"""The admin API: issuing and managing keys, and reading usage."""

from __future__ import annotations

from kindling.llm import FakeModel, ModelRegistry, ModelSpec, Turn

from .conftest import ADMIN

HELLO = {"messages": [{"role": "user", "content": "hello"}]}


def two_models() -> ModelRegistry:
    specs = [ModelSpec(id="fake", name="fake", aliases=["nemo"]), ModelSpec(id="big", name="big")]
    return ModelRegistry(specs, "fake", models={"fake": FakeModel(*[Turn(content="hi")] * 5), "big": FakeModel()})


def test_create_returns_the_secret_once(make_client):
    client = make_client()
    response = client.post(
        "/api/admin/keys",
        headers=ADMIN,
        json={
            "owner": "gopal",
            "name": "claude-code",
            "scopes": ["chat"],
            "models": ["fake"],
            "rate_limit_per_minute": 30,
            "daily_token_limit": 2_000_000,
            "expires_in_days": 90,
            "note": "laptop",
        },
    )
    assert response.status_code == 201
    created = response.json()
    secret = created.pop("key")
    assert secret.startswith(created["prefix"]) and "warning" in created
    assert created["models"] == ["fake"] and created["state"] == "active" and created["expires_at"]

    shown = client.get(f"/api/admin/keys/{created['id']}", headers=ADMIN).json()
    listed = client.get("/api/admin/keys", headers=ADMIN).json()
    assert secret not in str(shown) + str(listed)
    assert "hash" not in shown and [k["id"] for k in listed] == [created["id"]]
    assert shown["today"]["requests"] == 0


def test_create_validates(make_client):
    client = make_client(models=two_models())

    def create(**fields):
        return client.post("/api/admin/keys", headers=ADMIN, json={"owner": "gopal", "name": "x", **fields})

    assert create(owner="has space").status_code == 422
    assert create(name="").status_code == 422
    assert create(name="x" * 101).status_code == 422
    assert create(scopes=["root"]).status_code == 422
    assert create(scopes=[]).status_code == 422
    assert create(rate_limit_per_minute=0).status_code == 422
    assert create(expires_at="2001-01-01T00:00:00Z").status_code == 422
    unknown = create(models=["gpt-9"])
    assert unknown.status_code == 422 and "Available: fake, big" in unknown.json()["detail"]
    assert create(models=["nemo"]).json()["models"] == ["fake"]  # aliases are stored as ids


def test_list_filters(make_client):
    client = make_client()
    ids = {}
    for owner, name in [("gopal", "hearth"), ("gopal", "scripts"), ("evals", "runner")]:
        ids[name] = client.post("/api/admin/keys", headers=ADMIN, json={"owner": owner, "name": name}).json()["id"]
    client.post(f"/api/admin/keys/{ids['scripts']}/revoke", headers=ADMIN)
    names = lambda params: [k["name"] for k in client.get("/api/admin/keys", headers=ADMIN, params=params).json()]  # noqa: E731
    assert names({}) == ["runner", "hearth"]
    assert names({"owner": "gopal"}) == ["hearth"]
    assert names({"owner": "gopal", "include_revoked": "true"}) == ["hearth", "scripts"]


def test_patch_changes_only_what_is_sent(make_client):
    client = make_client(models=two_models())
    key = client.post(
        "/api/admin/keys",
        headers=ADMIN,
        json={"owner": "gopal", "name": "hearth", "models": ["fake"], "daily_token_limit": 5, "note": "a"},
    ).json()
    url = f"/api/admin/keys/{key['id']}"
    changed = client.patch(url, headers=ADMIN, json={"name": "hearth-2", "rate_limit_per_minute": 10}).json()
    assert (changed["name"], changed["rate_limit_per_minute"], changed["models"], changed["note"]) == (
        "hearth-2",
        10,
        ["fake"],
        "a",
    )
    cleared = client.patch(url, headers=ADMIN, json={"models": None, "daily_token_limit": None}).json()
    assert cleared["models"] is None and cleared["daily_token_limit"] is None
    assert client.patch(url, headers=ADMIN, json={"scopes": ["admin", "chat"]}).json()["scopes"] == ["admin", "chat"]
    assert client.patch(url, headers=ADMIN, json={"scopes": ["sudo"]}).status_code == 422
    assert client.patch(url, headers=ADMIN, json={"models": ["nope"]}).status_code == 422
    client.post(f"{url}/revoke", headers=ADMIN)
    assert client.patch(url, headers=ADMIN, json={"name": "back"}).status_code == 409


def test_rotate_and_revoke(make_client):
    client = make_client()
    key = client.post("/api/admin/keys", headers=ADMIN, json={"owner": "gopal", "name": "hearth"}).json()
    rotated = client.post(f"/api/admin/keys/{key['id']}/rotate", headers=ADMIN).json()
    assert rotated["replaces"] == key["id"] and rotated["key"] != key["key"] and rotated["name"] == "hearth"
    assert client.get(f"/api/admin/keys/{key['id']}", headers=ADMIN).json()["state"] == "revoked"
    assert client.post(f"/api/admin/keys/{key['id']}/rotate", headers=ADMIN).status_code == 409
    revoked = client.post(f"/api/admin/keys/{rotated['id']}/revoke", headers=ADMIN).json()
    assert revoked["state"] == "revoked"
    assert client.get("/api/admin/keys/00000000-0000-0000-0000-000000000000", headers=ADMIN).status_code == 404


def test_usage_report(make_client):
    client = make_client(models=two_models())
    hearth = client.post("/api/admin/keys", headers=ADMIN, json={"owner": "gopal", "name": "hearth"}).json()
    evals = client.post("/api/admin/keys", headers=ADMIN, json={"owner": "evals", "name": "runner"}).json()
    for key in (hearth, hearth, evals):
        client.post("/api/chat", headers={"Authorization": f"Bearer {key['key']}"}, json=HELLO)
    client.post("/api/chat", headers={"x-api-key": hearth["key"]}, json={**HELLO, "model": "big"})  # an error

    def usage(**params):
        response = client.get("/api/admin/usage", headers=ADMIN, params=params)
        assert response.status_code == 200, response.text
        return response.json()["rows"]

    [total] = usage()
    # The failed request is charged an estimate (2 prompt tokens), so failing doesn't dodge quotas.
    assert (total["requests"], total["errors"], total["prompt_tokens"], total["output_tokens"]) == (4, 1, 32, 15)
    by_key = {r["key_name"]: r for r in usage(group_by="key")}
    assert by_key["hearth"]["requests"] == 3 and by_key["hearth"]["errors"] == 1 and by_key["runner"]["requests"] == 1
    assert by_key["hearth"]["key_id"] == hearth["id"]
    assert {r["model"]: r["requests"] for r in usage(group_by="model")} == {"big": 1, "fake": 3}
    assert [r["requests"] for r in usage(group_by="day")] == [4]
    assert usage(owner="evals")[0]["requests"] == 1
    assert usage(key_id=hearth["id"])[0]["requests"] == 3
    assert usage(since="2000-01-01", until="2000-01-02")[0]["requests"] == 0
    assert client.get("/api/admin/usage", headers=ADMIN, params={"since": "whenever"}).status_code == 422
    assert client.get("/api/admin/usage", headers=ADMIN, params={"group_by": "planet"}).status_code == 422
