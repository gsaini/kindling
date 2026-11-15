"""Authentication, scopes, model allowlists, rate limits and quotas, through the HTTP API."""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import anyio

from kindling.keys import service_for
from kindling.llm import FakeModel, ModelRegistry, ModelSpec, Turn
from kindling.store import MemoryStore

from .conftest import ADMIN, read_events

HELLO = {"messages": [{"role": "user", "content": "hello"}]}


def issue(client, **fields) -> tuple[str, dict]:
    response = client.post("/api/admin/keys", headers=ADMIN, json={"owner": "gopal", "name": "hearth", **fields})
    assert response.status_code == 201, response.text
    body = response.json()
    return body["key"], body


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_bootstrap_key_is_admin(make_client):
    client = make_client()
    caller = client.get("/api/info", headers=ADMIN).json()["caller"]
    assert caller == {"owner": "admin", "key": "bootstrap", "scopes": ["admin", "chat"]}
    assert client.get("/api/admin/keys", headers=ADMIN).status_code == 200


def test_issued_key_works_with_either_header(make_client):
    client = make_client()
    key, _ = issue(client)
    for headers in (bearer(key), {"x-api-key": key}, {"authorization": f"bearer {key}"}):
        caller = client.get("/api/info", headers=headers).json()["caller"]
        assert caller == {"owner": "gopal", "key": "hearth", "scopes": ["chat"]}


def test_bad_keys_all_get_the_same_401(make_client):
    store = MemoryStore()
    client = make_client(store=store)
    revoked, meta = issue(client, name="revoked")
    client.post(f"/api/admin/keys/{meta['id']}/revoke", headers=ADMIN)
    expired, meta = issue(client, name="expired")
    # Expire it behind the API's back (the API refuses past expiries).
    keys = service_for(store).keys

    async def expire():
        key = await keys.get(uuid.UUID(meta["id"]))
        await keys.save(dataclasses.replace(key, expires_at=datetime.now(UTC) - timedelta(seconds=1)))

    anyio.run(expire)
    service_for(store)._cache.clear()

    answers = set()
    for headers in (
        {},
        bearer("wrong"),
        bearer("kdl_" + "x" * 43),
        bearer(revoked),
        bearer(expired),
        {"x-api-key": "test-admin-key-not"},
        {"Authorization": "Basic dGVzdA=="},
    ):
        response = client.get("/api/info", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        answers.add(response.text)
    assert len(answers) == 1  # nothing tells an attacker which keys exist


def test_revocation_through_the_api_takes_effect_at_once(make_client):
    client = make_client()
    key, meta = issue(client)
    assert client.get("/api/info", headers=bearer(key)).status_code == 200  # now cached
    client.post(f"/api/admin/keys/{meta['id']}/revoke", headers=ADMIN)
    assert client.get("/api/info", headers=bearer(key)).status_code == 401


def test_chat_scope_cannot_use_the_admin_api(make_client):
    client = make_client()
    key, meta = issue(client)
    assert client.get("/api/admin/keys", headers=bearer(key)).status_code == 403
    assert client.post("/api/admin/keys", headers=bearer(key), json={"owner": "x", "name": "y"}).status_code == 403
    assert client.get("/api/admin/usage", headers=bearer(key)).status_code == 403
    assert client.post(f"/api/admin/keys/{meta['id']}/revoke", headers=bearer(key)).status_code == 403
    admin, _ = issue(client, name="ops", scopes=["admin"])
    assert client.get("/api/admin/keys", headers=bearer(admin)).status_code == 200


def test_model_allowlist(make_client):
    specs = [ModelSpec(id="fake", name="fake"), ModelSpec(id="big", name="big")]
    models = ModelRegistry(specs, "fake", models={"fake": FakeModel(Turn(content="hi")), "big": FakeModel()})
    client = make_client(models=models)
    key, _ = issue(client, models=["fake"])
    response = client.post("/api/chat", headers=bearer(key), json={**HELLO, "model": "big"})
    assert response.status_code == 403
    assert [m["id"] for m in client.get("/api/info", headers=bearer(key)).json()["models"]] == ["fake"]
    response = client.post("/api/chat", headers=bearer(key), json={**HELLO, "model": "fake"})
    assert read_events(response)[-1]["type"] == "done"


def test_rate_limit_per_key(make_client):
    client = make_client()
    key, _ = issue(client, rate_limit_per_minute=2)
    other, _ = issue(client, name="scripts", rate_limit_per_minute=2)
    assert [client.get("/api/info", headers=bearer(key)).status_code for _ in range(2)] == [200, 200]
    response = client.get("/api/info", headers=bearer(key))
    assert response.status_code == 429 and int(response.headers["retry-after"]) >= 1
    assert client.get("/api/info", headers=bearer(other)).status_code == 200


def test_default_rate_limit_applies_to_the_bootstrap_key(make_client):
    client = make_client(settings_overrides={"rate_limit_per_minute": 3})
    assert [client.get("/api/info", headers=ADMIN).status_code for _ in range(4)] == [200, 200, 200, 429]


def test_daily_token_quota_stops_before_the_model_runs(make_client):
    model = FakeModel(Turn(content="one"))  # 15 tokens (10 prompt + 5 output)
    client = make_client(model)
    key, meta = issue(client, daily_token_limit=10)
    assert read_events(client.post("/api/chat", headers=bearer(key), json=HELLO))[-1]["type"] == "done"
    response = client.post("/api/chat", headers=bearer(key), json=HELLO)
    assert response.status_code == 429 and "daily quota" in response.json()["detail"]
    assert 1 <= int(response.headers["retry-after"]) <= 86400
    assert len(model.calls) == 1  # refused before the model was called
    # Reads still work, and raising the limit lets it chat again.
    assert client.get("/api/info", headers=bearer(key)).status_code == 200
    client.patch(f"/api/admin/keys/{meta['id']}", headers=ADMIN, json={"daily_token_limit": 1000})
    model.script.append(Turn(content="two"))
    assert read_events(client.post("/api/chat", headers=bearer(key), json=HELLO))[-1]["type"] == "done"


def test_quota_is_seeded_from_recorded_usage(make_client):
    store = MemoryStore()
    client = make_client(FakeModel(Turn(content="one")), store=store)
    key, _ = issue(client, daily_token_limit=10)
    client.post("/api/chat", headers=bearer(key), json=HELLO)
    # A restarted server (fresh counters) reads today's usage back.
    service = service_for(store)
    service.tokens._used.clear()
    assert client.post("/api/chat", headers=bearer(key), json=HELLO).status_code == 429


def test_last_used_at_is_recorded(make_client):
    client = make_client()
    key, meta = issue(client)
    assert meta["last_used_at"] is None
    client.get("/api/info", headers=bearer(key))
    assert client.get(f"/api/admin/keys/{meta['id']}", headers=ADMIN).json()["last_used_at"] is not None


def test_rotated_key_keeps_the_owners_conversations(make_client):
    client = make_client(FakeModel(Turn(content="saved")))
    key, meta = issue(client)
    client.post("/api/chat", headers=bearer(key), json={**HELLO, "save": True})
    rotated = client.post(f"/api/admin/keys/{meta['id']}/rotate", headers=ADMIN).json()
    assert client.get("/api/info", headers=bearer(key)).status_code == 401
    assert len(client.get("/api/conversations", headers=bearer(rotated["key"])).json()) == 1


def test_auth_disabled_skips_keys_and_limits(make_client):
    client = make_client(settings_overrides={"auth_disabled": True, "rate_limit_per_minute": 1})
    assert [client.get("/api/info").status_code for _ in range(3)] == [200, 200, 200]
