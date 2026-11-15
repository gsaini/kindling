"""The Cloudflare edge middleware: origin secret and CF-Connecting-IP. All offline."""

from __future__ import annotations

import pytest
from fastapi import Request

from kindling.edge import EdgeMiddleware, client_ip_from

from .conftest import ADMIN

SECRET = "s3cret-from-the-edge"
EDGE = {"X-Kindling-Origin-Secret": SECRET}


def with_echo(client):
    """Add a route that shows what the app sees: the headers and the client address."""

    @client.app.get("/api/_echo")
    async def echo(request: Request) -> dict:
        return {"headers": dict(request.headers), "client": request.client.host if request.client else None}

    return client


def edge_middleware_installed(client) -> bool:
    return any(m.cls is EdgeMiddleware for m in client.app.user_middleware)


def test_nothing_added_by_default(make_client):
    client = with_echo(make_client())
    assert not edge_middleware_installed(client)
    seen = client.get("/api/_echo", headers={**ADMIN, "CF-Connecting-IP": "203.0.113.7", **EDGE}).json()
    assert seen["client"] == "testclient"
    assert seen["headers"]["x-kindling-origin-secret"] == SECRET  # untouched when the feature is off


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Kindling-Origin-Secret": "wrong"}, {"X-Kindling-Origin-Secret": ""}],
    ids=["missing", "wrong", "empty"],
)
def test_origin_secret_refuses_requests_without_it(make_client, headers):
    client = make_client(settings_overrides={"origin_secret": SECRET})
    response = client.get("/api/info", headers={**ADMIN, **headers})
    assert response.status_code == 403
    assert response.json() == {"detail": "forbidden"}


def test_origin_secret_refuses_duplicates(make_client):
    client = make_client(settings_overrides={"origin_secret": SECRET})
    headers = [("authorization", "Bearer test-admin-key"), ("x-kindling-origin-secret", SECRET)]
    headers.append(("x-kindling-origin-secret", "attacker-value"))
    assert client.get("/api/info", headers=headers).status_code == 403


def test_origin_secret_lets_the_edge_through_and_strips_it(make_client):
    client = with_echo(make_client(settings_overrides={"origin_secret": SECRET}))
    assert client.get("/api/info", headers={**ADMIN, **EDGE}).status_code == 200
    seen = client.get("/api/_echo", headers={**ADMIN, **EDGE}).json()
    assert "x-kindling-origin-secret" not in seen["headers"]
    assert seen["headers"]["authorization"] == "Bearer test-admin-key"


def test_origin_secret_does_not_replace_api_keys(make_client):
    client = make_client(settings_overrides={"origin_secret": SECRET})
    assert client.get("/api/info", headers=EDGE).status_code == 401


def test_custom_header_name_is_case_insensitive(make_client):
    client = make_client(settings_overrides={"origin_secret": SECRET, "origin_secret_header": "X-Edge-Proof"})
    assert client.get("/api/info", headers={**ADMIN, **EDGE}).status_code == 403
    assert client.get("/api/info", headers={**ADMIN, "x-edge-proof": SECRET}).status_code == 200


def test_health_is_exempt(make_client):
    client = make_client(settings_overrides={"origin_secret": SECRET})
    assert client.get("/api/health").status_code == 200


def test_trust_cloudflare_takes_the_client_ip(make_client):
    client = with_echo(make_client(settings_overrides={"trust_cloudflare_headers": True}))
    seen = client.get("/api/_echo", headers={**ADMIN, "CF-Connecting-IP": " 2001:db8::1 "}).json()
    assert seen["client"] == "2001:db8::1"
    assert client.get("/api/_echo", headers=ADMIN).json()["client"] == "testclient"


def test_trust_cloudflare_ignores_garbage(make_client):
    client = with_echo(make_client(settings_overrides={"trust_cloudflare_headers": True}))
    seen = client.get("/api/_echo", headers={**ADMIN, "CF-Connecting-IP": "not-an-ip\r\nx"}).json()
    assert seen["client"] == "testclient"


def test_untrusted_by_default(make_client):
    client = with_echo(make_client(settings_overrides={"origin_secret": SECRET}))
    seen = client.get("/api/_echo", headers={**ADMIN, **EDGE, "CF-Connecting-IP": "203.0.113.7"}).json()
    assert seen["client"] == "testclient"


def test_both_settings(make_client):
    client = with_echo(make_client(settings_overrides={"origin_secret": SECRET, "trust_cloudflare_headers": True}))
    seen = client.get("/api/_echo", headers={**ADMIN, **EDGE, "CF-Connecting-IP": "203.0.113.7"}).json()
    assert seen["client"] == "203.0.113.7"


@pytest.mark.anyio
async def test_exempt_path_without_secret_does_not_trust_the_ip():
    """A probe that skipped the secret didn't come through Cloudflare, so its CF-Connecting-IP is ignored."""
    seen: dict = {}

    async def app(scope, receive, send):
        seen.update(scope)

    middleware = EdgeMiddleware(app, origin_secret=SECRET, trust_cloudflare_headers=True)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/health",
        "headers": [(b"cf-connecting-ip", b"203.0.113.7")],
        "client": ("127.0.0.1", 5000),
    }
    await middleware(scope, None, None)
    assert seen["client"] == ("127.0.0.1", 5000)


@pytest.mark.anyio
async def test_lifespan_passes_through():
    calls: list[str] = []

    async def app(scope, receive, send):
        calls.append(scope["type"])

    await EdgeMiddleware(app, origin_secret=SECRET)({"type": "lifespan"}, None, None)
    assert calls == ["lifespan"]


def test_client_ip_from():
    assert client_ip_from(b"198.51.100.4") == "198.51.100.4"
    assert client_ip_from(b"2001:DB8::1") == "2001:db8::1"
    assert client_ip_from(b"") is None
    assert client_ip_from(b"1.2.3.4, 5.6.7.8") is None
