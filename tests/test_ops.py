"""kindling.ops: request ids, headers, limits, CORS, metrics, logs. The gate has its own file."""

from __future__ import annotations

import io
import json
import logging

from fastapi import Request
from prometheus_client import REGISTRY

from kindling.llm import FakeModel, Turn, Usage
from kindling.ops import Overloaded
from kindling.ops.logs import build_handler

from .conftest import ADMIN


def add_echo_route(client) -> None:
    async def echo(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    client.app.add_api_route("/test/echo", echo, methods=["POST"])


# --- request ids ---------------------------------------------------------------------------------


def test_request_id_is_echoed(make_client):
    response = make_client().get("/livez", headers={"X-Request-ID": "trace-123.abc:9"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["x-request-id"] == "trace-123.abc:9"


def test_request_id_is_sanitised_and_capped(make_client):
    client = make_client()
    response = client.get("/livez", headers={"X-Request-ID": "abc def<script>\"';"})
    assert response.headers["x-request-id"] == "abcdefscript"
    response = client.get("/livez", headers={"X-Request-ID": "a" * 500})
    assert response.headers["x-request-id"] == "a" * 128


def test_request_id_is_generated(make_client):
    client = make_client()
    first = client.get("/livez").headers["x-request-id"]
    second = client.get("/livez", headers={"X-Request-ID": "<>"}).headers["x-request-id"]
    assert len(first) == 32 and int(first, 16) >= 0
    assert len(second) == 32 and second != first


# --- headers -------------------------------------------------------------------------------------


def test_security_headers(make_client):
    client = make_client()
    api = client.get("/api/info", headers=ADMIN)
    assert api.status_code == 200
    assert api.headers["x-content-type-options"] == "nosniff"
    assert api.headers["referrer-policy"] == "no-referrer"
    assert api.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in api.headers  # TLS ends at Cloudflare

    denied = client.get("/v1/models")  # errors get them too
    assert denied.headers["x-content-type-options"] == "nosniff"
    assert denied.headers["cache-control"] == "no-store"

    live = client.get("/livez")
    assert live.headers["x-content-type-options"] == "nosniff"
    assert "cache-control" not in live.headers


# --- body size limit -----------------------------------------------------------------------------


def test_body_over_content_length_limit_is_413(make_client):
    client = make_client(settings_overrides={"max_request_bytes": 1000})
    add_echo_route(client)
    assert client.post("/test/echo", content=b"x" * 1000).json() == {"bytes": 1000}
    response = client.post("/test/echo", content=b"x" * 1001)
    assert response.status_code == 413
    assert "1000 byte limit" in response.json()["detail"]
    assert response.headers["x-request-id"]
    # Before auth and validation: the route never sees it.
    assert client.post("/api/chat", content=b"x" * 5000).status_code == 413


def test_chunked_body_over_limit_is_413(make_client):
    client = make_client(settings_overrides={"max_request_bytes": 1000})
    add_echo_route(client)

    def chunks(n: int):
        for _ in range(n):
            yield b"x" * 400

    small = client.post("/test/echo", content=chunks(2))
    assert small.status_code == 200 and small.json() == {"bytes": 800}
    big = client.post("/test/echo", content=chunks(3))
    assert big.status_code == 413


def test_bad_content_length_is_400(make_client):
    client = make_client()
    add_echo_route(client)
    response = client.post("/test/echo", content=b"x", headers={"Content-Length": "lots"})
    assert response.status_code == 400


# --- CORS ----------------------------------------------------------------------------------------

PREFLIGHT = {
    "Origin": "https://app.example.com",
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "authorization, content-type, x-api-key, anthropic-version, x-request-id",
}


def test_cors_preflight_when_configured(make_client):
    client = make_client(settings_overrides={"cors_origins": ["https://app.example.com"]})
    response = client.options("/v1/chat/completions", headers=PREFLIGHT)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"
    allowed = response.headers["access-control-allow-headers"].lower()
    for header in ("authorization", "x-api-key", "content-type", "anthropic-version", "anthropic-beta"):
        assert header in allowed
    assert response.headers["x-request-id"]

    other = client.options("/v1/chat/completions", headers={**PREFLIGHT, "Origin": "https://evil.example"})
    assert other.headers.get("access-control-allow-origin") is None

    simple = client.get("/api/info", headers={**ADMIN, "Origin": "https://app.example.com"})
    assert simple.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "x-request-id" in simple.headers["access-control-expose-headers"].lower()


def test_no_cors_when_not_configured(make_client):
    client = make_client()
    preflight = client.options("/v1/chat/completions", headers=PREFLIGHT)
    assert "access-control-allow-origin" not in preflight.headers
    simple = client.get("/api/info", headers={**ADMIN, "Origin": "https://app.example.com"})
    assert "access-control-allow-origin" not in simple.headers


# --- Overloaded fallback -------------------------------------------------------------------------


def test_overloaded_escaping_a_route_is_503(make_client):
    client = make_client()

    async def busy() -> None:
        raise Overloaded("the server is busy", retry_after=7)

    client.app.add_api_route("/test/busy", busy, methods=["GET"])
    response = client.get("/test/busy")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"
    assert response.json() == {"detail": "the server is busy"}


# --- metrics -------------------------------------------------------------------------------------


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_need_an_admin_key(make_client):
    client = make_client()
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/metrics", headers=ADMIN)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "kindling_http_requests_total" in response.text
    assert "kindling_generations_in_flight" in response.text


def test_metrics_open_when_auth_disabled(make_client):
    client = make_client(settings_overrides={"auth_disabled": True})
    assert client.get("/metrics").status_code == 200


def test_metrics_can_be_turned_off(make_client):
    client = make_client(settings_overrides={"metrics_enabled": False})
    assert client.get("/metrics", headers=ADMIN).status_code == 404


def test_metrics_count_requests_by_route_template_and_tokens_by_model(make_client):
    model = FakeModel(Turn(content="hello there", usage=Usage(prompt_tokens=11, output_tokens=4)))
    client = make_client(model)
    labels = {"method": "GET", "route": "/livez", "status": "200"}
    requests_before = sample("kindling_http_requests_total", **labels)
    prompt_before = sample("kindling_prompt_tokens_total", model="fake")
    output_before = sample("kindling_output_tokens_total", model="fake")

    client.get("/livez")
    client.get("/no/such/path")
    chat = client.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": "hi"}]})
    assert chat.status_code == 200

    assert sample("kindling_http_requests_total", **labels) == requests_before + 1
    assert sample("kindling_http_requests_total", method="GET", route="unmatched", status="404") >= 1
    assert sample("kindling_prompt_tokens_total", model="fake") == prompt_before + 11
    assert sample("kindling_output_tokens_total", model="fake") == output_before + 4
    assert sample("kindling_generations_in_flight") == 0
    text = client.get("/metrics", headers=ADMIN).text
    assert 'route="/api/chat"' in text


# --- logs ----------------------------------------------------------------------------------------


def capture_logs(log_format: str = "json") -> tuple[io.StringIO, logging.Handler]:
    stream = io.StringIO()
    handler = build_handler(log_format, stream)
    logging.getLogger().addHandler(handler)
    return stream, handler


def test_access_log_line_and_no_secrets(make_client):
    client = make_client()
    stream, handler = capture_logs("json")
    try:
        response = client.get(
            "/api/info?api_key=query-secret",
            headers={**ADMIN, "x-api-key": "header-secret", "X-Request-ID": "req-42"},
        )
        assert response.status_code == 200
    finally:
        logging.getLogger().removeHandler(handler)
    output = stream.getvalue()
    for secret in ("test-admin-key", "header-secret", "query-secret", "Bearer"):
        assert secret not in output
    lines = [json.loads(line) for line in output.splitlines()]
    access = [line for line in lines if line["logger"] == "kindling.access"]
    assert len(access) == 1
    entry = access[0]
    assert entry["request_id"] == "req-42"
    assert entry["method"] == "GET" and entry["path"] == "/api/info" and entry["status"] == 200
    assert entry["key_owner"] == "admin" and entry["key_name"] == "bootstrap"
    assert entry["client_ip"] == "testclient"
    assert isinstance(entry["duration_ms"], float)
    assert {"ts", "level", "msg"} <= set(entry)


def test_text_log_format(make_client):
    client = make_client()
    stream, handler = capture_logs("text")
    try:
        client.get("/api/info", headers={**ADMIN, "X-Request-ID": "req-7"})
    finally:
        logging.getLogger().removeHandler(handler)
    line = next(line for line in stream.getvalue().splitlines() if "kindling.access" in line)
    assert "[req-7] GET /api/info 200" in line
    assert "key_owner=admin" in line
    assert "test-admin-key" not in stream.getvalue()


def test_request_id_reaches_log_records_inside_the_request(make_client):
    client = make_client()
    stream, handler = capture_logs("json")

    async def noisy() -> dict[str, bool]:
        logging.getLogger("kindling.test").warning("inside the route")
        return {"ok": True}

    client.app.add_api_route("/test/noisy", noisy, methods=["GET"])
    try:
        client.get("/test/noisy", headers={"X-Request-ID": "req-99"})
    finally:
        logging.getLogger().removeHandler(handler)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    inside = next(line for line in lines if line["msg"] == "inside the route")
    assert inside["request_id"] == "req-99"


# --- .env.example --------------------------------------------------------------------------------


def test_env_example_lists_every_setting_and_parses():
    from pathlib import Path

    from kindling.config import Settings

    path = Path(__file__).parents[1] / ".env.example"
    names = set()
    for line in path.read_text().splitlines():
        line = line.lstrip("# ").strip()
        if line.startswith("KINDLING_") and "=" in line:
            names.add(line.split("=", 1)[0].removeprefix("KINDLING_").lower())
    assert names == set(Settings.model_fields), "keep .env.example in step with kindling.config"
    settings = Settings(_env_file=path)
    assert settings.api_key == "" and not settings.auth_disabled  # secure as shipped
