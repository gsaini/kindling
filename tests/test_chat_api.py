from kindling.llm import FakeModel, ToolCall, Turn
from kindling.store import MemoryStore

from .conftest import ADMIN, read_events


def test_every_api_call_needs_a_key(make_client):
    client = make_client()
    assert client.get("/api/info").status_code == 401
    assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
    assert client.get("/api/info", headers={"x-api-key": "test-admin-key"}).status_code == 200
    assert client.get("/api/health").status_code == 200  # health stays open for load balancers


def test_auth_can_be_switched_off_for_a_private_machine(make_client):
    client = make_client(settings_overrides={"auth_disabled": True})
    assert client.get("/api/info").json()["caller"]["owner"] == "local"


def test_chat_streams_reasoning_tools_answer_and_stats(make_client):
    model = FakeModel(
        Turn(thinking="Use the calculator.", tool_calls=[ToolCall("calculate", {"expression": "6*7"})]),
        Turn(content="42"),
    )
    client = make_client(model)
    response = client.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": "6*7?"}]})
    events = read_events(response)
    assert response.headers["content-type"] == "application/x-ndjson"
    kinds = [e["type"] for e in events]
    assert [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]] == [
        "thinking",
        "tool_call",
        "tool_result",
        "content",
        "done",
    ]
    assert "".join(e["text"] for e in events if e["type"] == "content") == "42"
    assert events[2] == {"type": "tool_result", "name": "calculate", "output": "42", "ok": True}
    stats = events[-1]["stats"]
    assert stats["model"] == "fake" and stats["steps"] == 2 and stats["output_tokens"] == 10


def test_saved_chats_are_stored_and_owned(make_client):
    store = MemoryStore()
    client = make_client(FakeModel(Turn(content="Hi there"), Turn(content="Again")), store=store)
    first = read_events(
        client.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": "Hello"}], "save": True})
    )
    conversation_id = first[0]["id"]
    history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
    client.post(
        "/api/chat",
        headers=ADMIN,
        json={"messages": [*history, {"role": "user", "content": "Once more"}], "conversation_id": conversation_id},
    )
    saved = client.get(f"/api/conversations/{conversation_id}", headers=ADMIN).json()
    assert saved["title"] == "Hello" and saved["owner"] == "admin"
    assert [(m["role"], m["content"]) for m in saved["messages"]] == [
        ("user", "Hello"),
        ("assistant", "Hi there"),
        ("user", "Once more"),
        ("assistant", "Again"),
    ]
    assert saved["messages"][1]["stats"]["model"] == "fake"
    assert [c["id"] for c in client.get("/api/conversations", headers=ADMIN).json()] == [conversation_id]


def test_requests_are_validated_and_models_resolved(make_client):
    client = make_client()
    chat = lambda body: client.post("/api/chat", headers=ADMIN, json=body)  # noqa: E731
    assert chat({"messages": []}).status_code == 422
    assert chat({"messages": [{"role": "assistant", "content": "x"}]}).status_code == 422
    assert chat({"messages": [{"role": "user", "content": "x"}], "model": "gpt-9"}).status_code == 404
    missing = "00000000-0000-0000-0000-000000000000"
    assert chat({"messages": [{"role": "user", "content": "x"}], "conversation_id": missing}).status_code == 404


def test_a_model_failure_arrives_as_an_error_event(make_client):
    def boom(messages, kwargs):
        raise RuntimeError("Ollama went away")

    events = read_events(
        make_client(FakeModel(boom)).post(
            "/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": "x"}]}
        )
    )
    # The client learns what kind of failure it was; the details (hosts, ports) stay in the log.
    assert events == [{"type": "error", "message": "the model failed (RuntimeError); see the server log for details"}]


def test_a_silent_model_gets_keepalive_pings(make_client, monkeypatch):
    import asyncio

    from kindling.api import chat

    class SlowModel(FakeModel):
        async def stream(self, messages, **kwargs):
            await asyncio.sleep(0.25)
            async for event in super().stream(messages, **kwargs):
                yield event

    monkeypatch.setattr(chat, "KEEPALIVE_SECONDS", 0.05)
    events = read_events(
        make_client(SlowModel(Turn(content="ok"))).post(
            "/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": "x"}]}
        )
    )
    kinds = [e["type"] for e in events]
    assert kinds[0] == "ping" and kinds[-1] == "done" and "content" in kinds


def test_stopped_and_failed_requests_still_count_against_quotas(make_client):
    from kindling.agent import StepLimitReached  # noqa: F401  (the step limit carries its usage)
    from kindling.llm import ToolCall
    from kindling.usage import MemoryUsageLog  # noqa: F401

    looping = Turn(content="still going", tool_calls=[ToolCall("current_time", {})])
    client = make_client(FakeModel(*[looping] * 3), settings_overrides={"max_tool_steps": 3})
    events = read_events(client.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": "x"}]}))
    assert events[-1]["type"] == "error" and "after 3 steps" in events[-1]["message"]
    usage = client.get("/api/admin/usage", headers=ADMIN).json()
    rows = usage if isinstance(usage, list) else usage.get("rows", [usage])
    assert sum(r.get("output_tokens", 0) for r in rows) == 15  # 3 steps x 5 tokens, not 0
