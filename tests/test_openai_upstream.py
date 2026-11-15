"""OpenAICompatModel: a ChatModel over an OpenAI-compatible server, tested with httpx.MockTransport."""

from __future__ import annotations

import json

import anthropic
import httpx
import pytest

from kindling.llm import ModelNotReady, ModelRegistry, ModelSpec, complete
from kindling.llm.openai_upstream import OpenAICompatModel, to_openai_messages
from kindling.llm.registry import build_model

SPEC = ModelSpec(
    id="nemotron-super",
    provider="openai",
    name="nvidia/nemotron-super",
    host="http://gpu-box:8000/v1/",
    api_key_env="TEST_UPSTREAM_KEY",
)


def sse(*chunks: dict | str) -> bytes:
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]
    return "".join(lines).encode()


def delta(**fields) -> dict:
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": fields, "finish_reason": None}],
    }


REPLY = sse(
    delta(role="assistant", content=""),
    delta(reasoning_content="Check the "),
    delta(reasoning_content="weather."),
    delta(content="Looking it up."),
    delta(
        tool_calls=[
            {
                "index": 0,
                "id": "chatcmpl-tool-1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }
        ]
    ),
    delta(tool_calls=[{"index": 0, "function": {"arguments": '{"city": '}}]),
    delta(tool_calls=[{"index": 0, "function": {"arguments": '"Paris"}'}}]),
    {"id": "x", "choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59}},
    "[DONE]",
)


class Upstream:
    """Records requests and answers like vLLM."""

    def __init__(self, body: bytes = REPLY, status: int = 200):
        self.body, self.status = body, status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "nvidia/nemotron-super"}]})
        return httpx.Response(self.status, content=self.body, headers={"content-type": "text/event-stream"})

    def model(self, spec: ModelSpec = SPEC) -> OpenAICompatModel:
        return OpenAICompatModel(spec, transport=httpx.MockTransport(self))


@pytest.mark.anyio
async def test_stream_maps_reasoning_content_tool_calls_and_usage(monkeypatch):
    monkeypatch.setenv("TEST_UPSTREAM_KEY", "secret")
    upstream = Upstream()
    model = upstream.model()
    events = [
        e
        async for e in model.stream(
            [{"role": "user", "content": "Weather in Paris?"}],
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            think=True,
            format="json",
            options={"num_predict": 64, "temperature": 0.3, "stop": ["END"], "num_ctx": 8192},
        )
    ]
    assert [e.kind for e in events] == ["thinking", "thinking", "content", "tool_calls", "done"]
    assert "".join(e.text for e in events if e.kind == "thinking") == "Check the weather."
    [call] = events[3].tool_calls
    assert (call.name, call.arguments, call.id) == ("get_weather", {"city": "Paris"}, "chatcmpl-tool-1")
    assert (events[-1].usage.prompt_tokens, events[-1].usage.output_tokens) == (42, 17)

    [request] = upstream.requests
    assert str(request.url) == "http://gpu-box:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    payload = json.loads(request.content)
    assert payload["model"] == "nvidia/nemotron-super" and payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["response_format"] == {"type": "json_object"}
    assert (payload["max_tokens"], payload["temperature"], payload["stop"]) == (64, 0.3, ["END"])
    assert "num_ctx" not in payload and payload["tools"][0]["function"]["name"] == "get_weather"


@pytest.mark.anyio
async def test_reasoning_field_alias_schema_format_and_no_key(monkeypatch):
    monkeypatch.delenv("TEST_UPSTREAM_KEY", raising=False)
    upstream = Upstream(sse(delta(reasoning="Hmm."), delta(content="Hi"), "[DONE]"))
    result = await complete(
        upstream.model(), [{"role": "user", "content": "x"}], think=False, format={"type": "object"}
    )
    assert (result.thinking, result.content, result.usage.output_tokens) == ("Hmm.", "Hi", 0)
    payload = json.loads(upstream.requests[0].content)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "authorization" not in upstream.requests[0].headers


def test_messages_get_tool_call_ids():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "thinking": "dropped",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}},
                {"function": {"name": "get_time", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_name": "get_time", "content": "noon"},
        {"role": "tool", "tool_name": "get_weather", "content": "rain"},
    ]
    out = to_openai_messages(messages)
    calls = out[2]["tool_calls"]
    assert out[2]["content"] is None and "thinking" not in out[2]
    assert [c["function"]["arguments"] for c in calls] == ['{"city": "Oslo"}', "{}"]
    assert out[3] == {"role": "tool", "tool_call_id": calls[1]["id"], "content": "noon"}
    assert out[4]["tool_call_id"] == calls[0]["id"] and calls[0]["id"] != calls[1]["id"]


@pytest.mark.anyio
async def test_upstream_errors_are_raised_with_their_message():
    upstream = Upstream(json.dumps({"error": {"message": "model is overloaded"}}).encode(), status=503)
    with pytest.raises(RuntimeError, match="503: model is overloaded"):
        await complete(upstream.model(), [{"role": "user", "content": "x"}])

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    model = OpenAICompatModel(SPEC, transport=httpx.MockTransport(refuse))
    with pytest.raises(RuntimeError, match="Can't reach"):
        await complete(model, [{"role": "user", "content": "x"}])
    with pytest.raises(ModelNotReady, match="Can't reach"):
        await model.check()


@pytest.mark.anyio
async def test_check_confirms_the_model_is_served():
    await Upstream().model().check()
    other = SPEC.model_copy(update={"name": "nvidia/other"})
    with pytest.raises(ModelNotReady, match="does not serve 'nvidia/other'"):
        await Upstream().model(other).check()
    with pytest.raises(ModelNotReady, match="no host"):
        await OpenAICompatModel(SPEC.model_copy(update={"host": None})).check()


def test_the_registry_builds_it_from_a_spec():
    model = build_model(SPEC)
    assert isinstance(model, OpenAICompatModel) and model.name == "nvidia/nemotron-super"
    assert model.base_url == "http://gpu-box:8000/v1"


def test_kindling_serves_an_upstream_model_end_to_end(make_client):
    upstream = Upstream()
    registry = ModelRegistry([SPEC], SPEC.id, models={SPEC.id: upstream.model()})
    client = make_client(models=registry)
    sdk = anthropic.Anthropic(base_url="http://testserver", api_key="test-admin-key", http_client=client, max_retries=0)
    reply = sdk.messages.create(model="nemotron-super", max_tokens=64, messages=[{"role": "user", "content": "Paris?"}])
    assert [b.type for b in reply.content] == ["thinking", "text", "tool_use"]
    assert reply.stop_reason == "tool_use" and reply.usage.input_tokens == 42
