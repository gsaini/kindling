"""The OpenAI-compatible API, driven through the official openai SDK against a FakeModel."""

from __future__ import annotations

import json
from typing import Any

import anyio
import openai
import pytest

from kindling.llm import FakeModel, ModelEvent, ToolCall, Turn, Usage

from .conftest import ADMIN


def sdk(client, api_key: str = "test-admin-key") -> openai.OpenAI:
    return openai.OpenAI(base_url="http://testserver/v1", api_key=api_key, http_client=client, max_retries=0)


class Recorder:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def record(self, event) -> None:
        self.events.append(event)


def metered(client) -> Recorder:
    recorder = Recorder()
    client.app.state.services.usage = recorder
    return recorder


def test_plain_reply_with_reasoning_and_usage(make_client):
    model = FakeModel(Turn(content="Hello there", thinking="Say hi.", usage=Usage(12, 7)))
    client = make_client(model)
    usage = metered(client)
    reply = sdk(client).chat.completions.create(model="fake", messages=[{"role": "user", "content": "Hi"}])
    choice = reply.choices[0]
    assert choice.message.content == "Hello there" and choice.finish_reason == "stop"
    assert choice.message.model_extra["reasoning_content"] == "Say hi."
    assert reply.id.startswith("chatcmpl-") and reply.model == "fake"
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens) == (12, 7, 19)
    assert model.calls[0]["messages"] == [{"role": "user", "content": "Hi"}]
    assert model.calls[0]["tools"] is None and model.calls[0]["think"] is None
    [event] = usage.events
    assert (event.endpoint, event.model, event.status, event.prompt_tokens, event.output_tokens) == (
        "/v1/chat/completions",
        "fake",
        "ok",
        12,
        7,
    )


def test_streaming_sends_reasoning_content_and_a_usage_chunk(make_client):
    client = make_client(FakeModel(Turn(content="Hello there", thinking="Say hi.")))
    stream = sdk(client).chat.completions.create(
        model="fake",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    deltas = [c.choices[0].delta for c in chunks if c.choices]
    assert deltas[0].role == "assistant"
    assert "".join(d.content or "" for d in deltas) == "Hello there"
    assert "".join((d.model_extra or {}).get("reasoning_content") or "" for d in deltas) == "Say hi."
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert not chunks[-1].choices and chunks[-1].usage.total_tokens == 15
    assert all(c.usage is None for c in chunks[:-1])


def test_raw_stream_ends_with_done(make_client):
    client = make_client(FakeModel(Turn(content="ok")))
    response = client.post(
        "/v1/chat/completions",
        headers=ADMIN,
        json={"model": "fake", "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in response.text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    assert all(f.startswith("data: ") for f in frames)
    assert "usage" not in json.loads(frames[0][6:])


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "strict": True,
    },
}


def test_tool_call_round_trip(make_client):
    model = FakeModel(Turn(tool_calls=[ToolCall("get_weather", {"city": "Paris"})]), Turn(content="Sunny, 21C"))
    client = sdk(make_client(model))
    question = {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}
    first = client.chat.completions.create(model="fake", messages=[question], tools=[WEATHER])
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    [call] = choice.message.tool_calls
    assert call.id.startswith("call_") and call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}
    assert model.calls[0]["tools"] == [
        {"type": "function", "function": {k: v for k, v in WEATHER["function"].items() if k != "strict"}}
    ]

    second = client.chat.completions.create(
        model="fake",
        tools=[WEATHER],
        messages=[
            question,
            choice.message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": '{"sky": "clear"}'},
        ],
    )
    assert second.choices[0].message.content == "Sunny, 21C"
    assert model.calls[1]["messages"][1:] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "tool_name": "get_weather", "content": '{"sky": "clear"}'},
    ]


def test_a_stray_newline_before_a_tool_call_is_null_content(make_client):
    call = ToolCall("get_weather", {"city": "Paris"})
    model = FakeModel(Turn(content="\n", tool_calls=[call]), Turn(content="Let me check.", tool_calls=[call]))
    client = sdk(make_client(model))
    question = [{"role": "user", "content": "Weather in Paris?"}]
    blank = client.chat.completions.create(model="fake", messages=question, tools=[WEATHER]).choices[0].message
    assert blank.content is None and blank.tool_calls  # what OpenAI sends, so nothing blank reaches the user
    said = client.chat.completions.create(model="fake", messages=question, tools=[WEATHER]).choices[0].message
    assert said.content == "Let me check." and said.tool_calls  # real text alongside a call is kept


def test_streamed_tool_calls(make_client):
    model = FakeModel(
        Turn(tool_calls=[ToolCall("get_weather", {"city": "Oslo"}), ToolCall("get_weather", {"city": "Rome"})])
    )
    stream = sdk(make_client(model)).chat.completions.create(
        model="fake", messages=[{"role": "user", "content": "x"}], tools=[WEATHER], stream=True
    )
    chunks = list(stream)
    calls = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
    assert [c.index for c in calls] == [0, 1]
    assert [json.loads(c.function.arguments)["city"] for c in calls] == ["Oslo", "Rome"]
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_tool_choice_none_and_named(make_client):
    model = FakeModel(Turn(content="a"), Turn(content="b"))
    client = sdk(make_client(model))
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}]
    client.chat.completions.create(model="fake", messages=messages, tools=[WEATHER], tool_choice="none")
    assert model.calls[0]["tools"] is None
    client.chat.completions.create(
        model="fake",
        messages=messages,
        tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert [t["function"]["name"] for t in model.calls[1]["tools"]] == ["get_weather"]
    assert model.calls[1]["messages"][0]["content"].startswith("Be brief.\n\nYou must respond by calling the tool")


def test_options_format_and_reasoning_controls(make_client):
    model = FakeModel(Turn(content="{}"), Turn(content="{}"), Turn(content="{}"), Turn(content="x", usage=Usage(3, 5)))
    client = sdk(make_client(model))
    messages = [{"role": "developer", "content": "Answer in JSON."}, {"role": "user", "content": "x"}]
    client.chat.completions.create(
        model="fake",
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=50,
        stop="END",
        seed=7,
        response_format={"type": "json_object"},
        reasoning_effort="none",
    )
    call = model.calls[0]
    assert call["options"] == {"temperature": 0.2, "top_p": 0.9, "seed": 7, "num_predict": 50, "stop": ["END"]}
    assert call["format"] == "json" and call["think"] is False
    assert call["messages"][0] == {"role": "system", "content": "Answer in JSON."}

    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    client.chat.completions.create(
        model="fake",
        messages=messages,
        max_completion_tokens=20,
        response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}},
        reasoning_effort="high",
    )
    assert model.calls[1]["format"] == schema and model.calls[1]["think"] is True
    assert model.calls[1]["options"] == {"num_predict": 20}

    client.chat.completions.create(
        model="fake", messages=messages, reasoning_effort="high", extra_body={"think": False}
    )
    assert model.calls[2]["think"] is False

    capped = client.chat.completions.create(model="fake", messages=messages, max_tokens=5)
    assert capped.choices[0].finish_reason == "length"


def test_models_list_and_retrieve(make_client):
    client = sdk(make_client())
    assert [m.id for m in client.models.list()] == ["fake"]
    assert client.models.retrieve("fake").owned_by == "kindling"
    with pytest.raises(openai.NotFoundError):
        client.models.retrieve("gpt-9")


def test_errors_use_the_openai_shape(make_client):
    client = make_client()
    with pytest.raises(openai.AuthenticationError) as raised:
        sdk(client, api_key="wrong").chat.completions.create(model="fake", messages=[{"role": "user", "content": "x"}])
    assert raised.value.body["type"] == "invalid_request_error" and raised.value.body["code"] == "invalid_api_key"
    assert raised.value.response.headers["www-authenticate"] == "Bearer"

    with pytest.raises(openai.NotFoundError) as raised:
        sdk(client).chat.completions.create(model="gpt-9", messages=[{"role": "user", "content": "x"}])
    assert "gpt-9" in raised.value.body["message"]

    bad = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": []})
    assert bad.status_code == 400
    assert (
        set(bad.json()["error"]) == {"message", "type", "param", "code"} and bad.json()["error"]["param"] == "messages"
    )

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    with pytest.raises(openai.BadRequestError) as raised:
        sdk(client).chat.completions.create(model="fake", messages=[{"role": "user", "content": [image]}])
    assert "not supported yet" in raised.value.body["message"] and raised.value.body["param"] == "messages.0.content.0"

    orphan = [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "call_nope", "content": "y"}]
    with pytest.raises(openai.BadRequestError):
        sdk(client).chat.completions.create(model="fake", messages=orphan)

    # /api/* keeps FastAPI's own errors.
    assert client.post("/api/chat", headers=ADMIN, json={"messages": []}).status_code == 422
    assert "detail" in client.get("/api/info").json()


class Overloaded(RuntimeError):
    """Stands in for kindling.ops.Overloaded, which is matched by name."""


def test_a_full_queue_is_a_503(make_client):
    def busy(messages, kwargs):
        raise Overloaded("too many requests are waiting")

    client = make_client(FakeModel(busy, busy))
    usage = metered(client)
    with pytest.raises(openai.InternalServerError) as raised:
        sdk(client).chat.completions.create(model="fake", messages=[{"role": "user", "content": "x"}])
    assert raised.value.status_code == 503 and raised.value.body["code"] == "overloaded"
    streamed = client.post(
        "/v1/chat/completions",
        headers=ADMIN,
        json={"model": "fake", "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert streamed.status_code == 503 and streamed.json()["error"]["code"] == "overloaded"
    assert [e.status for e in usage.events] == ["error", "error"]


def test_a_failure_mid_stream_arrives_as_an_error_event(make_client):
    class Flaky:
        name, num_ctx = "fake", 4096

        async def check(self) -> None:
            pass

        async def stream(self, messages, **kwargs):
            yield ModelEvent("content", "partial")
            raise RuntimeError("Ollama went away")

    stream = sdk(make_client(Flaky())).chat.completions.create(
        model="fake", messages=[{"role": "user", "content": "x"}], stream=True
    )
    with pytest.raises(openai.APIError, match=r"the model failed \(RuntimeError\)"):
        list(stream)


# -- disconnects --------------------------------------------------------------------------------


class Endless:
    """A model that never stops on its own, and notes when its stream is closed."""

    name, num_ctx = "fake", 4096

    def __init__(self) -> None:
        self.closed = False

    async def check(self) -> None:
        pass

    async def stream(self, messages, **kwargs):
        try:
            while True:
                yield ModelEvent("content", "tick ")
                await anyio.sleep(0.005)
        finally:
            self.closed = True


def call_and_hang_up(client, path: str, body: dict[str, Any], *, after_chunks: int) -> list[dict[str, Any]]:
    """Call the app over raw ASGI and disconnect: after `after_chunks` body chunks (or, with 0,
    shortly after sending the request). Runs on the TestClient's event loop."""
    sent: list[dict[str, Any]] = []

    async def run() -> None:
        hang_up = anyio.Event()
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
            if after_chunks:
                await hang_up.wait()
            else:
                await anyio.sleep(0.05)
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)
            if sum(1 for m in sent if m.get("body")) >= after_chunks > 0:
                hang_up.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer test-admin-key"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
            "state": {},
        }
        with anyio.fail_after(5):
            await client.app(scope, receive, send)

    client.portal.call(run)
    return sent


@pytest.mark.parametrize("stream", [True, False])
def test_a_client_that_hangs_up_stops_the_model(make_client, stream):
    model = Endless()
    client = make_client(model)
    usage = metered(client)
    body = {"model": "fake", "messages": [{"role": "user", "content": "x"}], "stream": stream}
    sent = call_and_hang_up(client, "/v1/chat/completions", body, after_chunks=3 if stream else 0)
    assert model.closed
    assert [e.status for e in usage.events] == ["stopped"]
    if stream:
        assert sent[0]["status"] == 200
