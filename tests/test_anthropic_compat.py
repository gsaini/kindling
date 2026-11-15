"""The Anthropic-compatible API, driven through the official anthropic SDK against a FakeModel."""

from __future__ import annotations

import json

import anthropic
import pytest

from kindling.llm import FakeModel, ModelEvent, ToolCall, Turn, Usage

from .conftest import ADMIN
from .test_openai_compat import Endless, Overloaded, call_and_hang_up, metered

HI = [{"role": "user", "content": "Hi"}]


def sdk(client, api_key: str = "test-admin-key") -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url="http://testserver", api_key=api_key, http_client=client, max_retries=0)


def test_plain_reply_with_thinking_first(make_client):
    model = FakeModel(Turn(content="Hello there", thinking="Say hi.", usage=Usage(12, 7)))
    client = make_client(model)
    usage = metered(client)
    reply = sdk(client).messages.create(model="fake", max_tokens=100, system="Be warm.", messages=HI)
    assert reply.id.startswith("msg_") and reply.model == "fake" and reply.stop_reason == "end_turn"
    assert [b.type for b in reply.content] == ["thinking", "text"]
    assert reply.content[0].thinking == "Say hi." and reply.content[0].signature == ""
    assert reply.content[1].text == "Hello there"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (12, 7)
    assert model.calls[0]["messages"] == [{"role": "system", "content": "Be warm."}, {"role": "user", "content": "Hi"}]
    assert model.calls[0]["options"] == {"num_predict": 100}
    [event] = usage.events
    assert (event.endpoint, event.status, event.output_tokens) == ("/v1/messages", "ok", 7)


def test_streaming_follows_the_official_event_sequence(make_client):
    client = make_client(FakeModel(Turn(content="Hello there", thinking="Say hi.")))
    events = list(sdk(client).messages.create(model="fake", max_tokens=100, messages=HI, stream=True))
    kinds = [e.type for e in events]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    blocks = [(e.index, e.content_block.type) for e in events if e.type == "content_block_start"]
    assert blocks == [(0, "thinking"), (1, "text")]
    deltas = [e.delta.type for e in events if e.type == "content_block_delta"]
    assert set(deltas) == {"thinking_delta", "text_delta"}
    assert [e.index for e in events if e.type == "content_block_stop"] == [0, 1]
    assert events[-2].delta.stop_reason == "end_turn" and events[-2].usage.output_tokens == 5


def test_stream_helper_builds_the_final_message(make_client):
    client = make_client(FakeModel(Turn(content="Hello there", thinking="Say hi.")))
    with sdk(client).messages.stream(model="fake", max_tokens=100, messages=HI) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "Hello there"
    assert [b.type for b in final.content] == ["thinking", "text"] and final.usage.input_tokens == 10


WEATHER = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    "cache_control": {"type": "ephemeral"},
}


def test_tool_use_round_trip(make_client):
    model = FakeModel(
        Turn(content="Let me check.", tool_calls=[ToolCall("get_weather", {"city": "Paris"})]),
        Turn(content="Sunny, 21C"),
    )
    client = sdk(make_client(model))
    first = client.messages.create(
        model="fake", max_tokens=100, messages=[{"role": "user", "content": "Paris?"}], tools=[WEATHER]
    )
    assert first.stop_reason == "tool_use"
    text, use = first.content
    assert text.text == "Let me check." and use.type == "tool_use" and use.id.startswith("toolu_")
    assert (use.name, use.input) == ("get_weather", {"city": "Paris"})
    assert model.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": WEATHER["input_schema"],
            },
        }
    ]

    second = client.messages.create(
        model="fake",
        max_tokens=100,
        tools=[WEATHER],
        messages=[
            {"role": "user", "content": "Paris?"},
            {"role": "assistant", "content": [b.model_dump() for b in first.content]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": use.id, "content": [{"type": "text", "text": "clear"}]},
                    {"type": "text", "text": "And tomorrow?"},
                ],
            },
        ],
    )
    assert second.content[0].text == "Sunny, 21C"
    assert model.calls[1]["messages"][1:] == [
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "tool_name": "get_weather", "content": "clear"},
        {"role": "user", "content": "And tomorrow?"},
    ]


def test_streamed_tool_use_arrives_as_input_json(make_client):
    model = FakeModel(Turn(tool_calls=[ToolCall("get_weather", {"city": "Oslo"})]))
    with sdk(make_client(model)).messages.stream(model="fake", max_tokens=100, messages=HI, tools=[WEATHER]) as stream:
        events = list(stream)
        final = stream.get_final_message()
    assert any(e.type == "content_block_delta" and e.delta.type == "input_json_delta" for e in events)
    assert final.stop_reason == "tool_use" and final.content[0].input == {"city": "Oslo"}


def test_tool_errors_and_tool_choice(make_client):
    model = FakeModel(Turn(content="a"), Turn(content="b"))
    client = sdk(make_client(model))
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "boom", "is_error": True}],
        },
    ]
    client.messages.create(model="fake", max_tokens=10, messages=history, tools=[WEATHER], tool_choice={"type": "none"})
    assert model.calls[0]["tools"] is None
    assert model.calls[0]["messages"][-1] == {"role": "tool", "tool_name": "get_weather", "content": "Error: boom"}
    client.messages.create(model="fake", max_tokens=10, messages=HI, tools=[WEATHER], tool_choice={"type": "any"})
    assert (
        model.calls[1]["messages"][0]["role"] == "system" and "get_weather" in model.calls[1]["messages"][0]["content"]
    )


def test_thinking_switch_and_options(make_client):
    model = FakeModel(Turn(content="a"), Turn(content="b"), Turn(content="c"))
    client = sdk(make_client(model))
    client.messages.create(
        model="fake",
        max_tokens=2000,
        messages=HI,
        thinking={"type": "enabled", "budget_tokens": 1024},
        # Removed from the SDK with API version 2026-10-21; older clients and Claude Code builds still send them.
        extra_body={"temperature": 0.5, "top_p": 0.9, "top_k": 40},
    )
    assert model.calls[0]["think"] is True
    assert model.calls[0]["options"] == {"num_predict": 2000, "temperature": 0.5, "top_p": 0.9, "top_k": 40}
    client.messages.create(model="fake", max_tokens=10, messages=HI, thinking={"type": "disabled"})
    assert model.calls[1]["think"] is False
    client.messages.create(model="fake", max_tokens=10, messages=HI)
    assert model.calls[2]["think"] is None  # the model's configured default


def test_stop_sequences_and_max_tokens(make_client):
    model = FakeModel(
        Turn(content="one two STOP three"), Turn(content="one two STOP three"), Turn(content="long", usage=Usage(3, 8))
    )
    client = sdk(make_client(model))
    reply = client.messages.create(model="fake", max_tokens=100, messages=HI, stop_sequences=["STOP"])
    assert (reply.stop_reason, reply.stop_sequence, reply.content[0].text) == ("stop_sequence", "STOP", "one two ")
    assert "stop" not in model.calls[0]["options"]
    with client.messages.stream(model="fake", max_tokens=100, messages=HI, stop_sequences=["STOP"]) as stream:
        assert "".join(stream.text_stream) == "one two "
        assert stream.get_final_message().stop_reason == "stop_sequence"
    assert client.messages.create(model="fake", max_tokens=8, messages=HI).stop_reason == "max_tokens"


def test_claude_model_names_get_the_default_model(make_client):
    client = sdk(make_client(FakeModel(Turn(content="hi"))))
    reply = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=10,
        messages=HI,
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    )
    assert reply.content[0].text == "hi" and reply.model == "claude-sonnet-5-5"
    with pytest.raises(anthropic.NotFoundError) as raised:
        client.messages.create(model="gpt-9", max_tokens=10, messages=HI)
    assert raised.value.body["type"] == "error" and raised.value.body["error"]["type"] == "not_found_error"


def test_count_tokens_is_an_estimate(make_client):
    client = sdk(make_client())
    small = client.messages.count_tokens(model="claude-haiku-5-5", messages=HI)
    big = client.messages.count_tokens(
        model="fake", messages=[{"role": "user", "content": "word " * 400}], tools=[WEATHER]
    )
    assert 0 < small.input_tokens < 20 and big.input_tokens > 500


def test_models_answer_in_the_anthropic_shape(make_client):
    client = sdk(make_client())
    [model] = list(client.models.list())
    assert model.id == "fake" and model.type == "model"


def test_errors_use_the_anthropic_shape(make_client):
    client = make_client()
    with pytest.raises(anthropic.AuthenticationError) as raised:
        sdk(client, api_key="wrong").messages.create(model="fake", max_tokens=10, messages=HI)
    assert raised.value.body == {
        "type": "error",
        "error": {"type": "authentication_error", "message": "missing or invalid API key"},
    }

    missing = client.post("/v1/messages", headers=ADMIN, json={"model": "fake", "messages": HI})
    assert missing.status_code == 400 and missing.json()["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in missing.json()["error"]["message"]

    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    with pytest.raises(anthropic.BadRequestError, match="not supported yet"):
        sdk(client).messages.create(model="fake", max_tokens=10, messages=[{"role": "user", "content": [image]}])

    not_json = client.post("/v1/messages", headers={**ADMIN, "content-type": "application/json"}, content=b"{")
    assert not_json.status_code == 400 and not_json.json()["type"] == "error"


def test_a_full_queue_is_a_529(make_client):
    def busy(messages, kwargs):
        raise Overloaded("too many requests are waiting")

    client = make_client(FakeModel(busy, busy))
    with pytest.raises(anthropic.APIStatusError) as raised:
        sdk(client).messages.create(model="fake", max_tokens=10, messages=HI)
    assert raised.value.status_code == 529 and raised.value.body["error"]["type"] == "overloaded_error"
    streamed = client.post(
        "/v1/messages", headers=ADMIN, json={"model": "fake", "max_tokens": 10, "messages": HI, "stream": True}
    )
    assert streamed.status_code == 529


def test_model_failures_before_and_during_a_stream(make_client):
    client = make_client(FakeModel())  # no scripted turns: the model fails before any output...
    raw = client.post(
        "/v1/messages", headers=ADMIN, json={"model": "fake", "max_tokens": 10, "messages": HI, "stream": True}
    )
    assert raw.status_code == 502 and raw.json()["error"]["type"] == "api_error"  # ...so it's a plain error

    class Flaky:
        name, num_ctx = "fake", 4096

        async def check(self) -> None:
            pass

        async def stream(self, messages, **kwargs):
            yield ModelEvent("content", "partial")
            raise RuntimeError("Ollama went away")

    stream = sdk(make_client(Flaky())).messages.create(model="fake", max_tokens=10, messages=HI, stream=True)
    with pytest.raises(anthropic.APIStatusError, match=r"the model failed \(RuntimeError\)"):
        list(stream)


@pytest.mark.parametrize("stream", [True, False])
def test_a_client_that_hangs_up_stops_the_model(make_client, stream):
    model = Endless()
    client = make_client(model)
    usage = metered(client)
    body = {"model": "fake", "max_tokens": 1000, "messages": HI, "stream": stream}
    call_and_hang_up(client, "/v1/messages", body, after_chunks=4 if stream else 0)
    assert model.closed and [e.status for e in usage.events] == ["stopped"]


def test_message_start_carries_an_input_estimate(make_client):
    client = make_client(FakeModel(Turn(content="ok")))
    raw = client.post(
        "/v1/messages", headers=ADMIN, json={"model": "fake", "max_tokens": 10, "messages": HI, "stream": True}
    )
    frames = [f for f in raw.text.split("\n\n") if f]
    assert [f.splitlines()[0] for f in frames[:2]] == ["event: message_start", "event: ping"]
    first = json.loads(frames[0].split("data: ", 1)[1])
    assert first["message"]["usage"]["input_tokens"] > 0
