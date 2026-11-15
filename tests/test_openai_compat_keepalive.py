"""Keepalives while the model is silent: the helper, and both streaming endpoints."""

from __future__ import annotations

import asyncio

import anthropic
import anyio
import pytest

from kindling.api.compat import streaming
from kindling.api.compat.keepalive import with_keepalive
from kindling.llm import FakeModel, Turn

from .conftest import ADMIN
from .test_openai_compat import sdk as openai_sdk


class Source:
    """An async generator source that notes when it was closed."""

    def __init__(self, items, delay: float, fail: Exception | None = None):
        self.items, self.delay, self.fail = items, delay, fail
        self.closed = False

    async def __call__(self):
        try:
            for item in self.items:
                await asyncio.sleep(self.delay)
                yield item
            if self.fail:
                raise self.fail
        finally:
            self.closed = True


def tasks_left() -> set[asyncio.Task]:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()}


@pytest.mark.anyio
async def test_pings_fill_the_silences():
    source = Source(["a", "b"], delay=0.05)
    before = tasks_left()
    got = [x async for x in with_keepalive(source(), 0.015, lambda: "ping")]
    assert [x for x in got if x != "ping"] == ["a", "b"]
    assert got.index("a") >= 2 and got.count("ping") >= 4 and got[-1] == "b"
    assert source.closed and tasks_left() == before


@pytest.mark.anyio
async def test_first_interval_and_no_pings_when_busy():
    got = [x async for x in with_keepalive(Source(["a"], delay=0.03)(), 1.0, lambda: "ping", first_interval=0.01)]
    assert got[0] == "ping" and got[-1] == "a"
    assert [x async for x in with_keepalive(Source(["a", "b"], delay=0)(), 0.5, lambda: "ping")] == ["a", "b"]


@pytest.mark.anyio
async def test_errors_reach_the_consumer():
    source = Source(["a"], delay=0, fail=RuntimeError("boom"))
    got = []
    with pytest.raises(RuntimeError, match="boom"):
        async for x in with_keepalive(source(), 1.0, lambda: "ping"):
            got.append(x)
    assert got == ["a"] and source.closed


@pytest.mark.anyio
async def test_stopping_early_or_being_cancelled_closes_the_source():
    before = tasks_left()
    source = Source(range(1000), delay=0.001)
    merged = with_keepalive(source(), 1.0, lambda: "ping")
    async for x in merged:
        if x == 3:
            break
    await merged.aclose()
    assert source.closed and tasks_left() == before

    source = Source(range(1000), delay=0.01)
    with anyio.move_on_after(0.05):  # what a client disconnect looks like from inside the stream
        async for _ in with_keepalive(source(), 0.02, lambda: "ping"):
            pass
    assert source.closed and tasks_left() == before


class Slow(FakeModel):
    """A FakeModel that is silent for a while first (loading, queued, reading the prompt)."""

    def __init__(self, delay: float, *script):
        super().__init__(*script)
        self.delay = delay

    async def stream(self, messages, **kwargs):
        await anyio.sleep(self.delay)
        async for event in super().stream(messages, **kwargs):
            yield event


@pytest.fixture
def quick_keepalives(monkeypatch):
    monkeypatch.setattr(streaming, "PRIME_SECONDS", 0.01)
    monkeypatch.setattr(streaming, "KEEPALIVE_SECONDS", 0.02)


def test_chat_completions_sends_keepalive_comments(make_client, quick_keepalives):
    client = make_client(Slow(0.15, Turn(content="late"), Turn(content="late")))
    body = {"model": "fake", "messages": [{"role": "user", "content": "x"}], "stream": True}
    raw = client.post("/v1/chat/completions", headers=ADMIN, json=body).text
    frames = [f for f in raw.split("\n\n") if f]
    assert frames[0].startswith("data: ") and '"role": "assistant"' in frames[0]  # headers went out at once
    assert frames[1] == ": keepalive" and frames.count(": keepalive") >= 3
    assert frames[-1] == "data: [DONE]"

    stream = openai_sdk(client).chat.completions.create(model="fake", messages=body["messages"], stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "late"


def test_messages_sends_ping_events(make_client, quick_keepalives):
    client = make_client(Slow(0.15, Turn(content="late"), Turn(content="late")))
    body = {"model": "fake", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}], "stream": True}
    raw = client.post("/v1/messages", headers=ADMIN, json=body).text
    ping = 'event: ping\ndata: {"type": "ping"}'
    frames = [f for f in raw.split("\n\n") if f]
    assert frames[0].startswith("event: message_start") and frames[1] == ping
    assert frames.count(ping) >= 4 and frames[-1].startswith("event: message_stop")

    sdk = anthropic.Anthropic(base_url="http://testserver", api_key="test-admin-key", http_client=client, max_retries=0)
    with sdk.messages.stream(model="fake", max_tokens=10, messages=body["messages"]) as stream:
        assert "".join(stream.text_stream) == "late"
