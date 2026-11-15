"""Non-streaming replies padded open while the model is slow, and the context window check."""

from __future__ import annotations

import json

import anthropic
import anyio
import pytest

from kindling.api.compat import streaming
from kindling.llm import FakeModel, Turn

from .conftest import ADMIN
from .test_openai_compat import sdk as openai_sdk
from .test_openai_compat_keepalive import Slow

USER = [{"role": "user", "content": "x"}]


class SlowFailure(FakeModel):
    """Silent for a while, then fails: an error that arrives after the padding has started."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    async def stream(self, messages, **kwargs):
        await anyio.sleep(self.delay)
        raise RuntimeError("the model fell over")
        yield  # an async generator, like a real model


@pytest.fixture
def quick_padding(monkeypatch):
    monkeypatch.setattr(streaming, "PAD_AFTER_SECONDS", 0.02)
    monkeypatch.setattr(streaming, "KEEPALIVE_SECONDS", 0.02)


def test_a_slow_reply_is_padded_and_still_parses(make_client, quick_padding):
    client = make_client(Slow(0.15, Turn(content="late"), Turn(content="late")))
    response = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": USER})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.text.startswith("  ")  # at least two spaces went out while the model was silent
    assert json.loads(response.text)["choices"][0]["message"]["content"] == "late"

    reply = openai_sdk(client).chat.completions.create(model="fake", messages=USER)
    assert reply.choices[0].message.content == "late"


def test_a_quick_reply_is_not_padded(make_client):
    client = make_client(FakeModel(Turn(content="hi")))
    response = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": USER})
    assert response.status_code == 200 and response.text.startswith("{")
    assert response.headers["x-kindling-model"] == "fake"


def test_a_quick_failure_keeps_its_status(make_client):
    client = make_client(SlowFailure(0))
    response = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": USER})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"


def test_a_failure_after_padding_is_an_error_body_with_200(make_client, quick_padding):
    client = make_client(SlowFailure(0.1))
    response = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": USER})
    assert response.status_code == 200 and response.text.startswith(" ")
    error = json.loads(response.text)["error"]
    assert error["type"] == "server_error" and error["message"]


def test_messages_pads_a_slow_reply_too(make_client, quick_padding):
    client = make_client(Slow(0.15, Turn(content="late"), Turn(content="late")))
    body = {"model": "fake", "max_tokens": 10, "messages": USER}
    response = client.post("/v1/messages", headers=ADMIN, json=body)
    assert response.status_code == 200 and response.text.startswith("  ")
    assert json.loads(response.text)["content"][0]["text"] == "late"

    sdk = anthropic.Anthropic(base_url="http://testserver", api_key="test-admin-key", http_client=client, max_retries=0)
    assert sdk.messages.create(**body).content[0].text == "late"


def test_a_prompt_that_cannot_fit_is_refused(make_client):
    model = FakeModel(Turn(content="never"))
    client = make_client(model)
    too_long = [{"role": "user", "content": "a" * (model.num_ctx * streaming.CHARS_PER_TOKEN_AT_MOST + 600)}]

    response = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": too_long})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded" and error["param"] == "messages"
    assert f"maximum context length is {model.num_ctx} tokens" in error["message"]

    body = {"model": "fake", "max_tokens": 10, "messages": too_long}
    response = client.post("/v1/messages", headers=ADMIN, json=body)
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("prompt is too long: ")
    assert model.calls == []  # refused before the model was asked


def test_a_long_prompt_that_may_fit_goes_to_the_model(make_client):
    model = FakeModel(Turn(content="fine"))
    client = make_client(model)
    # About 1.5 times the window at four characters per token: too long by a rough count, but the check
    # only refuses what is certain, so the model gets to try.
    long = [{"role": "user", "content": "a" * (model.num_ctx * 6)}]
    response = client.post("/v1/chat/completions", headers=ADMIN, json={"model": "fake", "messages": long})
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "fine"


def test_tool_schemas_count_towards_the_context(make_client):
    client = make_client(FakeModel(Turn(content="never")))
    big = {"type": "function", "function": {"name": "f", "description": "d" * 30000, "parameters": {"type": "object"}}}
    body = {"model": "fake", "messages": USER, "tools": [big]}
    response = client.post("/v1/chat/completions", headers=ADMIN, json=body)
    assert response.status_code == 400 and response.json()["error"]["code"] == "context_length_exceeded"
