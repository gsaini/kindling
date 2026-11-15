"""Running one model call for a compat route: metering, client disconnects, and SSE.

Both APIs follow the same rules:

- Usage is recorded once per request, in a shielded finally, whatever happens.
- A client that goes away stops the generation: the model stream is closed (for Ollama that closes
  the HTTP response, which makes it stop generating).
- A streamed reply waits at most PRIME_SECONDS for the model's first event before sending headers,
  so a call that fails at once (the generation queue is full, Ollama is down) still gets a proper
  HTTP error status. After that, failures arrive as an error event in the stream, as both real APIs do.
- While the model is silent (loading, queued, reading a long prompt), a keepalive goes out every
  KEEPALIVE_SECONDS, so Cloudflare (which gives up after about 125 s without a byte) keeps the
  connection open. See keepalive.py.
- A non-streaming reply that isn't ready after PAD_AFTER_SECONDS commits to 200 and sends a space
  every KEEPALIVE_SECONDS until its JSON is ready (PaddedJSONResponse), for the same reason.
- A prompt that clearly can't fit the model's context window is refused up front (check_context).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio
from fastapi import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from ...llm import ChatModel, Message, ModelEvent, Usage
from ...usage import UsageEvent
from ..deps import Principal, Services
from .errors import InvalidRequest
from .keepalive import with_keepalive

PRIME_SECONDS = 1.0  # how long a stream waits for the first model event before committing to 200
KEEPALIVE_SECONDS = 15.0  # Cloudflare cuts a request after about 125 s without a byte from the origin
# How long a non-streaming reply may take before it commits to 200 and starts padding. Short enough
# to stay far from Cloudflare's limit, long enough that ordinary replies (and fast failures, such as
# a full queue) keep their real status code.
PAD_AFTER_SECONDS = 20.0
# Characters per token for the context check. Real text runs at about 3 to 5.5 characters per token,
# so dividing by 6 undercounts: the check only refuses prompts that are certainly too long, and leaves
# borderline ones to the model.
CHARS_PER_TOKEN_AT_MOST = 6


@dataclass
class ModelCall:
    """A model turn, already translated from the client's request."""

    model: ChatModel
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    think: bool | None = None
    format: dict[str, Any] | str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    # Filled in as the call runs, so usage can be charged even if it never reaches `done`.
    streamed_chars: int = 0
    usage: Usage | None = None

    async def events(self) -> AsyncIterator[ModelEvent]:
        stream = self.model.stream(
            self.messages, tools=self.tools, think=self.think, format=self.format, options=self.options or None
        )
        try:
            async for event in stream:
                self.streamed_chars += len(event.text) + sum(len(json.dumps(c.arguments)) for c in event.tool_calls)
                if event.usage is not None:
                    self.usage = event.usage
                yield event
        finally:
            await _close(stream)  # release the model (and its generation slot) right away

    def estimated_usage(self) -> Usage:
        prompt = sum(len(str(m.get("content") or "")) for m in self.messages)
        return Usage(prompt_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(self.streamed_chars))


async def _close(stream: AsyncIterator[Any]) -> None:
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        with anyio.CancelScope(shield=True):
            await closer()


class Meter:
    """One UsageEvent per request; `record()` is called exactly once, from a finally."""

    def __init__(
        self, svc: Services, principal: Principal, endpoint: str, model_id: str, call: ModelCall | None = None
    ):
        self.svc = svc
        self.event = UsageEvent.for_principal(principal, endpoint, model_id, status="stopped")
        self.started = time.perf_counter()
        self.call = call

    def finish(self, usage: Usage | None, *, error: bool = False) -> None:
        if usage is not None:
            self.event.prompt_tokens, self.event.output_tokens = usage.prompt_tokens, usage.output_tokens
        self.event.status = "error" if error else "ok"

    async def record(self) -> None:
        self.event.wall_seconds = time.perf_counter() - self.started
        if self.call is not None and not (self.event.prompt_tokens or self.event.output_tokens):
            # Stopped or failed before the final count: charge what the model reported, or an
            # estimate, so disconnecting early doesn't dodge a daily quota.
            usage = self.call.usage or (self.call.estimated_usage() if self.call.streamed_chars else None)
            if usage is not None:
                self.event.prompt_tokens, self.event.output_tokens = usage.prompt_tokens, usage.output_tokens
        with anyio.CancelScope(shield=True):
            await self.svc.usage.record(self.event)


class ClientGone(Exception):
    pass


async def _wait_for_disconnect(receive: Receive) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return


async def collect(request: Request, call: ModelCall, on_event: Callable[[ModelEvent], bool | None]) -> None:
    """Run a call to the end for a non-streaming reply. `on_event` returns False to stop early
    (a stop sequence). Raises ClientGone if the client disconnects first, or the model's error."""
    failure: BaseException | None = None
    gone = False
    async with anyio.create_task_group() as group:

        async def watch() -> None:
            nonlocal gone
            await _wait_for_disconnect(request.receive)
            gone = True
            group.cancel_scope.cancel()

        group.start_soon(watch)
        stream = call.events()
        try:
            async for event in stream:
                if on_event(event) is False:
                    break
        except Exception as error:  # re-raised below, outside the task group (no ExceptionGroup)
            failure = error
        finally:
            await _close(stream)
            group.cancel_scope.cancel()
    if gone:
        raise ClientGone
    if failure is not None:
        raise failure


class Encoder(Protocol):
    """Turns model events into one API's SSE frames."""

    finished: bool  # set when the reply is complete before the model stream ends (a stop sequence)
    usage: Usage | None  # the final counts, once known

    def opening(self) -> list[bytes]: ...
    def feed(self, event: ModelEvent) -> list[bytes]: ...
    def closing(self) -> list[bytes]: ...
    def failure(self, error: Exception) -> list[bytes]: ...
    def heartbeat(self) -> bytes: ...


_PING = object()
_END = object()


class ModelStreamResponse(Response):
    """An SSE response driven by a model call, with keepalives while the model is silent and a
    watcher that cancels everything when the client disconnects."""

    media_type = "text/event-stream"

    def __init__(
        self,
        call: ModelCall,
        encoder: Encoder,
        meter: Meter,
        *,
        on_error: Callable[[Exception], Response],
        headers: dict[str, str] | None = None,
    ):
        self.call = call
        self.encoder = encoder
        self.meter = meter
        self.on_error = on_error
        self.status_code = 200
        self.background = None
        self.init_headers({"Cache-Control": "no-store", "X-Accel-Buffering": "no", **(headers or {})})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        events = with_keepalive(self.call.events(), KEEPALIVE_SECONDS, lambda: _PING, first_interval=PRIME_SECONDS)
        failed = False
        try:
            async with anyio.create_task_group() as group:

                async def watch() -> None:
                    await _wait_for_disconnect(receive)
                    group.cancel_scope.cancel()

                group.start_soon(watch)
                try:
                    failed = await self._stream(events, send)
                except OSError:  # ASGI 2.4 servers raise from send() when the client has gone
                    pass
                group.cancel_scope.cancel()
        finally:
            await _close(events)  # stops the model if it is still going
            if self.encoder.usage is not None or failed:
                self.meter.finish(self.encoder.usage, error=failed)
            await self.meter.record()

    async def _stream(self, events: AsyncIterator[Any], send: Send) -> bool:
        """Send the response; True if the model failed."""
        try:
            item = await anext(events, _END)  # an event, or _PING if the model is slow to start
        except Exception as error:
            response = self.on_error(error)
            await send({"type": "http.response.start", "status": response.status_code, "headers": response.raw_headers})
            await send({"type": "http.response.body", "body": response.body, "more_body": False})
            return True

        async def emit(frames: list[bytes]) -> None:
            for frame in frames:
                await send({"type": "http.response.body", "body": frame, "more_body": True})

        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        await emit(self.encoder.opening())  # the opening frames stand in for the first keepalive
        if item is _PING:
            item = None
        failed = False
        while True:
            if item is _PING:
                await emit([self.encoder.heartbeat()])
            elif item is _END:
                await emit(self.encoder.closing())
                break
            elif item is not None:
                await emit(self.encoder.feed(item))
                if self.encoder.finished:
                    await emit(self.encoder.closing())
                    break
            try:
                item = await anext(events, _END)
            except Exception as error:
                failed = True
                await emit(self.encoder.failure(error))
                break
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return failed


class PaddedJSONResponse(Response):
    """A non-streaming JSON reply that keeps the connection open while it is being made.

    `produce()` runs the model call and returns the finished response (the reply, or the API's error).
    If it finishes within PAD_AFTER_SECONDS, that response goes out unchanged: its own status, headers
    and body. Otherwise 200 and this response's headers go out at once, then a space every
    KEEPALIVE_SECONDS, then the finished body. Leading whitespace is valid JSON, so clients parse it as
    usual. The cost: an error that arrives after the padding started keeps the API's error body but
    can't change the status, which is already 200 (streamed replies have the same limit).

    The client watcher lives in `collect`, which `produce` calls: one reader of `receive` at a time.
    """

    media_type = "application/json"

    def __init__(self, produce: Callable[[], Awaitable[Response]], *, headers: dict[str, str] | None = None):
        self.produce = produce
        self.status_code = 200
        self.background = None
        self.init_headers({"Cache-Control": "no-store", "X-Accel-Buffering": "no", **(headers or {})})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        result: Response | None = None
        failure: BaseException | None = None
        done = anyio.Event()
        started = False

        async def run() -> None:
            nonlocal result, failure
            try:
                result = await self.produce()
            except Exception as error:  # produce answers errors itself; anything else is re-raised below
                failure = error
            finally:
                done.set()

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            with anyio.move_on_after(PAD_AFTER_SECONDS):
                await done.wait()
            if not done.is_set():
                started = True
                await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
            while not done.is_set():
                with anyio.move_on_after(KEEPALIVE_SECONDS):
                    await done.wait()
                if not done.is_set():
                    try:
                        await send({"type": "http.response.body", "body": b" ", "more_body": True})
                    except OSError:  # the client has gone; collect's watcher stops the model
                        pass

        if failure is not None and not started:
            raise failure
        if not started:
            assert result is not None
            await result(scope, receive, send)
            return
        body = result.body if result is not None else b'{"error": {"message": "internal error"}}'
        await send({"type": "http.response.body", "body": body, "more_body": False})


def check_context(
    messages: list[Message], tools: list[dict[str, Any]] | None, num_ctx: int, *, anthropic: bool = False
) -> None:
    """Refuse a prompt that is certainly longer than the context window, with OpenAI's
    context_length_exceeded (or Anthropic's wording, which Claude Code recognises and compacts on).
    Some runners (Ollama's MLX engine) read the whole prompt anyway, slowly and with growing memory;
    others drop the oldest text without a word. Neither is what a client that trims its history on
    this error wants."""
    chars = len(json.dumps(tools, ensure_ascii=False)) if tools else 0
    for message in messages:
        chars += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            chars += len(json.dumps(call, ensure_ascii=False))
    at_least = chars // CHARS_PER_TOKEN_AT_MOST
    if at_least <= num_ctx:
        return
    if anthropic:
        message = f"prompt is too long: {at_least} tokens > {num_ctx} maximum"
    else:
        message = (
            f"This model's maximum context length is {num_ctx} tokens, but the messages and tools are at "
            f"least {at_least} tokens. Shorten the messages or the tools."
        )
    raise InvalidRequest(message, "messages", code="context_length_exceeded")


class StopScanner:
    """Finds stop sequences in streamed text ourselves, so the reply can report which one fired
    (Ollama strips the match and reports a plain "stop"). Text that could be the start of a stop
    sequence is held back until it can't be."""

    def __init__(self, stops: list[str] | None):
        self.stops = [s for s in (stops or []) if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.buffer = ""
        self.hit: str | None = None

    def feed(self, text: str) -> str:
        if not self.stops:
            return text
        self.buffer += text
        found = [(self.buffer.find(s), s) for s in self.stops if s in self.buffer]
        if found:
            index, stop = min(found)
            self.hit, out, self.buffer = stop, self.buffer[:index], ""
            return out
        cut = len(self.buffer) - self.hold
        if cut <= 0:
            return ""
        out, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return out

    def flush(self) -> str:
        out, self.buffer = self.buffer, ""
        return out


def estimate_tokens(chars: int) -> int:
    """A rough count (about four characters per token for English and code), for when no real count
    is available: count_tokens, and replies cut short by a stop sequence."""
    return (chars + 3) // 4


def tool_hint(names: list[str], forced: str | None) -> str:
    if forced:
        return f"You must respond by calling the tool `{forced}`."
    return f"You must respond by calling one of these tools: {', '.join(names)}."


def with_system_hint(messages: list[Message], hint: str) -> list[Message]:
    """Add an instruction to the system prompt (chat templates expect system first, so it goes there)."""
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0])
        first["content"] = f"{first['content']}\n\n{hint}" if first["content"] else hint
        return [first, *messages[1:]]
    return [{"role": "system", "content": hint}, *messages]
