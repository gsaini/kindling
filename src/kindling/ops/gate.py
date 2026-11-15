"""The concurrency gate: one shared limit on model streams, across every model and every API.

Ollama serves a few requests in parallel (OLLAMA_NUM_PARALLEL) and queues the rest without limit, so a
burst of requests would pile up there with no feedback. The gate keeps that queue here, bounded:
at most `limit` streams run, at most `max_queued` wait, each waits at most `timeout` seconds, and the
rest get `Overloaded` at once, which the APIs turn into 503 (529 for the Anthropic API).

A slot is taken on the first `__anext__` of a stream, not when `stream()` is called, and given back
when the model's `done` event arrives, when the stream ends, or when the consumer closes it early.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from . import metrics

if TYPE_CHECKING:
    from ..llm import ChatModel, Message, ModelEvent


class Overloaded(RuntimeError):
    """Too many generations running and waiting. `retry_after` is a hint in seconds for the client."""

    def __init__(self, message: str, *, retry_after: int = 5):
        super().__init__(message)
        self.retry_after = retry_after


class Slot:
    """A held place in the gate. `release()` may be called more than once."""

    def __init__(self, gate: GenerationGate):
        self._gate = gate
        self._held = True
        self.started = time.perf_counter()

    def release(self) -> None:
        if self._held:
            self._held = False
            self._gate._release()


class GenerationGate:
    def __init__(self, limit: int, max_queued: int, timeout: float):
        self.limit = max(1, limit)
        self.max_queued = max(0, max_queued)
        self.timeout = timeout
        self.running = 0
        self.waiting = 0
        self._semaphore = asyncio.Semaphore(self.limit)  # FIFO, so the queue is fair

    async def acquire(self) -> Slot:
        if self.waiting == 0 and not self._semaphore.locked():
            await self._semaphore.acquire()  # a free slot and nobody ahead: no wait
        else:
            if self.waiting >= self.max_queued:
                metrics.GENERATIONS_REJECTED.labels("queue_full").inc()
                raise Overloaded(
                    f"the server is busy: {self.running} generations running and {self.waiting} waiting. "
                    "Try again shortly."
                )
            self.waiting += 1
            metrics.GENERATIONS_QUEUED.inc()
            started = time.perf_counter()
            try:
                async with asyncio.timeout(self.timeout):
                    await self._semaphore.acquire()
            except TimeoutError:
                metrics.GENERATIONS_REJECTED.labels("timeout").inc()
                raise Overloaded(
                    f"the server is busy: no generation slot came free within {self.timeout:g}s. Try again shortly."
                ) from None
            finally:
                self.waiting -= 1
                metrics.GENERATIONS_QUEUED.dec()
                metrics.QUEUE_WAIT.observe(time.perf_counter() - started)
        self.running += 1
        metrics.GENERATIONS_IN_FLIGHT.inc()
        return Slot(self)

    def _release(self) -> None:
        self.running -= 1
        metrics.GENERATIONS_IN_FLIGHT.dec()
        self._semaphore.release()


class GatedModel:
    """A ChatModel whose streams go through the gate; also counts tokens for the metrics."""

    def __init__(self, inner: ChatModel, gate: GenerationGate):
        self.inner = inner
        self.gate = gate

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def num_ctx(self) -> int:
        return self.inner.num_ctx

    async def check(self) -> None:
        await self.inner.check()  # not gated: health checks must answer while the model is busy

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self.inner, attribute)  # anything else a provider offers

    async def stream(self, messages: Sequence[Message], **kwargs: Any) -> AsyncIterator[ModelEvent]:
        slot = await self.gate.acquire()
        events = self.inner.stream(messages, **kwargs)
        try:
            async for event in events:
                if event.kind == "done":
                    # Generation is over: free the slot now, even if the consumer never asks for more.
                    self._observe(slot, event)
                    slot.release()
                yield event
        finally:
            slot.release()
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    def _observe(self, slot: Slot, event: ModelEvent) -> None:
        model = self.inner.name
        metrics.GENERATIONS.labels(model).inc()
        metrics.GENERATION_SECONDS.labels(model).observe(time.perf_counter() - slot.started)
        if event.usage is not None:
            metrics.PROMPT_TOKENS.labels(model).inc(event.usage.prompt_tokens)
            metrics.OUTPUT_TOKENS.labels(model).inc(event.usage.output_tokens)
