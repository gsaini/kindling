"""The concurrency gate: limit, bounded queue, timeout, and slots given back on every way out."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from kindling.config import Settings
from kindling.llm import ModelEvent, Usage
from kindling.ops import GatedModel, GenerationGate, Overloaded, model_wrapper

pytestmark = pytest.mark.anyio


class HeldModel:
    """Yields one chunk, then waits for `go` before finishing, so tests control how long a slot is held."""

    name = "held"
    num_ctx = 1234

    def __init__(self) -> None:
        self.go = asyncio.Event()
        self.closed = 0
        self.checked = False

    async def check(self) -> None:
        self.checked = True

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[ModelEvent]:
        try:
            yield ModelEvent("content", "a")
            await self.go.wait()
            yield ModelEvent("done", usage=Usage(3, 2))
        finally:
            self.closed += 1


def gated(limit: int = 1, queued: int = 1, timeout: float = 5.0) -> tuple[GatedModel, HeldModel, GenerationGate]:
    inner = HeldModel()
    gate = GenerationGate(limit, queued, timeout)
    return GatedModel(inner, gate), inner, gate


async def test_wrapped_model_keeps_its_interface():
    model, inner, _ = gated()
    assert model.name == "held" and model.num_ctx == 1234
    await model.check()
    assert inner.checked
    assert model.go is inner.go  # other attributes pass through


async def test_limit_queue_and_overload():
    model, inner, gate = gated(limit=1, queued=1)
    first = model.stream([])
    assert (await anext(first)).text == "a"
    assert gate.running == 1

    second = model.stream([])
    waiting = asyncio.create_task(anext(second))
    await asyncio.sleep(0.01)
    assert gate.waiting == 1 and not waiting.done()

    with pytest.raises(Overloaded, match="busy"):
        await anext(model.stream([]))  # queue full: turned away at once

    inner.go.set()
    assert (await anext(first)).kind == "done"  # done frees the slot ...
    assert (await waiting).text == "a"  # ... and the next in line gets it
    assert gate.running == 1 and gate.waiting == 0
    assert [e.kind async for e in second] == ["done"]
    assert gate.running == 0


async def test_queue_timeout():
    model, _, gate = gated(limit=1, queued=4, timeout=0.05)
    first = model.stream([])
    await anext(first)
    with pytest.raises(Overloaded, match=r"0\.05s"):
        await anext(model.stream([]))
    assert gate.waiting == 0 and gate.running == 1
    await first.aclose()
    assert gate.running == 0


async def test_slot_released_when_closed_early():
    model, inner, gate = gated(limit=1, queued=0)
    stream = model.stream([])
    await anext(stream)
    assert gate.running == 1
    await stream.aclose()  # what a client disconnect does to the response's generator
    assert gate.running == 0
    assert inner.closed == 1  # the upstream stream is closed too
    await anext(model.stream([]))  # and the slot is usable again
    assert gate.running == 1


async def test_slot_released_when_cancelled():
    model, _, gate = gated(limit=1, queued=0)

    async def consume() -> None:
        async for _ in model.stream([]):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    assert gate.running == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.running == 0


async def test_cancelled_while_queued_leaves_the_queue():
    model, _, gate = gated(limit=1, queued=2)
    first = model.stream([])
    await anext(first)
    waiting = asyncio.create_task(anext(model.stream([])))
    await asyncio.sleep(0.01)
    assert gate.waiting == 1
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert gate.waiting == 0
    await first.aclose()
    assert gate.running == 0


async def test_model_wrapper_shares_one_gate(tmp_path):
    settings = Settings(_env_file=None, max_concurrent_generations=1, max_queued_generations=0)
    wrap = model_wrapper(settings)
    assert wrap is not None
    a, b = wrap(HeldModel()), wrap(HeldModel())
    stream = a.stream([])
    await anext(stream)
    with pytest.raises(Overloaded):
        await anext(b.stream([]))  # a different model, the same machine
    await stream.aclose()
