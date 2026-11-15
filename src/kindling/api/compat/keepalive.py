"""Keepalives for streamed replies: something to send while the model is silent.

Cloudflare cuts a proxied request (524) after about 125 s without a byte from the origin, and the
model can be silent that long: loading, waiting in the generation queue, reading a long prompt.

    async for item in with_keepalive(model.stream(...), 15.0, lambda: PING):
        ...  # a model event, or PING after 15 s of silence

The source is iterated in a task of its own, so a timer can run beside it, and that task is always
cancelled and awaited when the iteration ends: at the end of the source, on an error (re-raised to
the consumer), when the consumer stops early, and when the consumer is cancelled (a client disconnect).
Running the whole source in one task also keeps code inside it that is bound to a task (asyncio
timeouts, the ops gate) working as usual. asyncio only, like the rest of Kindling.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable

import anyio

_ITEM, _END, _ERROR = 0, 1, 2


async def _aclose(source: AsyncIterable[object]) -> None:
    closer = getattr(source, "aclose", None)
    if closer is not None:
        await closer()


async def with_keepalive[T, P](
    source: AsyncIterable[T],
    interval: float,
    make_ping: Callable[[], P],
    *,
    first_interval: float | None = None,
) -> AsyncIterator[T | P]:
    """Yield the source's items, and `make_ping()` after every `interval` seconds without one.
    `first_interval` (default: `interval`) is the wait before the first ping."""
    queue: asyncio.Queue[tuple[int, object]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in source:
                queue.put_nowait((_ITEM, item))
            queue.put_nowait((_END, None))
        except Exception as error:
            queue.put_nowait((_ERROR, error))
        finally:
            await _aclose(source)  # in this task, where the source has been running

    task = asyncio.get_running_loop().create_task(pump())
    wait = interval if first_interval is None else first_interval
    try:
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), wait)  # a cancelled get loses nothing
            except TimeoutError:
                yield make_ping()
                wait = interval
                continue
            if kind == _END:
                return
            if kind == _ERROR:
                assert isinstance(value, Exception)
                raise value
            wait = interval
            yield value  # type: ignore[misc]
    finally:
        task.cancel()
        with anyio.CancelScope(shield=True):  # let the source close even when we are being cancelled
            await asyncio.wait({task})
