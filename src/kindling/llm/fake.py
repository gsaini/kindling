"""A scripted stand-in for a real model, so tests and evals of the plumbing run without Ollama."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import Message, ModelEvent, ToolCall, Usage


@dataclass
class Turn:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(10, 5, 0.1, 0.5))


Step = Turn | Callable[[list[Message], dict[str, Any]], Turn]


class FakeModel:
    name = "fake"
    num_ctx = 4096

    def __init__(self, *script: Step):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def check(self) -> None:
        pass

    async def stream(self, messages: Sequence[Message], **kwargs: Any) -> AsyncIterator[ModelEvent]:
        messages = [dict(m) for m in messages]
        self.calls.append({"messages": messages, **kwargs})
        if not self.script:
            raise AssertionError("FakeModel ran out of scripted turns")
        step = self.script.pop(0)
        turn = step(messages, kwargs) if callable(step) else step
        if turn.thinking:
            yield ModelEvent("thinking", turn.thinking)
        # Split the answer in two so consumers see more than one chunk.
        half = len(turn.content) // 2
        for piece in (turn.content[:half], turn.content[half:]):
            if piece:
                yield ModelEvent("content", piece)
        if turn.tool_calls:
            yield ModelEvent("tool_calls", tool_calls=list(turn.tool_calls))
        yield ModelEvent("done", usage=turn.usage)
