"""The model interface the rest of Kindling talks to. Ollama is one implementation; tests use a fake."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    prompt_seconds: float = 0.0  # prefill: reading the prompt
    output_seconds: float = 0.0  # generation: writing the reply

    def __add__(self, other: Usage) -> Usage:
        # Prompt tokens are not summed: the last step's prompt is the whole context so far.
        return Usage(
            prompt_tokens=other.prompt_tokens or self.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            prompt_seconds=self.prompt_seconds + other.prompt_seconds,
            output_seconds=self.output_seconds + other.output_seconds,
        )

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.output_seconds if self.output_seconds else 0.0

    @property
    def prefill_per_second(self) -> float:
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0


@dataclass
class ModelEvent:
    """One piece of a streamed reply. `done` comes last and carries the usage."""

    kind: Literal["thinking", "content", "tool_calls", "done"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


@dataclass
class Completion:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


Message = dict[str, Any]  # Ollama chat format: role, content, optional thinking / tool_calls / tool_name


class ModelNotReady(RuntimeError):
    pass


class ChatModel(Protocol):
    name: str
    num_ctx: int

    async def check(self) -> None:
        """Raise ModelNotReady with a useful message if the model can't be used."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        think: bool | None = None,
        format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        """Stream one model turn. `format` is "json" or a JSON schema the reply must follow."""
        ...


async def complete(model: ChatModel, messages: Sequence[Message], **kwargs: Any) -> Completion:
    """Run a stream to the end and gather it into one Completion."""
    result = Completion()
    async for event in model.stream(messages, **kwargs):
        if event.kind == "thinking":
            result.thinking += event.text
        elif event.kind == "content":
            result.content += event.text
        elif event.kind == "tool_calls":
            result.tool_calls.extend(event.tool_calls)
        elif event.usage:
            result.usage = event.usage
    return result
