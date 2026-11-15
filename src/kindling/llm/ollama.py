"""ChatModel over Ollama's chat API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from .base import Message, ModelEvent, ModelNotReady, ToolCall, Usage


def _seconds(ns: int | None) -> float:
    return (ns or 0) / 1e9


class OllamaModel:
    def __init__(self, name: str, *, host: str | None = None, num_ctx: int = 16384, client: Any = None):
        if client is None:
            from ollama import AsyncClient

            client = AsyncClient(host=host)
        self.client = client
        self.name = name
        self.num_ctx = num_ctx

    async def check(self) -> None:
        try:
            listing = await self.client.list()
        except Exception as error:  # connection refused, mostly
            raise ModelNotReady(
                f"Can't reach Ollama ({error}). Start it with `ollama serve` or the Ollama app."
            ) from error
        names = {m.model for m in listing.models}
        wanted = self.name if ":" in self.name else f"{self.name}:latest"
        if wanted not in names:
            raise ModelNotReady(f"{self.name} isn't downloaded yet. Run: ollama pull {self.name}")

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        think: bool | None = None,
        format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        stream = await self.client.chat(
            model=self.name,
            messages=list(messages),
            tools=list(tools) if tools else None,
            think=think,
            format=format,
            stream=True,
            options={"num_ctx": self.num_ctx, **(options or {})},
        )
        async for chunk in stream:
            message = chunk.message
            if message.thinking:
                yield ModelEvent("thinking", message.thinking)
            if message.content:
                yield ModelEvent("content", message.content)
            if message.tool_calls:
                calls = [ToolCall(c.function.name, dict(c.function.arguments or {})) for c in message.tool_calls]
                yield ModelEvent("tool_calls", tool_calls=calls)
            if chunk.done:
                yield ModelEvent(
                    "done",
                    usage=Usage(
                        prompt_tokens=chunk.prompt_eval_count or 0,
                        output_tokens=chunk.eval_count or 0,
                        prompt_seconds=_seconds(chunk.prompt_eval_duration),
                        output_seconds=_seconds(chunk.eval_duration),
                    ),
                )
