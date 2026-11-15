"""The agent loop: stream a model turn, run the tools it asks for, send the results back, repeat
until it answers without a tool call (or runs out of steps).

Every step is reported as an event, so the HTTP layer, the eval runner and tests all see the same thing:

    thinking     a piece of reasoning
    content      a piece of the answer
    tool_call    the model asked for a tool (name, arguments)
    tool_result  what the tool returned (output, ok)
    done         the turn is over: usage, steps, and the messages to append to the history
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .llm import ChatModel, Message, ToolCall, Usage
from .tools import ToolRegistry


@dataclass
class AgentEvent:
    type: Literal["thinking", "content", "tool_call", "tool_result", "done"]
    text: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    usage: Usage | None = None
    steps: int = 0
    wall_seconds: float = 0.0
    new_messages: list[Message] = field(default_factory=list)


class StepLimitReached(RuntimeError):
    def __init__(self, message: str, usage: Usage):
        super().__init__(message)
        self.usage = usage  # tokens spent before giving up still count against quotas


async def run_agent(
    model: ChatModel,
    messages: Sequence[Message],
    *,
    tools: ToolRegistry | None = None,
    think: bool | None = None,
    max_steps: int = 8,
    options: dict[str, Any] | None = None,
) -> AsyncIterator[AgentEvent]:
    start = time.perf_counter()
    history = [dict(m) for m in messages]
    new_messages: list[Message] = []
    usage = Usage()
    schemas = tools.schemas() if tools and len(tools) else None

    for step in range(1, max_steps + 1):
        content, thinking, calls = "", "", []
        async for event in model.stream(history, tools=schemas, think=think, options=options):
            if event.kind == "thinking":
                thinking += event.text
                yield AgentEvent("thinking", text=event.text)
            elif event.kind == "content":
                content += event.text
                yield AgentEvent("content", text=event.text)
            elif event.kind == "tool_calls":
                calls.extend(event.tool_calls)
            elif event.usage:
                usage = usage + event.usage

        # Keep the reasoning within this turn so the model can follow its own plan between
        # tool calls; it is dropped from history the client sends next time.
        assistant: Message = {"role": "assistant", "content": content}
        if thinking:
            assistant["thinking"] = thinking
        if calls:
            assistant["tool_calls"] = [_wire_call(c) for c in calls]
        history.append(assistant)
        new_messages.append(assistant)

        if not calls or tools is None:
            yield AgentEvent(
                "done", usage=usage, steps=step, wall_seconds=time.perf_counter() - start, new_messages=new_messages
            )
            return

        for call in calls:
            yield AgentEvent("tool_call", name=call.name, arguments=call.arguments)
            result = await tools.call(call.name, call.arguments)
            yield AgentEvent("tool_result", name=call.name, text=result.output, ok=result.ok)
            tool_message: Message = {"role": "tool", "tool_name": call.name, "content": result.output}
            history.append(tool_message)
            new_messages.append(tool_message)

    raise StepLimitReached(f"the model was still calling tools after {max_steps} steps", usage)


def _wire_call(call: ToolCall) -> dict[str, Any]:
    return {"function": {"name": call.name, "arguments": call.arguments}}
