"""Tools the model can call. A tool is a plain function; its signature becomes the JSON schema.

registry = ToolRegistry()

@registry.tool
def add(a: Annotated[float, Field(description="first number")], b: float) -> float:
    "Add two numbers."
    return a + b
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio.to_thread
from pydantic import BaseModel, ValidationError, create_model


@dataclass
class ToolResult:
    name: str
    output: str
    ok: bool = True


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]

    @property
    def schema(self) -> dict[str, Any]:
        """The tool definition in Ollama / OpenAI function format."""
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }


def _args_model(fn: Callable[..., Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn, eval_str=True).parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    return create_model(f"{fn.__name__}_args", **fields)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
        """Register a function as a tool. Works bare (`@registry.tool`) or with arguments."""

        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            doc = description or inspect.getdoc(f) or ""
            tool = Tool(name or f.__name__, doc.strip(), f, _args_model(f))
            self._tools[tool.name] = tool
            return f

        return register(fn) if fn is not None else register

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool. Errors come back as text for the model to read, never as exceptions."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, f"error: no tool named {name!r}. Available: {', '.join(self._tools)}", ok=False)
        try:
            args = tool.args_model.model_validate(arguments or {}).model_dump()
        except ValidationError as error:
            return ToolResult(name, f"error: bad arguments: {error.errors(include_url=False)}", ok=False)
        try:
            if inspect.iscoroutinefunction(tool.fn):
                value = await tool.fn(**args)
            else:
                value = await anyio.to_thread.run_sync(lambda: tool.fn(**args))
        except Exception as error:  # the model should see what went wrong and may retry
            return ToolResult(name, f"error: {type(error).__name__}: {error}", ok=False)
        return ToolResult(name, value if isinstance(value, str) else json.dumps(value, default=str))
