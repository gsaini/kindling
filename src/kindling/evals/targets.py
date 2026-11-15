"""Where a case runs. Each target turns a case into an Output the checks can score.

    inproc   the model registry and the built-in tools in this process, through run_agent: model + tools
    http     a running Kindling's POST /api/chat with an API key: the whole stack
    openai   any OpenAI-compatible /v1/chat/completions (Kindling, Ollama, vLLM...); tools run here

API keys go into request headers only; they are never part of an Output or an error message.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anyio.to_thread
import httpx

from ..agent import run_agent
from ..llm import ChatModel, Message, ModelEvent, ModelRegistry, ToolCall, Usage
from ..tools import ToolRegistry, register_builtins
from .dataset import Case


@dataclass
class Output:
    text: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # name, arguments, output, ok
    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    first_token_s: float | None = None  # time to the first piece of reasoning or answer
    latency_s: float = 0.0
    steps: int = 0
    model: str = ""
    error: str | None = None


class Target(Protocol):
    name: str
    skip_tags: frozenset[str]  # cases with these tags can't run here (e.g. they need the fixture files)

    def describe(self) -> dict[str, Any]:
        """What to record about the target in the run summary. Never secrets."""

    async def run(self, case: Case, *, model: str | None, think: bool | None) -> Output: ...


# -- agent targets: run_agent in this process --------------------------------------------------


class AgentTarget:
    """Runs a case through run_agent with the built-in tools over a workspace folder."""

    skip_tags: frozenset[str] = frozenset()

    def __init__(
        self,
        name: str,
        resolve: Callable[[str | None], tuple[str, ChatModel, bool | None]],
        *,
        workspace: Path,
        max_steps: int = 8,
        system_prompt: str = "",
        url: str = "",
    ):
        self.name = name
        self.url = url
        self._resolve = resolve  # model id -> (id to report, model, its default think)
        self.workspace = workspace
        self.tools = register_builtins(ToolRegistry(), workspace)
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def describe(self) -> dict[str, Any]:
        return {"target": self.name, **({"url": self.url} if self.url else {}), "workspace": str(self.workspace)}

    async def run(self, case: Case, *, model: str | None, think: bool | None) -> Output:
        out = Output()
        started = time.perf_counter()
        try:
            out.model, chat_model, default_think = self._resolve(model)
            messages = case.chat_messages()
            if self.system_prompt and messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})  # as the server does
            async for event in run_agent(
                chat_model,
                messages,
                tools=self.tools if case.tools else None,
                think=think if think is not None else default_think,
                max_steps=self.max_steps,
            ):
                if event.type in ("thinking", "content") and out.first_token_s is None:
                    out.first_token_s = time.perf_counter() - started
                if event.type == "thinking":
                    out.thinking += event.text
                elif event.type == "content":
                    out.text += event.text
                elif event.type == "tool_call":
                    out.tool_calls.append({"name": event.name, "arguments": event.arguments})
                elif event.type == "tool_result":
                    _attach_result(out.tool_calls, event.name, event.text, event.ok)
                elif event.type == "done":
                    usage = event.usage or Usage()
                    out.prompt_tokens, out.output_tokens = usage.prompt_tokens, usage.output_tokens
                    out.tokens_per_second = round(usage.tokens_per_second, 1)
                    out.steps = event.steps
        except Exception as error:  # the model went away, the step limit, a bad reply: a failed case
            out.error = _error_text(error)
        out.latency_s = time.perf_counter() - started
        return out


def _attach_result(calls: list[dict[str, Any]], name: str, output: str, ok: bool) -> None:
    for call in calls:
        if call["name"] == name and "output" not in call:
            call["output"], call["ok"] = output[:2000], ok
            return


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def inproc_target(
    registry: ModelRegistry, *, workspace: Path, max_steps: int = 8, system_prompt: str = ""
) -> AgentTarget:
    def resolve(model_id: str | None) -> tuple[str, ChatModel, bool | None]:
        spec = registry.spec(model_id)
        return spec.id, registry.get(spec.id), spec.think

    return AgentTarget("inproc", resolve, workspace=workspace, max_steps=max_steps, system_prompt=system_prompt)


# -- http: a running Kindling ----------------------------------------------------------------


class HttpTarget:
    """POST /api/chat on a Kindling server and read its NDJSON events.

    It uses a synchronous httpx client on a worker thread, so a FastAPI TestClient can stand in for
    the network in tests."""

    name = "http"
    # The server's tools read the server's workspace, not evals/fixtures.
    skip_tags = frozenset({"fixtures"})

    def __init__(self, url: str, api_key: str = "", *, client: httpx.Client | None = None, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(base_url=self.url, timeout=timeout)

    def describe(self) -> dict[str, Any]:
        return {"target": self.name, "url": self.url}

    async def run(self, case: Case, *, model: str | None, think: bool | None) -> Output:
        return await anyio.to_thread.run_sync(lambda: self._run(case, model, think))

    def _run(self, case: Case, model: str | None, think: bool | None) -> Output:
        out = Output(model=model or "")
        body: dict[str, Any] = {"messages": case.chat_messages(), "tools": case.tools}
        if model:
            body["model"] = model
        if think is not None:
            body["think"] = think
        started = time.perf_counter()
        try:
            with self._client.stream("POST", "/api/chat", json=body, headers=self._headers) as response:
                if response.status_code != 200:
                    response.read()
                    raise RuntimeError(f"HTTP {response.status_code}: {_detail(response)}")
                for line in response.iter_lines():
                    if line.strip():
                        self._apply(out, json.loads(line), started)
        except Exception as error:
            out.error = out.error or _error_text(error)
        out.latency_s = time.perf_counter() - started
        return out

    @staticmethod
    def _apply(out: Output, event: dict[str, Any], started: float) -> None:
        kind = event.get("type")
        if kind in ("thinking", "content") and out.first_token_s is None:
            out.first_token_s = time.perf_counter() - started
        if kind == "thinking":
            out.thinking += event.get("text", "")
        elif kind == "content":
            out.text += event.get("text", "")
        elif kind == "tool_call":
            out.tool_calls.append({"name": event.get("name", ""), "arguments": event.get("arguments") or {}})
        elif kind == "tool_result":
            _attach_result(out.tool_calls, event.get("name", ""), event.get("output", ""), event.get("ok", True))
        elif kind == "done":
            stats = event.get("stats") or {}
            out.model = stats.get("model", out.model)
            out.prompt_tokens = stats.get("prompt_tokens", 0)
            out.output_tokens = stats.get("output_tokens", 0)
            out.tokens_per_second = stats.get("tokens_per_second", 0.0)
            out.steps = stats.get("steps", 0)
        elif kind == "error":
            out.error = event.get("message", "error")


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text
    return str(detail)[:300]


# -- openai: any OpenAI-compatible server ----------------------------------------------------


class OpenAIChatModel:
    """A ChatModel over /v1/chat/completions (streaming), so run_agent can drive any such server.

    Reasoning arrives as `reasoning_content` (vLLM), `reasoning` (Ollama) or `thinking`. To switch it,
    `think` is sent both as Ollama's `think` and as vLLM's chat_template_kwargs.enable_thinking (the
    Nemotron templates read that); servers ignore what they don't know."""

    def __init__(
        self,
        url: str,
        name: str,
        *,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 600.0,
    ):
        self.url = url.rstrip("/")
        self.name = name
        self.num_ctx = 0
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._transport = transport
        self._timeout = timeout

    async def check(self) -> None:
        pass

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        think: bool | None = None,
        **_: Any,
    ) -> AsyncIterator[ModelEvent]:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": to_openai_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = list(tools)
        if think is not None:
            body["think"] = think
            body["chat_template_kwargs"] = {"enable_thinking": think}
        started = time.perf_counter()
        first: float | None = None
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        async with (
            httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client,
            client.stream("POST", f"{self.url}/chat/completions", json=body, headers=self._headers) as response,
        ):
            if response.status_code != 200:
                await response.aread()
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"])[:300])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
                    for kind, text in (("thinking", reasoning), ("content", delta.get("content"))):
                        if text:
                            first = first or time.perf_counter()
                            yield ModelEvent(kind, text)
                    for piece in delta.get("tool_calls") or []:
                        call = calls.setdefault(piece.get("index", len(calls)), {"name": "", "arguments": ""})
                        function = piece.get("function") or {}
                        call["name"] += function.get("name") or ""
                        call["arguments"] += function.get("arguments") or ""
        if calls:
            yield ModelEvent("tool_calls", tool_calls=[_parse_call(c) for _, c in sorted(calls.items())])
        ended = time.perf_counter()
        yield ModelEvent(
            "done",
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                prompt_seconds=(first or ended) - started,
                output_seconds=ended - (first or ended),
            ),
        )


def _parse_call(call: dict[str, Any]) -> ToolCall:
    try:
        arguments = json.loads(call["arguments"]) if call["arguments"].strip() else {}
    except json.JSONDecodeError:
        arguments = {"_raw": call["arguments"]}  # let the tool report the bad arguments to the model
    return ToolCall(call["name"], arguments if isinstance(arguments, dict) else {"_raw": arguments})


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Ollama-style history (what run_agent keeps) to OpenAI's: tool calls need ids, results refer to them."""
    converted: list[dict[str, Any]] = []
    pending: list[str] = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            wire = []
            for n, call in enumerate(message["tool_calls"]):
                call_id = f"call_{index}_{n}"
                pending.append(call_id)
                function = call["function"]
                wire.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": function["name"], "arguments": json.dumps(function["arguments"])},
                    }
                )
            converted.append({"role": "assistant", "content": message.get("content") or None, "tool_calls": wire})
        elif role == "tool":
            call_id = pending.pop(0) if pending else f"call_{index}"
            converted.append({"role": "tool", "tool_call_id": call_id, "content": message.get("content", "")})
        else:
            converted.append({"role": role, "content": message.get("content", "")})
    return converted


def openai_target(
    url: str,
    *,
    api_key: str = "",
    workspace: Path,
    max_steps: int = 8,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 600.0,
) -> AgentTarget:
    models: dict[str, OpenAIChatModel] = {}

    def resolve(model_id: str | None) -> tuple[str, ChatModel, bool | None]:
        if not model_id:
            raise ValueError("the openai target needs a model name: pass --model or set it in the case")
        if model_id not in models:
            models[model_id] = OpenAIChatModel(url, model_id, api_key=api_key, transport=transport, timeout=timeout)
        return model_id, models[model_id], None

    return AgentTarget("openai", resolve, workspace=workspace, max_steps=max_steps, url=url.rstrip("/"))
