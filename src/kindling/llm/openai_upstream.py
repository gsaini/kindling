"""ChatModel over an OpenAI-compatible server (vLLM, NVIDIA NIM, llama.cpp, LM Studio...).

For a bigger Nemotron on a GPU box behind Kindling:

    [models.nemotron-super]
    provider = "openai"
    name = "nvidia/nemotron-super"          # the upstream model name, as its /models lists it
    host = "http://gpu-box:8000/v1"         # the server's base URL, including /v1
    api_key_env = "NIM_API_KEY"             # optional: sent as a Bearer token

`think` is sent as `chat_template_kwargs: {"enable_thinking": think}`, the switch vLLM (and NIM's
vLLM-based containers) use for Nemotron's chat template. Servers that don't know the field ignore it.
Reasoning is read from `reasoning_content` (vLLM, DeepSeek) or `reasoning` (newer vLLM, llama.cpp).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from ..media import to_part
from .base import Message, ModelEvent, ModelNotReady, ToolCall, Usage

if TYPE_CHECKING:
    from .registry import ModelSpec

# Ollama option -> OpenAI request field (vLLM accepts top_k, min_p and repetition_penalty as extras).
_OPTIONS = {
    "num_predict": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "repeat_penalty": "repetition_penalty",
}


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Ollama-format messages -> OpenAI chat messages. Ollama links a tool result to its call by
    name only, so ids are made up here and each result takes the oldest open call with its name."""
    out: list[dict[str, Any]] = []
    open_calls: list[tuple[str, str]] = []  # (id, name), oldest first
    counter = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                counter += 1
                call_id = call.get("id") or f"call_{counter}"
                open_calls.append((call_id, function.get("name", "")))
                arguments = function.get("arguments") or {}
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                        },
                    }
                )
            if calls:
                entry["tool_calls"] = calls
                entry["content"] = entry["content"] or None
            out.append(entry)
        elif role == "tool":
            name = message.get("tool_name") or message.get("name")
            match = next((c for c in open_calls if c[1] == name), open_calls[0] if open_calls else None)
            if match:
                open_calls.remove(match)
                call_id = match[0]
            else:
                counter += 1
                call_id = f"call_{counter}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": message.get("content") or ""})
        elif message.get("media"):
            # Media first, then the text: the order the model was tested with.
            parts = [to_part(item) for item in message["media"]]
            if message.get("content"):
                parts.append({"type": "text", "text": message["content"]})
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": message.get("content") or ""})
    return out


def _arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        value = None
    # Kindling's tool calls carry a dict; keep a malformed reply visible rather than dropping it.
    return value if isinstance(value, dict) else {"raw": raw}


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or body.get("detail") or body)[:500]


class OpenAICompatModel:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.spec = spec
        self.name = spec.name
        self.num_ctx = spec.num_ctx
        self.base_url = (spec.host or "").rstrip("/")
        # Generous read timeout: a big model can take a while before the first token.
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0)
        self.client = client or httpx.AsyncClient(transport=transport, timeout=timeout)

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.spec.api_key_env or "", "") if self.spec.api_key_env else ""
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _require_host(self) -> None:
        if not self.base_url:
            raise ModelNotReady(
                f"model {self.spec.id!r} uses the openai provider but has no host "
                '(the server\'s base URL, e.g. host = "http://gpu-box:8000/v1")'
            )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def check(self) -> None:
        self._require_host()
        try:
            response = await self.client.get(f"{self.base_url}/models", headers=self._headers())
        except httpx.HTTPError as error:
            raise ModelNotReady(f"Can't reach {self.base_url} ({type(error).__name__}: {error})") from error
        if response.status_code in (401, 403):
            hint = f"check {self.spec.api_key_env}" if self.spec.api_key_env else "set api_key_env for this model"
            raise ModelNotReady(f"{self.base_url} refused the API key ({response.status_code}): {hint}")
        if response.status_code >= 400:
            raise ModelNotReady(f"{self.base_url}/models returned {response.status_code}: {_error_message(response)}")
        served = [m.get("id") for m in response.json().get("data", []) if isinstance(m, dict)]
        if self.name not in served:
            raise ModelNotReady(
                f"{self.base_url} does not serve {self.name!r}; it serves: {', '.join(map(str, served))}"
            )

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        think: bool | None,
        format: dict[str, Any] | str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": to_openai_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = [{"type": "function", **t} if "type" not in t else t for t in tools]
        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": think}
        if format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(format, dict):
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": format}}
        for option, value in (options or {}).items():
            if option in _OPTIONS and value is not None:
                payload[_OPTIONS[option]] = value
        return payload

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        think: bool | None = None,
        format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        self._require_host()
        payload = self._payload(messages, tools, think, format, options)
        started = time.perf_counter()
        first: float | None = None
        calls: dict[int, dict[str, str]] = {}  # tool call deltas, gathered by index
        usage: dict[str, Any] = {}
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise RuntimeError(f"{self.base_url} returned {response.status_code}: {_error_message(response)}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # blank separators, comments, event: lines
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise RuntimeError(f"{self.base_url}: {chunk['error'].get('message', chunk['error'])}")
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if (reasoning or delta.get("content") or delta.get("tool_calls")) and first is None:
                            first = time.perf_counter()
                        if reasoning:
                            yield ModelEvent("thinking", reasoning)
                        if delta.get("content"):
                            yield ModelEvent("content", delta["content"])
                        for part in delta.get("tool_calls") or []:
                            slot = calls.setdefault(
                                part.get("index", len(calls)), {"id": "", "name": "", "arguments": ""}
                            )
                            function = part.get("function") or {}
                            slot["id"] = part.get("id") or slot["id"]
                            slot["name"] += function.get("name") or ""
                            slot["arguments"] += function.get("arguments") or ""
        except httpx.HTTPError as error:
            raise RuntimeError(f"Can't reach {self.base_url} ({type(error).__name__}: {error})") from error

        if calls:
            yield ModelEvent(
                "tool_calls",
                tool_calls=[
                    ToolCall(c["name"], _arguments(c["arguments"]), id=c["id"]) for _, c in sorted(calls.items())
                ],
            )
        ended = time.perf_counter()
        first = first or ended
        yield ModelEvent(
            "done",
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                prompt_seconds=first - started,  # time to first token: mostly prefill
                output_seconds=ended - first,
            ),
        )
