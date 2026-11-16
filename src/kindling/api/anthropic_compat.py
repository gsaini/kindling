"""Anthropic-compatible API: POST /v1/messages and /v1/messages/count_tokens, for Anthropic SDKs and Claude Code.

Like the OpenAI-compatible API, this is plain model access: `tools` are the client's own tools.

- Model names: an id or alias from models.toml. An unknown name starting with "claude-" is served by
  the default model (Claude Code asks for Claude models by name, for its main and its background
  work), and the reply echoes the name that was asked for. Any other unknown name is a 404.
- Thinking blocks come first and carry an empty `signature`; thinking sent back by the client is dropped.
- Stop sequences are matched here rather than by the model, so the reply can say which one fired.
- tool_choice: "none" offers no tools; "any" and {"type": "tool"} can't be enforced on Ollama, so they
  narrow the tool list and add an instruction to the system prompt (best effort).
See docs/api.md for the field mapping and the gaps.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from ..llm import ChatModel, Message, ModelEvent, ModelSpec, Usage
from .compat import errors
from .compat.errors import InvalidRequest
from .compat.streaming import (
    ClientGone,
    Meter,
    ModelCall,
    ModelStreamResponse,
    PaddedJSONResponse,
    StopScanner,
    check_context,
    collect,
    estimate_tokens,
    tool_hint,
    with_system_hint,
)
from .deps import Principal, Services, require_scope, services

router = APIRouter(prefix="/v1", tags=["anthropic"])

ENDPOINT = "/v1/messages"


class CountTokensRequest(BaseModel):
    # Claude Code sends fields this server doesn't use (metadata, context_management, cache_control...).
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None


class MessagesRequest(CountTokensRequest):
    max_tokens: int = Field(ge=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    output_config: dict[str, Any] | None = None


# -- request translation ------------------------------------------------------------------------


def _blocks(content: Any, param: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(b, dict) for b in content):
        return content
    raise InvalidRequest(f"{param}: content must be a string or an array of content blocks")


def _unsupported(kind: Any, param: str) -> InvalidRequest:
    if kind in ("image", "document"):
        # Images and audio work on /v1/chat/completions; this endpoint takes text only for now.
        return InvalidRequest(
            f"{param}: {kind} content is not supported yet here: send text, or use /v1/chat/completions"
        )
    return InvalidRequest(f"{param}: unsupported content block type {kind!r}")


def _system_text(system: str | list[dict[str, Any]] | None) -> str:
    if system is None or isinstance(system, str):
        return system or ""
    parts = []
    for i, block in enumerate(system):
        if block.get("type") != "text":
            raise _unsupported(block.get("type"), f"system.{i}")
        parts.append(block.get("text") or "")
    return "\n\n".join(p for p in parts if p)


def _tool_result_text(block: dict[str, Any], param: str) -> str:
    content = block.get("content")
    if content is None or isinstance(content, str):
        text = content or ""
    else:
        parts = []
        for i, inner in enumerate(_blocks(content, param)):
            if inner.get("type") != "text":
                raise _unsupported(inner.get("type"), f"{param}.content.{i}")
            parts.append(inner.get("text") or "")
        text = "\n".join(parts)
    return f"Error: {text}" if block.get("is_error") else text


def to_ollama_messages(system: Any, messages: list[dict[str, Any]]) -> list[Message]:
    """Anthropic messages -> Ollama's format. A tool_result becomes a tool message named after the
    tool_use (by id) in an earlier assistant turn."""
    out: list[Message] = []
    if text := _system_text(system):
        out.append({"role": "system", "content": text})
    names: dict[str, str] = {}  # tool_use id -> tool name
    for i, message in enumerate(messages):
        role = message.get("role")
        param = f"messages.{i}"
        if role not in ("user", "assistant"):
            raise InvalidRequest(f"{param}.role: must be user or assistant")
        texts: list[str] = []
        if role == "assistant":
            calls = []
            for j, block in enumerate(_blocks(message.get("content"), param)):
                kind = block.get("type")
                if kind == "text":
                    texts.append(block.get("text") or "")
                elif kind == "tool_use":
                    names[block.get("id", "")] = block.get("name", "")
                    calls.append({"function": {"name": block.get("name", ""), "arguments": block.get("input") or {}}})
                elif kind not in ("thinking", "redacted_thinking"):  # the model's old reasoning is not replayed
                    raise _unsupported(kind, f"{param}.content.{j}")
            entry: Message = {"role": "assistant", "content": "".join(texts)}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue

        # A user turn may mix text and tool results; keep their order.
        for j, block in enumerate(_blocks(message.get("content"), param)):
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text") or "")
            elif kind == "tool_result":
                if texts:
                    out.append({"role": "user", "content": "\n".join(texts)})
                    texts = []
                name = names.get(block.get("tool_use_id", ""))
                if name is None:
                    raise InvalidRequest(
                        f"{param}.content.{j}: tool_use_id {block.get('tool_use_id')!r} does not match "
                        "a tool_use block in an earlier assistant message"
                    )
                out.append(
                    {"role": "tool", "tool_name": name, "content": _tool_result_text(block, f"{param}.content.{j}")}
                )
            else:
                raise _unsupported(kind, f"{param}.content.{j}")
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def _tools(body: CountTokensRequest, messages: list[Message]) -> tuple[list[dict[str, Any]] | None, list[Message]]:
    # Tools without an input_schema are Anthropic's server tools (web search, code execution) or
    # schema-less built-ins: there is nothing to run them here, so they are left out.
    tools = [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]},
        }
        for t in body.tools or []
        if t.get("name") and isinstance(t.get("input_schema"), dict)
    ]
    choice = (body.tool_choice or {}).get("type", "auto")
    if not tools or choice == "auto":
        return tools or None, messages
    if choice == "none":
        return None, messages
    if choice == "any":
        return tools, with_system_hint(messages, tool_hint([t["function"]["name"] for t in tools], None))
    if choice == "tool":
        name = (body.tool_choice or {}).get("name")
        chosen = [t for t in tools if t["function"]["name"] == name]
        if not chosen:
            raise InvalidRequest(f"tool_choice: {name!r} is not in tools")
        return chosen, with_system_hint(messages, tool_hint([name], name))
    raise InvalidRequest("tool_choice.type must be auto, any, tool or none")


def _think(thinking: dict[str, Any] | None, spec: ModelSpec) -> bool | None:
    kind = (thinking or {}).get("type")
    if kind is None or kind == "adaptive":  # adaptive: the model decides, i.e. its own default
        return spec.think
    if kind == "enabled":
        return True
    if kind in ("disabled", "between_tools"):
        return False
    raise InvalidRequest("thinking.type must be enabled, disabled or adaptive")


def _format(output_config: dict[str, Any] | None) -> dict[str, Any] | None:
    fmt = (output_config or {}).get("format")
    if fmt is None:
        return None
    if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise InvalidRequest("output_config.format must be {type: json_schema, schema: {...}}")
    return fmt["schema"]


def _estimate_input(messages: list[Message], tools: list[dict[str, Any]] | None) -> int:
    """About four characters per token over what the model is sent, plus a few tokens per message
    for the chat template. Ollama has no tokenize endpoint, so this is an estimate."""
    chars = 0
    for message in messages:
        chars += len(message.get("content") or "")
        if message.get("tool_calls"):
            chars += len(json.dumps(message["tool_calls"], ensure_ascii=False))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    return estimate_tokens(chars) + 4 * len(messages)


def _resolve(svc: Services, principal: Principal, requested: str) -> tuple[ModelSpec, ChatModel, str]:
    """(spec, model, the name to echo). Unknown claude-* names get the default model."""
    try:
        spec, model = svc.model_for(requested, principal)
        return spec, model, spec.id
    except HTTPException as error:
        if error.status_code != 404 or not requested.startswith("claude-"):
            raise
    spec, model = svc.model_for(None, principal)
    return spec, model, requested


def build_call(body: MessagesRequest, spec: ModelSpec, model: ChatModel) -> ModelCall:
    messages = to_ollama_messages(body.system, body.messages)
    tools, messages = _tools(body, messages)
    check_context(messages, tools, spec.num_ctx, anthropic=True)
    options: dict[str, Any] = {"num_predict": body.max_tokens}
    for name in ("temperature", "top_p", "top_k"):
        if getattr(body, name) is not None:
            options[name] = getattr(body, name)
    # stop_sequences are not passed to the model: see StopScanner.
    return ModelCall(
        model=model,
        messages=messages,
        tools=tools,
        think=_think(body.thinking, spec),
        format=_format(body.output_config),
        options=options,
    )


# -- responses ----------------------------------------------------------------------------------


class _Blocks:
    """Model events -> content block operations, shared by the streamed and the plain reply:
    ("start", index, block), ("delta", index, delta), ("stop", index)."""

    def __init__(self, stop_sequences: list[str] | None, max_tokens: int, input_estimate: int):
        self.scanner = StopScanner(stop_sequences)
        self.max_tokens = max_tokens
        self.input_estimate = input_estimate
        self.index = -1
        self.open: str | None = None
        self.tool_uses = 0
        self.chars = 0
        self.usage: Usage | None = None
        self.finished = False

    def _start(self, block: dict[str, Any]) -> list[tuple]:
        ops = self._close()
        self.index += 1
        self.open = block["type"]
        return [*ops, ("start", self.index, block)]

    def _close(self) -> list[tuple]:
        ops: list[tuple] = []
        if self.open == "text" and (held := self.scanner.flush()):
            ops.append(("delta", self.index, {"type": "text_delta", "text": held}))
        if self.open:
            ops.append(("stop", self.index))
        self.open = None
        return ops

    def _text(self, text: str) -> list[tuple]:
        ops = [] if self.open == "text" else self._start({"type": "text", "text": ""})
        return [*ops, ("delta", self.index, {"type": "text_delta", "text": text})]

    def feed(self, event: ModelEvent) -> list[tuple]:
        self.chars += len(event.text)
        if event.kind == "thinking" and event.text:
            ops = [] if self.open == "thinking" else self._start({"type": "thinking", "thinking": "", "signature": ""})
            return [*ops, ("delta", self.index, {"type": "thinking_delta", "thinking": event.text})]
        if event.kind == "content" and event.text:
            text = self.scanner.feed(event.text)
            ops = self._text(text) if text else []
            if self.scanner.hit is not None:
                # Cut short: the model's own counts never arrive, so estimate them.
                self.finished = True
                self.usage = Usage(prompt_tokens=self.input_estimate, output_tokens=estimate_tokens(self.chars))
            return ops
        if event.kind == "tool_calls":
            ops = []
            for call in event.tool_calls:
                self.tool_uses += 1
                block = {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": call.name, "input": {}}
                ops += self._start(block)
                ops.append(
                    ("delta", self.index, {"type": "input_json_delta", "partial_json": json.dumps(call.arguments)})
                )
                ops += self._close()
            return ops
        if event.kind == "done":
            self.usage = event.usage or Usage()
        return []

    def end(self) -> list[tuple]:
        return self._close()

    def stop_reason(self) -> tuple[str, str | None]:
        if self.scanner.hit is not None:
            return "stop_sequence", self.scanner.hit
        if self.tool_uses:
            return "tool_use", None
        if self.usage and self.usage.output_tokens >= self.max_tokens:
            return "max_tokens", None
        return "end_turn", None

    def usage_body(self) -> dict[str, int]:
        usage = self.usage or Usage(prompt_tokens=self.input_estimate)
        return {"input_tokens": usage.prompt_tokens, "output_tokens": usage.output_tokens}


def _message(message_id: str, model: str, content: list[dict[str, Any]], usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
    }


class _EventEncoder:
    """The Messages streaming events: message_start, ping, content_block_start / _delta / _stop,
    message_delta, message_stop (and error)."""

    def __init__(self, message_id: str, model: str, blocks: _Blocks):
        self.message_id = message_id
        self.model = model
        self.blocks = blocks

    @property
    def usage(self) -> Usage | None:
        return self.blocks.usage

    @property
    def finished(self) -> bool:
        return self.blocks.finished

    @staticmethod
    def _event(payload: dict[str, Any]) -> bytes:
        data = json.dumps(payload, ensure_ascii=False)
        return f"event: {payload['type']}\ndata: {data}\n\n".encode()

    def _ops(self, ops: list[tuple]) -> list[bytes]:
        frames = []
        for op in ops:
            if op[0] == "start":
                frames.append(self._event({"type": "content_block_start", "index": op[1], "content_block": op[2]}))
            elif op[0] == "delta":
                frames.append(self._event({"type": "content_block_delta", "index": op[1], "delta": op[2]}))
            else:
                frames.append(self._event({"type": "content_block_stop", "index": op[1]}))
        return frames

    def opening(self) -> list[bytes]:
        usage = {"input_tokens": self.blocks.input_estimate, "output_tokens": 0}
        start = {"type": "message_start", "message": _message(self.message_id, self.model, [], usage)}
        return [self._event(start), self.heartbeat()]

    def feed(self, event: ModelEvent) -> list[bytes]:
        return self._ops(self.blocks.feed(event))

    def closing(self) -> list[bytes]:
        frames = self._ops(self.blocks.end())
        reason, sequence = self.blocks.stop_reason()
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": sequence},
            "usage": self.blocks.usage_body(),
        }
        return [*frames, self._event(delta), self._event({"type": "message_stop"})]

    def failure(self, error: Exception) -> list[bytes]:
        http = errors.model_error(error)
        _, body = errors.anthropic_error(http.status_code, str(http.detail))
        return [self._event(body)]

    def heartbeat(self) -> bytes:
        return self._event({"type": "ping"})


def _apply(ops: list[tuple], content: list[dict[str, Any]], partial: dict[int, str]) -> None:
    for op in ops:
        if op[0] == "start":
            content.append(dict(op[2]))
        elif op[0] == "delta":
            block, delta = content[op[1]], op[2]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] += delta["thinking"]
            else:
                partial[op[1]] = partial.get(op[1], "") + delta["partial_json"]
        elif op[1] in partial:
            content[op[1]]["input"] = json.loads(partial.pop(op[1]))


@router.post("/messages")
async def messages(
    body: MessagesRequest,
    request: Request,
    principal: Principal = Depends(require_scope("chat")),
    svc: Services = Depends(services),
) -> Response:
    spec, model, echoed = _resolve(svc, principal, body.model)
    call = build_call(body, spec, model)
    blocks = _Blocks(body.stop_sequences, body.max_tokens, _estimate_input(call.messages, call.tools))
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    meter = Meter(svc, principal, ENDPOINT, spec.id, call)
    headers = {"X-Kindling-Model": spec.id}

    if body.stream:
        encoder = _EventEncoder(message_id, echoed, blocks)
        return ModelStreamResponse(
            call, encoder, meter, on_error=lambda e: errors.response_for("anthropic", e), headers=headers
        )

    content: list[dict[str, Any]] = []
    partial: dict[int, str] = {}

    def feed(event: ModelEvent) -> bool:
        _apply(blocks.feed(event), content, partial)
        return not blocks.finished

    async def produce() -> Response:
        try:
            await collect(request, call, feed)
            _apply(blocks.end(), content, partial)
            meter.finish(blocks.usage)
        except ClientGone:
            return Response(status_code=499)  # nobody is listening; usage is recorded as "stopped"
        except Exception as error:
            meter.finish(None, error=True)
            return errors.response_for("anthropic", error)
        finally:
            await meter.record()

        reply = _message(message_id, echoed, content, blocks.usage_body())
        reply["stop_reason"], reply["stop_sequence"] = blocks.stop_reason()
        return JSONResponse(reply, headers=headers)

    # The call runs while the response is being sent, so a slow one can pad the connection open.
    return PaddedJSONResponse(produce, headers=headers)


@router.post("/messages/count_tokens")
async def count_tokens(
    body: CountTokensRequest,
    principal: Principal = Depends(require_scope("chat")),
    svc: Services = Depends(services),
) -> dict[str, int]:
    """An estimate: see _estimate_input."""
    _resolve(svc, principal, body.model)
    messages = to_ollama_messages(body.system, body.messages)
    tools, messages = _tools(body, messages)
    return {"input_tokens": _estimate_input(messages, tools)}


def install(app: FastAPI) -> None:
    errors.install(app)
