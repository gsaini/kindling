"""OpenAI-compatible API: GET /v1/models, GET /v1/models/{id}, POST /v1/chat/completions.

Plain model access, like any OpenAI-compatible server (vLLM, llama.cpp): `tools` are the client's
own tools, passed to the model and returned as `tool_calls`; Kindling's server-side tools and system
prompt are not used here. Reasoning comes back as `reasoning_content` (the vLLM / DeepSeek convention).
See docs/api.md for the field mapping and the gaps.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from ..llm import ChatModel, Message, ModelEvent, ModelSpec, ToolCall, Usage
from .compat import errors
from .compat.errors import InvalidRequest
from .compat.streaming import (
    ClientGone,
    Meter,
    ModelCall,
    ModelStreamResponse,
    PaddedJSONResponse,
    check_context,
    collect,
    tool_hint,
    with_system_hint,
)
from .deps import Principal, Services, require_scope, services

router = APIRouter(prefix="/v1", tags=["openai"])

ENDPOINT = "/v1/chat/completions"
_CREATED = int(time.time())  # models have no creation date here; a stable one keeps clients happy

# reasoning_effort -> think. Nemotron's reasoning is on or off, so every level of effort means on.
_EFFORT = {"none": False, "minimal": False, "low": True, "medium": True, "high": True, "xhigh": True}


class ChatCompletionRequest(BaseModel):
    # Clients send many fields this server doesn't use (user, metadata, store, logprobs...): accept them.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    think: bool | None = None  # Kindling extra: wins over reasoning_effort
    n: int | None = None


# -- models -------------------------------------------------------------------------------------


def _visible(svc: Services, principal: Principal) -> list[ModelSpec]:
    return [s for s in svc.models.specs() if principal.models is None or s.id in principal.models]


def _card(spec: ModelSpec, anthropic: bool) -> dict[str, Any]:
    if anthropic:  # the Anthropic Models API shape, for Anthropic SDKs and Claude Code
        return {
            "type": "model",
            "id": spec.id,
            "display_name": spec.description or spec.id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_CREATED)),
            "max_input_tokens": spec.num_ctx,
        }
    return {
        "id": spec.id,
        "object": "model",
        "created": _CREATED,
        "owned_by": "kindling",
        "max_model_len": spec.num_ctx,
    }


@router.get("/models")
async def list_models(
    request: Request, principal: Principal = Depends(require_scope("chat")), svc: Services = Depends(services)
) -> dict[str, Any]:
    """The models this caller may use. Answers in Anthropic's shape when the request carries anthropic-version."""
    anthropic = "anthropic-version" in request.headers
    cards = [_card(s, anthropic) for s in _visible(svc, principal)]
    if anthropic:
        return {
            "data": cards,
            "has_more": False,
            "first_id": cards[0]["id"] if cards else None,
            "last_id": cards[-1]["id"] if cards else None,
        }
    return {"object": "list", "data": cards}


@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    principal: Principal = Depends(require_scope("chat")),
    svc: Services = Depends(services),
) -> dict[str, Any]:
    spec, _ = svc.model_for(model_id, principal)
    return _card(spec, "anthropic-version" in request.headers)


# -- request translation ------------------------------------------------------------------------


def _text(content: Any, param: str) -> str:
    """Message content as plain text: a string, or an array of text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise InvalidRequest("content must be a string or an array of content parts", param)
    texts = []
    for i, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("text", "refusal"):
            texts.append(str(part.get(kind) or ""))
        elif kind in ("image_url", "input_image", "input_audio", "file"):
            raise InvalidRequest(f"{kind} content is not supported yet: send text", f"{param}.{i}")
        else:
            raise InvalidRequest(f"unsupported content part type {kind!r}", f"{param}.{i}")
    return "\n".join(texts)


def _arguments(raw: Any, param: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        raise InvalidRequest("tool call arguments must be a JSON object encoded as a string", param) from None
    if not isinstance(value, dict):
        raise InvalidRequest("tool call arguments must be a JSON object", param)
    return value


def to_ollama_messages(messages: list[dict[str, Any]]) -> list[Message]:
    """OpenAI chat messages -> Ollama's format. Tool results are matched to their call's name by
    tool_call_id, from the assistant message that made the call."""
    out: list[Message] = []
    names: dict[str, str] = {}  # tool_call_id -> tool name
    for i, message in enumerate(messages):
        param = f"messages.{i}"
        role = message.get("role")
        if role in ("system", "developer"):
            out.append({"role": "system", "content": _text(message.get("content"), f"{param}.content")})
        elif role == "user":
            out.append({"role": "user", "content": _text(message.get("content"), f"{param}.content")})
        elif role == "assistant":
            entry: Message = {"role": "assistant", "content": _text(message.get("content"), f"{param}.content")}
            calls = []
            for j, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function") or {}
                name = function.get("name")
                if not name:
                    raise InvalidRequest("a tool call needs function.name", f"{param}.tool_calls.{j}")
                if call.get("id"):
                    names[call["id"]] = name
                arguments = _arguments(function.get("arguments"), f"{param}.tool_calls.{j}.function.arguments")
                calls.append({"function": {"name": name, "arguments": arguments}})
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            name = names.get(call_id or "") or message.get("name")
            if not name:
                raise InvalidRequest(
                    f"tool_call_id {call_id!r} does not match a tool call in an earlier assistant message",
                    f"{param}.tool_call_id",
                )
            out.append(
                {"role": "tool", "tool_name": name, "content": _text(message.get("content"), f"{param}.content")}
            )
        else:
            raise InvalidRequest(f"unsupported role {role!r}", f"{param}.role")
    return out


def _tools(body: ChatCompletionRequest, messages: list[Message]) -> tuple[list[dict[str, Any]] | None, list[Message]]:
    """The tools to offer, after tool_choice. "required" and a named function can't be enforced on
    Ollama, so they narrow the tool list and add an instruction to the system prompt (best effort)."""
    tools = []
    for i, tool in enumerate(body.tools or []):
        function = tool.get("function") if isinstance(tool, dict) else None
        if tool.get("type", "function") != "function" or not isinstance(function, dict) or not function.get("name"):
            raise InvalidRequest("only function tools are supported", f"tools.{i}")
        clean = {k: v for k, v in function.items() if k in ("name", "description", "parameters")}
        clean.setdefault("parameters", {"type": "object", "properties": {}})
        tools.append({"type": "function", "function": clean})
    choice = body.tool_choice
    if not tools or choice in (None, "auto"):
        return tools or None, messages
    if choice == "none":
        return None, messages
    if choice == "required":
        return tools, with_system_hint(messages, tool_hint([t["function"]["name"] for t in tools], None))
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        chosen = [t for t in tools if t["function"]["name"] == name]
        if not chosen:
            raise InvalidRequest(f"tool_choice names {name!r}, which is not in tools", "tool_choice")
        return chosen, with_system_hint(messages, tool_hint([name], name))
    raise InvalidRequest(
        "tool_choice must be none, auto, required or {type: function, function: {name}}", "tool_choice"
    )


def _think(body: ChatCompletionRequest, spec: ModelSpec) -> bool | None:
    if body.think is not None:
        return body.think
    if body.reasoning_effort is not None:
        if body.reasoning_effort not in _EFFORT:
            raise InvalidRequest(f"reasoning_effort must be one of {', '.join(_EFFORT)}", "reasoning_effort")
        return _EFFORT[body.reasoning_effort]
    return spec.think


def _format(response_format: dict[str, Any] | None) -> dict[str, Any] | str | None:
    kind = (response_format or {}).get("type", "text")
    if kind == "text":
        return None
    if kind == "json_object":
        return "json"
    if kind == "json_schema":
        schema = ((response_format or {}).get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise InvalidRequest("json_schema.schema must be a JSON schema object", "response_format")
        return schema
    raise InvalidRequest("response_format.type must be text, json_object or json_schema", "response_format")


def _options(body: ChatCompletionRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for name in ("temperature", "top_p", "seed", "presence_penalty", "frequency_penalty"):
        if getattr(body, name) is not None:
            options[name] = getattr(body, name)
    limit = body.max_completion_tokens or body.max_tokens
    if limit:
        options["num_predict"] = limit
    if body.stop:
        options["stop"] = [body.stop] if isinstance(body.stop, str) else list(body.stop)
    return options


def build_call(body: ChatCompletionRequest, spec: ModelSpec, model: ChatModel) -> ModelCall:
    if body.n not in (None, 1):
        raise InvalidRequest("only n=1 is supported", "n")
    messages = to_ollama_messages(body.messages)
    tools, messages = _tools(body, messages)
    check_context(messages, tools, spec.num_ctx)
    return ModelCall(
        model=model,
        messages=messages,
        tools=tools,
        think=_think(body, spec),
        format=_format(body.response_format),
        options=_options(body),
    )


# -- responses ----------------------------------------------------------------------------------


def _wire_calls(calls: list[ToolCall], start: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "index": start + i,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
        }
        for i, c in enumerate(calls)
    ]


def _usage(usage: Usage | None) -> dict[str, int]:
    usage = usage or Usage()
    total = usage.prompt_tokens + usage.output_tokens
    return {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.output_tokens, "total_tokens": total}


def _finish_reason(has_calls: bool, usage: Usage | None, limit: int | None) -> str:
    if has_calls:
        return "tool_calls"
    if limit and usage and usage.output_tokens >= limit:
        return "length"
    return "stop"


class _Reply:
    """Gathers a non-streaming reply."""

    def __init__(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.calls: list[ToolCall] = []
        self.usage: Usage | None = None

    def feed(self, event: ModelEvent) -> None:
        if event.kind == "thinking":
            self.reasoning += event.text
        elif event.kind == "content":
            self.content += event.text
        elif event.kind == "tool_calls":
            self.calls.extend(event.tool_calls)
        elif event.kind == "done":
            self.usage = event.usage or Usage()


class _ChunkEncoder:
    """chat.completion.chunk frames: `data: {...}`, ending with `data: [DONE]`."""

    def __init__(self, completion_id: str, model: str, *, include_usage: bool, limit: int | None):
        self.id = completion_id
        self.model = model
        self.created = int(time.time())
        self.include_usage = include_usage
        self.limit = limit
        self.calls = 0
        self.usage: Usage | None = None
        self.finished = False

    def _frame(self, payload: dict[str, Any]) -> bytes:
        return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"

    def _payload(self, choices: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": None,
            "choices": choices,
        }
        if self.include_usage:
            payload["usage"] = None
        return payload

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        return self._frame(self._payload([choice]))

    def opening(self) -> list[bytes]:
        return [self._chunk({"role": "assistant", "content": ""})]

    def feed(self, event: ModelEvent) -> list[bytes]:
        if event.kind == "thinking" and event.text:
            return [self._chunk({"reasoning_content": event.text})]
        if event.kind == "content" and event.text:
            return [self._chunk({"content": event.text})]
        if event.kind == "tool_calls" and event.tool_calls:
            wire = _wire_calls(event.tool_calls, self.calls)
            self.calls += len(wire)
            return [self._chunk({"tool_calls": wire})]
        if event.kind == "done":
            self.usage = event.usage or Usage()
        return []

    def closing(self) -> list[bytes]:
        frames = [self._chunk({}, _finish_reason(self.calls > 0, self.usage, self.limit))]
        if self.include_usage:  # one last chunk with no choices, as OpenAI sends it
            last = self._payload([])
            last["usage"] = _usage(self.usage)
            frames.append(self._frame(last))
        return [*frames, b"data: [DONE]\n\n"]

    def failure(self, error: Exception) -> list[bytes]:
        http = errors.model_error(error)
        return [self._frame(errors.openai_error(http.status_code, str(http.detail)))]

    def heartbeat(self) -> bytes:
        return b": keepalive\n\n"  # an SSE comment: clients skip it


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    principal: Principal = Depends(require_scope("chat")),
    svc: Services = Depends(services),
) -> Response:
    spec, model = svc.model_for(body.model, principal)
    call = build_call(body, spec, model)
    limit = call.options.get("num_predict")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    meter = Meter(svc, principal, ENDPOINT, spec.id, call)
    headers = {"X-Kindling-Model": spec.id}

    if body.stream:
        include_usage = bool((body.stream_options or {}).get("include_usage"))
        encoder = _ChunkEncoder(completion_id, spec.id, include_usage=include_usage, limit=limit)
        return ModelStreamResponse(
            call, encoder, meter, on_error=lambda e: errors.response_for("openai", e), headers=headers
        )

    async def produce() -> Response:
        reply = _Reply()
        try:
            await collect(request, call, reply.feed)
            meter.finish(reply.usage)
        except ClientGone:
            return Response(status_code=499)  # nobody is listening; usage is recorded as "stopped"
        except Exception as error:
            meter.finish(None, error=True)
            return errors.response_for("openai", error)
        finally:
            await meter.record()

        # Models often write a stray newline before a tool call. OpenAI sends null content with tool calls,
        # and a client that forwards any non-empty content (to WhatsApp, say) would send a blank message.
        content = reply.content if reply.content.strip() or not reply.calls else ""
        message: dict[str, Any] = {"role": "assistant", "content": content or (None if reply.calls else "")}
        if reply.reasoning:
            message["reasoning_content"] = reply.reasoning
        if reply.calls:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in _wire_calls(reply.calls)]
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": spec.id,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "logprobs": None,
                        "finish_reason": _finish_reason(bool(reply.calls), reply.usage, limit),
                    }
                ],
                "usage": _usage(reply.usage),
            },
            headers=headers,
        )

    # The call runs while the response is being sent, so a slow one can pad the connection open.
    return PaddedJSONResponse(produce, headers=headers)


def install(app: FastAPI) -> None:
    errors.install(app)
