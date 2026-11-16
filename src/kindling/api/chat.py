"""POST /api/chat: the conversation in, newline-delimited JSON events out.

Compatible with Hearth (thinking / content / done / error) and adds:

    {"type": "conversation", "id": "..."}                     first, when the chat is being saved
    {"type": "tool_call", "name": "...", "arguments": {...}}
    {"type": "tool_result", "name": "...", "output": "...", "ok": true}
    {"type": "ping"}                                          after 15 s of silence; ignore it
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..agent import StepLimitReached, run_agent
from ..llm import Message, Usage
from ..media import MediaLimits, find_ffmpeg, prepare, split_content, think_default
from ..store import NewMedia, NewMessage, title_from
from ..usage import UsageEvent
from .compat.keepalive import with_keepalive
from .compat.streaming import estimate_tokens
from .deps import Principal, Services, require_scope, services
from .errors import public_error

router = APIRouter(prefix="/api", tags=["chat"])
KEEPALIVE_SECONDS = 15.0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # Text, or for a user message a list of content parts with images and audio (kindling.media).
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None  # a model id or alias from models.toml; None: the default
    think: bool | None = None  # None: the model's configured default
    tools: bool = True  # let the model call the server's tools
    conversation_id: UUID | None = None  # append this exchange to a saved conversation
    save: bool = False  # start a new saved conversation (ignored if conversation_id is set)

    @field_validator("messages")
    @classmethod
    def ends_with_user(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if messages[-1].role != "user":
            raise ValueError("the last message must come from the user")
        return messages


def _title(message: Message) -> str:
    if message["content"].strip() or not message.get("media"):
        return title_from(message["content"])
    return "Image" if message["media"][0]["kind"] == "image" else "Voice note"


def _saved_media(message: Message) -> list[NewMedia] | None:
    """What the model received, to keep with a saved conversation."""
    items = [
        NewMedia(kind=m["kind"], mime=m["mime"], name=m.get("name"), seconds=m.get("seconds"), data=m["bytes"])
        for m in message.get("media") or []
    ]
    return items or None


def _line(event: dict[str, Any]) -> bytes:
    return json.dumps(event, ensure_ascii=False).encode() + b"\n"


@router.post("/chat")
async def chat(
    body: ChatRequest,
    principal: Principal = Depends(require_scope("chat")),
    svc: Services = Depends(services),
) -> StreamingResponse:
    spec, model = svc.model_for(body.model, principal)
    owner = principal.owner
    limits = MediaLimits.from_settings(svc.settings)
    history: list[Message] = []
    for i, message in enumerate(body.messages):
        text, media = split_content(
            message.content, f"messages.{i}.content", media_allowed=message.role == "user", limits=limits
        )
        history.append({"role": message.role, "content": text, **({"media": media} if media else {})})
    await prepare(history, spec, limits, ffmpeg=svc.settings.ffmpeg or find_ffmpeg())
    newest = history[-1]

    conversation_id = body.conversation_id
    if conversation_id is not None and await svc.store.get(owner, conversation_id) is None:
        raise HTTPException(404, "no such conversation")
    if conversation_id is None and body.save:
        conversation_id = (await svc.store.create(owner, _title(newest))).id

    if svc.settings.system_prompt and history[0]["role"] != "system":
        history.insert(0, {"role": "system", "content": svc.settings.system_prompt})
    tools = svc.tools if body.tools and svc.settings.tools_enabled and spec.tools else None
    think = think_default(body.think, spec.think, history)

    async def stream() -> AsyncIterator[bytes]:
        started = time.perf_counter()
        saved: list[NewMessage] = [NewMessage(role="user", content=newest["content"], media=_saved_media(newest))]
        partial = {"content": "", "thinking": ""}
        usage = UsageEvent.for_principal(principal, "/api/chat", spec.id, status="stopped")
        error: str | None = None
        finished = False
        if conversation_id:
            yield _line({"type": "conversation", "id": str(conversation_id)})
        try:
            events = run_agent(model, history, tools=tools, think=think, max_steps=svc.settings.max_tool_steps)
            # Pings keep proxies (Cloudflare cuts after ~125 s of silence) from dropping a request
            # while the model loads, waits in the queue or reads a long prompt.
            async for event in with_keepalive(events, KEEPALIVE_SECONDS, lambda: None):
                if event is None:
                    yield _line({"type": "ping"})
                elif event.type in ("thinking", "content"):
                    partial[event.type] += event.text
                    yield _line({"type": event.type, "text": event.text})
                elif event.type == "tool_call":
                    yield _line({"type": "tool_call", "name": event.name, "arguments": event.arguments})
                elif event.type == "tool_result":
                    yield _line({"type": "tool_result", "name": event.name, "output": event.text, "ok": event.ok})
                elif event.type == "done":
                    counts = event.usage or Usage()
                    stats = {
                        "model": spec.id,
                        "prompt_tokens": counts.prompt_tokens,
                        "output_tokens": counts.output_tokens,
                        "prefill_per_second": round(counts.prefill_per_second, 1),
                        "tokens_per_second": round(counts.tokens_per_second, 1),
                        "wall_seconds": round(event.wall_seconds, 2),
                        "num_ctx": model.num_ctx,
                        "steps": event.steps,
                    }
                    saved += [NewMessage(**m) for m in event.new_messages]
                    saved[-1].stats = stats
                    usage.prompt_tokens, usage.output_tokens = counts.prompt_tokens, counts.output_tokens
                    usage.status = "ok"
                    finished = True
                    yield _line({"type": "done", "stats": stats})
        except Exception as exc:  # Ollama went away, the step limit, a bad model reply...
            error = public_error(exc)
            usage.status = "error"
            if isinstance(exc, StepLimitReached):
                usage.prompt_tokens, usage.output_tokens = exc.usage.prompt_tokens, exc.usage.output_tokens
            yield _line({"type": "error", "message": error})
        finally:
            # Runs on success, on error, and when the client disconnects (Stop): keep what we have.
            usage.wall_seconds = time.perf_counter() - started
            if not finished and not (usage.prompt_tokens or usage.output_tokens):
                # A stopped or failed request still used the model: charge an estimate, or a client
                # could dodge its daily quota by disconnecting before the final count arrives.
                usage.prompt_tokens = estimate_tokens(sum(len(str(m.get("content", ""))) for m in history))
                usage.output_tokens = estimate_tokens(len(partial["content"]) + len(partial["thinking"]))
            with anyio.CancelScope(shield=True):
                await svc.usage.record(usage)
                if conversation_id:
                    if not finished:
                        note = {"error": error} if error else {"stopped": True}
                        saved.append(NewMessage(role="assistant", **partial, stats=note))
                    await svc.store.append(owner, conversation_id, saved)

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})
