"""Saved conversations: list, read, rename, delete. Each caller sees only its owner's."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from ..store import Conversation, ConversationSummary
from .deps import Principal, Services, require_scope, services

router = APIRouter(prefix="/api/conversations", tags=["conversations"])
chat_scope = require_scope("chat")


class NewConversation(BaseModel):
    title: str = Field("New chat", max_length=200)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@router.get("")
async def list_conversations(
    limit: int = Query(50, ge=1, le=500),
    principal: Principal = Depends(chat_scope),
    svc: Services = Depends(services),
) -> list[ConversationSummary]:
    return await svc.store.list(principal.owner, limit)


@router.post("", status_code=201)
async def create_conversation(
    body: NewConversation, principal: Principal = Depends(chat_scope), svc: Services = Depends(services)
) -> ConversationSummary:
    return await svc.store.create(principal.owner, body.title)


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID, principal: Principal = Depends(chat_scope), svc: Services = Depends(services)
) -> Conversation:
    conversation = await svc.store.get(principal.owner, conversation_id)
    if conversation is None:
        raise HTTPException(404, "no such conversation")
    return conversation


@router.patch("/{conversation_id}", status_code=204)
async def rename_conversation(
    conversation_id: UUID,
    body: Rename,
    principal: Principal = Depends(chat_scope),
    svc: Services = Depends(services),
) -> Response:
    if not await svc.store.rename(principal.owner, conversation_id, body.title):
        raise HTTPException(404, "no such conversation")
    return Response(status_code=204)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID, principal: Principal = Depends(chat_scope), svc: Services = Depends(services)
) -> Response:
    if not await svc.store.delete(principal.owner, conversation_id):
        raise HTTPException(404, "no such conversation")
    return Response(status_code=204)


@router.get("/{conversation_id}/media/{media_id}")
async def get_media(
    conversation_id: UUID,
    media_id: UUID,
    principal: Principal = Depends(chat_scope),
    svc: Services = Depends(services),
) -> Response:
    """An image or audio clip of a saved message, as the model received it (audio as 16 kHz WAV)."""
    media = await svc.store.get_media(principal.owner, conversation_id, media_id)
    if media is None:
        raise HTTPException(404, "no such media")
    headers = {"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"}
    return Response(content=media.data, media_type=media.mime, headers=headers)
