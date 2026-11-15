"""Conversation data as the API and both stores see it."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class NewMessage(BaseModel):
    role: Role
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    stats: dict[str, Any] | None = None  # speed and token counts, on assistant replies


class StoredMessage(NewMessage):
    id: UUID
    seq: int
    created_at: datetime


class ConversationSummary(BaseModel):
    id: UUID
    owner: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class Conversation(ConversationSummary):
    messages: list[StoredMessage] = Field(default_factory=list)


def title_from(text: str, limit: int = 60) -> str:
    """A first-pass title: the first line of the first message, trimmed at a word."""
    line = " ".join(text.strip().split())
    if len(line) <= limit:
        return line or "New chat"
    return line[:limit].rsplit(" ", 1)[0] + "…"
