"""An in-memory store: for tests, and for running without a database. Lost on restart."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from .schemas import Conversation, ConversationSummary, MediaInfo, NewMessage, StoredMedia, StoredMessage


class MemoryStore:
    def __init__(self) -> None:
        self._conversations: dict[UUID, Conversation] = {}
        self._media: dict[UUID, tuple[UUID, StoredMedia]] = {}  # media id -> (conversation id, the media)

    def _owned(self, owner: str, conversation_id: UUID) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        return conversation if conversation and conversation.owner == owner else None

    async def create(self, owner: str, title: str = "New chat") -> ConversationSummary:
        now = datetime.now(UTC)
        conversation = Conversation(id=uuid4(), owner=owner, title=title, created_at=now, updated_at=now)
        self._conversations[conversation.id] = conversation
        return ConversationSummary(**conversation.model_dump(exclude={"messages"}))

    async def list(self, owner: str, limit: int = 50) -> list[ConversationSummary]:
        mine = [c for c in self._conversations.values() if c.owner == owner]
        ordered = sorted(mine, key=lambda c: c.updated_at, reverse=True)[:limit]
        return [ConversationSummary(**c.model_dump(exclude={"messages"})) for c in ordered]

    async def get(self, owner: str, conversation_id: UUID) -> Conversation | None:
        conversation = self._owned(owner, conversation_id)
        return conversation.model_copy(deep=True) if conversation else None

    async def rename(self, owner: str, conversation_id: UUID, title: str) -> bool:
        conversation = self._owned(owner, conversation_id)
        if conversation is None:
            return False
        conversation.title = title
        conversation.updated_at = datetime.now(UTC)
        return True

    async def delete(self, owner: str, conversation_id: UUID) -> bool:
        if self._owned(owner, conversation_id) is None:
            return False
        del self._conversations[conversation_id]
        self._media = {k: v for k, v in self._media.items() if v[0] != conversation_id}
        return True

    async def append(self, owner: str, conversation_id: UUID, messages: Sequence[NewMessage]) -> None:
        conversation = self._owned(owner, conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        now = datetime.now(UTC)
        for message in messages:
            seq = len(conversation.messages)
            infos = []
            for item in message.media or []:
                media = StoredMedia(**item.model_dump(), id=uuid4())
                self._media[media.id] = (conversation_id, media)
                infos.append(_info(media))
            fields = message.model_dump(exclude={"media"})
            stored = StoredMessage(**fields, id=uuid4(), seq=seq, created_at=now, media=infos or None)
            conversation.messages.append(stored)
        conversation.message_count = len(conversation.messages)
        conversation.updated_at = now

    async def get_media(self, owner: str, conversation_id: UUID, media_id: UUID) -> StoredMedia | None:
        if self._owned(owner, conversation_id) is None:
            return None
        found = self._media.get(media_id)
        return found[1] if found and found[0] == conversation_id else None

    async def close(self) -> None:
        pass


def _info(media: StoredMedia) -> MediaInfo:
    return MediaInfo(
        id=media.id, kind=media.kind, mime=media.mime, name=media.name, size=len(media.data), seconds=media.seconds
    )
