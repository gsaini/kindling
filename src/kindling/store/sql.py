"""The Postgres store, over SQLAlchemy's async engine and asyncpg."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload, undefer

from .models import ConversationRow, MediaRow, MessageRow
from .schemas import Conversation, ConversationSummary, MediaInfo, NewMessage, StoredMedia, StoredMessage


class SqlStore:
    def __init__(self, url: str | None = None, *, engine: AsyncEngine | None = None):
        if engine is None:
            if url is None:
                raise ValueError("pass a database URL or an engine")
            engine = create_async_engine(url, pool_pre_ping=True)
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def ping(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(select(1))

    async def create(self, owner: str, title: str = "New chat") -> ConversationSummary:
        async with self.sessions.begin() as session:
            row = ConversationRow(owner=owner, title=title)
            session.add(row)
            await session.flush()
            await session.refresh(row)
            return _summary(row, 0)

    async def list(self, owner: str, limit: int = 50) -> list[ConversationSummary]:
        counts = (
            select(MessageRow.conversation_id, func.count().label("n")).group_by(MessageRow.conversation_id).subquery()
        )
        query = (
            select(ConversationRow, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.conversation_id == ConversationRow.id)
            .where(ConversationRow.owner == owner)
            .order_by(ConversationRow.updated_at.desc())
            .limit(limit)
        )
        async with self.sessions() as session:
            return [_summary(row, n) for row, n in (await session.execute(query)).all()]

    async def get(self, owner: str, conversation_id: UUID) -> Conversation | None:
        query = (
            select(ConversationRow)
            .where(ConversationRow.id == conversation_id, ConversationRow.owner == owner)
            .options(selectinload(ConversationRow.messages).selectinload(MessageRow.media))
        )
        async with self.sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
            if row is None:
                return None
            messages = [_message(m) for m in row.messages]
            return Conversation(**_summary(row, len(messages)).model_dump(), messages=messages)

    async def rename(self, owner: str, conversation_id: UUID, title: str) -> bool:
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(ConversationRow)
                .where(ConversationRow.id == conversation_id, ConversationRow.owner == owner)
                .values(title=title, updated_at=func.now())
            )
            return result.rowcount > 0

    async def delete(self, owner: str, conversation_id: UUID) -> bool:
        async with self.sessions.begin() as session:
            result = await session.execute(
                delete(ConversationRow).where(ConversationRow.id == conversation_id, ConversationRow.owner == owner)
            )
            return result.rowcount > 0

    async def append(self, owner: str, conversation_id: UUID, messages: Sequence[NewMessage]) -> None:
        async with self.sessions.begin() as session:
            # Lock the conversation row so two appends can't take the same sequence numbers.
            locked = await session.execute(
                select(ConversationRow.id)
                .where(ConversationRow.id == conversation_id, ConversationRow.owner == owner)
                .with_for_update()
            )
            if locked.scalar_one_or_none() is None:
                raise KeyError(conversation_id)
            last = await session.scalar(
                select(func.coalesce(func.max(MessageRow.seq), -1)).where(MessageRow.conversation_id == conversation_id)
            )
            for offset, message in enumerate(messages, start=1):
                row = MessageRow(
                    conversation_id=conversation_id, seq=last + offset, **message.model_dump(exclude={"media"})
                )
                row.media = [
                    MediaRow(
                        position=i,
                        kind=m.kind,
                        mime=m.mime,
                        name=m.name,
                        size=len(m.data),
                        seconds=m.seconds,
                        data=m.data,
                    )
                    for i, m in enumerate(message.media or [])
                ]
                session.add(row)
            await session.execute(
                update(ConversationRow).where(ConversationRow.id == conversation_id).values(updated_at=func.now())
            )

    async def get_media(self, owner: str, conversation_id: UUID, media_id: UUID) -> StoredMedia | None:
        query = (
            select(MediaRow)
            .join(MessageRow, MediaRow.message_id == MessageRow.id)
            .join(ConversationRow, MessageRow.conversation_id == ConversationRow.id)
            .where(MediaRow.id == media_id, ConversationRow.id == conversation_id, ConversationRow.owner == owner)
            .options(undefer(MediaRow.data))
        )
        async with self.sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
            if row is None:
                return None
            return StoredMedia(
                id=row.id, kind=row.kind, mime=row.mime, name=row.name, seconds=row.seconds, data=row.data
            )  # type: ignore[arg-type]

    async def close(self) -> None:
        await self.engine.dispose()


def _summary(row: ConversationRow, count: int) -> ConversationSummary:
    return ConversationSummary(
        id=row.id,
        owner=row.owner,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        message_count=count,
    )


def _message(row: MessageRow) -> StoredMessage:
    return StoredMessage(
        id=row.id,
        seq=row.seq,
        created_at=row.created_at,
        role=row.role,  # type: ignore[arg-type]
        content=row.content,
        thinking=row.thinking,
        tool_calls=row.tool_calls,
        tool_name=row.tool_name,
        stats=row.stats,
        media=[
            MediaInfo(id=m.id, kind=m.kind, mime=m.mime, name=m.name, size=m.size, seconds=m.seconds)  # type: ignore[arg-type]
            for m in row.media
        ]
        or None,
    )
