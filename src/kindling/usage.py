"""Usage metering: one event per model request, so each key's (use case's) consumption is visible.

    class UsageRecorder(Protocol):
        async def record(self, event: UsageEvent) -> None    # must never raise into the request

    def build_usage(settings: Settings, store: ConversationStore) -> UsageRecorder

Events go to the `usage_events` table (SqlStore) or a list in memory (MemoryStore). Each one is
written as it happens: a model call takes seconds, so one INSERT per call costs nothing worth
batching, and nothing is lost if the process dies. The recorder also feeds the per-key daily
token counters that kindling.auth checks quotas against (see kindling.keys).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from .store.auth_models import UsageEventRow

if TYPE_CHECKING:
    from .api.deps import Principal
    from .config import Settings
    from .store import ConversationStore

log = logging.getLogger("kindling.usage")

GroupBy = Literal["key", "model", "day"]


@dataclass
class UsageEvent:
    owner: str
    key_id: str | None
    key_name: str
    endpoint: str  # "/api/chat", "/v1/chat/completions", "/v1/messages"
    model: str  # the model id the client asked for (after alias resolution)
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    status: Literal["ok", "error", "stopped"] = "ok"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def for_principal(cls, principal: Principal, endpoint: str, model: str, **kwargs) -> UsageEvent:
        return cls(
            owner=principal.owner,
            key_id=principal.key_id,
            key_name=principal.name,
            endpoint=endpoint,
            model=model,
            **kwargs,
        )

    @property
    def tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.output_tokens or 0)


class UsageRecorder(Protocol):
    async def record(self, event: UsageEvent) -> None: ...


class NoopUsage:
    async def record(self, event: UsageEvent) -> None:
        pass


# -- where events are kept ----------------------------------------------------------------


class UsageLog(Protocol):
    async def add(self, event: UsageEvent) -> None: ...

    async def tokens_since(self, key_id: str, since: datetime) -> int: ...

    async def summary(
        self,
        *,
        owner: str | None = None,
        key_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        group_by: GroupBy | None = None,
    ) -> list[dict[str, Any]]: ...


def _totals(rows: list[UsageEvent]) -> dict[str, Any]:
    prompt = sum(e.prompt_tokens for e in rows)
    output = sum(e.output_tokens for e in rows)
    return {
        "requests": len(rows),
        "errors": sum(1 for e in rows if e.status == "error"),
        "prompt_tokens": prompt,
        "output_tokens": output,
        "tokens": prompt + output,
        "wall_seconds": round(sum(e.wall_seconds for e in rows), 2),
    }


class MemoryUsageLog:
    """Usage in a list: for tests and for running without a database. Lost on restart."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    async def add(self, event: UsageEvent) -> None:
        self.events.append(event)

    async def tokens_since(self, key_id: str, since: datetime) -> int:
        return sum(e.tokens for e in self.events if e.key_id == key_id and e.created_at >= since)

    async def summary(self, *, owner=None, key_id=None, since=None, until=None, group_by=None) -> list[dict[str, Any]]:
        chosen = [
            e
            for e in self.events
            if (owner is None or e.owner == owner)
            and (key_id is None or e.key_id == key_id)
            and (since is None or e.created_at >= since)
            and (until is None or e.created_at < until)
        ]
        groups: dict[tuple, list[UsageEvent]] = {}
        labels: dict[tuple, dict[str, Any]] = {}
        for event in chosen:
            if group_by == "key":
                group: tuple = (event.owner, event.key_id)
                labels[group] = {"owner": event.owner, "key_id": event.key_id, "key_name": event.key_name}
            elif group_by == "model":
                group = (event.model,)
                labels[group] = {"model": event.model}
            elif group_by == "day":
                day = event.created_at.astimezone(UTC).date().isoformat()
                group = (day,)
                labels[group] = {"day": day}
            else:
                group = ()
                labels[group] = {}
            groups.setdefault(group, []).append(event)
        if group_by is None and not groups:
            return [_totals([])]
        return [{**labels[g], **_totals(rows)} for g, rows in sorted(groups.items(), key=lambda kv: str(kv[0]))]


def _uuid(value: str | None) -> uuid.UUID | None:
    try:
        return uuid.UUID(value) if value else None
    except ValueError:
        return None


class SqlUsageLog:
    def __init__(self, engine: AsyncEngine):
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def add(self, event: UsageEvent) -> None:
        async with self.sessions.begin() as session:
            session.add(
                UsageEventRow(
                    owner=event.owner,
                    key_id=_uuid(event.key_id),
                    key_name=event.key_name[:100],
                    endpoint=event.endpoint[:100],
                    model=event.model[:200],
                    prompt_tokens=event.prompt_tokens or 0,
                    output_tokens=event.output_tokens or 0,
                    wall_seconds=event.wall_seconds or 0.0,
                    status=event.status,
                    created_at=event.created_at,
                )
            )

    async def tokens_since(self, key_id: str, since: datetime) -> int:
        query = select(func.coalesce(func.sum(UsageEventRow.prompt_tokens + UsageEventRow.output_tokens), 0)).where(
            UsageEventRow.key_id == _uuid(key_id), UsageEventRow.created_at >= since
        )
        async with self.sessions() as session:
            return int(await session.scalar(query) or 0)

    async def summary(self, *, owner=None, key_id=None, since=None, until=None, group_by=None) -> list[dict[str, Any]]:
        row = UsageEventRow
        if group_by == "key":
            keys = [row.owner, row.key_id]
            labels = [*keys, func.max(row.key_name).label("key_name")]
        elif group_by == "model":
            keys = labels = [row.model]
        elif group_by == "day":
            day = func.date(func.timezone("UTC", row.created_at)).label("day")
            keys = labels = [day]
        else:
            keys, labels = [], []
        metrics = [
            func.count().label("requests"),
            func.coalesce(func.sum(case((row.status == "error", 1), else_=0)), 0).label("errors"),
            func.coalesce(func.sum(row.prompt_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(row.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(row.wall_seconds), 0.0).label("wall_seconds"),
        ]
        query = select(*labels, *metrics)
        if owner is not None:
            query = query.where(row.owner == owner)
        if key_id is not None:
            query = query.where(row.key_id == _uuid(key_id))
        if since is not None:
            query = query.where(row.created_at >= since)
        if until is not None:
            query = query.where(row.created_at < until)
        if keys:
            query = query.group_by(*keys).order_by(*keys)
        async with self.sessions() as session:
            result = (await session.execute(query)).mappings().all()
        out = []
        for r in result:
            item = dict(r)
            if "key_id" in item:
                item["key_id"] = str(item["key_id"]) if item["key_id"] else None
            if "day" in item:
                item["day"] = item["day"].isoformat()
            for name in ("requests", "errors", "prompt_tokens", "output_tokens"):
                item[name] = int(item[name])
            item["tokens"] = item["prompt_tokens"] + item["output_tokens"]
            item["wall_seconds"] = round(float(item["wall_seconds"]), 2)
            out.append(item)
        return out


# -- the recorder the routes call ---------------------------------------------------------


class Recorder:
    """Saves each event and tells the quota counters. Never raises: metering must not break a chat."""

    def __init__(self, usage_log: UsageLog, on_record: Callable[[UsageEvent], None] | None = None):
        self.log = usage_log
        self.on_record = on_record

    async def record(self, event: UsageEvent) -> None:
        try:
            if self.on_record:
                self.on_record(event)
        except Exception:
            log.exception("usage: couldn't update the quota counters")
        try:
            await self.log.add(event)
        except Exception as error:
            log.error(
                "usage: couldn't save an event for %s/%s (%s: %s)",
                event.owner,
                event.key_name,
                type(error).__name__,
                error,
            )

    async def aclose(self) -> None:
        pass  # events are written as they come; nothing is buffered


def build_usage(settings: Settings, store: ConversationStore) -> UsageRecorder:
    from .keys import service_for

    return service_for(store).recorder
