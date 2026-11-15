"""API keys: issuing, storing (hashed), looking up, and the per-process state that guards them.

A key looks like `kdl_<43 url-safe characters>` (32 random bytes). It is shown once, when it is
created; only its SHA-256 is stored, so a database leak doesn't leak working keys. Each key belongs
to an `owner` (whose conversations it sees) and has a `name` for the use case it was issued for
("hearth", "claude-code", "evals"), plus scopes, a model allowlist and limits.

`service_for(store)` returns the one `KeyService` for a conversation store: the key store (Postgres
for a SqlStore, memory otherwise), the usage log, a short lookup cache, the rate limiter and the
daily token counters. kindling.auth, kindling.usage and the admin API all share it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
import secrets
import time
import uuid
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from typing import TYPE_CHECKING, Any, Literal, Protocol

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from .store.auth_models import ApiKeyRow
from .usage import MemoryUsageLog, Recorder, SqlUsageLog, UsageEvent, UsageLog

if TYPE_CHECKING:
    from .llm import ModelRegistry

log = logging.getLogger("kindling.keys")

KEY_PREFIX = "kdl_"
SCOPES = frozenset({"chat", "admin"})
PREFIX_LENGTH = 12  # "kdl_" + 8 characters: enough to tell keys apart, far too few to guess the rest
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@:-]{0,99}")
LOOKUP_CACHE_SECONDS = 10.0  # how long another process's revocation (the CLI) can take to bite
TOUCH_EVERY_SECONDS = 60.0  # last_used_at is written at most this often per key

State = Literal["active", "revoked", "expired"]


class InvalidKeySettings(ValueError):
    pass


def now_utc() -> datetime:
    return datetime.now(UTC)


def new_secret() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


@dataclass(frozen=True)
class ApiKey:
    id: uuid.UUID
    owner: str
    name: str
    prefix: str
    hash: str
    scopes: frozenset[str] = frozenset({"chat"})
    models: frozenset[str] | None = None  # None: every model
    rate_limit_per_minute: int | None = None  # None: the server's default
    daily_token_limit: int | None = None  # None: unlimited
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime = field(default_factory=now_utc)
    last_used_at: datetime | None = None
    note: str = ""

    def state(self, now: datetime | None = None) -> State:
        now = now or now_utc()
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at is not None and self.expires_at <= now:
            return "expired"
        return "active"

    def public(self) -> dict[str, Any]:
        """What the admin API and CLI show. Never the hash."""
        return {
            "id": str(self.id),
            "owner": self.owner,
            "name": self.name,
            "prefix": self.prefix,
            "scopes": sorted(self.scopes),
            "models": sorted(self.models) if self.models is not None else None,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "daily_token_limit": self.daily_token_limit,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "note": self.note,
            "state": self.state(),
        }


# -- validation -----------------------------------------------------------------------------


def check_label(value: str, what: str) -> str:
    value = (value or "").strip()
    if not LABEL.fullmatch(value):
        raise InvalidKeySettings(
            f"{what} must be 1-100 characters: letters, digits and . _ @ : - (starting with a letter or digit)"
        )
    return value


def check_scopes(scopes: Iterable[str]) -> frozenset[str]:
    chosen = frozenset(s.strip() for s in scopes if s.strip())
    if not chosen:
        raise InvalidKeySettings(f"give at least one scope: {', '.join(sorted(SCOPES))}")
    unknown = chosen - SCOPES
    if unknown:
        raise InvalidKeySettings(
            f"unknown scope(s) {', '.join(sorted(unknown))}; choose from {', '.join(sorted(SCOPES))}"
        )
    return chosen


def check_models(models: Iterable[str] | None, registry: ModelRegistry | None) -> frozenset[str] | None:
    """Model ids (aliases become ids, since that is what requests are checked against)."""
    if models is None:
        return None
    wanted = [m.strip() for m in models if m.strip()]
    if registry is None:
        return frozenset(wanted)
    ids, unknown = set(), []
    for name in wanted:
        try:
            ids.add(registry.spec(name).id)
        except LookupError:
            unknown.append(name)
    if unknown:
        available = ", ".join(s.id for s in registry.specs())
        raise InvalidKeySettings(f"unknown model(s) {', '.join(unknown)}. Available: {available}")
    return frozenset(ids)


def check_limit(value: int | None, what: str) -> int | None:
    if value is not None and value < 1:
        raise InvalidKeySettings(f"{what} must be at least 1 (or unset)")
    return value


_DURATION = re.compile(r"(\d+)\s*([mhdw])")
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_duration(text: str) -> timedelta:
    """'90d', '12h', '30m', '2w'."""
    match = _DURATION.fullmatch(text.strip().lower())
    if not match:
        raise InvalidKeySettings(f"{text!r} is not a duration like 30m, 12h, 90d or 2w")
    return timedelta(**{_UNITS[match[2]]: int(match[1])})


def parse_time(text: str, *, ago: bool) -> datetime:
    """A duration ('7d': that long ago, or from now if not `ago`) or an ISO date / datetime (UTC if no zone)."""
    try:
        delta = parse_duration(text)
        return now_utc() - delta if ago else now_utc() + delta
    except InvalidKeySettings:
        pass
    try:
        when = datetime.fromisoformat(text.strip())
    except ValueError:
        raise InvalidKeySettings(f"{text!r} is neither a duration (7d) nor a date (2026-12-31)") from None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


# -- storage --------------------------------------------------------------------------------


class KeyStore(Protocol):
    async def add(self, key: ApiKey) -> None: ...

    async def get(self, key_id: uuid.UUID) -> ApiKey | None: ...

    async def by_hash(self, key_hash: str) -> ApiKey | None: ...

    async def find(self, ref: str) -> list[ApiKey]: ...  # by id, or by (the start of) the prefix

    async def list(self, owner: str | None = None, include_revoked: bool = False) -> list[ApiKey]: ...

    async def save(self, key: ApiKey) -> None: ...  # overwrite the mutable fields

    async def rotate(self, old: ApiKey, new: ApiKey) -> None: ...  # save `old` (revoked) and add `new`, atomically

    async def touch(self, key_id: uuid.UUID, when: datetime) -> None: ...


def _as_uuid(ref: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(ref)
    except ValueError:
        return None


class MemoryKeyStore:
    def __init__(self) -> None:
        self._keys: dict[uuid.UUID, ApiKey] = {}

    async def add(self, key: ApiKey) -> None:
        if any(k.hash == key.hash for k in self._keys.values()):
            raise ValueError("duplicate key hash")
        self._keys[key.id] = key

    async def get(self, key_id: uuid.UUID) -> ApiKey | None:
        return self._keys.get(key_id)

    async def by_hash(self, key_hash: str) -> ApiKey | None:
        return next((k for k in self._keys.values() if k.hash == key_hash), None)

    async def find(self, ref: str) -> list[ApiKey]:
        key_id = _as_uuid(ref)
        if key_id is not None:
            return [self._keys[key_id]] if key_id in self._keys else []
        return [k for k in self._keys.values() if ref and k.prefix.startswith(ref)]

    async def list(self, owner: str | None = None, include_revoked: bool = False) -> list[ApiKey]:
        keys = [
            k
            for k in self._keys.values()
            if (owner is None or k.owner == owner) and (include_revoked or k.revoked_at is None)
        ]
        return sorted(keys, key=lambda k: (k.owner, k.name, k.created_at))

    async def save(self, key: ApiKey) -> None:
        if key.id in self._keys:
            self._keys[key.id] = key

    async def rotate(self, old: ApiKey, new: ApiKey) -> None:
        await self.add(new)
        await self.save(old)

    async def touch(self, key_id: uuid.UUID, when: datetime) -> None:
        if key_id in self._keys:
            self._keys[key_id] = dataclasses.replace(self._keys[key_id], last_used_at=when)


def _from_row(row: ApiKeyRow) -> ApiKey:
    return ApiKey(
        id=row.id,
        owner=row.owner,
        name=row.name,
        prefix=row.prefix,
        hash=row.hash,
        scopes=frozenset(row.scopes),
        models=frozenset(row.models) if row.models is not None else None,
        rate_limit_per_minute=row.rate_limit_per_minute,
        daily_token_limit=row.daily_token_limit,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        note=row.note,
    )


def _mutable(key: ApiKey) -> dict[str, Any]:
    return {
        "name": key.name,
        "scopes": sorted(key.scopes),
        "models": sorted(key.models) if key.models is not None else None,
        "rate_limit_per_minute": key.rate_limit_per_minute,
        "daily_token_limit": key.daily_token_limit,
        "expires_at": key.expires_at,
        "revoked_at": key.revoked_at,
        "note": key.note,
    }


def _new_row(key: ApiKey) -> ApiKeyRow:
    return ApiKeyRow(
        id=key.id, owner=key.owner, prefix=key.prefix, hash=key.hash, created_at=key.created_at, **_mutable(key)
    )


class SqlKeyStore:
    def __init__(self, engine: AsyncEngine):
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def add(self, key: ApiKey) -> None:
        async with self.sessions.begin() as session:
            session.add(_new_row(key))

    async def get(self, key_id: uuid.UUID) -> ApiKey | None:
        async with self.sessions() as session:
            row = await session.get(ApiKeyRow, key_id)
            return _from_row(row) if row else None

    async def by_hash(self, key_hash: str) -> ApiKey | None:
        async with self.sessions() as session:
            row = (await session.execute(select(ApiKeyRow).where(ApiKeyRow.hash == key_hash))).scalar_one_or_none()
            return _from_row(row) if row else None

    async def find(self, ref: str) -> list[ApiKey]:
        key_id = _as_uuid(ref)
        if key_id is not None:
            found = await self.get(key_id)
            return [found] if found else []
        if not ref:
            return []
        escaped = ref.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = select(ApiKeyRow).where(ApiKeyRow.prefix.like(f"{escaped}%", escape="\\"))
        async with self.sessions() as session:
            return [_from_row(r) for r in (await session.execute(query)).scalars()]

    async def list(self, owner: str | None = None, include_revoked: bool = False) -> list[ApiKey]:
        query = select(ApiKeyRow).order_by(ApiKeyRow.owner, ApiKeyRow.name, ApiKeyRow.created_at)
        if owner is not None:
            query = query.where(ApiKeyRow.owner == owner)
        if not include_revoked:
            query = query.where(ApiKeyRow.revoked_at.is_(None))
        async with self.sessions() as session:
            return [_from_row(r) for r in (await session.execute(query)).scalars()]

    async def save(self, key: ApiKey) -> None:
        async with self.sessions.begin() as session:
            await session.execute(update(ApiKeyRow).where(ApiKeyRow.id == key.id).values(**_mutable(key)))

    async def rotate(self, old: ApiKey, new: ApiKey) -> None:
        async with self.sessions.begin() as session:
            await session.execute(update(ApiKeyRow).where(ApiKeyRow.id == old.id).values(**_mutable(old)))
            session.add(_new_row(new))

    async def touch(self, key_id: uuid.UUID, when: datetime) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                update(ApiKeyRow)
                .where(ApiKeyRow.id == key_id, or_(ApiKeyRow.last_used_at.is_(None), ApiKeyRow.last_used_at < when))
                .values(last_used_at=when)
            )


# -- operations (shared by the admin API and the CLI) ---------------------------------------


async def create_key(
    keys: KeyStore,
    *,
    owner: str,
    name: str,
    scopes: Iterable[str] = ("chat",),
    models: frozenset[str] | None = None,
    rate_limit_per_minute: int | None = None,
    daily_token_limit: int | None = None,
    expires_at: datetime | None = None,
    note: str = "",
) -> tuple[ApiKey, str]:
    """A new key and its secret. The secret exists only in the return value: show it once."""
    if expires_at is not None and expires_at <= now_utc():
        raise InvalidKeySettings("the expiry is in the past")
    secret = new_secret()
    key = ApiKey(
        id=uuid.uuid4(),
        owner=check_label(owner, "owner"),
        name=check_label(name, "name"),
        prefix=secret[:PREFIX_LENGTH],
        hash=hash_key(secret),
        scopes=check_scopes(scopes),
        models=models,
        rate_limit_per_minute=check_limit(rate_limit_per_minute, "rate_limit_per_minute"),
        daily_token_limit=check_limit(daily_token_limit, "daily_token_limit"),
        expires_at=expires_at,
        note=(note or "")[:1000],
    )
    await keys.add(key)
    return key, secret


async def revoke_key(keys: KeyStore, key: ApiKey) -> ApiKey:
    if key.revoked_at is not None:
        return key
    revoked = dataclasses.replace(key, revoked_at=now_utc())
    await keys.save(revoked)
    return revoked


async def rotate_key(keys: KeyStore, key: ApiKey) -> tuple[ApiKey, ApiKey, str]:
    """(the revoked old key, the new key, its secret): same owner, name, scopes, models, limits, expiry, note."""
    if key.revoked_at is not None:
        raise InvalidKeySettings("this key is revoked; create a new one instead")
    secret = new_secret()
    new = dataclasses.replace(
        key,
        id=uuid.uuid4(),
        prefix=secret[:PREFIX_LENGTH],
        hash=hash_key(secret),
        created_at=now_utc(),
        last_used_at=None,
    )
    old = dataclasses.replace(key, revoked_at=now_utc())
    await keys.rotate(old, new)
    return old, new, secret


async def find_one(keys: KeyStore, ref: str) -> ApiKey:
    """A key by id or prefix, for the CLI. LookupError if there is none, or more than one."""
    found = await keys.find(ref.strip())
    if not found:
        raise LookupError(f"no key matches {ref!r}")
    if len(found) > 1:
        raise LookupError(f"{ref!r} matches {len(found)} keys; give more of the prefix, or the id")
    return found[0]


# -- per-process state ----------------------------------------------------------------------


class RateLimiter:
    """A token bucket per key: `per_minute` requests a minute, in bursts of up to `per_minute`.

    In-process: each server process counts on its own. Kindling runs as one process.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # id -> (tokens left, when counted)

    def hit(self, bucket: str, per_minute: int) -> float:
        """0 if the request may go ahead, else the seconds until it could."""
        now = self.clock()
        tokens, then = self._buckets.get(bucket, (float(per_minute), now))
        tokens = min(float(per_minute), tokens + (now - then) * per_minute / 60)
        if tokens >= 1:
            self._buckets[bucket] = (tokens - 1, now)
            return 0.0
        self._buckets[bucket] = (tokens, now)
        return (1 - tokens) * 60 / per_minute


def utc_day_start(day: date) -> datetime:
    return datetime.combine(day, clock_time.min, tzinfo=UTC)


class TokenCounters:
    """Tokens used today (UTC) per key: read from the usage log once per key and day, then kept
    up to date by the usage recorder, so quota checks don't query the database on each request."""

    def __init__(self, usage_log: UsageLog):
        self.usage_log = usage_log
        self._used: dict[tuple[str, date], int] = {}

    async def used_today(self, key_id: str, now: datetime | None = None) -> int:
        today = (now or now_utc()).astimezone(UTC).date()
        slot = (key_id, today)
        if slot not in self._used:
            used = await self.usage_log.tokens_since(key_id, utc_day_start(today))
            self._used = {k: v for k, v in self._used.items() if k[1] == today}  # forget earlier days
            self._used.setdefault(slot, used)
        return self._used[slot]

    def add(self, event: UsageEvent) -> None:
        if event.key_id is None:
            return
        slot = (event.key_id, event.created_at.astimezone(UTC).date())
        if slot in self._used:
            self._used[slot] += event.tokens


class KeyService:
    def __init__(self, keys: KeyStore, usage_log: UsageLog):
        self.keys = keys
        self.usage_log = usage_log
        self.limiter = RateLimiter()
        self.tokens = TokenCounters(usage_log)
        self.recorder = Recorder(usage_log, on_record=self.tokens.add)
        self._cache: dict[str, tuple[float, ApiKey]] = {}  # hash -> (when fetched, key)
        self._touched: dict[uuid.UUID, float] = {}

    async def lookup(self, key_hash: str) -> ApiKey | None:
        cached = self._cache.get(key_hash)
        if cached and time.monotonic() - cached[0] < LOOKUP_CACHE_SECONDS:
            return cached[1]
        key = await self.keys.by_hash(key_hash)
        if key is None:
            self._cache.pop(key_hash, None)  # misses aren't cached: random keys can't fill memory
            return None
        if len(self._cache) > 10_000:
            self._cache.clear()
        self._cache[key_hash] = (time.monotonic(), key)
        return key

    def forget(self, key: ApiKey) -> None:
        """Drop a changed key from the cache, so a revocation through the admin API bites at once."""
        self._cache.pop(key.hash, None)

    async def touch(self, key: ApiKey, now: datetime) -> None:
        """Record last_used_at, at most about once a minute per key. Failures are only logged."""
        last = self._touched.get(key.id)
        if last is None and key.last_used_at is not None:
            if (now - key.last_used_at).total_seconds() < TOUCH_EVERY_SECONDS:
                self._touched[key.id] = time.monotonic()
                return
        elif last is not None and time.monotonic() - last < TOUCH_EVERY_SECONDS:
            return
        self._touched[key.id] = time.monotonic()
        try:
            await self.keys.touch(key.id, now)
        except Exception as error:
            log.warning("couldn't update last_used_at for key %s (%s)", key.id, type(error).__name__)


def build_backends(store: object) -> tuple[KeyStore, UsageLog]:
    """Postgres when the conversation store is (it has an `engine`), memory otherwise."""
    engine = getattr(store, "engine", None)
    if isinstance(engine, AsyncEngine):
        return SqlKeyStore(engine), SqlUsageLog(engine)
    return MemoryKeyStore(), MemoryUsageLog()


_services: weakref.WeakKeyDictionary[object, KeyService] = weakref.WeakKeyDictionary()


def service_for(store: object) -> KeyService:
    """The one KeyService for this conversation store (auth, usage and the admin API share it)."""
    service = _services.get(store)
    if service is None:
        service = KeyService(*build_backends(store))
        _services[store] = service
    return service
