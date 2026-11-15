"""Who may call the API, and how much.

    class Authenticator(Protocol):
        async def authenticate(self, request: Request) -> Principal   # raise HTTPException 401 / 429

    def build_auth(settings: Settings, store: ConversationStore) -> Authenticator

A request carries a key as `Authorization: Bearer <key>` or `x-api-key: <key>`. In order:

1. KINDLING_AUTH_DISABLED: everyone is LOCAL (a private machine only).
2. The bootstrap key (KINDLING_API_KEY): admin + chat, for first setup; rate limited at the default.
3. A key from `kindling keys create` / the admin API (see kindling.keys): its owner, scopes and models.
4. Anything else: 401, with the same message whether the key is unknown, revoked or expired.

Then per key: requests per minute (429 + Retry-After) and, for requests that run the model, the
daily token quota (429 until midnight UTC). Both are counted in this process.
"""

from __future__ import annotations

import logging
import math
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from fastapi import HTTPException, Request, status

from .keys import KEY_PREFIX, KeyService, hash_key, now_utc, service_for

if TYPE_CHECKING:
    from .api.deps import Principal
    from .config import Settings
    from .store import ConversationStore

log = logging.getLogger("kindling.auth")

# Requests that don't run the model; everything else that isn't a GET counts against a daily quota.
QUOTA_FREE_PREFIXES = ("/api/admin", "/api/conversations")


class Authenticator(Protocol):
    async def authenticate(self, request: Request) -> Principal: ...


def presented_key(request: Request) -> str:
    """The key from `Authorization: Bearer <key>` (OpenAI style) or `x-api-key: <key>` (Anthropic style)."""
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _unauthorized(reason: str) -> HTTPException:
    log.debug("rejected a request: %s", reason)
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"}
    )


def _too_many(message: str, retry_after: float) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS, message, headers={"Retry-After": str(max(1, math.ceil(retry_after)))}
    )


def spends_tokens(request: Request) -> bool:
    return request.method not in ("GET", "HEAD", "OPTIONS") and not request.url.path.startswith(QUOTA_FREE_PREFIXES)


class KeyAuthenticator:
    def __init__(self, settings: Settings, service: KeyService):
        self.settings = settings
        self.service = service

    async def authenticate(self, request: Request) -> Principal:
        from .api.deps import LOCAL, Principal

        if self.settings.auth_disabled:
            return LOCAL
        given = presented_key(request)
        if not given:
            raise _unauthorized("no key")
        bootstrap = self.settings.api_key
        if bootstrap and secrets.compare_digest(given.encode(), bootstrap.encode()):
            self._limit("bootstrap", self.settings.rate_limit_per_minute)
            return Principal(owner="admin", name="bootstrap", scopes=frozenset({"admin", "chat"}))
        if not given.startswith(KEY_PREFIX) or len(given) > 200:
            raise _unauthorized("not a kindling key")
        try:
            key = await self.service.lookup(hash_key(given))
        except Exception as error:
            log.error("can't look up API keys (%s: %s)", type(error).__name__, error)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "can't check API keys right now") from None
        if key is None:
            raise _unauthorized("unknown key")
        now = now_utc()
        state = key.state(now)
        if state != "active":
            raise _unauthorized(f"key {key.id} ({key.owner}/{key.name}) is {state}")

        self._limit(str(key.id), key.rate_limit_per_minute or self.settings.rate_limit_per_minute)
        if key.daily_token_limit is not None and spends_tokens(request):
            used = await self.service.tokens.used_today(str(key.id), now)
            if used >= key.daily_token_limit:
                tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
                raise _too_many(
                    f"this key has used its daily quota of {key.daily_token_limit} tokens; it resets at 00:00 UTC",
                    (tomorrow - now).total_seconds(),
                )
        await self.service.touch(key, now)
        return Principal(owner=key.owner, name=key.name, key_id=str(key.id), scopes=key.scopes, models=key.models)

    def _limit(self, bucket: str, per_minute: int) -> None:
        wait = self.service.limiter.hit(bucket, per_minute)
        if wait:
            raise _too_many(f"rate limit: {per_minute} requests per minute for this key", wait)


def build_auth(settings: Settings, store: ConversationStore) -> Authenticator:
    return KeyAuthenticator(settings, service_for(store))
