"""Admin API: API keys and usage, under /api/admin. Every route needs the "admin" scope.

POST  /api/admin/keys                  issue a key (the secret is in this response only)
GET   /api/admin/keys                  ?owner=  ?include_revoked=true
GET   /api/admin/keys/{id}             one key, with today's usage
PATCH /api/admin/keys/{id}             name, note, scopes, models, limits, expiry
POST  /api/admin/keys/{id}/revoke
POST  /api/admin/keys/{id}/rotate      a new secret with the same settings; the old one stops working
GET   /api/admin/usage                 ?owner= ?key_id= ?since= ?until= ?group_by=key|model|day
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..keys import (
    ApiKey,
    InvalidKeySettings,
    KeyService,
    check_label,
    check_limit,
    check_models,
    check_scopes,
    create_key,
    now_utc,
    parse_time,
    revoke_key,
    rotate_key,
    service_for,
    utc_day_start,
)
from .deps import Principal, Services, require_scope, services

log = logging.getLogger("kindling.admin")

admin_scope = require_scope("admin")
router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(admin_scope)])

SECRET_WARNING = "Store this key now: it is shown only once and can't be recovered (rotate it if it's lost)."


class NewKey(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=lambda: ["chat"])
    models: list[str] | None = None  # None: every model
    rate_limit_per_minute: int | None = None
    daily_token_limit: int | None = None
    expires_at: datetime | None = None
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)
    note: str = Field(default="", max_length=1000)


class KeyChanges(BaseModel):
    """Only the fields sent are changed; send null to clear models, limits or expiry."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=1000)
    scopes: list[str] | None = None
    models: list[str] | None = None
    rate_limit_per_minute: int | None = None
    daily_token_limit: int | None = None
    expires_at: datetime | None = None


def _svc(svc: Services) -> KeyService:
    return service_for(svc.store)


def _invalid(error: InvalidKeySettings) -> HTTPException:
    return HTTPException(422, str(error))


def _aware(when: datetime | None) -> datetime | None:
    return when.replace(tzinfo=UTC) if when is not None and when.tzinfo is None else when


async def _key(service: KeyService, key_id: UUID) -> ApiKey:
    key = await service.keys.get(key_id)
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such key")
    return key


def _actor(principal: Principal) -> str:
    return f"{principal.owner}/{principal.name}"


@router.post("/keys", status_code=201)
async def create(
    body: NewKey, principal: Principal = Depends(admin_scope), svc: Services = Depends(services)
) -> dict[str, Any]:
    service = _svc(svc)
    expires_at = _aware(body.expires_at)
    if body.expires_in_days is not None:
        if expires_at is not None:
            raise HTTPException(422, "give expires_at or expires_in_days, not both")
        expires_at = now_utc() + timedelta(days=body.expires_in_days)
    try:
        key, secret = await create_key(
            service.keys,
            owner=body.owner,
            name=body.name,
            scopes=body.scopes,
            models=check_models(body.models, svc.models),
            rate_limit_per_minute=body.rate_limit_per_minute,
            daily_token_limit=body.daily_token_limit,
            expires_at=expires_at,
            note=body.note,
        )
    except InvalidKeySettings as error:
        raise _invalid(error) from None
    log.info("key %s created for %s/%s by %s", key.id, key.owner, key.name, _actor(principal))
    return {"key": secret, "warning": SECRET_WARNING, **key.public()}


@router.get("/keys")
async def list_keys(
    owner: str | None = None, include_revoked: bool = False, svc: Services = Depends(services)
) -> list[dict[str, Any]]:
    return [k.public() for k in await _svc(svc).keys.list(owner=owner, include_revoked=include_revoked)]


@router.get("/keys/{key_id}")
async def show(key_id: UUID, svc: Services = Depends(services)) -> dict[str, Any]:
    service = _svc(svc)
    key = await _key(service, key_id)
    today = await service.usage_log.summary(key_id=str(key.id), since=utc_day_start(now_utc().date()))
    return {**key.public(), "today": today[0] if today else None}


@router.patch("/keys/{key_id}")
async def change(
    key_id: UUID, body: KeyChanges, principal: Principal = Depends(admin_scope), svc: Services = Depends(services)
) -> dict[str, Any]:
    service = _svc(svc)
    key = await _key(service, key_id)
    if key.revoked_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "this key is revoked")
    sent = body.model_fields_set
    changes: dict[str, Any] = {}
    try:
        if "name" in sent and body.name is not None:
            changes["name"] = check_label(body.name, "name")
        if "note" in sent:
            changes["note"] = body.note or ""
        if "scopes" in sent and body.scopes is not None:
            changes["scopes"] = check_scopes(body.scopes)
        if "models" in sent:
            changes["models"] = check_models(body.models, svc.models)
        if "rate_limit_per_minute" in sent:
            changes["rate_limit_per_minute"] = check_limit(body.rate_limit_per_minute, "rate_limit_per_minute")
        if "daily_token_limit" in sent:
            changes["daily_token_limit"] = check_limit(body.daily_token_limit, "daily_token_limit")
        if "expires_at" in sent:
            changes["expires_at"] = _aware(body.expires_at)
    except InvalidKeySettings as error:
        raise _invalid(error) from None
    updated = dataclasses.replace(key, **changes)
    await service.keys.save(updated)
    service.forget(key)
    log.info("key %s changed (%s) by %s", key.id, ", ".join(sorted(changes)) or "nothing", _actor(principal))
    return updated.public()


@router.post("/keys/{key_id}/revoke")
async def revoke(
    key_id: UUID, principal: Principal = Depends(admin_scope), svc: Services = Depends(services)
) -> dict[str, Any]:
    service = _svc(svc)
    revoked = await revoke_key(service.keys, await _key(service, key_id))
    service.forget(revoked)
    log.info("key %s (%s/%s) revoked by %s", revoked.id, revoked.owner, revoked.name, _actor(principal))
    return revoked.public()


@router.post("/keys/{key_id}/rotate")
async def rotate(
    key_id: UUID, principal: Principal = Depends(admin_scope), svc: Services = Depends(services)
) -> dict[str, Any]:
    service = _svc(svc)
    try:
        old, new, secret = await rotate_key(service.keys, await _key(service, key_id))
    except InvalidKeySettings as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None
    service.forget(old)
    log.info("key %s rotated to %s (%s/%s) by %s", old.id, new.id, new.owner, new.name, _actor(principal))
    return {"key": secret, "warning": SECRET_WARNING, "replaces": str(old.id), **new.public()}


@router.get("/usage")
async def usage(
    owner: str | None = None,
    key_id: UUID | None = None,
    since: str | None = Query(None, description="a duration back from now (7d, 12h) or an ISO date; default 7d"),
    until: str | None = Query(None, description="an ISO date or datetime; default now"),
    group_by: Literal["key", "model", "day"] | None = None,
    svc: Services = Depends(services),
) -> dict[str, Any]:
    try:
        start = parse_time(since or "7d", ago=True)
        end = parse_time(until, ago=True) if until else None
    except InvalidKeySettings as error:
        raise _invalid(error) from None
    rows = await _svc(svc).usage_log.summary(
        owner=owner, key_id=str(key_id) if key_id else None, since=start, until=end, group_by=group_by
    )
    return {"since": start, "until": end, "group_by": group_by, "rows": rows}


def install(app: FastAPI) -> None:
    pass  # nothing app-wide: the routes above are all there is
