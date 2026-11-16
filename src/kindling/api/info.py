from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from .deps import Principal, Services, require_scope, services

router = APIRouter(prefix="/api", tags=["info"])
log = logging.getLogger("kindling.health")


@router.get("/info")
async def info(
    principal: Principal = Depends(require_scope("chat")), svc: Services = Depends(services)
) -> dict[str, Any]:
    """What Hearth shows in its header, plus what this server offers the caller."""
    spec = svc.models.spec()
    visible = [s for s in svc.models.specs() if principal.models is None or s.id in principal.models]
    return {
        "model": spec.id,
        "num_ctx": spec.num_ctx,
        "system": svc.settings.system_prompt,
        "inputs": spec.inputs,  # what the default model reads: text, and images or audio where it can
        "models": [
            {
                "id": s.id,
                "name": s.name,
                "provider": s.provider,
                "num_ctx": s.num_ctx,
                "description": s.description,
                "inputs": s.inputs,
            }
            for s in visible
        ],
        "tools": svc.tools.names if svc.settings.tools_enabled else [],
        "caller": {"owner": principal.owner, "key": principal.name, "scopes": sorted(principal.scopes)},
    }


@router.get("/health")
async def health(svc: Services = Depends(services)) -> dict[str, Any]:
    """Readiness for load balancers and uptime checks. No key needed, so it says only ok or
    unavailable; the reason goes to the server log (`kindling check` shows it too)."""
    checks: dict[str, str] = {}
    try:
        await svc.models.get().check()
        checks["model"] = "ok"
    except Exception as error:
        log.warning("health: model not ready: %s", error)
        checks["model"] = "unavailable"
    ping = getattr(svc.store, "ping", None)
    if ping:
        try:
            await ping()
            checks["database"] = "ok"
        except Exception as error:
            log.warning("health: database unreachable: %s: %s", type(error).__name__, error)
            checks["database"] = "unavailable"
    return {"ok": all(v == "ok" for v in checks.values()), **checks}
