"""Shared state for the routes, and who is calling.

Routes declare what they need with `Depends(require_scope("chat"))`; that authenticates the caller
(via `Services.auth`, see kindling.auth) and returns a `Principal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status

from ..config import Settings
from ..llm import ChatModel, ModelNotAllowed, ModelRegistry, ModelSpec, UnknownModel
from ..store import ConversationStore
from ..tools import ToolRegistry


@dataclass(frozen=True)
class Principal:
    """The caller. Conversations belong to an `owner`, so rotating a key keeps its owner's history."""

    owner: str  # the account / team / app the key belongs to
    name: str  # the key's label: the use case it was issued for ("hearth", "claude-code", "evals")
    key_id: str | None = None  # None for the bootstrap key and when auth is disabled
    scopes: frozenset[str] = frozenset({"chat"})
    models: frozenset[str] | None = None  # model allowlist; None: every model

    def can(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


LOCAL = Principal(owner="local", name="local (auth disabled)", scopes=frozenset({"chat", "admin"}))


@dataclass
class Services:
    settings: Settings
    models: ModelRegistry
    store: ConversationStore
    tools: ToolRegistry
    auth: Authenticator
    usage: UsageRecorder
    extras: dict[str, object] = field(default_factory=dict)  # room for workstreams to hang shared objects

    def model_for(self, model_id: str | None, principal: Principal) -> tuple[ModelSpec, ChatModel]:
        """Resolve a requested model for this caller, as an HTTP error if it can't have it."""
        try:
            return self.models.resolve(model_id, principal.models)
        except UnknownModel:
            visible = [s.id for s in self.models.specs() if principal.models is None or s.id in principal.models]
            message = f"no model named {model_id!r}. Available: {', '.join(visible) or 'none'}"
            raise HTTPException(status.HTTP_404_NOT_FOUND, message) from None
        except ModelNotAllowed as error:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from None


def services(request: Request) -> Services:
    return request.app.state.services


async def current_principal(request: Request, svc: Services = Depends(services)) -> Principal:
    principal = await svc.auth.authenticate(request)  # raises 401 / 429 itself
    request.state.principal = principal
    return principal


def require_scope(scope: str):
    async def check(principal: Principal = Depends(current_principal)) -> Principal:
        if not principal.can(scope):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"this key lacks the {scope!r} scope")
        return principal

    return check


# Imported at the end: these modules import Principal from here.
from ..auth import Authenticator  # noqa: E402
from ..usage import UsageRecorder  # noqa: E402
