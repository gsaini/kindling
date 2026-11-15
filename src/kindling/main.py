"""The app factory: `create_app()` builds the FastAPI app and, on startup, its services."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from . import __version__, edge, ops
from .api import admin, anthropic_compat, chat, conversations, info, openai_compat
from .api.deps import Services
from .auth import build_auth
from .config import Settings, get_settings
from .llm import ChatModel, ModelRegistry
from .store import ConversationStore, MemoryStore, SqlStore
from .tools import ToolRegistry, register_builtins
from .usage import build_usage

log = logging.getLogger("kindling")


async def build_store(settings: Settings) -> ConversationStore:
    if settings.store == "memory":
        return MemoryStore()
    store = SqlStore(settings.database_url)
    try:
        await store.ping()
    except Exception as error:
        await store.close()
        raise RuntimeError(
            f"Can't reach Postgres at {settings.database_url.split('@')[-1]} ({type(error).__name__}: {error}). "
            "Start it with `docker compose up -d db`, then `kindling db upgrade`. "
            "Or run without a database: KINDLING_STORE=memory."
        ) from error
    return store


def build_tools(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    if settings.tools_enabled:
        settings.workspace.mkdir(parents=True, exist_ok=True)
        register_builtins(registry, settings.workspace)
    return registry


def create_app(
    settings: Settings | None = None,
    *,
    model: ChatModel | None = None,
    models: ModelRegistry | None = None,
    store: ConversationStore | None = None,
    tools: ToolRegistry | None = None,
    **extras: Any,
) -> FastAPI:
    """Pass `model`/`models`/`store`/`tools` to replace the real ones (tests, scripts)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        wrap = ops.model_wrapper(settings)
        if models is not None:
            registry = models
        elif model is not None:
            registry = ModelRegistry.single(model, wrap=wrap)
        else:
            registry = ModelRegistry.from_settings(settings, wrap=wrap)
        the_store = store if store is not None else await build_store(settings)
        app.state.services = Services(
            settings=settings,
            models=registry,
            store=the_store,
            tools=tools if tools is not None else build_tools(settings),
            auth=build_auth(settings, the_store),
            usage=build_usage(settings, the_store),
            extras=dict(extras),
        )
        if settings.api_key and len(settings.api_key) < 32:
            log.warning(
                "KINDLING_API_KEY is shorter than 32 characters: use a long random value, or unset it after setup"
            )
        if settings.auth_disabled:
            log.warning("auth is disabled: anyone who can reach %s:%s can use the model", settings.host, settings.port)
        try:
            yield
        finally:
            closer = getattr(app.state.services.usage, "aclose", None)
            if closer:
                await closer()
            await the_store.close()

    app = FastAPI(
        title="Kindling",
        version=__version__,
        summary="A local AI assistant backend: Nemotron (or any Ollama / OpenAI-compatible model) with tools, "
        "saved conversations, API keys and OpenAI / Anthropic compatible endpoints.",
        lifespan=lifespan,
        docs_url="/docs" if settings.api_docs or settings.auth_disabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.api_docs or settings.auth_disabled else None,
    )
    ops.install(app, settings)
    edge.install(app, settings)
    for module in (info, chat, conversations, openai_compat, anthropic_compat, admin):
        app.include_router(module.router)
        install = getattr(module, "install", None)
        if install:
            install(app)
    return app
