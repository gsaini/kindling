"""Running Kindling as a service: logging, request ids, CORS, limits, the concurrency gate, metrics.

    install(app, settings)     middleware, the Overloaded handler, GET /livez and GET /metrics
    model_wrapper(settings)    wraps every model so all streams share one concurrency gate

Middleware order, outermost first: request context (id, access log, security headers, metrics),
CORS, body size limit, then the app. So a 413 or a CORS preflight still gets a request id and a log line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware

from .gate import GatedModel, GenerationGate, Overloaded
from .logs import configure_logging, current_request_id
from .middleware import BodyLimitMiddleware, RequestContextMiddleware

if TYPE_CHECKING:
    from ..config import Settings
    from ..llm import ChatModel

__all__ = [
    "GatedModel",
    "GenerationGate",
    "Overloaded",
    "current_request_id",
    "install",
    "model_wrapper",
]

CORS_HEADERS = ["authorization", "x-api-key", "content-type", "anthropic-version", "anthropic-beta", "x-request-id"]


def install(app: FastAPI, settings: Settings) -> None:
    configure_logging(settings.log_format, settings.log_level)

    # add_middleware puts each new one outside the last: innermost first.
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=CORS_HEADERS,
            expose_headers=["X-Request-ID", "Retry-After"],
            allow_credentials=False,  # keys travel in headers, never cookies
            max_age=600,
        )
    app.add_middleware(RequestContextMiddleware, metrics_enabled=settings.metrics_enabled)

    async def overloaded(request: Request, error: Exception) -> Response:
        # A fallback for routes that let Overloaded escape before streaming; the standard APIs answer it themselves.
        retry_after = getattr(error, "retry_after", 5)
        return JSONResponse({"detail": str(error)}, status_code=503, headers={"Retry-After": str(retry_after)})

    app.add_exception_handler(Overloaded, overloaded)

    async def livez() -> dict[str, bool]:
        """Is the process up? No dependencies, so a container health check never flaps with Ollama or Postgres."""
        return {"ok": True}

    app.add_api_route("/livez", livez, methods=["GET"], include_in_schema=False)

    if settings.metrics_enabled:
        from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

        from ..api.deps import require_scope  # imported here: kindling.api imports a lot

        async def metrics_endpoint() -> Response:
            return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

        # Open when auth is off (nobody else can reach the server); otherwise an admin key, as for /api/admin.
        dependencies = [] if settings.auth_disabled else [Depends(require_scope("admin"))]
        app.add_api_route(
            "/metrics", metrics_endpoint, methods=["GET"], dependencies=dependencies, include_in_schema=False
        )


def model_wrapper(settings: Settings) -> Callable[[ChatModel], ChatModel] | None:
    """One gate per app, shared by every model, so the limit is on the machine, not per model."""
    gate = GenerationGate(
        settings.max_concurrent_generations, settings.max_queued_generations, settings.queue_timeout_seconds
    )

    def wrap(model: ChatModel) -> ChatModel:
        return GatedModel(model, gate)

    return wrap
