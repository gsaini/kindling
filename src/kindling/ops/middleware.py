"""ASGI middleware: request ids, the access log, security headers, metrics and the body size limit.

Plain ASGI rather than BaseHTTPMiddleware, so streaming replies pass straight through and the request
id contextvar is visible to everything the request runs, including the body of a streaming response.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import metrics
from .logs import request_id_var

access_log = logging.getLogger("kindling.access")

MAX_REQUEST_ID = 128
_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]")
_QUIET_PATHS = {"/livez", "/metrics"}  # probes and scrapes: access-logged at DEBUG


def request_id_from(headers: Headers) -> str:
    """The caller's X-Request-ID, cleaned so it can't forge log lines; a fresh one if there's none."""
    given = _UNSAFE.sub("", headers.get("x-request-id", ""))[:MAX_REQUEST_ID]
    return given or uuid.uuid4().hex


def route_template(scope: Scope) -> str:
    """`/v1/conversations/{id}` rather than the actual path, so metric labels stay few."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, metrics_enabled: bool = True):
        self.app = app
        self.metrics_enabled = metrics_enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_from(Headers(scope=scope))
        token = request_id_var.set(request_id)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        path: str = scope.get("path", "")
        no_store = path in ("/api", "/v1") or path.startswith(("/api/", "/v1/"))
        status = 500  # what the client gets if the app raises before responding
        started = time.perf_counter()

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                if no_store:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            seconds = time.perf_counter() - started
            if self.metrics_enabled:
                metrics.observe_request(scope["method"], route_template(scope), status, seconds)
            self._log(scope, state, path, status, seconds)
            request_id_var.reset(token)

    @staticmethod
    def _log(scope: Scope, state: dict, path: str, status: int, seconds: float) -> None:
        level = logging.DEBUG if path in _QUIET_PATHS else logging.INFO
        if not access_log.isEnabledFor(level):
            return
        client = scope.get("client")
        principal = state.get("principal")
        duration_ms = round(seconds * 1000, 1)
        # Only these fields. Never headers (Authorization, x-api-key), query strings or bodies.
        access_log.log(
            level,
            "%s %s %s %sms",
            scope["method"],
            path,
            status,
            duration_ms,
            extra={
                "method": scope["method"],
                "path": path,
                "status": status,
                "duration_ms": duration_ms,
                "client_ip": state.get("client_ip") or (client[0] if client else None),
                "key_owner": getattr(principal, "owner", None),
                "key_name": getattr(principal, "name", None),
            },
        )


class RequestTooLarge(HTTPException):
    def __init__(self, limit: int):
        super().__init__(413, f"request body is larger than the {limit} byte limit")


class BodyLimitMiddleware:
    """413 for bodies over `max_bytes`: at once from Content-Length, or while reading a chunked body."""

    def __init__(self, app: ASGIApp, *, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = Headers(scope=scope).get("content-length")
        if length is not None:
            try:
                too_large = int(length) > self.max_bytes
            except ValueError:
                await JSONResponse({"detail": "invalid Content-Length"}, status_code=400)(scope, receive, send)
                return
            if too_large:
                await self._reject(scope, receive, send)
                return

        received = 0
        started = False

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body parsing, so routes answer 413.
                    raise RequestTooLarge(self.max_bytes)
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except RequestTooLarge:
            if started:
                raise
            await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        detail = RequestTooLarge(self.max_bytes).detail
        await JSONResponse({"detail": detail}, status_code=413, headers={"Connection": "close"})(scope, receive, send)
