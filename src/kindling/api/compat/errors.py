"""Errors in each API's own shape, so SDKs raise their usual exceptions.

    OpenAI     {"error": {"message", "type", "param", "code"}}
    Anthropic  {"type": "error", "error": {"type", "message"}}

`install(app)` wraps the app's handlers for HTTPException and validation errors: on /v1/ paths they
answer in the right shape (validation errors become 400, as both APIs do); elsewhere the previous
handler (FastAPI's default, or whatever another module installed) still runs.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import public_error

Flavor = Literal["openai", "anthropic"]


class InvalidRequest(HTTPException):
    """A 400 that names the offending field (OpenAI's `param`) and, where OpenAI has one, its error
    `code` (such as context_length_exceeded), so SDK users can tell the cases apart."""

    def __init__(self, message: str, param: str | None = None, code: str | None = None):
        super().__init__(400, message)
        self.param = param
        self.code = code


def flavor(request: Request) -> Flavor | None:
    """Which API's error shape this request expects; None outside /v1/."""
    path = request.url.path
    if not path.startswith("/v1/"):
        return None
    if path == "/v1/messages" or path.startswith("/v1/messages/"):
        return "anthropic"
    # GET /v1/models is shared: Anthropic SDKs (and Claude Code) always send anthropic-version.
    if path.startswith("/v1/models") and "anthropic-version" in request.headers:
        return "anthropic"
    return "openai"


def is_overloaded(error: BaseException) -> bool:
    """kindling.ops raises `Overloaded` when the generation queue is full. Matched by name, so this
    module works whether or not (and wherever) that class exists."""
    return any(cls.__name__ == "Overloaded" for cls in type(error).__mro__)


def model_error(error: Exception) -> HTTPException:
    """An exception from a model call as the HTTP error a client should see."""
    if isinstance(error, HTTPException):
        return error
    if is_overloaded(error):
        retry_after = getattr(error, "retry_after", None)
        headers = {"Retry-After": str(int(retry_after))} if retry_after else None
        return HTTPException(503, str(error) or "the server is busy, try again shortly", headers=headers)
    return HTTPException(502, public_error(error))


_OPENAI_CODES = {401: "invalid_api_key", 403: "permission_denied", 404: "not_found", 413: "request_too_large"}

_ANTHROPIC_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def openai_error(status: int, message: str, param: str | None = None, code: str | None = None) -> dict[str, Any]:
    if status == 429:
        kind, default = "rate_limit_error", "rate_limit_exceeded"
    elif status == 503:
        kind, default = "server_error", "overloaded"
    elif status >= 500:
        kind, default = "server_error", None
    else:
        kind, default = "invalid_request_error", _OPENAI_CODES.get(status)
    return {"error": {"message": message, "type": kind, "param": param, "code": code or default}}


def anthropic_error(status: int, message: str) -> tuple[int, dict[str, Any]]:
    """(status, body). Anthropic signals overload with 529, so a 503 becomes one."""
    kind = _ANTHROPIC_TYPES.get(status) or ("api_error" if status >= 500 else "invalid_request_error")
    if kind == "overloaded_error":
        status = 529
    return status, {"type": "error", "error": {"type": kind, "message": message}}


def error_response(
    kind: Flavor,
    status: int,
    message: str,
    *,
    param: str | None = None,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    if kind == "anthropic":
        status, body = anthropic_error(status, message)
    else:
        body = openai_error(status, message, param, code)
    return JSONResponse(body, status_code=status, headers=headers)


def response_for(kind: Flavor, error: Exception) -> JSONResponse:
    """The error response for a failed model call (or any HTTPException)."""
    http = model_error(error)
    return error_response(
        kind,
        http.status_code,
        _detail(http.detail),
        param=getattr(http, "param", None),
        code=getattr(http, "code", None),
        headers=http.headers,
    )


def _detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    return json.dumps(detail, default=str)


def _validation_message(error: RequestValidationError) -> tuple[str, str | None]:
    problems = error.errors()
    if not problems:
        return "invalid request", None
    parts = []
    for problem in problems[:3]:
        loc = [str(p) for p in problem.get("loc", ()) if p != "body"]
        parts.append(f"{'.'.join(loc)}: {problem.get('msg', 'invalid')}" if loc else problem.get("msg", "invalid"))
    first = [str(p) for p in problems[0].get("loc", ()) if p != "body"]
    return "; ".join(parts), ".".join(first) or None


Handler = Callable[[Request, Exception], Response | Awaitable[Response]]


def _wrap(previous: Handler) -> Handler:
    async def handle(request: Request, error: Exception) -> Response:
        kind = flavor(request)
        if kind is None:
            result = previous(request, error)
            return await result if inspect.isawaitable(result) else result
        if isinstance(error, RequestValidationError):
            message, param = _validation_message(error)
            return error_response(kind, 400, message, param=param)
        assert isinstance(error, StarletteHTTPException)
        return error_response(
            kind,
            error.status_code,
            _detail(error.detail),
            param=getattr(error, "param", None),
            code=getattr(error, "code", None),
            headers=getattr(error, "headers", None),
        )

    return handle


def install(app: FastAPI) -> None:
    """Idempotent: both compat modules call it."""
    if getattr(app.state, "compat_errors_installed", False):
        return
    app.state.compat_errors_installed = True
    defaults: dict[type[Exception], Handler] = {
        StarletteHTTPException: http_exception_handler,
        RequestValidationError: request_validation_exception_handler,
    }
    for cls, default in defaults.items():
        app.add_exception_handler(cls, _wrap(app.exception_handlers.get(cls, default)))
