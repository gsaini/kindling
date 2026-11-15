"""Logging: one handler on the root logger, text for people or JSON lines for log shippers.

Every record carries the current request id (from a contextvar the request middleware sets), so a
model error deep in the agent loop can be tied back to the request that caused it.

What never gets logged: request bodies, prompts, replies, and the Authorization / x-api-key headers.
The access log is built from an explicit list of fields, not from the request, to keep it that way.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO

request_id_var: ContextVar[str | None] = ContextVar("kindling_request_id", default=None)


def current_request_id() -> str | None:
    return request_id_var.get()


# Attributes every LogRecord has; anything else on a record came from `extra=` and is worth showing.
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "request_id"}

# Access-log fields shown after the message in text format (method, path and status are in the message).
_TEXT_EXTRAS = ("client_ip", "key_owner", "key_name")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            record.request_id = request_id_var.get()
        return True


def _timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


_CONTROL = {c: f"\\x{c:02x}" for c in [*range(0x20), 0x7F]}


def _clean(text: str) -> str:
    """Control characters (newlines from a percent-decoded path, terminal escapes) as \\xNN."""
    return text.translate(_CONTROL)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = (
            f"{_timestamp(record)} {record.levelname:<7} {record.name} [{getattr(record, 'request_id', None) or '-'}] "
        )
        line += _clean(record.getMessage())
        extras = [
            f"{key}={_clean(str(getattr(record, key)))}"
            for key in _TEXT_EXTRAS
            if getattr(record, key, None) is not None
        ]
        if extras:
            line += " " + " ".join(extras)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields: dict[str, object] = {
            "ts": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if getattr(record, "request_id", None):
            fields["request_id"] = record.request_id
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_") and value is not None:
                fields[key] = value
        if record.exc_info:
            fields["exc"] = self.formatException(record.exc_info)
        return json.dumps(fields, ensure_ascii=False, default=str)


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is now, so swapped streams (pytest's capture) never go stale."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self) -> IO[str]:  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value: IO[str]) -> None:
        pass


def build_handler(log_format: str, stream: IO[str] | None = None) -> logging.Handler:
    handler: logging.Handler = logging.StreamHandler(stream) if stream is not None else _StderrHandler()
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    handler.addFilter(RequestIdFilter())
    return handler


def configure_logging(log_format: str, level: str) -> None:
    """Install Kindling's handler on the root logger. Safe to call again (each app in the tests does)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_kindling", False):
            root.removeHandler(handler)
    handler = build_handler(log_format)
    handler._kindling = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Our access log replaces uvicorn's, which would also log query strings.
    logging.getLogger("uvicorn.access").disabled = True
    # httpx logs every upstream call (each Ollama request) at INFO, with full URLs.
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
