"""Running behind Cloudflare (Tunnel, AI Gateway): client IPs and origin protection.

Two opt-in behaviours, both in one plain ASGI middleware so streaming replies pass straight through:

- Origin secret (KINDLING_ORIGIN_SECRET): every request must carry the secret in
  KINDLING_ORIGIN_SECRET_HEADER, which a Cloudflare Transform Rule adds at the edge. Requests without it
  get 403, so only traffic that came through Cloudflare reaches the app. The header is removed before
  the app sees it, so it never reaches logs, tools or the model. Health probes are exempt (see
  EXEMPT_PATHS). API keys are still required on top of this: the secret proves the route, not the caller.

- Cloudflare client IPs (KINDLING_TRUST_CLOUDFLARE_HEADERS): take the client address from
  CF-Connecting-IP. Through a tunnel every request arrives from cloudflared on 127.0.0.1, so without
  this all callers look alike. Only safe when the server is reachable through Cloudflare alone (bound
  to 127.0.0.1 with a tunnel, or with an origin secret): anyone who can reach the port directly can
  put any address in that header.

With neither setting, install() adds nothing.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Receive, Scope, Send

    from .config import Settings

log = logging.getLogger("kindling.edge")

# Probes that run on the Mac itself (launchd checks, uptime scripts) and don't know the secret. They
# need no key and reveal nothing sensitive, so letting them through costs nothing.
EXEMPT_PATHS = frozenset({"/livez", "/readyz", "/api/health"})

CF_CONNECTING_IP = b"cf-connecting-ip"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def client_ip_from(value: bytes) -> str | None:
    """A clean IP address from a CF-Connecting-IP value, or None if it isn't one."""
    try:
        return str(ipaddress.ip_address(value.decode("latin-1").strip()))
    except ValueError:
        return None


class EdgeMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        origin_secret: str = "",
        origin_secret_header: str = "X-Kindling-Origin-Secret",
        trust_cloudflare_headers: bool = False,
        exempt_paths: frozenset[str] = EXEMPT_PATHS,
    ):
        self.app = app
        self.secret = origin_secret.encode() if origin_secret else None
        self.header = origin_secret_header.strip().lower().encode("latin-1")
        self.trust_cloudflare = trust_cloudflare_headers
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        came_through_edge = True
        if self.secret is not None:
            presented: list[bytes] = []
            kept: list[tuple[bytes, bytes]] = []
            for name, value in scope["headers"]:
                if name == self.header:
                    presented.append(value)
                else:
                    kept.append((name, value))
            # Exactly one copy, compared in constant time; duplicates are refused rather than guessed at.
            came_through_edge = len(presented) == 1 and hmac.compare_digest(presented[0], self.secret)
            if not came_through_edge and scope.get("path", "") not in self.exempt_paths:
                await self._forbid(scope, send, "missing" if not presented else "wrong")
                return
            scope = dict(scope, headers=kept)

        # A spoofed CF-Connecting-IP can only come from off the edge, so trust it only when the secret
        # (if there is one) checked out.
        if self.trust_cloudflare and came_through_edge:
            for name, value in scope["headers"]:
                if name == CF_CONNECTING_IP:
                    ip = client_ip_from(value)
                    if ip:
                        port = (scope.get("client") or (None, 0))[1]
                        scope = dict(scope, client=(ip, port))
                    break

        await self.app(scope, receive, send)

    async def _forbid(self, scope: Scope, send: Send, why: str) -> None:
        client = (scope.get("client") or ("?", 0))[0]
        log.warning("refused %s %s from %s: %s origin secret", scope.get("method", "WS"), scope["path"], client, why)
        if scope["type"] == "websocket":
            # Closing before accept makes the server answer the handshake with 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": "forbidden"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


def install(app: FastAPI, settings: Settings) -> None:
    if not settings.origin_secret and not settings.trust_cloudflare_headers:
        return
    if settings.trust_cloudflare_headers and not settings.origin_secret and settings.host not in _LOOPBACK_HOSTS:
        log.warning(
            "trusting CF-Connecting-IP while listening on %s with no origin secret: anyone who can reach "
            "the port can claim any client IP",
            settings.host,
        )
    # Added last, so it runs first: refused requests never reach the app, and the ops access log
    # already sees the real client address.
    app.add_middleware(
        EdgeMiddleware,
        origin_secret=settings.origin_secret,
        origin_secret_header=settings.origin_secret_header,
        trust_cloudflare_headers=settings.trust_cloudflare_headers,
    )
