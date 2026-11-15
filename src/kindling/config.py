"""Settings, read from the environment (prefix KINDLING_) or a .env file.

Each block below belongs to one area of the code; add new settings to the block they belong to.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KINDLING_", env_file=".env", extra="ignore")

    # --- models (kindling.llm.registry) ------------------------------------------------------
    # A per-machine models.toml lists the models this server offers (see models.example.toml).
    # Without one, a single Ollama model is built from the three settings below.
    models_file: Path = Path("models.toml")
    model: str = "nemotron-3-nano:30b"
    ollama_host: str | None = None  # None: the ollama client's default (OLLAMA_HOST or localhost:11434)
    # Ollama's default window is small and it truncates silently, so ask for what we need.
    num_ctx: int = 16384
    system_prompt: str = ""

    # --- storage -----------------------------------------------------------------------------
    # "postgres" for real use, "memory" for tests and quick runs without a database.
    store: Literal["postgres", "memory"] = "postgres"
    database_url: str = "postgresql+asyncpg://kindling:kindling@localhost:5433/kindling"

    # --- tools -------------------------------------------------------------------------------
    tools_enabled: bool = True
    max_tool_steps: int = 8  # model turns per request before the agent loop gives up
    workspace: Path = Path("workspace")  # the only folder the file tools can read

    # --- auth (kindling.auth, kindling.usage) ------------------------------------------------
    # Secure by default: every /api and /v1 call needs a key. Set KINDLING_AUTH_DISABLED=true only
    # for a server nobody else can reach. (A tunnel to a 127.0.0.1 server counts as reachable.)
    auth_disabled: bool = False
    # A bootstrap admin key from the environment, for first setup (creating real keys) and CI.
    api_key: str = ""
    rate_limit_per_minute: int = 60  # default requests per minute per key; a key can override it

    # --- ops (kindling.ops) ------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_format: Literal["text", "json"] = "text"
    log_level: str = "INFO"
    cors_origins: list[str] = []  # browser origins allowed to call the API directly
    max_request_bytes: int = 2_000_000
    max_concurrent_generations: int = 2  # model calls running at once; the rest wait in a queue
    max_queued_generations: int = 16  # beyond this, new requests are turned away at once
    queue_timeout_seconds: float = 60.0  # how long a request may wait for a free slot
    metrics_enabled: bool = True
    api_docs: bool = False  # serve /docs and /openapi.json (they describe every route; off when exposed)

    # --- edge (kindling.edge): Cloudflare Tunnel / AI Gateway in front of the server ----------
    trust_cloudflare_headers: bool = False  # take the client IP from CF-Connecting-IP (only behind Cloudflare)
    origin_secret: str = ""  # if set, requests must carry it in `origin_secret_header` (only the edge knows it)
    origin_secret_header: str = "X-Kindling-Origin-Secret"

    # --- evals (kindling.evals) --------------------------------------------------------------
    eval_results_dir: Path = Path("evals/results")


@lru_cache
def get_settings() -> Settings:
    return Settings()
