"""Shared fixtures. Everything runs offline: a scripted FakeModel and an in-memory store.

Postgres tests run only when KINDLING_TEST_DATABASE_URL is set (CI, or `docker compose up -d db`):
    KINDLING_TEST_DATABASE_URL=postgresql+asyncpg://kindling:kindling@localhost:5433/kindling_test
Each test session works in its own scratch schema, so parallel runs don't collide.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kindling.config import Settings
from kindling.llm import FakeModel
from kindling.main import create_app
from kindling.store import MemoryStore

TEST_DATABASE_URL = os.environ.get("KINDLING_TEST_DATABASE_URL", "")
needs_postgres = pytest.mark.skipif(not TEST_DATABASE_URL, reason="set KINDLING_TEST_DATABASE_URL to run")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings for tests: memory store, a scratch workspace, auth on with a known bootstrap key."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Settings(
        _env_file=None,
        store="memory",
        workspace=workspace,
        api_key="test-admin-key",
        models_file=tmp_path / "no-models.toml",
        eval_results_dir=tmp_path / "eval-results",
    )


ADMIN = {"Authorization": "Bearer test-admin-key"}


@pytest.fixture
def make_client(settings: Settings):
    """make_client(FakeModel(...), **overrides) -> a started TestClient. Overrides go to create_app."""
    clients: list[TestClient] = []

    def make(model: FakeModel | None = None, *, settings_overrides: dict | None = None, **kwargs) -> TestClient:
        s = settings.model_copy(update=settings_overrides or {})
        app = create_app(s, model=model or FakeModel(), store=kwargs.pop("store", MemoryStore()), **kwargs)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def read_events(response) -> list[dict]:
    import json

    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.fixture
def pg_schema() -> Iterator[tuple[str, str]]:
    """(database_url, schema): a fresh schema in the test database, dropped afterwards."""
    if not TEST_DATABASE_URL:
        pytest.skip("set KINDLING_TEST_DATABASE_URL to run")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    yield TEST_DATABASE_URL, schema
