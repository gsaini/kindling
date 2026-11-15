"""Helpers for tests against a real Postgres, each in its own scratch schema."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

ROOT = Path(__file__).resolve().parents[1]


def alembic_config() -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["configure_logger"] = False
    return config


@asynccontextmanager
async def migrated_engine(url: str, schema: str):
    """An engine whose search_path is a fresh schema with every migration applied."""
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine: AsyncEngine = create_async_engine(
        url, connect_args={"server_settings": {"search_path": f"{schema},public"}}
    )
    try:
        async with engine.begin() as connection:

            def upgrade(sync_connection) -> None:
                config = alembic_config()
                config.attributes["connection"] = sync_connection
                command.upgrade(config, "head")

            await connection.run_sync(upgrade)
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
