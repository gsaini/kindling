"""The Postgres store and the migrations, against a real database (see conftest.py)."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from kindling.store import NewMessage, SqlStore
from kindling.store.models import Base

from .pg import migrated_engine

pytestmark = pytest.mark.anyio


async def test_conversations_round_trip_per_owner(pg_schema):
    async with migrated_engine(*pg_schema) as engine:
        store = SqlStore(engine=engine)
        mine = await store.create("alice", "First")
        await store.create("bob", "Not yours")
        await store.append("alice", mine.id, [NewMessage(role="user", content="hi")])
        await store.append(
            "alice", mine.id, [NewMessage(role="assistant", content="hello", stats={"output_tokens": 3})]
        )
        got = await store.get("alice", mine.id)
        assert [(m.seq, m.role, m.content) for m in got.messages] == [(0, "user", "hi"), (1, "assistant", "hello")]
        assert got.messages[1].stats == {"output_tokens": 3}
        assert [c.title for c in await store.list("alice")] == ["First"]
        assert (await store.list("alice"))[0].message_count == 2
        # Another owner can neither see nor change it.
        assert await store.get("bob", mine.id) is None
        assert not await store.rename("bob", mine.id, "mine now")
        assert not await store.delete("bob", mine.id)
        with pytest.raises(KeyError):
            await store.append("bob", mine.id, [NewMessage(role="user", content="sneaky")])
        assert await store.rename("alice", mine.id, "Renamed") and await store.delete("alice", mine.id)
        assert await store.get("alice", mine.id) is None


async def test_migrations_match_the_models(pg_schema):
    async with migrated_engine(*pg_schema) as engine:
        async with engine.connect() as connection:
            diff = await connection.run_sync(
                lambda sync: compare_metadata(MigrationContext.configure(sync), Base.metadata)
            )
    assert diff == [], f"models and migrations differ; write a migration: {diff}"
