"""Keys, the rate limiter and the token counters, without HTTP; and the key store on Postgres."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kindling.keys import (
    InvalidKeySettings,
    MemoryKeyStore,
    RateLimiter,
    SqlKeyStore,
    TokenCounters,
    check_label,
    check_models,
    check_scopes,
    create_key,
    find_one,
    hash_key,
    parse_duration,
    parse_time,
    revoke_key,
    rotate_key,
)
from kindling.llm import FakeModel, ModelRegistry, ModelSpec
from kindling.usage import MemoryUsageLog, SqlUsageLog, UsageEvent

from .pg import migrated_engine

pytestmark = pytest.mark.anyio


def registry() -> ModelRegistry:
    specs = [ModelSpec(id="fake", name="fake", aliases=["f"]), ModelSpec(id="other", name="other")]
    return ModelRegistry(specs, "fake", models={"fake": FakeModel(), "other": FakeModel()})


async def lifecycle(keys) -> None:
    key, secret = await create_key(keys, owner="gopal", name="hearth", scopes=["chat"], rate_limit_per_minute=30)
    assert secret.startswith("kdl_") and len(secret) > 40
    assert key.hash == hash_key(secret) and secret not in str(key.public()) and "hash" not in key.public()
    assert key.prefix == secret[:12] and key.state() == "active"
    assert await keys.by_hash(hash_key(secret)) == key
    assert await keys.by_hash(hash_key(secret + "x")) is None
    assert await find_one(keys, key.prefix) == key and await find_one(keys, str(key.id)) == key

    other, _ = await create_key(keys, owner="evals", name="runner", models=frozenset({"fake"}), daily_token_limit=10)
    assert [k.name for k in await keys.list()] == ["runner", "hearth"]
    assert [k.name for k in await keys.list(owner="gopal")] == ["hearth"]
    assert (await keys.get(other.id)).models == frozenset({"fake"})

    old, new, new_secret = await rotate_key(keys, key)
    assert old.state() == "revoked" and new.state() == "active" and new.id != key.id
    assert (new.owner, new.name, new.scopes, new.rate_limit_per_minute) == ("gopal", "hearth", key.scopes, 30)
    assert (await keys.by_hash(hash_key(secret))).state() == "revoked"
    assert (await keys.by_hash(hash_key(new_secret))).id == new.id
    with pytest.raises(InvalidKeySettings):
        await rotate_key(keys, old)

    revoked = await revoke_key(keys, new)
    assert (await keys.get(new.id)).revoked_at == revoked.revoked_at
    assert [k.name for k in await keys.list()] == ["runner"]
    assert len(await keys.list(include_revoked=True)) == 3

    when = datetime.now(UTC)
    await keys.touch(other.id, when)
    assert abs((await keys.get(other.id)).last_used_at - when) < timedelta(seconds=1)


async def test_key_lifecycle_in_memory():
    await lifecycle(MemoryKeyStore())


async def test_key_lifecycle_on_postgres(pg_schema):
    async with migrated_engine(*pg_schema) as engine:
        await lifecycle(SqlKeyStore(engine))
        # Prefix search treats LIKE wildcards literally.
        assert await SqlKeyStore(engine).find("kdl_%") == []


def test_validation():
    assert check_label("claude-code", "name") == "claude-code"
    for bad in ["", " ", "-x", "a b", "x" * 101, "drop;table", "ü"]:
        with pytest.raises(InvalidKeySettings):
            check_label(bad, "name")
    assert check_scopes(["chat", "admin"]) == {"chat", "admin"}
    for bad in [[], ["root"], ["chat", "write"]]:
        with pytest.raises(InvalidKeySettings):
            check_scopes(bad)
    assert check_models(["f", "other"], registry()) == {"fake", "other"}  # aliases become ids
    assert check_models(None, registry()) is None
    with pytest.raises(InvalidKeySettings, match="Available: fake, other"):
        check_models(["gpt-9"], registry())
    assert parse_duration("90d") == timedelta(days=90) and parse_duration("12h") == timedelta(hours=12)
    assert parse_time("2027-01-31", ago=False) == datetime(2027, 1, 31, tzinfo=UTC)
    with pytest.raises(InvalidKeySettings):
        parse_time("soon", ago=False)


async def test_an_expiry_in_the_past_is_refused():
    with pytest.raises(InvalidKeySettings):
        await create_key(MemoryKeyStore(), owner="o", name="n", expires_at=datetime.now(UTC) - timedelta(days=1))


def test_rate_limiter_is_a_token_bucket():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert [limiter.hit("k", 3) for _ in range(3)] == [0, 0, 0]
    assert limiter.hit("k", 3) == pytest.approx(20)  # one request every 20 s at 3/min
    assert limiter.hit("other", 3) == 0  # buckets are per key
    now[0] = 20.0
    assert limiter.hit("k", 3) == 0 and limiter.hit("k", 3) > 0


async def test_token_counters_seed_from_the_log_then_count_in_memory():
    log = MemoryUsageLog()
    yesterday = datetime.now(UTC) - timedelta(days=1, hours=1)
    await log.add(UsageEvent("o", "k1", "n", "/api/chat", "fake", 100, 100, created_at=yesterday))
    await log.add(UsageEvent("o", "k1", "n", "/api/chat", "fake", 10, 5))
    counters = TokenCounters(log)
    assert await counters.used_today("k1") == 15
    log.events.clear()  # from now on the counter doesn't read the log
    counters.add(UsageEvent("o", "k1", "n", "/api/chat", "fake", 3, 2))
    counters.add(UsageEvent("o", "k2", "n", "/api/chat", "fake", 3, 2))  # not seeded: ignored until asked
    assert await counters.used_today("k1") == 20
    assert await counters.used_today("k2") == 0


async def test_sql_usage_log_sums_tokens_per_key(pg_schema):
    async with migrated_engine(*pg_schema) as engine:
        usage = SqlUsageLog(engine)
        key, _ = await create_key(SqlKeyStore(engine), owner="o", name="n")
        await usage.add(UsageEvent("o", str(key.id), "n", "/api/chat", "fake", 10, 5))
        await usage.add(UsageEvent("o", None, "bootstrap", "/api/chat", "fake", 1, 1))
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        assert await usage.tokens_since(str(key.id), today) == 15
