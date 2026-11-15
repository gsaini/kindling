"""Usage metering: what /api/chat records, that recording never breaks a request, and the SQL log."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kindling.keys import service_for
from kindling.llm import FakeModel, Turn
from kindling.store import MemoryStore
from kindling.usage import MemoryUsageLog, Recorder, SqlUsageLog, UsageEvent

from .conftest import ADMIN, read_events
from .pg import migrated_engine

HELLO = {"messages": [{"role": "user", "content": "hello"}]}


def test_chat_records_usage_for_the_key(make_client):
    store = MemoryStore()
    client = make_client(FakeModel(Turn(content="hi")), store=store)
    key = client.post("/api/admin/keys", headers=ADMIN, json={"owner": "gopal", "name": "hearth"}).json()
    assert read_events(client.post("/api/chat", headers={"x-api-key": key["key"]}, json=HELLO))[-1]["type"] == "done"
    [event] = service_for(store).usage_log.events
    assert (event.owner, event.key_id, event.key_name, event.endpoint, event.model) == (
        "gopal",
        key["id"],
        "hearth",
        "/api/chat",
        "fake",
    )
    assert (event.prompt_tokens, event.output_tokens, event.status) == (10, 5, "ok") and event.wall_seconds >= 0


def test_failed_chat_is_recorded_as_an_error(make_client):
    store = MemoryStore()
    client = make_client(FakeModel(), store=store)  # no scripted turns: the model call fails
    events = read_events(client.post("/api/chat", headers=ADMIN, json=HELLO))
    assert events[-1]["type"] == "error"
    [event] = service_for(store).usage_log.events
    assert (event.key_name, event.key_id, event.status) == ("bootstrap", None, "error")


class BrokenLog(MemoryUsageLog):
    async def add(self, event: UsageEvent) -> None:
        raise ConnectionError("database went away")


@pytest.mark.anyio
async def test_recording_never_raises(caplog):
    def explode(event):
        raise RuntimeError("bug")

    await Recorder(BrokenLog(), on_record=explode).record(UsageEvent("o", None, "n", "/api/chat", "fake"))
    assert "couldn't save" in caplog.text


def test_a_broken_usage_log_does_not_break_chat(make_client):
    store = MemoryStore()
    service_for(store).recorder.log = BrokenLog()
    client = make_client(FakeModel(Turn(content="hi")), store=store)
    assert read_events(client.post("/api/chat", headers=ADMIN, json=HELLO))[-1]["type"] == "done"


@pytest.mark.anyio
async def test_sql_usage_summary(pg_schema):
    async with migrated_engine(*pg_schema) as engine:
        log = SqlUsageLog(engine)
        key_a, key_b = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
        yesterday = datetime.now(UTC) - timedelta(days=1)
        for event in [
            UsageEvent("gopal", key_a, "hearth", "/api/chat", "nemotron", 10, 5),
            UsageEvent("gopal", key_a, "hearth", "/v1/messages", "nemotron", 20, 10, status="error"),
            UsageEvent("gopal", key_a, "hearth", "/api/chat", "big", 1, 1, created_at=yesterday),
            UsageEvent("evals", key_b, "runner", "/api/chat", "nemotron", 100, 50),
            UsageEvent("admin", None, "bootstrap", "/api/chat", "nemotron", 1, 1),
        ]:
            await log.add(event)
        [total] = await log.summary()
        assert (total["requests"], total["errors"], total["prompt_tokens"], total["output_tokens"]) == (5, 1, 132, 67)
        assert total["tokens"] == 199
        by_key = {r["key_name"]: r for r in await log.summary(group_by="key")}
        assert by_key["hearth"]["requests"] == 3 and by_key["hearth"]["key_id"] == key_a
        assert by_key["bootstrap"]["key_id"] is None
        by_model = {r["model"]: r["requests"] for r in await log.summary(group_by="model")}
        assert by_model == {"big": 1, "nemotron": 4}
        days = await log.summary(group_by="day")
        assert [d["requests"] for d in days] == [1, 4] and days[1]["day"] == datetime.now(UTC).date().isoformat()
        assert (await log.summary(owner="evals"))[0]["tokens"] == 150
        since = datetime.now(UTC) - timedelta(hours=1)
        assert (await log.summary(key_id=key_a, since=since))[0]["requests"] == 2
        assert (await log.summary(until=since))[0]["requests"] == 1
