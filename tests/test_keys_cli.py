"""`kindling keys ...` and `kindling usage`: offline against memory stores, and against Postgres."""

from __future__ import annotations

import re

import anyio
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from kindling import auth_cli
from kindling.cli import main
from kindling.keys import MemoryKeyStore, hash_key
from kindling.llm import FakeModel, ModelRegistry
from kindling.usage import MemoryUsageLog, UsageEvent

from .pg import migrated_engine

SECRET = re.compile(r"kdl_[A-Za-z0-9_-]{43}")


@pytest.fixture
def offline(monkeypatch):
    """Point the CLI at memory stores (shared across calls) and a one-model registry."""
    keys, usage = MemoryKeyStore(), MemoryUsageLog()

    async def run(command, args):
        return await command(args, keys, usage)

    monkeypatch.setattr(auth_cli, "_run", run)
    monkeypatch.setattr(auth_cli, "_registry", lambda: ModelRegistry.single(FakeModel()))
    return keys, usage


def secret_in(text: str) -> str:
    [secret] = SECRET.findall(text)
    return secret


def test_keys_commands(offline, capsys):
    keys, usage = offline
    argv = "keys create --owner gopal --name claude-code --models fake --rpm 30 --daily-tokens 2000000 --expires 90d"
    assert main([*argv.split(), "--note", "work laptop"]) == 0
    out, err = capsys.readouterr()
    secret = secret_in(out)
    assert "shown only once" in err and "fake" in out and "30 requests/min" in out and "2,000,000" in out

    async def stored():
        return await keys.by_hash(hash_key(secret))

    key = anyio.run(stored)
    assert key.owner == "gopal" and key.note == "work laptop" and key.expires_at is not None

    assert main(["keys", "list"]) == 0
    assert key.prefix in capsys.readouterr().out
    assert main(["keys", "show", key.prefix]) == 0
    shown = capsys.readouterr().out
    assert str(key.id) in shown and secret not in shown

    assert main(["keys", "rotate", str(key.id)]) == 0
    new_secret = secret_in(capsys.readouterr().out)
    assert new_secret != secret
    assert main(["keys", "list"]) == 0
    assert key.prefix not in capsys.readouterr().out
    assert main(["keys", "list", "--all"]) == 0
    assert "revoked" in capsys.readouterr().out

    assert main(["keys", "revoke", new_secret[:12]]) == 0
    assert "revoked gopal/claude-code" in capsys.readouterr().out


def test_keys_command_errors(offline, capsys):
    assert main(["keys", "create", "--owner", "gopal", "--name", "x", "--models", "gpt-9"]) == 2
    assert "Available: fake" in capsys.readouterr().err
    assert main(["keys", "create", "--owner", "bad owner", "--name", "x"]) == 2
    assert main(["keys", "create", "--owner", "o", "--name", "x", "--scopes", "root"]) == 2
    assert main(["keys", "create", "--owner", "o", "--name", "x", "--expires", "someday"]) == 2
    assert main(["keys", "show", "kdl_nothing"]) == 2
    assert "no key matches" in capsys.readouterr().err


def test_usage_command(offline, capsys):
    keys, usage = offline
    usage.events += [
        UsageEvent("gopal", None, "hearth", "/api/chat", "fake", 1000, 500),
        UsageEvent("evals", None, "runner", "/api/chat", "fake", 10, 5, status="error"),
    ]
    assert main(["usage", "--by", "key"]) == 0
    out = capsys.readouterr().out
    assert "gopal/hearth" in out and "1,000" in out and "evals/runner" in out
    assert main(["usage", "--owner", "evals"]) == 0
    assert "total" in capsys.readouterr().out
    assert main(["usage", "--since", "someday"]) == 2


@pytest.mark.anyio
async def test_cli_against_postgres(pg_schema, monkeypatch, capsys):
    url, schema = pg_schema
    monkeypatch.setattr(auth_cli, "_registry", lambda: ModelRegistry.single(FakeModel()))
    monkeypatch.setattr(
        auth_cli,
        "_engine",
        lambda: create_async_engine(url, connect_args={"server_settings": {"search_path": f"{schema},public"}}),
    )
    async with migrated_engine(url, schema):
        # The CLI runs its own event loop, so give it a thread of its own.
        run = lambda *argv: anyio.to_thread.run_sync(main, list(argv))  # noqa: E731
        assert await run("keys", "create", "--owner", "gopal", "--name", "hearth", "--models", "fake") == 0
        secret = secret_in(capsys.readouterr().out)
        assert await run("keys", "show", secret[:12]) == 0
        assert "gopal/hearth" in capsys.readouterr().out
        assert await run("keys", "rotate", secret[:12]) == 0
        new_secret = secret_in(capsys.readouterr().out)
        assert await run("keys", "revoke", new_secret[:12]) == 0
        capsys.readouterr()
        assert await run("keys", "list", "--all") == 0
        assert capsys.readouterr().out.count(" revoked ") == 2
        assert await run("usage", "--by", "day") == 0
