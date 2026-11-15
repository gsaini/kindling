"""`kindling keys ...` and `kindling usage`: issue and manage API keys from the shell.

These talk to the database (KINDLING_DATABASE_URL) directly, so the first keys can be issued
before the server has ever run. A revocation here reaches a running server within about ten
seconds (its lookup cache); the admin API's takes effect at once.

    kindling keys create --owner gopal --name hearth --rpm 120
    kindling keys create --owner gopal --name claude-code --models nemotron --daily-tokens 2000000 --expires 90d
    kindling keys list [--owner gopal] [--all]
    kindling keys show|revoke|rotate <id or prefix>
    kindling usage [--owner gopal] [--key <id or prefix>] [--since 7d] [--by key|model|day]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from .keys import (
    ApiKey,
    InvalidKeySettings,
    KeyStore,
    check_models,
    create_key,
    find_one,
    parse_time,
    revoke_key,
    rotate_key,
)
from .usage import UsageLog

SECRET_WARNING = "Store this key now: it is shown only once. Anyone holding it can use the server as this key."


def _engine():
    """The database engine. Tests replace this to point at a scratch schema."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from .config import get_settings

    return create_async_engine(get_settings().database_url)


def _registry():
    from .config import get_settings
    from .llm import ModelRegistry

    return ModelRegistry.from_settings(get_settings())


def _when(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else "-"


def _describe(key: ApiKey) -> str:
    models = ", ".join(sorted(key.models)) if key.models is not None else "all"
    rpm = key.rate_limit_per_minute or "server default"
    daily = f"{key.daily_token_limit:,}" if key.daily_token_limit else "unlimited"
    lines = [
        f"id          {key.id}",
        f"owner/name  {key.owner}/{key.name}",
        f"prefix      {key.prefix}...",
        f"state       {key.state()}",
        f"scopes      {', '.join(sorted(key.scopes))}",
        f"models      {models}",
        f"limits      {rpm} requests/min, {daily} tokens/day",
        f"created     {_when(key.created_at)}",
        f"expires     {_when(key.expires_at)}",
        f"last used   {_when(key.last_used_at)}",
    ]
    if key.revoked_at:
        lines.append(f"revoked     {_when(key.revoked_at)}")
    if key.note:
        lines.append(f"note        {key.note}")
    return "\n".join(lines)


def _show_secret(secret: str, key: ApiKey) -> None:
    print(_describe(key))
    print()
    print(f"  {secret}")
    print()
    print(f"WARNING: {SECRET_WARNING}", file=sys.stderr)


# -- commands: each takes (args, keys, usage log) -----------------------------------------


async def _create(args: argparse.Namespace, keys: KeyStore, usage: UsageLog) -> int:
    models = check_models(args.models.split(","), _registry()) if args.models else None
    key, secret = await create_key(
        keys,
        owner=args.owner,
        name=args.name,
        scopes=args.scopes.split(","),
        models=models,
        rate_limit_per_minute=args.rpm,
        daily_token_limit=args.daily_tokens,
        expires_at=parse_time(args.expires, ago=False) if args.expires else None,
        note=args.note or "",
    )
    _show_secret(secret, key)
    return 0


async def _list(args: argparse.Namespace, keys: KeyStore, usage: UsageLog) -> int:
    found = await keys.list(owner=args.owner, include_revoked=args.all)
    if not found:
        print("no keys" + (f" for {args.owner}" if args.owner else ""))
        return 0
    print(f"{'prefix':<16} {'owner/name':<32} {'scopes':<12} {'state':<8} {'last used':<21} id")
    for k in found:
        label = f"{k.owner}/{k.name}"
        print(
            f"{k.prefix:<16} {label:<32} {','.join(sorted(k.scopes)):<12} {k.state():<8} "
            f"{_when(k.last_used_at):<21} {k.id}"
        )
    return 0


async def _show(args: argparse.Namespace, keys: KeyStore, usage: UsageLog) -> int:
    print(_describe(await find_one(keys, args.ref)))
    return 0


async def _revoke(args: argparse.Namespace, keys: KeyStore, usage: UsageLog) -> int:
    key = await revoke_key(keys, await find_one(keys, args.ref))
    print(f"revoked {key.owner}/{key.name} ({key.prefix}..., {key.id})")
    return 0


async def _rotate(args: argparse.Namespace, keys: KeyStore, usage: UsageLog) -> int:
    old, new, secret = await rotate_key(keys, await find_one(keys, args.ref))
    print(f"revoked the old key {old.prefix}... ({old.id}); its replacement:")
    _show_secret(secret, new)
    return 0


async def _usage(args: argparse.Namespace, keys: KeyStore, usage: UsageLog) -> int:
    key_id = str((await find_one(keys, args.key)).id) if args.key else None
    since = parse_time(args.since, ago=True)
    rows = await usage.summary(owner=args.owner, key_id=key_id, since=since, group_by=args.by)
    print(f"usage since {_when(since)}" + (f" for {args.owner}" if args.owner else ""))
    label: Callable[[dict[str, Any]], str] = {
        "key": lambda r: f"{r['owner']}/{r['key_name']}",
        "model": lambda r: r["model"],
        "day": lambda r: r["day"],
        None: lambda r: "total",
    }[args.by]
    print(f"{args.by or '':<32} {'requests':>9} {'errors':>7} {'prompt tok':>12} {'output tok':>12}")
    for r in rows:
        print(
            f"{label(r):<32} {r['requests']:>9,} {r['errors']:>7,} {r['prompt_tokens']:>12,} {r['output_tokens']:>12,}"
        )
    return 0


async def _run(command, args: argparse.Namespace) -> int:
    from .keys import SqlKeyStore
    from .usage import SqlUsageLog

    engine = _engine()
    try:
        return await command(args, SqlKeyStore(engine), SqlUsageLog(engine))
    finally:
        await engine.dispose()


def _command(command):
    def run(args: argparse.Namespace) -> int:
        try:
            return asyncio.run(_run(command, args))
        except (InvalidKeySettings, LookupError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        except (OSError, SQLAlchemyError) as error:  # no database, or not migrated
            print(f"error: database: {type(error).__name__}: {error}", file=sys.stderr)
            print("Is it running and migrated? `docker compose up -d db && kindling db upgrade`", file=sys.stderr)
            return 1

    return run


def register(subparsers: argparse._SubParsersAction) -> None:
    keys = subparsers.add_parser("keys", help="API keys: create, list, show, revoke, rotate")
    sub = keys.add_subparsers(dest="keys_command", required=True)

    create = sub.add_parser("create", help="issue a key (printed once)")
    create.add_argument("--owner", required=True, help="the account / team / app; its conversations go with it")
    create.add_argument("--name", required=True, help="the use case: hearth, claude-code, evals, ...")
    create.add_argument("--scopes", default="chat", help="comma-separated: chat, admin (default: chat)")
    create.add_argument("--models", help="comma-separated model ids or aliases it may use (default: all)")
    create.add_argument("--rpm", type=int, help="requests per minute (default: KINDLING_RATE_LIMIT_PER_MINUTE)")
    create.add_argument("--daily-tokens", type=int, help="prompt + output tokens per UTC day (default: unlimited)")
    create.add_argument("--expires", help="a duration (90d, 12h) or a date (2027-01-31); default: never")
    create.add_argument("--note", help="free text: who has it, where it's configured")
    create.set_defaults(func=_command(_create))

    listing = sub.add_parser("list", help="list keys (active ones unless --all)")
    listing.add_argument("--owner")
    listing.add_argument("--all", action="store_true", help="include revoked keys")
    listing.set_defaults(func=_command(_list))

    for name, command, text in (
        ("show", _show, "show a key's settings"),
        ("revoke", _revoke, "revoke a key: it stops working"),
        ("rotate", _rotate, "replace a key's secret, keeping its settings; the old one stops working"),
    ):
        parser = sub.add_parser(name, help=text)
        parser.add_argument("ref", help="the key's id, or its prefix (kdl_...)")
        parser.set_defaults(func=_command(command))

    usage = subparsers.add_parser("usage", help="requests and tokens per key, model or day")
    usage.add_argument("--owner")
    usage.add_argument("--key", help="a key id or prefix")
    usage.add_argument("--since", default="7d", help="a duration back from now (7d, 24h) or a date (default: 7d)")
    usage.add_argument("--by", choices=["key", "model", "day"], help="group the totals")
    usage.set_defaults(func=_command(_usage))
