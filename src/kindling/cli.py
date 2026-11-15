"""The `kindling` command.

kindling serve                 run the API server
kindling db upgrade            apply database migrations
kindling check                 is the model reachable, is the database up, is the models file valid?
kindling models                list the models this server offers
kindling keys ... / usage      API keys and usage (kindling.auth_cli)
kindling eval ...              the eval harness (kindling.evals.cli)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import get_settings


def _alembic_config():
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]  # the repo root, where alembic.ini lives
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    return config


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .main import create_app

    settings = get_settings()
    host, port = args.host or settings.host, args.port or settings.port
    uvicorn.run(create_app(settings), host=host, port=port, log_config=None, proxy_headers=False)
    return 0


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    from alembic import command

    command.upgrade(_alembic_config(), args.revision)
    print(f"database at {get_settings().database_url.split('@')[-1]} is at revision {args.revision}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .llm import ModelRegistry

    registry = ModelRegistry.from_settings(get_settings())
    for spec in registry.specs():
        mark = "*" if spec.id == registry.default else " "
        aliases = f"  (also: {', '.join(spec.aliases)})" if spec.aliases else ""
        print(f"{mark} {spec.id:<20} {spec.provider:<7} {spec.name}  ctx={spec.num_ctx}{aliases}")
    return 0


async def _check() -> int:
    from .llm import ModelNotReady, ModelRegistry
    from .main import build_store

    settings = get_settings()
    failures = 0
    try:
        registry = ModelRegistry.from_settings(settings)
    except ValueError as error:
        print(f"models file: {error}")
        return 1
    for spec in registry.specs():
        try:
            await registry.get(spec.id).check()
            print(f"model {spec.id} ({spec.name}): ok")
        except (ModelNotReady, NotImplementedError) as error:
            failures += 1
            print(f"model {spec.id} ({spec.name}): {error}")
    try:
        store = await build_store(settings)
        await store.close()
        print(f"store ({settings.store}): ok")
    except RuntimeError as error:
        failures += 1
        print(f"store: {error}")
    if settings.auth_disabled:
        print("auth: DISABLED (only safe if nobody else can reach this server)")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kindling", description="A local AI assistant backend.")
    parser.add_argument("--version", action="version", version=f"kindling {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", help="interface to listen on (default: KINDLING_HOST or 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default: KINDLING_PORT or 8000)")
    serve.set_defaults(func=cmd_serve)

    db = sub.add_parser("db", help="database migrations").add_subparsers(dest="db_command", required=True)
    upgrade = db.add_parser("upgrade", help="apply migrations")
    upgrade.add_argument("revision", nargs="?", default="head")
    upgrade.set_defaults(func=cmd_db_upgrade)

    sub.add_parser("check", help="check the model, the database and the models file").set_defaults(
        func=lambda args: asyncio.run(_check())
    )
    sub.add_parser("models", help="list the models this server offers").set_defaults(func=cmd_models)

    from . import auth_cli
    from .evals import cli as eval_cli

    auth_cli.register(sub)
    eval_cli.register(sub)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
