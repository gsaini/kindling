"""`kindling eval ...` commands.

kindling eval run smoke [--target inproc|http|openai] [--model nemotron] [--think on|off] [--label baseline]
kindling eval compare A.jsonl B.jsonl
kindling eval list

`run` exits 1 when the pass rate is below --min-pass, so CI can gate on it; 2 for a bad dataset or flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import httpx

from .. import __version__
from ..config import Settings, get_settings
from ..llm import ModelRegistry
from . import report
from .dataset import DatasetError, evals_dir, find_datasets, load_dataset, resolve_dataset
from .runner import RunOptions, compare, load_results, run_eval, run_stem, select, summarise, write_run
from .targets import HttpTarget, Target, inproc_target, openai_target

API_KEY_ENV = "KINDLING_EVAL_API_KEY"
THINK = {"on": True, "off": False, "default": None}


def register(subparsers: argparse._SubParsersAction) -> None:
    evals = subparsers.add_parser("eval", help="run evals against a model or a server, and compare runs")
    sub = evals.add_subparsers(dest="eval_command", required=True)

    run = sub.add_parser("run", help="run a dataset and save the results")
    run.add_argument("dataset", help="a .jsonl file, or a name in evals/datasets (e.g. smoke)")
    run.add_argument("--target", choices=["inproc", "http", "openai"], default="inproc")
    run.add_argument("--url", help="server URL (http: Kindling's root; openai: the base URL ending in /v1)")
    run.add_argument("--api-key", help=f"key for the http/openai target (default: ${API_KEY_ENV})")
    run.add_argument("--model", help="model id to use for every case (default: each case's, else the server's)")
    run.add_argument("--think", choices=list(THINK), default="default", help="reasoning on/off for every case")
    run.add_argument("--repeat", type=int, default=1, help="run each case N times (times its own repeat)")
    run.add_argument("--concurrency", type=int, default=1, help="cases in flight at once")
    run.add_argument("--label", help="a name for this run, used in the file names (default: dataset-target)")
    run.add_argument("--filter", action="append", default=[], help="only cases with this tag or id (repeatable)")
    run.add_argument("--min-pass", type=float, default=0.0, help="exit 1 if the pass rate is below this (0-1)")
    run.add_argument("--workspace", type=Path, help="folder the file tools read (default: evals/fixtures)")
    run.add_argument("--timeout", type=float, default=600.0, help="seconds per request (http/openai)")
    run.add_argument("--out", type=Path, help="results folder (default: KINDLING_EVAL_RESULTS_DIR)")
    run.set_defaults(func=cmd_run)

    diff = sub.add_parser("compare", help="compare two runs case by case")
    diff.add_argument("before", type=Path, help="results .jsonl (or .summary.json) of the earlier run")
    diff.add_argument("after", type=Path, help="results of the later run")
    diff.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any case regressed")
    diff.set_defaults(func=cmd_compare)

    sub.add_parser("list", help="list the datasets in evals/datasets").set_defaults(func=cmd_list)


def main(argv: list[str] | None = None, **inject: Any) -> int:
    """Run `kindling eval ...` on its own; tests pass a settings / registry / http client in `inject`."""
    parser = argparse.ArgumentParser(prog="kindling")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["eval", *(sys.argv[1:] if argv is None else argv)])
    return args.func(args, **inject)


# -- run -------------------------------------------------------------------------------------


def _workspace(args: argparse.Namespace, dataset: Path, settings: Settings) -> Path:
    if args.workspace:
        return args.workspace
    for candidate in (dataset.resolve().parent.parent / "fixtures", evals_dir() / "fixtures"):
        if candidate.is_dir():
            return candidate
    return settings.workspace


def build_target(
    args: argparse.Namespace,
    settings: Settings,
    workspace: Path,
    *,
    registry: ModelRegistry | None = None,
    http_client: httpx.Client | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Target:
    api_key = args.api_key or os.environ.get(API_KEY_ENV, "")
    if args.target == "http":
        url = args.url or f"http://{settings.host}:{settings.port}"
        return HttpTarget(url, api_key, client=http_client, timeout=args.timeout)
    if args.target == "openai":
        url = args.url or "http://localhost:11434/v1"  # Ollama's OpenAI-compatible API
        return openai_target(
            url,
            api_key=api_key,
            workspace=workspace,
            max_steps=settings.max_tool_steps,
            transport=transport,
            timeout=args.timeout,
        )
    return inproc_target(
        registry or ModelRegistry.from_settings(settings),
        workspace=workspace,
        max_steps=settings.max_tool_steps,
        system_prompt=settings.system_prompt,
    )


def cmd_run(
    args: argparse.Namespace,
    *,
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    http_client: httpx.Client | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    settings = settings or get_settings()
    try:
        dataset = resolve_dataset(args.dataset)
        cases = load_dataset(dataset)
        if args.target == "openai" and not args.model and not all(c.model for c in cases):
            raise DatasetError("the openai target needs --model (the model name on that server)")
        if args.repeat < 1 or args.concurrency < 1:
            raise DatasetError("--repeat and --concurrency must be at least 1")
        workspace = _workspace(args, dataset, settings)
        target = build_target(
            args, settings, workspace, registry=registry, http_client=http_client, transport=transport
        )
    except (DatasetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    options = RunOptions(
        model=args.model, think=THINK[args.think], repeat=args.repeat, concurrency=args.concurrency, filters=args.filter
    )
    runnable, skipped = select(cases, target, options.filters)
    if skipped:
        print(f"skipping {len(skipped)} case(s) the {target.name} target can't run: {', '.join(c.id for c in skipped)}")
    if not runnable:
        print("error: no cases to run", file=sys.stderr)
        return 2

    label = args.label or f"{dataset.stem}-{target.name}"
    started = datetime.now(UTC)
    print(f"running {len(runnable)} case(s) from {dataset} on {target.name}, think={args.think}")
    results = anyio.run(run_eval, runnable, target, options, lambda r: print(report.result_line(r), flush=True))
    summary = summarise(
        results,
        label=label,
        dataset=str(dataset),
        **target.describe(),
        model=args.model,
        think=args.think,
        repeat=args.repeat,
        concurrency=args.concurrency,
        filters=args.filter,
        skipped=[c.id for c in skipped],
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        kindling_version=__version__,
    )
    print(report.summary_text(summary))
    folder = args.out or settings.eval_results_dir
    results_path, summary_path = write_run(folder, run_stem(label, started), results, summary)
    print(f"\nresults: {results_path}\nsummary: {summary_path}")
    if summary["pass_rate"] < args.min_pass:
        print(f"pass rate {summary['pass_rate']:.0%} is below --min-pass {args.min_pass:.0%}")
        return 1
    return 0


# -- compare / list --------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace, **_: Any) -> int:
    try:
        before, after = load_results(args.before), load_results(args.after)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    comparison = compare(before, after)
    print(report.comparison_text(comparison, str(args.before), str(args.after)))
    return 1 if args.fail_on_regression and comparison.count("regressed") else 0


def cmd_list(args: argparse.Namespace, **_: Any) -> int:
    datasets = find_datasets()
    if not datasets:
        print(f"no datasets in {evals_dir() / 'datasets'}")
        return 0
    for path in datasets:
        try:
            cases = load_dataset(path)
        except DatasetError as error:
            print(f"{path.stem:<16} invalid: {error}")
            continue
        tags = sorted({t for c in cases for t in c.tags})
        print(f"{path.stem:<16} {len(cases):>3} cases  {path}  tags: {', '.join(tags) or '-'}")
    return 0
