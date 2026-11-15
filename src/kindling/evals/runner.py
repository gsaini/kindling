"""Run cases against a target, score them, summarise, save, and compare two runs."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from .dataset import Case
from .scoring import CheckResult, score
from .targets import Output, Target


@dataclass
class Result:
    """One run of one case: what the results JSONL holds, a line each."""

    id: str
    run: int  # 1..repeat
    tags: list[str]
    passed: bool
    checks: list[dict[str, Any]]
    output: str
    thinking_chars: int
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    first_token_s: float | None
    tokens_per_second: float
    steps: int
    model: str
    think: bool | None
    error: str | None = None

    @classmethod
    def build(cls, case: Case, run: int, out: Output, think: bool | None) -> Result:
        checks: list[CheckResult] = [] if out.error else [score(c, out) for c in case.expect]
        return cls(
            id=case.id,
            run=run,
            tags=list(case.tags),
            passed=out.error is None and all(c.passed for c in checks),
            checks=[c.to_dict() for c in checks],
            output=out.text,
            thinking_chars=len(out.thinking),
            tool_calls=out.tool_calls,
            prompt_tokens=out.prompt_tokens,
            output_tokens=out.output_tokens,
            latency_s=round(out.latency_s, 3),
            first_token_s=round(out.first_token_s, 3) if out.first_token_s is not None else None,
            tokens_per_second=out.tokens_per_second,
            steps=out.steps,
            model=out.model,
            think=think,
            error=out.error,
        )

    @property
    def reason(self) -> str:
        """Why it failed, in one line."""
        if self.error:
            return self.error
        return "; ".join(f"{c['check']}: {c['reason']}" for c in self.checks if not c["passed"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunOptions:
    model: str | None = None  # overrides each case's model
    think: bool | None = None  # overrides each case's think; None: the case's, else the model's default
    repeat: int = 1  # multiplies each case's own repeat
    concurrency: int = 1
    filters: list[str] = field(default_factory=list)  # keep cases with any of these tags or ids


def select(cases: Sequence[Case], target: Target, filters: Sequence[str]) -> tuple[list[Case], list[Case]]:
    """(cases to run, cases this target can't run)."""
    wanted = [c for c in cases if not filters or c.id in filters or set(c.tags) & set(filters)]
    runnable = [c for c in wanted if not set(c.tags) & target.skip_tags]
    return runnable, [c for c in wanted if c not in runnable]


async def run_eval(
    cases: Sequence[Case],
    target: Target,
    options: RunOptions,
    on_result: Callable[[Result], None] | None = None,
) -> list[Result]:
    jobs = [(case, n) for case in cases for n in range(1, case.repeat * options.repeat + 1)]
    results: list[Result | None] = [None] * len(jobs)
    limiter = anyio.Semaphore(max(1, options.concurrency))

    async def one(index: int, case: Case, run: int) -> None:
        async with limiter:
            think = options.think if options.think is not None else case.think
            out = await target.run(case, model=options.model or case.model, think=think)
            result = Result.build(case, run, out, think)
            results[index] = result
            if on_result:
                on_result(result)

    async with anyio.create_task_group() as group:
        for index, (case, run) in enumerate(jobs):
            group.start_soon(one, index, case, run)
    return [r for r in results if r is not None]  # in dataset order, whatever order they finished in


# -- summaries -------------------------------------------------------------------------------


def _rate(results: Sequence[Result]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def summarise(results: Sequence[Result], **about: Any) -> dict[str, Any]:
    """Pass rate, speed and per-tag pass rates. `about` (dataset, target, label...) is copied in."""
    ok = [r for r in results if not r.error]
    latencies = [r.latency_s for r in ok]
    speeds = [r.tokens_per_second for r in ok if r.tokens_per_second]
    first_tokens = [r.first_token_s for r in ok if r.first_token_s is not None]
    tags: dict[str, list[Result]] = {}
    for r in results:
        for tag in r.tags or ["(untagged)"]:
            tags.setdefault(tag, []).append(r)
    return {
        **about,
        "cases": len({r.id for r in results}),
        "runs": len(results),
        "passed": sum(r.passed for r in results),
        "errors": len(results) - len(ok),
        "pass_rate": round(_rate(results), 4),
        "latency_mean_s": _mean(latencies),
        "latency_median_s": _median(latencies),
        "first_token_median_s": _median(first_tokens),
        "tokens_per_second_mean": round(statistics.fmean(speeds), 1) if speeds else None,
        "output_tokens_mean": _mean([r.output_tokens for r in ok]),
        "by_tag": {
            tag: {"runs": len(rs), "passed": sum(r.passed for r in rs), "pass_rate": round(_rate(rs), 4)}
            for tag, rs in sorted(tags.items())
        },
    }


# -- saving and loading ----------------------------------------------------------------------


def run_stem(label: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]+', '-', label).strip('-') or 'run'}"


def write_run(folder: Path, stem: str, results: Sequence[Result], summary: dict[str, Any]) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    results_path, summary_path = folder / f"{stem}.jsonl", folder / f"{stem}.summary.json"
    with results_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return results_path, summary_path


def load_results(path: Path) -> list[dict[str, Any]]:
    """A results JSONL, or its .summary.json (whose results file sits next to it)."""
    if path.name.endswith(".summary.json"):
        path = path.with_name(path.name.removesuffix(".summary.json") + ".jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- comparing two runs ----------------------------------------------------------------------


@dataclass
class CaseDelta:
    id: str
    before: float | None  # pass rate over its runs; None: not in that run
    after: float | None
    latency_before: float | None
    latency_after: float | None

    @property
    def status(self) -> str:
        if self.before is None:
            return "new"
        if self.after is None:
            return "removed"
        if self.after > self.before:
            return "fixed"
        if self.after < self.before:
            return "regressed"
        return "pass" if self.after == 1 else "fail" if self.after == 0 else "flaky"


@dataclass
class Comparison:
    cases: list[CaseDelta]
    before: dict[str, Any]  # summaries of the two runs
    after: dict[str, Any]

    def count(self, status: str) -> int:
        return sum(c.status == status for c in self.cases)


def compare(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]) -> Comparison:
    def by_case(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["id"], []).append(row)
        return grouped

    a, b = by_case(before), by_case(after)
    deltas = []
    for case_id in [*a, *(k for k in b if k not in a)]:
        rows_a, rows_b = a.get(case_id), b.get(case_id)
        deltas.append(
            CaseDelta(
                case_id,
                _row_rate(rows_a),
                _row_rate(rows_b),
                _row_latency(rows_a),
                _row_latency(rows_b),
            )
        )
    return Comparison(deltas, _row_summary(before), _row_summary(after))


def _row_rate(rows: list[dict[str, Any]] | None) -> float | None:
    return sum(r["passed"] for r in rows) / len(rows) if rows else None


def _row_latency(rows: list[dict[str, Any]] | None) -> float | None:
    values = [r["latency_s"] for r in rows or [] if not r.get("error")]
    return _median(values)


def _row_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    speeds = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second")]
    return {
        "runs": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else 0.0,
        "latency_mean_s": _mean([r["latency_s"] for r in ok]),
        "latency_median_s": _median([r["latency_s"] for r in ok]),
        "tokens_per_second_mean": round(statistics.fmean(speeds), 1) if speeds else None,
    }
