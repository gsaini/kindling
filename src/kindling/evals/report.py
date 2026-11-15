"""Plain-text tables for the terminal."""

from __future__ import annotations

from typing import Any

from .runner import Comparison, Result


def _num(value: float | None, unit: str = "", digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}{unit}"


def result_line(r: Result) -> str:
    status = "ERROR" if r.error else "PASS" if r.passed else "FAIL"
    run = f"#{r.run}" if r.run > 1 else ""
    speed = f"{r.tokens_per_second:.0f} tok/s" if r.tokens_per_second else ""
    line = f"{status:<5} {r.id + run:<32} {r.latency_s:>7.1f}s {r.output_tokens:>6} tok {speed:>10}"
    return f"{line}  {r.reason[:100]}" if not r.passed else line


def summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "",
        f"passed {summary['passed']}/{summary['runs']} runs ({summary['pass_rate']:.0%})"
        f" over {summary['cases']} cases, {summary['errors']} errors",
        f"latency mean {_num(summary['latency_mean_s'], 's')}, median {_num(summary['latency_median_s'], 's')};"
        f" first token median {_num(summary['first_token_median_s'], 's')};"
        f" {_num(summary['tokens_per_second_mean'])} tok/s",
    ]
    if summary["by_tag"]:
        lines.append("by tag:")
        for tag, row in summary["by_tag"].items():
            lines.append(f"  {tag:<20} {row['passed']:>3}/{row['runs']:<3} {row['pass_rate']:>5.0%}")
    return "\n".join(lines)


def _delta(before: float | None, after: float | None, unit: str = "") -> str:
    if before is None or after is None:
        return f"{_num(before, unit)} -> {_num(after, unit)}"
    return f"{before:.2f}{unit} -> {after:.2f}{unit} ({after - before:+.2f}{unit})"


def comparison_text(comparison: Comparison, name_a: str, name_b: str) -> str:
    lines = [f"A: {name_a}", f"B: {name_b}", ""]
    changed = [c for c in comparison.cases if c.status in ("fixed", "regressed", "new", "removed")]
    for c in changed:
        rate = f"{_num(c.before, digits=2)} -> {_num(c.after, digits=2)}"
        lines.append(
            f"{c.status.upper():<10} {c.id:<32} pass {rate:<14} latency {_delta(c.latency_before, c.latency_after, 's')}"
        )
    if not changed:
        lines.append("no case changed its result")
    a, b = comparison.before, comparison.after
    counts = ", ".join(
        f"{comparison.count(s)} {s}" for s in ("fixed", "regressed", "pass", "fail", "flaky", "new", "removed")
    )
    lines += [
        "",
        counts,
        f"pass rate    {a['pass_rate']:.1%} -> {b['pass_rate']:.1%} ({(b['pass_rate'] - a['pass_rate']) * 100:+.1f} points)",
        f"latency mean   {_delta(a['latency_mean_s'], b['latency_mean_s'], 's')}",
        f"latency median {_delta(a['latency_median_s'], b['latency_median_s'], 's')}",
        f"tokens/s       {_delta(a['tokens_per_second_mean'], b['tokens_per_second_mean'])}",
    ]
    return "\n".join(lines)
