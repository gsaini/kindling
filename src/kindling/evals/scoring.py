"""Checks: does one model answer meet one expectation? Each returns a pass/fail with a reason a
person can read in the results table."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .dataset import Check

if TYPE_CHECKING:
    from .targets import Output


@dataclass
class CheckResult:
    check: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "passed": self.passed, "reason": self.reason}


def score(check: Check, output: Output) -> CheckResult:
    passed, reason = _SCORERS[check.kind](check, output)
    return CheckResult(check.describe(), passed, reason)


# -- text helpers ----------------------------------------------------------------------------

_ANSWER = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
# A number not glued to a digit or point before it (so "2026-11-15" gives 11 and 15, not -11).
_NUMBER = re.compile(r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?<![\d.])-?\.\d+")


def normalise(text: str) -> str:
    """Lower case, single spaces, without the quotes, markdown and full stop models like to add."""
    text = " ".join(text.lower().split())
    return text.strip(" \t\"'`*_").rstrip(".!").strip(" \"'`*_")


def final_number(text: str) -> float | None:
    """The first number after the last `ANSWER:` if there is one, else the last number in the text."""
    answers = _ANSWER.findall(text)
    in_answer = _NUMBER.findall(answers[-1]) if answers else []
    numbers = in_answer[:1] or _NUMBER.findall(text)[-1:]
    return float(numbers[0].replace(",", "")) if numbers else None


def _as_list(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# -- the checks ------------------------------------------------------------------------------


def _contains(check: Check, out: Output) -> tuple[bool, str]:
    text = out.text.lower()
    wanted = _as_list(check.value)
    missing = [w for w in wanted if w.lower() not in text]
    if check.options.get("mode", "all") == "any":
        return (len(missing) < len(wanted), "" if len(missing) < len(wanted) else f"none of {wanted} in the answer")
    return (not missing, f"missing {missing}" if missing else "")


def _not_contains(check: Check, out: Output) -> tuple[bool, str]:
    text = out.text.lower()
    found = [w for w in _as_list(check.value) if w.lower() in text]
    return (not found, f"found {found}" if found else "")


def _exact(check: Check, out: Output) -> tuple[bool, str]:
    got = normalise(out.text)
    ok = got in {normalise(v) for v in _as_list(check.value)}
    return (ok, "" if ok else f"got {_short(got)!r}")


def _regex(check: Check, out: Output) -> tuple[bool, str]:
    flags = 0
    for letter in check.options.get("flags", ""):
        flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}[letter]
    ok = re.search(str(check.value), out.text, flags) is not None
    return (ok, "" if ok else f"no match in {_short(out.text)!r}")


def _number(check: Check, out: Output) -> tuple[bool, str]:
    got = final_number(out.text)
    if got is None:
        return False, "no number in the answer"
    tolerance = float(check.options.get("tolerance", 0))
    ok = abs(got - float(check.value)) <= tolerance + 1e-9
    return (ok, "" if ok else f"got {got:g}, want {check.value:g}" + (f" ± {tolerance:g}" if tolerance else ""))


_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _json(check: Check, out: Output) -> tuple[bool, str]:
    text = out.text.strip()
    if not check.options.get("strict", False):
        fenced = _FENCE.match(text)  # a ```json block around nothing else is fine unless strict
        text = fenced.group(1) if fenced else text
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return False, f"not JSON: {error.msg} at {error.pos}"
    required = check.options.get("required", [])
    if required:
        if not isinstance(value, dict):
            return False, f"want an object with {required}, got {type(value).__name__}"
        missing = [k for k in required if k not in value]
        if missing:
            return False, f"missing keys {missing}"
    if "schema" in check.options:
        problem = schema_problem(value, check.options["schema"])
        if problem:
            return False, problem
    return True, ""


_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def schema_problem(value: Any, schema: dict[str, Any], where: str = "$") -> str:
    """A small JSON Schema subset (type, enum, required, properties, items). '' when it fits."""
    if "type" in schema:
        names = _as_list(schema["type"])
        fits = any(
            isinstance(value, _TYPES[n]) and not (isinstance(value, bool) and n in ("number", "integer"))
            for n in names
            if n in _TYPES
        )
        if not fits:
            return f"{where}: want {'/'.join(names)}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where}: {value!r} is not one of {schema['enum']}"
    if isinstance(value, dict):
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"{where}: missing keys {missing}"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                problem = schema_problem(value[key], sub, f"{where}.{key}")
                if problem:
                    return problem
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            problem = schema_problem(item, schema["items"], f"{where}[{index}]")
            if problem:
                return problem
    return ""


def _same_argument(got: Any, want: Any) -> bool:
    # Strings match ignoring case and spaces, so "6*7" and "6 * 7" are the same calculation.
    if isinstance(want, str) and isinstance(got, str):
        return "".join(got.lower().split()) == "".join(want.lower().split())
    if isinstance(want, int | float) and isinstance(got, int | float | str):
        try:
            return abs(float(got) - float(want)) < 1e-9
        except ValueError:
            return False
    return got == want


def _tool_called(check: Check, out: Output) -> tuple[bool, str]:
    names = _as_list(check.value)
    calls = [c for c in out.tool_calls if c["name"] in names]
    if not calls:
        made = [c["name"] for c in out.tool_calls]
        return False, f"{'/'.join(names)} not called" + (f" (called {made})" if made else " (no tool calls)")
    wanted = check.options.get("arguments") or {}
    for call in calls:
        args = call.get("arguments") or {}
        if all(k in args and _same_argument(args[k], v) for k, v in wanted.items()):
            return True, ""
    return False, f"called with {_short(json.dumps(calls[0].get('arguments')))}, want {json.dumps(wanted)}"


def _tool_not_called(check: Check, out: Output) -> tuple[bool, str]:
    names = _as_list(check.value)
    hits = [c["name"] for c in out.tool_calls if c["name"] in names]
    return (not hits, f"called {hits}" if hits else "")


def _no_tool_calls(check: Check, out: Output) -> tuple[bool, str]:
    made = [c["name"] for c in out.tool_calls]
    return (not made, f"called {made}" if made else "")


def _max_latency(check: Check, out: Output) -> tuple[bool, str]:
    ok = out.latency_s <= float(check.value)
    return (ok, "" if ok else f"took {out.latency_s:.1f}s")


def _max_output_tokens(check: Check, out: Output) -> tuple[bool, str]:
    ok = out.output_tokens <= int(check.value)
    return (ok, "" if ok else f"wrote {out.output_tokens} tokens")


_SCORERS = {
    "contains": _contains,
    "not_contains": _not_contains,
    "exact": _exact,
    "regex": _regex,
    "number": _number,
    "json": _json,
    "tool_called": _tool_called,
    "tool_not_called": _tool_not_called,
    "no_tool_calls": _no_tool_calls,
    "max_latency_s": _max_latency,
    "max_output_tokens": _max_output_tokens,
}
