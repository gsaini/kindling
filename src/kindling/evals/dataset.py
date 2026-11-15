r"""Eval datasets: JSONL files, one case per line.

    {"id": "capital-au", "prompt": "What is the capital of Australia?", "tags": ["facts"],
     "expect": [{"contains": "canberra"}]}

A check is an object with one check name as a key and, for some checks, a few options:

    {"contains": ["a", "b"], "mode": "any"}      case-insensitive substrings; mode "all" (default) or "any"
    {"not_contains": ["sydney"]}
    {"exact": "blue"}                            the whole answer, normalised; a list means any of them
    {"regex": "\\b20\\d\\d\\b", "flags": "i"}   re.search on the answer; flags from "imsx"
    {"number": 0.05, "tolerance": 0.001}         the ANSWER: value, or else the last number in the answer
    {"json": true, "required": ["k"], "schema": {...}, "strict": false}
    {"tool_called": "calculate", "arguments": {"expression": "6*7"}}
    {"tool_not_called": "calculate"}
    {"no_tool_calls": true}
    {"max_latency_s": 30}
    {"max_output_tokens": 2000}

Blank lines and lines starting with `#` are skipped, so a dataset can carry comments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Each check name and the extra keys it may carry.
CHECK_OPTIONS: dict[str, set[str]] = {
    "contains": {"mode"},
    "not_contains": set(),
    "exact": set(),
    "regex": {"flags"},
    "number": {"tolerance"},
    "json": {"required", "schema", "strict"},
    "tool_called": {"arguments"},
    "tool_not_called": set(),
    "no_tool_calls": set(),
    "max_latency_s": set(),
    "max_output_tokens": set(),
}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Check:
    kind: str
    value: Any
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any) -> Check:
        if not isinstance(raw, dict):
            raise ValueError(f"a check must be an object, not {raw!r}")
        kinds = [k for k in raw if k in CHECK_OPTIONS]
        if len(kinds) != 1:
            raise ValueError(f"a check needs exactly one of {', '.join(CHECK_OPTIONS)}; got {sorted(raw)}")
        kind = kinds[0]
        options = {k: v for k, v in raw.items() if k != kind}
        unknown = set(options) - CHECK_OPTIONS[kind]
        if unknown:
            raise ValueError(f"{kind}: unknown option(s) {', '.join(sorted(unknown))}")
        if kind == "contains" and options.get("mode", "all") not in ("all", "any"):
            raise ValueError("contains: mode must be 'all' or 'any'")
        if kind == "regex":
            if set(options.get("flags", "")) - set("imsx"):
                raise ValueError("regex: flags may only use the letters i, m, s, x")
            try:
                re.compile(str(raw[kind]))
            except re.error as error:
                raise ValueError(f"regex: {error}") from None
        if kind == "number" and not isinstance(raw[kind], int | float):
            raise ValueError("number: the expected value must be a number")
        return cls(kind, raw[kind], options)

    def describe(self) -> str:
        value = self.value if self.value is not True else ""
        text = f"{self.kind} {json.dumps(value)}" if value != "" else self.kind
        return text if len(text) <= 60 else text[:57] + "..."


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    system: str | None = None
    think: bool | None = None  # None: whatever the run (or the model) says
    tools: bool = True
    model: str | None = None
    tags: list[str] = Field(default_factory=list)
    repeat: int = Field(default=1, ge=1)
    notes: str = ""  # for the reader of the dataset; ignored by the runner
    expect: list[Check] = Field(default_factory=list)

    @field_validator("expect", mode="before")
    @classmethod
    def parse_checks(cls, raw: Any) -> list[Check]:
        if not isinstance(raw, list):
            raise ValueError("expect must be a list of checks")
        return [Check.parse(item) for item in raw]

    @model_validator(mode="after")
    def one_input(self) -> Case:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("give either prompt or messages")
        if self.messages is not None:
            for message in self.messages:
                if message.get("role") not in ("system", "user", "assistant") or not isinstance(
                    message.get("content"), str
                ):
                    raise ValueError("each message needs a role (system/user/assistant) and string content")
            if not self.messages or self.messages[-1]["role"] != "user":
                raise ValueError("the last message must come from the user")
        return self

    def chat_messages(self) -> list[dict[str, Any]]:
        """The conversation to send: the system prompt, then the prompt or the given messages."""
        messages = [dict(m) for m in self.messages] if self.messages else [{"role": "user", "content": self.prompt}]
        if self.system and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": self.system})
        return messages


def load_dataset(path: Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = Case.model_validate(json.loads(line))
        except json.JSONDecodeError as error:
            raise DatasetError(f"{path}:{number}: not valid JSON: {error.msg}") from None
        except ValidationError as error:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'case'}: {e['msg']}" for e in error.errors())
            raise DatasetError(f"{path}:{number}: {problems}") from None
        if case.id in seen:
            raise DatasetError(f"{path}:{number}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


# -- where datasets live ---------------------------------------------------------------------


def evals_dir() -> Path:
    """`evals/` under the current folder, or else the one in the source checkout."""
    here = Path("evals")
    if (here / "datasets").is_dir():
        return here
    return Path(__file__).resolve().parents[3] / "evals"


def find_datasets(folder: Path | None = None) -> list[Path]:
    folder = folder or evals_dir() / "datasets"
    return sorted(folder.glob("*.jsonl")) if folder.is_dir() else []


def resolve_dataset(name: str) -> Path:
    """A path, or a bare name like "smoke" for evals/datasets/smoke.jsonl."""
    path = Path(name)
    if path.is_file():
        return path
    candidate = evals_dir() / "datasets" / (name if name.endswith(".jsonl") else f"{name}.jsonl")
    if candidate.is_file():
        return candidate
    raise DatasetError(f"no dataset {name!r} (not a file, and not in {evals_dir() / 'datasets'})")
