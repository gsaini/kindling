"""The eval harness: datasets of cases with objective checks, run against a target, scored and saved.

    dataset   JSONL cases and their checks (kindling.evals.dataset)
    scoring   the checks themselves
    targets   inproc (run_agent here), http (a Kindling server), openai (any /v1 server)
    runner    running, summarising, saving and comparing runs
    cli       `kindling eval run | compare | list`

See docs/evals.md.
"""

from .dataset import Case, Check, DatasetError, load_dataset
from .runner import Result, RunOptions, compare, run_eval, summarise
from .scoring import CheckResult, score
from .targets import HttpTarget, OpenAIChatModel, Output, inproc_target, openai_target

__all__ = [
    "Case",
    "Check",
    "CheckResult",
    "DatasetError",
    "HttpTarget",
    "OpenAIChatModel",
    "Output",
    "Result",
    "RunOptions",
    "compare",
    "inproc_target",
    "load_dataset",
    "openai_target",
    "run_eval",
    "score",
    "summarise",
]
