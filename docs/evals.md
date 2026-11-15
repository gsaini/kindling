# Evals

Kindling's eval harness runs a dataset of prompts against a model or a server, checks each answer
against objective expectations, and saves the results so two runs can be compared case by case.

## Why

Every change worth making to Kindling changes what the model does: a new model version, reasoning on
or off, a different system prompt, a new tool, a bigger context window, a different server. Without
a fixed set of cases you are left with "it felt better in chat". With one you get numbers to compare:
which cases now pass, which broke, and how much slower or faster it got.

The checks are deliberately objective (a number, a word, valid JSON, the right tool called), so the
same run always scores the same way and no second model is needed to judge.

## Running

```sh
kindling eval list                                   # the datasets in evals/datasets
kindling eval run smoke                              # the in-process target, default model
kindling eval run smoke --label baseline --min-pass 0.8
kindling eval run reasoning --think off --label think-off
kindling eval run tools --filter fixtures --repeat 3 # only cases tagged "fixtures", 3 runs each
```

A run prints one line per case (PASS / FAIL / ERROR, latency, output tokens, tokens/s and why it
failed), then a summary: the pass rate, mean and median latency, median time to first token, mean
tokens/s, and the pass rate by tag. It writes two files to `KINDLING_EVAL_RESULTS_DIR`
(`evals/results/` by default, ignored by git):

- `<timestamp>-<label>.jsonl`: one line per case run, with the id, tags, pass/fail, each check and
  its reason, the answer, the length of the reasoning, the tool calls (with arguments and output),
  prompt and output tokens, latency, time to first token, tokens/s, steps, model, think and error.
- `<timestamp>-<label>.summary.json`: the summary, plus what was run (dataset, target, URL, model,
  think, filters, skipped cases, times, Kindling version).

API keys are sent in request headers only. They never appear in the results, the summary or the
printed output.

### Targets

| `--target` | What it calls | What it measures |
|---|---|---|
| `inproc` (default) | the models in `models.toml` and the built-in tools, through `run_agent` in this process | the model plus tools, with no HTTP |
| `http` | a running Kindling's `POST /api/chat` (`--url`, default `http://KINDLING_HOST:KINDLING_PORT`) | the whole stack: auth, queueing, streaming, the server's tools |
| `openai` | any OpenAI-compatible `/v1/chat/completions` (`--url`, default Ollama's `http://localhost:11434/v1`) | any server: Kindling's `/v1`, Ollama, vLLM, NIM |

- `http` and `openai` take a key from `--api-key` or the `KINDLING_EVAL_API_KEY` environment
  variable. Prefer the variable, so the key stays out of your shell history. Issue a key just for evals.
- `openai` needs `--model` (the model name on that server, e.g. `nemotron-3-nano:30b` for Ollama).
  It runs the tool loop here, offering the built-in tools to the model and running the calls it asks
  for, so tool cases mean the same thing as in-process. If a server runs its own tools, the harness
  sees only the final answer and `tool_called` checks will fail.
- Reasoning on `openai` is switched with both Ollama's `think` field and vLLM's
  `chat_template_kwargs.enable_thinking` (which Nemotron's chat template reads). Reasoning text is
  read from `reasoning_content`, `reasoning` or `thinking`.
- The file tools read `evals/fixtures/` on `inproc` and `openai` (change it with `--workspace`).
  On `http` they read the server's workspace, so cases tagged `fixtures` are skipped there.

### Which settings win

- `--model` and `--think on|off` override every case, so one dataset can be pointed at any model or
  run both ways. With `--think default`, a case's own `think` is used, and otherwise the model's
  default (the `think` in `models.toml`, or the model's own).
- `--repeat N` multiplies each case's own `repeat`. Models sample, so repeats show which cases are
  flaky; the pass rate counts every run.
- `--concurrency` runs that many cases at once. Keep it at 1 for latency numbers on one Mac: Ollama
  serves one request at a time unless `OLLAMA_NUM_PARALLEL` says otherwise, so concurrent cases queue
  and their latency includes the wait.

## Writing cases

A dataset is a JSONL file in `evals/datasets/`, one case per line. Lines starting with `#` are
comments.

```json
{"id": "capital-australia", "tags": ["facts"], "prompt": "What is the capital city of Australia?", "expect": [{"contains": "canberra"}]}
```

| Field | Meaning |
|---|---|
| `id` | unique in the file; used to match cases across runs |
| `prompt` or `messages` | a single user message, or a whole conversation (`role`: system / user / assistant) ending with the user |
| `system` | a system prompt, put first unless `messages` already has one |
| `think` | `true` / `false`; leave it out to follow `--think` and then the model's default |
| `tools` | `false` to offer no tools (default `true`) |
| `model` | a model id for this case (overridden by `--model`) |
| `tags` | for `--filter` and the by-tag summary |
| `repeat` | runs of this case (default 1) |
| `notes` | for people reading the file; ignored |
| `expect` | the checks; the case passes when all of them pass |

### Checks

| Check | Passes when |
|---|---|
| `{"contains": "x"}` / `{"contains": ["x", "y"], "mode": "any"}` | the answer contains all (or any) of the strings, ignoring case |
| `{"not_contains": ["x"]}` | none of them appear |
| `{"exact": "blue"}` / `{"exact": ["yes", "y"]}` | the whole answer equals one of them, ignoring case, extra spaces, quotes, markdown and a final full stop |
| `{"regex": "\\b20\\d\\d\\b", "flags": "i"}` | `re.search` finds it (flags from `imsx`) |
| `{"number": 0.05, "tolerance": 0.001}` | the first number after the last `ANSWER:`, or else the last number in the answer, is within the tolerance |
| `{"json": true, "required": ["k"], "schema": {...}, "strict": false}` | the answer is JSON (a lone `` ```json `` block is fine unless strict), has the keys, and fits the schema subset: `type`, `enum`, `required`, `properties`, `items` |
| `{"tool_called": "calculate", "arguments": {"expression": "6*7"}}` | that tool (or any of a list) was called, with at least these arguments; strings compare ignoring case and spaces |
| `{"tool_not_called": ["read_file"]}` | none of these tools were called |
| `{"no_tool_calls": true}` | no tool was called at all |
| `{"max_latency_s": 30}` | the whole case took at most this long |
| `{"max_output_tokens": 2000}` | it wrote at most this many tokens (reasoning included) |

Checks look at the answer only, never the reasoning. A dataset with a mistake (an unknown check, a
typo in an option, a regex that doesn't compile, a duplicate id) is rejected when it loads, with the
file and line.

Tips:

- Ask for a final `ANSWER: <x>` line when the answer is a number. Models show their working, and the
  last number in the working is often not the answer.
- Keep expectations objective. If you can't write a check a script can decide, the case belongs in
  the LLM-as-judge set (see the roadmap), not here.
- Tag cases by what they exercise (`tools`, `reasoning`, `format`...), so the by-tag summary shows
  where a change helped or hurt.
- Put files the tools should read in `evals/fixtures/` and tag those cases `fixtures`.

### The starter datasets

- `smoke.jsonl` (13 cases): the calculator, the clock, listing and reading fixture files, the
  bat-and-ball trick, multi-step arithmetic, one-word answers, JSON-only output, a fact, and a
  one-sentence summary. A quick check that nothing is broken.
- `reasoning.jsonl` (8 puzzles, tools off): meant to be run with `--think on` and `--think off`.
- `tools.jsonl` (10 cases): choosing the right tool, choosing no tool, chaining file reading with the
  calculator, and refusing to read outside the workspace.

## Comparing runs

```sh
kindling eval compare evals/results/20261115-101500-baseline.jsonl evals/results/20261115-103000-new-prompt.jsonl
```

It lists the cases that were fixed, regressed, added or removed (with their pass rates over repeats
and their median latency before and after), counts the unchanged ones, and prints the change in pass
rate, mean and median latency and tokens/s. A `.summary.json` works in place of its `.jsonl`.
`--fail-on-regression` exits 1 if any case got worse.

Some comparisons worth making:

- **Reasoning on versus off.** `kindling eval run reasoning --think off --label think-off`, then
  `--think on --label think-on`, then compare. Expect reasoning to fix the trick questions and cost
  latency and output tokens; the comparison says how many and how much.
- **Nano versus Super on another machine.** On the Mac: `kindling eval run smoke --label nano`. On the
  bigger machine (Ollama, or vLLM with the Nemotron 3 Super checkpoint), either run the in-process
  target there with its own `models.toml`, or from the Mac point the OpenAI target at it:
  `kindling eval run smoke --target openai --url http://gpu-box:8000/v1 --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --label super`.
  Then compare the two files. Kindling's own `/v1` works as the URL too, with `--model nemotron`.
- **In-process versus HTTP overhead.** Run the same dataset with `--target inproc` and with
  `--target http --url http://127.0.0.1:8000` against the same model. Pass rates should match (the
  same model and tools, apart from sampling); the latency difference is what auth, the queue,
  streaming and the network add.
- **Ollama directly versus through Kindling.** `--target openai --url http://localhost:11434/v1`
  against `--target openai --url http://127.0.0.1:8000/v1`.

The first case of a run also pays for loading the model if it isn't in memory (minutes for a 30B
model), which skews its latency and the mean. Load it first (`kindling check`, or one throwaway run)
when latency matters; the median is less affected.

Use `--repeat 3` or more before believing a difference of one or two cases: the model samples, and
a single run of a small dataset is noisy.

## In CI

The harness is a plain command with an exit code:

- `kindling eval run <dataset> --min-pass 0.8` exits 1 below that pass rate (and 2 for a broken dataset).
- `kindling eval compare <baseline> <new> --fail-on-regression` exits 1 if any case regressed.

The repository's tests (`uv run pytest`) cover the harness itself with a scripted `FakeModel` and no
network, so they belong in every CI run. A real-model eval needs a machine with the model, so run it
where the model lives: a self-hosted runner on the Mac or the GPU box, a nightly job, or by hand
before merging a change to prompts, tools or models. For example:

```sh
export KINDLING_EVAL_API_KEY=...   # a key issued for evals, from the CI secret store
kindling eval run smoke --target http --url "$KINDLING_URL" --label "ci-$GIT_SHA" --min-pass 0.8
```

Keep a baseline results file from the main branch, and compare each run against it, so a drop in one
area isn't hidden by a gain in another.

## Roadmap

- **LLM-as-judge.** A `judge` check that asks a second model (Super, or another model behind the
  OpenAI target) to grade an answer against a rubric, for things a regex can't decide: summary
  quality, tone, helpfulness. Record the judge's model and prompt in the results, and calibrate it
  against a few hand-graded cases first.
- **Agent trajectories and harness evals.** Checks on the whole sequence of steps, not just the end:
  the order of tool calls, the number of steps, wasted or repeated calls, recovering from a tool
  error, staying within `max_tool_steps`. The results already record every call with its arguments
  and output, so this is a matter of new checks and multi-turn cases with scripted tool failures.
- **Memory evals.** When Kindling remembers across conversations: store a fact in one saved
  conversation, ask about it in another, and check it's recalled (and not leaked to another owner).
  Needs cases that span several requests against the `http` target.
- **Long-context needle tests.** Bury a fact at different depths in documents of growing length (up
  to `num_ctx`), ask for it, and chart accuracy and prefill speed against length and depth. Generate
  the cases rather than writing them, and watch for Ollama truncating silently past its window.
- Also worth adding: per-case cost and memory figures from the server's metrics, a tiny HTML report
  of a comparison, and datasets for refusals and prompt injection through tool output.
