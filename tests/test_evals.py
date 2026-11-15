import json
from pathlib import Path

import anyio
import httpx
import pytest

from kindling.evals import cli
from kindling.evals.dataset import Case
from kindling.evals.runner import RunOptions, compare, load_results, run_eval, select, summarise
from kindling.evals.targets import HttpTarget, inproc_target, openai_target, to_openai_messages
from kindling.llm import FakeModel, ModelRegistry, ToolCall, Turn, Usage

KEY = "test-admin-key"


def case(**fields) -> Case:
    return Case.model_validate({"prompt": "q", **fields})


CALC = case(
    id="calc", tags=["tools"], prompt="6*7?", expect=[{"tool_called": "calculate"}, {"number": 42}, {"contains": "42"}]
)
CAPITAL = case(id="capital", tags=["facts"], expect=[{"contains": "canberra"}])


def calc_turns() -> list[Turn]:
    return [
        Turn(thinking="Use the tool.", tool_calls=[ToolCall("calculate", {"expression": "6*7"})]),
        lambda messages, kwargs: Turn(content=f"It is {messages[-1]['content']}.", usage=Usage(30, 8, 0.1, 0.5)),
    ]


def dataset_file(tmp_path: Path) -> Path:
    lines = [
        {"id": "calc", "tags": ["tools"], "prompt": "6*7?", "expect": [{"tool_called": "calculate"}, {"number": 42}]},
        {"id": "capital", "tags": ["facts"], "prompt": "Capital of Australia?", "expect": [{"contains": "canberra"}]},
    ]
    path = tmp_path / "mini.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


# -- inproc ----------------------------------------------------------------------------------


def test_inproc_runs_the_agent_with_tools_and_scores_it(tmp_path):
    model = FakeModel(*calc_turns(), Turn(content="Sydney"))
    target = inproc_target(ModelRegistry.single(model), workspace=tmp_path)
    results = anyio.run(run_eval, [CALC, CAPITAL], target, RunOptions(think=False))
    calc, capital = results
    assert calc.passed and calc.output == "It is 42." and calc.thinking_chars == len("Use the tool.")
    assert calc.tool_calls == [{"name": "calculate", "arguments": {"expression": "6*7"}, "output": "42", "ok": True}]
    assert calc.steps == 2 and calc.output_tokens == 13 and calc.prompt_tokens == 30 and calc.model == "fake"
    assert calc.tokens_per_second == 13.0 and calc.first_token_s is not None
    assert not capital.passed and capital.reason == "contains \"canberra\": missing ['canberra']"
    assert all(call["think"] is False for call in model.calls)

    summary = summarise(results, label="x")
    assert summary["pass_rate"] == 0.5 and summary["runs"] == 2 and summary["label"] == "x"
    assert summary["by_tag"] == {
        "facts": {"runs": 1, "passed": 0, "pass_rate": 0.0},
        "tools": {"runs": 1, "passed": 1, "pass_rate": 1.0},
    }


def test_inproc_case_settings_repeat_and_errors(tmp_path):
    no_tools = case(id="plain", tools=False, think=True, repeat=2, system="be brief", expect=[{"exact": "ok"}])
    model = FakeModel(Turn(content="OK"), Turn(content="ok."))
    results = anyio.run(
        run_eval, [no_tools], inproc_target(ModelRegistry.single(model), workspace=tmp_path), RunOptions()
    )
    assert [r.run for r in results] == [1, 2] and all(r.passed for r in results)
    assert model.calls[0]["tools"] is None and model.calls[0]["think"] is True
    assert model.calls[0]["messages"][0] == {"role": "system", "content": "be brief"}

    # A model that fails is a failed case with the error, not a crashed run.
    broken = inproc_target(ModelRegistry.single(FakeModel()), workspace=tmp_path)
    [result] = anyio.run(run_eval, [CAPITAL], broken, RunOptions())
    assert not result.passed and "ran out of scripted turns" in result.error and result.checks == []
    [unknown] = anyio.run(run_eval, [CAPITAL], broken, RunOptions(model="no-such-model"))
    assert "UnknownModel" in unknown.error


def test_filters_and_skipped_tags(make_client, tmp_path):
    fixture_case = case(id="files", tags=["fixtures"])
    inproc = inproc_target(ModelRegistry.single(FakeModel()), workspace=tmp_path)
    assert select([CALC, CAPITAL, fixture_case], inproc, ["facts", "calc"]) == ([CALC, CAPITAL], [])
    http = HttpTarget("http://testserver", KEY, client=make_client())
    assert select([CAPITAL, fixture_case], http, []) == ([CAPITAL], [fixture_case])


# -- http ------------------------------------------------------------------------------------


def test_http_target_runs_the_whole_stack(make_client):
    model = FakeModel(*calc_turns())
    target = HttpTarget("http://testserver", KEY, client=make_client(model))
    [result] = anyio.run(run_eval, [CALC], target, RunOptions(think=True))
    assert result.passed, result.reason
    assert result.output == "It is 42." and result.tool_calls[0]["output"] == "42"
    assert result.steps == 2 and result.model == "fake" and result.output_tokens == 13
    assert model.calls[0]["think"] is True


def test_http_target_reports_a_bad_key_without_leaking_it(make_client):
    target = HttpTarget("http://testserver", "wrong-secret-key", client=make_client())
    [result] = anyio.run(run_eval, [CAPITAL], target, RunOptions())
    assert not result.passed and result.error.startswith("RuntimeError: HTTP 401")
    assert "wrong-secret-key" not in json.dumps(result.to_dict()) and "wrong-secret-key" not in str(target.describe())


# -- openai ----------------------------------------------------------------------------------


def sse(*chunks: dict) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def test_openai_target_streams_and_runs_tools_here(tmp_path):
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"body": body, "auth": request.headers.get("authorization"), "url": str(request.url)})
        if len(requests) == 1:
            reply = sse(
                {"choices": [{"delta": {"reasoning_content": "Use the calculator."}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "x", "function": {"name": "calculate"}}]}}]},
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"expression": "6*7"}'}}]}}
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 6}},
            )
        else:
            reply = sse(
                {"choices": [{"delta": {"content": "It is "}}]},
                {"choices": [{"delta": {"content": "42."}}]},
                {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 4}},
            )
        return httpx.Response(200, content=reply, headers={"content-type": "text/event-stream"})

    target = openai_target(
        "http://gpu-box:8000/v1/", api_key="sk-secret", workspace=tmp_path, transport=httpx.MockTransport(handler)
    )
    [result] = anyio.run(run_eval, [CALC], target, RunOptions(model="nemotron-super", think=False))
    assert result.passed, result.reason
    assert result.thinking_chars == len("Use the calculator.") and result.tool_calls[0]["output"] == "42"
    assert result.prompt_tokens == 40 and result.output_tokens == 10 and result.steps == 2
    first, second = requests
    assert first["url"] == "http://gpu-box:8000/v1/chat/completions" and first["auth"] == "Bearer sk-secret"
    assert first["body"]["model"] == "nemotron-super" and first["body"]["stream"] is True
    assert first["body"]["chat_template_kwargs"] == {"enable_thinking": False} and first["body"]["tools"]
    # The tool result goes back with the id of the call it answers.
    call = second["body"]["messages"][1]["tool_calls"][0]
    assert call["function"] == {"name": "calculate", "arguments": '{"expression": "6*7"}'}
    assert second["body"]["messages"][2] == {"role": "tool", "tool_call_id": call["id"], "content": "42"}
    assert "sk-secret" not in json.dumps(result.to_dict()) + json.dumps(target.describe())


def test_openai_target_errors(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="model not found"))
    target = openai_target("http://x/v1", workspace=tmp_path, transport=transport)
    [result] = anyio.run(run_eval, [CAPITAL], target, RunOptions(model="nope"))
    assert result.error == "RuntimeError: HTTP 404: model not found"
    [result] = anyio.run(run_eval, [CAPITAL], target, RunOptions())
    assert "needs a model name" in result.error


def test_openai_message_conversion_pairs_calls_and_results():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "thinking": "hmm", "tool_calls": [
            {"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {"x": 1}}}]},
        {"role": "tool", "tool_name": "a", "content": "1"},
        {"role": "tool", "tool_name": "b", "content": "2"},
    ]  # fmt: skip
    converted = to_openai_messages(history)
    assert converted[1]["content"] is None and "thinking" not in converted[1]
    ids = [c["id"] for c in converted[1]["tool_calls"]]
    assert [m["tool_call_id"] for m in converted[2:]] == ids and len(set(ids)) == 2


# -- compare ---------------------------------------------------------------------------------


def rows(*items: tuple[str, bool, float]) -> list[dict]:
    return [{"id": i, "passed": p, "latency_s": s, "tokens_per_second": 10.0, "error": None} for i, p, s in items]


def test_compare_finds_fixed_and_regressed_cases():
    before = rows(("a", False, 2.0), ("b", True, 1.0), ("c", True, 1.0), ("d", True, 1.0), ("d", False, 1.0))
    after = rows(("a", True, 1.0), ("b", False, 3.0), ("c", True, 1.0), ("d", True, 1.0), ("e", True, 1.0))
    comparison = compare(before, after)
    assert {c.id: c.status for c in comparison.cases} == {
        "a": "fixed",
        "b": "regressed",
        "c": "pass",
        "d": "fixed",  # 1 of 2 runs passed before, 1 of 1 now
        "e": "new",
    }
    assert comparison.before["pass_rate"] == 0.6 and comparison.after["pass_rate"] == 0.8
    b = next(c for c in comparison.cases if c.id == "b")
    assert (b.latency_before, b.latency_after) == (1.0, 3.0)


# -- the CLI ---------------------------------------------------------------------------------


def test_cli_run_writes_results_and_gates_on_min_pass(tmp_path, settings, capsys):
    dataset = dataset_file(tmp_path)
    registry = ModelRegistry.single(FakeModel(*calc_turns(), Turn(content="Canberra.")))
    code = cli.main(
        ["run", str(dataset), "--label", "base line", "--min-pass", "1"], settings=settings, registry=registry
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "PASS  calc" in out and "passed 2/2 runs (100%)" in out and "tools" in out
    [results_path] = settings.eval_results_dir.glob("*-base-line.jsonl")
    [summary_path] = settings.eval_results_dir.glob("*-base-line.summary.json")
    assert [r["id"] for r in load_results(results_path)] == ["calc", "capital"]
    summary = json.loads(summary_path.read_text())
    assert summary["target"] == "inproc" and summary["pass_rate"] == 1.0 and summary["dataset"] == str(dataset)

    registry = ModelRegistry.single(FakeModel(*calc_turns(), Turn(content="Sydney.")))
    code = cli.main(
        ["run", str(dataset), "--label", "worse", "--min-pass", "0.8"], settings=settings, registry=registry
    )
    assert code == 1 and "below --min-pass" in capsys.readouterr().out

    [worse] = settings.eval_results_dir.glob("*-worse.jsonl")
    assert cli.main(["compare", str(results_path), str(worse)]) == 0
    out = capsys.readouterr().out
    assert "REGRESSED  capital" in out and "pass rate    100.0% -> 50.0% (-50.0 points)" in out
    assert cli.main(["compare", str(summary_path), str(worse), "--fail-on-regression"]) == 1


def test_cli_filter_repeat_and_bad_input(tmp_path, settings, capsys):
    dataset = dataset_file(tmp_path)
    registry = ModelRegistry.single(FakeModel(Turn(content="Canberra"), Turn(content="canberra")))
    assert (
        cli.main(["run", str(dataset), "--filter", "facts", "--repeat", "2"], settings=settings, registry=registry) == 0
    )
    assert "passed 2/2 runs" in capsys.readouterr().out
    assert cli.main(["run", "no-such-dataset"], settings=settings) == 2
    assert cli.main(["run", str(dataset), "--filter", "nothing"], settings=settings, registry=registry) == 2
    assert cli.main(["run", str(dataset), "--target", "openai"], settings=settings) == 2  # needs --model
    capsys.readouterr()


def test_cli_http_target_never_stores_the_key(tmp_path, settings, make_client, monkeypatch, capsys):
    dataset = dataset_file(tmp_path)
    client = make_client(FakeModel(*calc_turns(), Turn(content="Canberra")))
    monkeypatch.setenv(cli.API_KEY_ENV, KEY)
    code = cli.main(["run", str(dataset), "--target", "http", "--label", "h"], settings=settings, http_client=client)
    assert code == 0, capsys.readouterr().out
    for path in settings.eval_results_dir.iterdir():
        assert KEY not in path.read_text()
    assert KEY not in capsys.readouterr().out


def test_cli_list_shows_the_starter_datasets(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "smoke" in out and "reasoning" in out and "tools" in out


def test_kindling_cli_wires_the_eval_commands(capsys):
    from kindling.cli import main

    with pytest.raises(SystemExit):
        main(["eval", "--help"])
    assert "compare" in capsys.readouterr().out
