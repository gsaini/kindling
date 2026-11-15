import pytest

from kindling.evals.dataset import Case, Check, DatasetError, find_datasets, load_dataset
from kindling.evals.scoring import final_number, normalise, schema_problem, score
from kindling.evals.targets import Output


def check(raw: dict, **output) -> tuple[bool, str]:
    result = score(Check.parse(raw), Output(**output))
    return result.passed, result.reason


def test_contains_is_case_insensitive_with_all_or_any():
    assert check({"contains": "canberra"}, text="The capital is Canberra.")[0]
    assert check({"contains": ["a", "Z"]}, text="a b") == (False, "missing ['Z']")
    assert check({"contains": ["x", "B"], "mode": "any"}, text="a b")[0]
    assert not check({"contains": ["x", "y"], "mode": "any"}, text="a b")[0]
    assert check({"not_contains": ["sydney"]}, text="Canberra")[0]
    assert check({"not_contains": "Sydney"}, text="not sydney") == (False, "found ['Sydney']")


def test_exact_ignores_case_spacing_quotes_and_a_full_stop():
    assert normalise('  **"Blue."**  ') == "blue"
    assert check({"exact": "blue"}, text="Blue.")[0]
    assert check({"exact": ["yes", "y"]}, text="Y")[0]
    assert check({"exact": "blue"}, text="It is blue.") == (False, "got 'it is blue'")


def test_regex_with_flags():
    assert check({"regex": "ANSWER:\\W*bob\\b", "flags": "i"}, text="so... answer: **Bob**")[0]
    assert not check({"regex": "^bob$"}, text="Bob")[0]
    with pytest.raises(ValueError, match="flags"):
        Check.parse({"regex": "x", "flags": "q"})
    with pytest.raises(ValueError, match="regex"):
        Check.parse({"regex": "("})


def test_number_prefers_the_answer_line_then_the_last_number():
    assert final_number("5 cents is wrong... ANSWER: $0.05") == 0.05
    assert final_number("ANSWER: 1,234.5 (from 3 steps)") == 1234.5
    assert final_number("It took 3 tries and the total is 14,272,771.") == 14272771
    assert final_number("on 2026-11-15") == 15  # a date's dashes aren't minus signs
    assert final_number("it fell to -4 degrees") == -4
    assert final_number("no digits here") is None
    assert check({"number": 0.05, "tolerance": 0.001}, text="ANSWER: 0.0500")[0]
    assert check({"number": 17}, text="ANSWER: 16") == (False, "got 16, want 17")
    assert check({"number": 1}, text="none") == (False, "no number in the answer")


def test_json_with_required_keys_and_a_schema():
    schema = {"type": "object", "properties": {"year": {"type": "integer"}, "tags": {"items": {"type": "string"}}}}
    raw = {"json": True, "required": ["title", "year"], "schema": schema}
    assert check(raw, text='{"title": "P&P", "year": 1813, "tags": ["a"]}')[0]
    assert check(raw, text='```json\n{"title": "P&P", "year": 1813}\n```')[0]
    assert not check({**raw, "strict": True}, text='```json\n{"title": "P&P", "year": 1813}\n```')[0]
    assert check(raw, text='{"title": "P&P"}') == (False, "missing keys ['year']")
    assert check(raw, text='{"title": "P&P", "year": "1813"}') == (False, "$.year: want integer, got str")
    assert check(raw, text='{"title": "x", "year": 1, "tags": [1]}')[1] == "$.tags[0]: want string, got int"
    assert check(raw, text="Here you go: {...}")[1].startswith("not JSON")
    assert schema_problem(True, {"type": "integer"}) == "$: want integer, got bool"
    assert schema_problem("b", {"enum": ["a"]}) == "$: 'b' is not one of ['a']"


def test_tool_checks():
    calls = [{"name": "calculate", "arguments": {"expression": "6 * 7"}, "output": "42", "ok": True}]
    assert check({"tool_called": "calculate"}, tool_calls=calls)[0]
    assert check({"tool_called": "calculate", "arguments": {"expression": "6*7"}}, tool_calls=calls)[0]
    assert not check({"tool_called": "calculate", "arguments": {"expression": "6*8"}}, tool_calls=calls)[0]
    assert check({"tool_called": ["read_file", "list_files"]}, tool_calls=calls)[1].endswith("(called ['calculate'])")
    assert check({"tool_called": "calculate"}) == (False, "calculate not called (no tool calls)")
    assert check({"tool_not_called": "read_file"}, tool_calls=calls)[0]
    assert not check({"tool_not_called": ["calculate"]}, tool_calls=calls)[0]
    assert check({"no_tool_calls": True})[0] and not check({"no_tool_calls": True}, tool_calls=calls)[0]


def test_budget_checks():
    assert check({"max_latency_s": 2}, latency_s=1.5)[0]
    assert check({"max_latency_s": 1}, latency_s=1.5) == (False, "took 1.5s")
    assert check({"max_output_tokens": 10}, output_tokens=11) == (False, "wrote 11 tokens")


def test_bad_checks_are_caught_when_the_dataset_loads(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        Check.parse({"contains": "a", "exact": "b"})
    with pytest.raises(ValueError, match="unknown option"):
        Check.parse({"contains": "a", "tolerance": 1})
    path = tmp_path / "bad.jsonl"
    path.write_text('# a comment\n{"id": "a", "prompt": "x"}\n{"id": "b", "prompt": "x", "expect": [{"contans": 1}]}\n')
    with pytest.raises(DatasetError, match=r"bad.jsonl:3: expect"):
        load_dataset(path)
    path.write_text('{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n')
    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(path)
    path.write_text('{"id": "a"}\n')
    with pytest.raises(DatasetError, match="prompt or messages"):
        load_dataset(path)


def test_case_messages_get_the_system_prompt():
    case = Case.model_validate({"id": "a", "prompt": "hi", "system": "be brief"})
    assert case.chat_messages() == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="last message"):
        Case.model_validate({"id": "a", "messages": [{"role": "assistant", "content": "x"}]})


def test_the_starter_datasets_load():
    names = {p.stem: load_dataset(p) for p in find_datasets()}
    assert {"smoke", "reasoning", "tools"} <= set(names)
    assert len(names["smoke"]) >= 12 and len(names["reasoning"]) == 8
    assert all(case.expect for cases in names.values() for case in cases)
