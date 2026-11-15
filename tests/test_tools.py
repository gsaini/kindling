import pytest

from kindling.tools import ToolRegistry, register_builtins
from kindling.tools.builtin import evaluate


@pytest.fixture
def tools(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.txt").write_text("hello")
    (tmp_path.parent / "secret.txt").write_text("nope")
    return register_builtins(ToolRegistry(), tmp_path)


def test_calculator_does_arithmetic_and_nothing_else():
    assert evaluate("(1250.50 - 300) * 2") == 1901.0
    assert evaluate("2 ** 100") == 2**100
    bombs = ("(((9**99)**99)**99)**99", " * ".join(["9**99"] * 14), "2 ** 5000", "1" * 600)
    for bad in ("__import__('os')", "a + 1", "True + 1", *bombs):
        with pytest.raises(ValueError):
            evaluate(bad)


def test_schemas_come_from_signatures(tools):
    schema = next(s for s in tools.schemas() if s["function"]["name"] == "read_file")["function"]
    assert schema["description"].startswith("Read a text file")
    assert schema["parameters"]["required"] == ["path"]
    assert "description" in schema["parameters"]["properties"]["path"]


@pytest.mark.anyio
async def test_tools_run_and_report_errors_as_text(tools):
    assert (await tools.call("calculate", {"expression": "2+3"})).output == "5"
    assert (await tools.call("read_file", {"path": "notes/a.txt"})).output == "hello"
    escaped = await tools.call("read_file", {"path": "../secret.txt"})
    assert not escaped.ok and "outside the workspace" in escaped.output
    assert not (await tools.call("calculate", {})).ok  # missing argument
    assert not (await tools.call("rm_rf", {})).ok  # unknown tool
