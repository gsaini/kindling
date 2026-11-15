import pytest

from kindling.agent import StepLimitReached, run_agent
from kindling.llm import FakeModel, ToolCall, Turn
from kindling.tools import ToolRegistry, register_builtins


@pytest.mark.anyio
async def test_the_loop_runs_tools_and_feeds_results_back(tmp_path):
    tools = register_builtins(ToolRegistry(), tmp_path)
    model = FakeModel(
        Turn(thinking="I should add.", tool_calls=[ToolCall("calculate", {"expression": "17*23"})]),
        lambda messages, kwargs: Turn(content=f"It is {messages[-1]['content']}."),
    )
    events = [e async for e in run_agent(model, [{"role": "user", "content": "17*23?"}], tools=tools)]
    kinds = [e.type for e in events]
    assert kinds[:2] == ["thinking", "tool_call"] and kinds[-1] == "done"
    assert next(e for e in events if e.type == "tool_result").text == "391"
    assert "".join(e.text for e in events if e.type == "content") == "It is 391."
    done = events[-1]
    assert done.steps == 2 and done.usage.output_tokens == 10
    assert [m["role"] for m in done.new_messages] == ["assistant", "tool", "assistant"]
    # The tool schemas went to the model, and its reasoning stayed in the turn's history.
    assert model.calls[0]["tools"] and model.calls[1]["messages"][1]["thinking"] == "I should add."


@pytest.mark.anyio
async def test_a_model_that_never_stops_calling_tools_is_cut_off(tmp_path):
    tools = register_builtins(ToolRegistry(), tmp_path)
    loop = Turn(tool_calls=[ToolCall("current_time", {})])
    with pytest.raises(StepLimitReached):
        async for _ in run_agent(
            FakeModel(loop, loop, loop), [{"role": "user", "content": "x"}], tools=tools, max_steps=3
        ):
            pass
