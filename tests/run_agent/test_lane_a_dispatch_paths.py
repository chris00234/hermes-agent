"""Private policy regressions through the extracted production dispatch paths."""

import itertools
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


_SKIP_NAMES = (
    "pre_tool_block_checked",
    "skip_tool_request_middleware",
    "skip_tool_execution_middleware",
)
_SEAMS = ["sequential-registry", "sequential-inline", "concurrent-registry", "concurrent-inline"]
_SEAMS += list(itertools.product((False, True), repeat=len(_SKIP_NAMES)))
_DEFECTS = (
    "allowed", "public", "missing-attr", "missing-context", "mismatch",
    "expired", "malformed-lease", "accessor-error", "unknown-toolset", "disallowed",
)
_PAYLOAD = "LANE-A-PRIVATE-RESULT\n" + "z" * 2000


@pytest.fixture
def dispatch_agent(monkeypatch):
    from agent.agent_runtime_helpers import invoke_tool
    from tests.run_agent.test_tool_activity_heartbeat import _make_agent

    agent = _make_agent(monkeypatch)
    agent._context_engine_tool_names = set()
    agent._memory_manager = None
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call.return_value.allows_execution = True
    agent._append_guardrail_observation = lambda name, args, result, **kw: result
    agent._invoke_tool = MethodType(invoke_tool, agent)
    agent._dispatch_delegate_task = MagicMock(return_value="delegated")
    agent.tool_start_callback = MagicMock()
    agent.tool_complete_callback = MagicMock()
    return agent


@contextmanager
def _policy_state(agent, defect, monkeypatch):
    import gateway.private_execution_policy as pep
    from tools.registry import registry

    policy = pep.build_policy(
        run_id="lane-a", profile="design", execution_attempt_id="lane-a-attempt",
        lease_expires_at="2999-01-01T00:00:00+00:00", allowed_toolsets=("alpha",),
    )
    if defect == "expired":
        policy = replace(policy, lease_expires_at="2000-01-01T00:00:00+00:00")
    elif defect == "malformed-lease":
        policy = replace(policy, lease_expires_at="not-a-lease")
    if defect not in {"public", "missing-attr"}:
        agent._private_execution_policy = policy
    if defect == "mismatch":
        agent._private_execution_policy = replace(policy, policy_digest="different")
    toolset = None if defect == "unknown-toolset" else "alpha"
    if defect == "disallowed":
        toolset = "beta"
    monkeypatch.setattr(registry, "get_toolset_for_tool", lambda name: toolset)
    context = None if defect in {"public", "missing-context"} else policy
    token = pep.bind_private_policy(context)
    if defect == "accessor-error":
        monkeypatch.setattr(pep, "current_private_policy", MagicMock(side_effect=RuntimeError))
    try:
        yield
    finally:
        pep.reset_private_policy(token)


def _call(agent, seam, name):
    import agent.tool_executor as te

    args = {"command": "test-only"} if name == "terminal" else {"todos": []}
    if isinstance(seam, tuple):
        return agent._invoke_tool(
            name, args, "lane-a-task", "lane-a-call", **dict(zip(_SKIP_NAMES, seam)),
        )
    call = SimpleNamespace(
        id="lane-a-call", function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )
    execute = (
        te.execute_tool_calls_concurrent if seam.startswith("concurrent")
        else te.execute_tool_calls_sequential
    )
    messages = []
    execute(agent, SimpleNamespace(tool_calls=[call]), messages, "lane-a-task")
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "lane-a-call"
    return messages[0]["content"]


@pytest.mark.parametrize("seam", _SEAMS)
@pytest.mark.parametrize("defect", _DEFECTS)
def test_real_dispatch_policy_matrix(dispatch_agent, monkeypatch, seam, defect):
    sink = MagicMock(return_value="lane-a-ok")
    monkeypatch.setattr("tools.todo_tool.todo_tool", sink)
    monkeypatch.setattr("model_tools.handle_function_call", sink)
    name = "terminal" if isinstance(seam, str) and seam.endswith("registry") else "todo_list"
    with _policy_state(dispatch_agent, defect, monkeypatch):
        result = _call(dispatch_agent, seam, name)
    if defect in {"allowed", "public"}:
        assert result == "lane-a-ok"
        sink.assert_called_once()
    else:
        assert "private execution" in json.loads(result)["error"]
        sink.assert_not_called()
        dispatch_agent._tool_guardrails.before_call.assert_not_called()
        dispatch_agent.tool_start_callback.assert_not_called()
        dispatch_agent.tool_complete_callback.assert_not_called()


@pytest.mark.parametrize("seam", _SEAMS)
def test_denied_tool_never_reaches_inline_dispatch(dispatch_agent, monkeypatch, seam):
    with _policy_state(dispatch_agent, "allowed", monkeypatch):
        result = _call(dispatch_agent, seam, "delegate_task")
    assert "never allowed" in json.loads(result)["error"]
    dispatch_agent._dispatch_delegate_task.assert_not_called()


@pytest.mark.parametrize("seam", _SEAMS[:4])
@pytest.mark.parametrize("defect", ["allowed", "public"])
def test_shared_commit_spillover(dispatch_agent, monkeypatch, tmp_path, seam, defect):
    import agent.tool_executor as te
    from tools.budget_config import BudgetConfig

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    budget = BudgetConfig(tool_overrides={"terminal": 100, "todo_list": 100}, preview_size=40)
    monkeypatch.setattr(te, "_budget_for_agent", lambda agent: budget)
    monkeypatch.setattr("tools.todo_tool.todo_tool", MagicMock(return_value=_PAYLOAD))
    monkeypatch.setattr("model_tools.handle_function_call", MagicMock(return_value=_PAYLOAD))
    name = "terminal" if seam.endswith("registry") else "todo_list"
    with _policy_state(dispatch_agent, defect, monkeypatch):
        result = _call(dispatch_agent, seam, name)
    spillover = list((tmp_path / "cache" / "spillover").glob("*"))
    if defect == "allowed":
        assert "[Truncated in memory" in result
        assert len(result) < len(_PAYLOAD)
        assert spillover == []
    else:
        assert "Full output saved to:" in result
        assert len(spillover) == 1
        assert spillover[0].read_text(encoding="utf-8") == _PAYLOAD


@pytest.mark.parametrize("defect", ["missing-context", "missing-attr", "accessor-error"])
def test_shared_commit_without_readable_complete_policy(
    dispatch_agent, monkeypatch, tmp_path, defect,
):
    import agent.tool_executor as te
    from tools.budget_config import BudgetConfig

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with _policy_state(dispatch_agent, defect, monkeypatch):
        committed = te._commit_tool_result(
            dispatch_agent, [], te._ToolCallRef("terminal", {}, "task", "call", []),
            _PAYLOAD, budget=BudgetConfig(tool_overrides={"terminal": 100}, preview_size=40),
            tool_duration=0.0, is_error=False, blocked=False, effect_disposition=None,
        )
    assert committed is not None
    assert "[Truncated in memory" in committed[0]
    assert not (tmp_path / "cache" / "spillover").exists()


@pytest.mark.parametrize("defect", _DEFECTS)
def test_extracted_authorization_gate(dispatch_agent, monkeypatch, defect):
    import agent.tool_executor as te

    sink = MagicMock(return_value="lane-a-ok")
    advance = MagicMock(side_effect=lambda callback=None: callback() if callback else None)
    state = te._ManagedToolResult(None, {}, [], False, True)
    with _policy_state(dispatch_agent, defect, monkeypatch):
        result = te._dispatch_authorized_once(
            dispatch_agent, state, te._ToolCallRef("terminal", {}, "task", "call", []),
            execute=sink, scope_block=None, display_index=1,
            begin_execution=advance, authorization_gate=None,
        )
    advance.assert_called_once()
    if defect in {"allowed", "public"}:
        assert result == "lane-a-ok"
        sink.assert_called_once()
    else:
        assert state.blocked
        assert "private execution" in json.loads(result)["error"]
        sink.assert_not_called()
        dispatch_agent._tool_guardrails.before_call.assert_not_called()
        dispatch_agent.tool_start_callback.assert_not_called()


@pytest.mark.parametrize("defect", ["allowed", "missing-context", "accessor-error", "public"])
def test_aggregate_budget_policy_fence(dispatch_agent, monkeypatch, tmp_path, defect):
    import agent.tool_executor as te
    from tools.budget_config import BudgetConfig

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    messages = [{"role": "tool", "tool_call_id": "budget-call", "content": _PAYLOAD}]
    dispatch_agent._apply_pending_steer_to_tool_results = MagicMock()
    with _policy_state(dispatch_agent, defect, monkeypatch):
        te._finalize_tool_batch(
            dispatch_agent, messages, "task", 1, BudgetConfig(turn_budget=100, preview_size=40),
        )
    spillover = list((tmp_path / "cache" / "spillover").glob("*"))
    if defect == "public":
        assert len(spillover) == 1
        assert spillover[0].read_text(encoding="utf-8") == _PAYLOAD
    else:
        assert spillover == []
        assert "[Truncated in memory" in messages[0]["content"]
    dispatch_agent._apply_pending_steer_to_tool_results.assert_called_once_with(messages, 1)


if __name__ == "__main__":
    # Explicit in-memory compile preflight; no pyc or shared pytest cache writes.
    root = Path(__file__).resolve().parents[2]
    for source in (root / "agent/tool_executor.py", Path(__file__)):
        compile(source.read_bytes(), str(source), "exec")
        print(f"in-memory compile: PASS {source}")
