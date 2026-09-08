"""DE_DESIGN §1.1 #18 — no spillover writes for policy-bound runs.

Both ``maybe_persist_tool_result`` call sites in ``agent/tool_executor.py``
route through ``_persist_or_truncate_tool_result``: an oversized result from a
policy-bound run is truncated in memory and never touches
``$HERMES_HOME/cache/spillover``; public runs keep the persistence path.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from tests.run_agent.test_tool_activity_heartbeat import _make_agent
from tools.budget_config import BudgetConfig


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _policy():
    from gateway.private_execution_policy import build_policy

    lease = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    return build_policy(
        run_id="run-1",
        profile="design",
        execution_attempt_id="att-1",
        lease_expires_at=lease,
        allowed_toolsets=("alpha",),
    )


def _spillover_files(tmp_path) -> list:
    spill = tmp_path / ".hermes" / "cache" / "spillover"
    return sorted(spill.glob("*")) if spill.exists() else []


_CONFIG = BudgetConfig(tool_overrides={"tool_x": 100}, preview_size=40)
_MARKER = "PRIVATE-MARKER-8471"
_BIG = _MARKER + ("z" * 500)


def test_policy_bound_truncates_in_memory(tmp_path, monkeypatch):
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    agent._private_execution_policy = _policy()

    result = te._persist_or_truncate_tool_result(
        agent,
        content=_BIG,
        tool_name="tool_x",
        tool_use_id="tc1",
        effective_task_id="task",
        config=_CONFIG,
    )

    assert "[Truncated in memory" in result
    assert _MARKER in result  # preview keeps the head
    assert len(result) < len(_BIG)
    assert _spillover_files(tmp_path) == []


def test_policy_bound_small_result_untouched(tmp_path, monkeypatch):
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    agent._private_execution_policy = _policy()

    result = te._persist_or_truncate_tool_result(
        agent,
        content="small",
        tool_name="tool_x",
        tool_use_id="tc1",
        effective_task_id="task",
        config=_CONFIG,
    )

    assert result == "small"
    assert _spillover_files(tmp_path) == []


def test_accessor_error_fails_closed_to_truncation(tmp_path, monkeypatch):
    import agent.tool_executor as te
    import gateway.private_execution_policy as pep

    agent = _make_agent(monkeypatch)

    def _boom():
        raise RuntimeError("accessor broke")

    monkeypatch.setattr(pep, "current_private_policy", _boom)
    result = te._persist_or_truncate_tool_result(
        agent,
        content=_BIG,
        tool_name="tool_x",
        tool_use_id="tc1",
        effective_task_id="task",
        config=_CONFIG,
    )

    assert "[Truncated in memory" in result
    assert _spillover_files(tmp_path) == []


def test_public_run_still_spills_to_disk(tmp_path, monkeypatch):
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)

    result = te._persist_or_truncate_tool_result(
        agent,
        content=_BIG,
        tool_name="tool_x",
        tool_use_id="tc1",
        effective_task_id="task",
        config=_CONFIG,
    )

    files = _spillover_files(tmp_path)
    assert len(files) == 1
    assert _MARKER in files[0].read_text()
    assert "Full output saved to" in result


def test_concurrent_path_truncates_for_policy_bound_run(tmp_path, monkeypatch):
    """End-to-end through execute_tool_calls_concurrent: the oversized result
    of an allowed tool is truncated in memory, nothing hits spillover."""
    import run_agent as _ra
    from tools.registry import registry

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    monkeypatch.setattr(
        registry, "get_toolset_for_tool", {"tool_x": "alpha"}.get
    )
    monkeypatch.setattr(registry, "get_max_result_size", lambda name, default=None: 100)
    policy = _policy()
    agent._private_execution_policy = policy
    agent._append_guardrail_observation = lambda name, args, result, **kw: result
    agent._invoke_tool = MagicMock(return_value=_BIG)
    agent._execute_tool_calls_concurrent = (
        _ra.AIAgent._execute_tool_calls_concurrent.__get__(agent)
    )

    tool_call = MagicMock(id="tc_a")
    tool_call.function = MagicMock(arguments="{}")
    tool_call.function.name = "tool_x"
    msg = MagicMock()
    msg.tool_calls = [tool_call]
    messages: list = []

    from gateway.private_execution_policy import (
        bind_private_policy,
        reset_private_policy,
    )

    token = bind_private_policy(policy)
    try:
        agent._execute_tool_calls_concurrent(msg, messages, "task")
    finally:
        reset_private_policy(token)

    contents = [m.get("content") for m in messages if isinstance(m, dict)]
    assert any(
        isinstance(c, str) and "[Truncated in memory" in c for c in contents
    ), f"no in-memory truncation found in {contents!r}"
    assert _spillover_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Choke point inside maybe_persist_tool_result itself (tool_executor.py:2032
# finding): enforce_turn_budget calls maybe_persist_tool_result(threshold=0)
# directly, bypassing _persist_or_truncate_tool_result — the gate must live
# in tools/tool_result_storage.maybe_persist_tool_result so every caller is
# covered when the private-policy ContextVar is bound.
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _bound_policy():
    from gateway.private_execution_policy import (
        bind_private_policy,
        reset_private_policy,
    )

    token = bind_private_policy(_policy())
    try:
        yield
    finally:
        reset_private_policy(token)


def _turn_config():
    return BudgetConfig(turn_budget=100, preview_size=40)


def test_enforce_turn_budget_policy_bound_never_spills(tmp_path):
    from tools.tool_result_storage import enforce_turn_budget

    msgs = [{"content": _BIG, "tool_call_id": "tb1"}]
    with _bound_policy():
        enforce_turn_budget(msgs, env=None, config=_turn_config())

    assert _spillover_files(tmp_path) == []
    assert "[Truncated in memory" in msgs[0]["content"]
    assert len(msgs[0]["content"]) < len(_BIG)


def test_enforce_turn_budget_public_still_spills(tmp_path):
    from tools.tool_result_storage import enforce_turn_budget

    msgs = [{"content": _BIG, "tool_call_id": "tb1"}]
    enforce_turn_budget(msgs, env=None, config=_turn_config())

    assert len(_spillover_files(tmp_path)) == 1
    assert "Full output saved to:" in msgs[0]["content"]


def test_maybe_persist_policy_bound_truncates_in_memory(tmp_path):
    from tools.tool_result_storage import maybe_persist_tool_result

    with _bound_policy():
        result = maybe_persist_tool_result(
            content=_BIG,
            tool_name="tool_x",
            tool_use_id="tc9",
            env=None,
            config=_CONFIG,
        )

    assert "[Truncated in memory" in result
    assert _spillover_files(tmp_path) == []


def test_maybe_persist_accessor_error_fails_closed(tmp_path, monkeypatch):
    import gateway.private_execution_policy as pep
    from tools.tool_result_storage import maybe_persist_tool_result

    def _boom():
        raise RuntimeError("accessor broke")

    monkeypatch.setattr(pep, "current_private_policy", _boom)
    result = maybe_persist_tool_result(
        content=_BIG,
        tool_name="tool_x",
        tool_use_id="tc9",
        env=None,
        config=_CONFIG,
    )

    assert "[Truncated in memory" in result
    assert _spillover_files(tmp_path) == []
