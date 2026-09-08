"""DE_DESIGN §5 graft — invoke_tool consults the private-policy accessor.

Callers that pass the skip flags (``pre_tool_block_checked=True``,
``skip_tool_request_middleware=True``, ``skip_tool_execution_middleware=True``)
bypass the funnel's middleware layers; the seam inside ``invoke_tool`` still
blocks a call the private execution policy does not allow.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from tests.run_agent.test_tool_activity_heartbeat import _make_agent


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


_SKIP_FLAGS = dict(
    pre_tool_block_checked=True,
    skip_tool_request_middleware=True,
    skip_tool_execution_middleware=True,
)


def test_invoke_tool_seam_blocked(monkeypatch):
    from agent.agent_runtime_helpers import invoke_tool
    from gateway.private_execution_policy import (
        bind_private_policy,
        reset_private_policy,
    )
    from tools.registry import registry

    agent = _make_agent(monkeypatch)
    monkeypatch.setattr(
        registry, "get_toolset_for_tool", {"todo_list": "todo"}.get
    )
    monkeypatch.setattr(
        "tools.todo_tool.todo_tool",
        lambda *a, **k: pytest.fail("blocked tool must not run"),
    )
    policy = _policy()
    agent._private_execution_policy = policy
    token = bind_private_policy(policy)
    try:
        result = invoke_tool(
            agent,
            "todo_list",
            {"todos": []},
            "task",
            tool_call_id="tc1",
            **_SKIP_FLAGS,
        )
    finally:
        reset_private_policy(token)

    assert "private execution" in json.loads(result)["error"]


def test_invoke_tool_seam_denied_tool_blocked(monkeypatch):
    """delegate_task is denied by tool name even when its "delegation"
    toolset is in allowed_toolsets (finding: exclusion listed tool names
    but was applied to toolset names)."""
    from datetime import datetime, timedelta, timezone

    from agent.agent_runtime_helpers import invoke_tool
    from gateway.private_execution_policy import (
        PrivateExecutionPolicy,
        bind_private_policy,
        reset_private_policy,
    )
    from tools.registry import registry

    agent = _make_agent(monkeypatch)
    agent._dispatch_delegate_task = lambda *a, **k: pytest.fail(
        "denied tool must not dispatch"
    )
    monkeypatch.setattr(
        registry, "get_toolset_for_tool", {"delegate_task": "delegation"}.get
    )
    lease = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    policy = PrivateExecutionPolicy(
        run_id="run-1",
        profile="design",
        execution_attempt_id="att-1",
        lease_expires_at=lease,
        allowed_toolsets=frozenset({"delegation"}),
        policy_digest="digest-1",
        grant_version="v1",
    )
    agent._private_execution_policy = policy
    token = bind_private_policy(policy)
    try:
        result = invoke_tool(
            agent,
            "delegate_task",
            {"task": "x"},
            "task",
            tool_call_id="tc1",
            **_SKIP_FLAGS,
        )
    finally:
        reset_private_policy(token)

    assert "private execution" in json.loads(result)["error"]


def test_invoke_tool_seam_accessor_error_blocks(monkeypatch):
    import gateway.private_execution_policy as pep
    from agent.agent_runtime_helpers import invoke_tool

    agent = _make_agent(monkeypatch)

    def _boom():
        raise RuntimeError("accessor broke")

    monkeypatch.setattr(pep, "current_private_policy", _boom)
    monkeypatch.setattr(
        "tools.todo_tool.todo_tool",
        lambda *a, **k: pytest.fail("blocked tool must not run"),
    )
    result = invoke_tool(
        agent,
        "todo_list",
        {"todos": []},
        "task",
        tool_call_id="tc1",
        **_SKIP_FLAGS,
    )

    assert "private execution" in json.loads(result)["error"]


def test_invoke_tool_seam_no_policy_executes(monkeypatch):
    from agent.agent_runtime_helpers import invoke_tool

    agent = _make_agent(monkeypatch)
    monkeypatch.setattr("tools.todo_tool.todo_tool", lambda *a, **k: "todo-ok")
    result = invoke_tool(
        agent,
        "todo_list",
        {"todos": []},
        "task",
        tool_call_id="tc1",
        **_SKIP_FLAGS,
    )

    assert result == "todo-ok"
