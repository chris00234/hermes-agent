"""DE_DESIGN §5/§6.7 — private execution tool gate in _authorized_dispatch.

The gate reads the private-policy ContextVar AND ``agent._private_execution_policy``.
When either is set, both must be present and digest-equal, the tool's toolset
must be in ``allowed_toolsets``, and the lease must be unexpired. Every defect
blocks fail-closed with ``error_type='private_execution_grant_block'`` and the
denied call never reaches ``_begin_tool_execution``.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from tests.run_agent.test_tool_activity_heartbeat import _make_agent

BLOCK_ERROR_TYPE = "private_execution_grant_block"


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _policy(allowed=("alpha",), lease_offset_s=3600, lease=None, run_id="run-1"):
    from gateway.private_execution_policy import build_policy

    if lease is None:
        lease = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_offset_s)
        ).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return build_policy(
        run_id=run_id,
        profile="design",
        execution_attempt_id="att-1",
        lease_expires_at=lease,
        allowed_toolsets=allowed,
    )


@contextmanager
def _bound(policy):
    from gateway.private_execution_policy import (
        bind_private_policy,
        reset_private_policy,
    )

    token = bind_private_policy(policy)
    try:
        yield
    finally:
        reset_private_policy(token)


@pytest.fixture()
def gate_agent(monkeypatch):
    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    return agent


@pytest.fixture()
def toolset_map(monkeypatch):
    from tools.registry import registry

    mapping = {
        "terminal": "alpha",
        "web_search": "beta",
        "memory": "memory",
        "delegate_task": "delegation",
        "message_agent": "alpha",
    }
    monkeypatch.setattr(registry, "get_toolset_for_tool", mapping.get)
    return mapping


@pytest.fixture()
def emitted(monkeypatch):
    import agent.tool_executor as te

    calls: list = []
    monkeypatch.setattr(
        te, "_emit_terminal_post_tool_call", lambda *a, **k: calls.append(k)
    )
    return calls


@pytest.fixture()
def begun(monkeypatch):
    import agent.tool_executor as te

    calls: list = []
    monkeypatch.setattr(
        te, "_begin_tool_execution", lambda *a, **k: calls.append(k)
    )
    return calls


def _run(agent, name="terminal"):
    import agent.tool_executor as te

    executed: list = []

    def _execute(next_args):
        executed.append(next_args)
        return json.dumps({"ok": True})

    outcome = te._run_agent_tool_execution_middleware(
        agent,
        function_name=name,
        function_args={},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=_execute,
        display_index=1,
    )
    return outcome, executed


def _assert_blocked(outcome, executed, emitted):
    assert outcome.blocked is True
    assert executed == []
    assert "private execution" in json.loads(outcome.result)["error"]
    assert emitted and emitted[-1]["error_type"] == BLOCK_ERROR_TYPE
    assert emitted[-1]["status"] == "blocked"


def test_no_policy_executes(gate_agent, toolset_map, emitted, begun):
    outcome, executed = _run(gate_agent)
    assert outcome.blocked is False
    assert executed == [{}]


def test_allowed_tool_executes(gate_agent, toolset_map, emitted, begun):
    policy = _policy(allowed=("alpha",))
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "terminal")
    assert outcome.blocked is False
    assert executed == [{}]
    assert len(begun) == 1


def test_disallowed_tool_blocked(gate_agent, toolset_map, emitted, begun):
    policy = _policy(allowed=("alpha",))
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "web_search")
    _assert_blocked(outcome, executed, emitted)
    assert begun == []


def test_unregistered_tool_blocked(gate_agent, toolset_map, emitted, begun):
    policy = _policy(allowed=("alpha",))
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "not_a_tool")
    _assert_blocked(outcome, executed, emitted)


def test_excluded_toolset_never_allowed(gate_agent, toolset_map, emitted, begun):
    # build_policy strips memory even when the caller passes it.
    policy = _policy(allowed=("alpha", "memory"))
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "memory")
    _assert_blocked(outcome, executed, emitted)


def _raw_policy(allowed):
    """Policy built WITHOUT build_policy's stripping — models a grant whose
    allowed_toolsets still carries a denied tool's toolset."""
    from gateway.private_execution_policy import PrivateExecutionPolicy

    lease = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    return PrivateExecutionPolicy(
        run_id="run-1",
        profile="design",
        execution_attempt_id="att-1",
        lease_expires_at=lease,
        allowed_toolsets=frozenset(allowed),
        policy_digest="digest-1",
        grant_version="v1",
    )


def test_delegate_task_blocked_when_delegation_toolset_allowed(
    gate_agent, toolset_map, emitted, begun
):
    policy = _raw_policy({"alpha", "delegation"})
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "delegate_task")
    _assert_blocked(outcome, executed, emitted)
    assert begun == []


def test_message_agent_blocked_when_its_toolset_allowed(
    gate_agent, toolset_map, emitted, begun
):
    policy = _raw_policy({"alpha"})
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "message_agent")
    _assert_blocked(outcome, executed, emitted)
    assert begun == []


def test_expired_lease_blocked(gate_agent, toolset_map, emitted, begun):
    policy = _policy(allowed=("alpha",), lease_offset_s=-10)
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "terminal")
    _assert_blocked(outcome, executed, emitted)


def test_malformed_lease_blocked(gate_agent, toolset_map, emitted, begun):
    policy = _policy(allowed=("alpha",), lease="not-a-timestamp")
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome, executed = _run(gate_agent, "terminal")
    _assert_blocked(outcome, executed, emitted)


def test_contextvar_without_attr_blocked(gate_agent, toolset_map, emitted, begun):
    with _bound(_policy(allowed=("alpha",))):
        outcome, executed = _run(gate_agent, "terminal")
    _assert_blocked(outcome, executed, emitted)


def test_attr_without_contextvar_blocked(gate_agent, toolset_map, emitted, begun):
    gate_agent._private_execution_policy = _policy(allowed=("alpha",))
    outcome, executed = _run(gate_agent, "terminal")
    _assert_blocked(outcome, executed, emitted)


def test_digest_mismatch_blocked(gate_agent, toolset_map, emitted, begun):
    gate_agent._private_execution_policy = _policy(allowed=("alpha",), run_id="run-2")
    with _bound(_policy(allowed=("alpha",), run_id="run-1")):
        outcome, executed = _run(gate_agent, "terminal")
    _assert_blocked(outcome, executed, emitted)


def test_accessor_error_blocked(gate_agent, toolset_map, emitted, begun, monkeypatch):
    import gateway.private_execution_policy as pep

    def _boom():
        raise RuntimeError("accessor broke")

    monkeypatch.setattr(pep, "current_private_policy", _boom)
    outcome, executed = _run(gate_agent, "terminal")
    _assert_blocked(outcome, executed, emitted)


def test_sequential_offload_path_blocked(gate_agent, toolset_map, emitted, begun):
    """The sequential worker thread inherits the policy via
    propagate_context_to_thread and blocks a disallowed tool."""
    import agent.tool_executor as te

    executed: list = []
    policy = _policy(allowed=("alpha",))
    gate_agent._private_execution_policy = policy
    with _bound(policy):
        outcome = te._run_sequential_tool_execution_middleware(
            gate_agent,
            function_name="web_search",
            function_args={},
            effective_task_id="task",
            tool_call_id="tc1",
            execute=lambda next_args: executed.append(next_args)
            or json.dumps({"ok": True}),
            display_index=1,
        )
    _assert_blocked(outcome, executed, emitted)


def test_concurrent_batch_blocks_only_disallowed(
    gate_agent, toolset_map, emitted, begun, monkeypatch
):
    """A 2-tool batch through the concurrent path: the allowed tool runs, the
    disallowed one blocks in the worker thread (ContextVar propagated)."""
    import run_agent as _ra

    gate_agent._execute_tool_calls_concurrent = (
        _ra.AIAgent._execute_tool_calls_concurrent.__get__(gate_agent)
    )

    class _FakeToolCall:
        def __init__(self, name, call_id):
            self.function = MagicMock(arguments="{}")
            self.function.name = name
            self.id = call_id

    invoked: list = []

    def _invoke(name, *a, **kw):
        invoked.append(name)
        return json.dumps({"ok": name})

    gate_agent._invoke_tool = MagicMock(side_effect=_invoke)
    policy = _policy(allowed=("alpha",))
    gate_agent._private_execution_policy = policy

    msg = MagicMock()
    msg.tool_calls = [
        _FakeToolCall("terminal", "tc_a"),
        _FakeToolCall("web_search", "tc_b"),
    ]
    messages: list = []
    with _bound(policy):
        gate_agent._execute_tool_calls_concurrent(msg, messages, "task")

    assert invoked == ["terminal"]
    blocked_contents = [
        m.get("content") for m in messages if isinstance(m, dict)
    ]
    assert any(
        isinstance(c, str) and "private execution" in c for c in blocked_contents
    ), f"no blocked tool result found in {blocked_contents!r}"
    block_emits = [
        k for k in emitted if k.get("error_type") == BLOCK_ERROR_TYPE
    ]
    assert len(block_emits) == 1
