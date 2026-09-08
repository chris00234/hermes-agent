"""Gateway wiring for the working-frame scope sink (contract B).

A freshly created gateway agent must get the durable scope sink bound to its
memory manager, and the sink must read the LIVE ``agent.session_id`` at call
time (the agent rebinds session ids across rotations; a captured id would
persist ACKs against a dead session).
"""
import ast
import inspect
from types import SimpleNamespace

from agent.memory_manager import MemoryManager
from gateway import run as gateway_run
from gateway.working_frame_scope import attach_working_frame_scope_sink


class _RecordingStore:
    def __init__(self):
        self.calls = []

    def record_working_frame_scope(self, session_key, expected_session_id, reference,
                                   *, active_profile=None):
        self.calls.append((session_key, expected_session_id, reference, active_profile))
        return True


def test_attach_binds_sink_using_live_agent_session_id():
    store = _RecordingStore()
    runner = SimpleNamespace(session_store=store)
    manager = MemoryManager()
    agent = SimpleNamespace(_memory_manager=manager, session_id="sess-1")

    attach_working_frame_scope_sink(runner, agent, "telegram:c1")

    sink = manager._working_frame_scope_sink
    assert callable(sink)
    assert sink({"revision": 1}) is True
    agent.session_id = "sess-2"
    sink({"revision": 2})
    assert store.calls == [
        ("telegram:c1", "sess-1", {"revision": 1}, None),
        ("telegram:c1", "sess-2", {"revision": 2}, None),
    ]


def test_sink_passes_runner_active_profile_for_agent_fencing():
    store = _RecordingStore()
    runner = SimpleNamespace(
        session_store=store, _active_profile_name=lambda: "jenna"
    )
    manager = MemoryManager()
    agent = SimpleNamespace(_memory_manager=manager, session_id="sess-1")

    attach_working_frame_scope_sink(runner, agent, "telegram:c1")

    manager._working_frame_scope_sink({"revision": 1})
    assert store.calls == [("telegram:c1", "sess-1", {"revision": 1}, "jenna")]


def test_attach_is_defensive_about_missing_pieces():
    store = _RecordingStore()
    runner = SimpleNamespace(session_store=store)
    # No memory manager at all.
    attach_working_frame_scope_sink(
        runner, SimpleNamespace(session_id="s"), "key"
    )
    # Manager without the sink registration method.
    attach_working_frame_scope_sink(
        runner,
        SimpleNamespace(_memory_manager=object(), session_id="s"),
        "key",
    )
    # Runner without a session store.
    attach_working_frame_scope_sink(
        SimpleNamespace(),
        SimpleNamespace(_memory_manager=MemoryManager(), session_id="s"),
        "key",
    )
    assert store.calls == []


def _call_names(node):
    return {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
    }


def test_fresh_agent_creation_attaches_scope_sink():
    """AST pin: the ``agent = ctx.AIAgent(...)`` fresh-agent block in
    gateway/run.py must call ``attach_working_frame_scope_sink``."""
    tree = ast.parse(inspect.getsource(gateway_run))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        creates_agent = any(
            isinstance(sub, ast.Assign)
            and isinstance(sub.value, ast.Call)
            and isinstance(sub.value.func, ast.Attribute)
            and sub.value.func.attr == "AIAgent"
            and isinstance(sub.value.func.value, ast.Name)
            and sub.value.func.value.id == "ctx"
            for sub in ast.walk(node)
        )
        if not creates_agent:
            continue
        assert "attach_working_frame_scope_sink" in _call_names(node), (
            "the fresh-agent creation block in gateway/run.py must bind the "
            "working-frame scope sink (attach_working_frame_scope_sink) so "
            "checkpoint ACKs become durable for /reset //undo (contract B)"
        )
        found = True
    assert found, "could not locate the ctx.AIAgent(...) fresh-agent block"
