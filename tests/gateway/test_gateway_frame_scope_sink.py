"""Gateway wiring for the working-frame scope sink (contract B).

A freshly created gateway agent must get the durable scope sink bound to its
memory manager, and the sink must read the LIVE ``agent.session_id`` at call
time (the agent rebinds session ids across rotations; a captured id would
persist ACKs against a dead session).
"""
from types import SimpleNamespace

from agent.memory_manager import MemoryManager
from gateway import run as gateway_run
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.config import Platform
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


def test_fresh_agent_creation_attaches_scope_sink(monkeypatch):
    """Execute the extracted constructor, then checkpoint across a session rotation."""
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda config: {})
    store = _RecordingStore()
    manager = MemoryManager()
    agent = SimpleNamespace(_memory_manager=manager, session_id="sess-1")
    constructed = []

    def create(**kwargs):
        constructed.append(kwargs)
        return agent

    runner = SimpleNamespace(
        session_store=store, _active_profile_name=lambda: "jenna", _session_db=None,
        _prefill_messages=[], _service_tier=None, _refresh_fallback_model=lambda: None,
    )
    ctx = SimpleNamespace(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1"),
        AIAgent=create, user_config={}, enabled_toolsets=[], disabled_toolsets=[],
        session_id="sess-1", session_key="telegram:c1",
    )
    result = TurnRunner(runner, ctx)._build_fresh_agent(
        {"model": "test", "runtime": {}}, "telegram", "", 1, None, {}, False,
    )
    assert result is agent
    assert constructed[0]["load_soul_identity"] is True
    assert constructed[0]["gateway_session_key"] == ctx.session_key
    manager._working_frame_scope_sink({"revision": 1})
    agent.session_id = "sess-2"
    manager._working_frame_scope_sink({"revision": 2})
    assert store.calls == [
        ("telegram:c1", "sess-1", {"revision": 1}, "jenna"),
        ("telegram:c1", "sess-2", {"revision": 2}, "jenna"),
    ]
