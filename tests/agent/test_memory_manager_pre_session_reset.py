"""MemoryManager.on_pre_session_reset contract (working-frame retirement).

Pins the host-side ACK validation for the /reset and /undo prerequisite:
an advancing cleared ACK is accepted, an exact replay of an already-cleared
frozen reference is accepted (durable ACK landed, host crashed before
rotation), and everything else fails closed.
"""
import time

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


def _ref(**overrides):
    reference = {
        "provider": "frame",
        "principal": "chris",
        "agent": "jenna",
        "project": "proj",
        "session_id": "sess-1",
        "logical_task_id": "task-1",
        "revision": 3,
        "lifecycle_epoch": 1,
        "expires_at": 4102444800.0,
        "cleared": False,
    }
    reference.update(overrides)
    return reference


class _FrameProvider(MemoryProvider):
    """Provider stub with a controllable pre-session-reset result."""

    pre_session_reset_api_version = 1

    def __init__(self, ack=None, sync_delay=0.0):
        self.ack = ack
        self.sync_delay = sync_delay
        self.events = []

    @property
    def name(self):
        return "frame"

    def is_available(self):
        return True

    def initialize(self, session_id="", **kwargs):
        pass

    def system_prompt_block(self):
        return ""

    def prefetch(self, query, *, session_id=""):
        return ""

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        if self.sync_delay:
            time.sleep(self.sync_delay)
        self.events.append(("sync", assistant_content))

    def on_pre_session_reset(self, session_id, *, reference, rewound=False):
        self.events.append(("reset", session_id, rewound))
        if isinstance(self.ack, Exception):
            raise self.ack
        return dict(self.ack)


def _manager(provider, sink=None):
    manager = MemoryManager()
    manager.add_provider(provider)
    if sink is not None:
        manager.set_working_frame_scope_sink(sink)
    return manager


def test_advancing_cleared_ack_accepted_and_persisted():
    ack = _ref(revision=4, cleared=True)
    provider = _FrameProvider(ack=ack)
    recorded = []
    manager = _manager(provider, sink=lambda ref: recorded.append(ref) or True)

    result = manager.on_pre_session_reset("sess-1", reference=_ref())

    assert result["revision"] == 4
    assert result["cleared"] is True
    assert recorded == [result]
    assert provider.events == [("reset", "sess-1", False)]


def test_equal_already_cleared_ack_replay_accepted():
    frozen = _ref(revision=4, cleared=True)
    provider = _FrameProvider(ack=dict(frozen))
    recorded = []
    manager = _manager(provider, sink=lambda ref: recorded.append(ref) or True)

    result = manager.on_pre_session_reset("sess-1", reference=frozen)

    assert result == frozen
    assert recorded == [frozen]


def test_equal_non_cleared_ack_rejected():
    frozen = _ref(revision=4, cleared=False)
    provider = _FrameProvider(ack=dict(frozen))
    manager = _manager(provider)

    with pytest.raises(RuntimeError):
        manager.on_pre_session_reset("sess-1", reference=frozen)


def test_equal_revision_cleared_ack_rejected_when_frozen_not_cleared():
    frozen = _ref(revision=4, cleared=False)
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True))
    manager = _manager(provider)

    with pytest.raises(RuntimeError):
        manager.on_pre_session_reset("sess-1", reference=frozen)


def test_scope_mismatch_rejected():
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True, agent="liz"))
    manager = _manager(provider)

    with pytest.raises(RuntimeError):
        manager.on_pre_session_reset("sess-1", reference=_ref())


def test_reference_session_mismatch_rejected():
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True))
    manager = _manager(provider)

    with pytest.raises(ValueError):
        manager.on_pre_session_reset("other-session", reference=_ref())


def test_sink_refusal_raises():
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True))
    manager = _manager(provider, sink=lambda ref: False)

    with pytest.raises(RuntimeError):
        manager.on_pre_session_reset("sess-1", reference=_ref())


def test_no_capable_provider_raises():
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True))
    provider.pre_session_reset_api_version = 0
    manager = _manager(provider)

    with pytest.raises(RuntimeError):
        manager.on_pre_session_reset("sess-1", reference=_ref())


def test_reset_runs_fifo_behind_queued_sync():
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True), sync_delay=0.3)
    manager = _manager(provider)
    manager.initialize_all("sess-1")

    manager.sync_all("user", "turn-1", session_id="sess-1")
    result = manager.on_pre_session_reset("sess-1", reference=_ref())

    assert result["revision"] == 4
    assert provider.events == [("sync", "turn-1"), ("reset", "sess-1", False)]


def test_state_preserving_reset_error_propagates_unchanged():
    from agent.memory_provider import StatePreservingResetError

    provider = _FrameProvider(ack=StatePreservingResetError("working frame lifetime changed"))
    recorded = []
    manager = _manager(provider, sink=lambda ref: recorded.append(ref) or True)

    with pytest.raises(StatePreservingResetError):
        manager.on_pre_session_reset("sess-1", reference=_ref())

    assert recorded == []
    # Transition-neutral: the manager stays usable for later syncs.
    assert manager._memory_lifecycle.read_generation() is not None
    manager.initialize_all("sess-1")
    manager.sync_all("user", "turn-after", session_id="sess-1")
    assert manager.flush_pending(timeout=2)
    assert ("sync", "turn-after") in provider.events


def test_timed_out_retirement_is_abandoned_and_never_clears(monkeypatch):
    import agent.memory_manager as memory_manager

    monkeypatch.setattr(memory_manager, "_PRE_SESSION_RESET_TIMEOUT_S", 0.05)
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True), sync_delay=0.4)
    recorded = []
    manager = _manager(provider, sink=lambda ref: recorded.append(ref) or True)
    manager.initialize_all("sess-1")

    manager.sync_all("user", "turn-1", session_id="sess-1")
    with pytest.raises(TimeoutError):
        manager.on_pre_session_reset("sess-1", reference=_ref())

    assert manager.flush_pending(timeout=2)
    assert all(event[0] != "reset" for event in provider.events), (
        "a timed-out retirement must never execute later (it would clear the "
        "frame after /reset already failed closed)"
    )
    assert recorded == []
    # The lifecycle stays usable: abandonment is transition-neutral.
    assert manager._memory_lifecycle.read_generation() is not None
    manager.sync_all("user", "turn-2", session_id="sess-1")
    assert manager.flush_pending(timeout=2)
    assert ("sync", "turn-2") in provider.events


def test_explicit_timeout_caps_but_never_extends_retirement_deadline():
    from agent.memory_manager import _PRE_SESSION_RESET_TIMEOUT_S

    provider = _FrameProvider(ack=_ref(revision=4, cleared=True))
    manager = _manager(provider)
    seen = []
    original = manager._memory_lifecycle.call

    def spying_call(fn, **kwargs):
        seen.append(kwargs.get("timeout"))
        return original(fn, **kwargs)

    manager._memory_lifecycle.call = spying_call

    manager.on_pre_session_reset("sess-1", reference=_ref())
    manager.on_pre_session_reset("sess-1", reference=_ref(), timeout=3.5)
    manager.on_pre_session_reset("sess-1", reference=_ref(), timeout=99.0)

    assert seen == [_PRE_SESSION_RESET_TIMEOUT_S, 3.5, _PRE_SESSION_RESET_TIMEOUT_S]


def test_expired_caller_deadline_abandons_before_provider_runs():
    """timeout<=0 (the caller's deadline already passed) must abandon the
    queued retirement so its fn never executes late behind the FIFO."""
    provider = _FrameProvider(ack=_ref(revision=4, cleared=True), sync_delay=0.3)
    manager = _manager(provider)
    manager.initialize_all("sess-1")

    manager.sync_all("user", "turn-1", session_id="sess-1")
    with pytest.raises(TimeoutError):
        manager.on_pre_session_reset("sess-1", reference=_ref(), timeout=0.0)

    assert manager.flush_pending(timeout=2)
    assert all(event[0] != "reset" for event in provider.events), (
        "a retirement whose caller deadline already expired must never run"
    )
