"""Real worker ordering, bounded waits, and context isolation at boundaries."""

import threading
from concurrent.futures import Future, TimeoutError
from contextvars import ContextVar

import pytest

from agent import memory_lifecycle
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class RecordingProvider(MemoryProvider):
    pre_compress_checkpoint_api_version = 2

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.session = "old"
        self.fail_switch = False

    @property
    def name(self):
        return "recording"

    def initialize(self, session_id="", **kwargs):
        self.session = session_id

    def is_available(self):
        return True

    def get_tool_schemas(self):
        return []

    def system_prompt_block(self):
        return self.session

    def prefetch(self, query, *, session_id=""):
        return self.session

    def handle_tool_call(self, *args, **kwargs):
        return self.session

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        if user_content == "held":
            self.started.set()
            assert self.release.wait(5)
        self.calls.append(("sync", user_content, session_id))

    def on_session_switch(self, new_session_id, **kwargs):
        if self.fail_switch:
            raise RuntimeError("transition failed")
        self.session = new_session_id
        self.calls.append(("switch", new_session_id, kwargs))

    def on_pre_compress(self, messages, *, require_checkpoint=False):
        self.calls.append(("checkpoint", list(messages), require_checkpoint))
        return self.session

    def on_session_end(self, messages):
        self.calls.append(("end", list(messages)))


@pytest.fixture
def manager():
    manager = MemoryManager()
    provider = RecordingProvider()
    manager.add_provider(provider)
    manager._tool_to_provider["memory_search"] = provider
    yield manager, provider
    provider.release.set()
    manager.shutdown_all()


def hold_two_turns(manager, provider):
    manager.sync_all("held", "answer", session_id="old")
    assert provider.started.wait(2)
    manager.sync_all("queued-old", "answer", session_id="old")


@pytest.mark.parametrize("options", [
    {"rewound": True}, {"reason": "resume"}, {"reason": "branch"},
    {"reason": "compression"}, {"reset": True},
])
def test_transition_waits_behind_all_old_callbacks_and_blocks_stale_reads(manager, monkeypatch, options):
    mm, provider = manager
    monkeypatch.setattr(memory_lifecycle, "LIFECYCLE_TIMEOUT_SECONDS", 0.01)
    hold_two_turns(mm, provider)
    mm.on_session_switch("new", parent_session_id="old", **options)
    assert provider.calls == []
    assert mm.prefetch_all("continue", session_id="new") == ""
    assert mm.build_system_prompt() == ""
    assert "unavailable" in mm.handle_tool_call("memory_search", {})
    provider.release.set()
    assert mm.flush_pending(timeout=2)
    assert [call[0] for call in provider.calls] == ["sync", "sync", "switch"]
    assert mm.build_system_prompt() == mm.prefetch_all("continue", session_id="new") == "new"


def test_required_checkpoint_cannot_ack_queued_work_and_freezes_list(manager, monkeypatch):
    mm, provider = manager
    monkeypatch.setattr(memory_lifecycle, "LIFECYCLE_TIMEOUT_SECONDS", 0.01)
    hold_two_turns(mm, provider)
    evidence = [{"role": "user", "content": "original task"}]
    with pytest.raises(TimeoutError):
        mm.on_pre_compress([], evidence_messages=evidence, require_checkpoint=True)
    evidence.clear()
    assert provider.calls == []
    provider.release.set()
    assert mm.flush_pending(timeout=2)
    assert provider.calls[-1] == ("checkpoint", [{"role": "user", "content": "original task"}], True)
    assert [call[0] for call in provider.calls] == ["sync", "sync", "checkpoint"]


def test_async_boundary_and_same_worker_checkpoint_do_not_self_wait(manager):
    mm, provider = manager
    hold_two_turns(mm, provider)
    mm.commit_session_boundary_async([], new_session_id="new", parent_session_id="old")
    assert mm.build_system_prompt() == ""
    nested = mm._submit_background(lambda: mm.on_pre_compress([], require_checkpoint=True))
    provider.release.set()
    assert nested.result(timeout=2) == "new"
    assert [call[0] for call in provider.calls] == ["sync", "sync", "end", "switch", "checkpoint"]


def test_failed_newer_transition_is_not_unblocked_by_older_completion():
    callbacks = []

    def submit(fn, **kwargs):
        callbacks.append(fn)
        return Future()

    lifecycle = memory_lifecycle.MemoryLifecycle(submit)
    lifecycle.enqueue(lambda: None, transition=True)

    def fail():
        raise RuntimeError("newer failed")

    lifecycle.enqueue(fail, transition=True)
    with pytest.raises(RuntimeError):
        callbacks[1]()
    with pytest.raises(RuntimeError, match="superseded"):
        callbacks[0]()
    assert lifecycle.read_generation() is None
    lifecycle.enqueue(lambda: None, transition=True)
    callbacks[2]()
    assert lifecycle.read_generation() is not None


def test_failed_switch_stays_fenced_after_checkpoint_then_recovers_on_switch(manager):
    mm, provider = manager
    provider.fail_switch = True
    mm.on_session_switch("new")
    with pytest.raises(RuntimeError, match="prior memory session transition"):
        mm.on_pre_compress([], require_checkpoint=True)
    mm.sync_all("new-session turn", "answer", session_id="new")
    assert mm.flush_pending(timeout=2)
    assert provider.calls == []
    assert mm.build_system_prompt() == ""
    provider.fail_switch = False
    mm.on_session_switch("new")
    assert mm.build_system_prompt() == "new"


@pytest.mark.parametrize("checkpoint", [False, True])
def test_old_callback_that_is_admitted_late_cannot_reinsert_undone_turn(manager, monkeypatch, checkpoint):
    mm, provider = manager
    original_submit = mm._submit_background
    delayed = []
    monkeypatch.setattr(mm, "_submit_background", lambda fn, **kwargs: delayed.append(fn))
    mm.sync_all("undone", "answer", session_id="old")
    if checkpoint:
        mm.on_pre_compress([], require_checkpoint=True)
    else:
        mm.on_session_switch("old", rewound=True)
    original_submit(delayed[0])
    assert mm.flush_pending(timeout=2)
    assert [call[0] for call in provider.calls] == ["checkpoint" if checkpoint else "switch"]


def test_context_started_before_transition_is_discarded_when_it_finishes(manager):
    mm, provider = manager

    def switched():
        mm.on_session_switch("new")
        return "old context"

    provider.system_prompt_block = switched
    assert mm.build_system_prompt() == ""


def test_lifecycle_admission_is_bounded_and_shutdown_cannot_ack_checkpoint(manager, monkeypatch):
    mm, provider = manager
    monkeypatch.setattr(memory_lifecycle, "MAX_PENDING_LIFECYCLES", 1)
    hold_two_turns(mm, provider)
    mm._memory_lifecycle.enqueue(lambda: None, transition=True)
    with pytest.raises(RuntimeError, match="queue is full"):
        mm.on_pre_compress([], require_checkpoint=True)
    provider.release.set()
    mm.shutdown_all()
    with pytest.raises(RuntimeError, match="unavailable"):
        mm.on_pre_compress([], require_checkpoint=True)


def test_lifecycle_preserves_callers_profile_context(manager):
    mm, provider = manager
    scope = ContextVar("test_profile", default="wrong-profile")
    provider.on_pre_compress = lambda messages, **kwargs: scope.get()
    token = scope.set("correct-profile")
    try:
        assert mm.on_pre_compress([], require_checkpoint=True) == "correct-profile"
    finally:
        scope.reset(token)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_nested_boundary_cannot_roll_back_newer_binding_or_checkpoint_wrong_session(manager, checkpoint):
    mm, provider = manager
    started, release = threading.Event(), threading.Event()

    def nested():
        started.set()
        assert release.wait(2)
        if checkpoint:
            with pytest.raises(RuntimeError, match="cannot overtake"):
                mm.on_pre_compress([], require_checkpoint=True)
        else:
            mm.on_session_switch("newer")

    running = mm._submit_background(nested)
    assert started.wait(2)
    older = mm._memory_lifecycle.enqueue(lambda: mm._notify_session_switch("older"), transition=True)
    release.set()
    running.result(timeout=2)
    if checkpoint:
        older.result(timeout=2)
        assert mm.build_system_prompt() == "older"
    else:
        with pytest.raises(RuntimeError, match="superseded"):
            older.result(timeout=2)
        assert mm.build_system_prompt() == "newer"
    assert len(provider.calls) == 1
