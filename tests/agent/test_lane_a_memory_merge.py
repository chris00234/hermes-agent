"""Exercise the merged provider helpers and their real serialized worker."""

import threading
from concurrent.futures import Future
from contextvars import ContextVar
from unittest.mock import patch

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, RecallStatus

WAIT_SECONDS = 2


class LaneProvider(MemoryProvider):
    pre_compress_checkpoint_api_version = 2
    pre_session_reset_api_version = 1

    @property
    def name(self):
        return "lane-a"

    def __init__(self):
        self.events = []
        self.reference = {
            "provider": self.name, "principal": "test", "agent": "test",
            "project": "test", "session_id": "old", "logical_task_id": "task",
            "revision": 1, "lifecycle_epoch": 1, "expires_at": 4102444800.0,
            "cleared": False,
        }

    def is_available(self):
        return True

    def initialize(self, session_id="", **kwargs):
        pass

    def get_tool_schemas(self):
        return [{"name": "lane_a_recall", "parameters": {"type": "object"}}]

    def working_frame_reference(self):
        return dict(self.reference)

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self.events.append(("sync", messages))

    def queue_prefetch(self, query, *, session_id=""):
        self.events.append(("prefetch", query))

    def on_pre_compress(self, messages, *, require_checkpoint=False):
        self.events.append(("checkpoint", messages))
        return "checkpoint"

    def on_session_switch(self, new_session_id, **kwargs):
        self.reference["session_id"] = new_session_id
        self.events.append(("switch", new_session_id))

    def on_pre_session_reset(self, session_id, *, reference, rewound=False):
        self.events.append(("reset", session_id))
        self.reference.update(revision=reference["revision"] + 1, cleared=True)
        return dict(self.reference)


@pytest.fixture
def manager():
    manager = MemoryManager()
    provider = LaneProvider()
    manager.add_provider(provider)
    yield manager, provider
    manager.shutdown_all()


def hold_worker(manager):
    started, release = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert release.wait(WAIT_SECONDS)

    future = manager._submit_background(hold)
    assert started.wait(WAIT_SECONDS)
    return release, future


def test_sync_snapshot_ack_and_prefetch_share_fifo(manager):
    mm, provider = manager
    release, held = hold_worker(mm)
    messages = [{"role": "user", "content": "original"}]
    expected = list(messages)
    mm.set_working_frame_scope_sink(lambda ref: provider.events.append(("ack", ref)))
    try:
        mm.sync_all("question", "answer", session_id="old", messages=messages)
        mm.queue_prefetch_all("next", session_id="old")
        messages.clear()
        with mm._sync_executor_lock:
            assert sorted(mm._background_futures.values()) == ["prefetch", "write", "write"]
    finally:
        release.set()
    held.result(timeout=WAIT_SECONDS)
    assert mm.flush_pending(timeout=WAIT_SECONDS)
    assert [event[0] for event in provider.events] == ["sync", "ack", "prefetch"]
    assert provider.events[0][1] == expected
    with mm._sync_executor_lock:
        assert not mm._background_futures


@pytest.mark.parametrize("surface", ["prompt", "prefetch", "status", "tool"])
def test_reads_started_before_a_switch_discard_the_result(manager, surface):
    mm, provider = manager

    def change_session(*args, **kwargs):
        mm.on_session_switch("new")
        return RecallStatus("Lane", 1) if surface == "status" else "stale"

    bindings = {
        "prompt": ("system_prompt_block", mm.build_system_prompt, ()),
        "prefetch": ("prefetch", mm.prefetch_all, ("question",)),
        "status": ("recall_status", mm.describe_recall, ()),
        "tool": ("handle_tool_call", mm.handle_tool_call, ("lane_a_recall", {})),
    }
    attribute, call, args = bindings[surface]
    setattr(provider, attribute, change_session)
    result = call(*args)
    assert "session changed" in result if surface == "tool" else result == ""


def test_pending_transition_suppresses_status_and_turn_start(manager):
    mm, provider = manager
    release, held = hold_worker(mm)
    pending = mm._memory_lifecycle.enqueue(lambda: None, transition=True)
    provider.recall_status = lambda: pytest.fail("pending status was read")
    provider.on_turn_start = lambda *args: pytest.fail("pending turn was admitted")
    try:
        assert mm.describe_recall() == ""
        mm.on_turn_start(1, "question")
    finally:
        release.set()
    held.result(timeout=WAIT_SECONDS)
    pending.result(timeout=WAIT_SECONDS)


def test_worker_future_preserves_context_and_nested_checkpoint(manager):
    mm, provider = manager
    scope = ContextVar("lane_a_scope", default="wrong")
    provider.on_pre_compress = lambda *args, **kwargs: scope.get()
    token = scope.set("caller")
    try:
        future = mm._submit_background(lambda: mm.on_pre_compress([], require_checkpoint=True))
    finally:
        scope.reset(token)
    assert isinstance(future, Future)
    assert future.result(timeout=WAIT_SECONDS) == "caller"
    assert scope.get() == "wrong"


@pytest.mark.parametrize("failure", ["creation", "submission"])
def test_worker_unavailable_never_fakes_lifecycle_ack(manager, failure):
    mm, provider = manager
    target = "tools.daemon_pool.DaemonThreadPoolExecutor"
    if failure == "submission":
        class RejectedExecutor:
            def submit(self, fn):
                raise RuntimeError("closed")

            def shutdown(self, **kwargs):
                pass

        mm._sync_executor = RejectedExecutor()
    with patch(target, side_effect=RuntimeError("unavailable")):
        with pytest.raises(RuntimeError, match="worker is unavailable"):
            mm.on_pre_compress([], require_checkpoint=True)
        assert provider.events == []
        # Best-effort writes keep the upstream inline fail-safe.
        mm.sync_all("question", "answer")
    assert provider.events == [("sync", None)]


@pytest.mark.parametrize("operation", ["checkpoint", "switch"])
def test_scope_sink_refusal_is_not_a_successful_boundary(manager, operation):
    mm, provider = manager
    mm.set_working_frame_scope_sink(lambda ref: False)
    if operation == "checkpoint":
        with pytest.raises(RuntimeError, match="durably acknowledged"):
            mm.on_pre_compress([], require_checkpoint=True)
    else:
        mm.on_session_switch("new")
        assert mm._memory_lifecycle.read_generation() is None
        mm.set_working_frame_scope_sink(lambda ref: True)
        mm.on_session_switch("new")
        assert mm._memory_lifecycle.read_generation() is not None


def test_explicit_deadline_after_start_keeps_reads_fenced_until_durable_ack(manager):
    mm, provider = manager
    release, started = threading.Event(), threading.Event()
    original = provider.on_pre_session_reset
    recorded = []

    def reset(*args, **kwargs):
        started.set()
        assert release.wait(WAIT_SECONDS)
        return original(*args, **kwargs)

    provider.on_pre_session_reset = reset
    mm.set_working_frame_scope_sink(lambda ref: recorded.append(ref))
    try:
        with pytest.raises(TimeoutError):
            mm.on_pre_session_reset("old", reference=provider.reference, timeout=0.05)
        assert started.is_set()
        assert mm._memory_lifecycle.read_generation() is None
        assert recorded == []
    finally:
        release.set()
    assert mm.flush_pending(timeout=WAIT_SECONDS)
    assert recorded[0]["cleared"] is True
    assert mm._memory_lifecycle.read_generation() is not None


def test_expired_reset_deadline_cannot_execute_on_the_worker(manager):
    mm, provider = manager
    future = mm._submit_background(
        lambda: mm.on_pre_session_reset("old", reference=provider.reference, timeout=0),
    )
    with pytest.raises(TimeoutError):
        future.result(timeout=WAIT_SECONDS)
    assert provider.events == []
    assert mm._memory_lifecycle.read_generation() is not None
