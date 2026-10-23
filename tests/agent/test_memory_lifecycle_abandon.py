"""Per-call lifecycle timeouts and fail-closed abandonment (contract B item 4).

A caller that passes an explicit per-call timeout fails closed: on timeout
the queued generation is abandoned and its fn must NEVER run later (a /reset
retirement executing after the command already failed would clear the frame
behind the user's back). Default-timeout callers keep the legacy stay-queued
behavior. Abandonment and ``StatePreservingResetError`` are transition-neutral
so the lifecycle stays usable afterwards.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent import memory_lifecycle, memory_manager
from agent.memory_lifecycle import MemoryLifecycle
from agent.memory_provider import StatePreservingResetError


@pytest.fixture
def lifecycle():
    executor = ThreadPoolExecutor(max_workers=1)
    lifecycle = MemoryLifecycle(lambda fn, **kwargs: executor.submit(fn))
    yield lifecycle
    executor.shutdown(wait=False, cancel_futures=True)


def _drain(lifecycle, timeout=2.0):
    """Wait until every queued lifecycle task has finished."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with lifecycle._lock:
            if not lifecycle._pending:
                return
        time.sleep(0.01)
    raise AssertionError("lifecycle queue did not drain")


def test_explicit_timeout_abandons_queued_fn(lifecycle):
    release = threading.Event()
    ran = []
    lifecycle.enqueue(lambda: release.wait(5), transition=False)

    with pytest.raises(TimeoutError):
        lifecycle.call(lambda: ran.append(1), transition=True, required=True,
                       timeout=0.05)

    release.set()
    _drain(lifecycle)
    assert ran == [], "abandoned lifecycle fn must never run"
    # Transition-neutral: the abandoned transition is not a failed one.
    assert lifecycle.read_generation() is not None
    assert lifecycle.call(lambda: "ok", transition=False, required=True) == "ok"


def test_default_timeout_keeps_task_queued(lifecycle, monkeypatch):
    monkeypatch.setattr(memory_lifecycle, "LIFECYCLE_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    ran = []
    lifecycle.enqueue(lambda: release.wait(5), transition=False)

    with pytest.raises(TimeoutError):
        lifecycle.call(lambda: ran.append(1), transition=True, required=True)

    release.set()
    _drain(lifecycle)
    assert ran == [1], "default-timeout callers keep legacy stay-queued behavior"


def test_explicit_timeout_large_enough_succeeds(lifecycle, monkeypatch):
    monkeypatch.setattr(memory_lifecycle, "LIFECYCLE_TIMEOUT_SECONDS", 0.05)
    result = lifecycle.call(lambda: time.sleep(0.2) or "done",
                            transition=True, required=True, timeout=2.0)
    assert result == "done"


def test_state_preserving_reset_error_is_transition_neutral(lifecycle):
    def fail():
        raise StatePreservingResetError("frame lifetime changed")

    with pytest.raises(StatePreservingResetError):
        lifecycle.call(fail, transition=True, required=True)

    assert lifecycle.read_generation() is not None
    assert lifecycle.call(lambda: "ok", transition=False, required=True) == "ok"


def test_fn_raising_timeout_error_is_not_abandonment(lifecycle):
    def fail():
        raise TimeoutError("provider http timeout")

    with pytest.raises(TimeoutError):
        lifecycle.call(fail, transition=True, required=True, timeout=2.0)

    # A genuine failure (not a wait timeout) still fences background work.
    assert lifecycle.read_generation() is None


def test_pre_session_reset_uses_nine_second_deadline():
    assert memory_manager._PRE_SESSION_RESET_TIMEOUT_S == 9.0


def test_shutdown_all_without_drain_does_not_wait():
    from tests.agent.test_memory_manager_pre_session_reset import _FrameProvider

    manager = memory_manager.MemoryManager()
    provider = _FrameProvider(sync_delay=0.0)
    started, release = threading.Event(), threading.Event()
    provider.sync_turn = lambda *a, **k: (started.set(), release.wait(5))
    manager.add_provider(provider)
    manager.initialize_all("sess-1")
    manager.sync_all("user", "turn", session_id="sess-1")
    assert started.wait(2)

    begin = time.monotonic()
    manager.shutdown_all(drain=False)
    elapsed = time.monotonic() - begin
    release.set()

    assert elapsed < 1.0, "drain=False must not wait out the 5s drain window"
    assert manager._shutting_down is True
