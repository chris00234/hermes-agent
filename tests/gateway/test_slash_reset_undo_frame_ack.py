"""/reset and /undo must retire the durable working-frame scope FIRST.

Contract B: when a durable acknowledged working-frame scope exists for the
route, the handlers call ``_retire_working_frame_scope`` before any
invalidation/rotation/rewind; a retirement failure blocks the command with a
user-visible message; a missing scope keeps legacy behavior; without a cached
agent the retirement builds a bare (never-initialized) provider.
"""
import asyncio
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _ref(**overrides):
    reference = {
        "provider": "frame",
        "principal": "chris",
        "agent": "jenna",
        "project": "proj",
        "session_id": "sess-old",
        "logical_task_id": "task-1",
        "revision": 3,
        "lifecycle_epoch": 1,
        "expires_at": 4102444800.0,
        "cleared": False,
    }
    reference.update(overrides)
    return reference


def _ack_for(reference):
    ack = dict(reference)
    ack["revision"] += 1
    ack["cleared"] = True
    return ack


class _FakeManager:
    def __init__(self, *, ack=None, exc=None, delay=0.0, order=None, supports=True):
        self.ack = ack
        self.exc = exc
        self.delay = delay
        self.order = order
        self.supports = supports
        self.calls = []
        self.timeouts = []

    def supports_pre_session_reset(self):
        return self.supports

    def on_pre_session_reset(self, session_id, *, reference, rewound=False, timeout=None):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append((session_id, rewound))
        self.timeouts.append(timeout)
        if self.order is not None:
            self.order.append("retire")
        if self.exc is not None:
            raise self.exc
        return dict(self.ack if self.ack is not None else _ack_for(reference))


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner(*, scope=None, manager=None):
    """Bare GatewayRunner with a fake session store and optional cached agent."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key, session_id="sess-old",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    new_entry = SessionEntry(
        session_key=session_key, session_id="sess-new",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    store = MagicMock()
    store.reset_session.return_value = new_entry
    store._entries = {session_key: session_entry}
    store._generate_session_key.return_value = session_key
    store.get_working_frame_scope.return_value = scope
    store.record_working_frame_scope.return_value = True
    store.get_or_create_session.return_value = session_entry
    runner.session_store = store
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    runner._agent_cache_lock = threading.RLock()
    agent = MagicMock()
    agent.close = lambda: None
    agent.shutdown_memory_provider = MagicMock()
    if manager is not None:
        agent._memory_manager = manager
        runner._agent_cache = {session_key: agent}
    else:
        runner._agent_cache = {}
    return runner, session_key


@pytest.mark.asyncio
async def test_reset_without_scope_keeps_legacy_behavior():
    runner, _ = _make_runner(scope=None)
    result = await runner._handle_reset_command(_make_event("/new"))
    assert result is not None
    runner.session_store.reset_session.assert_called_once()


@pytest.mark.asyncio
async def test_reset_retires_scope_before_invalidation():
    order = []
    manager = _FakeManager(order=order)
    runner, session_key = _make_runner(scope=_ref(), manager=manager)
    runner._invalidate_session_run_generation = (
        lambda key, reason="": order.append("invalidate") or 1
    )

    result = await runner._handle_reset_command(_make_event("/new"))

    assert manager.calls == [("sess-old", False)]
    assert order[:2] == ["retire", "invalidate"]
    assert result is not None
    runner.session_store.reset_session.assert_called_once()


@pytest.mark.asyncio
async def test_reset_blocked_when_retirement_fails():
    manager = _FakeManager(exc=RuntimeError("frame conflict"))
    runner, _ = _make_runner(scope=_ref(), manager=manager)
    invalidated = []
    runner._invalidate_session_run_generation = (
        lambda key, reason="": invalidated.append(key) or 1
    )

    result = await runner._handle_reset_command(_make_event("/new"))

    assert result == t("gateway.reset.memory_scope_failed")
    assert invalidated == []
    runner.session_store.reset_session.assert_not_called()


@pytest.mark.asyncio
async def test_reset_blocked_on_timeout(monkeypatch):
    import gateway.slash_commands as _sc

    monkeypatch.setattr(_sc, "_WORKING_FRAME_RETIRE_TIMEOUT_S", 0.05)
    manager = _FakeManager(delay=0.5)
    runner, _ = _make_runner(scope=_ref(), manager=manager)

    result = await runner._handle_reset_command(_make_event("/new"))

    assert result == t("gateway.reset.memory_scope_failed")
    runner.session_store.reset_session.assert_not_called()


@pytest.mark.asyncio
async def test_undo_retires_scope_before_rewind():
    manager = _FakeManager()
    runner, _ = _make_runner(scope=_ref(), manager=manager)
    runner.session_store.rewind_session.return_value = {
        "target_text": "hello", "turns_undone": 1, "rewound_count": 2,
    }

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert manager.calls == [("sess-old", True)]
    runner.session_store.rewind_session.assert_called_once()
    assert "hello" in result


@pytest.mark.asyncio
async def test_undo_blocked_when_retirement_fails():
    manager = _FakeManager(exc=RuntimeError("frame conflict"))
    runner, _ = _make_runner(scope=_ref(), manager=manager)

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert result == t("gateway.undo.memory_scope_failed")
    runner.session_store.rewind_session.assert_not_called()


@pytest.mark.asyncio
async def test_undo_without_scope_keeps_legacy_behavior():
    runner, _ = _make_runner(scope=None)
    runner.session_store.rewind_session.return_value = None

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert result == t("gateway.undo.nothing")
    runner.session_store.rewind_session.assert_called_once()


class _BareProvider:
    """Configured-provider stub for the no-cached-agent path."""

    pre_session_reset_api_version = 1
    name = "frame"

    def __init__(self):
        self.initialized = False
        self.reset_calls = []
        self.shutdowns = 0
        self.context = SimpleNamespace(profile=None)

    def initialize(self, *args, **kwargs):
        self.initialized = True

    def on_pre_session_reset(self, session_id, *, reference, rewound=False):
        self.reset_calls.append((session_id, rewound))
        return _ack_for(reference)

    def shutdown(self):
        self.shutdowns += 1


@pytest.mark.asyncio
async def test_reset_no_cache_path_builds_bare_provider(monkeypatch):
    provider = _BareProvider()
    monkeypatch.setattr(
        "tools.memory_tool.get_builtin_memory_config",
        lambda config=None: {"provider": "frame"},
    )
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider",
        lambda name, **kwargs: provider if name == "frame" else None,
    )

    runner, _ = _make_runner(scope=_ref(), manager=None)
    runner._active_profile_name = lambda: "jenna"

    result = await runner._handle_reset_command(_make_event("/new"))

    assert provider.reset_calls == [("sess-old", False)]
    assert provider.initialized is False, "bare provider must never be initialize()d"
    assert provider.context.profile == "jenna"
    assert provider.shutdowns == 1
    assert result is not None
    runner.session_store.reset_session.assert_called_once()
    assert runner.session_store.record_working_frame_scope.called


@pytest.mark.asyncio
async def test_reset_drops_orphaned_scope_when_memory_disabled(monkeypatch):
    """No configured provider can EVER retire this scope: 'try again in a
    moment' would trap the session until the scope expires (up to 7 days).
    The explicit unrecoverable determination drops the orphaned scope and
    legacy /reset applies."""
    monkeypatch.setattr(
        "tools.memory_tool.get_builtin_memory_config",
        lambda config=None: {"provider": ""},
    )
    runner, session_key = _make_runner(scope=_ref(), manager=None)
    runner.session_store.expire_working_frame_scope.return_value = True

    result = await runner._handle_reset_command(_make_event("/new"))

    runner.session_store.expire_working_frame_scope.assert_called_once_with(
        session_key, "sess-old"
    )
    assert result != t("gateway.reset.memory_scope_failed")
    runner.session_store.reset_session.assert_called_once()


@pytest.mark.asyncio
async def test_noop_undo_leaves_scope_untouched():
    manager = _FakeManager()
    runner, _ = _make_runner(scope=_ref(), manager=manager)
    runner.session_store.has_rewindable_user_turn.return_value = False

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert result == t("gateway.undo.nothing")
    assert manager.calls == [], "a no-op /undo must not retire the frame scope"
    runner.session_store.rewind_session.assert_not_called()


@pytest.mark.asyncio
async def test_undo_probes_then_retires_then_rewinds():
    order = []
    manager = _FakeManager(order=order)
    runner, _ = _make_runner(scope=_ref(), manager=manager)
    runner.session_store.has_rewindable_user_turn.side_effect = (
        lambda session_id: order.append("probe") or True
    )
    runner.session_store.rewind_session.side_effect = lambda session_id, n: (
        order.append("rewind") or {
            "target_text": "hello", "turns_undone": 1, "rewound_count": 2,
        }
    )

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert order == ["probe", "retire", "rewind"]
    assert "hello" in result


@pytest.mark.asyncio
async def test_reset_no_cache_blocked_when_profile_stamp_fails(monkeypatch):
    provider = _BareProvider()
    monkeypatch.setattr(
        "tools.memory_tool.get_builtin_memory_config",
        lambda config=None: {"provider": "frame"},
    )
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider",
        lambda name, **kwargs: provider if name == "frame" else None,
    )
    runner, _ = _make_runner(scope=_ref(), manager=None)

    def _boom():
        raise RuntimeError("no profile")

    runner._active_profile_name = _boom

    result = await runner._handle_reset_command(_make_event("/new"))

    assert result == t("gateway.reset.memory_scope_failed")
    assert provider.reset_calls == [], (
        "a bare provider without a serving-profile actor must fail closed"
    )
    runner.session_store.reset_session.assert_not_called()


def test_retire_skips_drain_when_retirement_times_out(monkeypatch):
    import gateway.working_frame_scope as wfs

    class _Manager:
        def __init__(self, exc=None):
            self.shutdowns = []
            self.exc = exc

        def on_pre_session_reset(self, session_id, *, reference, rewound=False):
            if self.exc is not None:
                raise self.exc
            return dict(reference)

        def shutdown_all(self, *, drain=True):
            self.shutdowns.append(drain)

    timed_out = _Manager(exc=TimeoutError())
    monkeypatch.setattr(wfs, "_resolve_reset_manager", lambda *a: (timed_out, True))
    with pytest.raises(TimeoutError):
        wfs.retire_working_frame_scope_sync(None, "key", {"session_id": "s"}, False)
    assert timed_out.shutdowns == [False], (
        "a timed-out retirement must not wait out the bare manager's 5s drain"
    )

    clean = _Manager()
    monkeypatch.setattr(wfs, "_resolve_reset_manager", lambda *a: (clean, True))
    wfs.retire_working_frame_scope_sync(None, "key", {"session_id": "s"}, False)
    assert clean.shutdowns == [True]


@pytest.mark.asyncio
async def test_undo_reports_frame_reset_when_rewind_finds_nothing():
    """Retirement succeeded (durable frame cleared server-side) but the
    rewind then found no turn: the user must NOT be told nothing happened."""
    manager = _FakeManager()
    runner, _ = _make_runner(scope=_ref(), manager=manager)
    runner.session_store.has_rewindable_user_turn.return_value = True
    runner.session_store.rewind_session.return_value = None

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert manager.calls == [("sess-old", True)]
    assert result == t("gateway.undo.frame_reset_without_rewind")
    assert result != t("gateway.undo.nothing")


@pytest.mark.asyncio
async def test_lagged_retirement_never_clears_after_fail_closed_reply(monkeypatch):
    """Executor lag past the gateway's fail-closed reply must not let the
    retirement clear the frame afterwards (abandoned-then-executed)."""
    import gateway.slash_commands as _sc

    monkeypatch.setattr(_sc, "_WORKING_FRAME_RETIRE_TIMEOUT_S", 0.05)
    manager = _FakeManager()
    runner, _ = _make_runner(scope=_ref(), manager=manager)

    async def lagged_executor(func, *args):
        loop = asyncio.get_running_loop()

        def run():
            time.sleep(0.3)  # queue delay: the thread starts past the deadline
            return func(*args)

        return await loop.run_in_executor(None, run)

    runner._run_in_executor_with_context = lagged_executor

    result = await runner._handle_reset_command(_make_event("/new"))

    assert result == t("gateway.reset.memory_scope_failed")
    runner.session_store.reset_session.assert_not_called()
    await asyncio.sleep(0.5)  # let the abandoned worker thread finish
    assert manager.calls == [], (
        "a retirement that outlives the gateway's fail-closed reply must "
        "never reach the manager and clear the frame"
    )


@pytest.mark.asyncio
async def test_retirement_passes_remaining_deadline_to_manager():
    import gateway.slash_commands as _sc

    manager = _FakeManager()
    runner, session_key = _make_runner(scope=_ref(), manager=manager)

    ack = await runner._retire_working_frame_scope(session_key, rewound=False)

    assert isinstance(ack, dict)
    assert len(manager.timeouts) == 1
    assert manager.timeouts[0] is not None
    assert 0 < manager.timeouts[0] <= _sc._WORKING_FRAME_RETIRE_TIMEOUT_S


def test_retire_sync_raises_before_manager_when_deadline_passed(monkeypatch):
    import gateway.working_frame_scope as wfs
    from agent.memory_provider import StatePreservingResetError

    resolved = []
    monkeypatch.setattr(
        wfs, "_resolve_reset_manager", lambda *a: resolved.append(a) or (None, False)
    )
    with pytest.raises(StatePreservingResetError):
        wfs.retire_working_frame_scope_sync(
            None, "key", {"session_id": "s"}, False, time.monotonic() - 0.01
        )
    assert resolved == [], "an expired deadline must never resolve a manager"


@pytest.mark.asyncio
async def test_reset_drops_orphaned_scope_when_configured_provider_differs(monkeypatch):
    monkeypatch.setattr(
        "tools.memory_tool.get_builtin_memory_config",
        lambda config=None: {"provider": "other"},
    )
    loaded = []
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider",
        lambda name, **kwargs: loaded.append(name) or None,
    )
    runner, session_key = _make_runner(scope=_ref(), manager=None)
    runner._active_profile_name = lambda: "jenna"
    runner.session_store.expire_working_frame_scope.return_value = True

    result = await runner._handle_reset_command(_make_event("/new"))

    runner.session_store.expire_working_frame_scope.assert_called_once_with(
        session_key, "sess-old"
    )
    assert result != t("gateway.reset.memory_scope_failed")
    runner.session_store.reset_session.assert_called_once()
    assert loaded == [], "a provider that cannot serve the scope must not be loaded"


@pytest.mark.asyncio
async def test_undo_drops_orphaned_scope_and_proceeds_legacy(monkeypatch):
    monkeypatch.setattr(
        "tools.memory_tool.get_builtin_memory_config",
        lambda config=None: {"provider": "other"},
    )
    runner, session_key = _make_runner(scope=_ref(), manager=None)
    runner._active_profile_name = lambda: "jenna"
    runner.session_store.has_rewindable_user_turn.return_value = True
    runner.session_store.expire_working_frame_scope.return_value = True
    runner.session_store.rewind_session.return_value = {
        "target_text": "hello", "turns_undone": 1, "rewound_count": 2,
    }

    result = await runner._handle_undo_command(_make_event("/undo"))

    runner.session_store.expire_working_frame_scope.assert_called_once_with(
        session_key, "sess-old"
    )
    assert "hello" in result


@pytest.mark.asyncio
async def test_reset_stays_blocked_when_matching_provider_fails_to_load(monkeypatch):
    """Same configured provider, transient load failure: fail closed, keep
    the scope — this is NOT the unrecoverable orphan case."""
    monkeypatch.setattr(
        "tools.memory_tool.get_builtin_memory_config",
        lambda config=None: {"provider": "frame"},
    )
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider", lambda name, **kwargs: None
    )
    runner, _ = _make_runner(scope=_ref(), manager=None)
    runner._active_profile_name = lambda: "jenna"

    result = await runner._handle_reset_command(_make_event("/new"))

    assert result == t("gateway.reset.memory_scope_failed")
    runner.session_store.reset_session.assert_not_called()
    runner.session_store.expire_working_frame_scope.assert_not_called()
