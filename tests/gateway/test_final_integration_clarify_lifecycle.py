"""Clarify lifetime controls on the actual extracted gateway mixins."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from tools import clarify_gateway


@pytest.mark.asyncio
async def test_native_authorization_does_not_require_observer_hook(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized_for_source = lambda source: True
    accepted = object()
    authorize = Mock(return_value=accepted)
    invoke = Mock()
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    monkeypatch.setattr("gateway.telegram_clarify_evidence.authorize_candidate", authorize)
    source = SimpleNamespace()
    event = {"event_type": "clarify_callback_candidate"}
    assert await runner._handle_gateway_platform_event(event, source) is accepted
    authorize.assert_called_once_with(event, source)
    invoke.assert_not_called()


@pytest.mark.parametrize("state", ["idle", "running", "pending", "absent"])
def test_eviction_only_cancels_eligible_agent_clarify(state):
    runner = object.__new__(GatewayRunner)
    agent = object()
    runner._peek_session_state = lambda key: None
    runner._agent_cache_lock = None
    runner._agent_cache = {} if state == "absent" else {
        "key": _AGENT_PENDING_SENTINEL if state == "pending" else agent,
    }
    runner._running_agent_ids = lambda: {id(agent)} if state == "running" else set()
    runner._spawn_release_thread = Mock()
    entry = clarify_gateway.register("eviction-control", "key", "Continue?", ["yes", "no"])
    try:
        runner._evict_cached_agent("key")
        assert entry.event.is_set() is (state == "idle")
        assert clarify_gateway.has_pending("key") is (state != "idle")
        runner._evict_cached_agent("key")
        assert runner._spawn_release_thread.call_count == (1 if state == "idle" else 0)
    finally:
        clarify_gateway.clear_session("key")
