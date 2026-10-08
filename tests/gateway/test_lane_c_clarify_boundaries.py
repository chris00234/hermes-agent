"""Required clarify boundaries extracted outside lane C's source write grant.

These are regression tests, not xfails: the integration owner must port the
missing behavior into run_adapters/run_agent_cache before a release gate.
"""

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from tools import clarify_gateway


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [False, True])
async def test_gateway_authorizes_native_candidate_before_observer(monkeypatch, authorized):
    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized_for_source = lambda source: authorized
    accepted = object()
    calls = []
    source = SimpleNamespace()
    event = {"event_type": "clarify_callback_candidate", "platform": "telegram"}
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr("gateway.telegram_clarify_evidence.authorize_candidate", lambda event, source: accepted)
    result = await runner._handle_gateway_platform_event(event, source)
    assert result is (accepted if authorized else None)
    assert calls == [], "raw native candidates must not reach generic observers"


@pytest.mark.asyncio
async def test_public_platform_events_keep_observer_delivery(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized_for_source = lambda source: True
    calls = []
    event = {"event_type": "public_update", "platform": "telegram"}
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: calls.append(kw))
    await runner._handle_gateway_platform_event(event, SimpleNamespace())
    assert calls == [event]


def test_generation_invalidation_cancels_pending_clarify():
    runner = object.__new__(GatewayRunner)
    key = "lane-c-generation-boundary"
    entry = clarify_gateway.register("lane-c-boundary", key, "Continue?", ["yes", "no"])
    try:
        runner._invalidate_session_run_generation(key, reason="session_reset")
        assert not clarify_gateway.has_pending(key)
        assert entry.event.is_set()
        assert entry.response == clarify_gateway.CANCELLED
        assert clarify_gateway.resolve_gateway_clarify(entry.clarify_id, "yes") is False
    finally:
        clarify_gateway.clear_session(key)
