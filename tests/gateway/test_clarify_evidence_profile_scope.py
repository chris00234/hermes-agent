"""Multiplex invariant: clarify evidence closed by shutdown or cached-agent eviction reaches the
OWNING profile's plugins.

``ClarifyEvidence.close`` emits ``prompt_closed`` through ``invoke_hook``, which picks the plugin
manager from the ambient HERMES_HOME. Shutdown runs unscoped and an eviction caller may hold another
profile's scope, so the owning profile's evidence plugin never received ``prompt_closed``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools import clarify_gateway
from tools.clarify_evidence import ClarifyEvidence

_KEY = "agent:b:telegram:dm:1"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    prof_a, prof_b = default_home / "profiles" / "a", default_home / "profiles" / "b"
    prof_a.mkdir(parents=True)
    prof_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    yield prof_a, prof_b
    secret_scope.set_multiplex_active(False)


@pytest.fixture
def hook_homes(monkeypatch):
    seen = []

    def record(name, **kwargs):
        seen.append((kwargs["payload"]["kind"], get_hermes_home()))
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", record)
    return seen


@pytest.fixture
def pending(hook_homes):
    evidence = ClarifyEvidence("cid-b", "b", "Pick?", ["A", "B"], 600)
    evidence.delivered(question="Pick?", options=["A", "B"], rendered_text="Pick?",
                       bot_id="900", chat_id="1", thread_id=None, message_id="100")
    yield clarify_gateway.register("cid-b", _KEY, "Pick?", ["A", "B"], evidence=evidence)
    clarify_gateway.clear_session(_KEY)


def _runner(prof_b):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_store = SimpleNamespace(_profile_home_for_key={_KEY: prof_b}.get)
    return runner


@pytest.mark.asyncio
async def test_shutdown_closes_clarify_evidence_under_the_owning_profile(homes, hook_homes, pending):
    _, prof_b = homes
    runner = _runner(prof_b)
    runner._finalize_session_off_loop = AsyncMock()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    # Shutdown runs on the main loop with NO scope installed.
    assert secret_scope.current_secret_scope() is None
    await runner._finalize_shutdown_agents({_KEY: object()})
    assert pending.event.is_set()
    assert hook_homes[-1] == ("prompt_closed", prof_b)


def test_eviction_from_another_profiles_scope_closes_clarify_evidence_under_the_owner(
    homes, hook_homes, pending,
):
    prof_a, prof_b = homes
    runner = _runner(prof_b)
    runner._agent_cache_lock = None
    runner._agent_cache = {_KEY: object()}
    runner._peek_session_state = lambda key: None
    runner._running_agent_ids = lambda: set()
    runner._spawn_release_thread = Mock()
    # /login's free-tier sweep evicts every profile's cached sessions from profile A's scope.
    home_token = set_hermes_home_override(str(prof_a))
    scope_token = secret_scope.set_secret_scope({"MARKER": "profile-a"})
    try:
        runner._evict_cached_agent(_KEY)
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
    assert pending.event.is_set()
    assert hook_homes[-1] == ("prompt_closed", prof_b)
