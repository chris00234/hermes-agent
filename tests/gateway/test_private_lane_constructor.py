"""Private-lane constructor fence, stray-grant 400s, middleware default-deny.

DE_DESIGN §6.1 (grant honored only on POST /v1/runs), §6.3 (constructor
fence), §4 bind point 1 (middleware default-deny seed). No live generation,
no live ports.
"""

from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

import gateway.platforms.api_server as api_server_mod
from gateway.private_execution_policy import (
    bind_private_policy,
    build_policy,
    current_private_policy,
    reset_private_policy,
)
from tests.gateway.test_api_server import (
    _create_app,
    _make_adapter,
    _patch_create_agent_runtime,
)

GRANT_HEADER = "X-Hermes-Private-Grant"

PLATFORM_TOOLS = {"filesystem", "web", "shell", "memory", "delegate_task", "message_agent"}


def _policy():
    return build_policy(
        run_id="run-1",
        profile="default",
        execution_attempt_id="att-1",
        lease_expires_at="2099-01-01T00:00:00Z",
        allowed_toolsets=[
            "filesystem",
            "web",
            "not_configured",
            "memory",
            "delegate_task",
            "message_agent",
        ],
    )


def _prepare_create_agent(monkeypatch, captured: dict, observed: dict):
    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            observed["policy_during_init"] = current_private_policy()
            from hermes_cli.plugins import get_plugin_context_engine
            from plugins.context_engine import load_context_engine

            observed["plugin_engine"] = get_plugin_context_engine()
            observed["loaded_engine"] = load_context_engine("lcm")

    _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda *_: set(PLATFORM_TOOLS)
    )
    monkeypatch.setattr(
        "gateway.run._checkpoint_agent_kwargs",
        lambda cfg: {"checkpoints_enabled": True, "checkpoint_max_snapshots": 99},
    )
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    session_db_calls = []

    def _fake_session_db():
        session_db_calls.append(True)
        return "SESSION_DB"

    monkeypatch.setattr(adapter, "_ensure_session_db", _fake_session_db)
    return adapter, session_db_calls


class TestConstructorFence:
    def test_private_policy_rewrites_every_fenced_kwarg(self, monkeypatch):
        captured, observed = {}, {}
        adapter, session_db_calls = _prepare_create_agent(monkeypatch, captured, observed)
        policy = _policy()
        adapter._memory_sessions.checkout = MagicMock(side_effect=AssertionError("private memory checkout"))

        agent = adapter._create_agent(session_id="run-1", private_execution_policy=policy)

        assert captured["session_db"] is None
        assert session_db_calls == []
        assert captured["skip_memory"] is True
        assert captured["memory_manager"] is None
        adapter._memory_sessions.checkout.assert_not_called()
        assert captured["skip_background_review"] is True
        assert captured["save_trajectories"] is False
        assert captured["checkpoints_enabled"] is False
        assert "checkpoint_max_snapshots" not in captured
        assert captured["enabled_toolsets"] == ["filesystem", "web"]
        for excluded in ("memory", "delegate_task", "message_agent"):
            assert excluded not in captured["enabled_toolsets"]
        assert captured["quiet_mode"] is True
        # Policy bound during construction (belt-and-braces bind), reset after.
        assert observed["policy_during_init"] is policy
        assert current_private_policy() is None
        # Built-in compressor forced: both plugin-engine loaders refuse while
        # the policy is bound.
        assert observed["plugin_engine"] is None
        assert observed["loaded_engine"] is None
        # Post-construction flags + immutable policy stamp.
        assert agent._persist_disabled is True
        assert agent._session_json_enabled is False
        assert agent._end_session_on_close is False
        assert agent.suppress_status_output is True
        assert agent._private_execution_policy is policy

    def test_public_construction_is_untouched(self, monkeypatch):
        captured, observed = {}, {}
        adapter, session_db_calls = _prepare_create_agent(monkeypatch, captured, observed)

        agent = adapter._create_agent(session_id="s-public")

        assert captured["session_db"] == "SESSION_DB"
        assert session_db_calls == [True]
        assert captured["checkpoints_enabled"] is True
        assert captured["checkpoint_max_snapshots"] == 99
        assert "skip_memory" not in captured
        assert "skip_background_review" not in captured
        assert "save_trajectories" not in captured
        assert captured["enabled_toolsets"] == sorted(PLATFORM_TOOLS)
        assert observed["policy_during_init"] is None
        assert not hasattr(agent, "_private_execution_policy")
        assert not getattr(agent, "_persist_disabled", False)

    def test_engine_guard_installs_once_and_delegates_when_unbound(self, monkeypatch):
        api_server_mod._ensure_private_context_engine_guard()
        from hermes_cli import plugins as plugins_mod
        import plugins.context_engine as ce_mod

        guarded_get = plugins_mod.get_plugin_context_engine
        guarded_load = ce_mod.load_context_engine
        api_server_mod._ensure_private_context_engine_guard()
        assert plugins_mod.get_plugin_context_engine is guarded_get
        assert ce_mod.load_context_engine is guarded_load

        token = bind_private_policy(_policy())
        try:
            assert plugins_mod.get_plugin_context_engine() is None
            assert ce_mod.load_context_engine("anything") is None
        finally:
            reset_private_policy(token)
        # Unbound: load delegates to the original (unknown engine -> None via
        # the original's own not-found path, exercised without a policy).
        assert ce_mod.load_context_engine("definitely-not-an-engine") is None


class TestStrayGrantRejected:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/api/sessions/nope/chat",
            "/api/sessions/nope/chat/stream",
            "/v1/chat/completions",
            "/v1/responses",
        ],
    )
    async def test_agent_endpoints_400_reject_the_grant_header(self, path):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(path, json={}, headers={GRANT_HEADER: "v1.a.1.00"})
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == "private_grant_not_allowed"

            control = await cli.post(path, json={})
            control_body = await control.json()
            assert (
                (control_body.get("error") or {}).get("code")
                != "private_grant_not_allowed"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_key",
        [GRANT_HEADER, "x-hErmes-privATE-grant"],
    )
    async def test_agent_endpoints_400_reject_query_grants(self, query_key):
        # A query-delivered grant is the same stray-authority signal as the
        # header, and it additionally leaks into the path_qs audit log —
        # reject it case-insensitively (MultiDict keys are case-sensitive).
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/v1/chat/completions?{query_key}=v1.a.1.00", json={}
            )
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == "private_grant_not_allowed"

    @pytest.mark.asyncio
    async def test_run_job_400_rejects_the_grant_header(self, monkeypatch):
        adapter = _make_adapter()
        sentinel = web.json_response({"error": "auth"}, status=401)
        monkeypatch.setattr(adapter, "_check_auth", lambda request: sentinel)

        req = make_mocked_request(
            "POST", "/api/jobs/j1/run", headers={GRANT_HEADER: "v1.a.1.00"},
            match_info={"job_id": "j1"},
        )
        resp = await adapter._handle_run_job(req)
        assert resp.status == 400

        control = await adapter._handle_run_job(
            make_mocked_request("POST", "/api/jobs/j1/run", match_info={"job_id": "j1"})
        )
        assert control is sentinel

    @pytest.mark.asyncio
    async def test_cron_fire_400_rejects_the_grant_header(self):
        adapter = _make_adapter()
        req = make_mocked_request(
            "POST", "/api/cron/fire", headers={GRANT_HEADER: "v1.a.1.00"},
            match_info={},
        )
        resp = await adapter._handle_cron_fire(req)
        assert resp.status == 400


class TestMiddlewareDefaultDeny:
    @pytest.mark.asyncio
    async def test_seed_clears_inherited_policy_and_restores_it(self):
        adapter = _make_adapter()
        mw = adapter._make_profile_prefix_middleware()
        seen = []

        async def handler(request):
            seen.append(current_private_policy())
            return web.json_response({})

        outer = _policy()
        token = bind_private_policy(outer)
        try:
            req = make_mocked_request("GET", "/v1/test", match_info={})
            resp = await mw(req, handler)
            assert resp.status == 200
            # The default-deny seed masked the inherited policy for the
            # handler, and the middleware restored the outer binding.
            assert seen == [None]
            assert current_private_policy() is outer
        finally:
            reset_private_policy(token)

    @pytest.mark.asyncio
    async def test_slot_defaults_to_none_without_any_binding(self):
        adapter = _make_adapter()
        mw = adapter._make_profile_prefix_middleware()
        seen = []

        async def handler(request):
            seen.append(current_private_policy())
            return web.json_response({})

        await mw(make_mocked_request("GET", "/v1/test", match_info={}), handler)
        assert seen == [None]
