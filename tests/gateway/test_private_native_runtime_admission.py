"""A valid private grant must not start a native runtime outside Hermes' policy boundary."""

from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _patch_create_agent_runtime
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from tests.gateway.test_private_lane_runs import GRANT_HEADER, SECRET, _grant, _wait_status


@pytest.mark.asyncio
async def test_valid_private_grant_with_native_runtime_fails_before_agent_construction(monkeypatch):
    constructor = Mock(side_effect=AssertionError("agent construction reached"))
    _patch_create_agent_runtime(monkeypatch, {}, constructor)
    monkeypatch.setenv("PRIVATE_EXEC_GRANT_KEY", SECRET)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "openai-codex", "api_key": "stub", "base_url": "https://stub.invalid",
        "api_mode": "codex_app_server",
    })
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    session_db = Mock(side_effect=AssertionError("private session DB reached"))
    monkeypatch.setattr(adapter, "_ensure_session_db", session_db)
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        response = await client.post(
            "/v1/runs", json={"input": "private"}, headers={GRANT_HEADER: _grant()},
        )
        assert response.status == 202
        run_id = (await response.json())["run_id"]
        assert run_id in adapter._private_run_ids
        status = await _wait_status(client, run_id, {"failed"}, {GRANT_HEADER: _grant()})
        assert status["error"] == "run_failed"
    constructor.assert_not_called()
    session_db.assert_not_called()
    assert not adapter._active_run_agents
