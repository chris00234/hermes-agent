"""Exercise extracted run functions without importing the API facade."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import ContextVar
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms import api_server_runs as runs
from gateway.private_execution_policy import build_policy, current_private_policy, mint_grant
from gateway.session_context import get_session_env, set_session_vars

SECRET = "lane-b-test-signing-key-not-a-credential"
NOW = 1_800_000_000


def _api():
    return SimpleNamespace(
        _api_request_profile=ContextVar("lane-b-profile", default="default"),
        _openai_error=lambda message, **kw: {"error": {"message": message, **kw}},
        _publish_turn_process_ownership=Mock(),
        _clear_turn_process_ownership=Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize("grant", ["invalid", "expired", "wrong-profile", "v1"])
async def test_invalid_grant_before_session_parsing_or_json(grant, monkeypatch):
    token = mint_grant(
        SECRET, attempt_id="lane-b", expiry=NOW - 1 if grant == "expired" else NOW + 60,
        profile="other" if grant == "wrong-profile" else "default", toolsets=("shell",))
    if grant == "invalid":
        token = "invalid"
    if grant == "v1":
        token = "v1" + token[2:]
    monkeypatch.setattr(runs, "resolve_private_grant_secret", lambda _: SECRET)
    monkeypatch.setattr(runs.time, "time", lambda: NOW)
    request = SimpleNamespace(headers={"X-Hermes-Private-Grant": token}, query={}, json=AsyncMock())
    adapter = SimpleNamespace(_room_grant_token=lambda _: None, _parse_session_key_header=Mock())
    response = await runs._handle_runs(adapter, request, _api_server=_api())
    assert response.status == 403
    request.json.assert_not_called()
    adapter._parse_session_key_header.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"hosted_room_dispatch": None}, {"_room_execution_policy": None},
    {"session_id": "stored"}, {"previous_response_id": "stored"},
    {"conversation_history": [{"role": "user", "content": "stored"}]},
    {"allowed_toolsets": ["web"]}, {"allowed_toolsets": "shell"},
])
async def test_private_body_denial_before_normalization(body, monkeypatch):
    token = mint_grant(SECRET, attempt_id="lane-b", expiry=NOW + 60, profile="default", toolsets=("shell",))
    monkeypatch.setattr(runs, "resolve_private_grant_secret", lambda _: SECRET)
    monkeypatch.setattr(runs.time, "time", lambda: NOW)
    request = SimpleNamespace(
        headers={"X-Hermes-Private-Grant": token}, query={}, json=AsyncMock(return_value={"input": "x", **body}))
    normalize = AsyncMock(side_effect=AssertionError("side effect"))
    adapter = SimpleNamespace(
        _room_grant_token=lambda _: None, _parse_session_key_header=lambda _: (None, None),
        _normalize_room_dispatch=normalize)
    response = await runs._handle_runs(adapter, request, _api_server=_api())
    assert response.status == 400
    normalize.assert_not_called()


def _launch(adapter, policy):
    return runs._RunLaunch(
        owner=adapter, run_id="run-lane-b", queue=asyncio.Queue(), session_id="run-lane-b",
        gateway_session_key=None, declared_selected=False, user_message="synthetic private test",
        conversation_history=[], session_history_delivery=False, agent_kwargs={"room_dispatch": None},
        request_profile=None,
        private_policy=policy)


def _bind_session(**kwargs):
    return set_session_vars(
        platform="api_server", chat_id=kwargs["chat_id"],
        session_key=kwargs["session_key"], session_id=kwargs["session_id"])


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_actual_executor_body_resets_policy_and_session_on_reuse(cleanup_fails):
    policy = build_policy(
        run_id="run-lane-b", profile="default", execution_attempt_id="lane-b",
        lease_expires_at="2099-01-01T00:00:00+00:00", allowed_toolsets=("shell",))
    adapter = SimpleNamespace(_profile_scope=lambda _: nullcontext(), _bind_api_server_session=_bind_session, _memory_sessions=Mock())
    seen = []

    def converse(**kwargs):
        seen.append((current_private_policy(), get_session_env("HERMES_SESSION_ID"), kwargs["task_id"]))
        return {"final_response": "test"}

    agent = SimpleNamespace(run_conversation=converse)
    facade = _api()
    if cleanup_fails:
        facade._clear_turn_process_ownership.side_effect = RuntimeError("cleanup")
    with ThreadPoolExecutor(max_workers=1) as executor:
        call = partial(runs._run_agent_sync, adapter, _launch(adapter, policy), agent, lambda _: None, _api_server=facade)
        future = executor.submit(call)
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="cleanup"):
                future.result(timeout=5)
        else:
            assert future.result(timeout=5)[0]["final_response"] == "test"
        assert executor.submit(current_private_policy).result(timeout=5) is None
        assert executor.submit(get_session_env, "HERMES_SESSION_ID").result(timeout=5) != "run-lane-b"
        facade._clear_turn_process_ownership.side_effect = None
        executor.submit(partial(
            runs._run_agent_sync, adapter, _launch(adapter, None), agent, lambda _: None,
            _api_server=facade)).result(timeout=5)
        assert executor.submit(current_private_policy).result(timeout=5) is None
    assert seen == [(policy, "run-lane-b", "run-lane-b"), (None, "run-lane-b", "run-lane-b")]
