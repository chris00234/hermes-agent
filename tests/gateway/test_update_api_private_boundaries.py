"""Private grants on the extracted admission/task/executor paths."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server as api
from gateway.platforms import api_server_runs as runs
from gateway.private_execution_policy import (
    bind_private_policy,
    build_policy,
    current_private_policy,
    reset_private_policy,
)
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from tests.gateway.test_private_lane_runs import (
    GRANT_HEADER,
    SECRET,
    _done_agent,
    _grant,
    _wait_status,
)


@pytest.fixture(autouse=True)
def grant_secret(monkeypatch):
    monkeypatch.setenv("PRIVATE_EXEC_GRANT_KEY", SECRET)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra,grant,expected",
    [
        ({"previous_response_id": "stored"}, "invalid", 403),
        ({"hosted_room_dispatch": {}}, None, 400),
        ({"_room_execution_policy": {}}, None, 400),
        ({"previous_response_id": "stored"}, None, 400),
        ({"session_id": "stored"}, None, 400),
        ({"conversation_history": [{"role": "user", "content": "stored"}]}, None, 400),
        ({"allowed_toolsets": ["web"]}, None, 400),
    ],
)
async def test_private_denial_precedes_room_normalization(extra, grant, expected):
    adapter = _make_adapter()
    adapter._normalize_room_dispatch = AsyncMock(side_effect=AssertionError("normalization reached"))
    with patch.object(adapter, "_create_agent") as create:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs", json={"input": "private", **extra},
                headers={GRANT_HEADER: grant or _grant(), "Idempotency-Key": "denial"})
    assert response.status == expected
    adapter._normalize_room_dispatch.assert_not_called()
    create.assert_not_called()
    assert not adapter._run_owners
    assert not adapter._run_statuses
    assert not adapter._private_run_ids


@pytest.mark.asyncio
async def test_extracted_workers_isolate_concurrent_private_and_public_runs():
    adapter = _make_adapter()
    barrier = threading.Barrier(2)
    seen = {}
    agents = {}

    def create(**kwargs):
        policy = kwargs.get("private_execution_policy")
        name = "private" if policy else "public"
        seen[name + "-construction"] = current_private_policy()
        agent = agents[name] = _done_agent()

        def converse(**_kwargs):
            seen[name + "-worker"] = current_private_policy()
            barrier.wait(timeout=5)
            kwargs["stream_delta_callback"](name + "-delta")
            kwargs["tool_progress_callback"]("tool.started", "terminal", name + "-preview")
            return {"final_response": name + "-output"}

        agent.run_conversation.side_effect = converse
        return agent

    with patch.object(adapter, "_create_agent", side_effect=create):
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            private, public = await asyncio.gather(
                client.post("/v1/runs", json={"input": "private"}, headers={GRANT_HEADER: _grant()}),
                client.post("/v1/runs", json={"input": "public"}))
            assert private.status == public.status == 202
            private_id, public_id = (await private.json())["run_id"], (await public.json())["run_id"]
            await asyncio.gather(
                _wait_status(client, private_id, {"completed"}, {GRANT_HEADER: _grant()}),
                _wait_status(client, public_id, {"completed"}))
    assert seen["private-construction"] is seen["private-worker"]
    assert seen["private-worker"].run_id == private_id
    assert seen["public-construction"] is seen["public-worker"] is None
    assert current_private_policy() is None
    private_events = _drain(adapter._run_streams[private_id])
    public_events = _drain(adapter._run_streams[public_id])
    assert not any(event.get("event") == "message.delta" for event in private_events)
    assert any(event.get("delta") == "public-delta" for event in public_events)
    assert all("preview" not in event for event in private_events)
    assert any(event.get("preview") == "public-preview" for event in public_events)


def _drain(queue):
    events = []
    while not queue.empty():
        event = queue.get_nowait()
        if event is not None:
            events.append(event)
    return events


@pytest.mark.asyncio
async def test_repeated_task_cancellation_waits_for_private_executor_exit():
    adapter = _make_adapter()
    entered, release = threading.Event(), threading.Event()
    agent = _done_agent()

    def converse(**_kwargs):
        entered.set()
        assert release.wait(5)
        assert current_private_policy() is not None
        return {"final_response": "not-released-early"}

    agent.run_conversation.side_effect = converse
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(_create_runs_app(adapter))) as client:
                response = await client.post(
                    "/v1/runs", json={"input": "private"}, headers={GRANT_HEADER: _grant()})
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                assert await asyncio.to_thread(entered.wait, 3)
                task = adapter._active_run_tasks[run_id]
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0.02)
                    assert not task.done()
                    assert adapter._run_statuses[run_id]["status"] == "stopping"
                    assert adapter._active_run_agents[run_id] is agent
                    assert run_id in adapter._private_run_ids
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert adapter._run_statuses[run_id]["status"] == "cancelled"
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
    finally:
        release.set()
    assert current_private_policy() is None


def _launch(adapter, policy):
    return runs._RunLaunch(
        owner=adapter, run_id="run-context", queue=asyncio.Queue(), session_id="run-context",
        gateway_session_key=None, declared_selected=False, user_message="private",
        conversation_history=[], agent_kwargs={"room_dispatch": None}, request_profile=None,
        browser_control_principal=None, browser_control_transport_family=None, private_policy=policy)


def _policy():
    return build_policy(
        run_id="run-context", profile="default", execution_attempt_id="context",
        lease_expires_at="2099-01-01T00:00:00+00:00", allowed_toolsets=("shell",))


def test_worker_cleanup_failure_restores_reused_thread_policy():
    adapter, policy = _make_adapter(), _policy()
    agent = _done_agent()
    seen = []
    agent.run_conversation.side_effect = lambda **_kw: seen.append(current_private_policy()) or {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        with patch.object(api, "_clear_turn_process_ownership", side_effect=RuntimeError("cleanup")):
            with pytest.raises(RuntimeError, match="cleanup"):
                executor.submit(partial(
                    runs._run_agent_sync, adapter, _launch(adapter, policy), agent,
                    lambda _: None, _api_server=api)).result(timeout=5)
        assert executor.submit(current_private_policy).result(timeout=5) is None
        executor.submit(partial(
            runs._run_agent_sync, adapter, _launch(adapter, None), agent,
            lambda _: None, _api_server=api)).result(timeout=5)
        assert executor.submit(current_private_policy).result(timeout=5) is None
    assert seen == [policy, None]


def test_public_worker_clears_inherited_policy_and_restores_caller():
    adapter, inherited = _make_adapter(), _policy()
    agent = _done_agent()
    seen = []
    agent.run_conversation.side_effect = lambda **_kw: seen.append(current_private_policy()) or {}
    token = bind_private_policy(inherited)
    try:
        runs._run_agent_sync(adapter, _launch(adapter, None), agent, lambda _: None, _api_server=api)
        assert current_private_policy() is inherited
    finally:
        reset_private_policy(token)
    assert seen == [None]


def test_private_marker_survives_status_retirement_until_transport_is_gone():
    adapter = _make_adapter()
    run_id, now = "run-retirement", time.time()
    adapter._run_owners[run_id] = "owner"
    adapter._private_run_ids.add(run_id)
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._run_streams_created[run_id] = now
    adapter._run_statuses[run_id] = {
        "status": "completed", "updated_at": now - adapter._RUN_STATUS_TTL - 1}
    runs._sweep_orphaned_runs_once(adapter, now)
    assert run_id not in adapter._run_statuses
    assert run_id in adapter._private_run_ids
    assert run_id in adapter._run_owners
    runs._drop_run_transport(adapter, run_id)
    assert run_id not in adapter._private_run_ids
    assert run_id not in adapter._run_owners


def test_private_status_projection_preserves_hydrated_digest():
    projected = runs._project_private_run_status({"run_id": "run-digest", "output": "private"})
    assert projected["output_digest"]
    assert runs._project_private_run_status(projected) == projected
