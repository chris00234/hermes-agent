"""Real HTTP renewal reaches the policy already held by an executing worker."""

import asyncio
import threading
import time
from datetime import datetime, timezone, UTC
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_runs
from gateway.private_execution_policy import GrantError, build_policy, mint_grant, renew_private_lease
from tests.gateway.test_api_server_runs import _make_adapter

SECRET = "renewal-test-secret-0123456789abcdef"
HEADER = "X-Hermes-Private-Grant"


@pytest.mark.asyncio
async def test_http_renewal_reaches_running_tool_context_without_widening_authority(monkeypatch):
    import agent.tool_executor as executor
    from tools.registry import registry

    monkeypatch.setenv("PRIVATE_EXEC_GRANT_KEY", SECRET)
    monkeypatch.setattr(registry, "get_toolset_for_tool", lambda _: "file")
    adapter = _make_adapter(api_key="test-api-key")
    app = web.Application()
    for method, path, handler in api_server_runs._http_routes(adapter):
        app.router.add_route(method, path, handler)
    finish = threading.Event()
    ready = threading.Event()
    probe = threading.Event()
    observed = []
    captured = []
    original_expiry = int(time.time()) + 120

    class ToolClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(original_expiry + 1, UTC)

    monkeypatch.setattr(executor, "datetime", ToolClock)

    def create_agent(**kwargs):
        policy = kwargs["private_execution_policy"]
        captured.append(policy)
        agent = SimpleNamespace(_private_execution_policy=policy, session_prompt_tokens=0,
                                session_completion_tokens=0, session_total_tokens=0, interrupt=lambda *a: finish.set())

        def run_conversation(**kwargs):
            observed.append(executor._private_execution_block_message(agent, "read_file"))
            ready.set()
            if probe.wait(5):
                observed.append(executor._private_execution_block_message(agent, "read_file"))
            finish.wait(5)
            return {"final_response": "done"}

        agent.run_conversation = run_conversation
        return agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)

    def headers(expiry, *, attempt="attempt-1", toolsets=("file",)):
        return {"Authorization": "Bearer test-api-key", HEADER: mint_grant(
            SECRET, attempt_id=attempt, expiry=expiry, profile="default", toolsets=toolsets)}

    async with TestClient(TestServer(app)) as client:
        try:
            response = await client.post("/v1/runs", json={"input": "read"}, headers=headers(original_expiry))
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            assert await asyncio.to_thread(ready.wait, 3)
            policy = captured[0]
            digest = policy.policy_digest
            url = f"/v1/runs/{run_id}/lease"
            expiry = int(time.time()) + 900
            assert (await client.post(url)).status == 401
            assert (await client.post(url, headers=headers(expiry, attempt="other"))).status == 404
            assert (await client.post(url, headers=headers(expiry, toolsets=("file", "terminal")))).status == 403
            for _ in range(2):  # A lost response may replay the same renewal.
                response = await client.post(url, headers=headers(expiry))
                assert response.status == 200
                assert (await response.json())["execution_attempt_id"] == "attempt-1"
            assert (await client.post(url, headers=headers(original_expiry))).status == 403
            assert policy.policy_digest == digest
            assert policy.allowed_toolsets == frozenset({"file"})
            probe.set()
            finish.set()
            for _ in range(60):
                response = await client.get(f"/v1/runs/{run_id}", headers=headers(expiry))
                if (await response.json())["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            assert observed == ["Tool blocked: private execution lease expired", None]
            assert (await client.post(url, headers=headers(expiry))).status == 409
        finally:
            probe.set()
            finish.set()


@pytest.mark.parametrize("case", ["expired", "profile", "attempt", "toolsets", "unbounded", "bad_signature"])
def test_renewal_cannot_revive_or_reauthorize_a_policy(case):
    now = time.time()
    policy = build_policy(run_id="run", profile="liz", execution_attempt_id="attempt", allowed_toolsets=("file",),
                          lease_expires_at=datetime.fromtimestamp(now + (-1 if case == "expired" else 300), UTC).isoformat())
    grant = mint_grant(SECRET, attempt_id="other" if case == "attempt" else "attempt",
                       profile="ellie" if case == "profile" else "liz",
                       expiry=int(now) + (1801 if case == "unbounded" else 900),
                       toolsets=("file", "terminal") if case == "toolsets" else ("file",))
    with pytest.raises(GrantError):
        renew_private_lease(policy, grant + ("x" if case == "bad_signature" else ""), SECRET,
                            profile="ellie" if case == "profile" else "liz", now=now)
