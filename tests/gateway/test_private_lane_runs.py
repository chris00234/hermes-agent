"""Private /v1/runs lane: grant admission, linkage rejection, egress gating.

Covers DE_DESIGN §6.1, §6.2, §6.5, §6.6, §6.11, §6.12 for the
api_server_runs.py side of the private native execution lane.
"""

import asyncio
import contextlib
import json
import logging
import sqlite3
import threading
import time
from hashlib import sha256
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

import gateway.platforms.api_server as api_server_mod
from gateway.platforms import api_server_runs as runs_mod
from gateway.private_execution_policy import (
    bind_private_policy,
    build_policy,
    mint_grant,
    reset_private_policy,
)
from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _make_slow_agent,
    _use_idempotency_db,
)
from tools import approval as approval_mod

SECRET = "private-grant-secret-0123456789"
GRANT_HEADER = "X-Hermes-Private-Grant"


GRANT_TOOLSETS = ("shell",)


def _grant(
    attempt_id: str = "attempt-1",
    expiry_delta: int = 600,
    toolsets: tuple = GRANT_TOOLSETS,
) -> str:
    return mint_grant(
        SECRET,
        attempt_id=attempt_id,
        expiry=int(time.time()) + expiry_delta,
        profile="default",
        toolsets=toolsets,
    )


def _done_agent(output: str = "done"):
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": output}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


async def _wait_status(cli, run_id, want, headers=None):
    for _ in range(60):
        resp = await cli.get(f"/v1/runs/{run_id}", headers=headers or {})
        if resp.status == 200:
            body = await resp.json()
            if body["status"] in want:
                return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {want}")


def _status_rows(path):
    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        return conn.execute(
            "SELECT run_id, status_json FROM run_idempotency"
        ).fetchall()


@pytest.fixture(autouse=True)
def grant_secret(monkeypatch):
    monkeypatch.setenv("PRIVATE_EXEC_GRANT_KEY", SECRET)


@pytest.fixture
def adapter():
    return _make_adapter()


class TestGrantAdmission:
    @pytest.mark.asyncio
    async def test_valid_grant_starts_run_and_binds_policy(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: _grant(toolsets=("web", "shell"))},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_status(
                    cli, run_id, {"completed"}, headers={GRANT_HEADER: _grant()}
                )

        assert run_id in adapter._private_run_ids
        policy = create.call_args.kwargs["private_execution_policy"]
        assert policy.run_id == run_id
        assert policy.execution_attempt_id == "attempt-1"
        # Toolsets come from the grant, never the body.
        assert policy.allowed_toolsets == frozenset({"shell", "web"})
        assert create.call_args.kwargs["session_id"] == run_id
        task_id = create.return_value.run_conversation.call_args.kwargs["task_id"]
        assert task_id == run_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_grant",
        [
            "garbage",
            "v2.attempt-1.notanumber.shell.deadbeef",
            "v9" + _grant()[2:],
            # Well-formed v1 grant: the v1 lane is retired everywhere.
            "v1.attempt-1.2000000000." + "0" * 64,
            # Excluded toolset name signed into the segment.
            f"v2.attempt-1.2000000000.memory.{'0' * 64}",
            mint_grant(
                "wrong-secret-0123456789abcdef",
                attempt_id="attempt-1",
                expiry=2_000_000_000,
                profile="default",
                toolsets=GRANT_TOOLSETS,
            ),
        ],
    )
    async def test_defective_grant_403_no_run_no_reservation(
        self, adapter, tmp_path, bad_grant
    ):
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                rejected = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={
                        GRANT_HEADER: bad_grant,
                        "Idempotency-Key": "grant-retry",
                    },
                )
                assert rejected.status == 403
                assert (await rejected.json())["error"][
                    "code"
                ] == "invalid_private_grant"
                assert adapter._run_statuses == {}
                assert adapter._private_run_ids == set()
                create.assert_not_called()

                accepted = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={
                        GRANT_HEADER: _grant(),
                        "Idempotency-Key": "grant-retry",
                    },
                )
                assert accepted.status == 202
                assert "Idempotency-Replayed" not in accepted.headers

    @pytest.mark.asyncio
    async def test_expired_grant_403(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello"},
                headers={GRANT_HEADER: _grant(expiry_delta=-10)},
            )
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_missing_secret_disables_lane(self, adapter, monkeypatch):
        monkeypatch.delenv("PRIVATE_EXEC_GRANT_KEY", raising=False)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello"},
                headers={GRANT_HEADER: _grant()},
            )
        assert resp.status == 403

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_key",
        [
            GRANT_HEADER,
            GRANT_HEADER.lower(),
            "x-hErmes-privATE-grant",
            "X-HERMES-PRIVATE-GRANT",
        ],
    )
    async def test_query_string_grant_rejected_any_casing(self, adapter, query_key):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                resp = await cli.post(
                    f"/v1/runs?{query_key}={_grant()}",
                    json={"input": "hello"},
                )
        # Any query-delivered grant is a 400: the audit log records path_qs,
        # and a silently-ignored grant would fall back to a PUBLIC run.
        assert resp.status == 400
        assert adapter._run_statuses == {}
        assert adapter._private_run_ids == set()
        create.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/v1/runs/run-x"),
            ("get", "/v1/runs/run-x/events"),
            ("post", "/v1/runs/run-x/approval"),
            ("post", "/v1/runs/run-x/steer"),
            ("post", "/v1/runs/run-x/stop"),
        ],
    )
    async def test_query_grant_rejected_on_run_scoped_endpoints(
        self, adapter, method, path
    ):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            kwargs = {"json": {}} if method == "post" else {}
            resp = await getattr(cli, method)(
                f"{path}?{GRANT_HEADER}={_grant()}", **kwargs
            )
            body = await resp.json()
        # 400, not a silent ignore: path_qs lands in durable audit/access
        # logs, so a query-delivered grant is a leaked bearer credential.
        assert resp.status == 400
        assert body["error"]["code"] == "invalid_private_grant"

    def test_audit_context_scrubs_query_grant(self, adapter):
        grant = _grant()
        request = make_mocked_request(
            "GET", f"/v1/runs/run-x?x-hermes-private-grant={grant}&keep=1"
        )
        ctx = adapter._request_audit_context(request)
        assert grant not in ctx["path"]
        assert "keep=1" in ctx["path"]
        assert ctx["path"].startswith("/v1/runs/run-x")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "room_key", ["hosted_room_dispatch", "_room_execution_policy"]
    )
    async def test_room_body_keys_rejected_when_grant_present(
        self, adapter, room_key
    ):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        room_key: {"approval_mode": "never"},
                    },
                    headers={GRANT_HEADER: _grant()},
                )
                body = await resp.json()
        assert resp.status == 400
        assert body["error"]["code"] == "private_run_room_rejected"
        create.assert_not_called()
        assert adapter._run_statuses == {}
        assert adapter._private_run_ids == set()

    @pytest.mark.asyncio
    async def test_room_body_key_still_passes_through_without_grant(self, adapter):
        # Control: the pre-existing public pass-through (DE_DESIGN §1.2 H)
        # is not widened or narrowed by the private-lane rejection.
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "_room_execution_policy": {"approval_mode": "never"},
                    },
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_status(cli, run_id, {"completed"})

    @pytest.mark.asyncio
    async def test_room_token_cannot_ride_with_grant(self, adapter):
        request = make_mocked_request(
            "POST",
            "/v1/runs",
            headers={
                "Authorization": "HermesRoom some-room-token",
                GRANT_HEADER: _grant(),
            },
        )
        attempt_id, _expiry, _toolsets, err = runs_mod._verify_private_grant_admission(
            adapter, request, _api_server=api_server_mod
        )
        assert attempt_id is None
        assert err is not None
        assert err.status == 403

    @pytest.mark.asyncio
    async def test_wrong_profile_grant_rejected_on_single_profile_gateway(
        self, adapter, monkeypatch
    ):
        # Single-profile gateway: the URL-prefix ContextVar is None, but the
        # profile leg of the HMAC must bind to the gateway's own serving
        # profile identity, never collapse to "default".
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "servingp"
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                wrong = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: _grant()},  # minted for "default"
                )
                assert wrong.status == 403
                assert adapter._private_run_ids == set()

                right = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={
                        GRANT_HEADER: mint_grant(
                            SECRET,
                            attempt_id="attempt-1",
                            expiry=int(time.time()) + 600,
                            profile="servingp",
                            toolsets=GRANT_TOOLSETS,
                        )
                    },
                )
                assert right.status == 202
                run_id = (await right.json())["run_id"]
                assert run_id in adapter._private_run_ids
                await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_unknown_profile_identity_fails_closed(
        self, adapter, monkeypatch
    ):
        def _boom():
            raise RuntimeError("no profile identity")

        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", _boom
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello"},
                headers={GRANT_HEADER: _grant()},
            )
        assert resp.status == 403
        assert adapter._run_statuses == {}


class TestLinkageRejection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("body_extra", "headers_extra"),
        [
            ({"session_id": "linked"}, {}),
            ({"conversation_history": [{"role": "user", "content": "x"}]}, {}),
            ({"previous_response_id": "resp_1"}, {}),
            ({}, {"X-Hermes-Session-Key": "memory-a"}),
        ],
    )
    async def test_linkage_inputs_400_before_any_lookup(
        self, adapter, body_extra, headers_extra
    ):
        app = _create_runs_app(adapter)
        adapter._response_store.get = MagicMock()
        history = AsyncMock(return_value=[])
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_create_agent") as create,
                patch.object(adapter, "_conversation_history_for_session", history),
            ):
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", **body_extra},
                    headers={GRANT_HEADER: _grant(), **headers_extra},
                )
                body = await resp.json()

        assert resp.status == 400
        assert body["error"]["code"] == "private_run_linkage_rejected"
        create.assert_not_called()
        adapter._response_store.get.assert_not_called()
        history.assert_not_called()
        assert adapter._run_statuses == {}


class TestGrantToolsets:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body_toolsets",
        [
            ["web"],  # different set
            ["shell", "web", "extra"],  # superset of the grant
            [],  # empty differs from the grant's named set
            None,  # carried-but-null is not absent
            "shell",  # non-list fails closed
            ["shell", 42],  # non-string entry fails closed
        ],
    )
    async def test_body_toolsets_differing_from_grant_400(
        self, adapter, body_toolsets
    ):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "allowed_toolsets": body_toolsets,
                    },
                    headers={GRANT_HEADER: _grant(toolsets=("shell", "web"))},
                )
                body = await resp.json()
        assert resp.status == 400
        assert body["error"]["code"] == "private_grant_toolsets_mismatch"
        create.assert_not_called()
        assert adapter._run_statuses == {}
        assert adapter._private_run_ids == set()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body_toolsets",
        [
            ["shell", "web"],
            ["web", "shell", "shell"],  # same set, non-canonical order/dups
        ],
    )
    async def test_body_toolsets_matching_grant_admitted(
        self, adapter, body_toolsets
    ):
        app = _create_runs_app(adapter)
        grant = _grant(toolsets=("shell", "web"))
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "allowed_toolsets": body_toolsets,
                    },
                    headers={GRANT_HEADER: grant},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                await _wait_status(
                    cli, run_id, {"completed"}, headers={GRANT_HEADER: grant}
                )
        policy = create.call_args.kwargs["private_execution_policy"]
        assert policy.allowed_toolsets == frozenset({"shell", "web"})


class TestMetadataOnlyDurableStatus:
    @pytest.mark.asyncio
    async def test_completed_private_status_json_is_whitelisted(
        self, adapter, tmp_path
    ):
        path = tmp_path / "idem.db"
        _use_idempotency_db(adapter, path)
        app = _create_runs_app(adapter)
        grant = _grant()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent("MARKER-PRIVATE-OUTPUT")
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant, "Idempotency-Key": "priv-1"},
                )
                run_id = (await resp.json())["run_id"]
                status = await _wait_status(
                    cli, run_id, {"completed"}, headers={GRANT_HEADER: grant}
                )
                assert status["output"] == "MARKER-PRIVATE-OUTPUT"

        rows = _status_rows(path)
        assert len(rows) == 1
        stored = json.loads(rows[0][1])
        assert set(stored) == {
            "run_id",
            "status",
            "created_at",
            "updated_at",
            "usage",
            "error_class",
            "output_digest",
            "output_ref",
            "private",
        }
        assert stored["private"] is True
        assert stored["status"] == "completed"
        assert stored["output_digest"] == sha256(
            b"MARKER-PRIVATE-OUTPUT"
        ).hexdigest()
        assert "MARKER-PRIVATE-OUTPUT" not in rows[0][1]

    @pytest.mark.asyncio
    async def test_failure_stores_fixed_error_class_only(self, adapter, tmp_path):
        path = tmp_path / "idem.db"
        _use_idempotency_db(adapter, path)
        app = _create_runs_app(adapter)
        grant = _grant()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                agent = _done_agent()
                agent.run_conversation.side_effect = RuntimeError(
                    "SECRET-EXC-PROSE"
                )
                agent.run_conversation.return_value = None
                create.return_value = agent
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant, "Idempotency-Key": "priv-2"},
                )
                run_id = (await resp.json())["run_id"]
                status = await _wait_status(
                    cli, run_id, {"failed"}, headers={GRANT_HEADER: grant}
                )
                assert status["error"] == "run_failed"

        rows = _status_rows(path)
        stored = json.loads(rows[0][1])
        assert stored["error_class"] == "run_failed"
        assert "error" not in stored
        assert "SECRET-EXC-PROSE" not in rows[0][1]

    @pytest.mark.asyncio
    async def test_restart_hydration_stays_metadata_only(self, tmp_path):
        path = tmp_path / "idem.db"
        first = _make_adapter()
        _use_idempotency_db(first, path)
        grant = _grant()
        async with TestClient(TestServer(_create_runs_app(first))) as cli:
            with patch.object(first, "_create_agent") as create:
                create.return_value = _done_agent("MARKER-PRIVATE-OUTPUT")
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant, "Idempotency-Key": "priv-3"},
                )
                run_id = (await resp.json())["run_id"]
                await _wait_status(
                    cli, run_id, {"completed"}, headers={GRANT_HEADER: grant}
                )
        first._run_idempotency_store.close()

        restarted = _make_adapter()
        _use_idempotency_db(restarted, path)
        async with TestClient(TestServer(_create_runs_app(restarted))) as cli:
            denied = await cli.get(f"/v1/runs/{run_id}")
            hydrated = await cli.get(
                f"/v1/runs/{run_id}", headers={GRANT_HEADER: grant}
            )
            body = await hydrated.json()
            # Hydration re-derives privacy from the durable marker so the
            # steer/approval gates keep applying after a restart.
            steer = await cli.post(
                f"/v1/runs/{run_id}/steer",
                json={"input": "post-restart steer"},
                headers={GRANT_HEADER: grant},
            )
            steer_body = await steer.json()
            always = await cli.post(
                f"/v1/runs/{run_id}/approval",
                json={"choice": "always"},
                headers={GRANT_HEADER: grant},
            )
            always_body = await always.json()
        assert denied.status == 404
        assert hydrated.status == 200
        assert body["status"] == "completed"
        assert "output" not in body
        assert "MARKER-PRIVATE-OUTPUT" not in json.dumps(body)
        assert run_id in restarted._private_run_ids
        assert steer.status == 409
        assert steer_body["error"]["code"] == "private_run_no_steer"
        assert always.status == 400
        assert always_body["error"]["code"] == "invalid_approval_choice"

    @pytest.mark.asyncio
    async def test_dead_owner_rewrite_stays_whitelisted(self, tmp_path):
        from gateway.platforms.api_server import RunIdempotencyStore

        path = tmp_path / "idem.db"
        scope = sha256(
            "default\0unauthenticated-test-listener"
            "\0private-grant\0attempt-1".encode()
        ).hexdigest()
        store = RunIdempotencyStore(str(path))
        store.reserve(
            scope,
            "stale-priv",
            "fingerprint",
            "run_priv",
            {
                "run_id": "run_priv",
                "status": "running",
                "created_at": 1.0,
                "updated_at": 1.0,
                "usage": {},
                "error_class": "",
                "output_digest": "",
                "output_ref": "/v1/runs/run_priv",
                "private": True,
            },
            owner_pid=999_999_999,
            owner_started=1,
        )
        store.close()

        restarted = _make_adapter()
        _use_idempotency_db(restarted, path)
        async with TestClient(TestServer(_create_runs_app(restarted))) as cli:
            resp = await cli.get(
                "/v1/runs/run_priv", headers={GRANT_HEADER: _grant()}
            )
            body = await resp.json()
        assert resp.status == 200
        assert body["status"] == "interrupted"

        rows = _status_rows(path)
        stored = json.loads(rows[0][1])
        # The dead-owner rewrite is a durable write for a private run: it
        # must flow through the metadata-only projection instead of adding
        # raw error/last_event keys to status_json.
        assert set(stored) == {
            "run_id",
            "status",
            "created_at",
            "updated_at",
            "usage",
            "error_class",
            "output_digest",
            "output_ref",
            "private",
        }
        assert stored["status"] == "interrupted"
        assert stored["error_class"] == "gateway_restart_interrupted"
        assert stored["private"] is True


class TestEgressGating:
    @pytest.mark.asyncio
    async def test_get_and_events_require_grant(self, adapter):
        app = _create_runs_app(adapter)
        grant = _grant()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent("PRIVATE-OUTPUT")
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant},
                )
                run_id = (await resp.json())["run_id"]
                status = await _wait_status(
                    cli, run_id, {"completed"}, headers={GRANT_HEADER: grant}
                )
                assert status["output"] == "PRIVATE-OUTPUT"

                assert (await cli.get(f"/v1/runs/{run_id}")).status == 404
                assert (
                    await cli.get(f"/v1/runs/{run_id}/events")
                ).status == 404

    @pytest.mark.asyncio
    async def test_events_are_lifecycle_only_for_private_runs(self, adapter):
        app = _create_runs_app(adapter)
        grant = _grant()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:

                def _run(user_message=None, conversation_history=None, task_id=None):
                    kw = create.call_args.kwargs
                    kw["stream_delta_callback"]("SECRET-DELTA")
                    cb = kw["tool_progress_callback"]
                    cb("reasoning.available", preview="SECRET-REASONING")
                    cb("tool.started", tool_name="bash", preview="SECRET-ARGS")
                    cb(
                        "tool.completed",
                        tool_name="bash",
                        duration=0.1,
                        is_error=False,
                    )
                    return {"final_response": "PRIVATE-OUTPUT"}

                agent = _done_agent()
                agent.run_conversation.side_effect = _run
                create.return_value = agent
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant},
                )
                run_id = (await resp.json())["run_id"]
                events = await cli.get(
                    f"/v1/runs/{run_id}/events", headers={GRANT_HEADER: grant}
                )
                assert events.status == 200
                body = await events.text()

        assert "tool.started" in body
        assert "tool.completed" in body
        assert "run.completed" in body
        assert "PRIVATE-OUTPUT" in body
        assert "SECRET-DELTA" not in body
        assert "SECRET-REASONING" not in body
        assert "reasoning.available" not in body
        assert "SECRET-ARGS" not in body

    @pytest.mark.asyncio
    async def test_steer_409_private_run_no_steer(self, adapter):
        app = _create_runs_app(adapter)
        grant = _grant()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                agent, ready, interrupted = _make_slow_agent()
                create.return_value = agent
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant},
                )
                run_id = (await resp.json())["run_id"]
                ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                steer = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"input": "new direction"},
                    headers={GRANT_HEADER: grant},
                )
                body = await steer.json()
                interrupted.set()
                await asyncio.sleep(0.2)

        assert steer.status == 409
        assert body["error"]["code"] == "private_run_no_steer"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_private_approval_choices_limited_to_once_deny(self, adapter):
        app = _create_runs_app(adapter)
        grant = _grant()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                agent, ready, interrupted = _make_slow_agent()
                create.return_value = agent
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant},
                )
                run_id = (await resp.json())["run_id"]
                ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                entry = approval_mod._ApprovalEntry({
                    "command": "bash -c danger",
                    "description": "private approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [entry]

                always = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "always"},
                    headers={GRANT_HEADER: grant},
                )
                always_body = await always.json()
                assert always.status == 400
                assert always_body["error"]["code"] == "invalid_approval_choice"
                assert entry.result is None

                once = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                    headers={GRANT_HEADER: grant},
                )
                assert once.status == 200
                assert entry.result == "once"
                interrupted.set()
                await asyncio.sleep(0.2)


class TestStopVersusTermination:
    @pytest.mark.asyncio
    async def test_stop_ack_is_not_terminal_until_worker_exits(self, adapter):
        app = _create_runs_app(adapter)
        grant = _grant()
        run_can_finish = threading.Event()
        started = threading.Event()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                agent = _done_agent()

                def _run(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late", "interrupted": True}

                agent.run_conversation.side_effect = _run
                agent.run_conversation.return_value = None
                create.return_value = agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={GRANT_HEADER: grant},
                )
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop = await cli.post(
                    f"/v1/runs/{run_id}/stop", headers={GRANT_HEADER: grant}
                )
                assert stop.status == 200
                assert (await stop.json())["status"] == "stopping"
                await asyncio.sleep(0.1)

                bridge_view = await _wait_status(
                    cli, run_id, {"stopping"}, headers={GRANT_HEADER: grant}
                )
                assert bridge_view["status"] not in {
                    "completed",
                    "failed",
                    "cancelled",
                    "interrupted",
                }

                run_can_finish.set()
                terminal = await _wait_status(
                    cli, run_id, {"cancelled"}, headers={GRANT_HEADER: grant}
                )
                assert terminal["status"] == "cancelled"


class TestPrivateIdempotencyScope:
    @pytest.mark.asyncio
    async def test_same_key_without_grant_cannot_replay_private_run(
        self, adapter, tmp_path
    ):
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        grant = _grant()
        headers_private = {GRANT_HEADER: grant, "Idempotency-Key": "shared-key"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                private = await cli.post(
                    "/v1/runs", json={"input": "same"}, headers=headers_private
                )
                assert private.status == 202
                private_run = (await private.json())["run_id"]

                public = await cli.post(
                    "/v1/runs",
                    json={"input": "same"},
                    headers={"Idempotency-Key": "shared-key"},
                )
                assert public.status == 202
                public_body = await public.json()
                assert public_body["run_id"] != private_run
                assert "Idempotency-Replayed" not in public.headers

                replay = await cli.post(
                    "/v1/runs", json={"input": "same"}, headers=headers_private
                )
                assert replay.status == 202
                replay_body = await replay.json()
                assert replay_body["run_id"] == private_run
                assert replay.headers["Idempotency-Replayed"] == "true"
                await asyncio.sleep(0.2)


class TestPrivateGrantFingerprint:
    """The signed grant is part of the private request identity (finding
    api_server_runs.py:813): a same-key retry with a different grant must
    409, never replay a run whose policy came from the original grant."""

    @pytest.mark.asyncio
    async def test_same_key_same_grant_replays(self, adapter, tmp_path):
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        headers = {GRANT_HEADER: _grant(), "Idempotency-Key": "grant-fp"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                first = await cli.post(
                    "/v1/runs", json={"input": "same"}, headers=headers
                )
                assert first.status == 202
                run_id = (await first.json())["run_id"]
                replay = await cli.post(
                    "/v1/runs", json={"input": "same"}, headers=headers
                )
                assert replay.status == 202
                assert (await replay.json())["run_id"] == run_id
                assert replay.headers["Idempotency-Replayed"] == "true"
                await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_same_key_different_grant_toolsets_conflicts(
        self, adapter, tmp_path
    ):
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        expiry = int(time.time()) + 600

        def grant_for(toolsets):
            return mint_grant(
                SECRET,
                attempt_id="attempt-1",
                expiry=expiry,
                profile="default",
                toolsets=toolsets,
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                first = await cli.post(
                    "/v1/runs",
                    json={"input": "same"},
                    headers={
                        GRANT_HEADER: grant_for(("shell",)),
                        "Idempotency-Key": "grant-fp",
                    },
                )
                assert first.status == 202
                conflict = await cli.post(
                    "/v1/runs",
                    json={"input": "same"},
                    headers={
                        GRANT_HEADER: grant_for(("shell", "web")),
                        "Idempotency-Key": "grant-fp",
                    },
                )
                assert conflict.status == 409
                body = await conflict.json()
                assert body["error"]["code"] == "idempotency_key_conflict"
                await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_public_runs_fingerprint_unchanged(self, adapter, tmp_path):
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        headers = {"Idempotency-Key": "public-fp"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as create:
                create.return_value = _done_agent()
                first = await cli.post(
                    "/v1/runs", json={"input": "same"}, headers=headers
                )
                assert first.status == 202
                run_id = (await first.json())["run_id"]
                replay = await cli.post(
                    "/v1/runs", json={"input": "same"}, headers=headers
                )
                assert replay.status == 202
                assert (await replay.json())["run_id"] == run_id
                assert replay.headers["Idempotency-Replayed"] == "true"
                await asyncio.sleep(0.2)


class TestPrivateFailureLogVisibility:
    """Fixed-error-class private log lines must survive the private filter.

    Both directions (finding api_server_runs.py:1264): the allow marker is
    honored ONLY for fixed-template records with identifier-shaped args and
    no exception payload; everything else stays dropped.
    """

    def _policy(self):
        return build_policy(
            run_id="run-log",
            profile="default",
            execution_attempt_id="att-log",
            lease_expires_at="2099-01-01T00:00:00Z",
            allowed_toolsets=[],
        )

    def _record(self, msg, args, **attrs):
        factory = logging.getLogRecordFactory()
        record = factory(
            "gateway.platforms.api_server",
            logging.ERROR,
            __file__,
            1,
            msg,
            args,
            None,
        )
        for key, value in attrs.items():
            setattr(record, key, value)
        return record

    def test_allow_marker_passes_fixed_template_only(self):
        from hermes_logging import _PrivateRecordFilter

        flt = _PrivateRecordFilter()
        token = bind_private_policy(self._policy())
        try:
            allowed = self._record(
                "[api_server] private run %s failed: run_failed",
                ("run_abc123",),
                hermes_private_allow=True,
            )
            unmarked = self._record(
                "[api_server] private run %s failed: run_failed",
                ("run_abc123",),
            )
            prose_arg = self._record(
                "[api_server] private run %s failed: run_failed",
                ("SECRET private prose with spaces",),
                hermes_private_allow=True,
            )
            with_exc = self._record(
                "[api_server] private run %s failed: run_failed",
                ("run_abc123",),
                hermes_private_allow=True,
            )
            try:
                raise RuntimeError("SECRET-EXC")
            except RuntimeError:
                import sys as _sys

                with_exc.exc_info = _sys.exc_info()
        finally:
            reset_private_policy(token)

        public = self._record("public line %s", ("anything at all",))

        assert flt.filter(allowed) is True
        assert flt.filter(unmarked) is False
        assert flt.filter(prose_arg) is False
        assert flt.filter(with_exc) is False
        assert flt.filter(public) is True

    def test_allow_marker_rejects_unregistered_and_preinterpolated_msgs(self):
        from hermes_logging import _PrivateRecordFilter

        flt = _PrivateRecordFilter()
        token = bind_private_policy(self._policy())
        try:
            preinterpolated = self._record(
                "private run run_abc failed: SECRET-EXC-PROSE with spaces",
                (),
                hermes_private_allow=True,
            )
            unregistered = self._record(
                "unregistered fixed template %s",
                ("run_abc123",),
                hermes_private_allow=True,
            )
        finally:
            reset_private_policy(token)

        # A pre-interpolated f-string is a plain str at runtime,
        # indistinguishable from a literal template — only templates
        # registered in _PRIVATE_ALLOW_TEMPLATES are honored.
        assert flt.filter(preinterpolated) is False
        assert flt.filter(unregistered) is False

    @pytest.mark.asyncio
    async def test_private_failure_emits_a_filter_surviving_fixed_line(
        self, adapter
    ):
        from hermes_logging import _PrivateRecordFilter

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        handler.addFilter(_PrivateRecordFilter())
        target = logging.getLogger("gateway.platforms.api_server")
        target.addHandler(handler)
        try:
            app = _create_runs_app(adapter)
            grant = _grant()
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent") as create:
                    agent = _done_agent()
                    agent.run_conversation.side_effect = RuntimeError(
                        "SECRET-EXC-PROSE"
                    )
                    agent.run_conversation.return_value = None
                    create.return_value = agent
                    resp = await cli.post(
                        "/v1/runs",
                        json={"input": "hello"},
                        headers={GRANT_HEADER: grant},
                    )
                    run_id = (await resp.json())["run_id"]
                    await _wait_status(
                        cli, run_id, {"failed"}, headers={GRANT_HEADER: grant}
                    )
        finally:
            target.removeHandler(handler)

        messages = [record.getMessage() for record in records]
        assert any(
            message.endswith("failed: run_failed") for message in messages
        ), messages
        assert all("SECRET-EXC-PROSE" not in message for message in messages)
