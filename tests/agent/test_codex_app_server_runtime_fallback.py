"""Codex app-server outages hand the turn to the configured ``fallback_providers`` chain.

A codex login that expired, an app-server that stopped answering or exited, and a Codex backend that
kept failing after codex's own retries used to end the turn with an error and no fallback: only a
quota/rate-limit/billing text was classified (#71633). These tests drive the real
CodexAppServerSession over a scripted client through ``AIAgent.run_conversation`` so the whole
classify -> activate -> generic-loop retry path runs; the fallback is a local fake
OpenAI-compatible server. Cases that are not codex outages (user stop, approval denials, tool
errors, policy refusals, context-length, a Hermes-side close, the run budget, no fallback
configured) must keep today's behaviour.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

import agent.transports.codex_app_server_session as session_mod
import run_agent
from agent.transports.codex_app_server import CodexAppServerError, CodexAppServerTransportError
from tests.agent.transports.test_codex_app_server_session import FakeClient

FALLBACK_ANSWER = "fallback answered"
CODEX_ANSWER = "codex answered"


class _ScriptedClient(FakeClient):
    """FakeClient whose subprocess can die and whose requests can raise."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dead = False
        self.raise_on: dict[str, BaseException] = {}
        self.on_turn_start = None

    def request(self, method, params=None, timeout=30.0):
        if method == "turn/start" and self.on_turn_start is not None:
            self.on_turn_start()
        if method in self.raise_on:
            self.requests.append((method, params or {}))
            raise self.raise_on[method]
        return super().request(method, params, timeout)

    def is_alive(self) -> bool:
        return not self.dead and super().is_alive()

    def turn_failed(self, message: str, info=None) -> None:
        error = {"message": message}
        if info is not None:
            error["codexErrorInfo"] = info
        self.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "failed", "error": error})

    def turn_succeeded(self, text: str = CODEX_ANSWER) -> None:
        self.queue_notification("item/completed", threadId="t", turnId="tu1", item={
            "type": "agentMessage", "id": "msg-1", "text": text})
        self.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "completed"})

    def turn_starts(self) -> int:
        return sum(1 for method, _ in self.requests if method == "turn/start")


@pytest.fixture
def fallback_server():
    """Local OpenAI-compatible fallback that always answers FALLBACK_ANSWER; yields (base_url, calls)."""
    calls: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
            if self.path.endswith("/chat/completions"):  # not the provider's metadata probes
                calls.append(self.path)
            chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": FALLBACK_ANSWER},
                                  "finish_reason": "stop"}]}
            data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}/v1", calls
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def clients(monkeypatch):
    """Every codex spawn gets a fresh _ScriptedClient; a test scripts them through ``clients.next``."""
    spawned: list[_ScriptedClient] = []
    pending: list[_ScriptedClient] = []

    def factory(**kwargs):
        client = pending.pop(0) if pending else _ScriptedClient(**kwargs)
        spawned.append(client)
        return client

    def next_client() -> _ScriptedClient:
        client = _ScriptedClient()
        pending.append(client)
        return client

    monkeypatch.setattr(session_mod, "CodexAppServerClient", factory)
    factory.next = next_client
    factory.spawned = spawned
    return factory


def _make_agent(fallback_url: str | None, **kwargs):
    fallback = [{"provider": "custom", "model": "fake-fb", "api_key": "fb-key", "base_url": fallback_url}] \
        if fallback_url else None
    return run_agent.AIAgent(
        api_key="stub", base_url="https://stub.invalid", provider="openai", api_mode="codex_app_server",
        quiet_mode=True, skip_context_files=True, skip_memory=True, fallback_model=fallback, **kwargs,
    )


def _run(agent, text: str = "hello") -> dict:
    with patch.object(agent, "_spawn_background_review", return_value=None):
        return agent.run_conversation(text)


# --- trigger classes: each must activate the fallback for this turn --------------------------


def _auth_unauthorized_info(client, _agent):
    client.turn_failed("Your session has ended. Sign in again to continue.", "unauthorized")


def _auth_refresh_failed(client, _agent):
    client.turn_failed("Your access token could not be refreshed. Please log out and sign in again.", "other")


def _auth_login_expired(client, _agent):
    client.turn_failed("Provided authentication token is expired. Please try signing in again.")


def _auth_not_logged_in(client, _agent):
    client.raise_on["turn/start"] = CodexAppServerError(code=-32600, message="Not logged in. Run `codex login`.")


def _auth_forbidden(client, _agent):
    client.turn_failed("unexpected status 403 Forbidden", {"responseStreamConnectionFailed": {"httpStatusCode": 403}})


def _timeout_turn_start(client, _agent):
    client.raise_on["turn/start"] = TimeoutError("codex app-server method 'turn/start' timed out after 10s")


def _timeout_startup(client, _agent):
    client.raise_on["thread/start"] = TimeoutError("codex app-server method 'thread/start' timed out after 15s")


def _process_exit_mid_turn(client, _agent):
    client.set_stderr_tail(["thread 'tokio-runtime-worker' panicked at core/src/codex.rs:1:1"])

    def die():
        client.dead = True
    client.on_turn_start = die


def _transport_lost(client, _agent):
    client.raise_on["turn/start"] = CodexAppServerTransportError(code=-32000, message="codex app-server stdout closed")


def _transport_5xx_retries_exhausted(client, _agent):
    client.turn_failed("exceeded retry limit, last status: 502 Bad Gateway",
                       {"responseTooManyFailedAttempts": {"httpStatusCode": 502}})


def _transport_stream_disconnected(client, _agent):
    client.turn_failed("stream disconnected before completion: error sending request",
                       {"responseStreamDisconnected": {"httpStatusCode": None}})


def _transport_internal_server_error(client, _agent):
    client.turn_failed("We're currently experiencing high demand.", "internalServerError")


TRIGGERS = {
    "auth-unauthorized-info": (_auth_unauthorized_info, "auth"),
    "auth-refresh-failed": (_auth_refresh_failed, "auth"),
    "auth-login-expired": (_auth_login_expired, "auth"),
    "auth-not-logged-in": (_auth_not_logged_in, "auth"),
    "auth-403": (_auth_forbidden, "auth"),
    "timeout-turn-start": (_timeout_turn_start, "timeout"),
    "timeout-startup": (_timeout_startup, "timeout"),
    "process-exit-mid-turn": (_process_exit_mid_turn, "disconnect"),
    "transport-lost": (_transport_lost, "disconnect"),
    "transport-5xx-retries-exhausted": (_transport_5xx_retries_exhausted, "transport"),
    "transport-stream-disconnected": (_transport_stream_disconnected, "transport"),
    "transport-internal-server-error": (_transport_internal_server_error, "transport"),
}


@pytest.mark.parametrize("case", sorted(TRIGGERS))
def test_codex_outage_completes_turn_on_fallback(case, clients, fallback_server, caplog):
    script, kind = TRIGGERS[case]
    fallback_url, calls = fallback_server
    agent = _make_agent(fallback_url)
    script(clients.next(), agent)

    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        result = _run(agent)

    assert result["final_response"] == FALLBACK_ANSWER
    assert result["completed"] is True
    assert calls == ["/v1/chat/completions"], "the fallback answers once"
    assert agent.api_mode != "codex_app_server"
    assert len(clients.spawned) == 1, "no second codex attempt inside the same turn"
    logged = [r.getMessage() for r in caplog.records if "fallback" in r.getMessage()]
    assert any(f"({kind})" in line for line in logged), logged


# --- must not fall back --------------------------------------------------------------------


def _context_window_exceeded(client, _agent):
    # A stale refresh-token line in stderr must not turn a context-length failure into an auth fallback.
    client.set_stderr_tail(["ERROR codex_core::auth: token refresh failed: invalid_grant"])
    client.turn_failed("Codex ran out of room in the model's context window.", "contextWindowExceeded")


def _policy_refusal(client, _agent):
    client.turn_failed("This content was flagged for possible cybersecurity risk.", "cyberPolicy")


def _approval_denials(client, _agent):
    client.queue_server_request("item/commandExecution/requestApproval", request_id="r1", command="rm -rf /", cwd="/")
    respond = client.respond

    def respond_then_fail(request_id, result):
        # codex only moves on once the approval is answered (declined: nobody can approve here).
        respond(request_id, result)
        client.turn_failed("Too many approval requests were denied.", "tooManyDenials")
    client.respond = respond_then_fail


def _sandbox_error(client, _agent):
    client.turn_failed("sandbox error: command was killed by a signal", "sandboxError")


def _tool_failure_inside_turn(client, _agent):
    client.queue_notification("item/completed", threadId="t", turnId="tu1", item={
        "type": "commandExecution", "id": "exec-1", "command": "false", "status": "failed",
        "aggregatedOutput": "boom", "exitCode": 1})
    client.turn_succeeded("the command failed")


def _user_stop(client, agent):
    # The stop lands as the app-server goes away: the session itself never sees an interrupt.
    client.on_turn_start = agent.interrupt
    client.raise_on["turn/start"] = CodexAppServerTransportError(code=-32000, message="codex app-server stdout closed")


def _hermes_closed_session(client, agent):
    client.on_turn_start = lambda: agent._codex_session.close()


MUST_NOT = {
    "context-window": _context_window_exceeded,
    "policy-refusal": _policy_refusal,
    "approval-denials": _approval_denials,
    "sandbox-error": _sandbox_error,
    "tool-failure-inside-turn": _tool_failure_inside_turn,
    "user-stop": _user_stop,
    "hermes-closed-session": _hermes_closed_session,
}


@pytest.mark.parametrize("case", sorted(MUST_NOT))
def test_non_outage_failures_stay_on_codex(case, clients, fallback_server):
    fallback_url, calls = fallback_server
    agent = _make_agent(fallback_url)
    MUST_NOT[case](clients.next(), agent)

    with patch.object(agent, "_try_activate_fallback", wraps=agent._try_activate_fallback) as activate:
        result = _run(agent)

    assert activate.call_count == 0
    assert calls == []
    assert agent.api_mode == "codex_app_server"
    assert result["final_response"] != FALLBACK_ANSWER


def test_run_budget_deadline_is_not_a_codex_outage(clients, fallback_server):
    """The codex turn deadline exists only as the run budget: spending it does not restart the turn elsewhere."""
    fallback_url, calls = fallback_server
    agent = _make_agent(fallback_url, run_budget_seconds=0.3)
    clients.next()  # turn starts and never completes

    result = _run(agent)

    assert "timed out" in str(result["error"])
    assert calls == []
    assert agent.api_mode == "codex_app_server"


@pytest.mark.parametrize("case", ["auth-unauthorized-info", "process-exit-mid-turn", "transport-5xx-retries-exhausted"])
def test_outage_without_fallback_providers_returns_codex_error(case, clients):
    """AIRI/market have no fallback_providers: the codex error is returned as before."""
    agent = _make_agent(None)
    TRIGGERS[case][0](clients.next(), agent)

    with patch.object(agent, "_try_activate_fallback") as activate:
        result = _run(agent)

    activate.assert_not_called()
    assert result["completed"] is False
    assert result["error"]
    assert agent.api_mode == "codex_app_server"


def test_fallback_is_turn_scoped_and_primary_returns_next_turn(clients, fallback_server):
    fallback_url, calls = fallback_server
    agent = _make_agent(fallback_url)
    first_client = clients.next()
    _auth_unauthorized_info(first_client, agent)
    # The second turn succeeds on whichever app-server serves it (a retired one is respawned).
    first_client.turn_succeeded()
    clients.next().turn_succeeded()

    first = _run(agent, "first")
    second = _run(agent, "second")

    assert first["final_response"] == FALLBACK_ANSWER
    assert second["final_response"] == CODEX_ANSWER
    assert agent.api_mode == "codex_app_server"
    assert calls == ["/v1/chat/completions"]
    assert sum(c.turn_starts() for c in clients.spawned) == 2, "codex runs once per turn"
