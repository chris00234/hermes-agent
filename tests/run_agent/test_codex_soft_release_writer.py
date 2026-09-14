"""Soft eviction must release the native writer, not the durable conversation."""

import threading
from unittest.mock import Mock

import pytest

from agent.codex_runtime import _ensure_codex_session
from agent.codex_session_state import load_thread_id
from agent.transports.codex_app_server import CodexAppServerError
from gateway.run_agent_cache import GatewayAgentCacheMixin
from hermes_state import SessionDB
from run_agent import AIAgent


def bare_agent(db, session_id):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent.client = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent._session_messages = ["retained"]
    return agent


@pytest.mark.parametrize("close_raises", [False, True])
def test_soft_release_is_idempotent_and_preserves_session_resources(close_raises):
    agent = bare_agent(None, "soft-release")
    session = Mock()
    if close_raises:
        session.close.side_effect = RuntimeError("already dead")
    agent._codex_session = session
    agent._codex_thread_binding = {"session_id": agent.session_id, "thread_id": "native-thread"}
    agent._close_task_resources = Mock()
    agent.shutdown_memory_provider = Mock()
    agent._finalize_owned_session_row = Mock()

    agent.release_clients()
    agent.release_clients()

    session.close.assert_called_once_with()
    assert agent._codex_session is None
    assert load_thread_id(agent) == "native-thread"
    assert agent._session_messages == ["retained"]
    agent._close_task_resources.assert_not_called()
    agent.shutdown_memory_provider.assert_not_called()
    agent._finalize_owned_session_row.assert_not_called()


@pytest.mark.macos_only
def test_gateway_soft_eviction_releases_writer_before_durable_resume(tmp_path, monkeypatch):
    import fcntl
    from agent.transports import codex_app_server_session as transport

    clients = []

    class LockingClient:
        def __init__(self, **kwargs):
            self.writer = None
            self.requests = []
            clients.append(self)

        def initialize(self, **kwargs):
            pass

        def request(self, method, params, **kwargs):
            self.requests.append((method, params))
            thread_id = params.get("threadId", "native-thread")
            self.writer = (tmp_path / f"{thread_id}.lock").open("a")
            try:
                fcntl.flock(self.writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.writer.close()
                self.writer = None
                raise CodexAppServerError(-32600, f"thread {thread_id} already has an active writer")
            return {"thread": {"id": thread_id}}

        def close(self):
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    monkeypatch.setattr(transport, "CodexAppServerClient", LockingClient)
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="conversation", source="telegram", model="codex")
    first = bare_agent(db, "conversation")
    restored = bare_agent(db, "conversation")
    try:
        _ensure_codex_session(first)
        thread_id = first._codex_session.ensure_started()
        # Same resource-only cleanup used by TTL, pressure, cap and explicit eviction.
        GatewayAgentCacheMixin._release_evicted_agent_soft(None, first)
        _ensure_codex_session(restored)
        assert restored._codex_session.ensure_started() == thread_id
        assert clients[1].requests == [("thread/resume", {"threadId": thread_id})]
        assert db.get_session("conversation")["ended_at"] is None
    finally:
        for client in clients:
            client.close()
        db.close()
