"""Lane C frame/rewind transitions with the real SQLite store and memory manager."""

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig
from gateway.session import SessionStore

from .test_slash_reset_undo_frame_ack import _BareProvider, _make_event, _make_runner, _ref


@pytest.fixture
def frame_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, key = _make_runner()
    runner._active_profile_name = lambda: "jenna"
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner.session_store = store
    event = _make_event("/undo")
    entry = store.get_or_create_session(event.source)
    reference = _ref(session_id=entry.session_id)
    assert store.record_working_frame_scope(key, entry.session_id, reference, active_profile="jenna")
    db = store._db
    for text in ("first", "second"):
        db.append_message(entry.session_id, "user", text)
        db.append_message(entry.session_id, "assistant", f"reply to {text}")
    provider = _BareProvider()
    monkeypatch.setattr("tools.memory_tool.get_builtin_memory_config", lambda: {"provider": "frame"})
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: provider)
    try:
        yield runner, key, entry, provider
    finally:
        for handle in list(store._db_handles.values()):
            handle.close()


@pytest.mark.asyncio
async def test_repeated_undo_retires_and_rewinds_until_noop(frame_route):
    runner, key, entry, provider = frame_route
    store = runner.session_store
    for expected_text in ("second", "first"):
        assert store.has_rewindable_user_turn(entry.session_id)
        reply = await runner._handle_undo_command(_make_event("/undo"))
        assert expected_text in reply
        scope = store.get_working_frame_scope(key)
        assert scope["cleared"] is True
    assert not store.has_rewindable_user_turn(entry.session_id)
    assert store.load_transcript(entry.session_id) == []
    before = store.get_working_frame_scope(key)
    assert await runner._handle_undo_command(_make_event("/undo")) == t("gateway.undo.nothing")
    assert store.get_working_frame_scope(key) == before
    assert provider.reset_calls == [(entry.session_id, True), (entry.session_id, True)]
    assert provider.initialized is False
    assert provider.shutdowns == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/undo", "/new"])
async def test_retirement_failure_keeps_durable_route_and_transcript(frame_route, command):
    runner, key, entry, provider = frame_route
    store = runner.session_store
    before_scope = store.get_working_frame_scope(key)
    before_messages = store.load_transcript(entry.session_id)

    def reject(*args, **kwargs):
        raise RuntimeError("retirement unavailable")

    provider.on_pre_session_reset = reject
    handler = runner._handle_reset_command if command == "/new" else runner._handle_undo_command
    result = await handler(_make_event(command))
    error_key = "gateway.reset.memory_scope_failed" if command == "/new" else "gateway.undo.memory_scope_failed"
    assert result == t(error_key)
    assert store.lookup_by_session_key(key).session_id == entry.session_id
    assert store.get_working_frame_scope(key) == before_scope
    assert store.load_transcript(entry.session_id) == before_messages
    assert store.has_rewindable_user_turn(entry.session_id)
