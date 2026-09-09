"""Lane C regression controls through extracted gateway execution paths."""

import asyncio
from concurrent.futures import Future
import json
from types import SimpleNamespace

import pytest

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from tools import clarify_gateway
from tools.clarify_tool import mark_recommended

from .test_slash_reset_undo_frame_ack import _make_event, _make_runner, _ref


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/undo"])
async def test_scope_lookup_failure_preserves_state(command):
    runner, _ = _make_runner(scope=_ref())
    runner.session_store.get_working_frame_scope.side_effect = OSError("scope storage unavailable")
    invalidated = []
    runner._invalidate_session_run_generation = lambda *a, **kw: invalidated.append(a)
    handler = runner._handle_reset_command if command == "/new" else runner._handle_undo_command
    reply = await handler(_make_event(command))
    assert reply == t("gateway.reset.memory_scope_failed" if command == "/new" else "gateway.undo.memory_scope_failed")
    assert invalidated == []
    runner.session_store.reset_session.assert_not_called()
    runner.session_store.rewind_session.assert_not_called()


@pytest.mark.asyncio
async def test_orphan_scope_cas_failure_blocks_reset(monkeypatch):
    runner, _ = _make_runner(scope=_ref())
    monkeypatch.setattr("tools.memory_tool.get_builtin_memory_config", lambda: {"provider": ""})
    runner.session_store.expire_working_frame_scope.return_value = False
    result = await runner._handle_reset_command(_make_event("/new"))
    assert result == t("gateway.reset.memory_scope_failed")
    runner.session_store.reset_session.assert_not_called()


class _ClarifyAdapter:
    def __init__(self, resolve):
        self.resolve = resolve
        self.entries = []
        self.resumed = []

    def pause_typing_for_chat(self, chat_id):
        pass

    def resume_typing_for_chat(self, chat_id):
        self.resumed.append(chat_id)

    async def send_clarify(self, **kwargs):
        entry = clarify_gateway.get_pending_for_session(kwargs["session_key"], include_choice_prompts=True)
        self.entries.append(entry)
        if entry.evidence is not None:
            entry.evidence.delivered(
                question=kwargs["question"], options=kwargs["choices"], rendered_text="Pick a route",
                bot_id="bot", chat_id=kwargs["chat_id"], thread_id=None, message_id="prompt",
            )
        self.resolve(entry)
        return SendResult(success=True, message_id="prompt")


def _turn_for_adapter(adapter):
    ctx = SimpleNamespace(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="u", profile="liz"),
        session_key="lane-c-clarify", _status_adapter=adapter, _status_chat_id="chat",
        _status_thread_metadata={}, stream_consumer_holder=[None],
    )
    turn = TurnRunner(SimpleNamespace(), ctx)
    turn._close_native_stream_boundary = lambda *a, **kw: True

    def schedule(coro, *_):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    turn._schedule = schedule
    return turn


@pytest.mark.parametrize("mode", ["choice", "multi", "free", "other", "cancel", "timeout"])
def test_actual_clarify_callback_preserves_response_modes(monkeypatch, mode):
    events = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: events.append(kw))
    monkeypatch.setattr(clarify_gateway, "get_clarify_timeout", lambda: 0.01)
    choices = None if mode == "free" else mark_recommended(["staging", "production"])

    def resolve(entry):
        if mode == "timeout":
            return
        if mode == "cancel":
            clarify_gateway.resolve_gateway_clarify(entry.clarify_id, "")
            return
        response = "1,2" if mode == "multi" else "custom" if mode in {"free", "other"} else "2"
        if mode == "other":
            clarify_gateway.mark_awaiting_text(entry.clarify_id)
        assert clarify_gateway.attempt_text_response_for_session(entry.session_key, response) == clarify_gateway.TEXT_RESOLVED

    adapter = _ClarifyAdapter(resolve)
    turn = _turn_for_adapter(adapter)
    try:
        response = turn._clarify_callback_sync("Pick a route", choices, mode == "multi")
        entry = adapter.entries[0]
        if mode == "multi":
            assert entry.multi_select is True
            assert json.loads(response) == choices
        elif mode in {"free", "other"}:
            assert response == "custom"
        elif mode in {"cancel", "timeout"}:
            assert response.startswith("[user did not respond")
        else:
            assert response == "production"
        assert adapter.resumed == ([] if mode in {"cancel", "timeout"} else ["chat"])
        assert clarify_gateway.has_pending("lane-c-clarify") is False
        if mode in {"multi", "free"}:
            assert entry.evidence is None
            assert events == []
        else:
            assert entry.evidence.options == tuple(choices)
            assert [event["payload"]["kind"] for event in events] == ["prompt_delivered", "prompt_closed"]
            expected = mode if mode in {"timeout"} else "cancelled" if mode == "cancel" else "resolved"
            assert events[-1]["payload"]["reason"] == expected
    finally:
        clarify_gateway.clear_session("lane-c-clarify")
