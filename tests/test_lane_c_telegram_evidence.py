"""Lane C: evidence survives the decomposed adapter's real dispatch paths."""
import copy
import json
from unittest.mock import AsyncMock

import pytest
from telegram import CallbackQuery

from .lane_c_telegram_support import lane_c, receive, resolve_text, send, tap
from tools import clarify_gateway as cm

pytestmark = pytest.mark.asyncio


async def test_native_callback_keeps_original_update_and_decorated_choice(lane_c):
    assert (await send(lane_c)).success
    update = await tap(lane_c)
    assert lane_c.entry.response == "A (Recommended)"
    delivered, original, closed = lane_c.events
    assert delivered["options"] == lane_c.entry.choices
    assert delivered["rendered_text"] == lane_c.adapter._bot.send_message.call_args.kwargs["text"]
    assert delivered["choice_buttons"][0]["callback_data"] == update.callback_query.data
    assert original["update_id"] == str(update.update_id)
    assert original["callback_query_id"] == update.callback_query.id
    assert original["user_id"] == str(update.callback_query.from_user.id)
    assert original["option_index"] == 0
    assert closed["selected_callback_query_id"] == update.callback_query.id
    assert closed["selected_callback_update_id"] == str(update.update_id)
    assert closed["selected_reply_message_id"] is None
    assert not {"text", "click_time", "original_message_at", "owner"} & original.keys()
    assert [e["schema_version"] for e in lane_c.events] == [2, 2, 2]
    assert [e["sequence"] for e in lane_c.events] == [1, 2, 3]
    assert "lane-c" not in lane_c.adapter._clarify_state


async def test_dispatch_passes_same_native_update_to_capture(lane_c, monkeypatch):
    from gateway.telegram_clarify_callbacks import capture

    spy = AsyncMock(wraps=capture)
    monkeypatch.setattr("gateway.telegram_clarify_callbacks.capture", spy)
    await send(lane_c)
    update = await tap(lane_c)
    assert spy.call_args.args[1] is update
    assert lane_c.events[-1]["selected_callback_query_id"] == "lane-c-tap"


async def test_native_text_is_captured_before_trigger_normalization(lane_c):
    await send(lane_c)
    event = await receive(lane_c, "@test_bot 2")
    assert lane_c.events[-1]["text"] == "@test_bot 2"
    assert event.text == "2"
    assert resolve_text(event) == cm.TEXT_RESOLVED
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_reply_message_id"] is None


async def test_exact_native_text_keeps_selected_reply(lane_c):
    await send(lane_c)
    event = await receive(lane_c)
    assert resolve_text(event) == cm.TEXT_RESOLVED
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_reply_message_id"] == "101"
    assert lane_c.events[-1]["selected_reply_update_id"] == "80"
    assert lane_c.events[-1]["selected_callback_query_id"] is None


async def test_other_edit_invalidates_evidence_but_keeps_free_text(lane_c):
    await send(lane_c)
    await tap(lane_c, "other")
    assert lane_c.entry.awaiting_text
    assert lane_c.events[-1]["reason"] == "edited"
    text = CallbackQuery.edit_message_text.call_args.kwargs["text"]
    assert "&lt;one&gt;" in text and "<one>" not in text
    assert resolve_text(await receive(lane_c, "Use a different approach")) == cm.TEXT_RESOLVED
    assert lane_c.entry.response == "Use a different approach"
    assert [e["kind"] for e in lane_c.events] == ["prompt_delivered", "prompt_closed"]
    assert lane_c.events[-1]["selected_callback_query_id"] is None


@pytest.mark.parametrize("boundary", ["timeout", "cancelled", "reset", "shutdown", "edited"])
async def test_terminal_boundary_and_late_tap_do_not_reopen(lane_c, boundary):
    await send(lane_c)
    if boundary == "timeout":
        assert cm.wait_for_response("lane-c", 0.001) is None
    elif boundary == "edited":
        assert (await lane_c.adapter.edit_message("123", "100", "Changed")).success
    else:
        assert cm.clear_session("lane-c-session", reason=boundary) == 1
    assert lane_c.events[-1]["reason"] == boundary
    frozen = copy.deepcopy(lane_c.events)
    await tap(lane_c)
    assert lane_c.events == frozen
    assert lane_c.entry.evidence.closed


@pytest.mark.parametrize("mode", ["no_evidence", "free_text", "multi_select"])
async def test_public_and_ineligible_prompts_keep_original_response_contract(lane_c, mode):
    assert (await send(
        lane_c, choices=None if mode == "free_text" else ["A", "B"],
        evidence=mode != "no_evidence", multi_select=mode == "multi_select",
    )).success
    assert lane_c.entry.evidence is None
    if mode == "no_evidence":
        await tap(lane_c, "1")
        assert lane_c.entry.response == "B"
    else:
        text = "1, 2" if mode == "multi_select" else "My answer"
        assert resolve_text(await receive(lane_c, text)) == cm.TEXT_RESOLVED
        if mode == "multi_select":
            assert json.loads(lane_c.entry.response) == lane_c.entry.choices
        else:
            assert lane_c.entry.response == text
    assert lane_c.events == []


async def test_failed_send_and_late_ack_after_reset_emit_nothing(lane_c):
    lane_c.adapter._bot.send_message.side_effect = RuntimeError("transport failed")
    assert not (await send(lane_c)).success
    assert lane_c.events == []
    assert "lane-c" not in lane_c.adapter._clarify_state
    cm.clear_session("lane-c-session")

    async def late_ack(**kwargs):
        from .lane_c_telegram_support import message

        cm.clear_session("lane-c-session")
        return message(100, bot=True, reply_markup=kwargs["reply_markup"].to_dict())

    lane_c.adapter._bot.send_message.side_effect = late_ack
    assert (await send(lane_c)).success
    assert lane_c.events == []


async def test_ack_and_edit_failure_preserve_resolved_callback(lane_c, monkeypatch):
    await send(lane_c)
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock(side_effect=RuntimeError("ack failed")))
    with pytest.raises(RuntimeError, match="ack failed"):
        await tap(lane_c)
    frozen = copy.deepcopy(lane_c.events)
    assert lane_c.entry.event.is_set()
    assert frozen[-1]["selected_callback_query_id"] == "lane-c-tap"
    assert (await lane_c.adapter.edit_message("123", "100", "Already answered")).success
    cm.clear_session("lane-c-session")
    assert lane_c.events == frozen
