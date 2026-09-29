import copy
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_telegram_clarify_evidence import (
    Bot,
    Update,
    _message,
    _receive,
    _send,
    lane,
)
from telegram import CallbackQuery
from tools import clarify_gateway as cm


@pytest.fixture
def callback_lane(lane, monkeypatch):
    adapter, runner, entry, events, received = lane
    adapter.set_authorization_check(lambda *args, **kwargs: True)
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(CallbackQuery, "edit_message_text", AsyncMock())

    async def ack(**kwargs):
        return _message(
            100,
            bot=True,
            text="❓ Pick <one>\n\n1. A\n2. B",
            reply_markup=kwargs["reply_markup"].to_dict(),
        )

    adapter._bot.send_message.side_effect = ack
    return adapter, runner, entry, events, received


def _update(
    adapter,
    *,
    token="1",
    update_id=70,
    query_id="tap-1",
    user_id=777,
    message_changes=None,
    query_changes=None,
):
    kwargs = adapter._bot.send_message.call_args.kwargs
    prompt = _message(
        100,
        bot=True,
        text="❓ Pick <one>\n\n1. A\n2. B",
        reply_markup=kwargs["reply_markup"].to_dict(),
    ).to_dict()
    prompt.update(message_changes or {})
    query = {
        "id": query_id,
        "from": {"id": user_id, "is_bot": False, "first_name": "Owner"},
        "message": prompt,
        "chat_instance": "native-chat",
        "data": f"cl:cid:{token}",
    }
    query.update(query_changes or {})
    return Update.de_json(
        {"update_id": update_id, "callback_query": query}, Bot("123:test-token")
    )


async def _tap(adapter, **kwargs):
    update = _update(adapter, **kwargs)
    await adapter._handle_callback_query(update, None)
    return update


@pytest.mark.asyncio
async def test_actual_sdk_callback_closes_exact_choice_without_speech(callback_lane):
    adapter, _, entry, events, _ = callback_lane
    assert (await _send(adapter)).success
    await _tap(adapter)
    assert entry.response == "B"
    assert [event["kind"] for event in events] == [
        "prompt_delivered",
        "original_callback",
        "prompt_closed",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert {event["schema_version"] for event in events} == {2}
    assert events[0]["choice_buttons"] == [
        {"option_index": 0, "button_label": "1", "callback_data": "cl:cid:0"},
        {"option_index": 1, "button_label": "2", "callback_data": "cl:cid:1"},
    ]
    assert events[1]["callback_query_id"] == "tap-1"
    assert events[1]["update_id"] == "70"
    assert events[1]["prompt_message_id"] == "100"
    assert events[1]["option_index"] == 1
    assert events[1]["user_id"] == "777"
    assert (
        not {"text", "original_message_at", "message_id", "click_time", "owner"}
        & events[1].keys()
    )
    assert events[2]["selected_callback_query_id"] == "tap-1"
    assert events[2]["selected_callback_update_id"] == "70"
    assert events[2]["selected_reply_message_id"] is None
    assert events[2]["selected_reply_update_id"] is None


@pytest.mark.asyncio
async def test_localized_other_button_keeps_v2_callback_evidence(callback_lane, monkeypatch):
    from agent import i18n

    adapter, _, entry, events, _ = callback_lane
    monkeypatch.setenv("HERMES_LANGUAGE", "ko")
    i18n.reset_language_cache()
    assert (await _send(adapter)).success
    keyboard = adapter._bot.send_message.call_args.kwargs["reply_markup"].to_dict()
    other = keyboard["inline_keyboard"][-1][0]["text"]
    assert other == i18n.t("platform.telegram.prompt.other", lang="ko")
    assert other != i18n.t("platform.telegram.prompt.other", lang="en")
    await _tap(adapter)
    assert entry.response == "B"
    assert [event["kind"] for event in events] == [
        "prompt_delivered",
        "original_callback",
        "prompt_closed",
    ]
    assert {event["schema_version"] for event in events} == {2}
    assert events[2]["selected_callback_query_id"] == "tap-1"


@pytest.mark.asyncio
async def test_actual_v2_prompt_still_accepts_original_text(callback_lane):
    adapter, runner, _, events, received = callback_lane
    await _send(adapter)
    await _receive(adapter)
    await runner._handle_message(received[0])
    assert {event["schema_version"] for event in events} == {2}
    assert events[-1]["selected_reply_message_id"] == "101"
    assert events[-1]["selected_callback_query_id"] is None
    assert events[-1]["selected_callback_update_id"] is None


@pytest.mark.asyncio
async def test_missing_ack_keyboard_retains_v1_text_only(lane):
    adapter, runner, _, events, received = lane
    await _send(adapter)
    await _receive(adapter)
    await runner._handle_message(received[0])
    assert {event["schema_version"] for event in events} == {1}
    assert "choice_buttons" not in events[0]
    assert "selected_callback_query_id" not in events[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["label", "data", "other", "extra"])
async def test_actual_ack_entire_keyboard_must_match_sent(callback_lane, change):
    adapter, _, _, events, _ = callback_lane

    async def mismatched(**kwargs):
        markup = copy.deepcopy(kwargs["reply_markup"].to_dict())
        rows = markup["inline_keyboard"]
        if change == "label":
            rows[0][0]["text"] = "Different"
        elif change == "data":
            rows[0][0]["callback_data"] = "cl:cid:1"
        elif change == "other":
            rows.pop()
        else:
            rows.append([{"text": "More", "callback_data": "more"}])
        return _message(100, bot=True, reply_markup=markup)

    adapter._bot.send_message.side_effect = mismatched
    assert (await _send(adapter)).success
    await _tap(adapter)
    assert [event["kind"] for event in events] == ["prompt_delivered", "prompt_closed"]
    assert {event["schema_version"] for event in events} == {1}


@pytest.mark.asyncio
async def test_gateway_auth_rejection_prevents_callback_evidence(callback_lane):
    adapter, runner, entry, events, _ = callback_lane
    await _send(adapter)
    runner._is_user_authorized = lambda source: False
    await _tap(adapter)
    assert entry.response == "B"
    assert [event["kind"] for event in events] == ["prompt_delivered", "prompt_closed"]
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
async def test_native_callback_auth_rejection_preserves_pending(callback_lane):
    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)
    adapter.set_authorization_check(lambda *args, **kwargs: False)
    await _tap(adapter)
    assert not entry.event.is_set()
    assert len(events) == 1


@pytest.mark.asyncio
async def test_allowed_other_member_is_never_claimed_as_owner(callback_lane):
    adapter, _, _, events, _ = callback_lane
    await _send(adapter)
    await _tap(adapter, user_id=888)
    assert events[1]["user_id"] == "888"
    assert "owner" not in events[1]


@pytest.mark.asyncio
async def test_other_then_typed_answer_has_no_choice_label(callback_lane):
    adapter, runner, entry, events, received = callback_lane
    await _send(adapter)
    await _tap(adapter, token="other")
    assert entry.awaiting_text
    assert events[-1]["reason"] == "edited"
    await _receive(adapter)
    await runner._handle_message(received[0])
    assert len(events) == 2
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
async def test_duplicate_callback_and_postanswer_edit_do_not_rewrite(callback_lane):
    adapter, _, _, events, _ = callback_lane
    await _send(adapter)
    update = await _tap(adapter)
    accepted = copy.deepcopy(events)
    await adapter._handle_callback_query(update, None)
    adapter._rich_eligible = lambda content: False
    adapter._bot.edit_message_text = AsyncMock()
    await adapter.edit_message("123", "100", "Already answered")
    cm.clear_session("sk")
    assert events == accepted


@pytest.mark.asyncio
async def test_ui_ack_or_edit_failure_does_not_erase_owner_tap(
    callback_lane, monkeypatch
):
    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)
    monkeypatch.setattr(
        CallbackQuery,
        "edit_message_text",
        AsyncMock(side_effect=RuntimeError("failed")),
    )
    await _tap(adapter)
    assert entry.response == "B"
    assert events[-1]["selected_callback_query_id"] == "tap-1"


@pytest.mark.asyncio
async def test_capture_pipeline_bug_warns_class_name_and_still_resolves(
    callback_lane, monkeypatch, caplog
):
    import logging

    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)

    def boom(*args):
        raise RuntimeError("secret payload cl:cid:1")

    monkeypatch.setattr("gateway.telegram_clarify_callbacks._candidate", boom)
    with caplog.at_level(
        logging.WARNING, logger="gateway.telegram_clarify_callbacks"
    ):
        await _tap(adapter)
    assert entry.response == "B"
    records = [
        record
        for record in caplog.records
        if "Clarify callback observer failed" in record.getMessage()
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "RuntimeError" in records[0].getMessage()
    assert "secret payload" not in caplog.text
    assert "cl:cid:1" not in caplog.text
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
async def test_authorize_pipeline_bug_warns_class_name_and_still_resolves(
    callback_lane, monkeypatch, caplog
):
    import logging

    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)

    def boom(event, source):
        raise KeyError("secret snapshot text")

    monkeypatch.setattr("gateway.telegram_clarify_evidence.authorize_candidate", boom)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _tap(adapter)
    assert entry.response == "B"
    records = [
        record
        for record in caplog.records
        if "Clarify evidence authorization failed" in record.getMessage()
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "KeyError" in records[0].getMessage()
    assert "secret snapshot text" not in caplog.text
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
async def test_other_branch_edit_escapes_question_html(callback_lane):
    adapter, _, entry, _, _ = callback_lane
    await _send(adapter)
    await _tap(adapter, token="other")
    assert entry.awaiting_text
    text = CallbackQuery.edit_message_text.call_args.kwargs["text"]
    assert "Awaiting typed response" in text
    assert "&lt;one&gt;" in text
    assert "<one>" not in text
