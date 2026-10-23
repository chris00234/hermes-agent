import copy
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_telegram_clarify_callback_evidence import (
    CallbackQuery,
    _tap,
    _update,
    callback_lane,
)
from tests.gateway.test_telegram_clarify_evidence import _message, _receive, _send, lane
from tools import clarify_gateway as cm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_changes",
    [
        {"message_id": 999},
        {"chat": {"id": -100, "type": "group", "title": "Other group"}},
        {"message_thread_id": 42},
        {"from": {"id": 901, "is_bot": True, "first_name": "Other bot"}},
        {"from": {"id": 900, "is_bot": False, "first_name": "Human"}},
        {"edit_date": 1700000000},
        {
            "forward_origin": {
                "type": "hidden_user",
                "date": 1700000000,
                "sender_user_name": "Forwarder",
            }
        },
        {"sender_chat": {"id": -44, "type": "channel", "title": "Anonymous"}},
        {"date": 0},
        {"text": "Changed prompt"},
        {"reply_markup": None},
    ],
)
async def test_callback_prompt_must_match_original_eligible_ack(
    callback_lane, message_changes
):
    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)
    await _tap(adapter, message_changes=message_changes)
    assert entry.response == "B"
    assert [event["kind"] for event in events] == ["prompt_delivered", "prompt_closed"]
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_changes",
    [
        {"from": {"id": 777, "is_bot": True, "first_name": "Bot"}},
        {"inline_message_id": "inline-context"},
        {"id": ""},
        {"id": "q" * 129},
    ],
)
async def test_nonoriginal_query_cannot_become_owner_evidence(
    callback_lane, query_changes
):
    adapter, _, _, events, _ = callback_lane
    await _send(adapter)
    await _tap(adapter, query_changes=query_changes)
    assert len(events) == 2
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["01", "+1", " 1"])
async def test_parsed_numeric_alias_does_not_fake_delivered_button(
    callback_lane, token
):
    adapter, _, _, events, _ = callback_lane
    await _send(adapter)
    await _tap(adapter, token=token)
    assert len(events) == 2
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["99", "-1"])
async def test_out_of_range_live_index_never_resolves(callback_lane, token):
    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)
    await _tap(adapter, token=token)
    assert not entry.event.is_set()
    assert entry.response is None
    assert adapter._clarify_state.get("cid") == "sk"
    assert [event["kind"] for event in events] == ["prompt_delivered"]
    await _tap(adapter, token="1", update_id=71, query_id="tap-2")
    assert entry.response == "B"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("profile", "jenna"), ("thread_id", "42"), ("user_id", "888")]
)
async def test_callback_authorization_scope_cannot_change_original_route(
    callback_lane, field, value
):
    adapter, runner, _, events, _ = callback_lane
    await _send(adapter)

    async def rerouted(event, source):
        setattr(source, field, value)
        return await runner._handle_gateway_platform_event(event, source)

    adapter.set_platform_event_handler(rerouted)
    await _tap(adapter)
    assert len(events) == 2
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
async def test_pending_text_candidate_cannot_steal_winning_callback(callback_lane):
    adapter, _runner, _, events, received = callback_lane
    await _send(adapter)
    await _receive(adapter)
    await _tap(adapter)
    assert events[-1]["selected_reply_message_id"] is None
    assert events[-1]["selected_callback_query_id"] == "tap-1"
    accepted = copy.deepcopy(events)
    assert not cm.resolve_gateway_clarify(
        "cid", "B", selected_reply=received[0]._clarify_original_reply
    )
    assert events == accepted


@pytest.mark.asyncio
async def test_callback_that_loses_to_text_during_auth_cannot_be_selected(
    callback_lane,
):
    adapter, runner, _, events, received = callback_lane
    await _send(adapter)
    await _receive(adapter)

    async def interleave(event, source):
        candidate = await runner._handle_gateway_platform_event(event, source)
        await runner._handle_message(received[0])
        return candidate

    adapter.set_platform_event_handler(interleave)
    await _tap(adapter)
    assert events[-1]["selected_reply_message_id"] == "101"
    assert events[-1]["selected_callback_query_id"] is None
    assert [event["kind"] for event in events] == [
        "prompt_delivered",
        "original_reply",
        "original_callback",
        "prompt_closed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["reset", "timeout", "edited", "cancelled", "shutdown"]
)
async def test_terminal_boundary_before_callback_cannot_reopen_evidence(
    callback_lane, boundary
):
    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)
    if boundary == "timeout":
        cm.wait_for_response("cid", 0.001)
    elif boundary == "edited":
        adapter._rich_eligible = lambda content: False
        adapter._bot.edit_message_text = AsyncMock()
        await adapter.edit_message("123", "100", "Changed")
    else:
        cm.clear_session("sk", reason=boundary)
    closed = copy.deepcopy(events)
    await _tap(adapter)
    assert events == closed
    assert entry.evidence.closed


@pytest.mark.asyncio
async def test_ack_after_reset_or_failed_send_cannot_create_callback_evidence(
    callback_lane,
):
    adapter, _, _, events, _ = callback_lane

    async def late_ack(**kwargs):
        cm.clear_session("sk")
        return _message(100, bot=True, reply_markup=kwargs["reply_markup"].to_dict())

    adapter._bot.send_message.side_effect = late_ack
    assert (await _send(adapter)).success
    await _tap(adapter)
    assert events == []


@pytest.mark.asyncio
async def test_callback_answer_transport_failure_preserves_accepted_history(
    callback_lane, monkeypatch
):
    adapter, _, entry, events, _ = callback_lane
    await _send(adapter)
    monkeypatch.setattr(
        CallbackQuery, "answer", AsyncMock(side_effect=RuntimeError("failed"))
    )
    with pytest.raises(RuntimeError, match="failed"):
        await _tap(adapter)
    assert entry.event.is_set()
    assert events[-1]["selected_callback_query_id"] == "tap-1"


@pytest.mark.asyncio
async def test_replayed_sdk_query_or_update_never_emits_second_candidate(callback_lane):
    from gateway.telegram_clarify_callbacks import capture

    adapter, _, _, events, _ = callback_lane
    await _send(adapter)
    first = await capture(adapter, _update(adapter), "cid")
    assert first is not None
    assert await capture(adapter, _update(adapter, update_id=71), "cid") is None
    assert await capture(adapter, _update(adapter, query_id="tap-2"), "cid") is None
    cm.resolve_gateway_clarify("cid", "B", selected_callback=first)
    assert len(events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("both", [False, True])
async def test_mismatched_response_or_both_selection_types_abstain(callback_lane, both):
    from gateway.telegram_clarify_callbacks import capture

    adapter, _, _, events, received = callback_lane
    await _send(adapter)
    callback = await capture(adapter, _update(adapter), "cid")
    if both:
        await _receive(adapter)
    assert cm.resolve_gateway_clarify(
        "cid",
        "B" if both else "A",
        selected_callback=callback,
        selected_reply=received[0]._clarify_original_reply if both else None,
    )
    assert events[-1]["selected_callback_query_id"] is None


@pytest.mark.asyncio
async def test_nonforum_supergroup_thread_reply_is_admitted(lane):
    adapter, _, _, events, received = lane
    chat = {"id": -100777, "type": "supergroup", "title": "Team"}

    async def ack(**kwargs):
        return _message(100, bot=True, chat=chat, message_thread_id=55)

    adapter._bot.send_message.side_effect = ack
    assert (await _send(adapter)).success
    await _receive(adapter, message=_message(101, chat=chat, message_thread_id=55))
    assert [event["kind"] for event in events] == [
        "prompt_delivered",
        "original_reply",
    ]
    assert events[0]["thread_id"] == "55"
    assert received[0]._clarify_original_reply is not None


@pytest.mark.asyncio
async def test_reply_from_different_thread_is_rejected(lane):
    adapter, _, _, events, received = lane
    chat = {"id": -100777, "type": "supergroup", "title": "Team"}

    async def ack(**kwargs):
        return _message(100, bot=True, chat=chat, message_thread_id=55)

    adapter._bot.send_message.side_effect = ack
    assert (await _send(adapter)).success
    await _receive(adapter, message=_message(101, chat=chat, message_thread_id=56))
    assert [event["kind"] for event in events] == ["prompt_delivered"]
    assert getattr(received[0], "_clarify_original_reply", None) is None
