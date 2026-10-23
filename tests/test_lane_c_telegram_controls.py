"""Lane C: real dispatch rejects evidence forgery without breaking ordinary replies."""
import copy

import pytest

from .lane_c_telegram_support import lane_c, receive, resolve_text, send, tap
from tools import clarify_gateway as cm

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("token", ["99", "-1", "garbage"])
async def test_invalid_choice_leaves_prompt_open_then_real_tap_resolves(lane_c, token):
    await send(lane_c)
    await tap(lane_c, token)
    assert not lane_c.entry.event.is_set()
    assert lane_c.entry.response is None
    assert lane_c.adapter._clarify_state["lane-c"] == "lane-c-session"
    assert [e["kind"] for e in lane_c.events] == ["prompt_delivered"]
    await tap(lane_c, "1", update_id=71)
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_callback_update_id"] == "71"


@pytest.mark.parametrize("token", ["01", "+1", " 1"])
async def test_numeric_alias_resolves_normally_without_faking_delivered_button(lane_c, token):
    await send(lane_c)
    await tap(lane_c, token)
    assert lane_c.entry.response == "B"
    assert [e["kind"] for e in lane_c.events] == ["prompt_delivered", "prompt_closed"]
    assert lane_c.events[-1]["selected_callback_query_id"] is None


@pytest.mark.parametrize("changes", [
    {"message_id": 999}, {"message_thread_id": 42}, {"text": "Changed"},
    {"chat": {"id": 999, "type": "private"}}, {"date": 0},
    {"from": {"id": 901, "is_bot": True, "first_name": "Other"}},
    {"from": {"id": 900, "is_bot": False, "first_name": "Human"}},
    {"edit_date": 1700000000}, {"reply_markup": None},
    {"forward_origin": {"type": "hidden_user", "date": 1700000000, "sender_user_name": "Other"}},
    {"sender_chat": {"id": -44, "type": "channel", "title": "Anonymous"}},
])
async def test_changed_prompt_keeps_public_response_without_original_evidence(lane_c, changes):
    await send(lane_c)
    await tap(lane_c, "1", message_changes=changes)
    assert lane_c.entry.response == "B"
    assert [e["kind"] for e in lane_c.events] == ["prompt_delivered", "prompt_closed"]
    assert lane_c.events[-1]["selected_callback_query_id"] is None


@pytest.mark.parametrize("changes", [
    {"from": {"id": 777, "is_bot": True, "first_name": "Bot"}},
    {"inline_message_id": "inline"}, {"id": ""}, {"id": "q" * 129},
])
async def test_nonoriginal_queries_cannot_become_evidence(lane_c, changes):
    await send(lane_c)
    await tap(lane_c, "1", query_changes=changes)
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_callback_query_id"] is None


@pytest.mark.parametrize("kind", ["callback", "text"])
async def test_native_authorization_rejection_preserves_pending(lane_c, kind):
    await send(lane_c)
    lane_c.adapter.set_authorization_check(lambda *args, **kwargs: False)
    if kind == "callback":
        await tap(lane_c)
    else:
        # Unknown DMs deliberately pass the prefilter for gateway pairing.
        # An explicit adapter allowlist makes this a real denial instead.
        lane_c.adapter.config.extra["allow_from"] = ["999"]
        assert await receive(lane_c) is None
    assert not lane_c.entry.event.is_set()
    assert [e["kind"] for e in lane_c.events] == ["prompt_delivered"]
    assert lane_c.observations == []


@pytest.mark.parametrize("field,value", [("profile", "other"), ("thread_id", "42"), ("user_id", "888")])
async def test_authorized_source_still_must_match_original_route(lane_c, field, value):
    from gateway.telegram_clarify_evidence import authorize_candidate

    async def rerouted(event, source):
        setattr(source, field, value)
        return authorize_candidate(event, source)

    lane_c.adapter.set_platform_event_handler(rerouted)
    await send(lane_c)
    await tap(lane_c, "1")
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_callback_query_id"] is None


@pytest.mark.parametrize("changes", [
    {"forward_date": 1700000000}, {"edit_date": 1700000000},
    {"reply_to_message": None},
    {"sender_chat": {"id": -44, "type": "channel", "title": "Anonymous"}},
])
async def test_nonoriginal_text_still_flows_without_receipt(lane_c, changes):
    await send(lane_c)
    event = await receive(lane_c, **changes)
    assert event is not None
    assert resolve_text(event) == cm.TEXT_RESOLVED
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_reply_message_id"] is None


async def test_edited_update_invalidates_pending_evidence(lane_c):
    await send(lane_c)
    await receive(lane_c, edited=True, edit_date=1700000000)
    assert lane_c.events[-1]["reason"] == "edited"
    frozen = copy.deepcopy(lane_c.events)
    await tap(lane_c)
    assert lane_c.events == frozen


async def test_repeated_callback_does_not_rewrite_history(lane_c):
    await send(lane_c)
    update = await tap(lane_c)
    frozen = copy.deepcopy(lane_c.events)
    await lane_c.adapter._handle_callback_query(update, None)
    assert lane_c.events == frozen


async def test_other_authorized_user_is_not_relabelled_as_owner(lane_c):
    await send(lane_c)
    await tap(lane_c, query_changes={"from": {"id": 888, "is_bot": False, "first_name": "Other"}})
    assert lane_c.events[1]["user_id"] == "888"
    assert "owner" not in lane_c.events[1]
