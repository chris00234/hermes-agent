"""Real gateway integration contract; failures name out-of-lane porting gaps."""
import pytest

from gateway.run import GatewayRunner
from .lane_c_telegram_support import lane_c, receive, send, tap

pytestmark = pytest.mark.asyncio


async def test_gateway_platform_event_returns_native_callback_evidence(lane_c):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._is_user_authorized_for_source = lambda source: True
    lane_c.adapter.set_platform_event_handler(runner._handle_gateway_platform_event)
    assert (await send(lane_c)).success
    await tap(lane_c)
    assert lane_c.entry.response == "A (Recommended)"
    assert lane_c.events[-1]["selected_callback_query_id"] == "lane-c-tap"
    assert lane_c.events[-1]["selected_callback_update_id"] == "70"


async def test_gateway_text_intercept_passes_selected_original_reply(lane_c):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._adapter_for_source = lambda source: lane_c.adapter
    assert (await send(lane_c)).success
    # Isolate the second missing handoff: adapter -> real authorize_candidate
    # already produced the receipt, so this exercises the real inbound consumer.
    event = await receive(lane_c)
    assert event._clarify_original_reply is not None
    assert await runner._hm_clarify_reply(event, event.source, "lane-c-session") == ""
    assert lane_c.entry.response == "B"
    assert lane_c.events[-1]["selected_reply_message_id"] == "101"
    assert lane_c.events[-1]["selected_reply_update_id"] == "80"
