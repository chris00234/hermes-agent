from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_telegram_clarify_evidence import (
    Update,
    _message,
    _receive,
    _send,
    lane,
)
from tools import clarify_gateway as cm
from tools.clarify_evidence import selected_for_event


@pytest.mark.asyncio
async def test_reply_edit_before_resolution_invalidates_pending_evidence(lane):
    adapter, runner, _, events, received = lane
    await _send(adapter)
    await _receive(adapter)
    edited = _message(101, text="1", edit_date=1700000000)
    await adapter._handle_text_message(Update(51, edited_message=edited), None)
    assert events[-1]["reason"] == "edited"
    await runner._handle_message(received[0])
    assert len(events) == 3
    assert events[-1]["selected_reply_message_id"] is None


@pytest.mark.asyncio
async def test_same_text_does_not_bind_another_original_reply(lane):
    adapter, runner, _, events, received = lane
    await _send(adapter)
    await _receive(adapter, _message(101), update_id=50)
    await _receive(adapter, _message(102), update_id=51)
    await runner._handle_message(received[1])
    assert events[-1]["selected_reply_message_id"] == "102"
    assert events[-1]["selected_reply_update_id"] == "51"
    assert [e["sequence"] for e in events] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_real_batching_does_not_keep_original_selection_proof(lane):
    adapter, _, _, _, received = lane
    await _send(adapter)
    await _receive(adapter, _message(101), update_id=50)
    await _receive(adapter, _message(102), update_id=51)
    adapter._should_drop_delayed_delivery = lambda: False
    adapter._flush_text_batch = AsyncMock()
    for event in received:
        type(adapter)._enqueue_text_event(adapter, event)
    combined = next(iter(adapter._pending_text_batches.values()))
    assert combined.text == "2\n2"
    assert selected_for_event(combined, combined.text) is None
    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()


@pytest.mark.asyncio
async def test_profile_mismatch_cannot_claim_delivery(lane):
    adapter, runner, _, events, _ = lane
    await _send(adapter)

    async def routed(event, source):
        source.profile = "jenna"
        return await runner._handle_gateway_platform_event(event, source)

    adapter.set_platform_event_handler(routed)
    await _receive(adapter)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_callback_close_does_not_reuse_a_pending_text_candidate(lane):
    adapter, _, _, events, _ = lane
    await _send(adapter)
    await _receive(adapter)
    cm.resolve_gateway_clarify("cid", "B")
    assert events[-1]["reason"] == "resolved"
    assert events[-1]["selected_reply_message_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("chat_id", "999"),
        ("thread_id", "42"),
        ("profile", "jenna"),
    ],
)
async def test_changed_routing_cannot_reuse_original_candidate(lane, field, value):
    adapter, _, _, _, received = lane
    await _send(adapter)
    await _receive(adapter)
    setattr(received[0].source, field, value)
    assert selected_for_event(received[0], received[0].text) is None


@pytest.mark.asyncio
async def test_oversized_wire_evidence_preserves_native_delivery(lane):
    adapter, _, _, events, _ = lane
    from tools.clarify_evidence import ClarifyEvidence

    question = ">" * 2048
    evidence = ClarifyEvidence("large", "liz", question, ["A", "B"], 7200)
    cm.register("large", "sk", question, ["A", "B"], evidence=evidence)
    result = await adapter.send_clarify("123", question, ["A", "B"], "large", "sk")
    assert result.success
    assert events == []
