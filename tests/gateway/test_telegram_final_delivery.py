"""Regression coverage for Telegram final delivery after streamed edit failure."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = True
    adapter.RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.edit_message = AsyncMock()
    adapter.send = AsyncMock()
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_turn_final_flood_immediately_delivers_missing_tail():
    """A short visible preview must not suppress the completed answer."""
    adapter = _adapter()
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 180 seconds",
        retry_after=180.0,
    )
    adapter.send.return_value = SendResult(success=True, message_id="tail-1")

    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(cursor=" ▉"),
        metadata={"thread_id": "77"},
    )
    consumer._message_id = "preview-1"
    consumer._last_sent_text = ":("
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        ":( The completed answer follows here.",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._flood_strikes == 1
    assert consumer._fallback_final_send is True
    assert consumer.final_content_delivered is False
    assert adapter.edit_message.await_count == 1

    await consumer._send_fallback_final(":( The completed answer follows here.")

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "The completed answer follows here."
    assert adapter.send.await_args.kwargs["metadata"] == {
        "thread_id": "77",
        "notify": True,
    }
    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_non_opt_in_adapter_keeps_adaptive_final_edit_retry():
    """Immediate final fallback remains scoped to opted-in adapters."""
    adapter = _adapter()
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = False
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 30 seconds",
        retry_after=30.0,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "partial"
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        "partial plus final",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._flood_strikes == 1
    assert consumer._fallback_final_send is False


@pytest.mark.asyncio
async def test_empty_tail_commit_honors_retry_after(monkeypatch):
    adapter = _adapter()
    adapter.send.side_effect = [
        SendResult(
            success=False,
            error="Flood control exceeded",
            retry_after=3.0,
        ),
        SendResult(success=True, message_id="final-1"),
    ]
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    assert adapter.send.await_count == 2
    sleep.assert_awaited_once_with(3.0)
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_complete_preview_survives_long_flood_fallback_failure(monkeypatch):
    """A complete ACKed preview must not trigger a duplicate normal final."""
    adapter = _adapter()
    adapter.send.return_value = SendResult(
        success=False,
        error="flood_control:20.0",
        retry_after=20.0,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._already_sent = True
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.send.assert_awaited_once()
    sleep.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("Final answer") is True


@pytest.mark.asyncio
async def test_telegram_long_flood_result_keeps_retry_after():
    """The real adapter contract preserves the server delay for consumers."""
    class FloodError(Exception):
        retry_after = 30.0

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock(side_effect=FloodError("Retry after 30"))

    result = await adapter.edit_message("123", "456", "Final answer", finalize=False)

    assert result.success is False
    assert result.error == "flood_control:30.0"
    assert result.retry_after == 30.0




@pytest.mark.asyncio
async def test_empty_fallback_resend_preserves_reply_anchor():
    """The fresh-commit resend must carry the turn's reply anchor (#71047).

    With reply_to_mode='first' the streamed preview is delivered as a reply
    to the user's message. When a failed finalize edit forces the fresh
    resend, the replacement message must use the same anchor so the visible
    behavior matches the preview (and the non-streaming path).
    """
    adapter = _adapter()
    adapter.send.return_value = SendResult(success=True, message_id="final-1")

    consumer = GatewayStreamConsumer(
        adapter, "chat-1", initial_reply_to_id="111",
    )
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._already_sent = True
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.send.assert_awaited_once()
    kwargs = adapter.send.await_args.kwargs
    assert kwargs.get("reply_to") == "111"
    # Preview replaced: deleted after the fresh final succeeded.
    adapter.delete_message.assert_awaited_once_with("chat-1", "preview-1")
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_empty_fallback_preview_delete_retries_once(monkeypatch):
    """A False (flood-rejected) preview delete gets one bounded retry."""
    adapter = _adapter()
    adapter.send.return_value = SendResult(success=True, message_id="final-1")
    adapter.delete_message = AsyncMock(side_effect=[False, True])
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._already_sent = True
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    assert adapter.delete_message.await_count == 2
    sleep.assert_awaited_once_with(1.0)
    assert consumer.final_response_sent is True


@pytest.mark.asyncio
async def test_flood_capped_resend_keeps_single_bubble_reply_first(monkeypatch):
    """#71047 Problem B end-to-end shape: preview as reply, finalize edit and
    fresh resend both flood-capped — the gateway suppression decision must
    keep the complete ACKed preview as the single visible bubble instead of
    letting the normal final send create a second one.
    """
    adapter = _adapter()
    # Fresh-commit resend flood-capped past the inline retry budget.
    adapter.send.return_value = SendResult(
        success=False,
        error="flood_control:41.0",
        retry_after=41.0,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(
        adapter, "chat-1", initial_reply_to_id="111",
    )
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._already_sent = True
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    # Preview must NOT be deleted — it is the only copy of the answer.
    adapter.delete_message.assert_not_awaited()
    # Mirror the gateway/run.py suppression decision: content delivered and
    # the recorded payload reconciles, so the normal final send is skipped.
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("Final answer") is True


@pytest.mark.asyncio
async def test_long_edit_flood_blocks_stream_and_final_sends_until_deadline(monkeypatch):
    from gateway import delivery_ledger as ledger

    class FloodError(Exception):
        retry_after = 185.0

    now = [100.0]
    monkeypatch.setattr("plugins.platforms.telegram.adapter.time.monotonic", lambda: now[0])
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock(side_effect=FloodError("Retry after 185"))
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=789))
    consumer = GatewayStreamConsumer(adapter, "123", StreamConsumerConfig(cursor=""))
    consumer._message_id = "456"
    consumer._last_sent_text = "Partial"
    consumer._already_sent = True

    assert not await consumer._send_or_edit("Partial completed answer", finalize=True, is_turn_final=True)
    assert consumer._current_edit_interval == 185.0
    await consumer._send_fallback_final("Partial completed answer")
    assert not consumer.final_content_delivered
    adapter._bot.edit_message_text.assert_awaited_once()  # no Markdown-to-plain flood retry
    adapter._bot.send_message.assert_not_awaited()

    now[0] += 184.0
    result = await adapter.send("123", "Partial completed answer", metadata={"notify": True})
    assert not result.success and result.retry_after == 1.0
    adapter._bot.send_message.assert_not_awaited()
    ledger.record_obligation(obligation_id="flood-final", session_key="session", platform="telegram",
                             chat_id="123", thread_id=None, content="Partial completed answer")
    ledger.mark_failed("flood-final", result.error)
    import time
    assert ledger.sweep_failed_for_runtime("telegram", now=time.time()) == []
    due = ledger.sweep_failed_for_runtime("telegram", now=time.time() + 2.0)
    assert len(due) == 1 and due[0]["needs_marker"]

    now[0] += 2.0
    result = await adapter.send("123", due[0]["content"], metadata={"notify": True})
    assert result.success
    adapter._bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_flood_also_blocks_edits_and_other_sends():
    class FloodError(Exception):
        retry_after = 120.0

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(side_effect=FloodError("Retry after 120"))
    adapter._bot.edit_message_text = AsyncMock()
    assert not (await adapter.send("123", "answer")).success
    assert not (await adapter.edit_message("123", "456", "answer")).success
    result = await adapter.send("123", "answer")
    assert not result.success and 119 < result.retry_after <= 120
    adapter._bot.send_message.assert_awaited_once()
    adapter._bot.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_rich_edit_flood_keeps_canonical_delivery_deadline():
    class FloodError(Exception):
        retry_after = 185.0

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    result = adapter._rich_transient_result(FloodError("Retry after 185"), "rich editMessageText")
    assert result.error == "flood_control:185.0"
    assert result.retry_after == 185.0
    assert adapter._flood_cooldown_result() is not None
