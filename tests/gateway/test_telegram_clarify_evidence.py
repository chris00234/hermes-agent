from datetime import datetime, timezone, UTC
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Each file runs in an isolated process. Replace the directory's optional-SDK
# stubs so these transport tests exercise real PTB Message/Update semantics.
for module_name in list(sys.modules):
    if module_name == "telegram" or module_name.startswith("telegram."):
        if isinstance(sys.modules[module_name], MagicMock):
            del sys.modules[module_name]

pytest.importorskip("telegram", reason="Requires the native Telegram SDK")
from telegram import Bot, Message, Update

from gateway.config import PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import clarify_gateway as cm
from tools.clarify_evidence import ClarifyEvidence, selected_for_event


@pytest.fixture
def lane(monkeypatch):
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
    events = []

    def observe(name, **kwargs):
        if name == "gateway_platform_event":
            events.append(kwargs["payload"])
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", observe)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "liz")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:test-token"))
    adapter._bot = MagicMock(id=900, username="test_bot")
    adapter._bot.send_message = AsyncMock(return_value=_message(100, bot=True))
    adapter._is_user_authorized_from_message = lambda message: True
    adapter._should_process_message = lambda message: True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    received = []
    adapter._enqueue_text_event = received.append
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._session_key_for_source = lambda source: "sk"
    runner._adapter_for_source = lambda source: adapter
    runner._update_prompt_pending = {}
    adapter.set_platform_event_handler(runner._handle_gateway_platform_event)
    evidence = ClarifyEvidence("cid", "liz", "Pick <one>", ["A", "B"], 600)
    entry = cm.register("cid", "sk", "Pick <one>", ["A", "B"], evidence=evidence)
    yield adapter, runner, entry, events, received
    cm.clear_session("sk")


def _message(message_id, *, bot=False, text="2", user_id=777, reply_id=100, **extra):
    data = {
        "message_id": message_id,
        "date": int(datetime.now(UTC).timestamp()),
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 900 if bot else user_id, "is_bot": bot, "first_name": "User"},
        "text": text,
        **extra,
    }
    if not bot and reply_id is not None:
        data["reply_to_message"] = {
            "message_id": reply_id,
            "date": data["date"],
            "chat": data["chat"],
            "from": {"id": 900, "is_bot": True, "first_name": "Bot"},
            "text": "Prompt",
        }
    return Message.de_json(data, Bot("123:test-token"))


async def _send(adapter):
    return await adapter.send_clarify("123", "Pick <one>", ["A", "B"], "cid", "sk")


async def _receive(adapter, message=None, update_id=50):
    update = Update(update_id=update_id, message=message or _message(101))
    await adapter._handle_text_message(update, None)


@pytest.mark.asyncio
async def test_real_adapter_and_gateway_resolve_exact_original_reply(lane):
    adapter, runner, entry, events, received = lane
    assert (await _send(adapter)).success
    await _receive(adapter)
    assert await runner._handle_message(received[0]) == ""
    assert entry.response == "B"
    assert [e["kind"] for e in events] == [
        "prompt_delivered",
        "original_reply",
        "prompt_closed",
    ]
    assert [e["sequence"] for e in events] == [1, 2, 3]
    assert (
        events[0]["rendered_text"] == adapter._bot.send_message.call_args.kwargs["text"]
    )
    assert "&lt;one&gt;" in events[0]["rendered_text"]
    assert events[0]["options"] == ["A", "B"]
    assert events[1]["text"] == "2"
    assert events[2]["selected_reply_message_id"] == "101"
    assert events[2]["selected_reply_update_id"] == "50"
    assert events[2]["reason"] == "resolved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["batch", "internal", "actor", "update", "message"]
)
async def test_mutated_or_synthetic_event_cannot_claim_original_resolution(
    lane, mutation
):
    adapter, _, _, events, received = lane
    await _send(adapter)
    await _receive(adapter)
    event = received[0]
    if mutation == "batch":
        event.text += "\n2"
    elif mutation == "internal":
        event.internal = True
    elif mutation == "actor":
        event.source.user_id = "999"
    elif mutation == "update":
        event.platform_update_id = 51
    else:
        event.message_id = "102"
    selected = selected_for_event(event, event.text)
    assert selected is None
    cm.resolve_gateway_clarify("cid", "B", selected_reply=selected)
    assert events[-1]["selected_reply_message_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"forward_date": 1700000000},
        {"edit_date": 1700000000},
        {"sender_chat": {"id": -44, "type": "channel", "title": "Anonymous"}},
        {
            "forward_origin": {
                "type": "hidden_user",
                "date": 1700000000,
                "sender_user_name": "Other",
            }
        },
        {"from": {"id": 777, "is_bot": True, "first_name": "Bot"}},
    ],
)
async def test_nonoriginal_messages_preserve_flow_without_receipts(lane, extra):
    adapter, _, _, events, received = lane
    await _send(adapter)
    await _receive(adapter, _message(101, **extra))
    assert len(received) == 1
    assert [e["kind"] for e in events] == ["prompt_delivered"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_id", [None, 999])
async def test_reply_must_target_actual_prompt_ack(lane, reply_id):
    adapter, _, _, events, received = lane
    await _send(adapter)
    await _receive(adapter, _message(101, reply_id=reply_id))
    assert received
    assert len(events) == 1


@pytest.mark.asyncio
async def test_gateway_authorization_runs_before_original_event(lane):
    adapter, runner, _, events, received = lane
    await _send(adapter)
    runner._is_user_authorized = lambda source: False
    await _receive(adapter)
    assert received
    assert len(events) == 1


@pytest.mark.asyncio
async def test_other_allowed_group_member_keeps_real_identity(lane):
    adapter, _, _, events, _ = lane
    await _send(adapter)
    await _receive(adapter, _message(101, user_id=888))
    assert events[-1]["user_id"] == "888"
    assert "owner" not in events[-1]


@pytest.mark.asyncio
async def test_failed_send_has_no_delivery(lane):
    adapter, _, _, events, _ = lane
    adapter._bot.send_message.side_effect = RuntimeError("definitive failure")
    assert not (await _send(adapter)).success
    cm.clear_session("sk")
    assert events == []


@pytest.mark.asyncio
async def test_late_ack_after_reset_cannot_revive_prompt(lane):
    adapter, _, _, events, _ = lane

    async def send(**kwargs):
        cm.clear_session("sk")
        return _message(100, bot=True)

    adapter._bot.send_message.side_effect = send
    assert (await _send(adapter)).success
    assert events == []


@pytest.mark.asyncio
async def test_timeout_reset_and_prompt_edit_close_at_most_once(lane):
    adapter, _, entry, events, _ = lane
    await _send(adapter)
    adapter._rich_eligible = lambda content: False
    adapter._bot.edit_message_text = AsyncMock()
    await adapter.edit_message("123", "100", "changed prompt")
    assert events[-1]["reason"] == "edited"
    cm.clear_session("sk")
    assert len(events) == 2
    assert entry.evidence.closed


@pytest.mark.asyncio
async def test_actual_timeout_emits_terminal_event(lane):
    adapter, _, _, events, _ = lane
    await _send(adapter)
    assert cm.wait_for_response("cid", 0.001) is None
    assert events[-1]["reason"] == "timeout"
    cm.clear_session("sk")
    assert len(events) == 2


@pytest.mark.asyncio
async def test_postanswer_edit_preserves_historical_closure(lane):
    adapter, runner, _, events, received = lane
    await _send(adapter)
    await _receive(adapter)
    await runner._handle_message(received[0])
    adapter._rich_eligible = lambda content: False
    adapter._bot.edit_message_text = AsyncMock()
    await adapter.edit_message("123", "100", "Answered")
    assert len(events) == 3
    assert events[-1]["reason"] == "resolved"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["reset", "interrupt", "shutdown"])
async def test_real_gateway_boundaries_close_pending_prompt(lane, boundary):
    adapter, runner, entry, events, _ = lane
    await _send(adapter)
    runner._peek_session_state = lambda key: None
    if boundary == "reset":
        runner._clear_session_boundary_security_state("sk")
    elif boundary == "interrupt":
        runner._begin_session_run_generation = lambda key: 2
        runner._invalidate_session_run_generation("sk", reason="stop")
    else:
        runner._finalize_session_off_loop = AsyncMock()
        runner._cleanup_agent_resources_off_loop = AsyncMock()
        await runner._finalize_shutdown_agents({"sk": object()})
    assert entry.event.is_set()
    assert events[-1]["reason"] == (
        "cancelled" if boundary == "interrupt" else boundary
    )
    assert not cm.has_pending("sk")


@pytest.mark.asyncio
async def test_late_successful_ack_while_pending_remains_evidence(lane):
    import asyncio

    adapter, _, _, events, _ = lane
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_ack(**kwargs):
        entered.set()
        await release.wait()
        return _message(100, bot=True)

    adapter._bot.send_message.side_effect = delayed_ack
    task = asyncio.create_task(_send(adapter))
    await entered.wait()
    assert events == []
    release.set()
    assert (await task).success
    assert events[0]["transport_ack"] is True
