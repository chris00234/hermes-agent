"""Native SDK transport harness; only transport/observer sinks are replaced."""
import html
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Bot, CallbackQuery, Message, Update

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.telegram_clarify_evidence import authorize_candidate
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import clarify_gateway as cm
from tools.clarify_evidence import for_source, selected_for_event
from tools.clarify_tool import mark_recommended

QUESTION = "Pick <one>"
CHOICES = ["A", "B"]


def message(message_id=101, *, bot=False, text="2", **changes):
    data = {
        "message_id": message_id,
        "date": int(datetime.now(timezone.utc).timestamp()),
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 900 if bot else 777, "is_bot": bot, "first_name": "User"},
        "text": text,
    }
    if not bot:
        data["reply_to_message"] = message(100, bot=True).to_dict()
    data.update(changes)
    return Message.de_json(data, Bot("123:lane-c-test-token"))


@pytest.fixture
def lane_c(monkeypatch):
    monkeypatch.setattr(cm, "_entries", {})
    monkeypatch.setattr(cm, "_session_index", {})
    monkeypatch.setattr(cm, "get_clarify_timeout", lambda: 600)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "liz")
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    events = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: events.append(kwargs["payload"]),
    )
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(CallbackQuery, "edit_message_text", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:lane-c-test-token"))
    adapter._bot = MagicMock(id=900, username="test_bot")
    adapter._bot.send_message = AsyncMock()
    adapter._bot.edit_message_text = AsyncMock()
    adapter.set_authorization_check(lambda *args, **kwargs: True)
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    received, observations = [], []
    adapter._enqueue_text_event = received.append

    async def observe(event, source):
        observations.append((event, source))
        return authorize_candidate(event, source)

    async def ack(**kwargs):
        markup = kwargs.get("reply_markup")
        prompt = message(
            100, bot=True, text=html.unescape(kwargs["text"]),
            reply_markup=markup.to_dict() if markup else None,
        )
        state.prompt = prompt
        return prompt

    adapter.set_platform_event_handler(observe)
    adapter._bot.send_message.side_effect = ack
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", user_id="777",
        chat_type="dm", profile="liz",
    )
    state = SimpleNamespace(
        adapter=adapter, source=source, events=events, received=received,
        observations=observations, prompt=None, entry=None,
    )
    yield state
    cm.clear_session("lane-c-session")


async def send(state, *, choices=CHOICES, multi_select=False, evidence=True):
    decorated = mark_recommended(list(choices)) if choices else None
    receipt = for_source(state.source, "lane-c", QUESTION, decorated, multi_select) if evidence else None
    state.entry = cm.register(
        "lane-c", "lane-c-session", QUESTION, decorated,
        multi_select=multi_select, evidence=receipt,
    )
    return await state.adapter.send_clarify(
        "123", QUESTION, decorated, "lane-c", "lane-c-session",
    )


def callback(state, token="0", *, message_changes=None, query_changes=None, update_id=70):
    prompt = state.prompt.to_dict()
    prompt.update(message_changes or {})
    query = {
        "id": "lane-c-tap", "from": {"id": 777, "is_bot": False, "first_name": "User"},
        "chat_instance": "lane-c-native-chat", "message": prompt,
        "data": f"cl:lane-c:{token}",
    }
    query.update(query_changes or {})
    return Update.de_json(
        {"update_id": update_id, "callback_query": query}, Bot("123:lane-c-test-token"),
    )


async def tap(state, token="0", **kwargs):
    update = callback(state, token, **kwargs)
    await state.adapter._handle_callback_query(update, None)
    return update


async def receive(state, text="2", *, edited=False, **changes):
    incoming = message(text=text, **changes)
    payload = {"edited_message" if edited else "message": incoming}
    await state.adapter._handle_text_message(Update(update_id=80, **payload), None)
    return state.received[-1] if state.received else None


def resolve_text(event):
    return cm.attempt_text_response_for_session(
        "lane-c-session", event.text, selected_reply=selected_for_event(event, event.text),
    )
