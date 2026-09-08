"""Native button observations, distinct from user-authored Telegram messages."""

from __future__ import annotations

import logging
from datetime import datetime

from gateway.telegram_clarify_evidence import _identity, _route, _unmodified

logger = logging.getLogger(__name__)


def _snapshot(message):
    from telegram import Message

    markup = getattr(message, "reply_markup", None)
    date = getattr(message, "date", None)
    if (
        not isinstance(message, Message)
        or not _unmodified(message)
        or not isinstance(message.text, str)
        or markup is None
        or not isinstance(date, datetime)
        or date.tzinfo is None
        or date.timestamp() <= 0
    ):
        return None
    return {"text": message.text, "keyboard": markup.to_dict()}


def delivered_keyboard(message, sent_markup, clarify_id, choices):
    snapshot = _snapshot(message)
    if (
        not snapshot
        or sent_markup is None
        or not choices
        or snapshot["keyboard"] != sent_markup.to_dict()
    ):
        return {}
    expected = [
        [{"text": str(index + 1), "callback_data": f"cl:{clarify_id}:{index}"}]
        for index in range(len(choices))
    ]
    expected.append([
        {"text": "✏️ Other (type answer)", "callback_data": f"cl:{clarify_id}:other"}
    ])
    if snapshot["keyboard"] != {"inline_keyboard": expected} or any(
        len(row[0]["callback_data"].encode("utf-8")) > 64 for row in expected
    ):
        return {}
    return {
        "callback_snapshot": snapshot,
        "choice_buttons": [
            {
                "option_index": index,
                "button_label": row[0]["text"],
                "callback_data": row[0]["callback_data"],
            }
            for index, row in enumerate(expected[:-1])
        ],
    }


def _candidate(adapter, update, clarify_id):
    from telegram import CallbackQuery, Update

    query = getattr(update, "callback_query", None)
    if not isinstance(update, Update) or not isinstance(query, CallbackQuery):
        return None
    message, user = query.message, query.from_user
    route = _route(adapter._bot, message)
    snapshot = _snapshot(message)
    user_id, update_id = _identity(user.id), _identity(update.update_id)
    query_id = _identity(query.id)
    author = getattr(message, "from_user", None)
    if (
        not route
        or snapshot is None
        or not user_id
        or not update_id
        or not query_id
        or user.is_bot is not False
        or query.inline_message_id is not None
        or _identity(getattr(author, "id", None)) != route["bot_id"]
        or getattr(author, "is_bot", None) is not True
        or not isinstance(query.data, str)
        or not 0 < len(query.data.encode("utf-8")) <= 64
    ):
        return None
    prompt_message_id = route.pop("message_id")
    # forwarded/edited/sender_chat describe the eligible prompt message.
    # CallbackQuery provides no click timestamp or user-authored text.
    return {
        "platform": "telegram",
        "event_type": "clarify_callback_candidate",
        "callback_snapshot": snapshot,
        "payload": {
            "clarify_id": clarify_id,
            **route,
            "user_id": user_id,
            "update_id": update_id,
            "callback_query_id": query_id,
            "prompt_message_id": prompt_message_id,
            "callback_data": query.data,
            "original": True,
            "is_bot": False,
            "forwarded": False,
            "edited": False,
            "sender_chat": False,
            "synthetic": False,
        },
    }


async def capture(adapter, update, clarify_id):
    from gateway.config import Platform
    from gateway.session import SessionSource

    try:
        event = _candidate(adapter, update, clarify_id)
        handler = getattr(adapter, "_platform_event_handler", None)
        if event is None or handler is None:
            return None
        query, data = update.callback_query, event["payload"]
        chat_type = query.message.chat.type
        if chat_type == "private":
            chat_type = "dm"
        elif chat_type == "supergroup":
            chat_type = "forum" if data["thread_id"] is not None else "group"
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=data["chat_id"],
            chat_type=chat_type,
            user_id=data["user_id"],
            user_name=query.from_user.first_name,
            thread_id=data["thread_id"],
        )
        return await handler(event, source)
    except Exception as exc:
        # WARNING with only the exception class: a broken v2 evidence path must
        # be operator-visible, but never leak message text or callback data.
        logger.warning("Clarify callback observer failed: %s", type(exc).__name__)
        return None
