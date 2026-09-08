"""Extract exact Telegram SDK identities before gateway input normalization."""

from __future__ import annotations

from datetime import datetime


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value)
    return text if text and len(text) <= 128 else None


def _route(bot, message):
    bot_id = _identity(getattr(bot, "id", None))
    chat_id = _identity(getattr(getattr(message, "chat", None), "id", None))
    message_id = _identity(getattr(message, "message_id", None))
    thread = getattr(message, "message_thread_id", None)
    thread_id = _identity(thread) if thread is not None else None
    if (
        not bot_id
        or not chat_id
        or not message_id
        or (thread is not None and not thread_id)
    ):
        return None
    return dict(
        bot_id=bot_id, chat_id=chat_id, thread_id=thread_id, message_id=message_id
    )


def delivered(adapter, clarify_id, message, question, choices, rendered_text):
    from tools import clarify_gateway as cm

    route = _route(adapter._bot, message)
    author = getattr(message, "from_user", None)
    if (
        not route
        or _identity(getattr(author, "id", None)) != route["bot_id"]
        or getattr(author, "is_bot", None) is not True
    ):
        return
    with cm._lock:
        entry = cm._entries.get(clarify_id)
        if entry and entry.evidence and not entry.event.is_set():
            entry.evidence.delivered(
                question=question, options=choices, rendered_text=rendered_text, **route
            )


def _original(message, update):
    user = getattr(message, "from_user", None)
    legacy = getattr(message, "api_kwargs", {})
    return (
        getattr(user, "is_bot", None) is False
        and getattr(update, "message", None) is message
        and getattr(update, "edited_message", None) is None
        and getattr(message, "is_automatic_forward", None) is not True
        and all(
            getattr(message, key, None) is None
            for key in (
                "sender_chat",
                "forward_origin",
                "forward_date",
                "forward_from",
                "edit_date",
            )
        )
        and not any(
            key in legacy
            for key in (
                "forward_date",
                "forward_from",
                "forward_from_chat",
                "forward_sender_name",
            )
        )
    )


def _reply_data(adapter, update, message):
    route = _route(adapter._bot, message)
    user_id = _identity(getattr(getattr(message, "from_user", None), "id", None))
    update_id = _identity(getattr(update, "update_id", None))
    reply_id = _identity(
        getattr(getattr(message, "reply_to_message", None), "message_id", None)
    )
    date, text = getattr(message, "date", None), getattr(message, "text", None)
    if (
        not route
        or not user_id
        or not update_id
        or not reply_id
        or not isinstance(text, str)
        or not 0 < len(text) <= 4096
        or not isinstance(date, datetime)
        or date.tzinfo is None
    ):
        return None
    return dict(
        **route,
        user_id=user_id,
        update_id=update_id,
        text=text,
        reply_to_message_id=reply_id,
        original_message_at=date.timestamp(),
        original=True,
        is_bot=False,
        forwarded=False,
        edited=False,
        sender_chat=False,
        synthetic=False,
    )


def candidate(adapter, update, message):
    from tools import clarify_gateway as cm

    is_edit = getattr(update, "edited_message", None) is message
    if not is_edit and not _original(message, update):
        return None
    data = _reply_data(adapter, update, message)
    if data is None:
        return None
    event_type = "clarify_original_edited" if is_edit else "clarify_original_candidate"
    with cm._lock:
        for entry in cm._entries.values():
            if (
                entry.evidence
                and not entry.event.is_set()
                and entry.evidence.matches_transport(data)
            ):
                return dict(
                    platform="telegram",
                    event_type=event_type,
                    payload={"clarify_id": entry.clarify_id, **data},
                )
    return None


def authorize_candidate(event, source):
    from hermes_cli.profiles import get_active_profile_name
    from tools import clarify_gateway as cm

    data = dict(event["payload"])
    clarify_id = data.pop("clarify_id")
    if str(source.user_id) != data["user_id"] or str(source.chat_id) != data["chat_id"]:
        return None
    with cm._lock:
        entry = cm._entries.get(clarify_id)
        if (
            entry
            and entry.evidence
            and not entry.event.is_set()
            and entry.evidence.profile == (source.profile or get_active_profile_name())
        ):
            if event["event_type"] == "clarify_original_edited":
                if entry.evidence.matches_transport(data):
                    entry.evidence.close("edited")
                return None
            return entry.evidence.original_reply(data)
    return None


def invalidate(adapter, chat_id, message_id):
    from tools import clarify_gateway as cm

    bot_id = _identity(getattr(adapter._bot, "id", None))
    with cm._lock:
        for entry in cm._entries.values():
            evidence = entry.evidence
            if (
                evidence
                and evidence.delivery
                and all(
                    evidence.delivery[key] == value
                    for key, value in (
                        ("bot_id", bot_id),
                        ("chat_id", str(chat_id)),
                        ("message_id", str(message_id)),
                    )
                )
            ):
                evidence.close("edited")
