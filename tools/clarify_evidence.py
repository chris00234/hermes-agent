"""Host-owned, bounded transport evidence for native clarification prompts."""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass

_PROCESS_GENERATION = uuid.uuid4().hex
_MAX_EVIDENCE_TTL_SECONDS = 3600
_MAX_EVENT_UTF8_BYTES = 24 * 1024
_MAX_RENDERED_CHARACTERS = 8192
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OriginalReply:
    evidence: ClarifyEvidence
    message_id: str
    update_id: str
    user_id: str
    text: str

    def matches(self, event, response: str) -> bool:
        return (
            response == self.text == event.text
            and str(event.message_id) == self.message_id
            and str(event.platform_update_id) == self.update_id
            and str(event.source.user_id) == self.user_id
            and str(event.source.chat_id) == self.evidence.delivery["chat_id"]
            and event.source.thread_id == self.evidence.delivery["thread_id"]
            and event.source.profile in (None, self.evidence.profile)
            and str(event.reply_to_message_id) == self.evidence.delivery["message_id"]
            and getattr(event.source.platform, "value", event.source.platform)
            == "telegram"
            and not event.media_urls
            and event.internal is False
        )


@dataclass(frozen=True)
class OriginalCallback:
    evidence: ClarifyEvidence
    callback_query_id: str
    update_id: str
    option_index: int

    def matches(self, response: str) -> bool:
        return self.evidence.options[self.option_index] == response


class ClarifyEvidence:
    def __init__(self, clarify_id, profile, question, options, timeout):
        self.clarify_id = clarify_id
        self.profile = profile
        self.question = question
        self.options = tuple(options)
        lifetime = (
            min(timeout, _MAX_EVIDENCE_TTL_SECONDS)
            if timeout > 0
            else _MAX_EVIDENCE_TTL_SECONDS
        )
        self.expires_at = time.time() + lifetime
        self.lock = threading.RLock()
        self.sequence = 0
        self.delivery = None
        self.closed = False
        self.schema_version = 1
        self.choice_buttons = ()
        self.callback_snapshot = None
        self.callback_ids = set()
        self.callback_updates = set()

    def _emit(self, kind, **fields):
        from hermes_cli.lifecycle import invoke_hook

        self.sequence += 1
        payload = {
            "schema_version": self.schema_version,
            "kind": kind,
            "clarify_id": self.clarify_id,
            "process_generation": _PROCESS_GENERATION,
            "sequence": self.sequence,
            "profile": self.profile,
            "platform": "telegram",
            "occurred_at": time.time(),
            **{key: self.delivery[key] for key in ("bot_id", "chat_id", "thread_id")},
            **fields,
        }
        if (
            len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            > _MAX_EVENT_UTF8_BYTES
        ):
            return False
        try:
            invoke_hook(
                "gateway_platform_event",
                platform="telegram",
                event_type="clarify_evidence",
                payload=copy.deepcopy(payload),
            )
        except Exception:
            logger.debug("Clarify evidence observer failed", exc_info=True)
            return False
        return True

    def delivered(
        self,
        *,
        question,
        options,
        rendered_text,
        choice_buttons=(),
        callback_snapshot=None,
        **transport,
    ):
        with self.lock:
            if self.closed or self.delivery or time.time() > self.expires_at:
                return
            if question != self.question or tuple(options or []) != self.options:
                return
            if len(rendered_text) > _MAX_RENDERED_CHARACTERS:
                return
            self.delivery = transport
            if choice_buttons and callback_snapshot is not None:
                self.schema_version = 2
                self.choice_buttons = copy.deepcopy(tuple(choice_buttons))
                self.callback_snapshot = copy.deepcopy(callback_snapshot)
            if not self._emit(
                "prompt_delivered",
                message_id=transport["message_id"],
                question=self.question,
                options=list(self.options),
                rendered_text=rendered_text,
                expires_at=self.expires_at,
                transport_ack=True,
                **(
                    {"choice_buttons": list(self.choice_buttons)}
                    if self.schema_version == 2
                    else {}
                ),
            ):
                self.closed = True

    def matches_transport(self, data):
        with self.lock:
            return bool(
                self.delivery
                and not self.closed
                and time.time() <= self.expires_at
                and all(
                    data.get(k) == self.delivery[k]
                    for k in ("bot_id", "chat_id", "thread_id")
                )
                and data.get("reply_to_message_id") == self.delivery["message_id"]
            )

    def original_reply(self, data):
        with self.lock:
            if not self.matches_transport(data):
                return None
            self._emit("original_reply", **data)
            return OriginalReply(
                self,
                data["message_id"],
                data["update_id"],
                data["user_id"],
                data["text"],
            )

    def original_callback(self, data, snapshot):
        with self.lock:
            if (
                self.schema_version != 2
                or snapshot != self.callback_snapshot
                or not self.matches_transport({
                    **data,
                    "reply_to_message_id": data["prompt_message_id"],
                })
                or data["callback_query_id"] in self.callback_ids
                or data["update_id"] in self.callback_updates
                or len(self.callback_ids) >= 128
            ):
                return None
            button = next(
                (
                    button
                    for button in self.choice_buttons
                    if button["callback_data"] == data["callback_data"]
                ),
                None,
            )
            if button is None:
                return None
            self.callback_ids.add(data["callback_query_id"])
            self.callback_updates.add(data["update_id"])
            if not self._emit(
                "original_callback", **data, option_index=button["option_index"]
            ):
                return None
            return OriginalCallback(
                self,
                data["callback_query_id"],
                data["update_id"],
                button["option_index"],
            )

    def close(self, reason, selected=None):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if not self.delivery:
                return
            if (
                not isinstance(selected, (OriginalReply, OriginalCallback))
                or selected.evidence is not self
                or reason != "resolved"
            ):
                selected = None
            reply = selected if isinstance(selected, OriginalReply) else None
            callback = selected if isinstance(selected, OriginalCallback) else None
            self._emit(
                "prompt_closed",
                reason=reason,
                selected_reply_message_id=reply.message_id if reply else None,
                selected_reply_update_id=reply.update_id if reply else None,
                **(
                    {
                        "selected_callback_query_id": callback.callback_query_id
                        if callback
                        else None,
                        "selected_callback_update_id": callback.update_id
                        if callback
                        else None,
                    }
                    if self.schema_version == 2
                    else {}
                ),
            )


def for_source(source, clarify_id, question, choices, multi_select):
    from hermes_cli.lifecycle import has_hook
    from hermes_cli.profiles import get_active_profile_name
    from tools.clarify_gateway import get_clarify_timeout
    from tools.clarify_tool import mark_recommended

    if (
        getattr(source.platform, "value", source.platform) != "telegram"
        or multi_select
        or not choices
        or not 2 <= len(choices) <= 10
        or not all(isinstance(c, str) for c in choices)
        or not isinstance(question, str)
        or len(question) > 2048
        or not has_hook("gateway_platform_event")
    ):
        return None
    # clarify_tool decorates choices[0] with "(Recommended)" before the gateway
    # callback runs, so the strings the surfaces deliver are the decorated ones.
    # Normalize (idempotently) before the length gate and store the decorated
    # list, so validation and evidence labels match delivery exactly.
    delivered = mark_recommended(list(choices))
    if not all(0 < len(c) <= 512 for c in delivered):
        return None
    return ClarifyEvidence(
        clarify_id,
        source.profile or get_active_profile_name(),
        question,
        delivered,
        get_clarify_timeout(),
    )


def selected_for_event(event, response):
    candidate = getattr(event, "_clarify_original_reply", None)
    if isinstance(candidate, OriginalReply) and candidate.matches(event, response):
        return candidate
    return None
