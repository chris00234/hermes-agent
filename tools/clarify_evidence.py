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

    def _emit(self, kind, **fields):
        from hermes_cli.lifecycle import invoke_hook

        self.sequence += 1
        payload = {
            "schema_version": 1,
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

    def delivered(self, *, question, options, rendered_text, **transport):
        with self.lock:
            if self.closed or self.delivery or time.time() > self.expires_at:
                return
            if question != self.question or tuple(options or []) != self.options:
                return
            if len(rendered_text) > _MAX_RENDERED_CHARACTERS:
                return
            self.delivery = transport
            if not self._emit(
                "prompt_delivered",
                message_id=transport["message_id"],
                question=self.question,
                options=list(self.options),
                rendered_text=rendered_text,
                expires_at=self.expires_at,
                transport_ack=True,
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

    def close(self, reason, selected=None):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if not self.delivery:
                return
            if not isinstance(selected, OriginalReply) or selected.evidence is not self:
                selected = None
            self._emit(
                "prompt_closed",
                reason=reason,
                selected_reply_message_id=selected.message_id if selected else None,
                selected_reply_update_id=selected.update_id if selected else None,
            )


def for_source(source, clarify_id, question, choices, multi_select):
    from hermes_cli.lifecycle import has_hook
    from hermes_cli.profiles import get_active_profile_name
    from tools.clarify_gateway import get_clarify_timeout

    if (
        getattr(source.platform, "value", source.platform) != "telegram"
        or multi_select
        or not choices
        or not 2 <= len(choices) <= 10
        or not all(isinstance(c, str) and 0 < len(c) <= 512 for c in choices)
        or not isinstance(question, str)
        or len(question) > 2048
        or not has_hook("gateway_platform_event")
    ):
        return None
    return ClarifyEvidence(
        clarify_id,
        source.profile or get_active_profile_name(),
        question,
        choices,
        get_clarify_timeout(),
    )


def selected_for_event(event, response):
    candidate = getattr(event, "_clarify_original_reply", None)
    if isinstance(candidate, OriginalReply) and candidate.matches(event, response):
        return candidate
    return None
