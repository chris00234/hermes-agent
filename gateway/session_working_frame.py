"""Failure-atomic working-frame ACK metadata for the gateway routing index."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from agent.memory_provider import sanitize_working_frame_reference

logger = logging.getLogger("gateway.session")


def working_frame_scope_or_none(data: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """An invalid persisted scope must never drop the enclosing route."""
    try:
        return sanitize_working_frame_reference(data.get(key))
    except Exception as exc:
        logger.warning("Dropping invalid %s on session entry load: %s", key, type(exc).__name__)
        return None


class SessionWorkingFrameMixin:
    """Persist current/continuation ACKs before publishing them in memory."""

    def get_working_frame_scope(self, session_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            for reference in (entry.working_frame_scope, entry.working_frame_pending_scope):
                clean = sanitize_working_frame_reference(reference)
                if (clean is not None and clean["session_id"] == entry.session_id
                        and clean["expires_at"] > time.time()):
                    return clean
            return None

    @staticmethod
    def _scope_ack_advances(previous: Dict[str, Any], reference: Dict[str, Any]) -> bool:
        if any(previous[key] != reference[key] for key in
               ("provider", "principal", "agent", "project", "session_id")):
            return False
        return reference["revision"] > previous["revision"] or reference == previous

    @staticmethod
    def _scope_new_lifetime(current: Dict[str, Any], reference: Dict[str, Any]) -> bool:
        """Only a cleared ACK or expiry permits a new lifetime's revision restart."""
        return (
            reference["logical_task_id"] != current["logical_task_id"]
            and (reference["cleared"] or current["expires_at"] <= time.time())
        )

    @staticmethod
    def _scope_ack_continues(source: Dict[str, Any], reference: Dict[str, Any]) -> bool:
        return (
            reference.get("source_session_id") == source["session_id"]
            and reference.get("source_revision") == source["revision"]
            and reference["session_id"] != source["session_id"]
            and all(reference.get(key) == source.get(key) for key in
                    ("provider", "principal", "agent", "project", "logical_task_id",
                     "lifecycle_epoch", "cleared"))
        )

    def _merge_scope_reference(self, entry, expected_session_id: str, clean: Dict[str, Any]):
        current = sanitize_working_frame_reference(entry.working_frame_scope)
        pending = sanitize_working_frame_reference(entry.working_frame_pending_scope)
        if pending and pending["session_id"] == expected_session_id:
            current, pending = pending, None
        if clean["session_id"] == expected_session_id:
            if current and current["session_id"] == expected_session_id:
                if (not self._scope_new_lifetime(current, clean)
                        and not self._scope_ack_advances(current, clean)):
                    return None
                if clean["cleared"] or clean["logical_task_id"] != current["logical_task_id"]:
                    pending = None
            return clean, pending
        if pending and clean["session_id"] == pending["session_id"]:
            return (current, clean) if self._scope_ack_advances(pending, clean) else None
        if any(self._scope_ack_continues(source, clean) for source in
               (pending, current) if source is not None):
            return current, clean
        return None

    def record_working_frame_scope(
        self, session_key: str, expected_session_id: str, reference: Dict[str, Any],
        *, active_profile: Optional[str] = None,
    ) -> bool:
        """Route/profile CAS: record a current ACK or proven compression continuation."""
        clean = sanitize_working_frame_reference(reference)
        if clean is None:
            raise ValueError("Missing working frame reference")
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or entry.session_id != expected_session_id:
                return False
            profile = entry.origin.profile if entry.origin else None
            profile = profile or self._named_profile_for_key(session_key) or active_profile
            if profile and clean["agent"] != profile:
                return False
            merged = self._merge_scope_reference(entry, expected_session_id, clean)
            if merged is None:
                return False
            self._persist_working_frame_scope(session_key, entry, *merged)
            return True

    def _persist_working_frame_scope(self, session_key, entry, current, pending) -> None:
        candidate = entry.to_dict()
        candidate["working_frame_scope"] = current
        candidate["working_frame_pending_scope"] = pending
        self._save_entry(session_key, entry_data=candidate, lock_held=True)
        entry.working_frame_scope = current
        entry.working_frame_pending_scope = pending

    def expire_working_frame_scope(self, session_key: str, expected_session_id: str) -> bool:
        """CAS-drop only a scope whose configured provider can never retire it."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or entry.session_id != expected_session_id:
                return False
            self._persist_working_frame_scope(session_key, entry, None, None)
            return True
