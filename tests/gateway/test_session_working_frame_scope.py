"""SessionStore working-frame scope CAS (durable /reset //undo prerequisite).

Pins ``get_working_frame_scope`` / ``record_working_frame_scope``: route CAS,
profile fencing, revision monotonicity, cleared-ACK pending drop, compression
continuations, and metadata-only serialization.
"""
from datetime import datetime

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


def _ref(**overrides):
    reference = {
        "provider": "frame",
        "principal": "chris",
        "agent": "jenna",
        "project": "proj",
        "session_id": "sess-1",
        "logical_task_id": "task-1",
        "revision": 3,
        "lifecycle_epoch": 1,
        "expires_at": 4102444800.0,
        "cleared": False,
    }
    reference.update(overrides)
    return reference


def _make_store(tmp_path, *, session_id="sess-1", profile="jenna"):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db = None
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="c1",
        user_id="u1",
        chat_type="dm",
        profile=profile,
    )
    entry = SessionEntry(
        session_key="telegram:c1",
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
    )
    store._entries["telegram:c1"] = entry
    store._loaded = True
    return store, entry


def test_record_then_get_round_trip(tmp_path):
    store, _entry = _make_store(tmp_path)
    assert store.get_working_frame_scope("telegram:c1") is None

    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref()) is True
    assert store.get_working_frame_scope("telegram:c1") == _ref()
    assert store.get_working_frame_scope("missing-key") is None


def test_wrong_expected_session_id_rejected(tmp_path):
    store, _entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-other", _ref()) is False
    assert store.get_working_frame_scope("telegram:c1") is None


def test_agent_profile_mismatch_rejected(tmp_path):
    store, _entry = _make_store(tmp_path, profile="liz")
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref()) is False


def test_revision_regress_rejected(tmp_path):
    store, _entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref(revision=3))
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref(revision=2)) is False
    assert store.get_working_frame_scope("telegram:c1")["revision"] == 3


def test_cleared_ack_replaces_current_and_drops_pending(tmp_path):
    store, entry = _make_store(tmp_path)
    current = _ref(revision=3)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", current)

    continuation = _ref(
        session_id="sess-2",
        source_session_id="sess-1",
        source_revision=3,
        revision=1,
    )
    assert store.record_working_frame_scope("telegram:c1", "sess-1", continuation)
    assert entry.working_frame_pending_scope == continuation
    # Current-session scope still wins the route lookup.
    assert store.get_working_frame_scope("telegram:c1") == current

    ack = _ref(revision=4, cleared=True)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", ack) is True
    assert entry.working_frame_scope == ack
    assert entry.working_frame_pending_scope is None
    assert store.get_working_frame_scope("telegram:c1") == ack


def test_unrelated_reference_rejected(tmp_path):
    store, _entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref())
    stranger = _ref(session_id="sess-9", revision=7)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", stranger) is False


def test_serialization_round_trip_keeps_sanitized_metadata(tmp_path):
    store, entry = _make_store(tmp_path)
    noisy = _ref(revision=4, cleared=True)
    noisy["frame_content"] = "SECRET FRAME BODY"
    assert store.record_working_frame_scope("telegram:c1", "sess-1", noisy) is True

    data = entry.to_dict()
    assert data["working_frame_scope"] == _ref(revision=4, cleared=True)
    assert "frame_content" not in data["working_frame_scope"]

    restored = SessionEntry.from_dict(data)
    assert restored.working_frame_scope == _ref(revision=4, cleared=True)
    assert restored.working_frame_pending_scope is None


def test_get_ignores_scope_from_rotated_session(tmp_path):
    store, entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref())
    entry.session_id = "sess-2"
    assert store.get_working_frame_scope("telegram:c1") is None


def test_expired_scope_returns_none(tmp_path):
    import time as _time

    store, _entry = _make_store(tmp_path)
    assert store.record_working_frame_scope(
        "telegram:c1", "sess-1", _ref(expires_at=_time.time() - 60)
    ) is True
    assert store.get_working_frame_scope("telegram:c1") is None


def test_expired_current_accepts_fresh_lifetime(tmp_path):
    import time as _time

    store, _entry = _make_store(tmp_path)
    stale = _ref(revision=5, expires_at=_time.time() - 60)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", stale)

    fresh = _ref(revision=1, logical_task_id="task-2")
    assert store.record_working_frame_scope("telegram:c1", "sess-1", fresh) is True
    assert store.get_working_frame_scope("telegram:c1") == fresh


def test_unexpired_current_rejects_new_lifetime_regress(tmp_path):
    store, _entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref(revision=3))
    stranger = _ref(revision=1, logical_task_id="task-2")
    assert store.record_working_frame_scope("telegram:c1", "sess-1", stranger) is False
    assert store.get_working_frame_scope("telegram:c1")["revision"] == 3


def test_cleared_ack_may_restart_revision_for_new_lifetime(tmp_path):
    store, _entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref(revision=3))
    ack = _ref(revision=1, logical_task_id="task-2", cleared=True)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", ack) is True
    assert store.get_working_frame_scope("telegram:c1") == ack


def test_active_profile_keyword_fences_agent_when_origin_profile_missing(tmp_path):
    store, _entry = _make_store(tmp_path, profile=None)
    foreign = _ref(agent="liz")
    assert store.record_working_frame_scope(
        "telegram:c1", "sess-1", foreign, active_profile="jenna"
    ) is False
    assert store.record_working_frame_scope(
        "telegram:c1", "sess-1", _ref(), active_profile="jenna"
    ) is True


def test_from_dict_tolerates_corrupt_scope(tmp_path, caplog):
    import logging

    store, entry = _make_store(tmp_path)
    assert store.record_working_frame_scope("telegram:c1", "sess-1", _ref())
    data = entry.to_dict()
    data["working_frame_scope"] = {"provider": "frame", "revision": "NaN"}
    data["working_frame_pending_scope"] = ["not", "a", "dict"]

    with caplog.at_level(logging.WARNING, logger="gateway.session"):
        restored = SessionEntry.from_dict(data)

    assert restored.session_id == "sess-1"
    assert restored.working_frame_scope is None
    assert restored.working_frame_pending_scope is None
    assert any("ValueError" in record.getMessage() for record in caplog.records)
    assert all("frame" not in record.getMessage() or "SECRET" not in record.getMessage()
               for record in caplog.records)


class _StubTranscriptDB:
    """Durable-transcript stub for the /undo dry-run probe."""

    def __init__(self, messages):
        self._messages = messages

    def get_messages_as_conversation(self, session_id, **kwargs):
        return [dict(m) for m in self._messages]


def test_rewindable_probe_returns_false_when_classifier_raises(tmp_path, monkeypatch):
    """A malformed durable row must make the probe report nothing to undo,
    never crash the /undo handler."""
    import agent.context_compressor as cc

    store, _ = _make_store(tmp_path)
    store._db = _StubTranscriptDB([{"role": "user", "content": "hi", "_row_id": 1}])

    def _boom(message):
        raise ValueError("malformed composite payload")

    monkeypatch.setattr(cc, "user_originated_turn_view", _boom)
    monkeypatch.setattr(cc, "split_user_originated_turn", _boom)

    store._db_for_session_id = lambda session_id: store._db
    assert store.has_rewindable_user_turn("sess-1") is False


def test_rewindable_probe_requires_persisted_row_identity(tmp_path):
    """rewind_session refuses a target without an integer ``_row_id``; the
    probe must agree instead of using a weaker predicate."""
    store, _ = _make_store(tmp_path)
    store._db = _StubTranscriptDB([{"role": "user", "content": "hi"}])

    store._db_for_session_id = lambda session_id: store._db
    assert store.has_rewindable_user_turn("sess-1") is False


def test_rewindable_probe_true_for_rewindable_turn(tmp_path):
    store, _ = _make_store(tmp_path)
    store._db = _StubTranscriptDB([{"role": "user", "content": "hi", "_row_id": 7}])

    store._db_for_session_id = lambda session_id: store._db
    assert store.has_rewindable_user_turn("sess-1") is True
