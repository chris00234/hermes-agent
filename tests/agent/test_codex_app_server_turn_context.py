"""The context the turn prologue composes reaches the codex app-server turn.

A codex_app_server turn is ONE ``turn/start`` input. Memory-provider prefetch, ``pre_llm_call``
hook context and gateway must-deliver notes were computed every turn and then dropped on this
lane (only the bare user text was sent), and the skipped api_content stamp hid the loss from the
transcript. The fake session records the input and builds codex's echo of it with the real
projector, so the echo strip and the persisted user row are exercised as in production.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from agent.transports.codex_event_projector import CodexEventProjector
from hermes_state import SessionDB

LITERAL = "그리고 내 브렌드 로고 갖고 있지?"
BRAIN_ROW = "BRAIN-ROW: Chris final personal brand logo (approved 2026-10-14) at ~/brand/chris-cho-logo.svg"
SID = "sess-codex-ctx"


@pytest.fixture
def codex_inputs(monkeypatch):
    """Every ``turn/start`` input the fake codex session receives, in order."""
    inputs: list = []

    def fake_run_turn(self, user_input, **kwargs):
        inputs.append(user_input)
        item = {"type": "userMessage", "id": f"user-{len(inputs)}",
                "content": [{"type": "text", "text": user_input}]}
        echo = CodexEventProjector().project({"method": "item/completed", "params": {"item": item}}).messages[0]
        return TurnResult(
            final_text="done", projected_messages=[echo, {"role": "assistant", "content": "done"}],
            submitted_user_text=user_input, turn_id=f"turn-{len(inputs)}", thread_id="thread-ctx-1",
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-ctx-1")
    return inputs


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session(session_id=SID, source="cli")
    yield session_db
    session_db.close()


def _codex_agent(db, prefetch=lambda query, **_kw: BRAIN_ROW):
    agent = run_agent.AIAgent(
        api_key="stub", base_url="https://stub.invalid", provider="openai", api_mode="codex_app_server",
        quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id=SID,
    )
    agent._session_db_created = True
    agent._memory_manager = MagicMock()
    agent._memory_manager.build_system_prompt.return_value = ""
    agent._memory_manager.describe_recall.return_value = ""
    agent._memory_manager.prefetch_all.side_effect = prefetch
    return agent


def _run(agent, text, *, hook_context):
    def invoke_hook(name, **_kwargs):
        return [{"context": hook_context}] if name == "pre_llm_call" and hook_context else []

    with patch("hermes_cli.plugins.invoke_hook", side_effect=invoke_hook), \
            patch.object(agent, "_spawn_background_review", return_value=None):
        return agent.run_conversation(text)


def _user_rows(db) -> list[tuple]:
    with sqlite3.connect(db.db_path) as conn:
        return conn.execute(
            "SELECT content, api_content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id",
            (SID,),
        ).fetchall()


def test_brain_prefetch_and_hook_context_reach_the_codex_turn(db, codex_inputs):
    agent = _codex_agent(db)
    _run(agent, LITERAL, hook_context="LESSON-CTX")

    [sent] = codex_inputs
    assert sent.startswith(LITERAL)
    assert "<memory-context>" in sent and BRAIN_ROW in sent and "LESSON-CTX" in sent
    # One user row: the clean words, with the exact bytes codex received as its sidecar. The
    # echoed userMessage must not land as a second (context-bearing) user row.
    assert _user_rows(db) == [(LITERAL, sent)]
    assert not any("<memory-context>" in (content or "") for content, _ in _user_rows(db))


def test_context_rides_only_its_own_turn(db, codex_inputs):
    rows = {"logo": "ROW-ONE: logo file", "font": "ROW-TWO: brand font"}
    agent = _codex_agent(db, prefetch=lambda query, **_kw: rows["logo" if "logo" in query else "font"])
    _run(agent, "where is my logo file?", hook_context="")
    first_row = _user_rows(db)[0]
    _run(agent, "which brand font do I use?", hook_context="")

    first, second = codex_inputs
    assert "ROW-ONE" in first and "ROW-TWO" in second
    assert "ROW-ONE" not in second
    persisted = _user_rows(db)
    assert persisted[0] == first_row == ("where is my logo file?", first)
    assert persisted[1] == ("which brand font do I use?", second)


def test_gateway_note_reaches_the_codex_turn(db, codex_inputs):
    agent = _codex_agent(db)
    agent._gateway_turn_context_notes = "GW-NOTE"
    _run(agent, LITERAL, hook_context="")

    [sent] = codex_inputs
    assert "GW-NOTE" in sent
    assert agent._gateway_turn_context_notes == ""
    assert _user_rows(db) == [(LITERAL, sent)]


def test_trivial_turn_without_context_is_sent_bare(db, codex_inputs):
    agent = _codex_agent(db)
    _run(agent, "ok", hook_context="")

    assert codex_inputs == ["ok"]
    assert _user_rows(db) == [("ok", None)]
