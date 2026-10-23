"""Private fences on the extracted finalizer, text response and SQLite paths."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import private_execution_policy as policy_module


@contextmanager
def policy_state(agent, mode, monkeypatch):
    policy = policy_module.build_policy(
        run_id="lane-a", profile="test", execution_attempt_id="attempt",
        lease_expires_at="2099-01-01T00:00:00+00:00", allowed_toolsets=("terminal",),
    )
    if mode in {"agent", "both"}:
        agent._private_execution_policy = policy
    if mode == "error":
        monkeypatch.setattr(policy_module, "current_private_policy", Mock(side_effect=RuntimeError("unreadable")))
    token = policy_module.bind_private_policy(policy) if mode in {"context", "both"} else None
    try:
        yield
    finally:
        if token is not None:
            policy_module.reset_private_policy(token)


@pytest.mark.parametrize("mode", ["public", "agent", "context", "both", "error"])
def test_finalizer_preserves_response_cleanup_but_suppresses_sinks(mode, monkeypatch):
    from agent import conversation_loop, turn_finalizer
    from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent

    agent = FakeAgent()
    sinks = {
        name: Mock() for name in (
            "_save_trajectory", "_persist_session", "_sync_external_memory_for_turn",
            "_spawn_background_review", "_cleanup_task_resources",
        )
    }
    for name, sink in sinks.items():
        setattr(agent, name, sink)
    hooks = Mock(return_value=[])
    observer = Mock()
    compact = Mock()
    monkeypatch.setattr(turn_finalizer, "_invoke_hook_safely", hooks)
    monkeypatch.setattr(turn_finalizer, "_micro_compact_after_turn", compact)
    monkeypatch.setattr(conversation_loop, "_notify_context_engine_turn_complete", observer)
    messages = [{"role": "user", "content": "private input"}]
    with policy_state(agent, mode, monkeypatch):
        result = turn_finalizer.finalize_turn(
            agent, final_response="Completed.", api_call_count=1, interrupted=False,
            failed=False, messages=messages, conversation_history=[], effective_task_id="task",
            turn_id="turn", user_message="private input", original_user_message="private input",
            _should_review_memory=True, _turn_exit_reason="text_response(stop)",
        )
    assert result["final_response"] == "Completed."
    assert result["completed"] is True
    assert result["messages"][-1]["content"] == "Completed."
    assert agent._session_messages is messages
    sinks.pop("_cleanup_task_resources").assert_called_once_with("task")
    if mode == "public":
        for sink in sinks.values():
            sink.assert_called_once()
        assert [call.args[0] for call in hooks.call_args_list] == [
            "transform_llm_output", "post_llm_call", "on_session_end",
        ]
        observer.assert_called_once()
        compact.assert_called_once()
    else:
        for sink in [*sinks.values(), hooks, observer, compact]:
            sink.assert_not_called()


@pytest.mark.parametrize("mode", ["public", "agent", "context", "both", "error"])
def test_actual_sqlite_flush_and_trajectory_fences(mode, monkeypatch, tmp_path):
    from agent import session_persistence
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "lane-a.db")
    db.create_session("lane-a", source="cli")
    agent = session_persistence.SessionPersistenceMixin()
    agent._session_db = db
    agent.session_id = "lane-a"
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent.save_trajectories = True
    agent.model = "test"
    agent._convert_to_trajectory_format = Mock(return_value={"test": True})
    saved = Mock()
    monkeypatch.setattr(session_persistence, "_save_trajectory_to_file", saved)
    messages = [{"role": "user", "content": "private transcript"}]
    try:
        with policy_state(agent, mode, monkeypatch):
            for _ in range(2):
                agent._flush_messages_to_session_db(messages)
            agent._save_trajectory(messages, "private transcript", True)
            if mode != "public":
                agent._persist_session(messages)
                assert agent._session_messages is messages
        rows = db.get_messages_as_conversation("lane-a")
        if mode == "public":
            assert len(rows) == 1
            assert rows[0]["content"] == "private transcript"
            saved.assert_called_once()
        else:
            assert rows == []
            assert "_db_persisted" not in messages[0]
            saved.assert_not_called()
    finally:
        db.close()


@pytest.mark.parametrize("mode", ["public", "agent", "context", "both", "error"])
def test_text_response_flush_is_fenced_at_extracted_path(mode, monkeypatch):
    from agent import turn_final_response, turn_finalizer

    agent = SimpleNamespace(
        _has_content_after_think_block=lambda text: bool(text),
        _emit_pending_fallback_notice=Mock(), _clear_status_buffer=Mock(),
        _stall_guards=False, valid_tool_names=[], _strip_think_blocks=lambda text: text,
        _build_assistant_message=lambda msg, reason: {"role": "assistant", "content": msg.content},
        _flush_messages_to_session_db=Mock(), quiet_mode=True, session_id="lane-a", model="test",
    )
    output_hooks = Mock(return_value=[])
    monkeypatch.setattr(turn_finalizer, "_invoke_hook_safely", output_hooks)
    monkeypatch.setattr("agent.agent_runtime_helpers.intent_ack_continuation_mode", lambda agent: "off")
    monkeypatch.setattr(turn_final_response, "apply_stop_gates", lambda *a, **k: SimpleNamespace(
        pending_verification_response=None, pending_verification_response_previewed=False, continue_turn=False,
    ))
    messages = [{"role": "user", "content": "input"}]
    with policy_state(agent, mode, monkeypatch):
        verdict = turn_final_response.finish_text_response(
            agent, assistant_message=SimpleNamespace(content="Answer.", tool_calls=[]),
            response=None, finish_reason="stop", messages=messages, api_messages=messages,
            conversation_history=[], api_call_count=1, user_message="input", active_system_prompt="stable",
            final_response=None, _turn_exit_reason="unknown", _preflight_compression_blocked=False,
            codex_ack_continuations=0, truncated_response_parts=[], length_continue_retries=0,
            _pending_verification_response=None, _pending_verification_response_previewed=False,
            effective_task_id="lane-a-task",
        )
    assert verdict.action == "break"
    assert verdict.final_response == "Answer."
    assert messages[-1]["content"] == "Answer."
    assert verdict.active_system_prompt == "stable"
    if mode == "public":
        agent._flush_messages_to_session_db.assert_called_once()
        assert [call.args[0] for call in output_hooks.call_args_list] == ["transform_llm_output"]
    else:
        agent._flush_messages_to_session_db.assert_not_called()
        output_hooks.assert_not_called()


@pytest.mark.parametrize("mode", ["public", "agent", "context", "both", "error"])
def test_memory_end_and_sync_sinks_on_actual_agent(mode, monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = Mock()
    agent.context_compressor = Mock()
    agent.session_id = "lane-a"
    messages = [{"role": "user", "content": "remember the project constraint"}]
    with policy_state(agent, mode, monkeypatch):
        agent._sync_external_memory_for_turn(
            original_user_message=messages[0]["content"], final_response="Understood.",
            interrupted=False, messages=messages,
        )
        agent.commit_memory_session(messages)
        agent.shutdown_memory_provider(messages)
        agent.shutdown_memory_provider(messages)
    if mode == "public":
        agent._memory_manager.sync_all.assert_called_once()
        agent._memory_manager.queue_prefetch_all.assert_called_once()
        assert agent._memory_manager.on_session_end.call_count == 2
        agent._memory_manager.shutdown_all.assert_called_once()
        assert agent.context_compressor.on_session_end.call_count == 2
    else:
        assert agent._memory_manager.mock_calls == []
        assert agent.context_compressor.mock_calls == []


@pytest.mark.parametrize("mode", ["public", "agent", "context", "both", "error"])
def test_memory_init_cannot_reenable_store_for_private_toolset(mode, monkeypatch):
    from agent.agent_init import _init_memory
    from tools import memory_tool

    store = Mock()
    monkeypatch.setattr(memory_tool, "MemoryStore", store)
    monkeypatch.setattr(memory_tool, "get_builtin_memory_config", lambda config: {})
    monkeypatch.setattr(memory_tool, "get_builtin_memory_store_flags", lambda config: (True, True))
    monkeypatch.setattr("agent.memory_manager.inject_memory_provider_tools", Mock())
    agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=[])
    with policy_state(agent, mode, monkeypatch):
        _init_memory(agent, {}, skip_memory=True, platform="api")
    assert agent._memory_manager is None
    if mode == "public":
        store.return_value.load_from_disk.assert_called_once()
        assert agent._memory_enabled is True
    else:
        store.assert_not_called()
        assert agent._memory_store is None
        assert agent._memory_enabled is False
        assert agent._user_profile_enabled is False
