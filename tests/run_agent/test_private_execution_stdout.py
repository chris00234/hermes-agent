"""DE_DESIGN §1.2-B — gateway stdout is launchd-captured, so private runs
must never print tool previews or lifecycle lines.

Every launchd gateway plist sets StandardOutPath to the profile's
gateway.log, so process stdout IS a durable sink. The private constructor
fence (api_server._construct_private_agent) relies on two gates:
``quiet_mode=True`` silences the tool arg/result preview prints in
agent/tool_executor.py (_begin_tool_execution and both result-preview
sites), and ``suppress_status_output=True`` silences the
``_vprint(force=True)`` / ``_emit_status`` lifecycle lines that bypass
quiet_mode, plus the quiet-mode cute-message fallback. These tests pin each
gate with a printing control proving the assertions are not vacuous.
"""

from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent
from tests.run_agent.test_run_agent import (
    _make_tool_defs,
    _mock_assistant_msg,
    _mock_tool_call,
)

RESULT_MARKER = "STDOUT-TOOL-RESULT-5c1d"


def _real_agent(quiet_mode: bool) -> AIAgent:
    """Real AIAgent shaped like a /v1/runs private agent (platform, flags)."""
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=quiet_mode,
            skip_context_files=True,
            skip_memory=True,
            platform="api_server",
        )
    agent.client = MagicMock()
    if quiet_mode:
        agent.suppress_status_output = True  # _construct_private_agent post-flag
    return agent


def _run_tool_batch(agent, executor_method: str, tool_count: int) -> None:
    calls = [
        _mock_tool_call(name="web_search", arguments="{}", call_id=f"c{i}")
        for i in range(tool_count)
    ]
    msg = _mock_assistant_msg(content="", tool_calls=calls)
    with patch("run_agent.handle_function_call", return_value=RESULT_MARKER):
        getattr(agent, executor_method)(msg, [], "task-1")


class TestToolPreviewPrints:
    @pytest.mark.parametrize(
        ("executor_method", "tool_count"),
        [("_execute_tool_calls_sequential", 1), ("_execute_tool_calls_concurrent", 2)],
    )
    def test_private_shaped_agent_prints_nothing(
        self, capsys, executor_method, tool_count
    ):
        agent = _real_agent(quiet_mode=True)
        _run_tool_batch(agent, executor_method, tool_count)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert RESULT_MARKER not in captured.err

    @pytest.mark.parametrize(
        ("executor_method", "tool_count"),
        [("_execute_tool_calls_sequential", 1), ("_execute_tool_calls_concurrent", 2)],
    )
    def test_public_verbose_control_prints_previews(
        self, capsys, executor_method, tool_count
    ):
        agent = _real_agent(quiet_mode=False)
        agent.tool_progress_mode = "all"
        _run_tool_batch(agent, executor_method, tool_count)
        out = capsys.readouterr().out
        assert "web_search" in out
        assert RESULT_MARKER in out

    def test_begin_tool_execution_gated_on_quiet_mode(self, capsys):
        from agent.tool_executor import _begin_tool_execution

        for quiet in (True, False):
            agent = _real_agent(quiet_mode=quiet)
            agent.tool_progress_mode = "all"
            _begin_tool_execution(
                agent,
                function_name="web_search",
                function_args={"q": "SECRET-ARGS-91ab"},
                effective_task_id="task",
                tool_call_id="tc1",
                display_index=1,
            )
            out = capsys.readouterr().out
            if quiet:
                assert out == ""
            else:
                assert "web_search" in out and "SECRET-ARGS-91ab" in out


class TestLifecycleLines:
    def test_vprint_force_suppressed_for_private_agent(self, capsys):
        agent = _real_agent(quiet_mode=True)
        agent._vprint("lifecycle SECRET-LINE-77", force=True)
        assert capsys.readouterr().out == ""
        agent.suppress_status_output = False
        agent._vprint("lifecycle CONTROL-LINE-77", force=True)
        assert "CONTROL-LINE-77" in capsys.readouterr().out

    def test_emit_status_silent_but_callback_still_fires(self, capsys):
        agent = _real_agent(quiet_mode=True)
        events = []
        agent.status_callback = lambda kind, message: events.append((kind, message))
        agent._emit_status("provider fallback SECRET-STATUS-42")
        assert capsys.readouterr().out == ""
        assert events == [("lifecycle", "provider fallback SECRET-STATUS-42")]

    def test_quiet_cute_messages_never_fire_for_api_server_agents(self):
        agent = _real_agent(quiet_mode=True)
        assert agent._should_emit_quiet_tool_messages() is False
        # Even a CLI-shaped agent goes silent under the post-construction flag.
        agent.platform = "cli"
        agent.tool_progress_callback = None
        assert agent._should_emit_quiet_tool_messages() is False
        agent.suppress_status_output = False
        assert agent._should_emit_quiet_tool_messages() is True


def test_private_constructor_sets_both_stdout_gates():
    """api_server._construct_private_agent must keep quiet_mode(kwarg) and
    stamp suppress_status_output; losing either reopens launchd stdout."""
    from gateway.platforms.api_server import _construct_private_agent
    from gateway.private_execution_policy import build_policy

    policy = build_policy(
        run_id="run-1",
        profile="default",
        execution_attempt_id="att-1",
        lease_expires_at="2099-01-01T00:00:00Z",
        allowed_toolsets=("terminal",),
    )
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    agent = _construct_private_agent(FakeAgent, {"quiet_mode": True}, policy)
    assert captured["quiet_mode"] is True
    assert agent.suppress_status_output is True
