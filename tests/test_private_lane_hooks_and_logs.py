"""Private-lane fences: lifecycle hooks, request dumps, shared logs.

DE_DESIGN §6.8/§6.9/§6.10 — a run carrying a ``PrivateExecutionPolicy``
must never reach plugin hooks, request-debug dumps, or the shared log
files, while concurrent public runs in the same process stay untouched.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import hermes_cli.plugins as plugins_mod
import hermes_logging
from agent.agent_runtime_helpers import dump_api_request_debug
from agent.turn_finalizer import finalize_turn
from gateway.private_execution_policy import (
    bind_private_policy,
    build_policy,
    reset_private_policy,
)
from run_agent import AIAgent
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent


def _make_policy():
    return build_policy(
        run_id="run-1",
        profile="design",
        execution_attempt_id="attempt-1",
        lease_expires_at="2999-01-01T00:00:00+00:00",
        allowed_toolsets=["web_search"],
    )


@contextmanager
def _bound_policy():
    token = bind_private_policy(_make_policy())
    try:
        yield
    finally:
        reset_private_policy(token)


def _make_enabled_plugin(hermes_home: Path, name: str, register_body: str) -> None:
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n", encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {} if cfg_path.exists() else {}
    cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _finalize(fake_agent):
    return finalize_turn(
        fake_agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Done."},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )


class TestFinalizerHookGates:
    """§6.8 — finalize_turn never dispatches hooks for a private agent."""

    def _run(self, fake_agent, monkeypatch, hook_calls, engine_calls):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kw: (hook_calls.append(name), [])[1],
        )
        monkeypatch.setattr(
            "agent.conversation_loop._notify_context_engine_turn_complete",
            lambda *a, **kw: engine_calls.append(True),
        )
        return _finalize(fake_agent)

    def test_private_agent_skips_hooks_and_context_engine(self, monkeypatch):
        hook_calls, engine_calls = [], []
        fake = FakeAgent()
        fake._private_execution_policy = _make_policy()
        self._run(fake, monkeypatch, hook_calls, engine_calls)
        assert hook_calls == []
        assert engine_calls == []

    def test_public_agent_dispatches_hooks_and_context_engine(self, monkeypatch):
        hook_calls, engine_calls = [], []
        self._run(FakeAgent(), monkeypatch, hook_calls, engine_calls)
        assert hook_calls == [
            "transform_llm_output", "post_llm_call", "on_session_end",
        ]
        assert engine_calls == [True]


class TestPluginBackstop:
    """§6.8 — invoke_hook backstop with finalizer gates bypassed."""

    def _manager_with_transform(self, monkeypatch):
        home = Path(os.environ["HERMES_HOME"])
        _make_enabled_plugin(
            home, "recorder",
            'ctx.register_hook("transform_llm_output", lambda **kw: "transformed")',
        )
        mgr = plugins_mod.PluginManager()
        mgr.discover_and_load()
        return mgr

    def test_manager_invoke_hook_suppressed_when_policy_bound(self, monkeypatch):
        mgr = self._manager_with_transform(monkeypatch)
        kwargs = dict(response_text="x", session_id="s", model="m", platform="cli")
        assert mgr.invoke_hook("transform_llm_output", **kwargs) == ["transformed"]
        with _bound_policy():
            assert mgr.invoke_hook("transform_llm_output", **kwargs) == []

    def test_module_invoke_hook_skips_discovery_when_policy_bound(self, monkeypatch):
        mgr = MagicMock()
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: mgr)
        monkeypatch.setattr(plugins_mod, "_join_background_discovery", lambda: None)
        with _bound_policy():
            assert plugins_mod.invoke_hook("post_llm_call") == []
        mgr.discover_and_load.assert_not_called()
        mgr.invoke_hook.assert_not_called()
        plugins_mod.invoke_hook("post_llm_call")
        mgr.invoke_hook.assert_called_once()

    def test_finalizer_gate_bypassed_backstop_still_suppresses(self, monkeypatch):
        # The agent carries NO policy attribute (finalizer gate sees a public
        # run) but the ContextVar is bound — the module-level invoke_hook
        # backstop must still yield zero plugin dispatches and no discovery.
        mgr = MagicMock()
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: mgr)
        monkeypatch.setattr(plugins_mod, "_join_background_discovery", lambda: None)
        monkeypatch.setattr(
            "agent.conversation_loop._notify_context_engine_turn_complete",
            lambda *a, **kw: None,
        )
        with _bound_policy():
            _finalize(FakeAgent())
        mgr.invoke_hook.assert_not_called()
        mgr.discover_and_load.assert_not_called()

    def test_backstop_fails_closed_on_accessor_error(self, monkeypatch):
        mgr = self._manager_with_transform(monkeypatch)

        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "gateway.private_execution_policy.current_private_policy", _boom
        )
        assert mgr.invoke_hook(
            "transform_llm_output",
            response_text="x", session_id="s", model="m", platform="cli",
        ) == []


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def real_agent(tmp_path):
    with (
        patch("run_agent.get_tool_definitions",
              return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a.client.api_key = "test-key-1234567890"
        a.logs_dir = tmp_path
        return a


class TestDumpRefusal:
    """§6.9 — request-debug dumps refuse policy-stamped agents entirely."""

    _KWARGS = {
        "model": "m",
        "messages": [{"role": "user", "content": "PRIV-MARK-7731"}],
    }

    def test_private_agent_dump_refused_no_file_no_stdout(
        self, real_agent, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("HERMES_DUMP_REQUEST_STDOUT", "1")
        real_agent._private_execution_policy = _make_policy()
        result = dump_api_request_debug(
            real_agent, dict(self._KWARGS), reason="provider_4xx"
        )
        assert result is None
        assert list(tmp_path.glob("request_dump_*.json")) == []
        assert "PRIV-MARK-7731" not in capsys.readouterr().out

    def test_public_agent_dump_written(self, real_agent, tmp_path):
        result = dump_api_request_debug(
            real_agent, dict(self._KWARGS), reason="provider_4xx"
        )
        assert result is not None and result.exists()
        assert "provider_4xx" in result.read_text(encoding="utf-8")


class TestPrivateLogFilter:
    """§6.10 — marked records never reach shared log files; fail closed."""

    @pytest.fixture(autouse=True)
    def _reset_logging_state(self):
        hermes_logging._logging_initialized = False
        hermes_logging._reset_queued_handlers()
        root = logging.getLogger()
        prev_root_level = root.level
        root.setLevel(logging.NOTSET)
        pre_existing = list(root.handlers)
        hermes_logging._install_session_record_factory()
        yield
        hermes_logging._reset_queued_handlers()
        for h in list(root.handlers):
            if h not in pre_existing:
                root.removeHandler(h)
                h.close()
        root.setLevel(prev_root_level)
        hermes_logging._logging_initialized = False
        hermes_logging.clear_session_context()

    @pytest.fixture
    def hermes_home(self):
        return Path(os.environ["HERMES_HOME"])

    def test_private_records_dropped_public_records_present(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        lg = logging.getLogger("private_lane.log_test")
        with _bound_policy():
            lg.info("PRIVATE-MARK-abc123")
            lg.error("PRIVATE-ERR-abc123")
        lg.info("PUBLIC-MARK-def456")
        lg.error("PUBLIC-ERR-def456")
        hermes_logging.flush_log_queue()
        agent_log = (hermes_home / "logs" / "agent.log").read_text(encoding="utf-8")
        errors_log = (hermes_home / "logs" / "errors.log").read_text(encoding="utf-8")
        assert "PRIVATE-MARK-abc123" not in agent_log
        assert "PRIVATE-ERR-abc123" not in agent_log
        assert "PRIVATE-ERR-abc123" not in errors_log
        assert "PUBLIC-MARK-def456" in agent_log
        assert "PUBLIC-ERR-def456" in errors_log

    def test_factory_stamps_marker_before_queue_hop(self):
        factory = logging.getLogRecordFactory()
        with _bound_policy():
            rec = factory("n", logging.INFO, __file__, 1, "m", (), None)
        assert rec.hermes_private is True
        rec = factory("n", logging.INFO, __file__, 1, "m", (), None)
        assert rec.hermes_private is False

    def test_factory_fail_closed_marks_record_private(self, monkeypatch):
        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "gateway.private_execution_policy.current_private_policy", _boom
        )
        factory = logging.getLogRecordFactory()
        rec = factory("n", logging.INFO, __file__, 1, "m", (), None)
        assert rec.hermes_private is True

    def test_filter_exception_drops_record(self, hermes_home, monkeypatch):
        hermes_logging.setup_logging(hermes_home=hermes_home)

        def _boom(record):
            raise RuntimeError("boom")

        monkeypatch.setattr(hermes_logging, "_record_is_private", _boom)
        logging.getLogger("private_lane.log_test").info("DROP-ON-ERROR-xyz")
        hermes_logging.flush_log_queue()
        agent_log = (hermes_home / "logs" / "agent.log").read_text(encoding="utf-8")
        assert "DROP-ON-ERROR-xyz" not in agent_log

    def test_profile_router_passes_allow_marked_fixed_template(
        self, hermes_home, tmp_path
    ):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        assert hermes_logging.enable_profile_log_routing(
            [hermes_home, profile_home]
        ) is True

        lg = logging.getLogger("gateway.platforms.api_server")
        with _bound_policy():
            lg.error(
                "[api_server] private run %s failed: run_failed",
                "run_routed1",
                extra={"hermes_private_allow": True},
            )
            lg.error("PRIVATE-ROUTED-DROP %s", "run_routed1")
        hermes_logging.flush_log_queue()
        agent_log = (hermes_home / "logs" / "agent.log").read_text(
            encoding="utf-8"
        )
        assert "run run_routed1 failed: run_failed" in agent_log
        assert "PRIVATE-ROUTED-DROP" not in agent_log
