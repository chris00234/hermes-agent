"""DE_DESIGN §6.4 unique-marker leak sweep + §7.1 response-store verification.

A private /v1/runs agent drives every durable sink a real run touches with
distinct input/output/tool-result/error markers; all must stay out of
state.db, the sessions dir (JSON snapshots + request dumps),
runs_idempotency.db status_json, the shared logs after flush, and
cache/spillover — while a concurrent PUBLIC run's markers land in those
same sinks (no global muting). §7.1: no /v1/runs path writes
response_store.db (its only reference is a private-fenced read)."""

import asyncio
import io
import json
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import hermes_logging
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.gateway.test_api_server import _patch_create_agent_runtime
from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _use_idempotency_db,
)
from tests.gateway.test_private_lane_runs import GRANT_HEADER, SECRET, _grant
from tools.budget_config import BudgetConfig

_SPILL_CONFIG = BudgetConfig(tool_overrides={"leak_probe": 100}, preview_size=40)


class SinkExercisingAgent(AIAgent):
    """Real AIAgent whose run drives every fenced sink with markers."""

    def __init__(self, **kwargs):
        kwargs.setdefault("skip_memory", True)
        kwargs.setdefault("skip_context_files", True)
        super().__init__(**kwargs)

    def run_conversation(self, user_message=None, conversation_history=None, task_id=None):
        from agent.agent_runtime_helpers import dump_api_request_debug
        from agent.tool_executor import _persist_or_truncate_tool_result
        from tools.tool_result_storage import enforce_turn_budget

        base = str(user_message)
        out = base.replace("INPUT", "OUTPUT")
        tool_marker = base.replace("INPUT", "TOOLRESULT")
        transcript = [
            {"role": "user", "content": base},
            {"role": "assistant", "content": out},
        ]
        self._ensure_db_session()
        self._flush_messages_to_session_db(transcript, [])
        self._save_trajectory(transcript, base, completed=True)
        log = logging.getLogger("leak_sweep.agent")
        log.info("tool result %s", tool_marker)
        log.error("run error %s", base.replace("INPUT", "ERRLOG"))
        self.client.api_key = "sk-leak-sweep-dump"
        dump_api_request_debug(
            self,
            {"messages": [{"role": "user", "content": base}]},
            reason="leak-sweep",
        )
        _persist_or_truncate_tool_result(
            self,
            content=tool_marker + ("z" * 500),
            tool_name="leak_probe",
            tool_use_id=f"tc-{self.session_id}",
            effective_task_id=task_id or "leak-task",
            config=_SPILL_CONFIG,
        )
        enforce_turn_budget(
            [
                {
                    "tool_call_id": f"tb-{self.session_id}",
                    "content": base.replace("INPUT", "TURNBUDGET") + ("y" * 500),
                }
            ],
            env=None,
            config=BudgetConfig(turn_budget=100, preview_size=40),
        )
        if "RAISE" in base:
            raise RuntimeError(base.replace("INPUT", "EXC"))
        return {"final_response": out}


@pytest.fixture(autouse=True)
def grant_secret(monkeypatch):
    monkeypatch.setenv("PRIVATE_EXEC_GRANT_KEY", SECRET)


@pytest.fixture(autouse=True)
def _logging_state():
    hermes_logging._logging_initialized = False
    hermes_logging._reset_queued_handlers()
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.NOTSET)
    pre_existing = list(root.handlers)
    hermes_logging._install_session_record_factory()
    yield
    hermes_logging._reset_queued_handlers()
    for handler in list(root.handlers):
        if handler not in pre_existing:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(prev_level)
    hermes_logging._logging_initialized = False
    hermes_logging.clear_session_context()


@pytest.fixture
def hermes_home():
    home = Path(os.environ["HERMES_HOME"])
    hermes_logging.setup_logging(hermes_home=home)
    return home


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


@pytest.fixture
def sink_adapter(monkeypatch, session_db):
    _patch_create_agent_runtime(monkeypatch, {}, SinkExercisingAgent)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda *a, **k: MagicMock())
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda cfg: {})
    import hermes_cli.config as config_mod

    real_load = config_mod.load_config_readonly

    def _snapshots_enabled(*args, **kwargs):
        cfg = dict(real_load(*args, **kwargs) or {})
        sessions = dict(cfg.get("sessions") or {})
        sessions["write_json_snapshots"] = True
        cfg["sessions"] = sessions
        return cfg

    monkeypatch.setattr(config_mod, "load_config_readonly", _snapshots_enabled)
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: session_db)
    return adapter


async def _wait_status(cli, run_id, want, headers=None):
    for _ in range(120):
        resp = await cli.get(f"/v1/runs/{run_id}", headers=headers or {})
        if resp.status == 200:
            body = await resp.json()
            if body["status"] in want:
                return body
        await asyncio.sleep(0.1)
    raise AssertionError(f"run {run_id} never reached {want}")


def _tree_bytes(root: Path) -> bytes:
    if not root.exists():
        return b""
    return b"\n".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


def _db_bytes(db_path: Path) -> bytes:
    candidates = (Path(str(db_path) + s) for s in ("", "-wal", "-shm"))
    return b"\n".join(p.read_bytes() for p in candidates if p.exists())


def _markers(prefix: str) -> str:
    return f"{prefix}-INPUT-{uuid.uuid4().hex[:8]}"


class TestLeakSweep:  # §6.4
    @pytest.mark.asyncio
    async def test_private_markers_absent_public_markers_present(
        self, sink_adapter, session_db, hermes_home, tmp_path
    ):
        idem_path = tmp_path / "idem.db"
        _use_idempotency_db(sink_adapter, idem_path)
        app = _create_runs_app(sink_adapter)
        grant = _grant()
        priv_in = _markers("PRIVSWEEP")
        pub_in = _markers("PUBSWEEP")
        # §1.1 #19 stderr leg: launchd captures stderr into gateway.error.log,
        # so the verbose StreamHandler must drop private records end-to-end.
        hermes_logging.setup_verbose_logging()
        verbose = next(
            h
            for h in logging.getLogger().handlers
            if getattr(h, "_hermes_verbose", False)
        )
        stderr_stream = io.StringIO()
        verbose.setStream(stderr_stream)
        async with TestClient(TestServer(app)) as cli:
            priv = await cli.post(
                "/v1/runs",
                json={"input": priv_in},
                headers={GRANT_HEADER: grant, "Idempotency-Key": "sweep-priv"},
            )
            pub = await cli.post(
                "/v1/runs",
                json={"input": pub_in},
                headers={"Idempotency-Key": "sweep-pub"},
            )
            assert priv.status == 202 and pub.status == 202
            priv_id = (await priv.json())["run_id"]
            pub_id = (await pub.json())["run_id"]
            priv_status = await _wait_status(
                cli, priv_id, {"completed"}, headers={GRANT_HEADER: grant}
            )
            pub_status = await _wait_status(cli, pub_id, {"completed"})

        # Grant-gated in-memory egress still returns the private output.
        assert priv_status["output"] == priv_in.replace("INPUT", "OUTPUT")
        assert pub_status["output"] == pub_in.replace("INPUT", "OUTPUT")

        hermes_logging.flush_log_queue()
        state_raw = _db_bytes(tmp_path / "state.db").decode("utf-8", "replace")
        sessions_raw = _tree_bytes(hermes_home / "sessions").decode("utf-8", "replace")
        spill_raw = _tree_bytes(hermes_home / "cache" / "spillover").decode("utf-8", "replace")
        logs_raw = _tree_bytes(hermes_home / "logs").decode("utf-8", "replace")
        idem_raw = _db_bytes(idem_path).decode("utf-8", "replace")
        stderr_raw = stderr_stream.getvalue()

        for leg in ("INPUT", "OUTPUT", "TOOLRESULT", "TURNBUDGET", "ERRLOG"):
            marker = priv_in.replace("INPUT", leg)
            for sink_name, blob in (
                ("state.db", state_raw),
                ("sessions dir", sessions_raw),
                ("spillover", spill_raw),
                ("logs", logs_raw),
                ("stderr handler", stderr_raw),
                ("runs_idempotency.db", idem_raw),
            ):
                assert marker not in blob, f"private {leg} marker leaked into {sink_name}"

        # Concurrent public control: same process, same sinks, markers land.
        pub_rows = session_db.get_messages(pub_id)
        pub_text = json.dumps(pub_rows)
        assert pub_in in pub_text
        assert pub_in.replace("INPUT", "OUTPUT") in pub_text
        assert pub_in.replace("INPUT", "TOOLRESULT") in logs_raw
        assert pub_in.replace("INPUT", "ERRLOG") in logs_raw
        assert pub_in.replace("INPUT", "OUTPUT") in idem_raw
        assert pub_in in sessions_raw  # Request dump remains a durable sink.
        assert pub_in.replace("INPUT", "TOOLRESULT") in spill_raw
        assert pub_in.replace("INPUT", "TURNBUDGET") in spill_raw
        assert pub_in.replace("INPUT", "TOOLRESULT") in stderr_raw
        assert pub_in.replace("INPUT", "ERRLOG") in stderr_raw
        # Private rows never entered the transcript store at all.
        assert session_db.get_messages(priv_id) == []

    @pytest.mark.asyncio
    async def test_private_failure_exception_prose_absent(
        self, sink_adapter, hermes_home, tmp_path
    ):
        idem_path = tmp_path / "idem.db"
        _use_idempotency_db(sink_adapter, idem_path)
        app = _create_runs_app(sink_adapter)
        grant = _grant()
        priv_in = _markers("PRIVFAIL") + " RAISE"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": priv_in},
                headers={GRANT_HEADER: grant, "Idempotency-Key": "sweep-fail"},
            )
            run_id = (await resp.json())["run_id"]
            status = await _wait_status(
                cli, run_id, {"failed"}, headers={GRANT_HEADER: grant}
            )
        assert status["error"] == "run_failed"

        hermes_logging.flush_log_queue()
        logs_raw = _tree_bytes(hermes_home / "logs").decode("utf-8", "replace")
        idem_raw = _db_bytes(idem_path).decode("utf-8", "replace")
        assert "PRIVFAIL" not in logs_raw  # every leg, incl. the EXC prose
        assert "PRIVFAIL" not in idem_raw
        stored = json.loads(
            sqlite3.connect(str(idem_path))
            .execute("SELECT status_json FROM run_idempotency")
            .fetchone()[0]
        )
        assert stored["error_class"] == "run_failed"


class TestResponseStoreNeverWritten:
    """§7.1 — no /v1/runs path writes response_store.db for private runs."""

    @pytest.mark.asyncio
    async def test_private_run_never_touches_response_store(
        self, sink_adapter, hermes_home
    ):
        store = sink_adapter._response_store
        store.put = MagicMock()
        store.set_conversation = MagicMock()
        store.get = MagicMock(return_value=None)
        store.get_conversation = MagicMock(return_value=None)
        app = _create_runs_app(sink_adapter)
        grant = _grant()
        priv_in = _markers("PRIVRESP")
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs", json={"input": priv_in}, headers={GRANT_HEADER: grant}
            )
            run_id = (await resp.json())["run_id"]
            await _wait_status(cli, run_id, {"completed"}, headers={GRANT_HEADER: grant})

        store.put.assert_not_called()
        store.set_conversation.assert_not_called()
        store.get.assert_not_called()
        store.get_conversation.assert_not_called()
        raw = _db_bytes(hermes_home / "response_store.db").decode("utf-8", "replace")
        assert priv_in not in raw
        assert priv_in.replace("INPUT", "OUTPUT") not in raw
