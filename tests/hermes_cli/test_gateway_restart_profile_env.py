"""A detached manual restart loads only the selected profile's environment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_cli import gateway as gw


@pytest.mark.parametrize("restart_mode", ["profile", "captured", "captured_home"])
def test_restart_watcher_keeps_profile_environment_isolated(tmp_path, monkeypatch, restart_mode):
    root = tmp_path / ".hermes"
    source = root / "profiles" / "source"
    target = root / "profiles" / "target"
    for home in (root, source, target):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n")
    (root / ".env").write_text("ROOT_DEPLOYMENT_SECRET=root-only\n")
    (source / ".env").write_text("API_SERVER_PORT=8650\nCUSTOM_DEPLOYMENT_SECRET=source-only\n")
    (target / ".env").write_text("TARGET_SETTING=target-only\n")
    (root / "active_profile").write_text("target\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setenv("API_SERVER_PORT", "8650")
    monkeypatch.setenv("API_SERVER_HOST", "127.0.0.2")
    monkeypatch.setenv("CUSTOM_DEPLOYMENT_SECRET", "source-only")
    monkeypatch.setenv("ROOT_DEPLOYMENT_SECRET", "root-only")
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    assert gw._gateway_run_args_for_profile("default")[3:5] == ["--profile", "default"]

    real_popen = subprocess.Popen
    watchers = []

    def capture_watcher(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        watchers.append(process)
        return process

    monkeypatch.setattr(gw.subprocess, "Popen", capture_watcher)
    try:
        for index, profile in enumerate(("source", "target", "source", "default")):
            result_path = tmp_path / f"result-{index}.json"
            child = (
                "import json, os, pathlib\n"
                "import hermes_cli.main\n"
                "from hermes_cli.env_loader import load_hermes_dotenv\n"
                "load_hermes_dotenv(load_external_secrets=False)\n"
                f"pathlib.Path({str(result_path)!r}).write_text(json.dumps(dict(os.environ)))\n"
            )
            monkeypatch.setattr(
                gw, "_gateway_run_args_for_profile",
                lambda name: [sys.executable, "-c", child, "--profile", name],
            )
            if restart_mode == "captured_home":
                home = root if profile == "default" else root / "profiles" / profile
                old = real_popen(
                    [sys.executable, "-c", "import time; time.sleep(20)"],
                    env={**os.environ, "HERMES_HOME": str(home)},
                )
                try:
                    assert gw.launch_detached_gateway_restart_by_cmdline(old.pid, [sys.executable, "-c", child])
                finally:
                    old.terminate()
                    old.wait(timeout=5)
            elif restart_mode == "profile":
                exited = real_popen([sys.executable, "-c", "pass"])
                exited.wait(timeout=5)
                assert gw.launch_detached_profile_gateway_restart(profile, exited.pid)
            else:
                exited = real_popen([sys.executable, "-c", "pass"])
                exited.wait(timeout=5)
                assert gw.launch_detached_gateway_restart_by_cmdline(exited.pid, gw._gateway_run_args_for_profile(profile))
            deadline = time.monotonic() + 8
            while not result_path.exists():
                assert time.monotonic() < deadline, "restart child did not publish environment"
                time.sleep(0.02)
            env = json.loads(result_path.read_text())
            expected_home = root if profile == "default" else root / "profiles" / profile
            assert env["HERMES_HOME"] == str(expected_home)
            assert "_HERMES_GATEWAY" not in env
            if profile == "source":
                assert env["API_SERVER_PORT"] == "8650"
                assert env["API_SERVER_HOST"] == "127.0.0.2"
                assert env["CUSTOM_DEPLOYMENT_SECRET"] == "source-only"
            else:
                assert "API_SERVER_PORT" not in env
                assert "API_SERVER_HOST" not in env
                assert "CUSTOM_DEPLOYMENT_SECRET" not in env
            if profile == "target":
                assert env["TARGET_SETTING"] == "target-only"
                assert "ROOT_DEPLOYMENT_SECRET" not in env
            watchers[-1].wait(timeout=5)
    finally:
        for watcher in watchers:
            if watcher.poll() is None:
                watcher.terminate()
            watcher.wait(timeout=5)


def test_captured_restart_refuses_an_unknown_profile_home(monkeypatch):
    import psutil
    from types import SimpleNamespace

    monkeypatch.setattr(psutil, "Process", lambda _: SimpleNamespace(environ=lambda: {}))
    monkeypatch.setattr(gw.subprocess, "Popen", lambda *_a, **_k: pytest.fail("unknown profile must not spawn"))
    assert not gw.launch_detached_gateway_restart_by_cmdline(4242, [sys.executable, "-m", "hermes_cli.main", "gateway", "run"])
