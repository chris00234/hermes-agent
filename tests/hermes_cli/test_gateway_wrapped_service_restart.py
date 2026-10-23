"""Drain legacy launchd wrappers through their live gateway child."""

import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw


pytestmark = [pytest.mark.platforms("macos"), pytest.mark.spawns_gateway_lookalike]


@pytest.fixture
def legacy_service(tmp_path, monkeypatch):
    package = tmp_path / "hermes_cli"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "main.py").write_text(
        "import os, pathlib, signal, sys, time\n"
        "def drain(*_):\n"
        "    pathlib.Path('drained').write_text('1')\n"
        "    sys.exit(75)\n"
        "signal.signal(signal.SIGUSR1, drain)\n"
        "pathlib.Path('ready').write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    # The old installed wrapper forwarded TERM, but had no USR1 handler.
    (package / "stderr_timestamp.py").write_text(
        "import subprocess, sys\n"
        "child = subprocess.Popen(sys.argv[1:])\n"
        "sys.exit(child.wait())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["HERMES_HOME"] = str(tmp_path / "home")
    wrapper = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.stderr_timestamp", sys.executable,
         "-m", "hermes_cli.main", "gateway", "run", "--external-supervisor"],
        cwd=tmp_path, env=env, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        ready = tmp_path / "ready"
        while not ready.exists():
            assert time.monotonic() < deadline, "gateway child did not start"
            time.sleep(0.01)
        child_pid = int(ready.read_text())
        monkeypatch.setenv("HERMES_HOME", env["HERMES_HOME"])
        yield wrapper, child_pid, tmp_path
    finally:
        try:
            os.killpg(wrapper.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wrapper.wait(timeout=5)


def test_service_exclusion_contains_live_gateway_child(legacy_service, monkeypatch):
    wrapper, child_pid, _ = legacy_service
    monkeypatch.setattr(gw, "_locate_launchd_gateway_service", lambda _: ("gui/501", wrapper.pid))
    assert {wrapper.pid, child_pid} <= gw._get_service_pids()


def test_update_drains_child_before_legacy_wrapper_exits(legacy_service):
    wrapper, _, home = legacy_service
    assert gw._graceful_restart_via_sigusr1(wrapper.pid, 3)
    assert (home / "drained").exists(), "wrapper exited without draining its gateway"
    assert wrapper.wait(timeout=5) == 75


def test_self_restart_reaches_child_of_legacy_wrapper(legacy_service, monkeypatch):
    wrapper, _, home = legacy_service
    monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda _: True)
    assert gw._request_gateway_self_restart(wrapper.pid)
    assert wrapper.wait(timeout=5) == 75
    assert (home / "drained").exists()


def test_exited_descendant_never_redirects_drain_to_legacy_wrapper(monkeypatch):
    import psutil

    def exited():
        raise psutil.NoSuchProcess(4243)

    wrapper = SimpleNamespace(
        cmdline=lambda: [sys.executable, "-m", "hermes_cli.stderr_timestamp"],
        children=lambda **_: [SimpleNamespace(cmdline=exited)],
    )
    monkeypatch.setattr(psutil, "Process", lambda _: wrapper)
    monkeypatch.setattr(gw.os, "kill", lambda *_: pytest.fail("must not signal the wrapper"))
    assert not gw._graceful_restart_via_sigusr1(4242, 3)


def test_store_launcher_wrapper_is_not_mistaken_for_its_gateway(monkeypatch):
    """Store launchers run the timestamp wrapper as ``python -I -c <bootstrap>``; under the
    osascript job only its gateway child may be drained."""
    import psutil
    from hermes_cli._launchers import runtime_command

    gateway_argv = runtime_command(gw.PROJECT_ROOT, ["gateway", "run"], python=sys.executable)
    wrapper_argv = runtime_command(
        gw.PROJECT_ROOT, ["--error-log", "gateway.error.log", "--", *gateway_argv],
        module="hermes_cli.stderr_timestamp", python=sys.executable,
    )
    children = [
        SimpleNamespace(pid=4243, cmdline=lambda: wrapper_argv),
        SimpleNamespace(pid=4244, cmdline=lambda: gateway_argv),
    ]
    osascript = SimpleNamespace(
        cmdline=lambda: ["/usr/bin/osascript", "-l", "JavaScript", "-e", "system()"],
        children=lambda **_: children,
    )
    monkeypatch.setattr(psutil, "Process", lambda _: osascript)
    assert gw.gateway_pid_for_service(4242) == 4244


def test_manual_sweep_does_not_restart_a_new_service_child(legacy_service, monkeypatch):
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome, _restart_manual_gateways

    wrapper, child_pid, _ = legacy_service
    # Model service discovery just before the child was spawned, followed by
    # the process scan seeing that fresh child. It was absent pre-update.
    monkeypatch.setattr(gw, "_get_service_pids", lambda **_: {wrapper.pid})
    monkeypatch.setattr(gw, "find_gateway_pids", lambda **_: [child_pid])
    monkeypatch.setattr(gw, "find_profile_gateway_processes", lambda **_: [])
    outcome = _GatewayRestartOutcome(
        incomplete=False, phase_errors=[], restarted_services=[], failed_or_stale_units=[],
        relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        pre_restart_gateway_pids=[],
    )
    _restart_manual_gateways(outcome, 3)
    assert not outcome.killed_pids
    assert wrapper.poll() is None
