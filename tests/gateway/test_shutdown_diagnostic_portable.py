"""Execute the portable shutdown diagnostic supervisor without coreutils."""

import subprocess
import time

import pytest

from gateway.shutdown_forensics import _diagnostic_command


@pytest.mark.parametrize("script, timeout, expected", [
    ("printf bounded-diagnostic", 3, 0),
    ("sleep 30", 0.1, 124),
])
def test_stdlib_diagnostic_supervisor(monkeypatch, script, timeout, expected):
    monkeypatch.setattr("gateway.shutdown_forensics.shutil.which", lambda name: None)
    started = time.monotonic()
    result = subprocess.run(_diagnostic_command(script, timeout), capture_output=True, timeout=5)
    assert result.returncode == expected
    assert time.monotonic() - started < 5
    if expected == 0:
        assert result.stdout == b"bounded-diagnostic"


def test_coreutils_supervisor_remains_preferred(monkeypatch):
    monkeypatch.setattr("gateway.shutdown_forensics.shutil.which", lambda name: "/bin/timeout" if name == "timeout" else None)
    assert _diagnostic_command("printf diagnostic", 3) == [
        "/bin/timeout", "3", "bash", "-c", "printf diagnostic",
    ]
