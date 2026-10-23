"""MCP servers codex launches on the app-server runtime run under the driving profile's HERMES_HOME.

Codex launches stdio MCP servers from the shared ``config.toml`` with its env allowlist plus the
table's own env, so a server keyed on ``HERMES_HOME`` (Brain's MCP picks its credential by it)
acted as no profile at all. Hermes' own MCP client launches each enabled stdio server a profile
declares with that profile's ``HERMES_HOME``; the app-server launch must match.
"""

import json
import subprocess
import tomllib

import pytest

from agent.transports import codex_app_server as cas

_CODEX_CONFIG = """
[mcp_servers.brain]
command = "brain-mcp"

[mcp_servers.brain.env]
BRAIN_ACTOR = "codex_studio"

[mcp_servers.hermes-tools]
command = "hermes-tools-mcp"

[mcp_servers.off]
command = "off-mcp"

[mcp_servers.quiet]
command = "quiet-mcp"

[mcp_servers.other]
command = "other-mcp"
"""

_PROFILE_CONFIG = """
mcp_servers:
  brain: {command: brain-mcp}
  rea: {command: rea-mcp}
  off: {command: off-mcp, enabled: false}
  quiet: {command: quiet-mcp, enabled: "false"}
  penpot: {url: "http://127.0.0.1:9/mcp"}
"""


class _RecordingPopen:
    commands: list[list[str]] = []

    def __init__(self, cmd, *args, **kwargs):
        type(self).commands.append(list(cmd))
        self.stdin = self.stdout = self.stderr = None
        self.pid = 1
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "liz"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PROFILE_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """``launch(codex_config) -> list[str]`` of the ``mcp_servers.*`` overrides in the app-server argv."""
    _RecordingPopen.commands = []
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    codex_home = tmp_path / "codex"
    codex_home.mkdir()

    def _launch(codex_config):
        if codex_config is not None:
            (codex_home / "config.toml").write_text(codex_config, encoding="utf-8")
        client = cas.CodexAppServerClient(codex_bin="codex", codex_home=str(codex_home))
        client._closed = True
        cmd = _RecordingPopen.commands.pop()
        return [arg for arg in cmd if arg.startswith("mcp_servers.")]

    return _launch


def _server_of(override: str) -> str:
    return next(iter(tomllib.loads(override)["mcp_servers"]))


def test_each_declared_codex_server_gets_the_profile_home(launch, profile_home):
    overrides = launch(_CODEX_CONFIG)

    brain = [arg for arg in overrides if _server_of(arg) == "brain"]
    assert brain == [f"mcp_servers.brain.env.HERMES_HOME={json.dumps(str(profile_home))}"]
    targeted = [_server_of(arg) for arg in overrides]
    assert targeted.count("hermes-tools") == 1
    assert not {"rea", "off", "quiet", "penpot", "other"} & set(targeted)


def test_overrides_merge_into_existing_codex_tables(launch, profile_home):
    """Producer/consumer contract: every override names a server the codex config defines, and
    applied as codex applies ``-c`` (dotted-path sets) it only adds HERMES_HOME to that env."""
    config = tomllib.loads(_CODEX_CONFIG)
    before = set(config["mcp_servers"])
    for override in launch(_CODEX_CONFIG):
        name = _server_of(override)
        assert name in before
        env = tomllib.loads(override)["mcp_servers"][name]["env"]
        config["mcp_servers"][name].setdefault("env", {}).update(env)

    assert set(config["mcp_servers"]) == before
    assert config["mcp_servers"]["brain"]["env"] == {"BRAIN_ACTOR": "codex_studio", "HERMES_HOME": str(profile_home)}


def test_no_codex_config_adds_no_profile_overrides(launch, profile_home):
    assert [_server_of(arg) for arg in launch(None)] == ["hermes-tools"]
