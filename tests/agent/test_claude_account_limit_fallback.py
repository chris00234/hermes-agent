"""A Claude Max account at its plan limit, as the first fallback entry, must hand the turn on.

Opt-in end-to-end check of the managed profiles' fallback chain (``claude1``..``claude4``, served
by the out-of-tree ``claude-account-selector`` plugin over the official
``claude-subscription-directsdk-experimental`` transport). Point ``HERMES_CLAUDE_ACCOUNT_PLUGINS``
at a copy of both plugin directories outside the real Hermes home (the home I/O guard refuses
``~/.hermes``). Every Claude process is a fake CLI that prints the real CLI's limit shape; the
admission relay's upstream is a loopback 429 fixture and any other outbound connection fails.
"""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

PLUGINS = os.environ.get("HERMES_CLAUDE_ACCOUNT_PLUGINS", "")
pytestmark = pytest.mark.skipif(
    not (PLUGINS and (Path(PLUGINS) / "claude-account-selector").is_dir()),
    reason="set HERMES_CLAUDE_ACCOUNT_PLUGINS to a copy of the Claude account plugins")

OFFICIAL = "claude-subscription-directsdk-experimental"
LIMIT_TEXT = "You've hit your weekly limit · resets Oct 25 at 8am (America/Los_Angeles)"
UPSTREAM_429 = {"type": "error", "error": {"type": "rate_limit_error",
                "message": "This request would exceed your account's rate limit. Please try again later."}}

# Request-mode fake Claude Code: logs its account, then either relays one Messages call to the
# 429 fixture (``relay``, what the real CLI does) or fails without one (``synthetic``) and prints
# the CLI's limit shape (assistant error=rate_limit; result is_error/subtype success/
# api_error_status 429; exit 1). Unlisted accounts answer normally.
NATIVE = r'''
import json, os, sys
account = os.environ["CLAUDE_CONFIG_DIR"].rsplit("/", 1)[-1]
with open(os.environ["FAKE_SPAWN_LOG"], "a") as log:
    log.write(account + "\n")
for line in sys.stdin:
    if json.loads(line).get("shouldQuery") is False:
        print(json.dumps({"type": "result", "num_turns": 0, "is_error": False}), flush=True)
mode = os.environ.get("FAKE_MODE_" + account, "ok")
if mode != "ok":
    if mode == "relay":
        import urllib.error, urllib.request
        try:
            urllib.request.urlopen(urllib.request.Request(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages",
                                   data=b"{}", headers={"Content-Type": "application/json"}), timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 429, error.code
    text = os.environ["FAKE_LIMIT_TEXT"]
    print(json.dumps({"type": "assistant", "error": "rate_limit", "parent_tool_use_id": None, "message": {
        "id": "synthetic", "type": "message", "role": "assistant", "model": "<synthetic>",
        "stop_reason": "stop_sequence", "usage": {"input_tokens": 0, "output_tokens": 0},
        "content": [{"type": "text", "text": text}]}}), flush=True)
    print(json.dumps({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                      "num_turns": 1, "result": text, "stop_reason": "stop_sequence", "total_cost_usd": 0,
                      "usage": {"input_tokens": 0, "output_tokens": 0}, "modelUsage": {}}), flush=True)
    sys.exit(1)
reply = "reply from " + account
print(json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                  "delta": {"type": "text_delta", "text": reply}}}), flush=True)
print(json.dumps({"type": "assistant", "message": {"id": "msg", "role": "assistant", "model": "claude-opus-5-5",
                  "stop_reason": "end_turn", "content": [{"type": "text", "text": reply}]}}), flush=True)
print(json.dumps({"type": "stream_event", "event": {"type": "message_stop"}}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
                  "usage": {"input_tokens": 3, "output_tokens": 5}, "total_cost_usd": 0, "modelUsage": {}}), flush=True)
'''


class PrimaryRateLimit(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


@pytest.fixture
def spawns(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_FOUNDRY_API_KEY",
                 "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        monkeypatch.delenv(name, raising=False)
    native = tmp_path / "claude"
    native.write_text(f"#!{sys.executable}\n" + NATIVE)
    native.chmod(0o755)
    plugins = get_hermes_home() / "plugins"
    shutil.copytree(Path(PLUGINS) / OFFICIAL, plugins / OFFICIAL,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "tests", "evals"))
    shutil.copytree(Path(PLUGINS) / "claude-account-selector", plugins / "claude-account-selector",
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "tests"))
    for wrapper in (plugins / "claude-account-selector" / "bin").iterdir():
        wrapper.write_text(wrapper.read_text().replace("/Users/chrischo/.local/bin/claude", str(native)))
    log = tmp_path / "spawns.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_SPAWN_LOG", str(log))
    monkeypatch.setenv("FAKE_LIMIT_TEXT", LIMIT_TEXT)

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(UPSTREAM_429).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda host, port=None, **kw: http.client.HTTPConnection(
        "127.0.0.1", server.server_port, timeout=kw.get("timeout")))
    connect = socket.create_connection

    def loopback_only(address, *args, **kwargs):
        assert address[0] in ("127.0.0.1", "localhost"), f"outbound connection attempted: {address}"
        return connect(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", loopback_only)
    yield lambda: log.read_text().split()
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("mode", ["relay", "synthetic"])
def test_limited_first_fallback_entry_hands_the_turn_on(spawns, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE_max1", mode)
    primary = MagicMock()
    primary.chat.completions.create.side_effect = PrimaryRateLimit()
    tools = [{"type": "function", "function": {"name": "web_search", "description": "search",
              "parameters": {"type": "object", "properties": {}}}}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tools),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=primary),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("hermes_cli.observability.shared_metrics_events.record_fallback") as switches,
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12", base_url="https://open.bigmodel.cn/api/coding/paas/v4", provider="zai",
            model="glm-5.1", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "claude1", "model": "claude-opus-5-5[1m]"},
                            {"provider": "claude2", "model": "claude-opus-5-5[1m]"}])
        with (patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
              patch.object(agent, "_cleanup_task_resources")):
            turns = [agent.run_conversation("first")]
            agent._rate_limited_until = 0  # a later turn: the primary's cooldown has lapsed
            turns.append(agent.run_conversation("second"))

    for result in turns:
        assert result["final_response"] == "reply from max2"
        assert not result.get("failed")
    # Turn one spawns the limited account once; turn two refuses it without a spawn.
    assert spawns() == ["max1", "max2", "max2"]
    hops = [(c.kwargs["from_provider"], c.kwargs["to_provider"], c.kwargs["reason"].value)
            for c in switches.call_args_list]
    assert hops == [("zai", "claude1", "rate_limit"), ("claude1", "claude2", "rate_limit")] * 2
