"""B22: the isolated chat-only AIRI profile (api_server, no tools, sessions in ~) loaded
~/.hermes.md: coding action-bias rules, Brain admin URLs and the owner-credential path.
gateway.platforms.api_server.skip_context_files now drops cwd discovery for api_server
agents, as it does for the messaging platforms; the SOUL persona stays."""

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _captured_kwargs(monkeypatch, user_config):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "openai-codex"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.5")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: user_config)
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": None))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    adapter._create_agent(session_id="api-airi")
    return captured


def test_api_server_skip_context_files_keeps_only_the_persona(monkeypatch):
    kwargs = _captured_kwargs(monkeypatch, {"gateway": {"platforms": {"api_server": {"skip_context_files": True}}}})
    assert kwargs["skip_context_files"] is True and kwargs["load_soul_identity"] is True


def test_api_server_loads_context_files_by_default(monkeypatch):
    for config in ({}, {"gateway": {"platforms": ["api_server"]}}):
        assert "skip_context_files" not in _captured_kwargs(monkeypatch, config)
