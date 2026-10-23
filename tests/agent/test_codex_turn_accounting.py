import time
from types import SimpleNamespace

import pytest

from agent.codex_runtime import _codex_turn_options
from agent.codex_usage import canonical_usage, record_usage
from agent.transports.codex_app_server_session import TurnResult


def test_usage_excludes_history_deduplicates_updates_and_survives_counter_reset():
    result = TurnResult()
    first = {"inputTokens": 100, "cachedInputTokens": 40, "outputTokens": 20, "totalTokens": 120}
    second = {"inputTokens": 150, "cachedInputTokens": 80, "outputTokens": 30, "totalTokens": 180}
    history = {"inputTokens": 1000, "cachedInputTokens": 500, "outputTokens": 200, "totalTokens": 1200}
    total = {key: history[key] + first[key] for key in history}
    note = {"last": first, "total": total}
    record_usage(result, note)
    record_usage(result, note)
    total = {key: total[key] + second[key] for key in total}
    record_usage(result, {"last": second, "total": total})
    record_usage(result, {"last": second, "total": total})
    assert result.api_call_count == 2
    usage = canonical_usage(result.token_usage_total)
    assert usage.prompt_tokens == 250
    assert usage.input_tokens == 130
    assert usage.cache_read_tokens == 120
    assert usage.total_tokens == 300
    assert result.token_usage_last == second
    record_usage(result, {"last": first, "total": first})
    assert result.api_call_count == 3
    assert canonical_usage(result.token_usage_total).total_tokens == 420


@pytest.mark.parametrize("budget,elapsed,expected", [(None, 0, None), (1800, 100, 1700), (30, 40, 0)])
def test_runtime_options_use_profile_effort_and_explicit_budget(tmp_path, monkeypatch, budget, elapsed, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: high\n  gateway_timeout: 1800\n")
    monkeypatch.setattr(time, "time", lambda: 2000)
    agent = SimpleNamespace(model="gpt-6-astra", reasoning_config=None, run_budget_seconds=budget,
                            _run_budget_started_at=2000 - elapsed)
    assert _codex_turn_options(agent) == {
        "model": "gpt-6-astra", "reasoning_effort": "high", "reasoning_summary": "auto", "turn_timeout": expected,
    }


@pytest.mark.parametrize("model,reasoning_config,effort,summary", [
    ("gpt-6-astra", {"enabled": True, "effort": "high"}, "high", "auto"),
    ("gpt-5.4", {"enabled": False}, "none", "none"),
    # No ``none`` level: the disable leaves effort unset, and the summary must still be off.
    ("gpt-6-astra", {"enabled": False}, None, "none"),
])
def test_reasoning_summary_is_off_whenever_reasoning_is_disabled(tmp_path, monkeypatch, model, reasoning_config,
                                                                 effort, summary):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(model=model, reasoning_config=reasoning_config, provider="openai-codex",
                            base_url="https://chatgpt.com/backend-api/codex", run_budget_seconds=None)
    options = _codex_turn_options(agent)
    assert (options["reasoning_effort"], options["reasoning_summary"]) == (effort, summary)
