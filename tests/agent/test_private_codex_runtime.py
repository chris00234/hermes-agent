"""Native Codex cannot enforce Hermes private execution's tool and persistence fences."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import codex_runtime
from gateway import private_execution_policy as policies


@pytest.mark.parametrize("stamp", ["agent", "context", "both", "accessor_error"])
def test_private_native_turn_is_rejected_before_creating_a_session(stamp, monkeypatch):
    policy = policies.build_policy(
        run_id="private-native", profile="default", execution_attempt_id="private-native-attempt",
        lease_expires_at="2099-01-01T00:00:00Z", allowed_toolsets=["web"],
    )
    agent = SimpleNamespace()
    if stamp in {"agent", "both"}:
        agent._private_execution_policy = policy
    create = Mock(side_effect=AssertionError("native session creation reached"))
    monkeypatch.setattr(codex_runtime, "_ensure_codex_session", create)
    if stamp == "accessor_error":
        monkeypatch.setattr(policies, "current_private_policy", Mock(side_effect=RuntimeError))
    token = policies.bind_private_policy(policy if stamp in {"context", "both"} else None)
    try:
        with pytest.raises(RuntimeError, match="Private execution does not support codex_app_server"):
            codex_runtime.run_codex_app_server_turn(
                agent, user_message="private", original_user_message="private",
                messages=[], effective_task_id="private-native",
            )
    finally:
        policies.reset_private_policy(token)
    create.assert_not_called()
