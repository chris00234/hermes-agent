"""Fail-closed privacy checks shared by agent persistence and lifecycle sinks."""

from typing import Any


def private_execution_active(agent: Any) -> bool:
    """Either policy stamp suppresses sinks, even across a lost thread context.

    Dispatch requires both stamps to match; persistence must stay disabled even
    when that dispatch contract is broken or the policy accessor is unavailable.
    """
    try:
        from gateway.private_execution_policy import current_private_policy

        return (
            getattr(agent, "_private_execution_policy", None) is not None
            or current_private_policy() is not None
        )
    except Exception:
        return True
