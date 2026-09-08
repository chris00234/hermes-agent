"""Durable working-frame scope wiring for the gateway (contract B).

Two jobs: bind the session store's ``record_working_frame_scope`` writer to a
freshly created agent's memory manager, and retire the acknowledged frame
before ``/reset`` //``/undo`` — via the cached agent's manager when it opts in,
otherwise via a bare, never-``initialize()``d configured provider.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("gateway.run")


class OrphanedWorkingFrameScopeError(RuntimeError):
    """The configured memory provider can NEVER retire this durable scope.

    Raised only on the explicit unrecoverable determination — no provider is
    configured, or the configured provider differs from the scope's — so the
    gateway can drop the orphaned scope instead of blocking /reset //undo
    with 'try again' until the scope expires (up to 7 days). Transient load
    or retirement failures keep raising plain errors and stay fail-closed.
    """


def _bind_store_sink(manager, runner, session_key: str,
                     session_id_fn: Callable[[], str]) -> bool:
    store = getattr(runner, "session_store", None)
    record = getattr(store, "record_working_frame_scope", None)
    setter = getattr(manager, "set_working_frame_scope_sink", None)
    if not callable(record) or not callable(setter):
        return False
    # The store's agent-vs-profile fence needs a real profile source on
    # non-multiplexed gateways (entry.origin.profile is None there).
    profile_fn = getattr(runner, "_active_profile_name", None)
    setter(lambda reference: record(
        session_key, session_id_fn(), reference,
        active_profile=profile_fn() if callable(profile_fn) else None,
    ))
    return True


def attach_working_frame_scope_sink(runner, agent, session_key: str) -> None:
    """Bind durable scope persistence to a fresh agent's memory manager.

    Defensive by contract: agent creation must never fail because the
    manager, store, or session key is missing. Reads ``agent.session_id``
    at call time so ACKs always target the agent's live session.
    """
    try:
        manager = getattr(agent, "_memory_manager", None)
        if manager is None or not session_key:
            return
        _bind_store_sink(
            manager, runner, session_key,
            lambda: getattr(agent, "session_id", ""),
        )
    except Exception:
        logger.warning("Working frame scope sink attachment failed", exc_info=True)


def _cached_agent_manager(runner, session_key: str):
    cache = getattr(runner, "_agent_cache", None)
    if cache is None:
        return None
    lock = getattr(runner, "_agent_cache_lock", None)
    if lock is not None:
        with lock:
            cached = cache.get(session_key)
    else:
        cached = cache.get(session_key)
    agent = cached[0] if isinstance(cached, tuple) else cached
    manager = getattr(agent, "_memory_manager", None) if agent is not None else None
    supports = getattr(manager, "supports_pre_session_reset", None)
    try:
        return manager if callable(supports) and supports() else None
    except Exception:
        return None


def _build_bare_reset_manager(runner, session_key: str, scope: Dict[str, Any]):
    """Load the configured provider uninitialized and wrap it for retirement.

    Never calls ``provider.initialize()`` and never constructs an AIAgent:
    the provider works from ``scope`` identity plus its host-set profile.
    """
    from agent.memory_manager import MemoryManager
    from plugins.memory import load_memory_provider
    from tools.memory_tool import get_builtin_memory_config

    name = str(get_builtin_memory_config().get("provider") or "").strip()
    if name != str(scope.get("provider") or ""):
        raise OrphanedWorkingFrameScopeError(
            f"configured memory provider {name or None!r} cannot retire a "
            f"{scope.get('provider')!r} scope"
        )
    provider = load_memory_provider(name)
    if provider is None:
        return None
    context = getattr(provider, "context", None)
    if context is not None:
        # The provider uses ``context.profile`` as its HTTP actor. A bare
        # provider that cannot be stamped with the serving profile would
        # retire the frame as an unknown actor — fail closed instead.
        profile_fn = getattr(runner, "_active_profile_name", None)
        try:
            profile = profile_fn() if callable(profile_fn) else None
            if not profile:
                raise RuntimeError("no serving profile for bare reset provider")
            context.profile = profile
        except Exception as exc:
            logger.warning(
                "Could not stamp serving profile on bare reset provider: %s",
                type(exc).__name__,
            )
            return None
    manager = MemoryManager()
    manager._providers = [provider]
    _bind_store_sink(manager, runner, session_key, lambda: scope["session_id"])
    return manager


def _resolve_reset_manager(runner, session_key: str,
                           scope: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
    manager = _cached_agent_manager(runner, session_key)
    if manager is not None:
        return manager, False
    manager = _build_bare_reset_manager(runner, session_key, scope)
    return manager, manager is not None


def retire_working_frame_scope_sync(runner, session_key: str,
                                    scope: Dict[str, Any], rewound: bool,
                                    deadline: Optional[float] = None) -> Dict[str, Any]:
    """Retire the acknowledged frame; return the durable cleared ACK.

    Raises on any failure so the caller fails closed (no reset/rewind).
    Runs off-loop — provider retirement is a blocking HTTP round-trip.
    ``deadline`` (time.monotonic) is the gateway's own fail-closed deadline:
    a call that starts past it (executor queue lag) must never retire the
    frame the caller already reported unchanged, and the manager's per-call
    budget is derived from what remains so its abandonment fence can never
    extend past the gateway's reply.
    """
    from agent.memory_provider import StatePreservingResetError

    if deadline is not None and time.monotonic() >= deadline:
        raise StatePreservingResetError(
            "working frame retirement missed its caller deadline before starting"
        )
    manager, bare = _resolve_reset_manager(runner, session_key, scope)
    if manager is None:
        raise RuntimeError("No memory manager available for working frame retirement")
    kwargs: Dict[str, Any] = {}
    if deadline is not None:
        kwargs["timeout"] = max(0.0, deadline - time.monotonic())
    timed_out = False
    try:
        return manager.on_pre_session_reset(
            scope["session_id"], reference=dict(scope), rewound=rewound, **kwargs,
        )
    except TimeoutError:
        timed_out = True
        raise
    finally:
        if bare:
            try:
                # A timed-out retirement is abandoned but its worker may be
                # wedged: mark the manager shutting down without waiting out
                # the drain window (contract B item 4).
                manager.shutdown_all(drain=not timed_out)
            except Exception:
                logger.debug("Bare reset memory manager shutdown failed", exc_info=True)
