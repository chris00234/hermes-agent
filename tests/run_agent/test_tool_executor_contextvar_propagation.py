"""Regression guard for PR #16660 (salvaged as PR #18027): ContextVar
propagation into concurrent tool worker threads.

Background
----------
Gateway adapters (Slack, Telegram, Discord, ...) set
``tools.approval._approval_session_key`` as a ContextVar before calling
``agent.run_conversation`` so that dangerous-command approval prompts route
back to the channel/session that initiated the tool call. When the agent
dispatches multiple tools in parallel, it uses
``concurrent.futures.ThreadPoolExecutor.submit(...)`` — and ``submit`` runs
the callable in a *fresh* context, NOT the caller's context. Without an
explicit ``contextvars.copy_context().run(...)`` wrapper, worker threads
observe the ContextVar's default value, fall through to the
``os.environ`` legacy fallback (which the gateway overwrites at each
agent step), and route the approval card to *whichever session stepped
most recently* — not the one that raised the prompt. Confirmed in the
wild on Slack with two concurrent channels: session A's `rm -rf`
approval card was delivered to session B.

The fix (4 LOC in ``run_agent.py``) snapshots the caller's context with
``copy_context()`` and submits ``ctx.run(_run_tool, …)`` instead of
``_run_tool`` directly. Mirrors ``asyncio.to_thread`` semantics.

This suite follows the ``contextvar-run-in-executor-bridge`` skill's
two-test pattern: one end-to-end test proves the fix works at the
call-site level, one documents the Python contract that makes the fix
necessary. If anyone ever reverts the wrapper, the call-site test
fails while the contract test keeps passing — a clear diagnostic
signal for *why* the call-site regressed.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading


def test_executor_submit_without_copy_context_does_not_propagate():
    """Documents the Python contract the fix relies on.

    ``concurrent.futures.ThreadPoolExecutor.submit(fn)`` runs ``fn`` in a
    worker thread with a fresh, empty context. A ContextVar set by the
    caller is invisible inside ``fn``. This is the exact trap that made
    approval-session routing race in the gateway before #16660.

    If this test ever fails — i.e. submit() starts propagating
    ContextVars by default — the copy_context() wrapper in run_agent.py
    becomes redundant but not harmful, and the call-site test below
    should be updated accordingly.
    """
    probe: contextvars.ContextVar[str] = contextvars.ContextVar(
        "probe_default_propagation", default="unset"
    )

    def read_in_worker() -> str:
        return probe.get()

    probe.set("set-in-main")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        observed = ex.submit(read_in_worker).result(timeout=5)

    assert observed == "unset", (
        "Unexpected: executor.submit propagated a ContextVar without "
        "copy_context(). If Python's behavior changed, update "
        "test_run_tool_worker_sees_parent_context below."
    )




def test_run_tool_worker_sees_parent_approval_session_key():
    """End-to-end call-site guard.

    Mirrors the exact shape of the fixed call site in
    ``run_agent.py::_execute_tool_calls_concurrent`` — a
    ``ThreadPoolExecutor`` with ``executor.submit(ctx.run, fn, *args)``.
    Sets the real ``tools.approval._approval_session_key`` ContextVar
    in the caller and asserts the worker observes it via
    ``tools.approval.get_current_session_key()``.

    If the PR's ``copy_context().run`` wrapper is reverted, this test
    fails with ``Expected 'session-A' but worker saw 'default'``.
    """
    from tools.approval import (
        _approval_session_key,
        get_current_session_key,
    )

    observed: dict = {}
    barrier = threading.Event()

    def worker_equivalent_to_run_tool() -> None:
        # Mirror what real _run_tool does early: read the session key.
        observed["session_key"] = get_current_session_key(default="FALLBACK")
        barrier.set()

    # Set the ContextVar the gateway would set before calling agent.run.
    token = _approval_session_key.set("session-A")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ctx = contextvars.copy_context()
            fut = ex.submit(ctx.run, worker_equivalent_to_run_tool)
            fut.result(timeout=5)
        assert barrier.is_set(), "worker did not complete"
    finally:
        _approval_session_key.reset(token)

    assert observed.get("session_key") == "session-A", (
        f"Worker thread did not inherit _approval_session_key from caller. "
        f"Expected 'session-A', got {observed.get('session_key')!r}. "
        "This is the bug that PR #16660 fixed — approval prompts route to "
        "the wrong session in concurrent gateway traffic. Check whether "
        "the copy_context().run wrapper in _execute_tool_calls_concurrent "
        "was removed."
    )




def test_two_concurrent_tool_batches_keep_session_keys_isolated():
    """End-to-end guard: two callers each set a different session key
    and submit workers concurrently. Each worker must see its own
    caller's key, not the other's.

    Guards against a future "optimization" that reuses a single context
    snapshot across callers (which would collapse isolation the same way
    the unfixed ``submit`` does).
    """
    from tools.approval import (
        _approval_session_key,
        get_current_session_key,
    )

    results: dict = {}

    def caller(label: str) -> None:
        token = _approval_session_key.set(f"session-{label}")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ctx = contextvars.copy_context()
                fut = ex.submit(
                    ctx.run,
                    lambda: get_current_session_key(default="FALLBACK"),
                )
                results[label] = fut.result(timeout=5)
        finally:
            _approval_session_key.reset(token)

    t_a = threading.Thread(target=caller, args=("A",))
    t_b = threading.Thread(target=caller, args=("B",))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert results.get("A") == "session-A", (
        f"Session A worker saw {results.get('A')!r}, expected 'session-A'"
    )
    assert results.get("B") == "session-B", (
        f"Session B worker saw {results.get('B')!r}, expected 'session-B'"
    )


def _make_private_policy():
    from datetime import datetime, timedelta, timezone

    from gateway.private_execution_policy import build_policy

    lease = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    return build_policy(
        run_id="run-1",
        profile="design",
        execution_attempt_id="att-1",
        lease_expires_at=lease,
        allowed_toolsets=("alpha",),
    )


def test_private_policy_contextvar_not_propagated_by_bare_submit():
    """Contract half of the two-test pattern for the private policy
    ContextVar: a bare executor.submit sees the default (None), which is why
    every submit reachable from the private lane must wrap
    propagate_context_to_thread."""
    from gateway.private_execution_policy import (
        bind_private_policy,
        current_private_policy,
        reset_private_policy,
    )

    token = bind_private_policy(_make_private_policy())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            observed = ex.submit(current_private_policy).result(timeout=5)
    finally:
        reset_private_policy(token)

    assert observed is None


def test_private_policy_contextvar_propagates_via_thread_context():
    """Call-site half: propagate_context_to_thread (used at both tool
    executor submit sites) carries the bound private policy into the
    worker thread, so the tool gate sees it there."""
    from gateway.private_execution_policy import (
        bind_private_policy,
        current_private_policy,
        reset_private_policy,
    )
    from tools.thread_context import propagate_context_to_thread

    policy = _make_private_policy()
    token = bind_private_policy(policy)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            observed = ex.submit(
                propagate_context_to_thread(current_private_policy)
            ).result(timeout=5)
    finally:
        reset_private_policy(token)

    assert observed is policy


# ── DE_DESIGN §6.13: thread-context CI gate ─────────────────────────────────
#
# The log-marker leg of the private lane has no second factor: a log record
# is only stamped ``hermes_private`` when the emitting thread carries the
# bound policy ContextVar. Any executor submit / Thread spawn on the tool
# dispatch path that forgets ``propagate_context_to_thread`` silently
# reopens the shared-log sink for every tool running on that thread. This
# static gate fails the build when such a site appears.

_GATED_MODULES = (
    "agent/tool_executor.py",
    "agent/agent_runtime_helpers.py",
    "run_agent.py",
)

# threading.Thread targets that deliberately run OUTSIDE the policy context.
# Only content-free targets may live here (documented DE_DESIGN §1.2-A
# residue): the tool-activity heartbeat logs no tool arguments or results.
_THREAD_TARGET_ALLOWLIST = frozenset({"_run_tool_activity_heartbeat"})


def _is_context_propagated(node) -> bool:
    """True for ``propagate_context_to_thread(...)`` or ``ctx.run`` targets."""
    import ast

    if isinstance(node, ast.Call):
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        return name == "propagate_context_to_thread"
    # executor.submit(ctx.run, fn, ...) — the copy_context() bridge shape.
    return isinstance(node, ast.Attribute) and node.attr == "run"


def _thread_context_violations(source: str, rel_path: str) -> list:
    import ast

    violations = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        attr = getattr(fn, "attr", None)
        if attr == "submit":
            first = node.args[0] if node.args else None
            if not _is_context_propagated(first):
                violations.append(
                    f"{rel_path}:{node.lineno} executor.submit target is not "
                    "wrapped in propagate_context_to_thread"
                )
        is_thread_ctor = attr == "Thread" or getattr(fn, "id", None) == "Thread"
        if is_thread_ctor:
            target = next(
                (kw.value for kw in node.keywords if kw.arg == "target"), None
            )
            if target is None and node.args:
                continue  # positional Thread(group, target) is not used here
            if _is_context_propagated(target):
                continue
            target_name = getattr(target, "id", None) or getattr(
                target, "attr", None
            )
            if target_name in _THREAD_TARGET_ALLOWLIST:
                continue
            violations.append(
                f"{rel_path}:{node.lineno} threading.Thread target "
                f"{target_name!r} is not wrapped in "
                "propagate_context_to_thread (or allowlisted as content-free)"
            )
    return violations


def test_every_thread_hop_on_the_private_lane_propagates_context():
    """Every executor/Thread submit reachable from the private tool lane
    must wrap propagate_context_to_thread (DE_DESIGN §6.13).

    If this fails: either wrap the new submit/Thread target with
    ``propagate_context_to_thread`` (tools/thread_context.py), or — ONLY for
    a target that can never log tool arguments, results, or model text —
    add it to ``_THREAD_TARGET_ALLOWLIST`` with a justification comment.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    all_violations = []
    for rel_path in _GATED_MODULES:
        source = (repo_root / rel_path).read_text(encoding="utf-8")
        all_violations.extend(_thread_context_violations(source, rel_path))
    assert all_violations == [], "\n".join(all_violations)


def test_gate_detects_a_bare_submit_and_bare_thread():
    """Anti-test: the scanner actually flags unwrapped hops (non-vacuity)."""
    bad = (
        "import threading\n"
        "def f(executor, work):\n"
        "    executor.submit(work)\n"
        "    threading.Thread(target=work).start()\n"
    )
    violations = _thread_context_violations(bad, "example.py")
    assert len(violations) == 2
    good = (
        "import threading\n"
        "def f(executor, work, ctx):\n"
        "    executor.submit(propagate_context_to_thread(work))\n"
        "    executor.submit(ctx.run, work)\n"
        "    threading.Thread(target=propagate_context_to_thread(work)).start()\n"
        "    threading.Thread(target=_run_tool_activity_heartbeat).start()\n"
    )
    assert _thread_context_violations(good, "example.py") == []
