"""Private execution context propagation regression guards."""

import concurrent.futures
from datetime import UTC


def _make_private_policy():
    from datetime import datetime, timedelta, timezone

    from gateway.private_execution_policy import build_policy

    lease = (datetime.now(UTC) + timedelta(hours=1)).strftime(
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
