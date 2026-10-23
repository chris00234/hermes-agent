"""FIFO lifecycle barriers using the memory manager's existing worker."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable

from agent.memory_provider import StatePreservingResetError

logger = logging.getLogger(__name__)
LIFECYCLE_TIMEOUT_SECONDS = 5.0
MAX_PENDING_LIFECYCLES = 16


class MemoryLifecycle:
    def __init__(self, submit: Callable[..., Future | None]) -> None:
        self._submit = submit
        self._lock = threading.RLock()
        self._worker = threading.local()
        self._generation = 0
        self._applied_generation = 0
        self._successful_transition = 0
        self._failed_transition = 0
        self._pending: dict[int, bool] = {}
        # Generations whose caller gave up on an explicit per-call deadline.
        # Their fn must NEVER run (a late /reset retirement would clear the
        # frame after the command already failed closed).
        self._abandoned: set[int] = set()

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def read_generation(self) -> int | None:
        with self._lock:
            if self._pending or self._failed_transition > self._successful_transition:
                return None
            return self._generation

    def accepts_background(self, generation: int) -> bool:
        with self._lock:
            return (generation >= self._applied_generation
                    and self._failed_transition <= self._successful_transition)

    def run_worker(self, fn: Callable[[], Any]) -> Any:
        previous = getattr(self._worker, "active", False)
        self._worker.active = True
        try:
            return fn()
        finally:
            self._worker.active = previous

    def _finish(self, generation: int, transition: bool, outcome: str) -> None:
        with self._lock:
            self._pending.pop(generation, None)
            self._abandoned.discard(generation)
            if not transition or outcome == "neutral":
                return
            if outcome == "succeeded":
                self._successful_transition = max(self._successful_transition, generation)
            else:
                self._failed_transition = max(self._failed_transition, generation)

    def _execute(self, fn: Callable[[], Any], generation: int, transition: bool) -> Any:
        outcome = "failed"
        try:
            with self._lock:
                if generation in self._abandoned:
                    outcome = "neutral"
                    raise RuntimeError("lifecycle task abandoned")
                if generation < self._applied_generation:
                    raise RuntimeError("Memory lifecycle was superseded before execution")
                self._applied_generation = generation
            if not transition and not self.accepts_background(generation):
                raise RuntimeError("A prior memory session transition is incomplete")
            result = fn()
            outcome = "succeeded"
            return result
        except StatePreservingResetError:
            # Nothing was acknowledged or changed: transition-neutral.
            outcome = "neutral"
            raise
        finally:
            self._finish(generation, transition, outcome)

    def _enqueue(self, fn: Callable[[], Any], *, transition: bool) -> tuple[Future, int]:
        with self._lock:
            self._generation += 1
            generation = self._generation
            if len(self._pending) >= MAX_PENDING_LIFECYCLES:
                self._finish(generation, transition, "failed")
                raise RuntimeError("Memory lifecycle queue is full")
            if not transition and getattr(self._worker, "active", False) and any(
                changes_session and pending > self._applied_generation
                for pending, changes_session in self._pending.items()
            ):
                raise RuntimeError("Nested checkpoint cannot overtake a queued memory session transition")
            self._pending[generation] = transition
            run = lambda: self._execute(fn, generation, transition)
            if not getattr(self._worker, "active", False):
                try:
                    future = self._submit(run, inline_fallback=False)
                    if future is None:
                        raise RuntimeError("Memory lifecycle worker is unavailable")
                    return future, generation
                except Exception:
                    self._finish(generation, transition, "failed")
                    raise
        future = Future()
        try:
            future.set_result(run())
        except Exception as exc:
            future.set_exception(exc)
        return future, generation

    def enqueue(self, fn: Callable[[], Any], *, transition: bool) -> Future:
        return self._enqueue(fn, transition=transition)[0]

    def call(self, fn: Callable[[], Any], *, transition: bool, required: bool,
             timeout: float | None = None) -> Any:
        generation = None
        try:
            # A nested worker call can execute synchronously in _enqueue;
            # an already-expired deadline must reject before any provider runs.
            if timeout is not None and timeout <= 0:
                raise TimeoutError("Memory lifecycle deadline already expired")
            future, generation = self._enqueue(fn, transition=transition)
            wait = LIFECYCLE_TIMEOUT_SECONDS if timeout is None else timeout
            return future.result(timeout=wait)
        except Exception as exc:
            # Default-deadline callers keep the legacy behavior: a timed-out
            # callback stays queued, because earlier writes must finish before
            # it can safely clear or rebind provider state. An EXPLICIT
            # per-call deadline is a fail-closed contract instead — abandon
            # the still-queued generation so its fn never runs late.
            if (generation is not None and timeout is not None
                    and isinstance(exc, TimeoutError)):
                with self._lock:
                    if generation in self._pending:
                        self._abandoned.add(generation)
            logger.warning("Memory lifecycle incomplete; provider context is withheld")
            if required:
                raise
            return None
