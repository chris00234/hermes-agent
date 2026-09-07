"""FIFO lifecycle barriers using the memory manager's existing worker."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any, Callable

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

    def _finish(self, generation: int, transition: bool, succeeded: bool) -> None:
        with self._lock:
            self._pending.pop(generation, None)
            if transition and succeeded:
                self._successful_transition = max(self._successful_transition, generation)
            elif transition:
                self._failed_transition = max(self._failed_transition, generation)

    def _execute(self, fn: Callable[[], Any], generation: int, transition: bool) -> Any:
        succeeded = False
        try:
            with self._lock:
                if generation < self._applied_generation:
                    raise RuntimeError("Memory lifecycle was superseded before execution")
                self._applied_generation = generation
            if not transition and not self.accepts_background(generation):
                raise RuntimeError("A prior memory session transition is incomplete")
            result = fn()
            succeeded = True
            return result
        finally:
            self._finish(generation, transition, succeeded)

    def enqueue(self, fn: Callable[[], Any], *, transition: bool) -> Future:
        with self._lock:
            self._generation += 1
            generation = self._generation
            if len(self._pending) >= MAX_PENDING_LIFECYCLES:
                self._finish(generation, transition, False)
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
                    return future
                except Exception:
                    self._finish(generation, transition, False)
                    raise
        future = Future()
        try:
            future.set_result(run())
        except Exception as exc:
            future.set_exception(exc)
        return future

    def call(self, fn: Callable[[], Any], *, transition: bool, required: bool) -> Any:
        try:
            future = self.enqueue(fn, transition=transition)
            return future.result(timeout=LIFECYCLE_TIMEOUT_SECONDS)
        except Exception:
            # A timed-out callback stays queued: earlier writes must finish
            # before it can safely clear or rebind provider state.
            logger.warning("Memory lifecycle incomplete; provider context is withheld")
            if required:
                raise
            return None
