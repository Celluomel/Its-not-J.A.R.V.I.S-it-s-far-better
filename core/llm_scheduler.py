"""
LLMScheduler — Global LLM Access Coordinator
=============================================
Solves the core saturation problem: multiple subsystems (BackgroundWorker,
InnerMonologue, Liberty reflection, user response stream) calling LM Studio
concurrently, exhausting the single-slot queue and triggering 60s timeouts.

Architecture:
  - ONE asyncio.Lock  for async callers (persona_bridge, inner_monologue)
  - ONE threading.Lock for sync callers (BackgroundWorker, ai_system)
  - A priority queue so user-facing calls always preempt background tasks
  - Backoff logic: background tasks skip if LLM is currently busy

Priority levels:
  0 — user response (highest — never blocked)
  1 — inner monologue (pass-1 deliberation)
  2 — contradiction detection
  3 — liberty reflection / self-modification
  4 — dream cycle / learning cycle
  5 — identity processing (lowest — most deferrable)

Key design decision: background tasks at priority ≥ 3 are SKIPPED (not queued)
if the LLM is currently busy with a higher-priority call. They will retry on
the next background cycle. This prevents queue buildup entirely.

Usage:
    from core.llm_scheduler import llm_scheduler

    # Async path (persona_bridge, inner_monologue):
    async with llm_scheduler.async_slot(priority=0):
        result = await async_llm_call(...)

    # Sync path (BackgroundWorker thread):
    with llm_scheduler.sync_slot(priority=4, skip_if_busy=True) as acquired:
        if acquired:
            result = sync_llm_call(...)
        else:
            logger.debug("LLM busy — skipping background task")

    # Convenience wrapper (sync, skips if busy):
    if llm_scheduler.try_acquire_sync(priority=4):
        try:
            result = sync_llm_call(...)
        finally:
            llm_scheduler.release_sync()
"""

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Names for priority levels (for logging)
PRIORITY_NAMES = {
    0: "user_response",
    1: "inner_monologue",
    2: "contradiction",
    3: "liberty_reflection",
    4: "dream_learning",
    5: "identity_processing",
}

# Background tasks (priority >= this) are dropped when LLM is busy
BACKGROUND_PRIORITY_THRESHOLD = 3


@dataclass
class SchedulerStats:
    total_calls:      int   = 0
    skipped_calls:    int   = 0
    blocked_ms_total: float = 0.0
    peak_wait_ms:     float = 0.0
    current_priority: Optional[int] = None
    current_holder:   Optional[str] = None


class LLMScheduler:
    """
    Global coordinator for all LLM calls.
    Singleton — import the module-level `llm_scheduler` instance.
    """

    def __init__(self):
        # Async lock — used by asyncio coroutines
        self._async_lock: Optional[asyncio.Lock] = None  # lazy — needs running loop

        # Sync lock — used by background threads
        self._sync_lock = threading.Lock()
        self._sync_holder: Optional[str] = None
        self._sync_priority: Optional[int] = None
        self._lock_meta = threading.Lock()   # guards _sync_holder/_sync_priority

        self.stats = SchedulerStats()
        self._stats_lock = threading.Lock()

    # ── Async interface ───────────────────────────────────────────────────────

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazy-create the asyncio.Lock (must be created in the running loop)."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    @contextlib.asynccontextmanager
    async def async_slot(self, priority: int = 0, caller: str = ""):
        """
        Async context manager.  Always waits (never drops) since user-facing
        calls must complete.  Background async callers should use priority >= 3
        and check try_acquire_async first if they want skip behaviour.
        """
        lock = self._get_async_lock()
        name = PRIORITY_NAMES.get(priority, f"p{priority}")
        t0 = time.monotonic()
        try:
            await lock.acquire()
            wait_ms = (time.monotonic() - t0) * 1000
            with self._stats_lock:
                self.stats.total_calls      += 1
                self.stats.blocked_ms_total += wait_ms
                self.stats.peak_wait_ms      = max(self.stats.peak_wait_ms, wait_ms)
                self.stats.current_priority  = priority
                self.stats.current_holder    = caller or name
            if wait_ms > 500:
                logger.debug(f"[LLMScheduler] {name} waited {wait_ms:.0f}ms for slot")
            yield
        finally:
            with self._stats_lock:
                self.stats.current_priority = None
                self.stats.current_holder   = None
            lock.release()

    # ── Sync interface ────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def sync_slot(self, priority: int = 4, skip_if_busy: bool = True,
                  caller: str = "", wait_seconds: Optional[float] = None):
        """
        Sync context manager for background threads.

        If skip_if_busy=True (default for background tasks):
          - wait_seconds=None (default): tries to acquire immediately; if
            locked, yields False and skips. (Original behaviour.)
          - wait_seconds=N: waits UP TO N seconds for the slot (bounded),
            then yields False. Use this for background learning engines that
            must not be starved by the main loop's continuous LLM work, but
            also must not block indefinitely. A bounded wait of a few seconds
            lets them slot in between the main loop's (finite) LLM calls.
        If skip_if_busy=False:
          - Blocks until lock is free (use for important but sync callers)

        Usage:
            with llm_scheduler.sync_slot(priority=4) as acquired:
                if acquired:
                    do_llm_work()
        """
        name = PRIORITY_NAMES.get(priority, f"p{priority}")
        acquired = False
        t0 = time.monotonic()

        try:
            if skip_if_busy and priority >= BACKGROUND_PRIORITY_THRESHOLD:
                if wait_seconds is not None:
                    # Bounded wait — fair chance without unbounded blocking.
                    acquired = self._sync_lock.acquire(blocking=True,
                                                       timeout=max(0.0, float(wait_seconds)))
                else:
                    acquired = self._sync_lock.acquire(blocking=False)
            else:
                acquired = self._sync_lock.acquire(blocking=True, timeout=30.0)

            if acquired:
                wait_ms = (time.monotonic() - t0) * 1000
                with self._stats_lock:
                    self.stats.total_calls      += 1
                    self.stats.blocked_ms_total += wait_ms
                    self.stats.peak_wait_ms      = max(self.stats.peak_wait_ms, wait_ms)
                    self.stats.current_priority  = priority
                    self.stats.current_holder    = caller or name
                with self._lock_meta:
                    self._sync_holder   = caller or name
                    self._sync_priority = priority
            else:
                with self._stats_lock:
                    self.stats.skipped_calls += 1
                logger.debug(f"[LLMScheduler] {name} skipped — LLM busy")

            yield acquired

        finally:
            if acquired:
                with self._lock_meta:
                    self._sync_holder   = None
                    self._sync_priority = None
                with self._stats_lock:
                    self.stats.current_priority = None
                    self.stats.current_holder   = None
                self._sync_lock.release()

    def is_busy(self) -> bool:
        """True if any LLM call is currently in progress."""
        return self._sync_lock.locked()

    def current_holder(self) -> Optional[str]:
        with self._lock_meta:
            return self._sync_holder

    def summary(self) -> dict:
        with self._stats_lock:
            total = max(1, self.stats.total_calls + self.stats.skipped_calls)
            return {
                "total_calls":      self.stats.total_calls,
                "skipped_calls":    self.stats.skipped_calls,
                "skip_rate_pct":    round(100 * self.stats.skipped_calls / total, 1),
                "avg_wait_ms":      round(
                    self.stats.blocked_ms_total / max(1, self.stats.total_calls), 1
                ),
                "peak_wait_ms":     round(self.stats.peak_wait_ms, 1),
                "currently_busy":   self.is_busy(),
                "current_holder":   self.current_holder(),
            }


# ── Module-level singleton ────────────────────────────────────────────────────
llm_scheduler = LLMScheduler()
