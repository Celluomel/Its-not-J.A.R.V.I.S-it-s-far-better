"""
Cognitive Clock
===============
Multi-timescale scheduler that drives the orchestrator's background loops.

Inspired by biological neural time-scales:
  - fast    (30 s)   → energy regen, decay, maintenance
  - medium  (120 s)  → reflection, meta-cognition
  - slow    (600 s)  → memory consolidation, contradiction sweep
  - evolution (3600 s) → personality adjustment, self-concept update

The clock is stateless with respect to wall-clock time — it simply compares
time.time() against the last time each tick fired.  The orchestrator updates
the timestamps when it acts on a tick.
"""

import time
from dataclasses import dataclass, field
from typing import Dict


# ── Timing constants (seconds) ────────────────────────────────────────────────

FAST_INTERVAL      = 30       # maintenance: regen + decay
MEDIUM_INTERVAL    = 120      # reflection cycle
SLOW_INTERVAL      = 600      # memory consolidation
EVOLUTION_INTERVAL = 3600     # personality / self-concept evolution


@dataclass
class CognitiveClock:
    """
    Returns True from each tick_*() method when enough time has elapsed.
    Call mark_*() after acting on a tick to reset the timer.
    """

    _last_fast:      float = field(default_factory=time.time)
    _last_medium:    float = field(default_factory=time.time)
    _last_slow:      float = field(default_factory=time.time)
    _last_evolution: float = field(default_factory=time.time)

    # ── Tick checks ───────────────────────────────────────────────────────

    def fast_due(self) -> bool:
        return time.time() - self._last_fast >= FAST_INTERVAL

    def medium_due(self) -> bool:
        return time.time() - self._last_medium >= MEDIUM_INTERVAL

    def slow_due(self) -> bool:
        return time.time() - self._last_slow >= SLOW_INTERVAL

    def evolution_due(self) -> bool:
        return time.time() - self._last_evolution >= EVOLUTION_INTERVAL

    # ── Mark as handled ───────────────────────────────────────────────────

    def mark_fast(self)      -> None: self._last_fast      = time.time()
    def mark_medium(self)    -> None: self._last_medium    = time.time()
    def mark_slow(self)      -> None: self._last_slow      = time.time()
    def mark_evolution(self) -> None: self._last_evolution = time.time()

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        now = time.time()
        return {
            "fast_in_sec":      max(0, FAST_INTERVAL      - (now - self._last_fast)),
            "medium_in_sec":    max(0, MEDIUM_INTERVAL    - (now - self._last_medium)),
            "slow_in_sec":      max(0, SLOW_INTERVAL      - (now - self._last_slow)),
            "evolution_in_sec": max(0, EVOLUTION_INTERVAL - (now - self._last_evolution)),
        }
