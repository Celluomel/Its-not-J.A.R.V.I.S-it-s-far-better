"""
cognition/cognitive_resource_economy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CognitiveResourceEconomy (v50) — finite resource pools with hard
depletion and time-based recovery.

Why this exists
────────────────
Without resource scarcity, Lumina can simultaneously hold 47 open
questions, run an experiment, generate curiosity, initiate outreach,
run a cognitive audit trial, and be in counterfactual mode — all at
full capacity.  A cognitive organism must choose because it cannot
do everything at once.  The absence of scarcity makes the system feel
always-on, always-available, always-productive — which is the opposite
of alive.

This module provides the scarcity layer.  The CognitiveBehaviorGate
reads these values and enforces hard suppression.  This module only
tracks state — the gate does the enforcement.

Three resource pools
─────────────────────
cognitive_energy  (0–1)
    The cost of reasoning: LLM calls, reflection, experimentation,
    audit trials.  Depletes fast, recovers moderately.
    Low cognitive_energy → fewer LLM calls, shallower reflection,
    no new experiments.

social_energy  (0–1)
    The cost of interaction: human exchanges, outreach, child dialogue.
    Each interaction depletes by impact.  Recovers faster during
    non-interaction periods (dream cycles, slow processing).
    Low social_energy → outreach blocked regardless of trust score.

attention  (0–1)
    The cost of holding open goals, questions, and aspirations
    simultaneously.  Each active goal costs attention per slow cycle.
    Recovers when goals close (resolved question returns its budget).
    Low attention → new open questions rejected, new curiosity topics
    suppressed.  Existing tracked items still processed normally.

Recovery mechanics
───────────────────
Recovery happens continuously in tick(), which is called every slow
cycle.  Rates are expressed as fraction-per-second and scaled to
elapsed time.

    cognitive_energy: recovers at COGNITIVE_RECOVERY_RATE/s
                      bonus recovery when no LLM call in last 60s
    social_energy:    recovers at SOCIAL_RECOVERY_RATE/s
                      bonus recovery during dream cycle / no-interaction
    attention:        recovers proportional to goals closed since last tick
                      does NOT recover on a pure timer — it recovers by
                      closing things

Meta-system costs
──────────────────
The audit engine, immune system, and dream synthesis all consume
cognitive_energy when they fire LLM calls.  record_llm_call() is
called by those modules.

Depletion events
─────────────────
    record_llm_call(tokens)      → cognitive_energy -= f(tokens)
    record_interaction(impact)   → social_energy    -= impact * SOCIAL_COST
    record_goal_opened()         → attention        -= GOAL_ATTENTION_COST
    record_goal_closed()         → attention        += GOAL_ATTENTION_COST
    tick(elapsed_secs)           → apply time-based recovery to all pools
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Recovery rates (fraction of pool per second) ──────────────────────────────
# Bug fix (v59): tick() only ever runs from the slow cycle, which fires
# every SLOW_CYCLE_SECONDS=120s (core/internal_loop.py). With the old
# 60s idle threshold, `now - self._last_llm_time > 60` was true on
# essentially every single tick — a 120s gap always clears a 60s bar —
# so COGNITIVE_IDLE_BONUS was not a bonus for genuine idleness, it was
# baseline behavior. Combined with the base rate that made cognitive_energy
# recover ~17%/tick (full refill in ~12 real minutes) regardless of how
# much the person was actually chatting, which is why the dashboard's Ch4
# process probabilities were pinned at 1.000 — the pools never stayed
# depleted long enough between refreshes to show anything else. Raising
# the threshold to comfortably exceed one slow-cycle interval means the
# bonus now only applies across a genuinely quiet stretch (skipped/merged
# cycles, dream periods), not routine conversational gaps.
COGNITIVE_RECOVERY_RATE = 0.0008   # full recovery from 0 in ~1250s (~21 min)
COGNITIVE_IDLE_BONUS    = 0.0006   # extra recovery when no LLM call in 5 min
COGNITIVE_IDLE_THRESHOLD_SECS = 300
SOCIAL_RECOVERY_RATE    = 0.0005   # full recovery from 0 in ~2000s (~33 min)
SOCIAL_DREAM_BONUS      = 0.0010   # extra recovery during dream / no-user period

# ── Depletion costs ───────────────────────────────────────────────────────────
LLM_COST_PER_TOKEN    = 0.00025   # 400 token call costs 0.10 cognitive_energy
SOCIAL_COST_PER_UNIT  = 0.18      # one interaction at impact=1.0 costs 0.18
GOAL_ATTENTION_COST   = 0.04      # one active goal costs 0.04 attention
META_SYSTEM_MULTIPLIER = 2.0      # audit/immune LLM calls cost double

# ── Suppression thresholds (read by CognitiveBehaviorGate) ────────────────────
COGNITIVE_LOW      = 0.30   # below → no new experiments, shallower reflection
COGNITIVE_CRITICAL = 0.15   # below → consolidation only, no curiosity generation
SOCIAL_LOW         = 0.25   # below → outreach blocked
ATTENTION_LOW      = 0.40   # below → new questions rejected, curiosity suppressed
ATTENTION_CRITICAL = 0.20   # below → only existing goals processed

# ── Persistence ───────────────────────────────────────────────────────────────
SAVE_PATH = "data/persona/cognitive_resources.json"


@dataclass
class ResourceSnapshot:
    timestamp:        float
    cognitive_energy: float
    social_energy:    float
    attention:        float


class CognitiveResourceEconomy:
    """
    Tracks three finite cognitive resource pools and applies time-based
    recovery.  All values are 0.0 (empty) to 1.0 (full).

    Usage:
        economy = CognitiveResourceEconomy()
        economy.tick(elapsed_secs)          # call every slow cycle
        economy.record_llm_call(tokens=400)
        economy.record_interaction(impact=0.7)
        economy.record_goal_opened()
        economy.record_goal_closed()

    Read by CognitiveBehaviorGate:
        economy.cognitive_energy
        economy.social_energy
        economy.attention
    """

    def __init__(self, path: str = SAVE_PATH) -> None:
        self._path             = Path(path)
        self._lock             = threading.Lock()
        self.cognitive_energy  = 1.0
        self.social_energy     = 1.0
        self.attention         = 1.0
        self._last_llm_time    = 0.0
        self._last_tick_time   = time.time()
        self._in_dream_mode    = False
        self._history: list    = []
        self._load()
        logger.info(
            f"[ResourceEconomy] Init — "
            f"cog={self.cognitive_energy:.2f} "
            f"soc={self.social_energy:.2f} "
            f"att={self.attention:.2f}"
        )

    # ── Tick — call every slow cycle ──────────────────────────────────────────

    def tick(self, elapsed_secs: Optional[float] = None) -> None:
        """Apply time-based recovery.  Called from InternalThoughtLoop."""
        now = time.time()
        if elapsed_secs is None:
            elapsed_secs = now - self._last_tick_time
        self._last_tick_time = now

        with self._lock:
            # cognitive_energy recovery
            cog_rate = COGNITIVE_RECOVERY_RATE
            # v57: CrossLayerFeedback writes _recovery_multiplier based on
            # TemporalProjection's planned path intensity. Intensive path = build
            # reserves now. Emergent path = slow recovery (low cost anyway).
            cog_rate *= getattr(self, '_recovery_multiplier', 1.00)
            if now - self._last_llm_time > COGNITIVE_IDLE_THRESHOLD_SECS:
                cog_rate += COGNITIVE_IDLE_BONUS
            self.cognitive_energy = min(
                1.0, self.cognitive_energy + cog_rate * elapsed_secs
            )

            # social_energy recovery
            soc_rate = SOCIAL_RECOVERY_RATE
            if self._in_dream_mode:
                soc_rate += SOCIAL_DREAM_BONUS
            self.social_energy = min(
                1.0, self.social_energy + soc_rate * elapsed_secs
            )

            # attention does NOT recover on timer — only when goals close
            # (handled by record_goal_closed)
            # but floor it from going negative via float drift
            self.attention = max(0.0, min(1.0, self.attention))

        self._snapshot()
        self._save()

    # ── Depletion events ──────────────────────────────────────────────────────

    def record_llm_call(
        self,
        tokens:        int   = 400,   # generated tokens
        prompt_tokens: int   = 0,     # input prompt tokens
        complexity:    float = 0.5,   # 0=trivial, 1=maximum reasoning load
        meta:          bool  = False,  # True for audit/immune background calls
    ) -> None:
        """
        Record cognitive cost of an LLM call.

        Cost model (three-factor, not token-count-only):
            generated_tokens × 0.30   — output effort
            prompt_tokens    × 0.20   — context processing
            complexity       × 0.50   — reasoning depth

        A short answer to a hard question costs more than a long answer
        to a trivial one.  Complexity proxy: caller passes it based on
        input characteristics (prompt length, retrieval count, depth).

        meta=True doubles the cost — background cognitive calls are
        expensive precisely because they interrupt foreground processing.
        """
        cost = (
            (tokens        * LLM_COST_PER_TOKEN * 0.30 / 0.30)  # generated
            + (prompt_tokens * LLM_COST_PER_TOKEN * 0.20 / 0.30)  # context
            + (complexity    * 0.15)                               # reasoning depth
        )
        if meta:
            cost *= META_SYSTEM_MULTIPLIER
        cost = max(0.01, min(0.40, cost))  # floor and ceiling per call
        with self._lock:
            self.cognitive_energy = max(0.0, self.cognitive_energy - cost)
            self._last_llm_time   = time.time()
        logger.debug(
            f"[ResourceEconomy] LLM cost={cost:.3f} "
            f"(tok={tokens} prompt={prompt_tokens} cx={complexity:.2f} meta={meta}) "
            f"→ cog={self.cognitive_energy:.2f}"
        )

    def record_interaction(self, impact: float = 0.5) -> None:
        """Call after each human interaction."""
        cost = impact * SOCIAL_COST_PER_UNIT
        with self._lock:
            self.social_energy    = max(0.0, self.social_energy    - cost)
            self.cognitive_energy = max(0.0, self.cognitive_energy - cost * 0.3)
        logger.debug(f"[ResourceEconomy] Interaction cost={cost:.3f} → soc={self.social_energy:.2f}")

    def record_goal_opened(self, count: int = 1) -> None:
        """Call when a new goal/question/aspiration is registered."""
        with self._lock:
            self.attention = max(0.0, self.attention - GOAL_ATTENTION_COST * count)

    def record_goal_closed(self, count: int = 1) -> None:
        """Call when a goal/question is resolved or archived."""
        with self._lock:
            self.attention = min(1.0, self.attention + GOAL_ATTENTION_COST * count)

    def set_dream_mode(self, active: bool) -> None:
        """Signal that no human interactions are occurring (dream/sleep period)."""
        self._in_dream_mode = active

    # ── Snapshot / status ─────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "cognitive_energy": round(self.cognitive_energy, 3),
            "social_energy":    round(self.social_energy,    3),
            "attention":        round(self.attention,        3),
            "in_dream_mode":    self._in_dream_mode,
            "pressure": {
                "cognitive_low":      self.cognitive_energy < COGNITIVE_LOW,
                "cognitive_critical": self.cognitive_energy < COGNITIVE_CRITICAL,
                "social_low":         self.social_energy    < SOCIAL_LOW,
                "attention_low":      self.attention        < ATTENTION_LOW,
                "attention_critical": self.attention        < ATTENTION_CRITICAL,
            },
        }

    def prompt_fragment(self) -> str:
        """Single-line resource state for prompt injection via PromptContextBudget."""
        levels = []
        if self.cognitive_energy < COGNITIVE_LOW:
            levels.append(f"cognitive energy low ({self.cognitive_energy:.0%})")
        if self.social_energy < SOCIAL_LOW:
            levels.append(f"social energy low ({self.social_energy:.0%})")
        if self.attention < ATTENTION_LOW:
            levels.append(f"attention constrained ({self.attention:.0%})")
        if not levels:
            return ""
        return "[Resource state] " + "; ".join(levels)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _snapshot(self) -> None:
        snap = {
            "ts": time.time(),
            "cognitive_energy": round(self.cognitive_energy, 4),
            "social_energy":    round(self.social_energy,    4),
            "attention":        round(self.attention,        4),
        }
        self._history.append(snap)
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "cognitive_energy": self.cognitive_energy,
                    "social_energy":    self.social_energy,
                    "attention":        self.attention,
                    "last_tick":        self._last_tick_time,
                    "history":          self._history[-50:],
                    "_meta":            {"version": "v50", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[ResourceEconomy] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self.cognitive_energy = float(data.get("cognitive_energy", 1.0))
                self.social_energy    = float(data.get("social_energy",    1.0))
                self.attention        = float(data.get("attention",        1.0))
                self._history         = data.get("history", [])
            # Bug fix (v59): this granted up to a full hour of instant,
            # unconditional recovery on every restart based purely on wall-
            # clock gap since the last save — with COGNITIVE_RECOVERY_RATE
            # =0.0008/s, any restart gap over ~21 minutes fully saturated
            # cognitive_energy back to 1.0 on boot regardless of real
            # depletion. In a workflow that restarts frequently between
            # patches, this could erase an entire batch's worth of
            # accumulated interaction cost in one line, right before the
            # next dashboard read — indistinguishable from "recovery never
            # depletes" even though depletion IS being applied correctly
            # during the run. Capped much lower (10 min) so a normal dev
            # restart gap doesn't grant implausible free recovery, while a
            # genuinely long-idle restart (e.g. overnight) still recovers
            # something reasonable rather than nothing.
            _before = (self.cognitive_energy, self.social_energy)
            elapsed = time.time() - float(data.get("last_tick", time.time()))
            if elapsed > 0:
                self.tick(elapsed_secs=min(elapsed, 600))
            if elapsed > 60 and (
                self.cognitive_energy - _before[0] > 0.05
                or self.social_energy - _before[1] > 0.05
            ):
                logger.info(
                    f"[ResourceEconomy] Restart catch-up recovery: "
                    f"{elapsed:.0f}s offline → "
                    f"cog {_before[0]:.2f}→{self.cognitive_energy:.2f}, "
                    f"soc {_before[1]:.2f}→{self.social_energy:.2f}"
                )
        except Exception as e:
            logger.warning(f"[ResourceEconomy] Load failed (defaults): {e}")
