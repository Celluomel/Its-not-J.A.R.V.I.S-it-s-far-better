"""
DominantThoughtSelector (DTS)
==============================
At every cycle, selects ONE dominant ThoughtThread to drive behavior.

This solves the core pathology: "multiple weak signals competing, no dominant
theme, no commitment". The organism can have many thoughts, but only ONE is
acted upon per cycle. All others are secondary or deferred.

Scoring formula (tunable weights):
    score = (
        w_goal       * goal_priority
      + w_pressure   * total_pressure
      + w_curiosity  * thread.curiosity
      + w_dissonance * thread.dissonance
      - w_energy     * thread.energy_cost / available_energy
      - w_safety     * thread.safety_risk
    )

The selection result is consumed by:
  1. CognitivePreProcessor — injects dominant thread into system prompt
  2. InternalThoughtLoop   — advances dominant thread first
  3. ActivitySelector      — uses dominant goal to pick activity type
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Scoring weights ───────────────────────────────────────────────────────────
W_GOAL       = 0.30
W_PRESSURE   = 0.25
W_CURIOSITY  = 0.20
W_DISSONANCE = 0.15   # dissonance demands resolution attention
W_ENERGY     = 0.05   # penalise expensive threads
W_SAFETY     = 0.10   # penalise risky threads (safety_risk > 0)

# Threads below this score are never dominant
MIN_SCORE_FOR_DOMINANCE = 0.15

# Max secondary threads to include in result
MAX_SECONDARY = 3


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class SelectorContext:
    threads:            List[Any]          # ThoughtThread list
    metrics:            Dict[str, float]   = field(default_factory=dict)
    pressures:          Dict[str, float]   = field(default_factory=dict)
    curiosity_level:    float              = 0.30
    energy:             float              = 80.0
    safety_constraints: Dict[str, Any]    = field(default_factory=dict)
    identity_state:     Dict[str, Any]    = field(default_factory=dict)


@dataclass
class SelectionResult:
    dominant:   Optional[Any]           # ThoughtThread | None
    secondary:  List[Any]               = field(default_factory=list)
    scores:     Dict[str, float]        = field(default_factory=dict)
    rationale:  str                     = ""

    def to_prompt_block(self) -> str:
        """Compact string for injection into the system prompt."""
        if self.dominant is None:
            return ""
        lines = [
            f"[DOMINANT THOUGHT] {self.dominant.to_prompt_line()}",
            f"  Rationale: {self.rationale}",
        ]
        if self.secondary:
            tops = [t.topic for t in self.secondary[:2]]
            lines.append(f"  Background concerns: {', '.join(tops)}")
        return "\n".join(lines)


# ── Selector ──────────────────────────────────────────────────────────────────

class DominantThoughtSelector:
    """
    Filters, scores, and selects the dominant thought thread each cycle.
    Stateless — can be called freely from any thread.
    """

    def filter_threads(self, ctx: SelectorContext) -> List[Any]:
        """
        Eliminate threads that are:
        - too energy-costly given current level
        - safety-risky
        - not in ACTIVE or ENGAGED status
        """
        from cognition.thought_thread_engine import ThoughtThreadStatus
        available = ctx.energy
        filtered = []
        for t in ctx.threads:
            # Status filter
            if t.status not in (ThoughtThreadStatus.ACTIVE, ThoughtThreadStatus.ENGAGED):
                continue
            # Energy filter (allow low-cost threads even at low energy)
            if t.energy_cost > available * 1.5:
                continue
            # Safety filter
            if t.safety_risk > 0.8:
                continue
            filtered.append(t)
        return filtered

    def score_thread(self, thread: Any, ctx: SelectorContext) -> float:
        """
        Compute a [0, 1] priority score for a thread.
        Higher = more likely to be selected as dominant.
        """
        energy = max(1.0, ctx.energy)  # avoid div/zero

        # Goal priority: from thread's current_state or default
        goal_p = thread.current_state.get("goal_priority", 0.5)

        # Pressure alignment: sum of pressures that match thread source
        pressure_total = sum(ctx.pressures.values()) / max(1, len(ctx.pressures))

        # Dissonance urgency (meta-threads score high here)
        dissonance_urgency = thread.dissonance

        # Energy penalty
        energy_penalty = thread.energy_cost / energy

        # Safety penalty
        safety_penalty = thread.safety_risk

        score = (
            W_GOAL       * goal_p
          + W_PRESSURE   * pressure_total
          + W_CURIOSITY  * thread.curiosity
          + W_DISSONANCE * dissonance_urgency
          - W_ENERGY     * energy_penalty
          - W_SAFETY     * safety_penalty
        )
        return max(0.0, min(1.0, score))

    def select(self, ctx: SelectorContext) -> SelectionResult:
        """
        Full selection cycle:
        1. Filter ineligible threads
        2. Score each remaining thread
        3. Select 1 dominant + up to MAX_SECONDARY secondaries
        4. Mark dominant as ENGAGED, demote others to ACTIVE
        5. Produce rationale string for meta-cognition / logs
        """
        from cognition.thought_thread_engine import ThoughtThreadStatus

        eligible = self.filter_threads(ctx)

        if not eligible:
            return SelectionResult(
                dominant  = None,
                rationale = "No eligible threads — all deferred or too costly.",
            )

        # Score all threads
        scored: List[tuple[float, Any]] = [
            (self.score_thread(t, ctx), t)
            for t in eligible
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, dominant = scored[0]
        if best_score < MIN_SCORE_FOR_DOMINANCE:
            return SelectionResult(
                dominant  = None,
                rationale = f"Best score {best_score:.2f} below threshold {MIN_SCORE_FOR_DOMINANCE}.",
            )

        # Update thread status
        dominant.status = ThoughtThreadStatus.ENGAGED
        secondary = []
        for sc, t in scored[1: MAX_SECONDARY + 1]:
            t.status = ThoughtThreadStatus.ACTIVE
            secondary.append(t)

        # Build rationale
        rationale = (
            f"Selected '{dominant.topic}' "
            f"(score={best_score:.2f}, curiosity={dominant.curiosity:.2f}, "
            f"pressure={sum(ctx.pressures.values()):.2f}, "
            f"energy={ctx.energy:.0f}%)"
        )

        scores_dict = {sc_t[1].id: round(sc_t[0], 3) for sc_t in scored}

        logger.debug(f"[DTS] Dominant: {dominant.topic!r} score={best_score:.2f}")

        return SelectionResult(
            dominant  = dominant,
            secondary = secondary,
            scores    = scores_dict,
            rationale = rationale,
        )
