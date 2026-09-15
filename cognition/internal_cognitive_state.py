"""
InternalCognitiveState — epistemic pressure as a first-class cognitive
state, not a reactive control signal
=========================================================================
Implements the mission from the implementation prompt: move internal
cognitive condition from a flat, undifferentiated modulator into an
input that specifically favors GOALS ABOUT the organism's own cognitive
state, in continuous proportion to that state — not a threshold
controller (`if pressure > 0.8: choose(X)`), and not a duplicate
cognitive architecture.

Traced first (implementation prompt's own Step 1), before writing this:

  pressure source   -> PressureSystem.reservoirs['epistemic'].level
                        (cognition/pressure_system.py, real, already
                        boosted on Flux disagreement — v96)
  pressure storage  -> same object, persisted to disk
  goal creation     -> GoalEngine.derive_motivations() ALREADY creates a
                        real "resolve_uncertainty" goal when epistemic or
                        uncertainty pressure crosses PRESSURE_MOTIVATION_
                        THRESHOLDS (goal_engine.py — this predates this
                        module; Phase 5.1 already verified this chain
                        works end-to-end). This module does NOT need to
                        spawn new self-regulatory goals — real production
                        machinery already does, for "resolve_uncertainty"
                        specifically. Reused, not duplicated.
  goal scoring      -> workspace_competition.py::compete(), GOALS branch.
                        Existing pressure signal there
                        (`pressure.get('goal_pressure', 0.5) * 0.3`) is
                        exactly the "reactive control signal" the mission
                        describes: a FLAT boost applied to EVERY goal
                        equally, regardless of whether that goal has
                        anything to do with the pressure. This module's
                        job is the missing piece: a SEPARATE, bounded,
                        additive term that only engages for goals whose
                        own content is about cognitive self-regulation —
                        continuous, not thresholded, and ablatable to
                        exactly the previous behavior when disabled.
  arbitration/       -> compete()'s output feeds recursive_deliberation.
  deliberation/         deliberate() -> WorkspaceState (already the real,
  workspace              verified live path — v89/v90's causal proof
                        applies unchanged; this module changes what goes
                        INTO that pipeline, not the pipeline itself).
  prediction source -> PredictiveMind (predictive_mind.py) predicts USER
                        intent/emotion, not the organism's OWN future
                        internal state — genuinely a different question.
                        No existing meta-prediction-of-self mechanism
                        found. Smallest new state added here: a simple
                        trend-extrapolation self-model of P(t) itself
                        (NOT a second copy of PredictiveMind's intent
                        classifier — a one-number self-prediction).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Ablation / ordering knobs (Sections 11, 13-19 of the implementation
# prompt) — every one of these can be set to neutral values to fall back
# exactly to pre-existing behavior; that IS the OFF condition for the
# required ablation, not a separate code path.
INTERNAL_STATE_PROJECTION_ENABLED = True
SELF_STATE_GOAL_WEIGHT = 0.25   # bounded coefficient on the alignment term
MAX_INTERNAL_BIAS = 0.20        # hard clamp — never exceeds this contribution
DELTA_SCALE = 0.10              # tanh scale for pressure_delta normalization
EWMA_TREND_ALPHA = 0.3          # self-prediction trend-extrapolation weight

# Section 8 — functional self-regulatory goal names. Matched against a
# goal candidate's own name/topic text; NOT auto-prioritized (Section 8's
# own requirement) — matching only makes a goal ELIGIBLE for the bounded
# alignment term, which still has to win the same competition as any
# other goal. "resolve_uncertainty" is the one with real, already-
# verified production goal-creation machinery behind it (goal_engine.py);
# the rest are included for generality if/when equivalent goals exist.
SELF_REGULATORY_GOAL_NAMES = frozenset({
    "resolve_uncertainty", "verify_assumption", "resolve_dissonance",
    "resolve_epistemic_dissonance", "consolidate_context",
    "inspect_prediction_failure", "reduce_uncertainty",
    "review_recent_decision", "explore_missing_information",
    "reconcile_conflicting_hypotheses", "reassess_goal_priority",
})


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def is_self_regulatory_goal(name: str, topic: str) -> bool:
    text = f"{name or ''} {topic or ''}".lower().replace(" ", "_")
    return any(kw in text for kw in SELF_REGULATORY_GOAL_NAMES)


@dataclass
class InternalStateSnapshot:
    pressure: float           # normalized [0,1]
    pressure_delta: float     # tanh-normalized [-1,1]
    meta_error: float         # normalized [0,1]
    raw_pressure: float
    raw_delta: float
    raw_meta_error: float
    timestamp: float = field(default_factory=time.time)


class InternalCognitiveState:
    """
    Owned by the organism (see get_internal_cognitive_state() below — one
    shared instance, same singleton pattern as UniversalConnector).
    update() is meant to be called once per cognitive cycle with the
    CURRENT epistemic pressure reading; it returns the normalized state
    vector for THIS cycle and records how wrong last cycle's self-
    prediction was (meta_error) — a real, causally-computed quantity,
    not descriptive text.
    """

    def __init__(self) -> None:
        self._P_hat: Optional[float] = None   # this cycle's prediction, made last cycle
        self._P_prev: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

    def update(self, p_current: float) -> InternalStateSnapshot:
        p_current = clamp(p_current, 0.0, 1.0)

        meta_error_raw = abs(p_current - self._P_hat) if self._P_hat is not None else 0.0
        delta_raw = (p_current - self._P_prev) if self._P_prev is not None else 0.0

        snapshot = InternalStateSnapshot(
            pressure=p_current,
            pressure_delta=math.tanh(delta_raw / DELTA_SCALE),
            meta_error=clamp(meta_error_raw, 0.0, 1.0),
            raw_pressure=p_current,
            raw_delta=delta_raw,
            raw_meta_error=meta_error_raw,
        )

        # Form the prediction for NEXT cycle now, using only what's known
        # as of THIS cycle (trend-extrapolation) — a minimal, bounded,
        # non-learning self-model per the implementation prompt's explicit
        # "do not train yet" instruction (Section 19): fixed weight, no
        # gradient, no online fitting.
        trend = delta_raw
        self._P_hat = clamp(p_current + EWMA_TREND_ALPHA * trend, 0.0, 1.0)
        self._P_prev = p_current

        self.history.append({
            "t": snapshot.timestamp, "pressure": p_current,
            "pressure_delta": snapshot.pressure_delta,
            "meta_error": snapshot.meta_error,
            "predicted_next": self._P_hat,
        })
        self.history = self.history[-1000:]
        return snapshot

    def internal_alignment(self, goal_name: str, goal_topic: str,
                            enabled: bool = True,
                            efficacy_multiplier: float = 1.0) -> float:
        """
        The bounded, continuous, ablatable additive term (Sections 6, 7,
        11). Zero for any goal that isn't self-regulatory in content —
        Section 7's explicit requirement: a goal becomes more attractive
        because its OWN representation is compatible with internal
        state, never via a hard-coded name check that special-cases one
        goal. Zero entirely when enabled=False (the OFF condition for
        ablation — exactly pre-existing behavior, not a different code
        path).

        efficacy_multiplier: the recursive crossing (epistemic_efficacy_
        model.py). Defaults to 1.0 (neutral, exactly today's behavior)
        when no efficacy model is wired in. When provided, it's the
        self-model's OWN learned effectiveness belief scaling its own
        future boost — the causal loop the analysis asked for: an
        updated self-model changes future scoring of the very goal it's
        about, not just a static per-content boost.
        """
        if not enabled or not INTERNAL_STATE_PROJECTION_ENABLED:
            return 0.0
        if not self.history:
            return 0.0
        if not is_self_regulatory_goal(goal_name, goal_topic):
            return 0.0

        last = self.history[-1]
        raw_alignment = (
            last["pressure"] * 0.5
            + abs(last["pressure_delta"]) * 0.25
            + last["meta_error"] * 0.25
        )
        biased = SELF_STATE_GOAL_WEIGHT * raw_alignment * efficacy_multiplier
        return clamp(biased, -MAX_INTERNAL_BIAS, MAX_INTERNAL_BIAS)

    def cognitive_self_reference_index(self, selected_goal_internal_bias: float,
                                        selected_goal_total_score: float) -> float:
        """
        Section 25 — CSR(t): fraction of the SELECTED goal's utility
        attributable to internal-state alignment. Deliberately named to
        avoid the "avoid semantic cheating" trap (Section 20/25) — never
        called a consciousness score.
        """
        if selected_goal_total_score <= 0:
            return 0.0
        return clamp(selected_goal_internal_bias / selected_goal_total_score, 0.0, 1.0)


_states: Dict[int, InternalCognitiveState] = {}


def get_internal_cognitive_state(organism: Any) -> InternalCognitiveState:
    """One shared instance per organism — same rationale as
    get_universal_connector(): every caller (goal scoring, replay
    harness, future narrative/self-model integrations) needs the SAME
    running history, not one each."""
    existing = getattr(organism, "_internal_cognitive_state", None)
    if existing is not None:
        return existing
    state = InternalCognitiveState()
    try:
        organism._internal_cognitive_state = state
    except Exception:
        pass
    return state
