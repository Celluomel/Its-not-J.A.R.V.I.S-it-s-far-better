"""
EpistemicEfficacyModel — the recursive crossing
====================================================
Per the reframed mission (an external analysis reviewing the v104
self-model flatline): don't "implement a strange loop" as an engineered
feedback circuit that looks self-referential. Instead, make PandoraBOX's own
cognitive activity become an object of cognition in a way that CAUSALLY
alters the activity that generated it. The concrete test given:

  Can the self-model predict something about PandoraBOX's own future
  behaviour, be wrong, observe the error, update itself, and
  subsequently behave differently?

v104's self-model update was purely reactive (affirm a belief on a
keyword match in generated text) — it never predicted anything
falsifiable, so there was nothing for it to be WRONG about. This module
is the missing piece, using the analysis's own worked example directly:

  self-model: "I am effective at resolving epistemic uncertainty"
       -> resolve_uncertainty is chosen
       -> self-model PREDICTS how much epistemic pressure should drop
       -> next cycle, the ACTUAL drop is observed
       -> meta-prediction error = |predicted - actual|
       -> small error -> affirm effectiveness; large error -> erode it
       -> effectiveness belief scales THIS SAME goal's future
          internal_alignment boost (workspace_competition.py) ->
          future arbitration genuinely changes -> new behaviour ->
          new observation -> next prediction

Ablatable exactly like InternalCognitiveState: efficacy_model=None
(default) reproduces prior behavior with zero contribution.

Deliberately NOT done here (avoiding "optimizing to look self-aware",
the analysis's explicit warning): no narrative text about this process
is generated or injected into any prompt. The belief update and the
scoring effect are the only outputs; nothing here writes a sentence
claiming self-awareness.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PREDICTION_SCALE = 0.30      # confidence 0.5 -> predicts a 0.15 drop, matching
                              # the analysis's own worked example exactly
GOOD_PREDICTION_THRESHOLD = 0.05   # meta-error below this counts as "right"
MIN_MULTIPLIER = 0.4         # bounds on how much effectiveness belief can
MAX_MULTIPLIER = 1.6         # scale the goal's own alignment boost
EFFECTIVENESS_BELIEF_NAME = "effective_at_resolving_uncertainty"


@dataclass
class PendingPrediction:
    predicted_drop: float
    pressure_before: float
    made_at: float


class EpistemicEfficacyModel:
    """
    Owned by the organism (see get_epistemic_efficacy_model() below —
    same singleton pattern as InternalCognitiveState/UniversalConnector).
    Tracks ONE pending falsifiable prediction at a time, tied specifically
    to resolve_uncertainty (the one self-regulatory goal with real,
    already-verified production goal-creation machinery — same scoping
    choice made in internal_cognitive_state.py).
    """

    def __init__(self, self_concept: Any) -> None:
        self._sc = self_concept
        self._pending: Optional[PendingPrediction] = None
        self.history: list[Dict[str, Any]] = []
        self._ensure_belief_seeded()

    def _ensure_belief_seeded(self) -> None:
        try:
            if EFFECTIVENESS_BELIEF_NAME not in self._sc._beliefs:
                from cognition.self_concept import SelfBelief
                self._sc._beliefs[EFFECTIVENESS_BELIEF_NAME] = SelfBelief(
                    name=EFFECTIVENESS_BELIEF_NAME,
                    statement="I am effective at resolving epistemic uncertainty",
                    confidence=0.5,
                )
        except Exception as e:
            logger.debug(f"[EpistemicEfficacyModel] belief seed failed (non-fatal): {e}")

    def _confidence(self) -> float:
        try:
            return self._sc._beliefs[EFFECTIVENESS_BELIEF_NAME].confidence
        except Exception:
            return 0.5

    def observe_if_pending(self, current_pressure: float) -> Optional[float]:
        """
        Call once per cycle, BEFORE this cycle's own prediction/scoring.
        If a prediction from a previous resolve_uncertainty win is
        pending, compare it against what actually happened and update
        the effectiveness belief. Returns the meta-prediction-error if an
        observation was made, else None.
        """
        if self._pending is None:
            return None
        actual_drop = self._pending.pressure_before - current_pressure
        meta_error = abs(self._pending.predicted_drop - actual_drop)

        try:
            if meta_error < GOOD_PREDICTION_THRESHOLD:
                self._sc.record_affirmation(EFFECTIVENESS_BELIEF_NAME)
            else:
                self._sc.record_violation(EFFECTIVENESS_BELIEF_NAME, context="meta-prediction error")
        except Exception as e:
            logger.debug(f"[EpistemicEfficacyModel] belief update failed (non-fatal): {e}")

        self.history.append({
            "t": time.time(), "predicted_drop": self._pending.predicted_drop,
            "actual_drop": actual_drop, "meta_error": meta_error,
            "confidence_after": self._confidence(),
        })
        self.history = self.history[-500:]
        self._pending = None
        return meta_error

    def record_prediction_if_winner(self, winner_name: str, winner_topic: str,
                                     current_pressure: float) -> None:
        """Call once per cycle, AFTER the competition winner is known. If
        resolve_uncertainty won, the self-model makes ITS OWN falsifiable
        prediction about how effective that choice will be — genuinely
        derived from the current effectiveness belief, not a fixed
        constant, so a self-model that has learned it's less effective
        predicts smaller drops going forward. Checks name/topic, not id —
        a real GoalEngine goal's id is never literally "resolve_uncertainty",
        only its name/topic is (matching the same check used elsewhere,
        e.g. internal_cognitive_state.is_self_regulatory_goal)."""
        text = f"{winner_name or ''} {winner_topic or ''}".lower().replace(" ", "_")
        if "resolve_uncertainty" not in text:
            return
        predicted_drop = self._confidence() * PREDICTION_SCALE
        self._pending = PendingPrediction(
            predicted_drop=predicted_drop, pressure_before=current_pressure,
            made_at=time.time(),
        )

    def effectiveness_multiplier(self) -> float:
        """
        Scales resolve_uncertainty's own internal_alignment boost
        (workspace_competition.py) — THE causal crossing: an updated
        self-model changes future scoring of the very goal it's about.
        confidence=0.5 (neutral, unlearned) -> multiplier=1.0 (no change
        from today's v102/v103 behavior). Bounded both directions.
        """
        conf = self._confidence()
        mult = 0.4 + conf * 1.2  # confidence 0.5->1.0, 0.05->0.46, 0.95->1.54
        return max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, mult))


_models: Dict[int, EpistemicEfficacyModel] = {}


def get_epistemic_efficacy_model(organism: Any) -> Optional[EpistemicEfficacyModel]:
    existing = getattr(organism, "_epistemic_efficacy_model", None)
    if existing is not None:
        return existing
    # self_concept lives on ai_system, not directly on organism — verified
    # attribute path (v91 used the same lookup: getattr(getattr(organism,
    # "ai_system", None), "self_concept", None)). Getting this wrong here
    # would make the model permanently no-op (sc always None) without
    # ever raising an error to reveal it — exactly the class of bug this
    # whole project keeps finding, caught before shipping this time by
    # checking the real attribute path instead of assuming one.
    sc = getattr(getattr(organism, "ai_system", None), "self_concept", None)
    if sc is None:
        sc = getattr(organism, "self_concept", None)  # fallback for test
                                                         # harnesses that
                                                         # attach it directly
    if sc is None:
        return None
    model = EpistemicEfficacyModel(sc)
    try:
        organism._epistemic_efficacy_model = model
    except Exception:
        pass
    return model
