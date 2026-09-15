"""
cognition/prospective_deliberation.py

Priority 1 + 2 from the cognitive synthesis audit (2026-07), implemented
together because they're the same piece of work in practice:

  Priority 1 — make prediction prospective, not evaluative. Today:
    classify action type → generate ONE response → THEN predict what
    already happened. This module inserts prediction BEFORE generation.

  Priority 2 — stop discarding WorkspaceCompetition's runners_up. The
  cheapest real step toward genuine multi-candidate deliberation is to
  evaluate more than one candidate framing before committing to one.

Why this ISN'T "generate 3 full responses and pick the best" — that would
triple the LLM cost per turn, impractical for a local 7B model in
real-time chat. Instead, it reuses PCM/WSDM's existing k-NN prediction
machinery (already fast, already built, already correct) to compare ALL
7 action-type framings against the CURRENT live state, before the single
LLM call happens. If a framing other than the one the keyword classifier
picked scores meaningfully better, that's surfaced as a prompt advisory —
steering the ONE generation, rather than filtering after N generations.

This is deliberately the SAME joint_score formula, weights, and
calibration lookup pattern as cognition/counterfactual_simulator.py's
retroactive analysis — consistency between "what we predicted before
choosing" and "how we score choices after the fact" matters; a system
that judges its past choices by one standard and its future choices by
another isn't actually deliberating, it's rationalizing.

_predict_branch()-equivalent logic here is intentionally NOT imported
from CounterfactualSimulator, because that class's lifecycle is tied to
slow_cycle > 15 lazy-init (needs 15+ PCM records before it exists at
all). Prospective deliberation should work as soon as PCM/WSDM have
enough data individually — which can be earlier — so this module has no
dependency on CounterfactualSimulator's instantiation state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALL_ACTION_TYPES = [
    "philosophical", "technical", "emotional", "creative",
    "social", "analytical", "deep_reasoning",
]

PCM_ENERGY_WEIGHT       = 0.55   # same weights as counterfactual_simulator.py —
WSDM_ENGAGEMENT_WEIGHT  = 0.45   # deliberately consistent, see module docstring

MARGIN_THRESHOLD = 0.02   # minimum joint_score gap to surface an advisory —
                          # below this, the difference is noise, not signal
MIN_CONFIDENCE_TO_ACT = 0.3   # don't advise a framing shift on a near-guess


@dataclass
class FramingCandidate:
    action_type:     str
    pcm_energy_delta: Optional[float]
    pcm_confidence:   Optional[float]
    wsdm_engagement:  Optional[float]
    wsdm_confidence:  Optional[float]
    joint_score:      float
    is_default:       bool = False


@dataclass
class DeliberationResult:
    default_action:   str
    candidates:       List[FramingCandidate]
    best_action:      str
    best_score:       float
    default_score:    float
    margin:           float
    advisory:         str   # empty string if no meaningful advisory


def _score_candidate(
    pcm: Any,
    wsdm: Optional[Any],
    calibration_engine: Optional[Any],
    action_type: str,
    state: List[float],
    ctx: Optional[List[float]],
    is_default: bool,
) -> FramingCandidate:
    """Predict PCM/WSDM outcomes for one candidate action-type framing,
    using the CURRENT live state — not a historical record. This is the
    prospective counterpart to CounterfactualSimulator._predict_branch(),
    using identical scoring so retroactive and prospective views agree."""

    pcm_energy = pcm_conf_raw = pcm_conf_cal = None
    try:
        pcm_pred = pcm._predict(action_type, state)
        if pcm_pred is not None:
            pcm_energy   = round(float(pcm_pred.energy_delta), 4)
            pcm_conf_raw = round(float(pcm_pred.confidence), 4)
            if calibration_engine:
                cal_val, is_cal = calibration_engine.calibrated_confidence(
                    "pcm", pcm_conf_raw
                )
                pcm_conf_cal = round(cal_val, 4) if is_cal else None
    except Exception as e:
        logger.debug(f"[ProspectiveDeliberation] PCM predict error ({action_type}): {e}")

    wsdm_engage = wsdm_conf_raw = wsdm_conf_cal = None
    try:
        wsdm_ready = (wsdm is not None and ctx is not None and
                      len(getattr(wsdm, "_records", [])) >= 6)
        if wsdm_ready:
            wsdm_pred = wsdm._predict(action_type, ctx)
            if wsdm_pred is not None:
                wsdm_engage   = round(
                    float(wsdm_pred.world_deltas.get("engagement_signal", 0.0)), 4
                )
                wsdm_conf_raw = round(float(wsdm_pred.confidence), 4)
                if calibration_engine:
                    cal_val, is_cal = calibration_engine.calibrated_confidence(
                        "wsdm", wsdm_conf_raw
                    )
                    wsdm_conf_cal = round(cal_val, 4) if is_cal else None
    except Exception as e:
        logger.debug(f"[ProspectiveDeliberation] WSDM predict error ({action_type}): {e}")

    eff_pcm  = pcm_conf_cal  if pcm_conf_cal  is not None else (pcm_conf_raw  or 0.5)
    eff_wsdm = wsdm_conf_cal if wsdm_conf_cal is not None else (wsdm_conf_raw or 0.5)
    joint_score = round(
        (pcm_energy  or 0.0) * eff_pcm  * PCM_ENERGY_WEIGHT +
        (wsdm_engage or 0.0) * eff_wsdm * WSDM_ENGAGEMENT_WEIGHT,
        5,
    )

    return FramingCandidate(
        action_type       = action_type,
        pcm_energy_delta  = pcm_energy,
        pcm_confidence    = pcm_conf_raw,
        wsdm_engagement   = wsdm_engage,
        wsdm_confidence   = wsdm_conf_raw,
        joint_score       = joint_score,
        is_default        = is_default,
    )


def deliberate(
    pcm: Any,
    wsdm: Optional[Any],
    default_action_type: str,
    current_state: List[float],
    user_id: str = "default",
) -> Optional[DeliberationResult]:
    """
    Evaluate all 7 action-type framings against the CURRENT state, before
    a response is generated. Returns None if there isn't enough data to
    say anything meaningful (graceful degradation — same pattern PCM/WSDM
    already use when MIN_SIMILAR isn't met).

    Called from cognitive_organism.py's response path, BEFORE
    predict_and_advise() and BEFORE the LLM call — this is what makes it
    prospective rather than evaluative. predict_and_advise() still runs
    afterward as before, answering a different question ("is the chosen
    action risky") from this one ("was there a better choice available").
    """
    try:
        calibration_engine = getattr(pcm, "_calibration_engine", None)

        ctx = None
        if wsdm is not None:
            try:
                ctx = wsdm._read_context(user_id)
            except Exception:
                ctx = None

        candidates = [
            _score_candidate(
                pcm, wsdm, calibration_engine, action_type,
                current_state, ctx, is_default=(action_type == default_action_type),
            )
            for action_type in ALL_ACTION_TYPES
        ]

        # If every candidate has zero signal (no PCM data at all yet for
        # any action type), there's nothing to deliberate over.
        if all(c.joint_score == 0.0 for c in candidates):
            return None

        ranked = sorted(candidates, key=lambda c: c.joint_score, reverse=True)
        best = ranked[0]
        default = next(c for c in candidates if c.is_default)
        margin = round(best.joint_score - default.joint_score, 5)

        advisory = ""
        if (best.action_type != default.action_type
                and margin >= MARGIN_THRESHOLD
                and (best.pcm_confidence or 0) >= MIN_CONFIDENCE_TO_ACT):
            advisory = (
                f"Before responding: predicted outcomes favor a "
                f"{best.action_type.replace('_',' ')} framing over the "
                f"instinctive {default.action_type.replace('_',' ')} one "
                f"for this message (Δscore={margin:+.3f}) — worth leaning "
                f"that direction if it fits naturally."
            )

        return DeliberationResult(
            default_action = default.action_type,
            candidates     = candidates,
            best_action    = best.action_type,
            best_score     = best.joint_score,
            default_score  = default.joint_score,
            margin         = margin,
            advisory       = advisory,
        )

    except Exception as e:
        logger.debug(f"[ProspectiveDeliberation] deliberate error: {e}")
        return None
