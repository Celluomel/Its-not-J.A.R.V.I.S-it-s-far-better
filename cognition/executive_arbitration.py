"""
cognition/executive_arbitration.py

Phase 4.1 (Multiple Competing Futures) + 4.2 (Executive Arbitration).

WorkspaceCompetition.compete() already computes a full ranked candidate
list (result['competition']['all_candidates']) but only ever exposes the
raw-score winner downstream — confirmed by reading core/phase1_integration.py,
the only real caller: only `winner` is read, `all_candidates` is discarded
after being returned. This module closes that gap.

Design: take the top N candidates (winner + runners-up) that are goal-type
(the only type with a `topic` field prospective_deliberation can classify
an action framing for), predict each one's outcome using the SAME
prospective_deliberation machinery already wired into the live chat path,
and arbitrate: if a runner-up predicts a meaningfully better outcome than
the raw-score winner, override the selection.

This is deliberately NOT a new prediction model — it reuses
prospective_deliberation.deliberate()'s per-candidate scoring, applied to
workspace candidates instead of PCM action types directly. Two different
things get scored by the same underlying formula: prospective_deliberation
asks "which framing for this ONE decided-upon topic is best", this asks
"which of these N candidate topics should even BE the decided-upon one".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_CANDIDATES_TO_ARBITRATE = 3   # winner + top 2 runners-up
OVERRIDE_MARGIN_THRESHOLD   = 0.03   # utility gap needed to override raw
                                      # competition score — deliberately
                                      # higher than prospective_deliberation's
                                      # 0.02, since overriding WHICH topic
                                      # gets attention is a bigger move than
                                      # nudging HOW to frame one topic

# Phase 4.x: multi-factor executive utility weights. Replaces the old
# single-signal (predicted_score only) override check. Every term is
# computed from data that already exists elsewhere in the codebase — see
# _score_extra_factors() below for exactly where each one comes from and,
# where a subsystem doesn't yet expose a genuinely per-candidate signal
# (calibration), that's called out explicitly rather than faked.
UTILITY_WEIGHTS = {
    "activation":      0.25,   # workspace_competition's own raw score (normalized)
    "predicted":        0.35,   # prospective_deliberation's default_score (normalized)
    "counterfactual":   0.15,   # match in WorkspaceState.predicted_futures, if present
    "identity":         0.10,   # lexical overlap with WorkspaceState.narrative_state
    "goal_alignment":   0.10,   # candidate topic present among WorkspaceState.goals
    "calibration":      0.05,   # Phase 5.2: real per-domain skill confidence
                                 # (skill_registry.all_domain_confidence(), via
                                 # WorkspaceState.context) when the candidate's
                                 # topic matches a tracked skill; falls back to
                                 # WorkspaceState.confidence (a workspace-level
                                 # scalar) for untracked domains
}


def _emotional_modulation(valence: float, arousal: float) -> Dict[str, float]:
    """
    v78: emotion gets real causal power over WHICH candidate wins, not just
    narration in the prompt. Bounded additive nudges to UTILITY_WEIGHTS —
    max ~0.08 shift on any one weight — so mood can tilt the decision
    without ever being able to dominate it outright.

    valence in [-1, 1] (from EmotionalStateManager.get_overall_valence_arousal)
    arousal in [0, 1]
    """
    mod = {k: 0.0 for k in UTILITY_WEIGHTS}

    if valence < 0:
        # Anxious/frustrated: favor certainty and staying on known-good
        # ground over exploring the unknown.
        safety_push = min(0.08, -valence * 0.08)
        mod["calibration"]    += safety_push * 0.6
        mod["goal_alignment"] += safety_push * 0.4
        mod["counterfactual"] -= safety_push
    else:
        # Content/curious: favor exploring what might happen over what's
        # already certain.
        explore_push = min(0.08, valence * 0.08)
        mod["counterfactual"] += explore_push * 0.6
        mod["predicted"]      += explore_push * 0.4
        mod["calibration"]    -= explore_push

    # High arousal: more reactive to raw activation (act on what's grabbing
    # attention right now); low arousal: more deliberative (weigh predicted
    # outcome more heavily before committing).
    arousal_push = (arousal - 0.5) * 0.06   # roughly -0.03 .. +0.03
    mod["activation"] += arousal_push
    mod["predicted"]  -= arousal_push

    return mod


def _get_base_weights() -> Dict[str, float]:
    """
    Phase 5.4: UTILITY_WEIGHTS plus a bounded, learned, persisted delta
    from ArbitrationLearningTracker (whether past decisions weighted
    toward each term actually turned out well). Lazy import to avoid a
    module-load-time circular dependency — arbitration_learning.py itself
    imports UTILITY_WEIGHTS from here, also lazily. Falls back to the
    static UTILITY_WEIGHTS unchanged if learning isn't available yet.
    """
    try:
        from cognition.arbitration_learning import get_learned_weights
        learned = get_learned_weights()
        if learned:
            return learned
    except Exception:
        pass
    return UTILITY_WEIGHTS


def _apply_emotional_modulation(
    base_weights: Dict[str, float], valence: Optional[float], arousal: Optional[float]
) -> Dict[str, float]:
    """Return a modulated, renormalized copy of base_weights. No-op if
    valence/arousal weren't supplied (keeps every existing caller working
    unchanged)."""
    if valence is None or arousal is None:
        return base_weights
    mod = _emotional_modulation(valence, arousal)
    shifted = {k: max(0.0, base_weights[k] + mod.get(k, 0.0)) for k in base_weights}
    total = sum(shifted.values()) or 1.0
    return {k: v / total for k, v in shifted.items()}


def _tokens(text: str) -> set:
    raw = str(text or "").replace("_", " ").split()
    return {t.strip('.,!?;:\'"()[]').lower() for t in raw if len(t) >= 3}


def _score_extra_factors(cand: Dict[str, Any], workspace_state: Optional[Any]) -> Dict[str, float]:
    """
    Compute the non-prediction utility terms for one candidate from the
    shared WorkspaceState, if one was supplied. Every lookup degrades to
    0.0 (neutral) when the relevant field is empty — this function never
    raises, so arbitrate() stays safe to call with workspace_state=None.
    """
    if workspace_state is None:
        return {"counterfactual": 0.0, "identity": 0.0, "goal_alignment": 0.0, "calibration": 0.0}

    topic_tokens = _tokens(cand.get("topic") or cand.get("name") or cand.get("label"))

    # Counterfactual: does this candidate's topic appear among futures the
    # counterfactual simulator already flagged this cycle?
    cf_score = 0.0
    for fut in getattr(workspace_state, "predicted_futures", []) or []:
        if not isinstance(fut, dict):
            continue
        overlap = topic_tokens & _tokens(fut.get("topic") or fut.get("label"))
        if overlap:
            cf_score = max(cf_score, float(fut.get("joint_score", fut.get("delta_score", 0.5)) or 0.0))

    # Identity consistency: lexical overlap with the current narrative state
    # string (narrative_identity's prompt_fragment/self_description output).
    narrative = getattr(workspace_state, "narrative_state", None)
    identity_score = 0.0
    if narrative:
        narr_tokens = _tokens(narrative)
        union = topic_tokens | narr_tokens
        identity_score = len(topic_tokens & narr_tokens) / len(union) if union else 0.0

    # Goal alignment: is this candidate's topic already an active goal the
    # workspace knows about, and how does it rank there?
    goal_score = 0.0
    for g in getattr(workspace_state, "goals", []) or []:
        if not isinstance(g, dict):
            continue
        g_topic_tokens = _tokens(g.get("topic") or g.get("name"))
        if topic_tokens & g_topic_tokens:
            goal_score = max(goal_score, float(g.get("priority", 0.5) or 0.0))

    # Calibration (Phase 5.2): real per-domain skill confidence when this
    # candidate's topic matches a tracked skill domain — was previously
    # only ever a single global workspace-level scalar applied uniformly
    # to every candidate regardless of subject (documented as an
    # approximation in the original UTILITY_WEIGHTS comment). Falls back
    # to that same global scalar when no domain match exists, so this is
    # additive, not a behavior change for untracked domains.
    skill_confidence = (getattr(workspace_state, "context", None) or {}).get("skill_confidence", {})
    calibration_score = None
    if skill_confidence:
        for domain, conf in skill_confidence.items():
            if topic_tokens & _tokens(domain):
                calibration_score = max(calibration_score or 0.0, float(conf))
    if calibration_score is None:
        calibration_score = float(getattr(workspace_state, "confidence", 0.5) or 0.5)

    return {
        "counterfactual": cf_score,
        "identity": identity_score,
        "goal_alignment": goal_score,
        "calibration": calibration_score,
    }


@dataclass
class ArbitrationCandidate:
    candidate:        Dict[str, Any]   # original workspace_competition candidate dict
    raw_score:        float            # competition's own score
    predicted_score:  Optional[float]  # prospective_deliberation's best joint_score, or None
    predicted_action: Optional[str]    # which action-type framing scored that
    utility:          float = 0.0      # Phase 4.x: combined multi-factor utility
    utility_terms:    Dict[str, float] = None  # per-factor breakdown, for reportability


@dataclass
class ArbitrationResult:
    raw_winner:       Dict[str, Any]
    arbitrated_winner: Dict[str, Any]
    overridden:       bool
    margin:           float
    candidates:       List[ArbitrationCandidate]
    reasoning:        str


def arbitrate(
    competition_result: Dict[str, Any],
    pcm: Any,
    wsdm: Optional[Any],
    user_id: str = "default",
    workspace_state: Optional[Any] = None,
    valence: Optional[float] = None,
    arousal: Optional[float] = None,
) -> Optional[ArbitrationResult]:
    """
    Re-score the top workspace competition candidates using prospective
    prediction PLUS the Phase 4.x multi-factor utility (see
    UTILITY_WEIGHTS), and decide whether the raw-score winner should be
    overridden by a candidate with meaningfully higher overall utility.

    `workspace_state` is optional and additive: pass a
    cognition.global_workspace.WorkspaceState (e.g. from
    GlobalWorkspace.get_state()) to enable the counterfactual/identity/
    goal_alignment/calibration terms. Omitting it falls back to
    activation+predicted only — this keeps every existing caller working
    unchanged while they migrate to passing workspace_state.

    `valence`/`arousal` (v78, e.g. from EmotionalStateManager.
    get_overall_valence_arousal()) are optional and additive too: when
    supplied, they bound-modulate UTILITY_WEIGHTS per this call (see
    _emotional_modulation) so mood has a real, if capped, effect on which
    future gets chosen — not just how it's described afterward. Omitting
    them uses the static base weights, unchanged from before.

    Returns None if there's nothing to arbitrate (no candidates, no PCM
    data yet, or the winner isn't a goal-type candidate — thoughts don't
    have a topic prospective_deliberation can classify).
    """
    try:
        from cognition.prospective_deliberation import deliberate

        all_candidates = competition_result.get("competition", {}).get("all_candidates")
        winner = competition_result.get("winner")
        if not all_candidates or not winner:
            return None

        # Rank by the competition's own score, take top N — this is
        # "multiple futures" made concrete: not every candidate stays in
        # play, just the ones close enough to the winner to be worth
        # actually predicting outcomes for (predicting all 20-40 raw
        # candidates every cycle would be wasted k-NN calls for candidates
        # that were never realistically going to win anyway).
        ranked = sorted(all_candidates, key=lambda c: c.get("score", 0), reverse=True)
        top_n = ranked[:MAX_CANDIDATES_TO_ARBITRATE]

        goal_state = pcm.read_state() if pcm else None
        if goal_state is None:
            return None

        arb_candidates: List[ArbitrationCandidate] = []
        for cand in top_n:
            if cand.get("type") != "goal":
                # Thoughts have no `topic` field mappable to an action-type
                # framing — included in the returned list for visibility,
                # just never eligible to be predicted or to win arbitration.
                arb_candidates.append(ArbitrationCandidate(
                    candidate=cand, raw_score=cand.get("score", 0),
                    predicted_score=None, predicted_action=None,
                ))
                continue

            topic = cand.get("topic", "") or cand.get("name", "")
            default_action = pcm.classify(topic) if hasattr(pcm, "classify") else "social"
            delib = deliberate(pcm, wsdm, default_action, goal_state, user_id=user_id)

            # Fix: use delib.default_score, not delib.best_score. best_score
            # answers "what's the best POSSIBLE framing given this live
            # state" — since every candidate shares the same current_state,
            # that value converges to the same number regardless of which
            # candidate's topic was passed in, making it useless for
            # comparing candidates against each other. default_score
            # answers "how well does THIS candidate's own natural framing
            # score" — which genuinely differs per candidate, and is the
            # correct signal for "should candidate A win over candidate B".
            arb_candidates.append(ArbitrationCandidate(
                candidate       = cand,
                raw_score       = cand.get("score", 0),
                predicted_score = delib.default_score if delib else None,
                predicted_action= delib.default_action if delib else None,
            ))

        # Nothing predictable yet (e.g. fresh deployment, PCM has no
        # records) — arbitration can't act on nothing.
        predictable = [c for c in arb_candidates if c.predicted_score is not None]
        if not predictable:
            return None

        # ── Phase 4.x: combine into a multi-factor utility per candidate ──
        # v78: emotion-modulated weights, computed once for this call.
        # No-ops back to the static UTILITY_WEIGHTS if valence/arousal
        # weren't supplied.
        _weights = _apply_emotional_modulation(_get_base_weights(), valence, arousal)
        max_raw  = max((c.raw_score for c in arb_candidates), default=1.0) or 1.0
        max_pred = max((c.predicted_score for c in predictable), default=1.0) or 1.0
        for c in arb_candidates:
            extra = _score_extra_factors(c.candidate, workspace_state)
            activation_norm = c.raw_score / max_raw
            predicted_norm  = (c.predicted_score / max_pred) if c.predicted_score is not None else 0.0
            c.utility_terms = {
                "activation": activation_norm,
                "predicted": predicted_norm,
                **extra,
            }
            c.utility = sum(
                _weights[k] * v for k, v in c.utility_terms.items() if k in _weights
            )

        raw_winner_arb = next(
            (c for c in arb_candidates if c.candidate is winner), arb_candidates[0]
        )
        # Executive Arbitration should choose the future with the highest
        # utility, not merely the strongest raw activation — rank by
        # utility among candidates that had SOME predictable framing
        # (non-predictable, e.g. thought-type, candidates stay visible in
        # `candidates` for reportability but can't win — same constraint
        # as before, just applied to utility instead of predicted_score).
        best_utility = max(predictable, key=lambda c: c.utility)

        overridden = False
        margin = 0.0
        arbitrated_winner = winner
        reasoning = (
            f"Raw winner '{winner.get('label', winner.get('name',''))}' confirmed — "
            f"no runner-up scored meaningfully higher utility."
        )

        if best_utility.candidate is not raw_winner_arb.candidate:
            margin = round(best_utility.utility - raw_winner_arb.utility, 5)
            if margin >= OVERRIDE_MARGIN_THRESHOLD:
                overridden = True
                arbitrated_winner = best_utility.candidate
                reasoning = (
                    f"Overrode raw winner '{winner.get('label', winner.get('name',''))}' "
                    f"(competition score={raw_winner_arb.raw_score:.2f}, "
                    f"utility={raw_winner_arb.utility:.3f}) in favor of "
                    f"'{best_utility.candidate.get('label', best_utility.candidate.get('name',''))}' "
                    f"— utility margin={margin:+.3f} "
                    f"(terms={ {k: round(v,3) for k, v in best_utility.utility_terms.items()} })."
                )
        elif raw_winner_arb.predicted_score is None and predictable:
            # Winner itself wasn't predictable (e.g. a thought, not a goal)
            # but something else was — surface that as a strong signal
            # without silently overriding a non-goal candidate we can't
            # actually act on via prospective framing.
            reasoning = (
                f"Raw winner '{winner.get('label', winner.get('name',''))}' has no "
                f"predictable framing (not goal-type) — arbitration deferred to "
                f"competition's own ranking."
            )

        return ArbitrationResult(
            raw_winner        = winner,
            arbitrated_winner = arbitrated_winner,
            overridden        = overridden,
            margin            = margin,
            candidates        = arb_candidates,
            reasoning         = reasoning,
        )

    except Exception as e:
        logger.debug(f"[ExecutiveArbitration] arbitrate error: {e}")
        return None
