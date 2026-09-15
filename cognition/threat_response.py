"""
cognition/threat_response.py

v78b — calibrated threat detection, feeding anxiety through the existing
bounded emotional-influence path.

Design constraints (deliberate, not incidental):

1. The threat signal is computed ONLY from objective, already-existing
   subsystem state — resource depletion, unresolved coherence violations,
   sustained prediction failure. It is NEVER computed from anxiety itself,
   or from anything anxiety influences. This is the single most important
   property of this module: there is no closed loop where anxiety can
   inflate its own trigger. If that loop existed, this would be a genuine
   risk of a runaway/pathological spiral; without it, anxiety can rise in
   response to real pressure and fall when the pressure resolves, but it
   cannot manufacture its own justification.

2. Every nudge goes through EmotionalStateManager.receive_self_influence(),
   the same bounded, ceiling-respecting path used by self_model_influence.py
   and the prediction-error hook in predictive_mind.py. Nothing here writes
   to emotion state directly.

3. Every call immediately checks needs_regulation() and calls regulate()
   if needed, in addition to (not instead of) the periodic checks already
   in ai_system.py. This is an extra safety margin specifically because
   this driver is more targeted/sustained than generic interaction-driven
   nudges — belt and suspenders, not a replacement for the existing net.

4. The threat signal decays to 0 as soon as the underlying pressure
   resolves — there is no persistent "trauma" state here. Objective
   pressure gone => next cycle's signal is 0 => no further nudge. Existing
   emotion decay (2h half-life for anxiety) handles the rest.

5. This module does NOT wire into avoidance behavior (goal_action_executor,
   proactive_outreach, executive_arbitration candidate selection beyond
   what the existing v78 emotion-modulated-utility-weights already does
   generically for any negative valence). Anxiety from real threat flows
   into decisions exactly the same way anxiety from any other source
   already does — through get_overall_valence_arousal() into
   executive_arbitration's weight modulation. No new, threat-specific
   avoidance pathway was added.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Below this, the signal is treated as noise and produces no nudge at all —
# avoids a constant faint hum of anxiety from normal minor fluctuations.
THREAT_NOISE_FLOOR = 0.15

# Maximum single-cycle anxiety nudge, regardless of how high the computed
# threat level is. This is the hard ceiling that keeps one bad cycle from
# producing an outsized emotional swing.
MAX_ANXIETY_NUDGE = 0.05


def compute_threat_level(organism: Any) -> float:
    """
    Objective threat level in [0, 1], averaged across whichever of the
    three real pressure signals are currently available. Returns 0.0 if
    none are reachable (never raises — this must never be the reason a
    cycle fails).
    """
    signals = []

    # ── 1. Resource depletion ───────────────────────────────────────────
    try:
        econ = getattr(getattr(organism, "_loop", None), "_resource_economy", None)
        if econ and hasattr(econ, "status"):
            p = econ.status().get("pressure", {})
            if p.get("cognitive_critical") or p.get("attention_critical"):
                signals.append(0.7)
            elif p.get("cognitive_low") or p.get("social_low") or p.get("attention_low"):
                signals.append(0.3)
            else:
                signals.append(0.0)
    except Exception as e:
        logger.debug(f"[ThreatResponse] resource signal error (non-fatal): {e}")

    # ── 2. Unresolved coherence violations ──────────────────────────────
    try:
        cde = getattr(getattr(organism, "_loop", None), "_cde", None)
        if cde and hasattr(cde, "pending_count"):
            pending = cde.pending_count()
            signals.append(min(1.0, pending / 5.0))
    except Exception as e:
        logger.debug(f"[ThreatResponse] dissonance signal error (non-fatal): {e}")

    # ── 3. Sustained prediction failure ─────────────────────────────────
    try:
        pm = getattr(organism, "predictive_mind", None)
        if pm and hasattr(pm, "stability_metrics"):
            surprise_idx = pm.stability_metrics().get("surprise_idx", 0.0)
            signals.append(max(0.0, min(1.0, float(surprise_idx))))
    except Exception as e:
        logger.debug(f"[ThreatResponse] prediction signal error (non-fatal): {e}")

    if not signals:
        return 0.0
    return round(sum(signals) / len(signals), 3)


def apply_threat_response(organism: Any) -> Optional[float]:
    """
    Compute the current threat level and, if above the noise floor, nudge
    anxiety proportionally (capped at MAX_ANXIETY_NUDGE) through the
    existing safe influence path. Immediately checks and applies
    regulation as an extra safety margin. Returns the computed threat
    level (for logging/dashboard use), or None if nothing was reachable.

    Safe to call every slow cycle — genuinely a no-op most of the time
    (threat_level below the noise floor whenever things are actually
    fine, which is the common case).
    """
    threat_level = compute_threat_level(organism)

    emo = None
    try:
        emo = getattr(getattr(organism, "ai_system", None), "emotional_state", None)
    except Exception:
        pass

    if emo is None:
        return threat_level

    if threat_level >= THREAT_NOISE_FLOOR:
        try:
            delta = min(MAX_ANXIETY_NUDGE, threat_level * MAX_ANXIETY_NUDGE)
            emo.receive_self_influence({"anxiety": +delta})
            logger.debug(
                f"[ThreatResponse] threat_level={threat_level:.2f} -> "
                f"anxiety nudge +{delta:.3f}"
            )
        except Exception as e:
            logger.debug(f"[ThreatResponse] anxiety nudge failed (non-fatal): {e}")

    # Extra safety margin: check regulation immediately, on top of the
    # existing periodic checks in ai_system.py — this driver is more
    # targeted/sustained than generic interaction nudges.
    try:
        if hasattr(emo, "needs_regulation") and emo.needs_regulation():
            emo.regulate()
            logger.debug("[ThreatResponse] regulation applied (post-threat-check)")
    except Exception as e:
        logger.debug(f"[ThreatResponse] regulation check failed (non-fatal): {e}")

    return threat_level
