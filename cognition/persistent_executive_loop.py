"""
PersistentExecutiveLoop — Phase 5.1
=====================================
"Am I still trying to achieve this, should priorities change" as a
continuous process, distinct from what already exists:

  - GoalEngine.update_goals() decays energy mechanically every cycle and
    moves goals through active→dormant→abandoned purely on elapsed time
    and action counts — it never asks WHY a goal exists.
  - GoalQualityFilter (Phase 4.1) only ever boosts a goal (keyword overlap
    with the CURRENT dominant tension) or marks it as semantic noise — it
    never reduces priority because the reason a goal was created has gone
    away, and it only checks the goal's NAME against current tensions, not
    the goal's actual origin.

Both leave a real gap: a goal spawned because epistemic pressure was high
keeps its priority even after that pressure has long since resolved,
because nothing ever re-checks the ORIGINATING pressure/trait specifically.

This module closes that gap using only data that already exists:
goal.origin is set at creation time in GoalEngine.generate_candidates()
directly from Motivation.origin ("pressure:epistemic", "trait:curiosity",
etc — see derive_motivations()). For any goal with a traceable origin, this
re-checks the CURRENT value of that exact pressure/trait against the exact
threshold that created it (GoalEngine.PRESSURE_MOTIVATION_THRESHOLDS /
TRAIT_MOTIVATION_THRESHOLDS — the same constants derive_motivations() uses,
imported rather than duplicated, so the two can never drift apart) and, if
the reason has genuinely receded, reduces priority. It never abandons or
forces dormancy directly — that stays owned by GoalEngine's existing
lifecycle in update_goals(); this only ever adjusts the input (priority)
that lifecycle and the arbitration/deliberation scoring both already read.

Goals with non-traceable origins ("baseline:...", curiosity-resolver
"explore" goals with no single pressure/trait behind them) are left alone
and reported as such — inventing a plausible-sounding signal for them
would not be evidence-based, just guessing.

Zero LLM calls — pure Python comparison against already-live data, run
on a cycle cadence from InternalThoughtLoop, same pattern as GQF's
_phase4_maintenance() cadence.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How much to reduce priority per review when a goal's originating reason
# has receded. Small and repeatable (this runs periodically) rather than a
# single large cut — a goal whose reason keeps being absent drifts down
# over several reviews; one that recovers stops drifting immediately.
PRIORITY_STEP = 0.06
PRIORITY_FLOOR = 0.10  # never push a goal below this from here — GoalEngine's
                        # own dormancy/abandonment lifecycle takes it from there

# Small buffer below the creation threshold before reducing — avoids
# flapping a goal's priority up/down when the pressure is oscillating right
# at the boundary.
RECEDE_MARGIN = 0.05


class PersistentExecutiveLoop:
    """
    Owned by InternalThoughtLoop. review(organism) re-examines every active
    goal's origin against current pressures/traits and adjusts priority
    when the originating reason has receded.
    """

    def __init__(self) -> None:
        logger.info("[PersistentExecutiveLoop] initialised")

    def review(self, organism: Any) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "reviewed": 0, "reduced": [], "held": [], "not_traceable": 0,
        }
        try:
            from cognition.goal_engine import (
                PRESSURE_MOTIVATION_THRESHOLDS, TRAIT_MOTIVATION_THRESHOLDS,
            )
        except Exception as e:
            logger.debug(f"[PersistentExecutiveLoop] threshold import failed (non-fatal): {e}")
            return report

        ai_system = getattr(organism, "ai_system", None)
        goal_engine = getattr(ai_system, "goal_engine", None) if ai_system else None
        if goal_engine is None or not hasattr(goal_engine, "get_active_goals"):
            return report

        try:
            active_goals = goal_engine.get_active_goals() or []
        except Exception as e:
            logger.debug(f"[PersistentExecutiveLoop] get_active_goals failed (non-fatal): {e}")
            return report
        if not active_goals:
            return report

        # Same live pressures tick()/derive_motivations() use — must agree,
        # or a goal could be reduced for a reason that wouldn't have
        # stopped it from being created in the first place.
        try:
            current_pressures = goal_engine.get_current_pressures()
        except Exception as e:
            logger.debug(f"[PersistentExecutiveLoop] pressure gathering failed (non-fatal): {e}")
            current_pressures = {}

        trait_confidence: Dict[str, float] = {}
        try:
            identity = ai_system.identity_system.get_identity()
            for t in getattr(identity, "core_traits", []):
                trait_confidence[t.name] = t.confidence
        except Exception as e:
            logger.debug(f"[PersistentExecutiveLoop] identity read failed (non-fatal): {e}")

        for goal in active_goals:
            report["reviewed"] += 1
            origin = getattr(goal, "origin", "") or ""

            current_value: Optional[float] = None
            threshold: Optional[float] = None

            if origin.startswith("pressure:"):
                key = origin.split(":", 1)[1]
                threshold = PRESSURE_MOTIVATION_THRESHOLDS.get(key)
                if threshold is not None:
                    current_value = current_pressures.get(key, 0.0)
            elif origin.startswith("trait:"):
                key = origin.split(":", 1)[1]
                threshold = TRAIT_MOTIVATION_THRESHOLDS.get(key)
                if threshold is not None:
                    current_value = trait_confidence.get(key)

            if threshold is None or current_value is None:
                report["not_traceable"] += 1
                continue

            if current_value < (threshold - RECEDE_MARGIN):
                old_priority = goal.priority
                new_priority = max(PRIORITY_FLOOR, old_priority - PRIORITY_STEP)
                if new_priority < old_priority:
                    goal.priority = new_priority
                    report["reduced"].append({
                        "id": goal.id, "topic": goal.topic, "origin": origin,
                        "current_value": round(current_value, 3),
                        "threshold": threshold,
                        "priority": [round(old_priority, 3), round(new_priority, 3)],
                    })
                    logger.info(
                        f"[PersistentExecutiveLoop] '{goal.topic}' reduced "
                        f"{old_priority:.2f}\u2192{new_priority:.2f} "
                        f"({origin} receded: {current_value:.2f} < {threshold:.2f})"
                    )
                else:
                    report["held"].append(goal.id)
            else:
                report["held"].append(goal.id)

        if report["reduced"]:
            try:
                goal_engine.save_goals()
            except Exception as e:
                logger.debug(f"[PersistentExecutiveLoop] save failed (non-fatal): {e}")

        return report
