"""
Activity Selector
=================
The cortex prefrontal of the orchestrator.  Given a list of events and a
DriveVector, it returns the single activity PandoraBOX should perform next.

Priority order:
  1. User-facing events     (USER_MESSAGE, VOICE_TRANSCRIPTION, EXTERNAL_MESSAGE)
  2. Energy emergency       (energy < 15 %)
  3. Homeostasis crisis     (score < 0.50)
  4. Coherence emergency    (contradiction pressure very high)
  5. Scheduled clock ticks  (REFLECTION_DUE, CONSOLIDATION_DUE, EVOLUTION_DUE)
  6. Drive-based autonomous activity
  7. Default: idle_reflection

Activities returned (strings):
    respond_user          — reply to a user message
    respond_external      — reply on Telegram / WhatsApp
    restore_energy        — rest / light maintenance
    stabilise_homeostasis — corrective psychological action
    resolve_contradiction — address detected conflict
    reflect               — medium-cadence self-reflection
    consolidate_memory    — slow-cadence memory sweep
    evolve                — personality / self-concept update
    explore_curiosity     — follow high curiosity into research
    pursue_goal           — work on the highest-priority drive
    idle_reflection       — low-cost background thinking
"""

import logging
from typing import Any, List, Optional, Tuple

from .event_system import (
    Event, USER_MESSAGE, VOICE_TRANSCRIPTION, EXTERNAL_MESSAGE,
    PROACTIVE_TRIGGER, LOW_ENERGY, CONTRADICTION_FOUND, CURIOSITY_PEAK,
    REFLECTION_DUE, CONSOLIDATION_DUE, EVOLUTION_DUE,
)
from .drive_system import DriveVector

logger = logging.getLogger(__name__)


class ActivitySelector:
    """
    Selects the next activity + optional payload from events and drives.

    Returns (activity_name: str, payload: Any)
    """

    def select(
        self,
        events: List[Event],
        drives: DriveVector,
        organism: Any = None,
    ) -> Tuple[str, Any]:

        # ── 1. User-facing events always take priority ─────────────────────
        for e in events:
            if e.type in (USER_MESSAGE, VOICE_TRANSCRIPTION):
                p = e.payload or {}
                logger.debug(f"[Selector] user input → respond_user")
                return "respond_user", p

            if e.type == EXTERNAL_MESSAGE:
                logger.debug(f"[Selector] external message → respond_external")
                return "respond_external", e.payload

            if e.type == PROACTIVE_TRIGGER:
                return "respond_user", {"proactive": True}

        # ── 2. Energy emergency ────────────────────────────────────────────
        if drives.needs_rest():
            # Low energy must REDUCE INTENSITY, not block all cognition.
            # Use drives.social as proxy for "user recently active":
            # social > 0.35 means a user interacted in the last few minutes.
            # When social is low (truly idle), allow the restore_energy action.
            # This breaks the metabolic lock-loop observed on the dashboard.
            if drives.social < 0.35 and not events:
                logger.debug("[Selector] energy < 15% and idle → restore_energy")
                return "restore_energy", None
            logger.debug("[Selector] energy < 15% but socially active — continuing at low intensity")

        # ── 3. Homeostasis crisis ──────────────────────────────────────────
        if drives.needs_homeostasis():
            logger.debug("[Selector] homeostasis crisis → stabilise_homeostasis")
            return "stabilise_homeostasis", None

        # ── 4. Coherence emergency (from events or drive) ──────────────────
        for e in events:
            if e.type == CONTRADICTION_FOUND and e.priority >= 0.7:
                return "resolve_contradiction", e.payload
        if drives.needs_coherence():
            return "resolve_contradiction", None

        # ── 5. Scheduled clock ticks ──────────────────────────────────────
        for e in events:
            if e.type == REFLECTION_DUE:
                return "reflect", None
            if e.type == CONSOLIDATION_DUE:
                return "consolidate_memory", None
            if e.type == EVOLUTION_DUE:
                return "evolve", None
            if e.type == CURIOSITY_PEAK:
                return "explore_curiosity", e.payload

        # ── 6. Drive-based autonomous choice ─────────────────────────────
        if drives.is_curious():
            logger.debug("[Selector] curiosity high → explore_curiosity")
            return "explore_curiosity", None

        if drives.is_socially_hungry():
            logger.debug("[Selector] social drive → idle_reflection (will surface thought)")
            return "idle_reflection", {"social_hint": True}

        if drives.goal_progress < 0.40:
            return "pursue_goal", None

        # ── 7. Dominant thought action (TTE-driven) ───────────────────────
        # When TTE has selected a dominant thread, emit goal_exploration.
        # This fires even when curiosity/goal thresholds aren't crossed,
        # because DTS has already committed to a specific cognitive direction.
        if organism is not None:
            try:
                _dt = getattr(organism, '_dominant_thought', None)
                if _dt and getattr(_dt, 'dominant', None):
                    return "goal_exploration", {
                        "topic": _dt.dominant.topic,
                        "goal":  _dt.dominant.goal,
                    }
            except Exception:
                pass

        # ── 8. Default ────────────────────────────────────────────────────
        return "idle_reflection", None
