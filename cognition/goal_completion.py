"""cognition/goal_completion.py

Completion semantics (B + A + C) for PandoraBOX goals — the single source of truth.

A goal may only be *completed* when its work is real:

  B (action gate)  — It must have performed at least one world-facing
                     (non-introspective) action: web_search or user_question.
                     Pure introspection (self_question / bisociative_hypothesis /
                     memory_recall) is reflection, not completion. Completing a
                     goal after N insights — with nothing done in the world and
                     nothing resolved — is the "zombie metronome" bug this
                     module exists to kill.

  A (tension exit) — If the goal was spawned by a tension (``tension_source``),
                     that tension must have *resolved*: dropped to or below the
                     generation threshold for that tension (the same threshold
                     GoalQualityFilter uses to decide whether to spawn a goal
                     for it). That alignment is deliberate: a tension is only
                     "resolved" once it is low enough that the system would no
                     longer spawn a goal for it, which prevents the
                     complete-then-immediately-regenerate oscillation.
                     If the goal has no tension source (e.g. a curiosity-driven
                     goal), the action alone suffices.

  C (narrative)    — Completion is a story-worthy event. It is recorded as a
                     chapter in the life story via
                     ``narrative_identity.record_chapter`` so that closure
                     becomes part of her lived history rather than just a
                     status flag flipping in a JSON file.

Every completion path MUST route through :func:`goal_completion_eligible` and
:func:`record_goal_completion`:

  * GoalActionExecutor._decay_goal_energy  (insight-budget exhausted)
  * GoalEngine.update_goals                (long-lived auto-complete)
  * GoalQualityFilter.run                  (retroactive settlement)

This module is deliberately dependency-light: standard library only at import
time, package imports deferred into function bodies. That makes it safe to
import from anywhere in the package without circular-import risk.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# B: actions that reached the world (vs. introspective reflection)
NON_INTROSPECTIVE_ACTIONS = ("web_search", "user_question")

# A: per-tension "resolved" thresholds live in the SINGLE SOURCE OF TRUTH at
# cognition.tension_topics.TENSION_GOALS[*]['thresh'] — the level at which
# GoalQualityFilter generates a goal for a given tension. A tension counts as
# resolved when it falls to or below that same threshold, i.e. low enough that
# the system would no longer spawn a goal for it. goal_completion_eligible()
# reads the value from there at call time, so the A gate and the generator can
# never drift out of sync.
# Fallback for tensions not present in TENSION_GOALS.
DEFAULT_TENSION_THRESHOLD = 0.30


def _gget(goal: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a Goal dataclass instance or a goal dict.

    The goal system has two representations in flight (GoalEngine's in-memory
    Goal objects and the DAL/GQF dicts persisted to goals.json). This helper
    lets the shared logic operate on both without caring which it got.
    """
    if isinstance(goal, dict):
        return goal.get(key, default)
    return getattr(goal, key, default)


def goal_completion_eligible(
    goal: Any,
    current_tensions: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """Apply the B + A gate to a goal.

    Returns ``(eligible, reason)`` — ``reason`` is a human-readable
    explanation used for logging and the narrative chapter description.
    """
    current_tensions = current_tensions or {}

    # B: world-facing action gate
    try:
        non_intro = int(_gget(goal, "non_introspective_actions", 0) or 0)
    except (TypeError, ValueError):
        non_intro = 0
    if non_intro < 1:
        return (
            False,
            "no world-facing action yet (needs web_search or user_question)",
        )

    # A: tension-resolution gate
    tsrc = _gget(goal, "tension_source", None)
    if not tsrc:
        # No tension source (e.g. curiosity_autonomous goal): the action
        # alone is sufficient for completion.
        return (True, "action taken; no tension source to resolve")

    from cognition.tension_topics import TENSION_GOALS
    threshold = TENSION_GOALS.get(tsrc, {}).get('thresh', DEFAULT_TENSION_THRESHOLD)
    cur = current_tensions.get(tsrc)
    if cur is None:
        # Conservatively: we cannot confirm the tension has resolved, so we do
        # not complete the goal on an unverifiable A gate.
        return (
            False,
            "no current reading for tension '%s' "
            "(cannot confirm resolution <= %.2f)" % (tsrc, threshold),
        )
    try:
        cur = float(cur)
    except (TypeError, ValueError):
        return (False, "non-numeric current value for tension '%s'" % tsrc)

    if cur <= threshold:
        return (
            True,
            "action taken; tension '%s' resolved (%.2f <= %.2f)"
            % (tsrc, cur, threshold),
        )
    return (
        False,
        "action taken but tension '%s' still %.2f (needs <= %.2f to resolve)"
        % (tsrc, cur, threshold),
    )


# Project root = the directory that CONTAINS the cognition/ package. Resolving
# the persona dir from this (not from the process CWD) keeps the A gate working
# no matter where PandoraBOX is launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_current_tensions(persona_dir: Optional[str] = None) -> Dict[str, float]:
    """Read the live tension vector from the persona store.

    Uses the same DataAccess.get_tensions() source GoalQualityFilter scores
    against, so the A gate and the generator agree about tension levels.
    When persona_dir is omitted it defaults to <project_root>/data/persona
    (resolved from this file's location, not the CWD). Returns {} on any
    failure; callers treat an empty mapping conservatively.
    """
    if persona_dir is None:
        persona_dir = os.path.join(_PROJECT_ROOT, "data", "persona")
    try:
        from core.data.access import DataAccess
        dal = DataAccess(str(persona_dir))
        t = dal.get_tensions()
        return t if isinstance(t, dict) else {}
    except Exception:
        return {}


def get_narrative_identity(organism: Any = None) -> Any:
    """Locate the NarrativeIdentity instance.

    Tries an explicit organism first (cheapest, most direct), then falls back
    to the application-global state singleton (core.state.state.persona
    ._organism). Returns None when unavailable; callers then skip the chapter.
    """
    if organism is not None:
        ni = getattr(organism, "narrative_identity", None)
        if ni is not None:
            return ni
    try:
        from core.state import state as _st
        persona = getattr(_st, "persona", None)
        org = getattr(persona, "_organism", None) if persona else None
        return getattr(org, "narrative_identity", None) if org else None
    except Exception:
        return None


def record_goal_completion(
    goal: Any,
    goal_engine: Any = None,
    dal: Any = None,
    organism: Any = None,
    reason: str = "",
) -> bool:
    """Complete a goal through the single canonical path (C: + chapter).

    Marks the goal completed in whichever store is supplied (the GoalEngine
    object store and/or the DAL dict store) and records a life-story chapter
    via narrative_identity.record_chapter. Returns True if a store write
    succeeded.
    """
    gid = _gget(goal, "id", "")
    name = _gget(goal, "name") or _gget(goal, "topic") or gid or "goal"

    wrote = False
    # Prefer the object store when we have it (it does economy + persistence).
    if goal_engine is not None and hasattr(goal_engine, "complete_goal"):
        try:
            goal_engine.complete_goal(gid, completion=1.0)
            wrote = True
        except Exception as e:
            logger.debug("[Completion] goal_engine.complete_goal failed: %s" % e)
    if not wrote and dal is not None and hasattr(dal, "save_goal"):
        try:
            # Mark the goal completed in its own representation (dict or
            # object) so the persisted record reflects the completion.
            if isinstance(goal, dict):
                goal["status"] = "completed"
                goal["completion"] = 1.0
            else:
                try:
                    goal.status = "completed"
                    goal.completion = 1.0
                except Exception:
                    pass
            dal.save_goal(goal)
            wrote = True
        except Exception as e:
            logger.debug("[Completion] dal.save_goal failed: %s" % e)

    # C: record the life-story chapter (best-effort; never blocks completion).
    ni = get_narrative_identity(organism)
    if ni is not None and hasattr(ni, "record_chapter"):
        try:
            ni.record_chapter(
                title="Completed: %s" % name,
                description=("The goal '%s' was completed. %s" % (name, reason)).strip(),
                emotion="satisfaction",
                significance=0.6,
            )
        except Exception as e:
            logger.debug("[Completion] record_chapter failed: %s" % e)

    logger.info("[Completion] GOAL COMPLETED: '%s' — %s" % (name, reason))
    return wrote
