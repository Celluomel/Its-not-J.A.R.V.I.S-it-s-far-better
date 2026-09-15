"""
ArbitrationLearningTracker — Phase 5.4 (Recursive Cognitive Improvement)
==========================================================================
executive_arbitration.py's UTILITY_WEIGHTS are a static dict, documented
as "every term computed from data that already exists" — true for the
terms themselves, but the WEIGHTS on those terms have never been anything
but hand-picked constants, only ever bound-modulated per-call by emotion
(v78). Nothing has ever checked whether decisions weighted toward, say,
"counterfactual" actually turn out better than decisions weighted toward
"goal_alignment". cross_layer_feedback.py's four channels (checked before
starting this) tune _confidence_floor, WSDM motivation, IC policy, and the
PCM gate from real outcomes — none of them touch UTILITY_WEIGHTS at all.

This closes that gap using the exact same "record a decision, check back
later, nudge a bounded persisted delta" pattern MetaLearningAudit (v64)
already established for CognitiveAuditEngine — applied here to arbitration
instead of parameter proposals.

Tracked outcome (real, not invented): for arbitration decisions that
picked a goal-type candidate (the only type with an id traceable back to
a live, persistent GoalEngine.Goal), whether that goal was subsequently
acted on. update_goals() already prunes 'abandoned' goals out of GoalEngine
entirely while keeping 'completed' ones — so a goal_id that no longer
resolves via get_goal_by_id() reliably means it was abandoned without
being acted on, not merely dormant. A goal that gained actions_taken (or
completed) in the interim is a "good" outcome; one that was abandoned
without gaining actions_taken is "bad"; anything else (still active, no
change yet) is "neutral" and is not learned from — the evidence isn't in.

IMPORTANT — which live path this actually reaches: cognitive_organism.py
has a THIRD, unrelated "_arbitrate()" method (picks dominant drive/
reasoning style via self.arbitration.decide() — never touches
UTILITY_WEIGHTS at all, don't confuse the two). The learned weights here
reach the live per-turn decision through recursive_deliberation.deliberate()
(called synchronously in respond(), NOT through the async-only
executive_arbitration.arbitrate()) — deliberate() calls _compute_utility(),
which was updated (v89) to call executive_arbitration._get_base_weights()
instead of the static UTILITY_WEIGHTS dict, and its winner is written into
the same shared, mutable WorkspaceState object that _build_prompt_additions()
reads every turn. Verified with a direct counterfactual: two candidates
tied at baseline weights, a legal (within DELTA_CAP) learned shift between
two terms flips which one wins _compute_utility() — proof this has live
causal authority, not just a persisted number nobody reads.

The dominant utility term for the winning candidate (which single term
contributed most to its utility score) gets a small bounded nudge: up on
good outcomes, down on bad ones. This never touches the documented
UTILITY_WEIGHTS constant directly — it maintains a separate, persisted,
bounded delta (±0.06 per term, same order of magnitude as v78's emotional
modulation cap) that get_learned_weights() adds on top before emotional
modulation is applied. UTILITY_WEIGHTS itself stays the true floor
default, exactly as its docstring says.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REVIEW_DELAY_CYCLES = 15    # give a goal time to plausibly gain actions_taken
MAX_PENDING          = 200   # cap unreviewed history (memory bound)
DELTA_STEP            = 0.015  # per-review nudge — small; this compounds slowly
                                # over many reviews, same philosophy as GQF's
                                # +0.05 boosts and PEL's -0.06 reductions
DELTA_CAP              = 0.06  # never let learning move a weight more than
                                # this from its documented UTILITY_WEIGHTS default


def _now() -> float:
    return time.time()


class ArbitrationLearningTracker:
    """
    Owned by InternalThoughtLoop. record() is called right after a
    successful arbitrate() call (any goal-type winner, override or not —
    learning from every decision, not just overrides, gives far more
    signal per unit of engagement). review() is called periodically to
    check back on decisions old enough to have a real outcome.
    """

    def __init__(self, path: str = "data/persona/arbitration_learning.json") -> None:
        self._path = Path(path)
        self._lock = Lock()
        self._pending: List[Dict[str, Any]] = []
        self._delta: Dict[str, float] = {}
        self._good = 0
        self._bad = 0
        self._load()
        logger.info("[ArbitrationLearningTracker] initialised")

    # ── Recording ────────────────────────────────────────────────────────

    def record(self, arb_result: Any, cycle: int, goal_engine: Any = None) -> None:
        try:
            winner = arb_result.arbitrated_winner
            if not winner or winner.get("type") != "goal":
                return
            goal_id = winner.get("id")
            if not goal_id:
                return
            winner_arb = next(
                (c for c in arb_result.candidates if c.candidate is winner), None
            )
            if winner_arb is None or not winner_arb.utility_terms:
                return

            weights = get_learned_weights()
            contrib = {
                k: weights.get(k, 0.0) * v
                for k, v in winner_arb.utility_terms.items()
            }
            if not contrib:
                return
            dominant_term = max(contrib, key=contrib.get)

            # Snapshot actions_taken NOW so review() can compare a delta,
            # not an absolute — a goal that already had actions before this
            # decision shouldn't get credit for actions from before it won.
            actions_at_record = 0
            if goal_engine is not None:
                try:
                    g = goal_engine.get_goal_by_id(goal_id)
                    if g is not None:
                        actions_at_record = g.actions_taken
                except Exception:
                    pass

            with self._lock:
                self._pending.append({
                    "goal_id": goal_id,
                    "cycle_recorded": cycle,
                    "dominant_term": dominant_term,
                    "actions_taken_at_record": actions_at_record,
                    "recorded_at": _now(),
                })
                if len(self._pending) > MAX_PENDING:
                    self._pending = self._pending[-MAX_PENDING:]
        except Exception as e:
            logger.debug(f"[ArbitrationLearningTracker] record failed (non-fatal): {e}")

    # ── Review ───────────────────────────────────────────────────────────

    def review(self, goal_engine: Any, cycle: int) -> Dict[str, Any]:
        report = {"reviewed": 0, "good": 0, "bad": 0, "still_pending": 0}
        if goal_engine is None or not self._pending:
            return report

        with self._lock:
            due = [p for p in self._pending if cycle - p["cycle_recorded"] >= REVIEW_DELAY_CYCLES]
            still_pending = [p for p in self._pending if p not in due]

        for entry in due:
            report["reviewed"] += 1
            try:
                goal = goal_engine.get_goal_by_id(entry["goal_id"])
                term = entry["dominant_term"]
                if goal is None:
                    # Pruned → was abandoned without being acted on further.
                    self._nudge(term, good=False)
                    report["bad"] += 1
                    self._bad += 1
                elif goal.status == "completed" or goal.actions_taken > entry.get("actions_taken_at_record", 0):
                    self._nudge(term, good=True)
                    report["good"] += 1
                    self._good += 1
                else:
                    # Still active, no action yet — not enough evidence.
                    report["still_pending"] += 1
                    still_pending.append(entry)
            except Exception as e:
                logger.debug(f"[ArbitrationLearningTracker] review entry failed (non-fatal): {e}")

        with self._lock:
            self._pending = still_pending[-MAX_PENDING:]
        if report["good"] or report["bad"]:
            self._save()
        return report

    def _nudge(self, term: str, good: bool) -> None:
        with self._lock:
            cur = self._delta.get(term, 0.0)
            step = DELTA_STEP if good else -DELTA_STEP
            self._delta[term] = max(-DELTA_CAP, min(DELTA_CAP, cur + step))

    # ── Weight access ────────────────────────────────────────────────────

    def learned_delta(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._delta)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._pending),
                "good_total": self._good,
                "bad_total": self._bad,
                "delta": dict(self._delta),
            }

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "pending": self._pending,
                    "delta":   self._delta,
                    "good":    self._good,
                    "bad":     self._bad,
                    "_meta":   {"version": "v89", "ts": _now()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[ArbitrationLearningTracker] save failed (non-fatal): {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self._pending = data.get("pending", [])
                self._delta   = data.get("delta", {})
                self._good    = data.get("good", 0)
                self._bad     = data.get("bad", 0)
        except Exception as e:
            logger.warning(f"[ArbitrationLearningTracker] load failed (non-fatal): {e}")


# ── Module-level singleton + weight access ──────────────────────────────
# executive_arbitration.py imports get_learned_weights() lazily (inside
# arbitrate(), to avoid a module-load-time circular import — same pattern
# already used throughout this codebase, e.g. recursive_deliberation.py
# importing from executive_arbitration.py inside functions, not at the top).

_tracker: Optional[ArbitrationLearningTracker] = None


def get_tracker() -> ArbitrationLearningTracker:
    global _tracker
    if _tracker is None:
        _tracker = ArbitrationLearningTracker()
    return _tracker


def get_learned_weights() -> Dict[str, float]:
    """
    UTILITY_WEIGHTS (the documented floor default) plus a bounded, learned,
    persisted delta per term — renormalized so the terms still sum to 1.0,
    same invariant as the static UTILITY_WEIGHTS dict. Falls back cleanly
    to the static weights if anything goes wrong.
    """
    try:
        from cognition.executive_arbitration import UTILITY_WEIGHTS
    except Exception:
        return {}
    try:
        delta = get_tracker().learned_delta()
    except Exception:
        delta = {}
    shifted = {k: max(0.0, v + delta.get(k, 0.0)) for k, v in UTILITY_WEIGHTS.items()}
    total = sum(shifted.values()) or 1.0
    return {k: v / total for k, v in shifted.items()}
