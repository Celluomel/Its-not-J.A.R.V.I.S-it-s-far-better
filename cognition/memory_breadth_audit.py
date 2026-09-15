"""
MemoryBreadthAudit — Phase 6.6
================================
The peer-cognition proposal's own caution: don't build a smarter
retrieval strategy (temporal/divergent/associative neighbors) until the
existing 6->10 widen-on-surprise mechanism (v90) is shown to produce a
measurable benefit. This is that measurement, not the smarter strategy —
deliberately so; building the fancier version first would be exactly the
kind of decorative-but-unverified addition this whole engagement has
been finding and fixing.

Same "record a decision, check back later, compare" pattern already
established by MetaLearningAudit (v64) and ArbitrationLearningTracker
(v89), applied here to a different question: does widening memory
retrieval when a recent prediction was surprising actually correlate
with the NEXT prediction being less surprising — i.e. did the extra
context genuinely help disambiguate the situation, or did it just add
noise?

Real, already-computed signal used for "before"/"after": each
PredictionResult already carries error_level (predictive_mind.py). No
new metric invented — before = predictive_mind.recent_error_level() at
the moment retrieval widened (or didn't); after = the same call, read
back on the NEXT turn's prediction. Tracks two running averages (widened
vs baseline group) so the two groups can be honestly compared even
though the two conditions never happen on the exact same turn.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_PENDING = 200
REVIEW_DELAY_TURNS = 1   # compare against the very next prediction — this
                          # question is about immediate disambiguation, not
                          # a delayed outcome like a goal's actions_taken


class MemoryBreadthAudit:
    def __init__(self, path: str = "data/persona/memory_breadth_audit.json") -> None:
        self._path = Path(path)
        self._lock = Lock()
        self._pending: List[Dict[str, Any]] = []
        self._widened_deltas: List[float] = []
        self._baseline_deltas: List[float] = []
        self._turn_counter = 0
        self._load()
        logger.info("[MemoryBreadthAudit] initialised")

    def record_and_review(self, widened: bool, predictive_mind: Any) -> Dict[str, Any]:
        """
        Single entry point, called once per turn from persona_bridge.py
        right where the retrieval-widening decision is made. Combines
        record (this turn's decision) + review (checking any turn at
        least REVIEW_DELAY_TURNS old) so the caller doesn't need to
        thread a turn_id through — the counter here IS the turn clock.
        """
        report = {"reviewed": 0, "still_pending": 0}
        if predictive_mind is None or not hasattr(predictive_mind, "recent_error_level"):
            return report
        current_error = predictive_mind.recent_error_level()

        with self._lock:
            self._turn_counter += 1
            this_turn = self._turn_counter
            self._pending.append({
                "widened": widened,
                "pre_error_level": current_error,
                "turn_id": this_turn,
            })
            if len(self._pending) > MAX_PENDING:
                self._pending = self._pending[-MAX_PENDING:]

            due = [p for p in self._pending
                   if this_turn - p["turn_id"] >= REVIEW_DELAY_TURNS]
            still_pending = [p for p in self._pending if p not in due]

        for entry in due:
            report["reviewed"] += 1
            delta = entry["pre_error_level"] - current_error  # positive = improved
            if entry["widened"]:
                self._widened_deltas.append(delta)
                self._widened_deltas = self._widened_deltas[-100:]
            else:
                self._baseline_deltas.append(delta)
                self._baseline_deltas = self._baseline_deltas[-100:]

        with self._lock:
            self._pending = still_pending
            report["still_pending"] = len(still_pending)

        if report["reviewed"]:
            self._save()
        return report

    def summary(self) -> Dict[str, Any]:
        """
        Honest comparison, not a verdict — enough evidence to decide
        whether the smarter retrieval strategy is worth building, per the
        proposal's own condition. avg_delta > 0 means error level tended
        to drop after that condition; the widened-vs-baseline GAP is what
        actually answers "does widening help disambiguate", not either
        number alone.
        """
        def _avg(xs: List[float]) -> Optional[float]:
            return round(sum(xs) / len(xs), 4) if xs else None

        widened_avg = _avg(self._widened_deltas)
        baseline_avg = _avg(self._baseline_deltas)
        gap = (
            round(widened_avg - baseline_avg, 4)
            if widened_avg is not None and baseline_avg is not None else None
        )
        return {
            "widened_samples": len(self._widened_deltas),
            "baseline_samples": len(self._baseline_deltas),
            "widened_avg_error_delta": widened_avg,
            "baseline_avg_error_delta": baseline_avg,
            "gap": gap,  # positive gap = widening correlates with more improvement
        }

    # ── persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "pending":          self._pending,
                    "widened_deltas":   self._widened_deltas,
                    "baseline_deltas":  self._baseline_deltas,
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[MemoryBreadthAudit] save failed (non-fatal): {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self._pending = data.get("pending", [])
                self._widened_deltas = data.get("widened_deltas", [])
                self._baseline_deltas = data.get("baseline_deltas", [])
        except Exception as e:
            logger.warning(f"[MemoryBreadthAudit] load failed (non-fatal): {e}")


_audit: Optional[MemoryBreadthAudit] = None


def get_memory_breadth_audit() -> MemoryBreadthAudit:
    global _audit
    if _audit is None:
        _audit = MemoryBreadthAudit()
    return _audit
