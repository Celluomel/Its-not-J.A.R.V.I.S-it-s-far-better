"""
cognition/counterfactual_simulator.py  — Phase 3.3

CounterfactualSimulator: "What would have happened if I'd done X instead?"

Takes committed past interactions (ConsequenceRecords from PCM,
DynamicsRecords from WSDM) and retroactively queries PCM/WSDM's k-NN
prediction machinery against alternative action types, using the same
state_before / context_before vectors the actual interaction had.

Key design decisions:
  - Reuses PCM._predict() and WSDM._predict() directly — no new model.
    Both already accept arbitrary state vectors as input.
  - Weighted by calibrated confidence from Phase 3.1/3.2 (degrades
    gracefully to raw confidence before bins have filled).
  - Runs on consolidation cycle only, gated by MIN_RECORDS_FOR_CF and
    a minimum interval, because 7 action types × 10 interactions = 70
    k-NN lookups per trigger (cheap, but not free).
  - Does NOT modify PCM/WSDM — purely observational.
  - Primary output consumer: Phase 3.5 Introspective Observer.

Relationship to CognitiveImmuneSystem's "counterfactual mode":
  That is a defensive immune response to cognitive stagnation (inject
  opposing topics, bias toward disconfirmation). This is retroactive
  decision analysis ("was that action the best available?"). They are
  complementary and non-overlapping.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SAVE_PATH              = "data/persona/counterfactual_log.json"
MAX_LOG_ENTRIES        = 200     # keep last N completed queries
MIN_RECORDS_FOR_CF     = 15      # PCM records needed before simulator fires
MAX_RECENT_INTERACTIONS = 10     # how many past interactions to analyse per run
CF_INTERVAL_CYCLES     = 5       # consolidation cycles between runs (~50 min)
PCM_ENERGY_WEIGHT      = 0.55    # weight of PCM energy_delta in joint_score
WSDM_ENGAGEMENT_WEIGHT = 0.45    # weight of WSDM engagement_signal in joint_score

ALL_ACTION_TYPES = [
    "philosophical", "technical", "emotional", "creative",
    "social", "analytical", "deep_reasoning",
]


@dataclass
class CounterfactualBranch:
    """One alternative action's predicted outcome for a past interaction."""
    action_type:        str
    pcm_energy_delta:   Optional[float]   # PCM predicted energy change
    pcm_confidence:     Optional[float]   # PCM raw confidence
    pcm_calibrated:     Optional[float]   # PCM calibrated confidence (None if uncal.)
    wsdm_engagement:    Optional[float]   # WSDM predicted engagement_signal
    wsdm_confidence:    Optional[float]   # WSDM raw confidence
    wsdm_calibrated:    Optional[float]   # WSDM calibrated confidence (None if uncal.)
    joint_score:        float             # composite ranking score
    was_actual:         bool = False      # True for the branch that actually happened


@dataclass
class CounterfactualQuery:
    """One completed counterfactual analysis of a past interaction."""
    interaction_n:    int
    actual_action:    str
    actual_outcome:   str                                # positive/negative/neutral
    branches:         Dict[str, CounterfactualBranch]   # action_type -> branch
    best_alternative: Optional[str]      # highest joint_score non-actual branch
    actual_rank:      int                # rank of actual action among all branches (1=best)
    worst_actual:     bool               # True if actual was lowest-scoring branch
    computed_at:      float = field(default_factory=time.time)


class CounterfactualSimulator:
    """
    Retroactively evaluates past interactions against alternative action
    types using PCM and WSDM's existing k-NN prediction machinery.
    """

    def __init__(self, organism: Any) -> None:
        self._organism = organism
        self._lock     = threading.Lock()
        self._log:     List[Dict] = []
        self._path     = Path(SAVE_PATH)
        self._last_run_cycle: int = -CF_INTERVAL_CYCLES  # fire on first eligible cycle
        self._calibration_engine = None
        self._load()

        try:
            from cognition.calibration_engine import CalibrationEngine
            self._calibration_engine = CalibrationEngine()
        except Exception as e:
            logger.debug(f"[CounterfactualSim] CalibrationEngine unavailable: {e}")

        logger.info(
            f"[CounterfactualSim] Initialised — {len(self._log)} past queries"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Called every slow cycle from InternalLoop. Runs analysis on
        consolidation cycles only, gated by interval and data readiness."""
        is_consolidation = (slow_cycle % 5 == 0 and slow_cycle > 0)
        if not is_consolidation:
            return
        if slow_cycle - self._last_run_cycle < CF_INTERVAL_CYCLES:
            return

        threading.Thread(
            target=self._run,
            args=(slow_cycle,),
            daemon=True,
            name="cf-simulator",
        ).start()

    def recent_findings(self, n: int = 5) -> List[Dict]:
        """Return n most recent query results for prompt injection / dashboard."""
        with self._lock:
            return list(self._log[-n:])

    def worst_actual_streak(self) -> int:
        """Number of consecutive recent interactions where actual was worst-ranked.
        A streak > 3 is a concrete signal for Phase 3.5 Introspective Observer."""
        with self._lock:
            return self._worst_actual_streak_locked()

    def _worst_actual_streak_locked(self) -> int:
        """Same logic as worst_actual_streak() but assumes self._lock is
        ALREADY held by the caller — never acquires it itself. Used by
        status() to avoid the exact re-entrant-lock deadlock found in
        TemporalProjection.status() earlier: a plain threading.Lock() is
        not reentrant, so a method holding it must never call another
        method that tries to acquire it again — that call blocks forever,
        and since it runs inside InternalLoop's single background thread
        (via tick() -> _run(), or via _fetch_all() in the dashboard's
        to_thread()), the entire background loop or the dashboard fetch
        hangs permanently with no exception and nothing in the logs."""
        streak = 0
        for entry in reversed(self._log):
            if entry.get("worst_actual"):
                streak += 1
            else:
                break
        return streak

    def status(self) -> Dict:
        with self._lock:
            return {
                "total_queries": len(self._log),
                "worst_actual_streak": self._worst_actual_streak_locked(),
                "last_run_cycle": self._last_run_cycle,
                "recent": self._log[-3:] if self._log else [],
            }

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _run(self, slow_cycle: int) -> None:
        try:
            loop = getattr(self._organism, "_loop", None)
            if loop is None:
                return

            pcm  = getattr(loop, "_consequence_model", None)
            wsdm = getattr(loop, "_world_self_dynamics", None)

            if pcm is None:
                return
            if len(getattr(pcm, "_records", [])) < MIN_RECORDS_FOR_CF:
                return

            # Take the MAX_RECENT_INTERACTIONS most recent PCM records
            pcm_records = list(pcm._records)[-MAX_RECENT_INTERACTIONS:]
            wsdm_records = list(getattr(wsdm, "_records", []))

            # Build a lookup of WSDM records by interaction_n for fast pairing
            wsdm_by_n: Dict[int, Any] = {}
            for r in wsdm_records:
                n = getattr(r, "interaction_n", None)
                if n is not None:
                    wsdm_by_n[n] = r

            new_queries: List[CounterfactualQuery] = []
            for pcm_rec in pcm_records:
                query = self._analyse_interaction(pcm, wsdm, pcm_rec, wsdm_by_n)
                if query is not None:
                    new_queries.append(query)

            if new_queries:
                with self._lock:
                    for q in new_queries:
                        self._log.append(asdict(q))
                    if len(self._log) > MAX_LOG_ENTRIES:
                        self._log = self._log[-MAX_LOG_ENTRIES:]
                self._save()

                worst_streak = self.worst_actual_streak()
                logger.info(
                    f"[CounterfactualSim] Cycle {slow_cycle}: analysed "
                    f"{len(new_queries)} interactions. "
                    f"Worst-actual streak: {worst_streak}"
                )

            self._last_run_cycle = slow_cycle

        except Exception as e:
            logger.debug(f"[CounterfactualSim] _run error: {e}")

    def _analyse_interaction(
        self,
        pcm:       Any,
        wsdm:      Optional[Any],
        pcm_rec:   Any,
        wsdm_by_n: Dict[int, Any],
    ) -> Optional[CounterfactualQuery]:
        """Build a CounterfactualQuery for one past interaction."""
        try:
            actual_action  = pcm_rec.action_type
            state_before   = pcm_rec.state_before
            actual_outcome = pcm_rec.outcome
            interaction_n  = getattr(pcm_rec, "interaction_n", 0)

            wsdm_rec      = wsdm_by_n.get(interaction_n)
            ctx_before    = (
                getattr(wsdm_rec, "context_before", None)
                if wsdm_rec else None
            )

            branches: Dict[str, CounterfactualBranch] = {}
            for action_type in ALL_ACTION_TYPES:
                branch = self._predict_branch(
                    pcm, wsdm, action_type, state_before, ctx_before,
                    was_actual=(action_type == actual_action),
                )
                branches[action_type] = branch

            # Rank branches by joint_score (highest = best)
            ranked = sorted(
                branches.values(), key=lambda b: b.joint_score, reverse=True
            )
            actual_rank = next(
                (i + 1 for i, b in enumerate(ranked) if b.was_actual), len(ranked)
            )
            # Best alternative: highest-scoring branch that isn't the actual
            best_alt_branch = next(
                (b for b in ranked if not b.was_actual), None
            )
            best_alternative = best_alt_branch.action_type if best_alt_branch else None
            worst_actual = (actual_rank == len(ranked))

            return CounterfactualQuery(
                interaction_n    = interaction_n,
                actual_action    = actual_action,
                actual_outcome   = actual_outcome,
                branches         = branches,
                best_alternative = best_alternative,
                actual_rank      = actual_rank,
                worst_actual     = worst_actual,
            )

        except Exception as e:
            logger.debug(f"[CounterfactualSim] _analyse_interaction error: {e}")
            return None

    def _predict_branch(
        self,
        pcm:          Any,
        wsdm:         Optional[Any],
        action_type:  str,
        state_before: List[float],
        ctx_before:   Optional[List[float]],
        was_actual:   bool,
    ) -> CounterfactualBranch:
        """Get PCM and WSDM predictions for one hypothetical action type."""

        # ── PCM prediction ──────────────────────────────────────────────────
        pcm_energy   = None
        pcm_conf_raw = None
        pcm_conf_cal = None
        try:
            pcm_pred = pcm._predict(action_type, state_before)
            if pcm_pred is not None:
                pcm_energy   = round(float(pcm_pred.energy_delta), 4)
                pcm_conf_raw = round(float(pcm_pred.confidence), 4)
                if self._calibration_engine:
                    cal_val, is_cal = self._calibration_engine.calibrated_confidence(
                        "pcm", pcm_conf_raw
                    )
                    pcm_conf_cal = round(cal_val, 4) if is_cal else None
        except Exception as e:
            logger.debug(f"[CounterfactualSim] PCM predict error ({action_type}): {e}")

        # ── WSDM prediction ─────────────────────────────────────────────────
        wsdm_engage  = None
        wsdm_conf_raw = None
        wsdm_conf_cal = None
        try:
            wsdm_ready = (wsdm is not None and ctx_before is not None and
                          len(getattr(wsdm, "_records", [])) >= 6)
            if wsdm_ready:
                wsdm_pred = wsdm._predict(action_type, ctx_before)
                if wsdm_pred is not None:
                    wsdm_engage   = round(
                        float(wsdm_pred.world_deltas.get("engagement_signal", 0.0)), 4
                    )
                    wsdm_conf_raw = round(float(wsdm_pred.confidence), 4)
                    if self._calibration_engine:
                        cal_val, is_cal = self._calibration_engine.calibrated_confidence(
                            "wsdm", wsdm_conf_raw
                        )
                        wsdm_conf_cal = round(cal_val, 4) if is_cal else None
        except Exception as e:
            logger.debug(f"[CounterfactualSim] WSDM predict error ({action_type}): {e}")

        # ── Joint score: weighted combination, calibrated where available ───
        eff_pcm_conf  = pcm_conf_cal if pcm_conf_cal is not None else (pcm_conf_raw or 0.5)
        eff_wsdm_conf = wsdm_conf_cal if wsdm_conf_cal is not None else (wsdm_conf_raw or 0.5)
        pcm_term  = (pcm_energy or 0.0)   * eff_pcm_conf  * PCM_ENERGY_WEIGHT
        wsdm_term = (wsdm_engage or 0.0)  * eff_wsdm_conf * WSDM_ENGAGEMENT_WEIGHT
        joint_score = round(pcm_term + wsdm_term, 5)

        return CounterfactualBranch(
            action_type    = action_type,
            pcm_energy_delta = pcm_energy,
            pcm_confidence   = pcm_conf_raw,
            pcm_calibrated   = pcm_conf_cal,
            wsdm_engagement  = wsdm_engage,
            wsdm_confidence  = wsdm_conf_raw,
            wsdm_calibrated  = wsdm_conf_cal,
            joint_score      = joint_score,
            was_actual        = was_actual,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                self._log = d.get("queries", [])
        except Exception as e:
            logger.debug(f"[CounterfactualSim] load error: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"queries": self._log, "last_written": time.time()}
            _tmp = self._path.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(payload, indent=2))
            _tmp.replace(self._path)
        except Exception as e:
            logger.debug(f"[CounterfactualSim] save error: {e}")
