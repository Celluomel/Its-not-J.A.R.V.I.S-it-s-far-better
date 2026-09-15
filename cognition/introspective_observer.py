"""
cognition/introspective_observer.py  — Phase 3.5

IntrospectiveObserver: a genuine meta-cognitive process monitor.

Produces COMPUTED OBSERVATIONS from longitudinal data — not questions,
not LLM-generated introspection, but evidence-based findings like:

  "narrative_identity attention increased 42% over the last 120 cycles"
  "curiosity drive has declined from 0.66 to 0.63 over the last 7 days"
  "PCM predictions have been overconfident (ECE=0.27) for 30+ cycles"
  "emotional responses ranked worst in 6/10 recent counterfactual queries"

These observations are:
  - Grounded: computed from timestamped data, not asserted
  - Auditable: every observation carries the data slice it was computed from
  - Longitudinal: compare across cycles, not just the current snapshot
  - Structured: JSON rather than natural language, suitable for Phase 3.5+
    prompt injection and NarrativeCompression input

Relationship to existing modules:
  - CognitiveObservatory: tracks emergence metrics (CCS/RDI/GEI/IDX) in
    ObservatorySnapshots. Complementary — Observer reads different data
    sources (attention history, personality history, calibration, CF).
  - GroundedIntrospectionGenerator: generates self-QUESTIONS from current
    state. Observer generates OBSERVATIONS from longitudinal data. Different
    purpose, different output type.

Architecture (per Fred + collaborator Phase 3.5 redefinition):
  All upstream modules feed into the Observer, which produces
  meta_observations.json, which feeds NarrativeCompression, Dashboard,
  and self-question generation.

  Attention Engine → attention history →┐
  Calibration Engine ──────────────────→│
  Counterfactual Simulator ────────────→│ IntrospectiveObserver
  Personality History ─────────────────→│      ↓
  Emergence Metrics ───────────────────→┘ meta_observations.json
                                              ↓
                              NarrativeCompression / Dashboard / Prompts
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

META_OBS_PATH         = "data/persona/meta_observations.json"
ATTENTION_HIST_PATH   = "data/persona/cognitive_attention_history.json"
PERSONALITY_HIST_PATH = "data/persona/personality_history.json"
EMERGENCE_HIST_PATH   = "data/persona/emergence_metrics.json"

MAX_OBSERVATIONS      = 100    # keep most recent N in meta_observations.json
OBSERVER_INTERVAL     = 10     # consolidation cycles between observation runs
OBSERVATION_MIN_CYCLE = 20     # don't run until enough history exists
ATTENTION_WINDOW      = 30     # cycles to compare attention trends over
PERSONALITY_WINDOW    = 30     # personality_history entries to compare
                                # (was 50/~67h — lowered to 30/~24h to capture
                                #  meaningful drift within a realistic session window)
SIGNIFICANCE_THRESHOLD = 0.05  # minimum delta to report (avoids noise)
PERSONALITY_THRESHOLD  = 0.5   # % change threshold for personality observations
                                # (was 2.0% — real data shows most traits change
                                #  0.4-1.6% per 24h window; 2% was too aggressive)


@dataclass
class MetaObservation:
    """One computed longitudinal observation about Lumina's cognitive dynamics."""
    category:    str        # "attention" | "personality" | "calibration" |
                            # "counterfactual" | "emotion" | "emergence"
    observation: str        # human-readable finding
    metric:      str        # what was measured (e.g. "narrative_identity_attention")
    value_now:   float      # current value
    value_then:  float      # value N cycles/entries ago
    delta:       float      # value_now - value_then
    delta_pct:   float      # percentage change
    window_desc: str        # e.g. "last 30 cycles" or "last 7 days"
    significance: float     # abs(delta_pct) — for sorting by importance
    computed_at:  float = field(default_factory=time.time)
    slow_cycle:   int   = 0


class IntrospectiveObserver:
    """
    Computes longitudinal observations from accumulated cognitive data.
    Runs every OBSERVER_INTERVAL consolidation cycles (default ~100 min).
    """

    def __init__(self, organism: Any) -> None:
        self._organism = organism
        self._lock     = threading.Lock()
        self._observations: List[Dict] = []
        self._path     = Path(META_OBS_PATH)
        self._last_run_cycle = -OBSERVER_INTERVAL
        self._load()
        logger.info(
            f"[IntrospectiveObserver] Initialised — "
            f"{len(self._observations)} past observations"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Called every slow cycle from InternalLoop."""
        is_consolidation = (slow_cycle % 5 == 0 and slow_cycle > 0)
        if not is_consolidation:
            return
        if slow_cycle < OBSERVATION_MIN_CYCLE:
            return
        if slow_cycle - self._last_run_cycle < OBSERVER_INTERVAL:
            return

        threading.Thread(
            target=self._run,
            args=(slow_cycle,),
            daemon=True,
            name="introspective-observer",
        ).start()

    def latest_observations(self, n: int = 5,
                             category: Optional[str] = None) -> List[Dict]:
        """Return n most recent observations, optionally filtered by category."""
        with self._lock:
            obs = self._observations
            if category:
                obs = [o for o in obs if o.get("category") == category]
            return list(obs[-n:])

    def prompt_fragment(self, n: int = 3) -> str:
        """Return top-N most significant observations as a prompt fragment."""
        with self._lock:
            recent = sorted(
                self._observations[-20:],
                key=lambda o: o.get("significance", 0),
                reverse=True,
            )[:n]
        if not recent:
            return ""
        lines = [f"[Meta-observation] {o['observation']}" for o in recent]
        return "\n".join(lines)

    def status(self) -> Dict:
        with self._lock:
            total = len(self._observations)
            recent = self._observations[-5:] if self._observations else []
            by_cat: Dict[str, int] = {}
            for o in self._observations:
                c = o.get("category", "unknown")
                by_cat[c] = by_cat.get(c, 0) + 1
        return {
            "total_observations": total,
            "by_category":        by_cat,
            "last_run_cycle":     self._last_run_cycle if self._last_run_cycle >= 0 else None,
            "status":             "not_run_yet" if self._last_run_cycle < 0 else "ready",
            "recent":             recent,
        }

    # ── Observation pipeline ───────────────────────────────────────────────────

    def _run(self, slow_cycle: int) -> None:
        try:
            new_obs: List[MetaObservation] = []

            new_obs.extend(self._observe_attention(slow_cycle))
            new_obs.extend(self._observe_personality(slow_cycle))
            new_obs.extend(self._observe_calibration(slow_cycle))
            new_obs.extend(self._observe_counterfactual(slow_cycle))
            new_obs.extend(self._observe_emergence(slow_cycle))

            # Priority 3 (cognitive synthesis audit): close the
            # Attention→Prediction→Outcome→Calibration→Observer→Attention
            # loop for real, instead of only describing drift in prose for
            # the LLM to read. See _close_attention_loop() docstring for
            # the grounding that keeps this from being pure self-reinforcement.
            loop_obs = self._close_attention_loop(slow_cycle)
            if loop_obs:
                new_obs.append(loop_obs)

            if new_obs:
                with self._lock:
                    for o in new_obs:
                        self._observations.append(asdict(o))
                    if len(self._observations) > MAX_OBSERVATIONS:
                        self._observations = self._observations[-MAX_OBSERVATIONS:]
                self._save()
                logger.info(
                    f"[IntrospectiveObserver] Cycle {slow_cycle}: "
                    f"generated {len(new_obs)} observations"
                )
                for o in sorted(new_obs, key=lambda x: x.significance, reverse=True)[:3]:
                    logger.info(f"  → {o.observation}")

            self._last_run_cycle = slow_cycle

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] _run error: {e}")

    # ── Attention observations ─────────────────────────────────────────────────

    def _observe_attention(self, slow_cycle: int) -> List[MetaObservation]:
        """
        Compare current attention weights against weights ATTENTION_WINDOW
        cycles ago. Produces observations like:
        "narrative_identity attention increased 42% over the last 30 cycles"
        """
        obs: List[MetaObservation] = []
        try:
            hist_path = Path(ATTENTION_HIST_PATH)
            if not hist_path.exists():
                return obs
            history = json.loads(hist_path.read_text()).get("history", [])
            if len(history) < 2:
                return obs

            # Most recent entry
            current = history[-1]
            # Entry from ~ATTENTION_WINDOW cycles ago (or oldest available)
            lookback_idx = max(0, len(history) - ATTENTION_WINDOW)
            past = history[lookback_idx]

            current_weights = current.get("attention_weights", {})
            past_weights    = past.get("attention_weights", {})
            if not current_weights or not past_weights:
                return obs

            cycles_ago = current.get("slow_cycle", 0) - past.get("slow_cycle", 0)
            if cycles_ago <= 0:
                return obs
            window_desc = f"last {cycles_ago} cycles"

            for module, now_val in current_weights.items():
                then_val = past_weights.get(module, now_val)
                if then_val == 0:
                    continue
                delta     = round(now_val - then_val, 4)
                delta_pct = round((delta / then_val) * 100, 1)

                if abs(delta_pct) < 2.0:  # below 2% — attention weights diverge slowly
                    continue

                direction = "increased" if delta > 0 else "declined"
                obs.append(MetaObservation(
                    category     = "attention",
                    observation  = (
                        f"{module.replace('_',' ')} attention {direction} "
                        f"{abs(delta_pct):.1f}% over the {window_desc}"
                    ),
                    metric       = f"{module}_attention",
                    value_now    = round(now_val, 4),
                    value_then   = round(then_val, 4),
                    delta        = delta,
                    delta_pct    = delta_pct,
                    window_desc  = window_desc,
                    significance = abs(delta_pct),
                    slow_cycle   = slow_cycle,
                ))

            # Also observe bias drift (from Phase 2.9.1 learned_bias)
            current_bias = current.get("learned_bias", {})
            past_bias    = past.get("learned_bias", {})
            for module, now_bias in current_bias.items():
                then_bias = past_bias.get(module, 0.0)
                delta     = round(now_bias - then_bias, 5)
                if abs(delta) < 0.002:
                    continue
                direction = "strengthened" if delta > 0 else "weakened"
                obs.append(MetaObservation(
                    category     = "attention",
                    observation  = (
                        f"{module.replace('_',' ')} reinforcement bias {direction} "
                        f"by {abs(delta):.4f} over the {window_desc}"
                    ),
                    metric       = f"{module}_bias",
                    value_now    = round(now_bias, 5),
                    value_then   = round(then_bias, 5),
                    delta        = delta,
                    delta_pct    = round((delta / max(abs(then_bias), 0.001)) * 100, 1),
                    window_desc  = window_desc,
                    significance = abs(delta) * 100,
                    slow_cycle   = slow_cycle,
                ))

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] attention observe error: {e}")
        return obs

    def _close_attention_loop(self, slow_cycle: int) -> Optional[MetaObservation]:
        """
        Closes the Attention→Prediction→Outcome→Calibration→Observer→
        Attention loop with an actual numeric write, not just a sentence
        for the LLM to read.

        Why this needs real grounding, not just "attention drifted, so
        reinforce the drift": that would be pure self-reinforcement — a
        module could drift upward for no reason (noise) and this would
        amplify the noise, compounding it into a real bias with no
        connection to whether the drift was ever a good idea. That failure
        mode is exactly what this method is designed to avoid. Two
        independent gates must both pass before any write happens:

          1. TRUST GATE — calibration ECE for both PCM and WSDM must be
             below TRUST_ECE_THRESHOLD. If the system's own predictions
             aren't currently reliable, a trend observed during that period
             isn't trustworthy evidence of anything either.
          2. OUTCOME GATE — recent PCM ConsequenceRecord outcomes in the
             SAME window used for the attention-drift comparison must be
             net-positive or net-negative by a clear margin (not roughly
             50/50, which carries no signal). This is the actual "was the
             trend good or bad" evidence — grounded in real recorded
             outcomes, not in attention's own movement being used as its
             own justification.

        When both gates pass, reuses CognitiveAttentionEngine's EXISTING
        update_from_feedback() — the same Phase 2.9.1 REINFORCE-style
        credit assignment already used for per-interaction feedback — but
        called with the HISTORICAL weights snapshot from the start of the
        observation window, crediting or blaming whatever was elevated
        back then. This is a second, longer-window feedback pass layered
        on the same mechanism, not a new mutation pathway into learned_bias.
        """
        TRUST_ECE_THRESHOLD = 0.20
        OUTCOME_MARGIN      = 0.20   # required |positive - negative| fraction
        MIN_RECORDS         = 8      # need enough records for the margin to mean anything
        META_SIGNAL_SCALE   = 0.35   # meta-signal is weaker than a single interaction's ±1.0

        try:
            loop = getattr(self._organism, "_loop", None)
            if loop is None:
                return None

            attention_engine = getattr(loop, "_attention_engine", None)
            pcm = getattr(loop, "_consequence_model", None)
            cal = getattr(pcm, "_calibration_engine", None) if pcm else None
            if attention_engine is None or pcm is None or cal is None:
                return None

            # ── Trust gate ───────────────────────────────────────────────
            pcm_ece  = cal.compute_ece("pcm")
            wsdm_ece = cal.compute_ece("wsdm")
            if pcm_ece is None or wsdm_ece is None:
                return None  # not enough calibration data yet to trust anything
            if pcm_ece >= TRUST_ECE_THRESHOLD or wsdm_ece >= TRUST_ECE_THRESHOLD:
                return None  # predictions currently unreliable — don't act on a trend

            # ── Read the SAME attention-history window _observe_attention used ──
            hist_path = Path(ATTENTION_HIST_PATH)
            if not hist_path.exists():
                return None
            history = json.loads(hist_path.read_text()).get("history", [])
            if len(history) < 2:
                return None
            current = history[-1]
            lookback_idx = max(0, len(history) - ATTENTION_WINDOW)
            past = history[lookback_idx]
            past_weights = past.get("attention_weights", {})
            if not past_weights:
                return None
            window_start_ts = past.get("computed_at", 0)
            window_end_ts   = current.get("computed_at", time.time())

            # ── Outcome gate: recorded PCM outcomes within this exact window ──
            records = [
                r for r in getattr(pcm, "_records", [])
                if window_start_ts <= getattr(r, "recorded_at", 0) <= window_end_ts
            ]
            if len(records) < MIN_RECORDS:
                return None
            n_pos = sum(1 for r in records if r.outcome == "positive")
            n_neg = sum(1 for r in records if r.outcome == "negative")
            n_total = len(records)
            margin = (n_pos - n_neg) / n_total

            if abs(margin) < OUTCOME_MARGIN:
                return None  # not a clear enough signal either way

            signal = META_SIGNAL_SCALE if margin > 0 else -META_SIGNAL_SCALE

            # ── Apply: credit/blame whatever was elevated at window start ──
            attention_engine.update_from_feedback(
                signal=signal, weights_at_feedback=past_weights
            )

            direction = "reinforced" if signal > 0 else "dampened"
            top_module = max(past_weights, key=past_weights.get)
            return MetaObservation(
                category     = "attention",
                observation  = (
                    f"Closed the loop: {top_module.replace('_',' ')}-led attention "
                    f"from {ATTENTION_WINDOW} cycles ago {direction}, based on "
                    f"{n_pos} positive vs {n_neg} negative outcomes since "
                    f"(calibration trustworthy: PCM ECE={pcm_ece:.2f}, WSDM ECE={wsdm_ece:.2f})"
                ),
                metric       = "attention_loop_closure",
                value_now    = margin,
                value_then   = 0.0,
                delta        = margin,
                delta_pct    = round(margin * 100, 1),
                window_desc  = f"last {ATTENTION_WINDOW} cycles",
                significance = abs(margin) * 100,
                slow_cycle   = slow_cycle,
            )

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] close_attention_loop error: {e}")
            return None

    # ── Personality observations ───────────────────────────────────────────────

    def _observe_personality(self, slow_cycle: int) -> List[MetaObservation]:
        """
        Compare current personality traits against values PERSONALITY_WINDOW
        entries ago. 500 entries spans ~673 hours — can detect multi-day drift.
        """
        obs: List[MetaObservation] = []
        try:
            hist_path = Path(PERSONALITY_HIST_PATH)
            if not hist_path.exists():
                return obs
            history = json.loads(hist_path.read_text())
            if not isinstance(history, list) or len(history) < 2:
                return obs

            current  = history[-1]
            past_idx = max(0, len(history) - PERSONALITY_WINDOW)
            past     = history[past_idx]

            ts_now  = current.get("ts", time.time())
            ts_then = past.get("ts", ts_now)
            hours   = round((ts_now - ts_then) / 3600, 1)
            window_desc = f"last {hours}h"

            skip_keys = {"ts"}
            for trait, now_val in current.items():
                if trait in skip_keys or not isinstance(now_val, (int, float)):
                    continue
                then_val = past.get(trait)
                if then_val is None or then_val == 0:
                    continue
                delta     = round(float(now_val) - float(then_val), 4)
                delta_pct = round((delta / float(then_val)) * 100, 1)

                if abs(delta_pct) < PERSONALITY_THRESHOLD:
                    continue

                direction = "increased" if delta > 0 else "declined"
                obs.append(MetaObservation(
                    category     = "personality",
                    observation  = (
                        f"{trait.replace('_',' ')} {direction} "
                        f"from {then_val:.3f} to {now_val:.3f} "
                        f"({abs(delta_pct):.1f}%) over {window_desc}"
                    ),
                    metric       = trait,
                    value_now    = round(float(now_val), 4),
                    value_then   = round(float(then_val), 4),
                    delta        = delta,
                    delta_pct    = delta_pct,
                    window_desc  = window_desc,
                    significance = abs(delta_pct),
                    slow_cycle   = slow_cycle,
                ))

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] personality observe error: {e}")
        return obs

    # ── Calibration observations ───────────────────────────────────────────────

    def _observe_calibration(self, slow_cycle: int) -> List[MetaObservation]:
        """
        Report calibration ECE for PCM and WSDM, with a qualitative
        assessment. Observations only generated when bins have enough data.
        """
        obs: List[MetaObservation] = []
        try:
            loop = getattr(self._organism, "_loop", None)
            pcm  = getattr(loop, "_consequence_model", None) if loop else None
            cal  = getattr(pcm, "_calibration_engine", None) if pcm else None
            if cal is None:
                return obs

            for module in ("pcm", "wsdm"):
                ece = cal.compute_ece(module)
                if ece is None:
                    continue
                quality = (
                    "well-calibrated" if ece < 0.10 else
                    "slightly overconfident" if ece < 0.20 else
                    "significantly overconfident"
                )
                obs.append(MetaObservation(
                    category     = "calibration",
                    observation  = (
                        f"{module.upper()} prediction confidence is {quality} "
                        f"(ECE={ece:.3f}) — "
                        + ("predictions match actual accuracy well"
                           if ece < 0.10 else
                           "predicted confidence exceeds actual accuracy")
                    ),
                    metric       = f"{module}_ece",
                    value_now    = ece,
                    value_then   = 0.0,   # no historical ECE stored yet
                    delta        = ece,
                    delta_pct    = 0.0,
                    window_desc  = "current",
                    significance = ece * 100,  # higher ECE = more significant
                    slow_cycle   = slow_cycle,
                ))

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] calibration observe error: {e}")
        return obs

    # ── Counterfactual observations ────────────────────────────────────────────

    def _observe_counterfactual(self, slow_cycle: int) -> List[MetaObservation]:
        """
        Report persistent patterns in counterfactual analysis — which action
        types are systematically suboptimal, worst-actual streaks.
        """
        obs: List[MetaObservation] = []
        try:
            loop   = getattr(self._organism, "_loop", None)
            cf_sim = getattr(loop, "_counterfactual_sim", None) if loop else None
            if cf_sim is None:
                return obs

            status = cf_sim.status()
            streak = status.get("worst_actual_streak", 0)
            recent = status.get("recent", [])
            total  = status.get("total_queries", 0)

            if not recent or total < 5:
                return obs

            # Worst-actual streak
            if streak >= 3:
                obs.append(MetaObservation(
                    category     = "counterfactual",
                    observation  = (
                        f"Lumina has chosen the lowest-predicted action in "
                        f"{streak} consecutive interactions — "
                        f"consistently suboptimal action selection detected"
                    ),
                    metric       = "worst_actual_streak",
                    value_now    = float(streak),
                    value_then   = 0.0,
                    delta        = float(streak),
                    delta_pct    = 0.0,
                    window_desc  = f"last {streak} interactions",
                    significance = streak * 10.0,
                    slow_cycle   = slow_cycle,
                ))

            # Best alternative pattern
            from collections import Counter
            best_counts  = Counter(e.get("best_alternative") for e in recent
                                   if e.get("best_alternative"))
            worst_counts = Counter(e.get("actual_action") for e in recent
                                   if e.get("worst_actual"))
            if best_counts:
                top_best, top_count = best_counts.most_common(1)[0]
                pct = round(top_count / len(recent) * 100, 0)
                if pct >= 40:
                    obs.append(MetaObservation(
                        category     = "counterfactual",
                        observation  = (
                            f"'{top_best}' was the best predicted alternative "
                            f"in {int(pct)}% of recent interactions — "
                            f"consider shifting toward this framing"
                        ),
                        metric       = "best_alternative_frequency",
                        value_now    = top_count / len(recent),
                        value_then   = 0.0,
                        delta        = top_count / len(recent),
                        delta_pct    = 0.0,
                        window_desc  = f"last {len(recent)} interactions",
                        significance = pct,
                        slow_cycle   = slow_cycle,
                    ))

            if worst_counts:
                top_worst, worst_count = worst_counts.most_common(1)[0]
                worst_pct = round(worst_count / len(recent) * 100, 0)
                if worst_pct >= 40:
                    obs.append(MetaObservation(
                        category     = "counterfactual",
                        observation  = (
                            f"'{top_worst}' was the actual choice — and the "
                            f"worst-predicted alternative — in {int(worst_pct)}% "
                            f"of recent interactions — a recurring suboptimal pattern"
                        ),
                        metric       = "worst_actual_frequency",
                        value_now    = worst_count / len(recent),
                        value_then   = 0.0,
                        delta        = worst_count / len(recent),
                        delta_pct    = 0.0,
                        window_desc  = f"last {len(recent)} interactions",
                        significance = worst_pct,
                        slow_cycle   = slow_cycle,
                    ))

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] counterfactual observe error: {e}")
        return obs

    # ── Emergence observations ─────────────────────────────────────────────────

    def _observe_emergence(self, slow_cycle: int) -> List[MetaObservation]:
        """
        Report significant trends in emergence metrics (GEI, IDX, CCS, RDI)
        across the 200-entry history in emergence_metrics.json.
        """
        obs: List[MetaObservation] = []
        try:
            em_path = Path(EMERGENCE_HIST_PATH)
            if not em_path.exists():
                return obs
            em_data = json.loads(em_path.read_text())
            history = em_data.get("history", [])
            if len(history) < 10:
                return obs

            current = history[-1]
            past    = history[max(0, len(history) - 50)]  # ~50 entry lookback

            ts_now  = current.get("timestamp", time.time())
            ts_then = past.get("timestamp", ts_now)
            hours   = round((ts_now - ts_then) / 3600, 1)
            window_desc = f"last {hours}h"

            metric_labels = {
                "gei":  "Goal Emergence Index",
                "idx":  "Identity Drift Index",
                "csis": "Cross-System Integration Score",
                "wds":  "World-Dynamic Score",
            }
            for key, label in metric_labels.items():
                now_val  = current.get(key)
                then_val = past.get(key)
                if now_val is None or then_val is None or then_val == 0:
                    continue
                delta     = round(float(now_val) - float(then_val), 4)
                delta_pct = round((delta / float(then_val)) * 100, 1)
                if abs(delta_pct) < 5.0:
                    continue
                direction = "increased" if delta > 0 else "declined"
                obs.append(MetaObservation(
                    category     = "emergence",
                    observation  = (
                        f"{label} {direction} "
                        f"from {then_val:.3f} to {now_val:.3f} "
                        f"({abs(delta_pct):.1f}%) over {window_desc}"
                    ),
                    metric       = key,
                    value_now    = round(float(now_val), 4),
                    value_then   = round(float(then_val), 4),
                    delta        = delta,
                    delta_pct    = delta_pct,
                    window_desc  = window_desc,
                    significance = abs(delta_pct),
                    slow_cycle   = slow_cycle,
                ))

        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] emergence observe error: {e}")
        return obs

    # ── Persistence ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                self._observations = d.get("observations", [])
        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] load error: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "observations": self._observations,
                "last_written": time.time(),
            }
            _tmp = self._path.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(payload, indent=2))
            _tmp.replace(self._path)
        except Exception as e:
            logger.debug(f"[IntrospectiveObserver] save error: {e}")
