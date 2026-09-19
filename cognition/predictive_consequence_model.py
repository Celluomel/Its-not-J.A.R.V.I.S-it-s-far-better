"""
cognition/predictive_consequence_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PredictiveConsequenceModel (v53) — lightweight forward model that learns
statistical associations between action types and state changes, then
uses them to predict consequences before acting.

The gap this closes
────────────────────
The self-model currently records what happened.  "After that interaction,
cognitive_energy dropped by 0.08, social_energy recovered 0.03."
But PandoraBOX cannot reason: "If I engage deeply with this philosophical
question right now, I predict cognitive_energy will drop ~0.12, which
will put me in low-energy territory for the next two cycles — I should
note this affects available experimentation."

Predictive self-regulation requires a forward model.  Not a simulation
engine — a pattern store.  Learn from accumulated interactions what
consequences follow from what action types in what states.

Architecture
─────────────
Three components:

1. ConsequenceRecorder
   After every interaction and slow cycle, records a snapshot:
       action_type:      str   ("deep_reasoning"|"social"|"creative"|
                                "philosophical"|"technical"|"emotional")
   state_before:     Dict  (cognitive_energy, social_energy, attention,
                             self_model_confidence, dominant_drive)
   state_after:      Dict  (same fields, measured at next slow cycle)
   outcome:          str   ("positive"|"negative"|"neutral")
   delta:            Dict  (state_after - state_before per field)

   Stored in a rolling buffer of MAX_RECORDS interactions.

2. ConsequencePredictor
   Given a proposed action type and current state, queries the record
   store for similar (action_type, state_context) pairs and returns a
   predicted delta with confidence:

       prediction = model.predict(
           action_type="philosophical",
           current_state=state,
       )
       # → PredictionResult(
       #       energy_delta=-0.09,  confidence=0.72,
       #       social_delta=+0.02,  similar_cases=14,
       #       attention_delta=-0.05
       #   )

   Similarity is measured by state proximity (Euclidean distance on
   normalised [0,1] state vector).  Uses k=5 nearest neighbours from
   recent records weighted by recency.

3. PreemptiveAdvisory
   Before the LLM generates a response, predict_response_cost() is
   called with the inferred action type.  If the prediction indicates
   a consequence that would cross a critical threshold (cognitive_energy
   predicted to drop below 0.15, or attention below 0.20), a short
   advisory is appended to the system prompt:

       [Predictive note] This type of engagement (philosophical,
       confidence=0.68) typically reduces cognitive energy by ~0.09.
       Current level: 0.22. Post-engagement: ~0.13 (critical threshold).
       Consider scope-limiting the response.

   This is advisory, not blocking — the constraint layer (gate) still
   enforces hard limits.  The prediction gives the LLM context that
   was previously invisible: the consequence of its own effort.

Action type classification
───────────────────────────
Inferred from user input without LLM call (fast, keyword-based):
    philosophical:  abstract concepts, meaning, consciousness, existence
    technical:      code, implementation, architecture, debugging
    emotional:      personal situation, feelings, relationships
    creative:       writing, art, metaphor, story, imagine
    social:         casual, greeting, check-in, personal exchange
    analytical:     analysis, compare, evaluate, research, explain
    deep_reasoning: multi-step argument, reasoning chain, complex logic

State context
──────────────
State is read at action time and recorded as a 5-dim vector:
    [cognitive_energy, social_energy, attention,
     self_model_confidence, drive_novelty_level]

Minimum 8 similar records required before predictions are trusted.
Below that threshold the model returns None (no prediction).

Learning properties
────────────────────
The model learns continuously.  Early interactions produce no
predictions.  After ~20-30 interactions across action types, the model
develops a basic picture.  After 100+, predictions become reliable.
This is intentional — a cognitive organism should develop self-knowledge
from experience, not have it hardcoded.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_RECORDS             = 300    # rolling buffer of consequence records
MIN_SIMILAR_FOR_PREDICT = 8      # minimum similar records to trust prediction
K_NEAREST               = 5      # k-nearest-neighbours for prediction
RECENCY_HALF_LIFE       = 50     # records (not time) for recency weight decay
SAVE_PATH               = "data/persona/consequence_model.json"

# ── Action type keywords ──────────────────────────────────────────────────────
ACTION_KEYWORDS: Dict[str, List[str]] = {
    "philosophical":  [
        "consciousness", "meaning", "existence", "reality", "truth", "ethics",
        "free will", "identity", "purpose", "mind", "metaphysics", "philosophy",
        "what is", "what does it mean", "why do we",
    ],
    "technical":     [
        "code", "implement", "function", "class", "debug", "error", "algorithm",
        "architecture", "system", "database", "api", "script", "program",
        "build", "deploy", "configure",
    ],
    "emotional":     [
        "feel", "feeling", "worried", "anxious", "sad", "happy", "upset",
        "relationship", "friend", "family", "hurt", "scared", "lonely",
        "struggling", "difficult time", "going through",
    ],
    "creative":      [
        "write", "story", "poem", "imagine", "create", "art", "metaphor",
        "describe", "paint", "narrative", "fiction", "character", "scene",
    ],
    "social":        [
        "how are you", "what's up", "hello", "hi there", "good morning",
        "how's", "what do you think about", "your opinion", "do you like",
    ],
    "analytical":    [
        "analyze", "compare", "evaluate", "research", "explain", "why",
        "how does", "what causes", "effects of", "difference between",
        "pros and cons", "review",
    ],
    "deep_reasoning": [
        "step by step", "reason through", "argument for", "prove that",
        "demonstrate", "derive", "logical", "inference", "therefore",
        "because", "given that", "it follows",
    ],
}

# ── State fields tracked ──────────────────────────────────────────────────────
STATE_FIELDS = [
    "cognitive_energy",
    "social_energy",
    "attention",
    "self_model_confidence",
    "novelty_drive",
]


@dataclass
class ConsequenceRecord:
    """One recorded action → state_change pair."""
    action_type:  str
    state_before: List[float]   # 5-dim vector
    state_after:  List[float]   # 5-dim vector (recorded at next slow cycle)
    delta:        List[float]   # state_after - state_before
    outcome:      str           # "positive"|"negative"|"neutral"
    recorded_at:  float         = field(default_factory=time.time)
    interaction_n: int          = 0


@dataclass
class PredictionResult:
    """Predicted consequence of a proposed action."""
    action_type:     str
    energy_delta:    float        # predicted cognitive_energy change
    social_delta:    float        # predicted social_energy change
    attention_delta: float        # predicted attention change
    confidence:      float        # 0–1, based on similar_cases and state proximity
    similar_cases:   int
    will_cross_critical: bool     # True if predicted post-action state crosses threshold
    advisory:        str          # human-readable prediction for prompt injection


class PredictiveConsequenceModel:
    """
    Learns and predicts consequences of action types on PandoraBOX's
    cognitive state.

    Usage:
        pcm = PredictiveConsequenceModel(organism)
        pcm.record_action(action_type, state_before)   # before interaction
        pcm.record_outcome(state_after, outcome)        # after interaction

        # Before generating response:
        advisory = pcm.predict_and_advise(user_input, current_state)
        if advisory:
            prompt_additions += advisory
    """

    def __init__(self, organism: Any, path: str = SAVE_PATH) -> None:
        self._organism    = organism
        self._path        = Path(path)
        self._records:    List[ConsequenceRecord] = []
        self._pending:    Optional[Tuple[str, List[float], int]] = None
        self._interaction_count = 0
        self._load()

        # Phase 3.1: calibration engine, initialised here (not lazily inside
        # predict_and_advise) so record_outcome() can always resolve a
        # pending prediction even if it's called before predict_and_advise
        # has run once in this instance's lifetime.
        try:
            from cognition.calibration_engine import CalibrationEngine
            self._calibration_engine = CalibrationEngine()
        except Exception as _cal_init_e:
            logger.debug(f"[ConsequenceModel] CalibrationEngine init error: {_cal_init_e}")
            self._calibration_engine = None

        logger.info(
            f"[ConsequenceModel] Initialised — "
            f"{len(self._records)} records, "
            f"ready={'yes' if len(self._records) >= MIN_SIMILAR_FOR_PREDICT else 'building'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def record_action(
        self,
        action_type: str,
        state_before: List[float],
        interaction_n: int = 0,
    ) -> None:
        """Record the state immediately before an action fires."""
        self._pending = (action_type, state_before, interaction_n)

    def record_outcome(
        self,
        state_after: List[float],
        outcome: str = "neutral",
    ) -> None:
        """Record the state after action completion and store the record."""
        if self._pending is None:
            return
        action_type, state_before, interaction_n = self._pending
        self._pending = None

        delta = [
            round(a - b, 4)
            for a, b in zip(state_after, state_before)
        ]
        record = ConsequenceRecord(
            action_type    = action_type,
            state_before   = state_before,
            state_after    = state_after,
            delta          = delta,
            outcome        = outcome,
            interaction_n  = interaction_n,
        )
        self._records.append(record)
        if len(self._records) > MAX_RECORDS:
            self._records = self._records[-MAX_RECORDS:]

        self._save()
        logger.debug(
            f"[ConsequenceModel] Recorded: {action_type} → "
            f"energy Δ{delta[0]:+.3f} social Δ{delta[1]:+.3f} "
            f"att Δ{delta[2]:+.3f} ({outcome})"
        )

        # Phase 3.1: resolve any pending calibration prediction for this
        # action_type now that we know the actual outcome. This is what
        # closes the loop — predicted_confidence vs actual correctness.
        try:
            if self._calibration_engine is not None:
                self._calibration_engine.resolve_prediction(
                    module="pcm",
                    key=action_type,
                    actual_direction=outcome,
                )
        except Exception as _cal_e:
            logger.debug(f"[ConsequenceModel] Calibration resolve error: {_cal_e}")

    def predict_and_advise(
        self,
        user_input:    str,
        current_state: List[float],
    ) -> str:
        """
        Classify the action type from user_input, predict consequences,
        and return an advisory string if a critical threshold will be crossed.
        Returns empty string if no prediction or no concern.

        Phase 3.1: every prediction made here is logged to CalibrationEngine
        BEFORE the outcome is known — this is the only call site in the
        codebase that pairs naturally with record_outcome() for the same
        action_type, via self._pending set in record_action(). The previous
        behaviour silently discarded the confidence value whenever no
        critical threshold was crossed; predictions are now logged
        regardless, since calibration needs the full distribution of
        predictions, not just the alarming ones.
        """
        print(f"###PCM_DEBUG### predict_and_advise ENTRY user_input={user_input[:40]!r}", flush=True)
        try:
            action_type = self._classify_action(user_input)
            result      = self._predict(action_type, current_state)
            print(f"###PCM_DEBUG### classified={action_type} result_is_none={result is None}", flush=True)

            if result is None:
                logger.info(
                    f"[ConsequenceModel] predict_and_advise: no prediction for "
                    f"'{action_type}' (need {MIN_SIMILAR_FOR_PREDICT} same-type "
                    f"records, calibration log_prediction NOT reached this call)"
                )
                return ""

            # Phase 3.1: log this prediction for later calibration resolution.
            # 'direction' is derived from the predicted energy delta sign —
            # this is what record_outcome()'s actual outcome will be checked
            # against. Predictions with near-zero delta are logged as
            # "neutral" direction, matching how outcomes are classified
            # elsewhere in this file.
            try:
                if self._calibration_engine is not None:
                    _predicted_direction = (
                        "positive" if result.energy_delta > 0.01 else
                        "negative" if result.energy_delta < -0.01 else
                        "neutral"
                    )
                    self._calibration_engine.log_prediction(
                        module="pcm",
                        key=action_type,
                        predicted_confidence=result.confidence,
                        predicted_direction=_predicted_direction,
                    )
                    print(f"###PCM_DEBUG### logged pending pcm/{action_type}", flush=True)
                    # Diagnostic (v59): confirm this is actually reached and
                    # succeeding — calibration showed 0 resolved predictions
                    # across 192 real evaluations despite the wiring looking
                    # correct on inspection. This will confirm or rule out
                    # the logging side definitively on the next run.
                    logger.info(
                        f"[Calibration] logged pending pcm/{action_type} "
                        f"conf={result.confidence:.3f} dir={_predicted_direction} "
                        f"(pending queue now {len(self._calibration_engine._pending)})"
                    )
                else:
                    logger.warning(
                        "[Calibration] self._calibration_engine is None — "
                        "log_prediction NOT called this turn"
                    )
            except Exception as _cal_e:
                logger.warning(f"[ConsequenceModel] Calibration log error (was silent debug-level): {_cal_e}")

            if not result.will_cross_critical:
                return ""

            return result.advisory
        except Exception as e:
            print(f"###PCM_DEBUG### predict_and_advise EXCEPTION: {e!r}", flush=True)
            import traceback as _tb_pae
            print(_tb_pae.format_exc(), flush=True)
            # Bug fix (v59): this was logger.debug(), which is invisible at
            # the INFO level this deployment actually runs at. Every one of
            # my instrumented logs above sits INSIDE this try block, so if
            # _classify_action() or _predict() throws before reaching them,
            # NONE of that diagnostic logging ever fires — indistinguishable
            # from "this code path never runs at all". Given zero
            # [Calibration]/[ConsequenceModel] lines showed up across a run
            # with dozens of confirmed real interactions, this silent
            # catch-all is the far more likely explanation than a bug in
            # the log/resolve pairing itself. Bumped to warning with
            # traceback so the next run names the actual exception.
            import traceback
            logger.warning(
                f"[ConsequenceModel] predict_and_advise error: {e}\n"
                f"{traceback.format_exc()}"
            )
            return ""

    def predict(
        self, action_type: str, current_state: List[float]
    ) -> Optional[PredictionResult]:
        """Direct prediction for a given action type and state."""
        return self._predict(action_type, current_state)

    def classify(self, user_input: str) -> str:
        """Classify user input into an action type."""
        return self._classify_action(user_input)

    def read_state(self) -> List[float]:
        """Read current organism state as a 5-dim vector."""
        try:
            loop = getattr(self._organism, '_loop', None)
            ec   = getattr(loop, '_resource_economy', None) if loop else None
            sm   = getattr(self._organism, 'self_model', None)
            mf   = getattr(loop, '_motivational_field', None) if loop else None

            cog  = ec.cognitive_energy if ec else 0.75
            soc  = ec.social_energy    if ec else 0.75
            att  = ec.attention        if ec else 0.75
            conf = getattr(sm, 'confidence', 0.65) if sm else 0.65
            nov  = mf.drive_vector.get("novelty", 0.45) if mf else 0.45

            return [
                round(cog,  4),
                round(soc,  4),
                round(att,  4),
                round(conf, 4),
                round(nov,  4),
            ]
        except Exception:
            return [0.75, 0.75, 0.75, 0.65, 0.45]

    def summary(self) -> Dict:
        """Status for /state endpoint."""
        if not self._records:
            return {"records": 0, "ready": False}
        by_type: Dict[str, int] = {}
        for r in self._records:
            by_type[r.action_type] = by_type.get(r.action_type, 0) + 1
        return {
            "records": len(self._records),
            "ready":   len(self._records) >= MIN_SIMILAR_FOR_PREDICT,
            "by_type": by_type,
        }

    # ── Classification ────────────────────────────────────────────────────────

    def _classify_action(self, user_input: str) -> str:
        """Fast keyword-based classification. No LLM call."""
        text   = user_input.lower()
        scores: Dict[str, int] = {}
        for action_type, keywords in ACTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits:
                scores[action_type] = hits

        if not scores:
            return "social"   # default: low-cost conversational
        return max(scores, key=scores.get)

    # ── Prediction ────────────────────────────────────────────────────────────

    def _predict(
        self, action_type: str, current_state: List[float]
    ) -> Optional[PredictionResult]:
        """
        k-NN prediction using Euclidean distance on normalised state vector.
        Weights recent records more heavily.
        """
        # Filter to same action type
        same_type = [r for r in self._records if r.action_type == action_type]
        if len(same_type) < MIN_SIMILAR_FOR_PREDICT:
            return None

        # Compute distances
        distances: List[Tuple[float, float, ConsequenceRecord]] = []
        n_records = len(self._records)

        for i, record in enumerate(same_type):
            dist = self._euclidean(current_state, record.state_before)
            # Recency weight: most recent records weighted more
            recency_idx = self._records.index(record) if record in self._records else i
            recency_w   = math.exp(-(n_records - recency_idx) / RECENCY_HALF_LIFE)
            distances.append((dist, recency_w, record))

        # k nearest (weighted by recency)
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:K_NEAREST]

        if not k_nearest:
            return None

        # Weighted average delta
        total_weight = sum(rw for _, rw, _ in k_nearest)
        if total_weight == 0:
            return None

        delta_energy   = sum(rw * r.delta[0] for _, rw, r in k_nearest) / total_weight
        delta_social   = sum(rw * r.delta[1] for _, rw, r in k_nearest) / total_weight
        delta_attention = sum(rw * r.delta[2] for _, rw, r in k_nearest) / total_weight

        # Confidence: inverse of mean distance, scaled
        mean_dist  = sum(d for d, _, _ in k_nearest) / len(k_nearest)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + mean_dist * 3)))

        # Predict post-action state
        cog_after = current_state[0] + delta_energy
        att_after = current_state[2] + delta_attention

        # Critical threshold check
        CRIT_COG = 0.15
        CRIT_ATT = 0.20
        will_cross = (
            (current_state[0] > CRIT_COG and cog_after < CRIT_COG) or
            (current_state[2] > CRIT_ATT and att_after < CRIT_ATT)
        )

        # Build advisory
        advisory = ""
        if will_cross and confidence >= 0.45:
            concerns = []
            if current_state[0] > CRIT_COG and cog_after < CRIT_COG:
                concerns.append(
                    f"cognitive energy ({current_state[0]:.2f} → ~{cog_after:.2f}, "
                    f"below critical {CRIT_COG})"
                )
            if current_state[2] > CRIT_ATT and att_after < CRIT_ATT:
                concerns.append(
                    f"attention ({current_state[2]:.2f} → ~{att_after:.2f}, "
                    f"below critical {CRIT_ATT})"
                )
            advisory = (
                f"[Predictive note] {action_type.replace('_', ' ').capitalize()} "
                f"engagement (confidence={confidence:.2f}) typically reduces "
                + " and ".join(concerns)
                + f". Based on {len(k_nearest)} similar past interactions. "
                f"Scope-limiting may preserve capacity for remaining cycles."
            )

        return PredictionResult(
            action_type      = action_type,
            energy_delta     = round(delta_energy,    4),
            social_delta     = round(delta_social,    4),
            attention_delta  = round(delta_attention, 4),
            confidence       = round(confidence,      3),
            similar_cases    = len(same_type),
            will_cross_critical = will_cross,
            advisory         = advisory,
        )

    @staticmethod
    def _euclidean(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "records": [asdict(r) for r in self._records[-MAX_RECORDS:]],
                "_meta":   {"version": "v53", "ts": time.time()},
            }
            # Atomic write — dashboard reads this file from a background
            # thread on every refresh cycle; a non-atomic write here can be
            # caught mid-truncation, producing a JSONDecodeError that blanks
            # the entire dashboard render (render() has no per-panel recovery).
            _tmp = self._path.with_suffix('.json.tmp')
            with open(_tmp, "w") as f:
                json.dump(data, f, indent=2)
            _tmp.replace(self._path)
        except Exception as e:
            logger.debug(f"[ConsequenceModel] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data    = json.loads(self._path.read_text())
            records = data.get("records", [])
            self._records = [
                ConsequenceRecord(**{
                    k: v for k, v in r.items()
                    if k in ConsequenceRecord.__dataclass_fields__
                })
                for r in records
            ]
        except Exception as e:
            logger.warning(f"[ConsequenceModel] Load failed (empty): {e}")
