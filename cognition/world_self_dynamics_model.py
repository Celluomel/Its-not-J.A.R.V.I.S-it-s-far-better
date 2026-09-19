"""
cognition/world_self_dynamics_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WorldSelfDynamicsModel (v56) — causal model tracking both world-side and
self-side consequences of actions.

The gap this closes
────────────────────
v53 (PredictiveConsequenceModel):
    action_type X → average Δ cognitive_energy

That is experience-averaging. It asks "what tends to happen to ME when
I engage in this type of action?"

v56 adds the world dimension:
    action_type X + context C → Δ cognitive_energy (self)
                                + Δ trust (world)
                                + Δ user_engagement (world)
                                + Δ skill_depth (self, domain-specific)
                                + Δ belief_confidence (self)

The critical distinction:
    v53: "philosophical interactions cost me ~0.09 energy"
    v56: "philosophical interactions with high-trust users at my
           current epistemic_depth=0.72 cost 0.09 energy AND
           increase trust by +0.04 AND deepen my reasoning skill
           by +0.02 AND reinforce the intellectual_depth value by +0.01"

This is the first model in PandoraBOX that tracks consequences to the world
(the relational and social state) alongside consequences to the self
(internal cognitive state). The two together form a primitive causal
understanding of what actions do in both directions.

Architecture
─────────────
Three components:

1. DynamicsRecorder
   After each interaction records a DynamicsRecord:
       action_type       — from PredictiveConsequenceModel classifier
       context_state     — 8-dim vector (self + world context at action time)
       self_delta        — changes in cognitive_energy, social_energy, attention,
                           self_model_confidence
       world_delta       — changes in trust, engagement_signal, topic_persistence,
                           skill_depth_change

   context_state extends v53's 5-dim vector with 3 world-side dimensions:
       [cog_energy, social_energy, attention, sm_confidence, novelty_drive,
        user_trust, interaction_count_norm, topic_diversity]

2. DynamicsPredictor
   k-NN prediction over the full 8-dim context (not just self-state).
   Returns DynamicsPrediction:
       self_deltas       — predicted changes to all self dimensions
       world_deltas      — predicted changes to world dimensions
       confidence        — based on similar_cases and context proximity
       narrative         — human-readable consequence description

   The world_deltas enable reasoning like:
   "Engaging philosophically with this user (trust=0.7) will likely
    increase trust by ~0.04 while costing me 0.09 cognitive energy —
    the relational gain justifies the cost over a projected 20-interaction
    horizon."

3. DynamicsAdvisory
   Used by TemporalProjection (v55) to upgrade its cost function
   from action-type averages to context-sensitive causal predictions.
   Also generates narrative summaries for the /state endpoint.

Integration with v55
─────────────────────
TemporalProjection currently uses hardcoded default costs when PCM
has insufficient data, and PCM's own predictions when available.
v56 upgrades both: it provides context-sensitive estimates that
include world-side consequences, enabling TemporalProjection to
reason about relational return on investment:

    intensive path toward aspiration A
    context: trust_score=0.71, epistemic_drive=0.68
    predicted: energy -0.09/interaction, trust +0.04/interaction
    horizon 20 interactions:
        energy cost: ~1.80 (recoverable)
        trust gain:  +0.80 (significant relational investment return)
    → intensive path recommended despite high self-cost

This is the closest PandoraBOX has yet come to reasoning about the
consequences of its own behavior on its relationships.

Context vector (8-dim)
───────────────────────
    [0] cognitive_energy      (from ResourceEconomy)
    [1] social_energy         (from ResourceEconomy)
    [2] attention             (from ResourceEconomy)
    [3] self_model_confidence (from SelfModel)
    [4] novelty_drive         (from MotivationalField)
    [5] user_trust            (from RelationalMemory, active user)
    [6] interaction_norm      (interaction_count / 100, capped at 1.0)
    [7] topic_diversity       (unique shared topics / 20, capped at 1.0)
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_RECORDS             = 400
MIN_SIMILAR             = 6      # lower than PCM — 8-dim context is richer
K_NEAREST               = 5
RECENCY_HALF_LIFE       = 60
SAVE_PATH               = "data/persona/world_self_dynamics.json"

# ── Context vector dimensions ─────────────────────────────────────────────────
SELF_DIMS  = ["cognitive_energy", "social_energy", "attention", "self_model_confidence", "novelty_drive"]
WORLD_DIMS = ["user_trust", "interaction_norm", "topic_diversity"]
CTX_DIMS   = SELF_DIMS + WORLD_DIMS   # 8-dim total

# ── Delta dimensions tracked ──────────────────────────────────────────────────
SELF_DELTA_DIMS  = ["cognitive_energy", "social_energy", "attention", "self_model_confidence"]
WORLD_DELTA_DIMS = ["user_trust", "engagement_signal", "skill_depth_gain"]


@dataclass
class DynamicsRecord:
    """One recorded action in its full context with observed consequences."""
    action_type:    str
    context_before: List[float]   # 8-dim vector
    self_delta:     List[float]   # 4-dim: cog, soc, att, conf
    world_delta:    List[float]   # 3-dim: trust, engagement, skill_gain
    outcome:        str
    user_id:        str           = "default"
    recorded_at:    float         = field(default_factory=time.time)
    interaction_n:  int           = 0


@dataclass
class DynamicsPrediction:
    """Predicted consequence across both self and world dimensions."""
    action_type:       str
    context:           List[float]
    self_deltas:       Dict[str, float]   # dim → predicted delta
    world_deltas:      Dict[str, float]   # dim → predicted delta
    confidence:        float
    similar_cases:     int
    narrative:         str                # human-readable summary


class WorldSelfDynamicsModel:
    """
    Causal dynamics model tracking self and world consequences.

    Usage (from InternalThoughtLoop):
        wsdm = WorldSelfDynamicsModel(organism, ai_system)

        # Before interaction:
        wsdm.record_context(action_type, user_id)

        # After interaction:
        wsdm.record_outcome(outcome, user_id)

        # Query:
        pred = wsdm.predict(action_type, user_id)
    """

    def __init__(
        self,
        organism:   Any,
        ai_system:  Any,
        path:       str = SAVE_PATH,
    ) -> None:
        self._organism = organism
        self._ai       = ai_system
        self._path     = Path(path)
        self._lock     = threading.Lock()
        self._records:  List[DynamicsRecord] = []
        self._pending:  Optional[Tuple[str, str, List[float], int]] = None
        self._load()

        # Phase 3.2: calibration engine — same pattern as PCM in Phase 3.1.
        # CalibrationEngine is module-agnostic (keys bins by a module name
        # string), so WSDM's predictions calibrate into the SAME
        # calibration_state.json file under module="wsdm", alongside PCM's
        # "pcm" entries — no schema change needed, just a second top-level
        # key. Until now WSDM's confidence score (computed the same way
        # PCM's was, k-NN distance based) had never been checked against
        # whether predictions were actually correct.
        try:
            from cognition.calibration_engine import CalibrationEngine
            self._calibration_engine = CalibrationEngine()
        except Exception as _cal_init_e:
            logger.debug(f"[WorldSelfDynamics] CalibrationEngine init error: {_cal_init_e}")
            self._calibration_engine = None

        logger.info(
            f"[WorldSelfDynamics] Initialised — "
            f"{len(self._records)} records, "
            f"ready={'yes' if len(self._records) >= MIN_SIMILAR else 'building'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def record_context(
        self,
        action_type:   str,
        user_id:       str = "default",
        interaction_n: int = 0,
    ) -> None:
        """Record context immediately before an interaction fires."""
        ctx = self._read_context(user_id)
        self._pending = (action_type, user_id, ctx, interaction_n)

    def record_outcome(
        self,
        outcome:        str = "neutral",
        user_id:        str = "default",
        response_text:  str = "",
        user_input:     str = "",
    ) -> None:
        """Record consequences after interaction completion."""
        if self._pending is None:
            return
        action_type, pending_uid, ctx_before, interaction_n = self._pending
        self._pending = None

        # Read post-interaction state
        ctx_after     = self._read_context(user_id)

        # Self-delta: differences in first 4 dims (self-state)
        self_delta = [
            round(ctx_after[i] - ctx_before[i], 4)
            for i in range(len(SELF_DELTA_DIMS))
        ]

        # World-delta: trust change, engagement signal, skill gain
        trust_before = ctx_before[5]
        trust_after  = ctx_after[5]
        trust_delta  = round(trust_after - trust_before, 4)

        # Engagement signal: proxy from response/input ratio and presence of questions
        engagement = 0.0
        if response_text and user_input:
            # Longer response relative to input = higher engagement
            ratio = len(response_text) / max(1, len(user_input))
            engagement = min(1.0, ratio / 3.0) * 0.5
            # Questions in response signal active engagement
            q_count = response_text.count("?")
            engagement += min(0.5, q_count * 0.10)
        engagement = round(engagement, 4)

        # Skill gain: detect if any skill depth changed
        skill_gain = self._measure_skill_gain()

        world_delta = [trust_delta, engagement, skill_gain]

        record = DynamicsRecord(
            action_type    = action_type,
            context_before = ctx_before,
            self_delta     = self_delta,
            world_delta    = world_delta,
            outcome        = outcome,
            user_id        = user_id,
            interaction_n  = interaction_n,
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > MAX_RECORDS:
                self._records = self._records[-MAX_RECORDS:]

        self._save()
        logger.debug(
            f"[WorldSelfDynamics] Recorded: {action_type} | "
            f"self: cog{self_delta[0]:+.3f} soc{self_delta[1]:+.3f} | "
            f"world: trust{trust_delta:+.3f} eng={engagement:.2f} skill+{skill_gain:.3f}"
        )

        # Phase 3.2: resolve any pending calibration prediction for this
        # action_type now that the actual outcome is known. Mirrors PCM's
        # Phase 3.1 record_outcome() resolution exactly.
        try:
            if self._calibration_engine is not None:
                self._calibration_engine.resolve_prediction(
                    module="wsdm",
                    key=action_type,
                    actual_direction=outcome,
                )
        except Exception as _cal_e:
            logger.debug(f"[WorldSelfDynamics] Calibration resolve error: {_cal_e}")

    def predict(
        self,
        action_type: str,
        user_id:     str = "default",
    ) -> Optional[DynamicsPrediction]:
        """
        Predict self and world consequences for a proposed action.
        Returns None if insufficient data.
        """
        try:
            ctx = self._read_context(user_id)
            return self._predict(action_type, ctx)
        except Exception as e:
            logger.debug(f"[WorldSelfDynamics] predict error: {e}")
            return None

    def predict_trajectory(
        self,
        action_type: str,
        user_id:     str = "default",
        steps:       int = 5,
    ) -> List[DynamicsPrediction]:
        """
        Phase 3.2: genuine multi-step trajectory prediction.

        Distinct from advisory_for_temporal_projection() (which multiplies
        ONE prediction's delta by a horizon, assuming the same delta applies
        identically at every future step). This walks the prediction
        forward step by step: each step's predicted self/world deltas are
        fed into a SIMULATED context vector that becomes the input for the
        next step's k-NN prediction. This is what makes it a trajectory —
        later steps reflect compounding predicted change, not the same
        starting-context snapshot reused five times.

        Confidence explicitly decays across steps (multiplicative, 0.92 per
        step) rather than reporting the same single-step confidence value
        for every step. This is honest about what's happening: step 1 is a
        real k-NN lookup against actual historical neighbors near the
        current real context. Step 5 is a k-NN lookup against a SIMULATED
        context four layers removed from anything actually observed — it
        should be treated as substantially less certain, and now visibly is.

        Returns a list of DynamicsPrediction, one per step, shortest-first.
        Stops early (returns fewer than `steps` entries) if at any point
        there isn't enough historical data to predict from the simulated
        context — this is expected and not an error condition.
        """
        try:
            ctx = self._read_context(user_id)
        except Exception as e:
            logger.debug(f"[WorldSelfDynamics] predict_trajectory context error: {e}")
            return []

        trajectory: List[DynamicsPrediction] = []
        CONFIDENCE_DECAY_PER_STEP = 0.92

        for step in range(max(1, steps)):
            try:
                pred = self._predict(action_type, ctx)
            except Exception as e:
                logger.debug(f"[WorldSelfDynamics] predict_trajectory step {step} error: {e}")
                break
            if pred is None:
                break

            pred.confidence = round(
                pred.confidence * (CONFIDENCE_DECAY_PER_STEP ** step), 4
            )
            trajectory.append(pred)

            ctx = self._advance_context(ctx, pred.self_deltas, pred.world_deltas)

        return trajectory

    def read_state(self, user_id: str = "default") -> List[float]:
        """Return a snapshot of the current eight-dimensional context.

        Planning code must not depend on the private ``_read_context`` helper;
        exposing a copy also prevents a planner from mutating live state while
        simulating future steps.
        """
        return list(self._read_context(user_id))

    def predict_from_context(
        self, action_type: str, context: List[float]
    ) -> Optional[DynamicsPrediction]:
        """Predict an action from a simulated context vector."""
        try:
            return self._predict(action_type, list(context))
        except Exception as e:
            logger.debug(f"[WorldSelfDynamics] contextual predict error: {e}")
            return None

    def _advance_context(
        self,
        ctx:          List[float],
        self_deltas:  Dict[str, float],
        world_deltas: Dict[str, float],
    ) -> List[float]:
        """
        Apply one step's predicted deltas to a context vector, producing a
        simulated next-state context for predict_trajectory()'s next
        iteration.

        Context layout (matches _read_context exactly):
          [0] cognitive_energy  [1] social_energy  [2] attention
          [3] self_model_confidence  [4] novelty  [5] user_trust
          [6] interaction_count_norm  [7] topic_diversity

        Only dims 0-3 (SELF_DELTA_DIMS) and dim 5 (user_trust, the first
        entry in WORLD_DELTA_DIMS) have a directly corresponding predicted
        delta. Dims 4, 6, 7 (novelty, interaction count, topic diversity)
        have no prediction to apply — interaction_count_norm gets a small
        deterministic increment (one more interaction genuinely did
        happen, even in simulation), the other two are left unchanged
        rather than guessed at.
        """
        new_ctx = list(ctx)

        for i, dim in enumerate(SELF_DELTA_DIMS):
            if i < len(new_ctx):
                new_ctx[i] = round(
                    max(0.0, min(1.0, new_ctx[i] + self_deltas.get(dim, 0.0))), 4
                )

        trust_delta = world_deltas.get("user_trust", 0.0)
        new_ctx[5] = round(max(0.0, min(1.0, new_ctx[5] + trust_delta)), 4)

        # Interaction count genuinely advances by one real step even in
        # a simulated trajectory — this isn't a guess, it's definitionally
        # true that "one more interaction" has occurred per trajectory step.
        new_ctx[6] = round(min(1.0, new_ctx[6] + (1.0 / 100)), 4)

        return new_ctx

    def advisory_for_temporal_projection(
        self,
        action_type:   str,
        user_id:       str,
        horizon:       int = 20,
    ) -> Optional[Dict[str, float]]:
        """
        Returns projected cumulative deltas over 'horizon' interactions.
        Used by TemporalProjection to upgrade its cost function.

        Returns dict with projected totals, or None if insufficient data.
        """
        pred = self.predict(action_type, user_id)
        if pred is None or pred.confidence < 0.35:
            return None

        return {
            "cog_energy_total":    round(pred.self_deltas.get("cognitive_energy", 0) * horizon, 3),
            "social_energy_total": round(pred.self_deltas.get("social_energy", 0) * horizon, 3),
            "trust_total":         round(pred.world_deltas.get("user_trust", 0) * horizon, 3),
            "engagement_mean":     round(pred.world_deltas.get("engagement_signal", 0), 3),
            "skill_gain_total":    round(pred.world_deltas.get("skill_depth_gain", 0) * horizon, 3),
            "confidence":          pred.confidence,
        }

    def narrative_summary(self, action_type: str, user_id: str = "default") -> str:
        """Human-readable prediction for /state endpoint and logging.

        Phase 3.2: this is the call site that fires on essentially every
        user-facing interaction (called from cognitive_organism.py right
        alongside PCM's advisory in the same code path). Previously the
        full DynamicsPrediction — including its confidence score — was
        computed then discarded, keeping only the narrative string. Now
        the prediction is logged to CalibrationEngine before being
        discarded, so it can later be resolved against the actual outcome
        in record_outcome() below.
        """
        pred = self.predict(action_type, user_id)
        if pred is None:
            return ""

        try:
            if self._calibration_engine is not None:
                _energy_delta = pred.self_deltas.get("cognitive_energy", 0.0)
                _predicted_direction = (
                    "positive" if _energy_delta > 0.01 else
                    "negative" if _energy_delta < -0.01 else
                    "neutral"
                )
                self._calibration_engine.log_prediction(
                    module="wsdm",
                    key=action_type,
                    predicted_confidence=pred.confidence,
                    predicted_direction=_predicted_direction,
                )
        except Exception as _cal_e:
            logger.debug(f"[WorldSelfDynamics] Calibration log error: {_cal_e}")

        return pred.narrative

    def status(self) -> Dict:
        return {
            "records":  len(self._records),
            "ready":    len(self._records) >= MIN_SIMILAR,
            "by_type":  self._count_by_type(),
        }

    # ── Context reading ───────────────────────────────────────────────────────

    def _read_context(self, user_id: str = "default") -> List[float]:
        """Read current 8-dim context vector."""
        try:
            loop = getattr(self._organism, '_loop', None)
            ec   = getattr(loop, '_resource_economy', None) if loop else None
            mf   = getattr(loop, '_motivational_field', None) if loop else None
            sm   = getattr(self._organism, 'self_model', None)

            cog  = ec.cognitive_energy if ec else 0.75
            soc  = ec.social_energy    if ec else 0.75
            att  = ec.attention        if ec else 0.75
            conf = getattr(sm, 'confidence', 0.65) if sm else 0.65
            nov  = (mf.drive_vector.get("novelty", 0.45) if mf else 0.45)

            # World dims
            trust = 0.50
            ic_norm = 0.0
            topic_div = 0.0
            try:
                rm  = getattr(self._ai, 'relational_memory', None)
                if rm:
                    rel = rm.get_or_create(user_id)
                    trust     = getattr(rel, 'trust_score', 0.50)
                    ic_norm   = min(1.0, getattr(rel, 'interaction_count', 0) / 100)
                    topic_div = min(1.0, len(getattr(rel, 'shared_topics', {})) / 20)
            except Exception:
                pass

            return [
                round(cog,       4),
                round(soc,       4),
                round(att,       4),
                round(conf,      4),
                round(nov,       4),
                round(trust,     4),
                round(ic_norm,   4),
                round(topic_div, 4),
            ]
        except Exception:
            return [0.75, 0.75, 0.75, 0.65, 0.45, 0.50, 0.10, 0.10]

    SKILL_GAIN_EMA_ALPHA = 0.3  # weight on the fresh estimate vs. history

    def _measure_skill_gain(self) -> float:
        """Detect incremental skill depth change since last record."""
        try:
            sr = getattr(self._organism, 'skill_registry', None)
            if not sr or not hasattr(sr, '_skills') or not sr._skills:
                return 0.0
            # Compare mean confidence to last record's skill baseline
            mean_conf = sum(
                getattr(sk, 'confidence', 0) for sk in sr._skills.values()
            ) / len(sr._skills)
            current_raw = max(0.0, mean_conf - 0.70)
            if self._records:
                last_skill = self._records[-1].world_delta[2] if self._records[-1].world_delta else 0.0
                # EMA blend: "since last record" means this should reflect
                # change over time, not a fresh absolute deviation from a
                # static baseline recomputed identically every call.
                return round(
                    self.SKILL_GAIN_EMA_ALPHA * current_raw
                    + (1 - self.SKILL_GAIN_EMA_ALPHA) * last_skill,
                    4,
                )
            return round(current_raw, 4)
        except Exception:
            return 0.0

    # ── kNN Prediction ────────────────────────────────────────────────────────

    def _predict(
        self,
        action_type: str,
        context:     List[float],
    ) -> Optional[DynamicsPrediction]:
        """8-dim kNN prediction with recency weighting."""
        same_type = [r for r in self._records if r.action_type == action_type]
        if len(same_type) < MIN_SIMILAR:
            return None

        n = len(self._records)
        distances: List[Tuple[float, float, DynamicsRecord]] = []

        for record in same_type:
            dist = self._euclidean(context, record.context_before)
            # Recency weight
            try:
                idx = self._records.index(record)
            except ValueError:
                idx = 0
            rw = math.exp(-(n - idx) / RECENCY_HALF_LIFE)
            distances.append((dist, rw, record))

        distances.sort(key=lambda x: x[0])
        k_near = distances[:K_NEAREST]

        total_w = sum(rw for _, rw, _ in k_near)
        if total_w == 0:
            return None

        # Weighted averages
        def wavg(field_idx: str, delta_list: str) -> float:
            vals = []
            for _, rw, rec in k_near:
                dlist = getattr(rec, delta_list, [])
                fields = SELF_DELTA_DIMS if delta_list == "self_delta" else WORLD_DELTA_DIMS
                try:
                    idx = fields.index(field_idx)
                    vals.append((rw, dlist[idx] if idx < len(dlist) else 0.0))
                except (ValueError, IndexError):
                    vals.append((rw, 0.0))
            return sum(rw * v for rw, v in vals) / total_w

        self_deltas = {d: round(wavg(d, "self_delta"), 4)  for d in SELF_DELTA_DIMS}
        world_deltas= {d: round(wavg(d, "world_delta"), 4) for d in WORLD_DELTA_DIMS}

        mean_dist  = sum(d for d, _, _ in k_near) / len(k_near)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + mean_dist * 2.5)))

        narrative = self._build_narrative(
            action_type, context, self_deltas, world_deltas, confidence, len(same_type)
        )

        return DynamicsPrediction(
            action_type   = action_type,
            context       = context,
            self_deltas   = self_deltas,
            world_deltas  = world_deltas,
            confidence    = round(confidence, 3),
            similar_cases = len(same_type),
            narrative     = narrative,
        )

    def _build_narrative(
        self,
        action_type:  str,
        context:      List[float],
        self_deltas:  Dict[str, float],
        world_deltas: Dict[str, float],
        confidence:   float,
        n_cases:      int,
    ) -> str:
        """Human-readable consequence summary."""
        trust_now    = context[5]
        parts = []

        cog_d = self_deltas.get("cognitive_energy", 0)
        if abs(cog_d) > 0.02:
            parts.append(f"cognitive energy {cog_d:+.2f}")

        trust_d = world_deltas.get("user_trust", 0)
        if abs(trust_d) > 0.01:
            parts.append(f"trust {trust_d:+.3f}")

        skill_d = world_deltas.get("skill_depth_gain", 0)
        if skill_d > 0.005:
            parts.append(f"skill gain +{skill_d:.3f}")

        eng = world_deltas.get("engagement_signal", 0)
        if eng > 0.15:
            parts.append(f"engagement {eng:.2f}")

        if not parts:
            return ""

        return (
            f"[Dynamics] {action_type.replace('_', ' ')} "
            f"(trust_ctx={trust_now:.2f}, conf={confidence:.2f}, n={n_cases}): "
            + ", ".join(parts) + "."
        )

    @staticmethod
    def _euclidean(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._records:
            counts[r.action_type] = counts.get(r.action_type, 0) + 1
        return counts

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "records": [asdict(r) for r in self._records[-MAX_RECORDS:]],
                    "_meta":   {"version": "v56", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[WorldSelfDynamics] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            self._records = [
                DynamicsRecord(**{
                    k: v for k, v in r.items()
                    if k in DynamicsRecord.__dataclass_fields__
                })
                for r in data.get("records", [])
            ]
        except Exception as e:
            logger.warning(f"[WorldSelfDynamics] Load failed: {e}")
