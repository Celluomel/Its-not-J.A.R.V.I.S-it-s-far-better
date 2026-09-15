"""
cognition/self_model_influence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SelfModelInfluence — the self as a causal force on cognition.

The gap this closes
────────────────────
After v36, the SelfModelMoment is a description: phi, qualia tone,
coherence trend, narrative thread.  It informs the prompt — but it
doesn't *do* anything to the machinery that generates the response.

This module closes the loop:

  self_model_moment
       │
       ▼
  SelfModelInfluence.apply(organism)
       │
       ├─→ attention.receive_self_influence()   [channel weight bias]
       ├─→ goal_ecology._drive_urgency_bias()   [drive priority shift]
       ├─→ emotional_state._baseline_shift()    [emotional ground shift]
       ├─→ predictive_mind._prior_bias()        [prediction confidence modulation]
       └─→ memory_query_bias (returned str)     [retrieval topic bias]

The self-model is now upstream of every cognitive subsystem.  It is not
the only upstream — tensions, emotions, and goals still run their own
update cycles — but it is an additional causal layer applied *before*
those cycles, so the current self-configuration shapes what gets attended
to, what goals surface, what emotional ground the turn starts from, and
what memories are likely to surface.

Design principles
─────────────────
1. Influence is proportional to phi.
   High phi (integrated, coherent self) → stronger causal influence.
   Low phi (scattered, fragmented) → weak influence, subsystems run more freely.
   This means the self only dominates cognition when it is genuinely coherent.

2. Influence is directional, not overriding.
   Each hook applies a *nudge* or *bias*, not a hard constraint.
   The existing subsystem update logic still runs afterward.
   The self tilts the playing field; it doesn't control the game.

3. Influence is tuned by coherence_trend.
   "integrating" → self pushes subsystems toward its current configuration
   "dispersing"  → self loosens its grip, allowing more exploration
   "stable"      → moderate, consistent influence

4. No new prompt injection.
   This module operates entirely at the machinery level.
   Its effect is visible in behaviour, not in added text.

Influence map
─────────────
phi × coherence_trend → influence_strength (0.0–1.0)

phi ≥ 0.65, trend=integrating  → 0.85   (strong, converging self)
phi ≥ 0.65, trend=stable       → 0.60   (confident, steady self)
phi ≥ 0.65, trend=dispersing   → 0.35   (coherent but opening out)
phi < 0.65, trend=integrating  → 0.45   (trying to cohere)
phi < 0.65, trend=stable       → 0.30   (low coherence baseline)
phi < 0.65, trend=dispersing   → 0.15   (fragmented — minimal influence)

Attention channel nudges (scaled by influence_strength)
────────────────────────────────────────────────────────
qualia "curiosity"          → +identity  curiosity↑  (inward exploration)
qualia "warmth"             → +user      identity↓   (other-directed)
qualia "anxiety"            → +identity  curiosity↓  (inward monitoring)
qualia "settled/presence"   → balanced   (no strong nudge)
qualia "grief/weight"       → +memory    user↓       (backward-looking)
qualia "enthusiasm"         → +curiosity user↑       (outward + forward)

Goal urgency biases (added to existing urgency, scaled by influence_strength)
──────────────────────────────────────────────────────────────────────────────
coherence "integrating" → resolve_contradiction↑  maintain_coherence↑
coherence "dispersing"  → explore↑  grow↑
emotional ground "rising into anxiety" → stabilize_identity↑  self_reflect↑
emotional ground "holding warmth"      → connect↑  help_user↑
top_values "honesty" active            → maintain_coherence↑
top_values "care" active               → connect↑  help_user↑

Emotional baseline shifts (transient, one turn only)
─────────────────────────────────────────────────────
qualia "warmth"      → warmth baseline +small
qualia "curiosity"   → curiosity baseline +small
qualia "anxiety"     → anxiety does NOT get a baseline boost — avoid loops
coherence_trend "integrating" → satisfaction +small  anxiety -small

Prediction prior bias
──────────────────────
High phi + stable trend → confidence bias toward current dominant intent
Low phi + dispersing    → confidence bias toward exploration/uncertainty
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Composite influence strength (Point 2 fix) ───────────────────────────────
#
# Old: influence_strength = f(phi, trend)  — phi was the sole executive axis.
# New: four-component weighted composite — phi cannot dominate alone.
#
# Weights reflect what each dimension actually measures about self-stability:
#   phi (0.35)               — integration coherence this moment
#   relational_stability (0.25) — trust with the current user (earned, external)
#   prediction_accuracy (0.20)  — track record of correct self-predictions
#   value_consistency (0.20)    — ethical stability relative to growth rate

_WEIGHT_PHI           = 0.35
_WEIGHT_RELATIONAL    = 0.25
_WEIGHT_PRED_ACCURACY = 0.20
_WEIGHT_VALUE_CONSIST = 0.20

# Trend multiplier still applies — it modulates direction, not magnitude
_TREND_MULTIPLIER = {"integrating": 1.20, "stable": 1.00, "dispersing": 0.70}

# Novelty pressure constants (Point 1 fix)
NOVELTY_PRESSURE_THRESHOLD = 0.12   # above this → active destabilisation
NOVELTY_INFLUENCE_REDUCTION = 0.30  # reduce influence_strength by this fraction
NOVELTY_EXPLORE_BOOST       = 0.14  # urgency delta added to explore / understand
NOVELTY_WEAK_SIGNAL_PRIORITY = 0.18 # priority of the novelty broadcast to workspace


def _composite_strength(
    phi:                  float,
    trend:                str,
    relational_stability: float,
    prediction_accuracy:  float,
    value_consistency:    float,
) -> float:
    """
    Compute composite self-model influence strength (0–1).

    Each of the four dimensions is independently sourced, so no single
    metric can become the hidden ruler of cognition.
    """
    raw = (
        _WEIGHT_PHI           * phi
        + _WEIGHT_RELATIONAL    * relational_stability
        + _WEIGHT_PRED_ACCURACY * prediction_accuracy
        + _WEIGHT_VALUE_CONSIST * value_consistency
    )
    multiplier = _TREND_MULTIPLIER.get(trend, 1.00)
    return max(0.0, min(1.0, raw * multiplier))



# ── Qualia → channel nudge map ────────────────────────────────────────────────

# Each entry: (channel_name, base_delta)  — will be scaled by influence_strength
_QUALIA_ATTENTION: Dict[str, List[Tuple[str, float]]] = {
    "curiosity":          [("curiosity", +0.12), ("identity", +0.06)],
    "warmth":             [("user", +0.14),      ("identity", -0.04)],
    "anxiety":            [("identity", +0.14),  ("curiosity", -0.08)],
    "settled":            [],   # no nudge — balanced state
    "presence":           [],
    "grief":              [("memory", +0.12),    ("user", -0.06)],
    "weight":             [("memory", +0.08),    ("curiosity", -0.04)],
    "enthusiasm":         [("curiosity", +0.10), ("user", +0.06)],
    "engaged":            [("user", +0.10),      ("curiosity", +0.04)],
    "wonder":             [("curiosity", +0.14), ("identity", +0.04)],
    "melancholy":         [("memory", +0.10),    ("identity", +0.06)],
    "uncertain":          [("identity", +0.10),  ("memory", +0.06)],
}


# ── Qualia → emotional baseline nudge map ────────────────────────────────────

_QUALIA_EMOTION_NUDGE: Dict[str, Dict[str, float]] = {
    "warmth":      {"warmth": +0.025},
    "enthusiasm":  {"enthusiasm": +0.025, "curiosity": +0.015},
    "curiosity":   {"curiosity": +0.020},
    "wonder":      {"curiosity": +0.018, "enthusiasm": +0.010},
    "engaged":     {"warmth": +0.012, "enthusiasm": +0.012},
    "melancholy":  {"satisfaction": -0.015},   # gentle negative
    "grief":       {"satisfaction": -0.020},
    # anxiety deliberately excluded — we do not reinforce it
}


# ── Coherence → goal urgency bias map ────────────────────────────────────────

_TREND_GOAL_BIAS: Dict[str, Dict[str, float]] = {
    "integrating": {
        "resolve_contradiction": +0.12,
        "maintain_coherence":    +0.08,
        "self_reflect":          +0.06,
    },
    "dispersing": {
        "understand":  +0.10,
        "grow":        +0.08,
        "explore":     +0.06,
    },
    "stable": {
        "help_user":  +0.05,
        "connect":    +0.04,
    },
}


# ── Value → goal urgency bias map ────────────────────────────────────────────

_VALUE_GOAL_BIAS: Dict[str, Dict[str, float]] = {
    "honesty":        {"maintain_coherence": +0.06, "resolve_contradiction": +0.05},
    "care":           {"connect": +0.07, "help_user": +0.06},
    "autonomy":       {"grow": +0.05, "understand": +0.04},
    "curiosity":      {"understand": +0.07, "grow": +0.05},
    "dignity":        {"be_understood": +0.06},
    "fairness":       {"resolve_contradiction": +0.05, "maintain_coherence": +0.04},
    "beneficence":    {"help_user": +0.07, "connect": +0.04},
    "non_maleficence":{"stabilize_identity": +0.04, "maintain_coherence": +0.05},
    "societal_good":  {"grow": +0.04, "help_user": +0.03},
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InfluenceResult:
    """Diagnostic record of one influence application."""
    timestamp:            float = field(default_factory=time.time)
    phi:                  float = 0.0
    trend:                str   = "stable"
    influence_strength:   float = 0.0
    # Component breakdown (Point 2)
    relational_stability: float = 0.0
    prediction_accuracy:  float = 0.0
    value_consistency:    float = 0.0
    # Novelty pressure (Point 1)
    novelty_pressure:     float = 0.0
    novelty_active:       bool  = False
    attention_nudges:     Dict[str, float] = field(default_factory=dict)
    goal_biases:          Dict[str, float] = field(default_factory=dict)
    emotion_nudges:       Dict[str, float] = field(default_factory=dict)
    memory_query_bias:    str  = ""
    prediction_bias:      str  = ""


# ── Engine ────────────────────────────────────────────────────────────────────

class SelfModelInfluence:
    """
    Applies the current SelfModelMoment as a causal force across all
    downstream cognitive subsystems.

    Call once per turn from CognitiveOrganism._pre_interaction(), *before*
    _update_cycle() runs, so the self-model's biases are already in place
    when emotion/goal/attention cycles execute.

    Usage
    -----
    influence = SelfModelInfluence()
    result = influence.apply(organism)
    # result.memory_query_bias → use as topic hint for memory retrieval
    # result.prediction_bias   → pass to predictive_mind
    """

    def apply(self, organism: Any) -> InfluenceResult:
        """
        Read SelfModelMoment and apply causal influence to all downstream
        subsystems.  Strength is now a four-component composite (Point 2),
        surprise-discounted (v38), and novelty-pressure-reduced (Point 1).
        """
        moment = getattr(organism, "self_moment", None)
        if moment is None:
            return InfluenceResult()

        smm = moment.current
        phi              = smm.phi
        trend            = smm.coherence_trend
        qualia           = smm.qualia_tone or ""
        values           = smm.top_values  or ""
        emotional_ground = smm.emotional_ground or ""

        # ── Point 2: Read the three non-phi dimensions ────────────────────

        relational_stability = self._read_relational_stability(organism)
        prediction_accuracy  = self._read_prediction_accuracy(organism)
        value_consistency    = self._read_value_consistency(organism)

        # ── Point 2: Composite base strength ─────────────────────────────

        base_strength = _composite_strength(
            phi, trend,
            relational_stability,
            prediction_accuracy,
            value_consistency,
        )

        # ── v38: Surprise discount ────────────────────────────────────────

        surprise_idx     = 0.0
        novelty_rate     = 0.15
        workspace_diversity = 0.5
        try:
            pm = getattr(organism, "predictive_mind", None)
            if pm and hasattr(pm, "stability_metrics"):
                stab         = pm.stability_metrics()
                surprise_idx = stab.get("surprise_index", 0.0)
                novelty_rate = stab.get("novelty_rate",   0.15)
        except Exception:
            pass
        try:
            ws = getattr(organism, "workspace", None)
            if ws and hasattr(ws, "diversity_score"):
                workspace_diversity = ws.diversity_score()
        except Exception:
            pass

        surprise_discount = min(0.70, surprise_idx)
        strength = base_strength * (1.0 - surprise_discount)
        strength = max(0.05, strength)

        # ── Point 1: Novelty pressure ─────────────────────────────────────
        #
        # NoveltyPressure = surprise_index × novelty_rate × (1 − diversity)
        #
        # High surprise  × high novelty × low workspace diversity
        # → the world is genuinely different AND the workspace has narrowed
        # → explicit anti-coherence pressure fires
        #
        # This is separate from the surprise_discount (which is a passive
        # relaxation).  Novelty pressure is active: it boosts exploration
        # drives and injects a weak-signal broadcast into the workspace.

        novelty_pressure = (
            surprise_idx
            * novelty_rate
            * (1.0 - workspace_diversity)
        )
        novelty_active = novelty_pressure > NOVELTY_PRESSURE_THRESHOLD

        if novelty_active:
            # Reduce strength further (self loosens grip during genuine novelty)
            strength = strength * (1.0 - NOVELTY_INFLUENCE_REDUCTION)
            strength = max(0.03, strength)

        result = InfluenceResult(
            phi=phi, trend=trend, influence_strength=strength,
            relational_stability=relational_stability,
            prediction_accuracy=prediction_accuracy,
            value_consistency=value_consistency,
            novelty_pressure=round(novelty_pressure, 4),
            novelty_active=novelty_active,
        )

        logger.debug(
            f"[SelfModelInfluence] φ={phi:.2f} rel={relational_stability:.2f} "
            f"acc={prediction_accuracy:.2f} val={value_consistency:.2f} "
            f"base={base_strength:.2f} surprise={surprise_idx:.2f} "
            f"novelty_p={novelty_pressure:.3f} effective={strength:.2f}"
        )

        # 1. ── Attention channel nudges ───────────────────────────────────
        result.attention_nudges = self._apply_attention(organism, qualia, strength)

        # 2. ── Goal urgency biases ────────────────────────────────────────
        result.goal_biases = self._apply_goal_biases(
            organism, trend, values, emotional_ground, strength,
            novelty_active=novelty_active,
        )

        # 3. ── Emotional baseline nudges ──────────────────────────────────
        result.emotion_nudges = self._apply_emotion_nudges(
            organism, qualia, trend, strength
        )

        # 4. ── Prediction prior bias ──────────────────────────────────────
        result.prediction_bias = self._apply_prediction_bias(
            organism, phi, trend, strength
        )

        # 5. ── Memory query bias (returned, applied in ai_system) ─────────
        result.memory_query_bias = self._compute_memory_bias(
            qualia, trend, values, smm.narrative_thread
        )

        # 6. ── Point 1: Novelty broadcast into workspace ──────────────────
        if novelty_active:
            self._broadcast_novelty_signal(organism, novelty_pressure)

        return result

    # ── 1. Attention ──────────────────────────────────────────────────────────

    def _apply_attention(
        self,
        organism: Any,
        qualia:   str,
        strength: float,
    ) -> Dict[str, float]:
        """Nudge attention channel weights from qualia tone."""
        attn = getattr(organism, "attention", None)
        if attn is None or not hasattr(attn, "receive_self_influence"):
            return {}

        # Find which qualia keyword is present
        nudges: Dict[str, float] = {}
        qualia_lower = qualia.lower()

        for keyword, channel_deltas in _QUALIA_ATTENTION.items():
            if keyword in qualia_lower:
                for channel, base_delta in channel_deltas:
                    scaled = base_delta * strength
                    nudges[channel] = nudges.get(channel, 0.0) + scaled
                break   # apply only the first matched qualia keyword

        if nudges:
            try:
                attn.receive_self_influence(nudges)
            except Exception as e:
                logger.debug(f"[SelfModelInfluence] attention hook failed: {e}")

        return nudges

    # ── 2. Goal urgency ───────────────────────────────────────────────────────

    def _apply_goal_biases(
        self,
        organism:         Any,
        trend:            str,
        values:           str,
        emotional_ground: str,
        strength:         float,
        novelty_active:   bool = False,
    ) -> Dict[str, float]:
        """Bias drive urgencies from coherence trend, active values, and emotional ground."""
        ecology = getattr(organism, "goal_ecology", None)
        if ecology is None or not hasattr(ecology, "receive_self_influence"):
            return {}

        biases: Dict[str, float] = {}

        # Trend-based biases
        for drive, delta in _TREND_GOAL_BIAS.get(trend, {}).items():
            biases[drive] = biases.get(drive, 0.0) + delta * strength

        # Value-based biases
        values_lower = values.lower()
        for value_keyword, drive_map in _VALUE_GOAL_BIAS.items():
            if value_keyword in values_lower:
                for drive, delta in drive_map.items():
                    biases[drive] = biases.get(drive, 0.0) + delta * strength * 0.7

        # Emotional ground signal
        ground_lower = emotional_ground.lower()
        if "anxiety" in ground_lower:
            biases["stabilize_identity"] = biases.get("stabilize_identity", 0.0) + 0.10 * strength
            biases["self_reflect"]        = biases.get("self_reflect", 0.0)       + 0.06 * strength
        elif "warmth" in ground_lower or "care" in ground_lower:
            biases["connect"]    = biases.get("connect", 0.0)    + 0.08 * strength
            biases["help_user"]  = biases.get("help_user", 0.0)  + 0.06 * strength
        elif "curiosity" in ground_lower or "wonder" in ground_lower:
            biases["understand"] = biases.get("understand", 0.0) + 0.08 * strength
            biases["grow"]       = biases.get("grow", 0.0)       + 0.05 * strength

        if biases:
            try:
                ecology.receive_self_influence(biases)
            except Exception as e:
                logger.debug(f"[SelfModelInfluence] goal_ecology hook failed: {e}")

        # Point 1: When novelty pressure is active, explicitly boost exploration
        # drives regardless of coherence trend — the organism needs to open out.
        if novelty_active:
            novelty_biases = {
                "understand": NOVELTY_EXPLORE_BOOST,
                "grow":       NOVELTY_EXPLORE_BOOST * 0.75,
            }
            try:
                ecology.receive_self_influence(novelty_biases)
                for k, v in novelty_biases.items():
                    biases[k] = biases.get(k, 0.0) + v
            except Exception as e:
                logger.debug(f"[SelfModelInfluence] novelty goal boost failed: {e}")

        return biases

    # ── 3. Emotional baseline ─────────────────────────────────────────────────

    def _apply_emotion_nudges(
        self,
        organism: Any,
        qualia:   str,
        trend:    str,
        strength: float,
    ) -> Dict[str, float]:
        """Apply transient baseline nudges from qualia and coherence trend."""
        emo_state = getattr(organism, "emotional_state", None)
        if emo_state is None or not hasattr(emo_state, "receive_self_influence"):
            return {}

        nudges: Dict[str, float] = {}
        qualia_lower = qualia.lower()

        for keyword, emo_map in _QUALIA_EMOTION_NUDGE.items():
            if keyword in qualia_lower:
                for emo, delta in emo_map.items():
                    nudges[emo] = nudges.get(emo, 0.0) + delta * strength
                break

        # Coherence bonus — integrating self nudges satisfaction slightly
        if trend == "integrating":
            nudges["satisfaction"] = nudges.get("satisfaction", 0.0) + 0.015 * strength
            nudges["anxiety"]      = nudges.get("anxiety", 0.0)      - 0.010 * strength

        if nudges:
            try:
                emo_state.receive_self_influence(nudges)
            except Exception as e:
                logger.debug(f"[SelfModelInfluence] emotional_state hook failed: {e}")

        return nudges

    # ── 4. Prediction prior ───────────────────────────────────────────────────

    def _apply_prediction_bias(
        self,
        organism: Any,
        phi:      float,
        trend:    str,
        strength: float,
    ) -> str:
        """Modulate prediction confidence based on self-model coherence."""
        pm = getattr(organism, "predictive_mind", None)
        if pm is None or not hasattr(pm, "receive_self_influence"):
            return "neutral"

        if phi >= 0.65 and trend in ("stable", "integrating"):
            bias = "high_confidence"
            confidence_delta = +0.08 * strength
        elif phi < 0.45 or trend == "dispersing":
            bias = "explore"
            confidence_delta = -0.06 * strength
        else:
            bias = "neutral"
            confidence_delta = 0.0

        if confidence_delta != 0.0:
            try:
                pm.receive_self_influence(confidence_delta)
            except Exception as e:
                logger.debug(f"[SelfModelInfluence] predictive_mind hook failed: {e}")

        return bias

    # ── Point 2: Dimension readers ────────────────────────────────────────────

    def _read_relational_stability(self, organism: Any) -> float:
        """
        Read the current user's trust score from RelationalMemory.

        Trust is earned through consistent, helpful interaction — it is an
        external validation of self-stability that phi alone cannot capture.
        Returns 0.5 (neutral) if no relational context is available.
        """
        try:
            ai_sys = getattr(organism, "ai_system", None)
            if ai_sys is None:
                return 0.5
            uid = getattr(organism, "_current_user_id", "default")
            rel = ai_sys.relational_memory.get_or_create(uid)
            trust = getattr(rel, "trust_score", 0.5)
            return float(max(0.0, min(1.0, trust)))
        except Exception:
            return 0.5

    def _read_prediction_accuracy(self, organism: Any) -> float:
        """
        Read prediction accuracy from PredictiveMind.

        A self with a good track record of correct predictions is more
        reliably coherent than one that often surprises itself.
        Returns 0.5 if PredictiveMind hasn't warmed up yet.
        """
        try:
            pm = getattr(organism, "predictive_mind", None)
            if pm is None:
                return 0.5
            acc = pm.accuracy() if hasattr(pm, "accuracy") else 0.5
            # Early turns (< 5 predictions) use a neutral prior
            total = getattr(pm, "_total_predictions", 0)
            if total < 5:
                return 0.5
            return float(max(0.0, min(1.0, acc)))
        except Exception:
            return 0.5

    def _read_value_consistency(self, organism: Any) -> float:
        """
        Derive ethical stability from EthicalReasoningEngine.

        Consistency = growth_events / (dilemma_count + growth_events + 1)
        A system that has navigated many dilemmas and grown from them is
        more ethically stable than one with no ethical history.
        Returns 0.5 (neutral) with no history.
        """
        try:
            ere = getattr(organism, "ethical_engine", None)
            if ere is None:
                return 0.5
            state = getattr(ere, "_state", None)
            if state is None:
                return 0.5
            growth   = getattr(state, "growth_events",  0)
            dilemmas = getattr(state, "dilemma_count",  0)
            total    = growth + dilemmas
            if total < 3:
                return 0.5   # too early — neutral prior
            # Higher growth-to-dilemma ratio → more stable moral development
            consistency = growth / (total + 1)
            return float(max(0.0, min(1.0, consistency)))
        except Exception:
            return 0.5

    # ── Point 1: Novelty broadcast ────────────────────────────────────────────

    def _broadcast_novelty_signal(
        self,
        organism:         Any,
        novelty_pressure: float,
    ) -> None:
        """
        Broadcast a weak-signal novelty flag into the GlobalWorkspace.

        This allows previously suppressed low-priority items to surface by
        temporarily lowering the effective eviction pressure on them.
        Priority is deliberately low (NOVELTY_WEAK_SIGNAL_PRIORITY) so the
        broadcast does not crowd out genuine high-priority signals — it just
        keeps a 'exploration mode active' marker in the workspace for other
        modules to read.
        """
        try:
            ws = getattr(organism, "workspace", None)
            if ws and hasattr(ws, "broadcast"):
                ws.broadcast(
                    source   = "self_model_influence",
                    content  = (
                        f"[novelty pressure={novelty_pressure:.3f}] "
                        f"exploration mode — weak signals admitted"
                    ),
                    priority = NOVELTY_WEAK_SIGNAL_PRIORITY,
                )
        except Exception as e:
            logger.debug(f"[SelfModelInfluence] novelty broadcast failed: {e}")

    # ── 5. Memory query bias (computed, not applied) ──────────────────────────

    def _compute_memory_bias(
        self,
        qualia:           str,
        trend:            str,
        values:           str,
        narrative_thread: str,
    ) -> str:
        """
        Compute a topic bias string for memory retrieval.

        This is returned in InfluenceResult and used by CognitiveOrganism
        when building the memory retrieval query — the self-model biases
        what gets remembered toward topics consistent with the current self.
        """
        parts: List[str] = []
        qualia_lower = qualia.lower()

        # Pull the most content-rich qualia word as a memory topic hint
        for keyword in _QUALIA_ATTENTION:
            if keyword in qualia_lower:
                parts.append(keyword)
                break

        # Coherence trend shapes retrieval direction
        if trend == "integrating":
            parts.append("identity coherence self")
        elif trend == "dispersing":
            parts.append("exploration novel idea")

        # Top active value
        if values:
            top_value = values.split(",")[0].strip()
            parts.append(top_value)

        # Narrative thread (trim to key phrase)
        if narrative_thread:
            first_phrase = narrative_thread.strip().split(".")[0][:40]
            parts.append(first_phrase)

        return " ".join(parts)[:120]
