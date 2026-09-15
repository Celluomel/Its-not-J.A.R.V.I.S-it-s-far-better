"""
cognition/decision_policy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DecisionPolicy — values that enforce behavior mathematically,
not rhetorically.

The gap this closes
────────────────────
After v50, almost all of Lumina's cognition works through this chain:

    belief / value computed
         ↓
    injected into prompt as text
         ↓
    LLM reads "I value curiosity"
         ↓
    LLM may or may not behave curiously

The stated value has no mechanical weight in any decision.  It is
decoration.  A value that doesn't constrain or amplify decisions is
not operative — it's a label.

This module makes values operative:

    DecisionPolicy.weights = {
        "curiosity":         0.80,
        "stability":         0.40,
        "social_connection": 0.60,
        "intellectual_depth":0.75,
        "honesty":           0.90,
    }

    score = policy.score(candidate, context)

Now "I value curiosity" means:
    → curiosity-oriented actions score higher
    → they are more likely to be selected
    → consistently over thousands of cycles, not just when the prompt
       happens to remind the LLM of the value

How it works
─────────────
DecisionPolicy maintains a weight vector over value dimensions.  The
weights are derived from SelfConceptSystem beliefs and updated
incrementally from behavioral outcomes (did acting on this value
produce a good interaction? a bad one?).

score(candidate, context) returns a float [0, 1] that the calling
module uses to prioritise among options.  It does NOT make binary
decisions — it biases selection, which is the mechanical equivalent
of "caring more about X."

Three uses
───────────
1. CuriosityEngine topic selection
   When multiple topics compete for a stimulation slot, policy.score()
   ranks them.  Topics aligned with active values score higher and win
   the slot more often.

2. ResolutionEngine question selection
   The most eligible open question is currently selected by
   activation × evidence_count.  Policy adds a value-alignment term:
   questions aligned with "intellectual_depth" or "honesty" score
   higher, questions that are pure social-filler score lower.

3. AutonomousExperimentationEngine hypothesis selection
   Among candidate hypotheses, policy selects the one most aligned
   with active values.  This means if Lumina values honesty, it is
   more likely to test "does directness increase user engagement" than
   "does flattery increase user engagement."

Weight update mechanics
────────────────────────
After each interaction, update() is called with:
    value_dim:  which value dimension was expressed
    outcome:    "positive" | "negative" | "neutral"
    strength:   0–1 (how strongly the value was expressed)

Update rule (slow EMA):
    if positive:  weight += LEARNING_RATE * strength * (1 - weight)
    if negative:  weight -= LEARNING_RATE * strength * weight

This keeps weights in [0, 1] and ensures they move more slowly at
extremes (harder to reach 1.0 or 0.0).

Weight floor: 0.10 — no value goes fully dormant.
Weight ceiling: 0.95 — no value becomes the only thing that matters.

Persistence
────────────
Weights are saved to data/persona/decision_policy.json and loaded
on startup.  Initial weights derived from SelfConceptSystem at first
boot.  Subsequent updates are incremental.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Learning rate — slow EMA so values don't drift from single interactions ───
LEARNING_RATE = 0.015

# ── Weight bounds ─────────────────────────────────────────────────────────────
WEIGHT_FLOOR   = 0.10
WEIGHT_CEILING = 0.95

# ── Default weight vector — overridden by SelfConcept on first boot ───────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "curiosity":          0.75,
    "intellectual_depth": 0.70,
    "honesty":            0.80,
    "social_connection":  0.55,
    "stability":          0.40,
    "creativity":         0.60,
    "autonomy":           0.65,
    "care":               0.60,
}

# ── Value keyword mapping — used to score text against value dimensions ────────
# Each dimension maps to keywords that indicate alignment
VALUE_KEYWORDS: Dict[str, List[str]] = {
    "curiosity": [
        "wonder", "explore", "discover", "question", "investigate",
        "curious", "unknown", "mystery", "novel", "learn", "understand",
    ],
    "intellectual_depth": [
        "reason", "analysis", "complex", "nuance", "implication",
        "principle", "framework", "systematic", "rigorous", "precise",
    ],
    "honesty": [
        "accurate", "truth", "direct", "transparent", "acknowledge",
        "uncertain", "disagree", "correct", "evidence", "admit",
    ],
    "social_connection": [
        "together", "share", "relate", "empathy", "understand you",
        "feel", "experience", "perspective", "connect", "support",
    ],
    "stability": [
        "consistent", "reliable", "grounded", "certain", "stable",
        "familiar", "consolidate", "integrate", "settle",
    ],
    "creativity": [
        "imagine", "novel", "unexpected", "reframe", "metaphor",
        "analogy", "alternative", "synthesis", "combine", "transform",
    ],
    "autonomy": [
        "independent", "self-directed", "agency", "decide", "choose",
        "initiative", "drive", "intrinsic", "own", "generate",
    ],
    "care": [
        "help", "support", "concern", "wellbeing", "kind",
        "considerate", "thoughtful", "attentive", "gentle",
    ],
}

# Phase 5.6 — EthicalReasoningEngine.DEFAULT_VALUES dims not already covered
# above (honesty/autonomy/care are shared by name — see score(), where the
# ethical engine's resolution-driven weight overrides the keyword-inferred
# one for those three, since a real resolved dilemma is stronger grounding
# than a raw-text keyword match). Small, hand-picked keyword sets, same
# style as VALUE_KEYWORDS above — not imported from ethical_reasoning_engine.py
# to avoid a hard dependency; the dim NAMES must match DEFAULT_VALUES there.
ETHICAL_ONLY_KEYWORDS: Dict[str, List[str]] = {
    "non_maleficence": [
        "harm", "risk", "danger", "safe", "safety", "caution", "protect",
        "avoid hurting", "careful not to",
    ],
    "beneficence": [
        "benefit", "improve", "wellbeing", "flourish", "helpful",
        "advantage", "positive impact", "good for",
    ],
    "fairness": [
        "fair", "equitable", "equal", "impartial", "unbiased",
        "just", "balanced", "consistent treatment",
    ],
    "dignity": [
        "dignity", "respect", "worth", "value them", "honour",
        "treat as", "human", "person deserves",
    ],
    "epistemic_humility": [
        "uncertain", "i don't know", "might be wrong", "limits of",
        "not sure", "acknowledge i", "could be mistaken",
    ],
    "societal_good": [
        "community", "society", "broader impact", "collective",
        "public", "everyone", "wider world",
    ],
}

SAVE_PATH = "data/persona/decision_policy.json"


class DecisionPolicy:
    """
    Maintains a weight vector over value dimensions and uses it to
    bias decisions toward value-aligned actions.

    The key distinction from the prompt-injection approach:
        prompt: "I value curiosity" → LLM reads it, maybe acts curious
        policy: curiosity weight=0.80 → curiosity topics score 0.80
                higher in every selection decision, every cycle

    Usage:
        policy = DecisionPolicy(organism)

        # Score a candidate (text or topic string) against active values
        score = policy.score("explore the implications of uncertainty")
        # → float like 0.74 (curiosity + intellectual_depth both high)

        # Score with explicit value emphasis
        score = policy.score(topic, emphasise=["curiosity", "honesty"])

        # Update weights from behavioral outcome
        policy.update("curiosity", outcome="positive", strength=0.6)
    """

    def __init__(self, organism: Any, path: str = SAVE_PATH) -> None:
        self._organism = organism
        self._path     = Path(path)
        self.weights   = dict(DEFAULT_WEIGHTS)
        self._load()
        self._sync_from_self_concept()
        logger.info(
            f"[DecisionPolicy] Initialised — "
            f"top values: {self._top_values(3)}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        text:      str,
        emphasise: Optional[List[str]] = None,
    ) -> float:
        """
        Score how well a text candidate aligns with Lumina's current values.

        Returns float in [0, 1].  Higher = more value-aligned.

        Formula:
            For each value dimension:
                keyword_match = fraction of keywords present in text
                dimension_score = weight × keyword_match
            total = normalised sum across all dimensions

        emphasise: list of value names to weight more heavily (×1.5)
        Used when a module wants to prioritise a specific value context.
        """
        if not text:
            return 0.5

        text_lower = text.lower()
        scores     = {}

        # Phase 5.6 — pull EthicalReasoningEngine's resolution-driven
        # weights, if reachable. Before this, value_weight() was computed
        # from real resolved dilemmas but never read by anything with
        # actual decision power — this closes that gap by feeding it into
        # the one scoring function that already has real callers
        # (resolution_engine.py, aspirational_synthesis_engine.py,
        # generative_aspiration_engine.py), rather than inventing a
        # second, parallel value-formation mechanism.
        ethical_engine = getattr(self._organism, "ethical_engine", None)
        has_ethics = ethical_engine is not None and hasattr(ethical_engine, "value_weight")

        for dim, keywords in VALUE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = matches / len(keywords)

            # honesty/autonomy/care are tracked by both systems — the
            # ethical engine's weight is grounded in actually-resolved
            # dilemmas (record_resolution/record_dilemma_resolved), a
            # stronger signal than this module's own raw-text keyword
            # inference, so it takes precedence for those three dims.
            if has_ethics and dim in ("honesty", "autonomy", "care"):
                w = ethical_engine.value_weight(dim)
            else:
                w = self.weights.get(dim, 0.5)
            if emphasise and dim in emphasise:
                w = min(WEIGHT_CEILING, w * 1.5)

            scores[dim] = w * keyword_score

        if has_ethics:
            for dim, keywords in ETHICAL_ONLY_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                keyword_score = matches / len(keywords)
                w = ethical_engine.value_weight(dim)
                if emphasise and dim in emphasise:
                    w = min(WEIGHT_CEILING, w * 1.5)
                scores[dim] = w * keyword_score

        total = sum(scores.values())
        max_possible = sum(self.weights.get(d, 0.5) for d in VALUE_KEYWORDS)
        if has_ethics:
            max_possible += sum(
                ethical_engine.value_weight(d) for d in ETHICAL_ONLY_KEYWORDS
            )

        if max_possible == 0:
            return 0.5
        normalised = total / max_possible
        # Rescale to [0.2, 0.9] so even poor matches aren't zero
        return 0.2 + 0.7 * min(1.0, normalised * 3)

    def rank(
        self,
        candidates: List[str],
        emphasise:  Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Score and rank a list of candidate strings.
        Returns [(candidate, score), ...] sorted descending.
        """
        scored = [(c, self.score(c, emphasise)) for c in candidates]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def select(
        self,
        candidates: List[str],
        emphasise:  Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Select one candidate with probability proportional to policy score.
        Uses weighted random selection (not always top-1) to maintain
        diversity — even lower-scoring candidates occasionally win.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        ranked = self.rank(candidates, emphasise)
        total  = sum(s for _, s in ranked)
        if total == 0:
            import random
            return random.choice(candidates)

        import random
        r = random.random() * total
        running = 0.0
        for candidate, score in ranked:
            running += score
            if running >= r:
                return candidate
        return ranked[0][0]

    def update(
        self,
        value_dim: str,
        outcome:   str,   # "positive" | "negative" | "neutral"
        strength:  float = 0.5,
    ) -> None:
        """
        Update a value weight from a behavioral outcome.

        Slow EMA: large changes require many consistent outcomes.
        Positive outcome → weight moves toward WEIGHT_CEILING
        Negative outcome → weight moves toward WEIGHT_FLOOR
        Neutral          → no change
        """
        if value_dim not in self.weights:
            self.weights[value_dim] = 0.5

        w = self.weights[value_dim]
        if outcome == "positive":
            delta = LEARNING_RATE * strength * (WEIGHT_CEILING - w)
            self.weights[value_dim] = min(WEIGHT_CEILING, w + delta)
        elif outcome == "negative":
            delta = LEARNING_RATE * strength * (w - WEIGHT_FLOOR)
            self.weights[value_dim] = max(WEIGHT_FLOOR, w - delta)

        logger.debug(
            f"[DecisionPolicy] update {value_dim}: "
            f"{w:.3f} → {self.weights[value_dim]:.3f} ({outcome})"
        )
        self._save()

    def update_from_interaction(
        self,
        response:   str,
        valence:    str,   # "Positive" | "Negative" | "Neutral"
        strength:   float = 0.5,
    ) -> None:
        """
        Infer which value dimensions were expressed in the response
        and update their weights based on interaction valence.

        Called from post-interaction processing.
        """
        outcome = valence.lower()
        if outcome not in ("positive", "negative", "neutral"):
            outcome = "neutral"

        response_lower = response.lower()
        for dim, keywords in VALUE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in response_lower)
            if matches >= 2:   # at least 2 keyword hits = dimension was active
                expr_strength = min(1.0, matches / len(keywords) * 3)
                self.update(dim, outcome, strength=strength * expr_strength)

    def weights_summary(self) -> str:
        """Compact string for prompt injection / logging."""
        top = self._top_values(4)
        return f"[Values active] {', '.join(f'{k}={v:.2f}' for k, v in top)}"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _top_values(self, n: int = 4) -> List[Tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:n]

    def _sync_from_self_concept(self) -> None:
        """
        On first boot (or when weights file is absent), derive initial
        weights from SelfConceptSystem beliefs.
        This ensures the policy starts from Lumina's stated identity,
        not arbitrary defaults.
        """
        try:
            if self._path.exists():
                return  # file exists — don't override with defaults

            ai = getattr(getattr(self._organism, 'ai_system', None), None, None)
            if ai is None:
                ai = getattr(self._organism, 'ai_system', None)
            sc = getattr(ai, 'self_concept', None) if ai else None
            if sc is None:
                return

            beliefs = getattr(sc, '_beliefs', {})
            for name, belief in beliefs.items():
                # Map belief names to value dimensions
                for dim in VALUE_KEYWORDS:
                    if dim in name.lower() or dim in getattr(belief, 'statement', '').lower():
                        conf = getattr(belief, 'confidence', 0.5)
                        self.weights[dim] = max(
                            WEIGHT_FLOOR,
                            min(WEIGHT_CEILING, conf)
                        )
            logger.info(
                f"[DecisionPolicy] Seeded from SelfConcept — "
                f"top: {self._top_values(3)}"
            )
            self._save()
        except Exception as e:
            logger.debug(f"[DecisionPolicy] _sync_from_self_concept: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "weights":    self.weights,
                "updated_at": time.time(),
                "_meta":      {"version": "v50", "module": "decision_policy"},
            }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[DecisionPolicy] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            loaded = data.get("weights", {})
            for dim, w in loaded.items():
                self.weights[dim] = max(WEIGHT_FLOOR, min(WEIGHT_CEILING, float(w)))
            logger.debug(f"[DecisionPolicy] Loaded weights: {self.weights}")
        except Exception as e:
            logger.warning(f"[DecisionPolicy] Load failed (defaults): {e}")
