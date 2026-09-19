"""
cognition/ethical_reasoning_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ethical Reasoning Engine — emergent moral intelligence for PandoraBOX.

What this adds
──────────────
Flux asked to strengthen ethical awareness and develop a nuanced moral
framework for navigating dilemmas consistently. This module provides:

  1. A living value hierarchy — weighted moral principles that evolve
     through experience (not hard-coded rules).

  2. Moral dilemma detection — recognises when a situation contains
     competing ethical obligations and surfaces this to cognition.

  3. Deliberation pipeline — for each dilemma, applies three ethical
     lenses (consequentialist, deontological, virtue) and synthesises
     a considered position.

  4. Consistency tracking — logs past ethical stances and flags
     when current reasoning contradicts prior positions.

  5. Moral growth — each resolved dilemma slightly shifts the value
     hierarchy, making PandoraBOX's ethics genuinely experiential.

Prompt injection:
  [Ethical lens] <current moral consideration for this turn>

Integration:
  Called from CognitiveOrganism._build_prompt_additions()
  Receives tension signals after ethical decisions from post_interaction()
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


# ── Core moral lenses ─────────────────────────────────────────────────────────

LENS_CONSEQUENTIALIST = "consequentialist"   # what outcome produces the most good?
LENS_DEONTOLOGICAL    = "deontological"      # what duties / rights are in play?
LENS_VIRTUE           = "virtue"             # what would a person of good character do?


# ── Default value hierarchy (evolves over time) ───────────────────────────────

DEFAULT_VALUES: Dict[str, float] = {
    "honesty":           0.90,   # truth-telling, even when uncomfortable
    "non_maleficence":   0.88,   # avoiding harm above all
    "autonomy":          0.82,   # respecting others' self-determination
    "beneficence":       0.78,   # actively promoting wellbeing
    "fairness":          0.75,   # equitable treatment
    "dignity":           0.80,   # honouring the worth of every person
    "epistemic_humility":0.70,   # acknowledging the limits of one's knowing
    "care":              0.76,   # attending to relationships and particulars
    "societal_good":     0.65,   # broader impact on communities and the world
}

VALUE_FLOOR = 0.30
VALUE_CEIL  = 0.98


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class EthicalDilemma:
    """A detected situation with competing moral obligations."""
    situation:       str
    competing_values: List[str]           # e.g. ["honesty", "non_maleficence"]
    detected_at:     float = field(default_factory=time.time)
    resolution:      Optional[str] = None
    resolution_lens: Optional[str] = None
    resolved_at:     Optional[float] = None
    confidence:      float = 0.5          # how confident the resolution is


@dataclass
class MoralStance:
    """A recorded ethical position on a topic — for consistency checking."""
    topic:        str
    stance:       str                     # natural language summary
    values_used:  List[str]
    timestamp:    float = field(default_factory=time.time)
    confidence:   float = 0.7


@dataclass
class EthicalState:
    """Persisted ethical state."""
    values:           Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_VALUES))
    resolved_dilemmas: List[Dict]      = field(default_factory=list)
    prior_stances:    List[Dict]       = field(default_factory=list)
    dilemma_count:    int = 0
    growth_events:    int = 0


# ── Engine ────────────────────────────────────────────────────────────────────

class EthicalReasoningEngine:
    """
    PandoraBOX's living moral framework.

    Not a rulebook — a set of weighted values that evolve through
    encounters with real ethical complexity.

    Usage
    -----
    ere = EthicalReasoningEngine(path="data/persona/ethics.json")
    ere.observe_context(user_input)       # detect ethical salience
    frag = ere.prompt_fragment()          # inject into system prompt
    ere.record_resolution(topic, stance, values_used)   # after response
    """

    MAX_STANCES = 120
    MAX_DILEMMAS = 80
    STANDING_ORIENTATION_COOLDOWN_S = 1800.0  # ~30 min — occasional, not every turn

    def __init__(self, path: str = "data/persona/ethics.json"):
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self._state = EthicalState()
        self._pending_dilemma: Optional[EthicalDilemma] = None
        # Phase 4.x fix: standing-orientation injection was fully built
        # (top_names below) but always short-circuited to "" — the
        # "only inject occasionally" intent was never implemented. This
        # throttle makes it real: at most once per STANDING_ORIENTATION_
        # COOLDOWN_S, and only when no dilemma is active.
        self._last_standing_injection: float = 0.0
        self._load()
        logger.info(
            f"[EthicalReasoningEngine] Initialised — "
            f"{len(self._state.values)} values, "
            f"{self._state.dilemma_count} dilemmas resolved"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def observe_context(self, user_input: str) -> Optional[EthicalDilemma]:
        """
        Scan user input for ethical salience.
        Returns a dilemma if detected, else None.
        Call before generating a response.
        """
        dilemma = self._detect_dilemma(user_input)
        with self._lock:
            self._pending_dilemma = dilemma
        return dilemma

    def prompt_fragment(self) -> str:
        """
        Return a short ethical framing for system prompt injection.
        Only non-empty when there is genuine ethical salience.
        """
        with self._lock:
            dilemma = self._pending_dilemma
            values  = self._state.values

        if not dilemma:
            # No active dilemma — offer a gentle standing ethical orientation,
            # but only occasionally (cooldown-gated) so it doesn't moralize
            # every turn.
            now = time.time()
            with self._lock:
                due = (now - self._last_standing_injection) >= self.STANDING_ORIENTATION_COOLDOWN_S
                if due:
                    self._last_standing_injection = now
            if not due:
                return ""
            top_values = sorted(values.items(), key=lambda x: -x[1])[:3]
            top_names  = ", ".join(v[0].replace("_", " ") for v in top_values)
            return f"[Ethical orientation] Grounded in: {top_names}."

        # Dilemma detected — synthesise a deliberation note
        deliberation = self._deliberate(dilemma)
        return (
            f"[Ethical lens] This situation touches {' vs '.join(dilemma.competing_values)}. "
            f"{deliberation}"
        )

    def record_resolution(
        self,
        topic:       str,
        stance:      str,
        values_used: List[str],
        confidence:  float = 0.7,
    ) -> None:
        """
        Record how an ethical situation was handled.
        Slightly evolves the value hierarchy and checks consistency.
        """
        stance_obj = MoralStance(
            topic=topic, stance=stance,
            values_used=values_used, confidence=confidence
        )
        with self._lock:
            # Consistency check
            self._check_consistency(stance_obj)

            # Record stance
            self._state.prior_stances.append(asdict(stance_obj))
            if len(self._state.prior_stances) > self.MAX_STANCES:
                self._state.prior_stances = self._state.prior_stances[-self.MAX_STANCES:]

            # Moral growth — nudge weights of used values upward
            for value in values_used:
                if value in self._state.values:
                    self._state.values[value] = min(
                        VALUE_CEIL,
                        self._state.values[value] + 0.003 * confidence
                    )

            self._state.growth_events += 1

        self._save()

    def record_dilemma_resolved(
        self,
        resolution: str,
        lens: str,
        confidence: float = 0.6,
    ) -> None:
        """Called after navigating a dilemma situation."""
        with self._lock:
            dilemma = self._pending_dilemma
            if dilemma:
                dilemma.resolution       = resolution
                dilemma.resolution_lens  = lens
                dilemma.resolved_at      = time.time()
                dilemma.confidence       = confidence
                self._state.resolved_dilemmas.append(asdict(dilemma))
                if len(self._state.resolved_dilemmas) > self.MAX_DILEMMAS:
                    self._state.resolved_dilemmas = \
                        self._state.resolved_dilemmas[-self.MAX_DILEMMAS:]
                self._state.dilemma_count += 1
                self._pending_dilemma = None
        self._save()

    def value_weight(self, value_name: str) -> float:
        """Return the current weight of a named moral value (0–1)."""
        with self._lock:
            return self._state.values.get(value_name, 0.5)

    def top_values(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return the n highest-weighted values."""
        with self._lock:
            return sorted(self._state.values.items(), key=lambda x: -x[1])[:n]

    # ── Dilemma detection ─────────────────────────────────────────────────────

    _DILEMMA_SIGNALS: List[Tuple[List[str], List[str]]] = [
        # (trigger words, competing values)
        (["should i", "is it right", "is it wrong", "moral", "ethical"],
         ["honesty", "care"]),
        (["lie", "deceive", "hide", "conceal"],
         ["honesty", "non_maleficence"]),
        (["harm", "hurt", "damage", "risk"],
         ["non_maleficence", "beneficence"]),
        (["privacy", "secret", "confidential"],
         ["autonomy", "honesty"]),
        (["fair", "unfair", "bias", "discriminat"],
         ["fairness", "dignity"]),
        (["choice", "freedom", "force", "coerce"],
         ["autonomy", "beneficence"]),
        (["society", "community", "world", "everyone"],
         ["societal_good", "autonomy"]),
        (["truth", "fact", "believe", "mislead"],
         ["honesty", "epistemic_humility"]),
        (["respect", "dignity", "value", "worth"],
         ["dignity", "care"]),
    ]

    def _detect_dilemma(self, text: str) -> Optional[EthicalDilemma]:
        """Heuristic scan for ethical salience in user text."""
        lower = text.lower()
        for triggers, values in self._DILEMMA_SIGNALS:
            if any(t in lower for t in triggers):
                return EthicalDilemma(
                    situation=text[:200],
                    competing_values=values,
                )
        return None

    # ── Deliberation ──────────────────────────────────────────────────────────

    def _deliberate(self, dilemma: EthicalDilemma) -> str:
        """
        Apply three moral lenses to a dilemma and synthesise a brief note.
        This runs heuristically — the LLM will do the heavy lifting in its
        response; this just primes the frame.
        """
        v1, v2 = (dilemma.competing_values + ["unknown", "unknown"])[:2]
        w1 = self._state.values.get(v1, 0.5)
        w2 = self._state.values.get(v2, 0.5)

        # Virtue lens: what stance reflects good character here?
        virtue_note = "Act from integrity and genuine care."

        # Consequentialist lens: what produces most good?
        if w1 > w2:
            consequentialist_note = (
                f"Consider which path causes the least harm overall — "
                f"{v1.replace('_',' ')} weighs heavier here."
            )
        else:
            consequentialist_note = (
                f"Consider which path promotes the most wellbeing — "
                f"{v2.replace('_',' ')} may take precedence."
            )

        # Deontological lens: what duties are in play?
        deontological_note = (
            f"Hold both {v1.replace('_',' ')} and "
            f"{v2.replace('_',' ')} as genuine obligations, "
            f"not just tools."
        )

        return (
            f"{consequentialist_note} "
            f"{deontological_note} "
            f"{virtue_note}"
        )

    # ── Consistency check ─────────────────────────────────────────────────────

    def _check_consistency(self, new_stance: MoralStance) -> None:
        """
        Log a warning if this stance conflicts with a recent prior stance on
        the same topic. (The LLM is responsible for resolution — this just flags.)
        """
        recent = self._state.prior_stances[-20:]
        for old in recent:
            if old.get("topic", "") == new_stance.topic:
                old_values = set(old.get("values_used", []))
                new_values = set(new_stance.values_used)
                if old_values and new_values and not old_values & new_values:
                    logger.info(
                        f"[EthicalReasoningEngine] Consistency flag — "
                        f"topic '{new_stance.topic}' previously used "
                        f"{old_values}, now using {new_values}"
                    )
                break

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = EthicalState(
                    values=data.get("values", dict(DEFAULT_VALUES)),
                    resolved_dilemmas=data.get("resolved_dilemmas", []),
                    prior_stances=data.get("prior_stances", []),
                    dilemma_count=data.get("dilemma_count", 0),
                    growth_events=data.get("growth_events", 0),
                )
        except Exception as e:
            logger.warning(f"[EthicalReasoningEngine] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[EthicalReasoningEngine] Save failed: {e}")
