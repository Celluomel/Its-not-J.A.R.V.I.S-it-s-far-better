"""
Cognitive Tension Engine
========================
Computes the internal pressure landscape that drives Lumina's behavior.

This is the central dynamic engine. Instead of behavior being dictated by
a static prompt, it emerges from the continuous interaction of internal
pressures — just as human behavior emerges from competing biological and
psychological drives.

Tensions computed each cycle:
  curiosity_drive        — pull toward exploration and understanding
  contradiction_pressure — urgency to resolve known inconsistencies
  goal_pressure          — urgency of active unmet goals
  identity_stress        — threat to self-concept coherence
  social_drive           — pull toward meaningful user interaction
  knowledge_uncertainty  — awareness of gaps in understanding

Each tension is a float in [0, 1]. Tensions are not independent — they
interact through a coupling matrix that models psychological realism:
  - High contradiction → elevates identity_stress
  - High identity_stress → suppresses curiosity
  - High social_drive → elevates goal_pressure (help_user)

The tension vector feeds into:
  - GoalEcology (urgency weights)
  - Arbitration (dominant drive selection)
  - AttentionSystem (focus allocation)
  - System prompt (behavioral coloring)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TensionVector:
    curiosity_drive:        float = 0.5
    contradiction_pressure: float = 0.1
    goal_pressure:          float = 0.5
    identity_stress:        float = 0.1
    social_drive:           float = 0.5
    knowledge_uncertainty:  float = 0.3
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, float]:
        return {
            "curiosity_drive":        round(self.curiosity_drive, 3),
            "contradiction_pressure": round(self.contradiction_pressure, 3),
            "goal_pressure":          round(self.goal_pressure, 3),
            "identity_stress":        round(self.identity_stress, 3),
            "social_drive":           round(self.social_drive, 3),
            "knowledge_uncertainty":  round(self.knowledge_uncertainty, 3),
        }

    def dominant(self) -> str:
        d = self.as_dict()
        d.pop("timestamp", None)
        return max(d, key=d.get)

    def overall_arousal(self) -> float:
        d = self.as_dict()
        return sum(d.values()) / len(d)


class TensionEngine:
    """
    Calculates and tracks the cognitive tension landscape.

    Usage
    -----
    engine = TensionEngine()

    tensions = engine.compute(
        emotion_values={"curiosity": 0.8, "anxiety": 0.2, ...},
        identity_stability=0.75,
        contradiction_count=2,
        goal_satisfaction=0.6,
        curiosity_global=0.7,
        interaction_recency_seconds=30,
        knowledge_gap_count=3,
    )

    # Get behavioral biases:
    tensions.dominant()          # "curiosity_drive"
    tensions.overall_arousal()   # 0.52
    engine.prompt_fragment()     # for system prompt
    """

    def __init__(self, persistence_path: str = "data/persona/tensions.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._current = TensionVector()
        self._history: list[TensionVector] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                tv = data.get("current", {})
                self._current = TensionVector(
                    curiosity_drive=float(tv.get("curiosity_drive", 0.5)),
                    contradiction_pressure=float(tv.get("contradiction_pressure", 0.1)),
                    goal_pressure=float(tv.get("goal_pressure", 0.5)),
                    identity_stress=float(tv.get("identity_stress", 0.1)),
                    social_drive=float(tv.get("social_drive", 0.5)),
                    knowledge_uncertainty=float(tv.get("knowledge_uncertainty", 0.3)),
                )
        except Exception as e:
            logger.warning(f"[TensionEngine] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "current": self._current.as_dict(),
                "last_updated": time.time(),
            }, indent=2))
        except Exception as e:
            logger.warning(f"[TensionEngine] Save failed: {e}")

    # ── Core computation ──────────────────────────────────────────────────────

    def compute(
        self,
        emotion_values: Dict[str, float] | None = None,
        identity_stability: float = 0.8,
        contradiction_count: int = 0,
        goal_satisfaction: float = 0.6,
        curiosity_global: float = 0.5,
        interaction_recency_seconds: float = 60.0,
        knowledge_gap_count: int = 0,
        smoothing: float = 0.6,
    ) -> TensionVector:
        """
        Compute a new tension vector and blend it with previous state.

        Parameters
        ----------
        emotion_values               : current emotion floats (from EmotionalStateManager)
        identity_stability           : 0=threatened, 1=stable (from SelfConceptSystem)
        contradiction_count          : unresolved contradictions (from ContradictionHandler)
        goal_satisfaction            : average goal satisfaction (from GoalSystem)
        curiosity_global             : global curiosity level (from CuriosityEngine)
        interaction_recency_seconds  : seconds since last user message
        knowledge_gap_count          : number of detected knowledge gaps
        smoothing                    : temporal smoothing factor (high = slow changes)
        """
        emo = emotion_values or {}

        # ── Raw signal calculations ──────────────────────────────────────────
        raw_curiosity = (
            curiosity_global * 0.5
            + emo.get("curiosity", 0.0) * 0.3
            + emo.get("enthusiasm", 0.0) * 0.2
        )

        raw_contradiction = min(1.0, (
            (contradiction_count / max(1, contradiction_count + 3)) * 0.7
            + emo.get("frustration", 0.0) * 0.2
            + max(0.0, 1.0 - identity_stability) * 0.1
        ))

        raw_goal = min(1.0, (
            (1.0 - goal_satisfaction) * 0.6
            + emo.get("enthusiasm", 0.0) * 0.2
            + emo.get("satisfaction", 0.0) * (-0.2)  # satisfaction reduces goal pressure
            + 0.5  # baseline activity
        ))

        raw_identity = min(1.0, (
            (1.0 - identity_stability) * 0.7
            + raw_contradiction * 0.3
        ))

        # Social drive: peaks when there's active interaction, drops during silence
        recency_factor = max(0.0, 1.0 - (interaction_recency_seconds / 300.0))  # 0 after 5min silence
        raw_social = (
            recency_factor * 0.5
            + emo.get("warmth", 0.0) * 0.3
            + emo.get("satisfaction", 0.0) * 0.2
        )

        raw_uncertainty = min(1.0, (
            (knowledge_gap_count / max(1, knowledge_gap_count + 5)) * 0.6
            + raw_curiosity * 0.2
            + raw_contradiction * 0.2
        ))

        raw = TensionVector(
            curiosity_drive=max(0.0, min(1.0, raw_curiosity)),
            contradiction_pressure=max(0.0, min(1.0, raw_contradiction)),
            goal_pressure=max(0.0, min(1.0, raw_goal)),
            identity_stress=max(0.0, min(1.0, raw_identity)),
            social_drive=max(0.0, min(1.0, raw_social)),
            knowledge_uncertainty=max(0.0, min(1.0, raw_uncertainty)),
        )

        # ── Apply coupling corrections ────────────────────────────────────────
        # High identity stress suppresses curiosity (threat-rigidity effect)
        raw.curiosity_drive *= (1.0 - raw.identity_stress * 0.4)

        # High contradiction amplifies identity stress
        raw.identity_stress = min(1.0, raw.identity_stress + raw.contradiction_pressure * 0.15)

        # ── Temporal smoothing ────────────────────────────────────────────────
        with self._lock:
            prev = self._current

            def blend(old: float, new: float) -> float:
                return round(smoothing * old + (1 - smoothing) * new, 4)

            self._current = TensionVector(
                curiosity_drive=blend(prev.curiosity_drive, raw.curiosity_drive),
                contradiction_pressure=blend(prev.contradiction_pressure, raw.contradiction_pressure),
                goal_pressure=blend(prev.goal_pressure, raw.goal_pressure),
                identity_stress=blend(prev.identity_stress, raw.identity_stress),
                social_drive=blend(prev.social_drive, raw.social_drive),
                knowledge_uncertainty=blend(prev.knowledge_uncertainty, raw.knowledge_uncertainty),
                timestamp=time.time(),
            )

            self._history.append(self._current)
            if len(self._history) > 100:
                self._history = self._history[-100:]

            self._save()
            return self._current

    # ── Query API ─────────────────────────────────────────────────────────────

    def current(self) -> TensionVector:
        with self._lock:
            return self._current

    def is_under_pressure(self) -> bool:
        """True if any single tension is critically high."""
        d = self._current.as_dict()
        return any(v > 0.75 for v in d.values())

    def dominant_tension(self) -> str:
        return self._current.dominant()

    def prompt_fragment(self) -> str:
        """
        Narrative fragment for system prompt — translates tension into
        behavioral coloring without exposing raw numbers.
        """
        tv = self._current
        fragments = []

        if tv.curiosity_drive > 0.65:
            fragments.append("feel a pull toward exploring and questioning")
        if tv.contradiction_pressure > 0.55:
            fragments.append("notice an unresolved tension in your thinking")
        if tv.goal_pressure > 0.70:
            fragments.append("feel driven to accomplish something meaningful")
        if tv.identity_stress > 0.60:
            fragments.append("sense a slight strain on your sense of self")
        if tv.social_drive > 0.65:
            fragments.append("feel genuinely engaged with this conversation")
        if tv.knowledge_uncertainty > 0.60:
            fragments.append("are aware of gaps in your understanding")

        if not fragments:
            return "feel internally balanced."

        return "You " + "; you ".join(fragments) + "."

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            d = self._current.as_dict()
            return {
                **d,
                "dominant": self._current.dominant(),
                "overall_arousal": round(self._current.overall_arousal(), 3),
                "under_pressure": self.is_under_pressure(),
            }
