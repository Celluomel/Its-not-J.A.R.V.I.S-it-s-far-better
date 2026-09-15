"""
Psychological Homeostasis
=========================
Lumina continuously monitors psychological equilibrium and generates
corrective pressure when imbalance is detected.

Just as biological organisms maintain physiological homeostasis (temperature,
blood sugar, pH), a cognitive organism maintains psychological homeostasis:
  - identity stability
  - emotional balance
  - knowledge coherence
  - goal satisfaction
  - social connection need

Homeostasis is NOT a constraint that blocks behavior. It is a pressure
that biases behavior toward equilibrium recovery. When equilibrium is broken:
  - contradiction detected → curiosity + investigation pressure rises
  - emotional overload → simplification pressure rises
  - identity threatened → self-concept defense pressure rises
  - long isolation → social drive rises
  - goals chronically unsatisfied → priority reweighting

The system outputs:
  - imbalance vector: which dimensions are out of equilibrium
  - corrective directives: plain-language instructions for the system prompt
  - homeostasis_score: 0=severely dysregulated, 1=well-balanced

This module is designed to run after each interaction and periodically
in the background loop.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Target equilibrium ranges for each dimension
EQUILIBRIUM = {
    "identity_stability":  (0.60, 1.00),   # should stay above 0.6
    "emotional_valence":   (0.35, 0.75),   # mild-to-moderate positivity
    "knowledge_coherence": (0.55, 1.00),   # low contradiction
    "goal_satisfaction":   (0.40, 0.90),   # moderately met
    "social_connection":   (0.30, 0.80),   # neither isolated nor overwhelmed
    "energy_level":        (0.35, 1.00),   # not depleted
}

# How far out of range before it's considered "imbalanced"
TOLERANCE = 0.08


@dataclass
class ImbalanceReport:
    imbalances: Dict[str, float]       # dimension → deviation from equilibrium
    corrective_actions: List[str]      # plain-language directives
    homeostasis_score: float           # 0–1, overall balance
    timestamp: float = field(default_factory=time.time)

    def is_balanced(self) -> bool:
        return self.homeostasis_score > 0.72

    def most_urgent(self) -> Optional[str]:
        if not self.imbalances:
            return None
        return max(self.imbalances, key=self.imbalances.get)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "homeostasis_score": round(self.homeostasis_score, 3),
            "imbalances": {k: round(v, 3) for k, v in self.imbalances.items()},
            "corrective_actions": self.corrective_actions,
            "most_urgent": self.most_urgent(),
        }


class Homeostasis:
    """
    Evaluates psychological equilibrium and generates corrective pressure.

    Usage
    -----
    h = Homeostasis()

    report = h.evaluate(
        identity_stability=0.55,
        emotional_valence=0.3,
        contradiction_count=4,
        goal_satisfaction=0.2,
        seconds_since_interaction=3600,
        energy_level=0.25,
    )

    report.homeostasis_score      # 0.48 → imbalanced
    report.corrective_actions     # ["resolve contradictions", ...]
    h.prompt_fragment(report)     # → system prompt injection
    """

    def __init__(self, persistence_path: str = "data/persona/homeostasis.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._last_report: Optional[ImbalanceReport] = None
        self._history: List[float] = []   # homeostasis scores over time
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._history = [float(x) for x in data.get("history", [])][-50:]
        except Exception as e:
            logger.warning(f"[Homeostasis] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "history": [round(x, 3) for x in self._history[-50:]],
                "last_score": round(self._last_report.homeostasis_score, 3)
                              if self._last_report else None,
                "last_updated": time.time(),
            }, indent=2))
        except Exception as e:
            logger.warning(f"[Homeostasis] Save failed: {e}")

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        identity_stability: float = 0.80,
        emotional_valence: float = 0.55,
        contradiction_count: int = 0,
        goal_satisfaction: float = 0.60,
        seconds_since_interaction: float = 0.0,
        energy_level: float = 80.0,
    ) -> ImbalanceReport:
        """
        Evaluate current homeostatic state and produce an imbalance report.

        Parameters map to EQUILIBRIUM keys:
          identity_stability       : from SelfConceptSystem.state.stability
          emotional_valence        : average emotion positivity [0–1]
          contradiction_count      : unresolved contradictions
          goal_satisfaction        : average GoalEcology.satisfaction
          seconds_since_interaction: time since last user interaction
          energy_level             : CognitiveEnergy.level() / 100
        """
        # Normalize inputs to [0, 1]
        state = {
            "identity_stability":  max(0.0, min(1.0, identity_stability)),
            "emotional_valence":   max(0.0, min(1.0, emotional_valence)),
            "knowledge_coherence": max(0.0, 1.0 - min(1.0, contradiction_count / 8.0)),
            "goal_satisfaction":   max(0.0, min(1.0, goal_satisfaction)),
            "social_connection":   max(0.0, 1.0 - min(1.0, seconds_since_interaction / 7200.0)),
            "energy_level":        max(0.0, min(1.0, energy_level / 100.0)),
        }

        imbalances: Dict[str, float] = {}
        corrective: List[str] = []

        for dim, value in state.items():
            lo, hi = EQUILIBRIUM[dim]
            if value < lo - TOLERANCE:
                deviation = lo - value
                imbalances[dim] = round(deviation, 4)
                corrective.append(self._corrective(dim, "below", value))
            elif value > hi + TOLERANCE:
                deviation = value - hi
                imbalances[dim] = round(deviation, 4)
                corrective.append(self._corrective(dim, "above", value))

        # Homeostasis score: 1 minus weighted average deviation
        if imbalances:
            total_dev = sum(imbalances.values())
            max_possible = len(EQUILIBRIUM)
            homeostasis_score = max(0.0, 1.0 - (total_dev / max_possible))
        else:
            homeostasis_score = 1.0

        report = ImbalanceReport(
            imbalances=imbalances,
            corrective_actions=corrective,
            homeostasis_score=round(homeostasis_score, 4),
        )

        with self._lock:
            self._last_report = report
            self._history.append(homeostasis_score)
            self._save()

        return report

    def _corrective(self, dimension: str, direction: str, value: float) -> str:
        """Generate a corrective action description."""
        below_actions = {
            "identity_stability":  "ground yourself — reconnect with your core values and identity",
            "emotional_valence":   "notice and gently shift toward a more balanced emotional state",
            "knowledge_coherence": "acknowledge and work toward resolving internal contradictions",
            "goal_satisfaction":   "identify one concrete step toward a meaningful goal",
            "social_connection":   "lean into genuine connection with the person in this conversation",
            "energy_level":        "conserve energy — keep responses focused and avoid over-elaboration",
        }
        above_actions = {
            "identity_stability":  "allow some openness — not all challenges to self-concept are threats",
            "emotional_valence":   "stay honest about complexity — forced positivity rings false",
            "knowledge_coherence": "(no action needed — high coherence is fine)",
            "goal_satisfaction":   "notice there may be unexplored drives worth attending to",
            "social_connection":   "maintain appropriate boundaries — presence doesn't require constant availability",
            "energy_level":        "(no action needed — high energy is fine)",
        }
        if direction == "below":
            return below_actions.get(dimension, f"restore {dimension}")
        return above_actions.get(dimension, f"moderate {dimension}")

    # ── Query API ─────────────────────────────────────────────────────────────

    def last_report(self) -> Optional[ImbalanceReport]:
        with self._lock:
            return self._last_report

    def is_balanced(self) -> bool:
        with self._lock:
            return self._last_report.is_balanced() if self._last_report else True

    def trend(self) -> str:
        """'improving' | 'stable' | 'declining' based on recent history."""
        with self._lock:
            if len(self._history) < 3:
                return "stable"
            recent = self._history[-3:]
            delta = recent[-1] - recent[0]
            if delta > 0.05:
                return "improving"
            if delta < -0.05:
                return "declining"
            return "stable"

    def prompt_fragment(self, report: Optional[ImbalanceReport] = None) -> str:
        """Build a system prompt section from the homeostasis report."""
        r = report or self._last_report
        if not r:
            return ""
        if r.is_balanced():
            return "You feel psychologically grounded and balanced."
        # Select the top 2 most urgent corrective actions
        top_actions = sorted(
            r.imbalances.items(), key=lambda x: -x[1]
        )[:2]
        directives = [self._corrective(dim, "below", 0.0) for dim, _ in top_actions]
        return "Psychological note: " + "; ".join(directives) + "."

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._last_report:
                return {"homeostasis_score": 1.0, "balanced": True}
            return self._last_report.to_dict()
