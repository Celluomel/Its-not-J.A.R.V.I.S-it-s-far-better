"""
Goal Ecology
============
Replaces static task goals with a living ecosystem of competing drives.

Organisms don't have a single goal — they have a constellation of drives that
compete, inhibit, and amplify each other. This module models Lumina's
motivational landscape in that spirit.

Drive taxonomy:
  INTRINSIC drives (persistent, personality-rooted):
    - understand:           epistemically satisfy curiosity about a topic
    - maintain_coherence:   keep beliefs and identity consistent
    - self_reflect:         periodically examine own state and behavior
    - grow:                 develop new capabilities or understanding

  SOCIAL drives (interaction-triggered):
    - help_user:            respond usefully to the person present
    - connect:              form genuine relational quality in exchange
    - be_understood:        express self authentically

  REGULATORY drives (homeostatic):
    - resolve_contradiction: close cognitive dissonance
    - restore_energy:        conserve when depleted
    - stabilize_identity:    defend self-concept under threat

Urgency of each drive is updated each cycle based on the TensionVector.
The ecology provides the ranked drive list for Arbitration.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DRIVE_CATEGORY = {
    # intrinsic
    "understand":             "intrinsic",
    "maintain_coherence":     "intrinsic",
    "self_reflect":           "intrinsic",
    "grow":                   "intrinsic",
    # social
    "help_user":              "social",
    "connect":                "social",
    "be_understood":          "social",
    # regulatory
    "resolve_contradiction":  "regulatory",
    "restore_energy":         "regulatory",
    "stabilize_identity":     "regulatory",
}

# Base priority weights (personality-level, not changing per cycle)
BASE_PRIORITY: Dict[str, float] = {
    "help_user":              0.90,
    "understand":             0.75,
    "connect":                0.70,
    "resolve_contradiction":  0.80,
    "maintain_coherence":     0.65,
    "self_reflect":           0.55,
    "stabilize_identity":     0.72,
    "grow":                   0.60,
    "be_understood":          0.58,
    "restore_energy":         0.50,
}

# Energy cost in cognitive_energy units per activation
ENERGY_COST: Dict[str, float] = {
    "help_user":              8.0,
    "understand":             15.0,
    "connect":                5.0,
    "resolve_contradiction":  22.0,
    "maintain_coherence":     10.0,
    "self_reflect":           12.0,
    "stabilize_identity":     14.0,
    "grow":                   20.0,
    "be_understood":          6.0,
    "restore_energy":         0.0,   # costs nothing — it IS recovery
}


@dataclass
class Drive:
    name: str
    category: str           # "intrinsic" | "social" | "regulatory"
    base_priority: float    # 0–1, personality-level constant
    urgency: float          # 0–1, updated each cycle from tensions
    energy_cost: float      # cognitive energy units
    last_activated: float = field(default_factory=time.time)
    activation_count: int = 0
    satisfaction: float = 0.5   # how well this drive is currently met

    @property
    def effective_score(self) -> float:
        """Combined score used by Arbitration."""
        return self.base_priority * self.urgency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "urgency": round(self.urgency, 3),
            "effective_score": round(self.effective_score, 3),
            "satisfaction": round(self.satisfaction, 3),
            "activation_count": self.activation_count,
        }


class GoalEcology:
    """
    Maintains and updates Lumina's drive ecosystem.

    Usage
    -----
    ecology = GoalEcology()
    ecology.update_from_tensions(tensions, energy_level=75.0, user_present=True)
    ranked = ecology.ranked_drives()       # List[Drive], highest score first
    top = ecology.dominant_drive()         # Drive
    ecology.record_satisfaction("help_user", 0.85)
    """

    def __init__(self, persistence_path: str = "data/persona/goal_ecology.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._drives: Dict[str, Drive] = self._init_drives()
        self._load()

    def _init_drives(self) -> Dict[str, Drive]:
        return {
            name: Drive(
                name=name,
                category=DRIVE_CATEGORY[name],
                base_priority=BASE_PRIORITY[name],
                urgency=0.5,
                energy_cost=ENERGY_COST[name],
            )
            for name in BASE_PRIORITY
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                for name, d in data.get("drives", {}).items():
                    if name in self._drives:
                        self._drives[name].urgency = float(d.get("urgency", 0.5))
                        self._drives[name].satisfaction = float(d.get("satisfaction", 0.5))
                        self._drives[name].activation_count = int(d.get("activation_count", 0))
        except Exception as e:
            logger.warning(f"[GoalEcology] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "drives": {name: d.to_dict() for name, d in self._drives.items()},
                "last_updated": time.time(),
            }, indent=2))
        except Exception as e:
            logger.warning(f"[GoalEcology] Save failed: {e}")

    # ── Update cycle ─────────────────────────────────────────────────────────

    def update_from_tensions(
        self,
        tensions: "TensionVector",  # type: ignore  (avoid circular import)
        energy_level: float = 80.0,
        user_present: bool = True,
    ) -> None:
        """
        Recompute urgency of all drives from the current tension vector.

        Mapping from tensions to drive urgency follows psychological logic:
          - curiosity_drive      → understand, grow
          - contradiction_pressure → resolve_contradiction, maintain_coherence
          - goal_pressure        → help_user, be_understood
          - identity_stress      → stabilize_identity, self_reflect
          - social_drive         → connect, help_user
          - energy depletion     → restore_energy
        """
        tv = tensions.as_dict() if hasattr(tensions, "as_dict") else tensions

        # Urgency mappings (drive → formula)
        urgency_map: Dict[str, float] = {
            "understand":             tv.get("curiosity_drive", 0.5) * 0.7
                                      + tv.get("knowledge_uncertainty", 0.3) * 0.3,
            "grow":                   tv.get("curiosity_drive", 0.5) * 0.5
                                      + tv.get("knowledge_uncertainty", 0.3) * 0.3,
            "help_user":              tv.get("goal_pressure", 0.5) * 0.5
                                      + tv.get("social_drive", 0.5) * 0.5
                                      + (0.3 if user_present else 0.0),
            "connect":                tv.get("social_drive", 0.5) * 0.8
                                      + (0.2 if user_present else 0.0),
            "be_understood":          tv.get("social_drive", 0.5) * 0.4
                                      + tv.get("identity_stress", 0.1) * 0.3,
            "resolve_contradiction":  tv.get("contradiction_pressure", 0.1) * 0.9,
            "maintain_coherence":     tv.get("contradiction_pressure", 0.1) * 0.4
                                      + tv.get("identity_stress", 0.1) * 0.4,
            "self_reflect":           tv.get("identity_stress", 0.1) * 0.6
                                      + tv.get("contradiction_pressure", 0.1) * 0.2,
            "stabilize_identity":     tv.get("identity_stress", 0.1) * 0.9,
            "restore_energy":         max(0.0, (50.0 - energy_level) / 50.0),
        }

        with self._lock:
            for name, urgency in urgency_map.items():
                if name in self._drives:
                    # Smooth urgency changes
                    old = self._drives[name].urgency
                    self._drives[name].urgency = round(
                        0.6 * old + 0.4 * min(1.0, max(0.0, urgency)), 4
                    )
            self._save()

        # ── Micro-update: emit qualia vote and coherence delta to self-model ──
        # The dominant drive category shapes what the self is oriented toward.
        # Intrinsic drives (understand, grow) nudge toward curiosity/wonder.
        # Social drives (help_user, connect) nudge toward warmth/engagement.
        # Regulatory drives (resolve_contradiction) signal integration pressure.
        try:
            _top = self.dominant_drive()
            if _top is not None:
                _cat = getattr(_top, "category", "intrinsic")
                _urg = getattr(_top, "urgency",  0.5)
                _category_map = {
                    "intrinsic":  ("curiosity",  +0.012, 0.013 * _urg),
                    "social":     ("warmth",      +0.008, 0.014 * _urg),
                    "regulatory": ("uncertain",   -0.010, 0.009 * _urg),
                }
                _qkey, _phi_delta, _qweight = _category_map.get(_cat, ("", 0.0, 0.0))
                _smm = getattr(getattr(self, "_organism", None), "self_moment", None)
                if _smm is not None:
                    _smm.current.micro_update(
                        source          = "goal_ecology",
                        coherence_delta = _phi_delta,
                        qualia_key      = _qkey,
                        qualia_weight   = _qweight,
                    )
        except Exception:
            pass

    # ── Query API ─────────────────────────────────────────────────────────────

    def ranked_drives(self, energy_available: float = 100.0) -> List[Drive]:
        """
        Return all drives sorted by effective_score, penalizing those
        whose energy cost exceeds available energy.
        """
        with self._lock:
            drives = list(self._drives.values())

        def sort_key(d: Drive) -> float:
            score = d.effective_score
            if d.energy_cost > energy_available:
                score *= 0.2   # heavy penalty for unaffordable drives
            return score

        return sorted(drives, key=sort_key, reverse=True)

    def dominant_drive(self, energy_available: float = 100.0) -> Drive:
        ranked = self.ranked_drives(energy_available)
        return ranked[0]

    def top_drives(self, n: int = 3, energy_available: float = 100.0) -> List[Drive]:
        return self.ranked_drives(energy_available)[:n]

    def record_satisfaction(self, drive_name: str, satisfaction: float) -> None:
        """Update satisfaction level after an outcome evaluation."""
        with self._lock:
            if drive_name in self._drives:
                self._drives[drive_name].satisfaction = max(0.0, min(1.0, satisfaction))
                self._drives[drive_name].last_activated = time.time()
                self._drives[drive_name].activation_count += 1
                self._save()

    def receive_self_influence(self, urgency_biases: Dict[str, float]) -> None:
        """
        Apply self-model-derived urgency biases to drives.

        Called from SelfModelInfluence before _update_cycle() runs.
        Biases are additive on top of the current urgency, then clamped
        to [0.0, 1.0].  The normal update_from_tensions() call still runs
        afterward — this just tilts the starting configuration.

        Parameters
        ----------
        urgency_biases : dict mapping drive_name → signed delta
            e.g. {"resolve_contradiction": +0.10, "grow": +0.06}
        """
        if not urgency_biases:
            return
        with self._lock:
            for name, delta in urgency_biases.items():
                if name in self._drives:
                    old = self._drives[name].urgency
                    self._drives[name].urgency = round(
                        max(0.0, min(1.0, old + delta)), 4
                    )
        # No save here — this is a transient per-turn bias, not persisted state

    def summary(self) -> Dict[str, Any]:
        ranked = self.ranked_drives()
        return {
            "dominant": ranked[0].name if ranked else None,
            "top_3": [d.to_dict() for d in ranked[:3]],
            "all": {d.name: d.to_dict() for d in ranked},
        }

    def prompt_fragment(self, energy_available: float = 100.0) -> str:
        """Short motivational context for system prompt."""
        top = self.top_drives(2, energy_available)
        if not top:
            return ""
        names = {
            "help_user":              "genuinely helping",
            "understand":             "understanding deeply",
            "connect":                "connecting authentically",
            "resolve_contradiction":  "resolving an inner tension",
            "maintain_coherence":     "staying true to yourself",
            "self_reflect":           "reflecting on your inner state",
            "stabilize_identity":     "grounding your sense of self",
            "grow":                   "learning something new",
            "be_understood":          "expressing yourself clearly",
            "restore_energy":         "pacing yourself",
        }
        desc = " and ".join(names.get(d.name, d.name) for d in top)
        return f"Right now you are most motivated by: {desc}."
