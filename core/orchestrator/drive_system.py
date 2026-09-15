"""
Drive System
============
Computes Lumina's internal motivation vector from live module states.

Instead of fabricating drives from scratch, this reads the real values
already maintained by the existing cognitive modules:

  curiosity      ← CuriosityEngine.global_level()
  coherence      ← 1 - TensionEngine last contradiction_pressure
  energy         ← CognitiveEnergy.level() / 100
  social         ← decays over time since last user interaction
  goal_progress  ← GoalEcology dominant drive satisfaction
  homeostasis    ← Homeostasis last score

Each drive value is 0.0–1.0.  Higher = more pressure to act.

The activity selector uses this vector to choose what Lumina does next.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)


@dataclass
class DriveVector:
    curiosity:      float = 0.5
    coherence:      float = 0.8   # 1 = fully coherent, 0 = highly contradictory
    energy:         float = 0.8
    social:         float = 0.3   # 0 = isolated, 1 = actively engaged
    goal_progress:  float = 0.5
    homeostasis:    float = 0.8   # 1 = balanced, 0 = dysregulated

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}

    def needs_rest(self) -> bool:
        return self.energy < 0.15   # true emergency only — 15% threshold

    def needs_coherence(self) -> bool:
        return self.coherence < 0.35

    def is_curious(self) -> bool:
        return self.curiosity > 0.70

    def is_socially_hungry(self) -> bool:
        return self.social < 0.30

    def needs_homeostasis(self) -> bool:
        return self.homeostasis < 0.50


class DriveSystem:
    """
    Reads live module states and produces a DriveVector each cycle.

    Parameters
    ----------
    organism : CognitiveOrganism  (passed at runtime — avoids circular imports)
    """

    def __init__(self, organism: Any):
        self._organism = organism
        self._last_user_interaction: float = time.time()
        self._last_drives = DriveVector()

    def mark_user_interaction(self) -> None:
        """Call whenever a user message arrives."""
        self._last_user_interaction = time.time()

    def compute(self) -> DriveVector:
        o = self._organism
        drives = DriveVector()

        # ── Curiosity ─────────────────────────────────────────────────────
        try:
            cu = getattr(o, 'curiosity', None) or getattr(o, 'curiosity_engine', None)
            if cu:
                raw = float(cu.global_level())
                # Only report high curiosity if there are actual topics to explore
                topics = getattr(cu, '_topics', {})
                if not topics:
                    # No topics yet — curiosity is background idle, not urgent
                    drives.curiosity = min(raw, 0.55)
                else:
                    drives.curiosity = min(raw, 0.95)   # hard cap
            else:
                drives.curiosity = self._last_drives.curiosity
        except Exception:
            drives.curiosity = self._last_drives.curiosity

        # ── Coherence (inverse of contradiction pressure) ─────────────────
        try:
            # TensionEngine stores last tensions in _last_tensions
            tens = getattr(o, '_last_tensions', None)
            if tens is not None:
                drives.coherence = max(0.0, 1.0 - float(tens.contradiction_pressure))
            else:
                drives.coherence = self._last_drives.coherence
        except Exception:
            drives.coherence = self._last_drives.coherence

        # ── Energy ────────────────────────────────────────────────────────
        try:
            drives.energy = float(o.energy.level()) / 100.0
        except Exception:
            drives.energy = self._last_drives.energy

        # ── Social (time-decay since last user interaction) ───────────────
        idle_secs = time.time() - self._last_user_interaction
        # Full social drive after 2 hours without interaction
        drives.social = min(1.0, idle_secs / 7200.0)

        # ── Goal progress ─────────────────────────────────────────────────
        try:
            dominant = o.goal_ecology.dominant_drive(o.energy.level())
            drives.goal_progress = 1.0 - float(dominant.urgency)
        except Exception:
            drives.goal_progress = self._last_drives.goal_progress

        # ── Homeostasis ───────────────────────────────────────────────────
        try:
            report = getattr(o.homeostasis, '_last_report', None)
            if report is not None:
                drives.homeostasis = float(report.homeostasis_score)
            else:
                drives.homeostasis = self._last_drives.homeostasis
        except Exception:
            drives.homeostasis = self._last_drives.homeostasis

        self._last_drives = drives
        return drives
