"""
Cognitive Energy System
=======================
Simulates the biological constraint that deep cognition is costly.

Lumina cannot reason at full depth indefinitely. Energy drains during heavy
processing and regenerates during idle time or light tasks.

Architecture:
  - energy: float in [0, 100] — the current cognitive resource pool
  - drain rates are activity-specific (deep reasoning costs more than small talk)
  - regeneration is time-based: idle seconds → energy recovery
  - capacity modulates LLM temperature, response depth, curiosity activation,
    and internal loop activity — making behavior naturally variable

Energy levels produce three operational modes:
  DEEP   (70–100): full reasoning, high creativity, curiosity active
  NORMAL (40–70):  standard responses, moderate exploration
  LOW    (0–40):   concise answers, reduced internal loops, recovery priority

This is NOT a hard blocker — low energy degrades quality, it does not halt
cognition. Mimics human cognitive fatigue.
"""

import time
import json
import logging
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Dict, Any

logger = logging.getLogger(__name__)

# ── Drain constants (energy units per event) ─────────────────────────────────
DRAIN = {
    "deep_reasoning":         18.0,   # complex multi-step answer
    "contradiction_resolve":  22.0,   # epistemically costly
    "research_session":       25.0,   # external web search + synthesis
    "standard_reply":          8.0,   # normal conversational turn
    "self_reflection":        12.0,   # introspective cycle
    "memory_consolidation":    6.0,   # background maintenance
    "small_talk":              3.0,   # simple greeting / acknowledgement
}

# Regeneration rate: energy units per second of relative idle
REGEN_RATE_PER_SECOND = 0.04   # ~144 units/hour ≈ full tank from empty in ~42 min

OperationMode = Literal["deep", "normal", "low"]


@dataclass
class EnergySnapshot:
    energy: float
    mode: OperationMode
    timestamp: float


class CognitiveEnergy:
    """
    Manages Lumina's cognitive energy pool.

    Usage
    -----
    energy = CognitiveEnergy(persistence_path="data/persona/energy.json")

    # Before a response:
    energy.regenerate()           # apply time-based recovery first
    mode = energy.mode()          # "deep" | "normal" | "low"
    energy.drain("deep_reasoning")

    # Query for downstream modulation:
    energy.temperature_hint()     # float 0.4–0.9
    energy.depth_hint()           # int 1–3
    """

    def __init__(self, persistence_path: str = "data/persona/energy.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._energy: float = 80.0
        self._last_update: float = time.time()
        self._history: list[EnergySnapshot] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._energy = float(data.get("energy", 80.0))
                self._last_update = float(data.get("last_update", time.time()))
                logger.debug(f"[CognitiveEnergy] Loaded energy={self._energy:.1f}")
        except Exception as e:
            logger.warning(f"[CognitiveEnergy] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "energy": round(self._energy, 2),
                "last_update": self._last_update,
            }, indent=2))
        except Exception as e:
            logger.warning(f"[CognitiveEnergy] Save failed: {e}")

    # ── Core API ─────────────────────────────────────────────────────────────

    def regenerate(self) -> float:
        """Apply time-based regeneration. Call this before each interaction."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            gain = elapsed * REGEN_RATE_PER_SECOND
            self._energy = min(100.0, self._energy + gain)
            self._last_update = now
            self._save()
            return self._energy

    def drain(self, activity: str, multiplier: float = 1.0) -> float:
        """
        Consume energy for a cognitive activity.

        Parameters
        ----------
        activity   : key from DRAIN dict, or a custom float
        multiplier : scale factor (e.g. 0.5 for a quick version of the task)
        """
        with self._lock:
            amount = DRAIN.get(activity, 8.0) * multiplier
            self._energy = max(0.0, self._energy - amount)
            self._history.append(EnergySnapshot(
                energy=self._energy,
                mode=self._mode_raw(),
                timestamp=time.time(),
            ))
            if len(self._history) > 50:
                self._history = self._history[-50:]
            self._save()
            logger.debug(f"[CognitiveEnergy] Drained {amount:.1f} ({activity}), remaining={self._energy:.1f}")
            return self._energy

    def level(self) -> float:
        """Return current energy [0, 100]."""
        with self._lock:
            return self._energy

    def _mode_raw(self) -> OperationMode:
        if self._energy > 70:
            return "deep"
        if self._energy > 40:
            return "normal"
        return "low"

    def mode(self) -> OperationMode:
        """Return operational mode string."""
        with self._lock:
            return self._mode_raw()

    # ── Downstream hints ─────────────────────────────────────────────────────

    def temperature_hint(self) -> float:
        """
        Suggested LLM temperature based on energy level.
        High energy → more creative/exploratory. Low energy → more conservative.
        """
        m = self.mode()
        return {"deep": 0.85, "normal": 0.65, "low": 0.42}[m]

    def depth_hint(self) -> int:
        """
        Reasoning depth tier: 3=deep, 2=normal, 1=low.
        Use to select few-shot count, CoT instructions, etc.
        """
        return {"deep": 3, "normal": 2, "low": 1}[self.mode()]

    def curiosity_active(self) -> bool:
        """Curiosity-driven exploration is only affordable when energy is sufficient."""
        return self.level() > 45.0

    def verbose_ok(self) -> bool:
        """Whether Lumina should permit long elaborated answers."""
        return self.level() > 50.0

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "energy": round(self._energy, 1),
                "mode": self._mode_raw(),
                "temperature_hint": self.temperature_hint(),
                "depth_hint": self.depth_hint(),
                "curiosity_active": self.curiosity_active(),
            }

    def prompt_fragment(self) -> str:
        """
        One-liner for system prompt injection, so the LLM understands the
        current cognitive state.
        """
        m = self.mode()
        e = round(self._energy, 0)
        descriptions = {
            "deep":   f"cognitive energy is high ({e}/100) — think deeply, explore freely",
            "normal": f"cognitive energy is moderate ({e}/100) — balanced responses",
            "low":    f"cognitive energy is low ({e}/100) — be concise and recover naturally",
        }
        return descriptions[m]
