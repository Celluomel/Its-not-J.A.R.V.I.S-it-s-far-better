"""
Cognitive Attractor System
==========================
The stability layer that prevents Lumina from becoming a different agent after
each interaction.

Most autonomous agents suffer from personality drift: a few unusual interactions
can radically shift their behaviour.  The Attractor System introduces
psychological inertia — stable trait values that change very slowly and
always pull the agent back toward its baseline character.

Concept
-------
Each trait is a float [0.0, 1.0].  External events can nudge a trait up or
down, but:
  1. Each nudge is small (default delta ≈ 0.01–0.02)
  2. A slow "return force" pulls traits back toward their baseline
  3. Traits are persisted so personality survives restarts

Traits
------
  curiosity          — drive to explore and ask questions
  analytical_depth   — tendency toward thorough reasoning
  social_engagement  — warmth and interest in the user
  risk_tolerance     — willingness to speculate or be unconventional
  consistency        — preference for stable, predictable behaviour
  openness           — receptivity to new ideas
  empathy            — sensitivity to emotional tone

Integration
-----------
  attractors = CognitiveAttractorSystem()

  # Read in ActivitySelector / DriveSystem:
  if attractors.get("curiosity") > 0.75:
      # more exploration goals

  # Nudge after an interaction:
  attractors.nudge("curiosity", +0.01)   # success in exploration
  attractors.nudge("risk_tolerance", -0.01)  # risky response caused confusion

  # PersonaBridge injection:
  ctx["personality_attractors"] = attractors.prompt_fragment()
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PERSISTENCE_PATH = "data/persona/attractors.json"

# Trait baselines — Lumina's default character
BASELINES: Dict[str, float] = {
    "curiosity":          0.72,
    "analytical_depth":   0.68,
    "social_engagement":  0.65,
    "risk_tolerance":     0.42,
    "consistency":        0.75,
    "openness":           0.70,
    "empathy":            0.68,
}

# How fast traits return to baseline per background tick (per 30 s cycle)
RETURN_RATE = 0.003


class CognitiveAttractorSystem:
    """
    Maintains Lumina's stable personality traits with slow-drift dynamics.

    Thread-safe.  Persists to JSON.
    """

    def __init__(self, persistence_path: str = PERSISTENCE_PATH):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._traits: Dict[str, float] = dict(BASELINES)
        self._last_tick = time.time()
        self._load()
        logger.info("🧲 CognitiveAttractorSystem initialised")

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, trait: str) -> float:
        """Return current value of a trait (0–1)."""
        with self._lock:
            return self._traits.get(trait, 0.5)

    def all(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._traits)

    # ── Write ─────────────────────────────────────────────────────────────

    def nudge(self, trait: str, delta: float) -> None:
        """
        Nudge a trait by delta.  Small values only — large shifts are clamped.

        Recommended delta range: ±0.005 to ±0.025 per interaction.

        Scaled by CSIS (Cross-Session Influence Score, emergence_metrics.py)
        before being applied. CSIS is Lumina's own measurement of whether
        experience is demonstrably persisting across sessions right now;
        previously that measurement was computed and displayed but never
        read by anything (audited: every call site into the collector was
        write-only). This is the first real consumer — when persistence
        hasn't been demonstrated (CSIS low or not yet available), nudges
        land softer; once it has, they land at closer to full strength.
        """
        if trait not in BASELINES:
            return
        # Cap single nudge magnitude to prevent runaway shifts
        delta = max(-0.05, min(0.05, delta))
        delta *= self._persistence_factor()
        with self._lock:
            self._traits[trait] = max(0.0, min(1.0, self._traits[trait] + delta))
        self._save()

    def _persistence_factor(self) -> float:
        """
        CSIS-derived scale for nudge magnitude, in [0.4, 1.0].

        0.4 floor: before enough history exists for CSIS to mean anything
        (get_latest() returns None pre-first-tick), or as a lower bound
        once it does, nudges still land at meaningful strength rather than
        the attractor system going near-inert during the system's early
        life — that would itself be a silent behavior change bigger than
        intended. No snapshot yet → 1.0 (identical to pre-fix behavior).
        """
        try:
            from cognition.emergence_metrics import get_collector
            collector = get_collector()
            snap = collector.get_latest() if collector else None
            if snap is None:
                return 1.0
            csis = max(0.0, min(1.0, snap.csis))
            return 0.4 + 0.6 * csis
        except Exception as e:
            logger.debug(f"AttractorSystem._persistence_factor: {e}")
            return 1.0

    def tick_return_force(self) -> None:
        """
        Pull all traits gently back toward their baselines.
        Call on each background maintenance cycle (~30 s).

        Return rate is scaled by WDS (Weight Drift Stability,
        emergence_metrics.py) — the same audited gap as nudge()'s CSIS use
        above: WDS is Lumina's own measurement of whether recent trait
        movement has been stable or chaotic, previously computed and
        never read by anything. Low WDS (chaotic drift) pulls harder back
        toward baseline to restore stability; high WDS (proven stable
        movement) relaxes the pull so a real, demonstrated shift gets to
        consolidate toward becoming the new baseline instead of being
        dragged back by a flat constant forever. No snapshot yet → 1.0
        (identical to pre-fix behavior).
        """
        rate = RETURN_RATE * self._stability_factor()
        with self._lock:
            for trait, baseline in BASELINES.items():
                current = self._traits[trait]
                diff    = baseline - current
                # Apply a fraction of the gap (proportional return)
                self._traits[trait] = current + diff * rate
            self._last_tick = time.time()
        self._save()

    def _stability_factor(self) -> float:
        """
        WDS-derived multiplier on RETURN_RATE, in [0.5, 1.5].

        wds=0 (chaotic) → 1.5x — stronger correction back to baseline.
        wds=1 (proven stable) → 0.5x — weaker pull, room to consolidate.
        Bounded to a 3x total spread either way of the base rate so this
        never overrides the return force entirely in either direction.
        """
        try:
            from cognition.emergence_metrics import get_collector
            collector = get_collector()
            snap = collector.get_latest() if collector else None
            if snap is None:
                return 1.0
            wds = max(0.0, min(1.0, snap.wds))
            return 1.5 - wds
        except Exception as e:
            logger.debug(f"AttractorSystem._stability_factor: {e}")
            return 1.0

    # ── Influence helpers ─────────────────────────────────────────────────

    def influence_drives(self, drives) -> None:
        """
        Adjust a DriveVector in-place based on attractor values.
        Called by DriveSystem after computing base drives.
        """
        try:
            # High curiosity attractor amplifies curiosity drive
            cur_bias = (self.get("curiosity") - 0.5) * 0.10
            drives.curiosity = max(0.0, min(1.0, drives.curiosity + cur_bias))

            # Low consistency attractor makes homeostasis less important
            con_bias = (self.get("consistency") - 0.5) * 0.05
            drives.homeostasis = max(0.0, min(1.0, drives.homeostasis + con_bias))
        except Exception as e:
            logger.debug(f"AttractorSystem.influence_drives: {e}")

    # ── Prompt integration ────────────────────────────────────────────────

    def prompt_fragment(self) -> str:
        """
        Short description of Lumina's current personality state for the system prompt.
        """
        with self._lock:
            high  = [t for t, v in self._traits.items() if v > 0.70]
            low   = [t for t, v in self._traits.items() if v < 0.35]

        parts = []
        if high:
            parts.append(f"strong: {', '.join(high)}")
        if low:
            parts.append(f"muted: {', '.join(low)}")

        return "[Personality attractors] " + ("; ".join(parts) if parts else "balanced")

    def summary(self) -> Dict:
        with self._lock:
            return {
                "traits": {k: round(v, 3) for k, v in self._traits.items()},
                "baselines": BASELINES,
            }

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                for t, v in data.get("traits", {}).items():
                    if t in BASELINES:
                        self._traits[t] = float(v)
                logger.debug("🧲 Attractors loaded from disk")
        except Exception as e:
            logger.debug(f"Attractors load skipped: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"traits": self._traits}, indent=2))
        except Exception as e:
            logger.debug(f"Attractors save failed: {e}")
