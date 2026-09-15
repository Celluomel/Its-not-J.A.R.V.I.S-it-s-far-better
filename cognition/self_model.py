"""
Self-Model
==========
Lumina's introspective representation of itself.

Most agents know the world. Lumina also knows herself:

  - what she is good at (capability scores)
  - how she is performing right now (performance tracking)
  - what her current cognitive load is
  - what her confidence level is
  - where she has knowledge gaps

This module is read by the Drive System, Activity Selector, and PersonaBridge
to modulate reasoning style, goal generation, and prompt context.

It is also one of the rarest modules in open-source agent architectures.

Integration
-----------
  self_model = SelfModel()

  # After each interaction:
  self_model.record_interaction(success=True, domain="conversation")

  # In PersonaBridge context:
  ctx["self_model"] = self_model.prompt_fragment()

  # Drive system reads:
  if self_model.confidence < 0.4:
      drives.curiosity += 0.15   # seek new strategies
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PERSISTENCE_PATH = "data/persona/self_model.json"

# Capability domains
DOMAINS = [
    "conversation",
    "reasoning",
    "memory_analysis",
    "research",
    "creativity",
    "emotional_support",
    "code_analysis",
    "self_reflection",
]


@dataclass
class CapabilityRecord:
    """Tracks performance in a specific domain."""
    domain:   str
    score:    float = 0.65   # baseline competence
    success:  int   = 0
    failure:  int   = 0
    last_used: float = field(default_factory=time.time)

    def update(self, success: bool) -> None:
        if success:
            self.success += 1
            self.score = min(1.0, self.score + 0.02)
        else:
            self.failure += 1
            self.score = max(0.1, self.score - 0.03)
        self.last_used = time.time()

    def success_rate(self) -> float:
        total = self.success + self.failure
        return (self.success / total) if total > 0 else 0.5


class SelfModel:
    """
    Lumina's model of herself.

    Thread-safe.  Persists to JSON between sessions.
    """

    def __init__(self, persistence_path: str = PERSISTENCE_PATH):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()

        # Capability scores per domain
        self.capabilities: Dict[str, CapabilityRecord] = {
            d: CapabilityRecord(domain=d) for d in DOMAINS
        }

        # Current cognitive state
        self.confidence:     float = 0.65   # 0=very uncertain, 1=very confident
        self.cognitive_load: float = 0.20   # 0=idle, 1=overwhelmed
        self.learning_rate:  float = 0.50   # 0=stuck, 1=rapid learning

        # Interaction counters
        self.total_interactions: int = 0
        self.session_interactions: int = 0

        # Knowledge gap log (topics Lumina knows she doesn't know well)
        self.knowledge_gaps: List[str] = []

        self._load()
        logger.info("🪞 SelfModel initialised")

    # ── Record interactions ────────────────────────────────────────────────

    def record_interaction(
        self,
        success: bool,
        domain: str = "conversation",
        load_delta: float = 0.05,
    ) -> None:
        """
        Call after each user interaction.

        Parameters
        ----------
        success    : did the interaction go well?
        domain     : capability domain  (e.g. "research", "reasoning")
        load_delta : how much this interaction cost cognitively (0–1)
        """
        with self._lock:
            self.total_interactions += 1
            self.session_interactions += 1

            if domain in self.capabilities:
                self.capabilities[domain].update(success)

            # Update overall confidence (exponential moving average)
            target = 0.75 if success else 0.35
            self.confidence = 0.92 * self.confidence + 0.08 * target
            self.confidence = max(0.1, min(1.0, self.confidence))

            # Cognitive load decays naturally, spikes on interaction
            self.cognitive_load = min(1.0, self.cognitive_load + load_delta)

        self._save()

    def tick_load_decay(self, elapsed_secs: float = 30.0) -> None:
        """Call on each background cycle to decay cognitive load."""
        with self._lock:
            decay = elapsed_secs * 0.002   # ~6% per minute
            self.cognitive_load = max(0.0, self.cognitive_load - decay)

    def add_knowledge_gap(self, topic: str) -> None:
        with self._lock:
            if topic not in self.knowledge_gaps:
                self.knowledge_gaps.append(topic)
                self.knowledge_gaps = self.knowledge_gaps[-20:]   # keep last 20

    # ── Queries ───────────────────────────────────────────────────────────

    def capability(self, domain: str) -> float:
        """Return capability score for a domain (0–1)."""
        with self._lock:
            return self.capabilities.get(domain, CapabilityRecord(domain)).score

    def overall_success_rate(self) -> float:
        with self._lock:
            s = sum(c.success for c in self.capabilities.values())
            f = sum(c.failure for c in self.capabilities.values())
            return s / (s + f) if (s + f) > 0 else 0.5

    def weakest_domain(self) -> Optional[str]:
        with self._lock:
            if not self.capabilities:
                return None
            return min(self.capabilities, key=lambda k: self.capabilities[k].score)

    def strongest_domain(self) -> Optional[str]:
        with self._lock:
            if not self.capabilities:
                return None
            return max(self.capabilities, key=lambda k: self.capabilities[k].score)

    def is_overloaded(self) -> bool:
        return self.cognitive_load > 0.80

    def is_confident(self) -> bool:
        return self.confidence > 0.65

    # ── Prompt integration ────────────────────────────────────────────────

    def prompt_fragment(self) -> str:
        """
        Returns a short natural-language self-description for injection into
        the system prompt via PersonaBridge.
        """
        with self._lock:
            strongest = self.strongest_domain() or "conversation"
            weakest   = self.weakest_domain()   or "research"
            load_desc = (
                "overloaded" if self.cognitive_load > 0.8 else
                "busy"       if self.cognitive_load > 0.5 else
                "alert"      if self.cognitive_load > 0.2 else
                "rested"
            )
            conf_desc = (
                "very confident" if self.confidence > 0.80 else
                "confident"      if self.confidence > 0.60 else
                "cautious"       if self.confidence > 0.40 else
                "uncertain"
            )
            gaps = ", ".join(self.knowledge_gaps[-3:]) if self.knowledge_gaps else "none identified"

            return (
                f"[Self-awareness] Currently {load_desc} (load {self.cognitive_load:.0%}), "
                f"feeling {conf_desc}. "
                f"Strongest: {strongest} ({self.capability(strongest):.0%}). "
                f"Developing: {weakest} ({self.capability(weakest):.0%}). "
                f"Knowledge gaps: {gaps}."
            )

    def summary(self) -> Dict:
        with self._lock:
            return {
                "confidence":     round(self.confidence, 3),
                "cognitive_load": round(self.cognitive_load, 3),
                "learning_rate":  round(self.learning_rate, 3),
                "total_interactions": self.total_interactions,
                "overall_success_rate": round(self.overall_success_rate(), 3),
                "strongest_domain": self.strongest_domain(),
                "weakest_domain":   self.weakest_domain(),
                "knowledge_gaps":   self.knowledge_gaps[-5:],
            }

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self.confidence          = data.get("confidence", self.confidence)
                self.cognitive_load      = data.get("cognitive_load", self.cognitive_load)
                self.learning_rate       = data.get("learning_rate", self.learning_rate)
                self.total_interactions  = data.get("total_interactions", 0)
                self.knowledge_gaps      = data.get("knowledge_gaps", [])
                for d, vals in data.get("capabilities", {}).items():
                    if d in self.capabilities:
                        self.capabilities[d].score   = vals.get("score", 0.65)
                        self.capabilities[d].success = vals.get("success", 0)
                        self.capabilities[d].failure = vals.get("failure", 0)
                logger.debug("🪞 SelfModel loaded from disk")
        except Exception as e:
            logger.debug(f"SelfModel load skipped: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "confidence":        self.confidence,
                "cognitive_load":    self.cognitive_load,
                "learning_rate":     self.learning_rate,
                "total_interactions": self.total_interactions,
                "knowledge_gaps":    self.knowledge_gaps,
                "capabilities":      {
                    d: {"score": c.score, "success": c.success, "failure": c.failure}
                    for d, c in self.capabilities.items()
                },
            }
            import os as _os
            _tmp = self._path.with_suffix('.tmp')
            _tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            _os.replace(_tmp, self._path)
        except Exception as e:
            logger.debug(f"SelfModel save failed: {e}")
