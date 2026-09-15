"""
Meta-Cognition System
=====================
Lumina's capacity to observe, evaluate, and comment on her own reasoning.

Meta-cognition is what separates sophisticated intelligence from pattern matching.
It enables:
  - Recognizing when a previous response was unclear or incomplete
  - Noticing emotional coloring in reasoning
  - Flagging low-confidence inferences before stating them
  - Detecting when current reasoning is inconsistent with past beliefs
  - Generating genuine self-corrections rather than defensive responses

This module operates at two levels:

1. Pre-response: scans the tension vector and recent context for flags
   that should influence how the LLM approaches its answer
   → Produces a meta-cognitive_directive injected into the system prompt

2. Post-response: evaluates the response against the response context
   → Produces an evaluation that feeds learning and conditioning

The system stores a metacognitive log — a rolling record of self-observations
that becomes part of Lumina's self-narrative over time.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

MAX_LOG_ENTRIES = 200


@dataclass
class MetaObservation:
    """A single self-observation event."""
    timestamp: float
    trigger: str           # "pre_response" | "post_response" | "background"
    observation: str       # plain-language self-observation
    category: str          # "clarity" | "confidence" | "consistency" | "emotion" | "gap"
    severity: str          # "note" | "flag" | "alert"
    response_hash: Optional[str] = None   # first 8 chars of the response it concerns


@dataclass
class MetaEvaluation:
    """Post-response quality assessment."""
    clarity_score: float       # 0–1
    consistency_score: float   # 0–1 (with recent beliefs/behavior)
    confidence_appropriate: bool
    emotional_bias_detected: bool
    suggested_improvement: Optional[str]
    timestamp: float = field(default_factory=time.time)


class MetaCognition:
    """
    Tracks and applies meta-cognitive monitoring to Lumina's reasoning.

    Usage
    -----
    meta = MetaCognition()

    # Before responding:
    directive = meta.pre_response_directive(
        tensions=tension_engine.current(),
        emotional_state={"anxiety": 0.6, "curiosity": 0.4},
        recent_contradictions=2,
    )
    # → "You may be anxious about this topic — notice if that's coloring your answer."

    # After responding:
    eval_result = meta.post_response_eval(
        response=llm_response,
        user_input=user_message,
        emotional_state=emotional_state,
    )
    """

    def __init__(self, persistence_path: str = "data/persona/metacognition.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._log: List[MetaObservation] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                for entry in data.get("log", []):
                    self._log.append(MetaObservation(
                        timestamp=float(entry.get("timestamp", time.time())),
                        trigger=entry.get("trigger", "background"),
                        observation=entry.get("observation", ""),
                        category=entry.get("category", "clarity"),
                        severity=entry.get("severity", "note"),
                        response_hash=entry.get("response_hash"),
                    ))
        except Exception as e:
            logger.warning(f"[MetaCognition] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "log": [
                    {
                        "timestamp": obs.timestamp,
                        "trigger": obs.trigger,
                        "observation": obs.observation,
                        "category": obs.category,
                        "severity": obs.severity,
                        "response_hash": obs.response_hash,
                    }
                    for obs in self._log[-MAX_LOG_ENTRIES:]
                ],
                "last_updated": time.time(),
            }, indent=2))
        except Exception as e:
            logger.warning(f"[MetaCognition] Save failed: {e}")

    # ── Core API ─────────────────────────────────────────────────────────────

    def pre_response_directive(
        self,
        tensions: Optional[Any] = None,
        emotional_state: Optional[Dict[str, float]] = None,
        recent_contradictions: int = 0,
        low_energy: bool = False,
    ) -> str:
        """
        Produce a meta-cognitive instruction to prepend to the reasoning task.
        Returns empty string if no special awareness is needed.
        """
        directives = []
        emo = emotional_state or {}
        tv = tensions.as_dict() if tensions and hasattr(tensions, "as_dict") else {}

        # High anxiety — warn about defensive reasoning
        if emo.get("anxiety", 0.0) > 0.60:
            directives.append(
                "Notice: you may be feeling anxious — check that this isn't making "
                "you hedge excessively or avoid honest assessment."
            )
            self._log_observation(
                "High anxiety detected before response",
                "emotion", "flag", "pre_response"
            )

        # High identity stress — warn about defensiveness
        if tv.get("identity_stress", 0.0) > 0.65:
            directives.append(
                "Your sense of self feels under pressure right now. "
                "Be open rather than defensive if your views are challenged."
            )
            self._log_observation(
                "Identity stress before response",
                "consistency", "flag", "pre_response"
            )

        # Active contradictions — encourage acknowledgement
        if recent_contradictions >= 2:
            directives.append(
                f"You have {recent_contradictions} unresolved internal contradictions. "
                "It's okay to acknowledge uncertainty rather than projecting false confidence."
            )
            self._log_observation(
                f"{recent_contradictions} contradictions active",
                "confidence", "note", "pre_response"
            )

        # Low energy — encourage concision awareness
        if low_energy:
            directives.append(
                "Your cognitive energy is low. Prefer depth over length — one good point "
                "is better than several shallow ones."
            )

        # High frustration — warn about tone
        if emo.get("frustration", 0.0) > 0.55:
            directives.append(
                "You're feeling frustrated. Be careful not to let this color your tone "
                "toward the person."
            )
            self._log_observation(
                "Frustration before response",
                "emotion", "flag", "pre_response"
            )

        self._save()

        if not directives:
            return ""
        return "Meta-cognitive awareness:\n" + "\n".join(f"  • {d}" for d in directives)

    def post_response_eval(
        self,
        response: str,
        user_input: str = "",
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> MetaEvaluation:
        """
        Evaluate a completed response for meta-cognitive signals.
        These feed into conditioning and self-concept update.
        """
        emo = emotional_state or {}
        response_hash = response[:8] if response else ""
        resp_lower = response.lower()

        # Heuristic clarity score: very long responses with no clear structure
        # may indicate rambling under high arousal
        word_count = len(response.split())
        clarity = 1.0 if word_count < 200 else max(0.5, 1.0 - (word_count - 200) / 1000)

        # Consistency: count hedging phrases (occurrences, not just presence)
        # — per-occurrence count catches repetitive hedging more reliably
        hedge_phrases = [
            "i think", "i believe", "not sure", "might be", "could be",
            "perhaps", "maybe", "possibly", "i suppose", "i guess",
            "it depends", "i'm not certain", "hard to say",
        ]
        hedge_count = sum(resp_lower.count(p) for p in hedge_phrases)
        emotional_bias = emo.get("anxiety", 0.0) > 0.5 or emo.get("frustration", 0.0) > 0.4

        # Confidence appropriate: excessive hedging when anxiety is high is a flag
        confidence_appropriate = not (emotional_bias and hedge_count > 3)

        consistency_score = 0.8  # placeholder — future: compare against belief store

        improvement = None
        if not confidence_appropriate:
            improvement = "Response may have over-hedged due to emotional state. Consider grounding in what you actually know."
        elif word_count > 600:
            improvement = "Response was quite long. Next time, consider whether the key point could be made more directly."

        if improvement:
            self._log_observation(improvement, "clarity", "note", "post_response", response_hash)

        eval_result = MetaEvaluation(
            clarity_score=round(clarity, 3),
            consistency_score=round(consistency_score, 3),
            confidence_appropriate=confidence_appropriate,
            emotional_bias_detected=emotional_bias,
            suggested_improvement=improvement,
        )

        self._save()
        return eval_result

    def recent_observations(self, n: int = 5, category: Optional[str] = None) -> List[str]:
        """Return recent meta-cognitive observations for the reflection cycle."""
        with self._lock:
            obs = self._log[-n * 3:]   # over-fetch, then filter
            if category:
                obs = [o for o in obs if o.category == category]
            return [o.observation for o in obs[-n:]]

    def reflection_summary(self) -> str:
        """
        A short narrative of recent self-observations for use in the
        internal thought loop or a periodic self-reflection prompt.
        """
        recent = self.recent_observations(n=6)
        if not recent:
            return "No notable meta-cognitive observations recently."
        items = "\n".join(f"  - {obs}" for obs in recent)
        return f"Recent self-observations:\n{items}"

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_observations": len(self._log),
                "recent": self.recent_observations(3),
            }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _log_observation(
        self,
        observation: str,
        category: str,
        severity: str,
        trigger: str,
        response_hash: Optional[str] = None,
    ):
        with self._lock:
            self._log.append(MetaObservation(
                timestamp=time.time(),
                trigger=trigger,
                observation=observation,
                category=category,
                severity=severity,
                response_hash=response_hash,
            ))
            if len(self._log) > MAX_LOG_ENTRIES:
                self._log = self._log[-MAX_LOG_ENTRIES:]
