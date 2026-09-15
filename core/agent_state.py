"""
AgentState – runtime state for the embodied agent (Section 3 of spec).

Tracks reunion type, engagement level, environment vibe, idle cooldown,
and provides the time-classification helpers required by the controller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  VisionOutput  (Section 3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VisionOutput:
    user_present: bool = False
    attention_direction: str = "unknown"   # toward_robot | away | unknown
    activity: str = "unknown"
    expression: str = "neutral"
    notable_change: Optional[str] = None

    @classmethod
    def from_raw_text(cls, raw: str) -> "VisionOutput":
        """
        Parse natural-language vision analysis into a structured VisionOutput.
        Uses simple keyword heuristics so the LLM is *never* aware of tool output.
        """
        lower = raw.lower()

        user_present = any(w in lower for w in [
            "person", "human", "user", "face", "someone", "man", "woman",
            "individual", "people"
        ])

        if any(w in lower for w in ["looking at camera", "facing the robot",
                                     "toward the robot", "making eye contact",
                                     "facing you"]):
            attention = "toward_robot"
        elif any(w in lower for w in ["away", "turned", "looking elsewhere",
                                       "not looking", "distracted", "back to"]):
            attention = "away"
        else:
            attention = "unknown"

        # Activity – pick first recognisable keyword
        activity = "unknown"
        for keyword, label in [
            ("typing", "typing"), ("reading", "reading"), ("eating", "eating"),
            ("sleeping", "sleeping"), ("talking", "talking"),
            ("walking", "walking"), ("sitting", "sitting"),
            ("standing", "standing"), ("working", "working"),
            ("watching", "watching"), ("phone", "on phone"),
        ]:
            if keyword in lower:
                activity = label
                break

        # Expression
        expression = "neutral"
        for keyword, label in [
            ("smiling", "happy"), ("happy", "happy"), ("laughing", "happy"),
            ("sad", "sad"), ("frown", "sad"), ("crying", "sad"),
            ("angry", "angry"), ("frustrated", "angry"),
            ("surprised", "surprised"), ("confused", "confused"),
            ("focused", "focused"), ("tired", "tired"),
        ]:
            if keyword in lower:
                expression = label
                break

        notable = None
        for phrase in ["suddenly", "just", "now", "changed", "new", "different",
                        "entered", "left", "appeared", "moved"]:
            if phrase in lower:
                # Extract a short surrounding snippet as the notable change
                idx = lower.find(phrase)
                snippet = raw[max(0, idx - 10): idx + 40].strip()
                notable = snippet
                break

        return cls(
            user_present=user_present,
            attention_direction=attention,
            activity=activity,
            expression=expression,
            notable_change=notable,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MemoryEntry  (Section 3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    event: str
    user_emotion: str = "neutral"
    intensity: float = 0.5          # 0.0 – 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1         # higher = fades faster

    def effective_weight(self) -> float:
        """Return salience score after time-based decay."""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        return max(0.0, self.intensity - self.decay_rate * age_hours)


# ─────────────────────────────────────────────────────────────────────────────
#  AgentState  (Section 3)
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds for reunion classification (in seconds)
SHORT_ABSENCE_THRESHOLD = 5 * 60       # < 5 min  → continuous
LONG_ABSENCE_THRESHOLD  = 60 * 60      # > 60 min → long
IDLE_BEFORE_PROACTIVE   = 600          # 10 min of silence before proactive fires (was 180)


@dataclass
class AgentState:
    last_interaction: datetime = field(default_factory=datetime.now)
    reunion_type: str = "continuous"    # continuous | short | long
    engagement_level: float = 0.5      # 0.0 – 1.0
    environment_vibe: str = "neutral"
    user_present: bool = False
    idle_cooldown_until: datetime = field(default_factory=datetime.now)  # updated by set_idle_cooldown

    # ── time classification ───────────────────────────────────────────
    def classify_reunion(self) -> str:
        """
        Classify how long the user has been away since last interaction
        (Section 4, step 2).
        """
        gap = (datetime.now() - self.last_interaction).total_seconds()
        if gap < SHORT_ABSENCE_THRESHOLD:
            self.reunion_type = "continuous"
        elif gap < LONG_ABSENCE_THRESHOLD:
            self.reunion_type = "short"
        else:
            self.reunion_type = "long"
        return self.reunion_type

    # ── engagement tracking ───────────────────────────────────────────
    def update_engagement(self, vision: Optional[VisionOutput]) -> float:
        """
        Raise/lower engagement_level based on vision attention direction
        (Section 4, step 5).
        """
        if vision is None:
            return self.engagement_level

        delta = 0.0
        if vision.attention_direction == "toward_robot":
            delta = +0.15
        elif vision.attention_direction == "away":
            delta = -0.10

        self.engagement_level = max(0.0, min(1.0, self.engagement_level + delta))
        return self.engagement_level

    # ── idle cooldown check ───────────────────────────────────────────
    def is_idle_trigger_ready(self) -> bool:
        """
        Return True only when BOTH conditions are met:
        1. The post-proactive cooldown has expired (prevents back-to-back proactives)
        2. The user has actually been quiet for IDLE_BEFORE_PROACTIVE seconds
           (prevents firing mid-conversation just because cooldown expired)
        """
        if datetime.now() < self.idle_cooldown_until:
            return False
        seconds_since_last = (datetime.now() - self.last_interaction).total_seconds()
        return seconds_since_last >= IDLE_BEFORE_PROACTIVE

    def set_idle_cooldown(self, seconds: float = 90.0):
        from datetime import timedelta
        self.idle_cooldown_until = datetime.now() + timedelta(seconds=seconds)

    # ── touch last-interaction ────────────────────────────────────────
    def mark_interaction(self):
        """Reset both the last-interaction timestamp and the proactive cooldown."""
        from datetime import timedelta
        self.last_interaction = datetime.now()
        # Require IDLE_BEFORE_PROACTIVE seconds of silence before proactive can fire
        self.idle_cooldown_until = datetime.now() + timedelta(seconds=IDLE_BEFORE_PROACTIVE)


# Module-level singleton used by the controller
agent_state = AgentState()
