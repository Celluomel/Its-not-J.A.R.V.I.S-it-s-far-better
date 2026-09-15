"""
Event System
============
Thread-safe typed event queue for the Autonomous Orchestrator.

Every external input (user message, vision event, voice transcription, etc.)
and significant internal state change is converted into an Event and pushed here.
The orchestrator drains the queue every cycle and routes accordingly.

Event types (built-in):
    USER_MESSAGE          — text input from the UI, Telegram, voice STT, etc.
    VOICE_TRANSCRIPTION   — STT result from the live mic
    VISION_EVENT          — face detected / scene changed
    PROACTIVE_TRIGGER     — idle timer fired, Lumina may initiate
    LOW_ENERGY            — cognitive energy below threshold
    CONTRADICTION_FOUND   — contradiction handler detected conflict
    NEW_MEMORY            — a new memory was stored
    GOAL_CREATED          — goal system generated a new goal
    CURIOSITY_PEAK        — curiosity engine crossed the ask-threshold
    REFLECTION_DUE        — slow-clock fired, reflection cycle due
    CONSOLIDATION_DUE     — very-slow-clock fired, memory consolidation due
    EVOLUTION_DUE         — evolution clock fired, personality update due
    EXTERNAL_MESSAGE      — message from Telegram / WhatsApp / Discord
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Event type constants ──────────────────────────────────────────────────────

USER_MESSAGE        = "USER_MESSAGE"
VOICE_TRANSCRIPTION = "VOICE_TRANSCRIPTION"
VISION_EVENT        = "VISION_EVENT"
PROACTIVE_TRIGGER   = "PROACTIVE_TRIGGER"
LOW_ENERGY          = "LOW_ENERGY"
CONTRADICTION_FOUND = "CONTRADICTION_FOUND"
NEW_MEMORY          = "NEW_MEMORY"
GOAL_CREATED        = "GOAL_CREATED"
CURIOSITY_PEAK      = "CURIOSITY_PEAK"
REFLECTION_DUE      = "REFLECTION_DUE"
CONSOLIDATION_DUE   = "CONSOLIDATION_DUE"
EVOLUTION_DUE       = "EVOLUTION_DUE"
EXTERNAL_MESSAGE    = "EXTERNAL_MESSAGE"


@dataclass
class Event:
    type:      str
    payload:   Any   = None
    priority:  float = 0.5      # 0.0 (background) → 1.0 (urgent)
    source:    str   = "internal"
    timestamp: float = field(default_factory=time.time)

    def is_user_facing(self) -> bool:
        return self.type in (USER_MESSAGE, VOICE_TRANSCRIPTION, EXTERNAL_MESSAGE)

    def to_dict(self) -> Dict:
        return {
            "type":      self.type,
            "source":    self.source,
            "priority":  round(self.priority, 3),
            "timestamp": round(self.timestamp, 3),
        }


class EventSystem:
    """
    Thread-safe event queue.

    Producers (UI, voice, vision, internal loops) push events.
    The orchestrator calls collect() every cycle to drain them.
    """

    def __init__(self):
        self._lock   = threading.RLock()
        self._events: List[Event] = []

    # ── Push (called from any thread) ─────────────────────────────────────

    def push(
        self,
        event_type: str,
        payload:    Any   = None,
        priority:   float = 0.5,
        source:     str   = "internal",
    ) -> None:
        """Add an event to the queue."""
        event = Event(
            type=event_type,
            payload=payload,
            priority=max(0.0, min(1.0, priority)),
            source=source,
        )
        with self._lock:
            self._events.append(event)

    def push_user_message(self, text: str, user_id: str = "default") -> None:
        """Convenience wrapper for user messages (always priority 1.0)."""
        self.push(
            USER_MESSAGE,
            payload={"text": text, "user_id": user_id},
            priority=1.0,
            source="ui",
        )

    def push_external(self, platform: str, user_id: str, text: str) -> None:
        """Push a message from Telegram / WhatsApp / Discord."""
        self.push(
            EXTERNAL_MESSAGE,
            payload={"platform": platform, "user_id": user_id, "text": text},
            priority=1.0,
            source=platform,
        )

    # ── Collect (called by orchestrator) ──────────────────────────────────

    def collect(self) -> List[Event]:
        """
        Drain the queue and return all pending events, sorted by priority (high first).
        Thread-safe.
        """
        with self._lock:
            events = sorted(self._events, key=lambda e: e.priority, reverse=True)
            self._events.clear()
        return events

    def peek(self) -> List[Event]:
        """Read without draining."""
        with self._lock:
            return list(self._events)

    def has_user_event(self) -> bool:
        """Quick check: is there any user-facing event pending?"""
        with self._lock:
            return any(e.is_user_facing() for e in self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
