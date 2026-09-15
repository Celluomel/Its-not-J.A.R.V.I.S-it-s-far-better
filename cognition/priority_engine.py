"""
Priority Engine
===============
Implements the "cognitive competition" step from Global Workspace Theory
(Bernard Baars / Stanislas Dehaene).

In a biological brain, many signals compete for access to the global workspace.
Only the winner becomes "conscious" — i.e., is broadcast to all other modules
and becomes the focus of reasoning.

This module replicates that mechanism:

  1. Any part of the system can add a Stimulus with a priority score.
  2. select() pops the highest-priority stimulus.
  3. The winner is broadcast to the GlobalWorkspace.
  4. The orchestrator/activity-selector sees the broadcast and acts.

Stimulus sources
----------------
  user_input        — always priority 1.0
  memory_recall     — relevant memory surfaces (0.4–0.7)
  contradiction     — conflict detected (0.7–0.9)
  curiosity_peak    — curiosity threshold crossed (0.6–0.8)
  goal_urgency      — high-urgency goal (0.5–0.85)
  vision_event      — face / scene detected (0.5–0.7)
  tool_option       — available tool for the current context (0.6–0.9)
  emotional_spike   — strong emotion detected (0.5–0.8)
  self_model_alert  — confidence drop / overload (0.6–0.85)

Integration
-----------
  engine = PriorityEngine()

  # Add stimuli from anywhere:
  engine.add("user_input",   {"text": "..."}, priority=1.0)
  engine.add("contradiction", {"topic": "X"},  priority=0.85)

  # In the orchestrator cycle:
  winner = engine.select()
  if winner:
      workspace.broadcast(winner["source"], winner["content"], winner["priority"])
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Stimuli older than this are discarded (prevent stale signals lingering)
MAX_AGE_SECONDS = 30.0


@dataclass(order=True)
class _HeapItem:
    neg_priority: float                            # negative so heapq gives max-heap
    timestamp:    float    = field(compare=False)
    stimulus:     Dict     = field(compare=False)


class PriorityEngine:
    """
    Stimulus competition queue.  Thread-safe.
    """

    def __init__(self):
        self._lock  = threading.RLock()
        self._queue: List[_HeapItem] = []

    # ── Add stimuli ───────────────────────────────────────────────────────

    def add(
        self,
        source:   str,
        content:  Any,
        priority: float,
    ) -> None:
        """
        Register a new stimulus.

        Parameters
        ----------
        source   : module name  e.g. "memory", "user_input"
        content  : arbitrary payload (string, dict, …)
        priority : 0.0–1.0
        """
        priority = max(0.0, min(1.0, priority))
        stimulus = {
            "source":    source,
            "content":   content,
            "priority":  priority,
            "timestamp": time.time(),
        }
        item = _HeapItem(neg_priority=-priority, timestamp=stimulus["timestamp"], stimulus=stimulus)
        with self._lock:
            heapq.heappush(self._queue, item)

    # ── Select winner ─────────────────────────────────────────────────────

    def select(self) -> Optional[Dict]:
        """
        Pop and return the highest-priority non-expired stimulus.
        Returns None if the queue is empty or all stimuli are stale.
        """
        now = time.time()
        with self._lock:
            # Drain expired items
            fresh = [i for i in self._queue if now - i.timestamp < MAX_AGE_SECONDS]
            if len(fresh) != len(self._queue):
                heapq.heapify(fresh)
                self._queue = fresh

            if not self._queue:
                return None

            return heapq.heappop(self._queue).stimulus

    def peek(self) -> Optional[Dict]:
        """Return the highest-priority stimulus without removing it."""
        now = time.time()
        with self._lock:
            for item in sorted(self._queue, key=lambda x: x.neg_priority):
                if now - item.timestamp < MAX_AGE_SECONDS:
                    return item.stimulus
        return None

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    # ── Convenience helpers ───────────────────────────────────────────────

    def add_user_input(self, text: str, user_id: str = "default") -> None:
        self.add("user_input", {"text": text, "user_id": user_id}, priority=1.0)

    def add_memory_recall(self, topic: str, relevance: float = 0.55) -> None:
        self.add("memory_recall", {"topic": topic}, priority=relevance)

    def add_contradiction(self, description: str, severity: float = 0.80) -> None:
        self.add("contradiction", {"description": description}, priority=severity)

    def add_curiosity_peak(self, topic: str, level: float = 0.70) -> None:
        self.add("curiosity_peak", {"topic": topic}, priority=level)

    def add_self_model_alert(self, reason: str, urgency: float = 0.75) -> None:
        self.add("self_model_alert", {"reason": reason}, priority=urgency)

    def add_tool_option(self, tool: str, relevance: float = 0.70) -> None:
        self.add("tool_option", {"tool": tool}, priority=relevance)

    def summary(self) -> Dict:
        with self._lock:
            return {
                "queue_depth": len(self._queue),
                "top": [
                    {"source": i.stimulus["source"], "priority": i.stimulus["priority"]}
                    for i in sorted(self._queue, key=lambda x: x.neg_priority)[:3]
                ],
            }
