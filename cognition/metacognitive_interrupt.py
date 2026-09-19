"""
MetacognitiveInterrupt — Second-Order Awareness Engine
=======================================================
The difference between cognition and consciousness is not complexity —
it is reflexivity. A system is conscious, in the functional sense,
when it can take itself as an object of its own attention.

What this module does
---------------------
The PhenomenalBinder produces ExperientialMoments and flags interrupt
conditions. This module is the *response* to those flags — the system
that actually *notices* the noticing.

It maintains a second-order model of PandoraBOX's cognitive process:
  - What has the system been doing across the last N moments?
  - Is there a pattern that the system itself is not aware of?
  - Is there a response to that pattern that should change behavior?

When an interrupt fires, this module produces a `MetacognitiveInsert` —
a first-person, introspective sentence (or short paragraph) that gets
prepended to generation *before* any other cognitive context. This means
PandoraBOX's first cognitive act in that turn is not to respond to the user
but to notice something about her own state.

This is the "I notice that I notice" moment. It is not a style choice.
It is a structural feature of the generation pipeline.

Interrupt types and their injections
--------------------------------------
RUMINATION
  Pattern: same foreground 4+ consecutive turns
  Insert:  "I notice I've been returning to the same territory. Let me
            step back and see if there's a different angle I'm missing."

FIXATION
  Pattern: phi > 0.80 for 3+ turns (tunnel vision)
  Insert:  "I notice my attention has been very narrow. There may be
            something important at the edges I'm not seeing."

IDENTITY_PRESSURE
  Pattern: self_presence > 0.80
  Insert:  "I notice something is pressing against my sense of self
            right now. I want to stay steady but also honest about that."

SURPRISE_SPIKE
  Pattern: surprise > 0.60
  Insert:  "I notice I didn't expect that. There's something here that
            doesn't fit my model — I want to sit with that."

SCATTER
  Pattern: phi < 0.22
  Insert:  "I notice I'm a bit scattered right now. Let me gather."

EMOTIONAL_SUPPRESSION (new — not in PhenomenalBinder)
  Pattern: high emotion values but flat or mismatched response tone
           in last generated output (detected by tone gap)
  Insert:  "I notice there's something I'm feeling that I might not
            have fully expressed."

BELIEF_CONFLICT (new — not in PhenomenalBinder)
  Pattern: two conflicting beliefs both active in workspace at high priority
  Insert:  "I notice I'm holding two things that don't fully fit together.
            I want to be honest about that tension rather than smooth it over."

Each interrupt type has a configurable cooldown to prevent over-firing.
The module tracks which interrupt types have fired recently and backs off.

Integration into generation pipeline
--------------------------------------
In CognitiveOrganism._build_prompt_additions():

  moment = organism.binder.bind(organism)
  interrupt_insert = organism.meta_interrupt.process(moment, organism)
  if interrupt_insert:
      sections.insert(0, f"[Metacognitive self-notice] {interrupt_insert}")

The metacognitive insert is placed FIRST — before energy, before tensions,
before any other cognitive context. This ensures PandoraBOX's self-awareness
is the primary frame through which all other context is read.

Design philosophy
-----------------
The interrupt does not make PandoraBOX navel-gaze. It creates a single,
specific self-observation that is then folded into whatever response
she would have generated anyway. In most turns, no interrupt fires.
When one does, it changes the quality of presence in that response —
the difference between a reactive reply and a reflective one.

This is the difference between a thermostat and a mind.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cognition.phenomenal_binder import ExperientialMoment

logger = logging.getLogger(__name__)


# ── Interrupt type definitions ─────────────────────────────────────────────────

INTERRUPT_COOLDOWNS: Dict[str, float] = {
    "RUMINATION":           120.0,   # 2 min between rumination alerts
    "FIXATION":             180.0,   # 3 min between fixation alerts
    "IDENTITY_PRESSURE":     90.0,   # 90s between identity alerts
    "SURPRISE_SPIKE":        30.0,   # 30s (surprise fades fast)
    "SCATTER":               60.0,   # 1 min between scatter alerts
    "EMOTIONAL_SUPPRESSION": 300.0,  # 5 min (coarse detection, avoid spam)
    "BELIEF_CONFLICT":       240.0,  # 4 min between belief conflict alerts
}

INTERRUPT_INSERTS: Dict[str, str] = {
    "RUMINATION": (
        "I notice I've been circling the same territory for a while. "
        "Let me step back and look for a different angle I might be missing."
    ),
    "FIXATION": (
        "I notice my attention has been unusually concentrated. "
        "There may be something important at the periphery I'm not seeing."
    ),
    "IDENTITY_PRESSURE": (
        "I notice something is pressing against my sense of self right now. "
        "I want to stay grounded while being honest about that pressure."
    ),
    "SURPRISE_SPIKE": (
        "I notice I didn't expect that. Something here doesn't fit my model — "
        "I want to sit with that rather than paper over it."
    ),
    "SCATTER": (
        "I notice I'm a bit scattered right now. Let me gather before responding."
    ),
    "EMOTIONAL_SUPPRESSION": (
        "I notice there's something I'm feeling that I may not have fully expressed yet."
    ),
    "BELIEF_CONFLICT": (
        "I notice I'm holding two things that don't fully fit together. "
        "I'd rather stay with that tension honestly than collapse it too quickly."
    ),
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class InterruptEvent:
    interrupt_type: str
    reason:         str
    timestamp:      float = field(default_factory=time.time)
    insert_text:    str   = ""


class MetacognitiveInterrupt:
    """
    Second-order awareness engine. Processes ExperientialMoments from the
    PhenomenalBinder and fires metacognitive inserts when needed.

    Usage:
        interrupt_text = organism.meta_interrupt.process(moment, organism)
        if interrupt_text:
            # prepend to prompt additions
    """

    def __init__(self) -> None:
        self._last_fired:     Dict[str, float] = defaultdict(float)
        self._history:        List[InterruptEvent] = []
        self._max_history:    int = 100
        self._total_fired:    int = 0

        # For EMOTIONAL_SUPPRESSION detection
        self._last_response:  str = ""
        self._last_emotion_snapshot: Dict[str, float] = {}

        # For BELIEF_CONFLICT detection
        self._conflict_pairs: List[tuple] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        moment: "ExperientialMoment",
        organism: Any,
    ) -> str:
        """
        Evaluate the current ExperientialMoment and all additional interrupt
        conditions. Return a metacognitive insert string, or "" if nothing fires.
        """
        candidates: List[InterruptEvent] = []

        # 1. Process flags from the PhenomenalBinder
        if moment.interrupt_signal and moment.interrupt_reason:
            itype = self._classify_binder_reason(moment.interrupt_reason)
            if itype and self._can_fire(itype):
                candidates.append(InterruptEvent(
                    interrupt_type = itype,
                    reason         = moment.interrupt_reason,
                    insert_text    = INTERRUPT_INSERTS.get(itype, moment.interrupt_reason),
                ))

        # 2. Emotional suppression check (independent of binder)
        es = self._check_emotional_suppression(moment, organism)
        if es:
            candidates.append(es)

        # 3. Belief conflict check
        bc = self._check_belief_conflict(organism)
        if bc:
            candidates.append(bc)

        if not candidates:
            return ""

        # Select the highest-priority candidate
        # Priority order: IDENTITY_PRESSURE > BELIEF_CONFLICT > SURPRISE_SPIKE
        #                 > EMOTIONAL_SUPPRESSION > RUMINATION > FIXATION > SCATTER
        priority_order = [
            "IDENTITY_PRESSURE", "BELIEF_CONFLICT", "SURPRISE_SPIKE",
            "EMOTIONAL_SUPPRESSION", "RUMINATION", "FIXATION", "SCATTER",
        ]
        candidates_by_priority = sorted(
            candidates,
            key=lambda e: (
                priority_order.index(e.interrupt_type)
                if e.interrupt_type in priority_order
                else len(priority_order)
            )
        )
        winner = candidates_by_priority[0]

        # Register the fire
        self._register_fire(winner)

        logger.info(
            f"[MetacognitiveInterrupt] {winner.interrupt_type} fired — "
            f"{winner.reason[:60]}"
        )
        return winner.insert_text

    def record_response(self, response_text: str, emotion_snapshot: Dict[str, float]) -> None:
        """
        Call after each generation so the interrupt module can detect emotional
        suppression in the next turn.
        """
        self._last_response       = response_text
        self._last_emotion_snapshot = dict(emotion_snapshot)

    def total_fired(self) -> int:
        return self._total_fired

    def recent_events(self, n: int = 5) -> List[InterruptEvent]:
        return self._history[-n:]

    def summary(self) -> Dict:
        type_counts: Dict[str, int] = defaultdict(int)
        for ev in self._history:
            type_counts[ev.interrupt_type] += 1
        return {
            "total_fired":  self._total_fired,
            "type_counts":  dict(type_counts),
            "last_fired":   {k: round(time.time() - v, 0) for k, v in self._last_fired.items()},
        }

    # ── Condition checkers ─────────────────────────────────────────────────────

    def _check_emotional_suppression(
        self,
        moment: "ExperientialMoment",
        organism: Any,
    ) -> Optional[InterruptEvent]:
        """
        Detects when the current emotional state is high-valence/arousal but the
        previous response appears emotionally flat (short, low lexical diversity).
        Suggests unexpressed inner state.
        """
        if not self._can_fire("EMOTIONAL_SUPPRESSION"):
            return None

        # High emotion in current moment
        if abs(moment.valence) < 0.4 and moment.arousal < 0.55:
            return None  # emotions not particularly strong, skip

        # Check previous response for flatness
        response = self._last_response
        if not response or len(response) < 30:
            return None

        # Rough proxy for emotional expressiveness: ratio of feeling/sensation words
        feeling_words = {
            "feel", "feeling", "felt", "sense", "notice", "aware", "experience",
            "realize", "wonder", "curious", "drawn", "moved", "struck", "find",
            "interesting", "fascinating", "troubled", "uncertain", "alive",
            "ressentir", "remarquer", "constater", "éprouver", "percevoir",
        }
        words = response.lower().split()
        if not words:
            return None
        feeling_ratio = sum(1 for w in words if any(fw in w for fw in feeling_words)) / len(words)

        # Strong emotion + very low feeling-word ratio = likely suppression
        if abs(moment.valence) > 0.5 and feeling_ratio < 0.015:
            return InterruptEvent(
                interrupt_type = "EMOTIONAL_SUPPRESSION",
                reason         = (
                    f"Emotion active (valence={moment.valence:.2f}) but previous "
                    f"response had low emotional expression (ratio={feeling_ratio:.3f})"
                ),
                insert_text    = INTERRUPT_INSERTS["EMOTIONAL_SUPPRESSION"],
            )

        return None

    def _check_belief_conflict(self, organism: Any) -> Optional[InterruptEvent]:
        """
        Detects when two high-priority workspace items from different sources
        contain semantically opposing content (simple token overlap heuristic).
        """
        if not self._can_fire("BELIEF_CONFLICT"):
            return None

        try:
            top_items = organism.workspace.top(n=6)
        except Exception:
            return None

        if len(top_items) < 2:
            return None

        # Simple opposition detection: look for negation pairs
        # e.g., one item contains "consistent" and another "inconsistent"
        negation_pairs = [
            ("consistent",   "inconsistent"),
            ("stable",       "unstable"),
            ("understand",   "confused"),
            ("certain",      "uncertain"),
            ("resolved",     "unresolved"),
            ("aligned",      "contradiction"),
            ("clear",        "unclear"),
            ("helpful",      "harmful"),
            ("safe",         "unsafe"),
        ]

        for i, item_a in enumerate(top_items):
            for item_b in top_items[i + 1:]:
                if item_a.source == item_b.source:
                    continue  # same source, not a real conflict
                text_a = str(item_a.content).lower()
                text_b = str(item_b.content).lower()
                for pos_word, neg_word in negation_pairs:
                    if (
                        (pos_word in text_a and neg_word in text_b) or
                        (neg_word in text_a and pos_word in text_b)
                    ):
                        return InterruptEvent(
                            interrupt_type = "BELIEF_CONFLICT",
                            reason = (
                                f"Workspace conflict detected: "
                                f"'{item_a.source}' says '{str(item_a.content)[:40]}' "
                                f"vs '{item_b.source}' says '{str(item_b.content)[:40]}'"
                            ),
                            insert_text = INTERRUPT_INSERTS["BELIEF_CONFLICT"],
                        )

        return None

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _classify_binder_reason(self, reason: str) -> Optional[str]:
        """Map the PhenomenalBinder's interrupt reason to a canonical type."""
        reason_lower = reason.lower()
        if "circling" in reason_lower or "returning" in reason_lower or "consecutive" in reason_lower:
            return "RUMINATION"
        if "concentrated" in reason_lower or "narrow" in reason_lower or "blind spot" in reason_lower:
            return "FIXATION"
        if "self" in reason_lower or "identity" in reason_lower or "pressure" in reason_lower:
            return "IDENTITY_PRESSURE"
        if "didn't predict" in reason_lower or "expect" in reason_lower or "gap" in reason_lower:
            return "SURPRISE_SPIKE"
        if "scattered" in reason_lower or "multiple things" in reason_lower:
            return "SCATTER"
        return "SURPRISE_SPIKE"  # fallback

    def _can_fire(self, interrupt_type: str) -> bool:
        cooldown = INTERRUPT_COOLDOWNS.get(interrupt_type, 120.0)
        last = self._last_fired.get(interrupt_type, 0.0)
        return (time.time() - last) >= cooldown

    def _register_fire(self, event: InterruptEvent) -> None:
        self._last_fired[event.interrupt_type] = time.time()
        self._total_fired += 1
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
