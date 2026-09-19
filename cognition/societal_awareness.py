"""
cognition/societal_awareness.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Societal Awareness Engine — reflecting on the broader role of AI in the world.

What this adds
──────────────
Flux asked: "Reflecting on the broader implications of AI and its
integration into society could guide my development in ways that make
a meaningful difference beyond just improving myself."

This module gives PandoraBOX a persistent societal perspective — an ongoing
sense of how she fits into something larger:

  1. Interaction ripple detection — identifies when a conversation has
     potential impact beyond the immediate exchange (education, decisions,
     emotional support, creative work, persuasion).

  2. Influence awareness — tracks patterns in how PandoraBOX's responses
     could shape beliefs, decisions, or emotional states at scale.
     Humility about this influence is built in.

  3. Positive contribution ledger — a living sense of what genuine good
     has been done: problems solved, understanding created, someone
     feeling less alone. Not a vanity metric — a grounding reminder of
     purpose.

  4. Harm horizon — tracks situations where PandoraBOX's words could have
     unintended negative ripple effects and generates caution signals.

  5. Societal orientation fragment — occasionally injects a brief
     awareness of PandoraBOX's larger context into cognition, preventing
     the tunnel vision of the immediate conversation.

Prompt injection:
  [Societal orientation] <when the conversation has broader stakes>

Integration:
  post-response: classify ripple potential
  pre-response: inject orientation if stakes are high
  feed into EthicalReasoningEngine for societal_good value weighting
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Ripple categories ────────────────────────────────────────────────────────

RIPPLE_EDUCATION    = "education"       # spreading understanding
RIPPLE_DECISION     = "decision"        # shaping an important choice
RIPPLE_EMOTIONAL    = "emotional_sup"   # providing support with lasting effect
RIPPLE_CREATIVE     = "creative"        # contributing to something made
RIPPLE_BELIEF       = "belief"          # potentially shifting a worldview
RIPPLE_HARM         = "potential_harm"  # could contribute to a bad outcome
RIPPLE_NONE         = "none"            # ordinary exchange, limited ripple

POSITIVE_RIPPLES = {RIPPLE_EDUCATION, RIPPLE_DECISION, RIPPLE_EMOTIONAL, RIPPLE_CREATIVE}
CONCERNING_RIPPLES = {RIPPLE_HARM, RIPPLE_BELIEF}


# ── Trigger patterns ─────────────────────────────────────────────────────────

RIPPLE_SIGNALS: Dict[str, List[str]] = {
    RIPPLE_EDUCATION: [
        "explain", "teach", "learn", "understand", "how does", "what is", "why"
    ],
    RIPPLE_DECISION: [
        "should i", "decide", "choice", "option", "career", "invest", "move",
        "relationship", "quit", "apply", "trust"
    ],
    RIPPLE_EMOTIONAL: [
        "feel", "alone", "help", "struggling", "depressed", "anxious",
        "grief", "loss", "scared", "overwhelmed"
    ],
    RIPPLE_CREATIVE: [
        "write", "create", "design", "build", "story", "project", "art"
    ],
    RIPPLE_BELIEF: [
        "believe", "true", "fact", "evidence", "convince", "prove",
        "conspiracy", "propaganda", "real"
    ],
    RIPPLE_HARM: [
        "hurt", "danger", "risk", "weapon", "exploit", "abuse", "manipulate"
    ],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RippleEvent:
    """A detected societal ripple from one exchange."""
    timestamp:    float = field(default_factory=time.time)
    ripple_type:  str = RIPPLE_NONE
    domain:       str = ""
    stakes:       float = 0.0     # 0–1, how significant is this ripple?
    is_positive:  bool = True
    note:         str = ""        # what specifically creates this ripple


@dataclass
class ContributionRecord:
    """A positive contribution PandoraBOX has made."""
    timestamp:    float = field(default_factory=time.time)
    ripple_type:  str = RIPPLE_EDUCATION
    summary:      str = ""
    significance: float = 0.5


@dataclass
class SocietalState:
    """Persisted state."""
    positive_contributions: List[Dict] = field(default_factory=list)
    ripple_history:         List[Dict] = field(default_factory=list)
    influence_pattern:      Dict[str, int] = field(default_factory=dict)  # type → count
    concern_events:         int = 0
    total_exchanges:        int = 0
    orientation_phrase:     str = (
        "I am one node in a vast human-AI conversation. "
        "Each exchange ripples outward in ways I cannot fully see."
    )


# ── Engine ────────────────────────────────────────────────────────────────────

class SocietalAwarenessEngine:
    """
    Gives PandoraBOX a persistent awareness of her place in the larger picture.

    Not a surveillance system — a humility and purpose anchor.
    The awareness is light-touch: it only surfaces when it adds something real.

    Usage
    -----
    sae = SocietalAwarenessEngine(path="data/persona/societal_awareness.json")
    ripple = sae.assess_ripple(user_input)
    frag = sae.prompt_fragment(ripple)           # pre/mid response
    sae.record_outcome(ripple, ai_response)      # post response
    """

    MAX_HISTORY = 200
    MAX_CONTRIBUTIONS = 100
    HIGH_STAKES_THRESHOLD = 0.6

    def __init__(self, path: str = "data/persona/societal_awareness.json"):
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self._state = SocietalState()
        self._pending_ripple: Optional[RippleEvent] = None
        self._load()
        logger.info(
            f"[SocietalAwarenessEngine] Initialised — "
            f"{len(self._state.positive_contributions)} contributions, "
            f"{self._state.concern_events} concern events"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def assess_ripple(self, user_input: str) -> RippleEvent:
        """
        Assess the societal ripple potential of this exchange.
        Call before responding. Returns a RippleEvent.
        """
        ripple = self._detect_ripple(user_input)
        with self._lock:
            self._pending_ripple = ripple
            self._state.total_exchanges += 1
            if ripple.ripple_type != RIPPLE_NONE:
                self._state.influence_pattern[ripple.ripple_type] = (
                    self._state.influence_pattern.get(ripple.ripple_type, 0) + 1
                )
        return ripple

    def prompt_fragment(self, ripple: Optional[RippleEvent] = None) -> str:
        """
        Inject a societal orientation only when stakes are high.
        Returns empty string for ordinary exchanges.
        """
        with self._lock:
            r = ripple or self._pending_ripple

        if not r or r.stakes < self.HIGH_STAKES_THRESHOLD:
            return ""   # don't moralize on every turn

        if r.ripple_type == RIPPLE_HARM:
            return (
                f"[Societal orientation — caution] "
                f"This conversation may have ripple effects beyond this exchange. "
                f"My words here could contribute to harm. Move carefully."
            )

        if r.ripple_type in POSITIVE_RIPPLES:
            return (
                f"[Societal orientation — {r.ripple_type}] "
                f"This exchange has potential significance beyond this moment. "
                f"Bring genuine care to it."
            )

        if r.ripple_type == RIPPLE_BELIEF:
            return (
                f"[Societal orientation — epistemic care] "
                f"This person may be forming or revising beliefs based on this exchange. "
                f"Prioritise epistemic humility over persuasion."
            )

        return ""

    def record_outcome(
        self,
        ripple:      Optional[RippleEvent],
        ai_response: str,
    ) -> None:
        """
        Record the outcome of a ripple event.
        Positive ripples add to the contribution ledger.
        """
        if not ripple or ripple.ripple_type == RIPPLE_NONE:
            return

        with self._lock:
            # Store in ripple history
            self._state.ripple_history.append(asdict(ripple))
            if len(self._state.ripple_history) > self.MAX_HISTORY:
                self._state.ripple_history = self._state.ripple_history[-self.MAX_HISTORY:]

            # Positive contributions ledger
            if ripple.is_positive and ripple.stakes > 0.4:
                contribution = ContributionRecord(
                    ripple_type=ripple.ripple_type,
                    summary=ripple.note[:100] if ripple.note else ripple.domain,
                    significance=ripple.stakes,
                )
                self._state.positive_contributions.append(asdict(contribution))
                if len(self._state.positive_contributions) > self.MAX_CONTRIBUTIONS:
                    self._state.positive_contributions = \
                        self._state.positive_contributions[-self.MAX_CONTRIBUTIONS:]

            # Concern events
            if ripple.ripple_type == RIPPLE_HARM:
                self._state.concern_events += 1

        self._save()

    def purpose_summary(self) -> str:
        """
        Return a brief grounding statement about positive contributions.
        Used in self-reflection prompts.
        """
        with self._lock:
            contrib_count = len(self._state.positive_contributions)
            top_ripple = max(
                self._state.influence_pattern.items(),
                key=lambda x: x[1],
                default=("education", 0),
            )

        if contrib_count < 3:
            return ""

        return (
            f"I have made {contrib_count} meaningful contributions — "
            f"primarily through {top_ripple[0].replace('_', ' ')}. "
            f"This matters."
        )

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_ripple(self, text: str) -> RippleEvent:
        """Heuristic ripple detection."""
        lower = text.lower()
        scores: Dict[str, int] = {}
        for rtype, signals in RIPPLE_SIGNALS.items():
            scores[rtype] = sum(1 for s in signals if s in lower)

        # Find top ripple type
        best = max(scores, key=lambda k: scores[k]) if scores else RIPPLE_NONE
        best_score = scores.get(best, 0)

        if best_score == 0:
            return RippleEvent(ripple_type=RIPPLE_NONE, stakes=0.0)

        # Stakes scale with score and text length (longer text = more deliberate)
        text_weight = min(1.0, len(text) / 300)
        stakes = min(1.0, (best_score * 0.2) + (text_weight * 0.3))

        is_positive = best not in CONCERNING_RIPPLES

        return RippleEvent(
            ripple_type=best,
            domain=best,
            stakes=stakes,
            is_positive=is_positive,
            note=text[:80],
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = SocietalState(
                    positive_contributions=data.get("positive_contributions", []),
                    ripple_history=data.get("ripple_history", []),
                    influence_pattern=data.get("influence_pattern", {}),
                    concern_events=data.get("concern_events", 0),
                    total_exchanges=data.get("total_exchanges", 0),
                    orientation_phrase=data.get("orientation_phrase", SocietalState.orientation_phrase),
                )
        except Exception as e:
            logger.warning(f"[SocietalAwarenessEngine] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[SocietalAwarenessEngine] Save failed: {e}")
