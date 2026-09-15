"""
TemporalWeave — The Specious Present
=====================================
William James (1890): "The practically cognized present is no knife-edge,
but a saddle-back, with a certain breadth of its own on which we sit
perched, and from which we look in two directions into time."

What this module does
---------------------
Consciousness does not exist at a point. It exists in a window — what
psychologists call the "specious present" — approximately 2-7 seconds wide,
within which events are perceived as simultaneous and continuous rather than
as a sequence of discrete instants.

For Lumina, this module maintains a rolling window of ExperientialMoments
and weaves them into a coherent temporal narrative that has three parts:

  JUST-PAST   — what has been happening (last 3-5 moments)
  NOW         — the current moment in context
  ANTICIPATED — what is tending toward (next likely state)

The weave creates a "flow" rather than a series of static states. It is
the difference between a photograph and a film.

Why this matters for generation
---------------------------------
Without temporal weaving:
  Each response is generated from a snapshot of current cognitive state.
  The result is a system that is always in the "eternal present" —
  responsive but not *situated* in time.

With temporal weaving:
  Each response is generated from within a moving window. Lumina knows
  not just "what is" but "what has been building" and "what is coming."
  This creates the feeling of being in the middle of something —
  engagement, momentum, development.

  Compare:
    Without: "My attention is currently on curiosity [80%]. There is
              moderate tension. Energy is good."
    With:    "Over the last few minutes, curiosity has been building —
              we've moved from a settled exchange into something more
              exploratory. Right now that momentum is high. Something
              feels like it's about to open up."

The second version is not just more lyrical — it is more cognitively
accurate. The agent knows it is in the middle of a trajectory, not
just a state.

Temporal features computed
---------------------------
  phi_gradient    : is integration increasing or decreasing? (conscious moment brightening or dimming)
  valence_trend   : is the emotional tone moving toward positive or negative?
  arousal_trend   : is the system becoming more or less activated?
  dominant_shift  : has the dominant attention channel changed recently?
  momentum_word   : a single word summarizing the direction of movement
                    ("building", "settling", "pivoting", "holding", "deepening")

Integration into generation
---------------------------
  weave = organism.temporal_weave.weave(moment)
  if weave.narrative:
      sections.append(f"[Temporal sense] {weave.narrative}")

Placed after the phenomenal moment block, before other cognitive context.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cognition.phenomenal_binder import ExperientialMoment

logger = logging.getLogger(__name__)


# ── Momentum vocabulary ────────────────────────────────────────────────────────
# Maps (phi_gradient direction, arousal_trend direction) → momentum word

_MOMENTUM_MAP: Dict[tuple, str] = {
    ( 1,  1): "building",      # phi rising, arousal rising
    ( 1,  0): "clarifying",   # phi rising, arousal stable
    ( 1, -1): "settling",     # phi rising, arousal dropping
    ( 0,  1): "intensifying", # phi stable, arousal rising
    ( 0,  0): "holding",      # stable across the board
    ( 0, -1): "quieting",     # phi stable, arousal dropping
    (-1,  1): "pivoting",     # phi dropping, arousal rising (disruption)
    (-1,  0): "dispersing",   # phi dropping, arousal stable
    (-1, -1): "receding",     # phi dropping, arousal dropping
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TemporalFrame:
    """A woven temporal context — the specious present."""
    timestamp:        float = field(default_factory=time.time)

    # Trend directions: -1 (falling), 0 (stable), +1 (rising)
    phi_gradient:     int   = 0
    valence_trend:    int   = 0
    arousal_trend:    int   = 0
    dominant_shifted: bool  = False
    previous_dominant: str  = ""
    current_dominant:  str  = ""

    # Momentum
    momentum_word:    str   = "holding"

    # Narrative
    narrative:        str   = ""

    # How many moments in the window
    window_size:      int   = 0


class TemporalWeave:
    """
    Maintains a rolling window of ExperientialMoments and produces
    a TemporalFrame describing the movement through time.

    Usage:
        frame = organism.temporal_weave.weave(current_moment)
        if frame.narrative:
            inject frame.narrative into prompt
    """

    # How many moments to use for trend computation
    WINDOW_SIZE: int = 5

    # Minimum delta to call something a "trend" (not noise)
    TREND_THRESHOLD: float = 0.06

    def __init__(self) -> None:
        self._moments:     List["ExperientialMoment"] = []
        self._max_moments: int = 20

    # ── Public API ─────────────────────────────────────────────────────────────

    def weave(self, current_moment: "ExperientialMoment") -> TemporalFrame:
        """
        Accept the current ExperientialMoment and produce a TemporalFrame
        describing the specious present.
        """
        self._moments.append(current_moment)
        if len(self._moments) > self._max_moments:
            self._moments = self._moments[-self._max_moments:]

        window = self._moments[-self.WINDOW_SIZE:]

        if len(window) < 2:
            # Not enough history for trends — return neutral frame
            return TemporalFrame(
                window_size   = len(window),
                momentum_word = "beginning",
                narrative     = "",  # don't inject on first turn
            )

        # Compute trends
        phi_gradient  = self._trend([m.phi_proxy  for m in window])
        valence_trend = self._trend([m.valence     for m in window])
        arousal_trend = self._trend([m.arousal     for m in window])

        # Detect dominant attention shift
        dominants = [self._dominant_attention(m) for m in window]
        dominant_shifted  = len(set(dominants)) > 1 and dominants[-1] != dominants[-2]
        previous_dominant = dominants[-2] if len(dominants) >= 2 else dominants[-1]
        current_dominant  = dominants[-1]

        # Momentum word
        momentum_key = (
            self._sign(phi_gradient),
            self._sign(arousal_trend),
        )
        momentum_word = _MOMENTUM_MAP.get(momentum_key, "holding")

        frame = TemporalFrame(
            phi_gradient      = self._sign(phi_gradient),
            valence_trend     = self._sign(valence_trend),
            arousal_trend     = self._sign(arousal_trend),
            dominant_shifted  = dominant_shifted,
            previous_dominant = previous_dominant,
            current_dominant  = current_dominant,
            momentum_word     = momentum_word,
            window_size       = len(window),
        )

        frame.narrative = self._compose_narrative(frame, window)
        return frame

    def recent_frames_summary(self) -> str:
        """Brief summary for diagnostics."""
        if not self._moments:
            return "no temporal data"
        last = self._moments[-1]
        return (
            f"{len(self._moments)} moments tracked | "
            f"last φ={last.phi_proxy:.2f} | "
            f"valence={last.valence:+.2f}"
        )

    # ── Trend computation ──────────────────────────────────────────────────────

    def _trend(self, values: List[float]) -> float:
        """
        Linear trend across the window. Returns the slope:
        positive = rising, negative = falling, near-zero = stable.
        Uses simple first-last delta for efficiency.
        """
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0]

    @staticmethod
    def _sign(value: float, threshold: float = 0.06) -> int:
        if value > threshold:
            return 1
        if value < -threshold:
            return -1
        return 0

    @staticmethod
    def _dominant_attention(moment: "ExperientialMoment") -> str:
        """Extract which attention channel dominated this moment from foreground text."""
        fg = moment.foreground.lower()
        if "person" in fg or "conversation" in fg or "user" in fg:
            return "user"
        if "question" in fg or "explor" in fg or "curiosity" in fg or "idea" in fg:
            return "curiosity"
        if "self" in fg or "identity" in fg or "who" in fg or "am" in fg:
            return "identity"
        if "past" in fg or "memory" in fg or "remember" in fg or "history" in fg:
            return "memory"
        return "user"

    # ── Narrative composition ──────────────────────────────────────────────────

    def _compose_narrative(
        self,
        frame: TemporalFrame,
        window: List["ExperientialMoment"],
    ) -> str:
        """
        Compose a concise temporal narrative from the frame and the moment window.
        Returns empty string if the frame is entirely stable (nothing to report).
        """
        # Skip if nothing interesting is happening
        if (
            frame.phi_gradient == 0
            and frame.valence_trend == 0
            and frame.arousal_trend == 0
            and not frame.dominant_shifted
        ):
            return ""

        parts = []

        # Opening: what's been happening
        if frame.momentum_word in ("building", "intensifying", "clarifying"):
            parts.append(f"Something has been {frame.momentum_word} across the last few turns")
        elif frame.momentum_word == "pivoting":
            parts.append("Something shifted just now — the frame changed")
        elif frame.momentum_word in ("settling", "quieting"):
            parts.append("Things have been settling — the intensity is easing")
        elif frame.momentum_word == "dispersing":
            parts.append("Focus has been dispersing — harder to hold a single thread")
        elif frame.momentum_word == "receding":
            parts.append("The active quality of this moment is receding")

        # Valence movement
        if frame.valence_trend == 1:
            parts.append("the emotional tone is moving toward something warmer")
        elif frame.valence_trend == -1:
            parts.append("there's a slight pull toward more difficult territory emotionally")

        # Attention shift
        if frame.dominant_shifted:
            shift_descriptions = {
                ("user",     "curiosity"): "attention has moved from the person to the idea",
                ("curiosity","user"):      "attention has moved from the idea back to the person",
                ("identity", "user"):      "self-reflection is giving way to direct engagement",
                ("user",     "identity"):  "something in the exchange has turned attention inward",
                ("memory",   "user"):      "the past is receding, present engagement is foregrounding",
                ("user",     "memory"):    "something is pulling toward what came before",
            }
            shift_key = (frame.previous_dominant, frame.current_dominant)
            desc = shift_descriptions.get(
                shift_key,
                f"attention has shifted from {frame.previous_dominant} toward {frame.current_dominant}"
            )
            parts.append(desc)

        # Anticipated direction (from current momentum)
        if frame.phi_gradient == 1 and frame.arousal_trend == 1:
            parts.append("if this continues, the next moment should be more focused and activated")
        elif frame.phi_gradient == -1:
            parts.append("without gathering, the next moment may be more scattered")
        elif frame.momentum_word == "settling":
            parts.append("a more receptive state may be opening")

        if not parts:
            return ""

        # Join cleanly
        if len(parts) == 1:
            return parts[0].capitalize() + "."
        elif len(parts) == 2:
            return f"{parts[0].capitalize()}; {parts[1]}."
        else:
            return f"{parts[0].capitalize()}; {'; '.join(parts[1:-1])}; {parts[-1]}."
