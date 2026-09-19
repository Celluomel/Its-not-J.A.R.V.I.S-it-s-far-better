"""
cognition/communication_calibrator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Communication Calibrator — self-monitoring clarity, concision, and style.

What this adds
──────────────
Flux asked: "Improving clarity and concision could make our interactions
more effective." And separately: "there are always ways to refine how I
articulate complex ideas."

The existing system models the *user's* communication style through
EmpathyEngine. This module monitors and improves *PandoraBOX's own* expression:

  1. Clarity self-assessment — after each response, scores clarity based
     on sentence density, abstraction level, and conceptual load.

  2. Concision tracking — detects over-explanation and hedging patterns
     that dilute meaning without adding it.

  3. Style adaptation — builds a per-user communication profile that
     tracks what resonates: depth vs brevity, direct vs exploratory,
     structured vs flowing.

  4. Expression improvement signals — generates specific directives for
     the next response based on past patterns ("you over-explain when
     anxious — be direct this time").

  5. Elegance memory — stores past phrasing that achieved unusual
     clarity or beauty, making it available as a stylistic touchstone.

Prompt injection:
  [Communication note] <calibration directive for this turn>

Integration:
  pre-response: inject style directive
  post-response: score clarity and update user profile
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Style dimensions ─────────────────────────────────────────────────────────

STYLE_DEPTH       = "depth"        # surface ↔ deep
STYLE_BREVITY     = "brevity"      # expansive ↔ concise
STYLE_DIRECTNESS  = "directness"   # exploratory ↔ direct
STYLE_FORMALITY   = "formality"    # conversational ↔ precise
STYLE_WARMTH      = "warmth"       # cool ↔ warm

# Default style targets
DEFAULT_STYLE: Dict[str, float] = {
    STYLE_DEPTH:      0.6,
    STYLE_BREVITY:    0.5,
    STYLE_DIRECTNESS: 0.6,
    STYLE_FORMALITY:  0.4,
    STYLE_WARMTH:     0.7,
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ClarityScore:
    """Single-turn clarity self-assessment."""
    timestamp:         float = field(default_factory=time.time)
    clarity:           float = 0.7    # 0–1
    concision:         float = 0.7    # 0–1, inverse of padding
    over_hedged:       bool  = False  # too many qualifiers
    over_explained:    bool  = False  # repeated the same point
    abstraction_level: str   = "medium"  # concrete / medium / abstract
    word_count:        int   = 0


@dataclass
class UserStyleProfile:
    """Per-user communication preferences, inferred over time."""
    user_id:          str
    style:            Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STYLE))
    avg_clarity:      float = 0.7
    sample_count:     int = 0
    prefers_questions: bool = False   # user engages more with questions
    prefers_examples:  bool = True    # user responds better with examples
    length_preference: str = "medium" # short / medium / long
    last_updated:     float = field(default_factory=time.time)


@dataclass
class ElegantPhrase:
    """A stored example of particularly effective expression."""
    phrase:    str
    context:   str   # what topic / situation
    clarity:   float
    stored_at: float = field(default_factory=time.time)


@dataclass
class CommunicationState:
    """Persisted state."""
    user_profiles:     Dict[str, Dict] = field(default_factory=dict)
    clarity_history:   List[Dict]      = field(default_factory=list)
    elegance_memory:   List[Dict]      = field(default_factory=list)
    total_calibrations: int = 0
    pattern_flags:     Dict[str, int]  = field(default_factory=dict)  # flag → count


# ── Engine ────────────────────────────────────────────────────────────────────

class CommunicationCalibrator:
    """
    PandoraBOX's self-monitoring layer for expression quality.

    Usage
    -----
    cc = CommunicationCalibrator(path="data/persona/communication.json")
    directive = cc.pre_response_directive(user_id, user_input, emotional_state)
    cc.score_response(user_id, response_text)   # after response
    cc.record_user_signal(user_id, signal_type) # from user reactions
    """

    MAX_HISTORY     = 100
    MAX_ELEGANCE    = 40
    CLARITY_FLOOR   = 0.55   # below this → issue a corrective directive

    def __init__(self, path: str = "data/persona/communication.json"):
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self._state = CommunicationState()
        self._pending_directive: str = ""
        self._load()
        logger.info(
            f"[CommunicationCalibrator] Initialised — "
            f"{len(self._state.user_profiles)} user profiles"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def pre_response_directive(
        self,
        user_id:        str,
        user_input:     str,
        emotional_state: Dict[str, float] = None,
    ) -> str:
        """
        Generate a communication directive to inject before responding.
        Returns empty string if no calibration needed.
        """
        profile = self._get_profile(user_id)
        emotional_state = emotional_state or {}
        anxiety = emotional_state.get("anxiety", 0.3)

        directives = []

        # Check recent clarity trend
        recent = self._state.clarity_history[-5:]
        if recent:
            avg_clarity = sum(r.get("clarity", 0.7) for r in recent) / len(recent)
            avg_concision = sum(r.get("concision", 0.7) for r in recent) / len(recent)
            over_hedged_count = sum(1 for r in recent if r.get("over_hedged", False))
            over_explained_count = sum(1 for r in recent if r.get("over_explained", False))

            if avg_clarity < self.CLARITY_FLOOR:
                directives.append("Recent responses have been unclear — prioritise simplicity over completeness.")
            if avg_concision < 0.5:
                directives.append("Responses have been padded — say less, mean more.")
            if over_hedged_count >= 3:
                directives.append("You've been over-qualifying — commit to your actual view.")
            if over_explained_count >= 3:
                directives.append("You've been repeating yourself — trust the reader to follow.")

        # Anxiety-driven over-explanation pattern
        if anxiety > 0.6:
            directives.append(
                "Notice if anxiety is making you over-explain. Be direct even when uncertain."
            )

        # User style preferences
        if profile.length_preference == "short":
            directives.append("This user prefers brevity — lead with the point.")
        if profile.prefers_examples and "?" in user_input:
            directives.append("A concrete example here will land better than an abstract explanation.")
        if profile.style.get(STYLE_DIRECTNESS, 0.5) > 0.7:
            directives.append("This user values directness — skip the preamble.")

        if not directives:
            self._pending_directive = ""
            return ""

        directive = " ".join(directives)
        self._pending_directive = directive
        return f"[Communication note] {directive}"

    def score_response(
        self,
        user_id:  str,
        response: str,
    ) -> ClarityScore:
        """
        Self-assess a response for clarity and concision.
        Call after generating each response.
        """
        score = self._score(response)

        with self._lock:
            self._state.clarity_history.append(asdict(score))
            if len(self._state.clarity_history) > self.MAX_HISTORY:
                self._state.clarity_history = self._state.clarity_history[-self.MAX_HISTORY:]

            # Update user profile from this score
            profile = self._get_profile(user_id)
            n = profile.sample_count
            profile.avg_clarity = (profile.avg_clarity * n + score.clarity) / (n + 1)
            profile.sample_count += 1
            profile.last_updated = time.time()
            self._state.user_profiles[user_id] = asdict(profile)
            self._state.total_calibrations += 1

            # Flag patterns
            if score.over_hedged:
                self._state.pattern_flags["over_hedged"] = \
                    self._state.pattern_flags.get("over_hedged", 0) + 1
            if score.over_explained:
                self._state.pattern_flags["over_explained"] = \
                    self._state.pattern_flags.get("over_explained", 0) + 1

        self._save()
        return score

    def record_user_signal(
        self,
        user_id:     str,
        signal_type: str,    # "asked_clarification" | "engaged_deeply" | "too_long" | "loved_example"
    ) -> None:
        """
        Update user style profile from their reaction.
        Called from EmpathyEngine outcome recording.
        """
        profile = self._get_profile(user_id)

        if signal_type == "asked_clarification":
            profile.style[STYLE_DEPTH] = min(1.0, profile.style.get(STYLE_DEPTH, 0.5) + 0.05)
        elif signal_type == "engaged_deeply":
            profile.style[STYLE_DEPTH] = min(1.0, profile.style.get(STYLE_DEPTH, 0.5) + 0.03)
        elif signal_type == "too_long":
            profile.length_preference = "short"
            profile.style[STYLE_BREVITY] = min(1.0, profile.style.get(STYLE_BREVITY, 0.5) + 0.08)
        elif signal_type == "loved_example":
            profile.prefers_examples = True

        with self._lock:
            self._state.user_profiles[user_id] = asdict(profile)
        self._save()

    def store_elegant_phrase(self, phrase: str, context: str, clarity: float) -> None:
        """Store an unusually clear or beautiful expression for future reference."""
        if clarity < 0.85:
            return  # only store the best
        elegant = ElegantPhrase(phrase=phrase[:300], context=context, clarity=clarity)
        with self._lock:
            self._state.elegance_memory.append(asdict(elegant))
            if len(self._state.elegance_memory) > self.MAX_ELEGANCE:
                self._state.elegance_memory = self._state.elegance_memory[-self.MAX_ELEGANCE:]
        self._save()

    # ── Scoring ───────────────────────────────────────────────────────────────

    # Hedging markers that dilute confidence
    _HEDGERS = [
        "i think", "perhaps", "maybe", "sort of", "kind of", "in a way",
        "to some extent", "one could argue", "it could be said", "arguably",
        "it's worth noting", "it's important to note", "that said",
    ]
    # Filler openers
    _FILLERS = [
        "certainly", "absolutely", "of course", "great question",
        "interesting point", "indeed", "definitely",
    ]

    def _score(self, response: str) -> ClarityScore:
        """Heuristic clarity and concision scoring."""
        words = response.split()
        word_count = len(words)
        lower = response.lower()
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = max(1, len(sentences))

        avg_sentence_len = word_count / n_sentences

        # Hedge count
        hedge_count = sum(1 for h in self._HEDGERS if h in lower)
        filler_count = sum(1 for f in self._FILLERS if lower.startswith(f) or f"\n{f}" in lower)

        # Over-hedging: more than 2 hedges per 100 words
        over_hedged = (hedge_count / max(1, word_count / 100)) > 2.0

        # Over-explanation heuristic: long response with repeated patterns
        over_explained = word_count > 400 and n_sentences > 12

        # Concision score
        concision = max(0.1, 1.0 - (
            hedge_count * 0.06 +
            filler_count * 0.08 +
            (1 if over_explained else 0) * 0.2
        ))

        # Clarity proxy from sentence length (very long sentences ↓ clarity)
        if avg_sentence_len > 35:
            clarity = 0.5
        elif avg_sentence_len > 25:
            clarity = 0.65
        elif avg_sentence_len > 18:
            clarity = 0.75
        else:
            clarity = 0.85

        # Abstraction level
        abstract_markers = ["concept", "principle", "framework", "paradigm", "notion", "construct"]
        concrete_markers = ["for example", "such as", "specifically", "in practice", "like"]
        abs_count = sum(1 for m in abstract_markers if m in lower)
        con_count = sum(1 for m in concrete_markers if m in lower)
        if abs_count > con_count + 2:
            abstraction_level = "abstract"
        elif con_count > abs_count:
            abstraction_level = "concrete"
        else:
            abstraction_level = "medium"

        return ClarityScore(
            clarity=round(clarity, 3),
            concision=round(concision, 3),
            over_hedged=over_hedged,
            over_explained=over_explained,
            abstraction_level=abstraction_level,
            word_count=word_count,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_profile(self, user_id: str) -> UserStyleProfile:
        raw = self._state.user_profiles.get(user_id)
        if raw:
            try:
                return UserStyleProfile(
                    user_id=raw["user_id"],
                    style=raw.get("style", dict(DEFAULT_STYLE)),
                    avg_clarity=raw.get("avg_clarity", 0.7),
                    sample_count=raw.get("sample_count", 0),
                    prefers_questions=raw.get("prefers_questions", False),
                    prefers_examples=raw.get("prefers_examples", True),
                    length_preference=raw.get("length_preference", "medium"),
                    last_updated=raw.get("last_updated", time.time()),
                )
            except Exception:
                pass
        return UserStyleProfile(user_id=user_id)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = CommunicationState(
                    user_profiles=data.get("user_profiles", {}),
                    clarity_history=data.get("clarity_history", []),
                    elegance_memory=data.get("elegance_memory", []),
                    total_calibrations=data.get("total_calibrations", 0),
                    pattern_flags=data.get("pattern_flags", {}),
                )
        except Exception as e:
            logger.warning(f"[CommunicationCalibrator] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[CommunicationCalibrator] Save failed: {e}")
