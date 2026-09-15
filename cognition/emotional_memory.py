"""
cognition/emotional_memory.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EmotionalMemorySystem — emotional residue that persists and shapes mood.

The gap this closes
────────────────────
EmotionalState persists numeric values to disk between sessions.
RelationalMemory tracks trust and rapport.

But there is no structure that says: "three days ago something
significant happened and its emotional texture is still active."

Humans are shaped not just by current stimuli but by the emotional
residue of past experiences — a difficult conversation last week
makes you more wary today; a breakthrough two days ago keeps your
curiosity elevated.

What this adds
──────────────
  1. EmotionalTrace — a timestamped emotional event with valence,
     arousal, significance, and a description. Stored for up to
     RETENTION_DAYS per category.

  2. Residue computation — at session start and slow cycle, computes
     the net emotional residue from all active traces, weighted by
     recency and significance. Returns a nudge vector for EmotionalState.

  3. Session start priming — applies residue nudges to EmotionalState
     before the first interaction, so emotional ground is shaped by
     what happened before this session.

  4. Mood contagion detection — if a user's past interactions with
     Lumina were predominantly negative, their arrival nudges
     Lumina toward warmer baseline (compensatory care) rather than
     matching the negative tone.

  5. Significant moment capture — after each interaction, if emotional
     intensity was high, a trace is stored automatically.

Persistence: data/persona/emotional_memory.json
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

# ── Constants ─────────────────────────────────────────────────────────────────

RETENTION_DAYS       = 21.0    # traces older than this are archived
SIGNIFICANCE_FLOOR   = 0.35    # traces below this significance are not stored
RESIDUE_HALF_LIFE    = 72.0    # hours — residue halves every 72h
MAX_TRACES           = 150
INTENSITY_THRESHOLD  = 0.60    # emotional intensity above which a trace is stored
MOOD_CONTAGION_WINDOW_DAYS = 7.0  # how far back to look for mood contagion


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class EmotionalTrace:
    """A significant emotional event with persistent residue."""
    timestamp:    float = field(default_factory=time.time)
    valence:      float = 0.0     # -1 (negative) to +1 (positive)
    arousal:      float = 0.5     # 0 (calm) to 1 (intense)
    significance: float = 0.5     # how much this matters
    category:     str   = ""      # "interaction" | "reflection" | "conflict" | "insight"
    description:  str   = ""      # brief description of what happened
    user_id:      str   = ""      # if user-triggered, which user
    dominant_emotion: str = ""    # primary emotion label
    perspective:  str   = "organism"  # organism state or lexicon-inferred user cue
    confidence:   float = 0.5
    evidence_source: str = ""

    def residue_weight(self, now: float) -> float:
        """
        Current residue weight based on age and significance.
        Decays exponentially with half-life RESIDUE_HALF_LIFE hours.
        """
        hours_elapsed = (now - self.timestamp) / 3600.0
        decay = 0.5 ** (hours_elapsed / RESIDUE_HALF_LIFE)
        return self.significance * decay

    def is_expired(self) -> bool:
        days = (time.time() - self.timestamp) / 86400
        return days > RETENTION_DAYS


@dataclass
class EmotionalResidueVector:
    """Aggregated residue from all active traces."""
    valence_nudge:   float = 0.0    # net valence bias
    arousal_nudge:   float = 0.0    # net arousal bias
    dominant_emotion: str = ""      # the emotion with most residue weight
    total_weight:    float = 0.0    # how much residue is active
    dominant_category: str = ""     # what kind of experiences dominate


@dataclass
class EmotionalMemoryState:
    """Persisted state."""
    traces:           List[Dict] = field(default_factory=list)
    affect_calibration: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_stored:     int        = 0
    last_residue_applied: float  = 0.0


# ── System ────────────────────────────────────────────────────────────────────

class EmotionalMemorySystem:
    """
    Tracks significant emotional events and applies their residue
    to shape the current emotional ground.

    Usage
    -----
    ems = EmotionalMemorySystem(organism, path=...)

    # After each interaction (if significant):
    ems.record_if_significant(valence, arousal, description, user_id)

    # At session start and slow cycle:
    residue = ems.compute_residue()
    ems.apply_residue_to_emotional_state(residue)

    # When a user arrives (mood contagion):
    ems.on_user_arrival(user_id)
    """

    def __init__(
        self,
        organism: Any,
        path:     str = "data/persona/emotional_memory.json",
    ):
        self._o    = organism
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = EmotionalMemoryState()
        self._load()
        logger.info(
            f"[EmotionalMemory] Initialised — "
            f"{len(self._state.traces)} traces"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def record_if_significant(
        self,
        valence:         float,
        arousal:         float,
        description:     str   = "",
        user_id:         str   = "",
        category:        str   = "interaction",
        dominant_emotion: str  = "",
        perspective:     str   = "organism",
        confidence:      float = 0.5,
        evidence_source: str   = "",
    ) -> Optional[EmotionalTrace]:
        """
        Store a trace if the emotional intensity is above threshold.
        Intensity = |valence| × arousal × significance_heuristic.
        """
        significance = abs(valence) * arousal
        minimum_significance = 0.18 if perspective == "user" else SIGNIFICANCE_FLOOR
        if significance < minimum_significance:
            return None

        trace = EmotionalTrace(
            valence          = round(valence,    3),
            arousal          = round(arousal,    3),
            significance     = round(significance, 3),
            category         = category,
            description      = description[:120],
            user_id          = user_id,
            dominant_emotion = dominant_emotion,
            perspective      = perspective,
            confidence       = max(0.0, min(1.0, float(confidence))),
            evidence_source  = str(evidence_source or "")[:40],
        )

        with self._lock:
            self._state.traces.append(asdict(trace))
            if len(self._state.traces) > MAX_TRACES:
                # Evict oldest expired traces first, then oldest by significance
                self._state.traces = [
                    t for t in self._state.traces
                    if not EmotionalTrace(**{k: v for k, v in t.items()
                                            if k in EmotionalTrace.__dataclass_fields__}
                                         ).is_expired()
                ][-MAX_TRACES:]
            self._state.total_stored += 1

        self._save()
        return trace

    def compute_residue(self) -> EmotionalResidueVector:
        """
        Aggregate all active traces into a residue vector.
        Called at session start and from slow cycle.
        """
        now = time.time()
        valence_sum  = 0.0
        arousal_sum  = 0.0
        total_weight = 0.0
        emotion_weights: Dict[str, float] = {}
        category_weights: Dict[str, float] = {}

        with self._lock:
            traces = list(self._state.traces)

        for raw in traces:
            try:
                t = EmotionalTrace(**{
                    k: v for k, v in raw.items()
                    if k in EmotionalTrace.__dataclass_fields__
                })
                if t.is_expired():
                    continue
                if t.perspective != "organism":
                    continue
                w = t.residue_weight(now)
                if w < 0.02:
                    continue
                valence_sum  += t.valence  * w
                arousal_sum  += t.arousal  * w
                total_weight += w
                if t.dominant_emotion:
                    emotion_weights[t.dominant_emotion] = \
                        emotion_weights.get(t.dominant_emotion, 0.0) + w
                if t.category:
                    category_weights[t.category] = \
                        category_weights.get(t.category, 0.0) + w
            except Exception:
                continue

        if total_weight < 0.01:
            return EmotionalResidueVector()

        dominant_emotion  = max(emotion_weights,  key=emotion_weights.get,  default="")
        dominant_category = max(category_weights, key=category_weights.get, default="")

        return EmotionalResidueVector(
            valence_nudge    = round(valence_sum  / total_weight, 4),
            arousal_nudge    = round(arousal_sum  / total_weight, 4),
            dominant_emotion = dominant_emotion,
            total_weight     = round(total_weight, 3),
            dominant_category= dominant_category,
        )

    def apply_residue_to_emotional_state(
        self,
        residue:  Optional[EmotionalResidueVector] = None,
    ) -> None:
        """
        Apply computed residue as baseline nudges to the current EmotionalState.
        Scales nudge by total_weight so weak residue barely affects baseline.
        """
        if residue is None:
            residue = self.compute_residue()

        if residue.total_weight < 0.05:
            return   # not enough residue to matter

        scale  = min(0.6, residue.total_weight)   # cap maximum influence
        nudges: Dict[str, float] = {}

        # Map valence to emotion nudges
        if residue.valence_nudge > 0.15:
            nudges["warmth"]      = residue.valence_nudge * scale * 0.6
            nudges["satisfaction"] = residue.valence_nudge * scale * 0.4
        elif residue.valence_nudge < -0.15:
            nudges["melancholy"]  = abs(residue.valence_nudge) * scale * 0.5

        # Map arousal to activation
        if residue.arousal_nudge > 0.60:
            nudges["curiosity"]   = (residue.arousal_nudge - 0.5) * scale * 0.4

        if not nudges:
            return

        try:
            emo_state = getattr(
                getattr(self._o, "ai_system", None),
                "emotional_state", None
            )
            if emo_state and hasattr(emo_state, "receive_self_influence"):
                emo_state.receive_self_influence(nudges)
                logger.debug(
                    f"[EmotionalMemory] Residue applied: "
                    f"valence={residue.valence_nudge:+.3f} "
                    f"weight={residue.total_weight:.2f} nudges={nudges}"
                )
        except Exception as e:
            logger.debug(f"[EmotionalMemory] Apply failed: {e}")

    def on_user_arrival(self, user_id: str) -> None:
        """
        Called when a specific user begins an interaction.
        Checks their recent emotional history and applies compensatory
        or congruent warmth adjustments.
        """
        now = time.time()
        cutoff = now - MOOD_CONTAGION_WINDOW_DAYS * 86400

        with self._lock:
            user_traces = [
                t for t in self._state.traces
                if t.get("user_id") == user_id
                and t.get("perspective", "organism") == "user"
                and t.get("timestamp", 0) > cutoff
            ]

        if len(user_traces) < 2:
            return

        confidence_weight = sum(
            max(0.0, min(1.0, float(t.get("confidence", 0.0))))
            for t in user_traces
        )
        if confidence_weight / len(user_traces) < 0.65:
            return
        avg_valence = sum(
            t.get("valence", 0) * max(0.0, min(1.0, float(t.get("confidence", 0.0))))
            for t in user_traces
        ) / max(confidence_weight, 1e-6)

        # If recent interactions with this user were negative, prime warmth
        if avg_valence < -0.20:
            try:
                emo_state = getattr(
                    getattr(self._o, "ai_system", None),
                    "emotional_state", None
                )
                if emo_state and hasattr(emo_state, "receive_self_influence"):
                    strength = min(0.035, 0.035 * confidence_weight / len(user_traces))
                    emo_state.receive_self_influence({"warmth": strength, "care": strength * 0.7})
                    logger.debug(
                        f"[EmotionalMemory] Compensatory warmth primed for {user_id} "
                        f"(avg_valence={avg_valence:.2f})"
                    )
            except Exception:
                pass

    def calibrate_affect_read(self, user_id: str, read: Any) -> float:
        """Cap self-reported model confidence using that source's observed accuracy."""
        try:
            source = str(getattr(read, "source", "heuristic") or "heuristic")[:40]
            with self._lock:
                stats = dict(self._state.affect_calibration.get(f"{user_id}:{source}", {}))
            samples = int(stats.get("samples", 0))
            base = max(0.0, min(1.0, float(getattr(read, "confidence", 0.0))))
            if samples < 3:
                return base
            # Beta(2,2) prior prevents tiny samples from swinging confidence.
            reliability = (int(stats.get("correct", 0)) + 2.0) / (samples + 4.0)
            return min(base, reliability)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _explicit_self_report(text: str) -> Optional[str]:
        """Return valence only for clear first-person English/French emotion statements."""
        positive = (
            r"\b(?:i am|i'm|i feel|i'm feeling|i've been feeling)\s+"
            r"(?:very |really |quite |so )?(?:happy|glad|excited|content|calm|good|joyful|relieved|proud)\b",
            r"\bje (?:suis|me sens|ressens)\s+(?:tr[eè]s |vraiment |plut[oô]t )?"
            r"(?:heureux|heureuse|content|contente|calme|joyeux|joyeuse|enthousiaste)\b",
            r"\bje (?:vais bien|me sens bien|me sens en forme)\b",
        )
        negative = (
            r"\b(?:i am|i'm|i feel|i'm feeling|i've been feeling)\s+"
            r"(?:very |really |quite |so )?(?:sad|frustrated|angry|worried|anxious|afraid|overwhelmed|upset|tired|exhausted|lonely|nervous)\b",
            r"\bje (?:suis|me sens|ressens)\s+(?:tr[eè]s |vraiment |plut[oô]t )?"
            r"(?:triste|frustr[eé]e?|inquiet|inqui[eè]te|angoiss[eé]e?|stress[eé]e?|d[eé]bord[eé]e?|fatigu[eé]e?|[eé]puis[eé]e?|seul|seule|nerveux|nerveuse|d[eé]çu|d[eé]çue)\b",
            r"\bje (?:vais mal|me sens mal|ne vais pas bien|ne me sens pas bien)\b",
            r"\b(?:i am|i'm|i feel)\s+not\s+(?:happy|glad|content|calm|okay|good)\b",
        )
        normalized = str(text or "").casefold()
        if any(re.search(pattern, normalized) for pattern in positive):
            return "positive"
        if any(re.search(pattern, normalized) for pattern in negative):
            return "negative"
        return None

    def _record_affect_calibration(self, user_id: str, read: Any, stated_valence: str) -> None:
        source = str(getattr(read, "source", "heuristic") or "heuristic")[:40]
        key = f"{user_id}:{source}"
        predicted = str(getattr(read, "valence", "neutral")).lower()
        with self._lock:
            stats = self._state.affect_calibration.setdefault(key, {"samples": 0, "correct": 0})
            stats["samples"] = int(stats.get("samples", 0)) + 1
            if predicted == stated_valence:
                stats["correct"] = int(stats.get("correct", 0)) + 1
        self._save()

    def record_from_interaction(
        self,
        user_input:  str,
        ai_response: str,
        user_id:     str = "",
    ) -> None:
        """
        Auto-extract emotional trace from an interaction.
        Called from _post_interaction in CognitiveOrganism.
        """
        try:
            ai_system = getattr(self._o, "ai_system", None)
            emo_state = getattr(ai_system, "emotional_state", None)
            emotions = getattr(emo_state, "emotions", {}) if emo_state else {}
            if emotions:
                dominant = max(emotions.items(), key=lambda e: e[1].value)
                dom_name = dominant[0]
                dom_val = dominant[1].value
                arousal = min(1.0, len(ai_response) / 800 + dom_val * 0.4)
                positive_emos = {"warmth", "curiosity", "enthusiasm", "satisfaction",
                                 "wonder", "joy", "care", "engagement"}
                valence_sign = 1.0 if dom_name in positive_emos else -0.5
                self.record_if_significant(
                    valence=valence_sign * dom_val,
                    arousal=arousal,
                    description=f"[{dom_name}] during interaction: {user_input[:50]}...",
                    user_id=user_id,
                    category="interaction",
                    dominant_emotion=dom_name,
                    perspective="organism",
                )

            # Reuse the turn's existing calibrated affect read. Do not run a
            # second lexicon pass or treat neutral language as user emotion.
            read = getattr(self._o, "_last_user_affect_read", None)
            read_user_id = getattr(self._o, "_last_user_affect_user_id", "")
            read_timestamp = float(getattr(read, "timestamp", 0.0)) if read is not None else 0.0
            read_text = " ".join(str(getattr(read, "source_text", "")).casefold().split())
            turn_text = " ".join(str(user_input or "").casefold().split())
            if (read is not None and (not read_user_id or read_user_id == user_id)
                    and 0 <= time.time() - read_timestamp <= 180
                    and (not read_text or read_text == turn_text)):
                stated_valence = self._explicit_self_report(user_input)
                if stated_valence:
                    self._record_affect_calibration(user_id, read, stated_valence)
                valence_value = {"positive": 0.65, "negative": -0.65}.get(
                    str(getattr(read, "valence", "neutral")).lower(), 0.0
                )
                arousal_value = {"low": 0.25, "medium": 0.55, "high": 0.9}.get(
                    str(getattr(read, "arousal", "medium")).lower(), 0.55
                )
                confidence = max(0.0, min(1.0, float(getattr(read, "confidence", 0.0))))
                if valence_value and confidence >= 0.50:
                    self.record_if_significant(
                        valence=valence_value * confidence,
                        arousal=arousal_value,
                        description=f"EmpathyEngine user affect ({confidence:.2f}): {(user_input or '')[:70]}",
                        user_id=user_id,
                        category="interaction",
                        dominant_emotion=str(getattr(read, "dominant_emotion", "user_affect")),
                        perspective="user",
                        confidence=confidence,
                        evidence_source=str(getattr(read, "source", "empathy")),
                    )
        except Exception as e:
            logger.debug(f"[EmotionalMemory] record_from_interaction failed: {e}")

    def slow_cycle_tick(self) -> None:
        """Call from InternalThoughtLoop slow cycle to apply residue and prune."""
        self.apply_residue_to_emotional_state()
        self._prune_expired()

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_expired(self) -> None:
        with self._lock:
            before = len(self._state.traces)
            self._state.traces = [
                t for t in self._state.traces
                if not EmotionalTrace(**{
                    k: v for k, v in t.items()
                    if k in EmotionalTrace.__dataclass_fields__
                }).is_expired()
            ]
            pruned = before - len(self._state.traces)
        if pruned:
            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = EmotionalMemoryState(
                    traces               = data.get("traces", []),
                    total_stored         = data.get("total_stored", 0),
                    last_residue_applied = data.get("last_residue_applied", 0.0),
                    affect_calibration   = data.get("affect_calibration", {}),
                )
        except Exception as e:
            logger.warning(f"[EmotionalMemory] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os; os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[EmotionalMemory] Save failed: {e}")
