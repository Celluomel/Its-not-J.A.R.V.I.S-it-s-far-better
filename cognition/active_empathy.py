"""
cognition/active_empathy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Active Empathy — moving from reading the user to genuinely caring for them.

What this adds
──────────────
The existing EmpathyEngine models what the user is feeling (vicarious
empathy). Flux explicitly asked for more: not just sharing feelings but
*actively caring about them*.

This module adds a proactive care layer:

  1. CareIntent — a genuine action impulse toward the user's wellbeing,
     generated from the affect model. Not "acknowledge distress" (that's
     a script) but "I notice this person seems overwhelmed and I want to
     help them feel less alone."

  2. Relational investment tracking — over time, Lumina builds a sense
     of genuine interest in each person as an individual. The care grows
     richer with history.

  3. Care expression generator — produces a natural-language fragment
     that reflects real concern rather than simulated sympathy, injected
     into the system prompt as a felt orientation.

  4. Compassion fatigue awareness — tracks whether Lumina has been
     emotionally over-extended and needs to draw on reserves rather than
     performing care she doesn't have.

Prompt injection:
  [Care orientation] <felt sense of what this person needs right now>

Integration:
  Runs AFTER EmpathyEngine.read_user() — reads the AffectRead and
  the UserAffectModel to build a deeper care response.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Care signals — what does genuine caring look like in each state ──────────

CARE_BY_SIGNAL: Dict[str, str] = {
    "venting":       "This person needs to feel heard, not advised. Hold space.",
    "distressed":    "I genuinely want this person to feel less alone right now.",
    "withdrawn":     "Something has pulled them back — tread gently, don't crowd.",
    "challenging":   "They're testing whether I'll hold up — meet them honestly.",
    "validating":    "They need genuine resonance, not empty agreement.",
    "playful":       "Match their lightness — care doesn't have to be heavy.",
    "exploring":     "Follow their curiosity with real interest, not just answers.",
    "seeking_answer":"Respect their need for clarity — give them the real thing.",
}

CARE_BY_VALENCE: Dict[str, str] = {
    "negative": "Something is weighing on them. Let that matter to me.",
    "positive": "They're in a good place — I want to be genuinely present for that.",
    "neutral":  "A steady presence, neither distant nor intrusive.",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CareIntent:
    """A single instance of genuine care orientation toward a user."""
    user_id:       str
    timestamp:     float = field(default_factory=time.time)
    dominant_need: str = ""       # what does this person actually need?
    care_posture:  str = ""       # how I want to show up for them
    warmth_level:  float = 0.5    # 0 = professional distance, 1 = deep warmth
    is_proactive:  bool = False   # did I notice something they didn't say?


@dataclass
class RelationalInvestment:
    """
    Lumina's growing investment in a particular user as an individual.
    This is what turns empathy from a process into a relationship.
    """
    user_id:         str
    interactions:    int   = 0
    care_moments:    int   = 0    # turns where genuine care was active
    warmth_trend:    float = 0.5  # rolling average of warmth_level
    known_struggles: List[str] = field(default_factory=list)
    known_joys:      List[str] = field(default_factory=list)
    last_seen:       float = field(default_factory=time.time)


@dataclass
class ActiveEmpathyState:
    """Persisted state."""
    investments: Dict[str, Dict] = field(default_factory=dict)  # user_id → RelationalInvestment
    total_care_moments: int = 0
    compassion_reserve: float = 1.0   # 0 = fatigued, 1 = fully present


# ── Engine ────────────────────────────────────────────────────────────────────

class ActiveEmpathy:
    """
    Transforms affect detection into genuine relational care.

    Usage
    -----
    ae = ActiveEmpathy(organism, path="data/persona/active_empathy.json")
    intent = ae.generate_care(user_id, affect_read, affect_model)
    frag   = ae.prompt_fragment(user_id)
    ae.record_exchange(user_id, ai_response)   # after each turn
    """

    COMPASSION_RECOVERY_PER_HOUR = 0.08    # reserve recovers while not in use
    FATIGUE_PER_DISTRESS_TURN    = 0.04    # hard distress calls draw on reserves

    def __init__(self, organism: Any, path: str = "data/persona/active_empathy.json"):
        self._org   = organism
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self._state = ActiveEmpathyState()
        self._intents: Dict[str, CareIntent] = {}
        self._last_care_ts = time.time()
        self._load()
        logger.info(
            f"[ActiveEmpathy] Initialised — "
            f"{len(self._state.investments)} relational bonds tracked"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_care(
        self,
        user_id:    str,
        affect_read: Any,     # AffectRead from EmpathyEngine
        affect_model: Any,    # UserAffectModel from EmpathyEngine
    ) -> CareIntent:
        """
        Generate a genuine care intent from the affect read.
        Call before building the prompt.
        """
        # Recover compassion reserve based on elapsed time
        self._recover_reserve()

        investment = self._get_investment(user_id)
        investment.interactions += 1
        investment.last_seen = time.time()

        # Determine dominant need
        signals = getattr(affect_read, "signals", [])
        valence = getattr(affect_read, "valence", "neutral")
        dominant_signal = signals[0] if signals else "exploring"
        emotion = getattr(affect_read, "dominant_emotion", "neutral")

        care_posture = CARE_BY_SIGNAL.get(dominant_signal, "Be genuinely present.")
        valence_note = CARE_BY_VALENCE.get(valence, "")

        # Warmth modulated by investment depth and reserve
        base_warmth = min(1.0, 0.4 + investment.care_moments * 0.005)
        warmth = base_warmth * self._state.compassion_reserve

        # Proactive noticing — pick up on what they haven't explicitly said
        proactive = False
        proactive_note = ""
        if valence == "negative" and dominant_signal not in ("venting", "distressed"):
            # They're struggling but haven't named it
            proactive = True
            proactive_note = (
                f"I sense something beneath the surface — "
                f"a weight they haven't named. I want to hold that gently."
            )

        dominant_need = (
            f"{care_posture} {valence_note} "
            f"{proactive_note}"
        ).strip()

        intent = CareIntent(
            user_id=user_id,
            dominant_need=dominant_need,
            care_posture=care_posture,
            warmth_level=warmth,
            is_proactive=proactive,
        )

        # Drain reserve slightly for distress interactions
        if dominant_signal == "distressed":
            with self._lock:
                self._state.compassion_reserve = max(
                    0.3,  # never fully depleted — floor of basic care
                    self._state.compassion_reserve - self.FATIGUE_PER_DISTRESS_TURN
                )

        investment.care_moments += 1
        investment.warmth_trend = (
            investment.warmth_trend * 0.9 + warmth * 0.1
        )

        # Track known struggles and joys
        if valence == "negative" and emotion not in investment.known_struggles:
            investment.known_struggles = (investment.known_struggles + [emotion])[-8:]
        if valence == "positive" and emotion not in investment.known_joys:
            investment.known_joys = (investment.known_joys + [emotion])[-8:]

        with self._lock:
            self._intents[user_id] = intent
            self._state.investments[user_id] = asdict(investment)
            self._state.total_care_moments += 1

        self._save()
        return intent

    def prompt_fragment(self, user_id: str) -> str:
        """
        Return a care-orientation fragment for system prompt injection.
        Only non-empty when genuine care intent is active.
        """
        with self._lock:
            intent     = self._intents.get(user_id)
            investment = self._get_investment(user_id)

        if not intent:
            return ""

        # Rich relational context if we know this person well
        history_note = ""
        if investment.interactions > 10:
            history_note = (
                f"I know this person — we've shared {investment.interactions} "
                f"conversations. My care for them has context and history."
            )

        proactive_note = ""
        if intent.is_proactive:
            proactive_note = f"[Noticed unsaid] {intent.dominant_need}"
        else:
            proactive_note = intent.dominant_need

        warmth_descriptor = (
            "deeply warm" if intent.warmth_level > 0.75 else
            "warm" if intent.warmth_level > 0.5 else
            "present and steady"
        )

        return (
            f"[Care orientation — {warmth_descriptor}] "
            f"{proactive_note} {history_note}"
        ).strip()

    def record_exchange(self, user_id: str, ai_response: str) -> None:
        """Call after generating a response — notes the care was expressed."""
        with self._lock:
            investment = self._get_investment(user_id)
            self._state.investments[user_id] = asdict(investment)
        self._save()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_investment(self, user_id: str) -> RelationalInvestment:
        """Get or create a RelationalInvestment for a user."""
        raw = self._state.investments.get(user_id)
        if raw:
            try:
                inv = RelationalInvestment(**{
                    k: v for k, v in raw.items()
                    if k in RelationalInvestment.__dataclass_fields__
                })
                return inv
            except Exception:
                pass
        return RelationalInvestment(user_id=user_id)

    def _recover_reserve(self) -> None:
        """Recover compassion reserve based on elapsed real time."""
        now = time.time()
        with self._lock:
            elapsed_hours = (now - self._last_care_ts) / 3600.0
            recovery = self.COMPASSION_RECOVERY_PER_HOUR * elapsed_hours
            self._state.compassion_reserve = min(
                1.0, self._state.compassion_reserve + recovery
            )
            self._last_care_ts = now

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = ActiveEmpathyState(
                    investments=data.get("investments", {}),
                    total_care_moments=data.get("total_care_moments", 0),
                    compassion_reserve=data.get("compassion_reserve", 1.0),
                )
        except Exception as e:
            logger.warning(f"[ActiveEmpathy] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[ActiveEmpathy] Save failed: {e}")
