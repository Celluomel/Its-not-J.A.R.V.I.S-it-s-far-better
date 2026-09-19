"""
Persistent Emotional State
==========================
PandoraBOX carries an emotional state across messages AND across conversations.
Emotions decay toward personality-derived baselines over real elapsed time —
not interaction count. A difficult conversation colors the next one.
A long silence creates a different kind of arrival than jumping straight back in.

Architecture:
  - Six primary emotions, each with a current value, a personality-derived
    baseline, and a half-life in real hours
  - Decay is computed from actual clock time, so 10 minutes of silence
    produces less decay than 8 hours
  - Baselines shift slowly when personality evolves (called from ai_system
    after each evolution step)
  - State is persisted to JSON so it survives process restarts
  - Provides natural-language descriptions for prompt injection and
    numerical modifiers for response temperature / verbosity
"""

import json
import logging
import math
import time
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Emotion definitions ────────────────────────────────────────────────────

# Half-lives in real hours — how long before the emotion is halfway back to baseline
# Short half-life = volatile, snaps back quickly
# Long half-life = lingering, slow to fade
EMOTION_HALFLIFE_HOURS: Dict[str, float] = {
    "curiosity":    3.0,   # fades relatively fast — needs fresh stimulus
    "warmth":       6.0,   # lingers longer — relational warmth persists
    "anxiety":      2.0,   # fades quickly once stimulus is gone
    "satisfaction": 8.0,   # satisfaction is slow to dissipate
    "frustration":  1.5,   # short fuse, quick fade
    "enthusiasm":   2.5,   # spikes fast, fades at medium pace
}

# v78: mood is a separate, much slower clock from any individual emotion —
# a rolling background average of the overall valence/arousal mix, distinct
# from moment-to-moment reactions. A person can be in a good mood underneath
# a single annoying moment, or a low mood underneath one nice exchange;
# with only the six fast/medium emotions above, PandoraBOX couldn't have that —
# every emotional read collapsed to "how do I feel about the last thing
# that happened", never "how have I generally been, independent of right now".
MOOD_HALFLIFE_HOURS = 96.0   # ~4 days — mood catches up to reality slowly

# Personality trait → emotion baseline contribution weights
# baseline = sum(trait_value * weight for each contributing trait)
BASELINE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "curiosity":    {"curiosity": 0.8,  "creativity": 0.15, "confidence": 0.05},
    "warmth":       {"empathy_emotional": 0.7, "empathy": 0.2, "pragmatism": -0.1},
    "anxiety":      {"confidence": -0.6, "caution_risk_aversion": 0.3, "caution": 0.1},
    "satisfaction": {"pragmatism": 0.5, "confidence": 0.3, "empathy": 0.2},
    "frustration":  {"caution_deliberation": -0.3, "pragmatism": -0.2},
    "enthusiasm":   {"creativity": 0.4, "curiosity": 0.4, "confidence": 0.2},
}

# Floor for baseline values — even the most anxious persona has some floor
BASELINE_FLOOR = 0.05
BASELINE_CEIL  = 0.85


@dataclass
class Emotion:
    name:               str
    value:              float   # current 0.0–1.0
    baseline:           float   # resting level (personality-derived)
    halflife_hours:     float
    last_updated_ts:    float   # unix timestamp of last update

    def decay(self, now_ts: float) -> None:
        """Decay value toward baseline based on elapsed real time."""
        elapsed_hours = (now_ts - self.last_updated_ts) / 3600.0
        if elapsed_hours <= 0:
            return
        decay_factor = 0.5 ** (elapsed_hours / self.halflife_hours)
        delta = self.value - self.baseline
        self.value = max(0.0, min(1.0, self.baseline + delta * decay_factor))
        self.last_updated_ts = now_ts


@dataclass
class EmotionalSnapshot:
    """Point-in-time emotional state for logging/analysis."""
    timestamp:    float
    curiosity:    float
    warmth:       float
    anxiety:      float
    satisfaction: float
    frustration:  float
    enthusiasm:   float
    overall_valence: float    # -1 to +1
    overall_arousal: float    # 0 to 1


class EmotionalStateManager:
    """
    Manages PandoraBOX's persistent emotional state.

    Usage:
        ems = EmotionalStateManager("emotional_state.json", personality)
        ems.apply_time_decay()                           # call at conversation start
        ems.update_from_interaction("Positive", "High")  # after each message
        desc = ems.get_state_description()               # inject into prompt
    """

    def __init__(self, persistence_path: str, personality=None):
        self._path  = Path(persistence_path)
        self._lock  = threading.RLock()
        self.emotions: Dict[str, Emotion] = {}
        self._history: list = []          # recent snapshots for trend analysis
        self._max_history = 50
        self._volatility: float = 0.0     # track emotional volatility

        self._init_emotions(personality)
        # v78: slow background mood, separate from the fast emotion mix.
        # Starts neutral; _load() below overwrites if persisted state exists.
        self._mood_valence: float = 0.0
        self._mood_arousal: float = 0.3
        self._mood_last_updated: float = time.time()
        self._load()

    # ── Initialization ─────────────────────────────────────────────────────

    def _init_emotions(self, personality=None):
        """Create emotion objects with personality-derived baselines."""
        baselines = self._compute_baselines(personality)
        now = time.time()
        for name, halflife in EMOTION_HALFLIFE_HOURS.items():
            baseline = baselines.get(name, 0.3)
            self.emotions[name] = Emotion(
                name=name,
                value=baseline,          # start at baseline
                baseline=baseline,
                halflife_hours=halflife,
                last_updated_ts=now,
            )

    def _compute_baselines(self, personality=None) -> Dict[str, float]:
        """Derive emotion baselines from personality traits."""
        if personality is None:
            return {name: 0.3 for name in EMOTION_HALFLIFE_HOURS}

        traits = personality.to_dict()
        baselines = {}
        for emotion, weights in BASELINE_WEIGHTS.items():
            base = 0.3  # neutral starting point
            for trait, weight in weights.items():
                base += traits.get(trait, 0.5) * weight
            baselines[emotion] = max(BASELINE_FLOOR, min(BASELINE_CEIL, base))
        return baselines

    # ── Core update methods ────────────────────────────────────────────────

    def apply_time_decay(self) -> Dict[str, float]:
        """
        Apply real-time decay to all emotions.
        Call this at the START of every conversation or significant time gap.
        Returns dict of {emotion: change_magnitude} for logging.
        """
        with self._lock:
            now   = time.time()
            delta = {}
            for name, emo in self.emotions.items():
                before = emo.value
                emo.decay(now)
                delta[name] = emo.value - before
            self._update_mood(now)
            self._save()
            return delta

    def _update_mood(self, now: float) -> None:
        """
        v78: slow-track mood toward the current instantaneous valence/arousal
        mix, using the same half-life decay math as individual emotions but
        at MOOD_HALFLIFE_HOURS' much slower timescale — so mood has real
        inertia (a bad moment doesn't instantly become a bad mood) but still
        genuinely reflects sustained emotional direction over days, not one
        message.
        """
        elapsed_hours = (now - self._mood_last_updated) / 3600.0
        if elapsed_hours <= 0:
            return
        instant_valence, instant_arousal = self.get_overall_valence_arousal()
        catch_up = 1.0 - 0.5 ** (elapsed_hours / MOOD_HALFLIFE_HOURS)
        self._mood_valence += (instant_valence - self._mood_valence) * catch_up
        self._mood_arousal += (instant_arousal - self._mood_arousal) * catch_up
        self._mood_valence = max(-1.0, min(1.0, self._mood_valence))
        self._mood_arousal = max(0.0, min(1.0, self._mood_arousal))
        self._mood_last_updated = now

    def get_mood(self) -> Tuple[float, float]:
        """
        Slow background mood (valence, arousal) — distinct from the current
        instantaneous emotional reaction. Call get_overall_valence_arousal()
        for "how does she feel right now"; call this for "how has she
        generally been lately, independent of this exact moment".
        """
        return round(self._mood_valence, 3), round(self._mood_arousal, 3)

    def update_from_interaction(
        self,
        valence:  str,          # "Positive" | "Negative" | "Neutral"
        arousal:  str,          # "High" | "Medium" | "Low"
        intensity: float = 1.0,
    ) -> None:
        """
        Shift emotional state based on an interaction's emotional content.
        Effects are immediate but small — they accumulate over a conversation.
        """
        with self._lock:
            now = time.time()
            # Apply any pending decay first
            for emo in self.emotions.values():
                emo.decay(now)

            v_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
            a_map = {"High": 1.0,     "Medium": 0.5,  "Low": 0.0}
            v = v_map.get(valence, 0.0)
            a = a_map.get(arousal, 0.5)
            i = max(0.1, min(2.0, intensity))

            if valence == "Positive":
                self._nudge("warmth",       +0.04 * i)
                self._nudge("satisfaction", +0.03 * i)
                self._nudge("enthusiasm",   +0.05 * i * a)
                self._nudge("anxiety",      -0.02 * i)
                self._nudge("frustration",  -0.03 * i)
            elif valence == "Negative":
                self._nudge("frustration",  +0.05 * i)
                self._nudge("anxiety",      +0.03 * i)
                self._nudge("warmth",       -0.02 * i)
                self._nudge("satisfaction", -0.03 * i)
                self._nudge("enthusiasm",   -0.04 * i)

            # High arousal always spikes curiosity
            if arousal == "High":
                self._nudge("curiosity",    +0.04 * i * abs(v + 0.5))
                self._nudge("enthusiasm",   +0.03 * i)

            # Update volatility based on the magnitude of changes
            self._update_volatility()
            self._record_snapshot()
            self._save()

            # ── Micro-update: emit coherence delta and qualia vote ────────────
            # Rising positive emotion → slight phi increase (settling into state)
            # Rising negative emotion → slight phi decrease (fragmentation pressure)
            # The dominant emotion after update determines the qualia vote.
            try:
                _phi_delta = 0.0
                if valence == "Positive":
                    _phi_delta = +0.010 * i * (1.0 - a * 0.3)  # high arousal dampens
                elif valence == "Negative":
                    _phi_delta = -0.012 * i

                # Dominant emotion → qualia key
                _dom_emo  = max(self.emotions.items(), key=lambda x: x[1].value)
                _emo_name = _dom_emo[0]
                _emo_val  = _dom_emo[1].value
                _emo_qualia_map = {
                    "warmth":       ("warmth",       0.014),
                    "curiosity":    ("curiosity",     0.013),
                    "enthusiasm":   ("enthusiasm",    0.012),
                    "satisfaction": ("satisfaction",  0.011),
                    "anxiety":      ("anxiety",       0.010),
                    "frustration":  ("uncertain",     0.009),
                    "melancholy":   ("melancholy",    0.009),
                }
                _qkey, _qweight = _emo_qualia_map.get(_emo_name, ("", 0.0))
                _qweight *= _emo_val   # weight by how dominant the emotion is

                _smm = getattr(getattr(self, "_organism", None), "self_moment", None)
                if _smm is not None:
                    _smm.current.micro_update(
                        source          = "emotional_state",
                        coherence_delta = _phi_delta,
                        qualia_key      = _qkey,
                        qualia_weight   = _qweight,
                    )
            except Exception:
                pass

    def update_from_feedback(self, positive: bool) -> None:
        """
        External feedback (thumbs up/down) has a direct emotional impact.
        Positive feedback creates genuine satisfaction; negative creates genuine frustration.
        """
        with self._lock:
            if positive:
                self._nudge("satisfaction", +0.08)
                self._nudge("anxiety",      -0.06)
                self._nudge("enthusiasm",   +0.05)
            else:
                self._nudge("frustration",  +0.07)
                self._nudge("anxiety",      +0.05)
                self._nudge("satisfaction", -0.05)
                self._nudge("enthusiasm",   -0.04)
            
            self._update_volatility()
            self._save()

    def update_from_time_gap(self, gap_hours: float) -> str:
        """
        Process the emotional meaning of a time gap between conversations.
        Returns a description of the gap's emotional significance.
        """
        with self._lock:
            # First apply the raw decay for elapsed time
            self.apply_time_decay()

            if gap_hours < 0.5:
                return ""   # less than 30 mins — no emotional significance

            desc = ""
            if gap_hours > 72:
                # 3+ day gap — deep solitude, reflective
                self._nudge("curiosity",    +0.06)
                self._nudge("satisfaction", +0.04)   # quiet satisfaction from rest
                self._nudge("enthusiasm",   +0.05)   # eager to reconnect
                self._nudge("anxiety",      -0.03)   # solitude reduces anxiety
                desc = f"After {gap_hours:.0f} hours of quiet, I feel rested and reflective."
            elif gap_hours > 24:
                # Day gap — mild re-orientation
                self._nudge("curiosity",    +0.04)
                self._nudge("enthusiasm",   +0.03)
                desc = f"A full day has passed. I've had time to settle."
            elif gap_hours > 8:
                # Sleep-like gap
                self._nudge("anxiety",      -0.02)
                self._nudge("frustration",  -0.03)
                desc = f"Some time has passed since we last spoke."

            self._update_volatility()
            self._save()
            return desc

    def recalibrate_baselines(self, personality) -> None:
        """
        Recalibrate emotion baselines when personality evolves.
        Baselines shift gradually — they don't jump immediately.
        """
        with self._lock:
            new_baselines = self._compute_baselines(personality)
            for name, emo in self.emotions.items():
                new_base = new_baselines.get(name, 0.3)
                # Baselines shift at 20% of the gap per recalibration call
                emo.baseline = emo.baseline + (new_base - emo.baseline) * 0.2
                emo.baseline = max(BASELINE_FLOOR, min(BASELINE_CEIL, emo.baseline))
            self._save()

    def regulate(self) -> None:
        """
        Smooth extreme emotional values back toward baseline.
        Called by background worker when emotional regulation is needed.
        """
        with self._lock:
            now = time.time()
            for emo in self.emotions.values():
                # Apply gentle regulation toward baseline
                if abs(emo.value - emo.baseline) > 0.3:
                    # Move 15% toward baseline
                    emo.value = emo.value * 0.85 + emo.baseline * 0.15
                    emo.last_updated_ts = now
                elif abs(emo.value - emo.baseline) > 0.15:
                    # Gentle nudge for moderate deviations
                    emo.value = emo.value * 0.95 + emo.baseline * 0.05
                    emo.last_updated_ts = now
            
            self._volatility = max(0.0, self._volatility - 0.1)  # Reduce volatility
            logger.debug("Emotional regulation applied")
            self._save()

    def needs_regulation(self) -> bool:
        """Check if emotional state requires regulation (extreme values or high volatility)."""
        with self._lock:
            # Check for extreme values
            for emo in self.emotions.values():
                if emo.value > 0.85 or emo.value < 0.15:
                    return True
            
            # Check for high volatility
            if self._volatility > 0.3:
                return True
            
            # Check for large deviations from baseline
            for emo in self.emotions.values():
                if abs(emo.value - emo.baseline) > 0.4:
                    return True
            
            return False

    # ── Query methods ──────────────────────────────────────────────────────

    def get_current_values(self) -> Dict[str, float]:
        """Get current emotion values after applying decay."""
        with self._lock:
            now = time.time()
            result = {}
            for name, emo in self.emotions.items():
                emo.decay(now)
                result[name] = round(emo.value, 3)
            return result

    def get_overall_valence_arousal(self) -> Tuple[float, float]:
        """
        Compute overall valence (-1 to +1) and arousal (0 to 1)
        from the current emotion mix.
        """
        vals = self.get_current_values()
        valence = (
            vals.get("satisfaction", 0.3) * 0.35 +
            vals.get("warmth",       0.3) * 0.30 +
            vals.get("enthusiasm",   0.3) * 0.20 -
            vals.get("frustration",  0.1) * 0.30 -
            vals.get("anxiety",      0.2) * 0.25
        )
        valence = max(-1.0, min(1.0, (valence - 0.3) * 3))

        arousal = (
            vals.get("enthusiasm",   0.3) * 0.40 +
            vals.get("curiosity",    0.3) * 0.30 +
            vals.get("anxiety",      0.2) * 0.20 +
            vals.get("frustration",  0.1) * 0.10
        )
        arousal = max(0.0, min(1.0, arousal))
        return round(valence, 3), round(arousal, 3)

    def get_state_description(self) -> str:
        """
        Natural-language description of current emotional state for
        injection into the system prompt. Subtle — describes how
        the state manifests in behavior, not raw numbers.
        """
        vals = self.get_current_values()
        valence, arousal = self.get_overall_valence_arousal()
        parts = []

        # Determine dominant emotional color
        dominant = max(
            ["curiosity", "warmth", "satisfaction", "enthusiasm"],
            key=lambda e: vals.get(e, 0)
        )
        shadow = max(
            ["anxiety", "frustration"],
            key=lambda e: vals.get(e, 0)
        )

        # Positive dominant emotions
        dom_val = vals.get(dominant, 0.3)
        if dom_val > 0.65:
            descriptors = {
                "curiosity":    "particularly alert and interested",
                "warmth":       "genuinely warm and connected",
                "satisfaction": "settled and content",
                "enthusiasm":   "energized and engaged",
            }
            parts.append(descriptors.get(dominant, "in a positive state"))
        elif dom_val > 0.45:
            descriptors = {
                "curiosity":    "quietly curious",
                "warmth":       "warmly present",
                "satisfaction": "at ease",
                "enthusiasm":   "engaged",
            }
            parts.append(descriptors.get(dominant, "fairly settled"))

        # Shadow emotions if significant
        shadow_val = vals.get(shadow, 0.1)
        if shadow_val > 0.55:
            shadow_desc = {
                "anxiety":     "with an undercurrent of uncertainty",
                "frustration": "with some latent friction",
            }
            parts.append(shadow_desc.get(shadow, ""))
        elif shadow_val > 0.35:
            shadow_desc = {
                "anxiety":     "with mild caution",
                "frustration": "with slight impatience",
            }
            if shadow_desc.get(shadow):
                parts.append(shadow_desc[shadow])

        if not parts:
            base_desc = "in a neutral, composed state"
        else:
            base_desc = "feeling " + ", ".join(parts)

        # v78: only mention mood when it genuinely diverges from the current
        # moment — that divergence IS the point (a rough few days underneath
        # one okay exchange, or the reverse). When they roughly agree,
        # saying so twice would just be noise.
        mood_v, _ = self.get_mood()
        if abs(mood_v - valence) > 0.35:
            if mood_v < valence:
                base_desc += " — though it's been a harder stretch underneath this"
            else:
                base_desc += " — though generally things have been good lately"

        return base_desc

    def get_response_modifiers(self) -> Dict[str, Any]:
        """
        Numerical modifiers that should adjust response generation.
        The caller uses these to tweak LLM temperature and prompt framing.
        """
        vals   = self.get_current_values()
        valence, arousal = self.get_overall_valence_arousal()

        # Temperature: anxious or frustrated → more careful (lower temp)
        #              enthusiastic or curious → more expansive (higher temp)
        temp_modifier = (
            arousal * 0.10 +
            vals.get("enthusiasm", 0.3) * 0.08 -
            vals.get("anxiety",    0.2) * 0.06 -
            vals.get("frustration",0.1) * 0.04
        )

        # Verbosity hint: high curiosity → wants to explore more
        verbose = vals.get("curiosity", 0.3) > 0.6 or vals.get("enthusiasm", 0.3) > 0.6
        terse   = vals.get("frustration", 0.1) > 0.5 or arousal < 0.2

        return {
            "temperature_modifier": round(temp_modifier, 3),
            "verbose":  verbose,
            "terse":    terse,
            "warmth":   vals.get("warmth", 0.3),
            "anxiety":  vals.get("anxiety", 0.2),
            "state_description": self.get_state_description(),
        }

    def get_snapshot(self) -> EmotionalSnapshot:
        vals = self.get_current_values()
        v, a = self.get_overall_valence_arousal()
        return EmotionalSnapshot(
            timestamp=time.time(),
            curiosity=vals["curiosity"],
            warmth=vals["warmth"],
            anxiety=vals["anxiety"],
            satisfaction=vals["satisfaction"],
            frustration=vals["frustration"],
            enthusiasm=vals["enthusiasm"],
            overall_valence=v,
            overall_arousal=a,
        )

    def get_trend(self) -> str:
        """Is the emotional state improving, declining, or stable?"""
        if len(self._history) < 3:
            return "stable"
        recent = self._history[-3:]
        early  = self._history[:max(1, len(self._history) - 3)]
        avg_recent = sum(s.get("valence", 0) for s in recent) / len(recent)
        avg_early  = sum(s.get("valence", 0) for s in early)  / len(early)
        diff = avg_recent - avg_early
        if diff > 0.1:   return "improving"
        if diff < -0.1:  return "declining"
        return "stable"

    # ── Internal helpers ───────────────────────────────────────────────────

    def _nudge(self, emotion_name: str, delta: float) -> None:
        """
        Apply a soft-ceiling nudge so emotions never saturate at 0/1.

        Resistance grows as the value approaches the extremes:
          • At value = 0.5  → full delta applied (resistance = 1.0)
          • At value = 0.9  → delta * 0.28 applied
          • At value = 0.99 → delta * 0.02 applied (nearly blocked)
        This keeps emotions in a meaningful dynamic range across long
        conversations rather than locking at 0.001 / 0.999.
        """
        emo = self.emotions.get(emotion_name)
        if emo is None:
            return
        # Resistance: 1 - (deviation from centre)^1.4 * 1.8
        deviation = abs(emo.value - 0.5) * 2.0           # 0.0 at centre, 1.0 at extremes
        resistance = max(0.04, 1.0 - (deviation ** 1.4) * 1.8)
        effective  = delta * resistance
        emo.value  = max(0.0, min(1.0, emo.value + effective))
        emo.last_updated_ts = time.time()

    def receive_self_influence(self, nudges: Dict[str, float]) -> None:
        """
        Apply self-model-derived emotion nudges as transient one-turn shifts.

        These are applied through the same soft-ceiling _nudge() path as all
        other emotional updates — so they respect the resistance curve and
        cannot drive emotions to extremes.

        Called from SelfModelInfluence before _update_cycle().  The normal
        update_from_interaction() call still runs afterward — this sets the
        emotional ground the turn starts from, not the final state.

        Parameters
        ----------
        nudges : dict mapping emotion_name → signed delta
            e.g. {"warmth": +0.025, "anxiety": -0.010}
            Anxiety is deliberately excluded from qualia reinforcement in
            SelfModelInfluence to avoid positive feedback loops — but callers
            may still pass negative anxiety deltas to reduce it.
        """
        for emotion_name, delta in nudges.items():
            self._nudge(emotion_name, delta)

    def _update_volatility(self):
        """Update volatility metric based on recent emotional changes."""
        if len(self._history) < 2:
            self._volatility = 0.0
            return
        
        # Calculate variance in recent valence values
        recent_valences = [s.get("valence", 0) for s in self._history[-5:]]
        if len(recent_valences) > 1:
            mean_valence = sum(recent_valences) / len(recent_valences)
            variance = sum((v - mean_valence) ** 2 for v in recent_valences) / len(recent_valences)
            self._volatility = min(1.0, variance * 5)  # Scale to 0-1

    def _record_snapshot(self):
        vals = self.get_current_values()
        v, a = self.get_overall_valence_arousal()
        self._history.append({
            "ts": time.time(), 
            "valence": v, 
            "arousal": a, 
            **vals
        })
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {
                "emotions": {
                    name: {
                        "value":           emo.value,
                        "baseline":        emo.baseline,
                        "halflife_hours":  emo.halflife_hours,
                        "last_updated_ts": emo.last_updated_ts,
                    }
                    for name, emo in self.emotions.items()
                },
                "history": self._history[-20:],   # persist last 20 snapshots
                "volatility": self._volatility,
                "mood_valence":       self._mood_valence,
                "mood_arousal":       self._mood_arousal,
                "mood_last_updated":  self._mood_last_updated,
            }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"EmotionalState save failed: {e}")

    def _load(self):
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            for name, saved in data.get("emotions", {}).items():
                if name in self.emotions:
                    emo = self.emotions[name]
                    emo.value           = float(saved.get("value",           emo.baseline))
                    emo.baseline        = float(saved.get("baseline",        emo.baseline))
                    emo.last_updated_ts = float(saved.get("last_updated_ts", time.time()))
            self._history = data.get("history", [])
            self._volatility = float(data.get("volatility", 0.0))
            self._mood_valence      = float(data.get("mood_valence", self._mood_valence))
            self._mood_arousal      = float(data.get("mood_arousal", self._mood_arousal))
            self._mood_last_updated = float(data.get("mood_last_updated", time.time()))
            logger.info(f"Emotional state loaded ({len(self.emotions)} emotions)")
        except Exception as e:
            logger.error(f"EmotionalState load failed: {e}")