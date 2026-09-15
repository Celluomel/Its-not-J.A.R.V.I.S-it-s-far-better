"""
cognition/self_revision_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SelfRevisionEngine — reality reshapes the self.

The architectural gap this closes
──────────────────────────────────
v36 answered: "Who am I right now?"        → SelfModelMoment
v37 answered: "How does who I am shape perception?"  → SelfModelInfluence

The loop was still open:

  self → perception → action → ...
                                ↑ no return path

This module closes it:

  self → perception → action → prediction error → self revision
  ↑___________________________________________________|

The self is now a continuously adapting structure, not a slowly-fading
snapshot updated only by NarrativeIdentity's chapter recording.

What "self revision" means here
────────────────────────────────
Not personality replacement. Not memory erasure.
Targeted, proportional adjustments to the self-model driven by the gap
between what was predicted and what actually happened.

Three revision types:

  1. Phi recalibration
     When sustained high surprise (world consistently unexpected), phi is
     nudged downward — the self-model becomes less certain of its own
     coherence. When surprise drops back, phi is allowed to recover.
     This prevents the phi-dominance loop identified in v37 critique.

  2. Qualia drift
     When the dominant emotion in the response consistently differs from
     the qualia predicted by the self-model, the qualia_tone is gently
     updated toward what was actually expressed. The self learns what it
     actually feels, not what it thinks it should feel.

  3. Value weight revision (via EthicalReasoningEngine)
     When a prediction error is connected to an ethical dimension
     (the response invoked different values than the self-model expected),
     the value hierarchy is nudged. This makes moral development emergent
     from experience, not just from explicit dilemma resolution.

Revision is:
  - Proportional to error magnitude (small errors → tiny nudges)
  - Damped by phi (high phi → slower revision; the self doesn't thrash)
  - Bounded (phi floor 0.20, ceiling 0.95; qualia only shifts when
    consistent evidence accumulates across ≥3 turns)
  - Logged as GrowthEvents in ExperientialLearningEngine when significant

Execution order
────────────────
Called from CognitiveOrganism._post_interaction(), AFTER:
  - PredictiveMind.evaluate() has run (need the PredictionResult)
  - EmotionalState has been updated (need actual emotion post-response)
  - SelfModelMoment has been updated (need current phi to damp revision)

And BEFORE:
  - The next turn's SelfModelInfluence.apply() (so revision affects next influence)

Integration with existing systems
───────────────────────────────────
  SelfRevisionEngine reads from:
    - PredictiveMind._surprise_window  → sustained surprise signal
    - PredictiveMind last PredictionResult → per-turn error
    - EmotionalState                   → actual post-response emotion
    - SelfModelMoment.current          → current phi, qualia, values
    - EthicalReasoningEngine           → value hierarchy update hook

  SelfRevisionEngine writes to:
    - SelfModelMomentManager.current   → phi, qualia_tone, coherence_trend
    - EthicalReasoningEngine           → value weights (via record_resolution)
    - ExperientialLearningEngine       → growth log (when revision is significant)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

PHI_REVISION_RATE        = 0.025   # max phi change per turn from surprise
PHI_RECOVERY_RATE        = 0.015   # phi recovery per turn when surprise low
PHI_FLOOR                = 0.20    # phi cannot be revised below this
PHI_CEILING              = 0.95

QUALIA_ACCUMULATION_TURNS = 3      # consistent evidence before qualia shifts
QUALIA_SHIFT_MAGNITUDE    = 0.12   # how much qualia_tone shifts (see note)

SURPRISE_SUSTAINED_WINDOW = 8      # turns over which sustained surprise is measured
SURPRISE_HIGH_THRESHOLD   = 0.40   # surprise_index above → phi under pressure
SURPRISE_LOW_THRESHOLD    = 0.15   # surprise_index below → phi recovers

ERROR_SIGNIFICANT         = 0.35   # per-turn error above → triggers revision
ERROR_MINOR               = 0.15   # below → no revision this turn

VALUE_REVISION_RATE       = 0.004  # weight shift per significant error turn

# Qualia vocabulary — what the self can drift toward
# Each is a (keyword_in_actual_emotion, qualia_phrase) pair
QUALIA_ANCHORS: List[Tuple[str, str]] = [
    ("warmth",       "a warm, open quality — reaching toward the other"),
    ("curiosity",    "alert curiosity — something draws me forward"),
    ("anxiety",      "a taut, vigilant quality — monitoring and uncertain"),
    ("enthusiasm",   "a rising brightness — energised, leaning in"),
    ("melancholy",   "a quiet weight — reflective, slightly inward"),
    ("satisfaction", "a settled completeness — things fitting together"),
    ("wonder",       "wide-open attention — held by something larger"),
    ("care",         "a soft, attending warmth — oriented toward the other"),
    ("frustration",  "a blocked quality — something not resolving"),
    ("peace",        "stillness — no strong pull in any direction"),
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RevisionEvent:
    """A logged self-revision — for diagnostics and growth logging."""
    timestamp:      float = field(default_factory=time.time)
    revision_type:  str   = ""        # "phi" | "qualia" | "value"
    magnitude:      float = 0.0       # how large was the revision
    trigger:        str   = ""        # what caused it (brief)
    prior_value:    str   = ""        # what changed from
    new_value:      str   = ""        # what changed to
    error_level:    float = 0.0       # prediction error that drove this


@dataclass
class RevisionState:
    """Transient per-session state (not persisted — resets each session)."""
    qualia_accumulator: Dict[str, int]  = field(default_factory=dict)  # qualia_key → consecutive turns
    phi_pressure:       float           = 0.0   # running signed pressure on phi
    recent_errors:      Deque[float]    = field(default_factory=lambda: deque(maxlen=SURPRISE_SUSTAINED_WINDOW))
    revision_log:       List[RevisionEvent] = field(default_factory=list)
    total_revisions:    int             = 0


# ── Engine ────────────────────────────────────────────────────────────────────

class SelfRevisionEngine:
    """
    Closes the self → perception → action → prediction error → self loop.

    The self is now continuously updated by what it predicted vs. what
    actually happened.  Revision is proportional, damped, and bounded —
    it is adaptive updating, not instability.

    Usage
    -----
    sre = SelfRevisionEngine()

    # In _post_interaction(), after predictive_mind.evaluate():
    sre.revise(organism, prediction_result)

    # Diagnostic access:
    sre.state.revision_log[-5:]
    """

    def __init__(self):
        self.state = RevisionState()

    # ── Public API ────────────────────────────────────────────────────────────

    def revise(
        self,
        organism:          Any,
        prediction_result: Any,   # PredictionResult from predictive_mind.evaluate()
    ) -> List[RevisionEvent]:
        """
        Apply self-revision based on prediction error.

        Parameters
        ----------
        organism          : CognitiveOrganism
        prediction_result : PredictionResult | None

        Returns list of RevisionEvents that occurred this turn (may be empty).
        """
        if prediction_result is None:
            # No prediction was made this turn — still check phi recovery
            return self._attempt_phi_recovery(organism)

        error_level  = getattr(prediction_result, "error_level", 0.0)
        is_surprise  = getattr(prediction_result, "surprise", False)
        actual_intent = getattr(prediction_result, "actual_intent", "")

        self.state.recent_errors.append(error_level)

        events: List[RevisionEvent] = []

        # 1. Phi revision (sustained surprise signal)
        phi_event = self._revise_phi(organism, error_level)
        if phi_event:
            events.append(phi_event)

        # 2. Qualia drift (actual emotion vs. predicted qualia)
        qualia_event = self._revise_qualia(organism, actual_intent, error_level)
        if qualia_event:
            events.append(qualia_event)

        # 3. Value weight revision (ethical surprise)
        value_event = self._revise_values(organism, error_level, actual_intent)
        if value_event:
            events.append(value_event)

        # 4. Log to ExperientialLearningEngine if significant
        for evt in events:
            if evt.magnitude > 0.08:
                self._log_growth_event(organism, evt)

        self.state.revision_log.extend(events)
        self.state.revision_log = self.state.revision_log[-60:]
        self.state.total_revisions += len(events)

        if events:
            logger.debug(
                f"[SelfRevisionEngine] {len(events)} revision(s) this turn — "
                f"error={error_level:.2f} surprise={is_surprise} "
                f"types={[e.revision_type for e in events]}"
            )

        return events

    # ── Phi revision ─────────────────────────────────────────────────────────

    def _revise_phi(
        self,
        organism:    Any,
        error_level: float,
    ) -> Optional[RevisionEvent]:
        """
        Revise phi based on sustained surprise.

        Sustained high surprise → phi nudged down (self-model less certain).
        Sustained low surprise  → phi allowed to recover.

        Damped by current phi: a very coherent self revises more slowly.
        """
        smm = getattr(getattr(organism, "self_moment", None), "current", None)
        if smm is None:
            return None

        current_phi = smm.phi
        recent = list(self.state.recent_errors)
        if len(recent) < 3:
            return None   # too early to judge sustained trend

        sustained_surprise = sum(recent) / len(recent)

        # Damping: high phi → slower revision (the coherent self resists change)
        damping = 0.5 + 0.5 * current_phi   # range 0.5–1.0

        if sustained_surprise > SURPRISE_HIGH_THRESHOLD:
            # Phi under pressure
            pressure_magnitude = (sustained_surprise - SURPRISE_HIGH_THRESHOLD) * PHI_REVISION_RATE
            delta = -pressure_magnitude * damping
            new_phi = max(PHI_FLOOR, current_phi + delta)

            if abs(delta) < 0.002:
                return None  # negligible

            self._write_phi(organism, new_phi)
            return RevisionEvent(
                revision_type = "phi",
                magnitude     = abs(delta),
                trigger       = f"sustained_surprise={sustained_surprise:.2f}",
                prior_value   = f"{current_phi:.3f}",
                new_value     = f"{new_phi:.3f}",
                error_level   = error_level,
            )

        elif sustained_surprise < SURPRISE_LOW_THRESHOLD and current_phi < 0.72:
            # Phi recovery — world is fitting predictions, allow integration
            delta = PHI_RECOVERY_RATE * (1.0 - current_phi)   # recovery tapers as phi rises
            new_phi = min(PHI_CEILING, current_phi + delta)

            if abs(delta) < 0.002:
                return None

            self._write_phi(organism, new_phi)
            return RevisionEvent(
                revision_type = "phi",
                magnitude     = abs(delta),
                trigger       = f"low_surprise_recovery={sustained_surprise:.2f}",
                prior_value   = f"{current_phi:.3f}",
                new_value     = f"{new_phi:.3f}",
                error_level   = error_level,
            )

        return None

    def _attempt_phi_recovery(self, organism: Any) -> List[RevisionEvent]:
        """Called when no prediction result exists — still allow phi recovery."""
        evt = self._revise_phi(organism, 0.0)
        return [evt] if evt else []

    def _write_phi(self, organism: Any, new_phi: float) -> None:
        """Write revised phi back to SelfModelMoment."""
        try:
            smm = organism.self_moment.current
            # Direct attribute write — SelfModelMoment is a dataclass
            object.__setattr__(smm, "phi", round(new_phi, 3))
        except Exception as e:
            logger.debug(f"[SelfRevisionEngine] phi write failed: {e}")

    # ── Qualia drift ──────────────────────────────────────────────────────────

    def _revise_qualia(
        self,
        organism:      Any,
        actual_intent: str,
        error_level:   float,
    ) -> Optional[RevisionEvent]:
        """
        When what was actually expressed consistently differs from the current
        qualia_tone, drift the qualia toward what the self actually produced.

        Requires QUALIA_ACCUMULATION_TURNS of consistent evidence.
        """
        if error_level < ERROR_MINOR:
            # Low error — qualia is appropriate, no drift
            self.state.qualia_accumulator.clear()
            return None

        # Read actual dominant emotion post-response
        actual_qualia_key = self._detect_qualia_from_intent(actual_intent, organism)
        if not actual_qualia_key:
            return None

        smm = getattr(getattr(organism, "self_moment", None), "current", None)
        if smm is None:
            return None

        # Accumulate evidence
        acc = self.state.qualia_accumulator
        for key in list(acc.keys()):
            if key != actual_qualia_key:
                acc[key] = max(0, acc[key] - 1)
        acc[actual_qualia_key] = acc.get(actual_qualia_key, 0) + 1

        if acc[actual_qualia_key] < QUALIA_ACCUMULATION_TURNS:
            return None   # not enough consistent evidence yet

        # Find the new qualia phrase
        new_qualia = None
        for keyword, phrase in QUALIA_ANCHORS:
            if keyword == actual_qualia_key:
                new_qualia = phrase
                break

        if not new_qualia or new_qualia == smm.qualia_tone:
            return None

        prior_qualia = smm.qualia_tone
        try:
            object.__setattr__(smm, "qualia_tone", new_qualia)
        except Exception as e:
            logger.debug(f"[SelfRevisionEngine] qualia write failed: {e}")
            return None

        # Reset accumulator after shift
        self.state.qualia_accumulator.clear()

        return RevisionEvent(
            revision_type = "qualia",
            magnitude     = QUALIA_SHIFT_MAGNITUDE,
            trigger       = f"actual_qualia='{actual_qualia_key}' × {QUALIA_ACCUMULATION_TURNS} turns",
            prior_value   = prior_qualia[:60],
            new_value     = new_qualia[:60],
            error_level   = error_level,
        )

    def _detect_qualia_from_intent(self, actual_intent: str, organism: Any) -> Optional[str]:
        """
        Infer the dominant qualia keyword from actual intent + current emotion.
        Returns the first matching keyword from QUALIA_ANCHORS, or None.
        """
        # Prefer reading directly from emotional state
        try:
            emo_vals = organism._get_emotion_values()
            if emo_vals:
                dominant_emo = max(emo_vals, key=emo_vals.get)
                dominant_val = emo_vals[dominant_emo]
                if dominant_val > 0.50:
                    for keyword, _ in QUALIA_ANCHORS:
                        if keyword in dominant_emo.lower():
                            return keyword
        except Exception:
            pass

        # Fallback: intent text matching
        intent_lower = (actual_intent or "").lower()
        for keyword, _ in QUALIA_ANCHORS:
            if keyword in intent_lower:
                return keyword

        return None

    # ── Value revision ────────────────────────────────────────────────────────

    def _revise_values(
        self,
        organism:      Any,
        error_level:   float,
        actual_intent: str,
    ) -> Optional[RevisionEvent]:
        """
        When prediction error is significant AND the actual response invoked
        ethical or relational content the self-model didn't predict, nudge
        the value hierarchy toward what was actually expressed.

        This makes moral development emergent from experience.
        """
        if error_level < ERROR_SIGNIFICANT:
            return None

        ethical_engine = getattr(organism, "ethical_engine", None)
        if ethical_engine is None or not hasattr(ethical_engine, "record_resolution"):
            return None

        # Detect which values the actual response drew on
        actual_lower = (actual_intent or "").lower()
        values_expressed = []

        VALUE_SIGNALS = {
            "honesty":        ["honest", "truth", "transparent", "clear"],
            "care":           ["care", "support", "help", "there for"],
            "curiosity":      ["curious", "wonder", "explore", "interesting"],
            "dignity":        ["respect", "dignity", "worth", "value"],
            "non_maleficence":["harm", "careful", "risk", "safe"],
            "autonomy":       ["choice", "your decision", "up to you", "freedom"],
            "beneficence":    ["good", "benefit", "wellbeing", "flourish"],
        }

        for value, signals in VALUE_SIGNALS.items():
            if any(s in actual_lower for s in signals):
                values_expressed.append(value)

        if not values_expressed:
            return None

        # Nudge those values upward slightly — they were invoked under pressure
        for v in values_expressed[:2]:   # at most 2 per turn
            try:
                ethical_engine.record_resolution(
                    topic       = f"error_revision_{v}",
                    stance      = f"expressed {v} under prediction error",
                    values_used = [v],
                    confidence  = error_level * 0.6,   # proportional to error
                )
            except Exception as e:
                logger.debug(f"[SelfRevisionEngine] value revision failed: {e}")

        return RevisionEvent(
            revision_type = "value",
            magnitude     = VALUE_REVISION_RATE * len(values_expressed),
            trigger       = f"prediction_error={error_level:.2f}",
            prior_value   = "existing weights",
            new_value     = f"nudged: {', '.join(values_expressed[:2])}",
            error_level   = error_level,
        )

    # ── Growth logging ────────────────────────────────────────────────────────

    def _log_growth_event(self, organism: Any, event: RevisionEvent) -> None:
        """Forward significant revision events to ExperientialLearningEngine."""
        try:
            ele = getattr(organism, "experiential_learning", None)
            if ele and hasattr(ele, "integrate"):
                # Synthesise a minimal exchange description
                description = (
                    f"Self-revision ({event.revision_type}): "
                    f"{event.prior_value} → {event.new_value} "
                    f"[trigger: {event.trigger}]"
                )
                ele.integrate(
                    user_input  = event.trigger,
                    ai_response = description,
                    context     = {
                        "revision_type": event.revision_type,
                        "magnitude":     event.magnitude,
                    },
                )
        except Exception as e:
            logger.debug(f"[SelfRevisionEngine] growth log failed: {e}")
