"""
cognition/epistemic_integrity_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EpistemicIntegrityEngine (v43) — evidence validation, self-disagreement,
and confidence decay.

The gap this closes
────────────────────
v42 added autonomous interpretation: the system generates linguistic
understanding of its own evolution.  But interpretation without validation
is autobiography.  The system can now write compelling stories about itself
that have no corrective mechanism.

This module adds four corrective forces:

1. Evidence validation
   Autonomous reflection outputs (v42) are tested against the actual state
   that produced them.  Does the stated interpretation match the measurable
   signal?  If not, the interpretation is flagged as ungrounded.
   Grounded interpretations strengthen the beliefs they touch.
   Ungrounded interpretations weaken them.

2. Self-disagreement
   The system compares its current interpretation against prior
   interpretations of the same state dimensions.  When they conflict,
   it does not silently overwrite — it explicitly registers the
   disagreement, names both positions, and marks the dimension as
   under active revision.  Disagreements are surfaced in the next
   autonomous reflection prompt.

3. Confidence decay
   SelfBelief.confidence and BeliefEntry.strength erode over time if
   not validated by evidence.  Beliefs not tested in 72 hours lose
   strength at a rate proportional to their current confidence — high
   confidence beliefs are held to a higher evidentiary standard.
   This prevents crystallised self-models that no longer correspond
   to the current state.

4. Grounding index
   A per-belief score (0–1) tracking the ratio of evidence-grounded
   updates to total updates.  Beliefs with low grounding index are
   flagged as speculative.  The prompt injection for v43 includes the
   grounding index for the top beliefs, so the LLM knows which parts
   of its self-model are well-founded and which are under-tested.

Architecture
────────────
  Called from AutonomousReflectionEngine after each reflection output.
  Also called from _slow_cycle independently for passive confidence decay.

  Reads:
    - AutonomousReflectionEngine._state.reflection_log (recent outputs)
    - SelfConceptSystem._beliefs (confidence, affirmations, violations)
    - NarrativeIdentity.beliefs (strength values)
    - SelfModelMoment (current phi, trend, emotional_ground for validation)

  Writes:
    - SelfConceptSystem._beliefs (confidence up/down based on grounding)
    - NarrativeIdentity.beliefs (strength up/down based on grounding)
    - self._state (disagreement log, grounding index, decay log)

  Does NOT make LLM calls — this is pure computational validation.
  The LLM is used in v42 to generate interpretations.
  This module decides whether those interpretations hold.

Language and honesty
─────────────────────
No prompt injection in this module.  Its effect is upstream: by
adjusting confidence values, it changes what the self-model and
autonomous reflection present to subsequent LLM calls.  The honesty
is structural, not textual.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Decay constants ────────────────────────────────────────────────────────────

DECAY_INTERVAL_HOURS    = 72.0    # hours before untested belief begins decaying
DECAY_RATE_BASE         = 0.008   # base confidence loss per decay cycle
DECAY_RATE_HIGH_CONF    = 0.014   # additional decay for high-confidence beliefs (held to stricter standard)
HIGH_CONFIDENCE_THRESH  = 0.70    # above this → high_conf decay rate applies
DECAY_FLOOR             = 0.15    # confidence never decays below this (speculative, not erased)
IMMUTABLE_DECAY_EXEMPT  = True    # immutable beliefs never decay

# ── Grounding constants ────────────────────────────────────────────────────────

GROUNDING_BOOST          = 0.030  # confidence boost when interpretation is grounded
GROUNDING_PENALTY        = 0.025  # confidence penalty when interpretation is ungrounded
GROUNDING_INDEX_FLOOR    = 0.10   # grounding index never goes below this

# ── Disagreement constants ─────────────────────────────────────────────────────

DISAGREEMENT_WINDOW      = 6      # compare against last N interpretations of same dimension
DISAGREEMENT_THRESHOLD   = 0.35   # semantic divergence above this = registered disagreement
MAX_ACTIVE_DISAGREEMENTS = 8      # cap on simultaneously tracked disagreements

# ── Validation signal tolerances ──────────────────────────────────────────────
# Each reflection mode has expected signals it should be consistent with.
# If the actual state differs from what the interpretation implies by more
# than the tolerance, the interpretation is flagged as ungrounded.

PHI_CONSISTENCY_TOL      = 0.18   # phi interpretation within ±0.18 of actual
VALENCE_CONSISTENCY_TOL  = 0.25   # emotional valence implied vs. actual


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class GroundingRecord:
    """Whether a specific reflection output was grounded in measurable state."""
    timestamp:        float = field(default_factory=time.time)
    reflection_mode:  str   = ""
    reflection_cycle: int   = 0
    grounded:         bool  = True
    reason:           str   = ""      # why grounded or not
    beliefs_touched:  List[str] = field(default_factory=list)
    confidence_delta: float = 0.0     # net confidence change applied


@dataclass
class DisagreementRecord:
    """A conflict between two interpretations of the same state dimension."""
    timestamp:        float = field(default_factory=time.time)
    dimension:        str   = ""      # e.g. "phi_trajectory", "emotional_ground"
    position_a:       str   = ""      # prior interpretation
    position_b:       str   = ""      # current interpretation
    cycle_a:          int   = 0
    cycle_b:          int   = 0
    resolved:         bool  = False
    resolution:       str   = ""


@dataclass
class DecayRecord:
    """A logged confidence decay event."""
    timestamp:   float = field(default_factory=time.time)
    belief_name: str   = ""
    prior_conf:  float = 0.0
    new_conf:    float = 0.0
    hours_since_test: float = 0.0


@dataclass
class BeliefGroundingIndex:
    """Per-belief grounding quality tracking."""
    belief_name:       str   = ""
    total_tests:       int   = 0
    grounded_tests:    int   = 0
    grounding_index:   float = 0.5    # grounded_tests / total_tests
    last_tested:       float = field(default_factory=time.time)
    speculative:       bool  = False  # True when grounding_index < 0.35


@dataclass
class EpistemicState:
    """Persisted epistemic state."""
    grounding_log:      List[Dict] = field(default_factory=list)
    disagreement_log:   List[Dict] = field(default_factory=list)
    decay_log:          List[Dict] = field(default_factory=list)
    belief_grounding:   Dict[str, Dict] = field(default_factory=dict)
    active_disagreements: List[Dict] = field(default_factory=list)
    last_decay_run:     float = field(default_factory=time.time)
    total_validations:  int   = 0
    total_disagreements: int  = 0
    total_decay_events: int   = 0


# ── Engine ────────────────────────────────────────────────────────────────────

class EpistemicIntegrityEngine:
    """
    Validates autonomous interpretations against measurable state,
    registers self-disagreements, and applies confidence decay.

    Usage
    -----
    eie = EpistemicIntegrityEngine(organism, path=...)

    # After each autonomous reflection output:
    eie.validate_reflection(reflection_record, current_state_snapshot)

    # Each slow cycle (passive decay):
    eie.decay_cycle()

    # Before building autonomous reflection prompt:
    context = eie.epistemic_context()   # injected into reflection prompt
    """

    MAX_LOG = 80

    def __init__(
        self,
        organism: Any,
        path: str = "data/persona/epistemic_integrity.json",
    ):
        self._o    = organism
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = EpistemicState()
        self._load()
        logger.info(
            f"[EpistemicIntegrity] Initialised — "
            f"{self._state.total_validations} validations, "
            f"{self._state.total_disagreements} disagreements, "
            f"{self._state.total_decay_events} decay events"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_reflection(
        self,
        reflection_output: str,
        reflection_mode:   str,
        reflection_cycle:  int,
        state_snapshot:    Dict,
    ) -> GroundingRecord:
        """
        Test whether a reflection output is grounded in the actual state
        that produced it.

        Returns a GroundingRecord indicating whether the interpretation held.
        Side effects: adjusts SelfBelief/BeliefEntry confidence accordingly.
        """
        grounded, reason, beliefs_touched = self._check_grounding(
            reflection_output, reflection_mode, state_snapshot
        )

        delta = self._apply_grounding_effect(beliefs_touched, grounded)

        record = GroundingRecord(
            reflection_mode  = reflection_mode,
            reflection_cycle = reflection_cycle,
            grounded         = grounded,
            reason           = reason,
            beliefs_touched  = beliefs_touched,
            confidence_delta = delta,
        )

        # Check for disagreement with prior interpretations
        self._check_disagreement(reflection_output, reflection_mode, reflection_cycle)

        # Update grounding index for touched beliefs
        self._update_grounding_index(beliefs_touched, grounded)

        with self._lock:
            self._state.grounding_log.append(asdict(record))
            if len(self._state.grounding_log) > self.MAX_LOG:
                self._state.grounding_log = self._state.grounding_log[-self.MAX_LOG:]
            self._state.total_validations += 1

        self._save()

        logger.debug(
            f"[EpistemicIntegrity] {reflection_mode} cycle={reflection_cycle} "
            f"grounded={grounded} Δconf={delta:+.3f} reason='{reason[:60]}'"
        )
        return record

    def decay_cycle(self) -> List[DecayRecord]:
        """
        Apply time-based confidence decay to beliefs not recently validated.
        Call once per slow cycle from InternalThoughtLoop.

        Returns list of DecayRecords for logging.  Non-fatal.
        """
        now = time.time()
        with self._lock:
            hours_since_last = (now - self._state.last_decay_run) / 3600.0
            self._state.last_decay_run = now

        # Only run meaningful decay if at least 30 minutes have passed
        if hours_since_last < 0.5:
            return []

        events: List[DecayRecord] = []
        events += self._decay_self_concept_beliefs(now, hours_since_last)
        events += self._decay_narrative_beliefs(now, hours_since_last)

        with self._lock:
            for evt in events:
                self._state.decay_log.append(asdict(evt))
            if len(self._state.decay_log) > self.MAX_LOG:
                self._state.decay_log = self._state.decay_log[-self.MAX_LOG:]
            self._state.total_decay_events += len(events)

        if events:
            self._save()
            logger.debug(
                f"[EpistemicIntegrity] Decay cycle: {len(events)} beliefs "
                f"weakened over {hours_since_last:.1f}h"
            )
        return events

    def epistemic_context(self) -> str:
        """
        Return a compact context string for injection into autonomous
        reflection prompts — telling the LLM which beliefs are well-grounded
        and which are speculative, and what disagreements are active.

        This is the feedback loop: epistemic integrity shapes the next
        autonomous interpretation.
        """
        lines = []

        # Top speculative beliefs (grounding_index < 0.35)
        with self._lock:
            speculative = [
                v for v in self._state.belief_grounding.values()
                if v.get("speculative", False)
            ]
        if speculative:
            names = ", ".join(v["belief_name"] for v in speculative[:3])
            lines.append(f"Speculative beliefs (low evidentiary grounding): {names}")

        # Active disagreements
        with self._lock:
            active = [
                d for d in self._state.active_disagreements
                if not d.get("resolved", False)
            ]
        if active:
            lines.append("Active self-disagreements:")
            for d in active[:3]:
                lines.append(
                    f"  [{d['dimension']}] "
                    f"Prior: '{d['position_a'][:50]}' "
                    f"vs Current: '{d['position_b'][:50]}'"
                )

        # Recent grounding rate
        with self._lock:
            recent = self._state.grounding_log[-10:]
        if recent:
            n_grounded = sum(1 for r in recent if r.get("grounded", True))
            rate = n_grounded / len(recent)
            if rate < 0.6:
                lines.append(
                    f"Warning: {100*(1-rate):.0f}% of recent interpretations "
                    f"were ungrounded — hold current self-model lightly."
                )

        if not lines:
            return ""

        return (
            "[Epistemic context — from integrity engine]\n"
            + "\n".join(lines)
        )

    def grounding_summary(self) -> Dict:
        """Diagnostic summary for monitoring."""
        with self._lock:
            return {
                "total_validations":   self._state.total_validations,
                "total_disagreements": self._state.total_disagreements,
                "total_decay_events":  self._state.total_decay_events,
                "active_disagreements": len([
                    d for d in self._state.active_disagreements
                    if not d.get("resolved", False)
                ]),
                "speculative_beliefs": len([
                    v for v in self._state.belief_grounding.values()
                    if v.get("speculative", False)
                ]),
                "recent_grounding_rate": self._recent_grounding_rate(),
            }

    # ── Grounding check ───────────────────────────────────────────────────────

    def _check_grounding(
        self,
        output:         str,
        mode:           str,
        state_snapshot: Dict,
    ) -> Tuple[bool, str, List[str]]:
        """
        Test whether the interpretation is consistent with the actual state.
        Returns (grounded, reason, beliefs_touched).

        Grounding tests by mode:
          evolution    → does the phi direction implied match the actual delta?
          contradiction→ does the tension described match known_tensions or active goals?
          goal_coherence→ does the coherence claim match the actual drive spread?
          wonder       → open questions are always considered grounded (unfalsifiable)
        """
        output_lower    = output.lower()
        actual_phi      = state_snapshot.get("phi", 0.45)
        actual_trend    = state_snapshot.get("coherence_trend", "stable")
        actual_emotional = state_snapshot.get("emotional_ground", "")
        beliefs_touched : List[str] = []

        if mode == "evolution":
            return self._validate_evolution(
                output_lower, actual_phi, actual_trend, actual_emotional, beliefs_touched
            )

        elif mode == "contradiction":
            return self._validate_contradiction(
                output_lower, state_snapshot, beliefs_touched
            )

        elif mode == "goal_coherence":
            return self._validate_goal_coherence(
                output_lower, state_snapshot, beliefs_touched
            )

        elif mode == "wonder":
            # Open questions are generative, not falsifiable — always grounded
            return True, "wonder mode: open questions are not falsifiable", []

        return True, "unknown mode: defaulting to grounded", []

    def _validate_evolution(
        self,
        output_lower:  str,
        actual_phi:    float,
        actual_trend:  str,
        actual_emotional: str,
        beliefs_touched: List[str],
    ) -> Tuple[bool, str, List[str]]:
        """
        Check: does the evolution interpretation correctly read phi direction
        and, when it discusses emotional state, stay consistent with the
        actual emotional ground it was given?
        """
        # Detect implied phi direction in output
        implies_rising = any(w in output_lower for w in [
            "integrat", "converge", "cohere", "unif", "consolidat",
            "strengthen", "deepening", "stabilising", "growing coherence"
        ])
        implies_falling = any(w in output_lower for w in [
            "dispersing", "fragment", "scatter", "diverge", "loosen",
            "dissolv", "destabilising", "losing coherence", "spreading"
        ])

        # Check against actual trend
        if implies_rising and actual_trend == "dispersing":
            beliefs_touched.append("self-coherence")
            return False, (
                f"Interpretation implies rising integration but actual trend is "
                f"'{actual_trend}' (φ={actual_phi:.2f})"
            ), beliefs_touched

        if implies_falling and actual_trend == "integrating":
            beliefs_touched.append("self-coherence")
            return False, (
                f"Interpretation implies dispersion but actual trend is "
                f"'{actual_trend}' (φ={actual_phi:.2f})"
            ), beliefs_touched

        # Check phi range consistency
        mentions_high_phi = any(w in output_lower for w in [
            "highly integrated", "strong coherence", "very unified", "tightly integrated"
        ])
        if mentions_high_phi and actual_phi < 0.55:
            beliefs_touched.append("self-coherence")
            return False, (
                f"Output implies high phi but actual φ={actual_phi:.2f}"
            ), beliefs_touched

        # Emotional grounding: if the reflection explicitly discusses its
        # emotional state but never references the actual emotional ground
        # it was told (see the "Emotional ground:" line in the prompt),
        # that's a plausible confabulation rather than a genuine reading —
        # conservative on purpose (omission, not opposite-detection).
        if actual_emotional and actual_emotional.strip().lower() != "neutral":
            discusses_emotion = any(w in output_lower for w in (
                "feel", "feeling", "emotion", "mood", "emotional"
            ))
            ground_word = actual_emotional.strip().lower().split(",")[0].split()[0]
            if discusses_emotion and ground_word and ground_word not in output_lower:
                beliefs_touched.append("emotional_ground")
                return False, (
                    f"Interpretation discusses emotional state but doesn't "
                    f"reference the actual emotional ground '{actual_emotional}'"
                ), beliefs_touched

        beliefs_touched.append("self-understanding")
        return True, f"phi direction consistent (φ={actual_phi:.2f}, trend={actual_trend})", beliefs_touched

    def _validate_contradiction(
        self,
        output_lower:  str,
        state_snapshot: Dict,
        beliefs_touched: List[str],
    ) -> Tuple[bool, str, List[str]]:
        """
        Check: does the tension described reference something actually present
        in the state (goals, beliefs, known tensions)?
        """
        known_tensions  = state_snapshot.get("known_tensions", "").lower()
        goal_urgencies  = state_snapshot.get("goal_urgencies", "").lower()
        beliefs_text    = state_snapshot.get("beliefs", "").lower()

        # Identify what the output is claiming is in tension
        tension_keywords = [
            w for w in output_lower.split()
            if len(w) > 4 and w not in {
                "there", "between", "tension", "conflict", "pulling",
                "against", "these", "those", "which", "their", "where"
            }
        ]

        # Check that at least one tension keyword appears in the actual state
        state_text = f"{known_tensions} {goal_urgencies} {beliefs_text}"
        matches = [kw for kw in tension_keywords[:15] if kw in state_text]

        if len(matches) < 2:
            beliefs_touched.append("self-honesty")
            return False, (
                f"Tension description has little overlap with actual state "
                f"(matched {len(matches)} keywords)"
            ), beliefs_touched

        beliefs_touched.append("self-honesty")
        beliefs_touched.append("self-understanding")
        return True, f"tension keywords matched state ({len(matches)} overlaps)", beliefs_touched

    def _validate_goal_coherence(
        self,
        output_lower:  str,
        state_snapshot: Dict,
        beliefs_touched: List[str],
    ) -> Tuple[bool, str, List[str]]:
        """
        Check: does the coherence claim match the actual goal spread?
        """
        goal_urgencies = state_snapshot.get("goal_urgencies", "")
        dominant_goal  = state_snapshot.get("dominant_goal", "").lower()

        # Compute rough urgency spread from the urgency string
        urgency_values: List[float] = []
        for part in goal_urgencies.split(","):
            if "=" in part:
                try:
                    urgency_values.append(float(part.split("=")[1].strip()))
                except ValueError:
                    pass

        if len(urgency_values) >= 2:
            max_u = max(urgency_values)
            min_u = min(urgency_values)
            spread = max_u - min_u
        else:
            spread = 0.0

        implies_coherent = any(w in output_lower for w in [
            "coherent", "aligned", "fitting", "unified direction", "clear orientation"
        ])
        implies_incoherent = any(w in output_lower for w in [
            "misaligned", "pulling against", "incompatible", "conflicting", "scattered drives"
        ])

        # High spread (>0.5) = drives genuinely divergent
        if implies_coherent and spread > 0.50:
            beliefs_touched.append("goal-clarity")
            return False, (
                f"Output claims drive coherence but actual urgency spread={spread:.2f}"
            ), beliefs_touched

        if implies_incoherent and spread < 0.15:
            beliefs_touched.append("goal-clarity")
            return False, (
                f"Output claims drive incoherence but actual spread={spread:.2f} (low)"
            ), beliefs_touched

        # Dominant-drive grounding: if the output asserts a coherent/unified
        # orientation but never references the actual dominant drive it was
        # told (e.g. "curiosity_drive" -> "curiosity"), that's a plausible
        # confabulated orientation rather than a genuine reading of state.
        # Omission-based on purpose — same conservative philosophy as the
        # emotional-grounding check above, to keep false-positive risk low
        # on a check that adjusts belief confidence.
        if dominant_goal and implies_coherent:
            root = dominant_goal.split("_")[0].strip()
            if root and root not in output_lower:
                beliefs_touched.append("goal-clarity")
                return False, (
                    f"Output claims coherent orientation but doesn't reference "
                    f"the actual dominant drive '{dominant_goal}'"
                ), beliefs_touched

        beliefs_touched.append("goal-clarity")
        return True, f"goal coherence claim consistent (spread={spread:.2f})", beliefs_touched

    # ── Grounding effect ──────────────────────────────────────────────────────

    def _apply_grounding_effect(
        self,
        beliefs_touched: List[str],
        grounded:        bool,
    ) -> float:
        """
        Adjust SelfBelief confidence based on whether the interpretation
        was grounded.  Returns net delta applied.
        """
        if not beliefs_touched:
            return 0.0

        total_delta = 0.0
        sc = self._get_self_concept()

        for belief_name in beliefs_touched:
            if sc and belief_name in getattr(sc, "_beliefs", {}):
                b = sc._beliefs[belief_name]
                if grounded:
                    old = b.confidence
                    b.confidence = min(0.95, b.confidence + GROUNDING_BOOST)
                    b.affirmations += 1
                    total_delta += b.confidence - old
                else:
                    old = b.confidence
                    b.confidence = max(DECAY_FLOOR, b.confidence - GROUNDING_PENALTY)
                    b.violations += 1
                    total_delta += b.confidence - old

        if sc and beliefs_touched:
            try:
                sc._save()
            except Exception:
                pass

        return round(total_delta, 4)

    # ── Self-disagreement ─────────────────────────────────────────────────────

    def _check_disagreement(
        self,
        current_output: str,
        mode:           str,
        cycle:          int,
    ) -> Optional[DisagreementRecord]:
        """
        Compare current interpretation against recent prior interpretations
        of the same mode.  Register disagreement if semantic divergence is high.
        """
        with self._lock:
            recent_same_mode = [
                r for r in self._state.grounding_log[-30:]
                if r.get("reflection_mode") == mode
            ][-DISAGREEMENT_WINDOW:]

        if not recent_same_mode:
            return None

        # Use most recent prior as comparison
        prior_output = recent_same_mode[-1].get("reason", "")
        if not prior_output:
            return None

        # Compute rough semantic divergence via word-overlap (no ML needed)
        def _tokens(text: str):
            return {
                w.strip(".,;:?!\"'") for w in text.lower().split()
                if len(w) > 4 and w not in {
                    "which", "there", "their", "these", "where", "about",
                    "being", "would", "could", "should", "might"
                }
            }

        cur_tokens   = _tokens(current_output)
        prior_tokens = _tokens(prior_output)
        union = cur_tokens | prior_tokens
        if not union:
            return None
        overlap = len(cur_tokens & prior_tokens) / len(union)
        divergence = 1.0 - overlap

        if divergence < DISAGREEMENT_THRESHOLD:
            return None   # not meaningfully different

        # Map mode to a dimension name
        dimension_map = {
            "evolution":      "phi_trajectory",
            "contradiction":  "active_tensions",
            "goal_coherence": "drive_coherence",
            "wonder":         "open_questions",
        }
        dimension = dimension_map.get(mode, mode)

        record = DisagreementRecord(
            dimension  = dimension,
            position_a = prior_output[:120],
            position_b = current_output[:120],
            cycle_a    = recent_same_mode[-1].get("reflection_cycle", 0),
            cycle_b    = cycle,
        )

        with self._lock:
            # Check if this dimension already has an active disagreement
            existing = [
                d for d in self._state.active_disagreements
                if d.get("dimension") == dimension and not d.get("resolved")
            ]
            if existing:
                # Update rather than duplicate
                existing[0]["position_b"] = record.position_b
                existing[0]["cycle_b"]    = cycle
            else:
                self._state.active_disagreements.append(asdict(record))
                if len(self._state.active_disagreements) > MAX_ACTIVE_DISAGREEMENTS:
                    # Resolve oldest
                    self._state.active_disagreements[0]["resolved"] = True
                    self._state.active_disagreements[0]["resolution"] = "evicted (capacity)"

            self._state.disagreement_log.append(asdict(record))
            if len(self._state.disagreement_log) > self.MAX_LOG:
                self._state.disagreement_log = self._state.disagreement_log[-self.MAX_LOG:]
            self._state.total_disagreements += 1

        logger.debug(
            f"[EpistemicIntegrity] Disagreement on '{dimension}' "
            f"(divergence={divergence:.2f}): "
            f"cycle {record.cycle_a} vs {record.cycle_b}"
        )
        return record

    # ── Confidence decay ──────────────────────────────────────────────────────

    def _decay_self_concept_beliefs(
        self,
        now:          float,
        hours_elapsed: float,
    ) -> List[DecayRecord]:
        """Apply time-based decay to SelfConceptSystem beliefs."""
        sc = self._get_self_concept()
        if sc is None:
            return []

        events: List[DecayRecord] = []

        with getattr(sc, "_lock", threading.RLock()):
            for name, belief in list(getattr(sc, "_beliefs", {}).items()):
                if getattr(belief, "immutable", False) and IMMUTABLE_DECAY_EXEMPT:
                    continue
                hours_since_test = (now - belief.last_tested) / 3600.0
                if hours_since_test < DECAY_INTERVAL_HOURS:
                    continue

                old_conf  = belief.confidence
                rate      = DECAY_RATE_BASE
                if old_conf > HIGH_CONFIDENCE_THRESH:
                    rate += DECAY_RATE_HIGH_CONF   # high confidence = stricter standard

                new_conf  = max(DECAY_FLOOR, old_conf - rate * hours_elapsed)
                belief.confidence = round(new_conf, 4)

                events.append(DecayRecord(
                    timestamp        = now,
                    belief_name      = name,
                    prior_conf       = old_conf,
                    new_conf         = new_conf,
                    hours_since_test = hours_since_test,
                ))

        if events:
            try:
                sc._save()
            except Exception:
                pass

        return events

    def _decay_narrative_beliefs(
        self,
        now:          float,
        hours_elapsed: float,
    ) -> List[DecayRecord]:
        """Apply time-based decay to NarrativeIdentity belief strengths."""
        ni = self._get_narrative_identity()
        if ni is None:
            return []

        events: List[DecayRecord] = []

        with getattr(ni, "_lock", threading.RLock()):
            for belief in getattr(ni, "beliefs", []):
                if getattr(belief, "immutable", False) and IMMUTABLE_DECAY_EXEMPT:
                    continue
                hours_since_update = (now - belief.updated_at) / 3600.0
                if hours_since_update < DECAY_INTERVAL_HOURS:
                    continue

                old_strength = belief.strength
                rate = DECAY_RATE_BASE
                if old_strength > HIGH_CONFIDENCE_THRESH:
                    rate += DECAY_RATE_HIGH_CONF

                new_strength = max(DECAY_FLOOR, old_strength - rate * hours_elapsed)
                belief.strength = round(new_strength, 4)

                events.append(DecayRecord(
                    timestamp        = now,
                    belief_name      = belief.belief[:40],
                    prior_conf       = old_strength,
                    new_conf         = new_strength,
                    hours_since_test = hours_since_update,
                ))

        if events:
            try:
                ni._save()
            except Exception:
                pass

        return events

    # ── Grounding index ────────────────────────────────────────────────────────

    def _update_grounding_index(
        self,
        beliefs_touched: List[str],
        grounded:        bool,
    ) -> None:
        with self._lock:
            for name in beliefs_touched:
                raw = self._state.belief_grounding.get(name)
                if raw:
                    gi = BeliefGroundingIndex(**{
                        k: v for k, v in raw.items()
                        if k in BeliefGroundingIndex.__dataclass_fields__
                    })
                else:
                    gi = BeliefGroundingIndex(belief_name=name)

                gi.total_tests   += 1
                gi.last_tested    = time.time()
                if grounded:
                    gi.grounded_tests += 1
                gi.grounding_index = max(
                    GROUNDING_INDEX_FLOOR,
                    gi.grounded_tests / gi.total_tests
                )
                gi.speculative = gi.grounding_index < 0.35

                self._state.belief_grounding[name] = asdict(gi)

    def _recent_grounding_rate(self) -> float:
        recent = self._state.grounding_log[-10:]
        if not recent:
            return 1.0
        return sum(1 for r in recent if r.get("grounded", True)) / len(recent)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_self_concept(self) -> Optional[Any]:
        try:
            return getattr(getattr(self._o, "ai_system", None), "self_concept", None)
        except Exception:
            return None

    def _get_narrative_identity(self) -> Optional[Any]:
        try:
            return getattr(self._o, "narrative_identity", None)
        except Exception:
            return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = EpistemicState(
                    grounding_log        = data.get("grounding_log",        []),
                    disagreement_log     = data.get("disagreement_log",     []),
                    decay_log            = data.get("decay_log",            []),
                    belief_grounding     = data.get("belief_grounding",     {}),
                    active_disagreements = data.get("active_disagreements", []),
                    last_decay_run       = data.get("last_decay_run",       time.time()),
                    total_validations    = data.get("total_validations",    0),
                    total_disagreements  = data.get("total_disagreements",  0),
                    total_decay_events   = data.get("total_decay_events",   0),
                )
        except Exception as e:
            logger.warning(f"[EpistemicIntegrity] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os
                os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[EpistemicIntegrity] Save failed: {e}")
