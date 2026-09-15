"""
cognition/cross_layer_feedback.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CrossLayerFeedback (v57) — closes the upward causal loops between all
layers of the cognitive stack.

The gap this closes
────────────────────
v40–v56 built a layered stack where information flows mostly downward:

    Resources → BehaviorGate → MotivationalField → IdentityConstraint
    → PredictiveModel → TemporalProjection → AspirationalSynthesis

Each layer constrains the layers above it. But almost nothing flows
upward. The layers produce outputs that are not fed back into the
operating parameters of the layers below them.

v57 adds four upward feedback channels:

Channel 1: TemporalProjection → ResourceEconomy
────────────────────────────────────────────────
When TemporalProjection selects a preferred path (e.g. "intensive"),
it implies a planned expenditure schedule. ResourceEconomy should begin
building reserves for the upcoming intensive period by slightly boosting
its recovery multiplier now, before the path begins.

Signal:    preferred_path from TemporalProjection
Write:     _recovery_multiplier on ResourceEconomy
Logic:
    intensive   → multiplier 1.30 (build reserves faster)
    distributed → multiplier 1.10 (slight boost)
    emergent    → multiplier 0.90 (slow recovery, low cost path)
    none        → multiplier 1.00 (baseline)

This makes ResourceEconomy forward-looking rather than reactive.
Currently it recovers at a fixed rate regardless of what is planned.

Channel 2: WorldSelfDynamicsModel → MotivationalField
───────────────────────────────────────────────────────
When WorldSelfDynamicsModel consistently records that a specific
action type in a specific relational context produces world-side gains
(trust increase, skill growth), it should inform MotivationalField to
amplify the corresponding drive — not because the drive is intrinsically
stronger, but because the context makes acting on it productive.

Signal:    recent DynamicsRecords with world_delta[trust] > 0.02
Write:     contextual_boosts dict on MotivationalField
Logic:
    For each action_type with mean trust_delta > TRUST_GAIN_THRESHOLD
    and > MIN_RECORDS_FOR_FEEDBACK records in recent window:
        map action_type → drive dimension
        boost that drive by WORLD_FEEDBACK_BOOST (0.08, decays each cycle)

The boost is temporary and decays — it reflects present-context advantage,
not a permanent recalibration.

Channel 3: IdentityConstraint → DecisionPolicy
────────────────────────────────────────────────
When IdentityConstraintEngine fires (a draft fails a value check),
it is observational evidence that the corresponding value needs to be
more salient in the current context. DecisionPolicy should receive a
positive update for that value — not because the draft failed (that's
bad), but because the constraint caught a failure that the policy's
existing weight didn't prevent strongly enough.

Signal:    recent entries in IdentityConstraint._log where outcome=corrected
Write:     DecisionPolicy.update(value, "positive", strength=0.3)
Logic:
    For each value that triggered a correction in the last FEEDBACK_WINDOW
    slow cycles:
        call policy.update(value, "positive", strength=CONSTRAINT_FEEDBACK_STRENGTH)
    This tightens the policy weight so future response generation is more
    likely to produce compliant drafts in the first place.

    For fallbacks (constraint fired but correction didn't improve score):
        call policy.update(value, "negative", strength=0.1)
    A repeated fallback means the LLM reliably ignores this value —
    the policy weight shouldn't increase since it isn't helping.

Channel 4: PredictiveConsequenceModel → BehaviorGate
──────────────────────────────────────────────────────
When PCM's prediction confidence is low for the current action type
(fewer than MIN_SIMILAR_FOR_PREDICT similar records), it means Lumina
is in novel territory. The gate should be more conservative in novel
situations — not by hardcoding extra constraints, but by raising the
effective P_FLOOR used in _compute_p() so the gate is slightly more
permissive about reflection (consolidating new knowledge) and slightly
less permissive about experiments (testing hypotheses in unfamiliar
territory).

Signal:    PCM.summary()["ready"] and recent action_type confidence
Write:     _confidence_floor on BehaviorGate (adjusts _compute_p)
Logic:
    if pcm_ready and action_type_confidence > 0.60:
        confidence_floor = P_FLOOR           # 0.05, baseline
    elif pcm_ready and action_type_confidence > 0.35:
        confidence_floor = P_FLOOR * 1.5     # 0.075, slightly conservative
    else:
        confidence_floor = P_FLOOR * 2.0     # 0.10, novel territory — more cautious

Implementation approach
────────────────────────
CrossLayerFeedback runs every FEEDBACK_EVERY_N slow cycles. It reads
from all layer outputs and writes to layer operating parameters. It does
not modify any persistent cognitive state — it only modifies runtime
operating parameters. The modifications decay back to baseline if no
new signal arrives (except Channel 3 which uses DecisionPolicy's
existing slow EMA update mechanism).

All feedback writes are logged and persisted.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
FEEDBACK_EVERY_N      = 8    # slow cycles between feedback passes (~16 min)

# ── Channel 1: TemporalProjection → ResourceEconomy ──────────────────────────
RECOVERY_MULTIPLIERS = {
    "intensive":   1.30,
    "distributed": 1.10,
    "emergent":    0.90,
    None:          1.00,
}

# ── Channel 2: WorldSelfDynamicsModel → MotivationalField ────────────────────
TRUST_GAIN_THRESHOLD     = 0.015   # mean trust_delta above this triggers boost
MIN_RECORDS_FOR_FEEDBACK = 5       # minimum records in recent window
WORLD_FEEDBACK_BOOST     = 0.08    # added to drive weight (temporary)
FEEDBACK_WINDOW          = 30      # slow cycles to look back for records
DRIVE_DECAY_RATE         = 0.15    # fraction of boost lost each feedback cycle

ACTION_TO_DRIVE = {
    "philosophical":  "epistemic",
    "deep_reasoning": "epistemic",
    "analytical":     "epistemic",
    "social":         "social",
    "emotional":      "social",
    "creative":       "expression",
    "technical":      "purpose",
}

# ── Channel 3: IdentityConstraint → DecisionPolicy ───────────────────────────
CONSTRAINT_FEEDBACK_STRENGTH = 0.30   # policy update strength on correction
FALLBACK_FEEDBACK_STRENGTH   = 0.10   # policy update strength on fallback

# ── Channel 4: PCM → BehaviorGate ────────────────────────────────────────────
CONFIDENCE_FLOOR_NOVEL       = 0.10   # P_FLOOR when PCM has no data
CONFIDENCE_FLOOR_PARTIAL     = 0.075  # P_FLOOR when PCM has partial data
CONFIDENCE_FLOOR_BASELINE    = 0.05   # P_FLOOR when PCM is confident

SAVE_PATH = "data/persona/cross_layer_feedback.json"
MAX_LOG   = 200


@dataclass
class FeedbackEvent:
    """One recorded cross-layer feedback write."""
    cycle:     int
    channel:   str      # "tp_economy"|"wsdm_motivation"|"ic_policy"|"pcm_gate"
    signal:    str      # what was read
    write:     str      # what was written
    magnitude: float    # size of the change
    timestamp: float    = field(default_factory=time.time)


class CrossLayerFeedback:
    """
    Runs all four upward feedback channels each FEEDBACK_EVERY_N slow cycles.
    Reads layer outputs and writes to operating parameters of lower layers.

    Usage:
        clf = CrossLayerFeedback(organism, ai_system)
        clf.tick(slow_cycle_count)
    """

    def __init__(
        self,
        organism:  Any,
        ai_system: Any,
        path:      str = SAVE_PATH,
    ) -> None:
        self._organism = organism
        self._ai       = ai_system
        self._path     = Path(path)
        self._lock     = threading.Lock()
        self._log:     List[Dict] = []
        # Contextual drive boosts (decayed each feedback cycle)
        self._drive_boosts: Dict[str, float] = {}
        self._load()
        logger.info(
            f"[CrossLayerFeedback] Initialised — "
            f"{len(self._log)} past feedback events"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Run all feedback channels. Called from InternalThoughtLoop."""
        if slow_cycle % FEEDBACK_EVERY_N != 0 or slow_cycle == 0:
            return
        threading.Thread(
            target=self._run_all,
            args=(slow_cycle,),
            daemon=True,
            name="clf-feedback",
        ).start()

    def status(self) -> Dict:
        return {
            "drive_boosts":       dict(self._drive_boosts),
            "feedback_events":    len(self._log),
            "last_events":        self._log[-3:] if self._log else [],
        }

    # ── Main pass ─────────────────────────────────────────────────────────────

    def _run_all(self, slow_cycle: int) -> None:
        events: List[FeedbackEvent] = []
        try:
            events += self._channel_1_tp_economy(slow_cycle)
            events += self._channel_2_wsdm_motivation(slow_cycle)
            events += self._channel_3_ic_policy(slow_cycle)
            events += self._channel_4_pcm_gate(slow_cycle)

            if events:
                with self._lock:
                    for ev in events:
                        from dataclasses import asdict
                        self._log.append(asdict(ev))
                    if len(self._log) > MAX_LOG:
                        self._log = self._log[-MAX_LOG:]
                logger.info(
                    f"[CrossLayerFeedback] {len(events)} feedback events: "
                    + ", ".join(f"{e.channel}({e.write[:30]})" for e in events)
                )
            self._save()

        except Exception as e:
            logger.debug(f"[CrossLayerFeedback] _run_all error: {e}")

    # ── Channel 1: TemporalProjection → ResourceEconomy ──────────────────────

    def _channel_1_tp_economy(self, slow_cycle: int) -> List[FeedbackEvent]:
        """Set ResourceEconomy recovery multiplier based on planned path intensity."""
        events = []
        try:
            loop = getattr(self._organism, '_loop', None)
            tp   = getattr(loop, '_temporal_projection', None) if loop else None
            ec   = getattr(loop, '_resource_economy', None) if loop else None
            if not tp or not ec:
                return events

            # Read preferred path across all active projections
            projections = tp.active_projections()
            if not projections:
                # No active projections — reset to baseline
                ec._recovery_multiplier = 1.00
                return events

            # Use the projection with highest aspiration tension
            asp = getattr(self._organism, 'aspirational_self', None)
            preferred_path = None
            best_tension   = 0.0
            for domain, result in projections.items():
                tension = 0.0
                if asp:
                    a = asp.aspirations.get(domain)
                    tension = getattr(a, 'tension', 0.0) if a else 0.0
                if tension > best_tension:
                    best_tension   = tension
                    preferred_path = result.preferred_path

            multiplier = RECOVERY_MULTIPLIERS.get(preferred_path, 1.00)

            # Write to ResourceEconomy
            if not hasattr(ec, '_recovery_multiplier'):
                ec._recovery_multiplier = 1.00
            old_mult = ec._recovery_multiplier
            ec._recovery_multiplier = multiplier

            if abs(multiplier - old_mult) > 0.05:
                events.append(FeedbackEvent(
                    cycle     = slow_cycle,
                    channel   = "tp_economy",
                    signal    = f"preferred_path={preferred_path}",
                    write     = f"recovery_multiplier={multiplier:.2f}",
                    magnitude = abs(multiplier - old_mult),
                ))

        except Exception as e:
            logger.debug(f"[CrossLayerFeedback] ch1 error: {e}")
        return events

    # ── Channel 2: WorldSelfDynamicsModel → MotivationalField ────────────────

    def _channel_2_wsdm_motivation(self, slow_cycle: int) -> List[FeedbackEvent]:
        """Boost motivational drives when world-side gains are observed."""
        events = []
        try:
            loop = getattr(self._organism, '_loop', None)
            wsdm = getattr(loop, '_world_self_dynamics', None) if loop else None
            mf   = getattr(loop, '_motivational_field', None) if loop else None
            if not wsdm or not mf:
                return events

            # Decay existing boosts first
            for drive in list(self._drive_boosts.keys()):
                self._drive_boosts[drive] *= (1.0 - DRIVE_DECAY_RATE)
                if self._drive_boosts[drive] < 0.005:
                    del self._drive_boosts[drive]

            # Look at recent records for trust gains by action type
            cutoff_n = len(wsdm._records) - FEEDBACK_WINDOW * 3
            recent   = wsdm._records[max(0, cutoff_n):]

            # Group trust deltas by action type
            trust_by_action: Dict[str, List[float]] = {}
            for rec in recent:
                if rec.world_delta and len(rec.world_delta) >= 1:
                    at = rec.action_type
                    trust_by_action.setdefault(at, []).append(rec.world_delta[0])

            for action_type, trust_deltas in trust_by_action.items():
                if len(trust_deltas) < MIN_RECORDS_FOR_FEEDBACK:
                    continue
                mean_trust = sum(trust_deltas) / len(trust_deltas)
                if mean_trust < TRUST_GAIN_THRESHOLD:
                    continue

                drive = ACTION_TO_DRIVE.get(action_type)
                if not drive:
                    continue

                # Apply boost to MotivationalField drive_vector
                current = mf.drive_vector.get(drive, 0.5)
                boost   = WORLD_FEEDBACK_BOOST * (mean_trust / TRUST_GAIN_THRESHOLD)
                boosted = min(1.0, current + boost)
                mf.drive_vector[drive] = round(boosted, 4)
                self._drive_boosts[drive] = boost

                events.append(FeedbackEvent(
                    cycle     = slow_cycle,
                    channel   = "wsdm_motivation",
                    signal    = f"{action_type} mean_trust_Δ={mean_trust:+.3f}",
                    write     = f"drive[{drive}] {current:.3f}→{boosted:.3f}",
                    magnitude = boost,
                ))

        except Exception as e:
            logger.debug(f"[CrossLayerFeedback] ch2 error: {e}")
        return events

    # ── Channel 3: IdentityConstraint → DecisionPolicy ───────────────────────

    def _channel_3_ic_policy(self, slow_cycle: int) -> List[FeedbackEvent]:
        """Update DecisionPolicy weights based on IdentityConstraint outcomes."""
        events = []
        try:
            loop = getattr(self._organism, '_loop', None)
            ice  = getattr(self._organism, '_identity_constraint', None)
            if not ice:
                # Also try on the ai_system
                ice = getattr(self._organism, 'ai_system', None)
                ice = getattr(ice, '_identity_constraint', None) if ice else None
            dp = getattr(self._ai, '_decision_policy', None)
            if not ice or not dp:
                return events

            # Look at recent constraint checks
            cutoff = slow_cycle - FEEDBACK_EVERY_N
            recent_checks = [
                c for c in ice._log
                if c.get("cycle", 0) >= cutoff or
                   (time.time() - c.get("timestamp", 0)) < FEEDBACK_EVERY_N * 120
            ]

            if not recent_checks:
                return events

            # Corrections: value needed to be enforced — tighten the policy
            for check in recent_checks:
                value   = check.get("value", "")
                outcome = check.get("outcome", "")
                if not value:
                    continue

                if outcome == "corrected":
                    # Constraint caught a violation — policy should weight more
                    dp.update(value, "positive", strength=CONSTRAINT_FEEDBACK_STRENGTH)
                    events.append(FeedbackEvent(
                        cycle     = slow_cycle,
                        channel   = "ic_policy",
                        signal    = f"constraint corrected: {value}",
                        write     = f"policy.update({value}, positive, {CONSTRAINT_FEEDBACK_STRENGTH})",
                        magnitude = CONSTRAINT_FEEDBACK_STRENGTH,
                    ))

                elif outcome == "fallback":
                    # Correction failed — LLM ignores this value, don't inflate weight
                    dp.update(value, "negative", strength=FALLBACK_FEEDBACK_STRENGTH)
                    events.append(FeedbackEvent(
                        cycle     = slow_cycle,
                        channel   = "ic_policy",
                        signal    = f"constraint fallback: {value}",
                        write     = f"policy.update({value}, negative, {FALLBACK_FEEDBACK_STRENGTH})",
                        magnitude = FALLBACK_FEEDBACK_STRENGTH,
                    ))

        except Exception as e:
            logger.debug(f"[CrossLayerFeedback] ch3 error: {e}")
        return events

    # ── Channel 4: PCM confidence → BehaviorGate ─────────────────────────────

    def _channel_4_pcm_gate(self, slow_cycle: int) -> List[FeedbackEvent]:
        """Set BehaviorGate confidence floor based on PCM prediction quality."""
        events = []
        try:
            loop = getattr(self._organism, '_loop', None)
            pcm  = getattr(loop, '_consequence_model', None) if loop else None
            gate = getattr(self._organism, 'behavior_gate', None)
            if not pcm or not gate:
                return events

            summary = pcm.summary()
            ready   = summary.get("ready", False)

            if not ready:
                new_floor = CONFIDENCE_FLOOR_NOVEL
            else:
                # Check recent prediction confidence
                by_type  = summary.get("by_type", {})
                total    = sum(by_type.values())
                # Coverage: fraction of action types with ≥ MIN_SIMILAR records
                from cognition.predictive_consequence_model import MIN_SIMILAR_FOR_PREDICT
                covered  = sum(
                    1 for c in by_type.values()
                    if c >= MIN_SIMILAR_FOR_PREDICT
                )
                n_types  = len(by_type)
                coverage = covered / n_types if n_types else 0.0

                if coverage >= 0.7:
                    new_floor = CONFIDENCE_FLOOR_BASELINE
                elif coverage >= 0.4:
                    new_floor = CONFIDENCE_FLOOR_PARTIAL
                else:
                    new_floor = CONFIDENCE_FLOOR_NOVEL

            # Write to gate's confidence floor
            old_floor = getattr(gate, '_confidence_floor', None)
            gate._confidence_floor = new_floor

            if old_floor != new_floor:
                events.append(FeedbackEvent(
                    cycle     = slow_cycle,
                    channel   = "pcm_gate",
                    signal    = f"pcm_ready={ready}",
                    write     = f"confidence_floor={new_floor:.3f}",
                    magnitude = abs((old_floor or 0.05) - new_floor),
                ))

        except Exception as e:
            logger.debug(f"[CrossLayerFeedback] ch4 error: {e}")
        return events

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "log":          self._log[-MAX_LOG:],
                    "drive_boosts": self._drive_boosts,
                    "_meta":        {"version": "v57", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[CrossLayerFeedback] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self._log         = data.get("log", [])
                self._drive_boosts = data.get("drive_boosts", {})
        except Exception as e:
            logger.warning(f"[CrossLayerFeedback] Load failed: {e}")
