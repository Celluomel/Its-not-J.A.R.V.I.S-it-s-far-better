"""
cognition/cognitive_audit_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CognitiveAuditEngine (v48) — self-directed development.

The gap this closes
────────────────────
v47 (AutonomousExperimentationEngine) tests behavioural hypotheses:
  "shorter responses → more engagement"
  → run 8 interactions → measure → confirm/disconfirm

That's learning from interactions.  What v47 cannot do is look at the
cognitive architecture itself and ask:

  "The experiment confirmation rate has been 0.18 for three audits.
   That means hypotheses are failing at twice the expected rate.
   Why?  Possibly experiments run too infrequently — stale open
   questions produce poor hypotheses.  Reduce GENERATE_EVERY_N from 25
   to 18, trial it for 60 slow cycles, measure whether confirmation
   rate recovers."

That is self-directed development: observe self → find weakness →
propose change → experiment → measure → retain.

Architecture
────────────
  CognitiveAuditEngine reads five performance signals every AUDIT_EVERY_N
  slow cycles (~80 min):

    1. epistemic_grounding_rate  — ratio of evidence-grounded belief updates
       Source: AutonomousReflectionEngine._epistemic._recent_grounding_rate()
       Target: ≥ 0.60.  Low → beliefs form faster than evidence can support.

    2. experiment_confirmation_rate — confirmed / total experiments run
       Source: AutonomousExperimentationEngine._state
       Target: ≥ 0.40.  Low → hypotheses are poorly chosen or tests are
                               too short to resolve.

    3. question_resolution_rate — resolved / (resolved + unresolvable)
       Source: ResolutionEngine._state
       Target: ≥ 0.50.  Low → questions accumulate without closure.

    4. mean_skill_depth — average depth score across known skills
       Source: SkillRegistry._skills
       Target: ≥ 1.0.  Low → competencies stay surface-level.

    5. belief_confidence_mean — mean confidence across SelfBelief entries
       Source: SelfConceptSystem._beliefs
       Target: ≥ 0.50.  Low → self-model is structurally uncertain.

  When a signal falls below its target for TWO consecutive audits
  (not just one — transient dips are expected), the engine flags it as
  a confirmed underperformance and generates a DevelopmentProposal.

Proposal generation
────────────────────
  An LLM call (temperature 0.50, deliberate) receives:
    - the underperforming signal name, current value, and target
    - the full current cognitive_params dict
    - the bounded change space for each parameter
  It returns a JSON proposal: which parameter to change, from what
  value to what value, and why.

  The proposed change is applied immediately to the live module globals
  (no restart required) and written to data/persona/cognitive_params.json
  under "trial".

Trial evaluation
────────────────
  After TRIAL_DURATION_CYCLES slow cycles the engine re-reads the
  underperforming signal:
    - Improved by ≥ CONFIRM_THRESHOLD → proposal committed permanently.
      The parameter value is written to cognitive_params.json "params"
      and the module global is left at the new value.
    - Degraded by ≥ DISCONFIRM_THRESHOLD → proposal reverted.
      Module global is reset to prior value.
    - Otherwise → inconclusive; reverted (conservative default).

  All trials, outcomes, and committed changes are logged.

Cognitive parameter space
─────────────────────────
  Only six parameters are currently tunable.  Each has hard bounds to
  prevent runaway self-modification:

    resolve_every_n        int   [5, 50]    default 15
    generate_every_n       int   [10, 60]   default 25
    decay_rate_base        float [0.002, 0.025]  default 0.008
    grounding_boost        float [0.010, 0.080]  default 0.030
    curiosity_halflife_hrs float [12.0, 96.0]    default 48.0
    reflection_every_n     int   [3, 15]    default 6

  These map to module-level constants that the audit engine writes at
  runtime via Python module-namespace mutation — no restart required.

Safety constraints
──────────────────
  - MAX_PARAM_CHANGE_PCT = 0.30 (30% max per trial, in either direction)
  - ONE trial at a time — a new proposal cannot start while one is active
  - Parameters revert on inconclusive outcomes (conservative default)
  - Full audit history persisted to cognitive_params.json
  - All changes are human-readable and auditable
  - LLM priority = 3, skip_if_busy — audit never blocks interaction

What PandoraBOX can truthfully say about v48
─────────────────────────────────────────
  "My cognitive audit identified underperformance in [signal], proposed
   a calibration change to [parameter], tested it for [N] slow cycles,
   and the [confirmed/reverted] outcome [was committed/was discarded]."

  That is a precise, accurate description of what happened.  Nothing
  more is claimed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cognition.cognitive_organism import CognitiveOrganism

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
AUDIT_EVERY_N           = 40   # slow cycles between audits (~80 min)
TRIAL_DURATION_CYCLES   = 60   # slow cycles a trial runs before evaluation (~2 hrs)

# ── Thresholds for outcome classification ────────────────────────────────────
CONFIRM_THRESHOLD       = 0.10  # signal must improve by ≥10% of target to confirm
DISCONFIRM_THRESHOLD    = 0.05  # signal degraded by ≥5% of target to disconfirm

# ── Signal targets (the values we are aiming for) ─────────────────────────────
SIGNAL_TARGETS = {
    "epistemic_grounding_rate":     0.60,
    "experiment_confirmation_rate": 0.40,
    "question_resolution_rate":     0.50,
    "mean_skill_depth":             1.00,
    "belief_confidence_mean":       0.50,
}

# ── Cognitive parameter space (name → (default, min, max, type)) ──────────────
PARAM_SPACE: Dict[str, Tuple[Any, Any, Any, type]] = {
    "resolve_every_n":        (15,   5,    50,   int),
    "generate_every_n":       (25,   10,   60,   int),
    "decay_rate_base":        (0.008, 0.002, 0.025, float),
    "grounding_boost":        (0.030, 0.010, 0.080, float),
    "curiosity_halflife_hrs": (48.0,  12.0,  96.0,  float),
    "reflection_every_n":     (6,    3,    15,   int),
}

# ── Module paths for live global mutation ────────────────────────────────────
# Maps param name → (module_path, global_name)
PARAM_MODULE_MAP: Dict[str, Tuple[str, str]] = {
    "resolve_every_n":        ("cognition.resolution_engine",            "RESOLVE_EVERY_N"),
    "generate_every_n":       ("cognition.autonomous_experimentation",   "GENERATE_EVERY_N"),
    "decay_rate_base":        ("cognition.epistemic_integrity_engine",   "DECAY_RATE_BASE"),
    "grounding_boost":        ("cognition.epistemic_integrity_engine",   "GROUNDING_BOOST"),
    "curiosity_halflife_hrs": ("cognition.curiosity_engine",             "CURIOSITY_HALFLIFE_HOURS"),
    "reflection_every_n":     ("core.internal_loop",                     "REFLECTION_EVERY_N"),
}

MAX_AUDIT_LOG   = 50
MAX_TRIAL_LOG   = 30


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SignalSnapshot:
    """One audit's worth of performance readings."""
    slow_cycle:                    int
    timestamp:                     float
    epistemic_grounding_rate:      float = 0.0
    experiment_confirmation_rate:  float = 0.0
    question_resolution_rate:      float = 0.0
    mean_skill_depth:              float = 0.0
    belief_confidence_mean:        float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "epistemic_grounding_rate":     self.epistemic_grounding_rate,
            "experiment_confirmation_rate": self.experiment_confirmation_rate,
            "question_resolution_rate":     self.question_resolution_rate,
            "mean_skill_depth":             self.mean_skill_depth,
            "belief_confidence_mean":       self.belief_confidence_mean,
        }

    def deficit(self) -> Dict[str, float]:
        """Returns gap between signal and target (negative = underperforming)."""
        d = {}
        for name, target in SIGNAL_TARGETS.items():
            current = getattr(self, name, 0.0)
            d[name] = current - target
        return d

    def worst_signal(self) -> Optional[str]:
        """Name of the most underperforming signal, or None if all on target."""
        gaps = self.deficit()
        worst = min(gaps, key=gaps.get)
        if gaps[worst] < 0:
            return worst
        return None


@dataclass
class DevelopmentProposal:
    """One proposed cognitive parameter change, with trial tracking."""
    dimension:           str    # underperforming signal name
    signal_at_proposal:  float  # signal value when proposal was made
    signal_target:       float  # target value for that signal
    parameter:           str    # which param to change
    prior_value:         Any    # value before change
    proposed_value:      Any    # new value to trial
    rationale:           str    # LLM-generated explanation
    proposed_at_cycle:   int
    proposed_at_time:    float  = field(default_factory=time.time)
    outcome:             str    = "pending"   # "confirmed"|"disconfirmed"|"inconclusive"|"pending"
    signal_at_eval:      float  = 0.0
    evaluated_at_cycle:  int    = 0
    committed:           bool   = False


@dataclass
class AuditState:
    """Persisted state for the CognitiveAuditEngine."""
    last_audit_cycle:    int               = 0
    snapshot_log:        List[Dict]        = field(default_factory=list)
    active_trial:        Optional[Dict]    = None   # serialised DevelopmentProposal
    trial_log:           List[Dict]        = field(default_factory=list)
    committed_params:    Dict[str, Any]    = field(default_factory=dict)
    # track consecutive underperformance per signal (key → count)
    consecutive_under:   Dict[str, int]    = field(default_factory=dict)


# ── Engine ────────────────────────────────────────────────────────────────────

class CognitiveAuditEngine:
    """
    Monitors cognitive performance signals, proposes parameter calibrations,
    and evaluates whether those calibrations improved the underperforming signal.

    Usage (from InternalThoughtLoop slow cycle):
        cae = CognitiveAuditEngine(organism, ai_system)
        cae.tick(slow_cycle_count)
    """

    def __init__(
        self,
        organism:   "CognitiveOrganism",
        ai_system:  Any,
        path:       str = "data/persona/cognitive_params.json",
    ) -> None:
        self._organism  = organism
        self._ai        = ai_system
        self._path      = Path(path)
        self._state     = AuditState()
        self._lock      = threading.Lock()
        self._load()
        self._apply_committed_params()
        logger.info(
            f"[CognitiveAudit] Initialised — "
            f"{len(self._state.trial_log)} past trials, "
            f"{len(self._state.committed_params)} committed params"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Called every slow cycle from InternalThoughtLoop."""
        try:
            # 1. Evaluate any active trial first
            if self._state.active_trial:
                trial = DevelopmentProposal(**self._state.active_trial)
                if slow_cycle - trial.proposed_at_cycle >= TRIAL_DURATION_CYCLES:
                    threading.Thread(
                        target=self._evaluate_trial,
                        args=(trial, slow_cycle),
                        daemon=True,
                        name="cae-evaluate",
                    ).start()
                    return

            # 2. Run audit on schedule
            if slow_cycle % AUDIT_EVERY_N != 0 or slow_cycle == 0:
                return

            threading.Thread(
                target=self._run_audit,
                args=(slow_cycle,),
                daemon=True,
                name="cae-audit",
            ).start()

        except Exception as e:
            logger.debug(f"[CognitiveAudit] tick error (non-fatal): {e}")

    def status_summary(self) -> Dict[str, Any]:
        """Returns a human-readable status snapshot for UI/logging."""
        with self._lock:
            snap = self._state.snapshot_log[-1] if self._state.snapshot_log else {}
            trial = self._state.active_trial
            total_audits = len(self._state.snapshot_log)
            total_trials = len(self._state.trial_log)
            confirmed_trials = sum(
                1 for t in self._state.trial_log if t.get("outcome") == "confirmed"
            )
        worst = None
        if snap:
            try:
                worst = SignalSnapshot(**{
                    k: v for k, v in snap.items()
                    if k in SignalSnapshot.__dataclass_fields__
                }).worst_signal()
            except Exception:
                pass
        return {
            "last_snapshot": snap,
            "last_audit_cycle": self._state.last_audit_cycle,
            "total_audits": total_audits,
            "total_trials": total_trials,
            "confirmed_trials": confirmed_trials,
            "worst_signal_now": worst,
            "active_trial": trial,
            "committed_params": dict(self._state.committed_params),
            "consecutive_under": dict(self._state.consecutive_under),
        }

    def prompt_fragment(self) -> str:
        """Returns a short status string for optional prompt injection."""
        if not self._state.committed_params:
            return ""
        parts = [f"{k}={v}" for k, v in self._state.committed_params.items()]
        return f"[CognitiveAudit] Committed calibrations: {', '.join(parts)}"

    # ── Audit cycle ───────────────────────────────────────────────────────────

    def _run_audit(self, slow_cycle: int) -> None:
        try:
            snapshot = self._read_signals(slow_cycle)

            with self._lock:
                self._state.snapshot_log.append(asdict(snapshot))
                if len(self._state.snapshot_log) > MAX_AUDIT_LOG:
                    self._state.snapshot_log = self._state.snapshot_log[-MAX_AUDIT_LOG:]
                self._state.last_audit_cycle = slow_cycle

            self._update_consecutive_underperformance(snapshot)

            # Only generate a proposal when there's no active trial
            if self._state.active_trial:
                logger.debug("[CognitiveAudit] Skipping proposal — trial already active")
                self._save()
                return

            worst = snapshot.worst_signal()
            if worst is None:
                logger.info("[CognitiveAudit] ✅ All signals on target")
                self._save()
                return

            # Require TWO consecutive under-target readings before proposing
            if self._state.consecutive_under.get(worst, 0) < 2:
                logger.info(
                    f"[CognitiveAudit] {worst} underperforming "
                    f"(count={self._state.consecutive_under.get(worst,0)}), "
                    f"watching for next cycle"
                )
                self._save()
                return

            # Generate and activate a trial
            value = snapshot.as_dict()[worst]
            target = SIGNAL_TARGETS[worst]
            proposal = self._generate_proposal(worst, value, target, slow_cycle)
            if proposal:
                self._activate_trial(proposal)
                logger.info(
                    f"[CognitiveAudit] 🔬 Trial activated: {worst} "
                    f"({value:.3f} < {target:.2f}) → "
                    f"{proposal.parameter} {proposal.prior_value} → {proposal.proposed_value}"
                )

            self._save()

        except Exception as e:
            logger.debug(f"[CognitiveAudit] _run_audit error (non-fatal): {e}")

    def _update_consecutive_underperformance(self, snapshot: SignalSnapshot) -> None:
        deficits = snapshot.deficit()
        with self._lock:
            for name, gap in deficits.items():
                if gap < 0:
                    self._state.consecutive_under[name] = (
                        self._state.consecutive_under.get(name, 0) + 1
                    )
                else:
                    self._state.consecutive_under[name] = 0

    # ── Signal reading ────────────────────────────────────────────────────────

    def _read_signals(self, slow_cycle: int) -> SignalSnapshot:
        snap = SignalSnapshot(slow_cycle=slow_cycle, timestamp=time.time())

        # 1. Epistemic grounding rate
        try:
            ar = getattr(self._organism, '_loop', None)
            ar = getattr(ar, '_autonomous_reflection', None) if ar else None
            if ar and hasattr(ar, '_epistemic'):
                snap.epistemic_grounding_rate = ar._epistemic._recent_grounding_rate()
        except Exception:
            pass

        # 2. Experiment confirmation rate
        try:
            loop = getattr(self._organism, '_loop', None)
            ae = getattr(loop, '_auto_experiment', None) if loop else None
            if ae and hasattr(ae, '_state'):
                total = ae._state.total_experiments
                if total > 0:
                    snap.experiment_confirmation_rate = (
                        ae._state.total_confirmed / total
                    )
        except Exception:
            pass

        # 3. Question resolution rate
        try:
            loop = getattr(self._organism, '_loop', None)
            re = getattr(loop, '_resolution_engine', None) if loop else None
            if re and hasattr(re, '_state'):
                resolved     = re._state.total_resolved
                unresolvable = re._state.total_unresolvable
                total = resolved + unresolvable
                if total > 0:
                    snap.question_resolution_rate = resolved / total
        except Exception:
            pass

        # 4. Mean skill depth
        try:
            sr = getattr(self._organism, 'skill_registry', None)
            skills = getattr(getattr(sr, '_state', None), 'skills', {})
            if skills:
                depths = [s.get('depth', 0) for s in skills.values()]
                snap.mean_skill_depth = sum(depths) / len(depths)
        except Exception:
            pass

        # 5. Belief confidence mean
        try:
            sc = getattr(self._ai, 'self_concept', None)
            # Try multiple access paths
            if sc is None:
                ts = getattr(self._organism, 'thought_system', None)
                sc = getattr(ts, 'self_concept', None) if ts else None
            if sc and hasattr(sc, '_beliefs') and sc._beliefs:
                confs = [b.confidence for b in sc._beliefs.values()]
                snap.belief_confidence_mean = sum(confs) / len(confs)
        except Exception:
            pass

        logger.debug(
            f"[CognitiveAudit] Signals @ cycle {slow_cycle}: "
            f"grounding={snap.epistemic_grounding_rate:.2f} "
            f"exp_conf={snap.experiment_confirmation_rate:.2f} "
            f"q_res={snap.question_resolution_rate:.2f} "
            f"skill_depth={snap.mean_skill_depth:.2f} "
            f"belief_conf={snap.belief_confidence_mean:.2f}"
        )
        return snap

    # ── Proposal generation ───────────────────────────────────────────────────

    def _generate_proposal(
        self,
        dimension: str,
        current_value: float,
        target: float,
        slow_cycle: int,
    ) -> Optional[DevelopmentProposal]:
        """Ask the LLM to propose a specific parameter calibration."""
        current_params = self._current_params()

        # Build the parameter change bounds description
        param_desc_lines = []
        for pname, (default, pmin, pmax, ptype) in PARAM_SPACE.items():
            cur = current_params.get(pname, default)
            param_desc_lines.append(
                f"  {pname}: current={cur}, range=[{pmin},{pmax}], type={ptype.__name__}"
            )
        param_desc = "\n".join(param_desc_lines)

        # v64: MetaLearningAudit — bias away from parameters with poor effectiveness
        guidance_note = ""
        try:
            loop = getattr(self._organism, '_loop', None)
            mla  = getattr(loop, '_meta_learning_audit', None) if loop else None
            if mla:
                guidance = mla.parameter_guidance()
                do_not   = [p for p, w in guidance.items() if w == 0.0]
                low_eff  = [p for p, w in guidance.items() if 0 < w < 0.6]
                high_eff = [p for p, w in guidance.items() if w > 1.1]
                parts = []
                if do_not:  parts.append(f"DO NOT use: {do_not}")
                if low_eff: parts.append(f"LOW effectiveness history: {low_eff}")
                if high_eff: parts.append(f"HIGH effectiveness history: {high_eff}")
                if parts:
                    guidance_note = "\nHISTORICAL EFFECTIVENESS:\n" + "\n".join(parts) + "\n"
        except Exception:
            pass

        prompt = f"""You are the cognitive self-calibration layer of an adaptive AI system.

UNDERPERFORMING SIGNAL
  name:    {dimension}
  current: {current_value:.3f}
  target:  {target:.2f}
  deficit: {current_value - target:.3f}

CURRENT COGNITIVE PARAMETERS
{param_desc}

PARAMETER SEMANTICS
  resolve_every_n:        How often (slow cycles) questions are resolved. Lower = more frequent.
  generate_every_n:       How often new experiments are generated. Lower = more frequent.
  decay_rate_base:        Rate at which beliefs lose confidence without validation. Higher = faster decay.
  grounding_boost:        Confidence gain when a belief is evidence-validated. Higher = stronger grounding.
  curiosity_halflife_hrs: Hours for curiosity to halve. Higher = topics stay active longer.
  reflection_every_n:     How often autonomous reflection runs. Lower = more frequent.
{guidance_note}
TASK
Select one parameter to change to improve {dimension}.
Reason concisely (2 sentences max).
The change must stay within the stated range.
Change magnitude must not exceed 30% of current value.

Respond ONLY in JSON (no markdown, no preamble):
{{
  "parameter": "<param_name>",
  "proposed_value": <number>,
  "rationale": "<2-sentence explanation>"
}}"""

        response = self._llm_call(prompt, caller="cognitive_audit")
        if not response:
            return None

        try:
            # Strip any accidental markdown fences
            clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            pname = data.get("parameter", "")
            if pname not in PARAM_SPACE:
                logger.debug(f"[CognitiveAudit] Unknown parameter in proposal: {pname}")
                return None

            default, pmin, pmax, ptype = PARAM_SPACE[pname]
            prior = current_params.get(pname, default)
            raw   = data.get("proposed_value", prior)

            # Clamp to bounds
            proposed = ptype(max(pmin, min(pmax, raw)))

            # Enforce max-change-pct
            if prior != 0:
                change_pct = abs(proposed - prior) / abs(prior)
                if change_pct > 0.30:
                    # Clamp to 30%
                    direction = 1 if proposed > prior else -1
                    proposed = ptype(prior + direction * abs(prior) * 0.30)
                    proposed = ptype(max(pmin, min(pmax, proposed)))

            if proposed == prior:
                logger.debug(f"[CognitiveAudit] Proposal resulted in no change for {pname}")
                return None

            return DevelopmentProposal(
                dimension           = dimension,
                signal_at_proposal  = current_value,
                signal_target       = target,
                parameter           = pname,
                prior_value         = prior,
                proposed_value      = proposed,
                rationale           = data.get("rationale", ""),
                proposed_at_cycle   = slow_cycle,
            )

        except Exception as e:
            logger.debug(f"[CognitiveAudit] Proposal parse failed: {e} | raw={response[:200]}")
            return None

    # ── Trial management ──────────────────────────────────────────────────────

    def _activate_trial(self, proposal: DevelopmentProposal) -> None:
        """Apply proposed value to live module global and record trial."""
        self._set_module_global(proposal.parameter, proposal.proposed_value)
        with self._lock:
            self._state.active_trial = asdict(proposal)
        self._save()

    def _evaluate_trial(self, trial: DevelopmentProposal, slow_cycle: int) -> None:
        """Re-read the signal and determine whether the trial improved it."""
        try:
            snapshot = self._read_signals(slow_cycle)
            signal_now = snapshot.as_dict().get(trial.dimension, 0.0)
            target     = trial.signal_target
            prior      = trial.signal_at_proposal

            # Δ as fraction of target (normalised)
            delta = signal_now - prior

            if delta >= CONFIRM_THRESHOLD * target:
                outcome = "confirmed"
            elif delta <= -DISCONFIRM_THRESHOLD * target:
                outcome = "disconfirmed"
            else:
                outcome = "inconclusive"

            trial.outcome           = outcome
            trial.signal_at_eval    = signal_now
            trial.evaluated_at_cycle = slow_cycle

            if outcome == "confirmed":
                self._commit_trial(trial)
                logger.info(
                    f"[CognitiveAudit] ✅ Confirmed: {trial.parameter} "
                    f"{trial.prior_value} → {trial.proposed_value} "
                    f"({trial.dimension}: {prior:.3f} → {signal_now:.3f})"
                )
            else:
                self._revert_trial(trial)
                logger.info(
                    f"[CognitiveAudit] ↩ {outcome.capitalize()}: {trial.parameter} "
                    f"reverted to {trial.prior_value} "
                    f"({trial.dimension}: {prior:.3f} → {signal_now:.3f})"
                )

            with self._lock:
                self._state.trial_log.append(asdict(trial))
                if len(self._state.trial_log) > MAX_TRIAL_LOG:
                    self._state.trial_log = self._state.trial_log[-MAX_TRIAL_LOG:]
                self._state.active_trial = None
                # Reset consecutive underperformance for this signal so we don't
                # immediately re-propose after a revert.
                self._state.consecutive_under[trial.dimension] = 0

            self._save()

        except Exception as e:
            logger.debug(f"[CognitiveAudit] _evaluate_trial error (non-fatal): {e}")
            with self._lock:
                self._state.active_trial = None
            self._revert_trial(trial)
            self._save()

    def _commit_trial(self, trial: DevelopmentProposal) -> None:
        """Persist confirmed parameter change to committed_params."""
        trial.committed = True
        with self._lock:
            self._state.committed_params[trial.parameter] = trial.proposed_value
        # Module global is already at proposed_value — leave it.

    def _revert_trial(self, trial: DevelopmentProposal) -> None:
        """Restore the module global to the prior value."""
        self._set_module_global(trial.parameter, trial.prior_value)

    # ── Module global mutation ────────────────────────────────────────────────

    def _set_module_global(self, param: str, value: Any) -> None:
        """Mutate the live module-level constant for a cognitive parameter."""
        if param not in PARAM_MODULE_MAP:
            return
        mod_path, global_name = PARAM_MODULE_MAP[param]
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            setattr(mod, global_name, value)
            logger.debug(
                f"[CognitiveAudit] Set {mod_path}.{global_name} = {value}"
            )
        except Exception as e:
            logger.debug(f"[CognitiveAudit] Could not set {param}: {e}")

    def _apply_committed_params(self) -> None:
        """On startup, restore all previously committed parameter changes."""
        for param, value in self._state.committed_params.items():
            self._set_module_global(param, value)
            logger.info(f"[CognitiveAudit] Restored committed param: {param}={value}")

    # ── Current parameter snapshot ────────────────────────────────────────────

    def _current_params(self) -> Dict[str, Any]:
        """Read the current live values of all tunable parameters."""
        result = {}
        import importlib
        for pname, (default, pmin, pmax, ptype) in PARAM_SPACE.items():
            # First, use committed value if present
            if pname in self._state.committed_params:
                result[pname] = self._state.committed_params[pname]
                continue
            # Otherwise read from module
            if pname in PARAM_MODULE_MAP:
                mod_path, global_name = PARAM_MODULE_MAP[pname]
                try:
                    mod = importlib.import_module(mod_path)
                    result[pname] = getattr(mod, global_name, default)
                    continue
                except Exception:
                    pass
            result[pname] = default
        return result

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, caller: str = "cognitive_audit") -> Optional[str]:
        from core.llm_scheduler import llm_scheduler
        result = ""
        with llm_scheduler.sync_slot(
            priority     = 3,
            skip_if_busy = True,
            caller       = caller,
        ) as acquired:
            if not acquired:
                return None
            try:
                result = self._ai.get_response(
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 200,
                    temperature = 0.50,
                ) or ""
            except Exception as e:
                logger.debug(f"[CognitiveAudit] LLM failed: {e}")
        return result or None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = AuditState(
                    last_audit_cycle   = data.get("last_audit_cycle", 0),
                    snapshot_log       = data.get("snapshot_log", []),
                    active_trial       = data.get("active_trial"),
                    trial_log          = data.get("trial_log", []),
                    committed_params   = data.get("committed_params", {}),
                    consecutive_under  = data.get("consecutive_under", {}),
                )
        except Exception as e:
            logger.warning(f"[CognitiveAudit] Load failed (using defaults): {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "last_audit_cycle":  self._state.last_audit_cycle,
                    "snapshot_log":      self._state.snapshot_log,
                    "active_trial":      self._state.active_trial,
                    "trial_log":         self._state.trial_log,
                    "committed_params":  self._state.committed_params,
                    "consecutive_under": self._state.consecutive_under,
                    "_meta": {
                        "written_at":    time.time(),
                        "module":        "cognitive_audit_engine",
                        "version":       "v48",
                    },
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[CognitiveAudit] Save failed: {e}")
