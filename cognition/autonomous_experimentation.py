"""
cognition/autonomous_experimentation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AutonomousExperimentationEngine — hypothesis, test, measure, revise.

The gap this closes
────────────────────
Current PandoraBOX: observe → think → remember
Missing:        observe → hypothesize → test → compare → revise

This is the difference between a reflective system and one that
actively learns from interventions. Without it, PandoraBOX can accumulate
beliefs but cannot verify them against reality.

What an Experiment is
─────────────────────
  Experiment(
    hypothesis:       what is believed to be true
    action:           what change to make to test it
    expected_result:  what should be observed if hypothesis holds
    measurement_fn:   what metric to track across N interactions
    observed_results: actual measurements collected
    confidence_delta: how much the hypothesis was confirmed/disconfirmed
    outcome:          "confirmed" | "disconfirmed" | "inconclusive" | "pending"
  )

Example experiments PandoraBOX can run autonomously:
  - "Shorter responses increase user engagement"
    Action: reduce avg response length by 20% for 5 interactions
    Measure: user follow-up rate, message length

  - "Leading with a question increases relational depth"
    Action: open each response with a genuine question for 6 interactions
    Measure: conversation turns per session

  - "Surfacing preferences increases resonance"
    Action: include one preference reference per session
    Measure: session length, user return rate

  - "More concrete examples reduce clarification requests"
    Action: add at least one example per complex explanation
    Measure: clarification-seeking messages

Execution cycle
───────────────
  1. Hypothesis generation (LLM, priority=3) — from open questions,
     belief conflicts, or reflection insights. Generated every
     GENERATE_EVERY_N slow cycles.

  2. Experiment design (LLM) — turns hypothesis into an actionable
     test with a concrete measurement plan.

  3. Experiment activation — sets behavioural parameters in
     CognitiveOrganism that shape the next N interactions.

  4. Measurement collection — after each interaction, records
     the relevant metric against the experiment's expected result.

  5. Evaluation — after N interactions, evaluates whether the
     hypothesis held. Updates belief confidence accordingly.

  6. Storage — confirmed experiments update beliefs and skills.
     Disconfirmed experiments update beliefs in the opposite direction.
     All experiments stored as memories.

Safeguards
──────────
  - MAX_ACTIVE_EXPERIMENTS = 1 (never run two simultaneously)
  - MAX_INTERACTIONS_PER_EXPERIMENT = 8 (bounded test window)
  - LLM priority = 3, skip_if_busy
  - Experiment changes are nudges, not overrides
  - All changes are logged and reversible
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from managers.settings_manager import get_persona_name

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GENERATE_EVERY_N              = 25    # slow cycles between new experiment generation
EVALUATE_EVERY_N              = 5     # slow cycles between measurement checks
MAX_ACTIVE_EXPERIMENTS        = 1
MAX_INTERACTIONS_PER_EXP      = 8
MIN_MEASUREMENTS_FOR_EVAL     = 3
CONFIRMED_CONFIDENCE_BOOST    = 0.08
DISCONFIRMED_CONFIDENCE_DELTA = -0.10
INCONCLUSIVE_DELTA            = -0.02
MAX_EXPERIMENT_LOG            = 60


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    """A single autonomous experiment."""
    id:               str   = ""
    hypothesis:       str   = ""
    action:           str   = ""        # what behavioural change to make
    expected_result:  str   = ""
    metric:           str   = ""        # what to measure: "response_length" |
                                        # "question_rate" | "session_turns" |
                                        # "follow_up_rate" | "preference_surfaces"
    target_value:     float = 0.0       # expected direction: + means increase, - decrease
    measurements:     List[float] = field(default_factory=list)
    baseline:         float = 0.0       # metric value before experiment
    interactions_run: int   = 0
    max_interactions: int   = MAX_INTERACTIONS_PER_EXP
    started_at:       float = field(default_factory=time.time)
    completed_at:     Optional[float] = None
    outcome:          str   = "pending"    # pending|confirmed|disconfirmed|inconclusive
    confidence_delta: float = 0.0
    hypothesis_belief: str  = ""        # which belief this experiment tests
    slow_cycle_started: int = 0

    def is_complete(self) -> bool:
        return (
            self.interactions_run >= self.max_interactions
            or self.outcome != "pending"
        )

    def mean_measurement(self) -> float:
        return sum(self.measurements) / len(self.measurements) if self.measurements else 0.0

    def result_direction(self) -> float:
        """Positive = metric moved in expected direction."""
        if not self.measurements or self.baseline == 0:
            return 0.0
        delta = self.mean_measurement() - self.baseline
        # Normalise by baseline
        return delta / max(abs(self.baseline), 1.0)


@dataclass
class ExperimentState:
    """Persisted state."""
    active_experiment:  Optional[Dict] = None
    experiment_log:     List[Dict]     = field(default_factory=list)
    total_experiments:  int = 0
    total_confirmed:    int = 0
    total_disconfirmed: int = 0
    total_inconclusive: int = 0
    current_overrides:  Dict[str, Any] = field(default_factory=dict)
    # v50: persistence queue — confirmed experiments awaiting +50 interaction check
    persistence_queue:  List[Dict]     = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────

class AutonomousExperimentationEngine:
    """
    Runs bounded hypothesis-test-measure-revise cycles autonomously.

    Usage (from InternalThoughtLoop slow cycle):
        aee = AutonomousExperimentationEngine(organism, ai_system)
        aee.tick(slow_cycle_count)

    Usage (from CognitiveOrganism post-interaction):
        aee.record_interaction(user_input, response, session_context)
    """

    def __init__(
        self,
        organism:  Any,
        ai_system: Any,
        path:      str = "data/persona/autonomous_experimentation.json",
    ):
        self._o    = organism
        self._ai   = ai_system
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = ExperimentState()
        self._load()
        logger.info(
            f"[AutoExperiment] Initialised — "
            f"{self._state.total_experiments} total, "
            f"{self._state.total_confirmed} confirmed"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Drive experiment lifecycle from slow cycle. Non-fatal."""
        try:
            # Check if active experiment needs evaluation
            if slow_cycle % EVALUATE_EVERY_N == 0:
                self._evaluate_active()

            # Generate new experiment if none active
            if slow_cycle % GENERATE_EVERY_N == 0 and slow_cycle > 0:
                with self._lock:
                    has_active = (
                        self._state.active_experiment is not None
                        and self._state.active_experiment.get("outcome") == "pending"
                    )
                if not has_active:
                    # v50: gate check — experiments cost cognitive_energy;
                    # don't start one when the system is already depleted.
                    _gate_ok = True
                    try:
                        from core.state import state as _st
                        _org  = getattr(getattr(_st, 'persona', None), '_organism', None)
                        _gate = getattr(_org, 'behavior_gate', None) if _org else None
                        if _gate:
                            _gate_ok = _gate.may_start_experiment()
                    except Exception:
                        pass
                    if _gate_ok:
                        self._generate_experiment(slow_cycle)
        except Exception as e:
            logger.debug(f"[AutoExperiment] tick error (non-fatal): {e}")

    def record_interaction(
        self,
        user_input: str,
        response:   str,
        context:    Dict = None,
    ) -> None:
        """
        Called after each interaction to collect measurement for active experiment.
        Increments interactions_run and records the relevant metric.
        Also feeds measurements to any pending persistence verifications.
        """
        try:
            # v50: increment global interaction counter for persistence scheduling
            self._interaction_count = getattr(self, '_interaction_count', 0) + 1

            with self._lock:
                active_pending = (
                    self._state.active_experiment is not None
                    and self._state.active_experiment.get("outcome") == "pending"
                )

            if active_pending:
                with self._lock:
                    exp_dict = self._state.active_experiment

                metric  = exp_dict.get("metric", "")
                value   = self._measure(metric, user_input, response, context or {})

                with self._lock:
                    self._state.active_experiment["measurements"].append(value)
                    self._state.active_experiment["interactions_run"] += 1

                self._save()

            # v50: feed persistence verifications
            self._check_persistence_queue(user_input, response, context or {})

        except Exception as e:
            logger.debug(f"[AutoExperiment] record_interaction failed: {e}")

    def _check_persistence_queue(
        self, user_input: str, response: str, context: Dict
    ) -> None:
        """
        v50 — Experiment persistence verifier.
        Checks whether confirmed behavioral changes have held 50 interactions later.

        Outcomes:
          held    → SelfBelief.confidence += 0.03 (compounding, behavior is durable)
          decayed → SelfBelief.confidence -= 0.05, logged as "behavioral decay"
          partial → no change, noted in log
        """
        try:
            count = getattr(self, '_interaction_count', 0)
            with self._lock:
                queue = [p for p in self._state.persistence_queue if p["status"] == "pending"]

            for item in queue:
                # Collect measurements for pending verifications
                metric = item["metric"]
                value  = self._measure(metric, user_input, response, context)
                item["verify_measurements"].append(value)

                # Evaluate when due
                if count >= item["verify_at_count"] and len(item["verify_measurements"]) >= 5:
                    verify_mean   = sum(item["verify_measurements"]) / len(item["verify_measurements"])
                    original_mean = item["confirmed_mean"]
                    delta         = verify_mean - original_mean

                    if delta >= original_mean * 0.10:
                        result = "held"
                        conf_delta = +0.03
                        logger.info(
                            f"[AutoExperiment] 🔒 Persistence HELD: '{item['hypothesis'][:50]}' "
                            f"original={original_mean:.2f} verify={verify_mean:.2f}"
                        )
                    elif delta <= -original_mean * 0.10:
                        result = "decayed"
                        conf_delta = -0.05
                        logger.info(
                            f"[AutoExperiment] 📉 Behavioral DECAY: '{item['hypothesis'][:50]}' "
                            f"original={original_mean:.2f} verify={verify_mean:.2f}"
                        )
                    else:
                        result = "partial"
                        conf_delta = 0.0

                    item["status"]      = result
                    item["verify_mean"] = round(verify_mean, 4)

                    # Apply belief delta
                    if conf_delta != 0.0 and item.get("hypothesis_belief"):
                        self._adjust_belief(item["hypothesis_belief"], conf_delta,
                                            f"persistence_{result}")

                    with self._lock:
                        for p in self._state.persistence_queue:
                            if p["hypothesis"] == item["hypothesis"]:
                                p.update(item)

            self._save()
        except Exception as e:
            logger.debug(f"[AutoExperiment] _check_persistence_queue failed: {e}")

    def active_behavioural_overrides(self) -> Dict[str, Any]:
        """
        Return current experiment's behavioural nudges for prompt injection.
        Called from _build_prompt_additions.
        """
        with self._lock:
            overrides = dict(self._state.current_overrides)
        return overrides

    def experiment_prompt_fragment(self) -> str:
        """Return active experiment instruction for prompt injection."""
        with self._lock:
            exp = self._state.active_experiment
        if not exp or exp.get("outcome") != "pending":
            return ""
        action = exp.get("action", "")
        if not action:
            return ""
        return f"[Experiment active] {action}"

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate_experiment(self, slow_cycle: int) -> None:
        """Generate a new experiment from open questions and belief conflicts."""
        context = self._build_generation_context()
        if not context:
            return

        prompt = (
            f"You are {get_persona_name()}'s experimentation process.\n\n"
            f"Current cognitive context:\n{context}\n\n"
            f"Design one small, bounded experiment {get_persona_name()} can run over "
            f"the next {MAX_INTERACTIONS_PER_EXP} interactions to test "
            f"a specific hypothesis about how she engages.\n\n"
            f"The experiment must:\n"
            f"  - Be a concrete behavioural change (not a topic preference)\n"
            f"  - Be measurable from response text and interaction patterns\n"
            f"  - Have a clear expected result\n"
            f"  - Be achievable within {MAX_INTERACTIONS_PER_EXP} interactions\n\n"
            + (
                "IMPORTANT — COUNTERFACTUAL MODE ACTIVE: prefer a hypothesis that could "
                "DISCONFIRM an existing belief rather than confirm one. Choose the experiment "
                "most likely to challenge current assumptions.\n\n"
                if getattr(self, '_immune_force_disconfirm_bias', False) else ""
            )
            + (
                # DecisionPolicy: bias hypothesis selection toward active values
                lambda: (
                    f"VALUE ALIGNMENT: {get_persona_name()}'s current active values are "
                    f"{_policy_summary}. Prefer hypotheses that test or express "
                    f"these values in interaction.\n\n"
                    if (_policy_summary := self._get_policy_summary()) else ""
                )
            )()
            + f"Metric must be one of: response_length, question_rate, "
            f"example_rate, preference_surfaces, session_turns\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{"hypothesis": "...", "action": "...", '
            f'"expected_result": "...", "metric": "response_length", '
            f'"target_value": 0.8, "hypothesis_belief": "..."}}'
        )

        result = self._llm_call(prompt, caller="autoexp.generate")

        # Phase 3.4: if LLM unavailable, fall back to evidence-seeded
        # hypothesis derived directly from counterfactual data.
        if not result:
            evidence_data = self._evidence_hypothesis()
            if evidence_data is None:
                return
            logger.info(
                "[AutoExperiment] LLM unavailable — using evidence-seeded "
                "hypothesis from counterfactual data"
            )
            try:
                import uuid as _uuid2
                _metric   = evidence_data.get("metric", "response_length")
                _baseline = self._measure_baseline(_metric)
                _exp = Experiment(
                    id                 = str(_uuid2.uuid4())[:8],
                    hypothesis         = evidence_data.get("hypothesis", "")[:200],
                    action             = evidence_data.get("action", "")[:150],
                    expected_result    = evidence_data.get("expected_result", "")[:150],
                    metric             = _metric,
                    target_value       = float(evidence_data.get("target_value", 0.0)),
                    baseline           = _baseline,
                    hypothesis_belief  = evidence_data.get("hypothesis_belief", "")[:80],
                    slow_cycle_started = slow_cycle,
                )
                with self._lock:
                    self._state.active_experiment = asdict(_exp)
                    self._state.current_overrides = {"action": _exp.action}
                    self._state.total_experiments += 1
                self._save()
                logger.info(
                    f"[AutoExperiment] 🧪 Evidence-seeded experiment: "
                    f"'{_exp.hypothesis[:60]}'"
                )
            except Exception as _ev_e:
                logger.debug(f"[AutoExperiment] evidence hypothesis error: {_ev_e}")
            return

        try:
            import re, json as _j, uuid
            clean = re.sub(r"```json|```", "", result).strip()
            data  = _j.loads(clean)

            # Phase 3.4: quality gate — reject duplicates and belief conflicts
            if not self._hypothesis_quality_gate(data):
                logger.info("[AutoExperiment] Hypothesis rejected by quality gate")
                return

            metric = data.get("metric", "response_length")
            if metric not in {
                "response_length", "question_rate", "example_rate",
                "preference_surfaces", "session_turns"
            }:
                metric = "response_length"

            baseline = self._measure_baseline(metric)

            exp = Experiment(
                id                 = str(uuid.uuid4())[:8],
                hypothesis         = data.get("hypothesis",       "")[:200],
                action             = data.get("action",           "")[:150],
                expected_result    = data.get("expected_result",  "")[:150],
                metric             = metric,
                target_value       = float(data.get("target_value", 0.0)),
                baseline           = baseline,
                hypothesis_belief  = data.get("hypothesis_belief","")[:80],
                slow_cycle_started = slow_cycle,
            )

            with self._lock:
                self._state.active_experiment = asdict(exp)
                self._state.current_overrides = {"action": exp.action}
                self._state.total_experiments += 1

            self._save()
            logger.info(
                f"[AutoExperiment] 🧪 New experiment: '{exp.hypothesis[:60]}' "
                f"metric={exp.metric} baseline={exp.baseline:.2f}"
            )
        except Exception as e:
            logger.debug(f"[AutoExperiment] generation parse failed: {e}")


    def _build_generation_context(self) -> str:
        """Summarise current cognitive state for experiment generation.

        Phase 3.4: extended with three evidence fragments from Phases
        3.1/3.2/3.3 — calibration ECE, counterfactual action rankings,
        and WSDM trajectory predictions. These ground hypothesis
        generation in quantitative evidence rather than vague reflection,
        enabling the LLM to produce hypotheses like "emotional responses
        have ranked last 4 consecutive times — test whether analytical
        framing improves outcomes."
        """
        parts: List[str] = []

        # Open questions
        try:
            ts  = getattr(self._o, "thought_stream", None)
            oqs = sorted(
                getattr(ts, "_open_questions", []),
                key=lambda q: q.activation, reverse=True
            )[:2]
            if oqs:
                parts.append("Open questions: " + "; ".join(q.text[:60] for q in oqs))
        except Exception:
            pass

        # Active belief conflicts
        try:
            are = getattr(getattr(self._o, "_loop", None), "_autonomous_reflection", None)
            if are:
                active = [
                    d for d in
                    getattr(are._epistemic._state, "active_disagreements", [])
                    if not d.get("resolved", False)
                ][:2]
                if active:
                    parts.append(
                        "Belief conflicts: " +
                        "; ".join(d.get("dimension", "") for d in active)
                    )
        except Exception:
            pass

        # Recent reflection insights
        try:
            if are:
                recent = getattr(are._state, "reflection_log", [])[-2:]
                for r in recent:
                    if r.get("output"):
                        parts.append(f"Recent reflection: {r['output'][:60]}")
        except Exception:
            pass

        # ── Phase 3.4 Evidence Fragment 1: Calibration ECE ─────────────────
        # Tells the LLM which prediction modules are over/under-confident,
        # grounding hypotheses about reliability of self-assessment.
        try:
            loop = getattr(self._o, "_loop", None)
            pcm  = getattr(loop, "_consequence_model", None)
            cal  = getattr(pcm, "_calibration_engine", None) if pcm else None
            if cal is not None:
                cal_parts = []
                for module in ("pcm", "wsdm"):
                    ece = cal.compute_ece(module)
                    if ece is not None:
                        quality = (
                            "well-calibrated" if ece < 0.10 else
                            "slightly overconfident" if ece < 0.25 else
                            "significantly overconfident"
                        )
                        cal_parts.append(f"{module.upper()} ECE={ece:.3f} ({quality})")
                if cal_parts:
                    parts.append("Calibration evidence: " + "; ".join(cal_parts))
        except Exception:
            pass

        # ── Phase 3.4 Evidence Fragment 2: Counterfactual rankings ─────────
        # Surfaces which action types are systematically suboptimal,
        # enabling hypotheses like "try analytical instead of emotional."
        try:
            cf_sim = getattr(loop, "_counterfactual_sim", None) if loop else None
            if cf_sim is not None:
                cf_status = cf_sim.status()
                streak    = cf_status.get("worst_actual_streak", 0)
                recent_cf = cf_status.get("recent", [])
                if recent_cf:
                    # Tally which actions ranked best vs worst across recent queries
                    best_counts: dict = {}
                    worst_counts: dict = {}
                    for entry in recent_cf:
                        ba = entry.get("best_alternative")
                        aa = entry.get("actual_action")
                        wa = entry.get("worst_actual", False)
                        if ba:
                            best_counts[ba] = best_counts.get(ba, 0) + 1
                        if wa and aa:
                            worst_counts[aa] = worst_counts.get(aa, 0) + 1

                    cf_lines = []
                    if streak >= 2:
                        cf_lines.append(
                            f"worst-actual streak={streak} (repeatedly choosing "
                            f"lowest-predicted action)"
                        )
                    if best_counts:
                        top_best = max(best_counts, key=best_counts.get)
                        cf_lines.append(
                            f"'{top_best}' was best alternative in "
                            f"{best_counts[top_best]}/{len(recent_cf)} recent cases"
                        )
                    if worst_counts:
                        top_worst = max(worst_counts, key=worst_counts.get)
                        cf_lines.append(
                            f"'{top_worst}' ranked worst in "
                            f"{worst_counts[top_worst]}/{len(recent_cf)} recent cases"
                        )
                    if cf_lines:
                        parts.append(
                            "Counterfactual analysis (" +
                            f"last {len(recent_cf)} interactions): " +
                            "; ".join(cf_lines)
                        )
        except Exception:
            pass

        # ── Phase 3.4 Evidence Fragment 3: WSDM trajectory ─────────────────
        # Shows predicted energy trend over next 3 steps for the current
        # dominant action type — grounds hypotheses about sustainability.
        try:
            wsdm = getattr(loop, "_world_self_dynamics", None) if loop else None
            ai   = getattr(self._o, "ai_system", None)
            if wsdm is not None and hasattr(wsdm, "predict_trajectory"):
                # Use most common recent action type from PCM if available
                _pcm_recs = getattr(
                    getattr(loop, "_consequence_model", None), "_records", []
                )
                if _pcm_recs:
                    from collections import Counter
                    _action = Counter(
                        r.action_type for r in _pcm_recs[-10:]
                    ).most_common(1)[0][0]
                    uid = getattr(ai, "_current_user_id", "default") if ai else "default"
                    traj = wsdm.predict_trajectory(_action, user_id=uid, steps=3)
                    if traj:
                        trend = [
                            f"step{i+1}: cog{p.self_deltas.get('cognitive_energy',0):+.3f}"
                            f"(conf={p.confidence:.2f})"
                            for i, p in enumerate(traj)
                        ]
                        total_drift = sum(
                            p.self_deltas.get("cognitive_energy", 0) for p in traj
                        )
                        direction = "depleting" if total_drift < -0.05 else (
                            "sustaining" if total_drift > 0.02 else "neutral"
                        )
                        parts.append(
                            f"WSDM 3-step trajectory for '{_action}' "
                            f"({direction}): " + ", ".join(trend)
                        )
        except Exception:
            pass

        return "\n".join(parts) if parts else ""

    def _evidence_hypothesis(self) -> Optional[Dict]:
        """
        Phase 3.4: generate a structured hypothesis directly from Phase 3.3
        counterfactual data, without requiring an LLM call. Used as fallback
        when LLM is unavailable or _build_generation_context() returns empty.

        If worst_actual_streak >= 3, hypothesis: "replace the action type
        that keeps ranking worst with the consistently better alternative."
        This is the clearest signal in counterfactual data — repeated
        worst-case choices warrant a testable behavioural change.
        """
        try:
            loop   = getattr(self._o, "_loop", None)
            cf_sim = getattr(loop, "_counterfactual_sim", None) if loop else None
            if cf_sim is None:
                return None

            status = cf_sim.status()
            streak = status.get("worst_actual_streak", 0)
            if streak < 3:
                return None

            recent = status.get("recent", [])
            if not recent:
                return None

            # Find most common worst action and best alternative
            from collections import Counter
            worst_actions = Counter(
                e.get("actual_action") for e in recent
                if e.get("worst_actual") and e.get("actual_action")
            )
            best_alts = Counter(
                e.get("best_alternative") for e in recent
                if e.get("best_alternative")
            )
            if not worst_actions or not best_alts:
                return None

            worst = worst_actions.most_common(1)[0][0]
            best  = best_alts.most_common(1)[0][0]

            if worst == best:
                return None

            return {
                "hypothesis": (
                    f"'{best}' response framing produces better predicted outcomes "
                    f"than '{worst}' framing — counterfactual analysis shows "
                    f"'{worst}' ranked worst in {streak} consecutive interactions"
                )[:200],
                "action": (
                    f"Use '{best}' framing instead of '{worst}' for the next "
                    f"interactions"
                )[:150],
                "expected_result": (
                    f"Higher joint score (PCM energy + WSDM engagement) compared "
                    f"to recent '{worst}' baseline"
                )[:150],
                "metric": "response_length",   # most reliable built-in metric
                "target_value": 0.0,
                "hypothesis_belief": f"action_style:{best}_vs_{worst}",
            }
        except Exception:
            return None

    def _hypothesis_quality_gate(self, data: Dict) -> bool:
        """
        Phase 3.4: reject low-quality hypotheses before activating an
        experiment. Returns True if the hypothesis passes, False to reject.

        Rejection criteria:
          1. Duplicates an already-running experiment on the same action
             type — no value in running the same test twice concurrently.
          2. Conflicts with a high-confidence WorldModel causal belief
             (confidence > 0.85) that directly contradicts the expected
             direction — would be testing something the system already
             believes with high confidence is false.
        """
        try:
            with self._lock:
                active = self._state.active_experiment
            if active and active.get("outcome") == "pending":
                existing_action = active.get("action", "")
                new_action = data.get("action", "")
                if (existing_action and new_action and
                        existing_action[:30].lower() == new_action[:30].lower()):
                    logger.debug(
                        "[AutoExperiment] Hypothesis rejected: duplicates "
                        "active experiment action"
                    )
                    return False
        except Exception:
            pass

        try:
            loop = getattr(self._o, "_loop", None)
            wm   = getattr(loop, "_world_model", None) if loop else None
            if wm is not None:
                hypothesis_text = data.get("hypothesis", "").lower()
                for belief in getattr(wm, "causal_beliefs", []):
                    if getattr(belief, "confidence", 0) < 0.85:
                        continue
                    antecedent = getattr(belief, "antecedent", "").lower()
                    if antecedent and antecedent[:20] in hypothesis_text:
                        consequent = getattr(belief, "consequent", "").lower()
                        expected   = data.get("expected_result", "").lower()
                        # Simple heuristic: if belief says "X increases" but
                        # hypothesis expects "X decreases", flag as conflict
                        if (("increase" in consequent and "decrease" in expected) or
                                ("decrease" in consequent and "increase" in expected)):
                            logger.debug(
                                "[AutoExperiment] Hypothesis rejected: conflicts "
                                f"with high-confidence belief '{belief.antecedent[:40]}'"
                            )
                            return False
        except Exception:
            pass

        return True

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_active(self) -> None:
        """Check if active experiment has enough data to evaluate."""
        with self._lock:
            exp_dict = self._state.active_experiment
        if not exp_dict or exp_dict.get("outcome") != "pending":
            return

        interactions = exp_dict.get("interactions_run", 0)
        measurements = exp_dict.get("measurements", [])

        if (interactions < exp_dict.get("max_interactions", MAX_INTERACTIONS_PER_EXP)
                and len(measurements) < MIN_MEASUREMENTS_FOR_EVAL):
            return

        # Reconstruct Experiment object for evaluation
        exp = Experiment(**{
            k: v for k, v in exp_dict.items()
            if k in Experiment.__dataclass_fields__
        })

        outcome, confidence_delta = self._compute_outcome(exp)
        exp.outcome          = outcome
        exp.confidence_delta = confidence_delta
        exp.completed_at     = time.time()

        self._apply_experiment_result(exp)

        with self._lock:
            self._state.active_experiment  = asdict(exp)
            self._state.current_overrides  = {}   # clear behavioural overrides
            self._state.experiment_log.append(asdict(exp))
            if len(self._state.experiment_log) > MAX_EXPERIMENT_LOG:
                self._state.experiment_log = self._state.experiment_log[-MAX_EXPERIMENT_LOG:]
            if outcome == "confirmed":
                self._state.total_confirmed += 1
                # v50: schedule persistence check at +50 interactions.
                # If the behavioral change holds, belief gets +0.03 (compounding).
                # If it decays, belief loses -0.05 and we log "behavioral decay".
                self._state.persistence_queue.append({
                    "hypothesis":          exp.hypothesis,
                    "metric":              exp.metric,
                    "confirmed_mean":      exp.mean_measurement(),
                    "hypothesis_belief":   exp.hypothesis_belief,
                    "scheduled_at_count":  self._interaction_count,
                    "verify_at_count":     self._interaction_count + 50,
                    "verify_measurements": [],
                    "status":              "pending",
                })
            elif outcome == "disconfirmed":
                self._state.total_disconfirmed += 1
            else:
                self._state.total_inconclusive += 1

        self._save()
        logger.info(
            f"[AutoExperiment] {'✅' if outcome=='confirmed' else '❌' if outcome=='disconfirmed' else '❓'} "
            f"{outcome.upper()}: '{exp.hypothesis[:60]}' "
            f"Δconf={confidence_delta:+.3f} "
            f"({len(measurements)} measurements, mean={exp.mean_measurement():.2f})"
        )

    def _compute_outcome(self, exp: Experiment) -> Tuple[str, float]:
        """Determine experiment outcome from measurements vs baseline."""
        if len(exp.measurements) < MIN_MEASUREMENTS_FOR_EVAL:
            return "inconclusive", INCONCLUSIVE_DELTA

        direction = exp.result_direction()
        target    = exp.target_value

        # Did the metric move in the expected direction by ≥ 15%?
        if target > 0 and direction >= 0.15:
            return "confirmed", CONFIRMED_CONFIDENCE_BOOST
        elif target < 0 and direction <= -0.15:
            return "confirmed", CONFIRMED_CONFIDENCE_BOOST
        elif target > 0 and direction < -0.10:
            return "disconfirmed", DISCONFIRMED_CONFIDENCE_DELTA
        elif target < 0 and direction > 0.10:
            return "disconfirmed", DISCONFIRMED_CONFIDENCE_DELTA
        else:
            return "inconclusive", INCONCLUSIVE_DELTA

    def _apply_experiment_result(self, exp: Experiment) -> None:
        """Update beliefs and store memory based on experiment outcome."""
        # Update related belief in self_concept
        try:
            ai = getattr(self._o, "ai_system", None)
            sc = getattr(ai, "self_concept", None) if ai else None
            if sc and exp.hypothesis_belief:
                beliefs = getattr(sc, "_beliefs", {})
                for bname, bobj in beliefs.items():
                    if exp.hypothesis_belief.lower() in bname.lower():
                        old_conf = bobj.confidence
                        bobj.confidence = max(0.10, min(0.95,
                            old_conf + exp.confidence_delta
                        ))
                        if exp.outcome == "confirmed":
                            bobj.affirmations += 1
                        else:
                            bobj.violations += 1
                        bobj.last_tested = time.time()
                        break
        except Exception:
            pass

        # Store to memory
        try:
            ai = getattr(self._o, "ai_system", None)
            ms = getattr(ai, "memory_system", None) if ai else None
            if ms:
                summary = (
                    f"[Experiment {exp.outcome}] Hypothesis: '{exp.hypothesis}' "
                    f"Metric={exp.metric}, mean={exp.mean_measurement():.2f} "
                    f"vs baseline={exp.baseline:.2f}. "
                    f"Δconfidence={exp.confidence_delta:+.3f}"
                )
                ms.add_memory(
                    summary,
                    0.82 if exp.outcome == "confirmed" else 0.70,
                    "experiment", "Positive" if exp.outcome == "confirmed" else "Neutral",
                    "Low", memory_tier="cognitive"
                )
        except Exception:
            pass

        # Broadcast to workspace
        try:
            ws = getattr(self._o, "workspace", None)
            if ws:
                ws.broadcast(
                    source   = f"autoexperiment.{exp.outcome}",
                    content  = (
                        f"[Experiment {exp.outcome}] {exp.hypothesis[:80]} "
                        f"(Δ={exp.confidence_delta:+.3f})"
                    ),
                    priority = 0.52,
                )
        except Exception:
            pass

    # ── Measurement ───────────────────────────────────────────────────────────

    def _measure(
        self,
        metric:     str,
        user_input: str,
        response:   str,
        context:    Dict,
    ) -> float:
        """Measure the relevant metric from a single interaction."""
        if metric == "response_length":
            return len(response.split())

        elif metric == "question_rate":
            # Fraction of response sentences that are questions
            sentences = [s.strip() for s in response.split(".") if s.strip()]
            questions = [s for s in sentences if "?" in s]
            return len(questions) / max(1, len(sentences))

        elif metric == "example_rate":
            example_markers = ["for example", "such as", "for instance",
                               "like ", "e.g.", "consider "]
            return sum(1 for m in example_markers if m in response.lower())

        elif metric == "preference_surfaces":
            return 1.0 if "[Preference" in response or "consistently orients" in response else 0.0

        elif metric == "session_turns":
            return float(context.get("session_turn_count", 1))

        return 0.0

    def _measure_baseline(self, metric: str) -> float:
        """Estimate baseline metric from recent interactions."""
        log = getattr(self._state, "experiment_log", [])
        measurements = []
        for exp in log[-5:]:
            if exp.get("metric") == metric and exp.get("measurements"):
                measurements.extend(exp["measurements"][-3:])
        if measurements:
            return sum(measurements) / len(measurements)
        # Heuristic defaults
        defaults = {
            "response_length": 180.0,
            "question_rate":    0.15,
            "example_rate":     0.80,
            "preference_surfaces": 0.10,
            "session_turns":    3.0,
        }
        return defaults.get(metric, 1.0)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _get_policy_summary(self) -> str:
        """Return top active values string for experiment prompt injection."""
        try:
            from core.state import state as _st
            _org    = getattr(getattr(_st, 'persona', None), '_organism', None)
            _policy = getattr(getattr(_org, 'ai_system', None), '_decision_policy', None)
            if _policy:
                top = _policy._top_values(3)
                return ", ".join(f"{k}({v:.2f})" for k, v in top)
        except Exception:
            pass
        return ""

    def _llm_call(self, prompt: str, caller: str = "autoexp") -> Optional[str]:
        from core.llm_scheduler import llm_scheduler
        result = ""
        with llm_scheduler.sync_slot(
            priority    = 3,
            skip_if_busy= True,
            caller      = caller,
        ) as acquired:
            if not acquired:
                return None
            try:
                result = self._ai.get_response(
                    messages   = [{"role": "user", "content": prompt}],
                    max_tokens = 180,
                    temperature= 0.70,
                ) or ""
            except Exception as e:
                logger.debug(f"[AutoExperiment] LLM failed: {e}")
        return result or None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = ExperimentState(
                    active_experiment  = data.get("active_experiment"),
                    experiment_log     = data.get("experiment_log",    []),
                    total_experiments  = data.get("total_experiments", 0),
                    total_confirmed    = data.get("total_confirmed",   0),
                    total_disconfirmed = data.get("total_disconfirmed",0),
                    total_inconclusive = data.get("total_inconclusive",0),
                    current_overrides  = data.get("current_overrides", {}),
                )
        except Exception as e:
            logger.warning(f"[AutoExperiment] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os; os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[AutoExperiment] Save failed: {e}")
