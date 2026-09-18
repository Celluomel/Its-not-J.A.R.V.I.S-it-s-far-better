"""
cognition/long_horizon_planner.py  (v61)

Generates concrete action sequences toward aspirations.
Uses WSDM + CausalMechanismModel as transition function.

Unlike TemporalProjection (archetype selection over 20 interactions),
the LongHorizonPlanner:
    - Sequences specific action types step-by-step
    - Reserves resources for later steps before spending them
    - Revises the plan when early steps produce unexpected outcomes
    - Uses causal signatures to choose the *right* action given current context

Plan structure:
    Plan(
        aspiration_domain: str,
        steps: [PlanStep(action_type, expected_cost, expected_world_gain)],
        resource_reservation: Dict[str, float],  # energy held back
        revision_trigger: float,  # deviation that triggers replanning
        horizon: int,  # interactions
    )

Planning algorithm (simplified STRIPS-style):
    1. Goal state: aspiration.state_desired (target tension reduction)
    2. Current state: full 8-dim context from WSDM
    3. Available operators: action types with causal signatures
    4. For each step in horizon:
           Select action_type that maximises:
               world_gain * causal_relevance[user_trust] * trust_context
               - cost * causal_relevance[cognitive_energy]
           Reserve resources: don't spend below RESOURCE_FLOOR
           Add step to plan
    5. If actual outcome deviates > revision_trigger from expected: replan

Resource reservation: before executing any plan step, the planner
checks whether proceeding would leave enough resources for later steps.
If a deep_reasoning step in position 3 would drain cognitive_energy
below what position 5 (also deep_reasoning) needs, the planner either
reorders the steps or inserts a social/emergent step to allow recovery.
This is the architectural feature that TemporalProjection lacks.
"""

from __future__ import annotations
import json, logging, re, threading, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from managers.settings_manager import get_persona_name

logger = logging.getLogger(__name__)

PLAN_EVERY_N     = 70     # slow cycles between planning passes
HORIZON          = 30     # interactions
RESOURCE_FLOOR   = 0.22   # minimum cognitive_energy before inserting recovery step
REVISION_TRIGGER = 0.25   # deviation fraction that triggers replanning
SAVE_PATH        = "data/persona/long_horizon_plans.json"

RECOVERY_ACTIONS = {"social", "creative"}   # low-cost, allow energy recovery


@dataclass
class PlanStep:
    sequence:      int
    action_type:   str
    expected_cost: float          # cognitive_energy delta
    expected_gain: float          # world_delta (trust proxy)
    recovery_step: bool = False   # True if inserted for resource recovery
    operation:      str = "self_question"
    status:         str = "pending"
    attempts:       int = 0
    result:         str = ""


@dataclass
class Plan:
    aspiration_domain: str
    steps:             List[PlanStep]
    resource_reserve:  float         # energy floor maintained
    revision_trigger:  float
    horizon:           int
    created_at:        float = field(default_factory=time.time)
    last_revised:      float = field(default_factory=time.time)
    steps_completed:   int   = 0
    outcome_deviations: List[float] = field(default_factory=list)
    objective:         str = ""
    source:            str = "aspiration"
    source_id:         str = ""
    success_criteria:  str = "Generate and verify the planned cognitive steps"
    status:            str = "active"
    reasoning_episode_id: str = ""
    decision_rationale:   str = ""
    reasoning_confidence: float = 0.0
    uncertainties:        List[str] = field(default_factory=list)
    current_decision:     str = ""
    decision_revision:    int = 0
    decision_updated_at:  float = 0.0
    last_outcome:         Dict[str, Any] = field(default_factory=dict)


class LongHorizonPlanner:
    def __init__(self, organism: Any, ai_system: Any, path: str = SAVE_PATH):
        self._organism = organism
        self._ai       = ai_system
        self._path     = self._resolve_path(path)
        self._lock     = threading.Lock()
        self._plans:   Dict[str, Plan] = {}
        self._reasoner = None
        self._self_report_grounding_until = 0.0
        self._load()
        self._synchronize_goal_plans()
        active_count = sum(
            1 for plan in self._plans.values() if plan.status == "active"
        )
        logger.info(
            "[LongHorizonPlanner] Init — %d active plans (%d stored)",
            active_count,
            len(self._plans),
        )

    @staticmethod
    def _resolve_path(path: str) -> Path:
        """Keep independent persona processes from overwriting one plan file."""
        if path != SAVE_PATH:
            return Path(path)
        persona = re.sub(r"[^a-z0-9]+", "_", get_persona_name().lower()).strip("_") or "lumina"
        scoped = Path("data/persona") / f"long_horizon_plans_{persona}.json"
        legacy = Path(path)
        if persona == "lumina" and not scoped.exists() and legacy.exists():
            try:
                scoped.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
                logger.info("[LongHorizonPlanner] migrated legacy plan store to %s", scoped)
            except Exception as exc:
                logger.warning("[LongHorizonPlanner] plan store migration failed: %s", exc)
        return scoped

    def tick(self, slow_cycle: int) -> None:
        self._synchronize_goal_plans()
        # Aspiration plans must remain autonomous.  A recovery step labelled
        # ``social`` is an internal perspective-taking/prosociality probe, not
        # a request to wait indefinitely for a user message.  Execute it from
        # the background planner and use a real chat turn only as additional
        # evidence when one happens to be available.
        self._execute_autonomous_social_step()
        # Goal plans used to be created only as a side effect of the action
        # executor selecting a goal. That made autonomous goals invisible to
        # planning until after an action had already happened. Prepare the
        # highest-priority live goal in the background so the chain is visible
        # and persistent even between user turns.
        goal_engine = self._goal_engine()
        if goal_engine is not None:
            try:
                active_goals = goal_engine.get_active_goals(min_activation=0.10) or []
                missing = [
                    goal for goal in active_goals[:2]
                    if f"goal:{getattr(goal, 'id', '')}" not in self._plans
                ]
                if missing:
                    threading.Thread(
                        target=self._prepare_goal_plans,
                        args=(missing,),
                        daemon=True,
                        name="lhp-goal-plan",
                    ).start()
            except Exception as exc:
                logger.debug("[LongHorizonPlanner] goal plan preparation unavailable: %s", exc)
        has_aspiration_plan = any(
            plan.source == "aspiration" and plan.status == "active"
            for plan in self._plans.values()
        )
        # Bootstrap an empty/legacy store immediately. Once an aspiration plan
        # exists, retain the deliberately slow long-horizon planning cadence.
        if slow_cycle == 0 or (has_aspiration_plan and slow_cycle % PLAN_EVERY_N != 0):
            return
        threading.Thread(target=self._run, daemon=True, name="lhp-plan").start()

    def _execute_autonomous_social_step(self) -> None:
        """Complete one pending autonomous social/recovery step locally."""
        asp = getattr(self._organism, "aspirational_self", None)
        if not asp:
            return
        candidates: List[Tuple[float, str]] = []
        for domain, plan in self._plans.items():
            if plan.source != "aspiration" or plan.status != "active":
                continue
            step = self.current_step(domain)
            if step is None or (
                step.operation != "user_question" and step.action_type != "social"
            ):
                continue
            aspiration = next(
                (item for item in asp.aspirations.values()
                 if getattr(item, "domain", None) == domain),
                None,
            )
            candidates.append((float(getattr(aspiration, "tension", 0.0)), domain))
        if not candidates:
            return
        _, domain = max(candidates)
        plan = self._plans[domain]
        step = self.current_step(domain)
        self._broadcast(
            "long_horizon_planner.social_probe",
            f"[AutonomousSocialProbe] objective={plan.objective[:80]!r}; "
            f"operation={step.operation if step else 'social'}; no user turn required",
            0.55,
        )
        self.record_step_outcome(domain, actual_cost=0.02, actual_gain=0.005)
        plan = self._plans.get(domain)
        if plan:
            plan.last_outcome = {
                "operation": step.operation if step else "social",
                "success": True,
                "result": "autonomous internal social perspective probe",
                "observed_at": time.time(),
            }
            plan.decision_updated_at = time.time()
            self._save()

    def _prepare_goal_plans(self, goals: List[Any]) -> None:
        """Materialize autonomous goal plans without holding the slow-loop lock."""
        for goal in goals:
            try:
                self.ensure_goal_plan(goal)
            except Exception as exc:
                logger.debug(
                    "[LongHorizonPlanner] could not prepare plan for %s: %s",
                    getattr(goal, "topic", "goal"), exc,
                )

    def current_step(self, aspiration_domain: str) -> Optional[PlanStep]:
        """Return the next planned step for a given aspiration."""
        plan = self._plans.get(aspiration_domain)
        if not plan: return None
        idx = plan.steps_completed
        if idx < len(plan.steps):
            return plan.steps[idx]
        return None

    def ensure_goal_plan(self, goal: Any) -> Optional[Plan]:
        """Create or return the persistent intention plan for a live goal."""
        goal_id = str(getattr(goal, "id", ""))
        if not goal_id or str(getattr(goal, "status", "active")) != "active":
            return None
        key = f"goal:{goal_id}"
        plan = self._plans.get(key)
        if plan and plan.status == "active" and plan.steps_completed < len(plan.steps):
            return plan

        origin = str(getattr(goal, "origin", "") or "explore").lower()
        topic = str(getattr(goal, "topic", "") or "an unresolved goal")
        operations = self._operations_for_goal(origin, topic)
        episode = None
        try:
            reasoner = self._get_reasoner()
            episode = reasoner.deliberate(goal, operations) if reasoner else None
            if episode is not None:
                operations = episode.recommended_sequence
        except Exception as exc:
            logger.debug("[LongHorizonPlanner] intention reasoning unavailable: %s", exc)
        steps = [
            PlanStep(
                sequence=i,
                action_type=self._style_for_operation(operation),
                operation=operation,
                expected_cost=0.03 if operation in {"memory_recall", "user_question"} else 0.06,
                expected_gain=0.02 if operation == "memory_recall" else 0.05,
            )
            for i, operation in enumerate(operations)
        ]
        plan = Plan(
            aspiration_domain=key,
            steps=steps,
            resource_reserve=RESOURCE_FLOOR,
            revision_trigger=REVISION_TRIGGER,
            horizon=len(steps),
            objective=topic,
            source="goal",
            source_id=goal_id,
            success_criteria="A grounded insight or externally observable result is recorded",
            reasoning_episode_id=episode.id if episode else "",
            decision_rationale=episode.decision_rationale if episode else "",
            reasoning_confidence=episode.confidence if episode else 0.0,
            uncertainties=list(episode.uncertainties) if episode else [],
            current_decision=episode.decision_rationale if episode else "",
            decision_updated_at=time.time(),
        )
        with self._lock:
            self._plans[key] = plan
        self._save()
        logger.info(
            "[LongHorizonPlanner] Intention committed for goal %s: %s",
            goal_id, " -> ".join(operations),
        )
        self._broadcast(
            "long_horizon_planner",
            f"[PlanCommitted] goal='{topic[:60]}'; steps={' -> '.join(operations)}; "
            f"confidence={plan.reasoning_confidence:.2f}",
            0.60,
        )
        return plan

    def next_action_for_goal(self, goal: Any, available: Optional[List[str]] = None) -> Optional[str]:
        plan = self.ensure_goal_plan(goal)
        if not plan:
            return None
        step = self.current_step(plan.aspiration_domain)
        if not step:
            return None
        if available is not None and step.operation not in available:
            return None
        return step.operation

    def record_goal_action_outcome(
        self, goal_id: str, action_type: str, success: bool, result: str = ""
    ) -> None:
        """Advance a goal intention only after the operation really completed."""
        key = f"goal:{goal_id}"
        plan = self._plans.get(key)
        if not plan or plan.status != "active":
            return
        step = self.current_step(key)
        if not step or step.operation != action_type:
            return

        step.attempts += 1
        step.result = str(result)[:240]
        episode = None
        if plan.reasoning_episode_id:
            reasoner = self._get_reasoner()
            if reasoner is not None:
                episode = reasoner.record_outcome(
                    plan.reasoning_episode_id, action_type, success, result
                )
        revised = False
        if success:
            step.status = "completed"
            plan.steps_completed += 1
            if plan.steps_completed >= len(plan.steps):
                plan.status = "completed"
        else:
            step.status = "failed"
            plan.outcome_deviations.append(1.0)
            if step.attempts < 2:
                step.status = "pending"
            else:
                # Preserve completed work and move the repeatedly failing step
                # behind the remaining alternatives before trying it again.
                remaining = plan.steps[plan.steps_completed:]
                if len(remaining) > 1:
                    remaining.append(remaining.pop(0))
                    for offset, item in enumerate(remaining, plan.steps_completed):
                        item.sequence = offset
                        item.status = "pending"
                    plan.steps[plan.steps_completed:] = remaining
                plan.last_revised = time.time()
                revised = True
        next_step = self.current_step(key)
        state = "completed" if success else ("revised" if revised else "retry_pending")
        result_summary = str(result).strip()[:160] or "no result detail"
        if success and plan.status == "completed":
            current_decision = (
                f"{action_type} succeeded ({result_summary}). Decision: the plan is complete."
            )
        elif success:
            current_decision = (
                f"{action_type} succeeded ({result_summary}). Decision: preserve the sequence "
                f"and proceed with {next_step.operation if next_step else 'the next pending step'}."
            )
        elif revised:
            current_decision = (
                f"{action_type} failed after {step.attempts} attempts ({result_summary}). "
                f"Decision revised: defer it and proceed with "
                f"{next_step.operation if next_step else 'another available operation'}."
            )
        else:
            current_decision = (
                f"{action_type} failed once ({result_summary}). Decision: retry it once before "
                "changing the remaining sequence."
            )
        uncertainties = [
            item for item in plan.uncertainties
            if action_type not in item
        ]
        if not success:
            uncertainties.append(
                f"{action_type} was deferred after repeated failure"
                if revised else f"{action_type} has failed once and remains pending retry"
            )
        plan.current_decision = current_decision
        plan.decision_revision += 1
        plan.decision_updated_at = time.time()
        plan.last_outcome = {
            "operation": action_type,
            "success": bool(success),
            "result": result_summary,
            "observed_at": plan.decision_updated_at,
        }
        if episode is not None:
            plan.reasoning_confidence = episode.confidence
            plan.uncertainties = uncertainties
            reasoner.update_decision(
                plan.reasoning_episode_id,
                current_decision,
                uncertainties,
                "validated" if success else ("revised" if revised else "revision_needed"),
            )
        else:
            plan.uncertainties = uncertainties
        self._save()
        self._broadcast(
            "long_horizon_planner.outcome",
            f"[PlanStep] goal='{plan.objective[:60]}'; operation={action_type}; "
            f"status={state}; progress={plan.steps_completed}/{len(plan.steps)}",
            0.53 if success else 0.68,
        )
        if revised:
            self._broadcast(
                "long_horizon_planner.revision",
                f"[PlanRevised] goal='{plan.objective[:60]}'; failed={action_type}; "
                f"next={next_step.operation if next_step else 'none'}",
                0.66,
            )

    @staticmethod
    def _operations_for_goal(origin: str, topic: str) -> List[str]:
        combined = f"{origin} {topic}".lower()
        if "correct" in combined or "repair" in combined:
            return ["memory_recall", "self_question", "web_search", "self_question"]
        if "project" in combined or "future" in combined or "aspiration" in combined:
            return ["memory_recall", "bisociative_hypothesis", "user_question", "web_search"]
        if any(token in combined for token in ("user", "social", "connect", "relational", "help")):
            return ["memory_recall", "user_question", "self_question"]
        if "strengthen" in combined or "consolidat" in combined:
            return ["memory_recall", "bisociative_hypothesis", "self_question", "web_search"]
        if "become" in combined or "identity" in combined:
            return ["memory_recall", "self_question", "bisociative_hypothesis", "user_question"]
        if any(token in combined for token in ("uncertain", "research", "learn", "understand", "explore")):
            return ["memory_recall", "self_question", "bisociative_hypothesis", "web_search"]
        return ["memory_recall", "self_question", "bisociative_hypothesis"]

    @staticmethod
    def _style_for_operation(operation: str) -> str:
        return {
            "memory_recall": "analytical",
            "self_question": "philosophical",
            "bisociative_hypothesis": "creative",
            "web_search": "technical",
            "user_question": "social",
        }.get(operation, "analytical")

    def _get_reasoner(self):
        if self._reasoner is not None:
            return self._reasoner
        try:
            from cognition.intention_reasoner import IntentionReasoner
            reasoning_path = self._path.parent / "reasoning_episodes.json"
            self._reasoner = IntentionReasoner(
                self._organism, path=str(reasoning_path)
            )
        except Exception as exc:
            logger.debug("[LongHorizonPlanner] reasoner init failed: %s", exc)
            self._reasoner = False
        return self._reasoner or None

    def _broadcast(self, source: str, content: str, priority: float) -> None:
        """Publish compact plan state without bypassing workspace admission gates."""
        try:
            workspace = getattr(self._organism, "workspace", None)
            if workspace is not None and hasattr(workspace, "broadcast"):
                workspace.broadcast(
                    source=source,
                    content=content[:240],
                    priority=max(0.0, min(0.75, priority)),
                )
        except Exception as exc:
            logger.debug("[LongHorizonPlanner] workspace broadcast failed: %s", exc)

    def record_step_outcome(self, aspiration_domain: str,
                             actual_cost: float, actual_gain: float) -> None:
        """Update plan with actual outcome; trigger revision if needed."""
        plan = self._plans.get(aspiration_domain)
        if not plan: return
        idx = plan.steps_completed
        if idx >= len(plan.steps): return
        step = plan.steps[idx]
        deviation = abs(actual_cost - step.expected_cost) + \
                    abs(actual_gain - step.expected_gain)
        plan.outcome_deviations.append(deviation)
        step.status = "completed"
        plan.steps_completed += 1
        # A terminal outcome is a completion, not a reason to replan.  The
        # previous code advanced the counter but left the plan active forever,
        # which made the UI remain at 14/15 (or show an active 15/15 plan).
        if plan.steps_completed >= len(plan.steps):
            plan.status = "completed"
            plan.current_decision = (
                f"{step.operation} completed; the planned causal investigation "
                "reached its final step."
            )
            plan.decision_updated_at = time.time()
            self._broadcast(
                "long_horizon_planner.completed",
                f"[PlanCompleted] domain={aspiration_domain}; "
                f"progress={plan.steps_completed}/{len(plan.steps)}",
                0.66,
            )
        elif deviation > REVISION_TRIGGER and len(plan.outcome_deviations) >= 2:
            logger.info(f"[LongHorizonPlanner] Replanning {aspiration_domain} "
                        f"(deviation={deviation:.2f})")
            threading.Thread(target=self._replan,
                             args=(aspiration_domain,), daemon=True).start()
            self._broadcast(
                "long_horizon_planner.revision",
                f"[PlanRevisionRequested] domain={aspiration_domain}; "
                f"deviation={deviation:.2f}",
                0.66,
            )
        else:
            self._broadcast(
                "long_horizon_planner.outcome",
                f"[PlanStep] domain={aspiration_domain}; operation={step.operation}; "
                f"deviation={deviation:.2f}; progress={plan.steps_completed}/{len(plan.steps)}",
                0.52,
            )
        self._save()

    def record_interaction_outcome(self, result: str = "user interaction completed") -> None:
        """Use a real user turn to complete a pending social plan step.

        Aspiration plans are autonomous, so they are not owned by a GoalActionExecutor.
        Their ``social``/``user_question`` recovery step nevertheless needs a real-world
        observation.  A completed chat turn is that observation; this method deliberately
        advances only the current social step of the highest-tension aspiration.
        """
        asp = getattr(self._organism, "aspirational_self", None)
        if not asp:
            return
        candidates: List[Tuple[float, str]] = []
        for domain, plan in self._plans.items():
            if plan.source != "aspiration" or plan.status != "active":
                continue
            step = self.current_step(domain)
            if step is None or (step.operation != "user_question" and step.action_type != "social"):
                continue
            aspiration = next(
                (item for item in asp.aspirations.values()
                 if getattr(item, "domain", None) == domain),
                None,
            )
            candidates.append((float(getattr(aspiration, "tension", 0.0)), domain))
        if not candidates:
            return
        _, domain = max(candidates)
        self.record_step_outcome(domain, actual_cost=0.02, actual_gain=0.005)
        plan = self._plans.get(domain)
        if plan:
            plan.last_outcome = {
                "operation": "user_question",
                "success": True,
                "result": str(result)[:240],
                "observed_at": time.time(),
            }
            if plan.status != "completed":
                plan.current_decision = (
                    "user interaction completed the social step; "
                    f"proceed to {self.current_step(domain).operation if self.current_step(domain) else 'completion'}."
                )
            plan.decision_updated_at = time.time()
            self._save()

    def record_autonomous_action_outcome(
        self, operation: str, success: bool, result: str = ""
    ) -> None:
        """Resolve the current aspiration step from a completed autonomous action.

        Aspiration plans are not owned by a user goal, so the goal executor's
        goal-specific callback cannot advance them. Match only the current
        pending operation and advance one best-tension aspiration; never mark
        an unrelated or already-completed step as done.
        """
        asp = getattr(self._organism, "aspirational_self", None)
        if not asp:
            return
        candidates = []
        for domain, plan in self._plans.items():
            if plan.source != "aspiration" or plan.status != "active":
                continue
            step = self.current_step(domain)
            # GoalActionExecutor reports the coarse action type (for example
            # ``social``), while aspiration steps retain the concrete
            # operation (for example ``user_question``). Accept both names
            # so a valid completed action can advance the persisted plan.
            if step is None or (
                step.operation != operation and step.action_type != operation
            ):
                continue
            aspiration = next(
                (a for a in asp.aspirations.values()
                 if getattr(a, "domain", None) == domain), None
            )
            candidates.append((float(getattr(aspiration, "tension", 0.0)), domain))
        if not candidates:
            return
        _, domain = max(candidates)
        self.record_step_outcome(
            domain,
            actual_cost=0.05 if success else 0.08,
            actual_gain=0.05 if success else 0.0,
        )
        plan = self._plans.get(domain)
        if plan:
            plan.last_outcome = {
                "operation": operation,
                "success": bool(success),
                "result": str(result)[:240],
                "observed_at": time.time(),
            }
            plan.current_decision = (
                f"{operation} {'succeeded' if success else 'failed'}; "
                f"aspiration step {'advanced' if success else 'retained for revision'}."
            )
            plan.decision_updated_at = time.time()
            self._save()

    def preferred_next_action(self) -> Optional[str]:
        """Action type the planner most wants to execute next."""
        asp = getattr(self._organism, 'aspirational_self', None)
        if not asp: return None
        best_tension = 0.0
        best_action  = None
        for domain, plan in self._plans.items():
            if plan.source != "aspiration":
                continue
            a = next((item for item in asp.aspirations.values()
                      if getattr(item, "domain", None) == domain), None)
            tension = getattr(a, 'tension', 0.0) if a else 0.0
            if tension > best_tension:
                step = self.current_step(domain)
                if step:
                    best_tension = tension
                    best_action  = step.action_type
        return best_action

    def status(self) -> Dict:
        self._synchronize_goal_plans()
        active = [p for p in self._plans.values() if p.status == "active"]
        return {
            "active_plans": len(active),
            "plans": [
                {"domain": d,
                 "steps": len(p.steps),
                 "completed": p.steps_completed,
                 "objective": p.objective,
                 "reasoning_confidence": p.reasoning_confidence,
                 "uncertainties": p.uncertainties,
                 "next": self.current_step(d).action_type
                         if self.current_step(d) else "done"}
                for d, p in self._plans.items()
            ],
        }

    def persist_now(self) -> None:
        """Flush the latest in-memory plan before the host process exits."""
        self._save()

    def prompt_fragment(self) -> str:
        return self.prompt_fragment_for("")

    @staticmethod
    def is_self_report_query(user_input: str) -> bool:
        text = " ".join(str(user_input or "").lower().split())
        if not text:
            return False
        ownership_patterns = (
            r"\byour\s+(?:(?:actual|current|real|main|own|active)\s+)?goals?\b",
            r"\byour\s+(?:(?:actual|current|real|main|own|active)\s+)?plans?\b",
            r"\b(?:ton|votre)\s+(?:(?:veritable|véritable|actuel|réel|reel|principal|propre)\s+)?(?:but|objectif|plan)s?\b",
            r"\b(?:quel(?:le)?s?\s+(?:est|sont)\s+)(?:ton|votre|tes|vos)\s+(?:but|objectif|plan)s?\b",
        )
        if any(re.search(pattern, text) for pattern in ownership_patterns):
            return True
        phrases = (
            "your capability", "your capabilities", "your goal", "your goals",
            "your plan", "your planning", "your reasoning", "how do you reason",
            "how are you reasoning", "what are you planning", "what is your plan",
            "set goals", "plan for it", "your internal state", "your cognitive state",
            "factual elements", "not pure invention", "observable data",
        )
        return any(phrase in text for phrase in phrases)

    def should_ground_self_report(self, user_input: str) -> bool:
        text = " ".join(str(user_input or "").lower().split())
        explicit = self.is_self_report_query(text)
        short_follow_up = bool(text) and len(text.split()) <= 5
        continuing = time.time() < self._self_report_grounding_until and short_follow_up
        if explicit or continuing:
            self._self_report_grounding_until = time.time() + 300.0
            return True
        return False

    def prompt_fragment_for(self, user_input: str) -> str:
        """Return ordinary plan guidance or strict telemetry for self-reports."""
        if self.should_ground_self_report(user_input):
            return self.factual_self_report_fragment(user_input)
        action = self.preferred_next_action()
        if not action: return ""
        return f"[Plan] Next planned action type: {action}."

    def factual_self_report_snapshot(self) -> Dict[str, Any]:
        """Return inspectable state facts without prompt instructions or hidden reasoning."""
        self._synchronize_goal_plans()
        active_goals: List[Any] = []
        try:
            goal_engine = self._goal_engine()
            if goal_engine is not None:
                active_goals = goal_engine.get_active_goals(min_activation=0.10) or []
        except Exception:
            active_goals = []

        goal_topics = [
            str(getattr(goal, "topic", "")).strip()[:80]
            for goal in active_goals if str(getattr(goal, "topic", "")).strip()
        ]
        active_ids = {str(getattr(goal, "id", "")) for goal in active_goals}
        goal_plans = [
            plan for plan in self._plans.values()
            if plan.source == "goal" and plan.status == "active"
            and plan.source_id in active_ids
        ]
        plans_by_goal = {plan.source_id: plan for plan in goal_plans}
        plan = next(
            (
                plans_by_goal[str(getattr(goal, "id", ""))]
                for goal in active_goals
                if str(getattr(goal, "id", "")) in plans_by_goal
            ),
            None,
        )
        if plan is None and goal_plans:
            plan = max(goal_plans, key=lambda item: item.created_at)

        goals = [
            {
                "id": str(getattr(goal, "id", "")),
                "topic": str(getattr(goal, "topic", "")).strip()[:120],
                "origin": str(getattr(goal, "origin", "unknown")),
                "intent_kind": str(getattr(goal, "intent_kind", "learn")),
                "temporal_scope": str(getattr(goal, "temporal_scope", "ongoing")),
                "desired_effect": str(getattr(goal, "desired_effect", ""))[:180],
                "success_criteria": str(getattr(goal, "success_criteria", ""))[:180],
                "priority": round(float(getattr(goal, "priority", 0.0)), 3),
                "energy": round(float(getattr(goal, "energy", 0.0)), 3),
                "activation": round(
                    float(getattr(goal, "priority", 0.0))
                    * float(getattr(goal, "energy", 0.0)), 3
                ),
            }
            for goal in active_goals[:4]
            if str(getattr(goal, "topic", "")).strip()
        ]
        plan_data = None
        if plan is not None:
            completed = [step.operation for step in plan.steps if step.status == "completed"]
            next_step = self.current_step(plan.aspiration_domain)
            plan_data = {
                "objective": plan.objective[:160],
                "status": plan.status,
                "steps_completed": plan.steps_completed,
                "steps_total": len(plan.steps),
                "completed_operations": completed,
                "next_operation": next_step.operation if next_step else None,
                "reasoning_confidence": round(plan.reasoning_confidence, 3),
                "uncertainties": list(plan.uncertainties[:3]),
                "initial_decision": (plan.decision_rationale or "")[:240],
                "decision_summary": (
                    plan.current_decision or plan.decision_rationale or ""
                )[:500],
                "decision_revision": plan.decision_revision,
                "decision_updated_at": plan.decision_updated_at or plan.created_at,
                "last_outcome": dict(plan.last_outcome),
            }
        return {
            "captured_at": time.time(),
            "active_goals": goals,
            "dominant_plan": plan_data,
        }

    def _goal_engine(self) -> Any:
        """Return the live goal engine regardless of planner construction order."""
        organism_ai = getattr(self._organism, "ai_system", None)
        return (
            getattr(organism_ai, "goal_engine", None)
            or getattr(self._ai, "goal_engine", None)
        )

    def _synchronize_goal_plans(self, persist: bool = True) -> int:
        """Close plans whose source goals are no longer active."""
        goal_engine = self._goal_engine()
        get_goal = getattr(goal_engine, "get_goal_by_id", None)
        if not callable(get_goal):
            return 0

        transitions: List[Tuple[str, str]] = []
        now = time.time()
        with self._lock:
            for key, plan in self._plans.items():
                if plan.source != "goal" or plan.status != "active":
                    continue
                goal = get_goal(plan.source_id)
                goal_status = str(
                    getattr(goal, "status", "active") if goal is not None else "abandoned"
                ).lower()
                if goal_status == "active":
                    continue
                if goal_status not in {"completed", "dormant", "abandoned", "archived"}:
                    goal_status = "inactive"
                plan.status = goal_status
                plan.last_revised = now
                plan.current_decision = (
                    f"Plan closed because its source goal is {goal_status}."
                )
                plan.decision_updated_at = now
                transitions.append((key, goal_status))

        if transitions and persist:
            self._save()
        if transitions:
            logger.info(
                "[LongHorizonPlanner] Closed %d stale goal plans: %s",
                len(transitions),
                ", ".join(f"{key}={status}" for key, status in transitions[:8]),
            )
        return len(transitions)

    def factual_self_report_fragment(self, user_input: str = "") -> str:
        snapshot = self.factual_self_report_snapshot()
        goals = snapshot["active_goals"]
        plan = snapshot["dominant_plan"]
        goal_topics = [str(goal["topic"]) for goal in goals]

        lines = ["[Cognitive telemetry - factual self-report]"]
        lines.append(
            "Recorded active goals: " + ("; ".join(goal_topics[:4]) if goal_topics else "none")
        )
        if not plan:
            lines.append("Operational plan: none recorded for an active goal.")
        else:
            completed = plan["completed_operations"]
            lines.append(
                f"Dominant active internal plan (it may be independent of the current "
                f"conversation): objective={plan['objective'][:100]!r}; progress "
                f"{plan['steps_completed']}/{plan['steps_total']}; completed="
                f"{','.join(completed) if completed else 'none'}; next pending="
                f"{plan['next_operation'] or 'none'}; "
                "pending operation is not currently executing."
            )
            uncertainty = plan["uncertainties"][0] if plan["uncertainties"] else "none recorded"
            lines.append(
                f"Reasoning: confidence={plan['reasoning_confidence']:.3f}; "
                f"rationale={(plan['decision_summary'] or 'none recorded')[:180]}; "
                f"uncertainty={uncertainty[:100]}."
            )
        lines.append(
            "Persisted aspirations are desired directions, not active goals or completed plans. "
            "Conversation topics and the human's tasks are not your own goals. When describing "
            "your cognition, use only this telemetry. Answer the question directly before any "
            "reflection. Do not invent steps, "
            "exchange horizons, mode switches, progress, or priorities; label new ideas as proposals."
        )
        return "\n".join(lines)

    def _run(self) -> None:
        try:
            asp = getattr(self._organism, 'aspirational_self', None)
            if not asp: return
            top = sorted(asp.aspirations.values(),
                         key=lambda a: getattr(a,'tension',0), reverse=True)[:3]
            for aspiration in top:
                domain = aspiration.domain
                # Replan if no plan or plan is exhausted
                plan = self._plans.get(domain)
                if not plan or plan.steps_completed >= len(plan.steps):
                    new_plan = self._build_plan(aspiration)
                    if new_plan:
                        with self._lock:
                            self._plans[domain] = new_plan
            self._save()
        except Exception as e:
            logger.debug(f"[LongHorizonPlanner] _run error: {e}")

    def _replan(self, domain: str) -> None:
        asp = getattr(self._organism, 'aspirational_self', None)
        if not asp: return
        aspiration = next((item for item in asp.aspirations.values()
                           if getattr(item, "domain", None) == domain), None)
        if not aspiration: return
        new_plan = self._build_plan(aspiration)
        if new_plan:
            new_plan.last_revised = time.time()
            with self._lock:
                self._plans[domain] = new_plan
            self._save()
            self._broadcast(
                "long_horizon_planner.revision",
                f"[PlanRevised] domain={domain}; steps={len(new_plan.steps)}; "
                f"next={new_plan.steps[0].operation if new_plan.steps else 'none'}",
                0.62,
            )

    def _build_plan(self, aspiration: Any) -> Optional[Plan]:
        """Build a step sequence using WSDM costs and causal signatures."""
        loop = getattr(self._organism, '_loop', None)
        wsdm = getattr(loop, '_world_self_dynamics', None) if loop else None
        cm   = getattr(loop, '_causal_mechanism', None) if loop else None
        if not wsdm: return None

        ctx = wsdm.read_state()
        steps: List[PlanStep] = []
        energy = ctx[0]   # track projected energy through plan

        for seq in range(min(HORIZON, 15)):
            # Select best action given projected state
            best_at, best_score, best_cost, best_gain = "analytical", 0.0, 0.05, 0.01

            for at in ["philosophical","deep_reasoning","analytical",
                       "social","creative","technical","emotional"]:
                # Get predicted costs from WSDM
                pred = wsdm.predict_from_context(at, ctx)
                cost = abs(pred.self_deltas.get("cognitive_energy", 0.05)) if pred else 0.05
                gain = pred.world_deltas.get("user_trust", 0.01) if pred else 0.01
                conf = pred.confidence if pred else 0.3

                # Weight by causal relevance if available
                causal_mult = 1.0
                if cm:
                    sig = cm.get_signature(at)
                    if sig:
                        causal_mult = 0.5 + sig.causal_relevance.get(
                            "user_trust", 0.5
                        ) * 0.5

                score = (gain * causal_mult * conf) / max(cost, 0.01)

                # Resource reservation: skip expensive actions if energy too low
                if energy - cost < RESOURCE_FLOOR and at not in RECOVERY_ACTIONS:
                    continue

                if score > best_score:
                    best_at, best_score = at, score
                    best_cost, best_gain = cost, gain

            # Insert recovery step if energy projected too low next step
            next_step_needs = 0.06   # average cost of next non-recovery step
            if energy - best_cost < RESOURCE_FLOOR + next_step_needs:
                steps.append(PlanStep(
                    sequence=seq, action_type="social",
                    expected_cost=0.02, expected_gain=0.005,
                    recovery_step=True, operation="user_question",
                ))
                energy += 0.04   # social steps allow recovery
            else:
                steps.append(PlanStep(
                    sequence=seq, action_type=best_at,
                    expected_cost=best_cost, expected_gain=best_gain,
                    operation=self._operation_for_style(best_at),
                ))
                energy = max(0.0, energy - best_cost)

        if not steps: return None
        return Plan(
            aspiration_domain = getattr(aspiration, 'domain', 'unknown'),
            steps             = steps,
            resource_reserve  = RESOURCE_FLOOR,
            revision_trigger  = REVISION_TRIGGER,
            horizon           = HORIZON,
            objective         = getattr(aspiration, "description", ""),
            source            = "aspiration",
            source_id         = getattr(aspiration, "domain", "unknown"),
        )

    @staticmethod
    def _operation_for_style(style: str) -> str:
        return {
            "analytical": "memory_recall",
            "philosophical": "self_question",
            "deep_reasoning": "self_question",
            "creative": "bisociative_hypothesis",
            "technical": "web_search",
            "social": "user_question",
            "emotional": "user_question",
        }.get(style, "self_question")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {d: asdict(p) for d, p in self._plans.items()}
                data["_meta"] = {"version":"v62","persona":get_persona_name(),"ts":time.time()}
            temp = self._path.with_suffix(".tmp")
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
            temp.replace(self._path)
        except Exception as e:
            logger.debug(f"[LongHorizonPlanner] save error: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            for d, p in data.items():
                if d.startswith("_"): continue
                try:
                    steps = [PlanStep(**s) for s in p.get("steps",[])]
                    self._plans[d] = Plan(
                        aspiration_domain=p["aspiration_domain"],
                        steps=steps,
                        resource_reserve=p.get("resource_reserve",RESOURCE_FLOOR),
                        revision_trigger=p.get("revision_trigger",REVISION_TRIGGER),
                        horizon=p.get("horizon",HORIZON),
                        created_at=p.get("created_at",time.time()),
                        last_revised=p.get("last_revised",time.time()),
                        steps_completed=p.get("steps_completed",0),
                        outcome_deviations=p.get("outcome_deviations",[]),
                        objective=p.get("objective", ""),
                        source=p.get("source", "aspiration"),
                        source_id=p.get("source_id", ""),
                        success_criteria=p.get("success_criteria", "Generate and verify the planned cognitive steps"),
                        status=p.get("status", "active"),
                        reasoning_episode_id=p.get("reasoning_episode_id", ""),
                        decision_rationale=p.get("decision_rationale", ""),
                        reasoning_confidence=p.get("reasoning_confidence", 0.0),
                        uncertainties=p.get("uncertainties", []),
                        current_decision=p.get(
                            "current_decision", p.get("decision_rationale", "")
                        ),
                        decision_revision=p.get("decision_revision", 0),
                        decision_updated_at=p.get(
                            "decision_updated_at", p.get("created_at", time.time())
                        ),
                        last_outcome=p.get("last_outcome", {}),
                    )
                except Exception: pass
        except Exception as e:
            logger.warning(f"[LongHorizonPlanner] load error: {e}")
