"""Regression tests for PandoraBOX's intention -> plan -> outcome loop."""
import tempfile
import threading
import unittest
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace as NS

# Keep this test independent of optional numerical/audio packages imported by
# cognition/__init__.py; the planning modules themselves are stdlib-only.
ROOT = Path(__file__).resolve().parents[1]
cognition_package = ModuleType("cognition")
cognition_package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", cognition_package)
core_package = ModuleType("core")
core_package.__path__ = [str(ROOT / "core")]
sys.modules.setdefault("core", core_package)
from cognition.long_horizon_planner import LongHorizonPlanner
from cognition.goal_action_executor import GoalActionExecutor
from cognition.goal_engine import Goal, GoalEngine, Motivation
from cognition.intention_reasoner import IntentionReasoner
from cognition.prompt_context_budget import PromptContextBudget
from core.planning.multi_step_planner import MultiStepPlanner
from managers.llm_manager import LLMManager


class _Prediction:
    def __init__(self, cost=0.04, gain=0.03, confidence=0.8):
        self.self_deltas = {"cognitive_energy": -cost}
        self.world_deltas = {"user_trust": gain}
        self.confidence = confidence


class _WorldModel:
    def __init__(self):
        self.contexts = []

    def read_state(self):
        return [0.9, 0.8, 0.8, 0.7, 0.5, 0.6, 0.2, 0.3]

    def predict_from_context(self, action_type, context):
        self.contexts.append((action_type, list(context)))
        return _Prediction()


class _Workspace:
    def __init__(self):
        self.events = []

    def broadcast(self, **event):
        self.events.append(event)
        return True


class IntentionPlanningTests(unittest.TestCase):
    def _planner(self, folder):
        world_model = _WorldModel()
        loop = NS(_world_self_dynamics=world_model, _causal_mechanism=None)
        aspiration = NS(
            domain="epistemic",
            description="Understand an unresolved question",
            tension=0.8,
        )
        organism = NS(
            _loop=loop,
            aspirational_self=NS(aspirations={"legacy-id": aspiration}),
            workspace=_Workspace(),
        )
        planner = LongHorizonPlanner(
            organism, NS(), path=str(Path(folder) / "plans.json")
        )
        return planner, world_model, aspiration

    def test_aspiration_plan_uses_public_context_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, world_model, aspiration = self._planner(folder)
            plan = planner._build_plan(aspiration)

            self.assertIsNotNone(plan)
            self.assertEqual(plan.source, "aspiration")
            self.assertTrue(plan.steps)
            self.assertTrue(world_model.contexts)
            self.assertTrue(all(step.operation for step in plan.steps))

    def test_empty_store_bootstraps_and_persists_aspiration_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            planner._run()

            self.assertEqual(planner.status()["active_plans"], 1)
            self.assertTrue((Path(folder) / "plans.json").exists())
            self.assertIn(
                planner.preferred_next_action(),
                {"analytical", "philosophical", "deep_reasoning", "social",
                 "creative", "technical", "emotional"},
            )

    def test_goal_intention_advances_only_on_verified_success(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(id="g-1", topic="understand causal planning", origin="resolve_uncertainty")
            plan = planner.ensure_goal_plan(goal)

            self.assertEqual(
                [step.operation for step in plan.steps],
                ["memory_recall", "self_question", "bisociative_hypothesis", "web_search"],
            )
            self.assertTrue(plan.reasoning_episode_id)
            self.assertGreater(plan.reasoning_confidence, 0.0)
            self.assertIn("expected gain", plan.decision_rationale)
            initial_decision = plan.decision_rationale
            initial_confidence = plan.reasoning_confidence
            planner.record_goal_action_outcome(
                goal.id, "memory_recall", False, "no memory"
            )
            self.assertEqual(plan.steps_completed, 0)
            self.assertEqual(planner.next_action_for_goal(goal), "memory_recall")
            self.assertIn("retry it once", plan.current_decision)
            self.assertEqual(plan.decision_revision, 1)
            self.assertNotEqual(plan.reasoning_confidence, initial_confidence)

            planner.record_goal_action_outcome(
                goal.id, "memory_recall", True, "recalled evidence"
            )
            self.assertEqual(plan.steps_completed, 1)
            self.assertEqual(planner.next_action_for_goal(goal), "self_question")
            episode = planner._get_reasoner().get(plan.reasoning_episode_id)
            self.assertEqual(episode.status, "validated")
            self.assertEqual(episode.outcomes[-1]["operation"], "memory_recall")
            self.assertIn("proceed with self_question", plan.current_decision)
            self.assertEqual(plan.decision_revision, 2)
            self.assertEqual(plan.decision_rationale, initial_decision)
            self.assertEqual(episode.decision_rationale, plan.current_decision)

    def test_reasoning_and_planning_feed_global_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(
                id="g-workspace", topic="understand workspace integration",
                origin="resolve_uncertainty", priority=0.8,
            )

            plan = planner.ensure_goal_plan(goal)
            created = planner._organism.workspace.events
            self.assertTrue(any(
                event["source"] == "intention_reasoner"
                and "[IntentionDecision]" in event["content"]
                for event in created
            ))
            self.assertTrue(any(
                event["source"] == "long_horizon_planner"
                and "[PlanCommitted]" in event["content"]
                for event in created
            ))
            self.assertTrue(all(0.0 <= event["priority"] <= 0.75 for event in created))

            planner._organism.workspace.events.clear()
            planner.record_goal_action_outcome(
                goal.id, "memory_recall", False, "no evidence"
            )
            outcomes = planner._organism.workspace.events
            self.assertEqual(plan.steps_completed, 0)
            self.assertTrue(any(
                event["source"] == "intention_reasoner.outcome"
                and "result=failure" in event["content"]
                for event in outcomes
            ))
            self.assertTrue(any(
                event["source"] == "long_horizon_planner.outcome"
                and "status=retry_pending" in event["content"]
                and "progress=0/" in event["content"]
                for event in outcomes
            ))

            planner._organism.workspace.events.clear()
            planner.record_goal_action_outcome(
                goal.id, "memory_recall", False, "still no evidence"
            )
            self.assertTrue(any(
                event["source"] == "long_horizon_planner.revision"
                and "[PlanRevised]" in event["content"]
                for event in planner._organism.workspace.events
            ))

    def test_reasoning_episode_persists_evidence_hypotheses_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(
                id="reasoning-1", topic="understand uncertain evidence",
                origin="resolve_uncertainty", priority=0.8,
            )
            reasoner = IntentionReasoner(
                planner._organism, path=str(Path(folder) / "reasoning.json")
            )
            episode = reasoner.deliberate(
                goal, ["memory_recall", "self_question", "web_search"]
            )

            self.assertEqual(len(episode.hypotheses), 3)
            self.assertEqual(episode.recommended_sequence[0], "memory_recall")
            self.assertTrue(any(item.source == "goal_engine" for item in episode.evidence))
            self.assertNotIn("chain-of-thought", episode.decision_rationale.lower())

            reasoner.record_outcome(
                episode.id, "memory_recall", False, "no relevant evidence"
            )
            reloaded = IntentionReasoner(
                planner._organism, path=str(Path(folder) / "reasoning.json")
            )
            restored = reloaded.get(episode.id)
            self.assertEqual(restored.status, "revision_needed")
            self.assertEqual(restored.outcomes[-1]["success"], False)
            self.assertTrue(any(
                item.source == "verified_outcome" for item in restored.evidence
            ))

    def test_repeated_failure_repairs_remaining_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(id="g-2", topic="research uncertainty", origin="explore")
            planner.ensure_goal_plan(goal)

            planner.record_goal_action_outcome(goal.id, "memory_recall", False, "empty")
            planner.record_goal_action_outcome(goal.id, "memory_recall", False, "empty")

            self.assertEqual(planner.next_action_for_goal(goal), "self_question")
            plan = planner.ensure_goal_plan(goal)
            self.assertIn("Decision revised", plan.current_decision)
            self.assertIn("proceed with self_question", plan.current_decision)
            self.assertTrue(any(
                "deferred after repeated failure" in item
                for item in plan.uncertainties
            ))
            planner._organism.ai_system = NS(
                goal_engine=NS(
                    get_active_goals=lambda min_activation=0.1: [goal],
                    get_goal_by_id=lambda goal_id: goal if goal_id == goal.id else None,
                )
            )
            snapshot = planner.factual_self_report_snapshot()["dominant_plan"]
            self.assertEqual(snapshot["decision_summary"], plan.current_decision)
            self.assertEqual(snapshot["decision_revision"], 2)
            self.assertEqual(snapshot["last_outcome"]["success"], False)

    def test_goal_intention_survives_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(id="g-3", topic="understand persistence", origin="resolve_uncertainty")
            planner.ensure_goal_plan(goal)
            planner.record_goal_action_outcome(
                goal.id, "memory_recall", True, "persistent evidence"
            )

            reloaded, _, _ = self._planner(folder)
            plan = reloaded.ensure_goal_plan(goal)
            self.assertEqual(plan.steps_completed, 1)
            self.assertEqual(reloaded.next_action_for_goal(goal), "self_question")

    def test_executor_consumes_and_completes_planned_step(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(id="g-4", topic="understand execution", origin="resolve_uncertainty")
            marked = []
            goal_engine = NS(mark_action=lambda goal_id, action: marked.append((goal_id, action)))
            organism = NS(
                ai_system=NS(goal_engine=goal_engine),
                _loop=NS(_long_horizon_planner=planner),
            )
            executor = GoalActionExecutor.__new__(GoalActionExecutor)
            executor._o = organism
            executor._last_action_time = {}
            executor._pending_predictions = {}

            self.assertEqual(executor._planned_action(goal), "memory_recall")
            executor._finalize_async_action(
                goal, "memory_recall", True, "recalled evidence"
            )

            self.assertEqual(marked, [(goal.id, "memory_recall")])
            self.assertEqual(planner.next_action_for_goal(goal), "self_question")

    def test_self_capability_question_gets_only_recorded_plan_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(
                id="g-factual", topic="checking capability",
                origin="resolve_uncertainty", priority=0.78,
            )
            planner._organism.ai_system = NS(
                goal_engine=NS(get_active_goals=lambda min_activation=0.1: [goal])
            )
            plan = planner.ensure_goal_plan(goal)
            planner.record_goal_action_outcome(
                goal.id, "memory_recall", True, "recalled evidence"
            )

            fragment = planner.prompt_fragment_for(
                "checking your capability to set goals and plan for it"
            )
            self.assertIn("Recorded active goals: checking capability", fragment)
            self.assertIn("progress 1/4", fragment)
            self.assertIn("completed=memory_recall", fragment)
            self.assertIn("next pending=self_question", fragment)
            self.assertIn("not currently executing", fragment)
            self.assertIn(f"confidence={plan.reasoning_confidence:.3f}", fragment)
            self.assertIn("aspirations are desired directions, not active goals", fragment)
            self.assertIn("Do not invent steps", fragment)

            budget = PromptContextBudget(str(Path(folder) / "prompt_contributions.json"))
            budget.add("factual_telemetry", "live_plan", fragment)
            self.assertIn("Cognitive telemetry - factual self-report", budget.assemble())

            correction = planner.prompt_fragment_for("i need factual elements not pure invention")
            follow_up = planner.prompt_fragment_for("any")
            self.assertIn("Cognitive telemetry - factual self-report", correction)
            self.assertIn("Cognitive telemetry - factual self-report", follow_up)

    def test_actual_goal_wording_triggers_factual_self_report(self):
        self.assertTrue(LongHorizonPlanner.is_self_report_query("what's your actual goal"))
        self.assertTrue(LongHorizonPlanner.is_self_report_query("quel est ton objectif actuel"))
        self.assertTrue(LongHorizonPlanner.is_self_report_query("votre véritable but"))

    def test_user_practical_task_cannot_become_autonomous_goal(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = GoalEngine(
                NS(), persistence_file=str(Path(folder) / "goals.json")
            )
            existing = Goal(
                id="external-existing", topic="voiture lavage",
                origin="resolve_uncertainty", priority=0.8, energy=0.8,
                created_cycle=1, last_active=1,
            )
            engine._goals[existing.id] = existing

            retired = engine.register_external_task(
                "Je dois laver ma voiture: à pied ou en voiture?",
                "laver voiture",
                ["à pied", "en voiture"],
            )

            self.assertEqual(retired, 1)
            self.assertEqual(existing.status, "abandoned")
            candidates = engine.generate_candidates(
                [Motivation("explore", 0.8, "test")],
                ["voiture", "what's actual", "architecture cognitive"],
            )
            self.assertEqual(
                [candidate.topic for candidate in candidates],
                ["architecture cognitive"],
            )
            reloaded = GoalEngine(
                NS(), persistence_file=str(Path(folder) / "goals.json")
            )
            self.assertTrue(reloaded._is_external_task_topic("lavage voiture"))

    def test_fragment_topics_are_not_durable_goals(self):
        self.assertFalse(GoalEngine._is_valid_goal_topic("travaille certains"))
        self.assertTrue(GoalEngine._is_valid_goal_topic("cognitive architecture"))

        with tempfile.TemporaryDirectory() as folder:
            engine = GoalEngine(
                NS(), persistence_file=str(Path(folder) / "goals.json")
            )
            fragment = Goal(
                id="fragment", topic="travaille certains",
                origin="resolve_uncertainty", priority=0.9, energy=0.9,
                created_cycle=1, last_active=1,
            )
            engine.add_goal(fragment)
            self.assertNotIn(fragment.id, engine._goals)

    def test_self_report_preserves_dominant_internal_plan_even_if_unrelated(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            relevant = NS(
                id="g-relevant", topic="detail process",
                origin="resolve_uncertainty", priority=0.7,
            )
            unrelated = NS(
                id="g-newer", topic="modern",
                origin="resolve_uncertainty", priority=0.8,
            )
            planner.ensure_goal_plan(relevant)
            planner.ensure_goal_plan(unrelated)
            planner._organism.ai_system = NS(
                goal_engine=NS(
                    get_active_goals=lambda min_activation=0.1: [unrelated, relevant]
                )
            )

            fragment = planner.prompt_fragment_for(
                "checking your capability to plan, detail the full process"
            )

            self.assertIn("may be independent of the current conversation", fragment)
            self.assertIn("objective='modern'", fragment)

    def test_completed_goal_closes_plan_and_disappears_from_live_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            planner, _, _ = self._planner(folder)
            goal = NS(
                id="g-finished", topic="reddish", origin="resolve_uncertainty",
                priority=0.95, energy=0.4, status="active",
            )
            plan = planner.ensure_goal_plan(goal)
            plan.uncertainties = [
                "memory_recall was deferred after repeated failure",
                "self_question was deferred after repeated failure",
            ]
            goal.status = "completed"
            planner._organism.ai_system = NS(
                goal_engine=NS(
                    get_active_goals=lambda min_activation=0.1: [],
                    get_goal_by_id=lambda goal_id: goal if goal_id == goal.id else None,
                )
            )

            snapshot = planner.factual_self_report_snapshot()

            self.assertEqual(plan.status, "completed")
            self.assertEqual(snapshot["active_goals"], [])
            self.assertIsNone(snapshot["dominant_plan"])
            self.assertIn("source goal is completed", plan.current_decision)

    def test_background_llm_reports_contention_without_provider_failure(self):
        calls = []

        class _Provider:
            model = "test-model"

            def _generate_messages(self, messages, **kwargs):
                calls.append(messages)
                return "A grounded internal observation."

        manager = LLMManager.__new__(LLMManager)
        manager.provider = _Provider()
        manager.text_model = "test-model"
        manager._provider_lock = threading.Lock()
        manager._chat_active = threading.Event()

        manager._chat_active.set()
        result = manager.generate_bare_result("reflect")
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["reason"], "interactive turn active")
        self.assertEqual(calls, [])

        manager._chat_active.clear()
        manager._provider_lock.acquire()
        try:
            result = manager.generate_bare_result("reflect")
        finally:
            manager._provider_lock.release()
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["reason"], "provider busy")
        self.assertEqual(calls, [])

        result = manager.generate_bare_result("reflect")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["text"], "A grounded internal observation.")
        self.assertEqual(len(calls), 1)

    def test_embedding_text_model_falls_back_to_chat_model(self):
        class _Provider:
            model = "gemma-chat"

        manager = LLMManager(_Provider(), text_model="text-embedding-nomic-embed-text-v1.5")
        self.assertEqual(manager.text_model, "gemma-chat")

    def test_factual_telemetry_has_prompt_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = PromptContextBudget(str(Path(folder) / "contributions.json"))
            budget.add("identity", "large_narrative", "narrative " * 500)
            budget.add("factual_telemetry", "plan", "RECORDED PLAN FACTS")
            assembled = budget.assemble()

            self.assertIn("RECORDED PLAN FACTS", assembled)

    def test_template_planner_preserves_dependency_order(self):
        planner = MultiStepPlanner()
        plan = planner.generate_plan(
            [{"id": "help", "name": "help Fred", "type": "help_user", "priority": 1.0}],
            {"total": 0.5},
        )

        self.assertEqual(
            [step["action"] for step in plan],
            ["understand_need", "gather_resources", "formulate_response", "deliver_help"],
        )
        self.assertEqual([step["depends_on"] for step in plan], [None, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
