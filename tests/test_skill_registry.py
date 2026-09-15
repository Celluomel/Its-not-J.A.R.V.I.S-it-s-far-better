import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)
core_package = ModuleType("core")
core_package.__path__ = [str(ROOT / "core")]
sys.modules.setdefault("core", core_package)

from cognition.skill_registry import SkillRegistry


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def get_response(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "sub_skills": ["causal debugging", "architecture tracing"],
            "insights": ["Trace state transitions before changing behavior."],
            "keywords": ["debugging", "debogage"],
        })


class _FakeAISystem:
    def __init__(self, llm=None):
        self.llm = llm
        self.memory_system = type("Memory", (), {"embedding_model": None})()


class _FakeOrganism:
    def __init__(self, llm=None):
        self.ai_system = _FakeAISystem(llm)
        self._last_interaction_ts = 0.0


class SkillRegistryTests(unittest.TestCase):
    def test_legacy_registry_is_migrated_without_losing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            path.write_text(json.dumps({
                "skills": {
                    "technical_python_debugging": {
                        "skill_id": "technical_python_debugging",
                        "domain": "procedural",
                        "name": "Technical Python Debugging",
                        "description": "Tracing Python architecture and state transitions.",
                        "proficiency": 0.80,
                        "instances": 40,
                        "confidence": 0.85,
                        "last_demonstrated": 1234.0,
                    }
                }
            }), encoding="utf-8")

            registry = SkillRegistry(_FakeOrganism(), str(path))
            skill = registry.skills_snapshot()["technical_python_debugging"]

            self.assertEqual(skill.name, "Technical Python Debugging")
            self.assertEqual(skill.category, "procedural")
            self.assertEqual(skill.practice_count, 40)
            self.assertEqual(skill.depth, 3)
            self.assertEqual(skill.proficiency, 0.80)
            self.assertEqual(skill.growth_trajectory, "stable")
            self.assertEqual(skill.last_practiced, 1234.0)
            self.assertEqual(registry.summary()["total_practices"], 40)
            self.assertEqual(registry._state.total_deepenings, 3)

            persisted = json.loads(path.read_text(encoding="utf-8"))
            migrated = persisted["skills"]["technical_python_debugging"]
            self.assertEqual(migrated["practice_count"], 40)
            self.assertEqual(migrated["category"], "procedural")

    def test_seed_is_persisted_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            registry = SkillRegistry(_FakeOrganism(), str(path))

            registry.seed("Causal Reasoning", "Reasoning from causes to effects")

            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("causal_reasoning", stored["skills"])

    def test_reseeding_exposure_does_not_inflate_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_FakeOrganism(), str(Path(directory) / "skills.json"))
            registry.seed("reasoning", "Structured reasoning", initial_confidence=0.6)
            for _ in range(20):
                registry.seed("reasoning", "Structured reasoning", source_memory_id="same-source")

            self.assertEqual(registry.skills_snapshot()["reasoning"].confidence, 0.6)
            self.assertEqual(registry.summary()["verified_outcomes"], 0)

    def test_verified_skill_confidence_uses_cumulative_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_FakeOrganism(), str(Path(directory) / "skills.json"))
            registry.seed("reasoning", "Structured reasoning", initial_confidence=0.6)

            registry.practice("reasoning", success=True, schedule_enrichment=False)
            one_positive = registry.skills_snapshot()["reasoning"].confidence
            registry.practice("reasoning", success=False, schedule_enrichment=False)
            mixed = registry.skills_snapshot()["reasoning"].confidence

            self.assertAlmostEqual(one_positive, 0.68)
            self.assertAlmostEqual(mixed, 3.4 / 6.0)
            summary = registry.summary()
            self.assertEqual(summary["verified_outcomes"], 2)
            self.assertEqual(summary["positive_outcomes"], 1)
            self.assertEqual(summary["negative_outcomes"], 1)

    def test_legacy_confidence_is_preserved_on_first_new_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            path.write_text(json.dumps({"skills": {
                "reasoning": {
                    "confidence": 0.7, "proficiency": 0.7,
                    "positive_outcomes": 3, "negative_outcomes": 1,
                    "verified_practices": 3,
                }
            }}), encoding="utf-8")
            registry = SkillRegistry(_FakeOrganism(), str(path))
            before = registry.skills_snapshot()["reasoning"].confidence
            registry.practice("reasoning", success=True, schedule_enrichment=False)

            self.assertAlmostEqual(before, 0.7)
            self.assertAlmostEqual(registry.skills_snapshot()["reasoning"].confidence, 6.6 / 9.0)

    def test_reflection_candidate_requires_repeated_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            registry = SkillRegistry(_FakeOrganism(), str(path))

            self.assertIsNone(registry.propose("systems thinking", "first observation"))
            self.assertIsNone(registry.propose("systems thinking", "second observation"))
            created = registry.propose("systems thinking", "third observation")

            self.assertIsNotNone(created)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("systems_thinking", stored["skills"])
            self.assertNotIn("systems_thinking", stored["candidates"])

    def test_prompt_lookup_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            registry = SkillRegistry(_FakeOrganism(), str(path))
            registry.seed("technical_debugging", "Python code debugging", initial_confidence=0.8)
            for _ in range(5):
                registry.practice("technical_debugging", success=True, schedule_enrichment=False)
            before = path.read_text(encoding="utf-8")
            practices = registry.summary()["total_practices"]

            fragment = registry.prompt_fragment("help debug this Python code")

            self.assertIn("Technical Debugging", fragment)
            self.assertEqual(registry.summary()["total_practices"], practices)
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_completed_interaction_is_learned_on_background_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            registry = SkillRegistry(_FakeOrganism(), str(path))
            registry.seed("architecture_explanation", "Explain software architecture",  initial_confidence=0.7)

            registry.observe_interaction(
                "Can you explain this software architecture?",
                "Yes. The system has an event layer and a persistence layer.",
            )
            registry._observation_queue.join()

            skill = registry.skills_snapshot()["architecture_explanation"]
            self.assertEqual(skill.practice_count, 1)
            self.assertIn("explain", skill.aliases)
            self.assertEqual(skill.confidence, 0.7)

    def test_unrated_practice_does_not_raise_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_FakeOrganism(), str(Path(directory) / "skills.json"))
            registry.seed("reasoning", "Structured reasoning", initial_confidence=0.7)

            registry.practice("reasoning", success=None, schedule_enrichment=False)

            skill = registry.skills_snapshot()["reasoning"]
            self.assertEqual(skill.practice_count, 1)
            self.assertEqual(skill.confidence, 0.7)

    def test_explicit_feedback_updates_skill_outcome_without_counting_second_practice(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_FakeOrganism(), str(Path(directory) / "skills.json"))
            registry.seed("architecture_explanation", "Explain software architecture", initial_confidence=0.7)
            registry.relevant_skills = lambda query, n=2, semantic=False: [
                registry.skills_snapshot()["architecture_explanation"]
            ]
            registry._save = lambda: None
            registry._schedule_semantic_refresh = lambda: None

            registry.observe_interaction("Explain this architecture", "A layered service design.")
            registry._observation_queue.join()
            before = registry.skills_snapshot()["architecture_explanation"]
            self.assertEqual(before.practice_count, 1)
            self.assertEqual(before.confidence, 0.7)

            self.assertTrue(registry.observe_feedback("A layered service design.", True))
            registry._observation_queue.join()
            after = registry.skills_snapshot()["architecture_explanation"]

            self.assertEqual(after.practice_count, 1)
            self.assertGreater(after.confidence, before.confidence)
            self.assertEqual(after.positive_outcomes, 1)
            self.assertEqual(after.verified_practices, 1)

    def test_exposure_alone_does_not_inflate_skill_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_FakeOrganism(), str(Path(directory) / "skills.json"))
            registry.seed("reasoning", "Structured reasoning")

            for _ in range(10):
                registry.practice("reasoning", success=None, schedule_enrichment=False)

            skill = registry.skills_snapshot()["reasoning"]
            self.assertEqual(skill.practice_count, 10)
            self.assertEqual(skill.verified_practices, 0)
            self.assertEqual(skill.depth, 0)

    def test_clear_followup_correction_is_delayed_negative_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_FakeOrganism(), str(Path(directory) / "skills.json"))
            registry.seed("architecture_explanation", "Explain software architecture", initial_confidence=0.7)
            registry.relevant_skills = lambda query, n=2, semantic=False: [
                registry.skills_snapshot()["architecture_explanation"]
            ]
            registry._save = lambda: None
            registry._schedule_semantic_refresh = lambda: None

            registry.observe_interaction("Explain this architecture", "A layered service design.")
            registry._observation_queue.join()
            self.assertTrue(registry.observe_user_followup("That is not what I asked for."))
            registry._observation_queue.join()

            skill = registry.skills_snapshot()["architecture_explanation"]
            self.assertEqual(skill.negative_outcomes, 1)
            self.assertLess(skill.confidence, 0.7)

    def test_depth_enrichment_uses_background_llm_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            llm = _FakeLLM()
            registry = SkillRegistry(_FakeOrganism(llm), str(path))
            skill = registry.seed("technical_debugging", "Python debugging", initial_confidence=0.7)
            for _ in range(5):
                registry.practice("technical_debugging", success=True, schedule_enrichment=False)
            skill = registry.skills_snapshot()["technical_debugging"]

            registry._SkillRegistry__depth_llm_call(skill, "fixed a state transition")

            enriched = registry.skills_snapshot()["technical_debugging"]
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(enriched.enriched_depth, 1)
            self.assertIn("causal debugging", enriched.sub_skills)
            self.assertIn("debogage", enriched.aliases)

    def test_compatibility_view_feeds_other_cognitive_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(
                _FakeOrganism(), str(Path(directory) / "skills.json")
            )
            registry.seed("reasoning", "Structured reasoning", initial_confidence=0.7)
            for _ in range(5):
                registry.practice("reasoning", success=True, schedule_enrichment=False)

            self.assertEqual(registry._skills["reasoning"].depth, 1)


if __name__ == "__main__":
    unittest.main()
