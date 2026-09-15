"""Regression tests for quantitative constraint reasoning."""
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)

from cognition.quantitative_reasoning import (
    analyze_quantitative_question,
    guard_quantitative_response,
)


class QuantitativeReasoningTests(unittest.TestCase):
    def test_falling_object_question_builds_equation_and_missing_inputs(self):
        analysis = analyze_quantitative_question(
            "Je veux ratrapper un objet que je lache du2ième étage : à quelle vitesse dois-je courrir pour le ratepper ?",
            agent_name="Fred",
        )
        self.assertIsNotNone(analysis)
        self.assertIn("sqrt(2h/g)", analysis.relation)
        self.assertIn("route distance d", " ".join(analysis.unknowns))
        self.assertIn("1.1 s", analysis.fallback_answer)
        self.assertTrue(analysis.fallback_answer.startswith("Fred, aucune vitesse"))
        self.assertIn("tu te trouves encore au 2e étage à t=0", analysis.fallback_answer)
        self.assertIn("pas une distance horizontale inventée", analysis.fallback_answer)
        self.assertIn("dépasse les capacités physiques humaines", analysis.fallback_answer)
        self.assertIn("irréaliste et dangereux", analysis.fallback_answer)
        self.assertEqual("Fred", analysis.agent)
        self.assertFalse(analysis.feasible)

    def test_verbosity_selects_natural_verdict_or_full_derivation(self):
        analysis = analyze_quantitative_question(
            "Je lâche un objet du 2e étage, à quelle vitesse dois-je courir pour le rattraper ?",
            agent_name="Fred",
        )
        concise = analysis.answer_for_verbosity("concise")
        extended = analysis.answer_for_verbosity("verbose")
        self.assertLess(len(concise.split()), 80)
        self.assertNotIn("sqrt", concise)
        self.assertIn("Fred", concise)
        self.assertIn("sqrt(2h/g)", extended)
        self.assertGreater(len(extended), len(concise))

    def test_tautological_semantic_role_answer_is_replaced(self):
        analysis = analyze_quantitative_question(
            "Je lâche un objet du 2e étage, à quelle vitesse dois-je courir pour le rattraper ?"
        )
        answer, replaced = guard_quantitative_response(
            "Courir à la vitesse nécessaire permet au Agent d'atteindre le Patient/Thème.",
            analysis,
        )
        self.assertTrue(replaced)
        self.assertIn("v >= d/(t - t_r)", answer)
        self.assertNotIn("Agent", answer)

    def test_horizontal_distance_world_model_is_rejected(self):
        analysis = analyze_quantitative_question(
            "Je lâche un objet du deuxième étage, à quelle vitesse dois-je courir pour le rattraper ?"
        )
        candidate = (
            "Avec t = sqrt(2h/g), il faut parcourir une distance horizontale de 20 m. "
            "La distance et le temps de réaction donnent v >= d/(t-réaction)."
        )
        answer, replaced = guard_quantitative_response(candidate, analysis)
        self.assertTrue(replaced)
        self.assertIn("point de lâcher", answer)
        self.assertNotIn("20 m", answer)

    def test_grounded_candidate_is_preserved(self):
        analysis = analyze_quantitative_question(
            "I drop an object from the 2nd floor; how fast must I run to catch it?"
        )
        candidate = (
            "Use t = sqrt(2h/g) and v >= d/(t-reaction time). "
            "The same person starts upstairs and must descend the actual route; "
            "the distance information is missing, so no physically achievable human "
            "running speed can do this."
        )
        answer, replaced = guard_quantitative_response(candidate, analysis)
        self.assertFalse(replaced)
        self.assertEqual(candidate, answer)


if __name__ == "__main__":
    unittest.main()
