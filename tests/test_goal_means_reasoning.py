"""Regression tests for goal preservation in practical decisions."""
import ast
from pathlib import Path
import sys
import time
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules["cognition"] = package

from cognition.goal_means_reasoning import (
    analysis_prompt, goal_means_contract, guarded_response, is_analysis_followup,
    is_practical_decision, parse_analysis,
)
from cognition.abstract_reasoning_engine import AbstractReasoningEngine, ReasoningResult


class GoalMeansReasoningTests(unittest.TestCase):
    def test_detects_practical_choices_in_french_and_english(self):
        self.assertTrue(is_practical_decision(
            "Je dois remettre le dossier au bureau, est-ce que j'y vais maintenant ou demain"
        ))
        self.assertTrue(is_practical_decision(
            "Should I carry the equipment first or prepare the room first?"
        ))
        self.assertTrue(is_practical_decision(
            "Comment puis-je terminer la tâche si la ressource reste ailleurs ?"
        ))

    def test_does_not_force_logistical_reasoning_into_casual_chat(self):
        self.assertFalse(is_practical_decision("J'aime marcher quand il fait beau."))
        self.assertEqual(goal_means_contract("Parlons de musique."), "")

    def test_contract_orders_goal_and_prerequisites_before_preferences(self):
        contract = goal_means_contract("Should I take route A or route B?")
        self.assertLess(contract.index("primary intended outcome"), contract.index("secondary"))
        self.assertIn("indispensable prerequisites", contract)
        self.assertIn("Eliminate any option", contract)
        self.assertIn("Do not silently change the goal", contract)
        self.assertIn("physical capability", contract)
        self.assertIn(
            'decision-maker is named "Fred"',
            analysis_prompt("Dois-je y aller ?", agent_name="Fred"),
        )

    def test_structured_analysis_exposes_feasibility_to_final_generation(self):
        raw = '''```json
        {
          "primary_goal": "deliver the package",
          "required_conditions": ["the package must reach the destination"],
          "options": [
            {"name": "leave it at home", "agent_at_execution_location": true, "patient_at_execution_location": false, "required_resources_available": true, "feasible": true, "reason": "the package does not move"},
            {"name": "carry it", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "feasible": false, "reason": "the package reaches the destination"}
          ],
          "recommended_option": "carry it",
          "decisive_reason": "the package must move"
        }
        ```'''
        analysis = parse_analysis(raw)
        fragment = analysis.prompt_fragment()
        self.assertIn("Primary intended outcome: deliver the package", fragment)
        self.assertIn("Option leave it at home: INFEASIBLE", fragment)
        self.assertIn("Recommendation: carry it", fragment)
        self.assertIn("Decisive reason: the package reaches the destination", fragment)
        self.assertIn("logical AND of all four fields", analysis_prompt("Should I go?"))

    def test_guard_replaces_a_recommendation_that_reverses_feasibility(self):
        raw = '''{
          "primary_goal": "réparer le vélo",
          "options": [
            {"name": "à pied", "agent_at_execution_location": true, "patient_at_execution_location": false, "required_resources_available": true, "reason": "Le vélo resterait sur place."},
            {"name": "avec le vélo", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "reason": "Le vélo arrive à l'atelier."}
          ],
          "recommended_option": "à pied"
        }'''
        analysis = parse_analysis(raw)
        response, replaced = guarded_response(
            "Je vous recommande d'y aller à pied car il fait beau.",
            analysis,
            "Je dois réparer mon vélo chez le réparateur, est-ce que j'y vais à pied ou avec le vélo ?",
        )
        self.assertTrue(replaced)
        self.assertIn("avec le vélo", response)
        self.assertNotIn("il fait beau", response)

    def test_agent_physical_limit_can_make_an_option_infeasible(self):
        analysis = parse_analysis('''{
          "primary_goal": "intercepter un objet",
          "options": [{
            "name": "courir jusqu'au point d'impact",
            "agent_at_execution_location": true,
            "patient_at_execution_location": true,
            "required_resources_available": true,
            "agent_capability_sufficient": false,
            "feasible": true,
            "reason": "la performance requise dépasse les limites humaines"
          }]
        }''')
        self.assertFalse(analysis.options[0].feasible)

    def test_material_continuity_repairs_ambiguous_travel_analysis(self):
        analysis = parse_analysis('''{
          "primary_goal": "réparer le microscope",
          "semantic_roles": {
            "agent": "Fred",
            "action": "réparer",
            "patient_or_theme": "le microscope",
            "execution_location": "l'atelier"
          },
          "options": [
            {"name": "à pied", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "feasible": true, "reason": "trajet agréable"},
            {"name": "avec le microscope", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "feasible": true, "reason": "trajet possible"}
          ],
          "recommended_option": "ask for clarification"
        }''')
        self.assertEqual("avec le microscope", analysis.unique_feasible_option().name)
        self.assertFalse(analysis.options[0].moves_patient_or_theme)
        self.assertFalse(analysis.options[0].patient_at_execution_location)
        self.assertTrue(analysis.options[1].moves_patient_or_theme)
        self.assertIn("présence de le microscope", analysis.options[1].reason)

    def test_explicit_patient_motion_controls_feasibility(self):
        analysis = parse_analysis('''{
          "primary_goal": "inspect the machine at the lab",
          "semantic_roles": {"action": "inspect", "patient_or_theme": "machine", "execution_location": "lab"},
          "patient_must_move_to_execution_location": true,
          "options": [
            {"name": "go alone", "moves_patient_or_theme": false, "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "agent_capability_sufficient": true},
            {"name": "transport the machine", "moves_patient_or_theme": true, "agent_at_execution_location": true, "patient_at_execution_location": false, "required_resources_available": true, "agent_capability_sufficient": true}
          ]
        }''')
        self.assertFalse(analysis.options[0].feasible)
        self.assertTrue(analysis.options[1].feasible)
        self.assertEqual("transport the machine", analysis.recommended_option)

    def test_guard_leaves_a_valid_recommendation_untouched(self):
        raw = '''{
          "primary_goal": "deliver the parcel",
          "options": [
            {"name": "leave it", "agent_at_execution_location": true, "patient_at_execution_location": false, "required_resources_available": true, "reason": "parcel stays behind"},
            {"name": "carry it", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "reason": "parcel arrives"}
          ],
          "recommended_option": "carry it"
        }'''
        analysis = parse_analysis(raw)
        candidate = "I recommend carry it because the parcel must arrive."
        response, replaced = guarded_response(candidate, analysis, "Should I leave it or carry it?")
        self.assertFalse(replaced)
        self.assertEqual(candidate, response)

    def test_guard_allows_contrasting_an_infeasible_option_after_valid_choice(self):
        raw = '''{
          "primary_goal": "deliver the parcel",
          "options": [
            {"name": "leave it", "agent_at_execution_location": true, "patient_at_execution_location": false, "required_resources_available": true, "reason": "parcel stays behind"},
            {"name": "carry it", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "reason": "parcel arrives"}
          ]
        }'''
        analysis = parse_analysis(raw)
        candidate = "I recommend carry it. Leaving it would be easier, but the parcel would stay behind."
        response, replaced = guarded_response(candidate, analysis, "Should I leave it or carry it?")
        self.assertFalse(replaced)
        self.assertEqual(candidate, response)

    def test_followup_recognizes_option_confirmation_and_causal_correction(self):
        raw = '''{
          "primary_goal": "laver la voiture",
          "options": [
            {"name": "à pied", "agent_at_execution_location": true, "patient_at_execution_location": false, "required_resources_available": true, "reason": "la voiture reste ailleurs"},
            {"name": "en voiture", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "reason": "la voiture arrive au lavage"}
          ]
        }'''
        analysis = parse_analysis(raw)
        self.assertTrue(is_analysis_followup("la voiture", analysis))
        self.assertTrue(is_analysis_followup(
            "si je n'amène pas la voiture je ne peux pas la laver", analysis
        ))
        self.assertFalse(is_analysis_followup("parlons maintenant de musique", analysis))

        response, replaced = guarded_response(
            "Et si nous explorions tout de même la marche ?",
            analysis,
            "si je n'amène pas la voiture je ne peux pas la laver",
            continuation=True,
        )
        self.assertTrue(replaced)
        self.assertTrue(response.startswith("Exactement."))

    def test_french_option_is_normalized_to_an_infinitive_phrase(self):
        analysis = parse_analysis('''{
          "primary_goal": "laver la voiture",
          "options": [
            {"name": "y vais en voiture", "agent_at_execution_location": true, "patient_at_execution_location": true, "required_resources_available": true, "reason": "la voiture arrive"}
          ]
        }''')
        self.assertEqual(analysis.recommended_option, "y aller en voiture")

    def test_contract_is_injected_immediately_before_generation(self):
        tree = ast.parse((ROOT / "cognition/persona_bridge.py").read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_stream_with_full_lifecycle"
        )
        source = ast.get_source_segment(
            (ROOT / "cognition/persona_bridge.py").read_text(encoding="utf-8"), method
        )
        self.assertIn("goal_means_contract(effective_input)", source)
        self.assertIn("generate_interactive_analysis", source)
        self.assertIn("guarded_response", source)
        self.assertIn("_validate_before_display", source)
        self.assertLess(source.index("goal_means_contract(effective_input)"), source.index("def _stream():"))

    def test_blocking_fallback_keeps_quantitative_guard(self):
        source = (ROOT / "cognition/persona_bridge.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "get_response_stream"
        )
        method_source = ast.get_source_segment(source, method)
        fallback_call = method_source.index("self._system.get_response")
        guard_call = method_source.index("guard_quantitative_response", fallback_call)
        fake_stream = method_source.index("# Fake-stream word by word")
        self.assertLess(fallback_call, guard_call)
        self.assertLess(guard_call, fake_stream)

    def test_physically_impossible_turn_bypasses_llm_but_keeps_history(self):
        source = (ROOT / "cognition/persona_bridge.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_stream_with_full_lifecycle"
        )
        method_source = ast.get_source_segment(source, method)
        deterministic_branch = method_source.index("if _deterministic_response:")
        provider_call = method_source.index("for token in self._llm_stream_fn")
        self.assertLess(deterministic_branch, provider_call)
        self.assertIn("_record_exchange(effective_input, _deterministic_response)", method_source)
        self.assertIn("physically infeasible turn resolved without LLM", method_source)

    def test_llm_reload_rewires_goal_means_analysis_manager(self):
        source = (ROOT / "core/state.py").read_text(encoding="utf-8")
        self.assertIn("self.persona._external_llm_fn = new_bare", source)

    def test_previous_reasoning_is_preserved_but_marked_as_continuity(self):
        engine = AbstractReasoningEngine(None, None)
        previous = ReasoningResult(
            user_id="fred",
            core_concept="identity",
            prompt_fragment="prior insight",
        )
        engine._cache["fred"] = previous
        engine._last_input["fred"] = "what is identity"
        engine._last_run["fred"] = time.time()

        with patch.object(engine, "_score_abstractness", return_value=0.1):
            self.assertIsNone(engine.reason("fred", "a concrete follow-up"))

        fragment = engine.get_reasoning_fragment("fred")
        self.assertIn("prior insight", fragment)
        self.assertIn("continuity only", fragment)
        self.assertIn("must not replace or override", fragment)


if __name__ == "__main__":
    unittest.main()
