"""Health data regressions, without starting Lumina or calling an LLM."""
import ast
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
package = ModuleType('cognition')
package.__path__ = [str(ROOT / 'cognition')]
with patch.dict(sys.modules, {'cognition': package}):
    from cognition.cognitive_audit_engine import CognitiveAuditEngine
    from cognition.narrative_compression import NarrativeCompression
    from cognition.temporal_projection import TemporalProjection, PathProjection


class HealthDataTests(unittest.TestCase):
    def test_audit_reads_current_storage_contracts(self):
        audit = CognitiveAuditEngine.__new__(CognitiveAuditEngine)
        audit._organism = NS(skill_registry=NS(_state=NS(
            skills={'a': {'depth': 1}, 'b': {'depth': 3}})))
        audit._ai = NS(self_concept=NS(_beliefs={
            'a': NS(confidence=0.6), 'b': NS(confidence=0.8)}))
        snapshot = audit._read_signals(42)
        self.assertEqual(snapshot.mean_skill_depth, 2)
        self.assertAlmostEqual(snapshot.belief_confidence_mean, 0.7)

    def test_projection_restores_nested_paths_and_skips_bad_entry(self):
        path = dict(archetype='intensive', cumulative_cost=0.2,
                    progress_rate=0.1, efficiency=0.5, crosses_critical=False,
                    min_energy_predicted=0.8, action_sequence=['technical'])
        result = dict(aspiration_domain='cognitive', aspiration_desc='test',
                      projected_at_cycle=10, timestamp=100, paths=[path],
                      preferred_path='intensive', preferred_action='technical',
                      rationale='test')
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'projections.json'
            file.write_text(json.dumps({'active': {'bad': {}, 'cognitive': result},
                                        'log': [result]}))
            engine = TemporalProjection(None, None, str(file))
            self.assertEqual(engine.status()['active_projections'], 1)
            self.assertEqual(len(engine._log), 1)
            self.assertIsInstance(engine._active['cognitive'].paths[0], PathProjection)

    def test_narrative_uses_low_level_llm_not_chat_pipeline(self):
        engine = NarrativeCompression.__new__(NarrativeCompression)
        engine._ai = NS(llm=NS(get_response=Mock(return_value='I learn.')),
                        get_response=Mock(side_effect=AssertionError('chat called')))
        scheduler = ModuleType('core.llm_scheduler')
        scheduler.llm_scheduler = NS(sync_slot=Mock(return_value=nullcontext(True)))
        with patch.dict(sys.modules, {'core.llm_scheduler': scheduler}):
            self.assertEqual(engine._synthesise_theme([NS(title='t', description='d')]),
                             'I learn.')
        engine._ai.llm.get_response.assert_called_once()
        engine._ai.get_response.assert_not_called()

    def test_streaming_deliberation_precedes_prompt_build(self):
        tree = ast.parse((ROOT / 'cognition/persona_bridge.py').read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_build_prompt_and_cache')
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
        deliberate = next(n for n in calls if isinstance(n.func, ast.Name)
                          and n.func.id == 'deliberate')
        build = next(n for n in calls if isinstance(n.func, ast.Attribute)
                     and n.func.attr == '_build_prompt_additions')
        self.assertLess(deliberate.lineno, build.lineno)

    def test_live_cognitive_telemetry_precedes_narrative_prompt_sections(self):
        source = (ROOT / 'cognition/persona_bridge.py').read_text(encoding='utf-8')
        prompt_start = source.index('system_prompt = f"""')
        prompt_end = source.index('"""', prompt_start + len('system_prompt = f"""'))
        prompt_template = source[prompt_start:prompt_end]

        self.assertLess(
            prompt_template.index('{_organism_prompt_section}'),
            prompt_template.index('━━ CURRENT FUNCTIONAL STATE ━━'),
        )
        self.assertNotIn('━━ COGNITIVE STATE ━━', source[prompt_end:])

    def test_self_report_can_initialise_planner_without_background_cycle(self):
        source = (ROOT / 'cognition/cognitive_organism.py').read_text(encoding='utf-8')
        self.assertIn('LongHorizonPlanner.is_self_report_query(user_input)', source)
        self.assertIn('loop._long_horizon_planner = lhp', source)
        self.assertIn('factual planning telemetry unavailable', source)

    def test_factual_self_report_contract_is_adjacent_to_generation(self):
        source = (ROOT / 'cognition/persona_bridge.py').read_text(encoding='utf-8')
        self.assertIn('FINAL FACTUAL SELF-REPORT CONTRACT', source)
        self.assertIn(
            'Do not claim that composing or explaining this answer starts or completes',
            source,
        )
        self.assertIn('_generation_input, system_prompt,', source)
        self.assertIn('arb_temperature = min(', source)


if __name__ == '__main__':
    unittest.main()
