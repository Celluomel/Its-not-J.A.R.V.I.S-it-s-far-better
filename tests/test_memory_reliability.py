import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)

from cognition.memory_intrusion import MemoryIntrusionSystem
from cognition.emotional_memory import EmotionalMemorySystem
from cognition.empathy_engine import EmpathyEngine
from cognition.semantic_extractor import SemanticExtractor


class _Memory:
    def __init__(self):
        self.concepts = []
        self.capabilities = []

    def retrieve_memories(self, query, limit=None, current_valence=None):
        self.request = (query, limit)
        return [{
            "text": "A useful remembered event about the project.",
            "timestamp": "2026-09-14T12:00:00",
            "emotional_valence": "Positive",
            "arousal_level": "High",
            "relevance": 0.9,
        }]

    def upsert_concept(self, concept, strength_boost=0.04):
        self.concepts.append(concept)

    def upsert_relation(self, source, target, rel_type, weight_boost=0.05):
        pass

    def update_capability(self, name, value):
        self.capabilities.append((name, value))


class _Workspace:
    def broadcast(self, **kwargs):
        pass


class _EmotionalState:
    def __init__(self):
        self.influences = []

    def receive_self_influence(self, values):
        self.influences.append(values)


class _EmotionAnalyzer:
    def analyze_emotional_context(self, text):
        if "broken" in text.lower() or "wrong" in text.lower():
            return {"valence": "Negative", "arousal": "High"}
        return {"valence": "Neutral", "arousal": "Low"}


class MemoryReliabilityTests(unittest.TestCase):
    def test_intrusion_reads_memory_dict_and_uses_supported_limit_argument(self):
        memory = _Memory()
        organism = type("Organism", (), {
            "ai_system": type("AI", (), {"memory_system": memory})(),
        })()
        system = MemoryIntrusionSystem(organism)

        candidates = system._search_memories("a sufficiently detailed current context")

        self.assertEqual(memory.request[1], 8)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].content, "A useful remembered event about the project.")
        self.assertGreater(candidates[0].score, 0.8)

    def test_semantic_extractor_queues_rapid_turns_instead_of_dropping_them(self):
        memory = _Memory()
        extractor = SemanticExtractor(memory)
        extractor._parse = lambda user, response: ([user], [], user, 0.1)

        extractor.extract_async("first topic", "first answer", workspace=_Workspace())
        extractor.extract_async("second topic", "second answer", workspace=_Workspace())
        extractor._queue.join()

        self.assertEqual(memory.concepts, ["first topic", "second topic"])

    def test_semantic_extraction_rejects_ungrounded_concepts_and_relations(self):
        extractor = SemanticExtractor(_Memory())
        source = "The camera feed enables face recognition in this interface."
        concepts = extractor._ground_concepts([
            {"name": "camera feed", "evidence": "camera feed"},
            {"name": "face recognition", "evidence": "face recognition"},
            {"name": "quantum gravity", "evidence": "quantum gravity"},
        ], source)
        relations = extractor._ground_relations([
            {"from": "camera feed", "to": "face recognition", "type": "enables",
             "evidence": "camera feed enables face recognition"},
            {"from": "camera feed", "to": "face recognition", "type": "causes",
             "evidence": "camera feed causes face recognition"},
            {"from": "quantum gravity", "to": "face recognition", "type": "related_to",
             "evidence": "quantum gravity relates to face recognition"},
        ], concepts, source)

        self.assertEqual(concepts, ["camera feed", "face recognition"])
        self.assertEqual(relations, [{
            "from": "camera feed", "to": "face recognition", "type": "enables",
        }])

    def test_keyword_fallback_extracts_french_concepts(self):
        concepts = SemanticExtractor(_Memory())._keyword_extract(
            "Je dois laver ma voiture au système de lavage automatique."
        )
        self.assertIn("voiture", concepts)
        self.assertIn("lavage", concepts)
        self.assertNotIn("voiture au", concepts)

    def test_llm_extraction_keeps_user_topic_ahead_of_verbose_response_concepts(self):
        extractor = SemanticExtractor(_Memory(), llm_fn=lambda _prompt: json.dumps({
            "concepts": [
                {"name": "embodiment", "evidence": "future embodiment"},
                {"name": "camera movement", "evidence": "camera movement"},
            ],
            "relations": [],
            "dominant_topic": "embodiment",
            "cognitive_load": 0.4,
        }))

        concepts, _, topic, _ = extractor._parse(
            "I want to understand camera movement.",
            "This could support future embodiment. " * 20,
        )

        self.assertIn("camera movement", concepts)
        self.assertIn("embodiment", concepts)
        self.assertLess(concepts.index("camera movement"), concepts.index("embodiment"))
        self.assertEqual(topic, "camera movement")

    def test_capability_updates_do_not_mistake_cognitive_load_for_memory_accuracy(self):
        memory = _Memory()
        extractor = SemanticExtractor(memory)
        extractor._update_capabilities({"clarity": 1.4, "confidence": float("nan")}, 1.0)

        self.assertEqual(memory.capabilities, [("conversation", 1.0)])

    def test_user_affect_uses_confident_empathy_read_and_requires_repeated_evidence(self):
        emotional_state = _EmotionalState()
        read = type("Read", (), {
            "valence": "negative", "arousal": "high", "confidence": 0.82,
            "dominant_emotion": "frustrated", "source": "llm",
            "timestamp": time.time(),
        })()
        organism = type("Organism", (), {
            "ai_system": type("AI", (), {"emotional_state": emotional_state})(),
            "_last_user_affect_read": read,
            "_last_user_affect_user_id": "fred",
        })()
        with tempfile.TemporaryDirectory() as directory:
            memory = EmotionalMemorySystem(organism, str(Path(directory) / "emotion.json"))
            memory.record_from_interaction("That answer is wrong and frustrating.", "A short reply", "fred")

            self.assertEqual(memory._state.traces[0]["perspective"], "user")
            self.assertEqual(memory._state.traces[0]["evidence_source"], "llm")
            self.assertAlmostEqual(memory._state.traces[0]["confidence"], 0.82)
            self.assertEqual(memory.compute_residue().total_weight, 0.0)

            memory.record_from_interaction("I am still frustrated.", "A reply", "fred")
            memory.on_user_arrival("fred")
            self.assertTrue(emotional_state.influences)

    def test_low_confidence_affect_does_not_prime_arrival(self):
        emotional_state = _EmotionalState()
        read = type("Read", (), {
            "valence": "negative", "arousal": "high", "confidence": 0.55,
            "dominant_emotion": "frustrated", "source": "heuristic",
            "timestamp": time.time(),
        })()
        organism = type("Organism", (), {
            "ai_system": type("AI", (), {"emotional_state": emotional_state})(),
            "_last_user_affect_read": read,
            "_last_user_affect_user_id": "fred",
        })()
        with tempfile.TemporaryDirectory() as directory:
            memory = EmotionalMemorySystem(organism, str(Path(directory) / "emotion.json"))
            memory.record_from_interaction("I feel bad.", "A reply", "fred")
            memory.record_from_interaction("Still bad.", "A reply", "fred")
            memory.on_user_arrival("fred")

        self.assertEqual(emotional_state.influences, [])

    def test_affect_calibration_uses_only_matched_explicit_self_reports_and_persists(self):
        read = type("Read", (), {
            "valence": "negative", "arousal": "medium", "confidence": 0.9,
            "dominant_emotion": "sad", "source": "heuristic",
            "source_text": "I feel happy and content.", "timestamp": time.time(),
        })()
        organism = type("Organism", (), {
            "_last_user_affect_read": read,
            "_last_user_affect_user_id": "fred",
        })()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emotion.json"
            memory = EmotionalMemorySystem(organism, str(path))
            for _ in range(3):
                memory.record_from_interaction("I feel happy and content.", "A reply", "fred")

            calibrated = memory.calibrate_affect_read("fred", read)
            stored = json.loads(path.read_text(encoding="utf-8"))

            self.assertAlmostEqual(calibrated, 2 / 7)
            self.assertEqual(stored["affect_calibration"]["fred:heuristic"], {
                "samples": 3, "correct": 0,
            })
            reloaded = EmotionalMemorySystem(organism, str(path))
            self.assertAlmostEqual(reloaded.calibrate_affect_read("fred", read), 2 / 7)

    def test_affect_calibration_ignores_unmatched_and_ambiguous_text(self):
        read = type("Read", (), {
            "valence": "negative", "arousal": "medium", "confidence": 0.9,
            "dominant_emotion": "sad", "source": "llm",
            "source_text": "The answer is wrong.", "timestamp": time.time(),
        })()
        organism = type("Organism", (), {
            "_last_user_affect_read": read,
            "_last_user_affect_user_id": "fred",
        })()
        with tempfile.TemporaryDirectory() as directory:
            memory = EmotionalMemorySystem(organism, str(Path(directory) / "emotion.json"))
            memory.record_from_interaction("The answer is wrong.", "A reply", "fred")
            self.assertEqual(memory._state.affect_calibration, {})
            self.assertEqual(memory.calibrate_affect_read("fred", read), 0.9)

    def test_explicit_bilingual_self_report_overrides_inferred_affect_without_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EmotionalMemorySystem(type("Organism", (), {})(), str(Path(directory) / "emotion.json"))
            organism = type("Organism", (), {"emotional_memory": memory})()
            engine = EmpathyEngine(organism, llm=None)
            llm_calls = []
            engine._llm_read = lambda *_args: llm_calls.append(True)

            read = engine.read_user("fred", "Je me sens épuisé et inquiet.")

            self.assertEqual(read.valence, "negative")
            self.assertEqual(read.source, "explicit_self_report")
            self.assertEqual(read.confidence, 0.98)
            self.assertEqual(llm_calls, [])

    def test_explicit_affect_language_cases_distinguish_self_reports_from_context(self):
        cases = {
            "I'm feeling relieved.": "positive",
            "Je vais bien.": "positive",
            "Je ne me sens pas bien.": "negative",
            "Je suis déçue.": "negative",
            "I'm not happy today.": "negative",
            "That answer is wrong.": None,
            "Je parle d'une personne épuisée.": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(EmotionalMemorySystem._explicit_self_report(text), expected)

    def test_affect_read_flows_into_persisted_emotional_memory_and_calibration(self):
        organism = type("Organism", (), {
            "ai_system": type("AI", (), {"emotional_state": type("State", (), {"emotions": {}})()})(),
        })()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emotion.json"
            memory = EmotionalMemorySystem(organism, str(path))
            organism.emotional_memory = memory
            engine = EmpathyEngine(organism, llm=None)

            read = engine.read_user("fred", "Je me sens épuisé et inquiet.")
            memory.record_from_interaction("Je me sens épuisé et inquiet.", "Je comprends.", "fred")

            self.assertEqual(read.source, "explicit_self_report")
            self.assertEqual(memory._state.affect_calibration["fred:explicit_self_report"], {
                "samples": 1, "correct": 1,
            })
            self.assertEqual(memory._state.traces[0]["perspective"], "user")
            self.assertEqual(memory._state.traces[0]["evidence_source"], "explicit_self_report")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["affect_calibration"]["fred:explicit_self_report"]["correct"], 1)

    def test_multilingual_affect_is_background_and_does_not_call_chat_llm(self):
        class _Classifier:
            def classify(self, _text):
                return type("Classification", (), {
                    "valence": "negative", "confidence": 0.71,
                })()

        engine = EmpathyEngine(
            type("Organism", (), {})(),
            llm=type("LLM", (), {"generate_bare": lambda *_args, **_kwargs: self.fail("chat LLM must not be called")})(),
            multilingual_classifier_factory=_Classifier,
        )

        read = engine.read_user("nino", "Estoy agotado y preocupado.")
        engine._affect_queue.join()

        self.assertEqual(read.valence, "negative")
        self.assertEqual(read.source, "multilingual_embedding")
        self.assertAlmostEqual(read.confidence, 0.71)

    def test_newer_turn_is_not_overwritten_by_stale_multilingual_result(self):
        class _NeutralClassifier:
            def classify(self, _text):
                return type("Classification", (), {"valence": "neutral", "confidence": 0.0})()

        engine = EmpathyEngine(
            type("Organism", (), {})(), llm=None,
            multilingual_classifier_factory=_NeutralClassifier,
        )
        older = engine.read_user("fred", "First neutral turn.")
        older_generation = engine._affect_generation["fred"]
        newer = engine.read_user("fred", "Second neutral turn.")
        newer_generation = engine._affect_generation["fred"]
        engine._affect_queue.join()
        newer_state = (newer.valence, newer.source)

        engine._apply_multilingual_classification(
            "fred", older.source_text, older.timestamp, older_generation,
            type("Classification", (), {"valence": "negative", "confidence": 0.8})(),
        )

        self.assertGreater(newer_generation, older_generation)
        self.assertEqual((newer.valence, newer.source), newer_state)


if __name__ == "__main__":
    unittest.main()
