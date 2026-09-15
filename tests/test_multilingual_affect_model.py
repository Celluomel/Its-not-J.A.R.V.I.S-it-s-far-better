"""Optional local-model eval; skipped unless multilingual weights are cached."""

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)

from cognition.multilingual_affect import MultilingualAffectClassifier
from cognition.empathy_engine import EmpathyEngine


class MultilingualAffectModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.classifier = MultilingualAffectClassifier()
        except Exception as exc:
            raise unittest.SkipTest(f"multilingual affect model is not cached: {exc}")

    def test_positive_and_negative_affect_across_twelve_languages(self):
        cases = [
            ("I feel calm and content today.", "positive"),
            ("I feel exhausted and worried.", "negative"),
            ("Je me sens calme et contente aujourd’hui.", "positive"),
            ("Je me sens épuisé et inquiet.", "negative"),
            ("Me siento tranquila y feliz.", "positive"),
            ("Me siento agotado y preocupado.", "negative"),
            ("Ich fühle mich ruhig und zufrieden.", "positive"),
            ("Ich bin erschöpft und mache mir Sorgen.", "negative"),
            ("Mi sento sereno e felice.", "positive"),
            ("Mi sento esausto e preoccupato.", "negative"),
            ("Sinto-me calmo e contente.", "positive"),
            ("Estou exausto e preocupado.", "negative"),
            ("Я чувствую себя спокойно и хорошо.", "positive"),
            ("Я чувствую себя измученным и встревоженным.", "negative"),
            ("我感到平静和满足。", "positive"),
            ("我感到疲惫和担忧。", "negative"),
            ("落ち着いていて、満足しています。", "positive"),
            ("疲れ果てていて、不安です。", "negative"),
            ("أشعر بالهدوء والرضا.", "positive"),
            ("أشعر بالإرهاق والقلق.", "negative"),
            ("차분하고 만족스럽습니다.", "positive"),
            ("지치고 걱정됩니다.", "negative"),
            ("Tôi cảm thấy bình tĩnh và hài lòng.", "positive"),
            ("Tôi cảm thấy kiệt sức và lo lắng.", "negative"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.classifier.classify(text).valence, expected)

    def test_factual_or_ambiguous_text_abstains_instead_of_guessing_affect(self):
        cases = [
            "The meeting starts at three in the afternoon.",
            "La réunion commence à trois heures cet après-midi.",
            "La reunión empieza a las tres de la tarde.",
            "明日の会議は午後三時に始まります。",
            "I feel neither happy nor sad about this.",
        ]
        for text in cases:
            with self.subTest(text=text):
                result = self.classifier.classify(text)
                self.assertEqual(result.valence, "neutral")
                self.assertEqual(result.confidence, 0.0)

    def test_empathy_engine_applies_local_multilingual_result_asynchronously(self):
        engine = EmpathyEngine(
            organism=type("Organism", (), {})(),
            llm=None,
            multilingual_classifier_factory=lambda: self.classifier,
        )

        read = engine.read_user("nino", "Me siento agotado y preocupado.")
        engine._affect_queue.join()

        self.assertEqual(read.valence, "negative")
        self.assertEqual(read.source, "multilingual_embedding")
        self.assertGreaterEqual(read.confidence, 0.5)

    def test_configured_model_id_is_used_by_local_encoder(self):
        captured = []

        class Encoder:
            def __init__(self, model_id, **_kwargs):
                captured.append(model_id)

            def encode(self, texts, **_kwargs):
                import numpy as np
                return np.ones((len(texts), 4), dtype=np.float32)

        with patch.dict(sys.modules, {
            "sentence_transformers": ModuleType("sentence_transformers"),
        }) as modules:
            modules["sentence_transformers"].SentenceTransformer = Encoder
            MultilingualAffectClassifier("test/custom-affect-model")

        self.assertEqual(captured, ["test/custom-affect-model"])


if __name__ == "__main__":
    unittest.main()
