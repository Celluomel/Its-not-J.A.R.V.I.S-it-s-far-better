import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)

from cognition.memory_retrieval import rank_documents, tokenize


class MemoryRetrievalQualityTests(unittest.TestCase):
    DOCUMENTS = [
        "The Arduino Q development board was considered for PandoraBOX's future physical body and embodiment, with a movable camera and LiDAR.",
        "The car wash is 150 metres from Fred's home; walking would leave the vehicle behind, so he needs to drive it there.",
        "LM Studio loaded Gemma as a vision model that accepts image inputs for camera analysis.",
        "The face recognition camera has known people including Nino, Theodore, Fred, and Marion.",
        "Coqui is the selected text to speech provider, and its voice file must be selected in voice settings.",
        "Flux and PandoraBOX exchange messages through the network dialogue session.",
        "A cake needs to be placed in a preheated oven to bake properly.",
        "The cognitive dashboard shows planning, memory, and affect telemetry for the organism.",
        "Fred doit conduire sa voiture jusqu'au lavage auto ; à pied, il laisserait le véhicule chez lui.",
        "LM Studio a chargé Gemma, un modèle de vision capable de traiter les images.",
        "Le fournisseur Coqui doit utiliser le fichier de voix sélectionné dans les paramètres audio.",
    ]

    CASES = [
        ("Which development board was considered for PandoraBOX's physical embodiment?", 0),
        ("Why must Fred drive rather than walk to clean the vehicle?", 1),
        ("What computer vision model was loaded in LM Studio?", 2),
        ("Which familiar person does the face tracker know named Nino?", 3),
        ("Which speech provider needs its voice file selected?", 4),
        ("How do Flux and PandoraBOX exchange messages?", 5),
        ("Pourquoi Fred doit-il conduire sa voiture au lavage auto ?", 8),
        ("Quel modèle de vision est chargé dans LM Studio ?", 9),
        ("Quel fournisseur vocal utilise le fichier de voix sélectionné ?", 10),
    ]

    def test_bilingual_retrieval_benchmark_recall_and_rank(self):
        reciprocal_ranks = []
        recall_at_three = 0
        for query, expected_index in self.CASES:
            ranked = rank_documents(query, self.DOCUMENTS, limit=3)
            indices = [index for index, _ in ranked]
            if expected_index in indices:
                recall_at_three += 1
                reciprocal_ranks.append(1.0 / (indices.index(expected_index) + 1))
            else:
                reciprocal_ranks.append(0.0)

        self.assertEqual(recall_at_three, len(self.CASES))
        self.assertGreaterEqual(sum(reciprocal_ranks) / len(reciprocal_ranks), 0.8)

    def test_french_accents_are_normalized_and_empty_queries_return_nothing(self):
        self.assertEqual(tokenize("Épuisé, déjà, à côté"), ["epuise", "deja", "cote"])
        self.assertEqual(rank_documents("le et ou", self.DOCUMENTS), [])


if __name__ == "__main__":
    unittest.main()
