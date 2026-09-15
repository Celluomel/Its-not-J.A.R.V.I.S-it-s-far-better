import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)

from cognition.ai_system import EnhancedMemorySystem


class _Persistence:
    def __init__(self, rows):
        self.rows = rows

    def get_all_memories(self, tier=None):
        if tier is None:
            return list(self.rows)
        return [row for row in self.rows if row["memory_tier"] == tier]

    def get_memories_by_ids(self, _ids):
        return []


class _BrokenEmbedder:
    def encode(self, _text):
        raise RuntimeError("encoder unavailable")


class _Index:
    ntotal = 3

    def search(self, *_args):
        raise AssertionError("lexical fallback should run before vector search")


class _StaleIndex:
    ntotal = 1

    def search(self, *_args):
        return np.array([[0.1]], dtype="float32"), np.array([[999]], dtype="int64")


class _Embedder:
    def encode(self, _text):
        return np.ones(3, dtype="float32")


class MemorySystemFallbackTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now().isoformat()
        self.rows = [
            {"id": 1, "faiss_id": 1, "text": "Arduino Q camera movement and future embodiment",
             "timestamp": now, "impact_score": 0.5, "memory_type": "project",
             "memory_tier": "personal", "emotional_valence": "Neutral", "access_count": 0},
            {"id": 2, "faiss_id": 2, "text": "Coqui voice selection needs the configured speaker file",
             "timestamp": now, "impact_score": 0.5, "memory_type": "settings",
             "memory_tier": "interaction", "emotional_valence": "Neutral", "access_count": 0},
        ]

    def _system(self, embedder=None, index=None):
        system = EnhancedMemorySystem.__new__(EnhancedMemorySystem)
        system.persistence = _Persistence(self.rows)
        system.config = SimpleNamespace(max_memory_retrieval=5)
        system.embedding_dim = 3
        system.embedding_model = embedder
        system.faiss_index = index
        return system

    def test_no_embedder_still_recalls_persisted_memory_and_filters_tier(self):
        system = self._system()

        general = system.retrieve_memories("How can I move the Arduino camera?")
        personal = system.retrieve_by_tier("Arduino camera movement", "personal")

        self.assertEqual(general[0]["id"], 1)
        self.assertGreater(general[0]["relevance"], 0)
        self.assertEqual([row["id"] for row in personal], [1])

    def test_empty_vector_index_does_not_hide_database_memories(self):
        system = self._system(embedder=object(), index=SimpleNamespace(ntotal=0))

        result = system.retrieve_memories("configured Coqui speaker file")

        self.assertEqual(result[0]["id"], 2)

    def test_stale_vector_ids_do_not_hide_sqlite_memories(self):
        system = self._system(embedder=_Embedder(), index=_StaleIndex())

        result = system.retrieve_memories("configured Coqui speaker file")

        self.assertEqual(result[0]["id"], 2)

    def test_encoder_failure_disables_broken_vector_path_and_falls_back(self):
        system = self._system(embedder=_BrokenEmbedder(), index=_Index())

        result = system.retrieve_memories("configured Coqui speaker file")

        self.assertEqual(result[0]["id"], 2)
        self.assertIsNone(system.embedding_model)
        self.assertIsNone(system.faiss_index)


if __name__ == "__main__":
    unittest.main()
