import sys
import types
import unittest
from unittest.mock import patch

from cognition.voicemem_adapter import VoiceMemAdapter


class VoiceMemAdapterTests(unittest.TestCase):
    def test_disabled_adapter_is_a_noop(self):
        adapter = VoiceMemAdapter(enabled=False)
        self.assertEqual(adapter.search("hello"), [])
        self.assertFalse(adapter.ingest_text("hello"))
        self.assertFalse(adapter.status()["enabled"])

    def test_missing_package_is_non_fatal(self):
        adapter = VoiceMemAdapter(enabled=True)
        with patch("cognition.voicemem_adapter.importlib.util.find_spec", return_value=None):
            self.assertEqual(adapter.search("hello"), [])

    def test_fake_package_ingests_and_normalizes_results(self):
        class FakeVoiceMem:
            def __init__(self, **kwargs):
                self.items = []

            def ingest(self, text):
                self.items.append(text)

            def search(self, query, top_k=5):
                return [{"content": self.items[0]}] if self.items else []

        fake_module = types.SimpleNamespace(VoiceMem=FakeVoiceMem)
        adapter = VoiceMemAdapter(enabled=True, top_k=3)
        with patch("cognition.voicemem_adapter.importlib.util.find_spec", return_value=object()), \
             patch.dict(sys.modules, {"voicemem": fake_module}):
            self.assertTrue(adapter.ingest_text("I prefer concise answers", "fred"))
            result = adapter.search("preferences", "fred")
        self.assertEqual(result[0]["text"], "I prefer concise answers")
        self.assertEqual(result[0]["source"], "voicemem")


if __name__ == "__main__":
    unittest.main()
