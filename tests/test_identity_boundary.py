"""Regression tests for configured persona-name ownership."""
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
cognition_package = ModuleType("cognition")
cognition_package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", cognition_package)

from cognition.identity_boundary import claims_persona_identity, prompt_block


class IdentityBoundaryTests(unittest.TestCase):
    def test_detects_persona_claim_in_supported_phrasings(self):
        self.assertTrue(claims_persona_identity("je suis la Lumina", "Lumina"))
        self.assertTrue(claims_persona_identity("I am Lumina", "Lumina"))
        self.assertTrue(claims_persona_identity("Je m'appelle Nova", "Nova"))

    def test_plain_name_reference_is_not_an_identity_claim(self):
        self.assertFalse(claims_persona_identity("Lumina, peux-tu m'aider ?", "Lumina"))
        self.assertFalse(claims_persona_identity("je travaille sur le code de Lumina", "Lumina"))

    def test_prompt_uses_runtime_persona_and_interlocutor_names(self):
        block = prompt_block("Nova", "je suis Nova", "Fred")
        self.assertIn("configured self-name is Nova", block)
        self.assertIn("message sender is Fred", block)
        self.assertIn("ambiguous or mistaken", block)
        self.assertIn("Do not infer that they are another instance", block)


if __name__ == "__main__":
    unittest.main()
