"""Regression tests for camera identity and localized presence speech."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


presence = load_module("presence_engine_under_test", "cognition/presence_engine.py")
users = load_module("user_manager_under_test", "managers/user_manager.py")


class SequenceLlm:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_bare(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class PresenceIdentityLanguageTests(unittest.TestCase):
    def test_named_camera_face_creates_and_activates_interlocutor(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = users.UserManager(Path(directory) / "users.json")
            profile = manager.sync_active_from_faces([
                {"id": "face_17", "name": "Nino", "confidence": 0.72}
            ])

            self.assertEqual(profile.display_name, "Nino")
            self.assertEqual(manager.active_id, "guest")
            self.assertEqual(profile.face_ids, ["face_17"])

            same_profile = manager.sync_active_from_faces([
                {"id": "face_18", "name": "nInO", "confidence": 0.69}
            ])
            manager.sync_active_from_faces([
                {"id": "face_18", "name": "Nino", "confidence": 0.70}
            ])
            self.assertEqual(same_profile.id, profile.id)
            self.assertEqual(manager.active_id, "nino")
            self.assertCountEqual(same_profile.face_ids, ["face_17", "face_18"])
            self.assertEqual(len(manager.named_users()), 1)

    def test_single_false_face_match_does_not_switch_active_interlocutor(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = users.UserManager(Path(directory) / "users.json")
            fred = manager.create("Fred", face_ids=["face_fred"])
            theo = manager.create("Theodore", face_ids=["face_theo"])
            manager.set_active(fred.id)

            manager.sync_active_from_faces([
                {"id": "face_theo", "name": "Theodore", "confidence": 0.58}
            ])
            self.assertEqual(manager.active_id, "fred")

            for _ in range(2):
                manager.sync_active_from_faces([
                    {"id": "face_theo", "name": "Theodore", "confidence": 0.58}
                ])
            self.assertEqual(manager.active_id, theo.id)

    def test_persona_name_claim_is_not_learned_as_user_characteristic(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = users.UserManager(Path(directory) / "users.json")
            fred = manager.create("Fred")
            settings = SimpleNamespace(get_persona_name=lambda: "PandoraBOX")
            boundary = SimpleNamespace(
                claims_persona_identity=lambda text, name: "je suis la lumina" in text.casefold()
            )

            with patch.dict(sys.modules, {
                "managers.settings_manager": settings,
                "cognition.identity_boundary": boundary,
            }):
                changed = manager.learn_from_message(
                    fred.id, "Je suis la PandoraBOX, je travaille sur ton code."
                )

            self.assertFalse(changed)
            self.assertEqual(fred.characteristics, [])

    def test_french_presence_repairs_language_and_stale_name(self):
        engine = presence.PresenceEngine.__new__(presence.PresenceEngine)
        engine._org = None
        engine._llm = SequenceLlm("Hello Fred.", "Bonjour Nino.")
        engine._recent_utterances = []
        snapshot = presence._CognitiveSnapshot()
        face_state = presence.FacePresenceState("face_17", "Nino")
        settings = SimpleNamespace(config=SimpleNamespace(RESPONSE_LANGUAGE="FR"))

        with patch.dict(sys.modules, {"managers.settings_manager": settings}):
            result = engine._generate_reaction(
                "Nino", "ENTER", "engage", face_state, snapshot
            )

        self.assertEqual(result, "Bonjour Nino.")
        self.assertIn("person currently recognized by the camera is Nino", engine._llm.prompts[0])
        self.assertIn("exclusively in French", engine._llm.prompts[1])

    def test_french_fallback_is_localized(self):
        result = presence.PresenceEngine._fallback(
            "Nino", "ENTER", "neutral", "fr"
        )
        self.assertEqual(result, "Bonjour Nino.")


if __name__ == "__main__":
    unittest.main()
