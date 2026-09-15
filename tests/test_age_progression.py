"""Regression tests for human-paced persona age progression."""

from datetime import datetime, timedelta
import unittest

from cognition.ai_system import EnhancedAISystem, SystemConfig


class AgeProgressionTests(unittest.TestCase):
    def test_default_rate_is_one_age_year_per_calendar_year(self):
        self.assertAlmostEqual(SystemConfig().age_units_per_day, 1.0 / 365.2425)

    def test_persona_age_and_stage_follow_calendar_age(self):
        now = datetime.now()
        persona = object.__new__(EnhancedAISystem)
        persona.config = SystemConfig()
        persona._birth_date = now - timedelta(days=18 * 365.2425)

        self.assertAlmostEqual(persona.current_age, 18.0, delta=0.0001)
        self.assertEqual(persona.life_stage, "young_adult")


if __name__ == "__main__":
    unittest.main()
