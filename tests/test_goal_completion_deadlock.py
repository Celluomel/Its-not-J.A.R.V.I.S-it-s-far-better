"""Goal completion must not deadlock the cognitive loop or chat preparation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading
import unittest
from unittest.mock import Mock, patch
from types import ModuleType

# Import only the goal modules, without booting the cognition package's LLMs.
package = ModuleType('cognition')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'cognition')]
with patch.dict(sys.modules, {'cognition': package}):
    from cognition.goal_engine import Goal, GoalEngine
    from cognition import goal_completion


class CompletionLockTests(unittest.TestCase):
    def test_update_completes_goal_and_releases_lock(self):
        with patch.object(GoalEngine, '_load_goals'):
            engine = GoalEngine(None)
        engine._save_goals = Mock()
        goal = Goal(id='test', topic='test completion', origin='explore',
                    priority=0.8, energy=0.9, created_cycle=0, last_active=0,
                    persistence=9, actions_taken=3, non_introspective_actions=1)
        engine._goals[goal.id] = goal
        errors = []

        def run():
            try:
                engine.update_goals()
                engine.update_goals()
            except BaseException as exc:
                errors.append(exc)

        with patch.dict(sys.modules, {'cognition': package, 'cognition.goal_completion': goal_completion}), \
             patch.object(goal_completion, 'load_current_tensions', return_value={}), \
             patch.object(goal_completion, 'get_narrative_identity', return_value=None):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive(), 'Auto-completion deadlocked inside update_goals')
        self.assertEqual(errors, [])
        self.assertEqual(goal.status, 'completed')
        self.assertEqual(goal.completion, 1.0)
        self.assertTrue(engine._lock.acquire(timeout=0.2))
        engine._lock.release()


if __name__ == '__main__':
    unittest.main()
