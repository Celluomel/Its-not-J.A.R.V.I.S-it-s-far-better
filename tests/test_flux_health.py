"""Network telemetry uses actual exchanges, not human chat history."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
package = ModuleType('cognition')
package.__path__ = [str(ROOT / 'cognition')]
with patch.dict(sys.modules, {'cognition': package}):
    from cognition.flux_mind_model import FluxMindModel


class FluxHealthTests(unittest.TestCase):
    def test_profile_uses_network_pairs_once(self):
        messages = []
        for i in range(3):
            messages.extend([
                NS(direction='out', sender_id='master', text='Hello', timestamp=i),
                NS(direction='in', sender_id='child', text='Why this mechanism?', timestamp=i),
            ])
        state_module = ModuleType('core.state')
        state_module.state = NS(lumina_network=NS(_history={'one': messages}))
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                sys.modules, {'core.state': state_module}):
            model = FluxMindModel(None, None, str(Path(folder) / 'profile.json'))
            model._update()
            model._update()
            self.assertEqual(model.status()['exchanges_analysed'], 3)
            self.assertEqual(model.status()['expected_dialogue'], 1)

    def test_connection_or_failed_exchange_does_not_count(self):
        state_module = ModuleType('core.state')
        state_module.state = NS(lumina_network=NS(_history={'one': [
            NS(direction='out', sender_id='master', text='Hello', timestamp=1),
            NS(direction='in', sender_id='child', text='[Network error: offline]', timestamp=2),
        ]}))
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                sys.modules, {'core.state': state_module}):
            model = FluxMindModel(None, None, str(Path(folder) / 'profile.json'))
            self.assertEqual(model._read_exchanges(), [])
            self.assertEqual(model.status()['exchanges_analysed'], 0)


if __name__ == '__main__':
    unittest.main()
