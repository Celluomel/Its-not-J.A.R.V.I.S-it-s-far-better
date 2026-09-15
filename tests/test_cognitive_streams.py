"""Regression tests for pressure -> event bus -> cognitive reactions."""
import tempfile
import unittest
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace as NS

# Import the two lightweight modules without executing cognition/__init__.py,
# which eagerly imports optional numerical runtime dependencies.
ROOT = Path(__file__).resolve().parents[1]
package = ModuleType("cognition")
package.__path__ = [str(ROOT / "cognition")]
sys.modules.setdefault("cognition", package)
core_package = ModuleType("core")
core_package.__path__ = [str(ROOT / "core")]
sys.modules.setdefault("core", core_package)

from core.cognitive_event_bus import (
    CognitiveEventBus,
    CURIOSITY_PEAKED,
    DRIVE_SATISFIED,
    TENSION_RELEASED,
    wire_cognitive_bus,
)
from cognition.pressure_system import CRITICAL_THRESHOLD, PressureSystem


class _Curiosity:
    def __init__(self):
        self.calls = []

    def stimulate(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _Workspace:
    def __init__(self):
        self.events = []

    def broadcast(self, **payload):
        self.events.append(payload)


class CognitiveStreamTests(unittest.TestCase):
    def test_critical_pressure_reaches_curiosity_and_workspace(self):
        curiosity = _Curiosity()
        workspace = _Workspace()
        bus = CognitiveEventBus()
        organism = NS(event_bus=bus, curiosity=curiosity, workspace=workspace)
        wire_cognitive_bus(organism, bus)

        with tempfile.TemporaryDirectory() as folder:
            pressure = PressureSystem(organism, str(Path(folder) / "pressure.json"))
            pressure.reservoirs["epistemic"].level = CRITICAL_THRESHOLD
            pressure._emit_critical_events()

        events = bus.recent(10)
        self.assertTrue(any(e.event_type == CURIOSITY_PEAKED for e in events))
        self.assertTrue(curiosity.calls)
        self.assertTrue(any(e["source"] == "event_bus.curiosity_peak"
                            for e in workspace.events))

    def test_satiation_emits_satisfaction_and_release(self):
        bus = CognitiveEventBus()
        workspace = _Workspace()
        organism = NS(event_bus=bus, workspace=workspace)
        wire_cognitive_bus(organism, bus)

        with tempfile.TemporaryDirectory() as folder:
            pressure = PressureSystem(organism, str(Path(folder) / "pressure.json"))
            before = pressure.reservoirs["social"].level
            after = pressure.satiate("social", 0.10)

        self.assertLess(after, before)
        event_types = [event.event_type for event in bus.recent(10)]
        self.assertIn(DRIVE_SATISFIED, event_types)
        self.assertIn(TENSION_RELEASED, event_types)
        self.assertTrue(any(e["source"] == "event_bus.recovery"
                            for e in workspace.events))


if __name__ == "__main__":
    unittest.main()
