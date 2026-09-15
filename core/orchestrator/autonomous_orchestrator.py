"""
Autonomous Orchestrator
=======================
The central nervous system of Lumina — the loop that runs even when no user
is present and decides what Lumina does at every moment.

Architecture
------------

    Inputs (UI / Voice / Telegram / Vision)
               │
          EventSystem                 ← receives all external events
               │
     AutonomousOrchestrator
          │         │
    DriveSystem   CognitiveClock      ← motivation + timescales
          │         │
          └────┬────┘
               │
       ActivitySelector               ← chooses the next action
               │
       ExecutionLayer                 ← runs the action
               │
     CognitiveOrganism               ← existing cognitive modules

Integration
-----------
The orchestrator does NOT replace any existing module.  It wraps around the
CognitiveOrganism and PersonaBridge and adds:

  1. A background asyncio task that polls every 2 seconds
  2. Clock-driven background cycles (reflection, consolidation, evolution)
  3. A drive vector derived from live module states
  4. Priority-based activity selection

Usage (from app.py / state.py)
-------------------------------
    from core.orchestrator.autonomous_orchestrator import AutonomousOrchestrator

    orchestrator = AutonomousOrchestrator(organism)
    asyncio.create_task(orchestrator.start())

    # Push a user message into the event queue:
    orchestrator.notify_user_message(text, user_id)

    # Push from Telegram/WhatsApp:
    orchestrator.notify_external(platform, user_id, text)

    # Stop gracefully:
    await orchestrator.stop()
"""

import asyncio
import logging
import time
from typing import Any, Optional

from .event_system import (
    EventSystem, REFLECTION_DUE, CONSOLIDATION_DUE, EVOLUTION_DUE,
)
from .cognitive_clock import CognitiveClock
from .drive_system    import DriveSystem
from .activity_selector import ActivitySelector
from .execution_layer   import ExecutionLayer
from cognition.global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)

# Seconds between orchestrator cycles (CPU impact is negligible)
CYCLE_INTERVAL = 2.0


class AutonomousOrchestrator:
    """
    Central nervous system — reactive + proactive cognitive loop.

    Parameters
    ----------
    organism  : CognitiveOrganism
        The fully-initialised organism (energy, curiosity, goals, homeostasis, …)
    workspace : GlobalWorkspace | None
        Shared broadcast bus.  A new one is created if not supplied.
    """

    def __init__(self, organism: Any, workspace: Optional[GlobalWorkspace] = None):
        self._organism  = organism
        self._workspace = workspace or GlobalWorkspace()

        # Sub-systems
        self._events    = EventSystem()
        self._clock     = CognitiveClock()
        self._drives    = DriveSystem(organism)
        self._selector  = ActivitySelector()
        self._execution = ExecutionLayer(organism, self._workspace)

        self._running   = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_activity: str = "none"

        logger.info("🧠 AutonomousOrchestrator created")

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def event_system(self) -> EventSystem:
        return self._events

    @property
    def workspace(self) -> GlobalWorkspace:
        return self._workspace

    @property
    def execution(self) -> ExecutionLayer:
        return self._execution

    def notify_user_message(self, text: str, user_id: str = "default") -> None:
        """Call from the UI / voice pipeline when a user message arrives."""
        self._events.push_user_message(text, user_id)
        self._drives.mark_user_interaction()

    def notify_external(self, platform: str, user_id: str, text: str) -> None:
        """Call from MessagingManager when a Telegram/WhatsApp message arrives."""
        self._events.push_external(platform, user_id, text)
        self._drives.mark_user_interaction()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background cognitive loop as an asyncio task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="autonomous-orchestrator")
        logger.info("🧠 AutonomousOrchestrator started")

    async def stop(self) -> None:
        """Gracefully stop the loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 AutonomousOrchestrator stopped")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while self._running:
            try:
                await self._cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Orchestrator] cycle error: {e}")
            await asyncio.sleep(CYCLE_INTERVAL)

    async def _cycle(self) -> None:
        self._cycle_count += 1

        # ── 1. Collect pending events ────────────────────────────────────
        events = self._events.collect()

        # ── 2. Inject clock-driven events ───────────────────────────────
        if self._clock.medium_due():
            events.append(_clock_event(REFLECTION_DUE))
            self._clock.mark_medium()

        if self._clock.slow_due():
            events.append(_clock_event(CONSOLIDATION_DUE))
            self._clock.mark_slow()

        if self._clock.evolution_due():
            events.append(_clock_event(EVOLUTION_DUE))
            self._clock.mark_evolution()

        # ── 3. Compute drive vector ──────────────────────────────────────
        drives = self._drives.compute()

        # ── 4. Select activity ───────────────────────────────────────────
        activity, payload = self._selector.select(events, drives, organism=self._organism)

        # ── 5. Execute (skip idle on every cycle to reduce noise) ────────
        if activity != "idle_reflection" or self._cycle_count % 10 == 0:
            await self._execution.execute(activity, payload)
            self._last_activity = activity

            if self._cycle_count % 20 == 0:  # log every 40 s
                logger.debug(
                    f"[Orchestrator] cycle={self._cycle_count} "
                    f"activity={activity!r} "
                    f"drives={drives.as_dict()}"
                )

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running":       self._running,
            "cycle_count":   self._cycle_count,
            "last_activity": self._last_activity,
            "drives":        self._drives.compute().as_dict(),
            "clock":         self._clock.status(),
            "workspace":     self._workspace.summary(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clock_event(event_type: str):
    from .event_system import Event
    return Event(type=event_type, priority=0.4, source="cognitive_clock")
