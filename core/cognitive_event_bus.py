"""
CognitiveEventBus — Internal Module Communication
===================================================
Allows cognitive modules to react to each other's state changes
without direct coupling. Pure publish/subscribe.

Without this:
    Each module runs in its own loop.
    A contradiction detected by contradiction_handler
    doesn't automatically raise curiosity or increase tension.

With this:
    contradiction_detected  → curiosity.stimulate(topic)
                            → tension_engine.increase_pressure()
                            → goal_ecology.activate("resolve")

    prediction_error        → curiosity.stimulate(unexpected_domain)
                            → emotional_state.trigger("surprise")

    energy_critical         → goal_ecology.suppress_exploration()
                            → attention.focus("rest")

    memory_intrusion        → workspace.broadcast(memory)
                            → curiosity.stimulate(memory_topic)

This creates emergent cognitive chains — unscripted reactions
between modules that produce non-deterministic but coherent behaviour.

Usage:
    bus = CognitiveEventBus()

    # Subscribe
    bus.on("contradiction_detected", handler_fn)

    # Emit
    bus.emit("contradiction_detected", {
        "topic":    "identity",
        "pressure": 0.75,
        "source":   "self_concept",
    })

Thread-safe. Handlers are called synchronously in emit().
For async handlers, wrap with asyncio.create_task().
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CognitiveEvent:
    event_type: str
    payload:    Dict[str, Any]
    source:     str = "unknown"
    timestamp:  float = field(default_factory=time.time)


# ── Known event types ─────────────────────────────────────────────────────────
CONTRADICTION_DETECTED  = "contradiction_detected"
PREDICTION_ERROR        = "prediction_error"
ENERGY_CRITICAL         = "energy_critical"
ENERGY_RESTORED         = "energy_restored"
DRIVE_SPIKE             = "drive_spike"          # a drive jumped above threshold
DRIVE_SATISFIED         = "drive_satisfied"       # a drive was fulfilled
CURIOSITY_PEAKED        = "curiosity_peaked"      # curiosity hit high urgency
TENSION_RELEASED        = "tension_released"      # tension resolved
MEMORY_INTRUSION        = "memory_intrusion"      # spontaneous memory surfaced
LIFE_EVENT_RECORDED     = "life_event_recorded"
SURPRISE_DETECTED       = "surprise_detected"     # prediction error > threshold
IDENTITY_THREATENED     = "identity_threatened"   # self-concept violation
IDENTITY_AFFIRMED       = "identity_affirmed"


class CognitiveEventBus:
    """
    Decoupled event bus for inter-module cognitive communication.
    Singleton pattern — one bus per organism.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock      = threading.Lock()
        self._history:  List[CognitiveEvent] = []
        self._max_history = 200
        self._total_emitted  = 0
        self._total_handled  = 0

    # ── Registration ──────────────────────────────────────────────────────────

    def on(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type."""
        with self._lock:
            self._handlers[event_type].append(handler)
        logger.debug(f"[EventBus] registered handler for '{event_type}'")

    def off(self, event_type: str, handler: Callable) -> None:
        """Unregister a handler."""
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]

    # ── Emission ──────────────────────────────────────────────────────────────

    def emit(
        self,
        event_type: str,
        payload:    Dict[str, Any],
        source:     str = "unknown",
        priority:   Optional[float] = None,
    ) -> int:
        """
        Emit an event. All registered handlers are called synchronously.
        Returns the number of handlers that were called.
        """
        # Keep priority as event metadata when producers provide it.  Older
        # producers only pass the three original arguments, so this remains
        # backwards compatible while allowing pressure/perception signals to
        # preserve their urgency across the bus boundary.
        event_payload = dict(payload)
        if priority is not None:
            event_payload.setdefault("priority", priority)
        event = CognitiveEvent(
            event_type=event_type, payload=event_payload, source=source
        )

        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            self._total_emitted += 1

        called = 0
        for handler in handlers:
            try:
                handler(event)
                called += 1
            except Exception as e:
                logger.debug(f"[EventBus] handler error for '{event_type}': {e}")

        self._total_handled += called

        if called > 0:
            logger.debug(
                f"[EventBus] '{event_type}' from '{source}' → {called} handler(s)"
            )

        return called

    # ── Query ─────────────────────────────────────────────────────────────────

    def recent(self, n: int = 10) -> List[CognitiveEvent]:
        with self._lock:
            return list(self._history[-n:])

    def recent_of_type(self, event_type: str, n: int = 5) -> List[CognitiveEvent]:
        with self._lock:
            filtered = [e for e in self._history if e.event_type == event_type]
            return filtered[-n:]

    def summary(self) -> Dict:
        with self._lock:
            type_counts = defaultdict(int)
            for e in self._history:
                type_counts[e.event_type] += 1
            return {
                "total_emitted":  self._total_emitted,
                "total_handled":  self._total_handled,
                "registered_types": list(self._handlers.keys()),
                "recent_types":   dict(type_counts),
            }


def wire_cognitive_bus(organism: Any, bus: CognitiveEventBus) -> None:
    """
    Wire all standard cognitive reactions between modules.
    Called once after organism is initialised.

    This is the central place where emergent cognitive chains are defined.
    """

    # ── 1. Contradiction detected → raise tension + stimulate curiosity ───────
    def on_contradiction(event: CognitiveEvent):
        topic    = event.payload.get("topic", "")
        pressure = float(event.payload.get("pressure", 0.5))

        # Raise tension
        te = getattr(organism, 'tension_engine', None)
        if te and hasattr(te, 'add_contradiction_pressure'):
            try:
                te.add_contradiction_pressure(pressure * 0.6)
            except Exception:
                pass

        # Stimulate curiosity toward the contradiction topic
        cu = getattr(organism, 'curiosity', None)
        if cu and topic:
            try:
                cu.stimulate(topic, amount=pressure * 0.4, source="contradiction")
            except Exception:
                pass

        # Broadcast to workspace
        ws = getattr(organism, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.contradiction",
                    content=f"Contradiction detected on: {topic}",
                    priority=0.5 + pressure * 0.3,
                )
            except Exception:
                pass

    bus.on(CONTRADICTION_DETECTED, on_contradiction)

    # ── 2. Prediction error / surprise → curiosity boost + emotional note ─────
    def on_surprise(event: CognitiveEvent):
        domain     = event.payload.get("domain", "")
        error_lvl  = float(event.payload.get("error_level", 0.5))

        # Curiosity toward the unexpected domain
        cu = getattr(organism, 'curiosity', None)
        if cu and domain:
            try:
                cu.stimulate(domain, amount=error_lvl * 0.35, source="surprise")
            except Exception:
                pass

        # Broadcast to workspace
        ws = getattr(organism, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.surprise",
                    content=f"Unexpected: {domain or 'unknown domain'}",
                    priority=0.4 + error_lvl * 0.4,
                )
            except Exception:
                pass

    bus.on(SURPRISE_DETECTED, on_surprise)
    bus.on(PREDICTION_ERROR,  on_surprise)   # alias

    # Curiosity peaks are emitted by PressureSystem when epistemic or
    # uncertainty pressure becomes critical.  They are not prediction errors:
    # the drive itself is the signal, so route it into an explicit exploratory
    # workspace candidate instead of silently recording an unhandled event.
    def on_curiosity_peak(event: CognitiveEvent):
        drive = event.payload.get("drive", "epistemic")
        pressure = float(event.payload.get("pressure", 0.5))
        cu = getattr(organism, "curiosity", None)
        if cu and hasattr(cu, "stimulate"):
            try:
                cu.stimulate(str(drive), amount=min(0.35, pressure * 0.35),
                             source="pressure_peak")
            except Exception:
                pass
        ws = getattr(organism, "workspace", None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.curiosity_peak",
                    content=f"Curiosity peak: {drive}",
                    priority=min(0.9, 0.55 + pressure * 0.35),
                )
            except Exception:
                pass

    bus.on(CURIOSITY_PEAKED, on_curiosity_peak)

    # Recovery is also a cognitive signal.  Without these handlers, satiation
    # changed reservoir numbers but downstream attention and workspace state
    # never learned that the underlying need had been met.
    def on_drive_satisfied(event: CognitiveEvent):
        drive = event.payload.get("drive", "unknown")
        ws = getattr(organism, "workspace", None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.recovery",
                    content=f"Drive satisfied: {drive}",
                    priority=0.35,
                )
            except Exception:
                pass

    bus.on(DRIVE_SATISFIED, on_drive_satisfied)
    bus.on(TENSION_RELEASED, on_drive_satisfied)

    # ── 3. Energy critical → suppress exploration + signal workspace ──────────
    def on_energy_critical(event: CognitiveEvent):
        level = float(event.payload.get("level", 0.0))

        # Decay curiosity (no energy to explore)
        cu = getattr(organism, 'curiosity', None)
        if cu and hasattr(cu, 'decay_all'):
            try:
                cu.decay_all()
            except Exception:
                pass

        ws = getattr(organism, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.energy",
                    content=f"Energy critical ({level:.0f}%) — conserving",
                    priority=0.7,
                )
            except Exception:
                pass

    bus.on(ENERGY_CRITICAL, on_energy_critical)

    # ── 4. Energy restored → allow curiosity to resume ───────────────────────
    def on_energy_restored(event: CognitiveEvent):
        cu = getattr(organism, 'curiosity', None)
        if cu:
            try:
                cu.stimulate("general exploration", amount=0.15, source="energy_restored")
            except Exception:
                pass

    bus.on(ENERGY_RESTORED, on_energy_restored)

    # ── 5. Drive spike → workspace broadcast + attention focus ───────────────
    def on_drive_spike(event: CognitiveEvent):
        drive   = event.payload.get("drive", "unknown")
        urgency = float(event.payload.get("urgency", 0.6))

        ws = getattr(organism, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.drive",
                    content=f"Drive spike: {drive} urgency={urgency:.2f}",
                    priority=min(0.85, urgency),
                )
            except Exception:
                pass

        attn = getattr(organism, 'attention', None)
        if attn and hasattr(attn, 'set_focus'):
            try:
                attn.set_focus(drive)
            except Exception:
                pass

    bus.on(DRIVE_SPIKE, on_drive_spike)

    # ── 6. Memory intrusion → workspace broadcast + curiosity ────────────────
    def on_memory_intrusion(event: CognitiveEvent):
        content = event.payload.get("content", "")
        topic   = event.payload.get("topic", "")
        score   = float(event.payload.get("score", 0.5))

        ws = getattr(organism, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="memory_intrusion",
                    content=content[:120] if content else f"Memory: {topic}",
                    priority=score * 0.75,
                )
            except Exception:
                pass

        # Stimulate curiosity on the memory's topic
        cu = getattr(organism, 'curiosity', None)
        if cu and topic:
            try:
                cu.stimulate(topic, amount=score * 0.2, source="memory_intrusion")
            except Exception:
                pass

    bus.on(MEMORY_INTRUSION, on_memory_intrusion)

    # ── 7. Identity threatened → tension increase ────────────────────────────
    def on_identity_threatened(event: CognitiveEvent):
        belief   = event.payload.get("belief", "")
        severity = float(event.payload.get("severity", 0.5))

        te = getattr(organism, 'tension_engine', None)
        if te and hasattr(te, 'add_contradiction_pressure'):
            try:
                te.add_contradiction_pressure(severity * 0.5)
            except Exception:
                pass

        ws = getattr(organism, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="event_bus.identity",
                    content=f"Identity tension: {belief[:60]}",
                    priority=0.5 + severity * 0.35,
                )
            except Exception:
                pass

    bus.on(IDENTITY_THREATENED, on_identity_threatened)

    # ── 8. Life event recorded → narrative update + evolution trigger ─────────
    def on_life_event(event: CognitiveEvent):
        title       = event.payload.get("title", "event")
        significance = float(event.payload.get("significance", 0.5))

        ni = getattr(organism, 'narrative_identity', None)
        if ni:
            try:
                ni.record_chapter(
                    title=title,
                    description=event.payload.get("description", title),
                    emotion=event.payload.get("emotion", "neutral"),
                    significance=significance,
                )
            except Exception:
                pass

    bus.on(LIFE_EVENT_RECORDED, on_life_event)

    logger.info(
        f"[EventBus] Wired {len(bus._handlers)} cognitive reaction chains"
    )
