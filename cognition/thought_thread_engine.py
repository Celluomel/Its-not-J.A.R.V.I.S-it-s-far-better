"""
ThoughtThreadEngine (TTE)
=========================
Converts Lumina's isolated word-chains into persistent, stateful thought
processes — threads that evolve over multiple cycles, accumulate context,
pursue goals, and resolve into memory and identity updates.

The core insight from the diagnosis:
    Lumina produces:  word → likely next word → likely next word
    It should produce: goal → explore → discover → update → resolve

Each ThoughtThread is a cognitive unit that:
  - persists across interaction cycles
  - accumulates evidence and context
  - drives planning and action
  - resolves into memory + identity changes

The TTE step() method is called each slow cycle from InternalThoughtLoop.
The DominantThoughtSelector then picks the highest-priority thread to drive
the next LLM output via CognitivePreProcessor.

Thread lifecycle:
  spawn → advance → [branch] → resolve → archive → [reactivate]

Thread sources:
  - curiosity signal (passive curiosity → active inquiry)
  - user input (extracted topic from conversation)
  - pressure buildup (identity / epistemic / expression)
  - dissonance detection (CDE injects meta-threads)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ACTIVE_THREADS  = 12   # cap to avoid cognitive overload
MAX_HISTORY_ENTRIES = 20   # per thread
THREAD_IDLE_HOURS   = 2.0  # defer threads inactive for this long
MIN_CONFIDENCE      = 0.05 # below this → thread is failed/archived
ADVANCE_CONFIDENCE_GAIN = 0.05
PERSISTENCE_PATH    = "data/persona/thought_threads.json"


# ── Enums ─────────────────────────────────────────────────────────────────────

class ThoughtThreadStatus(str, Enum):
    ACTIVE    = "active"
    ENGAGED   = "engaged"    # dominant — being acted upon right now
    DEFERRED  = "deferred"   # paused, may reactivate
    COMPLETED = "completed"
    FAILED    = "failed"
    RETIRED   = "retired"    # gracefully retired after max iterations (Phase 3)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ThoughtThread:
    id:              str
    topic:           str
    goal:            str
    current_state:   Dict[str, Any]        = field(default_factory=dict)
    history:         List[Dict[str, Any]]  = field(default_factory=list)
    planned_actions: List[str]             = field(default_factory=list)
    memory_refs:     List[str]             = field(default_factory=list)
    confidence:      float                 = 0.60
    uncertainty:     float                 = 0.40
    pressure:        Dict[str, float]      = field(default_factory=dict)
    curiosity:       float                 = 0.30
    dissonance:      float                 = 0.0
    energy_cost:     float                 = 10.0
    safety_risk:     float                 = 0.0
    parent_id:       Optional[str]         = None
    children_ids:    List[str]             = field(default_factory=list)
    status:          ThoughtThreadStatus   = ThoughtThreadStatus.ACTIVE
    created_at:      float                 = field(default_factory=time.time)
    last_advanced:   float                 = field(default_factory=time.time)
    source:          str                   = "internal"  # curiosity / user / pressure / dissonance

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600.0

    def idle_hours(self) -> float:
        return (time.time() - self.last_advanced) / 3600.0

    def to_prompt_line(self) -> str:
        """Compact representation for injection into system prompt."""
        action = self.planned_actions[0] if self.planned_actions else "none"
        return (
            f"[THREAD:{self.topic}] goal={self.goal!r} "
            f"confidence={self.confidence:.0%} next_action={action!r}"
        )


@dataclass
class TTEContext:
    """Snapshot of global state consumed by the TTE each cycle."""
    metrics:        Dict[str, float]  = field(default_factory=dict)
    identity_state: Dict[str, Any]    = field(default_factory=dict)
    world_state:    Dict[str, Any]    = field(default_factory=dict)
    pressures:      Dict[str, float]  = field(default_factory=dict)
    curiosity_level: float            = 0.30
    energy:         float             = 80.0


# ── Engine ────────────────────────────────────────────────────────────────────

class ThoughtThreadEngine:
    """
    Manages the lifecycle of Lumina's thought threads.

    Wire-up (in InternalThoughtLoop._slow_cycle):
        tte_ctx = ThoughtThreadEngine.collect_context(organism)
        tte.step(tte_ctx)
        dominant = dts.select(threads=tte.active_threads(), ctx=...)
    """

    def __init__(self, persistence_path: str = PERSISTENCE_PATH) -> None:
        self._path     = Path(persistence_path)
        self._threads: Dict[str, ThoughtThread] = {}
        self._lock     = threading.RLock()
        self._load()
        logger.info(f"[TTE] Initialised — {len(self._threads)} threads loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def spawn_thread(self, ctx: TTEContext, seed: Dict[str, Any]) -> ThoughtThread:
        """
        Create a new thread from a trigger.
        seed keys: topic, goal, source, curiosity, pressure, planned_actions
        """
        with self._lock:
            # Deduplicate: don't spawn if same topic is already ACTIVE/ENGAGED
            topic = seed.get("topic", "unknown").lower()[:80]
            for t in self._threads.values():
                if t.topic.lower() == topic and t.status in (
                    ThoughtThreadStatus.ACTIVE, ThoughtThreadStatus.ENGAGED
                ):
                    t.confidence = min(1.0, t.confidence + 0.05)
                    t.last_advanced = time.time()
                    logger.debug(f"[TTE] Reinforced existing thread: {topic!r}")
                    return t

            thread = ThoughtThread(
                id              = uuid.uuid4().hex[:12],
                topic           = topic,
                goal            = seed.get("goal", f"explore {topic}"),
                curiosity       = seed.get("curiosity", ctx.curiosity_level),
                pressure        = seed.get("pressure", {}),
                energy_cost     = seed.get("energy_cost", 10.0),
                planned_actions = seed.get("planned_actions", ["gather information"]),
                source          = seed.get("source", "internal"),
            )
            # Energy scales confidence cap
            thread.confidence = min(0.70, 0.40 + ctx.energy / 200.0)

            self._threads[thread.id] = thread
            self._save()
            logger.debug(f"[TTE] Spawned thread {thread.id[:8]}: {topic!r} from {thread.source}")
            return thread

    def advance_thread(self, thread: ThoughtThread, ctx: TTEContext) -> ThoughtThread:
        """
        Evolve a thread one step: update state, confidence, history.
        Called by step() for each active thread.
        """
        with self._lock:
            if thread.status not in (ThoughtThreadStatus.ACTIVE, ThoughtThreadStatus.ENGAGED):
                return thread

            # Record history entry
            snapshot = {
                "ts":           time.time(),
                "current_state": dict(thread.current_state),
                "confidence":   thread.confidence,
                "action":       thread.planned_actions[0] if thread.planned_actions else None,
            }
            thread.history.append(snapshot)
            if len(thread.history) > MAX_HISTORY_ENTRIES:
                thread.history = thread.history[-MAX_HISTORY_ENTRIES:]

            # Rotate planned actions
            if len(thread.planned_actions) > 1:
                thread.planned_actions = thread.planned_actions[1:]

            # Confidence grows slightly each advance if energy available
            energy_mod = max(0.5, ctx.energy / 100.0)
            thread.confidence = min(
                0.95,
                thread.confidence + ADVANCE_CONFIDENCE_GAIN * energy_mod
            )
            thread.uncertainty = max(0.0, 1.0 - thread.confidence - 0.1)
            thread.last_advanced = time.time()

            # Auto-complete when goal is satisfied (confidence high, no actions left)
            if thread.confidence >= 0.90 and not thread.planned_actions:
                thread.status = ThoughtThreadStatus.COMPLETED
                thread.current_state["resolution"] = "goal_reached"
                logger.debug(f"[TTE] Thread {thread.id[:8]} auto-completed: {thread.topic!r}")

            self._save()
            return thread

    def branch_thread(
        self, thread: ThoughtThread, ctx: TTEContext, sub_goal: str
    ) -> ThoughtThread:
        """Create a child thread for a sub-goal discovered during exploration."""
        child = self.spawn_thread(ctx, {
            "topic":           f"{thread.topic}:{sub_goal[:40]}",
            "goal":            sub_goal,
            "source":          "branch",
            "curiosity":       thread.curiosity * 0.8,
            "pressure":        dict(thread.pressure),
            "energy_cost":     thread.energy_cost * 0.7,
            "planned_actions": ["investigate sub-goal"],
        })
        with self._lock:
            child.parent_id = thread.id
            thread.children_ids.append(child.id)
            self._save()
        logger.debug(f"[TTE] Branched {thread.id[:8]} → child {child.id[:8]}: {sub_goal!r}")
        return child

    def resolve_thread(self, thread: ThoughtThread, ctx: TTEContext) -> ThoughtThread:
        """Mark thread completed/failed and prepare it for archiving."""
        with self._lock:
            if thread.confidence >= 0.50:
                thread.status = ThoughtThreadStatus.COMPLETED
            else:
                thread.status = ThoughtThreadStatus.FAILED
            thread.current_state["resolved_at"] = time.time()
            self._save()
        logger.debug(f"[TTE] Resolved {thread.id[:8]}: {thread.status.value}")
        return thread

    def step(self, ctx: TTEContext) -> List[ThoughtThread]:
        """
        Main tick — called each slow cycle.
        Returns list of currently active threads after update.
        """
        with self._lock:
            threads = list(self._threads.values())

        # ── Auto-correction based on metrics ──────────────────────────────────
        ccs     = ctx.metrics.get("CCS", 0.5)
        energy  = ctx.energy

        for thread in threads:
            if thread.status not in (ThoughtThreadStatus.ACTIVE, ThoughtThreadStatus.ENGAGED):
                continue

            # Defer idle threads
            if thread.idle_hours() > THREAD_IDLE_HOURS:
                with self._lock:
                    thread.status = ThoughtThreadStatus.DEFERRED
                continue

            # Low CCS: don't advance fragile threads — favour focused ones
            if ccs < 0.15 and thread.confidence < 0.4:
                continue

            # Low energy: reduce cost, still advance (reduce intensity, not stop)
            cost_mod = max(0.3, energy / 100.0)
            thread.energy_cost = thread.energy_cost * cost_mod

            self.advance_thread(thread, ctx)

        # ── Spawn from curiosity with Goal Fitness scoring ─────────────────
        # Gemini insight: goal spawning should be weighted by drive alignment
        # AND tension relief, not just curiosity level. High-pressure topics
        # score higher and take priority.
        if energy > 20.0 and ctx.curiosity_level > 0.3:
            curiosity_topic = ctx.world_state.get("top_curiosity_topic")
            if curiosity_topic:
                # Goal Fitness = drive_alignment * 0.5 + tension_relief * 0.5
                # drive_alignment: curiosity level (how much organism wants this)
                # tension_relief: how much spawning this thread reduces dominant pressure
                dominant_pressure = max(ctx.pressures.values()) if ctx.pressures else 0.3
                tension_relief    = min(1.0, dominant_pressure * 1.5)
                goal_fitness      = (ctx.curiosity_level * 0.5) + (tension_relief * 0.5)

                # Only spawn if fitness clears threshold — prevents low-value spam
                if goal_fitness > 0.35:
                    # Guard: reject noise topics before spawning a thread.
                    # Topics like "statement", "question", "verbe" survive
                    # curiosity_engine's IDF filter (low df in small corpora)
                    # but produce deferred threads that never resolve.
                    _topic_ok = True
                    if len(curiosity_topic) < 4:
                        _topic_ok = False  # too short to be meaningful
                    elif curiosity_topic in {
                        "statement", "question", "answer", "exploration",
                        "uncertainty", "verbe", "verb", "noun", "word",
                        "thing", "something", "nothing", "feeling",
                        "emotion", "thought", "response", "sentence",
                        "continue", "continue...", "conversation",
                    }:
                        _topic_ok = False  # grammatical noise
                    elif len([t for t in self._threads.values()
                              if t.topic == curiosity_topic
                              and t.status.value == "deferred"]) >= 3:
                        _topic_ok = False  # deferred 3+ times — give up
                    if _topic_ok:
                        self.spawn_thread(ctx, {
                            "topic":           curiosity_topic,
                            "goal":            f"explore and understand: {curiosity_topic}",
                            "source":          "curiosity",
                            "curiosity":       ctx.curiosity_level,
                            "energy_cost":     max(5.0, 15.0 * (1.0 - goal_fitness)),
                            "planned_actions": ["research topic", "synthesize findings",
                                               "update world model", "resolve tension"],
                            "current_state":   {"goal_priority": goal_fitness},
                        })

        # ── Spawn from vision surprise if available ────────────────────────
        # Gemini insight: sensory surprise should spawn a thought thread,
        # not just broadcast to GW. Vision events that are novel/salient
        # become "Perceptual Threads" with their own goal and history.
        vision_surprise = ctx.world_state.get("vision_surprise")
        if vision_surprise and energy > 25.0:
            salience = vision_surprise.get("salience", 0.0)
            if salience > 0.6:
                self.spawn_thread(ctx, {
                    "topic":           f"visual_observation:{vision_surprise.get('summary','unknown')[:40]}",
                    "goal":            f"understand what was observed: {vision_surprise.get('summary','unknown')[:60]}",
                    "source":          "sensory_bridge",
                    "curiosity":       salience,
                    "energy_cost":     8.0,
                    "planned_actions": ["analyze observation", "link to known context", "update world model"],
                })
                ctx.world_state.pop("vision_surprise", None)  # consume event

        # ── Prune: cap at MAX_ACTIVE_THREADS ──────────────────────────────────
        with self._lock:
            active = [t for t in self._threads.values()
                      if t.status in (ThoughtThreadStatus.ACTIVE, ThoughtThreadStatus.ENGAGED)]
            if len(active) > MAX_ACTIVE_THREADS:
                # Archive lowest-confidence surplus
                active.sort(key=lambda t: t.confidence)
                for t in active[: len(active) - MAX_ACTIVE_THREADS]:
                    t.status = ThoughtThreadStatus.DEFERRED
            self._save()

        return self.active_threads()

    def active_threads(self) -> List[ThoughtThread]:
        with self._lock:
            return [
                t for t in self._threads.values()
                if t.status in (ThoughtThreadStatus.ACTIVE, ThoughtThreadStatus.ENGAGED)
            ]

    def all_threads(self) -> List[ThoughtThread]:
        with self._lock:
            return list(self._threads.values())

    def get_thread(self, thread_id: str) -> Optional[ThoughtThread]:
        with self._lock:
            return self._threads.get(thread_id)

    @staticmethod
    def collect_context(organism: Any) -> TTEContext:
        """
        Collect a TTEContext from the live organism object.
        Safe: any missing attribute returns a sensible default.
        """
        ctx = TTEContext()
        try:
            obs = getattr(organism, "observatory", None)
            if obs:
                snap = obs.snapshot()
                ctx.metrics = {
                    "CCS": snap.get("CCS", snap.get("cognitive_coherence", 0.5)),
                    "GEI": snap.get("GEI", snap.get("goal_engagement", 0.3)),
                    "IDX": snap.get("IDX", snap.get("identity_index", 0.5)),
                    "STR": snap.get("STR", snap.get("strangeness", 0.4)),
                }
        except Exception:
            pass
        try:
            ctx.energy = organism.energy.level() if hasattr(organism, "energy") else 80.0
        except Exception:
            pass
        try:
            pressure_sys = getattr(organism, "pressure", None)
            if pressure_sys:
                ctx.pressures = {
                    k: getattr(r, "level", 0.0)
                    for k, r in getattr(pressure_sys, "reservoirs", {}).items()
                }
        except Exception:
            pass
        try:
            curiosity = getattr(organism, "curiosity", None)
            if curiosity:
                ctx.curiosity_level = getattr(curiosity, "_global", 0.3)
                top = curiosity.top_topic()
                if top:
                    ctx.world_state["top_curiosity_topic"] = top
        except Exception:
            pass

        # ── Vision surprise → sensory bridge signal ──────────────────────────
        # If the GW has a recent high-priority ambient_vision item that is novel,
        # package it as a vision_surprise for TTE to spawn a perceptual thread.
        try:
            ws = getattr(organism, "workspace", None)
            if ws:
                recent_gw = ws.recent(8)
                for item in reversed(recent_gw):
                    src = getattr(item, "source", "")
                    pri = getattr(item, "priority", 0.0)
                    if src == "ambient_vision" and pri >= 0.50:
                        content = str(getattr(item, "content", ""))
                        # Salience from priority (ambient vision is capped at 0.28 normally,
                        # so anything >= 0.50 is a genuine surprise from the event bus)
                        ctx.world_state["vision_surprise"] = {
                            "salience": min(1.0, pri * 1.5),
                            "summary":  content[:80],
                        }
                        break
        except Exception:
            pass

        return ctx

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                tid: {**asdict(t), "status": t.status.value}
                for tid, t in self._threads.items()
            }
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[TTE] Save error: {e}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for tid, td in data.items():
                td["status"] = ThoughtThreadStatus(td.get("status", "active"))
                self._threads[tid] = ThoughtThread(**td)
            # Only keep non-completed threads
            self._threads = {
                tid: t for tid, t in self._threads.items()
                if t.status not in (ThoughtThreadStatus.COMPLETED, ThoughtThreadStatus.FAILED)
            }
        except Exception as e:
            logger.warning(f"[TTE] Load error (starting fresh): {e}")
