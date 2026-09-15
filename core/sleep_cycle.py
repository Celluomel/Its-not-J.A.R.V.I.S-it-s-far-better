"""
SleepCycleManager — Circadian Phase Controller
===============================================
Implements the biological insight: a cognitive organism cannot run all
cognitive tasks simultaneously.  It must alternate between phases, each
with a specific cognitive function and LLM budget.

Phases and their LLM policies:
─────────────────────────────────────────────────────────────────────
  ACTIVE  │ User interaction window.  LLM fully reserved for responses.
          │ Background tasks: BLOCKED (except fast maintenance).
          │ Trigger: user message received in last ACTIVE_WINDOW seconds.
─────────────────────────────────────────────────────────────────────
  IDLE    │ Light background cognition.  LLM available but rate-limited.
          │ Tasks: world model update, curiosity decay, light reflection.
          │ Trigger: no user for ACTIVE_WINDOW s, energy > 50%.
─────────────────────────────────────────────────────────────────────
  SLEEP   │ Heavy consolidation.  LLM used for deep tasks only.
          │ Tasks: memory summary, contradiction resolution,
          │        narrative identity update, learning cycle.
          │ Trigger: no user for SLEEP_ONSET s, energy > 30%.
─────────────────────────────────────────────────────────────────────
  DREAM   │ Generative exploration.  LLM used for simulation.
          │ Tasks: dream cycle, world model prediction,
          │        identity simulation.
          │ Trigger: after SLEEP phase, energy > 40%.
─────────────────────────────────────────────────────────────────────

Phase transitions are also driven by the PressureSystem:
  - high vitality pressure  → forces SLEEP regardless of timer
  - high epistemic pressure → accelerates DREAM onset

LLM scheduling integration:
  Each phase sets max_concurrent_background to 0 (ACTIVE) or 1 (others).
  Background tasks check can_run_background() before submitting to LLM.

Persists phase and timing state to: data/persona/sleep_cycle.json
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Timing constants (seconds) ────────────────────────────────────────────────
ACTIVE_WINDOW   = 300     # 5 min of interaction → ACTIVE
IDLE_ONSET      = 300     # 5 min of silence → IDLE
# Bug fix (v59): SLEEP_ONSET=900 + the old hardcoded 600s in-SLEEP-before-
# DREAM check required 25 continuous minutes of silence before a dream
# cycle could ever fire. sleep_cycle.json showed cycles_completed=4 and
# last_dream_at=null — confirming dream (and therefore the entire
# AspirationalSelf → LongHorizonPlanner pipeline) had never once
# triggered in this deployment's actual run history. Halved both so a
# dream cycle is reachable within realistic usage/dev-session gaps
# without eliminating the "real quiet time" design intent.
SLEEP_ONSET     = 600     # 10 min of silence → SLEEP
SLEEP_BEFORE_DREAM = 300  # 5 min in SLEEP → DREAM (was hardcoded 600 inline)
DREAM_DURATION  = 600     # DREAM lasts 10 min, then returns to ACTIVE/IDLE
MIN_PHASE_TIME  = 120     # minimum seconds in any phase before transitioning

# ── Energy thresholds ─────────────────────────────────────────────────────────
SLEEP_ENERGY_MIN  = 30.0   # don't enter SLEEP if energy critically low
DREAM_ENERGY_MIN  = 40.0   # don't enter DREAM if too depleted


class Phase(str, Enum):
    ACTIVE = "active"
    IDLE   = "idle"
    SLEEP  = "sleep"
    DREAM  = "dream"


# Tasks allowed per phase (priority = order in list)
PHASE_TASKS: Dict[Phase, List[str]] = {
    Phase.ACTIVE: [],                          # nothing background while user is here
    Phase.IDLE:   [
        "world_model_update",
        "curiosity_decay",
        "pressure_tick",
        "emotional_regulation",
    ],
    Phase.SLEEP:  [
        "memory_consolidation",               # session summary compression
        "contradiction_resolution",
        "narrative_identity_update",
        "learning_cycle",
        "identity_processing",
        "save_state",
    ],
    Phase.DREAM:  [
        "dream_cycle",
        "world_model_prediction",
        "predictive_mind_calibration",
    ],
}

# Which tasks make LLM calls (others are pure Python)
LLM_TASKS = {
    "memory_consolidation", "contradiction_resolution",
    "narrative_identity_update", "learning_cycle",
    "identity_processing", "dream_cycle",
    "world_model_prediction", "predictive_mind_calibration",
}


@dataclass
class SleepCycleState:
    phase:              str   = Phase.ACTIVE.value
    phase_entered_at:   float = field(default_factory=time.time)
    last_user_activity: float = field(default_factory=time.time)
    dream_started_at:   Optional[float] = None
    cycles_completed:   int   = 0
    last_sleep_at:      Optional[float] = None
    last_dream_at:      Optional[float] = None


class SleepCycleManager:
    """
    Controls which cognitive tasks can run and when, based on the
    current circadian phase.

    Integration points:
      - tick()             : called every slow cycle from internal_loop.py
      - mark_user_active() : called from _pre_interaction()
      - can_run_task(name) : checked by BackgroundWorker before each task
      - phase              : read by orchestrator dashboard
    """

    def __init__(
        self,
        organism:    Any,
        path:        str = "data/persona/sleep_cycle.json",
    ):
        self._o    = organism
        self._path = Path(path)
        self._lock = Lock()
        self._state = SleepCycleState()
        self._task_queue: List[Tuple[str, Callable]] = []   # (task_name, callable)
        self._last_tick = time.time()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"[SleepCycle] Initialised — phase: {self._state.phase}")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def phase(self) -> Phase:
        return Phase(self._state.phase)

    def mark_user_active(self) -> None:
        """Call whenever a user message arrives."""
        with self._lock:
            self._state.last_user_activity = time.time()
            if self._state.phase != Phase.ACTIVE.value:
                self._transition_to(Phase.ACTIVE)

    def can_run_task(self, task_name: str) -> bool:
        """
        Returns True if this task is permitted in the current phase.
        BackgroundWorker and internal_loop call this before every background task.
        """
        allowed = PHASE_TASKS.get(Phase(self._state.phase), [])
        return task_name in allowed

    def can_run_background_llm(self) -> bool:
        """True if the current phase permits background LLM calls at all."""
        return self._state.phase != Phase.ACTIVE.value

    def tick(self) -> Phase:
        """
        Evaluate phase transitions.
        Called every slow cycle (~120s) from internal_loop.py.
        Returns current phase after potential transition.
        """
        with self._lock:
            self._last_tick = time.time()
            new_phase = self._compute_target_phase()
            if new_phase.value != self._state.phase:
                min_elapsed = time.time() - self._state.phase_entered_at
                if min_elapsed >= MIN_PHASE_TIME:
                    self._transition_to(new_phase)
            self._save()
            return Phase(self._state.phase)

    def schedule(self, task_name: str, fn: Callable) -> bool:
        """
        Add a task to the sleep queue.
        It will run when the phase permits it.
        Returns True if queued, False if already queued.
        """
        with self._lock:
            if any(t == task_name for t, _ in self._task_queue):
                return False   # already pending
            self._task_queue.append((task_name, fn))
            logger.debug(f"[SleepCycle] Queued: {task_name}")
            return True

    def run_pending(self) -> List[str]:
        """
        Execute queued tasks that are permitted in the current phase.
        Returns list of executed task names.
        Called from internal_loop slow cycle.
        """
        from core.llm_scheduler import llm_scheduler

        executed = []
        allowed  = set(PHASE_TASKS.get(Phase(self._state.phase), []))

        with self._lock:
            remaining = []
            for task_name, fn in self._task_queue:
                if task_name not in allowed:
                    remaining.append((task_name, fn))   # keep for later
                    continue
                if task_name in LLM_TASKS and llm_scheduler.is_busy():
                    remaining.append((task_name, fn))   # LLM busy, retry later
                    continue
                remaining.append((task_name, fn))       # execute outside lock
            # Reset queue to only non-executed ones
            self._task_queue = remaining

        # Execute outside lock to avoid deadlock
        for task_name, fn in list(self._task_queue):
            if task_name not in allowed:
                continue
            if task_name in LLM_TASKS and llm_scheduler.is_busy():
                continue
            try:
                from core.llm_scheduler import llm_scheduler
                with llm_scheduler.sync_slot(
                    priority=4,
                    skip_if_busy=(task_name in LLM_TASKS),
                    caller=f"sleep:{task_name}",
                ) as acquired:
                    if acquired or task_name not in LLM_TASKS:
                        logger.debug(f"[SleepCycle] Running: {task_name} (phase={self._state.phase})")
                        fn()
                        executed.append(task_name)
                        with self._lock:
                            self._task_queue = [(n, f) for n, f in self._task_queue if n != task_name]
            except Exception as e:
                logger.warning(f"[SleepCycle] Task {task_name!r} failed: {e}")
                with self._lock:
                    self._task_queue = [(n, f) for n, f in self._task_queue if n != task_name]

        return executed

    def summary(self) -> Dict:
        with self._lock:
            elapsed = time.time() - self._state.phase_entered_at
            idle_s  = time.time() - self._state.last_user_activity
            return {
                "phase":            self._state.phase,
                "phase_elapsed_s":  round(elapsed),
                "idle_seconds":     round(idle_s),
                "queued_tasks":     [t for t, _ in self._task_queue],
                "cycles_completed": self._state.cycles_completed,
                "last_sleep":       self._state.last_sleep_at,
                "last_dream":       self._state.last_dream_at,
            }

    # ── Phase logic ───────────────────────────────────────────────────────────

    def _compute_target_phase(self) -> Phase:
        """Determine which phase Lumina should be in right now."""
        now        = time.time()
        idle_s     = now - self._state.last_user_activity
        energy     = self._get_energy()
        current    = Phase(self._state.phase)

        # ── DREAM → back to IDLE/ACTIVE after duration ──────────────────
        if current == Phase.DREAM:
            if self._state.dream_started_at is None:
                self._state.dream_started_at = now
            if now - self._state.dream_started_at >= DREAM_DURATION:
                return Phase.IDLE

        # ── Force SLEEP if vitality pressure is critical ─────────────────
        vitality_pressure = self._get_vitality_pressure()
        if vitality_pressure > 0.85 and energy >= SLEEP_ENERGY_MIN:
            return Phase.SLEEP

        # ── Normal timer transitions ─────────────────────────────────────
        if idle_s < IDLE_ONSET:
            return Phase.ACTIVE

        if idle_s < SLEEP_ONSET:
            # High epistemic pressure while idle → skip to DREAM
            epistemic = self._get_epistemic_pressure()
            if epistemic > 0.75 and energy >= DREAM_ENERGY_MIN and current == Phase.SLEEP:
                return Phase.DREAM
            return Phase.IDLE

        # idle >= SLEEP_ONSET
        if energy < SLEEP_ENERGY_MIN:
            return Phase.IDLE   # too depleted to even sleep properly

        if current == Phase.SLEEP:
            # If sleep has been running long enough → dream
            sleep_elapsed = now - self._state.phase_entered_at
            if sleep_elapsed > SLEEP_BEFORE_DREAM and energy >= DREAM_ENERGY_MIN:
                return Phase.DREAM
            return Phase.SLEEP

        return Phase.SLEEP

    def _transition_to(self, target: Phase) -> None:
        """Execute a phase transition. Must be called under _lock."""
        prev = self._state.phase
        self._state.phase          = target.value
        self._state.phase_entered_at = time.time()

        if target == Phase.SLEEP:
            self._state.last_sleep_at = time.time()
            self._state.dream_started_at = None
        elif target == Phase.DREAM:
            self._state.last_dream_at    = time.time()
            self._state.dream_started_at = time.time()
        elif target == Phase.ACTIVE:
            self._state.cycles_completed += 1

        logger.info(f"[SleepCycle] {prev} → {target.value} "
                    f"(idle={time.time()-self._state.last_user_activity:.0f}s)")

        # Notify PressureSystem of rest cycles
        try:
            ps = getattr(self._o, 'pressure', None)
            if ps:
                if target == Phase.SLEEP:
                    ps.satiate("vitality", 0.20)
                elif target == Phase.DREAM:
                    ps.record_rest_cycle()
                    ps.satiate("epistemic", 0.15)
        except Exception:
            pass

    # ── External state readers ────────────────────────────────────────────────

    def _get_energy(self) -> float:
        try:
            return float(getattr(self._o, 'energy').level())
        except Exception:
            return 80.0

    def _get_vitality_pressure(self) -> float:
        try:
            return float(getattr(self._o, 'pressure').pressure("vitality"))
        except Exception:
            return 0.0

    def _get_epistemic_pressure(self) -> float:
        try:
            return float(getattr(self._o, 'pressure').pressure("epistemic"))
        except Exception:
            return 0.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            import dataclasses
            self._path.write_text(
                json.dumps(dataclasses.asdict(self._state), indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"[SleepCycle] Save error: {e}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text(encoding="utf-8"))
            # Bug fix (v59): this used to unconditionally reset
            # last_user_activity to time.time() on every load, treating an
            # app restart as if the user had just interacted. That's fine
            # for a rarely-restarted always-on deployment, but for a dev
            # workflow that restarts the process between every patch (this
            # one), it meant the ACTIVE→IDLE→SLEEP idle clock got wiped
            # back to zero every single restart — so if restarts happen
            # more often than IDLE_ONSET+SLEEP_ONSET (15 min), the system
            # can NEVER reach SLEEP, let alone DREAM, no matter how long
            # the person actually leaves it alone, because the "leaving it
            # alone" clock keeps getting reset by the act of restarting.
            # Restoring the true last-activity timestamp means genuine
            # silence — including silence that spanned a restart — counts
            # the way it should.
            self._state.phase              = d.get("phase", Phase.ACTIVE.value)
            self._state.phase_entered_at   = d.get("phase_entered_at", time.time())
            self._state.cycles_completed   = d.get("cycles_completed", 0)
            self._state.last_sleep_at      = d.get("last_sleep_at")
            self._state.last_dream_at      = d.get("last_dream_at")
            self._state.last_user_activity = d.get("last_user_activity", time.time())
        except Exception as e:
            logger.warning(f"[SleepCycle] Load error: {e}")
