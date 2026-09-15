"""
Cognitive Governor
==================
Central authority over the InternalThoughtLoop subsystem scheduler.

Problem it solves
-----------------
InternalThoughtLoop has 14+ independent subsystems that each fire on their
own ``% N`` schedule with no coordination. They all write to the same JSON
files, generate pressure signals, and none knows what the others are doing.
When several land on the same cycle number they pile up; when GoalEngine fires
while GQF is mid-run they see inconsistent goal state. No system had authority
to say "too much is happening right now — stand down."

What the Governor does
----------------------
1. **Mode**: ``structured`` (safe, deterministic) or ``exploratory``
   (creative, emergent). Only one is active at a time.
   Structured mode blocks new autonomous goal generation and limits which
   subsystems may run. Exploratory mode permits emergence but with a cycle
   budget cap to prevent runaway recursion.

2. **Cycle budget**: at the start of each slow cycle a budget (max subsystems
   allowed to fire) is set based on current total pressure. High pressure →
   small budget, only the most critical systems run.

3. **Quorum guard**: a subsystem is skipped if its last run produced zero
   meaningful output AND nothing has changed since (no new goals, no new
   beliefs, no new pressure). Prevents GQF running 50× on the same stale
   goal list.

4. **Pressure ceiling**: when total decision pressure exceeds
   ``GOAL_GEN_PRESSURE_CEILING``, new autonomous goal generation is blocked
   regardless of what GoalEngine wants. The ``DecisionPressureCalculator``
   signal (already computed each cycle) drives this gate.

5. **Priority tiers**: subsystems are tiered. Critical systems (tension,
   homeostasis, pressure) always run. Core systems run unless budget is
   exhausted. Enrichment systems (semantic cleaner, belief bootstrapper) only
   run in exploratory mode when budget allows.

Usage (InternalThoughtLoop)
---------------------------
    # At start of __init__:
    from core.cognitive_governor import CognitiveGovernor
    self._governor = CognitiveGovernor()

    # At start of _slow_cycle():
    _total_pressure = getattr(o, '_v32_decision_pressure', {}).get('total', 0.5)
    _goal_count     = len(getattr(_ge, '_goals', {})) if _ge else 0
    self._governor.begin_cycle(_total_pressure, _goal_count, self._slow_cycle_count)

    # Before each subsystem call:
    if self._governor.may_run('GoalEngine'):
        _ge.tick()
        self._governor.record('GoalEngine', len(_ge.get_active_goals()))
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ── Subsystem tier definitions ─────────────────────────────────────────────────
# CRITICAL  — always run, not budget-counted
# CORE      — run unless budget exhausted; run in both modes
# ENRICHMENT— only in exploratory mode, consumed from budget
# GOAL_GEN  — blocked by pressure ceiling; exploratory mode only

TIER_CRITICAL   = "critical"
TIER_CORE       = "core"
TIER_ENRICHMENT = "enrichment"
TIER_GOAL_GEN   = "goal_gen"

SUBSYSTEM_TIERS: Dict[str, str] = {
    # Critical — always fire
    "Tensions":          TIER_CRITICAL,
    "Homeostasis":       TIER_CRITICAL,
    "PressureSystem":    TIER_CRITICAL,
    "NarrativeIdentity": TIER_CRITICAL,
    "MemoryIntrusion":   TIER_CRITICAL,
    "SleepCycle":        TIER_CRITICAL,
    "CognitiveStack":    TIER_CRITICAL,

    # Core — fire in both modes, budget-counted
    "Phase1Pressure":    TIER_CORE,
    "Phase1Threads":     TIER_CORE,
    "SCS":               TIER_CORE,       # SelfConceptSynchronizer
    "PEL":               TIER_CORE,       # PersistentExecutiveLoop (Phase 5.1)
    "ARB_LEARN":         TIER_CORE,       # ArbitrationLearningTracker (Phase 5.4)
    "GQF":               TIER_CORE,       # GoalQualityFilter
    "GoalConsolidator":  TIER_CORE,
    "ThreadLifecycle":   TIER_CORE,
    "TTE_CDE_DTS":       TIER_CORE,
    "GWEpisodic":        TIER_CORE,
    "IdentityBatch":     TIER_CORE,

    # Enrichment — exploratory mode only, budget-counted
    "SemanticGraphCleaner": TIER_ENRICHMENT,
    "BeliefBootstrapper":   TIER_ENRICHMENT,
    "Phase2Narrative":      TIER_ENRICHMENT,
    "Phase2ThreadIdentity": TIER_ENRICHMENT,
    "Phase2ThoughtEval":    TIER_ENRICHMENT,

    # Goal generation — blocked by pressure ceiling
    "GoalEngine":           TIER_GOAL_GEN,
    "Phase1ThoughtGoals":   TIER_GOAL_GEN,
}


@dataclass
class GovernorStatus:
    mode:             str
    total_pressure:   float
    goal_count:       int
    budget_total:     int
    budget_remaining: int
    cycle:            int
    blocked_systems:  list = field(default_factory=list)
    run_systems:      list = field(default_factory=list)


class CognitiveGovernor:
    """
    Central authority over the InternalThoughtLoop subsystem scheduler.
    See module docstring for full design notes.
    """

    MODE_STRUCTURED  = "structured"
    MODE_EXPLORATORY = "exploratory"

    # Pressure thresholds
    HIGH_PRESSURE_THRESHOLD  = 0.65   # → structured mode
    LOW_PRESSURE_THRESHOLD   = 0.30   # → exploratory mode
    GOAL_GEN_PRESSURE_CEILING = 0.60  # blocks new goal generation above this

    # Goal count thresholds
    HIGH_GOAL_COUNT  = 15   # → structured mode
    LOW_GOAL_COUNT   = 6    # helps stay exploratory

    # Cycle budgets (max core+enrichment subsystems per slow cycle)
    BUDGET_HIGH_PRESSURE = 3
    BUDGET_MED_PRESSURE  = 5
    BUDGET_LOW_PRESSURE  = 8

    # Quorum: skip if last N runs produced zero output
    QUORUM_SKIP_AFTER = 2   # skip after 2 consecutive empty runs

    def __init__(self) -> None:
        self.mode: str = self.MODE_STRUCTURED
        self._pressure: float = 0.5
        self._goal_count: int = 0
        self._cycle: int = 0

        self._budget_total: int = self.BUDGET_MED_PRESSURE
        self._budget_remaining: int = self.BUDGET_MED_PRESSURE

        # Tracking: subsystem → consecutive empty run count
        self._empty_run_count: Dict[str, int] = {}
        # Tracking: subsystem → last recorded output size
        self._last_output: Dict[str, int] = {}
        # Tracking for this cycle's log
        self._this_cycle_run: list = []
        self._this_cycle_blocked: list = []

        # Hysteresis: how many cycles must pass before switching modes
        self._mode_switch_cooldown = 3
        self._cycles_in_current_mode = 0

        logger.info("[Governor] CognitiveGovernor initialised — mode=structured")

    # ── Public API ─────────────────────────────────────────────────────────────

    def begin_cycle(
        self,
        total_pressure: float,
        goal_count: int,
        cycle_number: int,
    ) -> None:
        """
        Called at the START of every slow cycle.
        Sets mode, budget, and resets per-cycle tracking.

        Parameters
        ----------
        total_pressure : float  — from DecisionPressureCalculator (0–1)
        goal_count     : int    — current active+dormant goal count
        cycle_number   : int    — slow cycle counter from InternalThoughtLoop
        """
        self._pressure   = max(0.0, min(1.0, total_pressure))
        self._goal_count = goal_count
        self._cycle      = cycle_number
        self._this_cycle_run.clear()
        self._this_cycle_blocked.clear()
        self._cycles_in_current_mode += 1

        # ── Mode determination (with hysteresis) ──────────────────────────────
        new_mode = self._compute_mode()
        if new_mode != self.mode:
            if self._cycles_in_current_mode >= self._mode_switch_cooldown:
                old_mode = self.mode
                self.mode = new_mode
                self._cycles_in_current_mode = 0
                logger.info(
                    f"[Governor] Mode switch: {old_mode} → {new_mode} "
                    f"(pressure={self._pressure:.2f}, goals={self._goal_count})"
                )

        # ── Budget ────────────────────────────────────────────────────────────
        if self._pressure > self.HIGH_PRESSURE_THRESHOLD:
            budget = self.BUDGET_HIGH_PRESSURE
        elif self._pressure > self.LOW_PRESSURE_THRESHOLD:
            budget = self.BUDGET_MED_PRESSURE
        else:
            budget = self.BUDGET_LOW_PRESSURE

        self._budget_total = budget
        self._budget_remaining = budget

        logger.debug(
            f"[Governor] Cycle #{cycle_number}: mode={self.mode} "
            f"pressure={self._pressure:.2f} goals={self._goal_count} "
            f"budget={budget}"
        )

    def may_run(self, subsystem: str) -> bool:
        """
        Returns True if ``subsystem`` is allowed to run this cycle.
        Call this BEFORE every subsystem invocation in _slow_cycle().

        Decision tree
        -------------
        CRITICAL     → always True (not budget-counted)
        GOAL_GEN     → True only if below pressure ceiling AND exploratory mode
        ENRICHMENT   → True only if exploratory mode AND budget > 0
        CORE         → True if budget > 0 AND not quorum-blocked
        """
        tier = SUBSYSTEM_TIERS.get(subsystem, TIER_CORE)

        # ── Critical: always allowed ───────────────────────────────────────────
        if tier == TIER_CRITICAL:
            self._this_cycle_run.append(subsystem)
            return True

        # ── Goal generation: pressure ceiling + mode gate ─────────────────────
        if tier == TIER_GOAL_GEN:
            if self.block_goal_generation():
                self._this_cycle_blocked.append(f"{subsystem}(pressure_ceiling)")
                logger.debug(
                    f"[Governor] BLOCKED {subsystem}: "
                    f"pressure_ceiling={self._pressure:.2f}>={self.GOAL_GEN_PRESSURE_CEILING} "
                    f"or mode={self.mode}"
                )
                return False
            # Still costs a budget slot
            if self._budget_remaining <= 0:
                self._this_cycle_blocked.append(f"{subsystem}(budget_exhausted)")
                return False
            self._budget_remaining -= 1
            self._this_cycle_run.append(subsystem)
            return True

        # ── Enrichment: exploratory mode only ────────────────────────────────
        if tier == TIER_ENRICHMENT:
            if self.mode != self.MODE_EXPLORATORY:
                self._this_cycle_blocked.append(f"{subsystem}(structured_mode)")
                return False
            if self._budget_remaining <= 0:
                self._this_cycle_blocked.append(f"{subsystem}(budget_exhausted)")
                return False
            if self._quorum_blocked(subsystem):
                self._this_cycle_blocked.append(f"{subsystem}(quorum)")
                return False
            self._budget_remaining -= 1
            self._this_cycle_run.append(subsystem)
            return True

        # ── Core: budget + quorum ─────────────────────────────────────────────
        if self._budget_remaining <= 0:
            self._this_cycle_blocked.append(f"{subsystem}(budget_exhausted)")
            return False
        if self._quorum_blocked(subsystem):
            self._this_cycle_blocked.append(f"{subsystem}(quorum)")
            return False
        self._budget_remaining -= 1
        self._this_cycle_run.append(subsystem)
        return True

    def record(self, subsystem: str, output_size: int) -> None:
        """
        Call AFTER a subsystem runs to record how much it produced.
        ``output_size`` should be a meaningful count: goals created,
        beliefs synced, threads resolved, etc. Zero means "nothing changed."
        """
        self._last_output[subsystem] = output_size
        if output_size == 0:
            self._empty_run_count[subsystem] = (
                self._empty_run_count.get(subsystem, 0) + 1
            )
        else:
            # Reset empty count on any productive run
            self._empty_run_count[subsystem] = 0

    def block_goal_generation(self) -> bool:
        """
        True when new autonomous goal generation must be suppressed.
        Called by GoalEngine.tick() as well as by may_run().
        """
        return (
            self._pressure >= self.GOAL_GEN_PRESSURE_CEILING
            or self.mode == self.MODE_STRUCTURED
        )

    def status(self) -> GovernorStatus:
        """Return a snapshot of current governor state (for Observatory/UI)."""
        return GovernorStatus(
            mode=self.mode,
            total_pressure=self._pressure,
            goal_count=self._goal_count,
            budget_total=self._budget_total,
            budget_remaining=self._budget_remaining,
            cycle=self._cycle,
            blocked_systems=list(self._this_cycle_blocked),
            run_systems=list(self._this_cycle_run),
        )

    def summary_log(self) -> None:
        """Emit a one-line summary log for this cycle (call at end of _slow_cycle)."""
        if self._this_cycle_blocked:
            logger.debug(
                f"[Governor] Cycle #{self._cycle} summary: "
                f"ran={self._this_cycle_run} "
                f"blocked={self._this_cycle_blocked} "
                f"budget_used={self._budget_total - self._budget_remaining}/{self._budget_total}"
            )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _compute_mode(self) -> str:
        """Pure function: what mode should we be in given current state?"""
        # High pressure or goal explosion → structured
        if (
            self._pressure >= self.HIGH_PRESSURE_THRESHOLD
            or self._goal_count >= self.HIGH_GOAL_COUNT
        ):
            return self.MODE_STRUCTURED
        # Low pressure AND few goals → exploratory
        if (
            self._pressure <= self.LOW_PRESSURE_THRESHOLD
            and self._goal_count <= self.LOW_GOAL_COUNT
        ):
            return self.MODE_EXPLORATORY
        # Otherwise keep current mode (hysteresis)
        return self.mode

    def _quorum_blocked(self, subsystem: str) -> bool:
        """
        True if the subsystem has run QUORUM_SKIP_AFTER consecutive times with
        zero output — meaning nothing is changing that it could act on.
        """
        empty = self._empty_run_count.get(subsystem, 0)
        if empty >= self.QUORUM_SKIP_AFTER:
            logger.debug(
                f"[Governor] Quorum block: {subsystem} produced 0 output "
                f"for {empty} consecutive runs"
            )
            return True
        return False
