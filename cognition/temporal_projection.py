"""
cognition/temporal_projection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TemporalProjection (v55) — simulates multiple paths toward aspirations
and chooses between them based on predicted resource consequences.

The gap this closes
────────────────────
After v54 (AspirationalSynthesisEngine), Lumina can generate aspirations:
    "Develop affective audio synthesis from music theory + emotion modeling"

But it has no way to ask:
    "What sequence of actions gets me there?"
    "If I spend attention on deep reasoning this week, does that path
     advance this aspiration or deplete the resources I need for it?"
    "Which path has the best predicted trajectory?"

TemporalProjection adds this reasoning layer.

What "temporal projection" means here precisely
─────────────────────────────────────────────────
Not a full simulation of future states. That requires causal models that
don't exist yet (v56 will build them). Instead: a multi-path comparison
using the existing PredictiveConsequenceModel as the cost function.

Given an aspiration with a description and state_desired, generate N
candidate action sequences (paths) toward it, then for each path:
    1. Estimate cumulative resource cost from PredictiveConsequenceModel
    2. Estimate progress rate (how fast does this path reduce the gap?)
    3. Score: progress / cost  (efficiency)
    4. Flag paths that cross critical resource thresholds

The highest-efficiency path that doesn't cross critical thresholds
becomes the "projected path" — a preferred sequence of action types
that the system biases toward.

The projected path then influences:
    - GoalEngine: goals on the projected path get priority boost
    - MotivationalField: the dominant drive is weighted toward the path's
      primary action type
    - BehaviorGate.slow_cycle_scope(): the path's next action type is
      promoted in the scope list (on top of the motivational field's
      existing promotion)

Path generation
────────────────
Paths are not planned by the LLM. The LLM would produce plausible-
sounding plans that don't connect to the real cost model. Instead:

Three fixed path archetypes are generated for every aspiration:
    intensive:   high action_rate, prioritises depth (deep_reasoning,
                 philosophical, analytical) — fast progress, high cost
    distributed: balanced action mix — medium progress, medium cost
    emergent:    low action_rate, lets curiosity guide — slow progress,
                 very low cost, preserves resources for other goals

Each archetype is instantiated with specific action_type sequences
derived from the aspiration's skill components (from v54).

Then PredictiveConsequenceModel estimates the resource cost of each
archetype over PROJECTION_HORIZON interactions, and the efficiency
score selects the preferred path.

This is primitive temporal reasoning, not planning. It asks "which
mode of engaging with this aspiration best fits current resources?"
rather than computing optimal action sequences.

Importantly — it uses real data. The cost estimates come from
PredictiveConsequenceModel's learned records (v53), not hardcoded
assumptions. After 50+ interactions, these estimates become
increasingly accurate as the model learns Lumina's actual costs.

Integration
────────────
    tp = TemporalProjection(organism, ai_system)
    tp.tick(slow_cycle)

    # Read current projection for active aspirations
    proj = tp.active_projections()   # Dict[aspiration_domain → ProjectionResult]

    # The projected path is surfaced to BehaviorGate via:
    tp.preferred_action_type()   # str|None — promotes in scope
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
PROJECTION_EVERY_N   = 50    # slow cycles between projection updates (~100 min)
PROJECTION_HORIZON   = 20    # interactions to simulate per path

# ── Path archetypes ───────────────────────────────────────────────────────────
# Defined as weights over action types from PredictiveConsequenceModel
PATH_ARCHETYPES: Dict[str, Dict[str, float]] = {
    "intensive": {
        "deep_reasoning":  0.40,
        "analytical":      0.30,
        "philosophical":   0.20,
        "creative":        0.10,
    },
    "distributed": {
        "analytical":      0.25,
        "social":          0.20,
        "creative":        0.20,
        "deep_reasoning":  0.20,
        "technical":       0.15,
    },
    "emergent": {
        "social":          0.35,
        "creative":        0.30,
        "analytical":      0.20,
        "philosophical":   0.15,
    },
}

# ── Critical resource thresholds ─────────────────────────────────────────────
CRIT_COGNITIVE = 0.15
CRIT_ATTENTION = 0.20
SAVE_PATH      = "data/persona/temporal_projections.json"
MAX_PROJECTION_LOG = 60


@dataclass
class PathProjection:
    """Projection of one archetype path for one aspiration."""
    archetype:            str     # "intensive"|"distributed"|"emergent"
    cumulative_cost:      float   # total predicted cognitive_energy loss
    progress_rate:        float   # estimated aspiration progress per interaction
    efficiency:           float   # progress_rate / cumulative_cost
    crosses_critical:     bool    # True if path would push resource below threshold
    min_energy_predicted: float   # lowest predicted energy during path
    action_sequence:      List[str]  # dominant action types in order


@dataclass
class ProjectionResult:
    """Full projection for one aspiration."""
    aspiration_domain:   str
    aspiration_desc:     str
    projected_at_cycle:  int
    timestamp:           float
    paths:               List[PathProjection]
    preferred_path:      str     # archetype name of selected path
    preferred_action:    str     # primary action type of preferred path
    rationale:           str     # why this path was selected


class TemporalProjection:
    """
    Projects multiple paths toward active aspirations and selects the
    most efficient path that stays within resource constraints.

    Usage:
        tp = TemporalProjection(organism, ai_system)
        tp.tick(slow_cycle)
    """

    def __init__(
        self,
        organism:  Any,
        ai_system: Any,
        path:      str = SAVE_PATH,
    ) -> None:
        self._organism  = organism
        self._ai        = ai_system
        self._path      = Path(path)
        self._lock      = threading.Lock()
        self._log:      List[Dict] = []
        self._active:   Dict[str, ProjectionResult] = {}
        self._load()
        logger.info(
            f"[TemporalProjection] Initialised — "
            f"{len(self._active)} active projections"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Called every slow cycle from InternalThoughtLoop."""
        if slow_cycle % PROJECTION_EVERY_N != 0 or slow_cycle == 0:
            return
        threading.Thread(
            target=self._run_projections,
            args=(slow_cycle,),
            daemon=True,
            name="tp-project",
        ).start()

    def preferred_action_type(self) -> Optional[str]:
        """
        Primary action type of the highest-priority active projection.
        Used by BehaviorGate to promote this module in scope.
        """
        with self._lock:
            if not self._active:
                return None
            # Pick aspiration with highest state_desired tension
            asp = getattr(self._organism, 'aspirational_self', None)
            if not asp:
                return list(self._active.values())[0].preferred_action if self._active else None
            highest_tension = 0.0
            best_action = None
            for domain, result in self._active.items():
                aspiration = asp.aspirations.get(domain)
                tension    = getattr(aspiration, 'tension', 0.0) if aspiration else 0.0
                if tension > highest_tension:
                    highest_tension = tension
                    best_action = result.preferred_action
            return best_action

    def active_projections(self) -> Dict[str, ProjectionResult]:
        with self._lock:
            return dict(self._active)

    def status(self) -> Dict:
        # Fix: this previously called self.preferred_action_type() while
        # holding self._lock — preferred_action_type() ALSO acquires
        # self._lock, and self._lock is a plain threading.Lock() (not
        # reentrant). This was an unconditional deadlock: any call to
        # status() hung forever once TemporalProjection existed (lazy-inits
        # at slow_cycle > 10). The hang occurred inside _fetch_all(), which
        # runs in asyncio.to_thread() with no timeout — so the dashboard's
        # render() coroutine never completed, never threw, and never logged
        # anything. This is the exact "blank dashboard after cycle 11, no
        # error in logs" symptom. Fixed by computing preferred_action
        # INSIDE the single lock acquisition, not via a second locking call.
        with self._lock:
            active_snapshot = dict(self._active)

        preferred = self._preferred_action_type_locked(active_snapshot)

        return {
            "active_projections": len(active_snapshot),
            "preferred_action":   preferred,
            "domains": [
                {
                    "domain":    d,
                    "preferred": r.preferred_path,
                    "action":    r.preferred_action,
                }
                for d, r in active_snapshot.items()
            ],
        }

    def _preferred_action_type_locked(self, active_snapshot: Dict) -> Optional[str]:
        """Same logic as preferred_action_type() but operates on an already-
        acquired snapshot, never touching self._lock. Used internally by
        status() to avoid the re-entrant deadlock described above."""
        if not active_snapshot:
            return None
        asp = getattr(self._organism, 'aspirational_self', None)
        if not asp:
            return list(active_snapshot.values())[0].preferred_action if active_snapshot else None
        highest_tension = 0.0
        best_action = None
        for domain, result in active_snapshot.items():
            aspiration = asp.aspirations.get(domain)
            tension    = getattr(aspiration, 'tension', 0.0) if aspiration else 0.0
            if tension > highest_tension:
                highest_tension = tension
                best_action = result.preferred_action
        if best_action is None:
            # Active projections can outlive or use different aspiration keys.
            best_action = next(iter(active_snapshot.values())).preferred_action
        return best_action


    def prompt_fragment(self) -> str:
        """Short projection status for prompt injection."""
        action = self.preferred_action_type()
        if not action:
            return ""
        with self._lock:
            active = list(self._active.values())
        if not active:
            return ""
        top = active[0]
        return (
            f"[Temporal projection] Pursuing '{top.aspiration_desc[:60]}' "
            f"via {top.preferred_path} path ({action}-oriented)."
        )

    # ── Projection pipeline ───────────────────────────────────────────────────

    def _run_projections(self, slow_cycle: int) -> None:
        try:
            asp = getattr(self._organism, 'aspirational_self', None)
            if not asp or not asp.aspirations:
                return

            # Project top 3 aspirations by tension
            top_aspirations = sorted(
                asp.aspirations.values(),
                key=lambda a: getattr(a, 'tension', 0.0),
                reverse=True,
            )[:3]

            new_active: Dict[str, ProjectionResult] = {}

            for aspiration in top_aspirations:
                result = self._project_aspiration(aspiration, slow_cycle)
                if result:
                    new_active[aspiration.domain] = result
                    self._log.append(asdict(result))

            with self._lock:
                self._active = new_active
                if len(self._log) > MAX_PROJECTION_LOG:
                    self._log = self._log[-MAX_PROJECTION_LOG:]

            self._save()

            if new_active:
                preferred_action = self.preferred_action_type()
                logger.info(
                    f"[TemporalProjection] {len(new_active)} projections updated. "
                    f"Preferred action: {preferred_action}"
                )

        except Exception as e:
            logger.debug(f"[TemporalProjection] _run_projections error: {e}")

    def _project_aspiration(
        self, aspiration: Any, slow_cycle: int
    ) -> Optional[ProjectionResult]:
        """Project all three path archetypes for one aspiration."""
        try:
            # Get PredictiveConsequenceModel for cost estimates
            loop = getattr(self._organism, '_loop', None)
            pcm  = getattr(loop, '_consequence_model', None) if loop else None

            # Read current resource state
            pcm_state = pcm.read_state() if pcm else [0.75, 0.75, 0.75, 0.65, 0.45]
            cog_now   = pcm_state[0]

            # Infer relevant action types from aspiration domain
            domain_action = self._infer_action_type(
                getattr(aspiration, 'description', '')
                + " " + getattr(aspiration, 'domain', '')
            )

            paths = []
            for archetype_name, action_weights in PATH_ARCHETYPES.items():
                path = self._simulate_path(
                    archetype_name, action_weights, domain_action,
                    pcm, pcm_state, aspiration
                )
                paths.append(path)

            # Select preferred: best efficiency that doesn't cross critical
            viable = [p for p in paths if not p.crosses_critical]
            if not viable:
                # All paths cross critical — pick least bad
                viable = sorted(paths, key=lambda p: p.min_energy_predicted, reverse=True)

            preferred = max(viable, key=lambda p: p.efficiency)

            rationale = self._build_rationale(preferred, paths, cog_now)

            return ProjectionResult(
                aspiration_domain  = getattr(aspiration, 'domain', 'unknown'),
                aspiration_desc    = getattr(aspiration, 'description', '')[:80],
                projected_at_cycle = slow_cycle,
                timestamp          = time.time(),
                paths              = paths,
                preferred_path     = preferred.archetype,
                preferred_action   = preferred.action_sequence[0] if preferred.action_sequence else domain_action,
                rationale          = rationale,
            )

        except Exception as e:
            logger.debug(f"[TemporalProjection] _project_aspiration error: {e}")
            return None

    def _simulate_path(
        self,
        archetype:      str,
        action_weights: Dict[str, float],
        domain_action:  str,
        pcm:            Any,
        initial_state:  List[float],
        aspiration:     Any,
    ) -> PathProjection:
        """
        Simulate PROJECTION_HORIZON interactions using this path archetype.
        Accumulates resource cost from PredictiveConsequenceModel predictions.
        Returns PathProjection with cumulative cost, progress, and efficiency.
        """
        state       = list(initial_state)
        total_cost  = 0.0
        total_progress = 0.0
        min_energy  = state[0]
        crosses     = False
        action_seq  = []

        for step in range(PROJECTION_HORIZON):
            # Pick action type for this step (weighted by archetype)
            action = self._sample_action(action_weights, domain_action, step)
            action_seq.append(action)

            # Get predicted delta from PCM
            if pcm and len(pcm._records) >= 4:
                result = pcm.predict(action, state)
                if result:
                    energy_delta = result.energy_delta
                    att_delta    = result.attention_delta
                else:
                    # Default costs when PCM has insufficient data
                    energy_delta = -0.04
                    att_delta    = -0.02
            else:
                # Default action costs (before PCM has data)
                energy_delta = {
                    "deep_reasoning": -0.08,
                    "philosophical":  -0.07,
                    "analytical":     -0.05,
                    "technical":      -0.06,
                    "creative":       -0.04,
                    "social":         -0.02,
                    "emotional":      -0.03,
                }.get(action, -0.04)
                att_delta = energy_delta * 0.5

            # Apply to state
            state[0] = max(0.0, state[0] + energy_delta)
            state[2] = max(0.0, state[2] + att_delta)
            # Recovery: small natural recovery each step
            state[0] = min(1.0, state[0] + 0.008)
            state[2] = min(1.0, state[2] + 0.004)

            total_cost    += abs(energy_delta)
            min_energy     = min(min_energy, state[0])

            # Progress estimate: domain-aligned actions contribute more
            alignment = 1.0 if action == domain_action else 0.4
            total_progress += alignment * 0.05   # 5% per aligned step max

            if state[0] < CRIT_COGNITIVE or state[2] < CRIT_ATTENTION:
                crosses = True

        efficiency = (total_progress / total_cost) if total_cost > 0 else 0.0

        return PathProjection(
            archetype            = archetype,
            cumulative_cost      = round(total_cost,    3),
            progress_rate        = round(total_progress / PROJECTION_HORIZON, 4),
            efficiency           = round(efficiency,    3),
            crosses_critical     = crosses,
            min_energy_predicted = round(min_energy,    3),
            action_sequence      = list(dict.fromkeys(action_seq))[:4],  # unique, ordered
        )

    def _sample_action(
        self,
        weights:       Dict[str, float],
        domain_action: str,
        step:          int,
    ) -> str:
        """Sample action type from archetype weights, favouring domain on even steps."""
        if step % 3 == 0 and domain_action in weights:
            return domain_action
        sorted_actions = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        # Deterministic selection by step index for reproducibility
        idx = step % len(sorted_actions)
        return sorted_actions[idx][0]

    def _infer_action_type(self, text: str) -> str:
        """Classify aspiration description into primary action type."""
        from cognition.predictive_consequence_model import ACTION_KEYWORDS
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for action_type, keywords in ACTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits:
                scores[action_type] = hits
        if not scores:
            return "analytical"
        return max(scores, key=scores.get)

    def _build_rationale(
        self,
        preferred: PathProjection,
        all_paths: List[PathProjection],
        cog_now:   float,
    ) -> str:
        others = [p for p in all_paths if p.archetype != preferred.archetype]
        other_strs = [
            f"{p.archetype}(eff={p.efficiency:.2f}"
            + (", critical" if p.crosses_critical else "")
            + ")"
            for p in others
        ]
        return (
            f"Selected '{preferred.archetype}' path "
            f"(efficiency={preferred.efficiency:.2f}, "
            f"min_energy={preferred.min_energy_predicted:.2f}, "
            f"crosses_critical={preferred.crosses_critical}). "
            f"Current cog_energy={cog_now:.2f}. "
            f"Alternatives: {', '.join(other_strs)}."
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "active": {
                        d: asdict(r) for d, r in self._active.items()
                    },
                    "log":   self._log[-MAX_PROJECTION_LOG:],
                    "_meta": {"version": "v55", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[TemporalProjection] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            self._log = data.get("log", [])
            for domain, raw in data.get("active", {}).items():
                try:
                    fields = dict(raw)
                    fields['paths'] = [PathProjection(**p) for p in fields['paths']]
                    self._active[domain] = ProjectionResult(**fields)
                except (TypeError, KeyError, ValueError) as e:
                    logger.warning(f"[TemporalProjection] Invalid saved projection {domain}: {e}")
        except Exception as e:
            logger.warning(f"[TemporalProjection] Load failed: {e}")
