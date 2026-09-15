"""
cognition/cognitive_behavior_gate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CognitiveBehaviorGate — enforcement layer with soft degradation and
priority arbitration.

Two corrections from initial v50 implementation
─────────────────────────────────────────────────

A. Soft degradation instead of hard binary gates.
   Binary gates can create coma states:
       energy=0.29 → no curiosity, no experiment, no outreach
       nothing fires → less interaction → less recovery → stuck
   Humans rarely switch fully off.  Depleted systems become LESS LIKELY
   to act, not IMPOSSIBLE to act.  Each may_*() method now returns a
   probabilistic decision: at full resources p≈1.0, at zero p≈0.05
   (a small floor ensures the system never fully seizes).

B. Priority arbitration.
   Without a hierarchy, module A says yes while module B says no and
   the outcome is ambiguous.  Explicit priority order:
       1. Safety / immune (always wins — counterfactual mode forces reflection)
       2. Audit trial (blocks experiments — two calibration loops can't stack)
       3. Resource economy (depleted energy suppresses low-value work)
       4. Reflection (always permitted, depth varies)
       5. Resolution / curiosity / outreach / meta-systems (resource-gated)
       6. Experiment (lowest priority — highest cost)

   arbitrate(process) returns the canonical decision after consulting
   all relevant factors in order.  Each may_*() calls arbitrate().

Decision model
───────────────
Gate reads:
    1. CognitiveResourceEconomy — energy/attention pools
    2. SelfModel.cognitive_load — interaction load EMA
    3. Immune / audit state     — override conditions

Exposes:
    may_generate_curiosity()   → bool  (probabilistic)
    may_start_experiment()     → bool  (probabilistic, lowest priority)
    may_run_reflection()       → (bool, depth_str)
    may_run_resolution()       → bool
    may_initiate_outreach()    → bool
    may_run_meta_system(name)  → bool
    max_response_tokens()      → int   (hard cap on get_response)
    slow_cycle_scope()         → List[str]

Gate never raises — all methods return permissive defaults on error.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cognition.cognitive_resource_economy import CognitiveResourceEconomy

logger = logging.getLogger(__name__)

# ── Token caps ────────────────────────────────────────────────────────────────
TOKENS_NORMAL     = 800
TOKENS_LOADED     = 500
TOKENS_LOW_ENERGY = 300
TOKENS_CRITICAL   = 180

# ── Probability floor — depleted systems still occasionally act ───────────────
P_FLOOR = 0.05   # never go below 5% probability, even at zero resources

# ── Priority levels (lower number = higher priority) ─────────────────────────
PRIORITY = {
    "immune":      1,
    "reflection":  2,
    "audit":       3,
    "resolution":  4,
    "curiosity":   5,
    "outreach":    6,
    "meta":        6,
    "experiment":  7,   # most expensive, lowest priority
}


class CognitiveBehaviorGate:
    """
    Single point of truth for all behavioral go/no-go decisions.
    Uses probabilistic soft degradation and explicit priority arbitration.

    Usage:
        gate = organism.behavior_gate
        if not gate.may_generate_curiosity():
            return

        max_tok = gate.max_response_tokens()
        response = ai.get_response(..., max_tokens=max_tok)

    Non-fatal: returns permissive defaults on any read error.
    """

    def __init__(self, organism: Any) -> None:
        self._organism  = organism
        self._last_log: dict = {}

    # ── Public decisions ──────────────────────────────────────────────────────

    def may_generate_curiosity(self) -> bool:
        """
        Probabilistic — depleted attention/energy reduces probability,
        not a hard block.  Ensures the system never fully stops noticing
        things even when overloaded.
        """
        return self._arbitrate("curiosity")

    def may_start_experiment(self) -> bool:
        """
        Experiments are the most expensive cognitive operation.
        Lowest priority — blocked by immune counterfactual, audit trial,
        AND requires higher resource threshold than other processes.
        """
        return self._arbitrate("experiment")

    def may_run_reflection(self) -> Tuple[bool, str]:
        """
        Reflection always runs (priority 2, just below immune).
        Returns (True, depth) where depth varies with resource state:
            'deep'             — full LLM synthesis
            'surface'          — shorter prompt, half token budget
            'consolidate_only' — no LLM call, memory consolidation only
        """
        # Reflection always permitted — depth degrades gracefully
        ec   = self._economy()
        sm   = self._self_model()
        cog  = ec.cognitive_energy if ec else 1.0
        load = getattr(sm, 'cognitive_load', 0.5) if sm else 0.5

        if cog < 0.12 or load > 0.90:
            depth = "consolidate_only"
        elif cog < 0.28 or load > 0.80:
            depth = "surface"
        else:
            depth = "deep"

        return True, depth

    def may_run_resolution(self) -> bool:
        return self._arbitrate("resolution")

    def may_initiate_outreach(self) -> bool:
        return self._arbitrate("outreach")

    def may_run_meta_system(self, name: str = "meta") -> bool:
        return self._arbitrate("meta")

    def max_response_tokens(self) -> int:
        """
        Hard cap passed directly to get_response() as max_tokens.
        This is a constraint on the system, not a suggestion to the LLM.
        """
        sm   = self._self_model()
        ec   = self._economy()
        load = getattr(sm, 'cognitive_load', 0.5) if sm else 0.5
        cog  = ec.cognitive_energy if ec else 1.0

        if cog < 0.12 or load > 0.90:
            cap = TOKENS_CRITICAL
        elif cog < 0.28 or load > 0.80:
            cap = TOKENS_LOW_ENERGY
        elif cog < 0.50 or load > 0.65:
            cap = TOKENS_LOADED
        else:
            cap = TOKENS_NORMAL

        if cap < TOKENS_NORMAL:
            self._log_once("tokens",
                f"max_tokens={cap} (cog={cog:.2f} load={load:.2f})")
        return cap

    def slow_cycle_scope(self) -> List[str]:
        """
        Returns module names permitted this slow cycle.
        Modules check their name before firing.
        v51: MotivationalField dominant drive moves its corresponding
        module to the front of the scope list — it runs first and
        with full priority regardless of normal ordering.
        """
        ec   = self._economy()
        sm   = self._self_model()
        cog  = ec.cognitive_energy if ec else 1.0
        load = getattr(sm, 'cognitive_load', 0.5) if sm else 0.5

        if cog < 0.12 or load > 0.90:
            scope = ["consolidation"]
        elif cog < 0.28 or load > 0.80:
            scope = ["reflection", "consolidation"]
        elif cog < 0.50:
            scope = ["reflection", "resolution", "consolidation"]
        else:
            scope = [
                "reflection", "curiosity", "resolution",
                "experiment", "dream", "outreach",
                "audit", "immune",
            ]

        # v51: MotivationalField — promote dominant drive module to front
        # Drive → module mapping
        drive_module_map = {
            "epistemic":  "resolution",
            "novelty":    "curiosity",
            "social":     "outreach",
            "expression": "outreach",
            "coherence":  "reflection",
            "purpose":    "experiment",
        }
        try:
            loop = getattr(self._organism, '_loop', None)
            mf   = getattr(loop, '_motivational_field', None) if loop else None
            if mf:
                dominant = mf.dominant_drive()
                priority_module = drive_module_map.get(dominant)
                if priority_module and priority_module in scope:
                    scope = [priority_module] + [s for s in scope if s != priority_module]
                    self._log_once(
                        "mf_priority",
                        f"MotivationalField promoted '{priority_module}' "
                        f"(dominant drive: {dominant}={mf.drive_vector.get(dominant, 0):.2f})"
                    )
        except Exception:
            pass

        # v55: TemporalProjection — promote preferred action type module
        try:
            tp = getattr(loop, '_temporal_projection', None) if loop else None
            if tp:
                preferred_action = tp.preferred_action_type()
                action_to_module = {
                    "deep_reasoning":  "reflection",
                    "philosophical":   "reflection",
                    "analytical":      "resolution",
                    "creative":        "curiosity",
                    "technical":       "experiment",
                    "social":          "outreach",
                    "emotional":       "reflection",
                }
                tp_module = action_to_module.get(preferred_action)
                if tp_module and tp_module in scope:
                    scope = [tp_module] + [s for s in scope if s != tp_module]
                    self._log_once(
                        "tp_priority",
                        f"TemporalProjection promoted '{tp_module}' "
                        f"(preferred action: {preferred_action})"
                    )
        except Exception:
            pass
            self._log_once("scope", f"scope={scope} (cog={cog:.2f} load={load:.2f})")
        return scope

    def status(self) -> dict:
        ec = self._economy()
        return {
            "max_response_tokens": self.max_response_tokens(),
            "scope":               self.slow_cycle_scope(),
            "probabilities": {
                "curiosity":   round(self._compute_p("curiosity"),   3),
                "experiment":  round(self._compute_p("experiment"),  3),
                "resolution":  round(self._compute_p("resolution"),  3),
                "outreach":    round(self._compute_p("outreach"),    3),
            },
            "resources": ec.status() if ec else {},
        }

    # ── Arbitration engine ────────────────────────────────────────────────────

    def _arbitrate(self, process: str) -> bool:
        """
        Priority arbitration + soft probabilistic gate.

        Step 1 — Priority overrides (deterministic):
            Immune counterfactual → forces reflection, blocks experiment
            Audit trial active    → blocks experiment
            Safety flags          → block outreach

        Step 2 — Compute probability from resource state.

        Step 3 — Sample: return random() < p.
            At p=1.0 → always fires
            At p=0.3 → fires 30% of cycles
            At p=0.05 → rarely fires but never fully stops
        """
        # ── Step 1: deterministic overrides ───────────────────────────────────

        # Immune counterfactual mode
        loop = getattr(self._organism, '_loop', None)
        cis  = getattr(loop, '_cognitive_immune', None) if loop else None
        immune_active = cis and cis.is_counterfactual_active() if cis else False

        if immune_active:
            if process == "experiment":
                # Experiments during counterfactual would contaminate the
                # disconfirmation signal — block them deterministically
                return self._deny(process, "immune_counterfactual_active", p=0.0)
            if process == "curiosity":
                # Immune already injected counterfactual topics — don't
                # add more standard curiosity on top
                return self._deny(process, "immune_curiosity_suppressed", p=0.0)

        # Audit trial active → block experiment (can't calibrate and experiment)
        cae = getattr(loop, '_cognitive_audit', None) if loop else None
        if (process == "experiment" and cae
                and getattr(getattr(cae, '_state', None), 'active_trial', None)):
            return self._deny(process, "audit_trial_active", p=0.0)

        # ── Step 2: compute probability ───────────────────────────────────────
        p = self._compute_p(process)

        # ── Step 3: sample ────────────────────────────────────────────────────
        if random.random() < p:
            return True
        return self._deny(process, f"p={p:.2f}_suppressed")

    def _compute_p(self, process: str) -> float:
        """
        Compute probability [P_FLOOR, 1.0] for a process based on
        resource state.  Each process has a different resource sensitivity.

        Formula:
            p = weighted_score, clamped to [P_FLOOR, 1.0]

        Weights per process reflect cost and replaceability:
            experiment  — high cognitive_energy weight (most expensive)
            curiosity   — high attention weight (accumulates topics)
            outreach    — high social_energy weight (interpersonal cost)
            resolution  — moderate cognitive weight (LLM call required)
            meta        — moderate cognitive weight (background call)
        """
        ec   = self._economy()
        sm   = self._self_model()
        cog  = ec.cognitive_energy if ec else 1.0
        soc  = ec.social_energy    if ec else 1.0
        att  = ec.attention        if ec else 1.0
        load = getattr(sm, 'cognitive_load', 0.5) if sm else 0.5
        load_factor = max(0.0, 1.0 - load)

        weights = {
            "experiment":  cog * 0.50 + att * 0.30 + load_factor * 0.20,
            "curiosity":   att * 0.40 + cog * 0.40 + load_factor * 0.20,
            "outreach":    soc * 0.60 + cog * 0.25 + load_factor * 0.15,
            "resolution":  cog * 0.55 + att * 0.25 + load_factor * 0.20,
            "meta":        cog * 0.50 + att * 0.30 + load_factor * 0.20,
            "reflection":  1.0,   # always 1.0 — depth handled separately
            "immune":      1.0,
            "audit":       cog * 0.40 + att * 0.40 + load_factor * 0.20,
        }
        raw = weights.get(process, cog * 0.5 + att * 0.3 + load_factor * 0.2)
        # v57: CrossLayerFeedback writes _confidence_floor based on PCM coverage.
        # Novel territory (few prediction records) → higher floor → more cautious.
        floor = getattr(self, '_confidence_floor', P_FLOOR)
        # Priority arbitration (bounded, monotonic): higher-priority processes
        # (lower PRIORITY number) get a small floor bump so they stay more
        # resilient under resource pressure — this only ever raises the
        # floor, never suppresses, so it can't fight the resource weights.
        priority = PRIORITY.get(process, 5)
        floor += max(0, 5 - priority) * 0.01
        return max(floor, min(1.0, raw))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _economy(self) -> Optional["CognitiveResourceEconomy"]:
        try:
            loop = getattr(self._organism, '_loop', None)
            return getattr(loop, '_resource_economy', None) if loop else None
        except Exception:
            return None

    def _self_model(self) -> Any:
        try:
            return getattr(self._organism, 'self_model', None)
        except Exception:
            return None

    def _deny(self, process: str, reason: str, p: float = -1.0) -> bool:
        key = f"{process}:{reason}"
        now = time.time()
        if now - self._last_log.get(key, 0) > 120:
            if p == 0.0:
                logger.info(f"[BehaviorGate] ⛔ {process} blocked — {reason}")
            else:
                logger.debug(f"[BehaviorGate] ↓ {process} suppressed — {reason}")
            self._last_log[key] = now
        return False

    def _log_once(self, key: str, msg: str) -> None:
        now = time.time()
        if now - self._last_log.get(key, 0) > 120:
            logger.debug(f"[BehaviorGate] {msg}")
            self._last_log[key] = now
