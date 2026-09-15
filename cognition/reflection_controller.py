"""
ReflectionController — Phase C (scoped)
==========================================
One change from the original spec, flagged in review before implementing:
the original signature was `allowed_depth(meta_error, self_model_div=0.0)`,
but self_model_div only exists as a metric computed inside the v104
replay harness (test_internal_state_downstream.py) — there is no live,
production equivalent signal anywhere in the real codebase. Wiring a
harness-only metric into a live depth controller would either silently
always be 0.0 (dead parameter) or require inventing a new live self-model
divergence computation that doesn't exist yet and wasn't specified.

Scoped to what's real and live: meta_error alone, sourced directly from
InternalCognitiveState.history[-1]['meta_error'] — the same real,
already-verified (v102/v103) quantity. self_model_div is left out
explicitly rather than faked.

Wired into recursive_deliberation.py::deliberate()'s max_cycles: when the
caller relies on the static default (MAX_CYCLES=3), the actual depth
used is now controller.allowed_depth(meta_error) instead — under
sustained high meta-error, deliberation goes deeper (up to hard_cap);
otherwise it stays at the original 3. A caller that explicitly passes
its own max_cycles is respected as an override — "make bounds adaptive,
not replace them" per the design principle.

AutonomousReflection interval/mode-selection wiring (also mentioned in
the original spec) is NOT done here — tick()'s internal per-mode cadence
logic wasn't traced in this pass, and wiring into it without doing so
would risk changing behavior in code this session hasn't verified.
allowed_depth() is exposed as a reusable public function so that wiring
can be added later once traced, rather than guessed at now.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ReflectionController:
    def __init__(self, base_max_depth: int = 3, expand_threshold: float = 0.40,
                 hard_cap: int = 6) -> None:
        self.base_max_depth = base_max_depth
        self.expand_threshold = expand_threshold
        self.hard_cap = hard_cap
        self.max_depth_seen = 0   # Phase F observability — tracks the
                                    # deepest depth allowed_depth() has
                                    # ever actually returned, the one
                                    # piece of state added for
                                    # loop_observability.py; allowed_depth()
                                    # itself stays a pure function.

    def allowed_depth(self, meta_error: float) -> int:
        depth = self.base_max_depth
        if meta_error > self.expand_threshold:
            depth = min(self.hard_cap, self.base_max_depth + 2)
        self.max_depth_seen = max(self.max_depth_seen, depth)
        return depth


_controller: Optional[ReflectionController] = None


def get_reflection_controller() -> ReflectionController:
    global _controller
    if _controller is None:
        _controller = ReflectionController()
    return _controller
