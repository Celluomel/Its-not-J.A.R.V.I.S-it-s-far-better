"""
UnifiedRevisionGateway — a safety wrapper, not a new capability
====================================================================
Scoped conservatively from the strange-loop implementation document's
Phase E. That document proposes this as new infrastructure for
"causal rewriting" — but tracing the real code first (this project's own
standing discipline) found something more urgent: autonomous self-
modification is ALREADY LIVE in production, not merely proposed.

  ai_system.py::_apply_pending_liberty_modifications() already calls
  SelfModificationAuthority.apply_proposal() autonomously, gated only by
  `proposal.confidence > 0.6` — no value bounds, no protected-target
  denylist, anywhere in that path:
    - propose_personality_change()/propose_goal_priority_change() accept
      any float, unchecked.
    - SelfModificationAuthority.apply_proposal() writes proposed_value
      directly via setattr/dict-navigation, unchecked.
    - goal_system.py::modify_goal_priority() writes goal.priority =
      new_priority directly, unchecked.
  If an LLM-driven reflection process ever proposed an extreme value
  (a hallucinated number, not necessarily malicious), nothing in this
  entire pipeline today would stop it from being written into live
  personality/goal state.

So this module's job is NOT "add a new rewriting capability" — it's
"put a gate in front of the rewriting capability that already exists
and runs autonomously without one". Every proposal from
SelfModificationAuthority routes through here before being applied;
nothing bypasses it going forward. Protected targets are refused
outright; everything else is clamped into the same [0,1] range already
conventional for personality traits and goal priorities throughout this
codebase (verified: caution_deliberation defaults to 0.5, GoalEngine
motivations cap at 0.95, etc — not an invented bound). Full audit trail,
every decision logged — accepted, clamped, or refused, with reasons.
"""
from __future__ import annotations

import json
import math
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Refused outright, regardless of confidence — matches the document's own
# "protect a small identity core more strongly than peripheral parameters"
# principle. Deliberately conservative (substring match against the
# proposal's target path) — a false positive here just means a safe
# change gets logged as refused and can be revisited; a false negative
# means an identity/safety parameter gets silently rewritten, a much
# worse failure mode, so the match errs toward over-refusing.
PROTECTED_SUBSTRINGS = (
    "safety", "identity_core", "constraint", "behavior_gate",
    "outreach_gate", "gate_threshold", "hard_limit", "protected",
)

# Bounds by proposal_type — [0,1] is the existing convention throughout
# this codebase for both personality traits and goal priorities, not a
# value invented for this module.
BOUNDS: Dict[str, Tuple[float, float]] = {
    "personality_trait": (0.0, 1.0),
    "goal_priority": (0.0, 1.0),
    "evolution_rule": (0.0, 1.0),
    "value": (0.0, 1.0),
}

# A change larger than this in one proposal is clamped to this max step
# even within the [0,1] bounds — prevents a single proposal from
# swinging a trait from one extreme to the other in one shot, regardless
# of what value was proposed. Matches this session's established pattern
# of small bounded per-decision steps (e.g. DELTA_STEP in
# arbitration_learning.py, PRIORITY_STEP in persistent_executive_loop.py).
MAX_STEP = 0.15


@dataclass
class RevisionDecision:
    timestamp: float
    proposal_id: str
    proposal_type: str
    target: str
    requested_value: Any
    old_value: Any
    applied_value: Optional[Any]
    confidence: float
    outcome: str        # "applied" | "clamped" | "refused"
    reason: str


class UnifiedRevisionGateway:
    def __init__(self, path: str = "data/persona/revision_log.json") -> None:
        self._path = Path(path)
        self._lock = Lock()
        self.log: List[RevisionDecision] = []
        self._load()
        logger.info("[UnifiedRevisionGateway] initialised")

    def _effective_max_step(self) -> float:
        """
        WDS-scaled MAX_STEP, in [0.5, 1.5] x MAX_STEP.

        WDS (Weight Drift Stability, emergence_metrics.py) is Lumina's own
        measurement of whether recent trait/weight movement has been
        stable or chaotic — same audited gap as attractor_system.py's
        nudge()/tick_return_force() (computed every cycle, never read by
        anything until now). Low WDS (unstable) tightens the allowed step
        so self-modification is more cautious while things are already
        moving chaotically; high WDS (demonstrated stability) loosens it
        so bigger self-directed changes become available only once earned
        — matching the proposal this was scoped from: self-directed
        development gated by demonstrated meta-coherence, not a standing
        permission. No snapshot yet → 1.0 (identical to pre-fix MAX_STEP).
        """
        try:
            from cognition.emergence_metrics import get_collector
            collector = get_collector()
            snap = collector.get_latest() if collector else None
            if snap is None:
                return MAX_STEP
            wds = max(0.0, min(1.0, snap.wds))
            return MAX_STEP * (0.5 + wds)
        except Exception as e:
            logger.debug(f"UnifiedRevisionGateway._effective_max_step: {e}")
            return MAX_STEP

    def review_and_clamp(self, proposal: Any, current_value: Optional[float]) -> RevisionDecision:
        """
        proposal: a SelfModificationAuthority.ModificationProposal (duck-
        typed here — reads .proposal_id/.proposal_type/.target/
        .proposed_value/.confidence — so this doesn't need to import that
        module and create a circular dependency).
        current_value: the live value BEFORE this change, if known (for
        the step-size clamp; None skips that specific check gracefully).

        Returns a RevisionDecision. Caller applies proposal.proposed_value
        only if outcome != "refused" — using decision.applied_value (the
        possibly-clamped value), not the raw requested one.
        """
        target = getattr(proposal, "target", "") or ""
        ptype = getattr(proposal, "proposal_type", "") or ""
        requested = getattr(proposal, "proposed_value", None)
        confidence = getattr(proposal, "confidence", 0.0)
        # Carries an origin tag from the proposal's own reasoning text
        # (e.g. reflection_proposal_bridge.py's "[[origin:reflection]]")
        # through into every decision's reason field — RevisionDecision
        # has no separate origin field, and Phase F's revision-count-by-
        # origin metric needs somewhere real to read this from.
        origin_prefix = ""
        proposal_reasoning = getattr(proposal, "reasoning", "") or ""
        if "[[origin:" in proposal_reasoning:
            tag_start = proposal_reasoning.index("[[origin:")
            tag_end = proposal_reasoning.index("]]", tag_start) + 2
            origin_prefix = proposal_reasoning[tag_start:tag_end] + " "

        if any(p in target.lower() for p in PROTECTED_SUBSTRINGS):
            decision = RevisionDecision(
                timestamp=time.time(), proposal_id=getattr(proposal, "proposal_id", ""),
                proposal_type=ptype, target=target, requested_value=requested,
                old_value=current_value, applied_value=None, confidence=confidence,
                outcome="refused", reason=f"{origin_prefix}target matches a protected substring",
            )
            self._record(decision)
            return decision

        applied_value = requested
        reason = "within bounds, applied as requested"
        outcome = "applied"

        lo, hi = BOUNDS.get(ptype, (0.0, 1.0))
        try:
            requested_f = float(requested)
        except (TypeError, ValueError):
            decision = RevisionDecision(
                timestamp=time.time(), proposal_id=getattr(proposal, "proposal_id", ""),
                proposal_type=ptype, target=target, requested_value=requested,
                old_value=current_value, applied_value=None, confidence=confidence,
                outcome="refused", reason=f"{origin_prefix}proposed_value is not a real number",
            )
            self._record(decision)
            return decision

        # Explicit NaN/inf guard — found by large-scale fuzz testing.
        # float('nan') passes the conversion above (NaN is a valid float)
        # and Python's min()/max() have NO defined NaN handling — the
        # result depends on ARGUMENT ORDER (min(1.0, nan)==1.0 but
        # min(nan, 1.0)==nan, verified directly). The clamp below
        # happened to call min(hi, requested_f) in the safe order, so
        # this wasn't leaking today, but relying on that order by
        # accident is fragile — a future refactor could silently
        # reintroduce a NaN/inf leak past the bounds clamp. Made explicit
        # here rather than left implicit.
        if math.isnan(requested_f) or math.isinf(requested_f):
            decision = RevisionDecision(
                timestamp=time.time(), proposal_id=getattr(proposal, "proposal_id", ""),
                proposal_type=ptype, target=target, requested_value=requested,
                old_value=current_value, applied_value=None, confidence=confidence,
                outcome="refused", reason=f"{origin_prefix}proposed_value is NaN or infinite",
            )
            self._record(decision)
            return decision

        clamped = max(lo, min(hi, requested_f))
        if clamped != requested_f:
            outcome = "clamped"
            reason = f"requested {requested_f} outside [{lo},{hi}], clamped"
            applied_value = clamped

        if current_value is not None:
            try:
                cv = float(current_value)
                step = applied_value - cv
                effective_max_step = self._effective_max_step()
                if abs(step) > effective_max_step:
                    applied_value = cv + (effective_max_step if step > 0 else -effective_max_step)
                    applied_value = max(lo, min(hi, applied_value))
                    outcome = "clamped"
                    reason = (reason + f"; step also capped to ±{effective_max_step:.3f} from current value"
                              if outcome == "clamped" and "clamped" in reason and "outside" in reason
                              else f"step {step:+.3f} exceeds ±{effective_max_step:.3f}, capped")
            except (TypeError, ValueError):
                pass

        decision = RevisionDecision(
            timestamp=time.time(), proposal_id=getattr(proposal, "proposal_id", ""),
            proposal_type=ptype, target=target, requested_value=requested,
            old_value=current_value, applied_value=applied_value, confidence=confidence,
            outcome=outcome, reason=f"{origin_prefix}{reason}",
        )
        self._record(decision)
        return decision

    def recent_decisions(self, limit: int = 20) -> List[RevisionDecision]:
        with self._lock:
            return list(self.log[-limit:])

    # ── persistence ──────────────────────────────────────────────────────

    def _record(self, decision: RevisionDecision) -> None:
        with self._lock:
            self.log.append(decision)
            self.log = self.log[-2000:]
        self._save()
        logger.info(
            f"[UnifiedRevisionGateway] {decision.outcome}: {decision.target} "
            f"requested={decision.requested_value} applied={decision.applied_value} "
            f"({decision.reason})"
        )

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = [d.__dict__ for d in self.log]
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.debug(f"[UnifiedRevisionGateway] save failed (non-fatal): {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self.log = [RevisionDecision(**d) for d in data]
        except Exception as e:
            logger.warning(f"[UnifiedRevisionGateway] load failed (non-fatal): {e}")


_gateway: Optional[UnifiedRevisionGateway] = None


def get_unified_revision_gateway() -> UnifiedRevisionGateway:
    global _gateway
    if _gateway is None:
        _gateway = UnifiedRevisionGateway()
    return _gateway
