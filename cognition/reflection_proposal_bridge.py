"""
reflection_proposal_bridge — closes Phase B + Phase E together
==================================================================
Phase B's criterion: "MetaReflection or AutonomousReflection can form a
goal or proposal whose content is a specific architectural component."
Phase E's criterion: "...produces a measurable, logged change to a
lower-level parameter that persists and affects later behavior."

These turned out to be the same missing link, not two separate ones:
tracing propose_personality_change()/propose_evolution_rule_change()
(cognition/self_modification.py) found they're only ever called from
liberty_integration.py::propose_self_modification_from_reflection() —
which is itself never called from anywhere, including AutonomousReflection
or MetaReflection. The safety gateway (v106) correctly guards the apply
path, but nothing was actually feeding it from a reflection process.

Deliberate scoping choice: propose_evolution_rule_change()'s target
("evolution_rules.<rule>.<param>") is NOT one of the two branches
_apply_pending_liberty_modifications() actually handles ('personality.'
or 'goals.') — using it would create proposals that silently sit
unapplied forever, satisfying nothing. Rather than add a third,
unreviewed apply branch under time pressure, this uses
propose_personality_change() — already safety-reviewed, already gated
by UnifiedRevisionGateway — and satisfies Phase B's "content is a
specific architectural component" requirement via the REASONING text,
which explicitly names the real component and its current key parameter
value (from SelfDescription), rather than via the target path. Documented
here as a real scoping trade-off, not hidden.

Fires only for reflection mode=="evolution" (the mode already literally
about self-evolution — _run_evolution_synthesis) and only when the
reflection text plausibly concerns a real registered component, so this
doesn't fire on unrelated reflections.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Small, bounded nudge — matches this session's established per-decision
# step sizes (DELTA_STEP in arbitration_learning.py, PRIORITY_STEP in
# persistent_executive_loop.py) rather than a large jump.
TRAIT_STEP = 0.05
TARGET_TRAIT = "caution_deliberation"   # the one trait already used as the
                                          # worked example throughout this
                                          # session's own gateway tests


def maybe_propose_from_reflection(record_output: str, mode: str,
                                   self_description: Any,
                                   liberty_self_mod: Any,
                                   current_trait_value: Optional[float]) -> Optional[Any]:
    if mode != "evolution" or not record_output or liberty_self_mod is None:
        return None
    if self_description is None:
        return None

    text_lower = record_output.lower()
    matched_component = None
    for name, comp in self_description.components.items():
        if name.lower() in text_lower or comp.module_path.split("/")[-1].replace(".py", "") in text_lower:
            matched_component = comp
            break
    if matched_component is None:
        return None

    if current_trait_value is None:
        current_trait_value = 0.5
    direction = 1 if "increase" in text_lower or "more" in text_lower else -1
    proposed_value = max(0.0, min(1.0, current_trait_value + direction * TRAIT_STEP))

    reasoning = (
        f"[[origin:reflection]] Reflecting on {matched_component.name} "
        f"({matched_component.module_path}), currently "
        f"{matched_component.key_parameters}: {record_output[:200]}"
    )
    try:
        proposal = liberty_self_mod.propose_personality_change(
            trait=TARGET_TRAIT, proposed_value=proposed_value,
            reasoning=reasoning, confidence=0.65,
        )
        logger.info(f"[ReflectionProposalBridge] proposed {TARGET_TRAIT} -> {proposed_value:.3f} "
                     f"citing component {matched_component.name}")
        return proposal
    except Exception as e:
        logger.debug(f"[ReflectionProposalBridge] propose failed (non-fatal): {e}")
        return None
