"""
loop_observability — Phase F
================================
Five metrics, all computed from real, already-persisted state built this
session — nothing here is a new tracked quantity invented for this
module alone; each reads an existing structure.

  novel_symbols_created        <- len(SymbolSystem.symbols)
  max_reflection_depth_reached <- a running max tracked on
                                   ReflectionController itself (added
                                   here — allowed_depth() was previously
                                   stateless/pure, this adds the one
                                   piece of state needed to answer "what
                                   was the deepest reflection actually
                                   went", not to change its behavior)
  revision_count_by_origin     <- UnifiedRevisionGateway.log, grouped by
                                   whether reasoning text was tagged by
                                   reflection_proposal_bridge.py
                                   ("[[origin:reflection]]") vs anything
                                   else ("[[origin:other]]")
  symbol_activation_entropy    <- Shannon entropy over symbol
                                   activation_count (same formula already
                                   verified in test_flux_snowball_replay_v2.py
                                   for workspace-hypothesis entropy)
  self_description_drift_events <- len(SelfDescription._drift_events)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

logger = logging.getLogger(__name__)

ORIGIN_TAG = "[[origin:reflection]]"


def shannon_entropy(counts: list) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def snapshot(organism: Any) -> Dict[str, Any]:
    out = {
        "novel_symbols_created": 0,
        "max_reflection_depth_reached": 0,
        "revision_count_by_origin": {"reflection": 0, "other": 0},
        "symbol_activation_entropy": 0.0,
        "self_description_drift_events": 0,
    }

    try:
        from cognition.symbol_system import get_symbol_system
        sym_sys = get_symbol_system(organism)
        out["novel_symbols_created"] = len(sym_sys.symbols)
        out["symbol_activation_entropy"] = shannon_entropy(
            [s.activation_count for s in sym_sys.symbols.values()]
        )
    except Exception as e:
        logger.debug(f"[LoopObservability] symbol metrics failed (non-fatal): {e}")

    try:
        from cognition.reflection_controller import get_reflection_controller
        out["max_reflection_depth_reached"] = get_reflection_controller().max_depth_seen
    except Exception as e:
        logger.debug(f"[LoopObservability] reflection depth metric failed (non-fatal): {e}")

    try:
        from cognition.unified_revision_gateway import get_unified_revision_gateway
        gateway = get_unified_revision_gateway()
        for d in gateway.log:
            if ORIGIN_TAG in (d.reason or ""):
                out["revision_count_by_origin"]["reflection"] += 1
            else:
                out["revision_count_by_origin"]["other"] += 1
    except Exception as e:
        logger.debug(f"[LoopObservability] revision count metric failed (non-fatal): {e}")

    try:
        from cognition.self_description import get_self_description
        out["self_description_drift_events"] = len(get_self_description()._drift_events)
    except Exception as e:
        logger.debug(f"[LoopObservability] drift event metric failed (non-fatal): {e}")

    return out
