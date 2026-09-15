"""
cognition/cognitive_flux_engine.py  — Phase 2.8 Gap 1

CognitiveFluxEngine: per-module flux intensity from drives, tensions,
emotion, and trust — the original Phase 2.8 "organic ecosystem" signal
that preceded Phase 2.9's softmax-normalised attention allocation.

Note on relationship to CognitiveAttentionEngine (Phase 2.9): Phase 2.9
supersedes this engine's role in workspace competition and goal/curiosity
biasing — it does the same per-module weighting but with a proper
zero-sum softmax constraint that this engine lacks (its weights sum to an
arbitrary, unconstrained total). CognitiveAttentionEngine is preferred for
anything actually consuming attention/flux weights going forward.

This file is restored because several OTHER Phase 2.8 gaps (4, 5, 6) read
flux state to decide when to fire (e.g. "creative flux > 0.35" gates
CreativeDivergenceEngine.inject_to_threads()) — those gaps depend on this
engine's output existing, even though Phase 2.9 has superseded its role
in the competition/biasing pathway itself.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

FLUX_PATH = "data/persona/cognitive_flux.json"
FLUX_EMA_ALPHA = 0.35

# Same module → (drive, weight) mapping as the original Phase 2.8 design,
# also reused by cognitive_attention_engine.py as Stage 1 priors.
MODULE_INPUTS: Dict[str, list] = {
    "semantic_memory":    [("curiosity",0.40),("epistemic",0.35),("expression",0.25)],
    "thought_threads":    [("curiosity",0.35),("expression",0.35),("social",0.30)],
    "narrative_identity": [("identity", 0.40),("social",   0.35),("coherence",0.25)],
    "creative_divergence":[("expression",0.45),("curiosity",0.35),("vitality", 0.20)],
    "goal_ecology":       [("coherence",0.40),("epistemic",0.35),("vitality", 0.25)],
    "relational_memory":  [("social",  0.55),("identity", 0.25),("coherence",0.20)],
    "self_model":         [("identity",0.50),("coherence",0.30),("epistemic",0.20)],
    "curiosity_engine":   [("curiosity",0.55),("expression",0.25),("epistemic",0.20)],
    "emotional_state":    [("social",  0.40),("vitality", 0.35),("coherence",0.25)],
    "decision_policy":    [("coherence",0.40),("identity", 0.35),("epistemic",0.25)],
}


class CognitiveFluxEngine:
    """Computes per-module flux intensity (unconstrained, EMA-smoothed)
    and broadcasts the dominant flow to the workspace."""

    def __init__(self, organism: Any) -> None:
        self._o = organism
        self._lock = threading.Lock()
        self._path = Path(FLUX_PATH)
        self._prev: Dict[str, float] = {m: 0.3 for m in MODULE_INPUTS}
        logger.info("[CognitiveFluxEngine] Initialised")

    def compute_and_write(self) -> Dict[str, float]:
        try:
            state = self._read_state()
            flux: Dict[str, float] = {}
            for module, inputs in MODULE_INPUTS.items():
                score = sum(state.get(drive, 0.3) * weight for drive, weight in inputs)
                prev = self._prev.get(module, score)
                flux[module] = round(
                    FLUX_EMA_ALPHA * score + (1 - FLUX_EMA_ALPHA) * prev, 4
                )
            self._prev = dict(flux)
            self._write(flux)
            self._broadcast_dominant(flux)
            return flux
        except Exception as e:
            logger.debug(f"[CognitiveFluxEngine] compute error: {e}")
            return {}

    def get_module_flux(self, module: str) -> float:
        """Read the last computed flux intensity for a module (0.0 if unknown)."""
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                return float(d.get("flux", {}).get(module, 0.0))
        except Exception:
            pass
        return 0.0

    def _read_state(self) -> Dict[str, float]:
        state: Dict[str, float] = {}
        o = self._o
        ai = getattr(o, "ai_system", None)
        loop = getattr(o, "_loop", None)
        mf = getattr(loop, "_motivational_field", None) if loop else None
        if mf and hasattr(mf, "drive_vector"):
            state.update(mf.drive_vector)
        emo = getattr(ai, "emotional_state", None) if ai else None
        if emo and hasattr(emo, "get_overall_valence_arousal"):
            try:
                v, a = emo.get_overall_valence_arousal()
                state["valence"] = v
                state["arousal"] = a
            except Exception:
                pass
        return state

    def _broadcast_dominant(self, flux: Dict[str, float]) -> None:
        try:
            ws = getattr(self._o, "workspace", None)
            if not ws or not flux:
                return
            dominant = max(flux, key=flux.get)
            ws.broadcast(
                source="cognitive_flux.dominant",
                content=f"Cognitive flow: {dominant.replace('_',' ')} ({flux[dominant]:.2f})",
                priority=0.40,
            )
        except Exception:
            pass

    def _write(self, flux: Dict[str, float]) -> None:
        try:
            payload = {"flux": flux, "computed_at": time.time()}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            _tmp = self._path.with_suffix('.json.tmp')
            _tmp.write_text(json.dumps(payload, indent=2))
            _tmp.replace(self._path)
        except Exception as e:
            logger.debug(f"[CognitiveFluxEngine] write error: {e}")
