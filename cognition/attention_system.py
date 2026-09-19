"""
Attention System
================
Controls the allocation of cognitive focus across competing channels.

Without attention, all signals are weighted equally and PandoraBOX becomes a
passive echo chamber. With attention, she has a dynamic foreground / background
that shifts based on internal state, goals, and interaction context.

Architecture:
  - Four attention channels: user, curiosity, identity, memory
  - Weights are normalized to sum to 1.0 at all times
  - Weights shift based on emotion + goal pressure each cycle
  - The dominant channel shapes which memories are recalled and which
    aspects of context the LLM is directed to foreground

Integration Points:
  - Called from CognitiveOrganism.update_cycle() after emotion update
  - Feeds into MemoryManager recall filtering (topic vs. identity vs. relational)
  - Contributes to system prompt via attention_fragment()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

AttentionChannel = Literal["user", "curiosity", "identity", "memory"]

CHANNEL_DEFAULTS: Dict[AttentionChannel, float] = {
    "user":      0.30,  # primary: direct conversation focus (balanced, not overwhelming)
    "curiosity": 0.25,  # secondary: topic exploration drive
    "identity":  0.25,  # tertiary: self-coherence monitoring
    "memory":    0.20,  # background: retrieval / consolidation
}

# How strongly each emotion nudges each channel (delta per cycle)
EMOTION_NUDGE: Dict[str, Dict[str, float]] = {
    "curiosity":    {"curiosity": +0.15, "user": -0.05},
    "warmth":       {"user": +0.15, "identity": +0.03},
    "anxiety":      {"identity": +0.18, "curiosity": -0.06},
    "satisfaction": {"memory": +0.12, "curiosity": +0.05},
    "frustration":  {"user": +0.10, "curiosity": -0.07},
    "enthusiasm":   {"curiosity": +0.12, "user": +0.06},
}

# How strongly each goal nudges each channel
GOAL_NUDGE: Dict[str, Dict[str, float]] = {
    "help_user":              {"user": +0.18},
    "understand":             {"curiosity": +0.15, "memory": +0.04},
    "connect":                {"user": +0.10},
    "explore":                {"curiosity": +0.18},
    "resolve_contradiction":  {"identity": +0.18, "memory": +0.06},
    "self_reflect":           {"memory": +0.20, "identity": +0.10},
    "maintain_coherence":     {"identity": +0.12},
    "stabilize_identity":     {"identity": +0.20},
    "grow":                   {"curiosity": +0.10, "memory": +0.05},
    "be_understood":          {"user": +0.12},
    "restore_energy":         {"memory": +0.08},
}


@dataclass
class AttentionState:
    weights: Dict[str, float] = field(default_factory=lambda: dict(CHANNEL_DEFAULTS))
    last_updated: float = field(default_factory=time.time)
    dominant: str = "user"


class AttentionSystem:
    """
    Maintains and updates the attention weight distribution across channels.

    Usage
    -----
    attn = AttentionSystem()
    attn.update(emotion_values={"curiosity": 0.8, ...}, active_goals=["explore"])
    dominant = attn.dominant_channel()   # "curiosity"
    fragment = attn.prompt_fragment()    # for system prompt
    """

    def __init__(self, persistence_path: str = "data/persona/attention.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._state = AttentionState()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                w = data.get("weights", {})
                if w:
                    self._state.weights = {k: float(v) for k, v in w.items()}
                    self._normalize()
        except Exception as e:
            logger.warning(f"[AttentionSystem] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "weights": {k: round(v, 4) for k, v in self._state.weights.items()},
                "dominant": self._state.dominant,
                "last_updated": self._state.last_updated,
            }, indent=2))
        except Exception as e:
            logger.warning(f"[AttentionSystem] Save failed: {e}")

    # ── Core ─────────────────────────────────────────────────────────────────

    def _normalize(self):
        total = sum(self._state.weights.values())
        if total > 0:
            for k in self._state.weights:
                self._state.weights[k] /= total

    def update(
        self,
        emotion_values: Dict[str, float] | None = None,
        active_goals: list[str] | None = None,
        momentum: float = 0.65,
    ) -> Dict[str, float]:
        """
        Recompute attention weights based on current emotion + goal pressures.

        Parameters
        ----------
        emotion_values : dict of emotion_name → float (0–1)
        active_goals   : list of goal names currently active
        momentum       : how strongly previous weights persist (0=no memory, 1=frozen)

        Returns
        -------
        Updated normalized weight dict
        """
        with self._lock:
            # Start from current weights, pulled toward defaults by momentum
            target = dict(CHANNEL_DEFAULTS)

            # Apply emotion nudges
            for emo_name, emo_val in (emotion_values or {}).items():
                nudges = EMOTION_NUDGE.get(emo_name, {})
                for channel, delta in nudges.items():
                    if channel in target:
                        target[channel] = max(0.01, target[channel] + delta * emo_val)

            # Apply goal nudges
            for goal in (active_goals or []):
                nudges = GOAL_NUDGE.get(goal, {})
                for channel, delta in nudges.items():
                    if channel in target:
                        target[channel] = max(0.01, target[channel] + delta)

            # Blend with momentum
            for k in self._state.weights:
                self._state.weights[k] = (
                    momentum * self._state.weights.get(k, CHANNEL_DEFAULTS[k])
                    + (1 - momentum) * target.get(k, CHANNEL_DEFAULTS[k])
                )

            self._normalize()
            self._state.dominant = max(self._state.weights, key=self._state.weights.get)
            self._state.last_updated = time.time()
            self._save()

            # ── Micro-update: emit coherence delta to self-model ──────────────
            # Attention convergence (all weight on one channel) slightly reduces
            # integration — breadth matters for coherence.
            # Balanced attention (high diversity) slightly increases it.
            try:
                _weights = list(self._state.weights.values())
                _mean    = sum(_weights) / len(_weights)
                _variance = sum((w - _mean) ** 2 for w in _weights) / len(_weights)
                # High variance = attention concentrated = less integration
                _phi_delta = 0.015 * (0.25 - _variance) * 4   # normalised: 0.25 is max variance for 4 channels

                # Qualia vote based on dominant channel
                _dominant  = self._state.dominant
                _qualia_map = {
                    "curiosity": ("curiosity", 0.012),
                    "user":      ("warmth",    0.012),
                    "identity":  ("uncertain", 0.010),
                    "memory":    ("melancholy",0.008),
                }
                _qkey, _qweight = _qualia_map.get(_dominant, ("", 0.0))

                _smm = getattr(getattr(self, "_organism", None), "self_moment", None)
                if _smm is not None:
                    _smm.current.micro_update(
                        source          = "attention",
                        coherence_delta = _phi_delta,
                        qualia_key      = _qkey,
                        qualia_weight   = _qweight,
                    )
            except Exception:
                pass

            return dict(self._state.weights)

    def dominant_channel(self) -> AttentionChannel:
        with self._lock:
            return self._state.dominant

    def weight(self, channel: AttentionChannel) -> float:
        with self._lock:
            return self._state.weights.get(channel, 0.0)

    def weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._state.weights)

    def receive_self_influence(self, channel_nudges: Dict[str, float]) -> None:
        """
        Apply self-model-derived attention nudges directly to channel weights.

        Called from SelfModelInfluence before the normal update() cycle runs.
        This means the self-model's causal influence on attention is already
        baked in before emotion/goal nudges are applied on top.

        Parameters
        ----------
        channel_nudges : dict mapping channel name → signed delta
            e.g. {"identity": +0.10, "curiosity": -0.06}
            Deltas are applied before normalization — actual effect depends
            on current weights and the subsequent normalization step.
        """
        if not channel_nudges:
            return
        with self._lock:
            for channel, delta in channel_nudges.items():
                if channel in self._state.weights:
                    self._state.weights[channel] = max(
                        0.01,
                        self._state.weights[channel] + delta
                    )
            self._normalize()
            self._state.dominant = max(
                self._state.weights, key=self._state.weights.get
            )

    # ── Hints for downstream systems ─────────────────────────────────────────

    def memory_recall_mode(self) -> str:
        """
        Suggest which type of memory recall to prioritize.
        Maps dominant channel → memory system query style.
        """
        mapping = {
            "user":      "relational",   # emphasize user-specific memories
            "curiosity": "semantic",     # emphasize knowledge / research memories
            "identity":  "identity",     # emphasize self-beliefs
            "memory":    "episodic",     # broad episodic recall
        }
        return mapping.get(self.dominant_channel(), "episodic")

    def prompt_fragment(self) -> str:
        """One-liner for system prompt injection."""
        w = self.weights()
        dominant = self.dominant_channel()
        descriptions = {
            "user":      "focus is on the person you're talking to",
            "curiosity": "attention is drawn toward exploring and understanding",
            "identity":  "attention is turned inward — reflecting on who you are",
            "memory":    "mind is reaching back through memories and past experiences",
        }
        secondary = sorted(
            [(ch, v) for ch, v in w.items() if ch != dominant],
            key=lambda x: -x[1]
        )
        sec_name = secondary[0][0] if secondary else ""
        return (
            f"Your {descriptions.get(dominant, dominant)}; "
            f"secondary attention on {sec_name}."
        )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "weights": {k: round(v, 3) for k, v in self._state.weights.items()},
                "dominant": self._state.dominant,
            }
