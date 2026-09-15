"""
cognition/contextual_attention_memory.py  — Phase 3.0

ContextualAttentionMemory: experience-conditioned attention priors.

Phase 2.9.1 learns a GLOBAL bias: "relational_memory is often useful."
Phase 3.0 learns a CONDITIONAL one: "relational_memory is useful WHEN
trust=high AND conversation=philosophical."

This is a non-parametric context-indexed table, not a neural network.
No gradients, no training loop, no backprop — a context key maps to running
mean signal contributions per module, retrieved via discretised exact match
with embedding-based soft fallback for near-misses.

Architecture:
  1. extract_context()  — discretise live state into a context key + dict
  2. lookup()            — exact match, else cosine-similarity blend of top-3
  3. update()             — EWA running-mean update per (context, module)
  4. decay()              — slow forgetting (0.995/cycle, ~2 day half-life)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONTEXT_PATH   = "data/persona/contextual_attention.json"
DECAY_RATE     = 0.995    # slower than global bias (0.99) — contexts persist longer
PRUNE_FLOOR    = 0.001    # remove contributions below this magnitude
MIN_OBSERVATIONS_FOR_USE = 2   # need at least this many feedback events to trust a context
SOFT_MATCH_TOP_K = 3
SOFT_MATCH_MIN_SIM = 0.55   # below this similarity, don't blend the context in

# Domain classification anchors — used to discretise "conversation type"
_TYPE_ANCHORS: Dict[str, str] = {
    "philosophical": "consciousness identity meaning existence philosophy ethics",
    "technical":     "code debugging architecture implementation software systems",
    "emotional":     "feelings emotions support comfort distress care empathy",
    "creative":      "imagination story art creative writing novel ideas",
    "factual":       "facts information data history science explain define",
}


class ContextualAttentionMemory:
    """
    Stores and retrieves context-conditioned attention priors.
    A "context" is a discretised snapshot of trust/curiosity/valence/type.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = Path(CONTEXT_PATH)
        self._contexts: Dict[str, Dict] = {}
        self._type_anchor_vecs: Optional[Dict[str, np.ndarray]] = None
        self._load()
        logger.info(
            f"[ContextualAttentionMemory] Initialised — "
            f"{len(self._contexts)} contexts loaded"
        )

    # ── Context extraction (discretisation) ──────────────────────────────────

    def extract_context(self, state: Dict, emb_model: Any) -> Tuple[str, Dict]:
        """
        Discretise continuous state into a context key + context dict.
        Returns ("trust=high|curiosity=high|valence=positive|type=philosophical",
                 {"trust": "high", "curiosity": "high", ...})
        """
        trust     = state.get("trust", 0.5)
        curiosity = state.get("curiosity", 0.5)
        valence   = state.get("valence", 0.0)
        topic     = state.get("_workspace_topic", "")

        ctx = {
            "trust":     self._bin(trust,     0.5, 0.7),
            "curiosity": self._bin(curiosity, 0.4, 0.65),
            "valence":   self._bin_signed(valence, -0.2, 0.2),
            "type":      self._classify_type(topic, emb_model),
        }
        key = "|".join(f"{k}={v}" for k, v in ctx.items())
        return key, ctx

    @staticmethod
    def _bin(value: float, low: float, high: float) -> str:
        if value < low:  return "low"
        if value > high: return "high"
        return "medium"

    @staticmethod
    def _bin_signed(value: float, neg_thresh: float, pos_thresh: float) -> str:
        if value < neg_thresh: return "negative"
        if value > pos_thresh: return "positive"
        return "neutral"

    def _classify_type(self, topic: str, emb_model: Any) -> str:
        """Classify conversation type via embedding cosine to 5 domain anchors."""
        if not topic or emb_model is None:
            return "factual"   # safe default — least presumptive category
        try:
            if self._type_anchor_vecs is None:
                self._type_anchor_vecs = {
                    t: emb_model.encode(phrase).astype("float32")
                    for t, phrase in _TYPE_ANCHORS.items()
                }
            topic_vec = emb_model.encode(topic).astype("float32")

            def _cos(a, b):
                n = np.linalg.norm(a) * np.linalg.norm(b)
                return float(np.dot(a, b) / n) if n > 0 else 0.0

            scores = {t: _cos(topic_vec, v) for t, v in self._type_anchor_vecs.items()}
            return max(scores, key=scores.get)
        except Exception as e:
            logger.debug(f"[ContextualAttentionMemory] type classify error: {e}")
            return "factual"

    # ── Lookup (retrieval) ────────────────────────────────────────────────────

    def lookup(self, context_key: str, emb_model: Any) -> Dict[str, float]:
        """
        Return module prior scores for this context.
        1. Exact match — use directly if it has enough observations.
        2. Soft match — embed key text, cosine similarity vs stored keys,
           blend top-K weighted by similarity (only if similarity exceeds floor).
        3. No match — return {} (caller falls through to Stage 1 only).
        """
        with self._lock:
            # 1. Exact match
            exact = self._contexts.get(context_key)
            if exact and exact.get("total_feedback_events", 0) >= MIN_OBSERVATIONS_FOR_USE:
                return {
                    m: sig["mean"] for m, sig in exact.get("module_signals", {}).items()
                }

            if not self._contexts or emb_model is None:
                return {}

            # 2. Soft match via embedding similarity on the key text
            try:
                query_vec = emb_model.encode(context_key.replace("|", " ").replace("=", " ")
                                              ).astype("float32")

                def _cos(a, b):
                    n = np.linalg.norm(a) * np.linalg.norm(b)
                    return float(np.dot(a, b) / n) if n > 0 else 0.0

                scored = []
                for key, data in self._contexts.items():
                    if data.get("total_feedback_events", 0) < MIN_OBSERVATIONS_FOR_USE:
                        continue
                    key_vec = emb_model.encode(
                        key.replace("|", " ").replace("=", " ")
                    ).astype("float32")
                    sim = _cos(query_vec, key_vec)
                    if sim >= SOFT_MATCH_MIN_SIM:
                        scored.append((sim, data))

                if not scored:
                    return {}

                scored.sort(key=lambda x: -x[0])
                top = scored[:SOFT_MATCH_TOP_K]
                total_sim = sum(s for s, _ in top)
                if total_sim <= 0:
                    return {}

                blended: Dict[str, float] = {}
                for sim, data in top:
                    weight = sim / total_sim
                    for m, sig in data.get("module_signals", {}).items():
                        blended[m] = blended.get(m, 0.0) + sig["mean"] * weight
                return {m: round(v, 4) for m, v in blended.items()}

            except Exception as e:
                logger.debug(f"[ContextualAttentionMemory] soft match error: {e}")
                return {}

    # ── Update (learning) ─────────────────────────────────────────────────────

    def update(self, context_key: str, signal: float,
               weights_at_feedback: Dict[str, float]) -> None:
        """
        Record (signal × weight_above_baseline) for each module under this
        context key. Uses an exact-match running mean (EWA via count-weighting).
        """
        if not context_key:
            return
        baseline = 1.0 / max(len(weights_at_feedback), 1)

        with self._lock:
            entry = self._contexts.setdefault(context_key, {
                "module_signals": {},
                "last_updated": time.time(),
                "total_feedback_events": 0,
            })
            for module, weight in weights_at_feedback.items():
                contribution = signal * (weight - baseline)
                sig = entry["module_signals"].setdefault(
                    module, {"sum": 0.0, "count": 0, "mean": 0.0}
                )
                sig["sum"]  += contribution
                sig["count"] += 1
                sig["mean"]  = round(sig["sum"] / sig["count"], 5)

            entry["total_feedback_events"] += 1
            entry["last_updated"] = time.time()

        self._save()
        logger.info(
            f"[ContextualAttentionMemory] Updated context "
            f"'{context_key}' (signal={signal:+.2f}, "
            f"n={self._contexts[context_key]['total_feedback_events']})"
        )

    # ── Decay (forgetting) ────────────────────────────────────────────────────

    def decay(self) -> None:
        """
        Decay all module signal means toward zero. Slower than global bias
        (0.995 vs 0.99) — contextual patterns should persist longer since
        they're harder to acquire (need matching context + feedback).
        Prunes near-zero entries to keep the file bounded.
        """
        with self._lock:
            empty_contexts = []
            for ckey, data in self._contexts.items():
                empty_modules = []
                for m, sig in data.get("module_signals", {}).items():
                    sig["mean"] = round(sig["mean"] * DECAY_RATE, 5)
                    if abs(sig["mean"]) < PRUNE_FLOOR:
                        empty_modules.append(m)
                for m in empty_modules:
                    del data["module_signals"][m]
                if not data["module_signals"]:
                    empty_contexts.append(ckey)
            for ckey in empty_contexts:
                del self._contexts[ckey]

        self._save()

    # ── Dashboard support ──────────────────────────────────────────────────────

    def top_contexts(self, n: int = 5) -> List[Dict]:
        """Return n contexts with the most feedback events, for dashboard display."""
        with self._lock:
            sorted_ctx = sorted(
                self._contexts.items(),
                key=lambda x: -x[1].get("total_feedback_events", 0)
            )
            result = []
            for key, data in sorted_ctx[:n]:
                top_modules = sorted(
                    data.get("module_signals", {}).items(),
                    key=lambda x: -abs(x[1]["mean"])
                )[:3]
                result.append({
                    "context_key": key,
                    "observations": data.get("total_feedback_events", 0),
                    "top_modules": [
                        {"module": m, "mean": sig["mean"]} for m, sig in top_modules
                    ],
                })
            return result

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                self._contexts = d.get("contexts", {})
        except Exception as e:
            logger.debug(f"[ContextualAttentionMemory] load error: {e}")
            self._contexts = {}

    def _save(self) -> None:
        try:
            payload = {
                "contexts": self._contexts,
                "total_observations": sum(
                    c.get("total_feedback_events", 0) for c in self._contexts.values()
                ),
                "last_written": time.time(),
            }
            _tmp_path = self._path.with_suffix('.json.tmp')
            _tmp_path.write_text(json.dumps(payload, indent=2))
            _tmp_path.replace(self._path)  # atomic on POSIX — never a torn read
        except Exception as e:
            logger.debug(f"[ContextualAttentionMemory] save error: {e}")
