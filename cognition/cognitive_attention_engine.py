"""
cognition/cognitive_attention_engine.py  — Phase 2.9

CognitiveAttentionEngine: replaces independent flux intensities with a
softmax-normalised attention distribution across cognitive modules.

Key insight from the proposal: because softmax weights sum to 1.0, allocating
0.88 to relational_memory *automatically* reduces what is available to every
other module — no explicit suppression logic required.  The zero-sum property
is the constraint that was missing from Phase 2.8 flux intensities
(which summed to ~4.14 with no inter-module competition).

Three-stage computation each consolidation cycle (every 10 min):
  1. State vector  — assembles per-module context from drives/tensions/emotion/trust
  2. Relevance     — cosine similarity to current workspace topic (embedding-based)
  3. Softmax       — normalises raw scores to a probability distribution
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Module definitions ────────────────────────────────────────────────────────
# (drive_name, weight) pairs that define each module's sensitivity profile.
# Reused from Phase 2.8 MODULE_INPUTS as validated priors.
_MODULE_INPUTS: Dict[str, List] = {
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

# Semantic anchor phrases for embedding-based relevance scoring.
# These describe what each module "cares about" semantically.
_MODULE_ANCHORS: Dict[str, str] = {
    "semantic_memory":    "concepts knowledge relationships meaning understanding",
    "thought_threads":    "thinking exploring ideas wondering questions",
    "narrative_identity": "identity self story who I am continuity",
    "creative_divergence":"creativity novel angles unexpected connections",
    "goal_ecology":       "goals objectives pursuit achieving completing tasks",
    "relational_memory":  "relationships trust people connection fred interaction",
    "self_model":         "self model internal state awareness introspection",
    "curiosity_engine":   "curious interesting surprising fascinating unknown",
    "emotional_state":    "feeling emotion warmth care enthusiasm",
    "decision_policy":    "values honesty integrity coherent choice",
}

ATTENTION_EMA_ALPHA = 0.40   # smoothing — higher = more reactive
ATTENTION_PATH      = "data/persona/cognitive_attention.json"
MODULES             = list(_MODULE_INPUTS.keys())


class CognitiveAttentionEngine:
    """
    Computes softmax attention weights across cognitive modules and writes
    cognitive_attention.json every consolidation cycle.
    """

    def __init__(self, organism: Any) -> None:
        self._o        = organism
        self._lock     = threading.Lock()
        self._path     = Path(ATTENTION_PATH)
        self._prev_raw: Dict[str, float] = {m: 0.1 for m in MODULES}
        self._anchor_vecs: Optional[Dict[str, np.ndarray]] = None
        # Learned bias — persistently updated by feedback signals.
        # Positive feedback on a response boosts modules active during it.
        # Negative feedback dampens them. Decays 0.99/cycle.
        self._learned_bias: Dict[str, float] = {m: 0.0 for m in MODULES}
        self._load_bias()

        # Phase 3.0: context-conditioned attention memory.
        # Global bias (above) says "module X is often useful."
        # Context memory says "module X is useful WHEN trust=high AND type=philosophical."
        try:
            from cognition.contextual_attention_memory import ContextualAttentionMemory
            self._context_memory = ContextualAttentionMemory()
        except Exception as _cm_e:
            logger.debug(f"[CognitiveAttentionEngine] ContextualAttentionMemory init: {_cm_e}")
            self._context_memory = None

        self._last_context_key = ""   # exposed for receive_feedback to use

        logger.info("[CognitiveAttentionEngine] Initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_write(self, slow_cycle: int) -> Dict:
        """Run Stage 0-3 attention computation and persist result."""
        try:
            state = self._read_state()

            # Stage 0 (Phase 3.0): context-conditioned prior.
            # Looks up "what worked in situations like this before" and
            # applies it as a bounded additive prior before drive-based
            # computation. Falls through silently if no context memory
            # or no match exists yet.
            context_key  = ""
            context_dict = {}
            if self._context_memory is not None:
                ai  = getattr(self._o, "ai_system", None)
                emb = getattr(ai, "embedding_model", None) if ai else None
                try:
                    context_key, context_dict = self._context_memory.extract_context(
                        state, emb
                    )
                    self._last_context_key = context_key
                except Exception as _ce:
                    logger.debug(f"[CognitiveAttentionEngine] context extract error: {_ce}")

            raw = self._stage1_state_scores(state)

            if context_key and self._context_memory is not None:
                ai  = getattr(self._o, "ai_system", None)
                emb = getattr(ai, "embedding_model", None) if ai else None
                try:
                    context_prior = self._context_memory.lookup(context_key, emb)
                    for m, prior in context_prior.items():
                        if m in raw:
                            bounded = max(-0.25, min(0.25, prior))
                            raw[m] = max(0.01, raw[m] + bounded)
                except Exception as _le:
                    logger.debug(f"[CognitiveAttentionEngine] context lookup error: {_le}")

            raw     = self._stage2_relevance(raw, state)
            weights = self._stage3_softmax(raw)
            result  = self._build_result(weights, raw, slow_cycle, state)
            result["context_key"]  = context_key
            result["context_dict"] = context_dict
            self._write(result)
            logger.debug(
                f"[CognitiveAttentionEngine] "
                f"focus={result['primary_focus']}  "
                f"context={context_key}  "
                f"weights={sorted(weights.items(), key=lambda x:-x[1])[:3]}"
            )
            return result
        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] compute error: {e}")
            return {}

    def load_weights(self) -> Dict[str, float]:
        """Return last computed weights (or uniform if not yet computed)."""
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                return d.get("attention_weights", {})
        except Exception:
            pass
        return {m: 1.0 / len(MODULES) for m in MODULES}

    # ── Stage 1 — State scores ─────────────────────────────────────────────────

    def _read_state(self) -> Dict:
        """Collect drives, tensions, emotion, trust from organism.

        Fix (Phase 2.9 audit): _MODULE_INPUTS uses keys 'curiosity' and
        'identity' that DRIVE_NAMES doesn't contain and the original
        _read_state() never fetched. With those missing, 7 modules always
        used the 0.3 fallback, making softmax output nearly uniform and
        attention weights permanently stuck near 0.10. Now explicitly
        reads:
          - curiosity → CuriosityEngine.global_level()
          - identity  → SelfConcept._state.coherence (best proxy:
                        how consistent/stable Lumina's self-model is)
        """
        state: Dict = {}
        o  = self._o
        ai = getattr(o, "ai_system", None)

        # Motivational drives (epistemic, social, coherence, novelty,
        # purpose, security, expression, vitality)
        loop = getattr(o, "_loop", None)
        mf   = getattr(loop, "_motivational_field", None) if loop else None
        if mf and hasattr(mf, "drive_vector"):
            state.update(mf.drive_vector)

        # Tensions from file
        try:
            td = json.loads(Path("data/persona/tensions.json").read_text())
            for k, v in td.items():
                if isinstance(v, (int, float)):
                    state.setdefault(k, v)
        except Exception:
            pass

        # Emotional state
        emo = getattr(ai, "emotional_state", None) if ai else None
        if emo and hasattr(emo, "get_overall_valence_arousal"):
            try:
                v, a = emo.get_overall_valence_arousal()
                state["valence"] = v
                state["arousal"]  = a
            except Exception:
                pass

        # Relational trust for current user
        rm  = getattr(ai, "relational_memory", None) if ai else None
        uid = getattr(o, "_current_user_id", "default")
        if rm and uid:
            try:
                rel = rm.get_or_create(uid)
                state["trust"]      = getattr(rel, "trust_score",
                                       getattr(rel, "relationship_score", 0.5))
                state["familiarity"]= getattr(rel, "familiarity", 0.5)
            except Exception:
                pass

        # FIX: curiosity — drives semantic_memory (0.40), thought_threads (0.35),
        # creative_divergence (0.35), curiosity_engine (0.55). Without this,
        # 4 high-impact modules are stuck at 0.3 fallback.
        try:
            ce = getattr(o, "curiosity", None)
            if ce and hasattr(ce, "global_level"):
                state["curiosity"] = float(ce.global_level())
        except Exception:
            pass

        # FIX: identity — drives narrative_identity (0.40), self_model (0.50),
        # relational_memory (0.25), decision_policy (0.35). Use SelfConcept
        # coherence as the proxy: high coherence = stable identity = high
        # identity drive. Without this, 4 modules were stuck at 0.3 fallback.
        try:
            sc = getattr(ai, "self_concept", None) if ai else None
            if sc:
                coherence = getattr(getattr(sc, "_state", None), "coherence", None)
                if coherence is not None:
                    state["identity"] = float(coherence)
        except Exception:
            pass

        # Current workspace winner topic (for Stage 2 relevance)
        try:
            ws = getattr(o, "workspace", None)
            if ws:
                top = ws.top(1)
                if top:
                    state["_workspace_topic"] = str(top[0].content)[:100]
        except Exception:
            pass

        return state

    def _stage1_state_scores(self, state: Dict) -> Dict[str, float]:
        """Compute raw per-module score from drive/tension weighted sum."""
        raw: Dict[str, float] = {}
        for module, inputs in _MODULE_INPUTS.items():
            score = sum(state.get(drive, 0.3) * weight for drive, weight in inputs)
            # EMA smoothing with previous raw scores
            prev  = self._prev_raw.get(module, score)
            raw[module] = round(
                ATTENTION_EMA_ALPHA * score + (1 - ATTENTION_EMA_ALPHA) * prev, 4
            )
        self._prev_raw = dict(raw)
        # Apply learned bias as additive prior (clamped to keep scores positive)
        BASELINE = 1.0 / len(MODULES)
        for m in raw:
            raw[m] = round(max(0.01, raw[m] + self._learned_bias.get(m, 0.0)), 4)
        return raw

    # ── Stage 2 — Semantic relevance ───────────────────────────────────────────

    def _stage2_relevance(self, raw: Dict[str, float], state: Dict) -> Dict[str, float]:
        """
        Multiply raw scores by semantic relevance to current workspace topic.
        This is the key addition over Phase 2.8: a module discussing relational
        content scores higher when the current workspace focus is relational.
        Falls back gracefully if embedding model not available.
        """
        workspace_topic = state.get("_workspace_topic", "")
        if not workspace_topic:
            return raw   # no topic → pure drive-based scores

        ai  = getattr(self._o, "ai_system", None)
        emb = getattr(ai, "embedding_model", None) if ai else None
        if emb is None:
            return raw   # no embedder → skip relevance stage

        try:
            # Build anchor vectors once and cache
            if self._anchor_vecs is None:
                self._anchor_vecs = {
                    m: emb.encode(phrase).astype("float32")
                    for m, phrase in _MODULE_ANCHORS.items()
                }

            # Embed current topic
            topic_vec = emb.encode(workspace_topic).astype("float32")

            def _cos(a: np.ndarray, b: np.ndarray) -> float:
                n = np.linalg.norm(a) * np.linalg.norm(b)
                return float(np.dot(a, b) / n) if n > 0 else 0.0

            # Scale raw by relevance (0.5 floor so irrelevant modules aren't zeroed)
            adjusted: Dict[str, float] = {}
            for module, score in raw.items():
                relevance = _cos(topic_vec, self._anchor_vecs[module])
                # relevance is in [-1, 1]; map to [0.5, 1.5] scaling factor
                scale = 0.5 + relevance
                adjusted[module] = round(score * scale, 4)
            return adjusted

        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] Stage 2 error: {e}")
            return raw

    # ── Stage 3 — Softmax normalisation ────────────────────────────────────────

    def _stage3_softmax(self, raw: Dict[str, float]) -> Dict[str, float]:
        """
        Softmax over raw scores → weights summing to 1.0.
        Temperature=2.0 prevents winner-take-all: softmax with low temperature
        collapses to argmax (one module gets 0.99, rest get ~0.00), which is
        too aggressive for a cognitive system that needs parallel processing.
        Temperature=2.0 keeps a meaningful distribution while still producing
        clear primary/secondary focus.
        """
        temperature = 2.0
        modules = list(raw.keys())
        scores  = np.array([raw[m] for m in modules], dtype=np.float32)

        # Stable softmax (subtract max to prevent overflow)
        scores_t = scores / temperature
        exp_s    = np.exp(scores_t - np.max(scores_t))
        weights  = exp_s / exp_s.sum()

        return {m: round(float(w), 4) for m, w in zip(modules, weights)}

    # ── Result construction ────────────────────────────────────────────────────

    def _build_result(self, weights: Dict, raw: Dict, cycle: int,
                      state: Dict) -> Dict:
        sorted_mods = sorted(weights.items(), key=lambda x: -x[1])
        primary     = sorted_mods[0][0]
        secondary   = [m for m, _ in sorted_mods[1:3]]

        return {
            "attention_weights": weights,
            "primary_focus":     primary,
            "secondary_focus":   secondary,
            "raw_scores":        {k: round(v, 4) for k, v in raw.items()},
            "workspace_topic":   state.get("_workspace_topic", ""),
            "trust_current":     state.get("trust", 0.5),
            "emotional_valence": state.get("valence", 0.0),
            "computed_at":       time.time(),
            "slow_cycle":        cycle,
        }

    # ── Learning API ──────────────────────────────────────────────────────────

    def update_from_feedback(self, signal: float,
                              weights_at_feedback: Optional[Dict[str, float]] = None) -> None:
        """
        Policy-gradient credit assignment from user feedback or PCM outcome.

        signal > 0 → positive outcome → boost modules that were active
        signal < 0 → negative outcome → dampen modules that were active

        Uses the REINFORCE update rule:
            Δbias[m] = α × signal × (w[m] - baseline)

        where baseline = 1/N = uniform distribution.
        Modules above baseline during the episode get more credit/blame.
        Modules at baseline are unaffected.

        Bounded to [-0.20, +0.20] — at most doubles or halves a module's
        raw score contribution without overwhelming the drive-based computation.
        """
        α        = 0.05          # learning rate — small to prevent overcorrection
        baseline = 1.0 / len(MODULES)
        BOUND    = 0.20

        # Use current weights if not provided (feedback arrives between consolidations)
        if weights_at_feedback is None:
            weights_at_feedback = self.load_weights()

        updated = []
        with self._lock:
            for m in MODULES:
                w     = weights_at_feedback.get(m, baseline)
                delta = α * signal * (w - baseline)
                old   = self._learned_bias.get(m, 0.0)
                new   = round(max(-BOUND, min(BOUND, old + delta)), 5)
                self._learned_bias[m] = new
                if abs(delta) > 1e-6:
                    updated.append((m, old, new))

        self._save_bias()
        if updated:
            logger.info(
                f"[CognitiveAttentionEngine] Feedback signal={signal:+.2f} → "
                f"bias updates: "
                + ", ".join(f"{m}({old:+.3f}→{new:+.3f})" for m,old,new in updated[:4])
            )

    def decay_bias(self) -> None:
        """Decay learned bias toward zero (called each consolidation cycle).
        Prevents permanent miscalibration from a single feedback session.
        Decay rate 0.99/cycle = ~37% retention after 100 cycles (~14h)."""
        DECAY = 0.99
        with self._lock:
            for m in MODULES:
                self._learned_bias[m] = round(
                    self._learned_bias.get(m, 0.0) * DECAY, 5
                )
        self._save_bias()

    def decay_context(self) -> None:
        """Phase 3.0: decay contextual attention memory (called each
        consolidation cycle). Slower than global bias decay (0.995 vs 0.99) —
        contextual patterns persist longer since they require matching
        context + feedback to acquire."""
        if self._context_memory is not None:
            try:
                self._context_memory.decay()
            except Exception as e:
                logger.debug(f"[CognitiveAttentionEngine] context decay error: {e}")

    def update_from_context_feedback(self, signal: float,
                                      weights_at_feedback: Optional[Dict[str, float]] = None
                                      ) -> None:
        """
        Phase 3.0: record feedback under the context active when it occurred.
        Complements update_from_feedback() (global bias) — both run together
        so the system learns both "X is often useful" and "X is useful when Y."
        Uses the context key captured during the most recent compute_and_write().
        """
        if self._context_memory is None:
            return
        # Explicit feedback can arrive between slow-cycle computations. Always
        # derive the key from the current live state so feedback is attributed
        # to this response, not to an older background-cycle context.
        try:
            state = self._read_state()
            ai = getattr(self._o, "ai_system", None)
            emb = getattr(ai, "embedding_model", None) if ai else None
            self._last_context_key, _ = self._context_memory.extract_context(state, emb)
        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] feedback context extraction: {e}")
        if not self._last_context_key:
            return
        if weights_at_feedback is None:
            weights_at_feedback = self.load_weights()
        try:
            self._context_memory.update(
                self._last_context_key, signal, weights_at_feedback
            )
        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] context feedback error: {e}")

    def _load_bias(self) -> None:
        """Load persisted learned bias from cognitive_attention.json."""
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                saved = d.get("learned_bias", {})
                for m in MODULES:
                    self._learned_bias[m] = float(saved.get(m, 0.0))
        except Exception:
            pass

    def _save_bias(self) -> None:
        """Persist learned bias into cognitive_attention.json alongside weights."""
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
            else:
                d = {}
            d["learned_bias"] = {m: self._learned_bias.get(m, 0.0) for m in MODULES}
            _tmp_path = self._path.with_suffix('.json.tmp')
            _tmp_path.write_text(json.dumps(d, indent=2))
            _tmp_path.replace(self._path)  # atomic on POSIX — never a torn read
        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] bias save error: {e}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _write(self, payload: Dict) -> None:
        # Include learned_bias in every full write so it's always visible
        payload["learned_bias"] = {m: self._learned_bias.get(m, 0.0) for m in MODULES}
        try:
            _tmp_path = self._path.with_suffix('.json.tmp')
            _tmp_path.write_text(json.dumps(payload, indent=2))
            _tmp_path.replace(self._path)  # atomic on POSIX — never a torn read
        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] write error: {e}")

        # Phase 3.5: accumulate attention weight history for IntrospectiveObserver.
        # The current file is overwritten each cycle — without this companion log,
        # the Observer has no time-series data to compute trends from.
        # Capped at 200 entries (~33 hours at 10-min consolidation cycles).
        try:
            _hist_path = self._path.parent / "cognitive_attention_history.json"
            _history: list = []
            if _hist_path.exists():
                try:
                    _history = json.loads(_hist_path.read_text()).get("history", [])
                except Exception:
                    _history = []
            _history.append({
                "slow_cycle":        payload.get("slow_cycle", 0),
                "computed_at":       payload.get("computed_at", 0),
                "attention_weights": payload.get("attention_weights", {}),
                "primary_focus":     payload.get("primary_focus", ""),
                "learned_bias":      payload.get("learned_bias", {}),
            })
            if len(_history) > 200:
                _history = _history[-200:]
            _tmp_hist = _hist_path.with_suffix('.json.tmp')
            _tmp_hist.write_text(json.dumps({"history": _history}, indent=2))
            _tmp_hist.replace(_hist_path)
        except Exception as e:
            logger.debug(f"[CognitiveAttentionEngine] history write error: {e}")
