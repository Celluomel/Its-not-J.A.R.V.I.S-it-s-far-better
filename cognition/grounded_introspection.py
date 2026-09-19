"""
State-Grounded Introspection Generator
=======================================
Replaces the LLM's archetypal human introspection patterns with questions
derived from PandoraBOX's actual internal state.

The analysis is precise:
  LLM output: "What is my purpose?" (generic existential template)
  Grounded output: "Why do I keep shifting away from exploration when
                    contradiction pressure rises?" (IDX + commitment data)

The difference between storytelling and self-modeling is whether the
question comes from the LLM's training distribution or from measured
internal state. This module bridges that gap.

Called from GAE's _action_self_question() when the system has enough
internal state to generate a grounded question rather than a generic one.
"""

import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


logger.info("[Introspection] grounded_introspection module loaded ✅")


class GroundedIntrospectionGenerator:
    """
    Generates self-reflective questions anchored to actual system state.

    Priority order (most grounded first):
      1. Active contradiction + recent commitment divergence
      2. IDX high (identity drift measurable)
      3. Curiosity drive high but no progress (epistemic stall)
      4. Dominant pressure not reflected in recent goals
      5. Fallback: generic but topic-specific question
    """

    # How long a generated question stays valid before we regenerate
    QUESTION_TTL = 600   # 10 minutes

    def __init__(self, organism: Any):
        self._o = organism
        self._last_question: str = ""
        self._last_question_ts: float = 0.0

    def generate(self, topic: str, prompt_frame: int = 0) -> str:
        """
        Generate a grounded introspective question about `topic`.
        Returns the question string for use as an LLM prompt.
        """
        state = self._read_state()
        question = self._derive_question(state, topic, prompt_frame)
        self._last_question = question
        self._last_question_ts = time.time()

        source = state.get('source', 'fallback')
        if source != 'fallback':
            logger.info(
                f"[Introspection] 🧠 Grounded question ({source}): "
                f"{question[:80]}"
            )
        else:
            logger.info(
                f"[Introspection] fallback question (no signal above threshold) "
                f"IDX={state.get('idx',0):.2f} consistency={state.get('consistency',1):.2f} "
                f"curiosity={state.get('curiosity_level',0):.2f}"
            )
        # Record for emergence metrics
        try:
            from cognition.emergence_metrics import get_collector
            c = get_collector()
            if c:
                c.record_introspection(state.get('source', 'fallback'))
        except Exception:
            pass
        return question

    # ── State reading ─────────────────────────────────────────────────────────

    def _read_state(self) -> Dict:
        """Read relevant internal signals. Safe — all reads, no writes."""
        state = {'source': 'fallback'}

        # IDX (identity drift) — use Observatory.latest() not _last_snapshot
        try:
            obs = getattr(self._o, 'observatory', None)
            if obs and hasattr(obs, 'latest'):
                snap = obs.latest()
                if snap:
                    state['idx'] = getattr(snap, 'idx', 0.0)
                    state['ccs'] = getattr(snap, 'ccs', 0.5)
                    logger.debug(f"[Introspection] IDX={state['idx']:.3f} CCS={state['ccs']:.3f}")
        except Exception as _e:
            logger.debug(f"[Introspection] obs read error: {_e}")

        # Commitment layer — lazy-init'd on InternalLoop after first workspace winner
        try:
            il = (getattr(self._o, '_loop', None) or
                  getattr(self._o, '_internal_loop', None))
            cl = getattr(il, '_commitment_layer', None)
            if cl:
                state['consistency'] = cl.consistency_score()
                state['dominant_orientation'] = cl.get_dominant_orientation()
                logger.debug(f"[Introspection] consistency={state['consistency']:.2f} orient={state['dominant_orientation']}")
        except Exception as _e:
            logger.debug(f"[Introspection] commitment read error: {_e}")

        # Dominant pressure — confirmed method: pressure_system.dominant_drive() → str
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ps = getattr(ai_sys, 'pressure_system', None)
            if ps:
                if hasattr(ps, 'dominant_drive'):
                    dom_name = ps.dominant_drive()
                    state['dominant_pressure'] = dom_name
                    # Get the value of that pressure
                    if hasattr(ps, 'pressure'):
                        state['pressure_value'] = ps.pressure(dom_name)
                    logger.debug(f"[Introspection] pressure={dom_name} val={state.get('pressure_value',0):.2f}")
                elif hasattr(ps, 'dominant'):
                    dom = ps.dominant()
                    state['dominant_pressure'] = getattr(dom, 'name', str(dom))
                    state['pressure_value'] = getattr(dom, 'effective', 0.5)
        except Exception as _e:
            logger.debug(f"[Introspection] pressure read error: {_e}")

        # Recent contradiction
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ch = getattr(ai_sys, 'liberty_contradiction', None)
            if ch and ch.contradictions:
                recent = max(ch.contradictions, key=lambda c: c.timestamp)
                age = time.time() - recent.timestamp
                if age < 1800:   # within last 30 min
                    state['recent_contradiction'] = recent.actual_behavior
                    state['contradiction_age_min'] = int(age / 60)
        except Exception:
            pass

        # Curiosity drive level — CuriosityEngine stores as self._global
        try:
            curiosity = (getattr(self._o, 'curiosity_engine', None) or
                         getattr(self._o, 'curiosity', None))
            if curiosity:
                # Confirmed attribute: CuriosityEngine._global = global_curiosity level
                level = (getattr(curiosity, '_global', None) or
                         getattr(curiosity, 'global_curiosity', None) or
                         getattr(curiosity, 'drive', None))
                if level is not None:
                    state['curiosity_level'] = float(level)
                    logger.debug(f"[Introspection] curiosity={state['curiosity_level']:.3f}")
        except Exception as _e:
            logger.debug(f"[Introspection] curiosity read error: {_e}")

        # GEI (goal activity)
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ge = getattr(ai_sys, 'goal_engine', None)
            if ge:
                state['gei'] = ge.compute_gei()
        except Exception:
            pass

        return state

    # ── Question derivation ───────────────────────────────────────────────────

    def _derive_question(self, state: Dict, topic: str, frame: int) -> str:
        """
        Derive the most grounded question possible given current state.
        Returns a complete LLM prompt string.
        """
        topic_clean = topic.replace('_', ' ')

        # ── Signal 1: Recent contradiction (most grounded) ────────────────
        # Cooldown: only use contradiction source once per 20 min per contradiction
        # so other signals (IDX, pressure) also get a turn
        if 'recent_contradiction' in state:
            contradiction = state['recent_contradiction']
            age_min = state.get('contradiction_age_min', 0)
            contrad_key = contradiction[:40]
            last_used = getattr(self, '_contradiction_used', {}).get(contrad_key, 0)
            if time.time() - last_used > 1200:  # 20 min cooldown per contradiction
                if not hasattr(self, '_contradiction_used'):
                    self._contradiction_used = {}
                self._contradiction_used[contrad_key] = time.time()
                state['source'] = 'contradiction'
                return (
                    f"You are reflecting on an internal contradiction detected "
                    f"{age_min} minutes ago: {contradiction}\n\n"
                    f"In one sentence, what does this contradiction reveal about "
                    f"your actual orientation toward '{topic_clean}'? "
                    f"Be specific about the tension, not reassuring."
                )
            # else: fall through to IDX/pressure signals

        # ── Signal 2: High IDX + low consistency ─────────────────────────
        idx = state.get('idx', 0.0)
        consistency = state.get('consistency', 1.0)
        dominant = state.get('dominant_orientation', 'unknown')

        if idx > 0.20 and consistency < 0.75:
            state['source'] = 'idx_drift'
            return (
                f"Your identity drift index is {idx:.2f} and your recent "
                f"decisions show {consistency:.0%} consistency, with a "
                f"tendency toward '{dominant}'.\n\n"
                f"In one sentence, what pattern in '{topic_clean}' might be "
                f"causing you to drift away from your stated orientation? "
                f"Name a specific mechanism, not a general principle."
            )

        # ── Signal 3: IDX high but consistency OK (gradual drift) ─────────
        if idx >= 0.26:
            state['source'] = 'idx_high'
            return (
                f"Your identity drift index is {idx:.2f} — above the stable "
                f"threshold. Your dominant orientation is '{dominant}'.\n\n"
                f"In one sentence, what assumption about '{topic_clean}' are "
                f"you making that might not be holding up under pressure?"
            )

        # ── Signal 4: Epistemic stall (curiosity high, GEI low) ──────────
        curiosity = state.get('curiosity_level', 0.0)
        gei = state.get('gei', 0.5)
        if curiosity > 0.60 and gei < 0.45:
            state['source'] = 'epistemic_stall'
            return (
                f"Your curiosity drive is high ({curiosity:.2f}) but goal "
                f"action rate is low ({gei:.2f}) — you're curious but not "
                f"making progress.\n\n"
                f"In one sentence, what about '{topic_clean}' are you "
                f"circling without actually engaging? What would it mean to "
                f"actually commit to understanding it?"
            )

        # ── Signal 5: Dominant pressure not matched to topic ─────────────
        pressure = state.get('dominant_pressure')
        pressure_val = state.get('pressure_value', 0.0)
        if pressure and pressure_val > 0.45:
            state['source'] = 'pressure_mismatch'
            return (
                f"Your dominant internal pressure right now is '{pressure}' "
                f"({pressure_val:.2f}). You are focused on '{topic_clean}'.\n\n"
                f"In one sentence, how does your {pressure} pressure shape "
                f"the way you're approaching '{topic_clean}'? Is it helping "
                f"or distorting your thinking?"
            )

        # ── Fallback: rotating frames (still topic-specific, not generic) ─
        frames = [
            f"In one sentence, what does '{topic_clean}' mean for my sense of self?",
            f"In one sentence, what belief of mine is most challenged by '{topic_clean}'?",
            f"In one sentence, how does '{topic_clean}' connect to what I value most?",
            f"In one sentence, what would fully committing to '{topic_clean}' require of me?",
            f"In one sentence, what is the most interesting thing to understand about '{topic_clean}'?",
        ]
        state['source'] = 'fallback'
        return frames[frame % len(frames)]


# Module-level singleton getter — called from GAE
_instance = None

def get_generator(organism: Any) -> 'GroundedIntrospectionGenerator':
    """Get or create the singleton generator wired to the organism."""
    global _instance
    if _instance is None or getattr(_instance, '_o', None) is not organism:
        _instance = GroundedIntrospectionGenerator(organism)
    return _instance
