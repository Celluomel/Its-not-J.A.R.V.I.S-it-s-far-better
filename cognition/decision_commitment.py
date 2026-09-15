"""
Decision Commitment Layer
=========================
The missing wire between workspace competition and identity consistency.

The analysis is correct: workspace winners were stored but never compared
against previous decisions. Nothing enforced temporal consistency. This
module closes that loop.

Architecture:
  1. RECORD    — workspace winner → commitment entry with orientation type
  2. COMPARE   — new commitment vs recent history → detect inconsistency
  3. PENALISE  — inconsistency → pressure.record_contradiction() + coherence hit
  4. CONTROL   — IDX feeds back into goal weights (high drift → dampen novelty)

This does NOT suppress LLM output — it changes the INPUTS (pressures, goal
weights) so the cognitive system genuinely drifts less, rather than just
labelling the drift after the fact.
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Commitment:
    """A single recorded cognitive commitment."""
    timestamp:    float
    topic:        str           # workspace winner name/topic
    win_type:     str           # 'goal' | 'thought' | 'emotion' etc.
    score:        float         # winning score
    orientation:  str           # derived stable orientation: 'exploration' | 'stability' | 'social' | 'epistemic' | 'other'
    energy:       float         # system energy level at commit time
    cycle:        int           # slow-cycle counter


# Map topic keywords → stable orientation label
_ORIENTATION_MAP = {
    'curiosity':    'exploration',
    'explore':      'exploration',
    'learn':        'exploration',
    'discover':     'exploration',
    'research':     'exploration',
    'understand':   'epistemic',
    'knowledge':    'epistemic',
    'uncertain':    'epistemic',
    'reconcile':    'epistemic',
    'identity':     'stability',
    'coherence':    'stability',
    'belief':       'stability',
    'resolve':      'stability',
    'tension':      'stability',
    'relational':   'social',
    'connect':      'social',
    'social':       'social',
    'user':         'social',
    'warmth':       'social',
}


def _classify_orientation(topic: str) -> str:
    """Classify a topic into a stable orientation label."""
    t = topic.lower().replace('_', ' ')
    for kw, label in _ORIENTATION_MAP.items():
        if kw in t:
            return label
    return 'other'


class DecisionCommitmentLayer:
    """
    Records and enforces cognitive commitments.

    Instantiated once by InternalLoop, wired to the organism.
    tick() called after each workspace competition result.
    """

    WINDOW          = 10   # how many recent commitments to consider
    CONSISTENCY_THR = 0.45 # below this → inconsistency penalty fires
    PENALTY_COOLDOWN = 300  # seconds between penalties for same topic
    MAX_HISTORY     = 200  # on-disk cap

    def __init__(self, organism: Any, data_dir: str = "data/persona"):
        self._o           = organism
        self._path        = Path(data_dir) / "commitments.json"
        self._lock        = threading.Lock()
        self._history: List[Commitment] = []
        self._last_penalty: Dict[str, float] = {}   # orientation → last penalty ts
        self._cycle       = 0
        self._load()
        logger.info(f"[CommitmentLayer] Initialised — {len(self._history)} prior commitments")

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, winner: Dict, cycle: int) -> None:
        """
        Called after each workspace competition.
        winner: dict with keys 'name', 'type', 'score'
        """
        if not winner:
            return

        self._cycle = cycle
        # Fix: 'name' is a slug (needed by _classify_orientation's keyword
        # matching against prefixes like "curiosity_"/"identity_") — but using
        # it as the STORED topic produced cryptic entries like
        # "curiosity_wonder_dont" in commitments.json. 'label' (if present,
        # from workspace_competition's human-readable field) is used for the
        # stored/displayed topic instead; falls back to the slug only if no
        # label exists yet (older candidates / not-yet-repackaged sources).
        _slug  = winner.get('name', '')
        topic  = winner.get('label', _slug) or _slug
        wtype  = winner.get('type', 'unknown')
        score  = winner.get('score', 0.5)
        orient = _classify_orientation(_slug)

        # Fix: CognitiveEnergy exposes level() as a METHOD returning 0-100,
        # not a 'current' attribute. hasattr(energy, 'current') was always
        # False, so every commitment recorded a hardcoded 0.8.
        energy = 0.8
        try:
            energy_obj = getattr(self._o, 'energy', None)
            if energy_obj and hasattr(energy_obj, 'level'):
                energy = energy_obj.level() / 100.0
            energy = round(max(0.0, min(1.0, energy)), 3)
        except Exception:
            energy = 0.8

        commitment = Commitment(
            timestamp   = time.time(),
            topic       = topic[:60],
            win_type    = wtype,
            score       = round(score, 3),
            orientation = orient,
            energy      = round(float(energy), 2),
            cycle       = cycle,
        )

        with self._lock:
            self._history.append(commitment)
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]

        # Only evaluate consistency for goal-type winners (not transient thoughts)
        if wtype == 'goal':
            self._evaluate_consistency(commitment)

        # Apply IDX feedback — high drift damps novelty-seeking
        self._apply_idx_feedback()

        # Persist every 5 cycles (was 10) — halves data-loss window on restart.
        if cycle % 5 == 0:
            self._save()

    def get_dominant_orientation(self, window: int = 20) -> str:
        """Return the most common orientation in the last N commitments."""
        with self._lock:
            recent = [c for c in self._history[-window:] if c.win_type == 'goal']
        if not recent:
            return 'other'
        counts: Dict[str, int] = {}
        for c in recent:
            counts[c.orientation] = counts.get(c.orientation, 0) + 1
        return max(counts, key=counts.get)

    def consistency_score(self, window: int = None) -> float:
        """
        Fraction of recent goal-commitments matching the dominant orientation.
        1.0 = perfectly consistent, 0.0 = total drift.
        """
        w = window or self.WINDOW
        with self._lock:
            recent = [c for c in self._history[-w:] if c.win_type == 'goal']
        if len(recent) < 3:
            return 1.0  # not enough data — neutral
        dominant = self.get_dominant_orientation(w)
        matching = sum(1 for c in recent if c.orientation == dominant)
        return round(matching / len(recent), 3)

    def summary(self) -> Dict:
        """Dashboard-friendly summary."""
        with self._lock:
            total = len(self._history)
        score  = self.consistency_score()
        dom    = self.get_dominant_orientation()
        return {
            'total_commitments': total,
            'consistency_score': score,
            'dominant_orientation': dom,
            'window': self.WINDOW,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate_consistency(self, new_commit: Commitment) -> None:
        """
        Compare new commitment against recent history.
        If orientation shifts abruptly, fire penalty.
        """
        with self._lock:
            goal_history = [c for c in self._history[-(self.WINDOW+1):-1]
                            if c.win_type == 'goal']

        if len(goal_history) < 3:
            return  # not enough history to judge

        dominant = self.get_dominant_orientation(self.WINDOW)
        score    = self.consistency_score(self.WINDOW)

        if (score < self.CONSISTENCY_THR
                and new_commit.orientation != dominant
                and new_commit.orientation != 'other'):

            # Rate-limit penalty per orientation to avoid punishment flood
            now = time.time()
            last = self._last_penalty.get(new_commit.orientation, 0)
            if (now - last) < self.PENALTY_COOLDOWN:
                return

            self._last_penalty[new_commit.orientation] = now
            self._fire_penalty(new_commit, dominant, score)

    def _fire_penalty(self, commit: Commitment, dominant: str, score: float) -> None:
        """
        Apply consistency penalty to the pressure system and coherence.
        Uses existing infrastructure — no new moving parts.
        """
        try:
            # 1. Boost coherence + identity pressure (existing method)
            ai_sys = getattr(self._o, 'ai_system', None)
            pressure = getattr(ai_sys, 'pressure_system', None)
            if pressure and hasattr(pressure, 'record_contradiction'):
                pressure.record_contradiction()
                logger.info(
                    f"[CommitmentLayer] ⚠️  Consistency penalty: "
                    f"'{commit.orientation}' vs dominant '{dominant}' "
                    f"(score={score:.2f}) → coherence+identity pressure boosted"
                )

            # 2. Reduce self-concept coherence slightly (using existing mark_scs)
            organism = self._o
            sc = None
            if ai_sys:
                sc = getattr(ai_sys, 'self_concept', None)
            if sc and hasattr(sc, 'mark_scs_coherence'):
                current = getattr(sc, 'coherence', 0.6)
                sc.mark_scs_coherence(max(0.1, current - 0.04))

            # 3. Log to contradiction handler so it's visible in the audit trail
            ch = getattr(ai_sys, 'liberty_contradiction', None)
            if ch and hasattr(ch, 'contradictions'):
                from cognition.contradiction_handler import Contradiction
                ch.contradictions.append(Contradiction(
                    contradiction_id        = f"commit_{int(time.time()*1000)}",
                    timestamp               = time.time(),
                    claimed_belief          = f"I am oriented toward {dominant}",
                    actual_behavior         = f"workspace chose {commit.orientation} ({commit.topic[:40]})",
                    discrepancy_score       = round(1.0 - score, 2),
                    detected_at_interaction = self._cycle,
                ))

        except Exception as e:
            logger.debug(f"[CommitmentLayer] penalty error: {e}")

    def _apply_idx_feedback(self) -> None:
        """
        IDX → goal weight feedback.

        When identity drift is high (IDX > 0.40), increase priority of
        'stability' orientation goals and dampen 'exploration' goals slightly.
        This makes the cognitive system BEHAVE differently, not just measure drift.

        Called every tick but gate-checked so it only fires every 5 cycles.
        """
        if self._cycle % 5 != 0:
            return
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            obs    = getattr(self._o, 'observatory', None)
            ge     = getattr(ai_sys, 'goal_engine', None)
            if not obs or not ge:
                return

            snap = getattr(obs, '_last_snapshot', None)
            if not snap:
                return
            idx = getattr(snap, 'idx', 0.0)

            if idx < 0.30:
                return  # drift acceptable — no adjustment needed

            # Scale adjustment: IDX 0.30→0.60 maps to 0%→15% priority shift
            adjustment = min(0.15, (idx - 0.30) * 0.5)

            active_goals = [g for g in ge._goals.values() if g.status == 'active']
            adjusted = 0
            for goal in active_goals:
                orient = _classify_orientation(goal.topic)
                if orient == 'stability' and idx > 0.35:
                    # Boost stability-oriented goals when drifting
                    goal.priority = min(0.95, goal.priority + adjustment * 0.5)
                    adjusted += 1
                elif orient == 'exploration' and idx > 0.45:
                    # Slightly dampen pure exploration when heavily drifting
                    goal.priority = max(0.20, goal.priority - adjustment * 0.3)
                    adjusted += 1

            if adjusted:
                logger.debug(
                    f"[CommitmentLayer] IDX={idx:.2f} → adjusted {adjusted} "
                    f"goal priorities (±{adjustment:.3f})"
                )

        except Exception as e:
            logger.debug(f"[CommitmentLayer] IDX feedback error: {e}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._history = [Commitment(**c) for c in data.get('history', [])]
        except Exception as e:
            logger.debug(f"[CommitmentLayer] load error: {e}")
            self._history = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    'history': [asdict(c) for c in self._history[-self.MAX_HISTORY:]]
                }
            tmp = self._path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8', newline='') as f:
                json.dump(payload, f, indent=2)
            import os
            os.replace(tmp, self._path)
        except Exception as e:
            logger.debug(f"[CommitmentLayer] save error: {e}")
