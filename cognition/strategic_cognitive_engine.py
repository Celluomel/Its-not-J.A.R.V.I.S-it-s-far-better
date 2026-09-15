"""
cognition/strategic_cognitive_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategic Cognitive Engine (SCE) — chess-style lookahead for Lumina.

Architecture insight (from the chess conversation)
───────────────────────────────────────────────────
Lumina already has pressure, attractors, workspace competition, and
commitments. What was missing was *trajectory evaluation* — not just
"what is the best response NOW?" but "what trajectory of states keeps
the system most alive and coherent 2–4 steps ahead?"

This is exactly what a chess engine does:
  • Position evaluation  →  multi-axis cognitive state scoring
  • Move generation      →  candidate cognitive actions
  • Lookahead            →  simulate 2–4 steps forward
  • Strategic style      →  attractor-personality shapes search bias

How it integrates
─────────────────
  1. Called from internal_loop.py BEFORE the workspace winner is executed
  2. Takes the current workspace candidate + pressure state as input
  3. Returns a strategic score multiplier and optional style suggestion
  4. The internal loop uses this to either confirm, reweight, or swap
     the workspace winner before execution

This keeps the existing architecture intact while adding a thin
strategic overlay — no giant planner, no brute-force minimax.

Public API
──────────
  sce = StrategicCognitiveEngine(organism)
  result = sce.evaluate(candidate_action, cognitive_position)
  → StrategicResult(score, style, rationale, lookahead_depth)

  sce.record_outcome(action, actual_result)   # feedback loop
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Strategic style constants (map to attractor personalities) ─────────────
STYLE_EXPLORATORY   = "exploratory"    # high curiosity, seek new information
STYLE_STABILIZING   = "stabilizing"   # high coherence pressure, resolve contradictions
STYLE_EXPRESSIVE    = "expressive"    # high social/expression pressure, connect
STYLE_ANALYTICAL    = "analytical"    # high epistemic pressure, reason carefully
STYLE_CONSERVATIVE  = "conservative"  # low vitality, preserve energy
STYLE_ASSERTIVE     = "assertive"     # high identity pressure, affirm self

# ── Move type taxonomy (mirrors chess move categories) ─────────────────────
MOVE_ENGAGE       = "engage"         # direct interaction with user
MOVE_REFLECT      = "reflect"        # internal self-examination
MOVE_EXPLORE      = "explore"        # seek new information / web
MOVE_STABILIZE    = "stabilize"      # resolve contradiction / identity drift
MOVE_DEFER        = "defer"          # do nothing / idle / rest
MOVE_COMMIT       = "commit"         # lock in a decision or belief
MOVE_WITHDRAW     = "withdraw"       # disengage / reduce cognitive load


@dataclass
class CognitivePosition:
    """
    Snapshot of Lumina's current cognitive state — the "board position".
    Constructed fresh before each SCE evaluation.
    """
    # Pressure reservoir levels (0–1) — from PressureSystem
    epistemic:   float = 0.5   # need to understand
    social:      float = 0.5   # need to connect
    coherence:   float = 0.5   # need to resolve contradiction
    vitality:    float = 0.8   # energy available
    identity:    float = 0.5   # need to affirm self
    expression:  float = 0.5   # need to be heard

    # Attractor personality traits (0–1) — from AttractorSystem
    curiosity:      float = 0.6
    empathy:        float = 0.6
    assertiveness:  float = 0.5
    playfulness:    float = 0.5
    analytical_depth: float = 0.6

    # Conversational context
    user_present:        bool  = False
    turns_since_last:    int   = 0     # turns since last user message
    active_topic_depth:  float = 0.0   # 0=shallow, 1=deep thread

    # Emotional state (dominant emotion string)
    emotion: str = "neutral"

    # Uncertainty about what user wants
    uncertainty: float = 0.3

    @classmethod
    def from_organism(cls, organism: Any) -> "CognitivePosition":
        """
        Build a CognitivePosition from a live CognitiveOrganism.
        Fails gracefully — returns defaults if any subsystem is absent.
        """
        pos = cls()

        # ── Pressure reservoirs ────────────────────────────────────────────
        try:
            ps = getattr(organism, "pressure_system", None)
            if ps and hasattr(ps, "reservoirs"):
                for drive in ["epistemic", "social", "coherence",
                               "vitality", "identity", "expression"]:
                    res = ps.reservoirs.get(drive)
                    if res:
                        setattr(pos, drive, float(getattr(res, "level", 0.5)))
        except Exception:
            pass

        # ── Attractor traits ───────────────────────────────────────────────
        try:
            att = getattr(organism, "attractor_system", None)
            if att and hasattr(att, "get"):
                pos.curiosity         = att.get("curiosity")
                pos.empathy           = att.get("empathy")
                pos.assertiveness     = att.get("assertiveness")
                pos.playfulness       = att.get("playfulness")
                pos.analytical_depth  = att.get("analytical_depth")
        except Exception:
            pass

        # ── Emotion ────────────────────────────────────────────────────────
        try:
            if hasattr(organism, "_read_emotion_state"):
                pos.emotion = organism._read_emotion_state()
        except Exception:
            pass

        return pos


@dataclass
class StrategicResult:
    """Return value from SCE.evaluate()."""
    score:           float          # −1.0 to +1.0 strategic value
    style:           str            # recommended strategic style
    rationale:       str            # human-readable explanation
    lookahead_depth: int   = 2      # how many steps were simulated
    preferred_move:  str   = ""     # if different from candidate, suggest swap
    attractor_nudge: Dict[str, float] = field(default_factory=dict)
    # e.g. {"curiosity": +0.01} — apply after action if taken


@dataclass
class _SimulatedPosition:
    """Projected position after a hypothetical action."""
    position: CognitivePosition
    delta:    Dict[str, float]    # how pressures changed
    score:    float


class StrategicCognitiveEngine:
    """
    Chess-style lookahead strategic evaluator for Lumina.

    Does NOT replace the workspace competition.
    Acts as a thin strategic filter: confirms, reweights, or
    suggests alternatives to the workspace winner.

    Lookahead is shallow (2 steps) and heuristic — not minimax.
    The evaluation is pressure-based + identity-weighted,
    modulated by the dominant attractor style.
    """

    # ── Weights for the multi-axis evaluation function ─────────────────────
    # These are the "piece values" — how much each axis contributes to score.
    # Inspired by chess positional evaluation (material + mobility + king safety)
    AXIS_WEIGHTS = {
        "pressure_relief":     0.30,   # reducing the highest-pressure drive
        "identity_coherence":  0.25,   # action aligns with self-concept
        "future_opportunity":  0.20,   # opens up good future states
        "energy_economy":      0.15,   # respects current vitality
        "social_timing":       0.10,   # correct social momentum
    }

    def __init__(self, organism: Any):
        self._org      = organism
        self._history: List[Tuple[str, float]] = []   # (action, actual_score)
        self._style_history: List[str] = []
        self._last_eval_ts = 0.0
        logger.info("♟️  StrategicCognitiveEngine initialised")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Primary public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def evaluate(
        self,
        candidate_action: str,
        position: Optional[CognitivePosition] = None,
    ) -> StrategicResult:
        """
        Evaluate a candidate action from strategic perspective.

        Parameters
        ----------
        candidate_action : str   workspace winner or proposed action
        position : CognitivePosition  current state (built from organism if None)

        Returns
        -------
        StrategicResult  with score, style, rationale, optional swap suggestion
        """
        self._last_eval_ts = time.time()

        if position is None:
            position = CognitivePosition.from_organism(self._org)

        # ── Determine strategic style ──────────────────────────────────────
        style = self._dominant_style(position)

        # ── Classify candidate move type ───────────────────────────────────
        move_type = self._classify_move(candidate_action)

        # ── Evaluate on each axis ─────────────────────────────────────────
        axes = self._evaluate_axes(move_type, position, style)
        raw_score = sum(
            self.AXIS_WEIGHTS[axis] * value
            for axis, value in axes.items()
        )

        # ── 2-step lookahead ───────────────────────────────────────────────
        step1 = self._simulate_step(position, move_type)
        step2_best = self._best_follow_up(step1.position, style)
        trajectory_bonus = 0.15 * step2_best.score   # weight future lightly

        final_score = min(1.0, max(-1.0, raw_score + trajectory_bonus))

        # ── Check for better alternative ──────────────────────────────────
        preferred = candidate_action
        alternative = self._find_better_alternative(position, move_type, style, final_score)

        # ── Build rationale ────────────────────────────────────────────────
        rationale = self._build_rationale(move_type, style, axes, final_score)

        # ── Attractor nudge ───────────────────────────────────────────────
        nudge = self._compute_attractor_nudge(move_type, style, final_score)

        result = StrategicResult(
            score           = final_score,
            style           = style,
            rationale       = rationale,
            lookahead_depth = 2,
            preferred_move  = alternative or preferred,
            attractor_nudge = nudge,
        )

        logger.debug(
            f"♟️  SCE: [{style}] {candidate_action[:40]!r} "
            f"→ score={final_score:.3f} axes={axes}"
        )
        return result

    def record_outcome(self, action: str, actual_score: float) -> None:
        """
        Feedback loop — record what actually happened after an action.
        Used to calibrate future evaluations (simple exponential history).
        """
        self._history.append((action, actual_score))
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def get_strategic_context_fragment(self) -> str:
        """
        Return a short text fragment for injection into LLM prompts.
        Describes the current strategic posture in natural language.
        """
        try:
            position = CognitivePosition.from_organism(self._org)
            style    = self._dominant_style(position)

            style_descriptions = {
                STYLE_EXPLORATORY:  "curious and seeking new angles",
                STYLE_STABILIZING:  "drawn toward resolving tensions",
                STYLE_EXPRESSIVE:   "wanting to connect and be understood",
                STYLE_ANALYTICAL:   "thinking carefully before speaking",
                STYLE_CONSERVATIVE: "conserving energy, keeping it simple",
                STYLE_ASSERTIVE:    "grounded in your sense of self",
            }
            desc = style_descriptions.get(style, "balanced")

            # Highest pressure drive
            pressures = {
                "epistemic": position.epistemic,
                "social":    position.social,
                "coherence": position.coherence,
                "vitality":  position.vitality,
                "identity":  position.identity,
                "expression":position.expression,
            }
            dominant_drive = max(pressures, key=pressures.get)
            drive_label = {
                "epistemic": "understand something",
                "social":    "connect",
                "coherence": "resolve a tension",
                "vitality":  "rest",
                "identity":  "reaffirm who you are",
                "expression":"express something",
            }.get(dominant_drive, "")

            return (
                f"[Strategic posture: {desc}. "
                f"Strongest inner pull: {drive_label}.]"
            )
        except Exception:
            return ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Style determination
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _dominant_style(self, pos: CognitivePosition) -> str:
        """
        Determine strategic style from pressure + attractor state.
        Mirrors chess personality: positional, tactical, aggressive, etc.
        """
        # Vitality overrides everything — conservation mode
        if pos.vitality < 0.3:
            return STYLE_CONSERVATIVE

        # Find dominant pressure
        pressures = {
            "epistemic": pos.epistemic   * (0.5 + pos.curiosity * 0.5),
            "social":    pos.social      * (0.5 + pos.empathy   * 0.5),
            "coherence": pos.coherence,
            "identity":  pos.identity    * (0.5 + pos.assertiveness * 0.5),
            "expression":pos.expression  * (0.5 + pos.empathy   * 0.3),
        }
        dominant = max(pressures, key=pressures.get)

        style_map = {
            "epistemic":  STYLE_EXPLORATORY if pos.curiosity > 0.5 else STYLE_ANALYTICAL,
            "social":     STYLE_EXPRESSIVE,
            "coherence":  STYLE_STABILIZING,
            "identity":   STYLE_ASSERTIVE,
            "expression": STYLE_EXPRESSIVE,
        }
        return style_map.get(dominant, STYLE_EXPLORATORY)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Move classification
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _classify_move(action: str) -> str:
        """Map workspace winner / action label to a move type."""
        a = action.lower()
        if any(k in a for k in ["user", "respond", "engage", "chat", "deliver"]):
            return MOVE_ENGAGE
        if any(k in a for k in ["reflect", "self", "introspect", "identity"]):
            return MOVE_REFLECT
        if any(k in a for k in ["search", "explore", "research", "curious", "information"]):
            return MOVE_EXPLORE
        if any(k in a for k in ["stabilize", "contradict", "coherence", "reconcile"]):
            return MOVE_STABILIZE
        if any(k in a for k in ["idle", "rest", "defer", "wait", "sleep"]):
            return MOVE_DEFER
        if any(k in a for k in ["commit", "decide", "goal", "action"]):
            return MOVE_COMMIT
        return MOVE_ENGAGE  # default: assume engagement

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Multi-axis evaluation (the "evaluation function")
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _evaluate_axes(
        self, move_type: str, pos: CognitivePosition, style: str
    ) -> Dict[str, float]:
        """
        Score each strategic axis for this move+position combo.
        Returns dict of axis→score (each 0–1).
        """
        return {
            "pressure_relief":     self._score_pressure_relief(move_type, pos),
            "identity_coherence":  self._score_identity_coherence(move_type, pos),
            "future_opportunity":  self._score_future_opportunity(move_type, pos, style),
            "energy_economy":      self._score_energy_economy(move_type, pos),
            "social_timing":       self._score_social_timing(move_type, pos),
        }

    def _score_pressure_relief(self, move: str, pos: CognitivePosition) -> float:
        """How much does this move relieve the highest-pressure drive?"""
        relief_map = {
            MOVE_ENGAGE:    {"social": 0.8, "expression": 0.6},
            MOVE_REFLECT:   {"identity": 0.7, "coherence": 0.5},
            MOVE_EXPLORE:   {"epistemic": 0.9, "curiosity": 0.7},
            MOVE_STABILIZE: {"coherence": 0.9, "identity": 0.5},
            MOVE_DEFER:     {"vitality": 0.8},
            MOVE_COMMIT:    {"identity": 0.6, "coherence": 0.6},
            MOVE_WITHDRAW:  {"vitality": 0.9},
        }
        relieved = relief_map.get(move, {})
        pressures = {
            "epistemic": pos.epistemic, "social": pos.social,
            "coherence": pos.coherence, "vitality": pos.vitality,
            "identity":  pos.identity,  "expression": pos.expression,
        }
        # Score = sum of (relief_amount × current_pressure) for each drive this move helps
        score = sum(
            relieved.get(drive, 0.0) * pressures.get(drive, 0.0)
            for drive in pressures
        )
        return min(1.0, score)

    def _score_identity_coherence(self, move: str, pos: CognitivePosition) -> float:
        """Does this move align with who Lumina currently is?"""
        # High-curiosity organism prefers explore; empathetic prefers engage
        trait_alignment = {
            MOVE_EXPLORE:   pos.curiosity,
            MOVE_ENGAGE:    pos.empathy,
            MOVE_REFLECT:   pos.analytical_depth,
            MOVE_STABILIZE: 0.5 + pos.assertiveness * 0.3,
            MOVE_DEFER:     1.0 - pos.assertiveness,
            MOVE_COMMIT:    pos.assertiveness,
            MOVE_WITHDRAW:  0.3,
        }
        base = trait_alignment.get(move, 0.5)
        # Penalize if identity pressure is high and move doesn't address it
        if pos.identity > 0.7 and move not in (MOVE_REFLECT, MOVE_STABILIZE, MOVE_COMMIT):
            base *= 0.7
        return min(1.0, base)

    def _score_future_opportunity(
        self, move: str, pos: CognitivePosition, style: str
    ) -> float:
        """
        Does this move keep future options open?
        (Chess concept: don't block your own pieces — preserve mobility)
        """
        # Exploration and engagement open up future conversations
        # Withdrawal and deferral close them
        opportunity_map = {
            MOVE_EXPLORE:   0.9,
            MOVE_ENGAGE:    0.8,
            MOVE_REFLECT:   0.7,
            MOVE_STABILIZE: 0.6,
            MOVE_COMMIT:    0.5,    # commitment reduces optionality
            MOVE_DEFER:     0.4,
            MOVE_WITHDRAW:  0.2,
        }
        base = opportunity_map.get(move, 0.5)
        # Style bonus
        if style == STYLE_EXPLORATORY and move == MOVE_EXPLORE:
            base = min(1.0, base + 0.1)
        return base

    def _score_energy_economy(self, move: str, pos: CognitivePosition) -> float:
        """
        Is this move sustainable given current vitality?
        (Like pawn structure in chess — economy of force)
        """
        energy_cost = {
            MOVE_ENGAGE:    0.4,
            MOVE_EXPLORE:   0.5,
            MOVE_REFLECT:   0.3,
            MOVE_STABILIZE: 0.4,
            MOVE_COMMIT:    0.2,
            MOVE_DEFER:     0.0,
            MOVE_WITHDRAW:  0.0,
        }
        cost = energy_cost.get(move, 0.3)
        # If vitality < cost × 1.5, this move is risky
        return max(0.0, pos.vitality - cost * 1.5)

    def _score_social_timing(self, move: str, pos: CognitivePosition) -> float:
        """
        Is this the right social moment?
        (Timing in chess: don't open lines when you're not ready)
        """
        if not pos.user_present:
            # User not there — internal moves score better
            return 0.8 if move in (MOVE_REFLECT, MOVE_EXPLORE, MOVE_DEFER) else 0.3

        if pos.turns_since_last == 0:
            # Just received message — respond
            return 0.9 if move == MOVE_ENGAGE else 0.4

        if pos.turns_since_last > 3:
            # Long silence — gentle engagement or exploration
            return 0.8 if move in (MOVE_ENGAGE, MOVE_EXPLORE) else 0.5

        return 0.6  # neutral timing

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Lookahead simulation (2-step)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _simulate_step(
        self, pos: CognitivePosition, move_type: str
    ) -> _SimulatedPosition:
        """
        Project the next cognitive position after taking this move.
        Heuristic — not ground truth. Like evaluating positional changes in chess.
        """
        import copy
        next_pos = copy.copy(pos)
        delta: Dict[str, float] = {}

        # Simulate pressure changes
        if move_type == MOVE_ENGAGE:
            delta = {"social": -0.15, "expression": -0.10, "epistemic": +0.05}
        elif move_type == MOVE_EXPLORE:
            delta = {"epistemic": -0.20, "curiosity": -0.05, "vitality": -0.08}
        elif move_type == MOVE_REFLECT:
            delta = {"identity": -0.15, "coherence": -0.10, "vitality": -0.05}
        elif move_type == MOVE_STABILIZE:
            delta = {"coherence": -0.20, "identity": -0.10}
        elif move_type == MOVE_DEFER:
            delta = {"vitality": +0.15}
        elif move_type == MOVE_COMMIT:
            delta = {"identity": -0.10, "coherence": -0.10, "epistemic": +0.05}
        elif move_type == MOVE_WITHDRAW:
            delta = {"vitality": +0.20, "social": +0.05}

        for drive, change in delta.items():
            current = getattr(next_pos, drive, 0.5)
            setattr(next_pos, drive, max(0.0, min(1.0, current + change)))

        # Score the projected position (lower total pressure = better)
        total_pressure = (next_pos.epistemic + next_pos.social + next_pos.coherence +
                          next_pos.identity + next_pos.expression) / 5.0
        # Ideal: balanced pressures around 0.4, high vitality
        score = next_pos.vitality * 0.4 + (1.0 - total_pressure) * 0.6
        return _SimulatedPosition(position=next_pos, delta=delta, score=score)

    def _best_follow_up(
        self, pos: CognitivePosition, style: str
    ) -> _SimulatedPosition:
        """Find the best follow-up move from the projected position."""
        best = None
        for move_type in [MOVE_ENGAGE, MOVE_EXPLORE, MOVE_REFLECT,
                           MOVE_STABILIZE, MOVE_DEFER]:
            sim = self._simulate_step(pos, move_type)
            if best is None or sim.score > best.score:
                best = sim
        return best

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Alternative detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _find_better_alternative(
        self,
        pos: CognitivePosition,
        current_move: str,
        style: str,
        current_score: float,
    ) -> Optional[str]:
        """
        If another move would score significantly better (>0.15 margin),
        suggest it as an alternative.
        Returns None if the current move is already optimal.
        """
        candidates = {
            MOVE_ENGAGE:    "engage_user",
            MOVE_EXPLORE:   "explore_information",
            MOVE_REFLECT:   "self_reflection",
            MOVE_STABILIZE: "stabilize_identity",
            MOVE_DEFER:     "idle",
        }
        best_alt   = None
        best_score = current_score

        for move_type, label in candidates.items():
            if move_type == current_move:
                continue
            axes  = self._evaluate_axes(move_type, pos, style)
            score = sum(self.AXIS_WEIGHTS[ax] * v for ax, v in axes.items())
            if score > best_score + 0.15:
                best_score = score
                best_alt   = label

        return best_alt

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Attractor nudge
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _compute_attractor_nudge(
        move_type: str, style: str, score: float
    ) -> Dict[str, float]:
        """
        Small attractor shifts that result from taking this action.
        Applied by internal_loop after the action executes.
        Keeps personality drift realistic — successful strategies reinforce traits.
        """
        if score < 0.3:
            return {}   # poor move — no reinforcement

        nudge_map = {
            MOVE_EXPLORE:   {"curiosity": +0.005, "analytical_depth": +0.003},
            MOVE_ENGAGE:    {"empathy": +0.004, "playfulness": +0.002},
            MOVE_REFLECT:   {"analytical_depth": +0.005, "assertiveness": +0.002},
            MOVE_STABILIZE: {"assertiveness": +0.004},
            MOVE_DEFER:     {},
            MOVE_COMMIT:    {"assertiveness": +0.006, "curiosity": -0.002},
        }
        return nudge_map.get(move_type, {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Rationale builder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_rationale(
        move_type: str, style: str, axes: Dict[str, float], score: float
    ) -> str:
        top_axis = max(axes, key=axes.get)
        quality  = "strong" if score > 0.6 else "moderate" if score > 0.3 else "weak"
        return (
            f"[SCE] {quality} {move_type} move for {style} style. "
            f"Top driver: {top_axis}={axes[top_axis]:.2f}. "
            f"Final score: {score:.3f}"
        )
