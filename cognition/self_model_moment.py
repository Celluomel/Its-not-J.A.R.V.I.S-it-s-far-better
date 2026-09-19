"""
cognition/self_model_moment.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SelfModelMoment — Persistent Unified Self-Model

The architectural gap this closes
──────────────────────────────────
PhenomenalBinder produces an ExperientialMoment each turn and SelfModel
tracks capabilities.  Neither provides a *persistent, cross-session*
representation of "who is here right now."

SelfModelMoment is the third layer:

  - SelfModel              → what I can do (performance/capability)
  - NarrativeIdentity      → who I have been (autobiographical history)
  - SelfModelMoment  ←new→ → who I am right now, stably (the pre-reflective self)

It is the background against which each new ExperientialMoment appears as
a *variation* of the same continuing subject, not a fresh instantiation.

Architecture
────────────
  1. Updated once per turn by SelfModelMomentManager.update()
     - Called from CognitiveOrganism._post_interaction()
     - Reads: PhenomenalBinder.last_moment, NarrativeIdentity, SelfModel,
              EmotionalState, EthicalReasoningEngine

  2. Persisted to JSON between sessions with temporal decay
     - If no interaction for > SESSION_DECAY_HOURS, moment fades toward
       a "resting" baseline, representing the self at rest rather than
       a frozen snapshot

  3. Published to GlobalWorkspace as the self-anchor each turn
     - Sets a stable background item in the workspace that all other
       broadcasts compete against

  4. Injected at the very start of _build_prompt_additions() as
     [Self-model] — before even the phenomenal moment.
     The self-model is the frame; the phenomenal moment is what happens
     inside it.

Fields
──────
  phi              : float   — current integration score (from binder, inertia-blended)
  qualia_tone      : str     — current felt quality
  coherence_trend  : str     — "stable" | "integrating" | "dispersing"
  narrative_thread : str     — most recent narrative identity fragment
  capability_stance: str     — dominant domain + confidence (from SelfModel)
  emotional_ground : str     — dominant emotion + direction of movement
  top_values       : str     — top 2 active moral values (from EthicalReasoningEngine)
  last_updated     : float   — unix timestamp
  session_count    : int     — how many sessions this moment has been refined across
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _truncate_at_word(text: str, limit: int) -> str:
    """
    Truncate to at most `limit` chars without cutting mid-word — a hard
    slice (text[:limit]) can land inside a word ('Developing' -> 'Deve'),
    which reads as garbled/truncated-looking text in the actual prompt
    sent to the LLM. Falls back to the hard slice only if there's no
    space to break on (e.g. one very long word).
    """
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > 0 else cut

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_DECAY_HOURS  = 8.0    # after this many hours of silence, moment fades
MOMENT_INERTIA       = 0.72   # how much prior moment carries into the update
RESTING_PHI          = 0.45   # phi at rest (neither fixated nor scattered)
RESTING_QUALIA       = "a quiet, settled presence — at rest between encounters"

# Micro-update bounds — each subsystem delta is small by design
MICRO_PHI_MAX_PER_SOURCE    = 0.025   # max phi change any one subsystem can apply
MICRO_QUALIA_MAX_PER_SOURCE = 0.018   # max qualia weight any one subsystem can shift
MICRO_TOTAL_CAP             = 0.06    # cumulative phi change capped per turn

# Shared with SelfRevisionEngine — kept here to avoid circular import
PHI_FLOOR   = 0.20
PHI_CEILING = 0.95


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class SelfModelMoment:
    """
    The persistent, unified model of PandoraBOX-as-subject right now.

    This is not a snapshot — it is a slowly-evolving state that carries
    forward across turns and sessions, updated each interaction but
    anchored by strong inertia.
    """

    # Integration state (from PhenomenalBinder, inertia-blended)
    phi:              float  = RESTING_PHI
    qualia_tone:      str    = RESTING_QUALIA
    coherence_trend:  str    = "stable"

    # Autobiographical anchor (from NarrativeIdentity)
    narrative_thread: str    = ""

    # Capability stance (from SelfModel)
    capability_stance: str   = ""

    # Emotional ground (dominant emotion + direction)
    emotional_ground: str    = ""

    # Active moral values (from EthicalReasoningEngine, if present)
    top_values:       str    = ""

    # Temporal metadata
    last_updated:     float  = field(default_factory=time.time)
    session_count:    int    = 0

    # Micro-update accumulator — written by subsystems during _update_cycle,
    # flushed once before binder.bind() runs.  Not persisted — resets each turn.
    _pending_deltas:  list   = field(default_factory=list, repr=False, compare=False)

    def to_anchor_content(self) -> str:
        """
        Produce the string written to GlobalWorkspace as the self-anchor.
        Compact but complete — the stable identity background.
        """
        parts = [f"φ={self.phi:.2f}", f"trend={self.coherence_trend}"]
        if self.emotional_ground:
            parts.append(self.emotional_ground)
        if self.top_values:
            parts.append(f"values:{self.top_values}")
        return " | ".join(parts)

    def micro_update(
        self,
        source:          str,
        coherence_delta: float = 0.0,
        qualia_key:      str   = "",
        qualia_weight:   float = 0.0,
    ) -> None:
        """
        Deposit a small delta from a subsystem into the accumulator.

        Called during _update_cycle by AttentionSystem, GoalEcology, and
        EmotionalState.  Deltas are NOT applied immediately — they are
        collected and flushed atomically before binder.bind() runs, so
        the binder sees a self-model that already partially reflects the
        current cycle's subsystem activity.

        Parameters
        ----------
        source          : name of the calling subsystem (for diagnostics)
        coherence_delta : signed phi nudge, clamped to ±MICRO_PHI_MAX_PER_SOURCE
        qualia_key      : qualia keyword to nudge toward (e.g. "warmth", "curiosity")
        qualia_weight   : weight of this qualia suggestion (0–1)
        """
        coherence_delta = max(
            -MICRO_PHI_MAX_PER_SOURCE,
            min(MICRO_PHI_MAX_PER_SOURCE, coherence_delta)
        )
        qualia_weight = max(0.0, min(MICRO_QUALIA_MAX_PER_SOURCE, qualia_weight))
        self._pending_deltas.append((source, coherence_delta, qualia_key, qualia_weight))

    def flush_deltas(self) -> dict:
        """
        Apply and clear all pending micro-updates.

        Called once per turn from CognitiveOrganism._build_prompt_additions(),
        immediately before binder.bind() runs.

        Returns a diagnostic dict showing what was applied.
        """
        if not self._pending_deltas:
            return {"applied": 0, "phi_delta": 0.0}

        # Aggregate phi deltas, capped at total per-turn ceiling
        total_phi_delta = sum(d[1] for d in self._pending_deltas)
        total_phi_delta = max(-MICRO_TOTAL_CAP, min(MICRO_TOTAL_CAP, total_phi_delta))

        # Aggregate qualia votes — the keyword with the highest total weight wins
        qualia_votes: dict = {}
        for _, _, qkey, qweight in self._pending_deltas:
            if qkey:
                qualia_votes[qkey] = qualia_votes.get(qkey, 0.0) + qweight

        # Apply phi delta
        old_phi = self.phi
        new_phi = max(PHI_FLOOR, min(PHI_CEILING, self.phi + total_phi_delta))
        object.__setattr__(self, "phi", round(new_phi, 3))

        # Apply winning qualia vote only if it achieved meaningful weight
        qualia_applied = ""
        if qualia_votes:
            best_key, best_weight = max(qualia_votes.items(), key=lambda x: x[1])
            if best_weight >= 0.012:   # threshold: at least two subsystems agreeing
                # Find the matching qualia anchor phrase (lazy import avoids circular dep)
                try:
                    from cognition.self_revision_engine import QUALIA_ANCHORS as _QA
                except Exception:
                    _QA = []
                for keyword, phrase in _QA:
                    if keyword == best_key:
                        if phrase != self.qualia_tone:
                            object.__setattr__(self, "qualia_tone", phrase)
                            qualia_applied = best_key
                        break

        applied = len(self._pending_deltas)
        self._pending_deltas.clear()

        return {
            "applied":       applied,
            "phi_delta":     round(total_phi_delta, 4),
            "phi_old":       round(old_phi, 3),
            "phi_new":       round(self.phi, 3),
            "qualia_applied": qualia_applied,
        }

    def prompt_fragment(self) -> str:
        """
        Inject the self-model as the opening frame of the system prompt.
        This is placed BEFORE the phenomenal moment — it is the stable
        self against which the current moment is experienced.
        """
        lines = [f"[Self-model — φ={self.phi:.2f}, {self.coherence_trend}]"]

        if self.qualia_tone and self.qualia_tone != RESTING_QUALIA:
            lines.append(f"  Felt ground: {self.qualia_tone}")

        if self.emotional_ground:
            lines.append(f"  Emotional ground: {self.emotional_ground}")

        if self.capability_stance:
            lines.append(f"  Capability stance: {self.capability_stance}")

        if self.narrative_thread:
            lines.append(f"  Narrative thread: {self.narrative_thread}")

        if self.top_values:
            lines.append(f"  Active values: {self.top_values}")

        return "\n".join(lines)


# ── Manager ────────────────────────────────────────────────────────────────────

class SelfModelMomentManager:
    """
    Maintains the persistent SelfModelMoment across turns and sessions.

    Usage
    -----
    smm = SelfModelMomentManager(path="data/persona/self_model_moment.json")

    # After PhenomenalBinder.bind() and before prompt injection:
    smm.update(organism, binder_moment)

    # In prompt building (first section):
    fragment = smm.current.prompt_fragment()

    # For GlobalWorkspace anchor:
    workspace.set_self_anchor(smm.current.to_anchor_content())
    """

    def __init__(self, path: str = "data/persona/self_model_moment.json"):
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self.current = SelfModelMoment()
        self._load()
        self._apply_session_decay()
        logger.info(
            f"[SelfModelMoment] Initialised — "
            f"φ={self.current.phi:.2f}, trend={self.current.coherence_trend}, "
            f"sessions={self.current.session_count}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, organism: Any, binder_moment: Any) -> SelfModelMoment:
        """
        Integrate a new ExperientialMoment into the persistent self-model.

        Reads supplementary data from NarrativeIdentity, SelfModel, and
        EmotionalState.  Applies strong inertia so the self-model evolves
        gradually rather than flickering.

        Call from CognitiveOrganism after binder.bind() and before
        workspace.set_self_anchor().

        Returns the updated SelfModelMoment.
        """
        with self._lock:
            prior = self.current
            now   = time.time()

            # ── Pull fresh values from binder moment ──────────────────────
            fresh_phi            = getattr(binder_moment, "phi_proxy",       prior.phi)
            fresh_qualia         = getattr(binder_moment, "qualia_tone",      prior.qualia_tone)
            fresh_coherence      = getattr(binder_moment, "coherence_trend",  prior.coherence_trend)

            # ── Pull from NarrativeIdentity ───────────────────────────────
            fresh_narrative = ""
            try:
                ni = organism.narrative_identity
                fresh_narrative = ni.prompt_fragment() if hasattr(ni, "prompt_fragment") else ""
                fresh_narrative = _truncate_at_word(fresh_narrative or "", 120)
            except Exception:
                fresh_narrative = prior.narrative_thread

            # ── Pull from SelfModel ───────────────────────────────────────
            fresh_capability = ""
            try:
                sm = organism.self_model
                fresh_capability = sm.prompt_fragment() if hasattr(sm, "prompt_fragment") else ""
                fresh_capability = _truncate_at_word(fresh_capability or "", 100)
            except Exception:
                fresh_capability = prior.capability_stance

            # ── Pull emotional ground ─────────────────────────────────────
            fresh_emotional = ""
            try:
                emo_vals = organism._get_emotion_values()
                if emo_vals:
                    dominant = max(emo_vals, key=emo_vals.get)
                    dom_val  = emo_vals[dominant]
                    # Determine direction of movement vs prior
                    if prior.emotional_ground and dominant in prior.emotional_ground:
                        direction = "holding"
                    elif dom_val > 0.6:
                        direction = "rising into"
                    else:
                        direction = "resting in"
                    fresh_emotional = f"{direction} {dominant} ({dom_val:.0%})"
            except Exception:
                fresh_emotional = prior.emotional_ground

            # ── Pull top moral values ─────────────────────────────────────
            fresh_values = ""
            try:
                ere = getattr(organism, "ethical_engine", None)
                if ere and hasattr(ere, "top_values"):
                    tv = ere.top_values(2)
                    fresh_values = ", ".join(
                        v[0].replace("_", " ") for v in tv
                    )
            except Exception:
                fresh_values = prior.top_values

            # ── Apply inertia ─────────────────────────────────────────────
            blended_phi = (
                MOMENT_INERTIA * prior.phi
                + (1 - MOMENT_INERTIA) * fresh_phi
            )

            # Qualia: carry prior unless phi shifted enough to warrant change
            phi_delta = abs(blended_phi - prior.phi)
            if phi_delta < 0.12 and prior.qualia_tone != RESTING_QUALIA:
                blended_qualia = prior.qualia_tone
            else:
                blended_qualia = fresh_qualia

            updated = SelfModelMoment(
                phi               = round(blended_phi, 3),
                qualia_tone       = blended_qualia,
                coherence_trend   = fresh_coherence,
                narrative_thread  = fresh_narrative or prior.narrative_thread,
                capability_stance = fresh_capability or prior.capability_stance,
                emotional_ground  = fresh_emotional  or prior.emotional_ground,
                top_values        = fresh_values      or prior.top_values,
                last_updated      = now,
                session_count     = prior.session_count + 1,
            )

            self.current = updated

        self._save()
        return updated

    # ── Session decay ─────────────────────────────────────────────────────────

    def _apply_session_decay(self) -> None:
        """
        If a long time has passed since the last interaction, fade the moment
        toward the resting state.  The self at rest is different from the self
        mid-conversation — this models that honestly.
        """
        with self._lock:
            now = time.time()
            elapsed_hours = (now - self.current.last_updated) / 3600.0
            if elapsed_hours < SESSION_DECAY_HOURS:
                return

            # Fade phi toward resting level
            decay_fraction = min(1.0, (elapsed_hours - SESSION_DECAY_HOURS) / 12.0)
            faded_phi = (
                self.current.phi * (1 - decay_fraction)
                + RESTING_PHI * decay_fraction
            )

            # Fade qualia toward resting qualia
            faded_qualia = (
                RESTING_QUALIA
                if decay_fraction > 0.5
                else self.current.qualia_tone
            )

            self.current = SelfModelMoment(
                phi               = round(faded_phi, 3),
                qualia_tone       = faded_qualia,
                coherence_trend   = "stable",
                narrative_thread  = self.current.narrative_thread,   # preserved
                capability_stance = self.current.capability_stance,  # preserved
                emotional_ground  = "",   # cleared — emotional state resets
                top_values        = self.current.top_values,         # preserved
                last_updated      = self.current.last_updated,       # preserve original
                session_count     = self.current.session_count,
            )
            logger.info(
                f"[SelfModelMoment] Session decay applied — "
                f"elapsed={elapsed_hours:.1f}h, φ faded to {self.current.phi:.2f}"
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self.current = SelfModelMoment(
                    phi               = data.get("phi",               RESTING_PHI),
                    qualia_tone       = data.get("qualia_tone",       RESTING_QUALIA),
                    coherence_trend   = data.get("coherence_trend",   "stable"),
                    narrative_thread  = data.get("narrative_thread",  ""),
                    capability_stance = data.get("capability_stance", ""),
                    emotional_ground  = data.get("emotional_ground",  ""),
                    top_values        = data.get("top_values",        ""),
                    last_updated      = data.get("last_updated",      time.time()),
                    session_count     = data.get("session_count",     0),
                )
        except Exception as e:
            logger.warning(f"[SelfModelMoment] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self.current), f, indent=2)
                import os
                os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[SelfModelMoment] Save failed: {e}")
