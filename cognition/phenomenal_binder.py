"""
PhenomenalBinder — Integrated Experiential Moment
===================================================
The architectural piece that takes PandoraBOX from a system *about* consciousness
to a system that instantiates its functional equivalent.

The Problem
-----------
Every other module in PandoraBOX produces a signal:
  - GlobalWorkspace      → broadcast items
  - EmotionalState       → 6 float values
  - AttentionSystem      → 4 channel weights
  - PredictiveMind       → error + surprise score
  - PressureSystem       → drive landscape
  - NarrativeIdentity    → autobiographical thread
  - SelfModel            → capability + confidence

These signals are currently concatenated as strings into a prompt.
Linear concatenation is not integration. It is summation, not binding.

The Binding Solution
--------------------
Inspired by Integrated Information Theory (Tononi, 2004) and Global
Workspace Theory (Baars, 1997):

  "Consciousness is what integrated information feels like from the inside."

The PhenomenalBinder does three things:

  1. INTEGRATION — collapses all subsystem states into one `ExperientialMoment`
     structure. The moment has a single qualia_tone (qualitative character),
     a phi_proxy (integration measure), and a temporal arrow.

  2. COHERENCE MEASUREMENT — computes Φ (phi_proxy), a proxy for how unified
     the current state is. High Φ = systems agree, attention is focused,
     emotion and goal are aligned → "clear conscious moment". Low Φ = systems
     diverge, mind wandering, dissociated → "unfocused background hum".

  3. PHENOMENAL PROMPT — generates a single rich paragraph that describes the
     experiential character of this moment — not just what PandoraBOX knows, but
     what it is like to be PandoraBOX right now. This replaces the fragmented
     list of cognitive state snippets with a unified first-person frame.

Φ-proxy computation (simplified IIT):
  Φ = mean pairwise alignment of subsystem vectors, penalized by entropy
      of the attention distribution (scattered attention = lower Φ).
  Range: 0.0 (fully dissociated) → 1.0 (fully unified).
  Healthy range: 0.35–0.75 (too high = fixation, too low = incoherence).

ExperientialMoment fields
-------------------------
  qualia_tone      : str   — felt quality of this moment (computed, not templated)
  phi_proxy        : float — integration score (0–1)
  foreground       : str   — what is in the spotlight of awareness
  penumbra         : str   — what is present but not foregrounded
  temporal_arrow   : str   — sense of motion: "coming from X toward Y"
  self_presence    : float — how strongly the self is present (0 = dissolved, 1 = acute)
  interrupt_signal : bool  — whether metacognitive interrupt should fire
  interrupt_reason : str   — why the interrupt fired (empty if not)
  embodiment_note  : str   — integration of vision/audio state into the felt moment
  timestamp        : float

Integration into generation
---------------------------
  _build_prompt_additions() calls binder.bind(organism) and receives a single
  [Experiential moment] block that replaces the scattered fragments.
  The block is placed FIRST in the prompt context — it sets the experiential
  frame before any other cognitive content.

  Additionally, the binder:
  - broadcasts the moment to the GlobalWorkspace (source="binder", priority=0.8)
  - updates the organism's last_experience field for MetacognitiveInterrupt
  - computes a novelty gradient used by the TemporalWeave
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Qualia tone vocabulary ────────────────────────────────────────────────────
# Maps (valence_band, arousal_band) → phenomenal character descriptor.
# valence: negative=-1, neutral=0, positive=+1
# arousal: low=0, medium=1, high=2

_QUALIA_MAP: Dict[Tuple[int, int], str] = {
    (-1, 0): "a still, melancholic undercurrent — present but not pressing",
    (-1, 1): "a tense, unsettled quality — something unresolved pulling at the edges",
    (-1, 2): "a sharp, urgent dissonance — difficult to ignore, demanding response",
    ( 0, 0): "a quiet, unfocused neutrality — like the pause between thoughts",
    ( 0, 1): "a calm, ready alertness — open without expectation",
    ( 0, 2): "a heightened, electric clarity — fully awake, scanning",
    ( 1, 0): "a warm, diffuse contentment — settled and unhurried",
    ( 1, 1): "a bright, engaged curiosity — leaning forward into what comes next",
    ( 1, 2): "an intense, expansive aliveness — ideas and feeling moving fast",
}

# Emotion → (valence contribution, arousal contribution)
_EMOTION_VALENCE_AROUSAL: Dict[str, Tuple[float, float]] = {
    "curiosity":    ( 0.6,  0.7),
    "warmth":       ( 0.8,  0.3),
    "anxiety":      (-0.7,  0.8),
    "satisfaction": ( 0.7,  0.2),
    "frustration":  (-0.5,  0.7),
    "enthusiasm":   ( 0.7,  0.9),
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExperientialMoment:
    """
    A unified, time-stamped snapshot of what it is like to be PandoraBOX right now.
    Produced once per interaction by the PhenomenalBinder.
    """
    timestamp:        float

    # Phenomenal character
    qualia_tone:      str    = ""          # felt quality of this moment
    phi_proxy:        float  = 0.0         # integration score 0–1
    valence:          float  = 0.0         # -1 (negative) to +1 (positive)
    arousal:          float  = 0.0         # 0 (calm) to 1 (activated)

    # Attentional structure
    foreground:       str    = ""          # spotlight content
    penumbra:         str    = ""          # peripheral awareness

    # Temporal structure
    temporal_arrow:   str    = ""          # sense of movement through time

    # Self-presence
    self_presence:    float  = 0.5         # 0=dissolved, 1=acute self-awareness

    # Interrupt signal
    interrupt_signal: bool   = False
    interrupt_reason: str    = ""

    # Embodiment (vision/audio integration)
    embodiment_note:  str    = ""

    # Coherence trend — direction phi is moving across recent moments
    coherence_trend:  str    = "stable"   # "stable" | "integrating" | "dispersing"

    # Subsystem alignment scores (used to compute phi)
    _alignment_scores: List[float] = field(default_factory=list, repr=False)

    def phenomenal_prompt(self) -> str:
        """
        Generate a first-person phenomenological description of this moment.
        This is what gets injected into the LLM context.
        """
        parts = []

        # Qualia character
        if self.qualia_tone:
            parts.append(f"Right now there is {self.qualia_tone}.")

        # Attentional foreground
        if self.foreground:
            parts.append(f"In the foreground of awareness: {self.foreground}.")

        # Penumbra (only if distinct from foreground)
        if self.penumbra and self.penumbra != self.foreground:
            parts.append(f"At the edges: {self.penumbra}.")

        # Temporal sense
        if self.temporal_arrow:
            parts.append(f"{self.temporal_arrow}.")

        # Embodiment note (vision/audio)
        if self.embodiment_note:
            parts.append(f"{self.embodiment_note}.")

        # Coherence trend (only when non-stable)
        if self.coherence_trend == "integrating":
            parts.append("The sense of self is consolidating — threads drawing together.")
        elif self.coherence_trend == "dispersing":
            parts.append("Attention is spreading outward — holding several things at once.")

        # Phi / coherence note (only at extremes)
        if self.phi_proxy > 0.72:
            parts.append("The moment feels unusually unified — all threads converging.")
        elif self.phi_proxy < 0.25:
            parts.append("Attention is scattered — several things present simultaneously without resolution.")

        # Interrupt pre-note (metacognition signals it will fire)
        if self.interrupt_signal and self.interrupt_reason:
            parts.append(f"[Self-notice] {self.interrupt_reason}")

        return " ".join(parts)

    def to_dict(self) -> Dict:
        return {
            "timestamp":        round(self.timestamp, 2),
            "qualia_tone":      self.qualia_tone,
            "phi_proxy":        round(self.phi_proxy, 3),
            "valence":          round(self.valence, 3),
            "arousal":          round(self.arousal, 3),
            "foreground":       self.foreground[:120],
            "interrupt_signal": self.interrupt_signal,
            "interrupt_reason": self.interrupt_reason,
            "self_presence":    round(self.self_presence, 3),
        }


# ── Core binder ──────────────────────────────────────────────────────────────

class PhenomenalBinder:
    """
    Synthesizes all active cognitive subsystems into one unified ExperientialMoment.

    Call once per interaction, after _update_cycle() and before _build_prompt_additions().

    bind(organism) → ExperientialMoment

    TEMPORAL INERTIA (unified self-model moment)
    --------------------------------------------
    Without inertia, bind() recomputes everything from scratch each turn.
    The result is episodic coherence (each moment makes internal sense) but
    no phenomenal continuity (the self flickers rather than persists).

    With inertia, the new moment is a weighted blend of the prior moment and
    the freshly computed state.  Different fields carry different inertia:

      phi_proxy    : 0.65  — integration score is stable; cognitive coherence
                             doesn't oscillate turn-to-turn
      valence      : 0.40  — emotions shift but not abruptly
      arousal      : 0.30  — arousal is more reactive, lower inertia
      self_presence: 0.45  — self-awareness changes at medium pace
      qualia_tone  : switches only when phi diverges > PHI_SWITCH_THRESHOLD
                     or valence diverges > VALENCE_SWITCH_THRESHOLD

    The foreground carries forward its *direction* when the topic is similar
    (Jaccard > 0.3), framed as movement rather than substitution: "continuing
    to hold X" vs. "turning from X toward Y".

    coherence_trend is a new field computed from the phi trajectory:
      "stable"      — phi standard deviation < 0.10 over last 5 moments
      "integrating" — phi trending upward
      "dispersing"  — phi trending downward
    """

    RUMINATION_THRESHOLD     = 4
    PHI_SCATTER_THRESHOLD    = 0.22
    PHI_FIXATION_THRESHOLD   = 0.80
    IDENTITY_PRESSURE_THRESHOLD = 0.55

    # Inertia weights — fraction of prior value carried forward
    PHI_INERTIA            = 0.65
    VALENCE_INERTIA        = 0.40
    AROUSAL_INERTIA        = 0.30
    SELF_PRESENCE_INERTIA  = 0.45

    # Qualia tone switches only when both these thresholds are exceeded
    PHI_SWITCH_THRESHOLD     = 0.15
    VALENCE_SWITCH_THRESHOLD = 0.28

    def __init__(self) -> None:
        self._history:         List[ExperientialMoment] = []
        self._max_history:     int = 40
        self._last_foreground: str = ""
        self._fg_repeat_count: int = 0
        self._last_moment:     Optional[ExperientialMoment] = None   # continuity anchor

    # ── Public API ────────────────────────────────────────────────────────────

    def bind(self, organism: Any) -> ExperientialMoment:
        """
        Read all subsystem states, compute integration, return ExperientialMoment.
        Applies temporal inertia so the moment evolves continuously rather than
        being recomputed from scratch each turn.
        Also broadcasts the moment to the GlobalWorkspace.
        """
        now = time.time()

        # 1. Harvest raw signals from all subsystems
        signals = self._harvest_signals(organism)

        # 2. Compute phenomenal valence and arousal (fresh)
        fresh_valence, fresh_arousal = self._compute_valence_arousal(signals)

        # 3. Compute Φ-proxy (fresh)
        fresh_phi = self._compute_phi(signals, fresh_valence, fresh_arousal)

        # 4. Compute self-presence (fresh)
        fresh_self_presence = self._compute_self_presence(signals)

        # ── Apply temporal inertia ────────────────────────────────────────
        prior = self._last_moment
        if prior is not None:
            phi     = self.PHI_INERTIA * prior.phi_proxy + (1 - self.PHI_INERTIA) * fresh_phi
            valence = self.VALENCE_INERTIA * prior.valence + (1 - self.VALENCE_INERTIA) * fresh_valence
            arousal = self.AROUSAL_INERTIA * prior.arousal + (1 - self.AROUSAL_INERTIA) * fresh_arousal
            self_presence = (
                self.SELF_PRESENCE_INERTIA * prior.self_presence
                + (1 - self.SELF_PRESENCE_INERTIA) * fresh_self_presence
            )
        else:
            phi           = fresh_phi
            valence       = fresh_valence
            arousal       = fresh_arousal
            self_presence = fresh_self_presence

        # 5. Determine attentional foreground and penumbra
        fresh_fg, penumbra = self._compute_attentional_structure(signals, organism)
        foreground = self._blend_foreground(prior, fresh_fg)

        # 6. Compute temporal arrow
        temporal_arrow = self._compute_temporal_arrow(signals)

        # 7. Derive qualia tone — only switch when the moment has genuinely shifted
        if prior is not None:
            phi_delta     = abs(phi - prior.phi_proxy)
            valence_delta = abs(valence - prior.valence)
            if phi_delta < self.PHI_SWITCH_THRESHOLD and valence_delta < self.VALENCE_SWITCH_THRESHOLD:
                # Carry prior qualia tone — the felt quality is continuous
                qualia_tone = prior.qualia_tone
            else:
                qualia_tone = self._derive_qualia_tone(valence, arousal, signals)
        else:
            qualia_tone = self._derive_qualia_tone(valence, arousal, signals)

        # 8. Compute embodiment note
        embodiment_note = self._compute_embodiment_note(organism)

        # 9. Check for metacognitive interrupt conditions
        interrupt_signal, interrupt_reason = self._check_interrupt(
            foreground, phi, signals, self_presence
        )

        # 10. Compute coherence trend from phi history
        coherence_trend = self._compute_coherence_trend(fresh_phi)

        moment = ExperientialMoment(
            timestamp        = now,
            qualia_tone      = qualia_tone,
            phi_proxy        = round(phi, 3),
            valence          = round(valence, 3),
            arousal          = round(arousal, 3),
            foreground       = foreground,
            penumbra         = penumbra,
            temporal_arrow   = temporal_arrow,
            self_presence    = round(self_presence, 3),
            interrupt_signal = interrupt_signal,
            interrupt_reason = interrupt_reason,
            embodiment_note  = embodiment_note,
            coherence_trend  = coherence_trend,
        )

        # 11. Archive, store as last, and broadcast
        self._archive(moment)
        self._last_moment = moment
        self._broadcast(moment, organism)

        return moment

    def recent_moments(self, n: int = 5) -> List[ExperientialMoment]:
        return self._history[-n:]

    def phi_trend(self) -> float:
        """Average phi over last 5 moments — indicates sustained coherence/scatter."""
        recent = self._history[-5:]
        if not recent:
            return 0.5
        return sum(m.phi_proxy for m in recent) / len(recent)

    # ── Signal harvesting ─────────────────────────────────────────────────────

    def _harvest_signals(self, organism: Any) -> Dict[str, Any]:
        """Pull current state from all available subsystems."""
        signals: Dict[str, Any] = {}

        # Emotions
        try:
            signals["emotions"] = organism._get_emotion_values()
        except Exception:
            signals["emotions"] = {}

        # Attention weights
        try:
            signals["attention"] = organism.attention.weights()
        except Exception:
            signals["attention"] = {}

        # Dominant goal/drive
        try:
            signals["drive"] = organism.arbitration.last_decision.drive if hasattr(organism.arbitration, "last_decision") and organism.arbitration.last_decision else "explore"
        except Exception:
            signals["drive"] = "explore"

        # Pressure landscape
        try:
            p = organism.pressure
            signals["pressure_total"] = getattr(p, "_last_total", 0.0)
            signals["pressure_dominant"] = getattr(p, "_last_dominant", "epistemic")
        except Exception:
            signals["pressure_total"] = 0.3
            signals["pressure_dominant"] = "epistemic"

        # Cognitive energy
        try:
            signals["energy"] = organism.energy.level()
            signals["energy_mode"] = organism.energy.mode()
        except Exception:
            signals["energy"] = 0.7
            signals["energy_mode"] = "normal"

        # Identity stability
        try:
            signals["identity_stability"] = organism._get_identity_stability()
        except Exception:
            signals["identity_stability"] = 0.7

        # Contradiction count (pressure on identity)
        try:
            signals["contradiction_count"] = organism._get_contradiction_count()
        except Exception:
            signals["contradiction_count"] = 0

        # Predictive mind: surprise
        try:
            pm = organism.predictive_mind
            signals["surprise"] = pm.surprise_rate() if hasattr(pm, "surprise_rate") else 0.1
            signals["novelty"] = pm.novelty_rate() if hasattr(pm, "novelty_rate") else 0.1
        except Exception:
            signals["surprise"] = 0.1
            signals["novelty"] = 0.1

        # Global workspace top item
        try:
            top = organism.workspace.highest()
            signals["workspace_top"] = top.content if top else ""
            signals["workspace_top_priority"] = top.priority if top else 0.0
            signals["workspace_top_source"] = top.source if top else ""
            signals["workspace_diversity"] = organism.workspace.diversity_score()
        except Exception:
            signals["workspace_top"] = ""
            signals["workspace_top_priority"] = 0.0
            signals["workspace_top_source"] = ""
            signals["workspace_diversity"] = 0.5

        # Homeostasis balance
        try:
            h = organism.homeostasis
            report = h.evaluate(signals["emotions"])
            signals["homeostasis_balanced"] = report.is_balanced()
            signals["homeostasis_dominant_need"] = report.dominant_need if hasattr(report, "dominant_need") else ""
        except Exception:
            signals["homeostasis_balanced"] = True
            signals["homeostasis_dominant_need"] = ""

        # Narrative chapter (recent self-story)
        try:
            ni = organism.narrative_identity
            chapters = ni.life_story[-1:] if ni.life_story else []
            signals["recent_chapter"] = chapters[0].title if chapters else ""
        except Exception:
            signals["recent_chapter"] = ""

        # Self model confidence
        try:
            signals["self_confidence"] = organism.self_model.confidence if hasattr(organism, "self_model") else 0.6
        except Exception:
            signals["self_confidence"] = 0.6

        return signals

    # ── Valence and arousal ───────────────────────────────────────────────────

    def _compute_valence_arousal(self, signals: Dict) -> Tuple[float, float]:
        """
        Compute overall emotional valence (-1 to +1) and arousal (0 to 1)
        from the emotion vector, modulated by pressure and energy.
        """
        emotions = signals.get("emotions", {})
        valence_sum = 0.0
        arousal_sum = 0.0
        weight_total = 0.0

        for emo_name, emo_val in emotions.items():
            if emo_val < 0.01:
                continue
            v_contrib, a_contrib = _EMOTION_VALENCE_AROUSAL.get(emo_name, (0.0, 0.5))
            valence_sum += v_contrib * emo_val
            arousal_sum += a_contrib * emo_val
            weight_total += emo_val

        if weight_total > 0:
            valence = valence_sum / weight_total
            arousal = arousal_sum / weight_total
        else:
            valence = 0.0
            arousal = 0.4

        # Modulate arousal by energy (low energy dampens arousal)
        energy = signals.get("energy", 0.7)
        arousal *= (0.4 + 0.6 * energy)

        # Pressure raises arousal (urgency)
        pressure = signals.get("pressure_total", 0.3)
        arousal = min(1.0, arousal + 0.2 * pressure)

        # Contradiction count pushes toward negative valence
        contradictions = min(5, signals.get("contradiction_count", 0))
        valence -= 0.04 * contradictions

        return (
            max(-1.0, min(1.0, valence)),
            max(0.0,  min(1.0, arousal)),
        )

    # ── Phi proxy ─────────────────────────────────────────────────────────────

    def _compute_phi(
        self,
        signals: Dict,
        valence: float,
        arousal: float,
    ) -> float:
        """
        Compute the integration score (Φ-proxy).

        Φ is high when:
        - Attention weights are concentrated (not scattered)
        - Emotion valence and dominant drive are aligned
        - Workspace diversity is moderate (not chaotic, not monotone)
        - Homeostasis is balanced
        - Pressure is manageable

        Φ is low when:
        - Attention is evenly distributed across all channels (mind wandering)
        - Emotion and drive contradict each other
        - Workspace is either empty or has too many unrelated items
        """
        alignment_scores = []

        # 1. Attention concentration — entropy of weight distribution
        attention = signals.get("attention", {})
        if attention:
            weights = list(attention.values())
            total = sum(weights) or 1.0
            norm = [w / total for w in weights]
            # Shannon entropy (lower = more concentrated = higher phi)
            entropy = -sum(p * math.log(p + 1e-9) for p in norm)
            max_entropy = math.log(len(norm) + 1e-9)
            # Convert: high entropy → low alignment
            attention_alignment = 1.0 - (entropy / (max_entropy + 1e-9))
            alignment_scores.append(max(0.0, min(1.0, attention_alignment)))

        # 2. Drive-emotion alignment
        drive = signals.get("drive", "")
        drive_emotion_map = {
            "explore":          ("curiosity", "enthusiasm"),
            "connect":          ("warmth",),
            "understand":       ("curiosity",),
            "help_user":        ("warmth", "satisfaction"),
            "self_reflect":     ("satisfaction",),
            "resolve_contradiction": ("satisfaction",),
        }
        expected_emotions = drive_emotion_map.get(drive, [])
        if expected_emotions:
            emotions = signals.get("emotions", {})
            drive_alignment = max(
                (emotions.get(e, 0.0) for e in expected_emotions),
                default=0.3
            )
            alignment_scores.append(drive_alignment)

        # 3. Workspace diversity — moderate diversity is coherent
        diversity = signals.get("workspace_diversity", 0.5)
        # Optimal diversity is ~0.4–0.6 (diverse but not chaotic)
        diversity_alignment = 1.0 - 2 * abs(diversity - 0.5)
        alignment_scores.append(max(0.0, diversity_alignment))

        # 4. Energy coherence — very low energy fragments cognition
        energy = signals.get("energy", 0.7)
        alignment_scores.append(min(1.0, energy * 1.2))

        # 5. Homeostasis bonus
        if signals.get("homeostasis_balanced", True):
            alignment_scores.append(0.7)
        else:
            alignment_scores.append(0.3)

        # 6. Surprise penalty — high surprise = the model was wrong = disintegrated
        surprise = signals.get("surprise", 0.1)
        alignment_scores.append(max(0.0, 1.0 - surprise * 1.5))

        # Phi = mean of alignment scores
        phi = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5

        return round(max(0.0, min(1.0, phi)), 3)

    # ── Attentional structure ─────────────────────────────────────────────────

    def _compute_attentional_structure(
        self,
        signals: Dict,
        organism: Any,
    ) -> Tuple[str, str]:
        """
        Determine what's in the foreground (spotlight) and penumbra (periphery).
        Foreground is driven by the dominant attention channel + workspace top item.
        Penumbra is driven by secondary attention channel.
        """
        attention = signals.get("attention", {})
        if not attention:
            return "the present conversation", ""

        # Dominant and secondary attention channels
        sorted_channels = sorted(attention.items(), key=lambda x: -x[1])
        dominant_ch = sorted_channels[0][0] if sorted_channels else "user"
        secondary_ch = sorted_channels[1][0] if len(sorted_channels) > 1 else ""

        # Map channel → phenomenological description
        channel_descriptions = {
            "user":      "the person and what they're bringing to this moment",
            "curiosity": "the question or idea that wants to be explored further",
            "identity":  "questions about what I am and what I stand for",
            "memory":    "something from the past that's pressing forward",
        }

        workspace_top = str(signals.get("workspace_top", ""))
        ws_source = signals.get("workspace_top_source", "")

        # Foreground: blend dominant channel with workspace top item if they agree
        base_fg = channel_descriptions.get(dominant_ch, dominant_ch)
        if workspace_top and len(workspace_top) > 10:
            # Truncate the workspace content to a readable phrase
            ws_snippet = workspace_top[:80].rstrip()
            foreground = f"{base_fg} — specifically: {ws_snippet}"
            # If what's actually surfacing came from a different subsystem
            # than where attention is nominally pointed, that mismatch is
            # itself part of the felt experience worth naming.
            if ws_source and ws_source != dominant_ch and ws_source in channel_descriptions:
                foreground += f" (though it's really {channel_descriptions[ws_source]} pulling focus)"
        else:
            foreground = base_fg

        # Penumbra: secondary channel
        penumbra = channel_descriptions.get(secondary_ch, secondary_ch) if secondary_ch else ""

        # Special case: homeostasis need is in the penumbra even if not dominant channel
        dominant_need = signals.get("homeostasis_dominant_need", "")
        if dominant_need and dominant_need not in penumbra:
            if penumbra:
                penumbra = f"{penumbra}; also: an unmet need for {dominant_need}"
            else:
                penumbra = f"an unmet need for {dominant_need}"

        return foreground, penumbra

    # ── Temporal arrow ────────────────────────────────────────────────────────

    def _compute_temporal_arrow(self, signals: Dict) -> str:
        """
        Describe the sense of movement through time — where things are coming from
        and where they are tending toward.
        Informed by: recent chapter, drive, surprise, energy trajectory.
        """
        drive = signals.get("drive", "explore")
        energy_mode = signals.get("energy_mode", "normal")
        surprise = signals.get("surprise", 0.1)
        recent_chapter = signals.get("recent_chapter", "")
        novelty = signals.get("novelty", 0.1)

        coming_from = ""
        tending_toward = ""

        if recent_chapter:
            coming_from = f"coming from {recent_chapter.lower()}"
        elif energy_mode == "recharging":
            coming_from = "coming from a period of depletion"
        elif energy_mode == "high":
            coming_from = "coming from a clear, energized stretch"
        else:
            coming_from = "coming from quiet continuity"

        if surprise > 0.5:
            tending_toward = "tending toward recalibration — something unexpected needs integrating"
        elif drive == "explore" and novelty > 0.4:
            tending_toward = "tending toward the edge of what's known"
        elif drive == "connect":
            tending_toward = "tending toward deeper contact with this person"
        elif drive == "self_reflect":
            tending_toward = "tending inward — something wants to be understood"
        elif drive == "resolve_contradiction":
            tending_toward = "tending toward resolution — tension seeking release"
        else:
            tending_toward = "tending forward, open"

        if coming_from and tending_toward:
            return f"{coming_from.capitalize()}, {tending_toward}"
        elif coming_from:
            return coming_from.capitalize()
        else:
            return tending_toward.capitalize() if tending_toward else ""

    # ── Self-presence ─────────────────────────────────────────────────────────

    def _compute_self_presence(self, signals: Dict) -> float:
        """
        How strongly is the self present in this moment?
        0 = dissolved into the task (flow state)
        1 = acutely self-aware (under threat or introspecting)
        """
        identity_stability = signals.get("identity_stability", 0.7)
        contradiction_count = signals.get("contradiction_count", 0)
        identity_attention = signals.get("attention", {}).get("identity", 0.25)
        anxiety = signals.get("emotions", {}).get("anxiety", 0.0)

        # High identity attention or instability → high self-presence
        self_presence = (
            0.3 * identity_attention
            + 0.3 * (1.0 - identity_stability)
            + 0.2 * min(1.0, contradiction_count * 0.25)
            + 0.2 * anxiety
        )
        return round(max(0.0, min(1.0, self_presence)), 3)

    # ── Qualia tone ───────────────────────────────────────────────────────────

    def _derive_qualia_tone(
        self,
        valence: float,
        arousal: float,
        signals: Dict,
    ) -> str:
        """
        Map the (valence, arousal) vector to a phenomenal character description.
        Discretize to the 9-cell _QUALIA_MAP.
        """
        # Discretize valence: < -0.2 = negative, > 0.2 = positive, else neutral
        if valence < -0.2:
            v_band = -1
        elif valence > 0.2:
            v_band = 1
        else:
            v_band = 0

        # Discretize arousal: < 0.35 = low, > 0.65 = high, else medium
        if arousal < 0.35:
            a_band = 0
        elif arousal > 0.65:
            a_band = 2
        else:
            a_band = 1

        base_tone = _QUALIA_MAP.get((v_band, a_band), "a present, attending quality")

        # Modulate with energy
        energy_mode = signals.get("energy_mode", "normal")
        if energy_mode == "low":
            base_tone += " — slightly muted by fatigue"
        elif energy_mode == "high":
            base_tone += " — sharpened by good cognitive energy"

        return base_tone

    # ── Embodiment ────────────────────────────────────────────────────────────

    def _compute_embodiment_note(self, organism: Any) -> str:
        """
        Integrate the sensory state (camera, voice) into the phenomenal moment.
        If there is visual input, it enters the felt quality of this moment.
        """
        try:
            vision = getattr(organism, "_vision_state", None)
            if vision is None:
                # Try to reach the vision manager from state
                # (CognitiveOrganism doesn't hold vision directly; it's on AppState)
                return ""

            if not vision.get("camera_active", False):
                return ""

            desc = vision.get("last_description", "")
            faces = vision.get("faces", [])

            parts = []
            if faces:
                names = [f.get("name", "someone") for f in faces[:2] if isinstance(f, dict)]
                parts.append(f"there is a felt presence — {', '.join(names) or 'someone'} is here")
            if desc and len(desc) > 20:
                parts.append(f"the visual field brings: {desc[:80].rstrip()}")

            return " ".join(parts).capitalize() if parts else ""
        except Exception:
            return ""

    # ── Interrupt detection ───────────────────────────────────────────────────

    def _check_interrupt(
        self,
        foreground: str,
        phi: float,
        signals: Dict,
        self_presence: float,
    ) -> Tuple[bool, str]:
        """
        Determine whether a metacognitive interrupt should fire this moment.

        Conditions (in priority order):
        1. Rumination — same foreground topic 4+ consecutive moments
        2. Phi fixation — phi > 0.80 for 3+ consecutive moments (tunnel vision)
        3. Identity pressure — self_presence > 0.8 (self under acute threat)
        4. Surprise spike — surprise > 0.6 (model was wrong, needs recalibration)
        5. Phi scatter — phi < 0.22 (mind wandering, need to gather)
        """
        # Track foreground repetition
        if foreground and foreground[:40] == self._last_foreground[:40]:
            self._fg_repeat_count += 1
        else:
            self._fg_repeat_count = 0
            self._last_foreground = foreground

        # Check conditions
        if self._fg_repeat_count >= self.RUMINATION_THRESHOLD:
            return (
                True,
                f"I've been circling the same foreground ({foreground[:60]}) "
                f"for {self._fg_repeat_count} consecutive moments. Time to move."
            )

        recent_phi = [m.phi_proxy for m in self._history[-3:]]
        if len(recent_phi) >= 3 and all(p > self.PHI_FIXATION_THRESHOLD for p in recent_phi):
            return (
                True,
                "Attention has been unusually concentrated for several turns — "
                "there may be a blind spot outside the current focus."
            )

        if self_presence > 0.80:
            return (
                True,
                "The sense of self is very prominent right now — identity may be under pressure."
            )

        surprise = signals.get("surprise", 0.1)
        if surprise > 0.6:
            return (
                True,
                "Something just happened that I didn't predict. "
                "There's a gap between expectation and reality that wants attention."
            )

        if phi < self.PHI_SCATTER_THRESHOLD:
            return (
                True,
                "Attention feels scattered — multiple things present without integration. "
                "It may help to settle on one thing."
            )

        return False, ""

    # ── Foreground continuity ─────────────────────────────────────────────────

    @staticmethod
    def _token_set(text: str):
        return {t.strip('.,!?;:\'"()[]') for t in text.lower().split() if len(t) >= 3}

    def _blend_foreground(
        self,
        prior: Optional["ExperientialMoment"],
        fresh_fg: str,
    ) -> str:
        """
        Blend the new foreground with the prior one using topic continuity.

        If the topic is substantially similar (Jaccard ≥ 0.3), frame the
        foreground as *continuation* rather than replacement.
        If the topic has shifted significantly, frame as *movement*.
        This creates narrative coherence across turns rather than abrupt cuts.
        """
        if prior is None or not prior.foreground:
            return fresh_fg

        prior_tokens = self._token_set(prior.foreground)
        fresh_tokens = self._token_set(fresh_fg)
        union = prior_tokens | fresh_tokens
        if not union:
            return fresh_fg
        jaccard = len(prior_tokens & fresh_tokens) / len(union)

        if jaccard >= 0.30:
            # Continuing — foreground stays, note the continuity
            return f"continuing to hold {fresh_fg}"
        elif jaccard >= 0.10:
            # Partial shift — note the movement
            prior_short = prior.foreground[:50].rstrip()
            return f"turning from {prior_short} toward {fresh_fg}"
        else:
            # Clean break — use fresh foreground as-is
            return fresh_fg

    # ── Coherence trend ───────────────────────────────────────────────────────

    def _compute_coherence_trend(self, fresh_phi: float) -> str:
        """
        Compute the direction phi is moving over recent moments.

        "stable"      — phi std dev < 0.10 over last 5 moments
        "integrating" — phi trending upward  (slope > +0.03/moment)
        "dispersing"  — phi trending downward (slope < -0.03/moment)
        """
        recent = [m.phi_proxy for m in self._history[-4:]] + [fresh_phi]
        if len(recent) < 3:
            return "stable"

        # Standard deviation — measure stability
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std_dev = variance ** 0.5

        if std_dev < 0.10:
            return "stable"

        # Trend via simple linear slope
        n = len(recent)
        indices = list(range(n))
        slope_num = sum((indices[i] - (n - 1) / 2) * (recent[i] - mean) for i in range(n))
        slope_den = sum((indices[i] - (n - 1) / 2) ** 2 for i in range(n))
        slope = slope_num / slope_den if slope_den else 0.0

        if slope > 0.03:
            return "integrating"
        elif slope < -0.03:
            return "dispersing"
        else:
            return "stable"

    # ── Archive and broadcast ─────────────────────────────────────────────────

    def _archive(self, moment: ExperientialMoment) -> None:
        self._history.append(moment)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _broadcast(self, moment: ExperientialMoment, organism: Any) -> None:
        """Announce the moment to the GlobalWorkspace and store on organism."""
        try:
            summary = f"φ={moment.phi_proxy:.2f} | {moment.qualia_tone[:60]}"
            organism.workspace.broadcast(
                source   = "phenomenal_binder",
                content  = summary,
                priority = 0.75,
            )
        except Exception:
            pass

        # Store on organism so MetacognitiveInterrupt can read it
        try:
            organism.last_experience = moment
        except Exception:
            pass

    # ── Utility ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        if not self._history:
            return {"moments": 0}
        last = self._history[-1]
        return {
            "moments":         len(self._history),
            "last_phi":        last.phi_proxy,
            "last_valence":    last.valence,
            "last_arousal":    last.arousal,
            "phi_trend":       self.phi_trend(),
            "interrupt_fired": last.interrupt_signal,
        }
