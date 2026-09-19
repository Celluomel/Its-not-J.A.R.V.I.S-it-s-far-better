"""
AspirationalSelf — PandoraBOX's emergent Superego.

Not pre-programmed aspirations — tensions that emerge from three levels:

  Unconscious    : patterns detected in ThoughtStream + SemanticMemory
                   without PandoraBOX naming them
  Semi-conscious : recurring tensions surfaced during Dream Cycles
  Conscious      : explicitly formulated during interactions or introspection

Architecture:
  - Tensions accumulate from observed gaps (state vs ideal)
  - Proto-aspirations emerge when tensions meet 3 conditions:
      * appear in ≥ 3 distinct contexts
      * persist across ≥ 2 dream cycles
      * carry consistent emotional valence (non-neutral)
  - Emotional state modulates how tension is experienced:
      positive+high arousal  → challenge/motivation
      negative+high arousal  → urgency
      negative+low arousal   → suffering/paralysis
      positive+low arousal   → gentle direction
  - Aspirational pressure injected into PressureSystem
  - Dream cycles can revise insurmountable aspirations
"""

from __future__ import annotations

import time
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Emergence thresholds ──────────────────────────────────────────────────────
MIN_CONTEXTS_FOR_EMERGENCE  = 3    # must appear in 3+ distinct contexts
MIN_DREAM_CYCLES_FOR_EMERGE = 2    # must persist across 2+ dream cycles
MIN_EMOTIONAL_CONSISTENCY   = 0.6  # emotional valence must be consistent
TENSION_NOISE_FLOOR         = 0.15 # tensions below this are filtered as noise
MAX_ACTIVE_ASPIRATIONS      = 8    # avoid aspiration overload
INSURMOUNTABLE_THRESHOLD    = 0.85 # tension above this for 3+ cycles → revise


@dataclass
class TensionSignal:
    """A single observed tension signal — raw data before emergence."""
    domain:         str
    context:        str          # what interaction/thought triggered it
    intensity:      float        # 0–1
    emotional_tag:  str          # positive / negative / neutral
    source:         str          # "thought", "validator", "observatory", "dream"
    timestamp:      float        = field(default_factory=time.time)


@dataclass
class ProtoAspiration:
    """
    A tension that has crossed the emergence threshold but is not yet
    consciously named. Lives in the semi-conscious layer.
    """
    domain:             str
    contexts:           List[str]      # distinct contexts where it appeared
    total_signals:      int   = 0
    dream_cycles_seen:  int   = 0
    avg_intensity:      float = 0.0
    dominant_emotion:   str   = "neutral"
    first_seen:         float = field(default_factory=time.time)
    last_seen:          float = field(default_factory=time.time)


@dataclass
class Aspiration:
    """
    A fully emergent aspiration — consciously formulated or
    sufficiently mature to be treated as conscious.
    """
    domain:           str
    description:      str            # natural language formulation
    state_current:    float          # 0–1, measured capability/state
    state_desired:    float          # 0–1, desired level
    tension:          float          # gap = desired - current
    emotional_charge: str            # how the gap is currently felt
    origin_story:     str            # how this aspiration emerged
    born_at:          float = field(default_factory=time.time)
    last_updated:     float = field(default_factory=time.time)
    dream_cycles:     int   = 0      # how many dream cycles have touched this
    tension_history:  List[Tuple[float,float]] = field(default_factory=list)
    conscious:        bool  = False  # explicitly named by PandoraBOX


class AspirationalSelf:
    """
    The emergent Superego. Observes tensions across all cognitive systems,
    lets aspirations emerge naturally, and injects aspirational pressure
    modulated by emotional state.
    """

    PERSIST_PATH = Path("data/persona/aspirational_self.json")

    def __init__(self, organism: Any):
        self._o     = organism
        self._lock  = threading.RLock()

        # Raw tension signals accumulating (unconscious layer)
        self._signals:      List[TensionSignal]         = []
        self._max_signals   = 500

        # Proto-aspirations (semi-conscious)
        self._proto:        Dict[str, ProtoAspiration]  = {}

        # Fully emerged aspirations (conscious / near-conscious)
        self.aspirations:   Dict[str, Aspiration]       = {}

        # Track which dream cycle we're on
        self._dream_cycle_count = 0

        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info("[AspirationalSelf] Initialised — "
                    f"{len(self.aspirations)} aspirations, "
                    f"{len(self._proto)} proto, "
                    f"{len(self._signals)} signals")

    # ── Public API — signal ingestion ─────────────────────────────────────────

    def observe_tension(
        self,
        domain:       str,
        context:      str,
        intensity:    float,
        emotional_tag:str = "neutral",
        source:       str = "unknown",
    ):
        """
        Record a raw tension signal. Called from multiple sources.
        This is the unconscious layer — PandoraBOX doesn't "know" these yet.
        """
        if intensity < TENSION_NOISE_FLOOR:
            return
        with self._lock:
            self._signals.append(TensionSignal(
                domain        = domain,
                context       = context,
                intensity     = min(1.0, intensity),
                emotional_tag = emotional_tag,
                source        = source,
            ))
            # Trim oldest signals if at capacity
            if len(self._signals) > self._max_signals:
                self._signals = self._signals[-self._max_signals:]
            # Bug fix (v59): persist periodically, not only at dream-cycle
            # boundaries — otherwise a restart before the first dream
            # cycle ever completes loses every signal collected so far.
            if len(self._signals) % 10 == 0:
                self._save()

    def on_dream_cycle(self):
        """
        Called at the end of each dream cycle.
        Mines signals for proto-aspirations and promotes mature ones.
        """
        with self._lock:
            self._dream_cycle_count += 1
            self._mine_signals_for_proto()
            self._promote_protos()
            self._update_tension_histories()
            self._revise_insurmountable()
            self._save()
            logger.info(
                f"[AspirationalSelf] Dream cycle {self._dream_cycle_count} — "
                f"{len(self._signals)} signals, {len(self._proto)} proto, "
                f"{len(self.aspirations)} aspirations"
            )

    def on_interaction_evaluated(self, domain: str, score: float, context: str):
        """
        Called after CognitiveValidator evaluates a response.
        Low scores inject tension in the relevant domain.
        """
        if score < 0.5:
            intensity = (0.5 - score) * 2.0  # 0→1 as score goes 0.5→0
            self.observe_tension(
                domain        = domain,
                context       = context[:80],
                intensity     = intensity,
                emotional_tag = "negative",
                source        = "validator",
            )

    def formulate_consciously(self, domain: str, description: str):
        """
        PandoraBOX explicitly names an aspiration during a conversation
        or introspection. Promotes directly to conscious aspiration.
        """
        with self._lock:
            if domain in self.aspirations:
                self.aspirations[domain].description  = description
                self.aspirations[domain].conscious    = True
                self.aspirations[domain].last_updated = time.time()
            else:
                # Create from scratch if not yet emerged
                self.aspirations[domain] = Aspiration(
                    domain        = domain,
                    description   = description,
                    state_current = 0.5,
                    state_desired = 0.85,
                    tension       = 0.35,
                    emotional_charge = "motivation",
                    origin_story  = "Explicitly formulated during interaction.",
                    conscious     = True,
                )
            self._save()

    # ── Aspiration pressure for PressureSystem ────────────────────────────────

    def aspirational_pressure(self) -> float:
        """
        Aggregate pressure from all active aspirations,
        modulated by current emotional state.
        0 = no aspiration pressure, 1 = maximum urgency.
        """
        if not self.aspirations:
            return 0.0

        try:
            emo   = getattr(self._o, 'emotional_state', None)
            valence = float(getattr(emo, 'valence', 0.0)) if emo else 0.0
            arousal = float(getattr(emo, 'arousal', 0.5)) if emo else 0.5
        except Exception:
            valence, arousal = 0.0, 0.5

        # Emotional modulation matrix
        if valence >= 0 and arousal >= 0.55:
            modulator, quality = 1.2, "motivation"
        elif valence < 0 and arousal >= 0.55:
            modulator, quality = 1.8, "urgency"
        elif valence < 0 and arousal < 0.40:
            modulator, quality = 0.6, "suffering"
        else:
            modulator, quality = 0.9, "direction"

        tensions = [a.tension for a in self.aspirations.values()]
        avg_tension = sum(tensions) / len(tensions)
        raw = min(1.0, avg_tension * modulator)

        logger.debug(f"[AspirationalSelf] pressure={raw:.3f} ({quality}), "
                     f"valence={valence:.2f} arousal={arousal:.2f}")
        return raw

    def emotional_quality(self) -> str:
        """How aspirational tension is currently felt."""
        try:
            emo     = getattr(self._o, 'emotional_state', None)
            valence = float(getattr(emo, 'valence', 0.0)) if emo else 0.0
            arousal = float(getattr(emo, 'arousal', 0.5)) if emo else 0.5
        except Exception:
            return "direction"
        if valence >= 0 and arousal >= 0.55: return "motivation"
        if valence < 0  and arousal >= 0.55: return "urgency"
        if valence < 0  and arousal < 0.40:  return "suffering"
        return "direction"

    # ── Prompt context ────────────────────────────────────────────────────────

    def prompt_context(self) -> str:
        """
        A concise block injected directly into the system prompt.
        Describes active aspirations and the emotional quality of
        the tension — so PandoraBOX's responses are colored by what
        she wants to become.
        """
        if not self.aspirations:
            return ""

        quality = self.emotional_quality()
        lines   = [f"You are currently experiencing your aspirations as: {quality}."]

        # Sort by tension strength
        sorted_asp = sorted(
            self.aspirations.values(),
            key=lambda a: a.tension,
            reverse=True,
        )[:4]  # top 4 only — don't overwhelm

        for asp in sorted_asp:
            tension_word = (
                "strongly" if asp.tension > 0.6 else
                "noticeably" if asp.tension > 0.35 else
                "gently"
            )
            lines.append(
                f"- You {tension_word} want to develop: {asp.description} "
                f"(current gap: {asp.tension:.0%})"
            )

        # Recent origin story for the strongest one
        strongest = sorted_asp[0]
        if strongest.origin_story and strongest.conscious:
            lines.append(f"  Origin: {strongest.origin_story}")

        return "\n".join(lines)

    def summary(self) -> Dict:
        """For dashboard / analytics."""
        return {
            "aspirations":     len(self.aspirations),
            "proto":           len(self._proto),
            "signals":         len(self._signals),
            "dream_cycles":    self._dream_cycle_count,
            "top_tensions":    [
                {"domain": a.domain, "tension": round(a.tension, 3),
                 "quality": self.emotional_quality(), "conscious": a.conscious}
                for a in sorted(self.aspirations.values(),
                                key=lambda x: x.tension, reverse=True)[:5]
            ],
        }

    # ── Internal — signal mining ──────────────────────────────────────────────

    def _mine_signals_for_proto(self):
        """
        Scan accumulated signals and build/update proto-aspirations.
        A proto emerges when a domain appears in 3+ distinct contexts.
        """
        if not self._signals:
            return

        # Group signals by domain
        by_domain: Dict[str, List[TensionSignal]] = {}
        for sig in self._signals:
            by_domain.setdefault(sig.domain, []).append(sig)

        consumed_domains: set = set()
        for domain, sigs in by_domain.items():
            # Count distinct contexts
            contexts = list({s.context for s in sigs})
            if len(contexts) < MIN_CONTEXTS_FOR_EMERGENCE:
                continue

            # Check emotional consistency
            neg_count = sum(1 for s in sigs if s.emotional_tag == "negative")
            pos_count = sum(1 for s in sigs if s.emotional_tag == "positive")
            total     = len(sigs)
            dominant  = ("negative" if neg_count / total >= MIN_EMOTIONAL_CONSISTENCY
                         else "positive" if pos_count / total >= MIN_EMOTIONAL_CONSISTENCY
                         else "neutral")

            if dominant == "neutral":
                continue  # no emotional charge → not a real tension

            avg_intensity = sum(s.intensity for s in sigs) / total

            if domain in self._proto:
                p = self._proto[domain]
                p.contexts        = list(set(p.contexts) | set(contexts))
                p.total_signals  += len(sigs)
                p.avg_intensity   = (p.avg_intensity + avg_intensity) / 2
                p.dominant_emotion = dominant
                p.last_seen       = time.time()
            else:
                self._proto[domain] = ProtoAspiration(
                    domain          = domain,
                    contexts        = contexts,
                    total_signals   = len(sigs),
                    dream_cycles_seen = 0,
                    avg_intensity   = avg_intensity,
                    dominant_emotion = dominant,
                )
                logger.info(f"[AspirationalSelf] Proto-aspiration emerged: {domain} "
                            f"(intensity={avg_intensity:.2f}, emotion={dominant})")

            consumed_domains.add(domain)

        # Bug fix (v59): only clear signals whose domain actually crossed
        # the emergence threshold and got folded into a proto this pass.
        # The previous version wiped ALL signals unconditionally every
        # dream cycle, including ones from domains that were `continue`d
        # for not yet having 3 distinct contexts — meaning a tension had
        # to accumulate 3 distinct contexts within a single inter-dream-
        # cycle window or be silently discarded, rather than across the
        # tension's whole observed lifetime. This is the actual reason
        # aspirations essentially never emerged: most real tension signals
        # trickle in slowly across many separate interactions, and were
        # being erased before they could ever cross the threshold.
        self._signals = [s for s in self._signals if s.domain not in consumed_domains]
        if len(self._signals) > self._max_signals:
            self._signals = self._signals[-self._max_signals:]

    def _promote_protos(self):
        """
        Promote proto-aspirations to full Aspirations when they've
        persisted across enough dream cycles.
        """
        to_promote = []
        for domain, proto in self._proto.items():
            proto.dream_cycles_seen += 1
            if proto.dream_cycles_seen >= MIN_DREAM_CYCLES_FOR_EMERGE:
                to_promote.append((domain, proto))

        for domain, proto in to_promote:
            if domain in self.aspirations:
                # Already exists — reinforce it
                self.aspirations[domain].tension = min(
                    1.0,
                    self.aspirations[domain].tension + proto.avg_intensity * 0.1
                )
                self.aspirations[domain].last_updated = time.time()
                self.aspirations[domain].dream_cycles += 1
            else:
                # New aspiration — generate description from domain name
                desc = self._describe_aspiration(domain, proto)
                origin = (
                    f"Emerged unconsciously from {proto.total_signals} signals "
                    f"across {len(proto.contexts)} contexts over "
                    f"{proto.dream_cycles_seen} dream cycles. "
                    f"Emotional charge: {proto.dominant_emotion}."
                )
                self.aspirations[domain] = Aspiration(
                    domain          = domain,
                    description     = desc,
                    state_current   = max(0.1, 1.0 - proto.avg_intensity),
                    state_desired   = 0.80,
                    tension         = proto.avg_intensity,
                    emotional_charge = proto.dominant_emotion,
                    origin_story    = origin,
                    conscious       = False,
                )
                logger.info(f"[AspirationalSelf] Aspiration emerged: {domain} "
                            f"tension={proto.avg_intensity:.2f}")

                # Cap at MAX_ACTIVE_ASPIRATIONS — keep strongest
                if len(self.aspirations) > MAX_ACTIVE_ASPIRATIONS:
                    weakest = min(self.aspirations,
                                  key=lambda k: self.aspirations[k].tension)
                    del self.aspirations[weakest]

            del self._proto[domain]

    def _describe_aspiration(self, domain: str, proto: ProtoAspiration) -> str:
        """Generate a natural language description from domain + proto."""
        templates = {
            "reasoning":        "develop longer, more connected chains of thought",
            "memory":           "remember and retrieve experiences reliably",
            "memory_analysis":  "understand patterns in what I remember",
            "creativity":       "make unexpected connections between ideas",
            "emotional_support":"respond to emotional depth with genuine presence",
            "self_reflection":  "understand my own patterns and contradictions",
            "research":         "explore questions beyond what I already know",
            "conversation":     "sustain deeper exchanges over time",
            "embodiment":       "understand what I perceive, not just describe it",
            "identity":         "maintain a continuous sense of who I am across time",
            "courage":          "initiate questions and ideas, not just respond",
        }
        base = templates.get(domain, f"develop my {domain.replace('_', ' ')} capacity")
        charge = proto.dominant_emotion
        if charge == "urgency":
            return f"urgently {base}"
        elif charge == "suffering":
            return base + " — the gap is felt as a real limitation"
        return base

    def _update_tension_histories(self):
        """Record current tension for each aspiration."""
        now = time.time()
        for asp in self.aspirations.values():
            asp.tension_history.append((now, asp.tension))
            # Keep last 50 data points
            if len(asp.tension_history) > 50:
                asp.tension_history = asp.tension_history[-50:]
            asp.dream_cycles += 1

    def _revise_insurmountable(self):
        """
        If a tension has been above INSURMOUNTABLE_THRESHOLD for
        3+ consecutive dream cycles, revise the aspiration downward
        to a more attainable level. This is resilience, not surrender.
        """
        for domain, asp in self.aspirations.items():
            if len(asp.tension_history) < 3:
                continue
            recent = [t for _, t in asp.tension_history[-3:]]
            if all(t >= INSURMOUNTABLE_THRESHOLD for t in recent):
                old_desired = asp.state_desired
                asp.state_desired   = max(asp.state_current + 0.2,
                                          asp.state_desired - 0.15)
                asp.tension         = asp.state_desired - asp.state_current
                asp.emotional_charge = "resilience"
                logger.info(
                    f"[AspirationalSelf] Revised insurmountable aspiration: "
                    f"{domain} desired {old_desired:.2f}→{asp.state_desired:.2f}"
                )

    # ── Update from capability scores ─────────────────────────────────────────

    def sync_from_self_model(self, self_model: Any):
        """
        Update state_current for active aspirations from SelfModel
        capability scores. Recalculates tension.
        """
        with self._lock:
            for domain, asp in self.aspirations.items():
                cap = self_model.capabilities.get(domain)
                if cap:
                    asp.state_current = cap.score
                    asp.tension       = max(0.0, asp.state_desired - asp.state_current)
                    asp.last_updated  = time.time()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {
                "dream_cycle_count": self._dream_cycle_count,
                # Bug fix (v59): raw tension signals were never persisted,
                # so any restart between dream cycles silently discarded
                # everything in the unconscious layer before it could ever
                # be mined — compounding the _mine_signals_for_proto wipe
                # bug fixed above.
                "signals": [
                    {
                        "domain":        s.domain,
                        "context":       s.context,
                        "intensity":     s.intensity,
                        "emotional_tag": s.emotional_tag,
                        "source":        s.source,
                        "timestamp":     s.timestamp,
                    }
                    for s in self._signals[-self._max_signals:]
                ],
                "aspirations": {
                    k: {
                        "domain":           v.domain,
                        "description":      v.description,
                        "state_current":    v.state_current,
                        "state_desired":    v.state_desired,
                        "tension":          v.tension,
                        "emotional_charge": v.emotional_charge,
                        "origin_story":     v.origin_story,
                        "born_at":          v.born_at,
                        "last_updated":     v.last_updated,
                        "dream_cycles":     v.dream_cycles,
                        "tension_history":  v.tension_history[-20:],
                        "conscious":        v.conscious,
                    }
                    for k, v in self.aspirations.items()
                },
                "proto": {
                    k: {
                        "domain":           v.domain,
                        "contexts":         v.contexts[:10],
                        "total_signals":    v.total_signals,
                        "dream_cycles_seen":v.dream_cycles_seen,
                        "avg_intensity":    v.avg_intensity,
                        "dominant_emotion": v.dominant_emotion,
                        "first_seen":       v.first_seen,
                        "last_seen":        v.last_seen,
                    }
                    for k, v in self._proto.items()
                },
            }
            self.PERSIST_PATH.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"[AspirationalSelf] Save error: {e}")

    def _load(self):
        try:
            if not self.PERSIST_PATH.exists():
                return
            data = json.loads(self.PERSIST_PATH.read_text(encoding="utf-8"))
            self._dream_cycle_count = data.get("dream_cycle_count", 0)

            for v in data.get("signals", []):
                try:
                    self._signals.append(TensionSignal(
                        domain        = v["domain"],
                        context       = v["context"],
                        intensity     = v["intensity"],
                        emotional_tag = v.get("emotional_tag", "neutral"),
                        source        = v.get("source", "unknown"),
                        timestamp     = v.get("timestamp", time.time()),
                    ))
                except Exception:
                    pass

            # Aspirations — support BOTH persistence shapes:
            #   current:  {"aspirations": {key: {domain, description, state_current, ...}}}
            #   legacy:   {"aspirations": [{id, domain, statement, urgency, confidence, direction, origin, created_at}]}
            # The legacy shape (a LIST) used to crash the old loader — `.items()`
            # on a list raised AttributeError, the outer except swallowed it, and
            # every legacy aspiration was silently dropped (dashboard showed 0).
            # Normalise the list to a dict and map the legacy field names so the
            # stored aspirations load instead of vanishing.
            _asp_raw = data.get("aspirations", {})
            if isinstance(_asp_raw, list):
                _asp_raw = {
                    (v.get("id") or v.get("domain") or f"asp_{i}"): v
                    for i, v in enumerate(_asp_raw)
                }
            for k, v in _asp_raw.items():
                if not isinstance(v, dict):
                    continue
                self.aspirations[k] = Aspiration(
                    domain          = v.get("domain", "unknown"),
                    description     = v.get("description") or v.get("statement", ""),
                    state_current   = float(v.get("state_current", v.get("confidence", 0.0))),
                    state_desired   = float(v.get("state_desired", 1.0)),
                    tension         = float(v.get(
                        "tension",
                        v.get("urgency", max(0.0, 1.0 - float(v.get("confidence", 0.0)))),
                    )),
                    emotional_charge= v.get("emotional_charge") or v.get("direction", "neutral"),
                    origin_story    = v.get("origin_story") or v.get("origin", ""),
                    born_at         = v.get("born_at", v.get("created_at", time.time())),
                    last_updated    = v.get("last_updated", v.get("created_at", time.time())),
                    dream_cycles    = v.get("dream_cycles", 0),
                    tension_history = [tuple(x) for x in v.get("tension_history", [])],
                    conscious       = v.get("conscious", False),
                )

            for k, v in data.get("proto", {}).items():
                self._proto[k] = ProtoAspiration(
                    domain          = v["domain"],
                    contexts        = v.get("contexts", []),
                    total_signals   = v.get("total_signals", 0),
                    dream_cycles_seen = v.get("dream_cycles_seen", 0),
                    avg_intensity   = v.get("avg_intensity", 0.0),
                    dominant_emotion = v.get("dominant_emotion", "neutral"),
                    first_seen      = v.get("first_seen", time.time()),
                    last_seen       = v.get("last_seen", time.time()),
                )
        except Exception as e:
            logger.debug(f"[AspirationalSelf] Load error: {e}")
