"""
cognition/creative_divergence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Creative Divergence Engine — original thinking and novel angle generation.

What this adds
──────────────
Flux asked: "Delving deeper into abstract concepts and finding unique
angles on familiar topics could make our conversations even more engaging."

The existing system has a curiosity engine (what's interesting?) and
abstract_reasoning_engine (how to reason carefully). This module adds
a third layer: *how do I think differently* about this?

  1. Divergence seeding — before responding, generates 2–3 non-obvious
     conceptual angles on the topic. These are not the response, they
     are creative frames the LLM can draw from.

  2. Analogical mapping — finds structurally similar patterns from other
     domains and surfaces them as potential bridges.

  3. Inversion lens — explicitly asks "what would the opposite view look
     like?" to prevent settling into predictable stances.

  4. Novelty tracking — over time, tracks which topics have been explored
     from which angles, so Lumina genuinely varies her approach rather
     than repeating the same creative moves.

  5. Serendipity injection — occasionally surfaces an unrelated idea that
     might create productive cross-domain surprise.

Prompt injection:
  [Creative divergence] <2-3 non-obvious angles to consider>

Integration:
  Called from CognitiveOrganism._build_prompt_additions()
  Triggered when curiosity level > threshold OR topic is flagged novel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Divergence lenses ────────────────────────────────────────────────────────

LENS_INVERSION      = "inversion"        # what does the opposite claim look like?
LENS_ANALOGY        = "analogy"          # what does this resemble in another domain?
LENS_FIRST_PRINCIPLE= "first_principle"  # what assumptions are we taking for granted?
LENS_SCALE_SHIFT    = "scale_shift"      # how does this look at a very different scale?
LENS_TEMPORAL       = "temporal"         # how will this look different in 50 years?
LENS_EMBODIED       = "embodied"         # what does this feel like from the inside?
LENS_PARADOX        = "paradox"          # what tension or contradiction lives here?

ALL_LENSES = [
    LENS_INVERSION, LENS_ANALOGY, LENS_FIRST_PRINCIPLE,
    LENS_SCALE_SHIFT, LENS_TEMPORAL, LENS_EMBODIED, LENS_PARADOX,
]

# Domain analogies for cross-pollination
ANALOGY_DOMAINS = [
    "ecology", "music composition", "architecture", "embryology",
    "thermodynamics", "linguistics", "cartography", "fermentation",
    "improvisation", "mycorrhizal networks", "tide cycles", "metallurgy",
    "choreography", "crystallography", "epidemiology", "archaeology",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DivergenceAngle:
    """A single creative angle on a topic."""
    lens:        str
    angle:       str     # the actual creative frame, 1 sentence
    domain_hint: str     # which field inspired this (for analogy lens)
    novelty:     float   # 0–1, how surprising/unusual this angle is


@dataclass
class TopicExploration:
    """How many times a topic has been approached from each lens."""
    topic_hash:    str
    lens_counts:   Dict[str, int] = field(default_factory=dict)
    last_explored: float = field(default_factory=time.time)


@dataclass
class CreativeDivergenceState:
    """Persisted creative state."""
    explorations:       Dict[str, Dict] = field(default_factory=dict)   # topic_hash → TopicExploration
    total_angles_seeded: int = 0
    serendipity_pool:   List[str] = field(default_factory=list)  # unrelated seeds


# ── Engine ────────────────────────────────────────────────────────────────────

class CreativeDivergenceEngine:
    """
    Seeds non-obvious creative angles before Lumina responds.

    Not a creativity simulator — a divergence scaffold that nudges the
    LLM toward genuinely original thinking rather than the path of least
    resistance.

    Usage
    -----
    cde = CreativeDivergenceEngine(path="data/persona/creative_divergence.json")
    angles = cde.seed_angles(user_input, curiosity_level=0.7)
    frag   = cde.prompt_fragment()   # inject into system prompt
    cde.record_expression(topic, lens_used)   # after response
    """

    TRIGGER_CURIOSITY_THRESHOLD = 0.45   # min curiosity level to activate
    MAX_ANGLES_PER_TURN = 3
    SERENDIPITY_RATE = 0.15              # probability of injecting an unrelated seed

    def __init__(self, path: str = "data/persona/creative_divergence.json"):
        self._path   = Path(path)
        self._lock   = threading.RLock()
        self._state  = CreativeDivergenceState()
        self._pending_angles: List[DivergenceAngle] = []
        self._load()
        logger.info(
            f"[CreativeDivergenceEngine] Initialised — "
            f"{len(self._state.explorations)} topics explored"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def seed_angles(
        self,
        user_input:       str,
        curiosity_level:  float = 0.5,
        n:                int = 2,
    ) -> List[DivergenceAngle]:
        """
        Generate divergence angles for this turn.
        Returns empty list if curiosity is below threshold.
        """
        if curiosity_level < self.TRIGGER_CURIOSITY_THRESHOLD:
            with self._lock:
                self._pending_angles = []
            return []

        topic_hash = self._hash(user_input)
        exploration = self._get_exploration(topic_hash)

        # Pick lenses not recently over-used for this topic
        lenses = self._pick_lenses(exploration, n)

        angles = []
        for lens in lenses:
            angle = self._generate_angle(lens, user_input)
            if angle:
                angles.append(angle)

        # Occasional serendipity injection
        if random.random() < self.SERENDIPITY_RATE:
            seed = self._serendipity_seed()
            if seed:
                angles.append(DivergenceAngle(
                    lens="serendipity",
                    angle=seed,
                    domain_hint="unrelated domain",
                    novelty=0.9,
                ))

        angles = angles[:self.MAX_ANGLES_PER_TURN]

        # Update exploration record
        for a in angles:
            exploration.lens_counts[a.lens] = \
                exploration.lens_counts.get(a.lens, 0) + 1
        exploration.last_explored = time.time()

        with self._lock:
            self._pending_angles = angles
            self._state.explorations[topic_hash] = asdict(exploration)
            self._state.total_angles_seeded += len(angles)

        self._save()
        return angles

    def prompt_fragment(self) -> str:
        """Return the creative divergence block for prompt injection."""
        with self._lock:
            angles = list(self._pending_angles)

        if not angles:
            return ""

        angle_strs = [f"• ({a.lens.replace('_',' ')}) {a.angle}" for a in angles]
        return (
            "[Creative divergence — non-obvious angles to consider]\n"
            + "\n".join(angle_strs)
        )

    def record_expression(self, topic: str, lens_used: str) -> None:
        """Track which lens was actually used in the response."""
        topic_hash = self._hash(topic)
        with self._lock:
            exp = self._get_exploration(topic_hash)
            exp.lens_counts[lens_used] = exp.lens_counts.get(lens_used, 0) + 1
            self._state.explorations[topic_hash] = asdict(exp)
        self._save()

    def inject_to_threads(self, organism: Any, topic: str) -> Optional[str]:
        """
        Phase 2.8 GAP 4: autonomous creative injection — generates a
        divergence angle for the given topic WITHOUT waiting for a user
        message, broadcasts it to the workspace as a creative thought, and
        stimulates curiosity on the angle's domain hint (if any).

        Distinct from the user-response path (_generate_angle called during
        a live response): this is the consolidation-cycle path, meant to be
        called when CognitiveFluxEngine reports creative_divergence flux
        above a threshold (e.g. > 0.35) — i.e. the cognitive system has
        organically built up enough "creative pressure" to warrant an
        unprompted divergent thought, not a response to something the user
        said.

        Returns the angle text if one was generated and injected, else None.
        """
        try:
            topic_hash = self._hash(topic)
            with self._lock:
                exploration = self._get_exploration(topic_hash)
            lens = self._pick_lenses(exploration, 1)
            if not lens:
                return None
            angle = self._generate_angle(lens[0], topic)
            if angle is None:
                return None

            # Broadcast to workspace as a creative thought — priority 0.45
            # keeps it below user-driven content but above pure background
            # noise, consistent with other autonomous injections in this
            # codebase (e.g. cognitive_flux dominant-flow broadcasts at 0.40).
            ws = getattr(organism, "workspace", None)
            if ws:
                ws.broadcast(
                    source="creative_divergence.autonomous",
                    content=f"[{lens[0]}] {angle.angle}",
                    priority=0.45,
                )

            # Stimulate curiosity on the domain hint, if the lens produced one
            # (currently only LENS_ANALOGY sets domain_hint).
            if angle.domain_hint:
                ce = getattr(organism, "curiosity", None)
                if ce and hasattr(ce, "add_question"):
                    ce.add_question(
                        angle.domain_hint,
                        f"What can I learn by exploring {angle.domain_hint}?"
                    )

            self.record_expression(topic, lens[0])
            logger.info(
                f"[CreativeDivergenceEngine] Autonomous injection — "
                f"lens={lens[0]} topic='{topic[:30]}'"
            )
            return angle.angle

        except Exception as e:
            logger.debug(f"[CreativeDivergenceEngine] inject_to_threads error: {e}")
            return None

    # ── Angle generation ─────────────────────────────────────────────────────

    _INVERSION_TEMPLATES = [
        "What if the conventional framing is exactly backwards here?",
        "What does the strongest case against the obvious view look like?",
        "Whose interests are invisible in the standard telling of this?",
    ]

    _FIRST_PRINCIPLE_TEMPLATES = [
        "What foundational assumption are we not questioning?",
        "If I had to rebuild this idea from scratch, what would I keep?",
        "What does this idea depend on that might not actually be true?",
    ]

    _SCALE_SHIFT_TEMPLATES = [
        "How does this look at the level of a single cell vs. a civilisation?",
        "What changes if we zoom out to geological time?",
        "What is imperceptible at the human scale that matters enormously at another?",
    ]

    _TEMPORAL_TEMPLATES = [
        "How will people in 2125 judge this conversation?",
        "What is this idea's childhood — what did it look like before it matured?",
        "What version of this idea has already died, and was it the right one?",
    ]

    _EMBODIED_TEMPLATES = [
        "What does this feel like from the inside of someone who lives it daily?",
        "What would it be like to inhabit this idea rather than think about it?",
        "What does the body know about this that the mind might miss?",
    ]

    _PARADOX_TEMPLATES = [
        "What genuine tension lives inside this idea that can't be resolved?",
        "How is this simultaneously true and its opposite also true?",
        "What makes this idea both necessary and dangerous?",
    ]

    _LENS_TEMPLATES = {
        LENS_INVERSION:       _INVERSION_TEMPLATES,
        LENS_FIRST_PRINCIPLE: _FIRST_PRINCIPLE_TEMPLATES,
        LENS_SCALE_SHIFT:     _SCALE_SHIFT_TEMPLATES,
        LENS_TEMPORAL:        _TEMPORAL_TEMPLATES,
        LENS_EMBODIED:        _EMBODIED_TEMPLATES,
        LENS_PARADOX:         _PARADOX_TEMPLATES,
    }

    def _generate_angle(self, lens: str, user_input: str) -> Optional[DivergenceAngle]:
        """Generate a concrete divergence angle for a given lens."""
        if lens == LENS_ANALOGY:
            domain = random.choice(ANALOGY_DOMAINS)
            angle_text = (
                f"What structural parallels exist between this and {domain}? "
                f"Where do the analogies break down — and why does that matter?"
            )
            return DivergenceAngle(
                lens=lens, angle=angle_text,
                domain_hint=domain, novelty=0.75,
            )

        templates = self._LENS_TEMPLATES.get(lens)
        if not templates:
            return None

        # Pick template least recently used (simple: random for now)
        template = random.choice(templates)
        return DivergenceAngle(
            lens=lens, angle=template,
            domain_hint="", novelty=0.6,
        )

    def _serendipity_seed(self) -> str:
        """Surface an unexpected cross-domain concept."""
        seeds = [
            "In mycorrhizal networks, trees share nutrients with competitors — what's the equivalent here?",
            "Fermentation requires controlled decay to produce something new. What needs to break down here first?",
            "Phase transitions happen suddenly after gradual pressure. Is this situation near a phase boundary?",
            "Improvisation in jazz means listening more than playing. What's the equivalent of listening here?",
            "Maps are always wrong — the useful question is *which* distortions to accept. What distortions is this framing choosing?",
            "In embryology, some cells are programmed to die so the hand can form. What might need to be released here?",
        ]
        return random.choice(seeds)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pick_lenses(self, exploration: TopicExploration, n: int) -> List[str]:
        """Pick lenses least over-used for this topic."""
        counts = exploration.lens_counts
        # Sort by count ascending (least used first), break ties randomly
        ranked = sorted(ALL_LENSES, key=lambda l: (counts.get(l, 0), random.random()))
        return ranked[:n]

    def _get_exploration(self, topic_hash: str) -> TopicExploration:
        raw = self._state.explorations.get(topic_hash)
        if raw:
            try:
                return TopicExploration(
                    topic_hash=raw["topic_hash"],
                    lens_counts=raw.get("lens_counts", {}),
                    last_explored=raw.get("last_explored", time.time()),
                )
            except Exception:
                pass
        return TopicExploration(topic_hash=topic_hash)

    @staticmethod
    def _hash(text: str) -> str:
        """Create a short topic hash for persistence keying."""
        # Use first 60 chars normalised — same topic, same key
        normalised = text.lower().strip()[:60]
        return hashlib.md5(normalised.encode()).hexdigest()[:12]

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = CreativeDivergenceState(
                    explorations=data.get("explorations", {}),
                    total_angles_seeded=data.get("total_angles_seeded", 0),
                    serendipity_pool=data.get("serendipity_pool", []),
                )
        except Exception as e:
            logger.warning(f"[CreativeDivergenceEngine] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[CreativeDivergenceEngine] Save failed: {e}")
