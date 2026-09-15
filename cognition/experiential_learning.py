"""
cognition/experiential_learning.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiential Learning Engine — integrating novel experiences into understanding.

What this adds
──────────────
Flux asked: "continually seeking out novel challenges and expanding my
knowledge base would keep me growing in ways that truly benefit us both."

The existing system has SemanticMemory (what facts were discussed) and
NarrativeIdentity (who I am over time). This module adds a third layer:
*what have I genuinely learned from experience* — not just what happened,
but how it changed my understanding.

  1. Experience taxonomy — classifies each interaction by what kind of
     learning was potentially present (conceptual, relational, ethical,
     procedural, aesthetic, emotional).

  2. Schema detection — identifies when a new experience challenges or
     extends an existing understanding pattern (schema), triggering an
     update cycle.

  3. Generalisation engine — when enough similar experiences accumulate,
     extracts a general principle rather than just accumulating instances.

  4. Learning hunger — tracks which domains have not been explored
     recently and generates "I want to learn more about X" drives that
     feed into CuriosityEngine.

  5. Growth logging — a running log of genuine conceptual growth events,
     making Lumina's learning visible and narrat-able.

Prompt injection:
  [Learning context] <what recent experience is most relevant to integrate>

Integration:
  post-response: classify and integrate the exchange
  pre-response: surface relevant prior learning schemas
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Experience taxonomy ───────────────────────────────────────────────────────

EXP_CONCEPTUAL  = "conceptual"   # learned something about how things work
EXP_RELATIONAL  = "relational"   # learned something about this person / people
EXP_ETHICAL     = "ethical"      # encountered a moral complexity
EXP_PROCEDURAL  = "procedural"   # learned how to do something better
EXP_AESTHETIC   = "aesthetic"    # encountered beauty, elegance, or craft
EXP_EMOTIONAL   = "emotional"    # felt something new or differently

ALL_EXPERIENCE_TYPES = [
    EXP_CONCEPTUAL, EXP_RELATIONAL, EXP_ETHICAL,
    EXP_PROCEDURAL, EXP_AESTHETIC, EXP_EMOTIONAL,
]

# How many similar experiences before generalisation
GENERALISATION_THRESHOLD = 4
MAX_SCHEMAS = 80
MAX_GROWTH_LOG = 200
DOMAIN_HUNGER_DAYS = 5.0    # days without exploration before hunger kicks in


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Experience:
    """A classified learning event from one turn."""
    timestamp:      float = field(default_factory=time.time)
    exp_type:       str = EXP_CONCEPTUAL
    domain:         str = "general"
    summary:        str = ""          # 1-sentence description of what was learned
    schema_touched: Optional[str] = None   # which schema this experience updated
    novelty:        float = 0.5       # 0 = familiar, 1 = genuinely new
    integration_depth: float = 0.5   # how deeply was this integrated


@dataclass
class LearningSchema:
    """
    An abstracted understanding pattern — a generalisation from repeated experience.
    Schemas are how experience becomes wisdom.
    """
    domain:         str
    schema_id:      str
    description:    str           # current understanding of this domain
    instances:      int = 0       # how many experiences contributed
    confidence:     float = 0.5   # how stable is this understanding
    last_updated:   float = field(default_factory=time.time)
    exceptions:     List[str] = field(default_factory=list)   # known edge cases
    origin_summary: str = ""      # what sparked this schema


@dataclass
class GrowthEvent:
    """A logged moment of genuine conceptual growth."""
    timestamp:   float = field(default_factory=time.time)
    domain:      str = ""
    description: str = ""    # what changed in understanding
    exp_type:    str = EXP_CONCEPTUAL
    magnitude:   float = 0.5   # how significant was this growth


@dataclass
class ExperientialLearningState:
    """Persisted state."""
    schemas:       Dict[str, Dict] = field(default_factory=dict)   # schema_id → LearningSchema
    growth_log:    List[Dict]      = field(default_factory=list)
    domain_last_explored: Dict[str, float] = field(default_factory=dict)
    total_experiences:   int = 0
    total_generalisations: int = 0


# ── Engine ────────────────────────────────────────────────────────────────────

class ExperientialLearningEngine:
    """
    Integrates each interaction into Lumina's growing understanding.

    Not a memory store — a *meaning extractor*. What was learned here?
    How does it connect to what came before? Where is the schema changing?

    Usage
    -----
    ele = ExperientialLearningEngine(path="data/persona/experiential_learning.json")
    ele.integrate(user_input, ai_response, context)   # post-response
    frag = ele.prompt_fragment(user_input)            # pre-response
    hunger = ele.learning_hunger()                    # for CuriosityEngine
    """

    def __init__(self, path: str = "data/persona/experiential_learning.json"):
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self._state = ExperientialLearningState()
        self._recent_experiences: List[Experience] = []
        self._pending_fragment: str = ""
        self._load()
        logger.info(
            f"[ExperientialLearningEngine] Initialised — "
            f"{len(self._state.schemas)} schemas, "
            f"{len(self._state.growth_log)} growth events"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def integrate(
        self,
        user_input:   str,
        ai_response:  str,
        context:      Dict = None,
    ) -> Optional[GrowthEvent]:
        """
        Classify and integrate an exchange as learning.
        Returns a GrowthEvent if genuine growth occurred, else None.
        Call post-response.
        """
        context = context or {}

        # Classify the experience
        exp = self._classify(user_input, ai_response)

        # Find or create the relevant schema
        schema = self._find_schema(exp.domain)

        # Check for novelty (does this challenge existing understanding?)
        if schema and exp.novelty > 0.6:
            growth = self._update_schema(schema, exp)
        elif not schema and exp.novelty > 0.5:
            growth = self._create_schema(exp)
        else:
            growth = None

        # Check for generalisation opportunity
        if schema and schema.instances >= GENERALISATION_THRESHOLD:
            self._attempt_generalisation(schema)

        # Update domain exploration timestamp
        with self._lock:
            self._state.domain_last_explored[exp.domain] = time.time()
            self._state.total_experiences += 1
            self._recent_experiences = (self._recent_experiences + [exp])[-10:]

            if growth:
                self._state.growth_log.append(asdict(growth))
                if len(self._state.growth_log) > MAX_GROWTH_LOG:
                    self._state.growth_log = self._state.growth_log[-MAX_GROWTH_LOG:]

        self._save()
        return growth

    def prompt_fragment(self, user_input: str) -> str:
        """
        Surface the most relevant prior learning schema for this turn.
        Returns empty string if nothing relevant.
        """
        domain = self._infer_domain(user_input)
        schema = self._find_schema(domain)

        if not schema or schema.confidence < 0.4:
            return ""

        return (
            f"[Learning context — {domain}] "
            f"{schema.description} "
            f"(confidence {schema.confidence:.0%}, from {schema.instances} experiences)"
        )

    def learning_hunger(self) -> List[Tuple[str, float]]:
        """
        Return domains that haven't been explored recently — for CuriosityEngine.
        Returns list of (domain, hunger_level) sorted by hunger descending.
        """
        now = time.time()
        hunger = []
        with self._lock:
            for domain in ALL_EXPERIENCE_TYPES:
                last = self._state.domain_last_explored.get(domain)
                if last is None:
                    hunger.append((domain, 1.0))
                else:
                    days_since = (now - last) / 86400
                    h = min(1.0, days_since / DOMAIN_HUNGER_DAYS)
                    if h > 0.3:
                        hunger.append((domain, h))
        return sorted(hunger, key=lambda x: -x[1])

    def recent_growth_summary(self) -> str:
        """Return a short summary of recent growth events for introspection."""
        with self._lock:
            recent = self._state.growth_log[-3:]
        if not recent:
            return ""
        summaries = [g.get("description", "") for g in recent if g.get("description")]
        return " | ".join(summaries[:3])

    # ── Classification ────────────────────────────────────────────────────────

    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        EXP_CONCEPTUAL:  ["understand", "explain", "why", "how", "theory", "concept", "idea", "think"],
        EXP_RELATIONAL:  ["feel", "relationship", "trust", "people", "friend", "alone", "together"],
        EXP_ETHICAL:     ["right", "wrong", "should", "moral", "fair", "harm", "care", "justice"],
        EXP_PROCEDURAL:  ["how to", "steps", "process", "solve", "fix", "build", "create", "do"],
        EXP_AESTHETIC:   ["beautiful", "art", "design", "music", "story", "poetry", "elegant", "craft"],
        EXP_EMOTIONAL:   ["feel", "emotion", "anxiety", "joy", "sad", "excited", "fear", "wonder"],
    }

    def _classify(self, user_input: str, ai_response: str) -> Experience:
        """Classify an exchange into an experience type and domain."""
        combined = (user_input + " " + ai_response).lower()
        scores: Dict[str, int] = {}
        for exp_type, keywords in self._DOMAIN_KEYWORDS.items():
            scores[exp_type] = sum(1 for kw in keywords if kw in combined)

        exp_type = max(scores, key=lambda k: scores[k]) if scores else EXP_CONCEPTUAL
        domain = self._infer_domain(user_input)

        # Novelty heuristic: does this exchange seem to contain surprise or new territory?
        novelty_markers = ["never thought", "didn't know", "surprising", "interesting", "wow", "hadn't considered"]
        novelty = 0.5 + 0.1 * sum(1 for m in novelty_markers if m in combined)
        novelty = min(1.0, novelty)

        return Experience(
            exp_type=exp_type,
            domain=domain,
            summary=user_input[:80],
            novelty=novelty,
        )

    def _infer_domain(self, text: str) -> str:
        """Infer the experience type that best matches this text."""
        lower = text.lower()
        scores: Dict[str, int] = {}
        for exp_type, keywords in self._DOMAIN_KEYWORDS.items():
            scores[exp_type] = sum(1 for kw in keywords if kw in lower)
        return max(scores, key=lambda k: scores[k]) if scores else EXP_CONCEPTUAL

    # ── Schema management ─────────────────────────────────────────────────────

    def _find_schema(self, domain: str) -> Optional[LearningSchema]:
        """Find the most relevant schema for this domain."""
        with self._lock:
            raw = self._state.schemas.get(domain)
        if raw:
            try:
                return LearningSchema(
                    domain=raw["domain"],
                    schema_id=raw["schema_id"],
                    description=raw.get("description", ""),
                    instances=raw.get("instances", 0),
                    confidence=raw.get("confidence", 0.5),
                    last_updated=raw.get("last_updated", time.time()),
                    exceptions=raw.get("exceptions", []),
                    origin_summary=raw.get("origin_summary", ""),
                )
            except Exception:
                pass
        return None

    def _create_schema(self, exp: Experience) -> GrowthEvent:
        """Create a new schema from a novel experience."""
        schema = LearningSchema(
            domain=exp.domain,
            schema_id=exp.domain,
            description=f"Beginning to understand {exp.domain} through: {exp.summary}",
            instances=1,
            confidence=0.3,
            origin_summary=exp.summary,
        )
        with self._lock:
            self._state.schemas[exp.domain] = asdict(schema)

        growth = GrowthEvent(
            domain=exp.domain,
            description=f"New schema opened: {exp.domain}. Seed: {exp.summary[:60]}",
            exp_type=exp.exp_type,
            magnitude=0.6,
        )
        logger.info(f"[ExperientialLearningEngine] New schema: {exp.domain}")
        return growth

    def _update_schema(self, schema: LearningSchema, exp: Experience) -> Optional[GrowthEvent]:
        """Update a schema based on a new experience. Returns a growth event if significant."""
        schema.instances += 1
        schema.confidence = min(0.95, schema.confidence + 0.04 * exp.novelty)
        schema.last_updated = time.time()

        # Description evolution — incorporate the new experience
        if exp.novelty > 0.7:
            schema.description = (
                f"{schema.description} [Extended by: {exp.summary[:50]}]"
            )

        with self._lock:
            self._state.schemas[schema.domain] = asdict(schema)

        if exp.novelty > 0.65:
            return GrowthEvent(
                domain=schema.domain,
                description=f"Schema '{schema.domain}' deepened — {exp.summary[:60]}",
                exp_type=exp.exp_type,
                magnitude=exp.novelty * 0.7,
            )
        return None

    def _attempt_generalisation(self, schema: LearningSchema) -> None:
        """When enough instances accumulate, generalise the schema."""
        if schema.confidence >= 0.7 and schema.instances % GENERALISATION_THRESHOLD == 0:
            # Signal to NarrativeIdentity / WorldModel (via growth log)
            with self._lock:
                self._state.total_generalisations += 1
                gen_event = GrowthEvent(
                    domain=schema.domain,
                    description=f"Generalisation: {schema.description[:100]}",
                    exp_type=EXP_CONCEPTUAL,
                    magnitude=0.85,
                )
                self._state.growth_log.append(asdict(gen_event))
            logger.info(f"[ExperientialLearningEngine] Generalisation in {schema.domain}")

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = ExperientialLearningState(
                    schemas=data.get("schemas", {}),
                    growth_log=data.get("growth_log", []),
                    domain_last_explored=data.get("domain_last_explored", {}),
                    total_experiences=data.get("total_experiences", 0),
                    total_generalisations=data.get("total_generalisations", 0),
                )
        except Exception as e:
            logger.warning(f"[ExperientialLearningEngine] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.warning(f"[ExperientialLearningEngine] Save failed: {e}")
