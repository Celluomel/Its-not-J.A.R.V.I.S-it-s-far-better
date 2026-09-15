"""
Curiosity Engine
================
Tracks Lumina's evolving intellectual curiosity across topics and concepts.

Curiosity is not just a mood — it is a directed cognitive drive. This engine
maintains a weighted topic map where curiosity accumulates through:
  - knowledge gaps (encountered but not understood)
  - user interest signals (topics they raise repeatedly)
  - contradictions (unresolved cognitive tension generates curiosity)
  - novelty (new topics start with moderate curiosity by default)

Curiosity decays slowly over time if not re-stimulated, creating a
natural forgetting and re-discovery cycle.

The engine drives:
  - Spontaneous questions in responses
  - Background research selection in the internal loop
  - Topic revisiting behavior
  - Identity trait "intellectual interests"

Architecture:
  - topic_map: Dict[str, CuriosityNode]  — the living interest landscape
  - global_curiosity: float [0,1]        — overall arousal level
  - decay: slow exponential per topic, based on real elapsed time
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

from cognition.topic_quality import get_topic_filter

CURIOSITY_HALFLIFE_HOURS = 48.0    # topics lose half their curiosity in 2 days
NOVELTY_SEED = 0.45                # initial curiosity assigned to new topics
MAX_TOPICS = 200                   # memory ceiling
RESEARCH_THRESHOLD = 0.65         # curiosity level that triggers background research
QUESTION_THRESHOLD = 0.55         # curiosity level that triggers spontaneous questions


@dataclass
class CuriosityNode:
    topic: str
    curiosity: float           # 0.0 – 1.0
    times_encountered: int
    times_researched: int
    source: str                # "user_signal" | "gap" | "contradiction" | "novelty"
    last_stimulated: float
    last_researched: Optional[float] = None
    related_topics: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)    # pending questions about this topic


class CuriosityEngine:
    """
    Manages Lumina's intellectual curiosity landscape.

    Usage
    -----
    curiosity = CuriosityEngine()

    # After processing a user message:
    curiosity.stimulate("quantum entanglement", amount=0.3, source="user_signal")
    curiosity.decay_all()

    # Get what to research next:
    topic = curiosity.top_topic()

    # Check if a spontaneous question is warranted:
    if curiosity.wants_to_ask():
        q = curiosity.generate_question_hint()
    """

    def __init__(self, persistence_path: str = "data/persona/curiosity.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._topics: Dict[str, CuriosityNode] = {}
        self._global: float = 0.5
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if self._path.exists():
                # Use explicit utf-8 with 'replace' so Windows curly-quotes
                # (0x93/0x94 in cp1252) don't crash the entire load.
                # Replacement char (U+FFFD) in a topic string is harmless.
                try:
                    raw_text = self._path.read_text(encoding='utf-8', errors='replace')
                except TypeError:
                    # Older Python / pathlib fallback
                    raw_text = self._path.read_bytes().decode('utf-8', errors='replace')
                data = json.loads(raw_text)
                self._global = float(data.get("global_curiosity", 0.5))
                for t, d in data.get("topics", {}).items():
                    self._topics[t] = CuriosityNode(
                        topic=t,
                        curiosity=float(d.get("curiosity", NOVELTY_SEED)),
                        times_encountered=int(d.get("times_encountered", 1)),
                        times_researched=int(d.get("times_researched", 0)),
                        source=d.get("source", "novelty"),
                        last_stimulated=float(d.get("last_stimulated", time.time())),
                        last_researched=d.get("last_researched"),
                        related_topics=d.get("related_topics", []),
                        questions=d.get("questions", []),
                    )
            # Clean up low-quality topics using the statistical filter
            _tqf = get_topic_filter()
            bad = [k for k in list(self._topics.keys()) if not _tqf.is_valid(k)]
            for k in bad:
                del self._topics[k]
            if bad:
                logger.info(f"[CuriosityEngine] Cleaned {len(bad)} low-quality topics on load")
        except Exception as e:
            logger.warning(f"[CuriosityEngine] Load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            topics_dict = {}
            for t, node in self._topics.items():
                topics_dict[t] = {
                    "curiosity": round(node.curiosity, 4),
                    "times_encountered": node.times_encountered,
                    "times_researched": node.times_researched,
                    "source": node.source,
                    "last_stimulated": node.last_stimulated,
                    "last_researched": node.last_researched,
                    "related_topics": node.related_topics[:5],
                    "questions": node.questions[-3:],
                }
            _tmp_path = self._path.with_suffix('.json.tmp')
            _tmp_path.write_text(json.dumps({
                "global_curiosity": round(self._global, 4),
                "topics": topics_dict,
            }, indent=2))
            _tmp_path.replace(self._path)  # atomic — avoids torn reads by other consumers
        except Exception as e:
            logger.warning(f"[CuriosityEngine] Save failed: {e}")

    # ── Core operations ───────────────────────────────────────────────────────

    # Words that should never become curiosity topics
    _TOPIC_STOP_WORDS = {
        "just","this","inner","that","with","from","some","more","other","any",
        "all","each","very","here","there","now","only","also","even","still",
        "such","both","same","then","than","when","about","thing","things",
        "sentence","speak","response","user","okay","lumina","system",
        "j'ai","besoin","t'en","c'est","what","have","does","will","would",
        "could","should","said","tell","want","like","know","make","take",
        "come","look","feel","need","been","were","they","their","them",
        # French verb conjugations and common short words that leak through
        "était","avait","avoir","être","faire","dire","aller","venir",
        "arrêté","arrête","arrêter","s'est","c'est","n'est","qu'il",
        "parce","quand","comme","après","avant","aussi","mais","donc",
        "alors","enfin","voilà","voici","cela","celui","celle","ceux",
    }

    def stimulate(
        self,
        topic: str,
        amount: float = 0.25,
        source: str = "novelty",
        related: Optional[List[str]] = None,
    ) -> float:
        """
        Increase curiosity for a topic. Creates the topic if new.

        Returns the new curiosity value.
        """
        # v50: gate check — suppress new topic generation when attention or
        # cognitive_energy is critically low.  Immune counterfactual topics
        # bypass this (source="immune_counterfactual") — they serve recovery.
        if source != "immune_counterfactual":
            try:
                from core.state import state as _st
                _org  = getattr(getattr(_st, 'persona', None), '_organism', None)
                _gate = getattr(_org, 'behavior_gate', None) if _org else None
                if _gate and not _gate.may_generate_curiosity():
                    return self._global
            except Exception:
                pass

        topic = topic.lower().strip()
        # Reject low-quality topics using statistical IDF filter
        # (TopicQualityFilter uses WorldModel's live corpus — language agnostic)
        _tqf = get_topic_filter()
        if not _tqf.is_valid(topic):
            return self._global   # silently reject bad topic
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = CuriosityNode(
                    topic=topic,
                    curiosity=NOVELTY_SEED,
                    times_encountered=0,
                    times_researched=0,
                    source=source,
                    last_stimulated=time.time(),
                )
            node = self._topics[topic]
            node.curiosity = min(1.0, node.curiosity + amount)
            node.times_encountered += 1
            node.last_stimulated = time.time()
            node.source = source
            if related:
                for r in related:
                    if r not in node.related_topics:
                        node.related_topics.append(r)

            # Global curiosity rises when individual topics are stimulated
            self._global = min(1.0, self._global + amount * 0.15)

            # Cull oldest/lowest if over cap
            if len(self._topics) > MAX_TOPICS:
                self._cull()
            self._save()
            return node.curiosity

    def decay_all(self) -> None:
        """Apply time-based curiosity decay to all topics. Call once per cycle.
        Phase 2.5: floor=0.05, cap=60 topics, prune stale >7d.
        Phase 2.9 GAP 4: floor is now attention-weighted — curiosity_engine
        attention share scales the floor between 0.02 (low focus) and 0.15
        (high focus), so Lumina's curiosity concentration reflects overall
        cognitive priority rather than a fixed constant.
        """
        now = time.time()
        _MAX_TOPICS        = 60
        _PRUNE_STALE_HOURS = 168.0

        # Phase 2.9: attention-weighted floor
        try:
            import json as _jce
            from pathlib import Path as _Pce
            _ap = _Pce("data/persona/cognitive_attention.json")
            _cw = _jce.loads(_ap.read_text()).get(
                "attention_weights", {}
            ).get("curiosity_engine", 0.10) if _ap.exists() else 0.10
        except Exception:
            _cw = 0.10
        _FLOOR = round(max(0.02, min(0.15, _cw * 0.5)), 3)

        with self._lock:
            for node in self._topics.values():
                elapsed_h = (now - node.last_stimulated) / 3600.0
                if elapsed_h > 0:
                    factor = 0.5 ** (elapsed_h / CURIOSITY_HALFLIFE_HOURS)
                    node.curiosity = max(_FLOOR, node.curiosity * factor)

            # Prune stale topics at floor (>7 days without stimulation)
            _stale = [
                t for t, n in self._topics.items()
                if n.curiosity <= _FLOOR
                and (now - n.last_stimulated) / 3600.0 > _PRUNE_STALE_HOURS
            ]
            for t in _stale:
                del self._topics[t]

            # Cap at 60 topics — keep highest curiosity
            if len(self._topics) > _MAX_TOPICS:
                _sorted = sorted(self._topics.items(), key=lambda x: x[1].curiosity)
                for t, _ in _sorted[:len(self._topics) - _MAX_TOPICS]:
                    del self._topics[t]

            # Global decays toward baseline 0.30 (prevents permanent 100%)
            self._global = max(0.30, self._global * 0.992)
            self._save()

    def mark_researched(self, topic: str) -> None:
        topic = topic.lower().strip()
        with self._lock:
            if topic in self._topics:
                node = self._topics[topic]
                node.times_researched += 1
                node.last_researched = time.time()
                # Research partially satisfies curiosity
                node.curiosity = max(0.0, node.curiosity - 0.2)
                self._save()

    def add_question(self, topic: str, question: str) -> None:
        topic = topic.lower().strip()
        with self._lock:
            if topic in self._topics:
                node = self._topics[topic]
                if question not in node.questions:
                    node.questions.append(question)
                    node.questions = node.questions[-5:]   # keep last 5
                    self._save()

    def boost_from_contradiction(self, topic: str) -> None:
        """Contradictions generate strong curiosity spikes."""
        self.stimulate(topic, amount=0.40, source="contradiction")

    # ── Query API ─────────────────────────────────────────────────────────────

    def top_topic(self, exclude_recently_researched_hours: float = 6.0) -> Optional[str]:
        """Return the highest-curiosity topic not recently researched."""
        now = time.time()
        with self._lock:
            candidates = [
                (node.curiosity, node.topic)
                for node in self._topics.values()
                if node.curiosity >= RESEARCH_THRESHOLD
                and (
                    node.last_researched is None
                    or (now - node.last_researched) / 3600.0 > exclude_recently_researched_hours
                )
            ]
            if not candidates:
                return None
            candidates.sort(reverse=True)
            return candidates[0][1]

    def wants_to_ask(self) -> bool:
        """True if curiosity is high enough to generate a spontaneous question."""
        with self._lock:
            return self._global >= QUESTION_THRESHOLD or any(
                n.curiosity >= QUESTION_THRESHOLD for n in self._topics.values()
            )

    def top_questions(self, n: int = 3) -> List[Tuple[str, str]]:
        """Return (topic, question) pairs for the most curious topics with pending questions."""
        with self._lock:
            results = []
            for node in sorted(self._topics.values(), key=lambda x: -x.curiosity):
                if node.questions and node.curiosity >= QUESTION_THRESHOLD:
                    results.append((node.topic, node.questions[-1]))
                if len(results) >= n:
                    break
            return results

    def global_level(self) -> float:
        with self._lock:
            return self._global

    def top_interests(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-N (topic, curiosity) pairs — Lumina's current interests."""
        with self._lock:
            ranked = sorted(
                self._topics.items(),
                key=lambda kv: -kv[1].curiosity
            )
            return [(t, round(node.curiosity, 3)) for t, node in ranked[:n]]

    def prompt_fragment(self) -> str:
        """Inject Lumina's current intellectual interests into the system prompt."""
        interests = self.top_interests(4)
        if not interests:
            return ""
        items = ", ".join(f"{t} ({v:.0%})" for t, v in interests)
        return f"Current intellectual interests (curiosity level): {items}."

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "global_curiosity": round(self._global, 3),
                "top_interests": self.top_interests(5),
                "wants_to_ask": self.wants_to_ask(),
                "research_ready_topic": self.top_topic(),
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cull(self):
        """Remove lowest-curiosity topics when over the cap."""
        ranked = sorted(self._topics.items(), key=lambda kv: kv[1].curiosity)
        to_remove = len(self._topics) - MAX_TOPICS
        for topic, _ in ranked[:to_remove]:
            del self._topics[topic]
