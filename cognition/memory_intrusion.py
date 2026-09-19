"""
MemoryIntrusionSystem — Spontaneous Memory Surfacing
=====================================================
Passive memory doesn't make an organism — intrusive memory does.

In a human brain, memories surface without being asked.
A smell triggers a childhood memory. A word pattern resurfaces a conversation
from last week. This is not retrieval — it's intrusion.

This module watches the current cognitive context (workspace content,
recent user input, emotional state) and injects memories that are
structurally similar — even when no one asked for them.

How it works:
1. Every slow cycle, encode the current context into a query vector
2. Search memory by embedding similarity
3. Score each candidate by: similarity × recency_weight × emotional_weight
4. If score > INTRUSION_THRESHOLD: broadcast to workspace + emit bus event
5. The memory enters the ThoughtStream as a thought, colours the response

Connections:
    - Reads  : memory_system (embedding search)
    - Reads  : workspace (current cognitive context)
    - Reads  : emotional_state (emotional resonance)
    - Writes : workspace (memory broadcast)
    - Emits  : CognitiveEventBus (MEMORY_INTRUSION)

The intrusion system does NOT write to memory — only reads.
It is purely observational + broadcast.

Requires: shared_embedder (already in utils/)
"""

import logging
import math
import time
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryCandidate:
    content:    str
    topic:      str
    similarity: float
    recency:    float    # 0–1, 1 = very recent
    emotional:  float    # 0–1, emotional salience
    score:      float    # combined
    timestamp:  float = field(default_factory=time.time)


class MemoryIntrusionSystem:
    """
    Watches current context and injects emotionally / semantically
    similar memories into the workspace spontaneously.
    """

    INTRUSION_THRESHOLD = 0.58    # minimum combined score to intrude
    RECENCY_HALFLIFE    = 86400   # 1 day in seconds — recency decay
    COOLDOWN_SECONDS    = 120     # minimum gap between intrusions (avoid spam)
    MAX_PER_CYCLE       = 2       # max intrusions per slow cycle

    def __init__(self, organism: Any):
        self._o             = organism
        self._last_intrusion = 0.0
        self._total_intrusions = 0
        self._recent_intruded: List[str] = []   # avoid repeating same memory

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, context_text: Optional[str] = None) -> List[MemoryCandidate]:
        """
        Called during slow cycle. Returns list of memories that intruded.
        Side effects: broadcasts to workspace, emits bus events.
        """
        now = time.time()
        if now - self._last_intrusion < self.COOLDOWN_SECONDS:
            return []

        # Build context query from current cognitive state
        query = context_text or self._build_context_query()
        if not query or len(query) < 10:
            return []

        candidates = self._search_memories(query)
        if not candidates:
            return []

        intruded = []
        for candidate in candidates[:self.MAX_PER_CYCLE]:
            if candidate.score >= self.INTRUSION_THRESHOLD:
                if candidate.content[:50] not in self._recent_intruded:
                    self._intrude(candidate)
                    intruded.append(candidate)

        if intruded:
            self._last_intrusion = now
            self._total_intrusions += len(intruded)
            # Keep recent list bounded
            self._recent_intruded.extend(c.content[:50] for c in intruded)
            if len(self._recent_intruded) > 30:
                self._recent_intruded = self._recent_intruded[-30:]

        return intruded

    def summary(self) -> Dict:
        return {
            "total_intrusions": self._total_intrusions,
            "cooldown_remaining": max(
                0, self.COOLDOWN_SECONDS - (time.time() - self._last_intrusion)
            ),
        }

    # ── Memory Search ─────────────────────────────────────────────────────────

    def _search_memories(self, query: str) -> List[MemoryCandidate]:
        """
        Search memory by embedding similarity. Falls back gracefully
        if embedder or memory system unavailable.
        """
        candidates = []

        try:
            ai = getattr(self._o, 'ai_system', None)
            if not ai:
                return []
            mem = getattr(ai, 'memory_system', None)
            if not mem:
                return []

            # Try vector search first (FAISS)
            if hasattr(mem, 'retrieve_memories'):
                memories = mem.retrieve_memories(query, limit=8)
                if memories:
                    for m in memories:
                        c = self._score_candidate(m, query)
                        if c:
                            candidates.append(c)

        except Exception as e:
            logger.debug(f"[MemoryIntrusion] search error: {e}")

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _score_candidate(self, memory: Any, query: str) -> Optional[MemoryCandidate]:
        """Compute the combined intrusion score for a memory candidate."""
        try:
            if isinstance(memory, dict):
                content = str(memory.get("text", ""))
                relevance = memory.get("relevance", 0.5)
                timestamp = memory.get("timestamp", time.time())
                valence = memory.get("emotional_valence", "Neutral")
                arousal = memory.get("arousal_level", "Medium")
            else:
                content = str(getattr(memory, "content", memory))
                relevance = getattr(memory, "relevance", 0.5)
                timestamp = getattr(memory, "timestamp", time.time())
                valence = getattr(memory, "emotional_valence", 0.0)
                arousal = getattr(memory, "arousal", 0.0)
            if len(content) < 10:
                return None

            # Estimate semantic similarity from the retrieval score if available
            similarity = float(relevance)
            if similarity < 0.1:
                return None

            if isinstance(timestamp, str):
                try:
                    mem_time = datetime.datetime.fromisoformat(timestamp).timestamp()
                except ValueError:
                    mem_time = time.time()
            else:
                mem_time = float(timestamp)
            age_secs = max(0.0, time.time() - mem_time)
            recency  = 2 ** (-age_secs / self.RECENCY_HALFLIFE)

            # Emotional salience
            if isinstance(valence, str):
                valence_score = {"positive": 1.0, "negative": 1.0}.get(valence.lower(), 0.0)
            else:
                valence_score = min(1.0, abs(float(valence)))
            if isinstance(arousal, str):
                arousal_score = {"low": 0.25, "medium": 0.55, "high": 1.0}.get(arousal.lower(), 0.0)
            else:
                arousal_score = min(1.0, abs(float(arousal)))
            emotional = (valence_score + arousal_score) / 2.0

            # Combined score
            # Retrieval relevance is the strongest evidence.  Recency and
            # affect modulate an already relevant memory rather than drowning
            # it out after a few days, which made reliable recall look weak.
            score = (
                similarity * 0.90 +
                recency    * 0.07 +
                emotional  * 0.03
            )

            # Topic extraction (simple: first 3 content words)
            words = content.split()
            topic = " ".join(w.strip(".,!?") for w in words[:3] if len(w) > 3)

            return MemoryCandidate(
                content    = content[:200],
                topic      = topic,
                similarity = round(similarity, 3),
                recency    = round(recency, 3),
                emotional  = round(emotional, 3),
                score      = round(score, 3),
                timestamp  = mem_time,
            )
        except Exception as e:
            logger.debug(f"[MemoryIntrusion] score error: {e}")
            return None

    # ── Intrusion Broadcasting ────────────────────────────────────────────────

    def _intrude(self, candidate: MemoryCandidate) -> None:
        """Broadcast a memory into workspace + emit EventBus event."""

        # Broadcast to workspace
        ws = getattr(self._o, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="memory_intrusion",
                    content=f"Memory surfaces: {candidate.content[:100]}",
                    priority=candidate.score * 0.8,
                )
            except Exception:
                pass

        # Emit bus event
        bus = getattr(self._o, 'event_bus', None)
        if bus:
            try:
                from core.cognitive_event_bus import MEMORY_INTRUSION
                bus.emit(MEMORY_INTRUSION, {
                    "content":    candidate.content,
                    "topic":      candidate.topic,
                    "score":      candidate.score,
                    "similarity": candidate.similarity,
                }, source="memory_intrusion")
            except Exception:
                pass

        # Also push to ThoughtStream buffer
        ts = getattr(self._o, 'thought_stream', None)
        if ts:
            try:
                from cognition.thought_stream import Thought
                thought = Thought(
                    content      = f"Something surfaces from memory: {candidate.content[:80]}",
                    source       = "memory",
                    thought_type = "memory_intrusion",
                    priority     = candidate.score * 0.65,
                )
                ts._buffer.append(thought)
            except Exception:
                pass

        logger.debug(
            f"[MemoryIntrusion] intruded: score={candidate.score:.2f} "
            f"topic={candidate.topic!r}"
        )

    # ── Context Builder ───────────────────────────────────────────────────────

    def _build_context_query(self) -> str:
        """
        Build a context string from current organism state
        to use as the memory search query.
        """
        parts = []

        # Top workspace items
        ws = getattr(self._o, 'workspace', None)
        if ws:
            try:
                items = ws.recent(3)
                for item in items:
                    content = getattr(item, 'content', str(item))
                    if isinstance(content, str) and len(content) > 5:
                        parts.append(content[:60])
            except Exception:
                pass

        # Current curiosity topic
        cu = getattr(self._o, 'curiosity', None)
        if cu:
            try:
                top = cu.top_topic()
                if top:
                    parts.append(top)
            except Exception:
                pass

        # Recent thought from ThoughtStream
        ts = getattr(self._o, 'thought_stream', None)
        if ts:
            try:
                recent = ts.recent(1)
                if recent:
                    parts.append(recent[-1].content[:60])
            except Exception:
                pass

        return " ".join(parts)[:300]
