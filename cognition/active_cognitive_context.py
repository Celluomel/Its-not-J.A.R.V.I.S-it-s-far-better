"""
cognition/active_cognitive_context.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cache-Augmented Generation (CAG) — Layer 2 of PandoraBOX's cognitive stack.

Architecture position:
  Layer 1 — Long-term memory     FAISS / SQLite (slow, deep)
  Layer 2 — CAG ← THIS FILE      Active cognitive field (hot, continuous)
  Layer 3 — Workspace arbitration Attention competition
  Layer 4 — SCE                  Strategic trajectory
  Layer 5 — Embodiment grounding Vision / audio / sensors

What this solves
────────────────
Without CAG, every LLM call rebuilds context from scratch:
  event → retrieve → rebuild → respond

With CAG, cognition is continuous:
  ongoing mental state → conversation turn → state update → next turn

The difference: cognitive inertia.
PandoraBOX stops being a stateless improviser and starts having momentum of mind.

What lives in the CAG field
───────────────────────────
  current_topics       — what the conversation is actually about (rolling)
  unresolved_questions — threads opened but not closed
  emotional_tone       — inertial emotional trajectory (not just current)
  active_people        — who is present / recently present + their social weight
  current_environment  — vision / ambient signals (room state, time of day)
  strategic_posture    — from SCE, updated each turn
  dominant_attractors  — top personality traits dominating right now
  recent_visual_events — last 3 perception loop events
  ongoing_goals        — active goals from GoalEngine
  contradiction_tensions — unresolved contradictions (top 2)
  curiosity_threads    — open questions from ARE + curiosity engine
  social_field         — emotional resonance from EmpathyEngine per user

Update cadence
──────────────
  Per conversation turn : update_from_turn(user_text, ai_response, user_id)
  Per cognitive cycle   : update_from_organism(organism)  [every ~30s]
  Per vision event      : update_environment(description)
  Per presence event    : update_social_field(face_name, event)

Prompt injection
────────────────
  get_prompt_fragment() → compact natural-language block (~200 tokens)
  Injected near the top of system_prompt, before memory retrieval.

Public API
──────────
  acc = ActiveCognitiveContext(organism, llm)
  acc.update_from_turn(user_text, ai_response, user_id)
  acc.update_from_organism()
  acc.update_environment(vision_description)
  acc.update_social_field(name, event)
  fragment = acc.get_prompt_fragment()
  snapshot = acc.snapshot()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many items to keep per rolling field
MAX_TOPICS         = 5
MAX_QUESTIONS      = 6
MAX_VISUAL_EVENTS  = 3
MAX_GOALS          = 4
MAX_TENSIONS       = 2
MAX_CURIOSITY      = 4

# Inertia: how fast emotional tone shifts (0=instant, 1=never)
EMOTIONAL_INERTIA  = 0.65

# Topic weight decay per turn (topics fade if not reinforced)
TOPIC_DECAY        = 0.85


@dataclass
class TopicEntry:
    topic:      str
    weight:     float = 1.0
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    turn_count: int   = 1


@dataclass
class PersonField:
    name:             str
    user_id:          str   = ""
    present:          bool  = False
    last_seen:        float = 0.0
    emotional_signal: str   = "neutral"   # from EmpathyEngine
    social_weight:    float = 0.5         # 0=distant, 1=close
    dominant_signal:  str   = ""          # exploring/venting/challenging/etc.


class ActiveCognitiveContext:
    """
    The living cognitive field — PandoraBOX's working consciousness.

    Updated continuously from multiple sources.
    Read at every LLM call as a compact "hot context" fragment.
    """

    def __init__(self, organism: Any, llm: Any):
        self._org  = organism
        self._llm  = llm
        self._lock = threading.RLock()

        # ── Core CAG fields ───────────────────────────────────────────────
        self._topics:           Dict[str, TopicEntry]  = {}
        self._unresolved_qs:    Deque[str]             = deque(maxlen=MAX_QUESTIONS)
        self._emotional_tone:   str                    = "neutral"
        self._emotional_vector: Dict[str, float]       = {}   # {emotion: inertial_value}
        self._people:           Dict[str, PersonField] = {}
        self._environment:      str                    = ""   # last vision description
        self._strategic_posture:str                    = ""
        self._dominant_attractors: List[str]           = []
        self._visual_events:    Deque[str]             = deque(maxlen=MAX_VISUAL_EVENTS)
        self._ongoing_goals:    List[str]              = []
        self._tensions:         List[str]              = []
        self._curiosity_threads:Deque[str]             = deque(maxlen=MAX_CURIOSITY)

        # ── Meta ──────────────────────────────────────────────────────────
        self._turn_count:     int   = 0
        self._session_start:  float = time.time()
        self._last_org_sync:  float = 0.0
        self._last_fragment:  str   = ""   # cached for unchanged turns

        # Seed from organism immediately
        self._sync_from_organism()
        logger.info("🧠 ActiveCognitiveContext (CAG) initialised")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Update API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update_from_turn(
        self,
        user_text:   str,
        ai_response: str,
        user_id:     str = "default",
    ) -> None:
        """
        Called after every conversation turn.
        Extracts topics, updates emotional trajectory, pulls in new data.
        """
        with self._lock:
            self._turn_count += 1

        # ── Extract topics from this turn ─────────────────────────────────
        self._extract_and_merge_topics(user_text + " " + ai_response)

        # ── Decay old topics ───────────────────────────────────────────────
        self._decay_topics()

        # ── Pull ARE open question if fresh ───────────────────────────────
        try:
            from core.state import state as _st
            if _st.are:
                cached = list(_st.are._cache.values())
                if cached:
                    oq = cached[-1].open_question
                    if oq and oq not in self._unresolved_qs:
                        with self._lock:
                            self._unresolved_qs.appendleft(oq)
        except Exception:
            pass

        # ── Pull EmpathyEngine read for this user ─────────────────────────
        try:
            from core.state import state as _st
            if _st.empathy_engine:
                read = _st.empathy_engine.get_last_read(user_id)
                if read:
                    with self._lock:
                        if user_id in self._people:
                            pf = self._people[user_id]
                            pf.emotional_signal = read.dominant_emotion
                            pf.dominant_signal  = read.signals[0] if read.signals else ""
        except Exception:
            pass

        # ── Sync from organism every 30s ─────────────────────────────────
        if time.time() - self._last_org_sync > 30:
            self._sync_from_organism()

        # Invalidate cached fragment
        with self._lock:
            self._last_fragment = ""

    def update_from_organism(self) -> None:
        """Force a full sync from the organism. Call from cognitive cycle timer."""
        self._sync_from_organism()
        with self._lock:
            self._last_fragment = ""

    def update_environment(self, vision_description: str) -> None:
        """Called by VisionManager perception loop on each new event."""
        if not vision_description:
            return
        with self._lock:
            self._environment = vision_description[:200]
            self._visual_events.appendleft(vision_description[:100])
            self._last_fragment = ""

    def update_social_field(self, name: str, event: str, user_id: str = "") -> None:
        """
        Called by PresenceEngine on face events.
        Maintains who is present and updates their social weight.
        """
        with self._lock:
            uid = user_id or name.lower().replace(" ", "_")
            if uid not in self._people:
                self._people[uid] = PersonField(name=name, user_id=uid)
            pf = self._people[uid]
            pf.name      = name
            pf.last_seen = time.time()

            if event in ("ENTER", "RETURN"):
                pf.present       = True
                pf.social_weight = min(1.0, pf.social_weight + 0.05)
            elif event == "EXIT":
                pf.present = False
            elif event in ("DWELL", "SILENCE"):
                pf.social_weight = min(1.0, pf.social_weight + 0.02)

            self._last_fragment = ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Prompt fragment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_prompt_fragment(self) -> str:
        """
        Return a compact natural-language CAG block for system prompt injection.
        Cached per-turn to avoid redundant reconstruction.
        Approximately 150–250 tokens.
        """
        with self._lock:
            if self._last_fragment:
                return self._last_fragment

        fragment = self._build_fragment()

        with self._lock:
            self._last_fragment = fragment
        return fragment

    def _build_fragment(self) -> str:
        with self._lock:
            topics      = sorted(self._topics.values(), key=lambda t: t.weight, reverse=True)
            questions   = list(self._unresolved_qs)
            tone        = self._emotional_tone
            people      = [p for p in self._people.values() if p.present]
            environment = self._environment
            posture     = self._strategic_posture
            attractors  = self._dominant_attractors[:3]
            goals       = self._ongoing_goals[:3]
            tensions    = self._tensions[:2]
            curiosity   = list(self._curiosity_threads)[:3]
            v_events    = list(self._visual_events)[:2]

        lines = ["[CAG — Active Cognitive Field]"]

        # Topics
        if topics:
            top_topics = [f"{t.topic} ({t.weight:.1f})" for t in topics[:4]]
            lines.append(f"Active threads: {', '.join(top_topics)}")

        # Emotional inertia
        lines.append(f"Emotional inertia: {tone}")

        # Strategic posture
        if posture:
            lines.append(f"Posture: {posture}")

        # Attractors
        if attractors:
            lines.append(f"Dominant traits: {', '.join(attractors)}")

        # People
        if people:
            person_strs = []
            for p in people:
                s = p.name
                if p.dominant_signal:
                    s += f" ({p.dominant_signal})"
                person_strs.append(s)
            lines.append(f"Present: {', '.join(person_strs)}")

        # Environment
        if environment:
            lines.append(f"Environment: {environment[:80]}")

        # Visual events
        if v_events:
            lines.append(f"Recent vision: {v_events[0][:80]}")

        # Unresolved questions
        if questions:
            lines.append(f"Unresolved: {questions[0]}")
            if len(questions) > 1:
                lines.append(f"  Also open: {questions[1]}")

        # Curiosity threads
        if curiosity:
            lines.append(f"Curiosity pull: {curiosity[0]}")

        # Tensions
        if tensions:
            lines.append(f"Active tension: {tensions[0][:80]}")

        # Goals
        if goals:
            lines.append(f"Active goals: {'; '.join(goals)}")

        lines.append("[/CAG]")

        return "\n".join(lines)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Snapshot (for /state API and orchestrator page)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "turn_count":     self._turn_count,
                "session_age_s":  round(time.time() - self._session_start),
                "topics":         [
                    {"topic": t.topic, "weight": round(t.weight, 2), "turns": t.turn_count}
                    for t in sorted(self._topics.values(), key=lambda x: x.weight, reverse=True)[:5]
                ],
                "unresolved_questions": list(self._unresolved_qs)[:4],
                "emotional_tone":       self._emotional_tone,
                "strategic_posture":    self._strategic_posture,
                "dominant_attractors":  self._dominant_attractors[:3],
                "present_people":       [p.name for p in self._people.values() if p.present],
                "environment":          self._environment[:80],
                "ongoing_goals":        self._ongoing_goals[:3],
                "tensions":             self._tensions[:2],
                "curiosity_threads":    list(self._curiosity_threads)[:3],
                "visual_events":        list(self._visual_events)[:2],
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Internal sync and extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _sync_from_organism(self) -> None:
        """Pull live state from the cognitive organism."""
        org = self._org
        if org is None:
            return

        self._last_org_sync = time.time()

        # Emotional tone — inertial update
        try:
            raw_emotion = org._read_emotion_state() if hasattr(org, "_read_emotion_state") else "neutral"
            with self._lock:
                # Inertial blend: new tone pulls gradually toward raw
                if self._emotional_tone == "neutral" or self._emotional_tone == raw_emotion:
                    self._emotional_tone = raw_emotion
                else:
                    # Keep current unless raw is strongly different
                    self._emotional_tone = raw_emotion   # simplified — extend with blending later
        except Exception:
            pass

        # Strategic posture from SCE
        try:
            from core.state import state as _st
            if _st.sce:
                from cognition.strategic_cognitive_engine import CognitivePosition
                pos     = CognitivePosition.from_organism(org)
                posture = _st.sce._dominant_style(pos)
                with self._lock:
                    self._strategic_posture = posture
        except Exception:
            pass

        # Dominant attractors
        try:
            att = getattr(org, "attractor_system", None) or getattr(org, "attractors", None)
            if att and hasattr(att, "get_top"):
                top = att.get_top(3)
                with self._lock:
                    self._dominant_attractors = [f"{k}={v:.2f}" for k, v in top]
            elif att and hasattr(att, "_attractors"):
                attrs = sorted(att._attractors.items(), key=lambda x: x[1], reverse=True)[:3]
                with self._lock:
                    self._dominant_attractors = [f"{k}={v:.2f}" for k, v in attrs]
        except Exception:
            pass

        # Goals from GoalEngine
        try:
            ge = getattr(getattr(org, "ai_system", None), "goal_engine", None) or \
                 getattr(org, "goal_engine", None)
            if ge and hasattr(ge, "get_top_goals"):
                top_goals = ge.get_top_goals(MAX_GOALS)
                with self._lock:
                    self._ongoing_goals = [str(g)[:60] for g in top_goals]
            elif ge and hasattr(ge, "goals"):
                sorted_goals = sorted(
                    getattr(ge, "goals", []),
                    key=lambda g: getattr(g, "priority", 0),
                    reverse=True
                )[:MAX_GOALS]
                with self._lock:
                    self._ongoing_goals = [
                        getattr(g, "description", str(g))[:60] for g in sorted_goals
                    ]
        except Exception:
            pass

        # Active tensions from ContradictionHandler
        try:
            ch = getattr(getattr(org, "ai_system", None), "contradiction_handler", None) or \
                 getattr(org, "contradiction_handler", None)
            if ch and hasattr(ch, "get_open_contradictions"):
                tensions = ch.get_open_contradictions()[:MAX_TENSIONS]
                with self._lock:
                    self._tensions = [str(t)[:80] for t in tensions]
        except Exception:
            pass

        # Curiosity threads from ARE cache
        try:
            from core.state import state as _st
            if _st.are:
                cached = list(_st.are._cache.values())
                new_qs = [c.open_question for c in cached if c.open_question][-MAX_CURIOSITY:]
                with self._lock:
                    for q in new_qs:
                        if q not in self._curiosity_threads:
                            self._curiosity_threads.appendleft(q)
        except Exception:
            pass

    def _extract_and_merge_topics(self, text: str) -> None:
        """
        Extract topics from text using the LLM (best-effort, non-blocking).
        Falls back to keyword extraction if LLM is busy.
        """
        # Fast heuristic first
        heuristic_topics = self._heuristic_topics(text)

        with self._lock:
            for topic in heuristic_topics:
                if topic in self._topics:
                    entry = self._topics[topic]
                    entry.weight    = min(3.0, entry.weight + 0.5)
                    entry.last_seen = time.time()
                    entry.turn_count += 1
                else:
                    self._topics[topic] = TopicEntry(topic=topic)

            # Prune to MAX_TOPICS by weight
            if len(self._topics) > MAX_TOPICS * 2:
                sorted_topics = sorted(
                    self._topics.items(), key=lambda x: x[1].weight, reverse=True
                )
                self._topics = dict(sorted_topics[:MAX_TOPICS])

    def _decay_topics(self) -> None:
        """Apply weight decay to topics not mentioned this turn."""
        with self._lock:
            for entry in self._topics.values():
                if time.time() - entry.last_seen > 60:   # not updated in last minute
                    entry.weight *= TOPIC_DECAY
            # Remove very faded topics
            self._topics = {
                k: v for k, v in self._topics.items() if v.weight > 0.1
            }

    @staticmethod
    def _heuristic_topics(text: str) -> List[str]:
        """Extract noun-phrase-like topics using simple pattern matching."""
        import re
        # Remove punctuation, lowercase
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = clean.split()

        # Stop words
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "i", "you", "we", "they",
            "it", "this", "that", "what", "how", "why", "when", "where", "which",
            "and", "or", "but", "so", "yet", "for", "nor", "if", "then", "than",
            "of", "in", "on", "at", "to", "by", "with", "from", "about", "as",
            "my", "your", "his", "her", "its", "our", "their", "me", "him", "them",
            "up", "out", "not", "no", "just", "very", "also", "even", "like",
        }

        # Candidate words: length > 4, not stop words
        candidates = [w for w in words if len(w) > 4 and w not in stop]

        # Bigrams
        bigrams = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words)-1)
            if words[i] not in stop and words[i+1] not in stop
            and len(words[i]) > 3 and len(words[i+1]) > 3
        ]

        # Deduplicate + limit
        seen  = set()
        result = []
        for t in (bigrams[:3] + candidates[:4]):
            if t not in seen:
                seen.add(t)
                result.append(t)
            if len(result) >= 4:
                break

        return result
