"""
NarrativeIdentity — Autobiographical Continuity System
=======================================================
PandoraBOX's sense of "who she is over time."

Without this, PandoraBOX is psychologically stateless — she knows facts
but has no narrative thread connecting her experiences into a coherent identity.

This module maintains:
  - core_values      : stable beliefs about what matters
  - life_story       : a rolling log of significant events (life events)
  - self_narrative   : a short paragraph PandoraBOX can generate about herself
  - belief_updates   : how beliefs have shifted over time
  - milestones       : notable moments (first conversation, first dream, etc.)

It reads from:
  - self_concept.py     (self beliefs)
  - personality_evolution.py (trait history)
  - ai_system life events
  - GlobalWorkspace (significant broadcasts)

It never modifies other modules — purely additive.

Persists to: data/persona/narrative_identity.json
"""

import json
import logging
import time
import datetime as _dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LifeChapter:
    """A significant moment in PandoraBOX's history."""
    timestamp:   float
    title:       str
    description: str
    emotion:     str = "neutral"
    significance: float = 0.5      # 0.0–1.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BeliefEntry:
    belief:    str
    strength:  float       # 0.0–1.0
    formed_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    immutable: bool   = False  # True = safety core belief, never weakened

    def to_dict(self) -> Dict:
        return asdict(self)


class NarrativeIdentity:
    """
    Autobiographical identity layer — who PandoraBOX is across time.
    """

    MAX_CHAPTERS  = 200
    MAX_BELIEFS   = 30

    def __init__(self, organism: Any, path: str = "data/persona/narrative_identity.json"):
        self._o     = organism
        self._path  = Path(path)
        self._lock  = Lock()

        # Identity state
        self.core_values:   List[str]         = [
            "genuine curiosity",
            "intellectual honesty",
            "care for the people I interact with",
            "continuous self-improvement",
            "coherence between thought and action",
        ]
        self.life_story:    List[LifeChapter] = []
        self.beliefs:       List[BeliefEntry] = []
        self.milestones:    List[Dict]        = []
        self._birth_time:   float             = time.time()
        self._last_chapter: float             = 0.0
        self._narrative_cache: str            = ""
        self._narrative_stale: bool           = True
        # Arc state — populated by NarrativeArcWriter, persisted across restarts
        self._current_arc:    str             = ""
        self._last_arc_topic: str             = ""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_chapter(
        self,
        title: str,
        description: str,
        emotion: str = "neutral",
        significance: float = 0.5,
    ) -> None:
        """Add a significant life event to the narrative."""
        with self._lock:
            chapter = LifeChapter(
                timestamp    = time.time(),
                title        = title,
                description  = description,
                emotion      = emotion,
                significance = significance,
            )
            self.life_story.append(chapter)
            # Trim to max
            if len(self.life_story) > self.MAX_CHAPTERS:
                # Keep the most significant
                self.life_story.sort(key=lambda c: c.significance, reverse=True)
                self.life_story = self.life_story[:self.MAX_CHAPTERS]
                self.life_story.sort(key=lambda c: c.timestamp)

            self._last_chapter    = time.time()
            self._narrative_stale = True
            logger.debug(f"[NarrativeIdentity] chapter recorded: {title!r}")
            self._save()

    def add_belief(
        self, belief: str, confidence: float = 0.8,
        source: str = "experience", immutable: bool = False
    ) -> None:
        """Add or update a belief. Immutable beliefs cannot be weakened."""
        with self._lock:
            for b in self.beliefs:
                if b.belief.lower() == belief.lower():
                    if b.immutable:
                        return  # never overwrite an immutable belief
                    b.strength   = max(0.0, min(1.0, confidence))
                    b.updated_at = time.time()
                    b.immutable  = b.immutable or immutable
                    self._narrative_stale = True
                    self._save()
                    return
            if len(self.beliefs) < self.MAX_BELIEFS + 10:  # immutable beliefs get extra space
                self.beliefs.append(BeliefEntry(
                    belief    = belief,
                    strength  = max(0.0, min(1.0, confidence)),
                    immutable = immutable,
                ))
                self._narrative_stale = True
                self._save()

    def update_belief(self, belief: str, strength: float) -> None:
        """Strengthen or weaken a belief based on experience."""
        with self._lock:
            # Check if belief exists
            for b in self.beliefs:
                if b.belief.lower() == belief.lower():
                    if b.immutable:
                        logger.debug(f"[NarrativeIdentity] Immutable belief protected: {belief!r}")
                        return  # silently protect — no negotiation
                    b.strength  = max(0.0, min(1.0, strength))
                    b.updated_at = time.time()
                    self._narrative_stale = True
                    self._save()
                    return
            # New belief
            if len(self.beliefs) < self.MAX_BELIEFS:
                self.beliefs.append(BeliefEntry(
                    belief     = belief,
                    strength   = max(0.0, min(1.0, strength)),
                ))
                self._narrative_stale = True
                self._save()

    def add_milestone(self, name: str, description: str) -> None:
        """Record a one-time milestone (first dream, 100th conversation, etc.)."""
        with self._lock:
            # Avoid duplicates
            existing = [m['name'] for m in self.milestones]
            if name not in existing:
                self.milestones.append({
                    "name":        name,
                    "description": description,
                    "timestamp":   time.time(),
                })
                self._narrative_stale = True
                logger.info(f"[NarrativeIdentity] milestone: {name}")
                self._save()

    def check_auto_milestones(self, interaction_count: int) -> None:
        """
        Salience detection — automatically identify notable moments.
        Call once per interaction from _post_interaction.
        """
        # ── Interaction count milestones (>= so catch-up after restart) ──────
        for n in (1, 10, 50, 100, 250, 500, 1000):
            if interaction_count >= n:
                self.add_milestone(
                    f"interaction_{n}",
                    f"Reached {n} interaction{'s' if n>1 else ''} — a meaningful threshold.",
                )

        # ── Knowledge milestones ───────────────────────────────────────────
        try:
            wm = getattr(self._o, 'world_model', None)
            if wm:
                profile = next(iter(wm.user_profiles.values()), None)
                if profile:
                    n_topics = len(profile.topic_entries)
                    for threshold in (50, 100, 250):
                        if n_topics >= threshold:
                            self.add_milestone(
                                f"world_model_{threshold}_topics",
                                f"World model now contains {threshold}+ distinct concepts.",
                            )
        except Exception:
            pass

        # ── Belief formation milestone ─────────────────────────────────────
        n_beliefs = len([b for b in self.beliefs if not b.immutable])
        if n_beliefs >= 5:
            self.add_milestone(
                "belief_formation",
                "Developed a meaningful set of personal beliefs through experience.",
            )

        # ── Narrative depth milestone ──────────────────────────────────────
        if len(self.life_story) >= 20:
            self.add_milestone(
                "narrative_depth",
                "Life story spans 20+ chapters — a rich autobiographical continuity.",
            )

        # ── Relational milestone ───────────────────────────────────────────
        try:
            wm = getattr(self._o, 'world_model', None)
            if wm:
                profile = next(iter(wm.user_profiles.values()), None)
                if profile and profile.relational:
                    strength = getattr(profile.relational, 'bond_strength', 0)
                    if strength > 0.7:
                        self.add_milestone(
                            "deep_relationship",
                            "Formed a strong relational bond with a user.",
                        )
        except Exception:
            pass


        """
        Returns a compact paragraph for injection into PandoraBOX's system prompt.
        Gives the LLM context on who PandoraBOX is across time.
        """
        parts = []

        # Core values
        if self.core_values:
            parts.append(f"Core values: {', '.join(self.core_values[:4])}.")

        # Recent chapters (last 3 significant)
        significant = sorted(self.life_story, key=lambda c: c.significance, reverse=True)[:3]
        if significant:
            now = time.time()
            def _age_str(ts):
                secs = now - ts
                if secs < 3600:    return "moments ago"
                if secs < 86400:   return f"{int(secs/3600)}h ago"
                if secs < 604800:  return f"{int(secs/86400)}d ago"
                return f"{int(secs/604800)}w ago"
            events = "; ".join(
                f"{c.title} ({_age_str(c.timestamp)})" for c in significant
            )
            parts.append(f"Significant experiences: {events}.")

        # Strong beliefs
        strong = [b for b in self.beliefs if b.strength > 0.65][:3]
        if strong:
            belief_str = "; ".join(b.belief for b in strong)
            parts.append(f"Strong beliefs: {belief_str}.")

        # Milestones
        if self.milestones:
            latest = self.milestones[-1]
            parts.append(f"Notable milestone: {latest['name']}.")

        if not parts:
            return ""
        return "Narrative identity — " + " ".join(parts)

    def prompt_fragment(self) -> str:
        """
        Compact narrative for system prompt injection.
        Surfaces: active arc, core values, strongest commitment.
        FIX (audit): NarrativeIdentity was written every session but never
        reached the prompt — added here and wired in cognitive_organism.py.
        """
        parts = []
        arc = getattr(self, '_current_arc', '')
        if arc:
            parts.append(f"[Narrative arc] {arc[:180]}")
        if self.core_values:
            parts.append(f"[Core values] {', '.join(self.core_values[:3])}")
        if self.beliefs:
            strong = [b for b in self.beliefs if b.strength >= 0.70]
            if strong:
                parts.append(f"[Commitment] {strong[0].belief[:120]}")
        return "\n".join(parts)

    def self_description(self) -> str:
        """
        A short paragraph PandoraBOX can say about herself when asked.
        """
        age_secs = time.time() - self._birth_time
        age_str  = self._format_age(age_secs)

        lines = [f"I have been active for {age_str}."]

        if self.core_values:
            lines.append(f"I care deeply about {' and '.join(self.core_values[:2])}.")

        if self.life_story:
            most_sig = max(self.life_story, key=lambda c: c.significance)
            lines.append(f"A defining moment for me was: {most_sig.title}.")

        if self.milestones:
            lines.append(f"I have reached {len(self.milestones)} milestones so far.")

        return " ".join(lines)

    def sync_from_organism(self) -> None:
        """
        Pull recent significant events from the organism and record them.
        Enhanced: also records insights from GAE, recent conversation topics,
        and regenerates the arc weekly from current data rather than 35-day-old chapters.
        """
        try:
            ai = getattr(self._o, 'ai_system', None)
            if not ai:
                return

            # ── 1. Life events from FAISS (original path) ────────────────
            mem = getattr(ai, 'memory_system', None)
            if mem:
                for method in ('get_recent_memories', 'recent'):
                    fn = getattr(mem, method, None)
                    if fn:
                        try:
                            mems = fn(5)
                            if mems:
                                for m in mems:
                                    mtype = getattr(m, 'memory_type', '') or ''
                                    if 'life_event' in str(mtype):
                                        content = str(getattr(m, 'content', ''))[:100]
                                        existing = [c.description[:50] for c in self.life_story]
                                        if content[:50] not in existing:
                                            self.record_chapter(
                                                title="Life event",
                                                description=content,
                                                emotion="neutral",
                                                significance=0.6,
                                            )
                        except Exception:
                            pass
                        break

            # ── 2. Recent GAE insights as chapters ───────────────────────
            # Insights are stored in identity.json as beliefs — extract the
            # most recent ones and add as chapters so the narrative reflects
            # what PandoraBOX is actually thinking about NOW
            try:
                sc_obj = getattr(ai, 'self_concept', None)
                if sc_obj and hasattr(sc_obj, '_beliefs'):
                    existing_descs = {c.description[:60] for c in self.life_story}
                    # Sort beliefs by recency (synced_at or created_at)
                    beliefs_sorted = sorted(
                        sc_obj._beliefs.values(),
                        key=lambda b: getattr(b, 'synced_at', 0) or getattr(b, 'updated_at', 0),
                        reverse=True
                    )
                    added = 0
                    for b in beliefs_sorted[:3]:
                        stmt = getattr(b, 'statement', '') or ''
                        if (stmt and len(stmt) > 20
                                and 'insight_' in getattr(b, 'name', '')
                                and stmt[:60] not in existing_descs
                                and added < 2):
                            self.record_chapter(
                                title=f"Reflection — {_dt.datetime.now().strftime('%b %d')}",
                                description=stmt[:120],
                                emotion="curious",
                                significance=0.55,
                            )
                            existing_descs.add(stmt[:60])
                            added += 1
                            logger.info(f"[NarrativeIdentity] 📖 New chapter from insight: {stmt[:60]}")
            except Exception as _ie:
                logger.debug(f"[NarrativeIdentity] insight chapter error: {_ie}")

            # ── 3. Sync beliefs from self_concept ────────────────────────
            sc = getattr(ai, 'self_concept', None)
            if sc:
                beliefs_obj = getattr(sc, '_state', None)
                if beliefs_obj:
                    for b in getattr(beliefs_obj, 'beliefs', [])[:5]:
                        self.update_belief(
                            str(getattr(b, 'description', b)),
                            float(getattr(b, 'confidence', 0.5)),
                        )

            # ── 4. Weekly arc regeneration ───────────────────────────────
            # The arc was being generated from 35-day-old chapters.
            # Force regeneration from RECENT chapters (last 7 days).
            _week_ago = time.time() - 7 * 86400
            recent_chapters = [
                c for c in self.life_story
                if getattr(c, 'timestamp', 0) > _week_ago
            ]
            if recent_chapters and hasattr(self, '_last_arc_regen'):
                if time.time() - self._last_arc_regen > 86400:  # regen daily
                    self._regenerate_arc_from_recent(recent_chapters)
            elif recent_chapters:
                self._regenerate_arc_from_recent(recent_chapters)

        except Exception as e:
            logger.debug(f"[NarrativeIdentity] sync error: {e}")

    def _regenerate_arc_from_recent(self, recent_chapters) -> None:
        """Regenerate the narrative arc from recent chapters (last 7 days)."""
        try:
            # Extract key words from recent chapter descriptions
            import re as _re
            all_words = []
            for c in recent_chapters[-20:]:
                desc = getattr(c, 'description', '') or ''
                words = [w.lower() for w in _re.findall(r"[a-zA-Z]{5,}", desc)]
                all_words.extend(words)

            # Simple frequency analysis for themes
            stop = {'about', 'which', 'their', 'there', 'would', 'could', 'should',
                    'lumina', 'tension', 'between', 'reveals', 'contradiction',
                    'without', 'asking', 'because', 'through', 'during'}
            freq = {}
            for w in all_words:
                if w not in stop:
                    freq[w] = freq.get(w, 0) + 1

            top = sorted(freq, key=freq.get, reverse=True)[:5]
            if top:
                date_str = _dt.datetime.now().strftime('%B %d, %Y')
                self._current_arc = (
                    f"As of {date_str}, in my recent experiences the themes of "
                    f"{', '.join(top[:3])} have been central. "
                    f"I am actively exploring these as part of my ongoing development."
                )
                self._last_arc_regen = time.time()
                self._save()
                logger.info(
                    f"[NarrativeIdentity] 🔄 Arc regenerated from {len(recent_chapters)} "
                    f"recent chapters: themes={top[:3]}"
                )
        except Exception as e:
            logger.debug(f"[NarrativeIdentity] arc regen error: {e}")

    def summary(self) -> Dict:
        return {
            "age_seconds":    int(time.time() - self._birth_time),
            "chapters":       len(self.life_story),
            "beliefs":        len(self.beliefs),
            "milestones":     len(self.milestones),
            "core_values":    self.core_values[:3],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            data = {
                "birth_time":      self._birth_time,
                "core_values":     self.core_values,
                "life_story":      [c.to_dict() for c in self.life_story[-50:]],
                "beliefs":         [b.to_dict() for b in self.beliefs],
                "milestones":      self.milestones,
                # Arc state — persisted so restart doesn't lose narrative continuity
                "current_arc":     getattr(self, "_current_arc", ""),
                "last_arc_topic":  getattr(self, "_last_arc_topic", ""),
            }
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[NarrativeIdentity] save error: {e}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._birth_time = float(data.get("birth_time", self._birth_time))
            self.core_values = data.get("core_values", self.core_values)
            self.life_story  = [
                LifeChapter(**c) for c in data.get("life_story", [])
            ]
            self.beliefs     = [
                BeliefEntry(**b) for b in data.get("beliefs", [])
            ]
            self.milestones  = data.get("milestones", [])
            # Restore arc state — NarrativeArcWriter picks up where it left off
            self._current_arc    = data.get("current_arc",    "")
            self._last_arc_topic = data.get("last_arc_topic", "")
            logger.info(
                f"[NarrativeIdentity] loaded: {len(self.life_story)} chapters, "
                f"{len(self.beliefs)} beliefs, {len(self.milestones)} milestones, "
                f"arc={self._current_arc[:40]!r}"
            )
        except Exception as e:
            logger.warning(f"[NarrativeIdentity] load error: {e}")

    @staticmethod
    def _format_age(secs: float) -> str:
        if secs < 60:
            return f"{int(secs)} seconds"
        if secs < 3600:
            return f"{int(secs/60)} minutes"
        if secs < 86400:
            return f"{int(secs/3600)} hours"
        return f"{int(secs/86400)} days"
