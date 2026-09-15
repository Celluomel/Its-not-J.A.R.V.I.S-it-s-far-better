"""
MemoryIntegrator — writes completed Mode 3 research into:
  1. Research Journal  (data/persona/research_journal.json)
  2. Lumina's long-term memory  (via memory_system.add_memory)

Only called for Mode 3 (BACKGROUND) sessions, and only AFTER summarization.
Never stores raw HTML or page dumps — only structured summaries.
"""
from __future__ import annotations
import datetime
import json
import logging
import os
import threading
from typing import Optional, Any

from cognition.research_mcp.schemas import ResearchSession, ResearchStatus

logger = logging.getLogger("research_mcp.memory_integrator")

JOURNAL_PATH = "data/persona/research_journal.json"
MAX_JOURNAL_ENTRIES = 200    # rotate oldest when exceeded
CONFIDENCE_THRESHOLD = 0.45  # don't store if too uncertain


class MemoryIntegrator:

    def __init__(self, memory_system: Optional[Any] = None):
        """
        memory_system: EnhancedMemorySystem from ai_system.py (optional).
        If None, only the journal is written.
        """
        self._memory = memory_system
        self._lock   = threading.Lock()
        self._ensure_journal()

    # ── Public API ────────────────────────────────────────────────────────────
    def integrate(self, session: ResearchSession) -> int:
        """
        Write session to journal and Lumina memory.
        Returns number of knowledge nodes created.
        """
        if session.confidence_score < CONFIDENCE_THRESHOLD:
            logger.info(
                f"MemoryIntegrator: skipping low-confidence result "
                f"({session.confidence_score:.2f} < {CONFIDENCE_THRESHOLD})"
            )
            return 0

        nodes = 0

        # 1. Journal entry
        entry = self._build_entry(session)
        self._write_journal(entry)
        nodes += 1

        # 2. Lumina semantic memory (summary embedding)
        if self._memory and session.summary:
            try:
                emo = self._memory.analyze_emotional_context(session.summary)
                ok  = self._memory.add_memory(
                    text             = f"[Research knowledge] {session.goal}: {session.summary}",
                    impact_score     = min(0.95, 0.55 + session.confidence_score * 0.4),
                    memory_type      = "research",
                    emotional_valence= emo.get("valence", "Neutral"),
                    arousal_level    = emo.get("arousal", "Medium"),
                )
                if ok:
                    nodes += 1
                    logger.info(f"MemoryIntegrator: stored summary in Lumina memory")
            except Exception as e:
                logger.error(f"MemoryIntegrator: Lumina memory write failed: {e}")

        # 3. Per-fact memory nodes (high confidence only)
        if session.confidence_score >= 0.70 and self._memory:
            nodes += self._store_key_facts(session)

        session.knowledge_nodes = nodes
        logger.info(f"MemoryIntegrator: {nodes} knowledge nodes created for {session.research_id}")
        return nodes

    # ── Journal ───────────────────────────────────────────────────────────────
    def get_journal(self, limit: int = 20) -> list:
        """Return recent journal entries (newest first)."""
        journal = self._load_journal()
        return list(reversed(journal[-limit:]))

    def get_entry(self, research_id: str) -> Optional[dict]:
        journal = self._load_journal()
        for entry in reversed(journal):
            if entry.get("research_id") == research_id:
                return entry
        return None

    def get_by_topic(self, topic: str, limit: int = 10) -> list:
        """Find journal entries whose topic or influences_topics contains keyword."""
        journal = self._load_journal()
        topic_l = topic.lower()
        matches = [
            e for e in journal
            if topic_l in e.get("topic", "").lower()
            or any(topic_l in t.lower() for t in e.get("influences_topics", []))
        ]
        return list(reversed(matches[-limit:]))

    def has_new_research_since(self, iso_timestamp: str) -> list:
        """Return entries created after iso_timestamp."""
        journal = self._load_journal()
        try:
            since = datetime.datetime.fromisoformat(iso_timestamp)
            return [
                e for e in journal
                if datetime.datetime.fromisoformat(e.get("finished_at", "2000-01-01")) > since
            ]
        except Exception:
            return []

    # ── Private ───────────────────────────────────────────────────────────────
    def _build_entry(self, session: ResearchSession) -> dict:
        return {
            "research_id":      session.research_id,
            "trigger_reason":   session.trigger_reason,
            "topic":            self._extract_topic(session.goal),
            "goal":             session.goal,
            "summary":          session.summary,
            "sources":          session.sources,
            "confidence_score": round(session.confidence_score, 3),
            "knowledge_nodes":  session.knowledge_nodes,
            "finished_at":      session.finished_at or datetime.datetime.now().isoformat(),
            "started_at":       session.started_at,
            "influences_topics": session.influences_topics,
            "steps_taken":      len(session.steps),
            "pages_visited":    sum(len(s.pages) for s in session.steps),
        }

    def _write_journal(self, entry: dict):
        with self._lock:
            journal = self._load_journal()
            journal.append(entry)
            if len(journal) > MAX_JOURNAL_ENTRIES:
                journal = journal[-MAX_JOURNAL_ENTRIES:]
            self._save_journal(journal)

    def _store_key_facts(self, session: ResearchSession) -> int:
        """Parse summary for bullet points and store each as a memory node."""
        count = 0
        lines = session.summary.split("\n")
        for line in lines:
            line = line.strip().lstrip("•-*").strip()
            if len(line) > 40 and not line.endswith(":"):
                try:
                    ok = self._memory.add_memory(
                        text         = f"[Research fact — {session.goal[:60]}] {line}",
                        impact_score = 0.65,
                        memory_type  = "research_fact",
                        emotional_valence = "Neutral",
                        arousal_level     = "Low",
                    )
                    if ok:
                        count += 1
                    if count >= 5:  # cap
                        break
                except Exception:
                    pass
        return count

    def _extract_topic(self, goal: str) -> str:
        """Derive a short topic label from the goal."""
        words = goal.lower().split()[:6]
        stop  = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are"}
        return "_".join(w for w in words if w not in stop)[:50]

    def _ensure_journal(self):
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        if not os.path.exists(JOURNAL_PATH):
            self._save_journal([])

    def _load_journal(self) -> list:
        try:
            with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_journal(self, journal: list):
        with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(journal, f, indent=2, ensure_ascii=False)
