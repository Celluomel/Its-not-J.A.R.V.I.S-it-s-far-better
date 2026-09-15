"""
Research Activation Controller (RAC)

Observes conversation signals throughout the day and decides:
- Whether to schedule Mode 3 (Background Research)
- Which activation pattern to use (user-driven / deep-day / curiosity-drift)

Does NOT run research directly — it queues jobs for the controller.
Thread-safe (called from async NiceGUI context).
"""
from __future__ import annotations
import datetime
import json
import logging
import os
import threading
from typing import Optional, List, Tuple

from cognition.research_mcp.schemas import DailyMetrics, ResearchMode

logger = logging.getLogger("research_mcp.rac")

METRICS_PATH = "data/persona/rac_metrics.json"

# ── Thresholds ────────────────────────────────────────────────────────────────
COMPLEXITY_THRESHOLD    = 0.65   # daily complexity score → schedule Mode 3
TOPIC_REPEAT_THRESHOLD  = 3      # same topic N times → enrich background
CONFIDENCE_GAP_LIMIT    = 4      # N low-confidence events → schedule enrichment
MAX_BACKGROUND_PER_DAY  = 2      # hard limit on Mode 3 runs per day

# Words that signal a complex query (Mode 2 auto-trigger)
_COMPLEXITY_SIGNALS = [
    "analyze", "analyse", "compare", "explain why", "what causes",
    "future of", "history of", "impact of", "implications", "pros and cons",
    "difference between", "latest", "recent", "breakthrough", "deep dive",
    "comprehensive", "research", "evidence", "scientific",
]


class ResearchActivationController:

    def __init__(self):
        self._lock    = threading.Lock()
        self._metrics = self._load_metrics()
        self._pending_mode3: List[Tuple[str, str]] = []  # (goal, trigger_reason)

    # ── Mode 2 auto-detection ─────────────────────────────────────────────────
    def should_extend_answer(self, user_input: str) -> bool:
        """
        Returns True if the input is complex enough to warrant Mode 2 (Extended).
        Called from persona_bridge before generating a response.
        """
        text = user_input.lower()
        tokens = len(text.split())
        signal_count = sum(1 for s in _COMPLEXITY_SIGNALS if s in text)
        is_question  = "?" in user_input or text.startswith(("what", "why", "how", "who", "when", "where", "explain"))

        score = 0.0
        if tokens > 25:         score += 0.3
        if signal_count >= 1:   score += 0.25 * min(signal_count, 2)
        if is_question:         score += 0.15

        logger.debug(f"RAC Mode2 score={score:.2f} for: {user_input[:60]!r}")
        return score >= 0.55

    # ── Daily signal ingestion ────────────────────────────────────────────────
    def record_query(self, user_input: str, low_confidence: bool = False):
        """
        Called after every user query.
        Updates daily metrics and queues Mode 3 if thresholds crossed.
        """
        with self._lock:
            self._refresh_day()
            m = self._metrics
            m.total_queries += 1

            # complexity
            text   = user_input.lower()
            tokens = len(text.split())
            sigs   = sum(1 for s in _COMPLEXITY_SIGNALS if s in text)
            if tokens > 25 or sigs >= 2:
                m.complex_queries += 1

            # topic heat
            topic = self._extract_topic(user_input)
            if topic:
                m.topic_counts[topic] = m.topic_counts.get(topic, 0) + 1

            if low_confidence:
                m.confidence_gaps += 1

            # decide on Mode 3
            self._evaluate_and_queue(user_input, topic)
            self._save_metrics(m)

    def record_explicit_interest(self, topic: str):
        """User directly asked to research something — fast-track Mode 3."""
        with self._lock:
            self._refresh_day()
            if self._metrics.research_today < MAX_BACKGROUND_PER_DAY:
                self._pending_mode3.append((topic, "explicit_user_interest"))
                logger.info(f"RAC: explicit interest queued Mode 3 for: {topic!r}")

    # ── Mode 3 job queue ──────────────────────────────────────────────────────
    def pop_pending_mode3(self) -> Optional[Tuple[str, str]]:
        """Returns (goal, trigger_reason) or None. Called by background scheduler."""
        with self._lock:
            if self._pending_mode3:
                return self._pending_mode3.pop(0)
            return None

    def has_pending_mode3(self) -> bool:
        with self._lock:
            return bool(self._pending_mode3)

    def mark_mode3_ran(self):
        """Call after a Mode 3 run completes."""
        with self._lock:
            self._refresh_day()
            self._metrics.research_today += 1
            self._save_metrics(self._metrics)

    # ── Daily summary ─────────────────────────────────────────────────────────
    def daily_summary(self) -> dict:
        with self._lock:
            m = self._metrics
            return {
                "date":              m.date,
                "total_queries":     m.total_queries,
                "complex_queries":   m.complex_queries,
                "confidence_gaps":   m.confidence_gaps,
                "research_today":    m.research_today,
                "top_topics":        sorted(m.topic_counts.items(), key=lambda x: -x[1])[:5],
                "pending_mode3":     len(self._pending_mode3),
                "complexity_score":  round(self._complexity_score(m), 3),
            }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _evaluate_and_queue(self, user_input: str, topic: str):
        m = self._metrics
        if m.research_today >= MAX_BACKGROUND_PER_DAY:
            return

        score = self._complexity_score(m)

        # Pattern A: high-intensity day
        if score >= COMPLEXITY_THRESHOLD:
            goal = self._goal_from_topic(topic or user_input)
            if not self._already_queued(goal):
                self._pending_mode3.append((goal, "deep_day"))
                logger.info(f"RAC: deep_day → queued Mode 3 for: {goal!r}")
            return

        # Pattern B: topic repeated
        if topic and m.topic_counts.get(topic, 0) >= TOPIC_REPEAT_THRESHOLD:
            goal = self._goal_from_topic(topic)
            if not self._already_queued(goal):
                self._pending_mode3.append((goal, "topic_interest"))
                logger.info(f"RAC: topic_interest → queued Mode 3 for: {goal!r}")
            return

        # Pattern C: persistent confidence gaps
        if m.confidence_gaps >= CONFIDENCE_GAP_LIMIT:
            goal = f"fill knowledge gaps: {topic or user_input[:60]}"
            if not self._already_queued(goal):
                self._pending_mode3.append((goal, "confidence_drift"))
                logger.info(f"RAC: confidence_drift → queued Mode 3 for: {goal!r}")

    def _complexity_score(self, m: DailyMetrics) -> float:
        if m.total_queries == 0:
            return 0.0
        cq = m.complex_queries / m.total_queries
        rep= min(max(m.topic_counts.values(), default=0) / 5.0, 1.0)
        unc= min(m.confidence_gaps / 6.0, 1.0)
        return round(cq * 0.4 + rep * 0.3 + unc * 0.3, 3)

    def _already_queued(self, goal: str) -> bool:
        return any(g.lower() == goal.lower() for g, _ in self._pending_mode3)

    # Patterns that indicate an internal LLM prompt was passed as user_input
    # instead of an actual user query. These should not be logged as research topics.
    _INTERNAL_PROMPT_MARKERS = (
        "you are feeling", "without naming the emotion", "let it colour",
        "one sentence only", "you just had this inner thought",
        "naturally surface", "speak naturally", "you are lumina",
        "system prompt", "inner voice",
    )

    @staticmethod
    def _is_internal_prompt(text: str) -> bool:
        """True if text looks like an internal LLM prompt, not a user query."""
        t = text.lower()
        return any(marker in t for marker in
                   ResearchActivationController._INTERNAL_PROMPT_MARKERS)

    @staticmethod
    def _extract_topic(text: str) -> str:
        # Reject internal prompts — they pollute rac_metrics.json with
        # entries like "you_are_feeling_enthusiasm_without_naming".
        if ResearchActivationController._is_internal_prompt(text):
            return ""
        words = text.lower().split()
        stop  = {"the","a","an","of","in","on","for","and","or","to","is","are",
                 "what","how","why","tell","me","about","you","your","have","been",
                 "that","this","with","from","just","feel","said","very","like"}
        _punct = str.maketrans("", "", "'.,!?")
        keywords = [w.translate(_punct) for w in words
                    if len(w) > 3 and w.lower().translate(_punct) not in stop
                    and w.lower().translate(_punct)][:4]
        return "_".join(keywords) if keywords else ""

    @staticmethod
    def _goal_from_topic(topic: str) -> str:
        return f"Research and summarize: {topic.replace('_', ' ')}"

    def _refresh_day(self):
        today = datetime.date.today().isoformat()
        if self._metrics.date != today:
            self._metrics = DailyMetrics(date=today)

    def _load_metrics(self) -> DailyMetrics:
        try:
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)
            today = datetime.date.today().isoformat()
            if data.get("date") == today:
                m = DailyMetrics()
                m.__dict__.update(data)
                return m
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            pass
        return DailyMetrics()

    def _save_metrics(self, m: DailyMetrics):
        try:
            os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
            with open(METRICS_PATH, "w") as f:
                json.dump(m.__dict__, f, indent=2)
        except Exception as e:
            logger.warning(f"RAC: failed to save metrics: {e}")
