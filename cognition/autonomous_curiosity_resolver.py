"""
cognition/autonomous_curiosity_resolver.py  (Patch 05)

Converts high-priority curiosity topics into GoalEngine goals so
GoalActionExecutor can pursue them via self_question actions.
"""
from __future__ import annotations
import logging, threading, time
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)

RESOLVE_CHECK_EVERY_N   = 10
CURIOSITY_THRESHOLD     = 0.55
MAX_GOALS_FROM_CURIOSITY = 2
GOAL_PRIORITY_BASE      = 0.60

# A promoted topic must be a coherent phrase, not a 2-word conversation
# fragment. Fragments like 'are'nt turning' / 'sharing today' / 'thought
# wonder' were piling up as garbage goals (they score high curiosity because
# they repeat in chat, but they are not real topics). Require a minimum word
# count AND a minimum number of content (non-stop) words before a curiosity
# topic is allowed to become a goal.
MIN_TOPIC_WORDS         = 4
MIN_TOPIC_CONTENT_WORDS = 3

# Function words + generic introspection vocabulary — excluded when measuring
# content-word overlap for duplicate detection.
_RESOLVER_STOP = {
    'and', 'the', 'for', 'with', 'that', 'this', 'from', 'into', 'through',
    'about', 'have', 'more', 'will', 'been', 'your', 'our', 'their', 'its',
    'are', 'was', 'were', 'has', 'had', 'not', 'but', 'can', 'may', 'how',
    'all', 'any', 'each', 'both', 'such', 'than', 'then', 'when', 'what',
    'something', 'anything', 'nothing', 'true', 'know', 'think', 'believe',
    'assume', 'assuming', 'belief', 'beliefs', 'curiosity', 'question',
    'questions', 'answer', 'answers', 'thinking', 'thoughts',
}


class AutonomousCuriosityResolver:
    def __init__(self, organism: Any) -> None:
        self._organism = organism
        self._lock     = threading.Lock()
        self._promoted: Set[str] = set()
        logger.info("[CuriosityResolver] Initialised")

    def tick(self, slow_cycle: int) -> None:
        if slow_cycle % RESOLVE_CHECK_EVERY_N != 0 or slow_cycle == 0:
            return
        threading.Thread(target=self._check_and_promote, args=(slow_cycle,),
                         daemon=True, name="acr-promote").start()

    def _check_and_promote(self, slow_cycle: int) -> None:
        try:
            o    = self._organism
            gate = getattr(o, "behavior_gate", None)
            if gate and not gate.may_run_resolution():
                return
            ce = getattr(o, "curiosity", None)
            if not ce:
                return

            topics: List[tuple] = []
            try:
                with ce._lock:
                    for topic, node in ce._topics.items():
                        if node.curiosity >= CURIOSITY_THRESHOLD:
                            topics.append((topic, node.curiosity))
                topics.sort(key=lambda x: -x[1])
                topics = topics[:MAX_GOALS_FROM_CURIOSITY * 2]
            except Exception as _ce_e:
                logger.debug(f"[CuriosityResolver] topic read: {_ce_e}")
                return

            if not topics:
                return

            ai = getattr(o, "ai_system", None)
            ge = getattr(ai, "goal_engine", None) if ai else None
            if not ge:
                return

            existing: Set[str] = set()
            try:
                for g in ge._goals.values():
                    existing.add(g.topic.lower().strip())
            except Exception:
                pass

            promoted = 0
            rejected_fragments = 0
            for topic, curiosity_val in topics:
                if promoted >= MAX_GOALS_FROM_CURIOSITY:
                    break
                if topic in self._promoted:
                    continue
                if not self._is_valid_topic(topic):
                    rejected_fragments += 1
                    logger.debug(
                        "[CuriosityResolver] Skipping incoherent fragment "
                        "(not a real topic): " + repr(topic)
                    )
                    continue
                if self._is_duplicate(topic, existing):
                    continue
                self._create_goal(ge, topic, curiosity_val, slow_cycle)
                with self._lock:
                    self._promoted.add(topic)
                # Block a second near-duplicate in the same tick
                existing.add(topic.lower().strip())
                promoted += 1

            if promoted:
                logger.info(f"[CuriosityResolver] Promoted {promoted} topics to goals (cycle={slow_cycle})")
            if rejected_fragments:
                logger.info(
                    f"[CuriosityResolver] Rejected {rejected_fragments} "
                    f"fragment topic(s) as incoherent (cycle={slow_cycle})"
                )
        except Exception as e:
            logger.debug(f"[CuriosityResolver] error: {e}")

    def _is_valid_topic(self, topic: str) -> bool:
        """True only if the topic is a coherent phrase worth turning into a goal.

        Blocks 2-word conversation fragments (e.g. 'are'nt turning',
        'sharing today', 'thought wonder', 'sorry repeat') that were piling up
        as garbage goals. A valid topic needs at least MIN_TOPIC_WORDS words
        AND MIN_TOPIC_CONTENT_WORDS content (non-stop, alphabetic) words.
        """
        raw = (topic or "").split()
        if len(raw) < MIN_TOPIC_WORDS:
            return False
        content = set()
        for w in raw:
            w = w.strip(".,!?;:()[]{}").lower()
            if len(w) > 3 and w.isalpha() and w not in _RESOLVER_STOP:
                content.add(w)
        return len(content) >= MIN_TOPIC_CONTENT_WORDS

    # Calibrated dedup threshold (QUALITY tier / nomic): near-duplicate goal
    # topics score >= 0.643, distinct topics <= 0.550 (measured gap ~0.09).
    # 0.60 sits comfortably inside that gap.
    DEDUP_COSINE_THRESHOLD = 0.60

    def _is_duplicate(self, candidate: str, existing_topics) -> bool:
        """True if candidate is a duplicate/near-duplicate of an existing goal.

        Primary signal: semantic similarity via the QUALITY-tier embedder
        (nomic-embed-text) - robust to rewording and truncation. If the
        embedder/API is unavailable, falls back to the legacy word-overlap
        heuristic so dedup still works offline.
        """
        cand = (candidate or "").lower().strip()
        topics = [str(t) for t in (existing_topics or []) if t and str(t).strip()]
        if not cand or not topics:
            return False
        # Fast exact-match
        if cand in {t.lower().strip() for t in topics}:
            return True
        # Semantic similarity (QUALITY tier, single batched call)
        try:
            import numpy as np
            from utils.shared_embedder import get_embedder
            emb = get_embedder(quality=True)
            vecs = np.asarray(
                emb.encode([candidate] + topics), dtype="float32")
            cand_vec = vecs[0]
            others = vecs[1:]
            cn = float(np.linalg.norm(cand_vec)) + 1e-9
            on = np.linalg.norm(others, axis=1) + 1e-9
            sims = (others @ cand_vec) / (on * cn)
            if len(sims) and float(np.max(sims)) >= self.DEDUP_COSINE_THRESHOLD:
                return True
            return False
        except Exception as e:
            logger.debug(
                "[CuriosityResolver] semantic dedup unavailable (%s); "
                "using word-overlap" % e)
            return self._is_duplicate_word_overlap(cand, topics)

    def _is_duplicate_word_overlap(self, cand: str, topics) -> bool:
        """Legacy word-overlap dedup (offline fallback)."""
        def content_words(s: str):
            return {w for w in s.split()
                    if len(w) > 3 and w.isalpha()
                    and w not in _RESOLVER_STOP}
        cw = content_words(cand)
        if not cw:
            return True  # no content words -> nothing specific -> skip
        for ex in topics:
            ew = content_words(ex)
            if (cand in ex or ex in cand) and min(len(cand), len(ex)) > 12:
                return True
            if len(cw & ew) >= 2:
                return True
            if len(cw) == 1 and next(iter(cw)) in ew:
                return True
        return False

    def _create_goal(self, goal_engine: Any, topic: str,
                     curiosity: float, slow_cycle: int) -> None:
        try:
            from cognition.goal_engine import Goal
            import uuid
            priority = round(min(0.80, GOAL_PRIORITY_BASE + (curiosity - CURIOSITY_THRESHOLD) * 0.3), 3)
            goal = Goal(
                id            = f"curiosity_{uuid.uuid4().hex[:8]}",
                topic         = topic,
                priority      = priority,
                persistence   = 0,
                origin        = f"curiosity_autonomous_c{slow_cycle}",
                energy        = 0.8,
                created_cycle = slow_cycle,
                last_active   = slow_cycle,
            )
            goal_engine.add_goal(goal)
            logger.debug(f"[CuriosityResolver] Goal created: '{topic}' p={priority:.2f}")
        except Exception as e:
            logger.debug(f"[CuriosityResolver] _create_goal error '{topic}': {e}")
