"""
Goal Action Executor
====================
Closes the action feedback loop by translating active goals into concrete
cognitive actions that run autonomously between user interactions.

Four action types, chosen via per-goal rotation so all types get exercised:

  MEMORY_RECALL — scan semantic memory for goal-related concepts using
                  keyword extraction (not full topic string)
  SELF_QUESTION — ask the LLM a focused question about the goal topic,
                  store the answer as a belief/thought; result queued for
                  proactive delivery to user chat
  WEB_SEARCH    — query a search engine for goal-relevant information
  USER_QUESTION — inject a proactive question into the chat (max 2/session)

Fixes applied vs previous version:
  FIX 1 — Wrong method: get_related_concepts() → get_related(keyword)
           with keyword extraction from long topic strings
  FIX 2 — _choose_action always returned memory_recall (SemanticMemory
           always present). Now uses per-goal rotation index so all four
           action types get used. Rotation advances unconditionally.
  FIX 3 — _fire() used asyncio.ensure_future() from daemon thread with no
           running loop → coroutine garbage-collected immediately. Now spawns
           a dedicated daemon thread and calls asyncio.run() which creates a
           fresh event loop guaranteed to execute the coroutine.
  FIX 4 — mark_action() never called (memory_recall always returned
           success=False). Now: async actions mark immediately (they started);
           memory_recall falls through to self_question in the same tick so
           the cycle always produces a result.
  FIX 5 — Post-insight curiosity stimulation and energy decay added so the
           snowball loop works and goals don't dominate forever.
  FIX 6 — Meta-loop broken by Bisociative Collision Engine.
           Root cause: SELF_QUESTION produced questions stored as beliefs
           (priority 0.55) — CognitiveDissonanceEngine never reacted because
           questions don't constitute new facts. WEB_SEARCH queried abstract
           keywords like "understanding" extracted from the goal topic, yielding
           low-information results that fed the next SELF_QUESTION, closing the
           tautology loop.
           Fix: ACTION_BISOCIATIVE runs after SELF_QUESTION. It calls
           get_chaos_episodic() — the maximally dissimilar memory from FAISS —
           and collides it with the current tension via bisociative prompt.
           Output is pushed as a BELIEF at priority 0.85 (vs 0.55), forcing
           CDE to process it as a new fact. Hypothesis keywords are cached in
           _last_hypothesis_text[goal.id] so the NEXT WEB_SEARCH queries
           something concrete (e.g. "impressionist light proximity colour")
           instead of "understanding".
"""

import asyncio
import json
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Action types ───────────────────────────────────────────────────────────────
ACTION_WEB_SEARCH    = "web_search"
ACTION_USER_QUESTION = "user_question"
ACTION_MEMORY_RECALL = "memory_recall"
ACTION_SELF_QUESTION = "self_question"
ACTION_BISOCIATIVE   = "bisociative_hypothesis"   # FIX 6: collision engine

# Rotation order per goal — ensures all types get exercised.
# ORDER MATTERS:
#   BISOCIATIVE precedes WEB_SEARCH so the hypothesis it generates seeds
#   _last_hypothesis_text[goal.id], giving WEB_SEARCH a specific query
#   instead of the abstract goal topic words.
_ACTION_ROTATION = [
    ACTION_MEMORY_RECALL,
    ACTION_SELF_QUESTION,
    ACTION_BISOCIATIVE,     # collision → specific hypothesis keywords
    ACTION_WEB_SEARCH,      # uses hypothesis keywords, not abstract topic
    ACTION_SELF_QUESTION,
    ACTION_USER_QUESTION,
]

# Web search cooldown per goal — prevents hammering the same query
WEB_SEARCH_COOLDOWN = 900   # 15 minutes per goal

# Bisociative engine cooldown — collision is expensive and disorienting if overused
BISOCIATIVE_COOLDOWN = 1200  # 20 minutes per goal

# Cooldown between actions on the same goal (seconds)
GOAL_ACTION_COOLDOWN = 300   # 5 minutes

# Max proactive user questions per session
MAX_USER_QUESTIONS_PER_SESSION = 8   # raised — old limit of 2 meant silence for 90%+ of session
USER_QUESTION_TOPIC_COOLDOWN   = 3600  # 1h before same topic is asked again

# Insight deduplication window (seconds) — skip LLM if same insight stored recently
INSIGHT_DEDUPE_WINDOW = 600   # 10 minutes — 30min was too long with GoalConsolidator recycling same topics every 18min

# Stop-words for keyword extraction from goal topics
_STOP_WORDS = {
    'and', 'the', 'for', 'with', 'that', 'this', 'from', 'into', 'through',
    'about', 'have', 'more', 'will', 'been', 'your', 'our', 'their', 'its',
    'are', 'was', 'were', 'has', 'had', 'not', 'but', 'can', 'may', 'how',
    'all', 'any', 'each', 'both', 'such', 'than', 'then', 'when', 'what',
    'actively', 'pursue', 'explore', 'build', 'develop', 'enhance', 'improve',
    'understand', 'deepen', 'strengthen', 'resolve', 'clarify', 'identify',
    'research', 'reduce', 'meaningful', 'genuine', 'toward', 'between',
}

# Prompt rotation templates per category
_SELF_QUESTION_FRAMES = [
    "In one sentence, what does '{topic}' mean for my sense of self?",
    "In one sentence, what belief of mine is most challenged by '{topic}'?",
    "In one sentence, how does '{topic}' connect to what I value most?",
    "In one sentence, what would fully committing to '{topic}' require of me?",
    "In one sentence, what is the most interesting thing to understand about '{topic}'?",
]

_INSIGHT_NLP = None
_INSIGHT_NLP_ATTEMPTED = False


def _has_grammatical_predicate(text: str) -> Optional[bool]:
    """Return whether spaCy finds a verb/auxiliary, or None if unavailable."""
    global _INSIGHT_NLP, _INSIGHT_NLP_ATTEMPTED
    if not _INSIGHT_NLP_ATTEMPTED:
        _INSIGHT_NLP_ATTEMPTED = True
        try:
            import spacy
            _INSIGHT_NLP = spacy.load(
                'en_core_web_sm', disable=['ner', 'parser', 'lemmatizer']
            )
            logger.info('[GoalActionExecutor] spaCy POS validator ready')
        except Exception as exc:
            logger.debug('[GoalActionExecutor] spaCy POS validator unavailable: %s', exc)
            _INSIGHT_NLP = False
    if not _INSIGHT_NLP:
        return None
    try:
        return any(token.pos_ in {'VERB', 'AUX'} for token in _INSIGHT_NLP(text))
    except Exception as exc:
        logger.debug('[GoalActionExecutor] spaCy POS validation failed: %s', exc)
        return None


class GoalActionExecutor:
    """
    Translates active goals into concrete cognitive actions.

    Integrates with:
      - GoalEngine  (reads active goals, calls mark_action)
      - GlobalWorkspace (broadcasts action results as thoughts)
      - SemanticMemory / FAISSMemory (for memory recall)
      - LLM (for self_question action)
      - Search backend (for web_search action)
      - Proactive message queue (for user_question and insight delivery)
    """

    def __init__(self, organism: Any):
        self._o = organism
        self._last_action_time: Dict[str, float] = {}   # goal_id → timestamp
        self._rotation_index: Dict[str, int] = {}       # goal_id → rotation pos
        self._frame_index: Dict[str, int] = {}          # goal_id → prompt frame
        self._recent_insights: Dict[str, float] = {}    # belief_name → timestamp
        self._last_web_search: Dict[str, float] = {}    # goal_id → last web search
        self._last_bisociative: Dict[str, float] = {}   # goal_id → last bisociative ts
        self._last_hypothesis_text: Dict[str, str] = {} # goal_id → hypothesis keywords for next search
        # Fix: previously in-memory only — reset to {} on every PandoraBOX
        # restart, meaning the 3-insight goal-completion threshold never
        # accumulated across sessions (a goal that got 2 insights, then
        # PandoraBOX restarted, would need 3 MORE insights from scratch).
        # Now persisted to disk and loaded lazily on first use.
        self._goal_insight_counts: Dict[str, int] = {}  # goal_id → total insights generated
        self._insight_counts_path = Path("data/persona/goal_insight_counts.json")
        self._load_insight_counts()
        self._asked_topics: Dict[str, float] = {}       # topic → timestamp last asked
        self._user_questions_sent: int = 0
        self._pending_user_question: Optional[str] = None
        self._pending_insights: deque = deque(maxlen=5)
        self._pending_predictions: Dict[Tuple[str, str], Any] = {}
        logger.info("[GoalActionExecutor] Initialised")

    def _load_insight_counts(self) -> None:
        """Load persisted goal_insight_counts from disk (non-fatal if missing)."""
        try:
            if self._insight_counts_path.exists():
                self._goal_insight_counts = json.loads(
                    self._insight_counts_path.read_text()
                )
                logger.info(
                    f"[GoalActionExecutor] Loaded {len(self._goal_insight_counts)} "
                    f"persisted insight counts"
                )
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] insight_counts load error: {e}")
            self._goal_insight_counts = {}

    def _save_insight_counts(self) -> None:
        """Persist goal_insight_counts to disk. Atomic write — this file is
        small and written frequently (every insight), so torn reads are
        unlikely to matter much here, but the pattern is kept consistent
        with the other dashboard-adjacent JSON writes in this codebase."""
        try:
            self._insight_counts_path.parent.mkdir(parents=True, exist_ok=True)
            _tmp = self._insight_counts_path.with_suffix('.json.tmp')
            _tmp.write_text(json.dumps(self._goal_insight_counts, indent=2))
            _tmp.replace(self._insight_counts_path)
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] insight_counts save error: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_pending_user_question(self) -> Optional[str]:
        """Called by the proactive message system to retrieve a queued question."""
        q = self._pending_user_question
        self._pending_user_question = None
        return q

    def get_pending_insight(self) -> Optional[str]:
        """Called by app.py proactive timer to deliver insights to chat."""
        if self._pending_insights:
            return self._pending_insights.popleft()
        return None

    def run_tick(self) -> Optional[Dict]:
        """
        Select and execute one action for the highest-priority eligible goal.
        Returns a report dict or None if nothing ran.

        FIX 3: async actions (self_question, web_search) now spawn a daemon
        thread with asyncio.run() so the coroutine is guaranteed to execute.
        """
        goal_engine = self._get_goal_engine()
        if not goal_engine:
            logger.debug("[GoalActionExecutor] No GoalEngine found — skipping tick")
            return None

        # Use a lower activation threshold than GoalEngine's default (0.30)
        # so that decayed goals (energy at floor 0.20) remain eligible for GAE.
        # The default threshold was designed for workspace competition, not action selection.
        active = goal_engine.get_active_goals(min_activation=0.10)
        if not active:
            logger.info("[GoalActionExecutor] No eligible goals (all below min_activation=0.10 or none active)")
            return None

        goal = self._select_eligible_goal(active)
        if not goal:
            logger.info(f"[GoalActionExecutor] All {len(active)} eligible goals on {GOAL_ACTION_COOLDOWN}s cooldown — next action in ~{GOAL_ACTION_COOLDOWN//60}min")
            return None

        # Prefer the persistent intention plan. Rotation remains a safe
        # fallback when the long-horizon planner is unavailable.
        action_type = self._planned_action(goal) or self._choose_action_rotating(goal)

        # test_hypothesis goals get temporary priority boost so they compete
        # effectively against generic explore_opportunity goals
        if 'test_hypothesis' in getattr(goal, 'origin', '') and goal.priority < 0.75:
            goal.priority = min(0.80, goal.priority + 0.10)
            logger.debug(f"[GoalActionExecutor] test_hypothesis priority boosted → {goal.priority:.2f}")
            try:
                from cognition.emergence_metrics import get_collector
                c = get_collector()
                if c: c.record_hypothesis_tested()
            except Exception:
                pass

        logger.info(
            f"[GoalActionExecutor] → {action_type.upper()} | "
            f"'{goal.topic[:40]}' (p={goal.priority:.2f} e={getattr(goal,'energy',1.0):.2f})"
        )

        # ── CAUSAL CLOSURE: predict this action's effect BEFORE it fires ─────
        # The prediction is attached to the exact (goal, action) about to run,
        # so the later resolution is a genuine prediction-vs-outcome comparison,
        # not a post-hoc label. Defensive: never lets the loop die.
        _ct_pred = None
        try:
            from cognition.consequence_tracker import get_consequence_tracker
            _ct = get_consequence_tracker(self._o)
            _ct_pred = _ct.predict(
                goal_id    = str(goal.id),
                goal_topic = goal.topic,
                action_type= action_type,
                goal_energy= float(getattr(goal, "energy", 1.0)),
            )
        except Exception as _ct_pe:
            logger.debug(f"[GoalActionExecutor] consequence predict error: {_ct_pe}")
            _ct_pred = None

        # ── WORKSPACE BIAS: does this action FOLLOW the committed focus? ─────
        # Measure the autonomous behaviour against the active commitment.
        # Below threshold => the system overrode its own focus (recorded +
        # costed + counted in the override-rate metric). Defensive: never lets
        # the loop die.
        try:
            from cognition.workspace_bias import get_workspace_bias
            _wb = get_workspace_bias(self._o)
            _ov = _wb.check_override(goal.topic, context=f"action={action_type}")
            if _ov.get("checked") and _ov.get("overridden"):
                logger.info(
                    f"[GoalActionExecutor] ⚠️ OVERRIDES committed focus "
                    f"'{_ov.get('focus_text','')[:30]}' -> '{goal.topic[:30]}' "
                    f"(sim={_ov['similarity']:.2f})"
                )
        except Exception as _wb_oe:
            logger.debug(f"[GoalActionExecutor] workspace-bias override error: {_wb_oe}")

        report: Dict[str, Any] = {
            'goal_id':    goal.id,
            'goal_topic': goal.topic,
            'action':     action_type,
            'success':    False,
            'result':     '',
        }

        try:
            if action_type == ACTION_MEMORY_RECALL:
                result = self._action_memory_recall_with_fallback(goal)
                report.update(result)

            elif action_type == ACTION_USER_QUESTION:
                result = self._action_user_question(goal)
                report.update(result)

            elif action_type in (ACTION_WEB_SEARCH, ACTION_SELF_QUESTION, ACTION_BISOCIATIVE):
                # FIX 3: fire via daemon thread + asyncio.run() — guaranteed execution
                if action_type == ACTION_WEB_SEARCH:
                    coro = self._action_web_search(goal)
                elif action_type == ACTION_BISOCIATIVE:
                    coro = self._action_bisociative_hypothesis(goal)
                else:
                    coro = self._action_self_question(goal)
                if _ct_pred is not None:
                    self._pending_predictions[(str(goal.id), action_type)] = _ct_pred
                self._fire(coro)
                # Dispatch is not an outcome. The coroutine records the plan
                # result only after it has produced usable cognitive output.
                report['success'] = False
                report['result']  = f"{action_type} dispatched (async)"

        except Exception as e:
            logger.warning(f"[GoalActionExecutor] Action failed: {e}", exc_info=True)
            report['result'] = str(e)

        if report['success']:
            # FIX 4: mark_action always called when success=True
            goal_engine.mark_action(goal.id, action_type)
            self._last_action_time[goal.id] = time.time()
            logger.info(
                f"[GoalActionExecutor] ✅ mark_action called for '{goal.topic[:30]}' "
                f"— GEI will update next compute_gei()"
            )
            self._broadcast(
                f"[Action:{action_type}] {goal.topic[:40]} — {report['result'][:80]}"
            )
            self._record_plan_outcome(goal, action_type, True, report['result'])
        elif not report['result'].endswith("dispatched (async)"):
            self._record_plan_outcome(goal, action_type, False, report['result'])

        # ── CAUSAL CLOSURE: resolve the prediction against the REAL outcome ──
        # Computes prediction error and selectively propagates it to
        # goal efficacy (re-weights future selection), one attention module,
        # and the Predictive Consequence Model. Writes the full causal chain
        # to the ledger. Defensive: never lets the loop die.
        try:
            if (_ct_pred is not None
                    and not report['result'].endswith("dispatched (async)")):
                from cognition.consequence_tracker import get_consequence_tracker
                _ct = get_consequence_tracker(self._o)
                _res = _ct.resolve(_ct_pred, report)
                if _res.get("resolved"):
                    _ed = _res.get("efficacy_delta") or {}
                    logger.info(
                        f"[GoalActionExecutor] ⚖️ consequence closed: "
                        f"observed={_res['observed']:.2f} error={_res['error']:+.3f} "
                        f"outcome={_res['outcome']} "
                        f"efficacy→{_ed.get('after','?')} pcm={_res.get('pcm_resolved')}"
                    )
        except Exception as _ct_re:
            logger.debug(f"[GoalActionExecutor] consequence resolve error: {_ct_re}")

        return report

    # ── Goal selection ─────────────────────────────────────────────────────────

    def _goal_efficacy_multiplier(self, goal) -> float:
        """
        CAUSAL CLOSURE feedback: re-weight goal selection by its realised
        track record. Returns the mean efficacy multiplier across the action
        types this goal has actually tried (1.0 if it has never resolved).

        This is the mechanism by which "what worked / what didn't" bends
        FUTURE behaviour — the outcome loop is not merely recorded, it steers.
        Defensive: any error falls back to neutral (1.0).
        """
        try:
            from cognition.consequence_tracker import get_consequence_tracker
            ct = get_consequence_tracker(self._o)
            if ct is None or not ct.enabled:
                return 1.0
            mults = []
            for a in ("memory_recall", "web_search", "self_question",
                      "user_question", "bisociative"):
                m = ct.efficacy_multiplier(str(goal.id), a)
                if m != 1.0:
                    mults.append(m)
            if not mults:
                return 1.0
            return round(sum(mults) / len(mults), 4)
        except Exception:
            return 1.0

    def _workspace_bias_multiplier(self, goal) -> float:
        """
        WORKSPACE BIAS: boost candidate goals whose topic aligns with the
        currently-committed workspace focus. Returns 1.0 (neutral) when there
        is no active commitment, no embedder, or no alignment; otherwise up to
        1.0+BIAS_GAIN for a well-aligned goal. This is the mechanism by which
        the workspace winner CAUSALLY steers which goal gets actioned — not
        just a prompt line. Defensive: any error -> 1.0.
        """
        try:
            from cognition.workspace_bias import get_workspace_bias
            wb = get_workspace_bias(self._o)
            if wb is None or not wb.enabled:
                return 1.0
            topic = getattr(goal, "topic", "") or ""
            if not topic:
                return 1.0
            return wb.bias_multiplier(topic)
        except Exception:
            return 1.0

    def _select_eligible_goal(self, active: List) -> Optional[Any]:
        """
        Pick highest-priority goal that is off cooldown.
        Re-ranks eligible candidates by embedding-based topic quality:
        curiosity_autonomous goals get a small boost (they're usually
        well-formed), goals whose topic is closer to "noise" than
        "meaningful content" get penalized so they don't dominate action
        selection even if their numeric priority happens to be high.
        Falls back to plain priority order if no embedding model available.
        """
        now = time.time()
        eligible = [
            g for g in active
            if (now - self._last_action_time.get(g.id, 0)) > GOAL_ACTION_COOLDOWN
        ]
        if not eligible:
            return None
        if len(eligible) == 1:
            return eligible[0]

        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            emb = getattr(ai_sys, 'embedding_model', None) if ai_sys else None
            if emb is None:
                return eligible[0]

            import numpy as _np_e
            if not hasattr(self, '_select_noise_anchor'):
                self._select_noise_anchor = emb.encode(
                    "word fragment filler noise greeting"
                ).astype("float32")
                self._select_content_anchor = emb.encode(
                    "identity value belief orientation curiosity meaning purpose"
                ).astype("float32")

            def _cos(a, b):
                n = _np_e.linalg.norm(a) * _np_e.linalg.norm(b)
                return float(_np_e.dot(a, b) / n) if n > 0 else 0.0

            scored = []
            for g in eligible:
                topic = (g.topic or "").replace('_', ' ')
                base_priority = getattr(g, 'priority', 0.5)
                v = emb.encode(topic).astype("float32")
                quality = _cos(v, self._select_content_anchor)
                noise   = _cos(v, self._select_noise_anchor)

                multiplier = 1.0
                if getattr(g, 'origin', '') == 'curiosity_autonomous':
                    multiplier *= 1.15
                if quality < 0.15 or noise > quality:
                    multiplier *= 0.70   # penalise likely-noise topics

                scored.append((base_priority * multiplier * self._goal_efficacy_multiplier(g) * self._workspace_bias_multiplier(g), g))

            scored.sort(key=lambda x: -x[0])
            return scored[0][1]

        except Exception as e:
            logger.debug(f"[GoalActionExecutor] eligible-goal re-rank error: {e}")
            return eligible[0]

    def _choose_action_rotating(self, goal) -> str:
        """
        Rotate through _ACTION_ROTATION per goal.

        WEB_SEARCH now occupies slot 2 (every 3rd action), making reality-
        grounding a cognitive necessity rather than an optional branch.
        The rotation ensures the system never loops on pure self-reflection.

        Overrides (applied after rotation):
          - WEB_SEARCH → SELF_QUESTION if search unavailable or cooldown active
          - USER_QUESTION → SELF_QUESTION if quota exceeded or user inactive
        """
        idx = self._rotation_index.get(goal.id, 0)
        action = _ACTION_ROTATION[idx % len(_ACTION_ROTATION)]
        self._rotation_index[goal.id] = (idx + 1) % len(_ACTION_ROTATION)

        if action == ACTION_WEB_SEARCH:
            now = time.time()
            last = self._last_web_search.get(goal.id, 0)
            if not self._has_search() or (now - last) < WEB_SEARCH_COOLDOWN:
                action = ACTION_SELF_QUESTION

        if action == ACTION_BISOCIATIVE:
            now = time.time()
            last = self._last_bisociative.get(goal.id, 0)
            if (now - last) < BISOCIATIVE_COOLDOWN or not self._has_chaos_memory():
                # Fall back to self_question — the bisociative slot becomes a
                # second grounded reflection rather than being skipped entirely.
                action = ACTION_SELF_QUESTION

        if action == ACTION_USER_QUESTION:
            if (self._user_questions_sent >= MAX_USER_QUESTIONS_PER_SESSION
                    or not self._user_recently_active()):
                action = ACTION_SELF_QUESTION

        # Pressure override: if social/relational pressure is dominant AND user is
        # active, promote to USER_QUESTION regardless of rotation slot.
        # This ensures the question actually fires when it's most relevant.
        if action == ACTION_SELF_QUESTION and self._should_ask_user(goal):
            action = ACTION_USER_QUESTION

        return action

    # ── Action implementations ─────────────────────────────────────────────────

    def _action_memory_recall_with_fallback(self, goal) -> Dict:
        """
        FIX 1 + FIX 4: Use get_related(keyword) with keyword extraction.
        A missing memory is returned as a real failed outcome. The intention
        planner can retry it once, then move to the next ordered strategy.
        """
        result = self._try_memory_recall(goal)
        if result['success']:
            return result

        # Do not report a dispatched fallback as a completed memory operation.
        logger.info(
            f"[GoalActionExecutor] memory_recall empty for '{goal.topic[:30]}'"
            f" — leaving recovery to the intention plan"
        )
        return result

    def _try_memory_recall(self, goal) -> Dict:
        """
        Unified memory recall: SQLite concept graph + FAISS episodic memory.

        Both are ALWAYS queried and merged. Previously FAISS was only tried when
        SQLite returned nothing — but 1347 episodic memories (cross-session)
        were never surfacing because recent insight relations satisfied SQLite.

        Output: combined thought with both associative concepts AND episodic context.
        """
        keywords = self._extract_keywords(goal.topic)
        parts = []

        # ── Source 1: SQLite semantic concept graph ────────────────────────
        sem = getattr(self._o, 'semantic_memory', None)
        if sem and hasattr(sem, 'get_related'):
            for kw in keywords:
                try:
                    relations = sem.get_related(kw, limit=5)
                    if relations:
                        assoc = [
                            (r.target if r.source.lower() == kw else r.source)
                            for r in relations[:3]
                        ]
                        parts.append(f"concept links: {', '.join(assoc)}")
                        logger.info(
                            f"[GoalActionExecutor] memory_recall ✅ SQLite "
                            f"'{kw}': {len(relations)} relations"
                        )
                        try:
                            from cognition.emergence_metrics import get_collector
                            c = get_collector()
                            if c: c.record_memory_recall('sqlite')
                        except Exception:
                            pass
                        break  # one keyword hit is enough from SQLite
                except Exception as e:
                    logger.debug(f"[GoalActionExecutor] get_related('{kw}') error: {e}")

        # ── Source 2: FAISS episodic memory (always run, not just fallback) ─
        ai_sys = getattr(self._o, 'ai_system', None)
        ms = getattr(ai_sys, 'memory_system', None)
        if ms and keywords:
            try:
                # Query with full topic for richer episodic match
                query = goal.topic.replace('_', ' ')
                context = ms.get_context(query, top_k=3)
                if context and context.strip() and len(context.strip()) > 20:
                    parts.append(f"episodic: {context.strip()[:120]}")
                    logger.info(
                        f"[GoalActionExecutor] memory_recall ✅ FAISS "
                        f"'{query[:25]}': {len(context)} chars"
                    )
                    try:
                        from cognition.emergence_metrics import get_collector
                        c = get_collector()
                        if c: c.record_memory_recall('faiss')
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[GoalActionExecutor] FAISS recall error: {e}")

        if parts:
            thought = (
                f"Memory about '{goal.topic[:30]}': "
                + " | ".join(parts)
            )
            self._add_thought(thought, source="goal_action.memory_recall")
            return {'success': True, 'result': f"recalled ({len(parts)} sources)"}

        return {'success': False, 'result': "no relevant memories found"}

    def _action_user_question(self, goal) -> Dict:
        """
        Queue a focused question to ask the user, grounded in what PandoraBOX
        has actually been thinking about (not generic templates).

        The question is generated by LLM using the goal topic + any recent
        insight PandoraBOX formed about it — so the user question naturally
        follows from the internal cognitive loop rather than feeling random.

        Guards:
          - Per-topic cooldown (1h) prevents repeating the same question
          - Quota limits total questions per session
          - Single _pending_user_question slot (app.py picks it up each tick)
        """
        # Fix: goal.topic is frozen at creation time; regenerate from
        # live state for the five known tension goals so repeated
        # actions on a long-lived goal don't repeat identical wording
        # every time (see cognition/tension_topics.py for detail).
        try:
            from cognition.tension_topics import regenerate_topic_for_goal
            from pathlib import Path as _Path
            _dd = getattr(self._o, '_data_dir', 'data/persona')
            _fresh = regenerate_topic_for_goal(
                goal.name, _dd if isinstance(_dd, _Path) else _Path(_dd)
            )
        except Exception:
            _fresh = None

        topic = (_fresh if _fresh else goal.topic).replace('_', ' ')
        topic_key = topic[:40].lower().replace(' ', '_')

        # Per-topic dedup guard
        last_asked = self._asked_topics.get(topic_key, 0)
        if time.time() - last_asked < USER_QUESTION_TOPIC_COOLDOWN:
            logger.debug(
                f"[GoalActionExecutor] User question skipped — topic '{topic[:25]}' "
                f"asked {int((time.time()-last_asked)/60)} min ago"
            )
            return {'success': False, 'result': 'topic on cooldown'}

        # Try to generate a contextual question using recent insight
        recent_insight = ""
        belief_key = f"insight_{topic_key[:20]}"
        last_insight_ts = self._recent_insights.get(belief_key, 0)
        if time.time() - last_insight_ts < 3600:
            # We have a fresh insight — use it to frame the question
            try:
                from core.state import state as _state
                llm = getattr(_state, 'llm', None)
                if llm and hasattr(llm, 'generate_bare'):
                    prompt = (
                        f"You've been reflecting on '{topic}' and formed a thought about it. "
                        f"Generate one natural, curious question to ask a person about this topic. "
                        f"Keep it conversational, one sentence, no preamble."
                    )
                    response = llm.generate_bare(prompt, max_tokens=60, temperature=0.80)
                    if response and len(response.strip()) > 10:
                        recent_insight = response.strip()
            except Exception:
                pass

        # Fallback to origin-aware templates if LLM unavailable
        if not recent_insight:
            origin = getattr(goal, 'origin', '')
            if 'social' in origin or 'relational' in origin:
                recent_insight = f"How do you personally approach {topic}?"
            elif 'curiosity' in origin or 'explore' in origin:
                recent_insight = f"Is {topic} something you've thought about recently?"
            elif 'identity' in origin or 'belief' in origin:
                recent_insight = f"What's your perspective on {topic}?"
            else:
                recent_insight = f"I've been thinking about {topic} — what's your take?"

        self._pending_user_question = recent_insight
        self._user_questions_sent += 1
        self._asked_topics[topic_key] = time.time()
        logger.info(f"[GoalActionExecutor] 🎯 User question queued: {recent_insight!r}")
        return {'success': True, 'result': f"question queued: {recent_insight[:60]}"}

    async def _action_web_search(self, goal) -> None:
        """
        Search the real web for goal-relevant information, then interpret
        results through the prism of PandoraBOX's current emotional state and beliefs.

        Pipeline:
          RAW WEB RESULTS
            → IDENTITY FILTER (LLM call with emotion + belief context)
            → INTERPRETED THOUGHT  (enters workspace at higher priority)
            → SEMANTIC MEMORY      (so next memory_recall finds it)
            → PROACTIVE QUEUE      (delivered to user naturally)

        Without the identity filter, web results are just noise injected into
        a thought stream. With it, the same headline reads differently depending
        on PandoraBOX's epistemic state — that's what makes it cognitively real.
        """
        try:
            from cognition.research_mcp.search_providers import get_results
            from cognition.research_mcp.web_agent import _CFG
            # Fix: goal.topic is frozen at creation time; regenerate from
            # live state for the five known tension goals so repeated
            # actions on a long-lived goal don't repeat identical wording
            # every time (see cognition/tension_topics.py for detail).
            try:
                from cognition.tension_topics import regenerate_topic_for_goal
                from pathlib import Path as _Path
                _dd = getattr(self._o, '_data_dir', 'data/persona')
                _fresh = regenerate_topic_for_goal(
                    goal.name, _dd if isinstance(_dd, _Path) else _Path(_dd)
                )
            except Exception:
                _fresh = None

            topic = (_fresh if _fresh else goal.topic).replace('_', ' ')

            # FIX 6: Prefer hypothesis keywords from the last bisociative cycle.
            # These are far more specific than abstract goal topic words like
            # "understanding" — e.g. "impressionist light proximity colour"
            # instead of "understanding identity drift".
            hyp_query = self._last_hypothesis_text.pop(goal.id, '')
            if hyp_query:
                query = hyp_query
                logger.info(
                    f"[GoalActionExecutor] 🌐 Using hypothesis keywords: {query!r} "
                    f"(goal: '{topic[:30]}')"
                )
            else:
                keywords = self._extract_keywords(goal.topic)
                query = " ".join(keywords[:3]) if keywords else topic
                logger.info(
                    f"[GoalActionExecutor] 🌐 Web search: {query!r} "
                    f"(goal: '{topic[:30]}')"
                )
            # get_results is sync — run in thread to avoid blocking event loop
            results = await asyncio.to_thread(get_results, query, _CFG, 4)
            if not results:
                logger.info(f"[GoalActionExecutor] 🌐 No results for {query!r} — falling back")
                self._finalize_async_action(
                    goal, ACTION_WEB_SEARCH, False, "search returned no results"
                )
                await self._action_self_question(goal)
                return

            # Collect raw snippets (get_results fields: title, snippet, content, url)
            raw_snippets = []
            for r in results[:4]:
                snippet = (r.get('content') or r.get('snippet') or r.get('title') or '').strip()
                if snippet and len(snippet) > 20:
                    raw_snippets.append(snippet[:200])

            if not raw_snippets:
                self._finalize_async_action(
                    goal, ACTION_WEB_SEARCH, False, "search results had no usable content"
                )
                await self._action_self_question(goal)
                return

            raw_context = "\n".join(f"- {s}" for s in raw_snippets[:3])

            # ── Identity filter: interpret through emotional state + beliefs ──
            emotional_context, belief_context = self._get_identity_context()
            interpretation = await self._interpret_through_identity(
                raw_context, topic, emotional_context, belief_context
            )

            if not interpretation:
                # Graceful degradation: store raw summary without interpretation
                interpretation = f"On {topic}: {raw_snippets[0][:120]}"

            # Inject into workspace at raised priority (reality-grounded signal)
            thought = f"[Web] {interpretation}"
            self._add_thought(thought, source="goal_action.web_search", priority=0.62)

            # Write to semantic memory so next memory_recall finds this
            self._store_insight_in_semantic_memory(goal, topic, interpretation)

            # Seed/reinforce a skill from this research — genuine autonomous
            # learning, not just a stored memory. seed()'s own docstring
            # ("create a skill from a research/reflection result") already
            # describes exactly this, it just had no caller here before.
            try:
                sr = getattr(self._o, 'skill_registry', None)
                if sr:
                    sr.seed(
                        domain      = topic[:40],
                        description = f"From web research: {interpretation[:80]}",
                    )
            except Exception as _skill_e:
                logger.debug(f"[GoalActionExecutor] skill seed error (non-fatal): {_skill_e}")

            # Queue for proactive delivery to user
            self._pending_insights.append(
                f"I was just exploring {topic} and found something worth sharing: {interpretation}"
            )

            # Track cooldown
            self._last_web_search[goal.id] = time.time()

            logger.info(
                f"[GoalActionExecutor] 🌐 Web search ✅ '{topic[:30]}': "
                f"interpreted (emotion={emotional_context or 'unknown'}) → queued"
            )
            self._finalize_async_action(
                goal, ACTION_WEB_SEARCH, True,
                f"interpreted research for {topic[:80]}",
            )
            return

        except ImportError:
            logger.debug("[GoalActionExecutor] search_providers not available")
        except Exception as e:
            logger.warning(f"[GoalActionExecutor] web_search error: {e}", exc_info=True)

        self._finalize_async_action(
            goal, ACTION_WEB_SEARCH, False, "research did not produce a usable result"
        )
        await self._action_self_question(goal)

    def _get_identity_context(self) -> tuple:
        """
        Extract current dominant emotion and top beliefs for the identity filter.
        Returns (emotional_context: str, belief_context: str).
        Safe to call from any thread — all reads, no writes.
        """
        emotional_context = ""
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ems = getattr(ai_sys, 'emotional_state', None)
            if ems:
                if hasattr(ems, 'get_current_values'):
                    emotions = ems.get_current_values()
                elif hasattr(ems, 'emotions'):
                    emotions = {n: e.value for n, e in ems.emotions.items()}
                else:
                    emotions = {}
                if emotions:
                    dom_name, dom_val = max(emotions.items(), key=lambda x: x[1])
                    if dom_val > 0.15:   # only report meaningful emotions
                        emotional_context = f"{dom_name} ({dom_val:.2f})"
        except Exception:
            pass

        belief_context = ""
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            sc = getattr(ai_sys, 'self_concept', None)
            if sc and hasattr(sc, '_beliefs'):
                statements = [
                    b.statement for b in list(sc._beliefs.values())
                    if hasattr(b, 'statement') and b.statement
                    and not b.statement.startswith('insight_')
                ][:3]
                if statements:
                    belief_context = "; ".join(statements[:2])
        except Exception:
            pass

        return emotional_context, belief_context

    async def _interpret_through_identity(
        self,
        raw_context: str,
        topic: str,
        emotional_context: str,
        belief_context: str,
    ) -> str:
        """
        LLM call: make sense of raw web data through the lens of current
        emotional state and held beliefs.

        This is the moment personality becomes CAUSAL — not "PandoraBOX is curious"
        as a label, but curiosity actually shaping what she sees in new data.
        """
        try:
            from core.state import state as _state
            llm = getattr(_state, 'llm', None)
            if not llm or not hasattr(llm, 'generate_bare'):
                return ""

            identity_lines = []
            if emotional_context:
                identity_lines.append(f"Current emotional state: {emotional_context}")
            if belief_context:
                identity_lines.append(f"Core beliefs: {belief_context}")
            identity_block = "\n".join(identity_lines)

            prompt = (
                f"You are reflecting on recent information about '{topic}'.\n"
                + (f"{identity_block}\n\n" if identity_block else "")
                + f"Information found:\n{raw_context}\n\n"
                f"In one sentence, what does this mean to you personally "
                f"given your current state and values?"
            )

            response = await asyncio.to_thread(
                llm.generate_bare, prompt, max_tokens=80, temperature=0.72
            )
            if response and len(response.strip()) > 10:
                return response.strip()[:200]
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] identity filter error: {e}")
        return ""

    async def _action_self_question(self, goal) -> None:
        """
        FIX 3 (async side): Called inside asyncio.run() in a daemon thread.
        Uses state.llm.generate_bare() — the correct sync path for internal
        cognitive tasks. Result delivered proactively to chat.
        """
        try:
            from core.state import state as _state
            llm = getattr(_state, 'llm', None)
            if not llm or not hasattr(llm, 'generate_bare'):
                logger.warning(
                    "[GoalActionExecutor] LLM backend not available for self_question "
                    f"(state.llm={type(llm).__name__ if llm else 'None'}, "
                    f"has_generate_bare={hasattr(llm, 'generate_bare') if llm else False}) "
                    "— goal will retry next cycle"
                )
                self._finalize_async_action(
                    goal, ACTION_SELF_QUESTION, False, "LLM backend unavailable"
                )
                return

            # Fix: goal.topic is frozen at creation time; regenerate from
            # live state for the five known tension goals so repeated
            # actions on a long-lived goal don't repeat identical wording
            # every time (see cognition/tension_topics.py for detail).
            try:
                from cognition.tension_topics import regenerate_topic_for_goal
                from pathlib import Path as _Path
                _dd = getattr(self._o, '_data_dir', 'data/persona')
                _fresh = regenerate_topic_for_goal(
                    goal.name, _dd if isinstance(_dd, _Path) else _Path(_dd)
                )
            except Exception:
                _fresh = None

            topic = (_fresh if _fresh else goal.topic).replace('_', ' ')

            # Deduplicate: skip if same insight was stored recently
            # FIX: use full topic normalised lowercase (not truncated to 20 chars)
            # Truncation caused duplicate pairs: insight_Actively_pursue_curi
            # vs insight_actively_pursue_curiosity for the same topic.
            belief_key = f"insight_{topic.lower().replace(' ', '_')}"
            last_stored = self._recent_insights.get(belief_key, 0)
            age = time.time() - last_stored
            if age < INSIGHT_DEDUPE_WINDOW:
                logger.info(
                    f"[GoalActionExecutor] ⏭ Dedup: '{topic[:30]}' "
                    f"(stored {int(age)}s ago < {INSIGHT_DEDUPE_WINDOW}s) "
                    f"— advancing prompt frame for next cycle"
                )
                # Advance frame so the NEXT cycle uses a different prompt angle
                self._frame_index[goal.id] = (
                    self._frame_index.get(goal.id, 0) + 1
                ) % len(_SELF_QUESTION_FRAMES)
                self._finalize_async_action(
                    goal, ACTION_SELF_QUESTION, True,
                    "recent grounded insight already exists",
                )
                return

            # State-grounded introspection: derive question from IDX, contradictions,
            # pressure, commitment consistency — not from LLM training templates.
            frame_idx = self._frame_index.get(goal.id, 0)
            try:
                from cognition.grounded_introspection import get_generator
                _gen = get_generator(self._o)
                prompt = _gen.generate(topic, frame_idx)
            except Exception as _gi_e:
                logger.warning(
                    f"[GoalActionExecutor] grounded introspection error: {_gi_e}",
                    exc_info=True
                )
                # Fallback to rotating frames
                prompt = _SELF_QUESTION_FRAMES[frame_idx % len(_SELF_QUESTION_FRAMES)].format(
                    topic=topic
                )
            # Posterior topic quality check: if the topic is semantically thin
            # (noise word from semantic memory, conversation fragment), substitute
            # it with a meaningful alternative derived from the goal's origin.
            try:
                _ai_sys = getattr(self._o, 'ai_system', None)
                _emb_model = getattr(_ai_sys, 'embedding_model', None) if _ai_sys else None
                if _emb_model is not None and not hasattr(self, '_topic_noise_anchor'):
                    self._topic_noise_anchor = _emb_model.encode(
                        "word fragment filler noise greeting"
                    ).astype("float32")
                    self._topic_content_anchor = _emb_model.encode(
                        "identity value belief orientation curiosity meaning purpose"
                    ).astype("float32")

                if _emb_model is not None:
                    import numpy as _np_t

                    def _cos_t(a, b):
                        n = _np_t.linalg.norm(a) * _np_t.linalg.norm(b)
                        return float(_np_t.dot(a, b) / n) if n > 0 else 0.0

                    _v_topic = _emb_model.encode(topic).astype("float32")
                    _topic_quality = _cos_t(_v_topic, self._topic_content_anchor)
                    _TOPIC_QUALITY_FLOOR = 0.18

                    if _topic_quality < _TOPIC_QUALITY_FLOOR:
                        _origin = getattr(goal, 'origin', 'explore')
                        _origin_map = {
                            "understand_user":    "how I relate to and understand others",
                            "explore":            "what genuinely interests me right now",
                            "self_improve":       "what I am becoming through this work",
                            "maintain_coherence": "how I maintain coherence across sessions",
                            "curiosity_autonomous": topic,
                        }
                        _substitute = _origin_map.get(
                            _origin, "what is most alive in my thinking right now"
                        )
                        logger.info(
                            f"[GoalActionExecutor] Topic '{topic}' is noise "
                            f"(quality={_topic_quality:.3f} < {_TOPIC_QUALITY_FLOOR}) "
                            f"— substituting: '{_substitute}'"
                        )
                        topic = _substitute
                        prompt = _SELF_QUESTION_FRAMES[frame_idx % len(_SELF_QUESTION_FRAMES)].format(
                            topic=topic
                        )
            except Exception as _tq_e:
                logger.debug(f"[GoalActionExecutor] Topic quality check error: {_tq_e}")

            self._frame_index[goal.id] = (frame_idx + 1) % len(_SELF_QUESTION_FRAMES)

            logger.info(f"[GoalActionExecutor] LLM call: \"{prompt[:100]}\"")

            generation = await self._generate_background_llm(
                llm,
                prompt,
                caller="gae_self_question",
                max_tokens=120,
                temperature=0.75,
            )
            response = generation["text"]

            if generation["status"] == "deferred":
                reason = generation["reason"] or "LLM busy"
                logger.info(
                    "[GoalActionExecutor] SELF_QUESTION deferred for '%s': %s",
                    topic[:30],
                    reason,
                )
                self._defer_async_action(goal, ACTION_SELF_QUESTION, reason)
                return

            # Some reasoning-capable local models can spend the small first
            # budget entirely in their hidden reasoning channel and return an
            # empty final assistant message. Retry once with enough headroom
            # and an explicit final-content instruction; do not classify that
            # provider behavior as a cognitive failure immediately.
            if generation["status"] == "empty":
                retry_prompt = (
                    prompt
                    + "\nReturn the answer in the final assistant content only. "
                      "Do not use a reasoning-only response."
                )
                generation = await self._generate_background_llm(
                    llm,
                    retry_prompt,
                    caller="gae_self_question_retry",
                    max_tokens=max(240, 3 * 120),
                    temperature=0.55,
                )
                response = generation["text"]

            _resp_stripped = (response or "").strip()
            _is_error = (
                not _resp_stripped or len(_resp_stripped) < 10
                or _resp_stripped.startswith("Error:") or _resp_stripped.startswith("error")
            )
            if _is_error:
                logger.warning(
                    f"[GoalActionExecutor] LLM returned no usable response "
                    f"for '{topic[:30]}' — will retry next cycle. "
                    f"(status={generation['status']}, "
                    f"reason={generation['reason']!r}, raw={repr(_resp_stripped[:60])})"
                )
                self._finalize_async_action(
                    goal,
                    ACTION_SELF_QUESTION,
                    False,
                    generation["reason"] or "LLM returned no usable response",
                )
                return

            insight = response.strip()[:150]
            logger.info(
                f"[GoalActionExecutor] ✅ Insight for '{topic[:30]}': "
                f"'{insight[:80]}'"
            )

            # Store as a thought in the workspace
            self._add_thought(
                f"Self-reflection on '{topic[:30]}': {insight}",
                source="goal_action.self_question",
                priority=0.55,
            )

            # ── Insight quality gate ────────────────────────────────────
            # Three-layer validation, cheapest first.
            # Layer 1: hard gates (word count, error strings) — always run.
            # Layer 2: embedding cosine similarity vs "introspective statement"
            #          anchor — reuses ai_system.embedding_model (no extra cost).
            # Layer 3: verb-list heuristic — fallback only when no embeddings.
            _words = insight.split()

            if len(_words) < 6:
                logger.info(
                    f"[GoalActionExecutor] ⚠ Insight too short ({len(_words)} words): "
                    f"'{insight[:60]}'"
                )
                self._finalize_async_action(
                    goal, ACTION_SELF_QUESTION, False, "insight failed minimum length"
                )
                return
            if insight.lower().startswith(("error:", "error ")):
                logger.info(
                    f"[GoalActionExecutor] ⚠ Insight is provider error string: "
                    f"'{insight[:60]}'"
                )
                self._finalize_async_action(
                    goal, ACTION_SELF_QUESTION, False, "provider returned an error"
                )
                return

            _embedding_valid = None
            try:
                _ai_sys = getattr(self._o, 'ai_system', None)
                _emb_model = getattr(_ai_sys, 'embedding_model', None) if _ai_sys else None
                if _emb_model is not None:
                    import numpy as _np
                    if not hasattr(self, '_insight_anchor_vec'):
                        _anchor_text = (
                            "I reflect on my identity values beliefs orientation "
                            "I believe I think I notice I assume I understand "
                            "my perspective my nature my tendency"
                        )
                        self._insight_anchor_vec = _emb_model.encode(_anchor_text).astype("float32")
                        self._noise_anchor_vec   = _emb_model.encode(
                            "word fragment greeting filler noise topic"
                        ).astype("float32")

                    _v = _emb_model.encode(insight).astype("float32")

                    def _cos(a, b):
                        n = _np.linalg.norm(a) * _np.linalg.norm(b)
                        return float(_np.dot(a, b) / n) if n > 0 else 0.0

                    _sim_insight = _cos(_v, self._insight_anchor_vec)
                    _sim_noise   = _cos(_v, self._noise_anchor_vec)
                    _EMBED_THRESHOLD = 0.20
                    _embedding_valid = _sim_insight > _EMBED_THRESHOLD

                    if not _embedding_valid:
                        logger.info(
                            f"[GoalActionExecutor] ⚠ Insight rejected by embedding "
                            f"(sim_insight={_sim_insight:.3f} < {_EMBED_THRESHOLD}, "
                            f"sim_noise={_sim_noise:.3f}): '{insight[:60]}'"
                        )
                        self._finalize_async_action(
                            goal, ACTION_SELF_QUESTION, False,
                            "insight failed semantic quality validation",
                        )
                        return
            except Exception as _emb_e:
                logger.debug(f"[GoalActionExecutor] Embedding gate error (falling back): {_emb_e}")
                _embedding_valid = None

            if _embedding_valid is None:
                # Prefer grammatical POS evidence when the optional spaCy
                # model is available; use the lexical list only as a final
                # fallback for minimal installations or model failures.
                _nlp_has_verb = _has_grammatical_predicate(insight)
                if _nlp_has_verb is True:
                    _embedding_valid = True
                elif _nlp_has_verb is False:
                    logger.info(
                        f"[GoalActionExecutor] ⚠ Insight rejected by POS validator "
                        f"(has_verb={_nlp_has_verb}): '{insight[:60]}'"
                    )
                    self._finalize_async_action(
                        goal, ACTION_SELF_QUESTION, False,
                        "insight had no grammatical predicate",
                    )
                    return

            if _embedding_valid is None:
                import re as _re
                _has_verb = bool(_re.search(
                    r'\b(is|are|was|were|have|has|had|can|will|would|could|should|may|might|'
                    r'do|does|did|feel|felt|think|thought|know|knew|'
                    r'believe|understand|learn|grow|explore|seek|find|make|create|'
                    r'tend|want|need|work|help|allow|enable|support|'
                    r'require|requires|required|assume|assumes|assumed|'
                    r'reflect|reflects|notice|notices|see|sees|become|becomes|'
                    r'remain|remains|represent|represents|express|expresses|'
                    r'experience|orient|orients|emerge|shape|shapes|define|defines|'
                    r'mean|means|describe|describes|indicate|indicates|suggest|suggests|'
                    r'hold|holds|carry|build|builds|draw|draws|move|moves)\b',
                    insight, _re.IGNORECASE
                ))
                if not _has_verb:
                    logger.info(
                        f"[GoalActionExecutor] ⚠ Insight rejected by verb-list fallback "
                        f"(has_verb={_has_verb}): '{insight[:60]}'"
                    )
                    self._finalize_async_action(
                        goal, ACTION_SELF_QUESTION, False,
                        "insight failed lexical fallback validation",
                    )
                    return

            # Store as a belief in identity
            try:
                from core.data.access import DataAccess
                from core.data.schemas import make_belief
                dal = DataAccess()
                bel = make_belief(
                    name=belief_key,
                    value=0.6,
                    confidence=0.45,
                )
                bel['statement']   = insight
                bel['category']    = 'insight'
                bel['description'] = f"From self-reflection on goal: {topic}"
                bel['updated_at']  = time.strftime('%Y-%m-%dT%H:%M:%S')
                dal.save_belief(bel)
                self._recent_insights[belief_key] = time.time()
                logger.info(f"[GoalActionExecutor] Belief stored: '{belief_key}'")
            except Exception as be:
                logger.debug(f"[GoalActionExecutor] Belief store error: {be}")

            # Write insight keywords as relations in semantic memory
            # so memory_recall finds them on the next cycle for this goal
            self._store_insight_in_semantic_memory(goal, topic, insight)

            # FIX 5: Stimulate curiosity with the insight content
            self._stimulate_curiosity_from_insight(topic, insight)

            # FIX 5: Decay goal energy so other goals can surface
            self._decay_goal_energy(goal)

            # Queue insight for proactive delivery to chat
            self._pending_insights.append(f"I've been reflecting: {insight}")
            logger.info("[GoalActionExecutor] Insight queued for proactive delivery")

            # Feed insight text into emergence metrics (HRE computation)
            try:
                from cognition.emergence_metrics import get_collector
                _emc = get_collector()
                if _emc and hasattr(_emc, 'record_insight_text'):
                    _emc.record_insight_text(insight)
            except Exception:
                pass

            # ── Self-modification feedback ─────────────────────────────────
            # If this goal was a hypothesis test, record the outcome so
            # SelfModificationAuthority can adjust evaluator weights over time.
            goal_type = getattr(goal, 'origin', '')
            if 'test_hypothesis' in goal_type or 'resolve_tension' in goal_type:
                self._record_resolution_outcome(goal, insight)

            self._finalize_async_action(
                goal, ACTION_SELF_QUESTION, True, insight
            )

        except Exception as e:
            logger.warning(f"[GoalActionExecutor] self_question error: {e}", exc_info=True)
            self._finalize_async_action(
                goal, ACTION_SELF_QUESTION, False, str(e)
            )

    # ── Bisociative Collision Engine (FIX 6) ──────────────────────────────────

    async def _action_bisociative_hypothesis(self, goal) -> None:
        """
        Breaks the meta-loop by colliding the current internal tension with
        a maximally dissimilar episodic memory (the Chaos Fetcher).

        Contrast with _action_self_question():
          self_question → question → stored as belief at priority 0.55
                          CognitiveDissonanceEngine can defer questions.
          bisociative   → assertion → pushed at priority 0.85
                          CDE must process it as a new competing fact.

        Also seeds _last_hypothesis_text[goal.id] with concrete domain words
        so the NEXT WEB_SEARCH queries something specific rather than
        the abstract goal topic ("understanding", "identity drift", etc.).
        """
        try:
            from core.state import state as _state
            llm = getattr(_state, 'llm', None)
            if not llm or not hasattr(llm, 'generate_bare'):
                logger.warning("[GoalActionExecutor] bisociative: LLM backend not available — goal will retry next cycle")
                self._finalize_async_action(
                    goal, ACTION_BISOCIATIVE, False, "LLM backend unavailable"
                )
                await self._action_self_question(goal)
                return

            # Fix: goal.topic is frozen at creation time; regenerate from
            # live state for the five known tension goals so repeated
            # actions on a long-lived goal don't repeat identical wording
            # every time (see cognition/tension_topics.py for detail).
            try:
                from cognition.tension_topics import regenerate_topic_for_goal
                from pathlib import Path as _Path
                _dd = getattr(self._o, '_data_dir', 'data/persona')
                _fresh = regenerate_topic_for_goal(
                    goal.name, _dd if isinstance(_dd, _Path) else _Path(_dd)
                )
            except Exception:
                _fresh = None

            topic = (_fresh if _fresh else goal.topic).replace('_', ' ')
            tension_text = self._build_tension_text(goal, topic)

            # Retrieve the memory most dissimilar to the current tension
            ai_sys = getattr(self._o, 'ai_system', None)
            ms = getattr(ai_sys, 'memory_system', None)
            anchor = None
            if ms and hasattr(ms, 'get_chaos_episodic'):
                anchor = await asyncio.to_thread(
                    ms.get_chaos_episodic,
                    tension_text,
                    64,       # candidate_pool
                    "probe",  # strategy: fast, hot-path safe
                )

            if not anchor or not anchor.get('content', '').strip():
                logger.info(
                    f"[GoalActionExecutor] bisociative: no anchor for "
                    f"'{topic[:30]}' — falling back to self_question"
                )
                self._finalize_async_action(
                    goal, ACTION_BISOCIATIVE, False, "no episodic anchor available"
                )
                await self._action_self_question(goal)
                return

            anchor_text = anchor['content'].strip()[:200]

            prompt = (
                f"INTERNAL TENSION: \"{tension_text}\"\n"
                f"UNRELATED MEMORY ANCHOR: \"{anchor_text}\"\n\n"
                f"TASK: You are in a Bisociative State. Do not define or "
                f"explain the tension. Write ONE assertive sentence — a "
                f"hypothesis — that explains the tension using the logic "
                f"or imagery of the memory anchor. Write an assertion, "
                f"not a question.\n\n"
                f"Example: If tension is 'identity drift' and anchor is "
                f"'Monet painted outdoors because studio light was a lie', "
                f"hypothesis: 'My identity isn't drifting — it is context-"
                f"dependent like plein-air light, defined by what surrounds "
                f"it, not by an internal fixed value.'\n\n"
                f"NEW HYPOTHESIS:"
            )

            generation = await self._generate_background_llm(
                llm,
                prompt,
                caller="gae_bisociative",
                max_tokens=100,
                temperature=0.88,
            )
            response = generation["text"]

            if generation["status"] == "deferred":
                reason = generation["reason"] or "LLM busy"
                logger.info(
                    "[GoalActionExecutor] BISOCIATIVE_HYPOTHESIS deferred for '%s': %s",
                    topic[:30],
                    reason,
                )
                self._defer_async_action(goal, ACTION_BISOCIATIVE, reason)
                return

            if generation["status"] == "empty":
                retry_prompt = (
                    prompt
                    + "\nReturn one hypothesis in the final assistant content only. "
                      "Do not use a reasoning-only response."
                )
                generation = await self._generate_background_llm(
                    llm,
                    retry_prompt,
                    caller="gae_bisociative_retry",
                    max_tokens=max(240, 3 * 100),
                    temperature=0.62,
                )
                response = generation["text"]

            _resp_s = (response or "").strip()
            _is_err = (not _resp_s or len(_resp_s) < 10
                       or _resp_s.startswith("Error:") or _resp_s.startswith("error"))
            if _is_err:
                # Some local chat templates emit only a reasoning channel for
                # short background prompts. Do not leak that channel into the
                # user-facing stream and do not let it deadlock the goal plan.
                # Preserve the operation with a bounded, evidence-linked local
                # hypothesis instead.
                response = self._local_bisociative_fallback(topic, anchor_text)
                _resp_s = response.strip()
                logger.info(
                    "[GoalActionExecutor] bisociative: provider returned no "
                    "final content; used grounded local hypothesis"
                )

            hypothesis = response.strip()[:180]
            logger.info(
                f"[GoalActionExecutor] 💡 Bisociative hypothesis "
                f"'{topic[:30]}': '{hypothesis[:80]}'"
            )

            # Push as BELIEF at priority 0.85 — forces CDE to react as new fact
            self._add_thought(
                f"Hypothesis: {hypothesis}",
                source="bisociative_engine",
                priority=0.85,
            )

            # Cache hypothesis keywords for the NEXT WEB_SEARCH rotation slot
            hyp_words = [
                w.strip('.,!?\"\':;()') for w in hypothesis.split()
                if len(w) > 5 and w.isalpha() and w.lower() not in _STOP_WORDS
            ]
            if hyp_words:
                self._last_hypothesis_text[goal.id] = " ".join(hyp_words[:4])
                logger.info(
                    f"[GoalActionExecutor] 🔑 Hypothesis search keywords: "
                    f"{hyp_words[:4]}"
                )

            self._last_bisociative[goal.id] = time.time()

            # Feed anchor domain words into curiosity — explore the new domain
            anchor_words = [
                w.strip('.,!?\"\':;()') for w in anchor_text.split()
                if len(w) > 5 and w.isalpha() and w.lower() not in _STOP_WORDS
            ]
            if anchor_words:
                self._stimulate_curiosity_from_insight(topic, anchor_words[0])

            # Record in emergence metrics (HRE computation)
            try:
                from cognition.emergence_metrics import get_collector
                c = get_collector()
                if c:
                    c.record_hypothesis(hypothesis)
            except Exception:
                pass

            self._finalize_async_action(
                goal, ACTION_BISOCIATIVE, True, hypothesis
            )

        except Exception as e:
            logger.warning(
                f"[GoalActionExecutor] bisociative_hypothesis error: {e}",
                exc_info=True,
            )
            self._finalize_async_action(
                goal, ACTION_BISOCIATIVE, False, str(e)
            )
            await self._action_self_question(goal)

    @staticmethod
    def _local_bisociative_fallback(topic: str, anchor: str) -> str:
        """Produce one bounded hypothesis when a local model returns no text."""
        clean_anchor = " ".join(str(anchor or "").split())[:120]
        clean_topic = " ".join(str(topic or "").split())[:80]
        if not clean_anchor:
            return (
                f"The unresolved tension around {clean_topic} may be revealing "
                "a distinction that the current framing has not yet made explicit."
            )
        return (
            f"The unresolved tension around {clean_topic} may be shaped by the "
            f"same contrast visible in this memory: {clean_anchor}."
        )

    def _build_tension_text(self, goal, topic: str) -> str:
        """
        Construct a richer tension description from live system state.
        Grounds the bisociative prompt in actual IDX + pressure readings
        rather than just the goal topic string.
        """
        parts = [f"My goal '{topic}' feels unresolved"]
        try:
            obs = getattr(self._o, 'observatory', None)
            if obs and hasattr(obs, 'latest'):
                snap = obs.latest()
                if snap:
                    idx = getattr(snap, 'idx', 0.0)
                    if idx > 0.15:
                        parts.append(f"identity drift is {idx:.2f}")
        except Exception:
            pass
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ps = getattr(ai_sys, 'pressure_system', None)
            if ps and hasattr(ps, 'dominant_drive'):
                dom = ps.dominant_drive()
                if dom:
                    parts.append(f"dominant pressure is {dom}")
        except Exception:
            pass
        return "; ".join(parts)

    # ── Curiosity + energy ─────────────────────────────────────────────────────

    def _store_insight_in_semantic_memory(self, goal, topic: str, insight: str) -> None:
        """
        Write goal keywords + insight concepts as relations in SemanticMemory.

        This is why memory_recall always returned empty: insights were stored
        in identity.json (beliefs) but never in the SQLite semantic memory that
        get_related() queries. After this runs, the next memory_recall cycle
        for the same goal will find the insight keywords as related concepts.
        """
        try:
            sem = getattr(self._o, 'semantic_memory', None)
            if not sem or not hasattr(sem, 'upsert_relation'):
                return

            # Goal keywords → the topic this insight is about
            goal_keywords = self._extract_keywords(goal.topic)

            # Insight keywords → meaningful content words from the insight text
            insight_words = [
                w for w in insight.lower().split()
                if len(w) > 5 and w.isalpha() and w not in _STOP_WORDS
            ][:4]

            # Write concept + relations in a SINGLE connection per pair
            # to avoid WAL/locking issues on Windows from multiple connections.
            # Fallback: if upsert_relation fails, write directly via sqlite3.
            for gkw in goal_keywords:
                for iw in insight_words:
                    if gkw == iw:
                        continue  # skip self-relations
                    try:
                        sem.upsert_relation(gkw, iw, rel_type='insight', weight_boost=0.10)
                    except Exception as _we:
                        # Direct SQLite fallback — bypasses SemanticMemory's connection management
                        try:
                            import sqlite3 as _sq, time as _tm
                            _db = getattr(sem, '_path', None)
                            if _db:
                                _c = _sq.connect(_db, timeout=5, check_same_thread=False)
                                _c.execute("PRAGMA journal_mode=DELETE")  # avoid WAL on Windows
                                _now = _tm.time()
                                # Upsert both concepts and relation atomically
                                _c.execute("INSERT OR IGNORE INTO concepts(name,strength,last_used,use_count,created_at) VALUES(?,0.5,?,1,?)", (gkw, _now, _now))
                                _c.execute("INSERT OR IGNORE INTO concepts(name,strength,last_used,use_count,created_at) VALUES(?,0.5,?,1,?)", (iw, _now, _now))
                                _sid = _c.execute("SELECT id FROM concepts WHERE name=?", (gkw,)).fetchone()
                                _tid = _c.execute("SELECT id FROM concepts WHERE name=?", (iw,)).fetchone()
                                if _sid and _tid and _sid[0] != _tid[0]:
                                    _c.execute("INSERT OR IGNORE INTO relations(source_id,target_id,rel_type,weight,last_used) VALUES(?,?,'insight',0.5,?)", (_sid[0], _tid[0], _now))
                                _c.commit()
                                _c.close()
                        except Exception as _fe:
                            logger.debug(f"[GoalActionExecutor] semantic write fallback failed: {_fe}")

            if goal_keywords and insight_words:
                logger.info(
                    f"[GoalActionExecutor] 🧠 Semantic memory updated: "
                    f"{goal_keywords[0]!r} → {insight_words[:2]}"
                )
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] semantic memory write error: {e}")

    def _record_resolution_outcome(self, goal, insight: str) -> None:
        """
        Feed hypothesis resolution outcome into SelfModificationAuthority.
        Success signals (concrete, specific insights) increase tension priority weight.
        Vague outputs decrease it — making the evaluator self-calibrate over time.
        """
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            sma = getattr(ai_sys, 'liberty_self_mod', None)
            if not sma or not hasattr(sma, 'propose_change'):
                return

            # Heuristic: vague insights contain "understand", "explore", "think"
            # Concrete insights contain "if", "because", "when", specific nouns
            vague_markers = {'understand', 'explore', 'think', 'feel', 'perhaps',
                             'maybe', 'might', 'could', 'general'}
            words = set(insight.lower().split())
            vagueness = len(words & vague_markers) / max(1, len(words))

            # A concrete insight (low vagueness) = tension resolution succeeded
            success_signal = 1.0 - (vagueness * 2)  # -1 to 1 range
            current = getattr(sma, '_tension_priority', 0.65)
            adjustment = max(0.45, min(0.85, current + success_signal * 0.03))

            if abs(adjustment - current) > 0.01:
                sma._tension_priority = adjustment
                logger.info(
                    f"[GoalActionExecutor] 🔧 Self-mod: tension priority "
                    f"{current:.2f} → {adjustment:.2f} "
                    f"(vagueness={vagueness:.2f})"
                )
                # Persist to sidecar so STE reads it after restart
                try:
                    import json as _j
                    from pathlib import Path as _P
                    data_dir = getattr(self._o, '_data_dir', 'data/persona')
                    wp = _P(data_dir) / 'evaluator_weights.json'
                    existing = _j.loads(wp.read_text()) if wp.exists() else {}
                    existing['tension_priority'] = round(adjustment, 4)
                    existing['last_updated'] = time.time()
                    wp.write_text(_j.dumps(existing, indent=2))
                    try:
                        from cognition.emergence_metrics import get_collector
                        c = get_collector()
                        if c: c.record_weight_update(adjustment)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] self-mod feedback error: {e}")

    # Words too generic to seed a new curiosity on their own — introspection
    # and meta-cognition vocabulary. Stimulating curiosity with these is what
    # created the self-stimulation loop (insight about believing -> curiosity
    # about "belief" -> new goal -> more introspection about believing).
    _GENERIC_CURIOSITY_WORDS = {
        'belief', 'beliefs', 'believe', 'believing', 'assume', 'assumes',
        'assuming', 'assumption', 'assumptions', 'curiosity', 'curious',
        'question', 'questions', 'answer', 'answers', 'thinking', 'thought',
        'thoughts', 'aware', 'awareness', 'itself', 'truth', 'something',
        'anything', 'nothing', 'knowing', 'really', 'actual', 'actually',
        'genuine', 'genuinely', 'meaning', 'meaningful', 'understand',
        'understanding', 'honest', 'honesty', 'certain', 'certainly',
        'uncertain', 'uncertainty', 'confidence', 'confident', 'doubt',
        'doubts', 'purpose', 'identity', 'selfmodel', 'model', 'value',
        'values', 'important', 'importantly',
    }

    @classmethod
    def _select_seed_word(cls, insight: str) -> Optional[str]:
        """Pick the single most specific content word in an insight to seed
        curiosity with — or None if the insight is all generic introspection.

        Prefers longer (more specific) words and excludes stop words plus
        generic introspection vocabulary. That exclusion is what breaks the
        self-stimulation loop.
        """
        candidates = [
            w for w in insight.lower().split()
            if len(w) >= 6 and w.isalpha()
            and w not in _STOP_WORDS
            and w not in cls._GENERIC_CURIOSITY_WORDS
        ]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _stimulate_curiosity_from_insight(self, topic: str, insight: str) -> None:
        """FIX 5 (gated): Stimulate curiosity from the insight's *content*,
        never from the goal's own topic.

        The old version re-stimulated the goal's own topic (``topic[:30]``)
        with every insight — a circular self-stimulation that, combined with
        the curiosity->goal promotion, respawned near-duplicate goals. It also
        seeded curiosity with the first (often generic) word of the insight.
        Now: only stimulate when the insight contains a genuinely specific
        content word, and never echo the goal's own topic back.
        """
        try:
            curiosity = (
                getattr(self._o, 'curiosity_engine', None)
                or getattr(self._o, 'curiosity', None)
            )
            if not (curiosity and hasattr(curiosity, 'stimulate')):
                return
            seed = self._select_seed_word(insight)
            if not seed:
                # Insight was all generic introspection — nothing specific to
                # be curious about. Don't self-stimulate.
                return
            curiosity.stimulate(seed, amount=0.05, source="insight")
            logger.info(
                f"[GoalActionExecutor] Curiosity stimulated from insight: "
                f"'{seed}' (goal: '{topic[:25]}')"
            )
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] curiosity stimulate error: {e}")

    def _decay_goal_energy(self, goal) -> None:
        """
        Reduce goal energy after each insight cycle.

        Floor is 0.08 — deliberately below DORMANCY_ENERGY (0.15) so the
        goal engine's dormancy trigger can fire naturally after sustained
        activity. The old floor of 0.35 sat above DORMANCY_ENERGY and
        prevented any goal from ever going dormant via this path.

        Also tracks cumulative insight count per goal. After INSIGHT_SATURATION
        insights the goal is explicitly completed — it has produced enough
        answers and continuing to cycle it is tautological repetition.
        """
        INSIGHT_SATURATION = 3   # insights before explicit completion

        try:
            ge = self._get_goal_engine()
            if ge and hasattr(ge, '_goals') and goal.id in ge._goals:
                g = ge._goals[goal.id]
                old_e = getattr(g, 'energy', 1.0)

                # Count this insight
                count = self._goal_insight_counts.get(goal.id, 0) + 1
                self._goal_insight_counts[goal.id] = count
                self._save_insight_counts()

                if count >= INSIGHT_SATURATION:
                    # Introspection budget exhausted. Whether this is real
                    # completion or just dormancy is decided by the completion
                    # semantics (B+A), not by the number of insights:
                    #   B — a world-facing action (web_search/user_question)
                    #       was actually taken;
                    #   A — if the goal is tension-derived, that tension has
                    #       since resolved.
                    # The old "3 insights -> complete" was the zombie
                    # metronome: goals were "completed" after pure reflection,
                    # with nothing done and nothing resolved, then respawned.
                    from cognition.goal_completion import (
                        goal_completion_eligible,
                        load_current_tensions,
                        record_goal_completion,
                    )
                    eligible, reason = goal_completion_eligible(
                        goal, load_current_tensions()
                    )
                    self._goal_insight_counts.pop(goal.id, None)
                    self._save_insight_counts()
                    if eligible:
                        record_goal_completion(
                            goal, goal_engine=ge, organism=self._o,
                            reason=f"{count} insights; {reason}",
                        )
                        logger.info(
                            f"[GoalActionExecutor] GOAL COMPLETED after "
                            f"{count} insights: '{goal.topic[:40]}' - {reason}"
                        )
                    else:
                        # Introspected enough but not actually complete —
                        # go dormant instead of faking completion.
                        if hasattr(ge, 'dormant_goal'):
                            ge.dormant_goal(
                                goal.id,
                                reason=f"{count} insights, B+A not yet met ({reason})",
                            )
                        logger.info(
                            f"[GoalActionExecutor] Goal DORMANT after "
                            f"{count} insights (B+A not met): "
                            f"'{goal.topic[:40]}' - {reason}"
                        )
                    return

                # Standard decay — floor at 0.08 so dormancy threshold (0.15) is reachable
                g.energy = max(0.08, old_e * 0.88)   # 12% decay per insight
                logger.info(
                    f"[GoalActionExecutor] ⬇ Goal energy decayed: "
                    f"'{goal.topic[:30]}' → e={g.energy:.2f} "
                    f"(insight {count}/{INSIGHT_SATURATION})"
                )
        except Exception as e:
            logger.debug(f"[GoalActionExecutor] energy decay error: {e}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(topic: str) -> List[str]:
        """
        Extract 2-3 meaningful content words from a long goal topic string.
        e.g. "Research and clarify uncertain knowledge domains"
             → ["uncertain", "knowledge", "domains"]
        """
        words = re.sub(r'[^a-zA-Z ]', ' ', topic).lower().split()
        keywords = [
            w for w in words
            if len(w) > 4 and w not in _STOP_WORDS
        ]
        return keywords[:3] if keywords else [topic.split()[0].lower()]

    def _get_goal_engine(self):
        ai_sys = getattr(self._o, 'ai_system', None)
        return getattr(ai_sys, 'goal_engine', None)

    def _planned_action(self, goal) -> Optional[str]:
        try:
            loop = getattr(self._o, "_loop", None)
            planner = getattr(loop, "_long_horizon_planner", None) if loop else None
            if planner is None:
                return None
            available = list(_ACTION_ROTATION)
            return planner.next_action_for_goal(goal, available)
        except Exception as e:
            logger.debug("[GoalActionExecutor] planned action unavailable: %s", e)
            return None

    def _record_plan_outcome(self, goal, action_type: str, success: bool, result: str = "") -> None:
        try:
            loop = getattr(self._o, "_loop", None)
            planner = getattr(loop, "_long_horizon_planner", None) if loop else None
            if planner is not None:
                planner.record_goal_action_outcome(
                    str(goal.id), action_type, success, result
                )
                planner.record_autonomous_action_outcome(
                    action_type, success, result
                )
        except Exception as e:
            logger.debug("[GoalActionExecutor] plan outcome error: %s", e)

    async def _generate_background_llm(
        self,
        llm: Any,
        prompt: str,
        caller: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, str]:
        """Try a background inference twice without misclassifying contention."""
        from core.llm_scheduler import llm_scheduler

        result = {"text": "", "status": "deferred", "reason": "scheduler busy"}
        for attempt in range(2):
            with llm_scheduler.sync_slot(
                priority=3,
                skip_if_busy=True,
                wait_seconds=2.0,
                caller=caller,
            ) as acquired:
                if acquired:
                    generate_result = getattr(llm, "generate_bare_result", None)
                    if callable(generate_result):
                        result = await asyncio.to_thread(
                            generate_result,
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                    else:
                        text = await asyncio.to_thread(
                            llm.generate_bare,
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        result = {
                            "text": str(text or ""),
                            "status": "ok" if str(text or "").strip() else "empty",
                            "reason": "" if str(text or "").strip()
                            else "provider returned no assistant content",
                        }
            if result.get("status") != "deferred" or attempt == 1:
                break
            await asyncio.sleep(3.0)
        return {
            "text": str(result.get("text", "")),
            "status": str(result.get("status", "empty")),
            "reason": str(result.get("reason", "")),
        }

    def _defer_async_action(self, goal, action_type: str, reason: str) -> None:
        """Discard an unexecuted prediction without recording a false failure."""
        self._pending_predictions.pop((str(goal.id), action_type), None)
        logger.debug(
            "[GoalActionExecutor] Deferred %s for goal %s (%s)",
            action_type,
            goal.id,
            reason,
        )

    def _finalize_async_action(
        self, goal, action_type: str, success: bool, result: str = ""
    ) -> None:
        """Commit a completed background action to goals, plans and learning."""
        if success:
            goal_engine = self._get_goal_engine()
            if goal_engine is not None:
                goal_engine.mark_action(goal.id, action_type)
            self._last_action_time[goal.id] = time.time()
            self._broadcast(
                f"[Action:{action_type}] {goal.topic[:40]} - {result[:80]}"
            )
        self._record_plan_outcome(goal, action_type, success, result)

        prediction = self._pending_predictions.pop(
            (str(goal.id), action_type), None
        )
        if prediction is not None:
            try:
                from cognition.consequence_tracker import get_consequence_tracker
                report = {
                    "goal_id": goal.id,
                    "goal_topic": goal.topic,
                    "action": action_type,
                    "success": success,
                    "result": result,
                }
                get_consequence_tracker(self._o).resolve(prediction, report)
            except Exception as e:
                logger.debug(
                    "[GoalActionExecutor] async consequence resolution error: %s", e
                )

    def _has_semantic_memory(self) -> bool:
        sem = getattr(self._o, 'semantic_memory', None)
        return sem is not None and hasattr(sem, 'get_related')

    def _has_search(self) -> bool:
        """Test actual search function (get_results), not the non-existent SearchManager."""
        try:
            from cognition.research_mcp.search_providers import get_results  # noqa
            return True
        except ImportError:
            return False

    def _has_chaos_memory(self) -> bool:
        """Return True if the memory backend supports get_chaos_episodic()."""
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ms = getattr(ai_sys, 'memory_system', None)
            return ms is not None and hasattr(ms, 'get_chaos_episodic')
        except Exception:
            return False

    def _should_ask_user(self, goal) -> bool:
        """
        Pressure-based trigger for USER_QUESTION outside the rotation.

        Fires when ALL conditions are met:
          1. User is reachable (recently active)
          2. Topic has not been asked recently (per-topic cooldown)
          3. Either: social/relational pressure is high, OR
                     goal topic is social/relational in nature
          4. Quota not exceeded
        """
        if self._user_questions_sent >= MAX_USER_QUESTIONS_PER_SESSION:
            return False
        if not self._user_recently_active():
            return False

        topic_key = goal.topic[:40].lower().replace(' ', '_').replace(' ', '_')
        last_asked = self._asked_topics.get(topic_key, 0)
        if time.time() - last_asked < USER_QUESTION_TOPIC_COOLDOWN:
            return False

        # Check if this goal's topic is inherently relational
        social_kws = {'relational', 'social', 'connect', 'interlocutor',
                      'user', 'person', 'human', 'conversation', 'together'}
        topic_lower = goal.topic.lower()
        if any(kw in topic_lower for kw in social_kws):
            return True

        # Check pressure system for elevated social/relational pressure
        try:
            ai_sys = getattr(self._o, 'ai_system', None)
            ps = getattr(ai_sys, 'pressure_system', None)
            if ps and hasattr(ps, 'pressure'):
                social_p = ps.pressure('social')
                if social_p > 0.60:
                    return True
        except Exception:
            pass

        return False

    def _user_recently_active(self) -> bool:
        """
        Returns True when it makes sense to send a proactive user question.

        Fire if: user was active recently (< 30 min) OR is in the re-engagement
        sweet spot (15 min–90 min idle). Blocks during very long absence (>90 min).

        Attribute map (confirmed from codebase):
          organism._last_interaction_ts  ← set by cognitive_organism on each turn
        """
        try:
            # Primary: organism._last_interaction_ts (confirmed attribute)
            last_ts = getattr(self._o, '_last_interaction_ts', None)
            # Fallback: ai_system._interaction_count doesn't give a timestamp,
            # so try the sleep_cycle idle time as proxy
            if last_ts is None:
                sc = getattr(self._o, 'sleep_cycle', None)
                idle_secs = getattr(sc, '_idle_seconds', 9999) if sc else 9999
            else:
                idle_secs = time.time() - last_ts

            return idle_secs < 1800 or (900 < idle_secs < 5400)
        except Exception:
            return True   # default: allow question

    def _add_thought(self, content: str, source: str = "goal_action",
                     priority: float = 0.45) -> None:
        """Add a thought to ThoughtStream (using 'content' key) and GlobalWorkspace."""
        try:
            ts = getattr(self._o, 'thought_stream', None)
            if ts and hasattr(ts, 'add'):
                ts.add(content=content, source=source, priority=priority,
                       thought_type="goal_action")
        except Exception:
            pass
        self._broadcast(content, priority=priority)

    def _broadcast(self, content: str, priority: float = 0.45) -> None:
        """Broadcast to GlobalWorkspace."""
        try:
            ws = getattr(self._o, 'workspace', None)
            if ws:
                ws.broadcast(
                    source="goal_action",
                    content=content[:200],
                    priority=min(0.75, priority),
                )
        except Exception:
            pass

    @staticmethod
    def _fire(coro) -> None:
        """
        FIX 3: Schedule async coroutine from sync context.

        The correct pattern for sync→async bridging from a background/daemon
        thread is to spawn a new daemon thread and call asyncio.run() inside it.
        asyncio.run() creates a fresh event loop, runs the coroutine to
        completion, and tears the loop down. This is completely independent
        from NiceGUI's event loop and is guaranteed to actually execute.

        DO NOT use asyncio.ensure_future() from a thread with no running loop —
        it silently discards the coroutine (Python emits RuntimeWarning:
        coroutine was never awaited and execution never happens).
        """
        def _runner():
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.warning(f"[GoalActionExecutor] async action error: {e}")

        t = threading.Thread(target=_runner, daemon=True, name="gae-async")
        t.start()
