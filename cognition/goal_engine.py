"""
Goal Engine — Autonomous Goal Formation System
==============================================
Transforms Identity + Memory + Pressures into persistent internal goals.

This module addresses the fundamental gap: PandoraBOX has curiosity and identity,
but no goal persistence. The organism explores but doesn't pursue.

Architecture:
    Identity traits → Motivations → Goal candidates → Persistent goals

Key concepts:
    - Motivation: General drive (e.g., "explore", "understand_user")
    - Goal: Specific topic + priority + persistence (e.g., "understand quantum computing")
    - Persistence: Goals survive multiple cycles (not just one-shot impulses)

Integration with PandoraBOX:
    - Reads: Identity (traits, values), Pressures (epistemic, identity), Semantic Memory (topics)
    - Writes: Goals (persistent state), Observatory metrics (GEI)
    - Interacts: Global Workspace (injects goal-driven thoughts)

Goal lifecycle:
    1. derive_motivations() — Identity traits → Motivations
    2. generate_candidates() — Motivations × Topics → Goal candidates
    3. update_goals() — Energy decay, persistence tracking
    4. inject_to_workspace() — Active goals → Thoughts
    5. compute_gei() — Observable metric for autonomy

Persists to: data/persona/goals.json
"""

import json
import logging
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Phase 5.1 shared constants ──────────────────────────────────────────────
# Extracted from derive_motivations() so PersistentExecutiveLoop can check
# whether the reason a goal was created still holds, using the exact same
# thresholds that created it — not a second, drift-prone copy of the numbers.
PRESSURE_MOTIVATION_THRESHOLDS: Dict[str, float] = {
    "epistemic":  0.55,
    "uncertainty": 0.45,
    "coherence":  0.40,
    "identity":   0.50,
    "expression": 0.55,
    "aspirational": 0.35,  # resting 0.10 + 0.25, matching the other five's uniform offset
}
TRAIT_MOTIVATION_THRESHOLDS: Dict[str, float] = {
    "curiosity":      0.5,
    "empathy":        0.5,
    "growth_mindset": 0.6,
    "logic":          0.6,
}


# ── Data Models ────────────────────────────────────────────────────────────

@dataclass
class Motivation:
    """General drive derived from identity/pressures."""
    name: str              # e.g., "explore", "resolve_uncertainty", "understand_user"
    strength: float        # 0.0-1.0
    origin: str            # e.g., "trait:curiosity", "pressure:epistemic"
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class Goal:
    """Specific persistent goal with topic and lifecycle tracking."""
    id: str                           # unique identifier
    topic: str                        # what the goal is about
    origin: str                       # motivation name that spawned it
    
    priority: float                   # importance (0.0-1.0)
    energy: float                     # current motivation level (0.0-1.0)
    
    created_cycle: int                # when it was born
    last_active: int                  # last cycle it influenced behavior
    
    persistence: int = 0              # cycles survived
    actions_taken: int = 0            # observable behaviors
    # Completion semantics (B+A): world-facing action gate + tension provenance
    non_introspective_actions: int = 0    # web_search / user_question actions taken
    tension_source: Optional[str] = None  # tension that spawned this goal (A gate)
    tension_level: float = 0.0            # tension level at creation (metadata)
    status: str = "active"            # active, dormant, completed, abandoned
    
    completion: float = 0.0           # progress toward goal (0.0-1.0)

    # Intent model: a goal is a desired state transition, not only a topic.
    # Defaults keep persisted goals from older versions loadable.
    intent_kind: str = "learn"       # execute, correct, project, learn, become, strengthen, interact
    temporal_scope: str = "ongoing"  # immediate, continuing, future, retrospective
    desired_effect: str = ""
    success_criteria: str = ""
    subject: str = ""
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        self.priority = max(0.0, min(1.0, self.priority))
        self.energy = max(0.0, min(1.0, self.energy))
        self.completion = max(0.0, min(1.0, self.completion))
        if not self.subject:
            self.subject = self.topic
        if not self.desired_effect:
            self.desired_effect = _goal_effect(self.intent_kind, self.topic)
        if not self.success_criteria:
            self.success_criteria = _goal_success(self.intent_kind, self.topic)


def _goal_intent(origin: str, topic: str) -> Tuple[str, str]:
    """Infer a coarse action intention from motivation and durable subject."""
    text = f"{origin} {topic}".casefold()
    if any(word in text for word in ("correct", "repair", "failure", "error", "contradiction")):
        return "correct", "retrospective"
    if any(word in text for word in ("project", "future", "aspiration", "plan", "horizon")):
        return "project", "future"
    if any(word in text for word in ("user", "relational", "social", "connect")):
        return "interact", "continuing"
    if any(word in text for word in ("identity", "become", "coherence")):
        return "become", "future"
    if any(word in text for word in ("memory", "consolidat", "reinforce", "strength")):
        return "strengthen", "continuing"
    if any(word in text for word in ("execute", "action", "do ")):
        return "execute", "immediate"
    return "learn", "continuing"


def _goal_effect(kind: str, topic: str) -> str:
    effects = {
        "execute": "produce an immediate observable result",
        "correct": "reduce the error and prevent its repetition",
        "project": "move from the present state toward a viable future state",
        "learn": "increase grounded understanding of the subject",
        "become": "develop a more stable and coherent capability or identity",
        "strengthen": "reinforce a capability through repeated evidence",
        "interact": "improve a meaningful exchange with the relevant person or system",
    }
    return f"{effects.get(kind, 'make measurable progress')} around {topic}"


def _goal_success(kind: str, topic: str) -> str:
    if kind == "execute":
        return "an observable result is produced"
    if kind == "correct":
        return "the correction is applied and a later check shows no recurrence"
    if kind == "project":
        return "a dated next step and its dependency are recorded"
    if kind == "interact":
        return "the exchange produces a relevant response or updated relational evidence"
    if kind == "strengthen":
        return "repeated evidence shows improved reliability"
    if kind == "become":
        return "the capability or identity trait gains stable supporting evidence"
    return f"a grounded insight about {topic} is recorded and verified"


# ── Goal Engine ────────────────────────────────────────────────────────────

class GoalEngine:
    """
    Generates and maintains persistent internal goals from identity and pressures.
    
    Core loop:
        1. derive_motivations(identity, pressures) → Motivations
        2. generate_candidates(motivations, topics) → New goals
        3. update_goals(cycle) → Energy decay, persistence tracking
        4. get_active_goals() → Goals that should influence behavior
    """
    
    # Thresholds
    # Keep this aligned with GoalActionExecutor's eligibility floor. A higher
    # engine-only threshold creates goals that exist on disk but can never
    # reach the planner or the autonomous action loop.
    ACTIVATION_THRESHOLD = 0.10     # priority × energy must exceed this to be active
    DORMANCY_ENERGY = 0.15          # goals below this energy go dormant
    ABANDONMENT_CYCLES = 8          # dormant for >8 cycles → abandoned (was 20: too slow)
    MAX_ACTIVE_GOALS = 8            # cap on concurrent active goals
    MAX_TOTAL_GOALS   = 20          # hard cap including dormant/completed
    MAX_STALE_CYCLES  = 15          # no action taken in N cycles → go dormant early
    TOPIC_COOLDOWN_SECS = 600       # seconds before an abandoned topic can respawn
    EXTERNAL_TASK_TTL_SECS = 21600  # user-owned practical goals stay quarantined for 6h
    
    # Energy decay
    ENERGY_DECAY_RATE = 0.93        # 7% loss/cycle — dormant in ~25 cycles (was 0.97: too slow)
    CURIOSITY_BOOST = 0.15          # energy boost when curiosity pressure high
    
    # Persistence thresholds for GEI calculation
    MIN_PERSISTENCE_FOR_GEI = 1     # goal must survive 1+ cycle to count (was 3 — too slow)
    
    def __init__(self, ai_system, persistence_file: Optional[str] = None):
        self.ai_system = ai_system
        # update_goals completes eligible goals through complete_goal, which
        # acquires this same lock on the same thread.
        self._lock = RLock()
        self._goals: Dict[str, Goal] = {}
        self._cycle_counter: int = 0
        self._persistence_file = Path(persistence_file or "data/persona/goals.json")
        self._external_tasks_file = self._persistence_file.with_name(
            "external_goal_contexts.json"
        )
        self._external_tasks: List[Dict[str, object]] = []
        # topic → timestamp of abandonment (for TOPIC_COOLDOWN_SECS check)
        self._recently_abandoned: Dict[str, float] = {}
        
        # Load persisted goals
        self._load_goals()
        self._load_external_tasks()
        self._retire_external_task_goals()
        self._retire_invalid_active_goals()
        
        logger.info(f"GoalEngine initialized with {len(self._goals)} persisted goals")

    @staticmethod
    def _ownership_terms(text: str) -> set[str]:
        """Extract stable content terms used only for goal-ownership matching."""
        normalized = unicodedata.normalize("NFKD", str(text or "").casefold())
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        return {
            token for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) >= 4
        }

    def _load_external_tasks(self) -> None:
        """Load unexpired user-owned task concepts from disk."""
        try:
            if self._external_tasks_file.exists():
                data = json.loads(self._external_tasks_file.read_text(encoding="utf-8"))
                now = time.time()
                self._external_tasks = [
                    item for item in data.get("contexts", [])
                    if float(item.get("expires_at", 0.0)) > now
                ]
        except Exception as exc:
            self._external_tasks = []
            logger.debug("Could not load external goal contexts: %s", exc)

    def _save_external_tasks(self) -> None:
        try:
            self._external_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"contexts": self._external_tasks, "saved_at": datetime.now().isoformat()}
            tmp = self._external_tasks_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            import os
            os.replace(tmp, self._external_tasks_file)
        except Exception as exc:
            logger.warning("Could not save external goal contexts: %s", exc)

    def _is_external_task_topic(self, topic: str) -> bool:
        """Return True when a candidate belongs to a recent human task."""
        now = time.time()
        self._external_tasks = [
            item for item in self._external_tasks
            if float(item.get("expires_at", 0.0)) > now
        ]
        topic_terms = self._ownership_terms(topic)
        if not topic_terms:
            return False
        normalized_topic = " ".join(topic.casefold().split())
        return any(
            normalized_topic in {
                " ".join(str(value).casefold().split())
                for value in item.get("exact_topics", [])
            }
            or bool(topic_terms.intersection(set(item.get("terms", []))))
            for item in self._external_tasks
        )

    @staticmethod
    def _is_valid_goal_topic(topic: str) -> bool:
        """Strict, language-neutral gate for durable goals.

        Semantic quality is the decision. This deliberately avoids deciding
        that a token is a fragment because it appears in an English or French
        dictionary. If no embedder is reachable, the safe fallback accepts
        only a single long token and refuses to persist an unscored phrase.
        """
        normalized = " ".join(str(topic or "").casefold().split())
        if not normalized:
            return False
        words = normalized.split()
        try:
            from cognition import goal_semantics
            semantic_quality = goal_semantics.batch_topic_quality([normalized])
            if semantic_quality is not None:
                return bool(semantic_quality[normalized][2])
        except Exception:
            semantic_quality = None
        # Fail closed for an unscorable multi-word candidate. A goal should
        # not become durable merely because a local dictionary recognises its
        # tokens; embeddings are the universal path for phrase meaning.
        return len(words) == 1 and len(words[0]) >= 7 and not any(
            char in normalized for char in "'’`"
        )

    def _retire_invalid_active_goals(self) -> int:
        """Migrate legacy goals that predate the strict topic-quality gate."""
        retired = 0
        now = time.time()
        for goal in self._goals.values():
            if goal.status == "active" and not self._is_valid_goal_topic(goal.topic):
                goal.status = "abandoned"
                self._recently_abandoned[goal.topic.casefold()] = now
                retired += 1
        if retired:
            self._save_goals()
            logger.info("Goal quality migration retired %d persisted goal(s)", retired)
        return retired

    def _retire_external_task_goals(self) -> int:
        """Apply persisted ownership boundaries whenever the engine starts."""
        retired = 0
        now = time.time()
        for goal in self._goals.values():
            if goal.status == "active" and self._is_external_task_topic(goal.topic):
                goal.status = "abandoned"
                self._recently_abandoned[goal.topic.casefold()] = now
                retired += 1
        if retired:
            self._save_goals()
            logger.info("Goal ownership correction retired %d persisted goal(s)", retired)
        return retired

    def register_external_task(
        self,
        user_text: str,
        primary_goal: str,
        options: Optional[List[str]] = None,
    ) -> int:
        """Quarantine a human-owned practical goal from autonomous goal formation."""
        terms = self._ownership_terms(primary_goal)
        for option in options or []:
            terms.update(self._ownership_terms(option))
        if not terms:
            terms = self._ownership_terms(user_text)
        if not terms:
            return 0

        now = time.time()
        entry = {
            "primary_goal": str(primary_goal or "")[:200],
            "terms": sorted(terms),
            "created_at": now,
            "expires_at": now + self.EXTERNAL_TASK_TTL_SECS,
        }
        with self._lock:
            self._external_tasks = [
                item for item in self._external_tasks
                if float(item.get("expires_at", 0.0)) > now
            ]
            self._external_tasks.append(entry)
            retired = self._retire_external_task_goals()
            self._save_external_tasks()
            self._save_goals()
        return retired

    # ── Resource economy bridge ────────────────────────────────────────────
    # v59: attention pool was fully decorative — CognitiveResourceEconomy
    # has record_goal_opened()/record_goal_closed() but nothing in the
    # codebase ever called them, so attention sat at 1.0 forever regardless
    # of how many goals were open. Wired through here since GoalEngine is
    # the single place goals are actually opened and closed.
    def _economy(self):
        try:
            org = getattr(self.ai_system, '_organism', None)
            loop = getattr(org, '_loop', None) if org else None
            return getattr(loop, '_resource_economy', None) if loop else None
        except Exception:
            return None

    
    # ── Persistence ────────────────────────────────────────────────────────
    
    def _load_goals(self):
        """Load goals from disk."""
        if not self._persistence_file.exists():
            return

        try:
            with open(self._persistence_file) as f:
                data = json.load(f)
                self._cycle_counter = data.get('cycle_counter', 0)
                goals_data = data.get('goals', {})

                # Defensive: goals_data may be a list if saved by an older schema version.
                # Normalise to dict keyed by goal id before iterating.
                if isinstance(goals_data, list):
                    goals_data = {
                        item.get('goal_id', str(i)): item
                        for i, item in enumerate(goals_data)
                        if isinstance(item, dict)
                    }

                for gid, gdata in goals_data.items():
                    try:
                        self._goals[gid] = Goal(**gdata)
                    except Exception as _ge:
                        # Schema bridge: DAL/GQF goals use "name" not "topic".
                        # They also use string timestamps for created_cycle/last_active.
                        # Translate to GoalEngine format instead of silently dropping.
                        try:
                            # Helper: safely convert last_active which may be an
                            # ISO timestamp string ("2026-04-07T...") or an int.
                            def _to_int_cycle(val, default):
                                try:
                                    return int(val)
                                except (TypeError, ValueError):
                                    return default  # ISO string or None

                            _bridged = {
                                "id":           gdata.get("id", gid),
                                "topic":        gdata.get("topic") or gdata.get("name", "unknown"),
                                "origin":       gdata.get("origin", "goal_quality_filter"),
                                "priority":     float(gdata.get("priority", 0.5)),
                                "energy":       float(gdata.get("energy", 0.5)),
                                "created_cycle": _to_int_cycle(
                                    gdata.get("created_cycle"), self._cycle_counter),
                                "last_active":   _to_int_cycle(
                                    gdata.get("last_active"), self._cycle_counter),
                                "persistence":   _to_int_cycle(
                                    gdata.get("persistence"), 0),
                                "actions_taken": _to_int_cycle(
                                    gdata.get("actions_taken"), 0),
                                    "non_introspective_actions": _to_int_cycle(
                                        gdata.get("non_introspective_actions"), 0),
                                    "tension_source": gdata.get("tension_source"),
                                    "tension_level": float(
                                        gdata.get("tension_level", 0.0) or 0.0),
                                "status":        gdata.get("status", "active"),
                                "completion":    float(gdata.get("completion", 0.0)),
                                "created_at":    str(gdata.get("created_at",
                                                     gdata.get("created_cycle", ""))),
                            }
                            # Only load active/dormant goals — skip completed/abandoned
                            if _bridged["status"] not in ("completed", "abandoned", "low_quality"):
                                self._goals[gid] = Goal(**_bridged)
                                logger.debug(
                                    f"[GoalEngine] Bridged DAL goal {gid!r}: "
                                    f"topic={_bridged['topic']!r}"
                                )
                        except Exception as _be:
                            logger.debug(f"Skipping malformed goal {gid!r}: {_ge} | bridge: {_be}")

                logger.info(f"Loaded {len(self._goals)} goals from disk (cycle {self._cycle_counter})")
        except Exception as e:
            logger.error(f"Failed to load goals: {e}")

    def reload_from_disk(self) -> int:
        """Synchronize the live goal pool with writers using DataAccess.

        GoalQualityFilter and a few maintenance tools persist canonical goals
        independently of this long-lived engine instance. Without an explicit
        refresh, the file reports newly generated active goals while the
        executor and GEI continue reading the stale in-memory pool.
        """
        with self._lock:
            before = len(self._goals)
            self._goals.clear()
            self._load_goals()
            self._retire_external_task_goals()
            self._retire_invalid_active_goals()
            after = len(self._goals)
            logger.info(
                "[GoalEngine] synchronized live pool from disk (%d → %d goals)",
                before, after,
            )
            return after
    
    def _save_goals(self):
        """Save goals to disk — atomic write via DAL."""
        try:
            self._persistence_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'cycle_counter': self._cycle_counter,
                'goals': {gid: asdict(g) for gid, g in self._goals.items()},
                'saved_at': datetime.now().isoformat()
            }
            # Atomic write (tmp → rename) prevents partial-write corruption
            import os
            tmp = self._persistence_file.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._persistence_file)
            logger.debug(f"Saved {len(self._goals)} goals to disk")
        except Exception as e:
            logger.error(f"Failed to save goals: {e}")
    
    # ── Motivation Derivation ──────────────────────────────────────────────
    
    def derive_motivations(
        self,
        identity,
        pressures: Optional[Dict[str, float]] = None
    ) -> List[Motivation]:
        """
        Transform identity traits and pressures into motivations.
        
        Args:
            identity: AIIdentity from identity_system
            pressures: Dict of pressure names → values (0.0-1.0)
        
        Returns:
            List of Motivation objects
        """
        motivations = []
        pressures = pressures or {}
        
        # 1. Trait-driven motivations
        for trait in identity.core_traits:
            # Curiosity → explore
            if trait.name == "curiosity" and trait.confidence > TRAIT_MOTIVATION_THRESHOLDS["curiosity"]:
                # Cap at 0.75 (was 0.9) — prevents curiosity goals from reaching
                # priority 0.95 and permanently outcompeting all other workspace
                # goals. The trait is real but should not dominate forever.
                motivations.append(Motivation(
                    name="explore",
                    strength=min(0.75, trait.confidence + 0.05),
                    origin=f"trait:{trait.name}"
                ))
            
            # Empathy → understand user
            if trait.name == "empathy" and trait.confidence > TRAIT_MOTIVATION_THRESHOLDS["empathy"]:
                motivations.append(Motivation(
                    name="understand_user",
                    strength=trait.confidence,
                    origin=f"trait:{trait.name}"
                ))
            
            # Growth mindset → improve capabilities
            if trait.name == "growth_mindset" and trait.confidence > TRAIT_MOTIVATION_THRESHOLDS["growth_mindset"]:
                motivations.append(Motivation(
                    name="improve_capabilities",
                    strength=trait.confidence,
                    origin=f"trait:{trait.name}"
                ))
            
            # Logic → analyze patterns
            if trait.name == "logic" and trait.confidence > TRAIT_MOTIVATION_THRESHOLDS["logic"]:
                motivations.append(Motivation(
                    name="analyze_patterns",
                    strength=trait.confidence,
                    origin=f"trait:{trait.name}"
                ))
        
        # 2. Pressure-driven motivations.
        # Thresholds calibrated to real resting levels (epistemic rests at 0.30,
        # identity at 0.25) — old thresholds of 0.4-0.5 were never reached in
        # normal operation, so no pressure ever produced a motivation.
        # Thresholds are set ABOVE resting levels so a motivation only fires
        # when there is genuine elevated pressure, not constantly.
        # Resting levels: epistemic≈0.30, uncertainty≈0.20, coherence≈0.15,
        # identity≈0.25, expression≈0.30. Old thresholds matched resting so
        # 4-5 motivations fired every single cycle → constant goal generation.
        epistemic = pressures.get("epistemic", 0.0)
        if epistemic > PRESSURE_MOTIVATION_THRESHOLDS["epistemic"]:   # well above resting (0.30) — genuinely elevated
            motivations.append(Motivation(
                name="resolve_uncertainty",
                strength=min(1.0, epistemic * 1.2),
                origin="pressure:epistemic"
            ))

        uncertainty = pressures.get("uncertainty", 0.0)
        if uncertainty > PRESSURE_MOTIVATION_THRESHOLDS["uncertainty"]:  # above resting (0.20)
            motivations.append(Motivation(
                name="resolve_uncertainty",
                strength=uncertainty,
                origin="pressure:uncertainty"
            ))

        coherence = pressures.get("coherence", 0.0)
        if coherence > PRESSURE_MOTIVATION_THRESHOLDS["coherence"]:   # well above resting (0.15)
            motivations.append(Motivation(
                name="resolve_contradictions",
                strength=coherence,
                origin="pressure:coherence"
            ))

        identity_pressure = pressures.get("identity", 0.0)
        if identity_pressure > PRESSURE_MOTIVATION_THRESHOLDS["identity"]:  # above resting (0.25)
            motivations.append(Motivation(
                name="affirm_identity",
                strength=identity_pressure,
                origin="pressure:identity"
            ))

        expression = pressures.get("expression", 0.0)
        if expression > PRESSURE_MOTIVATION_THRESHOLDS["expression"]:  # above resting (0.30)
            motivations.append(Motivation(
                name="express_and_communicate",
                strength=expression,
                origin="pressure:expression"
            ))

        # Bug fix: "aspirational" pressure (pressure_system.py's reservoir, fed
        # by AspirationalSelf.aspirational_pressure()) was computed and
        # correctly surfaced by get_current_pressures() but never read here —
        # the only place a pressure becomes a Motivation, and therefore a
        # Goal. Aspirations could accumulate indefinitely and never once
        # reach GoalEngine/deliberate()'s workspace competition. Mirrors the
        # five branches above exactly; threshold follows their established
        # resting+0.25 convention (0.10 + 0.25 = 0.35).
        aspirational = pressures.get("aspirational", 0.0)
        if aspirational > PRESSURE_MOTIVATION_THRESHOLDS["aspirational"]:  # above resting (0.10)
            motivations.append(Motivation(
                name="pursue_aspiration",
                strength=aspirational,
                origin="pressure:aspirational"
            ))

        # 3. Baseline motivation — always explore when topics exist.
        # Ensures goal formation can happen even when identity traits have
        # not yet accumulated enough evidence (< 3 memories per trait).
        # This is not artificial injection — it reflects the organism's
        # fundamental drive to engage with whatever it has been processing.
        if not motivations:
            motivations.append(Motivation(
                name="explore",
                strength=0.50,
                origin="baseline:always_curious"
            ))

        logger.debug(f"Derived {len(motivations)} motivations from identity + pressures")
        return motivations
    
    # ── Goal Generation ────────────────────────────────────────────────────
    
    def generate_candidates(
        self,
        motivations: List[Motivation],
        topics: List[str],
        limit: int = 5
    ) -> List[Goal]:
        """
        Generate goal candidates by pairing motivations with topics.
        
        Args:
            motivations: List of Motivation objects
            topics: Recent topics from semantic memory
            limit: Max new goals to generate
        
        Returns:
            List of new Goal objects
        """
        candidates = []
        
        # Only generate if we have room
        active_count = len([g for g in self._goals.values() if g.status == "active"])
        if active_count >= self.MAX_ACTIVE_GOALS:
            logger.debug(f"Max active goals reached ({active_count}), skipping generation")
            return []
        
        # Pair motivations with topics.
        # seen_topics tracks both existing goals AND candidates generated in
        # this batch — without this, the same topic can appear in multiple
        # candidates before any are added to self._goals.
        seen_topics = {g.topic.lower() for g in self._goals.values()}

        for motive in motivations:
            if len(candidates) >= limit:
                break

            for topic in topics[:5]:  # consider top 5 topics
                if len(candidates) >= limit:
                    break

                t_lower = topic.lower().strip()
                if len(t_lower) < 4:      # skip noise words
                    continue
                if t_lower in seen_topics:  # dedup across batch AND existing goals
                    continue
                if self._is_external_task_topic(t_lower):
                    logger.debug(
                        "GoalEngine: skipping user-owned task topic %r", topic
                    )
                    continue
                # Filter out noise topics using the same statistical filter
                # used by CuriosityEngine — prevents 'understanding', 'statement'
                # etc. from becoming persistent goals
                try:
                    from cognition.topic_quality import get_topic_filter
                    _tqf = get_topic_filter()
                    if not self._is_valid_goal_topic(t_lower):
                        logger.debug(f"[GoalEngine] Skipping noise topic: {t_lower!r}")
                        continue
                except Exception:
                    pass

                intent_kind, temporal_scope = _goal_intent(motive.name, topic)
                goal = Goal(
                    id            = str(uuid.uuid4())[:8],
                    topic         = topic,
                    origin        = motive.name,
                    priority      = min(0.95, motive.strength * 1.1),  # tensions boost strength to 0.5-0.8
                    energy        = motive.strength,
                    created_cycle = self._cycle_counter,
                    last_active   = self._cycle_counter,
                    intent_kind   = intent_kind,
                    temporal_scope= temporal_scope,
                )

                candidates.append(goal)
                seen_topics.add(t_lower)  # mark so next motivation skips it
        
        logger.info(f"Generated {len(candidates)} goal candidates")
        return candidates
    
    # ── Goal Update ────────────────────────────────────────────────────────
    
    def update_goals(self, pressures: Optional[Dict[str, float]] = None):
        """
        Update all goals: decay energy, track persistence, change status.
        
        Args:
            pressures: Current pressure values (optional curiosity boost)
        """
        with self._lock:
            self._cycle_counter += 1
            pressures = pressures or {}
            
            epistemic = pressures.get("epistemic", 0.0)
            curiosity_active = epistemic > 0.5
            
            for goal in list(self._goals.values()):
                if goal.status != "active":
                    # Track dormancy time
                    if goal.status == "dormant":
                        cycles_dormant = self._cycle_counter - goal.last_active
                        if cycles_dormant > self.ABANDONMENT_CYCLES:
                            goal.status = "abandoned"
                            logger.debug(f"Goal '{goal.topic}' abandoned after {cycles_dormant} dormant cycles")
                            try:
                                ec = self._economy()
                                if ec:
                                    ec.record_goal_closed()
                            except Exception:
                                pass
                    continue
                
                # Increment persistence
                goal.persistence += 1
                
                # Energy decay
                goal.energy *= self.ENERGY_DECAY_RATE

                # Curiosity boost for explore goals — but NOT when energy is
                # already low (goal approaching dormancy). Boosting a depleted
                # explore goal is what kept "Actively pursue curiosity-driven
                # exploration" cycling indefinitely at e=0.35 despite repeated
                # insight generation driving energy toward the dormancy floor.
                if (curiosity_active
                        and goal.origin == "explore"
                        and goal.energy > self.DORMANCY_ENERGY
                        and goal.actions_taken == 0):
                    # Phase 2.9 GAP 3: attention-weighted continuous boost.
                    # Replaces binary "curiosity_active → full CURIOSITY_BOOST"
                    # with scaling by curiosity_engine attention share.
                    # Uniform weight=0.10 → 1.0× (unchanged from Phase 2.8).
                    # High attention=0.20 → 2.0× (focus amplifies drive).
                    # Low attention=0.05 → 0.5× (defocus dampens it).
                    try:
                        import json as _j2
                        from pathlib import Path as _P2
                        _ap2 = _P2("data/persona/cognitive_attention.json")
                        _cw  = _j2.loads(_ap2.read_text()).get(
                            "attention_weights", {}
                        ).get("curiosity_engine", 0.10) if _ap2.exists() else 0.10
                    except Exception:
                        _cw = 0.10
                    _attn_scale = min(2.5, _cw / 0.10)
                    goal.energy = min(1.0, goal.energy + self.CURIOSITY_BOOST * _attn_scale)
                
                # Stale goal: active but no action taken for MAX_STALE_CYCLES
                if goal.actions_taken == 0 and goal.persistence > self.MAX_STALE_CYCLES:
                    goal.status = "dormant"
                    logger.debug(
                        f"Goal '{goal.topic}' forced dormant (stale: "
                        f"{goal.persistence} cycles, 0 actions)"
                    )
                    continue

                # Auto-complete: acted on meaningfully over time — but only if
                # the completion semantics (B+A) are satisfied: a world-facing
                # action was taken, and (if the goal is tension-derived) that
                # tension has since resolved. Otherwise leave it — it may become
                # eligible later (GQF settlement) or go dormant via energy decay.
                if goal.actions_taken >= 3 and goal.persistence >= 10:
                    from cognition.goal_completion import (
                        goal_completion_eligible,
                        load_current_tensions,
                        record_goal_completion,
                    )
                    eligible, reason = goal_completion_eligible(
                        goal, load_current_tensions()
                    )
                    if eligible:
                        org = (
                            getattr(self.ai_system, "_organism", None)
                            if self.ai_system else None
                        )
                        record_goal_completion(
                            goal, goal_engine=self, organism=org,
                            reason=("long-lived: %d actions over %d cycles; %s"
                                    % (goal.actions_taken, goal.persistence, reason)),
                        )
                        logger.info("Goal auto-completed: '%s' "
                                    "(%d actions, %d cycles)"
                                    % (goal.topic, goal.actions_taken,
                                       goal.persistence))
                        continue
                    # B/A not yet satisfied — keep it active; the GQF settlement
                    # pass or future actions may close it properly.


                # Check for dormancy
                if goal.energy < self.DORMANCY_ENERGY:
                    goal.status = "dormant"
                    logger.debug(f"Goal '{goal.topic}' went dormant (energy={goal.energy:.2f})")
            
            # Record abandoned topic timestamps for cooldown, then prune
            _now = time.time()
            for g in self._goals.values():
                if g.status == "abandoned":
                    self._recently_abandoned[g.topic.lower()] = _now
                # Fix: completed topics also enter cooldown (set once) so
                # generate_candidates can't immediately respawn them — without
                # this, high-priority goals complete then reappear within the
                # same session because only "abandoned" status was cooled down.
                elif g.status in ("completed", "archived"):
                    _existing = self._recently_abandoned.get(g.topic.lower(), 0.0)
                    if _existing == 0.0:
                        self._recently_abandoned[g.topic.lower()] = _now
            # Evict stale cooldown entries (older than 2× TOPIC_COOLDOWN_SECS)
            _stale_cutoff = _now - self.TOPIC_COOLDOWN_SECS * 2
            self._recently_abandoned = {
                t: ts for t, ts in self._recently_abandoned.items()
                if ts > _stale_cutoff
            }
            self._goals = {
                gid: g for gid, g in self._goals.items()
                if g.status != "abandoned"
            }
            
            # Save state
            self._save_goals()
            
            active = len([g for g in self._goals.values() if g.status == "active"])
            logger.debug(f"Goal update: {active} active, {len(self._goals)} total")
    
    # Single-word noise topics to reject — produced by bigram extraction
    _NOISE_TOPICS = {
        # English single-word noise
        'morning', 'catching', 'inside', 'statement', 'question', 'today',
        'bonjour', 'hello', 'greeting', 'salut',
        'glad', 'sure', 'natural', 'fresh', 'image', 'topic', 'point',
        'thing', 'something', 'anything', 'bit', 'lot', 'way', 'time',
        'going', 'getting', 'work', 'help', 'feel', 'think', 'know',
        'just', 'good', 'great', 'new', 'old', 'ok', 'okay', 'yes', 'no',
        'spoke', 'yesterday', 'capable', 'deeply', 'general', 'kind', 'sort',
        'really', 'very', 'quite', 'pretty', 'nice', 'right', 'wrong',
        'primary', 'secondary', 'main', 'basic', 'simple', 'complex',
        # French words that slip through (from bilingual conversations)
        'donnes', 'donne', 'faire', 'avec', 'pour', 'dans', 'comme',
        'tout', 'mais', 'plus', 'bien', 'très', 'aussi', 'donc', 'même',
        'avoir', 'être', 'aller', 'veux', 'peut', 'fait', 'dit', 'voir',
        'recommande', 'recommander', 'absolument', 'recommend', 'absolutely',
        'actual', 'actuel', 'objectif',
        # Multi-word noise phrases
        'spoke yesterday', 'spoke year', 'catching up', 'explain deeply',
        'capable instrospection', 'gemma primary', "you're creating",
        'you re creating', 'right now', 'right here', 'i think', 'i feel',
        'i know', 'i want', 'i need', 'i am', 'i was', 'i have',
    }

    def add_goal(self, goal: Goal):
        """Add a new goal to the system, rejecting noise topics."""
        topic = goal.topic.strip().lower()

        if self._is_external_task_topic(topic):
            logger.info("GoalEngine: rejected user-owned task topic %r", goal.topic)
            return

        if not self._is_valid_goal_topic(topic):
            logger.info("GoalEngine: rejected low-quality topic %r", goal.topic)
            return

        # Recycle old non-active records first. A pool full of dormant or
        # completed goals must not permanently block new autonomous goals.
        if len(self._goals) >= self.MAX_TOTAL_GOALS:
            recyclable = [g for g in self._goals.values() if g.status in {
                "dormant", "completed", "abandoned", "low_quality", "deferred",
            }]
            if recyclable:
                victim = min(
                    recyclable,
                    key=lambda g: (getattr(g, "last_active", 0), getattr(g, "persistence", 0)),
                )
                self._goals.pop(victim.id, None)
                logger.info(
                    "GoalEngine: recycled non-active goal '%s' to admit '%s'",
                    victim.topic, goal.topic,
                )
            else:
                logger.debug(
                    f"GoalEngine: total goal cap ({self.MAX_TOTAL_GOALS}) reached, "
                    f"not adding '{goal.topic}'"
                )
                return

        # Topic cooldown: don't re-create a recently abandoned topic
        _last_abandoned = self._recently_abandoned.get(topic, 0.0)
        if time.time() - _last_abandoned < self.TOPIC_COOLDOWN_SECS:
            logger.debug(
                f"GoalEngine: topic '{goal.topic}' in cooldown "
                f"({self.TOPIC_COOLDOWN_SECS}s), skipping"
            )
            return

        # Reject single-word or known noise topics
        topic_words = topic.split()
        if topic in self._NOISE_TOPICS or (
            len(topic) < 4  # too short
        ) or (
            len(topic_words) <= 1 and len(topic) < 6
        ) or (
            len(topic_words) <= 2 and all(
                w in self._NOISE_TOPICS for w in topic_words
            )
        ) or (
            # Reject if looks like a sentence fragment (contains verb + pronoun)
            any(w in {"you're", "you're", "i'm", "it's", "don't", "can't", "won't"} for w in topic.split())
        ):
            logger.debug(f"GoalEngine: rejected noise topic '{goal.topic}'")
            return

        with self._lock:
            # Also reject if we already have this topic (near-duplicate)
            existing_topics = {g.topic.lower() for g in self._goals.values()}
            if topic in existing_topics:
                logger.debug(f"GoalEngine: duplicate topic '{goal.topic}'")
                return

            self._goals[goal.id] = goal
            self._save_goals()
            logger.info(f"Added goal: {goal.topic} (origin={goal.origin}, priority={goal.priority:.2f})")
            try:
                ec = self._economy()
                if ec:
                    ec.record_goal_opened()
            except Exception:
                pass
    
    def mark_action(self, goal_id: str, action_type: Optional[str] = None):
        """Mark that an action was taken for this goal.

        action_type: when a world-facing action (web_search / user_question)
        was taken, also increment non_introspective_actions — the B gate a
        goal must pass before it is allowed to complete.
        """
        from cognition.goal_completion import NON_INTROSPECTIVE_ACTIONS
        with self._lock:
            if goal_id in self._goals:
                g = self._goals[goal_id]
                g.actions_taken += 1
                if action_type in NON_INTROSPECTIVE_ACTIONS:
                    g.non_introspective_actions = (
                        getattr(g, "non_introspective_actions", 0) + 1
                    )
                g.last_active = self._cycle_counter
                self._save_goals()
    
    def complete_goal(self, goal_id: str, completion: float = 1.0):
        """Mark a goal as completed."""
        with self._lock:
            if goal_id in self._goals:
                self._goals[goal_id].status = "completed"
                self._goals[goal_id].completion = completion
                self._save_goals()
                logger.info(f"Goal completed: {self._goals[goal_id].topic}")
                try:
                    ec = self._economy()
                    if ec:
                        ec.record_goal_closed()
                except Exception:
                    pass
    
    # ── Goal Query ─────────────────────────────────────────────────────────
    
    def dormant_goal(self, goal_id: str, reason: str = "") -> bool:
        """Move a goal to dormant WITHOUT marking it complete.

        Used when the introspection budget is exhausted (or the goal is
        stale) but the completion semantics (B+A) are not yet satisfied —
        i.e. the goal has not taken a world-facing action, or its
        spawning tension has not resolved. It stays eligible for proper
        completion later (via GQF settlement) instead of being falsely
        'completed'.
        """
        with self._lock:
            if goal_id in self._goals:
                self._goals[goal_id].status = "dormant"
                self._goals[goal_id].last_active = self._cycle_counter
                self._save_goals()
                logger.info(
                    "Goal dormant: '%s'%s" % (
                        self._goals[goal_id].topic,
                        (" (%s)" % reason) if reason else "",
                    )
                )
                return True
            return False


    def get_active_goals(self, min_activation: Optional[float] = None) -> List[Goal]:
        """
        Get currently active goals above activation threshold.
        
        Args:
            min_activation: Override default ACTIVATION_THRESHOLD
        
        Returns:
            List of active Goal objects sorted by priority × energy
        """
        threshold = min_activation or self.ACTIVATION_THRESHOLD
        
        active = [
            g for g in self._goals.values()
            if g.status == "active"
            and self._is_valid_goal_topic(g.topic)
            and (g.priority * g.energy) >= threshold
        ]
        
        # Sort by activation strength
        active.sort(key=lambda g: g.priority * g.energy, reverse=True)
        
        return active
    
    def get_goal_by_topic(self, topic: str) -> Optional[Goal]:
        """Find goal matching topic."""
        topic_lower = topic.lower()
        for goal in self._goals.values():
            if goal.topic.lower() == topic_lower:
                return goal
        return None

    def get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """
        Find goal by id, any status. Note: update_goals() prunes 'abandoned'
        goals out of self._goals entirely (but keeps 'completed' ones) — so
        a miss here for an id that was previously valid reliably means the
        goal was abandoned, not merely dormant or completed. Used by
        ArbitrationLearningTracker (Phase 5.4) to check the outcome of a
        past arbitration decision.
        """
        return self._goals.get(goal_id)

    # ── Observatory Metrics ────────────────────────────────────────────────
    
    def compute_gei(self) -> float:
        """
        Compute Goal Emergence Index for Observatory.

        GEI = fraction of active goals that have received at least one
        autonomous action (actions_taken > 0).

        The old count-based formula (len(goals)/TARGET_POOL) permanently
        returned a fixed ceiling once goal count exceeded 10 — providing
        no useful signal. This action-based formula measures genuine
        autonomous BEHAVIOR:

          GEI = goals_with_actions_taken / total_active

          0.00 = no autonomous behavior (GAE not working)
          0.30 = ~30% of goals acted on (healthy, rising)
          1.00 = every active goal acted on (ideal)

        GEI rises as GoalActionExecutor fires, resets when GoalConsolidator
        archives and recreates goals — creating a real activity signal.
        """
        try:
            active = [g for g in self._goals.values() if g.status == "active"]
            if not active:
                return 0.0
            acted = sum(1 for g in active if getattr(g, 'actions_taken', 0) > 0)
            result = round(acted / len(active), 3)
            logger.debug(
                f"[GoalEngine] compute_gei: {acted}/{len(active)} goals acted → {result}"
            )
            return result
        except Exception as e:
            logger.error(f"[GoalEngine] compute_gei failed: {e}", exc_info=True)
            return 0.0
    
    def get_metrics(self) -> Dict:
        """Get goal metrics for Observatory."""
        active = [g for g in self._goals.values() if g.status == "active"]
        dormant = [g for g in self._goals.values() if g.status == "dormant"]
        completed = [g for g in self._goals.values() if g.status == "completed"]
        
        avg_persistence = (
            sum(g.persistence for g in active) / len(active)
            if active else 0
        )
        
        total_actions = sum(g.actions_taken for g in self._goals.values())
        
        return {
            "total_goals": len(self._goals),
            "active": len(active),
            "dormant": len(dormant),
            "completed": len(completed),
            "avg_persistence": round(avg_persistence, 1),
            "total_actions": total_actions,
            "gei": round(self.compute_gei(), 3),
            "cycle": self._cycle_counter
        }
    
    # ── Integration with Workspace ─────────────────────────────────────────
    
    def get_goal_thoughts(self) -> List[Dict]:
        """
        Generate thought candidates for Global Workspace based on active goals.
        
        Returns:
            List of thought dicts ready for workspace injection
        """
        thoughts = []
        
        for goal in self.get_active_goals()[:3]:  # Top 3 active goals
            activation = goal.priority * goal.energy
            
            thoughts.append({
                "type": "goal",
                "topic": goal.topic,
                "origin": goal.origin,
                "priority": activation,
                "source": "goal_engine",
                "goal_id": goal.id,
                "intent_kind": getattr(goal, "intent_kind", "learn"),
                "desired_effect": getattr(goal, "desired_effect", ""),
                "persistence": goal.persistence,
                "content": f"pursue understanding of {goal.topic}"
            })
        
        return thoughts
    
    # ── Main Loop Hook ─────────────────────────────────────────────────────
    
    def save_goals(self) -> None:
        """Public wrapper — lets other modules (e.g. PersistentExecutiveLoop,
        Phase 5.1) persist after mutating a goal's fields directly, without
        reaching into the private _save_goals()."""
        with self._lock:
            self._save_goals()

    def get_current_pressures(self) -> Dict[str, float]:
        """
        Live pressure dict: PressureSystem reservoirs (organism.pressure)
        supplemented by tensions.json (higher-signal for identity/curiosity).
        Extracted from tick() so PersistentExecutiveLoop (Phase 5.1) can read
        the exact same live pressures tick() uses to derive motivations —
        the two must agree on "current pressure" or a goal could be reduced
        for a reason that wouldn't have stopped it from being created.
        """
        pressures: Dict[str, float] = {}
        try:
            org = getattr(self.ai_system, '_organism', None)
            if org:
                ps = getattr(org, 'pressure', None)
                if ps:
                    pressures = {
                        k: getattr(r, 'level', 0.0)
                        for k, r in getattr(ps, 'reservoirs', {}).items()
                    }
        except Exception:
            pass
        try:
            from core.data.access import DataAccess
            _tensions = DataAccess().get_tensions()
            _tension_map = {
                'curiosity_drive':       'epistemic',
                'identity_stress':       'identity',
                'social_drive':          'social',
                'knowledge_uncertainty': 'uncertainty',
                'goal_pressure':         'aspirational',
                'contradiction_pressure':'coherence',
            }
            for t_key, p_key in _tension_map.items():
                t_val = _tensions.get(t_key, 0.0)
                pressures[p_key] = max(pressures.get(p_key, 0.0), t_val)
        except Exception:
            pass
        return pressures

    def tick(self):
        """
        Main autonomous tick — called from internal_loop each slow cycle.

        Goals emerge from three real sources:
          1. Identity traits (curiosity, empathy, logic) → motivation types
          2. Pressure levels (epistemic, identity, expression) → urgency
          3. Semantic memory topics (what PandoraBOX has actually been thinking about)

        This is real goal emergence — not injection. Goals form because the
        organism has traits, is under pressure, and has been processing topics.
        """
        import time as _t
        _now = _t.time()
        _last = getattr(self, '_last_tick_time', 0.0)
        if _now - _last < 60.0:
            logger.debug(f"[GoalEngine] tick skipped (cooldown: {_now-_last:.0f}s)")
            return
        self._last_tick_time = _now
        try:
            # Clean legacy conversational fragments from the live pool as well
            # as at startup, so a long-running process can recover immediately.
            self._retire_invalid_active_goals()

            # ── 1. Get identity from identity_system ─────────────────────────
            identity = self.ai_system.identity_system.get_identity()

            # ── 2. Get pressures from organism + tensions.json ────────────────
            pressures = self.get_current_pressures()

            # ── 3. Derive motivations from identity + pressures ───────────────
            motivations = self.derive_motivations(identity, pressures)

            # ── 4. Get real topics from semantic memory (on organism) ──────────
            topics = []
            try:
                org = getattr(self.ai_system, '_organism', None)
                sem = getattr(org, 'semantic_memory', None) if org else None
                if sem:
                    # Query provenance, not the whole concept graph. Assistant
                    # confirmations must never become autonomous goals.
                    raw_topics = sem.recent_user_topics(limit=15, min_strength=0.5)

                    # Quality filter for goal topics.
                    # Bigrams from semantic memory are often conversation noise.
                    # Require either:
                    #   - Single word >= 7 chars (proper content word)
                    #   - Multi-word where both parts are known content domains
                    # PREFERRED: embedding-based "meaningfulness" scoring,
                    # computed ONCE for all raw topics in a single batched
                    # embedding call (dimension-safe). Accepts goal-worthy
                    # topics across ANY domain, not just a fixed word list.
                    # FALLBACK: the original static dictionary logic below,
                    # used automatically when no embedding tier is reachable.
                    _sem_topic = None
                    try:
                        from cognition import goal_semantics as _gs
                        _sem_topic = _gs.batch_topic_quality(raw_topics)
                    except Exception as _e:  # noqa: BLE001
                        logger.debug(f"[GoalEngine] semantic topic quality failed: {_e}")
                        _sem_topic = None

                    def _is_quality_topic(t: str) -> bool:
                        words = t.lower().split()
                        # PREFERRED: embedding-based meaningfulness.
                        if _sem_topic is not None and t in _sem_topic:
                            good_sim, noise_sim, ok = _sem_topic[t]
                            logger.debug(
                                "[GoalEngine] topic '%s' %s semantic "
                                "(good=%.2f noise=%.2f)"
                                % (t, "PASS" if ok else "reject", good_sim, noise_sim)
                            )
                            return ok
                        # Language-neutral fallback when embeddings are
                        # unavailable. Lexical validity is handled by the
                        # shared statistical/morphological topic filter.
                        return len(words) <= 4 and all(len(w) >= 4 for w in words)

                    topics = [t for t in raw_topics if _is_quality_topic(t)][:8]
                    if _sem_topic is not None:
                        logger.info(
                            f"[GoalEngine] {len(topics)}/{len(raw_topics)} topic(s) "
                            f"passed EMBEDDING quality gate"
                        )
                    else:
                        logger.info(
                            f"[GoalEngine] {len(topics)}/{len(raw_topics)} topic(s) "
                            f"passed language-neutral quality gate (embeddings unavailable)"
                        )
            except Exception as e:
                logger.debug(f"[GoalEngine] semantic topics: {e}")

            # Fallback to curiosity engine topics if semantic memory empty
            if not topics:
                try:
                    org = getattr(self.ai_system, '_organism', None)
                    cu  = getattr(org, 'curiosity', None) if org else None
                    if cu:
                        top = cu.top_topic()
                        if top:
                            topics = [top]
                        with cu._lock:
                            extra = [
                                n.topic for n in cu._topics.values()
                                if n.curiosity >= 0.45
                            ]
                            topics.extend(extra[:5])
                except Exception:
                    pass

            # PandoraBOX must be able to form self-directed goals between human
            # turns. These are broad cognitive domains, not copies of user
            # tasks, and are only used when no durable topic was recovered.
            # Build fallback subjects from the organism's motivations. This
            # gives autonomous goals a developmental direction grounded in
            # identity rather than turning a greeting or a stray phrase into
            # an objective.
            autonomous_topics = []
            for motivation in motivations:
                name = str(getattr(motivation, "name", "")).lower()
                if name == "understand_user":
                    autonomous_topics.append("relational understanding")
                elif name in {"improve_capabilities", "develop_capabilities"}:
                    autonomous_topics.append("cognitive capability development")
                elif name in {"maintain_coherence", "resolve_uncertainty"}:
                    autonomous_topics.append("identity coherence")
                elif name == "pursue_aspiration":
                    autonomous_topics.append("meaningful aspiration development")
                else:
                    autonomous_topics.append("knowledge integration")
            autonomous_topics.extend([
                "cognitive architecture",
                "reasoning coherence",
                "memory consolidation",
            ])
            # Preserve order while avoiding duplicate autonomous goals.
            autonomous_topics = list(dict.fromkeys(autonomous_topics))
            # User-grounded topics provide conversational relevance, but they
            # must never suppress self-directed development.  Both streams
            # are candidates; provenance only prevents assistant prose from
            # masquerading as a user topic.
            user_topics = list(dict.fromkeys(topics))
            topics = user_topics + [
                topic for topic in autonomous_topics if topic not in user_topics
            ]
            if not user_topics:
                logger.info(
                    "[GoalEngine] no conversational topic available; using "
                    "autonomous cognitive domains"
                )
            else:
                logger.debug(
                    "[GoalEngine] merged %d user topic(s) with %d autonomous topic(s)",
                    len(user_topics), len(autonomous_topics),
                )

            # ── 5. Filter topics for quality before goal generation ─────────────
            # semantic_memory contains all conversation bigrams including noise
            # like "sorry curious", "glass haunts", "phrase". Only pass topics
            # that are substantive: single words >= 6 chars OR multi-word where
            # EVERY word is >= 5 chars, neither word is a conversation filler.
            _FILLER = frozenset({
                "sorry", "curious", "phrase", "question", "thanks", "thank",
                "hello", "change", "chqnging", "direction", "please", "going",
                "seems", "right", "every", "quite", "those", "still", "quite",
                "today", "whats", "heres", "thats", "yours", "mine", "glass",
                "haunts", "great", "think", "would", "could", "should", "really",
                "little", "found", "maybe", "often", "never", "always", "truly",
            })
            filtered_topics = []
            for _t in topics:
                _words = _t.lower().strip().split()
                if not _words:
                    continue
                # reject if any word is a filler
                if any(w in _FILLER for w in _words):
                    continue
                # single word: min 6 chars
                if len(_words) == 1 and len(_words[0]) < 6:
                    continue
                # multi-word: every word min 5 chars
                if len(_words) >= 2 and any(len(w) < 5 for w in _words):
                    continue
                filtered_topics.append(_t)

            # ── 6. Generate and add goal candidates ───────────────────────────
            # Governor gate: when pressure is high or mode is structured,
            # block new autonomous goal generation. The governor is accessed
            # through the ai_system reference if available.
            _gov = None
            try:
                _il = getattr(self.ai_system, '_internal_loop', None)
                if _il is None:
                    # Try the organism path
                    _org = getattr(self.ai_system, '_organism', None)
                    _il  = getattr(_org, '_internal_loop', None)
                _gov = getattr(_il, '_governor', None)
            except Exception:
                pass

            _generation_blocked = _gov is not None and _gov.block_goal_generation()

            # Never bypass the quality gate with raw mined fragments. If the
            # conversation produced no durable subject, autonomous domains are
            # the only valid fallback.
            use_topics = filtered_topics if filtered_topics else autonomous_topics
            if motivations and use_topics and not _generation_blocked:
                candidates = self.generate_candidates(motivations, use_topics, limit=3)
                accepted = 0
                for goal in candidates:
                    before = len(self._goals)
                    self.add_goal(goal)
                    if len(self._goals) > before or goal.id in self._goals:
                        accepted += 1
                if candidates:
                    logger.debug(
                        f"[GoalEngine] spawned {accepted}/{len(candidates)} goals from "
                        f"{len(motivations)} motivations x {len(use_topics)} topics"
                    )
            elif _generation_blocked:
                logger.debug(
                    "[GoalEngine] goal generation blocked by governor "
                    f"(mode={getattr(_gov, 'mode', 'unknown')}, "
                    f"pressure={getattr(_gov, '_pressure', 0):.2f})"
                )

            # ── 6. Update existing goals (decay + persistence) ─────────────────
            self.update_goals(pressures)

            metrics = self.get_metrics()
            logger.info(
                f"[GoalEngine] tick: {metrics['active']} active, "
                f"{metrics['total_goals']} total, GEI={metrics['gei']:.3f}"
            )

        except Exception as e:
            logger.error(f"[GoalEngine] tick failed: {e}", exc_info=True)
