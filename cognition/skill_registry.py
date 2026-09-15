"""
cognition/skill_registry.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SkillRegistry — persistent competencies that deepen with use.

The gap this closes
────────────────────
Lumina can research topics and store results as FAISS memories, but
has no mechanism to develop *skills* — persistent, callable competencies
that improve with use. Research results accumulate as facts, not as
evolving capability.

A human who repeatedly engages with music theory develops intuitions
about harmony that shape how they hear new pieces. Lumina's engagement
with a topic should similarly build structured capability, not just
add entries to a vector store.

What a Skill is
───────────────
  Skill(
    domain:         str    # e.g. "music_theory", "philosophy_of_mind"
    description:    str    # what this skill enables
    confidence:     float  # 0–1, grows with successful use
    depth:          int    # 0=surface, 1=working, 2=fluent, 3=deep
    last_practiced: float  # timestamp
    practice_count: int    # total uses
    sub_skills:     List[str]  # finer-grained competencies within domain
    source_memories: List[str] # memory IDs that informed this skill
    insights:       List[str]  # crystallised understanding (max 8)
  )

Skill lifecycle
───────────────
  Seeding     — a research result or autonomous reflection names a domain
                → skill created at confidence=0.25, depth=0

  Practice    — completed exchanges record exposure in the background;
                confidence changes only when an outcome signal is available

  Deepening   — at verified-success thresholds (5, 15, 40), depth increments
                and an LLM call generates new sub_skills and insights

  Decay       — skills not practiced in DECAY_DAYS begin losing confidence
                (floor = 0.15; skills don't disappear, they become rusty)

  Querying    — before responding on a topic, CognitiveOrganism queries
                the registry for relevant skills and injects them as
                [Skill context] in the prompt

Integration
───────────
  Seeded by:  AutonomousReflectionEngine, GoalActionExecutor research results
  Deepened by: post-response background observation
  Queried by:  CognitiveOrganism._build_prompt_additions()
  Persisted:  data/persona/skill_registry.json
"""

from __future__ import annotations

import json
import hashlib
import logging
import queue
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEPTH_THRESHOLDS    = [0, 5, 15, 40]   # verified outcomes → depth 0,1,2,3
DECAY_DAYS          = 14.0             # days without practice before decay starts
DECAY_RATE          = 0.006            # confidence lost per decay cycle
CONFIDENCE_FLOOR    = 0.15
CONFIDENCE_CEIL     = 0.97
MAX_INSIGHTS        = 8
MAX_SKILLS          = 60
MAX_ALIASES         = 16
SEMANTIC_MATCH_FLOOR = 0.34
BACKGROUND_IDLE_SECONDS = 12.0

_MATCH_STOPWORDS = {
    "about", "after", "again", "avec", "avoir", "been", "being", "cela",
    "cette", "comme", "dans", "does", "faire", "from", "have", "just",
    "mais", "more", "pour", "quoi", "sans", "that", "this", "tout",
    "tous", "tres", "une", "votre", "what", "when", "where", "which",
    "with", "would", "vous", "your",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Skill:
    domain:          str
    name:            str   = ""
    category:        str   = ""
    description:     str   = ""
    confidence:      float = 0.25
    proficiency:     float = 0.25
    confidence_prior: float = 0.25
    growth_trajectory: str = "emerging"
    depth:           int   = 0        # 0=surface 1=working 2=fluent 3=deep
    last_practiced:  float = field(default_factory=time.time)
    practice_count:  int   = 0
    verified_practices: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0
    sub_skills:      List[str] = field(default_factory=list)
    source_memories: List[str] = field(default_factory=list)
    insights:        List[str] = field(default_factory=list)
    transfer_links:  List[str] = field(default_factory=list)  # domains this transfers to
    aliases:         List[str] = field(default_factory=list)
    enriched_depth:  int = 0

    def depth_label(self) -> str:
        return ["surface", "working", "fluent", "deep"][self.depth]

    def prompt_fragment(self) -> str:
        """Compact skill context for prompt injection."""
        label = self.name or self.domain.replace("_", " ")
        parts = [
            f"[Skill: {label} — {self.depth_label()}, "
            f"confidence={self.confidence:.0%}]"
        ]
        if self.insights:
            parts.append("Key understanding: " + "; ".join(self.insights[:3]))
        if self.sub_skills:
            parts.append("Sub-skills: " + ", ".join(self.sub_skills[:4]))
        return "\n".join(parts)

    def is_rusty(self) -> bool:
        days_idle = (time.time() - self.last_practiced) / 86400
        return days_idle > DECAY_DAYS

    def next_depth_at(self) -> int:
        """Practice count needed to reach next depth level."""
        next_level = self.depth + 1
        if next_level >= len(DEPTH_THRESHOLDS):
            return 9999
        return DEPTH_THRESHOLDS[next_level]


@dataclass
class SkillState:
    skills: Dict[str, Dict] = field(default_factory=dict)  # domain → Skill dict
    candidates: Dict[str, Dict] = field(default_factory=dict)
    total_practices: int    = 0
    total_deepenings: int   = 0


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Maintains Lumina's growing competency map.

    Usage
    -----
    sr = SkillRegistry(organism, path="data/persona/skill_registry.json")

    # Seed a skill from research result:
    sr.seed("music_theory", description="harmonic analysis and chord relationships",
            source_memory_id="mem_123")

    # Queue practice after a completed exchange:
    sr.observe_interaction(user_input, response)

    # Query relevant skills before responding:
    skills = sr.relevant_skills(user_input, n=2)
    frag = "\n".join(s.prompt_fragment() for s in skills)

    # Decay idle skills (called from slow cycle):
    sr.decay_cycle()
    """

    def __init__(
        self,
        organism: Any,
        path: str = "data/persona/skill_registry.json",
    ):
        self._o    = organism
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = SkillState()
        self._observation_queue: queue.Queue = queue.Queue(maxsize=128)
        self._observation_worker_running = False
        self._recent_interactions: Dict[str, str] = {}
        self._semantic_vectors: Dict[str, Any] = {}
        self._semantic_generation = 0
        self._semantic_ready_generation = -1
        self._semantic_refresh_running = False
        self._enrichment_inflight = set()
        self._load()
        self._schedule_semantic_refresh()
        logger.info(
            f"[SkillRegistry] Initialised — "
            f"{len(self._state.skills)} skills, "
            f"{self._state.total_practices} practices"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def seed(
        self,
        domain:           str,
        description:      str  = "",
        source_memory_id: str  = "",
        initial_confidence: float = 0.25,
        name:             str = "",
        category:         str = "",
    ) -> Optional[Skill]:
        """Create or update a skill from a research/reflection result."""
        domain = self._canonical_domain(domain)
        if not domain:
            logger.debug("[SkillRegistry] Ignored empty or invalid skill domain")
            return None

        with self._lock:
            existing = self._get_skill(domain)
            if existing:
                # A repeated topic/source is exposure, not evidence of competence.
                if source_memory_id and source_memory_id not in existing.source_memories:
                    existing.source_memories = (
                        existing.source_memories + [source_memory_id]
                    )[-10:]
                if name and not existing.name:
                    existing.name = name[:80]
                if category and not existing.category:
                    existing.category = category[:40]
                self._put_skill(existing)
                result = existing
            else:
                skill = Skill(
                    domain           = domain,
                    name             = (name or domain.replace("_", " ").title())[:80],
                    category         = category[:40],
                    description      = description[:120],
                    confidence       = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEIL, initial_confidence)),
                    proficiency      = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEIL, initial_confidence)),
                    confidence_prior = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEIL, initial_confidence)),
                    source_memories  = [source_memory_id] if source_memory_id else [],
                )

                # Evict if at capacity
                if len(self._state.skills) >= MAX_SKILLS:
                    self._evict_weakest()

                self._put_skill(skill)
                result = skill
                logger.info(f"[SkillRegistry] Seeded: {domain} (conf={initial_confidence:.0%})")

            self._state.candidates.pop(domain, None)
            self._semantic_generation += 1

        # seed() previously only changed memory. A restart discarded every new
        # skill, which is why the logs repeatedly returned to the same 8 items.
        self._save()
        self._schedule_semantic_refresh()
        return result

    def propose(
        self,
        domain: str,
        description: str = "",
        source: str = "reflection",
        evidence_threshold: int = 3,
    ) -> Optional[Skill]:
        """Accumulate evidence before turning a reflection topic into a skill."""
        domain = self._canonical_domain(domain)
        if not domain:
            return None
        with self._lock:
            if domain in self._state.skills:
                existing = True
            else:
                existing = False
                candidate = dict(self._state.candidates.get(domain, {}))
                candidate["count"] = int(candidate.get("count", 0)) + 1
                candidate["description"] = str(description or candidate.get("description", ""))[:160]
                candidate["source"] = str(source or "reflection")[:40]
                candidate["last_seen"] = time.time()
                self._state.candidates[domain] = candidate
                if len(self._state.candidates) > 100:
                    oldest = min(
                        self._state.candidates.items(),
                        key=lambda item: item[1].get("last_seen", 0),
                    )[0]
                    self._state.candidates.pop(oldest, None)
                ready = candidate["count"] >= max(2, int(evidence_threshold))
        self._save()
        if existing:
            return self.seed(domain, description=description)
        if ready:
            return self.seed(domain, description=description)
        logger.debug(
            "[SkillRegistry] Candidate %s has %d/%d evidence events",
            domain, candidate["count"], max(2, int(evidence_threshold)),
        )
        return None

    def practice(
        self,
        domain:  str,
        success: Optional[bool] = None,
        context: str  = "",
        schedule_enrichment: bool = True,
        count_practice: bool = True,
    ) -> Optional[Skill]:
        """
        Record a practice event. Returns the skill if it deepened.
        Conversation callers should prefer observe_interaction(), which keeps
        persistence and enrichment off the interactive path.
        """
        domain = self._canonical_domain(domain)
        with self._lock:
            skill = self._get_skill(domain)
            if skill is None:
                return None

            if count_practice:
                skill.practice_count += 1
            skill.last_practiced   = time.time()
            if success is not None:
                if success:
                    skill.positive_outcomes += 1
                    skill.verified_practices += 1
                    skill.growth_trajectory = "improving"
                else:
                    skill.negative_outcomes += 1
                    skill.growth_trajectory = "declining"
                # Beta-binomial update centered on the confidence at first
                # seeding (four prior observations); only verified outcomes
                # move this estimate. Unrated exposures remain separate.
                evidence = skill.positive_outcomes + skill.negative_outcomes
                posterior = (
                    4.0 * skill.confidence_prior + skill.positive_outcomes
                ) / (4.0 + evidence)
                skill.confidence = max(
                    CONFIDENCE_FLOOR, min(CONFIDENCE_CEIL, posterior)
                )
                skill.proficiency = skill.confidence
            elif skill.growth_trajectory in ("emerging", "stable"):
                skill.growth_trajectory = "practiced"

            # Check for depth advancement
            deepened = False
            new_depth = self._compute_depth(skill.verified_practices)
            if new_depth > skill.depth:
                skill.depth = new_depth
                deepened    = True
                self._state.total_deepenings += 1
                logger.info(
                    f"[SkillRegistry] {domain} deepened to "
                    f"{skill.depth_label()} ({skill.verified_practices} verified outcomes)"
                )

            self._put_skill(skill)
            if count_practice:
                self._state.total_practices += 1
            self._semantic_generation += 1

        if deepened and schedule_enrichment:
            self._generate_depth_content(skill, context)

        self._save()
        return skill if deepened else None

    def observe_interaction(
        self,
        user_input: str,
        response: str,
        success: Optional[bool] = None,
    ) -> None:
        """Queue exposure learning; confidence changes require an outcome signal."""
        user_input = str(user_input or "")[:2000]
        full_response = str(response or "")
        response = full_response[:3000]
        item = (user_input, response, success, False)
        if full_response:
            key = self._response_key(full_response)
            with self._lock:
                self._recent_interactions.pop(key, None)
                self._recent_interactions[key] = user_input
                if len(self._recent_interactions) > 32:
                    self._recent_interactions.pop(next(iter(self._recent_interactions)))
        try:
            self._observation_queue.put_nowait(item)
        except queue.Full:
            try:
                self._observation_queue.get_nowait()
                self._observation_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._observation_queue.put_nowait(item)
            except queue.Full:
                return
        self._ensure_observation_worker()

    @staticmethod
    def _response_key(response: str) -> str:
        normalized = " ".join(str(response or "").casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def observe_feedback(self, response: str, positive: bool) -> bool:
        """Apply explicit feedback to the skill associated with that answer."""
        key = self._response_key(response)
        with self._lock:
            user_input = self._recent_interactions.get(key)
        if not user_input:
            return False
        item = (user_input, str(response or "")[:3000], bool(positive), True)
        try:
            self._observation_queue.put_nowait(item)
        except queue.Full:
            try:
                self._observation_queue.get_nowait()
                self._observation_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._observation_queue.put_nowait(item)
            except queue.Full:
                return False
        self._ensure_observation_worker()
        with self._lock:
            self._recent_interactions.pop(key, None)
        return True

    def observe_user_followup(self, text: str) -> bool:
        """Use an unambiguous next-turn correction or validation as delayed evidence."""
        normalized = " ".join(str(text or "").casefold().split())
        negative = (
            r"\b(that|this) (is|was) (wrong|incorrect|not right)\b",
            r"\bnot what i (asked|meant|wanted)\b",
            r"\byou (missed|misunderstood|got it wrong)\b",
            r"\b(c'est faux|c'est incorrect|tu as (rat[eé]|mal compris))\b",
            r"\b(ce n'est pas|ce n'est toujours pas) ce que je (demandais|voulais)\b",
            r"\b(ça ne répond pas|tu n'as pas compris)\b",
        )
        positive = (
            r"\b(that|this) (is|was) (correct|right|exactly right)\b",
            r"\b(that) worked\b|\bthat solved it\b",
            r"\bexactly what i needed\b|\byou got it right\b",
            r"\bc'est (exact|correct|bien ça)\b|\btu as compris\b",
            r"\b(ça marche|c'est exactement ça|probl[eè]me r[eé]solu)\b",
        )
        outcome = False if any(re.search(p, normalized) for p in negative) else (
            True if any(re.search(p, normalized) for p in positive) else None
        )
        if outcome is None:
            return False
        with self._lock:
            latest_key = next(reversed(self._recent_interactions), None)
            latest_user_input = self._recent_interactions.get(latest_key) if latest_key else None
        if not latest_user_input:
            return False
        try:
            self._observation_queue.put_nowait(
                (latest_user_input, str(text or "")[:3000], outcome, True)
            )
        except queue.Full:
            return False
        with self._lock:
            self._recent_interactions.pop(latest_key, None)
        self._ensure_observation_worker()
        return True

    def skills_snapshot(self) -> Dict[str, Skill]:
        """Return detached Skill objects for other cognitive subsystems."""
        with self._lock:
            return {
                domain: self._from_dict(domain, raw)
                for domain, raw in self._state.skills.items()
            }

    @property
    def _skills(self) -> Dict[str, Skill]:
        """Compatibility view for cognitive modules written against v46."""
        return self.skills_snapshot()

    def relevant_skills(
        self,
        query: str,
        n:     int = 2,
        semantic: bool = False,
    ) -> List[Skill]:
        """
        Return the n most relevant skills for a given query string.
        Uses keyword overlap (or embeddings if available).
        """
        query_words = self._terms(query)
        scores: Dict[str, Tuple[float, Skill]] = {}
        with self._lock:
            for domain, raw in self._state.skills.items():
                skill = self._from_dict(domain, raw)
                if skill.confidence < 0.20:
                    continue

                skill_text = " ".join([
                    domain.replace("_", " "), skill.name, skill.category,
                    skill.description, " ".join(skill.sub_skills),
                    " ".join(skill.insights), " ".join(skill.aliases),
                ])
                all_skill_words = self._terms(skill_text)
                overlap = len(query_words & all_skill_words)
                if overlap > 0:
                    coverage = overlap / max(1, min(len(query_words), 8))
                    score = (overlap + coverage) * skill.confidence * (1 + skill.depth * 0.3)
                    scores[domain] = (score, skill)

                # Transfer link boost — if a linked skill matches, this skill is relevant too
                for linked_domain in skill.transfer_links:
                    linked_words = self._terms(linked_domain.replace("_", " "))
                    if query_words & linked_words:
                        transfer_score = 0.35 * skill.confidence
                        if transfer_score > scores.get(domain, (0.0, skill))[0]:
                            scores[domain] = (transfer_score, skill)
                        break

        if semantic:
            self._add_semantic_scores(query, scores)

        ranked = sorted(scores.values(), key=lambda x: -x[0])
        return [skill for _, skill in ranked[:max(0, n)]]

    def decay_cycle(self) -> int:
        """
        Apply confidence decay to rusty skills.
        Returns count of skills decayed.
        """
        count = 0
        with self._lock:
            for domain, raw in list(self._state.skills.items()):
                skill = self._from_dict(domain, raw)
                if skill.is_rusty():
                    old = skill.confidence
                    skill.confidence = max(CONFIDENCE_FLOOR, skill.confidence - DECAY_RATE)
                    if skill.confidence < old:
                        self._put_skill(skill)
                        count += 1
        if count:
            self._save()
        return count

    def prompt_fragment(self, query: str) -> str:
        """Return skill context for prompt injection, or empty string."""
        skills = self.relevant_skills(query, n=2)
        if not skills:
            return ""
        frags = [s.prompt_fragment() for s in skills if s.depth >= 1]
        return "\n".join(frags) if frags else ""

    def all_domain_confidence(self) -> Dict[str, float]:
        """
        Phase 5.2 — domain -> confidence map for every tracked skill,
        including low-confidence/nascent ones (unlike relevant_skills(),
        which filters below 0.20 since it's about picking what to surface
        in a prompt fragment, not about knowing where the blind spots are).
        Used by workspace_state_sync.py to give executive_arbitration's
        calibration term real per-domain data instead of one global
        workspace-level scalar.
        """
        with self._lock:
            return {
                domain: float(raw.get("confidence", 0.25))
                for domain, raw in self._state.skills.items()
            }

    def summary(self) -> Dict:
        with self._lock:
            skills = [self._from_dict(domain, raw) for domain, raw in self._state.skills.items()]
            return {
                "total_skills":    len(self._state.skills),
                "candidate_skills": len(self._state.candidates),
                "total_practices": self._state.total_practices,
                "verified_outcomes": sum(s.positive_outcomes + s.negative_outcomes for s in skills),
                "positive_outcomes": sum(s.positive_outcomes for s in skills),
                "negative_outcomes": sum(s.negative_outcomes for s in skills),
                "skills_with_verified_outcomes": sum(
                    1 for s in skills if s.positive_outcomes + s.negative_outcomes > 0
                ),
                "deep_skills":     sum(
                    1 for r in self._state.skills.values()
                    if r.get("depth", 0) >= 2
                ),
            }

    def maintenance_cycle(self, slow_cycle: int = 0) -> Dict[str, int]:
        """Run bounded background maintenance; never performs an inline LLM call."""
        decayed = self.decay_cycle()
        self._schedule_semantic_refresh()
        pending = []
        with self._lock:
            for domain, raw in self._state.skills.items():
                skill = self._from_dict(domain, raw)
                if skill.depth > skill.enriched_depth:
                    pending.append(skill)
        if pending:
            pending.sort(key=lambda s: (s.enriched_depth, -s.depth, -s.confidence))
            self._generate_depth_content(pending[0], "background maintenance")
        return {"decayed": decayed, "pending_enrichment": len(pending)}

    # ── Depth content generation (LLM) ───────────────────────────────────────

    def _generate_depth_content(self, skill: Skill, context: str = "") -> None:
        """
        When a skill deepens, generate new sub_skills and insights via LLM.
        Runs in background — non-blocking.
        """
        with self._lock:
            if skill.domain in self._enrichment_inflight:
                return
            self._enrichment_inflight.add(skill.domain)

        def _run() -> None:
            try:
                self.__depth_llm_call(skill, context)
            finally:
                with self._lock:
                    self._enrichment_inflight.discard(skill.domain)

        threading.Thread(target=_run, daemon=True, name="lumina-skill-enrichment").start()

    def __depth_llm_call(self, skill: Skill, context: str) -> None:
        try:
            from core.llm_scheduler import llm_scheduler
            if not self._background_idle():
                return
            ai_system = getattr(self._o, "ai_system", None)
            llm = getattr(ai_system, "llm", None) or ai_system
            if llm is None or not hasattr(llm, "get_response"):
                return

            prompt = (
                f"Lumina has deepened her competency in: {skill.domain}\n"
                f"Description: {skill.description}\n"
                f"Current depth: {skill.depth_label()} "
                f"({skill.verified_practices} verified outcomes; "
                f"{skill.practice_count} exposures)\n"
                f"Existing insights: {'; '.join(skill.insights[:3]) or 'none yet'}\n"
                f"Context of recent use: {context[:120] or 'general engagement'}\n\n"
                f"Generate 2 new specific sub-skills, 1-2 new insights, and "
                f"up to 6 concise English/French retrieval keywords that emerge "
                f"at this depth level.\n"
                f"Respond ONLY with JSON:\n"
                f'{{"sub_skills": ["sub1","sub2"], '
                f'"insights": ["insight1","insight2"], '
                f'"keywords": ["keyword1","mot-cle1"]}}'
            )

            result = ""
            with llm_scheduler.sync_slot(
                priority    = 3,
                skip_if_busy= True,
                caller      = "skill_registry.deepen",
            ) as acquired:
                if acquired:
                    result = llm.get_response(
                        messages   = [{"role": "user", "content": prompt}],
                        max_tokens = 220,
                        temperature= 0.6,
                    ) or ""

            if result:
                import re, json as _json
                clean = re.sub(r"```json|```", "", result).strip()
                payload = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                data = _json.loads(payload.group(0) if payload else clean)
                with self._lock:
                    s = self._get_skill(skill.domain)
                    if s:
                        new_subs = data.get("sub_skills", [])[:2]
                        s.sub_skills = list(dict.fromkeys(
                            s.sub_skills + new_subs
                        ))[:8]
                        new_insights = data.get("insights", [])[:2]
                        s.insights = (s.insights + new_insights)[-MAX_INSIGHTS:]
                        new_aliases = [
                            str(value).strip() for value in data.get("keywords", [])[:6]
                            if str(value).strip()
                        ]
                        s.aliases = list(dict.fromkeys(s.aliases + new_aliases))[-MAX_ALIASES:]
                        s.enriched_depth = max(s.enriched_depth, skill.depth)
                        self._put_skill(s)
                        self._semantic_generation += 1
                self._save()
                self._schedule_semantic_refresh()
                logger.info(f"[SkillRegistry] {skill.domain} depth content generated")
        except Exception as e:
            logger.debug(f"[SkillRegistry] depth LLM failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_depth(verified_count: int) -> int:
        depth = 0
        for i, threshold in enumerate(DEPTH_THRESHOLDS):
            if verified_count >= threshold:
                depth = i
        return min(depth, 3)

    def _get_skill(self, domain: str) -> Optional[Skill]:
        raw = self._state.skills.get(domain)
        return self._from_dict(domain, raw) if raw else None

    def _put_skill(self, skill: Skill) -> None:
        self._state.skills[skill.domain] = asdict(skill)

    @staticmethod
    def _from_dict(domain: str, raw: Dict) -> Skill:
        practice_count = int(raw.get("practice_count", raw.get("instances", 0)) or 0)
        legacy_category = raw.get("domain", "") if raw.get("domain") != domain else ""
        raw_depth = raw.get("depth")
        if raw_depth is None:
            raw_depth = SkillRegistry._compute_depth(practice_count)
        confidence = float(raw.get("confidence", raw.get("proficiency", 0.25)))
        positive_outcomes = int(raw.get("positive_outcomes", 0) or 0)
        negative_outcomes = int(raw.get("negative_outcomes", 0) or 0)
        confidence_prior = raw.get("confidence_prior")
        if confidence_prior is None:
            # Back-solve the prior for existing records so the first Bayesian
            # update preserves their stored score instead of recounting history.
            evidence = positive_outcomes + negative_outcomes
            confidence_prior = (
                (confidence * (4.0 + evidence) - positive_outcomes) / 4.0
                if evidence else confidence
            )
        confidence_prior = max(
            CONFIDENCE_FLOOR, min(CONFIDENCE_CEIL, float(confidence_prior))
        )
        return Skill(
            domain          = domain,
            name            = raw.get("name", domain.replace("_", " ").title()),
            category        = raw.get("category", legacy_category),
            description     = raw.get("description",     ""),
            confidence      = confidence,
            proficiency     = raw.get("proficiency", raw.get("confidence", 0.25)),
            confidence_prior = confidence_prior,
            growth_trajectory = raw.get("growth_trajectory") or (
                "stable" if practice_count else "emerging"
            ),
            depth           = max(0, min(3, int(raw_depth or 0))),
            last_practiced  = raw.get("last_practiced", raw.get("last_demonstrated", time.time())),
            practice_count  = practice_count,
            verified_practices = int(raw.get("verified_practices", 0) or 0),
            positive_outcomes = positive_outcomes,
            negative_outcomes = negative_outcomes,
            sub_skills      = raw.get("sub_skills",      []),
            source_memories = raw.get("source_memories", []),
            insights        = raw.get("insights",        []),
            transfer_links  = raw.get("transfer_links",  []),
            aliases         = raw.get("aliases",         []),
            enriched_depth  = int(raw.get("enriched_depth", 0) or 0),
        )

    def generate_transfer_links(self) -> None:
        """
        For all skills at depth ≥ 1, run a background LLM call to discover
        which other domains they could transfer competence to.
        Called from slow cycle every ~60 cycles (~ 2 hours).
        Identifies structural similarities across domains rather than
        relying on surface-level keyword matching.
        """
        import threading as _th
        _th.Thread(
            target  = self.__transfer_link_cycle,
            daemon  = True,
        ).start()

    def __transfer_link_cycle(self) -> None:
        try:
            from core.llm_scheduler import llm_scheduler
            if not self._background_idle():
                return

            with self._lock:
                candidates = [
                    (d, r) for d, r in self._state.skills.items()
                    if r.get("depth", 0) >= 1 and r.get("confidence", 0) >= 0.45
                    and len(r.get("transfer_links", [])) < 3
                ]

            if not candidates:
                return

            # Process one skill per cycle to keep LLM budget low
            domain, raw = candidates[0]
            skill = self._from_dict(domain, raw)

            other_domains = [
                d for d in self._state.skills.keys()
                if d != domain
            ][:8]

            if not other_domains:
                return

            prompt = (
                f"Skill domain: {skill.domain}\n"
                f"Description: {skill.description}\n"
                f"Sub-skills: {', '.join(skill.sub_skills[:4])}\n"
                f"Insights: {'; '.join(skill.insights[:2])}\n\n"
                f"Other known domains: {', '.join(other_domains)}\n\n"
                f"Which of these other domains could meaningfully benefit from "
                f"competence in '{skill.domain}'? Consider structural similarities, "
                f"shared cognitive patterns, and transferable mental models.\n"
                f"Respond ONLY with JSON: "
                f'{{\"transfer_to\": [\"domain1\", \"domain2\"]}}'
            )

            result = ""
            with llm_scheduler.sync_slot(
                priority    = 3,
                skip_if_busy= True,
                caller      = "skill_registry.transfer",
            ) as acquired:
                if not acquired:
                    return
                ai_system = getattr(self._o, "ai_system", None)
                llm = getattr(ai_system, "llm", None) or ai_system
                if llm and hasattr(llm, "get_response"):
                    result = llm.get_response(
                        messages   = [{"role": "user", "content": prompt}],
                        max_tokens = 80,
                        temperature= 0.5,
                    ) or ""

            if result:
                import re, json as _j
                clean = re.sub(r"```json|```", "", result).strip()
                data  = _j.loads(clean)
                links = [
                    d for d in data.get("transfer_to", [])
                    if d in self._state.skills and d != domain
                ][:3]

                if links:
                    with self._lock:
                        s = self._get_skill(domain)
                        if s:
                            s.transfer_links = list(dict.fromkeys(
                                s.transfer_links + links
                            ))[:4]
                            self._put_skill(s)
                    self._save()
                    logger.info(
                        f"[SkillRegistry] Transfer links: {domain} → {links}"
                    )
        except Exception as e:
            logger.debug(f"[SkillRegistry] transfer_link_cycle failed: {e}")

    def _evict_weakest(self) -> None:
        """Remove the skill with lowest confidence × depth product."""
        if not self._state.skills:
            return
        worst = min(
            self._state.skills.items(),
            key=lambda kv: kv[1].get("confidence", 0) * (1 + kv[1].get("depth", 0))
        )
        del self._state.skills[worst[0]]

    # ── Background observation and semantic retrieval ───────────────────────

    def _ensure_observation_worker(self) -> None:
        with self._lock:
            if self._observation_worker_running:
                return
            self._observation_worker_running = True

        threading.Thread(
            target=self._observation_worker,
            daemon=True,
            name="lumina-skill-observer",
        ).start()

    def _observation_worker(self) -> None:
        try:
            while True:
                # Completed turns may arrive while the user immediately starts
                # speaking again. Keep embedding and persistence work out of
                # that window; queued observations can safely wait for idle.
                while not self._background_idle():
                    time.sleep(0.25)
                try:
                    user_input, response, success, feedback_only = self._observation_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._process_observation(user_input, response, success, feedback_only)
                except Exception as exc:
                    logger.debug("[SkillRegistry] observation failed: %s", exc)
                finally:
                    self._observation_queue.task_done()
        finally:
            with self._lock:
                self._observation_worker_running = False
                restart = not self._observation_queue.empty()
            if restart:
                self._ensure_observation_worker()

    def _process_observation(
        self, user_input: str, response: str,
        success: Optional[bool], feedback_only: bool = False,
    ) -> None:
        # The model's completed answer is included only for semantic matching;
        # learned aliases always come from the user's wording so Lumina can
        # recognise that person's vocabulary on a later turn.
        combined = f"{user_input}\n{response[:1200]}"
        matches = self.relevant_skills(
            combined, n=1 if success is not None else 2, semantic=True
        )
        if not matches:
            return

        aliases = [] if feedback_only else self._salient_aliases(user_input)
        deepened: List[Skill] = []
        for matched in matches:
            with self._lock:
                current = self._get_skill(matched.domain)
                if current and aliases:
                    current.aliases = list(dict.fromkeys(
                        current.aliases + aliases
                    ))[-MAX_ALIASES:]
                    self._put_skill(current)
                    self._semantic_generation += 1
            changed = self.practice(
                matched.domain,
                success=success,
                context=user_input[:160],
                schedule_enrichment=False,
                count_practice=not feedback_only,
            )
            if changed:
                deepened.append(changed)

        self._save()
        self._schedule_semantic_refresh()
        for skill in deepened:
            self._generate_depth_content(skill, user_input[:160])

    def _add_semantic_scores(
        self,
        query: str,
        scores: Dict[str, Tuple[float, Skill]],
    ) -> None:
        if not self._background_idle():
            return
        with self._lock:
            vectors = dict(self._semantic_vectors)
            ready = self._semantic_ready_generation == self._semantic_generation
        if not vectors or not ready:
            self._schedule_semantic_refresh()
            return

        model = self._embedding_model()
        if model is None:
            return
        try:
            import numpy as np
            query_vec = np.asarray(model.encode(query), dtype="float32").reshape(-1)
            norm = float(np.linalg.norm(query_vec))
            if norm <= 0:
                return
            query_vec /= norm
            for domain, skill_vec in vectors.items():
                similarity = float(np.dot(query_vec, skill_vec))
                if similarity < SEMANTIC_MATCH_FLOOR:
                    continue
                with self._lock:
                    raw = self._state.skills.get(domain)
                    skill = self._from_dict(domain, raw) if raw else None
                if skill is None or skill.confidence < 0.20:
                    continue
                semantic_score = similarity * skill.confidence * (1 + skill.depth * 0.2)
                if semantic_score > scores.get(domain, (0.0, skill))[0]:
                    scores[domain] = (semantic_score, skill)
        except Exception as exc:
            logger.debug("[SkillRegistry] semantic query failed: %s", exc)

    def _schedule_semantic_refresh(self) -> None:
        with self._lock:
            if (
                self._semantic_refresh_running
                or self._semantic_ready_generation == self._semantic_generation
            ):
                return
            self._semantic_refresh_running = True

        def _run() -> None:
            try:
                self._refresh_semantic_index()
            finally:
                with self._lock:
                    self._semantic_refresh_running = False

        threading.Thread(
            target=_run,
            daemon=True,
            name="lumina-skill-index",
        ).start()

    def _refresh_semantic_index(self) -> None:
        model = self._embedding_model()
        if model is None:
            with self._lock:
                self._semantic_vectors = {}
                self._semantic_ready_generation = self._semantic_generation
            return
        with self._lock:
            generation = self._semantic_generation
            items = [
                (domain, self._skill_search_text(self._from_dict(domain, raw)))
                for domain, raw in self._state.skills.items()
            ]
        if not items:
            with self._lock:
                self._semantic_vectors = {}
                self._semantic_ready_generation = generation
            return
        try:
            import numpy as np
            matrix = np.asarray(model.encode([text for _, text in items]), dtype="float32")
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
            vectors = {domain: matrix[index] for index, (domain, _) in enumerate(items)}
            with self._lock:
                if generation == self._semantic_generation:
                    self._semantic_vectors = vectors
                    self._semantic_ready_generation = generation
        except Exception as exc:
            logger.debug("[SkillRegistry] semantic index unavailable: %s", exc)

    def _embedding_model(self):
        ai_system = getattr(self._o, "ai_system", None)
        model = getattr(ai_system, "embedding_model", None)
        if model is None:
            model = getattr(getattr(ai_system, "memory_system", None), "embedding_model", None)
        return model

    def _background_idle(self) -> bool:
        last_interaction = float(getattr(self._o, "_last_interaction_ts", 0.0) or 0.0)
        if (time.time() - last_interaction) < BACKGROUND_IDLE_SECONDS:
            return False

        # ExternalLLMAdapter keeps the bound LLMManager method in _fn. Honour
        # its interactive marker as well as the time-based quiet period.
        ai_system = getattr(self._o, "ai_system", None)
        llm = getattr(ai_system, "llm", None)
        manager = getattr(getattr(llm, "_fn", None), "__self__", None)
        chat_active = getattr(manager, "_chat_active", None)
        return not (
            chat_active is not None
            and hasattr(chat_active, "is_set")
            and chat_active.is_set()
        )

    @staticmethod
    def _canonical_domain(domain: str) -> str:
        text = unicodedata.normalize("NFKD", str(domain or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
        text = re.sub(r"^new schema opened:\s*", "", text)
        text = text.split(". seed:", 1)[0]
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return text[:60]

    @staticmethod
    def _terms(text: str) -> set:
        normal = unicodedata.normalize("NFKD", str(text or ""))
        normal = "".join(ch for ch in normal if not unicodedata.combining(ch)).lower()
        return {
            word for word in re.findall(r"[a-z0-9]{3,}", normal)
            if word not in _MATCH_STOPWORDS
        }

    @classmethod
    def _salient_aliases(cls, text: str) -> List[str]:
        seen = set()
        aliases = []
        normal = unicodedata.normalize("NFKD", str(text or ""))
        normal = "".join(ch for ch in normal if not unicodedata.combining(ch)).lower()
        for word in re.findall(r"[a-z0-9]{4,}", normal):
            if word in _MATCH_STOPWORDS or word in seen:
                continue
            seen.add(word)
            aliases.append(word)
            if len(aliases) >= 6:
                break
        return aliases

    @staticmethod
    def _skill_search_text(skill: Skill) -> str:
        return " ".join([
            skill.domain.replace("_", " "), skill.name, skill.category,
            skill.description, " ".join(skill.sub_skills),
            " ".join(skill.insights), " ".join(skill.aliases),
        ]).strip()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                raw_skills = data.get("skills", {})
                migrated_skills = {}
                for domain, raw in raw_skills.items():
                    canonical = self._canonical_domain(domain)
                    if canonical:
                        migrated_skills[canonical] = asdict(
                            self._from_dict(canonical, raw)
                        )
                inferred_practices = sum(
                    raw.get("practice_count", raw.get("instances", 0)) or 0
                    for raw in raw_skills.values()
                )
                inferred_deepenings = sum(
                    self._from_dict(domain, raw).depth
                    for domain, raw in migrated_skills.items()
                )
                self._state = SkillState(
                    skills            = migrated_skills,
                    candidates        = data.get("candidates", {}),
                    total_practices   = data.get("total_practices", inferred_practices),
                    total_deepenings  = data.get("total_deepenings", inferred_deepenings),
                )
                self._semantic_generation += 1
                if migrated_skills != raw_skills or "total_practices" not in data:
                    self._save()
        except Exception as e:
            logger.warning(f"[SkillRegistry] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os; os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[SkillRegistry] Save failed: {e}")
