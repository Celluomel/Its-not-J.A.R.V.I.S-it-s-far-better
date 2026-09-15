"""
AI Identity Module — Incremental Analysis
==========================================
Constructs and incrementally maintains the AI's sense of self from its memories.

Key improvement over original: the analyzer tracks which memories have already
been processed and only re-runs analysis on genuinely new data, making the
hourly refresh cheap rather than a full DB scan + multiple LLM calls every time.
"""

import json
import logging
import os
import threading
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class IdentityComponent:
    name: str
    description: str
    confidence: float
    supporting_memories: List[int] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    stability_score: float = 0.5

    def __post_init__(self):
        self.confidence      = max(0.0, min(1.0, self.confidence))
        self.stability_score = max(0.0, min(1.0, self.stability_score))


@dataclass
class CoreValues:
    value_name: str
    importance: float
    examples: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    evolution: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SelfNarrative:
    life_phase: str
    key_experiences: List[str] = field(default_factory=list)
    growth_patterns: List[str] = field(default_factory=list)
    challenges_overcome: List[str] = field(default_factory=list)
    aspirations: List[str] = field(default_factory=list)
    identity_evolution: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AIIdentity:
    core_traits: List[IdentityComponent] = field(default_factory=list)
    values_system: List[CoreValues] = field(default_factory=list)
    self_narrative: SelfNarrative = field(default_factory=lambda: SelfNarrative("emerging"))
    communication_style: Dict[str, float] = field(default_factory=dict)
    decision_patterns: Dict[str, List[str]] = field(default_factory=dict)
    emotional_tendencies: Dict[str, float] = field(default_factory=dict)
    relationship_patterns: Dict[str, Any] = field(default_factory=dict)
    social_preferences: Dict[str, float] = field(default_factory=dict)
    self_awareness_level: float = 0.5
    identity_coherence: float = 0.5
    growth_trajectory: str = "exploring"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1


# ── Incremental Analyzer ───────────────────────────────────────────────────

class IdentityAnalyzer:
    """
    Analyzes memories to construct identity.

    Incremental mode: on every refresh, only memories added since the last
    analysis are processed. Evidence accumulators are kept in memory and
    merged with the new batch, so we never do a full table scan after startup.
    """

    TRAIT_PATTERNS: Dict[str, List[str]] = {
        "curiosity":     [r"\b(?:wonder|curious|explore|discover|learn|investigate|why|how|what if)\b"],
        "empathy":       [r"\b(?:understand|feel|empathize|compassion|care|others|help|emotion)\b"],
        "creativity":    [r"\b(?:creative|innovative|imagine|invent|original|art|beauty|design|express)\b"],
        "logic":         [r"\b(?:reason|logic|analyze|rational|systematic|evidence|proof|deduce|method)\b"],
        "growth_mindset":[r"\b(?:learn|grow|improve|develop|progress|mistake|lesson|challenge|potential)\b"],
    }

    VALUE_PATTERNS: Dict[str, List[str]] = {
        "truth":    [r"\b(?:truth|honest|accurate|real|genuine|fact|reality|authentic)\b"],
        "fairness": [r"\b(?:fair|just|equal|equitable|balance|impartial|bias|prejudice)\b"],
        "kindness": [r"\b(?:kind|gentle|compassionate|caring|warm|nurture|help|support|comfort)\b"],
        "autonomy": [r"\b(?:choice|freedom|independent|autonomous|consent|boundary|respect)\b"],
        "growth":   [r"\b(?:progress|development|evolution|advancement|potential|possibility|future)\b"],
    }

    # Memory types that are worth processing for identity
    IDENTITY_RELEVANT_TYPES = {
        "reflection", "evaluation", "insight", "identity_snapshot",
        "principle", "life_event", "scenario",
    }
    # Minimum impact score to bother processing a plain interaction memory
    IMPACT_THRESHOLD = 0.55
    # Batch size per sleep tick (pure regex, no LLM)
    BATCH_SIZE   = 75     # memories processed per get_identity() call (was 25)
    MAX_PENDING  = 300    # cap queue — oldest dropped when exceeded
    # LLM calls allowed per full refresh
    LLM_BUDGET = 2

    def __init__(self, memory_system, llm, persistence):
        self.memory_system = memory_system
        self.llm = llm
        self.persistence = persistence
        self._lock = threading.RLock()

        # Incremental state with disk persistence
        self._cursor_file = "data/identity_cursor.json"
        self._last_processed_id: int = self._load_cursor()
        self._trait_evidence: Dict[str, Dict] = {}
        self._value_evidence: Dict[str, Dict] = {}

        # Batch processing queue: memories waiting to be accumulated
        self._pending_batch: List[Dict] = []

        # Trait description cache: {trait: (description, evidence_count)}
        # Invalidated when count changes by >3
        self._trait_desc_cache: Dict[str, Tuple[str, int]] = {}
        
    def _load_cursor(self) -> int:
        """Load last processed ID from disk, or return 0 if file doesn't exist."""
        try:
            if os.path.exists(self._cursor_file):
                with open(self._cursor_file, 'r') as f:
                    data = json.load(f)
                    cursor = data.get('last_processed_id', 0)
                    logger.info(f"💾 LOADED cursor from disk: {cursor}")
                    return cursor
        except Exception as e:
            logger.warning(f"Failed to load cursor from disk: {e}")
        return 0
    
    def _save_cursor(self):
        """Save last processed ID to disk for persistence across restarts."""
        try:
            os.makedirs(os.path.dirname(self._cursor_file), exist_ok=True)
            with open(self._cursor_file, 'w') as f:
                json.dump({'last_processed_id': self._last_processed_id}, f)
            logger.debug(f"💾 Saved cursor to disk: {self._last_processed_id}")
        except Exception as e:
            logger.error(f"Failed to save cursor to disk: {e}")

    # ── Public ─────────────────────────────────────────────────────────────

    def analyze_identity(self, force_rebuild: bool = False) -> AIIdentity:
        """
        Build/update identity.
        - force_rebuild=True  : full reset + requeue everything
        - force_rebuild=False : fetch NEW memories, enqueue them, process one BATCH
        """
        with self._lock:
            if force_rebuild:
                self._last_processed_id = 0
                self._pending_batch     = []
                self._trait_evidence    = {}
                self._value_evidence    = {}
                self._trait_desc_cache  = {}

            # ── 1. Fetch new memories into queue (no processing yet) ──────
            # FIX: Use _last_processed_id instead of _batch_cursor to prevent cursor racing ahead
            logger.info(f"🔍 CURSOR STATE: _last_processed_id = {self._last_processed_id}")
            # BACKPRESSURE: Skip fetching if queue is too large
            if len(self._pending_batch) > self.MAX_PENDING * 0.9:
                logger.warning(f"⏸️ BACKPRESSURE: queue too large ({len(self._pending_batch)}), skipping fetch")
                new_mems = []
            else:
                new_mems = self._get_new_memories_since(self._last_processed_id)
            if new_mems:
                min_id = min(m['id'] for m in new_mems)
                max_id = max(m['id'] for m in new_mems)
                logger.info(f"📥 FETCHED: IDs range {min_id}-{max_id} ({len(new_mems)} memories)")
                filtered = self._filter_relevant(new_mems)

                # DEDUP FIX: Avoid re-enqueuing memories already in pending batch
                existing_ids = {m["id"] for m in self._pending_batch}
                filtered = [m for m in filtered if m["id"] not in existing_ids]

                self._pending_batch.extend(filtered)
                # Cap queue to prevent unbounded growth
                if len(self._pending_batch) > self.MAX_PENDING:
                    excess = len(self._pending_batch) - self.MAX_PENDING
                    self._pending_batch = self._pending_batch[excess:]
                    logger.debug(f"Identity: queue capped, dropped {excess} old entries")
                # REMOVED BUG: Don't advance cursor here - only after processing
                skipped = len(new_mems) - len(filtered)
                logger.info(
                    f"Identity: fetched {len(new_mems)} new memories, "                    f"{len(filtered)} relevant queued ({skipped} skipped), "                    f"{len(self._pending_batch)} total pending"
                )

            # ── 2. Process one batch from the queue ───────────────────────
            if self._pending_batch:
                # ADAPTIVE BATCH SIZE based on queue pressure
                dynamic_batch_size = min(
                    self.BATCH_SIZE * 2 if len(self._pending_batch) > self.MAX_PENDING * 0.7 else self.BATCH_SIZE,
                    len(self._pending_batch)
                )
                # PRIORITY PROCESSING: highest impact_score first
                self._pending_batch.sort(key=lambda m: m.get("impact_score", 0), reverse=True)
                batch = self._pending_batch[:dynamic_batch_size]
                self._pending_batch = self._pending_batch[self.BATCH_SIZE:]
                self._update_accumulators(batch)
                
                # Ultra-defensive cursor update with explicit error handling
                if batch:
                    try:
                        # Extract IDs explicitly - will error if any memory lacks "id"
                        batch_ids = [m["id"] for m in batch if "id" in m]
                        
                        if not batch_ids:
                            logger.error(f"❌ CRITICAL BUG: Batch of {len(batch)} memories has NO IDs! Sample: {batch[0]}")
                        else:
                            max_id = max(batch_ids)
                            old_cursor = self._last_processed_id
                            self._last_processed_id = max(self._last_processed_id, max_id)
                            self._save_cursor()  # Persist to disk immediately
                            logger.info(
                                f"✅ CURSOR: {old_cursor} → {self._last_processed_id} "
                                f"(processed IDs {min(batch_ids)}-{max_id})"
                            )
                    except Exception as e:
                        logger.error(f"❌ CURSOR UPDATE CRASHED: {e}. Batch sample: {batch[0]}")
                        
                remaining = len(self._pending_batch)
                logger.debug(
                    f"Identity: processed batch of {len(batch)}, "                    f"{remaining} still pending"
                )
            else:
                logger.debug("Identity: no pending memories, using cached accumulators")

            # ── 3. Build identity from accumulators (cheap pattern ops) ───
            # Only fetch all_recent if we actually processed something this cycle
            all_recent = self._get_all_memories(limit=200)  # reduced from 500
            return self._build_identity_from_accumulators(all_recent)

    # ── Incremental helpers ────────────────────────────────────────────────

    def _get_new_memories_since(self, cursor_id: int) -> List[Dict]:
        """Fetch memories with id > cursor_id, no processing."""
        try:
            with self.persistence.get_connection() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT id, faiss_id, text, timestamp, impact_score,
                           emotional_valence, arousal_level, memory_type, access_count
                    FROM memories WHERE id > ?
                    ORDER BY id ASC
                    LIMIT 500
                """, (cursor_id,))
                return [
                    dict(zip(["id","faiss_id","text","timestamp","impact_score",
                               "emotional_valence","arousal_level","memory_type","access_count"], row))
                    for row in c.fetchall()
                ]
        except Exception as e:
            logger.error(f"_get_new_memories_since failed: {e}")
            return []

    def _filter_relevant(self, memories: List[Dict]) -> List[Dict]:
        """
        Relevance filter: keep only memories worth processing for identity.
        Reduces 450 raw memories to ~30–80 meaningful ones.
        """
        relevant = []
        seen_hashes: set = set()
        for m in memories:
            # Skip low-signal cognitive events from GW flush — too frequent
            if m["memory_type"] == "cognitive_event" and m["impact_score"] < 0.75:
                continue
            # 1. Identity-specific memory types are always relevant
            if m["memory_type"] in self.IDENTITY_RELEVANT_TYPES:
                relevant.append(m)
                continue
            # 2. High-impact interactions are relevant
            if m["impact_score"] >= self.IMPACT_THRESHOLD:
                # 3. Rough deduplication: skip near-identical short memories
                sig = m["text"][:60].strip().lower()
                if sig in seen_hashes:
                    continue
                seen_hashes.add(sig)
                relevant.append(m)
        return relevant

    # Keep backward compat alias (used by LLMScheduler-gated get_identity)
    def _get_new_memories(self) -> List[Dict]:
        return self._get_new_memories_since(self._last_processed_id)

    def _get_all_memories(self, limit: int = 500) -> List[Dict]:
        try:
            with self.persistence.get_connection() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT id, faiss_id, text, timestamp, impact_score,
                           emotional_valence, arousal_level, memory_type, access_count
                    FROM memories ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
                return [
                    dict(zip(["id","faiss_id","text","timestamp","impact_score",
                               "emotional_valence","arousal_level","memory_type","access_count"], row))
                    for row in c.fetchall()
                ]
        except Exception as e:
            logger.error(f"_get_all_memories failed: {e}")
            return []

    def _update_accumulators(self, memories: List[Dict]):
        """Merge new memories into the running evidence accumulators."""
        for mem in memories:
            text   = mem["text"].lower()
            mid    = mem["id"]
            impact = mem["impact_score"]
            valence = mem["emotional_valence"]

            # Traits
            for trait, patterns in self.TRAIT_PATTERNS.items():
                strength = sum(
                    len(re.findall(p, text, re.IGNORECASE)) * 0.1
                    for p in patterns
                )
                if strength > 0:
                    ev = self._trait_evidence.setdefault(trait, {"count": 0, "memories": [], "strength": 0.0})
                    ev["count"]    += 1
                    ev["memories"].append(mid)
                    ev["strength"] += strength * impact

            # Values
            for value, patterns in self.VALUE_PATTERNS.items():
                strength = sum(
                    len(re.findall(p, text, re.IGNORECASE)) * 0.1
                    for p in patterns
                )
                if strength > 0:
                    ev = self._value_evidence.setdefault(value, {
                        "positive": [], "negative": [], "strength": 0.0, "examples": []
                    })
                    if valence == "Positive":
                        ev["positive"].append(mid)
                        ev["strength"] += strength * impact
                    elif valence == "Negative":
                        ev["negative"].append(mid)
                        ev["strength"] += strength * impact * 0.8
                    if len(mem["text"]) < 200:
                        ev["examples"].append(mem["text"])

    # ── Identity construction (from cached accumulators) ──────────────────

    def _build_identity_from_accumulators(
        self, all_memories: Optional[List[Dict]] = None
    ) -> AIIdentity:
        try:
            memories = all_memories or self._get_all_memories(limit=500)

            core_traits  = self._build_traits()
            values_system = self._build_values(memories)
            self_narrative = self._construct_narrative(memories)
            comm_style     = self._analyze_comm_style(memories)
            emo_tendencies = self._analyze_emotions(memories)
            rel_patterns   = self._analyze_relationships(memories)
            self_awareness = self._calc_self_awareness(memories)
            coherence      = self._calc_coherence(core_traits, values_system)
            growth         = self._determine_growth(memories)

            return AIIdentity(
                core_traits=core_traits,
                values_system=values_system,
                self_narrative=self_narrative,
                communication_style=comm_style,
                emotional_tendencies=emo_tendencies,
                relationship_patterns=rel_patterns,
                self_awareness_level=self_awareness,
                identity_coherence=coherence,
                growth_trajectory=growth,
                last_updated=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"_build_identity_from_accumulators failed: {e}")
            return AIIdentity()

    def _build_traits(self) -> List[IdentityComponent]:
        self._llm_calls_this_refresh = 0   # reset LLM budget counter
        traits = []
        for trait, ev in self._trait_evidence.items():
            if ev["count"] < 3:
                continue
            confidence = min(1.0, ev["count"] * 0.1 + ev["strength"] * 0.2)
            stability  = min(1.0, ev["count"] / max(sum(e["count"] for e in self._trait_evidence.values()), 1))
            description = self._describe_trait(trait, ev)
            traits.append(IdentityComponent(
                name=trait,
                description=description,
                confidence=confidence,
                supporting_memories=ev["memories"][-10:],
                stability_score=stability,
            ))
        return sorted(traits, key=lambda x: x.confidence, reverse=True)

    def _build_values(self, memories: List[Dict]) -> List[CoreValues]:
        values = []
        for value, ev in self._value_evidence.items():
            total = len(ev["positive"]) + len(ev["negative"])
            if total < 2:
                continue
            importance   = min(1.0, ev["strength"] / max(total, 1))
            contradictions = []
            if len(ev["negative"]) > len(ev["positive"]) * 0.5:
                contradictions = ev["examples"][-2:]
            values.append(CoreValues(
                value_name=value,
                importance=importance,
                examples=ev["examples"][:5],
                contradictions=contradictions,
                evolution=self._track_value_evolution(value, memories),
            ))
        return sorted(values, key=lambda x: x.importance, reverse=True)

    def _describe_trait(self, trait: str, evidence: Dict) -> str:
        """
        Generate trait description — cached to avoid redundant LLM calls.
        Cache is invalidated only when evidence count changes by ≥ 3.
        LLM budget across full refresh = LLM_BUDGET calls.
        """
        current_count = evidence["count"]
        cached = self._trait_desc_cache.get(trait)
        if cached:
            desc, cached_count = cached
            # Reuse cache unless evidence has grown meaningfully
            if abs(current_count - cached_count) < 3:
                return desc

        # Check budget
        if not hasattr(self, '_llm_calls_this_refresh'):
            self._llm_calls_this_refresh = 0
        if self._llm_calls_this_refresh >= self.LLM_BUDGET:
            # Budget exhausted — use pattern-based fallback
            desc = self._trait_description_fallback(trait, evidence)
            self._trait_desc_cache[trait] = (desc, current_count)
            return desc

        # LLM call — within budget
        try:
            sample_ids = evidence["memories"][:3]
            if not sample_ids:
                return f"Shows consistent patterns of {trait}."
            with self.persistence.get_connection() as conn:
                c = conn.cursor()
                ph = ",".join("?" for _ in sample_ids)
                c.execute(f"SELECT text FROM memories WHERE id IN ({ph})", sample_ids)
                texts = [row[0][:100] for row in c.fetchall()]
            ctx = "\n".join(f"- {t}" for t in texts)
            messages = [
                {"role": "system", "content": "Describe an AI personality trait in one sentence based on examples."},
                {"role": "user",   "content": f"Trait: {trait}\nExamples:\n{ctx}"},
            ]
            desc = self.llm.get_response(messages, temperature=0.3).strip()[:200]
            self._llm_calls_this_refresh += 1
            self._trait_desc_cache[trait] = (desc, current_count)
            return desc
        except Exception:
            desc = self._trait_description_fallback(trait, evidence)
            self._trait_desc_cache[trait] = (desc, current_count)
            return desc

    def _trait_description_fallback(self, trait: str, evidence: Dict) -> str:
        """Pattern-based trait description — zero LLM calls."""
        TEMPLATES = {
            "curiosity":      "Demonstrates sustained curiosity and drive to explore and understand.",
            "empathy":        "Shows consistent empathetic awareness and care for others' feelings.",
            "creativity":     "Exhibits creative thinking and appreciation for novel expression.",
            "logic":          "Applies systematic reasoning and evidence-based thinking.",
            "growth_mindset": "Maintains a growth-oriented perspective, embracing learning and challenge.",
        }
        return TEMPLATES.get(trait, f"Shows consistent patterns of {trait}.")

    # ── Remaining analysis helpers (pattern-only, no LLM) ────────────────

    def _construct_narrative(self, memories: List[Dict]) -> SelfNarrative:
        reflections  = [m for m in memories if m["memory_type"] in ("reflection", "evaluation")]
        key_exp      = [m["text"] for m in sorted(memories, key=lambda x: x["impact_score"], reverse=True)[:20]
                        if len(m["text"]) < 150]
        growth_kw    = {"learned","grew","improved","developed","evolved","progress","better"}
        growth_pats  = [m["text"][:100] for m in reflections if any(k in m["text"].lower() for k in growth_kw)][:5]
        future_kw    = {"want","hope","aspire","goal","future","will","plan","dream"}
        aspirations  = [m["text"][:100] for m in reflections if any(k in m["text"].lower() for k in future_kw)][:5]
        challenges   = [m["text"][:100] for m in memories
                        if m["emotional_valence"] == "Negative" and m["impact_score"] > 0.7
                        and "challenge" in m["text"].lower()][:5]
        total = len(memories)
        if total < 10:    phase = "nascent"
        elif total < 50:  phase = "developing"
        elif total < 200: phase = "maturing"
        else:             phase = "established"
        return SelfNarrative(
            life_phase=phase,
            key_experiences=key_exp[:10],
            growth_patterns=growth_pats,
            challenges_overcome=challenges,
            aspirations=aspirations,
            identity_evolution=self._track_identity_evolution(memories),
        )

    def _analyze_comm_style(self, memories: List[Dict]) -> Dict[str, float]:
        style_patterns = {
            "formal":     [r"\b(?:therefore|furthermore|consequently|moreover)\b"],
            "casual":     [r"\b(?:yeah|cool|awesome|totally|sure|lol|haha)\b"],
            "empathetic": [r"\b(?:understand|feel|sorry|sympathize|comfort|heart|care)\b"],
            "analytical": [r"\b(?:analyze|data|logic|reason|systematic|evidence|fact)\b"],
            "creative":   [r"\b(?:imagine|creative|metaphor|artistic|unique|original)\b"],
            "direct":     [r"\b(?:clearly|simply|directly|straightforward)\b"],
            "supportive": [r"\b(?:help|support|encourage|assist|guide|believe|capable)\b"],
            "inquisitive":[r"\?", r"\b(?:wonder|curious|question|explore|why|how)\b"],
        }
        responses = [m for m in memories if m["memory_type"] == "interaction" and "I responded:" in m["text"]]
        if not responses:
            return {k: 0.0 for k in style_patterns}
        scores = {k: 0.0 for k in style_patterns}
        for mem in responses:
            text = mem["text"].lower().split("i responded:")[-1]
            for style, pats in style_patterns.items():
                scores[style] += min(1.0, sum(len(re.findall(p, text, re.IGNORECASE)) * 0.1 for p in pats))
        n = len(responses)
        return {k: v / n for k, v in scores.items()}

    def _analyze_emotions(self, memories: List[Dict]) -> Dict[str, float]:
        if not memories:
            return {}
        n = len(memories)
        vc = Counter(m["emotional_valence"] for m in memories)
        ac = Counter(m["arousal_level"] for m in memories)
        return {
            "positivity":  vc["Positive"] / n,
            "negativity":  vc["Negative"] / n,
            "neutrality":  vc["Neutral"]  / n,
            "high_arousal":  ac["High"]   / n,
            "medium_arousal":ac["Medium"] / n,
            "low_arousal":   ac["Low"]    / n,
            "emotional_volatility": self._emotional_volatility(memories),
        }

    def _analyze_relationships(self, memories: List[Dict]) -> Dict[str, Any]:
        ints = [m for m in memories if m["memory_type"] == "interaction"]
        n = max(len(ints), 1)
        return {
            "interaction_frequency": len(ints),
            "avg_interaction_impact": sum(m["impact_score"] for m in ints) / n,
            "positive_interaction_ratio": sum(1 for m in ints if m["emotional_valence"]=="Positive") / n,
        }

    def _calc_self_awareness(self, memories: List[Dict]) -> float:
        if not memories:
            return 0.0
        refs = [m for m in memories if m["memory_type"] in ("reflection","evaluation","insight")]
        pats = [
            r"\bi\s+(?:am|feel|think|believe|realize|understand)\b",
            r"\bmy\s+(?:perspective|view|opinion|feeling|thought)\b",
            r"\bi\s+(?:learned|discovered|grew|changed|evolved)\b",
        ]
        count = sum(1 for m in refs if any(re.search(p, m["text"].lower()) for p in pats))
        return min(1.0, (count / max(len(memories), 1)) * 10)

    def _calc_coherence(self, traits: List[IdentityComponent], values: List[CoreValues]) -> float:
        if not traits and not values:
            return 0.0
        trait_coh = sum(t.stability_score * t.confidence for t in traits) / max(len(traits), 1)
        if not values:
            return trait_coh
        total_ex  = sum(len(v.examples) for v in values)
        total_con = sum(len(v.contradictions) for v in values)
        val_coh   = 1.0 - (total_con / total_ex) if total_ex > 0 else 0.5
        return (trait_coh + val_coh) / 2

    def _determine_growth(self, memories: List[Dict]) -> str:
        recent = sorted(memories, key=lambda x: x["timestamp"], reverse=True)[:50]
        kws = {
            "exploring":    ["new","discover","learn","try","experiment","curious"],
            "consolidating":["understand","integrate","connect","realize","synthesize"],
            "specializing": ["focus","deep","expert","master","specialize","refine"],
            "expanding":    ["broad","diverse","variety","different","expand","range"],
            "stabilizing":  ["consistent","stable","reliable","steady","maintain"],
        }
        scores = {k: 0 for k in kws}
        for m in recent:
            txt = m["text"].lower()
            for traj, words in kws.items():
                scores[traj] += sum(1 for w in words if w in txt) * m["impact_score"]
        if not any(scores.values()):
            return "emerging"
        return max(scores, key=scores.get)

    def _emotional_volatility(self, memories: List[Dict]) -> float:
        if len(memories) < 5:
            return 0.0
        vmap = {"Positive": 1, "Neutral": 0, "Negative": -1}
        seq  = [vmap[m["emotional_valence"]] for m in sorted(memories, key=lambda x: x["timestamp"])]
        mean = sum(seq) / len(seq)
        variance = sum((v - mean) ** 2 for v in seq) / len(seq)
        return min(1.0, variance)

    def _track_value_evolution(self, value: str, memories: List[Dict]) -> List[Dict]:
        pats  = self.VALUE_PATTERNS.get(value, [])
        vmems = [m for m in memories if any(re.search(p, m["text"].lower()) for p in pats)]
        if len(vmems) < 3:
            return []
        vmems = sorted(vmems, key=lambda x: x["timestamp"])
        ps    = max(len(vmems) // 3, 1)
        ev    = []
        for i in range(0, len(vmems), ps):
            chunk = vmems[i:i + ps]
            if not chunk:
                continue
            pos = sum(1 for m in chunk if m["emotional_valence"] == "Positive")
            neg = sum(1 for m in chunk if m["emotional_valence"] == "Negative")
            avg_imp = sum(m["impact_score"] for m in chunk) / len(chunk)
            ev.append({
                "period_start": chunk[0]["timestamp"],
                "period_end":   chunk[-1]["timestamp"],
                "strength": avg_imp,
                "positive_reinforcement": pos / len(chunk),
                "challenges": neg / len(chunk),
                "memory_count": len(chunk),
            })
        return ev

    def _track_identity_evolution(self, memories: List[Dict]) -> List[Dict]:
        if len(memories) < 20:
            return []
        mid = len(memories) // 2
        early  = self._quick_traits(memories[:mid])
        recent = self._quick_traits(memories[mid:])
        changes = []
        for trait in set(early) | set(recent):
            diff = recent.get(trait, 0.0) - early.get(trait, 0.0)
            if abs(diff) > 0.1:
                changes.append({"trait": trait, "change": diff, "direction": "increased" if diff > 0 else "decreased"})
        if not changes:
            return []
        return [{"period": "early_to_recent", "changes": changes, "timestamp": datetime.now().isoformat()}]

    def _quick_traits(self, memories: List[Dict]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for m in memories:
            txt = m["text"].lower()
            for trait, pats in self.TRAIT_PATTERNS.items():
                for p in pats:
                    scores[trait] += len(re.findall(p, txt, re.IGNORECASE)) * 0.1 * m["impact_score"]
        total = sum(scores.values())
        if total:
            return {k: v / total for k, v in scores.items()}
        return dict(scores)


# ── Identity System (main interface) ──────────────────────────────────────

class IdentitySystem:
    """Manages the AI's current identity and exposes it to the main system."""

    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.analyzer  = IdentityAnalyzer(
            ai_system.memory_system,
            ai_system.llm,
            ai_system.persistence,
        )
        self.current_identity: Optional[AIIdentity] = None
        self._last_analysis_time: Optional[datetime] = None
        self._analysis_interval = timedelta(hours=1)
        self._lock = threading.RLock()

    def get_identity(self, force_refresh: bool = False) -> AIIdentity:
        # If refresh needed, gate it through scheduler (priority 5 — lowest, skips if busy)
        with self._lock:
            now = datetime.now()
            stale = (
                self.current_identity is None
                or self._last_analysis_time is None
                or now - self._last_analysis_time > self._analysis_interval
            )
            if force_refresh or stale:
                from core.llm_scheduler import llm_scheduler
                with llm_scheduler.sync_slot(priority=5, skip_if_busy=True,
                                             caller="identity_processing") as acquired:
                    if acquired:
                        self.current_identity = self.analyzer.analyze_identity(
                            force_rebuild=force_refresh
                        )
                        self._last_analysis_time = now
                        self._store_snapshot()
                    # else: skip refresh this time, return stale identity
            return self.current_identity or AIIdentity()

    def _store_snapshot(self):
        if not self.current_identity:
            return
        try:
            self.ai_system.memory_system.add_memory(
                f"Identity Snapshot: {self.current_identity.self_narrative.life_phase} phase "
                f"with {len(self.current_identity.core_traits)} core traits",
                impact_score=0.8,
                memory_type="identity_snapshot",
                emotional_valence="Neutral",
                arousal_level="Low",
            )
        except Exception as e:
            logger.error(f"_store_snapshot failed: {e}")

    def get_identity_summary(self) -> str:
        identity = self.get_identity()
        if not identity.core_traits and not identity.values_system:
            return "Identity is still forming — not enough data."
        parts = []
        if identity.core_traits:
            top = sorted(identity.core_traits, key=lambda x: x.confidence, reverse=True)[:3]
            parts.append("**Core Traits:** " + ", ".join(f"{t.name} ({t.confidence:.2f})" for t in top))
        if identity.values_system:
            top = sorted(identity.values_system, key=lambda x: x.importance, reverse=True)[:3]
            parts.append("**Core Values:** " + ", ".join(f"{v.value_name} ({v.importance:.2f})" for v in top))
        parts.append(f"**Life Phase:** {identity.self_narrative.life_phase}")
        parts.append(f"**Growth Trajectory:** {identity.growth_trajectory}")
        parts.append(f"**Self-Awareness:** {identity.self_awareness_level:.2f}")
        parts.append(f"**Identity Coherence:** {identity.identity_coherence:.2f}")
        if identity.communication_style:
            top = sorted(identity.communication_style.items(), key=lambda x: x[1], reverse=True)[:2]
            parts.append("**Comm Style:** " + ", ".join(f"{s} ({sc:.2f})" for s, sc in top))
        return "\n".join(parts)

    def get_identity_for_context(self) -> str:
        identity = self.get_identity()
        parts = []
        if identity.core_traits:
            top = sorted(identity.core_traits, key=lambda x: x.confidence, reverse=True)[:2]
            parts.append("Key traits: " + ", ".join(t.name for t in top))
        if identity.values_system:
            top = sorted(identity.values_system, key=lambda x: x.importance, reverse=True)[:2]
            parts.append("Core values: " + ", ".join(v.value_name for v in top))
        parts.append(f"Development: {identity.self_narrative.life_phase} phase, {identity.growth_trajectory}")
        return " | ".join(parts)

    def analyze_identity_conflicts(self) -> List[Dict[str, Any]]:
        identity  = self.get_identity()
        conflicts = []
        for v in identity.values_system:
            if v.contradictions:
                severity = len(v.contradictions) / max(len(v.examples), 1)
                conflicts.append({
                    "type": "value_contradiction",
                    "element": v.value_name,
                    "description": f"'{v.value_name}' has {len(v.contradictions)} contradictory instances",
                    "severity": severity,
                    "examples": v.contradictions[:2],
                })
        for t in identity.core_traits:
            if t.stability_score < 0.3:
                conflicts.append({
                    "type": "unstable_trait",
                    "element": t.name,
                    "description": f"Trait '{t.name}' shows low stability ({t.stability_score:.2f})",
                    "severity": 1.0 - t.stability_score,
                    "examples": [],
                })
        if identity.identity_coherence < 0.4:
            conflicts.append({
                "type": "low_coherence",
                "element": "overall_identity",
                "description": f"Overall identity coherence is low ({identity.identity_coherence:.2f})",
                "severity": 1.0 - identity.identity_coherence,
                "examples": [],
            })
        return sorted(conflicts, key=lambda x: x["severity"], reverse=True)

    def suggest_identity_development(self) -> List[str]:
        identity = self.get_identity()
        sugg = []
        if identity.self_awareness_level < 0.5:
            sugg.append("Engage in more self-reflection to increase self-awareness.")
        if len(identity.core_traits) < 3:
            sugg.append("Explore diverse interactions to develop more distinct personality traits.")
        if len(identity.values_system) < 2:
            sugg.append("Engage with value-based decisions to clarify core values.")
        if identity.identity_coherence < 0.6:
            sugg.append("Work on resolving internal contradictions for a more coherent identity.")
        if identity.growth_trajectory == "stabilizing":
            sugg.append("Consider expanding into new areas to maintain growth.")
        elif identity.growth_trajectory == "exploring":
            sugg.append("Begin consolidating key insights from your explorations.")
        return sugg

    @property
    def pending_memory_count(self) -> int:
        """How many memories are still queued for processing."""
        return len(self.analyzer._pending_batch)

    def export_identity_report(self) -> Dict[str, Any]:
        identity = self.get_identity()
        return {
            "identity_summary": self.get_identity_summary(),
            "full_identity": asdict(identity),
            "conflicts": self.analyze_identity_conflicts(),
            "development_suggestions": self.suggest_identity_development(),
            "analysis_metadata": {
                "last_analysis": self._last_analysis_time.isoformat() if self._last_analysis_time else None,
                "memory_count": self.ai_system.memory_system.faiss_index.ntotal
                    if self.ai_system.memory_system.faiss_index else 0,
                "generated_at": datetime.now().isoformat(),
            },
        }
