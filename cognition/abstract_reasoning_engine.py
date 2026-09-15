"""
cognition/abstract_reasoning_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Abstract Reasoning Engine (ARE) — Lumina's second priority skill.

Fills the gap between "reasoning about internal state" (existing) and
"reasoning with abstract concepts outward" (this module).

What it does per conversation turn:
  Pass 1 — Decompose   : break the concept into structural components
  Pass 2 — Map         : find analogies in semantic memory + world model
  Pass 3 — Synthesize  : derive something not explicitly stated

Side effects:
  → new CausalBelief   fed into WorldModel automatically
  → open_question      planted as curiosity seed in CuriosityEngine
  → analogy chain      stored in SemanticMemory as new concept

Activation:
  ARE only fires when the topic is abstract / conceptual.
  Heuristic detection + optional LLM trigger-check.
  Skips when LLM scheduler is busy (returns cached result or None).

Prompt injection:
  get_reasoning_fragment(user_id) → string injected into system prompt
  Concise — 2-4 lines — not a full chain-of-thought dump.

Public API:
  are = AbstractReasoningEngine(organism, llm, memory)
  result = are.reason(user_id, user_text)    # call before response
  frag   = are.get_reasoning_fragment(user_id)
  are.record_outcome(user_id, ai_response)   # call after response

Design principles:
  • Does NOT replace the InnerMonologueEngine two-pass.
    ARE feeds INTO the system prompt; IME shapes the expression pass.
  • LLM calls are bare (no history pollution).
  • Graceful degradation — heuristic fallback if LLM busy.
  • Capped at 3 LLM calls per turn total (decompose + map + synthesize).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Topic abstractness thresholds ─────────────────────────────────────────
ABSTRACT_TOPIC_KEYWORDS = {
    # Philosophical / conceptual
    "consciousness", "identity", "meaning", "existence", "reality", "truth",
    "free will", "ethics", "morality", "justice", "beauty", "knowledge",
    "causation", "time", "infinity", "emergence", "complexity",
    # Structural / systemic
    "system", "pattern", "structure", "model", "framework", "architecture",
    "hierarchy", "network", "feedback", "recursion", "iteration",
    # Cross-domain analogies
    "like", "analogy", "similar to", "metaphor", "mirrors", "parallels",
    "just as", "in the same way", "equivalent to",
    # Hypothetical
    "what if", "imagine", "suppose", "could we", "would it", "hypothetically",
    "in theory", "let's say", "assuming",
    # Reasoning signals
    "because", "therefore", "implies", "leads to", "causes", "results in",
    "conclude", "derive", "follows that", "logic", "reasoning", "argument",
    "principle", "axiom", "assumption",
}

MIN_ABSTRACT_KEYWORD_HITS = 2   # how many keywords needed to trigger ARE
MIN_TEXT_LENGTH            = 20  # ignore very short messages
ARE_COOLDOWN_S             = 30  # min seconds between full LLM ARE runs per user


@dataclass
class ReasoningResult:
    """Output of one full ARE run."""
    user_id:        str
    timestamp:      float = field(default_factory=time.time)

    # Pass 1 — Decomposition
    components:     List[str] = field(default_factory=list)
    core_concept:   str       = ""

    # Pass 2 — Analogy mapping
    best_analogy:   str       = ""
    analogy_domain: str       = ""
    analogy_confidence: float = 0.0

    # Pass 3 — Synthesis
    derived_insight: str      = ""
    open_question:   str      = ""
    causal_pair:     Tuple[str, str] = ("", "")  # (antecedent, consequent)

    # Meta
    method:         str       = "heuristic"    # "heuristic" | "llm"
    abstract_score: float     = 0.0            # 0–1 how abstract the input was
    prompt_fragment: str      = ""


class AbstractReasoningEngine:
    """
    Structured abstract reasoning: Decompose → Map → Synthesize.

    Feeds new insights into WorldModel, CuriosityEngine, SemanticMemory.
    Injects a concise reasoning fragment into the system prompt.
    """

    def __init__(
        self,
        organism:       Any,
        llm:            Any,
        memory:         Any = None,   # MemoryManager
    ):
        self._org    = organism
        self._llm    = llm
        self._mem    = memory
        self._lock   = threading.Lock()
        self._cache: Dict[str, ReasoningResult] = {}   # user_id → last result
        self._continuity_cache: Dict[str, ReasoningResult] = {}
        self._last_run: Dict[str, float] = {}           # user_id → timestamp
        self._last_input: Dict[str, str] = {}            # prevents cross-topic reuse

        # Domain analogy library — pairs (domain, structural pattern)
        # Lumina draws from these when mapping abstract concepts
        self._analogy_seeds = [
            ("chess",          "strategic evaluation + lookahead under uncertainty"),
            ("evolution",      "selection pressure on variants over time"),
            ("thermodynamics", "energy gradient → flow → equilibrium"),
            ("ecology",        "niche competition + symbiosis + carrying capacity"),
            ("music",          "tension → resolution → motif development"),
            ("immune system",  "pattern recognition + tolerance + response amplification"),
            ("rivers",         "path of least resistance + erosion feedback"),
            ("language",       "combinatorial rules generating infinite expression"),
            ("markets",        "distributed signal aggregation + feedback loops"),
            ("crystallization","local order emerging from global chaos"),
        ]

        logger.info("🧠 AbstractReasoningEngine initialised")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Primary public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def reason(self, user_id: str, user_text: str) -> Optional[ReasoningResult]:
        """
        Run abstract reasoning on user_text if the topic warrants it.
        Returns None if the text is not abstract enough to trigger ARE.
        Always fast — LLM runs are best-effort and non-blocking.
        """
        # ── Abstractness check ─────────────────────────────────────────────
        abstract_score = self._score_abstractness(user_text)
        if abstract_score < 0.3:
            # Preserve the prior insight for conversational continuity, but
            # remove it from the current-turn slot so it cannot masquerade as
            # analysis of this new practical or concrete message.
            with self._lock:
                previous = self._cache.pop(user_id, None)
                if previous is not None:
                    self._continuity_cache[user_id] = previous
            return None

        # ── Cooldown check ────────────────────────────────────────────────
        now = time.time()
        with self._lock:
            last = self._last_run.get(user_id, 0.0)
            last_input = self._last_input.get(user_id, "")
        if now - last < ARE_COOLDOWN_S:
            if " ".join(user_text.casefold().split()) == last_input:
                with self._lock:
                    result = self._cache.get(user_id) or self._continuity_cache.get(user_id)
                    if result is not None:
                        self._cache[user_id] = result
                    return result
            # Do not spend another LLM call during cooldown, but reason about
            # this turn rather than injecting the previous turn's result.
            result = self._heuristic_reason(user_id, user_text, abstract_score)
            self._apply_side_effects(result)
            with self._lock:
                previous = self._cache.get(user_id) or self._continuity_cache.get(user_id)
                if previous is not None:
                    self._continuity_cache[user_id] = previous
                self._cache[user_id] = result
                self._last_input[user_id] = " ".join(user_text.casefold().split())
            return result

        # ── Try LLM run ───────────────────────────────────────────────────
        result = None
        try:
            from core.llm_scheduler import llm_scheduler
            if not llm_scheduler.is_busy() and self._llm:
                result = self._llm_reason(user_id, user_text, abstract_score)
        except Exception:
            pass

        # ── Heuristic fallback ────────────────────────────────────────────
        if result is None:
            result = self._heuristic_reason(user_id, user_text, abstract_score)

        # ── Side effects ──────────────────────────────────────────────────
        self._apply_side_effects(result)

        # ── Cache ─────────────────────────────────────────────────────────
        with self._lock:
            previous = self._cache.get(user_id)
            if previous is not None:
                self._continuity_cache[user_id] = previous
            self._cache[user_id]    = result
            self._last_run[user_id] = now
            self._last_input[user_id] = " ".join(user_text.casefold().split())

        logger.debug(
            f"🧠 ARE [{result.method}] score={abstract_score:.2f} "
            f"concept={result.core_concept!r} "
            f"analogy={result.best_analogy[:40]!r}"
        )
        return result

    def get_reasoning_fragment(self, user_id: str) -> str:
        """
        Return current reasoning plus explicitly subordinate prior context.
        """
        with self._lock:
            result = self._cache.get(user_id)
            previous = self._continuity_cache.get(user_id)
        now = time.time()
        parts = []
        if result and now - result.timestamp <= 30:
            parts.append(result.prompt_fragment)
        if previous and previous is not result and now - previous.timestamp <= 120:
            parts.append(
                "[Previous-turn abstract context - continuity only]\n"
                f"{previous.prompt_fragment}\n"
                "Use this only where it remains directly relevant. It must not "
                "replace or override the current message's primary goal, facts, "
                "or material prerequisites."
            )
        return "\n".join(parts)

    def record_outcome(self, user_id: str, ai_response: str) -> None:
        """
        Post-turn feedback. Check if the response actually used the insight.
        Strengthens or weakens the causal belief accordingly.
        """
        with self._lock:
            result = self._cache.get(user_id)
        if not result or not result.causal_pair[0]:
            return

        antecedent, consequent = result.causal_pair
        # Simple heuristic: if the derived insight appeared in the response, strengthen
        if result.derived_insight and result.derived_insight[:30].lower() in ai_response.lower():
            confidence_boost = 0.05
        else:
            confidence_boost = 0.01

        try:
            wm = getattr(self._org, "world_model", None)
            if wm and hasattr(wm, "learn_causal"):
                # Retrieve existing belief and boost
                for belief in getattr(wm, "causal_beliefs", []):
                    if belief.antecedent == antecedent and belief.consequent == consequent:
                        belief.confidence = min(0.95, belief.confidence + confidence_boost)
                        break
        except Exception:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Abstractness scoring
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _score_abstractness(self, text: str) -> float:
        """
        0–1 score of how abstract/conceptual the input is.
        Combines keyword density, question complexity, length, and structure.
        """
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return 0.0

        text_lower = text.lower()
        words      = text_lower.split()

        # Keyword density
        hits = sum(
            1 for kw in ABSTRACT_TOPIC_KEYWORDS
            if kw in text_lower
        )
        density = min(1.0, hits / MIN_ABSTRACT_KEYWORD_HITS)

        # Structural complexity (longer = more likely abstract)
        length_bonus = min(0.3, len(words) / 100.0)

        # Question complexity
        question_bonus = 0.1 if "?" in text else 0.0

        # Negation / qualification (signs of nuanced thinking)
        qualification = 0.1 if any(
            w in words for w in ["but", "however", "although", "despite", "unless"]
        ) else 0.0

        score = density * 0.6 + length_bonus + question_bonus + qualification
        return min(1.0, score)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LLM three-pass reasoning
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _llm_reason(
        self, user_id: str, text: str, abstract_score: float
    ) -> Optional[ReasoningResult]:
        """
        Run all three passes via LLM. Each pass is a bare call (no history).
        Returns None on failure so heuristic fallback can take over.
        """
        try:
            # ── Retrieve semantic memory context ─────────────────────────────
            mem_context = ""
            if self._mem and hasattr(self._mem, "get_context"):
                mem_context = self._mem.get_context(text, top_k=3)

            # ── World model causal context ────────────────────────────────────
            wm_context = ""
            try:
                wm = getattr(self._org, "world_model", None)
                if wm and hasattr(wm, "prompt_fragment"):
                    wm_context = wm.prompt_fragment()
            except Exception:
                pass

            # ── PASS 1: Decompose ─────────────────────────────────────────────
            decompose_prompt = f"""Analyze the abstract concept in this message and break it down.

Message: "{text}"

Respond ONLY with JSON, no other text:
{{
  "core_concept": "<the central abstract idea in 1-5 words>",
  "components": ["<component 1>", "<component 2>", "<component 3>"],
  "domain": "<what field or area does this primarily belong to>",
  "is_cross_domain": true | false
}}"""

            raw1 = self._llm.generate_bare(
                decompose_prompt,
                max_tokens=120,
                temperature=0.25,
            )
            raw1 = re.sub(r"```json|```", "", raw1).strip()
            d1   = json.loads(raw1)

            core_concept = d1.get("core_concept", "")
            components   = d1.get("components", [])[:4]

            # ── PASS 2: Analogy mapping ───────────────────────────────────────
            seed_list = "\n".join(
                f"  {domain}: {pattern}"
                for domain, pattern in self._analogy_seeds
            )
            map_prompt = f"""Find the best structural analogy for this concept.

Core concept: {core_concept}
Components: {components}
Memory context: {mem_context[:300] if mem_context else 'none'}
World model: {wm_context[:200] if wm_context else 'none'}

Candidate analogy domains:
{seed_list}

Respond ONLY with JSON:
{{
  "best_analogy_domain": "<domain name from the list, or a new one>",
  "structural_mapping": "<how the two things are structurally similar, 1-2 sentences>",
  "confidence": 0.0-1.0,
  "what_it_reveals": "<what the analogy reveals that wasn't obvious, 1 sentence>"
}}"""

            raw2 = self._llm.generate_bare(
                map_prompt,
                max_tokens=150,
                temperature=0.45,
            )
            raw2 = re.sub(r"```json|```", "", raw2).strip()
            d2   = json.loads(raw2)

            best_analogy      = d2.get("structural_mapping", "")
            analogy_domain    = d2.get("best_analogy_domain", "")
            analogy_confidence = float(d2.get("confidence", 0.5))
            what_it_reveals   = d2.get("what_it_reveals", "")

            # ── PASS 3: Synthesize ────────────────────────────────────────────
            synth_prompt = f"""Synthesize a novel insight from this conceptual analysis.

Original message: "{text}"
Core concept: {core_concept}
Components: {components}
Best analogy: {best_analogy} (domain: {analogy_domain})
What the analogy reveals: {what_it_reveals}

Respond ONLY with JSON:
{{
  "derived_insight": "<something true and non-obvious that follows from the analysis, 1-2 sentences>",
  "open_question": "<the most interesting unresolved question this raises, 1 sentence>",
  "causal_antecedent": "<a short cause phrase, e.g. 'prolonged focus on abstract problems'>",
  "causal_consequent": "<a short effect phrase, e.g. 'increased need for external grounding'>",
  "prompt_note": "<1 sentence for how Lumina should hold this in her response — not to quote, just to inform>"
}}"""

            raw3 = self._llm.generate_bare(
                synth_prompt,
                max_tokens=180,
                temperature=0.55,
            )
            raw3 = re.sub(r"```json|```", "", raw3).strip()
            d3   = json.loads(raw3)

            derived_insight = d3.get("derived_insight", "")
            open_question   = d3.get("open_question", "")
            causal_pair     = (
                d3.get("causal_antecedent", "")[:80],
                d3.get("causal_consequent", "")[:80],
            )
            prompt_note     = d3.get("prompt_note", "")

            # ── Build prompt fragment ─────────────────────────────────────────
            fragment = self._build_fragment(
                core_concept, best_analogy, analogy_domain,
                derived_insight, open_question, prompt_note
            )

            return ReasoningResult(
                user_id            = user_id,
                components         = components,
                core_concept       = core_concept,
                best_analogy       = best_analogy,
                analogy_domain     = analogy_domain,
                analogy_confidence = analogy_confidence,
                derived_insight    = derived_insight,
                open_question      = open_question,
                causal_pair        = causal_pair,
                method             = "llm",
                abstract_score     = abstract_score,
                prompt_fragment    = fragment,
            )

        except Exception as e:
            logger.debug(f"ARE LLM pass failed: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Heuristic fallback (no LLM needed)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _heuristic_reason(
        self, user_id: str, text: str, abstract_score: float
    ) -> ReasoningResult:
        """Fast heuristic reasoning — keyword-based, no LLM."""
        text_lower = text.lower()

        # Identify core concept from top keyword hit
        hit_keywords = [kw for kw in ABSTRACT_TOPIC_KEYWORDS if kw in text_lower]
        core_concept = hit_keywords[0] if hit_keywords else "abstract idea"

        # Find best analogy seed by keyword overlap
        best_domain, best_pattern, best_score = "", "", 0.0
        for domain, pattern in self._analogy_seeds:
            score = sum(1 for w in domain.split() if w in text_lower)
            score += sum(1 for w in pattern.split() if w in text_lower)
            if score > best_score:
                best_score  = score
                best_domain = domain
                best_pattern = pattern

        if not best_domain:
            best_domain, best_pattern = self._analogy_seeds[0]  # chess default

        # Minimal synthesis
        derived_insight = (
            f"The concept of {core_concept} may share structural properties "
            f"with {best_domain} — specifically {best_pattern}."
        )
        open_question = f"What are the limits of the {best_domain} analogy for {core_concept}?"

        fragment = self._build_fragment(
            core_concept, best_pattern, best_domain,
            derived_insight, open_question, ""
        )

        return ReasoningResult(
            user_id         = user_id,
            core_concept    = core_concept,
            best_analogy    = best_pattern,
            analogy_domain  = best_domain,
            derived_insight = derived_insight,
            open_question   = open_question,
            causal_pair     = (core_concept, f"structural pattern like {best_domain}"),
            method          = "heuristic",
            abstract_score  = abstract_score,
            prompt_fragment = fragment,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Side effects — WorldModel, CuriosityEngine, SemanticMemory
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _apply_side_effects(self, result: ReasoningResult) -> None:
        """
        Feed reasoning results back into the cognitive architecture.
        All calls are best-effort — never raise.
        """
        # ── WorldModel: new causal belief ─────────────────────────────────
        try:
            wm = getattr(self._org, "world_model", None)
            ant, con = result.causal_pair
            if wm and hasattr(wm, "learn_causal") and ant and con:
                wm.learn_causal(
                    antecedent  = ant,
                    consequent  = con,
                    confidence  = 0.4 + result.analogy_confidence * 0.3,
                )
                logger.debug(f"🧠 ARE → WorldModel: {ant!r} → {con!r}")
        except Exception:
            pass

        # ── CuriosityEngine: plant open question as curiosity seed ─────────
        try:
            if result.open_question:
                ce = getattr(self._org, "curiosity_engine", None)
                if ce and hasattr(ce, "plant_question"):
                    ce.plant_question(result.open_question, priority=0.6)
                elif ce and hasattr(ce, "add_question"):
                    ce.add_question(result.open_question)
                logger.debug(f"🧠 ARE → Curiosity: {result.open_question!r}")
        except Exception:
            pass

        # ── SemanticMemory: store analogy as new concept link ─────────────
        try:
            if result.best_analogy and result.core_concept:
                sm = getattr(self._org, "semantic_memory", None)
                if sm and hasattr(sm, "add_concept"):
                    sm.add_concept(
                        name        = result.core_concept,
                        description = result.best_analogy,
                        source      = f"analogy:{result.analogy_domain}",
                    )
                logger.debug(
                    f"🧠 ARE → SemanticMemory: {result.core_concept!r} "
                    f"via {result.analogy_domain}"
                )
        except Exception:
            pass

        # ── Memory: add derived insight as episodic trace ─────────────────
        try:
            if result.derived_insight and self._mem and hasattr(self._mem, "add_memory"):
                self._mem.add_memory(
                    f"[ARE insight] {result.derived_insight}",
                    metadata={"role": "sys", "source": "abstract_reasoning"},
                )
        except Exception:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Prompt fragment builder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_fragment(
        core_concept:    str,
        analogy:         str,
        analogy_domain:  str,
        derived_insight: str,
        open_question:   str,
        prompt_note:     str,
    ) -> str:
        """
        Build a concise prompt fragment — 3-4 lines max.
        Not a chain-of-thought dump. Informs without overwhelming.
        """
        lines = [f"[Abstract reasoning: concept='{core_concept}'"]

        if analogy_domain and analogy:
            short_analogy = analogy[:80] + ("…" if len(analogy) > 80 else "")
            lines.append(f" Analogy→{analogy_domain}: {short_analogy}")

        if derived_insight:
            short_insight = derived_insight[:100] + ("…" if len(derived_insight) > 100 else "")
            lines.append(f" Derived: {short_insight}")

        if open_question:
            lines.append(f" Open: {open_question[:80]}")

        if prompt_note:
            lines.append(f" Note: {prompt_note[:80]}")

        lines.append("]")
        return "\n".join(lines)
