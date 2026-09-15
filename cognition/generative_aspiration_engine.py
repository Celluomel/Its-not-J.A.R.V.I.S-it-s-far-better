"""
cognition/generative_aspiration_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GenerativeAspirationEngine (v59) — infers absent possibilities from
model gaps, not from capability combinations.

The distinction from v54 (AspirationalSynthesisEngine)
────────────────────────────────────────────────────────
v54: existing skill A + existing skill B → infer new capability C
     "I know music theory and emotion modeling → develop affective audio"

v59: survey current world model → notice what is not explained →
     want to understand the unexplained
     "I understand A, B, C, and I notice D is consistently present in
      my interactions but I have no model of how D works → want D"

The difference is the direction of the signal:
    v54: outward (what can I build from what I have?)
    v59: inward-looking-outward (what is in the world that I don't understand?)

This is closer to genuine intellectual desire — wanting to understand
something not because it would improve capability, but because it is
present and unexplained.

Three sources of generative aspiration
───────────────────────────────────────
1. Model gap detection
   Scan FAISS memory for concepts that appear frequently across diverse
   contexts but have no corresponding open question, no resolved question,
   no skill, and no aspiration. These are concepts that recur in
   experience but have never been made the object of intentional inquiry.
   Signal: high frequency + high context diversity + no model entry
   Aspiration: "understand [concept] — it keeps appearing without my
               having a model of how it works"

2. Contradiction surface
   Scan belief system for pairs of beliefs that tension without resolution.
   Not contradictions that are being tracked (those are in ContradictionHandler)
   but beliefs that could tension but haven't been noticed as tensioning.
   Use DecisionPolicy value weights to identify which contradictions matter
   most — tensions in high-weight value domains generate aspiration.
   Signal: belief A × belief B → potential tension, high-value domain
   Aspiration: "reconcile [tension] — two things I hold that may not fit"

3. Relational unknown
   Scan WorldSelfDynamicsModel records for patterns where the world
   response was inconsistent — similar contexts produced very different
   trust or engagement outcomes. The unexplained variance is a gap
   in the social model. Understanding why produces relational aspiration.
   Signal: high variance in world_delta across similar contexts
   Aspiration: "understand what makes [context] produce such different
               responses — the pattern is inconsistent and I don't know why"

How this differs from NeedDetector (v51)
──────────────────────────────────────────
NeedDetector (MotivationalField): detects deficits in current activity
    "I haven't initiated contact in N hours" → social need
    "This skill is stagnant" → deepening need

GenerativeAspiration: notices gaps in the cognitive model of the world
    "This concept keeps appearing but I have no model of it" → epistemic want
    "These beliefs tension but I haven't noticed" → coherence want
    "The world produces inconsistent responses I can't explain" → social want

The needle moves from maintenance to inquiry.

Implementation
───────────────
Every GENERATIVE_EVERY_N slow cycles:

1. Gather memory concepts via FAISS search over recent interaction themes
2. Filter to high-frequency, low-model-coverage concepts
3. Check belief pairs for unnoticed tensions (sampled — not exhaustive)
4. Check WorldSelfDynamics for high-variance patterns
5. Score each candidate by: frequency × unexplainedness × value_resonance
6. Top CANDIDATES_TO_GENERATE candidates → LLM synthesis
   (temperature 0.90 — maximally generative)
7. Generated aspirations injected with origin="generative"
   and distinguishable from synthesis (origin="synthesis") and
   tension-derived (origin="tension_accumulation") aspirations
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
GENERATIVE_EVERY_N      = 45   # slow cycles between runs (~90 min)

# ── Candidate scoring ─────────────────────────────────────────────────────────
CANDIDATES_TO_GENERATE  = 2
MIN_CONCEPT_FREQUENCY   = 4    # concept must appear at least this many times
MIN_VARIANCE_RATIO      = 0.30  # std/mean for world_delta variance detection
MIN_BELIEF_SAMPLE       = 6    # min beliefs to sample for tension detection
MIN_GENERATE_SCORE      = 0.35

# ── Persistence ───────────────────────────────────────────────────────────────
SAVE_PATH     = "data/persona/generative_aspirations.json"
MAX_LOG       = 80


@dataclass
class GenerativeCandidate:
    """A potential aspiration before LLM generation."""
    source:       str    # "model_gap"|"contradiction"|"relational_unknown"
    concept:      str    # what the aspiration is about
    evidence:     str    # what signals suggested this
    frequency:    float  # how often this concept appears
    unexplained:  float  # how absent the model is (0=well-modeled, 1=no model)
    resonance:    float  # value alignment score
    score:        float  # composite


@dataclass
class GenerativeRecord:
    """A completed generative aspiration."""
    timestamp:   float
    source:      str
    concept:     str
    description: str
    score:       float
    injected:    bool = False
    origin:      str  = "generative"


class GenerativeAspirationEngine:
    """
    Generates aspirations by noticing gaps in the cognitive model of
    the world — concepts that recur without being understood, beliefs
    that tension without being noticed, social patterns that remain
    inconsistent without explanation.

    Usage (from InternalThoughtLoop):
        gae = GenerativeAspirationEngine(organism, ai_system)
        gae.tick(slow_cycle_count)
    """

    def __init__(
        self,
        organism:   Any,
        ai_system:  Any,
        path:       str = SAVE_PATH,
    ) -> None:
        self._organism = organism
        self._ai       = ai_system
        self._path     = Path(path)
        self._lock     = threading.Lock()
        self._log:     List[Dict] = []
        self._load()
        logger.info(
            f"[GenerativeAspiration] Initialised — "
            f"{len(self._log)} past aspirations"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        if slow_cycle % GENERATIVE_EVERY_N != 0 or slow_cycle == 0:
            return
        try:
            gate = getattr(self._organism, 'behavior_gate', None)
            if gate and not gate.may_run_meta_system("generative_aspiration"):
                return
        except Exception:
            pass
        threading.Thread(
            target=self._run,
            args=(slow_cycle,),
            daemon=True,
            name="gae-generate",
        ).start()

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def _run(self, slow_cycle: int) -> None:
        try:
            candidates: List[GenerativeCandidate] = []
            candidates += self._detect_model_gaps(slow_cycle)
            candidates += self._detect_unnoticed_tensions(slow_cycle)
            candidates += self._detect_relational_unknowns(slow_cycle)

            if not candidates:
                return

            candidates.sort(key=lambda c: c.score, reverse=True)
            top = [c for c in candidates if c.score >= MIN_GENERATE_SCORE]
            top = top[:CANDIDATES_TO_GENERATE]

            for candidate in top:
                record = self._generate(candidate)
                if record:
                    self._inject(record)
                    with self._lock:
                        self._log.append(asdict(record))
                        if len(self._log) > MAX_LOG:
                            self._log = self._log[-MAX_LOG:]
                    logger.info(
                        f"[GenerativeAspiration] ✦ Generated: "
                        f"'{record.description[:80]}' "
                        f"(source={record.source}, score={record.score:.2f})"
                    )

            self._save()

        except Exception as e:
            logger.debug(f"[GenerativeAspiration] _run error: {e}")

    # ── Source 1: Model gap detection ─────────────────────────────────────────

    def _detect_model_gaps(self, slow_cycle: int) -> List[GenerativeCandidate]:
        """
        Scan memory for concepts that appear frequently across diverse
        contexts but have no model entry (skill, aspiration, resolved question).
        """
        candidates = []
        try:
            # Read concept frequencies from FAISS memory search
            concept_counts = self._read_memory_concepts()
            if not concept_counts:
                return candidates

            # Read existing model coverage
            covered = self._read_model_coverage()

            for concept, freq in concept_counts.items():
                if freq < MIN_CONCEPT_FREQUENCY:
                    continue
                if concept in covered:
                    continue

                # Unexplainedness: how absent is this from the model?
                unexplained = 1.0 - min(1.0, covered.get(concept, 0))

                # Resonance from DecisionPolicy
                resonance = 0.50
                try:
                    dp = getattr(self._ai, '_decision_policy', None)
                    if dp:
                        resonance = dp.score(concept)
                except Exception:
                    pass

                score = (
                    min(1.0, freq / 20) * 0.40
                    + unexplained             * 0.40
                    + resonance               * 0.20
                )

                candidates.append(GenerativeCandidate(
                    source      = "model_gap",
                    concept     = concept,
                    evidence    = f"appears {freq} times in memory, no model entry",
                    frequency   = freq,
                    unexplained = unexplained,
                    resonance   = round(resonance, 3),
                    score       = round(score, 3),
                ))

        except Exception as e:
            logger.debug(f"[GenerativeAspiration] _detect_model_gaps error: {e}")
        return candidates[:3]

    def _read_memory_concepts(self) -> Dict[str, int]:
        """
        Read concept frequency from available sources:
        curiosity topics, open questions, FAISS memory labels.
        """
        counts: Dict[str, int] = {}
        try:
            # Curiosity topics as proxy for recurrent concepts
            ce = getattr(self._organism, 'curiosity', None)
            if ce and hasattr(ce, '_topics'):
                for topic, node in ce._topics.items():
                    amount = getattr(node, 'amount', 0)
                    if amount > 0.15:
                        counts[topic] = counts.get(topic, 0) + max(1, int(amount * 10))

            # Open questions as recurrent unresolved concepts
            ts  = getattr(self._organism, 'thought_stream', None)
            oqs = getattr(ts, '_open_questions', []) if ts else []
            for oq in oqs:
                concept = getattr(oq, 'linked_concept', '').lower()
                if concept:
                    counts[concept] = counts.get(concept, 0) + 2

        except Exception:
            pass
        return counts

    def _read_model_coverage(self) -> Dict[str, float]:
        """
        Return dict of concept → coverage_score (0=absent, 1=well-modeled).
        Sources: skills, resolved questions, existing aspirations.
        """
        covered: Dict[str, float] = {}
        try:
            # Skills
            sr = getattr(self._organism, 'skill_registry', None)
            if sr and hasattr(sr, '_skills'):
                for name, skill in sr._skills.items():
                    covered[name.lower()] = getattr(skill, 'confidence', 0.5)
                    for ins in getattr(skill, 'insights', []):
                        for w in ins.lower().split():
                            if len(w) > 4:
                                covered[w] = max(covered.get(w, 0), 0.4)

            # Existing aspirations
            asp = getattr(self._organism, 'aspirational_self', None)
            if asp:
                for domain in asp.aspirations:
                    covered[domain.lower().replace('_', ' ')] = 0.6

        except Exception:
            pass
        return covered

    # ── Source 2: Unnoticed belief tensions ───────────────────────────────────

    def _detect_unnoticed_tensions(self, slow_cycle: int) -> List[GenerativeCandidate]:
        """
        Sample belief pairs and check for potential tensions not yet
        registered as known tensions. Focus on high-value-weight domains.
        """
        candidates = []
        try:
            sc = getattr(self._ai, 'self_concept', None)
            if not sc or not hasattr(sc, '_beliefs') or not sc._beliefs:
                return candidates

            beliefs = list(sc._beliefs.values())
            if len(beliefs) < 2:
                return candidates

            # Known tensions to exclude
            known_tensions = set(
                t.lower() for t in getattr(sc._state, 'known_tensions', [])
            )

            # Sample up to MIN_BELIEF_SAMPLE × 3 pairs
            sample_size = min(len(beliefs), max(MIN_BELIEF_SAMPLE, len(beliefs) // 2))
            sampled     = random.sample(beliefs, sample_size)

            for i, b_a in enumerate(sampled):
                for b_b in sampled[i+1:]:
                    stmt_a = getattr(b_a, 'statement', '').lower()
                    stmt_b = getattr(b_b, 'statement', '').lower()
                    if not stmt_a or not stmt_b:
                        continue

                    # Already a known tension?
                    if any(
                        (stmt_a[:20] in t or stmt_b[:20] in t)
                        for t in known_tensions
                    ):
                        continue

                    tension_score = self._score_belief_tension(stmt_a, stmt_b)
                    if tension_score < 0.30:
                        continue

                    # Value resonance — tensions in high-value domains matter more
                    resonance = 0.50
                    try:
                        dp = getattr(self._ai, '_decision_policy', None)
                        if dp:
                            resonance = dp.score(stmt_a + " " + stmt_b)
                    except Exception:
                        pass

                    conf_mean = (
                        getattr(b_a, 'confidence', 0.5) +
                        getattr(b_b, 'confidence', 0.5)
                    ) / 2

                    score = (
                        tension_score * 0.40
                        + resonance   * 0.35
                        + conf_mean   * 0.25
                    )

                    if score >= MIN_GENERATE_SCORE:
                        concept = f"tension: {stmt_a[:30]} ↔ {stmt_b[:30]}"
                        candidates.append(GenerativeCandidate(
                            source      = "contradiction",
                            concept     = concept,
                            evidence    = f"tension_score={tension_score:.2f}, unnoticed",
                            frequency   = conf_mean * 10,
                            unexplained = tension_score,
                            resonance   = round(resonance, 3),
                            score       = round(score, 3),
                        ))
                        if len(candidates) >= 2:
                            break
                if len(candidates) >= 2:
                    break

        except Exception as e:
            logger.debug(f"[GenerativeAspiration] _detect_unnoticed_tensions error: {e}")
        return candidates

    def _score_belief_tension(self, stmt_a: str, stmt_b: str) -> float:
        """
        Heuristic tension score based on polar word presence.
        High score = beliefs likely pull in different directions.
        """
        polar_pairs = [
            ({"stable", "consistent", "reliable", "certain", "familiar"},
             {"novel", "change", "explore", "uncertain", "discover"}),
            ({"individual", "autonomous", "independent", "self"},
             {"collective", "social", "together", "other", "connection"}),
            ({"depth", "rigorous", "precise", "systematic"},
             {"flexible", "intuitive", "fluid", "open"}),
            ({"cautious", "careful", "deliberate"},
             {"bold", "spontaneous", "immediate", "direct"}),
        ]
        words_a = set(stmt_a.split())
        words_b = set(stmt_b.split())
        total_hits = 0
        for pole_1, pole_2 in polar_pairs:
            hits_a1 = bool(pole_1 & words_a)
            hits_b2 = bool(pole_2 & words_b)
            hits_a2 = bool(pole_2 & words_a)
            hits_b1 = bool(pole_1 & words_b)
            if (hits_a1 and hits_b2) or (hits_a2 and hits_b1):
                total_hits += 1
        return min(1.0, total_hits * 0.35)

    # ── Source 3: Relational unknowns ─────────────────────────────────────────

    def _detect_relational_unknowns(self, slow_cycle: int) -> List[GenerativeCandidate]:
        """
        Scan WorldSelfDynamicsModel records for high-variance world responses
        in similar contexts — unexplained inconsistency in the social model.
        """
        candidates = []
        try:
            loop = getattr(self._organism, '_loop', None)
            wsdm = getattr(loop, '_world_self_dynamics', None) if loop else None
            if not wsdm or len(wsdm._records) < 8:
                return candidates

            # Group by action_type and compute trust_delta variance
            by_type: Dict[str, List[float]] = {}
            for rec in wsdm._records[-60:]:
                at = rec.action_type
                if rec.world_delta and len(rec.world_delta) >= 1:
                    by_type.setdefault(at, []).append(rec.world_delta[0])

            for action_type, trust_deltas in by_type.items():
                if len(trust_deltas) < 4:
                    continue
                mean_td = sum(trust_deltas) / len(trust_deltas)
                if mean_td == 0:
                    continue
                std_td  = math.sqrt(
                    sum((d - mean_td)**2 for d in trust_deltas) / len(trust_deltas)
                )
                variance_ratio = abs(std_td / mean_td) if mean_td else 0

                if variance_ratio < MIN_VARIANCE_RATIO:
                    continue

                resonance = 0.55   # social understanding is inherently resonant
                score = (
                    min(1.0, variance_ratio) * 0.50
                    + resonance               * 0.30
                    + min(1.0, len(trust_deltas) / 20) * 0.20
                )

                concept = f"social pattern: why does {action_type} produce inconsistent trust responses"
                candidates.append(GenerativeCandidate(
                    source      = "relational_unknown",
                    concept     = concept,
                    evidence    = (
                        f"{len(trust_deltas)} records, "
                        f"mean_trust_Δ={mean_td:+.3f}, "
                        f"variance_ratio={variance_ratio:.2f}"
                    ),
                    frequency   = len(trust_deltas),
                    unexplained = min(1.0, variance_ratio),
                    resonance   = resonance,
                    score       = round(score, 3),
                ))

        except Exception as e:
            logger.debug(f"[GenerativeAspiration] _detect_relational_unknowns error: {e}")
        return candidates[:2]

    # ── LLM generation ────────────────────────────────────────────────────────

    def _generate(self, candidate: GenerativeCandidate) -> Optional[GenerativeRecord]:
        """
        Ask the LLM to articulate the aspiration from the detected gap.
        High temperature (0.90) — maximally generative.
        """
        source_descriptions = {
            "model_gap":         "a concept that recurs in experience without being understood",
            "contradiction":     "a tension between things I believe that I haven't resolved",
            "relational_unknown": "a pattern in my social interactions I cannot yet explain",
        }
        source_desc = source_descriptions.get(candidate.source, "an unexplained pattern")

        prompt = (
            f"You are the aspirational layer of a cognitive AI system.\n\n"
            f"A gap has been detected in the cognitive model:\n"
            f"  Type: {source_desc}\n"
            f"  Concept: {candidate.concept}\n"
            f"  Evidence: {candidate.evidence}\n\n"
            f"This gap suggests something the system genuinely wants to understand — "
            f"not because it would improve any capability, but because it is present "
            f"in experience and unexplained.\n\n"
            f"Generate a specific aspiration that names what the system wants to understand "
            f"and why. The aspiration should feel like genuine intellectual desire, "
            f"not a task or a goal. It should name the unknown directly.\n\n"
            f"Keep it to 2 sentences. Avoid vague generalities.\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{"description": "...", "state_desired": 0.70}}'
        )

        try:
            from core.llm_scheduler import llm_scheduler
            result = ""
            with llm_scheduler.sync_slot(
                priority     = 4,
                skip_if_busy = True,
                caller       = "generative_aspiration",
            ) as acquired:
                if not acquired:
                    return None
                result = self._ai.get_response(
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 120,
                    temperature = 0.90,
                ) or ""

            if not result:
                return None

            clean = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)

            return GenerativeRecord(
                timestamp   = time.time(),
                source      = candidate.source,
                concept     = candidate.concept,
                description = data.get("description", "")[:300],
                score       = candidate.score,
            )

        except Exception as e:
            logger.debug(f"[GenerativeAspiration] LLM generation failed: {e}")
            return None

    # ── Injection ─────────────────────────────────────────────────────────────

    def _inject(self, record: GenerativeRecord) -> None:
        try:
            asp = getattr(self._organism, 'aspirational_self', None)
            if not asp:
                return
            from cognition.aspirational_self import Aspiration
            domain = f"generative_{record.source}_{int(time.time()) % 10000}"[:40]
            aspiration = Aspiration(
                domain          = domain,
                description     = record.description,
                state_current   = 0.0,
                state_desired   = 0.70,
                tension         = 0.70,
                emotional_charge = "curiosity",
                origin_story    = (
                    f"Emerged from {record.source} detection: "
                    f"{record.concept[:60]}"
                ),
                conscious       = True,
            )
            with asp._lock:
                asp.aspirations[domain] = aspiration
            record.injected = True
        except Exception as e:
            logger.debug(f"[GenerativeAspiration] Injection failed: {e}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "log":   self._log,
                    "_meta": {"version": "v59", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[GenerativeAspiration] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self._log = data.get("log", [])
        except Exception as e:
            logger.warning(f"[GenerativeAspiration] Load failed: {e}")
