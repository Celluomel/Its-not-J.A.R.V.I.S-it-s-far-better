"""
cognition/resolution_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ResolutionEngine — intentional closure for unresolved questions.

The gap this closes
────────────────────
Currently:

  curiosity → pressure → reactivation (indefinite)

Missing:

  curiosity → investigation → evidence → conclusion → archive

Without closure, Lumina risks becoming an engine for generating
unresolved internal activity — questions multiply faster than they
are answered, and the open_questions list grows without bound.

What this adds
──────────────
  1. Investigation trigger — when a question has been active long
     enough and has sufficient activation, the engine initiates an
     active investigation: queries semantic memory for related
     concepts, checks if any recent reflection outputs touched the
     domain, and synthesises a resolution attempt via LLM.

  2. Evidence aggregation — collects evidence from three sources:
       a. Semantic memory (stored knowledge on the concept)
       b. Recent autonomous reflections (v42 outputs)
       c. Recent FAISS memories related to the concept
     Evidence is weighted and scored for relevance.

  3. Resolution attempt — LLM synthesises a conclusion from the
     evidence. The conclusion is scored for confidence.
     High-confidence conclusions (≥ 0.65) → question archived.
     Medium confidence (0.40–0.65) → question marked "tentative"
     and continues with lower activation.
     Low confidence (< 0.40) → question persists unchanged.

  4. Archive — resolved questions are stored to NarrativeIdentity
     and as FAISS memory, then removed from _open_questions.
     This makes resolution visible and durable.

  5. Unresolvable detection — questions active for > MAX_AGE_DAYS
     with no evidence accumulation are marked "unresolvable" and
     archived without a conclusion. This prevents zombie questions.

Execution
─────────
  Called from InternalThoughtLoop slow cycle every RESOLVE_EVERY_N
  cycles. LLM priority = 3 (skip_if_busy). Non-fatal.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RESOLVE_EVERY_N         = 15    # slow cycles between resolution attempts
MIN_ACTIVATION_TO_RESOLVE = 0.45  # question must be active enough
MIN_AGE_TO_RESOLVE_HOURS  = 6.0   # must have existed for at least this long
MAX_AGE_DAYS              = 14.0  # questions older than this are archived regardless
MIN_EVIDENCE_COUNT        = 1     # must have at least one evidence hit

HIGH_CONFIDENCE_THRESHOLD = 0.65
MED_CONFIDENCE_THRESHOLD  = 0.40
MAX_RESOLUTION_LOG        = 100
MAX_TOKENS                = 200


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ResolutionAttempt:
    """A logged resolution attempt."""
    timestamp:      float = field(default_factory=time.time)
    question_text:  str   = ""
    concept:        str   = ""
    evidence_count: int   = 0
    conclusion:     str   = ""
    confidence:     float = 0.0
    outcome:        str   = ""    # "resolved" | "tentative" | "insufficient" | "unresolvable"
    slow_cycle:     int   = 0


@dataclass
class ResolutionState:
    """Persisted state."""
    resolution_log:    List[Dict] = field(default_factory=list)
    total_resolved:    int = 0
    total_tentative:   int = 0
    total_unresolvable: int = 0


# ── Engine ────────────────────────────────────────────────────────────────────

class ResolutionEngine:
    """
    Actively investigates and closes unresolved open questions.

    Usage (from InternalThoughtLoop slow cycle):
        re = ResolutionEngine(organism, ai_system)
        re.tick(slow_cycle_count)
    """

    def __init__(
        self,
        organism:  Any,
        ai_system: Any,
        path:      str = "data/persona/resolution_engine.json",
    ):
        self._o    = organism
        self._ai   = ai_system
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = ResolutionState()
        self._load()
        logger.info(
            f"[ResolutionEngine] Initialised — "
            f"{self._state.total_resolved} resolved, "
            f"{self._state.total_unresolvable} unresolvable"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Try to resolve the most eligible open question. Non-fatal."""
        if slow_cycle % RESOLVE_EVERY_N != 0 or slow_cycle == 0:
            return
        # v50: gate check — resolution requires an LLM synthesis call;
        # skip when cognitive_energy is critically low.
        try:
            from core.state import state as _st
            _org  = getattr(getattr(_st, 'persona', None), '_organism', None)
            _gate = getattr(_org, 'behavior_gate', None) if _org else None
            if _gate and not _gate.may_run_resolution():
                return
        except Exception:
            pass
        try:
            self._attempt_resolution(slow_cycle)
        except Exception as e:
            logger.debug(f"[ResolutionEngine] tick error (non-fatal): {e}")

    # ── Candidate selection ───────────────────────────────────────────────────

    def _attempt_resolution(self, slow_cycle: int) -> None:
        """Select the best candidate question and attempt resolution."""
        ts  = getattr(self._o, "thought_stream", None)
        oqs = getattr(ts, "_open_questions", []) if ts else []

        if not oqs:
            return

        now = time.time()
        candidates = []

        for oq in oqs:
            age_hours = (now - oq.created_at) / 3600

            # Unresolvable check — archive old questions with no evidence
            if (now - oq.created_at) / 86400 > MAX_AGE_DAYS:
                if oq.evidence_count < MIN_EVIDENCE_COUNT:
                    self._archive_unresolvable(oq, ts, slow_cycle)
                    continue

            # Eligibility: active enough, old enough, has some evidence
            if (oq.activation >= MIN_ACTIVATION_TO_RESOLVE
                    and age_hours >= MIN_AGE_TO_RESOLVE_HOURS
                    and oq.evidence_count >= MIN_EVIDENCE_COUNT):
                # Score: higher activation + more evidence = better candidate
                score = oq.activation * 0.6 + min(1.0, oq.evidence_count / 3) * 0.4
                candidates.append((score, oq))

        if not candidates:
            return

        # Add policy alignment score — questions aligned with active values
        # score higher, biasing resolution toward value-consistent exploration.
        try:
            _policy = getattr(getattr(self._o, 'ai_system', None), '_decision_policy', None)
            if _policy:
                candidates = [
                    (base_score + 0.25 * _policy.score(
                        oq.linked_concept + " " + getattr(oq, 'question', ''),
                        emphasise=["curiosity", "intellectual_depth"]
                    ), oq)
                    for base_score, oq in candidates
                ]
        except Exception:
            pass

        # Take highest-scored candidate
        candidates.sort(key=lambda x: -x[0])
        _, best = candidates[0]

        # Gather evidence
        evidence = self._gather_evidence(best.linked_concept)
        if not evidence:
            return

        # Attempt LLM resolution
        attempt = self._resolve_with_llm(best, evidence, slow_cycle)
        if attempt is None:
            return

        # Apply outcome
        self._apply_outcome(attempt, best, ts)

        with self._lock:
            self._state.resolution_log.append(asdict(attempt))
            if len(self._state.resolution_log) > MAX_RESOLUTION_LOG:
                self._state.resolution_log = self._state.resolution_log[-MAX_RESOLUTION_LOG:]
        self._save()

    # ── Evidence gathering ────────────────────────────────────────────────────

    def _gather_evidence(self, concept: str) -> List[str]:
        """
        Collect evidence from three sources:
          1. Semantic memory related concepts
          2. Recent autonomous reflection outputs
          3. Recent FAISS memory retrieval
        """
        evidence: List[str] = []

        # Source 1: Semantic memory
        try:
            ai  = getattr(self._o, "ai_system", None)
            sem = getattr(ai, "semantic_memory", None) if ai else None
            if sem:
                relations = sem.get_related(concept, limit=5)
                for rel in relations:
                    if rel.weight > 0.20:
                        evidence.append(
                            f"Semantic: '{rel.source}' relates to '{rel.target}' "
                            f"(weight={rel.weight:.2f})"
                        )
        except Exception:
            pass

        # Source 2: Recent autonomous reflections
        try:
            are = getattr(
                getattr(self._o, "_loop", None),
                "_autonomous_reflection", None
            )
            if are:
                recent = getattr(are._state, "reflection_log", [])[-8:]
                for record in recent:
                    output = record.get("output", "").lower()
                    if concept.lower() in output:
                        evidence.append(f"Reflection: {record.get('output','')[:80]}")
        except Exception:
            pass

        # Source 3: FAISS memory retrieval
        try:
            ai = getattr(self._o, "ai_system", None)
            ms = getattr(ai, "memory_system", None) if ai else None
            if ms:
                memories = ms.retrieve_memories(concept, limit=4)
                for m in memories:
                    content = m.get("content", "")
                    if content and concept.lower() in content.lower():
                        evidence.append(f"Memory: {content[:80]}")
        except Exception:
            pass

        return evidence[:8]   # cap at 8 evidence items

    # ── LLM resolution ────────────────────────────────────────────────────────

    def _resolve_with_llm(
        self,
        oq:         Any,
        evidence:   List[str],
        slow_cycle: int,
    ) -> Optional[ResolutionAttempt]:
        """Attempt to synthesise a conclusion from gathered evidence."""
        from core.llm_scheduler import llm_scheduler

        evidence_text = "\n".join(f"  - {e}" for e in evidence)

        prompt = (
            f"You are Lumina's resolution process, investigating an open question.\n\n"
            f"Open question: {oq.text}\n"
            f"Related concept: {oq.linked_concept}\n"
            f"Evidence gathered:\n{evidence_text}\n\n"
            f"Task: Based on this evidence, attempt a conclusion.\n"
            f"Be honest about confidence. Do not overclaim.\n"
            f"Respond ONLY with JSON:\n"
            f'{{"conclusion": "...", "confidence": 0.6, '
            f'"reasoning": "brief explanation"}}\n\n'
            f"Confidence guide: 0.7+ = well-supported, 0.4–0.7 = tentative, "
            f"<0.4 = insufficient evidence."
        )

        result = ""
        with llm_scheduler.sync_slot(
            priority    = 3,
            skip_if_busy= True,
            caller      = "resolution_engine",
        ) as acquired:
            if not acquired:
                return None
            try:
                result = self._ai.get_response(
                    messages   = [{"role": "user", "content": prompt}],
                    max_tokens = MAX_TOKENS,
                    temperature= 0.55,
                ) or ""
            except Exception as e:
                logger.debug(f"[ResolutionEngine] LLM failed: {e}")
                return None

        try:
            import re, json as _json
            clean = re.sub(r"```json|```", "", result).strip()
            data  = _json.loads(clean)
            conclusion = data.get("conclusion", "")
            confidence = float(data.get("confidence", 0.0))

            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                outcome = "resolved"
            elif confidence >= MED_CONFIDENCE_THRESHOLD:
                outcome = "tentative"
            else:
                outcome = "insufficient"

            return ResolutionAttempt(
                question_text  = oq.text,
                concept        = oq.linked_concept,
                evidence_count = len(evidence),
                conclusion     = conclusion[:200],
                confidence     = round(confidence, 3),
                outcome        = outcome,
                slow_cycle     = slow_cycle,
            )
        except Exception as e:
            logger.debug(f"[ResolutionEngine] parse failed: {e}")
            return None

    # ── Apply outcome ─────────────────────────────────────────────────────────

    def _apply_outcome(
        self,
        attempt: ResolutionAttempt,
        oq:      Any,
        ts:      Any,
    ) -> None:
        """Apply the resolution outcome to the open question and stores."""
        if attempt.outcome == "resolved":
            self._archive_resolved(attempt, oq, ts)
            with self._lock:
                self._state.total_resolved += 1

        elif attempt.outcome == "tentative":
            # Lower activation, mark partial resolution
            oq.activation   = max(0.2, oq.activation * 0.55)
            oq.uncertainty  = max(0.15, oq.uncertainty * 0.60)
            oq.revisit_bias = max(0.10, oq.revisit_bias * 0.70)
            with self._lock:
                self._state.total_tentative += 1
            # Store tentative conclusion to workspace
            try:
                ws = getattr(self._o, "workspace", None)
                if ws:
                    ws.broadcast(
                        source   = "resolution_engine.tentative",
                        content  = (
                            f"[Tentative resolution — {oq.linked_concept}] "
                            f"{attempt.conclusion} (confidence={attempt.confidence:.0%})"
                        ),
                        priority = 0.44,
                    )
            except Exception:
                pass
            logger.info(
                f"[ResolutionEngine] Tentative: '{oq.linked_concept}' "
                f"(conf={attempt.confidence:.0%})"
            )

        else:   # insufficient — persist unchanged
            logger.debug(
                f"[ResolutionEngine] Insufficient evidence for "
                f"'{oq.linked_concept}' (conf={attempt.confidence:.0%})"
            )

    def _archive_resolved(
        self,
        attempt: ResolutionAttempt,
        oq:      Any,
        ts:      Any,
    ) -> None:
        """Fully resolve and archive a question."""
        # Remove from open questions
        try:
            if ts and oq in getattr(ts, "_open_questions", []):
                ts._open_questions.remove(oq)
        except Exception:
            pass

        # Store conclusion to NarrativeIdentity
        try:
            ni = getattr(self._o, "narrative_identity", None)
            if ni and hasattr(ni, "record_chapter"):
                ni.record_chapter(
                    title       = f"Resolved: {oq.linked_concept}",
                    description = attempt.conclusion,
                    emotion     = "satisfaction",
                    significance= min(0.80, attempt.confidence),
                )
        except Exception:
            pass

        # Store to FAISS memory
        try:
            ai = getattr(self._o, "ai_system", None)
            ms = getattr(ai, "memory_system", None) if ai else None
            if ms:
                ms.add_memory(
                    f"[Resolved] {oq.text}: {attempt.conclusion}",
                    0.88, "principle", "Positive", "Low",
                    memory_tier="cognitive"
                )
        except Exception:
            pass

        # Broadcast resolution
        try:
            ws = getattr(self._o, "workspace", None)
            if ws:
                ws.broadcast(
                    source   = "resolution_engine.resolved",
                    content  = (
                        f"[Question resolved — {oq.linked_concept}] "
                        f"{attempt.conclusion}"
                    ),
                    priority = 0.60,
                )
        except Exception:
            pass

        logger.info(
            f"[ResolutionEngine] ✅ Resolved: '{oq.linked_concept}' "
            f"after {oq.evidence_count} evidence hits "
            f"(conf={attempt.confidence:.0%})"
        )

    def _archive_unresolvable(
        self,
        oq: Any,
        ts: Any,
        slow_cycle: int,
    ) -> None:
        """Archive a question that has exceeded max age with no evidence."""
        try:
            if ts and oq in getattr(ts, "_open_questions", []):
                ts._open_questions.remove(oq)
        except Exception:
            pass

        with self._lock:
            self._state.total_unresolvable += 1
            self._state.resolution_log.append(asdict(ResolutionAttempt(
                question_text  = oq.text,
                concept        = oq.linked_concept,
                evidence_count = oq.evidence_count,
                conclusion     = "Unresolvable — archived after max age with no evidence",
                confidence     = 0.0,
                outcome        = "unresolvable",
                slow_cycle     = slow_cycle,
            )))

        logger.info(
            f"[ResolutionEngine] 🗂  Unresolvable: '{oq.linked_concept}' "
            f"(age={( time.time()-oq.created_at)/86400:.1f}d, "
            f"evidence={oq.evidence_count})"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = ResolutionState(
                    resolution_log      = data.get("resolution_log",      []),
                    total_resolved      = data.get("total_resolved",      0),
                    total_tentative     = data.get("total_tentative",     0),
                    total_unresolvable  = data.get("total_unresolvable",  0),
                )
        except Exception as e:
            logger.warning(f"[ResolutionEngine] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os; os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[ResolutionEngine] Save failed: {e}")
