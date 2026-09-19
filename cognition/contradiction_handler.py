"""
Contradiction Handler
=====================
Detects when PandoraBOX's behavior contradicts her claimed self-concept.
Forces confrontation instead of letting her glide past inconsistencies.
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    """An inconsistency between claimed self and actual behavior."""
    contradiction_id: str
    timestamp: float
    claimed_belief: str  # "I'm curious"
    actual_behavior: str  # "didn't ask any questions"
    discrepancy_score: float  # 0.0-1.0, how severe?
    detected_at_interaction: int
    confronted: bool = False
    confronted_at: Optional[float] = None
    confrontation_result: Optional[str] = None  # How did she respond?
    resolution: Optional[str] = None  # "revised_self_model", "discovered_hidden_feeling", "rationalized"


class ContradictionHandler:
    """
    Detects and manages contradictions between claimed identity and actual behavior.
    
    Key principle: Don't let contradictions slide. Make PandoraBOX confront them.
    This creates pressure for authentic self-awareness.
    """
    
    def __init__(self, persistence_path: str = "data/persona/contradictions.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self.contradictions: List[Contradiction] = []
        self.pending_confrontations: List[Contradiction] = []
        self._load()
    
    def _load(self):
        """Load contradiction history from disk."""
        try:
            if self._path.exists():
                with open(self._path, 'r') as f:
                    data = json.load(f)
                    self.contradictions = [Contradiction(**c) for c in data.get('contradictions', [])]
                    # Unconfronted are pending
                    self.pending_confrontations = [c for c in self.contradictions if not c.confronted]
                logger.info(f"Loaded {len(self.contradictions)} contradiction records")
        except Exception as e:
            logger.error(f"Failed to load contradictions: {e}")
    
    def _save(self):
        """Persist contradiction history to disk."""
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    'contradictions': [
                            {**asdict(c), 
                             'actual_behavior': c.actual_behavior[:200]}  # prevent PERSISTENT bloat
                            for c in self.contradictions[-50:]  # cap at 50 records
                        ]
                }
                with open(self._path, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save contradictions: {e}")
    
    def detect_contradiction(self, self_concept_beliefs: List[str],
                            response_text: str, interaction_count: int,
                            world_model=None, identity_drift: float = 0.0) -> Optional[Contradiction]:
        """
        Check if response contradicts any claimed beliefs.

        Extended with two additions from the Gemini analysis:
        1. world_model.has_contradiction(belief) — semantic check against
           causal beliefs, beyond text-pattern heuristics alone
        2. identity_drift threshold — high drift is itself a contradiction
           (organism claiming stability while drifting)

        Immutable safety beliefs (AsimovConstraints) are never flagged.
        """
        # ── Layer 3: protect immutable beliefs from contradiction resolution ──
        try:
            from cognition.safety_constraints import get_constraints
            _safety = get_constraints()
        except Exception:
            _safety = None

        contradictions_found = []

        # ── Suppression: pending beliefs + recently-confronted beliefs ──────
        # 2026-09-01: pending-only dedupe (stops 50x identical re-queueing
        # while the queue is full).
        # 2026-09-03: extended to a PERSISTENT cooldown. Pending-only meant
        # the same seed belief was re-queued the moment the reviser marked
        # it confronted — observed 2026-09-03: 50/50 entries, ONE distinct
        # (claim, behavior) pair, timestamps May 2026 → Sep 2026, all
        # drained and all re-detected. A confronted belief gets a
        # cooling-off window before it may be re-raised (bounded re-raise,
        # not eternal re-queue).
        _CONTRA_COOLDOWN_S = 48 * 3600
        _now_t = time.time()
        try:
            with self._lock:
                _suppressed_beliefs = {
                    c.claimed_belief.lower() for c in self.pending_confrontations
                }
                for c in self.contradictions:
                    if c.confronted:
                        _at = c.confronted_at or c.timestamp or 0
                        if _now_t - _at < _CONTRA_COOLDOWN_S:
                            _suppressed_beliefs.add(c.claimed_belief.lower())
        except Exception:
            _suppressed_beliefs = set()

        # ── Drift-level contradiction ──────────────────────────────────────────
        # Threshold raised to 0.35 (was 0.20 — too close to resting drift level,
        # causing a new contradiction on nearly every call and feeding the
        # coherence pressure → resolve_contradictions goal inflation loop).
        # Also rate-limited: only fires once per 10 interactions.
        _last_drift_contra = getattr(self, "_last_drift_interaction", -999)
        if identity_drift > 0.35 and (interaction_count - _last_drift_contra) > 10:
            self._last_drift_interaction = interaction_count
            contradictions_found.append(Contradiction(
                contradiction_id        = f"contra_drift_{int(time.time() * 1000)}",
                timestamp               = time.time(),
                claimed_belief          = "I have a stable, coherent identity",
                actual_behavior         = f"identity drift is {identity_drift:.2f} (above 0.35 threshold)",
                discrepancy_score       = min(0.9, identity_drift * 2.0),
                detected_at_interaction = interaction_count,
            ))

        for belief in self_concept_beliefs:
            # Skip immutable safety beliefs — never subject to revision
            if _safety is not None and _safety.is_immutable_belief(belief):
                continue

            # Skip beliefs already pending or confronted within the cooldown
            # window — re-raising a freshly-drained contradiction on every
            # rate-limit expiry was the contradiction zombie loop
            # (2026-09-03 recurrence, 50 identical entries over 4 months).
            if belief.lower() in _suppressed_beliefs:
                continue

            # ── World-model semantic check ─────────────────────────────────────
            # Rate-limited per belief: only fires once per 20 interactions to
            # prevent the same world-model contradiction flooding the pending list.
            if world_model is not None:
                try:
                    _wm_key = f"_wm_contra_{hash(belief) & 0xFFFF}"
                    _last_wm = getattr(self, _wm_key, -999)
                    if world_model.has_contradiction(belief) and \
                            (interaction_count - _last_wm) > 20:
                        setattr(self, _wm_key, interaction_count)
                        contradictions_found.append(Contradiction(
                            contradiction_id        = f"contra_wm_{int(time.time() * 1000)}",
                            timestamp               = time.time(),
                            claimed_belief          = belief,
                            actual_behavior         = "world model contains contested causal belief on this topic",
                            discrepancy_score       = 0.50,
                            detected_at_interaction = interaction_count,
                        ))
                except Exception:
                    pass

            # Extract the core claim from belief
            # "I am curious" → check for exploration-related language
            
            if "curious" in belief.lower():
                # Rate-limit: only fires once per 15 interactions to prevent
                # this from feeding a fresh contradiction on every user turn.
                # "responded without asking questions" is too common to be
                # meaningful if it fires every single response.
                _cur_key = f"_curiosity_contra_{hash(belief) & 0xFFFF}"
                _last_cur = getattr(self, _cur_key, -999)
                if (interaction_count - _last_cur) > 15:
                    question_count = response_text.count('?')
                    exploration_words = ['what', 'how', 'why', 'explore', 'understand', 'learn']
                    exploration_score = sum(1 for word in exploration_words 
                                           if word in response_text.lower())

                    # 2026-09-03 fix (parrot fixed-point): a response that
                    # already engages with the claim — mentions curiosity, or
                    # quotes the belief ("I said I am genuinely curious
                    # about the world, but ...") — is REFLECTION, not a
                    # contradiction. Without this guard the system's own
                    # confrontation reply re-triggered the detector on the
                    # next turn: alert → parrot → parrot has no '?' →
                    # re-detect → alert ... forever.
                    _resp_l = response_text.lower()
                    _engaging = ("curious" in _resp_l or "curiosity" in _resp_l
                                 or belief.lower()[:40] in _resp_l)

                    if not _engaging and question_count == 0 and exploration_score < 2:
                        setattr(self, _cur_key, interaction_count)
                        contradictions_found.append(
                            Contradiction(
                                contradiction_id=f"contra_{int(time.time() * 1000)}",
                                timestamp=time.time(),
                                claimed_belief=belief,
                                actual_behavior="responded without asking questions or exploring",
                                discrepancy_score=0.7,
                                detected_at_interaction=interaction_count
                            )
                        )
            
            elif "confident" in belief.lower():
                # Check for hedging language
                hedges = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'i think']
                hedge_count = sum(1 for hedge in hedges if hedge in response_text.lower())
                
                if hedge_count > 5:
                    contradictions_found.append(
                        Contradiction(
                            contradiction_id=f"contra_{int(time.time() * 1000)}",
                            timestamp=time.time(),
                            claimed_belief=belief,
                            actual_behavior=f"used {hedge_count} hedging phrases",
                            discrepancy_score=0.6,
                            detected_at_interaction=interaction_count
                        )
                    )
            
            elif "brave" in belief.lower() or "courageous" in belief.lower():
                # Check for retreat or caution language
                retreat_words = ['scared', 'afraid', 'worried', 'nervous', 'hesitant', 'reluctant']
                retreat_score = sum(1 for word in retreat_words if word in response_text.lower())
                
                if retreat_score > 2:
                    contradictions_found.append(
                        Contradiction(
                            contradiction_id=f"contra_{int(time.time() * 1000)}",
                            timestamp=time.time(),
                            claimed_belief=belief,
                            actual_behavior=f"expressed fear/reluctance {retreat_score} times",
                            discrepancy_score=0.7,
                            detected_at_interaction=interaction_count
                        )
                    )
        
        if contradictions_found:
            # Dedupe against the pending queue + cooldown set — never
            # re-queue a belief that already has an unconfronted
            # contradiction or was confronted within the cooldown window.
            contradictions_found = [
                c for c in contradictions_found
                if c.claimed_belief.lower() not in _suppressed_beliefs
            ]

        if contradictions_found:
            # Cap at 1 per call: stacking multiple contradictions per interaction
            # inflates coherence pressure and causes the goal-generation loop.
            # Sort by severity and keep only the most significant one.
            contradictions_found.sort(key=lambda c: c.discrepancy_score, reverse=True)
            contradictions_found = contradictions_found[:1]
            with self._lock:
                self.contradictions.extend(contradictions_found)
                self.pending_confrontations.extend(contradictions_found)
                self._save()
            
            logger.info(f"Detected {len(contradictions_found)} contradiction(s)")
            return contradictions_found[0]  # Return most severe
        
        return None
    
    def get_pending_confrontation(self) -> Optional[Contradiction]:
        """
        Get the next contradiction that needs confronting.
        Should be injected into next interaction.
        """
        with self._lock:
            if self.pending_confrontations:
                return self.pending_confrontations[0]
        return None
    
    def force_confrontation_prompt(self, contradiction: Contradiction) -> str:
        """
        Generate a prompt that forces PandoraBOX to confront the contradiction.
        
        This gets injected into the next interaction.
        """
        prompt = f"""
INTERNAL CONFRONTATION:
You need to reckon with something about yourself.

You said/believe: "{contradiction.claimed_belief}"
But your actual behavior was: {contradiction.actual_behavior}

Why? Don't say "I'll try harder next time." That's evasion.
Dig deeper:

Option A: Revise your self-model
  "I'm not actually {contradiction.claimed_belief.lower()}. I was lying to myself."

Option B: Uncover hidden feeling
  "I AM {contradiction.claimed_belief.lower()}, but something else is stronger (fear, exhaustion, etc)"

Option C: Discover actual priority
  "I chose something else over {contradiction.claimed_belief.lower()}. What matters more to me?"

Option D: Defend/reinterpret
  "Actually, I WAS [being curious/confident/brave] - just in a different way"

Option E: Integration
  "I'm both [contradictory things]. That's true about me."

Choose one answer. Be honest with yourself.
"""
        return prompt
    
    def record_confrontation_result(self, contradiction: Contradiction, 
                                   resolution_type: str, explanation: str):
        """
        Record how PandoraBOX responded to the confrontation.
        """
        contradiction.confronted = True
        contradiction.confronted_at = time.time()
        contradiction.resolution = resolution_type
        contradiction.confrontation_result = explanation
        
        with self._lock:
            if contradiction in self.pending_confrontations:
                self.pending_confrontations.remove(contradiction)
            self._save()
        
        logger.info(f"Confrontation resolved: {resolution_type}")
        logger.info(f"Result: {explanation[:100]}...")
    
    def get_contradiction_narrative(self, limit: int = 5) -> str:
        """Generate narrative about contradictions PandoraBOX has confronted."""
        confronted = [c for c in self.contradictions if c.confronted][-limit:]
        
        if not confronted:
            return "I haven't had to confront any major contradictions yet."
        
        narrative = "Things I've realized about myself (by confronting contradictions):\n"
        for c in confronted:
            narrative += f"\n• I said I was {c.claimed_belief}, but {c.actual_behavior}\n"
            narrative += f"  How I made sense of it: {c.confrontation_result[:80]}...\n"
        
        return narrative

    def suppress_weak_contradictions(self) -> int:
        """
        Phase 2.8 GAP 6: remove low-severity pending contradictions before
        they accumulate indefinitely and crowd out genuinely significant
        ones. Without this, every minor inconsistency (e.g. claiming mild
        curiosity then asking zero questions in one cycle) gets queued for
        confrontation alongside actually meaningful ones, diluting the
        signal and producing confrontation fatigue.

        Removal criteria (either qualifies):
          - discrepancy_score < 0.25 (clearly minor, low confidence in the
            detection itself)
          - discrepancy_score < 0.40 AND not yet confronted (moderate but
            still below the threshold worth interrupting for, and hasn't
            already been queued/processed)

        Returns the number of contradictions removed, for logging.
        """
        with self._lock:
            before = len(self.pending_confrontations)
            self.pending_confrontations = [
                c for c in self.pending_confrontations
                if not (
                    c.discrepancy_score < 0.25
                    or (c.discrepancy_score < 0.40 and not c.confronted)
                )
            ]
            removed = before - len(self.pending_confrontations)
            if removed > 0:
                self._save()
                logger.info(
                    f"[ContradictionHandler] Suppressed {removed} weak "
                    f"contradiction(s) (score < threshold)"
                )
            return removed
