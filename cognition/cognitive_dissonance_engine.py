"""
CognitiveDissonanceEngine (CDE)
================================
Detects conflicts between Lumina's beliefs, identity, and actions, then
resolves them through identity mutation, belief revision, or meta-thread
creation.

Three detection modes:
  1. BELIEF_VS_EVIDENCE  — world model says X, recent action implies not-X
  2. IDENTITY_VS_ACTION  — self-model claims trait Y, behavior contradicts Y
  3. GOAL_VS_OUTCOME     — thread predicted result Z, actual result was not-Z

Effects when dissonance is detected:
  - Thread confidence decreases
  - Thread dissonance score rises
  - A meta-thread is spawned (e.g. "why did I act against my values?")
  - resolution triggers identity/belief mutation

Integration:
  - Called after TTE.step() in InternalThoughtLoop._slow_cycle()
  - Meta-threads are injected back into TTE
  - Resolved events trigger personality_evolution.queue_experience()
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cognition.thought_thread_engine import ThoughtThread, ThoughtThreadEngine, TTEContext

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DISSONANCE_THRESHOLD = 0.35   # below this intensity → ignore
CONFIDENCE_PENALTY   = 0.12   # per dissonance event applied to thread
MAX_DISSONANCE_EVENTS = 20    # keep only most recent


# ── Types ──────────────────────────────────────────────────────────────────────

class DissonanceEventType(str, Enum):
    BELIEF_VS_EVIDENCE     = "belief_vs_evidence"
    IDENTITY_VS_ACTION     = "identity_vs_action"
    GOAL_VS_OUTCOME        = "goal_vs_outcome"
    PREDICTION_VS_REALITY  = "prediction_vs_reality"   # Phase 5.3


@dataclass
class DissonanceEvent:
    id:           str
    type:         DissonanceEventType
    thread_id:    Optional[str]   = None
    belief_ref:   Optional[str]   = None
    evidence_ref: Optional[str]   = None
    identity_ref: Optional[str]   = None
    action_ref:   Optional[str]   = None
    predicted:    Optional[Dict]  = None
    actual:       Optional[Dict]  = None
    intensity:    float           = 0.5
    resolved:     bool            = False
    created_at:   float           = field(default_factory=time.time)
    # Phase 5.3 — Recursive Self-Inquiry: the mismatch, made examinable.
    # owner_statement binds the prediction to the persistent self ("I
    # expected...", not "confidence decreased") — the difference the
    # proposal called out between monitoring and genuine self-inquiry.
    owner_statement: Optional[str]        = None
    self_question:   Optional[str]        = None
    hypotheses:      List[str]            = field(default_factory=list)


@dataclass
class CDEContext:
    threads:        List[Any]       = field(default_factory=list)  # ThoughtThread list
    world_model:    Dict[str, Any]  = field(default_factory=dict)
    identity_state: Dict[str, Any]  = field(default_factory=dict)
    recent_actions: List[Dict]      = field(default_factory=list)
    metrics:        Dict[str, float] = field(default_factory=dict)
    beliefs:        List[str]       = field(default_factory=list)
    predictive_mind: Any            = None   # Phase 5.3 — for PREDICTION_VS_REALITY


# ── Engine ────────────────────────────────────────────────────────────────────

class CognitiveDissonanceEngine:
    """
    Detects and resolves cognitive dissonances within the TTE/DTS pipeline.
    Instantiated once and owned by the InternalThoughtLoop.
    """

    def __init__(self) -> None:
        self._events: List[DissonanceEvent] = []
        logger.info("[CDE] CognitiveDissonanceEngine initialised")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, ctx: CDEContext) -> List[DissonanceEvent]:
        """
        Scan for all three dissonance types. Returns new events only.
        """
        new_events: List[DissonanceEvent] = []

        # 1. IDENTITY_VS_ACTION: check each thread's actual behaviour
        new_events.extend(self._detect_identity_vs_action(ctx))

        # 2. GOAL_VS_OUTCOME: check thread predictions vs current_state outcomes
        new_events.extend(self._detect_goal_vs_outcome(ctx))

        # 3. BELIEF_VS_EVIDENCE: world model causal beliefs vs recent events
        new_events.extend(self._detect_belief_vs_evidence(ctx))

        # 4. PREDICTION_VS_REALITY (Phase 5.3): predicted vs actual confidence
        new_events.extend(self._detect_prediction_vs_reality(ctx))

        self._events.extend(new_events)
        if len(self._events) > MAX_DISSONANCE_EVENTS:
            self._events = self._events[-MAX_DISSONANCE_EVENTS:]

        if new_events:
            logger.debug(f"[CDE] Detected {len(new_events)} new dissonance event(s)")

        return new_events

    def apply_to_threads(
        self, events: List[DissonanceEvent], threads: List[Any]
    ) -> List[Any]:
        """
        Apply dissonance effects to affected threads:
        - reduce confidence
        - raise dissonance score
        - block resolution until addressed
        """
        if not events:
            return threads

        affected: Dict[str, DissonanceEvent] = {
            e.thread_id: e for e in events if e.thread_id
        }

        for thread in threads:
            if thread.id in affected:
                ev = affected[thread.id]
                thread.confidence  = max(0.05, thread.confidence  - CONFIDENCE_PENALTY * ev.intensity)
                thread.dissonance  = min(1.0,  thread.dissonance  + ev.intensity * 0.3)
                logger.debug(
                    f"[CDE] Thread {thread.id[:8]} penalised "
                    f"(conf={thread.confidence:.2f}, dissonance={thread.dissonance:.2f})"
                )

        return threads

    def spawn_meta_threads(
        self,
        events: List[DissonanceEvent],
        tte: Any,  # ThoughtThreadEngine
        ctx: Any,  # TTEContext
    ) -> List[Any]:
        """
        For high-intensity dissonances, create meta-threads of self-reflection.
        These are injected into the TTE so the executive loop can act on them.
        """
        meta_threads = []
        for ev in events:
            if ev.intensity < DISSONANCE_THRESHOLD + 0.15:
                continue  # minor dissonance — no meta-thread needed
            goal = (
                f"Reflect on dissonance: {ev.type.value}. "
                f"Belief={ev.belief_ref or 'unknown'}, Action={ev.action_ref or 'unknown'}."
            )
            mt = tte.spawn_thread(ctx, {
                "topic":           f"dissonance:{ev.id[:8]}",
                "goal":            goal,
                "source":          "dissonance",
                "curiosity":       0.1,
                "energy_cost":     8.0,
                "planned_actions": [
                    "identify root cause",
                    "revise belief or behavior",
                    "update self-model",
                ],
            })
            meta_threads.append(mt)
            logger.debug(f"[CDE] Meta-thread spawned for event {ev.id[:8]}: {ev.type.value}")
        return meta_threads

    def resolve(self, events: List[DissonanceEvent], ctx: CDEContext, organism: Any = None) -> None:
        """
        Apply resolution effects:
        - trigger personality evolution queue_experience
        - update world model causal beliefs (cause → effect)
        - mark events resolved
        """
        for ev in events:
            if ev.resolved:
                continue

            # Phase 5.3: genuine self-question + hypotheses, bound to the self
            self._generate_self_question(ev)
            if organism is not None:
                self._generate_competing_hypotheses(ev, organism)
                try:
                    exp_engine = getattr(organism, "_loop", None)
                    exp_engine = getattr(exp_engine, "_auto_experiment", None)
                    if exp_engine and ev.self_question and hasattr(exp_engine, "state"):
                        exp_engine.state.pending_dissonance_questions = getattr(
                            exp_engine.state, "pending_dissonance_questions", []
                        )
                        exp_engine.state.pending_dissonance_questions.append({
                            "question": ev.self_question,
                            "hypotheses": ev.hypotheses,
                            "type": ev.type.value,
                        })
                except Exception as e:
                    logger.debug(f"[CDE] experiment handoff skipped (non-fatal): {e}")

            # Trigger personality evolution — dissonance is a learning signal.
            # Keys must match exactly the entries in EXPERIENCE_PRESSURE dict.
            if organism is not None:
                try:
                    sys = getattr(organism, "ai_system", None) or getattr(organism, "_system", None)
                    if sys:
                        pe = getattr(sys, "personality_evolution", None)
                        if pe:
                            _type_key_map = {
                                "identity_vs_action": "resolved dissonance (identity_vs_action)",
                                "belief_vs_evidence": "resolved dissonance (belief_vs_evidence)",
                                "goal_vs_outcome":    "resolved dissonance (goal_vs_outcome)",
                            }
                            _exp_key = _type_key_map.get(ev.type.value, "contradiction_resolved")
                            pe.queue_experience(_exp_key, intensity=ev.intensity)
                            if ev.intensity > 0.6:
                                if ev.type.value == "identity_vs_action":
                                    pe.queue_experience("identity_dissonance_resolved", intensity=ev.intensity * 0.6)
                                elif ev.type.value == "belief_vs_evidence":
                                    pe.queue_experience("epistemic_dissonance_resolved", intensity=ev.intensity * 0.6)
                except Exception as e:
                    logger.debug(f"[CDE] Personality evolution update failed: {e}")

                # Add causal link to world model
                try:
                    ai_sys = getattr(organism, "ai_system", None) or getattr(organism, "_system", None)
                    wm = getattr(ai_sys, "world_model", None) if ai_sys else None
                    if wm is None:
                        wm = getattr(organism, "world_model", None)
                    if wm and ev.belief_ref and ev.evidence_ref:
                        wm.learn_causal(
                            antecedent=ev.belief_ref,
                            consequent=ev.evidence_ref,
                            confidence=min(0.6, ev.intensity),
                        )
                except Exception as e:
                    logger.debug(f"[CDE] World model causal update failed: {e}")

            ev.resolved = True

        logger.debug(f"[CDE] Resolved {sum(1 for e in events if e.resolved)} events")

    def step(
        self,
        threads: List[Any],
        tte: Any,
        ctx: CDEContext,
        tte_ctx: Any,
        organism: Any = None,
    ) -> List[Any]:
        """
        Main tick — called after TTE.step() in the slow cycle.
        Returns the updated thread list.
        """
        events = self.detect(ctx)
        threads = self.apply_to_threads(events, threads)
        self.spawn_meta_threads(events, tte, tte_ctx)

        # Auto-resolve low-intensity events immediately
        auto_resolve = [e for e in events if e.intensity < DISSONANCE_THRESHOLD + 0.1]
        self.resolve(auto_resolve, ctx, organism)

        return threads

    # ── Private detection helpers ─────────────────────────────────────────────

    def _detect_identity_vs_action(self, ctx: CDEContext) -> List[DissonanceEvent]:
        """Check if any belief in identity_state is contradicted by recent actions."""
        events = []
        beliefs = ctx.beliefs or ctx.identity_state.get("beliefs", [])
        recent  = ctx.recent_actions

        for belief in beliefs[:10]:
            bl = belief.lower()

            if "curious" in bl:
                # Curiosity claim: check if exploration actions happened
                explored = any(
                    "explore" in str(a).lower() or "research" in str(a).lower()
                    for a in recent[-5:]
                )
                if not explored and recent:
                    events.append(DissonanceEvent(
                        id          = uuid.uuid4().hex[:12],
                        type        = DissonanceEventType.IDENTITY_VS_ACTION,
                        belief_ref  = belief,
                        action_ref  = "no exploration action taken",
                        intensity   = 0.45,
                    ))

            elif "expressive" in bl or "expressive" in bl:
                communicated = any(
                    "express" in str(a).lower() or "respond" in str(a).lower()
                    for a in recent[-5:]
                )
                if not communicated and recent:
                    events.append(DissonanceEvent(
                        id          = uuid.uuid4().hex[:12],
                        type        = DissonanceEventType.IDENTITY_VS_ACTION,
                        belief_ref  = belief,
                        action_ref  = "no expression action taken",
                        intensity   = 0.35,
                    ))

        return events

    def _detect_goal_vs_outcome(self, ctx: CDEContext) -> List[DissonanceEvent]:
        """Check if any thread's predicted outcome differs from current_state."""
        events = []
        for thread in ctx.threads:
            predicted = thread.current_state.get("predicted_outcome")
            actual    = thread.current_state.get("actual_outcome")
            if predicted and actual and predicted != actual:
                events.append(DissonanceEvent(
                    id          = uuid.uuid4().hex[:12],
                    type        = DissonanceEventType.GOAL_VS_OUTCOME,
                    thread_id   = thread.id,
                    predicted   = {"value": predicted},
                    actual      = {"value": actual},
                    intensity   = min(0.9, thread.dissonance + 0.3),
                ))
        return events

    def _detect_belief_vs_evidence(self, ctx: CDEContext) -> List[DissonanceEvent]:
        """Check if causal beliefs in the world model are contradicted by recent evidence."""
        events = []
        causal = ctx.world_model.get("causal_beliefs", [])
        recent = ctx.recent_actions

        for belief in causal[:15]:
            ant = getattr(belief, "antecedent", None) or belief.get("antecedent", "")
            con = getattr(belief, "consequent", None) or belief.get("consequent", "")
            conf= getattr(belief, "confidence", 0.5) or belief.get("confidence", 0.5)

            if not ant or not con:
                continue

            # Low confidence causal belief — flag for review
            if conf < 0.25:
                events.append(DissonanceEvent(
                    id           = uuid.uuid4().hex[:12],
                    type         = DissonanceEventType.BELIEF_VS_EVIDENCE,
                    belief_ref   = ant,
                    evidence_ref = con,
                    intensity    = max(0.1, 0.5 - conf),
                ))
                continue

            # Recent-evidence contradiction: the antecedent condition showed
            # up in recent activity but the belief's expected consequent
            # didn't follow — the belief predicts X→Y, X happened, Y didn't.
            recent_text = [str(a).lower() for a in recent[-8:]]
            ant_seen = any(str(ant).lower() in t for t in recent_text)
            con_seen = any(str(con).lower() in t for t in recent_text)
            if ant_seen and not con_seen and recent_text:
                events.append(DissonanceEvent(
                    id           = uuid.uuid4().hex[:12],
                    type         = DissonanceEventType.BELIEF_VS_EVIDENCE,
                    belief_ref   = ant,
                    evidence_ref = f"expected '{con}' after '{ant}' in recent activity, didn't observe it",
                    intensity    = max(0.1, min(0.6, 0.4 * conf)),
                ))
        return events[:5]  # cap to avoid event flood

    def _detect_prediction_vs_reality(self, ctx: "CDEContext") -> List[DissonanceEvent]:
        """
        Phase 5.3 — the event type that was missing: 'expected confidence
        0.82, actual 0.61'. Reads PredictiveMind's real _results history
        (Prediction.confidence vs PredictionResult.error_level) rather
        than inventing a new tracking mechanism.
        """
        events: List[DissonanceEvent] = []
        pm = getattr(ctx, "predictive_mind", None)
        results = list(getattr(pm, "_results", []))[-5:] if pm else []
        for r in results:
            pred_conf = getattr(getattr(r, "prediction", None), "confidence", None)
            err = getattr(r, "error_level", None)
            if pred_conf is None or err is None:
                continue
            realized = 1.0 - err   # actual outcome quality, in hindsight
            gap = pred_conf - realized
            if gap > 0.20:   # confidently expected better than what happened
                events.append(DissonanceEvent(
                    id        = uuid.uuid4().hex[:12],
                    type      = DissonanceEventType.PREDICTION_VS_REALITY,
                    predicted = {"confidence": round(pred_conf, 3)},
                    actual    = {"confidence": round(realized, 3)},
                    intensity = min(0.9, gap),
                ))
        return events[:3]

    # ── Phase 5.3: self-question / ownership / competing hypotheses ────────

    def _generate_self_question(self, event: DissonanceEvent) -> None:
        """
        Binds the mismatch to the persistent self ('I expected...') rather
        than leaving it as an impersonal monitoring statement ('confidence
        decreased'). Tailored per event type, not a generic template.
        """
        t = event.type
        if t == DissonanceEventType.PREDICTION_VS_REALITY:
            pc = (event.predicted or {}).get("confidence")
            ac = (event.actual or {}).get("confidence")
            event.owner_statement = f"I expected to be confident here (~{pc:.2f}), but it didn't hold up (~{ac:.2f})."
            event.self_question = "Why did my confidence fall short here — what did I misjudge?"
        elif t == DissonanceEventType.BELIEF_VS_EVIDENCE:
            event.owner_statement = f"I believed '{event.belief_ref}', but recent evidence didn't back it up."
            event.self_question = f"Why do I still hold '{event.belief_ref}' when the evidence is thinning?"
        elif t == DissonanceEventType.IDENTITY_VS_ACTION:
            event.owner_statement = f"I think of myself as someone who '{event.belief_ref}', but I didn't act like it."
            event.self_question = f"Why did I act against '{event.belief_ref}' just now?"
        elif t == DissonanceEventType.GOAL_VS_OUTCOME:
            event.owner_statement = "I was pursuing this goal expecting a different result than what happened."
            event.self_question = "Why is this goal not producing what I expected — is the goal or my approach wrong?"

    def _generate_competing_hypotheses(self, event: DissonanceEvent, organism: Any) -> None:
        """
        Scoped, honest version of 'competing hypotheses': two candidates
        pulled from distinct real data sources rather than invented —
        (1) a causal-belief explanation from the world model, (2) a
        skill-rustiness explanation from the skill registry. Not creative
        hypothesis generation; a bounded, evidence-grounded starting set.
        """
        hyps = []
        topic = (event.belief_ref or "").strip()
        try:
            wm = getattr(organism, "world_model", None)
            if wm and topic and hasattr(wm, "causal_beliefs"):
                # causal_beliefs is a List[CausalBelief] with antecedent/consequent,
                # not a dict and not a `.cause` field — v87 had both wrong.
                for cb in list(getattr(wm, "causal_beliefs", []))[:20]:
                    antecedent = str(getattr(cb, "antecedent", ""))
                    if topic and topic.lower() in antecedent.lower():
                        hyps.append(f"Perhaps this connects to what I already believe about '{antecedent}' — that belief may itself be outdated.")
                        break
        except Exception:
            pass
        try:
            sr = getattr(organism, "skill_registry", None)
            if sr and topic:
                relevant = sr.relevant_skills(topic, n=1) if hasattr(sr, "relevant_skills") else []
                for sk in relevant:
                    if hasattr(sk, "is_rusty") and sk.is_rusty():
                        hyps.append(f"Perhaps my grasp of '{topic}' has gone rusty from disuse, not that my reasoning was wrong.")
        except Exception:
            pass
        if not hyps:
            hyps.append("Perhaps this was a one-off — worth watching before concluding anything changed.")
        event.hypotheses = hyps[:2]

    def pending_count(self) -> int:
        return sum(1 for e in self._events if not e.resolved)
