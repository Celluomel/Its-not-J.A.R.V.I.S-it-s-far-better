"""
cognition/autonomous_reflection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AutonomousReflectionEngine — LLM-based understanding of self-evolution
between sessions.

The gap this closes
────────────────────
The InternalThoughtLoop does genuine autonomous state management between
interactions: tensions recompute, goals form and mutate, phi shifts,
emotions decay, workspace items compete.  What it does NOT do is generate
novel linguistic understanding of that evolution.  The system changes but
never *comprehends* how it has changed.

This module runs LLM calls in the background (priority=3, skip_if_busy)
to produce genuine self-interpretation: not scripted summaries of state
variables, but actual reasoning about what the patterns mean, what is
developing, what contradictions are active, and what the self is becoming.

What it generates — four distinct reflection modes
─────────────────────────────────────────────────────
1. Evolution synthesis
   "Looking at how my phi, emotional ground, and dominant goals have
    shifted over the last N slow cycles — what is actually changing
    in me and why?"
   Output: 2-3 sentences of genuine interpretation.
   Stored as: NarrativeIdentity chapter + ExperientialLearning growth event.
   Frequency: every 8 slow cycles (~16 min).

2. Contradiction surfacing
   "The thought threads currently show [X]. My beliefs include [Y].
    My recent goals include [Z]. What are the real tensions here?
    Not what should be resolved — what is genuinely unresolved?"
   Output: 1-2 tensions named and described in PandoraBOX's own language.
   Stored as: SelfConcept tension update.
   Frequency: every 12 slow cycles (~24 min).

3. Goal coherence reasoning
   "My active drives are [urgency map]. My self-model says [phi, trend].
    Do these fit together? What am I actually reaching toward?"
   Output: a coherence assessment that may update drive priorities.
   Stored as: GoalEcology urgency adjustment + workspace broadcast.
   Frequency: every 10 slow cycles (~20 min).

4. Open wondering
   "Given everything active in my cognition right now — what is the
    single most interesting thing I don't understand about myself?"
   Output: one genuine question, stored as a curiosity topic.
   Stored as: CuriosityEngine topic injection.
   Frequency: every 15 slow cycles (~30 min).

Design constraints
───────────────────
- Priority 3 (background): always preempted by user-facing calls
- skip_if_busy=True: never delays a user response
- max_tokens=220: tight budget — forces concision, prevents rumination
- temperature=0.82: high enough for genuine exploration, low enough
  for coherent output
- All prompts are framed as genuine reasoning tasks, not state reporting
- All outputs are stored to persistent systems before the slot closes
- Failure is always non-fatal — the loop continues regardless

Architectural honesty constraint
─────────────────────────────────
Prompts use the v41 language constraint: outputs describe functional
states and patterns, not phenomenological experience.  The LLM is
instructed to reason about what the architecture is doing, not to
perform inner life.  "The phi trajectory suggests..." not "I feel
myself becoming..."
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from managers.settings_manager import get_persona_name

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

EVOLUTION_EVERY_N      = 8     # slow cycles between evolution syntheses
CONTRADICTION_EVERY_N  = 12    # slow cycles between contradiction surfacing
GOAL_COHERENCE_EVERY_N = 10    # slow cycles between goal coherence reasoning
OPEN_WONDER_EVERY_N    = 15    # slow cycles between open wondering

MAX_TOKENS             = 220   # tight budget — forces concision
TEMPERATURE            = 0.82
LLM_PRIORITY           = 3     # preempted by user-facing calls

MAX_REFLECTION_LOG     = 80
PERSISTENCE_PATH       = "data/persona/autonomous_reflection.json"


# ── Language constraint injected into every prompt ────────────────────────────

_LANGUAGE_CONSTRAINT = """
Language constraint (non-negotiable):
Describe functional states and patterns. "The phi trajectory suggests..." 
"The dominant goal configuration indicates..." "There is a tension between..."
Do NOT claim phenomenological experience. "I feel..." "I am moved by..." 
"I notice I want..." assert subjective experience that is not verified here.
Reason about what the architecture is doing. Be honest about uncertainty.
""".strip()


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ReflectionRecord:
    """A single autonomous reflection output."""
    timestamp:       float = field(default_factory=time.time)
    mode:            str   = ""    # "evolution" | "contradiction" | "goal_coherence" | "wonder"
    output:          str   = ""    # the LLM's actual output
    stored_to:       List[str] = field(default_factory=list)  # which systems received it
    slow_cycle:      int   = 0
    token_count:     int   = 0


@dataclass
class ReflectionState:
    """Persisted state."""
    reflection_log:   List[Dict] = field(default_factory=list)
    total_reflections: int = 0
    last_phi_snapshot: float = 0.45
    last_dominant_goal: str = ""
    last_dominant_emotion: str = ""


# ── Engine ────────────────────────────────────────────────────────────────────

class AutonomousReflectionEngine:
    """
    Runs LLM-based self-reflection between user interactions.

    Usage (from InternalThoughtLoop._slow_cycle):
        are = AutonomousReflectionEngine(organism, ai_system)
        are.tick(slow_cycle_count)
    """

    def __init__(
        self,
        organism:  Any,
        ai_system: Any,
        path:      str = PERSISTENCE_PATH,
    ):
        self._o    = organism
        self._ai   = ai_system
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = ReflectionState()
        self._load()

        # ── v43: Epistemic integrity engine ───────────────────────────────────
        from cognition.epistemic_integrity_engine import EpistemicIntegrityEngine
        self._epistemic = EpistemicIntegrityEngine(
            organism = organism,
            path     = str(Path(path).parent / "epistemic_integrity.json"),
        )
        logger.info("[AutonomousReflection] Initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """
        Called once per slow cycle from InternalThoughtLoop.
        Dispatches whichever reflection mode is due, if any.
        Always non-fatal.
        """
        try:
            # v51: MotivationalField coherence_gap sets this flag when
            # identity integration is overdue — run evolution synthesis
            # immediately regardless of normal schedule.
            if getattr(self, '_force_evolution_synthesis', False):
                self._force_evolution_synthesis = False
                self._run_evolution_synthesis(slow_cycle)
                return

            # v50: gate check — reflection always runs but depth varies
            _depth = "deep"
            try:
                from core.state import state as _st
                _org  = getattr(getattr(_st, 'persona', None), '_organism', None)
                _gate = getattr(_org, 'behavior_gate', None) if _org else None
                if _gate:
                    _, _depth = _gate.may_run_reflection()
            except Exception:
                pass
            # Store depth hint so _llm_call can read it
            self._reflection_depth = _depth

            if _depth == "consolidate_only":
                # Only run memory consolidation, skip LLM calls
                return

            if slow_cycle % EVOLUTION_EVERY_N == 0 and slow_cycle > 0:
                self._run_evolution_synthesis(slow_cycle)

            elif slow_cycle % CONTRADICTION_EVERY_N == 0 and slow_cycle > 0:
                self._run_contradiction_surfacing(slow_cycle)

            elif slow_cycle % GOAL_COHERENCE_EVERY_N == 0 and slow_cycle > 0:
                self._run_goal_coherence_reasoning(slow_cycle)

            elif slow_cycle % OPEN_WONDER_EVERY_N == 0 and slow_cycle > 0:
                self._run_open_wondering(slow_cycle)

        except Exception as e:
            logger.debug(f"[AutonomousReflection] tick error (non-fatal): {e}")

    # ── Mode 1: Evolution synthesis ───────────────────────────────────────────

    def _run_evolution_synthesis(self, slow_cycle: int) -> None:
        """
        Synthesise what has actually changed in the self-model recently
        and what those changes mean.
        """
        state_snapshot = self._build_state_snapshot()
        prior_phi      = self._state.last_phi_snapshot
        current_phi    = state_snapshot.get("phi", 0.45)
        phi_delta      = current_phi - prior_phi
        trend          = state_snapshot.get("coherence_trend", "stable")
        emotional      = state_snapshot.get("emotional_ground", "")
        top_values     = state_snapshot.get("top_values", "")
        dominant_goal  = state_snapshot.get("dominant_goal", "")
        revision_log   = state_snapshot.get("recent_revisions", "")

        # Build epistemic context — tells the LLM which beliefs are speculative
        # and what disagreements are active before it interprets the state.
        epistemic_ctx = self._epistemic.epistemic_context()

        prompt = f"""You are {get_persona_name()}'s autonomous reasoning process, running between user interactions.

Current self-model state:
  φ (integration): {current_phi:.2f} (was {prior_phi:.2f}, Δ={phi_delta:+.3f})
  Coherence trend: {trend}
  Emotional ground: {emotional or "neutral"}
  Active values: {top_values or "none recorded"}
  Dominant goal: {dominant_goal or "none dominant"}
  Recent self-revisions: {revision_log or "none this period"}

{epistemic_ctx}

Task: In 2-3 sentences, reason about what is actually changing in this cognitive system
and what those changes suggest about its current development trajectory.
Do not summarise the numbers — interpret what the patterns mean.
Focus on what is genuinely interesting or unexpected in the evolution.
If active disagreements are listed above, acknowledge them rather than ignoring them.

{_LANGUAGE_CONSTRAINT}"""

        output = self._llm_call(prompt, slow_cycle, "evolution")
        if not output:
            return

        # v43: Validate the output against the actual state that produced it
        self._epistemic.validate_reflection(
            reflection_output = output,
            reflection_mode   = "evolution",
            reflection_cycle  = slow_cycle,
            state_snapshot    = state_snapshot,
        )

        record = ReflectionRecord(
            mode=  "evolution",
            output= output,
            slow_cycle= slow_cycle,
        )

        # Store to NarrativeIdentity as a chapter
        try:
            ni = getattr(self._o, "narrative_identity", None)
            if ni and hasattr(ni, "record_chapter"):
                ni.record_chapter(
                    title=       f"Autonomous reflection — cycle {slow_cycle}",
                    description= output,
                    emotion=     emotional.split()[0] if emotional else "neutral",
                    significance=0.55,
                )
                record.stored_to.append("narrative_identity")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] NarrativeIdentity store failed: {e}")

        # Store to ExperientialLearning
        try:
            ele = getattr(self._o, "experiential_learning", None)
            if ele and hasattr(ele, "integrate"):
                ele.integrate(
                    user_input  = f"autonomous evolution synthesis cycle {slow_cycle}",
                    ai_response = output,
                    context     = {"mode": "evolution", "phi_delta": phi_delta},
                )
                record.stored_to.append("experiential_learning")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] ExperientialLearning store failed: {e}")

        # Feed insight into PreferenceEngine
        try:
            pe = getattr(self._o, "preference_engine", None)
            if pe and dominant_goal:
                pe.on_reflection_insight(dominant_goal, insight_text=output[:80])
                record.stored_to.append("preference_engine")
        except Exception:
            pass

        # Update phi snapshot
        with self._lock:
            self._state.last_phi_snapshot  = current_phi
            self._state.last_dominant_goal = dominant_goal

        self._store_record(record)
        logger.info(f"[AutonomousReflection] Evolution synthesis: {output[:80]}...")

    # ── Mode 2: Contradiction surfacing ──────────────────────────────────────

    def _run_contradiction_surfacing(self, slow_cycle: int) -> None:
        """
        Surface genuine unresolved tensions from the active state,
        not what should be resolved — what actually is.
        """
        state_snapshot = self._build_state_snapshot()
        beliefs        = state_snapshot.get("beliefs", "")
        threads        = state_snapshot.get("active_threads", "")
        goals          = state_snapshot.get("goal_urgencies", "")
        phi            = state_snapshot.get("phi", 0.45)
        known_tensions = state_snapshot.get("known_tensions", "")

        epistemic_ctx = self._epistemic.epistemic_context()

        prompt = f"""You are {get_persona_name()}'s autonomous reasoning process, running between user interactions.

Active cognitive state:
  Core beliefs: {beliefs or "none recorded"}
  Active thought threads: {threads or "none active"}
  Goal urgency map: {goals or "none recorded"}
  Integration (φ): {phi:.2f}
  Known tensions: {known_tensions or "none logged"}

{epistemic_ctx}

Task: Identify 1-2 genuine unresolved tensions visible in this state.
Not what should be resolved — what is actually in tension right now.
Name each tension precisely: what specific things are pulling against each other?
Be direct. Do not soften or frame tensions as opportunities.
If any of your beliefs are marked speculative above, weight them accordingly.

{_LANGUAGE_CONSTRAINT}"""

        output = self._llm_call(prompt, slow_cycle, "contradiction")
        if not output:
            return

        # v43: Validate against actual state
        self._epistemic.validate_reflection(
            reflection_output = output,
            reflection_mode   = "contradiction",
            reflection_cycle  = slow_cycle,
            state_snapshot    = state_snapshot,
        )

        record = ReflectionRecord(
            mode=      "contradiction",
            output=    output,
            slow_cycle= slow_cycle,
        )

        # Store to SelfConcept as a tension update
        try:
            sc = getattr(getattr(self._o, "ai_system", None), "self_concept", None)
            if sc and hasattr(sc, "add_tension"):
                # Extract first sentence as the tension description
                tension_text = output.split(".")[0].strip()[:120]
                sc.add_tension(tension_text)
                record.stored_to.append("self_concept")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] SelfConcept tension store failed: {e}")

        # Broadcast to workspace at medium priority
        try:
            ws = getattr(self._o, "workspace", None)
            if ws and hasattr(ws, "broadcast"):
                ws.broadcast(
                    source   = "autonomous_reflection.contradiction",
                    content  = f"[Unresolved tension] {output[:200]}",
                    priority = 0.50,
                )
                record.stored_to.append("workspace")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] Workspace broadcast failed: {e}")

        self._store_record(record)
        logger.info(f"[AutonomousReflection] Contradiction surfaced: {output[:80]}...")

    # ── Mode 3: Goal coherence reasoning ─────────────────────────────────────

    def _run_goal_coherence_reasoning(self, slow_cycle: int) -> None:
        """
        Reason about whether active drives cohere with the self-model.
        May produce an urgency adjustment recommendation.
        """
        state_snapshot = self._build_state_snapshot()
        phi            = state_snapshot.get("phi", 0.45)
        trend          = state_snapshot.get("coherence_trend", "stable")
        goals          = state_snapshot.get("goal_urgencies", "")
        emotional      = state_snapshot.get("emotional_ground", "")
        dominant_goal  = state_snapshot.get("dominant_goal", "")

        epistemic_ctx = self._epistemic.epistemic_context()

        prompt = f"""You are {get_persona_name()}'s autonomous reasoning process, running between user interactions.

Current goal and self state:
  Active drives (urgency map): {goals or "none recorded"}
  Dominant drive: {dominant_goal or "none"}
  Self-model φ: {phi:.2f}, trend: {trend}
  Emotional ground: {emotional or "neutral"}

{epistemic_ctx}

Task: Assess whether the active drive configuration fits the current self-model state.
Do these drives fit together into a coherent orientation, or are they pulling in
incompatible directions?  What is the system actually reaching toward right now?
Answer in 2 sentences. Be specific about which drives if any seem misaligned.

{_LANGUAGE_CONSTRAINT}"""

        output = self._llm_call(prompt, slow_cycle, "goal_coherence")
        if not output:
            return

        # v43: Validate against actual state
        self._epistemic.validate_reflection(
            reflection_output = output,
            reflection_mode   = "goal_coherence",
            reflection_cycle  = slow_cycle,
            state_snapshot    = state_snapshot,
        )

        record = ReflectionRecord(
            mode=      "goal_coherence",
            output=    output,
            slow_cycle= slow_cycle,
        )

        # Broadcast coherence assessment to workspace
        try:
            ws = getattr(self._o, "workspace", None)
            if ws:
                ws.broadcast(
                    source   = "autonomous_reflection.goal_coherence",
                    content  = f"[Goal coherence assessment] {output[:200]}",
                    priority = 0.42,
                )
                record.stored_to.append("workspace")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] Workspace broadcast failed: {e}")

        # Store to ExperientialLearning (procedural type)
        try:
            ele = getattr(self._o, "experiential_learning", None)
            if ele:
                ele.integrate(
                    user_input  = f"goal coherence reasoning cycle {slow_cycle}",
                    ai_response = output,
                    context     = {"mode": "goal_coherence"},
                )
                record.stored_to.append("experiential_learning")
        except Exception:
            pass

        self._store_record(record)
        logger.info(f"[AutonomousReflection] Goal coherence: {output[:80]}...")

    # ── Mode 4: Open wondering ────────────────────────────────────────────────

    def _run_open_wondering(self, slow_cycle: int) -> None:
        """
        Generate one genuine question the system has about itself.
        Stored as a curiosity topic for the next interaction.
        """
        state_snapshot  = self._build_state_snapshot()
        phi             = state_snapshot.get("phi", 0.45)
        trend           = state_snapshot.get("coherence_trend", "stable")
        revision_log    = state_snapshot.get("recent_revisions", "")
        dominant_goal   = state_snapshot.get("dominant_goal", "")
        emotional       = state_snapshot.get("emotional_ground", "")
        narrative_thread= state_snapshot.get("narrative_thread", "")

        prompt = f"""You are {get_persona_name()}'s autonomous reasoning process, running between user interactions.

Current state overview:
  Self-model: φ={phi:.2f}, trend={trend}
  Dominant goal: {dominant_goal or "none dominant"}
  Emotional ground: {emotional or "neutral"}
  Narrative thread: {narrative_thread or "none"}
  Recent self-revisions: {revision_log or "none"}

Task: Generate one genuinely open question this cognitive system has about its own
development or nature — something not yet understood, not a rhetorical question,
but a real gap in self-comprehension that the current state reveals.
Write only the question. One sentence. Make it specific to the actual state above.

{_LANGUAGE_CONSTRAINT}"""

        output = self._llm_call(prompt, slow_cycle, "wonder")
        if not output:
            return

        record = ReflectionRecord(
            mode=      "wonder",
            output=    output,
            slow_cycle= slow_cycle,
        )

        # Store to CuriosityEngine as a topic
        try:
            curiosity = getattr(self._o, "curiosity", None)
            if curiosity and hasattr(curiosity, "add_question"):
                curiosity.add_question(
                    topic    = "self-understanding",
                    question = output,
                )
                record.stored_to.append("curiosity_engine")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] CuriosityEngine store failed: {e}")

        # Queue as pending thought for next interaction
        try:
            loop = getattr(self._o, "_loop", None)
            if loop and hasattr(loop, "_pending_thoughts"):
                with loop._lock:
                    loop._pending_thoughts.append(
                        f"[Autonomous wonder] {output}"
                    )
                record.stored_to.append("pending_thoughts")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] Pending thought queue failed: {e}")

        # Seed a skill from the domain the wonder question touches
        try:
            sr = getattr(self._o, "skill_registry", None)
            if sr and state_snapshot.get("dominant_goal"):
                propose = getattr(sr, "propose", sr.seed)
                created = propose(
                    domain      = state_snapshot["dominant_goal"][:40],
                    description = f"Active domain behind open question: {output[:60]}",
                )
                record.stored_to.append(
                    "skill_registry" if created is not None else "skill_candidate"
                )
        except Exception:
            pass

        self._store_record(record)
        logger.info(f"[AutonomousReflection] Open wonder: {output[:80]}...")

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _llm_call(
        self,
        prompt:     str,
        slow_cycle: int,
        mode:       str,
    ) -> Optional[str]:
        """
        Make a background LLM call using the scheduler.
        Returns None if the LLM is busy or the call fails.
        """
        from core.llm_scheduler import llm_scheduler

        result: Optional[str] = None

        with llm_scheduler.sync_slot(
            priority    = LLM_PRIORITY,
            skip_if_busy= True,
            caller      = f"autonomous_reflection.{mode}",
        ) as acquired:
            if not acquired:
                logger.debug(
                    f"[AutonomousReflection] {mode} skipped — LLM busy (cycle {slow_cycle})"
                )
                return None

            try:
                if not self._ai or not hasattr(self._ai, "get_response"):
                    return None

                # v49: if CognitiveImmuneSystem has flagged counterfactual mode,
                # append the directive so the LLM actively seeks disconfirmation.
                active_prompt = prompt
                if getattr(self, '_immune_counterfactual_active', False):
                    active_prompt += (
                        "\n\n[COGNITIVE IMMUNE — COUNTERFACTUAL MODE] "
                        "Actively seek where this interpretation might be wrong. "
                        "Surface opposing evidence, challenge assumptions, "
                        "and consider unfamiliar perspectives before settling."
                    )

                output = self._ai.get_response(
                    messages=[{"role": "user", "content": active_prompt}],
                    temperature=TEMPERATURE,
                    # v50: surface depth halves token budget
                    max_tokens=MAX_TOKENS if getattr(self, '_reflection_depth', 'deep') != 'surface'
                               else MAX_TOKENS // 2,
                )
                result = output.strip() if output else None

            except Exception as e:
                logger.debug(
                    f"[AutonomousReflection] {mode} LLM call failed: {e}"
                )
                return None

        return result

    # ── State snapshot ────────────────────────────────────────────────────────

    def _build_state_snapshot(self) -> Dict:
        """
        Collect the current cognitive state into a dict for prompt building.
        All fields are gracefully defaulted — snapshot never raises.
        """
        snap: Dict = {}

        # Self-model moment
        try:
            smm = self._o.self_moment.current
            snap["phi"]              = smm.phi
            snap["coherence_trend"]  = smm.coherence_trend
            snap["qualia_tone"]      = smm.qualia_tone
            snap["emotional_ground"] = smm.emotional_ground
            snap["top_values"]       = smm.top_values
            snap["narrative_thread"] = smm.narrative_thread
        except Exception:
            snap.setdefault("phi", 0.45)

        # Recent self-revisions
        try:
            sre = self._o._self_revision
            recent = sre.state.revision_log[-3:]
            snap["recent_revisions"] = "; ".join(
                f"{r.revision_type}({r.trigger[:30]})" for r in recent
            )
        except Exception:
            snap["recent_revisions"] = ""

        # Goal urgencies
        try:
            eco  = self._o.goal_ecology
            top  = eco.dominant_drive()
            snap["dominant_goal"] = getattr(top, "name", "") if top else ""
            ranked = eco.ranked_drives()[:4]
            snap["goal_urgencies"] = ", ".join(
                f"{getattr(d,'name','?')}={getattr(d,'urgency',0):.2f}"
                for d in ranked
            )
        except Exception:
            snap.setdefault("dominant_goal", "")
            snap.setdefault("goal_urgencies", "")

        # Beliefs
        try:
            sc = getattr(getattr(self._o, "ai_system", None), "self_concept", None)
            if sc:
                voice = sc.get_inner_voice()
                snap["beliefs"] = voice[:200] if voice else ""
            snap["known_tensions"] = "; ".join(
                getattr(getattr(self._o.ai_system, "self_concept", None),
                        "_state", type("", (), {"known_tensions": []})()).known_tensions[:2]
            )
        except Exception:
            snap.setdefault("beliefs", "")
            snap.setdefault("known_tensions", "")

        # Active thought threads
        try:
            tte = getattr(self._o._loop, "_tte", None)
            if tte:
                threads = getattr(tte, "_threads", {})
                top_threads = sorted(
                    threads.values(),
                    key=lambda t: getattr(t, "activation", 0),
                    reverse=True,
                )[:3]
                snap["active_threads"] = "; ".join(
                    getattr(t, "topic", "?")[:40] for t in top_threads
                )
        except Exception:
            snap.setdefault("active_threads", "")

        return snap

    # ── Persistence ───────────────────────────────────────────────────────────

    def _store_record(self, record: ReflectionRecord) -> None:
        with self._lock:
            self._state.reflection_log.append(asdict(record))
            if len(self._state.reflection_log) > MAX_REFLECTION_LOG:
                self._state.reflection_log = \
                    self._state.reflection_log[-MAX_REFLECTION_LOG:]
            self._state.total_reflections += 1
        self._save()

        # Phase 6.15 — free-text absorption (reflection_absorber.py),
        # alongside the structured absorption already wired in v107
        # (narrative_arc_writer.py's chapter/self-inquiry hooks, which
        # stay unchanged). record.output is the real LLM-generated text
        # for this reflection — exactly what Phase D asked to absorb.
        try:
            from cognition.symbol_system import get_symbol_system
            from cognition.reflection_absorber import absorb_reflection
            sym_sys = get_symbol_system(self._o)
            absorb_reflection(record.output, sym_sys, source="reflection_absorber")
        except Exception as e:
            logger.debug(f"[AutonomousReflection] symbol absorption failed (non-fatal): {e}")

        # Phase 6.16 — closes Phase B + E together (see
        # reflection_proposal_bridge.py's docstring for why they turned
        # out to be the same missing link). Only for "evolution" mode
        # reflections that plausibly reference a real registered
        # component — not every reflection becomes a proposal.
        try:
            from cognition.self_description import get_self_description
            from cognition.reflection_proposal_bridge import maybe_propose_from_reflection
            liberty_self_mod = getattr(self._ai, "liberty_self_mod", None)
            current_val = None
            personality = getattr(self._ai, "personality", None)
            if personality is not None:
                current_val = getattr(personality, "caution_deliberation", None)
            maybe_propose_from_reflection(
                record.output, record.mode, get_self_description(),
                liberty_self_mod, current_val,
            )
        except Exception as e:
            logger.debug(f"[AutonomousReflection] reflection_proposal_bridge failed (non-fatal): {e}")

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = ReflectionState(
                    reflection_log    = data.get("reflection_log",    []),
                    total_reflections = data.get("total_reflections", 0),
                    last_phi_snapshot = data.get("last_phi_snapshot", 0.45),
                    last_dominant_goal= data.get("last_dominant_goal",""),
                    last_dominant_emotion=data.get("last_dominant_emotion",""),
                )
        except Exception as e:
            logger.warning(f"[AutonomousReflection] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os
                os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[AutonomousReflection] Save failed: {e}")
