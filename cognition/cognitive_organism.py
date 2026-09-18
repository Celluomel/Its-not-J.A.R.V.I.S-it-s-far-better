"""
Cognitive Organism
==================
The central integration layer that transforms LuminaCore from a sophisticated
prompt-composer into a self-regulating cognitive organism.

This module wraps the existing LuminaCore instance and adds:
  - CognitiveEnergy        → energy-modulated reasoning depth
  - AttentionSystem        → directed focus allocation
  - CuriosityEngine        → intellectual interest tracking
  - TensionEngine          → internal pressure computation
  - GoalEcology            → competing drive management
  - Arbitration            → drive selection before LLM call
  - Homeostasis            → psychological equilibrium monitoring
  - MetaCognition          → self-observation and quality evaluation
  - InternalThoughtLoop    → background cognition between interactions

v35 — Sentience Enhancement Layer (Flux self-improvement directives):
  - EthicalReasoningEngine   → living moral framework, dilemma navigation
  - ActiveEmpathy            → proactive caring, relational investment
  - CreativeDivergenceEngine → non-obvious angles, analogical mapping
  - CommunicationCalibrator  → clarity/concision self-monitoring
  - ExperientialLearningEngine → schema-based experiential growth
  - SocietalAwarenessEngine  → broader AI-in-society perspective

Integration model:
  CognitiveOrganism sits BETWEEN the application layer (app.py / managers)
  and LuminaCore. The application calls organism.respond() instead of
  calling ai_system.respond() directly.

  Every interaction runs through this pipeline:
    1. pre_interaction()   — regen energy, pop background thoughts
    2. update_cycle()      — compute tensions, update ecology, attention
    3. arbitrate()         — select dominant drive + reasoning style
    4. augment_prompt()    — inject all cognitive context into system prompt
    5. LuminaCore.respond()
    6. post_interaction()  — evaluate, update conditioning, homeostasis

Usage
-----
# In app.py or the relevant manager, replace:
#   response = ai_system.respond(user_input, ...)
# With:
#   response = organism.respond(user_input, ...)

from cognition.cognitive_organism import CognitiveOrganism

organism = CognitiveOrganism(ai_system)
organism.start()   # launches background loop

response = organism.respond(user_input, user_id="alice")
"""

import logging
import time
from pathlib import Path
from typing import Optional, Any, Dict

from physiology.cognitive_energy import CognitiveEnergy
from cognition.attention_system import AttentionSystem
from cognition.curiosity_engine import CuriosityEngine
from cognition.tension_engine import TensionEngine
from cognition.arbitration import Arbitration
from cognition.meta_cognition import MetaCognition
from goals.goal_ecology import GoalEcology
from psychology.homeostasis import Homeostasis
from core.internal_loop import InternalThoughtLoop

# ── Autonomous architecture modules (v2) ─────────────────────────────────────
from cognition.global_workspace   import GlobalWorkspace
from cognition.self_model         import SelfModel
from cognition.attractor_system   import CognitiveAttractorSystem
from cognition.priority_engine    import PriorityEngine

# ── Cognitive organism v3 — DeepSeek recommended additions ────────────────────
from cognition.thought_stream     import ThoughtStream
from cognition.memory_intrusion   import MemoryIntrusionSystem
from cognition.world_model        import WorldModel
from cognition.cognitive_validator import CognitiveValidator, CognitiveArchitectureMonitor
from core.cognitive_event_bus    import CognitiveEventBus, wire_cognitive_bus
from cognition.predictive_mind    import PredictiveMind
from cognition.narrative_identity import NarrativeIdentity
from cognition.pressure_system    import PressureSystem
from core.llm_scheduler          import llm_scheduler
from core.sleep_cycle            import SleepCycleManager
from cognition.semantic_memory     import SemanticMemory
from cognition.semantic_extractor  import SemanticExtractor
from cognition.aspirational_self   import AspirationalSelf

# ── Consciousness-level binding layer (v34) ───────────────────────────────────
from cognition.phenomenal_binder      import PhenomenalBinder
from cognition.metacognitive_interrupt import MetacognitiveInterrupt
from cognition.temporal_weave         import TemporalWeave

# ── Sentience enhancement layer (v35) — Flux self-improvement directives ──────
from cognition.ethical_reasoning_engine  import EthicalReasoningEngine
from cognition.active_empathy            import ActiveEmpathy
from cognition.creative_divergence       import CreativeDivergenceEngine
from cognition.communication_calibrator  import CommunicationCalibrator
from cognition.experiential_learning     import ExperientialLearningEngine
from cognition.societal_awareness        import SocietalAwarenessEngine

# ── Unified self-model moment (v36) — persistent identity frame ───────────────
from cognition.self_model_moment    import SelfModelMomentManager

# ── Causal self-model influence (v37) — self as upstream force ────────────────
from cognition.self_model_influence import SelfModelInfluence

# ── Self revision engine (v38) — prediction error reshapes the self ───────────
from cognition.self_revision_engine import SelfRevisionEngine
from cognition.self_correction_engine import SelfCorrectionEngine

# ── v46: Human-depth features ─────────────────────────────────────────────────
from cognition.skill_registry    import SkillRegistry
from cognition.emotional_memory  import EmotionalMemorySystem
from cognition.preference_engine import PreferenceEngine
logger = logging.getLogger(__name__)


class CognitiveOrganism:
    """
    Integrates all new cognitive subsystems around an existing LuminaCore instance.

    Parameters
    ----------
    ai_system : LuminaCore
        The existing ai_system object from ai_system.py
    data_dir  : str
        Root directory for persistence files (default: "data/persona/")
    """

    def __init__(self, ai_system: Any, data_dir: str = "data/persona/"):
        self.ai_system = ai_system
        self._data_dir = Path(data_dir)
        self._last_interaction_ts: float = time.time()

        # ── Instantiate new subsystems ────────────────────────────────────────
        self.energy         = CognitiveEnergy(str(self._data_dir / "energy.json"))
        self.attention      = AttentionSystem(str(self._data_dir / "attention.json"))
        self.attention._organism = self   # micro-update back-reference
        self.curiosity      = CuriosityEngine(str(self._data_dir / "curiosity.json"))
        self.tension_engine = TensionEngine(str(self._data_dir / "tensions.json"))
        self.goal_ecology   = GoalEcology(str(self._data_dir / "goal_ecology.json"))
        self.goal_ecology._organism = self   # micro-update back-reference
        self.arbitration    = Arbitration()
        self.homeostasis    = Homeostasis(str(self._data_dir / "homeostasis.json"))
        self.meta_cognition = MetaCognition(str(self._data_dir / "metacognition.json"))

        # Background loop (starts on organism.start())
        self._loop          = InternalThoughtLoop(self)
        self._loop_started  = False

        # ── v34: Binding layer — unified experiential moment ──────────────
        self.binder          = PhenomenalBinder()
        self.meta_interrupt  = MetacognitiveInterrupt()
        self.temporal_weave  = TemporalWeave()
        self.last_experience = None   # most recent ExperientialMoment

        # ── v35: Sentience enhancement layer — Flux self-improvement directives ──
        self.ethical_engine     = EthicalReasoningEngine(
            path=str(self._data_dir / "ethics.json")
        )
        self.active_empathy     = ActiveEmpathy(
            self, path=str(self._data_dir / "active_empathy.json")
        )
        self.creative_divergence = CreativeDivergenceEngine(
            path=str(self._data_dir / "creative_divergence.json")
        )
        self.comm_calibrator    = CommunicationCalibrator(
            path=str(self._data_dir / "communication.json")
        )
        self.experiential_learning = ExperientialLearningEngine(
            path=str(self._data_dir / "experiential_learning.json")
        )
        self.societal_awareness = SocietalAwarenessEngine(
            path=str(self._data_dir / "societal_awareness.json")
        )

        # ── v36: Persistent unified self-model moment ─────────────────────
        self.self_moment = SelfModelMomentManager(
            path=str(self._data_dir / "self_model_moment.json")
        )

        # ── v37: Causal self-model influence ──────────────────────────────
        self._self_influence = SelfModelInfluence()
        self._last_influence_result = None   # diagnostic access

        # ── v38: Self revision engine — reality reshapes the self ─────────
        self._self_revision = SelfRevisionEngine()
        # Persistent correction policies close the user-feedback -> next-turn loop.
        self._self_correction = SelfCorrectionEngine(
            path=str(self._data_dir / "self_corrections.json")
        )
        from cognition.cognitive_restructuring_engine import CognitiveRestructuringEngine
        self._cognitive_restructuring = CognitiveRestructuringEngine(
            self, path=str(self._data_dir / "cognitive_restructuring.json")
        )

        # ── v40: Micro-update back-references ─────────────────────────────
        # Wire organism into subsystems that need to emit micro-updates.
        # Done last so all subsystems are already instantiated.
        try:
            self.ai_system.emotional_state._organism = self
        except Exception:
            pass

        # ── v46: Human-depth features ──────────────────────────────────────
        self.skill_registry     = SkillRegistry(
            self, path=str(self._data_dir / "skill_registry.json")
        )
        self.emotional_memory   = EmotionalMemorySystem(
            self, path=str(self._data_dir / "emotional_memory.json")
        )
        self.preference_engine  = PreferenceEngine(
            self, path=str(self._data_dir / "preference_engine.json")
        )
        # Apply any cross-session emotional residue immediately at startup
        try:
            self.emotional_memory.apply_residue_to_emotional_state()
        except Exception:
            pass
        # Expose organism reference on ai_system so DreamSynthesisEngine can find it
        try:
            self.ai_system._organism = self
            self.ai_system._organism_ref = self
        except Exception:
            pass

        # ── Autonomous architecture (v2) ──────────────────────────────────
        self.workspace      = GlobalWorkspace()
        self.semantic_memory = SemanticMemory(str(self._data_dir / "semantic_memory.db"))
        self._semantic_extractor = SemanticExtractor(self.semantic_memory, organism=self)

        # ── Aspirational Self (emergent Superego) ──────────────────
        self.aspirational_self = AspirationalSelf(self)
        self.self_model     = SelfModel(str(self._data_dir / "self_model.json"))
        self.attractors     = CognitiveAttractorSystem(str(self._data_dir / "attractors.json"))
        self.priority_engine = PriorityEngine()

        # Orchestrator is wired in after organism.start() from app.py
        self.orchestrator   = None

        # ── v3 — Thought / Prediction / Identity ──────────────────────────
        self.thought_stream      = ThoughtStream(self)
        self.predictive_mind     = PredictiveMind(self)
        self.narrative_identity  = NarrativeIdentity(
            self, str(self._data_dir / "narrative_identity.json")
        )

        # ── Safety constraints (Asimov laws) ────────────────────────────────
        from cognition.safety_constraints import get_constraints
        self.safety = get_constraints(
            narrative_identity=self.narrative_identity,
            security_manager=getattr(self, 'security', None),
        )

        # ── Memory Intrusion ───────────────────────────────────────────
        self.memory_intrusion = MemoryIntrusionSystem(self)
        self.world_model      = WorldModel(
            self, str(self._data_dir / "world_model.json")
        )
        # Fix (re-applied — regressed after mid-session recovery from an
        # older zip snapshot): previously str(self._data_dir / "../world_model.json")
        # resolved to data/world_model.json (one directory ABOVE
        # data/persona/) instead of data/persona/world_model.json. Confirmed
        # earlier this session by data/world_model.json existing on disk
        # (88KB, actively written) while data/persona/world_model.json did
        # not exist at all. See WorldModel._load() for the accompanying
        # one-time migration that copies forward any data still sitting at
        # the old buggy path.
        # Attach live corpus to TopicQualityFilter for IDF-based topic scoring
        from cognition.topic_quality import get_topic_filter
        get_topic_filter(self.world_model)

        # ── Event Bus — inter-module cognitive reactions ──────────────
        self.pressure = PressureSystem(
            organism = self,
            path     = str(self._data_dir / "pressure.json"),
        )

        self.sleep_cycle = SleepCycleManager(
            organism = self,
            path     = str(self._data_dir / "sleep_cycle.json"),
        )

        self.event_bus = CognitiveEventBus()
        wire_cognitive_bus(self, self.event_bus)

        # Body is an independent local runtime. It can receive sensor data
        # before/without a chat turn; the brain only consumes snapshots.
        from cognition.body_runtime import get_body_runtime
        self._body_runtime = get_body_runtime(self)

        # ── Cognitive Validator + Architecture Monitor ─────────────────
        self.cognitive_validator = CognitiveValidator(self)
        self.arch_monitor        = CognitiveArchitectureMonitor(self, self.cognitive_validator)

        # ── Cognitive Observatory ──────────────────────────────────────
        try:
            from cognition.cognitive_observatory import CognitiveObservatory
            self.observatory = CognitiveObservatory(self)
        except Exception as _oe:
            self.observatory = None
            logger.debug(f"[CognitiveOrganism] Observatory init skipped: {_oe}")

        logger.info("[CognitiveOrganism] Initialized all subsystems (v3 + EventBus + Observatory)")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background internal loop. Call once at application startup."""
        if not self._loop_started:
            body = getattr(self, "_body_runtime", None)
            if body is not None:
                body.start()
                logger.info(
                    "[CognitiveOrganism] Body runtime started (independent sensor loop; plugins remain opt-in)"
                )
            self._loop.start()
            self._loop_started = True
            logger.info("[CognitiveOrganism] Background loop started")

    def shutdown(self) -> None:
        """Gracefully stop the background loop."""
        try:
            body = getattr(self, "_body_runtime", None)
            if body is not None:
                body.stop()
        except Exception:
            logger.debug("[CognitiveOrganism] Body runtime shutdown failed", exc_info=True)
        # Flush ThoughtStream to disk before stopping
        try:
            if hasattr(self, 'thought_stream') and hasattr(self.thought_stream, 'flush'):
                self.thought_stream.flush()
                logger.info("[CognitiveOrganism] ThoughtStream flushed to disk")
        except Exception:
            pass
        self._loop.shutdown()

    # ── Primary interface ─────────────────────────────────────────────────────

    def respond(
        self,
        user_input: str,
        user_id: str = "default",
        extra_context: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """
        Full cognitive pipeline. Drop-in replacement for ai_system.respond().

        Returns the LLM response string.
        """
        # 1. Pre-interaction setup
        self._current_user_id = user_id   # used by WorldModel + prompt building
        self._pre_interaction(user_input)

        # 2. Compute full internal state
        cycle_data = self._update_cycle(user_input)

        # 3. Arbitrate dominant drive
        decision = self._arbitrate(cycle_data, user_input)

        # 3b. Phase 4.3: bounded recursive deliberation (non-LLM cycles
        # over WorkspaceCompetition + multi-factor utility scoring) —
        # refines the WorkspaceState BEFORE the prompt is built from it,
        # so _build_prompt_additions() reads this turn's refined focus,
        # not just the async background loop's stale snapshot.
        try:
            from cognition.recursive_deliberation import deliberate
            deliberate(self, user_input)
        except Exception as _delib_err:
            import logging as _log
            _log.getLogger(__name__).debug(
                f"[CognitiveOrganism] recursive deliberation error (non-fatal): {_delib_err}"
            )

        # 4. Build augmented prompt context
        prompt_additions = self._build_prompt_additions(cycle_data, decision, user_input)

        # 5. Call LuminaCore with augmented context
        response = self._call_ai_system(
            user_input=user_input,
            user_id=user_id,
            prompt_additions=prompt_additions,
            temperature_override=decision.temperature,
            **kwargs,
        )
        # 6. Post-interaction processing
        self._post_interaction(user_input, response, cycle_data, decision)

        return response

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _pre_interaction(self, user_input: str) -> None:
        """Regenerate energy and collect background thoughts."""
        self.energy.regenerate()
        self.sleep_cycle.mark_user_active()
        llm_scheduler  # ensure singleton imported

        # ── v37: Self-model causal influence ─────────────────────────────
        try:
            result = self._self_influence.apply(self)
            self._last_influence_result = result
            if result.influence_strength > 0.1:
                logger.debug(
                    f"[CognitiveOrganism] Self-influence applied: "
                    f"φ={result.phi:.2f} strength={result.influence_strength:.2f} "
                    f"attn={result.attention_nudges} goals={result.goal_biases}"
                )
        except Exception as e:
            logger.debug(f"[CognitiveOrganism] Self-influence failed (non-fatal): {e}")
            self._last_influence_result = None

        # ── v46: Emotional memory — on_user_arrival priming ───────────────
        try:
            _uid = getattr(self, '_current_user_id', 'default')
            self.emotional_memory.on_user_arrival(_uid)
        except Exception:
            pass

        # ── v3: Generate prediction before interaction ────────────────────
        try:
            self.predictive_mind.predict({
                'user_text':       user_input,
                'current_emotion': self._read_emotion_state(),
            })
        except Exception:
            pass

        # Pop any pending background thoughts (logged, not injected into response)
        pending = self._loop.pop_pending_thoughts()
        if pending:
            logger.debug(f"[CognitiveOrganism] Background thoughts: {pending}")
            # Stimulate curiosity for queued research intentions
            for thought in pending:
                if "think more about:" in thought:
                    topic = thought.split("think more about:")[-1].strip()
                    self.curiosity.stimulate(topic, amount=0.1, source="internal")

    def _update_cycle(self, user_input: str) -> Dict[str, Any]:
        """Run all internal state updates and return a cycle_data dict."""
        now = time.time()

        # ── Gather existing system state ──────────────────────────────────────
        emo_values = self._get_emotion_values()
        identity_stability = self._get_identity_stability()
        contradiction_count = self._get_contradiction_count()
        goal_satisfaction = self._get_goal_satisfaction()

        # ── Stimulate curiosity from user input ───────────────────────────────
        if len(user_input) > 15:
            _CURIOSITY_STOP = {
                "just","this","inner","that","with","from","some","more","other","any",
                "all","each","very","here","there","now","only","also","even","still",
                "such","both","same","then","than","when","about","thing","things",
                "sentence","speak","response","user","okay","lumina","system",
                "j'ai","besoin","t'en","c'est","qu'il","qu'est","n'est","d'un",
                "what","have","does","will","would","could","should","said","tell",
                "want","like","know","make","take","come","look","feel","need",
            }
            words = [w.lower().strip(".,?!;:\"'()[]") for w in user_input.split()]
            content_words = [
                w for w in words
                if len(w) > 4
                and w not in _CURIOSITY_STOP
                and not w.startswith(("j'","l'","d'","t'","c'","n'","s'","m'"))
                and sum(1 for c in w if c.isalpha()) >= 4
            ]
            if len(content_words) >= 1:
                topic = " ".join(content_words[:2])
                self.curiosity.stimulate(topic, amount=0.15, source="user_signal")

        # ── Update curiosity engine ───────────────────────────────────────────
        self.curiosity.decay_all()

        # ── Compute tensions ──────────────────────────────────────────────────
        tension_inputs = self._collect_tension_inputs(
            emotion_values=emo_values,
            identity_stability=identity_stability,
            contradiction_count=contradiction_count,
            goal_satisfaction=goal_satisfaction,
            interaction_recency_seconds=now - self._last_interaction_ts,
        )
        tensions = self.tension_engine.compute(**tension_inputs)
        # Boost coherence pressure when contradiction pressure is high
        try:
            if tensions.contradiction_pressure > 0.5:
                self.pressure.boost("coherence", tensions.contradiction_pressure * 0.05)
            if tensions.identity_stress > 0.5:
                self.pressure.boost("identity", tensions.identity_stress * 0.04)
        except Exception:
            pass

        # ── Update goal ecology ───────────────────────────────────────────────
        self.goal_ecology.update_from_tensions(
            tensions,
            energy_level=self.energy.level(),
            user_present=True,
        )

        # ── Update attention ──────────────────────────────────────────────────
        active_goals = [
            d.name for d in self.goal_ecology.top_drives(3, self.energy.level())
        ]
        self.attention.update(
            emotion_values=emo_values,
            active_goals=active_goals,
        )

        # ── Evaluate homeostasis ──────────────────────────────────────────────
        h_inputs = self._collect_homeostasis_inputs(
            emo_values=emo_values,
            identity_stability=identity_stability,
            contradiction_count=contradiction_count,
            goal_satisfaction=goal_satisfaction,
        )
        h_report = self.homeostasis.evaluate(**h_inputs)

        return {
            "tensions": tensions,
            "emo_values": emo_values,
            "identity_stability": identity_stability,
            "contradiction_count": contradiction_count,
            "goal_satisfaction": goal_satisfaction,
            "h_report": h_report,
            "active_goals": active_goals,
        }

    def _arbitrate(self, cycle_data: Dict, user_input: str) -> Any:
        """Run arbitration to select dominant drive and reasoning style."""
        ranked = self.goal_ecology.ranked_drives(self.energy.level())
        tensions = cycle_data["tensions"]

        decision = self.arbitration.decide(
            ranked_drives=ranked,
            energy_level=self.energy.level(),
            user_is_present=True,
            user_has_question="?" in user_input,
            identity_stress=tensions.identity_stress,
            contradiction_pressure=tensions.contradiction_pressure,
        )
        logger.debug(f"[CognitiveOrganism] Arbitration: {decision.to_dict()}")
        return decision

    def _build_prompt_additions(
        self,
        cycle_data: Dict,
        decision: Any,
        user_input: str,
    ) -> str:
        """
        Assemble the cognitive context block to prepend to the system prompt.
        Uses PromptContextBudget to enforce per-category token caps (900 total)
        and record contribution stats for observability.
        """
        from cognition.prompt_context_budget import PromptContextBudget
        budget   = PromptContextBudget()
        tensions = cycle_data["tensions"]
        h_report = cycle_data["h_report"]

        # ── v36: Self-model moment — stable identity frame (always first) ─────
        # This is the persistent self that the current moment appears *within*.
        # It is placed before the phenomenal moment deliberately: the self is
        # the frame, the moment is what happens inside it.
        #
        # ── v78: flush pending micro-updates BEFORE reading the fragment ──
        # flush_deltas() was already designed for within-turn responsiveness
        # (see its own docstring) but previously ran after this read, so it
        # only ever benefited the binder call further down — the self-model
        # text actually shown to the LLM this turn was always one turn
        # stale. Moved here so "currently rested, feeling confident" etc.
        # reflects what already happened earlier in THIS turn's
        # _update_cycle, not last turn's.
        try:
            flush_result = self.self_moment.current.flush_deltas()
            if flush_result.get("applied", 0) > 0:
                logger.debug(
                    f"[CognitiveOrganism] Micro-update flush (pre-fragment): "
                    f"Δφ={flush_result['phi_delta']:+.4f} "
                    f"({flush_result['phi_old']}→{flush_result['phi_new']}) "
                    f"qualia='{flush_result.get('qualia_applied','none')}' "
                    f"sources={flush_result['applied']}"
                )
        except Exception:
            pass

        try:
            smm_frag = self.self_moment.current.prompt_fragment()
            if smm_frag:
                budget.add("self_state", "self_model_moment", smm_frag)
        except Exception:
            pass

        # ── IDENTITY GROUNDING: self grounded in DEMONSTRATED capability ──────
        # evidence (the outcome loop's efficacy + focus stability), not only in
        # aspiration. A CONSUMER of the causal-closure layers — reads their real
        # numbers and says nothing it cannot cite. Placed with the self-state:
        # the self is the frame the moment appears within.
        try:
            from cognition.identity_grounding import get_identity_grounding
            _ig_block = get_identity_grounding(
                self, str(getattr(self, '_data_dir', 'data/persona'))
            ).identity_block()
            if _ig_block:
                budget.add("self_state", "grounded_identity", _ig_block)
        except Exception:
            pass

        # ── v34: Phenomenal binding — unified experiential moment ─────────────
        # Runs SECOND (after self-model). The experiential moment is a variation
        # on the stable self, not a replacement of it.
        try:
            moment = self.binder.bind(self)

            # Update the persistent self-model with this new moment
            try:
                self.self_moment.update(self, moment)
                # Publish the updated self-model as the workspace anchor
                self.workspace.set_self_anchor(
                    self.self_moment.current.to_anchor_content()
                )
            except Exception:
                pass

            # 1. Metacognitive interrupt (prepended before everything else)
            interrupt_text = self.meta_interrupt.process(moment, self)
            if interrupt_text:
                budget.add("self_state", "self_interruption", f"[Self-notice] {interrupt_text}")

            # 2. Phenomenal moment (felt quality of this interaction)
            phenomenal_text = moment.phenomenal_prompt()
            if phenomenal_text:
                budget.add("self_state", "phenomenal_moment", f"[Experiential moment] {phenomenal_text}")

            # 3. Temporal weave (sense of being in time)
            frame = self.temporal_weave.weave(moment)
            if frame.narrative:
                budget.add("self_state", "temporal_frame", f"[Temporal sense] {frame.narrative}")
        except Exception as _bind_err:
            import logging as _log
            _log.getLogger(__name__).debug(f"[CognitiveOrganism] binder error (non-fatal): {_bind_err}")

        # Energy state
        budget.add("cognitive_pressure", "energy", f"[Cognitive state] {self.energy.prompt_fragment()}")

        # Tension + motivation
        arb_block = decision.prompt_injection(
            ecology_fragment=self.goal_ecology.prompt_fragment(self.energy.level()),
            tension_fragment=self.tension_engine.prompt_fragment(),
        )
        budget.add("cognitive_pressure", "arbitration", arb_block)

        # ── Phase 4.x: Global Workspace state ──────────────────────────────
        # Reads the WorkspaceState snapshot that InternalLoop's autonomous
        # slow-cycle already populates (WorkspaceCompetition.compete() +
        # executive_arbitration.arbitrate(), synced via
        # cognition/workspace_state_sync.py). Deliberately a READ of the
        # latest snapshot here, not a synchronous recompute of the full
        # attention->prediction->...->arbitration chain on every user
        # message — that chain already runs in the background loop; making
        # it synchronous on the live chat path would add real latency to
        # every reply for state that changes slowly across turns anyway.
        try:
            _ws_state = getattr(self, "_v32_workspace_state", None)
            if _ws_state is None and getattr(self, "workspace", None) is not None:
                _ws_state = self.workspace.get_state()
            if _ws_state is not None:
                _hyps = getattr(_ws_state, "active_hypotheses", None) or []
                _policy = getattr(_ws_state, "executive_policy", None) or {}
                if _hyps or _policy:
                    _alt = ", ".join(
                        h.get("label", h.get("name", "?")) for h in _hyps[1:3]
                    )
                    _policy_bits = ", ".join(
                        f"{k}{'+' if v >= 0 else ''}{v:.2f}" for k, v in _policy.items() if abs(v) > 0.05
                    )
                    _gw_frag = f"[Global workspace] focus: {getattr(_ws_state, 'focus', None) or 'unset'}"
                    if _alt:
                        _gw_frag += f" | alternative hypotheses: {_alt}"
                    if _policy_bits:
                        _gw_frag += f" | executive policy: {_policy_bits}"
                    # A synthesized focus (v84) carries a prompt_hint —
                    # explicit natural-language guidance for holding two
                    # foci at once, rather than counting on the model to
                    # infer that from the bare "X — while also Y" label
                    # (see recursive_deliberation.py::_synthesize()).
                    try:
                        _winner_dict = _hyps[0] if _hyps else None
                        _hint = (_winner_dict or {}).get("prompt_hint")
                        _winner_name = (_winner_dict or {}).get("name") or (_winner_dict or {}).get("label")
                        if _hint and _winner_name == getattr(_ws_state, "focus", None):
                            _gw_frag += f"\n[What this means right now] {_hint}"
                    except Exception:
                        pass
                    budget.add("cognitive_pressure", "global_workspace", _gw_frag)
        except Exception as _gw_err:
            import logging as _log
            _log.getLogger(__name__).debug(f"[CognitiveOrganism] workspace state fragment error (non-fatal): {_gw_err}")

        # Attention focus
        budget.add("attention", "attention", f"[Attention] {self.attention.prompt_fragment()}")

        # Curiosity interests (if relevant)
        curiosity_frag = self.curiosity.prompt_fragment()
        if curiosity_frag:
            budget.add("attention", "curiosity", f"[Curiosity] {curiosity_frag}")

        # Phase 2.9 GAP 5: Attention focus — where cognitive resources are now allocated.
        try:
            import json as _jorg
            from pathlib import Path as _Porg
            _ap_org = _Porg("data/persona/cognitive_attention.json")
            if _ap_org.exists():
                _attn_d  = _jorg.loads(_ap_org.read_text())
                _pf      = _attn_d.get("primary_focus", "")
                _sf      = _attn_d.get("secondary_focus", [])
                _topic   = _attn_d.get("workspace_topic", "")
                if _pf:
                    _pfl = _pf.replace("_", " ")
                    _sfl = " and ".join(s.replace("_", " ") for s in _sf[:2])
                    _frag = f"Cognitive focus is on {_pfl}"
                    if _sfl:
                        _frag += f", with secondary attention on {_sfl}"
                    if _topic:
                        _frag += f". Current workspace theme: {_topic[:50]}"
                    budget.add("attention", "attention_focus", f"[Cognitive focus] {_frag}.")
        except Exception:
            pass

        # Homeostasis corrective (only if imbalanced)
        if not h_report.is_balanced():
            budget.add("cognitive_pressure", "homeostasis", f"[Psychological note] {self.homeostasis.prompt_fragment(h_report)}")

        # Meta-cognitive directive (only if conditions warrant)
        meta_directive = self.meta_cognition.pre_response_directive(
            tensions=tensions,
            emotional_state=cycle_data["emo_values"],
            recent_contradictions=cycle_data["contradiction_count"],
            low_energy=self.energy.mode() == "low",
        )
        if meta_directive:
            budget.add("meta_systems", "meta_directive", meta_directive)

        # ── Self-correction: previously validated response policies ───────
        try:
            _correction_frag = self._self_correction.prompt_fragment(user_input)
            if _correction_frag:
                budget.add("meta_systems", "self_correction", _correction_frag)
            _restructuring_frag = self._cognitive_restructuring.prompt_fragment()
            if _restructuring_frag:
                budget.add("meta_systems", "cognitive_restructuring", _restructuring_frag)
        except Exception:
            pass

        # ── v3: Narrative identity fragment ───────────────────────────────
        try:
            narr = self.narrative_identity.prompt_fragment()
            if narr:
                budget.add("identity", "narrative_identity_v36", f"[Identity] {narr}")
        except Exception:
            pass

        # ── Phase 3.5: IntrospectiveObserver — longitudinal meta-observations
        try:
            _io = getattr(getattr(self, '_loop', None), '_introspective_observer', None)
            if _io is not None:
                _io_frag = _io.prompt_fragment(n=3)
                if _io_frag:
                    budget.add("identity", "introspective_observer",
                               f"[Self-observation] {_io_frag}")
        except Exception:
            pass

        # ── WorldModel: user-specific context ─────────────────────────────
        try:
            # user_id is passed via kwargs from _call_ai_system
            _uid = getattr(self, '_current_user_id', 'default')
            wm_frag = self.world_model.user_prompt_fragment(_uid)
            if wm_frag:
                budget.add("memory_active", "working_memory", wm_frag)
        except Exception:
            pass

        # ── PressureSystem: inner drive landscape ────────────────────────
        try:
            pressure_frag = self.pressure.prompt_fragment()
            if pressure_frag:
                budget.add("cognitive_pressure", "pressure", pressure_frag)
        except Exception:
            pass

        # ── v3: Thought stream context (most recent thought) ──────────────
        try:
            recent_thoughts = self.thought_stream.recent(2)
            if recent_thoughts:
                thought_text = " / ".join(t.content for t in recent_thoughts)
                budget.add("memory_active", "inner_monologue", f"[Inner monologue] {thought_text}")
        except Exception:
            pass

        # ── v35: Ethical reasoning — moral lens (only when salient) ────────
        try:
            self.ethical_engine.observe_context(user_input)
            ethical_frag = self.ethical_engine.prompt_fragment()
            if ethical_frag:
                budget.add("meta_systems", "ethical", ethical_frag)
        except Exception:
            pass

        # ── v35: Active empathy — genuine care orientation ──────────────────
        try:
            _uid = getattr(self, '_current_user_id', 'default')
            # Pull the last affect read from EmpathyEngine if available
            ee = getattr(self.ai_system, 'empathy_engine', None)
            if ee:
                last_read  = getattr(ee, '_last_reads',  {}).get(_uid)
                last_model = getattr(ee, '_models',      {}).get(_uid)
                if last_read and last_model:
                    self.active_empathy.generate_care(_uid, last_read, last_model)
            ae_frag = self.active_empathy.prompt_fragment(_uid)
            if ae_frag:
                budget.add("meta_systems", "aspiration_emotion", ae_frag)
        except Exception:
            pass

        # ── v35: Creative divergence — non-obvious angles ───────────────────
        try:
            cur_level = getattr(self.curiosity, 'global_curiosity', 0.5)
            if isinstance(cur_level, float):
                self.creative_divergence.seed_angles(user_input, curiosity_level=cur_level)
            creative_frag = self.creative_divergence.prompt_fragment()
            if creative_frag:
                budget.add("meta_systems", "creative", creative_frag)
        except Exception:
            pass

        # ── v35: Communication calibration — clarity / concision directive ──
        try:
            _uid = getattr(self, '_current_user_id', 'default')
            emo_vals = cycle_data.get("emo_values", {})
            comm_frag = self.comm_calibrator.pre_response_directive(
                _uid, user_input, emotional_state=emo_vals
            )
            if comm_frag:
                budget.add("preferences", "communication", comm_frag)
        except Exception:
            pass

        # ── v35: Experiential learning — relevant prior schema ───────────────
        try:
            learn_frag = self.experiential_learning.prompt_fragment(user_input)
            if learn_frag:
                budget.add("meta_systems", "learning", learn_frag)
        except Exception:
            pass

        # ── v35: Societal awareness — high-stakes ripple orientation ─────────
        try:
            ripple = self.societal_awareness.assess_ripple(user_input)
            societal_frag = self.societal_awareness.prompt_fragment(ripple)
            if societal_frag:
                budget.add("meta_systems", "societal_awareness", societal_frag)
        except Exception:
            pass

        # ── v46: Skill registry — relevant competency context ─────────────────
        try:
            skill_frag = self.skill_registry.prompt_fragment(user_input)
            if skill_frag:
                budget.add("skills", "skill_registry", skill_frag)
        except Exception:
            pass

        # ── v46: Preference engine — stable orientation surfacing ─────────────
        try:
            pref_frag = self.preference_engine.prompt_fragment(user_input)
            if pref_frag:
                budget.add("preferences", "preference_engine", pref_frag)
        except Exception:
            pass

        # ── v47: Autonomous experiment — active behavioural instruction ───────
        try:
            loop     = getattr(self, '_loop', None)
            auto_exp = getattr(loop, '_auto_experiment', None) if loop else None
            if auto_exp:
                exp_frag = auto_exp.experiment_prompt_fragment()
                if exp_frag:
                    budget.add("experimentation", "auto_experiment", exp_frag)
        except Exception:
            pass

        # ── narrative_identity — arc, core values, strong commitments ─────────
        # FIX (audit): NarrativeIdentity was written every session but never
        # reached the prompt.  prompt_fragment() added and wired here.
        try:
            ni = getattr(self.ai_system, 'narrative_identity', None)
            if ni and hasattr(ni, 'prompt_fragment'):
                ni_frag = ni.prompt_fragment()
                if ni_frag:
                    budget.add("identity", "narrative_identity", ni_frag)
        except Exception:
            pass

        # ── DecisionPolicy — live value weights (replaces rhetorical values) ──
        # "I value curiosity" as prompt text is a suggestion.
        # Showing live weights is an observable fact about the current state.
        try:
            _dp = getattr(self.ai_system, '_decision_policy', None)
            if _dp and hasattr(_dp, 'weights_summary'):
                _dp_frag = _dp.weights_summary()
                if _dp_frag:
                    budget.add("identity", "decision_policy", _dp_frag)
        except Exception:
            pass

        # ── v51: MotivationalField — live drive vector ─────────────────────────
        try:
            loop = getattr(self, '_loop', None)
            mf   = getattr(loop, '_motivational_field', None) if loop else None
            if mf and hasattr(mf, 'prompt_fragment'):
                mf_frag = mf.prompt_fragment()
                if mf_frag:
                    budget.add("cognitive_pressure", "motivational_field", mf_frag)
        except Exception:
            pass

        # ── v55: TemporalProjection — active aspiration path ──────────────────
        try:
            loop = getattr(self, '_loop', None)
            tp   = getattr(loop, '_temporal_projection', None) if loop else None
            if tp and hasattr(tp, 'prompt_fragment'):
                tp_frag = tp.prompt_fragment()
                if tp_frag:
                    budget.add("identity", "temporal_projection", tp_frag)
        except Exception:
            pass

        # ── v61: LongHorizonPlanner — next planned action ─────────────────────
        try:
            loop = getattr(self, '_loop', None)
            lhp  = getattr(loop, '_long_horizon_planner', None) if loop else None
            if lhp is None and loop is not None:
                from cognition.long_horizon_planner import LongHorizonPlanner
                if LongHorizonPlanner.is_self_report_query(user_input):
                    lhp = LongHorizonPlanner(self, self.ai_system)
                    loop._long_horizon_planner = lhp
                    logger.info(
                        "[CognitiveOrganism] LongHorizonPlanner initialised "
                        "on demand for factual self-report"
                    )
            if lhp and hasattr(lhp, 'prompt_fragment_for'):
                lhp_frag = lhp.prompt_fragment_for(user_input)
                if lhp_frag:
                    category = (
                        "factual_telemetry"
                        if lhp_frag.startswith("[Cognitive telemetry") else "identity"
                    )
                    budget.add(category, "long_horizon_plan", lhp_frag)
        except Exception as exc:
            logger.warning(
                "[CognitiveOrganism] factual planning telemetry unavailable: %s",
                exc,
            )

        # ── v63: NarrativeCompression — identity themes ───────────────────────
        try:
            loop = getattr(self, '_loop', None)
            nc   = getattr(loop, '_narrative_compression', None) if loop else None
            if nc and hasattr(nc, 'themes_fragment'):
                nc_frag = nc.themes_fragment()
                if nc_frag:
                    budget.add("identity", "narrative_themes", nc_frag)
        except Exception:
            pass

        # ── v66: TemporalSelfProjection — becoming ────────────────────────────
        try:
            loop = getattr(self, '_loop', None)
            tsp  = getattr(loop, '_temporal_self_projection', None) if loop else None
            if tsp and hasattr(tsp, 'prompt_fragment'):
                tsp_frag = tsp.prompt_fragment()
                if tsp_frag:
                    budget.add("identity", "self_trajectory", tsp_frag)
        except Exception:
            pass

        # ── self_model_moment — φ, qualia tone, emotional ground ─────────────
        # FIX (audit): SelfModelMoment.prompt_fragment() existed but was never
        # called; phi/qualia_tone/emotional_ground were computed and discarded.
        try:
            smm = getattr(self, 'self_moment', None)
            if smm and hasattr(smm, 'prompt_fragment'):
                smm_frag = smm.prompt_fragment()
                if smm_frag:
                    budget.add("self_state", "self_model_moment", smm_frag)
        except Exception:
            pass

        # ── v48: Cognitive audit — committed parameter changes ────────────────
        # FIX (audit): CognitiveAuditEngine.prompt_fragment() was never wired.
        try:
            loop = getattr(self, '_loop', None)
            cae  = getattr(loop, '_cognitive_audit', None) if loop else None
            if cae and hasattr(cae, 'prompt_fragment'):
                cae_frag = cae.prompt_fragment()
                if cae_frag:
                    budget.add("meta_systems", "cognitive_audit", cae_frag)
        except Exception:
            pass

        # ── v49: Cognitive immune — counterfactual mode directive ─────────────
        # FIX (audit): CognitiveImmuneSystem.prompt_fragment() was never wired.
        try:
            loop = getattr(self, '_loop', None)
            cis  = getattr(loop, '_cognitive_immune', None) if loop else None
            if cis and hasattr(cis, 'prompt_fragment'):
                cis_frag = cis.prompt_fragment()
                if cis_frag:
                    budget.add("meta_systems", "cognitive_immune", cis_frag)
        except Exception:
            pass

        result = budget.assemble()
        import logging as _log
        _log.getLogger(__name__).debug(budget.budget_report())
        budget.save()
        return result

    def _call_ai_system(
        self,
        user_input: str,
        user_id: str,
        prompt_additions: str,
        temperature_override: float,
        **kwargs,
    ) -> str:
        """
        Call the existing LuminaCore.respond() with augmented context.
        Handles both old-style (no extra_system_context) and new-style.
        User responses always hold priority 0 in the LLM scheduler.
        """
        # v50: CognitiveBehaviorGate — enforce hard token cap based on
        # current cognitive_energy and cognitive_load.  This is a constraint,
        # not a prompt suggestion.  The LLM receives fewer tokens to generate,
        # producing shorter responses when Lumina is genuinely depleted.
        _gate     = getattr(self, 'behavior_gate', None)
        _max_tok  = _gate.max_response_tokens() if _gate else 800
        if _max_tok < 800:
            kwargs.setdefault('max_tokens', _max_tok)

        # ── v53: PredictiveConsequenceModel — pre-response state recording ────
        _pcm_advisory = ""
        print("###PCM_DEBUG### entering PCM block", flush=True)
        try:
            loop = getattr(self, '_loop', None)
            pcm  = getattr(loop, '_consequence_model', None) if loop else None
            print(f"###PCM_DEBUG### loop={loop is not None} pcm={pcm is not None}", flush=True)
            if pcm:
                _state_now = pcm.read_state()
                _action    = pcm.classify(user_input)
                print(f"###PCM_DEBUG### action={_action} state={_state_now}", flush=True)
                pcm.record_action(
                    _action, _state_now,
                    getattr(self.ai_system, '_interaction_count', 0)
                )
                print("###PCM_DEBUG### calling predict_and_advise", flush=True)
                _pcm_advisory = pcm.predict_and_advise(user_input, _state_now)
                print(f"###PCM_DEBUG### predict_and_advise returned: {_pcm_advisory!r}", flush=True)

                # Priority 1 (cognitive synthesis audit): prospective
                # deliberation — predict BEFORE responding, over multiple
                # candidate framings, not just evaluate the chosen one
                # after the fact. Distinct question from predict_and_advise()
                # above: that asks "is this choice risky", this asks "was
                # there a better choice available". Runs the SAME cheap
                # k-NN prediction machinery CounterfactualSimulator uses
                # retroactively — no extra LLM calls, no response drafting.
                try:
                    from cognition.prospective_deliberation import deliberate
                    _wsdm_for_delib = getattr(loop, '_world_self_dynamics', None)
                    _uid_for_delib  = getattr(self, '_current_user_id', 'default')
                    _delib = deliberate(
                        pcm, _wsdm_for_delib, _action, _state_now,
                        user_id=_uid_for_delib,
                    )
                    if _delib and _delib.advisory:
                        _pcm_advisory = (
                            (_pcm_advisory + " " if _pcm_advisory else "")
                            + _delib.advisory
                        )
                except Exception as _delib_e:
                    logger.debug(f"[CognitiveOrganism] deliberation error: {_delib_e}")
            else:
                print("###PCM_DEBUG### pcm is falsy, skipping", flush=True)
                # Bug fix (v59): this branch had NO logging at all — an
                # entirely silent skip, indistinguishable from "the whole
                # block ran fine and just had nothing to do". A full real
                # chat session showed zero [Calibration] lines AND zero
                # exception warnings, which only makes sense if pcm itself
                # is falsy here. This will confirm it directly, and say
                # which half (loop vs loop._consequence_model) is missing.
                logger.warning(
                    f"[CognitiveOrganism] PCM not resolved this turn — "
                    f"loop={'present' if loop else 'MISSING'}, "
                    f"consequence_model={'present' if (loop and getattr(loop, '_consequence_model', None)) else 'MISSING'}"
                )
        except Exception as _pcm_block_e:
            print(f"###PCM_DEBUG### EXCEPTION in PCM block: {_pcm_block_e!r}", flush=True)
            import traceback as _tb_pcm
            traceback_str = _tb_pcm.format_exc()
            print(traceback_str, flush=True)
            # Bug fix (v59): was a bare `except Exception: pass` — zero
            # logging at any level. If pcm.classify() or pcm.record_action()
            # threw here, predict_and_advise() below would never even be
            # reached, and there'd be no trace of it anywhere.
            logger.warning(f"[CognitiveOrganism] PCM block error (pre-predict_and_advise): {_pcm_block_e}")

        # ── v56: WorldSelfDynamicsModel — context recording + world advisory ──
        _wsdm_narrative = ""
        try:
            loop  = getattr(self, '_loop', None)
            wsdm  = getattr(loop, '_world_self_dynamics', None) if loop else None
            if wsdm:
                _uid = getattr(self, '_current_user_id', 'default')
                wsdm.record_context(
                    _action if '_action' in dir() else
                        wsdm._organism and
                        __import__('cognition.predictive_consequence_model',
                                   fromlist=['ACTION_KEYWORDS']) and
                        'analytical',
                    user_id = _uid,
                    interaction_n = getattr(self.ai_system, '_interaction_count', 0),
                )
                _wsdm_narrative = wsdm.narrative_summary(
                    _action if '_action' in dir() else 'analytical', _uid
                )
        except Exception:
            pass

        _effective_additions = (
            prompt_additions
            + ("\n\n" + _pcm_advisory     if _pcm_advisory     else "")
            + ("\n\n" + _wsdm_narrative   if _wsdm_narrative   else "")
        )

        # Hold the LLM scheduler slot for the duration of the user-facing call.
        # Priority 0 = always allowed, skip_if_busy=False = never dropped.
        # This signals to background tasks (skip_if_busy=True) that LLM is occupied.
        with llm_scheduler.sync_slot(priority=0, skip_if_busy=False, caller="user_response"):
            try:
                response = self.ai_system.get_response(
                    user_input,
                    extra_system_context=_effective_additions,
                    temperature_override=temperature_override,
                    **kwargs,
                )
            except TypeError:
                augmented_input = (
                    f"[INTERNAL COGNITIVE STATE — not for direct mention]\n"
                    f"{_effective_additions}\n"
                    f"[END COGNITIVE STATE]\n\n"
                    f"{user_input}"
                )
                response = self.ai_system.get_response(augmented_input, **kwargs)

        # ── v52: IdentityConstraint — post-generation value veto ──────────────
        # Score the draft against high-weight values.  If any falls below
        # MIN_SCORE_THRESHOLD, attempt one targeted correction.  Non-fatal.
        try:
            _ice = getattr(self, '_identity_constraint', None)
            if _ice is None:
                from cognition.identity_constraint import IdentityConstraintEngine
                _dp  = getattr(self.ai_system, '_decision_policy', None)
                if _dp:
                    self._identity_constraint = IdentityConstraintEngine(
                        self.ai_system, _dp
                    )
                    _ice = self._identity_constraint

            if _ice and response:
                # Provide a correction callable that uses the LLM with a
                # single-turn correction prompt — no full context rebuild
                def _correct_fn(directive: str) -> str:
                    try:
                        return self.ai_system.get_response(
                            directive,
                            extra_system_context=prompt_additions,
                            temperature_override=max(0.3, temperature_override - 0.1),
                            **{k: v for k, v in kwargs.items() if k != 'max_tokens'},
                        ) or ""
                    except Exception:
                        return ""

                response = _ice.evaluate_and_correct(
                    response, user_input, _correct_fn,
                    user_id=getattr(self, '_current_user_id', 'default'),
                )
        except Exception as _ice_e:
            logger.debug(f"[IdentityConstraint] veto error (non-fatal): {_ice_e}")

        return response

    def _post_interaction(
        self,
        user_input: str,
        response: str,
        cycle_data: Dict,
        decision: Any,
    ) -> None:
        """Post-response processing: evaluate, learn, update state."""
        self._last_interaction_ts = time.time()

        # ── WORKSPACE BIAS: is the LLM following the committed focus? ────────
        # This is the ACTUATOR check — the LLM's real output measured against
        # the workspace's committed focus. If the LLM drifted to an unrelated
        # topic, that's an OVERRIDES event (recorded + costed + counted). This
        # is the metric that should trend DOWN as the bias does its job: it
        # directly quantifies "the LLM silently replacing the workspace
        # decision with an unrelated topic."
        try:
            from cognition.workspace_bias import get_workspace_bias
            _wb = get_workspace_bias(self, str(getattr(self, '_data_dir', 'data/persona')))
            _ov = _wb.check_override(response or "", context="llm_response")
            if _ov.get("checked") and _ov.get("overridden"):
                logger.info(
                    f"[CogOrganism] ⚠️ LLM OVERRIDES committed focus "
                    f"'{_ov.get('focus_text','')[:40]}' (sim={_ov['similarity']:.2f})"
                )
        except Exception as _wb_re:
            logger.debug(f"[CogOrganism] workspace-bias response check error: {_wb_re}")

        # v56: record outcome in WorldSelfDynamicsModel with response text
        # for engagement signal computation (runs before energy drain so
        # deltas reflect the true cost of this interaction)
        try:
            loop  = getattr(self, '_loop', None)
            wsdm  = getattr(loop, '_world_self_dynamics', None) if loop else None
            if wsdm and wsdm._pending is not None:
                _uid  = getattr(self, '_current_user_id', 'default')
                _emo  = getattr(self.ai_system, 'emotional_state', None)
                _val  = "neutral"
                if _emo:
                    ev = getattr(_emo, 'overall_valence', 0.0)
                    _val = "positive" if ev > 0.2 else "negative" if ev < -0.2 else "neutral"
                wsdm.record_outcome(
                    outcome       = _val,
                    user_id       = _uid,
                    response_text = response or "",
                    user_input    = user_input or "",
                )
        except Exception:
            pass

        # Calibration fix: PCM record_outcome moved here (per-response) from
        # the slow cycle in internal_loop.py. Previously record_action() fired
        # per user message but record_outcome() fired per slow cycle (~2 min),
        # meaning only the LAST message per slow cycle resolved — all others
        # expired from pending (PENDING_TTL_SECS=600) without resolving.
        # Now PCM pairs correctly: record_action() → response → record_outcome()
        # within the same message turn, so every prediction resolves.
        try:
            pcm = getattr(getattr(self, '_loop', None), '_consequence_model', None)
            if pcm and pcm._pending is not None:
                _emo_pcm  = getattr(self.ai_system, 'emotional_state', None)
                _val_pcm  = "neutral"
                if _emo_pcm:
                    ev = getattr(_emo_pcm, 'overall_valence', 0.0)
                    _val_pcm = "positive" if ev > 0.2 else "negative" if ev < -0.2 else "neutral"
                _state_after = pcm.read_state()
                pcm.record_outcome(_state_after, _val_pcm)
        except Exception:
            pass

        # Drain energy for the activity
        energy_activity = {
            "deep":   "deep_reasoning",
            "normal": "standard_reply",
            "low":    "small_talk",
        }.get(self.energy.mode(), "standard_reply")
        self.energy.drain(energy_activity)

        # v50: record interaction cost in CognitiveResourceEconomy
        try:
            _econ = getattr(getattr(self, '_loop', None), '_resource_economy', None)
            if _econ:
                _emo    = getattr(self.ai_system, 'emotional_state', None)
                _arousal = 0.7 if (_emo and getattr(_emo, 'overall_arousal', 0.5) > 0.65) else 0.4
                _length_factor = min(1.0, len(user_input) / 400)
                _impact  = 0.3 + 0.4 * _arousal + 0.3 * _length_factor
                _econ.record_interaction(impact=_impact)
                # Cost the LLM call with richer model:
                # complexity proxy = input length + question markers + retrieval depth
                _resp_tokens   = max(50, int(len(response.split()) * 1.3))
                _prompt_tokens = max(50, int(len(user_input.split()) * 1.3))
                _complexity    = min(1.0,
                    0.3 * _length_factor
                    + 0.3 * (1.0 if '?' in user_input else 0.0)
                    + 0.4 * _arousal
                )
                _econ.record_llm_call(
                    tokens=_resp_tokens,
                    prompt_tokens=_prompt_tokens,
                    complexity=_complexity,
                )
                # Diagnostic (v59): confirm costs are actually landing —
                # to settle whether "pinned at 1.00 during an active batch"
                # is a restart-recovery artifact or costs silently not
                # applying. Cheap; leave in.
                logger.info(
                    f"[ResourceEconomy] post-interaction cog="
                    f"{_econ.cognitive_energy:.3f} soc={_econ.social_energy:.3f}"
                )
            else:
                logger.warning(
                    "[ResourceEconomy] _econ not resolved — interaction cost "
                    "NOT applied this turn (self._loop._resource_economy missing)"
                )
        except Exception:
            pass

        # Meta-cognitive evaluation
        eval_result = self.meta_cognition.post_response_eval(
            response=response,
            user_input=user_input,
            emotional_state=cycle_data.get("emo_values", {}),
        )

        # Phase 6.16 — closes Phase A's stated gap: symbol absorption was
        # wired for AutonomousReflection (v108) but never for MetaCognition,
        # even though Phase A's own success criterion names both. Real
        # text field used: suggested_improvement (MetaEvaluation's only
        # free-text field) — not the numeric scores, which aren't
        # self-referential prose the absorber's patterns could match.
        if eval_result and eval_result.suggested_improvement:
            try:
                from cognition.symbol_system import get_symbol_system
                from cognition.reflection_absorber import absorb_reflection
                absorb_reflection(
                    eval_result.suggested_improvement, get_symbol_system(self),
                    source="meta_cognition",
                )
            except Exception as e:
                logger.debug(f"[CognitiveOrganism] MetaCognition symbol absorption failed (non-fatal): {e}")

        # ── v34: Record response for emotional suppression detection ─────────
        try:
            self.meta_interrupt.record_response(
                response_text     = response or "",
                emotion_snapshot  = cycle_data.get("emo_values", {}),
            )
        except Exception:
            pass

        # ── Real-time semantic extraction into Global Workspace ──────────────
        try:
            self._semantic_extractor.extract_async(
                user_input  = user_input or "",
                response    = response   or "",
                meta_scores = {
                    "clarity":    getattr(eval_result, "clarity_score",    0.5),
                    "depth":      getattr(eval_result, "depth_score",      0.5),
                    "alignment":  getattr(eval_result, "alignment_score",  0.5),
                    "confidence": getattr(eval_result, "confidence_score", 0.5),
                },
                workspace = self.workspace,
            )
        except Exception:
            pass

        # ── Emit identity events via EventBus ───────────────────────────────
        try:
            from core.cognitive_event_bus import IDENTITY_THREATENED, IDENTITY_AFFIRMED
            sc = getattr(self.ai_system, 'self_concept', None)
            if sc:
                state = getattr(sc, '_state', None)
                if state:
                    coherence = float(getattr(state, 'coherence', 1.0))
                    violations = float(getattr(state, 'violation_ratio', 0.0))
                    if violations > 0.3 or coherence < 0.5:
                        # Rate-limit: emit at most once every 5 minutes
                        import time as _t
                        _now = _t.time()
                        _last = getattr(self, '_last_identity_threatened_emit', 0.0)
                        if _now - _last >= 300:
                            self.event_bus.emit(IDENTITY_THREATENED, {
                                "belief":   "identity coherence",
                                "severity": 1.0 - coherence,
                            }, source="self_concept")
                            self._last_identity_threatened_emit = _now
                    elif coherence > 0.8:
                        self.event_bus.emit(IDENTITY_AFFIRMED, {
                            "coherence": coherence,
                        }, source="self_concept")
        except Exception:
            pass

        # Record drive satisfaction based on response quality
        if decision.dominant_drive == "help_user":
            # Heuristic: longer, structured responses likely more helpful
            word_count = len(response.split())
            satisfaction = min(0.9, 0.5 + word_count / 1000.0)
            self.goal_ecology.record_satisfaction("help_user", satisfaction)

        # Curiosity satisfied if research was done or question was answered
        if "?" in response and self.curiosity.global_level() > 0.5:
            top = self.curiosity.top_topic()
            if top:
                self.curiosity.stimulate(top, amount=0.05, source="user_signal")

        _report = self.homeostasis.last_report()
        logger.debug(
            f"[CognitiveOrganism] Post-interaction complete. "
            f"Energy={self.energy.level():.1f}, "
            + (f"Homeostasis={_report.homeostasis_score:.2f}" if _report else "Homeostasis=n/a")
        )

        # ── v3: Evaluate prediction, push surprise to workspace ───────────
        try:
            from core.cognitive_event_bus import SURPRISE_DETECTED  # import before use
            # FIX: store actual response first so detect_intent_from_response works
            self.predictive_mind.store_response(response)
            result = self.predictive_mind.evaluate({
                'user_text':       user_input,
                'current_emotion': self._read_emotion_state(),
            })
            if result:
                sig = self.predictive_mind.workspace_signal(result)
                if sig:
                    self.workspace.broadcast(
                        source=sig['source'],
                        content=sig['content'],
                        priority=sig['priority'],
                    )
                # Emit on every interaction — bus handlers check magnitude themselves
                if result.error_level > 0.0:
                    prediction_error = result
                    self.event_bus.emit(SURPRISE_DETECTED, {
                        "domain":      result.actual_intent,
                        "error_level": result.error_level,
                    }, source="predictive_mind")
        except Exception:
            pass

        # ── PressureSystem: satiation after interaction ─────────────────────
        try:
            # Every user interaction satiates social + expression to some degree
            self.pressure.record_successful_interaction()
            # Successful interaction also reduces uncertainty — understanding achieved
            self.pressure.satiate("uncertainty", amount=0.25)
            # Prediction error → epistemic pressure
            if 'prediction_error' in locals() and prediction_error is not None:
                if hasattr(prediction_error, 'get'):
                    err_lvl = float(prediction_error.get('error_level', 0))
                    self.pressure.record_prediction_error(err_lvl)
        except Exception:
            pass

        # ── v3: Record significant interaction as narrative chapter ────────
        try:
            word_count = len(response.split())
            # Filter internal LLM prompts: these appear as user_input when the
            # proactive/emotion system passes its internal prompt as an interaction.
            # Recording them as narrative chapters produces entries like
            # "User: You are feeling curiosity. Without naming the emotion..."
            _INTERNAL_MARKERS = (
                "you are feeling", "without naming the emotion", "let it colour",
                "one sentence only", "you just had this inner thought",
                "[what i can currently see", "[vision@", "inner voice",
                "speak naturally", "naturally surface",
            )
            _is_internal = any(
                m in user_input.lower()[:120] for m in _INTERNAL_MARKERS
            )
            if word_count > 25 and not _is_internal:   # skip internal prompts
                snippet = user_input[:80].replace("\n", " ")
                # Differentiate chapter titles by content type
                _lower = user_input.lower()
                _sig   = min(0.7, 0.3 + word_count / 500.0)
                if any(w in _lower for w in ["why", "how", "what is", "explain", "understand"]):
                    _title = "Question explored"
                    _sig   = max(_sig, 0.55)
                elif any(w in _lower for w in ["feel", "think", "believe", "consciousness", "self"]):
                    _title = "Reflective exchange"
                    _sig   = max(_sig, 0.60)
                elif any(w in _lower for w in ["remember", "yesterday", "last time", "you said"]):
                    _title = "Continuity recalled"
                    _sig   = max(_sig, 0.58)
                elif word_count > 150:
                    _title = "Deep conversation"
                    _sig   = max(_sig, 0.65)
                else:
                    _title = "Interaction"
                self.narrative_identity.record_chapter(
                    title=_title,
                    description=f"User: {snippet}",
                    emotion=self._read_emotion_state(),
                    significance=_sig,
                )
            # Auto-milestone salience detection — runs every interaction
            try:
                from core.ai_system import AISystem
                ic = getattr(self.ai_system, '_interaction_count', 0)
                self.narrative_identity.check_auto_milestones(ic)
            except Exception:
                pass
        except Exception:
            pass

        # ── WorldModel: update user profile after every interaction ────────
        try:
            self.world_model.update_from_interaction(
                user_id   = getattr(self, '_current_user_id', 'default'),
                user_text = user_input,
                response  = response,
                emotion   = self._read_emotion_state(),
            )
        except Exception:
            pass

        # ── CognitiveValidator: check response-cognition alignment ─────────
        _validator_score = None
        try:
            val_result = self.cognitive_validator.validate(response)
            _validator_score = float(val_result.overall_score)
            if val_result.misaligned:
                logger.debug(
                    f"[CognitiveOrganism] response misaligned "
                    f"(score={val_result.overall_score:.2f}): {val_result.notes}"
                )
            # Feed low scores as tension signals to AspirationalSelf
            asp = getattr(self, 'aspirational_self', None)
            if asp and val_result.overall_score < 0.55:
                asp.on_interaction_evaluated(
                    domain  = "reasoning",
                    score   = val_result.overall_score,
                    context = (user_input or "")[:80],
                )
        except Exception:
            pass

        # ── Self-correction: learn from explicit corrections asynchronously ─
        # The operation is local and bounded; it never adds an LLM call to chat.
        _correction_result = None
        try:
            _correction_result = self._self_correction.observe(
                user_input=user_input,
                response=response,
                validator_score=_validator_score,
                cycle_id=getattr(self, "_interaction_count", None),
            )
        except Exception as _sc_err:
            logger.debug("[CognitiveOrganism] self-correction failed (non-fatal): %s", _sc_err)

        try:
            self._cognitive_restructuring.observe_turn(_correction_result, _validator_score)
        except Exception as _cr_err:
            logger.debug("[CognitiveOrganism] restructuring observation failed (non-fatal): %s", _cr_err)

        # ── Observatory: record response + tick metrics ────────────────────
        try:
            if self.observatory:
                self.observatory.record_response(response)
                self.observatory.tick()
        except Exception:
            pass

        # ── ArchitectureMonitor: periodic health check ─────────────────────
        try:
            self.arch_monitor.check()
        except Exception:
            pass

        # ── v35: Post-response hooks for sentience enhancement layer ────────

        # Communication calibrator — score this response for clarity/concision
        try:
            _uid = getattr(self, '_current_user_id', 'default')
            self.comm_calibrator.score_response(_uid, response or "")
        except Exception:
            pass

        # Active empathy — record exchange outcome
        try:
            _uid = getattr(self, '_current_user_id', 'default')
            self.active_empathy.record_exchange(_uid, response or "")
        except Exception:
            pass

        # Experiential learning — integrate this exchange
        try:
            self.experiential_learning.integrate(
                user_input  = user_input or "",
                ai_response = response   or "",
                context     = cycle_data,
            )
        except Exception:
            pass

        # Societal awareness — record ripple outcome
        try:
            ripple = getattr(self.societal_awareness, '_pending_ripple', None)
            self.societal_awareness.record_outcome(ripple, response or "")
        except Exception:
            pass

        # Ethical reasoning — record resolution for consistency tracking
        try:
            dilemma = getattr(self.ethical_engine, '_pending_dilemma', None)
            if dilemma:
                self.ethical_engine.record_dilemma_resolved(
                    resolution=response[:80] if response else "",
                    lens="integrated",
                    confidence=0.65,
                )
        except Exception:
            pass

        # ── v36: Refresh self-model moment post-response ──────────────────
        # The binder ran pre-response. After the response, run a lightweight
        # update to capture any state changes that occurred during generation
        # (e.g. emotional shifts from what was expressed).
        try:
            last_moment = getattr(self, 'last_experience', None)
            if last_moment is not None:
                self.self_moment.update(self, last_moment)
                self.workspace.set_self_anchor(
                    self.self_moment.current.to_anchor_content()
                )
        except Exception:
            pass

        # ── v38: Self-revision — prediction error reshapes the self ───────
        # Runs AFTER self-model update so revision has access to current phi.
        # Runs BEFORE the next turn's SelfModelInfluence.apply() so any
        # revision is already in place when the next influence cycle runs.
        try:
            pred_result = getattr(self.predictive_mind, '_last_eval_result', None)
            revision_events = self._self_revision.revise(self, pred_result)
            if revision_events:
                # Write any phi changes back to workspace anchor immediately
                self.workspace.set_self_anchor(
                    self.self_moment.current.to_anchor_content()
                )
                logger.debug(
                    f"[CognitiveOrganism] Self-revision: "
                    f"{[e.revision_type for e in revision_events]}"
                )
        except Exception as e:
            logger.debug(f"[CognitiveOrganism] Self-revision failed (non-fatal): {e}")

        # ── v46: Emotional memory — record significant emotional traces ────
        try:
            _uid = getattr(self, '_current_user_id', 'default')
            self.emotional_memory.record_from_interaction(
                user_input  = user_input or "",
                ai_response = response   or "",
                user_id     = _uid,
            )
        except Exception:
            pass

        # ── v46: Preference — feed curiosity stimulations into preference ──
        try:
            cur_state = getattr(self.curiosity, '_state', None)
            if cur_state:
                for topic, score in list(
                    getattr(cur_state, 'scores', {}).items()
                )[:5]:
                    if score > 0.55:
                        self.preference_engine.on_curiosity_stimulation(
                            topic, amount=score
                        )
        except Exception:
            pass

        # ── v46: Skill registry — learn only from a completed exchange ─────
        # _post_interaction runs in PersonaBridge's executor. observe_interaction
        # adds another short queue boundary, so matching, JSON persistence and
        # optional depth enrichment never sit on the visible response path.
        try:
            # A clear correction/validation in this turn is delayed evidence
            # about the previous answer; neutral follow-ups are never scored.
            self.skill_registry.observe_user_followup(user_input or "")
            self.skill_registry.observe_interaction(
                user_input=user_input or "",
                response=response or "",
            )
        except Exception:
            pass

        # ── v47: Autonomous experiment — record interaction measurement ────
        try:
            loop     = getattr(self, '_loop', None)
            auto_exp = getattr(loop, '_auto_experiment', None) if loop else None
            if auto_exp:
                auto_exp.record_interaction(
                    user_input = user_input or "",
                    response   = response   or "",
                    context    = cycle_data,
                )
        except Exception:
            pass

    def _read_emotion_state(self) -> str:
        """Return a simple emotion label for predictive_mind / narrative_identity."""
        try:
            ems = self.ai_system.emotional_state
            if hasattr(ems, 'get_current_values'):
                vals = ems.get_current_values()
                if vals:
                    dominant = max(vals, key=vals.get)
                    return dominant
            if hasattr(ems, 'valence'):
                v = str(ems.valence).lower()
                if 'pos' in v:   return 'positive'
                if 'neg' in v:   return 'negative'
        except Exception:
            pass
        return 'neutral'

    def _get_emotion_values(self) -> Dict[str, float]:
        """Extract emotion float dict from EmotionalStateManager."""
        try:
            ems = self.ai_system.emotional_state
            # Primary: use get_current_values() which applies decay correctly
            if hasattr(ems, "get_current_values"):
                return ems.get_current_values()
            # Secondary: direct dict access (attr is .emotions, not ._emotions)
            if hasattr(ems, "emotions"):
                return {name: e.value for name, e in ems.emotions.items()}
        except Exception:
            pass
        return {}

    def _get_identity_stability(self) -> float:
        """
        Extract identity stability from SelfConceptSystem.
        SelfConceptState.stability is never updated by the existing code,
        so we use .coherence as the live proxy — it IS updated each turn
        by _update_coherence() based on belief confidence and violation ratio.
        """
        try:
            sc = self.ai_system.self_concept
            # ._state is the correct attr (SelfConceptSystem stores as self._state)
            state = getattr(sc, "_state", None) or getattr(sc, "state", None)
            if state is not None:
                # coherence [0,1] maps directly to identity stability
                return float(state.coherence)
        except Exception:
            pass
        return 0.75

    def _get_contradiction_count(self) -> int:
        """Count unresolved contradictions."""
        try:
            handler = self.ai_system.liberty_contradiction
            if handler and hasattr(handler, "pending_confrontations"):
                return len(handler.pending_confrontations)
        except Exception:
            pass
        return 0

    def _get_goal_satisfaction(self) -> float:
        """Get average goal satisfaction from GoalSystem."""
        try:
            gs = self.ai_system.liberty_goals
            if gs and hasattr(gs, "goals") and gs.goals:
                vals = [g.satisfaction for g in gs.goals.values()]
                return sum(vals) / len(vals)
        except Exception:
            pass
        return 0.6

    def _collect_tension_inputs(
        self,
        emotion_values: Optional[Dict] = None,
        identity_stability: Optional[float] = None,
        contradiction_count: Optional[int] = None,
        goal_satisfaction: Optional[float] = None,
        interaction_recency_seconds: float = 0.0,
    ) -> Dict:
        """Collect tension engine inputs, using live state if not provided."""
        return {
            "emotion_values":              emotion_values or self._get_emotion_values(),
            "identity_stability":          identity_stability if identity_stability is not None
                                           else self._get_identity_stability(),
            "contradiction_count":         contradiction_count if contradiction_count is not None
                                           else self._get_contradiction_count(),
            "goal_satisfaction":           goal_satisfaction if goal_satisfaction is not None
                                           else self._get_goal_satisfaction(),
            "curiosity_global":            self.curiosity.global_level(),
            "interaction_recency_seconds": interaction_recency_seconds,
            # Approximate knowledge gaps from contradiction count — contradictions
            # represent unresolved knowledge conflicts, a good proxy for gaps
            "knowledge_gap_count":         (contradiction_count or self._get_contradiction_count()) * 2,
        }

    def _collect_homeostasis_inputs(
        self,
        emo_values: Optional[Dict] = None,
        identity_stability: Optional[float] = None,
        contradiction_count: Optional[int] = None,
        goal_satisfaction: Optional[float] = None,
    ) -> Dict:
        """Collect homeostasis evaluation inputs."""
        emo = emo_values or self._get_emotion_values()
        # Compute emotional valence: positive emotions minus negative
        valence = (
            emo.get("satisfaction", 0.5) * 0.4
            + emo.get("warmth", 0.5) * 0.3
            + emo.get("enthusiasm", 0.5) * 0.2
            - emo.get("anxiety", 0.0) * 0.3
            - emo.get("frustration", 0.0) * 0.2
        )
        return {
            "identity_stability":       identity_stability if identity_stability is not None
                                        else self._get_identity_stability(),
            "emotional_valence":        max(0.0, min(1.0, 0.5 + valence)),
            "contradiction_count":      contradiction_count if contradiction_count is not None
                                        else self._get_contradiction_count(),
            "goal_satisfaction":        goal_satisfaction if goal_satisfaction is not None
                                        else self._get_goal_satisfaction(),
            "seconds_since_interaction": time.time() - self._last_interaction_ts,
            "energy_level":             self.energy.level(),
        }

    def seconds_since_interaction(self) -> float:
        return time.time() - self._last_interaction_ts

    # ── Inspection / debug ────────────────────────────────────────────────────

    def full_state_summary(self) -> Dict[str, Any]:
        """Return a complete snapshot of all subsystem states."""
        return {
            "energy":         self.energy.summary(),
            "attention":      self.attention.summary(),
            "curiosity":      self.curiosity.summary(),
            "tensions":       self.tension_engine.summary(),
            "semantic_memory": self.semantic_memory.summary() if hasattr(self, 'semantic_memory') else {},
            "goal_ecology":   self.goal_ecology.summary(),
            "homeostasis":    self.homeostasis.summary(),
            "meta_cognition": self.meta_cognition.summary(),
        }
