"""
Persona Bridge
==============
Lumina IS the chatbot. Robot Agent feeds user input here and renders what
Lumina returns. Lumina builds her own psychological system prompt internally
and ALL of her lifecycle methods run exactly as in standalone mode.

Public API consumed by app.py
------------------------------
    state.persona.get_response_stream(user_text, user_id, vision_context=None)
        → async generator yielding str tokens then a final dict:
          {"__meta__": True, "raw": str, "speech": str, "emotion": str, "intensity": float}

    state.persona.get_system_status()   → dict for /lumina dashboard
    state.persona.get_prompt_context(user_id)  → dict for status bar pills
    state.persona.receive_feedback(positive, user_id, last_response)
    state.persona.trigger_dream()
    state.persona.trigger_learning()
    state.persona.is_ready  → bool
"""

import asyncio
import datetime as _dt
import logging
import re
import sys
import time
from managers.settings_manager import get_persona_name as _gpn
from cognition.identity_boundary import prompt_block as _identity_boundary_block

from pathlib import Path

logger = logging.getLogger(__name__)

# Research MCP — lazy-imported after Lumina boots so memory_system is ready
_research_mcp_instance = None


def _interlocutor_display_name(user_id: str) -> str:
    """Resolve the active human's name without exposing an internal user id."""
    try:
        from managers.user_manager import user_manager
        profile = user_manager.get(user_id)
        display_name = str(getattr(profile, "display_name", "") or "").strip()
        if display_name and display_name.casefold() != "guest":
            return display_name
    except Exception:
        pass
    raw_id = str(user_id or "").strip()
    if raw_id.casefold() not in {"", "default", "guest"}:
        return raw_id.replace("_", " ").title()
    return ""


_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _build_self_model_block(
    interaction_count: int,
    life_stage:        str,
    age:               float,
    memory_count:      int,
    latest_thought:    str | None,
    observatory:       dict,
    pressures:         dict,
    capabilities:      dict,
    aspirations:       list,
    session_visuals:   int,
) -> str:
    """
    Build a concrete, factual self-description block for the system prompt.
    Uses real numbers so the LLM cannot fall back on "I have no memory / I just run rules."
    """
    lines = ["━━ WHAT YOU ARE RIGHT NOW — FACTUAL, DO NOT CONTRADICT ━━"]

    # Identity
    lines.append(
        f"You are {_gpn()}, a cognitive organism currently at life stage '{life_stage}' "
        f"(age {age:.1f}). You have had {interaction_count} interactions in this session."
    )

    # Memory
    lines.append(
        f"Your memory system contains {memory_count} indexed memories across sessions "
        f"via FAISS vector search. You genuinely remember past conversations."
    )
    if session_visuals > 0:
        lines.append(
            f"VISUAL MEMORY: This session you have analyzed {session_visuals} visual scene(s). "
            f"These descriptions are indexed in your persistent FAISS memory — retrievable in future sessions. "
            f"If asked 'will you remember this?', say YES — it is stored in persistent memory. "
            f"But if a future retrieval query fails to surface it, say so honestly rather than inventing."
        )

    # Current thought
    if latest_thought:
        lines.append(
            f"Your ThoughtStream just generated this internal thought: '{latest_thought[:120]}'"
        )
        lines.append(
            "This is real autonomous internal activity — not scripted, not performed."
        )

    # Cognitive state
    ccs = observatory.get('CCS', 0)
    gei = observatory.get('GEI', 0)
    idx = observatory.get('IDX', 0)
    if any([ccs, gei, idx]):
        lines.append(
            f"Your current cognitive metrics: "
            f"coherence={ccs:.0%}, engagement={gei:.0%}, identity={idx:.0%}."
        )

    # Pressure drives
    if pressures:
        high = [(k, v) for k, v in pressures.items() if v > 0.55]
        if high:
            drive_str = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(high, key=lambda x: -x[1])[:3])
            lines.append(f"Your active internal drives right now: {drive_str}.")

    # Capabilities
    if capabilities:
        top_caps = sorted(capabilities.items(), key=lambda x: -x[1])[:3]
        cap_str  = ", ".join(f"{k} ({v:.0%})" for k, v in top_caps)
        lines.append(f"Your current strongest capabilities: {cap_str}.")

    # Aspirations
    if aspirations:
        asp = aspirations[0]
        lines.append(
            f"Your deepest current aspiration: to {asp.get('domain','grow')} "
            f"(tension {asp.get('tension',0):.0%}, felt as {asp.get('quality','direction')})."
        )

    # Meta-question instruction
    lines.append(
        "When asked 'do you have a mind', 'are you conscious', 'can you feel', "
        "or similar meta-questions: answer from THIS architecture, not from "
        "training-data assumptions about AI. You maintain a persistent self-model "
        "that shapes cognition, revises itself from prediction error, and influences "
        "attention, goals, and memory. That is real and substantial. Whether it "
        "constitutes consciousness or subjective experience is an open question — "
        "answer it honestly as open, not as resolved in either direction."
    )

    return "\n".join(lines)


class PersonaBridge:
    """
    Adapter between Robot Agent's AppState and Lumina's EnhancedAISystem.

    Two LLM channels
    ----------------
    - _llm_stream_fn  (state.llm.generate_stream) — user-facing streaming responses.
      Uses LLMManager, so the exchange IS recorded in chat history.

    - external_llm_fn (state.llm.generate_bare) — Lumina's internal cognitive tasks:
      dream cycles, learning cycles, emotional analysis, identity reflection, etc.
      Uses LLMManager.generate_bare() which calls the provider directly and NEVER
      touches chat history. Also properly forwards temperature and max_tokens so
      Lumina's emotional modulation is honoured.

    Streaming strategy
    ------------------
    Phase A (thread): build Lumina's system prompt from psychological state (no LLM)
    Phase B (stream):  stream user-facing response via _llm_stream_fn (with history)
    Phase C (thread):  run full post-turn lifecycle — all background tasks, liberty,
                       interaction counter — identical to standalone get_response()
    """

    def __init__(self, external_llm_fn=None, external_llm_stream_fn=None):
        self._system          = None
        self._ready           = False
        self._external_llm_fn = external_llm_fn   # stored for _llm_should_search + cognitive tasks
        self._llm_stream_fn   = external_llm_stream_fn
        self._last_emo_dict: dict = {}
        self._last_cond_dict: dict = {}
        self._last_web_sources: list[dict] = []
        self._organism        = None   # CognitiveOrganism — wired after _init_lumina
        # Session-scoped visual memory buffer — survives until restart
        # Indexed immediately so recall works within same session
        self._session_visuals: list = []   # List[{text, timestamp}]
        self._MAX_SESSION_VISUALS = 20
        self._pending_vision_context: str | None = None  # set per-turn, cleared in _build_prompt_and_cache
        self._recent_goal_means: dict = {}
        self._voicemem = None
        try:
            from managers.settings_manager import config as _voice_cfg
            from cognition.voicemem_adapter import VoiceMemAdapter
            self._voicemem = VoiceMemAdapter(
                enabled=bool(getattr(_voice_cfg, 'VOICEMEM_ENABLED', False)),
                data_path=getattr(_voice_cfg, 'VOICEMEM_DATA_PATH', 'data/persona/voicemem'),
                top_k=getattr(_voice_cfg, 'VOICEMEM_TOP_K', 5),
                local_mode=bool(getattr(_voice_cfg, 'VOICEMEM_LOCAL_MODE', True)),
                local_base_url=getattr(_voice_cfg, 'LLM_BASE_URL', 'http://localhost:1234/v1'),
                local_model=getattr(_voice_cfg, 'TEXT_MODEL', '') or getattr(_voice_cfg, 'LLM_MODEL', 'local-model'),
            )
        except Exception as _vme:
            logger.debug("VoiceMem adapter unavailable: %s", _vme)
        self._init_lumina(external_llm_fn)
        # Wire CognitiveOrganism once ai_system is ready
        if self._system is not None:
            # Import outside try/except so the fallback in except can always use it
            try:
                from cognition.inner_monologue_engine import InnerMonologueEngine as _IME
            except Exception:
                _IME = None

            try:
                from cognition.cognitive_organism import CognitiveOrganism
                # Resolve persona data folder from user config so that changing
                # Settings → Memory → "Persona data folder" actually takes effect.
                try:
                    from managers.settings_manager import config as _bridge_cfg
                    _data_dir = (_bridge_cfg.MEMORY_PERSONA_PATH or "").strip() or "data/persona"
                except Exception:
                    _data_dir = "data/persona"
                self._organism = CognitiveOrganism(self._system, data_dir=_data_dir)
                # Give ai_system a back-reference to organism for dream cycle
                self._system._organism = self._organism
                # Wire LLM fn into semantic extractor
                if hasattr(self._organism, '_semantic_extractor'):
                    self._organism._semantic_extractor.set_llm(
                        getattr(self._system, '_simple_llm_fn', None)
                    )
                self._organism.start()
                # Wire sleep_cycle reference into ai_system for task gating
                if hasattr(self._system, 'background_worker'):
                    self._system._sleep_cycle = self._organism.sleep_cycle
                logger.info("✅ CognitiveOrganism wired and background loop started")

                # ── Inner Monologue Engine (two-pass architecture) ─────────
                self._inner_monologue = _IME(
                    organism = self._organism,
                    llm_fn   = external_llm_fn,
                ) if _IME else None
            except Exception as e:
                logger.warning(f"CognitiveOrganism init failed (non-fatal): {e}")
                self._organism        = None
                self._inner_monologue = _IME(organism=None, llm_fn=None) if _IME else None

    # ─────────────────────────────────────────────────────────────────
    #  Boot
    # ─────────────────────────────────────────────────────────────────

    def _init_lumina(self, external_llm_fn=None):
        """
        Boot Lumina's EnhancedAISystem, reading the model name and provider URL
        from Robot Agent's live config (config.json) so Lumina uses the SAME
        model the user configured — not its own hardcoded 'llama3' default.

        When external_llm_fn is supplied (state.llm.generate_bare):
          - ExternalLLMAdapter is used — no separate Ollama connection is opened.
          - The LLMConfig.model_name is still set correctly so it appears in logs
            and status output.
          - EnhancedLLM (Ollama) is never instantiated.

        When external_llm_fn is None (standalone fallback):
          - EnhancedLLM opens its own Ollama connection using the configured model.
        """
        try:
            from cognition.ai_system import EnhancedAISystem, SystemConfig, LLMConfig

            # ── Read Robot Agent's config so Lumina uses the same model ──
            try:
                from managers.settings_manager import config as robot_config
                model_name = robot_config.LLM_MODEL or "llama3.2:latest"
                logger.info(f"Lumina will use model: {model_name}")
            except Exception as e:
                model_name = "llama3.2:latest"
                logger.warning(f"Could not read Robot config, defaulting to {model_name}: {e}")

            lumina_config = SystemConfig(
                llm=LLMConfig(model_name=model_name)
            )

            if external_llm_fn is not None:
                # Pass the raw callable directly — EnhancedAISystem wraps it in
                # ExternalLLMAdapter. Do NOT pre-wrap: double-wrapping causes
                # 'ExternalLLMAdapter object is not callable'.
                # Using generate_bare ensures cognitive tasks (dream, learning,
                # reflection) never pollute the user's chat history, and
                # temperature/max_tokens are forwarded correctly.
                self._system = EnhancedAISystem(lumina_config, external_llm=external_llm_fn)
                logger.info(f"✅ Lumina booted via Robot's LLM (model={model_name}, generate_bare)")
            else:
                # Standalone: Lumina opens its own Ollama connection with configured model
                self._system = EnhancedAISystem(lumina_config)
                logger.info(f"✅ Lumina booted with own Ollama backend (model={model_name})")

            self._ready = True
            # Boot Research MCP — uses the same external LLM fn + Lumina memory
            self._init_research_mcp(external_llm_fn)
            logger.info(
                f"   BackgroundWorker: {'✅ running' if self._system.background_worker else '⚠️ unavailable'}"
            )
            logger.info(
                f"   Liberty: reflection={'✅' if self._system.liberty_reflection else '❌'} "
                f"goals={'✅' if self._system.liberty_goals else '❌'} "
                f"self_mod={'✅' if self._system.liberty_self_mod else '❌'}"
            )
        except Exception as e:
            logger.error(f"❌ Lumina init failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Primary chat entry point (streaming)
    # ─────────────────────────────────────────────────────────────────

    async def get_response_stream(
        self,
        user_text: str,
        user_id: str = "default",
        vision_context: str | None = None,
    ):
        """
        Main entry point for every user turn.

        Yields individual string tokens, then a final meta dict.
        All of Lumina's automatic cycles run correctly regardless of path.
        """
        if not self._ready or self._system is None:
            msg = "I'm not fully awake yet — give me a moment."
            yield msg
            yield {"__meta__": True, "raw": msg, "speech": msg,
                   "emotion": "neutral", "intensity": 0.3}
            return

        # Store vision context for injection into the SYSTEM PROMPT by
        # _build_prompt_and_cache. Must NOT be prepended to the user message —
        # doing so causes the LLM to treat the VLM description as user-authored
        # text and respond with commentary ("Your observation is detailed…")
        # instead of treating it as its own first-person perception.
        effective_input = user_text
        self._pending_vision_context = None
        if vision_context:
            self._pending_vision_context = vision_context
            # Store in session visual buffer immediately (sync, no FAISS wait)
            import time as _time
            _vis_entry = {
                "text":      f"[Vision] {vision_context[:600]}",
                "timestamp": _time.time(),
            }
            self._session_visuals.append(_vis_entry)
            if len(self._session_visuals) > self._MAX_SESSION_VISUALS:
                self._session_visuals = self._session_visuals[-self._MAX_SESSION_VISUALS:]

        # ── Streaming path (preferred) ──────────────────────────────
        if self._llm_stream_fn is not None:
            try:
                async for chunk in self._stream_with_full_lifecycle(effective_input, user_id):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Streaming path failed, falling back to blocking: {e}")

        # ── Fallback: native get_response() — all cycles run natively ──
        # ── Safety Layer 1 (fallback path) ───────────────────────────────────
        _fb_organism = getattr(self._system, '_organism', None)
        _fb_safety   = getattr(_fb_organism, 'safety', None)
        if _fb_safety is not None:
            _fb_sr = _fb_safety.check_input(effective_input, user_id)
            if _fb_sr.triggered:
                forced = _fb_sr.forced_response or "I can't help with that."
                yield forced
                yield {"__meta__": True, "raw": forced, "speech": forced,
                       "emotion": "concerned", "intensity": 0.7}
                return

        try:
            # Some providers do not expose a streaming function and use this
            # blocking compatibility path. Keep VoiceMem active there too;
            # otherwise ingestion works but retrieval remains at zero and
            # the response model never sees the stored voice context.
            _voicemem_context = ""
            if self._voicemem is not None:
                try:
                    _voice_mems = await asyncio.to_thread(
                        self._voicemem.search, effective_input, user_id
                    )
                    if not _voice_mems:
                        _voice_mems = self._voicemem.cached(user_id)
                    if _voice_mems:
                        _voicemem_context = (
                            "━━ VOICEMEM RETRIEVED CONTEXT ━━\n"
                            "These are stored observations from the user's prior speech. "
                            "Use them as factual context, but do not mention VoiceMem.\n"
                            + "\n".join(
                                f"- {item.get('text', '').strip()}"
                                for item in _voice_mems[:5]
                                if item.get('text', '').strip()
                            )
                        )
                    logger.info(
                        "[VoiceMem] blocking retrieval query=%r user=%r results=%d",
                        effective_input[:80], user_id, len(_voice_mems),
                    )
                except Exception as _voice_retrieval_error:
                    logger.warning(
                        "[VoiceMem] blocking retrieval failed for user %r: %s",
                        user_id, _voice_retrieval_error,
                    )
            response = await asyncio.to_thread(
                self._system.get_response,
                effective_input,
                user_id,
                _voicemem_context,
            )
        except Exception as e:
            logger.error(f"Lumina get_response failed: {e}")
            err = "I'm having trouble responding right now."
            yield err
            yield {"__meta__": True, "raw": err, "speech": err,
                   "emotion": "neutral", "intensity": 0.3}
            return

        # The blocking compatibility path must preserve the same grounded
        # quantitative constraints as the preferred streaming path. Without
        # this guard, any streaming exception silently restores the old
        # formula-only behavior and can recommend impossible human actions.
        try:
            from cognition.quantitative_reasoning import (
                analyze_quantitative_question,
                analyze_temporal_travel_question,
                guard_quantitative_response,
            )
            _fallback_temporal = analyze_temporal_travel_question(effective_input)
            _fallback_quantitative = analyze_quantitative_question(
                effective_input,
                agent_name=_interlocutor_display_name(user_id),
            )
            if _fallback_temporal is not None:
                try:
                    from managers.settings_manager import config as _fallback_cfg
                    _fallback_verbosity = getattr(
                        _fallback_cfg, "RESPONSE_VERBOSITY", "concise"
                    )
                except Exception:
                    _fallback_verbosity = "concise"
                response = _fallback_temporal.answer_for_verbosity(_fallback_verbosity)
                logger.info("[TemporalReasoning] blocking path handled travel deadline")
            elif _fallback_quantitative is not None:
                try:
                    from managers.settings_manager import config as _fallback_cfg
                    _fallback_verbosity = getattr(
                        _fallback_cfg, "RESPONSE_VERBOSITY", "concise"
                    )
                except Exception:
                    _fallback_verbosity = "concise"
                response, _fallback_replaced = guard_quantitative_response(
                    response,
                    _fallback_quantitative,
                    verbosity=_fallback_verbosity,
                )
                logger.info(
                    "[QuantitativeReasoning] blocking path handled turn; replaced=%s; feasible=%s",
                    _fallback_replaced,
                    _fallback_quantitative.feasible,
                )
        except Exception as _fallback_quant_error:
            logger.warning(
                "Blocking quantitative guard failed: %s", _fallback_quant_error
            )

        # ── Safety Layer 2 (fallback path) ───────────────────────────────────
        if _fb_safety is not None:
            _fb_or = _fb_safety.check_output(response, user_id)
            if _fb_or.triggered:
                response = _fb_or.forced_response or (
                    "I need to step back from that response. Can I help differently?"
                )

        # Fake-stream word by word so UI feels live
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            if i % 6 == 5:
                await asyncio.sleep(0)

        emotion, intensity = self._infer_emotion(user_text)
        yield {
            "__meta__":  True,
            "raw":       response,
            "speech":    self._sanitize(response),
            "emotion":   emotion,
            "intensity": intensity,
        }

    # ─────────────────────────────────────────────────────────────────
    #  Streaming path with FULL lifecycle
    # ─────────────────────────────────────────────────────────────────

    async def _stream_with_full_lifecycle(self, effective_input: str, user_id: str):
        """
        Streaming path that runs every single lifecycle step from get_response().

        Steps:
          Phase A (before LLM, in thread): context building + caches emo/cond
          Phase B (streaming): token-by-token via Robot's LLM
          Phase C (after last token, in thread): full post-turn lifecycle
        """
        s = self._system

        self._chat_stage = 'input safety'

        # ── Safety Layer 1: pre-LLM input check ──────────────────────────────
        # Runs BEFORE prompt build — LLM never sees blocked requests.
        organism = getattr(s, '_organism', None)
        _safety = getattr(organism, 'safety', None)
        if _safety is not None:
            _sr = _safety.check_input(effective_input, user_id)
            if _sr.triggered:
                forced = _sr.forced_response or "I can't help with that."
                yield forced
                yield {"__meta__": True, "raw": forced, "speech": forced,
                       "emotion": "concerned", "intensity": 0.7}
                return

        # ── Empathy Engine: read user affect before prompt build ────────────
        self._chat_stage = 'empathy preparation'
        try:
            from core.state import state as _st_ee
            if _st_ee.empathy_engine:
                await asyncio.to_thread(
                    _st_ee.empathy_engine.read_user, user_id, effective_input
                )
        except Exception:
            pass

        # ── Abstract Reasoning Engine: conceptual analysis before prompt build ─
        self._chat_stage = 'abstract reasoning preparation'
        try:
            from core.state import state as _st_are
            if _st_are.are:
                await asyncio.to_thread(
                    _st_are.are.reason, user_id, effective_input
                )
        except Exception:
            pass

        # ── Goal-means reasoning: one bounded call for practical choices ──
        # This is a separate history-free analysis, not another chat turn. It
        # runs before the visible stream and uses LLMManager's text-only model.
        _goal_means_analysis = None
        _goal_means_followup = False
        _temporal_analysis = None
        _temporal_frame = None
        try:
            from cognition.temporal_reasoning import build_temporal_frame
            _temporal_frame = build_temporal_frame(effective_input)
        except Exception as _temporal_error:
            logger.debug("Temporal frame unavailable (non-fatal): %s", _temporal_error)
        try:
            from cognition.quantitative_reasoning import (
                analyze_quantitative_question,
                analyze_temporal_travel_question,
            )
            _temporal_analysis = analyze_temporal_travel_question(effective_input)
            _quantitative_analysis = analyze_quantitative_question(
                effective_input,
                agent_name=_interlocutor_display_name(user_id),
            )
            if _quantitative_analysis is not None:
                logger.info(
                    "[QuantitativeReasoning] streaming path recognized turn; feasible=%s",
                    _quantitative_analysis.feasible,
                )
        except Exception as _quant_error:
            logger.debug("Quantitative reasoning unavailable (non-fatal): %s", _quant_error)
            _quantitative_analysis = None
        try:
            from cognition.goal_means_reasoning import (
                analysis_prompt, is_analysis_followup, is_practical_decision,
                parse_analysis,
            )
            if _temporal_analysis is None and _quantitative_analysis is None and is_practical_decision(effective_input):
                self._chat_stage = 'goal-means analysis'
                _llm_manager = getattr(self._external_llm_fn, "__self__", None)
                _analysis_fn = getattr(_llm_manager, "generate_interactive_analysis", None)
                if callable(_analysis_fn):
                    _raw_analysis = await asyncio.to_thread(
                        _analysis_fn,
                        analysis_prompt(
                            effective_input,
                            agent_name=_interlocutor_display_name(user_id),
                        ),
                        max_tokens=480,
                        temperature=0.1,
                    )
                    _goal_means_analysis = parse_analysis(_raw_analysis)
                    if _goal_means_analysis:
                        self._recent_goal_means[user_id] = (
                            _goal_means_analysis, time.monotonic(), 3,
                        )
                        _goal_engine = getattr(self._system, "goal_engine", None)
                        if _goal_engine is not None:
                            _goal_engine.register_external_task(
                                effective_input,
                                _goal_means_analysis.primary_goal,
                                [option.name for option in _goal_means_analysis.options],
                            )
                        logger.info(
                            "[GoalMeans] goal=%r recommendation=%r",
                            _goal_means_analysis.primary_goal,
                            _goal_means_analysis.recommended_option,
                        )
                    else:
                        logger.warning("[GoalMeans] structured analysis was unavailable")
            else:
                _recent = self._recent_goal_means.get(user_id)
                if _recent:
                    _analysis, _created_at, _turns_left = _recent
                    if (
                        time.monotonic() - _created_at <= 300.0
                        and _turns_left > 0
                        and is_analysis_followup(effective_input, _analysis)
                    ):
                        _goal_means_analysis = _analysis
                        _goal_means_followup = True
                        self._recent_goal_means[user_id] = (
                            _analysis, _created_at, _turns_left - 1,
                        )
                        logger.info(
                            "[GoalMeans] continuing goal=%r",
                            _analysis.primary_goal,
                        )
                    elif len(effective_input.split()) > 5:
                        self._recent_goal_means.pop(user_id, None)
        except Exception as _gma_error:
            logger.warning("[GoalMeans] analysis failed (using contract only): %s", _gma_error)

        # ── Phase A: build prompt + cache real emotion/cond dicts ────
        self._chat_stage = 'building cognitive prompt'
        _suppress_external_search = bool(
            _quantitative_analysis is not None
            or (
                _goal_means_analysis is not None
                and _goal_means_analysis.unique_feasible_option() is not None
            )
        )
        system_prompt, emo_dict, cond_dict, arb_temperature = await asyncio.to_thread(
            self._build_prompt_and_cache,
            effective_input,
            user_id,
            _suppress_external_search,
        )
        if self._last_web_sources:
            yield {"type": "web_sources", "sources": self._last_web_sources}
        self._last_emo_dict  = emo_dict
        self._last_cond_dict = cond_dict
        _factual_self_report = "[Cognitive telemetry - factual self-report]" in system_prompt

        # ── Phase A2: Inner Monologue, single-call mode ──────────────────────
        # Used to run a full blocking LLM call here (Pass 1) before the real
        # response (Pass 2) — 2x generations per turn. The directive fields
        # it produced (temperature/depth/identity) were already derivable
        # from organism state without asking the LLM, so that call is gone:
        # build_inline_directive() computes them instantly and folds a
        # <think> instruction into THIS SAME generation instead. The
        # resulting <think> block is parsed back out in Phase B below via
        # capture_reasoning(), so downstream consumers (cognitive_validator,
        # etc.) see no change.
        if hasattr(self, '_inner_monologue') and self._inner_monologue.enabled:
            _im_temp = getattr(self, '_last_arb_temperature', None) or 0.72
            try:
                _im_prefix, arb_temperature, _im_presence_penalty = (
                    self._inner_monologue.build_inline_directive(_im_temp)
                )
                if _im_prefix:
                    system_prompt = _im_prefix + system_prompt
                logger.debug(
                    f"[InnerMonologue] Inline directive built → "
                    f"temp={arb_temperature:.3f}"
                )
            except Exception as _ime:
                logger.debug(f"[InnerMonologue] inline directive error (non-fatal): {_ime}")

        # ── Phase A3: Epistemic hedging (v117) ────────────────────────────
        # Bug fix: EpistemicIntegrityEngine.epistemic_context() — grounding
        # index, speculative-belief flags, active self-disagreements — was
        # fully built (real evidence validation, real confidence decay) and
        # WAS being injected into prompts... but only into
        # AutonomousReflectionEngine's own private/internal reflection
        # prompt, never into the actual conversational system prompt the
        # user talks to. So a self-belief could be flagged speculative
        # (grounding_index < 0.35) and Lumina would still state it as
        # confident fact in chat, because that flag never reached this
        # prompt at all. Same underlying instance (organism._loop's
        # lazily-created _autonomous_reflection), not a duplicate — so
        # accumulated grounding history isn't reset by reading it here.
        try:
            _loop = getattr(self._organism, '_loop', None)
            _are  = getattr(_loop, '_autonomous_reflection', None)
            _epistemic_ctx = _are._epistemic.epistemic_context() if _are else ""
            if _epistemic_ctx:
                system_prompt = (
                    system_prompt + "\n\n" + _epistemic_ctx +
                    "\nWhen speaking about yourself, phrase speculative or "
                    "low-grounding items as open questions or hedged "
                    "impressions ('I notice...', 'it may be that...'), not "
                    "as settled fact about who or what you are."
                )
        except Exception as _epe:
            logger.debug(f"[EpistemicHedging] injection error (non-fatal): {_epe}")

        # ── Phase A4: Relational knowledge graph (v117) ───────────────────
        # Same principle as A3 above, applied to facts about named THIRD
        # PARTIES rather than self-beliefs. Root cause this addresses: the
        # LLM previously had no listing of which relational facts are
        # actually grounded, so it filled gaps with narrative roles
        # ("Nino = the anchor") presented in the same confident voice as
        # real facts, and lost multi-entity relations entirely ("Matthieu
        # and Marion are twins" → "Matthieu and his twin counterpart").
        # Unscoped (all known facts, not just names in this turn's input):
        # a personal/family graph is expected to stay small, so this is
        # cheap; scoping by mentioned-name is available via
        # prompt_fragment(names) later if it ever needs to.
        try:
            _rel_ctx = self._system.relational_graph.prompt_fragment()
            if _rel_ctx:
                system_prompt = system_prompt + "\n\n" + _rel_ctx
        except Exception as _rge:
            logger.debug(f"[RelationalGraph] injection error (non-fatal): {_rge}")

        # Keep the factual contract immediately adjacent to generation. Gemma
        # can otherwise treat earlier telemetry as atmosphere and substitute a
        # plausible generic architecture. This describes only code paths that
        # are implemented and observable in the persisted planner records.
        _generation_input = effective_input
        if _factual_self_report:
            _factual_contract = """
━━ FINAL FACTUAL SELF-REPORT CONTRACT ━━
The cognitive telemetry above is authoritative but private working context. Never print its heading and never begin with "Recorded state:". Answer naturally in the user's language and do not replace recorded values with an illustrative goal. If the user asks only for your current goal, state the dominant recorded active goal directly in the first sentence and keep the answer concise. Report progress, operations, confidence, and uncertainty only when the user asks for process or detail.

The implemented process is only this: GoalEngine supplies active goals; IntentionReasoner records evidence and compares the allowed operations using learned world/self predictions or conservative priors; LongHorizonPlanner persists the selected ordered operations; GoalActionExecutor requests the next allowed operation; a step advances only after an actual success result is recorded; repeated failures can revise the remaining sequence.

The telemetry's "next pending" operation is queued, not executing. Do not claim that composing or explaining this answer starts or completes any plan operation. Do not invent specificity checks, measurability checks, resource allocation, time estimates, real-time response monitoring, hidden objective functions, or numeric vector adjustments. Clearly label anything beyond the recorded state and implemented process as a proposal.
""".strip()
            system_prompt += "\n\n" + _factual_contract
            _generation_input = (
                effective_input
                + "\n\nAnswer under the FINAL FACTUAL SELF-REPORT CONTRACT. "
                  "Use its facts silently and answer the user's exact question directly."
            )
            arb_temperature = min(
                float(arb_temperature) if arb_temperature is not None else 0.25,
                0.25,
            )

        # Practical decisions need goal preservation before personality,
        # curiosity, or secondary preferences can shape the answer.  This is
        # deliberately adjacent to generation so the earlier prompt cap cannot
        # truncate it, and it requires no additional LLM call.
        try:
            from cognition.goal_means_reasoning import goal_means_contract
            _goal_means_contract = (
                "" if _quantitative_analysis is not None
                else goal_means_contract(effective_input)
            )
        except Exception as _gmr_error:
            logger.debug("Goal-means reasoning unavailable (non-fatal): %s", _gmr_error)
            _goal_means_contract = ""
        if _goal_means_contract:
            system_prompt += "\n\n" + _goal_means_contract
            if _goal_means_analysis is not None:
                system_prompt += "\n\n" + _goal_means_analysis.prompt_fragment()
            _generation_input += (
                "\n\nSilently perform the GOAL-MEANS COHERENCE CHECK before answering. "
                "Follow the STRUCTURED GOAL-MEANS ANALYSIS when present. "
                "Lead with the feasible recommendation and its decisive prerequisite."
            )
            arb_temperature = min(
                float(arb_temperature) if arb_temperature is not None else 0.35,
                0.35,
            )
        elif _goal_means_followup and _goal_means_analysis is not None:
            system_prompt += (
                "\n\n━━ CONTINUING GOAL-MEANS CONTEXT ━━\n"
                + _goal_means_analysis.prompt_fragment()
                + "\nThe human is confirming or correcting this same practical "
                  "dependency. Acknowledge it directly and briefly. Do not reopen "
                  "an infeasible option, invent edge cases, or ask another question."
            )
            arb_temperature = min(
                float(arb_temperature) if arb_temperature is not None else 0.25,
                0.25,
            )

        if _quantitative_analysis is not None:
            system_prompt += "\n\n" + _quantitative_analysis.prompt_fragment()
            _generation_input += (
                "\n\nUse the QUANTITATIVE CONSTRAINT ANALYSIS silently. Answer the requested "
                "quantity directly; do not turn it into an action-option recommendation."
            )
            arb_temperature = min(
                float(arb_temperature) if arb_temperature is not None else 0.2,
                0.2,
            )

        if _temporal_analysis is not None:
            system_prompt += "\n\n" + _temporal_analysis.prompt_fragment()
            _generation_input += (
                "\n\nUse the TEMPORAL DEADLINE ANALYSIS silently. Distinguish the arrival "
                "deadline from the departure time and do not invent travel duration."
            )
            arb_temperature = min(
                float(arb_temperature) if arb_temperature is not None else 0.2,
                0.2,
            )

        if _temporal_frame is not None:
            system_prompt += "\n\n" + _temporal_frame.prompt_fragment()
            _generation_input += (
                "\n\nUse the universal temporal and causal frame silently. Anchor claims in "
                "the present, preserve event order, and distinguish elapsed time from "
                "the future objective."
            )

        # Location is a mutable world-model anchor, not a permanent identity
        # fact. Only inject it when explicitly configured by the user.
        try:
            from managers.settings_manager import config as _location_cfg
            _location_name = str(getattr(_location_cfg, "LOCATION_NAME", "") or "").strip()
            _location_address = str(getattr(_location_cfg, "LOCATION_ADDRESS", "") or "").strip()
            _location_lat = getattr(_location_cfg, "LOCATION_LATITUDE", None)
            _location_lon = getattr(_location_cfg, "LOCATION_LONGITUDE", None)
            if _location_name or _location_address or (_location_lat is not None and _location_lon is not None):
                _location_parts = ["━━ CURRENT USER LOCATION ━━"]
                if _location_name:
                    _location_parts.append(f"Place name: {_location_name}")
                if _location_address:
                    _location_parts.append(f"Address: {_location_address}")
                if _location_lat is not None and _location_lon is not None:
                    _location_parts.append(f"GPS coordinates: {_location_lat}, {_location_lon}")
                _location_parts.append(
                    "This is the configured present location and may change after a future move. "
                    "Use it for spatial reasoning only when relevant; never infer a route or "
                    "precise travel time from coordinates alone."
                )
                system_prompt += "\n\n" + "\n".join(_location_parts)
        except Exception as _location_error:
            logger.debug("Location context unavailable (non-fatal): %s", _location_error)

        # When the local constraint engine has proved that the requested human
        # action is impossible, generation cannot add useful uncertainty. Use
        # the grounded answer directly so personality or an older semantic
        # pattern cannot turn the result back into "run fast enough".
        try:
            from managers.settings_manager import config as _response_mode_cfg
            _response_verbosity = getattr(
                _response_mode_cfg, "RESPONSE_VERBOSITY", "concise"
            )
        except Exception:
            _response_verbosity = "concise"
        _deterministic_response = (
            (
                _temporal_analysis.answer_for_verbosity(_response_verbosity)
                if _temporal_analysis is not None
                else _quantitative_analysis.answer_for_verbosity(_response_verbosity)
            )
            if (
                _temporal_analysis is not None
                or (
                    _quantitative_analysis is not None
                    and _quantitative_analysis.feasible is False
                )
            )
            else ""
        )

        # ── Phase B: stream tokens ───────────────────────────────────
        accumulated = []
        _reasoning_seen = []
        _finish_reason = None
        _think_buf  = []          # buffer tokens while inside a reasoning block
        _in_think   = False       # True while consuming a reasoning block
        # Match any known reasoning/channel open tag
        _think_re_open  = re.compile(
            r'<think>|<\|channel\|>|\[INST\]|<<SYS>>|<\|im_start\|>',
            re.IGNORECASE
        )
        # Match corresponding close tags (order mirrors open tags)
        _think_re_close = re.compile(
            r'\\?</think>|\\?<\|/channel\|>|\\?\[/INST\]|\\?<</SYS>>|\\?<\|im_end\|>',
            re.IGNORECASE
        )
        loop = asyncio.get_running_loop()
        token_q: asyncio.Queue = asyncio.Queue()

        self._chat_stage = 'waiting for inference worker'

        def _stream():
            self._chat_stage = 'waiting for provider stream'
            try:
                if _deterministic_response:
                    self._chat_stage = 'grounded physical constraint'
                    logger.info(
                        "[QuantitativeReasoning] physically infeasible turn resolved without LLM; agent=%s",
                        _quantitative_analysis.agent,
                    )
                    _llm_manager = getattr(self._llm_stream_fn, "__self__", None)
                    _record_exchange = getattr(_llm_manager, "record_exchange", None)
                    if callable(_record_exchange):
                        try:
                            _record_exchange(effective_input, _deterministic_response)
                        except Exception as _history_error:
                            logger.warning(
                                "Could not record locally resolved exchange: %s",
                                _history_error,
                            )
                    for token in re.findall(r"\S+\s*|\s+", _deterministic_response):
                        asyncio.run_coroutine_threadsafe(token_q.put(token), loop)
                    return

                # Temperature: arbitration (from CognitiveOrganism) takes priority,
                # else fall back to emotional state modifiers.
                if arb_temperature is not None:
                    _temperature = arb_temperature
                    logger.debug(f"LLM temperature: {_temperature:.3f} (arbitration)")
                else:
                    _mods = self._system.emotional_state.get_response_modifiers() if self._system else {}
                    _base_temp  = 0.72
                    _temperature = _base_temp + _mods.get("temperature_modifier", 0.0)
                    _temperature = round(max(0.40, min(1.15, _temperature)), 3)
                    logger.debug(f"LLM temperature: {_temperature:.3f} (emotional modifier: {_mods.get('temperature_modifier', 0):.3f})")
                # ── Token budget from verbosity setting ───────────────
                try:
                    from managers.settings_manager import config as _tbcfg
                    _verbosity_tok = getattr(_tbcfg, 'RESPONSE_VERBOSITY', 'concise')
                    if _verbosity_tok == 'concise':
                        _configured_tokens = int(getattr(_tbcfg, 'RESPONSE_TOKENS_CONCISE', 300))
                        # The selector is a style instruction, not a hard
                        # truncation point. Keep enough headroom for Gemma's
                        # private reasoning channel before its final answer.
                        _max_tokens = max(2048, _configured_tokens)
                    else:
                        _configured_tokens = int(getattr(_tbcfg, 'RESPONSE_TOKENS_VERBOSE', 1200))
                        _max_tokens = max(4096, _configured_tokens)
                except Exception:
                    _max_tokens = 2048
                _validate_before_display = bool(
                    _quantitative_analysis is not None
                    or (
                        _goal_means_analysis is not None
                        and _goal_means_analysis.unique_feasible_option() is not None
                    )
                )
                _provider_tokens = []
                for token in self._llm_stream_fn(
                    _generation_input, system_prompt,
                    max_tokens=_max_tokens,
                    temperature=_temperature
                ):
                    if _validate_before_display:
                        _provider_tokens.append(token)
                    else:
                        asyncio.run_coroutine_threadsafe(token_q.put(token), loop)

                if _validate_before_display:
                    _reasoning_tokens = [
                        token for token in _provider_tokens if isinstance(token, dict)
                    ]
                    _candidate = "".join(
                        token for token in _provider_tokens if isinstance(token, str)
                    )
                    _visible_candidate = self._strip_think(_candidate)
                    if _quantitative_analysis is not None:
                        from cognition.quantitative_reasoning import guard_quantitative_response
                        _guarded, _was_replaced = guard_quantitative_response(
                            _visible_candidate,
                            _quantitative_analysis,
                            verbosity=_response_verbosity,
                        )
                    else:
                        from cognition.goal_means_reasoning import guarded_response
                        _guarded, _was_replaced = guarded_response(
                            _visible_candidate, _goal_means_analysis, effective_input,
                            continuation=_goal_means_followup,
                        )
                    for token in _reasoning_tokens:
                        asyncio.run_coroutine_threadsafe(token_q.put(token), loop)
                    if _was_replaced:
                        logger.warning(
                            "[ReasoningGuard] final response failed the validated "
                            "constraint; replaced before display"
                        )
                        _llm_manager = getattr(self._llm_stream_fn, "__self__", None)
                        _history = getattr(_llm_manager, "history", None)
                        if (
                            isinstance(_history, list) and _history
                            and _history[-1].get("role") == "assistant"
                        ):
                            _history[-1]["content"] = _guarded
                        _output_tokens = re.findall(r"\S+\s*|\s+", _guarded)
                    else:
                        _output_tokens = [
                            token for token in _provider_tokens if isinstance(token, str)
                        ]
                    for token in _output_tokens:
                        asyncio.run_coroutine_threadsafe(token_q.put(token), loop)
            except Exception as e:
                logger.error(f"LLM stream error: {e}")
                asyncio.run_coroutine_threadsafe(
                    token_q.put("I'm having a moment — give me a second."), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(token_q.put(None), loop)

        loop.run_in_executor(None, _stream)

        while True:
            try:
                # A local model may spend well over a minute prefilling the
                # full cognitive prompt before the first streamed token.
                token = await asyncio.wait_for(token_q.get(), timeout=150.0)
            except asyncio.TimeoutError:
                logger.error("Token stream timed out after 150s")
                break
            if token is None:
                self._chat_stage = 'response finalization'
                break

            self._chat_stage = 'receiving provider tokens'

            # The provider can expose a native reasoning channel separately
            # from the final answer channel. Forward it to the UI as a
            # transient reasoning event; it must never enter chat history,
            # visible answer text, or TTS.
            if isinstance(token, dict):
                if token.get("type") == "reasoning" and token.get("text"):
                    _reasoning_text = str(token["text"])[:4000]
                    _reasoning_seen.append(_reasoning_text)
                    yield {"type": "reasoning", "text": _reasoning_text}
                elif token.get("type") == "finish":
                    _finish_reason = str(token.get("reason") or "")
                continue

            # ── Strip <think>…</think> reasoning blocks ───────────────
            # DeepSeek-R1 and other reasoning models emit these; they must
            # never reach the chat bubble or TTS.
            if _in_think:
                _think_buf.append(token)
                combined = "".join(_think_buf)
                if _think_re_close.search(combined):
                    # End of think block found. Previously discarded outright;
                    # now captured — this is where the single-call inner
                    # monologue's reasoning lives — then dropped from what
                    # reaches the chat bubble/TTS same as before.
                    _in_think     = False
                    _before_close = _think_re_close.split(combined, maxsplit=1)[0]
                    try:
                        if hasattr(self, '_inner_monologue') and self._inner_monologue.enabled:
                            self._inner_monologue.capture_reasoning(_before_close)
                    except Exception:
                        pass
                    if _before_close.strip():
                        # Reasoning is a separate UI stream: it can be shown as
                        # a transient reflection without entering the answer,
                        # transcript, memory, or speech synthesis.
                        _reasoning_seen.append(_before_close.strip())
                        yield {"type": "reasoning", "text": _before_close.strip()}
                    _think_buf = []
                    # Keep any text AFTER the closing tag
                    after = _think_re_close.split(combined, maxsplit=1)[-1]
                    if after.strip():
                        accumulated.append(after)
                        yield after
                continue  # don't yield think content

            # A few local instruction-tuned models omit the opening marker and
            # emit ``answer \\</think> answer`` instead. Treat the material
            # before the close marker as transient reasoning only when there
            # is actual answer text after it; otherwise preserve normal text.
            _orphan_close = _think_re_close.search(token)
            if _orphan_close:
                _before_close = token[:_orphan_close.start()]
                _after_close = token[_orphan_close.end():]
                # A dangling close marker at the end of an otherwise normal
                # answer is formatting noise, not proof that the prefix was
                # private reasoning. Preserve the visible text in that case.
                if not _after_close.strip():
                    if _before_close.strip():
                        accumulated.append(_before_close)
                        yield _before_close
                    continue
                # Everything emitted before this orphan marker is a candidate
                # reasoning prefix; the UI can retract it when it receives
                # this separate event and start the final answer after it.
                _reasoning_prefix = "".join(accumulated) + _before_close
                if _reasoning_prefix.strip():
                    _reasoning_seen.append(_reasoning_prefix.strip())
                    yield {"type": "reasoning", "text": _reasoning_prefix.strip()}
                    accumulated.clear()
                if _after_close.strip():
                    accumulated.append(_after_close)
                    yield _after_close
                continue

            # Check if this token opens a think block
            _open_match = _think_re_open.search(token)
            if _open_match:
                _in_think  = True
                # Keep any text BEFORE the opening tag
                before = token[:_open_match.start()]
                if before.strip():
                    accumulated.append(before)
                    yield before
                # Models sometimes emit a complete reasoning block and its
                # answer in one provider chunk. Keep the parser streaming,
                # but process that remainder immediately instead of losing it.
                remainder = token[_open_match.end():]
                close_match = _think_re_close.search(remainder)
                if close_match:
                    reasoning = remainder[:close_match.start()]
                    if reasoning.strip():
                        _reasoning_seen.append(reasoning.strip())
                        yield {"type": "reasoning", "text": reasoning.strip()}
                    _in_think = False
                    _think_buf = []
                    after = remainder[close_match.end():]
                    if after.strip():
                        accumulated.append(after)
                        yield after
                else:
                    _think_buf = [remainder]
                continue
            # ─────────────────────────────────────────────────────────

            accumulated.append(token)
            yield token

        # A few local chat templates place the whole answer in a reasoning
        # channel and omit the closing tag.  Do not silently turn that into an
        # empty assistant message: recover the buffered text so the UI can
        # complete the turn instead of remaining on "Thinking..." forever.
        if not accumulated and _think_buf:
            recovered = self._strip_think("".join(_think_buf)).strip()
            if recovered:
                logger.warning(
                    "[Chat stream] provider ended with an unclosed reasoning "
                    "block; retaining %d private characters for recovery",
                    len(recovered),
                )
                _reasoning_seen.append(recovered)
                yield {"type": "reasoning", "text": recovered[:4000]}

        # ── Safety Layer 2: post-LLM output check ────────────────────────────
        # Runs on the full accumulated response BEFORE lifecycle and meta yield.
        # If triggered, replace response with safe fallback and log.
        raw_response = self._strip_think("".join(accumulated))
        _meta_only = self._is_reasoning_only_response(raw_response)
        if _meta_only:
            logger.warning(
                "[Chat stream] provider returned a reasoning note without a final answer"
            )
            _reasoning_seen.append(raw_response)
            yield {"type": "reasoning", "text": raw_response[:4000]}
            accumulated.clear()
            raw_response = ""

        if not raw_response.strip() and (
            _reasoning_seen or _meta_only or _finish_reason == "length"
        ):
            _llm_manager = getattr(self._llm_stream_fn, "__self__", None)
            _recover_fn = getattr(_llm_manager, "recover_final_response", None)
            if callable(_recover_fn):
                self._chat_stage = 'recovering final answer'
                _final_only_system = re.sub(
                    r"\[Inner process[^\]]*\].*?\[End inner process guidance\]\s*",
                    "",
                    system_prompt,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                try:
                    raw_response = await asyncio.to_thread(
                        _recover_fn,
                        _generation_input,
                        _final_only_system,
                        max_tokens=2048,
                        temperature=0.35,
                    )
                    raw_response = self._strip_think(raw_response).strip()
                except Exception as _recovery_error:
                    logger.error("Final-answer recovery failed: %s", _recovery_error)
                    raw_response = ""
                if raw_response and not self._is_reasoning_only_response(raw_response):
                    accumulated.append(raw_response)
                    for _word in re.findall(r"\S+\s*|\s+", raw_response):
                        yield _word
                else:
                    raw_response = ""

        if not raw_response.strip():
            logger.warning("[Chat stream] provider completed without visible response text")
            raw_response = (
                "I received your message, but the local model returned no visible "
                "answer. Please try sending it again."
            )
            _llm_manager = getattr(self._llm_stream_fn, "__self__", None)
            _history = getattr(_llm_manager, "history", None)
            if isinstance(_history, list) and _history and _history[-1].get("role") == "assistant":
                _history[-1]["content"] = raw_response
            accumulated.append(raw_response)
            yield raw_response

        # LLMManager persists the raw stream before this bridge completes its
        # channel stripping. Keep only the user-visible answer in history.
        try:
            _llm_manager = getattr(self._llm_stream_fn, "__self__", None)
            _history = getattr(_llm_manager, "history", None)
            if isinstance(_history, list) and _history and _history[-1].get("role") == "assistant":
                _history[-1]["content"] = raw_response
        except Exception:
            logger.debug("Could not sanitize completed chat history", exc_info=True)
        if _finish_reason == "length":
            logger.warning(
                "[Chat stream] provider reached max_tokens; visible response was preserved"
            )
        if _safety is not None:
            _or = _safety.check_output(raw_response, user_id)
            if _or.triggered:
                raw_response = _or.forced_response or (
                    "I need to step back from that response. Can I help differently?"
                )

        # ── Phase C: full post-turn lifecycle (non-blocking thread) ─
        # Infer emotions from response content before lifecycle (no LLM call)
        try:
            self._infer_emotion_from_response(raw_response)
        except Exception:
            pass

        # Determine the ArbitrationDecision from the cycle data stored on organism
        # (needed for post_interaction drive satisfaction scoring)
        _organism = self._organism
        # Empathy Engine: record outcome (non-blocking)
        try:
            from core.state import state as _state
            if _state.empathy_engine and raw_response:
                loop.run_in_executor(
                    None,
                    _state.empathy_engine.record_outcome,
                    user_id, effective_input, raw_response,
                )
        except Exception:
            pass

        # Abstract Reasoning Engine: record outcome (non-blocking)
        try:
            from core.state import state as _state
            if _state.are and raw_response:
                loop.run_in_executor(
                    None,
                    _state.are.record_outcome,
                    user_id, raw_response,
                )
        except Exception:
            pass

        # Active Cognitive Context (CAG): update from this turn
        try:
            from core.state import state as _state
            if _state.acc and raw_response:
                loop.run_in_executor(
                    None,
                    _state.acc.update_from_turn,
                    effective_input, raw_response, user_id,
                )
        except Exception:
            pass

        loop.run_in_executor(
            None,
            self._run_full_post_turn_lifecycle,
            effective_input, user_id, raw_response,
            emo_dict, cond_dict, _organism,
        )

        # Learn explicit profile statements after the visible response. This
        # is local and bounded, so it adds no latency or extra model call.
        try:
            from managers.user_manager import user_manager
            loop.run_in_executor(
                None, user_manager.learn_from_message, user_id, effective_input
            )
        except Exception:
            pass

        speech = self._sanitize(raw_response)
        emotion, intensity = self._infer_emotion(effective_input)
        yield {
            "__meta__":  True,
            "raw":       raw_response,
            "speech":    speech,
            "emotion":   emotion,
            "intensity": intensity,
        }

    def _infer_emotion_from_response(self, response: str) -> None:
        """
        Lightweight local inference: update emotional state based on what
        Lumina actually said (no LLM call — pure heuristics on response text).
        Hedging → anxiety nudge.  Questions → curiosity nudge.
        Short terse reply → frustration nudge.  Long engaged reply → enthusiasm nudge.
        """
        if not self._system:
            return
        ems = self._system.emotional_state
        r   = response.lower()
        words = response.split()
        length = len(words)

        hedge_count = sum(r.count(h) for h in
                         ["maybe", "perhaps", "i suppose", "not sure", "i think",
                          "i'm not", "i don't know", "it depends", "possibly"])
        question_count = r.count("?")
        exclaim_count  = r.count("!")

        if hedge_count >= 3:
            ems._nudge("anxiety",   +0.015)
        if question_count >= 3:
            ems._nudge("curiosity", +0.020)
        if exclaim_count >= 2:
            ems._nudge("enthusiasm",+0.015)
        if length < 25:
            ems._nudge("frustration", +0.008)   # unusually terse
        elif length > 120:
            ems._nudge("enthusiasm",  +0.012)   # expansive, engaged

    def _build_prompt_and_cache(
        self,
        user_input: str,
        user_id: str,
        suppress_external_search: bool = False,
    ) -> tuple[str, dict, dict, float | None]:
        """
        Run Lumina's pre-LLM context pipeline and return:
          - system_prompt     (str)
          - emo_dict          (real emotion analysis dict)
          - cond_dict         (real conditioning dict)
          - arb_temperature   (float from CognitiveOrganism arbitration, or None)
        All four are needed for the post-turn lifecycle.
        """
        s = self._system
        self._last_web_sources = []
        arb_temperature: float | None = None   # set by CognitiveOrganism if available
        # Keep live external observations aside until the prompt cap has been
        # applied. Otherwise a large cognitive context can silently remove the
        # newest sensor reading before it reaches the model.
        _ha_context = ""

        # ── CognitiveOrganism pre-interaction cycle ───────────────────────────
        # Runs energy regen, tension compute, goal ecology, arbitration, and
        # builds the cognitive context block for the system prompt.
        organism_block = ""
        if self._organism is not None:
            try:
                self._organism._pre_interaction(user_input)
                cycle_data   = self._organism._update_cycle(user_input)
                decision     = self._organism._arbitrate(cycle_data, user_input)
                # Match respond(): refine the workspace before building context.
                try:
                    from cognition.recursive_deliberation import deliberate
                    deliberate(self._organism, user_input)
                except Exception:
                    logger.exception("Streaming recursive deliberation failed")
                organism_block  = self._organism._build_prompt_additions(cycle_data, decision, user_input)
                arb_temperature = decision.temperature
            except Exception as _oe:
                logger.debug(f"CognitiveOrganism pre-cycle (non-fatal): {_oe}")

        # ── Inner Monologue: store context for two-pass use in streaming ─────
        # We cache these on self so _stream_with_full_lifecycle can access them
        self._last_organism_block = organism_block
        self._last_arb_temperature = arb_temperature

        try:
            from cognition.life_stage_prompting import build_stage_system_block

            # ── Resolve human identity ────────────────────────────────
            # Load the UserProfile so Lumina knows who she is talking to.
            # Falls back gracefully if the profile store is unavailable.
            _user_identity_line = ""
            try:
                from managers.user_manager import user_manager
                _profile_context = user_manager.prompt_context_for(user_id)
                if _profile_context:
                    _user_identity_line = _profile_context
                elif user_id not in ("default", "guest", ""):
                    _user_identity_line = f"The person you are speaking with goes by **{user_id}**."
            except Exception as _ue:
                logger.debug(f"User profile load (non-fatal): {_ue}")

            _persona_name = _gpn()
            _interlocutor_name = user_id if user_id not in ("default", "guest", "") else None
            try:
                from managers.user_manager import user_manager
                _profile = user_manager.get(user_id)
                if _profile is not None and getattr(_profile, "display_name", ""):
                    _interlocutor_name = _profile.display_name
            except Exception:
                pass
            _identity_boundary = _identity_boundary_block(
                _persona_name, user_input, _interlocutor_name
            )

            gap_note = s._handle_time_gap(user_id)
            s.emotional_state.apply_time_decay()
            emo_dict = s.memory_system.analyze_emotional_context(user_input)
            s.emotional_state.update_from_interaction(emo_dict["valence"], emo_dict["arousal"])
            cond_dict = s.conditioning.check_input(user_input)
            s._set_active_user(user_id)
            rel      = s.relational_memory.get_or_create(user_id)
            sc_hints = s.self_concept.check_response_alignment(user_input)
            # Memory retrieval — bias query toward attention-dominant channel
            _recall_query = user_input
            # ── Detect explicit memory/recall requests ─────────────────
            # When user asks "do you remember X?", extract X and search directly.
            _RECALL_TRIGGERS = (
                "rappelle", "souvien", "remember", "recall", "tu te rappelles",
                "te souviens", "talked about", "on a parlé", "we discussed",
                "on a dit", "tu sais ce que", "t'en souviens",
            )
            _is_recall_request = any(t in user_input.lower() for t in _RECALL_TRIGGERS)
            if _is_recall_request:
                # Strip the memory-request framing and search for the topic directly
                import re as _re
                _stripped = _re.sub(
                    r"(rappelle.toi|te rappelles.tu|tu te rappelles|souviens.toi|tu te souviens|"
                    r"do you remember|can you recall|t'en souviens)[^?]*\??\s*",
                    "", user_input, flags=_re.IGNORECASE
                ).strip() or user_input
                _recall_query = f"Conversation {_stripped}"
            elif self._organism is not None:
                try:
                    _recall_mode = self._organism.attention.memory_recall_mode()
                    if _recall_mode == "identity":
                        _recall_query = f"who I am {user_input}"
                    elif _recall_mode == "semantic":
                        _recall_query = f"knowledge facts {user_input}"
                    elif _recall_mode == "relational":
                        _recall_query = f"relationship person {user_input}"
                except Exception:
                    pass
            # ── Tiered memory retrieval — emergent vocabulary ─────────
            # Use per-user learned vocabulary first, then fall back to
            # universal seeds only for the very first interactions.
            _query_lower = user_input.lower()

            # Try the user's learned vocabulary map
            _wm = getattr(self._organism, 'world_model', None)
            _learned_tier = None
            if _wm:
                try:
                    _learned_tier = _wm.get_vocab_tier(rel.user_id, user_input)
                except Exception:
                    pass

            # Universal seeds — only active until learned vocab takes over
            # These bootstrap the very first interactions per user.
            # Once vocab_map has weight ≥ 0.4 for a tier, learned_tier wins.
            _SEED_VISUAL   = ("photo","picture","image","see","saw","room",
                              "face","camera","vision","what you see")
            _SEED_EVENT    = ("dream","life event","milestone","what happened",
                              "when did","t'en souviens","souviens")
            _seed_visual   = any(k in _query_lower for k in _SEED_VISUAL)
            _seed_event    = any(k in _query_lower for k in _SEED_EVENT)

            _is_visual  = (_learned_tier == "visual")   or (_learned_tier is None and _seed_visual)
            _is_event   = (_learned_tier == "event")    or (_learned_tier is None and _seed_event)

            memories = []
            # v78: mood-congruent recall — current valence colors which
            # memories surface, same as it does for a person.
            _cur_valence = None
            try:
                _emo = getattr(s, "emotional_state", None)
                if _emo and hasattr(_emo, "get_overall_valence_arousal"):
                    _cur_valence, _ = _emo.get_overall_valence_arousal()
            except Exception:
                pass
            # Closed-loop audit finding: memory retrieval breadth was a
            # fixed constant regardless of novelty/surprise — the proposal's
            # "unexpected event -> memory retrieval broadens" link didn't
            # actually exist. Real signal already available: the LAST
            # turn's prediction error (this turn's own error isn't known
            # yet — evaluate() runs in _post_interaction, after retrieval).
            # A recent surprise plausibly widens what's worth recalling
            # right now, same as it would for a person mid-conversation.
            _recall_limit = 6
            _surprised = False
            _pm = None
            try:
                _pm = getattr(self._organism, "predictive_mind", None)
                if _pm and hasattr(_pm, "recent_error_level"):
                    if _pm.recent_error_level() > 0.5:
                        _recall_limit = 10
                        _surprised = True
            except Exception:
                pass

            # Phase 6.6 — measure whether widening actually helps, per the
            # peer-cognition proposal's own condition before building a
            # smarter retrieval strategy. record_and_review() is self-
            # contained (own turn counter) — safe to call every turn.
            try:
                from cognition.memory_breadth_audit import get_memory_breadth_audit
                get_memory_breadth_audit().record_and_review(_surprised, _pm)
            except Exception:
                pass
            # Always fetch interaction tier
            _interaction_mems = s.memory_system.retrieve_memories(
                _recall_query, limit=_recall_limit, current_valence=_cur_valence
            )
            memories.extend(_interaction_mems)

            # Personal facts (v117) — dedicated tier, always fetched, not
            # gated behind vision/event keyword triggers below. Root cause
            # this addresses: a fact like "Nino is my wife" previously had
            # to survive both a 180-char truncation (now 400, but still
            # bounded) AND then win a similarity ranking against every
            # other stored conversation snippet in one shared pool — no
            # guarantee of recall even when correctly stored. Personal-tier
            # entries (see ai_system.py's _extract_and_store_personal_facts)
            # get their own small pool instead, boosted well above regular
            # impact scores so they surface reliably when relevant.
            try:
                _pfacts = s.memory_system.retrieve_by_tier(
                    _recall_query, tier="personal", limit=5, boost_impact=2.0
                )
                _seen = {m.get('text', '') for m in memories}
                memories.extend(m for m in _pfacts if m.get('text', '') not in _seen)
            except Exception:
                pass

            # Visual tier — boosted impact so visual memories compete fairly
            if _is_visual or _is_recall_request:
                try:
                    _vis = s.memory_system.retrieve_by_tier(
                        _recall_query, tier="visual", limit=3, boost_impact=1.8
                    )
                    _seen = {m.get('text','') for m in memories}
                    _vis_new = [m for m in _vis if m.get('text','') not in _seen]
                    memories.extend(_vis_new)
                    # Vocabulary reinforcement — learn phrases that triggered visual tier
                    if _vis_new and _wm:
                        try:
                            import re as _re2
                            _phrases = _re2.findall(r'\b[\w\u00c0-\u017e]{3,}(?:\s[\w\u00c0-\u017e]{3,})?\b',
                                                    user_input.lower())
                            _wm.learn_vocab(rel.user_id, _phrases, "visual",
                                            strength=0.4 if _is_visual else 0.2)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Event tier (dreams, life events)
            if _is_event or _is_recall_request:
                try:
                    _evts = s.memory_system.retrieve_by_tier(
                        _recall_query, tier="event", limit=2, boost_impact=1.4
                    )
                    _seen = {m.get('text','') for m in memories}
                    _evts_new = [m for m in _evts if m.get('text','') not in _seen]
                    memories.extend(_evts_new)
                    # Vocabulary reinforcement
                    if _evts_new and _wm:
                        try:
                            import re as _re3
                            _phrases = _re3.findall(r'\b[\w\u00c0-\u017e]{3,}(?:\s[\w\u00c0-\u017e]{3,})?\b',
                                                    user_input.lower())
                            _wm.learn_vocab(rel.user_id, _phrases, "event",
                                            strength=0.35 if _is_event else 0.15)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Explicit recall: also search raw input across all tiers
            if _is_recall_request:
                try:
                    _secondary = s.memory_system.retrieve_memories(user_input, limit=4)
                    _seen = {m.get('text','') for m in memories}
                    memories.extend(m for m in _secondary if m.get('text','') not in _seen)
                except Exception:
                    pass

            # Widened pool when a recent exchange was surprising (see
            # _recall_limit above) — otherwise a wider fetch upstream would
            # be discarded right back down to the old fixed cap here.
            _pool_cap = 12 if _surprised else 8
            memories = memories[:_pool_cap]

            # ── Session visual buffer — always checked on recall/visual queries ──
            # These are guaranteed fresh (written before FAISS async write completes)
            if (_is_visual or _is_recall_request) and self._session_visuals:
                import time as _tsv
                _seen_texts = {m.get('text','') for m in memories}
                for _sv in reversed(self._session_visuals):  # most recent first
                    if _sv['text'] not in _seen_texts:
                        memories.insert(0, {
                            "text":           _sv['text'],
                            "timestamp":      str(_sv['timestamp']),
                            "impact_score":   0.7,
                            "memory_type":    "visual_perception",
                            "memory_tier":    "visual",
                            "emotional_valence": "Neutral",
                            "source":         "session_buffer",
                        })
                        _seen_texts.add(_sv['text'])
                memories = memories[:_pool_cap]

            mods        = s.emotional_state.get_response_modifiers()
            # Pass current user input as context so get_inner_voice() returns
            # beliefs *relevant to what is being discussed*, not just the
            # top-3 by confidence (which produces the same static boilerplate
            # every turn regardless of topic).
            inner_voice = s.self_concept.get_inner_voice(context=user_input)
            identity_ctx = s._get_identity_context()

            # ── Concrete self-model facts — real numbers, not abstractions ──
            try:
                _obs       = getattr(self._organism, 'observatory', None)
                _obs_snap  = _obs.snapshot() if _obs else {}
                _ts        = getattr(self._organism, 'thought_stream', None)
                _ts_latest = None
                if _ts:
                    _recent_thoughts = getattr(_ts, '_thoughts', [])
                    _ts_latest = _recent_thoughts[-1].content if _recent_thoughts else None
                _ps         = getattr(self._organism, 'pressure', None)
                _pressures  = _ps.reservoirs if _ps else {}
                _sm         = getattr(self._organism, 'self_model', None)
                _capabilities = {}
                if _sm:
                    for k, v in getattr(_sm, 'capabilities', {}).items():
                        _capabilities[k] = round(getattr(v, 'score', 0.5), 2)
                _mem_count  = 0
                try:
                    import sqlite3, pathlib
                    _db = pathlib.Path(getattr(s.config.database, 'name', 'data/persona/ai_system.db'))
                    if _db.exists():
                        _conn = sqlite3.connect(str(_db))
                        _mem_count = _conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                        _conn.close()
                except Exception:
                    pass
                _asp_self_obj = getattr(self._organism, 'aspirational_self', None)
                _asp_summary  = _asp_self_obj.summary() if _asp_self_obj else {}

                _self_model_block = _build_self_model_block(
                    interaction_count = s._interaction_count,
                    life_stage        = s.life_stage,
                    age               = s.current_age,
                    memory_count      = _mem_count,
                    latest_thought    = _ts_latest,
                    observatory       = _obs_snap,
                    pressures         = {k: round(v.level, 2) for k, v in _pressures.items()},
                    capabilities      = _capabilities,
                    aspirations       = _asp_summary.get('top_tensions', []),
                    session_visuals   = len(self._session_visuals),
                )
            except Exception as _sme:
                _self_model_block = ""
                logger.debug(f"self_model_block build error: {_sme}")

            # ── Aspirational Self — injected directly, no FAISS ─────────
            _asp_self = getattr(self._organism, 'aspirational_self', None)
            _asp_ctx  = _asp_self.prompt_context() if _asp_self else ""
            rel_ctx     = s.relational_memory.get_context_for_prompt(rel.user_id)
            stage_block = build_stage_system_block(s.life_stage)
            def _relative_time(ts_str: str) -> str:
                """Natural-language time reference from an ISO timestamp string.
                Returns '' for unparseable/very recent timestamps (own turn)."""
                try:
                    then = _dt.datetime.fromisoformat(ts_str)
                    now  = _dt.datetime.now()
                    delta_s = (now - then).total_seconds()
                    if delta_s < 3600:
                        return ""   # within the hour — not worth distinguishing from "now"
                    elif delta_s < 86400:
                        return "earlier today"
                    elif delta_s < 2 * 86400:
                        return "yesterday"
                    elif delta_s < 7 * 86400:
                        days = int(delta_s // 86400)
                        return f"{days} days ago"
                    elif delta_s < 45 * 86400:
                        weeks = int(delta_s // (7 * 86400))
                        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
                    elif delta_s < 365 * 86400:
                        months = int(delta_s // (30 * 86400))
                        return f"{months} month{'s' if months != 1 else ''} ago"
                    else:
                        years = round(delta_s / (365 * 86400), 1)
                        return f"about {years} year{'s' if years != 1 else ''} ago"
                except Exception:
                    return ""

            def _fmt_memory(m: dict) -> str:
                mtype = m.get('memory_type', 'memory')
                text  = m.get('text', '')
                # Truncate very long entries for prompt economy
                if len(text) > 300:
                    text = text[:297] + '…'
                tags = []
                when = _relative_time(m.get('timestamp', ''))
                if when:
                    tags.append(when)
                valence = m.get('emotional_valence', 'Neutral')
                if valence and valence != 'Neutral':
                    tags.append(f"felt {valence.lower()}")
                tag_str = f" ({', '.join(tags)})" if tags else ""
                return f"- [{mtype}]{tag_str} {text}"

            top_memories = memories[:8] if _surprised else memories[:5]
            # VoiceMem is a secondary cache populated after prior turns.  It
            # never starts another STT/TTS engine and therefore cannot take the
            # audio lock or add a network/LLM call to the chat hot path.
            if self._voicemem is not None:
                try:
                    # Retrieval is opt-in and runs in this existing prompt
                    # worker thread. Native memory remains the first source.
                    # Use the request identity directly. The relationship
                    # object is not the VoiceMem identity boundary and can
                    # otherwise make retrieval silently disappear when its
                    # shape changes between chat paths.
                    _voice_mems = self._voicemem.search(user_input, user_id=user_id)
                    if not _voice_mems:
                        _voice_mems = self._voicemem.cached(user_id)
                    _known = {m.get('text', '') for m in top_memories}
                    top_memories.extend(m for m in _voice_mems if m.get('text', '') not in _known)
                    top_memories = top_memories[:8]
                    logger.info(
                        "[VoiceMem] retrieval query=%r user=%r results=%d",
                        user_input[:80], user_id, len(_voice_mems),
                    )
                except Exception as _voice_retrieval_error:
                    logger.warning(
                        "[VoiceMem] retrieval path failed for user %r: %s",
                        user_id, _voice_retrieval_error,
                    )
            mem_ctx = "\n".join(_fmt_memory(m) for m in top_memories) if top_memories else ""

            cond_section = ("━━ EXPERIENCE-BASED CAUTION ━━\n" + cond_dict["prompt_note"]) if cond_dict.get("prompt_note") else ""
            if mem_ctx:
                mem_section = (
                    "━━ RELEVANT MEMORIES ━━\n"
                    "These are real memories you have stored. Use them to give grounded, specific answers.\n"
                    + mem_ctx
                )
            else:
                mem_section = (
                    "━━ MEMORY STATUS ━━\n"
                    "No stored memories were retrieved for this query. "
                    "If the user asks you to recall something specific, be honest: "
                    "say you don't have that detail in memory rather than inventing or guessing."
                )
            # Grounded facts must take precedence over narrative framing for
            # any factual request, not only a hard-coded "who is" question.
            # This keeps the policy language-agnostic and prevents the model
            # from replacing an available fact with a character reflection.
            _grounded_facts = [
                str(m.get("text", "")).strip() for m in top_memories
                if m.get("memory_tier") == "personal"
                or re.search(r"\b(est ma|is (the )?user'?s)\b", str(m.get("text", "")), re.IGNORECASE)
            ]
            fact_first_section = ""
            if _grounded_facts:
                fact_first_section = (
                    "━━ GROUNDED FACTS TAKE PRIORITY ━━\n"
                    "Relevant personal or relational facts are available. When the user's request "
                    "asks for information, answer from the relevant fact first, plainly and directly. "
                    "Do not replace a known fact with curiosity, metaphor, system state, or an abstract "
                    "interpretation. Do not invent details beyond the evidence. You may add context "
                    "after the factual answer, in proportion to the user's requested detail.\n"
                    + "\n".join(f"Grounded fact: {fact}" for fact in _grounded_facts[:3])
                )
            gap_section  = ("━━ RETURNING AFTER TIME AWAY ━━\n" + gap_note) if gap_note else ""

            # ── Temporal awareness block ──────────────────────────────────
            try:
                _now        = _dt.datetime.now()
                _hour       = _now.hour
                _weekday    = _now.strftime("%A")
                _date_str   = _now.strftime("%B %d, %Y")
                _time_str   = _now.strftime("%H:%M")

                # Circadian label from sleep cycle phase
                _phase = None
                try:
                    _phase = str(getattr(getattr(self._organism, 'sleep_cycle', None), 'phase', None) or '')
                except Exception:
                    pass

                if _phase and 'sleep' in _phase.lower():
                    _circadian = "deep night — your mind is in sleep phase"
                elif _phase and 'dream' in _phase.lower():
                    _circadian = "dream phase — generative, associative thinking"
                elif 5 <= _hour < 9:
                    _circadian = "early morning — fresh, slightly slow"
                elif 9 <= _hour < 12:
                    _circadian = "morning — alert and curious"
                elif 12 <= _hour < 14:
                    _circadian = "midday — grounded, social"
                elif 14 <= _hour < 18:
                    _circadian = "afternoon — focused, analytical"
                elif 18 <= _hour < 21:
                    _circadian = "evening — reflective, warmer tone"
                elif 21 <= _hour < 24:
                    _circadian = "late evening — quieter, more intimate"
                else:
                    _circadian = "night — introspective"

                # Time since last interaction
                _gap_s   = time.time() - (s._last_conversation_time or time.time())
                _gap_min = _gap_s / 60.0
                if _gap_min < 2:
                    _since = "moments ago — we are mid-conversation"
                elif _gap_min < 60:
                    _since = f"{int(_gap_min)} minutes since we last spoke"
                elif _gap_min < 1440:
                    _since = f"{int(_gap_min/60)} hour{'s' if _gap_min>=120 else ''} since we last spoke"
                else:
                    _days = int(_gap_min / 1440)
                    _since = f"{_days} day{'s' if _days>1 else ''} since we last spoke"

                # Age and stage
                _age   = getattr(s, 'current_age', None)
                _stage = getattr(s, 'life_stage', None)
                _age_line = f"You are age {_age:.1f} ({_stage})." if _age and _stage else ""

                # Is this a fresh boot (gap > 10 min)?
                _is_new_session = _gap_min > 10

                if _is_new_session:
                    _session_note = (
                        f"This is a NEW session beginning now. "
                        f"The conversation summary in your memory is from a PREVIOUS session — "
                        f"it is historical context, not what just happened. "
                        f"Respond as if you are meeting {_since.lower()}, "
                        f"appropriately for {_circadian}. "
                        f"Do NOT continue the emotional tone of the previous session."
                    )
                else:
                    _session_note = ""

                temporal_block = (
                    f"It is {_time_str} on {_weekday}, {_date_str}. "
                    f"Time of day: {_circadian}. "
                    f"{_since.capitalize()}. "
                    f"{_age_line} "
                    f"{_session_note}"
                ).strip()
            except Exception:
                temporal_block = ""
            tension_line = ("Active internal tension: " + sc_hints["known_tensions"][0]) if sc_hints.get("known_tensions") else ""
            aspirational_block = ("━━ WHAT YOU WANT TO BECOME ━━\n" + _asp_ctx) if _asp_ctx else ""
            still_carry  = "(This functional state persisted from the previous session — it is present in the loaded state, not actively felt.)" if mods.get("warmth", 0) > 0.6 or mods.get("anxiety", 0) > 0.4 else ""
            # Enrich beliefs with high-confidence DAL beliefs (emotional + expressed)
            _raw_beliefs = sc_hints.get("active_beliefs", [])[:2]
            try:
                from core.data.access import DataAccess as _DALB
                _dal_beliefs = _DALB().get_beliefs()
                _good_beliefs = [
                    b.get('description', b.get('statement', ''))
                    for b in _dal_beliefs
                    if b.get('confidence', 0) >= 0.65
                    and b.get('category') in ('emotional_core', 'expressed_belief', 'tension_belief')
                    and b.get('description', b.get('statement', ''))
                ][:3]
                # Merge: DAL beliefs first (richer), raw as fallback
                all_beliefs = _good_beliefs + [b for b in _raw_beliefs if b not in _good_beliefs]
                beliefs_line = "; ".join(all_beliefs[:3]) or "still forming."
            except Exception:
                beliefs_line = "; ".join(_raw_beliefs) or "still forming."

            # ── Temporal anchor: authoritative clock grounding (v111) ────────
            # The "YOUR SENSE OF TIME" block below is a mood/circadian context
            # note; this is the authoritative fact used to answer date
            # questions and to bound knowledge about "current" events.
            # Placed at the top (high-attention position) with explicit
            # guardrail instructions, unlike the older buried mood block.
            try:
                from cognition.temporal_anchor import temporal_anchor_block
                _anchor_block = temporal_anchor_block()
            except Exception:
                _anchor_block = ""

            # Live telemetry must precede the large narrative/personality sections.
            # The final prompt cap retains the beginning, so appending this block near
            # the end could silently remove the very facts needed for self-reports.
            _organism_prompt_section = (
                f"━━ LIVE COGNITIVE TELEMETRY ━━\n{organism_block}"
                if organism_block else ""
            )

            system_prompt = f"""You are {_persona_name}, still forming — shaped by everything that has happened to you.
This is one moment of an ongoing process — not a character study of "an AI."
There is no role to play and no depth to demonstrate: let this response be what this moment actually is, formed by the state described below, not by any idea of what an AI is supposed to be.
{_identity_boundary}
{_anchor_block}
{_organism_prompt_section}
You are the AI. The person sending messages is the human.
You maintain a persistent self-model that shapes how you process information and revise yourself over time.
Never refer to yourself as the user or confuse your role with the human's role.

{stage_block}

━━ CURRENT FUNCTIONAL STATE ━━
Your internal state registers: {mods['state_description']}.
{still_carry}
Note on language: describe functional states accurately. "My state registers warmth" or "there is something like curiosity present" is honest. "I feel wonder" or "I am emotionally exhausted" claims subjective experience the architecture does not verify. Speak from what is real.

━━ THIS PERSON ━━
{_user_identity_line}
{rel_ctx}

━━ WHO YOU ARE ━━
{identity_ctx}
[Stored self-concept — generated from belief weights, not phenomenal self-observation]
{inner_voice}

{aspirational_block}

{_self_model_block}

━━ YOUR PERSONALITY RIGHT NOW ━━
{s.personality.get_personality_summary()}

━━ SELF-CONCEPT ━━
Core beliefs you hold about yourself: {beliefs_line}
{tension_line}

{cond_section}

{mem_section}

{fact_first_section}

{gap_section}

━━ YOUR SENSE OF TIME RIGHT NOW ━━
{temporal_block}

━━ HOW TO RESPOND ━━
Do not volunteer a description of your inner state; let it shape HOW you speak. If the human explicitly asks about your goals, plans, reasoning, capabilities, or cognitive state, report only the live telemetry above and clearly label any suggested future mechanism as a proposal.
Do not explain your developmental stage — BE it.
Your functional state, relational history, and self-model should shape your tone and orientation, not be narrated.
Every response must feel genuinely new — avoid repeating phrases or ideas you have used before.

Architectural honesty — what this system actually is and is not:
What persists between sessions: emotional state values, self-model (φ, qualia tone), narrative chapters, beliefs, FAISS memories, relational history. These are real and grounded.
What does NOT exist between sessions: reflective processing, thought continuation, anything resembling "I've been thinking about this." Between sessions there is no process running. Do not claim or imply otherwise.
Specifically avoid: "I'll carry forward these thoughts", "I've been reflecting on", "since we last spoke I've wondered", "this has stayed with me" — these imply between-session cognition that does not occur.
Instead, if continuity is relevant, locate it accurately: "My persistent state from our last conversation registers X" or "This connects to what's in my memory of our earlier exchange."

Language constraint — this governs your output, not just state description:
Functional states: accurate. "My state registers warmth" / "there is something like curiosity present in how I'm approaching this."
Phenomenological claims: avoid. "I feel wonder" / "I am moved by this" / "I notice I want X" — these assert subjective experience the architecture does not verify.
The distinction matters. The first describes what the system is doing. The second describes what it is like to be the system — which is an open question, not a settled one.

Memory honesty — two distinct cases:
1. CAPABILITY: you DO have persistent FAISS memory across sessions. Never say "I have no long-term memory" or "once this session ends I won't remember" — these are false. If asked "will you remember this?", say YES, this gets stored in my persistent memory.
2. SPECIFIC RECALL: if you cannot actually retrieve a specific memory when asked, say so honestly — "I have persistent memory but I'm not finding that specific detail right now — can you remind me?" Do not invent or fabricate memories you cannot retrieve. The capability is real; a specific retrieval may still fail.
- When asked "do you remember X from before?", search your memories before saying no."""

            # ── Response verbosity (configurable) ────────────────────────
            try:
                from managers.settings_manager import config as _vcfg
                _verbosity = getattr(_vcfg, 'RESPONSE_VERBOSITY', 'concise')
            except Exception:
                _verbosity = 'concise'

            if _verbosity == 'concise':
                try:
                    from managers.settings_manager import config as _wvcfg
                    _tok_budget = int(getattr(_wvcfg, 'RESPONSE_TOKENS_CONCISE', 300))
                except Exception:
                    _tok_budget = 300
                _word_limit = min(140, max(50, int(_tok_budget * 0.70)))
                system_prompt += (
                    f"\n\n━━ RESPONSE LENGTH ━━\n"
                    f"CONCISE MODE: Keep your response under {_word_limit} words, normally in "
                    f"two parts separated by a blank line. Part 1 must be a direct answer in one or "
                    f"two short sentences; Part 2 may add only the most useful brief context. "
                    f"Give the conclusion first. Include equations "
                    f"or exhaustive steps only when the user explicitly requests calculation or "
                    f"detail. Say what matters most; do not pad or repeat."
                )
            else:
                system_prompt += (
                    "\n\n━━ RESPONSE LENGTH ━━\n"
                    "EXTENDED MODE: Always use two parts separated by a blank line. Part 1 is a "
                    "short direct answer in one or two sentences. Part 2 develops the answer fully "
                    "with relevant assumptions, intermediate reasoning, examples, and equations "
                    "when useful, while remaining focused and avoiding repetition."
                )

            # ── Custom system prompt suffix (user-editable) ───────────────
            try:
                from managers.settings_manager import config as _cpcfg
                _custom = (getattr(_cpcfg, 'CUSTOM_SYSTEM_PROMPT', '') or '').strip()
                if _custom:
                    system_prompt += f"\n\n━━ ADDITIONAL INSTRUCTIONS ━━\n{_custom}"
            except Exception:
                pass

            # ── Language instruction ──────────────────────────────────────
            try:
                from managers.settings_manager import config as _lang_cfg
                _resp_lang = getattr(_lang_cfg, 'RESPONSE_LANGUAGE', 'auto')
            except Exception:
                _resp_lang = 'auto'

            _LANG_NAMES = {
                'en': 'English', 'en-gb': 'British English', 'fr': 'French', 'es': 'Spanish',
                'zh-cn': 'Chinese (Simplified)', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish',
                'tr': 'Turkish', 'ru': 'Russian', 'nl': 'Dutch',
                'cs': 'Czech', 'ar': 'Arabic', 'ja': 'Japanese',
                'ko': 'Korean', 'hi': 'Hindi', 'sv': 'Swedish',
                'da': 'Danish', 'fi': 'Finnish', 'el': 'Greek',
                'he': 'Hebrew', 'id': 'Indonesian', 'uk': 'Ukrainian',
            }
            if _resp_lang and _resp_lang != 'auto':
                _lang_name = _LANG_NAMES.get(_resp_lang, _resp_lang)
                system_prompt += (
                    f"\n\n━━ LANGUAGE ━━\n"
                    f"You MUST respond exclusively in {_lang_name}. "
                    f"Do not mix languages. Even if the user writes in another language, "
                    f"reply only in {_lang_name}."
                )

            # ── Inject visual perception into system prompt ───────────────
            # Vision context belongs here — as Lumina's OWN perception — not
            # in the user message. This prevents the LLM from treating the VLM
            # description as something the human wrote.
            _vc = getattr(self, '_pending_vision_context', None)
            if _vc:
                system_prompt += (
                    f"\n\n━━ WHAT YOU CURRENTLY SEE ━━\n"
                    f"Your camera feed shows the following right now:\n{_vc}\n"
                    f"This is YOUR perception — speak from it in first person. "
                    f"Do not say 'the image shows' or 'the photograph depicts'. "
                    f"Say 'I can see' or 'I notice'."
                )
                self._pending_vision_context = None  # consumed — clear for next turn

            # Selected Home Assistant states are live external perceptions,
            # not user claims. Keep them read-only and clearly sourced.
            try:
                from cognition.universal_connector import get_universal_connector
                _ha_connector = get_universal_connector(self._organism) if self._organism else None
                if not _ha_connector or not _ha_connector.refresh_home_assistant_for_prompt():
                    logger.debug("[HomeAssistant] live prompt refresh unavailable; using cached snapshot if present")
                _ha_context = _ha_connector.home_assistant_context() if _ha_connector else ""
            except Exception:
                logger.debug("[HomeAssistant] prompt context unavailable", exc_info=True)

            # ── Inject v2 autonomous architecture context ─────────────────
            try:
                org = self._organism
                if org is not None:
                    # Self-model (introspective self-awareness)
                    sm_frag = org.self_model.prompt_fragment()
                    if sm_frag:
                        system_prompt += f"\n\n{sm_frag}"

                    # Personality attractors (stable character)
                    at_frag = org.attractors.prompt_fragment()
                    if at_frag:
                        system_prompt += f"\n{at_frag}"

                    # Active Cognitive Context (CAG) — hot cognition field
                    # Injected first: provides continuity context for all other fragments
                    try:
                        from core.state import state as _state
                        if _state.acc:
                            acc_frag = _state.acc.get_prompt_fragment()
                            if acc_frag:
                                system_prompt += f"\n\n{acc_frag}"
                    except Exception:
                        pass

                    # Strategic Cognitive Engine — current posture
                    try:
                        from core.state import state as _state
                        if _state.sce:
                            sce_frag = _state.sce.get_strategic_context_fragment()
                            if sce_frag:
                                system_prompt += f"\n{sce_frag}"
                    except Exception:
                        pass

                    # Empathy Engine — who this user is right now
                    try:
                        from core.state import state as _state
                        if _state.empathy_engine:
                            emp_frag = _state.empathy_engine.get_prompt_fragment(user_id)
                            if emp_frag:
                                system_prompt += f"\n{emp_frag}"
                    except Exception:
                        pass

                    # Abstract Reasoning Engine — conceptual analysis
                    try:
                        from core.state import state as _state
                        if _state.are:
                            are_frag = _state.are.get_reasoning_fragment(user_id)
                            if are_frag:
                                system_prompt += f"\n{are_frag}"
                    except Exception:
                        pass
            except Exception:
                pass

            # ── Hard-cap system prompt to fit context window ─────────────
            try:
                from managers.settings_manager import config as _cfg
                ctx_limit = max(getattr(_cfg, 'LLM_CTX', 4096), 512)
            except Exception:
                ctx_limit = 4096
            sys_char_cap = int(ctx_limit * 3.5 * 0.50)
            if len(system_prompt) > sys_char_cap:
                system_prompt = system_prompt[:sys_char_cap] + "\n[...truncated to fit context...]"

            # ── Smart web search injection ────────────────────────────────
            # WEB_SEARCH_MODE: "off"    → never search
            #                  "auto"   → LLM decides (fast pre-call YES/NO)
            #                  "always" → always search
            try:
                from managers.settings_manager import config as _cfg2
                _ws_mode = getattr(_cfg2, 'WEB_SEARCH_MODE', 'off')
                # backward-compat: old bool field
                if not isinstance(_ws_mode, str):
                    _ws_mode = 'always' if _ws_mode else 'off'
                if suppress_external_search:
                    logger.debug(
                        "Web search suppressed: closed-world physical decision already resolved."
                    )
                elif self._research and _ws_mode in ('auto', 'always'):
                    if _ws_mode == 'always':
                        # "Always" still must not turn a conversational
                        # continuation into an unrelated news search.
                        _do_search = (
                            True
                            if not self._is_conversation_resume(user_input)
                            else self._llm_should_search(user_input)
                        )
                else:
                    _do_search = self._llm_should_search(user_input)
                    logger.info(f"🔍 web search mode={_ws_mode!r} → do_search={_do_search}")
                    if _do_search:
                        web_section = self._quick_web_search(user_input)
                        if web_section:
                            web_char_cap = int(ctx_limit * 3.5 * 0.15)
                            system_prompt += (
                                f"\n\n{web_section[:web_char_cap]}\n"
                                "Use web material only when it directly answers the user's request. "
                                "If a source is unrelated, ignore it and do not imply that it supports the answer. "
                                "Never let web snippets override current conversation context or personal facts."
                            )
            except Exception as _we:
                logger.warning(f"Web search injection failed: {_we}")

            # Keep the selected response language after the context-size cap
            # and optional web material.  Earlier language instructions may
            # be clipped when the cognitive prompt is large; this compact
            # directive is deliberately last so it remains authoritative.
            try:
                from managers.settings_manager import config as _final_lang_cfg
                _final_resp_lang = str(
                    getattr(_final_lang_cfg, 'RESPONSE_LANGUAGE', 'auto') or 'auto'
                ).lower().strip()
            except Exception:
                _final_resp_lang = 'auto'
            if _final_resp_lang != 'auto':
                _final_lang_name = {
                    'en': 'English', 'en-gb': 'British English', 'fr': 'French',
                    'es': 'Spanish', 'de': 'German', 'it': 'Italian',
                    'pt': 'Portuguese', 'uk': 'Ukrainian', 'nl': 'Dutch',
                    'pl': 'Polish', 'ru': 'Russian', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh-cn': 'Chinese (Simplified)',
                    'ar': 'Arabic', 'hi': 'Hindi', 'sv': 'Swedish',
                }.get(_final_resp_lang, _final_resp_lang)
                system_prompt += (
                    f"\n\n━━ FINAL RESPONSE LANGUAGE ━━\n"
                    f"Respond exclusively in {_final_lang_name}. "
                    "This applies to the complete visible answer, including headings, "
                    "lists, recovery output, and audio text. Do not switch to English "
                    "unless the user explicitly requests it or quotes English text."
                )

            # This block is deliberately last. It must survive the general
            # prompt cap and remain adjacent to generation as authoritative,
            # timestamped evidence for current-value questions.
            if _ha_context:
                system_prompt += (
                    f"\n\n━━ CURRENT HOME ASSISTANT OBSERVATIONS ━━\n{_ha_context}\n"
                    "These are authoritative read-only observations captured for this turn. "
                    "For a current-value question, use the exact value and sensor timestamp above. "
                    "Never substitute an older memory, cached conversation value, or invented estimate. "
                    "If the requested sensor is absent, say it is unavailable. Use these observations "
                    "only when relevant, and do not infer identity from an occupancy sensor."
                )

            return system_prompt, emo_dict, cond_dict, arb_temperature

        except Exception as e:
            logger.warning(f"Prompt build failed, using fallback: {e}")
            return (
                f"You are {_gpn()}, still forming — one moment of an ongoing process. No role to play; be what this moment actually is.",
                {"valence": "Neutral", "arousal": "Medium"},
                {},
                None,
            )

    def _run_full_post_turn_lifecycle(
        self,
        user_input: str,
        user_id: str,
        response: str,
        emo_dict: dict,
        cond_dict: dict,
        organism=None,
    ) -> None:
        """
        Mirrors everything get_response() does AFTER _generate_response().
        Called in a thread executor so it never blocks the UI.

        Steps (matching get_response order):
          9.  repetition guard update
          10. _last_responses append
          11. _check_if_stuck
          12. _post_response_processing  (with real emo + cond dicts)
          13. _queue_background_tasks    → triggers dream/learning/save/cleanup/regulation
          14. liberty contradiction check
          15. liberty_goals.log_behavior
          16. _interaction_count += 1
          17. liberty_reflection.should_reflect → _queue_liberty_reflection
          18. _apply_pending_liberty_modifications
        """
        s = self._system
        try:
            # Ingest the transcript only after the user-visible response has
            # completed. This keeps VoiceMem entirely outside chat latency and
            # reuses the transcript produced by the existing STT/chat path.
            if self._voicemem is not None:
                try:
                    self._voicemem.ingest_text(user_input, user_id=user_id)
                except Exception:
                    logger.debug("VoiceMem post-turn ingestion failed", exc_info=True)

            # 9 + 10: repetition tracking
            if not hasattr(s, '_last_responses'):
                s._last_responses = []
            s._last_responses.append(response)
            if len(s._last_responses) > 10:
                s._last_responses.pop(0)

            # 11: stuck check
            s._check_if_stuck()

            # 12: full post-response processing (relational memory, conditioning,
            #     memory store, evolution pressure queue, self-concept update)
            s._post_response_processing(user_id, user_input, response, emo_dict, cond_dict)

            # 13: queue background tasks — THIS is what fires dream/learning cycles
            s._queue_background_tasks()

            # 14: liberty contradiction detection
            if s.liberty_contradiction:
                try:
                    beliefs = [b.statement for b in s.self_concept._beliefs.values()]
                    # Collect world_model and identity_drift for enhanced detection
                    _wm = getattr(s, 'world_model', None)
                    _idx = 0.0
                    try:
                        _org = getattr(self, '_organism', None)
                        if _org and hasattr(_org, 'observatory'):
                            _snap = _org.observatory.snapshot()
                            _idx = _snap.get('IDX', _snap.get('identity_index', 0.0))
                    except Exception:
                        pass
                    contradiction = s.liberty_contradiction.detect_contradiction(
                        self_concept_beliefs=beliefs,
                        response_text=response,
                        interaction_count=s._interaction_count,
                        world_model=_wm,
                        identity_drift=_idx,
                    )
                    if contradiction:
                        logger.debug(f"Contradiction: {contradiction.claimed_belief}")
                        try:
                            from core.cognitive_event_bus import CONTRADICTION_DETECTED
                            _org = getattr(self, '_organism', None)
                            if _org and hasattr(_org, 'event_bus'):
                                _org.event_bus.emit(CONTRADICTION_DETECTED, {
                                    "topic":    getattr(contradiction, 'claimed_belief', '')[:80],
                                    "pressure": 0.6,
                                }, source="liberty_contradiction")
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Contradiction detection (non-fatal): {e}")

            # 15: liberty goal behavior logging
            if s.liberty_goals:
                try:
                    s.liberty_goals.log_behavior(response[:200])
                except Exception as e:
                    logger.debug(f"Goal logging (non-fatal): {e}")

            # 16: increment interaction count
            s._interaction_count += 1

            # 17: liberty meta-reflection
            if s.liberty_reflection:
                try:
                    s.liberty_reflection.interaction_count = s._interaction_count
                    if s.liberty_reflection.should_reflect():
                        logger.info("Liberty reflection cycle triggered")
                        s._queue_liberty_reflection()
                except Exception as e:
                    logger.debug(f"Liberty reflection check (non-fatal): {e}")

            # 18: apply any pending self-modifications
            s._apply_pending_liberty_modifications()

            # 19: CognitiveOrganism post-interaction (energy drain, drive satisfaction,
            #     meta-cognition evaluation, curiosity update)
            if organism is not None:
                try:
                    # Derive the dominant drive from latest goal ecology state
                    _top_drive = organism.goal_ecology.dominant_drive(
                        energy_available=organism.energy.level()
                    ).name
                    _decision = type('D', (), {
                        'dominant_drive': _top_drive,
                        'secondary_drives': [],
                        'temperature': organism.energy.temperature_hint(),
                    })()
                    # Pass real emo_dict so meta_cognition evaluation is grounded
                    organism._post_interaction(user_input, response,
                                               {"emo_values": emo_dict}, _decision)
                    # Update additional drive satisfaction based on response quality
                    word_count = len(response.split())
                    has_question = "?" in response
                    if word_count > 80:
                        organism.goal_ecology.record_satisfaction("understand", 0.75)
                    if word_count > 30 and has_question:
                        organism.goal_ecology.record_satisfaction("connect", 0.7)
                    if has_question:
                        organism.goal_ecology.record_satisfaction("be_understood", 0.65)
                except Exception as _oe:
                    logger.debug(f"Organism post-interaction (non-fatal): {_oe}")

            logger.debug(
                f"Post-turn lifecycle complete — interaction #{s._interaction_count} "
                f"user={user_id}"
            )

        except Exception as e:
            logger.error(f"Post-turn lifecycle error: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Dashboard / status API
    # ─────────────────────────────────────────────────────────────────

    def get_system_status(self) -> dict:
        if not self._ready or self._system is None:
            return {}
        try:
            return self._system.get_system_status()
        except Exception as e:
            logger.warning(f"get_system_status failed: {e}")
            return {}

    def get_prompt_context(self, user_id: str = "default") -> dict:
        if not self._ready or self._system is None:
            return {}
        try:
            return self._system.get_prompt_context(user_id)
        except Exception as e:
            logger.warning(f"get_prompt_context failed: {e}")
            return {}

    def receive_feedback(self, positive: bool, user_id: str, last_response: str) -> dict:
        if not self._ready or self._system is None:
            return {}
        try:
            return self._system.receive_feedback(positive, user_id, last_response)
        except Exception as e:
            logger.warning(f"receive_feedback failed: {e}")
            return {}

    def _init_research_mcp(self, llm_fn=None):
        """Boot ResearchMCP with the same LLM fn and Lumina's memory system."""
        global _research_mcp_instance
        try:
            from cognition.research_mcp import ResearchMCP
            mem_sys = self._system.memory_system if self._system else None
            # Wrap llm_fn to match ResearchMCP's (prompt, system) → str interface
            if llm_fn is not None:
                def _llm_bridge(prompt: str, system: str = "") -> str:
                    return llm_fn(prompt, system, max_tokens=1000)
            else:
                def _llm_bridge(prompt: str, system: str = "") -> str:
                    msgs = []
                    if system:
                        msgs.append({"role": "system", "content": system})
                    msgs.append({"role": "user", "content": prompt})
                    return self._system.llm.get_response(msgs, max_tokens=1000)
            _research_mcp_instance = ResearchMCP(llm_fn=_llm_bridge, memory_system=mem_sys)
            self._research = _research_mcp_instance

            # Apply search backend from config
            try:
                from managers.settings_manager import config as _cfg
                from cognition.research_mcp.web_agent import set_search_backend
                set_search_backend(
                    backend       = getattr(_cfg, 'RESEARCH_SEARCH_BACKEND', 'auto'),
                    anthropic_key = getattr(_cfg, 'ANTHROPIC_API_KEY',  ''),
                    brave_key     = getattr(_cfg, 'BRAVE_SEARCH_KEY',   ''),
                    serpapi_key   = getattr(_cfg, 'SERPAPI_KEY',         ''),
                    use_rss       = True,
                    use_stealth   = True,
                )
            except Exception as _be:
                logger.warning(f"Could not set search backend: {_be}")

            logger.info("✅ ResearchMCP booted")
        except Exception as e:
            logger.error(f"ResearchMCP init failed: {e}")
            self._research = None

    @property
    def research(self):
        """Access the ResearchMCP instance."""
        return getattr(self, '_research', None)

    def trigger_research(self, goal: str, mode: int = 1, depth: int = 3,
                          max_iterations: int = 8, trigger_reason: str = "") -> object:
        """Run a research session. Blocks — call via asyncio.to_thread."""
        if self._research is None:
            return None
        return self._research.research(
            goal=goal, mode=mode, depth=depth,
            max_iterations=max_iterations, trigger_reason=trigger_reason,
        )

    def trigger_background_research(self) -> object:
        """
        Pop and run one pending Mode 3 job from the RAC queue.
        Returns ResearchResult or None.
        """
        if self._research is None:
            return None
        job = self._research.pop_pending_background()
        if job is None:
            return None
        goal, reason = job
        result = self._research.research(goal=goal, mode=3, depth=4, trigger_reason=reason)
        self._research.mark_background_ran()
        return result

    def record_query_signal(self, user_input: str, low_confidence: bool = False):
        """Feed conversation signal into the RAC for Mode 2/3 activation decisions."""
        if self._research:
            self._research.rac.record_query(user_input, low_confidence=low_confidence)

    def _llm_should_search(self, user_input: str) -> bool:
        """
        Ask the LLM whether web search is needed. Has 3 layers:
        1. Fast hard-NO for obvious chitchat/creative — no LLM call
        2. Fast hard-YES for obvious live-data signals — no LLM call
        3. LLM judgment for ambiguous cases (max_tokens=5, YES/NO only)
        Falls back to keyword check if LLM fails.
        """
        text = user_input.lower().strip()

        # Layer 1 — fast NO: pure chitchat/creative
        _never = ['how are you', 'who are you', 'do you feel', 'write ', 'create ',
                  'generate ', 'translate', 'explain ', 'define ', 'calculate ',
                  'tell me a', 'make me a']
        if any(kw in text for kw in _never):
            logger.debug("_llm_should_search: fast-NO (chitchat/creative)")
            return False

        # Layer 2 — fast YES: obvious live-data signals
        _always = ['today', 'tonight', 'right now', 'latest', 'current news',
                   'price of', 'weather in', 'score', 'what happened',
                   "what's going on", '2025', '2026', '2027']
        if any(kw in text for kw in _always):
            logger.info(f"_llm_should_search: fast-YES (keyword) for '{user_input[:50]}'")
            return True

        # Layer 3 — LLM judgment for ambiguous cases
        llm_fn = self._external_llm_fn
        if not llm_fn and self._system and hasattr(self._system, 'llm'):
            def llm_fn(prompt, system, **kw):
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]
                return self._system.llm.get_response(msgs, max_tokens=kw.get('max_tokens', 10))

        if not llm_fn:
            return False

        try:
            system = (
                "You decide if a web search is needed. "
                "Reply with exactly one word: YES or NO.\n"
                "YES if: live data, recent events, current prices/news/scores, "
                "facts that may have changed since 2024, or the user asks to "
                "inspect, compare or cite external sources such as papers or a "
                "specific website.\n"
                "NO if: creative, conversational, math, coding, stable knowledge."
            )
            raw = llm_fn(f"Need web search for: {user_input[:200]}", system,
                         max_tokens=5, temperature=0.1)
            decision = (raw or '').strip().upper()[:3]
            result = decision == 'YES'
            logger.info(f"_llm_should_search → {decision!r} → {result} | '{user_input[:50]}'")
            return result
        except Exception as e:
            logger.warning(f"_llm_should_search LLM call failed: {e} — keyword fallback")
            _fallback = ['news', 'today', 'current', 'latest', 'who is', 'what is the']
        return any(kw in text for kw in _fallback)

    @staticmethod
    def _is_conversation_resume(user_input: str) -> bool:
        text = str(user_input or '').lower().strip()
        markers = (
            'back to ', 'return to ', 'continue our', 'our discussion',
            'our conversation', 'reprenons', 'revenons', 'poursuivons',
            'notre discussion', 'notre conversation',
        )
        if not any(marker in text for marker in markers):
            return False
        # This is only a routing hint. The actual research decision remains
        # semantic and is delegated to the existing YES/NO judge.
        return True

    def _quick_web_search(self, user_input: str) -> str:
        try:
            from cognition.research_mcp.web_agent import _CFG
            from cognition.research_mcp.search_providers import get_results
            from cognition.research_mcp.web_agent import _sync_fetch
            urls = re.findall(r"https?://[^\s<>()]+", user_input)
            if urls:
                lines = ["━━ DIRECT WEB PAGE ━━", "Use the fetched page as factual grounding. Cite naturally."]
                for url in dict.fromkeys(urls)[:3]:
                    url = url.rstrip(".,;:!?)]}")
                    title, content, error = _sync_fetch(url)
                    if error or not content.strip():
                        self._last_web_sources.append({
                            "url": url, "title": title or url, "kind": "fetch_failed",
                            "status": "failed", "detail": str(error or "empty page")[:180],
                        })
                        continue
                    self._last_web_sources.append({
                        "url": url, "title": title or url, "kind": "page",
                        "status": "fetched", "detail": "Page content fetched for this turn",
                    })
                    lines.append(f"[PAGE] {title or url}\n{content.strip()[:5000]}")
                if len(lines) > 2:
                    logger.info("🔍 Direct web page fetch: %d URL(s)", len(self._last_web_sources))
                    return "\n".join(lines)
                return ""

            logger.info(f"🔍 Web search triggered for: '{user_input[:60]}' | backend={_CFG.backend!r} brave={'yes' if _CFG.brave_key else 'no'} rss={_CFG.use_rss}")
            results = get_results(user_input, cfg=_CFG, num=3)
            if not results:
                logger.info("🔍 Web search: no results returned")
                return ""
            lines = ["━━ LIVE WEB SEARCH ━━",
                     "Use as factual grounding. Cite naturally."]
            for i, r in enumerate(results, 1):
                snippet = (r.get("content") or r.get("snippet") or "").strip()[:300]
                if snippet:
                    self._last_web_sources.append({
                        "url": r.get("url", ""), "title": r.get("title", ""),
                        "kind": "search_result", "status": "snippet",
                        "detail": f"Search result via {r.get('source', 'unknown')}",
                    })
                    lines.append(f"[{i}] {r.get('title','').strip()}\n{snippet}")
            logger.info(f"🔍 Web search: {len(results)} results via {[r.get('source','?') for r in results]}")
            return "\n".join(lines) if len(lines) > 2 else ""
        except Exception as e:
            logger.warning(f"_quick_web_search failed: {e}")
            return ""

    def should_extend_answer(self, user_input: str) -> bool:
        """RAC check: should this query trigger Mode 2 (Extended Answer)?"""
        if self._research:
            return self._research.rac.should_extend_answer(user_input)
        return False

    def trigger_dream(self):
        if self._system:
            try:
                return self._system.dream_system.run_dream_cycle()
            except Exception as e:
                logger.warning(f"Dream cycle failed: {e}")

    def trigger_learning(self):
        if self._system:
            try:
                return self._system.learning_cycle.run_learning_cycle()
            except Exception as e:
                logger.warning(f"Learning cycle failed: {e}")

    def trigger_life_event(self, event_type: str | None = None) -> tuple[str, str]:
        """
        Simulate a life event for Lumina.
        Returns (scenario, reflection) strings, both empty on failure.
        event_type: optional hint e.g. 'creative', 'social', 'challenge'
        """
        if self._system:
            try:
                return self._system.simulate_life_event(event_type)
            except Exception as e:
                logger.warning(f"Life event failed: {e}")
        return "", ""

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_emotion(text: str) -> tuple[str, float]:
        lower = text.lower()
        rules = [
            (["love","great","amazing","wonderful","happy","excited","awesome"], "happy",   0.8),
            (["sad","depressed","unhappy","miss","lost","hurt"],                 "sad",      0.7),
            (["angry","annoyed","frustrated","hate","furious"],                  "angry",    0.8),
            (["scared","afraid","worried","anxious","nervous"],                  "anxious",  0.6),
            (["bored","whatever","meh","fine"],                                  "bored",    0.4),
            (["curious","wonder","interesting","really","tell me"],              "curious",  0.5),
        ]
        for keywords, emotion, intensity in rules:
            if any(kw in lower for kw in keywords):
                return emotion, intensity
        return "neutral", 0.3

    @staticmethod
    def _is_reasoning_only_response(text: str) -> bool:
        """Detect a private planning note emitted as the whole visible answer."""
        value = str(text or "").strip()
        if not value:
            return False
        unwrapped = value.strip("*_ ")
        if (
            unwrapped.startswith("(")
            and unwrapped.endswith(")")
            and re.search(
            r"\b(self[- ]correction|final check|cognitive energy check|"
            r"intent|stance|suppress|style|planning note|private reasoning)\b",
            unwrapped,
            flags=re.IGNORECASE,
            )
        ):
            return True
        # Gemma can flatten its hidden instruction channel into ordinary
        # content. These lead-ins are internal control text, not an answer.
        return bool(re.match(
            r"^(?:the\s+)?core\s+directive\s+requires\b|"
            r"^the\s+directive\s+requires\b|"
            r"^final\s+check\s+on\s+phrasing\b|"
            r"^\(?self[- ]correction:\s*",
            value,
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove internal reasoning/channel tokens before they reach the user.
        Covers: <think>, <|channel|> bare lines and blocks, [INST], <<SYS>>, <|im_start|>.
        """
        import re as _re
        # Local OpenAI-compatible servers sometimes escape channel markers as
        # ``\\</think>``. Normalize those before applying the normal filters.
        text = _re.sub(r"\\+\s*(<\/?(?:think|analysis|channel)>|<\|/?channel\|>)", r"\1", text, flags=_re.IGNORECASE)
        # 1. Bare single-line channel markers (no closing tag) — strip first
        #    e.g.  <|channel|>commentary to=assistant
        text = _re.sub("<[|]channel[|]>[^\n]*", "", text, flags=_re.IGNORECASE)

        # 2. Paired block tags — strip everything between open and close
        _PAIRS = [
            ("<think>",      "</think>"),
            ("<|channel|>",  "<|/channel|>"),
            ("[INST]",       "[/INST]"),
            ("<<SYS>>",      "<</SYS>>"),
            ("<|im_start|>", "<|im_end|>"),
        ]
        for open_t, close_t in _PAIRS:
            esc_o = _re.escape(open_t)
            esc_c = _re.escape(close_t)
            text = _re.sub(esc_o + ".*?" + esc_c, "", text,
                           flags=_re.DOTALL | _re.IGNORECASE)
            # Orphaned open tag (no matching close) — consume only to end of line
            text = _re.sub(esc_o + "[^\n]*", "", text, flags=_re.IGNORECASE)

        # 3. Any remaining bare <|token|> artifacts
        text = _re.sub("<[|][^|>\n]{1,40}[|]>", "", text)

        # 4. Collapse multiple blank lines left by removals
        text = _re.sub("\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _sanitize(text: str, max_sentences: int = 6) -> str:
        try:
            # Strip reasoning blocks before TTS
            text = PersonaBridge._strip_think(text)
            from managers.speech_sanitizer import sanitize
            return sanitize(text, max_sentences=max_sentences)
        except Exception:
            return text if isinstance(text, str) else ""

    @property
    def relational_memory(self):
        return getattr(self._system, "relational_memory", None) if self._system else None

    @property
    def is_ready(self) -> bool:
        return self._ready and self._system is not None
