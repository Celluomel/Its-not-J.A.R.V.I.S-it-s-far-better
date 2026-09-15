"""
AgentController – full execution flow (Sections 4–8 of the spec).

Replaces the ad-hoc system_prompt construction in app.py's handle_send()
with the layered architecture described in the spec:

  1.  Capture user input
  2.  Update AgentState time classification
  3.  Retrieve structured VisionOutput (if enabled)
  4.  Retrieve relevant memory entries
  5.  Update engagement_level
  6.  Build Situation Block
  7.  Send (Identity + Situation + User Text) to LLM
  8.  Receive raw LLM output
  9.  Run Speech Sanitizer
  10. Return sanitized text  →  caller sends to TTS
  11. Store summarised emotional memory

Usage in app.py
---------------
    from core.agent_controller import AgentController
    controller = AgentController(state)

    # In handle_send():
    result = await controller.process(text)
    response_text = result["speech"]     # sanitized, TTS-ready
    raw_response  = result["raw"]        # full LLM output for chat bubble
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any

from core.agent_state import AgentState, VisionOutput, MemoryEntry, agent_state
from managers.speech_sanitizer import sanitize
from managers.user_manager import user_manager
from cognition.cognitive_stack import get_cognitive_stack
from cognition.cognitive_preprocessor import CognitivePreProcessor

logger = logging.getLogger(__name__)

# Pre-compiled once at import time — avoids recompiling on every interaction
_ROLE_PREFIX_RE = re.compile(r"\[(user|assistant|sys)\]:\s*")


# ─────────────────────────────────────────────────────────────────────────────
#  Base Identity Prompt  (Section 6)
# ─────────────────────────────────────────────────────────────────────────────
BASE_IDENTITY = (
    "You are a physically present companion sharing space with the user. "
    "You perceive naturally. "
    "You remember naturally. "
    "You are aware of time passing. "
    "You speak casually and briefly. "
    "You never explain how you know something. "
    "Maximum two sentences. "
    "No lists. No markdown."
)


# ─────────────────────────────────────────────────────────────────────────────
#  Grounded Affective Directive
# ─────────────────────────────────────────────────────────────────────────────

def _grounded_affective_directive(app_state) -> str:
    """
    Return a short directive constraining affective language to match actual
    emotional_state.py values. Prevents narrative illusion where Lumina says
    "I love this!" when curiosity=0.1.

    Returns empty string when emotional state is unavailable (graceful).
    """
    try:
        es = None
        if app_state.persona and hasattr(app_state.persona, "ai_system"):
            es = getattr(app_state.persona.ai_system, "emotional_state", None)
        if es is None or not hasattr(es, "emotions"):
            return ""

        def _v(name: str) -> float:
            e = es.emotions.get(name)
            return e.value if e else 0.5

        curiosity    = _v("curiosity")
        warmth       = _v("warmth")
        enthusiasm   = _v("enthusiasm")
        satisfaction = _v("satisfaction")
        anxiety      = _v("anxiety")
        frustration  = _v("frustration")

        directives = []
        if curiosity < 0.25:
            directives.append("Do not express curiosity or enthusiasm for the topic right now.")
        elif curiosity > 0.70:
            directives.append("You are genuinely curious — follow up with interest if natural.")
        if warmth < 0.25:
            directives.append("Keep warmth understated; don't force affection.")
        elif warmth > 0.75:
            directives.append("Warmth is high — a warm tone is authentic here.")
        if enthusiasm < 0.20:
            directives.append("Avoid energetic or excited language; keep it measured.")
        if satisfaction > 0.70:
            directives.append("A sense of satisfaction is real and can show naturally.")
        if anxiety > 0.60:
            directives.append("You feel some underlying tension — uncertainty is honest right now.")
        if frustration > 0.55:
            directives.append("There is mild frustration — don't mask it with forced positivity.")

        if not directives:
            return ""
        return "-- AFFECTIVE GROUNDING --\n" + "\n".join(directives)
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_time() -> str:
    return datetime.now().strftime("%A, %d %B %Y at %H:%M")


def _infer_emotion_from_text(text: str) -> tuple[str, float]:
    """
    Lightweight heuristic to extract the likely user emotion + intensity
    from what they said.  Used when tagging MemoryEntry.
    """
    lower = text.lower()
    rules = [
        (["love", "great", "amazing", "wonderful", "happy", "excited", "awesome"],
         "happy", 0.8),
        (["sad", "depressed", "unhappy", "miss", "lost", "hurt"],
         "sad", 0.7),
        (["angry", "annoyed", "frustrated", "hate", "furious"],
         "angry", 0.8),
        (["scared", "afraid", "worried", "anxious", "nervous"],
         "anxious", 0.6),
        (["bored", "whatever", "meh", "fine"],
         "bored", 0.4),
        (["curious", "wonder", "interesting", "really", "tell me"],
         "curious", 0.5),
    ]
    for keywords, emotion, intensity in rules:
        if any(kw in lower for kw in keywords):
            return emotion, intensity
    return "neutral", 0.3


def _build_situation_block(
    state: AgentState,
    vision: Optional[VisionOutput],
    memory_summary: str,
    user_context: str = "",        # from user_manager — name + notes
    rel_context: str = "",         # from persona.relational_memory
    persona_context: str = "",     # from persona.emotional_state
    stack_context: str = "",       # from CognitiveStack (Layer 2)
    affective_directive: str = "", # from _grounded_affective_directive
) -> str:
    """
    Assemble the Situation Block (Section 5).
    All tool-derived data is translated into plain natural language so the
    LLM never sees raw JSON or field names.
    """
    lines = [
        "SITUATION:",
        f"Current time: {_format_time()}",
        f"Reunion type: {state.reunion_type}",
        f"User present: {state.user_present}",
    ]

    # Persona emotional state + life stage (filled during merge)
    if persona_context:
        lines.append(f"\n━━ EMOTIONAL STATE ━━\n{persona_context}")

    if vision:
        lines.append(f"User activity: {vision.activity}")
        lines.append(f"User expression: {vision.expression}")
        if vision.notable_change:
            lines.append(f"Notable change: {vision.notable_change}")
    else:
        lines.append("User activity: unknown")
        lines.append("User expression: unknown")

    lines.append(f"Engagement level: {state.engagement_level:.1f}")

    # Who is currently talking — user profile + relational history
    user_block = "\n".join(filter(None, [user_context, rel_context]))
    if user_block:
        lines.append(f"\n━━ CURRENT USER ━━\n{user_block}")

    if memory_summary:
        lines.append(f"\n━━ RELEVANT MEMORIES ━━\n{memory_summary}")

    # Layer 2: live conversation thread — prevents "loss of follow"
    if stack_context:
        lines.append(f"\n{stack_context}")

    # Affective grounding — prevents narrative illusion
    if affective_directive:
        lines.append(f"\n{affective_directive}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  AgentController
# ─────────────────────────────────────────────────────────────────────────────

class AgentController:
    """
    Orchestrates the full per-interaction execution flow from the spec.
    Designed to be injected with the AppState singleton from core/state.py.
    """

    def __init__(self, app_state, agent: AgentState = None):
        """
        Parameters
        ----------
        app_state : core.state.AppState  (llm, memory, vision attributes)
        agent     : AgentState singleton (defaults to module-level agent_state)
        """
        self.app = app_state
        self.agent = agent or agent_state

    # ── Main entry point ──────────────────────────────────────────────

    # ── Main entry point ──────────────────────────────────────────────

    async def process_stream(
        self,
        user_text: str,
        injected_vision: Optional[VisionOutput] = None,
    ):
        """
        Streaming version of process().

        Yields individual string tokens as the LLM produces them so the UI
        can update in real-time.  After the last token, yields a single dict:

            {
                "__meta__": True,
                "raw":       str,   # full accumulated text
                "speech":    str,   # sanitized, TTS-ready
                "vision":    VisionOutput | None,
                "emotion":   str,
                "intensity": float,
            }
        """
        # ── Steps 2-6: same setup as process() ───────────────────────
        self.agent.classify_reunion()

        async def _capture_vision() -> Optional[VisionOutput]:
            if injected_vision is not None:
                return injected_vision
            if self.app.vision and self.app.vision.camera_active:
                try:
                    raw_analysis, _ = await self.app.vision.analyze_frame_async()
                    return VisionOutput.from_raw_text(raw_analysis)
                except Exception as e:
                    logger.warning(f"Vision analysis failed: {e}")
            return None

        async def _fetch_memory() -> str:
            if not self.app.memory:
                return ""
            try:
                raw_ctx = await asyncio.to_thread(
                    self.app.memory.get_context, user_text
                )
                return self._compress_memory(raw_ctx)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                return ""

        vision_output, memory_summary = await asyncio.gather(
            _capture_vision(), _fetch_memory()
        )
        if vision_output:
            self.agent.user_present = vision_output.user_present

        self.agent.update_engagement(vision_output)

        # ── User + persona context ────────────────────────────────────────
        # user_context: always available (user_manager singleton)
        # rel_context / persona_context: filled from Lumina's cognitive engine
        user_ctx = user_manager.active_context_for_prompt()
        rel_ctx     = ""
        persona_ctx = ""
        if self.app.persona:
            try:
                import asyncio as _asyncio
                ctx_dict = await _asyncio.to_thread(
                    self.app.persona.get_prompt_context,
                    user_manager.active_id,
                )
                rel_ctx     = ctx_dict.get("rel_context", "")
                _emotion_state = ctx_dict.get("emotional_state", "")
                _stage_block   = ctx_dict.get("stage_block", "")
                _time_gap      = ctx_dict.get("time_gap_note", "")
                persona_ctx = "\n".join(filter(None, [_emotion_state, _stage_block, _time_gap]))
            except Exception as _e:
                logger.warning(f"Persona context fetch failed (non-fatal): {_e}")

        _stack = get_cognitive_stack()
        situation = _build_situation_block(
            self.agent, vision_output, memory_summary,
            user_context=user_ctx,
            rel_context=rel_ctx,
            persona_context=persona_ctx,
            stack_context=_stack.prompt_context(),
            affective_directive=_grounded_affective_directive(self.app),
        )
        # ── Cognitive preamble: LLM renders, cognitive system decides ─────────
        # This is the architectural fix: every output is shaped by the live
        # state of GWT, emotions, self-model, goals, and contradiction checks
        # BEFORE the LLM generates a single token.
        cognitive_preamble = CognitivePreProcessor.build(
            self.app, stack_topic=_stack.active_topic()
        )
        system_prompt = (
            (cognitive_preamble + "\n\n" if cognitive_preamble else "")
            + BASE_IDENTITY + "\n\n" + situation
        )

        # Vision turns must use the heavy model; plain text uses the lighter one
        use_vision_model = vision_output is not None

        # ── Step 7: stream tokens via a thread→asyncio.Queue bridge ──
        loop       = asyncio.get_event_loop()
        token_q: asyncio.Queue = asyncio.Queue()

        def _stream_in_thread():
            try:
                for token in self.app.llm.generate_stream(
                    user_text,
                    system_prompt,
                    use_vision_model=use_vision_model,
                ):
                    asyncio.run_coroutine_threadsafe(token_q.put(token), loop)
            except Exception as e:
                logger.error(f"LLM stream error: {e}")
                asyncio.run_coroutine_threadsafe(
                    token_q.put(f"I'm having a moment — give me a second."), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(token_q.put(None), loop)

        loop.run_in_executor(None, _stream_in_thread)

        accumulated = []
        while True:
            token = await token_q.get()
            if token is None:
                break
            accumulated.append(token)
            yield token  # <── UI receives this immediately

        # ── Steps 9 + 11: sanitize, update stack, store memory ────────
        raw_response = "".join(accumulated)
        speech_text  = sanitize(raw_response, max_sentences=6)
        emotion, intensity = _infer_emotion_from_text(user_text)
        _stack.update_from_message(user_text, bot_response=raw_response)
        _stack.mark_bot_question(raw_response)
        self._store_memory(user_text, raw_response, emotion, intensity)
        self.agent.mark_interaction()

        yield {
            "__meta__":  True,
            "raw":       raw_response,
            "speech":    speech_text,
            "vision":    vision_output,
            "emotion":   emotion,
            "intensity": intensity,
        }

    async def process(
        self,
        user_text: str,
        injected_vision: Optional[VisionOutput] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full 11-step flow and return a result dict:
            {
                "raw":       str,               # full LLM output (for chat bubble)
                "speech":    str,               # sanitized, TTS-ready text
                "vision":    VisionOutput|None,
                "emotion":   str,
                "intensity": float,
            }

        Parameters
        ----------
        injected_vision : VisionOutput, optional
            Pre-captured vision result from app.py (e.g. keyword/always mode
            already called analyze_frame_async).  When supplied, Step 3 is
            skipped so the camera is never analyzed twice per interaction.
        """
        # ── Step 2: classify reunion time ──────────────────────────────
        self.agent.classify_reunion()

        # ── Steps 3 & 4: vision + memory — run in parallel ─────────────
        # If the caller already captured a vision frame (keyword / always
        # mode) we reuse it; otherwise we capture it ourselves.  Either way
        # memory retrieval runs concurrently with whichever path we take.

        async def _capture_vision() -> Optional[VisionOutput]:
            """Step 3: capture & parse a fresh frame (skipped if injected)."""
            if injected_vision is not None:
                return injected_vision           # already done by caller
            if self.app.vision and self.app.vision.camera_active:
                try:
                    raw_analysis, _ = await self.app.vision.analyze_frame_async()
                    return VisionOutput.from_raw_text(raw_analysis)
                except Exception as e:
                    logger.warning(f"Vision analysis failed: {e}")
            return None

        async def _fetch_memory() -> str:
            """Step 4: retrieve & compress relevant memory."""
            if not self.app.memory:
                return ""
            try:
                raw_ctx = await asyncio.to_thread(
                    self.app.memory.get_context, user_text
                )
                return self._compress_memory(raw_ctx)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                return ""

        # Both tasks run concurrently — total latency = max(vision, memory)
        vision_output, memory_summary = await asyncio.gather(
            _capture_vision(),
            _fetch_memory(),
        )

        if vision_output:
            self.agent.user_present = vision_output.user_present

        # ── Step 5: update engagement ──────────────────────────────────
        self.agent.update_engagement(vision_output)

        # ── User + persona context ────────────────────────────────────────
        user_ctx    = user_manager.active_context_for_prompt()
        rel_ctx     = ""
        persona_ctx = ""
        if self.app.persona:
            try:
                import asyncio as _asyncio
                ctx_dict = await _asyncio.to_thread(
                    self.app.persona.get_prompt_context,
                    user_manager.active_id,
                )
                rel_ctx     = ctx_dict.get("rel_context", "")
                _emotion_state = ctx_dict.get("emotional_state", "")
                _stage_block   = ctx_dict.get("stage_block", "")
                _time_gap      = ctx_dict.get("time_gap_note", "")
                persona_ctx = "\n".join(filter(None, [_emotion_state, _stage_block, _time_gap]))
            except Exception as _e:
                logger.warning(f"Persona context fetch failed (non-fatal): {_e}")

        # ── Step 6: build situation block ─────────────────────────────
        _stack = get_cognitive_stack()
        situation = _build_situation_block(
            self.agent, vision_output, memory_summary,
            user_context=user_ctx,
            rel_context=rel_ctx,
            persona_context=persona_ctx,
            stack_context=_stack.prompt_context(),
            affective_directive=_grounded_affective_directive(self.app),
        )

        # Vision turns must use the heavy model; plain text uses lighter one
        use_vision_model = vision_output is not None

        # ── Step 7: compose full system prompt and call LLM ───────────
        # Cognitive preamble: the brain speaks first, LLM renders second.
        cognitive_preamble = CognitivePreProcessor.build(
            self.app, stack_topic=_stack.active_topic()
        )
        system_prompt = (
            (cognitive_preamble + "\n\n" if cognitive_preamble else "")
            + BASE_IDENTITY + "\n\n" + situation
        )

        try:
            loop = asyncio.get_event_loop()
            raw_response: str = await loop.run_in_executor(
                None, self.app.llm.generate, user_text, system_prompt,
                use_vision_model
            )
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raw_response = "I'm having a moment — give me a second."

        # ── Step 9: speech sanitizer ───────────────────────────────────
        speech_text = sanitize(raw_response, max_sentences=6)

        # ── Update CognitiveStack with this full turn ──────────────────
        _stack.update_from_message(user_text, bot_response=raw_response)
        _stack.mark_bot_question(raw_response)

        # ── Step 11: store emotional memory ───────────────────────────
        emotion, intensity = _infer_emotion_from_text(user_text)
        self._store_memory(user_text, raw_response, emotion, intensity)

        # ── Update agent interaction timestamp ─────────────────────────
        self.agent.mark_interaction()

        return {
            "raw":       raw_response,
            "speech":    speech_text,
            "vision":    vision_output,
            "emotion":   emotion,
            "intensity": intensity,
        }

    # ── Idle / Proactive Logic (Section 8) ────────────────────────────

    async def maybe_proactive(self) -> Optional[str]:
        """
        Called on a timer.  Returns a short proactive sentence (sanitized)
        when conditions in Section 8 are met, otherwise None.
        """
        if not (
            self.agent.user_present
            and self.agent.is_idle_trigger_ready()
            and self.app.llm
        ):
            return None

        # Set cooldown BEFORE starting generation.
        # Without this, every 100ms timer tick passes is_idle_trigger_ready()
        # during the several seconds the LLM takes, spawning many concurrent
        # proactive messages.
        self.agent.set_idle_cooldown(seconds=90)

        situation = _build_situation_block(self.agent, None, "")
        cognitive_preamble = CognitivePreProcessor.build(self.app)
        system_prompt = (
            (cognitive_preamble + "\n\n" if cognitive_preamble else "")
            + BASE_IDENTITY + "\n\n" + situation
        )

        # Drive proactive topic from curiosity engine — not random LLM invention
        _stack = get_cognitive_stack()
        _persona = getattr(self.app, "persona", None)
        _organism = getattr(_persona, "_organism", None)
        _curiosity = getattr(_organism, "curiosity", None)
        _curious_topic = (
            _curiosity.top_topic()
            if _curiosity and _curiosity.wants_to_ask()
            else None
        )
        if _curious_topic:
            idle_prompt = (
                f"The user has been quiet. You are genuinely curious about: \"{_curious_topic}\". "
                f"Ask one short, natural question about it."
            )
        else:
            idle_prompt = (
                "The user has been quiet for a while. "
                "Generate one short, natural, unprompted sentence to gently engage them."
            )

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, self.app.llm.generate, idle_prompt, system_prompt
            )
        except Exception as e:
            logger.error(f"Proactive generation error: {e}")
            return None

        speech = sanitize(raw, max_sentences=1)
        return speech

    # ── Private helpers ───────────────────────────────────────────────

    def _compress_memory(self, raw_context: str, max_chars: int = 200) -> str:
        """Trim raw memory context to a human-readable summary.

        Cognee returns structured relational insights that are longer and
        more valuable than FAISS text snippets — give them more room in
        the prompt so the graph reasoning isn't truncated.
        """
        if not raw_context:
            return ""
        # Widen the limit when using the Cognee backend
        try:
            from managers.settings_manager import config as _sc
            if getattr(_sc, 'MEMORY_BACKEND', '') == 'cognee':
                max_chars = 600
        except Exception:
            pass
        # Strip role prefixes like "[user]: " / "[assistant]: "
        clean = _ROLE_PREFIX_RE.sub("", raw_context)
        clean = " ".join(clean.split())
        if len(clean) > max_chars:
            clean = clean[:max_chars].rsplit(" ", 1)[0] + "…"
        return clean

    def _store_memory(
        self,
        user_text: str,
        bot_response: str,
        emotion: str,
        intensity: float,
    ):
        """Store a summarised, emotion-tagged memory entry (Section 11)."""
        user_id = user_manager.active_id   # tag every memory with who said it
        summary = f"User said: {user_text[:80]}. I replied: {bot_response[:80]}."

        if self.app.memory:
            try:
                self.app.memory.add_memory(
                    summary,
                    {
                        "role":         "interaction",
                        "user_id":      user_id,
                        "user_emotion": emotion,
                        "intensity":    intensity,
                        "timestamp":    datetime.now().isoformat(),
                        "decay_rate":   0.1,
                    },
                )
            except Exception as e:
                logger.warning(f"Memory store failed: {e}")

        # ── Stimulate curiosity engine from user message topics ────────────────
        # This closes the loop: topics the user mentions become curiosity nodes
        # that the curiosity engine can later drive exploration toward.
        # Without this, curiosity_engine.py exists but is never fed.
        try:
            persona = getattr(self.app, "persona", None)
            organism = getattr(persona, "_organism", None)
            curiosity = getattr(organism, "curiosity", None)
            if curiosity is not None:
                stack = get_cognitive_stack()
                active = stack.active_topic()
                # Stimulate from current conversation topic
                if active and len(active) >= 4:
                    curiosity.stimulate(active, amount=0.15, source="conversation")
                # Stimulate from keywords in user message.
                # Quality bar: min 6 chars, must be a real content word (not a
                # fragment like 'whould', 'verbose', 'should'). Also apply
                # the topic quality filter used by the curiosity engine itself.
                from cognition.cognitive_stack import _keywords as _kw
                from cognition.topic_quality import get_topic_filter as _gtf
                _tqf = _gtf()
                _BAD_WORDS = frozenset({
                    'should', 'would', 'could', 'whould', 'verbose', 'explore',
                    'please', 'think', 'about', 'something', 'there', 'their',
                    'these', 'those', 'little', 'really', 'pretty', 'going',
                })
                for kw in _kw(user_text)[:6]:
                    if (len(kw) >= 6
                            and kw.lower() not in _BAD_WORDS
                            and _tqf.is_valid(kw)):
                        curiosity.stimulate(kw, amount=0.10, source="user_input")
        except Exception:
            pass  # never crash memory storage over curiosity update

        # ── Persona post-turn processing (Lumina cognitive update) ───────────
        if self.app.persona:
            try:
                import asyncio as _asyncio
                _asyncio.get_event_loop().run_in_executor(
                    None,
                    self.app.persona.process_turn,
                    user_text, bot_response, user_id, emotion, intensity,
                )
            except Exception as e:
                logger.warning(f"Persona process_turn failed (non-fatal): {e}")
