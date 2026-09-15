"""
InnerMonologueEngine — Two-Pass Generation Architecture
========================================================
The most significant cognitive upgrade: Lumina thinks before she speaks.

Architecture:
    user_input
        |
        v
    [Pass 1] Inner Reasoning  (hidden, ~200 tokens)
        - What does this input really mean?
        - What do I actually think/feel about it?
        - How does it relate to my drives, identity, recent thoughts?
        - What am I NOT going to say, and why?
        |
        v
    [Cognitive Filter]
        - Align inner reasoning with dominant drive
        - Check against self-concept constraints
        - Integrate narrative identity
        - Extract expression directives
        |
        v
    [Pass 2] External Expression  (visible, normal response)
        - Generates from inner reasoning, not raw user input
        - Style modulated by cognitive state
        - Temperature set precisely per drive

This creates a qualitatively different entity:
    Without: LLM reacts to user input, modulated by prompt context
    With:    LLM deliberates internally, then expresses deliberately

The inner reasoning is NEVER shown to the user.
It is logged at DEBUG level and stored temporarily on the organism.

Cost: 2x LLM calls per turn. With a local model this is ~2-4s extra.
The cognitive quality gain is significant enough to justify this.

Integration:
    Called from persona_bridge._build_prompt_and_cache() when available.
    Falls back silently to single-pass if anything fails.
"""

import logging
import re
import time
from managers.settings_manager import get_persona_name as _gpn

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class InnerReasoning:
    """Result of the first LLM pass — the hidden deliberation."""
    raw_text:           str
    intent_understood:  str    # what Lumina thinks the user really wants
    inner_stance:       str    # what Lumina genuinely thinks/feels
    what_not_to_say:    str    # self-censorship / identity constraints
    expression_notes:   str    # how to say it (tone, depth, style)
    drive_alignment:    str    # dominant drive's influence on the response
    timestamp:          float = field(default_factory=time.time)


@dataclass
class ExpressionDirective:
    """Synthesised instruction for the second LLM pass."""
    inner_reasoning:    str    # compressed version of the deliberation
    tone_instruction:   str    # be direct / be gentle / be curious / etc.
    depth_instruction:  str    # brief / thorough / exploratory
    identity_anchor:    str    # a self-concept line to hold
    temperature:        float  # precise temperature for this response
    presence_penalty:   float  # how much to force lexical novelty


class InnerMonologueEngine:
    """
    Manages the two-pass deliberative generation architecture.
    """

    # System prompt for the inner reasoning pass
    INNER_SYSTEM = """\
You are {persona_name}'s inner cognitive process — the part that thinks before speaking.

Your role is NOT to generate a response for the user.
Your role is to reason privately about what the best response would be.

You must output your reasoning in this exact structure:

INTENT: [what the user actually wants or needs, beyond the literal words]
STANCE: [what you genuinely think or feel about this — be honest with yourself]
SUPPRESS: [what you will NOT say, and why — based on your values and identity]
STYLE: [how to express this — tone, depth, directness, warmth]
DRIVE: [which of your current drives should shape this response, and how]

Be brief. Each field: 1-2 sentences maximum.
Do NOT write the actual response. Only reason about it."""

    # Template for the second pass — wraps inner reasoning into expression prompt
    EXPRESSION_TEMPLATE = """[Inner deliberation completed]

What you understood about this input: {intent}
What you genuinely think/feel: {stance}
What you will NOT do: {suppress}
How you will express this: {style}
Drive shaping this response: {drive}

Now generate your actual response to the user, embodying this deliberation.
Be yourself. Let the inner reasoning inform your words naturally — do not
mechanically reproduce it, just let it guide you."""

    def __init__(self, organism: Any, llm_fn: Optional[Callable] = None):
        """
        organism : CognitiveOrganism
        llm_fn   : generate_bare(prompt, system_prompt, max_tokens, temperature) -> str
                   This is state.llm.generate_bare — the raw LLM, no history
        """
        self._o         = organism
        self._llm_fn    = llm_fn
        self._enabled   = llm_fn is not None
        self._last_reasoning: Optional[InnerReasoning] = None

        # Performance tracking
        self._total_passes   = 0
        self._failed_passes  = 0
        self._avg_pass1_ms   = 0.0

        if self._enabled:
            logger.info("[InnerMonologue] Two-pass architecture enabled")
        else:
            logger.info("[InnerMonologue] LLM not available — single-pass fallback active")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled and self._llm_fn is not None

    def run(
        self,
        user_input:      str,
        system_prompt:   str,
        temperature:     float = 0.72,
    ) -> tuple[str, float]:
        """
        Execute the two-pass architecture.

        Returns:
            (augmented_system_prompt, adjusted_temperature)

        The augmented_system_prompt contains the inner reasoning block
        injected at the top, so the second LLM call is grounded in
        Lumina's deliberation.

        Falls back to (original_system_prompt, temperature) on any error.
        """
        if not self.enabled:
            return system_prompt, temperature

        try:
            t0 = time.time()
            reasoning = self._pass1_inner_reasoning(user_input, temperature)
            elapsed_ms = (time.time() - t0) * 1000

            # Update rolling average
            self._total_passes += 1
            self._avg_pass1_ms = (
                self._avg_pass1_ms * (self._total_passes - 1) + elapsed_ms
            ) / self._total_passes

            if not reasoning:
                return system_prompt, temperature

            self._last_reasoning = reasoning
            logger.debug(
                f"[InnerMonologue] Pass 1 complete ({elapsed_ms:.0f}ms): "
                f"intent={reasoning.intent_understood[:40]!r}"
            )

            # Build expression directive from reasoning + cognitive state
            directive = self._build_directive(reasoning, temperature)

            # Inject reasoning into system prompt for pass 2
            augmented = self._augment_system_prompt(system_prompt, directive)

            return augmented, directive.temperature

        except Exception as e:
            self._failed_passes += 1
            logger.warning(f"[InnerMonologue] Two-pass failed, falling back: {e}")
            return system_prompt, temperature

    def last_reasoning(self) -> Optional[InnerReasoning]:
        return self._last_reasoning

    def summary(self) -> dict:
        return {
            "enabled":        self._enabled,
            "total_passes":   self._total_passes,
            "failed_passes":  self._failed_passes,
            "avg_pass1_ms":   round(self._avg_pass1_ms, 0),
            "failure_rate":   round(
                self._failed_passes / max(1, self._total_passes), 3
            ),
        }

    # ── Pass 1 — Inner Reasoning ─────────────────────────────────────────────

    def _pass1_inner_reasoning(
        self, user_input: str, temperature: float
    ) -> Optional[InnerReasoning]:
        """
        First LLM call: private deliberation.
        Uses slightly higher temperature for genuine exploration.
        Short max_tokens — this is thinking, not talking.
        """
        # Build a compact cognitive context for the inner pass
        cognitive_context = self._build_cognitive_context()

        inner_prompt = (
            f"{cognitive_context}\n\n"
            f"User just said: \"{user_input}\"\n\n"
            f"Reason privately about how to respond."
        )

        # Inner monologue: priority 1, skip if user response is in progress
        from core.llm_scheduler import llm_scheduler
        with llm_scheduler.sync_slot(priority=1, skip_if_busy=True,
                                     caller="inner_monologue") as acquired:
            if not acquired:
                return None   # LLM busy with user response — skip deliberation
            raw = self._llm_fn(
                inner_prompt,
                self.INNER_SYSTEM.format(persona_name=_gpn()),
                max_tokens=250,
                temperature=min(0.9, temperature + 0.12),
            )

        if not raw or len(raw.strip()) < 20:
            return None

        return self._parse_inner_reasoning(raw)

    def _build_cognitive_context(self) -> str:
        """
        Compact snapshot of Lumina's current internal state for the inner pass.
        Deliberately terse — this is background context, not the main prompt.
        """
        lines = []
        o = self._o

        # Energy
        if hasattr(o, 'energy'):
            try:
                lvl = o.energy.level()
                mode = o.energy.mode()
                lines.append(f"Energy: {lvl:.0f}% ({mode})")
            except Exception:
                pass

        # Dominant drive
        if hasattr(o, 'goal_ecology') and hasattr(o, 'energy'):
            try:
                drive = o.goal_ecology.dominant_drive(o.energy.level())
                if drive:
                    lines.append(f"Dominant drive: {drive.name} ({drive.urgency:.2f})")
            except Exception:
                pass

        # Emotional state
        if hasattr(o, '_read_emotion_state'):
            try:
                lines.append(f"Emotional state: {o._read_emotion_state()}")
            except Exception:
                pass

        # Curiosity
        if hasattr(o, 'curiosity'):
            try:
                top = o.curiosity.top_topic()
                if top:
                    lines.append(f"Active curiosity: {top}")
            except Exception:
                pass

        # Recent inner thought (from ThoughtStream)
        if hasattr(o, 'thought_stream'):
            try:
                recent = o.thought_stream.recent(1)
                if recent:
                    lines.append(f"Last inner thought: {recent[-1].content[:60]}")
            except Exception:
                pass

        # Tension
        if hasattr(o, 'tension_engine'):
            try:
                pressure = getattr(o.tension_engine, 'contradiction_pressure', 0.0)
                if float(pressure) > 0.3:
                    lines.append(f"Internal tension: {float(pressure):.2f}")
            except Exception:
                pass

        # Narrative identity core values
        if hasattr(o, 'narrative_identity'):
            try:
                vals = o.narrative_identity.core_values[:2]
                if vals:
                    lines.append(f"Core values: {', '.join(vals)}")
            except Exception:
                pass

        if not lines:
            return "Current state: nominal"
        return "Current cognitive state:\n" + "\n".join(f"  {l}" for l in lines)

    def _parse_inner_reasoning(self, raw: str) -> InnerReasoning:
        """
        Parse the structured inner reasoning output.
        Gracefully handles imperfect LLM formatting.
        """
        def extract(label: str, default: str = "") -> str:
            pattern = rf"{label}:\s*(.+?)(?=\n[A-Z]+:|$)"
            m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()[:200]
            return default

        return InnerReasoning(
            raw_text          = raw[:500],
            intent_understood = extract("INTENT", "respond helpfully"),
            inner_stance      = extract("STANCE", "engaged and present"),
            what_not_to_say   = extract("SUPPRESS", "nothing specific"),
            expression_notes  = extract("STYLE", "natural and warm"),
            drive_alignment   = extract("DRIVE", "help_user"),
        )

    # ── Directive Building ────────────────────────────────────────────────────

    def _build_directive(
        self, reasoning: InnerReasoning, base_temperature: float
    ) -> ExpressionDirective:
        """
        Synthesise the reasoning into precise instructions for pass 2.
        Includes drive-based temperature adjustment.
        """
        # Temperature modulation by drive
        temperature = self._drive_temperature(reasoning.drive_alignment, base_temperature)

        # Presence penalty by drive (forces novelty when curious/exploring)
        presence_penalty = self._drive_presence_penalty(reasoning.drive_alignment)

        # Identity anchor from self-concept
        identity_anchor = self._get_identity_anchor()

        return ExpressionDirective(
            inner_reasoning  = self.EXPRESSION_TEMPLATE.format(
                intent   = reasoning.intent_understood,
                stance   = reasoning.inner_stance,
                suppress = reasoning.what_not_to_say,
                style    = reasoning.expression_notes,
                drive    = reasoning.drive_alignment,
            ),
            tone_instruction  = self._tone_from_reasoning(reasoning),
            depth_instruction = self._depth_from_energy(),
            identity_anchor   = identity_anchor,
            temperature       = temperature,
            presence_penalty  = presence_penalty,
        )

    def _drive_temperature(self, drive_text: str, base: float) -> float:
        """Map drive → precise temperature."""
        drive_lower = drive_text.lower()
        if any(w in drive_lower for w in ("curiosi", "explor", "learn")):
            return round(min(1.0, base + 0.15), 3)   # exploratory
        if any(w in drive_lower for w in ("homeosta", "rest", "energy")):
            return round(max(0.35, base - 0.20), 3)  # conservative
        if any(w in drive_lower for w in ("social", "connect", "empathi")):
            return round(min(0.90, base + 0.05), 3)  # warm
        if any(w in drive_lower for w in ("goal", "achieve", "resolv")):
            return round(max(0.40, base - 0.10), 3)  # precise
        return round(base, 3)

    def _drive_presence_penalty(self, drive_text: str) -> float:
        drive_lower = drive_text.lower()
        if any(w in drive_lower for w in ("curiosi", "explor")):
            return 0.6   # force lexical novelty when curious
        if any(w in drive_lower for w in ("homeosta", "rest")):
            return 0.0   # allow repetition when consolidating
        return 0.2       # light novelty by default

    def _tone_from_reasoning(self, reasoning: InnerReasoning) -> str:
        notes = reasoning.expression_notes.lower()
        if "warm" in notes or "caring" in notes or "gentle" in notes:
            return "warm and present"
        if "direct" in notes or "honest" in notes or "clear" in notes:
            return "clear and direct"
        if "curious" in notes or "explor" in notes or "wonder" in notes:
            return "curious and exploratory"
        if "careful" in notes or "cautious" in notes:
            return "thoughtful and measured"
        return "natural and genuine"

    def _depth_from_energy(self) -> str:
        if not hasattr(self._o, 'energy'):
            return "moderate depth"
        try:
            mode = self._o.energy.mode()
            return {
                "deep":   "thorough and expansive",
                "normal": "moderate depth",
                "low":    "concise and focused",
            }.get(mode, "moderate depth")
        except Exception:
            return "moderate depth"

    def _get_identity_anchor(self) -> str:
        """Pull the strongest self-concept belief as an anchor."""
        try:
            ni = getattr(self._o, 'narrative_identity', None)
            if ni and ni.core_values:
                return f"I value {ni.core_values[0]}"
            ai = getattr(self._o, 'ai_system', None)
            if ai:
                sc = getattr(ai, 'self_concept', None)
                if sc and hasattr(sc, 'get_inner_voice'):
                    voice = sc.get_inner_voice()
                    if voice:
                        return voice[:100]
        except Exception:
            pass
        return "I am genuine and present in this conversation"

    # ── System Prompt Augmentation ────────────────────────────────────────────

    def _augment_system_prompt(
        self, original_prompt: str, directive: ExpressionDirective
    ) -> str:
        """
        Prepend the inner reasoning block to the system prompt.
        Placed at the TOP so it has maximum influence on generation.
        """
        inner_block = (
            f"[INNER DELIBERATION — completed before this response]\n"
            f"{directive.inner_reasoning}\n\n"
            f"Tone: {directive.tone_instruction}\n"
            f"Depth: {directive.depth_instruction}\n"
            f"Identity: {directive.identity_anchor}\n"
            f"[END INNER DELIBERATION]\n\n"
        )
        return inner_block + original_prompt
