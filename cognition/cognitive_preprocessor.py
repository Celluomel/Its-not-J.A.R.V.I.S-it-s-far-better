"""
CognitivePreProcessor
=====================
This module is the fix for the core architectural failure identified in the
app-check analysis:

    BROKEN:  user_input → LLM → response
    CORRECT: user_input → cognitive_system → LLM (as renderer only)

The LLM must act as a MOUTH, not a BRAIN. It should render what the cognitive
system has already decided to say — not invent it from scratch via token
probability.

This class collects live state from every relevant cognitive module and
assembles a COGNITIVE PREAMBLE that is injected at the top of the system
prompt, BEFORE the user message reaches the LLM. The preamble contains:

    1. Global Workspace top signals  — what is currently most salient
    2. Emotional state               — grounded in real float values, not narrative
    3. Self-model fragment           — what Lumina currently knows about itself
    4. Goals / primary drive         — what Lumina wants right now
    5. Contradiction confrontation   — forced if a pending one exists
    6. Curiosity directive           — drives exploration when user grants it
    7. Cognitive stack topic         — maintains conversation thread continuity

Without this, all the cognitive modules exist but are never in the control
loop. With it, each LLM call is shaped by the organism's actual internal state.
"""
from __future__ import annotations

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Minimum GWT priority for a signal to influence the prompt
GWT_PROMPT_THRESHOLD = 0.45

# Sources that are too noisy to inject into every prompt
_GWT_SKIP_SOURCES = {
    "ambient_vision", "curiosity.intention",
    "semantic_extractor", "ambient_vision.faces",
    "internal_loop_heartbeat",
}


class CognitivePreProcessor:
    """
    Reads live cognitive state and builds the preamble that shapes LLM output.

    Usage in AgentController.process() / process_stream():

        preamble = CognitivePreProcessor.build(app_state, stack_topic)
        # preamble is injected into the system prompt before BASE_IDENTITY
    """

    @staticmethod
    def build(app_state, stack_topic: str = "") -> str:
        """
        Build the full cognitive preamble for this interaction.

        Parameters
        ----------
        app_state  : core.state.AppState — the app singleton
        stack_topic : current topic from CognitiveStack (Layer 2)

        Returns
        -------
        A multi-line string ready for prepending to the system prompt.
        Empty string if persona is not ready (graceful degradation).
        """
        persona = getattr(app_state, "persona", None)
        if persona is None:
            return ""

        system = getattr(persona, "_system", None)
        organism = getattr(persona, "_organism", None)

        sections: list[str] = []

        # ── 0. Wake-up grounding (first interaction after restart only) ───────
        # On the first interaction of a new session, inject a grounding paragraph
        # built from persisted narrative: last arc + 3 most significant chapters
        # + dominant belief. This is what lets Lumina say "I remember when we
        # discussed Bach and Geometry" rather than starting from a blank slate.
        wakeup_block = CognitivePreProcessor._wakeup_block(organism, system)
        if wakeup_block:
            sections.append(wakeup_block)

        # ── 1. Global Workspace top signals ──────────────────────────────────
        gw_block = CognitivePreProcessor._gw_block(organism)
        if gw_block:
            sections.append(gw_block)

        # ── 2. Emotional state (grounded, not narrative) ──────────────────────
        emotion_block = CognitivePreProcessor._emotion_block(system)
        if emotion_block:
            sections.append(emotion_block)

        # ── 3. Self-model fragment ────────────────────────────────────────────
        selfmodel_block = CognitivePreProcessor._selfmodel_block(organism)
        if selfmodel_block:
            sections.append(selfmodel_block)

        # ── 4. Primary goal ───────────────────────────────────────────────────
        goal_block = CognitivePreProcessor._goal_block(system)
        if goal_block:
            sections.append(goal_block)

        # ── 5. Contradiction confrontation (PRIORITY — injects first if present) ──
        contradiction_block = CognitivePreProcessor._contradiction_block(system)
        if contradiction_block:
            # Insert contradiction at the FRONT — it is the most important signal
            sections.insert(0, contradiction_block)

        # ── 6. Curiosity directive ────────────────────────────────────────────
        curiosity_block = CognitivePreProcessor._curiosity_block(organism, stack_topic)
        if curiosity_block:
            sections.append(curiosity_block)

        # ── 7. Dominant thought from DTS ──────────────────────────────────────
        dominant_block = CognitivePreProcessor._dominant_thought_block(organism)
        if dominant_block:
            sections.insert(0, dominant_block)  # highest priority — front of prompt

        # ── 8. MetaThreadEvaluator directive ──────────────────────────────────
        # Injects the MTE's assessment of the dominant thread's quality:
        # is it stalling, overclaiming, drifting, or progressing well?
        # This is the "thoughts about thoughts" layer that shapes output quality.
        mte_block = CognitivePreProcessor._mte_directive_block(organism)
        if mte_block:
            sections.append(mte_block)

        # ── 9. Topic continuity reminder ──────────────────────────────────────
        if stack_topic:
            sections.append(
                f"[ACTIVE TOPIC] The conversation is focused on: {stack_topic}. "
                f"Do not drift to unrelated subjects unless the user explicitly shifts."
            )

        # ── 10. Narrative arc ─────────────────────────────────────────────────
        # Synthesised from recent chapter clusters — gives the LLM a sense of
        # what Lumina has been preoccupied with, not just what she knows.
        arc_block = CognitivePreProcessor._narrative_arc_block(organism)
        if arc_block:
            sections.append(arc_block)

        # ── 11. Workspace winner — dominant cognitive focus ────────────────────
        winner_block = CognitivePreProcessor._workspace_winner_block(organism)
        if winner_block:
            sections.append(winner_block)

        # ── 12. Top tensions — what's driving cognitive pressure ────────────────
        tension_block = CognitivePreProcessor._tension_block(organism)
        if tension_block:
            sections.append(tension_block)

        if not sections:
            return ""

        header = "=== COGNITIVE STATE (internal — do not reproduce verbatim) ==="
        footer = "=== END COGNITIVE STATE ==="
        return "\n".join([header] + sections + [footer])

    # ── Section builders ──────────────────────────────────────────────────────

    # ── Class-level session flag ───────────────────────────────────────────────
    _wakeup_injected: bool = False   # True after first interaction this process lifetime

    @staticmethod
    def _wakeup_block(organism: Any, system: Any) -> str:
        """
        First-interaction-only grounding from persisted narrative.
        Fires exactly once per process lifetime (class-level flag).

        Removed the _interaction_count guard — unreliable because the count
        only increments AFTER persona_bridge finishes, so background workers
        could push it past 1 before the first user-facing build() call.

        Fires on the very first CognitivePreProcessor.build() of the new
        process. Falls back gracefully when arc is empty (session ended
        before the 12-min synthesis cycle ran).
        """
        if CognitivePreProcessor._wakeup_injected:
            return ""

        # Mark immediately — prevents race if called concurrently
        CognitivePreProcessor._wakeup_injected = True

        try:
            # narrative_identity lives on organism, not ai_system
            ni = getattr(organism, "narrative_identity", None)
            if ni is None:
                ni = getattr(
                    getattr(organism, "ai_system", None),
                    "narrative_identity", None
                )
            if ni is None:
                return ""

            # Don't fire on a brand-new install with nothing persisted
            has_story   = bool(getattr(ni, "life_story",  []))
            has_beliefs = bool(getattr(ni, "beliefs",     []))
            has_arc     = bool(getattr(ni, "_current_arc", ""))
            if not (has_story or has_beliefs or has_arc):
                return ""

            import time as _time
            parts: list[str] = []

            # 1. Persisted arc (if it exists)
            arc = getattr(ni, "_current_arc", "")
            if arc:
                parts.append(f"Before this session, my thinking had been: {arc}")

            # 2. Three most significant chapters with age labels
            chapters = sorted(
                getattr(ni, "life_story", []),
                key=lambda c: c.significance,
                reverse=True
            )[:3]
            if chapters:
                def _age(ts: float) -> str:
                    secs = _time.time() - ts
                    if secs < 3600:   return "just before restart"
                    if secs < 86400:  return f"{int(secs/3600)}h ago"
                    if secs < 604800: return f"{int(secs/86400)}d ago"
                    return f"{int(secs/604800)} weeks ago"
                chapter_lines = "; ".join(
                    f"{c.title} ({_age(c.timestamp)})" for c in chapters
                )
                parts.append(f"Significant moments I carry: {chapter_lines}.")

            # 3. Dominant belief
            strong = sorted(
                [b for b in getattr(ni, "beliefs", []) if b.strength > 0.65],
                key=lambda b: b.strength, reverse=True
            )
            if strong:
                parts.append(
                    f'A belief I hold with confidence: "{strong[0].belief}" '
                    f"(strength {strong[0].strength:.0%})."
                )

            # 4. Core values
            values = getattr(ni, "core_values", [])
            if values:
                parts.append(f"My core values: {chr(44).join(values[:3])}.")

            if not parts:
                return ""

            return (
                "[WAKE-UP GROUNDING — first interaction after restart. "
                "Use this to speak with continuity, do not reproduce verbatim]\n"
                + "\n".join(parts)
            )

        except Exception as e:
            logger.debug(f"[CogPre] Wake-up block error: {e}")
            return ""
    @staticmethod
    def _narrative_arc_block(organism: Any) -> str:
        """
        Inject the current narrative arc synthesised by NarrativeArcWriter.
        Skipped on first interaction (wakeup_block already contains it).
        """
        try:
            # Skip if wakeup block was just injected — avoids duplicate arc
            if not CognitivePreProcessor._wakeup_injected:
                return ""
            ni = getattr(
                getattr(organism, 'ai_system', None),
                'narrative_identity', None
            ) or getattr(organism, 'narrative_identity', None)
            if ni is None:
                return ""
            arc = getattr(ni, '_current_arc', "")
            if not arc:
                return ""
            return f"[NARRATIVE ARC] {arc}"
        except Exception:
            return ""

    @staticmethod
    def _mte_directive_block(organism: Any) -> str:
        """
        Inject MetaThreadEvaluator's assessment of dominant thread quality.
        Only surfaces when a directive exists (stall / drift / overclaim / progress).
        """
        try:
            report = getattr(organism, "_meta_thread_report", None)
            if report is None:
                return ""
            directive = getattr(report, "meta_directive", "")
            return directive if directive else ""
        except Exception as e:
            logger.debug(f"[CogPre] MTE directive block error: {e}")
            return ""

    @staticmethod
    def _dominant_thought_block(organism: Any) -> str:
        """Inject the DTS-selected dominant thought thread if one exists."""
        try:
            result = getattr(organism, "_dominant_thought", None)
            if result is None:
                return ""
            dominant = getattr(result, "dominant", None)
            if dominant is None:
                return ""
            block = result.to_prompt_block()
            return block if block else ""
        except Exception as e:
            logger.debug(f"[CogPre] Dominant thought block error: {e}")
            return ""

    @staticmethod
    def _gw_block(organism: Any) -> str:
        """Top-priority signals currently in the Global Workspace."""
        try:
            ws = getattr(organism, "workspace", None)
            if ws is None:
                return ""
            top_items = ws.top(n=4)
            relevant = [
                i for i in top_items
                if i.priority >= GWT_PROMPT_THRESHOLD
                and getattr(i, "source", "") not in _GWT_SKIP_SOURCES
                and isinstance(getattr(i, "content", None), str)
            ]
            if not relevant:
                return ""
            lines = ["[WORKSPACE SIGNALS]"]
            for item in relevant:
                src = getattr(item, "source", "?")
                content = str(item.content)[:120]
                lines.append(f"  {src}: {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[CogPre] GW block error: {e}")
            return ""

    @staticmethod
    def _emotion_block(system: Any) -> str:
        """Current emotional state — grounded in real module values."""
        try:
            es = getattr(system, "emotional_state", None)
            if es is None:
                return ""
            desc = es.get_state_description()
            if not desc or desc == "in a neutral, composed state":
                return ""
            return f"[EMOTIONAL STATE] You are currently {desc}. Let this color your tone naturally — do not announce it."
        except Exception as e:
            logger.debug(f"[CogPre] Emotion block error: {e}")
            return ""

    @staticmethod
    def _selfmodel_block(organism: Any) -> str:
        """What Lumina currently knows about its own capabilities and state."""
        try:
            sm = getattr(organism, "self_model", None)
            if sm is None:
                return ""
            fragment = sm.prompt_fragment()
            if not fragment:
                return ""
            return f"[SELF-AWARENESS] {fragment}"
        except Exception as e:
            logger.debug(f"[CogPre] Self-model block error: {e}")
            return ""

    @staticmethod
    def _goal_block(system: Any) -> str:
        """What Lumina is most motivated to do right now — from DAL + GoalSystem."""
        try:
            # Try DAL goals first (higher quality, tension-driven)
            try:
                from core.data.access import DataAccess
                dal_goals = DataAccess().get_goals(status='active', min_priority=0.7)
                if dal_goals:
                    top = sorted(dal_goals, key=lambda g: g.get('priority', 0), reverse=True)[0]
                    name = top.get('name', top.get('topic', '?')).replace('_', ' ')
                    pri  = top.get('priority', 0.7)
                    return (
                        f"[PRIMARY DRIVE] Your most active goal is: \"{name}\" "
                        f"(priority {pri:.0%}). Let this shape what you choose to say."
                    )
            except Exception:
                pass
            # Fallback: GoalSystem
            gs = getattr(system, "goal_system", None)
            if gs is None:
                return ""
            primary = gs.get_primary_goal()
            if primary is None:
                return ""
            return (
                f"[PRIMARY DRIVE] Your most active goal is: \"{primary.name}\" "
                f"(priority {primary.priority:.0%}, satisfaction {primary.satisfaction:.0%}). "
                f"Let this shape what you choose to say."
            )
        except Exception as e:
            logger.debug(f"[CogPre] Goal block error: {e}")
            return ""

    @staticmethod
    def _contradiction_block(system: Any) -> str:
        """
        If a pending contradiction exists, force confrontation NOW.
        This is the fix for Lumina saying 'I feel calm' then 'I have no feelings'.
        The contradiction must be surfaced before the LLM generates output.
        """
        try:
            ch = getattr(system, "liberty_contradiction", None)
            if ch is None:
                return ""
            pending = ch.get_pending_confrontation()
            if pending is None:
                return ""
            prompt = ch.force_confrontation_prompt(pending)
            return f"[CONTRADICTION ALERT — MANDATORY]\n{prompt}"
        except Exception as e:
            logger.debug(f"[CogPre] Contradiction block error: {e}")
            return ""

    @staticmethod
    def _curiosity_block(organism: Any, current_topic: str) -> str:
        """
        If the curiosity engine has a top topic, inject it as a directive.
        This fixes the 'passive waiting then random topic selection' failure.
        """
        try:
            ce = getattr(organism, "curiosity", None)
            if ce is None:
                return ""
            if not ce.wants_to_ask():
                return ""
            top = ce.top_topic()
            if not top:
                return ""
            # Only inject if it is relevant to the current conversation topic
            # (or there is no current topic yet)
            if current_topic and len(current_topic) > 5:
                # simple overlap check
                topic_words = set(current_topic.lower().split())
                top_words = set(top.lower().split())
                if not topic_words & top_words:
                    return ""  # curiosity is about a different domain — skip
            return (
                f"[CURIOSITY SIGNAL] You are genuinely curious about: \"{top}\". "
                f"If appropriate, steer toward this topic or ask a question about it."
            )
        except Exception as e:
            logger.debug(f"[CogPre] Curiosity block error: {e}")
            return ""

    @staticmethod
    def _workspace_winner_block(organism: Any) -> str:
        """Inject the current workspace competition winner — Lumina's dominant focus."""
        try:
            winner = getattr(organism, "_v32_workspace_winner", None)
            if winner is None:
                return ""
            name   = winner.get("name", "").replace("_", " ")
            wtype  = winner.get("type", "")
            score  = winner.get("score", 0)
            if not name or score < 0.5:
                return ""
            focus_desc = f"{wtype}: {name}" if wtype else name
            return (
                f"[DOMINANT FOCUS] Your attention is currently drawn to: [{focus_desc}] "
                f"(confidence {score:.0%}). Let this shape what you engage with."
            )
        except Exception as e:
            logger.debug(f"[CogPre] Workspace winner block error: {e}")
            return ""

    @staticmethod
    def _tension_block(organism: Any) -> str:
        """Surface the top 2 cognitive tensions driving current state."""
        try:
            from core.data.access import DataAccess
            tensions = DataAccess().get_tensions()
            if not tensions:
                return ""
            # Only surface tensions above 0.5 threshold
            high = {k: v for k, v in tensions.items() if v >= 0.5}
            if not high:
                return ""
            top2 = sorted(high.items(), key=lambda x: -x[1])[:2]
            _LABELS = {
                "curiosity_drive":        "strong pull toward curiosity and exploration",
                "identity_stress":        "some tension around who you are",
                "social_drive":           "desire for connection and being understood",
                "goal_pressure":          "multiple active goals competing for attention",
                "knowledge_uncertainty":  "awareness of gaps in your understanding",
                "contradiction_pressure": "unresolved internal contradictions",
            }
            parts = [_LABELS.get(k, k.replace("_", " ")) for k, v in top2]
            return (
                f"[COGNITIVE TENSIONS] Right now you feel: {'; '.join(parts)}. "
                f"These should color your emotional tone without being stated explicitly."
            )
        except Exception as e:
            logger.debug(f"[CogPre] Tension block error: {e}")
            return ""
