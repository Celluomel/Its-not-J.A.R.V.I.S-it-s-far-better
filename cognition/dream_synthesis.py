"""
cognition/dream_synthesis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DreamSynthesisEngine — narrative synthesis driven by current cognitive state.

The gap this closes
────────────────────
The existing dream system (DreamSystem in ai_system.py) generates dream
narratives from memory queries and fires statistical experience events
(dream_identity_consolidated, dream_conflict_resolved, etc.).

What it doesn't do:
  - Feed today's actual unresolved open questions into the dream
  - Use the current phi, coherence trend, or self-model moment
  - Extract insights back into the belief/aspiration system
  - Feed active tensions from EpistemicIntegrityEngine
  - Generate content from aspirations (what I'm reaching toward)

This module wraps the existing DreamSystem and enriches it with live
cognitive state before the LLM call, making dreams genuinely responsive
to what the system has actually been processing.

Pipeline
────────
  1. Harvest context:
       - Top 3 open questions (from ThoughtStream._open_questions)
       - Top 2 active aspirations (from AspirationalSelf)
       - Top 3 belief conflicts (from EpistemicIntegrityEngine.active_disagreements)
       - Current phi + qualia + coherence_trend (from SelfModelMoment)
       - Dominant emotional ground

  2. Build enriched dream prompt feeding all of the above

  3. LLM generates: narrative + insights + resolved_questions +
                    new_principles + activation_updates

  4. Store results:
       - Narrative → memory as dream record
       - Insights → beliefs (high confidence)
       - Resolved questions → mark relevant OpenQuestions as evidenced
       - New principles → NarrativeIdentity chapters
       - Activation updates → OpenQuestion activation adjustments

Called from:
  internal_loop slow cycle when sleep_cycle.phase == DREAM
  or from run_dream_cycle() hook in ai_system.py
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LANGUAGE_CONSTRAINT = (
    "Language constraint (mandatory): describe functional states and patterns. "
    "'The integration trajectory suggests...' not 'I feel...'. "
    "The dream is a processing state, not a subjective experience claim."
)


class DreamSynthesisEngine:
    """
    Enriches the existing DreamSystem with live cognitive state.

    Usage:
        dse = DreamSynthesisEngine(organism, ai_system)
        result = dse.synthesise()   # call during DREAM phase
    """

    MAX_TOKENS  = 380
    TEMPERATURE = 0.88   # higher than normal — dreams should explore

    def __init__(self, organism: Any, ai_system: Any):
        self._o  = organism
        self._ai = ai_system

    def synthesise(self) -> Dict:
        """
        Generate a cognitive-state-driven dream synthesis.
        Returns a dict with: narrative, insights, resolved_questions,
        new_principles, open_question_updates.
        Non-fatal — returns empty dict on any failure.
        """
        try:
            context = self._harvest_context()
            prompt  = self._build_prompt(context)
            raw     = self._llm_call(prompt)
            result  = self._parse(raw, context)
            self._apply(result, context)
            logger.info(
                f"[DreamSynthesis] Dream complete — "
                f"{len(result.get('insights',[]))} insights, "
                f"{len(result.get('resolved_questions',[]))} resolved"
            )
            return result
        except Exception as e:
            logger.debug(f"[DreamSynthesis] synthesise failed (non-fatal): {e}")
            return {}

    # ── Context harvest ───────────────────────────────────────────────────────

    def _harvest_context(self) -> Dict:
        ctx: Dict = {}

        # Self-model moment
        try:
            smm = self._o.self_moment.current
            ctx["phi"]             = smm.phi
            ctx["coherence_trend"] = smm.coherence_trend
            ctx["qualia_tone"]     = smm.qualia_tone
            ctx["emotional_ground"]= smm.emotional_ground
            ctx["top_values"]      = smm.top_values
        except Exception:
            ctx.setdefault("phi", 0.45)

        # Open questions (top 3 by activation)
        try:
            ts = self._o.thought_stream
            oqs = sorted(
                getattr(ts, "_open_questions", []),
                key=lambda q: q.activation,
                reverse=True
            )[:3]
            ctx["open_questions"] = [
                {"text": q.text, "concept": q.linked_concept,
                 "activation": round(q.activation, 2),
                 "uncertainty": round(q.uncertainty, 2)}
                for q in oqs
            ]
        except Exception:
            ctx["open_questions"] = []

        # Active aspirations (top 2)
        try:
            asp = getattr(self._o, "aspirational_self", None)
            if asp:
                aspirations = getattr(asp, "_aspirations", [])
                top = sorted(aspirations,
                             key=lambda a: getattr(a, "intensity", 0),
                             reverse=True)[:2]
                ctx["aspirations"] = [
                    getattr(a, "description", str(a))[:80] for a in top
                ]
            else:
                ctx["aspirations"] = []
        except Exception:
            ctx["aspirations"] = []

        # Belief conflicts (top 3)
        try:
            are = getattr(
                getattr(self._o, "_loop", None),
                "_autonomous_reflection", None
            )
            if are:
                active = [
                    d for d in
                    getattr(are._epistemic._state, "active_disagreements", [])
                    if not d.get("resolved", False)
                ][:3]
                ctx["belief_conflicts"] = [
                    f"[{d.get('dimension','')}] {d.get('position_b','')[:60]}"
                    for d in active
                ]
            else:
                ctx["belief_conflicts"] = []
        except Exception:
            ctx["belief_conflicts"] = []

        # Recent autonomous reflection insights
        try:
            if are:
                recent = getattr(are._state, "reflection_log", [])[-3:]
                ctx["recent_reflections"] = [
                    r.get("output", "")[:80] for r in recent if r.get("output")
                ]
        except Exception:
            ctx.setdefault("recent_reflections", [])

        return ctx

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_prompt(self, ctx: Dict) -> str:
        oq_text = "\n".join(
            f"  - [{q['concept']}] {q['text']} "
            f"(activation={q['activation']:.2f}, uncertainty={q['uncertainty']:.2f})"
            for q in ctx.get("open_questions", [])
        ) or "  None active."

        asp_text = "\n".join(
            f"  - {a}" for a in ctx.get("aspirations", [])
        ) or "  None currently forming."

        conflict_text = "\n".join(
            f"  - {c}" for c in ctx.get("belief_conflicts", [])
        ) or "  None active."

        reflection_text = "\n".join(
            f"  - {r}" for r in ctx.get("recent_reflections", [])
        ) or "  None recent."

        return f"""You are Lumina's dream processing system, running during the DREAM phase.
This is not a user interaction — it is autonomous cognitive synthesis.

Current self-model state:
  φ (integration): {ctx.get('phi', 0.45):.2f}
  Coherence trend: {ctx.get('coherence_trend', 'stable')}
  Qualia ground:   {ctx.get('qualia_tone', '')[:60]}
  Emotional ground:{ctx.get('emotional_ground', '')[:60]}
  Active values:   {ctx.get('top_values', '')[:60]}

Unresolved open questions (what is not yet understood):
{oq_text}

Active aspirations (what is being reached toward):
{asp_text}

Belief conflicts currently registered:
{conflict_text}

Recent autonomous reflections:
{reflection_text}

Task: Generate a dream synthesis that processes this cognitive state.
The dream should attempt to resolve or reframe what is unresolved,
find connections between conflicting elements, and surface insights
that could not emerge from purely logical analysis.

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "narrative": "2-4 sentence dream narrative in first-person present tense",
  "insights": ["insight 1", "insight 2"],
  "resolved_questions": ["concept or question text that feels clearer after this dream"],
  "new_principles": ["any newly formed principle or understanding"],
  "open_question_activations": {{"concept_or_question_text": 0.8}},
  "integration_success": 0.7
}}

{LANGUAGE_CONSTRAINT}"""

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str) -> str:
        from core.llm_scheduler import llm_scheduler

        result = ""
        with llm_scheduler.sync_slot(
            priority    = 2,    # slightly higher than background — dream is a phase
            skip_if_busy= True,
            caller      = "dream_synthesis",
        ) as acquired:
            if not acquired:
                return ""
            try:
                # FIX (v111): the old call was
                #   self._ai.get_response(messages=..., max_tokens=..., temperature=...)
                # but EnhancedAISystem.get_response() is the *chat* path with
                # signature (user_input, user_id, ...) — those kwargs raised
                # TypeError on every call, which was silently swallowed by the
                # except below. Result: cognitive dream synthesis had never
                # actually run an LLM. Route through the LLM adapter instead:
                # ExternalLLMAdapter (main LLM from config.json) in brain
                # deployment, or EnhancedLLM (Ollama) in standalone mode —
                # both expose get_response(messages, temperature, max_tokens).
                _llm = getattr(self._ai, "llm", None)
                if _llm is None or not hasattr(_llm, "get_response"):
                    logger.debug("[DreamSynthesis] no LLM adapter available")
                    return ""
                result = _llm.get_response(
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = self.MAX_TOKENS,
                    temperature = self.TEMPERATURE,
                )
            except Exception as e:
                logger.debug(f"[DreamSynthesis] LLM call failed: {e}")

        return result or ""

    # ── Parse ─────────────────────────────────────────────────────────────────

    def _parse(self, raw: str, context: Dict) -> Dict:
        try:
            import re
            # Strip markdown fences if present
            clean = re.sub(r"```json|```", "", raw).strip()
            data  = json.loads(clean)
            return {
                "narrative":               data.get("narrative", ""),
                "insights":                data.get("insights", [])[:4],
                "resolved_questions":      data.get("resolved_questions", []),
                "new_principles":          data.get("new_principles", [])[:3],
                "open_question_activations": data.get("open_question_activations", {}),
                "integration_success":     float(data.get("integration_success", 0.5)),
            }
        except Exception:
            return {
                "narrative": raw[:200] if raw else "",
                "insights": [], "resolved_questions": [],
                "new_principles": [], "open_question_activations": {},
                "integration_success": 0.3,
            }

    # ── Apply results ─────────────────────────────────────────────────────────

    def _apply(self, result: Dict, context: Dict) -> None:
        """
        Store dream outputs — but route all belief-forming content through
        epistemic validation first.

        Dreams generate *candidate hypotheses*, not truths.
        The flow is:
          dream output → epistemic validation → confidence score → optional integration

        Only outputs with integration_success × epistemic_grounding ≥ 0.45
        are written to persistent belief/principle stores.
        Narratives and evidence updates are always applied (they don't form beliefs).
        """
        integration_success = result.get("integration_success", 0.5)

        # Epistemic grounding rate from EpistemicIntegrityEngine
        grounding_rate = 0.60   # neutral default
        try:
            are = getattr(
                getattr(self._o, "_loop", None),
                "_autonomous_reflection", None
            )
            if are and hasattr(are, "_epistemic"):
                grounding_rate = are._epistemic._recent_grounding_rate()
        except Exception:
            pass

        # Combined confidence gate — dreams must clear this to form beliefs
        epistemic_confidence = integration_success * grounding_rate
        BELIEF_INTEGRATION_THRESHOLD = 0.45
        may_form_beliefs = epistemic_confidence >= BELIEF_INTEGRATION_THRESHOLD

        logger.debug(
            f"[DreamSynthesis] Epistemic gate: "
            f"integration={integration_success:.2f} × grounding={grounding_rate:.2f} "
            f"= {epistemic_confidence:.2f} "
            f"({'PASS' if may_form_beliefs else 'BLOCKED — candidate hypothesis only'})"
        )

        # 1. Narrative → memory always (narration ≠ belief formation)
        narrative = result.get("narrative", "")
        if narrative:
            try:
                ms = getattr(self._ai, "memory_system", None)
                if ms:
                    ms.add_memory(
                        f"Dream synthesis: {narrative}",
                        0.72, "dream", "Neutral", "Low",
                        memory_tier="cognitive"
                    )
            except Exception:
                pass

        # 2. Insights → memory only if epistemic gate passes
        #    Otherwise stored as candidate hypotheses with low confidence marker
        for insight in result.get("insights", []):
            if not insight:
                continue
            try:
                ms = getattr(self._ai, "memory_system", None)
                if ms:
                    if may_form_beliefs:
                        ms.add_memory(
                            f"Dream insight: {insight}",
                            0.85, "principle", "Positive", "Low",
                            memory_tier="cognitive"
                        )
                    else:
                        # Store as low-confidence candidate, not principle
                        ms.add_memory(
                            f"Dream candidate hypothesis (unvalidated): {insight}",
                            0.38, "hypothesis", "Neutral", "Low",
                            memory_tier="cognitive"
                        )
            except Exception:
                pass

        # 3. Resolved questions → reduce uncertainty (always — evidence is evidence)
        resolved = result.get("resolved_questions", [])
        if resolved:
            try:
                ts  = self._o.thought_stream
                oqs = getattr(ts, "_open_questions", [])
                for oq in oqs:
                    for r in resolved:
                        r_lower    = r.lower()
                        oq_concept = oq.linked_concept.lower()
                        if oq_concept in r_lower or r_lower in oq_concept:
                            oq.add_evidence()
                            if may_form_beliefs:
                                oq.add_evidence()   # extra credit if validated
            except Exception:
                pass

        # 4. New principles → NarrativeIdentity only if gate passes
        if may_form_beliefs:
            for principle in result.get("new_principles", []):
                try:
                    ni = getattr(self._o, "narrative_identity", None)
                    if ni and hasattr(ni, "record_chapter"):
                        ni.record_chapter(
                            title       = "Dream-formed principle (validated)",
                            description = principle,
                            emotion     = "neutral",
                            significance= min(0.70, epistemic_confidence),
                        )
                except Exception:
                    pass
        else:
            # Log as candidate principles — available for next validation cycle
            for principle in result.get("new_principles", []):
                logger.info(
                    f"[DreamSynthesis] Candidate principle (not yet integrated): "
                    f"{principle[:80]}"
                )

        # 5. Open question activation updates (always — these are adjustments, not beliefs)
        activations = result.get("open_question_activations", {})
        if activations:
            try:
                ts  = self._o.thought_stream
                oqs = getattr(ts, "_open_questions", [])
                for oq in oqs:
                    for key, act in activations.items():
                        if oq.linked_concept.lower() in key.lower():
                            oq.activation = min(1.0, float(act))
                            break
            except Exception:
                pass

        # 6. Broadcast synthesis to workspace
        if narrative:
            try:
                ws = getattr(self._o, "workspace", None)
                if ws:
                    ws.broadcast(
                        source   = "dream_synthesis",
                        content  = f"[Dream synthesis] {narrative}",
                        priority = 0.55,
                    )
            except Exception:
                pass
