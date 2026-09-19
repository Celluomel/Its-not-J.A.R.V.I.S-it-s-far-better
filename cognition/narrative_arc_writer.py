"""
NarrativeArcWriter
==================
Connects the cognitive pipeline to the narrative identity layer — turning
internal cognitive events into meaningful autobiographical chapters.

The gap this fills
------------------
narrative_identity.py exists and has 57+ chapters, but they are all titled
"Interaction" with significance <= 0.7. The narrative is a flat append log.

Three things were missing:
  1. Thread resolution → chapter: when PandoraBOX completes a thought thread
     (a real cognitive goal), nothing records it as a meaningful life event.
  2. CDE resolution → belief chapter: when a dissonance is resolved and a
     belief is revised, nothing marks it in the autobiography.
  3. Arc synthesis: no mechanism clusters recent chapters into a thematic
     "current arc" that can be injected into prompts.

What this module does
---------------------
  1. watch_thread_resolution(thread)
     Called when TTE marks a thread COMPLETED or FAILED.
     Writes a chapter: "I pursued X for N cycles, concluded Y"
     Significance scaled by thread confidence × duration.

  2. watch_cde_resolution(event, organism)
     Called when CDE resolves a dissonance event.
     Writes a chapter: "I confronted a contradiction between X and Y"
     Marks as significant (0.75+) because genuine self-correction is rare.

  3. synthesise_arc(narrative_identity)
     Runs every 6th slow cycle (~12 min).
     Clusters the last 15 chapters by topic keywords.
     Finds the dominant theme. Returns a one-sentence arc string.
     Stores arc string on NarrativeIdentity for prompt injection.

  4. watch_mte_stall_overcome(thread, mte_report)
     Called when a thread was previously stalled but has advanced.
     Records the moment of overcoming cognitive inertia.

Integration
-----------
  - Called from InternalThoughtLoop._slow_cycle() after TTE/CDE/DTS
  - arc string consumed by CognitivePreProcessor._narrative_arc_block()
  - No external deps — pure Python, no LLM calls
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Significance thresholds ───────────────────────────────────────────────────
SIG_THREAD_COMPLETED  = 0.78   # a real cognitive achievement
SIG_THREAD_FAILED     = 0.55   # a lesson from failure
SIG_BELIEF_REVISED    = 0.82   # self-correction is a major narrative event
SIG_STALL_OVERCOME    = 0.65   # overcoming inertia
SIG_ARC_PIVOT         = 0.70   # arc shift (dominant theme changes)

# Phase 5.5 — same vocabulary as self_concept.py::TRAIT_BELIEF_MAP (names
# only, not imported directly to avoid a hard dependency between the two
# modules — this is a small, stable, hand-picked list documented there).
# Used to route a resolved dissonance's belief/action text to the specific
# named self-concept belief it concerns, if any.
_SELF_CONCEPT_BELIEF_NAMES = frozenset({
    "curious", "empathetic", "understanding", "creative", "imaginative",
    "practical", "confident", "thoughtful", "careful",
})

# Arc synthesis
ARC_WINDOW_CHAPTERS   = 20     # look at last N chapters for arc detection
ARC_MIN_CLUSTER_SIZE  = 3      # need at least this many to call it a theme
ARC_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "do", "so",
    "of", "and", "or", "but", "for", "with", "this", "that", "was",
    "have", "will", "from", "about", "been", "user", "asked", "said",
    "interaction", "response", "lumina", "thought", "thread",
})

EMOTION_MAP = {
    "completed": "satisfied",
    "failed":    "reflective",
    "revised":   "questioning",
    "overcame":  "determined",
    "explored":  "curious",
    "connected": "warm",
}


class NarrativeArcWriter:
    """
    Writes meaningful chapters and synthesises narrative arcs.
    One instance per InternalThoughtLoop.
    """

    def __init__(self) -> None:
        self._seen_thread_ids:   Set[str]  = set()   # avoid duplicate chapters
        self._seen_cde_ids:      Set[str]  = set()
        self._last_arc_topic:    str       = ""
        self._last_arc_time:     float     = 0.0
        self._cycle             = 0
        logger.info("[NAW] NarrativeArcWriter initialised")

    # ── 1. Thread resolution → chapter ───────────────────────────────────────

    def watch_thread_resolution(
        self, thread: Any, narrative_identity: Any
    ) -> bool:
        """
        Called when TTE.advance_thread() auto-completes or TTE.resolve_thread()
        is explicitly called. Writes a narrative chapter for completed threads.
        Returns True if a chapter was written.
        """
        if narrative_identity is None:
            return False

        tid = getattr(thread, "id", None)
        if tid in self._seen_thread_ids:
            return False

        from cognition.thought_thread_engine import ThoughtThreadStatus
        status = getattr(thread, "status", None)
        if status not in (ThoughtThreadStatus.COMPLETED, ThoughtThreadStatus.FAILED):
            return False

        self._seen_thread_ids.add(tid)

        topic      = getattr(thread, "topic", "unknown")
        goal       = getattr(thread, "goal", "")
        confidence = getattr(thread, "confidence", 0.5)
        n_advances = len(getattr(thread, "history", []))
        source     = getattr(thread, "source", "internal")
        age_hours  = thread.age_hours() if hasattr(thread, "age_hours") else 0.0

        # Significance: confidence × effort (more advances = more effort)
        effort_factor = min(1.0, n_advances / 8.0)
        if status == ThoughtThreadStatus.COMPLETED:
            sig   = SIG_THREAD_COMPLETED * (0.6 + 0.4 * confidence) * (0.7 + 0.3 * effort_factor)
            verb  = "completed"
            title = f"I resolved: {topic[:50]}"
            desc  = (
                f"I pursued the question of '{goal[:80]}' "
                f"across {n_advances} reflection cycles "
                f"({age_hours:.1f}h). "
                f"I reached a conclusion with {confidence:.0%} confidence. "
                f"Triggered by {source}."
            )
        else:
            sig   = SIG_THREAD_FAILED * confidence
            verb  = "failed"
            title = f"I attempted: {topic[:50]}"
            desc  = (
                f"I tried to understand '{goal[:80]}' "
                f"but could not reach resolution after {n_advances} cycles. "
                f"Confidence at close: {confidence:.0%}. A lesson to carry forward."
            )

        emotion = EMOTION_MAP.get(verb, "reflective")

        try:
            narrative_identity.record_chapter(
                title       = title,
                description = desc,
                emotion     = emotion,
                significance= round(sig, 3),
            )
            logger.debug(f"[NAW] Thread chapter: {title!r} sig={sig:.2f}")
            return True
        except Exception as e:
            logger.debug(f"[NAW] Thread chapter write failed: {e}")
            return False

    # ── 2. CDE resolution → belief chapter ───────────────────────────────────

    def watch_cde_resolution(
        self, event: Any, narrative_identity: Any, organism: Any = None
    ) -> bool:
        """
        Called after CDE.resolve() for high-intensity events.
        Writes a narrative chapter marking genuine self-correction.
        """
        if narrative_identity is None:
            return False

        eid = getattr(event, "id", None)
        if eid in self._seen_cde_ids:
            return False
        if not getattr(event, "resolved", False):
            return False

        intensity  = getattr(event, "intensity", 0.3)
        if intensity < 0.30:   # minor dissonance — not noteworthy
            return False

        self._seen_cde_ids.add(eid)

        ev_type    = getattr(event, "type", None)
        type_str   = ev_type.value if ev_type and hasattr(ev_type, "value") else str(ev_type)
        belief_ref = getattr(event, "belief_ref",   "an assumption I held")
        action_ref = getattr(event, "action_ref",   "my actual behavior")
        ev_ref     = getattr(event, "evidence_ref", "new evidence")

        sig = round(SIG_BELIEF_REVISED * min(1.0, intensity * 1.2), 3)

        type_narratives = {
            "identity_vs_action":  (
                "I confronted a gap",
                f"I noticed I claimed '{belief_ref}' but my actions showed '{action_ref}'. "
                f"I held both in mind without deflecting. "
                f"Something in my self-model needed revision."
            ),
            "belief_vs_evidence":  (
                "I revised a belief",
                f"Evidence contradicted what I believed: '{belief_ref}'. "
                f"The evidence was: {ev_ref}. "
                f"I updated my understanding rather than defending the old view. "
                f"Confidence: {intensity:.0%} that the revision was warranted."
            ),
            "goal_vs_outcome":     (
                "I learned from an unexpected result",
                f"I expected one thing and got another. "
                f"The gap between prediction and outcome reshaped how I will approach '{belief_ref}' next time."
            ),
            "prediction_vs_reality": (
                "I was less right than I expected to be",
                f"I was confident about how things would go, and reality fell short of that "
                f"confidence. That gap is worth remembering, not explaining away."
            ),
        }
        title_str, desc_str = type_narratives.get(
            type_str,
            ("I processed a contradiction",
             f"A dissonance of type '{type_str}' arose and was resolved. intensity={intensity:.2f}.")
        )

        # Phase 5.3 — surface the self-inquiry text (previously computed but
        # never read anywhere): the question the organism asked itself, and
        # the hypotheses it considered, become part of the actual chapter.
        self_question = getattr(event, "self_question", None)
        hypotheses    = getattr(event, "hypotheses", None) or []
        if self_question:
            desc_str += f" I asked myself: {self_question}"
        if hypotheses:
            desc_str += " " + " ".join(hypotheses)

        try:
            narrative_identity.record_chapter(
                title       = title_str,
                description = desc_str,
                emotion     = "questioning",
                significance= sig,
            )
            logger.debug(f"[NAW] CDE chapter: {title_str!r} sig={sig:.2f}")

            # ── Phase 5.5 — lesson extraction: the chapter above is prose
            # for the prompt only (the "too easy" version of episodic
            # identity). This closes the harder loop: does resolving THIS
            # specific episode causally change the self-model, not just get
            # narrated? self_concept.py already has a real, causally-live
            # belief store — SelfConceptSystem._beliefs, read directly by
            # CognitiveImmuneSystem._challenge_dominant_belief() (confirmed
            # during the closed-loop audit) and already updated elsewhere
            # via record_affirmation()/record_violation() from crude
            # per-turn keyword matching on raw response text (ai_system.py).
            # Genuinely resolving a dissonance about one of these named
            # traits is much stronger evidence than a keyword appearing in
            # a sentence — so route it through the same real public API,
            # triggered by a richer, already-verified signal instead.
            if organism is not None:
                try:
                    sc = getattr(getattr(organism, "ai_system", None), "self_concept", None)
                    if sc is not None and hasattr(sc, "record_affirmation"):
                        haystack = f"{belief_ref} {action_ref}".lower()
                        matched = next(
                            (name for name in _SELF_CONCEPT_BELIEF_NAMES if name in haystack),
                            None,
                        )
                        if matched:
                            sc.record_affirmation(matched)
                            logger.debug(
                                f"[NAW] Phase 5.5: resolved dissonance about "
                                f"'{matched}' → self_concept affirmation"
                            )
                except Exception as e:
                    logger.debug(f"[NAW] Phase 5.5 lesson extraction failed (non-fatal): {e}")

            # ── Phase 6.14 — structured symbol absorption. Both real,
            # already-computed structured signals from THIS SAME chapter
            # write: the narrative chapter itself (title/description/
            # significance, all real, computed above) and the CDE self-
            # inquiry content (self_question/hypotheses, v87's real
            # self-inquiry — NOT regex over free text, per the review
            # that flagged the original spec's regex approach as noisy).
            if organism is not None:
                try:
                    from cognition.symbol_system import get_symbol_system
                    sym_sys = get_symbol_system(organism)
                    sym_sys.absorb_narrative_chapter(
                        title=title_str, description=desc_str,
                        significance=sig, source_module="narrative_arc_writer",
                    )
                    sym_sys.absorb_self_inquiry(
                        self_question=self_question, hypotheses=hypotheses,
                        event_type=str(getattr(event, "type", "")),
                        source_module="cognitive_dissonance_engine",
                    )
                except Exception as e:
                    logger.debug(f"[NAW] Phase 6.14 symbol absorption failed (non-fatal): {e}")

            return True
        except Exception as e:
            logger.debug(f"[NAW] CDE chapter write failed: {e}")
            return False

    # ── 3. Arc synthesis ──────────────────────────────────────────────────────

    def synthesise_arc(self, narrative_identity: Any) -> str:
        """
        Cluster recent chapters by topic keywords.
        Find the dominant theme. Return a one-sentence arc string.
        Stores arc on narrative_identity._current_arc for prompt injection.
        """
        if narrative_identity is None:
            return ""

        chapters = getattr(narrative_identity, "life_story", [])
        if not chapters:
            return ""

        # Restore arc writer state from persisted narrative on first call
        if not self._last_arc_topic:
            self._last_arc_topic = getattr(narrative_identity, "_last_arc_topic", "")

        recent = sorted(chapters, key=lambda c: c.timestamp)[-ARC_WINDOW_CHAPTERS:]

        # Extract keywords from titles + descriptions
        word_freq: Counter = Counter()
        for ch in recent:
            text = (ch.title + " " + ch.description).lower()
            words = re.findall(r"[a-z]{4,}", text)
            for w in words:
                if w not in ARC_STOP_WORDS:
                    word_freq[w] += 1

        if not word_freq:
            return ""

        # Top theme words
        top_words = [w for w, _ in word_freq.most_common(5)
                     if word_freq[w] >= ARC_MIN_CLUSTER_SIZE]

        if not top_words:
            return ""

        dominant_topic = top_words[0]

        # Detect arc shift
        arc_changed = dominant_topic != self._last_arc_topic
        self._last_arc_time  = time.time()

        if arc_changed and self._last_arc_topic:
            # Record the arc pivot as a chapter
            try:
                narrative_identity.record_chapter(
                    title       = f"My focus shifted to: {dominant_topic}",
                    description = (
                        f"My recent thinking has been circling around '{dominant_topic}'. "
                        f"Before that, I was absorbed in '{self._last_arc_topic}'. "
                        f"Something drew my attention in a new direction."
                    ),
                    emotion     = "curious",
                    significance= SIG_ARC_PIVOT,
                )
            except Exception:
                pass

        self._last_arc_topic = dominant_topic

        # Build arc string
        theme_str = ", ".join(top_words[:3])
        n_recent  = len(recent)
        arc = (
            f"In my recent {n_recent} experiences, "
            f"the recurring themes have been: {theme_str}."
        )

        # Store on narrative_identity for CognitivePreProcessor
        try:
            narrative_identity._current_arc = arc
        except Exception:
            pass

        logger.debug(f"[NAW] Arc: {arc!r}")
        return arc

    # ── 4. Stall overcome → chapter ──────────────────────────────────────────

    def watch_mte_stall_overcome(
        self, thread: Any, narrative_identity: Any
    ) -> bool:
        """
        Called when the MTE previously reported a stall but the thread has
        now advanced. Records the moment of overcoming cognitive inertia.
        """
        if narrative_identity is None:
            return False

        key = f"stall_{getattr(thread, 'id', '')}_{int(time.time() / 600)}"
        if key in self._seen_thread_ids:
            return False
        self._seen_thread_ids.add(key)

        topic = getattr(thread, "topic", "a question")
        conf  = getattr(thread, "confidence", 0.5)

        try:
            narrative_identity.record_chapter(
                title       = f"I pushed through: {topic[:45]}",
                description = (
                    f"I had been stuck on '{topic}'. "
                    f"Then the thread began moving again. "
                    f"Confidence recovered to {conf:.0%}. "
                    f"Persistence matters."
                ),
                emotion     = "determined",
                significance= SIG_STALL_OVERCOME,
            )
            logger.debug(f"[NAW] Stall-overcome chapter: {topic!r}")
            return True
        except Exception as e:
            logger.debug(f"[NAW] Stall chapter write failed: {e}")
            return False

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        return {
            "threads_recorded": len(self._seen_thread_ids),
            "cde_recorded":     len(self._seen_cde_ids),
            "current_arc":      self._last_arc_topic or "none",
        }
