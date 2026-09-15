"""
MetaThreadEvaluator
===================
"Thoughts about thoughts" — the conscious oversight layer for TTE threads.

The gap this fills
------------------
meta_cognition.py evaluates LANGUAGE OUTPUT (clarity, hedging, length).
meta_reflection.py evaluates PERSONALITY EVOLUTION over time.

Neither evaluates the COGNITIVE PROCESS that produced the output:
  - Is the dominant thread actually advancing toward its goal?
  - Is confidence calibrated against real progress?
  - Is the thread drifting from its stated goal?
  - Is the system spending cycles on a stalled thread when a better one exists?
  - Are secondary threads being systematically ignored?

MetaThreadEvaluator runs each slow cycle (~2 min), reads the DTS
SelectionResult and active TTE threads, and produces:

  1. A quality assessment per thread  (goal_alignment, progress_rate,
     confidence_calibration, drift_score)
  2. Observations logged to MetaCognition (visible in dashboard + reflection)
  3. A "meta-directive" string injected into CognitivePreProcessor's preamble
     — telling the LLM how well its current thinking is working
  4. Direct thread mutations: threads that are stalling have their confidence
     penalised; threads that are drifting get a goal-reminder injected

This creates a genuine feedback loop:
  TTE generates threads → DTS selects dominant → MetaThreadEvaluator assesses
  → observations feed MetaCognition → CognitivePreProcessor injects directive
  → LLM output is shaped by awareness of its own reasoning quality
  → post_response_eval feeds back into MetaCognition
  → cycle repeats with calibration
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
STALL_IDLE_HOURS          = 0.5    # thread dominant but not advancing → stall
DRIFT_CONFIDENCE_DROP     = 0.15   # confidence falling fast → goal drift
HIGH_DISSONANCE_THRESHOLD = 0.45   # thread dissonance → flag for meta-reflection
MIN_ADVANCES_FOR_CALIBRATION = 3   # need at least this many history entries to judge
OVERCLAIM_THRESHOLD       = 0.85   # very high confidence with few advances → overclaim


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ThreadAssessment:
    """Quality assessment for one thought thread."""
    thread_id:               str
    topic:                   str
    goal_alignment:          float   # 0–1: how well current actions match stated goal
    progress_rate:           float   # 0–1: how fast confidence is growing per advance
    confidence_calibration:  float   # 0–1: 1=well-calibrated, 0=over/under confident
    drift_score:             float   # 0–1: 0=on track, 1=completely drifted
    is_stalled:              bool
    is_overclaiming:         bool
    quality_score:           float   # composite [0–1]
    observation:             str     # human-readable one-liner
    directive:               str     # instruction for the LLM ("pursue X", "reconsider Y")


@dataclass
class MetaThreadReport:
    """Full output of one MetaThreadEvaluator cycle."""
    timestamp:          float         = field(default_factory=time.time)
    dominant_assessment: Optional[ThreadAssessment] = None
    secondary_assessments: List[ThreadAssessment]   = field(default_factory=list)
    ignored_threads:    List[str]                   = field(default_factory=list)
    system_observation: str           = ""
    meta_directive:     str           = ""   # injected into system prompt preamble
    overall_quality:    float         = 0.5


# ── Evaluator ─────────────────────────────────────────────────────────────────

class MetaThreadEvaluator:
    """
    Evaluates thought thread quality and feeds findings back into
    MetaCognition and CognitivePreProcessor.

    Instantiated once in InternalThoughtLoop.__init__() alongside TTE/CDE/DTS.

    Usage (slow cycle):
        report = mte.evaluate(
            selection_result = dts_result,
            active_threads   = tte.active_threads(),
            meta_cognition   = organism.meta_cognition,
        )
        organism._meta_thread_report = report  # consumed by CognitivePreProcessor
    """

    def __init__(self) -> None:
        self._history: List[MetaThreadReport] = []
        self._cycle   = 0
        logger.info("[MTE] MetaThreadEvaluator initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        selection_result: Any,          # DTS SelectionResult
        active_threads:   List[Any],    # ThoughtThread list from TTE
        meta_cognition:   Any = None,   # MetaCognition instance (optional)
    ) -> MetaThreadReport:
        """
        Run one evaluation cycle. Returns a MetaThreadReport.
        """
        self._cycle += 1
        report = MetaThreadReport()

        dominant = getattr(selection_result, "dominant", None)
        secondary = getattr(selection_result, "secondary", [])

        # ── Assess dominant thread ────────────────────────────────────────────
        if dominant is not None:
            da = self._assess_thread(dominant, is_dominant=True)
            report.dominant_assessment = da

            # Log significant findings to MetaCognition
            if meta_cognition is not None and da.observation:
                severity = "alert" if da.is_stalled or da.is_overclaiming else \
                           "flag"  if da.quality_score < 0.40 else "note"
                try:
                    meta_cognition._log_observation(
                        observation   = f"[Thread:{dominant.topic[:40]}] {da.observation}",
                        category      = "consistency",
                        severity      = severity,
                        trigger       = "meta_thread_evaluator",
                    )
                except Exception:
                    pass

            # Directly penalise stalled threads
            if da.is_stalled:
                dominant.confidence = max(0.10, dominant.confidence - 0.08)
                logger.debug(f"[MTE] Stall penalty: {dominant.topic!r}")

            # Inject goal reminder into stalled thread's planned_actions
            if da.drift_score > 0.6 and dominant.planned_actions is not None:
                reminder = f"return to goal: {dominant.goal[:60]}"
                if reminder not in dominant.planned_actions:
                    dominant.planned_actions.insert(0, reminder)

        # ── Assess secondary threads ──────────────────────────────────────────
        for t in secondary[:3]:
            sa = self._assess_thread(t, is_dominant=False)
            report.secondary_assessments.append(sa)

        # ── Identify ignored threads ──────────────────────────────────────────
        dom_ids  = {dominant.id} if dominant else set()
        sec_ids  = {t.id for t in secondary}
        all_ids  = {t.id for t in active_threads}
        report.ignored_threads = list(all_ids - dom_ids - sec_ids)

        # ── System-level observation ──────────────────────────────────────────
        report.system_observation = self._system_observation(
            dominant, active_threads, report.ignored_threads
        )

        # ── Meta-directive for system prompt ──────────────────────────────────
        report.meta_directive = self._build_directive(
            report.dominant_assessment, report.system_observation
        )

        # ── Overall quality ───────────────────────────────────────────────────
        scores = [report.dominant_assessment.quality_score] if report.dominant_assessment else []
        scores += [a.quality_score for a in report.secondary_assessments]
        report.overall_quality = sum(scores) / max(1, len(scores))

        # Log to MetaCognition at system level
        if meta_cognition is not None and report.system_observation:
            try:
                severity = "alert" if report.overall_quality < 0.25 else \
                           "flag"  if report.overall_quality < 0.50 else "note"
                meta_cognition._log_observation(
                    observation = f"[MTE] {report.system_observation}",
                    category    = "clarity",
                    severity    = severity,
                    trigger     = "meta_thread_system",
                )
            except Exception:
                pass

        self._history.append(report)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        logger.debug(
            f"[MTE] cycle={self._cycle} quality={report.overall_quality:.2f} "
            f"dominant={dominant.topic[:30] if dominant else 'none'!r}"
        )
        return report

    def last_directive(self) -> str:
        """Return the most recent meta-directive, or empty string."""
        if self._history:
            return self._history[-1].meta_directive
        return ""

    def summary(self) -> Dict:
        recent = self._history[-5:] if self._history else []
        avg_q  = sum(r.overall_quality for r in recent) / max(1, len(recent))
        stalls = sum(
            1 for r in recent
            if r.dominant_assessment and r.dominant_assessment.is_stalled
        )
        return {
            "cycles":          self._cycle,
            "avg_quality":     round(avg_q, 3),
            "recent_stalls":   stalls,
            "last_directive":  self.last_directive()[:80],
        }

    # ── Thread assessment ─────────────────────────────────────────────────────

    def _assess_thread(self, thread: Any, is_dominant: bool) -> ThreadAssessment:
        """Produce a ThreadAssessment for one thread."""
        history   = getattr(thread, "history",    [])
        confidence= getattr(thread, "confidence", 0.5)
        goal      = getattr(thread, "goal",       "")
        topic     = getattr(thread, "topic",      "")
        actions   = getattr(thread, "planned_actions", [])
        dissonance= getattr(thread, "dissonance", 0.0)
        idle_h    = thread.idle_hours() if hasattr(thread, "idle_hours") else 0.0

        n_advances = len(history)

        # ── Goal alignment: do planned actions mention goal keywords? ─────────
        goal_words = set(goal.lower().split()[:6])  # first 6 words of goal
        action_text = " ".join(actions).lower()
        overlap = len(goal_words & set(action_text.split()))
        goal_alignment = min(1.0, overlap / max(1, len(goal_words)))

        # ── Progress rate: confidence delta per advance ───────────────────────
        if n_advances >= 2:
            conf_history = [
                h.get("confidence", confidence) for h in history[-5:]
                if isinstance(h, dict)
            ]
            if len(conf_history) >= 2:
                progress_rate = max(0.0, conf_history[-1] - conf_history[0]) / max(1, len(conf_history) - 1)
            else:
                progress_rate = 0.3
        else:
            progress_rate = 0.3  # not enough data

        # ── Confidence calibration ────────────────────────────────────────────
        is_overclaiming = confidence >= OVERCLAIM_THRESHOLD and n_advances < MIN_ADVANCES_FOR_CALIBRATION
        if is_overclaiming:
            confidence_calibration = 0.2   # very overconfident with little evidence
        elif n_advances == 0:
            confidence_calibration = 0.5   # neutral — no data yet
        else:
            # Good calibration: confidence tracks progress
            expected = min(0.9, 0.4 + n_advances * 0.08)
            calibration_error = abs(confidence - expected)
            confidence_calibration = max(0.0, 1.0 - calibration_error * 2)

        # ── Drift score ───────────────────────────────────────────────────────
        is_stalled  = is_dominant and idle_h > STALL_IDLE_HOURS
        drift_score = 0.0
        if n_advances >= 2:
            conf_series = [h.get("confidence", confidence) for h in history[-4:] if isinstance(h, dict)]
            if len(conf_series) >= 2 and (conf_series[0] - conf_series[-1]) > DRIFT_CONFIDENCE_DROP:
                drift_score = min(1.0, (conf_series[0] - conf_series[-1]) / 0.4)

        # ── Composite quality score ───────────────────────────────────────────
        quality = (
            goal_alignment          * 0.30
            + confidence_calibration * 0.25
            + progress_rate          * 0.25
            + (1.0 - drift_score)   * 0.10
            + (1.0 - dissonance)    * 0.10
        )
        if is_stalled:
            quality *= 0.6
        quality = max(0.0, min(1.0, quality))

        # ── Observation ───────────────────────────────────────────────────────
        obs_parts = []
        if is_stalled:
            obs_parts.append(f"stalled ({idle_h:.1f}h idle)")
        if is_overclaiming:
            obs_parts.append(f"overclaiming (conf={confidence:.2f}, only {n_advances} advances)")
        if drift_score > 0.5:
            obs_parts.append(f"drifting from goal (drift={drift_score:.2f})")
        if dissonance > HIGH_DISSONANCE_THRESHOLD:
            obs_parts.append(f"high dissonance ({dissonance:.2f})")
        if quality >= 0.70 and not obs_parts:
            obs_parts.append(f"progressing well (quality={quality:.2f})")
        observation = "; ".join(obs_parts) if obs_parts else f"quality={quality:.2f}"

        # ── Directive ─────────────────────────────────────────────────────────
        directive = ""
        if is_stalled:
            directive = f"Your '{topic}' thread is stalling. Actively advance it or defer it."
        elif is_overclaiming:
            directive = f"Your confidence on '{topic}' is higher than your evidence supports. Acknowledge uncertainty."
        elif drift_score > 0.5:
            directive = f"Your '{topic}' thread is drifting. Refocus on the goal: {goal[:60]}"
        elif dissonance > HIGH_DISSONANCE_THRESHOLD:
            directive = f"There is internal tension in your '{topic}' thread. Address the contradiction before continuing."

        return ThreadAssessment(
            thread_id              = getattr(thread, "id", "?"),
            topic                  = topic,
            goal_alignment         = round(goal_alignment,         3),
            progress_rate          = round(progress_rate,          3),
            confidence_calibration = round(confidence_calibration, 3),
            drift_score            = round(drift_score,            3),
            is_stalled             = is_stalled,
            is_overclaiming        = is_overclaiming,
            quality_score          = round(quality,                3),
            observation            = observation,
            directive              = directive,
        )

    # ── System-level analysis ─────────────────────────────────────────────────

    def _system_observation(
        self,
        dominant: Any,
        active_threads: List[Any],
        ignored_ids: List[str],
    ) -> str:
        n_active   = len(active_threads)
        n_ignored  = len(ignored_ids)

        if dominant is None:
            return "No dominant thread selected — cognitive focus absent."

        parts = []
        if n_ignored > 4:
            parts.append(
                f"{n_ignored} threads ignored this cycle — possible cognitive fragmentation."
            )
        if n_active > 8:
            parts.append(
                f"{n_active} active threads — workspace overloaded, consider pruning."
            )
        if not parts:
            return f"Dominant thread '{dominant.topic[:40]}' is the cognitive focus."
        return " ".join(parts)

    def _build_directive(
        self,
        da: Optional[ThreadAssessment],
        system_obs: str,
    ) -> str:
        """Build the string injected into CognitivePreProcessor preamble."""
        lines = []

        if da and da.directive:
            lines.append(f"[META-COGNITION] {da.directive}")

        if "fragmentation" in system_obs or "overloaded" in system_obs:
            lines.append(
                "[META-COGNITION] Multiple competing threads detected. "
                "Commit to the dominant topic rather than spreading attention."
            )

        if da and da.quality_score >= 0.70 and not da.is_stalled:
            lines.append(
                f"[META-COGNITION] Your reasoning on '{da.topic[:30]}' is well-calibrated. "
                "Continue building on it."
            )

        return "\n".join(lines)
