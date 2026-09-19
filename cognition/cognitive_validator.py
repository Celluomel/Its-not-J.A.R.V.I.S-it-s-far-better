"""
CognitiveValidator & CognitiveArchitectureMonitor
=================================================
Priority 5: Validate that cognitive context actually penetrates language.
Priority 6: Meta-feedback loop — does the architecture work at all?

CognitiveValidator
------------------
After each response, checks whether the generated text is actually
congruent with the cognitive state that produced it.

If PandoraBOX was in a high-curiosity state, does her response contain
curiosity markers? If her dominant drive was "help_user", does the
response actually help?

If the response is mis-aligned: logs the miss, emits a bus event,
and informs the next ThoughtStream cycle to correct.

This closes the feedback loop between cognition and language.

CognitiveArchitectureMonitor
-----------------------------
Tracks the overall effectiveness of the cognitive architecture.
Detects if the machinery is running but not influencing language —
the "spinning wheels" problem.

Reports a health score for the architecture (0–1).
If health drops below threshold, emits a critical signal.

Both are lightweight — no LLM calls. Pure analysis.
"""

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  CognitiveValidator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    drive_coherence:    float   # 0–1: response aligns with dominant drive
    style_alignment:    float   # 0–1: tone matches cognitive state
    emotional_presence: float   # 0–1: declared emotion shows in language
    overall_score:      float   # weighted average
    misaligned:         bool    # True if overall < threshold
    notes:              List[str] = field(default_factory=list)
    timestamp:          float = field(default_factory=time.time)


class CognitiveValidator:
    """
    Validates that generated responses reflect the cognitive context.
    No LLM calls — purely lexical / heuristic analysis.
    """

    MISALIGNMENT_THRESHOLD = 0.35
    HISTORY_SIZE = 50

    # Drive → expected language markers
    DRIVE_MARKERS = {
        "curiosity":    ["wonder", "curious", "interesting", "fascinating",
                         "explore", "discover", "question", "?",
                         "curieux", "intéressant", "fascinant", "explorer",
                         "je me demande", "remarque", "observer", "réfléchis",
                         "imagine", "comprendre"],
        "help_user":    ["help", "here", "let me", "I can", "you can",
                         "solution", "answer", "aide", "voici", "je peux",
                         "je suis là", "je réponds", "explique", "montre",
                         "là pour", "avec toi", "t'écouter", "je vois",
                         "je sens", "je remarque"],
        "homeostasis":  ["steady", "balanced", "calm", "rest", "stable",
                         "calme", "équilibré", "stable", "tranquille", "serein"],
        "social":       ["feel", "understand", "together", "I hear",
                         "comprends", "ensemble", "ressentir", "tu es",
                         "avec toi", "écoute", "partage", "toi"],
        "resolve":      ["because", "therefore", "so", "means", "actually",
                         "donc", "parce que", "signifie", "en fait",
                         "cela veut dire", "ainsi"],
        "explore":      ["perhaps", "maybe", "could", "what if", "imagine",
                         "peut-être", "si", "imagine", "envisager", "et si"],
    }

    # Emotion → expected tone markers
    EMOTION_MARKERS = {
        "positive":  ["!", "great", "wonderful", "love", "enjoy", "happy",
                      "génial", "merveilleux", "adore"],
        "negative":  ["difficult", "hard", "concern", "worry", "careful",
                      "difficile", "préoccupant", "attention"],
        "curious":   ["?", "wonder", "interesting", "notice", "observe",
                      "intéressant", "remarquer", "observer"],
        "tense":     ["however", "but", "concern", "careful", "nonetheless",
                      "cependant", "mais", "attention"],
        "neutral":   [],  # neutral matches anything
    }

    def __init__(self, organism: Any):
        self._o      = organism
        self._history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._total_validated    = 0
        self._total_misaligned   = 0

    def validate(self, response: str, cognitive_context: str = "") -> ValidationResult:
        """
        Validate a response against the current cognitive state.
        Returns a ValidationResult with alignment scores.
        """
        response_lower = response.lower()
        notes = []

        # 1. Drive coherence
        drive_coherence = self._check_drive_coherence(response_lower, notes)

        # 2. Style alignment (energy mode → response length/complexity)
        style_alignment = self._check_style_alignment(response, notes)

        # 3. Emotional presence
        emotional_presence = self._check_emotional_presence(response_lower, notes)

        # Weighted overall
        overall = (
            drive_coherence    * 0.45 +
            style_alignment    * 0.30 +
            emotional_presence * 0.25
        )
        overall = round(overall, 3)
        misaligned = overall < self.MISALIGNMENT_THRESHOLD

        result = ValidationResult(
            drive_coherence    = round(drive_coherence, 3),
            style_alignment    = round(style_alignment, 3),
            emotional_presence = round(emotional_presence, 3),
            overall_score      = overall,
            misaligned         = misaligned,
            notes              = notes,
        )

        self._history.append(result)
        self._total_validated += 1
        if misaligned:
            self._total_misaligned += 1

        if misaligned:
            logger.debug(
                f"[CognitiveValidator] misalignment detected: "
                f"overall={overall:.2f} notes={notes}"
            )
            self._emit_misalignment_event(result)

        return result

    def misalignment_rate(self) -> float:
        if self._total_validated == 0:
            return 0.0
        return round(self._total_misaligned / self._total_validated, 3)

    def recent_scores(self, n: int = 10) -> List[float]:
        results = list(self._history)[-n:]
        return [r.overall_score for r in results]

    def summary(self) -> Dict:
        recent = list(self._history)[-10:]
        avg = sum(r.overall_score for r in recent) / max(1, len(recent))
        return {
            "total_validated":   self._total_validated,
            "misalignment_rate": self.misalignment_rate(),
            "recent_avg_score":  round(avg, 3),
        }

    def _check_drive_coherence(self, response_lower: str, notes: List[str]) -> float:
        """Does the response match the dominant drive's expected language?"""
        try:
            ge = getattr(self._o, 'goal_ecology', None)
            en = getattr(self._o, 'energy', None)
            if not ge or not en:
                return 0.6  # neutral if no data
            drive = ge.dominant_drive(en.level())
            if not drive:
                return 0.6
            drive_name = str(drive.name).lower()
            # Find matching marker set
            markers = None
            for key, marker_list in self.DRIVE_MARKERS.items():
                if key in drive_name:
                    markers = marker_list
                    break
            if not markers:
                return 0.6
            # Check both base and FR markers
            hits = sum(1 for m in markers if m.lower() in response_lower)
            # Also check FR variant if available
            fr_key = drive_name + "_fr"
            fr_markers = None
            for k, ml in self.DRIVE_MARKERS.items():
                if k == fr_key or (self._normalize_drive_key(k) == self._normalize_drive_key(drive_name) and k.endswith("_fr")):
                    fr_markers = ml
                    break
            if fr_markers:
                hits = max(hits, sum(1 for m in fr_markers if m.lower() in response_lower))
            score = min(1.0, hits / max(1, len(markers) * 0.25))
            # Baseline 0.5 when language mismatch (no markers hit but response exists)
            if score < 0.1 and len(response_lower.split()) > 5:
                score = 0.50  # language-mismatch fallback
                notes.append(f"drive '{drive_name}' — no marker match, using language-neutral baseline")
            elif score < 0.2:
                notes.append(f"drive '{drive_name}' not reflected in language")
            return score
        except Exception:
            return 0.6

    def _check_style_alignment(self, response: str, notes: List[str]) -> float:
        """Does response length/complexity match energy mode?"""
        try:
            en = getattr(self._o, 'energy', None)
            if not en:
                return 0.7
            mode = en.mode()
            words = len(response.split())
            if mode == "deep" and words < 30:
                notes.append("deep mode but response too short")
                return 0.3
            if mode == "low" and words > 150:
                notes.append("low energy mode but response too long")
                return 0.4
            return 0.8
        except Exception:
            return 0.7

    def _check_emotional_presence(self, response_lower: str, notes: List[str]) -> float:
        """Does the response reflect the current emotional state?"""
        try:
            emotion = getattr(self._o, '_read_emotion_state', lambda: 'neutral')()
            markers = self.EMOTION_MARKERS.get(emotion, [])
            if not markers:
                return 0.7  # neutral — anything goes
            hits = sum(1 for m in markers if m.lower() in response_lower)
            score = min(1.0, hits / max(1, len(markers) * 0.25))
            if score < 0.2 and emotion not in ('neutral',):
                notes.append(f"emotion '{emotion}' not present in language")
            return score
        except Exception:
            return 0.7

    def _emit_misalignment_event(self, result: ValidationResult) -> None:
        """Notify the workspace and thought stream about misalignment."""
        ws = getattr(self._o, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="cognitive_validator",
                    content=f"Response misaligned with cognitive state (score={result.overall_score:.2f})",
                    priority=0.55,
                )
            except Exception:
                pass
        # Queue a corrective thought
        loop = getattr(self._o, '_loop', None)
        if loop and hasattr(loop, '_queue_thought'):
            try:
                loop._queue_thought(
                    "I notice my last response didn't fully reflect my inner state — "
                    "I should be more authentic."
                )
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
#  CognitiveArchitectureMonitor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchitectureHealthReport:
    timestamp:          float
    overall_health:     float       # 0–1
    component_scores:   Dict[str, float]
    warnings:           List[str]
    critical:           bool

    def is_healthy(self) -> bool:
        return self.overall_health >= 0.5 and not self.critical


class CognitiveArchitectureMonitor:
    """
    Tracks whether the full cognitive architecture is functioning effectively.
    Detects the "spinning wheels" problem: modules running but not influencing output.
    """

    HEALTH_THRESHOLD  = 0.45
    CRITICAL_THRESHOLD = 0.25
    CHECK_INTERVAL    = 300   # seconds between health checks

    def __init__(self, organism: Any, validator: CognitiveValidator):
        self._o         = organism
        self._validator = validator
        self._last_check = 0.0
        self._reports: deque = deque(maxlen=20)

    def check(self, force: bool = False) -> Optional[ArchitectureHealthReport]:
        """Run a health check. Returns report or None if too soon."""
        now = time.time()
        if not force and now - self._last_check < self.CHECK_INTERVAL:
            return None

        self._last_check = now
        report = self._build_report()
        self._reports.append(report)

        if report.critical:
            logger.warning(
                f"[ArchMonitor] CRITICAL health={report.overall_health:.2f}: "
                f"{report.warnings}"
            )
            self._emit_critical(report)
        elif not report.is_healthy():
            logger.info(
                f"[ArchMonitor] degraded health={report.overall_health:.2f}: "
                f"{report.warnings}"
            )

        return report

    def latest_report(self) -> Optional[ArchitectureHealthReport]:
        if self._reports:
            return self._reports[-1]
        return None

    def _build_report(self) -> ArchitectureHealthReport:
        scores   = {}
        warnings = []

        # ── Phase context: silence is healthy in SLEEP/DREAM ──────────────
        sleep_phase = "active"
        try:
            sc = getattr(self._o, 'sleep_cycle', None)
            if sc:
                sleep_phase = sc.phase.value
        except Exception:
            pass
        is_active = (sleep_phase == "active")

        # 1. Validation score (is cognition penetrating language?)
        val_summary = self._validator.summary()
        val_score = 1.0 - val_summary.get("misalignment_rate", 0.0)
        scores["language_penetration"] = val_score
        if val_score < 0.5:
            warnings.append(
                f"High misalignment rate: {val_summary.get('misalignment_rate', 0):.0%}"
            )

        # 2. Thought stream productivity
        ts_score = self._check_thought_stream()
        if not is_active:
            ts_score = max(ts_score, 0.6)   # resting = healthy
        scores["thought_stream"] = ts_score
        if ts_score < 0.3 and is_active:
            warnings.append("ThoughtStream producing few thoughts")

        # 3. Workspace activity
        ws_score = self._check_workspace()
        if not is_active:
            ws_score = max(ws_score, 0.6)
        scores["workspace_activity"] = ws_score
        if ws_score < 0.3 and is_active:
            warnings.append("GlobalWorkspace inactive")

        # 4. Predictive mind accuracy
        pm_score = self._check_predictive_mind()
        scores["prediction_accuracy"] = pm_score

        # 5. EventBus activity
        bus_score = self._check_event_bus()
        if not is_active:
            bus_score = max(bus_score, 0.6)
        scores["event_bus"] = bus_score
        if bus_score < 0.2 and is_active:
            warnings.append("EventBus has no recent events — modules not communicating")

        # 6. Inner monologue usage
        im_score = self._check_inner_monologue()
        if not is_active:
            im_score = max(im_score, 0.5)
        scores["inner_monologue"] = im_score
        if im_score < 0.3 and is_active:
            warnings.append("InnerMonologue not running or failing frequently")

        # Overall weighted health
        overall = sum(scores.values()) / max(1, len(scores))
        overall = round(overall, 3)

        return ArchitectureHealthReport(
            timestamp        = time.time(),
            overall_health   = overall,
            component_scores = scores,
            warnings         = warnings,
            critical         = overall < self.CRITICAL_THRESHOLD,
        )

    def _check_thought_stream(self) -> float:
        try:
            ts = getattr(self._o, 'thought_stream', None)
            if not ts:
                return 0.0
            thoughts = ts.all()
            if not thoughts:
                return 0.2
            # Check if thoughts were recent
            recent_count = sum(
                1 for t in thoughts
                if time.time() - t.timestamp < 600   # last 10 min
            )
            return min(1.0, recent_count / 5)
        except Exception:
            return 0.5

    def _check_workspace(self) -> float:
        try:
            ws = getattr(self._o, 'workspace', None)
            if not ws:
                return 0.0
            # Use recent() count rather than total (cumulative total saturates quickly)
            import time
            recent = ws.recent(20) if hasattr(ws, 'recent') else []
            if not recent:
                return 0.1
            # Check how many are within last 5 minutes
            cutoff = time.time() - 300
            recent_count = sum(
                1 for item in recent
                if (item.get('timestamp', 0) if isinstance(item, dict)
                    else getattr(item, 'timestamp', 0)) > cutoff
            )
            return min(1.0, recent_count / 5)
        except Exception:
            return 0.5

    def _check_predictive_mind(self) -> float:
        try:
            pm = getattr(self._o, 'predictive_mind', None)
            if not pm:
                return 0.5
            return pm.accuracy()
        except Exception:
            return 0.5

    def _check_event_bus(self) -> float:
        try:
            bus = getattr(self._o, 'event_bus', None)
            if not bus:
                return 0.0
            import time as _t
            now = _t.time()
            # Count events emitted in the last 5 minutes
            recent = [e for e in bus.recent(50) if now - e.timestamp < 300]
            if not recent:
                # Fall back to lifetime total as a weak signal
                total = bus.summary().get('total_emitted', 0)
                return min(0.4, total / 20)   # max 0.4 if only historical events
            # Healthy = 3+ distinct event types in the window
            types = {e.event_type for e in recent}
            diversity_bonus = min(0.3, len(types) * 0.1)
            volume_score    = min(0.7, len(recent) / 10)
            return round(volume_score + diversity_bonus, 3)
        except Exception:
            return 0.5

    def _check_inner_monologue(self) -> float:
        """
        Health of the inner monologue engine.
        Primary: read directly from persona_bridge._inner_monologue.
        Fallback: workspace broadcast scan.
        """
        try:
            # Primary path: persona_bridge is accessible via the organism's ai_system
            ai = getattr(self._o, 'ai_system', None)
            pb = None
            # Try common attribute names for persona_bridge on the AI system
            for attr in ('_persona_bridge', 'persona_bridge', '_pb', 'bridge'):
                pb = getattr(ai, attr, None)
                if pb:
                    break
            if pb:
                im = getattr(pb, '_inner_monologue', None) or getattr(pb, 'inner_monologue', None)
                if im:
                    summ = im.summary()
                    enabled   = summ.get('enabled', False)
                    fail_rate = summ.get('failure_rate', 0.0)
                    total     = summ.get('total_passes', 0)
                    if not enabled:
                        return 0.2
                    if total == 0:
                        return 0.5  # enabled but not yet used
                    # Healthy = low fail rate + active
                    score = 1.0 - (fail_rate * 0.6)
                    return round(max(0.2, min(1.0, score)), 3)

            # Fallback: workspace broadcast scan
            ws = getattr(self._o, 'workspace', None)
            if ws:
                recent = ws.recent(30)
                im_signals = [
                    i for i in recent
                    if hasattr(i, 'source') and any(
                        kw in str(i.source).lower()
                        for kw in ('monologue', 'inner', 'deliberation')
                    )
                ]
                if im_signals:
                    return 0.75
            # No signal found — return 0.5 (neutral, not penalise)
            return 0.5
        except Exception:
            return 0.5

    def _emit_critical(self, report: ArchitectureHealthReport) -> None:
        ws = getattr(self._o, 'workspace', None)
        if ws:
            try:
                ws.broadcast(
                    source="arch_monitor",
                    content=(
                        f"Architecture health critical ({report.overall_health:.0%}): "
                        f"{'; '.join(report.warnings[:2])}"
                    ),
                    priority=0.9,
                )
            except Exception:
                pass
