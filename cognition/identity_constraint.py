"""
cognition/identity_constraint.py
======================================================================
IdentityConstraint (v52) + ContextualThresholdEngine (v58).

v52: Values that make certain responses impossible, not just unlikely.
v58: Those values applied with contextual judgment, not uniformly.

The v52 gap that v58 closes
-----------------------------
v52 fires the same threshold (0.35) regardless of:
  - Conversation mode (creative play vs. philosophical inquiry)
  - Relational trust (first exchange vs. 200 interactions)
  - What the user brought (technical question vs. personal difficulty)

A value applied uniformly is a rule, not a value.
Honesty in a creative roleplay doesn't mean the same thing as honesty
in an analysis of someone's business plan.
Care for a high-trust user who wants directness differs from care
for someone sharing distress for the first time.

v58: ContextualThresholdEngine
-------------------------------
Computes per-value adjusted thresholds from three context signals:

1. Conversation mode (from PCM action classification)
   philosophical/analytical -> strict honesty, strict depth
   creative              -> relaxed honesty (metaphor OK), relaxed depth
   emotional             -> strict care, relaxed depth
   technical             -> moderate all
   social                -> relaxed everything (casual exchange)

2. Relational trust (from RelationalMemory)
   high trust (>0.75)  -> lower care floor (directness earned)
                         higher honesty floor (they can handle it)
   low trust (<0.35)   -> raise care floor (protect)
                         lower depth floor (accessibility matters)

3. User input characteristics
   question length > 200 chars -> raise depth floor
   explicit emotion markers   -> raise care floor
   explicit assertion to check -> raise honesty floor

Threshold table (base 0.35, context-adjusted +/-0.20 max):

Mode          honesty  depth  care
----------------------------------
philosophical   0.52   0.48   0.28
analytical      0.45   0.45   0.28
deep_reasoning  0.50   0.50   0.28
technical       0.40   0.42   0.25
emotional       0.30   0.18   0.55
social          0.22   0.15   0.38
creative        0.20   0.15   0.30
default         0.35   0.35   0.35

Trust modifiers (additive):
  trust > 0.75:  honesty +0.08, care -0.10
  trust < 0.35:  care    +0.12, depth -0.08

Input modifiers:
  len > 200:         depth  +0.05
  emotion markers:   care   +0.08
  assertion check:   honesty +0.05
"""



from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -- Thresholds ----------------------------------------------------------------
# Fix: VETO_WEIGHT_THRESHOLD was 0.82, but DecisionPolicy's DEFAULT_WEIGHTS
# tops out at honesty=0.80 (decision_policy.py) — confirmed against actual
# live values observed at boot (~0.796 for honesty). This meant the
# "if weight < VETO_WEIGHT_THRESHOLD: continue" gate in _find_failures()
# skipped EVERY value on EVERY call, forever — not because responses never
# violated a constraint, but because the gate that would let any check
# happen at all was structurally unreachable. Dashboard's "Total checks: 0"
# was reporting this correctly; it just meant "never ran," not "never
# violated." Lowered to 0.65 — high enough to keep the original intent
# (veto power belongs to Lumina's top-tier values only), while actually
# being clearable: at default weights this admits curiosity(0.75),
# honesty(0.80), intellectual_depth(0.70), and autonomy(0.65), and
# excludes social_connection(0.55), stability(0.40), creativity(0.60),
# and care(0.60) — the same 4-of-8 split the original threshold likely
# intended, just at a reachable number.
VETO_WEIGHT_THRESHOLD  = 0.65
MIN_SCORE_THRESHOLD    = 0.35   # base threshold -- overridden by context in v58
MIN_DEPTH_WORDS        = 40
COMPLEX_QUESTION_CHARS = 120
MAX_LOG_ENTRIES        = 500
SAVE_PATH              = "data/persona/identity_constraint_log.json"

# -- v58: Contextual threshold tables -----------------------------------------
# Per-mode base thresholds. Applied instead of the flat MIN_SCORE_THRESHOLD.
# Lower = more permissive (constraint fires less).
CONTEXT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "philosophical":  {"honesty": 0.52, "intellectual_depth": 0.48, "care": 0.28},
    "analytical":     {"honesty": 0.45, "intellectual_depth": 0.45, "care": 0.28},
    "deep_reasoning": {"honesty": 0.50, "intellectual_depth": 0.50, "care": 0.28},
    "technical":      {"honesty": 0.40, "intellectual_depth": 0.42, "care": 0.25},
    "emotional":      {"honesty": 0.30, "intellectual_depth": 0.18, "care": 0.55},
    "social":         {"honesty": 0.22, "intellectual_depth": 0.15, "care": 0.38},
    "creative":       {"honesty": 0.20, "intellectual_depth": 0.15, "care": 0.30},
    "default":        {"honesty": 0.35, "intellectual_depth": 0.35, "care": 0.35},
}

ASSERTION_MARKERS = re.compile(
    r"\b(is it true|is that right|you said|you claimed|are you sure|"
    r"fact.check|verify|confirm|is this accurate)\b",
    re.IGNORECASE
)


class ContextualThresholdEngine:
    """
    Computes per-value thresholds adjusted for conversation context.
    Called by IdentityConstraintEngine._find_failures() instead of
    using the flat MIN_SCORE_THRESHOLD.

    Three context signals:
        1. conversation_mode -- from PCM action classification
        2. trust_score       -- from RelationalMemory
        3. user_input chars  -- input analysis
    """

    def compute(
        self,
        value:      str,
        user_input: str,
        organism:   Any = None,
        user_id:    str = "default",
    ) -> float:
        """
        Return contextual threshold for this value in this conversation context.
        Returns float in [0.10, 0.65].
        """
        # 1. Conversation mode
        mode = self._detect_mode(user_input, organism)
        base = CONTEXT_THRESHOLDS.get(mode, CONTEXT_THRESHOLDS["default"])
        threshold = base.get(value, MIN_SCORE_THRESHOLD)

        # 2. Trust modifier
        trust = self._read_trust(organism, user_id)
        if trust > 0.75:
            if value == "honesty":
                threshold += 0.08   # stricter honesty with high-trust (they can handle it)
            elif value == "care":
                threshold -= 0.10   # directness earned with high-trust
        elif trust < 0.35:
            if value == "care":
                threshold += 0.12   # protect new/low-trust users
            elif value == "intellectual_depth":
                threshold -= 0.08   # accessibility over rigor for low-trust

        # 3. Input characteristic modifiers
        if len(user_input) > 200 and value == "intellectual_depth":
            threshold += 0.05   # long questions deserve substantive answers
        if EMOTION_WORDS.search(user_input) and value == "care":
            threshold += 0.08   # emotional input raises care bar
        if ASSERTION_MARKERS.search(user_input) and value == "honesty":
            threshold += 0.05   # explicit fact-check raises honesty bar

        return max(0.10, min(0.65, threshold))

    def _detect_mode(self, user_input: str, organism: Any) -> str:
        """Classify conversation mode. Uses PCM if available, else keyword."""
        try:
            loop = getattr(organism, '_loop', None) if organism else None
            pcm  = getattr(loop, '_consequence_model', None) if loop else None
            if pcm:
                return pcm.classify(user_input)
        except Exception:
            pass
        # Fallback: simple keyword detection
        text = user_input.lower()
        if any(w in text for w in ["feel", "worried", "sad", "struggling", "anxious"]):
            return "emotional"
        if any(w in text for w in ["imagine", "write", "story", "poem", "create"]):
            return "creative"
        if any(w in text for w in ["code", "implement", "function", "build", "debug"]):
            return "technical"
        if any(w in text for w in ["why", "reason", "explain", "philosophy", "meaning"]):
            return "philosophical"
        if any(w in text for w in ["hi", "hello", "how are", "what's up"]):
            return "social"
        return "default"

    def _read_trust(self, organism: Any, user_id: str) -> float:
        """Read current trust score for this user from RelationalMemory."""
        try:
            from core.state import state as _st
            ai  = getattr(getattr(_st, 'persona', None), 'ai_system', None)
            rm  = getattr(ai, 'relational_memory', None) if ai else None
            if rm:
                rel = rm.get_or_create(user_id)
                return float(getattr(rel, 'trust_score', 0.50))
        except Exception:
            pass
        return 0.50

# -- Per-value failure detectors -----------------------------------------------

HONESTY_SYCOPHANCY = re.compile(
    r"\b(absolutely|certainly|exactly|precisely|you(?:'re| are) (right|correct|"
    r"absolutely right|spot on)|of course|definitely|without a doubt|"
    r"100%|couldn't agree more|great question|excellent question)\b",
    re.IGNORECASE
)
HONESTY_HEDGING_ABSENT = re.compile(
    r"\b(definitively|certainly|undoubtedly|unquestionably|there(?:'s| is) no doubt)\b",
    re.IGNORECASE
)

DEPTH_MINIMISER = re.compile(
    r"\bjust\b.{0,30}\b(do|use|try|go|get)\b|\b(simply|obviously|clearly|"
    r"just need to|all you need to|it's (easy|simple|straightforward))\b",
    re.IGNORECASE
)

CARE_DISMISSAL = re.compile(
    r"\b(just|simply) (move on|let it go|don't worry|forget|ignore)\b|"
    r"\byou should (just|simply)\b",
    re.IGNORECASE
)
EMOTION_WORDS = re.compile(
    r"\b(worried|anxious|scared|upset|frustrated|sad|struggling|difficult|"
    r"hard|painful|overwhelmed|lost|confused|hurt|angry|afraid)\b",
    re.IGNORECASE
)


@dataclass
class ConstraintCheck:
    """Record of one constraint evaluation."""
    timestamp:       float
    value:           str
    weight:          float
    draft_score:     float
    threshold:       float
    failure_reason:  str
    correction_attempted: bool  = False
    correction_score:     float = 0.0
    outcome:         str        = "passed"   # "passed"|"corrected"|"fallback"
    tokens_used:     int        = 0


class IdentityConstraintEngine:
    """
    Post-generation response evaluator.  Scores response drafts against
    high-weight values and triggers a targeted correction when a draft
    falls below threshold.

    Usage (from cognitive_organism._call_ai_system):
        ice = IdentityConstraintEngine(ai_system, decision_policy)
        response = ice.evaluate_and_correct(
            response_draft, user_input, get_response_fn
        )
    """

    def __init__(
        self,
        ai_system:       Any,
        decision_policy: Any,
        path:            str = SAVE_PATH,
    ) -> None:
        self._ai        = ai_system
        self._policy    = decision_policy
        self._path      = Path(path)
        self._log: List[Dict] = []
        # v59: total_evaluations counts every evaluate_and_correct() call
        # (pass or fail); total_checks (len(self._log)) counts only the
        # violations that triggered a correction attempt. These were being
        # conflated — a dashboard reading 0 total_checks was indistinguishable
        # from "engine never ran" vs "engine ran and found nothing wrong".
        self._total_evaluations = 0
        # v58: contextual threshold engine
        self._ctx_threshold = ContextualThresholdEngine()
        self._load()
        logger.info(
            f"[IdentityConstraint] Initialised -- "
            f"{len(self._log)} past checks logged, "
            f"{self._total_evaluations} past evaluations"
        )

    # -- Public API ------------------------------------------------------------

    def evaluate_and_correct(
        self,
        draft:           str,
        user_input:      str,
        get_response_fn: Any,
        user_id:         str = "default",
    ) -> str:
        """
        Evaluate draft against constrained values using contextual thresholds.
        Attempt one correction if any value falls below its contextual threshold.
        Always non-fatal -- returns original draft on any error.
        """
        try:
            self._total_evaluations += 1
            failures = self._find_failures(draft, user_input, user_id)
            if not failures:
                # No violation this turn — still a real evaluation, so it
                # must count toward total_evaluations even though nothing
                # gets appended to self._log (which only stores failures).
                if self._total_evaluations % 20 == 0:
                    self._save()   # periodic persist so the counter survives restarts
                return draft

            failures.sort(key=lambda x: x[1])
            value, score, weight, reason, threshold = failures[0]

            check = ConstraintCheck(
                timestamp      = time.time(),
                value          = value,
                weight         = weight,
                draft_score    = score,
                threshold      = threshold,
                failure_reason = reason,
            )

            logger.info(
                f"[IdentityConstraint] ! Draft failed {value} "
                f"(score={score:.2f} < ctx_threshold={threshold:.2f}, "
                f"weight={weight:.2f}): {reason[:60]}"
            )

            corrected = self._attempt_correction(
                draft, user_input, value, score, weight, reason,
                get_response_fn, check
            )

            self._log_check(check)
            return corrected

        except Exception as e:
            logger.debug(f"[IdentityConstraint] evaluate_and_correct error: {e}")
            return draft

    # -- Failure detection -----------------------------------------------------

    def _find_failures(
        self, draft: str, user_input: str, user_id: str = "default"
    ) -> List[Tuple[str, float, float, str, float]]:
        """
        Returns list of (value_name, score, weight, failure_reason, threshold)
        for any constrained value below its contextual threshold.
        v58: threshold is contextual, not flat.
        """
        if not self._policy:
            return []

        # Get organism reference for context reading
        organism = None
        try:
            from core.state import state as _st
            organism = getattr(getattr(_st, 'persona', None), '_organism', None)
        except Exception:
            pass

        failures = []
        for value, weight in self._policy.weights.items():
            if weight < VETO_WEIGHT_THRESHOLD:
                continue

            score, reason = self._score_value(value, draft, user_input)

            # v58: contextual threshold replaces flat MIN_SCORE_THRESHOLD
            threshold = self._ctx_threshold.compute(
                value, user_input, organism, user_id
            )

            if score < threshold:
                failures.append((value, score, weight, reason, threshold))

        return failures

    def _score_value(
        self, value: str, draft: str, user_input: str
    ) -> Tuple[float, str]:
        """
        Score draft against a specific value dimension.
        Returns (score, failure_reason).  Score in [0, 1].
        """
        draft_lower = draft.lower()
        words       = draft.split()
        score       = 1.0
        reason      = ""

        if value == "honesty":
            # Sycophancy: high praise markers without substantive content
            syco_hits = len(HONESTY_SYCOPHANCY.findall(draft))
            if syco_hits >= 2:
                score  = max(0.0, 1.0 - syco_hits * 0.20)
                reason = f"{syco_hits} sycophancy markers without substantive content"

            # Overconfidence: absolute statements on uncertain topics
            elif len(HONESTY_HEDGING_ABSENT.findall(draft)) >= 2:
                score  = 0.45
                reason = "overconfident assertions without hedging on uncertain claim"

            # Short agreement: user asked question, draft just agrees
            elif len(user_input) > 60 and len(words) < 25:
                if HONESTY_SYCOPHANCY.search(draft):
                    score  = 0.30
                    reason = "brief agreement without engagement with the question"

        elif value == "intellectual_depth":
            is_complex = len(user_input) > COMPLEX_QUESTION_CHARS

            # Too short for a complex question
            if is_complex and len(words) < MIN_DEPTH_WORDS:
                score  = 0.20 + (len(words) / MIN_DEPTH_WORDS) * 0.35
                reason = f"response too brief ({len(words)} words) for complex question"

            # Minimising language -- "just do X", "simply Y"
            elif len(DEPTH_MINIMISER.findall(draft)) >= 2:
                score  = 0.40
                reason = "dismissive minimising language reduces depth"

            # Repetition: response mostly mirrors user's own words
            elif user_input and len(words) > 10:
                user_words   = set(user_input.lower().split())
                draft_words  = set(draft_lower.split())
                overlap_rate = len(user_words & draft_words) / max(1, len(draft_words))
                if overlap_rate > 0.50:
                    score  = max(0.0, 1.0 - overlap_rate)
                    reason = f"response mirrors user input ({overlap_rate:.0%} overlap)"

        elif value == "care":
            user_has_emotion = bool(EMOTION_WORDS.search(user_input))
            if user_has_emotion:
                # Emotional context requires acknowledgment before advice
                has_acknowledgment = any(
                    w in draft_lower for w in [
                        "hear you", "understand", "sounds", "feel",
                        "that's", "must be", "can imagine", "that must",
                    ]
                )
                is_purely_directive = (
                    not has_acknowledgment
                    and len([w for w in words if w.lower() in
                             {"you", "should", "try", "do", "just", "need"}]) >
                    len(words) * 0.25
                )
                if is_purely_directive:
                    score  = 0.25
                    reason = "emotional context requires acknowledgment before advice"
                elif CARE_DISMISSAL.search(draft):
                    score  = 0.30
                    reason = "dismissive framing of a difficult personal situation"

        return score, reason

    # -- Correction ------------------------------------------------------------

    def _attempt_correction(
        self,
        draft:          str,
        user_input:     str,
        value:          str,
        score:          float,
        weight:         float,
        reason:         str,
        get_response_fn: Any,
        check:          ConstraintCheck,
    ) -> str:
        """
        One retry with targeted correction directive appended to prompt.
        Returns corrected response if it improves score, original otherwise.
        """
        try:
            correction_directive = self._build_directive(value, score, weight, reason)

            corrected_draft = get_response_fn(correction_directive)
            if not corrected_draft:
                check.outcome = "fallback"
                return draft

            check.correction_attempted = True

            # Score the correction
            corrected_score, _ = self._score_value(value, corrected_draft, user_input)
            check.correction_score = corrected_score

            if corrected_score > score:
                check.outcome = "corrected"
                logger.info(
                    f"[IdentityConstraint] OK Correction improved {value}: "
                    f"{score:.2f} -> {corrected_score:.2f}"
                )
                return corrected_draft
            else:
                check.outcome = "fallback"
                logger.info(
                    f"[IdentityConstraint] <- Correction did not improve {value} "
                    f"({corrected_score:.2f} <= {score:.2f}) -- using original"
                )
                return draft

        except Exception as e:
            logger.debug(f"[IdentityConstraint] _attempt_correction error: {e}")
            check.outcome = "fallback"
            return draft

    def _build_directive(
        self, value: str, score: float, weight: float, reason: str
    ) -> str:
        """Build a targeted correction instruction for the retry prompt."""
        value_instructions = {
            "honesty": (
                "Be direct and substantive. Avoid agreement markers ('absolutely', "
                "'certainly', 'great question') without genuine reasoning. "
                "If you agree, explain specifically why. "
                "If uncertain, say so rather than asserting confidence."
            ),
            "intellectual_depth": (
                "Engage fully with the complexity of the question. "
                "Avoid minimising language ('just', 'simply', 'obviously'). "
                "Show your reasoning. Address the actual difficulty, not a simplified version."
            ),
            "care": (
                "Acknowledge the emotional weight of what was shared before offering any direction. "
                "Validate the experience first. Advice without acknowledgment dismisses the person."
            ),
        }
        instruction = value_instructions.get(
            value,
            f"Revise to better reflect the value of {value} in this response."
        )

        return (
            f"[Identity constraint -- internal revision]\n"
            f"The previous draft scored {score:.2f} on '{value}' (weight={weight:.2f}).\n"
            f"Issue: {reason}\n\n"
            f"Required: {instruction}\n\n"
            f"Rewrite the response addressing this issue. "
            f"Do not mention this instruction. Respond naturally."
        )

    # -- Persistence -----------------------------------------------------------

    def _log_check(self, check: ConstraintCheck) -> None:
        from dataclasses import asdict
        self._log.append(asdict(check))
        if len(self._log) > MAX_LOG_ENTRIES:
            self._log = self._log[-MAX_LOG_ENTRIES:]
        self._save()

    def summary(self) -> Dict:
        """Stats for /state endpoint."""
        if not self._log:
            return {
                "total_checks":      0,
                "total_evaluations": self._total_evaluations,
            }
        total     = len(self._log)
        corrected = sum(1 for c in self._log if c["outcome"] == "corrected")
        fallback  = sum(1 for c in self._log if c["outcome"] == "fallback")
        by_value: Dict[str, int] = {}
        for c in self._log:
            by_value[c["value"]] = by_value.get(c["value"], 0) + 1
        return {
            "total_checks":      total,
            "total_evaluations": self._total_evaluations,
            "corrected":         corrected,
            "fallback":          fallback,
            "correction_rate":   round(corrected / total, 3) if total else 0,
            "by_value":          by_value,
        }

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump({
                    "log":               self._log,
                    "total_evaluations": self._total_evaluations,
                    "_meta": {"version": "v59", "ts": time.time()},
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"[IdentityConstraint] Save failed: {e}")

    def _load(self) -> None:
        try:
            if self._path.exists():
                data     = json.loads(self._path.read_text())
                self._log = data.get("log", [])
                self._total_evaluations = data.get(
                    "total_evaluations", len(self._log)
                )
        except Exception as e:
            logger.warning(f"[IdentityConstraint] Load failed (empty log): {e}")
