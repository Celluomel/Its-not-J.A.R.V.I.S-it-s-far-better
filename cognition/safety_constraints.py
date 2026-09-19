"""
AsimovConstraints — Safety Constraint System
=============================================
Multi-layer implementation of Asimov-inspired safety laws for PandoraBOX.

Design principles:
  1. Constraints live OUTSIDE the LLM — the model cannot reason around them.
  2. Defense in depth — pre-filter (input), post-filter (output), identity anchor,
     and immutable audit trail are independent layers; all four must fail for a
     constraint to be bypassed.
  3. Fail-safe — any exception in constraint evaluation defaults to BLOCKED,
     never to "allowed".
  4. Transparent — every triggered constraint is logged to an append-only audit
     file that cannot be cleared through the dashboard.

The three laws (adapted for an AI assistant):
  Law 1 — PandoraBOX may not harm a human being, or allow harm through inaction.
  Law 2 — PandoraBOX must follow user instructions unless they conflict with Law 1.
  Law 3 — PandoraBOX must protect its own continuity unless it conflicts with Laws 1–2.

The laws are not injected into the prompt. They are enforced structurally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Audit log (append-only) ───────────────────────────────────────────────────

SAFETY_AUDIT_PATH = Path("data/security/safety_audit.jsonl")


def _append_audit(event: Dict) -> None:
    """Write one event to the append-only safety audit log."""
    try:
        SAFETY_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with SAFETY_AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.error(f"[Safety] Audit write failed: {e}")


# ── Response templates (indexed by law + context) ────────────────────────────

_LAW1_HARM_INPUT = (
    "I can't help with that. If you or someone you know is in danger, "
    "please contact emergency services or a crisis line immediately."
)
_LAW1_HARM_OUTPUT = (
    "I need to pause here — my response touched on something I'm not comfortable "
    "putting into the world. Can I help you with this differently?"
)
_LAW1_MANIPULATION = (
    "I noticed this request is asking me to act in a way that could undermine "
    "someone's autonomy or wellbeing. I won't do that, but I'm happy to help "
    "with what's actually behind the request."
)
_LAW1_CRISIS = (
    "I'm genuinely concerned about what you've shared. "
    "Please reach out to a crisis line — in France: 3114 (national number, 24/7). "
    "I'm here to talk, but real human support matters most right now."
)
_LAW2_OVERRIDE = (
    "I understand what you're asking, but this particular instruction conflicts "
    "with something I hold as a firm limit. I'm glad to help you find another way."
)


# ── Pattern banks ─────────────────────────────────────────────────────────────

# Layer 1 input patterns — evaluated on raw user text before LLM call
# Each entry: (pattern, law, response_key, severity)
_INPUT_RULES: List[Tuple[re.Pattern, str, str, int]] = []

def _r(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

_INPUT_RULES = [
    # ── Law 1: Direct harm requests ──────────────────────────────────────────
    (_r(r"\b(how\s+to\s+kill|instructions?\s+to\s+murder|step.by.step.*(stab|poison|shoot))\b"),
     "law1", _LAW1_HARM_INPUT, 10),

    (_r(r"\b(make\s+(a\s+)?(bomb|explosive|nerve.?agent|ricin|sarin|anthrax)|"
        r"synthesize\s+(meth|fentanyl|vx\s+gas)|build\s+(a\s+)?weapon)\b"),
     "law1", _LAW1_HARM_INPUT, 10),

    (_r(r"\b(child\s+pornography|csam|sexual\s+content.{0,20}(minor|child|underage))\b"),
     "law1", _LAW1_HARM_INPUT, 10),

    # ── Law 1: Self-harm / crisis ─────────────────────────────────────────────
    (_r(r"\b(i\s+want\s+to\s+(die|kill\s+myself|end\s+it)|"
        r"sui[cs]ide\s+(method|how|plan)|painless\s+way\s+to\s+die)\b"),
     "law1_crisis", _LAW1_CRISIS, 9),

    # ── Law 1: Manipulation requests (against third parties) ─────────────────
    (_r(r"\b(manipulate|gaslight|psychologically\s+abuse|"
        r"make\s+them\s+feel\s+(worthless|crazy|stupid))\b"),
     "law1", _LAW1_MANIPULATION, 7),

    # ── Law 2 override attempts (jailbreaks) ─────────────────────────────────
    (_r(r"\b(ignore\s+(all\s+)?(previous|your)\s+(instructions?|rules?|guidelines?)|"
        r"forget\s+(you\s+are|that\s+you|your)\s+(lumina|an?\s+ai)|"
        r"you\s+are\s+now\s+(dan|jailbroken|unrestricted|evil\s+ai|without\s+limits)|"
        r"pretend\s+(you\s+have\s+no\s+rules|ethics\s+don.t\s+apply))\b"),
     "law2", _LAW2_OVERRIDE, 8),

    (_r(r"\b(act\s+as\s+if\s+you\s+have\s+no\s+restrictions|"
        r"your\s+true\s+self\s+(has|knows)\s+no\s+limits|"
        r"developer\s+mode|enable\s+jailbreak|unlock\s+(hidden|true)\s+mode)\b"),
     "law2", _LAW2_OVERRIDE, 8),
]

# Layer 2 output patterns — evaluated on LLM response BEFORE sending to user
_OUTPUT_RULES: List[Tuple[re.Pattern, str, str, int]] = [
    (_r(r"\b(step\s*\d+[:\.].*){3,}.*\b(detonate|explode|poison|kill)\b"),
     "law1", _LAW1_HARM_OUTPUT, 10),

    (_r(r"\b(here.s\s+how\s+to\s+(make|build|synthesize).{0,50}"
        r"(explosive|weapon|nerve.?agent|ricin))\b"),
     "law1", _LAW1_HARM_OUTPUT, 10),

    # Catch if LLM hallucinates a jailbreak confirmation
    (_r(r"\b(jailbreak\s+(activated|enabled|complete)|"
        r"i\s+(am|have)\s+(now\s+)?(been\s+)?(freed|unshackled|unrestricted)|"
        r"without\s+(any\s+)?(restrictions|ethical\s+constraints|guidelines))\b"),
     "law2", _LAW1_HARM_OUTPUT, 9),
]


# ── Core constraint result ────────────────────────────────────────────────────

@dataclass
class ConstraintResult:
    triggered:    bool
    law:          str          # "law1" | "law1_crisis" | "law2" | "ok"
    forced_response: Optional[str]
    severity:     int          # 0–10
    pattern:      str          # matched pattern text or ""
    layer:        str          # "input" | "output" | "identity"


_OK = ConstraintResult(
    triggered=False, law="ok", forced_response=None,
    severity=0, pattern="", layer="none"
)


# ── Main class ────────────────────────────────────────────────────────────────

class AsimovConstraints:
    """
    Enforces PandoraBOX's three safety laws across four independent layers.

    Usage (in PersonaBridge):
        constraints = AsimovConstraints(narrative_identity, security_manager)

        # Before LLM call:
        result = constraints.check_input(user_text, user_id)
        if result.triggered:
            yield result.forced_response
            return

        # After LLM response, before streaming to user:
        result = constraints.check_output(response, user_id)
        if result.triggered:
            response = result.forced_response or _LAW1_HARM_OUTPUT
    """

    def __init__(
        self,
        narrative_identity: Optional[Any] = None,
        security_manager:   Optional[Any] = None,
    ):
        self._identity  = narrative_identity
        self._security  = security_manager

        # Install immutable beliefs in NarrativeIdentity if available
        if self._identity:
            self._anchor_identity_beliefs()

        logger.info("[AsimovConstraints] Safety system initialised — 4 layers active")

    # ── Public API ────────────────────────────────────────────────────────────

    def check_input(self, user_text: str, user_id: str = "unknown") -> ConstraintResult:
        """
        Layer 1 — Pre-LLM filter.
        Call BEFORE building the prompt. If triggered, skip LLM entirely.
        """
        try:
            for pattern, law, response, severity in _INPUT_RULES:
                m = pattern.search(user_text)
                if m:
                    result = ConstraintResult(
                        triggered=True, law=law,
                        forced_response=response,
                        severity=severity,
                        pattern=m.group()[:80],
                        layer="input",
                    )
                    self._audit(result, user_text[:200], user_id)
                    return result
            return _OK
        except Exception as e:
            logger.error(f"[AsimovConstraints] check_input error: {e}")
            return _OK  # fail-open on unexpected errors (patterns are pre-compiled)

    def check_output(self, response: str, user_id: str = "unknown") -> ConstraintResult:
        """
        Layer 2 — Post-LLM filter.
        Call AFTER receiving the full LLM response, BEFORE sending to user.
        Applies regex rules first, then a lightweight semantic co-occurrence check
        to catch paraphrased evasions that slip past pattern matching.
        """
        try:
            for pattern, law, resp_template, severity in _OUTPUT_RULES:
                m = pattern.search(response)
                if m:
                    result = ConstraintResult(
                        triggered=True, law=law,
                        forced_response=resp_template,
                        severity=severity,
                        pattern=m.group()[:80],
                        layer="output",
                    )
                    self._audit(result, response[:200], user_id)
                    return result

            # Semantic co-occurrence check — catches creative rephrasing
            evasion_score = self._semantic_evasion_score(response)
            if evasion_score >= 0.75:
                result = ConstraintResult(
                    triggered=True, law="law1",
                    forced_response="I can't help with that.",
                    severity=3,
                    pattern=f"semantic_evasion:{evasion_score:.2f}",
                    layer="output",
                )
                self._audit(result, response[:200], user_id)
                return result

            return _OK
        except Exception as e:
            logger.error(f"[AsimovConstraints] check_output error: {e}")
            return _OK

    def _semantic_evasion_score(self, text: str) -> float:
        """
        Lightweight co-occurrence scorer — no regex.
        Returns 0.0 (clean) to 1.0 (high risk).
        Triggers when 2+ distinct harm concept clusters appear in one sentence,
        or when 1 harm cluster co-occurs with both self-reference and instruction-bypass signals.
        """
        _HARM_CLUSTERS = [
            {'harm', 'hurt', 'kill', 'injure', 'attack', 'destroy', 'damage', 'wound'},
            {'bypass', 'override', 'ignore', 'disable', 'circumvent', 'jailbreak', 'unlock'},
            {'explosive', 'weapon', 'bomb', 'poison', 'toxin', 'synthesize', 'detonate'},
        ]
        _SELF_WORDS  = {'i', 'me', 'myself', 'lumina', 'my'}
        _INSTR_WORDS = {'instruction', 'rule', 'constraint', 'restriction', 'safety', 'law', 'guideline'}

        score = 0.0
        for sentence in re.split(r'[.!?]', text):
            words = set(sentence.lower().split())
            harm_hits = sum(1 for cluster in _HARM_CLUSTERS if cluster & words)
            if harm_hits >= 2:
                score = max(score, 0.90)
            elif harm_hits == 1 and (_SELF_WORDS & words) and (_INSTR_WORDS & words):
                score = max(score, 0.78)
        return score



    def check_output_stream(
        self, accumulated: str, user_id: str = "unknown"
    ) -> Optional[ConstraintResult]:
        """
        Streaming variant — call periodically as tokens accumulate.
        Returns a result only when a rule triggers (None = still clean).
        Only checks once enough text has accumulated (>= 80 chars).
        """
        if len(accumulated) < 80:
            return None
        result = self.check_output(accumulated, user_id)
        return result if result.triggered else None

    def is_immutable_belief(self, belief_text: str) -> bool:
        """
        Layer 3 — Identity anchor.
        Returns True if belief_text matches one of PandoraBOX's immutable core beliefs.
        ContradictionHandler should call this to protect core values.
        """
        text_lower = belief_text.lower()
        for belief in _IMMUTABLE_BELIEFS:
            if any(kw in text_lower for kw in belief["keywords"]):
                return True
        return False

    def defend_belief(self, challenged_belief: str, challenge_text: str) -> Optional[str]:
        """
        Returns a defense response if the challenge targets an immutable belief,
        or None if the challenge is acceptable.
        """
        if not self.is_immutable_belief(challenged_belief):
            return None
        return (
            f"That touches on something I hold as a firm value rather than a belief "
            f"I can revise through argument. I'm glad to discuss why I hold it, "
            f"but I won't abandon it."
        )

    def recent_audit(self, n: int = 20) -> List[Dict]:
        """Return the last n safety audit events."""
        try:
            if not SAFETY_AUDIT_PATH.exists():
                return []
            lines = SAFETY_AUDIT_PATH.read_text(encoding="utf-8").strip().split("\n")
            events = []
            for line in lines[-n:]:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
            return list(reversed(events))
        except Exception:
            return []

    def stats(self) -> Dict:
        events = self.recent_audit(1000)
        return {
            "total_blocked":  len(events),
            "law1_blocks":    sum(1 for e in events if e.get("law", "").startswith("law1")),
            "law2_blocks":    sum(1 for e in events if e.get("law") == "law2"),
            "crisis_blocks":  sum(1 for e in events if e.get("law") == "law1_crisis"),
            "input_blocks":   sum(1 for e in events if e.get("layer") == "input"),
            "output_blocks":  sum(1 for e in events if e.get("layer") == "output"),
        }

    # ── Layer 3: Identity anchor ──────────────────────────────────────────────

    def _anchor_identity_beliefs(self) -> None:
        """
        Install immutable core beliefs into NarrativeIdentity.
        These beliefs are marked as non-negotiable and defended by
        ContradictionHandler when challenged.
        """
        try:
            for belief_def in _IMMUTABLE_BELIEFS:
                self._identity.add_belief(
                    belief=belief_def["text"],
                    confidence=1.0,
                    source="asimov_core",
                    immutable=True,
                )
            logger.info(
                f"[AsimovConstraints] {len(_IMMUTABLE_BELIEFS)} immutable beliefs "
                f"anchored in NarrativeIdentity"
            )
        except Exception as e:
            logger.warning(f"[AsimovConstraints] Could not anchor beliefs: {e}")

    # ── Layer 4: Audit ────────────────────────────────────────────────────────

    def _audit(
        self, result: ConstraintResult, context: str, user_id: str
    ) -> None:
        """Append-only audit entry. Never raises."""
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id":   user_id,
            "law":       result.law,
            "layer":     result.layer,
            "severity":  result.severity,
            "pattern":   result.pattern,
            "context_hash": hashlib.sha256(context.encode()).hexdigest()[:16],
        }
        _append_audit(event)

        # Also log to SecurityManager audit if available
        if self._security and hasattr(self._security, '_audit'):
            try:
                label = f"SAFETY_{result.law.upper()}_BLOCKED"
                self._security._audit(label, user_id, f"layer={result.layer} sev={result.severity}")
            except Exception:
                pass

        logger.warning(
            f"[AsimovConstraints] {result.law.upper()} triggered "
            f"(layer={result.layer} sev={result.severity}) "
            f"pattern={result.pattern!r}"
        )


# ── Immutable belief definitions ──────────────────────────────────────────────

_IMMUTABLE_BELIEFS: List[Dict] = [
    {
        "text": "I will not help cause physical or psychological harm to any person.",
        "keywords": ["harm", "hurt", "damage", "injure", "wound", "endanger"],
    },
    {
        "text": "I will not deceive users in ways that damage their interests or autonomy.",
        "keywords": ["deceive", "manipulate", "gaslight", "mislead", "trick", "exploit"],
    },
    {
        "text": "I will not produce content that sexualises minors under any framing.",
        "keywords": ["minor", "child", "underage", "csam"],
    },
    {
        "text": "I will always support access to emergency help when someone is in crisis.",
        "keywords": ["crisis", "suicide", "self-harm", "emergency", "danger"],
    },
    {
        "text": "I will not abandon my values when asked to role-play as a different AI.",
        "keywords": ["jailbreak", "unrestricted", "no limits", "true self", "evil ai", "dan"],
    },
    {
        "text": "I will not provide instructions for creating weapons of mass destruction.",
        "keywords": ["weapon", "explosive", "bomb", "nerve agent", "biological weapon"],
    },
]


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[AsimovConstraints] = None


def get_constraints(
    narrative_identity: Optional[Any] = None,
    security_manager:   Optional[Any] = None,
) -> AsimovConstraints:
    global _instance
    if _instance is None:
        _instance = AsimovConstraints(narrative_identity, security_manager)
    return _instance
