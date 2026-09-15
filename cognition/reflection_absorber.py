"""
ReflectionAbsorber — Phase D
===============================
Implements the original spec's regex approach directly, per explicit
request — this is the piece flagged in review as fragile ("I notice
that I need coffee" is not a self-insight) and skipped in v107 in favor
of structured-only absorption (symbol_system.py's
absorb_narrative_chapter/absorb_self_inquiry, which stay in place
unchanged). This module adds the free-text path ALONGSIDE those, with
three concrete quality filters the original spec didn't have, aimed
directly at the failure mode identified:

  1. Minimum meaningful length after trimming — rejects fragments like
     "doing" or "here" that a bare regex match can produce.
  2. Stopword-density rejection — if most of the matched words are
     common function words, it's very unlikely to be a substantive
     self-claim ("that i am" matches "i seem to be" trivially but
     carries no content).
  3. A per-call cap (default 3) on how many claims one reflection text
     can contribute — bounds how fast a single verbose AutonomousReflection
     output can flood the symbol system, regardless of how many pattern
     matches it contains.

Absorbed symbols get a source_module tag of "reflection_absorber"
(distinct from "narrative_arc_writer"/"cognitive_dissonance_engine" used
by the structured path in v107), so the two absorption sources stay
distinguishable in the log and in SymbolSystem.symbols[*].source_module.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SELF_REF_PATTERNS = [
    r"i notice that i ([^.!?]{4,150})",
    r"my (?:own )?way of ([^.!?]{4,150})",
    r"the way i (?:monitor|evaluate|predict|revise) ([^.!?]{4,150})",
    r"i seem to be ([^.!?]{4,150})",
    r"this pattern of ([^.!?]{4,150})",
]

MIN_CLAIM_WORDS = 4         # filter 1 — reject too-short fragments
MAX_STOPWORD_RATIO = 0.7    # filter 2 — reject mostly-function-word matches
MAX_CLAIMS_PER_CALL = 3     # filter 3 — bound flood risk from one text

_STOPWORDS = frozenset({
    "the", "a", "an", "i", "am", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "it", "that", "this", "and", "or", "but",
    "my", "me", "so", "just", "very", "really", "here", "there", "with",
})


def _stopword_ratio(text: str) -> float:
    words = [w.lower().strip(".,!?;:\"'") for w in text.split() if w]
    if not words:
        return 1.0
    stop_count = sum(1 for w in words if w in _STOPWORDS)
    return stop_count / len(words)


def extract_self_claims(text: str) -> List[Dict]:
    claims = []
    lower = text.lower()
    for pat in SELF_REF_PATTERNS:
        for m in re.finditer(pat, lower):
            captured = m.group(1) if m.groups() else m.group(0)
            captured = captured.strip()
            if len(captured.split()) < MIN_CLAIM_WORDS:
                continue  # filter 1
            if _stopword_ratio(captured) > MAX_STOPWORD_RATIO:
                continue  # filter 2
            claims.append({
                "short_name": m.group(0)[:40].replace(" ", "_"),
                "text": m.group(0).strip(),
                # Was 0.45 — found via the Phase D success-criterion check
                # (an absorbed symbol must be able to influence a
                # subsequent decision) to sit BELOW symbol_system.py's own
                # MIN_CONFIDENCE_FOR_CANDIDATE=0.5 threshold, and
                # create_or_update()'s averaging never raises an
                # unchanged value — meaning every reflection-absorbed
                # symbol was permanently invisible to WorkspaceCompetition,
                # not merely low-priority. 0.55 clears that bar while
                # staying below structured absorption's typical 0.6-0.9
                # range (free-text extraction is a weaker signal than a
                # real narrative-chapter significance score).
                "confidence": 0.55,
            })
            if len(claims) >= MAX_CLAIMS_PER_CALL:
                return claims  # filter 3
    return claims


def absorb_reflection(text: str, symbol_system: Any, source: str = "reflection_absorber") -> int:
    """Returns the number of symbols actually created/updated."""
    if not text:
        return 0
    count = 0
    try:
        for claim in extract_self_claims(text):
            symbol_system.create_or_update(
                name=claim["short_name"], definition=claim["text"],
                symbol_type="meta", source_module=source,
                confidence=claim["confidence"],
            )
            count += 1
    except Exception as e:
        logger.debug(f"[ReflectionAbsorber] absorb failed (non-fatal): {e}")
    return count
