"""
Speech Sanitizer – Layer 8 of the embodied agent architecture (Section 7).

Rules (from spec):
  • Remove markdown characters (*, #, -, _)
  • Limit output to two sentences maximum
  • Remove bullet formatting
  • Remove excessive punctuation
  • Ensure conversational tone
"""
from __future__ import annotations

import re


# Characters and patterns to strip
_MARKDOWN_CHARS = re.compile(r"[*#_`]")
_BULLET_LINE    = re.compile(r"^\s*[-•·▸▶]\s+", re.MULTILINE)
_NUMBERED_ITEM  = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_EXCESS_PUNCT   = re.compile(r"[!?]{2,}")           # !! → !
_MULTIPLE_DOTS  = re.compile(r"\.{4,}")             # .... → ...
_MULTI_NEWLINE  = re.compile(r"\n{2,}")


def _sentence_split(text: str) -> list[str]:
    """
    Split into sentences on '. ', '! ', '? ' boundaries.
    Avoids splitting on abbreviations like "Mr." or decimal numbers.
    """
    # Protect common abbreviations
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|etc|vs|e\.g|i\.e)\.",
                       lambda m: m.group().replace(".", "\x00"), text)
    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', protected)
    # Restore protected dots
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def sanitize(text: str, max_sentences: int = 2) -> str:
    """
    Apply all speech sanitization rules and return TTS-ready text.

    Parameters
    ----------
    text          : raw LLM output
    max_sentences : hard cap on output sentences (default 2, per spec)
    """
    if not text:
        return ""

    # 1. Remove markdown characters
    text = _MARKDOWN_CHARS.sub("", text)

    # 2. Strip bullet / numbered-list formatting (flatten to prose)
    text = _BULLET_LINE.sub("", text)
    text = _NUMBERED_ITEM.sub("", text)

    # 3. Collapse multiple newlines → single space
    text = _MULTI_NEWLINE.sub(" ", text)
    text = text.replace("\n", " ")

    # 4. Clean excessive punctuation
    text = _EXCESS_PUNCT.sub(lambda m: m.group()[0], text)  # !! → !
    text = _MULTIPLE_DOTS.sub("...", text)

    # 5. Trim whitespace
    text = " ".join(text.split())

    # 6. Enforce sentence limit
    sentences = _sentence_split(text)
    if len(sentences) > max_sentences:
        text = " ".join(sentences[:max_sentences])
        # Ensure it ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."

    return text.strip()
