"""
cognition/temporal_anchor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TemporalAnchor — real-world clock grounding for LLM prompts.

Why this exists
───────────────
The model's weights encode the world only up to its training cutoff.
Without an explicit, authoritative date in the prompt, the model answers
"What day is today?" with whatever date its training distribution favoured
(e.g. 2023) and treats "the news today" as old references.

TemporalWeave (cognition/temporal_weave.py) is NOT a substitute: it weaves
the *specious present* — the felt flow of experience across turns — and says
nothing about the calendar. This module is the missing anchor: a single,
canonical, always-fresh statement of "when is now" that enters the top of
the user-facing system prompt.

Design
──────
- One helper, zero dependencies:  temporal_anchor_block(now=None) -> str
- Cheap: one datetime.now() call, ~70 tokens.
- Deterministic for a given `now` (unit-testable).
- Fails soft: on ANY error returns "" — chat must never break because of
  a grounding section.
"""

from __future__ import annotations

from datetime import datetime


def _tz_label() -> str:
    """Best-effort local timezone offset label, e.g. 'UTC+02:00'."""
    try:
        off = datetime.now().astimezone().utcoffset()
        if off is None:
            return "local time"
        total_min = int(off.total_seconds() // 60)
        sign = "+" if total_min >= 0 else "-"
        total_min = abs(total_min)
        return f"UTC{sign}{total_min // 60:02d}:{total_min % 60:02d}"
    except Exception:
        return "local time"


def temporal_anchor_block(now: datetime | None = None) -> str:
    """
    Canonical TEMPORAL ANCHOR section for the system prompt.

    Parameters
    ----------
    now : datetime, optional
        Pin the "current" moment (used by tests). Defaults to datetime.now().

    Returns
    -------
    str
        The anchor block, or "" on any failure (soft-fail by contract).
    """
    try:
        now = now or datetime.now()
        date_line = now.strftime("%A, %B %d, %Y")
        time_line = now.strftime("%H:%M")
        return (
            "━━ TEMPORAL ANCHOR (authoritative) ━━\n"
            f"Current date: {date_line}. Current time: {time_line} ({_tz_label()}).\n"
            "Your training data ends before this date — treat anything learned after your cutoff as unknown.\n"
            "If asked about today, the current date, or 'recent'/'current' events: use THIS date as the reference.\n"
            "Never infer the date from learned patterns, from memories, or from anything said earlier in the conversation.\n"
            "For real-time information (news, prices, releases): say plainly that your knowledge is limited to your training cutoff — do not present old facts as current."
        )
    except Exception:
        return ""


def temporal_date_line(now: datetime | None = None) -> str:
    """
    Compact one-line date fact — for internal prompts (dreams, life events)
    whose output is stored in memory and resurfaces in future context.
    Dreams that reference a wrong "today" would contaminate future prompts;
    this gives them the correct anchor.
    """
    try:
        now = now or datetime.now()
        return f"Current date: {now.strftime('%A, %B %d, %Y')}."
    except Exception:
        return ""
