"""Universal temporal and causal frame for conversational reasoning.

This is deliberately deterministic and inexpensive. It does not decide the
answer; it constrains generated reasoning to the real order of events.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class TemporalFrame:
    now: datetime
    horizon: Optional[timedelta] = None
    horizon_text: str = ""

    @property
    def deadline(self) -> Optional[datetime]:
        return self.now + self.horizon if self.horizon is not None else None

    def prompt_fragment(self) -> str:
        now = self.now.strftime("%Y-%m-%d %H:%M %Z")
        deadline = self.deadline.strftime("%Y-%m-%d %H:%M %Z") if self.deadline else "not established"
        horizon = self.horizon_text or "no explicit relative horizon detected"
        return "\n".join([
            "━━ UNIVERSAL TEMPORAL / CAUSAL FRAME ━━",
            f"Present reference t0: {now}",
            f"Stated elapsed-time constraint: {horizon}",
            f"Computed future boundary, if applicable: {deadline}",
            "Temporal rules:",
            "1. Treat t0 as the boundary between completed/present facts and future possibilities.",
            "2. A future objective must be reached by actions that occur after t0 and before its deadline.",
            "3. Elapsed time is a constraint: available time = deadline - t0, not a second arrival time.",
            "4. Causes precede effects; prerequisites must occur before the objective, never after it.",
            "5. Do not repair a missed deadline by assuming time travel, retroactive action, or an outcome already achieved.",
            "6. Separate known timestamps from estimates. If a duration, route, start time, or deadline is missing, say so.",
            "7. Test feasibility against physical, logistical, and human limits before recommending an action.",
            "Use this frame silently. Do not expose the label 'Temporal Frame' or internal rule numbering.",
        ])


def _duration(text: str) -> tuple[Optional[timedelta], str]:
    match = re.search(
        r"\b(?:dans|in)\s+(une|un|an|one|\d+)\s*"
        r"(heure|heures|hour|hours|h|minute|minutes)\b",
        text.casefold(),
    )
    if not match:
        return None, ""
    amount = 1 if match.group(1) in {"une", "un", "an", "one"} else int(match.group(1))
    unit = match.group(2)
    minutes = amount * 60 if unit.startswith(("heure", "hour", "h")) else amount
    label = f"{amount} hour(s)" if minutes % 60 == 0 else f"{minutes} minute(s)"
    return timedelta(minutes=minutes), label


def build_temporal_frame(text: str, now: Optional[datetime] = None) -> TemporalFrame:
    """Build a present-anchored frame for any user turn."""
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    horizon, label = _duration(str(text or ""))
    return TemporalFrame(current, horizon, label)

