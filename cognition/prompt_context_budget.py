"""
cognition/prompt_context_budget.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PromptContextBudget — token-budgeted prompt assembly with influence tracking.

Problem it solves
──────────────────
After the audit fixes, _build_prompt_additions() now assembles up to 15+
sections from different cognitive modules. Without a budget:

    memory + beliefs + narrative + skills + audit + immune + preferences
    + self_state + emotion + curiosity + pressure + ...

    → prompt grows unboundedly
    → signal-to-noise ratio falls
    → LLM context window fills with repeated state
    → PandoraBOX appears to use the "same state" regardless of input

And even with wiring fixed, there's no way to know which sections
actually entered the prompt vs were silently too long.

What this module does
──────────────────────
1. TOKEN BUDGET — each section category has a fixed token cap.
   Sections are assembled in priority order; when the budget for a
   category is exhausted, lower-priority items in that category drop.

2. CONTRIBUTION RECORD — every section that enters or is dropped is
   logged with: source, tokens, selected, reason.
   Persisted to data/persona/prompt_contributions.json (rolling 200).

3. INFLUENCE DASHBOARD — contribution_summary() returns per-source
   stats: mean tokens, selection rate, last_selected_at.
   Readable by brain.py /state endpoint and the audit engine.

Token budget (conservative — real tokenisation is ~4 chars/token):

    Category            Budget (tokens)   Priority
    ─────────────────────────────────────────────
    self_state          120               1  (φ, coherence, ground)
    identity            200               2  (narrative arc, values)
    cognitive_pressure  100               3  (tensions, energy, homeostasis)
    attention           80                4  (attention + curiosity)
    memory_active       150               5  (inner monologue, working memory)
    skills              120               6  (relevant competencies)
    preferences         80                7  (stable orientations)
    experimentation     100               8  (active experiment)
    meta_systems        120               9  (audit, immune, ripple, ethics)
    ─────────────────────────────────────────────
    TOTAL HARD CAP      900               (≈ 3600 chars / ~720 words)

Usage (in _build_prompt_additions):
    budget = PromptContextBudget()
    budget.add("identity", "narrative_identity", ni_frag)
    budget.add("skills",   "skill_registry",     skill_frag)
    ...
    final_text = budget.assemble()
    budget.save()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Token budget per category ──────────────────────────────────────────────────
# Approximate: 4 characters ≈ 1 token
CHARS_PER_TOKEN = 4

CATEGORY_BUDGETS: Dict[str, Tuple[int, int]] = {
    # category → (token_budget, priority)  lower priority = higher priority number
    "factual_telemetry":  (320, 0),
    "self_state":         (120, 1),
    "identity":           (200, 2),
    "cognitive_pressure": (100, 3),
    "attention":          (80,  4),
    "memory_active":      (150, 5),
    "skills":             (120, 6),
    "preferences":        (80,  7),
    "experimentation":    (100, 8),
    "meta_systems":       (120, 9),
}

TOTAL_TOKEN_CAP = 900   # hard ceiling across all categories

MAX_CONTRIBUTION_LOG = 200


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SectionEntry:
    """One candidate section before budget decision."""
    category:  str
    source:    str        # e.g. "narrative_identity", "skill_registry"
    text:      str
    tokens:    int        # estimated
    selected:  bool = False
    reason:    str  = ""  # "budget_ok" | "category_full" | "total_cap" | "empty"


@dataclass
class ContributionRecord:
    """One prompt assembly's full record."""
    timestamp:    float
    total_tokens: int
    sections:     List[Dict]  # serialised SectionEntries


# ── Budget ────────────────────────────────────────────────────────────────────

class PromptContextBudget:
    """
    Assembles prompt sections respecting per-category and global token budgets.
    Records every decision for observability.

    Usage:
        b = PromptContextBudget()
        b.add("identity", "narrative_identity", text)
        b.add("skills",   "skill_registry",     text)
        result = b.assemble()   # priority-ordered, budget-capped string
        b.save()                # persist contribution record
    """

    def __init__(self, path: str = "data/persona/prompt_contributions.json") -> None:
        self._path     = Path(path)
        self._entries:  List[SectionEntry] = []
        self._cat_used: Dict[str, int]     = {cat: 0 for cat in CATEGORY_BUDGETS}
        self._total_used = 0
        self._assembled  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, category: str, source: str, text: str) -> None:
        """Register a candidate section. Call before assemble()."""
        if not text or not text.strip():
            self._entries.append(SectionEntry(
                category=category, source=source, text="",
                tokens=0, selected=False, reason="empty",
            ))
            return

        tokens = max(1, len(text) // CHARS_PER_TOKEN)
        self._entries.append(SectionEntry(
            category=category, source=source,
            text=text.strip(), tokens=tokens,
        ))

    def assemble(self) -> str:
        """
        Apply budget rules in priority order.
        Returns the assembled prompt string.
        Must be called once; subsequent calls return the same result.
        """
        if self._assembled:
            return self._get_selected_text()

        # Sort entries: by category priority, then registration order
        priority_map = {cat: p for cat, (_, p) in CATEGORY_BUDGETS.items()}
        def sort_key(e: SectionEntry) -> Tuple[int, int]:
            return (priority_map.get(e.category, 99), self._entries.index(e))

        ordered = sorted(self._entries, key=sort_key)

        for entry in ordered:
            if entry.reason == "empty":
                continue

            cat_budget = CATEGORY_BUDGETS.get(entry.category, (80, 99))[0]
            cat_used   = self._cat_used.get(entry.category, 0)

            if self._total_used + entry.tokens > TOTAL_TOKEN_CAP:
                entry.selected = False
                entry.reason   = "total_cap"
            elif cat_used + entry.tokens > cat_budget:
                entry.selected = False
                entry.reason   = "category_full"
            else:
                entry.selected = True
                entry.reason   = "budget_ok"
                self._cat_used[entry.category] = cat_used + entry.tokens
                self._total_used += entry.tokens

        self._assembled = True
        return self._get_selected_text()

    def save(self) -> None:
        """Append this assembly's record to the rolling contribution log."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            existing: List[Dict] = []
            try:
                if self._path.exists():
                    existing = json.loads(self._path.read_text())
            except Exception:
                existing = []

            record = ContributionRecord(
                timestamp    = time.time(),
                total_tokens = self._total_used,
                sections     = [asdict(e) for e in self._entries],
            )
            existing.append(asdict(record))
            if len(existing) > MAX_CONTRIBUTION_LOG:
                existing = existing[-MAX_CONTRIBUTION_LOG:]
            self._path.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    def contribution_summary(self) -> Dict[str, Dict]:
        """
        Returns per-source selection stats from the persisted log.
        Usable by /state endpoint and CognitiveAuditEngine.

        Returns:
            {
                "narrative_identity": {
                    "appearances": 12,
                    "selected": 10,
                    "selection_rate": 0.83,
                    "mean_tokens": 42.1,
                    "last_selected_at": 1716000000.0,
                },
                ...
            }
        """
        try:
            if not self._path.exists():
                return {}
            records = json.loads(self._path.read_text())
            stats: Dict[str, Dict] = {}
            for rec in records:
                for sec in rec.get("sections", []):
                    src = sec.get("source", "unknown")
                    if src not in stats:
                        stats[src] = {
                            "appearances": 0, "selected": 0,
                            "total_tokens": 0, "last_selected_at": 0.0,
                        }
                    stats[src]["appearances"] += 1
                    if sec.get("selected"):
                        stats[src]["selected"] += 1
                        stats[src]["total_tokens"] += sec.get("tokens", 0)
                        stats[src]["last_selected_at"] = max(
                            stats[src]["last_selected_at"],
                            rec.get("timestamp", 0.0),
                        )
            # Compute derived fields
            for src, s in stats.items():
                n = s["appearances"]
                sel = s["selected"]
                s["selection_rate"] = round(sel / n, 3) if n else 0.0
                s["mean_tokens"]    = round(s["total_tokens"] / sel, 1) if sel else 0.0
            return stats
        except Exception:
            return {}

    def budget_report(self) -> str:
        """
        Human-readable budget usage for logging.
        Example:
            [PromptBudget] 312/900 tokens | selected=8/14
            identity=180 skills=80 meta_systems=52 ...
        """
        if not self._assembled:
            return "[PromptBudget] not assembled yet"
        selected = sum(1 for e in self._entries if e.selected)
        total    = len([e for e in self._entries if e.reason != "empty"])
        cats     = {cat: used for cat, used in self._cat_used.items() if used > 0}
        cat_str  = " ".join(f"{c}={v}" for c, v in sorted(cats.items()))
        return (
            f"[PromptBudget] {self._total_used}/{TOTAL_TOKEN_CAP} tokens "
            f"| selected={selected}/{total} | {cat_str}"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_selected_text(self) -> str:
        # Return sections in original registration order (not priority order)
        # so the prompt reads naturally
        return "\n\n".join(
            e.text for e in self._entries if e.selected
        )
