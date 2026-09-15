"""
Research MCP — Autonomous Research Plugin for Lumina.

Three activation modes:

  Mode 1 — ON_DEMAND   : explicit call, bounded loop, no memory write
  Mode 2 — EXTENDED    : auto-triggered for complex queries, richer answer
  Mode 3 — BACKGROUND  : async overnight, summarised + stored in journal & memory

Public surface (used by persona_bridge):

    from cognition.research_mcp import ResearchMCP
    mcp = ResearchMCP(llm_fn=..., memory_system=...)

    # Mode 1 — explicit
    result = mcp.research(goal="...", mode=1, depth=2)

    # Mode 2 — called automatically by persona_bridge
    result = mcp.research(goal="...", mode=2, depth=3)

    # Mode 3 — background, run in thread
    result = mcp.research(goal="...", mode=3, depth=4, trigger_reason="deep_day")

    # RAC — record interaction signals
    mcp.rac.record_query(user_input, low_confidence=False)
    mcp.rac.should_extend_answer(user_input) → bool

    # Journal
    mcp.journal()                    → list of entries
    mcp.journal_entry(research_id)   → dict or None
    mcp.journal_by_topic(topic)      → list
    mcp.new_research_since(ts)       → list

    # Background scheduler
    mcp.pop_pending_background()     → (goal, trigger_reason) or None
"""
from cognition.research_mcp.schemas      import (
    ResearchMode, ResearchSession, ResearchResult, DailyMetrics
)
from cognition.research_mcp.controller   import ResearchController
from cognition.research_mcp.rac          import ResearchActivationController
from cognition.research_mcp.memory_integrator import MemoryIntegrator

import logging
from typing import Callable, Optional, Any, Tuple, List

logger = logging.getLogger("research_mcp")


class ResearchMCP:
    """
    Top-level facade.
    Instantiate once in persona_bridge and share.
    """

    def __init__(
        self,
        llm_fn:        Callable[[str, str], str],
        memory_system: Optional[Any] = None,
    ):
        self._controller = ResearchController(llm_fn, memory_system)
        self._integrator = MemoryIntegrator(memory_system)
        self.rac          = ResearchActivationController()

    # ── Core research entry point ─────────────────────────────────────────────
    def research(
        self,
        goal:           str,
        mode:           int  = 1,
        depth:          int  = 3,
        max_iterations: int  = 8,
        trigger_reason: str  = "",
    ) -> ResearchResult:
        """
        Run a research session.

        mode: 1=ON_DEMAND, 2=EXTENDED, 3=BACKGROUND
        depth: 1–5 (breadth of sub-goal decomposition)
        max_iterations: hard cap on search loop
        """
        try:
            rmode = ResearchMode(mode)
        except ValueError:
            rmode = ResearchMode.ON_DEMAND

        # apply safety cap
        max_iterations = min(max(1, max_iterations), 12)
        depth          = min(max(1, depth), 5)

        session = ResearchSession(
            mode            = rmode,
            goal            = goal,
            depth           = depth,
            max_iterations  = max_iterations,
            trigger_reason  = trigger_reason,
        )
        return self._controller.run(session)

    # ── Journal access ────────────────────────────────────────────────────────
    def journal(self, limit: int = 20) -> List[dict]:
        return self._integrator.get_journal(limit=limit)

    def journal_entry(self, research_id: str) -> Optional[dict]:
        return self._integrator.get_entry(research_id)

    def journal_by_topic(self, topic: str, limit: int = 10) -> List[dict]:
        return self._integrator.get_by_topic(topic, limit=limit)

    def new_research_since(self, iso_timestamp: str) -> List[dict]:
        return self._integrator.has_new_research_since(iso_timestamp)

    # ── Background scheduler hook ─────────────────────────────────────────────
    def pop_pending_background(self) -> Optional[Tuple[str, str]]:
        """Returns (goal, trigger_reason) or None. Run Mode 3 in a thread."""
        return self.rac.pop_pending_mode3()

    def mark_background_ran(self):
        self.rac.mark_mode3_ran()
