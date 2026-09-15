"""
ResearchController — the main loop engine for all 3 research modes.

Architecture:
  caller (persona_bridge / background scheduler)
      ↓
  ResearchController.run(session)
      ↓
  [Planner → WebAgent → Evaluator → Synthesizer → MemoryIntegrator]
      ↓
  ResearchResult

Safety controls:
  - max_iterations hard cap (per session)
  - WebAgent.MAX_PAGES_PER_SESSION cap
  - loop detection via Evaluator
  - timeout-aware (WebAgent handles per-page timeout)
  - memory write only after summarization
"""
from __future__ import annotations
import datetime
import logging
import uuid
from typing import Callable, Optional, Any

from cognition.research_mcp.schemas import (
    ResearchMode, ResearchSession, ResearchStatus,
    ResearchStep, ResearchResult,
)
from cognition.research_mcp.planner          import Planner
from cognition.research_mcp.web_agent        import WebAgent
from cognition.research_mcp.evaluator        import Evaluator
from cognition.research_mcp.synthesizer      import Synthesizer
from cognition.research_mcp.memory_integrator import MemoryIntegrator

logger = logging.getLogger("research_mcp.controller")


class ResearchController:

    def __init__(
        self,
        llm_fn:        Callable[[str, str], str],
        memory_system: Optional[Any] = None,
    ):
        """
        llm_fn: bare LLM callable(prompt, system) → str
                Use generate_bare from LLMManager (no history pollution).
        memory_system: EnhancedMemorySystem instance (optional, needed for Mode 3).
        """
        self._planner    = Planner(llm_fn)
        self._evaluator  = Evaluator(llm_fn)
        self._synthesizer= Synthesizer(llm_fn)
        self._integrator = MemoryIntegrator(memory_system)

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, session: ResearchSession) -> ResearchResult:
        """
        Execute a full research session.
        Blocking — run in a thread (asyncio.to_thread) from the UI layer.
        """
        session.started_at = datetime.datetime.now().isoformat()
        session.status     = ResearchStatus.RUNNING
        logger.info(
            f"ResearchController.run — id={session.research_id} "
            f"mode={session.mode.name} goal={session.goal!r}"
        )

        web = WebAgent()

        try:
            # ── 1. Decompose goal into sub-goals ─────────────────────────────
            depth     = self._mode_depth(session)
            sub_goals = self._planner.decompose(session.goal, depth=depth)
            session.sub_goals = sub_goals
            remaining = list(sub_goals)
            completed = []

            iteration = 0

            # ── 2. Research loop ──────────────────────────────────────────────
            while iteration < session.max_iterations and remaining:
                iteration += 1
                logger.info(f"  iteration {iteration}/{session.max_iterations}")

                # pick next search query
                query = self._planner.next_query(session.goal, completed, remaining)
                if not query:
                    logger.info("  no more queries — exiting loop")
                    break

                # fetch pages
                pages = web.search_and_fetch(
                    query,
                    max_pages=self._pages_per_step(session.mode),
                )

                # extract relevant content from each page
                extracted_parts = []
                for page in pages:
                    extracted = self._evaluator.extract(session.goal, page)
                    if extracted:
                        extracted_parts.append(extracted)

                step = ResearchStep(
                    step_id   = uuid.uuid4().hex[:8],
                    query     = query,
                    pages     = pages,
                    extracted = "\n\n".join(extracted_parts),
                )
                session.steps.append(step)

                # mark next sub-goal as complete
                if remaining:
                    sg           = remaining.pop(0)
                    sg.completed = True
                    sg.findings  = step.extracted[:500]
                    completed.append(sg)

                # evaluate confidence + loop detection
                confidence, done = self._evaluator.assess(session)
                session.confidence_score = confidence

                if done:
                    logger.info(f"  goal satisfied at iteration {iteration} (confidence={confidence:.2f})")
                    break

                if self._evaluator.detect_loop(session):
                    logger.warning(f"  loop detected at iteration {iteration} — aborting")
                    session.status = ResearchStatus.ABORTED
                    break

            # ── 3. Synthesize ─────────────────────────────────────────────────
            summary = self._synthesizer.synthesize(session)
            session.sources  = self._synthesizer.build_sources(session)
            session.finished_at = datetime.datetime.now().isoformat()

            if session.status == ResearchStatus.RUNNING:
                session.status = ResearchStatus.COMPLETE

            # ── RSS feed outcome tracking ────────────────────────────────────
            # Attribute this session's real confidence_score (from
            # Evaluator.assess() above — not a new metric) back to whichever
            # RSS feeds contributed pages, so the management page's success
            # rate reflects actual research outcomes.
            try:
                from cognition.research_mcp.rss_registry import get_rss_registry
                sources_used = [
                    p.source for step in session.steps for p in step.pages if p.source
                ]
                if sources_used:
                    get_rss_registry().record_session_outcome(
                        sources_used, session.confidence_score
                    )
            except Exception as _rss_e:
                logger.debug(f"RSS outcome tracking failed (non-fatal): {_rss_e}")

            # ── 4. Memory integration (Mode 3 only) ──────────────────────────
            nodes = 0
            if session.mode == ResearchMode.BACKGROUND:
                nodes = self._integrator.integrate(session)
                session.knowledge_nodes = nodes

            logger.info(
                f"ResearchController done — {iteration} iterations, "
                f"confidence={session.confidence_score:.2f}, nodes={nodes}"
            )

            journal_entry = None
            if session.mode == ResearchMode.BACKGROUND:
                journal_entry = self._integrator.get_entry(session.research_id)

            return ResearchResult(
                research_id     = session.research_id,
                mode            = session.mode.value,
                goal            = session.goal,
                summary         = session.summary,
                sources         = session.sources,
                confidence      = session.confidence_score,
                status          = session.status.value,
                knowledge_nodes = session.knowledge_nodes,
                journal_entry   = journal_entry,
            )

        except Exception as e:
            logger.error(f"ResearchController critical error: {e}", exc_info=True)
            session.status     = ResearchStatus.FAILED
            session.finished_at= datetime.datetime.now().isoformat()
            return ResearchResult(
                research_id = session.research_id,
                mode        = session.mode.value,
                goal        = session.goal,
                summary     = "",
                sources     = [],
                confidence  = 0.0,
                status      = ResearchStatus.FAILED.value,
                error       = str(e),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _mode_depth(session: ResearchSession) -> int:
        """Depth controls sub-goal breadth. Mode 1 keeps it shallow."""
        if session.mode == ResearchMode.ON_DEMAND:
            return min(session.depth, 2)
        if session.mode == ResearchMode.EXTENDED:
            return min(session.depth, 4)
        return session.depth  # BACKGROUND: full depth

    @staticmethod
    def _pages_per_step(mode: ResearchMode) -> int:
        """How many pages to fetch per search step."""
        return {
            ResearchMode.ON_DEMAND: 2,
            ResearchMode.EXTENDED:  3,
            ResearchMode.BACKGROUND: 3,
        }.get(mode, 2)
