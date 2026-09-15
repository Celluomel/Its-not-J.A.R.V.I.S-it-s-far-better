"""
Planner — decomposes the user's research goal into ordered sub-goals.

Calls the LLM once per planning step.
All LLM calls use the injected `llm_fn` (generate_bare — no history pollution).
"""
from __future__ import annotations
import json
import logging
import re
import uuid
from typing import List, Callable

from cognition.research_mcp.schemas import SubGoal

logger = logging.getLogger("research_mcp.planner")


_DECOMPOSE_SYSTEM = """You are the planner module of an autonomous research system.
Given a research goal and a depth level, return a JSON array of sub-goals.
Each sub-goal is a focused, searchable question that together cover the main goal.
Return ONLY a JSON array — no markdown, no preamble.

Format:
[
  {"description": "...", "priority": 1},
  {"description": "...", "priority": 2}
]

Rules:
- Priority 1 = most essential, higher numbers = supplementary
- depth 1 → 2 sub-goals, depth 2 → 3, depth 3 → 4, depth 4 → 5, depth 5 → 6
- Each sub-goal must be independently searchable (concrete, specific)
- Avoid overlap between sub-goals
- Focus on verifiable, current information
"""

_NEXT_STEP_SYSTEM = """You are the research loop controller.
Given a goal, completed sub-goals with findings, and remaining sub-goals,
return the BEST next search query to advance the research.
Return ONLY the query string — no explanation, no quotes.
Make it specific and searchable (3–8 words ideal).
"""


class Planner:
    """
    Decomposes goals into sub-goals and selects the next search query.
    Stateless — takes all context as arguments.
    """

    def __init__(self, llm_fn: Callable[[str, str], str]):
        """
        llm_fn: callable(prompt: str, system: str) → str
        This should be generate_bare or equivalent (no history side-effects).
        """
        self._llm = llm_fn

    def decompose(self, goal: str, depth: int = 3) -> List[SubGoal]:
        """Break the goal into sub-goals via LLM. Returns sorted by priority."""
        depth = max(1, min(5, depth))
        prompt = f"Research goal: {goal}\nDepth level: {depth}\nDecompose into sub-goals:"

        try:
            raw = self._llm(prompt, _DECOMPOSE_SYSTEM)
            # extract JSON array even if LLM added surrounding text
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON array found in planner output")
            items = json.loads(match.group())
            sub_goals = []
            for item in items:
                sg = SubGoal(
                    goal_id     = uuid.uuid4().hex[:8],
                    description = item.get("description", ""),
                    priority    = int(item.get("priority", 99)),
                )
                if sg.description:
                    sub_goals.append(sg)
            sub_goals.sort(key=lambda x: x.priority)
            logger.info(f"Planner: decomposed into {len(sub_goals)} sub-goals")
            return sub_goals
        except Exception as e:
            logger.error(f"Planner.decompose failed: {e} — using goal as single sub-goal")
            return [SubGoal(goal_id=uuid.uuid4().hex[:8], description=goal, priority=1)]

    def next_query(
        self,
        goal: str,
        completed: List[SubGoal],
        remaining: List[SubGoal],
    ) -> str:
        """
        Given what's been done and what remains, return the best search query string.
        Falls back to the next sub-goal description if LLM fails.
        """
        if not remaining:
            return ""

        completed_summary = "\n".join(
            f"- {sg.description}: {sg.findings[:200]}" for sg in completed if sg.completed
        ) or "None yet."

        remaining_list = "\n".join(f"- {sg.description}" for sg in remaining)

        prompt = (
            f"Main goal: {goal}\n\n"
            f"Completed sub-goals:\n{completed_summary}\n\n"
            f"Remaining sub-goals:\n{remaining_list}\n\n"
            f"What is the best search query to run next?"
        )

        try:
            query = self._llm(prompt, _NEXT_STEP_SYSTEM).strip().strip('"').strip("'")
            if query:
                return query
        except Exception as e:
            logger.warning(f"Planner.next_query LLM failed: {e}")

        # fallback: use next sub-goal description directly
        return remaining[0].description
