"""
Evaluator — decides if the research goal is satisfied, detects runaway loops,
and extracts structured findings from raw page content.

All LLM calls use generate_bare (no history side-effects).
"""
from __future__ import annotations
import logging
import re
from typing import List, Callable, Tuple

from cognition.research_mcp.schemas import ResearchSession, SubGoal, WebPage

logger = logging.getLogger("research_mcp.evaluator")


_EXTRACT_SYSTEM = """You are a research extraction assistant.
Given raw web page content, extract the most relevant information for the research goal.
Return a clean, dense summary (2-5 sentences) focused on facts relevant to the goal.
If the page is irrelevant or empty, return: [IRRELEVANT]
No markdown, no preamble — just the extracted text or [IRRELEVANT].
"""

_CONFIDENCE_SYSTEM = """You are a research quality evaluator.
Given a research goal and collected findings so far, assess how well the goal is satisfied.
Return a JSON object: {"confidence": <float 0.0-1.0>, "reason": "<brief>", "done": <bool>}
- confidence: 0.0 = nothing useful found, 1.0 = goal fully covered
- done: true if confidence >= 0.75 OR you believe further searching won't help
Return ONLY the JSON object.
"""

_LOOP_DETECT_SYSTEM = """You are a loop detection module.
Given a list of recent search queries, determine if the research is looping
(repeating similar searches without progress).
Return: {"looping": <bool>, "reason": "<brief>"}
Return ONLY the JSON object.
"""


class Evaluator:

    def __init__(self, llm_fn: Callable[[str, str], str], confidence_threshold: float = 0.75):
        self._llm       = llm_fn
        self._threshold = confidence_threshold

    # ── Page extraction ───────────────────────────────────────────────────────
    def extract(self, goal: str, page: WebPage) -> str:
        """Extract goal-relevant content from a web page. Returns "" if irrelevant."""
        if not page.content or page.error:
            return ""

        prompt = f"Research goal: {goal}\n\nPage content:\n{page.content[:6000]}"
        try:
            result = self._llm(prompt, _EXTRACT_SYSTEM).strip()
            if result.startswith("[IRRELEVANT]"):
                return ""
            return result
        except Exception as e:
            logger.warning(f"Evaluator.extract failed: {e}")
            # simple fallback: take first 300 chars
            return page.content[:300]

    # ── Confidence assessment ─────────────────────────────────────────────────
    def assess(self, session: ResearchSession) -> Tuple[float, bool]:
        """
        Returns (confidence_score, should_stop).
        should_stop = True when confidence >= threshold or LLM says done.
        """
        findings = self._collect_findings(session)
        if not findings:
            return 0.0, False

        prompt = (
            f"Research goal: {session.goal}\n\n"
            f"Findings collected so far:\n{findings[:4000]}"
        )
        try:
            import json as _json
            raw = self._llm(prompt, _CONFIDENCE_SYSTEM).strip()
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                data = _json.loads(match.group())
                conf = float(data.get("confidence", 0.5))
                done = bool(data.get("done", False)) or conf >= self._threshold
                logger.info(f"Evaluator: confidence={conf:.2f} done={done} reason={data.get('reason','')}")
                return conf, done
        except Exception as e:
            logger.warning(f"Evaluator.assess failed: {e}")

        # fallback: count steps
        n = len(session.steps)
        conf = min(0.5 + n * 0.08, 0.90)
        return conf, n >= 4

    # ── Loop detection ────────────────────────────────────────────────────────
    def detect_loop(self, session: ResearchSession) -> bool:
        """Returns True if the research appears to be looping uselessly."""
        queries = [s.query for s in session.steps[-6:]]  # last 6 queries
        if len(queries) < 3:
            return False

        # fast check: near-duplicate queries (no LLM needed)
        seen: set = set()
        for q in queries:
            norm = re.sub(r'\s+', ' ', q.lower().strip())
            if norm in seen:
                logger.warning("Evaluator: loop detected via exact duplicate query")
                return True
            seen.add(norm)

        prompt = f"Recent search queries:\n" + "\n".join(f"- {q}" for q in queries)
        try:
            import json as _json
            raw = self._llm(prompt, _LOOP_DETECT_SYSTEM).strip()
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                data = _json.loads(match.group())
                looping = bool(data.get("looping", False))
                if looping:
                    logger.warning(f"Evaluator: loop detected — {data.get('reason','')}")
                return looping
        except Exception as e:
            logger.warning(f"Evaluator.detect_loop failed: {e}")
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _collect_findings(self, session: ResearchSession) -> str:
        parts = []
        for step in session.steps:
            if step.extracted:
                parts.append(f"[Query: {step.query}]\n{step.extracted}")
        return "\n\n".join(parts)
