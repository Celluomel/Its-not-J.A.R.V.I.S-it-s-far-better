"""
Synthesizer — produces the final output from a completed ResearchSession.

Mode 1: concise answer
Mode 2: deep structured analysis
Mode 3: structured knowledge packet for memory storage + journal
"""
from __future__ import annotations
import json
import logging
import re
from typing import Callable, List

from cognition.research_mcp.schemas import ResearchSession, ResearchMode

logger = logging.getLogger("research_mcp.synthesizer")


_SHORT_SYSTEM = """You are a research synthesis assistant.
Given a goal and findings from web research, write a clear, direct answer (3-6 sentences).
Cite key facts. No fluff, no markdown headers. Plain prose only.
"""

_DEEP_SYSTEM = """You are an expert research analyst.
Given a goal and multi-source findings, write a comprehensive structured analysis.
Use this structure:
1. Executive summary (2-3 sentences)
2. Key findings (3-5 bullet points)
3. Nuance / conflicting views (if any)
4. Confidence assessment
No preamble. Start directly with the summary.
"""

_STRUCTURED_SYSTEM = """You are a knowledge extraction specialist.
Given a research goal and findings, produce a structured knowledge packet.
Return ONLY valid JSON with this shape:
{
  "topic": "...",
  "summary": "...",
  "key_facts": ["...", "..."],
  "confidence": 0.82,
  "influences_topics": ["topic_a", "topic_b"],
  "knowledge_gaps": ["..."]
}
- topic: 2-4 word slug of the subject
- summary: 3-5 sentence synthesis
- key_facts: 3-6 concrete, verifiable facts discovered
- confidence: 0.0-1.0 honest assessment
- influences_topics: related topics this knowledge affects
- knowledge_gaps: what remains unknown / needs more research
Return ONLY the JSON object.
"""


class Synthesizer:

    def __init__(self, llm_fn: Callable[[str, str], str]):
        self._llm = llm_fn

    def synthesize(self, session: ResearchSession) -> str:
        """
        Produce the final summary text appropriate for the session mode.
        Stores result in session.summary and returns it.
        """
        findings = self._collect_findings(session)
        goal     = session.goal

        if not findings:
            session.summary = (
                f"Research completed but no substantive findings were collected "
                f"for: {goal}. This may be because web access is not yet configured "
                f"(see cognition/research_mcp/web_agent.py to plug in a real fetcher)."
            )
            session.confidence_score = 0.1
            return session.summary

        if session.mode == ResearchMode.ON_DEMAND:
            summary = self._run_llm(goal, findings, _SHORT_SYSTEM)

        elif session.mode == ResearchMode.EXTENDED:
            summary = self._run_llm(goal, findings, _DEEP_SYSTEM)

        else:  # BACKGROUND
            summary, structured = self._run_structured(goal, findings, session)
            if structured:
                session.influences_topics = structured.get("influences_topics", [])
                session.confidence_score  = float(structured.get("confidence", session.confidence_score))
            session.summary = summary
            return summary

        session.summary = summary
        return summary

    def _run_llm(self, goal: str, findings: str, system: str) -> str:
        prompt = f"Research goal: {goal}\n\nFindings:\n{findings[:5000]}"
        try:
            return self._llm(prompt, system).strip()
        except Exception as e:
            logger.error(f"Synthesizer LLM call failed: {e}")
            return f"Synthesis failed: {e}. Raw findings available in research journal."

    def _run_structured(
        self, goal: str, findings: str, session: ResearchSession
    ):
        """Returns (summary_text, structured_dict_or_None)."""
        prompt = f"Research goal: {goal}\n\nFindings:\n{findings[:5000]}"
        try:
            raw   = self._llm(prompt, _STRUCTURED_SYSTEM).strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data    = json.loads(match.group())
                summary = data.get("summary", "")
                facts   = data.get("key_facts", [])
                if facts:
                    summary += "\n\nKey facts:\n" + "\n".join(f"• {f}" for f in facts)
                return summary, data
        except Exception as e:
            logger.error(f"Synthesizer structured LLM failed: {e}")

        # fallback to deep mode
        return self._run_llm(goal, findings, _DEEP_SYSTEM), None

    def _collect_findings(self, session: ResearchSession) -> str:
        parts = []
        for step in session.steps:
            if step.extracted:
                parts.append(f"[Query: {step.query}]\n{step.extracted}")
        return "\n\n".join(parts)

    def build_sources(self, session: ResearchSession) -> List[dict]:
        """Build deduplicated source list for the journal."""
        seen: set = set()
        sources   = []
        for step in session.steps:
            for page in step.pages:
                if page.url not in seen and not page.error:
                    seen.add(page.url)
                    sources.append({
                        "url":         page.url,
                        "title":       page.title,
                        "credibility": round(page.credibility, 2),
                    })
        return sources
