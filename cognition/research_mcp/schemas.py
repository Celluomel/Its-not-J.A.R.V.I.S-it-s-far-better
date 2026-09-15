"""
Shared data structures for the Research MCP plugin.
All modules import from here — no circular deps.
"""
from __future__ import annotations
import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class ResearchMode(Enum):
    ON_DEMAND   = 1   # explicit call, no memory write
    EXTENDED    = 2   # auto-triggered for complex queries, richer synthesis
    BACKGROUND  = 3   # async overnight, summarised + stored in journal & memory


class ResearchStatus(Enum):
    PENDING     = "pending"
    RUNNING     = "running"
    COMPLETE    = "complete"
    FAILED      = "failed"
    ABORTED     = "aborted"   # hit safety limit


@dataclass
class SubGoal:
    goal_id:     str
    description: str
    priority:    int   = 1     # 1 = highest
    completed:   bool  = False
    findings:    str   = ""


@dataclass
class WebPage:
    url:        str
    title:      str        = ""
    content:    str        = ""
    fetched_at: str        = ""
    credibility: float     = 0.5   # 0..1
    error:      str        = ""
    source:     str        = ""    # provider tag, e.g. "rss_BBC", "brave", "claude"


@dataclass
class ResearchStep:
    step_id:   str
    query:     str
    pages:     List[WebPage] = field(default_factory=list)
    extracted: str           = ""   # cleaned text from pages
    confidence: float        = 0.0


@dataclass
class ResearchSession:
    research_id:      str   = field(default_factory=lambda: f"R-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}")
    mode:             ResearchMode = ResearchMode.ON_DEMAND
    goal:             str   = ""
    depth:            int   = 3     # 1–5  → controls sub-goal breadth
    max_iterations:   int   = 8
    trigger_reason:   str   = ""    # why RAC scheduled this
    status:           ResearchStatus = ResearchStatus.PENDING

    sub_goals:        List[SubGoal]      = field(default_factory=list)
    steps:            List[ResearchStep] = field(default_factory=list)

    summary:          str   = ""
    sources:          List[Dict[str, Any]] = field(default_factory=list)   # {url, title, credibility}
    confidence_score: float = 0.0
    knowledge_nodes:  int   = 0

    started_at:       str   = ""
    finished_at:      str   = ""
    influences_topics: List[str] = field(default_factory=list)

    # RAC daily tracking
    interaction_depth_at_trigger: float = 0.0


@dataclass
class DailyMetrics:
    """Tracked by RAC across a day to decide whether to schedule Mode 3."""
    date:             str   = field(default_factory=lambda: datetime.date.today().isoformat())
    complex_queries:  int   = 0
    total_queries:    int   = 0
    topic_counts:     Dict[str, int] = field(default_factory=dict)
    confidence_gaps:  int   = 0   # times LLM signalled low confidence
    research_today:   int   = 0   # how many Mode 3 runs already today


@dataclass
class ResearchResult:
    """Returned to caller for all modes."""
    research_id:  str
    mode:         int
    goal:         str
    summary:      str
    sources:      List[Dict[str, Any]]
    confidence:   float
    status:       str
    knowledge_nodes: int = 0
    error:        str   = ""
    journal_entry: Optional[Dict[str, Any]] = None
