"""
LUMINA V32 — Phase 3 : Canonical Data Schemas
==============================================
Defines the canonical format for every key data file.
Every component reads/writes through DataAccess, never raw JSON.

Canonical formats:
  goals.json         → { "goals": {id: GoalObject}, "cycle_counter": int, "saved_at": str }
  thought_stream.json→ { "thoughts": [ThoughtObject, ...] }
  thought_threads.json→{ id: ThreadObject, ... }
  tensions.json      → { "current": {name: float}, "history": [...], "last_updated": float }
  identity.json      → { "beliefs": {name: BeliefObject}, "updated_at": str }
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import time
import uuid


# ── TTL constants ─────────────────────────────────────────────────────────────
TTL_SHORT   = 3600        # 1 hour  — fleeting thoughts
TTL_MEDIUM  = 86400       # 1 day   — regular thoughts
TTL_LONG    = 604800      # 1 week  — important insights
TTL_FOREVER = 0           # no expiry


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def now_ts() -> float:
    return time.time()

def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ── Goal schema ───────────────────────────────────────────────────────────────

def make_goal(
    name: str,
    priority: float = 0.5,
    status: str = "active",
    origin: str = "manual",
    goal_id: Optional[str] = None,
    **extra
) -> Dict[str, Any]:
    """Return a canonical goal object."""
    return {
        "id":              goal_id or new_id("goal_"),
        "name":            name,
        "topic":           extra.get("topic", name),
        "origin":          origin,
        "priority":        max(0.0, min(1.0, priority)),
        "status":          status,          # active | completed | deferred | abandoned
        "energy":          extra.get("energy", 0.5),
        "created_at":      extra.get("created_at", now_iso()),
        "last_active":     extra.get("last_active", now_iso()),
        "completion":      extra.get("completion", 0.0),
        "persistence":     extra.get("persistence", 0.5),
        "actions_taken":   extra.get("actions_taken", 0),
        "origin_thoughts": extra.get("origin_thoughts", []),
        "priority_history":extra.get("priority_history", []),
        "merge_history":   extra.get("merge_history", []),
        "created_cycle":   extra.get("created_cycle", 0),
    }


def normalize_goal(raw: Any) -> Dict[str, Any]:
    """Ensure a goal object has all canonical fields."""
    if not isinstance(raw, dict):
        return make_goal(str(raw))
    defaults = make_goal(raw.get("name", raw.get("topic", "unknown")))
    return {**defaults, **raw}


# ── Thought schema ─────────────────────────────────────────────────────────────

def make_thought(
    content: str,
    thought_type: str = "reflection",
    priority: float = 0.5,
    source: str = "internal",
    ttl: int = TTL_MEDIUM,
    linked_goal: Optional[str] = None,
    conceptual_friction: float = 0.0,
    **extra
) -> Dict[str, Any]:
    """Return a canonical thought object."""
    return {
        "id":                   extra.get("id", new_id("th_")),
        "content":              content,
        "source":               source,
        "priority":             max(0.0, min(1.0, priority)),
        "thought_type":         thought_type,
        "timestamp":            extra.get("timestamp", now_iso()),
        # Phase 3 new fields
        "ttl":                  ttl,          # seconds until expiry (0 = never)
        "expires_at":           extra.get("expires_at",
                                    _expiry_ts(ttl) if ttl > 0 else None),
        "linked_goal":          linked_goal,  # goal id this thought maps to
        "conceptual_friction":  conceptual_friction,  # 0-1, how hard to resolve
        "compressed":           extra.get("compressed", False),
    }


def normalize_thought(raw: Dict) -> Dict[str, Any]:
    """Ensure a thought has all canonical fields."""
    defaults = make_thought(
        content=raw.get("content", ""),
        thought_type=raw.get("thought_type", "reflection"),
        priority=raw.get("priority", 0.5),
        source=raw.get("source", "internal"),
    )
    return {**defaults, **raw}


def is_expired(thought: Dict) -> bool:
    """Return True if thought has passed its TTL."""
    expires_at = thought.get("expires_at")
    if expires_at is None:
        return False
    return now_ts() > expires_at


def _expiry_ts(ttl_seconds: int) -> float:
    return now_ts() + ttl_seconds


# ── Thread schema ──────────────────────────────────────────────────────────────

def normalize_thread(raw: Dict) -> Dict[str, Any]:
    """Ensure a thread has all canonical fields."""
    return {
        "id":               raw.get("id", new_id("thr_")),
        "topic":            raw.get("topic", "unknown"),
        "goal":             raw.get("goal", ""),
        "current_state":    raw.get("current_state", ""),
        "history":          raw.get("history", []),
        "planned_actions":  raw.get("planned_actions", []),
        "memory_refs":      raw.get("memory_refs", []),
        "confidence":       raw.get("confidence", 0.5),
        "status":           raw.get("status", "active"),
        # Phase 3 new fields
        "iteration_count":  raw.get("iteration_count", len(raw.get("history", []))),
        "identity_link":    raw.get("identity_link", None),  # belief id updated by this thread
        "created_at":       raw.get("created_at", now_iso()),
        "resolved_at":      raw.get("resolved_at", None),
    }


# ── Belief / Identity schema ───────────────────────────────────────────────────

def make_belief(
    name: str,
    value: float = 0.5,
    confidence: float = 0.5,
    source_thread: Optional[str] = None,
    **extra
) -> Dict[str, Any]:
    """Return a canonical belief object."""
    return {
        "id":            extra.get("id", new_id("bel_")),
        "name":          name,
        "value":         max(0.0, min(1.0, value)),
        "confidence":    max(0.0, min(1.0, confidence)),
        "source_thread": source_thread,
        "created_at":    extra.get("created_at", now_iso()),
        "updated_at":    extra.get("updated_at", now_iso()),
        "history":       extra.get("history", []),
    }


# ── Goals file envelope ────────────────────────────────────────────────────────

def make_goals_envelope(goals: Dict[str, Dict], cycle_counter: int = 0) -> Dict:
    return {
        "cycle_counter": cycle_counter,
        "goals":         goals,
        "saved_at":      now_iso(),
    }


# ── Identity file envelope ─────────────────────────────────────────────────────

def make_identity_envelope(beliefs: Dict[str, Dict]) -> Dict:
    return {
        "beliefs":    beliefs,
        "updated_at": now_iso(),
        "version":    "v3",
    }
