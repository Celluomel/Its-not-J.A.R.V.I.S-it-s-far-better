"""
LUMINA V32 — Phase 3 : Data Access Layer (DAL)
===============================================
Single point of truth for all JSON reads and writes.

NEVER do raw json.load/json.dump on key persona files from cognition/core code.
Use this module instead:

    from core.data.access import DataAccess
    dal = DataAccess()

    goals  = dal.get_goals()          # → List[Dict]  always normalized
    dal.save_goal(goal_dict)          # creates or updates by id

    thoughts = dal.get_thoughts()     # → List[Dict]  always normalized, expired removed
    dal.add_thought(thought_dict)

    threads = dal.get_threads()       # → List[Dict]
    dal.save_thread(thread_dict)

    beliefs = dal.get_beliefs()       # → List[Dict]
    dal.save_belief(belief_dict)

All methods are thread-safe (RLock).
All writes are atomic (write to .tmp then rename).
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.data.schemas import (
    normalize_goal, normalize_thought, normalize_thread,
    make_belief, make_goals_envelope, make_identity_envelope,
    is_expired, now_iso, now_ts, new_id,
)

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()          # global lock shared by all instances


class DataAccess:
    """
    Unified, schema-aware access layer for Lumina persona data files.

    One instance per component is fine — all share the same RLock.
    """

    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.persona_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _path(self, filename: str) -> Path:
        return self.persona_dir / filename

    def _load(self, filename: str, default: Any = None) -> Any:
        """Load a JSON file safely. Returns default if missing or corrupt."""
        p = self._path(filename)
        if not p.exists():
            return default if default is not None else {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[DAL] Failed to load {filename}: {e}")
            return default if default is not None else {}

    def _save(self, filename: str, data: Any) -> bool:
        """Atomic write: write to .tmp then rename."""
        p = self._path(filename)
        tmp = p.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, p)
            return True
        except Exception as e:
            logger.error(f"[DAL] Failed to save {filename}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # GOALS
    # ──────────────────────────────────────────────────────────────────────────

    def _load_goals_raw(self) -> Dict:
        raw = self._load("goals.json", {})
        # Ensure envelope structure
        if "goals" not in raw:
            raw = {"cycle_counter": 0, "goals": raw, "saved_at": now_iso()}
        return raw

    def get_goals(
        self,
        status: Optional[str] = None,
        min_priority: float = 0.0,
    ) -> List[Dict]:
        """
        Return all goals as a normalized list.

        Args:
            status: filter by status ('active', 'completed', 'deferred', 'abandoned')
            min_priority: only return goals with priority >= this value
        """
        with _LOCK:
            raw = self._load_goals_raw()
            goals_raw = raw.get("goals", {})
            # Support both dict {id: goal} and list formats
            if isinstance(goals_raw, dict):
                goals = [normalize_goal(g) for g in goals_raw.values()]
            else:
                goals = [normalize_goal(g) for g in goals_raw]

            if status:
                goals = [g for g in goals if g.get("status") == status]
            if min_priority > 0.0:
                goals = [g for g in goals if g.get("priority", 0) >= min_priority]

            return goals

    def get_goal_by_id(self, goal_id: str) -> Optional[Dict]:
        """Return a single goal by id, or None."""
        with _LOCK:
            for g in self.get_goals():
                if g.get("id") == goal_id:
                    return g
            return None

    def save_goal(self, goal: Dict) -> bool:
        """Create or update a goal (matched by id)."""
        with _LOCK:
            raw = self._load_goals_raw()
            goals_raw = raw.get("goals", {})
            # Migrate list → dict if needed
            if isinstance(goals_raw, list):
                goals_raw = {g.get("id", new_id("goal_")): normalize_goal(g)
                             for g in goals_raw}

            goal_n = normalize_goal(goal)
            goal_id = goal_n["id"]
            goals_raw[goal_id] = goal_n
            raw["goals"] = goals_raw
            raw["saved_at"] = now_iso()
            return self._save("goals.json", raw)

    def delete_goal(self, goal_id: str) -> bool:
        """Remove a goal by id."""
        with _LOCK:
            raw = self._load_goals_raw()
            goals_raw = raw.get("goals", {})
            if isinstance(goals_raw, dict) and goal_id in goals_raw:
                del goals_raw[goal_id]
                raw["goals"] = goals_raw
                raw["saved_at"] = now_iso()
                return self._save("goals.json", raw)
            return False

    def increment_cycle_counter(self) -> int:
        """Increment and return the goal cycle counter."""
        with _LOCK:
            raw = self._load_goals_raw()
            raw["cycle_counter"] = raw.get("cycle_counter", 0) + 1
            self._save("goals.json", raw)
            return raw["cycle_counter"]

    # ──────────────────────────────────────────────────────────────────────────
    # THOUGHTS
    # ──────────────────────────────────────────────────────────────────────────

    def get_thoughts(
        self,
        limit: int = 50,
        skip_expired: bool = True,
        thought_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return thought stream as a normalized list.

        Args:
            limit: max number of thoughts (most recent first)
            skip_expired: if True, expired thoughts are filtered out
            thought_type: filter by type
        """
        with _LOCK:
            raw = self._load("thought_stream.json", {"thoughts": []})
            thoughts_raw = raw.get("thoughts", [])

            # Support root-level list
            if isinstance(raw, list):
                thoughts_raw = raw

            thoughts = [normalize_thought(t) for t in thoughts_raw]

            if skip_expired:
                thoughts = [t for t in thoughts if not is_expired(t)]

            if thought_type:
                thoughts = [t for t in thoughts if t.get("thought_type") == thought_type]

            # Most recent first
            thoughts.sort(key=lambda t: str(t.get("timestamp", "")), reverse=True)
            return thoughts[:limit]

    def add_thought(self, thought: Dict) -> bool:
        """Append a thought to the stream."""
        with _LOCK:
            raw = self._load("thought_stream.json", {"thoughts": []})
            if isinstance(raw, list):
                raw = {"thoughts": raw}
            if "thoughts" not in raw:
                raw["thoughts"] = []

            thought_n = normalize_thought(thought)
            raw["thoughts"].append(thought_n)

            # Keep stream bounded: max 200 thoughts, purge expired first
            all_thoughts = raw["thoughts"]
            active = [t for t in all_thoughts if not is_expired(normalize_thought(t))]
            if len(active) > 200:
                # Keep highest priority recent thoughts
                active.sort(key=lambda t: (t.get("priority", 0), t.get("timestamp", "")),
                            reverse=True)
                active = active[:200]
            raw["thoughts"] = active
            return self._save("thought_stream.json", raw)

    def expire_old_thoughts(self) -> int:
        """Remove expired thoughts. Returns count removed."""
        with _LOCK:
            raw = self._load("thought_stream.json", {"thoughts": []})
            if isinstance(raw, list):
                raw = {"thoughts": raw}
            before = len(raw.get("thoughts", []))
            active = [t for t in raw.get("thoughts", [])
                      if not is_expired(normalize_thought(t))]
            raw["thoughts"] = active
            self._save("thought_stream.json", raw)
            removed = before - len(active)
            if removed:
                logger.info(f"[DAL] Expired {removed} thoughts")
            return removed

    # ──────────────────────────────────────────────────────────────────────────
    # THREADS
    # ──────────────────────────────────────────────────────────────────────────

    def get_threads(self, status: Optional[str] = None) -> List[Dict]:
        """Return all thought threads as a normalized list."""
        with _LOCK:
            raw = self._load("thought_threads.json", {})
            threads = [normalize_thread(v) for v in raw.values()
                       if isinstance(v, dict)]
            if status:
                threads = [t for t in threads if t.get("status") == status]
            return threads

    def get_thread_by_id(self, thread_id: str) -> Optional[Dict]:
        with _LOCK:
            raw = self._load("thought_threads.json", {})
            entry = raw.get(thread_id)
            return normalize_thread(entry) if isinstance(entry, dict) else None

    def save_thread(self, thread: Dict) -> bool:
        """Create or update a thread."""
        with _LOCK:
            raw = self._load("thought_threads.json", {})
            thread_n = normalize_thread(thread)
            thread_id = thread_n["id"]
            raw[thread_id] = thread_n
            return self._save("thought_threads.json", raw)

    # ──────────────────────────────────────────────────────────────────────────
    # IDENTITY / BELIEFS
    # ──────────────────────────────────────────────────────────────────────────

    def get_beliefs(self) -> List[Dict]:
        """Return all beliefs as a list."""
        with _LOCK:
            raw = self._load("identity.json", {"beliefs": {}, "updated_at": now_iso()})
            beliefs_raw = raw.get("beliefs", {})
            if isinstance(beliefs_raw, dict):
                return list(beliefs_raw.values())
            return beliefs_raw

    def get_belief_by_name(self, name: str) -> Optional[Dict]:
        with _LOCK:
            for b in self.get_beliefs():
                if b.get("name") == name:
                    return b
            return None

    # Hard cap on beliefs stored in identity.json.
    # ThreadLifecycleManager extracts 4-5 beliefs per slow cycle; without a cap
    # the store grows to 2000+ between SCS runs, making every SCS pass O(n)
    # on a file it will immediately prune back to 40. Capping here keeps the
    # file small without affecting the 40-belief self_concept.json cap.
    BELIEF_MAX_IN_IDENTITY = 200

    def save_belief(self, belief: Dict) -> bool:
        """Create or update a belief (matched by name), capped at BELIEF_MAX_IN_IDENTITY."""
        with _LOCK:
            raw = self._load("identity.json", make_identity_envelope({}))
            beliefs = raw.get("beliefs", {})
            if isinstance(beliefs, list):
                beliefs = {b.get("name", new_id("bel_")): b for b in beliefs}

            # Normalize name: strip repeated "expressed_" prefix chains
            _raw_bname = belief.get("name", new_id("bel_"))
            import re as _bre
            name = _bre.sub(r'^(expressed_)+', 'expressed_', _raw_bname)
            # Update the belief dict itself so the stored name is clean
            if name != _raw_bname:
                belief = {**belief, "name": name}
            existing = beliefs.get(name, {})
            merged = {**existing, **belief, "updated_at": now_iso()}
            if "history" not in merged:
                merged["history"] = []
            if existing.get("value") is not None and existing["value"] != belief.get("value"):
                merged["history"].append({
                    "old_value": existing["value"],
                    "new_value": belief.get("value"),
                    "at": now_iso(),
                })
            beliefs[name] = merged

            # Hard cap: if over limit, prune weakest (lowest confidence) entries
            # but always keep the one we just wrote.
            if len(beliefs) > self.BELIEF_MAX_IN_IDENTITY:
                sorted_keys = sorted(
                    (k for k in beliefs if k != name),
                    key=lambda k: beliefs[k].get("confidence", beliefs[k].get("value", 0.0))
                )
                overflow = len(beliefs) - self.BELIEF_MAX_IN_IDENTITY
                for k in sorted_keys[:overflow]:
                    del beliefs[k]

            raw["beliefs"] = beliefs
            raw["updated_at"] = now_iso()
            return self._save("identity.json", raw)

    # ──────────────────────────────────────────────────────────────────────────
    # TENSIONS
    # ──────────────────────────────────────────────────────────────────────────

    def get_tensions(self) -> Dict[str, float]:
        """Return current tension values as {name: float}."""
        with _LOCK:
            raw = self._load("tensions.json", {"current": {}})
            return raw.get("current", {})

    def save_tensions(self, tensions: Dict[str, float]) -> bool:
        """Update tension values."""
        with _LOCK:
            raw = self._load("tensions.json", {"current": {}, "history": []})
            # Record snapshot in history (keep last 48)
            history = raw.get("history", [])
            if raw.get("current"):
                history.append({"snapshot": raw["current"], "at": now_ts()})
                history = history[-48:]
            raw["current"] = tensions
            raw["history"] = history
            raw["last_updated"] = now_ts()
            return self._save("tensions.json", raw)

    # ──────────────────────────────────────────────────────────────────────────
    # GENERIC KEY-VALUE (for other files)
    # ──────────────────────────────────────────────────────────────────────────

    def load_file(self, filename: str, default: Any = None) -> Any:
        """Raw load of any persona file."""
        with _LOCK:
            return self._load(filename, default)

    def save_file(self, filename: str, data: Any) -> bool:
        """Raw save of any persona file."""
        with _LOCK:
            return self._save(filename, data)

    # ──────────────────────────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Quick health snapshot."""
        goals = self.get_goals()
        thoughts = self.get_thoughts(skip_expired=False)
        active_thoughts = [t for t in thoughts if not is_expired(t)]
        threads = self.get_threads()
        beliefs = self.get_beliefs()

        return {
            "goals": {
                "total":    len(goals),
                "active":   len([g for g in goals if g.get("status") == "active"]),
                "high_pri": len([g for g in goals if g.get("priority", 0) >= 0.7]),
            },
            "thoughts": {
                "total":    len(thoughts),
                "active":   len(active_thoughts),
                "expired":  len(thoughts) - len(active_thoughts),
            },
            "threads": {
                "total":    len(threads),
                "active":   len([t for t in threads if t.get("status") == "active"]),
                "deferred": len([t for t in threads if t.get("status") == "deferred"]),
            },
            "beliefs": {
                "total":    len(beliefs),
            },
        }
