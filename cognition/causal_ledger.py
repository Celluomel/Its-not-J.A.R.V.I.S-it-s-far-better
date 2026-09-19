"""
PANDORABOX — CAUSAL LEDGER
======================
Append-only, structured record of every *causal* state transition the
ConsequenceTracker (and any future bound subsystem) produces.

DESIGN PRINCIPLE (anti-decorative):
  This ledger is NOT a standalone logging service. An entry is only ever
  created as the *trace of a real state change* that the ConsequenceTracker
  just performed (efficacy update / attention nudge / PCM resolution).
  If a mechanism doesn't change state, it doesn't write here. That is what
  keeps the ledger from becoming the 146th module nobody reads.

Storage:
  data/persona/causal_ledger.jsonl   (append-only, one JSON object per line)

Each event:
  {
    "cycle_id":        int | None,
    "event_id":        str,            # short unique id (this row)
    "ts":              iso8601,
    "source":          str,            # which mechanism emitted it
    "cause":           str,            # human-readable causal statement
    "kind":            str,            # prediction | action | outcome | error | update | override
    "input_state":     dict,
    "output_state":    dict,
    "delta":           dict,           # the *actual* change, if any
    "confidence":      float,
    "parent_event_ids":[str],          # causal ancestry -> enables chain() / why_goal()
    "refs":            dict            # goal_id, action, module, ...
  }

Query API (the "why did PandoraBOX choose X" capability):
  CausalLedger.chain(event_id)   -> [events from root .. event]
  CausalLedger.why_goal(goal_id) -> reconstructed causal story
  CausalLedger.recent(n)         -> tail
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class CausalLedger:
    def __init__(self, persona_dir: Optional[str] = None, enabled: bool = True):
        self.persona_dir = Path(persona_dir) if persona_dir else Path("data/persona")
        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.persona_dir / "causal_ledger.jsonl"
        self.enabled = enabled
        # in-memory index for chain()/why_goal() within a process
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._goal_index: Dict[str, List[str]] = {}  # goal_id -> [event_id,...]

    # ── write ────────────────────────────────────────────────────────────────
    def record(
        self,
        kind: str,
        cause: str,
        source: str = "consequence_tracker",
        cycle_id: Optional[int] = None,
        input_state: Optional[Dict] = None,
        output_state: Optional[Dict] = None,
        delta: Optional[Dict] = None,
        confidence: float = 1.0,
        parent_event_ids: Optional[List[str]] = None,
        refs: Optional[Dict] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        event_id = "ev_" + uuid.uuid4().hex[:10]
        ev = {
            "cycle_id": cycle_id,
            "event_id": event_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "cause": cause,
            "kind": kind,
            "input_state": input_state or {},
            "output_state": output_state or {},
            "delta": delta or {},
            "confidence": round(float(confidence), 4),
            "parent_event_ids": list(parent_event_ids or []),
            "refs": refs or {},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            # never let logging kill the loop
            return event_id
        self._by_id[event_id] = ev
        g = (refs or {}).get("goal_id")
        if g:
            self._goal_index.setdefault(str(g), []).append(event_id)
        return event_id

    # ── read ─────────────────────────────────────────────────────────────────
    def _load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        all_ev = list(self._by_id.values()) or self._load_all()
        # _by_id is unordered; if we had to load from disk, preserve order
        if all_ev and "event_id" in all_ev[0]:
            all_ev.sort(key=lambda e: e.get("ts", ""))
        return all_ev[-n:]

    def chain(self, event_id: str) -> List[Dict[str, Any]]:
        """Walk parent_event_ids up to the root; return [root .. event]."""
        index = dict(self._by_id)
        # top up from disk if needed
        if event_id not in index:
            for ev in self._load_all():
                index[ev["event_id"]] = ev
        seen: List[Dict[str, Any]] = []
        cur: Optional[str] = event_id
        guard = 0
        while cur and guard < 64:
            ev = index.get(cur)
            if ev is None:
                break
            seen.append(ev)
            parents = ev.get("parent_event_ids") or []
            cur = parents[0] if parents else None
            guard += 1
        seen.reverse()  # root first
        return seen

    def why_goal(self, goal_id: str, limit: int = 40) -> Dict[str, Any]:
        """
        Reconstruct the causal story for a goal from the ledger:
        the chain of predictions/actions/outcomes/errors that touched it.
        This is the concrete 'Why did PandoraBOX choose this goal?' answer.
        """
        index = dict(self._by_id)
        loaded = self._load_all()
        for ev in loaded:
            index[ev["event_id"]] = ev
        evs = [e for e in loaded if (e.get("refs") or {}).get("goal_id") == str(goal_id)]
        evs.sort(key=lambda e: e.get("ts", ""))
        evs = evs[-limit:]

        # aggregate the measurable consequence for this goal
        agg = {
            "goal_id": goal_id,
            "n_events": len(evs),
            "by_kind": {},
            "mean_abs_error": None,
            "efficacy_updates": [],
            "attention_updates": [],
        }
        errs = []
        for e in evs:
            agg["by_kind"][e.get("kind", "?")] = agg["by_kind"].get(e.get("kind", "?"), 0) + 1
            if e.get("kind") == "error":
                try:
                    errs.append(abs(float((e.get("delta") or {}).get("error", 0.0))))
                except Exception:
                    pass
            if e.get("kind") == "update" and (e.get("refs") or {}).get("target") == "efficacy":
                agg["efficacy_updates"].append(e.get("delta"))
            if e.get("kind") == "update" and (e.get("refs") or {}).get("target") == "attention":
                agg["attention_updates"].append(e.get("delta"))
        if errs:
            agg["mean_abs_error"] = round(sum(errs) / len(errs), 4)
        agg["events"] = evs
        return agg

    def stats(self) -> Dict[str, Any]:
        loaded = self._load_all()
        kinds: Dict[str, int] = {}
        for e in loaded:
            kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
        return {"n_events": len(loaded), "by_kind": kinds, "path": str(self.path)}
