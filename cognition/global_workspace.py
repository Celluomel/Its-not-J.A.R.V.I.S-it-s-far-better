"""
Global Workspace
================
Shared cognitive broadcast bus inspired by Bernard Baars' Global Workspace Theory.

Every module in Lumina can *broadcast* an item here.  The workspace acts as a
short-term attention buffer: the most recently / highly-prioritised signals are
visible to all other modules, enabling coordination without tight coupling.

SEMANTIC COMPETITION (2026-03-09 rewrite)
-----------------------------------------
Problem: pure priority-based eviction allowed semantically identical items
(same topic, same source) to accumulate → workspace became a fixation echo
chamber.  Lumina's CCS score was artificially high and her cognitive horizon
narrowed to the same recurring thoughts.

Fix: broadcast() now runs a two-stage competition before admitting a new item:

  Stage 1 — Source throttle
    A source cannot have more than SOURCE_MAX items in the buffer at once.
    Oldest items from that source are evicted first.

  Stage 2 — Semantic similarity gate
    The new item is compared to existing items via bag-of-words Jaccard
    similarity.  If any existing item scores above SIMILARITY_BLOCK threshold
    AND was broadcast recently (within RECENCY_WINDOW seconds), the new item
    is BLOCKED.  This prevents exact-duplicate and near-duplicate thoughts
    from re-entering the workspace.

    Exception: high-priority signals (priority ≥ PRIORITY_OVERRIDE) bypass
    the semantic gate — urgent system signals (safety, surprise) always land.

  Stage 3 — Classic priority eviction (unchanged)
    If the buffer is at capacity after stages 1–2, the lowest-priority item
    is evicted to make room.

Consequence: the workspace naturally rotates toward diverse content, forcing
different modules to compete for attention — exactly as GWT predicts.

SELF-ANCHOR SLOT (unified self-model integration)
--------------------------------------------------
A single protected slot outside the competition buffer holds the current
SelfModelMoment.  It is:

  - Never evicted (lives outside the CAPACITY deque)
  - Never blocked by the Jaccard semantic gate
  - Always visible in recent() and top() as a stable background item
  - Used as a coherence reference: new items whose content aligns with the
    anchor receive a small priority boost (ANCHOR_COHERENCE_BOOST)

This means every item in the workspace is interpreted against a stable
self-reference, rather than competing with each other in a vacuum.  The self
becomes a frame rather than just another competitor.

Integration model (unchanged API)
----------------------------------
  workspace = GlobalWorkspace()
  workspace.broadcast("memory", "contradiction detected", priority=0.8)
  workspace.broadcast("curiosity", "new topic: consciousness", priority=0.6)
  workspace.set_self_anchor("phi=0.63 | warm engaged curiosity")  # from SelfModelMoment
  items = workspace.recent(n=10)   # anchor always included at position 0
"""

import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ── Phase 4.x: structured live workspace state ──────────────────────────────
# Additive to the broadcast blackboard above, NOT a replacement. The
# blackboard (WorkspaceItem/broadcast/recent/top) keeps doing its existing
# job (diagnostic log + dedup gate + self-anchor). WorkspaceState is the
# "what is Lumina globally reasoning about right now" object: a live,
# structured record that multiple subsystems read AND write during a single
# cognitive cycle, instead of each subsystem only ever reading its own
# persisted state and writing a private text fragment.
#
# Design note (Phase 4.x audit): don't let active_hypotheses grow without
# bound — WorkspaceCompetition's stagnation penalty and GlobalWorkspace's
# Jaccard semantic gate exist specifically to prevent a fixation echo
# chamber (see module docstring above, 2026-03-09 rewrite). Hypotheses here
# are timestamped and HYPOTHESIS_TTL-bounded for the same reason.
HYPOTHESIS_TTL = 180.0  # seconds a hypothesis stays "active" without being re-asserted


@dataclass
class WorkspaceState:
    """
    Live, structured snapshot of Lumina's current global cognitive context.
    One instance lives on GlobalWorkspace and is mutated in place via
    update_state(); read via get_state() (returns a shallow copy).
    """
    focus:               Optional[str]      = None
    intention:            Optional[str]      = None
    active_hypotheses:    List[Dict]         = field(default_factory=list)
    uncertainty:          float              = 0.0
    predicted_futures:    List[Dict]         = field(default_factory=list)
    goals:                List[Dict]         = field(default_factory=list)
    motivations:          Dict[str, float]   = field(default_factory=dict)
    narrative_state:      Optional[str]      = None
    executive_policy:     Dict[str, float]   = field(default_factory=dict)
    confidence:           float              = 0.5
    context:              Dict[str, Any]     = field(default_factory=dict)

    # Reportability / provenance (not requested field-by-field in the spec,
    # but required for "Broadcast Participants" / "Recent Workspace Changes")
    updated_by:            List[Dict]         = field(default_factory=list)
    last_updated:          float              = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "focus":              self.focus,
            "intention":          self.intention,
            "active_hypotheses":  self.active_hypotheses,
            "uncertainty":        round(self.uncertainty, 4),
            "predicted_futures":  self.predicted_futures,
            "goals":              self.goals,
            "motivations":        self.motivations,
            "narrative_state":    self.narrative_state,
            "executive_policy":   self.executive_policy,
            "confidence":         round(self.confidence, 4),
            "context":            self.context,
            "updated_by":         self.updated_by[-10:],
            "last_updated":       round(self.last_updated, 3),
        }


@dataclass
class WorkspaceItem:
    source:    str
    content:   Any
    priority:  float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "source":    self.source,
            "content":   str(self.content)[:200],
            "priority":  round(self.priority, 3),
            "timestamp": round(self.timestamp, 3),
        }

    def _token_set(self) -> Set[str]:
        """Bag-of-words token set for Jaccard similarity."""
        text = str(self.content).lower()
        return {t.strip('.,!?;:\'"()[]') for t in text.split() if len(t) >= 3}


def _jaccard(a: WorkspaceItem, b: WorkspaceItem) -> float:
    """Jaccard similarity between two item token sets."""
    sa, sb = a._token_set(), b._token_set()
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


class GlobalWorkspace:
    """
    Shared broadcast bus for all cognitive modules — with semantic competition
    and a persistent self-anchor that provides a stable identity reference frame.
    Thread-safe.
    """

    CAPACITY              = 32    # max items in competition buffer
    SOURCE_MAX            = 4     # max items per source before oldest is evicted
    SIMILARITY_BLOCK      = 0.55  # Jaccard threshold above which a new item is blocked
    RECENCY_WINDOW        = 120.0 # seconds — only block if similar item is this recent
    PRIORITY_OVERRIDE     = 0.75  # items above this bypass the semantic gate

    # Self-anchor constants
    ANCHOR_COHERENCE_BOOST  = 0.06   # priority bonus for items that resonate with anchor
    ANCHOR_COHERENCE_THRESH = 0.25   # Jaccard threshold to qualify for the boost
    ANCHOR_SOURCE           = "self_anchor"

    def __init__(self, max_items: int = 32):
        self._lock     = threading.RLock()
        self._buffer: deque = deque(maxlen=max_items)
        self._listeners: List[Callable] = []
        self._blocked_count: int = 0   # diagnostic counter

        # ── Protected self-anchor slot — never evicted, never gated ───────────
        self._self_anchor: Optional[WorkspaceItem] = None

        # ── Phase 4.x: structured live state ───────────────────────────────
        self._state = WorkspaceState()
        self._state_lock = threading.RLock()

    # ── Phase 4.x: WorkspaceState read/write ─────────────────────────────────

    def get_state(self) -> WorkspaceState:
        """
        Return the current WorkspaceState. Callers should treat this as
        read-mostly; use update_state()/set_hypotheses()/edit_executive_policy()
        to mutate so provenance (updated_by) stays accurate.
        """
        with self._state_lock:
            return self._state

    def update_state(self, source: str, **fields) -> WorkspaceState:
        """
        Merge `fields` into the shared WorkspaceState and record provenance.

        Any subsystem can call this with just the fields it computed this
        cycle — e.g. world_model.py might call
            workspace.update_state("world_model", predicted_futures=[...])
        without needing to know or overwrite what identity/goals/calibration
        wrote. Unknown field names are ignored (logged at debug level via
        the caller, not raised) so a typo in one adapter can't crash others.
        """
        with self._state_lock:
            for key, value in fields.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._state.updated_by.append({
                "source": source,
                "fields": list(fields.keys()),
                "timestamp": round(time.time(), 3),
            })
            self._state.updated_by = self._state.updated_by[-50:]
            self._state.last_updated = time.time()
            return self._state

    def set_hypotheses(self, source: str, hypotheses: List[Dict]) -> None:
        """
        Replace active_hypotheses with a fresh ranked set (e.g. from
        WorkspaceCompetition.compete()), stamping each with an expiry so
        stale hypotheses fall out of the workspace instead of accumulating
        forever (see HYPOTHESIS_TTL note above the dataclass).
        """
        now = time.time()
        stamped = []
        for h in hypotheses:
            h = dict(h)
            h.setdefault("_ws_expires_at", now + HYPOTHESIS_TTL)
            stamped.append(h)
        with self._state_lock:
            self._state.active_hypotheses = stamped
            self._state.updated_by.append({
                "source": source, "fields": ["active_hypotheses"],
                "timestamp": round(now, 3),
            })
            self._state.updated_by = self._state.updated_by[-50:]
            self._state.last_updated = now

    def prune_expired_hypotheses(self) -> int:
        """Drop hypotheses past their TTL. Returns count removed."""
        now = time.time()
        with self._state_lock:
            before = len(self._state.active_hypotheses)
            self._state.active_hypotheses = [
                h for h in self._state.active_hypotheses
                if h.get("_ws_expires_at", now) > now
            ]
            return before - len(self._state.active_hypotheses)

    def edit_executive_policy(self, source: str, **deltas) -> Dict[str, float]:
        """
        Executive editing: nudge shared policy dimensions (e.g.
        increase_skepticism -> executive_policy['skepticism'] += 0.2)
        instead of writing to the LLM directly. Values are additive and
        clamped to [-1.0, 1.0].
        """
        with self._state_lock:
            pol = self._state.executive_policy
            for dim, delta in deltas.items():
                pol[dim] = max(-1.0, min(1.0, pol.get(dim, 0.0) + delta))
            self._state.updated_by.append({
                "source": source, "fields": [f"executive_policy.{d}" for d in deltas],
                "timestamp": round(time.time(), 3),
            })
            self._state.updated_by = self._state.updated_by[-50:]
            self._state.last_updated = time.time()
            return pol

    def state_summary(self) -> Dict:
        """Reportability: dashboard-facing summary of the structured state."""
        with self._state_lock:
            s = self._state.to_dict()
        participants = sorted({u["source"] for u in s["updated_by"]})
        return {
            **s,
            "broadcast_participants": participants,
            "workspace_entropy": self.diversity_score(),
        }

    # ── Write ─────────────────────────────────────────────────────────────────

    def broadcast(self, source: str, content: Any, priority: float = 0.5) -> bool:
        """
        Publish an item to the workspace.

        Returns True if admitted, False if blocked by semantic gate.

        Parameters
        ----------
        source   : name of the emitting module  e.g. "memory", "curiosity"
        content  : any Python value (str, dict, …)
        priority : 0.0–1.0  (higher = more salient)

        Self-anchor coherence boost
        ---------------------------
        If a self-anchor is set and the new item's content has Jaccard
        similarity ≥ ANCHOR_COHERENCE_THRESH with the anchor, its priority
        is nudged up by ANCHOR_COHERENCE_BOOST before entering competition.
        This means workspace items that resonate with the current self-model
        are slightly favoured — the self becomes an integrating reference
        frame rather than just another competitor.
        """
        item = WorkspaceItem(
            source   = source,
            content  = content,
            priority = max(0.0, min(1.0, priority)),
        )

        with self._lock:
            # ── Self-anchor coherence boost (before all gates) ────────────
            if self._self_anchor is not None:
                anchor_sim = _jaccard(item, self._self_anchor)
                if anchor_sim >= self.ANCHOR_COHERENCE_THRESH:
                    item = WorkspaceItem(
                        source   = item.source,
                        content  = item.content,
                        priority = min(1.0, item.priority + self.ANCHOR_COHERENCE_BOOST),
                        timestamp= item.timestamp,
                    )

            items = list(self._buffer)
            now   = time.time()

            # ── Stage 2: Semantic gate (bypass for urgent signals) ─────────
            if item.priority < self.PRIORITY_OVERRIDE:
                for existing in items:
                    age = now - existing.timestamp
                    if age > self.RECENCY_WINDOW:
                        continue
                    if _jaccard(item, existing) >= self.SIMILARITY_BLOCK:
                        self._blocked_count += 1
                        return False

            # ── Stage 1: Source throttle ───────────────────────────────────
            source_items = [i for i in items if i.source == source]
            if len(source_items) >= self.SOURCE_MAX:
                oldest = min(source_items, key=lambda i: i.timestamp)
                items.remove(oldest)
                self._buffer = deque(items, maxlen=self.CAPACITY)

            # ── Stage 3: Classic priority eviction ────────────────────────
            items = list(self._buffer)
            if len(items) >= self.CAPACITY:
                min_idx = min(range(len(items)), key=lambda i: items[i].priority)
                if items[min_idx].priority < item.priority:
                    del items[min_idx]
                    self._buffer = deque(items, maxlen=self.CAPACITY)

            self._buffer.append(item)

        # Notify listeners (non-blocking, outside lock)
        for cb in self._listeners:
            try:
                cb(item)
            except Exception:
                pass

        return True

    # ── Self-anchor ───────────────────────────────────────────────────────────

    def set_self_anchor(self, content: Any) -> None:
        """
        Set the persistent self-anchor.

        The anchor is a stable background item representing the current
        SelfModelMoment.  It is:
          - Never evicted
          - Never blocked by the Jaccard semantic gate
          - Always prepended to recent() and included in top()
          - Used as a coherence reference in broadcast()

        Call once per turn from CognitiveOrganism after updating the
        SelfModelMoment (i.e. after _build_prompt_additions resolves).
        """
        with self._lock:
            self._self_anchor = WorkspaceItem(
                source   = self.ANCHOR_SOURCE,
                content  = content,
                priority = 0.0,   # lowest priority — background frame, not competitor
            )

    def get_self_anchor(self) -> Optional[WorkspaceItem]:
        """Return the current self-anchor item, or None if not set."""
        with self._lock:
            return self._self_anchor

    # ── Read ──────────────────────────────────────────────────────────────────

    def recent(self, n: int = 10) -> List[WorkspaceItem]:
        """
        Return the n most recent items, with the self-anchor prepended if set.
        The anchor is always position 0 — a stable identity background against
        which all other items appear.
        """
        with self._lock:
            items = list(self._buffer)[-n:]
            anchor = self._self_anchor
        if anchor is not None:
            return [anchor] + items
        return items

    def top(self, n: int = 5) -> List[WorkspaceItem]:
        """
        Return the n highest-priority items from the last 30 entries.
        The self-anchor is included in the candidate pool but will only
        surface if n is large (priority=0.0 keeps it as background).
        """
        with self._lock:
            window = list(self._buffer)[-30:]
            anchor = self._self_anchor
        candidates = ([anchor] if anchor else []) + window
        return sorted(candidates, key=lambda i: i.priority, reverse=True)[:n]

    def by_source(self, source: str, n: int = 5) -> List[WorkspaceItem]:
        """Return the n most recent items from a specific source."""
        if source == self.ANCHOR_SOURCE:
            with self._lock:
                return [self._self_anchor] if self._self_anchor else []
        with self._lock:
            items = [i for i in self._buffer if i.source == source]
        return items[-n:]

    def highest(self) -> Optional[WorkspaceItem]:
        """Return the single most-salient item in the last 30 entries."""
        items = self.top(1)
        return items[0] if items else None

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            # Note: self-anchor is intentionally NOT cleared — it survives resets

    # ── Listeners ─────────────────────────────────────────────────────────────

    def add_listener(self, callback: Callable) -> None:
        """Register a callback(item: WorkspaceItem) for new broadcasts."""
        self._listeners.append(callback)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def diversity_score(self) -> float:
        """
        Fraction of distinct sources among the last 10 items.
        0 = single source dominates, 1 = fully diverse.
        Used as a health indicator.
        """
        with self._lock:
            recent = list(self._buffer)[-10:]
        if not recent:
            return 0.0
        sources = {i.source for i in recent}
        return round(len(sources) / max(1, len(recent)), 3)

    def summary(self) -> Dict:
        with self._lock:
            items = list(self._buffer)
            anchor = self._self_anchor
        return {
            "total_items":   len(items),
            "sources":       list({i.source for i in items}),
            "diversity":     self.diversity_score(),
            "blocked_total": self._blocked_count,
            "self_anchor":   str(anchor.content)[:80] if anchor else None,
            "recent_top":    [
                i.to_dict()
                for i in sorted(items[-10:], key=lambda x: x.priority, reverse=True)[:3]
            ],
        }
