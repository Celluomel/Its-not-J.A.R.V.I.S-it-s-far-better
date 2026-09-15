"""
LUMINA — WORKSPACE BIAS + OVERRIDE-RATE LAYER
=============================================
Closes the actuator gap the audit found:

    workspace winner  ->  "[DOMINANT FOCUS] X" (prompt text)  ->  LLM

...so that the committed focus is a *computational* influence, not merely
a sentence the LLM may or may not obey:

  1. COMMIT   — when the workspace competition produces a winner, it becomes
                an active, EMBEDDED commitment (focus text + vector) and a
                `commitment` event in the causal ledger.

  2. BIAS     — downstream SELECTION is re-weighted TOWARD the committed
                focus. bias_multiplier(candidate) > 1.0 for aligned
                candidates. This is applied to real goal selection, so the
                focus causally steers WHICH goal gets actioned — attention as
                resource allocation, not metadata.

  3. OVERRIDE — after real behaviour (an autonomous action AND/OR the LLM's
                response), we measure cosine(focus, behaviour). Below
                threshold = the system overrode its own committed focus.
                That is RECORDED (ledger), COSTED (focus_stability drops),
                and AGGREGATED into a single override-rate metric.

THE METRIC THAT MATTERS:
  override_rate = n_overrides / n_checks.  As the bias does its job, this
  number should trend DOWN. That is causal closure, measured — not claimed.

DESIGN RULES (consistent with the ConsequenceTracker layer):
  - The bias is a bounded BIAS (boost aligned, never hard-exclude others) —
    preserves the multi-candidate dynamics the rubric values (EMERGENT-CAPABLE).
  - Override is SOFT-costed (focus_stability EMA), never destabilising.
  - Every external effect is try/except wrapped — never kills the loop.
  - Pure & testable: commit/bias/check_override are deterministic given an
    (injectable) embedder. Verified without the LLM.

Stores:
  data/persona/workspace_bias.json   (commitment + metrics, persisted)
  data/persona/causal_ledger.jsonl   (commitment / comply / override events)
"""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cognition.causal_ledger import CausalLedger
except Exception:  # pragma: no cover
    CausalLedger = None  # type: ignore


# ── tunables (bounded, conservative) ────────────────────────────────────────
BIAS_GAIN          = 0.40   # max selection boost for a fully aligned candidate
NEUTRAL_SIM        = 0.35   # bias only engages above this cosine similarity
OVERRIDE_THRESHOLD = 0.35   # below this similarity => the focus was overridden
STABILITY_ALPHA    = 0.25   # EMA rate for focus_stability (cost/recovery)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class WorkspaceBias:
    def __init__(self,
                 organism: Any = None,
                 persona_dir: Optional[str] = None,
                 embedder: Any = None,
                 ledger: Optional["CausalLedger"] = None,
                 enabled: bool = True):
        self.persona_dir = Path(persona_dir) if persona_dir else Path("data/persona")
        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._organism = organism
        self._lock = threading.Lock()
        self.path = self.persona_dir / "workspace_bias.json"

        # embedder: injectable (for tests) else pulled from the organism
        self._embedder = embedder
        if self._embedder is None and organism is not None:
            self._embedder = self._resolve_embedder(organism)

        self.ledger = ledger if ledger is not None else (
            CausalLedger(str(self.persona_dir)) if CausalLedger is not None else None
        )

        # live state
        self._active: Optional[Dict[str, Any]] = None   # current commitment
        self._state: Dict[str, Any] = self._load()

    # ── embedder ─────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_embedder(organism: Any):
        ai = getattr(organism, "ai_system", None)
        if ai is not None:
            emb = getattr(ai, "embedding_model", None)
            if emb is not None and hasattr(emb, "encode"):
                return emb
        return None

    def _encode(self, text: str) -> Optional[List[float]]:
        # lazily (re)resolve the embedder in case it wasn't ready when the
        # singleton was first created
        if self._embedder is None and self._organism is not None:
            self._embedder = self._resolve_embedder(self._organism)
        emb = self._embedder
        if emb is None or not text:
            return None
        try:
            v = emb.encode(text)
            return [float(x) for x in v]
        except Exception:
            return None

    # ── 1. COMMIT ────────────────────────────────────────────────────────────
    def commit(self, winner: Dict[str, Any], cycle_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Turn a workspace winner into an active, embedded commitment.
        `winner` is the dict from workspace competition: {name, type, score,
        label, topic, ...}.
        """
        if not self.enabled or not winner:
            return {"committed": False}
        focus_text = self._winner_text(winner)
        if not focus_text:
            return {"committed": False, "reason": "no focus text"}

        vec = self._encode(focus_text)
        commitment = {
            "focus_text": focus_text,
            "vector": vec,                 # may be None if no embedder (bias degrades to 1.0)
            "type": winner.get("type", ""),
            "name": winner.get("name", ""),
            "label": winner.get("label", ""),
            "score": float(winner.get("score", 0.0)),
            "cycle_id": cycle_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self._active = commitment
            self._state["commitments_total"] = self._state.get("commitments_total", 0) + 1
            self._persist()

        if self.ledger:
            self.ledger.record(
                kind="commitment",
                cause=f"workspace committed focus: {focus_text[:60]} (score={commitment['score']:.2f})",
                source="workspace_bias",
                cycle_id=cycle_id,
                output_state={"focus_text": focus_text, "type": commitment["type"],
                              "score": commitment["score"], "has_vector": vec is not None},
                confidence=commitment["score"],
                refs={"focus_text": focus_text, "type": commitment["type"]},
            )
        return {"committed": True, "focus_text": focus_text, "has_vector": vec is not None}

    @staticmethod
    def _winner_text(winner: Dict[str, Any]) -> str:
        for key in ("topic", "name", "label"):
            v = winner.get(key)
            if v and isinstance(v, str) and len(v.strip()) > 2:
                return v.strip()
        return ""

    def get_active(self) -> Optional[Dict[str, Any]]:
        return self._active

    # ── 2. BIAS ──────────────────────────────────────────────────────────────
    def bias_multiplier(self, candidate_text: str) -> float:
        """
        Bounded selection boost for a candidate aligned with the committed
        focus: 1.0 + GAIN * max(0, cos(candidate, focus) - NEUTRAL).
        Returns 1.0 (neutral) when there is no commitment / no vectors / no
        alignment. This is the REAL causal influence of the focus on selection.
        """
        if not self.enabled:
            return 1.0
        active = self._active
        if not active or active.get("vector") is None:
            return 1.0
        vec = self._encode(candidate_text)
        if vec is None:
            return 1.0
        sim = _cosine(vec, active["vector"])
        if sim <= NEUTRAL_SIM:
            return 1.0
        boost = BIAS_GAIN * (sim - NEUTRAL_SIM)
        return round(1.0 + max(0.0, boost), 4)

    # ── 3. OVERRIDE ──────────────────────────────────────────────────────────
    def check_override(self, actual_text: str, context: str = "",
                       cycle_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Measure whether real behaviour (an action's goal topic, or the LLM's
        response text) followed the committed focus. Below threshold => the
        system overrode its own focus: recorded, costed, aggregated.
        """
        if not self.enabled:
            return {"checked": False, "reason": "disabled"}
        active = self._active
        if not active or active.get("vector") is None or not actual_text:
            return {"checked": False, "reason": "no_commitment_or_vector"}

        vec = self._encode(actual_text)
        if vec is None:
            return {"checked": False, "reason": "encode_failed"}

        sim = round(_cosine(vec, active["vector"]), 4)
        overridden = sim < OVERRIDE_THRESHOLD

        # SOFT cost: focus_stability EMA (1.0 = fully sticks, drops on override)
        prev_stab = self._state.get("focus_stability", 1.0)
        target = 0.0 if overridden else 1.0
        new_stab = STABILITY_ALPHA * target + (1 - STABILITY_ALPHA) * prev_stab
        new_stab = round(new_stab, 4)

        with self._lock:
            self._state["checks_total"] = self._state.get("checks_total", 0) + 1
            self._state["similarity_sum"] = self._state.get("similarity_sum", 0.0) + sim
            if overridden:
                self._state["overrides_total"] = self._state.get("overrides_total", 0) + 1
            self._state["focus_stability"] = new_stab
            self._state["last_check"] = {
                "context": context, "similarity": sim, "overridden": overridden,
                "focus_text": active.get("focus_text", ""),
                "actual_text": (actual_text or "")[:80],
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            self._persist()

        if self.ledger:
            kind = "override" if overridden else "comply"
            self.ledger.record(
                kind=kind,
                cause=(f"{'OVERRIDES' if overridden else 'follows'} committed focus "
                       f"'{active.get('focus_text','')[:40]}' -> behaviour "
                       f"'{(actual_text or '')[:40]}' (sim={sim:.2f}, thr={OVERRIDE_THRESHOLD:.2f})"
                       + (f" [{context}]" if context else "")),
                source="workspace_bias",
                cycle_id=cycle_id,
                input_state={"focus_text": active.get("focus_text", "")},
                output_state={"actual_text": (actual_text or "")[:120], "similarity": sim},
                delta={"similarity": sim, "overridden": overridden,
                       "focus_stability": new_stab},
                confidence=active.get("score", 1.0),
                refs={"focus_text": active.get("focus_text", ""),
                      "context": context, "similarity": sim,
                      "overridden": overridden},
            )
        return {
            "checked": True,
            "overridden": overridden,
            "similarity": sim,
            "threshold": OVERRIDE_THRESHOLD,
            "focus_text": active.get("focus_text", ""),
            "focus_stability": new_stab,
        }

    # ── METRIC ───────────────────────────────────────────────────────────────
    def override_rate(self) -> Dict[str, Any]:
        s = dict(self._state)
        checks = s.get("checks_total", 0)
        over = s.get("overrides_total", 0)
        s["override_rate"] = round(over / checks, 4) if checks else 0.0
        s["mean_similarity"] = round(s.get("similarity_sum", 0.0) / checks, 4) if checks else 0.0
        s["focus_stability"] = s.get("focus_stability", 1.0)
        s["active_focus"] = (self._active or {}).get("focus_text")
        return s

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> Dict[str, Any]:
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text(encoding="utf-8"))
                return d if isinstance(d, dict) else {}
        except Exception:
            pass
        return {"commitments_total": 0, "checks_total": 0, "overrides_total": 0,
                "similarity_sum": 0.0, "focus_stability": 1.0}

    def _persist(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# ── process-wide singleton ──────────────────────────────────────────────────
_wb: Optional[WorkspaceBias] = None
_wb_lock = threading.Lock()


def get_workspace_bias(organism: Any = None,
                       persona_dir: Optional[str] = None,
                       embedder: Any = None) -> WorkspaceBias:
    global _wb
    with _wb_lock:
        if _wb is None:
            _wb = WorkspaceBias(organism=organism, persona_dir=persona_dir,
                                embedder=embedder)
        else:
            # keep the freshest organism so the embedder can be (re)resolved
            # lazily even if it wasn't ready at first creation
            if organism is not None:
                _wb._organism = organism
            if embedder is not None and _wb._embedder is None:
                _wb._embedder = embedder
        return _wb
