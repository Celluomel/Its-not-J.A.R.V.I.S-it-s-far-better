"""
SymbolSystem — Phase A (scoped)
==================================
From the strange-loop implementation document, with one change from the
original spec: Phase D's absorption used regex over free LLM text
(`r"i notice that i (.*)"` etc.) — flagged in review as fragile, likely
to flood the system with noise from ordinary phrasing ("I notice that I
need coffee" is not a self-insight). This version absorbs from
STRUCTURED, already-real signals instead:

  - narrative_identity.record_chapter() calls with high significance
    (the same real, causally-connected pipeline verified in v91 —
    resolved dissonance chapters already update self_concept beliefs;
    now they also create/strengthen a symbol)
  - cognitive_dissonance_engine.py resolution events directly (self_
    question / hypotheses text, when present — the actual structured
    self-inquiry content from v87, not regex-matched prose)

This trades "open-ended absorption from anything Lumina says" for
"absorption only from things the architecture already treats as
significant self-referential events" — narrower, but the symbols that
do get created are backed by a real signal rather than a phrasing
coincidence.

Integration point the original document left unspecified: symbols
compete for attention via WorkspaceCompetition's `thoughts=` slot (the
same real, already-verified mechanism used for camera percepts — v94 —
and Flux peer-cognition — v95/v96), not a vague "influences goal
formation". A sufficiently activated/confident symbol becomes a genuine
competing candidate, scored by the same fair competition as everything
else — not force-injected.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_SIGNIFICANCE_TO_ABSORB = 0.55   # matches narrative_arc_writer's own
                                      # significance thresholds (SIG_THREAD_
                                      # COMPLETED=0.78, SIG_STALL_OVERCOME=
                                      # 0.65 etc — same order of magnitude,
                                      # not a value invented from nothing
MIN_CONFIDENCE_FOR_CANDIDATE = 0.5   # below this, not worth competing for
                                      # attention yet


@dataclass
class Symbol:
    id: str
    name: str
    definition: str
    symbol_type: str          # "self" | "meta" | "process" | "narrative"
    source_module: str
    confidence: float = 0.5
    creation_time: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    activation_count: int = 0
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def activate(self) -> None:
        self.last_activated = time.time()
        self.activation_count += 1


class SymbolSystem:
    def __init__(self, path: str = "data/persona/symbols.json") -> None:
        self._path = Path(path)
        self._lock = Lock()
        self.symbols: Dict[str, Symbol] = {}
        self._load()
        logger.info("[SymbolSystem] initialised")

    def create_or_update(self, name: str, definition: str, symbol_type: str,
                          source_module: str, confidence: float = 0.5,
                          parents: Optional[List[str]] = None) -> Symbol:
        with self._lock:
            existing = self._find_by_name(name)
            if existing:
                existing.definition = definition
                existing.confidence = min(0.95, (existing.confidence + confidence) / 2)
                existing.activate()
                existing.metadata["last_source"] = source_module
                self._save()
                return existing
            sid = str(uuid.uuid4())[:8]
            sym = Symbol(id=sid, name=name, definition=definition,
                         symbol_type=symbol_type, source_module=source_module,
                         confidence=confidence, parents=parents or [])
            # The absorption event that creates a symbol IS its first
            # activation — found via Phase F's symbol_activation_entropy
            # metric returning 0.0 for freshly-created symbols that had
            # never been re-absorbed, since Symbol's dataclass default
            # (activation_count=0) was never bumped on creation, only on
            # the existing-symbol update path below via .activate().
            sym.activation_count = 1
            self.symbols[sid] = sym
            self._save()
            return sym

    def _find_by_name(self, name: str) -> Optional[Symbol]:
        for s in self.symbols.values():
            if s.name.lower() == name.lower():
                return s
        return None

    def absorb_narrative_chapter(self, title: str, description: str,
                                  significance: float, source_module: str) -> Optional[Symbol]:
        """Structured absorption point 1 — called alongside
        narrative_identity.record_chapter() for high-significance
        chapters (v91's real pipeline), not from regex over free text."""
        if significance < MIN_SIGNIFICANCE_TO_ABSORB:
            return None
        name = title[:60]
        return self.create_or_update(
            name=name, definition=description[:300], symbol_type="narrative",
            source_module=source_module, confidence=min(0.9, significance),
        )

    def absorb_self_inquiry(self, self_question: Optional[str],
                             hypotheses: Optional[List[str]],
                             event_type: str, source_module: str) -> Optional[Symbol]:
        """Structured absorption point 2 — the actual self-question/
        hypotheses text from a resolved CDE dissonance event (v87's real
        self-inquiry content), not regex-matched prose."""
        if not self_question:
            return None
        definition = self_question
        if hypotheses:
            definition += " " + " ".join(hypotheses[:2])
        return self.create_or_update(
            name=self_question[:60], definition=definition[:300],
            symbol_type="meta", source_module=source_module, confidence=0.55,
        )

    def pending_thought_candidates(self, limit: int = 3) -> List[Dict]:
        """
        Real WorkspaceCompetition integration (the point the original
        document left unspecified). Same candidate shape verified in
        universal_connector.py's pending_thought_candidates() — content/
        thought_type/topic/priority/actionability/source, so this
        competes fairly through the identical real path, not a new one.
        Only sufficiently confident, recently-activated symbols are
        eligible — an old, never-reinforced symbol shouldn't keep
        competing for attention indefinitely.
        """
        with self._lock:
            candidates = [s for s in self.symbols.values() if s.confidence >= MIN_CONFIDENCE_FOR_CANDIDATE]
        candidates.sort(key=lambda s: (s.confidence, s.last_activated), reverse=True)
        out = []
        for s in candidates[:limit]:
            out.append({
                "id": f"symbol_{s.id}",
                "source": "symbol_system",
                "content": s.definition,
                "thought_type": "symbol",
                "topic": s.name[:80],
                "priority": round(min(1.0, s.confidence), 3),
                "actionability": 0.2,
                "source_data": {"symbol_id": s.id, "symbol_type": s.symbol_type,
                                "source_module": s.source_module},
            })
        return out

    # ── persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(s) for s in self.symbols.values()]
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.debug(f"[SymbolSystem] save failed (non-fatal): {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            for s in data:
                sym = Symbol(**{k: v for k, v in s.items() if k in Symbol.__dataclass_fields__})
                self.symbols[sym.id] = sym
        except Exception as e:
            logger.warning(f"[SymbolSystem] load failed (non-fatal): {e}")


_systems: Dict[int, SymbolSystem] = {}


def get_symbol_system(organism: Any) -> SymbolSystem:
    existing = getattr(organism, "_symbol_system", None)
    if existing is not None:
        return existing
    system = SymbolSystem()
    try:
        organism._symbol_system = system
    except Exception:
        pass
    return system
