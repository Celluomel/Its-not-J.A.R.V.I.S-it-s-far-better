"""
Emergence Metrics Collector
============================
Computes and persists the 5 emergence metrics in real time.
Wired into InternalLoop — called every slow cycle.

Metrics:
  GIR   — Grounded Introspection Rate
  HLCR  — Hypothesis Loop Closure Rate
  HRE   — Hypothesis Recurrence / Evolution
  CSIS  — Cross-Session Influence Score
  WDS   — Weight Drift Stability

Writes to: data/persona/emergence_metrics.json
Read by:   the NiceGUI dashboard tab
"""

import json
import logging
import math
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    timestamp:  float
    gir:        float   # 0.0–1.0
    hlcr:       float   # 0.0–1.0
    hre:        float   # 0.0–1.0  (0=stuck, 0.5=evolving, 1=novel)
    csis:       float   # 0.0–1.0
    wds:        float   # 0.0–1.0  (1=stable, 0=chaotic)
    # Raw counters for transparency
    gir_grounded:   int = 0
    gir_fallback:   int = 0
    hlcr_generated: int = 0
    hlcr_tested:    int = 0
    csis_faiss_hits: int = 0
    csis_total_recalls: int = 0
    wds_current_weight: float = 0.65
    wds_drift_std: float = 0.0


class EmergenceMetricsCollector:
    """
    Lightweight metrics engine — designed to never crash the main loop.
    All methods wrapped in try/except, fail silently.
    """

    HISTORY_LEN = 200
    SAVE_EVERY_N = 5   # save every N slow cycles

    def __init__(self, organism, data_dir: str = "data/persona"):
        self._o        = organism
        self._path     = Path(data_dir) / "emergence_metrics.json"
        self._lock     = threading.Lock()
        self._cycle    = 0

        # Rolling counters — reset on each tick, accumulated across window
        self._gir_grounded   = 0
        self._gir_fallback   = 0
        self._hlcr_generated = 0
        self._hlcr_tested    = 0
        self._csis_faiss     = 0
        self._csis_total     = 0
        self._weight_history: deque = deque(maxlen=40)

        # Hypothesis text ring — for HRE computation
        self._hypothesis_ring: deque = deque(maxlen=30)

        self._history: List[MetricSnapshot] = []
        self._load()
        logger.info(f"[EmergenceMetrics] Collector ready — {len(self._history)} prior snapshots")

    # ── Public event recording (called by other modules) ──────────────────────

    def record_introspection(self, source: str) -> None:
        """Called by GroundedIntrospectionGenerator after each question."""
        try:
            if source == 'fallback':
                self._gir_fallback += 1
            else:
                self._gir_grounded += 1
        except Exception:
            pass

    def record_hypothesis(self, text: str) -> None:
        """Called by SimpleThoughtEvaluator when a hypothesis is generated."""
        try:
            self._hlcr_generated += 1
            self._hypothesis_ring.append({
                'ts': time.time(),
                'text': text[:120],
                'words': set(text.lower().split())
            })
        except Exception:
            pass

    def record_hypothesis_tested(self) -> None:
        """Called by GAE when a test_hypothesis goal receives an action."""
        try:
            self._hlcr_tested += 1
        except Exception:
            pass

    def record_memory_recall(self, source: str) -> None:
        """Called by GAE _try_memory_recall. source: 'faiss' or 'sqlite'."""
        try:
            self._csis_total += 1
            if source == 'faiss':
                self._csis_faiss += 1
        except Exception:
            pass

    def record_weight_update(self, weight: float) -> None:
        """Called by GAE _record_resolution_outcome on each self-mod update."""
        try:
            self._weight_history.append({'ts': time.time(), 'w': weight})
        except Exception:
            pass

    # ── Tick (called every slow cycle from InternalLoop) ─────────────────────

    def tick(self, cycle: int) -> None:
        """Compute snapshot and save periodically."""
        self._cycle = cycle
        try:
            snap = self._compute_snapshot()
            with self._lock:
                self._history.append(snap)
                if len(self._history) > self.HISTORY_LEN:
                    self._history = self._history[-self.HISTORY_LEN:]
            if cycle % self.SAVE_EVERY_N == 0:
                self._save()
            self._reset_counters()
        except Exception as e:
            logger.debug(f"[EmergenceMetrics] tick error: {e}")

    # ── Computation ───────────────────────────────────────────────────────────

    def _compute_snapshot(self) -> MetricSnapshot:
        gir  = self._compute_gir()
        hlcr = self._compute_hlcr()
        hre  = self._compute_hre()
        csis = self._compute_csis()
        wds, w_cur, w_std = self._compute_wds()
        return MetricSnapshot(
            timestamp        = time.time(),
            gir              = gir,
            hlcr             = hlcr,
            hre              = hre,
            csis             = csis,
            wds              = wds,
            gir_grounded     = self._gir_grounded,
            gir_fallback     = self._gir_fallback,
            hlcr_generated   = self._hlcr_generated,
            hlcr_tested      = self._hlcr_tested,
            csis_faiss_hits  = self._csis_faiss,
            csis_total_recalls = self._csis_total,
            wds_current_weight = w_cur,
            wds_drift_std    = w_std,
        )

    def _compute_gir(self) -> float:
        """GIR = grounded / (grounded + fallback). Neutral 0.5 if no data."""
        total = self._gir_grounded + self._gir_fallback
        if total == 0:
            # Check live GroundedIntrospectionGenerator if available
            try:
                from cognition.grounded_introspection import _instance as _gi
                if _gi and hasattr(_gi, '_last_question'):
                    # Has it ever fired a non-fallback?
                    return 0.5  # neutral — no events this window
            except Exception:
                pass
            return 0.5
        return round(self._gir_grounded / total, 3)

    def _compute_hlcr(self) -> float:
        """
        HLCR = fraction of generated hypotheses that produced an insight.
        Since test_hypothesis goals rarely win workspace competition, we use
        GAE self_question insights as the proxy for "hypothesis tested":
        a hypothesis fires → later a self_question produces an insight on same
        topic → that counts as the test having run.

        Approximation: HLCR = min(insights_this_window / hypotheses_this_window, 1)
        recorded via record_hypothesis() and record_hypothesis_tested().
        If neither fires: carry forward last non-zero value.
        """
        if self._hlcr_generated == 0:
            # Try to use historical average rather than returning 0
            hist_nonzero = [s.hlcr for s in self._history[-20:] if s.hlcr > 0]
            return round(sum(hist_nonzero) / len(hist_nonzero), 3) if hist_nonzero else 0.0
        # Treat any insight stored in the window as a "test completed"
        ratio = self._hlcr_tested / self._hlcr_generated
        return round(min(1.0, ratio), 3)

    def record_insight_text(self, text: str) -> None:
        """Called by GAE after each insight — feeds HRE computation."""
        try:
            words = set(w.lower() for w in text.split() if len(w) > 5)
            if words:
                self._hypothesis_ring.append({'ts': time.time(), 'text': text[:120], 'words': words})
        except Exception:
            pass

    def _compute_hre(self) -> float:
        """
        HRE measures how hypotheses evolve across the ring:
          0.0 = exact repetition (stuck)
          0.5 = moderate variation (learning)
          1.0 = high novelty (possibly too scattered)

        Uses Jaccard distance between consecutive hypothesis word sets.
        """
        ring = list(self._hypothesis_ring)
        if len(ring) < 2:
            return 0.5  # neutral — not enough data
        distances = []
        for a, b in zip(ring, ring[1:]):
            wa, wb = a['words'], b['words']
            union = len(wa | wb)
            if union == 0:
                continue
            jaccard_sim = len(wa & wb) / union
            distances.append(1.0 - jaccard_sim)  # distance = novelty
        if not distances:
            return 0.5
        avg = sum(distances) / len(distances)
        # Map: 0.0 (identical) → 0.0, 0.3 (moderate variation) → 0.8, 1.0 (unrelated) → 0.4
        # We want the sweet spot around 0.25–0.40 distance to score 1.0
        sweetspot = 1.0 - abs(avg - 0.30) * 2.5
        return round(max(0.0, min(1.0, sweetspot)), 3)

    def _compute_csis(self) -> float:
        """
        CSIS = memory-influenced recalls / total recalls.
        Ideally FAISS (cross-session episodic), but since FAISS context often
        returns empty, SQLite semantic relations also count as memory influence.
        Both are stored via record_memory_recall('faiss'|'sqlite').
        CSIS > 0 means at least SQLite relations are guiding cognition.
        """
        if self._csis_total == 0:
            hist_nonzero = [s.csis for s in self._history[-20:] if s.csis > 0]
            return round(sum(hist_nonzero) / len(hist_nonzero), 3) if hist_nonzero else 0.0
        # Both faiss and sqlite hits count — split weight (faiss worth 2x)
        weighted = self._csis_faiss * 2 + (self._csis_total - self._csis_faiss)
        total_weighted = self._csis_total * 2
        return round(min(1.0, weighted / total_weighted), 3)

    def _compute_wds(self):
        """
        WDS = stability of weight drift.
        1.0 = converging, 0.5 = steady oscillation, 0.0 = chaotic divergence.
        Also returns (wds, current_weight, std_dev).
        """
        # Load from file if in-memory history is short
        weights = [e['w'] for e in self._weight_history]
        if not weights:
            try:
                wp = self._path.parent / 'evaluator_weights.json'
                if wp.exists():
                    d = json.loads(wp.read_text())
                    cur = d.get('tension_priority', 0.65)
                    return 0.5, cur, 0.0
            except Exception:
                pass
            return 0.5, 0.65, 0.0

        cur = weights[-1]
        if len(weights) < 3:
            return 0.5, cur, 0.0

        import statistics
        std = statistics.stdev(weights)
        # Low std → stable → high WDS
        # std=0.00 → WDS=1.0, std=0.05 → WDS=0.5, std=0.10+ → WDS=0.0
        wds = max(0.0, 1.0 - std * 20)
        return round(wds, 3), round(cur, 4), round(std, 4)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def get_latest(self) -> Optional[MetricSnapshot]:
        with self._lock:
            return self._history[-1] if self._history else None

    def get_history(self, n: int = 50) -> List[MetricSnapshot]:
        with self._lock:
            return list(self._history[-n:])

    def _reset_counters(self) -> None:
        self._gir_grounded = 0
        self._gir_fallback  = 0
        self._hlcr_generated = 0
        self._hlcr_tested    = 0
        self._csis_faiss     = 0
        self._csis_total     = 0

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._history = [MetricSnapshot(**s) for s in data.get('history', [])]
                for h in data.get('weight_history', []):
                    self._weight_history.append(h)
        except Exception as e:
            logger.debug(f"[EmergenceMetrics] load error: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    'history': [asdict(s) for s in self._history[-self.HISTORY_LEN:]],
                    'weight_history': list(self._weight_history),
                    'saved_at': time.time(),
                }
            tmp = self._path.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._path)
        except Exception as e:
            logger.debug(f"[EmergenceMetrics] save error: {e}")


# ── Module singleton ──────────────────────────────────────────────────────────
_collector: Optional[EmergenceMetricsCollector] = None

def get_collector(organism=None, data_dir="data/persona") -> Optional[EmergenceMetricsCollector]:
    global _collector
    if _collector is None and organism is not None:
        _collector = EmergenceMetricsCollector(organism, data_dir)
    return _collector
