"""
cognition/calibration_engine.py  — Phase 3.1

CalibrationEngine: closes the predicted-confidence → actual-outcome loop
that has never existed anywhere in this codebase.

PredictiveConsequenceModel._predict() computes a genuinely principled
confidence value (inverse k-NN distance over similar past actions), but
that confidence is never checked against what actually happened. This
engine logs every prediction as "pending", resolves it once the matching
ConsequenceRecord outcome is known, and bins resolved predictions by their
confidence range to compute EMPIRICAL accuracy per bin — standard
reliability-diagram / Expected Calibration Error (ECE) methodology.

Scope (explicit, see PHASE_3_1_CALIBRATION_UNCERTAINTY.md): PCM only, in
this pass. WorldModel and CausalMechanismModel are deliberately deferred —
neither currently commits to a single, timestamped (prediction, confidence)
record that pairs cleanly with one resolvable outcome.

This module does NOT change PCM's prediction formula. It does NOT yet
rewire Phase 2.9.1's reinforcement signal to use calibrated confidence
instead of raw outcome — that is an explicit Phase 3.1.1 follow-on once
calibration data actually exists to rewire against.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CALIBRATION_PATH   = "data/persona/calibration_state.json"
N_BINS             = 10        # standard reliability-diagram bin count
MIN_BIN_SAMPLES    = 5         # minimum resolved predictions before a bin is trusted
PENDING_TTL_SECS   = 600.0     # discard unresolved predictions older than this (10 min)
MAX_PENDING        = 50        # cap pending queue size


def _bin_for(confidence: float) -> str:
    """Map a confidence value in [0,1] to one of N_BINS reliability bins."""
    c = max(0.0, min(0.999, confidence))
    idx = int(c * N_BINS)
    lo  = idx / N_BINS
    hi  = (idx + 1) / N_BINS
    return f"[{lo:.1f}-{hi:.1f})"


def _bin_midpoint(bin_key: str) -> float:
    """Recover the midpoint confidence value for a bin key."""
    try:
        inner = bin_key.strip("[)")
        lo_s, hi_s = inner.split("-")
        return (float(lo_s) + float(hi_s)) / 2.0
    except Exception:
        return 0.5


class CalibrationEngine:
    """
    Tracks predicted confidence vs actual outcome correctness for PCM
    predictions, binned by confidence range, to compute empirical
    calibration accuracy and Expected Calibration Error (ECE) per module.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = Path(CALIBRATION_PATH)
        # pending: list of {module, key, predicted_confidence, direction, ts}
        # 'direction' is the predicted sign of the outcome ("positive"/"negative"/"neutral")
        # so resolution can check whether the actual outcome matched it.
        self._pending: List[Dict] = []
        # bins: {module: {bin_key: {"predicted_n": int, "correct_n": int, "accuracy": float}}}
        self._bins: Dict[str, Dict[str, Dict]] = {}
        self._global_ece: Dict[str, float] = {}
        self._load()
        logger.info(
            f"[CalibrationEngine] Initialised — "
            f"{len(self._pending)} pending, "
            f"{sum(len(b) for b in self._bins.values())} resolved bins"
        )

    # ── Logging a prediction (before outcome is known) ──────────────────────────

    def log_prediction(self, module: str, key: str,
                        predicted_confidence: float,
                        predicted_direction: str = "neutral") -> None:
        """
        Record a prediction that will later be resolved against an actual
        outcome. `key` identifies what's being predicted (e.g. action_type)
        so the matching resolution can find it. `predicted_direction` is
        the expected outcome sign — resolution checks whether the actual
        outcome's direction matches.
        """
        with self._lock:
            self._pending.append({
                "module": module,
                "key": key,
                "predicted_confidence": round(float(predicted_confidence), 4),
                "predicted_direction": predicted_direction,
                "logged_at": time.time(),
            })
            # Cap pending queue — oldest entries dropped if it grows unbounded
            # (e.g. predictions that are never resolved because record_outcome
            # was never called for that action_type).
            if len(self._pending) > MAX_PENDING:
                self._pending = self._pending[-MAX_PENDING:]
        self._save()

    # ── Resolving a prediction (once outcome is known) ──────────────────────────

    def resolve_prediction(self, module: str, key: str,
                            actual_direction: str) -> Optional[bool]:
        """
        Find the most recent matching pending prediction for (module, key)
        and resolve it: was the predicted direction correct?
        Returns True/False if a match was resolved, None if no pending
        prediction was found (nothing to resolve — not an error).
        """
        now = time.time()
        with self._lock:
            # Find most recent matching pending entry, prune stale ones along the way
            match_idx = None
            for i in range(len(self._pending) - 1, -1, -1):
                entry = self._pending[i]
                age = now - entry["logged_at"]
                if entry["module"] == module and entry["key"] == key and age <= PENDING_TTL_SECS:
                    match_idx = i
                    break

            # Prune stale entries regardless of whether we found a match
            self._pending = [
                e for e in self._pending
                if (now - e["logged_at"]) <= PENDING_TTL_SECS
            ]

            if match_idx is None:
                return None

            entry = self._pending.pop(match_idx) if match_idx < len(self._pending) else None
            # Re-find after prune (indices may have shifted) — safer to re-search
            if entry is None:
                for i, e in enumerate(self._pending):
                    if e["module"] == module and e["key"] == key:
                        entry = self._pending.pop(i)
                        break
            if entry is None:
                return None

            was_correct = (entry["predicted_direction"] == actual_direction)
            bin_key = _bin_for(entry["predicted_confidence"])

            module_bins = self._bins.setdefault(module, {})
            bin_data = module_bins.setdefault(bin_key, {
                "predicted_n": 0, "correct_n": 0, "accuracy": 0.0
            })
            bin_data["predicted_n"] += 1
            if was_correct:
                bin_data["correct_n"] += 1
            bin_data["accuracy"] = round(
                bin_data["correct_n"] / bin_data["predicted_n"], 4
            )

        self._save()
        logger.debug(
            f"[CalibrationEngine] Resolved {module}/{key}: "
            f"predicted={entry['predicted_direction']}({entry['predicted_confidence']:.2f}) "
            f"actual={actual_direction} correct={was_correct}"
        )
        return was_correct

    # ── Querying calibrated confidence ───────────────────────────────────────────

    def calibrated_confidence(self, module: str, raw_confidence: float) -> Tuple[float, bool]:
        """
        Return (calibrated_confidence, is_calibrated).
        If insufficient resolved data exists for this confidence range,
        returns (raw_confidence, False) — caller should treat it as
        unverified. Once MIN_BIN_SAMPLES resolutions exist in the matching
        bin, returns (empirical_accuracy, True) — the EMPIRICAL rate at
        which predictions in this confidence range were actually correct.
        """
        bin_key = _bin_for(raw_confidence)
        bin_data = self._bins.get(module, {}).get(bin_key)
        if bin_data is None or bin_data["predicted_n"] < MIN_BIN_SAMPLES:
            return raw_confidence, False
        return bin_data["accuracy"], True

    # ── Expected Calibration Error ───────────────────────────────────────────────

    def compute_ece(self, module: str) -> Optional[float]:
        """
        Expected Calibration Error: weighted mean |bin_midpoint - bin_accuracy|
        across all bins with sufficient samples. Lower = better calibrated.
        Returns None if no bins have enough samples yet.
        """
        module_bins = self._bins.get(module, {})
        eligible = {
            k: v for k, v in module_bins.items()
            if v["predicted_n"] >= MIN_BIN_SAMPLES
        }
        if not eligible:
            return None

        total_n = sum(v["predicted_n"] for v in eligible.values())
        if total_n == 0:
            return None

        weighted_error = sum(
            v["predicted_n"] * abs(_bin_midpoint(k) - v["accuracy"])
            for k, v in eligible.items()
        )
        ece = round(weighted_error / total_n, 4)
        self._global_ece[module] = ece
        return ece

    def summary(self) -> Dict:
        """Dashboard-facing summary: bins, ECE, pending count per module."""
        result = {}
        for module in set(list(self._bins.keys()) + list(self._global_ece.keys())):
            ece = self.compute_ece(module)
            bins = self._bins.get(module, {})
            total_resolved = sum(v["predicted_n"] for v in bins.values())
            result[module] = {
                "ece": ece,
                "total_resolved": total_resolved,
                "bins": bins,
            }
        return {
            "modules": result,
            "pending_count": len(self._pending),
            "last_updated": time.time(),
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                d = json.loads(self._path.read_text())
                self._pending = d.get("pending", [])
                self._bins    = d.get("bins", {})
                self._global_ece = d.get("global_ece", {})
        except Exception as e:
            logger.debug(f"[CalibrationEngine] load error: {e}")

    def _save(self) -> None:
        try:
            payload = {
                "pending": self._pending,
                "bins": self._bins,
                "global_ece": self._global_ece,
                "last_written": time.time(),
            }
            _tmp_path = self._path.with_suffix('.json.tmp')
            _tmp_path.write_text(json.dumps(payload, indent=2))
            _tmp_path.replace(self._path)  # atomic on POSIX — never a torn read
        except Exception as e:
            logger.debug(f"[CalibrationEngine] save error: {e}")
