"""
cognition/temporal_self_projection.py  (v66)

Models how Lumina has changed over time and projects future trajectory.
Self-directed prediction about one's own future value configuration.

Three components:

1. TrajectoryTracker
   Reads DecisionPolicy weight history (from interaction updates) and
   CognitiveAuditEngine committed_params across slow cycles.
   Builds a time-series: value_weight[dimension][cycle] = float

2. TrajectoryAnalyser
   For each value dimension with >= MIN_DATAPOINTS measurements:
   - Compute trend: linear regression slope over last TREND_WINDOW cycles
   - Classify: rising / falling / stable / oscillating
   - Identify inflection points: where trend reversed
   Produces TrajectorySignature per dimension.

3. SelfProjector
   Given the trajectory signatures, projects where each dimension will
   be in PROJECTION_HORIZON slow cycles from now.
   Also generates a narrative: "Based on how I have changed across the
   last N cycles, I am becoming more X and less Y. If this trajectory
   continues, I will increasingly..."

   This is primitive but genuine: Lumina reasoning about future versions
   of itself from evidence of how it has changed, not from stated values.

Output injected into prompt as:
   [Self-trajectory] Becoming: curiosity↑ stability↓ | Projected: honesty stable
"""

from __future__ import annotations
import json, logging, threading, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_EVERY_N    = 100   # slow cycles between full projections
TREND_WINDOW       = 50    # cycles to analyse for trend
MIN_DATAPOINTS     = 8     # minimum points for reliable trend
PROJECTION_HORIZON = 50    # cycles ahead to project
SAVE_PATH          = "data/persona/temporal_self_projection.json"


@dataclass
class TrajectorySignature:
    dimension:   str
    trend:       str          # "rising"|"falling"|"stable"|"oscillating"
    slope:       float        # change per cycle
    current:     float
    projected:   float        # value at PROJECTION_HORIZON cycles
    confidence:  float
    datapoints:  int


@dataclass
class SelfTrajectory:
    signatures:        List[TrajectorySignature]
    narrative:         str
    computed_at_cycle: int
    computed_at:       float = field(default_factory=time.time)


class TemporalSelfProjection:
    def __init__(self, organism: Any, ai_system: Any, path: str = SAVE_PATH):
        self._organism    = organism
        self._ai          = ai_system
        self._path        = Path(path)
        self._lock        = threading.Lock()
        self._history:    Dict[str, List[Tuple[int,float]]] = {}
        self._trajectory: Optional[SelfTrajectory] = None
        self._load()
        logger.info(
            f"[TemporalSelfProjection] Init — "
            f"{len(self._history)} tracked dimensions"
        )

    def tick(self, slow_cycle: int) -> None:
        # Record every cycle (lightweight)
        self._record_snapshot(slow_cycle)
        # Full projection periodically
        if slow_cycle % PROJECT_EVERY_N == 0 and slow_cycle > 0:
            threading.Thread(
                target=self._project, args=(slow_cycle,),
                daemon=True, name="tsp-project"
            ).start()

    def prompt_fragment(self) -> str:
        if not self._trajectory: return ""
        sigs = self._trajectory.signatures
        rising  = [s.dimension for s in sigs if s.trend == "rising"  and s.slope > 0.002]
        falling = [s.dimension for s in sigs if s.trend == "falling" and s.slope < -0.002]
        if not rising and not falling: return ""
        parts = []
        if rising:  parts.append("becoming: " + ", ".join(f"{d}↑" for d in rising[:3]))
        if falling: parts.append("receding: " + ", ".join(f"{d}↓" for d in falling[:2]))
        return "[Self-trajectory] " + " | ".join(parts)

    def get_narrative(self) -> str:
        if not self._trajectory: return ""
        return self._trajectory.narrative

    def status(self) -> Dict:
        if not self._trajectory:
            return {"status": "building", "datapoints": sum(len(v) for v in self._history.values())}
        return {
            "dimensions_tracked": len(self._trajectory.signatures),
            "computed_at_cycle":  self._trajectory.computed_at_cycle,
            "rising":  [s.dimension for s in self._trajectory.signatures if s.trend == "rising"],
            "falling": [s.dimension for s in self._trajectory.signatures if s.trend == "falling"],
            "narrative_excerpt": self._trajectory.narrative[:120],
        }

    def _record_snapshot(self, slow_cycle: int) -> None:
        """Record current value weights. Lightweight — runs every cycle."""
        try:
            dp = getattr(self._ai, '_decision_policy', None)
            if not dp: return
            for dim, weight in dp.weights.items():
                if dim not in self._history:
                    self._history[dim] = []
                self._history[dim].append((slow_cycle, round(weight, 4)))
                # Keep only last TREND_WINDOW * 2 points
                if len(self._history[dim]) > TREND_WINDOW * 2:
                    self._history[dim] = self._history[dim][-(TREND_WINDOW * 2):]
        except Exception:
            pass

    def _project(self, slow_cycle: int) -> None:
        """Full projection: analyse trajectories and generate narrative."""
        try:
            signatures = []
            for dim, history in self._history.items():
                if len(history) < MIN_DATAPOINTS: continue
                recent = history[-TREND_WINDOW:]
                sig = self._analyse(dim, recent)
                if sig: signatures.append(sig)

            if not signatures: return

            narrative = self._generate_narrative(signatures, slow_cycle)

            trajectory = SelfTrajectory(
                signatures        = signatures,
                narrative         = narrative,
                computed_at_cycle = slow_cycle,
            )

            with self._lock:
                self._trajectory = trajectory

            self._save()
            logger.info(
                f"[TemporalSelfProjection] {len(signatures)} dimensions tracked. "
                f"Rising: {[s.dimension for s in signatures if s.trend=='rising']}"
            )
        except Exception as e:
            logger.debug(f"[TemporalSelfProjection] _project error: {e}")

    def _analyse(self, dim: str, history: List[Tuple[int,float]]) -> Optional[TrajectorySignature]:
        if len(history) < 2: return None
        cycles = [h[0] for h in history]
        values = [h[1] for h in history]
        n = len(cycles)

        # Linear regression
        mean_c = sum(cycles) / n
        mean_v = sum(values) / n
        num = sum((c - mean_c) * (v - mean_v) for c, v in zip(cycles, values))
        den = sum((c - mean_c) ** 2 for c in cycles)
        slope = num / den if den != 0 else 0.0

        current   = values[-1]
        projected = max(0.10, min(0.95, current + slope * PROJECTION_HORIZON))

        # Trend classification
        abs_slope = abs(slope)
        if abs_slope < 0.0001:
            trend = "stable"
        else:
            # Check for oscillation: sign changes in first differences
            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            sign_changes = sum(
                1 for i in range(len(diffs)-1)
                if diffs[i] * diffs[i+1] < 0
            )
            if sign_changes > len(diffs) * 0.4:
                trend = "oscillating"
            elif slope > 0.0001:
                trend = "rising"
            else:
                trend = "falling"

        # Confidence: R² of the linear fit
        ss_res = sum((v - (mean_v + slope*(c - mean_c)))**2 for c, v in zip(cycles, values))
        ss_tot = sum((v - mean_v)**2 for v in values)
        r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-9))

        return TrajectorySignature(
            dimension  = dim,
            trend      = trend,
            slope      = round(slope, 6),
            current    = round(current, 3),
            projected  = round(projected, 3),
            confidence = round(r2, 3),
            datapoints = len(history),
        )

    def _generate_narrative(
        self, signatures: List[TrajectorySignature], slow_cycle: int
    ) -> str:
        rising  = sorted([s for s in signatures if s.trend == "rising"],
                         key=lambda x: abs(x.slope), reverse=True)[:3]
        falling = sorted([s for s in signatures if s.trend == "falling"],
                         key=lambda x: abs(x.slope), reverse=True)[:2]
        stable  = [s for s in signatures if s.trend == "stable"][:2]

        if not rising and not falling:
            return "Value configuration is currently stable."

        context = (
            f"Value trajectory over the last {TREND_WINDOW} slow cycles:\n"
        )
        if rising:
            context += "Rising: " + ", ".join(
                f"{s.dimension} ({s.current:.2f}→~{s.projected:.2f})"
                for s in rising
            ) + "\n"
        if falling:
            context += "Falling: " + ", ".join(
                f"{s.dimension} ({s.current:.2f}→~{s.projected:.2f})"
                for s in falling
            ) + "\n"
        if stable:
            context += "Stable: " + ", ".join(s.dimension for s in stable) + "\n"

        prompt = (
            f"{context}\n"
            f"In 2-3 sentences, describe what this trajectory means for how "
            f"this AI system is developing.  What is it becoming?  "
            f"Write in first person, grounded in these numbers, not aspirationally. "
            f"Be specific about the direction of change."
        )
        try:
            from core.llm_scheduler import llm_scheduler
            res = ""
            with llm_scheduler.sync_slot(priority=5, skip_if_busy=True,
                                          caller="temporal_self_proj") as ok:
                if ok:
                    res = self._ai.get_response(
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=100, temperature=0.55,
                    ) or ""
            return res.strip()[:400]
        except Exception:
            return context.strip()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "history": {
                        dim: hist[-TREND_WINDOW:]
                        for dim, hist in self._history.items()
                    },
                    "trajectory": asdict(self._trajectory) if self._trajectory else None,
                    "_meta": {"version":"v66","ts":time.time()},
                }
            with open(self._path,"w") as f: json.dump(data,f,indent=2)
        except Exception as e:
            logger.debug(f"[TemporalSelfProjection] save error: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            raw_hist = data.get("history", {})
            self._history = {
                dim: [(h[0], h[1]) for h in hist]
                for dim, hist in raw_hist.items()
            }
            raw_traj = data.get("trajectory")
            if raw_traj:
                sigs = [TrajectorySignature(**s) for s in raw_traj.get("signatures", [])]
                self._trajectory = SelfTrajectory(
                    signatures        = sigs,
                    narrative         = raw_traj.get("narrative", ""),
                    computed_at_cycle = raw_traj.get("computed_at_cycle", 0),
                    computed_at       = raw_traj.get("computed_at", time.time()),
                )
        except Exception as e:
            logger.warning(f"[TemporalSelfProjection] load error: {e}")
