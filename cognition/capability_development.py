"""Bounded, local experiments for developing observable organism capabilities.

This engine deliberately measures the Global Workspace instead of the chat
path.  A proposal, an active experiment and a verified capability are kept as
separate persisted states so an inconclusive trial cannot masquerade as a
working capability.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

BASELINE_STEPS = 3
TRIAL_STEPS = 6
MIN_IMPROVEMENT = 0.05


class CapabilityDevelopmentEngine:
    """Run a reversible Workspace observability experiment in the slow loop."""

    def __init__(self, organism: Any, path: str = "data/persona/capability_development.json"):
        self._organism = organism
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            "proposal": None,
            "experiment": None,
            "verified_capability": None,
            "history": [],
        }
        self._load()

    def start(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Accept one experiment proposal and begin a fresh baseline."""
        with self._lock:
            existing = self._state.get("experiment")
            if existing and existing.get("status") in {"baseline", "running"}:
                raise ValueError("A capability experiment is already active.")
            now = time.time()
            proposal_copy = dict(proposal)
            proposal_copy["status"] = "accepted"
            proposal_copy["accepted_at"] = now
            experiment = {
                "id": str(uuid.uuid4()),
                "proposal_id": proposal_copy.get("id"),
                "title": proposal_copy.get("title", "Capability development"),
                "subsystem": "workspace",
                "status": "baseline",
                "baseline": [],
                "trials": [],
                "baseline_score": None,
                "trial_score": None,
                "result": None,
                "started_at": now,
                "updated_at": now,
                "max_baseline_steps": BASELINE_STEPS,
                "max_trial_steps": TRIAL_STEPS,
            }
            self._state["proposal"] = proposal_copy
            self._state["experiment"] = experiment
            self._state["verified_capability"] = None
            self._save()
            return self.snapshot()

    def tick(self, slow_cycle: int) -> None:
        """Advance one bounded observation; never performs generative work."""
        if slow_cycle < 1 or slow_cycle % 2:
            return
        try:
            with self._lock:
                experiment = self._state.get("experiment")
                if not experiment or experiment.get("status") not in {"baseline", "running"}:
                    return
                phase = experiment["status"]

            before = self._observe()
            if phase == "baseline":
                with self._lock:
                    experiment["baseline"].append(before)
                    if len(experiment["baseline"]) >= experiment["max_baseline_steps"]:
                        experiment["baseline_score"] = round(self._mean_score(experiment["baseline"]), 4)
                        experiment["status"] = "running"
                        logger.info("[CapabilityExperiment] baseline complete: %.3f", experiment["baseline_score"])
                self._save()
                return

            # Measurement-only trials are intentional: artificial broadcasts
            # would change the very Workspace signal being evaluated.
            admitted = None
            after = self._observe()
            trial = {
                "step": len(experiment["trials"]) + 1,
                "slow_cycle": slow_cycle,
                "admitted": admitted,
                "before": before,
                "after": after,
                "score": round(after["score"], 4),
                "delta": round(after["score"] - before["score"], 4),
                "timestamp": time.time(),
            }
            with self._lock:
                experiment["trials"].append(trial)
                if len(experiment["trials"]) >= experiment["max_trial_steps"]:
                    self._finish_locked(experiment)
                experiment["updated_at"] = time.time()
            self._save()
        except Exception as exc:
            logger.debug("[CapabilityExperiment] tick failed: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state, ensure_ascii=False))

    def _observe(self) -> Dict[str, Any]:
        workspace = getattr(self._organism, "workspace", None)
        summary = workspace.summary() if workspace and hasattr(workspace, "summary") else {}
        sources = summary.get("sources", []) if isinstance(summary, dict) else []
        diversity = float(summary.get("diversity", 0.0) or 0.0) if isinstance(summary, dict) else 0.0
        total = float(summary.get("total_items", 0) or 0) if isinstance(summary, dict) else 0.0
        # A bounded composite signal: source diversity plus a saturated activity term.
        score = max(0.0, min(1.0, 0.7 * diversity + 0.3 * min(total / 10.0, 1.0)))
        return {"score": round(score, 4), "diversity": round(diversity, 4), "total_items": int(total), "sources": list(sources)[:20]}

    @staticmethod
    def _mean_score(observations) -> float:
        return sum(float(item.get("score", 0.0)) for item in observations) / max(1, len(observations))

    def _finish_locked(self, experiment: Dict[str, Any]) -> None:
        baseline = float(experiment.get("baseline_score") or 0.0)
        trials = experiment.get("trials", [])
        trial_score = sum(float(item.get("score", 0.0)) for item in trials) / max(1, len(trials))
        admitted_values = [item.get("admitted") for item in trials if item.get("admitted") is not None]
        admitted = sum(1 for value in admitted_values if value)
        improvement = trial_score - baseline
        verified = improvement >= MIN_IMPROVEMENT and (
            not admitted_values or admitted >= max(1, len(admitted_values) // 2)
        )
        experiment["trial_score"] = round(trial_score, 4)
        experiment["result"] = {
            "baseline_score": round(baseline, 4),
            "trial_score": round(trial_score, 4),
            "improvement": round(improvement, 4),
            "admitted_trials": admitted if admitted_values else None,
            "trial_count": len(trials),
            "evidence": "Workspace diversity/activity comparison across bounded repeated observations.",
        }
        experiment["status"] = "verified" if verified else "inconclusive"
        experiment["completed_at"] = time.time()
        if verified:
            self._state["verified_capability"] = {
                "name": experiment["title"],
                "subsystem": "workspace",
                "status": "verified",
                "evidence": experiment["result"],
                "verified_at": time.time(),
            }
        self._state["history"].append(dict(experiment))
        self._state["history"] = self._state["history"][-20:]
        logger.info("[CapabilityExperiment] %s: baseline=%.3f trials=%.3f improvement=%+.3f", experiment["status"], baseline, trial_score, improvement)

    def _load(self) -> None:
        try:
            if self._path.exists():
                value = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    self._state.update(value)
        except Exception as exc:
            logger.warning("[CapabilityExperiment] load failed: %s", exc)

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temp = self._path.with_suffix(".tmp")
                temp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
                temp.replace(self._path)
        except Exception as exc:
            logger.warning("[CapabilityExperiment] save failed: %s", exc)
