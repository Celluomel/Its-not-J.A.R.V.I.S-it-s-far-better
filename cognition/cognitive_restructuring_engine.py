"""Multi-causal restructuring above the per-turn self-correction loop."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CognitiveRestructuringEngine:
    """Cluster persistent tensions and test competing correction strategies."""

    def __init__(self, organism: Any, path: str = "data/persona/cognitive_restructuring.json") -> None:
        self._organism = organism
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            "version": 1, "clusters": [], "active_plan": None, "history": []
        }
        self._load()

    def tick(self, cycle_id: int) -> None:
        """Run infrequently in the background; no LLM call is made here."""
        if cycle_id <= 0 or cycle_id % 5:
            return
        with self._lock:
            signals = self._collect_signals()
            if len(signals) < 2:
                return
            clusters = self._cluster(signals)
            self._state["clusters"] = clusters[-12:]
            plan = self._state.get("active_plan")
            if plan is None:
                candidate = max(clusters, key=lambda c: (len(c["signals"]), c["strength"]))
                if len(candidate["signals"]) >= 2:
                    self._state["active_plan"] = self._make_plan(candidate, cycle_id)
                    logger.info(
                        "[CognitiveRestructuring] plan created for %s with %d signals",
                        candidate["label"], len(candidate["signals"]),
                    )
                    self._broadcast(self._state["active_plan"])
            self._save()

    def observe_turn(self, correction: Optional[Dict[str, Any]], score: Optional[float]) -> None:
        """Give the active trial real evidence from a completed interaction."""
        with self._lock:
            plan = self._state.get("active_plan")
            if not plan or not plan.get("trial"):
                return
            trial = plan["trial"]
            trial["samples"] = int(trial.get("samples", 0)) + 1
            if score is not None:
                trial.setdefault("scores", []).append(round(float(score), 3))
            successful = correction is None and (score is None or score >= 0.65)
            trial["successes"] = int(trial.get("successes", 0)) + int(successful)
            if trial["samples"] < 3:
                self._save()
                return
            rate = trial["successes"] / max(1, trial["samples"])
            scores = trial.get("scores", [])
            improvement = (float(scores[-1]) - float(scores[0])) if len(scores) >= 2 else 0.0
            # A strategy must either avoid correction and maintain quality, or
            # show measurable improvement over its first observed response.
            confirmed = rate >= 0.67 and (improvement >= 0.02 or not scores)
            verdict = "confirmed" if confirmed else "disconfirmed" if rate <= 0.33 else "inconclusive"
            step = plan["steps"][plan["active_step"]]
            step["status"] = verdict
            step["success_rate"] = round(rate, 3)
            step["improvement"] = round(improvement, 3)
            plan["results"].append({"strategy": step["strategy"], "verdict": verdict, "rate": round(rate, 3), "improvement": round(improvement, 3)})
            if verdict == "confirmed":
                plan["validated_strategy"] = step["strategy"]
                plan["active_plan_status"] = "validated"
                self._apply_belief_update(plan, "affirm")
                self._state["history"].append(dict(plan))
                self._state["active_plan"] = None
            else:
                plan["active_step"] += 1
                if plan["active_step"] >= len(plan["steps"]):
                    plan["active_plan_status"] = "unresolved"
                    self._apply_belief_update(plan, "challenge")
                    self._state["history"].append(dict(plan))
                    self._state["active_plan"] = None
                else:
                    plan["trial"] = {"samples": 0, "successes": 0}
            self._save()

    def prompt_fragment(self) -> str:
        with self._lock:
            plan = self._state.get("active_plan")
            if not plan:
                return ""
            step = plan["steps"][plan["active_step"]]
            return (
                "[Validated correction strategy] Apply this silently: "
                f"{step['strategy']} Do not mention internal plans or telemetry."
            )

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "clusters": len(self._state.get("clusters", [])),
                "active_plan": self._state.get("active_plan"),
                "completed_plans": len(self._state.get("history", [])),
            }

    def _collect_signals(self) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        sce = getattr(self._organism, "_self_correction", None)
        if sce:
            for rule in sce.status().get("top_rules", []):
                signals.append({
                    "source": "interaction_correction",
                    "text": rule.get("rule", ""),
                    "strength": float(rule.get("confidence", 0.0)),
                })
        asp = getattr(self._organism, "aspirational_self", None)
        if asp:
            for item in asp.summary().get("top_tensions", [])[:4]:
                signals.append({
                    "source": "aspiration_tension",
                    "text": str(item.get("domain", "")),
                    "strength": float(item.get("tension", 0.0)),
                })
        ai = getattr(self._organism, "ai_system", None)
        sc = getattr(ai, "self_concept", None) if ai else None
        if sc:
            for belief in sorted(
                getattr(sc, "_beliefs", {}).values(),
                key=lambda b: float(getattr(b, "confidence", 0.0)),
                reverse=True,
            )[:8]:
                signals.append({
                    "source": "self_concept_belief",
                    "id": str(getattr(belief, "name", "")),
                    "text": str(getattr(belief, "statement", "")),
                    "strength": float(getattr(belief, "confidence", 0.0)),
                })
        return signals

    @staticmethod
    def _cluster(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for signal in signals:
            words = set(signal["text"].lower().replace("_", " ").split())
            target = next((c for c in clusters if words & c["words"]), None)
            if target is None:
                target = {"id": "cluster_" + uuid.uuid4().hex[:8], "words": set(), "signals": [], "strength": 0.0}
                clusters.append(target)
            target["words"].update(words)
            target["signals"].append({k: v for k, v in signal.items()})
            target["strength"] = max(target["strength"], signal["strength"])
        for cluster in clusters:
            cluster["label"] = " + ".join(sorted({s["source"] for s in cluster["signals"]}))
            cluster.pop("words", None)
        return clusters

    @staticmethod
    def _make_plan(cluster: Dict[str, Any], cycle_id: int) -> Dict[str, Any]:
        beliefs = [s for s in cluster["signals"] if s.get("source") == "self_concept_belief"]
        belief_text = ", ".join(s.get("text", "")[:100] for s in beliefs[:3]) or "no specific belief identified"
        return {
            "id": "restructure_" + uuid.uuid4().hex[:10],
            "created_cycle": cycle_id,
            "cluster": cluster["label"],
            "signals": cluster["signals"],
            "target_beliefs": [s.get("id") for s in beliefs if s.get("id")],
            "causal_hypotheses": [
                {
                    "id": "hyp_1",
                    "mechanism": "response_priority",
                    "statement": "The observed tension is reduced when the primary user constraint is resolved before elaboration.",
                    "targets": belief_text,
                },
                {
                    "id": "hyp_2",
                    "mechanism": "precondition_check",
                    "statement": "The observed tension is reduced when relevant physical, temporal, and causal preconditions are checked first.",
                    "targets": belief_text,
                },
                {
                    "id": "hyp_3",
                    "mechanism": "uncertainty_calibration",
                    "statement": "The observed tension is reduced when missing information is stated explicitly instead of being filled with narrative assumptions.",
                    "targets": belief_text,
                },
            ],
            "active_step": 0,
            "active_plan_status": "testing",
            "validated_strategy": "",
            "results": [],
            "steps": [
                {"strategy": "Prioritize the user's explicit objective and constraints before secondary interpretation.", "status": "testing"},
                {"strategy": "Cross-check the response against relevant memory and physical or temporal preconditions before committing.", "status": "pending"},
                {"strategy": "When uncertainty remains, state the missing constraint and ask one focused clarifying question.", "status": "pending"},
            ],
            "trial": {"samples": 0, "successes": 0},
        }

    def _apply_belief_update(self, plan: Dict[str, Any], mode: str) -> None:
        """Make a bounded, auditable update only after the whole trial ends."""
        try:
            ai = getattr(self._organism, "ai_system", None)
            sc = getattr(ai, "self_concept", None) if ai else None
            if not sc:
                return
            for name in plan.get("target_beliefs", [])[:3]:
                if mode == "affirm" and hasattr(sc, "record_affirmation"):
                    sc.record_affirmation(name)
                elif mode == "challenge" and hasattr(sc, "record_violation"):
                    sc.record_violation(name, context=plan.get("cluster", ""))
            plan["belief_update"] = {"mode": mode, "targets": plan.get("target_beliefs", [])[:3], "updated_at": time.time()}
        except Exception as exc:
            logger.debug("[CognitiveRestructuring] belief update failed: %s", exc)

    def _broadcast(self, plan: Dict[str, Any]) -> None:
        try:
            ws = getattr(self._organism, "workspace", None)
            if ws:
                ws.broadcast(
                    source="cognitive_restructuring",
                    content=f"Testing correction strategy for {plan['cluster']}",
                    priority=0.72,
                )
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._state.update(data)
        except Exception as exc:
            logger.debug("[CognitiveRestructuring] load failed: %s", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(self._state)
            data["clusters"] = data.get("clusters", [])[-12:]
            data["history"] = data.get("history", [])[-12:]
            self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("[CognitiveRestructuring] save failed: %s", exc)
