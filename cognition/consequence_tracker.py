"""
LUMINA — CONSEQUENCE TRACKER  (the closed outcome loop)
=======================================================
THIS is the highest-priority repair: turning
    GOAL -> ACTION -> OUTCOME   (X)   PREDICTION ERROR
into
    GOAL -> PREDICTION -> ACTION -> OUTCOME -> RESOLUTION -> ERROR
          -> SELECTIVE UPDATE (efficacy / attention / PCM)

What it does, per autonomous action (memory_recall / web_search /
self_question / user_question / bisociative):

  1. PREDICT  (strictly BEFORE the action fires)
     Attach a structured prediction: what effect this action type should
     have, on which measurable probe, with what confidence.

  2. ACT      (the existing GoalActionExecutor runs unchanged)

  3. RESOLVE  (strictly AFTER the action)
     Evaluate the probe against the real result -> observed in {0, .5, 1}.

  4. ERROR
     prediction_error = 1 - observed   (0.0 = exactly as predicted)

  5. SELECTIVE UPDATE  (deliberately NOT a spaghetti feedback fan-out)
     a. GOAL EFFICACY  efficacy[(goal,action)] = EMA(observed)
        -> a REAL feedback signal that re-weights future action selection.
     b. ATTENTION      nudge the one module this action maps to (±),
        renormalise. Reuses the existing cognitive_attention.json store.
     c. PCM            feed a real ConsequenceRecord + calibration
        resolution into the (previously 0/192) Predictive Consequence Model.
     d. LEDGER         write the full causal chain (prediction->action->
        outcome->error->updates) with parent_event_ids so it is traceable.

  6. METRIC   closure_report() -> resolution rate, mean|error|, per-action
     efficacy. This is the "claims can be demonstrated from their own
     causal traces" deliverable.

DESIGN RULES
  - Selective propagation: one action updates at most efficacy + ONE
    attention module + one PCM record. No global fan-out.
  - Never kills the loop: every external side-effect is try/except wrapped.
  - The ledger is the *trace of* these updates, not a separate service.
  - Pure & testable: predict()/resolve() are deterministic given (prediction,
    report) — verified without the LLM.

Stores (all under data/persona/):
  causal_efficacy.json     { "goalid|action": {"eff":float,"n":int} }
  cognitive_attention.json (existing)  -> learned_bias + attention_weights
  causal_ledger.jsonl      (append-only causal chain)
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

# action type -> (measurable probe description, attention module it maps to,
#                expected effect label). The probe is evaluated in _observe().
ACTION_MODEL: Dict[str, Dict[str, str]] = {
    "memory_recall":  {"module": "semantic_memory",   "effect": "evidence_gain",   "probe": "recall_nonempty"},
    "web_search":     {"module": "goal_ecology",      "effect": "uncertainty_drop","probe": "success"},
    "self_question":  {"module": "self_model",        "effect": "clarity_gain",    "probe": "success"},
    "user_question":  {"module": "relational_memory", "effect": "external_feedback","probe": "success"},
    "bisociative":    {"module": "creative_divergence","effect": "hypothesis_fertility","probe": "success"},
}
DEFAULT_MODULE = "decision_policy"

_EFF_ALPHA = 0.30        # EMA rate for goal efficacy
_ATTN_NUDGE = 0.06       # max attention shift per resolution (bounded)
_CONF_BASE  = 0.55       # base prediction confidence for a dispatched action


class ConsequenceTracker:
    def __init__(self, persona_dir: Optional[str] = None,
                 consequence_model: Any = None,
                 ledger: Optional["CausalLedger"] = None,
                 enabled: bool = True):
        self.persona_dir = Path(persona_dir) if persona_dir else Path("data/persona")
        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._lock = threading.Lock()

        self.efficacy_path = self.persona_dir / "causal_efficacy.json"
        self.attention_path = self.persona_dir / "cognitive_attention.json"

        self._consequence_model = consequence_model
        self.ledger = ledger if ledger is not None else (
            CausalLedger(str(self.persona_dir)) if CausalLedger is not None else None
        )

        self._pending: Dict[str, Dict[str, Any]] = {}
        self._efficacy: Dict[str, Dict[str, float]] = self._load_efficacy()
        # closure metrics (in-memory, also persisted)
        self.metrics = {
            "n_predictions": 0,
            "n_resolved": 0,
            "abs_error_sum": 0.0,
            "per_action": {},
            "pcm_resolutions": 0,
        }

    # ────────────────────────────────────────────────────────────────────────
    # PREDICT  (called immediately BEFORE the action fires)
    # ────────────────────────────────────────────────────────────────────────
    def predict(self, goal_id: str, goal_topic: str, action_type: str,
                cycle_id: Optional[int] = None,
                goal_energy: float = 1.0) -> Dict[str, Any]:
        if not self.enabled:
            return {"disabled": True, "goal_id": str(goal_id),
                    "action_type": action_type, "goal_topic": goal_topic}
        am = ACTION_MODEL.get(action_type, {"module": DEFAULT_MODULE, "effect": "effect", "probe": "success"})
        pred = {
            "pred_id": f"pr_{abs(hash((goal_id, action_type, datetime.now().isoformat()))) % 10**10:010d}",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cycle_id": cycle_id,
            "goal_id": str(goal_id),
            "goal_topic": goal_topic,
            "action_type": action_type,
            "predicted_effect": am["effect"],
            "probe": am["probe"],
            "attention_module": am["module"],
            "confidence": _CONF_BASE,
            "goal_energy_before": round(float(goal_energy), 4),
        }
        with self._lock:
            self._pending[pred["pred_id"]] = pred
            self.metrics["n_predictions"] += 1
        # ledger: the PREDICTION event (root of this causal chain)
        if self.ledger:
            self.ledger.record(
                kind="prediction",
                cause=(f"predict {am['effect']} from {action_type} on "
                       f"'{goal_topic[:40]}' (conf {_CONF_BASE})"),
                source="consequence_tracker",
                cycle_id=cycle_id,
                output_state=pred,
                confidence=_CONF_BASE,
                refs={"goal_id": str(goal_id), "action": action_type},
            )
        return pred

    # ────────────────────────────────────────────────────────────────────────
    # RESOLVE  (called immediately AFTER the action)
    # ────────────────────────────────────────────────────────────────────────
    def resolve(self, pred: Optional[Dict[str, Any]], report: Dict[str, Any],
                cycle_id: Optional[int] = None) -> Dict[str, Any]:
        if pred is None:
            return {"resolved": False, "reason": "no prediction"}
        if not self.enabled:
            return {"resolved": False, "reason": "disabled"}

        observed = self._observe(pred, report)            # 0.0 / 0.5 / 1.0
        error = round(1.0 - observed, 4)                  # 0.0 = as predicted
        outcome = "positive" if observed >= 0.999 else ("neutral" if observed >= 0.499 else "negative")

        root_id = None
        if self.ledger:
            # find the prediction event id (most recent for this goal+action)
            root_id = self._find_prediction_event(pred)

        # 5a. GOAL EFFICACY (real feedback -> re-weights future selection)
        eff_delta = self._update_efficacy(pred["goal_id"], pred["action_type"], observed)

        # 5b. ATTENTION (nudge the ONE mapped module, renormalise)
        attn_delta = self._nudge_attention(pred["attention_module"], observed)

        # 5c. PCM (feed a real record + calibration resolution)
        pcm_ok = self._feed_pcm(pred, report, observed, outcome)

        # metrics
        with self._lock:
            self.metrics["n_resolved"] += 1
            self.metrics["abs_error_sum"] += error
            pa = self.metrics["per_action"].setdefault(
                pred["action_type"], {"n": 0, "abs_error_sum": 0.0, "efficacy": 0.0})
            pa["n"] += 1
            pa["abs_error_sum"] += error
            pa["efficacy"] = self._efficacy.get(
                f"{pred['goal_id']}|{pred['action_type']}", {}).get("eff", 0.5)

        # 5d. LEDGER (error + updates, chained to the prediction event)
        if self.ledger:
            self.ledger.record(
                kind="outcome",
                cause=f"{pred['action_type']} on '{pred['goal_topic'][:40]}' -> observed={observed:.2f} ({outcome})",
                cycle_id=cycle_id,
                input_state={"action": pred["action_type"], "success": report.get("success")},
                output_state={"observed": observed, "outcome": outcome},
                confidence=_CONF_BASE,
                parent_event_ids=[root_id] if root_id else [],
                refs={"goal_id": pred["goal_id"], "action": pred["action_type"], "pred_id": pred["pred_id"]},
            )
            self.ledger.record(
                kind="error",
                cause=f"prediction_error={error:+.3f} (predicted {pred['predicted_effect']}, got {outcome})",
                cycle_id=cycle_id,
                delta={"error": error, "observed": observed},
                confidence=_CONF_BASE,
                parent_event_ids=[root_id] if root_id else [],
                refs={"goal_id": pred["goal_id"], "action": pred["action_type"], "target": "error"},
            )
            if eff_delta is not None:
                self.ledger.record(
                    kind="update",
                    cause=(f"efficacy[{pred['goal_id']}|{pred['action_type']}] "
                           f"{eff_delta['before']:.4f} → {eff_delta['after']:.4f}"),
                    cycle_id=cycle_id, delta=eff_delta,
                    parent_event_ids=[root_id] if root_id else [],
                    refs={"goal_id": pred["goal_id"], "action": pred["action_type"], "target": "efficacy"},
                )
            if attn_delta is not None:
                self.ledger.record(
                    kind="update",
                    cause=(f"attention[{attn_delta['module']}] "
                           f"{attn_delta['before']:.4f} → {attn_delta['after']:.4f}"),
                    cycle_id=cycle_id, delta=attn_delta,
                    parent_event_ids=[root_id] if root_id else [],
                    refs={"goal_id": pred["goal_id"], "module": pred["attention_module"], "target": "attention"},
                )

        with self._lock:
            self._pending.pop(pred["pred_id"], None)

        return {
            "resolved": True,
            "pred_id": pred["pred_id"],
            "observed": observed,
            "outcome": outcome,
            "error": error,
            "efficacy_delta": eff_delta,
            "attention_delta": attn_delta,
            "pcm_resolved": pcm_ok,
        }

    # ────────────────────────────────────────────────────────────────────────
    # FEEDBACK HOOK  (REAL: re-weights future action selection)
    # ────────────────────────────────────────────────────────────────────────
    def efficacy_multiplier(self, goal_id: str, action_type: str) -> float:
        """
        Returns a multiplier in [0.70, 1.30] for selecting (goal, action).
        >1 = this action has worked well for this goal before; <1 = it has
        tended to fail. GoalActionExecutor multiplies a candidate's priority
        by this BEFORE choosing — so the loop actually bends future behaviour.
        """
        row = self._efficacy.get(f"{goal_id}|{action_type}")
        if not row or row.get("n", 0) < 1:
            return 1.0
        eff = row.get("eff", 0.5)
        # Map efficacy [0,1] -> multiplier [0.70,1.30] with the NEUTRAL point
        # (eff=0.5) mapping to exactly 1.0:  mult = 1.0 + (eff-0.5)*0.6
        mult = 1.0 + (eff - 0.5) * 0.6
        return round(max(0.70, min(1.30, mult)), 4)

    # ────────────────────────────────────────────────────────────────────────
    # METRIC
    # ────────────────────────────────────────────────────────────────────────
    def closure_report(self) -> Dict[str, Any]:
        m = dict(self.metrics)
        m["resolution_rate"] = round(m["n_resolved"] / m["n_predictions"], 4) if m["n_predictions"] else 0.0
        m["mean_abs_error"] = round(m["abs_error_sum"] / m["n_resolved"], 4) if m["n_resolved"] else 0.0
        for a, pa in m["per_action"].items():
            pa["mean_abs_error"] = round(pa["abs_error_sum"] / pa["n"], 4) if pa["n"] else 0.0
        if self.ledger:
            m["ledger"] = self.ledger.stats()
        return m

    # ────────────────────────────────────────────────────────────────────────
    # internals
    # ────────────────────────────────────────────────────────────────────────
    def _observe(self, pred: Dict[str, Any], report: Dict[str, Any]) -> float:
        """Map the real report to observed in {0, .5, 1} via the probe."""
        probe = pred.get("probe", "success")
        success = bool(report.get("success"))
        result = str(report.get("result", "")).lower()

        if probe == "recall_nonempty":
            # memory_recall is a real success only if it returned content
            if not success:
                return 0.0
            if result and ("no" not in result[:12] or "recalled" in result or "found" in result or len(result) > 25):
                return 1.0
            return 0.5
        # probe == "success"
        if not success:
            return 0.0
        # async-dispatched actions report "dispatched (async)" -> we only KNOW
        # it started; treat as partial evidence, not a confirmed effect.
        if "dispatched" in result or "async" in result:
            return 0.5
        return 1.0

    def _update_efficacy(self, goal_id: str, action_type: str, observed: float):
        key = f"{goal_id}|{action_type}"
        prev = self._efficacy.get(key, {"eff": 0.5, "n": 0})
        new_eff = _EFF_ALPHA * observed + (1 - _EFF_ALPHA) * prev["eff"]
        self._efficacy[key] = {"eff": round(new_eff, 4), "n": prev["n"] + 1,
                               "last_observed": round(observed, 3),
                               "ts": datetime.now().isoformat(timespec="seconds")}
        try:
            with open(self.efficacy_path, "w", encoding="utf-8") as f:
                json.dump(self._efficacy, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return {"before": prev["eff"], "after": round(new_eff, 4), "n": prev["n"] + 1}

    def _nudge_attention(self, module: str, observed: float):
        """Nudge one module's weight toward/away from the outcome, renormalise."""
        try:
            data = {}
            if self.attention_path.exists():
                data = json.loads(self.attention_path.read_text(encoding="utf-8"))
            weights = dict(data.get("attention_weights") or {})
            if not weights:
                return None
            bias = data.get("learned_bias") or {}
            sign = 1.0 if observed >= 0.5 else -1.0
            magnitude = _ATTN_NUDGE * (abs(observed - 0.5) * 2)  # 0.._ATTN_NUDGE
            old = weights.get(module, 0.10)
            new = max(0.02, min(0.40, old + sign * magnitude))
            weights[module] = new
            # renormalise to sum 1.0, absorbing the rounding residual so the
            # stored weights sum to EXACTLY 1.0 (downstream assumes a simplex)
            s = sum(weights.values()) or 1.0
            weights = {k: v / s for k, v in weights.items()}
            rounded = {k: round(v, 5) for k, v in weights.items()}
            residual = round(1.0 - sum(rounded.values()), 5)
            if residual != 0.0:
                # put the residual on the nudged module so its delta stays honest
                rounded[module] = round(rounded[module] + residual, 5)
            weights = rounded
            bias[module] = round(bias.get(module, 0.0) + sign * magnitude, 5)
            data["attention_weights"] = weights
            data["learned_bias"] = bias
            data["last_consequence_nudge"] = {
                "module": module, "delta": round(new - old, 5),
                "observed": observed, "ts": datetime.now().isoformat(timespec="seconds")}
            with open(self.attention_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return {"module": module, "before": round(old, 5), "after": round(weights[module], 5)}
        except Exception:
            return None

    def _feed_pcm(self, pred: Dict[str, Any], report: Dict[str, Any],
                  observed: float, outcome: str) -> bool:
        """Push a real ConsequenceRecord + calibration resolution into the PCM."""
        pcm = self._consequence_model
        if pcm is None:
            return False
        try:
            energy = float(pred.get("goal_energy_before", 1.0))
            social = 1.0 if pred["action_type"] == "user_question" else 0.3
            focus = 0.10
            state_before = [energy, social, focus]
            # state_after: energy unchanged, social up if engaged, focus per observed
            state_after = [energy, (1.0 if outcome == "positive" else social),
                           round(0.05 + observed * 0.10, 4)]
            pcm.record_action(pred["action_type"], state_before,
                              interaction_n=int(pred.get("cycle_id") or 0))
            pcm.record_outcome(state_after, outcome=outcome)
            with self._lock:
                self.metrics["pcm_resolutions"] += 1
            return True
        except Exception:
            return False

    def _find_prediction_event(self, pred: Dict[str, Any]) -> Optional[str]:
        if not self.ledger:
            return None
        # most recent prediction event for this goal+action
        try:
            for ev in reversed(self.ledger.recent(20)):
                if ev.get("kind") == "prediction" and \
                   (ev.get("refs") or {}).get("goal_id") == pred["goal_id"] and \
                   (ev.get("refs") or {}).get("action") == pred["action_type"]:
                    return ev["event_id"]
        except Exception:
            pass
        return None

    def _load_efficacy(self) -> Dict[str, Dict[str, float]]:
        try:
            if self.efficacy_path.exists():
                return json.loads(self.efficacy_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}


# ── process-wide singleton so the executor and the loop share one instance ──
_tracker: Optional[ConsequenceTracker] = None
_tracker_lock = threading.Lock()


def get_consequence_tracker(organism: Any = None,
                            consequence_model: Any = None,
                            persona_dir: Optional[str] = None) -> ConsequenceTracker:
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            persona_dir = persona_dir or "data/persona"
            if consequence_model is None and organism is not None:
                ai = getattr(organism, "ai_system", None)
                if ai is not None:
                    for attr in ("_consequence_model", "consequence_model",
                                 "predictive_consequence_model"):
                        consequence_model = getattr(ai, attr, None)
                        if consequence_model is not None:
                            break
            _tracker = ConsequenceTracker(persona_dir=persona_dir,
                                          consequence_model=consequence_model)
        return _tracker
