"""
SelfDescription — Phase B
============================
A queryable, hashable self-description of the architecture itself, built
from REAL current parameter values pulled from the actual modules
(internal_cognitive_state.py, epistemic_efficacy_model.py,
unified_revision_gateway.py, symbol_system.py) at registration time —
not placeholder text. drift detection (a parameter's hash changing since
last registration) is the concrete mechanism that lets a later reflection
process notice "my own configuration changed" as a fact, not a claim.

Per the design principle already established this session (every new
capability ablatable, extend don't replace): nothing here alters any
existing module. This only reads and records.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ComponentDesc:
    name: str
    module_path: str
    role: str
    key_parameters: Dict[str, Any]
    last_hash: str = ""
    notes: str = ""


class SelfDescription:
    def __init__(self, path: str = "data/persona/self_description.json") -> None:
        self._path = Path(path)
        self._lock = Lock()
        self.components: Dict[str, ComponentDesc] = {}
        self.last_updated: float = 0.0
        self._drift_events: list = []
        self._load()
        logger.info("[SelfDescription] initialised")

    def register_or_update(self, name: str, module_path: str, role: str,
                            key_parameters: Dict[str, Any], notes: str = "") -> bool:
        """Returns True if this registration is a genuine drift (params
        changed since the last registration of this component), False
        for a first-time registration or an unchanged re-registration."""
        param_str = json.dumps(key_parameters, sort_keys=True, default=str)
        h = hashlib.sha256(param_str.encode()).hexdigest()[:12]
        drifted = False
        with self._lock:
            prior = self.components.get(name)
            if prior is not None and prior.last_hash != h:
                drifted = True
                self._drift_events.append({
                    "t": time.time(), "component": name,
                    "old_hash": prior.last_hash, "new_hash": h,
                })
                self._drift_events = self._drift_events[-500:]
            self.components[name] = ComponentDesc(
                name=name, module_path=module_path, role=role,
                key_parameters=key_parameters, last_hash=h, notes=notes,
            )
            self.last_updated = time.time()
        self._save()
        return drifted

    def as_natural_language(self) -> str:
        lines = ["My current architectural self-description:"]
        with self._lock:
            for c in self.components.values():
                lines.append(
                    f"- {c.name} ({c.module_path}): {c.role}. "
                    f"Key params: {c.key_parameters}. Hash={c.last_hash}"
                )
        return "\n".join(lines)

    def detect_drift(self, name: str, current_params: Dict[str, Any]) -> bool:
        param_str = json.dumps(current_params, sort_keys=True, default=str)
        h = hashlib.sha256(param_str.encode()).hexdigest()[:12]
        with self._lock:
            if name not in self.components:
                return True
            return h != self.components[name].last_hash

    def recent_drift_events(self, limit: int = 20) -> list:
        with self._lock:
            return list(self._drift_events[-limit:])

    def scan_known_components(self) -> int:
        """
        Registers the real, current parameter values of every module
        built this session that exposes tunable module-level constants —
        not a generic/placeholder inventory. Returns the count of
        components that showed genuine drift since last scan (0 on a
        freshly-started process, since there's nothing to compare
        against yet — that's expected, not a bug).
        """
        drift_count = 0
        try:
            from cognition import internal_cognitive_state as ics
            if self.register_or_update(
                "InternalCognitiveState", "cognition/internal_cognitive_state.py",
                "Bounded, continuous epistemic-pressure bias on self-regulatory goals",
                {"SELF_STATE_GOAL_WEIGHT": ics.SELF_STATE_GOAL_WEIGHT,
                 "MAX_INTERNAL_BIAS": ics.MAX_INTERNAL_BIAS,
                 "DELTA_SCALE": ics.DELTA_SCALE,
                 "ENABLED": ics.INTERNAL_STATE_PROJECTION_ENABLED},
            ):
                drift_count += 1
        except Exception as e:
            logger.debug(f"[SelfDescription] scan InternalCognitiveState failed: {e}")

        try:
            from cognition import epistemic_efficacy_model as eem
            if self.register_or_update(
                "EpistemicEfficacyModel", "cognition/epistemic_efficacy_model.py",
                "Falsifiable self-prediction about resolve_uncertainty's effectiveness",
                {"PREDICTION_SCALE": eem.PREDICTION_SCALE,
                 "GOOD_PREDICTION_THRESHOLD": eem.GOOD_PREDICTION_THRESHOLD,
                 "MIN_MULTIPLIER": eem.MIN_MULTIPLIER, "MAX_MULTIPLIER": eem.MAX_MULTIPLIER},
            ):
                drift_count += 1
        except Exception as e:
            logger.debug(f"[SelfDescription] scan EpistemicEfficacyModel failed: {e}")

        try:
            from cognition import unified_revision_gateway as urg
            if self.register_or_update(
                "UnifiedRevisionGateway", "cognition/unified_revision_gateway.py",
                "Safety wrapper on the live autonomous self-modification path",
                {"MAX_STEP": urg.MAX_STEP, "BOUNDS": urg.BOUNDS,
                 "PROTECTED_SUBSTRINGS": list(urg.PROTECTED_SUBSTRINGS)},
            ):
                drift_count += 1
        except Exception as e:
            logger.debug(f"[SelfDescription] scan UnifiedRevisionGateway failed: {e}")

        try:
            from cognition import symbol_system as ss
            if self.register_or_update(
                "SymbolSystem", "cognition/symbol_system.py",
                "Open symbol repertoire absorbing structured self-referential events",
                {"MIN_SIGNIFICANCE_TO_ABSORB": ss.MIN_SIGNIFICANCE_TO_ABSORB,
                 "MIN_CONFIDENCE_FOR_CANDIDATE": ss.MIN_CONFIDENCE_FOR_CANDIDATE},
            ):
                drift_count += 1
        except Exception as e:
            logger.debug(f"[SelfDescription] scan SymbolSystem failed: {e}")

        return drift_count

    # ── persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {"last_updated": self.last_updated,
                           "components": [asdict(c) for c in self.components.values()],
                           "drift_events": self._drift_events}
            self._path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception as e:
            logger.debug(f"[SelfDescription] save failed (non-fatal): {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            self.last_updated = data.get("last_updated", 0.0)
            for c in data.get("components", []):
                self.components[c["name"]] = ComponentDesc(**c)
            self._drift_events = data.get("drift_events", [])
        except Exception as e:
            logger.warning(f"[SelfDescription] load failed (non-fatal): {e}")


_descriptions = {}


def get_self_description() -> SelfDescription:
    global _descriptions
    if "singleton" not in _descriptions:
        _descriptions["singleton"] = SelfDescription()
    return _descriptions["singleton"]
