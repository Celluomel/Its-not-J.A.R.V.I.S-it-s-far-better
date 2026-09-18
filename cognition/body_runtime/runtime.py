"""A small, local body process boundary.

The body owns normalized observations and an auditable command queue. It is
independent of the cognitive loop and remains useful without a brain attached.
Actuator execution is deliberately not implemented yet; commands stay queued
until a permissioned adapter is added.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from threading import Event, RLock, Thread
import time
from typing import Any, Dict, Optional
from secret_store import SECRET_FIELDS, config_reference, resolve, store


@dataclass
class BodyObservation:
    source: str
    kind: str
    subject: str
    value: Any
    unit: str = ""
    confidence: float = 1.0
    observed_at: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["age_seconds"] = round(max(0.0, time.time() - self.observed_at), 3)
        return result


@dataclass
class BodyCommand:
    target: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    requires_confirmation: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BodyRuntime:
    """Thread-safe local body loop shared by adapters and the brain."""

    def __init__(self, organism: Any = None, max_events: int = 256):
        self.organism = organism
        self._config_path = Path("data/body/config.json")
        self._config = self._load_config()
        self._lock = RLock()
        self._plugins: Dict[str, Dict[str, Any]] = {}
        self.register_plugin(
            "home_assistant",
            "Home Assistant",
            "BODY_PLUGIN_HOME_ASSISTANT_ENABLED",
            "Read-only environmental sensors and presence",
        )
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._bridge = None
        self._latest: Dict[str, BodyObservation] = {}
        self._events = deque(maxlen=max_events)
        self._commands = deque(maxlen=64)
        self._heartbeat = time.time()
        self._loop_count = 0

    _CONFIG_FIELDS = {
        "BODY_RUNTIME_ENABLED", "BODY_PLUGIN_HOME_ASSISTANT_ENABLED",
        "BODY_HOST", "BODY_PORT",
        "HOME_ASSISTANT_ENABLED", "HOME_ASSISTANT_PRESENCE_ENABLED",
        "HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN",
        "HOME_ASSISTANT_VERIFY_SSL", "HOME_ASSISTANT_POLL_INTERVAL",
        "HOME_ASSISTANT_ALLOWED_DOMAINS", "HOME_ASSISTANT_SELECTED_ENTITIES",
        "HOME_ASSISTANT_DISCOVERED_ENTITIES", "HOME_ASSISTANT_ENTITY_TAGS",
        "BODY_BRIDGE_ENABLED", "BODY_BRIDGE_URL", "BODY_BRIDGE_TOKEN",
        "BODY_BRIDGE_DEVICE_ID", "BODY_BRIDGE_VERIFY_TLS", "BODY_BRIDGE_RECONNECT_SECONDS",
    }

    def _load_config(self) -> Dict[str, Any]:
        try:
            if self._config_path.exists():
                payload = json.loads(self._config_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    for key in SECRET_FIELDS:
                        if key in payload:
                            payload[key] = resolve(key, payload[key])
                    return payload
                return {}
        except Exception:
            pass
        migrated: Dict[str, Any] = {}
        try:
            from managers.settings_manager import config
            for key in self._CONFIG_FIELDS:
                if hasattr(config, key):
                    migrated[key] = getattr(config, key)
        except Exception:
            pass
        self._write_config(migrated)
        return migrated

    def _write_config(self, payload: Dict[str, Any]) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            safe = dict(payload)
            for key in SECRET_FIELDS:
                if safe.get(key):
                    store(key, safe[key])
                safe[key] = config_reference(key, safe.get(key))
            self._config_path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def config_value(self, key: str, fallback: Any = None) -> Any:
        with self._lock:
            return self._config.get(key, fallback)

    def update_config(self, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            for key, value in values.items():
                if key in self._CONFIG_FIELDS:
                    self._config[key] = value
            self._write_config(self._config)
            bridge_enabled = bool(self._config.get("BODY_BRIDGE_ENABLED", False))
            bridge = self._bridge
        if bridge_enabled and self._thread and self._thread.is_alive():
            if bridge is None:
                try:
                    from cognition.body_runtime.bridge import BodyBrainBridge
                    self._bridge = BodyBrainBridge(self)
                    self._bridge.start()
                except Exception:
                    pass
        elif not bridge_enabled and bridge is not None:
            bridge.stop()
            self._bridge = None
        return dict(self._config)

    def config_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._config)

    def register_plugin(self, plugin_id: str, label: str, toggle_field: str, description: str) -> None:
        """Allow body adapters to announce themselves without changing brain UI."""
        with self._lock:
            self._plugins[plugin_id] = {
                "id": plugin_id,
                "label": label,
                "toggle_field": toggle_field,
                "description": description,
                "available": True,
            }

    def plugins(self) -> list[Dict[str, Any]]:
        with self._lock:
            plugins = list(self._plugins.values())
        for plugin in plugins:
            field = plugin["toggle_field"]
            fallback = self.config_value(
                "HOME_ASSISTANT_ENABLED", False
            ) if plugin["id"] == "home_assistant" else False
            plugin["enabled"] = bool(self.config_value(field, fallback))
            # Do not call status() here: status() includes plugin snapshots and
            # would recurse forever as soon as the Body toggle is enabled.
            running = bool(self._thread and self._thread.is_alive())
            plugin["runtime"] = "active" if plugin["enabled"] and running else "disabled"
        return plugins

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(target=self._run, name="lumina-body-runtime", daemon=True)
            self._thread.start()
            if bool(self.config_value("BODY_BRIDGE_ENABLED", False)):
                try:
                    from cognition.body_runtime.bridge import BodyBrainBridge
                    self._bridge = self._bridge or BodyBrainBridge(self)
                    self._bridge.start()
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning("[BodyRuntime] Brain bridge unavailable", exc_info=True)

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        self._thread = None
        if self._bridge is not None:
            self._bridge.stop(timeout)

    def publish_observation(self, observation: BodyObservation | None = None, forward: bool = True, **kwargs: Any) -> BodyObservation:
        item = observation or BodyObservation(**kwargs)
        key = f"{item.source}:{item.subject}"
        with self._lock:
            self._latest[key] = item
            self._events.append(item)
        if forward and self._bridge is not None:
            self._bridge.enqueue_observation(item.as_dict())
        return item

    def enqueue_command(self, target: str, action: str, payload: Optional[Dict[str, Any]] = None) -> BodyCommand:
        command = BodyCommand(target=target, action=action, payload=dict(payload or {}))
        with self._lock:
            self._commands.append(command)
        return command

    def snapshot(self, max_age: Optional[float] = None) -> list[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            observations = list(self._latest.values())
        if max_age is not None:
            observations = [item for item in observations if now - item.observed_at <= max_age]
        return [item.as_dict() for item in sorted(observations, key=lambda item: item.observed_at, reverse=True)]

    def recent_events(self, limit: int = 20) -> list[Dict[str, Any]]:
        with self._lock:
            return [item.as_dict() for item in list(self._events)[-max(1, limit):]]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            thread = self._thread
            return {
                "available": True,
                "running": bool(thread and thread.is_alive()),
                "mode": "standalone-local",
                "heartbeat": self._heartbeat,
                "loop_count": self._loop_count,
                "observation_count": len(self._latest),
                "event_count": len(self._events),
                "pending_commands": len(self._commands),
                "last_observation": max((item.observed_at for item in self._latest.values()), default=None),
                "plugins": {
                    item["id"]: item["enabled"] for item in self.plugins()
                },
                "brain_bridge": {
                    "enabled": bool(self.config_value("BODY_BRIDGE_ENABLED", False)),
                    "connected": bool(self._bridge and self._bridge.connected),
                    "url": str(self.config_value("BODY_BRIDGE_URL", "") or ""),
                },
            }

    def context_for_brain(self, max_age: float = 120.0, exclude_sources: Optional[set[str]] = None) -> str:
        if not bool(self.config_value("BODY_RUNTIME_ENABLED", True)):
            return ""
        observations = self.snapshot(max_age=max_age)
        if exclude_sources:
            observations = [item for item in observations if item.get("source") not in exclude_sources]
        # The Body may receive a permitted discovery list from a plugin, while
        # the user-facing selection is narrower. Keep unselected HA entities
        # out of the brain prompt even when the connector refreshed them.
        selected = {
            item.strip()
            for item in str(self.config_value("HOME_ASSISTANT_SELECTED_ENTITIES", "") or "").split(",")
            if item.strip()
        }
        if selected:
            observations = [
                item for item in observations
                if item.get("source") != "home_assistant"
                or str((item.get("provenance") or {}).get("entity_id", "")) in selected
            ]
        if not observations:
            return ""
        lines = [
            "BODY RUNTIME OBSERVATIONS (local, timestamped sensor evidence):",
            "Use these observations only when relevant; preserve exact values and timestamps.",
        ]
        for item in observations:
            value = item["value"]
            unit = f" {item['unit']}" if item.get("unit") else ""
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(item["observed_at"]))
            lines.append(f"- {item['subject']}: {value}{unit} (source {item['source']}, observed {stamp})")
        return "\n".join(lines)

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                self._heartbeat = time.time()
                self._loop_count += 1


def get_body_runtime(organism: Any = None) -> BodyRuntime:
    """Return one body runtime per organism and start its local loop lazily."""
    if organism is not None:
        existing = getattr(organism, "_body_runtime", None)
        if existing is not None:
            existing.start()
            return existing
    runtime = BodyRuntime(organism=organism)
    runtime.start()
    if organism is not None:
        try:
            organism._body_runtime = runtime
        except Exception:
            pass
    return runtime
