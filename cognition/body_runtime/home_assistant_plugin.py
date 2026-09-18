"""Home Assistant sensor adapter for the standalone Body Runtime."""
from __future__ import annotations

import json
import logging
import ssl
from threading import Event, Thread
import time
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


class HomeAssistantBodyPlugin:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_discovered_json = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="lumina-body-home-assistant", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def discover(self) -> Dict[str, Any]:
        url = str(self.runtime.config_value("HOME_ASSISTANT_URL", "") or "").rstrip("/")
        token = str(self.runtime.config_value("HOME_ASSISTANT_TOKEN", "") or "")
        if not url or not token:
            return {"ok": False, "message": "Home Assistant URL and token are required."}
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"ok": False, "message": "HOME_ASSISTANT_URL must be an HTTP(S) URL."}
        request = Request(f"{url}/api/states", headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        try:
            context = None
            if parsed.scheme == "https" and not bool(self.runtime.config_value("HOME_ASSISTANT_VERIFY_SSL", True)):
                context = ssl._create_unverified_context()
            with urlopen(request, timeout=5.0, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                return {"ok": False, "message": "Home Assistant returned an invalid state list."}
            allowed = {item.strip().lower() for item in str(self.runtime.config_value("HOME_ASSISTANT_ALLOWED_DOMAINS", "") or "").split(",") if item.strip()}
            selected = {item.strip() for item in str(self.runtime.config_value("HOME_ASSISTANT_SELECTED_ENTITIES", "") or "").split(",") if item.strip()}
            entities = [self._entity(item) for item in payload if isinstance(item, dict) and self._permitted(item, allowed, selected)]
            entities.sort(key=lambda item: item["entity_id"])
            return {"ok": True, "entities": entities, "count": len(entities)}
        except HTTPError as exc:
            return {"ok": False, "message": f"Home Assistant returned HTTP {exc.code}."}
        except (URLError, TimeoutError) as exc:
            return {"ok": False, "message": f"Home Assistant unreachable: {exc}."}
        except Exception as exc:
            logger.warning("[Body/HomeAssistant] discovery failed: %s", exc)
            return {"ok": False, "message": str(exc)}

    def _run(self) -> None:
        while not self._stop.is_set():
            enabled = self.runtime.config_value("BODY_PLUGIN_HOME_ASSISTANT_ENABLED", None)
            if enabled is None:
                enabled = self.runtime.config_value("HOME_ASSISTANT_ENABLED", False)
            if enabled:
                result = self.discover()
                if result.get("ok"):
                    for item in result["entities"]:
                        value = item.get("state")
                        if item.get("signal") == "presence_detected":
                            value = "presence detected"
                        elif item.get("signal") == "no_presence":
                            value = "no presence"
                        self.runtime.publish_observation(
                            source="home_assistant",
                            kind=item.get("domain", "sensor"),
                            subject=self._label(item),
                            value=value,
                            unit=item.get("unit") or "",
                            confidence=0.98,
                            observed_at=time.time(),
                            provenance={"entity_id": item["entity_id"], "sensor_last_updated": item.get("last_updated")},
                        )
                    discovered_json = json.dumps(result["entities"], ensure_ascii=False)
                    if discovered_json != self._last_discovered_json:
                        self.runtime.update_config({"HOME_ASSISTANT_DISCOVERED_ENTITIES": discovered_json})
                        self._last_discovered_json = discovered_json
                else:
                    logger.warning("[Body/HomeAssistant] %s", result.get("message", "poll failed"))
            interval = max(1, min(300, int(self.runtime.config_value("HOME_ASSISTANT_POLL_INTERVAL", 5) or 5)))
            self._stop.wait(interval)

    def _permitted(self, item: Dict[str, Any], allowed: set[str], selected: set[str]) -> bool:
        entity_id = str(item.get("entity_id", ""))
        return bool(entity_id) and (not selected or entity_id in selected) and (not allowed or entity_id.split(".", 1)[0].lower() in allowed)

    def _entity(self, item: Dict[str, Any]) -> Dict[str, Any]:
        entity_id = str(item.get("entity_id", ""))
        attributes = item.get("attributes") or {}
        device_class = attributes.get("device_class")
        state = item.get("state")
        signal = None
        if entity_id.startswith("binary_sensor.") and device_class in {"presence", "occupancy", "motion"}:
            signal = "presence_detected" if str(state).lower() == "on" else "no_presence"
        return {"entity_id": entity_id, "state": state, "friendly_name": attributes.get("friendly_name", entity_id), "unit": attributes.get("unit_of_measurement"), "domain": entity_id.split(".", 1)[0], "device_class": device_class, "signal": signal, "last_updated": item.get("last_updated"), "last_changed": item.get("last_changed")}

    def _label(self, item: Dict[str, Any]) -> str:
        entity_id = item["entity_id"]
        try:
            tags = json.loads(str(self.runtime.config_value("HOME_ASSISTANT_ENTITY_TAGS", "{}") or "{}"))
            if isinstance(tags, dict) and str(tags.get(entity_id, "")).strip():
                return str(tags[entity_id]).strip()
        except Exception:
            pass
        return str(item.get("friendly_name") or entity_id)
