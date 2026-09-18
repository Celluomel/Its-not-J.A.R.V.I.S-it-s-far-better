"""Independent Body process for sensor plugins and the optional Brain bridge.

This module intentionally uses only the Python standard library plus the
optional ``websockets`` package. It does not import the cognitive package.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import ssl
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

LOG = logging.getLogger("lumina.body")
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data" / "body" / "config.json"


def _load_dotenv() -> None:
    for path in (ROOT / "body_venv" / ".env", ROOT / ".body_venv" / ".env", ROOT / ".venv" / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class BodyHost:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.config = self._load_config()
        self.latest: dict[str, dict] = {}
        self.queue: Queue[dict] = Queue(maxsize=256)
        self.bridge_thread: threading.Thread | None = None

    def _load_config(self) -> dict:
        try:
            payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            LOG.warning("Could not load %s: %s", CONFIG_PATH, exc)
            return {}

    def value(self, key: str, fallback=None):
        value = self.config.get(key, fallback)
        if isinstance(value, str) and value.startswith("@env:"):
            return os.environ.get(value[5:], "")
        return value

    def save_discovery(self, entities: list[dict]) -> None:
        serialized = json.dumps(entities, ensure_ascii=False)
        if self.config.get("HOME_ASSISTANT_DISCOVERED_ENTITIES") == serialized:
            return
        self.config["HOME_ASSISTANT_DISCOVERED_ENTITIES"] = serialized
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        safe = dict(self.config)
        safe["HOME_ASSISTANT_TOKEN"] = "@env:HOME_ASSISTANT_TOKEN"
        safe["BODY_BRIDGE_TOKEN"] = "@env:BODY_BRIDGE_TOKEN"
        CONFIG_PATH.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")

    def discover(self) -> list[dict]:
        url = str(self.value("HOME_ASSISTANT_URL", "") or "").rstrip("/")
        token = str(self.value("HOME_ASSISTANT_TOKEN", "") or "")
        if not url or not token:
            return []
        parsed = urlparse(url)
        request = Request(f"{url}/api/states", headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        context = None
        if parsed.scheme == "https" and not bool(self.value("HOME_ASSISTANT_VERIFY_SSL", True)):
            context = ssl._create_unverified_context()
        with urlopen(request, timeout=5, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))
        allowed = {x.strip().lower() for x in str(self.value("HOME_ASSISTANT_ALLOWED_DOMAINS", "") or "").split(",") if x.strip()}
        selected = {x.strip() for x in str(self.value("HOME_ASSISTANT_SELECTED_ENTITIES", "") or "").split(",") if x.strip()}
        tags = self.value("HOME_ASSISTANT_ENTITY_TAGS", {})
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = {}
        result = []
        for raw in payload if isinstance(payload, list) else []:
            entity_id = str(raw.get("entity_id", ""))
            domain = entity_id.split(".", 1)[0].lower() if "." in entity_id else ""
            if not entity_id or (allowed and domain not in allowed) or (selected and entity_id not in selected):
                continue
            attrs = raw.get("attributes") or {}
            state = raw.get("state")
            device_class = attrs.get("device_class")
            signal_name = None
            if domain == "binary_sensor" and device_class in {"presence", "occupancy", "motion"}:
                signal_name = "presence_detected" if str(state).lower() == "on" else "no_presence"
            result.append({
                "entity_id": entity_id,
                "state": state,
                "friendly_name": attrs.get("friendly_name", entity_id),
                "label": str(tags.get(entity_id) or attrs.get("friendly_name", entity_id)),
                "unit": attrs.get("unit_of_measurement") or "",
                "domain": domain,
                "device_class": device_class,
                "signal": signal_name,
                "last_updated": raw.get("last_updated"),
            })
        return sorted(result, key=lambda item: item["entity_id"])

    def publish(self, item: dict) -> None:
        self.latest[item["entity_id"]] = item
        message = {"type": "observation", "observation": {
            "source": "home_assistant",
            "kind": item["domain"],
            "subject": item["label"],
            "value": item["signal"] or item["state"],
            "unit": item["unit"],
            "confidence": 0.98,
            "observed_at": time.time(),
            "provenance": {"entity_id": item["entity_id"], "sensor_last_updated": item["last_updated"]},
        }}
        try:
            self.queue.put_nowait(message)
        except Exception:
            LOG.warning("Body bridge queue is full; dropping observation")

    def poll_loop(self) -> None:
        while not self.stop_event.is_set():
            if bool(self.value("BODY_PLUGIN_HOME_ASSISTANT_ENABLED", self.value("HOME_ASSISTANT_ENABLED", False))):
                try:
                    entities = self.discover()
                    self.save_discovery(entities)
                    for item in entities:
                        self.publish(item)
                    LOG.info("Home Assistant: %d selected entities", len(entities))
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    LOG.warning("Home Assistant poll failed: %s", exc)
                except Exception:
                    LOG.exception("Home Assistant poll failed")
            interval = max(1, min(300, int(self.value("HOME_ASSISTANT_POLL_INTERVAL", 5) or 5)))
            self.stop_event.wait(interval)

    def bridge_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            LOG.warning("Brain bridge disabled: install body_requirements.txt")
            return
        asyncio.run(self._bridge_session(websockets))

    async def _bridge_session(self, websockets) -> None:
        while not self.stop_event.is_set():
            url = str(self.value("BODY_BRIDGE_URL", "") or "").strip()
            token = str(self.value("BODY_BRIDGE_TOKEN", "") or "").strip()
            if not bool(self.value("BODY_BRIDGE_ENABLED", False)) or not url or not token:
                await asyncio.sleep(2)
                continue
            try:
                headers = {"Authorization": f"Bearer {token}"}
                kwargs = {"ping_interval": 20, "ping_timeout": 10, "close_timeout": 2}
                if url.startswith("wss://") and not bool(self.value("BODY_BRIDGE_VERIFY_TLS", True)):
                    kwargs["ssl"] = ssl._create_unverified_context()
                try:
                    socket = websockets.connect(url, additional_headers=headers, **kwargs)
                except TypeError:
                    socket = websockets.connect(url, extra_headers=headers, **kwargs)
                async with socket as ws:
                    await ws.send(json.dumps({"type": "hello", "device_id": self.value("BODY_BRIDGE_DEVICE_ID", "body-local"), "protocol": 1}))
                    LOG.info("Body bridge connected to %s", url)
                    while not self.stop_event.is_set():
                        try:
                            await ws.send(json.dumps(await asyncio.to_thread(self.queue.get, True, 1), ensure_ascii=False))
                        except Empty:
                            continue
            except Exception as exc:
                LOG.warning("Body bridge disconnected: %s", exc)
            await asyncio.sleep(max(1, float(self.value("BODY_BRIDGE_RECONNECT_SECONDS", 3))))

    def run(self) -> None:
        LOG.info("Standalone Body host started with %s", CONFIG_PATH)
        threads = [threading.Thread(target=self.poll_loop, name="body-home-assistant", daemon=True)]
        if bool(self.value("BODY_BRIDGE_ENABLED", False)):
            threads.append(threading.Thread(target=self.bridge_loop, name="body-brain-bridge", daemon=True))
        for thread in threads:
            thread.start()
        while not self.stop_event.wait(1):
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _load_dotenv()
    host = BodyHost()
    signal.signal(signal.SIGINT, lambda *_: host.stop_event.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: host.stop_event.set())
    host.run()
    LOG.info("Standalone Body host stopped")
