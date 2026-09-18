"""Authenticated Body-to-Brain WebSocket transport.

The bridge is deliberately optional.  A Body keeps collecting observations
locally when the bridge is disabled or the Brain is unavailable, then retries
the outbound connection without blocking the cognitive or sensor loops.
"""
from __future__ import annotations

import asyncio
import json
import logging
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class BodyBrainBridge:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self._queue: Queue[Dict[str, Any]] = Queue(maxsize=256)
        self._stop = Event()
        self._thread: Thread | None = None
        self.connected = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="lumina-body-brain-bridge", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout))
        self._thread = None
        self.connected = False

    def enqueue_observation(self, observation: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait({"type": "observation", "observation": observation})
        except Exception:
            logger.debug("[BodyBridge] outbound queue full; dropping oldest observation")

    def _run(self) -> None:
        try:
            asyncio.run(self._session_loop())
        except Exception:
            logger.debug("[BodyBridge] stopped", exc_info=True)

    async def _session_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.warning("[BodyBridge] disabled: install the optional 'websockets' package")
            return

        while not self._stop.is_set():
            url = str(self.runtime.config_value("BODY_BRIDGE_URL", "") or "").strip()
            token = str(self.runtime.config_value("BODY_BRIDGE_TOKEN", "") or "").strip()
            if not url or not token:
                await asyncio.sleep(2.0)
                continue
            try:
                parsed = urlparse(url)
                if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
                    raise ValueError("BODY_BRIDGE_URL must use ws:// or wss://")
                headers = {"Authorization": f"Bearer {token}"}
                kwargs: Dict[str, Any] = {
                    "ping_interval": 20,
                    "ping_timeout": 10,
                    "close_timeout": 2,
                    "max_size": 2_000_000,
                }
                if parsed.scheme == "wss" and not bool(self.runtime.config_value("BODY_BRIDGE_VERIFY_TLS", True)):
                    import ssl
                    kwargs["ssl"] = ssl._create_unverified_context()
                try:
                    socket = websockets.connect(url, additional_headers=headers, **kwargs)
                except TypeError:  # websockets < 14
                    socket = websockets.connect(url, extra_headers=headers, **kwargs)
                async with socket as websocket:
                    await websocket.send(json.dumps({
                        "type": "hello",
                        "device_id": self.runtime.config_value("BODY_BRIDGE_DEVICE_ID", "body-local"),
                        "protocol": 1,
                    }))
                    logger.info("[BodyBridge] connected to Brain at %s", url)
                    self.connected = True
                    while not self._stop.is_set():
                        try:
                            message = await asyncio.to_thread(self._queue.get, True, 1.0)
                        except Empty:
                            continue
                        await websocket.send(json.dumps(message, ensure_ascii=False))
            except Exception as exc:
                self.connected = False
                logger.debug("[BodyBridge] disconnected: %s", exc)
            if not self._stop.is_set():
                await asyncio.sleep(max(1.0, float(self.runtime.config_value("BODY_BRIDGE_RECONNECT_SECONDS", 3))))
