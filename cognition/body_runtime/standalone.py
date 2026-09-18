"""Standalone Body host: local plugins plus optional Brain bridge."""
from __future__ import annotations

import logging
import signal
import time

from cognition.body_runtime import get_body_runtime
from cognition.body_runtime.home_assistant_plugin import HomeAssistantBodyPlugin


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    runtime = get_body_runtime()
    plugin = HomeAssistantBodyPlugin(runtime)
    plugin.start()
    stopping = False

    def stop(*_args):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    logging.getLogger(__name__).info("Standalone Body Runtime started; config=data/body/config.json")
    try:
        while not stopping:
            time.sleep(1)
    finally:
        plugin.stop()
        runtime.stop()
        logging.getLogger(__name__).info("Standalone Body Runtime stopped")
