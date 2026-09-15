"""Connection Monitoring

Fix (2026-06-21): the previous implementation used a single module-level
ConnectionMonitor instance shared across every page and every client.
`start_monitoring()` guarded on a single `self.monitoring` flag, so only
the FIRST client to ever call it got a monitoring loop at all — every
subsequent client/page navigation silently got nothing. Worse, that one
loop kept running and referencing `ui.context.client` (the original
client's context) forever, including after that client disconnected and
NiceGUI garbage-collected it. The loop's `ping()` then called
`run_javascript()` on a deleted client object every 30s, producing:

  "Client has been deleted but is still being used. This is most likely
   a bug in your application code."

The `has_socket_connection` polling check was also unreliable — wrapped
in a broad except that silently swallowed any exception raised by
accessing a stale client, so `handle_disconnect()` often never fired and
the loop never self-terminated.

Now: monitoring is PER-CLIENT (one ConnectionMonitor instance created
fresh on each `start_monitoring()` call, scoped to the calling page's own
client), and uses NiceGUI's `client.on_disconnect()` hook for immediate,
reliable teardown — the same proven pattern already used in
cognitive_dashboard_page.py — instead of polling a property that can
itself throw on a half-torn-down client.
"""
import asyncio
import logging
import time
from typing import Dict
from nicegui import ui

logger = logging.getLogger(__name__)


class ConnectionMonitor:
    """Monitors a single client's connection and keeps it alive with periodic pings.
    One instance per client — never shared across pages or clients."""

    def __init__(self, client_id: str = "") -> None:
        self._client_id = client_id
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_ping = time.time()
        self.ping_interval = 30  # seconds
        self.monitoring = False
        self._task = None
        self._client = None

    def start_monitoring(self) -> None:
        """Start monitoring the CURRENT page's client. Safe to call once per
        page load; binds to the client active in this request context and
        registers a disconnect hook for clean, immediate teardown."""
        if self.monitoring:
            return
        try:
            client = ui.context.client
        except Exception:
            return  # no running client context — nothing to monitor

        self._client = client
        self.monitoring = True
        self.last_ping = time.time()

        try:
            client.on_disconnect(self._on_disconnect)
        except Exception as e:
            logger.debug(f"[ConnectionMonitor] on_disconnect registration failed: {e}")

        try:
            self._task = asyncio.get_running_loop().create_task(self._monitor_loop())
        except RuntimeError:
            self.monitoring = False  # no running loop yet

    def _on_disconnect(self) -> None:
        """Called by NiceGUI the moment this client actually disconnects.
        Immediately stops the loop — no polling delay, no stale references."""
        self.monitoring = False
        if self._task is not None:
            try:
                self._task.cancel()
            except Exception:
                pass

    async def _monitor_loop(self) -> None:
        client = self._client
        while self.monitoring:
            try:
                # has_socket_connection can itself raise on a half-torn-down
                # client; on_disconnect (above) is the primary teardown
                # signal now, this check is a secondary safety net only.
                if client is None or not getattr(client, 'has_socket_connection', True):
                    self.monitoring = False
                    break
                if time.time() - self.last_ping > self.ping_interval:
                    await self.ping()
                    self.last_ping = time.time()
            except Exception as e:
                logger.debug(f"[ConnectionMonitor] loop check error: {e}")
                self.monitoring = False
                break
            await asyncio.sleep(5)

    async def ping(self) -> None:
        """Send a lightweight keepalive. Bound to THIS instance's own
        client reference, never the ambient ui.context (which may point
        to a different/stale client by the time this fires)."""
        client = self._client
        if client is None or self.monitoring is False:
            return
        try:
            if getattr(client, 'has_socket_connection', False):
                client.run_javascript('console.log("ping")')
        except Exception as e:
            logger.debug(f"[ConnectionMonitor] ping failed (client likely gone): {e}")
            self.monitoring = False

    def stop_monitoring(self) -> None:
        self.monitoring = False
        if self._task is not None:
            try:
                self._task.cancel()
            except Exception:
                pass


class _ConnectionMonitorFactory:
    """
    Drop-in replacement for the old singleton `connection_monitor` object.
    Existing call sites do `connection_monitor.start_monitoring()` — this
    proxy creates a NEW per-client ConnectionMonitor on each such call
    (scoped to whatever client is active in ui.context at that moment) so
    every page/client genuinely gets its own monitor, while keeping every
    existing call site's code unchanged.
    """

    def __init__(self) -> None:
        self._by_client: Dict[int, ConnectionMonitor] = {}

    def start_monitoring(self) -> None:
        try:
            client = ui.context.client
            key = id(client)
        except Exception:
            # No client context — create a throwaway monitor that will
            # simply no-op (mirrors old behaviour of silently skipping).
            ConnectionMonitor().start_monitoring()
            return

        existing = self._by_client.get(key)
        if existing is not None and existing.monitoring:
            return  # this specific client already has an active monitor

        mon = ConnectionMonitor(client_id=str(key))
        self._by_client[key] = mon
        mon.start_monitoring()

        # Prune dead entries opportunistically so this dict doesn't grow
        # unbounded across a long-running server with many client visits.
        if len(self._by_client) > 200:
            self._by_client = {
                k: v for k, v in self._by_client.items() if v.monitoring
            }

    def stop_monitoring(self) -> None:
        try:
            client = ui.context.client
            key = id(client)
            mon = self._by_client.get(key)
            if mon:
                mon.stop_monitoring()
        except Exception:
            pass

    @property
    def reconnect_attempts(self) -> int:
        """Backward-compat accessor for code that reads
        connection_monitor.reconnect_attempts directly (e.g. settings_page.py
        status display). Resolves to the CURRENT client's own monitor —
        each client now has an independent count, this returns the
        relevant one for whoever is asking, not a shared global value."""
        try:
            client = ui.context.client
            mon = self._by_client.get(id(client))
            return mon.reconnect_attempts if mon else 0
        except Exception:
            return 0


connection_monitor = _ConnectionMonitorFactory()
