"""
pages/orchestrator_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orchestrator dashboard.
"""
import logging
import threading
import queue as _queue
import os
import tempfile
import shutil
import asyncio
import base64
import platform
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
import psutil

from nicegui import ui, app

import managers.settings_manager as _settings_mod
from managers.settings_manager import (
    config, save_settings, load_settings, AppSettings,
    on_settings_changed, export_settings, import_settings,
    reset_to_defaults, get_persona_name as _gpn,
)
from managers.llm_manager import create_llm_manager
from managers.memory_manager import create_memory_manager
from managers.audio_manager import create_audio_manager
from managers.conversational_audio import ConversationalAudioManager
from managers.vision_manager import StreamingVisionManager, FACE_RECOGNITION_AVAILABLE
from managers.user_manager import user_manager
from managers.security_manager import security, RateLimitError, SecurityViolation

from core.state import state
from core.connection import connection_monitor
from core.agent_controller import AgentController
from core.agent_state import AgentState, VisionOutput as _VisionOutput

from utils.user_switcher_ui import build_user_chip, refresh_chip
from utils.organism_ui import build_organism, set_core_state, update_user_hue, set_camera_feed
from utils.enhanced_css import ENHANCED_GLOBAL_CSS

from pages.shared import GLOBAL_CSS, _safe_ui, _drain_ui_queue, _build_proactive_prompt

_agent_state = AgentState()
controller = AgentController(state, agent=_agent_state)

logger = logging.getLogger(__name__)

@ui.page('/orchestrator')
async def orchestrator_page():
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout. On a loaded system this can be exceeded, killing
    # the page coroutine before anything renders — confirmed in production
    # logs (TimeoutError at cognitive_dashboard_page.py:398, same pattern).
    # Raised to 10s and wrapped so a genuine failure shows a visible
    # message instead of a silently blank page.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(f"[orchestrator_page] Client connection timed out after 10s")
        ui.label('⚠ Connection timed out — please reload.').classes(
            'text-yellow-400 text-lg font-bold p-8'
        )
        return
    ui.add_css(GLOBAL_CSS)
    ui.add_css("""
        .orch-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px 15px;
        }
        .orch-card-title {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 8px;
        }
        .drive-bar-wrap { margin-bottom: 5px; }
        .drive-label { font-size: 11px; color: #94a3b8; min-width: 110px; display: inline-block; }
        .drive-bar-bg {
            display: inline-block; width: 100px; height: 7px;
            background: rgba(255,255,255,0.07); border-radius: 4px;
            vertical-align: middle; overflow: hidden;
        }
        .drive-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
        .drive-value { font-size: 11px; color: #64748b; margin-left: 6px; }
        .activity-badge {
            display: inline-block; padding: 2px 8px; border-radius: 20px;
            font-size: 11px; font-weight: 600;
            background: rgba(99,102,241,0.18); color: #a5b4fc;
            border: 1px solid rgba(99,102,241,0.3);
        }
        .ws-item { font-size: 10px; color: #94a3b8; padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
        .ws-source { display: inline-block; min-width: 80px; color: #6366f1; font-weight: 600; }
        .trait-row { margin-bottom: 4px; }
        .trait-label { font-size: 11px; color: #94a3b8; min-width: 120px; display: inline-block; }
        .log-line { font-size: 10px; font-family: monospace; color: #64748b; padding: 1px 0; }
        .log-line.activity { color: #a5b4fc; }
        .log-line.error    { color: #f87171; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
        .dot-green  { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
        .dot-yellow { background: #eab308; box-shadow: 0 0 6px #eab30888; }
        .dot-red    { background: #ef4444; box-shadow: 0 0 6px #ef444488; }
        .orch-log-box { max-height: 160px; overflow-y: auto; font-family: monospace; font-size: 10px; }
        body, .q-page { overflow-y: auto !important; }
    """)

    connection_monitor.start_monitoring()

    # ── Header ────────────────────────────────────────────────────────────────
    with ui.header().classes('app-header'):
        with ui.row().classes('items-center gap-3 w-full'):
            ui.button(icon='monitor_heart', on_click=lambda: ui.navigate.to('/cognitive-dashboard')).props('flat dense round').style('color:#64748b').tooltip('Cognitive Health')
            ui.html('''
                <div style="width:38px;height:38px;border-radius:10px;
                     background:linear-gradient(135deg,#6366f1,#8b5cf6);
                     display:flex;align-items:center;justify-content:center;
                     font-size:20px">🧠</div>
            ''')
            ui.label('Orchestrator Dashboard').classes('app-title')
            ui.space()
            # Live status indicator
            _status_html = ui.html('<span class="status-dot dot-yellow"></span> <span style="font-size:12px;color:#64748b">Connecting…</span>')

    # ── Helper: get safe orchestrator ref ─────────────────────────────────────
    def _orch():
        return getattr(state, 'orchestrator', None)

    def _organism():
        if state.persona and hasattr(state.persona, '_organism'):
            return state.persona._organism
        return None

    # ── Activity log (in-memory ring buffer) ──────────────────────────────────
    import collections, datetime
    _activity_log = collections.deque(maxlen=60)

    # ── Layout ────────────────────────────────────────────────────────────────
    # ── Layout: wide scrollable grid ──────────────────────────────────────────
    with ui.element('div').style(
        'width:100%;max-width:1600px;margin:0 auto;padding:16px 20px 40px;'
        'display:flex;flex-direction:column;gap:12px;'
    ):

        # ════ ROW A: Status | Controls | Sleep Cycle | LLM Scheduler ══════════
        with ui.row().classes('w-full gap-3 flex-wrap'):

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">⚙️ Status</div>')
                _status_label = ui.html('<div style="font-size:12px;color:#94a3b8">Loading…</div>')

            with ui.element('div').classes('orch-card').style('flex:3;min-width:280px'):
                ui.html('<div class="orch-card-title">🎮 Controls</div>')
                with ui.row().classes('gap-2 flex-wrap'):
                    async def _do_start():
                        o = _orch()
                        if o and not o._running:
                            await o.start()
                            _activity_log.append(('START', 'Orchestrator started'))
                            ui.notify('Orchestrator started', color='green')
                    async def _do_stop():
                        o = _orch()
                        if o and o._running:
                            await o.stop()
                            _activity_log.append(('STOP', 'Orchestrator stopped'))
                            ui.notify('Orchestrator stopped', color='orange')
                    def _do_clear_ws():
                        o = _orch()
                        if o:
                            o.workspace.clear()
                            _activity_log.append(('CLEAR', 'Workspace cleared'))
                            ui.notify('Workspace cleared')
                    def _do_push_reflect():
                        o = _orch()
                        if o:
                            from core.orchestrator.event_system import REFLECTION_DUE, Event
                            o.event_system._events.append(Event(type=REFLECTION_DUE, priority=0.9, source='dashboard'))
                            _activity_log.append(('TRIGGER', 'Reflection triggered'))
                            ui.notify('Reflection triggered', color='blue')
                    def _do_push_consolidate():
                        o = _orch()
                        if o:
                            from core.orchestrator.event_system import CONSOLIDATION_DUE, Event
                            o.event_system._events.append(Event(type=CONSOLIDATION_DUE, priority=0.9, source='dashboard'))
                            _activity_log.append(('TRIGGER', 'Consolidation triggered'))
                            ui.notify('Consolidation triggered', color='purple')
                    def _do_evolve():
                        o = _orch()
                        if o:
                            from core.orchestrator.event_system import EVOLUTION_DUE, Event
                            o.event_system._events.append(Event(type=EVOLUTION_DUE, priority=0.9, source='dashboard'))
                            _activity_log.append(('TRIGGER', 'Evolution triggered'))
                            ui.notify('Evolution triggered', color='teal')
                    ui.button('▶',              on_click=lambda: asyncio.create_task(_do_start())).props('color=green  dense flat').tooltip('Start')
                    ui.button('⏹',              on_click=lambda: asyncio.create_task(_do_stop())).props('color=orange dense flat').tooltip('Stop')
                    ui.button('🔄 Reflect',     on_click=_do_push_reflect).props('color=indigo dense flat')
                    ui.button('🧠 Consolidate', on_click=_do_push_consolidate).props('color=purple dense flat')
                    ui.button('🌱 Evolve',      on_click=_do_evolve).props('color=teal   dense flat')
                    ui.button('🗑 WS',          on_click=_do_clear_ws).props('color=red    dense flat').tooltip('Clear workspace')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:180px'):
                ui.html('<div class="orch-card-title">🌙 Sleep Cycle</div>')
                _sleep_html = ui.html('<div style="color:#64748b;font-size:12px">Initialising…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:180px'):
                ui.html('<div class="orch-card-title">⚙️ LLM Scheduler</div>')
                _scheduler_html = ui.html('<div style="color:#64748b;font-size:12px">Ready.</div>')

        # ════ ROW B: Drives | Attractors | Pressure | Clock ══════════════════
        with ui.row().classes('w-full gap-3 flex-wrap'):

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">⚡ Drive Vector</div>')
                _drives_html = ui.html('<div style="color:#64748b;font-size:12px">Waiting…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:180px'):
                ui.html('<div class="orch-card-title">🧲 Attractors</div>')
                _attrs_html = ui.html('<div style="color:#64748b;font-size:12px">Waiting…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:180px'):
                ui.html('<div class="orch-card-title">🫀 Pressure</div>')
                _pressure_html = ui.html('<div style="color:#64748b;font-size:12px">Initialising…</div>')

            with ui.element('div').classes('orch-card').style('flex:1;min-width:140px'):
                ui.html('<div class="orch-card-title">🕐 Clock</div>')
                _clock_html = ui.html('<div style="color:#64748b;font-size:12px">Waiting…</div>')

        # ════ ROW C: Self-model | Inner Monologue | Predictive | Arch Health ══
        with ui.row().classes('w-full gap-3 flex-wrap'):

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">🪞 Self-Model</div>')
                _selfmodel_html = ui.html('<div style="color:#64748b;font-size:12px">Waiting…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">🧠 Inner Monologue</div>')
                _innermonologue_html = ui.html('<div style="color:#64748b;font-size:12px">Two-pass ready.</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">🔮 Predictive Mind</div>')
                _predict_html = ui.html('<div style="color:#64748b;font-size:12px">No predictions yet.</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:180px'):
                ui.html('<div class="orch-card-title">🏥 Arch Health</div>')
                _archmonitor_html = ui.html('<div style="color:#64748b;font-size:12px">No check yet.</div>')

        # ════ ROW D: Thought Stream | World Model | Cognitive Validator ════════
        with ui.row().classes('w-full gap-3 flex-wrap'):

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">💭 Thought Stream</div>')
                _thoughts_html = ui.html('<div style="color:#64748b;font-size:12px">No thoughts yet.</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">🌍 World Model</div>')
                _worldmodel_html = ui.html('<div style="color:#64748b;font-size:12px">Building…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">✅ Cognitive Validator</div>')
                _validator_html = ui.html('<div style="color:#64748b;font-size:12px">Awaiting responses…</div>')

        # ════ ROW E: Global Workspace | Narrative Identity ════════════════════
        with ui.row().classes('w-full gap-3 flex-wrap'):

            with ui.element('div').classes('orch-card').style('flex:3;min-width:300px'):
                with ui.row().classes('items-center justify-between mb-2'):
                    ui.html('<div class="orch-card-title" style="margin-bottom:0">🌐 Global Workspace <span style="font-weight:400;font-size:10px;color:#475569">(last 12)</span></div>')
                    ui.button('Clear', on_click=_do_clear_ws).props('flat dense').style('font-size:11px;color:#64748b')
                _workspace_html = ui.html('<div style="color:#64748b;font-size:12px">No broadcasts yet.</div>')

            with ui.element('div').classes('orch-card').style('flex:4;min-width:300px'):
                ui.html('<div class="orch-card-title">📖 Narrative Identity</div>')
                _narrative_html = ui.html('<div style="color:#64748b;font-size:12px">Building identity…</div>')

        # ════ ROW F: Security | Activity Log | Inject Event ══════════════════
        with ui.row().classes('w-full gap-3 flex-wrap'):

            with ui.element('div').classes('orch-card').style('flex:2;min-width:220px'):
                with ui.row().classes('items-center justify-between mb-2'):
                    ui.html('<div class="orch-card-title" style="margin-bottom:0">🔒 Security</div>')
                    def _refresh_sec():
                        _render_security()
                    ui.button('Refresh', on_click=_refresh_sec).props('flat dense').style('font-size:11px;color:#64748b')
                _security_html = ui.html('<div style="color:#64748b;font-size:12px">Loading…</div>')

            with ui.element('div').classes('orch-card').style('flex:3;min-width:260px'):
                ui.html('<div class="orch-card-title">📋 Activity Log</div>')
                _log_html = ui.html('<div class="orch-log-box" style="color:#64748b;font-size:11px">No activity yet.</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:200px'):
                ui.html('<div class="orch-card-title">💉 Inject Event</div>')
                _inj_type   = ui.select({
                    'USER_MESSAGE': '💬 User Message',
                    'CONTRADICTION_FOUND': '⚡ Contradiction',
                    'CURIOSITY_PEAK': '🔍 Curiosity Peak',
                    'LOW_ENERGY': '🔋 Low Energy',
                    'PROACTIVE_TRIGGER': '🤖 Proactive',
                }, value='USER_MESSAGE', label='Event type').classes('w-full mb-2')
                _inj_payload = ui.input(label='Payload (optional)').classes('w-full mb-2')
                _inj_prio    = ui.slider(min=0.1, max=1.0, step=0.05, value=0.7).classes('w-full mb-1')
                ui.html('<div style="font-size:10px;color:#64748b;margin-bottom:6px">Priority</div>')

                def _inject_event():
                    o = _orch()
                    if not o:
                        ui.notify('Orchestrator not available', color='red')
                        return
                    from core.orchestrator.event_system import Event
                    etype   = _inj_type.value
                    payload = _inj_payload.value or None
                    prio    = _inj_prio.value
                    o.event_system._events.append(Event(type=etype, payload=payload, priority=prio, source='dashboard'))
                    _activity_log.append(('INJECT', f'{etype} prio={prio:.2f}'))
                    ui.notify(f'Injected {etype}', color='indigo')
                ui.button('Inject →', on_click=_inject_event).props('color=indigo dense').classes('w-full')

        # ════ ROW G: Cognitive Observatory ════════════════════════════════════
        with ui.row().classes('w-full gap-3 flex-wrap'):
            with ui.element('div').classes('orch-card').style('flex:1;min-width:180px'):
                ui.html('<div class=\"orch-card-title\">🔭 Observatory</div>')
                _obs_status_html = ui.html('<div style="color:#64748b;font-size:12px">Loading…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:220px'):
                ui.html('<div class=\"orch-card-title\">🧬 Cognitive Genome</div>')
                _genome_html = ui.html('<div style="color:#64748b;font-size:12px">Calibrating baseline…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:220px'):
                ui.html('<div class=\"orch-card-title\">🌡️ Identity Drift</div>')
                _drift_html = ui.html('<div style="color:#64748b;font-size:12px">Building baseline…</div>')

            with ui.element('div').classes('orch-card').style('flex:2;min-width:220px'):
                ui.html('<div class=\"orch-card-title\">⚡ Emergence Radar</div>')
                _emerge_html = ui.html('<div style="color:#64748b;font-size:12px">Observing…</div>')

    # ── Live update timer ─────────────────────────────────────────────────────
    def _colour_bar(value: float, low_is_bad: bool = False) -> str:
        """Return a CSS colour for a 0-1 bar."""
        if low_is_bad:
            # green when high, red when low
            r = int(239 * (1 - value) + 34 * value)
            g = int(68  * (1 - value) + 197 * value)
            b = int(68  * (1 - value) + 94  * value)
        else:
            # purple gradient
            r, g, b = 99, 102, 241
        return f"rgb({r},{g},{b})"

    def _drive_row(name: str, value: float, low_is_bad: bool = False) -> str:
        pct   = int(value * 100)
        width = max(2, pct)
        col   = _colour_bar(value, low_is_bad)
        label_nice = name.replace("_", " ").title()
        return (
            f'<div class="drive-bar-wrap">'
            f'<span class="drive-label">{label_nice}</span>'
            f'<span class="drive-bar-bg"><span class="drive-bar-fill" style="width:{width}%;background:{col}"></span></span>'
            f'<span class="drive-value">{pct}%</span>'
            f'</div>'
        )

    _sec_audit_cache = {"warns": None}   # run audit once, cache result

    def _render_security():
        from managers.security_manager import security as _sec
        from managers.settings_manager import config as _scfg
        lines = []
        # Config audit — cached after first run to avoid log spam
        if _sec_audit_cache["warns"] is None:
            cfg_dict = _scfg.model_dump() if hasattr(_scfg, 'model_dump') else _scfg.dict()
            _sec_audit_cache["warns"] = _sec.audit_config(cfg_dict)
        warns = _sec_audit_cache["warns"]
        if warns:
            for w in warns:
                lines.append(f'<div style="font-size:11px;color:#f87171;margin-bottom:3px">{w}</div>')
        else:
            lines.append('<div style="font-size:11px;color:#22c55e;margin-bottom:6px">✓ No hardcoded secrets found in config.json</div>')
        # Network binding
        lines.append('<div style="font-size:11px;color:#94a3b8;margin-bottom:3px">🌐 Network: <b style=\'color:#22c55e\'>localhost only (127.0.0.1)</b></div>')
        # Rate limiter
        remaining = _sec.remaining_requests('default')
        col = '#22c55e' if remaining > 10 else '#eab308' if remaining > 3 else '#ef4444'
        lines.append(f'<div style="font-size:11px;color:#94a3b8;margin-bottom:3px">⚡ Rate limit: <b style="color:{col}">{remaining}/20 requests remaining</b> (60s window)</div>')

        # ── Asimov safety stats ───────────────────────────────────────────
        try:
            from cognition.safety_constraints import get_constraints
            _sc = get_constraints()
            _stats = _sc.stats()
            total = _stats["total_blocked"]
            col_s = '#22c55e' if total == 0 else '#eab308'
            lines.append(f'<div style="font-size:11px;color:#94a3b8;margin-top:8px;margin-bottom:3px">🛡️ Safety laws: <b style="color:{col_s}">{total} blocked</b> '
                         f'(L1: {_stats["law1_blocks"]} | L2: {_stats["law2_blocks"]} | crisis: {_stats["crisis_blocks"]})</div>')
            # Recent safety events
            recent_safety = _sc.recent_audit(5)
            if recent_safety:
                lines.append('<div style="font-size:11px;color:#64748b;margin-top:4px;margin-bottom:2px">📋 Safety events:</div>')
                for ev in recent_safety:
                    lines.append(
                        f'<div style="font-family:monospace;font-size:10px;color:#f87171">'
                        f'{ev.get("timestamp","")} {ev.get("law","").upper()} '
                        f'[{ev.get("layer","")}] sev={ev.get("severity","")}'
                        f'</div>'
                    )
        except Exception:
            pass

        # Recent audit log
        recent = _sec.recent_audit(10)
        if recent:
            lines.append('<div style="font-size:11px;color:#64748b;margin-top:8px;margin-bottom:3px">📋 Recent audit events:</div>')
            for line in reversed(recent[-8:]):
                col2 = '#f87171' if any(k in line for k in ('INJECTION','TRAVERSAL','HARDCODED')) else '#64748b'
                lines.append(f'<div style="font-family:monospace;font-size:10px;color:{col2}">{line}</div>')
        else:
            lines.append('<div style="font-size:11px;color:#334155;margin-top:6px">No security events logged yet.</div>')
        _security_html.set_content(''.join(lines))

    def _render():
        # Guard: if any element has been deleted (tab switch, clear, reconnect)
        # between renders, catch silently instead of crashing the timer callback.
        import time as _time
        o    = _orch()
        org  = _organism()

        # ── Header status dot ──────────────────────────────────────────────
        if o and o._running:
            try:
                _status_html.set_content('<span class="status-dot dot-green"></span><span style="font-size:12px;color:#22c55e">Running</span>')
            except Exception: pass
        elif o:
            try:
                _status_html.set_content('<span class="status-dot dot-yellow"></span><span style="font-size:12px;color:#eab308">Stopped</span>')
            except Exception: pass
        else:
            try:
                _status_html.set_content('<span class="status-dot dot-red"></span><span style="font-size:12px;color:#ef4444">Not available</span>')
            except Exception: pass

        # ── Status card ────────────────────────────────────────────────────
        if o:
            s = o.status()
            running_str = "🟢 Running" if s["running"] else "🔴 Stopped"
            ws_div = s["workspace"].get("diversity", None)
            ws_div_html = (
                f' · diversity <b style="color:{"#22c55e" if ws_div and ws_div > 0.5 else "#f59e0b" if ws_div else "#ef4444"}">'
                f'{ws_div:.0%}</b>' if ws_div is not None else ""
            )
            lines = [
                f'<div style="font-size:13px;color:#e2e8f0;margin-bottom:6px"><b>{running_str}</b></div>',
                f'<div style="font-size:12px;color:#94a3b8">Cycles: <b style="color:#a5b4fc">{s["cycle_count"]}</b></div>',
                f'<div style="font-size:12px;color:#94a3b8">Last action: <span class="activity-badge">{s["last_activity"]}</span></div>',
                f'<div style="font-size:12px;color:#94a3b8;margin-top:4px">WS items: {s["workspace"]["total_items"]}{ws_div_html}</div>',
            ]
            try:
                _status_label.set_content("".join(lines))
            except Exception: pass
        else:
            try:
                _status_label.set_content('<div style="font-size:12px;color:#64748b">Orchestrator not initialised</div>')
            except Exception: pass

        # ── Drive vector ───────────────────────────────────────────────────
        if o:
            d = o._drives.compute()
            rows = "".join([
                _drive_row("curiosity",     d.curiosity,     True),
                _drive_row("coherence",     d.coherence,     True),
                _drive_row("energy",        d.energy,        True),
                _drive_row("social",        d.social,        False),
                _drive_row("goal_progress", d.goal_progress, True),
                _drive_row("homeostasis",   d.homeostasis,   True),
            ])
            try:
                _drives_html.set_content(rows)
            except Exception: pass

        # ── Attractor traits ───────────────────────────────────────────────
        if org and hasattr(org, 'attractors'):
            traits = org.attractors.all()
            rows = []
            for name, val in traits.items():
                pct = int(val * 100)
                rows.append(
                    f'<div class="trait-row">'
                    f'<span class="trait-label">{name.replace("_"," ").title()}</span>'
                    f'<span class="drive-bar-bg" style="width:120px"><span class="drive-bar-fill" style="width:{max(2,pct)}%;background:#8b5cf6"></span></span>'
                    f'<span class="drive-value">{pct}%</span>'
                    f'</div>'
                )
            try:
                _attrs_html.set_content("".join(rows))
            except Exception: pass

        # ── Self-model ─────────────────────────────────────────────────────
        if org and hasattr(org, 'self_model'):
            sm = org.self_model
            s  = sm.summary()
            conf_col   = "#22c55e" if sm.is_confident() else "#eab308" if s["confidence"] > 0.4 else "#ef4444"
            load_col   = "#ef4444" if sm.is_overloaded() else "#eab308" if s["cognitive_load"] > 0.5 else "#22c55e"
            caps = org.self_model.capabilities
            cap_html = "".join([
                f'<div style="font-size:11px;color:#64748b;margin-bottom:3px">'
                f'<span style="color:#94a3b8;min-width:120px;display:inline-block">{d}</span>'
                f'<span class="drive-bar-bg" style="width:80px"><span class="drive-bar-fill" style="width:{max(2,int(c.score*100))}%;background:#6366f1"></span></span>'
                f'<span class="drive-value">{int(c.score*100)}%</span></div>'
                for d, c in caps.items()
            ])
            try:
                _selfmodel_html.set_content(
                f'<div style="font-size:12px;color:#94a3b8;margin-bottom:8px">'
                f'Confidence: <b style="color:{conf_col}">{int(s["confidence"]*100)}%</b>  '
                f'Load: <b style="color:{load_col}">{int(s["cognitive_load"]*100)}%</b>  '
                f'Total interactions: <b style="color:#a5b4fc">{s["total_interactions"]}</b></div>'
                f'{cap_html}'
            )
            except Exception: pass

        # ── Cognitive clock ────────────────────────────────────────────────
        if o:
            clk = o._clock.status()
            def _fmt(secs):
                secs = int(secs)
                if secs < 60:   return f"{secs}s"
                if secs < 3600: return f"{secs//60}m {secs%60}s"
                return f"{secs//3600}h {(secs%3600)//60}m"
            rows = [
                f'<div style="font-size:12px;color:#94a3b8;margin-bottom:5px">⚡ Fast (30 s): <b style="color:#a5b4fc">{_fmt(clk["fast_in_sec"])}</b></div>',
                f'<div style="font-size:12px;color:#94a3b8;margin-bottom:5px">🔄 Reflect (2 min): <b style="color:#a5b4fc">{_fmt(clk["medium_in_sec"])}</b></div>',
                f'<div style="font-size:12px;color:#94a3b8;margin-bottom:5px">🧠 Consolidate (10 min): <b style="color:#a5b4fc">{_fmt(clk["slow_in_sec"])}</b></div>',
                f'<div style="font-size:12px;color:#94a3b8">🌱 Evolve (1 h): <b style="color:#a5b4fc">{_fmt(clk["evolution_in_sec"])}</b></div>',
            ]
            try:
                _clock_html.set_content("".join(rows))
            except Exception: pass

        # ── Global Workspace ───────────────────────────────────────────────
        if o:
            items = o.workspace.recent(12)
            if items:
                import time as _t
                def _age(ts):
                    a = int(_t.time() - ts)
                    return f"{a}s ago"
                ws_html = "".join([
                    f'<div class="ws-item">'
                    f'<span class="ws-source">{i.source}</span>'
                    f'<span style="color:#e2e8f0">{str(i.content)[:80]}</span>'
                    f'<span style="float:right;color:#334155;font-size:10px">{_age(i.timestamp)}</span>'
                    f'</div>'
                    for i in reversed(items)
                ])
                try:
                    _workspace_html.set_content(ws_html)
                except Exception: pass
            else:
                try:
                    _workspace_html.set_content('<div style="color:#334155;font-size:12px">No broadcasts yet.</div>')
                except Exception: pass

        # ── Activity log ───────────────────────────────────────────────────
        # Also pull from orchestrator last_activity
        if o and o._last_activity not in ('none', 'idle_reflection'):
            import time as _t
            _activity_log.append((o._last_activity, _t.strftime("%H:%M:%S")))

        if _activity_log:
            lines = []
            for entry in reversed(_activity_log):
                tag, info = entry
                col = "#a5b4fc" if tag not in ("START","STOP","ERROR") else ("#22c55e" if tag=="START" else "#ef4444" if tag=="ERROR" else "#eab308")
                lines.append(f'<div class="log-line" style="color:{col}">[{tag}] {info}</div>')
            try:
                _log_html.set_content(f'<div class="orch-log-box">{"".join(lines[:40])}</div>')
            except Exception: pass

        # ── v3: Thought Stream panel ───────────────────────────────────────
        try:
            ts = getattr(org, 'thought_stream', None) if org else None
            if ts:
                recent = ts.recent_dicts(6)
                if recent:
                    lines = []
                    for t in reversed(recent):
                        src   = t.get('source','?')
                        text  = t.get('content','')[:90]
                        pri   = t.get('priority', 0)
                        col   = {'curiosity':'#818cf8','emotion':'#fb7185','goals':'#34d399',
                                 'tension':'#f59e0b','memory':'#a78bfa','meta':'#94a3b8',
                                 'self_model':'#67e8f9'}.get(src, '#cbd5e1')
                        lines.append(
                            f'<div style="margin:1px 0;padding:2px 6px;background:#1e293b;'
                            f'border-left:3px solid {col};font-size:11px;color:#e2e8f0">'
                            f'<span style="color:{col};font-weight:600">{src}</span> '
                            f'<span style="color:#94a3b8">p={pri:.2f}</span> — {text}</div>'
                        )
                    try:
                        _thoughts_html.set_content("".join(lines))
                    except Exception: pass
                else:
                    try:
                        _thoughts_html.set_content('<div style="color:#64748b;font-size:12px">No thoughts yet.</div>')
                    except Exception: pass
        except Exception:
            pass

        # ── v3: Predictive Mind panel ──────────────────────────────────────
        try:
            pm = getattr(org, 'predictive_mind', None) if org else None
            if pm:
                summ = pm.summary()
                acc  = summ.get('accuracy', 1.0)
                tot  = summ.get('total_predictions', 0)
                conf = summ.get('avg_confidence', 0.65)
                surp = summ.get('recent_surprises', 0)
                acc_col  = '#22c55e' if acc > 0.7 else '#f59e0b' if acc > 0.4 else '#ef4444'
                surp_col = '#ef4444' if surp > 3 else '#f59e0b' if surp > 1 else '#22c55e'

                # Model Stability metrics
                stab = pm.stability_metrics() if hasattr(pm, 'stability_metrics') else {}
                si   = stab.get('surprise_index', 0.0)
                cb   = stab.get('confirmation_bias', 0.0)
                nr   = stab.get('novelty_rate', 0.0)
                tq   = stab.get('topic_quality', 1.0)
                drift= stab.get('model_drift', 0.0)
                hlth = stab.get('health', '')

                si_col = '#22c55e' if 0.10 <= si <= 0.40 else '#f59e0b' if si < 0.10 else '#ef4444'
                cb_col = '#ef4444' if cb > 0.85 else '#f59e0b' if cb > 0.70 else '#22c55e'
                tq_col = '#22c55e' if tq >= 0.7 else '#f59e0b' if tq >= 0.5 else '#ef4444'
                dr_col = '#ef4444' if drift > 0.3 else '#f59e0b' if drift > 0.1 else '#22c55e'

                try:
                    _predict_html.set_content(
                    f'<div style="font-size:12px;color:#cbd5e1">'
                    f'<div><span style="color:#94a3b8">Predictions:</span> <b>{tot}</b></div>'
                    f'<div><span style="color:#94a3b8">Accuracy:</span> '
                    f'<b style="color:{acc_col}">{acc:.0%}</b></div>'
                    f'<div><span style="color:#94a3b8">Confidence:</span> {conf:.0%}</div>'
                    f'<div><span style="color:#94a3b8">Recent surprises:</span> '
                    f'<b style="color:{surp_col}">{surp}</b></div>'
                    + (f'<div style="margin-top:6px;color:#64748b;font-size:10px">🧠 Model Stability {hlth}</div>'
                       f'<div style="font-size:10px"><span style="color:#94a3b8">Surprise idx:</span> '
                       f'<b style="color:{si_col}">{si:.2f}</b> '
                       f'<span style="color:#94a3b8">| Confirm bias:</span> '
                       f'<b style="color:{cb_col}">{cb:.2f}</b></div>'
                       f'<div style="font-size:10px"><span style="color:#94a3b8">Novelty:</span> {nr:.2f} '
                       f'<span style="color:#94a3b8">| Topic quality:</span> '
                       f'<b style="color:{tq_col}">{tq:.2f}</b> '
                       f'<span style="color:#94a3b8">| Drift:</span> '
                       f'<b style="color:{dr_col}">{drift:.2f}</b></div>'
                       if stab else '')
                    + f'</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── v3: Narrative Identity panel ───────────────────────────────────
        try:
            ni = getattr(org, 'narrative_identity', None) if org else None
            if ni:
                summ = ni.summary()
                desc = ni.self_description()
                vals = ", ".join(summ.get('core_values', [])[:3])
                try:
                    _narrative_html.set_content(
                    f'<div style="font-size:12px;color:#cbd5e1">'
                    f'<div style="color:#a5b4fc;margin-bottom:4px">{desc}</div>'
                    f'<div><span style="color:#94a3b8">Chapters:</span> {summ.get("chapters",0)} | '
                    f'<span style="color:#94a3b8">Beliefs:</span> {summ.get("beliefs",0)} | '
                    f'<span style="color:#94a3b8">Milestones:</span> {summ.get("milestones",0)}</div>'
                    f'<div style="color:#64748b;margin-top:4px">Values: {vals}</div>'
                    f'</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── World Model panel (extended) ────────────────────────────────
        try:
            wm = getattr(org, 'world_model', None) if org else None
            if wm:
                summ   = wm.summary()
                users  = summ.get('users', 0)
                causes = summ.get('causal_beliefs', 0)
                auto_c = summ.get('auto_causal', 0)
                topics = summ.get('topic_nodes', 0)
                # Current user profile
                _uid    = getattr(org, '_current_user_id', 'default')
                profile = wm.get_user(_uid)
                user_html = ""
                if profile and profile.interaction_count >= 2:
                    top_t = profile.top_topics(3)
                    top_e = profile.top_expertise(2)
                    trend = profile.relational.engagement_trend() if profile.relational else "unknown"
                    trend_col = "#22c55e" if trend=="rising" else "#ef4444" if trend=="declining" else "#94a3b8"
                    rel   = profile.relational.relationship_label() if profile.relational else "neutral"
                    exp_str = " | ".join(f"{d}:{l}" for d,l in top_e) or "-"
                    user_html = (
                        f'<div style="border-top:1px solid #334155;margin-top:5px;padding-top:4px">'
                        f'<div style="color:#a5b4fc">{profile.display_name or _uid}'
                        f' <span style="color:#64748b;font-size:10px">({profile.interaction_count} turns)</span></div>'
                        f'<div style="font-size:10px;color:#94a3b8">Topics: {", ".join(top_t) or "-"}</div>'
                        f'<div style="font-size:10px;color:#94a3b8">Expertise: {exp_str}</div>'
                        f'<div style="font-size:10px">'
                        f'<span style="color:#94a3b8">Rel:</span> {rel} | '
                        f'<span style="color:#94a3b8">Trend:</span> <span style="color:{trend_col}">{trend}</span>'
                        f'</div></div>'
                    )
                try:
                    _worldmodel_html.set_content(
                    f'<div style="font-size:12px;color:#cbd5e1">'
                    f'<div><span style="color:#94a3b8">Users:</span> {users} | '
                    f'<span style="color:#94a3b8">Topics:</span> {topics} | '
                    f'<span style="color:#94a3b8">Causal:</span> {causes} ({auto_c} auto)</div>'
                    f'{user_html}</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── Cognitive Validator panel ───────────────────────────────────
        try:
            cv = getattr(org, 'cognitive_validator', None) if org else None
            if cv:
                summ = cv.summary()
                total   = summ.get('total_validated', 0)
                misrate = summ.get('misalignment_rate', 0.0)
                avg     = summ.get('recent_avg_score', 0.0)
                mis_col = '#ef4444' if misrate > 0.4 else '#f59e0b' if misrate > 0.2 else '#22c55e'
                avg_col = '#22c55e' if avg > 0.6 else '#f59e0b' if avg > 0.4 else '#ef4444'
                recent_scores = cv.recent_scores(5)
                score_str = " ".join(
                    f'<span style="color:{"#22c55e" if s>0.6 else "#f59e0b" if s>0.4 else "#ef4444"}">{s:.2f}</span>'
                    for s in recent_scores
                )
                try:
                    _validator_html.set_content(
                    f'<div style="font-size:12px;color:#cbd5e1">'
                    f'<div><span style="color:#94a3b8">Validated:</span> {total}</div>'
                    f'<div><span style="color:#94a3b8">Misalignment:</span> '
                    f'<b style="color:{mis_col}">{misrate:.0%}</b></div>'
                    f'<div><span style="color:#94a3b8">Avg alignment:</span> '
                    f'<b style="color:{avg_col}">{avg:.0%}</b></div>'
                    f'<div style="margin-top:4px;font-size:10px;color:#64748b">Recent: {score_str}</div>'
                    f'</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── Inner Monologue panel ────────────────────────────────────────
        try:
            # InnerMonologueEngine is on persona_bridge, access via state.persona
            _pb = getattr(state, 'persona', None)
            _im = getattr(_pb, '_inner_monologue', None) if _pb else None
            if _im:
                summ = _im.summary()
                enabled   = summ.get('enabled', False)
                total     = summ.get('total_passes', 0)
                fail_rate = summ.get('failure_rate', 0.0)
                avg_ms    = summ.get('avg_pass1_ms', 0.0)
                status_col = '#22c55e' if enabled else '#ef4444'
                status_txt = 'Active' if enabled else 'Disabled'
                fail_col  = '#ef4444' if fail_rate > 0.3 else '#f59e0b' if fail_rate > 0.1 else '#22c55e'
                last_reasoning = ""
                if _im.last_reasoning():
                    lr = _im.last_reasoning()
                    last_reasoning = (
                        f'<div style="color:#64748b;font-size:10px;margin-top:4px">'
                        f'Intent: {lr.intent_understood[:50]}</div>'
                    )
                try:
                    _innermonologue_html.set_content(
                    f'<div style="font-size:12px;color:#cbd5e1">'
                    f'<div><span style="color:#94a3b8">Status:</span> '
                    f'<b style="color:{status_col}">{status_txt}</b></div>'
                    f'<div><span style="color:#94a3b8">Passes:</span> {total} | '
                    f'<span style="color:#94a3b8">Fail rate:</span> '
                    f'<b style="color:{fail_col}">{fail_rate:.0%}</b></div>'
                    f'<div><span style="color:#94a3b8">Avg pass 1:</span> {avg_ms:.0f}ms</div>'
                    f'{last_reasoning}</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── PressureSystem panel ────────────────────────────────────────
        try:
            ps = getattr(org, 'pressure', None) if org else None
            if ps:
                summ = ps.summary()
                top  = ps.top_drives(4)   # show top 4 to include uncertainty
                PCOL = {"urgent":"#ef4444","elevated":"#f59e0b","present":"#60a5fa","quiet":"#475569"}
                bars = ""
                for drive, ep in top:
                    info  = summ.get(drive, {})
                    if not info:
                        continue
                    color = PCOL.get(info.get("label","quiet"), "#475569")
                    pct   = int(ep * 100)
                    esc   = " ↑" if info.get("escalating") else ""
                    mult  = info.get("urgency_multiplier", 1.0)
                    bars += (
                        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                        f'<span style="color:#94a3b8;min-width:74px;font-size:10px">{drive}</span>'
                        f'<div style="background:#1e293b;flex:1;border-radius:3px;height:7px">'
                        f'<div style="height:100%;width:{pct}%;background:{color};border-radius:3px"></div>'
                        f'</div>'
                        f'<span style="color:{color};font-size:10px;min-width:52px">{pct}%{esc} ×{mult:.1f}</span>'
                        f'</div>'
                    )
                try:
                    _pressure_html.set_content(
                    f'<div style="font-size:11px;color:#cbd5e1">'
                    f'<div style="color:#94a3b8;margin-bottom:5px">Dominant: '
                    f'<span style="color:#e2e8f0">{ps.dominant_drive()}</span></div>'
                    f'{bars}</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── Sleep Cycle panel ────────────────────────────────────────────
        try:
            sc = getattr(org, 'sleep_cycle', None) if org else None
            if sc:
                summ  = sc.summary()
                phase = summ["phase"]
                elapsed = summ["phase_elapsed_s"]
                idle    = summ["idle_seconds"]
                queued  = summ["queued_tasks"]
                PHASE_COL = {"active":"#22c55e","idle":"#60a5fa","sleep":"#818cf8","dream":"#f472b6"}
                PHASE_ICON= {"active":"🟢","idle":"🔵","sleep":"🟣","dream":"🩷"}
                col   = PHASE_COL.get(phase, "#94a3b8")
                icon  = PHASE_ICON.get(phase, "⚪")
                qstr  = ", ".join(queued) if queued else "none"
                mins  = elapsed // 60; secs = elapsed % 60
                try:
                    _sleep_html.set_content(
                    f'<div style="font-size:11px;color:#cbd5e1">'
                    f'<div style="font-size:14px;color:{col};margin-bottom:4px">'
                    f'{icon} <strong>{phase.upper()}</strong></div>'
                    f'<div style="color:#94a3b8">In phase: {mins}m {secs}s</div>'
                    f'<div style="color:#94a3b8">Idle: {idle//60}m {idle%60}s</div>'
                    f'<div style="color:#64748b;margin-top:3px">Queued: {qstr}</div>'
                    f'</div>'
                )
                except Exception: pass
        except Exception:
            pass

        # ── LLM Scheduler panel ──────────────────────────────────────────
        try:
            from core.llm_scheduler import llm_scheduler as _sched
            ss = _sched.summary()
            busy_col = "#ef4444" if ss["currently_busy"] else "#22c55e"
            busy_txt = ss.get("current_holder","—") if ss["currently_busy"] else "free"
            try:
                _scheduler_html.set_content(
                f'<div style="font-size:11px;color:#cbd5e1">'
                f'<div style="color:{busy_col};margin-bottom:4px">● {busy_txt}</div>'
                f'<div style="color:#94a3b8">Calls: {ss["total_calls"]} '
                f'| Skipped: {ss["skipped_calls"]} ({ss["skip_rate_pct"]}%)</div>'
                f'<div style="color:#94a3b8">Avg wait: {ss["avg_wait_ms"]}ms '
                f'| Peak: {ss["peak_wait_ms"]}ms</div>'
                f'</div>'
            )
            except Exception: pass
        except Exception:
            pass

        # ── Architecture Monitor panel ───────────────────────────────────
        try:
            am = getattr(org, 'arch_monitor', None) if org else None
            if am:
                report = am.latest_report()
                if report:
                    health = report.overall_health
                    h_col  = '#22c55e' if health > 0.65 else '#f59e0b' if health > 0.4 else '#ef4444'
                    status = '✅ Healthy' if report.is_healthy() else ('🔴 Critical' if report.critical else '⚠️ Degraded')
                    comps  = report.component_scores
                    comp_html = " ".join(
                        f'<span style="color:#94a3b8">{k[:8]}:</span>'
                        f'<span style="color:{"#22c55e" if v>0.6 else "#f59e0b" if v>0.35 else "#ef4444"}">{v:.0%}</span>'
                        for k, v in comps.items()
                    )
                    warn_html = ""
                    if report.warnings:
                        warn_html = f'<div style="color:#f59e0b;font-size:10px;margin-top:3px">{report.warnings[0][:60]}</div>'
                    # Identity queue status
                    id_html = ""
                    try:
                        ai_sys = getattr(org, 'ai_system', None)
                        id_sys = getattr(ai_sys, 'identity_system', None) if ai_sys else None
                        if id_sys:
                            pending = id_sys.pending_memory_count
                            id_col  = "#ef4444" if pending > 50 else "#f59e0b" if pending > 10 else "#22c55e"
                            id_html = f'<div style="color:{id_col};font-size:10px;margin-top:2px">🧬 Identity queue: {pending}</div>'
                    except Exception:
                        pass
                    try:
                        _archmonitor_html.set_content(
                        f'<div style="font-size:12px;color:#cbd5e1">'
                        f'<div><b style="color:{h_col}">{status}</b> '
                        f'({health:.0%})</div>'
                        f'<div style="margin-top:4px;font-size:10px">{comp_html}</div>'
                        f'{warn_html}{id_html}</div>'
                    )
                    except Exception: pass
        except Exception:
            pass

    def _render_observatory():
        """Render all 4 Cognitive Observatory panels."""
        # Guard against deleted elements between 2s render ticks
        try:
            obs = getattr(_organism(), 'observatory', None) if _organism() else None
            if not obs:
                try:
                    _obs_status_html.set_content('<div style="color:#64748b;font-size:12px">Observatory not available</div>')
                except Exception: pass
                return

            summ = obs.summary()
            if not summ.get('ready'):
                try:
                    _obs_status_html.set_content('<div style="color:#64748b;font-size:12px">Calibrating…</div>')
                except Exception: pass
                return

            # ── Status panel ────────────────────────────────────────────────
            ccs = summ['ccs'];  rdi = summ['rdi']
            gei = summ['gei'];  idx = summ['idx']
            str_s = summ['strangeness']; emg = summ['emergence']

            def _col(v, lo=0.3, hi=0.7):
                if v < lo:   return '#ef4444'
                if v > hi:   return '#f59e0b'
                return '#22c55e'
            def _trend_arrow(t):
                if t >  0.03: return '<span style="color:#22c55e">↑</span>'
                if t < -0.03: return '<span style="color:#ef4444">↓</span>'
                return '<span style="color:#64748b">→</span>'

            locked = "✅ locked" if summ['baseline_locked'] else f"⏳ calibrating ({summ['ticks']}/{obs.BASELINE_MIN_SAMPLES})"
            notes_html = ""
            if summ['notes']:
                notes_html = "<br>" + "<br>".join(f'<span style="color:#f59e0b;font-size:10px">• {n}</span>' for n in summ['notes'][:3])

            status_html = (
                f'<div style="font-size:12px;color:#cbd5e1">'
                f'<div style="margin-bottom:4px"><span style="color:#94a3b8">Baseline:</span> {locked}</div>'
                f'<div><span style="color:#94a3b8">CCS</span> <b style="color:{_col(ccs,0.4,0.85)}">{ccs:.2f}</b> {_trend_arrow(summ["ccs_trend"])}</div>'
                f'<div><span style="color:#94a3b8">RDI</span> <b style="color:{_col(rdi,0.25,0.75)}">{rdi:.2f}</b> {_trend_arrow(summ["rdi_trend"])}</div>'
                f'<div><span style="color:#94a3b8">GEI</span> <b style="color:{_col(gei,0.05,0.30)}">{gei:.2f}</b> {_trend_arrow(summ["gei_trend"])}</div>'
                f'<div><span style="color:#94a3b8">STR</span> <b style="color:{"#ef4444" if str_s>0.5 else "#f59e0b" if str_s>0.25 else "#22c55e"}">{str_s:.2f}</b></div>'
                f'{notes_html}</div>'
            )
            try:
                _obs_status_html.set_content(status_html)
            except Exception: pass

            # ── Genome panel ────────────────────────────────────────────────
            genome = obs.genome_vector()
            g_rows = ""
            for gname, gval in genome.items():
                pct = int(gval * 100)
                col = '#818cf8'
                g_rows += (
                    f'<div class="drive-bar-wrap">'
                    f'<span class="drive-label">{gname.replace("_"," ").title()}</span>'
                    f'<span class="drive-bar-bg" style="width:90px">'
                    f'<span class="drive-bar-fill" style="width:{max(2,pct)}%;background:{col}"></span></span>'
                    f'<span class="drive-value">{pct}%</span></div>'
                )
            try:
                _genome_html.set_content(f'<div style="font-size:12px">{g_rows}</div>')
            except Exception: pass

            # ── Identity Drift panel ────────────────────────────────────────
            if not summ['baseline_locked']:
                try:
                    _drift_html.set_content(
                    f'<div style="color:#64748b;font-size:12px">'
                    f'Building baseline… ({summ["ticks"]}/{obs.BASELINE_MIN_SAMPLES} samples)</div>'
                )
                except Exception: pass
            else:
                drift_pct = int(idx * 100)
                drift_col = '#ef4444' if idx > 0.35 else '#f59e0b' if idx > 0.20 else '#22c55e'
                drift_label = 'Stable' if idx < 0.15 else ('Moderate drift' if idx < 0.30 else '⚠️ High drift')
                history = obs.history_list(10)
                spark = "".join(
                    f'<span style="display:inline-block;width:8px;height:{max(3,int(s.idx*40))}px;'
                    f'background:{("#ef4444" if s.idx>0.35 else "#f59e0b" if s.idx>0.20 else "#6366f1")};'
                    f'margin-right:2px;vertical-align:bottom;border-radius:1px"></span>'
                    for s in history
                )
                try:
                    _drift_html.set_content(
                    f'<div style="font-size:12px;color:#cbd5e1">'
                    f'<div style="margin-bottom:6px"><b style="color:{drift_col}">{drift_label}</b> '
                    f'(IDX={idx:.3f})</div>'
                    f'<div style="margin-bottom:4px">'
                    f'<span class="drive-bar-bg" style="width:140px">'
                    f'<span class="drive-bar-fill" style="width:{max(2,drift_pct)}%;background:{drift_col}"></span></span>'
                    f'<span class="drive-value">{drift_pct}%</span></div>'
                    f'<div style="margin-top:6px;font-size:10px;color:#64748b">History:</div>'
                    f'<div style="display:flex;align-items:flex-end;height:44px;margin-top:2px">{spark}</div>'
                    f'</div>'
                )
                except Exception: pass

            # ── Emergence Radar panel ───────────────────────────────────────
            emg_pct  = int(emg * 100)
            str_pct  = int(str_s * 100)
            emg_col  = '#ef4444' if emg > 0.6 else '#f59e0b' if emg > 0.35 else '#22c55e'
            str_col  = '#ef4444' if str_s > 0.5 else '#f59e0b' if str_s > 0.25 else '#22c55e'
            emg_label= ('🌟 Emergent!' if emg > 0.6 else
                        '⚡ Rising'    if emg > 0.35 else
                        '✅ Baseline')
            history  = obs.history_list(10)
            emg_spark = "".join(
                f'<span style="display:inline-block;width:8px;height:{max(3,int(s.emergence*40))}px;'
                f'background:{("#ef4444" if s.emergence>0.6 else "#f59e0b" if s.emergence>0.35 else "#6366f1")};'
                f'margin-right:2px;vertical-align:bottom;border-radius:1px"></span>'
                for s in history
            )
            try:
                _emerge_html.set_content(
                f'<div style="font-size:12px;color:#cbd5e1">'
                f'<div style="margin-bottom:6px">{emg_label}</div>'
                f'<div><span style="color:#94a3b8">Emergence:</span> '
                f'<b style="color:{emg_col}">{emg:.3f}</b></div>'
                f'<div style="margin:3px 0">'
                f'<span class="drive-bar-bg" style="width:140px">'
                f'<span class="drive-bar-fill" style="width:{max(2,emg_pct)}%;background:{emg_col}"></span></span>'
                f'<span class="drive-value">{emg_pct}%</span></div>'
                f'<div><span style="color:#94a3b8">Strangeness:</span> '
                f'<b style="color:{"#ef4444" if str_s>0.5 else "#f59e0b" if str_s>0.25 else "#22c55e"}">{str_s:.3f}</b></div>'
                f'<div style="margin:3px 0">'
                f'<span class="drive-bar-bg" style="width:140px">'
                f'<span class="drive-bar-fill" style="width:{max(2,str_pct)}%;background:{str_col}"></span></span>'
                f'<span class="drive-value">{str_pct}%</span></div>'
                f'<div style="display:flex;align-items:flex-end;height:44px;margin-top:6px">{emg_spark}</div>'
                f'</div>'
            )
            except Exception: pass
        except Exception as _oe:
            pass

    def _render_all():
        # Drain thread-safe UI updates first — these come from background threads
        # (cognitive loop, async workers) that cannot touch NiceGUI elements directly.
        _drain_ui_queue()
        _render()
        _render_security()
        _render_observatory()

    ui.timer(2.0, _render_all)

    # Add link to main navigation
    # (also add it to the main page header nav)


# ─────────────────────────────────────────────
