"""
pages/research_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Research Cortex.
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

@ui.page('/research')
async def research_page():
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout. On a loaded system this can be exceeded, killing
    # the page coroutine before anything renders — confirmed in production
    # logs (TimeoutError at cognitive_dashboard_page.py:398, same pattern).
    # Raised to 10s and wrapped so a genuine failure shows a visible
    # message instead of a silently blank page.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(f"[research_page] Client connection timed out after 10s")
        ui.label('⚠ Connection timed out — please reload.').classes(
            'text-yellow-400 text-lg font-bold p-8'
        )
        return
    ui.add_css(GLOBAL_CSS)
    connection_monitor.start_monitoring()

    with ui.header().classes('app-header'):
        with ui.row().classes('items-center gap-3'):
            ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat dense round').style('color:#64748b')
            ui.html(
                '<div style="width:38px;height:38px;border-radius:10px;'
                + 'background:linear-gradient(135deg,#0ea5e9,#6366f1);'
                + 'display:flex;align-items:center;justify-content:center;font-size:20px">&#x1F52C;</div>'
            )
            ui.label('Research Cortex').classes('app-title')
            ui.space()
            ui.button('Refresh', icon='refresh', on_click=lambda: ui.navigate.to('/research')).props('flat dense').style('color:#64748b')

    with ui.column().classes('w-full max-w-3xl mx-auto p-6 gap-5 settings-page-content'):

        with ui.element('div').classes('settings-card w-full p-5'):
            ui.label('MANUAL RESEARCH TRIGGER').classes('section-label')
            ui.label('Activate any research mode on a specific goal.').classes('text-slate-400 text-sm mb-4')

            with ui.column().classes('w-full gap-4 mt-3'):
                goal_input = ui.input(
                    'Research Goal',
                    placeholder='e.g. Latest breakthroughs in quantum error correction'
                ).props('outlined dense').classes('w-full dark-input')

                with ui.row().classes('w-full gap-4 items-end'):
                    mode_select = ui.select(
                        {1: 'Mode 1 - On-Demand', 2: 'Mode 2 - Extended Analysis', 3: 'Mode 3 - Background (stores to memory)'},
                        value=1, label='Research Mode'
                    ).props('outlined dense').classes('flex-1 dark-input')
                    depth_select = ui.select(
                        {1: 'Depth 1', 2: 'Depth 2', 3: 'Depth 3', 4: 'Depth 4', 5: 'Depth 5'},
                        value=3, label='Depth'
                    ).props('outlined dense').classes('dark-input')

                with ui.element('div').classes('info-card').style('padding:10px 14px;border-color:#1e3a5f;background:#080f1a'):
                    ui.markdown(
                        '**Mode 1** - Single bounded loop, no memory write. Best for quick facts.\n\n'
                        '**Mode 2** - Multi-step, cross-source, richer synthesis. Best for analysis.\n\n'
                        '**Mode 3** - Autonomous loop, LLM-summarised, stored in journal & memory. '
                        'Influences future answers. Best for background enrichment.'
                    ).style('font-size:0.78rem;color:#475569;line-height:1.6')

                result_area = ui.element('div').classes('w-full')

                async def _run_research():
                    goal = goal_input.value.strip()
                    if not goal:
                        ui.notify('Please enter a research goal', type='warning')
                        return
                    if not state.persona or not state.persona.research:
                        ui.notify('Research system not yet initialised - start a conversation first', type='warning')
                        return

                    mode  = int(mode_select.value)
                    depth = int(depth_select.value)
                    ui.notify(f'Running Mode {mode} research...', type='info', timeout=3000)

                    result = await asyncio.to_thread(
                        state.persona.trigger_research, goal, mode, depth
                    )

                    try:
                        result_area.clear()
                    except Exception:
                        logger.debug("Research page closed before results — skipping UI update")
                        return
                    with result_area:
                        if result and result.summary:
                            with ui.element('div').classes('settings-card w-full p-4 mt-4'):
                                conf_pct = int(result.confidence * 100)
                                stat_col = '#22c55e' if result.status == 'complete' else '#f97316'
                                with ui.row().classes('items-center gap-3 mb-3'):
                                    ui.html(f'<span style="color:{stat_col};font-weight:600;font-size:0.85rem">{result.status.upper()}</span>')
                                    ui.html(f'<span style="color:#94a3b8;font-size:0.8rem">confidence {conf_pct}%</span>')
                                    ui.html(f'<span style="color:#94a3b8;font-size:0.8rem">{len(result.sources)} sources</span>')
                                ui.label('RESULT').classes('section-label')
                                ui.markdown(result.summary).classes('text-slate-200 text-sm mt-2')
                                if result.sources:
                                    ui.label('SOURCES').classes('section-label mt-4')
                                    for src in result.sources:
                                        src_title = (src.get('title') or src['url'])[:80]
                                        cred_pct = int(src.get('credibility', 0.5) * 100)
                                        ui.html(
                                            f'<div class="status-row" style="margin-bottom:4px">'
                                            f'<a href="{src["url"]}" target="_blank" style="color:#818cf8;font-size:0.78rem">'
                                            f'{src_title}</a>'
                                            f'<span style="color:#475569;font-size:0.75rem">cred {cred_pct}%</span></div>'
                                        )
                                if result.knowledge_nodes:
                                    ui.html(f'<div style="color:#22c55e;font-size:0.8rem;margin-top:8px">+ {result.knowledge_nodes} knowledge nodes stored</div>')
                        else:
                            ui.label('No result returned — check console logs.').classes('text-red-400 text-sm mt-4')

                ui.button('Start Research', icon='science', on_click=_run_research).props('color=indigo').classes('w-full')

        if state.persona and state.persona.research:
            rac_data = state.persona.research.rac.daily_summary()
            with ui.element('div').classes('settings-card w-full p-5'):
                ui.label('RESEARCH ACTIVATION METRICS (TODAY)').classes('section-label')
                rows = [
                    ('Complexity score', f"{int(rac_data.get('complexity_score', 0) * 100)}%"),
                    ('Total queries', str(rac_data.get('total_queries', 0))),
                    ('Complex queries', str(rac_data.get('complex_queries', 0))),
                    ('Background runs today', f"{rac_data.get('research_today', 0)} / 2 max"),
                    ('Mode 3 jobs queued', str(rac_data.get('pending_mode3', 0))),
                ]
                for label, val in rows:
                    ui.html(
                        f'<div class="status-row" style="margin-bottom:5px">'
                        f'<span style="color:#94a3b8;font-size:0.82rem">{label}</span>'
                        f'<span style="color:#c4b5fd;font-size:0.82rem;font-weight:600">{val}</span></div>'
                    )
                top_topics = rac_data.get('top_topics', [])
                if top_topics:
                    ui.label('TOP TOPICS TODAY').classes('section-label mt-3')
                    for topic, count in top_topics:
                        ui.html(
                            f'<div class="status-row"><span style="color:#94a3b8;font-size:0.8rem">'
                            f'{topic.replace("_"," ")}</span>'
                            f'<span style="color:#818cf8;font-size:0.8rem">{count}x</span></div>'
                        )

        with ui.element('div').classes('settings-card w-full p-5'):
            ui.label('RESEARCH JOURNAL').classes('section-label')
            ui.label('Mode 3 background research — summarised and stored in long-term memory.').classes('text-slate-400 text-sm mb-3')

            if not state.persona or not state.persona.research:
                ui.label('Research system not initialised yet.').classes('text-slate-500 text-sm')
            else:
                entries = state.persona.research.journal(limit=15)
                if not entries:
                    with ui.element('div').classes('info-card'):
                        ui.html(
                            '<span style="color:#475569;font-size:0.85rem">'
                            'No journal entries yet. Run a Mode 3 research or wait for background activation.</span>'
                        )
                else:
                    for entry in entries:
                        conf = int(entry.get('confidence_score', 0) * 100)
                        conf_col = '#22c55e' if conf >= 70 else ('#f97316' if conf >= 45 else '#ef4444')
                        with ui.element('div').classes('info-card mb-3'):
                            with ui.row().classes('items-center gap-2 mb-2 flex-wrap'):
                                ui.html(f'<span style="color:{conf_col};font-size:0.75rem;font-weight:700">conf {conf}%</span>')
                                ui.html(f'<span style="color:#475569;font-size:0.72rem">{entry.get("finished_at", "")[:16]}</span>')
                                trigger = entry.get('trigger_reason', '')
                                if trigger:
                                    ui.html(f'<span style="color:#818cf8;font-size:0.72rem">trigger: {trigger}</span>')
                            ui.html(f'<div style="color:#c4b5fd;font-size:0.85rem;font-weight:600;margin-bottom:6px">{entry.get("goal","")}</div>')
                            summary = entry.get('summary', '')
                            if summary:
                                ui.markdown((summary[:600] + '...' if len(summary) > 600 else summary)).classes('text-slate-300 text-sm')
                            topics = entry.get('influences_topics', [])
                            if topics:
                                ui.html(f'<div style="color:#475569;font-size:0.72rem;margin-top:6px">topics: {", ".join(topics)}</div>')
                            nodes = entry.get('knowledge_nodes', 0)
                            if nodes:
                                ui.html(f'<div style="color:#22c55e;font-size:0.72rem;margin-top:4px">{nodes} memory nodes stored</div>')



