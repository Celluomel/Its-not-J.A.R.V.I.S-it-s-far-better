"""
pages/lumina_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lumina inner-world.
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

# Fix: referenced in the EMOTIONAL STATE panel below (color = EMO_COLORS.get(...))
# but was never defined anywhere in this file — a guaranteed NameError every
# time that panel renders with at least one emotion above the 0.25 display
# threshold. Names match EmotionalSnapshot's 6 fields in emotional_state.py
# (curiosity, warmth, anxiety, satisfaction, frustration, enthusiasm).
EMO_COLORS = {
    "curiosity":    "#38bdf8",  # sky blue — alert, exploratory
    "warmth":       "#fb923c",  # warm orange — relational closeness
    "anxiety":      "#f87171",  # red — tension/threat signal
    "satisfaction": "#34d399",  # green — contentment/completion
    "frustration":  "#f59e0b",  # amber — friction, distinct from anxiety's red
    "enthusiasm":   "#a78bfa",  # violet — energised engagement
}

@ui.page('/lumina')
async def lumina_page():
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout, raised here to 10s for the same load-related
    # reason documented in cognitive_dashboard_page.py — an unguarded
    # TimeoutError here kills the page coroutine before anything renders.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(
            "[LuminaPage] Client connection timed out after 10s — "
            "system may be under heavy load."
        )
        ui.label('⚠ Connection timed out — please reload.').classes(
            'text-yellow-400 text-lg font-bold p-8'
        )
        return
    ui.add_css(GLOBAL_CSS)
    connection_monitor.start_monitoring()

    with ui.header().classes('app-header'):
        with ui.row().classes('items-center gap-3'):
            ui.button(
                icon='arrow_back', on_click=lambda: ui.navigate.to('/')
            ).props('flat dense round').style('color:#64748b')
            ui.html('''
                <div style="width:38px;height:38px;border-radius:10px;
                     background:linear-gradient(135deg,#8b5cf6,#c084fc);
                     display:flex;align-items:center;justify-content:center;
                     font-size:20px">✨</div>
            ''')
            ui.label('Lumina Mind').classes('app-title')
        with ui.row().classes('items-center gap-2 ml-auto'):
            ui.button(
                icon='settings', on_click=lambda: ui.navigate.to('/settings?tab=lumina')
            ).props('flat round dense').style('color:#64748b')

    with ui.column().classes('w-full max-w-4xl mx-auto p-6 gap-5 settings-page-content'):

        if not (state.persona and state.persona.is_ready):
            with ui.element('div').classes('info-card'):
                ui.html('<p class="text-yellow-400 text-lg">⚠️ Lumina is not yet active.</p>'
                        '<p class="text-slate-400 mt-2">Start a conversation on the main page to '
                        'wake her up. The cognitive dashboard will populate after the first exchange.</p>')
            ui.button('← Back to Chat', on_click=lambda: ui.navigate.to('/')).props('flat')
            return

        # Load status
        status = {}
        try:
            status = await asyncio.to_thread(state.persona.get_system_status)
        except Exception as e:
            ui.notify(f'Dashboard load error: {e}', type='negative')

        status.setdefault('emotional_state', {})
        status.setdefault('evolution', {})
        status.setdefault('identity', {})
        status.setdefault('self_concept', '')
        status.setdefault('relationships', {})
        status.setdefault('conditioning', {})
        status.setdefault('life_stage', 'unknown')

        # ── Header info row ──────────────────────────────────────────────
        with ui.row().classes('w-full gap-4'):
            with ui.element('div').classes('settings-card flex-1 p-4'):
                ui.label('LIFE STAGE').classes('section-label')
                stage = status.get('life_stage', 'unknown').replace('_', ' ').title()
                ui.label(stage).style('font-size:1.6rem;font-weight:700;color:#c4b5fd')

            with ui.element('div').classes('settings-card flex-1 p-4'):
                ui.label('INTERACTION COUNT').classes('section-label')
                cnt = status.get('interaction_count', 0)  # top-level key in get_system_status()
                ui.label(str(cnt)).style('font-size:1.6rem;font-weight:700;color:#818cf8')

            with ui.element('div').classes('settings-card flex-1 p-4'):
                ui.label('ACTIVE USER').classes('section-label')
                ui.label(user_manager.active_id).style('font-size:1.1rem;font-weight:600;color:#94a3b8')

        # ── Emotions ─────────────────────────────────────────────────────
        with ui.element('div').classes('settings-card w-full p-4'):
            ui.label('EMOTIONAL STATE').classes('section-label')
            _emo_raw  = status.get('emotional_state', {})
            # get_system_status() spreads numeric emotions AND string metadata into one dict.
            # Filter to only renderable float values; exclude computed/string fields.
            _emo_skip = {'description','trend','overall_valence','overall_arousal','state_description'}
            _emo_nums = {k: float(v) for k, v in _emo_raw.items()
                         if k not in _emo_skip and isinstance(v, (int, float))}
            _emo_desc = _emo_raw.get('description', '')
            if _emo_desc:
                ui.label(_emo_desc).classes('text-slate-400 text-sm italic mt-1 mb-3')
            if _emo_nums:
                with ui.grid(columns=3).classes('w-full gap-3 mt-3'):
                    for emo_name, val in _emo_nums.items():
                        pct   = int(val * 100)
                        color = EMO_COLORS.get(emo_name, '#9ca3af') if val > 0.25 else '#334155'
                        ui.html(
                            f'<div style="background:#0f172a;border-radius:8px;padding:12px">'
                            f'  <div style="display:flex;justify-content:space-between;margin-bottom:6px">'
                            f'    <span style="color:#94a3b8;font-size:0.8rem;text-transform:capitalize">{emo_name}</span>'
                            f'    <span style="color:{color};font-size:0.8rem;font-weight:600">{pct}%</span>'
                            f'  </div>'
                            f'  <div style="background:#1e293b;border-radius:3px;height:5px">'
                            f'    <div style="width:{pct}%;background:{color};height:5px;border-radius:3px;transition:width 0.4s"></div>'
                            f'  </div>'
                            f'</div>'
                        )
            else:
                ui.label('No emotional data yet.').classes('text-slate-500 text-sm mt-2')

        # ── Personality traits ───────────────────────────────────────────
        # traits are in status['personality'] (PersonalityState.to_dict()),
        # NOT in status['evolution'] which has cycles/records/avg_score
        traits = {k: v for k, v in status.get('personality', {}).items()
                  if isinstance(v, (int, float))}
        if traits:
            with ui.element('div').classes('settings-card w-full p-4'):
                ui.label('PERSONALITY TRAITS').classes('section-label')
                with ui.grid(columns=2).classes('w-full gap-2 mt-3'):
                    for t_name, t_val in sorted(traits.items(), key=lambda x: -float(x[1])):
                        pct = int(float(t_val) * 100)
                        color = '#22c55e' if pct >= 70 else ('#3b82f6' if pct >= 40 else '#f97316')
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:8px">'
                            f'  <span style="color:#94a3b8;font-size:0.78rem;min-width:120px;text-transform:capitalize">'
                            f'    {t_name.replace("_"," ")}</span>'
                            f'  <div style="flex:1;background:#1e293b;border-radius:3px;height:5px">'
                            f'    <div style="width:{pct}%;background:{color};height:5px;border-radius:3px"></div>'
                            f'  </div>'
                            f'  <span style="color:{color};font-size:0.78rem;min-width:32px;text-align:right">{pct}%</span>'
                            f'</div>'
                        )

        # ── Personality history chart ────────────────────────────────────
        try:
            import json as _json
            from pathlib import Path as _Path
            _hist_path = _Path('data/persona/personality_history.json')
            if _hist_path.exists():
                _hist = _json.loads(_hist_path.read_text())
                if len(_hist) >= 3:
                    # Sample up to 40 points for readability
                    _step = max(1, len(_hist) // 40)
                    _sampled = _hist[::_step][-40:]
                    _labels = [f"#{i*_step+1}" for i in range(len(_sampled))]
                    _chart_traits = ['confidence', 'curiosity', 'empathy_emotional', 'pragmatism']
                    _colors = ['#818cf8', '#34d399', '#f97316', '#c084fc']
                    _datasets = []
                    for trait, color in zip(_chart_traits, _colors):
                        _vals = [round(_s.get(trait, 0.5) * 100, 1) for _s in _sampled]
                        _datasets.append({
                            'label': trait.replace('_', ' ').title(),
                            'data': _vals,
                            'borderColor': color,
                            'backgroundColor': color + '22',
                            'borderWidth': 2,
                            'pointRadius': 2,
                            'tension': 0.3,
                            'fill': False,
                        })
                    with ui.element('div').classes('settings-card w-full p-4'):
                        ui.label('PERSONALITY EVOLUTION').classes('section-label')
                        ui.label(f'{len(_hist)} snapshots recorded across {status.get("evolution", {}).get("cycles", 0)} evolution cycles').classes('text-slate-500 text-xs mt-1 mb-3')
                        ui.chart({
                            'type': 'line',
                            'data': {'labels': _labels, 'datasets': _datasets},
                            'options': {
                                'responsive': True,
                                'animation': False,
                                'plugins': {
                                    'legend': {'labels': {'color': '#94a3b8', 'font': {'size': 11}}},
                                    'tooltip': {'mode': 'index'},
                                },
                                'scales': {
                                    'x': {'ticks': {'color': '#64748b', 'maxTicksLimit': 10},
                                          'grid': {'color': '#1e293b'}},
                                    'y': {'min': 0, 'max': 100,
                                          'ticks': {'color': '#64748b', 'callback': None},
                                          'grid': {'color': '#1e293b'},
                                          'title': {'display': True, 'text': 'Trait Value %', 'color': '#64748b'}},
                                },
                            },
                        }).classes('w-full').style('height:240px')
        except Exception as _ce:
            logger.debug(f"Personality chart error (non-fatal): {_ce}")

        # ── Self concept ─────────────────────────────────────────────────
        sc = status.get('self_concept', '')
        if sc:
            with ui.element('div').classes('settings-card w-full p-4'):
                ui.label('SELF CONCEPT').classes('section-label')
                ui.markdown(str(sc)).classes('text-slate-300 text-sm mt-2')

        # ── Relationships ────────────────────────────────────────────────
        rels = status.get('relationships', {})
        if rels:
            with ui.element('div').classes('settings-card w-full p-4'):
                ui.label('RELATIONSHIPS').classes('section-label')
                for uid, rel_info in list(rels.items())[:5]:
                    rap  = rel_info.get('rapport', 0) if isinstance(rel_info, dict) else 0
                    ints = rel_info.get('interactions', 0) if isinstance(rel_info, dict) else 0
                    ui.html(
                        f'<div class="status-row" style="margin-bottom:8px">'
                        f'  <span style="color:#e2e8f0;font-weight:500">{uid}</span>'
                        f'  <span style="color:#94a3b8;font-size:0.8rem">{ints} interactions · rapport {int(rap*100)}%</span>'
                        f'</div>'
                    )

        # ── Actions ──────────────────────────────────────────────────────
        with ui.element('div').classes('settings-card w-full p-4'):
            ui.label('COGNITIVE ACTIONS').classes('section-label')

            _last_response_for_feedback = getattr(state.persona, '_last_user_response', '')

            async def _refresh():
                ui.navigate.to('/lumina')

            async def _dream():
                ui.notify('Running dream cycle…', type='info', timeout=2000)
                result = await asyncio.to_thread(state.persona.trigger_dream)
                insights = result if isinstance(result, list) else []
                with ui.dialog() as _dr, ui.card().classes('p-5 min-w-96 max-w-xl'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.label('🌙 Dream Cycle Complete').classes('font-semibold text-lg text-purple-300')
                    if insights:
                        ui.label(f'{len(insights)} insight{"s" if len(insights) != 1 else ""} generated:').classes('text-slate-400 text-sm mb-2')
                        for ins in insights:
                            ui.label(f'• {ins}').classes('text-sm text-slate-200 mb-1')
                    else:
                        ui.label('Dream cycle ran — no new insights this time (Lumina may need more memories first).').classes('text-slate-400 text-sm')
                    ui.button('Close', on_click=_dr.close).classes('mt-4').props('color=purple')
                _dr.open()

            async def _learn():
                ui.notify('Running learning cycle…', type='info', timeout=2000)
                result = await asyncio.to_thread(state.persona.trigger_learning)
                with ui.dialog() as _lr, ui.card().classes('p-5 min-w-96 max-w-xl'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.label('📚 Learning Cycle Complete').classes('font-semibold text-lg text-indigo-300')
                    if result and isinstance(result, dict):
                        assessment = result.get('assessment', '')
                        insights   = result.get('insights', [])
                        scores = {k: v for k, v in result.items()
                                  if k.endswith('_score') and isinstance(v, (int, float))}
                        if assessment:
                            ui.label('Assessment').classes('text-slate-400 text-xs uppercase tracking-wide mb-1')
                            ui.label(assessment).classes('text-sm text-slate-200 mb-3')
                        if scores:
                            ui.label('Scores').classes('text-slate-400 text-xs uppercase tracking-wide mb-1')
                            with ui.grid(columns=3).classes('gap-2 mb-3'):
                                for k, v in scores.items():
                                    pct = int(float(v) * 100)
                                    ui.html(f'<div style="background:#1e293b;border-radius:6px;padding:8px;text-align:center">'
                                            f'<div style="color:#94a3b8;font-size:0.7rem">{k.replace("_score","").title()}</div>'
                                            f'<div style="color:#c4b5fd;font-weight:600">{pct}%</div></div>')
                        if insights:
                            ui.label('Insights').classes('text-slate-400 text-xs uppercase tracking-wide mb-1')
                            for ins in insights:
                                ui.label(f'• {ins}').classes('text-sm text-slate-200 mb-1')
                    else:
                        ui.label('Learning cycle ran successfully.').classes('text-slate-400 text-sm')
                    ui.button('Close', on_click=_lr.close).classes('mt-4').props('color=indigo')
                _lr.open()

            async def _feedback(positive: bool):
                last = getattr(state.persona, '_system', None)
                if last:
                    resp = getattr(last, '_last_responses', [''])
                    last_resp = resp[-1] if resp else ''
                    await asyncio.to_thread(
                        state.persona.receive_feedback, positive, user_manager.active_id, last_resp
                    )
                    lbl = 'Positive' if positive else 'Negative'
                    ui.notify(f'{lbl} feedback recorded', type='positive')

            async def _life_event():
                with ui.dialog() as _d, ui.card().classes('p-5 min-w-80'):
                    ui.label('🌱 Simulate Life Event').classes('font-semibold text-lg mb-3')
                    ui.label('Optionally hint the type of event (creative, social, challenge…)').classes('text-slate-400 text-sm mb-2')
                    _inp = ui.input(placeholder='Event type (optional)').classes('w-full')
                    with ui.row().classes('gap-2 mt-4'):
                        ui.button('Cancel', on_click=_d.close).props('flat')
                        async def _go():
                            _d.close()
                            ui.notify('Simulating life event…', type='info', timeout=2000)
                            scenario, reaction = await asyncio.to_thread(
                                state.persona.trigger_life_event, _inp.value or None
                            )
                            if scenario:
                                with ui.dialog() as _d2, ui.card().classes('p-5 max-w-lg'):
                                    ui.label('🌱 Life Event').classes('font-semibold text-lg text-green-400 mb-3')
                                    ui.label('What happened:').classes('text-slate-400 text-xs uppercase tracking-wide mb-1')
                                    ui.label(scenario).classes('text-sm text-slate-200 mb-4')
                                    ui.label("Lumina's Reflection:").classes('text-slate-400 text-xs uppercase tracking-wide mb-1')
                                    ui.label(reaction).classes('text-sm text-slate-200')
                                    ui.button('Close', on_click=_d2.close).classes('mt-5').props('color=green')
                                _d2.open()
                                # Refresh dashboard ONLY after user closes the result dialog
                            else:
                                ui.notify('Life event could not be generated — Lumina may need more context first.', type='warning', timeout=6000)
                        ui.button('Simulate', on_click=_go).props('color=purple')
                _d.open()

            with ui.row().classes('gap-3 flex-wrap mt-3'):
                ui.button('🌙 Dream Cycle', on_click=_dream).props('color=deep-purple')
                ui.button('📚 Learning Cycle', on_click=_learn).props('color=indigo')
                ui.button('🌱 Life Event', on_click=_life_event).props('color=green')
                ui.button('👍 Rate last response', on_click=lambda: asyncio.create_task(_feedback(True))).props('color=green flat').tooltip('Send positive feedback on the most recent chat response')
                ui.button('👎 Rate last response', on_click=lambda: asyncio.create_task(_feedback(False))).props('color=red flat').tooltip('Send negative feedback on the most recent chat response')
                ui.button('🔄 Refresh', on_click=_refresh).props('flat color=slate')




# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator Dashboard  /orchestrator
# ─────────────────────────────────────────────────────────────────────────────


