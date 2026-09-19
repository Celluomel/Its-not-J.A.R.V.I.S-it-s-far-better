"""
pages/main_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main chat interface.
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

@ui.page('/')
async def main_page():
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout. On a loaded system this can be exceeded, killing
    # the page coroutine before anything renders — confirmed in production
    # logs (TimeoutError at cognitive_dashboard_page.py:398, same pattern).
    # Raised to 10s and wrapped so a genuine failure shows a visible
    # message instead of a silently blank page.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(f"[main_page] Client connection timed out after 10s")
        ui.label('⚠ Connection timed out — please reload.').classes(
            'text-yellow-400 text-lg font-bold p-8'
        )
        return
    ui.add_css(GLOBAL_CSS)
    
    # Start connection monitoring
    connection_monitor.start_monitoring()
    
    # Add connection status indicator
    connection_indicator = ui.html('''
        <div class="connection-status connected">
            ● Connected
        </div>
    ''')

    # ── Brain readiness check ─────────────────────────────────────────────
    # Normally state.initialize() runs at server startup via app.on_startup()
    # in app.py — by the time a browser connects the brain is already ready.
    # This block is a safety net for the rare case where someone connects
    # within the first few seconds before startup completes.
    if not state.ready:
        if state.initializing:
            # Already booting — poll until ready then reload
            with ui.column().classes('w-full h-screen items-center justify-center gap-4'):
                ui.spinner(size='lg', color='indigo')
                ui.label('Starting up…').classes('text-slate-400 text-sm')
                ui.label('Almost ready').classes('text-slate-600 text-xs')
            # Poll every 500ms, reload when ready
            async def _wait_and_reload():
                for _ in range(120):   # wait up to 60 seconds
                    await asyncio.sleep(0.5)
                    if state.ready:
                        ui.navigate.to('/')
                        return
            asyncio.create_task(_wait_and_reload())
            return
        else:
            # Not started at all — shouldn't happen in normal operation
            # but handle gracefully
            await state.initialize()
            ui.navigate.to('/')
            return

    # ════════════════════════════════════════════════════════════════
    #  UI TIMER FUNCTIONS - Pull-based architecture
    # ════════════════════════════════════════════════════════════════

    # ── Guard against concurrent handle_send / proactive calls ──────────
    # _is_processing  : True while handle_send OR _speak is running
    # _is_proactive_running : True while maybe_proactive LLM call is in-flight
    # Together they prevent all three forms of double-fire:
    #   1. Timer re-entrancy (handle_send takes seconds, timer fires every 100ms)
    #   2. STT double-fire (VAD emits on_transcription twice per utterance)
    #   3. Proactive spam (cooldown set before generation, plus flag guard)
    _is_processing: bool = False
    _is_speaking:   bool = False   # True only during TTS playback
    _is_proactive_running: bool = False
    _last_transcription: str = ""
    _last_transcription_time: float = 0.0
    _DEDUP_WINDOW: float = 3.0  # seconds — identical text within this window is dropped

    _processing_started_at: float = 0.0  # watchdog for stuck _is_processing

    async def check_transcription_queue():
        """Check for new transcriptions - called by ui.timer every 100ms"""
        nonlocal _is_processing, _is_speaking, _is_proactive_running
        nonlocal _last_transcription, _last_transcription_time
        nonlocal _processing_started_at
        import time as _time

        # Watchdog: if LLM processing stuck for >90s, force reset
        if _is_processing:
            elapsed = _time.monotonic() - _processing_started_at
            if elapsed > 90:
                logger.warning(f"_is_processing stuck for {elapsed:.0f}s — forcing reset")
                _is_processing = False
                _is_speaking   = False
                if state.conv_audio:
                    state.conv_audio._tts_active = False
                return

        # Skip entirely if LLM inference is already in-flight
        if _is_processing or _is_proactive_running:
            return

        try:
            while not state.transcription_queue.empty():
                text = state.transcription_queue.get_nowait()
                if not text or not text.strip():
                    continue

                now = _time.monotonic()

                # Drop identical transcription if it arrived within the dedup window
                if (
                    text.strip().lower() == _last_transcription
                    and (now - _last_transcription_time) < _DEDUP_WINDOW
                ):
                    logger.info(f"🔇 Duplicate transcription suppressed: '{text[:60]}'")
                    continue

                _last_transcription = text.strip().lower()
                _last_transcription_time = now

                logger.info(f"📝 Transcription from queue: {text}")

                # If the bot is currently speaking, show the transcription as a
                # chat bubble immediately so the user can see their words, then
                # interrupt the TTS and send to LLM as normal.
                if _is_speaking:
                    state.tts_stop_event.set()          # interrupt current TTS
                    logger.info("🗣 User spoke during TTS — interrupting")

                _is_processing = True
                _processing_started_at = _time.monotonic()
                try:
                    await handle_send(text)
                finally:
                    _is_processing = False

        except Exception as e:
            _is_processing = False
            logger.debug(f"Transcription queue check error: {e}")

        # ── Idle / proactive — PandoraBOX-aware ────────────────────────────
        # Skip if anything else is running
        if _is_processing or _is_proactive_running:
            return

        # ── GoalActionExecutor user question ────────────────────────────
        # Check if the action executor has queued a goal-driven question
        # to ask the user. Deliver it before the normal proactive check.
        try:
            _org_ref = getattr(getattr(state, 'persona', None), '_organism', None)
            # BUG FIX: organism stores the loop as '_loop', not '_internal_loop'
            # '_internal_loop' was always None → GAE pickup silently dead since written
            _il_ref  = getattr(_org_ref, '_loop', None) or getattr(_org_ref, '_internal_loop', None)
            _gae_ref = getattr(_il_ref, '_goal_action_executor', None)
            if _gae_ref:
                _pending_q = _gae_ref.get_pending_user_question()
                if _pending_q and not _is_proactive_running:
                    _is_proactive_running = True
                    _bubble(_pending_q, 'bot')
                    logger.info(f"🎯 Goal question delivered: {_pending_q!r}")
                    if voice_select.value != 'off' and state.audio:
                        set_core_state('speaking')
                        await _speak(_pending_q)
                    _is_proactive_running = False

                # ── GoalActionExecutor insight delivery ─────────────────────
                # Insights from autonomous web_search/self_question actions
                # were queued (self._pending_insights) but had no consumer
                # anywhere — they accumulated in a maxlen=5 deque and were
                # silently evicted as new ones arrived, never reaching chat.
                _pending_insight = _gae_ref.get_pending_insight()
                if _pending_insight and not _is_proactive_running:
                    _is_proactive_running = True
                    _bubble(_pending_insight, 'bot')
                    logger.info(f"💡 Goal insight delivered: {_pending_insight[:60]!r}")
                    if voice_select.value != 'off' and state.audio:
                        set_core_state('speaking')
                        await _speak(_pending_insight)
                    _is_proactive_running = False
        except Exception as _gae_ex:
            logger.debug(f"GAE pickup error: {_gae_ex}")

        try:
            _is_proactive_running = True
            if _agent_state.is_idle_trigger_ready() and state.persona and state.persona.is_ready:
                # ── Anti-loop cooldown: matches IDLE_BEFORE_PROACTIVE so proactive
                #    can fire at most once per idle window (not 4× in 30 min) ──
                _agent_state.set_idle_cooldown(seconds=900)   # 15 min between proactives

                # ── Mark used thought so the same thought is not reused ──
                try:
                    pb  = getattr(state, 'persona', None)
                    org = getattr(pb, '_organism', None) if pb else None
                    if org and hasattr(org, 'thought_stream'):
                        recent = org.thought_stream.recent(3)
                        for t in recent:
                            if t.priority > 0.45 and t.thought_type not in ('meta', 'self_model'):
                                t.priority = 0.20   # demote so same thought won't be picked again
                                break
                except Exception:
                    pass

                _proactive_prompt = _build_proactive_prompt(state)
                proactive_msg = None

                # Inject working memory so PandoraBOX's proactive message can
                # reference what she is currently observing without an
                # explicit on-demand vision capture.
                _proactive_vision = None
                if state.vision and state.vision.camera_active:
                    _wm = state.vision.get_visual_context_for_prompt()
                    if _wm:
                        _proactive_vision = _wm

                async for chunk in state.persona.get_response_stream(
                    _proactive_prompt,
                    user_id=user_manager.active_id,
                    vision_context=_proactive_vision,
                ):
                    if isinstance(chunk, dict) and chunk.get("__meta__"):
                        proactive_msg = chunk.get("speech", "")
                        break

                if proactive_msg:
                    # ── Semantic deduplication: skip if too similar to last proactive ──
                    _last_pro = getattr(main_page, '_last_proactive_msg', "")
                    _words_new = set(proactive_msg.lower().split())
                    _words_old = set(_last_pro.lower().split())
                    _overlap = len(_words_new & _words_old) / max(1, len(_words_new | _words_old))
                    if _overlap > 0.55 and _last_pro:
                        logger.debug(f"Proactive suppressed (similarity {_overlap:.2f}): {proactive_msg!r}")
                    else:
                        main_page._last_proactive_msg = proactive_msg
                        logger.info(f"🗣️ Proactive (PandoraBOX): {proactive_msg!r}")
                        _bubble(proactive_msg, 'bot')
                        if voice_select.value != 'off' and state.audio:
                            set_core_state('speaking')
                            main_page._current_state = 'speaking'
                            await _speak(proactive_msg)
        except Exception as e:
            logger.debug(f"Proactive check error: {e}")
        finally:
            _is_proactive_running = False

    # ── Voice clone status pill ───────────────────────────────────────
    def _tts_pill_label() -> str:
        if state.audio and state.audio.tts_engine:
            t  = state.audio.tts_engine['type'].upper()
            vr = state.audio.tts_engine.get('voice_ref') if state.audio.tts_engine['type'] == 'coqui' else None
            if vr and os.path.exists(vr):
                return f'🎙️ {t} · Cloning'
            return f'🎙️ {t}'
        return f'🎙️ {config.TTS_PROVIDER.upper()}'

    def _tts_pill_class() -> str:
        if state.audio and state.audio.tts_engine:
            vr = state.audio.tts_engine.get('voice_ref') if state.audio.tts_engine['type'] == 'coqui' else None
            if vr and os.path.exists(vr):
                return 'status-pill pill-clone-on'
        return 'status-pill pill-tts'

    # ── Vision status pill ────────────────────────────────────────────
    def _vision_pill_label() -> str:
        if state.vision and state.vision.camera_active:
            faces = len(state.vision.face_encodings) if hasattr(state.vision, 'face_encodings') else 0
            return f'👁️ Camera · {faces} faces'
        return '👁️ Camera off'

    def _vision_pill_class() -> str:
        if state.vision and state.vision.camera_active:
            return 'status-pill pill-vision'
        return 'status-pill pill-memory'

    # ══════════════════════════════════════════════════════════════
    #  BUILD ORGANISM UI - Living Interface
    # ══════════════════════════════════════════════════════════════
    
    # Notice we unpack 4 variables now to acquire the input container
    core_elem, camera_elem, chat_container, input_container, status_bar = build_organism(show_header=True)
    
    # Store references for state updates
    main_page.camera_elem = camera_elem
    main_page.core_elem = core_elem
    
    # Enhanced camera feed update with state management
    def update_camera_feed():
        """Pull latest camera frame and update organism eye"""
        if state.vision and state.vision.camera_active:
            try:
                encoded_frame = state.vision.get_latest_encoded_frame()
                if encoded_frame:
                    set_camera_feed(encoded_frame)
                    # Update organism state based on CURRENTLY DETECTED faces,
                    # not stored encodings (which are non-empty after any registration).
                    if (hasattr(state.vision, 'last_detected_faces')
                            and state.vision.last_detected_faces):
                        set_core_state('face')
                    else:
                        if hasattr(main_page, '_current_state') and main_page._current_state == 'face':
                            set_core_state('')
                            main_page._current_state = ''
                    # Auto-switch active user from face recognition
                    if hasattr(state.vision, 'last_detected_faces') and state.vision.last_detected_faces:
                        match = user_manager.resolve_faces(state.vision.last_detected_faces)
                        if user_manager.set_active_from_face(match):
                            refresh_chip(main_page._chip_refs, user_manager)
                            ui.notify(f'👤 Recognised: {match.display_name}',
                                      type='info', position='bottom-right', timeout=3000)
            except Exception as e:
                logger.error(f"Camera feed error: {e}")
    
    # ── Fixed status bar — lives in the floating-core-bar (never scrolls away) ──
    with status_bar:
        # Left: persona name + emotional state
        with ui.row().classes('items-center gap-3 flex-shrink-0'):
            ui.label(f'✨ {_gpn()}').classes('text-lg font-bold glow-text')
            # Bug fix: this label was built once at page render and never
            # updated again — no ui.refreshable, no timer, not even assigned
            # to a variable, so nothing could ever call .set_text() on it.
            # It showed whatever PandoraBOX's dominant emotion was the instant
            # the page loaded, frozen until a manual browser reload. Now
            # kept live via the ui.timer registered near _cag_sync below,
            # same pattern already used for that periodic sync.
            emotion_label = ui.label('').classes('text-xs opacity-60 italic max-w-xs truncate')
            emotion_label.set_visibility(False)

            def _sync_emotion_label():
                try:
                    if state.persona and state.persona.is_ready:
                        _ctx = state.persona.get_prompt_context(user_manager.active_id)
                        _txt = _ctx.get('emotional_state', '')
                        emotion_label.set_text(_txt)
                        emotion_label.set_visibility(bool(_txt))
                except Exception:
                    pass

            _sync_emotion_label()  # populate immediately, don't wait for first tick

        # Right: pills + user chip + action buttons
        with ui.row().classes('items-center gap-2 flex-wrap'):
            if state.persona and state.persona.is_ready:
                try:
                    _stage = state.persona.get_prompt_context(user_manager.active_id).get('life_stage', '')
                    if _stage:
                        ui.html(f'<span class="status-pill" style="background:rgba(139,92,246,0.3);color:#c4b5fd">✨ {_stage}</span>')
                except Exception:
                    pass
            ui.html(f'<span class="status-pill pill-active">💾 {config.MEMORY_BACKEND}</span>')
            if state.audio:
                ui.html(f'<span class="status-pill pill-active">🎙️ {config.TTS_PROVIDER}</span>')
            if state.vision and state.vision.camera_active:
                _faces = len(state.vision.face_encodings) if hasattr(state.vision, 'face_encodings') else 0
                ui.html(f'<span class="status-pill pill-active">👁️ {_faces} faces</span>')

            _chip_elem, main_page._chip_refs = build_user_chip(user_manager, vision=state.vision)

        # Nav buttons — props color= overrides Quasar's default light-blue
        ui.button(icon='hub',
                  on_click=lambda: ui.navigate.to('/orchestrator')
                  ).props('flat round dense color=grey-6').tooltip('Orchestrator')
        ui.button(icon='monitor_heart',
                  on_click=lambda: ui.navigate.to('/cognitive-dashboard')
                  ).props('flat round dense color=grey-6').tooltip('Cognitive Health')
        ui.button(icon='rss_feed',
                  on_click=lambda: ui.navigate.to('/rss-feeds')
                  ).props('flat round dense color=grey-6').tooltip('RSS Feed Management')
        ui.button(icon='settings',
                  on_click=lambda: ui.navigate.to('/settings')
                  ).props('flat round dense color=grey-6').tooltip('Settings')

        # Master mode toggle — activates / deactivates PandoraBOX Network
        _net_enabled = getattr(config, 'LUMINA_NETWORK_ENABLED', False)
        _master_btn = ui.button(
            icon='lan',
        ).props(
            f'flat round dense color={"purple" if _net_enabled else "grey-6"}'
        ).tooltip('Master Mode: ON' if _net_enabled else 'Master Mode: OFF')

        async def _toggle_master_mode():
            _current = getattr(config, 'LUMINA_NETWORK_ENABLED', False)
            _new_val  = not _current
            # Update config object
            config.__dict__['LUMINA_NETWORK_ENABLED'] = _new_val
            # Persist to config.json
            try:
                save_settings(config)
            except Exception as _e:
                ui.notify(f'Save failed: {_e}', type='negative', position='top')
                return
            # Update button appearance
            if _new_val:
                _master_btn.props('flat round dense color=purple')
                _master_btn.tooltip('Master Mode: ON')
                ui.notify('🌐 Master Mode activated — restart to connect children', type='positive', position='top')
                # Re-init network if not already running
                if not state.lumina_network:
                    try:
                        from cognition.lumina_network import LuminaNetwork
                        state.lumina_network = LuminaNetwork(state.persona, state.llm)
                        children_cfg = getattr(config, 'LUMINA_CHILDREN', [])
                        if children_cfg:
                            state.lumina_network.load_from_config(children_cfg)
                        master_name = getattr(config, 'LUMINA_MASTER_NAME', 'PandoraBOX-Master')
                        master_url  = f"http://127.0.0.1:{getattr(config, 'NICEGUI_PORT', 8080)}"
                        state.lumina_network.set_master_info(master_url, master_name)
                        state.lumina_network.set_bubble_fn(_network_bubble)
                        ui.notify(f'🌐 Network connector started — {len(children_cfg)} child(ren)', type='positive', position='top')
                    except Exception as _e:
                        ui.notify(f'Network init error: {_e}', type='warning', position='top')
            else:
                _master_btn.props('flat round dense color=grey-6')
                _master_btn.tooltip('Master Mode: OFF')
                ui.notify('🌐 Master Mode deactivated', type='info', position='top')
                # Gracefully stop network
                if state.lumina_network:
                    try:
                        await state.lumina_network.stop() if hasattr(state.lumina_network, 'stop') else None
                    except Exception:
                        pass
                    state.lumina_network = None

        _master_btn.on('click', _toggle_master_mode)

        async def _new_conversation():
            from managers.session_manager import get_session_manager
            sess = get_session_manager()
            with ui.dialog() as _nd, ui.card().classes('p-5'):
                ui.label('Start a New Conversation?').classes('font-semibold text-lg mb-2')
                ui.label('This will clear the current conversation history. '
                         'Long-term memory and the research journal are kept.').classes('text-slate-400 text-sm mb-4')
                with ui.row().classes('gap-3'):
                    def _confirm_new():
                        sess.clear(state.llm)
                        chat_area.clear()
                        try:
                            app.storage.tab['chat_log'] = []
                        except Exception:
                            pass
                        _bubble(f"New conversation started. How can I help?", 'bot')
                        _nd.close()
                        ui.notify('Conversation cleared', type='info', position='bottom-right')
                    ui.button('Clear & Start Fresh', on_click=_confirm_new).props('color=red')
                    ui.button('Cancel', on_click=_nd.close).props('flat')
            _nd.open()

        ui.button(icon='add_comment',
                  on_click=lambda: asyncio.create_task(_new_conversation())
                  ).props('flat round dense color=grey-6').tooltip('New Conversation')

        # Web Search: 3-level cycle  off → auto → always → off
        _WS_CYCLE = ['off', 'auto', 'always']
        _ws_state = [getattr(config, 'WEB_SEARCH_MODE', 'off')]
        # backward-compat: old bool field
        if isinstance(_ws_state[0], bool):
            _ws_state[0] = 'always' if _ws_state[0] else 'off'
        def _ws_cls(mode):
            return {'off': 'toggle-btn-off',
                    'auto': 'toggle-btn-amber',
                    'always': 'toggle-btn-red'}[mode]
        def _ws_tip(mode):
            return {'off':    '🔍 Web Search: OFF — click for Auto',
                    'auto':   '🔍 Web Search: AUTO (LLM decides) — click for Always',
                    'always': '🔍 Web Search: ALWAYS — click to disable'}[mode]
        web_search_btn = ui.button(icon='travel_explore').props('flat round dense')
        web_search_btn.classes(_ws_cls(_ws_state[0]))
        web_search_btn.tooltip(_ws_tip(_ws_state[0]))
        def _toggle_web_search():
            idx = _WS_CYCLE.index(_ws_state[0])
            _ws_state[0] = _WS_CYCLE[(idx + 1) % 3]
            config.WEB_SEARCH_MODE = _ws_state[0]
            save_settings(config, silent=True)
            web_search_btn.classes(remove='toggle-btn-off toggle-btn-amber toggle-btn-red')
            web_search_btn.classes(add=_ws_cls(_ws_state[0]))
            web_search_btn.tooltip(_ws_tip(_ws_state[0]))
            labels = {'off': '🔍 OFF', 'auto': '🔍 AUTO — LLM decides when to search', 'always': '🔍 ALWAYS searching'}
            ui.notify(labels[_ws_state[0]], type='info', position='bottom-right', timeout=2500)
        web_search_btn.on('click', _toggle_web_search)

        # Verbosity toggle
        _verb_active = [getattr(config, 'RESPONSE_VERBOSITY', 'concise') == 'verbose']
        def _verb_cls(on): return 'toggle-btn-amber' if on else 'toggle-btn-off'
        verb_btn = ui.button(icon='unfold_more').props('flat round dense')
        verb_btn.classes(_verb_cls(_verb_active[0]))
        verb_btn.tooltip('Verbose ON' if _verb_active[0] else 'Concise mode')
        def _toggle_verbosity():
            _verb_active[0] = not _verb_active[0]
            config.RESPONSE_VERBOSITY = 'verbose' if _verb_active[0] else 'concise'
            from managers.settings_manager import save_settings
            save_settings(config, silent=True)
            verb_btn.classes(remove='toggle-btn-amber toggle-btn-off')
            verb_btn.classes(add=_verb_cls(_verb_active[0]))
            verb_btn.tooltip('Verbose ON' if _verb_active[0] else 'Concise mode')
            ui.notify('📝 Verbose ON' if _verb_active[0] else '📝 Concise ON',
                      type='positive' if _verb_active[0] else 'info',
                      position='bottom-right', timeout=2000)
        verb_btn.on('click', _toggle_verbosity)

    # Build chat interface inside the chat container
    with chat_container:
        chat_area = ui.column().classes('w-full gap-3 pb-8')

    # Floating scroll-to-bottom button
    scroll_btn = ui.button(
        icon='keyboard_arrow_down', on_click=lambda: _scroll_bottom()
    ).props('round dense color=indigo-8').classes('scroll-to-bottom-btn').style(
        'position:fixed;bottom:90px;right:24px;z-index:999;'
        'display:none;box-shadow:0 2px 12px rgba(99,102,241,.5);'
    )

    def _check_scroll_position():
        try:
            ui.context.client.run_javascript('(function(){var c=document.querySelector("#lumina-chat")||document.querySelector(".scroll-area__content");if(!c)return;var atBottom=(c.scrollHeight-c.scrollTop-c.clientHeight)<60;var btn=document.querySelector(".scroll-to-bottom-btn");if(btn)btn.style.display=atBottom?"none":"flex";})()')
        except Exception:
            pass


    with input_container:
        with ui.row().classes('w-full items-center gap-2 chat-input-container'):

            voice_select = ui.select(
                {'off': '⌨️ Text', 'always': '🔴 Live'},
                value='off'
            ).props('dense outlined borderless').classes('w-28 mode-select')

            # ── VAD on/off toggle — visible only when Live mode is active ──
            _vad_active = [False]   # mutable ref so closures can update it

            def _vad_btn_style(on: bool) -> str:
                return 'color:#ef4444;background:#3f0a0a' if on else 'color:#475569'

            vad_btn = ui.button(icon='mic').props('flat round dense').style(_vad_btn_style(False))
            vad_btn.tooltip('Mic active — click to mute')
            vad_btn.set_visibility(False)   # hidden until Live mode selected

            def _toggle_vad():
                if not state.conv_audio or not state.conv_audio.is_listening:
                    return
                _vad_active[0] = not _vad_active[0]
                # Use dedicated _muted flag — NOT _tts_active.
                # _tts_active triggers post-TTS flush/grace on the falling edge,
                # which was causing 3 clicks needed to unmute.
                state.conv_audio._muted = _vad_active[0]
                icon = 'mic_off' if _vad_active[0] else 'mic'
                vad_btn.props(f'flat round dense icon={icon}')
                vad_btn.style(_vad_btn_style(_vad_active[0]))
                tip = 'Muted — click to unmute' if _vad_active[0] else 'Mic active — click to mute'
                vad_btn.tooltip(tip)
                ui.notify('🎙️ Mic muted' if _vad_active[0] else '🎙️ Mic active',
                          type='info', position='bottom-right', timeout=1500)

            vad_btn.on('click', _toggle_vad)

            with ui.element('div').classes('input-wrap flex-grow'):
                input_field = ui.input(
                    placeholder='Message… (Enter to send, Shift+Enter for newline)'
                ).classes('flex-grow chat-textarea').props(
                    'outlined borderless type=textarea autogrow'
                )

            # ── Response language picker ──────────────────────────────────
            _CHAT_LANGS = {
                'auto': '🌐 Auto',
                'en':    '🇬🇧 EN',
                'fr':    '🇫🇷 FR',
                'es':    '🇪🇸 ES',
                'de':    '🇩🇪 DE',
                'it':    '🇮🇹 IT',
                'pt':    '🇵🇹 PT',
                'zh-cn': '🇨🇳 ZH',
                'ru':    '🇷🇺 RU',
                'ar':    '🇸🇦 AR',
                'ja':    '🇯🇵 JA',
                'ko':    '🇰🇷 KO',
                'hi':    '🇮🇳 HI',
                'nl':    '🇳🇱 NL',
                'pl':    '🇵🇱 PL',
                'tr':    '🇹🇷 TR',
                'sv':    '🇸🇪 SV',
                'uk':    '🇺🇦 UK',
            }
            _init_lang = getattr(config, 'RESPONSE_LANGUAGE', 'auto')
            chat_lang_select = ui.select(
                _CHAT_LANGS,
                value=_init_lang if _init_lang in _CHAT_LANGS else 'auto',
            ).props('dense outlined borderless').classes('w-24 mode-select').tooltip(
                'Response language — Auto follows the user\'s language'
            )

            def _on_chat_lang_change():
                from managers.settings_manager import config as _cfg, save_settings
                _cfg.RESPONSE_LANGUAGE = chat_lang_select.value
                save_settings(_cfg)
                _label = _CHAT_LANGS.get(chat_lang_select.value, chat_lang_select.value)
                _msg = f'🌐 Language set to {_label}' if chat_lang_select.value != 'auto' \
                       else '🌐 Language: Auto (follows user)'
                ui.notify(_msg, type='info', position='bottom-right', timeout=2000)

            chat_lang_select.on('update:model-value', _on_chat_lang_change)
            # ─────────────────────────────────────────────────────────────

            ui.button(
                icon='send', on_click=lambda: asyncio.create_task(handle_send())
            ).classes('send-btn').props('dense round')

        status_label = ui.label('').classes('text-center w-full mt-2 text-sm text-slate-400 font-medium')

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────

    def _scroll_bottom():
        """
        Scroll chat to bottom. Uses a multi-strategy approach that stays
        reliable across many rounds:
        1. Direct #lumina-chat ID (fastest)
        2. Stored reference from first successful scroll (stable across rounds)
        3. Re-scan fallback with longest scrollable div
        Also installs a MutationObserver after first call to auto-scroll
        on future DOM changes without needing explicit calls.
        """
        try:
            ui.context.client.run_javascript(
                '(function(){'
                # ── Strategy: find and scroll the chat container ──
                '  function _findChat(){'
                '    var c=document.getElementById("lumina-chat");'
                '    if(c&&c.scrollHeight>200)return c;'
                # Stored reference from previous call
                '    if(window._lumina_chat&&document.contains(window._lumina_chat))return window._lumina_chat;'
                # Re-scan: find deepest scrollable div with most content
                '    var best=null,bh=0;'
                '    document.querySelectorAll("div").forEach(function(d){'
                '      var s=window.getComputedStyle(d);'
                '      var ov=s.overflowY;'
                '      if((ov==="auto"||ov==="scroll")&&d.scrollHeight>500&&d.scrollHeight>bh){'
                '        best=d;bh=d.scrollHeight;}'
                '    });'
                '    if(best){window._lumina_chat=best;}' 
                '    return best;'
                '  }'
                '  function _doScroll(){'
                '    var c=_findChat();'
                '    if(c){c.scrollTop=c.scrollHeight+9999;}'
                '  }'
                # Scroll immediately, then again after renders settle
                '  _doScroll();'
                '  requestAnimationFrame(function(){'
                '    _doScroll();'
                '    requestAnimationFrame(_doScroll);'
                '  });'
                '  setTimeout(_doScroll,100);'
                '  setTimeout(_doScroll,400);'
                # Install MutationObserver once — auto-scrolls on every future DOM change
                '  if(!window._lumina_obs){'
                '    var chat=_findChat();'
                '    if(chat){'
                '      window._lumina_obs=new MutationObserver(function(){'
                '        if(window._lumina_autoscroll!==false){'
                '          var c=window._lumina_chat||document.getElementById("lumina-chat");'
                '          if(c){var dist=c.scrollHeight-c.scrollTop-c.clientHeight;'
                '            if(dist<400){c.scrollTop=c.scrollHeight+9999;}}'  # only if near bottom
                '        }'
                '      });'
                '      window._lumina_obs.observe(chat,{childList:true,subtree:true,characterData:true});'
                '    }'
                '  }'
                '})();'
            )
        except Exception:
            pass

    def _bubble(text: str, who: str, show_feedback: bool = False):
        # ── Persist to tab storage so navigation doesn't clear the chat ──────
        try:
            _log = app.storage.tab.get('chat_log', [])
            _log.append({'who': who, 'text': text,
                         'ts': datetime.now().strftime('%H:%M')})
            if len(_log) > 200:           # cap to avoid bloat
                _log = _log[-200:]
            app.storage.tab['chat_log'] = _log
        except Exception:
            pass

        with chat_area:
            css = 'bubble-user' if who == 'user' else 'bubble-bot'
            stamp = datetime.now().strftime('%H:%M')
            align = 'items-end' if who == 'user' else 'items-start'
            with ui.column().classes(f'w-full gap-0 {align}'):
                md = ui.markdown(text).classes(css)
                with ui.row().classes('gap-2 items-center opacity-40 hover:opacity-100 transition-opacity mt-0.5'):
                    ui.label(stamp).classes('text-xs text-slate-500')
                    if who == 'user':
                        async def _edit_msg(t=text):
                            input_field.value = t
                            input_field.run_method('focus')
                        ui.button(icon='edit', on_click=_edit_msg).props(
                            'flat dense round size=xs'
                        ).style('color:#64748b').tooltip('Edit & resubmit')
            feedback_row = None
            if show_feedback and who == 'bot':
                feedback_row = ui.row().classes('gap-1 ml-2 mb-1 opacity-50 hover:opacity-100 transition-opacity')
                with feedback_row:
                    async def _fb_pos(fb_row=feedback_row):
                        last = getattr(state.persona, '_system', None)
                        if last:
                            resp = getattr(last, '_last_responses', [''])
                            last_resp = resp[-1] if resp else ''
                            await asyncio.to_thread(
                                state.persona.receive_feedback, True, user_manager.active_id, last_resp
                            )
                        ui.notify('👍 Positive feedback recorded', type='positive', position='bottom-right')
                        try:
                            fb_row.delete()
                        except Exception:
                            pass
                    async def _fb_neg(fb_row=feedback_row):
                        last = getattr(state.persona, '_system', None)
                        if last:
                            resp = getattr(last, '_last_responses', [''])
                            last_resp = resp[-1] if resp else ''
                            await asyncio.to_thread(
                                state.persona.receive_feedback, False, user_manager.active_id, last_resp
                            )
                        ui.notify('👎 Feedback noted — PandoraBOX will learn from this', type='warning', position='bottom-right')
                        try:
                            fb_row.delete()
                        except Exception:
                            pass
                    ui.button('👍', on_click=_fb_pos).props('flat dense round size=xs color=green')
                    ui.button('👎', on_click=_fb_neg).props('flat dense round size=xs color=red')
        _scroll_bottom()
        return md  # caller can update content for streaming


    # ── Network bubble function — routes child/master messages into chat ───────
    def _network_bubble(text: str, who: str, child_id: str, child_name: str):
        try:
            _log = app.storage.tab.get('chat_log', [])
            _log.append({'who': who, 'text': text, 'ts': datetime.now().strftime('%H:%M')})
            if len(_log) > 200: _log = _log[-200:]
            app.storage.tab['chat_log'] = _log
        except Exception:
            pass
        with chat_area:
            css   = 'bubble-child' if who == 'child' else 'bubble-master'
            stamp = datetime.now().strftime('%H:%M')
            _color = 'rgba(139,92,246,0.15);color:#a78bfa' if who == 'child' else 'rgba(20,184,166,0.15);color:#5eead4'
            with ui.column().classes('w-full gap-0 items-start'):
                ui.label(child_name).classes('text-xs font-semibold px-2 py-0.5 rounded-full mb-0.5').style(f'background:{_color}')
                ui.markdown(text).classes(css)
                ui.label(stamp).classes('text-xs text-slate-600 mt-0.5')
        _scroll_bottom()

    try:
        if state.lumina_network:
            state.lumina_network.set_bubble_fn(_network_bubble)
    except Exception:
        pass

    # ── Wire presence engine with proper per-client async bridge ────────────
    _presence_loop = asyncio.get_running_loop()
    _presence_queue = asyncio.Queue()

    async def _drain_presence_queue():
        while not _presence_queue.empty():
            try:
                txt = _presence_queue.get_nowait()
                _bubble(txt, 'bot')
            except Exception:
                pass

    def _presence_chat_fn(text):
        try:
            _presence_loop.call_soon_threadsafe(_presence_queue.put_nowait, text)
        except Exception:
            pass

    try:
        if state.vision and getattr(state.vision, 'presence_engine', None):
            state.vision.presence_engine.set_chat_fn(_presence_chat_fn)
    except Exception:
        pass

    def _typing_indicator():
        """Show animated typing dots; return the element to delete later."""
        with chat_area:
            el = ui.html('''
                <div class="typing-bubble">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            ''')
        _scroll_bottom()
        return el

    # ─────────────────────────────────────────────────────────────────
    #  Core send/respond
    # ─────────────────────────────────────────────────────────────────

    async def handle_send(user_text: Optional[str] = None, show_in_chat: bool = True):
        try:
            text = user_text or input_field.value.strip()
            if not text:
                return
            input_field.value = ''

            # ── Security gate — before bubble, LLM, or any processing ────
            try:
                if not security.allow_request(user_manager.active_id):
                    ui.notify('Too many requests — please wait a moment.', type='warning')
                    return
                text = security.sanitise_input(text, user_manager.active_id)
            except SecurityViolation as _sv:
                set_core_state('listening')
                main_page._current_state = 'listening'
                _bubble(text, 'user')
                _bubble(str(_sv), 'bot')
                set_core_state('')
                main_page._current_state = ''
                return
            except Exception as _se:
                logger.warning(f"Security gate error: {_se}")

            # Reset presence-engine silence detector — user is interacting
            try:
                if state.vision and state.vision.presence_engine:
                    state.vision.presence_engine.notify_user_interaction()
            except Exception:
                pass

            # If a network dialogue is active, inject user message into it
            try:
                if state.lumina_network and hasattr(state.lumina_network, 'inject_user_message'):
                    # Check if any dialogue task is running
                    if any(not t.done() for t in _dialogue_tasks.values()):
                        state.lumina_network.inject_user_message(text)
                        return  # message handled by dialogue — skip normal LLM response
            except Exception:
                pass

            # 🧬 Set organism to listening state
            set_core_state('listening')
            main_page._current_state = 'listening'

            if show_in_chat:
                _bubble(text, 'user')
            
            # Vision handling based on VISION_MODE setting
            vision_keywords = ['what do you see', 'look', 'camera', 'what\'s in front', 'describe the room', 'show me', 'can you see']
            vision_requested = any(keyword in text.lower() for keyword in vision_keywords)
            vision_enabled = state.vision is not None and state.vision.camera_active

            # ── Determine vision intent for this message ─────────────────
            # should_capture : call analyze_frame_async this turn?
            # show_bubble    : display the raw vision frame in chat?
            #
            # keyword  → capture + show bubble only when user asks for it
            # always   → capture + show bubble on every message
            # context  → capture silently on every message; LLM sees it but
            #            no separate vision bubble is shown to the user
            should_capture = False
            show_bubble    = False   # always False: PandoraBOX's LLM response IS the description now.
                                     # The raw VLM bubble was a pre-system-prompt workaround — retired.

            if vision_enabled:
                if config.VISION_MODE == 'keyword':
                    should_capture = vision_requested
                    # show_bubble stays False — PandoraBOX describes in her own response
                elif config.VISION_MODE == 'always':
                    should_capture = True
                    # show_bubble stays False
                elif config.VISION_MODE == 'context':
                    should_capture = True
                    # show_bubble stays False (was already False)

            # ── Capture vision exactly once (if needed) ──────────────────
            # Result is passed into the controller so it never calls
            # analyze_frame_async a second time on the same turn.
            injected_vision = None
            if should_capture:
                status_label.set_text('👁️ Analyzing camera feed…')
                set_core_state('face')
                main_page._current_state = 'face'

                try:
                    from core.agent_state import VisionOutput as _VO
                    # First-person prompt: output sounds like PandoraBOX's perception,
                    # not a photography critique. "I see…" not "The image shows…"
                    _capture_prompt = (
                        "Describe what you see right now in first person, "
                        "as if you are the one observing. Focus on the person, "
                        "objects, and any notable action. One concise paragraph."
                    )
                    raw_vision, faces = await state.vision.analyze_frame_async(
                        prompt=_capture_prompt
                    )
                    injected_vision = _VO.from_raw_text(raw_vision)
                    injected_vision._raw_text = raw_vision   # PandoraBOX reads this

                    if show_bubble and show_in_chat:
                        display_analysis = raw_vision
                        if faces:
                            face_info = f"\n\n**Detected {len(faces)} face(s):**\n"
                            for face in faces:
                                face_info += f"- {face['name']} (confidence: {face.get('confidence', 0):.0%})\n"
                            display_analysis += face_info
                        _bubble(display_analysis, 'bot')
                except Exception as e:
                    logger.warning(f"Vision capture failed: {e}")

            # ── Controller-driven flow (Sections 4-11 of spec) ───────────
            # The LLM always runs — when vision was captured it receives
            # injected_vision and can respond conversationally about what
            # it sees.  The old early-return on keyword/always is gone.
            status_label.set_text('🤖 Thinking…')
            typing = _typing_indicator() if show_in_chat else None

            # 🧬 Set thinking state
            set_core_state('thinking')
            main_page._current_state = 'thinking'

            # ── PandoraBOX is the cognitive engine — stream her response ──────
            bot_bubble = _bubble("▋", 'bot') if show_in_chat else None
            accumulated_text = ""
            result = None

            # ── Build vision_context for this turn ───────────────────────
            # Priority 1: on-demand capture (user triggered / VISION_MODE=always).
            #   Use the raw analysis text as before.
            # Priority 2: autonomous perception loop working memory.
            #   The PerceptionLoop ticks every 5 s and keeps working_memory current.
            #   Injected on every turn so PandoraBOX always knows what she sees —
            #   even when the user never mentioned vision.
            # Priority 3: nothing — vision off or camera not running.
            raw_vision_text = None
            if injected_vision is not None:
                raw_vision_text = getattr(injected_vision, '_raw_text', None)
            elif state.vision and state.vision.camera_active:
                _wm_block = state.vision.get_visual_context_for_prompt()
                if _wm_block:
                    raw_vision_text = _wm_block

            # ── Guard: persona must be ready ─────────────────────────────
            if not state.persona:
                if typing is not None:
                    typing.delete()
                if bot_bubble is not None:
                    bot_bubble.set_content("⚠️ PandoraBOX is still initialising — please wait a moment.")
                logger.warning("handle_send: state.persona is None — PandoraBOX not yet ready")
                return

            async for chunk in state.persona.get_response_stream(
                text,
                user_id=user_manager.active_id,
                vision_context=raw_vision_text,
            ):
                if isinstance(chunk, dict) and chunk.get("__meta__"):
                    result = chunk
                    break

                # First token arriving — remove typing indicator + status
                if not accumulated_text and typing is not None:
                    typing.delete()
                    typing = None
                    status_label.set_text('')
                    set_core_state('thinking')

                accumulated_text += chunk
                if bot_bubble is not None:
                    bot_bubble.set_content(accumulated_text + "▋")
                    # Scroll during streaming every ~200 chars to keep latest text visible
                    if len(accumulated_text) % 200 < len(chunk):
                        _scroll_bottom()

            # Streaming complete — show final text (stripped of reasoning blocks)
            if bot_bubble is not None:
                import re as _re
                def _strip_think_ui(t):
                    import re as _re2
                    _pairs = [
                        ("<think>",      "</think>"),
                        ("<|channel|>",  "<|/channel|>"),
                        ("[INST]",       "[/INST]"),
                        ("<<SYS>>",      "<</SYS>>"),
                        ("<|im_start|>", "<|im_end|>"),
                    ]
                    for op, cl in _pairs:
                        eo = _re2.escape(op); ec = _re2.escape(cl)
                        t = _re2.sub(eo + r".*?" + ec, "", t, flags=_re2.DOTALL | _re2.IGNORECASE)
                        t = _re2.sub(eo + r".*$",      "", t, flags=_re2.DOTALL | _re2.IGNORECASE)
                    t = _re2.sub("<[|]channel[|]>[^\n]*", "", t, flags=_re2.IGNORECASE)
                    t = _re2.sub(r"<\|[^|>]{1,40}\|>",  "", t)
                    return t.strip()
                final_text = (result["raw"] if result else None) or accumulated_text
                if final_text:
                    bot_bubble.set_content(_strip_think_ui(final_text))
                elif not accumulated_text:
                    bot_bubble.set_content("...")
            if bot_bubble is not None and result:
                pass  # already set above
                # Add 👍/👎 below this response so user can rate it in-context
                with chat_area:
                    _fb_row = ui.row().classes('gap-1 ml-2 mb-1 opacity-50 hover:opacity-100 transition-opacity')
                    with _fb_row:
                        async def _fb_pos_chat(_row=_fb_row):
                            last = getattr(state.persona, '_system', None)
                            if last:
                                resp = getattr(last, '_last_responses', [''])
                                last_resp = resp[-1] if resp else ''
                                await asyncio.to_thread(
                                    state.persona.receive_feedback, True, user_manager.active_id, last_resp
                                )
                            ui.notify('👍 Positive feedback recorded', type='positive', position='bottom-right')
                            try:
                                _row.delete()
                            except Exception:
                                pass
                        async def _fb_neg_chat(_row=_fb_row):
                            last = getattr(state.persona, '_system', None)
                            if last:
                                resp = getattr(last, '_last_responses', [''])
                                last_resp = resp[-1] if resp else ''
                                await asyncio.to_thread(
                                    state.persona.receive_feedback, False, user_manager.active_id, last_resp
                                )
                            ui.notify('👎 Feedback noted — PandoraBOX will learn from this', type='warning', position='bottom-right')
                            try:
                                _row.delete()
                            except Exception:
                                pass
                        ui.button('👍', on_click=_fb_pos_chat).props('flat dense round size=xs color=green')
                        ui.button('👎', on_click=_fb_neg_chat).props('flat dense round size=xs color=red')

            if typing is not None:
                typing.delete()
            status_label.set_text('')

            if not result:
                if not accumulated_text and bot_bubble is not None:
                    bot_bubble.set_content("I'm having trouble responding right now.")
                set_core_state('')
                main_page._current_state = ''
                return

            if state.persona and state.persona.research:
                low_conf = result.get("confidence", 1.0) < 0.5 if isinstance(result, dict) else False
                state.persona.record_query_signal(text, low_confidence=low_conf)

            # ── Security: validate LLM response before display ─────────
            if accumulated_text:
                try:
                    accumulated_text = security.validate_response(accumulated_text)
                except Exception:
                    pass

            # ── Update self-model and goal ecology after each response ────
            try:
                org = getattr(state.persona, '_organism', None)
                if org is not None:
                    success = bool(accumulated_text and len(accumulated_text) > 10)

                    # Self-model: record interaction outcome
                    if hasattr(org, 'self_model'):
                        # Guess domain from content
                        domain = 'conversation'
                        low = (text or '').lower()
                        if any(w in low for w in ['code','python','function','debug','error']):
                            domain = 'code_analysis'
                        elif any(w in low for w in ['search','find','research','look up']):
                            domain = 'research'
                        elif any(w in low for w in ['feel','emotion','sad','happy','stress']):
                            domain = 'emotional_support'
                        elif any(w in low for w in ['think','reason','why','explain','analyse']):
                            domain = 'reasoning'
                        org.self_model.record_interaction(success=success, domain=domain)

                    # Goal ecology: record satisfaction on dominant drive
                    if hasattr(org, 'goal_ecology') and hasattr(org, 'energy') and success:
                        dominant = org.goal_ecology.dominant_drive(org.energy.level())
                        org.goal_ecology.record_satisfaction(dominant.name, 0.6)

                    # Attractor nudge: successful social interaction
                    if hasattr(org, 'attractors') and success:
                        org.attractors.nudge('social_engagement', +0.005)
                        org.attractors.nudge('consistency', +0.003)
            except Exception as _e:
                logger.debug(f"Post-response organism update: {_e}")

            # Scroll to bottom after full response is rendered
            _scroll_bottom()

            tts_text = result.get("speech", accumulated_text)

            if voice_select.value != 'off' and state.audio:
                # 🧬 Speaking state
                set_core_state('speaking')
                main_page._current_state = 'speaking'
                asyncio.create_task(_speak(tts_text))
            else:
                # 🧬 Return to idle
                set_core_state('')
                main_page._current_state = ''
                
        except Exception as e:
            # Catch any UI-related errors (e.g., client disconnected)
            if 'deleted' in str(e).lower() or 'client' in str(e).lower():
                logger.debug(f"Client disconnected during message handling: {e}")
            else:
                logger.error(f"Error in handle_send: {e}")
            # 🧬 Reset to idle on error
            set_core_state('')
            main_page._current_state = ''

    # ─────────────────────────────────────────────────────────────────
    #  TTS playback with interruption
    # ─────────────────────────────────────────────────────────────────

    async def _speak(text: str):
        nonlocal _is_processing, _is_speaking, _is_proactive_running
        if not isinstance(text, str):
            logger.error(f"_speak received non-string ({type(text).__name__}) — skipping TTS")
            return
        text = text.strip()
        if not text:
            return
        state.tts_stop_event.clear()
        status_label.set_text('🔊 Speaking…')

        # 🧬 Set speaking state
        set_core_state('speaking')
        main_page._current_state = 'speaking'

        # _is_speaking blocks new VAD→LLM calls during TTS but still allows
        # transcribed text to appear in the chat bubble.
        _is_speaking   = True
        _is_processing = True   # also block LLM inference while speaking

        # Mute the VAD so the bot's own speaker audio doesn't self-trigger.
        if state.conv_audio:
            state.conv_audio._tts_active = True
        # Signal vision to back off face tracking while TTS is generating/playing.
        if state.vision:
            state.vision.audio_busy = True

        loop = asyncio.get_running_loop()

        audio_path = await loop.run_in_executor(None, state.audio.text_to_speech, text)
        if not audio_path:
            logger.warning("TTS returned no audio — check TTS provider settings")
            ui.notify('⚠️ TTS failed — check Settings → Voice', type='warning',
                      position='bottom-right', timeout=4000)
            status_label.set_text('')
            set_core_state('')
            main_page._current_state = ''
            if state.conv_audio:
                state.conv_audio._tts_active = False
            _is_speaking   = False
            _is_processing = False
            return

        try:
            import sounddevice as sd
            import soundfile as sf

            data, sr = sf.read(audio_path)

            def _play_blocking():
                sd.play(data, sr)
                try:
                    stream = sd.get_stream()
                    while stream.active:
                        if state.tts_stop_event.is_set():
                            sd.stop()
                            logger.info("🛑 TTS interrupted")
                            return
                        time.sleep(0.05)  # poll interval — already in executor thread
                except Exception:
                    pass
                sd.wait()

            await loop.run_in_executor(None, _play_blocking)

        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            status_label.set_text('')
            # 🧬 Return to idle after speaking
            set_core_state('')
            main_page._current_state = ''
            try:
                os.remove(audio_path)
            except Exception:
                pass
            # Un-mute VAD — process_loop will flush stale audio before resuming
            if state.conv_audio:
                state.conv_audio._tts_active = False
            # Re-enable full-speed face tracking now that audio is done.
            if state.vision:
                state.vision.audio_busy = False
            # Release both locks so voice input and LLM are accepted again
            _is_speaking   = False
            _is_processing = False

    # ─────────────────────────────────────────────────────────────────
    #  Voice mode switching
    # ─────────────────────────────────────────────────────────────────

    def on_voice_change():
        mode = voice_select.value
        # Show the VAD mute button only when Live mode is on
        vad_btn.set_visibility(mode == 'always')
        if mode != 'always':
            _vad_active[0] = False
            if state.conv_audio:
                state.conv_audio._muted = False

        # Stop any active conversation
        if state.conv_audio and state.conv_audio.is_listening:
            state.conv_audio.stop_conversation()
            logger.info("Stopped existing conversation due to mode change")

        if mode != 'off':
            if not state.ensure_audio_ready():
                ui.notify('Audio initialisation failed — check console', type='negative')
                voice_select.set_value('off')
                return

        if mode == 'always':
            _start_always_listening()
        # If mode is 'off' or any other value, conversation is already stopped above

    def _start_always_listening():
        # Stop any existing conversation first
        if state.conv_audio and state.conv_audio.is_listening:
            state.conv_audio.stop_conversation()
            # stop_conversation() joins its own threads — no sleep needed here
        
        if not state.conv_audio or not state.conv_audio.vad_available:
            ui.notify('VAD not available — check console for details', type='warning')
            voice_select.set_value('off')
            return

        def _on_speech_start():
            state.tts_stop_event.set()
            # Reset idle timer — user is actively speaking, proactive must not fire
            _agent_state.mark_interaction()
            # Notify orchestrator so drives / social decay reset correctly
            if state.orchestrator is not None:
                try:
                    # No transcript exists yet at speech-start — this just
                    # signals user presence/activity to the orchestrator.
                    state.orchestrator.notify_user_message("", user_manager.active_id)
                except Exception:
                    pass
                # 🧬 Set listening state when speech detected
                set_core_state('listening')
                main_page._current_state = 'listening'

        def _on_transcription(text: str):
            # ✅ Use the global queue instead of per-page queue
            state.transcription_queue.put(text)
        
        def _on_speech_end():
            # 🧬 Return to idle after speech ends
            set_core_state('')
            main_page._current_state = ''

        try:
            # Flush any audio already in the queue from mic warmup before
            # starting — prevents startup hallucinations like "thanks for watching"
            if hasattr(state, 'transcription_queue'):
                while not state.transcription_queue.empty():
                    try: state.transcription_queue.get_nowait()
                    except Exception: break

            state.conv_audio.start_conversation(
                on_transcription=_on_transcription,
                on_speech_start=_on_speech_start,
                on_speech_end=_on_speech_end
            )

            ui.notify('🔴 Always-Listen active', type='info', position='bottom-right')
            logger.info("✅ Live mode activated successfully")
        except Exception as e:
            logger.error(f"Failed to start always-listening: {e}")
            ui.notify(f'Failed to start live mode: {e}', type='negative')
            voice_select.set_value('off')

    voice_select.on('update:model-value', on_voice_change)
    input_field.on('keydown.enter', lambda e: 
        asyncio.create_task(handle_send())
        if not (isinstance(e, dict) and e.get('shiftKey'))
        else None
    )
    # Reset idle timer on any keypress — proactive must not fire while user is typing
    input_field.on('keydown', lambda: _agent_state.mark_interaction())

    # ── Boot session restore ──────────────────────────────────────────────────
    # Load previous conversation context from disk so PandoraBOX remembers
    if state.llm:
        from managers.session_manager import get_session_manager
        sess = get_session_manager()
        # Give session manager access to generate_bare for LLM compression
        sess.set_llm_fn(state.llm.generate_bare)
        boot_info = sess.load_on_boot(state.llm)
        if boot_info.get("restored"):
            turns   = boot_info.get("turns", 0)
            has_sum = boot_info.get("has_summary", False)
            age     = boot_info.get("session_age", "")[:16]
            parts   = []
            if has_sum:
                parts.append("earlier context compressed")
            if turns:
                parts.append(f"{turns} recent turns")
            detail  = " + ".join(parts) if parts else "session"
            ui.notify(
                f"Session restored ({detail})",
                type='info', position='bottom-right', timeout=4000
            )
            logger.info(f"Session restored: {boot_info}")

    # ── Restore chat bubbles from tab storage (navigation-safe replay) ─────────
    _chat_log = []
    try:
        _chat_log = app.storage.tab.get('chat_log', [])
    except Exception:
        pass

    if _chat_log:
        # Re-render all persisted bubbles silently (no re-save, no scroll storm)
        with chat_area:
            for _entry in _chat_log:
                _who   = _entry.get('who', 'bot')
                _text  = _entry.get('text', '')
                _stamp = _entry.get('ts', '')
                if not _text:
                    continue
                _css   = 'bubble-user' if _who == 'user' else 'bubble-bot'
                _align = 'items-end' if _who == 'user' else 'items-start'
                with ui.column().classes(f'w-full gap-0 {_align}'):
                    ui.markdown(_text).classes(_css)
                    with ui.row().classes('gap-2 items-center opacity-40 mt-0.5'):
                        ui.label(_stamp).classes('text-xs text-slate-500')
        _scroll_bottom()
        # Skip the welcome bubble — chat is already visible
        pass
    else:
            # Welcome bubble — grounded wake-up when narrative exists, generic otherwise
        if not (state.llm and state.llm.history):
            _bubble(f"Hello. I'm **{_gpn()}**. What's on your mind?", 'bot')
        else:
            # Try to build a grounded wake-up from persisted narrative
            _wake_text = "Welcome back. I remember our previous conversation."
            try:
                if state.persona and state.persona.is_ready:
                    _org = getattr(state.persona, '_organism', None)
                    _ni  = getattr(_org, 'narrative_identity', None)
                    _has_content = (
                        _ni is not None and (
                            bool(getattr(_ni, 'life_story',   [])) or
                            bool(getattr(_ni, '_current_arc', ''))
                        )
                    )
                    if _has_content and state.llm:
                        import time as _wt
                        _parts = []
                        _arc = getattr(_ni, '_current_arc', '')
                        if _arc:
                            _parts.append(f"Last narrative arc: {_arc}")
                        _chapters = sorted(
                            getattr(_ni, 'life_story', []),
                            key=lambda c: c.significance, reverse=True
                        )[:2]
                        for _c in _chapters:
                            _secs = _wt.time() - _c.timestamp
                            _age  = (f"{int(_secs/3600)}h ago"  if _secs < 86400
                                     else f"{int(_secs/86400)}d ago")
                            _parts.append(f"  - {_c.title} ({_age}): {_c.description[:80]}")
                        _values = getattr(_ni, 'core_values', [])
                        if _values:
                            _parts.append(f"Core values: {', '.join(_values[:3])}")

                        # ── Visual narrative memory ────────────────────────────
                        if state.vision and hasattr(state.vision, 'query_visual_memory'):
                            try:
                                _vis_hits = state.vision.query_visual_memory(
                                    "what I observed recently", top_k=2
                                )
                                if _vis_hits:
                                    _vis_lines = []
                                    for _vh in _vis_hits:
                                        _vt  = _vh.get('time', '?')
                                        _vd  = _vh.get('description', '')[:80]
                                        _vf  = ', '.join(_vh.get('faces', []))
                                        _vf  = f" [{_vf}]" if _vf else ""
                                        _vis_lines.append(f"  - {_vt}: {_vd}{_vf}")
                                    _parts.append(
                                        "Last visual memories:\n" + "\n".join(_vis_lines)
                                    )
                                _known = (state.vision.get_known_faces()
                                          if hasattr(state.vision, 'get_known_faces') else [])
                                _named = [f for f in _known if f['name'] != 'Unknown Person']
                                if _named:
                                    _name_list = ', '.join(
                                        f"{f['name']} (seen {f['seen_count']}×)"
                                        for f in _named[:4]
                                    )
                                    _parts.append(f"People I recognise: {_name_list}")
                            except Exception:
                                pass

                        _grounding = "\n".join(_parts)
                        _wake_prompt = (
                            f"You are waking up after a restart. "
                            f"Here is what you remember:\n{_grounding}\n\n"
                            f"Write ONE short sentence (under 20 words) as your opening. "
                            f"Reference something specific and sensory — a face, a moment, "
                            f"a visual detail you recall. Do not say 'I remember' or "
                            f"'Welcome back'. Sound awake and present, not archival."
                        )
                        _generated = state.llm.generate_bare(
                            _wake_prompt,
                            f"You are {_gpn()}, an AI with genuine memory, continuity, and visual perception.",
                            max_tokens=60,
                            temperature=0.7,
                        )
                        if _generated and len(_generated.strip()) > 5:
                            _wake_text = _generated.strip().strip('"')
            except Exception as _we:
                logger.debug(f"[app] Wake-up generation failed (non-fatal): {_we}")
            _bubble(_wake_text, 'bot')
    # end of else (no chat log)

    # ── PandoraBOX Network panel ─────────────────────────────────────────────────
    _dialogue_tasks: dict = {}
    _dialogue_stops: dict = {}

    try:
        if state.lumina_network:
            with ui.expansion('🌐 PandoraBOX Network', icon='hub').classes('w-full'):
                with ui.column().classes('w-full gap-3 p-3'):
                    _clist = state.lumina_network.list_children()
                    if not _clist:
                        ui.label('No children — add LUMINA_CHILDREN to config.json').classes('text-slate-500 text-xs')
                    else:
                        for _ch in _clist:
                            _cid = _ch['id']
                            _cnm = _ch['name']
                            with ui.card().classes('w-full p-3').style('background:#1e293b;border:1px solid #334155'):
                                with ui.row().classes('w-full items-center justify-between'):
                                    with ui.row().classes('items-center gap-2'):
                                        _sc = '#22c55e' if _ch['online'] else '#64748b'
                                        ui.html(f'<div style="width:8px;height:8px;border-radius:50%;background:{_sc}"></div>')
                                        ui.label(_cnm).classes('font-semibold text-sm text-slate-200')
                                    ui.label(f"{_ch['latency_ms']:.0f}ms").classes('text-xs text-slate-500')
                                ui.label(_ch['url']).classes('text-xs text-slate-600 font-mono mt-1')

                                with ui.row().classes('gap-2 mt-2 flex-wrap'):

                                    # ── Ping ──────────────────────────────────────
                                    async def _ping(__cid=_cid, __cn=_cnm):
                                        ok = await state.lumina_network.ping_child(__cid)
                                        ui.notify(f"{__cn}: {'✅ online' if ok else '❌ offline'}",
                                                  type='positive' if ok else 'warning', position='bottom-right')
                                    ui.button('Ping', icon='wifi', on_click=_ping).props('flat dense size=sm color=indigo')

                                    # ── Ask ───────────────────────────────────────
                                    async def _ask(__cid=_cid, __cn=_cnm):
                                        q = input_field.value.strip()
                                        if not q:
                                            ui.notify('Type a message first', type='warning', position='bottom-right')
                                            return
                                        input_field.value = ''
                                        await state.lumina_network.send_to_child(__cid, q)
                                    ui.button('Ask', icon='send', on_click=_ask).props('flat dense size=sm color=purple')

                                    # ── Dialogue start/stop toggle ─────────────────
                                    _dlg_btn = ui.button('Dialogue ▶', icon='forum').props('flat dense size=sm color=teal')
                                    _dlg_running = [False]  # mutable flag — survives closure

                                    async def _toggle_dlg(_b=_dlg_btn, __cid=_cid, __cn=_cnm, _flag=_dlg_running):
                                        if _flag[0]:
                                            # ── STOP ──────────────────────────────────────
                                            _flag[0] = False
                                            if __cid in _dialogue_stops:
                                                _dialogue_stops[__cid].set()
                                            _b.props('flat dense size=sm color=teal')
                                            _b.set_text('Dialogue ▶')
                                            ui.notify(f'Dialogue with {__cn} stopped', position='bottom-right')
                                            # Notify child of closure
                                            try:
                                                asyncio.create_task(
                                                    state.lumina_network.send_to_child(
                                                        __cid,
                                                        "Our dialogue is pausing here. Thank you for the exchange.",
                                                        show_in_ui=True,
                                                    )
                                                )
                                            except Exception:
                                                pass
                                        else:
                                            # ── START ─────────────────────────────────────
                                            _flag[0] = True
                                            topic   = input_field.value.strip()
                                            input_field.value = ''
                                            stop_ev = asyncio.Event()
                                            _dialogue_stops[__cid] = stop_ev
                                            _b.props('flat dense size=sm color=red')
                                            _b.set_text('Stop ■')
                                            ui.notify(f'Dialogue with {__cn} started', position='bottom-right')

                                            async def _run_and_reset(__flag=_flag, __b=_b, __t=topic, __c=__cid):
                                                try:
                                                    await state.lumina_network.start_dialogue(
                                                        __c, topic=__t, stop_event=_dialogue_stops[__c])
                                                except Exception:
                                                    pass
                                                # Reset button when dialogue ends naturally
                                                if __flag[0]:
                                                    __flag[0] = False
                                                    __b.props('flat dense size=sm color=teal')
                                                    __b.set_text('Dialogue ▶')

                                            _dialogue_tasks[__cid] = asyncio.create_task(_run_and_reset())

                                    _dlg_btn.on('click', _toggle_dlg)

                                    # ── Turing Test ───────────────────────────────
                                    # Store child refs for the upload callback
                                    _t_cid = _cid
                                    _t_cnm = _cnm

                                    async def _run_turing(e):
                                        try:
                                            txt = e.content.read().decode('utf-8', errors='replace')
                                        except Exception as _re:
                                            ui.notify(f'Read error: {_re}', type='negative', position='bottom-right')
                                            return
                                        qs = [
                                            ln.strip().lstrip('-* ').strip()
                                            for ln in txt.splitlines()
                                            if ln.strip()
                                            and not ln.strip().startswith('#')
                                            and len(ln.strip()) > 8
                                        ]
                                        if not qs:
                                            ui.notify('No questions found in script', type='warning', position='bottom-right')
                                            return
                                        ui.notify(f'Turing: {len(qs)} Qs → {_t_cnm}', position='bottom-right')
                                        _network_bubble(f'📋 Turing test — {len(qs)} questions', 'master', 'master', 'System')
                                        for i, q in enumerate(qs, 1):
                                            _network_bubble(f'Q{i}: {q}', 'master', 'master', 'Evaluator')
                                            await state.lumina_network.send_to_child(_t_cid, q, show_in_ui=True)
                                            await asyncio.sleep(2.0)
                                        _network_bubble('📋 Turing test complete.', 'master', 'master', 'System')

                                    _turing_upload = ui.upload(
                                        on_upload=_run_turing,
                                        auto_upload=True,
                                        max_files=1,
                                    ).props('accept=.md,.txt').classes('hidden')

                                    ui.button('Turing', icon='psychology',
                                              on_click=lambda: _turing_upload.run_method('pickFiles')
                                              ).props('flat dense size=sm color=amber')
    except Exception:
        pass

    # ✅ REGISTER TIMERS
    ui.timer(0.1, update_camera_feed)        # 10 FPS camera polling
    ui.timer(0.1, check_transcription_queue)  # 10 Hz transcription polling

    # Background research — check RAC queue every 30 minutes
    async def _bg_research_tick():
        if state.persona and state.persona.research and state.persona.research.rac.has_pending_mode3():
            try:
                logger.info("Background research tick: running queued Mode 3 job")
                result = await asyncio.to_thread(state.persona.trigger_background_research)
                if result and result.summary:
                    logger.info(f"Background research done: {result.research_id} conf={result.confidence:.2f} nodes={result.knowledge_nodes}")
                    # Notify user if they're on the main page
                    ui.notify(
                        f"PandoraBOX completed background research: {result.goal[:50]}",
                        type='info', position='bottom-right', timeout=8000
                    )
            except Exception as _e:
                logger.error(f"Background research tick error: {_e}")

    ui.timer(1800, _bg_research_tick)  # every 30 minutes
    ui.timer(0.5, _drain_presence_queue)  # presence-engine bubble drain

    # CAG: periodic sync from organism cognitive state
    def _cag_sync():
        try:
            if state.acc:
                state.acc.update_from_organism()
        except Exception:
            pass
    ui.timer(30.0, _cag_sync)  # sync CAG from organism every 30s
    ui.timer(30.0, _sync_emotion_label)  # keep header emotion text live (was frozen at page load)
    ui.timer(1.0, _check_scroll_position)  # show/hide scroll-to-bottom btn
    ui.add_css(
        '.chat-textarea .q-field__native,'
        '.chat-textarea textarea {'
        'min-height:40px!important;'
        'max-height:160px!important;'
        'overflow-y:auto!important;'
        'resize:none!important;'
        'line-height:1.6!important;'
        'padding:6px 8px!important;}'
        '.chat-textarea{align-self:stretch;}'
    )



# ─────────────────────────────────────────────
#  Vision Page with Fixed Size Container
# ─────────────────────────────────────────────
