"""
pages/vision_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vision page.
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

@ui.page('/vision')
async def vision_page():
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout. On a loaded system this can be exceeded, killing
    # the page coroutine before anything renders — confirmed in production
    # logs (TimeoutError at cognitive_dashboard_page.py:398, same pattern).
    # Raised to 10s and wrapped so a genuine failure shows a visible
    # message instead of a silently blank page.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(f"[vision_page] Client connection timed out after 10s")
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

    with ui.header().classes('app-header'):
        with ui.row().classes('items-center gap-3'):
            ui.button(
                icon='arrow_back', on_click=lambda: ui.navigate.to('/')
            ).props('flat dense round').style('color:#64748b')
            ui.html('''
                <div style="width:38px;height:38px;border-radius:10px;
                     background:linear-gradient(135deg,#6366f1,#8b5cf6);
                     display:flex;align-items:center;justify-content:center;
                     font-size:20px">👁️</div>
            ''')
            ui.label('Camera View').classes('app-title')
        
        with ui.row().classes('items-center gap-2'):
            camera_status = ui.label('').classes('status-pill pill-vision')
            uptime_label = ui.label('').classes('text-xs text-slate-600')
            ui.button(
                icon='settings',
                on_click=lambda: ui.navigate.to('/settings?tab=vision')
            ).props('flat dense round').tooltip('Vision settings').style('color:#64748b')

    # Create UI elements first so they exist
    with ui.column().classes('w-full max-w-4xl mx-auto p-4 gap-4'):
        with ui.row().classes('w-full items-center gap-3 flex-wrap'):
            camera_switch = ui.switch(
                'Enable Camera',
                value=state.vision and state.vision.camera_active
            )
            
            # Camera test button
            test_camera_btn = ui.button('📋 Test Camera').props('flat color=blue')
            
            # Capture button
            capture_btn = ui.button('📸 Capture & Analyze').props('color=purple outline')
            
            # Reset camera button
            reset_camera_btn = ui.button('🔄 Reset Camera').props('flat')

        # Camera preview - fixed size container
        preview_container = ui.card().classes('w-full fixed-video-container').style('background: #000;')
        
        with preview_container:
            # Use interactive_image which is better for live video updates
            camera_image = ui.interactive_image().classes('w-full h-full').style('object-fit: contain;')
            # Set placeholder
            camera_image.set_source('https://via.placeholder.com/800x600/1e293b/64748b?text=No+Camera+Feed')
            
            # Status overlay
            overlay = ui.html('''
                <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                    📷 Camera Off
                </div>
            ''')

        # Analysis results
        with ui.expansion('Latest Analysis', icon='analytics').classes('w-full'):
            analysis_text = ui.label('No analysis yet').classes('text-slate-400 p-2')

        # Face recognition panel
        with ui.expansion('Face Recognition', icon='face').classes('w-full'):
            with ui.row().classes('w-full justify-between items-center mb-2'):
                ui.label('Known Faces').classes('text-sm font-semibold')
                register_btn = ui.button('➕ Register New Person').props('flat dense color=green').classes('text-xs')
            faces_container = ui.column().classes('w-full gap-2 p-2')

        # Vision memory
        with ui.expansion('Vision Memory', icon='memory').classes('w-full'):
            memory_container = ui.column().classes('w-full gap-2 p-2')
            
            with ui.row().classes('w-full items-center gap-2'):
                memory_query = ui.input('Search memories...').classes('flex-grow dark-input').props('outlined dense')
                search_btn = ui.button('🔍').props('flat')

        # ── Quick Settings (inline shortcut — full config at /settings?tab=vision) ──
        with ui.expansion('⚙️ Quick Settings', icon='tune').classes('w-full'):
            with ui.column().classes('w-full gap-4 p-3'):

                with ui.grid(columns=2).classes('w-full gap-3'):
                    ui.select(
                        {0: 'Camera 0 (Built-in)', 1: 'Camera 1 (External)',
                         2: 'Camera 2', 3: 'Camera 3'},
                        label='Camera',
                        value=getattr(config, 'CAMERA_ID', 0),
                        on_change=lambda e: setattr(config, 'CAMERA_ID', int(e.value))
                    ).props('outlined dense').classes('dark-input')

                    ui.select(
                        {5: '5 fps (low)', 10: '10 fps (balanced)', 15: '15 fps (high)'},
                        label='Frame Rate',
                        value=getattr(config, 'CAMERA_FPS', 5),
                        on_change=lambda e: setattr(config, 'CAMERA_FPS', int(e.value))
                    ).props('outlined dense').classes('dark-input')

                ui.select(
                    {'640x480': '640×480 SD', '1280x720': '1280×720 HD'},
                    label='Resolution',
                    value=getattr(config, 'CAMERA_RESOLUTION', '640x480'),
                    on_change=lambda e: setattr(config, 'CAMERA_RESOLUTION', e.value)
                ).props('outlined dense').classes('w-full dark-input')

                ui.select(
                    {
                        'keyword': '🔤 Keyword-Triggered',
                        'always':  '👁️ Always Analyse',
                        'context': '🧠 Context (Background)',
                    },
                    label='Vision Mode',
                    value=getattr(config, 'VISION_MODE', 'keyword'),
                    on_change=lambda e: setattr(config, 'VISION_MODE', e.value)
                ).props('outlined dense').classes('w-full dark-input')

                ui.input(
                    'Vision Model (LLaVA)',
                    value=getattr(config, 'LAVA_MODEL', 'llava:latest'),
                    on_change=lambda e: setattr(config, 'LAVA_MODEL', e.value)
                ).props('outlined dense').classes('w-full dark-input')

                with ui.row().classes('w-full justify-between items-center'):
                    async def _qs_save():
                        try:
                            save_settings(config)
                            ui.notify('Settings saved', type='positive', position='top')
                        except Exception as ex:
                            ui.notify(f'Save failed: {ex}', type='negative', position='top')

                    ui.button('💾 Save', on_click=_qs_save).props('color=indigo dense')
                    ui.button(
                        '⚙️ Full Vision Settings',
                        on_click=lambda: ui.navigate.to('/settings?tab=vision')
                    ).props('flat dense color=purple')


    # ─────────────────────────────────────────────────────────────────
    #  Function Definitions
    # ─────────────────────────────────────────────────────────────────

    async def test_camera():
        """Test camera functionality"""
        if not state.vision:
            ui.notify('Vision system not initialized', type='warning')
            return
        
        with ui.dialog() as dialog, ui.card():
            with ui.column().classes('items-center p-4'):
                ui.spinner(size='lg', color='blue')
                ui.label('Testing camera...').classes('text-center mt-2')
        dialog.open()
        
        try:
            # Test current camera
            success = await asyncio.to_thread(state.vision.start_camera)
            if success:
                ui.notify('✅ Camera test successful', type='positive')
                camera_switch.value = True
                camera_status.set_text('👁️ Active')
            else:
                ui.notify('❌ Camera test failed - check console for details', type='negative')
                camera_switch.value = False
        except Exception as e:
            ui.notify(f'Test error: {e}', type='negative')
            logger.error(f"Camera test error: {e}")
        finally:
            dialog.close()

    async def reset_camera():
        """Reset the camera connection"""
        if not state.vision:
            ui.notify('Vision system not initialized', type='warning')
            return
        
        # Show loading
        with ui.dialog() as dialog, ui.card():
            with ui.column().classes('items-center p-4'):
                ui.spinner(size='lg', color='purple')
                ui.label('Resetting camera...').classes('text-center mt-2')
        dialog.open()
        
        try:
            # Run in thread to avoid blocking
            def _reset():
                if state.vision.camera_active:
                    state.vision.stop_camera()
                time.sleep(1)
                return state.vision.start_camera()
            
            success = await asyncio.to_thread(_reset)
            
            if success:
                ui.notify('Camera reset successful', type='positive')
                camera_status.set_text('👁️ Active')
            else:
                ui.notify('Camera reset failed', type='negative')
                camera_status.set_text('👁️ Failed')
        except Exception as e:
            logger.error(f"Reset error: {e}")
            ui.notify(f'Reset error: {e}', type='negative')
        finally:
            dialog.close()

    def update_faces_panel():
        """Update the faces panel — thumbnails, stats, first_seen, seen_count."""
        try:
            faces_container.clear()

            if not state.vision or not getattr(state.vision, 'face_encodings', None):
                with faces_container:
                    ui.label('No faces recognised yet').classes('text-slate-400 text-sm')
                return

            stats = (state.vision.get_face_stats()
                     if hasattr(state.vision, 'get_face_stats') else {})
            known_list = (state.vision.get_known_faces()
                          if hasattr(state.vision, 'get_known_faces')
                          else [{"id": fid, "name": n, "first_seen": None,
                                 "seen_count": 0, "thumb_path": None}
                                for fid, n in state.vision.face_names.items()])

            with faces_container:
                if stats:
                    with ui.row().classes('w-full gap-4 mb-2 text-xs text-slate-400'):
                        ui.label(f"👥 Total: {stats.get('total_registered', 0)}")
                        ui.label(f"✅ Named: {stats.get('named', 0)}")
                        ui.label(f"❓ Unnamed: {stats.get('unnamed', 0)}")

                for face in known_list:
                    face_id    = face['id']
                    name       = face['name']
                    seen_count = face.get('seen_count', 0)
                    first_seen = face.get('first_seen', '')
                    thumb_path = face.get('thumb_path')

                    age_label = ''
                    if first_seen:
                        try:
                            from datetime import datetime as _dt
                            import time as _tt
                            _secs = _tt.time() - _dt.fromisoformat(first_seen).timestamp()
                            age_label = (f"{int(_secs/60)}m ago" if _secs < 3600
                                         else f"{int(_secs/3600)}h ago" if _secs < 86400
                                         else f"{int(_secs/86400)}d ago")
                        except Exception:
                            pass

                    is_unknown = name in ('Unknown Person', 'Unknown', '')
                    row_cls    = 'bg-slate-900 border border-slate-700' if is_unknown else 'bg-slate-800'

                    with ui.row().classes(f'w-full items-center gap-3 p-2 {row_cls} rounded mb-1'):
                        if thumb_path:
                            try:
                                ui.image(thumb_path).classes(
                                    'w-10 h-10 rounded-full object-cover border border-slate-600'
                                )
                            except Exception:
                                ui.icon('person').classes('text-2xl text-slate-500')
                        else:
                            ui.icon('person').classes('text-2xl text-slate-500')

                        with ui.column().classes('flex-1 gap-0'):
                            ui.label(name).classes(
                                f"text-sm font-medium "
                                f"{'text-amber-400' if is_unknown else 'text-slate-100'}"
                            )
                            parts = []
                            if seen_count:
                                parts.append(f"seen {seen_count}x")
                            if age_label:
                                parts.append(f"first: {age_label}")
                            if parts:
                                ui.label(' - '.join(parts)).classes('text-xs text-slate-500')

                        with ui.row().classes('gap-1'):
                            async def rename(fid=face_id):
                                result = [None]
                                cur = state.vision.face_names.get(fid, '')
                                with ui.dialog() as _dlg, ui.card().classes('p-5 gap-3'):
                                    ui.label('Name This Person').classes('font-semibold text-lg')
                                    _inp = ui.input('Name', value=cur,
                                                    placeholder='e.g. Alice').props(
                                        'outlined dense'
                                    ).classes('w-full dark-input')
                                    with ui.row().classes('gap-3 mt-2'):
                                        def _save():
                                            result[0] = _inp.value.strip()
                                            _dlg.close()
                                        ui.button('Save', on_click=_save).props('color=indigo')
                                        ui.button('Cancel', on_click=_dlg.close).props('flat')
                                await _dlg
                                if result[0]:
                                    state.vision.update_face_name(fid, result[0])
                                    try:
                                        existing = user_manager.find_by_face(fid)
                                        if not existing:
                                            nm = result[0]
                                            match = next(
                                                (u for u in user_manager.named_users()
                                                 if u.display_name.lower() == nm.lower()),
                                                None
                                            )
                                            if match:
                                                user_manager.link_face(match.id, fid)
                                                ui.notify(f'Linked to {match.display_name}', type='positive')
                                            else:
                                                profile = user_manager.create(nm, face_ids=[fid])
                                                user_manager.set_active(profile.id)
                                                ui.notify(f'Created profile for {nm}', type='positive')
                                    except Exception as _ue:
                                        logger.debug(f"[FacePanel] user link: {_ue}")
                                    update_faces_panel()

                            async def delete(fid=face_id, nm=name):
                                confirmed = [False]
                                with ui.dialog() as _dlg, ui.card().classes('p-5 gap-3'):
                                    ui.label('Remove Face').classes('font-semibold text-lg')
                                    ui.label(f'Remove "{nm}"? Cannot be undone.').classes(
                                        'text-slate-400 text-sm'
                                    )
                                    with ui.row().classes('gap-3 mt-2'):
                                        def _del():
                                            confirmed[0] = True
                                            _dlg.close()
                                        ui.button('Remove', on_click=_del).props('color=red')
                                        ui.button('Cancel', on_click=_dlg.close).props('flat')
                                await _dlg
                                if confirmed[0]:
                                    state.vision.delete_face(fid)
                                    update_faces_panel()
                                    ui.notify(f'Removed {nm}', type='info')

                            ui.button('Name', on_click=lambda fid=face_id: rename(fid)).props(
                                'flat dense'
                            ).classes('text-xs text-indigo-400')
                            ui.button('X', on_click=lambda fid=face_id, nm=name: delete(fid, nm)).props(
                                'flat dense round color=red'
                            ).classes('text-xs')

        except RuntimeError as e:
            if 'slot stack' not in str(e).lower() and 'context' not in str(e).lower():
                logger.error(f"Update faces panel error: {e}")
        except Exception as e:
            logger.error(f"Update faces panel error: {e}")

    async def register_new_person():
        """Capture current frame and register person — or upload a photo."""
        try:
            if not state.vision:
                ui.notify('Vision not initialised', type='warning')
                return

            mode = [None]
            with ui.dialog() as _mode_dlg, ui.card().classes('p-5 gap-4 min-w-64'):
                ui.label('Register Person').classes('font-semibold text-lg')
                ui.label('How would you like to register?').classes('text-slate-400 text-sm')
                with ui.column().classes('gap-2 w-full'):
                    def _use_camera():
                        mode[0] = 'camera'
                        _mode_dlg.close()
                    def _use_upload():
                        mode[0] = 'upload'
                        _mode_dlg.close()
                    ui.button('Use current camera frame', on_click=_use_camera).props(
                        'color=indigo outline'
                    ).classes('w-full')
                    ui.button('Upload a photo', on_click=_use_upload).props(
                        'color=teal outline'
                    ).classes('w-full')
                    ui.button('Cancel', on_click=_mode_dlg.close).props('flat').classes('w-full')
            await _mode_dlg

            if mode[0] is None:
                return

            # MODE: upload photo
            if mode[0] == 'upload':
                reg_result = [None]
                with ui.dialog() as _up_dlg, ui.card().classes('p-5 gap-3 min-w-72'):
                    ui.label('Upload Photo').classes('font-semibold text-lg')
                    ui.label('Choose a clear single-person photo.').classes('text-slate-400 text-sm')
                    _name_inp = ui.input("Person's name", placeholder='e.g. Alice').props(
                        'outlined dense'
                    ).classes('w-full dark-input')
                    _upload_path = [None]

                    def _on_upload(e):
                        import tempfile, os as _os
                        tmp = tempfile.NamedTemporaryFile(
                            suffix='.jpg', delete=False, dir='data/faces'
                        )
                        tmp.write(e.content.read())
                        tmp.close()
                        _upload_path[0] = tmp.name

                    ui.upload(
                        label='Choose photo',
                        on_upload=_on_upload,
                        max_files=1,
                        auto_upload=True,
                    ).props('accept=".jpg,.jpeg,.png"').classes('w-full')

                    with ui.row().classes('gap-3 mt-2'):
                        def _do_upload_reg():
                            reg_result[0] = (_name_inp.value.strip(), _upload_path[0])
                            _up_dlg.close()
                        ui.button('Register', on_click=_do_upload_reg).props('color=teal')
                        ui.button('Cancel', on_click=_up_dlg.close).props('flat')
                await _up_dlg

                if reg_result[0]:
                    reg_name, reg_path = reg_result[0]
                    if not reg_name:
                        ui.notify('Please enter a name', type='warning')
                        return
                    if not reg_path:
                        ui.notify('Please upload a photo', type='warning')
                        return
                    if not hasattr(state.vision, 'register_face_from_image'):
                        ui.notify('register_face_from_image not available', type='negative')
                        return
                    ok, msg = await asyncio.to_thread(
                        state.vision.register_face_from_image, reg_path, reg_name
                    )
                    if ok:
                        ui.notify(f'{msg}', type='positive')
                        update_faces_panel()
                    else:
                        ui.notify(f'{msg}', type='negative')
                return

            # MODE: camera frame
            if not state.vision.camera_active:
                ui.notify('Camera must be active to capture a face', type='warning')
                return

            frame = state.vision.get_current_frame()
            if frame is None:
                ui.notify('No camera feed available', type='warning')
                return

            loop  = asyncio.get_running_loop()
            faces = await loop.run_in_executor(None, state.vision.detect_faces, frame)

            if not faces:
                ui.notify('No faces detected — move closer to the camera', type='warning')
                return

            target_face = faces[0]
            if len(faces) > 1:
                ui.notify(f'{len(faces)} faces detected — registering the most prominent', type='info')

            if target_face['name'] not in ('Unknown', 'Unknown Person', ''):
                ui.notify(
                    f'Already registered as "{target_face["name"]}"',
                    type='warning'
                )
                return

            result_name = [None]
            with ui.dialog() as _dlg, ui.card().classes('p-5 gap-3'):
                ui.label('Who is this?').classes('font-semibold text-lg')
                thumb = state.vision.faces_file.parent / f"{target_face['id']}_thumb.jpg"
                if thumb.exists():
                    ui.image(str(thumb)).classes('w-24 h-24 rounded object-cover mx-auto')
                _inp = ui.input('Name', placeholder='e.g. Alice').props(
                    'outlined dense'
                ).classes('w-full dark-input')
                with ui.row().classes('gap-3 mt-2'):
                    def _do_reg():
                        result_name[0] = _inp.value.strip()
                        _dlg.close()
                    ui.button('Register', on_click=_do_reg).props('color=indigo')
                    ui.button('Cancel', on_click=_dlg.close).props('flat')
            await _dlg

            name = result_name[0]
            if name:
                face_id = target_face['id']
                state.vision.update_face_name(face_id, name)
                existing = user_manager.find_by_face(face_id)
                if not existing:
                    name_match = next(
                        (u for u in user_manager.named_users()
                         if u.display_name.lower() == name.lower()),
                        None
                    )
                    if name_match:
                        user_manager.link_face(name_match.id, face_id)
                        ui.notify(f'Linked face to {name_match.display_name}', type='positive')
                    else:
                        profile = user_manager.create(name, face_ids=[face_id])
                        user_manager.set_active(profile.id)
                        ui.notify(f'Registered {profile.display_name}', type='positive')
                else:
                    ui.notify(f'Face already linked to {existing.display_name}', type='info')
                update_faces_panel()

        except Exception as e:
            logger.error(f"Register person error: {e}")
            try:
                ui.notify(f'Registration failed: {e}', type='negative')
            except Exception:
                logger.error(f"Could not show notification: {e}")

    async def capture_and_analyze():
        """Capture current frame and analyze with LAVA"""
        if not state.vision or not state.vision.camera_active:
            ui.notify('Camera not active', type='warning')
            return
        
        analysis_text.set_text('Analyzing...')
        
        try:
            analysis, faces = await state.vision.analyze_frame_async()
            analysis_text.set_text(analysis)
            
            if faces:
                ui.notify(f'✅ Detected {len(faces)} faces', type='positive')
            
            # Update memory panel
            update_memory_panel()
            
        except Exception as e:
            analysis_text.set_text(f'Error: {e}')
            logger.error(f"Analysis error: {e}")

    def update_memory_panel():
        """Update the vision memory panel"""
        try:
            memory_container.clear()
            
            if not state.vision or not hasattr(state.vision, 'vision_meta') or not state.vision.vision_meta:
                with memory_container:
                    ui.label('No vision memories yet').classes('text-slate-400 text-sm')
                return
            
            with memory_container:
                for i, memory in enumerate(state.vision.vision_meta[-5:]):  # Last 5 memories
                    with ui.card().classes('w-full bg-slate-800 p-2'):
                        ui.label(f'📸 Memory #{memory["id"]}').classes('text-xs text-purple-400')
                        ui.label(memory['analysis'][:100] + '...').classes('text-xs text-slate-300')
                        ui.label(f'👤 {len(memory["faces"])} faces').classes('text-xs text-slate-400')
        except Exception as e:
            logger.error(f"Update memory panel error: {e}")

    async def search_memory():
        """Search vision memories"""
        query = memory_query.value
        if not query or not state.vision:
            return
        
        try:
            results = await asyncio.to_thread(state.vision.search_vision_memory, query, 3)
            
            memory_container.clear()
            with memory_container:
                if not results:
                    ui.label('No matching memories').classes('text-slate-400 text-sm')
                else:
                    for dist, memory in results:
                        with ui.card().classes('w-full bg-slate-800 p-2'):
                            ui.label(f'📸 Memory #{memory["id"]}').classes('text-xs text-purple-400')
                            ui.label(memory['analysis'][:150] + '...').classes('text-xs text-slate-300')
                            ui.label(f'Score: {1/(1+dist):.2f}').classes('text-xs text-slate-400')
        except Exception as e:
            logger.error(f"Search memory error: {e}")

    async def update_uptime():
        """Update uptime display periodically - runs in UI context"""
        while True:
            try:
                uptime = state.get_uptime()
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                uptime_label.set_text(f'Uptime: {hours}h {minutes}m')
            except:
                pass
            await asyncio.sleep(60)

    async def on_camera_toggle(e):
        """Handle camera switch toggle - runs in UI context"""
        is_active = camera_switch.value
        
        if is_active:
            if state.vision:
                # Show loading
                camera_status.set_text('👁️ Starting...')
                overlay.set_content('''
                    <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                        ⏳ Starting camera...
                    </div>
                ''')
                
                # Run in thread to avoid blocking
                success = await asyncio.to_thread(state.vision.start_camera)
                
                if success:
                    ui.notify('Camera started', type='positive')
                    camera_status.set_text('👁️ Active')
                    # Wire vision onto organism so brain bridge VIS reads camera state
                    try:
                        organism = getattr(state.persona, '_organism', None) if state.persona else None
                        if organism is not None:
                            organism._vision_manager = state.vision
                            loop = getattr(organism, '_loop', None)
                            if loop and hasattr(loop, 'attach_vision'):
                                loop.attach_vision(state.vision)
                            # Reset last_ambient_vision so next slow cycle fires immediately
                            if loop and hasattr(loop, '_last_ambient_vision'):
                                loop._last_ambient_vision = 0.0
                    except Exception as _vw:
                        logger.debug(f"Vision→organism wiring on toggle: {_vw}")
                    overlay.set_content('''
                        <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                            📷 Camera active
                        </div>
                    ''')
                else:
                    ui.notify('Failed to start camera', type='negative')
                    camera_switch.value = False
                    camera_status.set_text('👁️ Failed')
                    overlay.set_content('''
                        <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                            ❌ Camera failed
                        </div>
                    ''')
        else:
            if state.vision:
                await asyncio.to_thread(state.vision.stop_camera)
                ui.notify('Camera stopped', type='info')
                camera_status.set_text('👁️ Off')
                camera_image.set_source('https://via.placeholder.com/800x600/1e293b/64748b?text=No+Camera+Feed')
                overlay.set_content('''
                    <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                        📷 Camera Off
                    </div>
                ''')

    # Simplified camera update loop - just uses the pre-encoded frames
    async def update_camera_feed():
        """Update camera feed in UI with pre-encoded frames"""
        while True:
            try:
                if state.vision and state.vision.camera_active:
                    # Get pre-encoded frame - no processing on UI thread!
                    encoded_frame = state.vision.get_latest_encoded_frame()
                    
                    if encoded_frame:
                        camera_image.set_source(f'data:image/jpeg;base64,{encoded_frame}')
                        
                        # Update status periodically
                        try:
                            if state.vision.face_encodings:
                                overlay.set_content(f'''
                                    <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                                        👤 {len(state.vision.face_encodings)} face(s) known
                                    </div>
                                ''')
                            else:
                                overlay.set_content('''
                                    <div class="vision-overlay" style="background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 20px; position: absolute; top: 10px; right: 10px; z-index: 10;">
                                        📷 Camera active
                                    </div>
                                ''')
                        except Exception:
                            pass  # element deleted — camera tab navigated away
                else:
                    # Camera off - ensure placeholder is shown
                    if camera_image.source != 'https://via.placeholder.com/800x600/1e293b/64748b?text=No+Camera+Feed':
                        camera_image.set_source('https://via.placeholder.com/800x600/1e293b/64748b?text=No+Camera+Feed')
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Camera feed error: {e}")
            
            await asyncio.sleep(0.1)  # 10 Hz — matches main page timer, reduces event-loop pressure

    # Connect button handlers
    test_camera_btn.on('click', test_camera)
    reset_camera_btn.on('click', reset_camera)
    capture_btn.on('click', capture_and_analyze)
    register_btn.on('click', register_new_person)
    search_btn.on('click', search_memory)
    camera_switch.on('update:model-value', on_camera_toggle)

    # Populate known faces immediately on page load
    update_faces_panel()

    # Start background tasks
    asyncio.create_task(update_camera_feed())
    asyncio.create_task(update_uptime())


# ─────────────────────────────────────────────
#  Settings Page with Fixed Status Tab
# ─────────────────────────────────────────────



