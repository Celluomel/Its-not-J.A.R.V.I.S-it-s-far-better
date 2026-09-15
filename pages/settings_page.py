"""
pages/settings_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Settings.
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
    get_birth_date, set_birth_date,
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

@ui.page('/settings')
async def settings_page(tab: str = 'llm'):
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout. On a loaded system this can be exceeded, killing
    # the page coroutine before anything renders — confirmed in production
    # logs (TimeoutError at cognitive_dashboard_page.py:398, same pattern).
    # Raised to 10s and wrapped so a genuine failure shows a visible
    # message instead of a silently blank page.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(f"[settings_page] Client connection timed out after 10s")
        ui.label('⚠ Connection timed out — please reload.').classes(
            'text-yellow-400 text-lg font-bold p-8'
        )
        return
    ui.add_css(GLOBAL_CSS)
    # Force scroll on settings page — override chat layout's overflow:hidden
    ui.add_head_html('''
        <style>
        html, body {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            height: auto !important;
        }
        .q-page-container, .q-page { overflow: visible !important; height: auto !important; }
        </style>
        <script>
        document.documentElement.style.overflowY = "auto";
        document.body.style.overflowY = "auto";
        document.body.style.height = "auto";
        </script>
    ''')

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
                     font-size:20px">⚙️</div>
            ''')
            ui.label('Settings').classes('app-title')

    with ui.column().classes('w-full max-w-4xl mx-auto px-4 gap-5 settings-page-content'):

        with ui.element('div').classes('settings-card w-full'):
            with ui.tabs().props('dense align=left scrollable').classes('settings-tabs') as tabs:
                tab_llm    = ui.tab('llm',    label='🧠 LLM')
                tab_memory = ui.tab('memory', label='💾 Memory')
                tab_voice  = ui.tab('voice',  label='🎙️ Voice')
                tab_vision = ui.tab('vision', label='👁️ Vision')
                tab_status = ui.tab('status', label='📊 Status')
                tab_lumina = ui.tab('lumina', label='✨ Lumina')
                tab_tools    = ui.tab('tools',    label='🔧 Tools')
                tab_analytics = ui.tab('analytics', label='📈 Analytics')

            with ui.tab_panels(tabs, value=tab).classes('w-full p-6 settings-panel'):

                # ── LLM tab ───────────────────────────────────────────
                with ui.tab_panel('llm'):
                    # ── Persona Identity ───────────────────────────────
                    ui.label('PERSONA IDENTITY').classes('section-label')
                    with ui.grid(columns=2).classes('w-full gap-4 mb-4'):
                        persona_name_input = ui.input(
                            'Persona Name',
                            value=getattr(config, 'PERSONA_NAME', 'Lumina'),
                            placeholder='Lumina'
                        ).props('outlined dense').classes('dark-input')
                        with ui.element('div').classes('flex items-center'):
                            ui.html('<div style="color:#64748b;font-size:11px;line-height:1.5">'
                                    'The name your AI persona uses for itself.<br>'
                                    'Used in all prompts and responses. Restart to apply.</div>')

                    ui.separator().classes('mb-4')
                    ui.label('LLM PROVIDER').classes('section-label')
                    # Default URLs per provider — auto-filled on change, still editable
                    _S_PROVIDER_URLS = {
                        'ollama':   'http://localhost:11434',
                        'lmstudio': 'http://localhost:1234/v1',
                        'openai':   'https://api.openai.com/v1',
                    }
                    with ui.grid(columns=2).classes('w-full gap-4'):
                        llm_url   = ui.input(
                            'Base URL', value=config.LLM_BASE_URL
                        ).props('outlined dense').classes('col-span-2 dark-input')

                        def _s_on_provider_change(e):
                            default = _S_PROVIDER_URLS.get(e.value, '')
                            current = llm_url.value.strip()
                            if current in _S_PROVIDER_URLS.values() or current == '':
                                llm_url.value = default

                        llm_prov  = ui.select(
                            ['ollama', 'lmstudio', 'openai'],
                            label='Provider', value=config.LLM_PROVIDER,
                            on_change=_s_on_provider_change
                        ).props('outlined dense').classes('dark-input')
                        llm_model = ui.input(
                            'Model Name', value=config.LLM_MODEL
                        ).props('outlined dense').classes('dark-input')
                        openai_key = ui.input(
                            'OpenAI API Key', value=config.OPENAI_API_KEY, password=True
                        ).props('outlined dense').classes('col-span-2 dark-input')

                    # Provider hints
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('PROVIDER NOTES').classes('section-label')
                        ui.markdown(
                            "- **ollama** → `http://localhost:11434` · run `ollama serve` first\n"
                            "- **lmstudio** → `http://localhost:1234/v1` · start LM Studio server\n"
                            "- **openai** → requires API key above"
                        ).style('font-size:0.82rem;color:#94a3b8;line-height:1.7')

                # ── Memory tab ────────────────────────────────────────
                with ui.tab_panel('memory'):
                    ui.label('MEMORY BACKEND').classes('section-label')
                    mem_backend = ui.select(
                        ['faiss', 'dict', 'cognee'],
                        label='Backend', value=config.MEMORY_BACKEND
                    ).props('outlined dense').classes('w-full dark-input')

                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('BACKEND NOTES').classes('section-label')
                        ui.markdown(
                            "- **faiss** — semantic vector search, persists to `~/.robot_agent/` ✅ recommended\n"
                            "- **dict** — simple recency buffer, in-memory only\n"
                            "- **cognee** — knowledge graph memory, requires `pip install cognee`\n""  · Ollama: set embed model to blank (auto-uses nomic-embed-text)\n""  · LM Studio: load an embedding model in LM Studio, enter its name above"
                        ).style('font-size:0.82rem;color:#94a3b8;line-height:1.7')

                    # ── Storage paths ──────────────────────────────────
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('STORAGE PATHS').classes('section-label')
                        ui.html('<div style="color:#64748b;font-size:11px;margin-bottom:10px">'
                                'Paths are relative to the Lumina working directory. '
                                'Change and restart to take effect. '
                                'Defaults: <code>data/persona/</code></div>')

                        mem_db_path = ui.input(
                            'SQLite DB path',
                            value=getattr(config, 'MEMORY_DB_PATH', 'data/persona/ai_system.db'),
                            placeholder='data/persona/ai_system.db'
                        ).props('outlined dense').classes('w-full dark-input mb-2')

                        mem_faiss_path = ui.input(
                            'FAISS index path',
                            value=getattr(config, 'MEMORY_FAISS_PATH', 'data/persona/faiss_index.bin'),
                            placeholder='data/persona/faiss_index.bin'
                        ).props('outlined dense').classes('w-full dark-input mb-2')

                        mem_world_path = ui.input(
                            'World model path',
                            value=getattr(config, 'MEMORY_WORLD_PATH', 'data/persona/world_model.json'),
                            placeholder='data/persona/world_model.json'
                        ).props('outlined dense').classes('w-full dark-input mb-2')

                        mem_persona_path = ui.input(
                            'Persona data folder',
                            value=getattr(config, 'MEMORY_PERSONA_PATH', 'data/persona'),
                            placeholder='data/persona'
                        ).props('outlined dense').classes('w-full dark-input mb-2')

                        mem_cognee_path = ui.input(
                            'Cognee data folder',
                            value=getattr(config, 'MEMORY_COGNEE_PATH', 'data/persona/cognee'),
                            placeholder='data/persona/cognee'
                        ).props('outlined dense').classes('w-full dark-input mb-2') \
                         .tooltip('Root folder for Cognee\'s graph, vector and metadata databases. '
                                  'Relative to the Lumina working directory. Restart to apply.')

                        mem_cognee_embed = ui.input(
                            'Cognee embedding model',
                            value=getattr(config, 'MEMORY_COGNEE_EMBED_MODEL', ''),
                            placeholder='nomic-embed-text  (leave blank = auto)'
                        ).props('outlined dense').classes('w-full dark-input mb-3') \
                         .tooltip(
                            'Embedding model for Cognee graph memory.\n'
                            'Ollama: leave blank — uses nomic-embed-text automatically.\n'
                            'LM Studio: enter the name of the embedding model you loaded '
                            '(e.g. nomic-embed-text, text-embedding-nomic-embed-text-v1-5).\n'
                            'Must be loaded separately from the chat model in LM Studio.'
                        )

                        # Show / hide the Cognee fields depending on selected backend
                        def _toggle_cognee_path(e=None):
                            is_cognee = mem_backend.value == 'cognee'
                            mem_cognee_path.set_visibility(is_cognee)
                            mem_cognee_embed.set_visibility(is_cognee)

                        _toggle_cognee_path()                        # set initial state
                        mem_backend.on('update:model-value', _toggle_cognee_path)

                        # Save memory path settings
                        def _save_memory_paths():
                            config.MEMORY_BACKEND           = mem_backend.value
                            config.MEMORY_DB_PATH           = mem_db_path.value.strip()    or 'data/persona/ai_system.db'
                            config.MEMORY_FAISS_PATH        = mem_faiss_path.value.strip() or 'data/persona/faiss_index.bin'
                            config.MEMORY_WORLD_PATH        = mem_world_path.value.strip() or 'data/persona/world_model.json'
                            config.MEMORY_PERSONA_PATH      = mem_persona_path.value.strip() or 'data/persona'
                            config.MEMORY_COGNEE_PATH       = mem_cognee_path.value.strip() or 'data/persona/cognee'
                            config.MEMORY_COGNEE_EMBED_MODEL = mem_cognee_embed.value.strip()
                            config.save()
                            ui.notify('Memory paths saved — restart Lumina to apply', type='positive')

                        ui.button('Save paths', on_click=_save_memory_paths,
                                  icon='save').props('flat dense color=indigo')

                        # Show current DB stats
                        async def _show_db_stats():
                            try:
                                import sqlite3, pathlib
                                db = pathlib.Path(config.MEMORY_DB_PATH)
                                if not db.exists():
                                    ui.notify(f'DB not found: {db}', type='warning')
                                    return
                                conn = sqlite3.connect(str(db))
                                total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                                tiers = conn.execute(
                                    "SELECT memory_tier, COUNT(*) FROM memories GROUP BY memory_tier"
                                ).fetchall()
                                conn.close()
                                tier_str = ' · '.join(f'{t}:{n}' for t,n in tiers)
                                ui.notify(f'DB: {total} memories — {tier_str}', timeout=5000)
                            except Exception as e:
                                ui.notify(f'DB error: {e}', type='negative')

                        ui.button('DB stats', on_click=_show_db_stats,
                                  icon='storage').props('flat dense color=cyan').classes('ml-2')

                    # ── Cognee graph visualisation ────────────────────
                    with ui.element('div').classes('info-card mt-4') as _cognee_viz_card:
                        ui.label('COGNEE GRAPH EXPLORER').classes('section-label')
                        ui.html(
                            '<div style="color:#64748b;font-size:11px;margin-bottom:10px">'
                            'Generates an interactive HTML graph from the Cognee knowledge graph. '
                            'Only available when <b style="color:#a78bfa">Backend = cognee</b>. '
                            'Opens in your default browser.'
                            '</div>'
                        )

                        _viz_status = ui.label('').classes('text-xs text-slate-400 mt-1')

                        async def _open_cognee_graph():
                            """Generate cognee visualize_graph() HTML and open it."""
                            if config.MEMORY_BACKEND != 'cognee':
                                ui.notify(
                                    'Switch Memory Backend to "cognee" first, then restart.',
                                    type='warning'
                                )
                                return

                            _viz_status.set_text('⏳ Generating graph…')
                            try:
                                import asyncio, pathlib, webbrowser

                                cognee_dir = pathlib.Path(
                                    getattr(config, 'MEMORY_COGNEE_PATH', 'data/persona/cognee')
                                )
                                cognee_dir.mkdir(parents=True, exist_ok=True)
                                out_path = str((cognee_dir / 'cognee_graph.html').resolve())

                                # Import Cognee's visualize function
                                try:
                                    from cognee.api.v1.visualize.visualize import visualize_graph
                                except ImportError:
                                    try:
                                        from cognee import visualize_graph  # older API
                                    except ImportError:
                                        ui.notify(
                                            'cognee not installed — run: pip install cognee',
                                            type='negative'
                                        )
                                        _viz_status.set_text('❌ cognee not installed')
                                        return

                                # Run async visualize in the current loop
                                await visualize_graph(out_path)

                                _viz_status.set_text(f'✅ Saved → {out_path}')
                                ui.notify('Graph generated! Opening in browser…', type='positive')
                                webbrowser.open(f'file://{pathlib.Path(out_path).resolve()}')

                            except Exception as e:
                                logger.error(f'Cognee visualize error: {e}')
                                _viz_status.set_text(f'❌ Error: {e}')
                                ui.notify(f'Graph error: {e}', type='negative')

                        async def _cognee_graph_stats():
                            """Print node/edge count from the Cognee graph without rendering."""
                            if config.MEMORY_BACKEND != 'cognee':
                                ui.notify('Backend must be set to "cognee".', type='warning')
                                return
                            try:
                                import cognee as _cog
                                g = await _cog.get_graph_engine()
                                nodes = await g.get_number_of_nodes()
                                edges = await g.get_number_of_edges()
                                _viz_status.set_text(f'📊 {nodes} nodes · {edges} edges')
                                ui.notify(
                                    f'Cognee graph: {nodes} nodes, {edges} edges',
                                    timeout=5000
                                )
                            except Exception as e:
                                ui.notify(f'Stats error: {e}', type='negative')

                        # Show/hide the whole card based on selected backend
                        def _toggle_cognee_viz(e=None):
                            _cognee_viz_card.set_visibility(mem_backend.value == 'cognee')

                        _toggle_cognee_viz()
                        mem_backend.on('update:model-value', _toggle_cognee_viz)

                        with ui.row().classes('gap-2 mt-2'):
                            ui.button(
                                'Open Graph', on_click=_open_cognee_graph, icon='hub'
                            ).props('flat dense color=purple')
                            ui.button(
                                'Graph stats', on_click=_cognee_graph_stats, icon='analytics'
                            ).props('flat dense color=indigo')

                    # ── Cognee sync ───────────────────────────────────
                    with ui.element('div').classes('info-card mt-4') as _cognee_sync_card:
                        ui.label('SYNC ALL SOURCES → COGNEE').classes('section-label')
                        ui.html(
                            '<div style="color:#64748b;font-size:11px;margin-bottom:8px">'
                            'Injects all Lumina data sources into the Cognee knowledge graph in one pass. '
                            'Safe to run multiple times — Cognee deduplicates.'
                            '</div>'
                        )

                        # Source checklist (user can deselect sources)
                        with ui.grid(columns=2).classes('w-full gap-1 mb-3'):
                            _sync_src_memories   = ui.checkbox('Memories (SQLite)',    value=True)
                            _sync_src_semantic   = ui.checkbox('Semantic graph (DB)',  value=True)
                            _sync_src_narrative  = ui.checkbox('Narrative identity',   value=True)
                            _sync_src_self       = ui.checkbox('Self-concept',         value=True)
                            _sync_src_relational = ui.checkbox('Relational memory',    value=True)
                            _sync_src_goals      = ui.checkbox('Goals',                value=True)
                            _sync_src_emotional  = ui.checkbox('Emotional state',      value=True)
                            _sync_src_thoughts   = ui.checkbox('Thought stream',       value=True)

                        # Scrollable live log
                        _sync_log = ui.textarea('').props(
                            'outlined dense readonly rows=8'
                        ).classes('w-full dark-input font-mono text-xs').style('font-size:10px')

                        async def _sync_all_to_cognee():
                            import pathlib, json, sqlite3

                            if config.MEMORY_BACKEND != 'cognee':
                                ui.notify('Switch backend to "cognee" first.', type='warning')
                                return

                            try:
                                import cognee as _cog
                                # Apply LM Studio / Ollama config before any cognee call
                                try:
                                    from managers.memory_manager import CogneeMemoryBackend as _CMB
                                    _tmp = object.__new__(_CMB)
                                    _tmp._configure_cognee(_tmp)
                                    # Apply via API too
                                    _lm_provider = getattr(config, 'LLM_PROVIDER', 'ollama')
                                    _embed_model  = getattr(config, 'MEMORY_COGNEE_EMBED_MODEL', '') or 'nomic-embed-text'
                                    _base_url     = getattr(config, 'LLM_BASE_URL', 'http://localhost:11434')
                                    _api_key      = 'lm-studio' if _lm_provider == 'lmstudio' else 'ollama'
                                    _prov         = 'openai' if _lm_provider == 'lmstudio' else 'ollama'
                                    _cog.config.set_embedding_provider(_prov)
                                    _cog.config.set_embedding_model(_embed_model)
                                    _cog.config.set_embedding_endpoint(_base_url + ('' if _base_url.endswith('/v1') else '/v1'))
                                    _cog.config.set_embedding_api_key(_api_key)
                                    _cog.config.set_embedding_dimensions(768)
                                except Exception as _cfg_e:
                                    pass  # config errors are non-fatal
                            except ImportError:
                                ui.notify('pip install cognee', type='negative')
                                return

                            persona_dir = pathlib.Path(
                                getattr(config, 'MEMORY_PERSONA_PATH', 'data/persona')
                            )
                            ai_db_path = pathlib.Path(
                                getattr(config, 'MEMORY_DB_PATH', 'data/persona/ai_system.db')
                            )
                            lines = []

                            def log(msg):
                                lines.append(msg)
                                _sync_log.value = '\n'.join(lines)

                            log('── Cognee sync started ──')

                            # ── 1. semantic_memory.db → direct triplets ──────────
                            if _sync_src_semantic.value:
                                sem_db = persona_dir / 'semantic_memory.db'
                                if sem_db.exists():
                                    try:
                                        conn = sqlite3.connect(str(sem_db))
                                        concepts = conn.execute(
                                            'SELECT name, strength FROM concepts'
                                        ).fetchall()
                                        relations = conn.execute('''
                                            SELECT c1.name, c2.name, r.rel_type, r.weight
                                            FROM relations r
                                            JOIN concepts c1 ON r.source_id = c1.id
                                            JOIN concepts c2 ON r.target_id = c2.id
                                        ''').fetchall()
                                        conn.close()
                                        try:
                                            g = await _cog.get_graph_engine()
                                            for name, strength in concepts:
                                                await g.add_node(
                                                    name, {'strength': strength, 'src': 'semantic_memory'}
                                                )
                                            for src, tgt, rel_type, weight in relations:
                                                await g.add_edge(src, tgt, rel_type, {'weight': weight})
                                            log(f'✅ semantic_memory.db  {len(concepts)} nodes · {len(relations)} edges')
                                        except Exception as _ge:
                                            # Fallback: convert to text and let cognify do it
                                            sem_text = '\n'.join(
                                                f'{s} {rt} {t} (weight={w:.2f})'
                                                for s, t, rt, w in relations[:300]
                                            )
                                            await _cog.add(sem_text)
                                            log(f'✅ semantic_memory.db  {len(relations)} relations (text fallback)')
                                    except Exception as e:
                                        log(f'⚠️  semantic_memory.db  {e}')
                                else:
                                    log('–  semantic_memory.db  not found')

                            # ── 2. Collect text chunks ───────────────────────────
                            texts = []

                            # ai_system.db → memories table
                            if _sync_src_memories.value and ai_db_path.exists():
                                try:
                                    conn = sqlite3.connect(str(ai_db_path))
                                    rows = conn.execute(
                                        'SELECT text, timestamp, memory_type, emotional_valence '
                                        'FROM memories ORDER BY timestamp DESC LIMIT 500'
                                    ).fetchall()
                                    conn.close()
                                    added = 0
                                    for text, ts, mtype, valence in rows:
                                        if text and len(text.strip()) > 15:
                                            texts.append(
                                                f'[{mtype}/{valence} @{ts[:10]}] {text.strip()}'
                                            )
                                            added += 1
                                    log(f'✅ ai_system.db memories  {added} entries queued')
                                except Exception as e:
                                    log(f'⚠️  ai_system.db  {e}')

                            # narrative_identity.json
                            if _sync_src_narrative.value:
                                ni_path = persona_dir / 'narrative_identity.json'
                                if ni_path.exists():
                                    try:
                                        ni = json.loads(ni_path.read_text(encoding='utf-8'))
                                        chunks = []
                                        vals = ni.get('core_values', [])
                                        if vals:
                                            chunks.append('Core values: ' + ', '.join(vals))
                                        arc = ni.get('current_arc', '')
                                        if arc:
                                            chunks.append(f'Current narrative arc: {arc}')
                                        for entry in ni.get('life_story', [])[:60]:
                                            desc = entry.get('description', '') if isinstance(entry, dict) else str(entry)
                                            if desc and len(desc) > 10:
                                                chunks.append(f'Life event: {desc}')
                                        for b in (ni.get('beliefs', []) or [])[:20]:
                                            if isinstance(b, str) and len(b) > 5:
                                                chunks.append(f'Belief: {b}')
                                        texts.extend(chunks)
                                        log(f'✅ narrative_identity.json  {len(chunks)} entries queued')
                                    except Exception as e:
                                        log(f'⚠️  narrative_identity.json  {e}')

                            # self_concept.json
                            if _sync_src_self.value:
                                sc_path = persona_dir / 'self_concept.json'
                                if sc_path.exists():
                                    try:
                                        sc = json.loads(sc_path.read_text(encoding='utf-8'))
                                        chunks = []
                                        beliefs = sc.get('beliefs', {})
                                        items = beliefs.values() if isinstance(beliefs, dict) else beliefs
                                        for v in items:
                                            stmt = v.get('statement', '') if isinstance(v, dict) else str(v)
                                            if stmt and len(stmt) > 5:
                                                chunks.append(f'Self-belief: {stmt}')
                                        texts.extend(chunks)
                                        log(f'✅ self_concept.json  {len(chunks)} beliefs queued')
                                    except Exception as e:
                                        log(f'⚠️  self_concept.json  {e}')

                            # relational_memory.json
                            if _sync_src_relational.value:
                                rm_path = persona_dir / 'relational_memory.json'
                                if rm_path.exists():
                                    try:
                                        rm = json.loads(rm_path.read_text(encoding='utf-8'))
                                        chunks = []
                                        for user_id, data in rm.items():
                                            topics  = data.get('shared_topics', [])
                                            arc     = data.get('emotional_arc', '')
                                            count   = data.get('interaction_count', 0)
                                            trust   = data.get('trust_score', 0)
                                            prefs   = data.get('communication_prefs', {})
                                            # shared_topics can be a dict {topic: count}
                                            # or a list — handle both
                                            if isinstance(topics, dict):
                                                top_topics = [t for t, _ in sorted(
                                                    topics.items(),
                                                    key=lambda x: x[1], reverse=True
                                                )[:12]]
                                            else:
                                                top_topics = list(topics)[:12]
                                            summary = (
                                                f'User "{user_id}": {count} interactions, '
                                                f'trust={trust:.2f}, arc="{arc}". '
                                                f'Topics: {", ".join(top_topics)}. '
                                                f'Prefs: {json.dumps(prefs)}'
                                            )
                                            chunks.append(summary)
                                        texts.extend(chunks)
                                        log(f'✅ relational_memory.json  {len(chunks)} user summaries queued')
                                    except Exception as e:
                                        log(f'⚠️  relational_memory.json  {e}')

                            # goals.json
                            if _sync_src_goals.value:
                                goals_path = persona_dir / 'goals.json'
                                if goals_path.exists():
                                    try:
                                        gdata  = json.loads(goals_path.read_text(encoding='utf-8'))
                                        goals  = gdata.get('goals', {})
                                        items  = goals.values() if isinstance(goals, dict) else goals
                                        chunks = []
                                        for g in items:
                                            topic    = g.get('topic', '')
                                            origin   = g.get('origin', '')
                                            priority = g.get('priority', 0)
                                            if topic:
                                                chunks.append(
                                                    f'Active goal [{origin}]: "{topic}" (priority={priority:.2f})'
                                                )
                                        texts.extend(chunks)
                                        log(f'✅ goals.json  {len(chunks)} goals queued')
                                    except Exception as e:
                                        log(f'⚠️  goals.json  {e}')

                            # emotional_state.json → last snapshot + trend prose
                            if _sync_src_emotional.value:
                                em_path = persona_dir / 'emotional_state.json'
                                if em_path.exists():
                                    try:
                                        em   = json.loads(em_path.read_text(encoding='utf-8'))
                                        hist = em.get('history', [])
                                        if hist:
                                            last = hist[-1]
                                            first = hist[0]
                                            trend_val  = last.get('valence', 0) - first.get('valence', 0)
                                            trend_str  = 'rising' if trend_val > 0.05 else 'falling' if trend_val < -0.05 else 'stable'
                                            summary = (
                                                f'Emotional state snapshot: '
                                                f'valence={last.get("valence",0):.2f} ({trend_str}), '
                                                f'arousal={last.get("arousal",0):.2f}, '
                                                f'curiosity={last.get("curiosity",0):.2f}, '
                                                f'warmth={last.get("warmth",0):.2f}, '
                                                f'anxiety={last.get("anxiety",0):.2f}, '
                                                f'satisfaction={last.get("satisfaction",0):.2f}.'
                                            )
                                            texts.append(summary)
                                        log(f'✅ emotional_state.json  snapshot queued ({len(hist)} history entries)')
                                    except Exception as e:
                                        log(f'⚠️  emotional_state.json  {e}')

                            # thought_stream.json → last 20 high-salience thoughts
                            if _sync_src_thoughts.value:
                                ts_path = persona_dir / 'thought_stream.json'
                                if ts_path.exists():
                                    try:
                                        ts_data  = json.loads(ts_path.read_text(encoding='utf-8'))
                                        thoughts = ts_data if isinstance(ts_data, list) else ts_data.get('thoughts', [])
                                        chunks   = []
                                        for t in thoughts[-20:]:
                                            content = (
                                                t.get('content', t.get('text', ''))
                                                if isinstance(t, dict) else str(t)
                                            )
                                            if content and len(content.strip()) > 15:
                                                chunks.append(f'Thought: {content.strip()}')
                                        texts.extend(chunks)
                                        log(f'✅ thought_stream.json  {len(chunks)} thoughts queued')
                                    except Exception as e:
                                        log(f'⚠️  thought_stream.json  {e}')

                            # ── 3. Batch add + cognify ───────────────────────────
                            if not texts:
                                log('⚠️  No text chunks collected — nothing to cognify.')
                                ui.notify('No data collected.', type='warning')
                                return

                            log(f'\n⏳ Ingesting {len(texts)} chunks into Cognee…')
                            BATCH = 40
                            n_batches = (len(texts) + BATCH - 1) // BATCH
                            try:
                                for i in range(0, len(texts), BATCH):
                                    batch    = texts[i : i + BATCH]
                                    combined = '\n\n'.join(batch)
                                    await _cog.add(combined)
                                    log(f'  batch {i // BATCH + 1}/{n_batches} added')

                                log('⏳ Running cognify()…  (this may take a minute)')
                                # Prefer force_cognify() via state.memory if available
                                # so the _cognified_once flag is properly set
                                if (
                                    hasattr(state, 'memory')
                                    and state.memory
                                    and hasattr(state.memory, 'force_cognify')
                                ):
                                    await state.memory.force_cognify()
                                else:
                                    await _cog.cognify()
                                log(f'\n✅ Sync complete — {len(texts)} chunks processed into the knowledge graph.')
                                ui.notify(f'Cognee sync complete: {len(texts)} chunks', type='positive')
                            except Exception as e:
                                log(f'❌ cognify error: {e}')
                                ui.notify(f'Sync error: {e}', type='negative')

                        # Toggle sync card visibility with backend selector
                        def _toggle_cognee_sync(e=None):
                            _cognee_sync_card.set_visibility(mem_backend.value == 'cognee')

                        _toggle_cognee_sync()
                        mem_backend.on('update:model-value', _toggle_cognee_sync)

                        with ui.row().classes('gap-2 mt-2 items-center'):
                            ui.button(
                                'Sync all → Cognee',
                                on_click=_sync_all_to_cognee,
                                icon='sync'
                            ).props('flat dense color=green')
                            ui.button(
                                'Clear log',
                                on_click=lambda: setattr(_sync_log, 'value', ''),
                                icon='delete_sweep'
                            ).props('flat dense color=grey')

                # ── Voice Lab tab ─────────────────────────────────────
                with ui.tab_panel('voice'):

                    # ── TTS provider selector ─────────────────────────
                    ui.label('TTS ENGINE').classes('section-label')
                    tts_select = ui.select(
                        ['pyttsx3', 'coqui', 'kokoro', 'edge', 'openai', 'elevenlabs'],
                        label='TTS Provider', value=config.TTS_PROVIDER
                    ).props('outlined dense').classes('w-full dark-input')

                    ui.separator().classes('my-5 opacity-10')

                    # ── STT provider selector ─────────────────────────
                    ui.label('STT ENGINE').classes('section-label')
                    with ui.grid(columns=2).classes('w-full gap-4'):
                        stt_select = ui.select(
                            ['faster_whisper', 'whisper', 'openai'],
                            label='STT Provider', value=config.STT_PROVIDER
                        ).props('outlined dense').classes('dark-input')
                        whisper_model = ui.select(
                            ['tiny', 'base', 'small', 'medium', 'large'],
                            label='Whisper Model', value=getattr(config, 'WHISPER_MODEL', 'base')
                        ).props('outlined dense').classes('dark-input')

                    ui.separator().classes('my-5 opacity-10')

                    # ── VAD / Always-listen settings ──────────────────
                    ui.label('ALWAYS-LISTEN SENSITIVITY').classes('section-label')

                    with ui.element('div').classes('info-card'):
                        with ui.column().classes('gap-4'):

                            with ui.grid(columns=2).classes('w-full gap-4'):
                                vad_aggr = ui.select(
                                    {0: '0 — Very sensitive', 1: '1 — Sensitive',
                                     2: '2 — Balanced', 3: '3 — Noise resistant ✅'},
                                    label='VAD Aggressiveness',
                                    value=getattr(config, 'VAD_AGGRESSIVENESS', 3)
                                ).props('outlined dense').classes('dark-input')

                                vad_onset = ui.select(
                                    {2: '2 chunks — 60ms (fast)', 4: '4 chunks — 120ms (balanced)',
                                     6: '6 chunks — 180ms (cautious)', 8: '8 chunks — 240ms (very cautious)'},
                                    label='Onset Confirmation',
                                    value=getattr(config, 'VAD_ONSET_CHUNKS', 4)
                                ).props('outlined dense').classes('dark-input')

                                vad_silence = ui.select(
                                    {0.6: '0.6s — Short pause', 1.0: '1.0s — Normal',
                                     1.2: '1.2s — Relaxed ✅', 1.8: '1.8s — Long pause', 2.5: '2.5s — Very long'},
                                    label='Silence Before End',
                                    value=getattr(config, 'VAD_SILENCE_DURATION', 1.2)
                                ).props('outlined dense').classes('dark-input')

                                vad_min_dur = ui.select(
                                    {0.2: '0.2s — Allow short words', 0.4: '0.4s — Normal ✅',
                                     0.6: '0.6s — Ignore short bursts', 1.0: '1.0s — Full sentences only'},
                                    label='Min Speech Duration',
                                    value=getattr(config, 'VAD_MIN_SPEECH_DURATION', 0.4)
                                ).props('outlined dense').classes('dark-input')

                            vad_gate = ui.select(
                                {2.0: '2.0× — Low gate (sensitive room)', 3.5: '3.5× — Normal gate ✅',
                                 5.0: '5.0× — High gate (noisy room)', 8.0: '8.0× — Very high gate (loud environment)'},
                                label='Energy Gate (× noise floor)',
                                value=getattr(config, 'VAD_ENERGY_GATE_FACTOR', 3.5)
                            ).props('outlined dense').classes('w-full dark-input')

                            with ui.element('div').classes('info-card mt-1').style(
                                'border-color:#1e3a5f;background:#080f1a;padding:10px 14px'
                            ):
                                ui.markdown(
                                    '**Noisy room / headset** → aggressiveness **3**, energy gate **5×–8×**, onset **6–8 chunks**\\n\\n'
                                    '**Quiet room / good mic** → aggressiveness **2**, energy gate **3.5×**, onset **4 chunks**\\n\\n'
                                    '**Recalibrate** after moving rooms using the 🔇 button in the chat header.'
                                ).style('font-size:0.78rem;color:#475569;line-height:1.6')

                    ui.separator().classes('my-5 opacity-10')

                    # ── Coqui-only section ────────────────────────────
                    coqui_section = ui.column().classes('w-full gap-4')
                    coqui_section.visible = (config.TTS_PROVIDER == 'coqui')

                    with coqui_section:
                        ui.label('VOICE CLONING').classes('section-label')

                        # Current voice status
                        ref = config.COQUI_VOICE_REFERENCE
                        ref_exists = bool(ref and os.path.exists(ref))
                        voice_status_label = ui.html(
                            f'<span class="clone-badge">🎙️ Active: {Path(ref).name}</span>'
                            if ref_exists else
                            '<span class="clone-badge clone-badge-warn">⚠️ No sample — default Coqui voice</span>'
                        )

                        ui.separator().classes('my-3 opacity-10')

                        # Language selector
                        ui.label('OUTPUT LANGUAGE').classes('section-label')
                        voice_language = ui.select(
                            {
                                'en': '🇬🇧 English',
                                'fr': '🇫🇷 French',
                                'es': '🇪🇸 Spanish',
                                'zh-cn': '🇨🇳 Chinese',
                                'de': '🇩🇪 German',
                                'it': '🇮🇹 Italian',
                                'pt': '🇵🇹 Portuguese',
                                'pl': '🇵🇱 Polish',
                                'tr': '🇹🇷 Turkish',
                                'ru': '🇷🇺 Russian',
                                'nl': '🇳🇱 Dutch',
                                'cs': '🇨🇿 Czech',
                                'ar': '🇸🇦 Arabic',
                                'ja': '🇯🇵 Japanese',
                                'ko': '🇰🇷 Korean',
                                'hi': '🇮🇳 Hindi'
                            },
                            label='TTS Language',
                            value=config.VOICE_LANGUAGE
                        ).props('outlined dense').classes('w-full dark-input')
                        
                        with ui.element('div').classes('info-card mt-2').style(
                            'border-color:#1e3a5f;background:#080f1a;padding:8px 12px'
                        ):
                            ui.markdown(
                                '💡 **Language Tips:**\n'
                                '- Your cloned voice works across all languages\n'
                                '- For best accent: record in the target language\n'
                                '- Multi-lingual voices work but may have slight accent'
                            ).style('font-size:0.75rem;color:#475569;line-height:1.5')

                        ui.separator().classes('my-4 opacity-10')

                        # Read-aloud script card
                        CLONE_SCRIPT = (
                            "The quick brown fox jumps over the lazy dog near the riverbank. "
                            "Every morning, I wake up and look out the window at the sky. "
                            "Sometimes it is bright and clear; other times, clouds gather on the horizon. "
                            "I enjoy a warm cup of coffee while thinking about the day ahead. "
                            "Technology has changed the way we communicate with each other around the world. "
                            "I believe that clear speech and a steady pace make all the difference."
                        )

                        with ui.element('div').classes('info-card info-card-amber'):
                            with ui.column().classes('gap-2'):
                                with ui.row().classes('items-center justify-between w-full'):
                                    ui.label('📖 Read This Aloud').style('font-weight:600;color:#fbbf24;font-size:0.9rem')
                                    ui.html('<span style="background:#291d08;color:#f59e0b;border-radius:20px;padding:2px 10px;font-size:0.7rem;font-weight:600">~25 seconds</span>')
                                ui.label(
                                    'Reading this passage gives Coqui the full range of your voice — '
                                    'rhythm, intonation, and natural pauses — for the best clone quality.'
                                ).style('font-size:0.78rem;color:#d97706')
                                ui.html(
                                    f'<div style="background:#0f172a;border:1px solid #92400e;border-radius:8px;'
                                    f'padding:12px;color:#e2e8f0;font-size:0.875rem;line-height:1.65;'
                                    f'cursor:text;user-select:all">{CLONE_SCRIPT}</div>'
                                )
                                ui.label(
                                    '💡 Speak naturally — not too fast. Avoid pausing mid-sentence.'
                                ).style('font-size:0.75rem;color:#92400e;font-style:italic')

                        # Recording section
                        rec_state = {
                            'recording': False, 'elapsed': 0, 'chunks': [],
                            'stream_stop': None, 'capture_thread': None, 'timer_ref': None,
                        }

                        with ui.element('div').classes('info-card info-card-purple'):
                            with ui.column().classes('gap-3'):
                                ui.label('🔴 Record Your Voice').style('font-weight:600;color:#a78bfa;font-size:0.9rem')
                                ui.label(
                                    'Click Record, speak for 15–30 seconds, then Stop. '
                                    'The recording saves automatically as your voice sample.'
                                ).style('font-size:0.78rem;color:#7c3aed')

                                with ui.row().classes('items-center gap-3'):
                                    ui.label('Duration:').style('font-size:0.82rem;color:#94a3b8;width:62px')
                                    rec_duration = ui.select(
                                        {15: '15 sec', 20: '20 sec', 30: '30 sec', 45: '45 sec'},
                                        value=20, label='Duration'
                                    ).props('dense outlined').classes('w-32 dark-input')

                                rec_progress = ui.linear_progress(value=0).props('color=purple rounded')
                                rec_progress.visible = False

                                with ui.row().classes('items-center gap-3'):
                                    rec_btn    = ui.button('⏺ Start Recording').props('color=purple outline')
                                    rec_status = ui.label('Ready').style('font-size:0.8rem;color:#64748b')

                        def _stop_mic():
                            if rec_state.get('timer_ref'):
                                rec_state['timer_ref'].cancel()
                                rec_state['timer_ref'] = None
                            if rec_state.get('stream_stop'):
                                rec_state['stream_stop'].set()

                        def _save_and_update():
                            try:
                                import numpy as np
                                import soundfile as sf

                                chunks = rec_state.get('chunks', [])
                                if not chunks:
                                    rec_status.set_text('⚠️ No audio captured — check microphone')
                                    rec_progress.visible = False
                                    return

                                audio = np.concatenate(chunks, axis=0)
                                peak  = np.max(np.abs(audio))
                                if peak > 0:
                                    audio = audio / peak * 0.95

                                save_dir  = Path('data/voices')
                                save_dir.mkdir(parents=True, exist_ok=True)
                                save_path = save_dir / 'recorded_voice.wav'
                                sf.write(str(save_path), audio, 22050)

                                duration_s = len(audio) / 22050
                                size_kb    = save_path.stat().st_size // 1024

                                config.COQUI_VOICE_REFERENCE = str(save_path)
                                save_settings(config, silent=True)

                                if state.audio:
                                    state.audio.update_voice_reference(str(save_path))

                                rec_status.set_text(f'✅ Saved ({duration_s:.1f}s, {size_kb} KB)')
                                rec_progress.visible = False
                                voice_status_label.set_content(
                                    f'<span class="clone-badge">🎙️ Active: {save_path.name}</span>'
                                )
                                ui.notify(
                                    f'✅ Recorded {duration_s:.1f}s — voice clone ready!',
                                    type='positive'
                                )
                            except Exception as ex:
                                logger.error(f"Recording save error: {ex}")
                                rec_status.set_text(f'❌ Save error: {ex}')
                                rec_progress.visible = False
                                ui.notify(f'Save error: {ex}', type='negative')

                        def _do_stop():
                            if not rec_state['recording']:
                                return
                            rec_state['recording'] = False
                            _stop_mic()
                            rec_btn.props('color=purple outline')
                            rec_btn.set_text('⏺ Start Recording')
                            rec_status.set_text('⏳ Processing…')
                            rec_progress.set_value(1.0)
                            ui.timer(0.4, _save_and_update, once=True)

                        def _tick():
                            if not rec_state['recording']:
                                return
                            rec_state['elapsed'] += 1
                            elapsed = rec_state['elapsed']
                            total   = rec_duration.value
                            rec_progress.set_value(min(elapsed / total, 1.0))
                            rec_status.set_text(f'🔴 {elapsed} / {total}s')
                            if elapsed >= total:
                                _do_stop()

                        def _on_rec_btn_click():
                            if not rec_state['recording']:
                                duration = rec_duration.value
                                rec_state.update({'recording': True, 'elapsed': 0, 'chunks': []})
                                rec_btn.props('color=red')
                                rec_btn.set_text('⏹ Stop')
                                rec_status.set_text(f'🔴 0 / {duration}s')
                                rec_progress.set_value(0)
                                rec_progress.visible = True

                                stop_evt = threading.Event()
                                rec_state['stream_stop'] = stop_evt

                                def _capture():
                                    import sounddevice as sd
                                    def _cb(indata, frames, t, status):
                                        if not stop_evt.is_set():
                                            rec_state['chunks'].append(indata.copy())
                                    try:
                                        with sd.InputStream(
                                            samplerate=22050, channels=1,
                                            dtype='float32', callback=_cb
                                        ):
                                            stop_evt.wait(timeout=duration + 2)
                                    except Exception as e:
                                        logger.error(f"Mic capture error: {e}")

                                threading.Thread(target=_capture, daemon=True).start()
                                rec_state['timer_ref'] = ui.timer(1.0, _tick)
                            else:
                                _do_stop()

                        rec_btn.on('click', _on_rec_btn_click)

                        # Upload section
                        with ui.element('div').classes('info-card info-card-blue'):
                            with ui.column().classes('gap-2'):
                                ui.label('📁 Upload Existing WAV').style('font-weight:600;color:#7dd3fc;font-size:0.9rem')
                                ui.label('Already have a recording? Upload a 15–30s WAV directly.').style('font-size:0.78rem;color:#1d4ed8')

                                async def handle_voice_upload(e):
                                    if hasattr(e, 'content') and e.content:
                                        content = e.content
                                        save_dir = Path('data/voices')
                                        save_dir.mkdir(parents=True, exist_ok=True)
                                        save_path = save_dir / e.name
                                        save_path.write_bytes(content)
                                        config.COQUI_VOICE_REFERENCE = str(save_path)
                                        save_settings(config, silent=True)
                                        if state.audio:
                                            state.audio.update_voice_reference(str(save_path))
                                        voice_status_label.set_content(
                                            f'<span class="clone-badge">🎙️ Active: {save_path.name}</span>'
                                        )
                                        ui.notify(f'✅ Uploaded: {save_path.name}', type='positive')
                                    else:
                                        ui.notify('❌ Upload failed - no file content', type='negative')

                                ui.upload(
                                    label='Choose .wav file',
                                    on_upload=handle_voice_upload,
                                    auto_upload=True
                                ).props('accept=.wav flat outlined color=blue')

                        # ── Sample voice section ─────────────────────────────
                        with ui.element('div').classes('info-card mt-0').style(
                            'border-color:#1e3a5f;background:#080f1a'
                        ):
                            with ui.column().classes('gap-2'):
                                with ui.row().classes('items-center justify-between w-full'):
                                    ui.label('🧪 Use a Sample Voice').style(
                                        'font-weight:600;color:#93c5fd;font-size:0.9rem'
                                    )
                                    ui.html(
                                        '<span style="background:#0c1e3a;color:#60a5fa;border-radius:20px;'
                                        'padding:2px 10px;font-size:0.7rem;font-weight:600">Pipeline test</span>'
                                    )

                                ui.label(
                                    'No mic? Distorted headset? Generate a clean synthetic voice sample '
                                    'to verify the Coqui pipeline works before recording your own voice. '
                                    'Uses your system TTS (Windows SAPI / pyttsx3) — no internet needed.'
                                ).style('font-size:0.78rem;color:#3b82f6;line-height:1.5')

                                with ui.grid(columns=2).classes('w-full gap-3 mt-1'):
                                    sample_voice_select = ui.select(
                                        {
                                            'zira':   '🔵 Zira (Female EN)',
                                            'david':  '🟢 David (Male EN)',
                                            'hazel':  '🟣 Hazel (Female EN-GB)',
                                            'mark':   '🟠 Mark (Male EN)',
                                        },
                                        value='zira',
                                        label='Voice'
                                    ).props('dense outlined').classes('dark-input')

                                    sample_duration_select = ui.select(
                                        {15: '15 sec', 20: '20 sec', 30: '30 sec'},
                                        value=20,
                                        label='Duration'
                                    ).props('dense outlined').classes('dark-input')

                                sample_status = ui.label('').style('font-size:0.78rem;color:#64748b')

                                async def generate_sample_voice():
                                    sample_status.set_text('⏳ Generating sample voice…')
                                    loop = asyncio.get_running_loop()

                                    target_voice = sample_voice_select.value
                                    duration_s   = sample_duration_select.value

                                    # Build a script long enough for the chosen duration
                                    SCRIPT_SENTENCES = [
                                        "The quick brown fox jumps over the lazy dog near the riverbank.",
                                        "Every morning, I wake up and look out the window at the sky.",
                                        "Sometimes it is bright and clear; other times, clouds gather on the horizon.",
                                        "I enjoy a warm cup of coffee while thinking about the day ahead.",
                                        "Technology has changed the way we communicate with each other around the world.",
                                        "I believe that clear speech and a steady pace make all the difference.",
                                        "The rain in Spain falls mainly on the plain, or so they say.",
                                        "A journey of a thousand miles begins with a single step forward.",
                                        "She sells seashells by the seashore every single morning.",
                                        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
                                    ]
                                    # Repeat until we have roughly enough text for the target duration
                                    # ~130 words per minute speaking rate → ~2.2 words/sec
                                    words_needed = int(duration_s * 2.2)
                                    text_parts, word_count = [], 0
                                    for sentence in SCRIPT_SENTENCES * 3:
                                        text_parts.append(sentence)
                                        word_count += len(sentence.split())
                                        if word_count >= words_needed:
                                            break
                                    script_text = ' '.join(text_parts)

                                    def _generate():
                                        try:
                                            import pyttsx3
                                            import soundfile as sf
                                            import numpy as np

                                            engine = pyttsx3.init()
                                            voices = engine.getProperty('voices') or []

                                            # Find the requested voice by name keyword
                                            chosen = None
                                            for v in voices:
                                                if target_voice.lower() in v.name.lower():
                                                    chosen = v.id
                                                    break
                                            # Fallback: any English female/male voice
                                            if not chosen:
                                                fallbacks = (
                                                    ['zira', 'female', 'hazel', 'susan']
                                                    if target_voice in ('zira', 'hazel')
                                                    else ['david', 'male', 'mark', 'james']
                                                )
                                                for v in voices:
                                                    if any(k in v.name.lower() for k in fallbacks):
                                                        chosen = v.id
                                                        break
                                            if not chosen and voices:
                                                chosen = voices[0].id

                                            engine.setProperty('rate',   155)
                                            engine.setProperty('volume', 1.0)
                                            if chosen:
                                                engine.setProperty('voice', chosen)

                                            save_dir = Path('data/voices')
                                            save_dir.mkdir(parents=True, exist_ok=True)
                                            out_path = save_dir / f'sample_{target_voice}.wav'

                                            engine.save_to_file(script_text, str(out_path))
                                            engine.runAndWait()
                                            engine.stop()

                                            if not out_path.exists() or out_path.stat().st_size < 2000:
                                                return None, "pyttsx3 produced an empty file — check system TTS"

                                            # Re-sample to 22050 Hz mono (XTTS requirement)
                                            data, sr = sf.read(str(out_path))
                                            if data.ndim > 1:
                                                data = data.mean(axis=1)
                                            if sr != 22050:
                                                try:
                                                    import librosa
                                                    data = librosa.resample(data, orig_sr=sr, target_sr=22050)
                                                    sr   = 22050
                                                except Exception:
                                                    pass  # keep original sr — still usable
                                            # Normalise
                                            peak = np.max(np.abs(data))
                                            if peak > 0:
                                                data = data / peak * 0.92
                                            sf.write(str(out_path), data, sr)

                                            actual_duration = len(data) / sr
                                            size_kb = out_path.stat().st_size // 1024
                                            return str(out_path), (
                                                f"✅ {actual_duration:.1f}s · {size_kb} KB · "
                                                f"{sr} Hz mono"
                                            )
                                        except Exception as ex:
                                            logger.error(f"Sample voice generation error: {ex}")
                                            return None, str(ex)

                                    out_path, msg = await loop.run_in_executor(None, _generate)

                                    if out_path:
                                        config.COQUI_VOICE_REFERENCE = out_path
                                        save_settings(config, silent=True)
                                        if state.audio:
                                            state.audio.update_voice_reference(out_path)
                                        voice_status_label.set_content(
                                            f'<span class="clone-badge">🧪 Sample: '
                                            f'{Path(out_path).name}</span>'
                                        )
                                        sample_status.set_text(f'{msg} — set as active reference')
                                        ui.notify(
                                            '✅ Sample voice generated and set as reference. '
                                            'Use "Test Now" below to verify the pipeline.',
                                            type='positive'
                                        )
                                    else:
                                        sample_status.set_text(f'❌ {msg}')
                                        ui.notify(f'Sample generation failed: {msg}', type='negative')

                                ui.button(
                                    '🎲 Generate Sample Voice',
                                    on_click=generate_sample_voice
                                ).props('color=blue outline')

                        # Test voice section
                        with ui.element('div').classes('info-card info-card-green'):
                            with ui.column().classes('gap-2'):
                                ui.label('🔊 Test Cloned Voice').style('font-weight:600;color:#4ade80;font-size:0.9rem')
                                ui.label('Generate a quick preview with the current voice reference.').style('font-size:0.78rem;color:#16a34a')

                                test_text = ui.input(
                                    'Test phrase',
                                    value='Hello, this is a test of my cloned voice.'
                                ).props('outlined dense').classes('w-full dark-input')

                                test_status = ui.label('').style('font-size:0.78rem;color:#64748b')

                                async def test_voice():
                                    provider   = tts_select.value
                                    voice_path = config.COQUI_VOICE_REFERENCE

                                    if provider == 'coqui':
                                        if not voice_path or not os.path.exists(voice_path):
                                            ui.notify('No voice sample set — record or upload one first', type='warning')
                                            return

                                    test_status.set_text('⏳ Generating…')
                                    loop = asyncio.get_running_loop()

                                    def _generate():
                                        try:
                                            # Reuse state.audio if already loaded with the right provider
                                            # — avoids reloading the 7s XTTS model from disk
                                            mgr = None
                                            if (state.audio and state.audio.tts_engine and
                                                    state.audio.tts_engine.get('type') == provider):
                                                mgr = state.audio
                                                # Hot-swap voice ref in case it changed
                                                if provider == 'coqui':
                                                    mgr.tts_engine['voice_ref'] = voice_path
                                                logger.info("Test: reusing existing AudioManager (no reload)")
                                            else:
                                                from managers.audio_manager import AudioManager
                                                orig_provider = config.TTS_PROVIDER
                                                config.TTS_PROVIDER = provider
                                                mgr = AudioManager()
                                                config.TTS_PROVIDER = orig_provider
                                                if provider == 'coqui' and mgr.tts_engine:
                                                    mgr.tts_engine['voice_ref'] = voice_path
                                            return mgr.text_to_speech(test_text.value)
                                        except Exception as ex:
                                            logger.error(f"Test voice error: {ex}")
                                            return None

                                    path = await loop.run_in_executor(None, _generate)

                                    if path:
                                        try:
                                            import sounddevice as sd
                                            import soundfile as sf
                                            data, sr = sf.read(path)
                                            test_status.set_text('🔊 Playing…')
                                            await loop.run_in_executor(None, lambda: (sd.play(data, sr), sd.wait()))
                                            os.remove(path)
                                            test_status.set_text('✅ Playback complete')
                                        except Exception as ex:
                                            test_status.set_text(f'❌ {ex}')
                                    else:
                                        test_status.set_text('❌ TTS generation failed — check console')
                                        ui.notify('TTS failed — check provider settings and console', type='negative')

                                ui.button('🔊 Test Now', on_click=test_voice).props('color=green outline')

                    # ElevenLabs section (hidden unless selected)
                    elevenlabs_section = ui.column().classes('w-full gap-4')
                    elevenlabs_section.visible = (config.TTS_PROVIDER == 'elevenlabs')

                    with elevenlabs_section:
                        ui.separator().classes('my-4 opacity-10')
                        ui.label('ELEVENLABS API').classes('section-label')
                        elevenlabs_key = ui.input(
                            'ElevenLabs API Key', value=config.ELEVENLABS_API_KEY, password=True
                        ).props('outlined dense').classes('w-full dark-input')

                    def _on_tts_change(e):
                        v = tts_select.value
                        coqui_section.visible      = (v == 'coqui')
                        elevenlabs_section.visible = (v == 'elevenlabs')

                    tts_select.on('update:model-value', _on_tts_change)

                # ── Vision tab ────────────────────────────────────────
                with ui.tab_panel('vision'):
                    ui.label('VISION SETTINGS').classes('section-label')
                    
                    # Camera settings
                    with ui.element('div').classes('info-card'):
                        ui.label('Camera').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        
                        with ui.column().classes('gap-4 mt-3'):
                            camera_enabled = ui.switch(
                                'Enable Camera on Startup',
                                value=getattr(config, 'CAMERA_AUTOSTART', False)
                            )
                            
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                camera_id = ui.select(
                                    {0: 'Camera 0 (Built-in)', 1: 'Camera 1 (External)'},
                                    label='Camera ID',
                                    value=getattr(config, 'CAMERA_ID', 0)
                                ).props('outlined dense').classes('dark-input')
                                
                                camera_fps = ui.select(
                                    {5: '5 fps (low)', 10: '10 fps (balanced)', 15: '15 fps (high)'},
                                    label='Frame Rate',
                                    value=getattr(config, 'CAMERA_FPS', 5)
                                ).props('outlined dense').classes('dark-input')
                            
                            camera_resolution = ui.select(
                                {'640x480': '640x480 (SD)', '1280x720': '1280x720 (HD)'},
                                label='Resolution',
                                value=getattr(config, 'CAMERA_RESOLUTION', '640x480')
                            ).props('outlined dense').classes('w-full dark-input')
                            
                            ui.label('Note: Lower FPS improves stability').classes('text-xs text-amber-400')
                    
                    # Vision Mode settings
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('Vision Integration').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        
                        with ui.column().classes('gap-4 mt-3'):
                            vision_mode = ui.select(
                                {
                                    'keyword': '🔤 Keyword-Triggered (Default)',
                                    'always': '👁️ Always Show Vision',
                                    'context': '🧠 Context Mode (Background)'
                                },
                                label='Vision Mode',
                                value=getattr(config, 'VISION_MODE', 'keyword')
                            ).props('outlined dense').classes('w-full dark-input')
                            
                            ui.markdown(
                                '**Vision Modes:**\n'
                                '- **Keyword**: Vision only triggers with phrases like "what do you see", "look", "camera"\n'
                                '- **Always**: Every message analyzes camera and shows vision response\n'
                                '- **Context**: Camera analyzed in background, LLM aware but vision not shown separately'
                            ).style('font-size:0.78rem;color:#94a3b8')
                    
                    # Face recognition settings
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('Face Recognition').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        
                        with ui.column().classes('gap-4 mt-3'):
                            def _on_face_toggle(e):
                                if state.vision:
                                    state.vision.face_detection_enabled = e.value
                                    logger.info(f"Face detection {'enabled' if e.value else 'disabled'} via settings")

                            face_enabled = ui.switch(
                                'Enable Face Detection',
                                value=state.vision.face_detection_enabled if state.vision else (True if FACE_RECOGNITION_AVAILABLE else False),
                                on_change=_on_face_toggle
                            )
                            
                            if not FACE_RECOGNITION_AVAILABLE:
                                ui.label('⚠️ face_recognition not installed').classes('text-amber-400 text-sm')
                            
                            if state.vision and hasattr(state.vision, 'face_encodings'):
                                ui.label(f'Known Faces: {len(state.vision.face_encodings)}').classes('text-sm text-green-400')
                    
                    # Vision memory
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('Vision Memory').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        
                        with ui.column().classes('gap-4 mt-3'):
                            if state.vision and hasattr(state.vision, 'vision_meta'):
                                ui.label(f'Stored Memories: {len(state.vision.vision_meta)}').classes('text-sm')
                            
                            async def clear_vision_memory():
                                if state.vision:
                                    if hasattr(state.vision, 'vision_meta'):
                                        state.vision.vision_meta = []
                                    if hasattr(state.vision, 'vision_index') and state.vision.vision_index:
                                        import faiss
                                        state.vision.vision_index = faiss.IndexFlatL2(state.vision.embedding_dim)
                                    ui.notify('Vision memory cleared', type='info')
                                    clear_memory_btn.visible = False
                            
                            clear_memory_btn = ui.button(
                                'Clear Vision Memory',
                                on_click=clear_vision_memory
                            ).props('flat color=red')
                    
                    # LAVA model settings
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('Vision Model (LAVA)').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        
                        with ui.column().classes('gap-4 mt-3'):
                            lava_model = ui.input(
                                'Model Name',
                                value=getattr(config, 'LAVA_MODEL', 'llava:latest')
                            ).props('outlined dense').classes('w-full dark-input')
                            
                            ui.markdown(
                                '**Available models:**\n'
                                '- `llava:latest` (recommended)\n'
                                '- `bakllava:latest`\n'
                                '- `llava-llama3:latest`'
                            ).style('font-size:0.78rem;color:#94a3b8')

                    # Vision LLM routing
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('Vision LLM Routing').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        ui.label(
                            'How the camera image reaches the language model.'
                        ).classes('text-slate-400 text-sm mt-1 mb-3')

                        vision_llm_mode = ui.select(
                            {
                                'separate': '🔀 Separate  — dedicated LLaVA model describes, text sent to main LLM',
                                'direct':   '⚡ Direct    — image sent straight to main model (must be multimodal)',
                            },
                            value=getattr(config, 'VISION_LLM_MODE', 'separate'),
                            label='Vision routing'
                        ).props('outlined dense').classes('w-full dark-input')

                        with ui.element('div').classes('mt-2').style('font-size:0.78rem;color:#94a3b8;line-height:1.6'):
                            ui.markdown(
                                '**Separate** — LLaVA (or any Ollama vision model) analyses the frame '
                                'and produces a text description. That text is injected into the main '
                                'model\'s context. Use this if your main model is text-only (e.g. Mistral Small).\n\n'
                                '**Direct** — the image is sent in OpenAI `image_url` format directly '
                                'to your main model. Requires a multimodal model in LM Studio '
                                '(LLaVA, Qwen-VL, Pixtral, InternVL, etc.).'
                            )

                    # Ambient vision
                    with ui.element('div').classes('info-card mt-4'):
                        ui.label('Ambient Vision').style('font-weight:600;color:#c084fc;font-size:0.9rem')
                        ui.label(
                            'Periodic silent scene analysis fed to the cognitive workspace. '
                            'Never runs during active conversation.'
                        ).classes('text-slate-400 text-sm mt-1 mb-3')

                        ambient_vision_interval = ui.select(
                            {
                                0:   '🚫 Disabled',
                                60:  '⚡ 60s  — frequent (higher CPU)',
                                90:  '✅ 90s  — recommended',
                                120: '🐢 2 min — relaxed',
                                300: '💤 5 min — minimal',
                            },
                            value=getattr(config, 'AMBIENT_VISION_INTERVAL', 90),
                            label='Analysis interval'
                        ).props('outlined dense').classes('w-full dark-input')

                        with ui.element('div').classes('mt-2').style('font-size:0.78rem;color:#94a3b8;line-height:1.6'):
                            ui.markdown(
                                'Each ambient tick fires a brief LLM call (1-2 sentences) '
                                'and broadcasts the result to the Global Workspace at low priority (0.28). '
                                'Detected known faces are broadcast at higher priority (0.55).\n\n'
                                'The orchestrator automatically skips ambient analysis if the user '
                                'interacted within the last 30 seconds — LM Studio is never overloaded.'
                            )



                # ── Status tab ────────────────────────────────────────
                with ui.tab_panel('status'):
                    ui.label('SYSTEM STATUS').classes('section-label')

                    # Safe check for LLM availability
                    llm_ok = False
                    try:
                        if state.llm:
                            # Try different method names that might exist
                            if hasattr(state.llm, 'is_available'):
                                llm_ok = state.llm.is_available()
                            elif hasattr(state.llm, 'is_ready'):
                                llm_ok = state.llm.is_ready()
                            elif hasattr(state.llm, 'model'):
                                llm_ok = True
                            else:
                                llm_ok = True  # Assume it's working if it exists
                    except:
                        llm_ok = False

                    mem_ok = bool(state.memory)
                    aud_ok = bool(state.audio and state.audio.tts_engine)
                    vad_ok = bool(state.conv_audio and state.conv_audio.vad_available)
                    stt_ok = bool(state.audio and state.audio.stt_engine)
                    clone_ok = (
                        aud_ok and state.audio and state.audio.tts_engine and 
                        state.audio.tts_engine.get('type') == 'coqui' and
                        bool(state.audio.tts_engine.get('voice_ref'))
                    )
                    vision_ok = bool(state.vision)
                    camera_ok = bool(state.vision and state.vision.camera_active)
                    faces_ok = bool(state.vision and hasattr(state.vision, 'face_encodings') and len(state.vision.face_encodings) > 0)

                    rows = [
                        ('LLM',          llm_ok,   f"{config.LLM_PROVIDER} / {config.LLM_MODEL}" if llm_ok else "Not reachable"),
                        ('Memory',       mem_ok,   config.MEMORY_BACKEND if mem_ok else "Failed"),
                        ('TTS',          aud_ok,   config.TTS_PROVIDER if aud_ok else "Failed (lazy-init: switch to voice mode)"),
                        ('STT',          stt_ok,   config.STT_PROVIDER if stt_ok else "Failed (lazy-init: switch to voice mode)"),
                        ('VAD',          vad_ok,   "Ready" if vad_ok else "Install webrtcvad"),
                        ('Voice Clone',  clone_ok, "Active ✓" if clone_ok else "No sample or not Coqui"),
                        ('Vision',       vision_ok, "Initialized ✓" if vision_ok else "Failed"),
                        ('Camera',       camera_ok, "Active" if camera_ok else "Off"),
                        ('Face Recognition', faces_ok, f"{len(state.vision.face_encodings) if faces_ok else 0} known faces" if faces_ok else "No faces"),
                    ]

                    for name, ok, detail in rows:
                        icon = '✅' if ok else '❌'
                        cls = 'status-ok' if ok else 'status-fail'
                        ui.html(
                            f'<div class="status-row">'
                            f'  <span style="font-weight:500;color:#94a3b8;width:140px;font-size:0.85rem">{name}</span>'
                            f'  <span class="{cls}" style="font-size:0.82rem">{icon} {detail}</span>'
                            f'</div>'
                        )

                    # Add connection info
                    ui.label('CONNECTION STATUS').classes('section-label mt-4')
                    ui.html(
                        f'<div class="status-row">'
                        f'  <span style="font-weight:500;color:#94a3b8">WebSocket</span>'
                        f'  <span class="status-ok">✅ Connected</span>'
                        f'</div>'
                    )
                    ui.html(
                        f'<div class="status-row">'
                        f'  <span style="font-weight:500;color:#94a3b8">Reconnect attempts</span>'
                        f'  <span class="status-ok">{connection_monitor.reconnect_attempts}</span>'
                        f'</div>'
                    )
                    ui.html(
                        f'<div class="status-row">'
                        f'  <span style="font-weight:500;color:#94a3b8">System uptime</span>'
                        f'  <span class="status-ok">{int(state.get_uptime() // 60)} minutes</span>'
                        f'</div>'
                    )
                    ui.html(
                        f'<div class="status-row">'
                        f'  <span style="font-weight:500;color:#94a3b8">Platform</span>'
                        f'  <span class="status-ok">{platform.system()} {platform.release()}</span>'
                        f'</div>'
                    )

                    # Show current config file info
                    ui.label('CONFIGURATION FILE').classes('section-label mt-4')
                    config_path = Path('config.json')
                    if config_path.exists():
                        size_kb = config_path.stat().st_size // 1024
                        modified = datetime.fromtimestamp(config_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        ui.html(
                            f'<div class="status-row">'
                            f'  <span style="font-weight:500;color:#94a3b8">Location</span>'
                            f'  <span class="status-ok">{config_path.absolute()}</span>'
                            f'</div>'
                        )
                        ui.html(
                            f'<div class="status-row">'
                            f'  <span style="font-weight:500;color:#94a3b8">Size</span>'
                            f'  <span class="status-ok">{size_kb} KB</span>'
                            f'</div>'
                        )
                        ui.html(
                            f'<div class="status-row">'
                            f'  <span style="font-weight:500;color:#94a3b8">Modified</span>'
                            f'  <span class="status-ok">{modified}</span>'
                            f'</div>'
                        )
                    else:
                        ui.html('<div class="status-row"><span style="color:#f87171">⚠️ No config.json found</span></div>')

                    ui.html('<div style="height:8px"></div>')
                    ui.label(
                        '* TTS/STT/VAD/Clone status only available after switching to a voice mode '
                        'at least once (lazy initialisation).'
                    ).style('font-size:0.72rem;color:#475569;font-style:italic')

                    ui.button(
                        '🔄 Refresh Status', on_click=lambda: ui.navigate.to('/settings')
                    ).props('flat color=indigo').classes('mt-4')

                # ── Lumina tab ──────────────────────────────────────────
                with ui.tab_panel('lumina'):
                    ui.label('LUMINA COGNITIVE ENGINE').classes('section-label')
                    ui.markdown(
                        "Lumina is the psychological core of this agent — she builds her own "
                        "system prompt using emotional state, relational memory, personality "
                        "evolution, and life-stage cognition. Use **✨ Lumina Mind** in the "
                        "petal menu for the full live dashboard."
                    ).classes('text-sm opacity-70 mb-4')

                    # ── Live status ──────────────────────────────────────
                    if state.persona and state.persona.is_ready:
                        try:
                            _status = state.persona.get_system_status()
                            # get_system_status() merges float emotion values (curiosity,
                            # warmth…) with string metadata keys (description, trend,
                            # overall_valence, overall_arousal) into ONE dict via **emo_vals.
                            # Filter to numeric-only before any .get() / float() calls.
                            _NON_EMO = {'description','trend','overall_valence',
                                        'overall_arousal','state_description'}
                            _emo_raw = _status.get('emotional_state', {})
                            _emo_num = {k: float(v) for k, v in _emo_raw.items()
                                        if k not in _NON_EMO and isinstance(v, (int, float))}
                            _stage   = _status.get('life_stage', 'unknown')
                            # traits are in status['personality'], NOT status['evolution']['traits']
                            _traits  = {k: v for k, v in _status.get('personality', {}).items()
                                        if isinstance(v, (int, float))}

                            with ui.element('div').classes('info-card mb-4'):
                                ui.label('CURRENT STATE').classes('section-label')
                                with ui.grid(columns=2).classes('w-full gap-3 mt-2'):
                                    with ui.element('div').classes('stat-card'):
                                        ui.label('Life Stage').classes('stat-label')
                                        ui.label(_stage.replace('_',' ').title()).classes('stat-value')
                                    with ui.element('div').classes('stat-card'):
                                        ui.label('Mood').classes('stat-label')
                                        if _emo_num:
                                            dominant = max(_emo_num, key=lambda k: _emo_num[k])
                                            ui.label(dominant.replace('_',' ').title()).classes('stat-value')
                                        elif _emo_raw.get('description'):
                                            ui.label(_emo_raw['description'][:30]).classes('stat-value text-xs')
                                        else:
                                            ui.label('—').classes('stat-value')

                            if _traits:
                                with ui.element('div').classes('info-card mb-4'):
                                    ui.label('PERSONALITY TRAITS').classes('section-label')
                                    for trait, val in sorted(_traits.items(), key=lambda x: -x[1])[:6]:
                                        try:
                                            pct = int(float(val) * 100)
                                        except (TypeError, ValueError):
                                            continue
                                        ui.html(
                                            f'<div class="status-row" style="margin-bottom:6px">'
                                            f'  <span style="color:#94a3b8;font-size:0.8rem">{trait.replace("_"," ").title()}</span>'
                                            f'  <div style="flex:1;margin:0 10px;background:#1e293b;border-radius:4px;height:6px">'
                                            f'    <div style="width:{pct}%;background:#8b5cf6;height:6px;border-radius:4px"></div>'
                                            f'  </div>'
                                            f'  <span style="color:#c4b5fd;font-size:0.8rem">{pct}%</span>'
                                            f'</div>'
                                        )
                                        
                        except Exception as _e:
                            ui.label(f'Could not load Lumina status: {_e}').classes('text-red-400 text-sm')
                    else:
                        with ui.element('div').classes('info-card'):
                            ui.html('<span class="text-yellow-400">⚠️ Lumina not yet initialised. '
                                    'Start a conversation to activate the cognitive engine.</span>')

                    ui.separator().classes('my-4')

                    # ── Birth date (v117: age/life_stage computed from this) ──
                    ui.label('BIRTH DATE').classes('section-label')
                    ui.markdown(
                        "Age and life stage are computed live from elapsed real time "
                        "since this date — not from usage. Set once automatically on "
                        "first boot (or migrated from a pre-v117 install's accumulated "
                        "age); editable here if you want to adjust it directly."
                    ).classes('text-sm opacity-70 mb-2')
                    _existing_birth = get_birth_date()
                    with ui.grid(columns=2).classes('w-full gap-4 items-end'):
                        birth_date_input = ui.input(
                            'Birth date/time (ISO format)',
                            value=_existing_birth.isoformat() if _existing_birth else '',
                            placeholder='e.g. 2026-08-01T00:00:00'
                        ).props('outlined dense').classes('dark-input')
                        with ui.row().classes('gap-2'):
                            def _save_birth_date():
                                raw = (birth_date_input.value or '').strip()
                                try:
                                    parsed = datetime.fromisoformat(raw)
                                except ValueError:
                                    ui.notify('Invalid ISO datetime — example: 2026-08-01T00:00:00',
                                               type='negative', timeout=4000)
                                    return
                                set_birth_date(parsed)
                                birth_date_input.value = parsed.isoformat()
                                ui.notify('Birth date saved. Restart to apply.',
                                          type='positive', timeout=3000)
                            def _reset_birth_date_now():
                                now = datetime.now()
                                set_birth_date(now)
                                birth_date_input.value = now.isoformat()
                                ui.notify('Birth date reset to now — age restarts from 0.',
                                          type='warning', timeout=4000)
                            ui.button('Save', on_click=_save_birth_date).props('color=purple dense')
                            ui.button('Reset to now', on_click=_reset_birth_date_now).props('flat dense')

                    # ── Trigger actions ──────────────────────────────────
                    ui.label('COGNITIVE ACTIONS').classes('section-label')
                    with ui.row().classes('gap-3 mt-2'):
                        async def _dream():
                            if state.persona:
                                ui.notify('Running dream cycle…', type='info', timeout=2000)
                                result = await asyncio.to_thread(state.persona.trigger_dream)
                                insights = result if isinstance(result, list) else []
                                with ui.dialog() as _dr, ui.card().classes('p-5 min-w-96 max-w-xl'):
                                    ui.label('🌙 Dream Cycle Complete').classes('font-semibold text-lg text-purple-300 mb-3')
                                    if insights:
                                        ui.label(f'{len(insights)} insight{"s" if len(insights) != 1 else ""} generated:').classes('text-slate-400 text-sm mb-2')
                                        for ins in insights:
                                            ui.label(f'• {ins}').classes('text-sm text-slate-200 mb-1')
                                    else:
                                        ui.label('Dream cycle ran — no new insights this time.').classes('text-slate-400 text-sm')
                                    ui.button('Close', on_click=_dr.close).classes('mt-4').props('color=purple')
                                _dr.open()
                        async def _learn():
                            if state.persona:
                                ui.notify('Running learning cycle…', type='info', timeout=2000)
                                result = await asyncio.to_thread(state.persona.trigger_learning)
                                with ui.dialog() as _lr, ui.card().classes('p-5 min-w-96 max-w-xl'):
                                    ui.label('📚 Learning Cycle Complete').classes('font-semibold text-lg text-indigo-300 mb-3')
                                    if result and isinstance(result, dict):
                                        assessment = result.get('assessment', '')
                                        insights   = result.get('insights', [])
                                        if assessment:
                                            ui.label(assessment).classes('text-sm text-slate-200 mb-3')
                                        for ins in insights:
                                            ui.label(f'• {ins}').classes('text-sm text-slate-200 mb-1')
                                    else:
                                        ui.label('Learning cycle ran successfully.').classes('text-slate-400 text-sm')
                                    ui.button('Close', on_click=_lr.close).classes('mt-4').props('color=indigo')
                                _lr.open()
                        async def _life_event_settings():
                            with ui.dialog() as _d, ui.card().classes('p-5 min-w-80'):
                                ui.label('🌱 Simulate Life Event').classes('font-semibold text-lg mb-3')
                                ui.label('Optionally hint the type of event (creative, social, challenge…)').classes('text-slate-400 text-sm mb-2')
                                _inp = ui.input(placeholder='Event type (optional)').classes('w-full')
                                with ui.row().classes('gap-2 mt-4'):
                                    ui.button('Cancel', on_click=_d.close).props('flat')
                                    async def _go_settings():
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
                                        else:
                                            ui.notify('Life event could not be generated — Lumina may need more context first.', type='warning', timeout=6000)
                                    ui.button('Simulate', on_click=_go_settings).props('color=purple')
                            _d.open()

                        ui.button('🌙 Run Dream Cycle', on_click=_dream).props('flat color=indigo')
                        ui.button('📚 Run Learning Cycle', on_click=_learn).props('flat color=purple')
                        ui.button('🌱 Life Event', on_click=_life_event_settings).props('flat color=green')

                    ui.separator().classes('my-4')

                    # ── Data paths ───────────────────────────────────────
                    ui.label('DATA & PERSISTENCE').classes('section-label')
                    with ui.element('div').classes('info-card mt-2'):
                        for label, path in [
                            ('Persona DB',       'data/persona/ai_system.db'),
                            ('Emotional State',  'data/persona/emotional_state.json'),
                            ('Relational Memory','data/persona/relational_memory.json'),
                            ('Goals',            'data/persona/goals.json'),
                            ('Self Concept',     'data/persona/self_concept.json'),
                        ]:
                            exists = Path(path).exists()
                            color  = '#22c55e' if exists else '#f87171'
                            icon   = '✅' if exists else '❌'
                            ui.html(
                                f'<div class="status-row">'
                                f'  <span style="color:#94a3b8;font-size:0.8rem">{label}</span>'
                                f'  <span style="color:{color};font-size:0.8rem">{icon} {path}</span>'
                                f'</div>'
                            )

                    ui.button(
                        '✨ Open Lumina Dashboard', on_click=lambda: ui.navigate.to('/lumina')
                    ).props('color=deep-purple').classes('mt-4')


                # ── Tools tab ─────────────────────────────────────────
                with ui.tab_panel('tools'):
                    ui.label('🔧 TOOLS').classes('section-label')
                    ui.html('<div style="color:#64748b;font-size:12px;margin-bottom:16px">'
                            'External tools, cognitive observatory, and diagnostic utilities. '
                            'These are independent of core LLM/voice/memory settings.</div>')

                    # ── Cognitive Observatory ────────────────────────────
                    with ui.expansion('🔭 Cognitive Observatory', icon='psychology').classes('w-full mb-3'):
                        ui.label('OBSERVATORY SETTINGS').classes('section-label')
                        obs_enabled = ui.switch(
                            'Enable Observatory metrics (CCS, RDI, GEI, IDX)',
                            value=getattr(config, 'OBSERVATORY_ENABLED', True)
                        ).classes('mb-2')
                        with ui.row().classes('gap-3 w-full flex-wrap'):
                            obs_baseline = ui.number(
                                'Baseline calibration samples',
                                value=getattr(config, 'OBSERVATORY_BASELINE_SAMPLES', 20),
                                min=5, max=100, step=5
                            ).classes('flex-1')
                            obs_emg_threshold = ui.number(
                                'Emergence alert threshold (0–1)',
                                value=getattr(config, 'OBSERVATORY_EMERGENCE_THRESHOLD', 0.5),
                                min=0.1, max=1.0, step=0.05, format='%.2f'
                            ).classes('flex-1')
                        ui.label('Metrics reference:').classes('text-slate-400 text-xs mt-2')
                        ui.html('''
                            <div style="font-size:11px;color:#64748b;margin-top:4px">
                            <b style="color:#a5b4fc">CCS</b> Coherence (0.6–0.9 healthy) ·
                            <b style="color:#a5b4fc">RDI</b> Reflection depth (0.3–0.7) ·
                            <b style="color:#a5b4fc">GEI</b> Goal emergence (0.05–0.25) ·
                            <b style="color:#a5b4fc">IDX</b> Identity drift (below 0.25) ·
                            <b style="color:#a5b4fc">STR</b> Strangeness (below 0.35)
                            </div>''')
                        async def _save_obs():
                            if hasattr(config, 'OBSERVATORY_ENABLED'):
                                config.OBSERVATORY_ENABLED = obs_enabled.value
                            if hasattr(config, 'OBSERVATORY_BASELINE_SAMPLES'):
                                config.OBSERVATORY_BASELINE_SAMPLES = int(obs_baseline.value)
                            if hasattr(config, 'OBSERVATORY_EMERGENCE_THRESHOLD'):
                                config.OBSERVATORY_EMERGENCE_THRESHOLD = float(obs_emg_threshold.value)
                            save_settings(config, silent=True)
                            ui.notify('Observatory settings saved', type='positive')
                        ui.button('Save Observatory Settings', on_click=_save_obs).props('flat dense color=indigo').classes('mt-2')

                    # ── Brain Visualizer ─────────────────────────────────
                    with ui.expansion('🧠 Brain Visualizer', icon='biotech').classes('w-full mb-3'):
                        ui.label('HOLOGRAPHIC BRAIN').classes('section-label')
                        ui.html('''
                            <div style="font-size:12px;color:#64748b;margin-bottom:10px">
                            3D holographic render of Lumina's live cognitive state mapped onto
                            7 functional brain networks.  Requires <code>PyOpenGL</code> and
                            <code>glfw</code> — install with the command below if not yet present.
                            </div>
                            <div style="background:#0f172a;border-radius:6px;padding:8px 12px;
                                        font-size:11px;color:#94a3b8;font-family:monospace;margin-bottom:10px">
                            pip install PyOpenGL PyOpenGL_accelerate glfw
                            </div>
                            <div style="font-size:11px;color:#475569;margin-bottom:8px">
                            <b style="color:#60a5fa">ECN</b> Executive Control &nbsp;·&nbsp;
                            <b style="color:#a78bfa">DMN</b> Default Mode &nbsp;·&nbsp;
                            <b style="color:#fb923c">SN</b> Salience &nbsp;·&nbsp;
                            <b style="color:#f472b6">SCN</b> Social Cognition<br>
                            <b style="color:#fbbf24">RMS</b> Reward/Motivation &nbsp;·&nbsp;
                            <b style="color:#4ade80">MS</b> Memory System &nbsp;·&nbsp;
                            <b style="color:#22d3ee">SIS</b> Sensory Integration
                            </div>
                        ''')
                        _brain_status = ui.html('<div style="font-size:12px;color:#64748b">Not started.</div>')

                        def _launch_brain_viz():
                            try:
                                from brain_visualizer.lumina_brain_bridge import LuminaBrainBridge
                                from core.state import state as _lumina_state
                                # Same path used everywhere in app.py
                                _org = None
                                pb   = getattr(_lumina_state, 'persona', None)
                                if pb:
                                    _org = getattr(pb, '_organism', None)
                                if not _org:
                                    _brain_status.set_content(
                                        '<div style="color:#ef4444;font-size:12px">'
                                        '❌ Organism not found — start Lumina first.</div>'
                                    )
                                    return
                                bridge = LuminaBrainBridge(_org)
                                # TEMPORARY DIAGNOSTIC — remove after debugging
                                try:
                                    from brain_visualizer.debug_bridge import debug_organism
                                    debug_organism(_org)
                                except Exception as _de:
                                    print(f"[debug_bridge] {_de}")
                                bridge.start()
                                _brain_status.set_content(
                                    '<div style="color:#22c55e;font-size:12px">'
                                    '✅ Brain visualizer running — SPACE to pause, ESC to close.</div>'
                                )
                            except ImportError as _ie:
                                _brain_status.set_content(
                                    f'<div style="color:#ef4444;font-size:12px">'
                                    f'❌ Missing dependency: {_ie}<br>'
                                    f'Run: pip install PyOpenGL PyOpenGL_accelerate glfw</div>'
                                )
                            except Exception as _be:
                                _brain_status.set_content(
                                    f'<div style="color:#ef4444;font-size:12px">❌ {str(_be)[:120]}</div>'
                                )

                        ui.button('🧠 Launch Brain Visualizer', on_click=_launch_brain_viz) \
                            .props('flat color=indigo').classes('mb-2')
                        _brain_status

                    # ── Research / MCP ───────────────────────────────────
                    with ui.expansion('🔬 Research & MCP', icon='search').classes('w-full mb-3'):
                        ui.label('RESEARCH TOOLS').classes('section-label')
                        ui.html('<div style="color:#64748b;font-size:12px;margin-bottom:8px">Configure autonomous research, web scraping, and MCP tool access.</div>')
                        with ui.row().classes('gap-3 w-full'):
                            ui.button(
                                '📖 Open Research Page',
                                on_click=lambda: ui.navigate.to('/research')
                            ).props('flat color=indigo')
                        ui.label('MCP STATUS').classes('section-label mt-3')
                        mcp_status_html = ui.html('<div style="color:#64748b;font-size:12px">Checking…</div>')
                        def _refresh_mcp():
                            try:
                                from cognition.research_mcp.rac import ResearchAgentCore
                                mcp_status_html.set_content('<div style="color:#22c55e;font-size:12px">✅ Research MCP available</div>')
                            except Exception as _me:
                                mcp_status_html.set_content(f'<div style="color:#ef4444;font-size:12px">❌ {str(_me)[:60]}</div>')
                        _refresh_mcp()
                        ui.button('Refresh', on_click=_refresh_mcp).props('flat dense').style('font-size:11px;color:#64748b')

                    # ── Security Tools ───────────────────────────────────
                    with ui.expansion('🔒 Security Tools', icon='security').classes('w-full mb-3'):
                        ui.label('SECURITY AUDIT').classes('section-label')
                        sec_audit_html = ui.html('<div style="color:#64748b;font-size:12px">Run audit to see results.</div>')
                        def _run_sec_audit():
                            try:
                                from managers.security_manager import security as _sec
                                from managers.settings_manager import config as _sc
                                cfg_dict = _sc.model_dump() if hasattr(_sc, 'model_dump') else _sc.dict()
                                warns = _sec.audit_config(cfg_dict)
                                if warns:
                                    lines = "".join(f'<div style="color:#f59e0b;font-size:11px">⚠️ {w}</div>' for w in warns[:5])
                                    sec_audit_html.set_content(lines)
                                else:
                                    sec_audit_html.set_content('<div style="color:#22c55e;font-size:12px">✅ No security issues found</div>')
                            except Exception as _e:
                                sec_audit_html.set_content(f'<div style="color:#ef4444;font-size:12px">Error: {str(_e)[:80]}</div>')
                        ui.button('Run Config Audit', on_click=_run_sec_audit).props('flat dense color=red')

                        ui.label('ASIMOV SAFETY LAYERS').classes('section-label mt-3')
                        ui.html('''<div style="font-size:11px;color:#64748b">
                            <div>Layer 1: Input regex filter (pre-LLM)</div>
                            <div>Layer 2: Output regex + semantic evasion detector (post-LLM)</div>
                            <div>Layer 3: Immutable beliefs in NarrativeIdentity</div>
                            <div>Layer 4: Rate limiting + injection detection audit log</div>
                        </div>''')

                        ui.label('RECENT AUDIT EVENTS').classes('section-label mt-3')
                        sec_events_html = ui.html('<div style="color:#64748b;font-size:12px">Loading…</div>')
                        def _refresh_sec_events():
                            try:
                                from managers.security_manager import security as _sec
                                events = _sec.recent_audit(15)
                                if events:
                                    cols = {'INJECTION': '#ef4444', 'TRUNCATED': '#f59e0b', 'BLOCK': '#f87171'}
                                    lines = []
                                    for e in events[-10:]:
                                        col = next((v for k, v in cols.items() if k in str(e).upper()), '#64748b')
                                        lines.append(f'<div style="font-size:10px;font-family:monospace;color:{col}">{str(e)[:100]}</div>')
                                    sec_events_html.set_content("".join(lines))
                                else:
                                    sec_events_html.set_content('<div style="color:#22c55e;font-size:12px">No events logged</div>')
                            except Exception as _e:
                                sec_events_html.set_content(f'<div style="color:#ef4444;font-size:12px">{str(_e)[:80]}</div>')
                        _refresh_sec_events()
                        ui.button('Refresh Events', on_click=_refresh_sec_events).props('flat dense').style('font-size:11px;color:#64748b')

                    # ── Vision Tools ─────────────────────────────────────
                    with ui.expansion('👁️ Vision Tools', icon='visibility').classes('w-full mb-3'):
                        ui.label('VISION DIAGNOSTICS').classes('section-label')
                        ui.html('<div style="color:#64748b;font-size:12px;margin-bottom:8px">Camera test, face management, and vision format diagnostics. Full vision settings are in the Vision tab.</div>')
                        with ui.row().classes('gap-2 flex-wrap'):
                            ui.button(
                                '📷 Open Vision Page',
                                on_click=lambda: ui.navigate.to('/vision')
                            ).props('flat color=indigo')
                        vision_diag_html = ui.html('<div style="color:#64748b;font-size:12px">Click to test.</div>')
                        def _test_vision():
                            try:
                                if not state.vision:
                                    vision_diag_html.set_content('<div style="color:#ef4444">Vision manager not initialized</div>')
                                    return
                                ok = state.vision.camera_active
                                mode = getattr(config, 'VISION_LLM_MODE', 'separate')
                                model = getattr(config, 'LAVA_MODEL', 'qwen/qwen3-vl-4b')
                                vision_diag_html.set_content(
                                    f'<div style="font-size:11px;color:#cbd5e1">'
                                    f'<div>Camera: <b style="color:{"#22c55e" if ok else "#ef4444"}">{"Active" if ok else "Off"}</b></div>'
                                    f'<div>LLM mode: <b style="color:#a5b4fc">{mode}</b></div>'
                                    f'<div>Vision model: <b style="color:#a5b4fc">{model}</b></div>'
                                    f'<div>Ambient interval: <b style="color:#a5b4fc">{getattr(config,"AMBIENT_VISION_INTERVAL",90)}s</b></div>'
                                    f'</div>'
                                )
                            except Exception as _e:
                                vision_diag_html.set_content(f'<div style="color:#ef4444">{str(_e)[:80]}</div>')
                        ui.button('Run Vision Diagnostics', on_click=_test_vision).props('flat dense color=indigo')
                        ui.html('').bind_content(vision_diag_html, 'content')  # live update placeholder
                        vision_diag_html  # keep reference

                    # ── Cognitive Health Report ──────────────────────────
                    with ui.expansion('🏥 Cognitive Health Report', icon='monitor_heart').classes('w-full mb-3'):
                        ui.label('HEALTH SNAPSHOT').classes('section-label')
                        health_html = ui.html('<div style="color:#64748b;font-size:12px">Click to generate.</div>')
                        def _gen_health():
                            try:
                                org = state.persona._organism if state.persona and hasattr(state.persona, '_organism') else None
                                if not org:
                                    health_html.set_content('<div style="color:#ef4444">Organism not available — Lumina must be running</div>')
                                    return
                                lines = []

                                # ── Architecture Monitor ─────────────────
                                am = getattr(org, 'arch_monitor', None)
                                if am:
                                    report = am.latest_report()
                                    if report:
                                        h = report.overall_health
                                        col = '#22c55e' if h > 0.65 else '#f59e0b' if h > 0.4 else '#ef4444'
                                        st = '✅ Healthy' if h > 0.65 else ('⚠️ Degraded' if h > 0.4 else '🔴 Critical')
                                        lines.append(f'<div style="margin-bottom:6px"><b style="color:{col}">{st} ({h:.0%})</b></div>')
                                        for k, v in report.component_scores.items():
                                            vc = '#22c55e' if v>0.6 else '#f59e0b' if v>0.35 else '#ef4444'
                                            lines.append(
                                                f'<div style="font-size:11px;color:#94a3b8">'
                                                f'{k.replace("_"," ").title()}: '
                                                f'<b style="color:{vc}">{v:.0%}</b></div>'
                                            )
                                    else:
                                        lines.append('<div style="color:#64748b;font-size:11px">Arch monitor: no report yet (waiting for interactions)</div>')

                                # ── Cognitive Validator (organism-level) ──
                                cv = getattr(org, 'cognitive_validator', None)
                                if cv:
                                    summ_cv = cv.summary()
                                    total   = summ_cv.get('total_validated', 0)
                                    avg     = summ_cv.get('recent_avg_score', 0.0)
                                    misrate = summ_cv.get('misalignment_rate', 0.0)
                                    lines.append(f'<div style="margin-top:8px;margin-bottom:3px"><b style="color:#a5b4fc">Cognitive Validator</b></div>')
                                    if total == 0:
                                        lines.append('<div style="font-size:11px;color:#64748b">No responses validated yet — send a message first</div>')
                                    else:
                                        avc = '#22c55e' if avg > 0.6 else '#f59e0b' if avg > 0.4 else '#ef4444'
                                        mc  = '#22c55e' if misrate < 0.2 else '#f59e0b' if misrate < 0.4 else '#ef4444'
                                        lines.append(f'<div style="font-size:11px;color:#94a3b8">Validated: <b style="color:#a5b4fc">{total}</b></div>')
                                        lines.append(f'<div style="font-size:11px;color:#94a3b8">Avg alignment: <b style="color:{avc}">{avg:.0%}</b></div>')
                                        lines.append(f'<div style="font-size:11px;color:#94a3b8">Misalignment rate: <b style="color:{mc}">{misrate:.0%}</b></div>')
                                        recent = cv.recent_scores(5)
                                        if recent:
                                            sc_html = " ".join(
                                                f'<span style="color:{"#22c55e" if s>0.6 else "#f59e0b" if s>0.4 else "#ef4444"}">{s:.2f}</span>'
                                                for s in recent
                                            )
                                            lines.append(f'<div style="font-size:10px;color:#64748b;margin-top:2px">Recent: {sc_html}</div>')

                                # ── Observatory ───────────────────────────
                                obs = getattr(org, 'observatory', None)
                                if obs:
                                    summ = obs.summary()
                                    lines.append(f'<div style="margin-top:8px;margin-bottom:3px"><b style="color:#a5b4fc">Observatory</b></div>')
                                    if not summ.get('ready'):
                                        lines.append(f'<div style="font-size:11px;color:#64748b">Calibrating baseline ({summ.get("ticks",0)}/{obs.BASELINE_MIN_SAMPLES} ticks)</div>')
                                    else:
                                        for key, lo, hi in [('ccs',0.4,0.85),('rdi',0.25,0.75),('gei',0.05,0.30),('idx',0.0,0.25)]:
                                            v = summ.get(key, 0)
                                            vc = '#22c55e' if lo <= v <= hi else '#f59e0b' if abs(v - (lo+hi)/2) < 0.2 else '#ef4444'
                                            lines.append(f'<div style="font-size:11px;color:#94a3b8">{key.upper()}: <b style="color:{vc}">{v:.3f}</b></div>')
                                        emg = summ.get('emergence', 0)
                                        ec = '#ef4444' if emg > 0.6 else '#f59e0b' if emg > 0.35 else '#22c55e'
                                        lines.append(f'<div style="font-size:11px;color:#94a3b8">Emergence: <b style="color:{ec}">{emg:.3f}</b></div>')

                                # ── Goal Engine ───────────────────────────────
                                goal_engine = getattr(org.ai_system if hasattr(org, 'ai_system') else None, 'goal_engine', None)
                                if goal_engine:
                                    try:
                                        metrics = goal_engine.get_metrics()
                                        active_goals = goal_engine.get_active_goals()
                                        
                                        lines.append(f'<div style="margin-top:12px;margin-bottom:3px"><b style="color:#a5b4fc">🎯 Goal Engine</b></div>')
                                        
                                        # Metrics
                                        gei_color = '#22c55e' if metrics['gei'] >= 0.12 else '#f59e0b' if metrics['gei'] >= 0.05 else '#64748b'
                                        lines.append(f'<div style="font-size:11px;color:#94a3b8">GEI: <b style="color:{gei_color}">{metrics["gei"]:.3f}</b> · Active: <b>{metrics["active"]}</b> · Total: {metrics["total_goals"]} · Persist: {metrics["avg_persistence"]:.1f} cycles</div>')
                                        
                                        # Active goals
                                        if active_goals:
                                            lines.append(f'<div style="font-size:11px;color:#94a3b8;margin-top:4px">Active Goals:</div>')
                                            for goal in active_goals[:3]:  # Top 3
                                                activation = goal.priority * goal.energy
                                                bar_color = '#22c55e' if activation > 0.5 else '#f59e0b' if activation > 0.3 else '#64748b'
                                                lines.append(
                                                    f'<div style="font-size:10px;color:#94a3b8;margin-left:8px">'
                                                    f'• <b style="color:{bar_color}">{goal.topic[:30]}</b> '
                                                    f'<span style="color:#64748b">({goal.origin} · p={goal.persistence})</span>'
                                                    f'</div>'
                                                )
                                            if metrics['active'] > 3:
                                                lines.append(f'<div style="font-size:10px;color:#64748b;margin-left:8px">+ {metrics["active"] - 3} more...</div>')
                                        else:
                                            lines.append(f'<div style="font-size:11px;color:#64748b;margin-top:4px">No active goals</div>')
                                    except Exception as e:
                                        lines.append(f'<div style="font-size:11px;color:#ef4444">Goal Engine error: {str(e)[:50]}</div>')

                                if not lines:
                                    lines.append('<div style="color:#64748b">No data available — start Lumina and send a message first</div>')

                                health_html.set_content('<div style="font-size:12px;color:#cbd5e1">' + "".join(lines) + '</div>')
                            except Exception as _e:
                                health_html.set_content(f'<div style="color:#ef4444">Error: {str(_e)[:120]}</div>')
                        ui.button('Generate Health Report', on_click=_gen_health).props('flat dense color=indigo')
                        health_html  # keep reference



                # ── Analytics tab ─────────────────────────────────────

                with ui.tab_panel('analytics'):
                    ui.label('📈 LOG ANALYTICS & SELF-ANALYSIS').classes('section-label')
                    ui.html('<div style="color:#64748b;font-size:12px;margin-bottom:16px">'
                            'Lumina analyzes her own cognitive trends and log patterns. '
                            'Trend data comes from the Observatory history (last 200 ticks). '
                            'Self-analysis uses the LLM to interpret what the numbers mean.</div>')

                    # ── Metric Trend Charts ──────────────────────────────
                    with ui.expansion('📊 Cognitive Metric Trends', icon='show_chart',
                                      value=True).classes('w-full mb-3'):
                        ui.label('OBSERVATORY HISTORY').classes('section-label')
                        trend_html = ui.html('<div style="color:#64748b;font-size:12px">'
                                             'Click Refresh to load trend data.</div>').classes('w-full')

                        async def _refresh_trends():
                            try:
                                org = getattr(getattr(state, 'persona', None), '_organism', None)
                                obs = getattr(org, 'observatory', None) if org else None
                                if not obs:
                                    trend_html.set_content('<div style="color:#64748b">Observatory not active.</div>')
                                    return
                                hist = obs.history_list(200)
                                if len(hist) < 3:
                                    trend_html.set_content(
                                        f'<div style="color:#64748b">Only {len(hist)} ticks collected — '
                                        f'need at least 3. Keep chatting!</div>')
                                    return

                                # Build sparkline bars for each metric
                                metrics = [
                                    ('CCS',  'ccs',       '#8b5cf6', (0.40, 0.85)),
                                    ('GEI',  'gei',       '#06b6d4', (0.05, 0.30)),
                                    ('IDX',  'idx',       '#f59e0b', (0.00, 0.25)),
                                    ('STR',  'str_score', '#ef4444', (0.05, 0.50)),
                                ]
                                html_parts = ['<div style="font-family:monospace;font-size:11px">']

                                for label, attr, color, (lo, hi) in metrics:
                                    vals = [getattr(s, attr, 0.0) for s in hist]
                                    cur  = vals[-1]
                                    avg  = sum(vals) / len(vals)
                                    mn, mx = min(vals), max(vals)
                                    trend_dir = '↑' if vals[-1] > vals[-3] else ('↓' if vals[-1] < vals[-3] else '→')
                                    status_c = '#22c55e' if lo <= cur <= hi else '#f59e0b' if abs(cur-(lo+hi)/2)<0.15 else '#ef4444'

                                    # Mini sparkline (40 chars wide)
                                    n_bars = 40
                                    step = max(1, len(vals) // n_bars)
                                    sampled = vals[::step][-n_bars:]
                                    rng = mx - mn if mx != mn else 0.001
                                    bar_chars = '▁▂▃▄▅▆▇█'
                                    spark = ''.join(bar_chars[min(7, int((v - mn) / rng * 7))] for v in sampled)

                                    html_parts.append(
                                        f'<div style="margin-bottom:10px">'
                                        f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
                                        f'    <span style="color:{color};font-weight:bold;width:32px">{label}</span>'
                                        f'    <span style="color:{status_c};font-size:13px;font-weight:bold">'
                                        f'      {cur:.3f} {trend_dir}</span>'
                                        f'    <span style="color:#475569;font-size:10px">'
                                        f'      avg={avg:.3f} min={mn:.3f} max={mx:.3f}</span>'
                                        f'    <span style="color:#64748b;font-size:10px">'
                                        f'      healthy [{lo}–{hi}]</span>'
                                        f'  </div>'
                                        f'  <div style="color:{color};letter-spacing:1px;font-size:12px">{spark}</div>'
                                        f'</div>'
                                    )

                                # Semantic memory summary
                                sm = getattr(org, 'semantic_memory', None)
                                if sm:
                                    summ = sm.summary()
                                    html_parts.append(
                                        f'<div style="margin-top:12px;padding-top:8px;border-top:1px solid #1e293b">'
                                        f'  <span style="color:#a5b4fc;font-weight:bold">Semantic Memory: </span>'
                                        f'  {summ.get("concepts",0)} concepts · '
                                        f'  {summ.get("relations",0)} relations · '
                                        f'  top: {", ".join(summ.get("top_concepts",[])[:5])}</div>'
                                    )

                                html_parts.append('</div>')
                                trend_html.set_content(''.join(html_parts))
                                ui.notify('Trends refreshed', type='positive', timeout=1500)
                            except Exception as e:
                                trend_html.set_content(f'<div style="color:#ef4444">Error: {str(e)[:200]}</div>')

                        ui.button('Refresh Trends', on_click=_refresh_trends,
                                  icon='refresh').props('flat dense color=violet').classes('mt-2')

                    # ── Log Pattern Scanner ──────────────────────────────
                    with ui.expansion('🔍 Log Pattern Scanner', icon='search').classes('w-full mb-3'):
                        ui.label('RECENT LOG ANALYSIS').classes('section-label')
                        log_scan_html = ui.html('<div style="color:#64748b;font-size:12px">'
                                                'Click Scan to analyze recent logs.</div>').classes('w-full')

                        async def _scan_logs():
                            import pathlib, re, collections
                            try:
                                log_path = pathlib.Path('data/persona/ai_system.log')
                                if not log_path.exists():
                                    log_scan_html.set_content('<div style="color:#64748b">No log file found at data/persona/ai_system.log</div>')
                                    return

                                lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-500:]

                                errors   = [l for l in lines if ' - ERROR - '   in l]
                                warnings = [l for l in lines if ' - WARNING - ' in l]
                                infos    = [l for l in lines if ' - INFO - '    in l]

                                # Extract module names from errors/warnings
                                mod_pattern = re.compile(r' - ([\.\w]+) - (?:ERROR|WARNING) - ')
                                mod_counter = collections.Counter()
                                for l in errors + warnings:
                                    m = mod_pattern.search(l)
                                    if m: mod_counter[m.group(1)] += 1

                                # Recurring phrases
                                phrase_counter = collections.Counter()
                                for l in errors + warnings:
                                    # Extract text after level
                                    parts = l.split(' - ', 3)
                                    if len(parts) == 4:
                                        msg = parts[3][:80]
                                        phrase_counter[msg] += 1

                                html = ['<div style="font-family:monospace;font-size:11px">']
                                html.append(
                                    f'<div style="margin-bottom:8px">'
                                    f'  <span style="color:#22c55e">INFO: {len(infos)}</span> · '
                                    f'  <span style="color:#f59e0b">WARN: {len(warnings)}</span> · '
                                    f'  <span style="color:#ef4444">ERROR: {len(errors)}</span>'
                                    f'  <span style="color:#475569;font-size:10px"> (last 500 lines)</span>'
                                    f'</div>'
                                )

                                if mod_counter:
                                    html.append('<div style="color:#a5b4fc;margin:6px 0 3px">Top noisy modules:</div>')
                                    for mod, cnt in mod_counter.most_common(5):
                                        html.append(f'<div style="color:#94a3b8;margin-left:8px">'
                                                    f'{mod} <b style="color:#f59e0b">×{cnt}</b></div>')

                                if phrase_counter:
                                    html.append('<div style="color:#a5b4fc;margin:6px 0 3px">Recurring messages:</div>')
                                    for msg, cnt in phrase_counter.most_common(5):
                                        if cnt > 1:
                                            col = '#ef4444' if cnt > 5 else '#f59e0b'
                                            html.append(
                                                f'<div style="color:#94a3b8;margin-left:8px;margin-bottom:2px">'
                                                f'  <b style="color:{col}">×{cnt}</b> {msg[:70]}</div>'
                                            )

                                if errors:
                                    html.append('<div style="color:#a5b4fc;margin:6px 0 3px">Last 3 errors:</div>')
                                    for l in errors[-3:]:
                                        short = l.split(' - ', 3)[-1][:120]
                                        html.append(f'<div style="color:#ef4444;margin-left:8px;margin-bottom:2px">{short}</div>')

                                html.append('</div>')
                                log_scan_html.set_content(''.join(html))
                                ui.notify('Log scan complete', type='positive', timeout=1500)
                            except Exception as e:
                                log_scan_html.set_content(f'<div style="color:#ef4444">Scan error: {str(e)[:200]}</div>')

                        ui.button('Scan Logs', on_click=_scan_logs,
                                  icon='search').props('flat dense color=cyan').classes('mt-2')

                    # ── Lumina Self-Analysis ─────────────────────────────
                    with ui.expansion('🧠 Lumina Self-Analysis', icon='psychology',
                                      value=True).classes('w-full mb-3'):
                        ui.label('AI-POWERED COGNITIVE SELF-ANALYSIS').classes('section-label')
                        ui.html('<div style="color:#64748b;font-size:11px;margin-bottom:8px">'
                                'Lumina reads her own metrics and writes a free-form analysis '
                                'of trends, anomalies, and cognitive health. Requires LLM.</div>')
                        self_analysis_html = ui.html(
                            '<div style="color:#64748b;font-size:12px">Click Analyze to generate.</div>'
                        ).classes('w-full')

                        async def _self_analyze():
                            import datetime as _dt
                            try:
                                if not state.llm:
                                    self_analysis_html.set_content(
                                        '<div style="color:#ef4444">LLM not available.</div>')
                                    return

                                self_analysis_html.set_content(
                                    '<div style="color:#64748b;font-style:italic">⏳ Analyzing...</div>')

                                org = getattr(getattr(state, 'persona', None), '_organism', None)

                                # ── Gather all available metrics ──────────
                                context_parts = []

                                # Observatory trends
                                obs = getattr(org, 'observatory', None) if org else None
                                if obs:
                                    hist = obs.history_list(50)
                                    if hist:
                                        metrics = [('CCS','ccs'),('GEI','gei'),('IDX','idx'),('STR','str_score')]
                                        for label, attr in metrics:
                                            vals = [getattr(s, attr, 0.0) for s in hist]
                                            avg = sum(vals)/len(vals)
                                            trend = vals[-1] - vals[0] if len(vals)>1 else 0
                                            context_parts.append(
                                                f'{label}: current={vals[-1]:.3f} avg={avg:.3f} '
                                                f'trend={trend:+.3f} over {len(vals)} ticks'
                                            )

                                # Capabilities from semantic memory
                                sm = getattr(org, 'semantic_memory', None) if org else None
                                if sm:
                                    caps = sm.get_capabilities()
                                    for name, data in caps.items():
                                        c, t = data['confidence'], data['trend']
                                        context_parts.append(
                                            f'Capability {name}: confidence={c:.2f} trend={t:+.3f}')
                                    summ = sm.summary()
                                    context_parts.append(
                                        f'Semantic memory: {summ.get("concepts",0)} concepts, '
                                        f'{summ.get("relations",0)} relations')

                                # Emotional state
                                if org:
                                    try:
                                        emo = org.emotional_state
                                        if emo:
                                            ev = getattr(emo, 'valence', None)
                                            ea = getattr(emo, 'arousal', None)
                                            if ev is not None:
                                                context_parts.append(f'Emotional state: valence={ev:.2f} arousal={ea:.2f}')
                                    except Exception:
                                        pass

                                # Recent thought topics
                                ts_mod = getattr(org, 'thought_stream', None) if org else None
                                if ts_mod:
                                    recent_thoughts = ts_mod.recent(5)
                                    if recent_thoughts:
                                        topics = [t.source for t in recent_thoughts]
                                        context_parts.append(f'Recent thought sources: {", ".join(topics)}')

                                if not context_parts:
                                    self_analysis_html.set_content(
                                        '<div style="color:#64748b">No metrics available yet — '
                                        'send some messages first.</div>')
                                    return

                                metrics_text = '\n'.join(context_parts)
                                ts_str = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')

                                prompt = f"""You are {_gpn()}, an AI cognitive organism performing introspective self-analysis.
The time is {ts_str}. Here are your current cognitive metrics:

{metrics_text}

Write a thoughtful self-analysis (150-250 words) covering:
1. What the trends in CCS, GEI, IDX, STR tell you about your current cognitive state
2. Which capabilities feel strong or weak based on confidence scores
3. Any anomalies or patterns you notice
4. One concrete thing you would want to improve

Write in first person, as Lumina reflecting on herself. Be specific, not generic."""

                                try:
                                    # Use generate_bare — no history read/write,
                                    # avoids polluting chat + no 400 from oversized history
                                    _gen_fn = getattr(state.llm, 'generate_bare',
                                               getattr(state.llm, 'generate', None))
                                    result = await asyncio.to_thread(
                                            _gen_fn, prompt, max_tokens=350
                                        ) if _gen_fn else None

                                    if not result:
                                        self_analysis_html.set_content(
                                            '<div style="color:#64748b">LLM returned no response.</div>')
                                        return

                                    # Format nicely
                                    paragraphs = [p.strip() for p in result.strip().split('\n\n') if p.strip()]
                                    html_out = ['<div style="font-size:12px;color:#cbd5e1;line-height:1.6">',
                                                f'<div style="color:#64748b;font-size:10px;margin-bottom:8px">Generated {ts_str}</div>']
                                    for p in paragraphs:
                                        html_out.append(f'<p style="margin-bottom:8px">{p}</p>')
                                    html_out.append('</div>')
                                    self_analysis_html.set_content(''.join(html_out))
                                    ui.notify('Self-analysis complete', type='positive', timeout=2000)
                                except Exception as llm_e:
                                    self_analysis_html.set_content(
                                        f'<div style="color:#ef4444">LLM error: {str(llm_e)[:150]}</div>')
                            except Exception as e:
                                self_analysis_html.set_content(
                                    f'<div style="color:#ef4444">Error: {str(e)[:200]}</div>')

                        with ui.row().classes('gap-2 mt-2'):
                            ui.button('Analyze Now', on_click=_self_analyze,
                                      icon='psychology').props('flat dense color=purple')
                            ui.html('<span style="color:#475569;font-size:10px;line-height:2.5">'
                                    '~5-10s · uses 1 LLM call</span>')









        # ── Save, Reset, Export, Import buttons ─────────────────────────
        with ui.row().classes('w-full justify-between items-center mt-2 flex-wrap gap-2'):
            ui.button(
                '← Back to Chat', on_click=lambda: ui.navigate.to('/')
            ).props('flat').style('color:#64748b')
            
            with ui.row().classes('gap-2'):
                # Export button
                async def export_config():
                    # Save to a default location with timestamp
                    export_dir = Path('config_backups')
                    export_dir.mkdir(exist_ok=True)
                    export_file = export_dir / f'config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    
                    if export_settings(export_file):
                        ui.notify(f'✅ Exported to {export_file}', type='positive', position='bottom-right')
                    else:
                        ui.notify('❌ Export failed', type='negative', position='bottom-right')
                
                ui.button(
                    '📤 Export', on_click=export_config
                ).props('flat color=blue').classes('text-sm')
                
                # Import button with custom file picker dialog
                async def import_config():
                    # Create a dialog with file upload
                    with ui.dialog() as dialog, ui.card().classes('p-4 w-96 max-w-full'):
                        ui.label('Import Configuration').classes('text-lg font-bold mb-4')
                        ui.label('Select a config.json file to import:').classes('text-sm text-slate-400 mb-2')
                        
                        # Store uploaded file info
                        upload_result = {'path': None, 'name': None}
                        
                        async def handle_upload(e):
                            try:
                                # Check for content in different possible attributes
                                file_content = None
                                file_name = getattr(e, 'name', 'config.json')
                                
                                if hasattr(e, 'content') and e.content:
                                    file_content = e.content
                                elif hasattr(e, 'data') and e.data:
                                    file_content = e.data
                                elif hasattr(e, 'bytes') and e.bytes:
                                    file_content = e.bytes
                                else:
                                    ui.notify('❌ Could not read file content - unknown format', type='negative', position='bottom-right')
                                    return
                                
                                # Save uploaded file temporarily
                                temp_dir = Path('temp_imports')
                                temp_dir.mkdir(exist_ok=True)
                                temp_file = temp_dir / file_name
                                
                                # Write the content (handle both bytes and string)
                                if isinstance(file_content, bytes):
                                    temp_file.write_bytes(file_content)
                                else:
                                    # If it's a string, encode to bytes
                                    temp_file.write_bytes(str(file_content).encode('utf-8'))
                                
                                upload_result['path'] = temp_file
                                upload_result['name'] = file_name
                                ui.notify(f'✅ File uploaded: {file_name}', type='positive', position='bottom-right')
                                
                            except Exception as ex:
                                logger.error(f"Upload error: {ex}")
                                ui.notify(f'❌ Upload error: {str(ex)}', type='negative', position='bottom-right')
                        
                        # Create the upload element
                        ui.upload(
                            label='Click to select file or drag & drop',
                            on_upload=handle_upload,
                            auto_upload=True,
                            max_file_size=10_000_000  # 10MB max
                        ).props('accept=.json flat').classes('w-full mb-4')
                        
                        with ui.row().classes('w-full justify-end gap-2'):
                            ui.button('Cancel', on_click=dialog.close).props('flat')
                            ui.button('Import', on_click=lambda: dialog.submit('import')).props('color=primary')
                        
                        result = await dialog
                        
                        if result == 'import' and upload_result['path']:
                            import_path = upload_result['path']
                            imported = import_settings(import_path)
                            
                            # Clean up temp file
                            try:
                                import_path.unlink()
                                # Try to remove temp directory if empty
                                try:
                                    Path('temp_imports').rmdir()
                                except:
                                    pass
                            except:
                                pass
                            
                            if imported:
                                # Update UI with new values
                                llm_prov.value = imported.LLM_PROVIDER
                                llm_model.value = imported.LLM_MODEL
                                llm_url.value = imported.LLM_BASE_URL
                                openai_key.value = imported.OPENAI_API_KEY
                                mem_backend.value = imported.MEMORY_BACKEND
                                tts_select.value = imported.TTS_PROVIDER
                                stt_select.value = imported.STT_PROVIDER
                                whisper_model.value = imported.WHISPER_MODEL
                                elevenlabs_key.value = imported.ELEVENLABS_API_KEY
                                voice_language.value = imported.VOICE_LANGUAGE
                                camera_enabled.value = imported.CAMERA_AUTOSTART
                                camera_id.value = imported.CAMERA_ID
                                camera_fps.value = imported.CAMERA_FPS
                                camera_resolution.value = imported.CAMERA_RESOLUTION
                                vision_mode.value = getattr(imported, 'VISION_MODE', 'keyword')
                                lava_model.value = imported.LAVA_MODEL
                                
                                ui.notify('✅ Settings imported! Click Save & Apply to use them.', type='positive', position='bottom-right')
                            else:
                                ui.notify('❌ Import failed - invalid config file', type='negative', position='bottom-right')
                
                ui.button(
                    '📥 Import', on_click=import_config
                ).props('flat color=green').classes('text-sm')
                
                # Reset button
                async def reset_settings():
                    confirmed = [False]
                    with ui.dialog() as _dlg, ui.card().classes('p-5 gap-3'):
                        ui.label('Reset Settings').classes('font-semibold text-lg')
                        ui.label('Reset all settings to defaults? This cannot be undone.').classes('text-slate-400 text-sm')
                        with ui.row().classes('gap-3 mt-2'):
                            def _do_reset():
                                confirmed[0] = True
                                _dlg.close()
                            ui.button('Reset', on_click=_do_reset).props('color=orange')
                            ui.button('Cancel', on_click=_dlg.close).props('flat')
                    await _dlg
                    if confirmed[0]:
                        new_config = reset_to_defaults()
                        # Update UI with defaults
                        llm_prov.value = new_config.LLM_PROVIDER
                        llm_model.value = new_config.LLM_MODEL
                        llm_url.value = new_config.LLM_BASE_URL
                        openai_key.value = new_config.OPENAI_API_KEY
                        mem_backend.value = new_config.MEMORY_BACKEND
                        tts_select.value = new_config.TTS_PROVIDER
                        stt_select.value = new_config.STT_PROVIDER
                        whisper_model.value = new_config.WHISPER_MODEL
                        elevenlabs_key.value = new_config.ELEVENLABS_API_KEY
                        voice_language.value = new_config.VOICE_LANGUAGE
                        camera_enabled.value = new_config.CAMERA_AUTOSTART
                        camera_id.value = new_config.CAMERA_ID
                        camera_fps.value = new_config.CAMERA_FPS
                        camera_resolution.value = new_config.CAMERA_RESOLUTION
                        vision_mode.value = new_config.VISION_MODE
                        lava_model.value = new_config.LAVA_MODEL
                        
                        ui.notify('Settings reset to defaults. Click Save & Apply to use them.', type='warning', position='bottom-right')
                
                ui.button(
                    '🔄 Reset to Defaults', on_click=reset_settings
                ).props('flat color=orange').classes('text-sm')
                
                # Save button
                async def save_all():
                    # ── 1. Push UI values into config ────────────────────────
                    config.PERSONA_NAME   = (persona_name_input.value or "Lumina").strip()
                    config.LLM_PROVIDER = llm_prov.value
                    config.LLM_MODEL = llm_model.value
                    config.LLM_BASE_URL = llm_url.value
                    config.OPENAI_API_KEY = openai_key.value
                    config.MEMORY_BACKEND = mem_backend.value
                    config.TTS_PROVIDER = tts_select.value
                    config.STT_PROVIDER = stt_select.value
                    config.WHISPER_MODEL = whisper_model.value
                    config.ELEVENLABS_API_KEY = elevenlabs_key.value
                    config.VOICE_LANGUAGE = voice_language.value
                    config.VAD_AGGRESSIVENESS = int(vad_aggr.value)
                    config.VAD_ONSET_CHUNKS = int(vad_onset.value)
                    config.VAD_SILENCE_DURATION = float(vad_silence.value)
                    config.VAD_MIN_SPEECH_DURATION = float(vad_min_dur.value)
                    config.VAD_ENERGY_GATE_FACTOR = float(vad_gate.value)
                    config.CAMERA_AUTOSTART = camera_enabled.value
                    config.CAMERA_ID = int(camera_id.value)
                    config.CAMERA_FPS = int(camera_fps.value)
                    config.CAMERA_RESOLUTION = camera_resolution.value
                    config.VISION_MODE = vision_mode.value
                    config.LAVA_MODEL = lava_model.value
                    config.VISION_LLM_MODE = vision_llm_mode.value
                    config.AMBIENT_VISION_INTERVAL = int(ambient_vision_interval.value or 90)
                    # Re-attach vision with updated interval (no restart needed)
                    try:
                        organism = getattr(state.persona, '_organism', None) if state.persona else None
                        if organism and state.vision:
                            loop = getattr(organism, '_loop', None)
                            if loop and hasattr(loop, 'attach_vision'):
                                loop.attach_vision(state.vision)
                    except Exception:
                        pass

                    # ── 2. Persist to disk ───────────────────────────────────
                    # JSON is now the single source of truth. Each reload_*
                    # method calls state._refresh_config() which reads fresh
                    # from JSON and updates _settings_mod.config atomically.
                    save_settings(config, silent=True)

                    ui.notify('⏳ Applying settings…', type='info', position='bottom-right')

                    # ── 3. Reload components off the event loop ───────────────
                    # LLM and memory are fast; audio can take ~7s (Coqui XTTS).
                    # Run them in a thread so the UI stays responsive.
                    loop = asyncio.get_running_loop()

                    await loop.run_in_executor(None, state.reload_llm)
                    await loop.run_in_executor(None, state.reload_memory)

                    # Remember if voice was active before tearing down audio.
                    # voice_select lives in main_page's scope — we check state instead.
                    voice_was_active = (
                        state.conv_audio is not None and state.conv_audio.is_listening
                    )

                    # Stop active listener before reload to avoid thread leaks
                    if state.conv_audio and state.conv_audio.is_listening:
                        await loop.run_in_executor(None, state.conv_audio.stop_conversation)

                    await loop.run_in_executor(None, lambda: state.reload_audio(force=True))
                    await loop.run_in_executor(None, state.reload_vision)

                    # ── 4. Notify user to re-enable voice if it was active ────
                    # on_voice_change() is scoped to main_page and cannot be called
                    # from here. Audio is reloaded with fresh instances — the user
                    # must toggle voice mode off/on once to re-bind the callbacks.
                    if voice_was_active:
                        ui.notify(
                            '🎤 Voice was reset — please toggle voice mode off then back on',
                            type='warning', position='bottom-right', timeout=6000
                        )

                    # ── 5. Re-start camera if needed ──────────────────────────
                    if config.CAMERA_AUTOSTART and state.vision:
                        try:
                            asyncio.create_task(asyncio.to_thread(state.vision.start_camera))
                        except Exception as e:
                            logger.error(f"Auto-start camera error: {e}")

                    ui.notify('✅ Settings saved and applied!', type='positive', position='bottom-right')
                    print(f"\n✅ Settings applied: {config.LLM_PROVIDER}/{config.LLM_MODEL}")

                ui.button(
                    '💾 Save & Apply', on_click=save_all
                ).classes('save-btn').props('color=indigo')

# ─────────────────────────────────────────────
#  /lumina — Lumina Cognitive Dashboard
# ─────────────────────────────────────────────

EMO_COLORS = {
    "curiosity":    "#3b82f6",
    "warmth":       "#f97316",
    "enthusiasm":   "#eab308",
    "satisfaction": "#22c55e",
    "anxiety":      "#f87171",
    "frustration":  "#dc2626",
}


