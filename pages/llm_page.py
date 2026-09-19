"""
pages/llm_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM diagnostics.
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

@ui.page('/llm')
async def llm_page():
    """Standalone LLM Parameters page — shown by the Neural Core petal."""
    # Fix: ui.context.client.connected() defaults to a 3.0s WebSocket
    # handshake timeout. On a loaded system this can be exceeded, killing
    # the page coroutine before anything renders — confirmed in production
    # logs (TimeoutError at cognitive_dashboard_page.py:398, same pattern).
    # Raised to 10s and wrapped so a genuine failure shows a visible
    # message instead of a silently blank page.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(f"[llm_page] Client connection timed out after 10s")
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
            ui.html(
                '<div style="width:38px;height:38px;border-radius:10px;'
                'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
                'display:flex;align-items:center;justify-content:center;'
                'font-size:20px">&#x1F9E0;</div>'
            )
            ui.label('Neural Core — LLM Parameters').classes('app-title')
            ui.space()
            ui.button(
                icon='settings', on_click=lambda: ui.navigate.to('/settings')
            ).props('flat dense round').style('color:#64748b').tooltip('All Settings')

    with ui.column().classes('w-full max-w-2xl mx-auto p-6 gap-5 settings-page-content'):

        with ui.element('div').classes('settings-card w-full p-6'):
            # ── Persona Identity ─────────────────────────────────────────
            ui.label('PERSONA IDENTITY').classes('section-label')
            with ui.grid(columns=2).classes('w-full gap-4 mb-4'):
                persona_name_input = ui.input(
                    'Persona Name',
                    value=getattr(config, 'PERSONA_NAME', 'PandoraBOX'),
                    placeholder='PandoraBOX'
                ).props('outlined dense').classes('dark-input')
                with ui.element('div').classes('flex items-center'):
                    ui.html(
                        '<div style="color:#64748b;font-size:11px;line-height:1.5">'
                        'The name your AI persona uses for itself.<br>'
                        'Used in all prompts and responses.</div>'
                    )
            ui.separator().classes('mb-4')

            ui.label('LLM PROVIDER').classes('section-label')
            with ui.grid(columns=2).classes('w-full gap-4 mt-3'):
                # Default URLs per provider — auto-filled on change, still editable
                _PROVIDER_URLS = {
                    'ollama':   'http://localhost:11434',
                    'lmstudio': 'http://localhost:1234/v1',
                    'openai':   'https://api.openai.com/v1',
                }

                llm_url = ui.input(
                    'Base URL', value=config.LLM_BASE_URL
                ).props('outlined dense').classes('col-span-2 dark-input')

                def _on_provider_change(e):
                    default = _PROVIDER_URLS.get(e.value, '')
                    # Only auto-fill if the current value matches another provider's
                    # default (i.e. user hasn't typed a custom URL)
                    current = llm_url.value.strip()
                    is_default = current in _PROVIDER_URLS.values() or current == ''
                    if is_default:
                        llm_url.value = default
                    # Show/hide OpenAI key based on provider
                    openai_key.visible = (e.value == 'openai')

                llm_prov = ui.select(
                    ['ollama', 'lmstudio', 'openai'],
                    label='Provider', value=config.LLM_PROVIDER,
                    on_change=_on_provider_change
                ).props('outlined dense').classes('dark-input')

                llm_model = ui.input(
                    'Model Name', value=config.LLM_MODEL
                ).props('outlined dense').classes('dark-input')

                openai_key = ui.input(
                    'OpenAI API Key', value=config.OPENAI_API_KEY, password=True
                ).props('outlined dense').classes('col-span-2 dark-input')
                openai_key.visible = (config.LLM_PROVIDER == 'openai')

                text_model = ui.input(
                    'Text-only Model (lightweight)', value=getattr(config, 'TEXT_MODEL', '')
                ).props('outlined dense').classes('col-span-2 dark-input').tooltip(
                    'Optional lighter model for text-only turns. Leave blank to use Model Name for everything.'
                )
                llm_ctx = ui.number(
                    'Context Window (tokens)',
                    value=getattr(config, 'LLM_CTX', 4096), min=512, max=131072,
                ).props('outlined dense').classes('col-span-2 dark-input').tooltip(
                    'Must match the context size set in LM Studio / Ollama. Default 4096.'
                )

        with ui.element('div').classes('info-card'):
            ui.label('PROVIDER NOTES').classes('section-label')
            ui.markdown(
                '- **ollama** -> `http://localhost:11434` - run `ollama serve` first\n'
                '- **lmstudio** -> `http://localhost:1234/v1` - start LM Studio server\n'
                '- **openai** -> requires API key above\n'
                '- **Text-only model**: if set, lighter inference for text turns; vision turns still use main model'
            ).style('font-size:0.82rem;color:#94a3b8;line-height:1.7')

        ui.separator().classes('my-5 opacity-10')

        with ui.element('div').classes('settings-card w-full p-5'):
            ui.label('RESEARCH WEB SEARCH BACKEND').classes('section-label')
            ui.label('How the Research Cortex searches the web. AUTO tries all sources in order.').classes('text-slate-400 text-sm mb-3')

            research_backend = ui.select(
                {
                    'auto':    '🔄 AUTO — tries all sources (recommended)',
                    'brave':   '🦁 Brave Search API (free 2k/month)',
                    'claude':  '⚡ Claude Web Search (Anthropic API)',
                    'rss':     '📰 RSS Feeds only (news headlines)',
                    'stealth': '🕶️ Stealth Browser only (DDG/Bing scrape)',
                },
                value=getattr(config, 'RESEARCH_SEARCH_BACKEND', 'auto'),
                label='Search Backend'
            ).props('outlined dense').classes('w-full dark-input')

            anthropic_key = ui.input(
                'Anthropic API Key (for Claude search)',
                value=getattr(config, 'ANTHROPIC_API_KEY', ''),
                password=True
            ).props('outlined dense').classes('w-full dark-input mt-3')

            brave_key = ui.input(
                'Brave Search API Key (free at api.search.brave.com)',
                value=getattr(config, 'BRAVE_SEARCH_KEY', ''),
                password=True
            ).props('outlined dense').classes('w-full dark-input mt-2')

            serpapi_key = ui.input(
                'SerpAPI Key (optional)',
                value=getattr(config, 'SERPAPI_KEY', ''),
                password=True
            ).props('outlined dense').classes('w-full dark-input mt-2')

            with ui.element('div').classes('info-card mt-3').style('padding:10px 14px;border-color:#1e3a5f;background:#080f1a'):
                ui.markdown(
                    '**AUTO (recommended)** — tries Brave → Claude → SerpAPI → RSS feeds → Stealth browser in order.\n\n'
                    '**Brave API** — free 2 000 searches/month. Sign up at `api.search.brave.com` (2 min). Most reliable.\n\n'
                    '**Claude** — Anthropic built-in web search. Fast, no browser. Needs `pip install anthropic`.\n\n'
                    '**RSS** — Direct news feeds (BBC/Reuters/AP/Guardian). Zero bot risk, great for news.\n\n'
                    '**Stealth** — Headless Chromium with anti-detection. Last resort, may still get blocked.'
                ).style('font-size:0.78rem;color:#475569;line-height:1.6')

        ui.separator().classes('my-5 opacity-10')

        # ── Response verbosity ────────────────────────────────────────────
        with ui.element('div').classes('settings-card w-full p-5'):
            ui.label('RESPONSE LENGTH').classes('section-label')
            ui.label('Controls response verbosity — both the prompt instruction and the token budget sent to the LLM.').classes('text-slate-400 text-sm mb-3')

            verbosity_select = ui.select(
                {
                    'concise': '📝 Concise  — quick chat replies',
                    'verbose': '📖 Verbose  — detailed answers, research, code',
                },
                value=getattr(config, 'RESPONSE_VERBOSITY', 'concise'),
                label='Default mode'
            ).props('outlined dense').classes('w-full dark-input')

            ui.label('TOKEN BUDGETS').classes('section-label mt-4')
            ui.label('Max tokens sent to the LLM for each mode. 1 token ≈ 0.75 word.').classes('text-slate-400 text-sm mb-2')

            with ui.grid(columns=2).classes('w-full gap-4'):
                tokens_concise = ui.number(
                    'Concise — max tokens',
                    value=getattr(config, 'RESPONSE_TOKENS_CONCISE', 300),
                    min=50, max=2000, step=50,
                ).props('outlined dense').classes('dark-input')
                tokens_verbose = ui.number(
                    'Verbose — max tokens',
                    value=getattr(config, 'RESPONSE_TOKENS_VERBOSE', 1200),
                    min=200, max=4000, step=100,
                ).props('outlined dense').classes('dark-input')

            with ui.element('div').classes('info-card mt-3').style('padding:10px 14px'):
                ui.markdown(
                    '**Concise** injects *"Keep your response under N words"* and caps generation at the token budget.\n\n'
                    '**Verbose** sends no word-count instruction and uses the verbose token budget.\n\n'
                    'Recommended: **300 / 1200** for Mistral Small 22B · **400 / 2000** for larger models.\n\n'
                    'The chat toolbar toggle switches mode instantly without reloading.'
                ).style('font-size:0.78rem;color:#94a3b8;line-height:1.6')

        ui.separator().classes('my-5 opacity-10')

        # ── Custom system prompt editor ───────────────────────────────────
        with ui.element('div').classes('settings-card w-full p-5'):
            ui.label('CUSTOM SYSTEM PROMPT').classes('section-label')
            ui.label(
                'Text appended after the built-in "HOW TO RESPOND" block. '
                'Leave empty to use the default prompt unchanged.'
            ).classes('text-slate-400 text-sm mb-3')

            custom_prompt_area = ui.textarea(
                label='Additional instructions (appended to system prompt)',
                value=getattr(config, 'CUSTOM_SYSTEM_PROMPT', ''),
                placeholder=(
                    'e.g. "Always include a concrete example in your answers."\n'
                    'Leave empty to use default prompt only.'
                )
            ).props('outlined rows=6').classes('w-full dark-input font-mono text-sm')
            custom_prompt_area.style('font-family: monospace; font-size: 0.82rem;')

            with ui.row().classes('w-full items-center gap-3 mt-2'):
                def _clear_custom_prompt():
                    custom_prompt_area.value = ''
                    ui.notify('Custom prompt cleared', type='info', position='bottom-right', timeout=2000)
                ui.button('Clear', icon='clear', on_click=_clear_custom_prompt).props('flat dense size=sm').classes('text-slate-400')
                ui.label('Changes apply on next Save & Apply.').classes('text-slate-500 text-xs')

            with ui.element('div').classes('info-card mt-3').style('padding:10px 14px'):
                ui.markdown(
                    '**The base prompt structure is preserved** — this text is only appended at the end.\n\n'
                    'Keep additional instructions short and directive. '
                    'Long injections risk pushing the cognitive state context out of the window.\n\n'
                    'Example uses: enforce a specific output format, add domain expertise hints, '
                    'set a language register.'
                ).style('font-size:0.78rem;color:#94a3b8;line-height:1.6')

        async def save_llm():
            config.PERSONA_NAME   = (persona_name_input.value or "PandoraBOX").strip()
            config.LLM_PROVIDER   = llm_prov.value
            config.LLM_MODEL      = llm_model.value
            config.LLM_CTX        = int(llm_ctx.value or 4096)
            config.LLM_BASE_URL   = llm_url.value
            config.OPENAI_API_KEY = openai_key.value
            if hasattr(config, 'TEXT_MODEL'):
                config.TEXT_MODEL = text_model.value
            config.ANTHROPIC_API_KEY       = anthropic_key.value
            config.BRAVE_SEARCH_KEY         = brave_key.value
            config.SERPAPI_KEY              = serpapi_key.value
            config.RESEARCH_SEARCH_BACKEND  = research_backend.value
            # ── Verbosity + custom prompt ─────────────────────────────────
            config.RESPONSE_VERBOSITY        = verbosity_select.value
            config.RESPONSE_TOKENS_CONCISE   = int(tokens_concise.value or 300)
            config.RESPONSE_TOKENS_VERBOSE   = int(tokens_verbose.value or 1200)
            config.CUSTOM_SYSTEM_PROMPT  = (custom_prompt_area.value or '').strip()
            # Sync chat toolbar toggle to match saved value
            # Apply immediately to running web_agent (no restart needed)
            try:
                from cognition.research_mcp.web_agent import set_search_backend
                set_search_backend(
                    backend       = research_backend.value,
                    anthropic_key = anthropic_key.value,
                    brave_key     = brave_key.value,
                    serpapi_key   = serpapi_key.value,
                )
            except Exception:
                pass
            save_settings(config, silent=True)
            ui.notify('Applying LLM settings...', type='info', position='bottom-right')
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, state.reload_llm)
            # Update PandoraBOX's adapter model label to match
            if state.persona and state.persona.is_ready:
                try:
                    state.persona._system.llm.config.model_name = llm_model.value
                except Exception:
                    pass
            ui.notify(
                f'LLM set to {llm_prov.value} / {llm_model.value}',
                type='positive', position='bottom-right'
            )

        ui.button('Save & Apply', icon='save', on_click=save_llm).classes('save-btn').props('color=indigo')



