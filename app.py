"""
app.py — Lumina Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This file is the skeleton of the application.
It does exactly four things and nothing else:

  1. Security bootstrap  — generate/load .env + session secret
  2. Platform setup      — logging, UTF-8 fix on Windows
  3. Route registration  — `import pages` triggers every @ui.page(...)
                           decorator across all files in pages/
  4. Server start        — ui.run() with production-safe settings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture
  Brain (cognitive organism)  →  core/  managers/  cognition/
  Body  (UI surface)          →  pages/  (one file per route)
  Spine (this file)           →  wires brain to body, starts the server

Nothing cognitive belongs here.
Nothing presentational belongs in core/.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ══════════════════════════════════════════════════════════════════════
#  1.  SECURITY BOOTSTRAP  (must run before anything reads os.environ)
# ══════════════════════════════════════════════════════════════════════
from core.bootstrap import _bootstrap_security
_bootstrap_security()


# ══════════════════════════════════════════════════════════════════════
#  2.  PLATFORM SETUP
# ══════════════════════════════════════════════════════════════════════
import sys
import os
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Windows UTF-8 fix — reconfigure stdout/stderr in-place so emoji in log
# messages never raise UnicodeEncodeError. No-op on Linux/macOS.
if sys.platform == 'win32':
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# ── Config + manager wiring ───────────────────────────────────────────
import managers.settings_manager as _settings_mod
from managers.settings_manager import config, on_settings_changed, get_persona_name

from core.state import state
from core.agent_controller import AgentController
from core.agent_state import AgentState

_agent_state = AgentState()
controller   = AgentController(state, agent=_agent_state)

# Debug: print active configuration at startup
print('\n' + '=' * 50)
print('🔧 LOADED CONFIGURATION:')
print(f'   LLM:    {config.LLM_PROVIDER}/{config.LLM_MODEL}')
print(f'   Memory: {config.MEMORY_BACKEND}')
print(f'   TTS:    {config.TTS_PROVIDER}')
print(f'   STT:    {config.STT_PROVIDER}')
print('=' * 50 + '\n')

# ── External settings-change callback (import/reset only, not normal saves) ───
def _on_settings_changed():
    """
    Triggered only on external config import or reset — NOT on normal
    in-UI saves (those use silent=True to avoid double-reload).
    """
    logger.info('🔄 Settings changed externally — scheduling component reload')

    def _reload():
        import time as _t; _t.sleep(0.3)   # let caller finish writing config
        state.reload_llm()
        state.reload_memory()
        if state.conv_audio and state.conv_audio.is_listening:
            try: state.conv_audio.stop_conversation()
            except Exception: pass
        state.reload_audio(force=True)
        state.reload_vision()
        logger.info('🔄 External reload complete')

    threading.Thread(target=_reload, daemon=True).start()

on_settings_changed(_on_settings_changed)


# ══════════════════════════════════════════════════════════════════════
#  3.  ROUTE REGISTRATION
#      Importing `pages` triggers every @ui.page(...) decorator across
#      all modules in the pages/ package — all routes go live here.
# ══════════════════════════════════════════════════════════════════════
import pages   # noqa: F401


# ── Debug endpoint: raw _fetch_all() data for the cognitive dashboard ──
# Exposes exactly which tile keys have data and which are missing, bypassing
# the UI layer so tile failures can be diagnosed in-process. Localhost only
# (server host is pinned to 127.0.0.1 below).
from nicegui import app as _nicegui_app
import asyncio as _asyncio

# Optional React interface shares the existing runtime and remains opt-in.
from core.interface_api import router as _interface_router
from pathlib import Path as _Path
from starlette.staticfiles import StaticFiles as _StaticFiles
_nicegui_app.include_router(_interface_router)
_interface_dist = _Path(__file__).parent / 'frontend' / 'dist'
if _interface_dist.is_dir():
    _nicegui_app.mount('/next', _StaticFiles(directory=str(_interface_dist), html=True), name='interface')


@_nicegui_app.get('/diag-dashboard')
async def _diag_dashboard():
    from pages.cognitive_dashboard_page import _fetch_all
    try:
        data = await _asyncio.to_thread(_fetch_all)
        return {
            'ok': True,
            'key_count': len(data),
            'keys': sorted(data.keys()),
            'loop_cycle': data.get('slow_cycle'),
        }
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


@_nicegui_app.get('/diag-dashboard-detail')
async def _diag_dashboard_detail():
    """Full _fetch_all() payload (values included) for deep inspection."""
    from pages.cognitive_dashboard_page import _fetch_all
    try:
        data = await _asyncio.to_thread(_fetch_all)
        return {'ok': True, 'data': data}
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


@_nicegui_app.post('/diag-trigger-chat')
async def _diag_trigger_chat():
    """One-shot: run a real chat turn through the full response path so the
    lazy response-path tiles (ICE, internal_cognitive_state,
    arbitration_learning) populate — same as a user message in the UI."""
    def _work():
        from core.state import state as _st
        org = getattr(getattr(_st, 'persona', None), '_organism', None)
        if org is None:
            return {'ok': False, 'error': 'no organism'}
        fn = getattr(org, 'respond', None) or getattr(org, 'generate_response', None)
        if fn is None:
            return {'ok': False, 'error': 'no respond() method found'}
        try:
            resp = fn(f"Hi {get_persona_name()} — quick dashboard test, just say hi back.")
            return {'ok': True, 'response_preview': str(resp)[:200]}
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    return await _asyncio.to_thread(_work)


@_nicegui_app.get('/diag-ics')
async def _diag_ics():
    """Definitive check for the internal_cognitive_state tile:
    report ICS history length, run deliberate() once directly, report
    whether it early-returned or updated the history."""
    def _work():
        from core.state import state as _st
        org = getattr(getattr(_st, 'persona', None), '_organism', None)
        if org is None:
            return {'ok': False, 'error': 'no organism'}
        from cognition.internal_cognitive_state import get_internal_cognitive_state
        ics = get_internal_cognitive_state(org)
        before = len(ics.history)
        try:
            from cognition.recursive_deliberation import (
                deliberate, _gather_live_goals, _gather_live_percepts, _gather_live_symbols)
            gw = getattr(org, 'workspace', None)
            goals = _gather_live_goals(org)
            percepts = _gather_live_percepts(org)
            symbols = _gather_live_symbols(org)
            result = {'gw_exists': gw is not None,
                      'live_goals': len(goals),
                      'percepts': len(percepts),
                      'symbols': len(symbols),
                      'would_early_return': (gw is None) or (not goals and not percepts and not symbols),
                      'ics_history_before': before}
            summary = deliberate(org, 'diag-ics-check')
            result['deliberate_summary'] = summary
            result['ics_history_after'] = len(ics.history)
            result['ok'] = True
            return result
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    return await _asyncio.to_thread(_work)


@_nicegui_app.get('/diag-loop-attrs')
async def _diag_loop_attrs():
    """Introspect the live loop/organism objects to find where the
    dashboard's expected engines actually live (or why they're absent)."""
    def _work():
        from core.state import state as _st
        out = {}
        persona = getattr(_st, 'persona', None)
        org = getattr(persona, '_organism', None) if persona else None
        ai = getattr(persona, 'ai_system', None) if persona else None
        loop = getattr(org, '_loop', None) if org else None
        out['has_persona'] = persona is not None
        out['has_org'] = org is not None
        out['org_type'] = type(org).__name__ if org else None
        out['has_ai'] = ai is not None
        out['has_loop'] = loop is not None
        out['loop_type'] = type(loop).__name__ if loop else None
        expected = [
            '_consequence_model', '_world_self_dynamics', '_cross_layer_feedback',
            '_motivational_field', '_resource_economy', '_temporal_projection',
            '_aspiration_synthesis', '_generative_aspiration', '_flux_mind_model',
            '_causal_mechanism', '_long_horizon_planner', '_narrative_compression',
            '_meta_learning_audit', '_structural_coupling', '_counterfactual_sim',
            '_introspective_observer', '_temporal_self_projection',
            '_arbitration_runs', '_arbitration_overrides', '_arbitration_skipped',
            '_slow_cycle_count',
        ]
        if loop is not None:
            loop_attrs = {a: type(getattr(loop, a)).__name__ for a in dir(loop)
                          if a.startswith('_') and a not in ('__class__', '__dict__', '__weakref__')}
            out['loop_private_attrs'] = sorted(loop_attrs)
            out['loop_expected_missing'] = [e for e in expected if not hasattr(loop, e)]
        if org is not None:
            org_attrs = {a: type(getattr(org, a)).__name__ for a in dir(org)
                         if a.startswith('_') and a not in ('__class__', '__dict__', '__weakref__')}
            out['org_private_attrs'] = sorted(org_attrs)
            out['org_expected_present'] = [e for e in expected if hasattr(org, e)]
        if ai is not None:
            ai_attrs = [a for a in dir(ai) if a.startswith('_') and a not in ('__class__', '__dict__', '__weakref__')]
            out['ai_private_attrs'] = sorted(ai_attrs)
        return out
    return await _asyncio.to_thread(_work)


# ══════════════════════════════════════════════════════════════════════
#  4.  SERVER START
# ══════════════════════════════════════════════════════════════════════
from nicegui import ui, app

if __name__ in {'__main__', '__mp_main__'}:

    # ── Pre-warm brain on server startup — before any browser connects ──
    async def _on_startup():
        """
        Boot the full cognitive organism at server start.
        By the time a browser opens, state.ready is True and the
        page renders immediately — no splash screen wait.
        """
        if not state.ready and not state.initializing:
            logger.info('🧠 Pre-warming Lumina brain on startup…')
            await state.initialize()

            try:
                from managers.settings_manager import config as _cfg
                _warnings = security.audit_config(
                    _cfg.model_dump() if hasattr(_cfg, 'model_dump') else _cfg.dict()
                )
                if _warnings:
                    logger.warning(f'Security: {len(_warnings)} config issue(s) — check logs')
            except Exception:
                pass

            if state.orchestrator is not None:
                import asyncio as _asyncio
                _asyncio.create_task(state.orchestrator.start())
                logger.info('🧠 Orchestrator started')

            if config.CAMERA_AUTOSTART and state.vision:
                import asyncio as _asyncio
                await _asyncio.to_thread(state.vision.start_camera)

            logger.info('✅ Brain pre-warm complete — ready for connections')

    from managers.security_manager import security
    app.on_startup(_on_startup)

    # ── Graceful shutdown ─────────────────────────────────────────────
    def _on_shutdown():
        logger.info('App shutting down — saving session...')
        try:
            organism = getattr(getattr(state, 'persona', None), '_organism', None)
            if organism is not None and hasattr(organism, 'shutdown'):
                organism.shutdown()
            else:
                loop = getattr(organism, '_loop', None) if organism else None
                if loop is not None and hasattr(loop, 'shutdown'):
                    loop.shutdown()
        except Exception as exc:
            logger.warning('Background loop shutdown save failed: %s', exc)
        from managers.session_manager import get_session_manager
        sess = get_session_manager()
        if state and state.llm:
            sess.on_shutdown(state.llm.history)
        try:
            from cognition.research_mcp.stealth_browser import _close_browser
            import asyncio as _asyncio
            _asyncio.run(_close_browser())
        except Exception:
            pass
        logger.info('Shutdown complete.')

    app.on_shutdown(_on_shutdown)

    # ── Init search backend with saved config keys ────────────────────
    def _init_search_backend():
        try:
            from cognition.research_mcp.web_agent import set_search_backend
            set_search_backend(
                backend       = getattr(config, 'RESEARCH_SEARCH_BACKEND', 'auto'),
                anthropic_key = getattr(config, 'ANTHROPIC_API_KEY', ''),
                brave_key     = getattr(config, 'BRAVE_SEARCH_KEY', ''),
                serpapi_key   = getattr(config, 'SERPAPI_KEY', ''),
                use_rss       = True,
                use_stealth   = True,
            )
            logger.info('✅ Search backend initialised at startup')
        except Exception as e:
            logger.warning(f'Search backend init failed: {e}')

    _init_search_backend()

    # ── Launch ────────────────────────────────────────────────────────
    ui.run(
        title=f'{get_persona_name()} — Embodied AI',
        host=getattr(config, 'NICEGUI_HOST', '127.0.0.1'),
        port=config.NICEGUI_PORT,
        reload=False,
        show=False,
        favicon='✨',
        dark=True,
        storage_secret=os.environ.get('LUMINA_SESSION_SECRET'),
        binding_refresh_interval=0.5,
    )
