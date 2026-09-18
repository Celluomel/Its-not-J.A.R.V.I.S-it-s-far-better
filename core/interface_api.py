"""Incremental frontend adapter. Owns no cognitive engines or NiceGUI elements."""
import asyncio
import faulthandler
from collections import deque
import json
import logging
from pathlib import Path
import tempfile
import time
import uuid
import ipaddress
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_turn_lock = asyncio.Lock()
_audio_lock = asyncio.Lock()
_workers = set()
_events = deque(maxlen=40)
_presence_messages = deque(maxlen=40)
_timings = {}
_dialogues = {}


def _local_request(request: Request):
    # Browser requests must originate from this local application, including Vite.
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, 'Cross-site requests are not allowed')
    origin = request.headers.get('origin')
    if origin:
        hostname = (urlparse(origin).hostname or '').lower()
        allowed = hostname in {'localhost', '127.0.0.1', '::1'}
        if not allowed:
            from managers.settings_manager import config
            if getattr(config, 'NICEGUI_HOST', '127.0.0.1') == '0.0.0.0':
                try:
                    allowed = ipaddress.ip_address(hostname).is_private
                except ValueError:
                    allowed = False
        if not allowed:
            raise HTTPException(403, 'Local application origin required')


router = APIRouter(prefix='/api/interface', dependencies=[Depends(_local_request)])


class BodySettingsUpdate(BaseModel):
    values: dict[str, object] = Field(default_factory=dict)


def _runtime():
    from core.state import state
    return state


def _body_runtime_for_state():
    """Return the body even while the cognitive organism is booting.

    Body configuration and plugin discovery are intentionally independent from
    brain readiness.  A brain-attached runtime is reused when available;
    otherwise a standalone runtime is kept on the shared state until the brain
    attaches to it.
    """
    state = _runtime()
    organism = getattr(getattr(state, 'persona', None), '_organism', None)
    if organism is not None:
        from cognition.body_runtime import get_body_runtime
        body = get_body_runtime(organism)
        state.body_runtime = body
        return body, organism

    body = getattr(state, 'body_runtime', None)
    if body is None:
        from cognition.body_runtime import get_body_runtime
        body = get_body_runtime()
        state.body_runtime = body
        logger.info('[BodyRuntime] Standalone body started while brain is unavailable')
    return body, None


def _event(label, turn_id=None):
    _events.append({'id': str(uuid.uuid4()), 'label': label, 'timestamp': time.time(), 'turn_id': turn_id})


def record_presence(text: str):
    """Publish a proactive PresenceEngine utterance to interface clients."""
    # Presence may finish an older camera-triggered reaction while a user turn
    # is already in progress. Never inject that message into the active
    # transcript; it can make a normal response look interrupted.
    if _turn_lock.locked():
        logger.debug('Suppressed proactive presence utterance during interactive turn')
        return
    if text and text.strip():
        _presence_messages.append({'id': str(uuid.uuid4()), 'text': text.strip(), 'timestamp': time.time()})
        _event('Proactive presence utterance')


def interactive_turn_active() -> bool:
    """Return whether the incremental interface owns the chat turn."""
    return _turn_lock.locked()


def _json_safe(value, depth=0):
    """Keep diagnostic endpoints serializable without leaking engine objects."""
    if depth > 5:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in list(value)[:100]]
    return str(value)


def _cognitive_telemetry_snapshot():
    """Return the live, user-visible cognitive trace for the chat surface."""
    state = _runtime()
    persona = getattr(state, 'persona', None)
    organism = getattr(persona, '_organism', None) if persona else None
    loop = getattr(organism, '_loop', None) if organism else None
    planner = getattr(loop, '_long_horizon_planner', None) if loop else None

    # The slow loop normally creates the planner after its warm-up period. The
    # telemetry control must also work before that threshold and after a fresh
    # restart, so initialise the same persisted planner on demand.
    if planner is None and organism is not None and loop is not None:
        ai_system = getattr(organism, 'ai_system', None)
        if ai_system is not None:
            try:
                from cognition.long_horizon_planner import LongHorizonPlanner
                planner = LongHorizonPlanner(organism, ai_system)
                loop._long_horizon_planner = planner
                logger.info('LongHorizonPlanner initialised for cognitive telemetry')
            except Exception as exc:
                logger.debug('Cognitive telemetry planner unavailable: %s', exc)

    if planner is None or not hasattr(planner, 'factual_self_report_snapshot'):
        return None
    try:
        return _json_safe(planner.factual_self_report_snapshot())
    except Exception as exc:
        logger.debug('Cognitive telemetry snapshot unavailable: %s', exc)
        return None


def _orchestrator_snapshot():
    """Expose a small, serializable view of the live autonomous scheduler.

    This is deliberately observational: the interface may render the state, but
    never controls the orchestrator through the routine health refresh.
    """
    try:
        orchestrator = getattr(_runtime(), 'orchestrator', None)
        if orchestrator is None:
            return {'available': False, 'running': False}
        status = orchestrator.status()
        pending = orchestrator.event_system.peek()
        return _json_safe({
            'available': True,
            'running': status.get('running', False),
            'cycle_count': status.get('cycle_count', 0),
            'last_activity': status.get('last_activity', 'none'),
            'drives': status.get('drives', {}),
            'clock': status.get('clock', {}),
            'workspace': status.get('workspace', {}),
            'queued_events': len(pending),
            'pending_event_types': [event.type for event in pending[:6]],
        })
    except Exception as exc:
        logger.debug('Orchestrator snapshot unavailable: %s', exc)
        return {'available': False, 'running': False}


@router.get('/status')
async def status():
    state = _runtime()
    persona = state.persona
    org = getattr(persona, '_organism', None)
    loop = getattr(org, '_loop', None)
    from managers.settings_manager import config
    audio = getattr(state, 'audio', None)
    active_tts = (audio.tts_engine or {}).get('type') if audio else None
    active_stt = (audio.stt_engine or {}).get('type') if audio else None
    context = (
        persona.get_prompt_context('default')
        if persona and persona.is_ready and hasattr(persona, 'get_prompt_context')
        else {}
    )
    system = getattr(persona, '_system', None)
    body = getattr(org, '_body_runtime', None) if org is not None else getattr(state, 'body_runtime', None)
    return {
        'version': 1,
        'ready': bool(state.ready and persona and persona.is_ready),
        'busy': _turn_lock.locked(),
        'chat_stage': getattr(persona, '_chat_stage', 'unknown') if _turn_lock.locked() else 'idle',
        'timestamp': time.time(),
        'uptime': round(state.get_uptime()),
        'cycle': getattr(loop, '_slow_cycle_count', None),
        'engines': {'persona': persona is not None, 'organism': org is not None, 'loop': loop is not None},
        'providers': {
            'stt': active_stt or config.STT_PROVIDER,
            'tts': active_tts or config.TTS_PROVIDER,
            'configured_stt': config.STT_PROVIDER,
            'configured_tts': config.TTS_PROVIDER,
        },
        'response_language': getattr(config, 'RESPONSE_LANGUAGE', 'auto'),
        'web_search_mode': getattr(config, 'WEB_SEARCH_MODE', 'off'),
        'response_verbosity': getattr(config, 'RESPONSE_VERBOSITY', 'concise'),
        'emotional_state': context.get('emotional_state', ''),
        'life_stage': context.get('life_stage', ''),
        'current_age': getattr(system, 'current_age', None),
        'timings': dict(_timings),
        'events': list(_events),
        'presence_messages': list(_presence_messages),
        'body_runtime': body.status() if body is not None else {'available': False, 'running': False},
    }


@router.get('/body/status')
async def body_status():
    body, _ = _body_runtime_for_state()
    return _json_safe({
        'status': body.status(),
        'plugins': body.plugins(),
        'observations': body.snapshot(),
        'recent_events': body.recent_events(),
    })


@router.get('/body/plugins')
async def body_plugins():
    body, _ = _body_runtime_for_state()
    return _json_safe({'plugins': body.plugins(), 'config_path': 'data/body/config.json'})


@router.get('/body/settings')
async def body_settings():
    body, _ = _body_runtime_for_state()
    values = body.config_snapshot()
    for secret_name in ('HOME_ASSISTANT_TOKEN', 'BODY_BRIDGE_TOKEN'):
        if values.get(secret_name):
            values[secret_name] = '••••••••'
    return _json_safe({'values': values, 'config_path': 'data/body/config.json'})


@router.post('/body/settings')
async def update_body_settings(payload: BodySettingsUpdate):
    body, organism = _body_runtime_for_state()
    values = dict(payload.values)
    for secret_name in ('HOME_ASSISTANT_TOKEN', 'BODY_BRIDGE_TOKEN'):
        if values.get(secret_name) == '••••••••':
            values.pop(secret_name, None)
    saved = body.update_config(values)
    if organism is not None:
        from cognition.universal_connector import get_universal_connector
        await get_universal_connector(organism).reconcile_home_assistant_monitor()
    for secret_name in ('HOME_ASSISTANT_TOKEN', 'BODY_BRIDGE_TOKEN'):
        if saved.get(secret_name):
            saved[secret_name] = '••••••••'
    return _json_safe({'values': saved, 'config_path': 'data/body/config.json'})


@router.post('/body/discover')
async def discover_body_home_assistant():
    _, organism = _body_runtime_for_state()
    if organism is None:
        raise HTTPException(503, 'Body is ready; Home Assistant discovery is waiting for the cognitive connector.')
    from cognition.universal_connector import get_universal_connector
    result = await asyncio.to_thread(get_universal_connector(organism).discover_home_assistant)
    if result.get('ok'):
        from cognition.body_runtime import get_body_runtime
        get_body_runtime(organism).update_config({
            'HOME_ASSISTANT_DISCOVERED_ENTITIES': json.dumps(result.get('entities', []), ensure_ascii=False),
        })
    return _json_safe(result)


@router.websocket('/body/bridge')
async def body_bridge(websocket: WebSocket):
    """Receive authenticated observations from a remote Body Runtime."""
    body, _ = _body_runtime_for_state()
    from managers.settings_manager import config
    expected = str(body.config_value('BODY_BRIDGE_TOKEN', getattr(config, 'BODY_BRIDGE_TOKEN', '')) or '')
    supplied = websocket.headers.get('authorization', '')
    token = supplied[7:].strip() if supplied.lower().startswith('bearer ') else ''
    if not expected or token != expected:
        await websocket.close(code=1008, reason='Body bridge authentication failed')
        return
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            if message.get('type') == 'hello':
                await websocket.send_json({'type': 'hello_ack', 'protocol': 1, 'brain': 'ready'})
                continue
            if message.get('type') != 'observation' or not isinstance(message.get('observation'), dict):
                await websocket.send_json({'type': 'error', 'reason': 'unsupported message'})
                continue
            item = message['observation']
            from cognition.body_runtime import BodyObservation
            observation = BodyObservation(
                source=f"remote:{message.get('device_id') or 'body'}:{item.get('source') or 'unknown'}",
                kind=str(item.get('kind') or 'sensor'),
                subject=str(item.get('subject') or 'observation'),
                value=item.get('value'),
                unit=str(item.get('unit') or ''),
                confidence=float(item.get('confidence', 1.0)),
                observed_at=float(item.get('observed_at') or time.time()),
                provenance=dict(item.get('provenance') or {}),
            )
            body.publish_observation(observation, forward=False)
            await websocket.send_json({'type': 'observation_ack', 'subject': observation.subject})
    except WebSocketDisconnect:
        logger.info('[BodyBridge] remote Body disconnected')


@router.get('/telemetry')
async def cognitive_telemetry():
    """Fetch the current trace when its chat visibility control is enabled."""
    snapshot = _cognitive_telemetry_snapshot()
    return snapshot or {
        'captured_at': time.time(),
        'active_goals': [],
        'dominant_plan': None,
    }


@router.get('/lumina')
async def lumina_status():
    state = _runtime()
    persona = state.persona
    if not persona:
        raise HTTPException(503, 'Lumina is not ready.')
    try:
        return _json_safe(await asyncio.to_thread(persona.get_system_status))
    except Exception as exc:
        logger.warning('Lumina status read failed: %s', exc)
        raise HTTPException(503, 'Lumina status is unavailable.')


class LuminaAction(BaseModel):
    action: str = Field(pattern=r'^(dream|learning|life_event|feedback_positive|feedback_negative)$')
    event_type: str = Field(default='creative', max_length=40)


@router.post('/lumina/action')
async def lumina_action(payload: LuminaAction):
    state = _runtime()
    persona = state.persona
    if not persona:
        raise HTTPException(503, 'Lumina is not ready.')
    try:
        if payload.action == 'dream':
            result = await asyncio.to_thread(persona.trigger_dream)
        elif payload.action == 'learning':
            result = await asyncio.to_thread(persona.trigger_learning)
        elif payload.action == 'life_event':
            result = await asyncio.to_thread(persona.trigger_life_event, payload.event_type)
        else:
            result = await asyncio.to_thread(persona.receive_feedback, payload.action == 'feedback_positive', 'default', '')
        _event(f'Lumina action: {payload.action}')
        return {'ok': True, 'action': payload.action, 'result': _json_safe(result)}
    except Exception as exc:
        logger.warning('Lumina action failed: %s', exc)
        raise HTTPException(500, 'Lumina action failed.')


class ResponseFeedback(BaseModel):
    positive: bool
    response: str = Field(default='', max_length=8000)
    message_id: str = Field(default='', max_length=120)


@router.post('/feedback')
async def response_feedback(payload: ResponseFeedback):
    """Record explicit feedback for the exact assistant response being rated."""
    state = _runtime()
    persona = state.persona
    if not persona:
        raise HTTPException(503, 'Lumina is not ready.')
    try:
        from managers.user_manager import user_manager
        result = await asyncio.to_thread(
            persona.receive_feedback,
            payload.positive,
            user_manager.active_id,
            payload.response,
        )
        if not result.get('feedback_received', False):
            raise RuntimeError(result.get('error', 'Feedback was not accepted.'))
        _event(
            f"Response feedback: {'positive' if payload.positive else 'negative'}",
        )
        return {'ok': True, 'message_id': payload.message_id, 'result': _json_safe(result)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning('Response feedback failed: %s', exc)
        raise HTTPException(500, 'Response feedback could not be recorded.')


@router.get('/settings/analytics')
async def settings_analytics():
    state = _runtime()
    org = getattr(getattr(state, 'persona', None), '_organism', None)
    obs = getattr(org, 'observatory', None) if org else None
    history = []
    if obs:
        try:
            history = [_json_safe(item.__dict__ if hasattr(item, '__dict__') else item) for item in obs.history_list(200)]
        except Exception:
            history = []
    log_path = Path('data/persona/ai_system.log')
    lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-500:] if log_path.exists() else []
    semantic = getattr(org, 'semantic_memory', None) if org else None
    memory_summary = {}
    capabilities = {}
    if semantic:
        try: memory_summary = _json_safe(semantic.summary())
        except Exception: pass
        try: capabilities = _json_safe(semantic.get_capabilities())
        except Exception: pass
    trends = {}
    for key in ('ccs', 'rdi', 'gei', 'idx', 'str_score'):
        values = [float(item.get(key, 0) or 0) for item in history if isinstance(item, dict)]
        if values: trends[key] = {'current': values[-1], 'average': sum(values) / len(values), 'minimum': min(values), 'maximum': max(values), 'delta': values[-1] - values[0]}
    return {'history': history, 'trends': trends, 'memory': memory_summary, 'capabilities': capabilities, 'log': {'available': log_path.exists(), 'lines': len(lines), 'errors': sum(' - ERROR - ' in line for line in lines), 'warnings': sum(' - WARNING - ' in line for line in lines), 'infos': sum(' - INFO - ' in line for line in lines), 'recent_errors': [line[-240:] for line in lines if ' - ERROR - ' in line][-5:]}}


@router.post('/settings/analytics/self-analysis')
async def analytics_self_analysis():
    state = _runtime()
    if not getattr(state, 'llm', None) or not hasattr(state.llm, 'generate_bare'):
        raise HTTPException(503, 'LLM self-analysis is unavailable.')
    snapshot = await settings_analytics()
    prompt = 'Analyze these Lumina cognitive telemetry metrics in 3 concise paragraphs. Mention trends, anomalies, and one practical observation. Do not invent missing data. Telemetry:\n' + json.dumps(snapshot, default=str)[:7000]
    try:
        result = await asyncio.to_thread(state.llm.generate_bare, prompt, max_tokens=300, temperature=0.35)
        return {'analysis': str(result).strip()}
    except Exception as exc:
        logger.warning('Self-analysis failed: %s', exc)
        raise HTTPException(500, 'Self-analysis failed.')


@router.get('/settings/tools')
async def settings_tools():
    state = _runtime()
    org = getattr(getattr(state, 'persona', None), '_organism', None)
    from managers.settings_manager import config
    from managers.security_manager import security
    from managers.vision_manager import FACE_RECOGNITION_AVAILABLE, FACE_RECOGNITION_ERROR
    try:
        cfg = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        security_warnings = security.audit_config(cfg)
    except Exception as exc:
        security_warnings = [str(exc)]
    mcp_error = ''
    try:
        from cognition.research_mcp import ResearchMCP
        mcp_available = True
    except Exception as exc:
        mcp_available = False
        mcp_error = str(exc)
    vision = getattr(state, 'vision', None)
    vision_resolution = f'{getattr(vision, "frame_width", 640)}x{getattr(vision, "frame_height", 480)}' if vision else 'Unavailable'
    effective_resolution = 'Unavailable'
    if vision is not None:
        frame = vision.get_current_frame()
        if frame is not None and len(frame.shape) >= 2:
            effective_resolution = f'{frame.shape[1]}x{frame.shape[0]}'
    return _json_safe({'observatory': {'enabled': getattr(config, 'OBSERVATORY_ENABLED', False), 'active': bool(getattr(org, 'observatory', None)), 'baseline_samples': getattr(config, 'OBSERVATORY_BASELINE_SAMPLES', 20), 'emergence_threshold': getattr(config, 'OBSERVATORY_EMERGENCE_THRESHOLD', 0.5)}, 'brain_visualizer': {'available': Path('brain_visualizer').exists(), 'dependencies': 'PyOpenGL + glfw', 'networks': ['ECN', 'DMN', 'SN', 'SCN', 'RMS', 'MS', 'SIS']}, 'research': {'mcp_available': mcp_available, 'mcp_error': mcp_error, 'backend': getattr(config, 'RESEARCH_SEARCH_BACKEND', 'auto'), 'brave_configured': bool(getattr(config, 'BRAVE_SEARCH_KEY', ''))}, 'security': {'local_only': True, 'warnings': security_warnings, 'audit_events': security.recent_audit(15)}, 'vision': {'initialized': bool(vision), 'camera_active': bool(getattr(vision, 'camera_active', False)), 'face_detection': bool(getattr(vision, 'face_detection_enabled', False)), 'face_recognition_available': FACE_RECOGNITION_AVAILABLE, 'face_recognition_error': FACE_RECOGNITION_ERROR, 'known_faces': len(getattr(vision, 'face_encodings', {})), 'vision_memories': len(getattr(vision, 'vision_meta', [])), 'resolution': vision_resolution, 'effective_resolution': effective_resolution, 'model': getattr(config, 'LAVA_MODEL', ''), 'llm_mode': getattr(config, 'VISION_LLM_MODE', 'separate'), 'vision_supported': bool(getattr(vision, 'vision_supported', False)), 'dedicated_model_ready': bool(getattr(vision, 'lava_model', '')), 'ambient_interval': getattr(config, 'AMBIENT_VISION_INTERVAL', 90)}, 'engines': {'persona': bool(getattr(state, 'persona', None)), 'organism': bool(org), 'research': bool(getattr(getattr(state, 'persona', None), '_research', None)), 'vision': bool(vision), 'memory': bool(getattr(org, 'semantic_memory', None) if org else None)}, 'events': list(_events)[-12:]})


class RSSFeedCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=30)


class RSSFeedUpdate(BaseModel):
    active: bool | None = None
    tags: list[str] | None = Field(default=None, max_length=30)


def _rss_feed_dict(feed):
    return _json_safe(feed.to_dict())


@router.get('/rss-feeds')
async def rss_feeds():
    """Return the persisted RSS registry used by the research pipeline."""
    from cognition.research_mcp.rss_registry import get_rss_registry
    return {'feeds': [_rss_feed_dict(feed) for feed in get_rss_registry().list_all()]}


@router.post('/rss-feeds')
async def add_rss_feed(payload: RSSFeedCreate):
    from cognition.research_mcp.rss_registry import get_rss_registry
    url = payload.url.strip()
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(400, 'RSS URL must start with http:// or https://')
    feed = get_rss_registry().add_feed(url, payload.name, payload.tags)
    _event(f'RSS feed added: {feed.name}')
    return {'feed': _rss_feed_dict(feed)}


@router.patch('/rss-feeds/{feed_id}')
async def update_rss_feed(feed_id: str, payload: RSSFeedUpdate):
    from cognition.research_mcp.rss_registry import get_rss_registry
    registry = get_rss_registry()
    changed = False
    if payload.active is not None:
        changed = registry.set_active(feed_id, payload.active) or changed
    if payload.tags is not None:
        changed = registry.set_tags(feed_id, payload.tags) or changed
    if not changed:
        raise HTTPException(404, 'RSS feed not found')
    feed = next((item for item in registry.list_all() if item.feed_id == feed_id), None)
    _event(f'RSS feed updated: {feed.name if feed else feed_id}')
    return {'feed': _rss_feed_dict(feed)} if feed else {'ok': True}


@router.post('/rss-feeds/{feed_id}/reset')
async def reset_rss_feed(feed_id: str):
    from cognition.research_mcp.rss_registry import get_rss_registry
    registry = get_rss_registry()
    if not registry.reset_stats(feed_id):
        raise HTTPException(404, 'RSS feed not found')
    return {'ok': True}


@router.delete('/rss-feeds/{feed_id}')
async def delete_rss_feed(feed_id: str):
    from cognition.research_mcp.rss_registry import get_rss_registry
    if not get_rss_registry().remove_feed(feed_id):
        raise HTTPException(404, 'RSS feed not found')
    _event(f'RSS feed removed: {feed_id}')
    return {'ok': True}


@router.get('/cognitive-health')
async def cognitive_health():
    try:
        # Reuse the dashboard's canonical collector.  The general persona
        # status is intentionally small and does not include the 23 learning
        # and self-observation panels shown by the legacy dashboard.
        from pages.cognitive_dashboard_page import _fetch_all, _compute_self_awareness_index
        data = await asyncio.to_thread(_fetch_all)
        if isinstance(data, dict):
            data['orchestrator'] = _orchestrator_snapshot()
            index = _compute_self_awareness_index(data)
            data['self_awareness_index'] = (
                {'score': index[0], 'components': [
                    {'name': name, 'value': value, 'weight': weight}
                    for name, value, weight in index[1]
                ], 'interpretation': 'Heuristic activity index, not a consciousness measurement.'}
                if index else {'status': 'Insufficient recorded activity'}
            )
        return data if isinstance(data, dict) else {'error': 'Invalid cognitive status.'}
    except Exception as exc:
        logger.warning('Cognitive health read failed: %s', exc)
        raise HTTPException(503, 'Cognitive dashboard is unavailable.')


@router.get('/network')
async def network_status():
    state = _runtime()
    from managers.settings_manager import config
    network = getattr(state, 'lumina_network', None)
    configured = getattr(config, 'LUMINA_CHILDREN', []) or []
    children = network.list_children() if network else [
        {'id': c.get('id', 'child'), 'name': c.get('name', 'Flux'), 'url': c.get('url', ''),
         'role': c.get('role', 'child'), 'online': False, 'latency_ms': 0, 'turn_count': 0}
        for c in configured
    ]
    return {'enabled': bool(getattr(config, 'LUMINA_NETWORK_ENABLED', False)), 'children': children}


@router.post('/connector/home-assistant/discover')
async def discover_home_assistant():
    """Test the configured Home Assistant bridge and list allowed entities."""
    state = _runtime()
    organism = getattr(getattr(state, 'persona', None), '_organism', None)
    if organism is None:
        raise HTTPException(503, 'Cognitive organism is not ready.')
    from cognition.universal_connector import get_universal_connector
    result = await asyncio.to_thread(get_universal_connector(organism).discover_home_assistant)
    if result.get('ok'):
        from managers.settings_manager import config, save_settings
        config.HOME_ASSISTANT_DISCOVERED_ENTITIES = json.dumps(result.get('entities', []), ensure_ascii=False)
        save_settings(config, silent=True)
    return _json_safe(result)


class NetworkAction(BaseModel):
    child_id: str = Field(default='', max_length=80)
    text: str = Field(default='', max_length=8000)
    topic: str = Field(default='', max_length=800)


class LanguageUpdate(BaseModel):
    language: str = Field(default='auto', pattern=r'^[a-z]{2,5}(?:-[a-z]{2,4})?$')


class PreferenceUpdate(BaseModel):
    web_search_mode: str | None = Field(default=None, pattern=r'^(off|auto|always)$')
    response_verbosity: str | None = Field(default=None, pattern=r'^(concise|verbose)$')


class SettingsUpdate(BaseModel):
    values: dict[str, object] = Field(default_factory=dict)


_SETTING_FIELDS = {
    'PERSONA_NAME', 'LOCATION_NAME', 'LOCATION_ADDRESS', 'LOCATION_LATITUDE', 'LOCATION_LONGITUDE',
    'LLM_PROVIDER', 'LLM_MODEL', 'TEXT_MODEL', 'QUALITY_EMBED_MODEL', 'EMBED_API_BASE_URL',
    'AFFECT_EMBEDDING_MODEL', 'LLM_BASE_URL',
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'RESEARCH_SEARCH_BACKEND',
    'BRAVE_SEARCH_KEY', 'SERPAPI_KEY', 'MEMORY_BACKEND', 'MEMORY_DB_PATH',
    'MEMORY_FAISS_PATH', 'MEMORY_WORLD_PATH', 'MEMORY_PERSONA_PATH',
    'MEMORY_COGNEE_PATH', 'MEMORY_COGNEE_EMBED_MODEL', 'TTS_PROVIDER',
    'VOICEMEM_ENABLED', 'VOICEMEM_DATA_PATH', 'VOICEMEM_TOP_K', 'VOICEMEM_LOCAL_MODE',
    'STT_PROVIDER', 'WHISPER_MODEL', 'ELEVENLABS_API_KEY', 'ELEVENLABS_VOICE_ID', 'COQUI_VOICE_REFERENCE',
    'VOICE_LANGUAGE', 'RESPONSE_LANGUAGE', 'VAD_AGGRESSIVENESS',
    'VAD_ONSET_CHUNKS', 'VAD_SILENCE_DURATION', 'VAD_MIN_SPEECH_DURATION',
    'VAD_ENERGY_GATE_FACTOR', 'PARTIAL_STT_ENABLED', 'VOICE_MAX_TOKENS', 'TTS_STREAMING', 'TTS_BARGE_IN',
    'BARGE_IN_SENSITIVITY', 'INTER_SENTENCE_PAUSE_MS', 'TTS_POST_ROLL_MS',
    'CAMERA_AUTOSTART', 'CAMERA_ID', 'CAMERA_FPS', 'CAMERA_RESOLUTION',
    'VISION_MODE', 'LAVA_MODEL', 'VISION_LLM_MODE', 'AMBIENT_VISION_INTERVAL',
    'RESPONSE_VERBOSITY', 'RESPONSE_TOKENS_CONCISE', 'RESPONSE_TOKENS_VERBOSE',
    'CUSTOM_SYSTEM_PROMPT', 'LOG_FULL_PROMPTS', 'OBSERVATORY_ENABLED',
    'OBSERVATORY_BASELINE_SAMPLES', 'OBSERVATORY_EMERGENCE_THRESHOLD',
    'NICEGUI_HOST', 'UNIVERSAL_CONNECTOR_ENABLED', 'BODY_RUNTIME_ENABLED', 'BODY_PLUGIN_HOME_ASSISTANT_ENABLED',
    'BODY_HOST', 'BODY_PORT',
    'HOME_ASSISTANT_ENABLED', 'HOME_ASSISTANT_PRESENCE_ENABLED',
    'HOME_ASSISTANT_URL', 'HOME_ASSISTANT_TOKEN', 'HOME_ASSISTANT_VERIFY_SSL',
    'HOME_ASSISTANT_POLL_INTERVAL', 'HOME_ASSISTANT_ALLOWED_DOMAINS', 'HOME_ASSISTANT_SELECTED_ENTITIES', 'HOME_ASSISTANT_DISCOVERED_ENTITIES', 'HOME_ASSISTANT_ENTITY_TAGS',
    'BODY_BRIDGE_ENABLED', 'BODY_BRIDGE_URL', 'BODY_BRIDGE_TOKEN', 'BODY_BRIDGE_DEVICE_ID', 'BODY_BRIDGE_VERIFY_TLS', 'BODY_BRIDGE_RECONNECT_SECONDS',
}
_SECRET_FIELDS = {
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'BRAVE_SEARCH_KEY', 'SERPAPI_KEY',
    'ELEVENLABS_API_KEY', 'HOME_ASSISTANT_TOKEN', 'WHATSAPP_WEBHOOK_SECRET', 'BODY_BRIDGE_TOKEN',
}


# Development proposals are deliberately independent from chat state. They are
# a durable research backlog that can later be reviewed by the user or handed
# to a coding agent as an explicit implementation brief.
_PROPOSAL_PATH = Path('data/persona/capability_proposals.json')
_PROPOSAL_CATEGORIES = {'cognitive', 'tool', 'sensor', 'embodiment', 'learning', 'interaction', 'experiment'}
_PROPOSAL_STATUSES = {'draft', 'accepted', 'in_progress', 'completed', 'deferred', 'rejected'}


class CapabilityProposalCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(default='', max_length=8000)
    category: str = Field(default='cognitive', pattern=r'^(cognitive|tool|sensor|embodiment|learning|interaction|experiment)$')
    motivation: str = Field(default='', max_length=3000)
    expected_capability: str = Field(default='', max_length=3000)
    constraints: str = Field(default='', max_length=3000)
    success_criteria: str = Field(default='', max_length=3000)
    priority: int = Field(default=3, ge=1, le=5)
    source: str = Field(default='user', pattern=r'^(user|organism)$')


class CapabilityProposalUpdate(BaseModel):
    status: str | None = Field(default=None, pattern=r'^(draft|accepted|in_progress|completed|deferred|rejected)$')
    user_evaluation: str | None = Field(default=None, max_length=3000)


def _load_capability_proposals():
    try:
        if _PROPOSAL_PATH.exists():
            value = json.loads(_PROPOSAL_PATH.read_text(encoding='utf-8'))
            return value if isinstance(value, list) else []
    except Exception as exc:
        logger.warning('Capability proposal store unreadable: %s', exc)
    return []


def _save_capability_proposals(items):
    _PROPOSAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = _PROPOSAL_PATH.with_suffix('.tmp')
    temp.write_text(json.dumps(items[-200:], ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(_PROPOSAL_PATH)


def _proposal_analysis(item):
    text = ' '.join(str(item.get(key, '')) for key in ('title', 'description', 'expected_capability')).lower()
    mapping = {
        'memory': ('memory', 'episodic and semantic memory'),
        'reason': ('reasoning', 'causal, temporal or abstract reasoning'),
        'plan': ('planning', 'intention and long-horizon planning'),
        'goal': ('goals', 'autonomous goal formation and execution'),
        'camera': ('vision', 'camera perception and visual grounding'),
        'face': ('vision', 'face recognition and identity grounding'),
        'sensor': ('embodiment', 'sensor integration and physical context'),
        'arduino': ('embodiment', 'hardware I/O and actuator control'),
        'robot': ('embodiment', 'robotic embodiment and action'),
        'voice': ('audio', 'speech recognition and synthesis'),
        'tool': ('tools', 'external tools and research actions'),
        'learn': ('learning', 'evidence-based adaptation and self-correction'),
    }
    affected = []
    for needle, result in mapping.items():
        if needle in text and result[0] not in [name for name, _ in affected]:
            affected.append(result)
    if not affected:
        affected = [('workspace', 'global workspace routing and observable cognitive state')]
    subsystem_names = ', '.join(name for name, _ in affected)
    outputs = '; '.join(description for _, description in affected)
    prompt = f'''Implement the capability proposal "{item['title']}" in the cognitive organism.

Objective:
{item.get('description') or 'Define and implement the proposed capability.'}

Category: {item.get('category', 'cognitive')}
Motivation: {item.get('motivation') or 'Make the organism more capable while preserving inspectability.'}
Expected capability: {item.get('expected_capability') or item.get('description') or 'A working, observable capability.'}
Affected subsystems: {subsystem_names} ({outputs})
Constraints: {item.get('constraints') or 'Keep existing chat, audio, perception and background loops working; keep slow work asynchronous.'}
Success criteria: {item.get('success_criteria') or 'Add focused tests, expose observable output, persist required state, and verify no regression in existing workflows.'}

Implementation requirements:
- Inspect the existing architecture before editing.
- Reuse existing managers, event streams and persistence patterns.
- Keep hardware or external-service integrations optional with a local/mock fallback.
- Separate proposal, experiment and verified capability states.
- Report files changed, tests run, evidence collected and unresolved risks.'''
    return {'affected_subsystems': [name for name, _ in affected], 'analysis': f'Likely affected: {subsystem_names}.', 'generated_prompt': prompt}


@router.get('/capability-proposals')
async def list_capability_proposals():
    return {'proposals': _json_safe(list(reversed(_load_capability_proposals())))}


@router.post('/capability-proposals')
async def create_capability_proposal(payload: CapabilityProposalCreate):
    item = payload.model_dump()
    item.update({'id': str(uuid.uuid4()), 'status': 'draft', 'created_at': time.time(), 'updated_at': time.time()})
    item.update(_proposal_analysis(item))
    proposals = _load_capability_proposals()
    proposals.append(item)
    _save_capability_proposals(proposals)
    _event(f'Capability proposal created: {payload.title[:80]}')
    return _json_safe(item)


@router.post('/capability-proposals/generate')
async def generate_capability_proposal():
    """Let Lumina suggest one bounded capability for the research backlog."""
    state = _runtime()
    organism = getattr(getattr(state, 'persona', None), '_organism', None)
    llm = getattr(state, 'llm', None)
    if organism is None or llm is None:
        raise HTTPException(503, 'Lumina cognitive engine is not ready.')
    workspace = getattr(organism, 'workspace', None)
    workspace_state = workspace.summary() if workspace and hasattr(workspace, 'summary') else {}
    prompt = (
        'Propose one genuinely new capability for this cognitive organism. It may be a '
        'cognitive function, external tool, sensor, embodiment or learning mechanism. '
        'Base it on a plausible limitation or opportunity in the current architecture, '
        'not on a generic chatbot feature. Return JSON only with exactly these string '
        'fields: title, description, category, motivation, expected_capability, '
        'constraints, success_criteria; category must be one of cognitive, tool, sensor, '
        'embodiment, learning, interaction, experiment; priority must be an integer 1-5. '
        f'Current workspace snapshot: {json.dumps(_json_safe(workspace_state), ensure_ascii=False)[:2500]}'
    )
    system_prompt = (
        'You are Lumina proposing research directions for your own development. Be '
        'concrete, falsifiable and modest. Do not claim that anything is implemented. '
        'Prefer a small prototype with observable evidence and a safe fallback.'
    )
    try:
        result = await asyncio.to_thread(
            llm.generate_bare_result,
            prompt,
            system_prompt=system_prompt,
            max_tokens=700,
            temperature=0.35,
            reasoning_format='none',
            json_mode=True,
        )
        # A separate lightweight text model may be configured for background
        # work. If it cannot produce structured content, retry once with the
        # primary model rather than presenting a false "no proposal" state.
        if result.get('status') != 'ok' and getattr(llm, 'text_model', '') != getattr(getattr(llm, 'provider', None), 'model', ''):
            result = await asyncio.to_thread(
                llm.generate_bare_result,
                prompt,
                system_prompt=system_prompt,
                model=getattr(getattr(llm, 'provider', None), 'model', ''),
                max_tokens=700,
                temperature=0.35,
                reasoning_format='none',
                json_mode=True,
            )
    except Exception as exc:
        logger.warning('Autonomous capability proposal failed: %s', exc)
        raise HTTPException(502, 'Lumina could not generate a proposal.') from exc
    raw = result.get('text', '').strip() if isinstance(result, dict) else ''
    generated = {}
    try:
        start, end = raw.find('{'), raw.rfind('}')
        generated = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
    except (json.JSONDecodeError, TypeError):
        generated = {}
    required = ('title', 'description', 'category', 'motivation', 'expected_capability', 'constraints', 'success_criteria')
    generation_mode = 'llm'
    if not all(str(generated.get(key, '')).strip() for key in required):
        # Reasoning-capable local models can return an empty final channel even
        # when the organism has enough state to form a useful next proposal.
        # Preserve the user action by creating a transparent, state-grounded
        # draft rather than fabricating an LLM response.
        generation_mode = 'organism_state_fallback'
        focus = str((workspace_state or {}).get('focus') or '').strip()
        active = 'cognitive capability development'
        generated = {
            'title': f'Investigate {focus or active}',
            'description': f'Create a bounded experiment that improves or measures {focus or active}.',
            'category': 'experiment',
            'motivation': 'Turn the organism\'s current internal focus into a testable development direction.',
            'expected_capability': 'A measurable improvement with observable evidence in the relevant cognitive stream.',
            'constraints': 'Keep the experiment local, asynchronous and reversible; do not disrupt user chat.',
            'success_criteria': 'Define a baseline, run repeated trials, compare outcomes and record the result.',
            'priority': 3,
        }
    category = str(generated['category']).lower().strip()
    if category not in _PROPOSAL_CATEGORIES:
        category = 'experiment'
    item = {key: str(generated[key]).strip() for key in required}
    item.update({'category': category, 'priority': max(1, min(5, int(generated.get('priority', 3)))), 'source': 'organism', 'generation_mode': generation_mode, 'status': 'draft', 'id': str(uuid.uuid4()), 'created_at': time.time(), 'updated_at': time.time()})
    item.update(_proposal_analysis(item))
    proposals = _load_capability_proposals()
    proposals.append(item)
    _save_capability_proposals(proposals)
    _event(f"Lumina generated capability proposal: {item['title'][:80]}")
    return _json_safe(item)


@router.patch('/capability-proposals/{proposal_id}')
async def update_capability_proposal(proposal_id: str, payload: CapabilityProposalUpdate):
    proposals = _load_capability_proposals()
    item = next((entry for entry in proposals if entry.get('id') == proposal_id), None)
    if item is None:
        raise HTTPException(404, 'Capability proposal not found.')
    changes = payload.model_dump(exclude_none=True)
    item.update(changes)
    if changes.get('status') == 'in_progress' and item.get('category') == 'experiment':
        state = _runtime()
        organism = getattr(getattr(state, 'persona', None), '_organism', None)
        if organism is None:
            raise HTTPException(503, 'Lumina cognitive engine is not ready.')
        try:
            from cognition.capability_development import CapabilityDevelopmentEngine
            loop = getattr(organism, '_loop', None)
            engine = getattr(loop, '_capability_development', None) if loop else None
            if engine is None:
                engine = CapabilityDevelopmentEngine(organism)
                if loop is not None:
                    loop._capability_development = engine
            snapshot = engine.start(item)
            experiment = snapshot.get('experiment') or {}
            item['experiment_id'] = experiment.get('id')
            item['experiment_status'] = experiment.get('status')
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            logger.warning('Capability experiment start failed: %s', exc)
            raise HTTPException(500, 'Capability experiment could not start.') from exc
    item['updated_at'] = time.time()
    _save_capability_proposals(proposals)
    return _json_safe(item)


@router.post('/capability-proposals/{proposal_id}/start')
async def start_capability_experiment(proposal_id: str):
    """Start a proposal-driven local experiment without entering chat."""
    proposals = _load_capability_proposals()
    item = next((entry for entry in proposals if entry.get('id') == proposal_id), None)
    if item is None:
        raise HTTPException(404, 'Capability proposal not found.')
    if item.get('category') != 'experiment':
        raise HTTPException(400, 'Only experiment proposals can be started directly.')
    state = _runtime()
    organism = getattr(getattr(state, 'persona', None), '_organism', None)
    if organism is None:
        raise HTTPException(503, 'Lumina cognitive engine is not ready.')
    try:
        from cognition.capability_development import CapabilityDevelopmentEngine
        loop = getattr(organism, '_loop', None)
        engine = getattr(loop, '_capability_development', None) if loop else None
        if engine is None:
            engine = CapabilityDevelopmentEngine(organism)
            if loop is not None:
                loop._capability_development = engine
        snapshot = engine.start(item)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.warning('Capability experiment start failed: %s', exc)
        raise HTTPException(500, 'Capability experiment could not start.') from exc
    experiment = snapshot.get('experiment') or {}
    item.update({
        'status': 'in_progress',
        'experiment_id': experiment.get('id'),
        'experiment_status': experiment.get('status'),
        'updated_at': time.time(),
    })
    _save_capability_proposals(proposals)
    _event(f"Capability experiment started: {item['title'][:80]}")
    return _json_safe({**item, 'experiment': experiment, 'verified_capability': snapshot.get('verified_capability')})


@router.get('/capability-proposals/{proposal_id}/experiment')
async def capability_experiment_status(proposal_id: str):
    """Return the persisted experiment state for one proposal."""
    proposals = _load_capability_proposals()
    item = next((entry for entry in proposals if entry.get('id') == proposal_id), None)
    if item is None:
        raise HTTPException(404, 'Capability proposal not found.')
    state = _runtime()
    organism = getattr(getattr(state, 'persona', None), '_organism', None)
    loop = getattr(organism, '_loop', None) if organism else None
    try:
        from cognition.capability_development import CapabilityDevelopmentEngine
        engine = getattr(loop, '_capability_development', None) if loop else None
        if engine is None:
            engine = CapabilityDevelopmentEngine(organism)
            if loop is not None:
                loop._capability_development = engine
        snapshot = engine.snapshot()
        if not snapshot.get('experiment') and item.get('status') == 'in_progress' and item.get('category') == 'experiment':
            snapshot = engine.start(item)
            experiment = snapshot.get('experiment') or {}
            item['experiment_id'] = experiment.get('id')
            item['experiment_status'] = experiment.get('status')
            item['updated_at'] = time.time()
            _save_capability_proposals(proposals)
    except Exception as exc:
        raise HTTPException(500, 'Capability experiment status unavailable.') from exc
    return _json_safe({'proposal': item, 'experiment': snapshot.get('experiment'), 'verified_capability': snapshot.get('verified_capability')})


@router.post('/capability-proposals/{proposal_id}/analyze')
async def analyze_capability_proposal(proposal_id: str):
    """Submit a proposal to Lumina's cognitive layer without adding chat history."""
    proposals = _load_capability_proposals()
    item = next((entry for entry in proposals if entry.get('id') == proposal_id), None)
    if item is None:
        raise HTTPException(404, 'Capability proposal not found.')
    state = _runtime()
    organism = getattr(getattr(state, 'persona', None), '_organism', None)
    llm = getattr(state, 'llm', None)
    if organism is None or llm is None:
        raise HTTPException(503, 'Lumina cognitive engine is not ready.')

    submission = (
        f"Development proposal: {item['title']}\n"
        f"Category: {item.get('category', 'cognitive')}\n"
        f"Description: {item.get('description', '')}\n"
        f"Motivation: {item.get('motivation', '')}\n"
        f"Expected capability: {item.get('expected_capability', '')}\n"
        f"Constraints: {item.get('constraints', '')}\n"
        f"Success criteria: {item.get('success_criteria', '')}"
    )
    workspace = getattr(organism, 'workspace', None)
    if workspace is not None:
        workspace.broadcast('development_proposal', submission, priority=0.78)
    persona = getattr(state, 'persona', None)
    persona_context = persona.get_prompt_context('default') if persona and hasattr(persona, 'get_prompt_context') else {}
    memory_context = []
    try:
        memory = getattr(persona, 'memory_system', None)
        if memory is not None and hasattr(memory, 'retrieve_memories'):
            memory_context = memory.retrieve_memories(item.get('title', ''), limit=5)
    except Exception as exc:
        logger.debug('Development dialogue memory retrieval unavailable: %s', exc)
    active_goals = []
    try:
        goal_engine = getattr(persona, 'goal_engine', None)
        if goal_engine is not None and hasattr(goal_engine, 'get_active_goals'):
            active_goals = [getattr(goal, 'topic', str(goal)) for goal in goal_engine.get_active_goals()[:5]]
    except Exception as exc:
        logger.debug('Development dialogue goal context unavailable: %s', exc)
    context_block = (
        f"Persona context: {json.dumps(_json_safe(persona_context), ensure_ascii=False)[:3000]}\n"
        f"Active goals: {json.dumps(_json_safe(active_goals), ensure_ascii=False)}\n"
        f"Relevant memories: {json.dumps(_json_safe(memory_context), ensure_ascii=False)[:3000]}\n"
        f"Workspace: {json.dumps(_json_safe(workspace.summary() if workspace and hasattr(workspace, 'summary') else {}), ensure_ascii=False)[:2500]}"
    )
    system_prompt = (
        'You are the cognitive development analyst inside a persistent cognitive organism. '
        'Analyze the proposal as a possible new capability, tool, sensor or embodiment. '
        'Do not claim implementation. Return a concise assessment with: interpretation, '
        'cognitive value, affected systems, dependencies, risks, prototype steps and '
        'measurable evidence of success.'
    )
    try:
        result = await asyncio.to_thread(
            llm.generate_bare_result,
            f"{context_block}\n\n{submission}",
            system_prompt=system_prompt + ' This is a private Development dialogue, not a user-facing chat turn.',
            max_tokens=700,
            temperature=0.25,
            reasoning_format='none',
        )
    except Exception as exc:
        logger.warning('Capability proposal analysis failed: %s', exc)
        raise HTTPException(502, 'Lumina could not analyze the proposal.') from exc
    if result.get('status') != 'ok' or not result.get('text', '').strip():
        raise HTTPException(503, f"Lumina returned no analysis ({result.get('reason', 'unknown reason')}).")
    item['organism_analysis'] = result['text'].strip()
    item['analysis_at'] = time.time()
    item['updated_at'] = time.time()
    _save_capability_proposals(proposals)
    _event(f"Capability proposal analyzed: {item['title'][:80]}")
    return _json_safe(item)


def _settings_snapshot():
    from managers.settings_manager import config
    values = {}
    for name in _SETTING_FIELDS:
        value = getattr(config, name, None)
        values[name] = '••••••••' if name in _SECRET_FIELDS and value else value
    return values


@router.get('/settings')
async def settings_snapshot():
    return {'values': _settings_snapshot(), 'secret_fields': sorted(_SECRET_FIELDS)}


@router.get('/voicemem/status')
async def voicemem_status():
    """Report optional VoiceMem availability without importing it at boot."""
    try:
        from cognition.voicemem_adapter import VoiceMemAdapter
        from managers.settings_manager import config
        adapter = VoiceMemAdapter(
            enabled=bool(getattr(config, 'VOICEMEM_ENABLED', False)),
            data_path=getattr(config, 'VOICEMEM_DATA_PATH', 'data/persona/voicemem'),
            top_k=getattr(config, 'VOICEMEM_TOP_K', 5),
            local_mode=bool(getattr(config, 'VOICEMEM_LOCAL_MODE', True)),
            local_base_url=getattr(config, 'LLM_BASE_URL', 'http://localhost:1234/v1'),
            local_model=getattr(config, 'TEXT_MODEL', '') or getattr(config, 'LLM_MODEL', 'local-model'),
        )
        result = adapter.status()
        result['evaluation'] = adapter.evaluation()
        return result
    except Exception as exc:
        logger.debug('VoiceMem status unavailable: %s', exc)
        return {'enabled': False, 'available': False, 'ready': False, 'error': str(exc)}


@router.get('/voicemem/space')
async def voicemem_space():
    """Expose the active local VoiceMem space for the main interface."""
    try:
        from cognition.voicemem_adapter import VoiceMemAdapter
        from managers.settings_manager import config
        adapter = VoiceMemAdapter(
            enabled=bool(getattr(config, 'VOICEMEM_ENABLED', False)),
            data_path=getattr(config, 'VOICEMEM_DATA_PATH', 'data/persona/voicemem'),
            top_k=getattr(config, 'VOICEMEM_TOP_K', 5),
            local_mode=bool(getattr(config, 'VOICEMEM_LOCAL_MODE', True)),
            local_base_url=getattr(config, 'LLM_BASE_URL', 'http://localhost:1234/v1'),
            local_model=getattr(config, 'TEXT_MODEL', '') or getattr(config, 'LLM_MODEL', 'local-model'),
        )
        return _json_safe(adapter.space_snapshot())
    except Exception as exc:
        logger.debug('VoiceMem space unavailable: %s', exc)
        return {'status': {'enabled': False, 'available': False, 'ready': False, 'error': str(exc)},
                'evaluation': {}, 'observations': [], 'retrievals': [], 'cached': []}


@router.post('/settings')
async def update_settings(payload: SettingsUpdate):
    from managers.settings_manager import config, save_settings
    # Do not persist an enabled VoiceMem state that cannot work.  This keeps
    # config.json truthful and avoids a confusing silent fallback on restart.
    if 'VOICEMEM_ENABLED' in payload.values and bool(payload.values.get('VOICEMEM_ENABLED')):
        try:
            from importlib.util import find_spec
            if find_spec('voicemem') is None:
                raise HTTPException(
                    400,
                    'VoiceMem is not installed. Set LUMINA_INSTALL_VOICEMEM=1 and rerun setup, then enable it.'
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, f'VoiceMem availability check failed: {exc}') from exc
    changed = []
    for name, value in payload.values.items():
        if name not in _SETTING_FIELDS:
            continue
        if name in _SECRET_FIELDS and value == '••••••••':
            continue
        try:
            setattr(config, name, value)
            changed.append(name)
        except Exception as exc:
            raise HTTPException(400, f'Invalid setting {name}: {exc}')
    save_settings(config, silent=True)
    body_fields = {
        'BODY_RUNTIME_ENABLED', 'BODY_PLUGIN_HOME_ASSISTANT_ENABLED',
        'BODY_HOST', 'BODY_PORT',
        'HOME_ASSISTANT_ENABLED', 'HOME_ASSISTANT_PRESENCE_ENABLED',
        'HOME_ASSISTANT_URL', 'HOME_ASSISTANT_TOKEN', 'HOME_ASSISTANT_VERIFY_SSL',
        'HOME_ASSISTANT_POLL_INTERVAL', 'HOME_ASSISTANT_ALLOWED_DOMAINS',
        'HOME_ASSISTANT_SELECTED_ENTITIES', 'HOME_ASSISTANT_ENTITY_TAGS',
    }
    if body_fields.intersection(changed):
        organism = getattr(getattr(_runtime(), 'persona', None), '_organism', None)
        if organism is not None:
            from cognition.body_runtime import get_body_runtime
            get_body_runtime(organism).update_config({
                key: getattr(config, key) for key in body_fields if hasattr(config, key)
            })
    audio_fields = {
        'TTS_PROVIDER', 'STT_PROVIDER', 'WHISPER_MODEL',
        'COQUI_VOICE_REFERENCE', 'VOICE_LANGUAGE',
        'VAD_AGGRESSIVENESS', 'VAD_ONSET_CHUNKS', 'VAD_SILENCE_DURATION',
        'VAD_MIN_SPEECH_DURATION', 'VAD_ENERGY_GATE_FACTOR',
        'TTS_STREAMING', 'TTS_BARGE_IN', 'BARGE_IN_SENSITIVITY',
        'INTER_SENTENCE_PAUSE_MS', 'TTS_POST_ROLL_MS',
    }
    audio_changed = bool(audio_fields.intersection(changed))
    if audio_changed:
        # Apply the selection to the live process immediately.  Provider
        # changes rebuild the engine; VAD/language/reference changes are
        # applied by the same path without reloading a heavy model.
        state = _runtime()
        if state.audio is not None:
            await asyncio.to_thread(state.reload_audio)
    state = _runtime()
    if {'UNIVERSAL_CONNECTOR_ENABLED', 'BODY_RUNTIME_ENABLED', 'BODY_PLUGIN_HOME_ASSISTANT_ENABLED',
        'HOME_ASSISTANT_ENABLED', 'HOME_ASSISTANT_PRESENCE_ENABLED', 'HOME_ASSISTANT_URL',
        'HOME_ASSISTANT_TOKEN', 'HOME_ASSISTANT_VERIFY_SSL', 'HOME_ASSISTANT_POLL_INTERVAL',
        'HOME_ASSISTANT_ALLOWED_DOMAINS', 'HOME_ASSISTANT_SELECTED_ENTITIES', 'HOME_ASSISTANT_DISCOVERED_ENTITIES', 'HOME_ASSISTANT_ENTITY_TAGS'}.intersection(changed):
        organism = getattr(getattr(state, 'persona', None), '_organism', None)
        if organism is not None:
            from cognition.universal_connector import get_universal_connector
            await get_universal_connector(organism).reconcile_home_assistant_monitor()
    _event(f'Settings saved ({len(changed)} fields)')
    state = _runtime()
    active_audio = getattr(state, 'audio', None)
    return {
        'values': _settings_snapshot(),
        'changed': changed,
        'audio_applied': audio_changed and active_audio is not None,
        'active_tts': (active_audio.tts_engine or {}).get('type') if active_audio else None,
        'active_stt': (active_audio.stt_engine or {}).get('type') if active_audio else None,
        'restart_recommended': bool(set(changed) - audio_fields),
    }


@router.post('/preferences')
async def update_preferences(payload: PreferenceUpdate):
    from managers.settings_manager import config, save_settings
    if payload.web_search_mode is not None:
        config.WEB_SEARCH_MODE = payload.web_search_mode
    if payload.response_verbosity is not None:
        config.RESPONSE_VERBOSITY = payload.response_verbosity
    save_settings(config, silent=True)
    return {
        'web_search_mode': getattr(config, 'WEB_SEARCH_MODE', 'off'),
        'response_verbosity': getattr(config, 'RESPONSE_VERBOSITY', 'concise'),
    }


@router.post('/language')
async def update_language(payload: LanguageUpdate):
    from managers.settings_manager import config, save_settings
    allowed = {'auto', 'en', 'en-gb', 'fr', 'es', 'de', 'it', 'pt', 'zh-cn', 'ru', 'ar', 'ja', 'ko', 'hi', 'nl', 'pl', 'tr', 'sv', 'uk'}
    if payload.language not in allowed:
        raise HTTPException(400, 'Unsupported response language.')
    config.RESPONSE_LANGUAGE = payload.language
    # The main selector controls the language of both the generated answer
    # and the following TTS playback.  Keep VOICE_LANGUAGE in sync so Whisper,
    # Coqui, SAPI, Edge, and the next audio request all see the same choice.
    if payload.language != 'auto':
        config.VOICE_LANGUAGE = payload.language
        audio = getattr(_runtime(), 'audio', None)
        if audio:
            audio.update_language(payload.language)
    save_settings(config, silent=True)
    _event(f"Response language set to {payload.language}")
    return {'language': payload.language}


@router.post('/network/toggle')
async def network_toggle():
    from managers.settings_manager import config, save_settings
    enabled = not bool(getattr(config, 'LUMINA_NETWORK_ENABLED', False))
    config.__dict__['LUMINA_NETWORK_ENABLED'] = enabled
    save_settings(config, silent=True)
    state = _runtime()
    if enabled and not getattr(state, 'lumina_network', None):
        from cognition.lumina_network import LuminaNetwork
        state.lumina_network = LuminaNetwork(state.persona, state.llm)
        state.lumina_network.load_from_config(getattr(config, 'LUMINA_CHILDREN', []) or [])
        state.lumina_network.set_master_info(f"http://127.0.0.1:{getattr(config, 'NICEGUI_PORT', 8080)}", getattr(config, 'LUMINA_MASTER_NAME', 'Lumina-Master'))
    elif not enabled and getattr(state, 'lumina_network', None):
        await state.lumina_network.stop()
        state.lumina_network = None
    _event(f"Network {'enabled' if enabled else 'disabled'}")
    return await network_status()


@router.post('/network/action')
async def network_action(payload: NetworkAction):
    state = _runtime()
    network = getattr(state, 'lumina_network', None)
    if not network:
        raise HTTPException(409, 'Flux network is disabled.')
    if not payload.child_id:
        raise HTTPException(400, 'A Flux child is required.')
    if payload.text:
        return {'ok': True, 'action': 'ask', 'response': await network.send_to_child(payload.child_id, payload.text, show_in_ui=False)}
    if payload.topic:
        dialogue_id = str(uuid.uuid4())[:8]
        child = next((item for item in network.list_children() if item['id'] == payload.child_id), None)
        _dialogues[dialogue_id] = {'child_id': payload.child_id, 'child_name': child['name'] if child else payload.child_id, 'topic': payload.topic, 'started_at': time.time()}
        async def run_dialogue():
            try:
                await network.start_dialogue(payload.child_id, topic=payload.topic, conversation_id=dialogue_id)
            finally:
                _dialogues.get(dialogue_id, {})['finished_at'] = time.time()
        task = asyncio.create_task(run_dialogue())
        _workers.add(task)
        task.add_done_callback(_workers.discard)
        _event('Flux dialogue started', payload.child_id)
        return {'ok': True, 'action': 'dialogue_started', 'dialogue_id': dialogue_id}
    online = await network.ping_child(payload.child_id)
    _event('Flux ping completed', payload.child_id)
    return {'ok': True, 'action': 'ping', 'online': online}


@router.get('/network/dialogue/{dialogue_id}')
async def network_dialogue(dialogue_id: str):
    state = _runtime()
    session = _dialogues.get(dialogue_id)
    if not session:
        raise HTTPException(410, 'This Flux dialogue is no longer available, possibly because Lumina restarted. Start a new dialogue from the main window.')
    network = getattr(state, 'lumina_network', None)
    history = network.get_history(dialogue_id) if network else []
    return {
        **session,
        'active': dialogue_id in _dialogues and 'finished_at' not in session,
        'messages': [
            {'text': message.text, 'sender_id': message.sender_id, 'sender_name': message.sender_name,
             'direction': message.direction, 'timestamp': message.timestamp}
            for message in history
        ],
    }


@router.get('/camera/frame')
async def camera_frame():
    state = _runtime()
    vision = getattr(state, 'vision', None)
    frame = vision.get_latest_clean_encoded_frame() if vision and vision.camera_active else None
    faces = []
    if vision and vision.camera_active:
        faces = [{'name': face.get('name', 'Unknown'), 'confidence': face.get('confidence', 0), 'location': face.get('location', {})} for face in (getattr(vision, 'last_detected_faces', None) or [])]
    return {'active': bool(vision and vision.camera_active), 'frame': frame, 'faces': faces}


@router.post('/camera/toggle')
async def camera_toggle():
    state = _runtime()
    vision = getattr(state, 'vision', None)
    if not vision:
        raise HTTPException(503, 'Camera is unavailable.')
    active = await asyncio.to_thread(vision.stop_camera if vision.camera_active else vision.start_camera)
    return {'active': bool(vision.camera_active), 'started': active if isinstance(active, bool) else None}


class VisionSettingsUpdate(BaseModel):
    camera_id: int | None = Field(default=None, ge=0, le=20)
    fps: int | None = Field(default=None, ge=1, le=30)
    resolution: str | None = Field(default=None, pattern=r'^\d{3,4}x\d{3,4}$')
    face_detection: bool | None = None


@router.post('/vision/settings')
async def update_vision_settings(payload: VisionSettingsUpdate):
    state = _runtime()
    vision = getattr(state, 'vision', None)
    from managers.vision_manager import FACE_RECOGNITION_AVAILABLE, FACE_RECOGNITION_ERROR
    if not vision:
        raise HTTPException(503, 'Vision manager is unavailable.')
    was_active = bool(vision.camera_active)
    if was_active:
        await asyncio.to_thread(vision.stop_camera)
    if payload.camera_id is not None: vision.camera_id = payload.camera_id
    if payload.fps is not None: vision.target_fps = payload.fps
    if payload.resolution:
        width, height = (int(value) for value in payload.resolution.split('x'))
        vision.frame_width, vision.frame_height = width, height
    if payload.face_detection is not None: vision.face_detection_enabled = payload.face_detection
    from managers.settings_manager import config, save_settings
    config.CAMERA_ID = vision.camera_id
    config.CAMERA_FPS = vision.target_fps
    config.CAMERA_RESOLUTION = f'{vision.frame_width}x{vision.frame_height}'
    save_settings(config, silent=True)
    if was_active:
        await asyncio.to_thread(vision.start_camera)
    _event('Vision settings updated')
    return {'camera_id': vision.camera_id, 'fps': vision.target_fps, 'resolution': f'{vision.frame_width}x{vision.frame_height}', 'face_detection': vision.face_detection_enabled, 'initialized': bool(getattr(vision, 'face_detection_enabled', False) and FACE_RECOGNITION_AVAILABLE), 'face_recognition_available': FACE_RECOGNITION_AVAILABLE, 'face_recognition_error': FACE_RECOGNITION_ERROR, 'known_faces': len(getattr(vision, 'face_encodings', {})), 'vision_memories': len(getattr(vision, 'vision_meta', []))}


@router.post('/vision/clear-memory')
async def clear_vision_memory():
    vision = getattr(_runtime(), 'vision', None)
    if not vision:
        raise HTTPException(503, 'Vision manager is unavailable.')
    vision.vision_meta = []
    return {'ok': True, 'vision_memories': 0}


@router.get('/vision/faces')
async def vision_faces():
    vision = getattr(_runtime(), 'vision', None)
    if not vision:
        raise HTTPException(503, 'Vision manager is unavailable.')
    return {'faces': _json_safe(vision.get_known_faces()), 'stats': _json_safe(vision.get_face_stats())}


@router.get('/vision/memories')
async def vision_memories(query: str = ''):
    vision = getattr(_runtime(), 'vision', None)
    if not vision: raise HTTPException(503, 'Vision manager is unavailable.')
    if query.strip():
        result = await asyncio.to_thread(vision.search_vision_memory, query.strip(), 10)
    else:
        result = list(getattr(vision, 'vision_meta', []))[-10:]
    return {'memories': _json_safe(result)}


@router.post('/vision/analyze')
async def analyze_vision():
    vision = getattr(_runtime(), 'vision', None)
    if not vision or not vision.camera_active: raise HTTPException(409, 'Start the camera before analyzing a frame.')
    result = await vision.analyze_frame_async(
        prompt=(
            "Describe the visible scene naturally and concretely for a human. "
            "Mention the main person or objects, their setting, posture or activity, "
            "and notable background details. Start with the scene description, not "
            "image resolution, brightness, channels, or face-recognition diagnostics."
        )
    )
    return {'analysis': result[0], 'faces': _json_safe(result[1])}


@router.post('/vision/reset')
async def reset_vision():
    vision = getattr(_runtime(), 'vision', None)
    if not vision: raise HTTPException(503, 'Vision manager is unavailable.')
    if vision.camera_active: await asyncio.to_thread(vision.stop_camera)
    await asyncio.sleep(.1)
    started = await asyncio.to_thread(vision.start_camera)
    return {'active': bool(vision.camera_active), 'started': bool(started)}


class FaceCapture(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@router.post('/vision/capture-face')
async def capture_face(payload: FaceCapture):
    vision = getattr(_runtime(), 'vision', None)
    if not vision or not vision.camera_active:
        raise HTTPException(409, 'Start the camera before capturing a face.')
    frame = None
    with getattr(vision, 'frame_lock', asyncio.Lock()):
        current = getattr(vision, 'current_frame', None)
        if current is not None:
            frame = current.copy()
    if frame is None:
        raise HTTPException(409, 'No camera frame is available yet.')
    try:
        import cv2
        path = Path(tempfile.gettempdir()) / f'lumina-face-{uuid.uuid4().hex}.jpg'
        cv2.imwrite(str(path), frame)
        ok, detail = await asyncio.to_thread(vision.register_face_from_image, str(path), payload.name.strip())
        path.unlink(missing_ok=True)
        if not ok: raise HTTPException(422, detail)
        return {'ok': True, 'message': detail, 'faces': _json_safe(vision.get_known_faces())}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning('Face capture failed: %s', exc)
        raise HTTPException(500, 'Face capture failed.')


class Turn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    # Voice turns use a latency-optimized cognitive path. Text chat keeps
    # the full auxiliary reasoning pipeline unchanged.
    voice_mode: bool = False


@router.post('/voice/coqui-recording')
async def coqui_recording(file: UploadFile = File(...)):
    state = _runtime()
    audio = getattr(state, 'audio', None)
    if not audio:
        raise HTTPException(503, 'Audio manager is unavailable.')
    raw = await file.read()
    if len(raw) < 1000:
        raise HTTPException(422, 'Recording is empty.')
    voice_dir = Path('data/voices')
    voice_dir.mkdir(parents=True, exist_ok=True)
    target = (voice_dir / 'recorded_voice.wav').resolve()
    source = voice_dir / f'coqui-upload-{uuid.uuid4().hex}.webm'
    source.write_bytes(raw)
    try:
        import subprocess
        converted = False
        try:
            subprocess.run(['ffmpeg', '-y', '-i', str(source), '-ar', '22050', '-ac', '1', str(target)], check=True, capture_output=True, timeout=30)
            converted = target.exists() and target.stat().st_size > 1000
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        if not converted:
            raise HTTPException(422, 'Audio conversion requires ffmpeg. Install ffmpeg and try again.')
        from managers.settings_manager import config, save_settings
        config.COQUI_VOICE_REFERENCE = str(target)
        save_settings(config, silent=True)
        # The upload may happen immediately after selecting Coqui, while the
        # live manager is still using the previous provider. Reconcile the
        # provider and reference together so the next test uses this file.
        await asyncio.to_thread(state.reload_audio)
        active_audio = getattr(state, 'audio', None)
        active_reference = None
        if active_audio and hasattr(active_audio, 'get_voice_clone_status'):
            active_reference = active_audio.get_voice_clone_status().get('voice_ref')
        _event('Coqui voice recording saved')
        return {
            'ok': True,
            'path': str(target),
            'active_provider': (active_audio.tts_engine or {}).get('type') if active_audio else None,
            'active_reference': active_reference,
            'duration_hint': round(len(raw) / 16000, 1),
        }
    finally:
        source.unlink(missing_ok=True)


@router.post('/chat')
async def chat(payload: Turn, request: Request):
    state = _runtime()
    if not state.ready or not state.persona or not state.persona.is_ready:
        raise HTTPException(503, 'Lumina is still starting. Try again shortly.')
    if _turn_lock.locked():
        raise HTTPException(409, 'Lumina is finishing another turn. Please wait.')
    from managers.user_manager import user_manager
    from managers.security_manager import security, SecurityViolation
    try:
        if state.vision and state.vision.camera_active:
            user_manager.sync_active_from_faces(
                getattr(state.vision, 'last_detected_faces', []) or []
            )
    except Exception:
        pass
    user_id = user_manager.active_id
    if not security.allow_request(user_id):
        raise HTTPException(429, 'Too many requests. Please wait a moment.')
    try:
        text = security.sanitise_input(payload.text.strip(), user_id)
    except SecurityViolation as exc:
        raise HTTPException(400, str(exc)) from exc
    if not text:
        raise HTTPException(400, 'Enter a message first.')
    # Keep ambient perception socially grounded on the migrated interface.
    # The legacy page already notified PresenceEngine; this API path did not.
    try:
        presence = getattr(state.vision, 'presence_engine', None) if state.vision else None
        if presence is not None:
            presence.notify_user_interaction(topic=text)
    except Exception as exc:
        logger.debug('Presence interaction notification unavailable: %s', exc)
    await _turn_lock.acquire()
    queue = asyncio.Queue(maxsize=128)
    stopped = asyncio.Event()
    turn_id = str(uuid.uuid4())

    async def produce():
        started = time.perf_counter()
        seq = 0
        llm_turn_reserved = False

        async def emit(kind, **fields):
            nonlocal seq
            seq += 1
            item = {'type': kind, 'turn_id': turn_id, 'seq': seq, **fields}
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(queue.put(item), .2)
                    return
                except asyncio.TimeoutError:
                    continue

        try:
            # Reserve the provider before any cognitive pre-processing. This
            # prevents perception/presence from acquiring LM Studio in the
            # small gap before PersonaBridge starts its visible stream.
            if hasattr(state.llm, 'begin_interactive_turn'):
                state.llm.begin_interactive_turn()
                llm_turn_reserved = True
            _event('Turn started', turn_id)
            await emit('start')
            vision = None
            if state.vision and state.vision.camera_active:
                vision = state.vision.get_visual_context_for_prompt()
            stream = state.persona.get_response_stream(
                text,
                user_id=user_id,
                vision_context=vision,
                voice_mode=payload.voice_mode,
            )
            first = True
            iterator = stream.__aiter__()
            try:
                while True:
                    next_task = asyncio.create_task(iterator.__anext__())
                    stop_task = asyncio.create_task(stopped.wait())
                    try:
                        done, _ = await asyncio.wait(
                            {next_task, stop_task},
                            timeout=120.0 if first else 180.0,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if stop_task in done:
                            next_task.cancel()
                            await asyncio.gather(next_task, return_exceptions=True)
                            break
                        if next_task not in done:
                            stage = getattr(state.persona, '_chat_stage', 'unknown')
                            logger.error('Chat stalled: turn=%s stage=%s', turn_id, stage)
                            diagnostic = Path('logs/chat-stall.log')
                            diagnostic.parent.mkdir(parents=True, exist_ok=True)
                            with diagnostic.open('a', encoding='utf-8') as output:
                                output.write(f'\nTurn {turn_id}: {stage}\n')
                                output.flush()
                                faulthandler.dump_traceback(file=output, all_threads=True)
                            next_task.cancel()
                            await asyncio.gather(next_task, return_exceptions=True)
                            raise asyncio.TimeoutError
                        chunk = next_task.result()
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        stage = getattr(state.persona, '_chat_stage', 'unknown')
                        await emit('error', message=f'Lumina stalled during {stage}. Diagnostic saved to logs/chat-stall.log.')
                        stopped.set()
                        break
                    except asyncio.CancelledError:
                        next_task.cancel()
                        await asyncio.gather(next_task, return_exceptions=True)
                        raise
                    finally:
                        stop_task.cancel()
                        await asyncio.gather(stop_task, return_exceptions=True)
                    if stopped.is_set():
                        break
                    if isinstance(chunk, dict):
                        if chunk.get('type') == 'reasoning' and chunk.get('text'):
                            await emit('reasoning', text=str(chunk['text'])[:4000])
                        elif chunk.get('type') == 'web_sources' and chunk.get('sources'):
                            await emit('web_sources', sources=chunk['sources'])
                        continue
                    if first:
                        _timings['first_token_ms'] = round((time.perf_counter() - started) * 1000)
                        first = False
                    await emit('delta', text=chunk)
            finally:
                # A provider-backed generator may still be unwinding a
                # thread after the browser, STT, or timeout has stopped the
                # request.  Never let cleanup hold the global turn lock
                # indefinitely and make every later chat appear busy.
                try:
                    await asyncio.wait_for(stream.aclose(), timeout=2.0)
                except Exception as close_error:
                    logger.warning('Interface chat stream cleanup delayed: %s', close_error)
            _timings['turn_ms'] = round((time.perf_counter() - started) * 1000)
            telemetry = _cognitive_telemetry_snapshot()
            if telemetry is not None:
                await emit('telemetry', data=telemetry)
            _event('Turn interrupted' if stopped.is_set() else 'Turn completed', turn_id)
            await emit('done')
        except Exception:
            logger.exception('Interface chat failed')
            _event('Turn failed', turn_id)
            await emit('error', message='The response failed. Please try again.')
        finally:
            if llm_turn_reserved and hasattr(state.llm, 'end_interactive_turn'):
                state.llm.end_interactive_turn()
            _turn_lock.release()

    worker = asyncio.create_task(produce())
    _workers.add(worker)
    worker.add_done_callback(_workers.discard)

    async def events():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), .5)
                except asyncio.TimeoutError:
                    if worker.done():
                        break
                    continue
                yield json.dumps(event, ensure_ascii=False) + '\n'
                if event['type'] in {'done', 'error'}:
                    break
        finally:
            # Do not cancel a thread-backed inference call mid-flight. The worker
            # keeps the runtime lock until the current call returns and closes.
            stopped.set()

    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


async def _audio_work(fn):
    if _audio_lock.locked():
        raise HTTPException(409, 'Audio is busy. Please wait.')
    await _audio_lock.acquire()

    async def work():
        try:
            return await asyncio.to_thread(fn)
        finally:
            _audio_lock.release()

    task = asyncio.create_task(work())
    _workers.add(task)
    task.add_done_callback(_workers.discard)
    return await asyncio.shield(task)


@router.post('/transcribe')
async def transcribe(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 4_000_000:
            raise HTTPException(413, 'Recording is too long. Keep it under one minute.')
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise HTTPException(400, 'A WAV recording is required.')

    def work():
        state = _runtime()
        if not state.ensure_audio_ready():
            raise HTTPException(503, 'Audio is unavailable.')
        started = time.perf_counter()
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as file:
                file.write(data)
                path = Path(file.name)
            text = state.audio.speech_to_text(str(path))
            _timings['stt_ms'] = round((time.perf_counter() - started) * 1000)
            return {'text': text or ''}
        finally:
            if path:
                path.unlink(missing_ok=True)

    return await _audio_work(work)


@router.post('/speak')
async def speak(payload: Turn):
    def work():
        state = _runtime()
        if not state.ensure_audio_ready():
            raise HTTPException(503, 'Audio is unavailable.')
        # Reconcile the live audio object on every utterance.  The audio
        # manager may have been created before the language selector changed,
        # especially when the always-listening loop is active.
        from managers.settings_manager import config
        requested_language = getattr(config, 'RESPONSE_LANGUAGE', 'auto')
        if requested_language != 'auto' and getattr(state, 'audio', None):
            state.audio.update_language(requested_language)
        started = time.perf_counter()
        path = state.audio.text_to_speech(payload.text)
        if not path:
            raise HTTPException(503, 'Speech synthesis is unavailable.')
        # Provider owns its output file and cache; only read it here.
        data = Path(path).read_bytes()
        _timings['tts_ms'] = round((time.perf_counter() - started) * 1000)
        media = 'audio/mpeg' if Path(path).suffix.lower() == '.mp3' else 'audio/wav'
        return Response(data, media_type=media, headers={'Cache-Control': 'no-store'})

    return await _audio_work(work)
