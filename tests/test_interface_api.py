"""Contract checks use fake dependencies; no models, devices or user data."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

spec = importlib.util.spec_from_file_location('interface_under_test', Path(__file__).parents[1] / 'core/interface_api.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class Persona:
    is_ready = True
    _organism = SimpleNamespace(_loop=SimpleNamespace(_slow_cycle_count=9))

    async def get_response_stream(self, text, **kwargs):
        yield 'Bonjour '
        yield 'Fred'
        yield {'__meta__': True, 'raw': 'Bonjour Fred'}


class InterfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        api._turn_lock = asyncio.Lock()
        api._events.clear()
        self.runtime = SimpleNamespace(ready=True, persona=Persona(), vision=None, llm=None, get_uptime=lambda: 120)
        self.modules = {
            'managers.settings_manager': SimpleNamespace(
                config=SimpleNamespace(
                    STT_PROVIDER='whisper', TTS_PROVIDER='pyttsx3',
                    AFFECT_EMBEDDING_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                ),
                save_settings=lambda *_args, **_kwargs: None,
            ),
            'managers.user_manager': SimpleNamespace(user_manager=SimpleNamespace(active_id='test-user')),
            'managers.security_manager': SimpleNamespace(security=SimpleNamespace(allow_request=lambda _: True, sanitise_input=lambda text, _: text), SecurityViolation=ValueError),
        }
        self.patches = [patch.object(api, '_runtime', return_value=self.runtime), patch.dict(sys.modules, self.modules)]
        for item in self.patches:
            item.start()
        self.app = FastAPI()
        self.app.include_router(api.router)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def client(self):
        return AsyncClient(transport=ASGITransport(app=self.app), base_url='http://localhost')

    def test_cognitive_trace_returns_planner_snapshot(self):
        snapshot = {'active_goals': [], 'dominant_plan': {'objective': 'Test objective'}}
        planner = SimpleNamespace(factual_self_report_snapshot=lambda: snapshot)
        loop = self.runtime.persona._organism._loop
        with patch.object(loop, '_long_horizon_planner', planner, create=True):
            self.assertEqual(api._cognitive_telemetry_snapshot(), snapshot)

    def test_orchestrator_snapshot_exposes_live_scheduler(self):
        self.runtime.orchestrator = SimpleNamespace(
            status=lambda: {'running': True, 'cycle_count': 12,
                            'last_activity': 'reflection', 'drives': {'coherence': .7},
                            'clock': {'medium_in_sec': 42}},
            event_system=SimpleNamespace(peek=lambda: [SimpleNamespace(type='user_message')]),
        )
        result = api._orchestrator_snapshot()
        self.assertTrue(result['available'])
        self.assertEqual(result['queued_events'], 1)
        self.assertEqual(result['pending_event_types'], ['user_message'])
        self.assertEqual(result['clock']['medium_in_sec'], 42)

    def test_missing_orchestrator_is_unavailable_not_active(self):
        self.assertEqual(api._orchestrator_snapshot(), {'available': False, 'running': False})

    async def test_chat_notifies_presence_engine_of_recent_user_activity(self):
        calls = []
        self.runtime.vision = SimpleNamespace(
            camera_active=False,
            presence_engine=SimpleNamespace(
                notify_user_interaction=lambda **kwargs: calls.append(kwargs),
            ),
        )
        async with self.client() as client:
            response = await client.post('/api/interface/chat', json={'text': 'Current topic'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [{'topic': 'Current topic'}])

    async def test_missing_dialogue_is_expired(self):
        with patch.object(api, '_dialogues', {}):
            async with self.client() as client:
                response = await client.get('/api/interface/network/dialogue/expired')
        self.assertEqual(response.status_code, 410)
        self.assertIn('Start a new dialogue', response.json()['detail'])

    async def test_snapshot_is_read_only_and_contains_only_allowlisted_data(self):
        before = dict(vars(self.runtime.persona._organism._loop))
        async with self.client() as client:
            result = (await client.get('/api/interface/status')).json()
        self.assertEqual(result['cycle'], 9)
        self.assertEqual(before, vars(self.runtime.persona._organism._loop))
        self.assertNotIn('config', result)

    async def test_settings_snapshot_exposes_affect_embedding_model(self):
        async with self.client() as client:
            result = (await client.get('/api/interface/settings')).json()
        self.assertEqual(
            result['values']['AFFECT_EMBEDDING_MODEL'],
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        )

    async def test_affect_embedding_model_can_be_saved_through_settings_api(self):
        selected = 'sentence-transformers/alternate-model'
        async with self.client() as client:
            response = await client.post('/api/interface/settings', json={
                'values': {'AFFECT_EMBEDDING_MODEL': selected},
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['values']['AFFECT_EMBEDDING_MODEL'], selected)

    async def test_stream_has_ids_order_and_terminal_event(self):
        async with self.client() as client:
            response = await client.post('/api/interface/chat', json={'text': 'Hello'})
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(response.status_code, 200)
            self.assertEqual([e['type'] for e in events], ['start', 'delta', 'delta', 'done'])
            self.assertEqual([e['seq'] for e in events], [1, 2, 3, 4])
            self.assertEqual(len({e['turn_id'] for e in events}), 1)
            self.assertFalse(api._turn_lock.locked())

    async def test_rejects_unready_busy_empty_and_cross_site(self):
        async with self.client() as client:
            self.runtime.ready = False
            self.assertEqual((await client.post('/api/interface/chat', json={'text': 'Hi'})).status_code, 503)
            self.runtime.ready = True
            self.assertEqual((await client.post('/api/interface/chat', json={'text': '  '})).status_code, 400)
            self.assertEqual((await client.post('/api/interface/chat', json={'text': ''})).status_code, 422)
            self.assertEqual((await client.post('/api/interface/chat', json={'text': 'Hi'}, headers={'Origin': 'https://untrusted.example'})).status_code, 403)
            self.assertEqual((await client.get('/api/interface/status', headers={'Sec-Fetch-Site': 'cross-site'})).status_code, 403)
            api._turn_lock = SimpleNamespace(locked=lambda: True)
            self.assertEqual((await client.post('/api/interface/chat', json={'text': 'Hi'})).status_code, 409)

    async def test_consecutive_turns_release_reservation(self):
        from unittest.mock import Mock
        self.runtime.llm = Mock()
        async with self.client() as client:
            for number in range(6):
                response = await client.post('/api/interface/chat', json={'text': f'Turn {number}'})
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual(events[-1]['type'], 'done')
                self.assertFalse(api._turn_lock.locked())
        self.assertEqual(self.runtime.llm.begin_interactive_turn.call_count, 6)
        self.assertEqual(self.runtime.llm.end_interactive_turn.call_count, 6)

    async def test_transcription_rejects_non_wav_and_oversized_uploads(self):
        async with self.client() as client:
            self.assertEqual((await client.post('/api/interface/transcribe', content=b'not audio')).status_code, 400)
            self.assertEqual((await client.post('/api/interface/transcribe', content=b'x' * 4_000_001)).status_code, 413)


if __name__ == '__main__':
    unittest.main()
