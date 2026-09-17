"""Application State Management"""
import asyncio
import threading
import logging
import time
import os
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

import managers.settings_manager as _settings_mod
from managers.settings_manager import config, load_settings
from managers.llm_manager import create_llm_manager
from managers.memory_manager import create_memory_manager
from managers.audio_manager import create_audio_manager
from managers.conversational_audio import ConversationalAudioManager
from managers.vision_manager import StreamingVisionManager

logger = logging.getLogger(__name__)


def _init_persona(llm_generate_fn=None, llm_stream_fn=None):
    """
    Bootstrap Lumina's EnhancedAISystem via PersonaBridge.
    Returns a PersonaBridge instance, or None on failure.
    """
    try:
        from cognition.persona_bridge import PersonaBridge
        persona = PersonaBridge(external_llm_fn=llm_generate_fn, external_llm_stream_fn=llm_stream_fn)
        if persona.is_ready:
            logger.info("✅ Persona (Lumina cognitive engine) ready")
        else:
            logger.warning("⚠️  Persona initialised in degraded mode (no Lumina)")
        return persona
    except Exception as e:
        logger.error(f"❌ Persona init failed: {e}")
        return None

def _cfg():
    """Always return the live config object (updated after every save)."""
    return _settings_mod.config

class AppState:
    def __init__(self):
        self.llm        = None
        self.memory     = None
        self.audio      = None
        self.conv_audio = None
        self.vision     = None
        self.presence_engine = None
        self.persona    = None   # PersonaBridge (Lumina cognitive engine) — wired in initialize()
        self.tts_stop_event = threading.Event()
        self.ready      = False
        self.start_time = time.time()
        self.initializing = False
        
        # Thread-safe audio→UI queue
        self.transcription_queue = queue.Queue()

    async def initialize(self):
        if self.initializing:
            return
        
        self.initializing = True
        
        try:
            self.llm = create_llm_manager()
            logger.info(f"✅ LLM: {_cfg().LLM_PROVIDER}/{_cfg().LLM_MODEL}")
        except Exception as e:
            logger.error(f"LLM init failed: {e}")

        # ── Persona (Lumina cognitive engine) ─────────────────────────────
        # Pass Robot's LLM fns so Lumina shares the same backend:
        #   generate_bare — for Lumina's internal cognitive tasks (dream, learning,
        #                   reflection). Calls the provider directly, NO history pollution,
        #                   temperature & max_tokens fully forwarded.
        #   generate_stream — for user-facing streaming responses (records history).
        try:
            llm_fn        = getattr(self.llm, "generate_bare",   None) if self.llm else None
            llm_stream_fn = getattr(self.llm, "generate_stream",  None) if self.llm else None
            # Fallback: if generate_bare unavailable (old provider), use plain generate
            if llm_fn is None:
                llm_fn = getattr(self.llm, "generate", None) if self.llm else None
                if llm_fn:
                    logger.warning("generate_bare not found on LLMManager — falling back to generate (history may be polluted by cognitive tasks)")
            self.persona = _init_persona(llm_generate_fn=llm_fn, llm_stream_fn=llm_stream_fn)
        except Exception as e:
            logger.error(f"Persona init failed (non-fatal): {e}")

        # ── Autonomous Orchestrator (v2) ───────────────────────────────────
        self.sce = None
        self.empathy_engine = None
        self.are = None
        self.acc = None   # Active Cognitive Context (CAG Layer 2)
        self.lumina_network = None
        self.orchestrator = None
        try:
            organism = getattr(self.persona, '_organism', None) if self.persona else None
            if organism is not None:
                from core.orchestrator import AutonomousOrchestrator
                self.orchestrator = AutonomousOrchestrator(
                    organism=organism,
                    workspace=getattr(organism, 'workspace', None),
                )
                # Wire vision into the internal loop for ambient perception
                if self.vision is not None:
                    try:
                        loop = getattr(organism, '_loop', None)
                        if loop and hasattr(loop, 'attach_vision'):
                            loop.attach_vision(self.vision)
                        # Also wire vision onto organism so brain bridge can read it
                        try:
                            if hasattr(loop, '_organism_ref'):
                                org = loop._organism_ref
                            else:
                                from cognition.cognitive_organism import CognitiveOrganism
                                org = None
                                for attr in ('_organism', 'organism'):
                                    org = getattr(loop, attr, None)
                                    if org: break
                            if org is not None:
                                org._vision_manager = self.vision
                        except Exception:
                            pass
                    except Exception as _ve:
                        logger.warning(f"Ambient vision wiring failed (non-fatal): {_ve}")
                # Wire messaging manager into execution layer (if configured later)
                organism.orchestrator = self.orchestrator
                logger.info("✅ Autonomous Orchestrator created (will start on first use)")
            else:
                logger.info("⚠️  Orchestrator skipped — organism not available yet")
        except Exception as e:
            logger.error(f"Orchestrator init failed (non-fatal): {e}")

        try:
            self.memory = create_memory_manager()
            logger.info(f"✅ Memory: {_cfg().MEMORY_BACKEND}")
        except Exception as e:
            logger.error(f"Memory init failed: {e}")

        # Initialize vision with retry logic
        vision_retries = 3
        for attempt in range(vision_retries):
            try:
                self.vision = StreamingVisionManager(
                    llm_manager=self.llm,
                    memory_manager=self.memory,
                    lava_model=getattr(_cfg(), 'LAVA_MODEL', 'llava:latest'),
                    camera_id=getattr(_cfg(), 'CAMERA_ID', 0),
                    target_fps=getattr(_cfg(), 'CAMERA_FPS', 5)
                )
                logger.info("✅ Vision initialized")
                break
            except Exception as e:
                logger.error(f"Vision init attempt {attempt+1} failed: {e}")
                if attempt < vision_retries - 1:
                    await asyncio.sleep(2)
                else:
                    self.vision = None

        self.audio      = None
        self.conv_audio = None
        self.ready = bool(self.llm)
        self.initializing = False

        # ── Wire Strategic Cognitive Engine ─────────────────────────────────
        try:
            from cognition.strategic_cognitive_engine import StrategicCognitiveEngine
            organism_ref = getattr(self.persona, '_organism', None) if self.persona else None
            if organism_ref:
                self.sce = StrategicCognitiveEngine(organism_ref)
                logger.info("♟️  StrategicCognitiveEngine wired")
            else:
                self.sce = None
        except Exception as _e:
            logger.warning(f"SCE init failed (non-fatal): {_e}")
            self.sce = None

        # ── Wire Lumina Network ─────────────────────────────────────────────────
        try:
            _net_cfg = __import__('managers.settings_manager', fromlist=['config']).config
            if getattr(_net_cfg, 'LUMINA_NETWORK_ENABLED', False):
                from cognition.lumina_network import LuminaNetwork
                self.lumina_network = LuminaNetwork(
                    persona = self.persona,
                    llm     = self.llm,
                )
                _children_cfg = getattr(_net_cfg, 'LUMINA_CHILDREN', [])
                if _children_cfg:
                    self.lumina_network.load_from_config(_children_cfg)
                _master_name = getattr(_net_cfg, 'LUMINA_MASTER_NAME', 'Lumina-Master')
                _master_url  = f"http://127.0.0.1:{getattr(_net_cfg, 'NICEGUI_PORT', 8080)}"
                self.lumina_network.set_master_info(_master_url, _master_name)
                logger.info(f"🌐 LuminaNetwork ready — {len(_children_cfg)} child(ren)")
            else:
                self.lumina_network = None
        except Exception as _e:
            logger.warning(f"LuminaNetwork init failed (non-fatal): {_e}")
            self.lumina_network = None

        # ── Wire Abstract Reasoning Engine ────────────────────────────────────
        try:
            from cognition.abstract_reasoning_engine import AbstractReasoningEngine
            organism_ref = getattr(self.persona, '_organism', None) if self.persona else None
            if organism_ref:
                self.are = AbstractReasoningEngine(
                    organism = organism_ref,
                    llm      = self.llm,
                    memory   = self.memory,
                )
                logger.info("🧠 AbstractReasoningEngine wired")
            else:
                self.are = None
        except Exception as _e:
            logger.warning(f"ARE init failed (non-fatal): {_e}")
            self.are = None

        # ── Wire Active Cognitive Context (CAG Layer 2) ─────────────────────────
        try:
            from cognition.active_cognitive_context import ActiveCognitiveContext
            organism_ref = getattr(self.persona, '_organism', None) if self.persona else None
            if organism_ref:
                self.acc = ActiveCognitiveContext(organism=organism_ref, llm=self.llm)
                logger.info("🧠 ActiveCognitiveContext (CAG) wired")
            else:
                self.acc = None
        except Exception as _e:
            logger.warning(f"ACC init failed (non-fatal): {_e}")
            self.acc = None

        # ── Wire Empathy Engine ──────────────────────────────────────────────
        try:
            from cognition.empathy_engine import EmpathyEngine
            organism_ref = getattr(self.persona, '_organism', None) if self.persona else None
            rm_ref = getattr(organism_ref, 'relational_memory', None) if organism_ref else None
            if organism_ref:
                self.empathy_engine = EmpathyEngine(
                    organism          = organism_ref,
                    llm               = self.llm,
                    relational_memory = rm_ref,
                    multilingual_model_id = getattr(
                        _cfg(), "AFFECT_EMBEDDING_MODEL",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    ),
                )
                logger.info("❤️  EmpathyEngine wired")
            else:
                self.empathy_engine = None
        except Exception as _e:
            logger.warning(f"EmpathyEngine init failed (non-fatal): {_e}")
            self.empathy_engine = None

        # ── Wire social presence engine ──────────────────────────────────
        # Needs: vision manager + organism (from persona) + LLM
        # Called after all three are initialised so nothing is None-dependent.
        if self.vision is not None:
            organism = getattr(self.persona, '_organism', None) if self.persona else None

            def _tts_speak(text: str) -> None:
                """Push text to TTS — walk all available audio handles."""
                try:
                    try:
                        from core.interface_api import record_presence
                        record_presence(text)
                    except Exception:
                        pass
                    # FluidVoice / AudioManager both expose .speak(text)
                    for _handle in (
                        getattr(self, 'audio', None),
                        getattr(self, 'conv_audio', None),
                    ):
                        if _handle is None:
                            continue
                        # Try .speak() directly (FluidVoice, AudioManager)
                        if hasattr(_handle, 'speak'):
                            _handle.speak(text)
                            return
                        # Try .tts.speak() (some wrappers)
                        _tts = getattr(_handle, 'tts', None)
                        if _tts and hasattr(_tts, 'speak'):
                            _tts.speak(text)
                            return
                    logger.info(f"[PresenceEngine TTS] {text}")
                except Exception as _e:
                    logger.debug(f"PresenceEngine TTS: {_e}")

            try:
                self.vision.attach_presence_engine(
                    organism = organism,
                    llm      = self.llm,
                    tts_fn   = _tts_speak,
                )
            except Exception as _e:
                logger.error(f"PresenceEngine wire failed: {_e}")

        # Sensor-only mode keeps external occupancy available when the
        # camera manager/stream is disabled. It has no face identity and does
        # not generate speech by itself.
        if self.vision is None and getattr(_cfg(), 'HOME_ASSISTANT_PRESENCE_ENABLED', False):
            try:
                from cognition.presence_engine import PresenceEngine
                self.presence_engine = PresenceEngine(
                    organism=getattr(self.persona, '_organism', None) if self.persona else None,
                    llm=self.llm,
                )
                logger.info("✅ Sensor-only PresenceEngine attached")
            except Exception as _e:
                logger.warning("Sensor-only PresenceEngine unavailable (non-fatal): %s", _e)

        # Start the optional Home Assistant monitor only when explicitly
        # enabled in config.json. It is read-only and remains outside chat.
        try:
            organism = getattr(self.persona, '_organism', None) if self.persona else None
            if organism is not None:
                from cognition.universal_connector import get_universal_connector
                await get_universal_connector(organism).reconcile_home_assistant_monitor()
        except Exception as _e:
            logger.warning("Home Assistant monitor startup failed (non-fatal): %s", _e)

    def ensure_audio_ready(self) -> bool:
        if self.audio:
            return True
        try:
            self.audio = create_audio_manager()
            self.conv_audio = ConversationalAudioManager(self.audio, self.tts_stop_event)
            self.conv_audio._vision_ref = self.vision
            logger.info("✅ Audio ready (lazy init)")
            return True
        except Exception as e:
            logger.error(f"Audio init failed: {e}")
            self.audio = None
            self.conv_audio = None
            return False

    def reload_audio(self, force: bool = False):
        """
        Reload AudioManager only when TTS/STT provider actually changed.
        VAD settings, voice reference, and other hot-swappable params are
        applied directly without a full reload to avoid the 7s XTTS reload.
        Pass force=True to always reload (e.g. first init).
        """
        cfg = self._refresh_config()

        if self.audio is None:
            return

        # Apply hot-swappable settings directly — no reload needed
        if self.conv_audio:
            self.conv_audio.set_aggressiveness(cfg.VAD_AGGRESSIVENESS)
            self.conv_audio.speech_onset_chunks = cfg.VAD_ONSET_CHUNKS
            self.conv_audio.silence_duration    = cfg.VAD_SILENCE_DURATION
            self.conv_audio.min_speech_duration = cfg.VAD_MIN_SPEECH_DURATION
            self.conv_audio.energy_gate_factor  = cfg.VAD_ENERGY_GATE_FACTOR
            self.conv_audio.tts_post_roll_ms    = getattr(cfg, 'TTS_POST_ROLL_MS', 250)
            logger.info("✅ VAD settings applied (no reload)")

        # Only reload the heavy audio stack if TTS/STT provider changed
        def _canonical_tts(provider):
            return 'edge_tts' if provider == 'edge' else provider

        current_tts = self.audio.tts_engine.get('type') if self.audio.tts_engine else None
        current_stt = self.audio.stt_engine.get('type') if self.audio.stt_engine else None
        tts_changed = current_tts != _canonical_tts(cfg.TTS_PROVIDER)
        stt_changed = current_stt != cfg.STT_PROVIDER

        if not force and not tts_changed and not stt_changed:
            # Hot-swap voice reference if it changed
            if self.audio.tts_engine and cfg.COQUI_VOICE_REFERENCE:
                self.audio.update_voice_reference(cfg.COQUI_VOICE_REFERENCE)
            # Hot-swap language if it changed (Coqui only)
            if self.audio.tts_engine and self.audio.tts_engine.get('type') == 'coqui':
                self.audio.update_language(cfg.VOICE_LANGUAGE)
            logger.info("✅ Audio settings applied (no model reload needed)")
            return

        try:
            logger.info(f"Reloading audio (TTS changed={tts_changed}, STT changed={stt_changed})")
            if self.conv_audio and self.conv_audio.is_listening:
                self.conv_audio.stop_conversation()
            # Explicitly tear down Coqui/PyTorch before initialising the new
            # provider — prevents heap corruption when switching Coqui → pyttsx3
            if self.audio is not None:
                try:
                    self.audio.teardown()
                except Exception as _te:
                    logger.debug(f"Audio teardown warning (non-fatal): {_te}")
                self.audio = None
                self.conv_audio = None
            self.audio = create_audio_manager()
            if self.audio:
                self.conv_audio = ConversationalAudioManager(self.audio, self.tts_stop_event)
                self.conv_audio._vision_ref = self.vision
            logger.info("✅ Audio reloaded")
        except Exception as e:
            logger.error(f"Audio reload failed: {e}")

    def _refresh_config(self):
        """Reload config from JSON and update the module-level singleton.
        This is the single source of truth — no in-memory object sync needed."""
        fresh = load_settings()
        _settings_mod.config = fresh
        return fresh

    def reload_llm(self):
        cfg = self._refresh_config()
        try:
            # Save history BEFORE creating new LLMManager (which starts empty)
            from managers.session_manager import get_session_manager
            sess = get_session_manager()
            old_history = sess.save_before_reload(self.llm.history if self.llm else [])

            self.llm = create_llm_manager()

            # Restore history into the fresh LLMManager
            restored = sess.restore_after_reload(self.llm)
            # Give session manager a reference to the new LLM fn for compression
            sess.set_llm_fn(self.llm.generate_bare)

            # ── Rewire PersonaBridge so it uses the new LLM ────────────────
            # Without this, persona keeps calling the OLD provider even after
            # the user changes model/URL in settings.
            if self.persona is not None:
                new_bare   = getattr(self.llm, "generate_bare",   None) or getattr(self.llm, "generate", None)
                new_stream = getattr(self.llm, "generate_stream", None)
                # Swap the streaming fn used by the chat path
                self.persona._llm_stream_fn = new_stream
                # Goal-means analysis resolves its history-free interactive
                # method through this bound callable. Keep it on the same new
                # manager as the visible stream after an LLM settings reload.
                self.persona._external_llm_fn = new_bare
                # Swap the bare fn inside Lumina's ExternalLLMAdapter
                try:
                    if new_bare and self.persona._system and self.persona._system.llm:
                        self.persona._system.llm._fn = new_bare
                        self.persona._system.llm.config.model_name = cfg.LLM_MODEL
                except Exception as _re:
                    logger.warning(f"Persona LLM adapter swap warning (non-fatal): {_re}")
                logger.info("✅ PersonaBridge LLM rewired to new provider")

            logger.info(f"✅ LLM reloaded: {cfg.LLM_PROVIDER}/{cfg.LLM_MODEL} ({restored} history turns restored)")
        except Exception as e:
            logger.error(f"LLM reload failed: {e}")

    def reload_memory(self):
        cfg = self._refresh_config()
        try:
            self.memory = create_memory_manager()
            logger.info(f"✅ Memory reloaded: {cfg.MEMORY_BACKEND}")
        except Exception as e:
            logger.error(f"Memory reload failed: {e}")

    def reload_vision(self):
        """Reload vision with new settings"""
        cfg = self._refresh_config()
        try:
            if self.vision:
                self.vision.cleanup()
            self.vision = StreamingVisionManager(
                llm_manager=self.llm,
                memory_manager=self.memory,
                lava_model=getattr(cfg, 'LAVA_MODEL', 'llava:latest'),
                camera_id=getattr(cfg, 'CAMERA_ID', 0),
                target_fps=getattr(cfg, 'CAMERA_FPS', 5)
            )
            logger.info("✅ Vision reloaded")
        except Exception as e:
            logger.error(f"Vision reload failed: {e}")
            self.vision = None

    def add_memory(self, content: str, role: str):
        if self.memory:
            try:
                self.memory.add_memory(content, {'role': role, 'timestamp': datetime.now().isoformat()})
            except Exception as e:
                logger.warning(f"Memory add failed: {e}")

    def get_context(self, query: str) -> str:
        if self.memory:
            try:
                return self.memory.get_context(query)
            except Exception as e:
                logger.warning(f"Memory get failed: {e}")
        return ""
    
    def get_uptime(self):
        """Get system uptime"""
        return time.time() - self.start_time


state = AppState()
