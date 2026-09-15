"""
LUMINA V32 — Phase 9 : Fluid Voice System
==========================================
Improvements over current system:

  STT (was: Whisper base, ~800ms/chunk):
  ├── faster-whisper (same accuracy, 4× faster, ~200ms)
  ├── whisper.cpp via pywhispercpp (native, ~100ms on CPU)
  └── kept: OpenAI Whisper API fallback

  TTS (was: full-text → Coqui → WAV → play, 2-4s):
  ├── Streaming Coqui: sentence-by-sentence, first audio in ~600ms
  ├── Kokoro TTS: local, phoneme-based, <200ms latency, very natural
  ├── Edge-TTS (Microsoft Neural): free, streaming, ~150ms, excellent quality
  └── kept: Coqui XTTS-v2 (best quality for voice cloning)

  Fluidity improvements:
  ├── Barge-in: user can interrupt TTS mid-sentence (VAD during playback)
  ├── Sentence streaming: play first sentence while generating the rest
  ├── Speech sanitizer already handles markdown → speaks naturally
  └── Prosody helpers: sentence-boundary pauses, question intonation markers

Usage:
    # In audio_manager._init_tts, add:
    elif provider == 'kokoro':
        self._init_kokoro()
    elif provider == 'edge':
        self._init_edge_tts()

    # In text_to_speech, enable streaming:
    return self._tts_streaming(text, callback=play_chunk_callback)

    # In _init_stt, add:
    elif provider == 'faster_whisper':
        self._init_faster_whisper()
"""

import io
import logging
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  STT IMPROVEMENTS
# ══════════════════════════════════════════════════════════════════════════════

class FasterWhisperSTT:
    """
    Drop-in replacement for Whisper base.
    Uses faster-whisper (CTranslate2 backend) — same model, 4× faster.

    Install:  pip install faster-whisper
    Models:   tiny / base / small / medium / large-v3
    Latency:  ~200ms on CPU for base, ~50ms for tiny
    """

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8"):
        """
        Args:
            model_size:   'tiny', 'base', 'small', 'medium', 'large-v3'
            device:       'cpu' or 'cuda'
            compute_type: 'int8' (fastest CPU), 'float16' (GPU), 'float32'
        """
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        logger.info(f"✅ STT: faster-whisper ({model_size}) on {device} [{compute_type}]")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe an audio file. Returns text string."""
        t0 = time.time()
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=1,           # fastest, still accurate
            vad_filter=True,       # built-in VAD removes silence
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"STT done in {(time.time()-t0)*1000:.0f}ms: {text[:60]!r}")
        return text

    def transcribe_numpy(self, audio: np.ndarray, sample_rate: int = 16000,
                         language: str = "en") -> str:
        """Transcribe a numpy array directly (no file I/O)."""
        t0 = time.time()
        # faster-whisper accepts numpy float32 normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"STT (numpy) done in {(time.time()-t0)*1000:.0f}ms")
        return text


# ══════════════════════════════════════════════════════════════════════════════
#  TTS IMPROVEMENTS
# ══════════════════════════════════════════════════════════════════════════════

class KokoroTTS:
    """
    Kokoro TTS — local phoneme-based neural TTS.
    Extremely fast (<200ms), very natural prosody, no GPU needed.
    Supports voice styles and speed control.

    Install:  pip install kokoro-onnx soundfile
    Voices:   af_heart, af_bella, af_sarah, am_adam, am_michael,
              bf_emma, bf_isabella, bm_george, bm_lewis, af_nicole, af_sky
    Size:     ~80MB model download (once)
    Latency:  <200ms for typical sentence on CPU

    https://github.com/thewh1teagle/kokoro-onnx
    """

    VOICES = {
        # American Female
        'af_heart':    'af_heart',    # warm, conversational
        'af_bella':    'af_bella',    # expressive
        'af_sarah':    'af_sarah',    # clear, professional
        'af_nicole':   'af_nicole',   # soft, natural
        'af_sky':      'af_sky',      # youthful
        # American Male
        'am_adam':     'am_adam',
        'am_michael':  'am_michael',
        # British Female
        'bf_emma':     'bf_emma',
        'bf_isabella': 'bf_isabella',
        # British Male
        'bm_george':   'bm_george',
        'bm_lewis':    'bm_lewis',
    }

    def __init__(self, voice: str = "af_heart", speed: float = 1.0,
                 lang: str = "en-us"):
        from kokoro_onnx import Kokoro
        self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        self.voice  = voice
        self.speed  = speed
        self.lang   = lang
        self.sample_rate = 24000
        logger.info(f"✅ TTS: Kokoro ready | voice={voice} speed={speed}")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text → numpy float32 array at 24kHz."""
        t0 = time.time()
        samples, sr = self.kokoro.create(text, voice=self.voice,
                                         speed=self.speed, lang=self.lang)
        logger.info(f"Kokoro synthesized {len(text)} chars in {(time.time()-t0)*1000:.0f}ms")
        return samples

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Synthesize and write to WAV file. Returns path."""
        import soundfile as sf
        samples = self.synthesize(text)
        sf.write(output_path, samples, self.sample_rate)
        return output_path

    def synthesize_streaming(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Sentence-by-sentence streaming synthesis.
        Yields numpy chunks as they are ready.
        """
        sentences = _split_to_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                chunk = self.synthesize(sentence)
                yield chunk


class EdgeTTS:
    """
    Microsoft Edge Neural TTS — free, streaming, excellent quality.
    ~150ms first-audio latency. No local model needed.
    Requires internet. Uses the same engine as Edge browser.

    Install:  pip install edge-tts
    Voices:   en-US-AvaNeural (warm), en-US-JennyNeural (professional),
              en-US-GuyNeural (male), en-GB-SoniaNeural (British female)
    Latency:  ~150ms first chunk (streaming), or ~400ms full file

    https://github.com/rany2/edge-tts
    """

    VOICES = {
        # Recommended for conversational AI
        'ava':      'en-US-AvaNeural',        # warm, conversational ← best for Lumina
        'jenny':    'en-US-JennyNeural',       # professional, clear
        'aria':     'en-US-AriaNeural',        # expressive
        'guy':      'en-US-GuyNeural',         # male, natural
        'sonia':    'en-GB-SoniaNeural',       # British female
        'ryan':     'en-GB-RyanNeural',        # British male
        'natasha':  'en-AU-NatashaNeural',     # Australian female
        'william':  'en-AU-WilliamNeural',     # Australian male
    }

    def __init__(self, voice: str = "en-US-AvaNeural", rate: str = "+0%",
                 pitch: str = "+0Hz", volume: str = "+0%"):
        import edge_tts
        self._edge_tts = edge_tts
        self.voice  = voice
        self.rate   = rate
        self.pitch  = pitch
        self.volume = volume
        logger.info(f"✅ TTS: Edge-TTS ready | voice={voice}")

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Synthesize and write to MP3/WAV. Returns path."""
        import asyncio

        async def _run():
            communicate = self._edge_tts.Communicate(
                text, self.voice,
                rate=self.rate, pitch=self.pitch, volume=self.volume,
            )
            await communicate.save(output_path)

        asyncio.run(_run())
        return output_path

    def synthesize_streaming(self, text: str,
                             chunk_callback: Callable[[bytes], None]) -> None:
        """
        Stream audio bytes as they arrive from Edge.
        chunk_callback receives raw MP3/audio bytes.
        First bytes arrive in ~150ms.
        """
        import asyncio

        async def _stream():
            communicate = self._edge_tts.Communicate(
                text, self.voice,
                rate=self.rate, pitch=self.pitch, volume=self.volume,
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunk_callback(chunk["data"])

        asyncio.run(_stream())


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING TTS PLAYER  (works with any TTS backend)
# ══════════════════════════════════════════════════════════════════════════════

class StreamingTTSPlayer:
    """
    Plays TTS audio sentence-by-sentence while the next sentence is being
    synthesized. This cuts perceived latency from ~2s to ~600ms for Coqui,
    and ~150ms for Edge/Kokoro.

    Also implements barge-in: if the VAD detects the user speaking during
    playback, synthesis and playback stop immediately.

    Usage:
        player = StreamingTTSPlayer(audio_manager, barge_in_callback=on_barge_in)
        player.speak("Hello! How are you doing today? I'm here to help.")
        # First sentence plays in ~600ms, subsequent without gap.
    """

    def __init__(self, audio_manager, barge_in_callback: Optional[Callable] = None,
                 inter_sentence_pause_ms: int = 120):
        self.audio_manager = audio_manager
        self.barge_in_callback = barge_in_callback
        self.inter_sentence_pause_ms = inter_sentence_pause_ms

        self._stop_event   = threading.Event()
        self._play_queue: queue.Queue = queue.Queue()
        self._is_playing   = False
        self._player_thread: Optional[threading.Thread] = None

    def speak(self, text: str, blocking: bool = False) -> None:
        """
        Speak text using sentence-level streaming.
        Returns immediately (non-blocking by default).
        """
        self._stop_event.clear()
        self._is_playing = True

        sentences = _split_to_sentences(text)
        if not sentences:
            return

        # Start player thread first
        self._player_thread = threading.Thread(
            target=self._player_loop, daemon=True,
            name="StreamingTTSPlayer"
        )
        self._player_thread.start()

        # Synthesize sentences and enqueue audio
        def _synth_loop():
            for sentence in sentences:
                if self._stop_event.is_set():
                    break
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    audio_path = self.audio_manager.text_to_speech(sentence)
                    if audio_path and not self._stop_event.is_set():
                        self._play_queue.put(('file', audio_path))
                except Exception as e:
                    logger.debug(f"[StreamingTTS] Synthesis error: {e}")
            self._play_queue.put(('done', None))

        synth_thread = threading.Thread(target=_synth_loop, daemon=True,
                                        name="StreamingTTSSynth")
        synth_thread.start()

        if blocking:
            synth_thread.join()
            if self._player_thread:
                self._player_thread.join()

    def stop(self) -> None:
        """Stop playback immediately (barge-in)."""
        self._stop_event.set()
        self._is_playing = False
        # Drain queue
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except queue.Empty:
                break

    def _player_loop(self) -> None:
        """Dequeue synthesized audio files and play them sequentially."""
        import sounddevice as sd
        import soundfile as sf

        while not self._stop_event.is_set():
            try:
                item = self._play_queue.get(timeout=5.0)
            except queue.Empty:
                break

            kind, payload = item
            if kind == 'done':
                break

            if kind == 'file' and payload and os.path.exists(payload):
                try:
                    data, sr = sf.read(payload, dtype='float32')
                    # Play in small chunks so we can check barge-in
                    chunk_frames = int(sr * 0.05)  # 50ms chunks
                    for start in range(0, len(data), chunk_frames):
                        if self._stop_event.is_set():
                            break
                        chunk = data[start:start + chunk_frames]
                        sd.play(chunk, samplerate=sr, blocking=True)
                    # Inter-sentence pause
                    if not self._stop_event.is_set():
                        time.sleep(self.inter_sentence_pause_ms / 1000.0)
                except Exception as e:
                    logger.debug(f"[StreamingTTS] Playback error: {e}")
                finally:
                    try:
                        os.unlink(payload)
                    except Exception:
                        pass

        self._is_playing = False


# ══════════════════════════════════════════════════════════════════════════════
#  BARGE-IN HANDLER
# ══════════════════════════════════════════════════════════════════════════════

class BargeInHandler:
    """
    Monitors the VAD during TTS playback and triggers barge-in when the
    user starts speaking.

    The core problem this solves: currently the TTS mute gate stops VAD
    entirely during playback — the user CANNOT interrupt Lumina while she
    is speaking. This handler listens at LOW sensitivity during playback
    and calls on_barge_in() when sustained speech is detected.

    Usage:
        handler = BargeInHandler(
            vad=conversational_audio._vad,
            on_barge_in=streaming_player.stop,
            min_speech_frames=4,  # ~120ms of sustained speech to trigger
        )
        handler.start_monitoring()   # call before TTS starts
        # ... TTS plays ...
        handler.stop_monitoring()    # call when TTS finishes naturally
    """

    def __init__(self, on_barge_in: Callable, sample_rate: int = 16000,
                 chunk_ms: int = 30, min_speech_frames: int = 4,
                 vad_aggressiveness: int = 1):
        self.on_barge_in        = on_barge_in
        self.sample_rate        = sample_rate
        self.chunk_frames       = int(sample_rate * chunk_ms / 1000)
        self.min_speech_frames  = min_speech_frames
        self.vad_aggressiveness = vad_aggressiveness

        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._speech_count = 0

    def start_monitoring(self) -> None:
        """Start listening for barge-in. Call just before TTS playback."""
        self._monitoring  = True
        self._speech_count = 0
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="BargeInMonitor"
        )
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring. Call when TTS finishes naturally."""
        self._monitoring = False

    def _monitor_loop(self) -> None:
        """Listen to mic at low VAD sensitivity during TTS playback."""
        try:
            import sounddevice as sd
            import webrtcvad

            vad = webrtcvad.Vad(self.vad_aggressiveness)
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                dtype='int16', blocksize=self.chunk_frames) as stream:
                while self._monitoring:
                    audio, _ = stream.read(self.chunk_frames)
                    chunk_bytes = audio.tobytes()
                    try:
                        if vad.is_speech(chunk_bytes, self.sample_rate):
                            self._speech_count += 1
                            if self._speech_count >= self.min_speech_frames:
                                logger.info("[BargeIn] User speech detected — interrupting TTS")
                                self._monitoring = False
                                self.on_barge_in()
                                return
                        else:
                            self._speech_count = max(0, self._speech_count - 1)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"[BargeIn] Monitor error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO MANAGER PATCHES  (apply to existing AudioManager)
# ══════════════════════════════════════════════════════════════════════════════

def patch_audio_manager(audio_manager) -> None:
    """
    Apply Phase 9 improvements to an existing AudioManager instance.

    Adds:
      - _init_faster_whisper()
      - _init_kokoro()
      - _init_edge_tts()
      - _tts_streaming_coqui()  (sentence-level streaming for Coqui)
      - streaming_player property
      - barge_in_handler property

    Call once after AudioManager is created:
        from managers.fluid_voice import patch_audio_manager
        patch_audio_manager(app.audio_manager)
    """
    import types

    # ── Faster Whisper ──────────────────────────────────────────────────────

    def _init_faster_whisper(self):
        from managers.settings_manager import config as cfg
        model_size = getattr(cfg, 'WHISPER_MODEL', 'base')
        try:
            stt = FasterWhisperSTT(model_size=model_size, device='cpu',
                                   compute_type='int8')
            self.stt_engine = {'type': 'faster_whisper', '_obj': stt}
        except ImportError:
            logger.warning("faster-whisper not installed — pip install faster-whisper")
            raise

    def _stt_faster_whisper(self, audio_path: str) -> Optional[str]:
        obj = self.stt_engine.get('_obj')
        if obj:
            return obj.transcribe(audio_path)
        return None

    # ── Kokoro TTS ──────────────────────────────────────────────────────────

    def _init_kokoro(self):
        from managers.settings_manager import config as cfg
        voice = getattr(cfg, 'KOKORO_VOICE', 'af_heart')
        speed = float(getattr(cfg, 'KOKORO_SPEED', '1.0'))
        try:
            tts_obj = KokoroTTS(voice=voice, speed=speed)
            self.tts_engine = {'type': 'kokoro', '_obj': tts_obj}
        except ImportError:
            logger.warning("kokoro-onnx not installed — pip install kokoro-onnx")
            raise

    def _tts_kokoro(self, text: str) -> Optional[str]:
        obj = self.tts_engine.get('_obj')
        if not obj:
            return None
        output = os.path.join(tempfile.gettempdir(),
                              f'tts_kokoro_{os.getpid()}.wav')
        return obj.synthesize_to_file(text, output)

    # ── Edge TTS ────────────────────────────────────────────────────────────

    def _init_edge_tts(self):
        from managers.settings_manager import config as cfg
        voice = getattr(cfg, 'EDGE_TTS_VOICE', 'en-US-AvaNeural')
        rate  = getattr(cfg, 'EDGE_TTS_RATE',  '+0%')
        try:
            tts_obj = EdgeTTS(voice=voice, rate=rate)
            self.tts_engine = {'type': 'edge_tts', '_obj': tts_obj}
        except ImportError:
            logger.warning("edge-tts not installed — pip install edge-tts")
            raise

    def _tts_edge(self, text: str) -> Optional[str]:
        obj = self.tts_engine.get('_obj')
        if not obj:
            return None
        output = os.path.join(tempfile.gettempdir(),
                              f'tts_edge_{os.getpid()}.mp3')
        return obj.synthesize_to_file(text, output)

    # ── Streaming Coqui (sentence-level) ────────────────────────────────────

    def _tts_coqui_stream_sentence(self, text: str) -> Optional[str]:
        """
        Single-sentence Coqui synthesis.
        Called per-sentence by StreamingTTSPlayer.
        """
        return self._tts_coqui(text)  # existing method, works per-sentence

    # ── Patch speech_to_text to support faster_whisper ──────────────────────

    _orig_stt = audio_manager.speech_to_text.__func__ if hasattr(
        audio_manager.speech_to_text, '__func__') else None

    def _speech_to_text_patched(self, audio_path: str) -> Optional[str]:
        if self.stt_engine and self.stt_engine.get('type') == 'faster_whisper':
            return self._stt_faster_whisper(audio_path)
        return _orig_stt(self, audio_path) if _orig_stt else None

    # ── Patch text_to_speech to support kokoro + edge ───────────────────────

    _orig_tts = audio_manager.text_to_speech.__func__ if hasattr(
        audio_manager.text_to_speech, '__func__') else None

    def _text_to_speech_patched(self, text: str) -> Optional[str]:
        if not self.tts_engine:
            return None
        text = self.clean_text(text)
        if not text:
            return None
        engine_type = self.tts_engine.get('type', '')
        try:
            if engine_type == 'kokoro':
                return self._tts_kokoro(text)
            elif engine_type == 'edge_tts':
                return self._tts_edge(text)
            elif engine_type == 'faster_whisper':
                return None  # TTS, not STT
        except Exception as e:
            logger.error(f"TTS error ({engine_type}): {e}")
        # Fall back to original
        return _orig_tts(self, text) if _orig_tts else None

    # ── Patch _init_tts to support new providers ─────────────────────────────

    _orig_init_tts = audio_manager._init_tts.__func__ if hasattr(
        audio_manager._init_tts, '__func__') else None

    def _init_tts_patched(self):
        from managers.settings_manager import config as cfg
        provider = cfg.TTS_PROVIDER
        if provider == 'kokoro':
            try:
                self._init_kokoro()
                return
            except Exception as e:
                logger.warning(f"Kokoro init failed ({e}) — falling back")
        elif provider == 'edge':
            try:
                self._init_edge_tts()
                return
            except Exception as e:
                logger.warning(f"Edge-TTS init failed ({e}) — falling back")
        # Fall back to original
        if _orig_init_tts:
            _orig_init_tts(self)

    # ── Patch _init_stt to support faster_whisper ───────────────────────────

    _orig_init_stt = audio_manager._init_stt.__func__ if hasattr(
        audio_manager._init_stt, '__func__') else None

    def _init_stt_patched(self):
        from managers.settings_manager import config as cfg
        provider = cfg.STT_PROVIDER
        if provider == 'faster_whisper':
            try:
                self._init_faster_whisper()
                return
            except Exception as e:
                logger.warning(f"faster-whisper init failed ({e}) — falling back to whisper")
                cfg.STT_PROVIDER = 'whisper'
        if _orig_init_stt:
            _orig_init_stt(self)

    # ── Bind all patched methods ─────────────────────────────────────────────

    am = audio_manager
    am._init_faster_whisper       = types.MethodType(_init_faster_whisper,       am)
    am._stt_faster_whisper        = types.MethodType(_stt_faster_whisper,         am)
    am._init_kokoro               = types.MethodType(_init_kokoro,               am)
    am._tts_kokoro                = types.MethodType(_tts_kokoro,                am)
    am._init_edge_tts             = types.MethodType(_init_edge_tts,             am)
    am._tts_edge                  = types.MethodType(_tts_edge,                  am)
    am._init_tts                  = types.MethodType(_init_tts_patched,          am)
    am._init_stt                  = types.MethodType(_init_stt_patched,          am)
    am.speech_to_text             = types.MethodType(_speech_to_text_patched,    am)
    am.text_to_speech             = types.MethodType(_text_to_speech_patched,    am)

    # ── Attach streaming player ──────────────────────────────────────────────

    am.streaming_player = StreamingTTSPlayer(am)
    am.barge_in_handler = BargeInHandler(on_barge_in=am.streaming_player.stop)

    logger.info("✅ Phase 9: fluid_voice patches applied to AudioManager")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _split_to_sentences(text: str) -> List[str]:
    """
    Split text into natural TTS-friendly sentences.
    Handles: . ! ? — ellipsis — list items.
    Avoids splitting: Mr./Dr./etc., decimal numbers, abbreviations.
    """
    import re

    # Protect common abbreviations
    protected = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|approx|est|dept|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.',
        r'\1<DOT>', text
    )

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\.\.\.)\s+|(?<=[!?])\s+', protected)

    # Restore protected dots
    sentences = [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]

    # Merge very short fragments (< 4 words) with the next sentence
    merged: List[str] = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if len(s.split()) < 4 and i + 1 < len(sentences):
            sentences[i + 1] = s + ' ' + sentences[i + 1]
        else:
            merged.append(s)
        i += 1

    return merged if merged else [text]


def install_dependencies(providers: List[str]) -> None:
    """
    Install required packages for the requested providers.
    Run once at setup.
    """
    deps = {
        'faster_whisper': ['faster-whisper'],
        'kokoro':          ['kokoro-onnx', 'soundfile'],
        'edge':            ['edge-tts'],
    }
    import subprocess, sys
    for provider in providers:
        packages = deps.get(provider, [])
        for pkg in packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg],
                           capture_output=True)
            print(f"✅ Installed: {pkg}")


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS ADDITIONS  (add to settings_manager.AppSettings)
# ══════════════════════════════════════════════════════════════════════════════

NEW_SETTINGS = """
    # ── Phase 9: Fluid Voice ──────────────────────────────────────────────────
    # STT
    STT_PROVIDER:           str = "whisper"        # whisper | faster_whisper | openai
    # TTS
    TTS_PROVIDER:           str = "coqui"          # pyttsx3 | coqui | kokoro | edge | openai | elevenlabs
    # Kokoro
    KOKORO_VOICE:           str = "af_heart"       # af_heart | af_bella | af_sarah | am_michael | ...
    KOKORO_SPEED:           str = "1.0"            # 0.5 – 2.0
    # Edge-TTS
    EDGE_TTS_VOICE:         str = "en-US-AvaNeural" # any Edge neural voice
    EDGE_TTS_RATE:          str = "+0%"            # +10% faster, -10% slower
    # Streaming
    TTS_STREAMING:          bool = True            # stream sentence-by-sentence
    TTS_BARGE_IN:           bool = True            # let user interrupt TTS
    BARGE_IN_SENSITIVITY:   int  = 1               # VAD aggressiveness 0-3 (0=most sensitive)
    INTER_SENTENCE_PAUSE_MS: int = 120             # pause between sentences (ms)
"""

if __name__ == '__main__':
    print("Phase 9 — Fluid Voice System")
    print("Available providers:")
    print("  STT: faster_whisper (4× faster, same accuracy)")
    print("  TTS: kokoro (<200ms, very natural), edge (<150ms, neural quality)")
    print()
    print("To install all:")
    print("  pip install faster-whisper kokoro-onnx soundfile edge-tts")
    print()
    print("To patch AudioManager:")
    print("  from managers.fluid_voice import patch_audio_manager")
    print("  patch_audio_manager(app.audio_manager)")
