"""
Audio Manager
- TTS: pyttsx3 (offline) | Coqui XTTS-v2 (voice cloning) | OpenAI | ElevenLabs
- STT: Whisper (offline) | OpenAI Whisper API
- Thread-safe, with Windows SAPI fallback for pyttsx3
"""
import os
import re
import logging
import tempfile
import threading
import subprocess
import sys
import queue
from typing import Optional
from pathlib import Path

# Phase 9: fluid voice patches
try:
    from managers.fluid_voice import patch_audio_manager as _patch_fluid_voice
except Exception:
    _patch_fluid_voice = None

logger = logging.getLogger(__name__)

# Global Coqui instance cache to prevent multiple 1.8GB loads
_COQUI_INSTANCE = None
_COQUI_INSTANCE_LOCK = threading.RLock()
_COQUI_LOADED = threading.Event()


def _get_coqui_instance():
    """Return cached Coqui instance to avoid 1.8GB reloads"""
    global _COQUI_INSTANCE
    
    if _COQUI_LOADED.is_set():
        return _COQUI_INSTANCE
        
    with _COQUI_INSTANCE_LOCK:
        if not _COQUI_LOADED.is_set():
            logger.info("Loading Coqui XTTS-v2 (first load ~1.8GB)...")
            from TTS.api import TTS
            device = _detect_best_device()
            _COQUI_INSTANCE = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True,
                gpu=(device == "cuda")
            )
            _COQUI_LOADED.set()
            logger.info(f"✅ Coqui XTTS-v2 loaded on {device}")
    return _COQUI_INSTANCE


def _release_coqui_model():
    """
    Explicitly release the Coqui/PyTorch model from the global cache and
    force garbage collection + GPU cache flush BEFORE pyttsx3 initialises.

    Without this, switching Coqui → pyttsx3 causes heap corruption:
      - Coqui brings in libgomp (OpenMP) and optionally libcudart
      - pyttsx3 loads libespeak-ng which uses the system C allocator
      - When Python's GC finalises Coqui tensors while espeak is initialising,
        two different allocators race on the same heap → 'corrupted double-linked list'
    """
    global _COQUI_INSTANCE, _COQUI_LOADED
    import gc
    try:
        with _COQUI_INSTANCE_LOCK:
            if _COQUI_INSTANCE is not None:
                logger.info("Releasing Coqui model from memory before TTS switch...")
                # Drop all references to the model and its tensors
                try:
                    # Try to move model to CPU first to free GPU VRAM
                    if hasattr(_COQUI_INSTANCE, 'synthesizer') and _COQUI_INSTANCE.synthesizer:
                        synth = _COQUI_INSTANCE.synthesizer
                        if hasattr(synth, 'tts_model') and synth.tts_model:
                            synth.tts_model.cpu()
                        if hasattr(synth, 'vocoder_model') and synth.vocoder_model:
                            synth.vocoder_model.cpu()
                except Exception:
                    pass
                _COQUI_INSTANCE = None
                _COQUI_LOADED.clear()
                # Force Python GC to finalise the model before native libs conflict
                gc.collect()
                # Flush PyTorch GPU allocator if torch is loaded
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
                # Small sleep to let the OS settle the allocator
                import time as _time
                _time.sleep(0.15)
                logger.info("✅ Coqui model released")
    except Exception as e:
        logger.warning(f"Coqui release warning (non-fatal): {e}")


def _patch_torch_safe_globals():
    """
    PyTorch 2.6 changed torch.load() to weights_only=True by default.
    The old Coqui TTS package (pip install TTS) saves XttsConfig as a Python
    class inside the checkpoint, which is now blocked.
    The maintained fork (pip install coqui-tts >=0.26.0) fixes this internally.
    If the old package is installed, we patch safe_globals as a best-effort fix.
    """
    try:
        import torch
        # Only needed for torch >= 2.6
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        if (major, minor) < (2, 6):
            return
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])
            logger.info("✅ PyTorch 2.6+ safe_globals patch applied for XTTS checkpoint")
        except Exception:
            # coqui-tts fork handles this itself; or classes not available — skip
            pass
    except Exception:
        pass


def _detect_best_device() -> str:
    """
    Detect the best available compute device for Coqui / PyTorch.
    """
    try:
        import torch

        # On ROCm builds: torch.version.hip is set; CUDA builds: torch.version.cuda
        is_rocm  = bool(getattr(torch.version, 'hip', None))
        is_cuda  = bool(getattr(torch.version, 'cuda', None))
        gpu_avail = torch.cuda.is_available()

        if gpu_avail:
            name = torch.cuda.get_device_name(0)
            if is_rocm:
                logger.info(f"✅ AMD GPU detected via ROCm/HIP: {name}")
            else:
                logger.info(f"✅ NVIDIA GPU detected: {name}")
            return "cuda"
        else:
            if is_rocm:
                logger.warning(
                    "ROCm-built PyTorch is installed but no GPU was found. "
                    "Falling back to CPU."
                )
            elif is_cuda:
                logger.warning(
                    "CUDA-built PyTorch installed but no NVIDIA GPU found. "
                    "Falling back to CPU."
                )
            else:
                logger.info("No GPU found (CPU-only PyTorch). Coqui will run on CPU.")
            return "cpu"
    except Exception as e:
        logger.error(f"Device detection error: {e}")
        return "cpu"


def _check_coqui_installed() -> tuple:
    """
    Detect whether the MAINTAINED Coqui TTS fork is installed and importable.
    Returns (is_importable: bool, error_message: str).
    """
    try:
        import TTS  # noqa: F401

        # Try to detect whether it's the old dead package
        try:
            from TTS.tts.layers.xtts import stream_generator  # noqa: F401
        except ImportError as e:
            if 'BeamSearchScorer' in str(e) or 'LogitsWarper' in str(e):
                return False, (
                    "You have the OLD, abandoned 'TTS' package installed. "
                    "Fix — uninstall it and install the maintained fork:\n"
                    "  pip uninstall TTS -y\n"
                    "  pip install coqui-tts librosa pydub"
                )

        return True, ""

    except ImportError as e:
        msg = str(e)
        if "No module named 'TTS'" in msg:
            return False, (
                "Coqui TTS is not installed. Install the maintained fork:\n"
                "  pip install coqui-tts librosa pydub"
            )
        elif "torchaudio" in msg:
            return False, (
                "torchaudio is not installed. It must match your torch version.\n"
                "Install for your hardware."
            )
        elif "librosa" in msg:
            return False, "librosa is not installed. Run: pip install librosa"
        elif "pydub" in msg:
            return False, "pydub is not installed. Run: pip install pydub"
        elif "numpy" in msg:
            return False, "numpy is not installed or incompatible. Run: pip install --upgrade numpy"
        elif "soundfile" in msg:
            return False, "soundfile is not installed. Run: pip install soundfile"
        else:
            return False, f"Coqui TTS sub-dependency missing: {msg}"
    except Exception as e:
        return False, f"Coqui TTS unexpected import error: {e}"


def _warmup_coqui(tts, language: str) -> bool:
    """
    Warm up Coqui model with a short generation to load it into memory.
    Returns True if warmup successful.
    """
    try:
        import numpy as np
        import soundfile as sf
        import tempfile
        import os
        
        # Create a tiny silent audio file
        warmup_path = os.path.join(tempfile.gettempdir(), f'coqui_warmup_{os.getpid()}.wav')
        
        # Generate a very short phrase to warm up the model
        tts.tts_to_file(
            text="Hello",
            language=language,
            file_path=warmup_path
        )
        
        # Clean up
        if os.path.exists(warmup_path):
            os.remove(warmup_path)
        
        logger.info("✅ Coqui model warmed up successfully")
        return True
    except Exception as e:
        logger.debug(f"Coqui warmup skipped (non-critical): {e}")
        return False


class AudioManager:
    def __init__(self):
        self._tts_lock  = threading.RLock()  # Use RLock for reentrant locking
        self._stt_lock  = threading.RLock()
        self.tts_engine: Optional[dict] = None
        self.stt_engine: Optional[dict] = None
        self.sample_rate = 16000
        self._generation_timeout = 45  # seconds

        from managers.settings_manager import config
        self._cfg = config

        self._apply_fluid_patches()
        self._init_tts()
        self._init_stt()

    # ─────────────────────────────────────────────────────────────────
    #  Init
    # ─────────────────────────────────────────────────────────────────

    def _init_tts(self):
        provider = self._cfg.TTS_PROVIDER
        logger.info(f"Initialising TTS: {provider}")
        try:
            if provider == 'coqui':
                self._init_coqui()
            elif provider == 'openai' and self._cfg.OPENAI_API_KEY:
                from openai import OpenAI
                self.tts_engine = {
                    'type':   'openai',
                    'client': OpenAI(api_key=self._cfg.OPENAI_API_KEY)
                }
                logger.info("✅ TTS: OpenAI ready")
            elif provider == 'elevenlabs' and self._cfg.ELEVENLABS_API_KEY:
                self.tts_engine = {
                    'type':     'elevenlabs',
                    'api_key':  self._cfg.ELEVENLABS_API_KEY,
                    'voice_id': self._cfg.ELEVENLABS_VOICE_ID
                }
                logger.info("✅ TTS: ElevenLabs ready")
            elif provider == 'kokoro' and hasattr(self, '_init_kokoro'):
                self._init_kokoro()
            elif provider in ('edge', 'edge_tts') and hasattr(self, '_init_edge_tts'):
                self._init_edge_tts()
            elif provider == 'pyttsx3':
                self._init_pyttsx3()
            else:
                logger.warning(f"TTS provider '{provider}' unavailable — falling back to pyttsx3")
                self._init_pyttsx3()
        except Exception as e:
            logger.error(f"TTS init failed ({provider}): {e} — trying pyttsx3 fallback")
            try:
                self._init_pyttsx3()
            except Exception as e2:
                logger.error(f"pyttsx3 fallback also failed: {e2}")

    def _apply_fluid_patches(self):
        """Apply Phase 9 streaming/barge-in patches if available."""
        if _patch_fluid_voice:
            try:
                _patch_fluid_voice(self)
            except Exception as _e:
                logger.debug(f"fluid_voice patch skipped: {_e}")

    def _init_coqui(self):
        """
        Initialise Coqui XTTS-v2 with cached instance to prevent multiple loads.
        """
        importable, err_msg = _check_coqui_installed()
        if not importable:
            logger.error(f"❌ Coqui TTS unavailable — {err_msg}")
            raise RuntimeError(err_msg)

        try:
            import torch
            # Apply PyTorch 2.6+ safe_globals patch BEFORE loading the model.
            _patch_torch_safe_globals()

            # Get cached Coqui instance (this is the key fix)
            tts = _get_coqui_instance()
            
            # Get device from the loaded model
            device = "cuda" if hasattr(tts, 'device') and 'cuda' in str(tts.device) else "cpu"

            voice_ref = self._cfg.COQUI_VOICE_REFERENCE
            if voice_ref:
                voice_ref = str(Path(voice_ref))

            if voice_ref and Path(voice_ref).exists():
                logger.info(f"✅ Coqui voice clone sample loaded: {voice_ref}")
            else:
                if voice_ref:
                    logger.warning(
                        f"Voice reference not found at: {voice_ref!r} — "
                        "using Coqui default speaker."
                    )
                else:
                    logger.info(
                        "No Coqui voice reference configured. "
                        "Record a sample in Settings → Voice Lab to enable cloning."
                    )
                voice_ref = None

            # When no speaker_wav is available XTTS-v2 requires a named speaker.
            # Pick the first available built-in speaker so the model never
            # receives an ambiguous call without either speaker_wav or speaker.
            default_speaker = None
            if voice_ref is None:
                try:
                    speakers = tts.speakers  # list of built-in speaker names
                    if speakers:
                        default_speaker = speakers[0]
                        logger.info(f"XTTS default speaker: {default_speaker!r}")
                except Exception:
                    pass

            self.tts_engine = {
                'type':            'coqui',
                'model':           tts,
                'device':          device,
                'voice_ref':       voice_ref,
                'language':        self._cfg.VOICE_LANGUAGE,
                'default_speaker': default_speaker,
            }
            
            # Warm up the model in a background thread to avoid blocking
            if not hasattr(tts, '_warmed_up'):
                threading.Thread(
                    target=_warmup_coqui,
                    args=(tts, self._cfg.VOICE_LANGUAGE),
                    daemon=True
                ).start()
                tts._warmed_up = True

            clone_info = f"YES — {Path(voice_ref).name}" if voice_ref else "NO (default speaker)"
            logger.info(f"✅ TTS: Coqui XTTS-v2 ready | device={device} | clone={clone_info}")

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Coqui XTTS-v2 load error: {e}", exc_info=True)
            raise RuntimeError(f"Coqui model load failed: {e}")

    def teardown(self):
        """Explicitly release native resources before creating a new AudioManager.
        Critical when switching from Coqui (PyTorch/libgomp) to pyttsx3 (espeak/SAPI):
        without this, GC races between the two native allocators corrupt the heap.
        """
        if self.tts_engine and self.tts_engine.get('type') == 'coqui':
            _release_coqui_model()
        self.tts_engine = None
        self.stt_engine = None

    def _init_pyttsx3(self):
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices') or []
        engine.stop()

        preferred = None
        for v in voices:
            if any(x in v.name.lower() for x in ['zira', 'female', 'hazel', 'susan']):
                preferred = v.id
                break

        self.tts_engine = {
            'type':     'pyttsx3',
            'voice_id': preferred,
            'rate':     170,
            'volume':   1.0
        }
        logger.info(f"✅ TTS: pyttsx3 ready | voice={preferred}")

    def _init_stt(self):
        provider = self._cfg.STT_PROVIDER
        try:
            if provider == 'whisper':
                import whisper
                model_size = getattr(self._cfg, 'WHISPER_MODEL', 'base')
                # Force CPU to avoid GPU memory conflict with Coqui/XTTS-v2.
                # Whisper on CPU is fast enough for STT and prevents cuFFT crashes.
                self.stt_engine = {
                    'type':  'whisper',
                    'model': whisper.load_model(model_size, device='cpu')
                }
                logger.info(f"✅ STT: Whisper ({model_size}) ready on CPU")
            elif provider == 'faster_whisper' and hasattr(self, '_init_faster_whisper'):
                self._init_faster_whisper()
            elif provider == 'openai' and self._cfg.OPENAI_API_KEY:
                from openai import OpenAI
                self.stt_engine = {
                    'type':   'openai',
                    'client': OpenAI(api_key=self._cfg.OPENAI_API_KEY)
                }
                logger.info("✅ STT: OpenAI Whisper API ready")
            else:
                logger.warning(f"STT provider '{provider}' unavailable")
        except Exception as e:
            logger.error(f"STT init failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Text cleaning
    # ─────────────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        text = re.sub(r'```[\s\S]*?```', ' code block omitted. ', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'#{1,6}\s', '', text)
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        text = re.sub(r'https?://\S+', 'link', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_sentences(self, text: str, max_chars: int = 200) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 <= max_chars:
                current += (" " if current else "") + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        if current:
            chunks.append(current.strip())
        return chunks or [text]

    # ─────────────────────────────────────────────────────────────────
    #  TTS — public entry point
    # ─────────────────────────────────────────────────────────────────

    def text_to_speech(self, text: str) -> Optional[str]:
        if not self.tts_engine:
            logger.warning("TTS not initialised")
            return None

        text = self.clean_text(text)
        if not text:
            return None

        logger.info(f"TTS → {text[:80]}{'...' if len(text) > 80 else ''}")

        with self._tts_lock:
            engine_type = self.tts_engine['type']
            try:
                if engine_type == 'coqui':
                    return self._tts_coqui(text)
                elif engine_type == 'openai':
                    return self._tts_openai(text)
                elif engine_type == 'elevenlabs':
                    return self._tts_elevenlabs(text)
                elif engine_type == 'pyttsx3':
                    return self._tts_pyttsx3(text)
            except Exception as e:
                logger.error(f"TTS error ({engine_type}): {e}")
        return None

    # ─────────────────────────────────────────────────────────────────
    #  Coqui XTTS-v2 (with timeout and non-blocking generation)
    # ─────────────────────────────────────────────────────────────────

    def _tts_coqui(self, text: str) -> Optional[str]:
        """Generate speech with timeout to prevent blocking"""
        result_queue = queue.Queue()
        
        def _generate():
            try:
                import numpy as np
                import soundfile as sf

                tts       = self.tts_engine['model']
                voice_ref = self.tts_engine['voice_ref']
                language  = self.tts_engine['language']
                default_speaker = self.tts_engine.get('default_speaker')
                logger.info(
                    "Coqui synthesis voice source: %s",
                    voice_ref if voice_ref else f"built-in speaker {default_speaker!r}",
                )
                output    = os.path.join(
                    tempfile.gettempdir(),
                    f'tts_coqui_{os.getpid()}_{threading.get_ident()}.wav'
                )

                chunks = self._split_sentences(text, max_chars=200)
                logger.info(f"Coqui: {len(chunks)} chunk(s) | clone={'YES' if voice_ref else 'NO'}")

                segments = []
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunk_path = os.path.join(
                        tempfile.gettempdir(),
                        f'tts_chunk_{os.getpid()}_{i}.wav'
                    )
                    if voice_ref and os.path.exists(voice_ref):
                        tts.tts_to_file(
                            text=chunk,
                            speaker_wav=voice_ref,
                            language=language,
                            file_path=chunk_path
                        )
                    else:
                        # No voice clone — must pass a named speaker for multi-speaker XTTS-v2.
                        # default_speaker is populated at init from tts.speakers[0].
                        kwargs = dict(text=chunk, language=language, file_path=chunk_path)
                        if default_speaker:
                            kwargs['speaker'] = default_speaker
                        tts.tts_to_file(**kwargs)
                    if os.path.exists(chunk_path):
                        data, sr = sf.read(chunk_path)
                        segments.append((data, sr))
                        os.remove(chunk_path)

                if not segments:
                    result_queue.put(('error', "No audio segments generated"))
                    return

                sr    = segments[0][1]
                pause = np.zeros(int(sr * 0.15))
                parts = []
                for i, (d, _) in enumerate(segments):
                    parts.append(d)
                    if i < len(segments) - 1:
                        parts.append(pause)

                sf.write(output, np.concatenate(parts), sr)
                size_kb = os.path.getsize(output) // 1024
                logger.info(f"✅ Coqui done: {size_kb} KB")
                result_queue.put(('success', output))
                
            except Exception as e:
                err_str = str(e)
                logger.error(f"Coqui generation error: {e}", exc_info=True)
                # ── GPU/cuFFT failure → retry on CPU ─────────────────────
                if "CUFFT" in err_str or "CUDA" in err_str or "cuda" in err_str.lower():
                    logger.warning("GPU error detected — retrying Coqui on CPU...")
                    try:
                        import torch
                        tts = self.tts_engine['model']
                        if hasattr(tts, 'synthesizer') and tts.synthesizer:
                            synth = tts.synthesizer
                            if hasattr(synth, 'tts_model') and synth.tts_model:
                                synth.tts_model.cpu()
                                synth.tts_model.device = torch.device('cpu')
                            if hasattr(synth, 'vocoder_model') and synth.vocoder_model:
                                synth.vocoder_model.cpu()
                        self.tts_engine['device'] = 'cpu'
                        torch.cuda.empty_cache()
                        chunk_path_cpu = os.path.join(
                            tempfile.gettempdir(),
                            f'tts_coqui_cpu_{os.getpid()}.wav'
                        )
                        voice_ref = self.tts_engine['voice_ref']
                        language  = self.tts_engine['language']
                        if voice_ref and os.path.exists(voice_ref):
                            tts.tts_to_file(text=text, speaker_wav=voice_ref,
                                            language=language, file_path=chunk_path_cpu)
                        else:
                            ds = self.tts_engine.get('default_speaker')
                            kwargs = dict(text=text, language=language, file_path=chunk_path_cpu)
                            if ds:
                                kwargs['speaker'] = ds
                            tts.tts_to_file(**kwargs)
                        if os.path.exists(chunk_path_cpu) and os.path.getsize(chunk_path_cpu) > 1000:
                            logger.info("✅ Coqui CPU fallback succeeded")
                            result_queue.put(('success', chunk_path_cpu))
                            return
                    except Exception as cpu_err:
                        logger.error(f"Coqui CPU fallback also failed: {cpu_err}")
                result_queue.put(('error', err_str))

        # Run generation in thread with timeout
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        thread.join(timeout=self._generation_timeout)
        
        if thread.is_alive():
            logger.error(f"Coqui generation timed out after {self._generation_timeout}s")
            return None
        
        try:
            result_type, result = result_queue.get_nowait()
            if result_type == 'success':
                return result
            else:
                logger.error(f"Coqui generation failed: {result}")
                return None
        except queue.Empty:
            return None

    def update_voice_reference(self, new_path: str) -> bool:
        """Hot-swap voice clone reference without restart."""
        new_path = str(Path(new_path))  # normalise separators
        if self.tts_engine and self.tts_engine['type'] == 'coqui':
            if os.path.exists(new_path):
                self.tts_engine['voice_ref'] = new_path
                logger.info(f"✅ Voice reference updated: {new_path}")
                return True
            else:
                logger.warning(f"Voice reference file not found: {new_path!r}")
        return False

    def update_language(self, language: str) -> bool:
        """Hot-swap the active TTS language without rebuilding the model."""
        if not self.tts_engine:
            return False
        language = (language or 'en').lower()
        engine_type = self.tts_engine.get('type')
        if engine_type == 'coqui':
            self.tts_engine['language'] = language
        elif engine_type == 'pyttsx3':
            # SAPI voices are selected by installed locale, not by a language
            # argument. Resolve it once when the user changes the selector.
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty('voices') or []
                selected = None
                for voice in voices:
                    haystack = f"{voice.id} {voice.name} {getattr(voice, 'languages', '')}".lower()
                    if language in haystack or (language == 'zh-cn' and 'zh' in haystack):
                        selected = voice.id
                        break
                engine.stop()
                if selected:
                    self.tts_engine['voice_id'] = selected
                else:
                    logger.warning("No installed SAPI voice matched language %s; keeping current voice", language)
            except Exception as exc:
                logger.warning("Could not select SAPI voice for %s: %s", language, exc)
        elif engine_type == 'edge_tts':
            edge_voices = {
                'en': 'en-US-AvaNeural', 'fr': 'fr-FR-DeniseNeural',
                'de': 'de-DE-KatjaNeural', 'es': 'es-ES-ElviraNeural',
                'it': 'it-IT-ElsaNeural', 'pt': 'pt-BR-FranciscaNeural',
                'ru': 'ru-RU-SvetlanaNeural', 'uk': 'uk-UA-PolinaNeural',
                'nl': 'nl-NL-ColetteNeural', 'pl': 'pl-PL-ZofiaNeural',
                'ja': 'ja-JP-NanamiNeural', 'ko': 'ko-KR-SunHiNeural',
                'zh-cn': 'zh-CN-XiaoxiaoNeural',
            }
            obj = self.tts_engine.get('_obj')
            if obj:
                obj.voice = edge_voices.get(language, edge_voices['en'])
        elif engine_type == 'kokoro':
            # Kokoro's bundled model is English-only; retain the selected
            # language for the next compatible provider rather than failing.
            logger.info("Kokoro has no native %s voice; leaving its English voice active", language)
        self.tts_engine['language'] = language
        logger.info(f"✅ TTS language updated: {language} ({engine_type})")
        return True

    def get_voice_clone_status(self) -> dict:
        """Return a dict describing the current voice clone state."""
        if not self.tts_engine:
            return {'provider': 'none', 'cloning': False, 'voice_ref': None}
        t  = self.tts_engine['type']
        vr = self.tts_engine.get('voice_ref') if t == 'coqui' else None
        return {
            'provider':  t,
            'cloning':   bool(vr and os.path.exists(vr)),
            'voice_ref': vr,
            'device':    self.tts_engine.get('device', 'cpu'),
        }

    # ─────────────────────────────────────────────────────────────────
    #  pyttsx3 (with Windows SAPI fallback)
    # ─────────────────────────────────────────────────────────────────

    # ── Subprocess helper script (written once per process) ─────────────────
    _PYTTSX3_SCRIPT: Optional[str] = None
    _PYTTSX3_SCRIPT_LOCK = threading.Lock()

    @classmethod
    def _get_pyttsx3_script(cls) -> str:
        """Write the pyttsx3 worker script to a temp file once per process."""
        with cls._PYTTSX3_SCRIPT_LOCK:
            if cls._PYTTSX3_SCRIPT and os.path.exists(cls._PYTTSX3_SCRIPT):
                return cls._PYTTSX3_SCRIPT
            script = """
import sys, pyttsx3
text, output, rate, volume, voice_id = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
engine = pyttsx3.init()
engine.setProperty('rate', rate)
engine.setProperty('volume', volume)
if voice_id != '__none__':
    engine.setProperty('voice', voice_id)
engine.save_to_file(text, output)
engine.runAndWait()
engine.stop()
"""
            path = os.path.join(tempfile.gettempdir(), f'_pyttsx3_worker_{os.getpid()}.py')
            with open(path, 'w') as f:
                f.write(script)
            cls._PYTTSX3_SCRIPT = path
            return path

    def _tts_pyttsx3(self, text: str) -> Optional[str]:
        """
        Run pyttsx3 synthesis in a subprocess to fully isolate it from any
        PyTorch/Coqui native libraries (libgomp, libcudart) still resident in
        the main process.  Direct thread-based pyttsx3.init() after Coqui causes
        heap corruption ('corrupted double-linked list') on Linux.
        """
        output = os.path.join(
            tempfile.gettempdir(),
            f'tts_pyttsx3_{os.getpid()}_{threading.get_ident()}.wav'
        )
        rate     = str(self.tts_engine.get('rate', 170))
        volume   = str(self.tts_engine.get('volume', 1.0))
        voice_id = self.tts_engine.get('voice_id') or '__none__'

        def _run_in_process() -> Optional[str]:
            err = [None]

            def _run():
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate',   int(rate))
                    engine.setProperty('volume', float(volume))
                    if voice_id != '__none__':
                        engine.setProperty('voice', voice_id)
                    engine.save_to_file(text, output)
                    engine.runAndWait()
                    engine.stop()
                except Exception as ex:
                    err[0] = ex

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=20)
            if not err[0] and os.path.exists(output) and os.path.getsize(output) > 2000:
                logger.info("✅ pyttsx3 (in-process) TTS success")
                return output
            if err[0]:
                logger.error(f"pyttsx3 in-process error: {err[0]}")
            return None

        coqui_resident = _COQUI_LOADED.is_set()

        # On Windows, avoid per-utterance Python process startup unless Coqui is loaded.
        if sys.platform == 'win32' and not coqui_resident:
            in_process_output = _run_in_process()
            if in_process_output:
                return in_process_output

        # Subprocess path stays available for native-library isolation after Coqui.
        try:
            script = self._get_pyttsx3_script()
            result = subprocess.run(
                [sys.executable, script, text, output, rate, volume, voice_id],
                timeout=25, capture_output=True
            )
            if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 2000:
                logger.info("✅ pyttsx3 (subprocess) TTS success")
                return output
            if result.returncode != 0:
                logger.warning(f"pyttsx3 subprocess error: {result.stderr.decode()[:200]}")
        except subprocess.TimeoutExpired:
            logger.error("pyttsx3 subprocess timed out")
        except Exception as e:
            logger.warning(f"pyttsx3 subprocess failed: {e} — trying in-process fallback")

        # Second try: in-process (only if subprocess failed AND Coqui not loaded)
        if not coqui_resident:
            in_process_output = _run_in_process()
            if in_process_output:
                return in_process_output

        # Final fallback
        logger.warning("pyttsx3 all methods failed — using system fallback")
        if sys.platform == 'win32':
            return self._tts_sapi_powershell(text)
        return self._tts_espeak(text)

    def _tts_sapi_powershell(self, text: str) -> Optional[str]:
        output    = os.path.join(tempfile.gettempdir(), f'tts_sapi_{os.getpid()}.wav')
        safe_text = text.replace("'", "''").replace('"', '""')
        from managers.settings_manager import config
        language = getattr(config, 'VOICE_LANGUAGE', 'en').split('-')[0].lower()
        ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name.ToLower().StartsWith('{language}') }} | Select-Object -First 1
if ($voice) {{ $s.SelectVoice($voice.VoiceInfo.Name) }}
$s.SetOutputToWaveFile('{output}')
$s.Rate = 1
$s.Volume = 100
$s.Speak('{safe_text}')
$s.Dispose()
"""
        try:
            r = subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 2000:
                logger.info("✅ SAPI PowerShell TTS success")
                return output
            logger.error(f"SAPI error: {r.stderr.strip()}")
        except Exception as e:
            logger.error(f"SAPI exception: {e}")
        return None

    def _tts_espeak(self, text: str) -> Optional[str]:
        output = os.path.join(tempfile.gettempdir(), f'tts_espeak_{os.getpid()}.wav')
        try:
            subprocess.run(
                ['espeak', '-w', output, '-s', '150', text],
                capture_output=True, timeout=15
            )
            if os.path.exists(output) and os.path.getsize(output) > 1000:
                return output
        except Exception as e:
            logger.error(f"espeak error: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────
    #  OpenAI TTS
    # ─────────────────────────────────────────────────────────────────

    def _tts_openai(self, text: str) -> Optional[str]:
        output = os.path.join(tempfile.gettempdir(), f'tts_openai_{os.getpid()}.mp3')
        try:
            self.tts_engine['client'].audio.speech.create(
                model="tts-1", voice="nova", input=text
            ).stream_to_file(output)
            if os.path.exists(output) and os.path.getsize(output) > 1000:
                return output
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────
    #  ElevenLabs TTS
    # ─────────────────────────────────────────────────────────────────

    def _tts_elevenlabs(self, text: str) -> Optional[str]:
        output = os.path.join(tempfile.gettempdir(), f'tts_el_{os.getpid()}.wav')
        try:
            import requests as req
            r = req.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.tts_engine['voice_id']}",
                headers={
                    "xi-api-key":   self.tts_engine['api_key'],
                    "Content-Type": "application/json"
                },
                json={
                    "text":     text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
                },
                timeout=30
            )
            if r.status_code == 200:
                Path(output).write_bytes(r.content)
                return output
            logger.error(f"ElevenLabs error: {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────
    #  STT
    # ─────────────────────────────────────────────────────────────────

    def speech_to_text(self, audio_path: str) -> Optional[str]:
        if not self.stt_engine:
            logger.warning("STT not available")
            return None
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        # Known Whisper hallucinations on silence/noise — discard these entirely.
        # Whisper is trained on YouTube data and hallucinates typical video phrases
        # when given quiet or near-silence audio at startup or between sentences.
        _HALLUCINATIONS = {
            "thanks for watching", "thank you for watching",
            "thanks for listening", "thank you for listening",
            "please subscribe", "like and subscribe",
            "don't forget to subscribe", "see you in the next video",
            "see you next time", "bye", "goodbye", "bye bye",
            "[ music ]", "[music]", "[ applause ]", "[applause]",
            "music", "applause", "music playing",
            "subtitles by", "transcribed by",
            "you", ".", "..", "...", "- ", "—",
        }

        try:
            t = self.stt_engine['type']
            if t == 'whisper':
                from managers.settings_manager import config
                lang = getattr(config, 'VOICE_LANGUAGE', 'en')
                if lang.startswith('zh'):
                    lang = 'zh'
                result = self.stt_engine['model'].transcribe(
                    audio_path,
                    fp16=False,
                    language=lang,
                    # Raise no-speech threshold: if Whisper is not confident
                    # there is actual speech, suppress the output entirely.
                    # Default is 0.6 — raising to 0.8 prevents most hallucinations.
                    no_speech_threshold=0.80,
                    # Suppress blank audio token (token 220 in multilingual models)
                    # which is associated with silence hallucinations.
                    suppress_tokens=[-1],
                    # condition_on_previous_text=False prevents the model from
                    # repeating its own previous output (another source of hallucinations).
                    condition_on_previous_text=False,
                )
                text = result.get('text', '').strip()

                # Filter known hallucinations (case-insensitive, exact or prefix match)
                if text:
                    text_lower = text.lower().strip(' .,!?')
                    if text_lower in _HALLUCINATIONS:
                        logger.debug(f"STT hallucination suppressed: {text!r}")
                        return None
                    # Also suppress very short single-word outputs that are
                    # almost always noise artifacts (less than 3 chars)
                    if len(text_lower) < 3:
                        logger.debug(f"STT output too short, suppressed: {text!r}")
                        return None

                return text or None

            elif t == 'openai':
                with open(audio_path, 'rb') as f:
                    tr = self.stt_engine['client'].audio.transcriptions.create(
                        model="whisper-1", file=f
                    )
                text = tr.text.strip()

                # Apply same hallucination filter for OpenAI Whisper API
                if text:
                    text_lower = text.lower().strip(' .,!?')
                    if text_lower in _HALLUCINATIONS or len(text_lower) < 3:
                        logger.debug(f"STT hallucination suppressed (OpenAI): {text!r}")
                        return None

                return text or None

        except Exception as e:
            logger.error(f"STT error: {e}")
        return None


def create_audio_manager() -> AudioManager:
    return AudioManager()
