"""
Conversational Audio Manager
- Always-listen VAD loop (WebRTC VAD + energy gate)
- Noise-resistant: onset confirmation, min speech duration, adaptive energy gate
- Fires on_transcription via queue — NEVER touches NiceGUI UI directly
- Interrupts TTS playback immediately when speech is detected
"""
import logging
import threading
import queue
import time
import tempfile
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ConversationalAudioManager:
    def __init__(self, audio_manager, tts_stop_event: threading.Event):
        self.audio_manager  = audio_manager
        self.tts_stop_event = tts_stop_event

        self.is_listening  = False
        self._stop_event   = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue()

        # ── VAD core ──────────────────────────────────────────────────
        self.vad_available = False
        self._vad          = None
        self.sample_rate   = 16000
        self.chunk_size    = 480          # 30 ms at 16 kHz (webrtcvad requirement)

        # ── Load settings from config ─────────────────────────────────
        try:
            from managers.settings_manager import config as _cfg
            self.vad_aggressiveness  = _cfg.VAD_AGGRESSIVENESS
            self.speech_onset_chunks = _cfg.VAD_ONSET_CHUNKS
            self.silence_duration    = _cfg.VAD_SILENCE_DURATION
            self.min_speech_duration = _cfg.VAD_MIN_SPEECH_DURATION
            self.energy_gate_factor  = _cfg.VAD_ENERGY_GATE_FACTOR
            self.tts_post_roll_ms    = getattr(_cfg, 'TTS_POST_ROLL_MS', 250)
        except Exception:
            self.vad_aggressiveness  = 3
            self.speech_onset_chunks = 4
            self.silence_duration    = 1.2
            self.min_speech_duration = 0.4
            self.energy_gate_factor  = 3.5
            self.tts_post_roll_ms    = 250

        self.energy_threshold    = 0.02

        # ── Adaptive noise floor ──────────────────────────────────────
        self._noise_floor        = None   # calibrated on startup
        self._noise_calibrated   = False

        # ── Callbacks ─────────────────────────────────────────────────
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_speech_start:  Optional[Callable[[], None]]    = None
        self.on_speech_end:    Optional[Callable[[], None]]    = None
        
        # ── Transcription state ───────────────────────────────────────
        self._transcribing = False           # Flag to prevent overlapping transcriptions
        self._last_transcription_time = 0.0  # For cooldown
        self._last_transcribed_text   = ""   # For text-level dedup (mid-sentence pause splits)
        self._dedup_window            = 3.0  # Seconds — suppress identical text within this window

        # ── TTS mute gate ─────────────────────────────────────────────
        # Set True by the TTS player before playback starts, False after.
        # While True: the VAD loop drains the audio queue without processing,
        # so the bot's own speaker audio never triggers speech detection.
        # After cleared: queue is flushed to discard any remaining stale audio.
        self._tts_active = False

        # ── Manual mute gate (user-controlled VAD on/off button) ──────
        # Separate from _tts_active so toggling it does NOT trigger the
        # post-TTS flush/grace period that was causing 3-click syndrome.
        # When True: chunks are drained without processing (same as _tts_active)
        # but clearing it does NOT cause a queue flush.
        self._muted = False

        # ── Dedup lock ────────────────────────────────────────────────
        # _last_transcription_time and _last_transcribed_text are read/written
        # from both the VAD process thread and the STT transcription thread.
        # Without a lock the cooldown check is a data race.
        self._dedup_lock = threading.Lock()

        self._init_vad()

    # ─────────────────────────────────────────────────────────────────
    #  VAD init
    # ─────────────────────────────────────────────────────────────────

    def _init_vad(self):
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.vad_aggressiveness)
            self.vad_available = True
            logger.info(f"✅ WebRTC VAD initialised (aggressiveness={self.vad_aggressiveness})")
        except ImportError:
            logger.warning(
                "webrtcvad not installed — using energy VAD fallback. "
                "Install: pip install webrtcvad"
            )
            self._vad = None  # energy VAD fallback will be used
            self.vad_available = True  # Energy fallback is still available
        except Exception as e:
            logger.error(f"VAD init error: {e}")
            self.vad_available = False

    def set_aggressiveness(self, level: int):
        """Hot-change VAD aggressiveness (0–3). 3 = most noise resistant."""
        level = max(0, min(3, int(level)))
        self.vad_aggressiveness = level
        if self._vad is not None:
            try:
                import webrtcvad
                self._vad = webrtcvad.Vad(level)
                logger.info(f"VAD aggressiveness updated to {level}")
            except Exception as e:
                logger.error(f"VAD aggressiveness update failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Noise floor calibration
    # ─────────────────────────────────────────────────────────────────

    def _calibrate_noise_floor(self, calibration_seconds: float = 1.5):
        """
        Sample ambient noise for calibration_seconds and set the noise floor.
        Called once when always-listen starts. The mic must be open already.
        We collect chunks from the queue (which is already filling from the
        listen thread) until we have enough samples.
        """
        logger.info(f"🔇 Calibrating noise floor ({calibration_seconds}s)…")
        chunks_needed = int(calibration_seconds * self.sample_rate / self.chunk_size)
        samples = []
        deadline = time.time() + calibration_seconds + 1.0  # 1s grace

        while len(samples) < chunks_needed and time.time() < deadline:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                samples.append(chunk)
            except queue.Empty:
                continue

        if samples:
            energies = [float(np.sqrt(np.mean(c ** 2))) for c in samples]
            # Use 90th percentile as noise floor (robust against occasional spikes)
            self._noise_floor = float(np.percentile(energies, 90))
            self._noise_calibrated = True
            logger.info(
                f"✅ Noise floor calibrated: {self._noise_floor:.5f} "
                f"(speech gate at {self._noise_floor * self.energy_gate_factor:.5f})"
            )
        else:
            self._noise_floor = self.energy_threshold
            logger.warning("Noise floor calibration got no samples — using default threshold")

    def _energy_gate(self, chunk: np.ndarray) -> bool:
        """
        Returns True only if chunk energy is significantly above the noise floor.
        Acts as a pre-filter before WebRTC VAD to block constant low-level noise.
        """
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        if self._noise_calibrated and self._noise_floor is not None:
            return energy > (self._noise_floor * self.energy_gate_factor)
        return energy > self.energy_threshold

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """
        Two-stage check: energy gate first, then WebRTC VAD.
        Both must agree for a chunk to be classified as speech.
        This dramatically reduces false positives from ambient noise.
        """
        # Stage 1: energy gate — fast, blocks most background noise
        if not self._energy_gate(chunk):
            return False

        # Stage 2: WebRTC VAD — phoneme-level speech detection
        if self._vad is not None:
            try:
                pcm = (chunk * 32767).astype(np.int16).tobytes()
                return self._vad.is_speech(pcm, self.sample_rate)
            except Exception:
                pass

        # Fallback: energy gate alone was sufficient
        return True

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def start_conversation(
        self,
        on_transcription: Callable[[str], None],
        on_speech_start:  Callable[[], None],
        on_speech_end:    Callable[[], None]
    ):
        if self.is_listening:
            logger.warning("Already listening — ignoring start_conversation")
            return

        self.on_transcription = on_transcription
        self.on_speech_start  = on_speech_start
        self.on_speech_end    = on_speech_end

        self.is_listening = True
        self._stop_event.clear()

        threading.Thread(target=self._listen_loop,  daemon=True, name="vad-listen").start()
        threading.Thread(target=self._process_loop, daemon=True, name="vad-process").start()
        logger.info("🎤 Always-listen started")

    def stop_conversation(self):
        self.is_listening = False
        self._stop_event.set()
        logger.info("🛑 Always-listen stopped")

    def recalibrate(self):
        """Force a new noise floor calibration. Call after moving to a noisier/quieter room."""
        self._noise_calibrated = False
        self._noise_floor      = None
        logger.info("Noise floor reset — will recalibrate on next speech activity")

    # ─────────────────────────────────────────────────────────────────
    #  Internal threads
    # ─────────────────────────────────────────────────────────────────

    def _listen_loop(self):
        """Reads mic continuously into audio queue. Never touches UI."""
        try:
            import sounddevice as sd

            def _cb(indata, frames, ts, status):
                if status:
                    logger.debug(f"Mic status: {status}")
                if not self._stop_event.is_set():
                    self._audio_queue.put(indata.copy().flatten())

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=_cb
            ):
                # Calibrate noise floor before entering main loop
                self._calibrate_noise_floor(calibration_seconds=1.5)

                while not self._stop_event.is_set():
                    time.sleep(0.05)

        except Exception as e:
            logger.error(f"Microphone error: {e}")
            self.is_listening = False

    def _process_loop(self):
        """
        VAD loop with:
        - onset confirmation (N consecutive speech chunks before triggering)
        - minimum speech duration gate (filters short noise bursts)
        - adaptive silence detection
        - cooldown period to prevent duplicate transcriptions
        """
        speech_buffer:    list = []
        silence_chunks        = 0
        onset_counter         = 0    # consecutive speech chunks seen
        in_speech             = False

        silence_limit = int(self.silence_duration * self.sample_rate / self.chunk_size)
        onset_needed  = self.speech_onset_chunks
        cooldown_period = 0.8  # seconds - minimum gap between transcriptions
        _was_tts_active = False  # tracks edge: TTS just finished

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Manual mute gate (user button) ────────────────────────
            # Silently drain — no post-flush, no edge detection.
            if self._muted:
                speech_buffer  = []
                silence_chunks = 0
                onset_counter  = 0
                in_speech      = False
                continue

            # ── TTS mute gate ─────────────────────────────────────────
            # While the bot is speaking, silently drain every chunk so the
            # bot's own voice never reaches the VAD or starts a cooldown.
            if self._tts_active:
                _was_tts_active = True
                # Reset all in-progress speech state so we start clean
                speech_buffer  = []
                silence_chunks = 0
                onset_counter  = 0
                in_speech      = False
                continue  # discard chunk — do NOT call _is_speech()

            # TTS just finished — flush residual speaker audio and wait
            # a grace period before resuming VAD detection.
            # Without this, the tail end of the bot's own voice triggers speech.
            if _was_tts_active:
                _was_tts_active = False
                # Drain everything currently in the queue (bot-voice residue)
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                # Extra grace: discard a short tail of audio to let
                # speaker reverb and room echo die down before VAD resumes
                post_roll_s = max(0.0, float(self.tts_post_roll_ms) / 1000.0)
                grace_chunks = int(post_roll_s * self.sample_rate / self.chunk_size)
                for _ in range(grace_chunks):
                    try:
                        self._audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        break
                logger.debug("🔇 Audio queue flushed after TTS — VAD resuming")
                continue
            # ─────────────────────────────────────────────────────────

            is_speech = self._is_speech(chunk)

            if is_speech:
                # Check if we're in cooldown period after a recent transcription
                with self._dedup_lock:
                    time_since_last = time.time() - self._last_transcription_time
                if time_since_last < cooldown_period:
                    # Still in cooldown - ignore this speech to prevent duplicates
                    continue
                
                onset_counter += 1

                if not in_speech:
                    # Accumulate pre-speech context but wait for onset confirmation
                    speech_buffer.append(chunk)

                    if onset_counter >= onset_needed:
                        # Confirmed — genuine speech onset
                        in_speech      = True
                        silence_chunks = 0
                        self.tts_stop_event.set()
                        if self.on_speech_start:
                            try:
                                self.on_speech_start()
                            except Exception as e:
                                logger.error(f"on_speech_start error: {e}")
                        logger.debug("🗣 Speech onset confirmed")
                else:
                    # Already in speech
                    silence_chunks = 0
                    speech_buffer.append(chunk)

            else:
                # Not speech
                onset_counter = max(0, onset_counter - 1)  # decay counter

                if in_speech:
                    silence_chunks += 1
                    speech_buffer.append(chunk)   # keep trailing silence natural

                    if silence_chunks >= silence_limit:
                        in_speech     = False
                        onset_counter = 0

                        if self.on_speech_end:
                            try:
                                self.on_speech_end()
                            except Exception as e:
                                logger.error(f"on_speech_end error: {e}")

                        # Minimum duration gate — don't send tiny noise bursts to Whisper
                        speech_duration = len(speech_buffer) * self.chunk_size / self.sample_rate
                        if speech_duration >= self.min_speech_duration and speech_buffer:
                            # Check if we're already transcribing
                            if self._transcribing:
                                logger.debug("Skipping transcription - already in progress")
                                speech_buffer  = []
                                silence_chunks = 0
                            else:
                                buf_copy      = speech_buffer[:]
                                speech_buffer = []
                                silence_chunks = 0
                                with self._dedup_lock:
                                    self._last_transcription_time = time.time()
                                threading.Thread(
                                    target=self._transcribe,
                                    args=(buf_copy,),
                                    daemon=True,
                                    name="transcribe"
                                ).start()
                        else:
                            logger.debug(
                                f"Speech too short ({speech_duration:.2f}s < "
                                f"{self.min_speech_duration}s) — discarded"
                            )
                            speech_buffer  = []
                            silence_chunks = 0
                else:
                    # Not in speech — slowly drain pre-onset buffer
                    if speech_buffer:
                        speech_buffer.pop(0)

    def _transcribe(self, buffer: list):
        """
        STT in background thread.
        CRITICAL: on_transcription must only do queue.put() — never touch NiceGUI UI.
        """
        self._transcribing = True
        # Signal vision to back off face-tracking (HOG model) during Whisper STT
        # so both can share the CPU without starving each other.
        _vision = getattr(self, '_vision_ref', None)
        if _vision is not None:
            _vision.audio_busy = True
        try:
            import soundfile as sf
            import os

            audio_data = np.concatenate(buffer)
            tmp_path   = tempfile.mktemp(suffix='.wav')
            sf.write(tmp_path, audio_data, self.sample_rate)

            text = self.audio_manager.speech_to_text(tmp_path)

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            if text and text.strip() and self.on_transcription:
                logger.info(f"Transcribed: {text!r}")

                # ── Text-level deduplication (thread-safe) ────────────────
                now = time.time()
                normalized = text.strip().lower()
                with self._dedup_lock:
                    is_dup = (
                        normalized == self._last_transcribed_text.lower()
                        and (now - self._last_transcription_time) < self._dedup_window
                    )
                    if not is_dup:
                        self._last_transcribed_text   = text.strip()
                        self._last_transcription_time = now

                if is_dup:
                    logger.info(f"Duplicate transcription suppressed (within {self._dedup_window}s window)")
                else:
                    self.on_transcription(text.strip())

        except Exception as e:
            logger.error(f"Transcription error: {e}")
        finally:
            self._transcribing = False
            if _vision is not None:
                _vision.audio_busy = False
