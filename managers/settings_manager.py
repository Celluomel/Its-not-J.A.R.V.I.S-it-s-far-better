"""settings_manager.py - Complete with all features (NO app.py changes needed)"""
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable
from pydantic import BaseModel, validator

# Configuration file path
CONFIG_FILE = Path('config.json')
BACKUP_DIR = Path('config_backups')

class AppSettings(BaseModel):
    """Application settings with defaults and validation"""
    # LLM Settings
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "llama2"
    # Separate fast text-only model used when vision is NOT needed.
    # Set to "" to always use LLM_MODEL for everything.
    # Example: LLM_MODEL = "llava:latest", TEXT_MODEL = "llama3.2:3b"
    TEXT_MODEL: str = ""
    # Local multilingual sentence encoder for background affect estimation.
    AFFECT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    RESEARCH_SEARCH_BACKEND: str = "auto"   # "auto"|"claude"|"brave"|"rss"|"stealth"
    BRAVE_SEARCH_KEY:        str = ""
    SERPAPI_KEY:             str = ""
    
    # Persona Identity
    PERSONA_NAME: str = "Lumina"   # The name of the AI persona — shown in UI and used in prompts
    LUMINA_BIRTH_DATE: str = ""    # ISO datetime, set once on first boot. See get_birth_date() /
                                    # ai_system.py's age-as-computed-property model (v117): age and
                                    # life_stage are now derived purely from elapsed real time since
                                    # this date, not from usage/idle-triggered increments.
    # Physical location is an explicit, user-editable world-model anchor.
    # Coordinates are authoritative when present; address is descriptive.
    LOCATION_NAME: str = ""
    LOCATION_ADDRESS: str = ""
    LOCATION_LATITUDE: Optional[float] = None
    LOCATION_LONGITUDE: Optional[float] = None

    # Memory Settings
    MEMORY_BACKEND:      str  = "simple"
    MEMORY_DB_PATH:      str  = "data/persona/ai_system.db"
    MEMORY_FAISS_PATH:   str  = "data/persona/faiss_index.bin"
    MEMORY_WORLD_PATH:   str  = "data/persona/world_model.json"
    MEMORY_PERSONA_PATH: str  = "data/persona"
    MEMORY_COGNEE_PATH:  str  = "data/persona/cognee"  # Cognee knowledge-graph storage root
    MEMORY_COGNEE_EMBED_MODEL: str = ""  # Cognee embedding model

    # Optional VoiceMem secondary voice-memory index.  Native memory and the
    # configured STT/TTS pipeline remain canonical; this is opt-in and local.
    VOICEMEM_ENABLED: bool = False
    VOICEMEM_DATA_PATH: str = "data/persona/voicemem"
    VOICEMEM_TOP_K: int = 5
    VOICEMEM_LOCAL_MODE: bool = True

    # Two-tier embedding stack (see utils/shared_embedder.py).
    # QUALITY tier: high-fidelity embeddings via an OpenAI-compatible
    # /v1/embeddings endpoint (LM Studio). Used for batch / semantic-critical
    # work (dedup, pruning, relevance ranking). FAST tier (local MiniLM) is
    # the default for hot paths and needs no config here.
    QUALITY_EMBED_MODEL: str = "text-embedding-nomic-embed-text-v1.5"
    EMBED_API_BASE_URL:  str = "http://localhost:1234/v1"

    # ── Phase 9: Fluid Voice ───────────────────────────────────────────────
    KOKORO_VOICE:            str  = "af_heart"        # Kokoro voice ID
    KOKORO_SPEED:            str  = "1.0"             # 0.5–2.0
    EDGE_TTS_VOICE:          str  = "en-US-AvaNeural" # Edge-TTS neural voice
    EDGE_TTS_RATE:           str  = "+0%"             # speaking rate adjustment
    TTS_STREAMING:           bool = True              # sentence-level streaming
    TTS_BARGE_IN:            bool = True              # allow user to interrupt
    BARGE_IN_SENSITIVITY:    int  = 1                 # VAD aggressiveness 0-3
    INTER_SENTENCE_PAUSE_MS: int  = 120               # ms pause between sentences (LM Studio: 'nomic-embed-text', Ollama: auto)
    TTS_POST_ROLL_MS:        int  = 250               # discard speaker echo after TTS before VAD resumes
    
    # Audio Settings
    TTS_PROVIDER: str = "pyttsx3"
    STT_PROVIDER: str = "whisper"
    WHISPER_MODEL: str = "base"
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "EXAVITQu4vr4xnSDxMaL"
    VOICE_LANGUAGE: str = "en"
    COQUI_VOICE_REFERENCE: str = ""
    RESPONSE_LANGUAGE: str = "auto"   # "auto" = follow user, else force language e.g. "fr"
    
    # VAD Settings
    VAD_AGGRESSIVENESS: int = 3
    VAD_ONSET_CHUNKS: int = 4
    VAD_SILENCE_DURATION: float = 1.2
    VAD_MIN_SPEECH_DURATION: float = 0.6   # raised from 0.4s — short bursts hallucinate
    VAD_ENERGY_GATE_FACTOR: float = 3.5
    
    # Vision Settings
    CAMERA_AUTOSTART: bool = False
    CAMERA_ID: int = 0
    CAMERA_FPS: int = 5
    CAMERA_RESOLUTION: str = "640x480"
    LAVA_MODEL: str = "llava:latest"
    LLM_CTX: int = 4096  # context window tokens — must match LM Studio/Ollama
    WEB_SEARCH_MODE: str = "off"  # "off" | "auto" | "always"
    VISION_MODE: str = "keyword"  # keyword, always, context

    # Vision LLM routing:
    #   "separate" → analyze via dedicated LLaVA model (LAVA_MODEL), text result
    #                fed to main LLM. Works even if main model is text-only.
    #   "direct"   → send image directly to main model (must be multimodal:
    #                LLaVA, Qwen-VL, Pixtral, etc. loaded in LM Studio)
    VISION_LLM_MODE: str = "separate"

    # Ambient vision: how often (seconds) the orchestrator silently analyzes
    # a frame and feeds the description to the GlobalWorkspace.
    # 0 = disabled. Recommended: 60–180s. Never runs during active interaction.
    AMBIENT_VISION_INTERVAL: int = 90
    
    # ── Messaging connectors ───────────────────────────────────────────────
    # Telegram
    TELEGRAM_TOKEN: str = ""           # BotFather token  e.g. 123456:ABC-DEF...

    # WhatsApp via Twilio
    WHATSAPP_TWILIO_SID:     str = ""  # Twilio Account SID
    WHATSAPP_TWILIO_TOKEN:   str = ""  # Twilio Auth Token
    WHATSAPP_FROM:           str = "whatsapp:+14155238886"  # Twilio sandbox number
    WHATSAPP_WEBHOOK_SECRET: str = ""  # optional: validate Twilio X-Twilio-Signature

    # Lumina-to-Lumina network children
    # List of child Lumina instances this master can connect to
    # Format: [{"id":"child_a","name":"Lumina-A","url":"http://IP:8765","api_key":"","role":"child"}]
    LUMINA_CHILDREN: list = []
    LUMINA_NETWORK_ENABLED: bool = False   # set True to activate the connector
    LUMINA_MASTER_NAME: str = "Lumina-Master"
    # Phase 6.4 — real cognitive differentiation for a peer instance (Flux).
    # "collaborative" (default): no change to prompt behavior.
    # "epistemic_challenger": this instance's own system prompt is biased
    # toward alternative interpretation, contradiction, counterfactual,
    # and epistemic challenge — set this on the CHILD/Flux instance's own
    # config.json, not the master's. See ai_system.py's system prompt
    # builder for where this is read.
    PEER_COGNITIVE_STANCE: str = "collaborative"

    # Headless brain API (brain.py)
    BRAIN_API_PORT: int = 8765         # REST + webhook listen port
    BRAIN_API_HOST: str = "127.0.0.1"  # change to 0.0.0.0 to expose externally

    # NiceGUI
    NICEGUI_PORT: int = 8080
    NICEGUI_HOST: str = "127.0.0.1"

    # Persona system (Lumina psychological subsystems)
    PERSONA_ENABLED: bool = True

    # Response verbosity is a prompt-level style instruction. These values are
    # provider safety floors/ceilings, never a word limit for the user answer.
    RESPONSE_VERBOSITY: str = "concise"
    RESPONSE_TOKENS_CONCISE: int = 2048  # inference safety budget
    RESPONSE_TOKENS_VERBOSE: int = 4096  # inference safety budget

    # Custom system prompt suffix — appended after the built-in HOW TO RESPOND block.
    # Empty string = use built-in prompt only (recommended default).
    CUSTOM_SYSTEM_PROMPT: str = ""

    # Debug: log the FULL assembled system prompt (cognitive state + emotional
    # state + identity + relational context) on every response, so what
    # Lumina/the LLM actually sees can be inspected directly. Off by default
    # since prompts run ~900+ tokens and this fires every turn. When on,
    # logs at INFO (visible by default) and also writes the latest prompt
    # to data/persona/last_prompt_debug.txt (overwritten each turn — easy
    # to tail/open without scrolling logs).
    LOG_FULL_PROMPTS: bool = False

    # Cognitive Observatory settings
    OBSERVATORY_ENABLED:            bool  = True
    OBSERVATORY_BASELINE_SAMPLES:   int   = 20
    OBSERVATORY_EMERGENCE_THRESHOLD: float = 0.50
    
    # ----- VALIDATION -----
    @validator('NICEGUI_PORT')
    def validate_port(cls, v):
        if not 1024 <= v <= 65535:
            return 8080  # Auto-correct instead of raising error
        return v

    @validator('NICEGUI_HOST')
    def validate_nicegui_host(cls, v):
        return v if v in {'127.0.0.1', '0.0.0.0', '::1'} else '127.0.0.1'
    
    @validator('CAMERA_FPS')
    def validate_fps(cls, v):
        valid_fps = [5, 10, 15, 30]
        if v not in valid_fps:
            return 5  # Default to 5 if invalid
        return v
    
    @validator('VAD_AGGRESSIVENESS')
    def validate_vad(cls, v):
        if v not in [0, 1, 2, 3]:
            return 3  # Default to balanced
        return v
    
    @validator('WHISPER_MODEL')
    def validate_whisper(cls, v):
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if v not in valid_models:
            return 'base'
        return v
    
    @validator('VISION_MODE')
    def validate_vision_mode(cls, v):
        valid_modes = ['keyword', 'always', 'context']
        if v not in valid_modes:
            return 'keyword'  # Default to keyword-triggered
        return v
    
    class Config:
        arbitrary_types_allowed = True

# ----- BACKUP FUNCTIONALITY -----
def create_backup():
    """Create a backup of current config if it exists"""
    if not CONFIG_FILE.exists():
        return None
    
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f'config_{timestamp}.json'
    
    shutil.copy(CONFIG_FILE, backup_path)
    
    # Clean old backups (keep last 10)
    backups = sorted(BACKUP_DIR.glob('config_*.json'))
    for old in backups[:-10]:
        old.unlink()
    
    return backup_path

# ----- ENVIRONMENT OVERRIDE -----
def apply_env_overrides(settings: AppSettings) -> AppSettings:
    """Override settings from environment variables if they exist"""
    env_map = {
        'LLM_PROVIDER': os.getenv('ROBOT_LLM_PROVIDER'),
        'LLM_MODEL': os.getenv('ROBOT_LLM_MODEL'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'TTS_PROVIDER': os.getenv('ROBOT_TTS_PROVIDER'),
        'NICEGUI_PORT': os.getenv('ROBOT_PORT'),
    }
    
    modified = False
    for key, value in env_map.items():
        if value is not None and hasattr(settings, key):
            # Convert type based on current value
            current = getattr(settings, key)
            try:
                if isinstance(current, bool):
                    setattr(settings, key, value.lower() == 'true')
                elif isinstance(current, int):
                    setattr(settings, key, int(value))
                elif isinstance(current, float):
                    setattr(settings, key, float(value))
                else:
                    setattr(settings, key, value)
                modified = True
                print(f"🌐 Env override: {key}={value}")
            except:
                print(f"⚠️ Failed to override {key} with '{value}'")
    
    return settings, modified

# ----- CORE LOAD/SAVE -----
def load_settings() -> AppSettings:
    """Load settings from config.json with validation and env overrides"""
    settings = AppSettings()  # Start with defaults
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
            # Update defaults with loaded data
            for key, value in data.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            print(f"✅ Loaded settings from {CONFIG_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading config.json: {e}, using defaults")
            # Create backup of corrupted file
            if CONFIG_FILE.exists():
                corrupt_path = BACKUP_DIR / f'corrupt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                BACKUP_DIR.mkdir(exist_ok=True)
                shutil.copy(CONFIG_FILE, corrupt_path)
                print(f"💾 Backed up corrupted config to {corrupt_path}")
    else:
        print(f"📝 No config.json found, using defaults")
        # Save defaults
        save_settings(settings)
    
    # Apply environment overrides
    settings, overridden = apply_env_overrides(settings)
    if overridden:
        print("⚡ Environment overrides applied")
    
    return settings

def save_settings(settings: AppSettings):
    """Save settings to config.json with automatic backup"""
    # Create backup before overwriting
    if CONFIG_FILE.exists():
        create_backup()
    
    # Save new config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings.dict(), f, indent=2)
        print(f"💾 Saved settings to {CONFIG_FILE}")
    except Exception as e:
        print(f"❌ Error saving settings: {e}")

# ----- RESET FUNCTIONALITY -----
def reset_to_defaults() -> AppSettings:
    """Reset all settings to defaults"""
    # Create backup before reset
    if CONFIG_FILE.exists():
        create_backup()
    
    # Create new default settings
    settings = AppSettings()
    save_settings(settings)
    print("🔄 Settings reset to defaults")
    return settings

# ----- EXPORT/IMPORT (no UI, just functions) -----
def export_settings(file_path: Path) -> bool:
    """Export settings to a specific file"""
    if CONFIG_FILE.exists():
        shutil.copy(CONFIG_FILE, file_path)
        print(f"📤 Exported settings to {file_path}")
        return True
    return False

def import_settings(file_path: Path) -> Optional[AppSettings]:
    """Import settings from a file"""
    if file_path.exists():
        try:
            # Create backup of current
            if CONFIG_FILE.exists():
                create_backup()
            
            # Copy imported file to config location
            shutil.copy(file_path, CONFIG_FILE)
            
            # Load and return new settings
            settings = load_settings()
            print(f"📥 Imported settings from {file_path}")
            return settings
        except Exception as e:
            print(f"❌ Import failed: {e}")
    return None

# ----- CALLBACK SYSTEM (optional, but no app.py changes needed) -----
_callbacks: List[Callable] = []
_suppress_notify: bool = False   # set True during UI-triggered saves to prevent double reload

def on_settings_changed(callback: Callable):
    """Register callback (component must handle its own reload)"""
    _callbacks.append(callback)

def _notify_changed():
    """Notify callbacks — skipped when _suppress_notify is set (UI-triggered saves)."""
    global _suppress_notify
    if _suppress_notify:
        return
    for callback in _callbacks:
        try:
            callback()
        except Exception as e:
            print(f"⚠️ Callback error: {e}")

# Override save_settings to include notifications
original_save = save_settings
def save_settings_with_notify(settings: AppSettings, silent: bool = False):
    """Save settings.
    Pass silent=True from UI handlers to suppress the external-reload callback
    (UI handlers call reload_* themselves; the callback would cause a double reload
    and heap corruption when switching TTS providers like Coqui → pyttsx3).
    """
    original_save(settings)
    if not silent:
        _notify_changed()
save_settings = save_settings_with_notify

# Create global config instance
config = load_settings()

# Optional: Auto-save on attribute changes? Too magic, better explicit
def get_persona_name() -> str:
    """Return the configured persona name. Always use this instead of hardcoding 'Lumina'."""
    return (config.PERSONA_NAME or "Lumina").strip()


def get_birth_date() -> "Optional[datetime]":
    """Parsed LUMINA_BIRTH_DATE, or None if not yet set (first boot, or an
    install from before v117's birth_date-based age model). ai_system.py
    handles the one-time migration via set_birth_date()."""
    raw = (config.LUMINA_BIRTH_DATE or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def set_birth_date(dt: "datetime") -> None:
    """Persist LUMINA_BIRTH_DATE to config.json. Called once, either on a
    genuinely fresh install (dt=now) or as a back-dated migration from a
    pre-v117 install's accumulated current_age (see ai_system.py)."""
    config.LUMINA_BIRTH_DATE = dt.isoformat()
    save_settings(config)
