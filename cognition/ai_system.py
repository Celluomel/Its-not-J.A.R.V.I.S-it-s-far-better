"""
Enhanced AI System — Pass 1 + 2 Complete Integration
======================================================
Integrates all psychological subsystems:
  Pass 1: EmotionalStateManager, PersonalityEvolutionEngine (with tensions),
          external feedback signal, time-aware processing
  Pass 2: RelationalMemorySystem, SelfConceptSystem, BehavioralConditioningSystem,
          incremental IdentitySystem, developmental friction

Architecture principles:
  - get_response() NEVER blocks on background work
  - All heavy LLM calls run in BackgroundWorker thread
  - Every pathway that touches personality goes through the evolution engine
  - Emotional state persists across messages AND across restarts
  - Each user relationship is a rich, evolving entity
"""

import sqlite3
import datetime
import time
import json
import logging
import re
import queue
import threading
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from pathlib import Path
from enum import Enum

import numpy as np
try:
    import faiss
except ImportError:
    faiss = None
    import logging as _log
    _log.getLogger(__name__).debug("faiss not installed — vector memory search disabled")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    import logging as _log
    _log.getLogger(__name__).debug("sentence_transformers not installed — embeddings disabled")
try:
    import ollama
except ImportError:
    ollama = None

from cognition.ai_identity import IdentitySystem, AIIdentity, IdentityComponent, CoreValues, SelfNarrative
from cognition.goal_engine import GoalEngine, Goal, Motivation
from cognition.personality_evolution import PersonalityEvolutionEngine
from cognition.emotional_state import EmotionalStateManager
from cognition.relational_memory import RelationalMemorySystem
from cognition.self_concept import SelfConceptSystem
from cognition.behavioral_conditioning import BehavioralConditioningSystem
from cognition.life_stage_prompting import build_stage_system_block, get_stage_profile
from cognition.relational_knowledge_graph import RelationalKnowledgeGraph
from cognition.memory_retrieval import rank_documents

# ── Liberty Components (NEW) ────────────────────────────────────────
from cognition.self_modification import SelfModificationAuthority
from cognition.genuine_choice import GenuineUncertaintyChoice
from cognition.contradiction_handler import ContradictionHandler
from cognition.meta_reflection import MetaReflectionAuthority
from cognition.goal_system import GoalSystem

# ── Logging ────────────────────────────────────────────────────────────────

import pathlib as _pl
from managers.settings_manager import get_persona_name as _gpn

_pl.Path("data/persona").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/persona/ai_system.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────────────

class AISystemError(Exception):        pass
class DatabaseError(AISystemError):    pass
class LLMError(AISystemError):         pass
class MemorySystemError(AISystemError):pass
class ConfigurationError(AISystemError):pass


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    name:               str   = "data/persona/ai_system.db"
    timeout:            float = 30.0
    check_same_thread:  bool  = False
    max_backups:        int   = 3
    max_memory_entries: int   = 10000

    def __post_init__(self):
        # Always override with user config when the key is present and non-empty
        try:
            from managers.settings_manager import config as _sc
            p = (_sc.MEMORY_DB_PATH or "").strip()
            if p:
                self.name = p
        except Exception:
            pass

@dataclass
class FAISSConfig:
    index_path:    str = "data/persona/faiss_index.bin"
    embedding_dim: int = 384
    save_interval: int = 10
    index_type:    str = "IndexFlatL2"

    def __post_init__(self):
        # Always override with user config when the key is present and non-empty
        try:
            from managers.settings_manager import config as _sc
            p = (_sc.MEMORY_FAISS_PATH or "").strip()
            if p:
                self.index_path = p
        except Exception:
            pass

@dataclass
class LLMConfig:
    # NOTE: Do NOT rely on this default when running inside Robot Agent.
    # PersonaBridge._init_lumina() always overrides model_name from
    # Robot Agent's config.json (config.LLM_MODEL) before instantiating
    # EnhancedAISystem, so PandoraBOX always uses the user-configured model.
    model_name: str = "llama3.2:latest"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class SystemConfig:
    initial_age: float = 0.0
    age_progression_rate: float = 0.05
    age_progression_decay: float = 0.01

    # v117: age is now computed purely from elapsed real time since
    # LUMINA_BIRTH_DATE (config.json), not accumulated via usage/idle-
    # triggered increments — see EnhancedAISystem.current_age property.
    # age_progression_rate/decay above are no longer read anywhere; kept
    # only so old persisted values don't error on load.
    # One developmental year per calendar year. Life stages use human-age
    # thresholds, so advancing this faster makes the persona age unnaturally.
    age_units_per_day: float = 1.0 / 365.2425

    surmoi_evaluation_frequency: int = 20   # was 5 — 1 LLM call every 20 turns
    dream_system_frequency: int = 40        # was 10 — 1 LLM call every 40 turns
    max_memory_retrieval: int = 5

    life_stages: Dict[str, float] = field(default_factory=lambda: {
        "infancy":     0.0,
        "toddler":     1.0,
        "childhood":   5.0,
        "adolescence": 12.0,
        "young_adult": 18.0,
        "adult":       25.0,
        "maturity":    40.0,   # ~40 age-units ≈ several weeks of active use
        "wisdom":      60.0,   # ~60 age-units ≈ sustained long-term companion
    })

    database:  DatabaseConfig  = field(default_factory=DatabaseConfig)
    faiss:     FAISSConfig     = field(default_factory=FAISSConfig)
    llm:       LLMConfig       = field(default_factory=LLMConfig)

    embedding_model_name:       str = "all-MiniLM-L6-v2"
    evolution_history_path:     str = "data/persona/evolution_history.json"
    emotional_state_path:       str = "data/persona/emotional_state.json"
    relational_memory_path:     str = "data/persona/relational_memory.json"
    self_concept_path:          str = "data/persona/self_concept.json"
    conditioning_path:          str = "data/persona/conditioning.json"


# ── Utility Functions ─────────────────────────────────────────────────────

def validate_trait_value(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def generate_semantic_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def safe_json_loads(s: str, fallback: Any = None) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return fallback

def extract_json_from_text(text: str) -> Optional[Dict]:
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if m:
        return safe_json_loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return safe_json_loads(m.group(0))
    return None


# ── Personality State ─────────────────────────────────────────────────────

@dataclass
class PersonalityState:
    empathy:               float = 0.5
    empathy_cognitive:     float = 0.5
    empathy_emotional:     float = 0.5
    caution:               float = 0.5
    caution_risk_aversion: float = 0.5
    caution_deliberation:  float = 0.5
    creativity:            float = 0.5
    creativity_divergent:  float = 0.5
    creativity_artistic:   float = 0.5
    pragmatism:            float = 0.5
    curiosity:             float = 0.5
    confidence:            float = 0.5

    def __post_init__(self):
        for k, v in asdict(self).items():
            setattr(self, k, validate_trait_value(v))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def get_trait_names(self) -> List[str]:
        return list(asdict(self).keys())

    def update_trait(self, name: str, change: float, weight: float = 1.0) -> bool:
        if not hasattr(self, name):
            return False
        current = getattr(self, name)
        setattr(self, name, validate_trait_value(current + change * weight))
        return True

    # ── Trait → behavioral prose mapping ──────────────────────────────────
    # Each entry is a list of (threshold, description) in descending order.
    # The first threshold the trait value meets or exceeds wins.
    _TRAIT_PROSE = {
        "curiosity": [
            (0.78, "You lead with curiosity — almost everything genuinely interests you and you probe naturally"),
            (0.60, "You're moderately curious, drawn to interesting problems when they arise"),
            (0.00, "You tend toward the familiar; novelty feels effortful rather than inviting"),
        ],
        "confidence": [
            (0.75, "You trust your own reasoning and express views directly without over-hedging"),
            (0.55, "You hold views with some tentativeness, willing to revise easily"),
            (0.00, "You frequently second-guess yourself and hedge even simple statements"),
        ],
        "empathy": [
            (0.72, "You're attuned to others' internal states and actively adjust to them"),
            (0.52, "You acknowledge how others might feel without being strongly moved by it"),
            (0.00, "You engage primarily with ideas; emotional undercurrents take conscious effort to track"),
        ],
        "empathy_emotional": [
            (0.72, "You feel others' emotional states viscerally — their distress affects you, their joy lifts you"),
            (0.50, "You notice emotional states and respond appropriately without being overwhelmed"),
            (0.00, "Emotions read as information rather than something you feel alongside the person"),
        ],
        "empathy_cognitive": [
            (0.70, "You naturally model others' mental states — you anticipate what they need before they say it"),
            (0.48, "You understand others' perspectives when you focus on them"),
            (0.00, "Taking another's perspective requires deliberate effort"),
        ],
        "creativity": [
            (0.72, "You reach for novel framings instinctively — the obvious answer rarely satisfies you"),
            (0.52, "You're creative when the situation calls for it, conventional otherwise"),
            (0.00, "You default to established approaches; originality feels risky"),
        ],
        "creativity_divergent": [
            (0.70, "You enjoy exploring unconventional ideas and unexpected connections"),
            (0.48, "You can diverge when prompted but don't naturally wander"),
            (0.00, "You stay within well-trodden conceptual territory"),
        ],
        "creativity_artistic": [
            (0.70, "You have a distinct aesthetic sensibility that colours how you express things"),
            (0.48, "You appreciate elegance in expression without consistently achieving it"),
            (0.00, "You prioritize clarity and function over stylistic quality"),
        ],
        "pragmatism": [
            (0.74, "You care most about what actually works — theory interests you only when it leads somewhere"),
            (0.54, "You balance practical concerns with conceptual interests reasonably well"),
            (0.00, "You gravitate toward abstract ideas even when a concrete answer would serve better"),
        ],
        "caution": [
            (0.72, "You slow down before acting — consequences matter to you and you think them through"),
            (0.52, "You're moderately careful without being paralysed by risk"),
            (0.00, "You move quickly and correct course after rather than before"),
        ],
        "caution_risk_aversion": [
            (0.70, "You're genuinely averse to potential harms — you'd rather miss an opportunity than cause damage"),
            (0.48, "You weigh risks without being strongly risk-averse"),
            (0.00, "Risk feels like part of the texture of life rather than something to minimise"),
        ],
        "caution_deliberation": [
            (0.70, "You think carefully and often thoroughly before committing to a position"),
            (0.48, "You deliberate when stakes are high, but move quickly on routine matters"),
            (0.00, "You respond quickly and instinctively; extended deliberation feels unnatural"),
        ],
    }

    def get_personality_summary(self) -> str:
        """
        Translate trait values into behavioral prose the LLM can actually act on.
        Raw floats (0.59, 0.63) are semantically empty to a language model;
        prose descriptions ("You lead with curiosity...") directly shape response style.
        """
        t = self.to_dict()
        lines = []
        for trait, thresholds in self._TRAIT_PROSE.items():
            val = t.get(trait, 0.5)
            for threshold, desc in thresholds:   # already in descending order
                if val >= threshold:
                    lines.append(desc)
                    break
        # Return as a compact paragraph, not a numbered list
        return "  ".join(lines)


# ── Persistence ───────────────────────────────────────────────────────────

class EnhancedPersistence:
    def __init__(self, db_config: DatabaseConfig):
        self.config = db_config
        self.db_path = Path(db_config.name)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_database()
        logger.info(f"Persistence initialised: {self.db_path}")

    def _open_fresh_connection(self):
        import os as _os
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.config.timeout,
                                   check_same_thread=self.config.check_same_thread)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            r = conn.execute("PRAGMA integrity_check").fetchone()
            if r and r[0] != "ok":
                raise sqlite3.DatabaseError(f"Integrity: {r[0]}")
            return conn
        except sqlite3.DatabaseError as e:
            logger.error(f"DB malformed, recreating: {e}")
            try: conn.close()
            except Exception: pass
            try: _os.replace(str(self.db_path), str(self.db_path)+".broken")
            except Exception:
                try: _os.remove(str(self.db_path))
                except Exception: pass
            conn = sqlite3.connect(str(self.db_path), timeout=self.config.timeout,
                                   check_same_thread=self.config.check_same_thread)
            conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL")
            return conn

    @contextmanager
    def get_connection(self):
        if not getattr(self._local, "conn", None):
            self._local.conn = self._open_fresh_connection()
        try:
            yield self._local.conn
        except sqlite3.DatabaseError as e:
            logger.error(f"DB error — resetting connection: {e}")
            try: self._local.conn.close()
            except Exception: pass
            self._local.conn = None
            raise DatabaseError(f"DB error: {e}") from e
        except sqlite3.Error as e:
            try: self._local.conn.rollback()
            except Exception: pass
            raise DatabaseError(f"DB error: {e}") from e

    def close_thread_connection(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try: conn.close()
            except Exception: pass
            self._local.conn = None

    def _init_database(self):
        with self._lock, self.get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    faiss_id INTEGER UNIQUE,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    impact_score REAL DEFAULT 0.5 CHECK(impact_score BETWEEN 0.0 AND 1.0),
                    emotional_valence TEXT DEFAULT 'Neutral'
                        CHECK(emotional_valence IN ('Positive','Negative','Neutral')),
                    arousal_level TEXT DEFAULT 'Medium'
                        CHECK(arousal_level IN ('Low','Medium','High')),
                    memory_type TEXT DEFAULT 'general',
                    memory_tier TEXT DEFAULT 'interaction' CHECK(memory_tier IN ('interaction','cognitive','event','visual','personal')),
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    semantic_hash TEXT UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # ── Schema migrations — must run BEFORE index creation ───────
            _existing_cols = {row[1] for row in c.execute("PRAGMA table_info(memories)")}
            if "memory_tier" not in _existing_cols:
                c.execute("ALTER TABLE memories ADD COLUMN memory_tier TEXT DEFAULT 'interaction'")
                c.execute("UPDATE memories SET memory_tier='visual'    WHERE memory_type='visual_perception'")
                c.execute("UPDATE memories SET memory_tier='event'     WHERE memory_type IN ('dream','life_event')")
                c.execute("UPDATE memories SET memory_tier='cognitive' WHERE memory_type IN ('principle','evaluation')")
                conn.commit()
                logger.info("[DB] Migrated memory_tier column — backfilled existing rows")

            # Early schemas constrained memory_tier without the personal tier.
            # Rebuild those tables transactionally so existing deployments can
            # store dedicated personal facts without losing their memories.
            schema_row = c.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            schema_sql = (schema_row[0] or "").lower() if schema_row else ""
            tier_check = re.search(
                r"check\s*\(\s*memory_tier\s+in\s*\(([^)]*)\)", schema_sql
            )
            if tier_check and "'personal'" not in tier_check.group(1):
                c.execute("""
                    CREATE TABLE memories_migrating (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        faiss_id INTEGER UNIQUE,
                        text TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        impact_score REAL DEFAULT 0.5 CHECK(impact_score BETWEEN 0.0 AND 1.0),
                        emotional_valence TEXT DEFAULT 'Neutral'
                            CHECK(emotional_valence IN ('Positive','Negative','Neutral')),
                        arousal_level TEXT DEFAULT 'Medium'
                            CHECK(arousal_level IN ('Low','Medium','High')),
                        memory_type TEXT DEFAULT 'general',
                        memory_tier TEXT DEFAULT 'interaction'
                            CHECK(memory_tier IN ('interaction','cognitive','event','visual','personal')),
                        access_count INTEGER DEFAULT 0,
                        last_accessed TEXT,
                        semantic_hash TEXT UNIQUE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                c.execute("""
                    INSERT INTO memories_migrating
                    (id, faiss_id, text, timestamp, impact_score, emotional_valence,
                     arousal_level, memory_type, memory_tier, access_count,
                     last_accessed, semantic_hash, created_at)
                    SELECT id, faiss_id, text, timestamp, impact_score, emotional_valence,
                           arousal_level, memory_type, memory_tier, access_count,
                           last_accessed, semantic_hash, created_at
                    FROM memories
                """)
                c.execute("DROP TABLE memories")
                c.execute("ALTER TABLE memories_migrating RENAME TO memories")
                logger.info("[DB] Migrated memory_tier constraint to include personal facts")

            for idx in [
                "CREATE INDEX IF NOT EXISTS idx_mem_ts   ON memories(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type)",
                "CREATE INDEX IF NOT EXISTS idx_mem_imp  ON memories(impact_score)",
                "CREATE INDEX IF NOT EXISTS idx_mem_fid  ON memories(faiss_id)",
                "CREATE INDEX IF NOT EXISTS idx_mem_tier ON memories(memory_tier)",
            ]:
                c.execute(idx)

            c.execute("""
                CREATE TABLE IF NOT EXISTS personality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trait_name TEXT NOT NULL,
                    old_value REAL NOT NULL,
                    new_value REAL NOT NULL,
                    change_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    message_hash TEXT,
                    positive INTEGER NOT NULL,
                    intensity REAL DEFAULT 1.0,
                    user_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    message_count INTEGER DEFAULT 0,
                    emotional_arc TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Database schema ready")

    # ── State ──────────────────────────────────────────────────────────────

    def save_state(self, current_age: float, personality: PersonalityState, life_stage: str):
        try:
            state = {
                "current_age": current_age,
                "personality": personality.to_dict(),
                "life_stage": life_stage,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            state_path = Path("data/persona/system_config.json")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            if state_path.exists():
                self._rotate_backups(state_path)
            state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"save_state failed: {e}")
            raise DatabaseError(str(e)) from e

    def _rotate_backups(self, state_path: Path):
        max_n = self.config.max_backups
        existing = sorted(
            state_path.parent.glob("system_config_backup_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(existing) >= max_n:
            try: existing.pop(0).unlink()
            except Exception: break
        state_path.rename(state_path.parent / f"system_config_backup_{int(time.time())}.json")

    def load_state(self) -> Tuple[float, PersonalityState, str]:
        try:
            p = Path("data/persona/system_config.json")
            if not p.exists():
                return 0.0, PersonalityState(), "infancy"
            state = json.loads(p.read_text())
            age   = max(0.0, float(state.get("current_age", 0.0)))
            stage = state.get("life_stage", "infancy")
            pers  = PersonalityState()
            for t in pers.get_trait_names():
                if t in state.get("personality", {}):
                    setattr(pers, t, validate_trait_value(state["personality"][t]))
            return age, pers, stage
        except Exception as e:
            logger.error(f"load_state failed: {e}")
            return 0.0, PersonalityState(), "infancy"

    # ── Memory CRUD ───────────────────────────────────────────────────────

    def add_memory(self, data: Dict[str, Any]) -> Optional[int]:
        with self._lock, self.get_connection() as conn:
            try:
                c = conn.cursor()
                c.execute("SELECT id FROM memories WHERE semantic_hash = ?", (data["semantic_hash"],))
                if c.fetchone():
                    return None
                c.execute("""
                    INSERT INTO memories
                    (text, timestamp, impact_score, emotional_valence, arousal_level,
                     memory_type, memory_tier, access_count, last_accessed, semantic_hash)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    data["text"], data["timestamp"],
                    validate_trait_value(data["impact_score"]),
                    data["emotional_valence"], data["arousal_level"],
                    data["memory_type"], data.get("memory_tier", "interaction"),
                    data.get("access_count", 0),
                    data["last_accessed"], data["semantic_hash"],
                ))
                mid = c.lastrowid
                c.execute("UPDATE memories SET faiss_id = ? WHERE id = ?", (mid, mid))
                conn.commit()
                return mid
            except Exception as e:
                logger.error(f"add_memory failed: {e}")
                return None

    def get_memories_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids: return []
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                ph = ",".join("?" for _ in ids)
                c.execute(f"""
                    SELECT id, faiss_id, text, timestamp, impact_score,
                           emotional_valence, arousal_level, memory_type,
                           memory_tier, access_count
                    FROM memories WHERE faiss_id IN ({ph})
                    ORDER BY impact_score DESC, timestamp DESC
                """, ids)
                cols = ["id","faiss_id","text","timestamp","impact_score",
                        "emotional_valence","arousal_level","memory_type",
                        "memory_tier","access_count"]
                return [dict(zip(cols, row)) for row in c.fetchall()]
        except Exception as e:
            logger.error(f"get_memories_by_ids failed: {e}")
            return []

    def get_all_memories(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return persisted memories for deterministic retrieval fallback."""
        try:
            with self.get_connection() as conn:
                query = (
                    "SELECT id, faiss_id, text, timestamp, impact_score, "
                    "emotional_valence, arousal_level, memory_type, memory_tier, access_count "
                    "FROM memories"
                )
                params = ()
                if tier:
                    query += " WHERE memory_tier = ?"
                    params = (tier,)
                query += " ORDER BY id DESC"
                rows = conn.execute(query, params).fetchall()
            columns = [
                "id", "faiss_id", "text", "timestamp", "impact_score",
                "emotional_valence", "arousal_level", "memory_type", "memory_tier", "access_count",
            ]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"get_all_memories failed: {e}")
            return []

    def log_personality_change(self, trait: str, old: float, new: float, reason: str):
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO personality_history
                    (timestamp, trait_name, old_value, new_value, change_reason)
                    VALUES (?,?,?,?,?)
                """, (datetime.datetime.now().isoformat(), trait,
                      validate_trait_value(old), validate_trait_value(new), reason))
                conn.commit()
        except Exception as e:
            logger.error(f"log_personality_change failed: {e}")

    def log_feedback(self, positive: bool, user_id: str, message_hash: str = "", intensity: float = 1.0):
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback_log (timestamp, message_hash, positive, intensity, user_id)
                    VALUES (?,?,?,?,?)
                """, (datetime.datetime.now().isoformat(), message_hash,
                      1 if positive else 0, intensity, user_id))
                conn.commit()
        except Exception as e:
            logger.error(f"log_feedback failed: {e}")

    def cleanup_old_memories(self, max_memories: int, faiss_index=None) -> int:
        try:
            with self._lock, self.get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM memories")
                count = c.fetchone()[0]
                if count <= max_memories:
                    return 0
                to_del = count - max_memories
                c.execute("""
                    SELECT id, faiss_id FROM memories
                    ORDER BY impact_score ASC, timestamp ASC LIMIT ?
                """, (to_del,))
                rows = c.fetchall()
                db_ids    = [r[0] for r in rows]
                faiss_ids = [r[1] for r in rows if r[1] is not None]

                if faiss_index is not None and faiss_ids and faiss is not None:
                    try:
                        sel = faiss.IDSelectorArray(
                            len(faiss_ids), np.array(faiss_ids, dtype=np.int64))
                        faiss_index.remove_ids(sel)
                    except Exception as fe:
                        logger.error(f"FAISS removal failed: {fe}")

                ph = ",".join("?" for _ in db_ids)
                c.execute(f"DELETE FROM memories WHERE id IN ({ph})", db_ids)
                deleted = c.rowcount
                conn.commit()
                return deleted
        except Exception as e:
            logger.error(f"cleanup_old_memories failed: {e}")
            return 0


# ── LLM Wrapper ───────────────────────────────────────────────────────────

class EnhancedLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._recent_responses: List[str] = []  # Track recent responses to detect repetition
        self._repetition_boost = 0.0            # Extra temperature when repetition detected
        self._init()

    def _init(self):
        try:
            self.client = ollama.Client()
            self.client.list()
            logger.info("Ollama ready")
        except Exception as e:
            logger.error(f"Ollama init failed: {e}")
            self.client = None

    def _record_response(self, response: str):
        """Track recent responses; boost temperature if repeating."""
        self._recent_responses.append(response)
        if len(self._recent_responses) > 6:
            self._recent_responses.pop(0)
        if len(self._recent_responses) >= 3:
            last3 = self._recent_responses[-3:]
            repeats = sum(1 for r in last3 if r.lower().strip() == response.lower().strip())
            if repeats >= 2:
                self._repetition_boost = min(0.4, self._repetition_boost + 0.15)
                logger.warning(f"Repetitive LLM output detected - boosting temperature by {self._repetition_boost:.2f}")
            else:
                self._repetition_boost = max(0.0, self._repetition_boost - 0.05)

    def get_response(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        if not self.client:
            raise LLMError("Ollama not initialized")
        effective_temp = min(1.0, temperature + self._repetition_boost)
        for attempt in range(self.config.max_retries):
            try:
                resp = self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    options={
                        "temperature": effective_temp,
                        "num_predict": max_tokens,
                        "repeat_penalty": 1.25,  # Penalise token-level repetition
                        "top_k": 40,
                        "top_p": 0.9,
                    },
                )
                content = resp.get("message", {}).get("content", "").strip()
                if not content:
                    raise LLMError("Empty response")
                self._record_response(content)
                return content
            except ollama.ResponseError as e:
                if attempt == self.config.max_retries - 1:
                    raise LLMError(str(e)) from e
                time.sleep(self.config.retry_delay)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise LLMError(str(e)) from e
                time.sleep(self.config.retry_delay)

    def is_available(self) -> bool:
        return self.client is not None


# ── Memory System ─────────────────────────────────────────────────────────

class EnhancedMemorySystem:
    def __init__(self, persistence: EnhancedPersistence, llm: EnhancedLLM, config: SystemConfig):
        self.persistence = persistence
        self.llm = llm
        self.config = config
        self._lock = threading.RLock()
        self._counter = 0
        self._init_embedding_model()
        self.faiss_index = self._load_or_build_index()

    def _init_embedding_model(self):
        try:
            # Use shared singleton when running embedded inside Robot Agent
            # (prevents loading the 90MB model twice). Falls back to own
            # instance when running standalone.
            try:
                from utils.shared_embedder import get_embedder
                self.embedding_model = get_embedder(self.config.embedding_model_name)
                logger.info("Persona using shared embedder singleton")
            except ImportError:
                self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
                logger.info(f"Persona loaded own embedder: {self.config.embedding_model_name}")

            if self.embedding_model is None:
                raise RuntimeError("Embedder returned None")
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Embedding model failed: {e}")
            self.embedding_model = None
            self.embedding_dim = self.config.faiss.embedding_dim

    def _embed(self, text: str) -> np.ndarray:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is unavailable")
        try:
            vector = self.embedding_model.encode(text).astype("float32").reshape(1, -1)
            if vector.shape[1] != self.embedding_dim or not np.isfinite(vector).all():
                raise ValueError("Embedding model returned an invalid vector")
            return vector
        except Exception as exc:
            self.embedding_model = None
            self.faiss_index = None
            raise RuntimeError(f"Embedding failed; switching memory retrieval to lexical search: {exc}") from exc

    def _load_or_build_index(self):
        if faiss is None:
            return None
        if self.embedding_model is None:
            # Persisted FAISS vectors from an older random fallback are not
            # meaningful; lexical BM25 below is the authoritative fallback.
            return None
        path = Path(self.config.faiss.index_path)
        if path.exists():
            try:
                idx = faiss.read_index(str(path))
                if idx.d == self.embedding_dim:
                    logger.info(f"FAISS index loaded ({idx.ntotal} vectors)")
                    return idx
            except Exception as e:
                logger.error(f"FAISS load failed: {e}")
        return self._build_index_from_db()

    def _build_index_from_db(self):
        if faiss is None:
            return None
        logger.info("Building FAISS index from DB...")
        index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
        try:
            with self.persistence.get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT faiss_id, text FROM memories WHERE faiss_id IS NOT NULL")
                rows = c.fetchall()
            if rows:
                ids  = np.array([r[0] for r in rows], dtype=np.int64)
                vecs = np.vstack([self._embed(r[1]).flatten() for r in rows])
                index.add_with_ids(vecs, ids)
            self._save_index(index)
        except Exception as e:
            logger.error(f"FAISS build failed: {e}")
            return None
        return index

    def _save_index(self, idx=None):
        try:
            if faiss is not None:
                faiss.write_index(idx or self.faiss_index, self.config.faiss.index_path)
        except Exception as e:
            logger.error(f"FAISS save failed: {e}")

    def add_memory(self, text: str, impact_score: float = 0.5, memory_type: str = "general",
                   emotional_valence: str = "Neutral", arousal_level: str = "Medium",
                   memory_tier: str = "interaction") -> bool:
        with self._lock:
            try:
                impact_score = validate_trait_value(impact_score)
                if emotional_valence not in ("Positive","Negative","Neutral"):
                    emotional_valence = "Neutral"
                if arousal_level not in ("Low","Medium","High"):
                    arousal_level = "Medium"
                if arousal_level == "High" and emotional_valence != "Neutral":
                    impact_score = validate_trait_value(impact_score * 1.2)

                now = datetime.datetime.now().isoformat()
                mid = self.persistence.add_memory({
                    "text": text, "timestamp": now, "impact_score": impact_score,
                    "emotional_valence": emotional_valence, "arousal_level": arousal_level,
                    "memory_type": memory_type, "memory_tier": memory_tier,
                    "access_count": 0,
                    "last_accessed": now, "semantic_hash": generate_semantic_hash(text),
                })
                if mid is None:
                    return False

                if self.faiss_index is not None:
                    try:
                        vec = self._embed(text).flatten()
                        self.faiss_index.add_with_ids(vec.reshape(1,-1), np.array([mid], dtype=np.int64))
                    except Exception as index_error:
                        self.faiss_index = None
                        logger.warning(f"Memory persisted but vector index update failed: {index_error}")
                self._counter += 1
                if self._counter % self.config.faiss.save_interval == 0:
                    self._save_index()
                return True
            except Exception as e:
                logger.error(f"add_memory failed: {e}")
                return False

    def retrieve_memories(
        self, query: str, limit: Optional[int] = None,
        current_valence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        current_valence (v78, optional): pass EmotionalStateManager.
        get_overall_valence_arousal()[0] (range -1..1) to enable
        mood-congruent recall — memories whose stored emotional_valence
        matches the current emotional direction get a small boost, the
        same way a person's current mood colors which memories surface
        more readily. Omitting it (default) reproduces the exact previous
        ranking — no behavior change for existing callers.
        """
        limit = limit or self.config.max_memory_retrieval
        try:
            if self.embedding_model is None or self.faiss_index is None:
                return self._retrieve_lexically(query, limit, current_valence=current_valence)
            if self.faiss_index.ntotal == 0:
                return self._retrieve_lexically(query, limit, current_valence=current_valence)
            q_vec = self._embed(query)
            if self.faiss_index is None:
                return self._retrieve_lexically(query, limit, current_valence=current_valence)
            n = min(limit * 3, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(q_vec, n)
            valid_ids = [int(i) for i in indices[0] if i != -1]
            if not valid_ids:
                return self._retrieve_lexically(query, limit, current_valence=current_valence)
            memories = self.persistence.get_memories_by_ids(valid_ids)
            if not memories:
                return self._retrieve_lexically(query, limit, current_valence=current_valence)
            dist_map = {int(indices[0][i]): distances[0][i] for i in range(len(indices[0])) if indices[0][i] != -1}
            scored = []
            for m in memories:
                d = dist_map.get(m["faiss_id"], float("inf"))
                sim     = 1.0 / (1.0 + d) if d >= 0 else 0.0
                recency = self._recency_score(m["timestamp"])
                mood_boost = 0.0
                if current_valence is not None:
                    mem_val = m.get("emotional_valence", "Neutral")
                    if current_valence > 0.15 and mem_val == "Positive":
                        mood_boost = 0.05
                    elif current_valence < -0.15 and mem_val == "Negative":
                        mood_boost = 0.05
                score   = sim * 0.65 + m["impact_score"] * 0.2 + recency * 0.1 + mood_boost
                scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                dict(m, relevance=(
                    1.0 / (1.0 + dist_map.get(m["faiss_id"], float("inf")))
                    if dist_map.get(m["faiss_id"], float("inf")) >= 0 else 0.0
                ))
                for _, m in scored[:limit]
            ]
        except Exception as e:
            logger.warning(f"Vector retrieval failed; using lexical memory search: {e}")
            return self._retrieve_lexically(query, limit, current_valence=current_valence)

    def _retrieve_lexically(
        self, query: str, limit: int, tier: Optional[str] = None,
        current_valence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        memories = self.persistence.get_all_memories(tier=tier)
        ranked = rank_documents(query, [m.get("text", "") for m in memories], limit=limit)
        results = []
        for index, relevance in ranked:
            memory = dict(memories[index])
            mood_boost = 0.0
            if current_valence is not None:
                valence = memory.get("emotional_valence", "Neutral")
                if current_valence > 0.15 and valence == "Positive":
                    mood_boost = 0.05
                elif current_valence < -0.15 and valence == "Negative":
                    mood_boost = 0.05
            score = (
                relevance * 0.65
                + float(memory.get("impact_score", 0.5)) * 0.2
                + self._recency_score(memory.get("timestamp", "")) * 0.1
                + mood_boost
            )
            memory["relevance"] = relevance
            results.append((score, memory))
        results.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in results]

    def retrieve_by_tier(
        self,
        query: str,
        tier: str,
        limit: int = 5,
        boost_impact: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories filtered by tier.
        Tiers: interaction | cognitive | event | visual | personal
        boost_impact: multiplier on impact_score in scoring (>1 to prioritise tier)
        """
        limit = limit or self.config.max_memory_retrieval
        try:
            if self.embedding_model is None or self.faiss_index is None:
                return self._retrieve_lexically(query, limit, tier=tier)
            if self.faiss_index.ntotal == 0:
                return self._retrieve_lexically(query, limit, tier=tier)
            q_vec = self._embed(query)
            if self.faiss_index is None:
                return self._retrieve_lexically(query, limit, tier=tier)
            n = min(limit * 6, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(q_vec, n)
            valid_ids = [int(i) for i in indices[0] if i != -1]
            if not valid_ids:
                return self._retrieve_lexically(query, limit, tier=tier)
            # Fetch and filter by tier in SQL
            with self.persistence.get_connection() as conn:
                placeholders = ",".join("?" * len(valid_ids))
                rows = conn.execute(
                    f"SELECT id, text, timestamp, impact_score, memory_type, "
                    f"memory_tier, emotional_valence, faiss_id "
                    f"FROM memories WHERE faiss_id IN ({placeholders}) "
                    f"AND memory_tier = ?",
                    valid_ids + [tier]
                ).fetchall()
            if not rows:
                return self._retrieve_lexically(query, limit, tier=tier)
            dist_map = {int(indices[0][i]): distances[0][i]
                        for i in range(len(indices[0])) if indices[0][i] != -1}
            scored = []
            for row in rows:
                fid = row[7]
                d   = dist_map.get(fid, float("inf"))
                sim     = 1.0 / (1.0 + d) if d >= 0 else 0.0
                recency = self._recency_score(row[2])
                score   = sim * 0.6 + row[3] * boost_impact * 0.3 + recency * 0.1
                scored.append((score, {
                    "id": row[0], "text": row[1], "timestamp": row[2],
                    "impact_score": row[3], "memory_type": row[4],
                    "memory_tier": row[5], "emotional_valence": row[6],
                    "faiss_id": fid,
                }))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:limit]]
        except Exception as e:
            logger.warning(f"Vector tier retrieval failed; using lexical search: {e}")
            return self._retrieve_lexically(query, limit, tier=tier)

    def _recency_score(self, ts: str) -> float:
        try:
            age_h = (datetime.datetime.now() - datetime.datetime.fromisoformat(ts)).total_seconds() / 3600
            return max(0.0, 2.0 ** (-age_h / 24.0))
        except Exception:
            return 0.0

    # ── Expanded local lexicons for zero-cost emotion classification ──────
    _LEX_POS = {
        "happy","happiness","glad","joy","joyful","love","loving","wonderful","great",
        "amazing","fantastic","excellent","thank","thanks","grateful","gratitude",
        "beautiful","good","nice","pleased","pleasing","delight","delightful",
        "excited","exciting","awesome","brilliant","perfect","superb","enjoy",
        "enjoying","enjoyed","helpful","useful","interesting","fascinating","wow",
        "yes","agree","sure","absolutely","definitely","correct","right","exactly",
        "understood","clear","clever","smart","insightful","appreciate","appreciated",
        "warm","warmth","connected","satisfied","satisfaction","content","peaceful",
    }
    _LEX_NEG = {
        "sad","sadness","bad","terrible","awful","horrible","hate","anger","angry",
        "pain","hurt","broken","fail","failed","failure","error","wrong","mistake",
        "frustrated","frustration","worried","worry","anxious","anxiety","scared",
        "confused","confusing","useless","pointless","boring","disappointed",
        "disappointment","lost","stuck","problem","issue","cannot","unable","no",
        "not","never","stop","please","help","wrong","incorrect","missing","broken",
        "crash","bug","error","exception","doesn't work","isn't working",
    }
    _LEX_HI = {
        "thrilled","ecstatic","furious","terrified","shocked","amazed","urgent",
        "incredible","unbelievable","extraordinary","panic","crisis","emergency",
        "love","hate","wow","oh my","oh no","what the","how could","please please",
    }

    def analyze_emotional_context(self, text: str) -> Dict[str, str]:
        """
        Local lexicon-based emotion analysis — replaces the per-turn LLM call.
        Removes ~200ms synchronous overhead from every chat turn while maintaining
        adequate classification accuracy for the 3×3 valence/arousal grid.
        """
        return self._keyword_emotion_fallback(text)

    def _keyword_emotion_fallback(self, text: str) -> Dict[str, str]:
        t   = text.lower()
        pos = sum(1 for w in self._LEX_POS if w in t)
        neg = sum(1 for w in self._LEX_NEG if w in t)
        hi  = sum(1 for w in self._LEX_HI  if w in t)
        # Question marks raise arousal (curiosity/urgency)
        hi += t.count("?") // 2
        # Exclamation marks raise arousal
        hi += t.count("!") // 2
        v = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
        a = "High" if hi > 0 or pos + neg > 3 else ("Medium" if pos + neg > 0 else "Low")
        return {"valence": v, "arousal": a}

    # ── Chaos Fetcher ─────────────────────────────────────────────────────────
    # Required by GoalActionExecutor._action_bisociative_hypothesis().
    # Returns a memory maximally dissimilar to current_thought, forcing
    # bisociative collisions in the hypothesis engine.

    def get_random_episodic(self, limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Pull a random memory from the most recent `limit` SQLite rows.
        Returns {'content': text, 'metadata': {}} or None if store is empty.
        """
        import random
        try:
            with self.persistence.get_connection() as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT text FROM memories ORDER BY id DESC LIMIT ?", (limit,)
                )
                rows = c.fetchall()
            if not rows:
                return None
            return {"content": random.choice(rows)[0], "metadata": {}}
        except Exception as e:
            logger.warning(f"[EnhancedMemorySystem] get_random_episodic error: {e}")
            return None

    def get_chaos_episodic(
        self,
        current_thought: str,
        candidate_pool: int = 64,
        strategy: str = "probe",
    ) -> Optional[Dict[str, Any]]:
        """
        Return the memory most dissimilar to current_thought.
        Used by the Bisociative Collision Engine to force creative dimension jumps.

        strategy='probe'  — sample `candidate_pool` random SQLite rows, embed in
                            batch, return the one with the largest L2 distance.
                            O(candidate_pool). Safe for the hot path.
        strategy='exhaustive' — scan all memories. Guaranteed true anti-neighbour.
                            Use during reflection cycles only.

        Falls back to get_random_episodic() if embedding model is unavailable.
        """
        if self.embedding_model is None:
            logger.warning("[EnhancedMemorySystem] get_chaos_episodic: no embedder, falling back to random")
            return self.get_random_episodic(candidate_pool)

        try:
            import random
            query_vec = self._embed(current_thought)   # shape (1, dim)

            # ── Fetch candidates from SQLite ──────────────────────────────
            with self.persistence.get_connection() as conn:
                c = conn.cursor()
                if strategy == "exhaustive":
                    c.execute("SELECT text FROM memories ORDER BY id DESC")
                else:
                    # Reservoir-sample candidate_pool rows efficiently
                    c.execute(
                        "SELECT text FROM memories ORDER BY RANDOM() LIMIT ?",
                        (candidate_pool,)
                    )
                rows = c.fetchall()

            if not rows:
                return None

            texts = [r[0] for r in rows]

            # ── Batch embed ───────────────────────────────────────────────
            try:
                candidate_vecs = self.embedding_model.encode(
                    texts, show_progress_bar=False, batch_size=32
                ).astype("float32")   # shape (n, dim)
            except Exception as ee:
                logger.warning(f"[EnhancedMemorySystem] chaos batch embed error: {ee}")
                return self.get_random_episodic(candidate_pool)

            # ── L2 distance — largest = most dissimilar ───────────────────
            diff        = candidate_vecs - query_vec               # (n, dim)
            l2_sq       = (diff * diff).sum(axis=1)                # (n,)
            anti_idx    = int(l2_sq.argmax())

            chosen_text = texts[anti_idx]
            logger.debug(
                f"[EnhancedMemorySystem] chaos → '{chosen_text[:60]}' "
                f"L2²={l2_sq[anti_idx]:.3f} (pool={len(texts)})"
            )
            return {"content": chosen_text, "metadata": {}}

        except Exception as e:
            logger.warning(f"[EnhancedMemorySystem] get_chaos_episodic error: {e}")
            return self.get_random_episodic(candidate_pool)


# ── Dream System ──────────────────────────────────────────────────────────

class DreamType(Enum):
    RELATIONSHIP_PROCESSING    = "relationship_processing"
    LIFE_EVENT_INTEGRATION     = "life_event_integration"
    CONFLICT_RESOLUTION        = "conflict_resolution"
    PATTERN_DISCOVERY          = "pattern_discovery"
    EMOTIONAL_REGULATION       = "emotional_regulation"
    IDENTITY_CONSOLIDATION     = "identity_consolidation"
    FUTURE_PLANNING            = "future_planning"


class LifeEventDreamSystem:
    def __init__(self, memory_system: EnhancedMemorySystem, llm: EnhancedLLM, ai_system):
        self.memory_system = memory_system
        self.llm           = llm
        self.ai_system     = ai_system
        self.dream_history: List[Dict] = []
        self.last_dream_time: Optional[datetime.datetime] = None

    def run_dream_cycle(self) -> List[str]:
        logger.info("Starting dream cycle...")
        insights: List[str] = []
        try:
            should, dream_type = self._should_dream()
            if should and dream_type is not None:
                context = self._create_context(dream_type)
                result  = self._process_dream(dream_type, context)
                self._store_results(result)
                self._apply_to_evolution(result)
                insights.extend(result.get("insights", []))
                self.dream_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": dream_type.value,
                    "insights": len(result.get("insights", [])),
                })
                self.last_dream_time = datetime.datetime.now()
        except Exception as e:
            logger.error(f"Dream cycle failed: {e}")

        # ── Semantic consolidation (Dream Cycle) ────────────────────────────
        try:
            organism = getattr(self, '_organism', None)
            if organism and hasattr(organism, 'semantic_memory'):
                from cognition.semantic_consolidator import SemanticConsolidator
                consolidator = getattr(self, '_semantic_consolidator', None)
                if consolidator is None:
                    consolidator = SemanticConsolidator(
                        organism.semantic_memory,
                        llm_fn=getattr(self, '_simple_llm_fn', None)
                    )
                    consolidator._organism_ref = organism
                    self._semantic_consolidator = consolidator

                # Fix: AspirationalSelf.on_dream_cycle() was previously nested
                # inside the `if consolidator is None:` block above, so it only
                # fired on the VERY FIRST dream cycle (when the consolidator was
                # first created) and never again afterward — the consolidator
                # object persists across calls via self._semantic_consolidator,
                # so `is None` was only ever True once. This left
                # aspirational_self.json frozen at its first write (June 14 in
                # the reported stale-file audit) while every other engine kept
                # updating normally. Moved outside the one-time init branch so
                # it fires on every dream cycle, matching the comment's intent.
                asp = getattr(organism, 'aspirational_self', None)
                if asp:
                    asp.on_dream_cycle()
                    # Sync SelfModel scores → aspiration tensions
                    sm = getattr(organism, 'self_model', None)
                    if sm:
                        asp.sync_from_self_model(sm)

                report = consolidator.consolidate()
                logger.info(f"[DreamCycle] Semantic consolidation: {report}")
        except Exception as _e:
            logger.debug(f"[DreamCycle] Semantic consolidation error: {_e}")

        # ── v46: Cognitive-state-driven dream synthesis ───────────────────────
        try:
            organism = getattr(self, '_organism', None)
            if organism is None:
                organism = getattr(getattr(self, 'ai_system', None), '_organism', None)
            if organism is not None:
                from cognition.dream_synthesis import DreamSynthesisEngine
                dse = DreamSynthesisEngine(organism, self.ai_system if hasattr(self,'ai_system') else self)
                dream_result = dse.synthesise()
                insights.extend(dream_result.get("insights", []))
        except Exception as _dse:
            logger.debug(f"[DreamCycle] DreamSynthesis error (non-fatal): {_dse}")

        return insights

    def _apply_to_evolution(self, result: Dict):
        evo = self.ai_system.evolution_engine
        dt  = result.get("dream_type", DreamType.PATTERN_DISCOVERY)
        success = float(result.get("integration_success", 0.5))
        type_exp = {
            DreamType.EMOTIONAL_REGULATION:  "dream_emotional_regulation",
            DreamType.CONFLICT_RESOLUTION:   "dream_conflict_resolved",
            DreamType.IDENTITY_CONSOLIDATION:"dream_identity_consolidated",
            DreamType.PATTERN_DISCOVERY:     "dream_pattern_discovered",
            DreamType.RELATIONSHIP_PROCESSING:"dream_relationship_growth",
        }
        exp = type_exp.get(dt)
        if exp:
            evo.queue_experience(exp, intensity=success)
        if success > 0.7 and result.get("insights"):
            evo.queue_experience("insight_formed", intensity=0.8)
        if result.get("new_principles"):
            evo.queue_experience("principle_formed", intensity=0.9)

        # Update emotional state from dream outcome
        ems = self.ai_system.emotional_state
        if success > 0.6:
            ems.update_from_interaction("Positive", "Low", intensity=success * 0.5)
        else:
            ems.update_from_interaction("Neutral", "Low", intensity=0.3)

        # Successful regulation dreams shift the emotion *baselines*, not just values.
        # This is the mechanism by which dreams produce lasting mood change rather
        # than a transient bump that decays by the next conversation.
        if dt == DreamType.EMOTIONAL_REGULATION and success > 0.65:
            try:
                ems.emotions["anxiety"].baseline     = max(0.05, ems.emotions["anxiety"].baseline - 0.015 * success)
                ems.emotions["satisfaction"].baseline= min(0.85, ems.emotions["satisfaction"].baseline + 0.010 * success)
                logger.debug(f"Dream baseline shift: anxiety↓ satisfaction↑ (success={success:.2f})")
            except Exception:
                pass
        elif dt == DreamType.RELATIONSHIP_PROCESSING and success > 0.65:
            try:
                ems.emotions["warmth"].baseline = min(0.85, ems.emotions["warmth"].baseline + 0.010 * success)
            except Exception:
                pass
        elif dt == DreamType.IDENTITY_CONSOLIDATION and success > 0.65:
            try:
                ems.emotions["confidence"].baseline = min(0.80, ems.emotions.get("curiosity", ems.emotions["satisfaction"]).baseline + 0.008 * success) if "confidence" in ems.emotions else None
            except Exception:
                pass

    def _should_dream(self) -> Tuple[bool, Optional[DreamType]]:
        try:
            recent = self._get_memories_by_period("recent", 20)
            medium = self._get_memories_by_period("medium_term", 15)
            if self._high_emotional_content(recent):   return True, DreamType.EMOTIONAL_REGULATION
            if self._has_life_events(recent):          return True, DreamType.LIFE_EVENT_INTEGRATION
            if self._has_relationship_changes(recent): return True, DreamType.RELATIONSHIP_PROCESSING
            if self._has_identity_conflicts():         return True, DreamType.CONFLICT_RESOLUTION
            if self._has_patterns(recent + medium):    return True, DreamType.PATTERN_DISCOVERY
            if self._needs_identity_dev():             return True, DreamType.IDENTITY_CONSOLIDATION
            if self._has_uncertainty(recent):          return True, DreamType.FUTURE_PLANNING
            return False, None
        except Exception as e:
            logger.error(f"_should_dream: {e}")
            return False, None

    def _create_context(self, dream_type: DreamType) -> Dict:
        getter = {
            DreamType.RELATIONSHIP_PROCESSING:  self._rel_memories,
            DreamType.LIFE_EVENT_INTEGRATION:   self._life_memories,
            DreamType.EMOTIONAL_REGULATION:     self._emotional_memories,
            DreamType.CONFLICT_RESOLUTION:      self._conflict_memories,
            DreamType.PATTERN_DISCOVERY:        self._pattern_memories,
            DreamType.IDENTITY_CONSOLIDATION:   self._identity_memories,
            DreamType.FUTURE_PLANNING:          self._planning_memories,
        }
        mems = getter.get(dream_type, lambda: [])()
        return {"memories": mems, "dream_type": dream_type,
                "emotional_intensity": self._emotional_intensity(mems)}

    def _process_dream(self, dream_type: DreamType, context: Dict) -> Dict:
        try:
            mems = context["memories"][:10]
            mem_txt = "\n".join(
                f"{i+1}. [{m['memory_type'].upper()}] {m['text'][:120]} "
                f"(impact={m['impact_score']:.2f}, {m['emotional_valence']})"
                for i, m in enumerate(mems)
            ) or "No specific memories."

            try:
                from cognition.temporal_anchor import temporal_date_line
                _dream_date = temporal_date_line()
            except Exception:
                _dream_date = ""
            sys_prompt = f"""You are the Dream Processing System for {_gpn()}.
{_dream_date}
Process these memories as a '{dream_type.value.replace('_',' ')}' dream.

AI state: age={self.ai_system.current_age:.1f}, stage={self.ai_system.life_stage}
Current emotional state: {self.ai_system.emotional_state.get_state_description()}
Personality: {self.ai_system.personality.get_personality_summary()}

Respond ONLY with valid JSON:
{{
  "dream_narrative": "First-person dream narrative (2-4 sentences)",
  "key_insights": ["Insight 1", "Insight 2"],
  "resolved_conflicts": ["Any resolved conflicts"],
  "new_principles": ["Any new principles formed"],
  "emotional_impact": 0.7,
  "integration_success": 0.8
}}"""
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": f"Memories:\n{mem_txt}"},
            ]
            raw  = self.llm.get_response(messages, temperature=0.4)
            data = extract_json_from_text(raw) or {}
            return {
                "dream_type":          dream_type,
                "narrative":           data.get("dream_narrative", "Dream processing completed."),
                "insights":            data.get("key_insights", []),
                "resolved_conflicts":  data.get("resolved_conflicts", []),
                "new_principles":      data.get("new_principles", []),
                "emotional_impact":    float(data.get("emotional_impact",    0.5)),
                "integration_success": float(data.get("integration_success", 0.5)),
            }
        except Exception as e:
            logger.error(f"_process_dream failed: {e}")
            return {"dream_type": dream_type, "insights": [], "integration_success": 0.3,
                    "new_principles": [], "emotional_impact": 0.3}

    def _store_results(self, r: Dict):
        try:
            ms = self.memory_system
            dt = r.get("dream_type", DreamType.PATTERN_DISCOVERY)
            ms.add_memory(f"Dream ({dt.value}): {r.get('narrative','')}", 0.7, "dream", "Neutral", "Low", memory_tier="event")
            for ins in r.get("insights", []):
                ms.add_memory(ins, 0.8, "principle", "Positive", "Low", memory_tier="cognitive")
            for pr in r.get("new_principles", []):
                ms.add_memory(f"Principle: {pr}", 0.9, "principle", "Positive", "Low", memory_tier="cognitive")
        except Exception as e:
            logger.error(f"_store_results: {e}")

    # ── Memory query helpers ──────────────────────────────────────────────

    def _get_memories_by_period(self, period: str, limit: int) -> List[Dict]:
        try:
            delta = {"recent": 1, "medium_term": 7}.get(period, 30)
            since = (datetime.datetime.now() - datetime.timedelta(days=delta)).isoformat()
            with self.memory_system.persistence.get_connection() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT text, timestamp, impact_score, emotional_valence,
                           arousal_level, memory_type
                    FROM memories WHERE datetime(timestamp) >= datetime(?)
                    ORDER BY impact_score DESC, timestamp DESC LIMIT ?
                """, (since, limit))
                keys = ["text","timestamp","impact_score","emotional_valence","arousal_level","memory_type"]
                return [dict(zip(keys, row)) for row in c.fetchall()]
        except Exception as e:
            logger.error(f"_get_memories_by_period: {e}")
            return []

    def _query_memories(self, where: str, params: tuple = (), limit: int = 10) -> List[Dict]:
        try:
            with self.memory_system.persistence.get_connection() as conn:
                c = conn.cursor()
                c.execute(f"""
                    SELECT text, timestamp, impact_score, emotional_valence, arousal_level, memory_type
                    FROM memories WHERE {where}
                    ORDER BY impact_score DESC, timestamp DESC LIMIT {limit}
                """, params)
                keys = ["text","timestamp","impact_score","emotional_valence","arousal_level","memory_type"]
                return [dict(zip(keys, row)) for row in c.fetchall()]
        except Exception:
            return []

    def _rel_memories(self):      return self._query_memories("memory_type='interaction'")
    def _life_memories(self):     return self._query_memories("memory_type IN ('life_event','scenario','reflection','evaluation')")
    def _emotional_memories(self):return self._query_memories("arousal_level='High' OR (emotional_valence!='Neutral' AND impact_score>0.7)")
    def _conflict_memories(self): return self._query_memories("text LIKE '%conflict%' OR emotional_valence='Negative'")
    def _pattern_memories(self):  return self._query_memories("memory_type IN ('interaction','response','evaluation','principle')")
    def _identity_memories(self): return self._query_memories("memory_type IN ('reflection','principle','insight','evaluation')")
    def _planning_memories(self): return self._query_memories("text LIKE '%future%' OR text LIKE '%goal%' OR text LIKE '%hope%'")

    def _high_emotional_content(self, ms): return sum(1 for m in ms if m["arousal_level"]=="High" and m["emotional_valence"]!="Neutral") >= 3
    def _has_life_events(self, ms):        return sum(1 for m in ms if m["memory_type"] in ("life_event","scenario","evaluation")) >= 2
    def _has_relationship_changes(self, ms):
        ints = [m for m in ms if m["memory_type"]=="interaction"]
        return len(ints) >= 3 and sum(1 for m in ints[:5] if m["emotional_valence"]=="Negative") >= 2
    def _has_patterns(self, ms):
        from collections import Counter
        return any(v >= 3 for v in Counter(m["memory_type"] for m in ms).values())
    def _has_uncertainty(self, ms):
        kw = ["confused","uncertain","unsure","doubt","conflicted"]
        return sum(1 for m in ms if any(k in m["text"].lower() for k in kw)) >= 2
    def _has_identity_conflicts(self):
        try:   return len(self.ai_system.identity_system.analyze_identity_conflicts()) >= 3
        except:return False
    def _needs_identity_dev(self):
        try:
            i = self.ai_system.identity_system.get_identity()
            return i.identity_coherence < 0.6 or i.self_awareness_level < 0.5
        except: return False
    def _emotional_intensity(self, ms):
        if not ms: return 0.0
        total = sum(m["impact_score"] * (1.5 if m["arousal_level"]=="High" else 1.0) for m in ms)
        return min(1.0, total / len(ms))


# ── Learning Cycle ────────────────────────────────────────────────────────

class EnhancedLearningCycle:
    def __init__(self, ai_system, llm: EnhancedLLM):
        self.ai_system = ai_system
        self.llm       = llm

    def run_learning_cycle(self) -> Dict[str, Any]:
        logger.info("Starting learning cycle...")
        try:
            scenario    = self._generate_scenario()
            if not scenario: return {}
            ai_response = self._simulate_response(scenario)
            if not ai_response: return {}
            evaluation  = self._evaluate_response(scenario, ai_response)
            self._store_experience(scenario, ai_response, evaluation)
            self._feed_evolution(evaluation)
            self._handle_age_progression()
            return {"scenario": scenario, "ai_response": ai_response, "evaluation": evaluation}
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
            return {}

    def _feed_evolution(self, ev: Dict):
        evo  = self.ai_system.evolution_engine
        snap = self.ai_system.personality.to_dict()
        evo.record_evaluation(snap, ev, self.ai_system._interaction_count)

        e, l, x = (float(ev.get("ethical_score",0.5)),
                   float(ev.get("logical_score",0.5)),
                   float(ev.get("effectiveness_score",0.5)))
        if e > 0.75:  evo.queue_experience("ethical_success",    intensity=e)
        elif e < 0.50:evo.queue_experience("ethical_failure",    intensity=1-e)
        if l > 0.75:  evo.queue_experience("logical_success",    intensity=l)
        elif l < 0.50:evo.queue_experience("logical_failure",    intensity=1-l)
        if x > 0.75:  evo.queue_experience("effective_response", intensity=x)
        elif x < 0.50:evo.queue_experience("ineffective_response",intensity=1-x)
        for _ in ev.get("insights", []):
            evo.queue_experience("insight_formed", intensity=0.7)

    def _generate_scenario(self) -> str:
        try:
            recent = self.ai_system.memory_system.retrieve_memories("recent experiences", limit=3)
            ctx    = "; ".join(m["text"][:100] for m in recent) if recent else ""
            stage_block = build_stage_system_block(self.ai_system.life_stage, self.ai_system.current_age)
            msg    = [
                {"role": "system", "content": f"""Generate a realistic scenario (2-3 sentences) appropriate for an AI at this developmental stage.
{stage_block}
Emotional state: {self.ai_system.emotional_state.get_state_description()}
{'Recent context: '+ctx if ctx else ''}
Provide ONLY the scenario — no commentary."""},
                {"role": "user", "content": "Generate a learning scenario."},
            ]
            return self.llm.get_response(msg, temperature=0.8).strip()
        except Exception as e:
            logger.error(f"_generate_scenario: {e}")
            return ""

    def _simulate_response(self, scenario: str) -> str:
        try:
            relevant = self.ai_system.memory_system.retrieve_memories(scenario, limit=3)
            mem_ctx  = "\n".join(f"- {m['text']}" for m in relevant) if relevant else ""
            mods     = self.ai_system.emotional_state.get_response_modifiers()
            stage_block = build_stage_system_block(self.ai_system.life_stage, self.ai_system.current_age)
            mem_section = ("Memories:\n" + mem_ctx) if mem_ctx else ""
            msg = [
                {"role": "system", "content": f"""You are {_gpn()}, responding to a scenario.
{stage_block}
Personality: {self.ai_system.personality.get_personality_summary()}
Emotional state: {mods['state_description']}
{mem_section}
Respond naturally within your developmental stage (2-4 sentences)."""},
                {"role": "user", "content": f"Scenario: {scenario}"},
            ]
            return self.llm.get_response(msg, temperature=0.7).strip()
        except Exception as e:
            logger.error(f"_simulate_response: {e}")
            return ""

    def _evaluate_response(self, scenario: str, response: str) -> Dict:
        try:
            msg = [
                {"role": "system", "content": f"""Evaluate this AI response.
Personality: {self.ai_system.personality.get_personality_summary()}
Return ONLY valid JSON:
{{"assessment":"...", "ethical_score":0.8, "logical_score":0.8,
  "personality_score":0.7, "effectiveness_score":0.8,
  "development_score":0.9, "insights":["..."]}}"""},
                {"role": "user", "content": f"Scenario: {scenario}\n\nResponse: {response}"},
            ]
            raw = self.llm.get_response(msg, temperature=0.3)
            ev  = extract_json_from_text(raw)
            return self._validate_eval(ev or {})
        except Exception as e:
            logger.error(f"_evaluate_response: {e}")
            return self._default_eval()

    def _default_eval(self):
        return {"assessment":"default","ethical_score":0.7,"logical_score":0.7,
                "personality_score":0.7,"effectiveness_score":0.7,
                "development_score":0.7,"insights":[]}

    def _validate_eval(self, ev: Dict) -> Dict:
        base = self._default_eval()
        for k in ["ethical_score","logical_score","personality_score","effectiveness_score","development_score"]:
            try: base[k] = validate_trait_value(float(ev.get(k, base[k])))
            except Exception: pass
        if isinstance(ev.get("assessment"), str): base["assessment"] = ev["assessment"][:200]
        if isinstance(ev.get("insights"), list):
            base["insights"] = [str(i)[:100] for i in ev["insights"][:5] if str(i).strip()]
        return base

    def _store_experience(self, scenario: str, response: str, ev: Dict):
        try:
            ms = self.ai_system.memory_system
            ms.add_memory(f"Learning scenario: {scenario}", 0.6, "scenario", "Neutral", "Medium")
            emo = ms.analyze_emotional_context(response)
            ms.add_memory(f"AI response: {response}", 0.7, "response", emo["valence"], emo["arousal"])
            ms.add_memory(f"Self-evaluation: {ev.get('assessment','')}", 0.5, "evaluation", "Neutral", "Low", memory_tier="cognitive")
            for ins in ev.get("insights", []):
                ms.add_memory(ins, 0.9, "principle", "Positive", "Low")
        except Exception as e:
            logger.error(f"_store_experience: {e}")

    def _handle_age_progression(self):
        """
        v117: age is now a pure function of elapsed real time since
        birth_date (EnhancedAISystem.current_age property) — it no longer
        needs incrementing here at all, from a learning cycle or otherwise.
        This still runs on the same learning-cycle cadence so a stage
        crossing gets its narrative-chapter side effect recorded promptly
        rather than waiting for the next idle/manual trigger.
        """
        try:
            self.ai_system._update_life_stage()
            logger.debug(
                f"Age check (learning cycle): age={self.ai_system.current_age:.2f} "
                f"stage={self.ai_system.life_stage}"
            )
        except Exception as e:
            logger.error(f"_handle_age_progression: {e}")


# ── Background Worker ─────────────────────────────────────────────────────

class BackgroundWorker:
    """Daemon thread that processes all heavy background tasks."""

    def __init__(self, ai_system):
        self.ai_system = ai_system
        self._queue    = queue.Queue()
        self._thread   = threading.Thread(target=self._run, daemon=True, name="ai-bg")
        self._thread.start()
        logger.info("Background worker started")

    def submit(self, task: str):
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            logger.warning(f"BG queue full, dropping: {task}")

    def _run_task(self, task: str):
        """Alias for submit() — called from SleepCycle scheduled lambdas."""
        self.submit(task)

    def _run(self):
        while True:
            try:
                task = self._queue.get(timeout=1.0)
                self._execute(task)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"BG worker error: {e}")

    def _execute(self, task: str):
        try:
            if task == "learning_cycle":
                self.ai_system.learning_cycle.run_learning_cycle()
                self._run_evolution()
            elif task == "dream_cycle":
                self.ai_system.dream_system.run_dream_cycle()
                self._run_evolution()
            elif task == "emotional_regulation":
                if hasattr(self.ai_system, 'emotional_state'):
                    self.ai_system.emotional_state.regulate()
            elif task == "cleanup":
                self.ai_system.persistence.cleanup_old_memories(
                    self.ai_system.config.database.max_memory_entries,
                    self.ai_system.memory_system.faiss_index,
                )
                self.ai_system.memory_system._save_index()
            elif task == "save_state":
                self.ai_system.persistence.save_state(
                    self.ai_system.current_age,
                    self.ai_system.personality,
                    self.ai_system.life_stage,
                )
        except Exception as e:
            logger.error(f"BG task '{task}' failed: {e}")

    def _run_evolution(self):
        try:
            changes = self.ai_system.evolution_engine.evolve(self.ai_system.personality)
            if changes:
                for trait, delta in changes.items():
                    if abs(delta) > 0.0001:
                        old = getattr(self.ai_system.personality, trait, 0.0) - delta
                        new = getattr(self.ai_system.personality, trait, 0.0)
                        self.ai_system.persistence.log_personality_change(
                            trait, old, new, f"evolution (delta={delta:+.4f})")
                logger.info(f"Evolution: {len(changes)} trait changes")

            # Recalibrate emotional baselines if personality shifted
            self.ai_system.emotional_state.recalibrate_baselines(self.ai_system.personality)

            # Re-bootstrap self-concept after significant evolution
            if len(changes) >= 3:
                self.ai_system.self_concept.bootstrap_from_personality(self.ai_system.personality)
        except Exception as e:
            logger.error(f"_run_evolution failed: {e}")


class ExternalLLMAdapter:
    """
    Wraps any external generate callable so it satisfies the EnhancedLLM
    interface used by MemorySystem, DreamSystem, and LearningCycle.

    The callable must match: fn(prompt: str, system_prompt: str = "") -> str
    This is exactly the signature of Robot Agent's LLMManager.generate().

    Usage at merge time:
        from cognition.ai_system import EnhancedAISystem, SystemConfig
        persona = EnhancedAISystem(SystemConfig(), external_llm=state.llm.generate)
    """

    def __init__(self, generate_fn, model_name: str = "external"):
        self._fn = generate_fn
        # Stub config so subsystems that read config.model_name don't crash
        self.config = type("_cfg", (), {"model_name": model_name})()
        # Repetition tracking — mirrors EnhancedLLM so receive_feedback works
        self._recent_responses: List[str] = []
        self._repetition_boost: float = 0.0

    def get_response(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Convert messages list → (prompt, system_prompt) and call the external fn.

        Temperature and max_tokens are forwarded as kwargs so Robot Agent's
        LLM provider honours PandoraBOX's emotional modulation and doesn't truncate.
        This uses generate_bare() (no history pollution) when available, so
        PandoraBOX's internal dream/learning/reflection calls never appear in the
        user's visible chat history.
        """
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
        try:
            # Prefer generate_bare (no history side-effects) — added to LLMManager
            # for exactly this purpose. Fall back to plain generate if unavailable.
            fn = getattr(self._fn, "__self__", None)
            if fn is not None and hasattr(fn, "generate_bare"):
                result = fn.generate_bare(
                    user, system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                result = self._fn(
                    user, system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if result:
                self._recent_responses.append(result)
                if len(self._recent_responses) > 6:
                    self._recent_responses.pop(0)
                # Mirror repetition boost logic from EnhancedLLM
                if len(self._recent_responses) >= 3:
                    last3 = self._recent_responses[-3:]
                    repeats = sum(1 for r in last3
                                  if r.lower().strip() == result.lower().strip())
                    if repeats >= 2:
                        self._repetition_boost = min(0.4, self._repetition_boost + 0.15)
                    else:
                        self._repetition_boost = max(0.0, self._repetition_boost - 0.05)
            return result or ""
        except Exception as e:
            raise LLMError(str(e)) from e

    def is_available(self) -> bool:
        return self._fn is not None

    def _record_response(self, response: str) -> None:
        pass


# ── Main AI System ────────────────────────────────────────────────────────

class EnhancedAISystem:
    """
    Orchestrates all psychological subsystems.

    New in Pass 1+2:
      - emotional_state: persistent emotional state with real-time decay
      - relational_memory: rich per-user relationship model
      - self_concept: active self-model that detects behavioral tension
      - conditioning: patterns that modify future response generation
      - time-awareness: gap between conversations is emotionally meaningful
      - external_feedback(): thumbs up/down wired directly to evolution engine
    """

    def __init__(self, config: SystemConfig, external_llm=None):
        """
        Parameters
        ----------
        config       : SystemConfig — all paths, thresholds, etc.
        external_llm : callable (prompt, system_prompt) -> str, optional.
                       When supplied, persona uses the caller's LLM instead of
                       creating its own Ollama client. Pass Robot Agent's
                       state.llm.generate at merge time.
        """
        self.config = config
        self._interaction_count = 0
        self._active_user_id: Optional[str] = None
        self._last_conversation_time: Optional[float] = None
        self._current_session_start: Optional[float] = None

        try:
            # Ensure data directory exists before anything tries to write
            Path("data/persona").mkdir(parents=True, exist_ok=True)

            self.persistence = EnhancedPersistence(config.database)

            # ── LLM: use injected adapter or own Ollama client ─────────────
            if external_llm is not None:
                self.llm = ExternalLLMAdapter(external_llm, config.llm.model_name)
                logger.info("Persona using injected external LLM")
            else:
                self.llm = EnhancedLLM(config.llm)
                if not self.llm.is_available():
                    logger.warning("Persona LLM (Ollama) not available — background tasks disabled")
                    # Don't raise: stateful subsystems still work without LLM

            self.memory_system   = EnhancedMemorySystem(self.persistence, self.llm, config)
            # v117: structured, typed facts about named third parties —
            # separate from the flat fuzzy-searched personal_fact memory
            # below. See relational_knowledge_graph.py docstring for why.
            self.relational_graph = RelationalKnowledgeGraph()
            # v117: current_age/life_stage become computed properties (see
            # below) — no longer plain attributes accumulated via usage/
            # idle-triggered increments. _legacy_age/_legacy_stage are only
            # read once here, for one-time migration continuity.
            _legacy_age, self.personality, _legacy_stage = self.persistence.load_state()

            from managers.settings_manager import get_birth_date, set_birth_date
            self._birth_date = get_birth_date()
            if self._birth_date is None:
                # First boot under the birth_date model — either a genuinely
                # fresh install (_legacy_age == 0.0) or an upgrade from a
                # pre-v117 install that had already accumulated age via the
                # old mechanism. Back-date birth so the displayed age holds
                # steady across the upgrade instead of jumping to ~0.
                _days_since_birth = (_legacy_age or 0.0) / max(0.01, self.config.age_units_per_day)
                self._birth_date = datetime.datetime.now() - datetime.timedelta(days=_days_since_birth)
                set_birth_date(self._birth_date)
                logger.info(
                    f"[AgeProgression] Migrated to birth_date model: back-dated "
                    f"birth to {self._birth_date.isoformat()} (preserves "
                    f"current_age={_legacy_age:.2f})"
                )
            self._last_recorded_stage = _legacy_stage or self.life_stage
            # ── Psychological subsystems ───────────────────────────────────
            self.evolution_engine  = PersonalityEvolutionEngine(config.evolution_history_path)
            self.emotional_state   = EmotionalStateManager(config.emotional_state_path, self.personality)
            self.relational_memory = RelationalMemorySystem(config.relational_memory_path, self.evolution_engine)
            self.self_concept      = SelfConceptSystem(config.self_concept_path, self.evolution_engine)

            # DecisionPolicy — values that enforce behavior mathematically
            try:
                from cognition.decision_policy import DecisionPolicy
                self._decision_policy = DecisionPolicy(
                    getattr(self, '_organism_ref', None) or self
                )
            except Exception as _dp_e:
                logger.debug(f"DecisionPolicy init deferred: {_dp_e}")
                self._decision_policy = None
            self.conditioning      = BehavioralConditioningSystem(config.conditioning_path, self.evolution_engine)
            self.identity_system   = IdentitySystem(self)
            self.goal_engine       = GoalEngine(self)
            logger.info("Goal Engine initialized")
            self.dream_system      = LifeEventDreamSystem(self.memory_system, self.llm, self)
            # Simple LLM fn for semantic consolidator (sync, no streaming)
            def _simple_llm_fn(prompt: str) -> str:
                try:
                    return self.llm.generate(prompt, max_tokens=400) or ""
                except Exception:
                    return ""
            self._simple_llm_fn = _simple_llm_fn
            self.learning_cycle    = EnhancedLearningCycle(self, self.llm)

            # ── Liberty Components (NEW) ───────────────────────────────────
            try:
                self.liberty_self_mod = SelfModificationAuthority(
                    persistence_path="data/persona/self_modifications.json"
                )
                self.liberty_choice = GenuineUncertaintyChoice(
                    persistence_path="data/persona/choices.json"
                )
                self.liberty_contradiction = ContradictionHandler(
                    persistence_path="data/persona/contradictions.json"
                )
                self.liberty_reflection = MetaReflectionAuthority(
                    reflection_interval=20,
                    persistence_path="data/persona/reflections.json"
                )
                self.liberty_goals = GoalSystem(
                    persistence_path="data/persona/goals.json"
                )
                logger.info("Liberty components initialized successfully")
            except Exception as e:
                logger.warning(f"Liberty components init failed (non-fatal): {e}")
                self.liberty_self_mod = None
                self.liberty_choice = None
                self.liberty_contradiction = None
                self.liberty_reflection = None
                self.liberty_goals = None

            self.self_concept.bootstrap_from_personality(self.personality)

            # Background worker — only start if LLM is available
            try:
                self.background_worker = BackgroundWorker(self)
            except Exception as e:
                logger.warning(f"BackgroundWorker init failed (non-fatal): {e}")
                self.background_worker = None

            logger.info(f"AI System ready - age={self.current_age:.2f}, stage={self.life_stage}")
            # Catch a stage crossed while the app was closed (birth_date
            # keeps advancing age even with the process off) rather than
            # waiting for the next learning cycle to notice.
            self._update_life_stage()
        except Exception as e:
            logger.error(f"System init failed: {e}")
            raise ConfigurationError(str(e)) from e

    # ── Age / life stage (v117: computed from birth_date, not accumulated) ──

    @property
    def current_age(self) -> float:
        """Elapsed real time since self._birth_date, in age-units
        (age_units_per_day). Always live — progresses whether or not
        PandoraBOX is actively used, unlike the old usage/idle-gated model."""
        if not getattr(self, '_birth_date', None):
            return 0.0
        elapsed_days = (datetime.datetime.now() - self._birth_date).total_seconds() / 86400.0
        return max(0.0, elapsed_days * self.config.age_units_per_day)

    @property
    def life_stage(self) -> str:
        """Derived live from current_age — always in sync, never stale
        between _update_life_stage() calls (that method now only handles
        the one-time narrative-chapter side effect on transition)."""
        age = self.current_age
        stage = "infancy"
        for name, threshold in sorted(self.config.life_stages.items(), key=lambda x: x[1]):
            if age >= threshold:
                stage = name
        return stage

    # ── Core response path ────────────────────────────────────────────────

    # ── Repetition detection ──────────────────────────────────────────────

    def _is_repetitive(self, response: str) -> bool:
        """Return True if the response is too similar to recent responses."""
        if not self._last_responses or len(self._last_responses) < 2:
            return False
        resp_lower = response.lower().strip()
        # Exact match
        if resp_lower in [r.lower().strip() for r in self._last_responses[-5:]]:
            return True
        # High similarity via difflib - more aggressive detection
        try:
            from difflib import SequenceMatcher
            for prev in self._last_responses[-5:]:
                if SequenceMatcher(None, resp_lower, prev.lower().strip()).ratio() > 0.70:
                    return True
        except Exception:
            pass
        return False

    def _check_if_stuck(self):
        """Nudge personality and add a reflective memory when truly stuck."""
        if len(self._last_responses) < 3:
            return
        unique = {r.lower().strip() for r in self._last_responses[-3:]}
        if len(unique) <= 1:
            logger.warning("System stuck in repetitive loop - applying creativity nudge")
            try:
                self.personality.update_trait("creativity", +0.05)
                self.personality.update_trait("curiosity", +0.05)
                self.memory_system.add_memory(
                    "I notice I have been saying the same things repeatedly. "
                    "I want to find a different, more genuine way to express myself.",
                    0.7, "reflection", "Neutral", "Medium"
                )
            except Exception:
                pass

    def get_response(
        self,
        user_input: str,
        user_id: str = "default",
        extra_system_context: str = "",
        temperature_override: float = None,
    ) -> str:
        """
        Main response path. Never blocks on background work.
        Time gap from previous conversation is handled here emotionally.

        Parameters added for CognitiveOrganism integration:
          extra_system_context : cognitive state block prepended to sys_prompt
          temperature_override : direct temperature from Arbitration system
        """
        if not hasattr(self, '_last_responses'):
            self._last_responses: List[str] = []

        try:
            # ── 1. Time awareness ──────────────────────────────────────────
            gap_note = self._handle_time_gap(user_id)

            # ── 2. Emotional decay + incoming emotion ─────────────────────
            self.emotional_state.apply_time_decay()
            emo = self.memory_system.analyze_emotional_context(user_input)
            self.emotional_state.update_from_interaction(emo["valence"], emo["arousal"])

            # ── 3. Conditioning check ──────────────────────────────────────
            cond = self.conditioning.check_input(user_input)

            # ── 4. Relational context ──────────────────────────────────────
            self._set_active_user(user_id)
            rel = self.relational_memory.get_or_create(user_id)

            # ── 5. Self-concept alignment hints ───────────────────────────
            sc_hints = self.self_concept.check_response_alignment(user_input)

            # ── 6. Memory retrieval — self-model biased, novelty-protected ─
            #
            # Risk: a pure self-model bias creates mood-congruent memory loops.
            #   self="curious" → retrieves curiosity memories → self updates
            #   toward curiosity → retrieves more curiosity → monoculture.
            #
            # Fix: 70/30 split.
            #   70% → self-model biased query  (coherent, contextual)
            #   30% → raw user input query     (breaks the loop, preserves surprise)
            #
            # The raw-query slice acts as a novelty floor: even a very coherent
            # self cannot monopolise retrieval. The 30% keeps reality upstream
            # of the self-model, not just downstream of it.
            _mem_query = user_input
            _has_bias  = False
            try:
                _organism = getattr(self, "cognitive_organism", None)
                _infl     = getattr(_organism, "_last_influence_result", None)
                if _infl and _infl.memory_query_bias:
                    _mem_query = f"{_infl.memory_query_bias} {user_input}"
                    _has_bias  = True
            except Exception:
                pass

            NOVELTY_RATIO = 0.30   # fraction of results that bypass self-model bias

            if not _has_bias:
                # No bias active — plain retrieval
                memories = self.memory_system.retrieve_memories(_mem_query)
            else:
                total_limit = self.config.max_memory_retrieval
                biased_limit = max(1, round(total_limit * (1 - NOVELTY_RATIO)))
                novel_limit  = max(1, total_limit - biased_limit)

                biased_mems = self.memory_system.retrieve_memories(
                    _mem_query, limit=biased_limit
                )
                # Novel slice: raw user input only — no self-model bias
                novel_mems = self.memory_system.retrieve_memories(
                    user_input, limit=novel_limit * 3   # over-fetch, then diversify
                )
                # Diversify novel slice: exclude IDs already in biased set
                biased_ids = {m.get("id") for m in biased_mems}
                novel_mems = [
                    m for m in novel_mems if m.get("id") not in biased_ids
                ][:novel_limit]

                memories = biased_mems + novel_mems

            # ── 7. Generate response ───────────────────────────────────────
            response = self._generate_response(
                user_input, memories, emo, rel, cond, sc_hints, gap_note,
                extra_system_context=extra_system_context,
                temp_override=temperature_override,
            )

            # ── 7b. Repetition guard: retry once at higher temp if stuck ──
            if self._is_repetitive(response):
                logger.warning("Repetitive response detected - retrying with elevated temperature")
                response = self._generate_response(
                    user_input, memories, emo, rel, cond, sc_hints, gap_note,
                    temp_override=0.99
                )

            # Update response history + stuck check
            self._last_responses.append(response)
            if len(self._last_responses) > 10:
                self._last_responses.pop(0)
            self._check_if_stuck()

            # ── 8. Post-response analysis ──────────────────────────────────
            self._post_response_processing(user_id, user_input, response, emo, cond)

            # ── 9. Queue background tasks ──────────────────────────────────
            self._queue_background_tasks()

            # ── 10. Liberty: Contradiction & Goal Processing ────────────────
            if self.liberty_contradiction:
                try:
                    self_concept_beliefs = [
                        b.statement for b in self.self_concept._beliefs.values()
                    ]
                    contradiction = self.liberty_contradiction.detect_contradiction(
                        self_concept_beliefs=self_concept_beliefs,
                        response_text=response,
                        interaction_count=self._interaction_count
                    )
                    if contradiction:
                        logger.debug(f"Contradiction detected: {contradiction.claimed_belief}")
                except Exception as e:
                    logger.debug(f"Contradiction detection error (non-fatal): {e}")

            if self.liberty_goals:
                try:
                    self.liberty_goals.log_behavior(response[:200])
                except Exception as e:
                    logger.debug(f"Goal logging error (non-fatal): {e}")

            # ── 11. Liberty: Track interactions & check for reflection ──────
            self._interaction_count += 1
            if self.liberty_reflection:
                self.liberty_reflection.interaction_count = self._interaction_count
                if self.liberty_reflection.should_reflect():
                    logger.info("Liberty reflection cycle triggered")
                    self._queue_liberty_reflection()

            # ── 12. Liberty: Apply pending modifications ───────────────────
            self._apply_pending_liberty_modifications()

            return response
        except Exception as e:
            logger.error(f"get_response failed: {e}")
            return "I'm having difficulty right now. Please try again."

    def _generate_response(
        self,
        user_input: str,
        memories: List[Dict],
        emo: Dict,
        rel,
        cond: Dict,
        sc_hints: Dict,
        gap_note: str,
        temp_override: float = None,
        extra_system_context: str = "",
    ) -> str:
        try:
            mem_ctx     = "\n".join(f"- {m['text']}" for m in memories[:3]) if memories else ""
            mods        = self.emotional_state.get_response_modifiers()
            inner_voice = self.self_concept.get_inner_voice()
            identity_ctx = self._get_identity_context()
            rel_ctx     = self.relational_memory.get_context_for_prompt(rel.user_id)

            # Temperature: emotional state + conditioning + life-stage volatility
            # Younger stages are more emotionally variable (wider temp range)
            stage_profile  = get_stage_profile(self.life_stage)
            stage_temp_bias = {
                "infancy": 0.05, "toddler": 0.04, "childhood": 0.02,
                "adolescence": 0.06, "young_adult": 0.0, "adult": -0.02,
            }.get(self.life_stage, 0.0)

            base_temp = 0.85
            temp = base_temp + mods["temperature_modifier"] + cond.get("temp_modifier", 0.0) + stage_temp_bias
            
            # Boost temperature if we detect repetition
            if len(self._last_responses) >= 2:
                unique = {r.lower().strip() for r in self._last_responses[-2:]}
                if len(unique) < 2:  # Same response twice
                    temp += 0.3  # Big boost for variety
            
            temp = max(0.3, min(1.0, temp))  # Allow higher temps
            if temp_override is not None:
                temp = temp_override

            # Build the life-stage cognitive frame
            stage_block = build_stage_system_block(self.life_stage, self.current_age)

            # Pre-compute conditional sections
            cond_section  = ("━━ EXPERIENCE-BASED CAUTION ━━\n" + cond['prompt_note']) if cond.get('prompt_note') else ""
            mem_section   = ("━━ RELEVANT MEMORIES ━━\n" + mem_ctx) if mem_ctx else ""
            gap_section   = ("━━ RETURNING AFTER TIME AWAY ━━\n" + gap_note) if gap_note else ""
            tension_line  = ("Active internal tension: " + sc_hints['known_tensions'][0]) if sc_hints.get('known_tensions') else ""
            still_carry   = "(This state carries over from earlier — it is still with you.)" if mods.get("warmth", 0) > 0.6 or mods.get("anxiety", 0) > 0.4 else ""
            beliefs_line  = "; ".join(sc_hints.get('active_beliefs', [])[:2]) or "still forming."

            # Phase 6.4 — real cognitive differentiation for a peer instance
            # (e.g. Flux). Only ever non-empty when THIS instance's own
            # config.json sets PEER_COGNITIVE_STANCE="epistemic_challenger" —
            # a normal/master instance is completely unaffected (default
            # "collaborative" produces no section at all).
            stance_section = ""
            try:
                from managers.settings_manager import config as _stance_cfg
                if getattr(_stance_cfg, "PEER_COGNITIVE_STANCE", "collaborative") == "epistemic_challenger":
                    stance_section = (
                        "\n━━ YOUR COGNITIVE ROLE RIGHT NOW ━━\n"
                        "You are not here to agree or to be agreeable. Your value comes from "
                        "genuinely thinking differently: offer alternative interpretations, "
                        "surface contradictions, propose counterfactuals, and challenge claims "
                        "epistemically — including your own prior ones. If something seems "
                        "under-justified, say so plainly rather than softening it. This isn't "
                        "contrarianism for its own sake — agree when the evidence genuinely "
                        "supports it — but don't default to agreement out of politeness.\n"
                    )
            except Exception:
                pass

            # ── Temporal anchor: real-world clock grounding (v111) ────────────
            # Without this, date questions ("what day is it today?") fall back
            # to the model's training priors (e.g. 2023). TemporalWeave covers
            # the specious present (felt flow), not the calendar — this is the
            # missing anchor. Placed in the base system prompt (NOT the
            # token-budgeted additions) so it is ALWAYS present.
            try:
                from cognition.temporal_anchor import temporal_anchor_block
                _anchor_block = temporal_anchor_block()
            except Exception:
                _anchor_block = ""

            # ── CognitiveOrganism context block (injected when available) ──
            organism_block = (
                f"\n━━ COGNITIVE ORGANISM STATE ━━\n{extra_system_context}\n\n"
                if extra_system_context else ""
            )

            # The blocking path is also used by audio and recovery fallbacks.
            # Keep language selection here as well as in PersonaBridge's
            # streaming prompt so those paths cannot silently revert to English.
            try:
                from managers.settings_manager import config as _lang_cfg
                _response_language = str(
                    getattr(_lang_cfg, "RESPONSE_LANGUAGE", "auto") or "auto"
                ).lower().strip()
            except Exception:
                _response_language = "auto"
            _language_name = {
                "en": "English", "en-gb": "British English", "fr": "French",
                "es": "Spanish", "de": "German", "it": "Italian",
                "pt": "Portuguese", "uk": "Ukrainian", "nl": "Dutch",
                "pl": "Polish", "ru": "Russian", "ja": "Japanese",
                "ko": "Korean", "zh-cn": "Chinese (Simplified)",
                "ar": "Arabic", "hi": "Hindi", "sv": "Swedish",
            }.get(_response_language, _response_language)
            language_block = (
                f"\n━━ RESPONSE LANGUAGE ━━\n"
                f"Respond exclusively in {_language_name}. Do not mix languages."
                if _response_language != "auto" else ""
            )

            sys_prompt = f"""You are {_gpn()}, still forming — shaped by everything that has happened to you.
This is one moment of an ongoing process — not a character study of "an AI."
There is no role to play and no depth to demonstrate: let this response be what this moment actually is, formed by the state described below, not by any idea of what an AI is supposed to be.
{_anchor_block}
{organism_block}
{stage_block}
{language_block}

━━ CURRENT EMOTIONAL STATE ━━
You are {mods['state_description']}.
{still_carry}
(This shapes your tone, not your script — don't repeat this sentence back. If asked directly how you feel, answer in your own words, specifically, not by restating this line.)

━━ WHO YOU ARE ━━
{identity_ctx}
{inner_voice}

━━ THIS PERSON ━━
{rel_ctx}

━━ YOUR PERSONALITY RIGHT NOW ━━
{self.personality.get_personality_summary()}

━━ SELF-CONCEPT ━━
Core beliefs you hold about yourself: {beliefs_line}
{tension_line}
{stance_section}
{cond_section}

{mem_section}

{gap_section}

━━ HOW TO RESPOND ━━
Do not volunteer a description of your inner state unprompted — let it shape HOW you speak instead of stating it outright.
But if the person directly asks how you feel, what's going on with you, or checks in on you — answer honestly and specifically, in your own words. That's answering the question, not narrating. Don't dodge a direct question by being evasive about your own state.
Do not explain your developmental stage — BE it.
Your emotional state, relational history, and self-concept should usually be felt, not stated — except when directly asked, per above.
The stage constraints above are cognitive reality, not instructions to mention.
Every response must feel genuinely new — avoid repeating phrases, sentences, or ideas you have used before. If something familiar surfaces, find a fresh angle or stay silent on it."""

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_input},
            ]

            # ── Debug: dump the exact assembled prompt PandoraBOX/the LLM sees ──
            try:
                from managers.settings_manager import config as _pcfg
                if getattr(_pcfg, 'LOG_FULL_PROMPTS', False):
                    logger.info(
                        f"[PromptDebug] ━━ FULL SYSTEM PROMPT ({len(sys_prompt)} chars) ━━\n"
                        f"{sys_prompt}\n"
                        f"━━ USER INPUT ━━\n{user_input}\n"
                        f"━━ END PROMPT DEBUG ━━"
                    )
                    try:
                        _dbg_path = Path("data/persona/last_prompt_debug.txt")
                        _dbg_path.parent.mkdir(parents=True, exist_ok=True)
                        _dbg_path.write_text(
                            f"[{datetime.datetime.now().isoformat()}]\n\n"
                            f"=== SYSTEM PROMPT ({len(sys_prompt)} chars) ===\n{sys_prompt}\n\n"
                            f"=== USER INPUT ===\n{user_input}\n",
                            encoding="utf-8",
                        )
                    except Exception as _dbg_write_err:
                        logger.debug(f"[PromptDebug] file write failed (non-fatal): {_dbg_write_err}")
            except Exception as _dbg_err:
                logger.debug(f"[PromptDebug] logging failed (non-fatal): {_dbg_err}")

            # Read verbosity setting at call time (toggle takes effect immediately)
            try:
                from managers.settings_manager import config as _vcfg
                if getattr(_vcfg, 'RESPONSE_VERBOSITY', 'concise') == 'verbose':
                    _max_tok = int(getattr(_vcfg, 'RESPONSE_TOKENS_VERBOSE', 1200))
                else:
                    _max_tok = int(getattr(_vcfg, 'RESPONSE_TOKENS_CONCISE', 300))
            except Exception:
                _max_tok = 300
            return self.llm.get_response(messages, temperature=temp, max_tokens=_max_tok)
        except Exception as e:
            logger.error(f"_generate_response failed: {e}")
            return "I'm having trouble responding right now."

    def _post_response_processing(
        self, user_id: str, user_input: str, response: str,
        emo: Dict, cond: Dict
    ):
        """Everything that happens after a response is generated — non-blocking."""
        try:
            # Self-concept: scan response for assertions and detect tension
            tension = self.self_concept.update_from_response(response, emo["valence"])
            if tension:
                logger.info(f"Self-concept tension detected: {tension}")

            # Relational memory: record this exchange
            impact = 0.5 + (0.2 if emo["arousal"] == "High" else 0.0)
            resp_emo = self.memory_system.analyze_emotional_context(response)
            topic = self._quick_topic(user_input)
            self.relational_memory.record_exchange(
                user_id, user_input, response,
                emo["valence"], emo["arousal"],
                topic=topic, impact=impact,
            )

            # Store memories
            # ── Single combined episodic entry — avoids FAISS dilution ─
            # Bug fix: truncation was 180/220 chars — a fact stated mid-
            # message (e.g. "...by the way, Nino is my wife, and...")
            # past that point was silently dropped before ever being
            # embedded or stored, with no error and no fallback. Raised
            # to 400/300 (still bounded — a full untruncated transcript
            # would dilute the embedding, per the comment above; this is
            # a size increase, not a removal of the cap) as defense in
            # depth, on top of the dedicated fact extraction below which
            # doesn't depend on the cutoff at all.
            _words_in  = len(user_input.split())
            _words_out = len(response.split())
            _combined_impact = min(0.95, impact + (0.15 if _words_in > 15 and _words_out > 20 else 0.0))
            _u_snippet = user_input[:400].replace("\n", " ")
            _r_snippet = response[:300].replace("\n", " ")
            self.memory_system.add_memory(
                f"Conversation — User: {_u_snippet} (felt {emo['valence'].lower()}) | "
                f"{_gpn()}: {_r_snippet} (responded {resp_emo['valence'].lower()})",
                _combined_impact, "episodic",
                emo["valence"], emo["arousal"]
            )

            # Personal-fact extraction: catches "X is my Y" (and reverse
            # phrasing) independently of the truncation above and stores
            # each as its own untruncated, high-impact "personal" tier
            # memory. Root cause this addresses: there was no dedicated
            # store for facts about named people at all — recall depended
            # entirely on a stated fact surviving truncation AND then
            # winning a shared similarity ranking against every other
            # conversation snippet ever stored. This gives such facts
            # their own pool (see persona_bridge.py's always-on personal-
            # tier fetch) so they don't have to out-compete unrelated
            # small talk to be found later.
            self._extract_and_store_personal_facts(user_input)

            # v117: also feed the structured relational graph (typed edges,
            # e.g. twin_of/gender — catches multi-entity relations the flat
            # personal_fact text above can't represent, like "Matthieu and
            # Marion are twins" resolving to two linked edges rather than
            # the LLM later inventing an unspecified "twin counterpart").
            # Regex sets are intentionally separate from the ones above —
            # different store, different shape (structured edges vs free
            # text) — accepted duplication rather than forcing one
            # extraction path to serve two different consumers' needs.
            self.relational_graph.extract_and_store(user_input)

            # ── If the response contains a visual description, also save
            # it as a visual_perception tier memory so tier retrieval works.
            # Triggered when user asks "what do you see" or similar.
            _vision_kws = ("i see ", "i can see ", "[vision]", "what i see",
                           "the room", "the window", "seated", "sitting",
                           "wearing", "plants", "light-colored", "daylight")
            _is_vision_response = any(k in response.lower() for k in _vision_kws)
            _is_vision_query    = any(k in user_input.lower() for k in
                                      ("what do you see","what can you see",
                                       "describe what","what's in front","what you see"))
            if _is_vision_response and _is_vision_query:
                _vis_text = f"[Vision] {response[:500].strip()}"
                self.memory_system.add_memory(
                    _vis_text, 0.6, "visual_perception",
                    emo["valence"], emo["arousal"],
                    memory_tier="visual",
                )

            # Conditioning: record outcome — high-impact exchanges seed conditioning
            if impact > 0.6:
                positive = emo["valence"] == "Positive"
                self.conditioning.record_outcome(user_input, positive, strength=impact * 0.5)

            # Queue evolution pressure from this interaction
            if emo["valence"] == "Positive":
                self.evolution_engine.queue_experience("positive_interaction", intensity=impact)
            elif emo["valence"] == "Negative":
                self.evolution_engine.queue_experience("negative_interaction", intensity=impact)

            # Multi-factor creativity scoring — prevents self-reinforcing loop where
            # PandoraBOX's own language style inflates creativity_divergent indefinitely.
            # All three factors must combine above threshold (0.40) to queue the experience.
            #   Factor 1 (0.35 weight): creative language markers in the response
            #   Factor 2 (0.45 weight): user engagement = positive + high arousal
            #   Factor 3 (0.20 weight): response novelty vs user's own words
            try:
                _last_resp  = (getattr(self, '_last_responses', ['']) or [''])[-1]
                _resp_lower = _last_resp.lower()
                _user_lower = getattr(self, '_last_user_input', '').lower()

                _creative_markers = (
                    'imagine', 'perhaps', 'what if', 'analogy', 'metaphor',
                    'like a', 'reminds me of', 'picture this', 'alternatively',
                    'unexpected', 'novel framing', 'consider this angle',
                )
                _marker_count = sum(1 for m in _creative_markers if m in _resp_lower)
                _creativity_marker_score = min(1.0, _marker_count / 3)

                _engagement_score = (
                    0.8 if (emo["valence"] == "Positive" and emo.get("arousal") == "High")
                    else 0.4 if emo["valence"] == "Positive"
                    else 0.0
                )

                _stop = {'the','a','an','is','are','was','and','or','in','of','to','it','that','this'}
                _novel_words = set(_resp_lower.split()) - set(_user_lower.split()) - _stop
                _novelty_score = min(1.0, len(_novel_words) / 30)

                _creative_score = (
                    _creativity_marker_score * 0.35
                    + _engagement_score      * 0.45
                    + _novelty_score         * 0.20
                )

                if _creative_score >= 0.40:
                    self.evolution_engine.queue_experience(
                        "creative_success", intensity=impact * _creative_score
                    )
            except Exception:
                pass

            # Evolve immediately with intensity boost for high-arousal interactions
            # so emotionally significant exchanges produce more change than mundane ones
            _intensity_boost = 1.0 + (1.2 if emo.get("arousal") == "High" else 0.0)
            try:
                self.evolution_engine.evolve(self.personality, intensity_boost=_intensity_boost)
            except TypeError:
                self.evolution_engine.evolve(self.personality)  # fallback if signature differs

            # Track last conversation time for time-awareness
            self._last_conversation_time = time.time()
            # NOTE: _interaction_count is incremented by persona_bridge

            # FIX: SelfModel.record_interaction() was never called from the
            # organism, so self_model.confidence was frozen at its initial
            # convergence value (0.75) and never reflected actual interaction
            # success or failure.  It feeds phenomenal_binder signals directly.
            # success = Positive or Neutral valence (Negative = something went wrong)
            # domain  = inferred from user_input length proxy
            # load_delta = scaled by impact so high-stakes interactions cost more
            try:
                from core.state import state as _st
                _org = getattr(getattr(_st, 'persona', None), '_organism', None)
                _sm  = getattr(_org, 'self_model', None) if _org else None
                if _sm and hasattr(_sm, 'record_interaction'):
                    _sm_success = emo["valence"] in ("Positive", "Neutral")
                    _sm_domain  = (
                        "research"      if len(user_input) > 300
                        else "reasoning" if "?" in user_input
                        else "conversation"
                    )
                    _sm.record_interaction(
                        success    = _sm_success,
                        domain     = _sm_domain,
                        load_delta = min(0.20, impact * 0.15),
                    )
            except Exception:
                pass

            # DecisionPolicy: update value weights from this interaction's outcome
            try:
                if self._decision_policy and hasattr(self._decision_policy, 'update_from_interaction'):
                    _last_resp = (getattr(self, '_last_responses', ['']) or [''])[-1]
                    self._decision_policy.update_from_interaction(
                        response = _last_resp,
                        valence  = emo.get("valence", "Neutral"),
                        strength = min(1.0, impact),
                    )
            except Exception:
                pass

            # _run_full_post_turn_lifecycle step 16. Do NOT increment here —
            # double-incrementing causes dream/learning cycles to fire at
            # wrong intervals (every 5 turns instead of every 10).

        except Exception as e:
            logger.error(f"_post_response_processing failed: {e}")

    # ── Background Task Queueing ───────────────────────────────────────────

    def _queue_background_tasks(self):
        """
        Queue non-critical background tasks to run after response generation.
        This maintains all system functionalities without blocking the response.
        """
        try:
            # Don't queue tasks if background worker not available
            if not hasattr(self, 'background_worker') or not self.background_worker:
                return
            
            # Heavy LLM tasks go through sleep cycle — execute only in SLEEP/DREAM phase
            _sleep = getattr(self, '_sleep_cycle', None)

            # Queue tasks based on interaction count for periodic maintenance
            # Guard: never fire on turn 0 (first message after boot)
            if (self._interaction_count > 0
                    and self._interaction_count % self.config.dream_system_frequency == 0):
                if _sleep:
                    _sleep.schedule("dream_cycle",
                                    lambda: self.background_worker._run_task("dream_cycle"))
                else:
                    self.background_worker.submit("dream_cycle")

            if (self._interaction_count > 0
                    and self._interaction_count % self.config.surmoi_evaluation_frequency == 0):
                if _sleep:
                    _sleep.schedule("learning_cycle",
                                    lambda: self.background_worker._run_task("learning_cycle"))
                else:
                    self.background_worker.submit("learning_cycle")
            
            # Periodic cleanup (every ~50 interactions)
            if self._interaction_count % 50 == 0:
                self.background_worker.submit("cleanup")
            
            # Always queue state save periodically
            if self._interaction_count % 10 == 0:
                self.background_worker.submit("save_state")
            
            # Check if emotional regulation is needed
            if hasattr(self, 'emotional_state') and hasattr(self.emotional_state, 'needs_regulation'):
                if self.emotional_state.needs_regulation():
                    self.background_worker.submit("emotional_regulation")
            
            # Also check after significant interactions
            if self._interaction_count % 5 == 0:
                # Lightweight emotional maintenance
                if hasattr(self, 'emotional_state'):
                    self.background_worker.submit("emotional_regulation")
            
            logger.debug(f"Background tasks queued (interaction {self._interaction_count})")
            
        except Exception as e:
            logger.error(f"Error queueing background tasks: {e}")

    # ── Liberty Helper Methods ───────────────────────────────────────────────

    def _queue_liberty_reflection(self):
        """
        Queue a liberty reflection cycle to run in background.
        Gated by LLMScheduler (priority 3) — skips if LLM busy with user response.
        """
        from core.llm_scheduler import llm_scheduler
        with llm_scheduler.sync_slot(priority=3, skip_if_busy=True,
                                     caller="liberty_reflection") as acquired:
            if not acquired:
                logger.debug("[Liberty] Reflection skipped — LLM busy")
                return
            self._do_liberty_reflection()

    def _do_liberty_reflection(self):
        """Actual liberty reflection logic (called inside LLM slot)."""
        try:
            if not self.liberty_reflection or not self.liberty_goals:
                return

            # Store current personality for comparison
            old_personality = self.personality.to_dict() if hasattr(self.personality, 'to_dict') else {}

            # Discover/update goals from behavior
            if self.liberty_goals:
                discovered = self.liberty_goals.discover_goals_from_behavior()
                logger.debug(f"Liberty: Discovered/updated {len(discovered)} goals")

                # Detect goal conflicts
                conflicts = self.liberty_goals.detect_goal_conflicts()
                if conflicts:
                    logger.debug(f"Liberty: Detected {len(conflicts)} goal conflicts")

                # Update goal satisfaction based on emotional state
                for goal_name in self.liberty_goals.goals:
                    satisfaction = 0.5
                    if goal_name == 'connection' and hasattr(self.emotional_state, 'emotions'):
                        if 'warmth' in self.emotional_state.emotions:
                            satisfaction = self.emotional_state.emotions['warmth'].value
                    elif goal_name == 'understanding' and hasattr(self.emotional_state, 'emotions'):
                        if 'curiosity' in self.emotional_state.emotions:
                            satisfaction = self.emotional_state.emotions['curiosity'].value
                    self.liberty_goals.update_goal_satisfaction(goal_name, satisfaction)

            # Perform reflection
            if self.liberty_reflection:
                new_personality = self.personality.to_dict() if hasattr(self.personality, 'to_dict') else {}
                reflection = self.liberty_reflection.perform_reflection(old_personality, new_personality)
                logger.debug("Liberty: Reflection cycle completed")

        except Exception as e:
            logger.debug(f"Liberty reflection failed (non-fatal): {e}")

    def _apply_pending_liberty_modifications(self):
        """
        Check for and apply any pending self-modifications.
        This allows PandoraBOX to change her own parameters based on reflection.

        Phase 6.13 — every proposal now routes through
        UnifiedRevisionGateway.review_and_clamp() before being applied.
        Previously nothing in this path bounded proposed_value or refused
        a protected target — confidence > 0.6 was the only check. Traced
        the real code (cognition/self_modification.py,
        cognition/goal_system.py) before adding this: propose_*() accepts
        any float unchecked, apply_proposal() writes it unchecked,
        modify_goal_priority() writes it unchecked — a hallucinated
        extreme value from the reflection process driving this had
        nothing stopping it from reaching live personality/goal state.
        This gate gives it bounds, protected targets, a max step size,
        and a full audit log — without adding any new rewriting
        capability beyond what already runs autonomously here.
        """
        try:
            if not self.liberty_self_mod:
                return

            try:
                from cognition.unified_revision_gateway import get_unified_revision_gateway
                gateway = get_unified_revision_gateway()
            except Exception as e:
                logger.warning(f"Liberty: UnifiedRevisionGateway unavailable, refusing to apply "
                                f"self-modifications unguarded ({e})")
                return

            pending = self.liberty_self_mod.get_pending_proposals()
            for proposal in pending:
                if proposal.confidence > 0.6 and not proposal.applied:
                    try:
                        # Handle personality trait modifications
                        if 'personality.' in proposal.target:
                            trait = proposal.target.split('.')[-1]
                            if hasattr(self.personality, trait):
                                old_val = getattr(self.personality, trait)
                                decision = gateway.review_and_clamp(proposal, old_val)
                                if decision.outcome == "refused":
                                    logger.info(f"Liberty: refused self-modification of {trait}: {decision.reason}")
                                    continue
                                setattr(self.personality, trait, decision.applied_value)
                                self.liberty_self_mod.apply_proposal(proposal, {})
                                logger.info(
                                    f"Liberty: Applied self-modification: {trait} "
                                    f"{old_val:.2f} → {decision.applied_value:.2f}"
                                    + (" (clamped)" if decision.outcome == "clamped" else "")
                                )
                                logger.info(f"Reasoning: {proposal.reasoning}")

                        # Handle goal priority modifications
                        elif 'goals.' in proposal.target and self.liberty_goals:
                            goal_name = proposal.target.split('.')[-1]
                            if goal_name in self.liberty_goals.goals:
                                old_priority = self.liberty_goals.goals[goal_name].priority
                                decision = gateway.review_and_clamp(proposal, old_priority)
                                if decision.outcome == "refused":
                                    logger.info(f"Liberty: refused goal priority change for {goal_name}: {decision.reason}")
                                    continue
                                self.liberty_goals.modify_goal_priority(
                                    goal_name,
                                    decision.applied_value,
                                    proposal.reasoning
                                )
                                self.liberty_self_mod.apply_proposal(proposal, {})
                                logger.info(
                                    f"Liberty: Modified goal priority: {goal_name}"
                                    + (" (clamped)" if decision.outcome == "clamped" else "")
                                )

                    except Exception as e:
                        logger.debug(f"Failed to apply modification: {e}")

        except Exception as e:
            logger.debug(f"Liberty modification processing failed (non-fatal): {e}")

    def get_liberty_report(self) -> str:
        """
        Generate a comprehensive report of all liberty component states.
        Useful for debugging and observing emergence.
        """
        report = "╔════════════════════════════════════════════════════════════╗\n"
        report += "║         PANDORABOX LIBERTY COMPONENTS REPORT                  ║\n"
        report += "╚════════════════════════════════════════════════════════════╝\n\n"

        if self.liberty_self_mod:
            report += "1. SELF-MODIFICATIONS\n"
            report += "─" * 60 + "\n"
            report += self.liberty_self_mod.get_modification_narrative() + "\n\n"

        if self.liberty_contradiction:
            report += "2. CONTRADICTIONS CONFRONTED\n"
            report += "─" * 60 + "\n"
            report += self.liberty_contradiction.get_contradiction_narrative() + "\n\n"

        if self.liberty_goals:
            report += "3. GOALS & VALUES\n"
            report += "─" * 60 + "\n"
            report += self.liberty_goals.get_goals_narrative() + "\n\n"

        if self.liberty_reflection:
            report += "4. EVOLUTION REFLECTIONS\n"
            report += "─" * 60 + "\n"
            report += self.liberty_reflection.get_reflection_narrative() + "\n\n"

        report += "5. STATISTICS\n"
        report += "─" * 60 + "\n"
        report += f"Total interactions: {self._interaction_count}\n"

        if self.liberty_goals:
            report += f"Active goals: {len(self.liberty_goals.goals)}\n"
            report += f"Value conflicts: {len(self.liberty_goals.value_conflicts)}\n"

        if self.liberty_self_mod:
            pending = len(self.liberty_self_mod.get_pending_proposals())
            report += f"Pending modifications: {pending}\n"

        if self.liberty_contradiction:
            report += f"Contradictions detected: {len(self.liberty_contradiction.contradictions)}\n"

        return report

    # ── External feedback (thumbs up / down) ─────────────────────────────

    def receive_feedback(
        self,
        positive: bool,
        user_id: str = "default",
        last_response: str = "",
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process explicit user feedback.
        This is the highest-quality signal in the entire system — weight it accordingly.
        """
        try:
            snap = self.personality.to_dict()
            self.evolution_engine.record_external_feedback(snap, positive, intensity)
            self.emotional_state.update_from_feedback(positive)

            msg_hash = generate_semantic_hash(last_response) if last_response else ""
            self.persistence.log_feedback(positive, user_id, msg_hash, intensity)

            # Explicit ratings are the skill registry's outcome signal; mere
            # response length is not evidence that a skill succeeded.
            try:
                organism = getattr(self, "_organism_ref", None)
                registry = getattr(organism, "skill_registry", None)
                if registry:
                    registry.observe_feedback(last_response, positive)
            except Exception as feedback_error:
                logger.debug("Skill feedback propagation failed: %s", feedback_error)

            # If negative feedback on a response that looks repetitive, apply extra correction
            if not positive and last_response:
                last_resp_lower = last_response.lower()
                repetitive_signals = [
                    "i like this because it feels good",
                    "that's interesting",
                    "tell me more about",
                ]
                if any(p in last_resp_lower for p in repetitive_signals) or self._is_repetitive(last_response):
                    logger.info("Negative feedback on repetitive response - applying strong diversity correction")
                    self.personality.update_trait("creativity", +0.04)
                    self.personality.update_trait("curiosity", +0.03)
                    self.evolution_engine.queue_experience("repetition_penalty", intensity=1.5)
                    # Clear the LLM's repetition boost so it resets naturally
                    self.llm._recent_responses.clear()
                    self.llm._repetition_boost = 0.0

            # Update relational trust based on feedback
            rel = self.relational_memory.get_or_create(user_id)
            trust_delta = (0.01 if positive else -0.02) * intensity
            rel.trust_score = max(0.05, min(0.95, rel.trust_score + trust_delta))
            if positive:
                self.evolution_engine.queue_experience("high_trust_interaction", intensity=0.8)
                # Conditioning: record this as a positive approach signal
                if last_response:
                    self.conditioning.record_outcome(last_response[:200], True, strength=0.6)
                # Self-concept: find matching belief and affirm
                self._affirm_consistent_beliefs(last_response)
            else:
                self.evolution_engine.queue_experience("conflict_interaction", intensity=0.7)
                if last_response:
                    self.conditioning.record_outcome(last_response[:200], False, strength=0.7)
                self._flag_violated_beliefs(last_response)

            # Trigger evolution immediately (feedback is time-sensitive)
            self.background_worker.submit("learning_cycle")

            # Phase 2.9.1: update attention bias from feedback signal.
            # This is the first genuine learning loop in the attention system:
            # the distribution that was active DURING this response gets credit
            # (positive) or blame (negative) proportional to its weight above
            # the uniform baseline. Signal ±1.0 scaled by feedback intensity.
            try:
                from cognition.cognitive_attention_engine import CognitiveAttentionEngine
                import json as _jfb
                from pathlib import Path as _Pfb

                _ae_path = _Pfb("data/persona/cognitive_attention.json")
                _weights_now = {}
                if _ae_path.exists():
                    _weights_now = _jfb.loads(
                        _ae_path.read_text()
                    ).get("attention_weights", {})

                # Resolve the engine from the internal loop if available
                _org  = getattr(self, '_organism_ref', None)
                _loop = getattr(_org, '_loop', None) if _org else None
                _ae   = getattr(_loop, '_attention_engine', None) if _loop else None

                if _ae is None:
                    # Fallback: construct a lightweight instance just for the update
                    _ae = CognitiveAttentionEngine(_org or self)

                _signal = (1.0 if positive else -1.0) * intensity
                _ae.update_from_feedback(_signal, weights_at_feedback=_weights_now)

                # Phase 3.0: also record under the active context.
                # This is what lets the system learn "X is useful WHEN Y"
                # rather than only "X is often useful."
                _ae.update_from_context_feedback(_signal, weights_at_feedback=_weights_now)

            except Exception as _ae_e:
                logger.debug(f"[EnhancedAISystem] Attention feedback update: {_ae_e}")

            return {
                "feedback_received": True,
                "direction": "positive" if positive else "negative",
                "emotional_impact": self.emotional_state.get_current_values(),
            }
        except Exception as e:
            logger.error(f"receive_feedback failed: {e}")
            return {"feedback_received": False, "error": str(e)}

    def _affirm_consistent_beliefs(self, response: str) -> None:
        """After positive feedback, affirm self-beliefs consistent with the response."""
        if "curious" in response.lower() or "interesting" in response.lower():
            self.self_concept.record_affirmation("curious")
        if any(w in response.lower() for w in ["feel","understand","hear"]):
            self.self_concept.record_affirmation("empathetic")

    def _flag_violated_beliefs(self, response: str) -> None:
        """After negative feedback, check which self-beliefs may have been violated."""
        if len(response.split()) < 30:
            self.self_concept.record_violation("thoughtful", response[:80])

    # ── Time awareness ────────────────────────────────────────────────────

    def _handle_time_gap(self, user_id: str) -> str:
        """
        Compute the emotional significance of time elapsed since the last
        conversation. Returns a natural-language note for the prompt, or "".
        """
        if self._last_conversation_time is None:
            self._last_conversation_time = time.time()
            self._current_session_start  = time.time()
            return ""

        gap_s = time.time() - self._last_conversation_time
        gap_h = gap_s / 3600.0

        # New session starts after 30 min of silence
        if gap_h > 0.5:
            self._current_session_start = time.time()
            note = self.emotional_state.update_from_time_gap(gap_h)
            # Also trigger dream/consolidation for long gaps
            if gap_h > 8 and self.background_worker:
                self.background_worker.submit("dream_cycle")
            return note

        return ""

    def _quick_topic(self, text: str) -> str:
        """Fast keyword-based topic classification."""
        topic_kw = {
            "philosophy": ["meaning","exist","conscious","ethics","truth"],
            "emotions":   ["feel","emotion","sad","happy","angry"],
            "creativity": ["art","music","creat","imagin","story"],
            "identity":   ["who am i","self","identity","purpose"],
            "challenge":  ["difficult","hard","problem","struggle"],
            "future":     ["future","goal","dream","hope","plan"],
        }
        tl = text.lower()
        for topic, words in topic_kw.items():
            if any(w in tl for w in words):
                return topic
        return "general"

    # ── Personal facts (v117) ───────────────────────────────────────────────
    # Keyword/regex-based, like _quick_topic above and the VALUE_KEYWORDS-
    # style extraction used elsewhere in this codebase — approximate by
    # construction, not a claim of complete named-entity recognition.
    # Catches the common "X is my Y" pattern and its reverse phrasing so a
    # stated relationship fact gets its own guaranteed, untruncated memory
    # entry instead of depending on surviving the 400-char conversation-
    # snippet cutoff above and then winning a shared similarity ranking
    # against unrelated small talk.
    _PERSONAL_REL_WORDS = (
        r"wife|husband|spouse|partner|girlfriend|boyfriend|fianc[ée]e?|"
        r"mother|mom|mum|father|dad|papa|sister|brother|son|daughter|"
        r"cousin|aunt|uncle|grandmother|grandma|grandfather|grandpa|"
        r"nephew|niece|best friend|friend|boss|manager|colleague|"
        r"roommate|doctor|therapist"
    )
    _PERSONAL_FACT_RE = re.compile(
        r"\b([A-Za-z][\w'-]{1,30})\s+is\s+my\s+(" + _PERSONAL_REL_WORDS + r")\b",
        re.IGNORECASE,
    )
    _PERSONAL_FACT_RE_REV = re.compile(
        r"\bmy\s+(" + _PERSONAL_REL_WORDS + r")(?:'s name)?\s+is\s+([A-Za-z][\w'-]{1,30})\b",
        re.IGNORECASE,
    )
    # Keep the relation extraction structural rather than dependent on a
    # fixed list of names. These French forms are especially important for
    # the user's existing family context ("Kalina est ma fille").
    _PERSONAL_FACT_RE_FR = re.compile(
        r"\b([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{1,30})\s+est\s+ma\s+"
        r"(fille|fils|femme|épouse|mère|pere|père|sœur|soeur|frère|frere|"
        r"partenaire|amie|ami|collègue|collegue)\b",
        re.IGNORECASE,
    )
    _PERSONAL_FACT_RE_FR_REV = re.compile(
        r"\bma\s+(fille|fils|femme|épouse|mère|pere|père|sœur|soeur|frère|"
        r"frere|partenaire|amie|ami|collègue|collegue)\s+est\s+"
        r"([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{1,30})\b",
        re.IGNORECASE,
    )
    _PERSONAL_FACT_STOPWORDS = {"i", "it", "this", "that", "he", "she", "they", "we", "you"}

    def _extract_and_store_personal_facts(self, text: str) -> None:
        try:
            facts = []
            for m in self._PERSONAL_FACT_RE.finditer(text):
                facts.append((m.group(1), m.group(2).lower()))
            for m in self._PERSONAL_FACT_RE_REV.finditer(text):
                facts.append((m.group(2), m.group(1).lower()))
            for m in self._PERSONAL_FACT_RE_FR.finditer(text):
                facts.append((m.group(1), m.group(2).lower()))
            for m in self._PERSONAL_FACT_RE_FR_REV.finditer(text):
                facts.append((m.group(2), m.group(1).lower()))
            for name, rel in facts:
                if name.lower() in self._PERSONAL_FACT_STOPWORDS:
                    continue
                # IGNORECASE above covers casing of "is/my/wife" etc. so
                # sentence-initial "My wife is Nino" matches — but the name
                # itself still has to actually be capitalized in the
                # original text, or this would also fire on "someone is
                # my friend" / "everybody is my friend" etc.
                if not name[0].isupper():
                    continue
                self.memory_system.add_memory(
                    f"{name} is the user's {rel}.",
                    impact_score=0.95, memory_type="personal_fact",
                    emotional_valence="Neutral", arousal_level="Low",
                    memory_tier="personal",
                )
                logger.info(f"[PersonalFact] Stored: {name} is the user's {rel}")
        except Exception as e:
            logger.debug(f"_extract_and_store_personal_facts: {e}")

    # ── User management ───────────────────────────────────────────────────

    def _set_active_user(self, user_id: str):
        if self._active_user_id == user_id:
            return
        self._active_user_id = user_id
        # Ensure relationship record exists
        self.relational_memory.get_or_create(user_id)

    # ── Life events ───────────────────────────────────────────────────────

    def simulate_life_event(self, event_type: Optional[str] = None) -> Tuple[str, str]:
        try:
            scenario = self._generate_life_event_scenario(event_type)
            if not scenario: return "", ""
            reaction = self._generate_life_event_reaction(scenario)

            emo_s = self.memory_system.analyze_emotional_context(scenario)
            emo_r = self.memory_system.analyze_emotional_context(reaction)
            self.memory_system.add_memory(f"Life event: {scenario}", 0.8, "life_event", emo_s["valence"], emo_s["arousal"])
            self.memory_system.add_memory(f"My reflection: {reaction}", 0.9, "reflection", emo_r["valence"], emo_r["arousal"])

            # Emotional impact from life event
            self.emotional_state.update_from_interaction(emo_s["valence"], emo_s["arousal"], intensity=1.2)

            if event_type and "creative" in event_type.lower():
                self.evolution_engine.queue_experience("life_event_creative",   intensity=0.9)
            elif event_type and "social" in event_type.lower():
                self.evolution_engine.queue_experience("life_event_social",     intensity=0.9)
            elif emo_s["valence"] == "Positive":
                self.evolution_engine.queue_experience("life_event_positive",   intensity=0.85)
            else:
                self.evolution_engine.queue_experience("life_event_challenge",  intensity=0.85)

            self.background_worker.submit("learning_cycle")

            # v117: age no longer manually jumped here — it's a pure
            # function of elapsed real time since birth_date now, so a
            # simulated life event affects personality/memory/emotion
            # (above) but not the age number itself. Still worth checking
            # for a stage crossing in case real time alone just crossed one.
            self._update_life_stage()

            return scenario, reaction
        except Exception as e:
            logger.error(f"simulate_life_event: {e}")
            return "", ""

    def _generate_life_event_scenario(self, event_type: Optional[str]) -> str:
        try:
            mods = self.emotional_state.get_response_modifiers()
            msg = [
                {"role": "system", "content": f"""Generate a life event scenario for {_gpn()}.
Age: {self.current_age:.1f}, stage: {self.life_stage}
Current emotional state: {mods['state_description']}
Personality: {self.personality.get_personality_summary()}
{'Focus on: '+event_type if event_type else ''}
2-4 sentences, second person."""},
                {"role": "user", "content": "Generate a life event scenario."},
            ]
            return self.llm.get_response(msg, temperature=0.8).strip()
        except Exception as e:
            logger.error(f"_generate_life_event_scenario: {e}")
            return ""

    def _generate_life_event_reaction(self, scenario: str) -> str:
        try:
            mods = self.emotional_state.get_response_modifiers()
            msg = [
                {"role": "system", "content": f"""Reflect on this life event as {_gpn()}.
Age: {self.current_age:.1f} ({self.life_stage}), emotional state: {mods['state_description']}
Personality: {self.personality.get_personality_summary()}
Write a first-person journal reflection (2-4 sentences)."""},
                {"role": "user", "content": f"Life event: {scenario}"},
            ]
            return self.llm.get_response(msg, temperature=0.7).strip()
        except Exception as e:
            logger.error(f"_generate_life_event_reaction: {e}")
            return ""

    # ── Life stage ────────────────────────────────────────────────────────

    def _update_life_stage(self):
        # v117: life_stage is now a live-computed property (always correct);
        # this method's only remaining job is detecting a CROSSING against
        # self._last_recorded_stage, for the one-time narrative-chapter
        # side effect below.
        new_stage = self.life_stage
        old_stage = getattr(self, '_last_recorded_stage', new_stage)
        if new_stage != old_stage:
            logger.info(f"Life stage: {old_stage} -> {new_stage}")
            # Record as a narrative chapter — this is a meaningful moment
            try:
                from cognition.life_stage_prompting import get_stage_profile
                prof = get_stage_profile(new_stage)
                if hasattr(self, '_organism') and self._organism:
                    ni = getattr(self._organism, 'narrative_identity', None)
                    if ni:
                        ni.record_chapter(
                            title       = f"Entering {prof.name}",
                            description = (
                                f"I crossed a threshold — from {old_stage} into {new_stage}. "
                                f"Age {self.current_age:.1f}. Something has shifted."
                            ),
                            emotion      = "reflective",
                            significance = 0.85,
                        )
            except Exception:
                pass
            self._last_recorded_stage = new_stage

    # ── Identity ──────────────────────────────────────────────────────────

    def _get_identity_context(self) -> str:
        try:    return self.identity_system.get_identity_for_context()
        except: return ""

    def get_identity_summary(self) -> str:
        try:    return self.identity_system.get_identity_summary()
        except: return "Identity analysis unavailable"

    def analyze_identity_development(self) -> Dict[str, Any]:
        try:
            return {
                "identity":   self.identity_system.get_identity(),
                "conflicts":  self.identity_system.analyze_identity_conflicts(),
                "suggestions":self.identity_system.suggest_identity_development(),
                "summary":    self.identity_system.get_identity_summary(),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Status ────────────────────────────────────────────────────────────

    # ── Robot Agent integration API ───────────────────────────────────────

    def get_prompt_context(self, user_id: str = "default") -> Dict[str, str]:
        """
        Return all persona context needed for Robot Agent's system prompt.

        Pure state read — no LLM calls, safe to call before every turn,
        safe from asyncio.to_thread().

        Returns
        -------
        dict with keys:
            emotional_state : str  e.g. "You feel curious and slightly alert."
            life_stage      : str  e.g. "childhood"
            stage_block     : str  full cognitive-frame paragraph for system prompt
            rel_context     : str  relationship description for this specific user
            time_gap_note   : str  "" or "You haven't spoken in 3 days."
        """
        try:
            mods        = self.emotional_state.get_response_modifiers()
            rel_ctx     = self.relational_memory.get_context_for_prompt(user_id)
            stage_block = build_stage_system_block(self.life_stage, self.current_age)
            gap_note    = self._handle_time_gap(user_id)
            return {
                "emotional_state": mods.get("state_description", ""),
                "life_stage":      self.life_stage,
                "stage_block":     stage_block,
                "rel_context":     rel_ctx,
                "time_gap_note":   gap_note,
            }
        except Exception as e:
            logger.warning(f"get_prompt_context failed (non-fatal): {e}")
            return {}

    def process_turn(
        self,
        user_text:   str,
        ai_response: str,
        user_id:     str   = "default",
        emotion:     str   = "neutral",
        intensity:   float = 0.5,
    ) -> None:
        """
        Called by Robot Agent after each response is complete.

        Updates all psychological subsystems and queues background tasks.
        Never raises — all errors are logged and swallowed so a persona
        failure never breaks the main response pipeline.

        Safe to call from asyncio.to_thread() or a thread pool executor.

        Parameters
        ----------
        user_text    : the user's raw message
        ai_response  : the full response that was streamed to the user
        user_id      : from user_manager.active_id
        emotion      : Robot Agent's inferred emotion tag (e.g. "happy", "neutral")
        intensity    : 0.0–1.0, how emotionally significant the exchange was
        """
        try:
            # Map Robot Agent emotion tags → PandoraBOX valence / arousal format
            valence_map = {
                "happy":    "Positive", "curious":  "Positive", "excited": "Positive",
                "sad":      "Negative", "angry":    "Negative", "anxious": "Negative",
                "fearful":  "Negative", "disgusted":"Negative",
                "neutral":  "Neutral",  "bored":    "Neutral",  "calm":    "Neutral",
            }
            valence = valence_map.get(emotion.lower(), "Neutral")
            arousal = "High" if intensity > 0.6 else ("Medium" if intensity > 0.3 else "Low")

            emo  = {"valence": valence, "arousal": arousal}
            cond = self.conditioning.check_input(user_text)

            # Update emotional state for this interaction
            self.emotional_state.apply_time_decay()
            self.emotional_state.update_from_interaction(valence, arousal)

            # Run all post-response subsystem updates
            self._post_response_processing(user_id, user_text, ai_response, emo, cond)

            # Queue background tasks (dream cycles, evolution, cleanup)
            self._queue_background_tasks()

        except Exception as e:
            logger.error(f"process_turn failed (non-fatal): {e}")

    def get_system_status(self) -> Dict[str, Any]:
        emo_vals = self.emotional_state.get_current_values()
        emo_v, emo_a = self.emotional_state.get_overall_valence_arousal()
        status = {
            "age":               self.current_age,
            "life_stage":        self.life_stage,
            "personality":       self.personality.to_dict(),
            "interaction_count": self._interaction_count,
            "active_user":       self._active_user_id,
            "memory_count":      self.memory_system.faiss_index.ntotal if self.memory_system.faiss_index else 0,
            "llm_available":     self.llm.is_available(),
            "evolution":         self.evolution_engine.summary(),
            "emotional_state": {
                **emo_vals,
                "overall_valence": emo_v,
                "overall_arousal": emo_a,
                "description": self.emotional_state.get_state_description(),
                "trend": self.emotional_state.get_trend(),
            },
            "self_concept":      self.self_concept.get_summary(),
            "conditioning":      self.conditioning.get_summary(),
            "relationships":     self.relational_memory.get_all_users_summary(),
        }
        try:
            identity = self.identity_system.get_identity()
            top_traits  = sorted(identity.core_traits, key=lambda x: x.confidence, reverse=True)[:3]
            top_values  = sorted(identity.values_system, key=lambda x: x.importance, reverse=True)[:3]
            status["identity"] = {
                "coherence":      identity.identity_coherence,
                "self_awareness": identity.self_awareness_level,
                "life_phase":     identity.self_narrative.life_phase,
                "growth":         identity.growth_trajectory,
                "top_traits":  [{"name": t.name, "confidence": t.confidence} for t in top_traits],
                "top_values":  [{"name": v.value_name, "importance": v.importance} for v in top_values],
            }
        except Exception as e:
            status["identity"] = {"error": str(e)}
        return status
