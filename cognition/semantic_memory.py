"""
semantic_memory.py — Lumina Semantic Memory
============================================

Two-level knowledge structure:

  SemanticMemory   — SQLite-backed concept graph with relations and weights
  CognitiveMemory  — capability map with confidence + trend tracking

Both are written by:
  - SemanticExtractor  (real-time, called from GW post-response hook)
  - SemanticConsolidator (offline, called from Dream Cycle)

Schema
------
  concepts   (id, name, strength, last_used, use_count, created_at)
  relations  (id, source_id, target_id, relation_type, weight, last_used)
  aliases    (concept_id, alias)   — populated during consolidation
  capabilities (name, confidence, trend, last_updated, sample_count)
"""

import json
import logging
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,
    strength    REAL    DEFAULT 0.5,
    last_used   REAL    DEFAULT 0,
    use_count   INTEGER DEFAULT 1,
    created_at  REAL    DEFAULT 0
);
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    rel_type    TEXT    NOT NULL,
    weight      REAL    DEFAULT 0.5,
    last_used   REAL    DEFAULT 0,
    UNIQUE(source_id, target_id, rel_type)
);
CREATE TABLE IF NOT EXISTS aliases (
    concept_id  INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    alias       TEXT    NOT NULL,
    UNIQUE(concept_id, alias)
);
CREATE TABLE IF NOT EXISTS capabilities (
    name         TEXT PRIMARY KEY,
    confidence   REAL DEFAULT 0.5,
    trend        REAL DEFAULT 0.0,
    sample_count INTEGER DEFAULT 0,
    last_updated REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name);
CREATE TABLE IF NOT EXISTS concept_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    role TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    observed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observation_role ON concept_observations(role, observed_at);
"""

# Capability names that mirror the Self-Model
CAPABILITY_NAMES = [
    "conversation", "reasoning", "memory_analysis", "research",
    "creativity", "emotional_support", "code_analysis", "self_reflection",
]

# Concept pruning thresholds
PRUNE_STRENGTH_MIN  = 0.08    # concepts weaker than this get removed
PRUNE_AGE_DAYS      = 30      # concepts not used in 30 days get weakened
MERGE_SIM_THRESHOLD = 0.85    # string similarity to trigger alias merging
RELATION_DECAY      = 0.97    # per-consolidation decay on unused relations
CONCEPT_DECAY       = 0.96    # per-consolidation decay on unused concepts


@dataclass
class Concept:
    id:        int
    name:      str
    strength:  float
    use_count: int
    last_used: float


@dataclass
class Relation:
    source: str
    target: str
    rel_type: str
    weight: float


class SemanticMemory:
    """
    Persistent SQLite-backed concept graph.
    Thread-safe via connection-per-call pattern.
    """

    def __init__(self, db_path: str):
        self._path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"[SemanticMemory] Loaded — {self.concept_count()} concepts")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Seed capabilities if empty
            cur = conn.execute("SELECT COUNT(*) FROM capabilities")
            if cur.fetchone()[0] == 0:
                now = time.time()
                conn.executemany(
                    "INSERT OR IGNORE INTO capabilities (name, confidence, trend, last_updated) VALUES (?,0.5,0.0,?)",
                    [(n, now) for n in CAPABILITY_NAMES]
                )

    # ── Concepts ────────────────────────────────────────────────────────────

    def upsert_concept(self, name: str, strength_boost: float = 0.05) -> int:
        """Add or reinforce a concept. Returns concept id."""
        name = name.lower().strip()[:80]
        if not name: return -1
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO concepts (name, strength, last_used, use_count, created_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(name) DO UPDATE SET
                    strength  = MIN(1.0, strength + ?),
                    last_used = ?,
                    use_count = use_count + 1
            """, (name, 0.5 + strength_boost, now, now, strength_boost, now))
            row = conn.execute("SELECT id FROM concepts WHERE name=?", (name,)).fetchone()
            return row[0] if row else -1

    def record_observation(self, name: str, source: str = "unknown",
                           role: str = "context", confidence: float = 0.5) -> None:
        """Persist provenance separately from the shared concept graph."""
        concept_id = self.upsert_concept(name, strength_boost=0.0)
        if concept_id < 0:
            return
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO concept_observations "
                "(concept_id, source, role, confidence, observed_at) VALUES (?,?,?,?,?)",
                (concept_id, str(source)[:80], str(role)[:40], confidence, time.time()),
            )

    def recent_user_topics(self, limit: int = 15, min_strength: float = 0.5) -> List[str]:
        """Return concepts grounded in human-authored turns."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.name, MAX(o.observed_at), MAX(o.confidence), c.strength
                   FROM concept_observations o JOIN concepts c ON c.id = o.concept_id
                   WHERE o.role = 'user_topic' AND c.strength >= ?
                   GROUP BY c.id
                   ORDER BY MAX(o.confidence) DESC, MAX(o.observed_at) DESC, c.strength DESC
                   LIMIT ?""", (min_strength, limit)
            ).fetchall()
        return [row[0] for row in rows]

    def upsert_relation(self, source: str, target: str, rel_type: str,
                        weight_boost: float = 0.05):
        """Add or reinforce a directed relation between two concepts."""
        now = time.time()
        sid = self.upsert_concept(source)
        tid = self.upsert_concept(target)
        if sid < 0 or tid < 0 or sid == tid: return
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO relations (source_id, target_id, rel_type, weight, last_used)
                VALUES (?,?,?,?,?)
                ON CONFLICT(source_id, target_id, rel_type) DO UPDATE SET
                    weight    = MIN(1.0, weight + ?),
                    last_used = ?
            """, (sid, tid, rel_type, 0.4 + weight_boost, now, weight_boost, now))

    def get_related(self, concept: str, limit: int = 10) -> List[Relation]:
        """Return concepts related to the given concept."""
        concept = concept.lower().strip()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT c1.name as src, c2.name as tgt, r.rel_type, r.weight
                FROM relations r
                JOIN concepts c1 ON r.source_id = c1.id
                JOIN concepts c2 ON r.target_id = c2.id
                WHERE c1.name = ? OR c2.name = ?
                ORDER BY r.weight DESC LIMIT ?
            """, (concept, concept, limit)).fetchall()
        return [Relation(r["src"], r["tgt"], r["rel_type"], r["weight"]) for r in rows]

    def top_concepts(self, n: int = 20) -> List[Concept]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, strength, use_count, last_used FROM concepts "
                "ORDER BY strength DESC LIMIT ?", (n,)).fetchall()
        return [Concept(r["id"], r["name"], r["strength"], r["use_count"], r["last_used"]) for r in rows]

    def concept_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

    def relation_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    # ── Capabilities ────────────────────────────────────────────────────────

    def update_capability(self, name: str, new_confidence: float):
        """EMA update of a capability confidence score and trend."""
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT confidence, trend, sample_count FROM capabilities WHERE name=?", (name,)
            ).fetchone()
            if row:
                old_conf  = row["confidence"]
                old_trend = row["trend"]
                samples   = row["sample_count"]
                alpha     = max(0.1, 1.0 / (samples + 1))  # shrinking alpha
                new_conf  = old_conf + alpha * (new_confidence - old_conf)
                new_trend = 0.8 * old_trend + 0.2 * (new_confidence - old_conf)
                conn.execute("""
                    UPDATE capabilities
                    SET confidence=?, trend=?, sample_count=sample_count+1, last_updated=?
                    WHERE name=?
                """, (new_conf, new_trend, now, name))
            else:
                conn.execute("""
                    INSERT INTO capabilities (name, confidence, trend, sample_count, last_updated)
                    VALUES (?,?,0.0,1,?)
                """, (name, new_confidence, now))

    def get_capabilities(self) -> Dict[str, Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, confidence, trend, sample_count FROM capabilities"
            ).fetchall()
        return {
            r["name"]: {
                "confidence":   round(r["confidence"], 3),
                "trend":        round(r["trend"], 4),
                "sample_count": r["sample_count"],
            }
            for r in rows
        }

    # ── Summary (for dashboard / bridge) ────────────────────────────────────

    def summary(self) -> Dict:
        caps = self.get_capabilities()
        top  = self.top_concepts(8)
        return {
            "concepts":      self.concept_count(),
            "relations":     self.relation_count(),
            "top_concepts":  [c.name for c in top],
            "capabilities":  {k: round(v["confidence"], 2) for k, v in caps.items()},
        }

    # ── Pruning (called by consolidator) ────────────────────────────────────

    def apply_decay(self):
        """Decay unused concepts/relations. Called during Dream Cycle."""
        now  = time.time()
        age_threshold = now - PRUNE_AGE_DAYS * 86400
        with self._conn() as conn:
            # Decay concepts not used recently
            conn.execute("""
                UPDATE concepts SET strength = strength * ?
                WHERE last_used < ?
            """, (CONCEPT_DECAY, age_threshold))
            # Decay all relation weights slightly each cycle
            conn.execute("UPDATE relations SET weight = weight * ?", (RELATION_DECAY,))
            # Prune very weak concepts (cascade deletes their relations + aliases)
            conn.execute("DELETE FROM concepts WHERE strength < ?", (PRUNE_STRENGTH_MIN,))
            # Prune dead relations
            conn.execute("DELETE FROM relations WHERE weight < 0.05")
        logger.debug("[SemanticMemory] Decay applied")

    def merge_aliases(self, alias: str, canonical: str):
        """Merge 'alias' concept into 'canonical' — called during consolidation."""
        with self._conn() as conn:
            alias_row = conn.execute(
                "SELECT id FROM concepts WHERE name=?", (alias,)).fetchone()
            canon_row = conn.execute(
                "SELECT id FROM concepts WHERE name=?", (canonical,)).fetchone()
            if not alias_row or not canon_row:
                return
            aid, cid = alias_row["id"], canon_row["id"]
            # Redirect all relations from alias to canonical
            conn.execute(
                "UPDATE OR IGNORE relations SET source_id=? WHERE source_id=?", (cid, aid))
            conn.execute(
                "UPDATE OR IGNORE relations SET target_id=? WHERE target_id=?", (cid, aid))
            # Record alias
            conn.execute(
                "INSERT OR IGNORE INTO aliases (concept_id, alias) VALUES (?,?)", (cid, alias))
            # Merge strength
            conn.execute("""
                UPDATE concepts SET
                    strength  = MIN(1.0, strength + (SELECT strength FROM concepts WHERE id=?)),
                    use_count = use_count + (SELECT use_count FROM concepts WHERE id=?)
                WHERE id=?
            """, (aid, aid, cid))
            conn.execute("DELETE FROM concepts WHERE id=?", (aid,))
        logger.debug(f"[SemanticMemory] Merged '{alias}' → '{canonical}'")
