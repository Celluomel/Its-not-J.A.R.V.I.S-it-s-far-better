"""
semantic_consolidator.py — Offline Dream Cycle Consolidation
=============================================================

Called from LifeEventDreamSystem.run_dream_cycle() (ai_system.py).
Performs heavy consolidation that would be too slow for real-time:

  1. Concept merging  — detects near-duplicates via string similarity,
                        merges them into canonical forms with aliases
  2. Decay            — weakens concepts/relations not used recently
  3. Pruning          — removes concepts below strength threshold
  4. Path discovery   — finds indirect concept connections (2-hop)
                        and creates weak direct links (concept spreading)
  5. Capability trend — updates long-term capability trends in CognitiveMemory

The consolidator uses the LLM optionally for semantic similarity
(more accurate than string similarity but slower). Falls back to
string-edit-distance if LLM is unavailable.
"""

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from cognition.semantic_memory import (
    SemanticMemory, MERGE_SIM_THRESHOLD, CAPABILITY_NAMES
)

logger = logging.getLogger(__name__)

_MERGE_PROMPT = """Given these concept pairs, decide which should be merged.
Return ONLY valid JSON list of merges. No extra text.

Concepts: {concept_list}

Return format:
[
  {{"keep": "canonical_name", "merge": "duplicate_name"}},
  ...
]

Rules:
- Only merge concepts with the same meaning (synonyms, plural/singular, abbreviation)
- Do NOT merge concepts that are merely related (sound != music)
- keep: the more general/common form
- Return empty list [] if no merges needed"""


class SemanticConsolidator:
    """
    Dream-cycle consolidator for SemanticMemory.
    Designed to be called at most once per sleep phase.
    """

    def __init__(self, semantic_memory: SemanticMemory, llm_fn=None):
        self._memory = semantic_memory
        self._llm    = llm_fn
        self._last_consolidation = 0.0

    def set_llm(self, llm_fn):
        self._llm = llm_fn

    def consolidate(self) -> Dict:
        """
        Full consolidation pass. Returns a report dict.
        Typically takes 1-5 seconds depending on graph size.
        """
        t0 = time.time()
        logger.info("[SemanticConsolidator] Starting consolidation pass...")

        report = {
            "merges":        0,
            "pruned":        0,
            "purged":        0,
            "spread":        0,
            "caps_updated":  0,
            "elapsed_ms":    0,
        }

        concepts_before = self._memory.concept_count()

        # 0. Purge stopwords / generic words that slipped in
        purged = self._purge_stopwords()
        report["purged"] = purged

        # 1. Merge near-duplicates
        merges = self._find_merges()
        for canonical, alias in merges:
            self._memory.merge_aliases(alias, canonical)
            report["merges"] += 1

        # 2. Apply decay and pruning
        self._memory.apply_decay()
        concepts_after = self._memory.concept_count()
        report["pruned"] = max(0, concepts_before - concepts_after - report["merges"])

        # 3. Concept spreading — reinforce indirect connections
        spread = self._spread_activation()
        report["spread"] = spread

        # 4. Update capability trends
        report["caps_updated"] = self._consolidate_capabilities()

        report["elapsed_ms"] = int((time.time() - t0) * 1000)
        self._last_consolidation = time.time()

        logger.info(
            f"[SemanticConsolidator] Done — "
            f"merged={report['merges']} pruned={report['pruned']} "
            f"spread={report['spread']} in {report['elapsed_ms']}ms"
        )
        return report

    # ── Merge detection ─────────────────────────────────────────────────────

    def _purge_stopwords(self) -> int:
        """Delete concepts that are stopwords or too short to be meaningful."""
        from cognition.semantic_extractor import _STOP, _GENERIC_CONCEPTS, _MIN_CONCEPT_LEN
        try:
            import sqlite3
            conn = sqlite3.connect(self._memory._path)
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.execute("SELECT id, name FROM concepts")
            to_delete = []
            for row in cur.fetchall():
                cid, name = row
                if (name in _STOP or name in _GENERIC_CONCEPTS
                        or len(name) < _MIN_CONCEPT_LEN):
                    to_delete.append(cid)
            if to_delete:
                conn.executemany("DELETE FROM concepts WHERE id=?",
                                 [(cid,) for cid in to_delete])
                conn.commit()
            conn.close()
            if to_delete:
                logger.info(f"[SemanticConsolidator] Purged {len(to_delete)} stopword concepts")
            return len(to_delete)
        except Exception as e:
            logger.debug(f"[SemanticConsolidator] Purge error: {e}")
            return 0

    def _find_merges(self) -> List[Tuple[str, str]]:
        """Find concept pairs that should be merged."""
        concepts = self._memory.top_concepts(n=200)
        names = [c.name for c in concepts]
        merges = []

        # String similarity pass (fast, no LLM)
        string_candidates = self._string_similarity_merges(names)
        merges.extend(string_candidates)

        # LLM semantic pass (slow, optional, batched)
        if self._llm and len(names) >= 5:
            # Only check concepts not already handled by string pass
            already = {a for _, a in string_candidates} | {c for c, _ in string_candidates}
            remaining = [n for n in names if n not in already][:40]
            if len(remaining) >= 4:
                llm_merges = self._llm_similarity_merges(remaining)
                merges.extend(llm_merges)

        # Deduplicate
        seen = set()
        result = []
        for canonical, alias in merges:
            key = tuple(sorted([canonical, alias]))
            if key not in seen:
                seen.add(key)
                result.append((canonical, alias))
        return result

    def _string_similarity_merges(self, names: List[str]) -> List[Tuple[str, str]]:
        """Find merges using string edit distance."""
        merges = []
        checked = set()
        for i, a in enumerate(names):
            for b in names[i+1:]:
                if (a, b) in checked or (b, a) in checked:
                    continue
                checked.add((a, b))

                sim = SequenceMatcher(None, a, b).ratio()
                if sim >= MERGE_SIM_THRESHOLD:
                    # Keep the shorter/simpler one as canonical
                    canonical = a if len(a) <= len(b) else b
                    alias     = b if canonical == a else a
                    merges.append((canonical, alias))
        return merges

    def _llm_similarity_merges(self, names: List[str]) -> List[Tuple[str, str]]:
        """Use LLM to find semantic duplicates."""
        try:
            import json
            prompt = _MERGE_PROMPT.format(concept_list=", ".join(names))
            raw = self._llm(prompt)
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            return [(item["keep"], item["merge"]) for item in data
                    if "keep" in item and "merge" in item
                    and item["keep"] in names and item["merge"] in names]
        except Exception as e:
            logger.debug(f"[SemanticConsolidator] LLM merge failed: {e}")
            return []

    # ── Concept spreading ───────────────────────────────────────────────────

    def _spread_activation(self) -> int:
        """
        Find 2-hop concept paths (A→B→C) and create a weak direct
        link A→C if none exists. Simulates concept generalization.
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self._memory._path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")

            # Find 2-hop paths with combined weight > threshold
            rows = conn.execute("""
                SELECT r1.source_id, r2.target_id,
                       r1.rel_type, (r1.weight * r2.weight) as combined
                FROM relations r1
                JOIN relations r2 ON r1.target_id = r2.source_id
                WHERE combined > 0.25
                  AND r1.source_id != r2.target_id
                LIMIT 100
            """).fetchall()

            count = 0
            now = time.time()
            for row in rows:
                # Only create if no direct link exists
                existing = conn.execute("""
                    SELECT id FROM relations
                    WHERE source_id=? AND target_id=?
                """, (row["source_id"], row["r2.target_id"] if "r2.target_id" in row.keys()
                      else row[1])).fetchone()
                if not existing:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO relations
                            (source_id, target_id, rel_type, weight, last_used)
                            VALUES (?,?,?,?,?)
                        """, (row[0], row[1], "inferred_" + row[2],
                              row[3] * 0.6, now))
                        count += 1
                    except Exception:
                        pass
            conn.commit()
            conn.close()
            return count
        except Exception as e:
            logger.debug(f"[SemanticConsolidator] Spread error: {e}")
            return 0

    # ── Capability consolidation ─────────────────────────────────────────────

    def _sync_to_self_model(self):
        """Push SemanticMemory confidence scores into SelfModel so both stay in sync."""
        try:
            organism = getattr(self, '_organism_ref', None)
            if organism is None:
                return
            sm_obj = getattr(organism, 'self_model', None)
            if sm_obj is None:
                return
            caps = self._memory.get_capabilities()
            for name, data in caps.items():
                if name in sm_obj.capabilities and data['sample_count'] >= 3:
                    # Blend: 30% SemanticMemory, 70% SelfModel (SelfModel has richer signal)
                    old_score = sm_obj.capabilities[name].score
                    new_score = old_score * 0.70 + data['confidence'] * 0.30
                    sm_obj.capabilities[name].score = round(new_score, 3)
            logger.debug("[SemanticConsolidator] Synced capabilities to SelfModel")
        except Exception as e:
            logger.debug(f"[SemanticConsolidator] SelfModel sync error: {e}")

    def _consolidate_capabilities(self) -> int:
        """
        Re-compute long-term trends from capability history.
        Currently just smooths any outlier values.
        """
        caps = self._memory.get_capabilities()
        updated = 0
        for name, data in caps.items():
            conf  = data["confidence"]
            trend = data["trend"]
            # Clamp extreme trends
            if abs(trend) > 0.3:
                self._memory.update_capability(name, conf)  # resets trend smoothing
                updated += 1
        return updated
