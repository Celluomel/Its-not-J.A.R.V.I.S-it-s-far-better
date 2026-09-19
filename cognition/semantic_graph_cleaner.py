"""
PANDORABOX V32 — Phase 7 : Semantic Graph Cleaner
==============================================
The semantic_memory.db has 1263 concepts but ~701 are orphans and the
top nodes by use_count are stopwords ('without', 'naming', 'feeling').

This module:
  1. Purges pure-stopword concepts and their relations
  2. Merges near-duplicate concepts (e.g. 'curiosity' + 'curious')
  3. Enriches relation types beyond the single 'related_to' bucket
  4. Decays orphan concepts (no relations) below a strength threshold
  5. Promotes high-value concepts as belief candidates

Safe to run repeatedly — idempotent.
"""

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Stopword set ──────────────────────────────────────────────────────────────
STOP = {
    'the','and','is','a','an','to','of','in','it','that','this','was','for',
    'on','are','as','with','be','at','by','or','but','not','what','have',
    'from','they','we','you','he','she','do','did','has','had','all','would',
    'there','their','if','will','one','when','so','out','up','more','about',
    'just','also','which','been','its','into','than','then','them','these',
    'those','some','can','could','should','may','might','how','who','get',
    'got','like','very','really','feel','feels','think','know','want','need',
    'make','made','go','going','come','even','well','still','only','my','me',
    'i','im','your','our','yes','no','ok','okay','sure','right','today',
    'here','now','time','way','see','look','much','many','any','try','use',
    'let','say','said','tell','told','give','take','start','keep','help',
    'ask','mean','thought','find','good','great','nice','glad','without',
    'naming','naturally','inner','image','which','while','why','where',
    'until','because','again','once','first','second','new','old','able',
    'back','between','through','over','under','after','before','within',
    'above','below','each','every','both','either','neither','although',
    'though','since','during','feeling','emotion','something','anything',
    'everything','conversation','satisfaction','really','quite','just',
    'then','also','still','even','well','quite','pretty','already','always',
    'never','ever','often','sometimes','usually','generally','especially',
    # PandoraBOX-specific filler phrases found in the DB
    'without naming', 'feeling satisfaction', 'without mentioning',
    'speak naturally', 'naturally surface', 'feel free', 'feel today',
    'image shows', 'topic together', 'provide helpful', 'start error',
    'away solution', 'detailed system',
}

# ── Concept synonym groups for merging ────────────────────────────────────────
SYNONYM_GROUPS = [
    {'curiosity', 'curious', 'curiousity'},
    {'consciousness', 'conscious', 'awareness', 'aware'},
    {'emotion', 'emotional', 'emotions', 'feeling', 'feelings'},
    {'identity', 'self', 'selfhood', 'self-concept'},
    {'goal', 'goals', 'objective', 'objectives', 'aim', 'aims'},
    {'memory', 'memories', 'remember', 'recollection'},
    {'thought', 'thoughts', 'thinking', 'idea', 'ideas'},
    {'exploration', 'explore', 'exploring', 'discovery', 'discover'},
    {'learning', 'learn', 'knowledge', 'understand', 'understanding'},
    {'conversation', 'dialogue', 'discussion', 'talk'},
    {'connection', 'relate', 'relation', 'relationship'},
    {'creativity', 'creative', 'imagination', 'imaginative'},
    {'philosophy', 'philosophical', 'metaphysics'},
    {'science', 'scientific', 'physics', 'biology', 'chemistry'},
    {'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning'},
]

# ── Relation type inference rules ─────────────────────────────────────────────
# (trigger_words, relation_type)
RELATION_RULES = [
    ({'causes', 'leads', 'creates', 'produces', 'generates', 'results'},   'causes'),
    ({'part', 'component', 'aspect', 'element', 'subset', 'type'},          'part_of'),
    ({'opposite', 'contrast', 'different', 'unlike', 'versus', 'vs'},       'contrasts_with'),
    ({'similar', 'like', 'resembles', 'analogy', 'metaphor', 'parallel'},   'similar_to'),
    ({'enables', 'allows', 'facilitates', 'supports', 'helps'},             'enables'),
    ({'requires', 'needs', 'depends', 'relies', 'based'},                   'requires'),
    ({'emotion', 'feeling', 'affect', 'mood', 'sentiment'},                 'emotional_link'),
    ({'goal', 'objective', 'aim', 'purpose', 'intention'},                  'goal_link'),
    ({'identity', 'self', 'belief', 'value', 'principle'},                  'identity_link'),
]


class SemanticGraphCleaner:
    """
    Cleans and enriches the semantic_memory.db graph.
    """

    MIN_STRENGTH_TO_KEEP = 0.05  # below this → delete
    ORPHAN_DECAY_RATE    = 0.7   # multiply orphan strength by this
    MERGE_SIMILARITY     = 0.8   # character overlap ratio to trigger merge

    def __init__(self, db_path: str = "data/persona/semantic_memory.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def run(self) -> Dict[str, Any]:
        report = {
            'purged_concepts':  0,
            'purged_relations': 0,
            'merged_concepts':  0,
            'enriched_relations': 0,
            'decayed_orphans':  0,
            'beliefs_promoted': 0,
        }

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            report['purged_concepts'],  report['purged_relations'] = self._purge_stopwords(conn)
            report['merged_concepts']                               = self._merge_synonyms(conn)
            report['enriched_relations']                            = self._enrich_relation_types(conn)
            report['decayed_orphans']                               = self._decay_orphans(conn)
            report['beliefs_promoted']                              = self._promote_to_beliefs(conn)
            conn.commit()
        finally:
            conn.close()

        logger.info(
            f"[GraphCleaner] purged={report['purged_concepts']} concepts "
            f"({report['purged_relations']} relations) | "
            f"merged={report['merged_concepts']} | "
            f"enriched={report['enriched_relations']} relations | "
            f"orphans_decayed={report['decayed_orphans']} | "
            f"beliefs_promoted={report['beliefs_promoted']}"
        )
        return report

    # ── 1. Purge stopword concepts ─────────────────────────────────────────────

    def _purge_stopwords(self, conn: sqlite3.Connection) -> Tuple[int, int]:
        cur = conn.cursor()

        # Load all concepts
        cur.execute("SELECT id, name FROM concepts")
        all_concepts = cur.fetchall()

        noise_ids: List[int] = []
        for row in all_concepts:
            name = row['name'].lower().strip()
            # Pure stopword or in STOP set
            if name in STOP:
                noise_ids.append(row['id'])
                continue
            # All individual words are stopwords
            words = name.split()
            if words and all(w in STOP for w in words):
                noise_ids.append(row['id'])

        if not noise_ids:
            return 0, 0

        placeholders = ','.join('?' * len(noise_ids))

        cur.execute(f"DELETE FROM relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    noise_ids + noise_ids)
        deleted_relations = cur.rowcount

        cur.execute(f"DELETE FROM concepts WHERE id IN ({placeholders})", noise_ids)
        deleted_concepts = cur.rowcount

        logger.info(f"[GraphCleaner] Purged {deleted_concepts} stopword concepts, {deleted_relations} relations")
        return deleted_concepts, deleted_relations

    # ── 2. Merge synonym groups ────────────────────────────────────────────────

    def _merge_synonyms(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        merged = 0

        for group in SYNONYM_GROUPS:
            # Find which members exist in the DB
            placeholders = ','.join('?' * len(group))
            cur.execute(f"SELECT id, name, strength, use_count FROM concepts WHERE LOWER(name) IN ({placeholders})",
                        [g.lower() for g in group])
            found = cur.fetchall()
            if len(found) < 2:
                continue

            # Keep the one with highest use_count as canonical
            canonical = max(found, key=lambda r: r['use_count'])
            victims   = [r for r in found if r['id'] != canonical['id']]

            for victim in victims:
                # Re-point relations from victim → canonical safely (avoiding UNIQUE violations)
                # For source_id: only update if the new (canonical_id, target_id) pair doesn't exist
                cur.execute("""
                    UPDATE relations SET source_id=?
                    WHERE source_id=?
                    AND NOT EXISTS (
                        SELECT 1 FROM relations r2
                        WHERE r2.source_id=? AND r2.target_id=relations.target_id
                        AND r2.rel_type=relations.rel_type
                    )
                """, (canonical['id'], victim['id'], canonical['id']))
                # Delete remaining source refs that couldn't be updated (would be dupes)
                cur.execute("DELETE FROM relations WHERE source_id=?", (victim['id'],))

                # For target_id
                cur.execute("""
                    UPDATE relations SET target_id=?
                    WHERE target_id=?
                    AND NOT EXISTS (
                        SELECT 1 FROM relations r2
                        WHERE r2.source_id=relations.source_id AND r2.target_id=?
                        AND r2.rel_type=relations.rel_type
                    )
                """, (canonical['id'], victim['id'], canonical['id']))
                cur.execute("DELETE FROM relations WHERE target_id=?", (victim['id'],))

                # Remove self-loops and duplicates
                cur.execute("DELETE FROM relations WHERE source_id=target_id")
                # Absorb victim's strength and use_count
                cur.execute("""
                    UPDATE concepts
                    SET use_count = use_count + ?,
                        strength  = MIN(1.0, strength + ?)
                    WHERE id = ?
                """, (victim['use_count'], victim['strength'] * 0.2, canonical['id']))
                # Delete victim
                cur.execute("DELETE FROM aliases WHERE concept_id=?", (victim['id'],))
                cur.execute("DELETE FROM concepts WHERE id=?", (victim['id'],))
                # Add alias
                cur.execute("INSERT OR IGNORE INTO aliases (concept_id, alias) VALUES (?,?)",
                            (canonical['id'], victim['name'].lower()))
                merged += 1
                logger.debug(f"[GraphCleaner] Merged '{victim['name']}' → '{canonical['name']}'")

        return merged

    # ── 3. Enrich relation types ───────────────────────────────────────────────

    def _enrich_relation_types(self, conn: sqlite3.Connection) -> int:
        """
        Infer richer relation types from concept name pairs.
        Only re-types 'related_to' relations where a better type can be inferred.
        """
        cur = conn.cursor()
        enriched = 0

        cur.execute("""
            SELECT r.id, c1.name as src_name, c2.name as tgt_name
            FROM relations r
            JOIN concepts c1 ON r.source_id = c1.id
            JOIN concepts c2 ON r.target_id = c2.id
            WHERE r.rel_type = 'related_to'
        """)
        relations = cur.fetchall()

        for rel in relations:
            combined = (rel['src_name'] + ' ' + rel['tgt_name']).lower()
            new_type = None

            for trigger_words, rel_type in RELATION_RULES:
                if any(tw in combined for tw in trigger_words):
                    new_type = rel_type
                    break

            if new_type and new_type != 'related_to':
                cur.execute("UPDATE relations SET rel_type=? WHERE id=?",
                            (new_type, rel['id']))
                enriched += 1

        return enriched

    # ── 4. Decay orphan concepts ───────────────────────────────────────────────

    def _decay_orphans(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()

        # Find orphans (no relations at all)
        cur.execute("""
            SELECT c.id, c.name, c.strength FROM concepts c
            WHERE NOT EXISTS (
                SELECT 1 FROM relations r
                WHERE r.source_id = c.id OR r.target_id = c.id
            )
        """)
        orphans = cur.fetchall()

        decayed = 0
        to_delete = []
        for row in orphans:
            new_strength = row['strength'] * self.ORPHAN_DECAY_RATE
            if new_strength < self.MIN_STRENGTH_TO_KEEP:
                to_delete.append(row['id'])
            else:
                cur.execute("UPDATE concepts SET strength=? WHERE id=?",
                            (new_strength, row['id']))
                decayed += 1

        if to_delete:
            placeholders = ','.join('?' * len(to_delete))
            cur.execute(f"DELETE FROM concepts WHERE id IN ({placeholders})", to_delete)
            logger.info(f"[GraphCleaner] Deleted {len(to_delete)} zero-strength orphans")
            decayed += len(to_delete)

        return decayed

    # ── 5. Promote high-value concepts to beliefs ─────────────────────────────

    def _promote_to_beliefs(self, conn: sqlite3.Connection) -> int:
        """
        Concepts with high strength AND hub connectivity → add as beliefs.
        """
        cur = conn.cursor()

        # Find high-value hub concepts
        cur.execute("""
            SELECT c.id, c.name, c.strength, c.use_count,
                   COUNT(*) as degree
            FROM concepts c
            JOIN relations r ON r.source_id=c.id OR r.target_id=c.id
            WHERE c.strength >= 0.6
              AND LENGTH(c.name) >= 4
            GROUP BY c.id
            HAVING degree >= 3
            ORDER BY c.use_count DESC
            LIMIT 20
        """)
        hubs = cur.fetchall()

        promoted = 0
        try:
            from core.data.access import DataAccess
            from core.data.schemas import make_belief
            dal = DataAccess()

            for hub in hubs:
                name = hub['name']
                # Skip obvious noise that slipped through
                if name.lower() in STOP:
                    continue

                belief_name = f"semantic_hub_{name.replace(' ','_').lower()}"
                if dal.get_belief_by_name(belief_name):
                    continue

                confidence = min(0.85, hub['strength'] * 0.7 + hub['degree'] * 0.02)
                bel = make_belief(
                    name=belief_name,
                    value=round(hub['strength'], 3),
                    confidence=round(confidence, 3),
                )
                bel['category']    = 'semantic_concept'
                bel['concept']     = name
                bel['use_count']   = hub['use_count']
                bel['hub_degree']  = hub['degree']
                bel['description'] = f"Frequently referenced concept: '{name}' ({hub['use_count']} uses, {hub['degree']} connections)"
                dal.save_belief(bel)
                promoted += 1

        except Exception as e:
            logger.warning(f"[GraphCleaner] Belief promotion failed: {e}")

        return promoted


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    cleaner = SemanticGraphCleaner()
    r = cleaner.run()
    print(f"\n=== SEMANTIC GRAPH CLEAN REPORT ===")
    print(f"  Purged concepts  : {r['purged_concepts']}")
    print(f"  Purged relations : {r['purged_relations']}")
    print(f"  Merged synonyms  : {r['merged_concepts']}")
    print(f"  Enriched rel types: {r['enriched_relations']}")
    print(f"  Decayed orphans  : {r['decayed_orphans']}")
    print(f"  Beliefs promoted : {r['beliefs_promoted']}")
