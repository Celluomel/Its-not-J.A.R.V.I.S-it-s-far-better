"""
LUMINA V32 — Phase 6 : Goal Consolidator
=========================================
The goal pool currently has 9 'opportunity_*_unknown' variants that are
semantically identical. This module:

  1. Groups goals by semantic similarity (word overlap + origin)
  2. Merges duplicates into a single representative goal
  3. Renames poorly-named goals using the best available description
  4. Archives goals that are truly redundant

Run periodically from internal_loop to keep the goal pool clean.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.data.access import DataAccess
from core.data.schemas import make_goal

logger = logging.getLogger(__name__)

# Patterns that indicate generic/noise goal names
NOISE_PATTERNS = [
    r'^opportunity_.*_unknown$',
    r'^resolve_.*_unknown$',
    r'^explore_.*_unknown$',
    r'_unknown$',
    r'^(thought|insight)_[0-9a-f]{6,}$',
]

# Minimum word overlap ratio to consider two goals as duplicates
MERGE_THRESHOLD = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_words(name: str) -> Set[str]:
    """Extract meaningful words from a goal name slug."""
    stop = {'opportunity', 'resolve', 'explore', 'understand', 'unknown',
            'thought', 'insight', 'tension', 'uncertainty', 'the', 'and',
            'for', 'with', 'from', 'that', 'this', 'into'}
    words = set(re.split(r'[_\s]+', name.lower()))
    return words - stop - {''}


def _is_noise_name(name: str) -> bool:
    for pat in NOISE_PATTERNS:
        if re.match(pat, name):
            return True
    return False


def _similarity(a: str, b: str) -> float:
    wa, wb = _slug_words(a), _slug_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class GoalConsolidator:
    """
    Deduplicates and renames goals in the DAL goal pool.
    """

    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.dal = DataAccess(str(persona_dir))

    def run(self) -> Dict[str, Any]:
        """Run consolidation pass. Returns report."""
        report = {
            'merged':   [],
            'renamed':  [],
            'archived': [],
        }

        goals = self.dal.get_goals()
        active = [g for g in goals if g.get('status') == 'active']

        # Step 1: Group by origin cluster
        clusters = self._cluster(active)

        # Step 2: Merge each cluster → keep highest-priority, archive rest
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            merged, archived = self._merge_cluster(cluster)
            if merged:
                report['merged'].append(merged['name'])
            report['archived'].extend(a['name'] for a in archived)

        # Step 3: Rename noise-named goals that survived
        goals_after = self.dal.get_goals(status='active')
        for goal in goals_after:
            new_name = self._suggest_rename(goal)
            if new_name and new_name != goal.get('name'):
                old_name = goal['name']
                goal['name'] = new_name
                goal['renamed_from'] = old_name
                goal['renamed_at'] = _now()
                self.dal.save_goal(goal)
                report['renamed'].append(f"{old_name} → {new_name}")
                logger.info(f"[GoalConsolidator] Renamed: {old_name} → {new_name}")

        report['summary'] = {
            'merged':   len(report['merged']),
            'renamed':  len(report['renamed']),
            'archived': len(report['archived']),
        }
        return report

    # ── Clustering ────────────────────────────────────────────────────────────

    def _cluster(self, goals: List[Dict]) -> List[List[Dict]]:
        """Group semantically similar goals into clusters."""
        used: Set[str] = set()
        clusters: List[List[Dict]] = []

        for i, g1 in enumerate(goals):
            if g1['id'] in used:
                continue
            cluster = [g1]
            used.add(g1['id'])

            for g2 in goals[i + 1:]:
                if g2['id'] in used:
                    continue
                if self._should_merge(g1, g2):
                    cluster.append(g2)
                    used.add(g2['id'])

            clusters.append(cluster)

        return clusters

    def _should_merge(self, a: Dict, b: Dict) -> bool:
        """True if two goals are close enough to merge."""
        na, nb = a.get('name', ''), b.get('name', '')

        # Both noise-named with same origin → definitely merge
        if _is_noise_name(na) and _is_noise_name(nb):
            oa, ob = a.get('origin', ''), b.get('origin', '')
            if oa == ob:
                return True

        # High word overlap
        sim = _similarity(na, nb)
        if sim >= MERGE_THRESHOLD:
            return True

        # Same concept field
        ca, cb = a.get('concept', ''), b.get('concept', '')
        if ca and cb and ca.lower() == cb.lower():
            return True

        return False

    # ── Merging ───────────────────────────────────────────────────────────────

    def _merge_cluster(
        self, cluster: List[Dict]
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Keep the best goal in a cluster; archive the rest.
        'Best' = highest priority + best name quality.
        """
        # Deduplicate by id first — prevent self-merging
        seen_ids: set = set()
        unique_cluster = []
        for g in cluster:
            gid = g.get('id', '')
            if gid not in seen_ids:
                seen_ids.add(gid)
                unique_cluster.append(g)
        cluster = unique_cluster

        if len(cluster) < 2:
            return None, []

        # Score each goal: priority + name quality
        def score(g: Dict) -> float:
            pri  = g.get('priority', 0.5)
            name = g.get('name', '')
            name_q = 0.0 if _is_noise_name(name) else 0.3
            qs = g.get('quality_score', 0.5)
            return pri + name_q + qs * 0.2

        ranked = sorted(cluster, key=score, reverse=True)
        keeper  = ranked[0]
        victims = ranked[1:]

        # Merge origin_thoughts from victims into keeper
        all_origins = list(keeper.get('origin_thoughts', []))
        for v in victims:
            all_origins.extend(v.get('origin_thoughts', []))
        keeper['origin_thoughts'] = all_origins
        keeper['merged_from'] = [v['id'] for v in victims]
        keeper['consolidated_at'] = _now()

        # Boost priority slightly (merged goals are more important)
        keeper['priority'] = min(1.0, keeper.get('priority', 0.5) + 0.05)
        self.dal.save_goal(keeper)

        # Archive victims
        for v in victims:
            v['status'] = 'archived'
            v['archived_reason'] = f"merged_into:{keeper['id']}"
            v['archived_at'] = _now()
            self.dal.save_goal(v)
            logger.info(f"[GoalConsolidator] Archived {v['name']} → merged into {keeper['name']}")

        return keeper, victims

    # ── Renaming ──────────────────────────────────────────────────────────────

    def _suggest_rename(self, goal: Dict) -> Optional[str]:
        """
        Suggest a better name for a noise-named goal, or None if name is fine.
        """
        name = goal.get('name', '')
        if not _is_noise_name(name):
            return None

        # Try to build a name from available fields
        concept = goal.get('concept', '')
        topic   = goal.get('topic', '')
        gtype   = goal.get('type', goal.get('goal_type', ''))
        tension = goal.get('tension_source', '')

        # Priority order for name construction
        parts = []
        if gtype and gtype not in ('unknown', 'opportunity', 'resolve', 'explore'):
            parts.append(gtype)

        if concept and concept not in ('unknown', '') and not _is_noise_name(concept):
            parts.append(concept.replace(' ', '_'))
        elif topic and topic not in ('unknown', ''):
            # Extract meaningful words from topic
            words = [w for w in re.split(r'\s+', topic.lower())
                     if len(w) >= 4 and w not in ('with', 'from', 'that', 'this', 'into', 'about')]
            if words:
                parts.extend(words[:2])

        if not parts and tension:
            parts = [tension.replace('_drive', '').replace('_pressure', ''), 'exploration']

        if not parts:
            # Use origin_thoughts content if available
            origins = goal.get('origin_thoughts', [])
            if origins:
                parts = ['explore', 'thought_insight']

        if parts:
            new_name = '_'.join(p.lower() for p in parts if p)
            # Truncate if too long
            if len(new_name) > 50:
                new_name = new_name[:50]
            return new_name

        return None


if __name__ == '__main__':
    gc = GoalConsolidator()
    r = gc.run()
    print(f"\nGoal consolidation:")
    print(f"  Merged  : {r['summary']['merged']} groups")
    print(f"  Renamed : {r['summary']['renamed']}")
    print(f"  Archived: {r['summary']['archived']}")
    if r['renamed']:
        for ren in r['renamed']:
            print(f"    {ren}")
