"""
LUMINA V32 — Phase 5.1 : Self-Concept Synchronizer
====================================================
Bridges the gap between:
  - identity.json  (beliefs accumulated by Phase 4)
  - self_concept.json (the self-model Lumina uses in responses)

Self-concept coherence is currently 0.286 because the two stores
are disconnected. This module syncs them and rebuilds coherence.

Also extracts milestones from life_story (currently 50 entries, 0 milestones).
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.data.access import DataAccess

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SelfConceptSynchronizer:
    """
    Synchronizes identity.json → self_concept.json and
    extracts milestones from life_story.
    """

    COHERENCE_WEIGHTS = {
        'emotional_core':   0.25,
        'expressed_belief': 0.30,
        'tension_belief':   0.20,
        'thread_learning':  0.15,
        'insight':          0.10,
    }

    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.dal = DataAccess(str(persona_dir))

    def run(self) -> Dict[str, Any]:
        report = {
            'beliefs_synced':  0,
            'milestones_found': 0,
            'coherence_before': 0.0,
            'coherence_after':  0.0,
        }

        sc = self._load_json('self_concept.json')
        # coherence_before: use the actual saved state from disk.
        # The old cache restoration (cached > current+0.05) caused a misleading
        # 'before' reading every session when the cache held a past better value.
        # The live computation is ground truth — don't override it with stale cache.
        report['coherence_before'] = sc.get('state', {}).get('coherence', 0.0)

        # 1. Sync beliefs → self_concept.beliefs
        synced = self._sync_beliefs(sc)
        report['beliefs_synced'] = synced

        # 2. Recompute coherence
        new_coherence = self._compute_coherence(sc)
        sc.setdefault('state', {})['coherence'] = new_coherence
        sc['state']['last_updated'] = _now()
        sc['state']['confidence'] = min(0.85, new_coherence + 0.1)
        report['coherence_after'] = new_coherence

        # 3. Extract milestones from life_story
        milestones = self._extract_milestones()
        if milestones:
            sc['state']['milestones'] = milestones
            report['milestones_found'] = len(milestones)

        # 4. Build expressed_values from emotional core beliefs
        expressed_values = self._build_expressed_values(sc)
        if expressed_values:
            sc['state']['expressed_values'] = expressed_values

        # 5. Feed coherence back into stability — this directly reduces identity_stress
        # tension_engine uses: raw_identity = (1 - identity_stability)*0.7 + contradiction*0.3
        # So raising stability from 0.5 → 0.7 drops identity_stress from ~0.84 → ~0.70
        old_stability = sc.get('state', {}).get('stability', 0.5)
        # Stability converges toward coherence slowly (smoothing factor 0.85)
        new_stability = round(0.85 * old_stability + 0.15 * new_coherence, 4)
        sc['state']['stability'] = new_stability
        report['stability_before'] = old_stability
        report['stability_after']  = new_stability
        # Cache coherence so it survives restarts (SelfConcept resets it on load)
        sc['_scs_coherence_cache'] = new_coherence

        self._save_json('self_concept.json', sc)
        
        # Also write stability into tensions.json so tension_engine sees it immediately
        if new_stability != old_stability:
            try:
                tensions_path = self.persona_dir / 'tensions.json'
                if tensions_path.exists():
                    import json as _j
                    with open(tensions_path) as f:
                        t_data = _j.load(f)
                    # Recompute identity_stress from new stability
                    contradiction = t_data.get('current', {}).get('contradiction_pressure', 0.1)
                    new_id_stress = round((1.0 - new_stability) * 0.7 + contradiction * 0.3, 4)
                    t_data.setdefault('current', {})['identity_stress'] = new_id_stress
                    import os
                    tmp = tensions_path.with_suffix('.tmp')
                    with open(tmp, 'w') as f:
                        _j.dump(t_data, f, indent=2)
                    os.replace(tmp, tensions_path)
            except Exception as _e:
                pass
        # ── Reload the live in-memory SelfConceptSystem ──────────────────────
        # SCS previously only wrote to disk. The in-memory object that drives
        # get_inner_voice() and check_response_alignment() never saw the updates,
        # so syncing was effectively a no-op for behavior. We now reload the
        # live object if it is accessible through the singleton pattern.
        try:
            from cognition.self_concept import SelfConceptSystem as _SCS
            # Walk all live SCS instances registered on the AISystem/organism
            # via the module-level registry we add below, or via the ai_system ref.
            reloaded = False
            # Strategy 1: check if a global registry exists
            _registry = getattr(_SCS, '_live_instances', [])
            for _inst in list(_registry):
                try:
                    n = _inst.reload_from_disk()
                    # Tell the live instance that SCS just set coherence so
                    # _update_coherence() won't overwrite it for 4 minutes
                    if hasattr(_inst, 'mark_scs_coherence'):
                        _inst.mark_scs_coherence(new_coherence)
                    logger.debug(f"[SCS] Reloaded live SCS instance: {n} beliefs")
                    reloaded = True
                except Exception:
                    pass
            if not reloaded:
                logger.debug("[SCS] No live SCS instance found to reload (will take effect on next load)")
        except Exception as _re:
            logger.debug(f"[SCS] Live reload skipped: {_re}")

        pruned_count = len([b for b in sc.get('beliefs', {}).values()
                            if isinstance(b, dict)])
        logger.info(
            f"[SCS] Coherence: {report['coherence_before']:.3f} → {new_coherence:.3f} "
            f"({synced} beliefs synced, {pruned_count} total, "
            f"{len(milestones)} milestones)"
        )
        return report

    # ── Belief sync ───────────────────────────────────────────────────────────

    # Minimum confidence for a belief to survive a sync pass
    BELIEF_MIN_CONFIDENCE = 0.35
    # How much to decay confidence per sync for unsupported beliefs
    BELIEF_DECAY_PER_SYNC = 0.02
    # Decay starts after this many days without reinforcement
    BELIEF_DECAY_AFTER_DAYS = 30  # beliefs survive 30 days without reinforcement
    # Hard cap on total beliefs stored (prune lowest if exceeded)
    BELIEF_MAX_COUNT = 40
    # Categories that are too noisy to sync from identity.json
    BELIEF_SKIP_CATEGORIES = frozenset({'thread_learning', 'raw_extraction', 'noise'})

    def _sync_beliefs(self, sc: Dict) -> int:
        """
        Merge identity.json beliefs into self_concept.beliefs dict.

        Three additions vs the original:
        1. Quality gate: only sync beliefs with confidence >= BELIEF_MIN_CONFIDENCE
        2. Confidence decay: beliefs not reinforced in BELIEF_DECAY_AFTER_DAYS
           lose BELIEF_DECAY_PER_SYNC per sync pass — weak beliefs die naturally
        3. Hard cap: if belief count exceeds BELIEF_MAX_COUNT, prune the weakest
        """
        import time as _time
        _now_ts = _time.time()
        _decay_cutoff = _now_ts - self.BELIEF_DECAY_AFTER_DAYS * 86400

        beliefs = self.dal.get_beliefs()
        # Build a lookup by name for fast dedup
        incoming = {
            b.get('name', ''): b for b in (beliefs or [])
            if b.get('name') and b.get('category') not in self.BELIEF_SKIP_CATEGORIES
            and b.get('confidence', 0) >= self.BELIEF_MIN_CONFIDENCE
        }

        existing = sc.get('beliefs', {})
        if isinstance(existing, list):
            existing = {b.get('name', str(i)): b for i, b in enumerate(existing)}

        synced = 0

        # ── Pass 1: decay existing beliefs that haven't been reinforced ──────
        pruned = []
        for key, eb in list(existing.items()):
            # Skip metadata fields stored at the same level as beliefs
            if not isinstance(eb, dict):
                continue
            # Parse synced_at to check age
            try:
                from datetime import datetime as _dt
                synced_at_str = eb.get('synced_at', '')
                if synced_at_str:
                    # Handle both ISO format and plain timestamps
                    if 'T' in synced_at_str:
                        synced_ts = _dt.fromisoformat(
                            synced_at_str.replace('Z', '+00:00')
                        ).timestamp()
                    else:
                        synced_ts = float(synced_at_str)
                else:
                    synced_ts = _decay_cutoff  # treat missing as stale
            except Exception:
                synced_ts = _decay_cutoff

            # If the incoming set has a fresher version, don't decay
            if key in incoming:
                continue

            # Decay if stale (not in incoming AND older than cutoff)
            if synced_ts < _decay_cutoff:
                old_conf = eb.get('confidence', 0.5)
                new_conf = round(old_conf - self.BELIEF_DECAY_PER_SYNC, 4)
                if new_conf < self.BELIEF_MIN_CONFIDENCE:
                    pruned.append(key)
                else:
                    existing[key]['confidence'] = new_conf

        for key in pruned:
            del existing[key]
        if pruned:
            logger.info(f"[SCS] Pruned {len(pruned)} decayed beliefs: {pruned[:5]}")

        # ── Pass 2: merge incoming beliefs ────────────────────────────────────
        for name, b in incoming.items():
            cat = b.get('category', '')
            key = name
            # Normalise belief name: strip repeated "expressed_" prefix chains
            # that accumulate when BeliefBootstrapper reads self_concept names
            # and prepends "expressed_" each cycle without checking.
            import re as _re
            _norm_name = _re.sub(r'^(expressed_)+', 'expressed_', name)
            # Also normalise the key (used as dict key in self_concept.json)
            _norm_key = _re.sub(r'^(expressed_)+', 'expressed_', key)

            if _norm_key not in existing:
                existing[_norm_key] = {
                    'name':       _norm_name,
                    'statement':  b.get('statement', b.get('description', _norm_name)),
                    'confidence': b.get('confidence', 0.5),
                    'value':      b.get('value', 0.5),
                    'category':   cat,
                    'valence':    b.get('valence', 'positive'),
                    'source':     b.get('source', 'identity.json'),
                    'synced_at':  _now(),
                }
                synced += 1
            else:
                # Update confidence if improved
                if b.get('confidence', 0) > existing[_norm_key].get('confidence', 0):
                    existing[_norm_key]['confidence'] = b['confidence']
                    synced += 1
                existing[_norm_key]['name'] = _norm_name  # repair corrupted name in place
                existing[_norm_key].setdefault('valence', b.get('valence', 'positive'))
                existing[_norm_key].setdefault('source',  b.get('source',  'identity.json'))
                # KEY FIX: always refresh synced_at when belief is in incoming.
                # Previously only refreshed on confidence improvement — so 26 beliefs
                # that existed with unchanged confidence never got synced_at updated,
                # were treated as stale on the very next SCS run, and decayed every
                # single cycle. This caused the persistent 0.658 → 0.521 coherence drop.
                existing[_norm_key]['synced_at'] = _now()

        # ── Pass 3: hard cap — prune weakest if over limit ────────────────────
        if len(existing) > self.BELIEF_MAX_COUNT:
            sorted_keys = sorted(
                existing.keys(),
                key=lambda k: existing[k].get('confidence', 0)
            )
            overflow = len(existing) - self.BELIEF_MAX_COUNT
            for k in sorted_keys[:overflow]:
                del existing[k]
            logger.info(
                f"[SCS] Hard cap: pruned {overflow} weakest beliefs "
                f"(kept top {self.BELIEF_MAX_COUNT})"
            )

        sc['beliefs'] = existing
        return synced

    # ── Coherence ─────────────────────────────────────────────────────────────

    def _compute_coherence(self, sc: Dict) -> float:
        """
        Coherence = weighted average of belief confidence per category.
        More high-confidence beliefs of diverse categories = higher coherence.
        """
        beliefs = sc.get('beliefs', {})
        if isinstance(beliefs, dict):
            # self_concept.json has metadata keys at the top level
            # (confidence, version, last_validated, last_updated) — skip them
            belief_list = [v for v in beliefs.values() if isinstance(v, dict)]
        else:
            belief_list = [b for b in beliefs if isinstance(b, dict)]

        if not belief_list:
            return 0.1

        # Group by category
        by_cat: Dict[str, List[float]] = {}
        for b in belief_list:
            if not isinstance(b, dict):
                continue
            cat  = b.get('category', 'other')
            conf = b.get('confidence', 0.5)
            by_cat.setdefault(cat, []).append(conf)

        if not by_cat:
            return 0.1

        # Weighted score
        total_weight = 0.0
        total_score  = 0.0
        for cat, confs in by_cat.items():
            w     = self.COHERENCE_WEIGHTS.get(cat, 0.05)
            score = sum(confs) / len(confs)
            total_score  += w * score
            total_weight += w

        base = total_score / total_weight if total_weight > 0 else 0.3

        # Diversity bonus: more categories = higher coherence
        diversity = min(1.0, len(by_cat) / 5.0)
        coherence = base * 0.7 + diversity * 0.3

        return round(max(0.0, min(1.0, coherence)), 4)

    # ── Milestones ────────────────────────────────────────────────────────────

    def _extract_milestones(self) -> List[Dict]:
        """
        Extract significant milestones from narrative_identity.life_story.
        Criteria: entries with 'Deep conversation', unique insight, or high word count.
        """
        ni = self._load_json('narrative_identity.json')
        life_story = ni.get('life_story', [])

        milestones = []
        seen_titles = set()

        for entry in life_story:
            if not isinstance(entry, dict):
                continue
            title = entry.get('title', '')
            desc  = str(entry.get('description', ''))
            ts    = entry.get('timestamp', 0)

            # Select significant entries
            is_milestone = (
                'deep' in title.lower() or
                'insight' in title.lower() or
                'explored' in title.lower() or
                len(desc) > 80
            )

            if is_milestone and title not in seen_titles:
                seen_titles.add(title)
                milestones.append({
                    'title':     title,
                    'timestamp': ts,
                    'summary':   desc[:100],
                    'type':      'conversation_milestone',
                })

        # Keep top 10 most recent
        milestones.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        # Update narrative_identity.milestones too
        if milestones:
            ni['milestones'] = milestones[:10]
            self._save_json('narrative_identity.json', ni)

        return milestones[:10]

    # ── Expressed values ──────────────────────────────────────────────────────

    def _build_expressed_values(self, sc: Dict) -> List[str]:
        """Extract value statements from emotional_core and expressed_belief categories."""
        beliefs = sc.get('beliefs', {})
        if isinstance(beliefs, dict):
            belief_list = list(beliefs.values())
        else:
            belief_list = beliefs

        values = []
        for b in belief_list:
            if not isinstance(b, dict):
                continue
            cat  = b.get('category', '')
            conf = b.get('confidence', 0.0)
            if cat in ('emotional_core', 'expressed_belief') and conf >= 0.5:
                stmt = b.get('statement', b.get('description', ''))
                if stmt and len(stmt) > 8:
                    values.append(stmt[:80])

        return values[:8]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_json(self, filename: str) -> Dict:
        p = self.persona_dir / filename
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_json(self, filename: str, data: Dict):
        import os
        p = self.persona_dir / filename
        tmp = p.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)


if __name__ == '__main__':
    scs = SelfConceptSynchronizer()
    r = scs.run()
    print(f"\nSelf-concept sync:")
    print(f"  Beliefs synced   : {r['beliefs_synced']}")
    print(f"  Milestones found : {r['milestones_found']}")
    print(f"  Coherence        : {r['coherence_before']:.3f} → {r['coherence_after']:.3f}")
