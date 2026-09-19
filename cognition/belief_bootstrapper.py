"""
PANDORABOX V32 — Phase 4.2 : Belief Bootstrapper
=============================================
Extracts meaningful beliefs from existing rich data sources and
populates identity.json.

Sources mined:
  - emotional_state.json   → stable high-value emotions become beliefs
  - self_concept.json      → expressed beliefs from conversations
  - tensions.json          → persistent tensions = belief conflicts
  - homeostasis.json       → cognitive stability patterns
  - thought_stream.json    → high-priority insights

Run once, then ongoing via internal_loop integration.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.data.access import DataAccess
from core.data.schemas import make_belief

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BeliefBootstrapper:
    """
    Mines existing persona data to populate identity.json with
    meaningful beliefs that reflect PandoraBOX's actual cognitive state.
    """

    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.dal = DataAccess(str(persona_dir))

    def run(self) -> Dict[str, Any]:
        """Bootstrap all belief sources. Returns report."""
        report = {'sources': {}, 'total_added': 0, 'total_updated': 0}

        sources = [
            ('emotions',     self._extract_from_emotions),
            ('self_concept', self._extract_from_self_concept),
            ('tensions',     self._extract_from_tensions),
            ('thoughts',     self._extract_from_thoughts),
        ]

        for name, extractor in sources:
            try:
                added, updated = extractor()
                report['sources'][name] = {'added': added, 'updated': updated}
                report['total_added']   += added
                report['total_updated'] += updated
                logger.info(f"[BeliefBoot] {name}: +{added} added, {updated} updated")
            except Exception as e:
                report['sources'][name] = {'error': str(e)}
                logger.warning(f"[BeliefBoot] {name} failed: {e}")

        total = len(self.dal.get_beliefs())
        report['total_beliefs_now'] = total
        return report

    # ── Emotion → Beliefs ─────────────────────────────────────────────────────

    def _extract_from_emotions(self) -> tuple:
        """Stable, high-value emotions become core beliefs."""
        p = self.persona_dir / 'emotional_state.json'
        if not p.exists():
            return 0, 0

        with open(p) as f:
            data = json.load(f)
        emotions = data.get('emotions', {})

        added = updated = 0
        for emo_name, emo_data in emotions.items():
            if not isinstance(emo_data, dict):
                continue
            value    = emo_data.get('value', 0.0)
            baseline = emo_data.get('baseline', 0.0)

            # Only stable emotions (value ≈ baseline) at meaningful levels
            stability = 1.0 - abs(value - baseline)
            if value < 0.3 or stability < 0.7:
                continue

            belief_name = f"emotional_core_{emo_name}"
            existing = self.dal.get_belief_by_name(belief_name)

            bel = make_belief(
                name=belief_name,
                value=round(value, 3),
                confidence=round(stability * 0.8, 3),
                source_thread=None,
            )
            bel['category']    = 'emotional_core'
            bel['emotion']     = emo_name
            bel['baseline']    = round(baseline, 3)
            bel['description'] = f"Core emotional disposition: {emo_name} (stable at {value:.2f})"

            self.dal.save_belief(bel)
            if existing:
                updated += 1
            else:
                added += 1

        return added, updated

    # ── Self-Concept → Beliefs ────────────────────────────────────────────────

    def _extract_from_self_concept(self) -> tuple:
        """Extract expressed beliefs from self_concept.json."""
        p = self.persona_dir / 'self_concept.json'
        if not p.exists():
            return 0, 0

        with open(p) as f:
            data = json.load(f)

        added = updated = 0

        # Expressed beliefs from introspective statements
        beliefs_raw = data.get('beliefs', {})
        if isinstance(beliefs_raw, dict):
            items = beliefs_raw.values()
        else:
            items = beliefs_raw

        for raw in items:
            if not isinstance(raw, dict):
                continue
            stmt = raw.get('statement', '').strip()
            if len(stmt) < 10:
                continue

            # Skip semantic hub concepts — these belong in semantic_memory,
            # not identity.json. They inflate the identity store with graph
            # nodes (semantic_hub_goal, semantic_hub_explore, etc.) that are
            # corpus statistics, not self-beliefs.
            _raw_name = raw.get('name', '')
            if _raw_name.startswith('semantic_hub_'):
                continue

            conf = raw.get('confidence', 0.5)
            if conf < 0.2:
                continue

            belief_name = f"expressed_{raw.get('name', stmt[:20].replace(' ', '_'))}"
            existing = self.dal.get_belief_by_name(belief_name)

            bel = make_belief(
                name=belief_name,
                value=round((conf + 0.5) / 2, 3),
                confidence=round(conf, 3),
            )
            bel['category']    = 'expressed_belief'
            bel['statement']   = stmt
            bel['valence']     = raw.get('valence', 'neutral')
            bel['description'] = f"Expressed: {stmt[:60]}"

            self.dal.save_belief(bel)
            if existing:
                updated += 1
            else:
                added += 1

        # State coherence as a meta-belief
        state = data.get('state', {})
        coherence = state.get('coherence', None)
        if coherence is not None:
            bel = make_belief(
                name='identity_coherence',
                value=round(float(coherence), 3),
                confidence=0.7,
            )
            bel['category']    = 'meta'
            bel['description'] = f"Self-model coherence level: {coherence:.2f}"
            self.dal.save_belief(bel)
            added += 1

        return added, updated

    # ── Tensions → Belief Conflicts ───────────────────────────────────────────

    def _extract_from_tensions(self) -> tuple:
        """Persistent tensions signal unresolved belief conflicts."""
        tensions = self.dal.get_tensions()
        added = updated = 0

        TENSION_DESCRIPTIONS = {
            'curiosity_drive':        "Strong drive to explore and understand the world",
            'contradiction_pressure': "Awareness of internal contradictions needing resolution",
            'goal_pressure':          "Pressure from multiple active goals competing for attention",
            'identity_stress':        "Tension in self-model — identity is actively evolving",
            'social_drive':           "Drive toward meaningful connection and understanding others",
            'knowledge_uncertainty':  "Awareness of gaps in knowledge that create discomfort",
        }

        for tension_key, level in tensions.items():
            if level < 0.2:
                continue
            belief_name = f"tension_{tension_key}"
            existing    = self.dal.get_belief_by_name(belief_name)

            desc = TENSION_DESCRIPTIONS.get(tension_key, f"Persistent tension: {tension_key}")
            bel = make_belief(
                name=belief_name,
                value=round(level, 3),
                confidence=0.8,
            )
            bel['category']    = 'tension_belief'
            bel['tension_key'] = tension_key
            bel['description'] = desc

            self.dal.save_belief(bel)
            if existing:
                updated += 1
            else:
                added += 1

        return added, updated

    # ── High-priority Thoughts → Insight Beliefs ─────────────────────────────

    def _extract_from_thoughts(self) -> tuple:
        """High-priority insights become beliefs."""
        thoughts = self.dal.get_thoughts(limit=100, skip_expired=True)
        added = updated = 0

        for t in thoughts:
            if t.get('thought_type') not in ('insight', 'meta', 'self_observation'):
                continue
            if t.get('priority', 0) < 0.7:
                continue
            content = t.get('content', '').strip()
            if len(content) < 15:
                continue

            tid = t.get('id', '')
            belief_name = f"insight_{tid[:8]}" if tid else f"insight_{content[:15].replace(' ','_')}"
            if self.dal.get_belief_by_name(belief_name):
                continue  # already exists

            bel = make_belief(
                name=belief_name,
                value=round(t.get('priority', 0.6), 3),
                confidence=0.5,
                source_thread=t.get('linked_goal'),
            )
            bel['category']    = 'insight'
            bel['statement']   = content[:120]
            bel['thought_id']  = tid
            bel['description'] = f"Insight: {content[:60]}"

            self.dal.save_belief(bel)
            added += 1

        return added, updated


if __name__ == '__main__':
    bb = BeliefBootstrapper()
    report = bb.run()
    print('\n=== BELIEF BOOTSTRAP REPORT ===')
    for src, result in report['sources'].items():
        if 'error' in result:
            print(f"  {src:<16} ❌ {result['error']}")
        else:
            print(f"  {src:<16} +{result['added']} added, {result['updated']} updated")
    print(f"\n  Total beliefs now: {report['total_beliefs_now']}")
