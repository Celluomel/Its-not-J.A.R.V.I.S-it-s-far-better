"""
PANDORABOX V32 — Phase 4.1 : Goal Quality Filter
=============================================
Scores goals on semantic richness and tension alignment.
Marks noise goals (single keywords, low-signal) and promotes
goals that align with live tensions and emotional state.

Usage:
    from cognition.goal_quality_filter import GoalQualityFilter
    gqf = GoalQualityFilter()
    report = gqf.run()
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cognition.tension_topics import TENSION_TOPIC_GENERATORS, TENSION_GOALS

logger = logging.getLogger(__name__)

# Words that are clearly keyword-noise from NLP extraction
NOISE_WORDS = {
    'able', 'warm', 'coffee', 'caffeine', 'tackle', 'plateaued', 'personal',
    'action', 'thing', 'something', 'anything', 'question', 'statement',
    'exploration', 'discussion', 'conversation', 'topic', 'point', 'stuff',
    'bit', 'lot', 'way', 'time', 'day', 'going', 'getting', 'making',
    'work', 'help', 'need', 'want', 'feel', 'think', 'know', 'like',
    'just', 'good', 'great', 'new', 'old', 'big', 'small', 'sure', 'ok',
}

# Tension keys → goal themes that are meaningful
TENSION_THEMES = {
    'curiosity_drive':        ['understand', 'explore', 'learn', 'discover', 'investigate', 'research'],
    'identity_stress':        ['identity', 'self', 'values', 'purpose', 'meaning', 'consciousness'],
    'goal_pressure':          ['complete', 'achieve', 'resolve', 'progress', 'advance'],
    'contradiction_pressure': ['resolve', 'reconcile', 'integrate', 'clarify', 'align'],
    'social_drive':           ['connect', 'relate', 'understand_user', 'empathize', 'engage'],
    'knowledge_uncertainty':  ['understand', 'clarify', 'learn', 'research', 'resolve_uncertainty'],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GoalQualityFilter:
    """
    Analyses and improves the goal pool quality.

    Phases:
      1. Score each goal (semantic richness + tension alignment)
      2. Mark low-quality goals as 'low_quality'
      3. Boost high-alignment goals
      4. Generate replacement goals for pure noise
    """

    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.quality_threshold = 0.3   # below this → low_quality
        self.boost_threshold   = 0.6   # above this → priority boosted

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run full goal quality pass. Returns a report."""
        from core.data.access import DataAccess
        dal = DataAccess(str(self.persona_dir))

        goals   = dal.get_goals()
        tensions = dal.get_tensions()
        emotions = self._load_emotions()

        report = {
            'total': len(goals),
            'scored': [],
            'marked_noise': [],
            'boosted': [],
            'generated': [],
        }

        # Precompute embedding-based tension alignment for ALL goals in a single
        # batched embedding call (dimension-safe, one API round-trip). Returns
        # None if no embedding tier is reachable -> _score_goal transparently
        # falls back to the original static dictionary matching.
        sem_align: Optional[Dict[str, Tuple[float, str]]] = None
        try:
            from cognition import goal_semantics as _gs
            sem_align = _gs.batch_tension_alignment(
                [goal.get("name", "") for goal in goals], tensions
            )
        except Exception as e:  # noqa: BLE001 - never break the quality pass
            logger.warning(f"[GQF] semantic alignment failed ({e}); using dictionary fallback")
            sem_align = None
        if sem_align is not None:
            logger.info(f"[GQF] using EMBEDDING-based tension alignment for {len(goals)} goal(s)")
        else:
            logger.info(f"[GQF] embeddings unavailable; using DICT fallback for tension alignment")

        for goal in goals:
            _name = goal.get("name", "")
            semantic_align = sem_align.get(_name) if sem_align is not None else None
            score, reasons = self._score_goal(goal, tensions, emotions, semantic_align=semantic_align)
            goal['quality_score'] = round(score, 3)
            goal['quality_reasons'] = reasons
            report['scored'].append({'id': goal['id'], 'name': goal.get('name',''), 'score': score})

            if score < self.quality_threshold and goal.get('status') == 'active':
                # Don't mark low-quality if we're already below the minimum active threshold
                current_active = len([g for g in goals if g.get('status') == 'active'
                                      and g.get('quality_score', 0.5) >= self.quality_threshold
                                      and g['id'] != goal['id']])
                if current_active < 4:  # keep at least 4 high-quality goals active
                    pass
                else:
                    goal['status'] = 'low_quality'
                    goal['low_quality_reason'] = '; '.join(reasons)
                    report['marked_noise'].append(goal['id'])
                    logger.info(f"[GQF] Marked low-quality: {goal.get('name','')} (score={score:.2f})")

            elif score >= self.boost_threshold and goal.get('status') == 'active':
                old_pri = goal.get('priority', 0.5)
                # Skip boost if already at the ceiling — prevents "1.00→1.00" spam
                # and stops goals from being permanently locked at maximum priority,
                # which keeps goal_pressure pinned high and the governor in structured mode.
                PRIORITY_CEILING = 0.90  # hard cap so there is always room to breathe
                if old_pri >= PRIORITY_CEILING:
                    pass  # already at or above ceiling — no boost
                else:
                    new_pri = min(PRIORITY_CEILING, old_pri + 0.05)
                    goal['priority'] = new_pri
                    report['boosted'].append({'id': goal['id'], 'old': old_pri, 'new': new_pri})
                    logger.info(f"[GQF] Boosted: {goal.get('name','')} {old_pri:.2f}→{new_pri:.2f}")

            goal['quality_evaluated_at'] = _now()
            dal.save_goal(goal)

        # Completion semantics (A): retroactive settlement.
        # A goal that has already done real work (B: took a world-facing
        # action) whose spawning tension has since resolved (A) is eligible
        # for completion here — even if it went dormant. This is the
        # tension-resolution exit valve that replaces the old "3 insights =
        # done" shortcut. It is checked on every quality pass, so a goal that
        # was not yet eligible when it went dormant can still close properly
        # once its tension eases.
        from cognition.goal_completion import (
            goal_completion_eligible,
            record_goal_completion,
        )
        settled = []
        for goal in goals:
            if goal.get('status') not in ('active', 'dormant'):
                continue
            eligible, reason = goal_completion_eligible(goal, tensions)
            if eligible:
                record_goal_completion(goal, dal=dal, reason=reason)
                settled.append(goal.get('name') or goal.get('id', ''))
                logger.info(f"[GQF] Settled: '{goal.get('name', '')}' - {reason}")
        report['settled'] = settled

        # Generate replacement goals for dominant un-covered tensions
        # Hard cap: never exceed HARD_CAP_GOALS active goals across all generators
        HARD_CAP_GOALS = 12
        current_active = sum(1 for g in goals if g.get('status') == 'active')
        if current_active >= HARD_CAP_GOALS:
            logger.info(f"[GQF] ⛔ Cap reached ({current_active} active ≥ {HARD_CAP_GOALS}), skipping generation")
            new_goals = []
        else:
            new_goals = self._generate_tension_goals(tensions, goals, dal)
        report['generated'] = [g['name'] for g in new_goals]

        report['summary'] = {
            'noise_marked':   len(report['marked_noise']),
            'boosted':        len(report['boosted']),
            'settled':        len(report.get('settled', [])),
            'generated':      len(report['generated']),
            'high_quality':   sum(1 for s in report['scored'] if s['score'] >= self.boost_threshold),
        }
        return report

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_goal(
        self,
        goal: Dict,
        tensions: Dict[str, float],
        emotions: Dict[str, float],
        semantic_align: Optional[Tuple[float, str]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Score a goal 0-1.
        Returns (score, [reasons]).

        semantic_align: optional precomputed (best_alignment, best_tension_key)
            from the embedding-based scorer (cognition.goal_semantics). When
            provided, tension alignment uses real semantic similarity; when
            None, the original static dictionary matching is used as fallback.
        """
        reasons = []
        score = 0.5  # neutral baseline

        name  = goal.get('name', '').lower().strip()
        words = set(name.replace('_', ' ').split())

        # ── Semantic richness ──────────────────────────────────────────────
        if len(words) <= 1 and name in NOISE_WORDS:
            score -= 0.4
            reasons.append(f"single noise keyword '{name}'")
        elif len(words) <= 1:
            score -= 0.2
            reasons.append("single-word goal")
        elif len(words) >= 3:
            score += 0.1
            reasons.append("multi-word goal")

        # ── Origin quality ─────────────────────────────────────────────────
        origin = goal.get('origin', '')
        if origin == 'thought_evaluation':
            score += 0.1
            reasons.append("from thought evaluation")
        elif origin == 'thought_goal_linker':
            score += 0.15
            reasons.append("from semantic thought linker")
        elif origin == 'understand_user':
            score -= 0.15
            reasons.append("raw keyword from user message")
        elif origin == 'manual':
            score += 0.05

        # ── Tension alignment ──────────────────────────────────────────────
        # Preferred: embedding-based semantic similarity (goal name × tension
        # concept anchor). Fallback: original static dictionary matching when
        # no embedding tier is reachable (semantic_align is None).
        best_tension_match = 0.0
        if semantic_align is not None:
            best_tension_match, best_tension_key = semantic_align
            if best_tension_key:
                tension_level = tensions.get(best_tension_key, 0.0)
                reasons.append(
                    f"semantically aligns with {best_tension_key}="
                    f"{tension_level:.2f} (cosine)"
                )
        else:
            for tension_key, themes in TENSION_THEMES.items():
                tension_level = tensions.get(tension_key, 0.0)
                if tension_level < 0.3:
                    continue
                if any(theme in name for theme in themes):
                    alignment = tension_level * 0.3
                    best_tension_match = max(best_tension_match, alignment)
                    reasons.append(f"aligns with {tension_key}={tension_level:.2f}")
        score += best_tension_match

        # ── Emotional resonance ────────────────────────────────────────────
        if emotions.get('curiosity', 0) > 0.7 and any(
            t in name for t in ['understand', 'explore', 'learn', 'discover']
        ):
            score += 0.05
            reasons.append("resonates with high curiosity")

        # ── Origin thoughts ────────────────────────────────────────────────
        if goal.get('origin_thoughts'):
            score += 0.05 * min(3, len(goal['origin_thoughts']))
            reasons.append(f"{len(goal['origin_thoughts'])} origin thought(s)")

        # ── Priority signal ────────────────────────────────────────────────
        if goal.get('priority', 0.5) >= 0.7:
            score += 0.05

        return max(0.0, min(1.0, score)), reasons

    # ── Tension-aligned goal generation ───────────────────────────────────────

    def _generate_tension_goals(
        self,
        tensions: Dict[str, float],
        existing_goals: List[Dict],
        dal: Any,
    ) -> List[Dict]:
        """
        For each dominant tension that has no strong aligned goal,
        create a new meaningful goal.

        DIVERSITY FIX: Skip generation if a goal with a similar name/topic
        already exists in active pool (same name prefix = same concept).
        This breaks the infinite cycle of the same 5 goals regenerating every 18 min.
        """
        # Build set of already-active topic stems for dedup
        active_stems = set()
        for g in existing_goals:
            if g.get('status') == 'active':
                name = (g.get('name') or g.get('topic') or '').lower()[:25]
                active_stems.add(name)
        from core.data.schemas import make_goal

        existing_names = {g.get('name', '').lower() for g in existing_goals
                          if g.get('quality_score', 0.5) >= self.quality_threshold}
        new_goals = []

        for tension_key, template in TENSION_GOALS.items():
            level = tensions.get(tension_key, 0.0)
            if level < template['thresh']:
                continue
            # Check against ALL goals (active + archived + low_quality), not just high-quality
            all_names = {g.get('name', '').lower() for g in dal.get_goals()}
            if template['name'] in all_names:
                continue

            # Fix: every tension previously used a single frozen topic
            # string, which every downstream question-generator
            # (goal_action_executor.py, grounded_introspection.py) embedded
            # verbatim — producing the "always asking the same question"
            # symptom for whichever tension crossed threshold most often.
            # Generators now live in cognition/tension_topics.py (shared
            # with goal_action_executor.py, which regenerates a fresh topic
            # at ACTION time too — not just here at creation time — since a
            # goal can persist across several action cycles before
            # completing, and a topic frozen only at creation would still
            # repeat verbatim for the goal's whole lifetime otherwise).
            generator = TENSION_TOPIC_GENERATORS.get(tension_key)
            topic = generator(self.persona_dir) if generator else template['topic']

            new_goal = make_goal(
                name=template['name'],
                # Cap at 0.75 so newly generated tension goals have room to grow
                # (or decay) and don't immediately pin goal_pressure at ceiling.
                priority=min(0.75, 0.4 + level * 0.3),
                origin='goal_quality_filter',
                topic=topic,
            )
            new_goal['tension_source'] = tension_key
            new_goal['tension_level']  = level
            new_goal['quality_score']  = 0.75
            # Skip if same topic already active — breaks the infinite 5-goal cycle
            new_stem = new_goal.get('name', '')[:25].lower()
            if new_stem in active_stems:
                logger.debug(f"[GQF] Skipping duplicate topic: {new_stem!r}")
                continue
            active_stems.add(new_stem)
            dal.save_goal(new_goal)
            new_goals.append(new_goal)
            logger.info(f"[GQF] Generated: {new_goal['name']} (pri={new_goal['priority']:.2f})")

        return new_goals

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_emotions(self) -> Dict[str, float]:
        """Load current emotion values."""
        try:
            import json
            p = self.persona_dir / 'emotional_state.json'
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                return {k: v.get('value', 0.5) for k, v in data.get('emotions', {}).items()
                        if isinstance(v, dict)}
        except Exception:
            pass
        return {}


if __name__ == '__main__':
    import json as _j
    gqf = GoalQualityFilter()
    report = gqf.run()
    print('\n=== GOAL QUALITY REPORT ===')
    print(f"  Total goals     : {report['total']}")
    print(f"  Noise marked    : {report['summary']['noise_marked']}")
    print(f"  Boosted         : {report['summary']['boosted']}")
    print(f"  Generated       : {report['summary']['generated']} → {report['generated']}")
    print(f"  High quality    : {report['summary']['high_quality']}")
    print('\nTop scored:')
    for s in sorted(report['scored'], key=lambda x: x['score'], reverse=True)[:6]:
        print(f"  {s['score']:.2f}  {s['name']}")
