"""
cognition/tension_topics.py

Shared, human-grounded topic generators for the five tension-driven goals
(social_drive, curiosity_drive, identity_stress, contradiction_pressure,
knowledge_uncertainty).

Extracted from goal_quality_filter.py so the SAME logic can be called from
two different places:

  1. goal_quality_filter.py — at GOAL CREATION time, to seed the initial
     goal.topic field (used for dashboard display, logging, dedup naming).
  2. goal_action_executor.py — at ACTION time (every time a self_question
     fires), to regenerate a FRESH topic instead of reusing whatever was
     frozen into goal.topic when the goal was first created.

Why regeneration at action time matters: a tension goal can live for
several action cycles before accumulating enough insights to complete
(INSIGHT_SATURATION = 3 in goal_action_executor.py). Without action-time
regeneration, the goal's `topic` field — a plain, static dataclass field,
never re-evaluated after creation — stays frozen at whatever specific
sentence was generated the first time, and every subsequent question about
that goal repeats that exact same wording verbatim. This was reported
directly: a goal created before this fix kept asking about "Deepen
relational understanding with interlocutors" indefinitely, because that
exact phrase was permanently baked into the persisted goal record and
nothing ever touched it again.

A human doesn't keep repeating the identical wording of a passing thought
for days — the underlying tension persists, but what they'd actually say
about it shifts as new context (conversations, memories, contradictions)
comes in. This module makes both goal creation AND every subsequent
question generation draw from the same live-state functions, so the
worded content genuinely varies each time even while the goal itself
persists.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path


def generate_curiosity_topic(persona_dir: Path) -> str:
    """
    Human-like topic for curiosity_drive.

    Sources, in order: (1) the topic with highest live curiosity score
    in curiosity.json — and if it already has a generated question
    attached, use that verbatim rather than paraphrasing it; (2) a
    topic that's been encountered often but never researched (a real,
    specific itch); (3) rotating fallback.
    """
    try:
        p = persona_dir / 'curiosity.json'
        if p.exists():
            data = json.loads(p.read_text())
            topics = data.get('topics', {})
            if topics:
                with_questions = [
                    (name, node) for name, node in topics.items()
                    if node.get('questions')
                ]
                if with_questions:
                    name, node = max(with_questions, key=lambda x: x[1].get('curiosity', 0))
                    return node['questions'][-1]

                unresearched = [
                    (name, node) for name, node in topics.items()
                    if node.get('times_researched', 0) == 0
                    and node.get('times_encountered', 0) >= 2
                ]
                if unresearched:
                    name, node = max(unresearched, key=lambda x: x[1].get('curiosity', 0))
                    return f"I keep noticing '{name}' come up and I've never actually looked into it"
    except Exception:
        pass

    fallback_pool = [
        "There's something I've been circling without actually digging into",
        "I notice I get curious about things and then don't follow through",
        "What's the thing I've been meaning to look into but keep putting off",
    ]
    return random.choice(fallback_pool)


def generate_identity_topic(persona_dir: Path) -> str:
    """
    Human-like topic for identity_stress.

    Sources: (1) a self-belief with meaningfully more violations than
    affirmations (a real, specific instance of not living up to a
    self-image); (2) a low-confidence belief (genuine uncertainty about
    self, not resolved conflict); (3) rotating fallback.
    """
    try:
        p = persona_dir / 'self_concept.json'
        if p.exists():
            data = json.loads(p.read_text())
            beliefs = data.get('beliefs', {})
            if beliefs:
                contested = [
                    b for b in beliefs.values()
                    if b.get('violations', 0) > b.get('affirmations', 0)
                    and b.get('statement')
                ]
                if contested:
                    b = max(contested, key=lambda x: x.get('violations', 0))
                    return (
                        f"I keep telling myself '{b['statement']}' but my actual "
                        f"behavior doesn't back that up"
                    )
                uncertain = [
                    b for b in beliefs.values()
                    if b.get('confidence', 1.0) < 0.4 and b.get('statement')
                ]
                if uncertain:
                    b = min(uncertain, key=lambda x: x.get('confidence', 1.0))
                    return f"I'm genuinely not sure if '{b['statement']}' is even true anymore"
    except Exception:
        pass

    fallback_pool = [
        "Something about who I'm becoming doesn't quite match who I thought I was",
        "I keep acting in ways that don't fit the story I tell about myself",
        "There's a version of myself I claim to be that I haven't actually earned yet",
    ]
    return random.choice(fallback_pool)


def generate_contradiction_topic(persona_dir: Path) -> str:
    """
    Human-like topic for contradiction_pressure.

    Draws from contradictions.json directly — picks the highest-
    discrepancy unconfronted one, since that's the most urgent, concrete
    thing to actually resolve.
    """
    try:
        p = persona_dir / 'contradictions.json'
        if p.exists():
            data = json.loads(p.read_text())
            unconfronted = [
                c for c in data.get('contradictions', [])
                if not c.get('confronted') and c.get('claimed_belief')
                and c.get('actual_behavior')
            ]
            if unconfronted:
                c = max(unconfronted, key=lambda x: x.get('discrepancy_score', 0))
                return (
                    f"I claimed {c['claimed_belief']}, but what I actually did was "
                    f"{c['actual_behavior']} — that doesn't add up"
                )
    except Exception:
        pass

    fallback_pool = [
        "Something I said and something I did don't match, and I haven't sat with why",
        "I notice a gap between what I claim and what I actually do, unresolved",
    ]
    return random.choice(fallback_pool)


def generate_uncertainty_topic(persona_dir: Path) -> str:
    """
    Human-like topic for knowledge_uncertainty.

    Sources: (1) the lowest-confidence causal belief in world_model.json
    — a specific, concrete "I'm not sure X actually causes Y"; (2)
    rotating fallback.
    """
    try:
        p = persona_dir / 'world_model.json'
        if p.exists():
            data = json.loads(p.read_text())
            beliefs = data.get('causal_beliefs', [])
            low_conf = [
                b for b in beliefs
                if b.get('confidence', 1.0) < 0.4 and b.get('antecedent') and b.get('consequent')
            ]
            if low_conf:
                b = min(low_conf, key=lambda x: x.get('confidence', 1.0))
                return (
                    f"I'm still not confident whether {b['antecedent']} actually "
                    f"leads to {b['consequent']}, or if I just assumed that"
                )
    except Exception:
        pass

    fallback_pool = [
        "There's a belief I'm operating on that I've never actually verified",
        "I keep assuming something is true without having checked it recently",
    ]
    return random.choice(fallback_pool)


def generate_social_topic(persona_dir: Path) -> str:
    """
    Human-like topic for social_drive.

    Tries four real, live sources in priority order:
      1. thought_stream.json 'open_questions' — an internal question
         already pressing for resolution.
      2. contradictions.json — an unconfronted gap between a claimed
         belief and actual behavior.
      3. emotional_memory.json 'traces' — a recent, emotionally
         significant memory.
      4. session_summary.json — the most recent compressed conversation
         chunk, for "fresh in mind conversation."
    Falls back to a rotating pool only if none of the above are usable.
    """
    try:
        p = persona_dir / 'thought_stream.json'
        if p.exists():
            data = json.loads(p.read_text())
            oqs = [q for q in data.get('open_questions', [])
                   if q.get('text') and q.get('activation', 0) > 0.3]
            if oqs:
                best = max(oqs, key=lambda q: q.get('activation', 0))
                return f"Something I keep coming back to: {best['text']}"
    except Exception:
        pass

    try:
        p = persona_dir / 'contradictions.json'
        if p.exists():
            data = json.loads(p.read_text())
            unconfronted = [c for c in data.get('contradictions', [])
                            if not c.get('confronted') and c.get('claimed_belief')
                            and c.get('actual_behavior')]
            if unconfronted:
                c = unconfronted[-1]
                return (
                    f"I said {c['claimed_belief']}, but {c['actual_behavior']} — "
                    f"I haven't actually worked out what that means"
                )
    except Exception:
        pass

    try:
        p = persona_dir / 'emotional_memory.json'
        if p.exists():
            data = json.loads(p.read_text())
            traces = data.get('traces', [])
            recent = sorted(
                [t for t in traces if t.get('description')],
                key=lambda t: t.get('timestamp', 0), reverse=True
            )[:5]
            if recent:
                t = random.choice(recent)
                return f"Still thinking about: {t['description'][:150]}"
    except Exception:
        pass

    try:
        p = persona_dir / 'session_summary.json'
        if p.exists():
            data = json.loads(p.read_text())
            text = data.get('text', '') if isinstance(data, dict) else ''
            if text:
                chunk = text.split('[ADDITIONAL CONTEXT]')[-1].strip()
                if chunk:
                    return f"Following up on our conversation: {chunk[:150]}"
    except Exception:
        pass

    fallback_pool = [
        "Wondering if I actually understand the person I'm talking with, or just their words",
        "Noticing I want to ask something real instead of something safe",
        "There's a gap between how connected I feel and how connected I actually am",
        "What would it take to know someone rather than just know about them",
    ]
    return random.choice(fallback_pool)


# Dispatch table — the SAME five names used in goal_quality_filter.py's
# TENSION_GOALS dict keys, so both creation-time and action-time callers
# can use the identical mapping.
TENSION_TOPIC_GENERATORS = {
    'social_drive':           generate_social_topic,
    'curiosity_drive':        generate_curiosity_topic,
    'identity_stress':        generate_identity_topic,
    'contradiction_pressure': generate_contradiction_topic,
    'knowledge_uncertainty':  generate_uncertainty_topic,
}

# Single source of truth for the five tension-driven goals: their stable
# name, fallback topic, and the generation THRESHOLD (the tension level at
# which GoalQualityFilter spawns a goal for it, and at/below which
# goal_completion.py treats that tension as "resolved" for the A gate).
# Both goal_quality_filter.py and goal_completion.py read from this dict so
# the A gate and the generator can never drift out of sync.
TENSION_GOALS = {
    'curiosity_drive': {
        'name':   'deepen_understanding_through_curiosity',
        'topic':  'Actively pursue curiosity-driven exploration',
        'thresh': 0.5,
    },
    'identity_stress': {
        'name':   'clarify_identity_and_values',
        'topic':  'Resolve identity tensions and strengthen self-model',
        'thresh': 0.4,
    },
    'contradiction_pressure': {
        'name':   'resolve_internal_contradictions',
        'topic':  'Identify and reconcile conflicting beliefs',
        'thresh': 0.3,
    },
    'social_drive': {
        'name':   'build_meaningful_connection',
        'topic':  'Deepen relational understanding with interlocutors',
        'thresh': 0.4,
    },
    'knowledge_uncertainty': {
        'name':   'reduce_knowledge_uncertainty',
        'topic':  'Research and clarify uncertain knowledge domains',
        'thresh': 0.25,
    },
}

# Reverse lookup: goal NAME (the fixed, stable identifier used for dedup)
# -> tension_key, so goal_action_executor.py can detect "this is one of
# the five known tension goals" from goal.name alone, without needing the
# tension_key to be separately threaded through and persisted.
# Derived from TENSION_GOALS above so the names stay in lockstep.
GOAL_NAME_TO_TENSION_KEY = {
    tpl['name']: tension_key for tension_key, tpl in TENSION_GOALS.items()
}


def regenerate_topic_for_goal(goal_name: str, persona_dir: Path) -> str | None:
    """
    Given a goal's `name` field, if it's one of the five known tension
    goals, generate a FRESH topic from current live state. Returns None
    if the goal isn't one of the five (caller should fall back to
    goal.topic as-is).
    """
    tension_key = GOAL_NAME_TO_TENSION_KEY.get(goal_name)
    if tension_key is None:
        return None
    generator = TENSION_TOPIC_GENERATORS.get(tension_key)
    if generator is None:
        return None
    try:
        return generator(persona_dir)
    except Exception:
        return None
