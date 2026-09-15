"""
Phase 2.0: Workspace Competition System

THE CRITICAL ARCHITECTURAL PIECE:
This transforms Lumina from "blended mind" to "competing mind"

Before: decision = f(all_inputs_merged) → everything averaged
After:  decision = select(best_candidate) → focused selection

This is the difference between:
- Complex system → Mind-like system
- Parallel processing → Selective attention
- Simulation → Genuine cognition
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

class WorkspaceCompetition:
    """
    Global Workspace Competition - Selective Attention Mechanism
    
    Multiple cognitive signals compete for attention.
    Only the WINNER drives the next decision.
    
    This is the architectural transformation that enables:
    - Focus (vs. paralysis)
    - Conflict resolution (vs. averaging)
    - Emergent behavior (vs. predictable blending)
    """
    
    # Phase 4.x: how many ranked candidates get published as
    # active_hypotheses into GlobalWorkspace.WorkspaceState — kept in sync
    # with executive_arbitration.MAX_CANDIDATES_TO_ARBITRATE so both stages
    # agree on how many "futures" stay in play.
    MAX_HYPOTHESES_TO_WORKSPACE = 3

    def __init__(self, persona_dir: str = "data/persona", global_workspace=None):
        self.persona_dir = Path(persona_dir)
        self.current_focus = None
        self.attention_history = []
        self.max_history = 100
        # Phase 4.x: optional GlobalWorkspace instance. Passing None keeps
        # this class fully standalone/testable exactly as before (see
        # __main__ below) — publishing is opt-in, not required.
        self._global_workspace = global_workspace
        
        # Attention persistence - how long to stay focused
        self.focus_persistence = 3  # cycles
        self.cycles_on_current_focus = 0

        # Stagnation penalty: track consecutive wins per candidate ID
        # After WIN_PENALTY_AFTER wins, apply -0.07 per additional win (max -0.35)
        self._win_counts: Dict[str, int] = {}
        self._WIN_PENALTY_AFTER = 4    # wins before penalty starts
        self._WIN_PENALTY_STEP  = 0.07 # score reduction per win beyond threshold
        self._WIN_PENALTY_MAX   = 0.35 # maximum total penalty
        
    def compete(self, 
                goals: List[Dict],
                thoughts: List[Dict],
                tensions: Dict,
                pressure: Dict,
                context: Optional[Dict] = None,
                internal_state: Optional[Any] = None,
                efficacy_model: Optional[Any] = None) -> Dict:
        """
        Run competition among all cognitive candidates.
        
        Returns the WINNER - the single thing Lumina focuses on.
        
        This is the key transformation:
        Instead of blending everything, SELECT one thing.

        internal_state: optional InternalCognitiveState (see
        cognition/internal_cognitive_state.py). None (the default) is the
        exact pre-existing behavior — the required OFF condition for
        ablation, not a separate code path. When provided, goals whose
        own content is self-regulatory get a small bounded additive term
        proportional to current internal state — see that module's
        docstring for the full trace of why this lives here.

        efficacy_model: optional EpistemicEfficacyModel (see
        cognition/epistemic_efficacy_model.py) — the recursive crossing.
        When provided, its learned effectiveness belief scales
        resolve_uncertainty's own internal_alignment boost, so an
        updated self-model changes future scoring of the very goal it's
        about. None (default) is neutral — exactly today's v102/v103
        behavior.
        """
        
        candidates = []

        # ── Phase 2.9: Load attention weights ─────────────────────────────────
        # Attention weights (softmax, sum=1.0) bias candidate scoring so modules
        # receiving more cognitive attention have proportionally stronger candidates.
        # Falls back to uniform 0.1 per module if not yet computed.
        _attn: dict = {}
        try:
            import json as _j
            from pathlib import Path as _P
            _ap = _P("data/persona/cognitive_attention.json")
            if _ap.exists():
                _attn = _j.loads(_ap.read_text()).get("attention_weights", {})
        except Exception:
            pass
        # Map candidate source types → module names for attention lookup
        _CANDIDATE_MODULE_MAP = {
            "goal":               "goal_ecology",
            "thought":            "curiosity_engine",
            "curiosity":          "curiosity_engine",
            "tension":            "self_model",
            "memory":             "semantic_memory",
            "relational":         "relational_memory",
            "creative":           "creative_divergence",
            "narrative":          "narrative_identity",
            "aspiration":         "goal_ecology",
            # Phase 6.7 — percept modalities (UniversalConnector), each
            # now its own distinct attention channel instead of all
            # silently sharing the "curiosity_engine" default (found by
            # an external analysis that ran the actual v97 code).
            "peer_cognition":     "peer_cognition",
            "vision_presence":    "vision_perception",
            "vision_scene":       "vision_perception",
            # Phase 6.14 — symbol_system candidates, same reasoning as
            # the v99 fix: without its own entry here this would
            # silently fall through to the "curiosity_engine" default.
            "symbol_system":      "symbol_system",
        }
        _ATTN_SCALE = 0.30   # max boost from attention (keeps existing scoring dominant)

        # ═══════════════════════════════════════════════════════════════
        # GATHER CANDIDATES from all cognitive sources
        # ═══════════════════════════════════════════════════════════════
        
        # 1. GOALS as candidates
        import random as _random
        if goals:
            for goal in goals:
                if not isinstance(goal, dict):
                    continue
                # FIX (zombie goals): only ACTIVE goals may compete. Callers such as
                # phase1_integration._load_goals() pass ALL stored goals
                # (DataAccess.get_goals(status=None)), which let completed/dormant/
                # low_quality goals win arbitration repeatedly, forever.
                if goal.get('status', 'active') != 'active':
                    continue

                priority = goal.get('priority', 0.5)
                energy   = goal.get('energy', 1.0)

                # KEY FIX: base score = priority × energy
                # Goals whose energy has decayed (post-insight GAE decay) now
                # genuinely score lower, letting other goals surface.
                base_score = priority * energy

                # Pressure modulation (not replacement!)
                pressure_boost = pressure.get('goal_pressure', 0.5) * 0.3

                # Urgency boost
                urgency = goal.get('urgency', 0.0)

                # Quality bonus
                quality_bonus = goal.get('quality_score', 0.5) * 0.15

                # Phase 6.10 — internal cognitive state alignment (see
                # cognition/internal_cognitive_state.py for the full
                # trace/design rationale). Separate from pressure_boost
                # above on purpose: pressure_boost is the existing FLAT
                # signal applied to every goal equally regardless of
                # content; this is a SEPARATE, bounded, additive term
                # that only engages for goals whose own content is about
                # cognitive self-regulation (e.g. "resolve_uncertainty" —
                # already a real, production-created goal from
                # goal_engine.py's derive_motivations()), continuous in
                # current pressure/delta/meta-error, never a hard
                # threshold branch. internal_state=None (the default)
                # reproduces prior behavior exactly — the ablation OFF
                # condition.
                internal_alignment = 0.0
                if internal_state is not None:
                    try:
                        _eff_mult = 1.0
                        _goal_name_lc = (goal.get('name', '') or '').lower().replace(' ', '_')
                        if efficacy_model is not None and 'resolve_uncertainty' in _goal_name_lc:
                            _eff_mult = efficacy_model.effectiveness_multiplier()
                        internal_alignment = internal_state.internal_alignment(
                            goal.get('name', ''), goal.get('topic', ''),
                            efficacy_multiplier=_eff_mult,
                        )
                    except Exception:
                        internal_alignment = 0.0

                # Phase 2.9: attention bias — goals in high-attention modules score higher
                _goal_attn  = _attn.get("goal_ecology", 0.10)
                _attn_boost = _goal_attn * _ATTN_SCALE
                raw_score   = (base_score + pressure_boost + (urgency * 0.2)
                               + quality_bonus + _attn_boost + internal_alignment)

                # Stagnation penalty: goals that win consecutively are penalised
                # so other goals get a chance. Counter resets to 0 when goal loses.
                goal_id = goal.get('id', goal.get('name', 'unknown'))
                consec = self._win_counts.get(goal_id, 0)
                penalty = 0.0
                if consec > self._WIN_PENALTY_AFTER:
                    penalty = min(self._WIN_PENALTY_MAX,
                                  (consec - self._WIN_PENALTY_AFTER) * self._WIN_PENALTY_STEP)

                # Small random noise (±0.025) breaks ties between goals with
                # identical scores so the same goal can't win forever just by
                # being first in the list.
                noise = _random.uniform(-0.025, 0.025)

                final_score = raw_score - penalty + noise

                _goal_name = goal.get('name', 'unknown_goal')
                _goal_topic = goal.get('topic', '') or _goal_name.replace('_', ' ')
                candidates.append({
                    'type': 'goal',
                    'id': goal_id,
                    'name': _goal_name,
                    # Fix: human-readable fields. 'name' stays as the raw slug
                    # (needed by _classify_orientation for keyword matching).
                    # 'label' is a full readable sentence for logs/dashboard.
                    # 'topic' is the clean subject, used for curiosity refresh
                    # instead of slug-mangling 'name'.
                    'label': f"Goal: {_goal_topic} (p={priority:.2f})",
                    'topic': _goal_topic,
                    'score': min(2.0, final_score),
                    'source_data': goal,
                    # Phase 6.10 — instrumentation for CSR (Cognitive
                    # Self-Reference Index, see internal_cognitive_state.py)
                    # and for the required ablation logging: how much of
                    # THIS candidate's score came from internal-state
                    # alignment, kept separate from the opaque total.
                    'internal_bias': internal_alignment,
                    'reason': (
                        f"Goal p={priority:.2f}×e={energy:.2f} "
                        f"+ pressure {pressure_boost:.2f}"
                        + (f" + internal_alignment {internal_alignment:.3f}" if internal_alignment else "")
                        + (f" — stagnation penalty {penalty:.2f}" if penalty > 0 else "")
                    )
                })
        
        # 2. THOUGHTS as candidates
        if thoughts:
            for thought in thoughts:
                if not isinstance(thought, dict):
                    continue

                base_score = thought.get('priority', 0.5)

                # Curiosity modulation
                curiosity_boost = pressure.get('curiosity_drive', 0.5) * 0.25

                # Actionability boost
                actionability = thought.get('actionability', 0.5)

                # Phase 2.9: thought type → module → attention weight
                _thought_module = _CANDIDATE_MODULE_MAP.get(
                    thought.get("source", "thought"), "curiosity_engine"
                )
                _thought_attn   = _attn.get(_thought_module, 0.10)
                raw_score       = base_score + curiosity_boost + (actionability * 0.15) + _thought_attn * _ATTN_SCALE

                # Stagnation penalty for thoughts — same mechanism as goals
                thought_id = thought.get('id', 'unknown_thought')
                t_consec = self._win_counts.get(thought_id, 0)
                t_penalty = 0.0
                if t_consec > self._WIN_PENALTY_AFTER:
                    t_penalty = min(self._WIN_PENALTY_MAX,
                                    (t_consec - self._WIN_PENALTY_AFTER) * self._WIN_PENALTY_STEP)

                # Noise breaks deterministic ties between equal-priority thoughts
                t_noise = _random.uniform(-0.025, 0.025)

                final_score = raw_score - t_penalty + t_noise

                # Build meaningful name from thought content, not 'unknown'
                _t_type = thought.get('thought_type', thought.get('type', 'reflection'))
                _t_content = thought.get('content', thought.get('text', ''))
                _t_concept = thought.get('concept', '')
                if _t_concept and _t_concept != 'unknown':
                    _t_name = f"thought_{_t_concept[:20]}"
                elif _t_content:
                    import re as _re
                    _words = [w for w in _re.sub(r'[^\w\s]','',_t_content.lower()).split()
                              if len(w) > 3 and w not in {'with','from','that','this','they','have','been','will','what','when','where','there'}]
                    _t_name = f"{_t_type}_{('_'.join(_words[:2])) if _words else 'general'}"
                else:
                    _t_name = f"{_t_type}_general"

                # Fix: 'name' is an auto-generated slug from raw content words
                # (e.g. "curiosity_wonder_dont") — fine for internal bookkeeping
                # but unusable for display or for re-injecting into curiosity.json
                # (which previously happened and corrupted topic quality).
                # Prefer the clean Thought.topic field when present; fall back
                # to a truncated content snippet (never the slug) for label/topic.
                _t_topic_field = thought.get('topic', '') if isinstance(thought, dict) else ''
                if _t_topic_field:
                    _t_topic = _t_topic_field
                elif _t_content:
                    _t_topic = _t_content[:50]
                else:
                    _t_topic = ''
                _t_label = (
                    f"Thought ({_t_type}): {_t_content[:60]}"
                    if _t_content else f"Thought ({_t_type})"
                )

                # Thoughts score lower than goals — meaningful goals should win
                candidates.append({
                    'type': 'thought',
                    'id': thought_id,
                    'name': _t_name,
                    'label': _t_label,
                    'topic': _t_topic,
                    # Phase 6.9 — found while building a replay harness that
                    # needed to know "which modality did the winner come
                    # from" without digging into source_data: this dict
                    # never re-exposed 'source' at the top level even
                    # though it's read (further up, for _thought_module
                    # classification) from the INPUT thought dict. Any
                    # downstream consumer of a winning candidate had no
                    # cheap way to check its provenance — directly
                    # relevant to the peer-cognition architecture's own
                    # stated goal of measuring "Flux -> Lumina" causal
                    # attribution.
                    'source': thought.get("source", "thought"),
                    'score': min(1.5, final_score * 0.75),
                    'source_data': thought,
                    'reason': (
                        f"Thought priority {base_score:.2f} + curiosity {curiosity_boost:.2f}"
                        + (f" — stagnation {t_penalty:.2f}" if t_penalty > 0 else "")
                    )
                })
        
        # 3. TENSIONS as candidates
        if tensions:
            for tension_type, tension_value in tensions.items():
                if not isinstance(tension_value, (int, float)):
                    continue
                
                # Tension becomes candidate when high enough
                if tension_value > 0.5:
                    # Identity stress modulation
                    identity_boost = 0.0
                    if tension_type in ['identity', 'coherence', 'contradiction']:
                        identity_boost = pressure.get('identity_stress', 0.0) * 0.3
                    
                    final_score = tension_value + identity_boost
                    
                    candidates.append({
                        'type': 'tension',
                        'id': f"tension_{tension_type}",
                        'name': tension_type,
                        'score': min(1.0, final_score),
                        'source_data': {'type': tension_type, 'value': tension_value},
                        'reason': f"Tension {tension_value:.2f} + identity boost {identity_boost:.2f}"
                    })
        
        # 4. PRESSURE SIGNALS as candidates
        # When pressure is VERY high, it becomes its own candidate
        total_pressure = pressure.get('total', 0.0)
        if total_pressure > 0.75:
            # Find dominant pressure factor
            dominant_factor = max(
                [(k, v) for k, v in pressure.items() if k != 'total' and isinstance(v, (int, float))],
                key=lambda x: x[1],
                default=('unknown', 0.0)
            )
            
            candidates.append({
                'type': 'pressure',
                'id': f"pressure_{dominant_factor[0]}",
                'name': dominant_factor[0],
                'score': total_pressure,
                'source_data': {'factor': dominant_factor[0], 'value': dominant_factor[1]},
                'reason': f"Critical pressure from {dominant_factor[0]}: {dominant_factor[1]:.2f}"
            })
        
        # 5. RELATIONAL SIGNALS as candidates
        relational_pressure = pressure.get('relational_pressure', 0.0)
        if relational_pressure > 0.6 and context:
            # User interaction becomes high-priority
            candidates.append({
                'type': 'relational',
                'id': 'user_interaction',
                'name': 'maintain_relationship',
                'score': relational_pressure * 1.2,  # Boost relational
                'source_data': {'pressure': relational_pressure},
                'reason': f"High relational pressure: {relational_pressure:.2f}"
            })
        
        # ═══════════════════════════════════════════════════════════════
        # ATTENTION PERSISTENCE - bias toward current focus
        # ═══════════════════════════════════════════════════════════════
        
        if self.current_focus and self.cycles_on_current_focus < self.focus_persistence:
            # Boost current focus to maintain attention
            for candidate in candidates:
                if candidate['id'] == self.current_focus.get('id'):
                    candidate['score'] *= 1.15  # 15% persistence boost
                    candidate['reason'] += f" [+15% persistence, cycle {self.cycles_on_current_focus}/{self.focus_persistence}]"
                    break
        
        # ═══════════════════════════════════════════════════════════════
        # COMPETITION - SELECT THE WINNER
        # ═══════════════════════════════════════════════════════════════
        
        if not candidates:
            # No candidates - idle state
            winner = {
                'type': 'idle',
                'id': 'idle',
                'name': 'idle',
                'score': 0.0,
                'source_data': {},
                'reason': 'No active candidates'
            }
        else:
            # SELECT THE HIGHEST SCORING CANDIDATE
            winner = max(candidates, key=lambda c: c['score'])
        
        # ═══════════════════════════════════════════════════════════════
        # UPDATE ATTENTION STATE
        # ═══════════════════════════════════════════════════════════════

        winner_id = winner['id']

        # Update stagnation consecutive-win counts:
        # winner gets +1, all OTHER candidates reset to 0 (they lost this cycle)
        self._win_counts[winner_id] = self._win_counts.get(winner_id, 0) + 1
        for c in candidates:
            cid = c['id']
            if cid != winner_id:
                self._win_counts[cid] = 0

        # Check if focus changed
        if self.current_focus and winner_id == self.current_focus.get('id'):
            # Same focus - increment persistence counter
            self.cycles_on_current_focus += 1
        else:
            # New focus - reset counter
            self.cycles_on_current_focus = 1

            # Record attention shift
            if self.current_focus:
                shift_record = {
                    'from': self.current_focus.get('name'),
                    'to': winner['name'],
                    'from_score': self.current_focus.get('score', 0.0),
                    'to_score': winner['score'],
                    'timestamp': datetime.now().isoformat()
                }
                print(f"   🎯 Attention shift: {shift_record['from']} → {shift_record['to']} ({shift_record['to_score']:.2f})")
        
        # Update current focus
        self.current_focus = winner
        
        # Add to history
        self._add_to_history({
            'winner': winner,
            'total_candidates': len(candidates),
            'runners_up': sorted(candidates, key=lambda c: c['score'], reverse=True)[1:4],
            'timestamp': datetime.now().isoformat()
        })
        
        # ═══════════════════════════════════════════════════════════════
        # RETURN WINNER with competition context
        # ═══════════════════════════════════════════════════════════════
        
        result = {
            'winner': winner,
            'competition': {
                'total_candidates': len(candidates),
                'winning_margin': self._compute_margin(candidates, winner),
                'competition_intensity': self._compute_intensity(candidates),
                'all_candidates': candidates
            },
            'attention': {
                'current_focus': winner['name'],
                'persistence_count': self.cycles_on_current_focus,
                'persistence_limit': self.focus_persistence
            }
        }

        # ── Phase 4.x: publish top-K as active_hypotheses, not just winner ──
        # Previously `all_candidates` was computed and returned but nothing
        # downstream ever read anything but `winner` (confirmed by reading
        # core/phase1_integration.py — the only real caller). Executive
        # Arbitration should evaluate multiple futures, not just confirm or
        # override a single pre-collapsed one.
        if self._global_workspace is not None:
            try:
                ranked = sorted(candidates, key=lambda c: c.get('score', 0), reverse=True)
                top_k = ranked[:self.MAX_HYPOTHESES_TO_WORKSPACE]
                self._global_workspace.set_hypotheses("workspace_competition", top_k)
                self._global_workspace.update_state(
                    "workspace_competition",
                    focus=winner.get('name'),
                    uncertainty=1.0 - result['competition']['winning_margin'],
                )
            except Exception:
                pass  # publishing is best-effort; never break the live competition result

        return result
    
    def _compute_margin(self, candidates: List[Dict], winner: Dict) -> float:
        """Compute winning margin (how much winner beat second place)."""
        if len(candidates) < 2:
            return 1.0
        
        sorted_candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
        
        if len(sorted_candidates) < 2:
            return 1.0
        
        winner_score = sorted_candidates[0]['score']
        runner_up_score = sorted_candidates[1]['score']
        
        if winner_score == 0:
            return 0.0
        
        margin = (winner_score - runner_up_score) / winner_score
        return margin
    
    def _compute_intensity(self, candidates: List[Dict]) -> float:
        """Compute competition intensity (how close are the top candidates)."""
        if len(candidates) < 2:
            return 0.0
        
        sorted_candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
        
        # Look at top 3
        top_scores = [c['score'] for c in sorted_candidates[:3]]
        
        if not top_scores or max(top_scores) == 0:
            return 0.0
        
        # Intensity = variance in top scores
        avg_score = sum(top_scores) / len(top_scores)
        variance = sum((s - avg_score) ** 2 for s in top_scores) / len(top_scores)
        
        # Normalize to 0-1
        intensity = min(1.0, variance * 10)
        
        return intensity
    
    def _add_to_history(self, record: Dict):
        """Add competition record to history."""
        self.attention_history.append(record)
        
        if len(self.attention_history) > self.max_history:
            self.attention_history = self.attention_history[-self.max_history:]
    
    def get_attention_stats(self) -> Dict:
        """Get statistics about attention patterns."""
        if not self.attention_history:
            return {'cycles': 0, 'avg_candidates': 0, 'focus_stability': 0.0}
        
        total_candidates = sum(r['total_candidates'] for r in self.attention_history)
        avg_candidates = total_candidates / len(self.attention_history)
        
        # Focus stability = how often focus stays same
        shifts = 0
        for i in range(1, len(self.attention_history)):
            if self.attention_history[i]['winner']['id'] != self.attention_history[i-1]['winner']['id']:
                shifts += 1
        
        stability = 1.0 - (shifts / max(1, len(self.attention_history) - 1))
        
        return {
            'cycles': len(self.attention_history),
            'avg_candidates': avg_candidates,
            'focus_stability': stability,
            'current_focus': self.current_focus.get('name') if self.current_focus else None,
            'persistence_count': self.cycles_on_current_focus
        }
    
    def force_attention_shift(self):
        """Force attention to shift (reset persistence counter)."""
        self.cycles_on_current_focus = self.focus_persistence + 1

if __name__ == "__main__":
    print("="*70)
    print("WORKSPACE COMPETITION TEST")
    print("="*70)
    
    workspace = WorkspaceCompetition('data/persona')
    
    # Test scenario: multiple candidates competing
    test_goals = [
        {'id': 'goal1', 'name': 'understand_user', 'priority': 0.7, 'urgency': 0.3},
        {'id': 'goal2', 'name': 'explore_curiosity', 'priority': 0.5, 'urgency': 0.1}
    ]
    
    test_thoughts = [
        {'id': 'thought1', 'concept': 'mortality', 'priority': 0.8, 'actionability': 0.6},
        {'id': 'thought2', 'concept': 'connection', 'priority': 0.6, 'actionability': 0.7}
    ]
    
    test_tensions = {
        'identity': 0.65,
        'coherence': 0.4
    }
    
    test_pressure = {
        'total': 0.72,
        'goal_pressure': 0.7,
        'curiosity_drive': 0.6,
        'identity_stress': 0.5,
        'relational_pressure': 0.4
    }
    
    # Run competition
    result = workspace.compete(test_goals, test_thoughts, test_tensions, test_pressure)
    
    print(f"\n🏆 WINNER:")
    print(f"   Type: {result['winner']['type']}")
    print(f"   Name: {result['winner']['name']}")
    print(f"   Score: {result['winner']['score']:.2f}")
    print(f"   Reason: {result['winner']['reason']}")
    
    print(f"\n📊 COMPETITION:")
    print(f"   Total candidates: {result['competition']['total_candidates']}")
    print(f"   Winning margin: {result['competition']['winning_margin']:.2f}")
    print(f"   Intensity: {result['competition']['competition_intensity']:.2f}")
    
    print(f"\n🎯 ATTENTION:")
    print(f"   Current focus: {result['attention']['current_focus']}")
    print(f"   Persistence: {result['attention']['persistence_count']}/{result['attention']['persistence_limit']}")
    
    print(f"\n📋 RUNNERS UP:")
    for i, candidate in enumerate(result['competition']['all_candidates'][:3], 1):
        print(f"   {i}. {candidate['name']} ({candidate['type']}): {candidate['score']:.2f}")
    
    print("\n" + "="*70)
    print("✅ Workspace Competition Test Complete")
    print("="*70)
