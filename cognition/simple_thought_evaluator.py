"""
Quick Win #4: Thought Evaluator
Transforms thoughts from word-focused logs into concept-focused decision inputs.
This is the BRIDGE between thinking and acting.
"""
import json
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict, Any

class SimpleThoughtEvaluator:
    
    def __init__(self, persona_dir):
        self.persona_dir = Path(persona_dir)
        
        self.uncertainty_keywords = [
            'wonder', 'curious', 'curiosity', 'why', 'how', 'what', 'uncertain',
            'unclear', 'confus', 'question', 'understand', 'don\'t know',
            'unsatisfied', 'not yet', 'still thinking', 'hasn\'t been',
            'fascinat', 'intrigue', 'puzzle', 'mystery', 'wonder',
            'me demande', 'curieux', 'comprendre', 'pourquoi', 'comment',
        ]
        
        self.tension_keywords = [
            'conflict', 'tension', 'contradiction', 'disagree',
            'paradox', 'struggle', 'difficult', 'but', 'yet',
            'however', 'despite', 'although', 'contrast', 'pull',
            'keep thinking', 'returning', 'unresolved', 'stuck',
        ]
        
        self.opportunity_keywords = [
            'could', 'might', 'explore', 'try', 'experiment',
            'discover', 'learn', 'grow', 'should', 'pursue',
            'opportunity', 'potential', 'possible', 'consider',
            'reflection', 'insight', 'realise', 'realize', 'notice',
        ]
        
        self.insight_sources = {
            'goal_action.self_question', 'goal_action.memory_recall',
            'goal_action.web_search', 'insight', 'reflection',
        }
    
    def evaluate_recent_thoughts(self) -> List[Dict]:
        try:
            from core.data.access import DataAccess
            _dal = DataAccess(str(self.persona_dir))
            thoughts = _dal.get_thoughts(skip_expired=True)
            goals_list = _dal.get_goals()
            goals_data = {'goals': {g['id']: g for g in goals_list}}
            _use_dal = True
        except Exception:
            thoughts_file = self.persona_dir / 'thought_stream.json'
            goals_file = self.persona_dir / 'goals.json'
            if not thoughts_file.exists():
                print("⚠️  thought_stream.json not found")
                return []
            with open(thoughts_file, 'r', encoding='utf-8') as f:
                thoughts_data = json.load(f)
            goals_data = {}
            if goals_file.exists():
                with open(goals_file, 'r', encoding='utf-8') as f:
                    goals_data = json.load(f)
            thoughts = thoughts_data.get('thoughts', [])
            _use_dal = False

        suggestions = []
        evaluated_count = 0
        
        print(f"\n🧠 Evaluating {len(thoughts)} thoughts...")
        
        import time as _time
        _now = _time.time()
        REEVAL_INTERVAL = 1800

        for thought in thoughts:
            if thought.get('evaluated', False):
                evaluated_at = thought.get('evaluated_at', '')
                try:
                    from datetime import datetime, timezone
                    ev_ts = datetime.fromisoformat(
                        evaluated_at.replace('Z', '+00:00')
                    ).timestamp() if evaluated_at else 0
                    age = _now - ev_ts
                    if age < REEVAL_INTERVAL:
                        continue
                    thought['evaluated'] = False
                except Exception:
                    continue
            
            try:
                evaluation = self._evaluate_thought(thought)
            except Exception as _eval_err:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    f"[ThoughtEvaluator] _evaluate_thought error: {_eval_err}"
                )
                continue
            
            if evaluation:
                if evaluation['should_create_goal']:
                    if not self._has_existing_goal(goals_data, evaluation['concept']):
                        _hyp = evaluation.get('hypothesis', '')
                        suggestions.append({
                            'type': evaluation['goal_type'],
                            'concept': evaluation['concept'],
                            'priority': evaluation['priority'],
                            'origin_thought': thought.get('id', self._generate_id(thought)),
                            'created_at': datetime.now().isoformat(),
                            'conceptual_friction': evaluation.get('friction', ''),
                            'actionability': evaluation.get('actionability', 0.5),
                            'hypothesis': _hyp,
                            'name': (
                                f"test_hypothesis_{evaluation['concept'][:30]}"
                                if _hyp else evaluation['concept']
                            ),
                        })
                
                thought['evaluated'] = True
                thought['evaluation'] = {
                    'type': evaluation['type'],
                    'concept': evaluation['concept'],
                    'priority': evaluation['priority'],
                    'actionability': evaluation.get('actionability', 0.5),
                    'conceptual_friction': evaluation.get('friction', '')
                }

                concept = evaluation['concept']
                if concept and concept != 'unknown' and _use_dal:
                    self._inject_followup_thoughts(_dal, concept, evaluation['type'])

                evaluated_count += 1

        if _use_dal:
            for thought in thoughts:
                if thought.get('evaluated'):
                    _dal.add_thought(thought)
        else:
            import os
            tmp = thoughts_file.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(thoughts_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, thoughts_file)
        
        if evaluated_count == 0 and _use_dal:
            import time as _t, random as _rand
            seeds = [
                "What is currently most important for me to understand?",
                "What recent interaction has stayed with me and why?",
                "What feels unresolved in my current state of mind?",
                "What assumption am I making that I should question?",
            ]
            try:
                _dal.add_thought({
                    'content': _rand.choice(seeds),
                    'source': 'thought_evaluator.stagnation_guard',
                    'priority': 0.50,
                    'thought_type': 'reflection',
                    'timestamp': _t.time(),
                    'evaluated': False,
                })
            except Exception:
                pass

        print(f"   ✅ Evaluated {evaluated_count} new thoughts")
        print(f"   🎯 Generated {len(suggestions)} goal suggestions")
        
        return suggestions
    
    def _evaluate_thought(self, thought: Dict) -> Dict[str, Any]:
        text = (thought.get('content') or thought.get('text') or '').lower()
        
        if not text:
            return None
        
        # ✅ FIX: pass thought here
        thought_type = self._classify_thought(text, thought)
        
        if thought_type == 'mundane':
            return None
        
        concept = self._extract_concept(text)
        priority = self._compute_priority(text, thought_type)
        actionability = self._compute_actionability(text, thought_type)
        friction = self._identify_friction(text, thought_type)
        goal_type = self._determine_goal_type(thought_type)
        
        should_create_goal = (
            thought_type in ['uncertainty', 'tension', 'opportunity'] and
            priority > 0.4 and
            actionability > 0.3
        )

        # For high-priority tensions: generate a concrete hypothesis instead of
        # just labelling the tension. This shifts action from 'resolve_tension'
        # to 'test_hypothesis' — the key upgrade the analysis identified.
        hypothesis = None
        if thought_type == 'tension' and priority >= 0.65:
            hypothesis = self._generate_hypothesis(text, concept)

        return {
            'type': thought_type,
            'concept': concept,
            'priority': priority,
            'actionability': actionability,
            'friction': friction,
            'goal_type': 'test_hypothesis' if hypothesis else goal_type,
            'hypothesis': hypothesis,
            'should_create_goal': should_create_goal
        }
    
    # ✅ FIX: added thought parameter
    def _classify_thought(self, text: str, thought: Dict) -> str:
        source = thought.get('source', '')
        
        # insight_sources that are GENERATIVE (produce their own follow-up goals)
        # must be 'mundane' to break circular amplification loops.
        # But memory_recall is RETRIEVIVE — it surfaces past context that should
        # be evaluated as new input, potentially generating hypotheses.
        GENERATIVE_SOURCES = {'goal_action.self_question', 'goal_action.web_search',
                               'insight', 'reflection'}
        RETRIEVIVE_SOURCES = {'goal_action.memory_recall'}
        
        if source in GENERATIVE_SOURCES:
            return 'mundane'
        if source in RETRIEVIVE_SOURCES:
            # Treat FAISS/SQLite recalled content as external input — can generate hypotheses
            if any(kw in text for kw in self.tension_keywords):
                return 'tension'
            return 'opportunity'

        if any(kw in text for kw in self.uncertainty_keywords):
            return 'uncertainty'
        elif any(kw in text for kw in self.tension_keywords):
            return 'tension'
        elif any(kw in text for kw in self.opportunity_keywords):
            return 'opportunity'
        else:
            return 'mundane'
    
    def _extract_concept(self, text: str) -> str:
        import hashlib, re

        stop_words = {'i','me','my','the','and','but','if','or','because','as'}

        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean.split()

        meaningful = [w for w in words if w not in stop_words and len(w) >= 4]

        if meaningful:
            concept = '_'.join(meaningful[:3])
        elif words:
            h = hashlib.md5(text.encode()).hexdigest()[:6]
            concept = f"thought_{h}"
        else:
            concept = f"thought_{hashlib.md5(b'empty').hexdigest()[:6]}"

        return concept
    
    def _compute_priority(self, text: str, thought_type: str) -> float:
        """
        Priority based on type + keyword density.
        
        The 'tension' base priority is READ from sma._tension_priority if available,
        so that _record_resolution_outcome() feedback in GAE actually affects
        how aggressively new tensions get prioritized. Also loads from the
        sidecar JSON so weight survives restarts.
        """
        tension_base = self._load_tension_priority()
        base = {
            'uncertainty': 0.50,
            'tension':     tension_base,
            'opportunity': 0.55,
            'mundane':     0.20
        }.get(thought_type, 0.30)

        high_signal = ['contradiction', 'paradox', 'unresolved', 'stuck',
                       'conflict', 'fascinat', 'mystery', 'puzzle']
        boost = sum(0.05 for kw in high_signal if kw in text)
        return min(0.95, base + boost)

    def _load_tension_priority(self) -> float:
        """
        Load tension priority weight from SelfModificationAuthority or sidecar JSON.
        Sidecar at data/persona/evaluator_weights.json survives restarts.
        """
        # Try live sma reference first
        try:
            from core.state import state as _st
            ai_sys = getattr(getattr(_st, 'persona', None), 'ai_system', None)
            sma = getattr(ai_sys, 'liberty_self_mod', None)
            if sma and hasattr(sma, '_tension_priority'):
                return float(sma._tension_priority)
        except Exception:
            pass
        # Fall back to persisted sidecar
        try:
            import json as _json
            weights_path = self.persona_dir / 'evaluator_weights.json'
            if weights_path.exists():
                w = _json.loads(weights_path.read_text())
                return float(w.get('tension_priority', 0.65))
        except Exception:
            pass
        return 0.65  # hardcoded default
    
    def _compute_actionability(self, text: str, thought_type: str) -> float:
        return {
            'uncertainty': 0.7,
            'tension': 0.6,
            'opportunity': 0.8,
            'mundane': 0.2
        }.get(thought_type, 0.3)
    
    def _identify_friction(self, text: str, thought_type: str) -> str:
        if thought_type == 'uncertainty':
            return 'Lexical uncertainty vs. Conceptual understanding'
        elif thought_type == 'tension':
            return 'Conflicting beliefs or values'
        elif thought_type == 'opportunity':
            return 'Current state vs. Potential state'
        return ''
    
    def _determine_goal_type(self, thought_type: str) -> str:
        return {
            'uncertainty': 'resolve_uncertainty',
            'tension': 'resolve_tension',
            'opportunity': 'explore_opportunity',
            'mundane': 'routine'
        }.get(thought_type, 'general')
    
    def _generate_hypothesis(self, text: str, concept: str) -> str:
        """
        For high-priority tensions: generate a concrete testable hypothesis
        using the LLM. This upgrades the action from 'resolve_tension' to
        'test_hypothesis' — active simulation rather than passive tension-holding.

        Returns the hypothesis string, or empty string if LLM unavailable.
        """
        try:
            from core.state import state as _state
            llm = getattr(_state, 'llm', None)
            if not llm or not hasattr(llm, 'generate_bare'):
                return ""
            prompt = (
                f"A cognitive tension has been identified around: '{concept.replace('_', ' ')}'.\n"
                f"Context: {text[:200]}\n\n"
                f"Generate ONE concrete, testable hypothesis that could resolve this tension. "
                f"Format: 'If [action], then [expected outcome].' One sentence only."
            )
            response = llm.generate_bare(prompt, max_tokens=60, temperature=0.75)
            if response and len(response.strip()) > 15:
                hyp = response.strip()
                import logging as _lg
                _lg.getLogger(__name__).info(
                    f"[ThoughtEvaluator] 🔬 Hypothesis for '{concept}': {hyp[:80]}"
                )
                # Record for emergence metrics
                try:
                    from cognition.emergence_metrics import get_collector
                    c = get_collector()
                    if c:
                        c.record_hypothesis(hyp)
                except Exception:
                    pass
                return hyp
        except Exception:
            pass
        return ""

    def _has_existing_goal(self, goals_data: Dict, concept: str) -> bool:
        """
        Check if a goal with this concept already exists.

        BUG FIX: when using DAL, goals_data['goals'] is a dict {id: goal_dict},
        not a list. Iterating a dict yields string keys, so isinstance(goal, dict)
        was always False → silently skipped every goal → always returned False
        → duplicate goals created every cycle.
        """
        raw = goals_data.get('goals', {})
        concept_lower = concept.lower()

        # Handle both dict {id: goal_dict} (DAL path) and list [goal_dict] (file path)
        if isinstance(raw, dict):
            goal_iter = raw.values()
        else:
            goal_iter = raw

        for goal in goal_iter:
            if not isinstance(goal, dict):
                continue
            name = (goal.get('name') or goal.get('topic') or '').lower()
            if concept_lower in name or name in concept_lower:
                return True

        return False

    def _inject_followup_thoughts(self, dal, concept: str, thought_type: str) -> None:
        """
        Inject a follow-up thought, with dedup to prevent spam.

        BUG FIX: previously fired 'What about X?' every 30 min for the same concept,
        flooding the thought stream with noise and triggering re-evaluation loops.
        Now tracks last injection per concept and skips if within 2 hours.
        Also skips GAE insight sources — those already generate their own follow-ups.
        """
        import time as _t

        # Skip for GAE insight sources — they self-generate follow-ups
        if thought_type == 'opportunity':
            return

        # Per-concept dedup: only inject once per 2 hours
        if not hasattr(self, '_last_injected'):
            self._last_injected = {}
        now = _t.time()
        last = self._last_injected.get(concept, 0)
        if now - last < 7200:
            return

        self._last_injected[concept] = now
        content = (
            f"I keep returning to the idea of {concept.replace('_', ' ')} — "
            f"what's the core tension there?"
            if thought_type == 'tension'
            else f"There's still something to understand about {concept.replace('_', ' ')}."
        )
        try:
            dal.add_thought({
                'content': content,
                'source': 'thought_evaluator.followup',
                'priority': 0.50,
                'thought_type': 'reflection',
                'timestamp': now,
                'evaluated': False,
            })
        except Exception:
            pass

    # ✅ FIX: use content instead of text
    def _generate_id(self, thought: Dict) -> str:
        content = thought.get('content', '') + str(datetime.now().timestamp())
        return hashlib.md5(content.encode()).hexdigest()[:12]


if __name__ == "__main__":
    evaluator = SimpleThoughtEvaluator('data/persona')
    evaluator.evaluate_recent_thoughts()