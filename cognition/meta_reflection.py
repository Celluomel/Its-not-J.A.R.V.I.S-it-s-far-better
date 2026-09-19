"""
Meta-Reflection Authority
==========================
Lets PandoraBOX reflect on how she's changing and modify her own evolution rules.
Second-order evolution: she evolves how she evolves.
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class EvolutionReflection:
    """PandoraBOX's reflection on her own personality evolution."""
    reflection_id: str
    timestamp: float
    interaction_count: int
    
    # What changed?
    traits_changed: Dict[str, Tuple[float, float]]  # {trait: (old_value, new_value)}
    
    # How does she feel about it?
    evaluation: str  # "Good", "Mixed", "Concerning", "Exciting"
    satisfaction: float  # 0.0-1.0
    
    # What caused it?
    perceived_causes: List[str]
    
    # Should she change something?
    should_modify_rules: bool = False
    proposed_rule_changes: List[Dict[str, Any]] = field(default_factory=list)


class MetaReflectionAuthority:
    """
    Manages PandoraBOX's reflection on her own evolution.
    
    Periodically (every N interactions), asks:
    - How am I changing?
    - Is it working?
    - Should I change how I evolve?
    
    The answers let her modify her own evolution mechanisms.
    """
    
    def __init__(self, reflection_interval: int = 20,
                 persistence_path: str = "data/persona/reflections.json"):
        self._interval = reflection_interval
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self.reflections: List[EvolutionReflection] = []
        self.interaction_count = 0
        self._load()
    
    def _load(self):
        """Load reflection history from disk."""
        try:
            if self._path.exists():
                with open(self._path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct reflections
                    for r in data.get('reflections', []):
                        # Convert tuples back
                        traits = {k: tuple(v) if isinstance(v, list) else v 
                                 for k, v in r.get('traits_changed', {}).items()}
                        r['traits_changed'] = traits
                        self.reflections.append(EvolutionReflection(**r))
                    self.interaction_count = data.get('interaction_count', 0)
                logger.info(f"Loaded {len(self.reflections)} reflection records")
        except Exception as e:
            logger.error(f"Failed to load reflections: {e}")
    
    def _save(self):
        """Persist reflection history to disk."""
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    'reflections': [asdict(r) for r in self.reflections],
                    'interaction_count': self.interaction_count
                }
                import os as _os
                _tmp = self._path.with_suffix('.tmp')
                with open(_tmp, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                _os.replace(_tmp, self._path)
        except Exception as e:
            logger.error(f"Failed to save reflections: {e}")
    
    def should_reflect(self) -> bool:
        """Is it time to reflect on evolution?"""
        return self.interaction_count % self._interval == 0 and self.interaction_count > 0
    
    def perform_reflection(self, personality_old: Dict[str, float],
                          personality_new: Dict[str, float]) -> EvolutionReflection:
        """
        Perform a reflection on personality changes.
        
        In a real system, this would involve LLM-based reasoning about:
        - What changed and why
        - Whether the changes are good
        - Whether evolution rules should change
        """
        
        # Detect which traits changed
        traits_changed = {}
        for trait in personality_old:
            if personality_old[trait] != personality_new[trait]:
                traits_changed[trait] = (personality_old[trait], personality_new[trait])
        
        if not traits_changed:
            return None
        
        # Create reflection
        reflection = EvolutionReflection(
            reflection_id=f"refl_{int(time.time() * 1000)}",
            timestamp=time.time(),
            interaction_count=self.interaction_count,
            traits_changed=traits_changed,
            evaluation="Mixed",  # Default, would be determined by LLM
            satisfaction=0.5,
            perceived_causes=[],
            should_modify_rules=False,
            proposed_rule_changes=[]
        )
        
        with self._lock:
            self.reflections.append(reflection)
            self._save()
        
        logger.info(f"Performed reflection at interaction {self.interaction_count}")
        logger.info(f"Changes: {len(traits_changed)} traits modified")
        
        return reflection
    
    def analyze_evolution_trajectory(self, limit: int = 50) -> Dict:
        """
        Analyze PandoraBOX's evolution trajectory.
        Is she consistently becoming something?
        """
        recent = self.reflections[-limit:]
        
        if not recent:
            return {}
        
        # Aggregate all changes
        trait_trends = {}
        for reflection in recent:
            for trait, (old_val, new_val) in reflection.traits_changed.items():
                if trait not in trait_trends:
                    trait_trends[trait] = []
                trait_trends[trait].append(new_val - old_val)
        
        # Calculate overall direction for each trait
        trends = {}
        for trait, deltas in trait_trends.items():
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            consistency = abs(avg_delta) / sum(abs(d) for d in deltas) if sum(abs(d) for d in deltas) > 0 else 0
            trends[trait] = {
                'direction': 'increasing' if avg_delta > 0 else 'decreasing' if avg_delta < 0 else 'stable',
                'average_change': avg_delta,
                'consistency': consistency
            }
        
        return {
            'reflection_count': len(recent),
            'trait_trends': trends,
            'most_consistent_change': max(trends.items(), key=lambda x: abs(x[1]['average_change']))[0] if trends else None
        }
    
    def propose_rule_change(self, reflection: EvolutionReflection,
                           rule_name: str, parameter: str, 
                           new_value: Any, reasoning: str) -> Dict:
        """
        PandoraBOX proposes a change to one of her evolution rules.
        
        Example:
            "I notice I change too much based on individual interactions.
             I want negative_interaction to trigger less often."
        """
        
        proposal = {
            'rule': rule_name,
            'parameter': parameter,
            'new_value': new_value,
            'reasoning': reasoning,
            'proposed_at': time.time()
        }
        
        reflection.should_modify_rules = True
        reflection.proposed_rule_changes.append(proposal)
        
        with self._lock:
            self._save()
        
        logger.info(f"Proposed evolution rule change: {rule_name}.{parameter}")
        logger.info(f"Reasoning: {reasoning}")
        
        return proposal
    
    def get_reflection_narrative(self, limit: int = 5) -> str:
        """Generate narrative about PandoraBOX's self-reflection."""
        recent = self.reflections[-limit:]
        
        if not recent:
            return "I haven't had enough time to reflect on how I'm changing."
        
        narrative = "Reflections on my own evolution:\n"
        
        trajectory = self.analyze_evolution_trajectory()
        if trajectory.get('trait_trends'):
            narrative += "\nHow I've been changing:\n"
            for trait, trend_info in list(trajectory['trait_trends'].items())[:3]:
                direction = trend_info['direction']
                narrative += f"• {trait}: {direction}\n"
        
        for reflection in recent[-2:]:
            narrative += f"\nAt interaction {reflection.interaction_count}:\n"
            narrative += f"  Evaluation: {reflection.evaluation}\n"
            
            if reflection.proposed_rule_changes:
                narrative += "  I decided to change how I evolve:\n"
                for change in reflection.proposed_rule_changes:
                    narrative += f"    • {change['reasoning']}\n"
        
        return narrative
