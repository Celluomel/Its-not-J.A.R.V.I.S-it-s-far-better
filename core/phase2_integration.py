"""
LUMINA V33 - PHASE 2 INTEGRATION

Phase 2 cognitive enhancements integrated:
- Advanced thought evaluation (semantic similarity)
- Thought→Goal mutation system (dynamic goal creation)
- Narrative synthesis (coherent internal monologue)
- Thread→Identity linking (beliefs evolve from threads)

Usage:
    from core.phase2_integration import Phase2Orchestrator
    
    orchestrator = Phase2Orchestrator()
    result = orchestrator.enhanced_cycle(cycle_data)
"""
from pathlib import Path
from typing import Dict, Any, Optional, List

class Phase2Orchestrator:
    """
    Integration orchestrator for Phase 2 enhancements.
    
    Builds on Phase 1 by adding:
    - Advanced (semantic) thought evaluation
    - Dynamic goal creation and mutation
    - Narrative synthesis for coherent thoughts
    - Thread-identity linkage for belief evolution
    """
    
    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.enabled = True
        
        # Initialize Phase 2 components
        self.components = self._initialize_components()
        
        print(f"🔧 Phase 2 Orchestrator initialized")
        print(f"   Active components: {len([c for c, v in self.components.items() if v])}")
    
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize Phase 2 components if available."""
        components = {
            'advanced_evaluator': None,
            'thought_goal_linker': None,
            'narrative_synthesizer': None,
            'thread_identity_linker': None
        }
        
        try:
            from cognition.advanced_thought_evaluator import AdvancedThoughtEvaluator
            components['advanced_evaluator'] = AdvancedThoughtEvaluator(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from cognition.thought_goal_linker import ThoughtGoalLinker
            components['thought_goal_linker'] = ThoughtGoalLinker(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from cognition.narrative_synthesizer import NarrativeSynthesizer
            components['narrative_synthesizer'] = NarrativeSynthesizer(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from cognition.thread_identity_linker import ThreadIdentityLinker
            components['thread_identity_linker'] = ThreadIdentityLinker(str(self.persona_dir))
        except ImportError:
            pass
        
        return components
    
    def evaluate_thoughts_advanced(self) -> List[Dict]:
        """
        Evaluate thoughts using advanced semantic analysis.
        
        Returns list of goal suggestions with enhanced metadata.
        """
        if self.components['advanced_evaluator']:
            return self.components['advanced_evaluator'].evaluate_recent_thoughts()
        return []
    
    def create_and_mutate_goals(self, evaluated_thoughts: List[Dict]) -> Dict:
        """
        Create new goals and mutate existing ones from evaluated thoughts.
        
        Returns dict with created_goals, updated_goals, merged_goals.
        """
        if not self.components['thought_goal_linker']:
            return {'created_goals': [], 'updated_goals': [], 'merged_goals': []}
        
        linker = self.components['thought_goal_linker']
        
        total_result = {
            'created_goals': [],
            'updated_goals': [],
            'linked_goals': [],
            'merged_goals': []
        }
        
        for thought_eval in evaluated_thoughts:
            result = linker.process_thought_evaluation(thought_eval)
            
            total_result['created_goals'].extend(result['created_goals'])
            total_result['updated_goals'].extend(result['updated_goals'])
            total_result['linked_goals'].extend(result['linked_goals'])
            total_result['merged_goals'].extend(result['merged_goals'])
        
        return total_result
    
    def synthesize_narrative(self, cycle_data: Dict) -> Optional[str]:
        """
        Synthesize a coherent narrative from cognitive cycle data.
        
        Returns narrative string or None.
        """
        if self.components['narrative_synthesizer']:
            return self.components['narrative_synthesizer'].create_narrative_from_cycle(cycle_data)
        return None
    
    def link_resolved_threads(self, resolved_threads: List[Dict]) -> List[Dict]:
        """
        Link resolved threads to identity changes.
        
        Returns list of identity update results.
        """
        if not self.components['thread_identity_linker']:
            return []
        
        linker = self.components['thread_identity_linker']
        
        results = []
        for thread in resolved_threads:
            result = linker.process_resolved_thread(thread)
            results.append(result)
        
        return results
    
    def enhanced_cycle(self, cycle_data: Dict) -> Dict:
        """
        Run enhanced Phase 2 cognitive processing.
        
        This extends Phase 1 with:
        - Advanced (semantic) thought evaluation
        - Dynamic goal mutation
        - Narrative synthesis
        - Thread-identity linkage
        
        Args:
            cycle_data: Dict from Phase 1 or cognitive cycle
        
        Returns:
            Dict with Phase 2 results
        """
        result = {
            'phase': 2,
            'evaluated_thoughts': [],
            'goal_mutations': {},
            'narrative': None,
            'identity_updates': []
        }
        
        # 1. Advanced thought evaluation
        if self.components['advanced_evaluator']:
            evaluated = self.evaluate_thoughts_advanced()
            result['evaluated_thoughts'] = evaluated
            
            # 2. Create/mutate goals from thoughts
            if evaluated and self.components['thought_goal_linker']:
                mutations = self.create_and_mutate_goals(evaluated)
                result['goal_mutations'] = mutations
        
        # 3. Synthesize narrative
        if self.components['narrative_synthesizer']:
            narrative = self.synthesize_narrative(cycle_data)
            result['narrative'] = narrative
        
        # 4. Link resolved threads to identity
        resolved_threads = cycle_data.get('resolved_threads', [])
        if resolved_threads and self.components['thread_identity_linker']:
            identity_updates = self.link_resolved_threads(resolved_threads)
            result['identity_updates'] = identity_updates
        
        return result
    
    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        active_components = [name for name, comp in self.components.items() if comp is not None]
        
        return {
            'enabled': self.enabled,
            'active_components': active_components,
            'component_count': len(active_components),
            'version': 'v33_phase2'
        }

# Convenience function
def get_enhanced_phase2_result(cycle_data: Dict) -> Dict:
    """
    Convenience function to get Phase 2 enhancements.
    
    Can be called from anywhere in existing Lumina code.
    """
    orchestrator = Phase2Orchestrator()
    return orchestrator.enhanced_cycle(cycle_data)

if __name__ == "__main__":
    print("="*70)
    print("PHASE 2 INTEGRATION TEST")
    print("="*70)
    
    orchestrator = Phase2Orchestrator()
    
    print(f"\nStatus: {orchestrator.get_status()}")
    
    # Test enhanced cycle
    test_cycle_data = {
        'pressure': {
            'total': 0.68,
            'goal_pressure': 0.72,
            'curiosity_drive': 0.65,
            'identity_stress': 0.45,
            'energy_level': 0.58
        },
        'goals': [
            {'name': 'understand_consciousness', 'priority': 0.75, 'status': 'active'}
        ],
        'emotions': {'curiosity': 0.7, 'calm': 0.5},
        'traits': {'empathy': 0.82, 'curiosity': 0.78},
        'energy_level': 0.58
    }
    
    result = orchestrator.enhanced_cycle(test_cycle_data)
    
    print(f"\nPhase 2 Result:")
    print(f"  Evaluated thoughts: {len(result['evaluated_thoughts'])}")
    print(f"  Goal mutations: {result['goal_mutations']}")
    print(f"  Narrative: {result['narrative'][:80] if result['narrative'] else 'None'}...")
    print(f"  Identity updates: {len(result['identity_updates'])}")
    
    print("\n" + "="*70)
    print("✅ Phase 2 Integration Test Complete")
    print("="*70)
