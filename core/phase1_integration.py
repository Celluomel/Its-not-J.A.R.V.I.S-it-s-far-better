"""
LUMINA V32 - PHASE 1 INTEGRATION

This module integrates the Phase 1 cognitive enhancements into the existing
Lumina system without breaking existing functionality.

New in V32:
- Unified decision pressure from all cognitive factors
- Thought evaluation that creates actionable goals
- Thread resolution to prevent infinite loops
- Enhanced perception gathering
- Multi-step action planning
- Complete cognitive cycle orchestration

Usage:
    from core.phase1_integration import Phase1Orchestrator
    
    orchestrator = Phase1Orchestrator()
    decision = orchestrator.enhanced_cycle(external_input)
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

class Phase1Orchestrator:
    """
    Integration orchestrator for Phase 1 enhancements.
    
    This provides a clean interface to use Phase 1 components
    alongside existing Lumina functionality.
    """
    
    def __init__(self, persona_dir: str = "data/persona", global_workspace=None):
        self.persona_dir = Path(persona_dir)
        self.cycle_count = 0
        self.enabled = True
        # Phase 4.x: optional GlobalWorkspace, threaded through to
        # WorkspaceCompetition below so compete() can publish
        # active_hypotheses. None keeps this class fully standalone.
        self._global_workspace = global_workspace
        
        # Try to load Phase 1 components
        self.components = self._initialize_components()
        
        print(f"🔧 Phase 1 Orchestrator initialized")
        print(f"   Active components: {len([c for c, v in self.components.items() if v])}")
    
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize Phase 1 components if available."""
        components = {
            'perception_hub': None,
            'pressure_calculator': None,
            'thread_resolver': None,
            'thought_evaluator': None,
            'planner': None,
            'actuator': None,
            'workspace': None  # NEW: Phase 2.0
        }
        
        # Try to load each component
        try:
            from core.perception.unified_perception_hub import UnifiedPerceptionHub
            components['perception_hub'] = UnifiedPerceptionHub(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from cognition.decision_pressure import DecisionPressureCalculator
            components['pressure_calculator'] = DecisionPressureCalculator(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from cognition.thread_resolver import ThreadResolver
            components['thread_resolver'] = ThreadResolver(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from cognition.simple_thought_evaluator import SimpleThoughtEvaluator
            components['thought_evaluator'] = SimpleThoughtEvaluator(str(self.persona_dir))
        except ImportError:
            pass
        
        try:
            from core.planning.multi_step_planner import MultiStepPlanner
            components['planner'] = MultiStepPlanner()
        except ImportError:
            pass
        
        try:
            from core.execution.actuator_interface import ActuatorInterface
            components['actuator'] = ActuatorInterface(str(self.persona_dir))
        except ImportError:
            pass
        
        # NEW: Phase 2.0 - Workspace Competition
        try:
            from cognition.workspace_competition import WorkspaceCompetition
            components['workspace'] = WorkspaceCompetition(
                str(self.persona_dir), global_workspace=self._global_workspace
            )
            print("   ✅ Phase 2.0: Workspace Competition loaded")
        except ImportError:
            pass
        
        return components
    
    def get_decision_pressure(self) -> Dict[str, float]:
        """
        Get unified decision pressure from all cognitive factors.
        
        This is the key innovation - all factors converge into one metric.
        """
        if self.components['pressure_calculator']:
            return self.components['pressure_calculator'].compute()
        else:
            # Fallback minimal pressure
            return {
                'total': 0.5,
                'goal_pressure': 0.5,
                'curiosity_drive': 0.5,
                'identity_stress': 0.3,
                'emotional_valence': 0.0,
                'energy_level': 0.5,
                'relational_pressure': 0.3,
                'interpretation': 'MODERATE (fallback)'
            }
    
    def resolve_stuck_threads(self) -> tuple:
        """
        Resolve threads stuck in infinite loops.
        Should be called periodically (e.g., every 10 cycles).
        """
        if self.components['thread_resolver']:
            return self.components['thread_resolver'].resolve_stuck_threads()
        else:
            return (0, 0)
    
    def evaluate_thoughts_and_create_goals(self) -> List[Dict]:
        """
        Evaluate recent thoughts and create goals from them.
        This bridges thought → action gap.
        """
        if self.components['thought_evaluator']:
            return self.components['thought_evaluator'].evaluate_recent_thoughts()
        else:
            return []
    
    def generate_action_plan(self, pressure: Dict, goals: List[Dict]) -> List[Dict]:
        """
        Generate multi-step action plan based on pressure and goals.
        """
        if self.components['planner']:
            return self.components['planner'].generate_plan(goals, pressure)
        else:
            # Fallback plan
            return [{
                'action_type': 'continue',
                'reason': 'No planner available',
                'priority': 0.5
            }]
    
    def gather_perception(self, external_input: Optional[Dict] = None) -> Dict:
        """
        Gather all perceptual inputs (external + internal).
        """
        if self.components['perception_hub']:
            return self.components['perception_hub'].gather_all(external_input)
        else:
            # Fallback minimal perception
            return {
                'external': external_input or {},
                'internal': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def enhanced_cycle(self, external_input: Optional[Dict] = None) -> Dict:
        """
        Run enhanced cognitive cycle with Phase 1 components.
        
        V32 PHASE 2.0 UPDATE:
        Now uses workspace competition for selective attention.
        
        KEY CHANGE:
        OLD: decision = f(all_inputs_merged)
        NEW: decision = select(winner_from_competition)
        
        This transforms Lumina from "blended mind" to "competing mind"
        
        Returns:
            Dict with 'pressure', 'plan', 'perception', 'recommendation', 'competition'
        """
        self.cycle_count += 1
        
        # Step 1: Gather perception
        perception = self.gather_perception(external_input)
        
        # Step 2: Compute decision pressure
        pressure = self.get_decision_pressure()
        
        # Step 3: Resolve threads (periodically)
        if self.cycle_count % 10 == 0:
            resolved, deferred = self.resolve_stuck_threads()
            if resolved > 0 or deferred > 0:
                print(f"   🔧 Thread resolution: {resolved} resolved, {deferred} deferred")
        
        # Step 4: Evaluate thoughts (periodically)
        goal_suggestions = []
        if self.cycle_count % 5 == 0:
            goal_suggestions = self.evaluate_thoughts_and_create_goals()
            if goal_suggestions:
                print(f"   💡 Created {len(goal_suggestions)} new goals from thoughts")
        
        # Step 5: Load current goals and thoughts
        goals = self._load_goals()
        thoughts = self._load_recent_thoughts()
        tensions = self._load_tensions()
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2.0: WORKSPACE COMPETITION
        # ═══════════════════════════════════════════════════════════════
        
        competition_result = None
        winner = None
        
        if self.components['workspace']:
            # Run competition among all candidates
            competition_result = self.components['workspace'].compete(
                goals=goals,
                thoughts=thoughts,
                tensions=tensions,
                pressure=pressure,
                context=external_input
            )
            
            winner = competition_result['winner']
            
            print(f"   🏆 Attention focus: {winner['name']} ({winner['type']}, score: {winner['score']:.2f})")
            
            # Generate plan BASED ON WINNER (not averaged inputs)
            plan = self._plan_for_winner(winner, pressure)
        else:
            # Fallback: Generate action plan (old way)
            plan = self.generate_action_plan(pressure, goals)
        
        # ═══════════════════════════════════════════════════════════════
        
        # Step 6: Make recommendation BASED ON WINNER
        if winner:
            recommendation = self._recommend_from_winner(winner, pressure, plan)
        else:
            recommendation = self._make_recommendation(pressure, plan)
        
        return {
            'cycle': self.cycle_count,
            'perception': perception,
            'pressure': pressure,
            'competition': competition_result,  # NEW
            'winner': winner,  # NEW
            'plan': plan,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_goals(self) -> List[Dict]:
        """Load current goals via DAL."""
        try:
            from core.data.access import DataAccess
            return DataAccess(str(self.persona_dir)).get_goals()
        except Exception:
            goals_file = self.persona_dir / 'goals.json'
            if goals_file.exists():
                try:
                    with open(goals_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    raw = data.get('goals', {})
                    return list(raw.values()) if isinstance(raw, dict) else raw
                except Exception:
                    pass
            return []
    
    def _load_recent_thoughts(self) -> List[Dict]:
        """Load recent thoughts for competition via DAL."""
        try:
            from core.data.access import DataAccess
            return DataAccess(str(self.persona_dir)).get_thoughts(limit=20)
        except Exception:
            pass
        thought_file = self.persona_dir / 'thought_stream.json'
        if thought_file.exists():
            try:
                with open(thought_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    thoughts = data.get('thoughts', [])
                    # Return last 10 thoughts
                    return thoughts[-10:] if len(thoughts) > 10 else thoughts
            except:
                pass
        
        return []
    
    def _load_tensions(self) -> Dict:
        """Load current tensions."""
        tension_file = self.persona_dir / 'tensions.json'
        
        if tension_file.exists():
            try:
                with open(tension_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('tensions', {})
            except:
                pass
        
        return {}
    
    def _plan_for_winner(self, winner: Dict, pressure: Dict) -> List[Dict]:
        """
        Generate plan based on competition WINNER.
        
        KEY DIFFERENCE:
        This plans for ONE focused thing, not averaged inputs.
        """
        plan = []
        
        winner_type = winner['type']
        winner_name = winner['name']
        
        if winner_type == 'goal':
            # Winner is a goal - focus on achieving it
            plan.append({
                'action_type': 'focus_on_goal',
                'goal': winner_name,
                'reason': f"Attention winner: {winner_name} (score: {winner['score']:.2f})",
                'priority': winner['score']
            })
            
            plan.append({
                'action_type': 'execute_goal_action',
                'goal': winner_name,
                'reason': 'Take action on focused goal',
                'priority': winner['score'] * 0.9
            })
        
        elif winner_type == 'thought':
            # Winner is a thought - explore it
            plan.append({
                'action_type': 'explore_thought',
                'thought': winner_name,
                'reason': f"Attention winner: {winner_name} (score: {winner['score']:.2f})",
                'priority': winner['score']
            })
            
            # Create goal from thought if actionable
            source_data = winner.get('source_data', {})
            if source_data.get('actionability', 0) > 0.6:
                plan.append({
                    'action_type': 'convert_thought_to_goal',
                    'thought': winner_name,
                    'reason': 'Thought is actionable',
                    'priority': winner['score'] * 0.8
                })
        
        elif winner_type == 'tension':
            # Winner is a tension - resolve it
            plan.append({
                'action_type': 'resolve_tension',
                'tension': winner_name,
                'reason': f"Attention winner: {winner_name} (score: {winner['score']:.2f})",
                'priority': winner['score']
            })
        
        elif winner_type == 'pressure':
            # Winner is pressure itself - urgent mode
            plan.append({
                'action_type': 'address_pressure',
                'pressure_type': winner_name,
                'reason': f"Critical pressure: {winner_name}",
                'priority': winner['score']
            })
        
        elif winner_type == 'relational':
            # Winner is relational - engage with user
            plan.append({
                'action_type': 'engage_user',
                'reason': f"High relational pressure: {winner['score']:.2f}",
                'priority': winner['score']
            })
        
        else:
            # Fallback
            plan.append({
                'action_type': 'continue',
                'reason': f"Focus on: {winner_name}",
                'priority': winner['score']
            })
        
        return plan
    
    def _recommend_from_winner(self, winner: Dict, pressure: Dict, plan: List[Dict]) -> Dict:
        """
        Make recommendation based on competition winner.
        
        KEY DIFFERENCE:
        Decision is FOCUSED on winner, not blended.
        """
        winner_type = winner['type']
        winner_name = winner['name']
        winner_score = winner['score']
        
        # Select action from plan (highest priority)
        if plan:
            action = max(plan, key=lambda x: x.get('priority', 0))
        else:
            action = {'action_type': 'idle', 'reason': 'No plan', 'priority': 0.0}
        
        return {
            'action': action.get('action_type', 'continue'),
            'reason': f"Focus: {winner_name} | {action.get('reason', '')}",
            'priority': winner_score,
            'pressure_level': pressure.get('total', 0.5),
            'pressure_interpretation': pressure.get('interpretation', 'UNKNOWN'),
            'focus': {
                'type': winner_type,
                'name': winner_name,
                'score': winner_score
            }
        }
    
    def _make_recommendation(self, pressure: Dict, plan: List[Dict]) -> Dict:
        """
        Make recommendation for action based on pressure and plan.
        """
        total_pressure = pressure.get('total', 0.5)
        
        if not plan:
            return {
                'action': 'idle',
                'reason': 'No plan generated',
                'priority': 0.1
            }
        
        # Select highest priority action
        best_action = max(plan, key=lambda x: x.get('priority', 0))
        
        return {
            'action': best_action.get('action_type', 'continue'),
            'reason': best_action.get('reason', 'Based on current pressure'),
            'priority': best_action.get('priority', 0.5),
            'pressure_level': total_pressure,
            'pressure_interpretation': pressure.get('interpretation', 'UNKNOWN')
        }
    
    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        active_components = [name for name, comp in self.components.items() if comp is not None]
        
        return {
            'enabled': self.enabled,
            'cycle_count': self.cycle_count,
            'active_components': active_components,
            'component_count': len(active_components),
            'version': 'v32_phase1'
        }

# Convenience function for easy integration
def get_enhanced_decision(external_input: Optional[Dict] = None) -> Dict:
    """
    Convenience function to get enhanced decision with Phase 1 components.
    
    Can be called from anywhere in existing Lumina code:
    
        from core.phase1_integration import get_enhanced_decision
        
        result = get_enhanced_decision({'text': 'user input'})
        pressure = result['pressure']
        recommendation = result['recommendation']
    """
    orchestrator = Phase1Orchestrator()
    return orchestrator.enhanced_cycle(external_input)

if __name__ == "__main__":
    print("="*70)
    print("PHASE 1 INTEGRATION TEST")
    print("="*70)
    
    orchestrator = Phase1Orchestrator()
    
    print(f"\nStatus: {orchestrator.get_status()}")
    
    # Test enhanced cycle
    result = orchestrator.enhanced_cycle({'text': 'Test input', 'source': 'test'})
    
    print(f"\nDecision Pressure: {result['pressure']['total']:.2f}")
    print(f"Interpretation: {result['pressure']['interpretation']}")
    print(f"Recommendation: {result['recommendation']['action']}")
    print(f"Reason: {result['recommendation']['reason']}")
    
    print("\n" + "="*70)
    print("✅ Phase 1 Integration Test Complete")
    print("="*70)
