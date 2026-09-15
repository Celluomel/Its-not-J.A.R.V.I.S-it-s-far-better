"""
Phase 1: MultiStepPlanner - Goal Decomposition and Planning

Transforms high-level goals into executable action sequences.

Example:
    Goal: "Understand quantum computing"
    Plan: 
        1. Search for introductory resources
        2. Read basic concepts
        3. Identify key terminology
        4. Store semantic definitions
        5. Synthesize understanding
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class MultiStepPlanner:
    """
    Decomposes goals into multi-step executable plans.
    
    Takes high-level goals and pressure context,
    generates concrete action sequences.
    """
    
    def __init__(self):
        # Action templates for common goal types
        self.templates = {
            'resolve_uncertainty': [
                {'action': 'search_information', 'priority': 0.8},
                {'action': 'analyze_results', 'priority': 0.7},
                {'action': 'synthesize_understanding', 'priority': 0.6},
                {'action': 'update_beliefs', 'priority': 0.5}
            ],
            'resolve_tension': [
                {'action': 'identify_conflict', 'priority': 0.9},
                {'action': 'analyze_both_sides', 'priority': 0.8},
                {'action': 'seek_resolution', 'priority': 0.7},
                {'action': 'update_beliefs', 'priority': 0.6}
            ],
            'explore_opportunity': [
                {'action': 'explore_possibilities', 'priority': 0.7},
                {'action': 'evaluate_options', 'priority': 0.6},
                {'action': 'select_approach', 'priority': 0.5},
                {'action': 'execute_experiment', 'priority': 0.8}
            ],
            'help_user': [
                {'action': 'understand_need', 'priority': 0.9},
                {'action': 'gather_resources', 'priority': 0.7},
                {'action': 'formulate_response', 'priority': 0.8},
                {'action': 'deliver_help', 'priority': 0.9}
            ],
            'learn_skill': [
                {'action': 'identify_prerequisites', 'priority': 0.6},
                {'action': 'gather_learning_resources', 'priority': 0.7},
                {'action': 'practice_incrementally', 'priority': 0.8},
                {'action': 'assess_progress', 'priority': 0.5}
            ],
            'rest': [
                {'action': 'reduce_activity', 'priority': 0.9},
                {'action': 'consolidate_memories', 'priority': 0.6},
                {'action': 'restore_energy', 'priority': 0.8}
            ]
        }
        
        print("MultiStepPlanner initialized")
    
    def generate_plan(
        self,
        goals: List[Dict],
        pressure: Dict,
        max_steps: int = 5
    ) -> List[Dict]:
        """
        Generate action plan based on goals and pressure.
        
        Args:
            goals: List of active goals
            pressure: Decision pressure context
            max_steps: Maximum steps in plan
        
        Returns:
            List of action steps, ordered by priority
        """
        if not goals:
            # No goals - use pressure to suggest action
            return self._plan_from_pressure(pressure, max_steps)
        
        # Get highest priority goal
        sorted_goals = sorted(
            goals,
            key=lambda g: g.get('priority', 0.5),
            reverse=True
        )
        
        primary_goal = sorted_goals[0]
        
        # Generate plan for this goal
        plan = self._plan_for_goal(primary_goal, pressure)
        
        # Add steps from secondary goals if space
        if len(plan) < max_steps and len(sorted_goals) > 1:
            secondary_plan = self._plan_for_goal(sorted_goals[1], pressure)
            # Add lower-priority steps
            for step in secondary_plan[:max_steps - len(plan)]:
                step['priority'] *= 0.7  # Reduce priority
                plan.append(step)
        
        # Preserve causal template order. Priority scores express importance;
        # sorting by them used to put delivery before understanding/gathering.
        plan = plan[:max_steps]
        
        # Add metadata
        for i, step in enumerate(plan):
            step['step_number'] = i + 1
            step['total_steps'] = len(plan)
            step['planned_at'] = datetime.now().isoformat()
            step['status'] = 'pending'
            step['depends_on'] = i if i > 0 else None
        
        return plan
    
    def _plan_for_goal(self, goal: Dict, pressure: Dict) -> List[Dict]:
        """
        Generate plan for a specific goal.
        """
        goal_type = goal.get('type', 'general')
        goal_name = goal.get('name', 'unknown')
        goal_priority = goal.get('priority', 0.5)
        
        # Get template for this goal type
        template = self.templates.get(goal_type, self.templates['resolve_uncertainty'])
        
        # Create plan steps
        plan = []
        for step_template in template:
            step = {
                'action': step_template['action'],
                'priority': step_template['priority'] * goal_priority,
                'goal_id': goal.get('id', ''),
                'goal_name': goal_name,
                'goal_type': goal_type,
                'reason': f"Step toward {goal_name}"
            }
            
            # Adjust priority based on pressure
            step['priority'] = self._adjust_for_pressure(
                step['priority'],
                pressure
            )
            
            plan.append(step)
        
        return plan
    
    def _plan_from_pressure(self, pressure: Dict, max_steps: int) -> List[Dict]:
        """
        Generate plan based on pressure when no explicit goals.
        """
        plan = []
        total_pressure = pressure.get('total', 0.5)
        
        # High total pressure
        if total_pressure > 0.7:
            plan.append({
                'action': 'identify_priority_goals',
                'priority': 0.9,
                'reason': f'High total pressure ({total_pressure:.2f})',
                'pressure_driven': True
            })
        
        # High curiosity
        if pressure.get('curiosity_drive', 0) > 0.6:
            plan.append({
                'action': 'explore_new_information',
                'priority': 0.7,
                'reason': f'High curiosity ({pressure["curiosity_drive"]:.2f})',
                'pressure_driven': True
            })
        
        # High identity stress
        if pressure.get('identity_stress', 0) > 0.5:
            plan.append({
                'action': 'self_reflection',
                'priority': 0.6,
                'reason': f'Identity stress ({pressure["identity_stress"]:.2f})',
                'pressure_driven': True
            })
        
        # Low energy
        if pressure.get('energy_level', 0.5) < 0.3:
            plan.append({
                'action': 'rest_or_idle',
                'priority': 0.8,
                'reason': f'Low energy ({pressure["energy_level"]:.2f})',
                'pressure_driven': True
            })
        
        # High relational pressure
        if pressure.get('relational_pressure', 0) > 0.6:
            plan.append({
                'action': 'improve_trust',
                'priority': 0.65,
                'reason': f'Relational pressure ({pressure["relational_pressure"]:.2f})',
                'pressure_driven': True
            })
        
        # Default if nothing else
        if not plan:
            plan.append({
                'action': 'continue_current_task',
                'priority': 0.5,
                'reason': 'Balanced state, maintain course',
                'pressure_driven': True
            })
        
        return plan[:max_steps]
    
    def _adjust_for_pressure(self, base_priority: float, pressure: Dict) -> float:
        """
        Adjust action priority based on pressure context.
        """
        adjusted = base_priority
        
        # Boost if high goal pressure
        if pressure.get('goal_pressure', 0) > 0.7:
            adjusted *= 1.2
        
        # Reduce if low energy
        if pressure.get('energy_level', 0.5) < 0.3:
            adjusted *= 0.8
        
        # Ensure within bounds
        return min(1.0, max(0.0, adjusted))
    
    def validate_plan(self, plan: List[Dict]) -> Dict:
        """
        Validate a plan for feasibility.
        
        Returns:
            {
                'valid': bool,
                'issues': List[str],
                'suggestions': List[str]
            }
        """
        issues = []
        suggestions = []
        
        if not plan:
            issues.append("Plan is empty")
            suggestions.append("Generate at least one action step")
        
        if len(plan) > 10:
            issues.append("Plan is too long")
            suggestions.append("Limit to 5-7 steps for better focus")
        
        # Check for duplicate actions
        actions = [s.get('action') for s in plan]
        if len(actions) != len(set(actions)):
            issues.append("Plan has duplicate actions")
            suggestions.append("Remove redundant steps")
        
        # Check priority distribution
        priorities = [s.get('priority', 0) for s in plan]
        if priorities and max(priorities) == min(priorities):
            suggestions.append("Consider varying step priorities")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def explain_plan(self, plan: List[Dict]) -> str:
        """
        Generate human-readable explanation of plan.
        """
        if not plan:
            return "No plan generated"
        
        explanation = "Action Plan:\n"
        
        for step in plan:
            step_num = step.get('step_number', 0)
            action = step.get('action', 'unknown')
            priority = step.get('priority', 0)
            reason = step.get('reason', '')
            
            explanation += f"\n{step_num}. {action} (priority: {priority:.2f})"
            if reason:
                explanation += f"\n   → {reason}"
        
        return explanation

if __name__ == "__main__":
    print("="*60)
    print("Testing MultiStepPlanner")
    print("="*60)
    
    planner = MultiStepPlanner()
    
    # Test 1: Plan with explicit goals
    print("\n1. Planning with explicit goals...")
    goals = [
        {
            'id': 'g1',
            'name': 'understand_quantum_computing',
            'type': 'resolve_uncertainty',
            'priority': 0.8
        },
        {
            'id': 'g2',
            'name': 'help_user_with_question',
            'type': 'help_user',
            'priority': 0.9
        }
    ]
    
    pressure = {
        'total': 0.6,
        'goal_pressure': 0.7,
        'curiosity_drive': 0.8,
        'energy_level': 0.5
    }
    
    plan = planner.generate_plan(goals, pressure)
    print(planner.explain_plan(plan))
    
    # Test 2: Plan from pressure alone
    print("\n2. Planning from pressure (no explicit goals)...")
    pressure_high = {
        'total': 0.75,
        'goal_pressure': 0.3,
        'curiosity_drive': 0.85,
        'energy_level': 0.6,
        'identity_stress': 0.7
    }
    
    plan = planner.generate_plan([], pressure_high)
    print(planner.explain_plan(plan))
    
    # Test 3: Validate plan
    print("\n3. Validating plan...")
    validation = planner.validate_plan(plan)
    print(f"   Valid: {validation['valid']}")
    if validation['issues']:
        print(f"   Issues: {validation['issues']}")
    if validation['suggestions']:
        print(f"   Suggestions: {validation['suggestions']}")
    
    print("\n" + "="*60)
    print("✅ MultiStepPlanner test complete")
