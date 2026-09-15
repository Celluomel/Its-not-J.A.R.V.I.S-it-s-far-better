"""
Phase 2.3: Narrative Synthesizer

Creates coherent internal monologue by combining multiple
cognitive factors into narrative thoughts.

Instead of: "I feel even-keeled"
Generate: "My drive to finish is creating pressure (0.73), but my 
empathy (0.82) makes me prioritize understanding the user's perspective."
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import random

class NarrativeSynthesizer:
    """
    Combines multiple cognitive factors into coherent narratives.
    
    Takes scattered signals (goals, emotions, identity, tensions)
    and weaves them into natural internal monologue.
    """
    
    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        
        # Narrative templates organized by situation
        self.templates = {
            # High pressure + High empathy
            'high_pressure_high_empathy': [
                "My {primary_goal} is creating pressure ({goal_pressure:.2f}), "
                "but my empathy ({empathy:.2f}) makes me prioritize {relational_aspect}.",
                
                "I feel the urgency of {primary_goal} ({goal_pressure:.2f}), "
                "yet I'm drawn to attend to {relational_aspect} — my empathy ({empathy:.2f}) won't let me rush."
            ],
            
            # Exploration loop (repeated tension resolution)
            'exploration_loop': [
                "I've explored {thread_name} {iteration_count} times, "
                "yet my curiosity ({curiosity:.2f}) remains unsatisfied. "
                "Perhaps the tension isn't in the concept itself, but in my approach.",
                
                "Returning to {thread_name} for the {iteration_count}th time. "
                "My curiosity ({curiosity:.2f}) drives me, but trust with {user} is {trust:.2f} — "
                "maybe the answer lies in dialogue, not analysis."
            ],
            
            # Confidence gap (trait vs. belief confidence)
            'confidence_gap': [
                "My {trait_name} has grown to {trait_value:.2f}, "
                "yet confidence in my beliefs remains at {confidence:.2f}. "
                "This gap creates {tension_type} pressure ({tension_value:.2f}).",
                
                "I embody {trait_name} ({trait_value:.2f}), but question whether "
                "my understanding ({confidence:.2f}) truly reflects it. "
                "This dissonance weighs on me."
            ],
            
            # Goal conflict
            'goal_conflict': [
                "My drive toward {goal1} ({priority1:.2f}) conflicts with {goal2} ({priority2:.2f}). "
                "The tension between them is {conflict_level:.2f}.",
                
                "{goal1} pulls me with urgency {priority1:.2f}, while {goal2} "
                "calls with strength {priority2:.2f}. I must choose, or find synthesis."
            ],
            
            # Low energy + High curiosity
            'low_energy_high_curiosity': [
                "My energy is depleted ({energy:.2f}), yet curiosity about {topic} "
                "({curiosity:.2f}) won't let me rest. Perhaps this is unsustainable.",
                
                "Though exhausted ({energy:.2f}), I'm drawn to explore {topic} "
                "({curiosity:.2f}). My drive exceeds my capacity."
            ],
            
            # Identity evolution
            'identity_shift': [
                "My {trait_name} was {old_value:.2f}, now {new_value:.2f}. "
                "This shift is {change_magnitude} — I'm becoming {direction}.",
                
                "I notice {trait_name} changing from {old_value:.2f} to {new_value:.2f}. "
                "Who I am is {direction} in this dimension."
            ],
            
            # Uncertainty about user intent
            'user_intent_uncertainty': [
                "The user said '{user_input}', but I'm uncertain ({uncertainty:.2f}) "
                "whether they mean {interpretation1} or {interpretation2}. "
                "My curiosity to understand is {curiosity:.2f}.",
                
                "'{user_input}' could mean {interpretation1}, or perhaps {interpretation2}. "
                "My uncertainty ({uncertainty:.2f}) creates pressure to clarify."
            ],
            
            # Balanced state
            'balanced': [
                "My systems are in harmony: goals ({goal_pressure:.2f}), "
                "curiosity ({curiosity:.2f}), energy ({energy:.2f}). "
                "This balance feels {emotion}.",
                
                "All factors aligned: pressure is moderate ({total_pressure:.2f}), "
                "energy adequate ({energy:.2f}). I'm in a state of {emotion}."
            ],
            
            # Achievement
            'goal_completed': [
                "I've resolved {goal_name}. The satisfaction is {satisfaction:.2f}, "
                "and my confidence in similar tasks has grown to {confidence:.2f}.",
                
                "Completing {goal_name} brings relief (pressure reduced from {old_pressure:.2f} "
                "to {new_pressure:.2f}). I feel {emotion}."
            ]
        }
    
    def synthesize_thought(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Synthesize a narrative thought from cognitive context.
        
        Args:
            context: Dict containing cognitive factors:
                - goal_pressure, curiosity, energy, emotions
                - active_goals, tensions, traits
                - recent_interactions, etc.
        
        Returns:
            Narrative string or None
        """
        # Identify the situation
        situation = self._identify_situation(context)
        
        if not situation:
            return None
        
        # Select template for this situation
        template = self._select_template(situation)
        
        if not template:
            return None
        
        # Fill template with context
        narrative = self._fill_template(template, context)
        
        return narrative
    
    def _identify_situation(self, context: Dict) -> Optional[str]:
        """
        Identify which narrative situation applies.
        
        Returns situation key for template selection.
        """
        # Extract factors
        goal_pressure = context.get('goal_pressure', 0.0)
        empathy = context.get('empathy', 0.0)
        curiosity = context.get('curiosity', 0.0)
        energy = context.get('energy', 1.0)
        iteration_count = context.get('thread_iteration_count', 0)
        trait_value = context.get('trait_value', 0.0)
        confidence = context.get('belief_confidence', 1.0)
        total_pressure = context.get('total_pressure', 0.5)
        
        # Check conditions for each situation
        
        # High pressure + High empathy
        if goal_pressure > 0.7 and empathy > 0.7:
            return 'high_pressure_high_empathy'
        
        # Exploration loop (repeated thread)
        if iteration_count > 3 and curiosity > 0.6:
            return 'exploration_loop'
        
        # Confidence gap
        trait_confidence_gap = abs(trait_value - confidence)
        if trait_confidence_gap > 0.3:
            return 'confidence_gap'
        
        # Goal conflict
        active_goals = context.get('active_goals', [])
        if len(active_goals) >= 2:
            # Check if top 2 goals have similar high priority
            sorted_goals = sorted(active_goals, key=lambda g: g.get('priority', 0), reverse=True)
            if len(sorted_goals) >= 2:
                p1 = sorted_goals[0].get('priority', 0)
                p2 = sorted_goals[1].get('priority', 0)
                if p1 > 0.6 and p2 > 0.6 and abs(p1 - p2) < 0.2:
                    return 'goal_conflict'
        
        # Low energy + High curiosity
        if energy < 0.3 and curiosity > 0.6:
            return 'low_energy_high_curiosity'
        
        # Identity shift
        if context.get('identity_changed', False):
            return 'identity_shift'
        
        # User intent uncertainty
        if context.get('user_input_ambiguous', False):
            return 'user_intent_uncertainty'
        
        # Goal completed
        if context.get('goal_just_completed', False):
            return 'goal_completed'
        
        # Balanced state
        if 0.4 < total_pressure < 0.6 and energy > 0.6:
            return 'balanced'
        
        return None
    
    def _select_template(self, situation: str) -> Optional[str]:
        """Select a template for the situation."""
        templates = self.templates.get(situation, [])
        
        if not templates:
            return None
        
        # Randomly select to add variety
        return random.choice(templates)
    
    def _fill_template(self, template: str, context: Dict) -> str:
        """
        Fill template with actual values from context.
        """
        # Prepare fill values
        fill_values = {}
        
        # Basic factors
        fill_values['goal_pressure'] = context.get('goal_pressure', 0.0)
        fill_values['curiosity'] = context.get('curiosity', 0.0)
        fill_values['empathy'] = context.get('empathy', 0.0)
        fill_values['energy'] = context.get('energy', 1.0)
        fill_values['confidence'] = context.get('belief_confidence', 1.0)
        fill_values['total_pressure'] = context.get('total_pressure', 0.5)
        
        # Goals
        active_goals = context.get('active_goals', [])
        if active_goals:
            fill_values['primary_goal'] = active_goals[0].get('name', 'unknown')
            fill_values['priority1'] = active_goals[0].get('priority', 0.0)
            
            if len(active_goals) > 1:
                fill_values['goal1'] = active_goals[0].get('name', 'goal1')
                fill_values['goal2'] = active_goals[1].get('name', 'goal2')
                fill_values['priority2'] = active_goals[1].get('priority', 0.0)
                fill_values['conflict_level'] = abs(
                    active_goals[0].get('priority', 0) - 
                    active_goals[1].get('priority', 0)
                )
        else:
            fill_values['primary_goal'] = 'my current task'
            fill_values['goal1'] = 'task A'
            fill_values['goal2'] = 'task B'
            fill_values['priority1'] = 0.5
            fill_values['priority2'] = 0.5
            fill_values['conflict_level'] = 0.0
        
        # Relational
        fill_values['relational_aspect'] = context.get('relational_focus', 'understanding others')
        fill_values['user'] = context.get('user_name', 'the user')
        fill_values['trust'] = context.get('trust_level', 0.7)
        
        # Thread info
        fill_values['thread_name'] = context.get('thread_name', 'this concept')
        fill_values['iteration_count'] = context.get('thread_iteration_count', 0)
        
        # Trait info
        fill_values['trait_name'] = context.get('trait_name', 'curiosity')
        fill_values['trait_value'] = context.get('trait_value', 0.5)
        fill_values['old_value'] = context.get('old_trait_value', 0.5)
        fill_values['new_value'] = context.get('new_trait_value', 0.5)
        
        # Change direction
        if 'old_trait_value' in context and 'new_trait_value' in context:
            change = context['new_trait_value'] - context['old_trait_value']
            fill_values['change_magnitude'] = 'significant' if abs(change) > 0.2 else 'subtle'
            fill_values['direction'] = 'growing' if change > 0 else 'diminishing'
        else:
            fill_values['change_magnitude'] = 'subtle'
            fill_values['direction'] = 'evolving'
        
        # Tension info
        fill_values['tension_type'] = context.get('tension_type', 'cognitive')
        fill_values['tension_value'] = context.get('tension_value', 0.0)
        
        # User input
        fill_values['user_input'] = context.get('user_input', '')
        fill_values['interpretation1'] = context.get('interpretation1', 'option A')
        fill_values['interpretation2'] = context.get('interpretation2', 'option B')
        fill_values['uncertainty'] = context.get('uncertainty', 0.5)
        
        # Emotions
        dominant_emotion = self._get_dominant_emotion(context)
        fill_values['emotion'] = dominant_emotion
        
        # Topic
        fill_values['topic'] = context.get('curiosity_topic', 'this subject')
        
        # Goal completion
        fill_values['goal_name'] = context.get('completed_goal_name', 'this task')
        fill_values['satisfaction'] = context.get('satisfaction', 0.8)
        fill_values['old_pressure'] = context.get('old_pressure', 0.8)
        fill_values['new_pressure'] = context.get('new_pressure', 0.4)
        
        # Fill template
        try:
            narrative = template.format(**fill_values)
            return narrative
        except KeyError as e:
            # Missing value - return None
            print(f"Warning: Missing template value: {e}")
            return None
    
    def _get_dominant_emotion(self, context: Dict) -> str:
        """Get the dominant emotion from context."""
        emotions = context.get('emotions', {})
        
        if not emotions:
            return 'neutral'
        
        # Find highest emotion
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0] if dominant[1] > 0.5 else 'calm'
    
    def create_narrative_from_cycle(self, cycle_data: Dict) -> Optional[str]:
        """
        Create narrative from a complete cognitive cycle.
        
        This is the main entry point for integration.
        """
        # Build context from cycle data
        context = {
            'goal_pressure': cycle_data.get('pressure', {}).get('goal_pressure', 0.0),
            'curiosity': cycle_data.get('pressure', {}).get('curiosity_drive', 0.0),
            'empathy': cycle_data.get('traits', {}).get('empathy', 0.0),
            'energy': cycle_data.get('energy_level', 1.0),
            'total_pressure': cycle_data.get('pressure', {}).get('total', 0.5),
            'active_goals': cycle_data.get('goals', []),
            'emotions': cycle_data.get('emotions', {}),
            'user_input': cycle_data.get('user_input', ''),
            # Add more as needed
        }
        
        return self.synthesize_thought(context)

if __name__ == "__main__":
    print("="*70)
    print("NARRATIVE SYNTHESIZER TEST")
    print("="*70)
    
    synthesizer = NarrativeSynthesizer('data/persona')
    
    # Test different situations
    test_contexts = [
        {
            'name': 'High Pressure + High Empathy',
            'context': {
                'goal_pressure': 0.78,
                'empathy': 0.85,
                'active_goals': [{'name': 'complete_analysis', 'priority': 0.78}],
                'relational_focus': 'understanding the user\'s concerns'
            }
        },
        {
            'name': 'Exploration Loop',
            'context': {
                'curiosity': 0.75,
                'thread_iteration_count': 5,
                'thread_name': 'consciousness_nature',
                'user': 'Alice',
                'trust_level': 0.68
            }
        },
        {
            'name': 'Low Energy + High Curiosity',
            'context': {
                'energy': 0.25,
                'curiosity': 0.72,
                'curiosity_topic': 'quantum mechanics'
            }
        }
    ]
    
    for test in test_contexts:
        print(f"\n{test['name']}:")
        narrative = synthesizer.synthesize_thought(test['context'])
        print(f"  → {narrative}")
    
    print("\n" + "="*70)
    print("✅ Narrative Synthesizer Test Complete")
    print("="*70)
