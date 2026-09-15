"""
Phase 1: ActuatorInterface - Action Execution Abstraction

Provides unified interface for executing actions:
- Speech output
- UI updates
- File creation
- Web search
- System commands

Decouples decision-making from execution.
"""
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json
from pathlib import Path

class ActuatorInterface:
    """
    Unified interface for action execution.
    
    Actions are executed through registered handlers.
    Each handler is a callable that takes action parameters
    and returns execution result.
    
    This decouples the cognitive engine from specific
    execution mechanisms (NiceGUI, console, API, etc.)
    """
    
    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        
        # Action handlers registry
        self.handlers = {}
        
        # Execution history
        self.execution_history = []
        self.max_history = 100
        
        # Default handlers
        self._register_default_handlers()
        
        print("⚙️  ActuatorInterface initialized")
    
    def register_handler(self, action_type: str, handler: Callable):
        """
        Register a handler for an action type.
        
        Args:
            action_type: Type of action (e.g., 'speak', 'ui_update')
            handler: Callable that executes the action
        """
        self.handlers[action_type] = handler
        print(f"   ✓ Registered handler for '{action_type}'")
    
    async def execute(self, action: Dict) -> Dict[str, Any]:
        """
        Execute an action.
        
        Args:
            action: Action specification with 'action' type and parameters
        
        Returns:
            Execution result with status, output, duration
        """
        action_type = action.get('action', 'unknown')
        start_time = datetime.now()
        
        # Get handler
        handler = self.handlers.get(action_type)
        
        if not handler:
            # No handler - log and return failure
            result = {
                'status': 'failed',
                'reason': f'No handler for action type: {action_type}',
                'action': action_type,
                'timestamp': start_time.isoformat()
            }
        else:
            try:
                # Execute handler
                output = await handler(action) if hasattr(handler, '__call__') else handler(action)
                
                result = {
                    'status': 'success',
                    'action': action_type,
                    'output': output,
                    'timestamp': start_time.isoformat(),
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            
            except Exception as e:
                result = {
                    'status': 'error',
                    'action': action_type,
                    'error': str(e),
                    'timestamp': start_time.isoformat()
                }
        
        # Record in history
        self._record_execution(action, result)
        
        return result
    
    def _register_default_handlers(self):
        """Register default action handlers."""
        
        # Thought/reflection actions
        self.register_handler('self_reflection', self._handle_reflection)
        self.register_handler('identify_priority_goals', self._handle_identify_goals)
        
        # Information gathering
        self.register_handler('search_information', self._handle_search)
        self.register_handler('explore_new_information', self._handle_explore)
        
        # Analysis
        self.register_handler('analyze_results', self._handle_analyze)
        self.register_handler('synthesize_understanding', self._handle_synthesize)
        
        # State management
        self.register_handler('update_beliefs', self._handle_update_beliefs)
        self.register_handler('rest_or_idle', self._handle_rest)
        self.register_handler('reduce_activity', self._handle_reduce_activity)
        
        # User interaction
        self.register_handler('deliver_help', self._handle_deliver_help)
        self.register_handler('formulate_response', self._handle_formulate_response)
        
        # Default/fallback
        self.register_handler('continue_current_task', self._handle_continue)
        self.register_handler('idle', self._handle_idle)
    
    # ═══════════════════════════════════════════════════════════════════
    # Default Handler Implementations
    # ═══════════════════════════════════════════════════════════════════
    
    def _handle_reflection(self, action: Dict) -> str:
        """Handle self-reflection action."""
        return "Engaging in self-reflection about recent experiences and goals"
    
    def _handle_identify_goals(self, action: Dict) -> str:
        """Handle goal identification."""
        return "Analyzing current state to identify priority goals"
    
    def _handle_search(self, action: Dict) -> str:
        """Handle information search."""
        query = action.get('query', 'general information')
        return f"Searching for information about: {query}"
    
    def _handle_explore(self, action: Dict) -> str:
        """Handle exploration."""
        return "Exploring new information and possibilities"
    
    def _handle_analyze(self, action: Dict) -> str:
        """Handle analysis."""
        return "Analyzing gathered information"
    
    def _handle_synthesize(self, action: Dict) -> str:
        """Handle synthesis."""
        return "Synthesizing understanding from multiple sources"
    
    def _handle_update_beliefs(self, action: Dict) -> str:
        """Handle belief update."""
        return "Updating beliefs based on new evidence"
    
    def _handle_rest(self, action: Dict) -> str:
        """Handle rest action."""
        return "Entering rest mode to restore energy"
    
    def _handle_reduce_activity(self, action: Dict) -> str:
        """Handle activity reduction."""
        return "Reducing cognitive activity level"
    
    def _handle_deliver_help(self, action: Dict) -> str:
        """Handle help delivery."""
        return "Delivering assistance to user"
    
    def _handle_formulate_response(self, action: Dict) -> str:
        """Handle response formulation."""
        return "Formulating response based on context"
    
    def _handle_continue(self, action: Dict) -> str:
        """Handle continue action."""
        return "Continuing current task"
    
    def _handle_idle(self, action: Dict) -> str:
        """Handle idle action."""
        return "Idling - maintaining awareness"
    
    # ═══════════════════════════════════════════════════════════════════
    # History and Status
    # ═══════════════════════════════════════════════════════════════════
    
    def _record_execution(self, action: Dict, result: Dict):
        """Record action execution in history."""
        record = {
            'action': action,
            'result': result,
            'recorded_at': datetime.now().isoformat()
        }
        
        self.execution_history.append(record)
        
        # Prune history
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
        
        # Save to file periodically
        if len(self.execution_history) % 10 == 0:
            self._save_history()
    
    def _save_history(self):
        """Save execution history to file."""
        history_file = self.persona_dir / 'execution_history.json'
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'history': self.execution_history[-100:],  # Keep last 100
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Could not save execution history: {e}")
    
    def get_recent_executions(self, limit: int = 10) -> list:
        """Get recent action executions."""
        return self.execution_history[-limit:]
    
    def get_success_rate(self, action_type: Optional[str] = None) -> float:
        """
        Calculate success rate for actions.
        
        Args:
            action_type: Optional filter by action type
        
        Returns:
            Success rate (0.0 - 1.0)
        """
        relevant = self.execution_history
        
        if action_type:
            relevant = [
                e for e in relevant
                if e.get('action', {}).get('action') == action_type
            ]
        
        if not relevant:
            return 0.0
        
        successes = sum(
            1 for e in relevant
            if e.get('result', {}).get('status') == 'success'
        )
        
        return successes / len(relevant)
    
    def get_stats(self) -> Dict:
        """Get actuator statistics."""
        total = len(self.execution_history)
        
        if total == 0:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'registered_handlers': len(self.handlers)
            }
        
        successes = sum(
            1 for e in self.execution_history
            if e.get('result', {}).get('status') == 'success'
        )
        
        # Most common actions
        action_counts = {}
        for e in self.execution_history:
            action_type = e.get('action', {}).get('action', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        most_common = sorted(
            action_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_executions': total,
            'success_rate': successes / total,
            'registered_handlers': len(self.handlers),
            'most_common_actions': dict(most_common)
        }

if __name__ == "__main__":
    import asyncio
    
    async def test_actuator():
        print("="*60)
        print("Testing ActuatorInterface")
        print("="*60)
        
        actuator = ActuatorInterface('data/persona')
        
        # Test executions
        test_actions = [
            {'action': 'self_reflection'},
            {'action': 'search_information', 'query': 'AI basics'},
            {'action': 'rest_or_idle'},
            {'action': 'unknown_action'}  # Should fail gracefully
        ]
        
        print("\n📊 Executing test actions...")
        for action in test_actions:
            result = await actuator.execute(action)
            status_icon = '✅' if result['status'] == 'success' else '❌'
            print(f"\n{status_icon} {action['action']}")
            print(f"   Status: {result['status']}")
            if 'output' in result:
                print(f"   Output: {result['output']}")
            elif 'reason' in result:
                print(f"   Reason: {result['reason']}")
        
        # Stats
        print("\n📈 Actuator Stats:")
        stats = actuator.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n" + "="*60)
        print("✅ ActuatorInterface test complete")
    
    asyncio.run(test_actuator())
