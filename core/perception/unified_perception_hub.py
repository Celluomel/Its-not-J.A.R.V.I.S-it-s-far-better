"""
Phase 1: Unified Perception Hub

Consolidates all sensory inputs into normalized perception objects.
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

class UnifiedPerceptionHub:
    """
    Unified perception gathering system for PandoraBOX.
    
    Consolidates:
    - External: vision, audio, text, web
    - Internal: emotions, energy, homeostasis, goals
    """
    
    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.perception_history = []
        self.max_history = 100
        
    def gather_all(self, external_input: Optional[Dict] = None) -> Dict[str, Any]:
        """Gather all available perceptual inputs."""
        perception = {
            'timestamp': datetime.now().isoformat(),
            'external': self._gather_external(external_input),
            'internal': self._gather_internal()
        }
        
        self._add_to_history(perception)
        return perception
    
    def _gather_external(self, external_input: Optional[Dict]) -> Dict:
        """Gather external perceptual inputs."""
        external = {'text': None, 'vision': None, 'audio': None, 'web': None}
        
        if external_input:
            if 'text' in external_input:
                external['text'] = {
                    'content': external_input['text'],
                    'source': external_input.get('source', 'user'),
                    'timestamp': datetime.now().isoformat()
                }
        
        return external
    
    def _gather_internal(self) -> Dict:
        """Gather internal perceptual inputs."""
        return {
            'emotional_state': self._load_json('emotional_state.json'),
            'energy_level': self._load_json('energy.json'),
            'homeostasis': self._load_json('homeostasis.json'),
            'goals_status': self._compute_goals_status()
        }
    
    def _compute_goals_status(self) -> Dict:
        """Compute current goals status via DAL."""
        try:
            from core.data.access import DataAccess
            dal = DataAccess(str(self.persona_dir))
            goals = dal.get_goals()
        except Exception:
            goals_data = self._load_json('goals.json')
            raw = goals_data.get('goals', {})
            goals = list(raw.values()) if isinstance(raw, dict) else raw

        return {
            'total': len(goals),
            'active': len([g for g in goals if g.get('status') == 'active']),
            'high_priority': len([g for g in goals if g.get('priority', 0) > 0.7])
        }
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file safely."""
        filepath = self.persona_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _add_to_history(self, perception: Dict):
        """Add perception to history."""
        self.perception_history.append(perception)
        if len(self.perception_history) > self.max_history:
            self.perception_history = self.perception_history[-self.max_history:]
    
    def has_new_external_input(self, perception: Dict) -> bool:
        """Check if perception contains new external input."""
        external = perception.get('external', {})
        return any([external.get('text'), external.get('vision'), external.get('audio')])
