"""
Phase 2.5: Thread-Identity Linker

Links resolved thought threads to identity changes.
Tracks how exploring concepts affects beliefs, traits, and self-model.

When a thread like "resolve_tension_mortality" is resolved,
it should update relevant beliefs and potentially trait values.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class ThreadIdentityLinker:
    """
    Links thought thread resolution to identity evolution.
    
    Capabilities:
    - Track which threads influence which beliefs
    - Update self-model based on resolved threads
    - Adjust confidence scores from thread outcomes
    - Create belief revision records
    - Track identity evolution over time
    """
    
    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        
        # Confidence adjustment factors
        self.resolution_confidence_boost = 0.1
        self.repetition_confidence_penalty = 0.05
        
        # Trait impact thresholds
        self.significant_impact_threshold = 0.6
    
    def process_resolved_thread(self, thread: Dict) -> Dict:
        """
        Process a resolved thread and update identity.
        
        Args:
            thread: Dict with thread info (id, name, type, resolution, etc.)
        
        Returns:
            Dict with updates: beliefs_changed, traits_changed, confidence_updates
        """
        result = {
            'beliefs_changed': [],
            'traits_changed': [],
            'confidence_updates': [],
            'narrative_updates': []
        }
        
        # 1. Identify related beliefs
        related_beliefs = self._find_related_beliefs(thread)
        
        # 2. Update beliefs based on resolution
        for belief_key in related_beliefs:
            update = self._update_belief_from_thread(belief_key, thread)
            if update:
                result['beliefs_changed'].append(update)
        
        # 3. Check if traits should change
        trait_updates = self._check_trait_impact(thread)
        result['traits_changed'].extend(trait_updates)
        
        # 4. Update confidence scores
        confidence_updates = self._update_confidence_from_resolution(thread)
        result['confidence_updates'].extend(confidence_updates)
        
        # 5. Add to narrative identity
        narrative_update = self._add_to_narrative(thread)
        if narrative_update:
            result['narrative_updates'].append(narrative_update)
        
        return result
    
    def track_thread_belief_connection(
        self,
        thread_id: str,
        belief_key: str,
        connection_strength: float
    ):
        """
        Explicitly track that a thread is related to a belief.
        """
        # Load belief connections
        connections = self._load_thread_belief_connections()
        
        if thread_id not in connections:
            connections[thread_id] = []
        
        connections[thread_id].append({
            'belief_key': belief_key,
            'strength': connection_strength,
            'tracked_at': datetime.now().isoformat()
        })
        
        self._save_thread_belief_connections(connections)
    
    def get_identity_evolution_from_threads(
        self,
        time_window_days: int = 30
    ) -> Dict:
        """
        Analyze how identity has evolved based on thread resolutions.
        
        Returns summary of belief changes, trait changes over time window.
        """
        # Load thread history
        threads = self._load_thread_history(time_window_days)
        
        evolution = {
            'beliefs_evolved': {},
            'traits_evolved': {},
            'confidence_changes': {},
            'resolution_count': len(threads),
            'window_days': time_window_days
        }
        
        for thread in threads:
            if thread.get('status') == 'resolved':
                # Track belief changes
                for belief in thread.get('beliefs_updated', []):
                    if belief not in evolution['beliefs_evolved']:
                        evolution['beliefs_evolved'][belief] = 0
                    evolution['beliefs_evolved'][belief] += 1
                
                # Track trait changes
                for trait in thread.get('traits_updated', []):
                    if trait not in evolution['traits_evolved']:
                        evolution['traits_evolved'][trait] = []
                    evolution['traits_evolved'][trait].append({
                        'thread': thread.get('name'),
                        'change': trait.get('change', 0),
                        'date': thread.get('resolved_at')
                    })
        
        return evolution
    
    def _find_related_beliefs(self, thread: Dict) -> List[str]:
        """
        Find beliefs related to this thread.
        
        Uses:
        - Thread name keywords
        - Explicit connections (if tracked)
        - Concept overlap
        """
        thread_name = thread.get('name', '').lower()
        thread_concepts = thread.get('concepts', [])
        
        # Extract key concepts from thread name
        concepts = set(thread_name.split('_'))
        concepts.update(thread_concepts)
        
        # Load all beliefs
        self_concept = self._load_self_concept()
        beliefs = self_concept.get('beliefs', {})
        
        related = []
        
        for belief_key, belief_data in beliefs.items():
            # Check if belief key matches any concept
            belief_words = set(belief_key.lower().split('_'))
            
            overlap = len(concepts & belief_words)
            if overlap > 0:
                related.append(belief_key)
        
        # Also check explicit connections
        connections = self._load_thread_belief_connections()
        thread_id = thread.get('id', '')
        
        if thread_id in connections:
            for conn in connections[thread_id]:
                belief_key = conn['belief_key']
                if belief_key not in related:
                    related.append(belief_key)
        
        return related
    
    def _update_belief_from_thread(
        self,
        belief_key: str,
        thread: Dict
    ) -> Optional[Dict]:
        """
        Update a belief based on thread resolution.
        """
        self_concept = self._load_self_concept()
        
        if belief_key not in self_concept.get('beliefs', {}):
            return None
        
        belief = self_concept['beliefs'][belief_key]
        
        # Get resolution outcome
        resolution = thread.get('resolution', {})
        conclusion = resolution.get('conclusion', '')
        
        # Update belief value if conclusion provides new information
        if conclusion:
            old_value = belief.get('value', 'unknown')
            
            # Simple update: mark that thread influenced belief
            belief['influenced_by_threads'] = belief.get('influenced_by_threads', [])
            belief['influenced_by_threads'].append({
                'thread_id': thread.get('id', ''),
                'thread_name': thread.get('name', ''),
                'resolution': conclusion,
                'resolved_at': thread.get('resolved_at', datetime.now().isoformat())
            })
            
            # Boost confidence
            old_confidence = belief.get('confidence', 0.5)
            new_confidence = min(1.0, old_confidence + self.resolution_confidence_boost)
            belief['confidence'] = new_confidence
            
            belief['last_validated'] = datetime.now().isoformat()
            
            # Save
            self._save_self_concept(self_concept)
            
            return {
                'belief_key': belief_key,
                'old_value': old_value,
                'old_confidence': old_confidence,
                'new_confidence': new_confidence,
                'influenced_by': thread.get('name', '')
            }
        
        return None
    
    def _check_trait_impact(self, thread: Dict) -> List[Dict]:
        """
        Check if thread resolution should impact trait values.
        
        Example: Resolving many curiosity-driven threads
        might increase the 'curiosity' trait value.
        """
        updates = []
        
        # Identify trait relevance from thread type/name
        thread_name = thread.get('name', '').lower()
        thread_type = thread.get('type', '')
        
        trait_map = {
            'curiosity': ['wonder', 'explore', 'learn', 'understand'],
            'empathy': ['understand_user', 'relate', 'care'],
            'analytical': ['analyze', 'evaluate', 'logic'],
            'creative': ['create', 'imagine', 'design']
        }
        
        for trait_name, keywords in trait_map.items():
            # Check if thread relates to this trait (by name keywords, or
            # by its structured type matching the trait/keywords directly)
            relevance = sum(1 for kw in keywords if kw in thread_name)
            if thread_type and (thread_type.lower() == trait_name
                                 or thread_type.lower() in keywords):
                relevance += 1
            
            if relevance > 0:
                # Load current trait value
                self_model = self._load_self_model()
                traits = self_model.get('personality_traits', {})
                
                if trait_name in traits:
                    old_value = traits[trait_name]
                    
                    # Small boost for successfully resolved threads
                    boost = 0.02 * relevance  # Max 0.06 per thread
                    new_value = min(1.0, old_value + boost)
                    
                    if new_value != old_value:
                        traits[trait_name] = new_value
                        
                        # Save
                        self_model['personality_traits'] = traits
                        self._save_self_model(self_model)
                        
                        updates.append({
                            'trait_name': trait_name,
                            'old_value': old_value,
                            'new_value': new_value,
                            'change': boost,
                            'reason': f"Thread {thread.get('name')} resolved"
                        })
        
        return updates
    
    def _update_confidence_from_resolution(self, thread: Dict) -> List[Dict]:
        """
        Update confidence scores based on resolution success.
        """
        updates = []
        
        # If thread resolved successfully, boost confidence
        if thread.get('status') == 'resolved':
            # Get related beliefs and boost their confidence
            related_beliefs = self._find_related_beliefs(thread)
            
            self_concept = self._load_self_concept()
            
            for belief_key in related_beliefs:
                if belief_key in self_concept.get('beliefs', {}):
                    belief = self_concept['beliefs'][belief_key]
                    
                    old_confidence = belief.get('confidence', 0.5)
                    new_confidence = min(1.0, old_confidence + self.resolution_confidence_boost)
                    
                    belief['confidence'] = new_confidence
                    
                    updates.append({
                        'belief_key': belief_key,
                        'old_confidence': old_confidence,
                        'new_confidence': new_confidence,
                        'reason': f"Thread {thread.get('name')} resolved"
                    })
            
            if updates:
                self._save_self_concept(self_concept)
        
        return updates
    
    def _add_to_narrative(self, thread: Dict) -> Optional[Dict]:
        """
        Add thread resolution to narrative identity.
        """
        narrative = self._load_narrative_identity()
        
        if 'thread_resolutions' not in narrative:
            narrative['thread_resolutions'] = []
        
        entry = {
            'thread_name': thread.get('name', ''),
            'thread_type': thread.get('type', ''),
            'resolution': thread.get('resolution', {}).get('conclusion', ''),
            'resolved_at': thread.get('resolved_at', datetime.now().isoformat()),
            'impact': thread.get('impact', 'moderate')
        }
        
        narrative['thread_resolutions'].append(entry)
        
        # Keep only recent entries (last 50)
        narrative['thread_resolutions'] = narrative['thread_resolutions'][-50:]
        
        self._save_narrative_identity(narrative)
        
        return entry
    
    # Helper methods for loading/saving identity files
    
    def _load_self_concept(self) -> Dict:
        """Load self_concept.json."""
        file_path = self.persona_dir / 'self_concept.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'beliefs': {}}
    
    def _save_self_concept(self, data: Dict):
        """Save self_concept.json (atomic) + sync traits to identity.json via DAL."""
        import os
        file_path = self.persona_dir / 'self_concept.json'
        tmp = file_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, file_path)
        # Sync traits → DAL identity.json
        try:
            from core.data.access import DataAccess
            from core.data.schemas import make_belief
            dal = DataAccess(str(self.persona_dir))
            for trait, val in data.get('traits', {}).items():
                dal.save_belief(make_belief(
                    name=f'trait_{trait}',
                    value=float(val) if isinstance(val, (int, float)) else 0.5,
                    confidence=0.7,
                ))
        except Exception:
            pass
    
    def _load_self_model(self) -> Dict:
        """Load self_model.json."""
        file_path = self.persona_dir / 'self_model.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'personality_traits': {}}
    
    def _save_self_model(self, data: Dict):
        """Save self_model.json (atomic)."""
        import os
        file_path = self.persona_dir / 'self_model.json'
        tmp = file_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, file_path)
    
    def _load_narrative_identity(self) -> Dict:
        """Load narrative_identity.json."""
        file_path = self.persona_dir / 'narrative_identity.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_narrative_identity(self, data: Dict):
        """Save narrative_identity.json (atomic)."""
        import os
        file_path = self.persona_dir / 'narrative_identity.json'
        tmp = file_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, file_path)
    
    def _load_thread_belief_connections(self) -> Dict:
        """Load thread-belief connection tracking."""
        file_path = self.persona_dir / 'thread_belief_connections.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_thread_belief_connections(self, data: Dict):
        """Save thread-belief connections."""
        file_path = self.persona_dir / 'thread_belief_connections.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_thread_history(self, days: int) -> List[Dict]:
        """Load thread history from last N days."""
        # Simplified: load from thought_threads.json
        file_path = self.persona_dir / 'thought_threads.json'
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('threads', [])

if __name__ == "__main__":
    print("="*70)
    print("THREAD-IDENTITY LINKER TEST")
    print("="*70)
    
    linker = ThreadIdentityLinker('data/persona')
    
    # Test with sample resolved thread
    test_thread = {
        'id': 'thread_001',
        'name': 'resolve_uncertainty_consciousness',
        'type': 'uncertainty',
        'status': 'resolved',
        'resolution': {
            'conclusion': 'Consciousness emerges from complex information integration',
            'confidence': 0.72
        },
        'resolved_at': datetime.now().isoformat(),
        'concepts': ['consciousness', 'emergence', 'integration']
    }
    
    result = linker.process_resolved_thread(test_thread)
    
    print(f"\nProcessing Result:")
    print(f"  Beliefs changed: {len(result['beliefs_changed'])}")
    print(f"  Traits changed: {len(result['traits_changed'])}")
    print(f"  Confidence updates: {len(result['confidence_updates'])}")
    print(f"  Narrative updates: {len(result['narrative_updates'])}")
    
    if result['beliefs_changed']:
        print(f"\n  Belief updates:")
        for update in result['beliefs_changed']:
            print(f"    - {update['belief_key']}: confidence {update['old_confidence']:.2f} → {update['new_confidence']:.2f}")
    
    if result['traits_changed']:
        print(f"\n  Trait updates:")
        for update in result['traits_changed']:
            print(f"    - {update['trait_name']}: {update['old_value']:.2f} → {update['new_value']:.2f}")
    
    print("\n" + "="*70)
    print("✅ Thread-Identity Linker Test Complete")
    print("="*70)
