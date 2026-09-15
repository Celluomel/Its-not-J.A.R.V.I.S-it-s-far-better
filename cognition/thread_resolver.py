"""
Quick Win #2: Fix infinite "resolve tension" loops.
Adds resolution criteria and max iterations to prevent looping.
"""
import json
from pathlib import Path
from datetime import datetime

class ThreadResolver:
    """
    Fixes thought threads stuck in infinite 'resolve tension' loops
    by adding resolution criteria and maximum iteration limits.
    """
    
    def __init__(self, persona_dir):
        self.persona_dir = Path(persona_dir)
        self.threads_file = self.persona_dir / 'thought_threads.json'
        self.tensions_file = self.persona_dir / 'tensions.json'
        
        # Resolution criteria
        self.max_iterations = 20
        self.resolution_threshold = 0.2  # Tension below this = resolved
    
    def resolve_stuck_threads(self):
        """
        Check all active threads and resolve/defer infinite loops.
        Returns number of threads resolved.
        """
        # Load via DAL (thread-safe, schema-aware)
        try:
            from core.data.access import DataAccess
            _dal = DataAccess(str(self.threads_file.parent))
            threads_list = _dal.get_threads()
            tensions_flat = _dal.get_tensions()
            tensions = {'current': tensions_flat}
            _use_dal = True
        except Exception:
            if not self.threads_file.exists():
                print("⚠️  thought_threads.json not found")
                return 0
            with open(self.threads_file, 'r', encoding='utf-8') as f:
                threads_data = json.load(f)
            threads_list = list(threads_data.values()) if isinstance(threads_data, dict) else threads_data.get('threads', [])
            tensions = self._load_tensions()
            _use_dal = False

        resolved_count = 0
        deferred_count = 0

        for thread in threads_list:
            # Skip if already resolved
            if thread.get('status') == 'resolved':
                continue
            
            # Only process 'resolve tension' actions
            if thread.get('action') != 'resolve tension':
                continue
            
            # Get iteration count
            iteration_count = len(thread.get('history', []))
            if not iteration_count:
                # Try counting from other fields
                iteration_count = thread.get('iteration_count', 0)
            
            # Get current tension level for this object
            tension_object = thread.get('object', '')
            current_tension = self._get_tension_level(tensions, tension_object)
            
            # Apply resolution criteria
            if current_tension < self.resolution_threshold:
                # Tension is low enough - mark as resolved
                thread['status'] = 'resolved'
                thread['resolution_criteria_met'] = True
                thread['resolved_at'] = datetime.now().isoformat()
                thread['resolution_reason'] = f'Tension below threshold ({current_tension:.2f} < {self.resolution_threshold})'
                resolved_count += 1
                print(f"   ✅ Resolved thread '{tension_object}' - tension: {current_tension:.2f}")
                
            elif iteration_count > self.max_iterations:
                # Too many iterations without resolution - defer
                thread['status'] = 'deferred'
                thread['deferred_reason'] = f'Cannot resolve with current confidence after {iteration_count} iterations'
                thread['deferred_at'] = datetime.now().isoformat()
                thread['current_tension'] = current_tension
                deferred_count += 1
                print(f"   ⏸️  Deferred thread '{tension_object}' - {iteration_count} iterations")
            
            else:
                # Still active, add metadata for tracking
                thread['current_tension'] = current_tension
                thread['iteration_count'] = iteration_count
        
        # Save updated threads via DAL
        if _use_dal:
            for thread in threads_list:
                _dal.save_thread(thread)
        else:
            import os
            tmp = self.threads_file.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(threads_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.threads_file)

        return resolved_count, deferred_count
    
    def _load_tensions(self):
        """Load current tensions via DAL."""
        try:
            from core.data.access import DataAccess
            flat = DataAccess(str(self.tensions_file.parent)).get_tensions()
            return {'current': flat}
        except Exception:
            if not self.tensions_file.exists():
                return {}
            try:
                with open(self.tensions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load tensions: {e}")
                return {}
    
    def _get_tension_level(self, tensions, object_name):
        """
        Get tension level for a specific object.
        Looks for matching keys in tensions dict.
        """
        if not tensions or not object_name:
            return 0.5  # Default medium tension
        
        object_lower = object_name.lower()
        
        # Direct match
        if object_lower in tensions:
            return tensions[object_lower]
        
        # Partial match
        for key, value in tensions.items():
            if isinstance(value, (int, float)):
                if object_lower in key.lower() or key.lower() in object_lower:
                    return value
        
        # Check if tensions has a structured format
        if isinstance(tensions, dict):
            for key in ['goal_pressure', 'identity_stress', 'contradiction_pressure']:
                if key in tensions and isinstance(tensions[key], (int, float)):
                    return tensions[key]
        
        return 0.5  # Default if not found

if __name__ == "__main__":
    print("="*60)
    print("QUICK WIN #2: Fixing Infinite Tension Resolution Loops")
    print("="*60)
    
    resolver = ThreadResolver('data/persona')
    resolved, deferred = resolver.resolve_stuck_threads()
    
    print(f"\n📊 Results:")
    print(f"   ✅ Resolved: {resolved} threads")
    print(f"   ⏸️  Deferred: {deferred} threads")
    print(f"   Total processed: {resolved + deferred}")
    
    print("\n" + "="*60)
    print("✅ COMPLETE - Thread resolution loops fixed")
    print("="*60)
