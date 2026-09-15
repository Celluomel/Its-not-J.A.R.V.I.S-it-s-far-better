"""
Quick Win #3: Decision Pressure Calculator
THE KEY TRANSFORMATION: Converge all cognitive factors into unified decision pressure.
This is what transforms simulation into execution.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class DecisionPressureCalculator:
    """
    Converges all cognitive factors into unified decision pressure.
    
    This is the CORE of turning simulation into execution:
    - Goals, tensions, emotions, curiosity, energy, relationships
    - All become forces that drive decisions
    - Single numerical output per factor
    - Combined into total pressure that drives action selection
    """
    
    def __init__(self, persona_dir):
        self.persona_dir = Path(persona_dir)
        
        # Weight factors (sum to 1.0)
        self.weights = {
            'goal_pressure': 0.30,      # Primary driver
            'curiosity_drive': 0.20,    # Exploration
            'identity_stress': 0.20,    # Self-coherence
            'emotional_valence': 0.10,  # Emotional state
            'energy_deficit': 0.10,     # Resource constraint
            'relational_pressure': 0.10 # Social context
        }
    
    def compute(self) -> Dict[str, float]:
        """
        Compute decision pressure from all sources.
        Returns a dict of pressure values and total.
        """
        print("\n🧠 Computing Decision Pressure...")
        
        # Load all relevant files — use DAL for canonical files
        try:
            from core.data.access import DataAccess
            _dal = DataAccess(str(self.persona_dir))
            _goals_list = _dal.get_goals(status='active')
            goals = {'goals': _goals_list}
            tensions = _dal.get_tensions()  # flat {name: float} dict
        except Exception:
            tensions = self._load_json('tensions.json')
            goals = self._load_json('goals.json')
        emotional = self._load_json('emotional_state.json')
        curiosity = self._load_json('curiosity.json')
        energy = self._load_json('energy.json')
        relational = self._load_json('relational_memory.json')
        
        # Compute individual pressures
        pressure = {
            'goal_pressure': self._compute_goal_pressure(goals),
            'curiosity_drive': self._compute_curiosity(curiosity),
            'identity_stress': self._compute_identity_stress(tensions),
            'emotional_valence': self._compute_emotional_valence(emotional),
            'energy_level': self._compute_energy(energy),
            'relational_pressure': self._compute_relational_pressure(relational),
            'timestamp': datetime.now().isoformat()
        }
        
        # Compute energy deficit (low energy = high pressure)
        pressure['energy_deficit'] = 1.0 - pressure['energy_level']
        
        # Compute total weighted pressure
        pressure['total'] = sum(
            pressure[key] * self.weights[key]
            for key in self.weights.keys()
        )
        
        # Add interpretation
        pressure['interpretation'] = self._interpret_pressure(pressure)
        
        return pressure
    
    # Reference pool for normalising goal pressure.
    # 8 active goals at avg priority 0.7 → pressure ≈ 0.50 (healthy midpoint).
    _GOAL_REFERENCE_POOL = 8
    # Hard ceiling — goal pressure never hits 1.0 so the governor can breathe.
    _GOAL_PRESSURE_CEILING = 0.80

    def _compute_goal_pressure(self, goals_data: Dict) -> float:
        """
        Compute pressure from active goals.

        Uses weighted quality score (priority × energy) per goal instead of
        headcount so that 14 stale GQF template goals at priority=1.0/energy≈0
        don't permanently peg pressure at 1.00 and lock the governor in
        structured mode forever.

        Healthy range: 0.20–0.60. Above 0.65 → governor goes structured.
        Hard ceiling: 0.80 (leaves room for pressure to vary).
        """
        if not goals_data:
            return 0.0

        active_goals = goals_data.get('goals', [])
        if not active_goals:
            return 0.0

        # Filter to genuinely active goals
        active = [
            g for g in active_goals
            if isinstance(g, dict)
            and g.get('status', 'active') in ('active', 'dormant')
        ]
        if not active:
            return 0.0

        # Quality score per goal: priority × energy (both 0-1).
        # GQF template goals have energy=0.5 (default) and priority=1.0
        # → quality=0.50, not 1.0. Goals with recent actions get higher energy.
        # Goals that have never been acted on get an "inertia penalty".
        def _goal_quality(g: dict) -> float:
            pri    = float(g.get('priority', 0.5))
            energy = float(g.get('energy',   0.5))
            acted  = int(g.get('actions_taken', 0))
            # Inertia: goals never acted on get 20% quality reduction
            inertia = 0.8 if acted == 0 else 1.0
            return pri * energy * inertia

        total_quality = sum(_goal_quality(g) for g in active)
        # Normalise against reference pool of 8 goals at avg quality 0.5
        ref_quality = self._GOAL_REFERENCE_POOL * 0.5
        pressure = total_quality / ref_quality
        pressure = min(self._GOAL_PRESSURE_CEILING, pressure)

        high_priority_count = sum(
            1 for g in active if g.get('priority', 0) > 0.7
        )
        print(
            f"   📊 Goal Pressure: {pressure:.2f} "
            f"({len(active)} goals, {high_priority_count} high-priority, "
            f"total_quality={total_quality:.2f})"
        )
        return pressure
    
    def _compute_curiosity(self, curiosity_data: Dict) -> float:
        """
        Compute curiosity drive from global level + active high-curiosity topics.
        """
        if not curiosity_data:
            return 0.5

        # Base: global_curiosity (already 0-1)
        base = float(curiosity_data.get('global_curiosity',
                     curiosity_data.get('current_level',
                     curiosity_data.get('level',
                     curiosity_data.get('drive', 0.5)))))
        if base > 1.0:
            base = base / 100.0
        base = max(0.0, min(1.0, base))

        # Boost from actively-curious topics (high curiosity × encountered multiple times)
        topics = curiosity_data.get('topics', {})
        if topics:
            hot_topics = [
                v.get('curiosity', 0)
                for v in topics.values()
                if isinstance(v, dict)
                   and v.get('curiosity', 0) > 0.7
                   and v.get('times_encountered', 0) > 3
            ]
            if hot_topics:
                topic_boost = min(0.2, len(hot_topics) * 0.05)
                base = min(1.0, base + topic_boost)

        print(f"   🔍 Curiosity Drive: {base:.2f}")
        return base
    
    def _compute_identity_stress(self, tensions_data: Dict) -> float:
        """
        Extract identity stress from tensions.
        """
        if not tensions_data:
            return 0.0
        
        stress = tensions_data.get('identity_stress',
                 tensions_data.get('stress',
                 tensions_data.get('identity', 0.0)))
        
        if isinstance(stress, (int, float)):
            stress = float(stress)
        else:
            stress = 0.0
        
        print(f"   🎭 Identity Stress: {stress:.2f}")
        return stress
    
    def _compute_emotional_valence(self, emotional_data: Dict) -> float:
        """
        Compute emotional valence (-1 to 1, but we use absolute for pressure).
        Positive emotions = low pressure, negative = high pressure.
        """
        if not emotional_data:
            return 0.0
        
        emotions = emotional_data.get('emotions', {})
        
        if not emotions:
            return 0.0
        
        # Support both flat {name: value} and nested {name: {value: float}} formats
        def _eval(v):
            if isinstance(v, dict): return v.get('value', 0.0)
            return float(v) if isinstance(v, (int, float)) else 0.0
        flat = {k: _eval(v) for k, v in emotions.items()}

        # Identify positive and negative emotions (expanded to actual names used)
        positive_emotions = ['joy', 'contentment', 'satisfaction', 'peace', 'love',
                             'enthusiasm', 'warmth', 'curiosity', 'delight', 'calm']
        negative_emotions = ['frustration', 'anxiety', 'fear', 'sadness', 'anger',
                             'distress', 'guilt', 'shame', 'boredom']

        positive = sum(
            flat.get(e, 0) for e in positive_emotions
            if flat.get(e, 0) > 0
        )
        
        negative = sum(
            flat.get(e, 0) for e in negative_emotions
            if flat.get(e, 0) > 0
        )
        
        total = positive + negative

        if total == 0:
            valence = 0.0
            arousal = 0.0
        else:
            # Valence: -1 (all negative) to +1 (all positive)
            valence = (positive - negative) / total
            # Arousal: how emotionally activated overall (0 = flat, 1 = intense)
            # High arousal in any direction creates cognitive pressure
            n_emotions = len(flat)
            arousal = min(1.0, total / max(1, n_emotions))

        # Pressure model: arousal-based
        # - High negative valence → high pressure (distress)
        # - Very high positive valence → moderate pressure (enthusiasm drives action)
        # - Emotional flatness → low pressure
        neg_pressure  = max(0.0, -valence)            # 0–1: distress pressure
        pos_pressure  = max(0.0, valence - 0.3) * 0.4 # 0–0.28: enthusiasm pressure
        pressure = min(1.0, neg_pressure + pos_pressure + arousal * 0.1)

        print(f"   ❤️  Emotional Valence: {valence:.2f} arousal={arousal:.2f} (pressure: {pressure:.2f})")
        return pressure
    
    def _compute_energy(self, energy_data: Dict) -> float:
        """
        Extract energy level.
        """
        if not energy_data:
            return 0.5
        
        level = energy_data.get('current',
                energy_data.get('level',
                energy_data.get('energy', 0.5)))
        
        if isinstance(level, (int, float)):
            level = float(level)
            # energy.json stores 0-100; normalize to 0-1
            if level > 1.0:
                level = level / 100.0
            level = max(0.0, min(1.0, level))
        else:
            level = 0.5

        print(f"   ⚡ Energy Level: {level:.2f}")
        return level
    
    def _compute_relational_pressure(self, relational_data: Dict) -> float:
        """
        Compute pressure from relationships.
        Low trust = high pressure to build trust.
        """
        if not relational_data:
            return 0.0
        
        # Support both list format and dict {person: {trust_score,...}} format
        raw = relational_data.get('relationships', relational_data)
        relationships = list(raw.values()) if isinstance(raw, dict) else raw

        if not relationships:
            return 0.0

        # Calculate average trust
        trust_scores = [
            r.get('trust_score', r.get('trust', 0.5))
            for r in relationships
            if isinstance(r, dict)
        ]
        
        if not trust_scores:
            return 0.0
        
        avg_trust = sum(trust_scores) / len(trust_scores)
        
        # Low trust = high pressure
        pressure = 1.0 - avg_trust
        
        print(f"   🤝 Relational Pressure: {pressure:.2f} (avg trust: {avg_trust:.2f})")
        return pressure
    
    def _interpret_pressure(self, pressure: Dict) -> str:
        """
        Interpret total pressure level.
        """
        total = pressure['total']
        
        if total > 0.8:
            return "CRITICAL - Immediate action required"
        elif total > 0.6:
            return "HIGH - Strong drive to act"
        elif total > 0.4:
            return "MODERATE - Balanced state"
        elif total > 0.2:
            return "LOW - Exploration mode"
        else:
            return "MINIMAL - Resting state"
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file safely."""
        filepath = self.persona_dir / filename
        
        if not filepath.exists():
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load {filename}: {e}")
            return {}
    
    def display_pressure(self, pressure: Dict):
        """Display pressure analysis in a readable format."""
        print("\n" + "="*60)
        print("🎯 DECISION PRESSURE ANALYSIS")
        print("="*60)
        print(f"\n{'Factor':<25} {'Value':>8} {'Weight':>8} {'Contribution':>12}")
        print("-"*60)
        
        for key in self.weights.keys():
            value = pressure.get(key.replace('_deficit', '_level'), 0.0)
            if key == 'energy_deficit':
                value = pressure.get('energy_deficit', 0.0)
            weight = self.weights[key]
            contribution = value * weight
            
            print(f"{key:<25} {value:>7.2f}  {weight:>7.2f}  {contribution:>11.3f}")
        
        print("-"*60)
        print(f"{'TOTAL PRESSURE':<25} {pressure['total']:>7.2f}")
        print(f"{'Interpretation':<25} {pressure['interpretation']}")
        print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("QUICK WIN #3: Decision Pressure Calculator")
    print("="*60)
    
    calculator = DecisionPressureCalculator('data/persona')
    pressure = calculator.compute()
    calculator.display_pressure(pressure)
    
    print("\n✅ COMPLETE - Decision pressure computed successfully")
