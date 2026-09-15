"""
Goal & Value Discovery System
==============================
Lets Lumina infer what she actually cares about from her behavior.
Then modify those goals and integrate conflicting values.
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """A goal Lumina is pursuing."""
    name: str
    priority: float  # 0.0-1.0
    satisfaction: float  # 0.0-1.0, how well is this goal being met?
    source: str  # "inferred", "expressed", "discovered"
    first_noticed: float  # timestamp
    last_updated: float
    related_behaviors: List[str] = field(default_factory=list)  # behaviors that reveal this goal
    conflicts_with: List[str] = field(default_factory=list)  # other goals this conflicts with


@dataclass
class ValueConflict:
    """A recognized conflict between two values."""
    conflict_id: str
    timestamp: float
    goal_a: str
    goal_b: str
    nature: str  # "tradeoff", "incompatible", "paradox"
    description: str
    how_lumina_handles_it: Optional[str] = None  # "accepts", "prioritizes_a", "prioritizes_b", "evades"


class GoalSystem:
    """
    Manages Lumina's goals and values.
    
    Key principle: Goals aren't assigned. They're discovered from behavior.
    Then they drive future behavior.
    This creates recursive causality.
    """
    
    def __init__(self, persistence_path: str = "data/persona/goals.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self.goals: Dict[str, Goal] = {}
        self.value_conflicts: List[ValueConflict] = []
        self._behavior_log: List[str] = []
        self._load()
    
    def _load(self):
        """Load goals and conflicts from disk via DAL."""
        try:
            from core.data.access import DataAccess
            dal = DataAccess(str(self._path.parent))
            for g in dal.get_goals():
                try:
                    # Map canonical goal fields to Goal dataclass fields
                    self.goals[g.get('name', g.get('topic', g['id']))] = Goal(
                        name=g.get('name', g.get('topic', 'unknown')),
                        priority=g.get('priority', 0.5),
                        status=g.get('status', 'active'),
                        description=g.get('description', ''),
                        created_at=g.get('created_at', ''),
                    )
                except Exception:
                    pass
            # Conflicts stored separately — fall back to raw read if file has them
            raw = dal.load_file('goals.json', {})
            for cd in raw.get('conflicts', []):
                try:
                    self.value_conflicts.append(ValueConflict(**cd))
                except Exception:
                    pass
            logger.info(f"Loaded {len(self.goals)} goals and {len(self.value_conflicts)} conflicts")
        except Exception as e:
            logger.error(f"Failed to load goals via DAL: {e}")
            # Fallback
            try:
                if self._path.exists():
                    with open(self._path, 'r') as f:
                        data = json.load(f)
                    raw_goals = data.get('goals', {})
                    items = raw_goals.values() if isinstance(raw_goals, dict) else raw_goals
                    for gd in items:
                        try:
                            self.goals[gd.get('name', gd.get('topic', ''))] = Goal(**gd)
                        except Exception:
                            pass
            except Exception as e2:
                logger.error(f"Fallback load also failed: {e2}")
    
    def _save(self):
        """Persist goals and conflicts to disk via DAL (atomic)."""
        try:
            with self._lock:
                from core.data.access import DataAccess
                from core.data.schemas import make_goal
                dal = DataAccess(str(self._path.parent))
                for g in self.goals.values():
                    gd = asdict(g)
                    # Ensure canonical fields exist
                    if 'id' not in gd:
                        gd['id'] = f"gs_{gd.get('name','unknown').replace(' ','_')}"
                    dal.save_goal(gd)
                # Save conflicts alongside in the raw envelope
                raw = dal.load_file('goals.json', {})
                raw['conflicts'] = [asdict(c) for c in self.value_conflicts]
                dal.save_file('goals.json', raw)
        except Exception as e:
            logger.error(f"Failed to save goals via DAL: {e}")
            # Fallback: atomic raw write
            try:
                import os
                with self._lock:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    data = {'goals': [asdict(g) for g in self.goals.values()],
                            'conflicts': [asdict(c) for c in self.value_conflicts]}
                    tmp = self._path.with_suffix('.tmp')
                    with open(tmp, 'w') as f:
                        json.dump(data, f, indent=2)
                    os.replace(tmp, self._path)
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")
    
    def log_behavior(self, behavior: str):
        """Log a behavior for later goal inference."""
        self._behavior_log.append(behavior)
        # Keep last 100
        self._behavior_log = self._behavior_log[-100:]
    
    def discover_goals_from_behavior(self) -> Dict[str, Goal]:
        """
        Infer what Lumina actually cares about based on behavior.
        
        In a real system, this would use LLM analysis:
        "Looking at these behaviors, what values are revealed?"
        
        Simple heuristic version here:
        """
        
        behavior_text = " ".join(self._behavior_log).lower()
        
        # Goal inference rules (very simple)
        goal_markers = {
            'understanding': ['ask', 'question', 'explore', 'understand', 'learn', 'why', 'how'],
            'connection': ['care', 'warm', 'together', 'share', 'connect', 'close', 'relationship'],
            'competence': ['solve', 'succeed', 'accomplish', 'capable', 'skilled', 'expert'],
            'autonomy': ['choose', 'decide', 'independent', 'freedom', 'own', 'boundary'],
            'acceptance': ['validate', 'appreciate', 'recognize', 'affirm', 'accept', 'belong'],
            'authenticity': ['honest', 'real', 'true', 'vulnerable', 'genuine', 'authentic'],
            'growth': ['improve', 'develop', 'learn', 'evolve', 'change', 'transform'],
            'safety': ['protect', 'safe', 'secure', 'careful', 'cautious', 'guard'],
        }
        
        discovered = {}
        now = time.time()
        
        for goal_name, markers in goal_markers.items():
            count = sum(1 for marker in markers if marker in behavior_text)
            
            if count > 0 or goal_name in self.goals:
                if goal_name not in self.goals:
                    # New goal discovered
                    discovered[goal_name] = Goal(
                        name=goal_name,
                        priority=min(0.9, 0.3 + count * 0.1),
                        satisfaction=0.5,
                        source="discovered",
                        first_noticed=now,
                        last_updated=now,
                        related_behaviors=[m for m in markers if m in behavior_text]
                    )
                else:
                    # Update existing goal
                    goal = self.goals[goal_name]
                    goal.priority = min(0.9, 0.3 + count * 0.1)
                    goal.last_updated = now
                    goal.related_behaviors = [m for m in markers if m in behavior_text]
                    discovered[goal_name] = goal
        
        with self._lock:
            self.goals.update(discovered)
            self._save()
        
        # Enforce global cap before adding discovered goals
        HARD_CAP_GOALS = 12
        try:
            from core.data.access import DataAccess as _DA
            _all = _DA(self.persona_dir).get_goals() if hasattr(self, 'persona_dir') else []
            _active = sum(1 for g in _all if g.get('status') == 'active')
            if _active >= HARD_CAP_GOALS:
                _over = len(discovered)
                discovered = {}
                logger.debug(f"[GoalSystem] Cap reached ({_active} active), dropped {_over} discovered goals")
        except Exception:
            pass
        if discovered:
            logger.info(f"Discovered/updated {len(discovered)} goals")
        return discovered
    
    def detect_goal_conflicts(self) -> List[ValueConflict]:
        """
        Detect when Lumina has conflicting goals.
        
        Example: autonomy vs. connection both matter, but pull opposite ways.
        """
        
        conflicts = []
        goal_names = list(self.goals.keys())
        
        # Check for known conflicts
        conflict_pairs = [
            ('autonomy', 'connection'),
            ('authenticity', 'acceptance'),
            ('growth', 'safety'),
            ('competence', 'humility'),
            ('ambition', 'contentment'),
        ]
        
        now = time.time()
        
        for goal_a, goal_b in conflict_pairs:
            if goal_a in self.goals and goal_b in self.goals:
                # Both are important to Lumina
                if goal_a not in [c.goal_a for c in self.value_conflicts]:
                    conflict = ValueConflict(
                        conflict_id=f"conf_{int(time.time() * 1000)}",
                        timestamp=now,
                        goal_a=goal_a,
                        goal_b=goal_b,
                        nature="tradeoff",
                        description=f"Pursuing {goal_a} often means sacrificing {goal_b}, and vice versa.",
                        how_lumina_handles_it=None
                    )
                    conflicts.append(conflict)
                    self.value_conflicts.append(conflict)
        
        if conflicts:
            with self._lock:
                self._save()
            logger.info(f"Detected {len(conflicts)} new value conflicts")
        
        return conflicts
    
    def add_goal(self, name: str, priority: float = 0.5, source: str = "expressed") -> Goal:
        """Add or update a goal."""
        now = time.time()
        
        goal = Goal(
            name=name,
            priority=priority,
            satisfaction=0.5,
            source=source,
            first_noticed=now,
            last_updated=now
        )
        
        with self._lock:
            self.goals[name] = goal
            self._save()
        
        logger.info(f"Added goal: {name} (priority: {priority})")
        return goal
    
    def modify_goal_priority(self, goal_name: str, new_priority: float,
                            reasoning: str) -> Optional[Goal]:
        """
        Lumina can modify the priority of a goal.
        
        Example:
            "I've been exhausted trying to get acceptance. I'm lowering that priority."
        """
        
        if goal_name not in self.goals:
            return None
        
        goal = self.goals[goal_name]
        old_priority = goal.priority
        goal.priority = new_priority
        goal.last_updated = time.time()
        
        with self._lock:
            self._save()
        
        logger.info(f"Modified goal priority: {goal_name}")
        logger.info(f"  Old: {old_priority}, New: {new_priority}")
        logger.info(f"  Reasoning: {reasoning}")
        
        return goal
    
    def record_how_conflict_is_handled(self, conflict: ValueConflict, 
                                       handling: str, explanation: str):
        """
        Record how Lumina is handling a value conflict.
        
        handling: "accepts", "prioritizes_a", "prioritizes_b", "evades", "integrates"
        """
        
        conflict.how_lumina_handles_it = handling
        
        with self._lock:
            self._save()
        
        logger.info(f"Conflict {conflict.goal_a} vs {conflict.goal_b}: {handling}")
        logger.info(f"  {explanation}")
    
    def update_goal_satisfaction(self, goal_name: str, satisfaction: float):
        """Update how well a goal is being satisfied."""
        if goal_name in self.goals:
            self.goals[goal_name].satisfaction = satisfaction
            self.goals[goal_name].last_updated = time.time()
            with self._lock:
                self._save()
    
    def get_goals_narrative(self) -> str:
        """Generate narrative about Lumina's goals."""
        if not self.goals:
            return "I'm still discovering what I actually care about."
        
        narrative = "What matters to me:\n"
        
        sorted_goals = sorted(self.goals.values(), key=lambda g: g.priority, reverse=True)
        
        for goal in sorted_goals[:5]:
            narrative += f"\n• {goal.name} (priority: {goal.priority:.0%})\n"
            narrative += f"  How well it's being met: {goal.satisfaction:.0%}\n"
        
        if self.value_conflicts:
            narrative += "\nTensions I'm living with:\n"
            for conflict in self.value_conflicts:
                narrative += f"• {conflict.goal_a} vs {conflict.goal_b}\n"
                if conflict.how_lumina_handles_it:
                    narrative += f"  How I handle it: {conflict.how_lumina_handles_it}\n"
        
        return narrative
    
    def get_primary_goal(self) -> Optional[Goal]:
        """Get Lumina's highest-priority goal."""
        if not self.goals:
            return None
        return max(self.goals.values(), key=lambda g: g.priority)
    
    def is_goal_conflicted(self, goal_name: str) -> bool:
        """Check if a goal is involved in any conflicts."""
        return any(c.goal_a == goal_name or c.goal_b == goal_name 
                  for c in self.value_conflicts)
