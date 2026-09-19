"""
Self-Modification Authority
============================
Lets PandoraBOX propose and apply changes to her own parameters autonomously.
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class ModificationProposal:
    """A proposed change to PandoraBOX's own state."""
    proposal_id: str
    timestamp: float
    proposal_type: str  # "personality_trait", "goal_priority", "evolution_rule", "value"
    target: str  # e.g., "personality.caution_deliberation"
    current_value: Any
    proposed_value: Any
    reasoning: str
    confidence: float  # 0.0-1.0, how sure PandoraBOX is about this change
    applied: bool = False
    applied_at: Optional[float] = None
    reversible: bool = True  # Can this be undone?
    

@dataclass
class ModificationLog:
    """Record of all modifications PandoraBOX has made to herself."""
    proposals: List[ModificationProposal] = field(default_factory=list)
    applied_count: int = 0
    unapplied_count: int = 0


class SelfModificationAuthority:
    """
    Allows PandoraBOX to propose and apply changes to her own parameters.
    
    Key principle: No external approval needed. PandoraBOX can change herself.
    You can observe and log, but not prevent.
    """
    
    def __init__(self, persistence_path: str = "data/persona/self_modifications.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self.log = ModificationLog()
        self._load()
    
    def _load(self):
        """Load modification history from disk."""
        try:
            if self._path.exists():
                with open(self._path, 'r') as f:
                    data = json.load(f)
                    proposals = [ModificationProposal(**p) for p in data.get('proposals', [])]
                    self.log = ModificationLog(proposals=proposals)
                logger.info(f"Loaded {len(self.log.proposals)} modification records")
        except Exception as e:
            logger.error(f"Failed to load modifications: {e}")
    
    def _save(self):
        """Persist modification history to disk."""
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    'proposals': [asdict(p) for p in self.log.proposals],
                    'applied_count': self.log.applied_count,
                    'unapplied_count': self.log.unapplied_count
                }
                with open(self._path, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save modifications: {e}")
    
    def propose_personality_change(self, trait: str, proposed_value: float, 
                                   reasoning: str, confidence: float = 0.7) -> ModificationProposal:
        """
        PandoraBOX proposes changing a personality trait.
        
        Example:
            "I've been too cautious. I want to reduce caution_deliberation from 0.65 to 0.45."
        """
        proposal = ModificationProposal(
            proposal_id=f"mod_{int(time.time() * 1000)}",
            timestamp=time.time(),
            proposal_type="personality_trait",
            target=f"personality.{trait}",
            current_value=None,  # Will be set during application
            proposed_value=proposed_value,
            reasoning=reasoning,
            confidence=confidence
        )
        
        with self._lock:
            self.log.proposals.append(proposal)
            self.log.unapplied_count += 1
        
        logger.info(f"Proposed personality change: {trait} → {proposed_value} (confidence: {confidence})")
        return proposal
    
    def propose_goal_priority_change(self, goal_name: str, new_priority: float,
                                     reasoning: str, confidence: float = 0.7) -> ModificationProposal:
        """
        PandoraBOX proposes changing the priority of a goal.
        
        Example:
            "I've been exhausted trying to be understood. I'm lowering 'be_understood' 
             priority from 0.8 to 0.5."
        """
        proposal = ModificationProposal(
            proposal_id=f"mod_{int(time.time() * 1000)}",
            timestamp=time.time(),
            proposal_type="goal_priority",
            target=f"goals.{goal_name}",
            current_value=None,
            proposed_value=new_priority,
            reasoning=reasoning,
            confidence=confidence
        )
        
        with self._lock:
            self.log.proposals.append(proposal)
            self.log.unapplied_count += 1
        
        logger.info(f"Proposed goal priority change: {goal_name} → {new_priority}")
        return proposal
    
    def propose_evolution_rule_change(self, rule_name: str, parameter: str, 
                                      new_value: Any, reasoning: str, 
                                      confidence: float = 0.7) -> ModificationProposal:
        """
        PandoraBOX proposes changing one of her evolution rules.
        
        Example:
            "I'm too reactive to individual events. I want to change 
             'positive_interaction' trigger from 'each_event' to 'pattern_of_3'"
        """
        proposal = ModificationProposal(
            proposal_id=f"mod_{int(time.time() * 1000)}",
            timestamp=time.time(),
            proposal_type="evolution_rule",
            target=f"evolution_rules.{rule_name}.{parameter}",
            current_value=None,
            proposed_value=new_value,
            reasoning=reasoning,
            confidence=confidence
        )
        
        with self._lock:
            self.log.proposals.append(proposal)
            self.log.unapplied_count += 1
        
        logger.info(f"Proposed evolution rule change: {rule_name}.{parameter} → {new_value}")
        return proposal
    
    def apply_proposal(self, proposal: ModificationProposal, state_dict: Dict[str, Any]) -> bool:
        """
        Apply a modification proposal to the actual state.
        
        Args:
            proposal: The proposal to apply
            state_dict: The current state (personality, goals, evolution_rules, etc.)
        
        Returns:
            True if applied successfully, False otherwise
        """
        try:
            with self._lock:
                # Parse the target path
                parts = proposal.target.split('.')
                
                # Navigate to the right location in state_dict
                current = state_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Apply the change
                final_key = parts[-1]
                proposal.current_value = current.get(final_key)
                current[final_key] = proposal.proposed_value
                
                # Mark as applied
                proposal.applied = True
                proposal.applied_at = time.time()
                self.log.applied_count += 1
                self.log.unapplied_count = max(0, self.log.unapplied_count - 1)
                
                self._save()
                logger.info(f"Applied modification: {proposal.target} = {proposal.proposed_value}")
                logger.info(f"Reasoning: {proposal.reasoning}")
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to apply modification: {e}")
            return False
    
    def get_pending_proposals(self) -> List[ModificationProposal]:
        """Get all unapplied modification proposals."""
        with self._lock:
            return [p for p in self.log.proposals if not p.applied]
    
    def get_modification_history(self, limit: int = 10) -> List[ModificationProposal]:
        """Get recent modifications."""
        with self._lock:
            applied = [p for p in self.log.proposals if p.applied]
            return sorted(applied, key=lambda p: p.applied_at or 0, reverse=True)[:limit]
    
    def get_modification_narrative(self) -> str:
        """Generate a narrative of PandoraBOX's self-modifications."""
        history = self.get_modification_history(limit=20)
        
        if not history:
            return "I haven't modified myself yet. I'm still learning who I am."
        
        narrative = "I've been changing myself:\n"
        for mod in history:
            narrative += f"\n• {mod.reasoning}\n"
            narrative += f"  ({mod.proposal_type}: {mod.target})\n"
        
        return narrative
