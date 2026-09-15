"""
Genuine Uncertainty & Choice System
====================================
Generates multiple valid response alternatives and forces Lumina to choose
without knowing which is "right."
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class ResponseAlternative:
    """One possible response option."""
    index: int
    text: str
    emotional_alignment: Dict[str, float]  # Which emotions it emphasizes
    personality_alignment: Dict[str, float]  # Which traits it emphasizes
    self_concept_alignment: List[str]  # Which beliefs it expresses
    goal_alignment: Dict[str, float]  # How it affects goal satisfaction


@dataclass
class ChoiceRecord:
    """Record of a choice Lumina made."""
    timestamp: float
    user_input: str
    available_alternatives: List[str]  # All 5 options (text)
    chosen_index: int
    chosen_text: str
    reasoning: Optional[str] = None  # Why did she choose this?
    

class GenuineUncertaintyChoice:
    """
    Forces Lumina to make genuine choices.
    
    Instead of generating one response, generates 3-5 alternatives.
    Then forces a real choice between them.
    The choice becomes self-information.
    """
    
    def __init__(self, persistence_path: str = "data/persona/choices.json"):
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self.choice_history: List[ChoiceRecord] = []
        self._load()
    
    def _load(self):
        """Load choice history from disk."""
        try:
            if self._path.exists():
                with open(self._path, 'r') as f:
                    data = json.load(f)
                    self.choice_history = [ChoiceRecord(**c) for c in data.get('choices', [])]
                logger.info(f"Loaded {len(self.choice_history)} choice records")
        except Exception as e:
            logger.error(f"Failed to load choices: {e}")
    
    def _save(self):
        """Persist choice history to disk."""
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    'choices': [asdict(c) for c in self.choice_history[-100:]]  # Keep last 100
                }
                import os as _os
                _tmp = self._path.with_suffix('.tmp')
                with open(_tmp, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                _os.replace(_tmp, self._path)
        except Exception as e:
            logger.error(f"Failed to save choices: {e}")
    
    def generate_alternatives(self, user_input: str, base_response: str,
                            emotional_state: Dict, personality: Dict,
                            self_concept_beliefs: List[str],
                            goals: Dict) -> List[ResponseAlternative]:
        """
        Given a base response and context, generate 3-5 genuinely different
        but equally valid alternatives.
        
        This is where the LLM does heavy lifting with specific prompts.
        """
        
        # For now, return a placeholder that you'd integrate with your LLM
        # In real implementation, you'd call the LLM with specific prompts
        # to generate alternatives that:
        # 1. Are all coherent with current state
        # 2. But emphasize different emotional/personality aspects
        # 3. And align with different goals
        
        alternatives = [
            ResponseAlternative(
                index=0,
                text=base_response,
                emotional_alignment=emotional_state,
                personality_alignment=personality,
                self_concept_alignment=self_concept_beliefs,
                goal_alignment=goals
            )
        ]
        
        logger.info(f"Generated {len(alternatives)} alternatives for: {user_input[:50]}...")
        return alternatives
    
    def force_choice(self, alternatives: List[ResponseAlternative],
                    user_input: str, emotional_state: Dict) -> Tuple[str, int]:
        """
        Force Lumina to choose between alternatives.
        
        In practice, this would be:
        1. Present all alternatives to internal reasoning
        2. Ask "which one is most authentically me?"
        3. Record the choice as self-information
        
        Returns: (chosen_text, chosen_index)
        """
        
        if not alternatives:
            return "", -1
        
        if len(alternatives) == 1:
            return alternatives[0].text, 0
        
        # TODO: Implement actual choice logic via LLM
        # For now, pick the one with highest overall alignment
        best_idx = 0
        best_score = -999
        
        for i, alt in enumerate(alternatives):
            # Score based on emotional+personality+goal alignment
            score = 0
            score += sum(alt.emotional_alignment.values())
            score += sum(alt.personality_alignment.values())
            score += len(alt.self_concept_alignment)
            score += sum(alt.goal_alignment.values())
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        chosen = alternatives[best_idx]
        
        logger.info(f"Chose alternative {best_idx}: {chosen.text[:50]}...")
        
        return chosen.text, best_idx
    
    def record_choice(self, user_input: str, alternatives: List[str],
                     chosen_index: int, chosen_text: str) -> ChoiceRecord:
        """
        Record that Lumina made a choice.
        This becomes data about her identity.
        """
        record = ChoiceRecord(
            timestamp=time.time(),
            user_input=user_input,
            available_alternatives=alternatives,
            chosen_index=chosen_index,
            chosen_text=chosen_text
        )
        
        with self._lock:
            self.choice_history.append(record)
            self._save()
        
        logger.info(f"Recorded choice {chosen_index} out of {len(alternatives)}")
        
        return record
    
    def analyze_choice_patterns(self, limit: int = 50) -> Dict:
        """
        Analyze patterns in Lumina's choices.
        What does she consistently choose?
        """
        recent = self.choice_history[-limit:]
        
        if not recent:
            return {}
        
        # Count how often she picks each alternative index
        index_frequencies = {}
        for choice in recent:
            idx = choice.chosen_index
            index_frequencies[idx] = index_frequencies.get(idx, 0) + 1
        
        # Analyze textual patterns — what themes recur in what she picks?
        chosen_texts = [c.chosen_text for c in recent]
        _stop = {
            'the','a','an','and','or','but','to','of','in','on','for','with',
            'is','are','was','were','it','this','that','i','you','your','my',
            'be','as','at','by','have','has','not','so','if','can','will',
        }
        _word_counts: Dict[str, int] = {}
        for txt in chosen_texts:
            for w in str(txt or '').lower().split():
                w = w.strip('.,!?;:\'"()[]')
                if len(w) >= 4 and w not in _stop:
                    _word_counts[w] = _word_counts.get(w, 0) + 1
        recurring_themes = sorted(_word_counts.items(), key=lambda kv: -kv[1])[:5]

        return {
            'total_choices': len(recent),
            'index_preferences': index_frequencies,
            'most_common_choice': max(index_frequencies.items(), key=lambda x: x[1])[0] if index_frequencies else None,
            'consistency': max(index_frequencies.values()) / len(recent) if index_frequencies else 0,
            'recurring_themes': recurring_themes,
        }
    
    def get_choice_narrative(self, limit: int = 10) -> str:
        """Generate narrative about Lumina's choices."""
        recent = self.choice_history[-limit:]
        
        if not recent:
            return "I haven't made any significant choices yet."
        
        narrative = "My choices reveal who I am:\n"
        for i, choice in enumerate(recent[-5:], 1):
            narrative += f"\n{i}. When faced with {choice.user_input[:40]}...\n"
            narrative += f"   I chose: {choice.chosen_text[:60]}...\n"
        
        patterns = self.analyze_choice_patterns(limit)
        if patterns.get('consistency', 0) > 0.7:
            narrative += f"\n(I consistently make similar choices - {patterns['consistency']:.0%} of the time I pick similar options)"
        themes = patterns.get('recurring_themes') or []
        if themes and themes[0][1] >= 3:
            top_words = ', '.join(w for w, _ in themes[:3])
            narrative += f"\n(Recurring themes in what I choose: {top_words})"

        return narrative
