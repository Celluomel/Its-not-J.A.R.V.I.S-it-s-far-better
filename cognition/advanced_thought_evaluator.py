"""
Phase 2.1: Advanced Thought Evaluator

Goes beyond keyword matching to use semantic similarity,
conceptual friction detection, and multi-factor priority scoring.

This replaces the simple keyword-based evaluator with true
conceptual understanding.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import math

class AdvancedThoughtEvaluator:
    """
    Advanced thought evaluation using semantic understanding.
    
    Improvements over simple evaluator:
    - Semantic similarity to active goals (not just keywords)
    - Conceptual friction detection (conflicting concepts)
    - Multi-factor priority scoring (urgency × impact × actionability)
    - Goal impact prediction
    - Better type classification
    """
    
    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        
        # Semantic similarity thresholds
        self.high_similarity_threshold = 0.7
        self.moderate_similarity_threshold = 0.4
        
        # Enhanced classification patterns
        self.uncertainty_patterns = {
            'explicit': ['wonder', 'curious', 'unsure', 'unclear', 'question', 'confusing'],
            'implicit': ['might', 'perhaps', 'maybe', 'possibly', 'could be', 'not sure'],
            'epistemic': ['how', 'why', 'what if', 'whether', 'which', 'understand']
        }
        
        self.tension_patterns = {
            'conflict': ['conflict', 'tension', 'contradiction', 'opposing', 'clash'],
            'dissonance': ['inconsistent', 'incompatible', 'mismatch', 'discrepancy'],
            'dilemma': ['choice', 'decide', 'torn', 'struggle', 'difficult']
        }
        
        self.opportunity_patterns = {
            'exploration': ['explore', 'try', 'experiment', 'discover', 'learn'],
            'growth': ['improve', 'develop', 'enhance', 'expand', 'grow'],
            'creation': ['create', 'build', 'make', 'design', 'construct']
        }
        
        # Conceptual friction indicators
        self.friction_indicators = [
            'but', 'however', 'yet', 'although', 'despite', 'whereas',
            'on the other hand', 'nevertheless', 'nonetheless'
        ]
    
    def evaluate_recent_thoughts(self, max_thoughts: int = 10) -> List[Dict]:
        """
        Evaluate recent thoughts using advanced semantic analysis.
        
        Returns list of goal suggestions with enhanced metadata.
        """
        thoughts = self._load_recent_thoughts(max_thoughts)
        
        if not thoughts:
            return []
        
        # Load active goals for semantic comparison
        active_goals = self._load_active_goals()
        
        suggestions = []
        
        for thought in thoughts:
            # Skip already processed thoughts
            if thought.get('evaluated', False):
                continue
            
            evaluation = self._evaluate_thought_advanced(thought, active_goals)
            
            # Only create suggestion if actionable
            if evaluation['actionability'] > 0.5:
                suggestion = self._create_goal_suggestion(thought, evaluation)
                suggestions.append(suggestion)
                
                # Mark thought as evaluated
                self._mark_thought_evaluated(thought.get('id', ''))
        
        return suggestions
    
    def _evaluate_thought_advanced(
        self, 
        thought: Dict, 
        active_goals: List[Dict]
    ) -> Dict:
        """
        Advanced evaluation using multiple factors.
        """
        text = thought.get('content', thought.get('text', ''))
        
        # 1. Classify thought type with confidence
        thought_type, type_confidence = self._classify_thought_advanced(text)
        
        # 2. Extract concepts (not just keywords)
        concepts = self._extract_concepts(text)
        
        # 3. Compute semantic similarity to active goals
        goal_similarity = self._compute_goal_similarity(concepts, active_goals)
        
        # 4. Detect conceptual friction
        friction_score = self._detect_conceptual_friction(text)
        
        # 5. Compute multi-factor priority
        priority = self._compute_priority_multifactor(
            type_confidence=type_confidence,
            goal_similarity=goal_similarity,
            friction_score=friction_score
        )
        
        # 6. Compute urgency (time-sensitive vs. exploratory)
        urgency = self._compute_urgency(thought_type, friction_score)
        
        # 7. Compute actionability
        actionability = self._compute_actionability(
            thought_type=thought_type,
            concepts=concepts,
            goal_similarity=goal_similarity
        )
        
        # 8. Predict goal impact
        goal_impact = self._predict_goal_impact(
            goal_similarity=goal_similarity,
            priority=priority
        )
        
        return {
            'type': thought_type,
            'type_confidence': type_confidence,
            'concepts': concepts,
            'goal_similarity': goal_similarity,
            'friction_score': friction_score,
            'priority': priority,
            'urgency': urgency,
            'actionability': actionability,
            'goal_impact': goal_impact
        }
    
    def _classify_thought_advanced(self, text: str) -> tuple:
        """
        Classify thought type with confidence score.
        
        Returns: (type, confidence)
        """
        text_lower = text.lower()
        
        scores = {
            'uncertainty': 0.0,
            'tension': 0.0,
            'opportunity': 0.0
        }
        
        # Score uncertainty
        for category, patterns in self.uncertainty_patterns.items():
            matches = sum(1 for p in patterns if p in text_lower)
            weight = {'explicit': 1.0, 'implicit': 0.7, 'epistemic': 0.9}[category]
            scores['uncertainty'] += matches * weight
        
        # Score tension
        for category, patterns in self.tension_patterns.items():
            matches = sum(1 for p in patterns if p in text_lower)
            weight = {'conflict': 1.0, 'dissonance': 0.9, 'dilemma': 0.8}[category]
            scores['tension'] += matches * weight
        
        # Score opportunity
        for category, patterns in self.opportunity_patterns.items():
            matches = sum(1 for p in patterns if p in text_lower)
            weight = {'exploration': 1.0, 'growth': 0.9, 'creation': 1.0}[category]
            scores['opportunity'] += matches * weight
        
        # Normalize scores
        max_score = max(scores.values()) if any(scores.values()) else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        # Get dominant type
        thought_type = max(scores, key=scores.get)
        confidence = scores[thought_type]
        
        # If confidence too low, classify as general
        if confidence < 0.3:
            thought_type = 'general'
            confidence = 0.5
        
        return thought_type, confidence
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract meaningful concepts (not just keywords).
        
        Uses simple noun phrase extraction and important words.
        """
        concepts = []
        
        # Remove common stop words
        stop_words = {'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
                     'of', 'to', 'in', 'for', 'on', 'with', 'as', 'by'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter and get meaningful words
        meaningful = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Simple noun phrase extraction (adjacent meaningful words)
        i = 0
        while i < len(meaningful):
            if i < len(meaningful) - 1:
                # Try bigram
                bigram = f"{meaningful[i]} {meaningful[i+1]}"
                concepts.append(bigram)
                i += 2
            else:
                concepts.append(meaningful[i])
                i += 1
        
        # Add important single words
        concepts.extend([w for w in meaningful if len(w) > 5])
        
        # Deduplicate
        return list(set(concepts))
    
    def _compute_goal_similarity(
        self,
        concepts: List[str],
        active_goals: List[Dict]
    ) -> float:
        """
        Compute semantic similarity to active goals.

        Uses the QUALITY-tier embedder (nomic-embed-text) cosine similarity -
        true semantic understanding, robust to rewording. The class thresholds
        (high=0.7 / moderate=0.4) map naturally onto the cosine scale. Falls
        back to the legacy word-overlap heuristic if the embedder/API is
        unavailable, so evaluation never breaks offline.
        """
        if not concepts or not active_goals:
            return 0.0
        concept_text = ' '.join(concepts)
        goal_texts = [goal.get('name', goal.get('description', '')) for goal in active_goals]
        goal_texts = [g for g in goal_texts if g]
        if not goal_texts:
            return 0.0
        try:
            import numpy as np
            from utils.shared_embedder import get_embedder
            emb = get_embedder(quality=True)
            vecs = np.asarray(emb.encode([concept_text] + goal_texts), dtype="float32")
            c = vecs[0]
            others = vecs[1:]
            cn = float(np.linalg.norm(c)) + 1e-9
            on = np.linalg.norm(others, axis=1) + 1e-9
            sims = (others @ c) / (on * cn)
            if not len(sims):
                return 0.0
            return float(min(1.0, max(0.0, float(np.max(sims)))))
        except Exception:
            return self._compute_goal_similarity_word_overlap(concepts, active_goals)

    def _compute_goal_similarity_word_overlap(
        self,
        concepts: List[str],
        active_goals: List[Dict]
    ) -> float:
        """Legacy word-overlap similarity (offline fallback)."""
        max_similarity = 0.0
        for goal in active_goals:
            goal_text = goal.get('name', goal.get('description', '')).lower()
            goal_words = set(re.findall(r'\b[a-z]+\b', goal_text))
            concept_words = set(' '.join(concepts).split())
            overlap = len(concept_words & goal_words)
            if goal_words:
                similarity = overlap / len(goal_words)
                max_similarity = max(max_similarity, similarity)
        return min(1.0, max_similarity)

    def _detect_conceptual_friction(self, text: str) -> float:
        """
        Detect conflicting concepts or tension indicators.
        
        Higher score = more friction = more cognitive tension.
        """
        text_lower = text.lower()
        
        friction_count = sum(1 for indicator in self.friction_indicators 
                           if indicator in text_lower)
        
        # Normalize to 0-1
        friction_score = min(1.0, friction_count * 0.3)
        
        return friction_score
    
    def _compute_priority_multifactor(
        self,
        type_confidence: float,
        goal_similarity: float,
        friction_score: float
    ) -> float:
        """
        Compute priority from multiple factors.
        
        Priority = weighted combination of:
        - Type confidence (40%)
        - Goal similarity (40%)
        - Friction score (20%)
        """
        priority = (
            type_confidence * 0.4 +
            goal_similarity * 0.4 +
            friction_score * 0.2
        )
        
        return min(1.0, priority)
    
    def _compute_urgency(self, thought_type: str, friction_score: float) -> float:
        """
        Compute urgency (how time-sensitive is this thought).
        
        Tension thoughts with high friction are most urgent.
        """
        base_urgency = {
            'tension': 0.8,
            'uncertainty': 0.5,
            'opportunity': 0.4,
            'general': 0.3
        }.get(thought_type, 0.3)
        
        # Friction increases urgency
        urgency = base_urgency + (friction_score * 0.2)
        
        return min(1.0, urgency)
    
    def _compute_actionability(
        self,
        thought_type: str,
        concepts: List[str],
        goal_similarity: float
    ) -> float:
        """
        Compute how actionable this thought is.
        
        More concepts + higher goal similarity = more actionable.
        """
        # Base actionability by type
        base = {
            'opportunity': 0.8,
            'uncertainty': 0.6,
            'tension': 0.7,
            'general': 0.4
        }.get(thought_type, 0.4)
        
        # More concepts = more actionable
        concept_factor = min(1.0, len(concepts) / 5.0)
        
        # Higher goal similarity = more actionable
        actionability = (base * 0.5) + (concept_factor * 0.2) + (goal_similarity * 0.3)
        
        return min(1.0, actionability)
    
    def _predict_goal_impact(self, goal_similarity: float, priority: float) -> float:
        """
        Predict impact on existing goals if this thought becomes a goal.
        """
        # Simple prediction: high similarity × high priority = high impact
        impact = (goal_similarity * 0.6) + (priority * 0.4)
        
        return min(1.0, impact)
    
    def _create_goal_suggestion(self, thought: Dict, evaluation: Dict) -> Dict:
        """
        Create enhanced goal suggestion from thought evaluation.
        """
        concepts = evaluation['concepts']
        primary_concept = concepts[0] if concepts else 'unknown'
        
        suggestion = {
            'type': f"{evaluation['type']}_{primary_concept}".replace(' ', '_'),
            'concept': primary_concept,
            'priority': evaluation['priority'],
            'urgency': evaluation['urgency'],
            'actionability': evaluation['actionability'],
            'goal_impact': evaluation['goal_impact'],
            'created_at': datetime.now().isoformat(),
            'origin_thought_id': thought.get('id', ''),
            'evaluation': {
                'type_confidence': evaluation['type_confidence'],
                'goal_similarity': evaluation['goal_similarity'],
                'friction_score': evaluation['friction_score'],
                'concepts_extracted': len(concepts)
            },
            'metadata': {
                'source': 'advanced_thought_evaluator',
                'version': '2.0'
            }
        }
        
        return suggestion
    
    def _load_recent_thoughts(self, max_thoughts: int = 10) -> List[Dict]:
        """Load recent thoughts via DAL."""
        try:
            from core.data.access import DataAccess
            return DataAccess(str(self.persona_dir)).get_thoughts(limit=max_thoughts)
        except Exception:
            thought_stream_file = self.persona_dir / 'thought_stream.json'
            if not thought_stream_file.exists():
                return []
            try:
                with open(thought_stream_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    thoughts = data.get('thoughts', [])
                    return thoughts[-max_thoughts:] if thoughts else []
            except Exception:
                return []
    
    def _load_active_goals(self) -> List[Dict]:
        """Load active goals via DAL."""
        try:
            from core.data.access import DataAccess
            return DataAccess(str(self.persona_dir)).get_goals(status='active')
        except Exception:
            goals_file = self.persona_dir / 'goals.json'
            if not goals_file.exists():
                return []
            try:
                with open(goals_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw = data.get('goals', {})
                    goals = list(raw.values()) if isinstance(raw, dict) else raw
                    return [g for g in goals if g.get('status') == 'active']
            except Exception:
                return []
    
    def _mark_thought_evaluated(self, thought_id: str):
        """Mark a thought as evaluated via DAL (atomic update)."""
        if not thought_id:
            return
        try:
            from core.data.access import DataAccess
            dal = DataAccess(str(self.persona_dir))
            thoughts = dal.get_thoughts(skip_expired=False, limit=200)
            for t in thoughts:
                if t.get('id') == thought_id:
                    t['evaluated'] = True
                    t['evaluated_at'] = datetime.now().isoformat()
                    t['evaluator_version'] = '2.0_advanced'
                    dal.add_thought(t)
                    break
        except Exception:
            pass

if __name__ == "__main__":
    print("="*70)
    print("ADVANCED THOUGHT EVALUATOR TEST")
    print("="*70)
    
    evaluator = AdvancedThoughtEvaluator('data/persona')
    
    # Test evaluation
    suggestions = evaluator.evaluate_recent_thoughts()
    
    print(f"\nGenerated {len(suggestions)} goal suggestions:\n")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. Type: {suggestion['type']}")
        print(f"   Concept: {suggestion['concept']}")
        print(f"   Priority: {suggestion['priority']:.2f}")
        print(f"   Urgency: {suggestion['urgency']:.2f}")
        print(f"   Actionability: {suggestion['actionability']:.2f}")
        print(f"   Goal Impact: {suggestion['goal_impact']:.2f}")
        print(f"   Type Confidence: {suggestion['evaluation']['type_confidence']:.2f}")
        print(f"   Goal Similarity: {suggestion['evaluation']['goal_similarity']:.2f}")
        print(f"   Friction: {suggestion['evaluation']['friction_score']:.2f}")
        print()
    
    print("="*70)
    print("✅ Advanced Thought Evaluator Test Complete")
    print("="*70)
