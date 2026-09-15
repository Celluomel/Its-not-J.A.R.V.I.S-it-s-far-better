"""
Core AI System Package — Full Pass 1 + Pass 2
"""

from .ai_system import (
    EnhancedAISystem,
    ExternalLLMAdapter,
    SystemConfig,
    PersonalityState,
    validate_trait_value,
    DreamType,
    LifeEventDreamSystem,
    EnhancedLLM,
    EnhancedMemorySystem,
    EnhancedPersistence,
    DatabaseConfig,
    FAISSConfig,
    LLMConfig,
    AISystemError,
    DatabaseError,
    LLMError,
    MemorySystemError,
    ConfigurationError,
)

from .ai_identity import (
    IdentitySystem,
    AIIdentity,
    IdentityComponent,
    CoreValues,
    SelfNarrative,
    IdentityAnalyzer,
)

from .personality_evolution import PersonalityEvolutionEngine
from .emotional_state import EmotionalStateManager
from .relational_memory import RelationalMemorySystem, UserRelationship
from .self_concept import SelfConceptSystem, SelfBelief
from .behavioral_conditioning import BehavioralConditioningSystem, ConditioningSignal
from .life_stage_prompting import (
    build_stage_system_block,
    get_stage_profile,
    STAGE_PROFILES,
    StageProfile,
)

__all__ = [
    # Main system
    "EnhancedAISystem",
    "ExternalLLMAdapter",
    "SystemConfig",
    "PersonalityState",
    "LifeEventDreamSystem",
    "DreamType",
    # Persistence / LLM / Memory
    "EnhancedLLM",
    "EnhancedMemorySystem",
    "EnhancedPersistence",
    "DatabaseConfig",
    "FAISSConfig",
    "LLMConfig",
    # Identity
    "IdentitySystem",
    "AIIdentity",
    "IdentityComponent",
    "CoreValues",
    "SelfNarrative",
    "IdentityAnalyzer",
    # Pass 1
    "PersonalityEvolutionEngine",
    "EmotionalStateManager",
    # Pass 2
    "RelationalMemorySystem",
    "UserRelationship",
    "SelfConceptSystem",
    "SelfBelief",
    "BehavioralConditioningSystem",
    "ConditioningSignal",
    # Life stage
    "build_stage_system_block",
    "get_stage_profile",
    "STAGE_PROFILES",
    "StageProfile",
    # Utilities
    "validate_trait_value",
    # Exceptions
    "AISystemError",
    "DatabaseError",
    "LLMError",
    "MemorySystemError",
    "ConfigurationError",
]
