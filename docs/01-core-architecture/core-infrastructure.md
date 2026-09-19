# Core Infrastructure Documentation

## app.py — Main Application Entry Point

**Purpose:** Streamlit web interface + main orchestration logic for PandoraBOX Cognitive Organism

### Key Components:

#### 1. UI Entry Points (Streamlit Pages)
- Located in `pages/` subdirectory
- Handle user interactions through Streamlit callbacks
- Trigger cognitive events when users interact with UI elements

#### 2. Module Instantiation Pattern
```python
# At startup, these cognitive subsystems are instantiated:
ai_identity = AIIdentity()                    # Identity building
semantic_memory = SemanticMemoryController()  # Memory management  
goal_engine = GoalEngine()                   # Goal derivation
curiosity_engine = CuriosityEngine()         # Intrinsic exploration
meta_cognition = MetaCognition()             # Self-monitoring
```

#### 3. Event Broadcasting System
- Uses `cognitive_event_bus` for pub/sub pattern
- UI actions publish events to cognitive modules
- Modules subscribe to relevant event types

### Main Execution Flow:
1. User interacts with Streamlit UI → creates event
2. Event published to cognitive_event_bus
3. Global workspace receives event → processes through cognitive loop
4. Output generated → displayed back in UI

### Connection Pattern:
```
UI Layer (app.py) 
    ↓ triggers events
Event Bus (cognitive_event_bus.py)
    ↓ distributes to subscribers
Cognitive Modules (53+ modules in cognition/)
    ↓ process & update state
Global Workspace State (shared memory buffer)
    ↓ feeds back results
UI Output Display
```

---

## core/ — Core Orchestration Layer

### cognitive_event_bus.py — Event Dispatch System
**Purpose:** Pub/sub event distribution connecting all cognitive modules

**Key Features:**
- Event classes for each message type (USER_INPUT, PERCEPTION_UPDATE, GOAL_ACHIEVED)
- Subscriber registration system
- Thread-safe event queue management

### global_workspace_state.py — Shared Memory Buffer
**Purpose:** Persistent state maintained across cognitive cycles

**State Components:**
- Current perceptual inputs
- Active goals and pressure levels
- Recent memory activations
- Cognitive process logs

### orchestrator.py — Main Cycle Logic
**Purpose:** Coordinates each cognitive processing cycle

**Cycle Steps:**
1. Integrate new perceptions into workspace
2. Evaluate goal pressures and motivations
3. Allocate attention to most relevant processes
4. Run cognitive modules in parallel
5. Select insight for broadcasting (global workspace)
6. Consolidate learning experiences
7. Generate appropriate output/actions

---

## Summary

The core infrastructure establishes the **event-driven architecture** that allows 53+ cognitive modules to communicate and coordinate their activities through:
- A global workspace as shared state
- An event bus for inter-module messaging  
- Orchestrated cycles that integrate perception, cognition, action

This modular design enables emergent intelligence from distributed specializations.