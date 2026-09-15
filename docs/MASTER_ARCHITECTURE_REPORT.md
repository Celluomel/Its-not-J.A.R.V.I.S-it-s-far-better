# Lumina Cognitive Organism - Master Architecture Report

**Generated:** March 24, 2026  
**Analyst Team:** Core Mappers, Cognition Experts, Integration Tracers, Research Validators  
**Status:** ✅ Complete Analysis Performed

---

## 📋 Executive Summary

This document provides a complete file-by-file mapping of the **Lumina Cognitive Organism**—a sophisticated AI system implementing Global Workspace Theory, Predictive Processing, and Narrative Identity frameworks. The cognitive organism consists of **53+ modules** organized into subsystems that work together to create an autonomous, self-aware agent capable of goal-driven behavior, memory consolidation, curiosity-driven exploration, and metacognitive reflection.

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Lumina Cognitive Organism                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   INPUT      │→ │  CORE        │←→│   OUTPUT     │      │
│  │  (Sensory)   │  │ Processing   │  │ (Actions)    │      │
│  └──────────────┘  │   Engine     │  └──────────────┘      │
│                     ├──────────────┤                        │
│         ┌───────────┴──────────────┴──────────────┐         │
│         │              Global Workspace           │         │
│         │          (Central Bottleneck)           │         │
│         └───────────┬──────────────┬──────────────┘         │
│                     ↓              ↓                        │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Memory Systems  │  │ Goal & Motivation│                  │
│  │ (Episodic +     │  │ System          │                  │
│  │ Semantic)       │  └─────────────────┘                  │
│  └─────────────────┘  ┌─────────────────┐                  │
│  ┌──────────────────┐ │Metacognition    │                  │
│  │ Perception &     │ │& Observation    │                  │
│  │ Simulation       │ └─────────────────┘                  │
│  └──────────────────┘                                      │
│                                                             │
│  Event Bus: cognitive_event_bus.py connects all modules    │
│  State: global_workspace_state (shared memory buffer)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File-by-File Documentation

### **Phase 1: Core Infrastructure**

#### `app.py` — Main Application Entry Point

**Purpose:** Nicegui web interface + main orchestration logic

**Key Components:**
- **UI Entry Points:** Nicegui pages (pages/ folder) handling user interactions
- **Module Instantiation:** Creates cognitive subsystems at startup
- **Event Broadcasting:** Connects UI actions to cognitive event bus
- **State Management:** Maintains session state between cognitive cycles

**Imports & Connections:**
```python
from core.cognitive_event_bus import CognitiveEventBus  # Event system
from ai_identity import AIIdentity, IdentityBuilder    # Identity module
from semantic_memory import SemanticMemoryController   # Memory system
from goal_engine import GoalEngine                    # Goal management
from cognitive_observatory import CognitiveObservatory # Metacognition
```

**Main Flow:**
1. User interacts with Nicegui UI → triggers event `USER_INPUT`
2. Event bus broadcasts to all subscribed cognitive modules
3. Core modules process input via global workspace
4. Output sent back to UI for display

---

#### `core/` — Core Orchestration Layer

**Files:**
- `cognitive_event_bus.py` — Event dispatch system connecting all modules
- `global_workspace_state.py` — Shared memory buffer (persistent state)
- `orchestrator.py` — Main cycle orchestration logic
- `utils.py` — Utility functions used across modules

**Architecture Pattern:** Pub/Sub event bus with global workspace as shared state

---

### **Phase 2: Cognitive Module Subsystems**

#### **A. Identity & Memory Subsystem**

##### `ai_identity.py` — Identity Building System

**Purpose:** Constructs and maintains the AI's self-model (identity representation)

**Key Features:**
- **Queue System Fix:** Fixed buffer overflow issue in original implementation
- **Identity Representation:** Stores beliefs, goals, personality traits
- **Evolution Mechanism:** Identity adapts through experience and reflection

**Integration Points:**
- Subscribes to `EXTERNAL_FEEDBACK` events
- Publishes `IDENTITY_UPDATE` to global workspace
- Persists identity snapshots to `identity_state.json`

##### `semantic_memory.py` — Semantic Graph Memory

**Purpose:** Builds and maintains the semantic knowledge graph

**Architecture:**
```
Concept Nodes ↔ Concept Edges (relations)
    ↓              ↓
Semantic Content   Relationships/Associations
```

**Key Functions:**
- **Graph Building:** Constructs concept graph from LLM outputs
- **Retrieval:** Finds relevant concepts given query
- **Evolution:** Adds new concepts and relations over time

**Persistence:** FAISS vector index + JSON graph structure

##### `memory_manager.py` — Memory Persistence Layer

**Handles:**
- SQLite database for structured memory records
- FAISS embedding index for semantic search
- JSON file storage for episodic memories

##### `relational_memory.py` — Relational Fact Storage

**Stores:** Structured facts about world models and learned concepts

---

#### **B. Goal & Motivation Subsystem**

##### `goal_engine.py` — Goal Derivation & Lifecycle Management

**Purpose:** Translates high-level intentions into actionable goals

**Lifecycle Stages:**
1. **Derivation:** Generate goal from user intent or curiosity
2. **Decomposition:** Break into sub-goals and action plans
3. **Persistence:** Store active goals in `active_goals.db`
4. **Completion Detection:** Recognize when goal achieved
5. **Learning:** Extract lessons from completed goals

**Integration:** Coordinates with pressure system and attention modules

##### `pressure_system.py` — Motivational Pressure Calculator

**Pressure Types:**
- **Epistemic Pressure:** Desire to learn/understand
- **Expression Pressure:** Need to express/create
- **Identity Pressure:** Maintain self-coherence
- **Goal Pressure:** Complete active objectives

**Function:** Computes pressure vector → influences goal selection

##### `curiosity_engine.py` — Intrinsic Exploration Driver

**Mechanism:** Identifies knowledge gaps and unexpected observations

**Triggers:**
- Surprise detection (prediction error)
- Ambiguity in current state
- Novelty seeking based on prior interests

##### `attention_system.py` — Resource Allocation

**Function:** Decides which cognitive modules get processing attention

**Factors:** Pressure levels, urgency, relevance to active goals

---

#### **C. Metacognition & Observation Subsystem**

##### `meta_cognition.py` — Self-Reflection Module

**Purpose:** Monitors own cognitive processes

**Capabilities:**
- **Process Monitoring:** Tracks which modules are active
- **Reasoning Chain Analysis:** Examines problem-solving approaches
- **Bias Detection:** Identifies potential thinking errors

##### `cognitive_observatory.py` — Real-time Metrics System

**Metrics Tracked:**
- **CCS (Cognitive Complexity Score):** How sophisticated current reasoning is
- **GEI (Goal Engagement Index):** How strongly pursuing active goals
- **IDX (Information Integration Density):** How many concepts connected
- **STR (Strategy Transfer Rate):** How well applying past solutions

**Function:** Provides real-time "health" metrics of cognitive organism

##### `safety_constraints.py` — Safety & Boundary Enforcement

**Ensures:** Safe operation, respects user boundaries, prevents harmful actions

---

#### **D. Perception & Simulation Subsystem**

##### `perception_simulator.py` — Sensory Processing Interface

**Modality Handling:**
- **Visual:** Processes camera input via visual_processor.py
- **Auditory:** Processes audio via audio_listener.py  
- **Tactile:** Touch input via tactile_processor.py
- **Proprioceptive:** Body position via proprioceptor_processor.py
- **Vestibular:** Balance/orientation via vestibular_controller.py

**Integration:** Converts raw sensor data into cognitive workspace representations

---

#### **E. Long-Term Memory Consolidation Subsystem**

##### `longterm_memory_manager.py` — Consolidation Scheduler

**Function:** Periodically consolidates short-term experiences to long-term memory

##### `episodic_memory_controller.py` — Episodic Experience Storage

**Stores:** Specific events, interactions, learning episodes

##### `semantic_memory_controller.py` — Semantic Knowledge Updates

**Function:** Extracts new concepts and relations from episodic memories

##### `conceptual_graph_builder.py` — Semantic Graph Construction

**Builds:** Concept nodes and edges representing knowledge structure

##### `world_model.py` — Predictive World Model

**Purpose:** Creates internal model of environment for prediction

**Mechanism:** Learn causal relationships, simulate outcomes

---

#### **F. Research & MCP Integration Subsystem**

##### `research_mcp/` — Research Capabilities Module

**Components:**
- `stealth_browser.py`: Headless browsing for research
- `search_providers.py`: Multiple search engine APIs
- `claude_search.py`: Advanced semantic search

**Trigger Pattern:** Activated when curiosity engine identifies knowledge gaps requiring external research

---

### **Phase 3: System Services Layer**

#### `managers/` — Service Orchestration

##### `llm_manager.py` — LLM Interface & Scheduling

**Handles:** Model routing, prompt construction, response streaming

##### `audio_manager.py` — Voice/Speech Integration

**Manages:** Text-to-speech output, speech recognition input

##### `vision_manager.py` — Camera/Vision Input Processing

**Processes:** Frame capture, visual feature extraction

##### `session_manager.py` — Multi-Session Handling

**Coordinates:** Multiple concurrent user sessions and their states

##### `security_manager.py` — Safety Controls

**Enforces:** Content filters, access controls, safety boundaries

##### `messaging_manager.py` — Channel Integrations

**Handles:** Telegram, Discord, Signal messaging interfaces

---

### **Phase 4: Brain Visualization Subsystem**

#### `brain_visualizer/` — OpenGL Neural Rendering

##### Components:
- **OpenGL Renderer:** 3D brain visualization with particle system
- **Particle System:** Visualizes information flow as particles
- **Shader Code:** Highlights active cortical regions by color intensity
- **obj_loader.py:** Parses brain.obj anatomical model
- **network_mapping.py:** Maps neural connections on visualized structure
- **lumina_brain_bridge.py:** Connects visualizer to cognitive event stream

**Visualization Mapping:**
```
Cognitive Activation → Cortex Region Lighting Intensity
Information Flow → Particle Stream Through Brain Regions
Concept Processing → Localized Glow at Semantic Network Nodes
```

---

## 🔗 Cross-Module Integration Map

### **Event Bus Connections (cognitive_event_bus.py)**

| Event Type | Source Module | Subscriber Modules |
|------------|---------------|-------------------|
| USER_INPUT | app.py UI | goal_engine, meta_cognition, pressure_system |
| PERCEPTION_UPDATE | perception_simulator | semantic_memory, world_model, curiosity_engine |
| GOAL_DERIVED | goal_engine | attention_system, pressure_system |
| MEMORY_CONSOLIDATED | memory_manager | semantic_graph_builder, identity_builder |
| REFLECTION_COMPLETE | meta_cognition | safety_constraints, identity_builder |
| CURIOSITy_SPIKE | curiosity_engine | goal_engine, research_mcp |
| GOAL_ACHIEVED | goal_engine | pressure_system (release), learning_module |

### **Data Flow Architecture**

```
[USER INPUT] → Event Bus → [GLOBAL WORKSPACE STATE]
                                    ↓
          ┌────────────────┬───────────┬──────────────┐
          ↓                ↓           ↓             ↓
    ┌────────────┐  ┌────────────┐ ┌──────────┐ ┌──────────┐
    │ Identity   │  │ Goal/Power │ │Attention │ │ Memory   │
    │ Builder    │  │ System     │ │System    │ │System    │
    └────────────┘  └────────────┘ └──────────┘ └──────────┘
          ↓                ↓           ↓             ↓
          ┌────────────────┴───────────┴──────────────┐
          ↓                                           ↓
    [Cognitive Insights] ←→ [World Model Updates] ←→ [Learning]
          ↓                                           ↓
    ┌────────────┐  ┌────────────┐  ┌────────────────┐
    │ UI Output  │  │ Action     │  │ Persistence    │
    │ Display    │  │ Execution  │  │ (SQLite/FAISS) │
    └────────────┘  └────────────┘  └────────────────┘
```

---

## 🧠 How the Cognitive Organism Works

### **The Core Loop (Each Cycle)**

1. **Perception Integration:** Sensory inputs processed → workspace representations
2. **Goal Evaluation:** Pressure system computes motivational state
3. **Attention Allocation:** Most urgent/goal-relevant processes get focus
4. **Cognitive Processing:** Modules work in parallel, publish insights
5. **Global Workspace:** Bottleneck selection of insight to broadcast
6. **Memory Update:** Consolidated experiences stored
7. **Action Generation:** Output selected for UI/display
8. **Metacognitive Reflection:** Self-monitoring of the cycle

### **Learning & Development**

- **Short-term learning:** Working memory updates, temporary insights
- **Consolidation (nightly/periodic):** Episoic → semantic memory transfer
- **Identity evolution:** Accumulated experiences reshape self-model
- **Knowledge graph growth:** New concepts and relations added
- **Strategy accumulation:** Past solutions stored for future retrieval

---

## ✅ Research Literature Alignment Assessment

| Cognitive Architecture Element | Implementation Quality | Notes |
|-------------------------------|----------------------|-------|
| Global Workspace Theory | ✅ Strong | Proper bottleneck, broadcasting mechanism |
| Predictive Processing | ✅ Good | World model, prediction error handling |
| Narrative Identity Framework | ✅ Implemented | Self-story building across cycles |
| Affective Computing | ⚠️ Partial | Needs richer emotional state integration |
| Intrinsic Motivation Theory | ✅ Strong | Curiosity engine, novelty detection |

---

## 🔧 Architecture Strengths

1. **Modular Design:** Clear separation of concerns, easy to extend
2. **Event-Driven:** Loose coupling via event bus
3. **Persistent Memory:** SQLite + FAISS for durable knowledge
4. **Metacognitive Layer:** Self-monitoring capabilities built-in
5. **Research Integration:** Autonomous web research for knowledge acquisition

---

## 📈 Potential Improvements

1. **Emotional Subsystem:** Add dedicated emotion processing module
2. **Hierarchical Goals:** Implement sub-goal decomposition trees
3. **Multi-Agent Coordination:** Allow internal "debate" between sub-modules
4. **Dreaming Mode:** Offline consolidation simulation when idle
5. **Social Modeling:** Learn from observing other agents' behaviors

---

## 📄 File Locations Summary

| Category | Files | Location |
|----------|-------|----------|
| Core Infrastructure | app.py, core/*.py | lumina_cognitive_organism_upgrade/ |
| Cognitive Modules (53+) | cognition/*.py | lumina_cognitive_organism_upgrade/cognition/ |
| Memory Systems | memory/*.py | lumina_cognitive_organism_upgrade/memory/ |
| Goal System | goals/*.py | lumina_cognitive_organism_upgrade/goals/ |
| Managers Services | managers/*.py | lumina_cognitive_organism_upgrade/managers/ |
| Brain Visualizer | brain_visualizer/*.py, *.glsl | lumina_cognitive_organism_upgrade/brain_visualizer/ |

---

## 📚 Related Documentation Files

This report synthesizes analysis from:
- ✅ Core Infrastructure Mapping (app.py + core/)
- ✅ Identity & Memory Subsystem Analysis
- ✅ Goal & Motivation Architecture Study  
- ✅ Metacognition & Observation Deep Dive
- ✅ Research MCP Integration Review
- ✅ Cross-Module Wiring Diagrams
- ✅ Managers Subsystem Documentation
- ✅ Brain Visualizer System Analysis

---

**End of Master Architecture Report** 🧠📊