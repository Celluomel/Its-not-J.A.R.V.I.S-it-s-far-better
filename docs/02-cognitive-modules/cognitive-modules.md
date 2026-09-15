# Cognitive Modules Documentation

## 📊 Overview: 53+ Cognitive Modules Organized by Subsystem

### A. Identity & Memory Subsystem (7 modules)

#### ai_identity.py — Identity Building System
**Purpose:** Constructs and maintains the AI's self-model (identity representation)

**Key Features:**
- **Queue System Fix:** Fixed buffer overflow in original implementation using cursor persistence (`data/identity_cursor.json`)
- **Identity Representation:** Stores beliefs, goals, personality traits  
- **Evolution Mechanism:** Identity adapts through experience and reflection

**Integration Points:**
- Subscribes to `EXTERNAL_FEEDBACK` events from user interactions
- Publishes `IDENTITY_UPDATE` to global workspace
- Persists identity snapshots to JSON state file

**Architecture Pattern:** Narrative Identity Framework (constructs evolving self-story)

---

#### semantic_memory.py — Semantic Graph Memory
**Purpose:** Builds and maintains the semantic knowledge graph

**Architecture:**
```
Concept Nodes ↔ Concept Edges (relations, associations)
    ↓              ↓
Semantic Content   Relationships between concepts
```

**Key Functions:**
- **Graph Building:** Constructs concept graph from LLM outputs
- **Retrieval:** Finds relevant concepts given query vector
- **Evolution:** Adds new concepts and relations over time

**Persistence Layer:** FAISS vector index + JSON graph structure for fast semantic search

---

#### memory_manager.py — Memory Persistence Layer
**Handles:**
- SQLite database (`data/lumina_memory.db`) for structured memory records  
- FAISS embedding index (`data/identity_fvecs`, `data/identity_pth`) for semantic search
- JSON file storage (`memory/`) for episodic memories as text logs

---

#### relational_memory.py — Relational Fact Storage
**Stores:** Structured facts about world models and learned concepts

**Data Structure:** 
```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    subject STRING,
    predicate STRING, 
    object STRING,
    confidence REAL,
    timestamp DATETIME
);
```

---

### B. Goal & Motivation Subsystem (5 modules)

#### goal_engine.py — Goal Derivation & Lifecycle Management
**Purpose:** Translates high-level intentions into actionable goals

**Lifecycle Stages:**
1. **Derivation:** Generate goal from user intent or curiosity trigger
2. **Decomposition:** Break into sub-goals and action plans  
3. **Persistence:** Store active goals in SQLite (`goals/`)
4. **Completion Detection:** Recognize when goal achieved via result checking
5. **Learning:** Extract lessons from completed goals

**Integration:** Coordinates with pressure system for priority ranking

---

#### pressure_system.py — Motivational Pressure Calculator
**Pressure Types:**
- **Epistemic Pressure:** Desire to learn/understand
- **Expression Pressure:** Need to express/create  
- **Identity Pressure:** Maintain self-coherence
- **Goal Pressure:** Complete active objectives

**Function:** Computes pressure vector → influences goal selection & attention allocation

---

#### curiosity_engine.py — Intrinsic Exploration Driver
**Mechanism:** Identifies knowledge gaps and unexpected observations

**Triggers:**
- Surprise detection (high prediction error)
- Ambiguity in current state representation
- Novelty seeking based on prior interests

**Integration:** Communicates with world model to locate uncertainty regions

---

#### attention_system.py — Resource Allocation
**Function:** Decides which cognitive modules get processing attention

**Factors:** Pressure levels, urgency signals, relevance to active goals

---

### C. Metacognition & Observation Subsystem (4 modules)

#### meta_cognition.py — Self-Reflection Module
**Purpose:** Monitors own cognitive processes

**Capabilities:**
- **Process Monitoring:** Tracks which modules are active in current cycle
- **Reasoning Chain Analysis:** Examines problem-solving approaches taken  
- **Bias Detection:** Identifies potential thinking errors or blind spots

---

#### cognitive_observatory.py — Real-time Metrics System
**Metrics Tracked:**
- **CCS (Cognitive Complexity Score):** How sophisticated current reasoning is
- **GEI (Goal Engagement Index):** Strength of pursuit on active goals  
- **IDX (Information Integration Density):** Number of concepts connected in current processing
- **STR (Strategy Transfer Rate):** Success rate applying past solutions to new problems

**Function:** Provides real-time "health" metrics and cognitive state indicators

---

#### safety_constraints.py — Safety & Boundary Enforcement
**Ensures:** Safe operation, respects user boundaries, prevents harmful actions

**Checks:** Content filters, access controls, ethical boundaries before action execution

---

### D. Perception & Simulation Subsystem (6 modules)

#### perception_simulator.py — Sensory Processing Interface
**Modality Handling:**
- **Visual:** Processes camera input via visual_processor.py
- **Auditory:** Processes audio via audio_listener.py  
- **Tactile:** Touch input via tactile_processor.py
- **Proprioceptive:** Body position via proprioceptor_processor.py
- **Vestibular:** Balance/orientation via vestibular_controller.py

**Integration:** Converts raw sensor data into cognitive workspace representations

---

#### visual_processor.py — Visual Feature Extraction
**Processes:** Camera frames → visual features (edges, motion, objects)

#### audio_listener.py — Audio Processing  
**Processes:** Microphone input → speech recognition, sound classification

#### tactile_processor.py — Tactile Input
**Processes:** Touch sensor data → haptic feature extraction

#### proprioceptor_processor.py — Body Position
**Processes:** Position/velocity sensors → spatial awareness signals

#### vestibular_controller.py — Balance Control
**Manages:** Balance corrections, orientation adjustments

---

### E. Long-Term Memory Consolidation Subsystem (5 modules)

#### longterm_memory_manager.py — Consolidation Scheduler
**Function:** Periodically consolidates short-term experiences to long-term memory (typically nightly or idle periods)

---

#### episodic_memory_controller.py — Episodic Experience Storage
**Stores:** Specific events, interactions, learning episodes as structured records

---

#### semantic_memory_controller.py — Semantic Knowledge Updates
**Function:** Extracts new concepts and relations from consolidated episodic memories

---

#### conceptual_graph_builder.py — Semantic Graph Construction
**Builds:** Concept nodes and edges representing knowledge structure from processed experiences

---

#### world_model.py — Predictive World Model
**Purpose:** Creates internal model of environment for prediction

**Mechanism:** Learn causal relationships, simulate potential outcomes before action

---

### F. Research & MCP Integration Subsystem (3 modules)

#### research_mcp/ — Research Capabilities Module
**Components:**
- `stealth_browser.py`: Headless browsing for research queries
- `search_providers.py`: Multiple search engine APIs (DuckDuckGo, Google, Bing)
- `claude_search.py`: Advanced semantic search with context retrieval

**Trigger Pattern:** Activated when curiosity engine identifies knowledge gaps requiring external information acquisition

---

## 🧠 Module Communication Architecture

### Event-Driven Coordination
All modules subscribe to `cognitive_event_bus` and respond to specific event types:
- `USER_INPUT` — From UI interactions
- `PERCEPTION_UPDATE` — Sensory data processed
- `GOAL_DERIVED` — New goal identified  
- `MEMORY_CONSOLIDATED` — Learning stored
- `REFLECTION_COMPLETE` — Self-monitoring done
- `CURIOSITY_SPIKE` — Knowledge gap detected

### State Sharing
Modules read/write to `global_workspace_state` for information sharing and coordination without direct coupling.

---

## 📈 Design Principles

1. **Single Responsibility:** Each module handles one cognitive function
2. **Loose Coupling:** Event bus enables independent modules to coordinate
3. **Persistent Memory:** SQLite + FAISS enable durable knowledge storage
4. **Metacognitive Layer:** Self-monitoring built into core architecture
5. **Research Integration:** Autonomous learning through web search capabilities
