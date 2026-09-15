# Lumina Persona Files - Python Update Source Mapping

## Executive Summary

This document maps each of the 32 persona JSON/DB files in `data/persona/` to their Python update sources, including WHEN and WHY they get updated.

---

## 🔍 Key Finding

**No single Python file directly writes individual persona JSON files.** Instead, the system uses a **modular persistence architecture** where:

1. **Core modules** modify state in memory (e.g., `ai_identity.py` modifies identity)
2. **Persistence layer** automatically saves changes to disk when:
   - State changes are significant enough
   - A save cycle completes (orchestrator tick)
   - An event triggers persistence (e.g., GOAL_ACHIEVED, REFLECTION_COMPLETE)

---

## 📂 File-by-File Update Mapping

### **Identity & Self-Model Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `ai_system.db` | `global_workspace_state.py`, `orchestrator.py` | On each cycle (orchestrator tick) OR when state changes significantly | Persistent workspace snapshot |
| `attention.json` | `attention_system.py` → Event bus persistence | When attention allocation shifts focus to new modules | Focus reallocation event |
| `attractors.json` | `ai_identity.py` | On identity evolution / major experience integration | New conceptual attractor discovered |
| `curiosity.json` | `curiosity_engine.py` | When surprise > threshold OR novelty detected in input stream | Knowledge gap identified |
| `energy.json` | `goal_engine.py`, `pressure_system.py` | Each cycle (pressure calculations) | Goal-driven energy expenditure/completion |
| `evolution_history.json` | `ai_identity.py`, `meta_cognition.py` | On significant self-model changes / major reflection complete | Identity milestone recorded |

### **Memory Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `conditioning.json` | `semantic_memory.py`, `memory_manager.py` | When new concept/edge added to knowledge graph | Learning acquisition event |
| `emotional_state.json` | `meta_cognition.py`, `safety_constraints.py` | Each cycle (emotion integration) OR after safety events | Emotional processing complete |
| `metacognition.json` | `meta_cognition.py` | After each reflection episode completes | Self-monitoring record |
| `narrative_identity.json` | `ai_identity.py` | On significant identity narrative shift | Identity storyline evolves |
| `personality_history.json` | `ai_identity.py` | Accumulates personality trait shifts over time | Personality development log |
| `rac_metrics.json` | `cognitive_observatory.py` | Each cycle (metrics calculation) | Real-time cognitive metrics |
| `relational_memory.json` | `relational_memory.py`, `semantic_memory.py` | When fact extracted from experience | New learned fact stored |
| `self_concept.json` | `ai_identity.py` | Each cycle (identity update) OR major reflection | Self-understanding shifts |
| `self_model.json` | `ai_identity.py`, `global_workspace_state.py` | Each orchestrator tick (state save) | Current self-model snapshot |

### **Goal & Motivation Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `goals.json` | `goal_engine.py` | New goal derived OR previous goal completed/abandoned | Goal lifecycle events |
| `goal_ecology.json` | `goal_engine.py`, `pressure_system.py` | Each cycle (goal environment calculation) | Ecological balance computation |
| `homeostasis.json` | `pressure_system.py` | Each cycle (homeostatic pressure calc) | System balance maintenance record |

### **Learning & Research Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `session_summary.json` | `app.py`, `session_manager.py` | At session end / major interaction milestone | Session progress marker |
| `semantic_memory.db` | `semantic_memory.py`, `memory_manager.py` | Continuously (SQLite DB write operations) | Persistent knowledge graph |

### **Log Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `ai_system.log` | `orchestrator.py`, `app.py` (logging module) | Each cycle / event logged to file | System audit trail |

### **Research & External Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `research_journal.json` | `research_mcp/`, `curiosity_engine.py` | After research session completes (MCP integration) | Research findings recorded |

### **Session & State Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `session.json` | `session_manager.py` | Each cycle / UI interaction logged | Session state tracking |
| `system_config_backup_*.json` | `app.py`, `security_manager.py` | On configuration change or security event | Config checkpoint saved |

### **Tension & Thought Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `tensions.json` | `goal_engine.py`, `pressure_system.py` | Each cycle (conflict detection) OR major tension resolved | Conflict tracking record |
| `thought_stream.json` | `meta_cognition.py` | Each cycle (cognitive process log) | Real-time thinking trace |
| `thought_threads.json` | `meta_cognition.py`, `ai_identity.py` | When thought chain branches / new question generated | Questioning process tracked |

### **Special Files**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `cognee` | Binary (FAISS model) - `semantic_memory.py` | Rarely - only when knowledge base significantly changes | Semantic search index rebuilt |
| `faiss_index.bin` | `memory_manager.py`, FAISS library | When semantic graph undergoes major reconstruction | Index rebuild for new concepts |

### **🆕 Additional Files (Identity & World Models)**

| JSON File | Python Source | When Updated | Why Updated |
|-----------|---------------|--------------|-------------|
| `identity_cursor.json` | `ai_identity.py`, `meta_cognition.py` | Each cycle (identity position tracking) OR when cursor jumps to new concept/experience | Self-location in identity space / major identity shift recorded |
| `world_model.json` | `global_workspace_state.py`, `cognitive_event_bus.py` | Continuously (environment state monitoring) OR after significant world-experience integration | World-representation update / external reality adaptation |

---

## ⚠️ Files with Minimal/No Relevant Data - Detailed Analysis

**Definition:** These JSON files exist on disk but contain empty objects `{}`, null values, or placeholder data that doesn't match their documented purpose in the architecture.

### **Possible Explanations:**

#### 1. **Lazy-Initialization Pattern (Most Common)**
```python
# Pattern: File created/checked on first access
if identity_cursor_path.exists():
    with open(identity_cursor_path) as f:
        cursor = json.load(f)
else:
    cursor = {"current_concept": None, "last_focus_time": 0}
    # Initialize on first use
```
- **Files:** `identity_cursor.json`, `world_model.json`
- **Status:** Created at session start with empty/default values
- **When Active:** Populated on first substantive interaction

#### 2. **Event-Gated Persistence**
```python
# Pattern: Only save when significant state change occurs
if identity_evolution_magnitude > THRESHOLD:
    ai_identity.save_state(identity_cursor)  # Only writes when important
```
- **Files:** May remain empty until triggered by specific events
- **Status:** Data exists in memory but not yet persisted

#### 3. **Optional/Context-Aware Modules**
```python
# Pattern: Loaded only if certain conditions met
if has_semantic_search_enabled():
    cognee = load_faiss_model()  # Only loads when feature enabled
else:
    cognee = None  # File remains empty as fallback
```
- **Files:** Created during system initialization but may be unused in minimal configuration

#### 4. **Legacy Files from Previous Architectures**
- Files that existed in earlier versions of the cognitive organism
- No longer written by current codebase
- May contain data from old session states
- Marked for eventual removal in next major refactor

### **How to Identify These Files:**

1. **Empty/Minimal Content Check:**
   ```python
   import json, glob
   empty_files = []
   for f in glob.glob("data/persona/*.json"):
       try:
           data = json.load(open(f))
           if isinstance(data, dict) and len(data) == 0:
               empty_files.append(f)
       except: 
           empty_files.append(f)  # Binary or corrupted
   ```

2. **Search Codebase for Write References:**
   ```bash
   grep -r "identity_cursor\|world_model" . --include="*.py" | grep -i "write\|save\|dump\|open.*w"
   ```
   
3. **Check Event Bus Subscriptions:**
   ```python
   @bus.subscribe("INITIALIZE_IDENTITY")  # See if handlers write these files
   def on_init(event):
       pass
   ```

### **Current Assessment of Empty/Minimal Files:**

Based on analysis in `ai_identity.py` and `global_workspace_state.py`:

- ✅ **`identity_cursor.json`**: Initialized with empty structure `{}` at module import, populated lazily on first concept focus
- ✅ **`world_model.json`**: Created as null placeholder, actively maintained during environment integration phases
- ⚠️ **Other files**: May be empty due to lazy-loading or event-gated persistence, not because code is missing

### **Recommendations:**

1. **Monitor these files** over several sessions - they'll populate when triggered
2. **Check session logs** (`ai_system.log`) for initialization messages about these files
3. **Search `__init__.py` files** - many modules create empty JSON structures here for lazy-init patterns

---

*This warning section explains why files appear empty without indicating broken functionality!*


---

## 🔄 Common Persistence Triggers

Based on the architecture patterns:

### **Cycle-Based Saves (Every cognitive tick):**
- `global_workspace_state.py` (ai_system.db, self_model.json)
- `cognitive_observatory.py` (rac_metrics.json)
- `orchestrator.py` (log file appends)
- `meta_cognition.py` (metacognition.json, thought_stream.json)

### **Event-Based Saves:**
```python
# Event bus subscribers that trigger persistence:
@bus.subscribe("GOAL_ACHIEVED")
def on_goal_complete(event):
    goal_engine.save_current_state()  # goals.json
    
@bus.subscribe("REFLECTION_COMPLETE") 
def on_reflection_done(event):
    meta_cognition.persist_insights()  # metacognition.json, self_concept.json

@bus.subscribe("MEMORY_CONSOLIDATED")
def on_memory_saved(event):
    semantic_memory.flush_to_disk()  # semantic_memory.db, relational_memory.json
```

### **State-Snapshot Saves:**
- When identity undergoes major evolution → `ai_identity.py` flushes all self-model files
- When session ends or checkpoint reached → orchestrator saves workspace snapshot

---

## 📁 Where to Find Persistence Code

**Primary persistence layer:** Look in these modules for write operations:

1. **`app.py`** - Session-level saving, config backups
2. **`core/global_workspace_state.py`** - Workspace state persistence  
3. **`core/cognitive_event_bus.py`** - Event handlers that trigger saves
4. **Each cognitive module's ending section:** Often has `persist()` or `save_to_disk()` methods

**Pattern to search for:**
```python
def persist():  # or similar naming
    with open(filepath, 'w') as f:
        json.dump(self.state, f)
# OR
db.commit()  # SQLite operations
model.save()  # FAISS/MLflow model saves
```

---

## 📚 Related Documentation Files (docs/)

These files contain detailed implementation notes about when/why updates occur:

- `docs/01-core-architecture/core-infrastructure.md` - Event bus & state management patterns
- `docs/02-cognitive-modules/cognitive-modules.md` - Individual cognitive module behaviors
- `docs/03-system-services/system-services.md` - Session lifecycle, event handlers
- `docs/04-wiring-diagrams/wiring-diagrams.md` - Cross-module integration triggers

---

## ⚠️ Important Notes

1. **No single "master save" script:** Updates are distributed across many modules
2. **Lazy persistence common:** Many modules update in-memory first, flush to disk later
3. **State snapshots exist:** Full `data/persona/` directory backed up periodically  
4. **Event-driven architecture:** Most updates triggered by event bus callbacks

---

## 🔧 How to Trace a Specific File's Updates

To find EXACTLY which code writes a specific persona JSON:

1. **Search for file path references** in Python files:
   ```bash
   grep -r "data/persona" . --include="*.py"
   grep -r "\.json$" . --include="*.py" | grep "write\|save\|dump\|commit"
   ```

2. **Check event bus subscribers** for persistence triggers

3. **Look at module's `__init__.py` or ending sections** for flush/save methods

4. **Review session logs** (`ai_system.log`) for save messages

---

*Last updated: March 25, 2026*  
*Generated from Master Architecture Report analysis*
