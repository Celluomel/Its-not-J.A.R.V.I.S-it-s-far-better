# LUMINA PHASE 2.9 — ATTENTION-BASED COGNITIVE ALLOCATION
## Implementation Plan

**Date:** 2026-06-16  
**Basis:** Lumina_evolution.txt proposal (Fred + Hermes/Gemma analysis)  
**Built on:** Phase 2.8 organic ecosystem (CognitiveFluxEngine, consolidation cycles, trust→drives)

---

## EVALUATION OF THE PROPOSAL

### What it gets exactly right

The core diagnosis is correct. The current workspace competition in
`workspace_competition.py` is threshold-gated scoring:

```python
# Current code (workspace_competition.py ~line 140)
if tension_value > 0.5:           # hard gate
    final_score = tension_value + identity_boost
    candidates.append(...)
```

And `MotivationalField._integrate_drives()` produces:

```python
if curiosity > threshold:
    → semantic_memory flux ↑
if trust > 0.5:
    → social drive += trust * 0.20
```

182 such threshold comparisons across 18 cognitive modules. Each one is a
discontinuous step function: nothing happens below, something discrete happens
above. The proposal's critique — "État → seuil → action, le cerveau ne
fonctionne pas réellement comme ça" — is architecturally accurate.

The attention matrix framing is also precisely right:

> "Chaque élément évalue l'importance de tous les autres éléments et produit
> des poids. Le système choisit alors où investir ses ressources."

This is exactly what `workspace_competition.compete()` *tries* to do but
implements imperfectly — candidates compete by score arithmetic rather than
mutual relevance weighting. A goal doesn't know it's competing against a
curiosity thread with overlapping semantic content.

**The zero-sum softmax insight is the strongest part.** Because attention
weights sum to 1.0, allocating 0.82 to relational_memory *automatically*
reduces what's available to everything else. There is no equivalent constraint
in the current system — goals, thoughts, and tensions all score independently
and a high-scoring goal doesn't reduce any thought's score. This allows
simultaneous high-salience in multiple candidates, which produces the
"everything fires at once" incoherence we've repeatedly patched around.

### Where the proposal overstates

**Replacing workspace_competition with a neural attention matrix** — the
full `AttentionWorkspace` class in the proposal with `W_q`, `W_k`, `W_v`
random projection matrices — is too much. Those matrices would need training
to be meaningful. With random initialization they're just random projections
of the input embeddings, producing random attention weights that are no
better than the current noise-breaking `random.uniform(-0.025, 0.025)`.

The proposal says "let a neural/mathematical attention matrix compute the
weights" but doesn't specify how the matrices are learned. Without a gradient
signal (there's none here — this is not a training loop), the W_q/W_k/W_v
matrices are decoration, not computation.

**The FAISS dynamic pruning claim** — "when relational_memory dominates,
the search radius for relational vectors opens wide" — doesn't map to how
FAISS works. FAISS `IndexFlatL2` has no variable search radius. The proposal
conflates attention weights with FAISS query parameters, which are separate
systems.

**"Lumina ceases to be a chat script trying to look human"** — this overstates
what an attention allocation layer achieves. The LLM is still the generator;
the attention layer governs what context reaches it, not how it generates.

### The right interpretation

The proposal's valuable insight is not "replace workspace_competition with a
Transformer" — it's:

> **Add a softmax normalization step to the existing workspace competition
> outputs so that module salience weights are mutually constraining (sum to 1)
> rather than independent scores.**

And separately:

> **Replace the binary threshold gates in MotivationalField, CuriosityEngine,
> and ContradictionHandler with continuous attention-weighted allocation.**

This is implementable in ~200 lines using the existing embedding model
(all-MiniLM-L6-v2 already loaded) and numpy. No new neural networks, no
training loops, no FAISS changes.

---

## ARCHITECTURE: ATTENTION ALLOCATION LAYER

```
INPUTS (already computed each slow cycle):
  MotivationalField.drive_vector     {curiosity:0.73, social:0.83, ...}
  TensionEngine.tensions             {identity:0.62, coherence:0.41, ...}
  EmotionalState.overall_valence     0.42
  RelationalMemory.trust             0.833 (current user)
  GoalEngine.active goals            [{priority:0.83, energy:0.51, ...}]
  GlobalWorkspace.recent(10)         [WorkspaceItem, ...]
  CognitiveFluxEngine.last_flux      {modules: {semantic_memory:0.72, ...}}
        ↓
CognitiveAttentionEngine.compute()
        ↓
  Step 1: Build state vector for each cognitive module
          (drives + tensions + emotional + relational relevant to that module)
  Step 2: Compute pairwise relevance using embedding cosine similarity
          (semantic_memory embedding vs current goal topic embedding)
  Step 3: Apply softmax → attention_weights summing to 1.0
  Step 4: Derive focus_priority scaling factors from weights
        ↓
OUTPUTS:
  attention_weights: {semantic_memory:0.82, relational_memory:0.88, ...}
  primary_focus: "relational_memory"
  secondary_focus: ["semantic_memory", "narrative_identity"]
  Written to: data/persona/cognitive_attention.json (every consolidation)
  Broadcast to: GlobalWorkspace (high priority)
  Fed to: workspace_competition.compete() as attention_bias
          MotivationalField._integrate_drives() replaces hard thresholds
          CuriosityEngine.decay_all() uses attention weight instead of fixed floor
```

---

## GAPS TO IMPLEMENT (6 items, ranked)

### GAP 1 — CognitiveAttentionEngine (new file) [CRITICAL]
**What:** `cognition/cognitive_attention_engine.py`
**Replaces:** Static flux weights table in `cognitive_flux_engine.py`
**Adds:** Softmax normalization, pairwise module relevance, primary/secondary focus

The engine has three stages:

**Stage 1 — State embedding.** Each module gets a context vector assembled
from the drives/tensions/emotional/relational values most relevant to it.
Uses the existing mapping from Phase 2.8 `MODULE_INPUTS` as the basis.

**Stage 2 — Relevance scoring.** For each module, compute relevance to the
current workspace winner's topic using `embedding_model.encode()` (already
loaded). A module discussing "relational depth" scores higher when the
current workspace winner is a relational thought. This is the key improvement
over Phase 2.8 — flux intensity was purely drive-based, not topic-sensitive.

**Stage 3 — Softmax allocation.** Apply softmax to raw scores. The zero-sum
property enforces the cognitive resource constraint the proposal identified:
a 0.88 weight on relational_memory arithmetically reduces what's available
to goal_ecology. No explicit suppression logic needed — it emerges from the
math.

**Output schema:**
```json
{
  "attention_weights": {
    "semantic_memory": 0.12,
    "thought_threads": 0.09,
    "narrative_identity": 0.11,
    "creative_divergence": 0.05,
    "goal_ecology": 0.08,
    "relational_memory": 0.18,
    "self_model": 0.10,
    "curiosity_engine": 0.11,
    "emotional_state": 0.07,
    "decision_policy": 0.09
  },
  "primary_focus": "relational_memory",
  "secondary_focus": ["semantic_memory", "narrative_identity"],
  "raw_scores": {...},
  "computed_at": 1234567890.0,
  "slow_cycle": 42
}
```

### GAP 2 — workspace_competition.compete() attention bias [HIGH]
**File:** `cognition/workspace_competition.py`
**What:** Feed attention_weights into candidate scoring so modules with higher
attention weight have their candidates boosted proportionally.

Current: `raw_score = base_score + pressure_boost + urgency * 0.2`
New: `raw_score = base_score + pressure_boost + urgency * 0.2 + attention_bias`

where `attention_bias = attention_weights.get(candidate_source_module, 0.5) * 0.3`

This is additive (preserves existing scoring) but makes the zero-sum
constraint operative: if relational_memory has 0.88 weight, its candidates
get +0.264 boost; goal_ecology at 0.08 gets +0.024 — a 10x difference in
the same competition cycle.

### GAP 3 — MotivationalField threshold replacement [HIGH]
**File:** `cognition/motivational_field.py`
**What:** Replace 5 hard threshold gates with continuous attention-weighted scaling.

Current:
```python
if curiosity_active and goal.origin == "explore" and energy > DORMANCY_ENERGY:
    goal.energy = min(1.0, goal.energy + CURIOSITY_BOOST)
```

New:
```python
curiosity_weight = attention_weights.get("curiosity_engine", 0.5)
boost = CURIOSITY_BOOST * (curiosity_weight / 0.10)  # scale by attention share
goal.energy = min(1.0, goal.energy + boost * (energy / DORMANCY_ENERGY))
```

The boost is now proportional to how much attention the curiosity module
is currently receiving — no hard binary gate.

### GAP 4 — CuriosityEngine attention-weighted floor [MEDIUM]
**File:** `cognition/curiosity_engine.py`
**What:** Replace fixed decay floor (0.05) with attention-weighted floor.
When curiosity_engine has high attention weight, topics decay more slowly
(floor rises to 0.12). When it has low weight, topics decay faster (floor
stays at 0.02). This means Lumina's curiosity is genuinely concentrated
when the cognitive system is focused on curiosity, and allowed to settle
when focus is elsewhere.

### GAP 5 — Prompt budget attention injection [MEDIUM]
**File:** `cognition/cognitive_organism.py`
**What:** Inject `primary_focus` and `secondary_focus` into prompt budget
so the LLM knows where cognitive attention is allocated at generation time.

Current prompt budget includes: workspace winner, motivational state,
aspiration tensions. Adding: attention focus (1-2 sentences max).

Example injection: *"Current cognitive focus: relational_memory (primary),
semantic_memory (secondary). Identity stress is high; relational depth is
the primary processing priority this cycle."*

### GAP 6 — Internal loop wiring + cognitive_attention.json [LOW]
**File:** `core/internal_loop.py`
**What:** Instantiate CognitiveAttentionEngine lazily (like Phase 2.8 flux engine),
call `compute()` on consolidation cycle (every 5 slow = 10 min), write
`cognitive_attention.json`.

---

## IMPLEMENTATION PLAN

### Files modified: 4
### Files created: 2

| # | File | Type | Change |
|---|---|---|---|
| 1 | `cognition/cognitive_attention_engine.py` | NEW | Full engine with 3-stage computation |
| 2 | `data/persona/cognitive_attention.json` | NEW | Initial state file |
| 3 | `cognition/workspace_competition.py` | MODIFY | Add attention_bias to scoring |
| 4 | `cognition/motivational_field.py` | MODIFY | Replace 3 threshold gates |
| 5 | `cognition/curiosity_engine.py` | MODIFY | Attention-weighted decay floor |
| 6 | `core/internal_loop.py` | MODIFY | Lazy-init + consolidation wiring |
| 7 | `cognition/cognitive_organism.py` | MODIFY | Prompt budget injection |

---

## IMPLEMENTATION APPROACH

### What uses existing infrastructure (no new dependencies)
- `embedding_model` (all-MiniLM-L6-v2, already loaded for FAISS)
- `numpy` (already used in FAISS and embedding operations)
- `cognitive_flux_engine.py` MODULE_INPUTS (reused as attention prior)
- `GlobalWorkspace.recent()` (existing API)
- `RelationalMemory.get_or_create()` (existing API)

### What is explicitly NOT done
- No W_q/W_k/W_v random projection matrices (no training = noise)
- No changes to FAISS search (proposal's "dynamic pruning" claim is unsound)
- No replacement of workspace_competition (attention is additive bias, not replacement)
- No changes to safety constraints (cross_layer_feedback.py thresholds stay hard)
- No changes to CognitiveGovernor (structural safety thresholds stay hard)

### Safety boundary
The two modules with all-safety thresholds (`cross_layer_feedback.py`,
`cognitive_governor.py`) are explicitly excluded from attention replacement.
Safety and structural boundaries remain hard thresholds by design.
Attention allocation applies only to cognitive content selection, not
safety enforcement.

---

## EXPECTED BEHAVIOURAL CHANGES

**Before (Phase 2.8):**
- All 10 modules have independent flux intensities summing to ~4.14 (arbitrary)
- Workspace competition: goal scores 0.83, thought scores 0.62, tension scores 0.71 — all
  can be high simultaneously, winner is the highest single value
- Curiosity boost fires if curiosity > 0.5 (binary gate)

**After (Phase 2.9):**
- All 10 modules have attention weights summing exactly to 1.0
- When relational_memory has 0.88 weight, cognitive system is mathematically
  constrained to allocate less to goal_ecology (e.g. 0.05)
- Workspace candidates from high-attention modules get measurably stronger scores
- Curiosity boost scales continuously with curiosity attention weight
- Dashboard shows: primary_focus, secondary_focus, attention_weights heatmap

**What this enables:**
- Coherent cognitive sessions: when relational trust is high, relational
  content dominates the workspace for multiple cycles, producing thematically
  consistent responses
- Graceful focus transitions: attention shifts gradually via softmax EMA,
  not via discrete threshold crossings
- Diagnosable cognitive state: `cognitive_attention.json` shows exactly where
  the system's attention is allocated and why

**What this does not enable:**
- Consciousness (no change to the fundamental LLM generation process)
- Better language generation (LLM capability is unchanged)
- Self-modification of attention weights (weights are computed, not learned)

---

## CRITICAL DESIGN DECISION: NO LEARNED WEIGHTS

The proposal's `W_q`, `W_k`, `W_v` matrices require a training signal to be
meaningful. Without one, they are random projections — worse than the current
hand-engineered scoring because at least the current scoring reflects domain
knowledge about what matters cognitively.

Phase 2.9 uses **handcrafted relevance priors** (from MODULE_INPUTS in
cognitive_flux_engine.py, validated against actual codebase behaviour) combined
with **live semantic similarity** (embedding cosine to current workspace topic).
This is more honest than claiming neural self-organization from untrained matrices.

If in a future phase there is a training signal (e.g. user feedback on response
quality, or outcome scores from PCM), the attention weights could be learned.
Phase 2.9 establishes the infrastructure for that — a computed attention vector
that influences behaviour — without overclaiming what the computation does.

