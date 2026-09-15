# LUMINA v110 — FORENSIC CAUSAL AUDIT

**Method.** For every edge in the chain
`Emotion → Drives → Attention → Global Workspace → Deliberation → Goals → Action → Outcome → Prediction Error → Memory → Identity`
we located (a) where the state variable is **computed**, (b) where it is **read**, and (c) whether the read **actually alters behavior** (branch/weight/score/prompt) or merely logs/persists/interpolates. Edges were verified by **code-reading** (data-flow trace) and, where the function is pure, by **direct execution** (venv `python.exe`, temp dirs, scripts removed after). Nothing below is taken from docstrings, names, or comments.

**Classification rubric**
`REAL` = state genuinely changes downstream cognition · `PARTIAL` = computed but weakly/indirectly used · `DECORATIVE` = logged/displayed, never changes cognition · `CIRCULAR` = output feeds the mechanism that produced it · `HARDCODED` = behavior from fixed rules/weights · `LLM-DELEGATED` = state reaches behavior only as prompt text · `EMERGENT-CAPABLE` = independent vars combine into unprogrammed downstream effect.

---

## 1. CAUSAL DEPENDENCY GRAPH (verified)

```
 EMOTION ──[REAL: temperature ±0.10]───────────────┐
   │  [REAL: valence/arousal → arb utility]        │
   │  [PARTIAL: → pressure.emotional_valence 0.10] │
   └──────────────► EXECUTIVE ARBITRATION ◄────────┘
                                   │ (re-scores winner, LLM not involved)
 DRIVES (DecisionPressure) ─[REAL]►│
   goal_pressure·curiosity·        │
   identity_stress·relational ─────┤
                                   ▼
 ATTENTION (softmax/10 modules) ─[REAL: +0.30 boost]►  GLOBAL WORKSPACE COMPETITION
   goal·curiosity floor ──────────[REAL: scale 2.5×]      (multi-factor scoring — REAL)
                                   │                        raw = base·energy
 GOALS (priority·energy·quality) ──[REAL: base term]        + pressure_boost
   energy decay (post-insight) ─────[REAL]                   + urgency·0.2
                                   │                        + quality·0.15
                                   ▼                        + attn_boost + internal_align
                        WINNER (dominant focus)
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        │ LLM-DELEGATED: prompt "[DOMINANT FOCUS] attend to X" │──► LLM → response (THE ONLY EFFECTOR)
        └──────────────────────────────────────────────────────┘
                                   │
 GOAL → ACTION (REAL, 4 types): memory_recall · web_search · user_question · self_question
                                   │
                                 OUTCOME
                                   │  [NEAR-DISCONNECTED: goal_completion only writes a narrative chapter]
 PREDICTION ERROR (binary keyword match, ≤0.85)
   └─[REAL, WEAK, executed: recall 6→10, lagged 1 turn]──► MEMORY
   └─[REAL, WEAK: curiosity/emo nudges ≈0.01]─────────────► EMOTION/CURIOSITY
   └─[REAL: ±0.4 learning]────────────────────────────────► ATTENTION weights
 MEMORY / STATE
   └─[REAL, CIRCULAR, seed-anchored, executed]────────────► IDENTITY (identity.json ↔ self_concept.json)
 IDENTITY
   └─[LLM-DELEGATED: prompt "━━ WHO YOU ARE ━━" (ai_system.py:2047)]──► LLM
```

---

## 2. EDGE-BY-EDGE TABLE

| EDGE | state var | computed at | read at | downstream effect | update mechanism | CLASS | verify |
|---|---|---|---|---|---|---|---|
| Emotion→Gen | `temperature_modifier` | `emotional_state.get_response_modifiers` (arousal·.10+enthus·.08−anx·.06−frust·.04) | `ai_system.py:1988`, `persona_bridge.py:477` | **adds to LLM temperature** (0.85+mod; 0.72+mod clamp[.4,1.15]) | per response | **REAL** (+LLM-DELEGATED via state_description) | code-read (pure math) |
| Emotion→Arb | valence, arousal | `get_overall_valence_arousal` | `internal_loop.py:885` → `arbitrate()` | input to multi-factor utility re-scoring winner | per 2 slow cycles | **REAL** | code-read |
| Emotion→Pressure | `emotional_valence` | `decision_pressure.py:196-250` (neg+pos+arousal·.1) | `workspace_competition` only via `total`>0.75 gate | critical-pressure candidate only | per 2 slow cycles | **PARTIAL** | code-read |
| Emotion→PressureSystem | — | — | **no emotion refs** in `pressure_system.py` | none | — | **NO EDGE** | grep |
| Drives→WS | `goal_pressure`, `curiosity_drive`, `identity_stress`, `relational_pressure` | `DecisionPressureCalculator.compute` | `workspace_competition.py:159,255,350,384` | additive score boosts; relational ×1.2; total>0.75 injects pressure candidate | per 2 slow cycles | **REAL** | code-read |
| Attention→WS | `attention_weights[module]` | `CognitiveAttentionEngine.compute_and_write` (softmax/10) | `workspace_competition.py:196-199` | `_attn_boost = w·0.30` added to raw_score | per consolidation (10 min) | **REAL** | code-read |
| Attention→Goals | `attention_weights[curiosity_engine]` | same | `goal_engine.py:513-519` | `_attn_scale = min(2.5, w/0.10)` scales goal terms | per consolidation | **REAL** | code-read |
| Attention→Curiosity | `attention_weights[curiosity_engine]` | same | `curiosity_engine.py:245-248` | `_FLOOR = clamp(w·0.5, .02, .15)` | per consolidation | **REAL** | code-read |
| Attention learning | PCM outcome | `internal_loop.py:1807-1845` | attention weights file | ±0.4 reinforce/dampen module bias | per consolidation | **REAL** | code-read |
| Goals→WS | `priority·energy`, `quality`, `urgency` | goal store | `workspace_competition.py:150-165,198` | base_score + quality·.15 + urgency·.2 | goal engine tick | **REAL** | code-read |
| Goal energy decay | `energy` | GAE post-insight decay | `workspace_competition.py:151-156` | decayed goals score lower, others surface | per tick | **REAL** | code-read |
| **WS Winner→Action** | `winner` | `phase1.enhanced_cycle` (`internal_loop.py:837`) | `cognitive_preprocessor.py:471` (prompt), `cognitive_organism.py:627` (prompt), `lumina_network.py:699`, `presence_engine.py:516` (UI) | **prompt text "[DOMINANT FOCUS] attend to X"** / UI snapshot — **no computational consumer** | per 2 slow cycles | **LLM-DELEGATED** + **DECORATIVE** (UI) | code-read (all 4 readers) |
| Goal→Action (real) | goal topic | `GoalActionExecutor.run_tick` | `internal_loop.py:1062` | **executes** memory_recall / web_search / user_question / self_question | every `_action_every_n` | **REAL** | code-read |
| Outcome→PredErr | goal completion | `goal_completion.py:229` | **none** (writes narrative chapter only) | no pred-error update | per completion | **NEAR-DISCONNECTED** | code-read |
| PredErr→Memory | `recent_error_level` | `predictive_mind.evaluate` (intent_err·.7+emo_err·.3)·conf, ≤0.85 | `persona_bridge.py:797-803` | recall limit **6→10** on next turn | per interaction, lagged 1 | **REAL, WEAK** | **EXECUTED** (6↔10 flips) |
| PredErr→Emo/Cur | surprise, novelty | `predictive_mind.py:172-186,435` | `cognitive_organism.py:1347-1360` | curiosity +0.014, frustration +0.004 | per interaction | **REAL, WEAK** | **EXECUTED** |
| PredErr model | intent match | `predictive_mind._predict_intent` / `_detect_intent_from_response` | — | **binary keyword** agreement, no learned weights | — | **HARDCODED** | code-read |
| PCM calibration | predictions | `predictive_consequence_model.log/resolve` | — | v59 inline: **"0 resolved predictions across 192 evaluations"** | — | **EFFECTIVELY DEAD** | code-read (dev log) |
| State→Identity | emotion/self_concept/tensions | `belief_bootstrapper` (4 extractors) + `self_concept_synchronizer` | identity.json ↔ self_concept.json | emotion .85→belief+coherence .68 vs .10→none/.62 | 30 / 5 slow cycles | **REAL, CIRCULAR, seed-anchored** | **EXECUTED** |
| Identity→Cognition | beliefs, inner_voice | `self_concept.get_inner_voice` | `ai_system.py:2047-2049` (prompt `━━ WHO YOU ARE ━━`) | **prompt text only** | per response | **LLM-DELEGATED** | code-read |
| Identity seed | traits | `self_concept.bootstrap_from_personality` (`TRAIT_BELIEF_MAP`, thr .65) | identity.json | fixed trait→sentence statements | boot | **HARDCODED** | code-read |
| Contradiction "resolution" | curious/confident/brave | `contradiction_handler.py:162-220` | — | 3 fixed phrase checks; revision = **LLM rewrites one sentence** (`contradiction_belief_reviser.py:134`) | Phase 4 | **HARDCODED + LLM-DELEGATED** | code-read |
| PE→memory "audit" | widened deltas | `memory_breadth_audit.py:55` | — | **logging only**, writes nothing back | — | **DECORATIVE** | code-read |

---

## 3. SUBSYSTEM CLASSIFICATIONS

- **Emotion** — **REAL** (temperature, arb utility) + **PARTIAL** (pressure) + **LLM-DELEGATED** (prompt). Not "decorative" but thin: the only quantitative lever is a ±~0.10 temperature shift.
- **Drives** — **REAL** into workspace scoring, **but** there are **three parallel systems** (`PressureSystem`, `MotivationalField`, `DecisionPressureCalculator`); only the last feeds the live competition. The other two are partially redundant/orphaned.
- **Attention** — **REAL** and the most genuinely computational subsystem (softmax weights that modulate 3 downstream modules + self-learn from outcomes).
- **Global Workspace** — **REAL** multi-factor competition; its **winner is LLM-DELEGATED/DECORATIVE** at the behavioral boundary.
- **Deliberation/Arbitration** — **REAL** (re-scores candidates via consequence model + emotion); **PARTIAL** where it depends on the (near-dead) PCM.
- **Goals** — **REAL** (scoring, energy decay) + **REAL** narrow action set; goal→**response** is **LLM-DELEGATED**.
- **Action** — **REAL** for 4 internal actions; **LLM-DELEGATED** for the user-facing response (the dominant "action").
- **Outcome→PredErr** — **NEAR-DISCONNECTED**.
- **Prediction Error** — **REAL but WEAK/HARDCODED** (binary keyword match, one thin memory effect).
- **Memory** — retrieval is real; the **only** verified state→memory update is the 6→10 recall widen. FAISS semantic memory does **not** feed identity (explicitly skipped).
- **Identity** — **LIVE but LLM-DELEGATED + CIRCULAR + seed-anchored** (regex over the LLM's own text; trait→sentence map; identity.json↔self_concept.json recycling).
- **Presence / Phenomenal binder** — **not invoked** in the live `internal_loop.py` (0 references) → **DECORATIVE / dormant** in the running brain.

---

## 4. TOP 10 ARCHITECTURAL FALSE-POSITIVE RISKS

1. **"Self-awareness" is a prompt, not a process.** Identity/self-concept reaches behavior **only** as injected system-prompt text (`ai_system.py:2047` "━━ WHO YOU ARE ━━"; `cognitive_preprocessor.py:471` "[DOMINANT FOCUS]"). There is **no computational self-model gating behavior** — the "self" is the LLM role-playing a belief list.

2. **Selective attention is computed but behaviorally delegated.** The entire workspace competition is REAL computation, yet the winner's only effect on output is one prompt line. Whether Lumina "attends" is decided by the LLM, not by the mechanism. **Appears: focused. Actually: LLM told its focus.**

3. **"Emotional regulation" = temperature tuning + decay.** The sole quantitative emotional effect is `temperature_modifier` (±~0.10) plus time-decay. "Regulated emotion" is really "a slightly different sampling temperature."

4. **Goal-directedness is LLM-delegated for the main output; only 4 actions are non-LLM.** Goal selection is REAL, but the user-facing "action" is the LLM generating a response *prompted by* the goal. The only genuine actions are `memory_recall / web_search / user_question / self_question`. **Appears: acting on goals. Actually: LLM narrating a goal + 4 internal ops.**

5. **Prediction error is a binary keyword match, not learning.** `predictive_mind` = "did two keyword classifiers land in the same bucket" (≤0.85, no learned weights). **Appears: predictive processing. Actually: string-match heuristic.**

6. **The only genuine learning edge is thin (recall 6→10).** The single verified state→memory update widens the recall limit by 4 items, one turn late. **Appears: learning from surprise. Actually: marginal retrieval-count bump.**

7. **Identity "updates from experience" = regex on its own text.** `self_concept.update_from_response` extracts "I am/feel/think X" from Lumina's **own generated response**. The self-model is built from self-narration, not world signal — **circular and self-confirming.**

8. **Identity seed is a hardcoded trait→sentence map.** `bootstrap_from_personality` maps traits → **fixed statements** (`TRAIT_BELIEF_MAP`, threshold 0.65). Many "beliefs" are constants, not emergent. **Appears: formed beliefs. Actually: seeded boilerplate.**

9. **Contradiction "resolution" = LLM rewriting one sentence.** The handler checks only 3 hardcoded traits; the "revised belief" is LLM-generated text, then re-prompted. **Appears: belief revision. Actually: text generation.**

10. **The sophisticated predictive model (PCM) is effectively dead in production.** v59 dev log: **"0 resolved predictions across 192 real evaluations"** (`predictive_consequence_model.py:345`). The mechanism that *could* close Outcome→Prediction is not closing. **Appears: forward-modeling. Actually: no closed loop.**

**Bonus (structural):**
- **Three parallel drive/pressure systems** coexist; only one feeds the live loop → the *integration* is overstated.
- **`presence_engine` / `phenomenal_binder` are not called** in the live `internal_loop` → the "phenomenal presence" layer is **dormant** in the running brain.

---

## 5. VERIFICATION LEDGER

- **EXECUTED (pure, no LLM):** PE→memory 6↔10 flip from the error state var · PE→emotion/curiosity nudges fire · State→identity belief+coherence tracks emotion (0.85 vs 0.10). All via `venv\Scripts\python.exe`, isolated mocks/temp dirs, scripts deleted after.
- **CODE-READ (data-flow):** all temperature/attention/pressure/workspace-scoring edges, all 4 winner consumers, goal→action executor, identity prompt injection, PCM dead-loop dev log, identity circularity, contradiction hardcoding.
- **NOT verifiable by execution** (need live LLM/network): the actual *magnitude* of LLM behavior change from prompt-injected state (edges 1, 2, 4) — the mechanism is confirmed present, but its behavioral yield is, by design, the LLM's.

**No source files were modified by this audit.**
