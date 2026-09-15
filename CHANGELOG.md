# Lumina — Consolidated Changelog (v67 → v80)

Every zip delivered since the Phase 4.x Global Workspace redesign was built
cumulatively on the same working copy — `lumina_v80_rss_management.zip`
already contains everything below, not just the RSS panel. This document
lists what actually changed, file by file, for merging into your real
codebase.

---

## Phase 4.x — Global Workspace redesign (v67)

- **`cognition/global_workspace.py`** — added `WorkspaceState` dataclass
  (focus, intention, active_hypotheses, uncertainty, predicted_futures,
  goals, motivations, narrative_state, executive_policy, confidence,
  context, provenance) plus `get_state()`, `update_state()`,
  `set_hypotheses()` (TTL-bounded), `prune_expired_hypotheses()`,
  `edit_executive_policy()`, `state_summary()`. Additive — existing
  blackboard API (`broadcast()`/`recent()`/`top()`) untouched.
- **`cognition/workspace_competition.py`** — `compete()` now publishes
  top-K ranked candidates as `active_hypotheses` instead of only the
  collapsed winner.
- **`cognition/executive_arbitration.py`** — replaced single-signal
  margin check with a multi-factor utility function (activation,
  predicted, counterfactual, identity, goal_alignment, calibration).
- **`cognition/workspace_state_sync.py`** *(new)* — centralized adapter
  syncing world_model/counterfactual_simulator/calibration_engine/
  introspective_observer/narrative_identity/goal_engine into
  `WorkspaceState` without modifying those six files.
- **`core/phase1_integration.py`**, **`core/internal_loop.py`** — thread
  `GlobalWorkspace` through; store full state (`_v32_workspace_state`).
- **`cognition/cognitive_organism.py`** — live `respond()` reads the
  async-populated snapshot (focus/hypotheses/executive_policy) into the
  prompt.
- **`pages/cognitive_dashboard_page.py`** — added Panel 16 (Global
  Workspace).

## Bug audits (v68–v71)

- **`cognition/goal_engine.py`** — fixed `passs` typo (was raising
  `NameError` in an except handler).
- **`cognition/semantic_extractor.py`** — fixed `_update_capabilities()`
  referencing undefined `result`/`user_input`; threaded `user_input`
  through from the caller.
- **`cognition/persona_bridge.py`** — added missing `datetime`/`time`
  imports; the entire temporal-awareness feature (time of day, circadian
  phase, session-gap detection) had been silently dead since inception.
- **`pages/main_page.py`** — fixed `_on_speech_start` callback
  indentation (was firing once at mode-toggle instead of per speech
  event) and an undefined `user_text` reference.
- **`cognition/lumina_network.py`** — fixed topic-logging fallback
  (`'current_topic' in dir()` guard always false, always logged literal
  "dialogue"); fixed `receive_from_child()` never logging inbound
  exchanges to `_history`; cleaned up a `deque` type-annotation issue.
- **`cognition/genuine_choice.py`** — completed the promised-but-missing
  textual choice-pattern analysis in `analyze_choice_patterns()`.
- **`cognition/ethical_reasoning_engine.py`** — the "standing ethical
  orientation" prompt fragment was built then permanently silenced via
  `return ""`; added a real 30-minute cooldown so it actually fires.
- **`cognition/structural_coupling.py`** — wired peer `aspiration_seeds`
  into `aspirational_self.observe_tension()` (previously dropped).
- **`cognition/cognitive_behavior_gate.py`** — the `PRIORITY` dict was
  defined and referenced once, never used; added as a bounded monotonic
  floor bump in `_compute_p()`.
- **`cognition/introspective_observer.py`** — added the missing
  `worst_counts` observation block (exact mirror of the existing
  `best_counts` block).
- **`cognition/thread_identity_linker.py`** — added the promised-but-
  missing `thread_type` check alongside `thread_name`.
- **`cognition/narrative_arc_writer.py`** — wired `ev_ref` into the
  `belief_vs_evidence` narrative template.
- **`cognition/cognitive_dissonance_engine.py`** — added a second,
  additive detection path in `_detect_belief_vs_evidence` (checks actual
  recent evidence, not just isolated low confidence).
- **`cognition/phenomenal_binder.py`** — wired `ws_source` into the
  foreground-description blending.
- **`cognition/motivational_field.py`** — fixed `created_at` →
  `timestamp` (the real `LifeChapter` field name; always fell to the 0.0
  default before), added real elapsed-time display.
- **`cognition/ai_system.py`** — fixed `perform_reflection(old, old)`
  (same argument passed twice, so the Liberty personality-reflection
  system never recorded a single reflection since creation); fixed
  "update relational trust based on feedback" being a no-op; wired
  `resp_emo` into episodic memory text.
- **`cognition/epistemic_integrity_engine.py`** — added grounding checks
  for `emotional_ground`/`dominant_goal` in `_validate_evolution`/
  `_validate_goal_coherence` (conservative, omission-based).
- **`cognition/abstract_reasoning_engine.py`**,
  **`cognition/proactive_outreach.py`** — removed confirmed-dead
  variables after full re-verification (no behavior change).
- **Deleted** (user-confirmed unused): `cognition/cognition_engine.py`,
  `cognition/cognition_engine_v2.py`,
  `cognition/simple_thought_evaluator - Copy.py`.

## Prompt logging (v73)

- **`managers/settings_manager.py`** — added `LOG_FULL_PROMPTS: bool =
  False` config toggle.
- **`cognition/ai_system.py`** — logs the full assembled prompt at INFO
  level and writes `data/persona/last_prompt_debug.txt` when enabled.

## "Will to act" (v74–v76)

- **`pages/main_page.py`** — wired `GoalActionExecutor.get_pending_insight()`
  delivery (was queued, never consumed).
- **`brain.py`** — added, then removed in favor of the v76 fix (see
  below) a log-only surfacing of GAE content for Flux.
- **`cognition/proactive_outreach.py`** — added a 4th outreach trigger
  ("gae_action") reusing all existing safety gates (trust threshold,
  interaction minimum, cooldown, daily cap, topic-relevance matching).
- **`cognition/goal_action_executor.py`** — wired `_action_web_search()`
  into `skill_registry.seed()`; removed a duplicate `return None`.

## Real sandbox execution testing (v77)

- **`cognition/executive_arbitration.py`** — fixed `_tokens()` not
  splitting on underscores (made every `goal_ecology` topic name
  structurally unmatchable against narrative prose).
- **`cognition/cognitive_organism.py`** — fixed duplicate `[Attention]`
  label on two unrelated signals (renamed one to `[Cognitive focus]`).
- **`cognition/self_model_moment.py`** — added `_truncate_at_word()`
  helper, replacing two hard character slices that cut mid-word.
- **`cognition/proactive_outreach.py`** — **the most significant fix of
  the whole engagement**: `_evaluate()` read `rel_mem._relationships`,
  an attribute that never existed on `RelationalMemorySystem` (real name:
  `_rels`). The entire outreach engine, every trigger type, had been
  iterating over an empty dict since the method was written.

## Emotion / self-awareness (v78, v78b)

- **`cognition/executive_arbitration.py`** — added `_emotional_modulation()`
  / `_apply_emotional_modulation()`; `arbitrate()` gained optional
  `valence`/`arousal` params that bound-modulate `UTILITY_WEIGHTS`
  (max ±0.08/weight, always renormalized).
- **`core/internal_loop.py`** — wired real valence/arousal from
  `organism.ai_system.emotional_state` into the `arbitrate()` call.
- **`cognition/ai_system.py`** — `retrieve_memories()` gained an optional
  `current_valence` param for mood-congruent recall (backward compatible).
- **`cognition/persona_bridge.py`** — wired `current_valence` into the
  main conversational memory recall call.
- **`cognition/cognitive_organism.py`** — moved `flush_deltas()` to
  before the self-model fragment is read (was one turn stale).
- **`cognition/emotional_state.py`** — added `MOOD_HALFLIFE_HOURS`,
  `_mood_valence`/`_mood_arousal` (slow EMA distinct from the 6 fast
  emotions), `get_mood()`, mood persistence in `_save()`/`_load()`,
  mood-vs-moment contrast in `get_state_description()`.
- **`cognition/predictive_mind.py`** — `evaluate()` now nudges
  curiosity/frustration via `receive_self_influence()` when
  `result.surprise` is true (anxiety deliberately excluded).
- **`cognition/threat_response.py`** *(new)* — objective threat
  detection (resource depletion, unresolved dissonance, sustained
  prediction failure — never anxiety itself) feeding anxiety through the
  same bounded path, tied to existing `regulate()`.
- **`core/internal_loop.py`** — wired `apply_threat_response()` into the
  end of `_slow_cycle_impl()`.

## Memory temporal/emotional context (v79)

- **`cognition/persona_bridge.py`** — `_fmt_memory()` now includes
  `_relative_time()` (natural-language temporal reference) and emotional
  tagging when non-neutral, e.g. `(2 months ago, felt positive)`.

## RSS feed management (v80)

- **`cognition/research_mcp/schemas.py`** — added `source: str = ""` to
  `WebPage`.
- **`cognition/research_mcp/web_agent.py`** — populated the new `source`
  field on both `WebPage` construction paths.
- **`cognition/research_mcp/rss_registry.py`** *(new)* — persisted,
  taggable, activatable RSS feed registry with `success_rate` derived
  from real `ResearchSession.confidence_score`.
- **`cognition/research_mcp/search_providers.py`** — `_rss()` now reads
  from the registry (active-only, tag-ranked) instead of a hardcoded
  list; added `_guess_tag_from_query()` heuristic.
- **`cognition/research_mcp/controller.py`** — records feed outcomes
  right when a session's `confidence_score` is finalized.
- **`pages/rss_management_page.py`** *(new)* — the `/rss-feeds`
  configuration panel.
- **`pages/__init__.py`** — registered the new page.
- **`pages/main_page.py`** — added a nav button for it.

## Dashboard visibility + changelog (this round)

- **`pages/cognitive_dashboard_page.py`** — added Panel 17 (Emotion,
  Mood & Threat): instant vs. mood valence/arousal, mood/moment
  divergence, individual emotion badges, threat-level badge.
- **This document.**

---

## Files touched, complete list

```
brain.py
core/internal_loop.py
core/phase1_integration.py
cognition/abstract_reasoning_engine.py
cognition/ai_system.py
cognition/cognitive_behavior_gate.py
cognition/cognitive_dissonance_engine.py
cognition/cognitive_organism.py
cognition/emotional_state.py
cognition/epistemic_integrity_engine.py
cognition/ethical_reasoning_engine.py
cognition/executive_arbitration.py
cognition/genuine_choice.py
cognition/global_workspace.py
cognition/goal_action_executor.py
cognition/goal_engine.py
cognition/introspective_observer.py
cognition/lumina_network.py
cognition/motivational_field.py
cognition/narrative_arc_writer.py
cognition/persona_bridge.py
cognition/phenomenal_binder.py
cognition/predictive_mind.py
cognition/proactive_outreach.py
cognition/research_mcp/controller.py
cognition/research_mcp/rss_registry.py            (new)
cognition/research_mcp/schemas.py
cognition/research_mcp/search_providers.py
cognition/research_mcp/web_agent.py
cognition/self_model_moment.py
cognition/semantic_extractor.py
cognition/structural_coupling.py
cognition/thread_identity_linker.py
cognition/threat_response.py                      (new)
cognition/workspace_competition.py
cognition/workspace_state_sync.py                 (new)
managers/settings_manager.py
pages/__init__.py
pages/cognitive_dashboard_page.py
pages/main_page.py
pages/rss_management_page.py                      (new)

Deleted: cognition/cognition_engine.py, cognition/cognition_engine_v2.py,
         cognition/simple_thought_evaluator - Copy.py
```
