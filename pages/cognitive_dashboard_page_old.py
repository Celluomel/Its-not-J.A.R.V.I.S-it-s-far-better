"""
pages/cognitive_dashboard_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cognitive Health Dashboard — GET /cognitive-dashboard

A diagnostic instrument for a learning system.

Not: "what is PandoraBOX thinking right now" (that was the old dashboard)
But: "are the learning models reliable, are the feedback channels firing,
     where is the architecture healthy and where is it calibrating"

Seven panels:

1. Predictive Model Readiness
   PCM records per action type — which predictions are reliable vs building.
   WSDM coverage and relational context count.
   Shows the threshold: MIN_SIMILAR_FOR_PREDICT = 8 per type.

2. Identity Constraint Health
   Total checks, correction rate, fallback rate per value.
   High fallback rate = classifier firing but LLM ignoring the correction.
   Actionable: if honesty fallback > 40%, the correction directive needs work.

3. Cross-Layer Feedback Activity
   Which channels have fired recently and with what magnitude.
   Recovery multiplier (Ch1), active drive boosts (Ch2),
   policy deltas from constraint outcomes (Ch3), gate floor (Ch4).

4. Aspiration Pipeline
   Active aspirations by origin (synthesis/generative/tension/internal_need).
   Temporal projection preferred path and action type.
   Next synthesis and generative scan cycles.

5. Motivational State
   Live drive vector with values and trend indicators.
   Dominant drive. Context boosts from CrossLayerFeedback.

6. Resource Economy
   Current pool levels with visual bars.
   Recovery multiplier from temporal projection.
   Last N interactions cost profile.

7. PandoraBOX-Flux Network
   Exchange count, PCM records contributed by network dialogues.
   Action type distribution from network vs human interactions.
   Trust trajectory with Flux (lumina_child_*).

Auto-refreshes every 30 seconds. Can be manually refreshed.
"""

import logging
import math
import time
from typing import Any, Dict, Optional

from nicegui import ui

from core.state import state
from pages.shared import GLOBAL_CSS

# These constants are imported at module level (not inline inside _fetch_all)
# to prevent an import-lock race condition: at cycle 11 both engines lazy-init
# in the slow-cycle thread while _fetch_all() runs concurrently in to_thread().
# Python's import lock caused _fetch_all() to block waiting for the slow-cycle
# thread to finish the first import, which could stall long enough to blank
# the dashboard. Module-level imports run once at startup, eliminating the race.
try:
    from cognition.aspirational_synthesis_engine import SYNTHESIS_EVERY_N
except Exception:
    SYNTHESIS_EVERY_N = 35  # fallback matches the actual default

try:
    from cognition.generative_aspiration_engine import GENERATIVE_EVERY_N
except Exception:
    GENERATIVE_EVERY_N = 45  # fallback matches the actual default

logger = logging.getLogger(__name__)

# ── Refresh interval ──────────────────────────────────────────────────────────
REFRESH_INTERVAL = 60   # seconds — was 30; halved refresh rate to reduce
                        # WebSocket message volume and event-loop pressure
                        # from 18 panels worth of UI elements rebuilt per cycle

# ── PCM action types (same as predictive_consequence_model.py) ────────────────
ACTION_TYPES = [
    "philosophical", "technical", "emotional", "creative",
    "social", "analytical", "deep_reasoning",
]
MIN_RELIABLE = 8   # records needed for reliable prediction


# ── Data fetcher — reads all cognitive module statuses ────────────────────────

def _import_build(module_name: str, cls_name: str, *args):
    """Import a class from a module and construct it with the given args.

    Used by the v112 cold-start warm-up in _fetch_all() to build-and-adopt the
    loop's lazy engines using the exact same module/class the runtime uses.
    Returns None (never raises) if the import, the class lookup, or the
    construction fails — a failed warm-up must never take the dashboard down.
    """
    try:
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        return cls(*args)
    except Exception:
        return None


def _fetch_all() -> Dict[str, Any]:
    """Collect status from all cognitive modules. Non-fatal."""
    result: Dict[str, Any] = {}
    try:
        org  = getattr(getattr(state, 'persona', None), '_organism', None)
        # FIX (v111): PersonaBridge stores EnhancedAISystem as `self._system` —
        # `persona.ai_system` never existed, so `ai` was always None and every
        # ai-backed tile (emotion_mood, flux_relations, ice fallback) was dead,
        # showing "No data yet" even though the live system had emotional
        # state, relational memory, etc. The organism holds the canonical
        # reference (CognitiveOrganism.ai_system) — use it as primary source,
        # with the legacy names as fallbacks.
        ai   = getattr(org, 'ai_system', None) if org else None
        if ai is None:
            ai = getattr(getattr(state, 'persona', None), 'ai_system', None)
        if ai is None:
            ai = getattr(getattr(state, 'persona', None), '_system', None)
        loop = getattr(org, '_loop', None) if org else None

        if not loop:
            return result

        # ── FIX (v112 cold-start): warm the lazy engines the loop owns ────────
        # The loop builds these on its FIRST slow cycle (~2 min after boot) or
        # on the first LLM turn, and each one only loads its persisted state
        # at construction. Until then the corresponding tiles rendered empty /
        # "engine has not run" even though the data already existed on disk.
        # We build-and-adopt them here — using the SAME constructor + args the
        # runtime path uses — so the persisted state is loaded and displayed
        # from the very first dashboard render. The runtime's
        # `if self._x is None` guards then reuse this exact instance, so there
        # is exactly ONE engine object per module (no divergent in-memory
        # copies, no double state). Each constructor is a light disk load;
        # none performs LLM work at init. All non-fatal.
        _ai_sys = getattr(org, 'ai_system', None)

        def _adopt(attr, builder):
            """If loop.<attr> is missing, build it (light disk load) and adopt
            it back onto the loop so the runtime reuses this instance."""
            if getattr(loop, attr, None) is not None:
                return
            try:
                inst = builder()
            except Exception:
                return
            if inst is None:
                return
            try:
                setattr(loop, attr, inst)
            except Exception:
                pass

        _adopt('_consequence_model',
               lambda: _import_build('cognition.predictive_consequence_model',
                                     'PredictiveConsequenceModel', org))
        if _ai_sys:
            _adopt('_world_self_dynamics',
                   lambda: _import_build('cognition.world_self_dynamics_model',
                                         'WorldSelfDynamicsModel', org, _ai_sys))
            _adopt('_motivational_field',
                   lambda: _import_build('cognition.motivational_field',
                                         'MotivationalField', org, _ai_sys))
            _adopt('_cross_layer_feedback',
                   lambda: _import_build('cognition.cross_layer_feedback',
                                         'CrossLayerFeedback', org, _ai_sys))
            _adopt('_temporal_projection',
                   lambda: _import_build('cognition.temporal_projection',
                                         'TemporalProjection', org, _ai_sys))
        _adopt('_resource_economy',
               lambda: _import_build('cognition.cognitive_resource_economy',
                                     'CognitiveResourceEconomy'))

        # Phase 4.x: Global Workspace state
        gw = getattr(org, 'workspace', None)
        if gw and hasattr(gw, 'state_summary'):
            result['global_workspace'] = gw.state_summary()

        # v78/v78b: Emotion — instant vs. mood, and threat-response level
        try:
            emo = getattr(ai, 'emotional_state', None)
            if emo and hasattr(emo, 'get_overall_valence_arousal'):
                instant_v, instant_a = emo.get_overall_valence_arousal()
                mood_v, mood_a = emo.get_mood() if hasattr(emo, 'get_mood') else (None, None)
                result['emotion_mood'] = {
                    'instant_valence': instant_v, 'instant_arousal': instant_a,
                    'mood_valence': mood_v, 'mood_arousal': mood_a,
                    'description': emo.get_state_description() if hasattr(emo, 'get_state_description') else '',
                    'emotions': {
                        name: round(e.value, 3)
                        for name, e in getattr(emo, 'emotions', {}).items()
                    },
                }
        except Exception:
            pass

        try:
            from cognition.threat_response import compute_threat_level
            result['threat_level'] = compute_threat_level(org)
        except Exception:
            pass

        # PCM
        pcm = getattr(loop, '_consequence_model', None)
        if pcm:
            result['pcm'] = pcm.summary()

        # WSDM
        wsdm = getattr(loop, '_world_self_dynamics', None)
        if wsdm:
            result['wsdm'] = wsdm.status()
            # Cap scan to last 200 records — unbounded scan of a growing
            # collection (400+ and increasing) was O(n) and growing every
            # cycle, lengthening _fetch_all()'s blocking time on each refresh.
            _recent_wsdm = list(wsdm._records)[-200:]
            net_records = [
                r for r in _recent_wsdm
                if 'lumina_child' in r.user_id
            ]
            result['wsdm_network_records'] = len(net_records)

        # Identity constraint
        # FIX (v112 cold-start): the engine is lazy-instantiated on the FIRST
        # LLM response (cognitive_organism._call_ai_system) and only then
        # loads its persisted log. Until that fires the tile showed
        # "No evaluations yet — engine has not run" even though a full log
        # (data/persona/identity_constraint_log.json) existed on disk.
        # Build-and-adopt it here (same ctor + args as the runtime path), so
        # the persisted state is loaded and displayed from the first render.
        # The runtime's `if _ice is None` check then reuses this instance.
        ice = getattr(org, '_identity_constraint', None)
        if ice is None and ai:
            try:
                from cognition.identity_constraint import IdentityConstraintEngine
                _dp = getattr(ai, '_decision_policy', None)
                if _dp:
                    ice = IdentityConstraintEngine(ai, _dp)
                    org._identity_constraint = ice
            except Exception:
                ice = None
        if ice is None and ai:
            ice = getattr(ai, '_identity_constraint', None)
        if ice:
            result['ice'] = ice.summary()
            # Per-value breakdown
            result['ice_by_value'] = {}
            for check in ice._log[-100:]:
                v = check.get('value', 'unknown')
                o = check.get('outcome', 'passed')
                if v not in result['ice_by_value']:
                    result['ice_by_value'][v] = {
                        'checks': 0, 'corrected': 0, 'fallback': 0
                    }
                result['ice_by_value'][v]['checks'] += 1
                if o == 'corrected':
                    result['ice_by_value'][v]['corrected'] += 1
                elif o == 'fallback':
                    result['ice_by_value'][v]['fallback'] += 1

        # CrossLayerFeedback
        clf = getattr(loop, '_cross_layer_feedback', None)
        if clf:
            result['clf'] = clf.status()

        # Executive arbitration (Phase 4.1/4.2) — counters set in
        # core/internal_loop.py's _slow_cycle_impl on each workspace-winner
        # cycle. Lazily created on first cycle, so absent until then.
        result['arbitration'] = {
            'runs':      getattr(loop, '_arbitration_runs', 0),
            'overrides': getattr(loop, '_arbitration_overrides', 0),
            'skipped':   getattr(loop, '_arbitration_skipped', 0),
        }

        # MotivationalField
        mf = getattr(loop, '_motivational_field', None)
        if mf:
            result['mf'] = mf.status()

        # ResourceEconomy
        ec = getattr(loop, '_resource_economy', None)
        if ec:
            result['ec'] = ec.status()
            result['ec_multiplier'] = getattr(ec, '_recovery_multiplier', 1.0)

        # BehaviorGate
        gate = getattr(org, 'behavior_gate', None)
        if gate:
            result['gate'] = gate.status()

        # TemporalProjection
        tp = getattr(loop, '_temporal_projection', None)
        if tp:
            result['tp'] = tp.status()

        # Aspiration pipeline
        asp = getattr(org, 'aspirational_self', None)
        if asp:
            aspirations = asp.aspirations
            by_origin: Dict[str, int] = {}
            for a in aspirations.values():
                o = getattr(a, 'origin_story', '')
                ol = o.lower()
                if 'generative' in ol:
                    by_origin['generative'] = by_origin.get('generative', 0) + 1
                elif 'synthes' in ol:   # matches "Synthesised" and legacy "synthesis"
                    by_origin['synthesis'] = by_origin.get('synthesis', 0) + 1
                elif 'internal_need' in str(a):
                    by_origin['internal_need'] = by_origin.get('internal_need', 0) + 1
                else:
                    by_origin['tension'] = by_origin.get('tension', 0) + 1
            result['aspirations'] = {
                'total': len(aspirations),
                'by_origin': by_origin,
            }

        # Synthesis/Generative scan cycles
        # Fix: previously only computed when the engine object already existed.
        # Both engines lazy-init at _slow_cycle_count > 10 / > 15 — before that
        # this showed "in ? cycles" with zero information. Now always computed.
        ase = getattr(loop, '_aspiration_synthesis', None)
        gae = getattr(loop, '_generative_aspiration', None)
        cur = getattr(loop, '_slow_cycle_count', 0)

        ASE_UNLOCK_CYCLE = 10
        GAE_UNLOCK_CYCLE = 15

        if ase:
            result['next_synthesis_cycle'] = SYNTHESIS_EVERY_N - (cur % SYNTHESIS_EVERY_N)
        elif cur <= ASE_UNLOCK_CYCLE:
            result['next_synthesis_cycle'] = (ASE_UNLOCK_CYCLE - cur) + 1
            result['synthesis_locked'] = True

        if gae:
            result['next_generative_cycle'] = GENERATIVE_EVERY_N - (cur % GENERATIVE_EVERY_N)
        elif cur <= GAE_UNLOCK_CYCLE:
            result['next_generative_cycle'] = (GAE_UNLOCK_CYCLE - cur) + 1
            result['generative_locked'] = True

        # Network: Flux relational record
        rm = getattr(ai, 'relational_memory', None) if ai else None
        if rm and hasattr(rm, '_rels'):
            flux_rels = {
                uid: rel for uid, rel in rm._rels.items()
                if 'lumina_child' in uid
            }
            result['flux_relations'] = [
                {
                    'uid': uid,
                    'trust': round(getattr(rel, 'trust_score', 0.5), 3),
                    'interactions': getattr(rel, 'interaction_count', 0),
                    'topics': len(getattr(rel, 'shared_topics', {})),
                }
                for uid, rel in flux_rels.items()
            ]

        # FluxMindModel
        fmm = getattr(loop, '_flux_mind_model', None)
        if fmm:
            result['flux_mind'] = fmm.status()

        # CausalMechanismModel
        cm = getattr(loop, '_causal_mechanism', None)
        if cm:
            result['causal_mechanism'] = cm.status()

        # LongHorizonPlanner
        lhp = getattr(loop, '_long_horizon_planner', None)
        if lhp:
            result['long_horizon'] = lhp.status()

        # NarrativeCompression
        nc = getattr(loop, '_narrative_compression', None)
        if nc:
            result['narrative_compression'] = nc.status()

        # MetaLearningAudit
        mla = getattr(loop, '_meta_learning_audit', None)
        if mla:
            result['meta_learning'] = mla.status()

        # StructuralCoupling
        sc = getattr(loop, '_structural_coupling', None)
        if sc:
            result['structural_coupling'] = sc.status()

        # Phase 3.3: CounterfactualSimulator
        cfs = getattr(loop, '_counterfactual_sim', None)
        if cfs:
            result['counterfactual'] = cfs.status()

        # Phase 3.5: IntrospectiveObserver
        io = getattr(loop, '_introspective_observer', None)
        if io:
            result['introspective'] = io.status()

        # TemporalSelfProjection
        tsp = getattr(loop, '_temporal_self_projection', None)
        if tsp:
            result['temporal_self'] = tsp.status()

        result['slow_cycle'] = cur
        result['fetched_at'] = time.time()

        # Phase 2.9 / 2.9.1: cognitive attention weights + learned bias
        try:
            import json as _jda
            from pathlib import Path as _Pda
            _cap = _Pda("data/persona/cognitive_attention.json")
            if _cap.exists():
                _cad = _jda.loads(_cap.read_text())
                result['attention'] = {
                    'weights':         _cad.get('attention_weights', {}),
                    'primary':         _cad.get('primary_focus', ''),
                    'secondary':       _cad.get('secondary_focus', []),
                    'learned_bias':    _cad.get('learned_bias', {}),
                    'workspace_topic': _cad.get('workspace_topic', ''),
                    'slow_cycle':      _cad.get('slow_cycle', 0),
                    'context_key':     _cad.get('context_key', ''),
                    'context_dict':    _cad.get('context_dict', {}),
                }
        except Exception:
            pass

        # Phase 3.0: top learned contexts
        try:
            import json as _jctx
            from pathlib import Path as _Pctx
            _ctxp = _Pctx("data/persona/contextual_attention.json")
            if _ctxp.exists():
                _ctxd = _jctx.loads(_ctxp.read_text())
                _contexts = _ctxd.get("contexts", {})
                _sorted_ctx = sorted(
                    _contexts.items(),
                    key=lambda x: -x[1].get("total_feedback_events", 0)
                )[:5]
                result['top_contexts'] = [
                    {
                        "context_key": k,
                        "observations": v.get("total_feedback_events", 0),
                        "top_modules": sorted(
                            v.get("module_signals", {}).items(),
                            key=lambda x: -abs(x[1].get("mean", 0.0) if isinstance(x[1], dict) else 0.0)
                        )[:3],
                    }
                    for k, v in _sorted_ctx
                ]
                result['total_context_observations'] = _ctxd.get('total_observations', 0)
        except Exception:
            pass

        # Phase 3.1/3.2: calibration — predicted confidence vs actual accuracy.
        # Reads ALL modules present in the file (currently "pcm" from Phase
        # 3.1, "wsdm" added in Phase 3.2) rather than hardcoding "pcm" —
        # CalibrationEngine itself is module-agnostic, the dashboard should
        # be too. Each module gets its own entry; the panel renders one
        # card per module rather than assuming only PCM exists.
        try:
            import json as _jcal
            from pathlib import Path as _Pcal
            _calp = _Pcal("data/persona/calibration_state.json")
            if _calp.exists():
                _cald = _jcal.loads(_calp.read_text())
                _all_bins = _cald.get("bins", {})
                _all_ece  = _cald.get("global_ece", {})
                _pending_count = len(_cald.get("pending", []))
                _modules = {}
                for _mod_name, _mod_bins in _all_bins.items():
                    _modules[_mod_name] = {
                        "ece": _all_ece.get(_mod_name),
                        "bins": _mod_bins,
                        "total_resolved": sum(
                            v.get("predicted_n", 0) for v in _mod_bins.values()
                        ),
                    }
                result['calibration'] = {
                    "modules": _modules,
                    "pending_count": _pending_count,
                }
        except Exception:
            pass

        # ── Phase 6.10-6.16: the mechanisms built since the dashboard was
        # last updated had zero visibility here — added below. Every
        # section reads real, already-persisted state (same singletons
        # the live code paths use — get_internal_cognitive_state(org),
        # get_epistemic_efficacy_model(org), etc.), nothing computed
        # freshly just for display.

        # InternalCognitiveState (v102) — epistemic pressure as a
        # first-class cognitive state, not just a reactive control signal
        try:
            from cognition.internal_cognitive_state import get_internal_cognitive_state
            ics = get_internal_cognitive_state(org)
            if ics.history:
                last = ics.history[-1]
                result['internal_cognitive_state'] = {
                    'pressure': last.get('pressure', 0.0),
                    'pressure_delta': last.get('pressure_delta', 0.0),
                    'meta_error': last.get('meta_error', 0.0),
                    'predicted_next': last.get('predicted_next', 0.0),
                    'history_len': len(ics.history),
                }
        except Exception:
            pass

        # EpistemicEfficacyModel (v105) — the recursive crossing: a
        # falsifiable self-prediction that actually updates from its own
        # error and feeds back into future goal scoring
        try:
            from cognition.epistemic_efficacy_model import get_epistemic_efficacy_model
            eff = get_epistemic_efficacy_model(org)
            if eff is not None:
                recent = eff.history[-5:] if eff.history else []
                result['epistemic_efficacy'] = {
                    'confidence': eff._confidence(),
                    'multiplier': eff.effectiveness_multiplier(),
                    'pending_prediction': eff._pending is not None,
                    'recent_predictions': [
                        {'predicted': round(h['predicted_drop'], 3),
                         'actual': round(h['actual_drop'], 3),
                         'meta_error': round(h['meta_error'], 3)}
                        for h in recent
                    ],
                    'total_observations': len(eff.history),
                }
        except Exception:
            pass

        # SymbolSystem (v107) — the open symbol repertoire, and whether
        # anything is actually competing for attention through it
        try:
            from cognition.symbol_system import get_symbol_system
            sym_sys = get_symbol_system(org)
            syms = list(sym_sys.symbols.values())
            top_syms = sorted(syms, key=lambda s: (s.confidence, s.activation_count), reverse=True)[:5]
            result['symbol_system'] = {
                'total_symbols': len(syms),
                'top_symbols': [
                    {'name': s.name, 'confidence': round(s.confidence, 2),
                     'activations': s.activation_count, 'type': s.symbol_type,
                     'source': s.source_module}
                    for s in top_syms
                ],
                'eligible_for_competition': sum(1 for s in syms if s.confidence >= 0.5),
            }
        except Exception:
            pass

        # SelfDescription (v108/110) — the architecture as an object of
        # its own reflection, and whether it's actually noticed changing
        try:
            from cognition.self_description import get_self_description
            sd = get_self_description()
            result['self_description'] = {
                'components_tracked': len(sd.components),
                'drift_events': len(sd._drift_events),
                'recent_drift': sd.recent_drift_events(3),
                'last_updated': sd.last_updated,
            }
        except Exception:
            pass

        # UnifiedRevisionGateway (v106) — every self-modification
        # decision, not just the ones that succeeded
        try:
            from cognition.unified_revision_gateway import get_unified_revision_gateway
            gw_rev = get_unified_revision_gateway()
            recent_dec = gw_rev.recent_decisions(10)
            outcome_counts = {'applied': 0, 'clamped': 0, 'refused': 0}
            reflection_originated = 0
            for d_dec in recent_dec:
                outcome_counts[d_dec.outcome] = outcome_counts.get(d_dec.outcome, 0) + 1
                if '[[origin:reflection]]' in (d_dec.reason or ''):
                    reflection_originated += 1
            result['revision_gateway'] = {
                'total_logged': len(gw_rev.log),
                'recent_outcomes': outcome_counts,
                'reflection_originated_recent': reflection_originated,
                'recent': [
                    {'target': d_dec.target, 'outcome': d_dec.outcome,
                     'applied_value': d_dec.applied_value, 'reason': (d_dec.reason or '')[:80]}
                    for d_dec in recent_dec[-5:]
                ],
            }
        except Exception:
            pass

        # ReflectionController (v108) — adaptive depth, and how deep it's
        # actually ever gone under real meta-error, not just its ceiling
        try:
            from cognition.reflection_controller import get_reflection_controller
            rc = get_reflection_controller()
            result['reflection_controller'] = {
                'base_max_depth': rc.base_max_depth,
                'hard_cap': rc.hard_cap,
                'expand_threshold': rc.expand_threshold,
                'max_depth_seen': rc.max_depth_seen,
            }
        except Exception:
            pass

        # ArbitrationLearningTracker (v89) — has UTILITY_WEIGHTS actually
        # moved from their documented static defaults, and in which
        # direction
        try:
            from cognition.arbitration_learning import get_tracker
            arb_tracker = get_tracker()
            status = arb_tracker.status()
            if status.get('delta'):
                result['arbitration_learning'] = status
        except Exception:
            pass

        # Phase F (v110) — reuse the already-built observability module
        # directly rather than re-deriving its metrics from partial data
        # shown elsewhere on this page (e.g. only top-5 symbols are listed
        # above; the real entropy needs the full set, which this module
        # already computes correctly).
        try:
            from cognition import loop_observability
            result['loop_observability'] = loop_observability.snapshot(org)
        except Exception:
            pass

    except Exception as e:
        logger.debug(f"[CognitiveDashboard] fetch error: {e}")
    return result


# ── UI helpers ────────────────────────────────────────────────────────────────

def _pct_bar(value: float, color: str = 'blue', height: str = '6px') -> None:
    """Render a horizontal percentage bar.
    Uses ui.html() instead of add_slot() — add_slot() creates a persistent
    server-side listener per call that is NOT reliably released by
    content_col.clear(). With ~15-20 bars rendered every refresh cycle,
    this leaks listeners continuously: 18 bars × 120 refreshes/hour (at
    30s interval) = ~2,160 leaked listeners/hour, eventually degrading the
    WebSocket connection until it becomes unresponsive — the "works fine
    at first, dies after extended use" failure pattern. ui.html() is a
    plain DOM injection with no server-side listener at all.
    """
    pct = max(0.0, min(1.0, value))
    color_map = {
        'blue':   '#3b82f6',
        'green':  '#10b981',
        'yellow': '#f59e0b',
        'red':    '#ef4444',
        'purple': '#8b5cf6',
        'teal':   '#14b8a6',
    }
    bar_color  = color_map.get(color, color)
    warn_color = '#ef4444' if pct < 0.20 else ('#f59e0b' if pct < 0.40 else bar_color)
    ui.html(
        f'<div style="width:100%;height:{height};background:#1e293b;'
        f'border-radius:4px;overflow:hidden;">'
        f'<div style="width:{pct*100:.1f}%;height:100%;background:{warn_color};'
        f'border-radius:4px;transition:width 0.4s;"></div></div>'
    )


def _badge(text: str, color: str = 'gray') -> None:
    color_map = {
        'green':  ('bg-emerald-900', 'text-emerald-300', 'border-emerald-700'),
        'yellow': ('bg-yellow-900',  'text-yellow-300',  'border-yellow-700'),
        'red':    ('bg-red-900',     'text-red-300',     'border-red-700'),
        'blue':   ('bg-blue-900',    'text-blue-300',    'border-blue-700'),
        'purple': ('bg-purple-900',  'text-purple-300',  'border-purple-700'),
        'gray':   ('bg-slate-800',   'text-slate-300',   'border-slate-600'),
    }
    bg, fg, border = color_map.get(color, color_map['gray'])
    ui.label(text).classes(
        f'{bg} {fg} border {border} text-xs px-2 py-0.5 rounded-full font-mono'
    )


def _card(title: str, icon: str):
    """Context manager yielding a styled card."""
    return ui.element('div').classes(
        'bg-slate-900 border border-slate-700 rounded-xl p-4 w-full'
    )


def _section(title: str, icon: str) -> None:
    with ui.row().classes('items-center gap-2 mb-3'):
        ui.label(icon).style('font-size:16px;')
        ui.label(title).classes('text-slate-300 font-semibold text-sm uppercase tracking-wide')


def _kv(key: str, value: Any, mono: bool = False) -> None:
    with ui.row().classes('items-baseline gap-2 w-full'):
        ui.label(key).classes('text-slate-500 text-xs w-40 shrink-0')
        cls = 'text-slate-200 text-sm font-mono' if mono else 'text-slate-200 text-sm'
        ui.label(str(value)).classes(cls)


def _compute_self_awareness_index(d: Dict[str, Any]):
    """
    Combines several already-real, already-logged quantities into one
    number for a quick glance. Deliberately NOT a new measurement of its
    own — every input here is read straight from data already fetched
    above (loop_observability.py's Phase F metrics, epistemic_efficacy's
    real prediction history, the revision gateway's real log). Returns
    None if there isn't enough real data yet to compute anything
    meaningful, rather than a placeholder number.

    This is a heuristic engagement index, not a consciousness metric —
    it answers "how much is the self-referential machinery actually
    doing right now", nothing more. Framed that way in the panel label
    and the italic caption above it, not just in this docstring.
    """
    components = []

    eff = d.get('epistemic_efficacy', {})
    recent_preds = eff.get('recent_predictions', [])
    if recent_preds:
        accurate = sum(1 for p in recent_preds if p['meta_error'] < 0.05)
        self_pred_accuracy = accurate / len(recent_preds)
        components.append(('Self-prediction accuracy', self_pred_accuracy, 0.30))

    lo = d.get('loop_observability', {})
    if lo:
        n_symbols = lo.get('novel_symbols_created', 0)
        if n_symbols >= 2:
            max_possible_entropy = math.log2(n_symbols)
            entropy_norm = (lo.get('symbol_activation_entropy', 0.0) / max_possible_entropy
                             if max_possible_entropy > 0 else 0.0)
            components.append(('Symbol activation diversity', min(1.0, entropy_norm), 0.20))

        depth_ceiling = d.get('reflection_controller', {}).get('hard_cap', 6)
        depth_used = lo.get('max_reflection_depth_reached', 0)
        if depth_ceiling:
            components.append(('Adaptive reflection engagement', min(1.0, depth_used / depth_ceiling), 0.20))

        rev_by_origin = lo.get('revision_count_by_origin', {})
        total_rev = rev_by_origin.get('reflection', 0) + rev_by_origin.get('other', 0)
        if total_rev > 0:
            reflection_share = rev_by_origin.get('reflection', 0) / total_rev
            components.append(('Self-modification from reflection', reflection_share, 0.15))

        drift = lo.get('self_description_drift_events', 0)
        components.append(('Architectural self-observation', min(1.0, drift / 5.0), 0.15))

    if not components:
        return None

    total_weight = sum(w for _, _, w in components)
    score = sum(v * w for _, v, w in components) / total_weight if total_weight else 0.0
    return score, components


# ── Page ──────────────────────────────────────────────────────────────────────

@ui.page('/cognitive-dashboard')
async def cognitive_dashboard_page():
    # Fix: ui.context.client.connected() defaults to a 3.0s timeout for
    # the WebSocket handshake to complete. On a loaded system (local LLM
    # inference, FAISS, vision, 30+ background cognitive modules all
    # competing for the event loop) this can genuinely take longer than
    # 3s, especially when navigating to this page — the heaviest one in
    # the app (13 panels, multiple background JSON reads on first render).
    # An unguarded TimeoutError here kills the page coroutine before the
    # header or content_col are ever built, producing a fully blank page
    # — not even the header renders, since this line runs before it.
    # Raised to 10s and wrapped so a genuine failure (e.g. browser tab
    # closed mid-navigation) shows a clear message instead of nothing.
    try:
        await ui.context.client.connected(timeout=10.0)
    except TimeoutError:
        logger.warning(
            "[CognitiveDashboard] Client connection timed out after 10s — "
            "system may be under heavy load. Page will not render; "
            "navigating back and retrying usually resolves this."
        )
        ui.label('⚠ Connection timed out').classes(
            'text-yellow-400 text-lg font-bold p-8'
        )
        ui.label(
            'The dashboard could not establish a connection in time — '
            'this usually means the system is under heavy load right now. '
            'Try navigating back and opening the dashboard again in a moment.'
        ).classes('text-slate-400 text-sm px-8')
        ui.button(
            'Back to main', icon='arrow_back',
            on_click=lambda: ui.navigate.to('/'),
        ).props('flat').classes('text-slate-400 mx-8 mt-4')
        return

    ui.add_css(GLOBAL_CSS)
    ui.add_css("""
        .dash-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 16px;
        }
        .dash-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 16px;
            width: 100%;
        }
        .reliable  { color: #10b981; }
        .building  { color: #f59e0b; }
        .starved   { color: #ef4444; }
        .dash-page { padding: 72px 24px 80px; max-width: 1400px; margin: 0 auto; }
    """)

    # ── Header ────────────────────────────────────────────────────────────────
    with ui.element('div').classes('dash-page'):
        with ui.row().classes('items-center gap-4 mb-6 w-full'):
            ui.button(
                icon='arrow_back',
                on_click=lambda: ui.navigate.to('/'),
            ).props('flat round').classes('text-slate-400')
            ui.label('Cognitive Health Dashboard').classes(
                'text-xl font-bold text-slate-100'
            )
            ui.space()
            last_refresh_label = ui.label('').classes('text-slate-500 text-xs font-mono')

        # ── Main content container ────────────────────────────────────────────
        content_col = ui.column().classes('w-full gap-4')

        async def render(d: Dict):
            # Fix: render() previously had NO internal exception handling
            # across its ~550 lines of panel construction. Any single
            # exception anywhere — a KeyError from a malformed dict, a
            # torn read mid-write on a JSON data file, a future schema
            # drift — would propagate up, get caught by the bare
            # try/except at the call site, and leave content_col cleared
            # with nothing rebuilt. Result: only the header (back button,
            # title) survives, because it lives outside content_col,
            # while the entire panel area stays permanently blank until
            # the NEXT successful refresh (if one ever comes).
            #
            # Fix has two parts:
            #  1. All JSON writes this dashboard reads are now atomic
            #     (temp file + os.replace) — eliminates torn-read crashes,
            #     the most likely cause given symptoms appearing "after
            #     several chats" (more write activity = more race exposure).
            #  2. This try/except wraps the actual panel-building work so
            #     ANY remaining exception still leaves a visible, readable
            #     error message instead of a blank screen — and the page
            #     remains navigable (back button still works either way,
            #     but now you also know WHY the dashboard is empty).
            # KEY FIX: clear() is now inside the try block. If anything
            # throws during rebuild, the OLD content remains visible rather
            # than the screen going blank. This is the core difference vs
            # the orchestrator page which never clears at all — the
            # orchestrator updates element content in-place via set_content().
            try:
                now_str = time.strftime('%H:%M:%S')
                last_refresh_label.set_text(f'Last refresh: {now_str}  •  cycle #{d.get("slow_cycle", "?")}')

                content_col.clear()
                with content_col:
                    with ui.element('div').classes('dash-grid'):

                        # ── Panel 1: Predictive Model Readiness ───────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Predictive Model Readiness', '🧠')
                            pcm  = d.get('pcm', {})
                            wsdm = d.get('wsdm', {})

                            ui.label('PCM — consequence model').classes(
                                'text-slate-400 text-xs mb-2 mt-1 block'
                            )
                            by_type = pcm.get('by_type', {})
                            for at in ACTION_TYPES:
                                count = by_type.get(at, 0)
                                reliable = count >= MIN_RELIABLE
                                cls = 'reliable' if count >= 30 else ('building' if reliable else 'starved')
                                with ui.row().classes('items-center gap-2 mb-1 w-full'):
                                    ui.label(at[:14].ljust(14)).classes(
                                        f'{cls} text-xs font-mono w-32 shrink-0'
                                    )
                                    _pct_bar(count / 50, 'green' if reliable else 'yellow')
                                    ui.label(f'{count}').classes(f'{cls} text-xs font-mono w-6 text-right')

                            ui.separator().classes('my-3 border-slate-700')
                            ui.label('WSDM — world/self dynamics').classes(
                                'text-slate-400 text-xs mb-2 block'
                            )
                            wsdm_records = wsdm.get('records', 0)
                            wsdm_ready   = wsdm.get('ready', False)
                            net_records  = d.get('wsdm_network_records', 0)
                            _kv('Total records', wsdm_records)
                            _kv('Network dialogue records', net_records)
                            _kv('Relational contexts', len(wsdm.get('by_type', {})))
                            with ui.row().classes('gap-2 mt-2'):
                                _badge('ready' if wsdm_ready else 'building',
                                       'green' if wsdm_ready else 'yellow')
                                if net_records > 0:
                                    _badge(f'{net_records} from Flux', 'purple')

                        # ── Panel 2: Identity Constraint Health ───────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Identity Constraint Health', '🛡')
                            ice = d.get('ice', {})
                            ice_bv = d.get('ice_by_value', {})

                            total       = ice.get('total_checks', 0)
                            evaluations = ice.get('total_evaluations', 0)
                            correction_rate = ice.get('correction_rate', 0)
                            _kv('Total evaluations', evaluations)
                            _kv('Total checks (violations)', total)

                            if evaluations == 0:
                                ui.label('No evaluations yet — engine has not run').classes(
                                    'text-slate-500 text-sm italic mt-2'
                                )
                            elif total == 0:
                                ui.label(
                                    f'{evaluations} evaluations run, 0 violations — '
                                    f'drafts have been passing cleanly'
                                ).classes('text-emerald-400 text-sm italic mt-2')
                            else:
                                with ui.row().classes('gap-2 mb-3'):
                                    corrected = ice.get('corrected', 0)
                                    fallback  = ice.get('fallback', 0)
                                    _badge(f'corrected {corrected}', 'green')
                                    _badge(f'fallback {fallback}',
                                           'red' if (fallback / max(1,total)) > 0.35 else 'yellow')
                                    _badge(f'correction rate {correction_rate:.0%}', 'blue')

                                for value, stats in ice_bv.items():
                                    checks    = stats['checks']
                                    corrected = stats['corrected']
                                    fallback  = stats['fallback']
                                    fb_rate   = fallback / max(1, checks)
                                    color     = 'red' if fb_rate > 0.40 else ('yellow' if fb_rate > 0.20 else 'green')
                                    with ui.row().classes('items-center gap-2 mb-1 w-full'):
                                        ui.label(value[:16].ljust(16)).classes(
                                            'text-slate-300 text-xs font-mono w-32 shrink-0'
                                        )
                                        _pct_bar(corrected / max(1, checks), color)
                                        ui.label(
                                            f'{corrected}/{checks}'
                                        ).classes('text-xs font-mono text-slate-400 w-10 text-right')
                                    if fb_rate > 0.40:
                                        ui.label(
                                            f'  ⚠ High fallback ({fb_rate:.0%}) — '
                                            f'correction directive may need revision'
                                        ).classes('text-red-400 text-xs ml-2')

                        # ── Panel 3: Cross-Layer Feedback ─────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Cross-Layer Feedback', '⚡')
                            clf = d.get('clf', {})
                            ec  = d.get('ec', {})

                            rec_mult = d.get('ec_multiplier', 1.0)
                            mult_color = 'green' if rec_mult > 1.05 else ('yellow' if rec_mult < 0.95 else 'gray')
                            with ui.row().classes('gap-2 items-center mb-2'):
                                ui.label('Ch1 recovery multiplier').classes('text-slate-400 text-xs')
                                _badge(f'{rec_mult:.2f}×', mult_color)

                            boosts = clf.get('drive_boosts', {})
                            if boosts:
                                ui.label('Ch2 active drive boosts').classes('text-slate-400 text-xs mt-2 mb-1')
                                for drive, boost in boosts.items():
                                    with ui.row().classes('items-center gap-2 w-full mb-1'):
                                        ui.label(drive[:14].ljust(14)).classes(
                                            'text-purple-300 text-xs font-mono w-32 shrink-0'
                                        )
                                        _pct_bar(boost / 0.15, 'purple')
                                        ui.label(f'+{boost:.3f}').classes(
                                            'text-purple-300 text-xs font-mono'
                                        )
                            else:
                                ui.label('Ch2: no active drive boosts').classes(
                                    'text-slate-500 text-xs italic mt-1'
                                )

                            gate  = d.get('gate', {})
                            probs = gate.get('probabilities', {})
                            # Fix: this badge previously showed min(live probabilities)
                            # mislabeled as "floor ~ X" with inverted colour logic.
                            # P_FLOOR (cognitive_behavior_gate.py) = 0.05 — the hard
                            # minimum any process probability can reach. The live
                            # probabilities (each in [0.05, 1.0]) reflect CURRENT
                            # resource health: 1.000 = healthy, not "floor reached".
                            ui.label('Ch4 process probabilities').classes('text-slate-400 text-xs mt-3 mb-1')
                            if probs:
                                min_p = min(probs.values())
                                near_floor_color = (
                                    'red'    if min_p <= 0.10 else
                                    'yellow' if min_p <= 0.30 else
                                    'green'
                                )
                                _badge(f'lowest = {min_p:.3f}  (hard floor = 0.05)', near_floor_color)
                                with ui.row().classes('gap-1 flex-wrap mt-1'):
                                    for proc, p in probs.items():
                                        _badge(f'{proc} {p:.2f}', 'gray')

                            arb = d.get('arbitration', {})
                            arb_runs = arb.get('runs', 0)
                            arb_skip = arb.get('skipped', 0)
                            if arb_runs or arb_skip:
                                ui.label('Executive arbitration').classes(
                                    'text-slate-400 text-xs mt-3 mb-1'
                                )
                                with ui.row().classes('gap-1 flex-wrap'):
                                    _badge(f"runs {arb_runs}", 'gray')
                                    _badge(f"overrides {arb.get('overrides', 0)}", 'purple')
                                    _badge(f"skipped {arb_skip}", 'gray')
                            else:
                                ui.label(
                                    'No arbitration cycles yet'
                                ).classes('text-slate-500 text-xs italic mt-2')

                            recent = clf.get('last_events', [])
                            if recent:
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('Recent events').classes('text-slate-400 text-xs mb-1')
                                for ev in recent[-3:]:
                                    ui.label(
                                        f'{ev.get("channel","?")}  {ev.get("write","?")[:45]}'
                                    ).classes('text-slate-500 text-xs font-mono')

                        # ── Panel 4: Aspiration Pipeline ──────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Aspiration Pipeline', '✦')
                            asp   = d.get('aspirations', {})
                            tp    = d.get('tp', {})
                            total_asp = asp.get('total', 0)
                            by_origin = asp.get('by_origin', {})

                            _kv('Active aspirations', total_asp)
                            if by_origin:
                                with ui.row().classes('gap-2 flex-wrap mt-1'):
                                    color_map_orig = {
                                        'synthesis':     'blue',
                                        'generative':    'purple',
                                        'internal_need': 'teal',
                                        'tension':       'yellow',
                                    }
                                    for origin, count in by_origin.items():
                                        _badge(
                                            f'{origin} ×{count}',
                                            color_map_orig.get(origin, 'gray')
                                        )

                            ui.separator().classes('my-3 border-slate-700')
                            pref_path   = tp.get('preferred_action') or '—'
                            pref_arch   = ''
                            tp_domains  = tp.get('domains', [])
                            if tp_domains:
                                pref_arch = tp_domains[0].get('preferred', '')
                            _kv('Temporal path', pref_arch or '—')
                            _kv('Preferred action', pref_path)

                            ui.separator().classes('my-3 border-slate-700')
                            ns = d.get('next_synthesis_cycle', '?')
                            ng = d.get('next_generative_cycle', '?')
                            ns_label = (f'unlocks in {ns} cycles'
                                        if d.get('synthesis_locked') else f'in {ns} cycles')
                            ng_label = (f'unlocks in {ng} cycles'
                                        if d.get('generative_locked') else f'in {ng} cycles')
                            _kv('Next synthesis scan', ns_label)
                            _kv('Next generative scan', ng_label)

                        # Yield event loop between panel groups — creating 13
                        # panels' worth of UI elements synchronously can hold
                        # the event loop long enough to miss WebSocket
                        # keepalives on a loaded system. Spread across 3
                        # groups rather than building all 13 panels in one
                        # uninterrupted block.
                        import asyncio as _aio
                        await _aio.sleep(0)

                        # ── Panel 5: Motivational State ───────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Motivational State', '🧭')
                            mf = d.get('mf', {})
                            dv = mf.get('drive_vector', {})
                            dominant = mf.get('dominant', '—')
                            clf_boosts = d.get('clf', {}).get('drive_boosts', {})

                            _kv('Dominant drive', dominant)

                            if dv:
                                ui.label('Drive vector').classes('text-slate-400 text-xs mt-3 mb-2')
                                sorted_drives = sorted(dv.items(), key=lambda x: x[1], reverse=True)
                                for drive, val in sorted_drives:
                                    boost = clf_boosts.get(drive, 0)
                                    color = 'purple' if drive == dominant else 'blue'
                                    with ui.row().classes('items-center gap-2 mb-1 w-full'):
                                        ui.label(drive[:14].ljust(14)).classes(
                                            f'text-xs font-mono w-32 shrink-0 '
                                            f'{"text-purple-300" if drive == dominant else "text-slate-300"}'
                                        )
                                        _pct_bar(val, color)
                                        ui.label(f'{val:.2f}').classes(
                                            'text-xs font-mono text-slate-400 w-10 text-right'
                                        )
                                        if boost > 0.005:
                                            ui.label(f'+{boost:.3f}').classes(
                                                'text-purple-400 text-xs font-mono'
                                            )

                        # ── Panel 6: Resource Economy ─────────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Resource Economy', '⚡')
                            ec   = d.get('ec', {})
                            gate = d.get('gate', {})
                            res  = gate.get('resources', ec)

                            pools = [
                                ('cognitive_energy', 'blue'),
                                ('social_energy',    'teal'),
                                ('attention',        'purple'),
                            ]
                            pressure = res.get('pressure', {})
                            for pool, color in pools:
                                val = res.get(pool, 0)
                                is_low = pressure.get(f'{pool.split("_")[0]}_low', False)
                                is_crit = pressure.get(f'{pool.split("_")[0]}_critical', False)
                                disp_color = 'red' if is_crit else ('yellow' if is_low else color)
                                with ui.row().classes('items-center gap-2 mb-2 w-full'):
                                    ui.label(pool[:16].replace('_',' ').ljust(16)).classes(
                                        'text-slate-300 text-xs font-mono w-32 shrink-0'
                                    )
                                    _pct_bar(val, disp_color, '8px')
                                    badge_color = 'red' if is_crit else ('yellow' if is_low else 'green')
                                    _badge(f'{val:.2f}', badge_color)

                            max_tok = gate.get('max_response_tokens', 800)
                            ui.separator().classes('my-3 border-slate-700')
                            _kv('Max response tokens', max_tok, mono=True)
                            _kv('Recovery multiplier', f'{d.get("ec_multiplier", 1.0):.2f}×', mono=True)
                            _kv('Dream mode', str(res.get('in_dream_mode', False)))

                        # ── Panel 7: PandoraBOX-Flux Network ──────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('PandoraBOX-Flux Network', '🌐')
                            flux_rels = d.get('flux_relations', [])

                            if not flux_rels:
                                ui.label('No Flux connection active').classes(
                                    'text-slate-500 text-sm italic'
                                )
                            else:
                                for rel in flux_rels:
                                    uid          = rel['uid']
                                    trust        = rel['trust']
                                    interactions = rel['interactions']
                                    topics       = rel['topics']
                                    trust_color = (
                                        'green'  if trust > 0.70 else
                                        'yellow' if trust > 0.45 else 'red'
                                    )
                                    with ui.row().classes('gap-2 items-center mb-1'):
                                        ui.label(uid.replace('lumina_child_', '')).classes(
                                            'text-slate-300 text-xs font-mono'
                                        )
                                        _badge(f'trust {trust:.2f}', trust_color)
                                    _kv('Interactions', interactions)
                                    _kv('Shared topics', topics)
                                    with ui.row().classes('items-center gap-2 w-full mb-3'):
                                        ui.label('Trust').classes('text-slate-400 text-xs w-32 shrink-0')
                                        _pct_bar(trust, trust_color)

                            net_r = d.get('wsdm_network_records', 0)
                            pcm_total = d.get('pcm', {}).get('records', 0)
                            # Fix: these are CUMULATIVE historical records, independent
                            # of whether a Flux peer is connected RIGHT NOW. Previously
                            # rendered directly under "No Flux connection active" with
                            # no distinction, implying an active connection.
                            if net_r > 0:
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('Historical Flux contribution (all-time)').classes(
                                    'text-slate-500 text-xs mb-1 italic'
                                )
                                _kv('PCM records (total)', pcm_total)
                                _kv('WSDM from Flux (all-time)', net_r)
                                if pcm_total > 0:
                                    share = net_r / pcm_total
                                    with ui.row().classes('items-center gap-2 w-full'):
                                        ui.label('Flux share').classes('text-slate-400 text-xs w-32 shrink-0')
                                        _pct_bar(share, 'purple')
                                        _badge(f'{share:.0%}', 'purple')
                            # FluxMindModel
                            fmm = d.get('flux_mind', {})
                            if fmm.get('exchanges_analysed', 0) > 0:
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('Flux mind model').classes('text-slate-400 text-xs mb-1')
                                _kv('Exchanges analysed', fmm['exchanges_analysed'])
                                _kv('Expected depth', f"{fmm.get('expected_depth',0):.2f}")
                                _kv('Expected dialogue', f"{fmm.get('expected_dialogue',0):.2f}")
                                top_cur = fmm.get('top_curiosities', [])
                                if top_cur:
                                    _kv('Top curiosities', ', '.join(t[0] for t in top_cur[:3]))

                        # ── Panel 8: Causal & Planning ────────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Causal & Long-Horizon Planning', '🔗')
                            cm  = d.get('causal_mechanism', {})
                            lhp = d.get('long_horizon', {})

                            ui.label('Causal mechanisms').classes('text-slate-400 text-xs mb-2')
                            sigs = cm.get('signatures', 0)
                            _kv('Action types with signatures', sigs)
                            at_list = cm.get('action_types', [])
                            if at_list:
                                with ui.row().classes('gap-1 flex-wrap'):
                                    for at in at_list:
                                        _badge(at[:12], 'teal')

                            ui.separator().classes('my-3 border-slate-700')
                            ui.label('Long-horizon plans').classes('text-slate-400 text-xs mb-2')
                            plans = lhp.get('plans', [])
                            if not plans:
                                ui.label('No active plans').classes('text-slate-500 text-xs italic')
                            for plan in plans[:3]:
                                completed = plan.get('completed', 0)
                                total     = plan.get('steps', 0)
                                nxt       = plan.get('next', '—')
                                domain    = plan.get('domain', '')[:25]
                                with ui.row().classes('items-center gap-2 w-full mb-1'):
                                    ui.label(domain).classes('text-slate-300 text-xs font-mono w-32 shrink-0')
                                    _pct_bar(completed / max(1, total), 'blue')
                                    ui.label(f'{completed}/{total}').classes('text-xs text-slate-400 font-mono')
                                _kv('Next action', nxt)

                        await _aio.sleep(0)

                        # ── Panel 9: Narrative & Self-Trajectory ──────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Narrative & Self-Trajectory', '📖')
                            nc  = d.get('narrative_compression', {})
                            tsp = d.get('temporal_self', {})

                            ui.label('Identity themes').classes('text-slate-400 text-xs mb-2')
                            themes = nc.get('top', [])
                            if not themes:
                                ui.label('Compressing — needs more chapters').classes(
                                    'text-slate-500 text-xs italic'
                                )
                            for theme in themes:
                                ui.label(f'• {theme}').classes('text-slate-300 text-xs mb-1')

                            ui.separator().classes('my-3 border-slate-700')
                            ui.label('Value trajectory').classes('text-slate-400 text-xs mb-2')
                            rising  = tsp.get('rising',  [])
                            falling = tsp.get('falling', [])
                            if not rising and not falling:
                                ui.label('Building trajectory — needs more cycles').classes(
                                    'text-slate-500 text-xs italic'
                                )
                            else:
                                if rising:
                                    with ui.row().classes('gap-1 flex-wrap mb-1'):
                                        ui.label('Rising:').classes('text-emerald-400 text-xs')
                                        for d_name in rising:
                                            _badge(f'{d_name}↑', 'green')
                                if falling:
                                    with ui.row().classes('gap-1 flex-wrap mb-1'):
                                        ui.label('Receding:').classes('text-red-400 text-xs')
                                        for d_name in falling:
                                            _badge(f'{d_name}↓', 'red')
                                excerpt = tsp.get('narrative_excerpt', '')
                                if excerpt:
                                    ui.label(excerpt).classes('text-slate-400 text-xs italic mt-2')

                        # ── Panel 10: Meta-Learning & Structural Coupling ──────
                        with ui.element('div').classes('dash-card'):
                            _section('Meta-Learning & Structural Coupling', '🔄')
                            mla = d.get('meta_learning', {})
                            sc  = d.get('structural_coupling', {})

                            ui.label('Cognitive audit meta-learning').classes('text-slate-400 text-xs mb-2')
                            by_v = mla.get('by_verdict', {})
                            total_evals = mla.get('total_evaluations', 0)
                            if total_evals == 0:
                                ui.label('No evaluations yet').classes('text-slate-500 text-xs italic')
                            else:
                                with ui.row().classes('gap-2'):
                                    _badge(f"effective {by_v.get('effective',0)}", 'green')
                                    _badge(f"transient {by_v.get('transient',0)}", 'yellow')
                                    _badge(f"ineffective {by_v.get('ineffective',0)}", 'red')
                                dnr = mla.get('do_not_repeat', [])
                                if dnr:
                                    ui.label(f'Do-not-repeat: {", ".join(dnr)}').classes(
                                        'text-red-400 text-xs mt-1'
                                    )
                                pending = mla.get('pending_evaluation', 0)
                                if pending:
                                    _kv('Pending evaluation', pending)

                            ui.separator().classes('my-3 border-slate-700')
                            ui.label('Structural coupling').classes('text-slate-400 text-xs mb-2')
                            _kv('Updates sent', sc.get('sent', 0))
                            _kv('Updates received', sc.get('received', 0))

                        await _aio.sleep(0)

                        # ── Panel 11: Cognitive Attention (Phase 2.9 / 2.9.1) ─────
                        with ui.element('div').classes('dash-card'):
                            _section('Cognitive Attention Allocation', '🎯')
                            attn = d.get('attention', {})
                            if not attn:
                                ui.label(
                                    'Not yet computed — fires on first consolidation cycle (10 min)'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                pf = attn.get('primary', '')
                                sf = attn.get('secondary', [])
                                ui.label('Softmax attention (sum = 1.0)').classes('text-slate-400 text-xs mb-2')
                                if pf:
                                    _badge(pf.replace('_', ' '), 'teal')
                                if sf:
                                    with ui.row().classes('gap-1 flex-wrap mt-1'):
                                        for s in sf:
                                            _badge(s.replace('_', ' '), 'blue')
                                wt = attn.get('weights', {})
                                if wt:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Module weights').classes('text-slate-400 text-xs mb-2')
                                    for mod, val in sorted(wt.items(), key=lambda x: -x[1]):
                                        with ui.row().classes('items-center gap-2 w-full mb-1'):
                                            ui.label(mod.replace('_', ' ')).classes(
                                                'text-slate-300 text-xs w-36 shrink-0'
                                            )
                                            _pct_bar(val * 5, 'teal', '4px')
                                            ui.label(f'{val:.3f}').classes(
                                                'text-slate-400 text-xs w-12 text-right'
                                            )
                                bias = attn.get('learned_bias', {})
                                nonzero_bias = {k: v for k, v in bias.items() if abs(v) > 0.001}
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('Learned bias (from feedback)').classes('text-slate-400 text-xs mb-2')
                                if nonzero_bias:
                                    with ui.row().classes('gap-1 flex-wrap'):
                                        for mod, val in sorted(nonzero_bias.items(), key=lambda x: -abs(x[1])):
                                            colour = 'green' if val > 0 else 'red'
                                            _badge(f"{mod.replace('_',' ')} {val:+.3f}", colour)
                                else:
                                    ui.label(
                                        'No learned bias yet — needs feedback signal'
                                    ).classes('text-slate-500 text-xs italic')
                                topic = attn.get('workspace_topic', '')
                                if topic:
                                    ui.separator().classes('my-3 border-slate-700')
                                    _kv('Workspace topic at last compute', topic[:60])

                                ckey = attn.get('context_key', '')
                                if ckey:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Current context (Phase 3.0)').classes(
                                        'text-slate-400 text-xs mb-1'
                                    )
                                    ui.label(ckey).classes(
                                        'text-teal-400 text-xs font-mono'
                                    )

                        # ── Panel 12: Contextual Attention Learning (Phase 3.0) ───
                        with ui.element('div').classes('dash-card'):
                            _section('Contextual Attention Learning', '🧭')
                            top_ctx = d.get('top_contexts', [])
                            total_obs = d.get('total_context_observations', 0)
                            if not top_ctx:
                                ui.label(
                                    'No learned contexts yet — needs feedback events '
                                    '(min 2 per context to activate retrieval)'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                _kv('Total context observations', total_obs)
                                ui.separator().classes('my-3 border-slate-700')
                                for ctx in top_ctx:
                                    # Defensive .get() access — ctx is read from a JSON
                                    # file written by a separate background thread; even
                                    # with atomic writes, defend against schema drift
                                    # rather than assume every key is always present.
                                    with ui.column().classes('w-full mb-3 gap-1'):
                                        ui.label(ctx.get('context_key', '?')).classes(
                                            'text-teal-400 text-xs font-mono'
                                        )
                                        with ui.row().classes('items-center gap-2'):
                                            _badge(f"{ctx.get('observations', 0)} obs", 'blue')
                                            for mod, sig in ctx.get('top_modules', []):
                                                _mean = sig.get('mean', 0.0) if isinstance(sig, dict) else 0.0
                                                colour = 'green' if _mean > 0 else 'red'
                                                _badge(
                                                    f"{mod.replace('_',' ')} {_mean:+.3f}",
                                                    colour
                                                )

                        # ── Panel 13: Calibration & Uncertainty (Phase 3.1/3.2) ───
                        with ui.element('div').classes('dash-card'):
                            _section('Calibration & Uncertainty', '📐')
                            cal = d.get('calibration', {})
                            modules = cal.get('modules', {}) if cal else {}
                            any_resolved = any(
                                m.get('total_resolved', 0) > 0 for m in modules.values()
                            )
                            if not modules or not any_resolved:
                                ui.label(
                                    'No resolved predictions yet — calibration data '
                                    'accumulates as predictions (PCM, WSDM) are checked '
                                    'against actual outcomes (needs ≥5 per confidence '
                                    'bin to trust)'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                _kv('Pending (awaiting outcome, all modules)',
                                    cal.get('pending_count', 0))
                                _module_labels = {"pcm": "PCM", "wsdm": "WSDM"}
                                for _mod_name in sorted(modules.keys()):
                                    _mod_data = modules[_mod_name]
                                    if _mod_data.get('total_resolved', 0) == 0:
                                        continue
                                    ui.separator().classes('my-3 border-slate-700')
                                    _mod_label = _module_labels.get(_mod_name, _mod_name.upper())
                                    ece = _mod_data.get('ece')
                                    with ui.row().classes('items-center gap-2 mb-2'):
                                        ui.label(_mod_label).classes(
                                            'text-slate-300 text-xs font-bold'
                                        )
                                        if ece is not None:
                                            ece_colour = (
                                                'green' if ece < 0.10 else
                                                'yellow' if ece < 0.25 else
                                                'red'
                                            )
                                            _badge(f'ECE = {ece:.3f}', ece_colour)
                                    _kv(f'{_mod_label} resolved predictions',
                                        _mod_data.get('total_resolved', 0))
                                    bins = _mod_data.get('bins', {})
                                    if bins:
                                        for bin_key, bin_data in sorted(bins.items()):
                                            n = bin_data.get('predicted_n', 0)
                                            acc = bin_data.get('accuracy', 0)
                                            trusted = n >= 5
                                            with ui.row().classes('items-center gap-2 w-full mb-1'):
                                                ui.label(bin_key).classes(
                                                    'text-slate-300 text-xs w-24 shrink-0 font-mono'
                                                )
                                                _pct_bar(acc, 'teal' if trusted else 'gray', '4px')
                                                ui.label(f'{acc:.2f} (n={n})').classes(
                                                    'text-slate-400 text-xs w-20 text-right'
                                                )

                        # ── Panel 14: Counterfactual Simulator (Phase 3.3) ──────
                        with ui.element('div').classes('dash-card'):
                            _section('Counterfactual Simulator', '🔀')
                            cf = d.get('counterfactual', {})
                            total_q = cf.get('total_queries', 0)
                            if total_q == 0:
                                ui.label(
                                    'No queries yet — simulator activates at cycle 15, '
                                    'then runs every 5 consolidation cycles (~50 min). '
                                    'Requires ≥15 PCM records.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                streak = cf.get('worst_actual_streak', 0)
                                _kv('Total interactions analysed', total_q)
                                if streak > 0:
                                    streak_colour = 'red' if streak >= 3 else 'yellow'
                                    with ui.row().classes('items-center gap-2 mb-2'):
                                        ui.label('Worst-actual streak').classes(
                                            'text-slate-400 text-xs'
                                        )
                                        _badge(f'{streak} consecutive', streak_colour)
                                    if streak >= 3:
                                        ui.label(
                                            '⚠ PandoraBOX has repeatedly chosen the '
                                            'lowest-predicted action — worth reviewing.'
                                        ).classes('text-yellow-400 text-xs mb-2')
                                else:
                                    _badge('No streak', 'green')
                                recent = cf.get('recent', [])
                                if recent:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Recent counterfactual queries').classes(
                                        'text-slate-400 text-xs mb-2'
                                    )
                                    for entry in recent[-3:]:
                                        actual   = entry.get('actual_action', '?')
                                        best_alt = entry.get('best_alternative', '?')
                                        rank     = entry.get('actual_rank', '?')
                                        n_total  = len(ACTION_TYPES)
                                        outcome  = entry.get('actual_outcome', '?')
                                        a_score  = entry.get('actual_score')
                                        b_score  = entry.get('best_score')
                                        # Bug fix (v59): CounterfactualQuery never had
                                        # actual_score/best_score fields — asdict(q)
                                        # only produces interaction_n, actual_action,
                                        # branches, best_alternative, actual_rank,
                                        # worst_actual. The real per-branch scores live
                                        # at branches[action_type]['joint_score']. The
                                        # two .get() calls above always returned None,
                                        # so Δscore was hardcoded to 0-0=+0.0000 on
                                        # every single query since this panel existed.
                                        if a_score is None or b_score is None:
                                            _branches = entry.get('branches', {}) or {}
                                            _a_b = _branches.get(actual, {}) or {}
                                            _b_b = _branches.get(best_alt, {}) or {}
                                            a_score = _a_b.get('joint_score', a_score)
                                            b_score = _b_b.get('joint_score', b_score)
                                        worst    = entry.get('worst_actual', False)
                                        rank_colour = (
                                            'green' if rank == 1 else
                                            'yellow' if rank <= 3 else
                                            'red' if worst else 'gray'
                                        )
                                        with ui.column().classes('w-full mb-2 gap-1'):
                                            with ui.row().classes('items-center gap-2'):
                                                ui.label(actual.replace('_', ' ')).classes(
                                                    'text-slate-300 text-xs font-mono'
                                                )
                                                _badge(f'rank {rank}/{n_total}', rank_colour)
                                                _badge(outcome, 'green' if outcome == 'positive' else 'red' if outcome == 'negative' else 'gray')
                                            if best_alt and best_alt != actual:
                                                ui.label(
                                                    f'best alt: {best_alt.replace("_"," ")} '
                                                    f'(Δscore {round((b_score or 0) - (a_score or 0), 4):+.4f})'
                                                ).classes('text-slate-500 text-xs ml-2')

                        await _aio.sleep(0)

                        # ── Panel 15: Introspective Observer (Phase 3.5) ─────────
                        with ui.element('div').classes('dash-card'):
                            _section('Introspective Observer', '🔭')
                            io = d.get('introspective', {})
                            total_obs = io.get('total_observations', 0)
                            if total_obs == 0:
                                ui.label(
                                    'No observations yet — Observer activates at cycle 20, '
                                    'then runs every 10 consolidation cycles (~100 min). '
                                    'Computes trends from attention history, personality, '
                                    'calibration, counterfactual, and emergence data.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                by_cat = io.get('by_category', {})
                                _kv('Total observations', total_obs)
                                if by_cat:
                                    with ui.row().classes('flex-wrap gap-1 mb-2'):
                                        for cat, count in sorted(by_cat.items()):
                                            cat_colours = {
                                                'attention':      'blue',
                                                'personality':    'purple',
                                                'calibration':    'teal',
                                                'counterfactual': 'yellow',
                                                'emergence':      'green',
                                                'emotion':        'orange',
                                            }
                                            _badge(f'{cat} ×{count}',
                                                   cat_colours.get(cat, 'gray'))
                                recent_obs = io.get('recent', [])
                                if recent_obs:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Recent observations (most recent first)').classes(
                                        'text-slate-400 text-xs mb-2'
                                    )
                                    for obs_entry in reversed(recent_obs[-5:]):
                                        cat        = obs_entry.get('category', '?')
                                        obs_text   = obs_entry.get('observation', '')
                                        sig        = obs_entry.get('significance', 0)
                                        delta_pct  = obs_entry.get('delta_pct', 0)
                                        sig_colour = (
                                            'red'    if sig >= 20 else
                                            'yellow' if sig >= 10 else
                                            'blue'
                                        )
                                        with ui.column().classes('w-full mb-2 gap-0'):
                                            with ui.row().classes('items-center gap-1'):
                                                cat_colours = {
                                                    'attention':      'blue',
                                                    'personality':    'purple',
                                                    'calibration':    'teal',
                                                    'counterfactual': 'yellow',
                                                    'emergence':      'green',
                                                }
                                                _badge(cat, cat_colours.get(cat, 'gray'))
                                                if delta_pct != 0:
                                                    _badge(f'{delta_pct:+.1f}%', sig_colour)
                                            ui.label(obs_text).classes(
                                                'text-slate-300 text-xs mt-1'
                                            )

                        # ── Panel 16: Global Workspace (Phase 4.x) ────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Global Workspace', '🌐')
                            gw = d.get('global_workspace', {})
                            if not gw:
                                ui.label(
                                    'No workspace state yet — populated by the '
                                    'autonomous slow-cycle (WorkspaceCompetition + '
                                    'Executive Arbitration) every 2 slow cycles.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                _kv('Current focus', gw.get('focus') or '—')
                                _kv('Confidence', f"{gw.get('confidence', 0):.2f}", mono=True)
                                _kv('Uncertainty', f"{gw.get('uncertainty', 0):.2f}", mono=True)
                                _kv('Workspace entropy', f"{gw.get('workspace_entropy', 0):.2f}", mono=True)

                                participants = gw.get('broadcast_participants', [])
                                if participants:
                                    ui.label('Broadcast participants').classes(
                                        'text-slate-400 text-xs mb-1 mt-2 block'
                                    )
                                    with ui.row().classes('flex-wrap gap-1 mb-2'):
                                        for p in participants:
                                            _badge(p, 'blue')

                                policy = gw.get('executive_policy', {})
                                if policy:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Executive policy').classes(
                                        'text-slate-400 text-xs mb-1 block'
                                    )
                                    with ui.row().classes('flex-wrap gap-1 mb-2'):
                                        for dim, val in policy.items():
                                            if abs(val) < 0.05:
                                                continue
                                            _badge(f'{dim} {val:+.2f}',
                                                   'green' if val >= 0 else 'red')

                                hyps = gw.get('active_hypotheses', [])
                                if hyps:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label(
                                        f'Active hypotheses ({len(hyps)})'
                                    ).classes('text-slate-400 text-xs mb-1 block')
                                    _source_colors = {
                                        'goal': 'blue', 'thought': 'gray',
                                        'peer_cognition': 'purple',
                                        'vision_presence': 'orange', 'vision_scene': 'orange',
                                        'symbol_system': 'teal',
                                    }
                                    for h in hyps[:5]:
                                        label = h.get('label', h.get('name', '?'))
                                        score = h.get('score', 0)
                                        # Provenance badge — the data was already
                                        # here (v99's fix exposes 'source' at the
                                        # candidate's top level), just never
                                        # rendered. Seeing WHERE the current focus
                                        # came from (own goal vs. a percept vs. a
                                        # symbol vs. Flux) is real, useful signal
                                        # for judging what's actually driving
                                        # attention, not decoration.
                                        src = h.get('source', 'goal')
                                        with ui.row().classes('items-center gap-1'):
                                            _badge(src, _source_colors.get(src, 'gray'))
                                            ui.label(f'{label}  (score={score:.2f})').classes(
                                                'text-slate-300 text-xs font-mono'
                                            )

                                changes = gw.get('updated_by', [])
                                if changes:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Recent workspace changes').classes(
                                        'text-slate-400 text-xs mb-1 block'
                                    )
                                    for c in reversed(changes[-5:]):
                                        src    = c.get('source', '?')
                                        fields = ', '.join(c.get('fields', []))
                                        ui.label(f'{src}: {fields}').classes(
                                            'text-slate-500 text-xs font-mono'
                                        )

                        # ── Panel 17: Emotion, Mood & Threat Response (v78/v78b) ──
                        with ui.element('div').classes('dash-card'):
                            _section('Emotion, Mood & Threat', '🌡️')
                            em = d.get('emotion_mood', {})
                            threat = d.get('threat_level', None)
                            if not em:
                                ui.label(
                                    'No emotional-state data yet.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                iv = em.get('instant_valence')
                                ia = em.get('instant_arousal')
                                mv = em.get('mood_valence')
                                ma = em.get('mood_arousal')
                                _kv('Right now (instant)', f"valence={iv:.2f} arousal={ia:.2f}" if iv is not None else '—', mono=True)
                                _kv('Background mood', f"valence={mv:.2f} arousal={ma:.2f}" if mv is not None else '—', mono=True)
                                if iv is not None and mv is not None:
                                    divergence = abs(mv - iv)
                                    _kv('Mood/moment divergence', f"{divergence:.2f}" + (" (notable)" if divergence > 0.35 else ""), mono=True)
                                desc = em.get('description', '')
                                if desc:
                                    ui.label(f'"{desc}"').classes('text-slate-300 text-xs italic mt-2 mb-2')

                                emotions = em.get('emotions', {})
                                if emotions:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Individual emotions').classes(
                                        'text-slate-400 text-xs mb-1 block'
                                    )
                                    with ui.row().classes('flex-wrap gap-1 mb-2'):
                                        for name, val in sorted(emotions.items(), key=lambda x: -x[1]):
                                            color = 'green' if val > 0.6 else 'yellow' if val > 0.35 else 'gray'
                                            _badge(f'{name} {val:.2f}', color)

                                ui.separator().classes('my-3 border-slate-700')
                                if threat is not None:
                                    t_color = 'red' if threat >= 0.5 else 'yellow' if threat >= 0.15 else 'green'
                                    with ui.row().classes('items-center gap-2'):
                                        ui.label('Threat level:').classes('text-slate-400 text-xs')
                                        _badge(f'{threat:.2f}', t_color)
                                    if threat < 0.15:
                                        ui.label(
                                            'Below the noise floor — no anxiety response triggered.'
                                        ).classes('text-slate-600 text-xs mt-1')
                                else:
                                    ui.label('Threat level: unavailable this cycle').classes(
                                        'text-slate-500 text-xs'
                                    )

                        # ── Panel 18: Internal Cognitive State (v102) ──────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Epistemic Self-Model', '🪞')
                            ics = d.get('internal_cognitive_state', {})
                            if not ics:
                                ui.label(
                                    'No data yet — needs at least one deliberate() cycle.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                _kv('Current pressure (P)', f"{ics['pressure']:.3f}", mono=True)
                                _pct_bar(ics['pressure'], 'amber')
                                _kv('Pressure delta (dP/dt)', f"{ics['pressure_delta']:+.3f}", mono=True)
                                _kv('Meta-prediction error', f"{ics['meta_error']:.3f}", mono=True)
                                _kv('Self-predicted next P', f"{ics['predicted_next']:.3f}", mono=True)
                                _kv('History depth', ics['history_len'], mono=True)
                                ui.label(
                                    'meta_error is how wrong the organism\'s own trend-'
                                    'extrapolation guess about its NEXT pressure reading '
                                    'turned out to be — not a claim, a measured miss.'
                                ).classes('text-slate-600 text-xs mt-2 italic')

                        # ── Panel 19: Epistemic Efficacy Model (v105) ──────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Recursive Crossing (Self-Efficacy)', '🔁')
                            eff = d.get('epistemic_efficacy', {})
                            if not eff:
                                ui.label(
                                    'No data yet — needs resolve_uncertainty to win at least once.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                _kv('Effectiveness belief', f"{eff['confidence']:.3f}", mono=True)
                                _pct_bar(eff['confidence'], 'purple')
                                _kv('Scoring multiplier (self-effect)', f"{eff['multiplier']:.3f}×", mono=True)
                                _kv('Prediction pending observation', 'yes' if eff['pending_prediction'] else 'no')
                                _kv('Total falsifiable predictions made', eff['total_observations'], mono=True)
                                recent = eff.get('recent_predictions', [])
                                if recent:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Recent predicted vs actual').classes(
                                        'text-slate-400 text-xs mb-1 block'
                                    )
                                    for r in reversed(recent):
                                        hit = r['meta_error'] < 0.05
                                        _badge('accurate' if hit else 'missed', 'green' if hit else 'red')
                                        ui.label(
                                            f"predicted {r['predicted']:+.3f}  actual {r['actual']:+.3f}  "
                                            f"error {r['meta_error']:.3f}"
                                        ).classes('text-slate-300 text-xs font-mono inline ml-1')
                                ui.label(
                                    'The multiplier is the causal crossing: this number, '
                                    'learned from whether past self-predictions were right, '
                                    'is scaling resolve_uncertainty\'s OWN future score right now.'
                                ).classes('text-slate-600 text-xs mt-2 italic')

                        # ── Panel 20: Symbol System (v107) ─────────────────────────
                        with ui.element('div').classes('dash-card'):
                            _section('Open Symbol Repertoire', '🔣')
                            sym = d.get('symbol_system', {})
                            if not sym or not sym.get('total_symbols'):
                                ui.label(
                                    'No symbols absorbed yet — needs a high-significance '
                                    'narrative chapter, a resolved self-inquiry, or an '
                                    'evolution-mode reflection.'
                                ).classes('text-slate-500 text-xs italic')
                            else:
                                _kv('Total symbols', sym['total_symbols'], mono=True)
                                _kv('Eligible to compete for attention', sym['eligible_for_competition'], mono=True)
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('Top symbols (by confidence × activation)').classes(
                                    'text-slate-400 text-xs mb-1 block'
                                )
                                for s in sym.get('top_symbols', []):
                                    with ui.row().classes('items-center gap-1'):
                                        _badge(s['type'], 'teal')
                                        ui.label(
                                            f"{s['name'][:40]}  conf={s['confidence']:.2f}  "
                                            f"×{s['activations']}  ({s['source']})"
                                        ).classes('text-slate-300 text-xs font-mono')

                        # ── Panel 21: Self-Description & Revision Gateway (v106-110) ─
                        with ui.element('div').classes('dash-card'):
                            _section('Architecture as Object of Reflection', '🧬')
                            sd_data = d.get('self_description', {})
                            gw_rev = d.get('revision_gateway', {})
                            if not sd_data and not gw_rev:
                                ui.label('No data yet.').classes('text-slate-500 text-xs italic')
                            else:
                                if sd_data:
                                    _kv('Components tracked', sd_data.get('components_tracked', 0), mono=True)
                                    _kv('Drift events (total)', sd_data.get('drift_events', 0), mono=True)
                                    if sd_data.get('drift_events', 0) > 0:
                                        ui.label(
                                            'A drift event means the organism\'s own real '
                                            'config differs from what it last registered '
                                            'about itself — a genuine self-observation, not a claim.'
                                        ).classes('text-slate-600 text-xs mt-1 italic')
                                if gw_rev:
                                    ui.separator().classes('my-3 border-slate-700')
                                    ui.label('Self-modification decisions (recent)').classes(
                                        'text-slate-400 text-xs mb-1 block'
                                    )
                                    oc = gw_rev.get('recent_outcomes', {})
                                    with ui.row().classes('flex-wrap gap-1 mb-2'):
                                        _badge(f"applied {oc.get('applied', 0)}", 'green')
                                        _badge(f"clamped {oc.get('clamped', 0)}", 'yellow')
                                        _badge(f"refused {oc.get('refused', 0)}", 'red')
                                    _kv('From reflection (recent)', gw_rev.get('reflection_originated_recent', 0), mono=True)
                                    _kv('Total logged (all-time)', gw_rev.get('total_logged', 0), mono=True)
                                    for r in gw_rev.get('recent', [])[-3:]:
                                        ui.label(
                                            f"{r['target']}: {r['outcome']} → {r['applied_value']}"
                                        ).classes('text-slate-500 text-xs font-mono')

                        # ── Panel 22: Reflection Depth & Arbitration Learning ──────
                        with ui.element('div').classes('dash-card'):
                            _section('Adaptive Reflection & Learned Weights', '⚙️')
                            rc = d.get('reflection_controller', {})
                            arb = d.get('arbitration_learning', {})
                            if rc:
                                _kv('Base depth / hard cap', f"{rc.get('base_max_depth')} / {rc.get('hard_cap')}", mono=True)
                                _kv('Deepest reflection actually reached', rc.get('max_depth_seen', 0), mono=True)
                                _kv('Expansion threshold (meta_error)', f"{rc.get('expand_threshold', 0):.2f}", mono=True)
                            if arb and arb.get('delta'):
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('UTILITY_WEIGHTS learned delta').classes(
                                    'text-slate-400 text-xs mb-1 block'
                                )
                                with ui.row().classes('flex-wrap gap-1'):
                                    for term, val in arb['delta'].items():
                                        _badge(f'{term} {val:+.3f}', 'green' if val >= 0 else 'red')
                                _kv('Good / bad outcomes', f"{arb.get('good_total', 0)} / {arb.get('bad_total', 0)}", mono=True)
                            if not rc and not (arb and arb.get('delta')):
                                ui.label('No data yet.').classes('text-slate-500 text-xs italic')

                        # ── Panel 23: Cognitive Self-Awareness Index (composite) ───
                        with ui.element('div').classes('dash-card'):
                            _section('Cognitive Self-Awareness Index', '🎯')
                            ui.label(
                                'A heuristic aggregate of measurable self-referential '
                                'activity — not a claim about consciousness or experience. '
                                'Each component is a real, already-logged quantity; this '
                                'panel only combines them for one glance.'
                            ).classes('text-slate-600 text-xs italic mb-2')
                            _csi = _compute_self_awareness_index(d)
                            if _csi is None:
                                ui.label('Insufficient data yet to compute.').classes(
                                    'text-slate-500 text-xs italic'
                                )
                            else:
                                score, components = _csi
                                _color = 'green' if score >= 0.6 else 'yellow' if score >= 0.3 else 'gray'
                                with ui.row().classes('items-center gap-2 mb-2'):
                                    ui.label(f'{score:.2f}').classes('text-2xl font-mono')
                                    _badge('/ 1.00', _color)
                                _pct_bar(score, _color)
                                ui.separator().classes('my-3 border-slate-700')
                                ui.label('Components').classes('text-slate-400 text-xs mb-1 block')
                                for name, val, weight in components:
                                    ui.label(
                                        f"• {name}: {val:.2f}  (weight {weight:.0%})"
                                    ).classes('text-slate-300 text-xs font-mono')

            except Exception as _panel_err:
                    logger.warning(
                        f"[CognitiveDashboard] Panel render error: {_panel_err}",
                        exc_info=True
                    )
                    with content_col:
                        ui.label(
                            f'⚠ Dashboard render error: {type(_panel_err).__name__}: {_panel_err}'
                        ).classes('text-yellow-400 text-sm p-4 font-mono')
                        ui.label(
                            'A panel failed to render — likely a transient data read issue. '
                            'Click "Refresh now" below, or wait for the next auto-refresh.'
                        ).classes('text-slate-500 text-xs px-4 pb-4')
        # ── Initial render ────────────────────────────────────────────────
        import asyncio as _asyncio
        try:
            data = await _asyncio.to_thread(_fetch_all)
        except Exception as _fe:
            logger.warning(f"[CognitiveDashboard] Initial fetch error: {_fe}")
            data = {}
        try:
            await render(data)
        except Exception as _ie:
            logger.warning(f"[CognitiveDashboard] Initial render error: {_ie}")

        # ── Auto-refresh ──────────────────────────────────────────────────
        _timer_ref  = [None]
        _last_data  = [data]   # keep last successful fetch as fallback

        async def do_refresh():
            try:
                client = ui.context.client
                if not getattr(client, 'id', None):
                    if _timer_ref[0]: _timer_ref[0].cancel()
                    return
            except Exception:
                if _timer_ref[0]: _timer_ref[0].cancel()
                return
            try:
                fresh = await _asyncio.to_thread(_fetch_all)
                if fresh:  # only update fallback when fetch returned real data
                    _last_data[0] = fresh
                else:
                    fresh = _last_data[0]  # use last good data if fetch empty
            except Exception as _fe:
                logger.debug(f"[CognitiveDashboard] Fetch error (using last data): {_fe}")
                fresh = _last_data[0]   # render with previous data rather than blank
            try:
                await render(fresh)
            except Exception as _re:
                logger.debug(f"[CognitiveDashboard] Render error: {_re}")

        async def _on_disconnect():
            if _timer_ref[0]:
                try: _timer_ref[0].cancel()
                except Exception: pass

        ui.context.client.on_disconnect(_on_disconnect)

        refresh_btn = ui.button(
            'Refresh now', icon='refresh',
            on_click=do_refresh,
        ).props('flat').classes('text-slate-400 text-sm mt-4')

        _timer_ref[0] = ui.timer(REFRESH_INTERVAL, do_refresh)
