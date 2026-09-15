"""debug_bridge.py — diagnostic for brain bridge module reads"""
import time, math

def _clamp(v): return max(0.0, min(1.0, float(v)))
def _log_scale(value, max_val):
    if value <= 0: return 0.0
    return _clamp(math.log1p(value) / math.log1p(max_val))

def debug_organism(o):
    print("\n" + "="*60)
    print("BRAIN BRIDGE DIAGNOSTIC v2")
    print("="*60)
    ai = getattr(o, 'ai_system', None)

    # Top-level organism attributes
    for attr in ['thought_stream','narrative_identity','predictive_mind','tension_engine',
                 'goal_ecology','energy','curiosity','pressure','observatory',
                 'world_model','meta_cognition','workspace','event_bus']:
        val = getattr(o, attr, None)
        print(f"  organism.{attr:<25} {'✅ '+type(val).__name__ if val else '❌ NOT FOUND'}")

    # ai_system attributes
    print()
    for attr in ['relational_memory','_active_user_id','memory_system']:
        val = getattr(ai, attr, None) if ai else None
        print(f"  ai_system.{attr:<25} {'✅ '+type(val).__name__ if val else '❌ NOT FOUND'}")

    print()
    # Deep reads
    ts = getattr(o, 'thought_stream', None)
    if ts:
        try:
            recent = ts.recent(8); now = time.time()
            fresh = sum(1 for t in recent if now - getattr(t,'timestamp',now) < 90)
            print(f"  thought_stream.recent(8)      → {len(recent)} thoughts, {fresh} fresh (<90s)")
        except Exception as e: print(f"  thought_stream.recent(8)      → ERROR: {e}")

    ni = getattr(o, 'narrative_identity', None)
    if ni:
        try:
            s = ni.summary()
            print(f"  narrative_identity.summary()  → chapters={s.get('chapters',0)} beliefs={s.get('beliefs',0)}")
        except Exception as e: print(f"  narrative_identity.summary()  → ERROR: {e}")

    pm = getattr(o, 'predictive_mind', None)
    if pm:
        try:
            m = pm.stability_metrics()
            print(f"  predictive_mind.stability     → surprise_index={m.get('surprise_index',0):.3f} novelty={m.get('novelty',0):.3f}")
        except Exception as e: print(f"  predictive_mind.stability     → ERROR: {e}")

    te = getattr(o, 'tension_engine', None)
    if te:
        try:
            tv = te.current(); arousal = tv.overall_arousal()
            print(f"  tension_engine.current()      → overall_arousal={arousal:.3f}")
        except Exception as e: print(f"  tension_engine.current()      → ERROR: {e}")

    cu = getattr(o, 'curiosity', None)
    if cu:
        for meth in ('global_level','level','get_level'):
            fn = getattr(cu, meth, None)
            if callable(fn):
                try: print(f"  curiosity.{meth}()         → {fn():.3f}")
                except Exception as e: print(f"  curiosity.{meth}()         → ERROR: {e}")
                break

    pressure = getattr(o, 'pressure', None)
    if pressure:
        try:
            for k, v in getattr(pressure,'reservoirs',{}).items():
                try: print(f"  pressure.{k:<20}   → {v.effective_pressure():.3f}")
                except: pass
        except Exception as e: print(f"  pressure.reservoirs           → ERROR: {e}")

    ge = getattr(o, 'goal_ecology', None); en = getattr(o, 'energy', None)
    if ge and en:
        try:
            d = ge.dominant_drive(en.level())
            print(f"  goal_ecology.dominant_drive   → urgency={getattr(d,'urgency',0):.3f}")
        except Exception as e: print(f"  goal_ecology.dominant_drive   → ERROR: {e}")

    wm = getattr(o, 'world_model', None)
    if wm:
        try:
            s = wm.summary() if callable(getattr(wm,'summary',None)) else {}
            print(f"  world_model.summary()         → topics={s.get('topics',s.get('topic_count','?'))}")
        except Exception as e: print(f"  world_model.summary()         → ERROR: {e}")

    rm = getattr(ai, 'relational_memory', None) if ai else None
    uid = getattr(ai, '_active_user_id', None) if ai else None
    if rm and uid:
        try:
            rel = rm.get_or_create(uid)
            print(f"  relational_memory[{uid}]   → interactions={getattr(rel,'interaction_count',0)}")
        except Exception as e: print(f"  relational_memory             → ERROR: {e}")

    obs = getattr(o, 'observatory', None)
    if obs:
        try:
            snap = obs.tick()
            ccs = getattr(snap,'ccs',None) if snap else None
            print(f"  observatory.tick().ccs        → {ccs}")
        except Exception as e: print(f"  observatory.tick()            → ERROR: {e}")

    mc = getattr(o, 'meta_cognition', None)
    if mc:
        try:
            s = mc.summary() if callable(getattr(mc,'summary',None)) else {}
            print(f"  meta_cognition.summary()      → cycles={s.get('reflection_cycles',s.get('cycles','?'))}")
        except Exception as e: print(f"  meta_cognition.summary()      → ERROR: {e}")

    print("="*60 + "\n")
