import warnings, logging, os, time, threading
warnings.filterwarnings('ignore')
os.chdir(r'C:/Users/frede2/test/lumina_v110_temp')

# Build the REAL stack (same path the app uses)
t0 = time.time()
from core.state import _init_persona
persona = _init_persona()
print(f"persona built in {time.time()-t0:.1f}s, ready={getattr(persona,'is_ready',None)}")

org = getattr(persona, '_organism', None)
ai  = getattr(org, 'ai_system', None)
print(f"organism={org is not None}, ai_system={ai is not None}")

# ── Test 1: NarrativeCompression (LLM starvation fix) ────────────────────────
print("\n=== NarrativeCompression._run() ===")
nc = None
try:
    from cognition.narrative_compression import NarrativeCompression
    nc = NarrativeCompression(org, ai)
    print("life_story chapters:", len(getattr(getattr(org,'narrative_identity',None),'life_story',[]) or []))
    # Run on a thread (as the loop does), then join
    th = threading.Thread(target=nc._run, daemon=True)
    th.start(); th.join(timeout=60)
    print("status:", nc.status())
except Exception as e:
    import traceback; traceback.print_exc()

# ── Test 2: LongHorizonPlanner (unblocked by Aspirations fix) ────────────────
print("\n=== LongHorizonPlanner._run() ===")
try:
    from cognition.long_horizon_planner import LongHorizonPlanner
    # Ensure WSDM exists (it's lazy in the loop)
    loop = getattr(org, '_loop', None)
    if loop and getattr(loop, '_world_self_dynamics', None) is None:
        from cognition.world_self_dynamics_model import WorldSelfDynamicsModel
        loop._world_self_dynamics = WorldSelfDynamicsModel(org, ai)
        print("constructed WSDM")
    asp = getattr(org, 'aspirational_self', None)
    print("aspirations:", len(getattr(asp,'aspirations',{}) or {}))
    lhp = LongHorizonPlanner(org, ai)
    th = threading.Thread(target=lhp._run, daemon=True)
    th.start(); th.join(timeout=60)
    print("status:", lhp.status())
except Exception as e:
    import traceback; traceback.print_exc()
