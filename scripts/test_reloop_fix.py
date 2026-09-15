# -*- coding: utf-8 -*-
"""2026-09-03 — contradiction zombie-loop recurrence: verify the three fixes
against the REAL modules (no mocks of the code under test).

Run from project root:  venv\\Scripts\\python.exe scripts\\test_reloop_fix.py
"""
import json, os, sys, time, tempfile, shutil, logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)  # keep output clean; assert on behavior

BELIEF = "I am genuinely curious about the world"
PARROT = ("I was just exploring I said I am genuinely curious about the world, "
          "but responded without asking questions or exploring — I haven't "
          "actually worked out what that means and found something worth sharing")
PLAIN  = "The weather is nice today."          # no ?, no exploration words, no 'curious'
EXPL   = "Let me understand how that works, what do you think?"  # genuine exploration

results = []
def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")

from cognition.contradiction_handler import ContradictionHandler

# ── handler factory (fresh in-memory state, no disk) ─────────────────────────
def fresh_handler():
    h = ContradictionHandler.__new__(ContradictionHandler)
    import threading
    # NOTE: must be an RLock, exactly like the real __init__ (L44) —
    # _save() re-acquires self._lock from inside a locked section.
    h._lock = threading.RLock()
    h.contradictions = []
    h.pending_confrontations = []
    h._path = None          # _save() will raise+log; harmless for these tests
    return h

print("== Fix 1+2: detector (contradiction_handler.detect_contradiction) ==")

# T1: true positive preserved — plain non-exploratory response still flags
h = fresh_handler()
c = h.detect_contradiction([BELIEF], PLAIN, interaction_count=100,
                           world_model=None, identity_drift=0.0)
check("T1 plain non-exploratory response still fires (true positive kept)",
      c is not None and c.claimed_belief == BELIEF)

# T2: parrot guard — the canned confrontation reply must NOT re-trigger
h = fresh_handler()
c = h.detect_contradiction([BELIEF], PARROT, interaction_count=100,
                           world_model=None, identity_drift=0.0)
check("T2 parroted confrontation reply does NOT re-trigger (Fix 2)", c is None)

# T3: genuine exploration still does not fire (was never a contradiction)
h = fresh_handler()
c = h.detect_contradiction([BELIEF], EXPL, interaction_count=100,
                           world_model=None, identity_drift=0.0)
check("T3 genuine exploratory response does not fire", c is None)

# T4: pending suppression (original 2026-09-01 behaviour preserved)
h = fresh_handler()
c0 = h.detect_contradiction([BELIEF], PLAIN, interaction_count=100,
                            world_model=None, identity_drift=0.0)
assert c0 is not None
h.pending_confrontations.append(c0)          # still pending
c = h.detect_contradiction([BELIEF], PLAIN, interaction_count=130,
                           world_model=None, identity_drift=0.0)
check("T4 pending belief suppressed (dedupe preserved)", c is None)

# T5: RECENTLY-CONFRONTED suppression (the 2026-09-03 fix)
# Simulate the real confrontation path (reviser fallback): mark confronted AND
# remove from pending — so suppression can ONLY come from the cooldown.
h = fresh_handler()
c0 = h.detect_contradiction([BELIEF], PLAIN, interaction_count=100,
                            world_model=None, identity_drift=0.0)
assert c0 is not None
c0.confronted = True
c0.confronted_at = time.time()               # just confronted
h.pending_confrontations = [c for c in h.pending_confrontations if c is not c0]
c = h.detect_contradiction([BELIEF], PLAIN, interaction_count=130,
                           world_model=None, identity_drift=0.0)
check("T5 recently-confronted belief suppressed for 48h (Fix 1)", c is None)

# T6: cooldown EXPIRES — confronted >48h ago may be re-raised (bounded, not eternal)
h = fresh_handler()
c0 = h.detect_contradiction([BELIEF], PLAIN, interaction_count=100,
                            world_model=None, identity_drift=0.0)
assert c0 is not None
c0.confronted = True
c0.confronted_at = time.time() - 49 * 3600   # 49h ago
h.pending_confrontations = [c for c in h.pending_confrontations if c is not c0]
c = h.detect_contradiction([BELIEF], PLAIN, interaction_count=130,
                           world_model=None, identity_drift=0.0)
check("T6 >48h-old confrontation may be re-raised (bounded re-raise)", c is not None)

# T7: other beliefs unaffected by suppression of one belief
h = fresh_handler()
c0 = h.detect_contradiction(["I am genuinely curious about the world"], PLAIN,
                            interaction_count=100, world_model=None, identity_drift=0.0)
c0.confronted = True; c0.confronted_at = time.time()
h.contradictions.append(c0)
other = "I am brave in the face of the unknown"
c = h.detect_contradiction([other], "I'm scared, I'm afraid and nervous about this",
                           interaction_count=130, world_model=None, identity_drift=0.0)
# brave branch fires on >=1 retreat word; ensure it was NOT suppressed
check("T7 unrelated belief not suppressed by another's cooldown",
      c is not None and c.claimed_belief == other)

# ── Fix 3: reviser source erosion ────────────────────────────────────────────
print("\n== Fix 3: reviser erodes the SOURCE stores (not just narrative_identity) ==")

from cognition.contradiction_belief_reviser import ContradictionBeliefReviser
from cognition.self_concept import SelfBelief

tmp = tempfile.mkdtemp(prefix="reloop_test_")
persona = os.path.join(tmp, "persona")
os.makedirs(persona, exist_ok=True)
ident_path = os.path.join(persona, "identity.json")
seed_conf = 0.95
json.dump({"version": "v3", "updated_at": "2026-09-03T00:00:00+00:00",
           "beliefs": {
               "expressed_curious": {"statement": BELIEF, "confidence": seed_conf,
                                     "category": "expressed_belief"},
               "thread_learning_x": {"statement":
                                     f"Explored 'i am genuinely curious about the world' "
                                     "extensively without full resolution",
                                     "confidence": 0.75, "category": "thread_learning"},
               "unrelated": {"statement": "I like warm evenings by the river",
                             "confidence": 0.9, "category": "insight"}},
          }, open(ident_path, "w", encoding="utf-8"))

class _SC:  # minimal stand-in for the SelfConceptSystem interface used
    def __init__(self):
        self._beliefs = {
            "expressed_curious": SelfBelief(name="expressed_curious",
                                            statement=BELIEF, confidence=seed_conf,
                                            valence="neutral", source="expressed"),
            "unrelated": SelfBelief(name="unrelated",
                                    statement="I like warm evenings by the river",
                                    confidence=0.9, valence="positive", source="x"),
        }
        self.saved = 0
    def _save(self):
        self.saved += 1

class _Cfg:
    self_concept_path = os.path.join(persona, "self_concept.json")

class _AI:
    pass
ai = _AI(); ai.self_concept = _SC(); ai.config = _Cfg()

rev = ContradictionBeliefReviser(organism=None, ai_system=ai)
rev._erode_source_belief(BELIEF)

sc_b = ai.self_concept._beliefs["expressed_curious"]
check("T8 self_concept seed belief eroded 0.95 → 0.57",
      abs(sc_b.confidence - 0.57) < 1e-6)
check("T9 violation recorded on the seed belief", sc_b.violations == 1)
check("T10 self_concept persisted (_save called)", ai.self_concept.saved == 1)

ident = json.load(open(ident_path, encoding="utf-8"))
b = ident["beliefs"]["expressed_curious"]
check("T11 identity.json seed belief eroded 0.95 → 0.57",
      abs(b["confidence"] - 0.57) < 1e-6 and b.get("updated_at"))
check("T12 same-topic thread learning eroded (cluster weakens together)",
      abs(ident["beliefs"]["thread_learning_x"]["confidence"] - 0.45) < 1e-6)
check("T13 unrelated belief untouched (0.9 in both stores)",
      ident["beliefs"]["unrelated"]["confidence"] == 0.9
      and ai.self_concept._beliefs["unrelated"].confidence == 0.9)
# floor at 0.20
rev._erode_source_belief(BELIEF); rev._erode_source_belief(BELIEF)
# 0.57 → 0.342 → 0.2052 → floor check: after 3 erosions from 0.95: .57/.342/.2052 (still >.2)
# one more: .123 → floor .20
rev._erode_source_belief(BELIEF)
check("T14 confidence floor 0.20 respected",
      ai.self_concept._beliefs["expressed_curious"].confidence >= 0.20 - 1e-9)

shutil.rmtree(tmp, ignore_errors=True)

fails = [n for n, ok in results if not ok]
print(f"\n{'='*60}\n{len(results)-len(fails)}/{len(results)} passed"
      + (f"  — FAILURES: {fails}" if fails else "  — ALL PASS"))
sys.exit(1 if fails else 0)
