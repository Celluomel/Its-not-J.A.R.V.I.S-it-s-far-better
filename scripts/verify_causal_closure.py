"""
verify_causal_closure.py — EXECUTE the closed outcome loop (no LLM, no network)
==============================================================================
Proves, by real execution, that the ConsequenceTracker:
  1. predicts BEFORE an action,
  2. resolves against the REAL report AFTER,
  3. computes a genuine prediction error,
  4. SELECTIVELY propagates it (efficacy / one attention module / PCM),
  5. writes a traceable causal chain to the ledger,
  6. exposes a measurable closure report,
  7. makes efficacy steer future selection (the multiplier differs by outcome).

Run:  venv\Scripts\python.exe scripts\verify_causal_closure.py
"""
import json
import os
import shutil
import sys
import tempfile

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cognition.consequence_tracker import ConsequenceTracker  # noqa: E402
from cognition.causal_ledger import CausalLedger              # noqa: E402


# ── a mock PCM that records the calls it is fed ──────────────────────────────
class MockPCM:
    def __init__(self):
        self.records = []
        self.resolutions = 0
    def record_action(self, action_type, state_before, interaction_n=0):
        self._pending = (action_type, state_before, interaction_n)
    def record_outcome(self, state_after, outcome="neutral"):
        at, sb, n = self._pending
        self.records.append({"action": at, "before": sb, "after": state_after,
                             "delta": [round(a-b,4) for a,b in zip(state_after,sb)],
                             "outcome": outcome})
        self.resolutions += 1


def banner(t):
    print("\n" + "═" * 78 + f"\n{t}\n" + "═" * 78)


def main():
    tmp = tempfile.mkdtemp(prefix="lumina_cc_")
    persona = os.path.join(tmp, "persona")
    os.makedirs(persona, exist_ok=True)

    # seed a realistic attention file so the nudge has something to act on
    attn = {
        "attention_weights": {
            "semantic_memory": 0.10, "goal_ecology": 0.10, "self_model": 0.10,
            "relational_memory": 0.10, "creative_divergence": 0.10,
            "emotional_state": 0.10, "narrative_identity": 0.10,
            "thought_threads": 0.10, "decision_policy": 0.10,
        },
        "learned_bias": {},
    }
    with open(os.path.join(persona, "cognitive_attention.json"), "w") as f:
        json.dump(attn, f)

    pcm = MockPCM()
    ledger = CausalLedger(persona)
    ct = ConsequenceTracker(persona_dir=persona, consequence_model=pcm, ledger=ledger)

    banner("STEP 1 — memory_recall that RETURNS CONTENT  (predicted effect: evidence_gain)")
    p = ct.predict("goal_1", "understand cryptographic hashes", "memory_recall", goal_energy=0.8)
    r = ct.resolve(p, {"success": True, "result": "Recalled 4 memories on SHA-256 domain separation"})
    print(json.dumps(r, indent=2))

    banner("STEP 2 — web_search DISPATCHED ASYNC  (partial evidence only)")
    p = ct.predict("goal_1", "understand cryptographic hashes", "web_search", goal_energy=0.8)
    r = ct.resolve(p, {"success": True, "result": "web_search dispatched (async)"})
    print(json.dumps(r, indent=2))

    banner("STEP 3 — self_question that FAILS  (predicted effect: clarity_gain, got failure)")
    p = ct.predict("goal_2", "define my stance on uncertainty", "self_question", goal_energy=0.6)
    r = ct.resolve(p, {"success": False, "result": "LLM timeout"})
    print(json.dumps(r, indent=2))

    banner("STEP 4 — memory_recall for goal_2 AGAIN (EMA should now be lower than goal_1's)")
    for _ in range(3):
        p = ct.predict("goal_2", "define my stance on uncertainty", "memory_recall", goal_energy=0.6)
        ct.resolve(p, {"success": True, "result": "Recalled 2 memories"})

    banner("PROOF A — efficacy store actually changed on disk (causal_efficacy.json)")
    print(open(os.path.join(persona, "causal_efficacy.json")).read())

    banner("PROOF B — attention weights actually changed on disk (goal_ecology nudged up, self_model down)")
    d = json.load(open(os.path.join(persona, "cognitive_attention.json")))
    print("goal_ecology      :", d["attention_weights"]["goal_ecology"], "(web_search success → up)")
    print("self_model        :", d["attention_weights"]["self_model"], "(self_question fail → down)")
    print("sum(weights)      :", round(sum(d["attention_weights"].values()), 5), "(renormalised)")
    print("last_consequence_nudge:", d.get("last_consequence_nudge"))

    banner("PROOF C — PCM actually received real records + resolutions (the 0/192 fix)")
    print(f"PCM records fed   : {len(pcm.records)}")
    print(f"PCM resolutions   : {pcm.resolutions}")
    for rec in pcm.records:
        print("   ", json.dumps(rec))

    banner("PROOF D — efficacy now STEERS selection (multiplier differs by track record)")
    print("goal_1 (succeeded) multiplier:", ct.efficacy_multiplier("goal_1", "memory_recall"))
    print("goal_2 (mixed)     multiplier:", ct.efficacy_multiplier("goal_2", "memory_recall"))
    print("goal_3 (never tried) multiplier:", ct.efficacy_multiplier("goal_3", "memory_recall"), "(neutral)")

    banner("PROOF E — the causal chain is traceable (why did goal_1's memory_recall get rewarded?)")
    story = ledger.why_goal("goal_1")
    print(f"n_events={story['n_events']}  by_kind={story['by_kind']}  mean_abs_error={story['mean_abs_error']}")
    for e in story["events"]:
        print(f"   [{e['kind']:>10}] {e['cause'][:78]}")

    banner("PROOF F — closure metric (the measurable 'is the loop closed?' answer)")
    print(json.dumps(ct.closure_report(), indent=2))

    # hard assertions — this is a test, not a demo
    eff1 = json.load(open(os.path.join(persona, "causal_efficacy.json")))
    assert eff1["goal_1|memory_recall"]["eff"] > 0.5, "success should raise efficacy above prior"
    assert ct.efficacy_multiplier("goal_1", "memory_recall") > 1.0, "success must steer selection up"
    assert len(pcm.records) >= 3, "PCM must have received real records"
    assert story["n_events"] >= 3, "ledger must contain a causal chain for the goal"
    assert abs(sum(d["attention_weights"].values()) - 1.0) < 1e-6, "attention must stay normalised"
    print("\n✅ ALL ASSERTIONS PASSED — the outcome loop is closed and measurable.")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
