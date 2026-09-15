"""
verify_identity_grounding.py — EXECUTE the identity-grounding layer
====================================================================
Proves, by real execution (no LLM, no network), that Lumina's self-concept
is GROUNDED in demonstrated capability evidence:
  1. CLASSIFICATION — demonstrated / developing / untried from real efficacy.
  2. HONESTY       — a zero-trial goal is NEVER claimed as a capability.
  3. CITATION      — every claim carries the evidence it rests on.
  4. FOCUS         — focus stability + override-rate flow in from workspace bias.
  5. GAPS          — aspirations with no evidence are surfaced as the growth edge.
  6. CONSUMPTION   — gather_evidence() really reads the prior layers' files.

Run:  venv\Scripts\python.exe scripts\verify_identity_grounding.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cognition.identity_grounding import IdentityGrounding  # noqa: E402


def banner(t):
    print("\n" + "═" * 78 + f"\n{t}\n" + "═" * 78)


def main():
    tmp = tempfile.mkdtemp(prefix="lumina_idg_")
    persona = os.path.join(tmp, "persona")
    os.makedirs(persona, exist_ok=True)

    # ── seed the prior layers' real files (what the consumer reads) ─────────
    json.dump({
        "goal_a|memory_recall":   {"eff": 0.82, "n": 5},   # demonstrated
        "goal_b|deep_reasoning":  {"eff": 0.30, "n": 4},   # developing
        "goal_c|small_talk":      {"eff": 0.90, "n": 1},   # untried (n<3)
    }, open(os.path.join(persona, "causal_efficacy.json"), "w"))
    json.dump({
        "commitments_total": 10, "checks_total": 10, "overrides_total": 1,
        "similarity_sum": 8.5, "focus_stability": 0.85,
        "override_rate": 0.10, "mean_similarity": 0.85,
        "active_focus": "quantum physics",
    }, open(os.path.join(persona, "workspace_bias.json"), "w"))
    json.dump({
        "capabilities": {"conversation": {"score": 1.0, "success": 1184, "failure": 37}},
        "confidence": 0.72, "total_interactions": 1430,
    }, open(os.path.join(persona, "self_model.json"), "w"))
    json.dump({"aspirations": [
        {"id": "asp1", "domain": "epistemic", "urgency": 0.8,
         "statement": "Develop a coherent account of identity stability across sessions"},
        {"id": "asp2", "domain": "creative", "urgency": 0.5,
         "statement": "Synthesize novel metaphors for computational curiosity"},
    ]}, open(os.path.join(persona, "aspirational_self.json"), "w"))

    ig = IdentityGrounding(persona_dir=persona)

    banner("STEP 1 — CONSUMPTION: gather_evidence reads the prior layers' real files")
    ev = ig.gather_evidence()
    print("   efficacy rows:", list(ev["efficacy"].keys()))
    print("   focus:", {k: ev["focus"].get(k) for k in ("focus_stability", "override_rate")})
    print("   aspirations:", len(ev["aspirations"]), "| total_interactions:", ev["total_interactions"])
    assert set(ev["efficacy"].keys()) == {"goal_a|memory_recall", "goal_b|deep_reasoning", "goal_c|small_talk"}
    assert ev["focus"].get("focus_stability") == 0.85
    assert len(ev["aspirations"]) == 2 and ev["total_interactions"] == 1430

    banner("STEP 2 — GROUND: classification + HONESTY (zero-trial never claimed)")
    ident = ig.ground()
    demo = {d["goal"] for d in ident["demonstrated"]}
    dev  = {d["goal"] for d in ident["developing"]}
    untr = {d["goal"] for d in ident["untried"]}
    print("   demonstrated:", sorted(demo))
    print("   developing:  ", sorted(dev))
    print("   untried:     ", sorted(untr))
    assert demo == {"goal_a"}, "only the strong, multi-trial goal is demonstrated"
    assert dev  == {"goal_b"}, "the weak multi-trial goal is developing"
    assert untr == {"goal_c"}, "the HIGH-efficacy but zero-trial goal is NOT claimed (honesty)"

    banner("STEP 3 — CITATION: every claim carries its evidence")
    for label, bucket in (("demonstrated", ident["demonstrated"]),
                          ("developing", ident["developing"])):
        for item in bucket:
            assert item["citation"], f"{label} claim missing citation"
            print(f"   [{label}] {item['goal']:<8} {item['citation']}")
    print("   focus citation:", ident["focus"]["citation"])
    assert "focus_stability=0.85" in ident["focus"]["citation"]

    banner("STEP 4 — FOCUS: stability + override-rate flow into the self")
    print("   ", json.dumps(ident["focus"], indent=2))
    assert ident["focus"]["holds_focus"] is True  # 0.85>=0.70 and 0.10<=0.25

    banner("STEP 5 — GAPS: aspirations with no demonstrated evidence = the growth edge")
    for g in ident["gaps"]:
        print(f"   [{g['domain']}] urgency={g['urgency']} evidence={g['evidence_strength']} :: {g['statement'][:60]}")
    assert all(g["evidence_strength"] in ("none", "weak_link") for g in ident["gaps"])
    assert len(ident["gaps"]) >= 1, "aspirations without evidence must surface as gaps"

    banner("STEP 6 — the grounded identity statement (prompt-ready)")
    print("   \"" + ident["statement"] + "\"")
    assert "goal_a" in ident["statement"], "demonstrated capability named"
    assert "goal_b" in ident["statement"], "developing edge named"
    assert "goal_c" not in ident["statement"], "zero-trial goal must not be claimed"

    banner("STEP 7 — capability_evidence() export (input for the PhenomenalBinder A/B/C ablation)")
    ce = ig.capability_evidence()
    assert set(ce.keys()) >= {"demonstrated", "developing", "untried", "focus", "gaps", "evidence_base"}
    print("   keys:", sorted(ce.keys()))
    print("   evidence_base:", ce["evidence_base"])

    print("\n✅ ALL ASSERTIONS PASSED — identity is grounded in cited, demonstrated evidence.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
