"""
verify_workspace_bias.py — EXECUTE the workspace-bias + override-rate layer
===========================================================================
Proves, by real execution (no LLM, no network), that:
  1. COMMIT   turns a workspace winner into an active embedded commitment.
  2. BIAS     causally steers SELECTION — the aligned goal wins, the
              misaligned one loses (A/B: with vs without the commitment).
  3. OVERRIDE is detected against REAL behaviour (action goal + LLM response),
              recorded, costed (focus_stability drops), and aggregated.
  4. override_rate + focus_stability are the measurable causal-closure signal.
  5. the causal chain is traceable in the ledger.

Run:  venv\Scripts\python.exe scripts\verify_workspace_bias.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cognition.workspace_bias import WorkspaceBias  # noqa: E402
from cognition.causal_ledger import CausalLedger   # noqa: E402

# ── deterministic mock embedder: controlled cosines ─────────────────────────
# focus = "quantum physics" -> [1,0,0]
#   aligned   "quantum computing" -> [0.9,0.4358,0]  cos≈0.90  (engages bias)
#   misaligned "baking bread"     -> [0,1,0]         cos=0.00  (no bias)
VECTORS = {
    "quantum physics":   [1.0, 0.0, 0.0],
    "quantum computing": [0.9, 0.4358, 0.0],
    "baking bread":      [0.0, 1.0, 0.0],
    "bread recipes":     [0.1, 0.95, 0.0],
}


class MockEmbedder:
    def encode(self, text):
        return VECTORS.get(text, [0.0, 0.0, 0.0])


def banner(t):
    print("\n" + "═" * 78 + f"\n{t}\n" + "═" * 78)


class FakeGoal:
    def __init__(self, topic):
        self.id = topic
        self.topic = topic
        self.priority = 0.5  # equal base priority for all candidates (A/B fairness)


def select_with_bias(goals, wb):
    """Mirror of goal_action_executor._select_eligible_goal scoring:
    score = base_priority * workspace_bias_multiplier(topic)."""
    scored = [(g.priority * wb.bias_multiplier(g.topic), g, wb.bias_multiplier(g.topic))
              for g in goals]
    scored.sort(key=lambda x: -x[0])
    return scored


def main():
    tmp = tempfile.mkdtemp(prefix="lumina_wb_")
    persona = os.path.join(tmp, "persona")
    os.makedirs(persona, exist_ok=True)

    emb = MockEmbedder()
    ledger = CausalLedger(persona)
    wb = WorkspaceBias(persona_dir=persona, embedder=emb, ledger=ledger)

    aligned   = FakeGoal("quantum computing")
    misaligned = FakeGoal("baking bread")
    goals = [aligned, misaligned]

    banner("STEP 0 — A/B selection WITHOUT a commitment (bias must be neutral)")
    s0 = select_with_bias(goals, wb)
    for score, g, mult in s0:
        print(f"   {g.topic:20s} bias={mult:.4f}  score={score:.4f}")
    assert s0[0][2] == 1.0 and s0[1][2] == 1.0, "no commitment => neutral bias for all"

    banner("STEP 1 — COMMIT the workspace winner (focus: 'quantum physics')")
    winner = {"topic": "quantum physics", "type": "goal",
              "name": "quantum_physics", "score": 0.82}
    c = wb.commit(winner, cycle_id=42)
    print(json.dumps(c, indent=2))
    assert c["committed"] and c["has_vector"], "commitment must be embedded"

    banner("STEP 2 — A/B selection WITH the commitment (bias must steer the aligned goal to win)")
    s1 = select_with_bias(goals, wb)
    for score, g, mult in s1:
        print(f"   {g.topic:20s} bias={mult:.4f}  score={score:.4f}")
    assert s1[0][1] is aligned, "aligned goal must win when the focus is committed"
    assert s1[0][2] > 1.0, "aligned candidate must get a real bias boost"
    assert s1[1][2] == 1.0, "misaligned candidate must stay neutral"
    print(f"   → WINNER: {s1[0][1].topic}  (bias flipped the selection: "
          f"{s0[0][1].topic if s0[0][1] is misaligned else 'tie'} → {s1[0][1].topic})")

    banner("STEP 3 — OVERRIDE: action follows the focus (COMPLY)")
    r_comply = wb.check_override("quantum computing", context="action=memory_recall")
    print(json.dumps(r_comply, indent=2))
    assert r_comply["checked"] and not r_comply["overridden"], "aligned action must comply"

    banner("STEP 4 — OVERRIDE: LLM response DRIFTS from the focus (OVERRIDE)")
    r_over = wb.check_override("baking bread", context="llm_response")
    print(json.dumps(r_over, indent=2))
    assert r_over["checked"] and r_over["overridden"], "drifted response must be an override"

    banner("STEP 5 — the metric: override_rate + focus_stability (the causal-closure signal)")
    m = wb.override_rate()
    print(json.dumps(m, indent=2))
    assert m["override_rate"] == 0.5, "1 override of 2 checks => rate 0.5"
    assert m["focus_stability"] < 1.0, "an override must cost focus_stability"
    assert m["active_focus"] == "quantum physics"

    banner("STEP 6 — causal chain in the ledger (commitment -> comply -> override)")
    evs = ledger.recent(12)
    for e in evs:
        print(f"   [{e['kind']:>12}] {e['cause'][:80]}")
    kinds = [e["kind"] for e in evs]
    assert "commitment" in kinds and "comply" in kinds and "override" in kinds, \
        "ledger must contain the full causal chain"

    banner("PROOF — bias is BOUNDED (never hard-excludes, preserves competition)")
    print("   max possible bias multiplier =", wb.bias_multiplier("quantum physics"),
          "(focus itself, cos=1.0)")
    print("   min bias for misaligned      =", wb.bias_multiplier("baking bread"), "(neutral 1.0)")
    print("   → aligned boosted, others untouched: selection stays a competition, not a gate.")

    print("\n✅ ALL ASSERTIONS PASSED — workspace bias steers selection; overrides are measured.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
