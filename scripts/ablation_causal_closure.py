"""
ablation_causal_closure.py — the controlled-ablation experiment
===============================================================
Runs the SAME fixed scenario through the causal-closure stack with each layer
toggled OFF in turn, and prints the CONTRIBUTION TABLE — which layers are
causally active (ablating them changes behaviour) vs trace-only (ablating
them changes the trace, not the behaviour).

This is the deliverable the analysis asked for:
    "an architecture whose claims can be experimentally demonstrated
     from its own causal traces."

Pure: no LLM, no network. Every number is produced by real execution of the
real layers (WorkspaceBias / ConsequenceTracker / IdentityGrounding /
CausalLedger) under hermetic, isolated conditions.

Run:  venv\Scripts\python.exe scripts\ablation_causal_closure.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cognition.ablation import run_conditions  # noqa: E402


def banner(t):
    print("\n" + "═" * 82 + f"\n{t}\n" + "═" * 82)


def main():
    R = run_conditions(ROOT)

    banner("CONTRIBUTION TABLE  (same fixed scenario; each row toggles one layer OFF)")
    cols = ["condition", "selection", "efficacy(n,eff)", "override", "focus_stab",
            "identity(demo)", "ledger_events"]
    widths = [14, 11, 15, 10, 11, 14, 13]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for name in ["FULL", "NO_BIAS", "NO_OUTCOME", "NO_IDENTITY", "NO_LEDGER", "NO_EVERYTHING"]:
        r = R[name]
        eff = r["efficacy"]
        row = [
            name,
            r["selection_winner"],
            f"({eff['n']},{eff['eff']})",
            f"{r['override_rate']:.2f}" if r["override_rate"] is not None else "-",
            f"{r['focus_stability']:.2f}",
            str(r["identity_demonstrated"]),
            str(r["n_ledger_events"]),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))

    # ── VERDICT: which layers are causally active? ───────────────────────────
    F, NB, NO, NI, NL = (R["FULL"], R["NO_BIAS"], R["NO_OUTCOME"],
                         R["NO_IDENTITY"], R["NO_LEDGER"])
    verdict = {
        "BIAS": (
            "ACTIVE",
            f"selection flips {F['selection_winner']}→{NB['selection_winner']}; "
            f"override_rate {F['override_rate']}→{NB['override_rate']}; "
            f"focus_stab {F['focus_stability']}→{NB['focus_stability']}"
        ),
        "OUTCOME": (
            "ACTIVE",
            f"efficacy n {F['efficacy']['n']}→{NO['efficacy']['n']}, "
            f"eff {F['efficacy']['eff']}→{NO['efficacy']['eff']}; "
            f"identity_demonstrated {F['identity_demonstrated']}→{NO['identity_demonstrated']}"
        ),
        "IDENTITY": (
            "ACTIVE",
            f"identity_demonstrated {F['identity_demonstrated']}→{NI['identity_demonstrated']} "
            f"(efficacy unchanged: {F['efficacy']['n']}→{NI['efficacy']['n']})"
        ),
        "LEDGER": (
            "TRACE-ONLY",
            f"events {F['n_ledger_events']}→{NL['n_ledger_events']}, but selection "
            f"{F['selection_winner']}→{NL['selection_winner']} and efficacy "
            f"{F['efficacy']['n']}→{NL['efficacy']['n']} unchanged"
        ),
    }

    banner("VERDICT — causal role of each layer (from measured deltas)")
    for layer, (role, why) in verdict.items():
        print(f"  {layer:<9} {role:<10}  {why}")

    # ── HARD CHECKS (the ablation must actually discriminate) ────────────────
    banner("ASSERTIONS (the ablation must show real, non-zero deltas)")
    assert F["selection_winner"] == "aligned",    "FULL: bias must steer selection to aligned"
    assert NB["selection_winner"] == "misaligned", "NO_BIAS: tie must fall to misaligned (no steering)"
    assert F["efficacy"]["n"] == 5 and (F["efficacy"]["eff"] or 0) >= 0.65, "FULL: outcome loop must learn (n=5, strong)"
    assert NO["efficacy"]["n"] == 0,              "NO_OUTCOME: no efficacy learning"
    assert F["identity_demonstrated"] == 1,       "FULL: identity must demonstrate the learned capability"
    assert NO["identity_demonstrated"] == 0,      "NO_OUTCOME: no demonstrated capability (no evidence)"
    assert NI["identity_demonstrated"] == 0,      "NO_IDENTITY: self-model not grounded"
    assert NI["efficacy"]["n"] == 5,              "NO_IDENTITY: efficacy intact (identity is a consumer)"
    assert F["n_ledger_events"] > 0,              "FULL: causal trace present"
    assert NL["n_ledger_events"] == 0,            "NO_LEDGER: trace disabled"
    assert NL["selection_winner"] == F["selection_winner"] and NL["efficacy"]["n"] == F["efficacy"]["n"], \
        "NO_LEDGER: behaviour must be UNCHANGED (ledger is trace, not actuator)"
    assert F["override_rate"] > NB["override_rate"], "BIAS: override measurement only exists with bias on"

    print("\n  [x] BIAS steers selection + measures override   (ACTIVE)")
    print("  [x] OUTCOME loop learns efficacy + feeds identity (ACTIVE)")
    print("  [x] IDENTITY grounds the self in that evidence   (ACTIVE)")
    print("  [x] LEDGER traces without altering behaviour     (TRACE-ONLY, by design)")
    print("  [x] Each delta is non-zero and correctly attributed to its layer")

    print("\n✅ ABLATION PASSED — the causal-closure stack is demonstrably active,")
    print("   and every claim is now backed by a measured delta from its own traces.")


if __name__ == "__main__":
    main()
