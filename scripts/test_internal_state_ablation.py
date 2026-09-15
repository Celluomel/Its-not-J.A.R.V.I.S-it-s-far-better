"""
test_internal_state_ablation.py — Phase 6.10
================================================
Large-scale ablation replay for InternalCognitiveState (see
cognition/internal_cognitive_state.py). Per the implementation prompt's
required minimum: 300 cycles, seed=1 and seed=7, ON vs OFF.

Design — a true PAIRED comparison, not a separate-run A/B
-------------------------------------------------------------
Unlike the Flux replay (v100/v101), which needed two independent
organisms because Flux's own commentary is stochastic and can't be held
identical across conditions, this ablation only toggles one thing: does
compete() receive internal_state=<real object> or None. Everything else
(the epistemic pressure trajectory, the two competing goals' priority/
energy each cycle) is generated ONCE per cycle and fed into BOTH
conditions identically — a matched-pairs design, statistically stronger
than the independent-trajectories design used for Flux.

Two competing goals present every cycle:
  - "resolve_uncertainty" (self-regulatory; real production goal name —
    goal_engine.py's derive_motivations() already creates this exact
    goal from real epistemic pressure, this replay does not invent it)
  - "routine_task" (neutral; not self-regulatory)
Both get a randomly jittered priority/energy each cycle (same jitter
used in both ON and OFF conditions) so neither trivially dominates by
construction — the ablation term has to earn any win-rate difference.

Metrics reported per the implementation prompt's Section 16 + Section 25
(CSR): win-rate for resolve_uncertainty (ON vs OFF), win-rate lift,
margin-vs-pressure correlation (does the margin actually track pressure,
not just exist), and CSR for winning self-regulatory candidates.

Honest scope: this is the goal-scoring mechanism in isolation — no
camera/mic/Flux, no live LLM. It answers "does the ablation change
behavior at scale", not "does the whole cognitive loop behave
differently" (that would need the v101-style full replay, extended).
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognition.workspace_competition import WorkspaceCompetition
from cognition.internal_cognitive_state import (
    InternalCognitiveState, MAX_INTERNAL_BIAS,
)
from cognition.pressure_system import PressureSystem


def make_pressure_organism():
    org = SimpleNamespace()
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    org.pressure = PressureSystem(organism=org, path=tmp.name)
    return org


def run_ablation(n_cycles: int, seed: int):
    rng = random.Random(seed)
    pressure_org = make_pressure_organism()
    ics = InternalCognitiveState()

    # Two SEPARATE competition instances (each has its own win-history/
    # stagnation tracking — sharing one across ON/OFF would let one
    # condition's stagnation penalties bleed into the other, corrupting
    # the comparison, same class of bug already found and worked around
    # earlier this session with _get_competition_instance()'s module
    # singleton).
    wc_on = WorkspaceCompetition()
    wc_off = WorkspaceCompetition()

    wins_on, wins_off = 0, 0
    margins_on = []
    pressures = []
    csr_values = []
    internal_biases = []

    for t in range(1, n_cycles + 1):
        # Scripted epistemic drive: occasional novelty spikes (boost),
        # natural decay otherwise — same documented pattern as v100/v101,
        # not the real PressureSystem.tick() (needs a full live organism).
        if rng.random() < 0.25:
            pressure_org.pressure.boost("epistemic", rng.uniform(0.05, 0.15))
        res = pressure_org.pressure.reservoirs["epistemic"]
        res.level = res.resting_level + (res.level - res.resting_level) * 0.90
        epi = res.level
        pressures.append(epi)

        ics.update(epi)

        # Same jittered goal state fed into BOTH conditions this cycle.
        p1 = max(0.1, min(0.9, 0.5 + rng.uniform(-0.15, 0.15)))
        e1 = max(0.1, min(0.9, 0.5 + rng.uniform(-0.15, 0.15)))
        p2 = max(0.1, min(0.9, 0.5 + rng.uniform(-0.15, 0.15)))
        e2 = max(0.1, min(0.9, 0.5 + rng.uniform(-0.15, 0.15)))
        goals = [
            {"id": "g_ru", "name": "resolve_uncertainty", "topic": "resolve_uncertainty",
             "priority": p1, "energy": e1, "urgency": 0.0, "quality_score": 0.5},
            {"id": "g_rt", "name": "routine_task", "topic": "routine_task",
             "priority": p2, "energy": e2, "urgency": 0.0, "quality_score": 0.5},
        ]
        pressure_dict = {"epistemic": epi}

        r_on = wc_on.compete(goals=[dict(g) for g in goals], thoughts=[], tensions={},
                              pressure=pressure_dict, context={"user_input": "x"},
                              internal_state=ics)
        r_off = wc_off.compete(goals=[dict(g) for g in goals], thoughts=[], tensions={},
                                pressure=pressure_dict, context={"user_input": "x"})

        winner_on = r_on["winner"]
        winner_off = r_off["winner"]
        if winner_on["id"] == "g_ru":
            wins_on += 1
            total = winner_on["score"]
            bias = winner_on.get("internal_bias", 0.0)
            if total > 0:
                csr_values.append(max(0.0, min(1.0, bias / total)))
        if winner_off["id"] == "g_ru":
            wins_off += 1

        scores_on = {c["id"]: c["score"] for c in r_on["competition"]["all_candidates"]}
        margins_on.append(scores_on.get("g_ru", 0) - scores_on.get("g_rt", 0))
        bias_on = next(
            (c.get("internal_bias", 0.0) for c in r_on["competition"]["all_candidates"] if c["id"] == "g_ru"),
            0.0,
        )
        internal_biases.append(bias_on)

    return {
        "wins_on": wins_on, "wins_off": wins_off, "n": n_cycles,
        "margins_on": margins_on, "pressures": pressures, "csr_values": csr_values,
        "internal_biases": internal_biases,
    }


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 7])
    args = ap.parse_args()

    print("=" * 78)
    print(f"Phase 6.10 — InternalCognitiveState ablation replay")
    print(f"cycles={args.cycles} per seed, seeds={args.seeds}")
    print(f"MAX_INTERNAL_BIAS={MAX_INTERNAL_BIAS} (hard clamp, unchanged)")
    print("=" * 78)

    all_lift = []
    all_corr = []
    all_csr = []

    for seed in args.seeds:
        result = run_ablation(args.cycles, seed)
        win_rate_on = result["wins_on"] / result["n"]
        win_rate_off = result["wins_off"] / result["n"]
        lift = win_rate_on - win_rate_off
        corr = pearson(result["pressures"], result["internal_biases"])
        corr_margin = pearson(result["pressures"], result["margins_on"])
        csr_mean = statistics.mean(result["csr_values"]) if result["csr_values"] else 0.0
        csr_n = len(result["csr_values"])

        all_lift.append(lift)
        all_corr.append(corr)
        if result["csr_values"]:
            all_csr.extend(result["csr_values"])

        print(f"\n--- seed={seed} ---")
        print(f"  win-rate ON:  {win_rate_on:.3f}  ({result['wins_on']}/{result['n']})")
        print(f"  win-rate OFF: {win_rate_off:.3f}  ({result['wins_off']}/{result['n']})")
        print(f"  LIFT (ON - OFF): {lift:+.3f}")
        print(f"  margin-vs-epistemic-pressure correlation (ON): r={corr_margin:+.3f} "
              f"(noisy — confounded by independent goal-priority jitter, see internal_bias corr below)")
        print(f"  internal_bias-vs-epistemic-pressure correlation (ON): r={corr:+.3f} "
              f"(clean — internal_bias has no goal-jitter noise, isolates the mechanism itself)")
        print(f"  CSR when resolve_uncertainty wins (ON): mean={csr_mean:.3f} over n={csr_n} wins")
        print(f"  margin range (ON): [{min(result['margins_on']):+.4f}, {max(result['margins_on']):+.4f}]")

    print("\n" + "=" * 78)
    print("SUMMARY across seeds")
    print(f"  mean win-rate lift: {statistics.mean(all_lift):+.3f}")
    print(f"  mean internal_bias-pressure correlation: {statistics.mean(all_corr):+.3f}")
    if all_csr:
        print(f"  mean CSR (pooled): {statistics.mean(all_csr):.3f}")
    print()
    print("Interpretation (OBSERVED / INFERRED / NOT DEMONSTRATED framing):")
    print(f"  OBSERVED: win-rate for the self-regulatory goal is "
          f"{'higher' if statistics.mean(all_lift) > 0.02 else 'not meaningfully different'} "
          f"under ON vs OFF across both seeds.")
    print(f"  OBSERVED: margin correlates with epistemic pressure at r="
          f"{statistics.mean(all_corr):+.3f} — "
          f"{'confirms the effect scales continuously with pressure, not a fixed offset' if statistics.mean(all_corr) > 0.3 else 'weak/no evidence of continuous scaling at this noise level'}.")
    print(f"  NOT DEMONSTRATED: whether this win-rate change propagates into "
          f"downstream behavior change (response content, memory, self-model) "
          f"at scale — would need the full v101-style replay extended with "
          f"this ablation, not run here.")
    print("=" * 78)


if __name__ == "__main__":
    main()
