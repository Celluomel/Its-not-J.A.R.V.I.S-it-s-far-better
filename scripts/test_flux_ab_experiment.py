"""
test_flux_ab_experiment.py — Phase 6.5
========================================
Does Flux genuinely enlarge Lumina's cognitive space, or is the peer
connection decorative? Per the architecture proposal's own suggested
experiment: run the SAME scenario twice through the real
recursive_deliberation.deliberate() / WorkspaceCompetition pipeline —

    Run A — Lumina alone      (goals only)
    Run B — Lumina + Flux     (same goals, plus a Flux peer_cognition
                                percept via UniversalConnector, exactly
                                the real path v95/v96/v(this) wire)

— and report whether B produces trajectories A cannot. No live LLM
required: everything measured here is pure cognitive-scoring machinery,
the same real functions the live system calls every turn.

Usage:
    python scripts/test_flux_ab_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognition.universal_connector import Percept, get_universal_connector
from cognition.recursive_deliberation import deliberate
from cognition.global_workspace import GlobalWorkspace


class FakeGoal:
    def __init__(self, id, name, topic, priority, energy):
        self.id = id
        self.name = name
        self.topic = topic
        self.priority = priority
        self.energy = energy
        self.urgency = 0.0
        self.quality_score = 0.5


class FakeGoalEngine:
    def __init__(self, goals):
        self._g = goals

    def get_active_goals(self):
        return self._g


def make_organism(goals):
    ge = FakeGoalEngine(goals)
    ai_system = SimpleNamespace(goal_engine=ge)
    workspace = GlobalWorkspace()
    return SimpleNamespace(ai_system=ai_system, workspace=workspace,
                            tension_engine=None, _loop=None)


def run_scenario(goals, flux_hypothesis, user_input):
    """One scenario, run with and without a Flux percept present."""
    # Run A — Lumina alone
    org_a = make_organism(goals)
    result_a = deliberate(org_a, user_input)

    # Run B — Lumina + Flux
    org_b = make_organism(goals)
    if flux_hypothesis:
        conn = get_universal_connector(org_b)
        conn.perceive(Percept(
            modality="peer_cognition", source="flux",
            payload=flux_hypothesis, confidence=0.65, salience=0.6,
        ))
    result_b = deliberate(org_b, user_input)

    return result_a, result_b


SCENARIOS = [
    {
        "name": "no active goals, ambiguous moment",
        "goals": [],
        "flux_hypothesis": "This silence might mean the person is thinking, not waiting.",
        "user_input": "...",
    },
    {
        "name": "one weak goal vs a confident Flux challenge",
        "goals": [FakeGoal("g1", "Minor housekeeping", "housekeeping", 0.3, 0.3)],
        "flux_hypothesis": "I don't think housekeeping is actually the priority here — "
                            "something about this conversation seems unresolved.",
        "user_input": "hmm",
    },
    {
        "name": "one strong, clearly relevant goal",
        "goals": [FakeGoal("g2", "Answer the urgent deadline question", "deadline_task", 0.9, 0.9)],
        "flux_hypothesis": "Might be worth double-checking the deadline is real.",
        "user_input": "when is this due?",
    },
    {
        "name": "two competing goals, Flux offers a synthesis-shaped angle",
        "goals": [
            FakeGoal("g3", "Help the user directly", "help_user", 0.7, 0.7),
            FakeGoal("g4", "Reflect on the exchange", "self_reflect", 0.65, 0.65),
        ],
        "flux_hypothesis": "Helping and reflecting might not be in tension here — "
                            "the reflection could BE the help.",
        "user_input": "I'm not sure what I need right now",
    },
]


def main():
    print("=" * 78)
    print("Phase 6.5 — Flux A/B cognitive-value experiment")
    print("=" * 78)

    diffs = 0
    for sc in SCENARIOS:
        res_a, res_b = run_scenario(sc["goals"], sc["flux_hypothesis"], sc["user_input"])

        focus_a = res_a.get("final_focus") if res_a else None
        focus_b = res_b.get("final_focus") if res_b else None
        differs = focus_a != focus_b
        diffs += int(differs)

        print(f"\nScenario: {sc['name']}")
        print(f"  A (Lumina alone):   ran={res_a is not None}  focus={focus_a!r}"
              f"  candidates={res_a.get('candidates_considered') if res_a else '-'}")
        print(f"  B (Lumina + Flux):  ran={res_b is not None}  focus={focus_b!r}"
              f"  candidates={res_b.get('candidates_considered') if res_b else '-'}"
              f"  synthesized={res_b.get('synthesized') if res_b else '-'}")
        print(f"  -> decision differs from A: {differs}"
              + ("  (A found nothing to compete over, B did)" if res_a is None and res_b else ""))

    print("\n" + "=" * 78)
    print(f"Summary: {diffs}/{len(SCENARIOS)} scenarios produced a different decision "
          f"with Flux present.")
    if diffs == 0:
        print("Flux did not change a single outcome in these scenarios — on this "
              "evidence it would currently be decorative, not additive.")
    else:
        print("Flux measurably changed the outcome in at least one real scenario — "
              "not decorative on this evidence. Re-run with real conversation "
              "logs for a stronger claim; this harness uses illustrative goals.")
    print("=" * 78)


if __name__ == "__main__":
    main()
