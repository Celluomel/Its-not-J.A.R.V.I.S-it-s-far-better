"""
test_flux_snowball_replay.py — Phase 6.8
===========================================
The external analysis's proposed experiment: a long replay (100-1000
cycles) comparing two trajectories — PandoraBOX alone vs PandoraBOX+Flux — under
the SAME sensory event stream, tracking whether their divergence
amplifies, stays as bounded local perturbations, or explodes.

Honest scope, read this before trusting the numbers
-----------------------------------------------------
This sandbox has no live LLM, no live camera/mic, and no live Flux
process. Of the seven divergence dimensions the analysis asked for, this
harness measures three with real fidelity, using the actual verified
cognitive modules (GlobalWorkspace, WorkspaceCompetition via
recursive_deliberation.deliberate(), UniversalConnector, PressureSystem —
all the real classes, not stand-ins):

  MEASURED (real mechanics, real code paths):
    - Workspace focus divergence      (recursive_deliberation.deliberate)
    - Epistemic pressure divergence   (PressureSystem.boost, same call
                                        UniversalConnector already makes
                                        on a disagreement percept)
    - Goal-set divergence             (a SIMPLIFIED goal-spawn rule keyed
                                        off each organism's own pressure —
                                        NOT the real GoalEngine.derive_
                                        motivations()/generate_candidates()
                                        pipeline, which needs identity
                                        traits + tensions.json + a live
                                        organism this sandbox doesn't have.
                                        Documented as simplified, not
                                        real, so this number should be
                                        read as illustrative of the
                                        MECHANISM, not a production
                                        measurement.)

  NOT measured here (would need a live deployment, not a sandbox):
    - Prediction-error divergence (needs real emotional_state +
      intent-classification on real LLM responses)
    - Memory content divergence beyond count (needs real embeddings/FAISS)
    - Emotion/mood divergence (needs the full EmotionalStateManager)
    - Self-model divergence (needs SelfConceptSynchronizer's batch cycle)
    - Actions (no real action executor in this sandbox)

The sensory event stream and Flux's commentary are SCRIPTED, not live —
a small deterministic template library, seeded for reproducibility, so
the SAME run can be repeated and compared. Flux's commentary is
explicitly template-generated text, not a real LLM's output — labelled
as such everywhere it appears.

Usage:
    python scripts/test_flux_snowball_replay.py --cycles 300 --seed 7
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognition.universal_connector import Percept, get_universal_connector
from cognition.recursive_deliberation import deliberate
from cognition.global_workspace import GlobalWorkspace
from cognition.pressure_system import PressureSystem

WINDOW = 20  # windowed divergence rate — smooths per-cycle noise into a trend


class FakeGoal:
    _next_id = 0

    def __init__(self, name, topic, priority, energy):
        FakeGoal._next_id += 1
        self.id = f"g{FakeGoal._next_id}"
        self.name = name
        self.topic = topic
        self.priority = priority
        self.energy = energy
        self.urgency = 0.0
        self.quality_score = 0.5


class FakeGoalEngine:
    def __init__(self, goals):
        self._g = list(goals)

    def get_active_goals(self):
        return self._g

    def spawn_if_absent(self, name, topic, priority, energy):
        if any(g.topic == topic for g in self._g):
            return
        self._g.append(FakeGoal(name, topic, priority, energy))


SCENE_TEMPLATES = [
    ("someone unfamiliar entered the room", 0.7),
    ("routine background movement", 0.15),
    ("Fred looked directly at the camera without speaking", 0.5),
    ("a familiar face returned after a long absence", 0.6),
    ("nothing changed since the last check", 0.05),
    ("an object was moved that is usually still", 0.55),
]

# Explicitly template-based, NOT a real Flux/LLM output — every entry
# here is one of two rhetorical stances (agree/challenge), matching
# PEER_COGNITIVE_STANCE's two real modes (v96), so this stays honest
# about simulating the MECHANISM, not the content quality of a real model.
FLUX_AGREE_TEMPLATES = [
    "That matches what I'd expect given the pattern so far.",
    "Seems consistent — nothing here surprises me.",
]
FLUX_CHALLENGE_TEMPLATES = [
    "I don't think that's the full picture — something about this doesn't add up.",
    "I'd push back on reading this the obvious way.",
    "Actually, I think there's a more interesting explanation being missed.",
]


def make_organism(seed_goals):
    ge = FakeGoalEngine(seed_goals)
    ai_system = SimpleNamespace(goal_engine=ge)
    workspace = GlobalWorkspace()
    org = SimpleNamespace(ai_system=ai_system, workspace=workspace,
                           tension_engine=None, _loop=None)
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    org.pressure = PressureSystem(organism=org, path=tmp.name)
    return org, ge


def run_replay(n_cycles: int, seed: int, flux_comment_prob: float = 0.4):
    rng = random.Random(seed)

    seed_goals = [
        FakeGoal("General attentiveness", "attentiveness", 0.5, 0.5),
    ]
    org_a, ge_a = make_organism([FakeGoal(g.name, g.topic, g.priority, g.energy) for g in seed_goals])
    org_b, ge_b = make_organism([FakeGoal(g.name, g.topic, g.priority, g.energy) for g in seed_goals])

    focus_history_a, focus_history_b = [], []
    divergence_events = []       # 1 if focus differed this cycle, else 0
    goal_count_a, goal_count_b = [], []
    pressure_a, pressure_b = [], []

    for t in range(1, n_cycles + 1):
        scene_text, novelty = rng.choice(SCENE_TEMPLATES)
        salience = min(1.0, 0.35 + novelty)

        # SAME sensory percept fed to both — shared ground truth.
        for org in (org_a, org_b):
            get_universal_connector(org).perceive(Percept(
                modality="vision_scene", source="camera", payload=scene_text,
                confidence=0.8, salience=salience, novelty=novelty,
            ))

        # Flux only ever perceives on B — this IS the manipulation.
        if rng.random() < flux_comment_prob:
            challenging = rng.random() < 0.5
            text = rng.choice(FLUX_CHALLENGE_TEMPLATES if challenging else FLUX_AGREE_TEMPLATES)
            get_universal_connector(org_b).perceive(Percept(
                modality="peer_cognition", source="flux", payload=text,
                confidence=0.65, salience=0.55,
            ))

        res_a = deliberate(org_a, scene_text)
        res_b = deliberate(org_b, scene_text)
        focus_a = res_a.get("final_focus") if res_a else None
        focus_b = res_b.get("final_focus") if res_b else None
        focus_history_a.append(focus_a)
        focus_history_b.append(focus_b)
        divergence_events.append(1 if focus_a != focus_b else 0)

        # Simplified goal-spawn/prune rule (see module docstring — NOT the
        # real GoalEngine pipeline): sustained epistemic pressure above
        # threshold spawns ONE investigation goal (stable topic key, so
        # spawn_if_absent's dedup actually works — an earlier version used
        # a per-cycle-unique topic and produced unbounded, artifact-driven
        # goal growth, not a real dynamic); pressure dropping back below
        # threshold prunes it, mirroring real dormancy/pruning instead of
        # only ever accumulating.
        for org, ge, label in ((org_a, ge_a, "a"), (org_b, ge_b, "b")):
            epi = org.pressure.reservoirs["epistemic"].level
            topic = f"investigate_unresolved_{label}"
            # Threshold picked to actually be reachable under this
            # harness's realistic pressure dynamics (boost=0.08, 15%
            # decay/cycle) — an earlier value of 0.6 never triggered in
            # 300-cycle runs across 4 different seeds (max divergence
            # observed ~0.2), which would have made this dimension a dead
            # illustration rather than a working one.
            if epi > 0.45:
                ge.spawn_if_absent(
                    f"Investigate unresolved question ({label})",
                    topic, priority=min(0.9, epi), energy=0.6,
                )
            else:
                ge._g = [g for g in ge._g if g.topic != topic]

            # Manual, documented decay toward resting_level (0.30) — NOT a
            # call to the real PressureSystem.tick(), which reads signals
            # from a full live organism this synthetic harness doesn't
            # have; this is the same "gravity toward equilibrium" concept
            # tick() implements, applied directly and transparently here.
            res = org.pressure.reservoirs["epistemic"]
            res.level = res.resting_level + (res.level - res.resting_level) * 0.85

        goal_count_a.append(len(ge_a.get_active_goals()))
        goal_count_b.append(len(ge_b.get_active_goals()))
        pressure_a.append(org_a.pressure.reservoirs["epistemic"].level)
        pressure_b.append(org_b.pressure.reservoirs["epistemic"].level)

    return {
        "divergence_events": divergence_events,
        "goal_count_a": goal_count_a, "goal_count_b": goal_count_b,
        "pressure_a": pressure_a, "pressure_b": pressure_b,
        "focus_history_a": focus_history_a, "focus_history_b": focus_history_b,
    }


def windowed_rate(events, window=WINDOW):
    out = []
    for i in range(len(events)):
        lo = max(0, i - window + 1)
        seg = events[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out


def classify_trend(d_t):
    """
    Simple, honest classification — not a formal statistical test.
    Compares mean divergence in the first third vs the last third of the
    run. Rising -> amplification; falling -> settling; flat -> bounded
    local perturbations (the "sparse nonlinear snowball" the analysis
    predicted, neither exploding nor absent).
    """
    n = len(d_t)
    third = max(1, n // 3)
    first = sum(d_t[:third]) / third
    last = sum(d_t[-third:]) / third
    delta = last - first
    if delta > 0.15:
        return "AMPLIFYING", first, last, delta
    if delta < -0.15:
        return "SETTLING", first, last, delta
    return "BOUNDED / OSCILLATING (local perturbations, no runaway)", first, last, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data/flux_snowball_replay.png")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Phase 6.8 — Flux snowball replay: {args.cycles} cycles, seed={args.seed}")
    print("=" * 78)

    result = run_replay(args.cycles, args.seed)
    d_t = windowed_rate(result["divergence_events"])
    verdict, first, last, delta = classify_trend(d_t)

    goal_div = [abs(a - b) for a, b in zip(result["goal_count_a"], result["goal_count_b"])]
    pressure_div = [abs(a - b) for a, b in zip(result["pressure_a"], result["pressure_b"])]

    print(f"\nWorkspace focus divergence rate (windowed, W={WINDOW}):")
    print(f"  first third avg: {first:.3f}   last third avg: {last:.3f}   delta: {delta:+.3f}")
    print(f"  VERDICT: {verdict}")
    print(f"\nFinal goal-set sizes: A={result['goal_count_a'][-1]}  B={result['goal_count_b'][-1]}"
          f"  (max divergence seen: {max(goal_div)})")
    print(f"Final epistemic pressure: A={result['pressure_a'][-1]:.3f}  B={result['pressure_b'][-1]:.3f}"
          f"  (max divergence seen: {max(pressure_div):.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(d_t, color="#c0392b")
        axes[0].set_ylabel("Focus divergence rate")
        axes[0].set_title(f"Phase 6.8 replay — {args.cycles} cycles, seed={args.seed} — {verdict}")
        axes[0].axhline(0, color="gray", linewidth=0.5)

        axes[1].plot(result["goal_count_a"], label="A (Flux OFF)", color="#2980b9")
        axes[1].plot(result["goal_count_b"], label="B (Flux ON)", color="#e67e22")
        axes[1].set_ylabel("Goal-set size\n(simplified rule)")
        axes[1].legend(loc="upper left", fontsize=8)

        axes[2].plot(result["pressure_a"], label="A (Flux OFF)", color="#2980b9")
        axes[2].plot(result["pressure_b"], label="B (Flux ON)", color="#e67e22")
        axes[2].set_ylabel("Epistemic pressure")
        axes[2].set_xlabel("cycle")
        axes[2].legend(loc="upper left", fontsize=8)

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved graph to {args.out}")
    except Exception as e:
        print(f"\n(matplotlib graph skipped: {e})")

    print("=" * 78)


if __name__ == "__main__":
    main()
