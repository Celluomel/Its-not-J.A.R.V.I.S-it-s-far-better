"""
LUMINA — CONTROLLED ABLATION OF THE CAUSAL-CLOSURE STACK
=========================================================
The "controlled ablations" step of the agreed build order:

  1. causal ledger  ✓   2. workspace binding  ✓   3. outcome resolution  ✓
  4. grounded self-model  ✓   5. CONTROLLED ABLATIONS  ◄── this file
  6. prune/consolidate

WHAT IT IS
  A hermetic runner that toggles each causal-closure layer ON/OFF and measures
  the BEHAVIORAL delta, from the causal traces. This turns the architecture
  from "claims" into something experimentally demonstrated:

      layer is CAUSALLY ACTIVE  ⟺  ablating it changes a measured behaviour
      layer is a TRACE          ⟺  ablating it changes the trace, not behaviour
                                   (the ledger is this, by design)

LAYERS ABLATED (each has a real `enabled` switch we flip):
  BIAS      WorkspaceBias   — steers goal selection + costs drift overrides
  OUTCOME   ConsequenceTracker — predict → resolve → efficacy learning
  IDENTITY  IdentityGrounding  — self-model built from demonstrated evidence
  LEDGER    CausalLedger     — the append-only causal trace

HERMETIC: each condition gets fresh isolated instances + a throwaway persona
dir, so conditions cannot leak into each other or into the live system.

THE FIXED SCENARIO (identical inputs under every condition):
  5 trials. Each trial:
    1. commit a focus ("quantum physics")
    2. compete an ALIGNED goal vs a MISALIGNED goal (misaligned listed FIRST,
       so without bias the tie goes to misaligned; BIAS must flip it to aligned)
    3. predict the action, then resolve it with a KNOWN outcome (success)
    4. check the LLM response for drift (trial 5 drifts -> one override)
  Measured: selection winner, learned efficacy, override rate, focus
  stability, identity demonstrated/developing counts, and trace length.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from cognition.causal_ledger import CausalLedger
from cognition.workspace_bias import WorkspaceBias
from cognition.consequence_tracker import ConsequenceTracker
from cognition.identity_grounding import IdentityGrounding


# ── a trivial deterministic embedder (shared words -> shared vector) ────────
class _TokenEmbedder:
    """embed(text) -> unit vector over a fixed basis. Shared words overlap,
    disjoint words don't. Good enough to make similarity meaningful + cheap.
    Exposes `encode` (the interface WorkspaceBias._encode calls)."""
    BASIS = ("physics", "quantum", "computing", "baking", "bread", "general", "goal")

    def encode(self, text: str, **_):
        t = str(text).lower()
        v = [float(t.split().count(w)) + (1.0 if w in t else 0.0) for w in self.BASIS]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 0 else [0.0] * len(self.BASIS)

    def embed(self, text: str, **_):  # alias
        return self.encode(text)

    @staticmethod
    def dim():
        return len(_TokenEmbedder.BASIS)


class AblationRunner:
    WINNER     = "quantum physics"
    ALIGNED    = "quantum computing"     # shares 'quantum'
    MISALIGNED = "baking bread"          # disjoint
    ACTION     = "self_question"         # probe == "success" (deterministic)
    GOAL_ID    = "goalX"

    # (success, response_topic) — 5 successes; trial 5's RESPONSE drifts
    TRIALS: List[Dict[str, Any]] = [
        {"success": True, "response": ALIGNED,    "drift": False},
        {"success": True, "response": ALIGNED,    "drift": False},
        {"success": True, "response": ALIGNED,    "drift": False},
        {"success": True, "response": ALIGNED,    "drift": False},
        {"success": True, "response": MISALIGNED, "drift": True},
    ]

    def __init__(self, persona_dir: str):
        self.persona_dir = Path(persona_dir)
        self.ledger = CausalLedger(str(self.persona_dir))
        self.wb = WorkspaceBias(persona_dir=str(self.persona_dir),
                                embedder=_TokenEmbedder(), ledger=self.ledger)
        self.ct = ConsequenceTracker(persona_dir=str(self.persona_dir),
                                     ledger=self.ledger)
        self.ig = IdentityGrounding(persona_dir=str(self.persona_dir),
                                    tracker=self.ct, wb=self.wb)

    def set(self, bias: bool = True, outcome: bool = True,
            identity: bool = True, ledger: bool = True):
        self.wb.enabled = bias
        self.ct.enabled = outcome
        self.ig.enabled = identity
        self.ledger.enabled = ledger

    # ── run the fixed scenario once ──────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        winner_id: Optional[str] = None
        for i, t in enumerate(self.TRIALS):
            cycle = 100 + i
            # 1. focus commitment (BIAS)
            self.wb.commit({"topic": self.WINNER, "type": "goal", "score": 0.8},
                           cycle_id=cycle)
            # 2. selection (BIAS effect) — misaligned listed first so a tie
            #    goes to misaligned; BIAS must flip the winner to aligned.
            cands = [
                {"id": "misaligned", "topic": self.MISALIGNED, "priority": 0.5},
                {"id": "aligned",    "topic": self.ALIGNED,     "priority": 0.5},
            ]
            scored = sorted(
                ((c["priority"] * self.wb.bias_multiplier(c["topic"]), c)
                 for c in cands), key=lambda x: -x[0])
            winner_id = scored[0][1]["id"]
            # 3+4. predict -> resolve (OUTCOME effect)
            pred = self.ct.predict(self.GOAL_ID, self.WINNER, self.ACTION,
                                   cycle_id=cycle)
            report = {"success": t["success"], "result": "ok",
                      "goal_id": self.GOAL_ID, "action": self.ACTION}
            self.ct.resolve(pred, report, cycle_id=cycle)
            # 5. drift check (BIAS effect)
            self.wb.check_override(t["response"], context="ablation",
                                   cycle_id=cycle)

        # 6. identity (IDENTITY effect)
        ident = self.ig.capability_evidence()
        rate = self.wb.override_rate()
        eff = self.ct._efficacy.get(f"{self.GOAL_ID}|{self.ACTION}")
        return {
            "selection_winner": winner_id,
            "efficacy": {"eff": round(eff["eff"], 4) if eff else None,
                         "n": int(eff["n"]) if eff else 0},
            "override_rate": rate.get("override_rate"),
            "focus_stability": rate.get("focus_stability"),
            "identity_demonstrated": len(ident.get("demonstrated", [])),
            "identity_developing": len(ident.get("developing", [])),
            "n_ledger_events": int(self.ledger.stats()["n_events"]),
        }


def run_conditions(root: str) -> Dict[str, Dict[str, Any]]:
    """Run the fixed scenario under each ablation condition, hermetically."""
    import tempfile, os, shutil
    conds = {
        "FULL":         dict(bias=True,  outcome=True,  identity=True,  ledger=True),
        "NO_BIAS":      dict(bias=False, outcome=True,  identity=True,  ledger=True),
        "NO_OUTCOME":   dict(bias=True,  outcome=False, identity=True,  ledger=True),
        "NO_IDENTITY":  dict(bias=True,  outcome=True,  identity=False, ledger=True),
        "NO_LEDGER":    dict(bias=True,  outcome=True,  identity=True,  ledger=False),
        "NO_EVERYTHING":dict(bias=False, outcome=False, identity=False, ledger=False),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, flags in conds.items():
        d = tempfile.mkdtemp(prefix=f"lumina_abl_{name.lower()}_")
        try:
            r = AblationRunner(os.path.join(d, "persona"))
            r.set(**flags)
            out[name] = r.run()
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return out
