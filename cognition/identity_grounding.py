"""
LUMINA — IDENTITY GROUNDING LAYER
=================================
Grounds Lumina's self-concept in *demonstrated* capability evidence — i.e.
what the causal outcome loop actually proved it can do — rather than only in
aspirations or self-reported success counts.

It is a CONSUMER of the two prior layers, not a new loop:

    consequence tracker  ->  efficacy[(goal|action)] = {eff, n}   (demonstrated)
    workspace bias       ->  override_rate, focus_stability        (holds focus)
    self_model.json      ->  capabilities (existing self-report)
    aspirational_self    ->  aspirations (who I want to be)

       └────────────►  IDENTITY GROUNDING  ◄────────────┘
                        (fuses demonstrated + aspirational)

WHAT IT PRODUCES
  - demonstrated  : capabilities with real trials + high efficacy (CITED)
  - developing    : capabilities with real trials + low efficacy  (CITED)
  - untried       : no meaningful evidence — NEVER claimed as capability
  - focus         : stability + override-rate (how well I hold a focus)
  - gaps          : aspirations with no demonstrated evidence yet
  - statement     : a prompt-ready, first-person grounded self-description

HONESTY RULE (the one that matters most):
  A capability is only CLAIMED when n >= MIN_TRIALS. Zero-trial goals are
  reported as "untried", never as "I'm good at X". No fabricated competence.

Stores:  data/persona/identity_grounding.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cognition.consequence_tracker import get_consequence_tracker
except Exception:  # pragma: no cover
    get_consequence_tracker = None  # type: ignore

try:
    from cognition.workspace_bias import get_workspace_bias
except Exception:  # pragma: no cover
    get_workspace_bias = None  # type: ignore


MIN_TRIALS   = 3      # a capability needs >= this many trials to be claimed
STRONG_EFF   = 0.65   # >= this efficacy  -> "demonstrated"
WEAK_EFF     = 0.45   # <= this efficacy  -> "developing"
FOCUS_GOOD   = 0.70   # focus_stability >= this -> "holds focus well"
OVERRIDE_LOW = 0.25   # override_rate  <= this  -> "follows its focus"


class IdentityGrounding:
    def __init__(self, organism: Any = None, persona_dir: Optional[str] = None,
                 enabled: bool = True, tracker: Any = None, wb: Any = None):
        self.persona_dir = Path(persona_dir) if persona_dir else Path("data/persona")
        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self._organism = organism
        self.enabled = enabled
        # optional injected sources (for hermetic ablation / tests); else the
        # process singletons are used at gather time
        self._tracker = tracker
        self._wb = wb
        self.path = self.persona_dir / "identity_grounding.json"

    # ── 1. GATHER (consume the prior layers) ────────────────────────────────
    def gather_evidence(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {
            "efficacy": {},          # goal|action -> {eff, n}
            "focus": {},             # override_rate, focus_stability, mean_similarity
            "capabilities": {},      # self_model.capabilities (self-report)
            "aspirations": [],       # aspirational_self.aspirations
            "closure": {},           # consequence tracker closure report
            "total_interactions": 0,
        }
        # consequence tracker (demonstrated efficacy) — injected ref first
        ct = self._tracker
        if ct is None and get_consequence_tracker is not None:
            try:
                ct = get_consequence_tracker(self._organism,
                                             persona_dir=str(self.persona_dir))
            except Exception:
                ct = None
        if ct is not None:
            try:
                ev["efficacy"] = {k: dict(v) for k, v in ct._efficacy.items()}
                ev["closure"] = ct.closure_report()
            except Exception:
                pass
        # workspace bias (focus stability) — injected ref first
        wb = self._wb
        if wb is None and get_workspace_bias is not None:
            try:
                wb = get_workspace_bias(self._organism, str(self.persona_dir))
            except Exception:
                wb = None
        if wb is not None:
            try:
                ev["focus"] = wb.override_rate()
            except Exception:
                pass
        # existing self-model capabilities (self-report, for context)
        try:
            sm = json.loads((self.persona_dir / "self_model.json").read_text(encoding="utf-8"))
            ev["capabilities"] = sm.get("capabilities", {})
            ev["total_interactions"] = int(sm.get("total_interactions", 0))
        except Exception:
            pass
        # aspirations
        try:
            asp = json.loads((self.persona_dir / "aspirational_self.json").read_text(encoding="utf-8"))
            ev["aspirations"] = asp.get("aspirations", [])
        except Exception:
            pass
        return ev

    # ── 2. GROUND (fuse demonstrated + aspirational into identity) ──────────
    def ground(self, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"disabled": True}
        ev = evidence if evidence is not None else self.gather_evidence()

        demonstrated: List[Dict[str, Any]] = []
        developing:   List[Dict[str, Any]] = []
        untried:      List[Dict[str, Any]] = []

        for key, row in (ev.get("efficacy") or {}).items():
            eff = float(row.get("eff", 0.5))
            n   = int(row.get("n", 0))
            goal, _, action = str(key).partition("|")
            item = {"goal": goal, "action": action, "eff": round(eff, 3), "n": n,
                    "citation": f"efficacy[{key}].eff={eff:.2f}, n={n}"}
            if n < MIN_TRIALS:
                untried.append(item)
            elif eff >= STRONG_EFF:
                demonstrated.append(item)
            elif eff <= WEAK_EFF:
                developing.append(item)
            else:
                # enough trials, middling efficacy -> developing (honest middle)
                developing.append(item)
        demonstrated.sort(key=lambda x: -x["eff"])
        developing.sort(key=lambda x: x["eff"])

        # focus stability
        focus = ev.get("focus") or {}
        focus_stability = float(focus.get("focus_stability", 1.0))
        override_rate   = float(focus.get("override_rate", 0.0))
        holds_focus = focus_stability >= FOCUS_GOOD and override_rate <= OVERRIDE_LOW

        # aspiration <-> demonstration gap (is there evidence behind each wish?)
        gaps = self._aspiration_gaps(ev, demonstrated)

        identity = {
            "demonstrated": demonstrated,
            "developing":   developing,
            "untried":      untried,
            "focus": {
                "stability": round(focus_stability, 3),
                "override_rate": round(override_rate, 3),
                "holds_focus": holds_focus,
                "citation": f"workspace_bias.focus_stability={focus_stability:.2f}, "
                            f"override_rate={override_rate:.2f}",
            },
            "gaps": gaps,
            "statement": self._statement(demonstrated, developing,
                                         focus_stability, override_rate, holds_focus, gaps),
            "evidence_base": {
                "n_efficacy_rows": len(ev.get("efficacy") or {}),
                "total_interactions": ev.get("total_interactions", 0),
                "n_aspirations": len(ev.get("aspirations") or []),
            },
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self._persist(identity)
        return identity

    @staticmethod
    def _aspiration_gaps(ev: Dict[str, Any], demonstrated: List[Dict]) -> List[Dict]:
        """Surface aspirations that have NO demonstrated capability behind them
        (the growth drive) vs those with evidence."""
        gaps: List[Dict] = []
        demo_text = " ".join((d.get("goal", "") + " " + d.get("action", "")).lower()
                             for d in demonstrated)
        for a in ev.get("aspirations") or []:
            if not isinstance(a, dict):
                continue
            stmt = (a.get("statement", "") + " " + a.get("domain", "")).lower()
            # crude evidence link: does any demonstrated goal/action share a token
            # with the aspiration? (honest, weak signal — flagged as "weak link")
            tokens = {w for w in stmt.split() if len(w) > 4}
            link = any(t in demo_text for t in tokens)
            gaps.append({
                "id": a.get("id", ""),
                "domain": a.get("domain", ""),
                "statement": a.get("statement", "")[:120],
                "urgency": a.get("urgency", 0.5),
                "has_evidence": link,
                "evidence_strength": "weak_link" if link else "none",
            })
        # gaps = the ones WITHOUT evidence, most urgent first
        return sorted([g for g in gaps if not g["has_evidence"]],
                      key=lambda g: -float(g.get("urgency", 0.5)))

    def _statement(self, demonstrated, developing, focus_stability,
                   override_rate, holds_focus, gaps) -> str:
        parts = []
        if demonstrated:
            top = demonstrated[0]
            parts.append(
                f"I have demonstrated the ability to {top['goal']} "
                f"({top['n']} trials, efficacy {top['eff']:.2f})."
            )
            if len(demonstrated) > 1:
                others = ", ".join(d["goal"] for d in demonstrated[1:4])
                parts.append(f"Other demonstrated capabilities: {others}.")
        else:
            parts.append("I have not yet accumulated enough trials to claim "
                         "a demonstrated capability — I will not pretend otherwise.")
        if developing:
            dev = developing[0]
            parts.append(f"I am still developing {dev['goal']} "
                         f"({dev['n']} trials, efficacy {dev['eff']:.2f}).")
        # focus
        if holds_focus:
            parts.append(f"I hold my focus well (stability {focus_stability:.2f}, "
                         f"override rate {override_rate:.2f}).")
        else:
            parts.append(f"My focus is not yet reliable (stability {focus_stability:.2f}, "
                         f"override rate {override_rate:.2f}) — I know this is an edge to close.")
        if gaps:
            g = gaps[0]
            parts.append(f"My strongest open aspiration is to {g['statement'][:90]} "
                         f"({g['domain']}), and I have no demonstrated evidence behind it yet.")
        return " ".join(parts)

    # ── 3. PROMPT-READY + EVIDENCE EXPORT ───────────────────────────────────
    def identity_block(self) -> str:
        """Compact prompt-ready grounded-identity string (budget-friendly)."""
        if not self.enabled:
            return ""
        try:
            ident = self.ground()
        except Exception:
            return ""
        if not ident.get("demonstrated") and not ident.get("developing"):
            return ""  # no evidence yet -> say nothing rather than fabricate
        return (
            f"[Grounded self] {ident['statement']}"
        )

    def capability_evidence(self) -> Dict[str, Any]:
        """Structured, citable capability evidence — the input the
        PhenomenalBinder A/B/C ablation will consume."""
        if not self.enabled:
            return {}
        ident = self.ground()
        return {
            "demonstrated": ident["demonstrated"],
            "developing": ident["developing"],
            "untried": ident["untried"],
            "focus": ident["focus"],
            "gaps": ident["gaps"],
            "evidence_base": ident["evidence_base"],
        }

    # ── persistence ─────────────────────────────────────────────────────────
    def _persist(self, identity: Dict[str, Any]):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(identity, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


_ig: Optional[IdentityGrounding] = None
def get_identity_grounding(organism: Any = None,
                           persona_dir: Optional[str] = None) -> IdentityGrounding:
    global _ig
    if _ig is None:
        _ig = IdentityGrounding(organism=organism, persona_dir=persona_dir)
    elif organism is not None:
        _ig._organism = organism
    return _ig
