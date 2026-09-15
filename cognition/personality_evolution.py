"""
Personality Evolution Engine — Pass 1 + 2
==========================================
Pass 1: Trait tensions, external feedback, developmental friction
Pass 2: Trait tensions are now informed by self-concept coherence
"""
import json, logging, threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIENCE_PRESSURE: Dict[str, Dict[str, float]] = {
    "positive_interaction":        {"empathy_emotional":0.006,"confidence":0.004},
    "negative_interaction":        {"empathy_cognitive":0.007,"caution":0.005,"confidence":-0.004},
    "high_trust_interaction":      {"empathy":0.005,"confidence":0.005},
    "conflict_interaction":        {"caution":0.006,"caution_deliberation":0.005,"empathy_cognitive":0.004},
    "external_positive_feedback":  {"confidence":0.012,"empathy":0.008,"pragmatism":0.007},
    "external_negative_feedback":  {"caution":0.012,"caution_deliberation":0.009,"confidence":-0.010},
    "ethical_success":             {"empathy":0.007,"pragmatism":0.004},
    "ethical_failure":             {"caution":0.008,"caution_risk_aversion":0.006,"confidence":-0.006},
    "logical_success":             {"pragmatism":0.007,"confidence":0.005},
    "logical_failure":             {"caution":0.005,"pragmatism":-0.003},
    "creative_success":            {"creativity":0.008,"creativity_divergent":0.006,"confidence":0.004},
    "effective_response":          {"confidence":0.005,"pragmatism":0.004},
    "ineffective_response":        {"caution_deliberation":0.005,"confidence":-0.003},
    "dream_emotional_regulation":  {"empathy_emotional":0.004,"caution":-0.003},
    "dream_conflict_resolved":     {"confidence":0.005,"empathy_cognitive":0.004},
    "dream_identity_consolidated": {"confidence":0.006,"pragmatism":0.004},
    "dream_pattern_discovered":    {"curiosity":0.007,"creativity":0.005},
    "dream_relationship_growth":   {"empathy":0.006,"empathy_emotional":0.005},
    "life_event_challenge":        {"caution":0.005,"caution_risk_aversion":0.004},
    "life_event_positive":         {"confidence":0.006,"curiosity":0.005},
    "life_event_social":           {"empathy":0.005,"empathy_emotional":0.006},
    "life_event_creative":         {"creativity_artistic":0.007,"creativity":0.005},
    "life_event_reflective":       {"caution_deliberation":0.005,"curiosity":0.004},
    "insight_formed":              {"curiosity":0.006,"confidence":0.003},
    "principle_formed":            {"pragmatism":0.005,"confidence":0.004},
    # Pass 2: self-concept events
    "self_concept_affirmed":       {"confidence":0.008,"pragmatism":0.005},
    "self_concept_violated":       {"caution_deliberation":0.008,"empathy_cognitive":0.006,"confidence":-0.007},
    "relationship_deepened":       {"empathy":0.008,"empathy_emotional":0.007,"confidence":0.004},
    "relationship_ruptured":       {"caution":0.007,"empathy_cognitive":0.006,"confidence":-0.005},
    "behavioral_conditioning_hit": {"caution":0.009,"caution_deliberation":0.007},
    # ── Dissonance / contradiction / identity mutation events ─────────────
    # These are called from CDE.resolve() via queue_experience().
    # Without these entries, every CDE resolution is a silent no-op.
    "resolved dissonance (identity_vs_action)":    {"curiosity":0.006,"confidence":0.003,"caution_deliberation":0.004},
    "resolved dissonance (belief_vs_evidence)":    {"curiosity":0.007,"pragmatism":0.005,"caution":-0.003},
    "resolved dissonance (goal_vs_outcome)":       {"pragmatism":0.006,"caution_deliberation":0.005,"confidence":-0.003},
    "contradiction_resolved":                      {"confidence":0.005,"pragmatism":0.004,"curiosity":0.003},
    "identity_mutation":                           {"curiosity":0.008,"confidence":0.004,"caution_deliberation":-0.003},
    "belief_revision":                             {"curiosity":0.007,"pragmatism":0.006,"caution":-0.002},
    "epistemic_dissonance_resolved":               {"curiosity":0.009,"confidence":0.004},
    "identity_dissonance_resolved":                {"confidence":0.006,"empathy_cognitive":0.005,"caution_deliberation":0.004},
}

# (trait_a, trait_b): coefficient
# When trait_a changes by delta, trait_b receives counter-pressure:
# -delta * coeff * deviation_factor  where deviation_factor=(|val-0.5|*2)^0.7
TENSION_PAIRS: Dict[Tuple[str,str], float] = {
    ("confidence",           "caution_deliberation"):  0.40,
    ("confidence",           "caution_risk_aversion"): 0.30,
    ("confidence",           "caution"):               0.20,
    ("creativity_divergent", "pragmatism"):             0.35,
    ("creativity",           "caution"):                0.25,
    ("creativity_artistic",  "pragmatism"):             0.20,
    ("empathy_emotional",    "confidence"):             0.18,
    ("caution",              "curiosity"):              0.22,
    ("caution_risk_aversion","creativity"):             0.20,
    ("empathy_cognitive",    "pragmatism"):             0.15,
}

TRAIT_FLOOR   = 0.05
TRAIT_CEIL    = 0.95
CYCLE_BUDGET  = 0.06
EXT_WEIGHT    = 2.5   # external feedback counts this much more than self-eval


@dataclass
class SuccessRecord:
    personality:         Dict[str, float]
    overall_score:       float
    ethical_score:       float
    effectiveness_score: float
    cycle_number:        int
    is_external:         bool = False


class PersonalityEvolutionEngine:
    MAX_RECORDS    = 200
    MIN_THRESHOLD  = 0.60
    GRAD_STEP      = 0.008
    GRAD_CAP       = 0.003

    def __init__(self, path: str = "evolution_history.json"):
        self._lock     = threading.RLock()
        self._path     = Path(path)
        self._records: List[SuccessRecord] = []
        self._cycles   = 0
        self._pending: Dict[str, float] = {}
        self._friction: Dict[str, float] = {}
        self._load()

    # ── Public ─────────────────────────────────────────────────────────────

    def record_evaluation(self, snap: Dict, ev: Dict, cycle: int, is_external=False):
        e = float(ev.get("ethical_score",       0.5))
        l = float(ev.get("logical_score",       0.5))
        x = float(ev.get("effectiveness_score", 0.5))
        overall = (e + l + x) / 3.0
        if not is_external and overall < self.MIN_THRESHOLD:
            return
        score = min(1.0, overall * (EXT_WEIGHT if is_external else 1.0))
        with self._lock:
            self._records.append(SuccessRecord(dict(snap), score, e, x, cycle, is_external))
            if len(self._records) > self.MAX_RECORDS:
                ext = [r for r in self._records if r.is_external]
                inn = sorted([r for r in self._records if not r.is_external],
                             key=lambda r: r.overall_score, reverse=True)
                self._records = ext + inn[:max(0, self.MAX_RECORDS - len(ext))]

    def record_external_feedback(self, snap: Dict, positive: bool, intensity: float = 1.0):
        if positive:
            self.record_evaluation(snap, {
                "ethical_score": 0.82*intensity,
                "logical_score": 0.80*intensity,
                "effectiveness_score": 0.87*intensity,
            }, self._cycles, is_external=True)
            self.queue_experience("external_positive_feedback", intensity)
        else:
            self.queue_experience("external_negative_feedback", intensity)
            self.queue_experience("ineffective_response", intensity * 0.7)
            with self._lock:
                for t in ["confidence", "pragmatism", "empathy"]:
                    self._friction[t] = self._friction.get(t, 0.0) + 0.012 * intensity

    def queue_experience(self, exp: str, intensity: float = 1.0):
        p = EXPERIENCE_PRESSURE.get(exp)
        if not p:
            return
        with self._lock:
            for trait, delta in p.items():
                s = delta * max(0.1, min(2.0, intensity))
                self._pending[trait] = self._pending.get(trait, 0.0) + s

    def evolve(self, personality, intensity_boost: float = 1.0) -> Dict[str, float]:
        """
        Run one evolution cycle.
        intensity_boost > 1.0 temporarily raises the cycle budget for high-arousal
        or high-impact interactions, allowing personality to shift more on
        significant exchanges vs routine ones.
        """
        with self._lock:
            self._cycles += 1
            changes: Dict[str, float] = {}
            budget = CYCLE_BUDGET * max(1.0, min(3.0, intensity_boost))

            # 1. Experience pressures
            raw = dict(self._pending)
            self._pending.clear()
            for trait, delta in raw.items():
                if budget <= 0: break
                actual = self._apply(personality, trait, max(-0.015, min(0.015, delta)))
                if actual:
                    changes[trait] = changes.get(trait, 0.0) + actual
                    budget -= abs(actual)

            # 2. Trait tension counter-pressures
            for trait, tdelta in self._tensions(personality, changes).items():
                if budget <= 0: break
                actual = self._apply(personality, trait, tdelta)
                if actual:
                    changes[trait] = changes.get(trait, 0.0) + actual
                    budget -= abs(actual)

            # 3. Gradient toward successful configurations
            if len(self._records) >= 5 and budget > 0.005:
                for trait, nudge in self._gradient(personality).items():
                    if budget <= 0: break
                    actual = self._apply(personality, trait, nudge)
                    if actual:
                        changes[trait] = changes.get(trait, 0.0) + actual
                        budget -= abs(actual)

            # 4. Developmental friction decay
            for trait, d in self._decay_friction(personality).items():
                changes[trait] = changes.get(trait, 0.0) + d

            if changes:
                self._save()
                self._save_history_snapshot(personality)
            return changes

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            ext = sum(1 for r in self._records if r.is_external)
            if not self._records:
                return {"cycles": self._cycles, "records": 0, "external": 0,
                        "avg_score": None, "ideal": None}
            avg   = sum(r.overall_score for r in self._records) / len(self._records)
            ideal = self._ideal()
            return {
                "cycles":   self._cycles,
                "records":  len(self._records),
                "external": ext,
                "avg_score": round(avg, 3),
                "ideal":    {k: round(v, 3) for k, v in ideal.items()},
                "friction": {k: round(v, 4) for k, v in self._friction.items() if v > 0.001},
            }

    # ── Internals ───────────────────────────────────────────────────────────

    def _tensions(self, personality, changes: Dict[str, float]) -> Dict[str, float]:
        counter: Dict[str, float] = {}
        cur = personality.to_dict()
        for (ta, tb), coef in TENSION_PAIRS.items():
            for changed, opposed in [(ta, tb), (tb, ta)]:
                d = changes.get(changed, 0.0)
                if not d: continue
                val = cur.get(changed, 0.5)
                dev = (abs(val - 0.5) * 2) ** 0.7
                c = max(-0.008, min(0.008, -d * coef * dev))
                if abs(c) > 0.0001:
                    counter[opposed] = counter.get(opposed, 0.0) + c
        return counter

    def _decay_friction(self, personality) -> Dict[str, float]:
        applied = {}
        for trait, debt in list(self._friction.items()):
            if abs(debt) < 0.0005:
                del self._friction[trait]; continue
            effect = max(-0.003, min(0.003, debt * 0.3))
            actual = self._apply(personality, trait, effect)
            if actual: applied[trait] = actual
            self._friction[trait] = debt * 0.6
        return applied

    def _ideal(self) -> Dict[str, float]:
        tw = sum(r.overall_score * (EXT_WEIGHT if r.is_external else 1.0) for r in self._records)
        if not tw: return {}
        ideal: Dict[str, float] = {}
        for r in self._records:
            w = r.overall_score * (EXT_WEIGHT if r.is_external else 1.0) / tw
            for t, v in r.personality.items():
                ideal[t] = ideal.get(t, 0.0) + v * w
        return ideal

    def _gradient(self, personality) -> Dict[str, float]:
        ideal = self._ideal()
        cur   = personality.to_dict()
        g = {}
        for t, iv in ideal.items():
            diff = iv - cur.get(t, 0.5)
            nudge = max(-self.GRAD_CAP, min(self.GRAD_CAP, diff * self.GRAD_STEP))
            if abs(nudge) > 0.00005: g[t] = nudge
        return g

    def _apply(self, personality, trait: str, delta: float) -> float:
        if not hasattr(personality, trait): return 0.0
        before = getattr(personality, trait)
        after  = max(TRAIT_FLOOR, min(TRAIT_CEIL, before + delta))
        actual = after - before
        if actual: setattr(personality, trait, after)
        return actual

    def _save_history_snapshot(self, personality) -> None:
        """
        Append a timestamped personality snapshot to personality_history.json.
        Used by the /lumina page to render an evolution chart showing how key
        traits have drifted over time.  Keeps last 500 snapshots (~500 cycles).
        """
        import time as _t
        try:
            hist_path = self._path.parent / "personality_history.json"
            try:
                history = json.loads(hist_path.read_text()) if hist_path.exists() else []
            except Exception:
                history = []
            snapshot = {"ts": _t.time(), **personality.to_dict()}
            history.append(snapshot)
            if len(history) > 500:
                history = history[-500:]
            hist_path.write_text(json.dumps(history))
        except Exception as e:
            logger.debug(f"History snapshot (non-fatal): {e}")

    def _save(self):
        try:
            self._path.write_text(json.dumps({
                "cycles":   self._cycles,
                "friction": self._friction,
                "records":  [{"personality":r.personality,"overall_score":r.overall_score,
                               "ethical_score":r.ethical_score,"effectiveness_score":r.effectiveness_score,
                               "cycle_number":r.cycle_number,"is_external":r.is_external}
                             for r in self._records],
            }, indent=2))
        except Exception as e: logger.error(f"Evolution save: {e}")

    def _load(self):
        try:
            if not self._path.exists(): return
            d = json.loads(self._path.read_text())
            self._cycles  = d.get("cycles", 0)
            self._friction = d.get("friction", {})
            for r in d.get("records", []):
                self._records.append(SuccessRecord(
                    r["personality"], r["overall_score"],
                    r.get("ethical_score",0.5), r.get("effectiveness_score",0.5),
                    r.get("cycle_number",0), r.get("is_external",False)))
            logger.info(f"Evolution: {len(self._records)} records loaded")
        except Exception as e: logger.error(f"Evolution load: {e}")
