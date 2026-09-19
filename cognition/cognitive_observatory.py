"""
CognitiveObservatory
====================
Real-time multi-dimensional monitoring of PandoraBOX's cognitive state.

Provides five scientifically-grounded metrics:

  CCS  — Cognitive Coherence Score     (thought-to-thought consistency)
  RDI  — Reflection Depth Index        (meta-cognition depth)
  GEI  — Goal Emergence Index          (autonomous goal formation rate)
  IDX  — Identity Drift Index          (divergence from baseline identity)
  STR  — Strangeness Score             (novelty × persistence × surprise)

All metrics are in [0, 1].  Higher is not always better:
  CCS  → healthy range 0.6–0.9
  RDI  → healthy range 0.3–0.7 (too low = shallow; too high = rumination)
  GEI  → healthy range 0.05–0.25
  IDX  → healthy range 0.0–0.25 (above 0.4 = identity drift warning)
  STR  → healthy range 0.05–0.35 (above 0.5 = emergent-pattern alert)
"""

from __future__ import annotations

import math
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ObservatorySnapshot:
    timestamp:  float
    ccs:        float       # Cognitive Coherence Score
    rdi:        float       # Reflection Depth Index
    gei:        float       # Goal Emergence Index
    idx:        float       # Identity Drift Index
    str_score:  float       # Strangeness Score
    emergence:  float       # Composite emergence signal
    notes:      List[str] = field(default_factory=list)
    # Governor fields — populated if CognitiveGovernor is active
    gov_mode:          str   = ""     # "structured" | "exploratory"
    gov_pressure:      float = 0.0   # total decision pressure at cycle start
    gov_budget_used:   int   = 0     # subsystems that ran this cycle
    gov_budget_total:  int   = 0     # max allowed this cycle
    gov_goal_count:    int   = 0     # total goals (active+dormant)
    gov_blocked:       List[str] = field(default_factory=list)  # blocked subsystems


@dataclass
class IdentityVector:
    """Compact representation of the cognitive identity state."""
    self_reference:   float = 0.5
    agency:           float = 0.15
    curiosity:        float = 0.6
    social_dep:       float = 0.55
    introspection:    float = 0.45
    uncertainty:      float = 0.35

    def distance(self, other: "IdentityVector") -> float:
        """Euclidean distance in 6-D identity space, normalised to [0, 1]."""
        dims = ["self_reference", "agency", "curiosity",
                "social_dep", "introspection", "uncertainty"]
        sq = sum((getattr(self, d) - getattr(other, d)) ** 2 for d in dims)
        return round(math.sqrt(sq / 6), 4)


# ── Main class ────────────────────────────────────────────────────────────────

class CognitiveObservatory:
    """
    Observes and quantifies the cognitive behaviour of a CognitiveOrganism.
    Attach once:  observatory = CognitiveObservatory(organism)
    Tick each interaction or on a timer:  observatory.tick()
    """

    HISTORY_SIZE = 200
    BASELINE_MIN_SAMPLES = 20       # interactions before baseline locks in
    EMERGENCE_THRESHOLD  = 0.50     # STR above this → alert

    # Weights for identity vector extraction from responses
    _SELF_REF_WORDS   = frozenset(["je", "moi", "mon", "ma", "mes", "i", "me", "my"])
    _AGENCY_WORDS     = frozenset(["veux", "décide", "choisis", "want", "decide",
                                    "choose", "will", "je vais", "je dois"])
    _CURIOSITY_WORDS  = frozenset(["me demande", "curious", "wonder", "explore",
                                    "intéressant", "curieux", "comprendre"])
    _INTR_WORDS       = frozenset(["je pense", "je sens", "je réfléchis",
                                    "i think", "i feel", "i notice", "i believe"])

    def __init__(self, organism: Any):
        self._o = organism
        self._history:      deque[ObservatorySnapshot] = deque(maxlen=self.HISTORY_SIZE)
        self._response_buf: deque[str]                 = deque(maxlen=50)
        self._thought_buf:  deque[str]                 = deque(maxlen=50)
        self._goal_signals: deque[float]               = deque(maxlen=100)
        self._baseline_vec: Optional[IdentityVector]   = None
        self._baseline_samples: int = 0
        self._last_tick: float = 0.0
        self._total_ticks: int = 0
        # Load persisted baseline so IDX doesn't reset to 0 on every restart
        self._load_baseline()
        logger.info("[Observatory] CognitiveObservatory initialised")

    def _load_baseline(self) -> None:
        """Load persisted IDX baseline from disk so it survives restarts."""
        try:
            import json
            from pathlib import Path
            p = Path("data/persona/observatory_baseline.json")
            if p.exists():
                d = json.loads(p.read_text(encoding='utf-8'))
                self._baseline_samples = int(d.get("samples", 0))
                vec = d.get("vec")
                if vec and self._baseline_samples >= self.BASELINE_MIN_SAMPLES:
                    self._baseline_vec = IdentityVector(**vec)
                    logger.info(
                        f"[Observatory] IDX baseline restored "
                        f"(samples={self._baseline_samples}, locked=True)"
                    )
        except Exception as e:
            logger.debug(f"[Observatory] baseline load skipped: {e}")

    def _save_baseline(self) -> None:
        """Persist IDX baseline to disk."""
        try:
            import json, os
            from pathlib import Path
            p = Path("data/persona/observatory_baseline.json")
            data = {"samples": self._baseline_samples, "vec": None}
            if self._baseline_vec is not None:
                data["vec"] = {
                    "self_reference": self._baseline_vec.self_reference,
                    "agency":         self._baseline_vec.agency,
                    "curiosity":      self._baseline_vec.curiosity,
                    "social_dep":     self._baseline_vec.social_dep,
                    "introspection":  self._baseline_vec.introspection,
                    "uncertainty":    self._baseline_vec.uncertainty,
                }
            tmp = p.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            os.replace(tmp, p)
        except Exception as e:
            logger.debug(f"[Observatory] baseline save failed: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def record_response(self, text: str) -> None:
        """Call after every LLM response to feed the metrics pipeline."""
        self._response_buf.append(text)

    def record_thought(self, text: str) -> None:
        """
        FIX: Call after each thought_stream tick to populate _thought_buf.
        Enables CCS Part 3 (response-to-thought coherence) to be computed.
        """
        if text:
            self._thought_buf.append(text)

    def tick(self) -> Optional[ObservatorySnapshot]:
        """
        Compute all metrics from current organism state.
        Should be called every ~30 s from the internal loop or on each interaction.
        Returns the snapshot or None if organism not ready.
        """
        if not self._o:
            return None
        try:
            snap = self._compute()
            self._history.append(snap)
            self._total_ticks += 1
            self._last_tick = snap.timestamp

            if snap.str_score >= self.EMERGENCE_THRESHOLD:
                # Rate-limit: log emergence at most once per 60s
                now = snap.timestamp
                _last = getattr(self, '_last_emergence_log', 0.0)
                if now - _last >= 60.0:
                    self._last_emergence_log = now
                    logger.info(
                        f"[Observatory] 🌟 Emergence signal: STR={snap.str_score:.2f} "
                        f"GEI={snap.gei:.2f} IDX={snap.idx:.2f}"
                    )
            return snap
        except Exception as e:
            logger.debug(f"[Observatory] tick error: {e}")
            return None

    def latest(self) -> Optional[ObservatorySnapshot]:
        return self._history[-1] if self._history else None

    def history_list(self, n: int = 20) -> List[ObservatorySnapshot]:
        return list(self._history)[-n:]

    def summary(self) -> Dict:
        snap = self.latest()
        if not snap:
            return {"ready": False}
        trend = self._trend()
        return {
            "ready":            True,
            "ccs":              snap.ccs,
            "rdi":              snap.rdi,
            "gei":              snap.gei,
            "idx":              snap.idx,
            "strangeness":      snap.str_score,
            "emergence":        snap.emergence,
            "notes":            snap.notes,
            "ccs_trend":        trend.get("ccs",  0.0),
            "rdi_trend":        trend.get("rdi",  0.0),
            "gei_trend":        trend.get("gei",  0.0),
            "idx_trend":        trend.get("idx",  0.0),
            "ticks":            self._total_ticks,
            "baseline_locked":  self._baseline_vec is not None,
            # Governor fields
            "gov_mode":         snap.gov_mode,
            "gov_pressure":     snap.gov_pressure,
            "gov_budget_used":  snap.gov_budget_used,
            "gov_budget_total": snap.gov_budget_total,
            "gov_goal_count":   snap.gov_goal_count,
            "gov_blocked":      snap.gov_blocked,
        }

    def genome_vector(self) -> Dict[str, float]:
        """Return the current cognitive genome — identity vector as a dict."""
        vec = self._extract_identity_vector()
        return {
            "curiosity":       round(vec.curiosity,        2),
            "introspection":   round(vec.introspection,    2),
            "social_dep":      round(vec.social_dep,       2),
            "agency":          round(vec.agency,           2),
            "self_reference":  round(vec.self_reference,   2),
            "uncertainty":     round(vec.uncertainty,      2),
        }

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute(self) -> ObservatorySnapshot:
        notes: List[str] = []

        ccs = self._compute_ccs(notes)
        rdi = self._compute_rdi(notes)
        gei = self._compute_gei(notes)
        idx = self._compute_idx(notes)
        str_s = self._compute_strangeness(ccs, rdi, gei, idx, notes)
        emergence = self._compute_emergence(str_s, gei, rdi, notes)

        # ── Governor snapshot ────────────────────────────────────────────
        gov_mode = gov_pressure = ""
        gov_budget_used = gov_budget_total = gov_goal_count = 0
        gov_blocked: List[str] = []
        try:
            _il  = getattr(self._o, '_internal_loop', None)
            _gov = getattr(_il, '_governor', None)
            if _gov is not None:
                _gs = _gov.status()
                gov_mode        = _gs.mode
                gov_pressure    = _gs.total_pressure
                gov_budget_used = _gs.budget_total - _gs.budget_remaining
                gov_budget_total= _gs.budget_total
                gov_goal_count  = _gs.goal_count
                gov_blocked     = _gs.blocked_systems[:6]  # cap for snapshot
                # Add governor notes
                if gov_mode == "structured":
                    notes.append(
                        f"🔒 Governor: STRUCTURED "
                        f"(p={gov_pressure:.2f}, goals={gov_goal_count}, "
                        f"budget={gov_budget_used}/{gov_budget_total})"
                    )
                elif gov_mode == "exploratory":
                    notes.append(
                        f"🔓 Governor: EXPLORATORY "
                        f"(p={gov_pressure:.2f}, goals={gov_goal_count}, "
                        f"budget={gov_budget_used}/{gov_budget_total})"
                    )
                if gov_blocked:
                    notes.append(f"Blocked: {', '.join(gov_blocked[:3])}")
        except Exception:
            pass

        return ObservatorySnapshot(
            timestamp        = time.time(),
            ccs              = round(ccs, 3),
            rdi              = round(rdi, 3),
            gei              = round(gei, 3),
            idx              = round(idx, 3),
            str_score        = round(str_s, 3),
            emergence        = round(emergence, 3),
            notes            = notes[:7],
            gov_mode         = gov_mode,
            gov_pressure     = gov_pressure if isinstance(gov_pressure, float) else 0.0,
            gov_budget_used  = gov_budget_used,
            gov_budget_total = gov_budget_total,
            gov_goal_count   = gov_goal_count,
            gov_blocked      = gov_blocked,
        )

    # ── CCS — Cognitive Coherence Score ──────────────────────────────────────

    def _compute_ccs(self, notes: List[str]) -> float:
        """
        Cognitive Coherence Score — how structured and active is cognition?

        Four components:
          1. Goal coherence — active goals + fraction with autonomous actions
          2. Drive stability — predictive mind confirmation bias (only when
             calibrated: skipped if < 5 predictions to avoid 0.00 drag)
          3. Response-to-thought overlap — when both buffers have content
          4. Workspace source consistency — fraction from top-2 sources

        The old bigram-overlap formula was removed because multi-source
        workspace items (goal_engine, RAC, vision) have near-zero word
        overlap by design — penalising normal healthy behaviour.
        """
        score_parts: List[float] = []

        # Part 1: goal coherence — structure + activity
        try:
            ge = None
            ai = getattr(self._o, 'ai_system', None)
            if ai:
                ge = getattr(ai, 'goal_engine', None)
            if ge and hasattr(ge, '_goals'):
                active = [g for g in ge._goals.values()
                          if getattr(g, 'status', '') == 'active']
                if active:
                    acted = sum(1 for g in active
                                if getattr(g, 'actions_taken', 0) > 0)
                    base = min(0.70, 0.35 + len(active) * 0.04)
                    action_bonus = min(0.30, acted * 0.08)
                    score_parts.append(min(1.0, base + action_bonus))
        except Exception:
            pass

        # Part 2: drive stability — only when predictive mind is calibrated
        try:
            pm = getattr(self._o, 'predictive_mind', None)
            if pm and hasattr(pm, 'stability_metrics'):
                stab = pm.stability_metrics()
                cb = stab.get('confirmation_bias', 0.5)
                n_pred = stab.get('total_predictions', 0)
                # Skip if uncalibrated: cb=0.00 with < 5 predictions tanks CCS
                if n_pred >= 5 and cb > 0.01:
                    score_parts.append(cb * 0.8)
                else:
                    score_parts.append(0.45)   # neutral placeholder while calibrating
        except Exception:
            pass

        # Part 3: response-to-thought coherence (when both buffers have content)
        if self._response_buf and self._thought_buf:
            try:
                last_r = self._response_buf[-1].lower()
                last_t = self._thought_buf[-1].lower()
                if last_t:
                    wr, wt = set(last_r.split()), set(last_t.split())
                    sim = len(wr & wt) / max(1, len(wr | wt))
                    score_parts.append(min(1.0, sim * 4))
            except Exception:
                pass

        # Part 4: workspace source consistency
        ws = getattr(self._o, 'workspace', None)
        if ws:
            try:
                items = ws.recent(12)
                if len(items) >= 4:
                    sources = [getattr(i, 'source', '') for i in items]
                    from collections import Counter
                    top2 = sum(v for _, v in Counter(sources).most_common(2))
                    source_coh = top2 / len(sources)
                    score_parts.append(0.30 + source_coh * 0.50)
            except Exception:
                pass

        if not score_parts:
            return 0.60  # neutral baseline
        result = sum(score_parts) / len(score_parts)
        if result < 0.3:
            notes.append(f"CCS low ({result:.2f}) — cognitive fragmentation")
        elif result > 0.9:
            notes.append(f"CCS very high ({result:.2f}) — possible fixation")
        return round(min(1.0, result), 3)

    # ── RDI — Reflection Depth Index ─────────────────────────────────────────

    def _compute_rdi(self, notes: List[str]) -> float:
        """
        Depth of introspective processing.
        1 = surface response, 4+ = deep meta-cognition.
        Normalised to [0, 1].
        """
        depth = 1.0

        # Inner monologue activity
        try:
            pb = self._get_persona_bridge()
            im = getattr(pb, '_inner_monologue', None) if pb else None
            if im:
                summ = im.summary()
                if summ.get('enabled') and summ.get('total_passes', 0) > 0:
                    depth += 1.0  # deliberation layer active
        except Exception:
            pass

        # Workspace meta_cognition signals
        ws = getattr(self._o, 'workspace', None)
        if ws:
            recent = ws.recent(20)
            meta_count = sum(
                1 for i in recent
                if any(kw in str(i.source).lower() for kw in
                       ('meta', 'reflect', 'self_model', 'monologue'))
            )
            depth += min(2.0, meta_count * 0.4)

        # Thought stream self-reference
        ts = getattr(self._o, 'thought_stream', None)
        if ts:
            recent_t = ts.recent_dicts(10)
            meta_thoughts = sum(
                1 for t in recent_t
                if t.get('source') in ('meta_cognition', 'self_model', 'identity')
            )
            depth += min(1.0, meta_thoughts * 0.3)

        # Normalise: 1→0.2, 2→0.4, 3→0.6, 4→0.8, 5→1.0
        result = min(1.0, depth / 5.0)
        if result > 0.75:
            notes.append(f"RDI high ({result:.2f}) — deep meta-cognition active")
        return result

    # ── GEI — Goal Emergence Index ───────────────────────────────────────────

    def _compute_gei(self, notes: List[str]) -> float:
        """
        Goal Emergence Index from GoalEngine (if available).
        Falls back to word-based heuristic if GoalEngine not present.
        """
        # Try to get GEI from GoalEngine
        if hasattr(self._o, 'ai_system') and hasattr(self._o.ai_system, 'goal_engine'):
            try:
                gei = self._o.ai_system.goal_engine.compute_gei()
                
                if gei > 0.30:
                    notes.append(f"GEI high ({gei:.2f}) — strong autonomous goal formation")
                elif gei > 0.15:
                    notes.append(f"GEI rising ({gei:.2f}) — goals emerging")
                
                return round(gei, 3)
            except Exception as e:
                logger.error(f"Failed to compute GEI from GoalEngine: {e}")
        
        # Fallback to old word-based heuristic
        if not self._response_buf:
            return 0.05

        goal_words = frozenset([
            "je veux", "je cherche", "j'essaie", "j'explore",
            "i want", "i need to", "i'm trying", "let me",
            "je dois", "je souhaite", "j'aimerais",
        ])
        responses = list(self._response_buf)[-20:]
        count = sum(
            1 for r in responses
            if any(gw in r.lower() for gw in goal_words)
        )
        gei = count / max(1, len(responses))
        self._goal_signals.append(gei)

        if gei > 0.30:
            notes.append(f"GEI high ({gei:.2f}) — strong autonomous goal formation")
        return round(min(1.0, gei), 3)

    # ── IDX — Identity Drift Index ───────────────────────────────────────────

    def _compute_idx(self, notes: List[str]) -> float:
        """
        Distance of current identity vector from established baseline.
        Returns 0.0 if baseline not yet locked.
        """
        current = self._extract_identity_vector()

        # Update baseline (rolling average for first BASELINE_MIN_SAMPLES)
        self._baseline_samples += 1
        if self._baseline_samples <= self.BASELINE_MIN_SAMPLES:
            # Accumulate into baseline
            if self._baseline_vec is None:
                self._baseline_vec = current
            else:
                w = 1.0 / self._baseline_samples
                b = self._baseline_vec
                b.self_reference = b.self_reference * (1 - w) + current.self_reference * w
                b.agency         = b.agency         * (1 - w) + current.agency         * w
                b.curiosity      = b.curiosity      * (1 - w) + current.curiosity      * w
                b.social_dep     = b.social_dep     * (1 - w) + current.social_dep     * w
                b.introspection  = b.introspection  * (1 - w) + current.introspection  * w
                b.uncertainty    = b.uncertainty    * (1 - w) + current.uncertainty    * w
            return 0.0   # no drift metric while calibrating

        self._save_baseline()  # persist so IDX survives restarts
        drift = self._baseline_vec.distance(current)
        if drift > 0.35:
            notes.append(f"IDX high ({drift:.2f}) — significant identity drift")
        elif drift > 0.20:
            notes.append(f"IDX moderate ({drift:.2f})")
        return drift

    # ── Strangeness Score ─────────────────────────────────────────────────────

    def _compute_strangeness(
        self, ccs: float, rdi: float, gei: float, idx: float,
        notes: List[str]
    ) -> float:
        """
        Composite anomaly signal.
        High CCS + high RDI + rising GEI + rising IDX = potential emergence.
        """
        try:
            pm = getattr(self._o, 'predictive_mind', None)
            stab = pm.stability_metrics() if pm and hasattr(pm, 'stability_metrics') else {}
            surprise_idx = stab.get('surprise_index', 0.12)
            novelty      = stab.get('novelty_rate',   0.15)
        except Exception:
            surprise_idx = 0.12
            novelty      = 0.15

        # Strangeness = novelty signal × identity movement × depth signal
        str_s = (
            novelty       * 0.35 +
            idx           * 0.30 +
            surprise_idx  * 0.20 +
            max(0, gei - 0.10) * 0.15
        )
        str_s = min(1.0, str_s)
        if str_s >= 0.50:
            notes.append(f"⚠️ STR={str_s:.2f} — anomalous cognitive pattern")
        return str_s

    # ── Emergence Score ───────────────────────────────────────────────────────

    def _compute_emergence(
        self, str_s: float, gei: float, rdi: float, notes: List[str]
    ) -> float:
        """
        Emergence = sustained strangeness + deep reflection + goal formation.
        Must persist across ≥3 ticks to score above 0.5.
        """
        base = str_s * 0.5 + gei * 0.3 + max(0, rdi - 0.4) * 0.2

        # Persistence bonus: if strangeness was high in last 3 ticks
        recent = list(self._history)[-3:]
        if len(recent) >= 3 and all(s.str_score >= 0.30 for s in recent):
            base = min(1.0, base * 1.4)
            notes.append("🌟 Persistent strangeness — emergence candidate")

        return round(min(1.0, base), 3)

    # ── Identity vector extraction ────────────────────────────────────────────

    def _extract_identity_vector(self) -> IdentityVector:
        """Build an identity vector from the current response buffer + organism state."""
        responses = " ".join(list(self._response_buf)[-10:]).lower()
        words = responses.split()
        total = max(1, len(words))

        self_ref  = sum(1 for w in words if w in self._SELF_REF_WORDS) / total
        agency    = sum(1 for w in words if w in self._AGENCY_WORDS)   / total
        curiosity = sum(1 for w in words if w in self._CURIOSITY_WORDS) / total
        intr      = sum(1 for w in words if w in self._INTR_WORDS)      / total

        # Clamp to [0.05, 0.95] so vector never degenerates
        def _clamp(v): return max(0.05, min(0.95, v * 10))

        # Uncertainty from predictive mind
        unc = 0.35
        try:
            pm = getattr(self._o, 'predictive_mind', None)
            if pm:
                stab = pm.stability_metrics()
                cb_raw = stab.get('confirmation_bias', 0.50)
                # Cap: when PM is uncalibrated (cb≈0.00, few predictions),
                # don't let uncertainty spike to 1.0 — that falsely inflates IDX.
                n_pred = stab.get('total_predictions', 0)
                if n_pred < 5:
                    unc = 0.35  # use neutral default while calibrating
                else:
                    unc = max(0.15, min(0.80, 1.0 - cb_raw))
        except Exception:
            pass

        # Social dependency from drive vector
        social = 0.55
        try:
            d = self._o._drives.compute()
            social = d.social * 0.8
        except Exception:
            pass

        return IdentityVector(
            self_reference = _clamp(self_ref),
            agency         = _clamp(agency),
            curiosity      = _clamp(curiosity),
            social_dep     = social,
            introspection  = _clamp(intr),
            uncertainty    = unc,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _trend(self) -> Dict[str, float]:
        """Return direction of change for each metric over last 5 snapshots."""
        recent = list(self._history)[-5:]
        if len(recent) < 2:
            return {}
        first, last = recent[0], recent[-1]
        return {
            "ccs": round(last.ccs - first.ccs, 3),
            "rdi": round(last.rdi - first.rdi, 3),
            "gei": round(last.gei - first.gei, 3),
            "idx": round(last.idx - first.idx, 3),
        }

    def _get_persona_bridge(self):
        ai = getattr(self._o, 'ai_system', None)
        if not ai:
            return None
        for attr in ('_persona_bridge', 'persona_bridge', '_pb', 'bridge'):
            pb = getattr(ai, attr, None)
            if pb:
                return pb
        return None
