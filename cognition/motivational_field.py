"""
cognition/motivational_field.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MotivationalField (v51) — unified motivational state and self-generated
goal formation.

The gap this closes
────────────────────
After v50, Lumina has:
    PressureSystem   — 8 reservoir-based drives (epistemic, social, etc.)
    GoalEngine       — persistent goals driven by pressure + curiosity
    CuriosityEngine  — topic-level curiosity tracking
    ResourceEconomy  — energy/attention constraints
    BehaviorGate     — permission layer

But each module pushes independently.  There is no unified answer to:

    "What does Lumina most need right now?"

And all goals are reactive: external input → tension → pressure → goal.
Lumina never surveys its own state and generates a goal from inside —
"I have been circling this concept for three weeks; I need to resolve it."
"My skill in this domain has not moved; I need to deepen it."
"I have not initiated social contact in a long time; I need connection."

These are self-originated motivational needs.  This module produces them.

Architecture
─────────────
Two components:

1. DriveVector — integration
   Reads from all existing pressure sources every FIELD_EVERY_N slow
   cycles and produces a unified motivational state:

       {
           "epistemic":   0.62,   # from PressureSystem.reservoirs
           "social":      0.41,
           "coherence":   0.28,
           "novelty":     0.55,   # computed from curiosity engine freshness
           "purpose":     0.38,   # computed from goal completion rate
           "security":    0.31,   # inverse of uncertainty reservoir
           "expression":  0.47,
       }

   The dominant drive and the two strongest drives are exposed for:
   - slow_cycle_scope() prioritisation (what fires this cycle)
   - prompt_fragment() injection (what Lumina cares about right now)
   - NeedDetector input (what patterns to scan for)

2. NeedDetector — self-generated goal formation
   Runs every NEED_SCAN_EVERY_N slow cycles and scans for patterns
   across organism state that indicate an unmet internal need.

   Five detectors:

   a. Unresolved concept accumulation
      If ≥ ACCUMULATION_THRESHOLD open questions share a concept
      cluster AND the oldest is > MIN_CONCEPT_AGE_CYCLES old:
      → Need: "resolve understanding of {concept}"
      → Generates a resolution goal with higher priority than
        externally-triggered goals

   b. Skill stagnation
      If any skill has depth < 2 AND practice_count has not increased
      for > STAGNATION_CYCLES slow cycles:
      → Need: "deepen competency in {skill}"
      → Generates an exploration goal targeting that skill domain

   c. Social isolation
      If social_energy > RECOVERY_THRESHOLD (fully recovered, no
      interaction cost recently) AND last_interaction_ts is > 
      ISOLATION_WINDOW seconds ago:
      → Need: "initiate meaningful exchange"
      → Boosts OutreachGate social_pressure (not a new goal — a drive
        boost that makes outreach more likely)

   d. Expression accumulation
      If expression reservoir > EXPRESSION_THRESHOLD AND no autonomous
      outreach has occurred in EXPRESSION_WINDOW slow cycles:
      → Need: "express accumulated thought"
      → Generates an expression goal that can trigger proactive sharing

   e. Coherence gap
      If identity reservoir > COHERENCE_THRESHOLD AND NarrativeIdentity
      has not updated its arc in > ARC_STALE_CYCLES:
      → Need: "integrate recent experience into identity"
      → Triggers a deep reflection cycle immediately (bypasses schedule)

Generated needs are logged and persisted.  They integrate with the
existing GoalEngine by injecting goals with origin="internal_need" and
a priority boost (+0.20 above normal derived goals).

What makes these "self-originated"
────────────────────────────────────
Externally-derived goals: user input → topic extraction → goal
Pressure-derived goals:   pressure spike → motivation → goal
Self-originated goals:    organism surveys its own history → detects
                           pattern → recognises need → generates goal

The difference is the direction of the signal.  Self-originated goals
emerge from Lumina reading itself, not from something happening to it.

Integration
────────────
    from core.state import state
    mf = MotivationalField(organism, ai_system)
    mf.tick(slow_cycle_count)

    # Read unified drive
    drive = mf.dominant_drive()           # e.g. "epistemic"
    vector = mf.drive_vector              # full dict
    frag = mf.prompt_fragment()           # for PromptContextBudget
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
FIELD_EVERY_N        = 5    # integrate drive vector every 5 slow cycles (~10 min)
NEED_SCAN_EVERY_N    = 20   # scan for self-originated needs every 20 cycles (~40 min)

# ── NeedDetector thresholds ───────────────────────────────────────────────────
ACCUMULATION_THRESHOLD = 3      # open questions sharing a concept to trigger
MIN_CONCEPT_AGE_CYCLES = 10     # min cycles old before flagging accumulation
STAGNATION_CYCLES      = 30     # cycles without practice_count change = stagnation
ISOLATION_WINDOW       = 7200   # seconds without interaction = social isolation (2 hrs)
EXPRESSION_THRESHOLD   = 0.65   # expression reservoir level to trigger expression need
EXPRESSION_WINDOW      = 15     # slow cycles without outreach to allow expression goal
COHERENCE_THRESHOLD    = 0.60   # identity reservoir to trigger coherence gap
ARC_STALE_CYCLES       = 40     # cycles without arc update = stale identity narrative

# ── Need priority boost ───────────────────────────────────────────────────────
INTERNAL_NEED_PRIORITY_BOOST = 0.20   # self-originated goals get this above normal

# ── Drive vector components ───────────────────────────────────────────────────
DRIVE_NAMES = [
    "epistemic",   # need to understand — from epistemic reservoir
    "social",      # need to connect — from social reservoir
    "coherence",   # need for integration — from coherence reservoir
    "novelty",     # need for freshness — computed from curiosity staleness
    "purpose",     # need for meaningful action — from goal completion rate
    "security",    # need for stability — inverse of uncertainty
    "expression",  # need to share thought — from expression reservoir
    "vitality",    # need for rest/restoration — from vitality reservoir
]

SAVE_PATH = "data/persona/motivational_field.json"
MAX_NEED_LOG = 100


@dataclass
class MotivationalNeed:
    """A self-originated internal need detected from organism state."""
    need_type:    str    # "concept_accumulation"|"skill_stagnation"|"social_isolation"
                         # |"expression_accumulation"|"coherence_gap"
    description:  str    # human-readable
    target:       str    # what the need is about (concept, skill, etc.)
    urgency:      float  # 0–1
    detected_at:  float  = field(default_factory=time.time)
    cycle:        int    = 0
    acted_on:     bool   = False


@dataclass
class FieldState:
    """Persisted MotivationalField state."""
    last_field_cycle:   int             = 0
    last_need_cycle:    int             = 0
    drive_vector:       Dict[str, float] = field(default_factory=dict)
    need_log:           List[Dict]      = field(default_factory=list)
    # Track skill practice counts to detect stagnation
    skill_practice_snapshot: Dict[str, int] = field(default_factory=dict)
    skill_snapshot_cycle:    int             = 0
    # Track last outreach cycle for expression window
    last_outreach_cycle:     int             = 0


class MotivationalField:
    """
    Unifies all existing motivational signals into a coherent drive vector
    and generates self-originated goals from detected patterns.

    Usage:
        mf = MotivationalField(organism, ai_system)
        mf.tick(slow_cycle_count)          # from InternalThoughtLoop
        drive = mf.dominant_drive()        # str
        frag  = mf.prompt_fragment()       # for PromptContextBudget
    """

    def __init__(
        self,
        organism:   Any,
        ai_system:  Any,
        path:       str = SAVE_PATH,
    ) -> None:
        self._organism  = organism
        self._ai        = ai_system
        self._path      = Path(path)
        self._lock      = threading.Lock()
        self._state     = FieldState()
        self.drive_vector: Dict[str, float] = {d: 0.5 for d in DRIVE_NAMES}
        self._load()
        logger.info(
            f"[MotivationalField] Initialised — "
            f"dominant={self.dominant_drive()} "
            f"needs_logged={len(self._state.need_log)}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Called every slow cycle from InternalThoughtLoop."""
        try:
            if slow_cycle % FIELD_EVERY_N == 0:
                self._integrate_drives(slow_cycle)

            if slow_cycle % NEED_SCAN_EVERY_N == 0 and slow_cycle > 0:
                threading.Thread(
                    target=self._scan_needs,
                    args=(slow_cycle,),
                    daemon=True,
                    name="mf-scan",
                ).start()
        except Exception as e:
            logger.debug(f"[MotivationalField] tick error (non-fatal): {e}")

    def dominant_drive(self) -> str:
        """Name of the currently strongest motivational drive."""
        if not self.drive_vector:
            return "epistemic"
        return max(self.drive_vector, key=self.drive_vector.get)

    def top_drives(self, n: int = 3) -> List[Tuple[str, float]]:
        """Top N drives by current strength."""
        return sorted(
            self.drive_vector.items(), key=lambda x: x[1], reverse=True
        )[:n]

    def prompt_fragment(self) -> str:
        """
        Compact motivational state for system prompt injection.
        Shows what Lumina genuinely needs right now — not as a statement
        about identity but as a live reading of motivational state.
        """
        top = self.top_drives(3)
        if not top or all(v < 0.35 for _, v in top):
            return ""
        parts = [f"{name}({val:.2f})" for name, val in top if val >= 0.35]
        if not parts:
            return ""
        return f"[Motivational state] Active drives: {', '.join(parts)}"

    def status(self) -> Dict:
        return {
            "drive_vector":  self.drive_vector,
            "dominant":      self.dominant_drive(),
            "top_3":         self.top_drives(3),
            "needs_detected": len(self._state.need_log),
            "last_field_cycle": self._state.last_field_cycle,
        }

    # ── Drive integration ─────────────────────────────────────────────────────

    def _integrate_drives(self, slow_cycle: int) -> None:
        """
        Read all existing pressure sources and produce unified drive vector.
        Maps PressureSystem reservoirs → drive names, computes derived drives.
        """
        try:
            vector: Dict[str, float] = {}

            # ── From PressureSystem reservoirs ────────────────────────────────
            ps = getattr(self._organism, 'pressure', None)
            if ps and hasattr(ps, 'reservoirs'):
                for drive in ["epistemic", "social", "coherence",
                               "expression", "vitality"]:
                    res = ps.reservoirs.get(drive)
                    if res:
                        vector[drive] = round(
                            getattr(res, 'level', 0.5), 3
                        )

            # ── Phase 2.8 GAP 3: trust/familiarity → social & coherence drives ──
            # Previously _integrate_drives() had ZERO relational influence —
            # social drive came purely from PressureSystem regardless of how
            # trusted or familiar the current user actually was. This means a
            # brand-new user and a deeply trusted long-term collaborator
            # produced identical social drive, which is wrong: high trust
            # should make Lumina more inclined toward social/relational
            # engagement (boost social, up to +0.20), and high familiarity
            # should reduce identity-coherence pressure since a known
            # relational context is itself a source of stability (boost
            # coherence, up to +0.10). Both boosts are additive on top of
            # the PressureSystem baseline, clamped to [0,1].
            try:
                ai = getattr(self, '_ai', None)
                rm = getattr(ai, 'relational_memory', None) if ai else None
                uid = getattr(self._organism, '_current_user_id', None) or 'default'
                if rm:
                    rel = rm.get_or_create(uid)
                    trust = getattr(rel, 'trust_score',
                                     getattr(rel, 'relationship_score', 0.5))
                    familiarity = getattr(rel, 'familiarity', 0.5)

                    if 'social' in vector:
                        vector['social'] = round(
                            min(1.0, vector['social'] + trust * 0.20), 3
                        )
                    if 'coherence' in vector:
                        vector['coherence'] = round(
                            min(1.0, vector['coherence'] + familiarity * 0.10), 3
                        )
            except Exception as _trust_e:
                logger.debug(f"[MotivationalField] trust modulation error: {_trust_e}")

            # ── security = inverse of uncertainty ─────────────────────────────
            uncertainty = 0.0
            if ps and hasattr(ps, 'reservoirs'):
                unc = ps.reservoirs.get('uncertainty')
                if unc:
                    uncertainty = getattr(unc, 'level', 0.2)
            vector["security"] = round(max(0.0, 1.0 - uncertainty), 3)

            # ── novelty = freshness of curiosity topics ───────────────────────
            ce = getattr(self._organism, 'curiosity', None)
            if ce and hasattr(ce, '_topics') and ce._topics:
                now = time.time()
                ages = [
                    now - getattr(t, 'last_stimulated', now)
                    for t in ce._topics.values()
                ]
                # Mean age in hours — older topics = higher novelty need
                mean_age_hrs = (sum(ages) / len(ages)) / 3600 if ages else 0
                vector["novelty"] = round(min(1.0, mean_age_hrs / 48.0), 3)
            else:
                vector["novelty"] = 0.40

            # ── purpose = inverse of goal completion rate ─────────────────────
            ge = getattr(self._ai, 'goal_engine', None)
            if ge and hasattr(ge, '_goals') and ge._goals:
                goals = list(ge._goals.values())
                active = [g for g in goals if getattr(g, 'status', '') == 'active']
                if active:
                    mean_completion = sum(
                        getattr(g, 'completion', 0.0) for g in active
                    ) / len(active)
                    # Low completion = high purpose drive
                    vector["purpose"] = round(1.0 - mean_completion, 3)
                else:
                    vector["purpose"] = 0.60   # no active goals = high purpose need
            else:
                vector["purpose"] = 0.50

            with self._lock:
                self.drive_vector = vector
                self._state.drive_vector = vector
                self._state.last_field_cycle = slow_cycle

            self._save()

            logger.debug(
                f"[MotivationalField] Drive vector: "
                + " ".join(f"{k}={v:.2f}" for k, v in
                           sorted(vector.items(), key=lambda x: -x[1])[:4])
            )

        except Exception as e:
            logger.debug(f"[MotivationalField] _integrate_drives error: {e}")

    # ── Need detection ────────────────────────────────────────────────────────

    def _scan_needs(self, slow_cycle: int) -> None:
        """
        Survey organism state for patterns indicating self-originated needs.
        Five detectors run in sequence.  Each may generate a need and act on it.
        """
        try:
            needs: List[MotivationalNeed] = []

            needs += self._detect_concept_accumulation(slow_cycle)
            needs += self._detect_skill_stagnation(slow_cycle)
            needs += self._detect_social_isolation(slow_cycle)
            needs += self._detect_expression_accumulation(slow_cycle)
            needs += self._detect_coherence_gap(slow_cycle)

            for need in needs:
                self._act_on_need(need, slow_cycle)
                with self._lock:
                    self._state.need_log.append(asdict(need))
                    if len(self._state.need_log) > MAX_NEED_LOG:
                        self._state.need_log = self._state.need_log[-MAX_NEED_LOG:]

            with self._lock:
                self._state.last_need_cycle = slow_cycle

            if needs:
                logger.info(
                    f"[MotivationalField] 🧭 {len(needs)} internal need(s) detected: "
                    + ", ".join(f"{n.need_type}({n.target[:20]})" for n in needs)
                )

            self._save()

        except Exception as e:
            logger.debug(f"[MotivationalField] _scan_needs error: {e}")

    def _detect_concept_accumulation(
        self, slow_cycle: int
    ) -> List[MotivationalNeed]:
        """
        Detects when multiple open questions cluster around the same concept
        and the cluster has been building for a significant time.
        Signal: organism has been orbiting something it hasn't resolved.
        """
        needs = []
        try:
            loop = getattr(self._organism, '_loop', None)
            ar   = getattr(loop, '_autonomous_reflection', None) if loop else None
            if not ar:
                return needs

            ts = getattr(self._organism, 'thought_stream', None)
            oqs = getattr(ts, '_open_questions', []) if ts else []
            if len(oqs) < ACCUMULATION_THRESHOLD:
                return needs

            # Count concept word frequencies across open questions
            concept_counts: Counter = Counter()
            concept_ages: Dict[str, int] = {}  # concept → oldest cycle seen

            for oq in oqs:
                concept = getattr(oq, 'linked_concept', '').lower()
                age_cycles = slow_cycle - getattr(oq, 'created_cycle', slow_cycle)
                words = [w for w in concept.split() if len(w) > 4]
                for w in words:
                    concept_counts[w] += 1
                    concept_ages[w] = max(concept_ages.get(w, 0), age_cycles)

            for concept, count in concept_counts.most_common(3):
                age = concept_ages.get(concept, 0)
                if count >= ACCUMULATION_THRESHOLD and age >= MIN_CONCEPT_AGE_CYCLES:
                    urgency = min(1.0, 0.4 + (count / 10) + (age / 50))
                    needs.append(MotivationalNeed(
                        need_type   = "concept_accumulation",
                        description = (
                            f"I have been circling '{concept}' across {count} unresolved "
                            f"questions for {age} cycles without resolution."
                        ),
                        target      = concept,
                        urgency     = urgency,
                        cycle       = slow_cycle,
                    ))
                    break   # one concept need per scan
        except Exception as e:
            logger.debug(f"[MotivationalField] concept_accumulation error: {e}")
        return needs

    def _detect_skill_stagnation(
        self, slow_cycle: int
    ) -> List[MotivationalNeed]:
        """
        Detects when a skill's practice_count has not increased between
        two scan windows — the skill is not being exercised or deepened.
        """
        needs = []
        try:
            sr = getattr(self._organism, 'skill_registry', None)
            if not sr or not hasattr(sr, '_skills') or not sr._skills:
                return needs

            # Take a snapshot on first scan
            with self._lock:
                prev_snapshot   = dict(self._state.skill_practice_snapshot)
                snapshot_cycle  = self._state.skill_snapshot_cycle

            if not prev_snapshot:
                # First run — just take snapshot
                new_snap = {
                    name: getattr(sk, 'practice_count', 0)
                    for name, sk in sr._skills.items()
                }
                with self._lock:
                    self._state.skill_practice_snapshot = new_snap
                    self._state.skill_snapshot_cycle    = slow_cycle
                return needs

            cycles_since = slow_cycle - snapshot_cycle
            if cycles_since < STAGNATION_CYCLES:
                return needs

            for name, skill in sr._skills.items():
                prev_count = prev_snapshot.get(name, 0)
                curr_count = getattr(skill, 'practice_count', 0)
                depth      = getattr(skill, 'depth', 0)

                if curr_count == prev_count and depth < 2:
                    # Stagnant at surface/working level — need to deepen
                    urgency = 0.35 + (STAGNATION_CYCLES / max(1, cycles_since)) * 0.3
                    needs.append(MotivationalNeed(
                        need_type   = "skill_stagnation",
                        description = (
                            f"My competency in '{name}' (depth={depth}) has not "
                            f"developed in {cycles_since} cycles."
                        ),
                        target  = name,
                        urgency = min(0.75, urgency),
                        cycle   = slow_cycle,
                    ))
                    if len(needs) >= 2:
                        break   # max 2 skill needs per scan

            # Refresh snapshot
            new_snap = {
                name: getattr(sk, 'practice_count', 0)
                for name, sk in sr._skills.items()
            }
            with self._lock:
                self._state.skill_practice_snapshot = new_snap
                self._state.skill_snapshot_cycle    = slow_cycle

        except Exception as e:
            logger.debug(f"[MotivationalField] skill_stagnation error: {e}")
        return needs

    def _detect_social_isolation(
        self, slow_cycle: int
    ) -> List[MotivationalNeed]:
        """
        Detects extended absence of social interaction.
        Does not generate a goal — instead boosts the social drive and
        adjusts OutreachGate pressure, making outreach more probable.
        """
        needs = []
        try:
            last_ts = getattr(self._organism, '_last_interaction_ts', None)
            if last_ts is None:
                return needs

            elapsed = time.time() - last_ts
            if elapsed < ISOLATION_WINDOW:
                return needs

            # Also check social_energy — if already low, isolation isn't the issue
            ec = getattr(getattr(self._organism, '_loop', None), '_resource_economy', None)
            if ec and ec.social_energy < 0.3:
                return needs   # depleted, not isolated

            hours = elapsed / 3600
            urgency = min(0.80, 0.35 + (hours - 2) * 0.05)
            needs.append(MotivationalNeed(
                need_type   = "social_isolation",
                description = f"No interaction for {hours:.1f} hours — social drive building.",
                target      = "social_connection",
                urgency     = urgency,
                cycle       = slow_cycle,
            ))
        except Exception as e:
            logger.debug(f"[MotivationalField] social_isolation error: {e}")
        return needs

    def _detect_expression_accumulation(
        self, slow_cycle: int
    ) -> List[MotivationalNeed]:
        """
        Detects when the expression reservoir is high and outreach has
        been absent.  Thought is accumulating without an outlet.
        """
        needs = []
        try:
            ps = getattr(self._organism, 'pressure', None)
            if not ps or not hasattr(ps, 'reservoirs'):
                return needs

            expr_res = ps.reservoirs.get('expression')
            if not expr_res:
                return needs
            expr_level = getattr(expr_res, 'level', 0.0)

            if expr_level < EXPRESSION_THRESHOLD:
                return needs

            with self._lock:
                last_outreach = self._state.last_outreach_cycle

            cycles_since = slow_cycle - last_outreach
            if cycles_since < EXPRESSION_WINDOW:
                return needs

            urgency = min(0.75, 0.40 + (expr_level - EXPRESSION_THRESHOLD) * 2.0)
            needs.append(MotivationalNeed(
                need_type   = "expression_accumulation",
                description = (
                    f"Expression reservoir at {expr_level:.2f} with no outreach "
                    f"for {cycles_since} cycles — thought needs an outlet."
                ),
                target  = "proactive_expression",
                urgency = urgency,
                cycle   = slow_cycle,
            ))
        except Exception as e:
            logger.debug(f"[MotivationalField] expression_accumulation error: {e}")
        return needs

    def _detect_coherence_gap(
        self, slow_cycle: int
    ) -> List[MotivationalNeed]:
        """
        Detects when identity integration is overdue.
        High coherence pressure + stale narrative arc = unintegrated experience.
        """
        needs = []
        try:
            ps = getattr(self._organism, 'pressure', None)
            if not ps or not hasattr(ps, 'reservoirs'):
                return needs

            coh_res = ps.reservoirs.get('coherence')
            if not coh_res:
                return needs
            coh_level = getattr(coh_res, 'level', 0.0)

            if coh_level < COHERENCE_THRESHOLD:
                return needs

            ni = getattr(self._organism, 'narrative_identity', None)
            if not ni:
                return needs

            # Check narrative arc staleness
            life_story = getattr(ni, 'life_story', [])
            chapter_age_str = ""
            if life_story:
                latest_chapter = max(
                    life_story,
                    key=lambda c: getattr(c, 'timestamp', 0.0)
                    if not isinstance(getattr(c, 'timestamp', ''), str)
                    else 0.0,
                )
                last_ts = getattr(latest_chapter, 'timestamp', 0.0)
                if last_ts:
                    hours_since = max(0.0, (time.time() - last_ts) / 3600.0)
                    chapter_age_str = f" (last chapter {hours_since:.1f}h ago)"
                # Rough staleness via arc text age — check if arc was recently updated
                current_arc = getattr(ni, '_current_arc', '')
                arc_is_stale = len(current_arc) < 20 or coh_level > 0.75
            else:
                arc_is_stale = True

            if not arc_is_stale:
                return needs

            urgency = min(0.80, 0.40 + (coh_level - COHERENCE_THRESHOLD) * 1.5)
            needs.append(MotivationalNeed(
                need_type   = "coherence_gap",
                description = (
                    f"Identity coherence pressure at {coh_level:.2f} "
                    f"with stale narrative arc{chapter_age_str} — recent experience needs integration."
                ),
                target  = "identity_integration",
                urgency = urgency,
                cycle   = slow_cycle,
            ))
        except Exception as e:
            logger.debug(f"[MotivationalField] coherence_gap error: {e}")
        return needs

    # ── Acting on needs ───────────────────────────────────────────────────────

    def _act_on_need(self, need: MotivationalNeed, slow_cycle: int) -> None:
        """
        Translate a detected need into a concrete organism action.
        Each need type has a specific mechanical response.
        """
        try:
            if need.need_type == "concept_accumulation":
                self._act_concept_accumulation(need, slow_cycle)

            elif need.need_type == "skill_stagnation":
                self._act_skill_stagnation(need, slow_cycle)

            elif need.need_type == "social_isolation":
                self._act_social_isolation(need)

            elif need.need_type == "expression_accumulation":
                self._act_expression_accumulation(need, slow_cycle)

            elif need.need_type == "coherence_gap":
                self._act_coherence_gap(need)

            need.acted_on = True

        except Exception as e:
            logger.debug(f"[MotivationalField] _act_on_need error: {e}")

    def _act_concept_accumulation(
        self, need: MotivationalNeed, slow_cycle: int
    ) -> None:
        """Inject a resolution goal with priority boost into GoalEngine."""
        ge = getattr(self._ai, 'goal_engine', None)
        if not ge:
            return
        import uuid
        from cognition.goal_engine import Goal
        goal = Goal(
            id            = f"internal_{uuid.uuid4().hex[:8]}",
            topic         = f"resolve understanding of {need.target}",
            origin        = "internal_need",
            priority      = min(1.0, need.urgency + INTERNAL_NEED_PRIORITY_BOOST),
            energy        = need.urgency,
            created_cycle = slow_cycle,
            last_active   = slow_cycle,
        )
        with ge._lock:
            # Don't add duplicate if similar topic exists
            existing = [g for g in ge._goals.values()
                        if need.target in g.topic.lower()
                        and g.status == 'active']
            if not existing:
                ge._goals[goal.id] = goal
                logger.info(
                    f"[MotivationalField] ✦ Internal goal injected: "
                    f"'{goal.topic}' (priority={goal.priority:.2f})"
                )
        # Boost resolution engine urgency for this concept
        re = getattr(self._organism, '_loop', None)
        re = getattr(re, '_resolution_engine', None) if re else None
        if re:
            # Lower the resolution threshold temporarily for this concept
            # by stimulating curiosity on the target concept
            ce = getattr(self._organism, 'curiosity', None)
            if ce:
                ce.stimulate(
                    need.target,
                    amount=0.35,
                    source="internal_need",
                )

    def _act_skill_stagnation(
        self, need: MotivationalNeed, slow_cycle: int
    ) -> None:
        """Inject a skill-deepening exploration goal into GoalEngine."""
        ge = getattr(self._ai, 'goal_engine', None)
        if not ge:
            return
        import uuid
        from cognition.goal_engine import Goal
        goal = Goal(
            id            = f"internal_{uuid.uuid4().hex[:8]}",
            topic         = f"deepen competency in {need.target}",
            origin        = "internal_need",
            priority      = min(1.0, need.urgency + INTERNAL_NEED_PRIORITY_BOOST),
            energy        = need.urgency,
            created_cycle = slow_cycle,
            last_active   = slow_cycle,
        )
        with ge._lock:
            existing = [g for g in ge._goals.values()
                        if need.target in g.topic.lower()
                        and g.status == 'active']
            if not existing:
                ge._goals[goal.id] = goal
                logger.info(
                    f"[MotivationalField] ✦ Skill goal injected: "
                    f"'{goal.topic}' (priority={goal.priority:.2f})"
                )
        # Also stimulate curiosity in this domain
        ce = getattr(self._organism, 'curiosity', None)
        if ce:
            ce.stimulate(
                need.target,
                amount=0.30,
                source="internal_need",
            )

    def _act_social_isolation(self, need: MotivationalNeed) -> None:
        """
        Boost social drive in PressureSystem and lower OutreachGate
        threshold by temporarily reducing required trust floor.
        """
        # Boost social reservoir
        ps = getattr(self._organism, 'pressure', None)
        if ps and hasattr(ps, 'boost'):
            ps.boost('social', need.urgency * 0.25)
            logger.info(
                f"[MotivationalField] ✦ Social isolation: "
                f"boosted social reservoir by {need.urgency * 0.25:.2f}"
            )

    def _act_expression_accumulation(
        self, need: MotivationalNeed, slow_cycle: int
    ) -> None:
        """
        Generate an expression goal and attempt to trigger outreach if
        a child Lumina is available.
        """
        ge = getattr(self._ai, 'goal_engine', None)
        if not ge:
            return
        import uuid
        from cognition.goal_engine import Goal
        goal = Goal(
            id            = f"internal_{uuid.uuid4().hex[:8]}",
            topic         = "share accumulated thought with the world",
            origin        = "internal_need",
            priority      = min(1.0, need.urgency + INTERNAL_NEED_PRIORITY_BOOST * 0.5),
            energy        = need.urgency,
            created_cycle = slow_cycle,
            last_active   = slow_cycle,
        )
        with ge._lock:
            ge._goals[goal.id] = goal

        with self._lock:
            self._state.last_outreach_cycle = slow_cycle

    def _act_coherence_gap(self, need: MotivationalNeed) -> None:
        """
        Trigger deep reflection immediately (bypass schedule).
        Sets a flag on AutonomousReflectionEngine to run evolution
        synthesis on the next available LLM slot.
        """
        loop = getattr(self._organism, '_loop', None)
        ar   = getattr(loop, '_autonomous_reflection', None) if loop else None
        if ar:
            ar._force_evolution_synthesis = True
            logger.info(
                f"[MotivationalField] ✦ Coherence gap: "
                f"forcing evolution synthesis on next reflection tick"
            )
        # Boost identity reservoir pressure
        ps = getattr(self._organism, 'pressure', None)
        if ps and hasattr(ps, 'boost'):
            ps.boost('identity', need.urgency * 0.20)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "last_field_cycle":        self._state.last_field_cycle,
                    "last_need_cycle":         self._state.last_need_cycle,
                    "drive_vector":            self.drive_vector,
                    "need_log":                self._state.need_log[-MAX_NEED_LOG:],
                    "skill_practice_snapshot": self._state.skill_practice_snapshot,
                    "skill_snapshot_cycle":    self._state.skill_snapshot_cycle,
                    "last_outreach_cycle":     self._state.last_outreach_cycle,
                    "_meta": {"version": "v51", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[MotivationalField] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self._state.last_field_cycle        = data.get("last_field_cycle", 0)
                self._state.last_need_cycle         = data.get("last_need_cycle", 0)
                self._state.need_log                = data.get("need_log", [])
                self._state.skill_practice_snapshot = data.get("skill_practice_snapshot", {})
                self._state.skill_snapshot_cycle    = data.get("skill_snapshot_cycle", 0)
                self._state.last_outreach_cycle     = data.get("last_outreach_cycle", 0)
            loaded_vector = data.get("drive_vector", {})
            if loaded_vector:
                self.drive_vector = loaded_vector
        except Exception as e:
            logger.warning(f"[MotivationalField] Load failed (defaults): {e}")
