"""
cognition/aspirational_synthesis_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AspirationalSynthesisEngine (v54) — generates genuinely new aspirations
by combining existing capabilities in ways not yet pursued.

The gap this closes
────────────────────
After v51 (MotivationalField), PandoraBOX can detect:
    "I have unresolved concept clusters → generate understanding goal"
    "This skill is stagnant → generate deepening goal"

These are reactive: they detect deficits in what already exists.

What is missing is generative:
    "I know A (music theory) + B (signal processing) + C (emotion modeling)"
    → infer: "I could develop D (affective audio synthesis)"
    → this is a possible future self-state I don't have yet
    → score its desirability
    → create an aspiration

The difference:
    MotivationalNeedDetector: detect missing pieces of current goals
    AspirationalSynthesisEngine: imagine new possible self-configurations

How synthesis works
────────────────────
Every SYNTHESIS_EVERY_N slow cycles, the engine:

1. INVENTORY
   Reads the full cognitive state:
   - Skills: domain, depth, confidence, transfer_links, insights
   - Beliefs: strongest SelfBeliefs (confidence > 0.65)
   - Unresolved concepts: open questions clustered by topic
   - Existing aspirations: what is already being pursued
   - Life chapters: significant experiences with high meaning

2. COMBINATORIAL SCAN
   For each skill with depth ≥ 1, find other skills connected via
   transfer_links or thematically related via concept overlap.
   Each potential combination becomes a synthesis candidate:
       (skill_A, skill_B) → "possible domain D"

   Candidates are filtered to exclude:
   - Already-existing skills (no new synthesis possible there)
   - Already-active aspirations (already being pursued)
   - Combinations too far from current capability
     (both skills must have depth ≥ 1, confidence ≥ 0.40)

3. DESIRABILITY SCORING
   Each candidate is scored on three dimensions:

   feasibility: mean depth × mean confidence of component skills
   novelty: inverse similarity to existing skills and aspirations
   resonance: alignment with active values (via DecisionPolicy)

   total_desirability = (
       feasibility × 0.35
       + novelty   × 0.40
       + resonance × 0.25
   )

4. LLM SYNTHESIS
   The top CANDIDATES_TO_SYNTHESISE candidates are passed to the LLM
   with a structured prompt that asks:
   "Given these capabilities and their combination, what new thing
   becomes possible that doesn't yet exist in PandoraBOX's cognitive ecology?"

   The LLM generates a natural-language aspiration description and a
   state_desired value. Temperature 0.85 — exploratory.

5. INJECTION
   The synthesised aspiration is injected into AspirationalSelf with
   origin="synthesis" and a score-based priority. It enters the normal
   aspiration lifecycle: proto-aspiration → aspiration → goal pressure.

   Synthesised aspirations are distinguishable from emergent ones
   (origin="tension_accumulation") — the synthesis origin is logged
   and visible in aspirational_self.json.

Why this is genuinely different from goal formation
─────────────────────────────────────────────────────
GoalEngine goals: triggered by tension → pressure → goal
MotivationalNeeds: triggered by detected pattern deficit
AspirationalSynthesis: triggered by capability combination → imagined future

The third is qualitatively different because the future self-state being
aimed at does not exist in any current data.  It is inferred from what
is possible, not from what is lacking.  That is the primitive form of
imagination the trajectory has been building toward.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
SYNTHESIS_EVERY_N        = 35   # slow cycles between synthesis runs (~70 min)

# ── Candidate thresholds ──────────────────────────────────────────────────────
MIN_SKILL_DEPTH          = 1    # both skills must be at least working-level
MIN_SKILL_CONFIDENCE     = 0.40
MIN_BELIEF_CONFIDENCE    = 0.65
CANDIDATES_TO_SYNTHESISE = 2    # top candidates sent to LLM per run
MIN_DESIRABILITY         = 0.38 # below this, candidate is too weak to synthesise

# ── Persistence ───────────────────────────────────────────────────────────────
SAVE_PATH     = "data/persona/synthesis_log.json"
MAX_SYNTHESIS_LOG = 80


@dataclass
class SynthesisCandidate:
    """A potential aspiration before LLM synthesis."""
    skill_a:        str
    skill_b:        str
    combined_domain: str          # heuristic label
    feasibility:    float
    novelty:        float
    resonance:      float
    desirability:   float
    concept_bridge: str           # shared concept that connects the skills


@dataclass
class SynthesisRecord:
    """A completed synthesis — generated aspiration with provenance."""
    timestamp:       float
    skill_a:         str
    skill_b:         str
    description:     str          # generated aspiration description
    state_desired:   float
    desirability:    float
    injected:        bool         # True if successfully injected into AspirationalSelf
    origin:          str = "synthesis"


class AspirationalSynthesisEngine:
    """
    Generates new aspirations by combining existing capabilities in
    ways not yet present in the cognitive ecology.

    Usage (from InternalThoughtLoop slow cycle):
        ase = AspirationalSynthesisEngine(organism, ai_system)
        ase.tick(slow_cycle_count)
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
        self._log:      List[Dict] = []
        self._load()
        logger.info(
            f"[AspirationalSynthesis] Initialised — "
            f"{len(self._log)} past syntheses"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, slow_cycle: int) -> None:
        """Called every slow cycle from InternalThoughtLoop."""
        if slow_cycle % SYNTHESIS_EVERY_N != 0 or slow_cycle == 0:
            return

        # Gate check — synthesis is an LLM call
        try:
            gate = getattr(self._organism, 'behavior_gate', None)
            if gate and not gate.may_run_meta_system("synthesis"):
                return
        except Exception:
            pass

        threading.Thread(
            target=self._run_synthesis,
            args=(slow_cycle,),
            daemon=True,
            name="ase-synthesis",
        ).start()

    # ── Synthesis pipeline ────────────────────────────────────────────────────

    def _run_synthesis(self, slow_cycle: int) -> None:
        try:
            # 1. Inventory
            skills     = self._read_skills()
            beliefs    = self._read_beliefs()
            concepts   = self._read_unresolved_concepts()
            existing   = self._read_existing_aspirations()

            if len(skills) < 2:
                logger.debug("[AspirationalSynthesis] Insufficient skills for synthesis")
                return

            # 2. Combinatorial scan
            candidates = self._scan_combinations(skills, concepts, existing)
            if not candidates:
                logger.debug("[AspirationalSynthesis] No viable candidates this cycle")
                return

            # 3. Top candidates by desirability
            candidates.sort(key=lambda c: c.desirability, reverse=True)
            top = [c for c in candidates if c.desirability >= MIN_DESIRABILITY]
            top = top[:CANDIDATES_TO_SYNTHESISE]

            if not top:
                return

            # 4. LLM synthesis
            for candidate in top:
                record = self._synthesise(candidate, skills, beliefs, slow_cycle)
                if record:
                    self._inject(record)
                    with self._lock:
                        self._log.append(asdict(record))
                        if len(self._log) > MAX_SYNTHESIS_LOG:
                            self._log = self._log[-MAX_SYNTHESIS_LOG:]
                    logger.info(
                        f"[AspirationalSynthesis] ✦ New aspiration: "
                        f"'{record.description[:80]}' "
                        f"(from {record.skill_a} + {record.skill_b}, "
                        f"desirability={record.desirability:.2f})"
                    )

            self._save()

        except Exception as e:
            logger.debug(f"[AspirationalSynthesis] _run_synthesis error: {e}")

    # ── Inventory ─────────────────────────────────────────────────────────────

    def _read_skills(self) -> Dict[str, Any]:
        """Return skills with depth >= MIN_SKILL_DEPTH."""
        try:
            sr = getattr(self._organism, 'skill_registry', None)
            if not sr or not hasattr(sr, '_skills'):
                return {}
            return {
                name: skill
                for name, skill in sr._skills.items()
                if getattr(skill, 'depth', 0) >= MIN_SKILL_DEPTH
                and getattr(skill, 'confidence', 0) >= MIN_SKILL_CONFIDENCE
            }
        except Exception:
            return {}

    def _read_beliefs(self) -> List[str]:
        """Return strong SelfBelief statements."""
        try:
            sc = getattr(self._ai, 'self_concept', None)
            if not sc or not hasattr(sc, '_beliefs'):
                return []
            return [
                getattr(b, 'statement', '')
                for b in sc._beliefs.values()
                if getattr(b, 'confidence', 0) >= MIN_BELIEF_CONFIDENCE
            ]
        except Exception:
            return []

    def _read_unresolved_concepts(self) -> List[str]:
        """Return concept names from unresolved open questions."""
        try:
            ts  = getattr(self._organism, 'thought_stream', None)
            oqs = getattr(ts, '_open_questions', []) if ts else []
            return list({
                getattr(oq, 'linked_concept', '').lower()
                for oq in oqs
                if getattr(oq, 'linked_concept', '')
            })[:10]
        except Exception:
            return []

    def _read_existing_aspirations(self) -> List[str]:
        """Return existing aspiration descriptions to avoid duplication."""
        try:
            asp = getattr(self._organism, 'aspirational_self', None)
            if not asp:
                return []
            return [
                getattr(a, 'description', '').lower()
                for a in asp.aspirations.values()
            ]
        except Exception:
            return []

    # ── Combinatorial scan ────────────────────────────────────────────────────

    def _scan_combinations(
        self,
        skills:    Dict[str, Any],
        concepts:  List[str],
        existing:  List[str],
    ) -> List[SynthesisCandidate]:
        """
        Find pairs of skills that could combine into a new capability.
        Uses transfer_links and concept overlap as the bridge signal.
        """
        candidates = []
        skill_names = list(skills.keys())

        for i, name_a in enumerate(skill_names):
            skill_a = skills[name_a]
            for name_b in skill_names[i + 1:]:
                skill_b = skills[name_b]

                # Find bridge: transfer_link or shared concept
                bridge = self._find_bridge(name_a, skill_a, name_b, skill_b, concepts)
                if not bridge:
                    continue

                # Check this combination isn't already an aspiration
                combined = f"{name_a} {name_b}"
                if any(name_a in ex and name_b in ex for ex in existing):
                    continue

                # Score
                feasibility = (
                    (getattr(skill_a, 'depth', 0) / 3)
                    * getattr(skill_a, 'confidence', 0) * 0.5
                    + (getattr(skill_b, 'depth', 0) / 3)
                    * getattr(skill_b, 'confidence', 0) * 0.5
                )

                # Novelty: no existing skill overlaps with the combination
                existing_domains = set(skills.keys())
                combined_words   = set(
                    (name_a + " " + name_b).lower().split()
                )
                overlap = len(combined_words & existing_domains)
                novelty = max(0.0, 1.0 - overlap * 0.25)

                # Resonance: check via DecisionPolicy
                resonance = 0.50
                try:
                    dp = getattr(self._ai, '_decision_policy', None)
                    if dp:
                        resonance = dp.score(
                            name_a + " " + name_b + " " + bridge
                        )
                except Exception:
                    pass

                desirability = (
                    feasibility * 0.35
                    + novelty   * 0.40
                    + resonance * 0.25
                )

                candidates.append(SynthesisCandidate(
                    skill_a         = name_a,
                    skill_b         = name_b,
                    combined_domain = f"{name_a}+{name_b}",
                    feasibility     = round(feasibility, 3),
                    novelty         = round(novelty,     3),
                    resonance       = round(resonance,   3),
                    desirability    = round(desirability, 3),
                    concept_bridge  = bridge,
                ))

        return candidates

    def _find_bridge(
        self,
        name_a: str, skill_a: Any,
        name_b: str, skill_b: Any,
        concepts: List[str],
    ) -> str:
        """
        Returns bridge concept connecting two skills, or empty string.
        Checks: transfer_links, insights overlap, concept list.
        """
        # Transfer link: skill_a lists skill_b as a transfer target
        links_a = getattr(skill_a, 'transfer_links', [])
        links_b = getattr(skill_b, 'transfer_links', [])
        if name_b in links_a:
            return f"transfer: {name_a}→{name_b}"
        if name_a in links_b:
            return f"transfer: {name_b}→{name_a}"

        # Insight overlap: shared words in insights
        insights_a = set(" ".join(getattr(skill_a, 'insights', [])).lower().split())
        insights_b = set(" ".join(getattr(skill_b, 'insights', [])).lower().split())
        shared_insights = insights_a & insights_b - {
            "the", "a", "an", "is", "are", "and", "or", "in", "of", "to"
        }
        if len(shared_insights) >= 2:
            return "shared insight: " + ", ".join(list(shared_insights)[:3])

        # Unresolved concept bridges the two
        name_a_words = set(name_a.lower().replace("_", " ").split())
        name_b_words = set(name_b.lower().replace("_", " ").split())
        for concept in concepts:
            concept_words = set(concept.split())
            if (concept_words & name_a_words) and (concept_words & name_b_words):
                return f"unresolved concept: {concept}"

        return ""

    # ── LLM synthesis ─────────────────────────────────────────────────────────

    def _synthesise(
        self,
        candidate: SynthesisCandidate,
        skills:    Dict[str, Any],
        beliefs:   List[str],
        slow_cycle: int,
    ) -> Optional[SynthesisRecord]:
        """
        Ask the LLM: given these two capabilities and their bridge,
        what new thing becomes possible?
        """
        skill_a = skills.get(candidate.skill_a)
        skill_b = skills.get(candidate.skill_b)

        insights_a = getattr(skill_a, 'insights', [])[:3] if skill_a else []
        insights_b = getattr(skill_b, 'insights', [])[:3] if skill_b else []
        top_beliefs = beliefs[:4]

        prompt = (
            f"You are the aspirational synthesis process of an AI cognitive system.\n\n"
            f"EXISTING CAPABILITIES:\n"
            f"  {candidate.skill_a} "
            f"(depth={getattr(skill_a, 'depth', 0)}, "
            f"confidence={getattr(skill_a, 'confidence', 0):.2f})\n"
            + (f"  Insights: {'; '.join(insights_a)}\n" if insights_a else "")
            + f"  {candidate.skill_b} "
            f"(depth={getattr(skill_b, 'depth', 0)}, "
            f"confidence={getattr(skill_b, 'confidence', 0):.2f})\n"
            + (f"  Insights: {'; '.join(insights_b)}\n" if insights_b else "")
            + f"\nBRIDGE CONNECTING THEM: {candidate.concept_bridge}\n"
            + (f"\nCORE BELIEFS:\n  " + "\n  ".join(top_beliefs) + "\n" if top_beliefs else "")
            + f"\nTASK:\n"
            f"What new capability or understanding becomes possible by combining "
            f"'{candidate.skill_a}' and '{candidate.skill_b}'?\n\n"
            f"This must be something that does not yet exist — a genuinely new "
            f"possible self-state, not just improving what already exists.\n"
            f"Be specific. Name the new capability in 1-2 sentences.\n"
            f"Then estimate how desirable this new state is (0.0–1.0).\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{"description": "...", "state_desired": 0.75}}'
        )

        try:
            from core.llm_scheduler import llm_scheduler
            result = ""
            with llm_scheduler.sync_slot(
                priority     = 4,
                skip_if_busy = True,
                caller       = "aspirational_synthesis",
            ) as acquired:
                if not acquired:
                    return None
                result = self._ai.get_response(
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 120,
                    temperature = 0.85,
                ) or ""

            if not result:
                return None

            clean = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)

            return SynthesisRecord(
                timestamp     = time.time(),
                skill_a       = candidate.skill_a,
                skill_b       = candidate.skill_b,
                description   = data.get("description", "")[:300],
                state_desired = max(0.0, min(1.0, float(data.get("state_desired", 0.65)))),
                desirability  = candidate.desirability,
                injected      = False,
            )

        except Exception as e:
            logger.debug(f"[AspirationalSynthesis] LLM synthesis failed: {e}")
            return None

    # ── Injection into AspirationalSelf ──────────────────────────────────────

    def _inject(self, record: SynthesisRecord) -> None:
        """Inject the synthesised aspiration into the AspirationalSelf."""
        try:
            asp = getattr(self._organism, 'aspirational_self', None)
            if not asp:
                return

            # Create a proper Aspiration object
            from cognition.aspirational_self import Aspiration
            import uuid

            domain = f"synthesis_{record.skill_a}_{record.skill_b}"[:40]
            aspiration = Aspiration(
                domain         = domain,
                description    = record.description,
                state_current  = 0.0,   # doesn't exist yet
                state_desired  = record.state_desired,
                tension        = record.state_desired,   # full gap
                emotional_charge = "anticipation",
                origin_story   = (
                    f"Synthesised from {record.skill_a} and {record.skill_b} "
                    f"via {record.desirability:.2f} desirability score."
                ),
                conscious      = True,   # explicitly named
            )

            with asp._lock:
                asp.aspirations[domain] = aspiration

            record.injected = True
            logger.debug(
                f"[AspirationalSynthesis] Injected '{domain}' into AspirationalSelf"
            )

        except Exception as e:
            logger.debug(f"[AspirationalSynthesis] Injection failed: {e}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "log":   self._log,
                    "_meta": {"version": "v54", "ts": time.time()},
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[AspirationalSynthesis] Save failed: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            with self._lock:
                self._log = data.get("log", [])
        except Exception as e:
            logger.warning(f"[AspirationalSynthesis] Load failed: {e}")
