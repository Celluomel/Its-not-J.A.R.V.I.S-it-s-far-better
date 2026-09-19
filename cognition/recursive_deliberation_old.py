"""
cognition/recursive_deliberation.py

Phase 4.3 — Recursive Deliberation Workspace, extended to Parallel
Deliberation + Synthesis (v84).

v83 kept a top-3 list structurally but only ever RE-SCORED the same
candidates across cycles toward a single winner — closer to "improve A"
than genuine multi-hypothesis reasoning. This version adds what was
actually missing: candidates are evaluated on separate, kept-apart
dimensions (not collapsed into one number until comparison time), the
top-2 are checked for compatibility, and when they're compatible and
close contenders, a genuinely NEW candidate is synthesized from both —
selection becomes creation, not just choice among existing options.

Design principle (unchanged from v83, still the reason this is feasible
on a local 7B model): the recursion and the synthesis both happen across
PandoraBOX's own cognitive modules — WorkspaceCompetition, ExecutiveArbitration's
multi-factor scoring — never across extra LLM generations. Synthesis here
is STRUCTURAL (a new candidate object: merged topic, merged label,
combined priority) not linguistic — the actual prose fusion ("help Fred
while learning from the interaction") happens for free when the single,
already-scheduled response-generation LLM call reads a workspace fragment
that clearly frames the synthesis, not from an extra call in this loop.

This deliberately does NOT reuse Phase1Orchestrator (core/phase1_
integration.py) — that class loads goals/thoughts/tensions from JSON
files on a background cadence and has its own periodic side effects
(thread resolution every 10 cycles, goal creation every 5). Calling it
synchronously per-turn would both be slower than needed and risk
interleaving its cycle_count-gated side effects with a live response.
This module reads live organism state directly instead, and uses a
DEDICATED WorkspaceCompetition instance (not the one, if any, the
background loop is already using) so per-turn calls don't interfere
with the background loop's own focus-persistence tracking
(self.current_focus / cycles_on_current_focus / attention_history).

Bounded by design:
  - max_cycles (default 3): hard ceiling, matching the "usually 2-4
    iterations are sufficient, don't use unlimited recursion" principle.
  - early stop when utility improvement between cycles falls below
    MIN_IMPROVEMENT — diminishing returns, don't spend cycles for
    negligible gain.
  - synthesis is attempted at most once per deliberate() call (not once
    per cycle) — it's evaluated fresh each cycle against the current
    top-2, but there's no unbounded "synthesize a synthesis of a
    synthesis" recursion; a synthesized candidate competes on equal
    footing with originals in subsequent cycles, it doesn't get special
    re-synthesis treatment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_CYCLES = 3
MIN_IMPROVEMENT = 0.02   # stop early if utility gain between cycles is below this
SYNTHESIS_UTILITY_MARGIN = 0.10   # top-2 must be within this utility gap to be "close contenders"

# One dedicated WorkspaceCompetition instance for this module, separate
# from anything core/phase1_integration.py's background loop uses.
_deliberation_competition = None


def _get_competition_instance():
    global _deliberation_competition
    if _deliberation_competition is None:
        from cognition.workspace_competition import WorkspaceCompetition
        _deliberation_competition = WorkspaceCompetition()
    return _deliberation_competition


def _gather_live_goals(organism: Any) -> List[Dict]:
    """Same live-state goal gathering as workspace_state_sync.py, for
    consistency — GoalEngine.get_active_goals(), not the JSON-file
    loader Phase1Orchestrator uses."""
    try:
        ge = getattr(getattr(organism, "ai_system", None), "goal_engine", None)
        if ge and hasattr(ge, "get_active_goals"):
            active = ge.get_active_goals() or []
            return [
                {"id": g.id, "name": getattr(g, "name", g.id), "topic": g.topic,
                 "priority": g.priority, "energy": g.energy,
                 "urgency": getattr(g, "urgency", 0.0),
                 "quality_score": getattr(g, "quality_score", 0.5)}
                for g in active
            ]
    except Exception as e:
        logger.debug(f"[RecursiveDeliberation] goal gathering failed (non-fatal): {e}")
    return []


def _gather_live_percepts(organism: Any) -> List[Dict]:
    """
    Phase 6.2 — recent salient percepts (camera, and any future sensor
    routed through the same connector) as thought-candidates. The
    `thoughts` parameter of WorkspaceCompetition.compete() already exists
    for exactly this and was always called with thoughts=[] before this —
    not a new competition mechanism, just its first real client.
    """
    try:
        uc = getattr(organism, "_universal_connector", None)
        if uc and hasattr(uc, "pending_thought_candidates"):
            return uc.pending_thought_candidates()
    except Exception as e:
        logger.debug(f"[RecursiveDeliberation] percept gathering failed (non-fatal): {e}")
    return []


def _gather_live_symbols(organism: Any) -> List[Dict]:
    """
    Phase 6.14 — sufficiently confident symbols (symbol_system.py) as
    thought-candidates, same real competition mechanism as percepts
    above. Not a new pathway — the same thoughts= slot, same fair
    scoring."""
    try:
        from cognition.symbol_system import get_symbol_system
        sym_sys = get_symbol_system(organism)
        return sym_sys.pending_thought_candidates()
    except Exception as e:
        logger.debug(f"[RecursiveDeliberation] symbol gathering failed (non-fatal): {e}")
    return []


def _gather_live_tensions(organism: Any) -> Dict[str, float]:
    try:
        te = getattr(organism, "tension_engine", None)
        if te and hasattr(te, "current"):
            return te.current().as_dict()
    except Exception as e:
        logger.debug(f"[RecursiveDeliberation] tension gathering failed (non-fatal): {e}")
    return {}


def _gather_live_pressure(organism: Any) -> Dict[str, float]:
    """Best-effort pressure dict — compete() reads these with safe
    .get(key, default) fallbacks throughout, so a partial/empty dict
    degrades gracefully rather than breaking anything."""
    pressure = {}
    try:
        econ = getattr(getattr(organism, "_loop", None), "_resource_economy", None)
        if econ and hasattr(econ, "status"):
            st = econ.status()
            pressure["cognitive_pressure"] = 1.0 - st.get("cognitive_energy", 0.75)
            pressure["social_pressure"]    = 1.0 - st.get("social_energy", 0.75)
    except Exception:
        pass
    try:
        tensions = _gather_live_tensions(organism)
        if tensions:
            pressure["goal_pressure"] = tensions.get("goal_pressure", 0.5)
    except Exception:
        pass
    try:
        # Phase 6.10 — same real PressureSystem reservoir Phase 5.1's
        # GoalEngine.get_current_pressures() reads (organism.pressure,
        # not ai_system.pressure — verified attribute path). Needed for
        # InternalCognitiveState's update() in deliberate().
        ps = getattr(organism, "pressure", None)
        if ps and hasattr(ps, "reservoirs"):
            epi = ps.reservoirs.get("epistemic")
            if epi is not None:
                pressure["epistemic"] = epi.level
    except Exception:
        pass
    return pressure


def _profile_candidate(c: Dict, ws_state: Any, prev_focus: Optional[str]) -> Dict[str, float]:
    """
    Per-dimension profile for one candidate, kept apart rather than
    immediately collapsed — this is the "simulate consequences / estimate
    uncertainty / estimate identity alignment / estimate goal alignment /
    estimate novelty" step. Reuses the existing, already-tested
    _score_extra_factors() (counterfactual match, identity, goal_
    alignment, calibration) and adds novelty, which didn't exist before:
    a candidate whose topic is literally what the workspace was already
    focused on last turn is not novel; a fresh topic is.
    """
    from cognition.executive_arbitration import _score_extra_factors
    extra = _score_extra_factors(c, ws_state)
    topic = (c.get("topic") or c.get("name") or "").lower()
    novelty = 0.2 if (prev_focus and topic and topic in str(prev_focus).lower()) else 0.8
    return {**extra, "novelty": novelty}


def _compute_utility(raw_score: float, max_raw: float, profile: Dict[str, float]) -> float:
    # Phase 5.4: use the learned weights (UTILITY_WEIGHTS + bounded delta
    # from ArbitrationLearningTracker), not the static dict directly —
    # this and executive_arbitration.arbitrate() both score "which
    # candidate deserves focus" from the same UTILITY_WEIGHTS source, so
    # leaving this one unlearned would silently fork the two once learning
    # actually moves a weight. Already imports a private helper from this
    # module below (_score_extra_factors), so this follows the same
    # existing pattern, not a new one.
    from cognition.executive_arbitration import _get_base_weights
    weights = _get_base_weights()
    activation_norm = raw_score / max_raw if max_raw else 0.0
    # novelty isn't in UTILITY_WEIGHTS (it's new, v84) — give it a small,
    # explicit weight taken from calibration's share rather than inventing
    # a new total that could silently stop summing to 1.0 elsewhere.
    novelty_weight = 0.05
    base = weights["activation"] * activation_norm + sum(
        weights[k] * v for k, v in profile.items() if k in weights
    )
    return base + novelty_weight * profile.get("novelty", 0.0)


def _synthesis_eligible(a: Dict, b: Dict) -> bool:
    """Only merge two real goal-type candidates with genuinely different
    topics — merging a goal with a raw tension/thought, or merging a
    candidate with itself, isn't a meaningful synthesis."""
    if a.get("type") != "goal" or b.get("type") != "goal":
        return False
    topic_a = (a.get("topic") or "").lower()
    topic_b = (b.get("topic") or "").lower()
    return bool(topic_a) and bool(topic_b) and topic_a != topic_b


def _synthesize(a: Dict, b: Dict) -> Dict:
    """
    Structural synthesis: a genuinely new candidate object combining two
    compatible ones. Also carries a prompt_hint — see below; a real
    "act as the LLM" simulation found the bare mechanical label alone
    wasn't reliably enough for a small local model to actually fuse both
    halves in its response, so this no longer counts on that happening
    "for free".

    Score includes a small explicit synergy bonus (+8%) over the average
    of the two originals — addressing two real drives with one action is
    genuinely more valuable than addressing either alone, and without
    this bonus a synthesized candidate could only ever tie the better
    original at best (max(a,b)), never actually win, which would make
    synthesis structurally unable to ever be selected.
    """
    label_a = a.get("label") or a.get("name") or a.get("topic")
    label_b = b.get("label") or b.get("name") or b.get("topic")
    # For the natural-language hint specifically, prefer topic — label is
    # documented (workspace_competition.py) as "a full readable sentence
    # for logs/dashboard" (e.g. "Goal: help the user (p=0.70)"), which
    # reads as diagnostic text inside a hint meant for the LLM, not the
    # clean subject a hint sentence needs.
    hint_a = a.get("topic") or a.get("name") or label_a
    hint_b = b.get("topic") or b.get("name") or label_b
    # Light readability normalization only — real GoalEngine topics are
    # often slug-like (e.g. "curiosity_drive", "housekeeping"), not
    # invented or reworded, just de-slugged for a hint sentence.
    if hint_a:
        hint_a = str(hint_a).replace("_", " ")
    if hint_b:
        hint_b = str(hint_b).replace("_", " ")
    synergy_score = ((a.get("score", 0) + b.get("score", 0)) / 2.0) * 1.08
    return {
        "id": f"synth_{a.get('id','a')}_{b.get('id','b')}",
        "name": f"{a.get('topic')}+{b.get('topic')}",
        "label": f"{label_a} — while also {label_b}",
        # Space, not '+': _tokens() only splits on underscores/whitespace,
        # so '+' was gluing the second topic into one unmatchable token
        # (e.g. "social_drive+curiosity_drive" -> {'drive+curiosity', ...}
        # instead of {'social','drive','curiosity'}), silently breaking
        # the identity/goal_alignment terms for every synthesized candidate.
        "topic": f"{a.get('topic')} {b.get('topic')}",
        "type": "goal",
        "score": synergy_score,
        "source_data": {"synthesized_from": [a.get("id"), b.get("id")]},
        # A genuine "act as the LLM" simulation (not just static reading)
        # showed the docstring's "fusion happens for free" assumption was
        # optimistic for a small local model: "X — while also Y" reached
        # the prompt as a bare mechanical label (often snake_case drive
        # names), and the model had no explicit instruction for what to
        # DO with two foci at once — it tended to lean on whichever half
        # read as more natural language and underweight the other. This
        # hint gives the response-generation prompt fragment an explicit,
        # general instruction (works regardless of what a/b actually are —
        # real goal topics, drive names, or percept labels) rather than
        # counting on the LLM to correctly infer the fusion unprompted.
        "prompt_hint": (
            f"Two things matter here at once: {hint_a}, and {hint_b} — "
            f"let your response hold both rather than picking one."
        ),
    }


def deliberate(
    organism: Any,
    user_input: str,
    max_cycles: int = MAX_CYCLES,
) -> Optional[Dict]:
    """
    Run a bounded (1-3 cycle) synchronous parallel-deliberation pass:
    keeps the top candidates' evaluation dimensions separate, checks
    whether the top-2 are compatible and close enough to be worth
    combining, and — if so — synthesizes a genuinely new candidate that
    competes alongside the originals rather than the loop only ever
    selecting among pre-existing options.

    Publishes into organism.workspace's WorkspaceState — the same
    GlobalWorkspace that _build_prompt_additions() already reads to
    build the "[Global workspace] focus: ..." fragment (wired in v67).

    Returns a small summary dict for logging/dashboard use, or None if
    there was nothing to deliberate over this turn (e.g. no active goals
    — not an error, just nothing to refine).
    """
    gw = getattr(organism, "workspace", None)
    if gw is None:
        return None

    goals = _gather_live_goals(organism)
    percepts = _gather_live_percepts(organism)
    symbols = _gather_live_symbols(organism)
    thought_candidates = percepts + symbols
    if not goals and not thought_candidates:
        return None   # nothing to compete over — a quiet, expected no-op

    tensions = _gather_live_tensions(organism)
    pressure = _gather_live_pressure(organism)
    competition = _get_competition_instance()

    # Phase 6.10 — update the organism's InternalCognitiveState once per
    # deliberation call (not once per inner cycle below — the epistemic
    # reading doesn't change within a single bounded deliberation pass,
    # and updating it multiple times would corrupt the pressure_delta/
    # meta_error trend calculation, which is meant to track cognitive
    # cycles, not competition-loop iterations).
    internal_state = None
    try:
        from cognition.internal_cognitive_state import get_internal_cognitive_state
        internal_state = get_internal_cognitive_state(organism)
        _epi = pressure.get("epistemic", 0.0)
        internal_state.update(_epi)
    except Exception as e:
        logger.debug(f"[RecursiveDeliberation] internal_cognitive_state update failed (non-fatal): {e}")
        internal_state = None

    # Phase 6.12 — the recursive crossing (epistemic_efficacy_model.py).
    # Observe any prediction the self-model made when resolve_uncertainty
    # last won, BEFORE this cycle's own scoring — the self-model's
    # accuracy about ITS OWN LAST prediction has to be settled before it
    # gets to influence this cycle's competition, or the loop would be
    # scoring against evidence from the future.
    efficacy_model = None
    try:
        from cognition.epistemic_efficacy_model import get_epistemic_efficacy_model
        efficacy_model = get_epistemic_efficacy_model(organism)
        if efficacy_model is not None:
            efficacy_model.observe_if_pending(_epi)
    except Exception as e:
        logger.debug(f"[RecursiveDeliberation] epistemic_efficacy_model observe failed (non-fatal): {e}")

    # Phase 6.15 — adaptive reflection depth (reflection_controller.py).
    # Only overrides max_cycles when the caller relied on the static
    # default — an explicit caller-provided value is still respected as
    # a real ceiling, per "make bounds adaptive, not replace them".
    effective_max_cycles = max_cycles
    if max_cycles == MAX_CYCLES:
        try:
            from cognition.reflection_controller import get_reflection_controller
            meta_err = internal_state.history[-1]["meta_error"] if (internal_state and internal_state.history) else 0.0
            effective_max_cycles = get_reflection_controller().allowed_depth(meta_err)
        except Exception as e:
            logger.debug(f"[RecursiveDeliberation] reflection_controller failed (non-fatal): {e}")
            effective_max_cycles = max_cycles
        efficacy_model = None

    ws_state = gw.get_state()
    prev_focus = ws_state.focus

    prev_utility = None
    cycles_run = 0
    final_ranked: List[Tuple[float, Dict]] = []
    synthesis_used: Optional[Dict] = None

    for cycle in range(1, effective_max_cycles + 1):
        cycles_run = cycle
        try:
            result = competition.compete(
                goals=goals, thoughts=thought_candidates, tensions=tensions,
                pressure=pressure, context={"user_input": user_input},
                internal_state=internal_state, efficacy_model=efficacy_model,
            )
        except Exception as e:
            logger.debug(f"[RecursiveDeliberation] compete() failed cycle {cycle} (non-fatal): {e}")
            break

        candidates = result.get("competition", {}).get("all_candidates", [])
        if not candidates:
            break

        max_raw = max((c.get("score", 0) for c in candidates), default=1.0) or 1.0

        # ── Parallel evaluation: profile each candidate on separate
        # dimensions before any collapsing/comparison — "compare all
        # three simultaneously", not narrow-then-refine-one.
        profiled = [(c, _profile_candidate(c, ws_state, prev_focus)) for c in candidates]
        scored = [(_compute_utility(c.get("score", 0), max_raw, p), c) for c, p in profiled]
        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Synthesis: is it worth letting the top-2 influence each
        # other instead of just picking one?
        if len(scored) >= 2:
            (u1, c1), (u2, c2) = scored[0], scored[1]
            if abs(u1 - u2) <= SYNTHESIS_UTILITY_MARGIN and _synthesis_eligible(c1, c2):
                d = _synthesize(c1, c2)
                d_profile = _profile_candidate(d, ws_state, prev_focus)
                d_utility = _compute_utility(d.get("score", 0), max_raw, d_profile)
                scored.append((d_utility, d))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored[0][1] is d:
                    synthesis_used = d
                    logger.debug(
                        f"[RecursiveDeliberation] synthesized '{d['label']}' "
                        f"from '{c1.get('label')}' + '{c2.get('label')}', "
                        f"utility={d_utility:.3f} beat both originals "
                        f"({u1:.3f}, {u2:.3f})"
                    )
                else:
                    synthesis_used = None   # synthesized but didn't win — don't claim it did

        final_ranked = scored
        top_utility = scored[0][0] if scored else 0.0

        gw.set_hypotheses(
            "recursive_deliberation",
            [{**c, "score": round(u, 4)} for u, c in scored[:3]],
        )
        ws_state = gw.get_state()

        if prev_utility is not None and (top_utility - prev_utility) < MIN_IMPROVEMENT:
            break   # diminishing returns — stop early
        prev_utility = top_utility

    if not final_ranked:
        return None

    winner_utility, winner = final_ranked[0]
    _update_kwargs = {"focus": winner.get("name") or winner.get("label")}
    if prev_utility is not None:
        _update_kwargs["uncertainty"] = round(1.0 - prev_utility, 4)
    if winner is synthesis_used:
        _update_kwargs["context"] = {
            **(ws_state.context or {}),
            "synthesis": {
                "label": winner.get("label"),
                "from": winner.get("source_data", {}).get("synthesized_from"),
            },
        }
    gw.update_state("recursive_deliberation", **_update_kwargs)

    # Phase 6.12 — now that the winner is known, let the self-model make
    # its own falsifiable prediction if resolve_uncertainty won this
    # cycle. Deliberately AFTER update_state(), not before — the
    # prediction is about what happens BECAUSE this goal was chosen, so
    # it has to follow the choice being finalized.
    if efficacy_model is not None:
        try:
            efficacy_model.record_prediction_if_winner(
                winner.get("name", ""), winner.get("topic", ""),
                pressure.get("epistemic", 0.0),
            )
        except Exception as e:
            logger.debug(f"[RecursiveDeliberation] efficacy prediction record failed (non-fatal): {e}")

    summary = {
        "cycles_run": cycles_run,
        "final_focus": winner.get("name") or winner.get("label"),
        "final_utility": round(winner_utility, 4),
        "candidates_considered": len(final_ranked),
        "synthesized": winner is synthesis_used,
    }
    logger.debug(f"[RecursiveDeliberation] {summary}")
    return summary
