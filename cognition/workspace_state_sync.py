"""
cognition/workspace_state_sync.py

Phase 4.x — thin adapter layer for GlobalWorkspace.WorkspaceState.

Deliberately centralized in ONE new file instead of editing six existing
modules (world_model, counterfactual_simulator, calibration_engine,
introspective_observer, narrative_identity, goal_engine). Every one of
those already exposes a public status()/summary()/get_metrics()/
prompt_fragment() method — this module just reads those existing outputs
and republishes the relevant pieces into the shared WorkspaceState via
GlobalWorkspace.update_state(). No subsystem internals are touched, no
new prediction/scoring logic is added here — this is pure recomposition,
per the "reuse existing infrastructure" constraint.

Call sync_workspace_state(workspace, organism, internal_loop) once per
slow cycle, after the subsystems' own tick() calls have already run
(internal_loop.py already ticks them in this order — see the call site
added at the same point _v32_workspace_winner is stored).

Every read is wrapped individually so one subsystem being unavailable
(e.g. fresh deployment, module not yet initialized) never blocks the
others from publishing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def sync_workspace_state(workspace, organism: Any = None, internal_loop: Any = None) -> None:
    """
    Pull current status from Lumina's existing cognitive subsystems and
    write it into GlobalWorkspace's WorkspaceState. Best-effort: any
    subsystem that isn't present or raises is skipped silently (logged at
    debug level), never blocking the others or the caller's cycle.
    """
    if workspace is None:
        return

    # ── World Model (3.2) → predicted_futures / context ────────────────────
    try:
        wm = getattr(organism, "world_model", None)
        if wm is not None and hasattr(wm, "summary"):
            wm_summary = wm.summary()
            workspace.update_state("world_model", context={"world_model": wm_summary})
    except Exception as e:
        logger.debug(f"[WorkspaceSync] world_model skipped: {e}")

    # ── Counterfactual Simulator (3.3) → predicted_futures ─────────────────
    try:
        cfs = getattr(internal_loop, "_counterfactual_sim", None)
        if cfs is not None and hasattr(cfs, "recent_findings"):
            findings = cfs.recent_findings(n=5) or []
            workspace.update_state("counterfactual_simulator", predicted_futures=findings)
    except Exception as e:
        logger.debug(f"[WorkspaceSync] counterfactual_simulator skipped: {e}")

    # ── Calibration Engine (3.1) → confidence ───────────────────────────────
    # Note (Phase 4.x audit): CalibrationEngine is currently instantiated
    # separately inside world_self_dynamics_model.py, counterfactual_
    # simulator.py, and predictive_consequence_model.py — three private
    # instances, not one shared engine. This reads whichever one the PCM
    # on the live internal loop holds, since that's the instance already
    # wired into the live chat path via prospective_deliberation. A real
    # fix (sharing one CalibrationEngine instance) is a separate, smaller
    # cleanup this adapter deliberately does not attempt.
    try:
        pcm = getattr(internal_loop, "_consequence_model", None)
        cal = getattr(pcm, "_calibration_engine", None)
        if cal is not None and hasattr(cal, "summary"):
            cal_summary = cal.summary()
            modules = cal_summary.get("modules", {})
            eces = [m["ece"] for m in modules.values() if m.get("ece") is not None]
            if eces:
                mean_ece = sum(eces) / len(eces)
                # Lower ECE = better calibrated = higher confidence proxy.
                confidence = max(0.0, min(1.0, 1.0 - mean_ece))
                workspace.update_state("calibration_engine", confidence=confidence)
    except Exception as e:
        logger.debug(f"[WorkspaceSync] calibration_engine skipped: {e}")

    # ── Introspective Observer (3.5) → intention / uncertainty context ─────
    try:
        obs = getattr(internal_loop, "_introspective_observer", None)
        if obs is not None and hasattr(obs, "status"):
            obs_status = obs.status()
            workspace.update_state("introspective_observer", context={"introspection": obs_status})
    except Exception as e:
        logger.debug(f"[WorkspaceSync] introspective_observer skipped: {e}")

    # ── Narrative Identity → narrative_state ────────────────────────────────
    try:
        narr = getattr(organism, "narrative_identity", None)
        if narr is not None and hasattr(narr, "self_description"):
            workspace.update_state("narrative_identity", narrative_state=narr.self_description())
    except Exception as e:
        logger.debug(f"[WorkspaceSync] narrative_identity skipped: {e}")

    # ── Goal Engine → goals / motivations ───────────────────────────────────
    try:
        ge = getattr(getattr(organism, "ai_system", None), "goal_engine", None)
        if ge is not None and hasattr(ge, "get_active_goals"):
            active = ge.get_active_goals() or []
            goals = [
                {"id": g.id, "topic": g.topic, "priority": g.priority, "energy": g.energy}
                for g in active
            ]
            workspace.update_state("goal_engine", goals=goals)
    except Exception as e:
        logger.debug(f"[WorkspaceSync] goal_engine skipped: {e}")

    # ── Skill Registry → domain confidence (Phase 5.2) ──────────────────────
    # Real per-domain confidence was already tracked (seeded from web
    # research since v76) but never read anywhere outside where it was
    # written. Stored in context so executive_arbitration's calibration
    # term can look up a candidate's actual domain confidence instead of
    # only ever reading one global workspace-level scalar.
    try:
        sr = getattr(organism, "skill_registry", None)
        if sr is not None and hasattr(sr, "all_domain_confidence"):
            workspace.update_state(
                "skill_registry",
                context={
                    **(workspace.get_state().context or {}),
                    "skill_confidence": sr.all_domain_confidence(),
                },
            )
    except Exception as e:
        logger.debug(f"[WorkspaceSync] skill_registry skipped: {e}")

    # Bound hypothesis lifetime every sync pass (see HYPOTHESIS_TTL in
    # global_workspace.py) so this is the one place staleness gets swept,
    # rather than scattering prune calls across callers.
    try:
        workspace.prune_expired_hypotheses()
    except Exception:
        pass
