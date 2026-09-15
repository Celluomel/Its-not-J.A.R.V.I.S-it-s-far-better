"""
cognition/contradiction_belief_reviser.py  (Patch 01)

Closes the contradiction feedback loop: detected self-inconsistencies
feed back into the value/belief system rather than accumulating in a
write-only file.
"""
from __future__ import annotations
import logging, os, threading, time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REVISION_EVERY_N   = 15
REVISION_THRESHOLD = 0.35
MAX_PER_CYCLE      = 2

_BELIEF_TO_VALUE_DIM: Dict[str, str] = {
    "curious": "curiosity", "curiosity": "curiosity",
    "honest":  "honesty",   "honesty":   "honesty",
    "care":    "care",      "caring":    "care",
    "consistent": "consistency", "authentic": "authenticity",
    "creative":   "creativity",  "helpful":   "helpfulness",
}


class ContradictionBeliefReviser:
    def __init__(self, organism: Any, ai_system: Any) -> None:
        self._organism = organism
        self._ai       = ai_system
        self._lock     = threading.Lock()
        self._revisions: List[Dict] = []
        logger.info("[ContradictionReviser] Initialised")

    def tick(self, slow_cycle: int) -> None:
        if slow_cycle % REVISION_EVERY_N != 0 or slow_cycle == 0:
            return
        threading.Thread(target=self._run_revision, args=(slow_cycle,),
                         daemon=True, name="cbr-revision").start()

    def _run_revision(self, slow_cycle: int) -> None:
        try:
            handler = self._get_handler()
            if not handler:
                return
            pending = sorted(
                [c for c in handler.pending_confrontations
                 if c.discrepancy_score >= REVISION_THRESHOLD],
                key=lambda c: -c.discrepancy_score
            )
            for contradiction in pending[:MAX_PER_CYCLE]:
                self._revise_one(contradiction, slow_cycle)
        except Exception as e:
            logger.debug(f"[ContradictionReviser] error: {e}")

    def _revise_one(self, contradiction: Any, slow_cycle: int) -> None:
        try:
            cid, claimed, actual, score = (
                contradiction.contradiction_id,
                contradiction.claimed_belief,
                contradiction.actual_behavior,
                contradiction.discrepancy_score,
            )
            dim = next((v for k, v in _BELIEF_TO_VALUE_DIM.items()
                        if k in claimed.lower()), None)
            dp = getattr(self._ai, "_decision_policy", None)
            if dp and dim:
                dp.update(dim, "negative", strength=min(0.6, score * 0.8))
                logger.info(f"[ContradictionReviser] DecisionPolicy: {dim} ← negative")

            revision = self._llm_revision(claimed, actual, score)
            if revision:
                # NarrativeIdentity.add_belief (signature: belief, confidence, source)
                # SelfConceptSystem uses a bootstrap-only API, not add_belief
                _ni_b = getattr(self._ai, "narrative_identity", None)
                if _ni_b and hasattr(_ni_b, "add_belief"):
                    _ni_b.add_belief(
                        belief     = revision,
                        confidence = max(0.4, 0.8 - score * 0.4),
                        source     = f"contradiction_revision_{slow_cycle}",
                    )
                ge = getattr(self._organism, "goal_ecology", None)
                if ge and hasattr(ge, "record_satisfaction"):
                    ge.record_satisfaction("maintain_coherence", 0.65)

            # Phase 4 repair: contradiction → aspirations
            try:
                _asp = getattr(self._organism, 'aspirational_self', None)
                if _asp and dim:
                    _asp.observe_tension(
                        domain        = f"resolve_{dim}_discrepancy",
                        context       = f"Contradiction detected: {claimed[:80]}",
                        intensity     = min(0.85, score * 1.1),
                        emotional_tag = "dissonance",
                        source        = "contradiction_reviser",
                    )
            except Exception as _asp_e:
                logger.debug(f"[ContradictionReviser] AspirationalSelf tension: {_asp_e}")

            # Phase 4 repair: contradiction → identity themes
            try:
                _ai  = getattr(self._organism, 'ai_system', None)
                _ni  = getattr(_ai, 'narrative_identity', None) if _ai else None
                if _ni and revision and hasattr(_ni, 'record_chapter'):
                    _ni.record_chapter(
                        title       = f"Self-consistency: {dim or 'values'}",
                        description = f"Contradiction resolved: {revision[:120]}",
                        emotion     = "resolution",
                        significance= min(0.75, score),
                    )
            except Exception as _ni_e:
                logger.debug(f"[ContradictionReviser] NarrativeIdentity chapter: {_ni_e}")

            handler = self._get_handler()
            if handler:
                if hasattr(handler, "mark_confronted"):
                    handler.mark_confronted(cid,
                        result=revision or "policy_adjustment_only",
                        resolution="revised_self_model" if revision else "value_policy_adjusted")
                else:
                    for c in handler.contradictions:
                        if c.contradiction_id == cid:
                            c.confronted = True; c.confronted_at = time.time()
                            c.confrontation_result = revision or "policy_adjustment_only"
                            c.resolution = "revised_self_model" if revision else "value_policy_adjusted"
                            break
                    handler.pending_confrontations = [
                        c for c in handler.pending_confrontations if c.contradiction_id != cid]
                    try: handler._save()
                    except Exception: pass

            # ── 2026-09-03: close the loop at the SOURCE ─────────────────────
            # The detector reads ai.self_concept._beliefs (self_concept.json,
            # kept topped up from identity.json by SelfConceptSynchronizer).
            # Previously the "revision" only landed in narrative_identity.json
            # — a store nothing in this loop reads — so the seed belief kept
            # its 0.95 confidence and the contradiction re-fired forever
            # (50 identical entries, May→Sep 2026). Erode the source belief
            # so each confrontation has a real, persistent effect.
            try:
                self._erode_source_belief(claimed)
            except Exception as _er_e:
                logger.debug(f"[ContradictionReviser] source erosion: {_er_e}")
        except Exception as e:
            logger.debug(f"[ContradictionReviser] _revise_one error: {e}")

    # ── 2026-09-03: source-store erosion (the actual loop-closure) ─────────
    def _erode_source_belief(self, claimed: str) -> None:
        """Erode the matched belief in the stores the DETECTOR reads.

        1. SelfConceptSystem._beliefs (self_concept.json) — the direct input
           of ContradictionHandler.detect_contradiction (ai_system builds
           self_concept_beliefs from self.self_concept._beliefs).
        2. identity.json — the sync source; SelfConceptSynchronizer merges it
           INTO self_concept.json, so a stale high-confidence copy there would
           otherwise resurrect the belief.

        Each matched belief's confidence is multiplied by 0.6 (floor 0.20)
        and a violation recorded — a real belief update, not just a queue
        drain. Matching is by statement containment, so the whole
        same-topic cluster (seed + duplicate thread learnings + captured
        parrots) weakens together.
        """
        claimed_l = (claimed or "").strip().lower()
        if not claimed_l:
            return

        def _match(st: Any) -> bool:
            st = (st or "").strip().lower()
            if not st:
                return False
            return claimed_l in st or st in claimed_l

        # 1) In-memory SelfConceptSystem — the detector's input
        sc = getattr(self._ai, "self_concept", None)
        if sc is None and self._organism is not None:
            sc = getattr(getattr(self._organism, "ai_system", None),
                         "self_concept", None)
        if sc is not None:
            beliefs = getattr(sc, "_beliefs", None) or {}
            hit = False
            for _name, b in list(beliefs.items()):
                if _match(getattr(b, "statement", "")):
                    old = float(getattr(b, "confidence", 0.5))
                    new = max(0.20, round(old * 0.6, 4))
                    b.confidence = new
                    try:
                        b.violations = int(getattr(b, "violations", 0)) + 1
                    except Exception:
                        pass
                    hit = True
                    logger.info(
                        f"[ContradictionReviser] source belief eroded: "
                        f"{_name} {old:.2f} → {new:.2f}")
            if hit:
                try:
                    sc._save()
                except Exception as e:
                    logger.debug(f"[ContradictionReviser] self_concept save: {e}")

        # 2) identity.json — the sync source
        ipath = self._identity_json_path()
        if ipath:
            try:
                import json as _json
                import datetime as _dt
                with open(ipath, encoding="utf-8") as f:
                    d = _json.load(f)
                changed = False
                for _name, b in (d.get("beliefs") or {}).items():
                    if isinstance(b, dict) and _match(b.get("statement", "")):
                        old = float(b.get("confidence", 0.5))
                        new = max(0.20, round(old * 0.6, 4))
                        b["confidence"] = new
                        b["updated_at"] = _dt.datetime.now(
                            _dt.timezone.utc).isoformat()
                        changed = True
                        logger.info(
                            f"[ContradictionReviser] identity.json eroded: "
                            f"{_name} {old:.2f} → {new:.2f}")
                if changed:
                    tmp = ipath + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        _json.dump(d, f, indent=2, ensure_ascii=False)
                    os.replace(tmp, ipath)
            except Exception as e:
                logger.debug(f"[ContradictionReviser] identity.json erode: {e}")

    def _identity_json_path(self) -> Optional[str]:
        """Locate data/persona/identity.json relative to the brain's cwd."""
        cfg = getattr(self._ai, "config", None)
        scp = getattr(cfg, "self_concept_path", None)
        d = os.path.dirname(scp) if scp else "data/persona"
        if not d:
            d = "data/persona"
        p = os.path.join(d, "identity.json")
        return p if os.path.exists(p) else None

    def _llm_revision(self, claimed: str, actual: str, score: float) -> Optional[str]:
        try:
            from core.llm_scheduler import llm_scheduler
            prompt = (
                f"Self-inconsistency detected (severity {score:.2f}/1.0).\n"
                f"Claimed belief: \"{claimed}\"\nObserved: \"{actual}\"\n\n"
                f"Write one revised first-person belief statement that resolves "
                f"the inconsistency. No preamble.\n"
                f"Example: \"I am curious, but primarily when cognitive energy is above 40%.\""
            )
            result = ""
            with llm_scheduler.sync_slot(priority=3, skip_if_busy=True,
                                         caller="contradiction_reviser") as acquired:
                if not acquired: return None
                result = self._ai.get_response(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=80, temperature=0.45) or ""
            return result.strip().strip('"') if result else None
        except Exception as e:
            logger.debug(f"[ContradictionReviser] LLM failed: {e}")
            return None

    def _get_handler(self) -> Optional[Any]:
        ai = getattr(self._organism, "ai_system", None) or self._ai
        h  = (getattr(ai, "contradiction_handler", None)
              or getattr(self._organism, "contradiction_handler", None))
        li = getattr(ai, "liberty", None) or getattr(ai, "_liberty", None)
        # ai_system actually stores the handler as `liberty_contradiction`
        h = h or getattr(ai, "liberty_contradiction", None)
        return h or (getattr(li, "contradiction_handler", None) if li else None)
