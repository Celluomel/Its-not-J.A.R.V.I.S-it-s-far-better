"""
Execution Layer
===============
Translates activity names into actual calls on Lumina's cognitive modules.

The orchestrator calls execute(activity, payload) and this layer does the work,
using whatever modules are available on the CognitiveOrganism.

Activities and their implementations:

  respond_user          → queued for the UI handler (puts in persona_queue)
  respond_external      → calls MessagingManager (if attached)
  restore_energy        → calls organism._fast_cycle() (regen + decay)
  stabilise_homeostasis → runs homeostasis evaluation + logs correctives
  resolve_contradiction → triggers contradiction handler sweep
  reflect               → runs organism._slow_cycle() + reflection
  consolidate_memory    → sweeps memory manager for patterns
  evolve                → triggers personality / self-concept update
  explore_curiosity     → stimulates curiosity engine with top topic
  pursue_goal           → records urgency in goal ecology
  idle_reflection       → lightweight background cycle

All activities are fire-and-forget from the orchestrator's perspective.
Heavy LLM calls are NOT made here — those happen in the persona_bridge pipeline
when the UI drains the persona_queue.
"""

import asyncio
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cognition.cognitive_organism import CognitiveOrganism
    from cognition.global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)


class ExecutionLayer:
    """
    Runs activities on behalf of the orchestrator.

    Parameters
    ----------
    organism  : CognitiveOrganism
    workspace : GlobalWorkspace
    """

    def __init__(self, organism: Any, workspace: Any):
        self._o  = organism
        self._ws = workspace
        # Optional: response queue for UI to drain (set by app.py)
        self.persona_queue: asyncio.Queue = asyncio.Queue()
        # Optional messaging manager (set externally for Telegram/WhatsApp)
        self.messaging_manager: Optional[Any] = None

        # ── Workspace broadcast dedup ──────────────────────────────────────
        # Prevents same source+content flooding the workspace every 2s
        # source_key → (last_content_sig, last_emit_time)
        self._ws_last: dict = {}
        # Minimum seconds between broadcasts from the same source
        self._WS_COOLDOWN = {
            "goal_ecology":  30.0,   # goal status: at most once per 30s
            "physiology":    30.0,   # energy/rest: at most once per 30s
            "idle":          20.0,
            "curiosity":     15.0,
            "tension":       30.0,
            "contradiction": 20.0,
            "meta_cognition":60.0,
            "memory":        60.0,
            "personality":   120.0,
        }
        self._WS_DEFAULT_COOLDOWN = 10.0

    # ── Main dispatch ─────────────────────────────────────────────────────

    async def execute(self, activity: str, payload: Any = None) -> None:
        try:
            if activity == "respond_user":
                await self._respond_user(payload)
            elif activity == "respond_external":
                await self._respond_external(payload)
            elif activity == "restore_energy":
                await self._restore_energy()
            elif activity == "stabilise_homeostasis":
                await self._stabilise_homeostasis()
            elif activity == "resolve_contradiction":
                await self._resolve_contradiction()
            elif activity == "reflect":
                await self._reflect()
            elif activity == "consolidate_memory":
                await self._consolidate_memory()
            elif activity == "evolve":
                await self._evolve()
            elif activity == "explore_curiosity":
                await self._explore_curiosity(payload)
            elif activity == "pursue_goal":
                await self._pursue_goal()
            elif activity == "goal_exploration":
                await self._goal_exploration(payload)
            elif activity == "idle_reflection":
                await self._idle_reflection(payload)
            else:
                logger.debug(f"[ExecutionLayer] unknown activity: {activity!r}")
        except Exception as e:
            logger.warning(f"[ExecutionLayer] {activity} failed: {e}")

    # ── Activity implementations ──────────────────────────────────────────

    async def _respond_user(self, payload: Any) -> None:
        """
        User message: put it in the persona_queue so the UI handler can
        pick it up and run the full persona_bridge pipeline.
        The UI's check_transcription_queue already handles this path —
        we only need to queue it here if the message came via the orchestrator
        (e.g., from external messaging).
        """
        if payload and payload.get("proactive"):
            # Proactive message — queue a None to signal the UI to initiate
            await self.persona_queue.put({"proactive": True, "user_id": "default"})
            self._ws.broadcast("orchestrator", "proactive trigger queued", priority=0.6)
        # Regular user messages go through the existing UI pipeline — no action needed

    async def _respond_external(self, payload: Any) -> None:
        """Route external platform message through MessagingManager."""
        if not self.messaging_manager or not payload:
            return
        platform = payload.get("platform", "unknown")
        user_id  = payload.get("user_id", "external")
        text     = payload.get("text", "")
        logger.info(f"[ExecutionLayer] external message from {platform}/{user_id}")
        try:
            await self.messaging_manager.receive_message(platform, user_id, text)
        except Exception as e:
            logger.error(f"[ExecutionLayer] messaging error: {e}")

    async def _restore_energy(self) -> None:
        """Trigger a fast maintenance cycle — regen energy, decay stale states."""
        try:
            loop = getattr(self._o, '_loop', None)
            if loop:
                await asyncio.to_thread(loop._fast_cycle)
                self._ws_broadcast("physiology", "energy restoration cycle", priority=0.3)
                logger.debug("[ExecutionLayer] restore_energy: fast cycle ran")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] restore_energy: {e}")

    async def _stabilise_homeostasis(self) -> None:
        """Evaluate homeostasis and log corrective actions."""
        try:
            h = self._o.homeostasis
            report = getattr(h, '_last_report', None)
            if report and not report.is_balanced():
                actions = report.corrective_actions[:2]
                self._ws.broadcast("homeostasis", f"corrective: {actions}", priority=0.7)
                logger.info(f"[ExecutionLayer] homeostasis correctives: {actions}")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] stabilise_homeostasis: {e}")

    async def _resolve_contradiction(self) -> None:
        """Ask the contradiction handler to sweep recent memories."""
        try:
            ch = getattr(self._o, 'contradiction_handler', None) or \
                 getattr(self._o.ai_system, 'contradiction_handler', None)
            if ch and hasattr(ch, 'sweep'):
                await asyncio.to_thread(ch.sweep)
                self._ws.broadcast("contradiction", "contradiction sweep completed", priority=0.6)
        except Exception as e:
            logger.debug(f"[ExecutionLayer] resolve_contradiction: {e}")

    async def _reflect(self) -> None:
        """Run a slow + reflection cycle on the internal loop."""
        try:
            loop = getattr(self._o, '_loop', None)
            if loop:
                await asyncio.to_thread(loop._slow_cycle)
                self._ws.broadcast("meta_cognition", "reflection cycle completed", priority=0.4)
                logger.debug("[ExecutionLayer] reflect: slow cycle ran")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] reflect: {e}")

    async def _consolidate_memory(self) -> None:
        """Trigger memory consolidation if available."""
        try:
            mm = getattr(self._o, 'memory_manager', None) or \
                 getattr(self._o.ai_system, 'memory_manager', None)
            if mm:
                fn = getattr(mm, 'consolidate', None) or getattr(mm, 'analyze', None)
                if fn:
                    await asyncio.to_thread(fn)
                    self._ws.broadcast("memory", "consolidation cycle completed", priority=0.35)
                    logger.debug("[ExecutionLayer] consolidate_memory: done")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] consolidate_memory: {e}")

    async def _evolve(self) -> None:
        """Trigger personality evolution / self-concept update."""
        try:
            # PersonalityEvolution
            pe = getattr(self._o.ai_system, 'personality', None)
            if pe and hasattr(pe, 'evolve'):
                await asyncio.to_thread(pe.evolve)
                self._ws.broadcast("personality", "evolution cycle", priority=0.3)
            # SelfConcept
            sc = getattr(self._o.ai_system, 'self_concept', None)
            if sc and hasattr(sc, 'update_coherence'):
                await asyncio.to_thread(sc.update_coherence)
                self._ws.broadcast("self_concept", "coherence updated", priority=0.3)
            # ── v3: NarrativeIdentity sync during evolution ──────────
            ni = getattr(self._o, 'narrative_identity', None)
            if ni:
                await asyncio.to_thread(ni.sync_from_organism)
                self._ws.broadcast("narrative_identity", "identity synced", priority=0.25)

            logger.debug("[ExecutionLayer] evolve: done")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] evolve: {e}")

    async def _goal_exploration(self, payload: Any) -> None:
        """Advance the dominant TTE thread — broadcasts to GW and marks curiosity."""
        try:
            topic = (payload or {}).get("topic", "")
            goal  = (payload or {}).get("goal",  "")
            if topic:
                self._ws_broadcast(
                    "goal_exploration",
                    f"pursuing: {topic[:60]}",
                    priority=0.60,
                )
                # Also stimulate curiosity so the engine tracks this topic
                cu = getattr(self._o, 'curiosity', None)
                if cu and topic:
                    cu.stimulate(topic[:40], amount=0.08, source="goal_exploration")
                logger.debug(f"[ExecutionLayer] goal_exploration: {topic!r} (goal={goal!r})")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] goal_exploration: {e}")

    async def _explore_curiosity(self, payload: Any) -> None:
        """Log curiosity topic to workspace — do NOT re-stimulate (prevents runaway loop)."""
        try:
            cu = self._o.curiosity
            topic = cu.top_topic()
            if topic:
                # Mark as researched so urgency drops and the cycle can move on
                cu.mark_researched(topic)
                self._ws.broadcast("curiosity", f"noted topic: {topic}", priority=0.45)
                logger.debug(f"[ExecutionLayer] explore_curiosity noted: {topic!r}")
            else:
                # No topics — decay global to baseline
                cu.decay_all()
                self._ws.broadcast("curiosity", "no topics — decaying", priority=0.2)
        except Exception as e:
            logger.debug(f"[ExecutionLayer] explore_curiosity: {e}")

    def _ws_broadcast(self, source: str, content: str, priority: float) -> bool:
        """
        Rate-limited workspace broadcast.
        Returns True if emitted, False if suppressed by cooldown or dedup.
        """
        import time as _t
        now = _t.time()
        cooldown = self._WS_COOLDOWN.get(source, self._WS_DEFAULT_COOLDOWN)
        last_content, last_time = self._ws_last.get(source, ("", 0.0))
        sig = content[:60]
        # Skip if same content AND within cooldown window
        if sig == last_content and (now - last_time) < cooldown:
            return False
        # Skip if different content but still within half the cooldown
        if sig != last_content and (now - last_time) < cooldown * 0.5:
            return False
        self._ws.broadcast(source, content, priority=priority)
        self._ws_last[source] = (sig, now)
        return True

    async def _pursue_goal(self) -> None:
        """Record that the dominant goal needs attention — rate limited."""
        try:
            ge = self._o.goal_ecology
            dominant = ge.dominant_drive(self._o.energy.level())
            emitted = self._ws_broadcast(
                "goal_ecology",
                f"pursuing: {dominant.name} urgency={dominant.urgency:.2f}",
                priority=0.5,
            )
            if emitted:
                logger.debug(f"[ExecutionLayer] pursue_goal: {dominant.name}")
        except Exception as e:
            logger.debug(f"[ExecutionLayer] pursue_goal: {e}")

    async def _idle_reflection(self, payload: Any) -> None:
        """Very lightweight — broadcast current mood to workspace (rate limited)."""
        try:
            social_hint = (payload or {}).get("social_hint", False)
            energy = self._o.energy.level()
            self._ws_broadcast(
                "idle",
                f"idle reflection — energy={energy:.0f}% social_hint={social_hint}",
                priority=0.2,
            )
        except Exception as e:
            logger.debug(f"[ExecutionLayer] idle_reflection: {e}")
