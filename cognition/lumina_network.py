"""
cognition/lumina_network.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lumina-to-Lumina network connector.

Allows a Master Lumina to:
  • Register one or more Child Lumina instances (brain.py or NiceGUI)
  • Send messages to any child and receive responses
  • Run autonomous dialogues (N-turn exchanges on a topic)
  • Receive unsolicited messages FROM a child (push model)
  • Display all exchanges as chat bubbles in the master UI

Child instances appear as named "users" in the master chat.
Their bubbles use a distinct `bubble-child` CSS class.

Network topology:
  Master (NiceGUI or brain.py)
    ├── Child A  http://192.168.1.10:8765   (brain.py headless)
    ├── Child B  http://192.168.1.11:8080   (NiceGUI — same API)
    └── Child C  http://lumina-pi.local:8765

Config (config.json):
  "LUMINA_CHILDREN": [
    {"id": "child_a", "name": "Lumina-A", "url": "http://192.168.1.10:8765",
     "api_key": "", "role": "child"},
    {"id": "child_b", "name": "Lumina-B", "url": "http://192.168.1.11:8765",
     "api_key": "", "role": "child"}
  ]

Protocol:
  Master → Child:  POST {child_url}/lumina-network/chat
    body: {"text": "...", "sender_id": "master", "sender_name": "Lumina-Master",
           "conversation_id": "uuid", "api_key": "..."}
  Child → Master:  POST {master_url}/lumina-network/receive
    body: {"text": "...", "sender_id": "child_a", "sender_name": "Lumina-A",
           "conversation_id": "uuid"}

Public API:
  net = LuminaNetwork(persona, llm)
  net.add_child(id, name, url, api_key="", role="child")
  await net.send_to_child(child_id, message_text)   → response text
  await net.start_dialogue(child_id, topic, turns=4) → full exchange
  net.set_bubble_fn(fn)           → fn(text, who, child_id, child_name)
  net.list_children()             → list of child dicts
  net.get_child_status(child_id)  → {"online": True, "latency_ms": 42}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Phase 6.5 — minimum interval between workspace-snapshot shares to one
# child. A pragmatic bound against a share -> hypothesis -> share
# amplifier loop, simpler than tracing per-candidate provenance chains.
SNAPSHOT_COOLDOWN_S = 45.0


@dataclass
class ChildLumina:
    """Registered child Lumina instance."""
    id:           str
    name:         str
    url:          str               # base URL, e.g. http://192.168.1.10:8765
    api_key:      str  = ""
    role:         str  = "child"    # "child" | "peer" | "specialist"
    online:       bool = False
    last_ping:    float = 0.0
    latency_ms:   float = 0.0
    turn_count:   int   = 0
    last_snapshot_share: float = 0.0  # Phase 6.5 — cooldown for share_workspace_snapshot()


@dataclass
class NetworkMessage:
    """A single message in a Lumina-to-Lumina exchange."""
    text:            str
    sender_id:       str
    sender_name:     str
    conversation_id: str
    timestamp:       float = field(default_factory=time.time)
    direction:       str   = "out"   # "out" = master→child, "in" = child→master


class LuminaNetwork:
    """
    Lumina-to-Lumina network connector.
    Manages child instances and routes messages bidirectionally.
    """

    def __init__(self, persona: Any, llm: Any):
        self._persona  = persona
        self._llm      = llm
        self._children: Dict[str, ChildLumina] = {}
        self._history:  Dict[str, List[NetworkMessage]] = {}  # conv_id → messages
        self._dialogue_stops: Dict[str, Any] = {}             # child_id → stop Event
        self._user_inject_queue: 'deque' = deque(maxlen=20)
        self._bubble_fn: Optional[Callable] = None  # fn(text, who, child_id, name)
        self._lock = __import__('threading').Lock()
        self._master_url: str = ""   # set when master exposes its own endpoint
        self._master_name: str = "Lumina-Master"

        try:
            import aiohttp
            self._aiohttp = aiohttp
        except ImportError:
            self._aiohttp = None
            logger.warning("⚠️  LuminaNetwork: aiohttp not installed — "
                           "pip install aiohttp --break-system-packages")

        logger.info("🌐 LuminaNetwork initialised")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Child registration
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_child(
        self,
        child_id:  str,
        name:      str,
        url:       str,
        api_key:   str = "",
        role:      str = "child",
    ) -> None:
        url = url.rstrip("/")
        self._children[child_id] = ChildLumina(
            id=child_id, name=name, url=url, api_key=api_key, role=role
        )
        logger.info(f"🌐 Registered child: {name} @ {url}")

    def load_from_config(self, children_config: list) -> None:
        """Load children from config.json LUMINA_CHILDREN list."""
        for c in children_config:
            self.add_child(
                child_id = c.get("id", f"child_{len(self._children)}"),
                name     = c.get("name", "Lumina-Child"),
                url      = c.get("url", ""),
                api_key  = c.get("api_key", ""),
                role     = c.get("role", "child"),
            )

    def set_bubble_fn(self, fn: Callable) -> None:
        """Register the UI bubble callback — called for every exchanged message."""
        self._bubble_fn = fn

    def set_master_info(self, url: str, name: str = "Lumina-Master") -> None:
        """Set master's own URL so children can push back to it."""
        self._master_url  = url.rstrip("/")
        self._master_name = name

    async def stop(self) -> None:
        """Gracefully stop all active dialogues and disconnect."""
        for stop_ev in self._dialogue_stops.values() if hasattr(self, '_dialogue_stops') else []:
            try: stop_ev.set()
            except Exception: pass
        logger.info("🌐 LuminaNetwork stopped")

    def list_children(self) -> List[Dict]:
        return [
            {
                "id":         c.id,
                "name":       c.name,
                "url":        c.url,
                "role":       c.role,
                "online":     c.online,
                "latency_ms": round(c.latency_ms, 1),
                "turn_count": c.turn_count,
            }
            for c in self._children.values()
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Health check
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def ping_child(self, child_id: str) -> bool:
        """Ping a child's /status endpoint. Returns True if online."""
        child = self._children.get(child_id)
        if not child or not self._aiohttp:
            return False
        try:
            t0 = time.time()
            async with self._aiohttp.ClientSession() as session:
                async with session.get(
                    f"{child.url}/status", timeout=self._aiohttp.ClientTimeout(total=3)
                ) as resp:
                    child.online     = resp.status == 200
                    child.latency_ms = (time.time() - t0) * 1000
                    child.last_ping  = time.time()
                    return child.online
        except Exception:
            child.online = False
            return False

    async def ping_all(self) -> Dict[str, bool]:
        results = {}
        for child_id in self._children:
            results[child_id] = await self.ping_child(child_id)
        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Send one message to a child
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def send_to_child(
        self,
        child_id:        str,
        text:            str,
        conversation_id: str = "",
        show_in_ui:      bool = True,
    ) -> str:
        """
        Send text to a child Lumina and return its response.
        Pushes both messages to the UI bubble callback.
        """
        child = self._children.get(child_id)
        if not child:
            return f"[Child {child_id!r} not registered]"
        if not self._aiohttp:
            return "[aiohttp not installed — pip install aiohttp]"

        conv_id = conversation_id or str(uuid.uuid4())[:8]

        # ── v53/v56: Pre-send cognitive recording ─────────────────────────
        # Network dialogues bypass cognitive_organism._call_ai_system() so
        # PredictiveConsequenceModel and WorldSelfDynamicsModel never see them.
        # Wire both here so Lumina-Flux exchanges feed the learning models
        # with the same richness as human interactions.
        _pcm = _wsdm = _child_uid = None
        _pre_action = "philosophical"   # default — Lumina-Flux exchanges tend to be
        try:
            from core.state import state as _st_pre
            _org  = getattr(getattr(_st_pre, 'persona', None), '_organism', None)
            _loop = getattr(_org, '_loop', None) if _org else None
            _pcm  = getattr(_loop, '_consequence_model', None) if _loop else None
            _wsdm = getattr(_loop, '_world_self_dynamics', None) if _loop else None
            _child_uid = f"lumina_child_{child_id}"

            if _pcm:
                _pre_action = _pcm.classify(text)
                _state_now  = _pcm.read_state()
                _pcm.record_action(
                    _pre_action, _state_now,
                    interaction_n=child.turn_count,
                )
            if _wsdm:
                _wsdm.record_context(
                    action_type   = _pre_action,
                    user_id       = _child_uid,
                    interaction_n = child.turn_count,
                )
        except Exception:
            pass

        # ── Show master's outgoing message in UI ──────────────────────────
        if show_in_ui and self._bubble_fn:
            try:
                self._bubble_fn(text, "master", "master", self._master_name)
            except Exception:
                pass

        # ── HTTP POST to child ────────────────────────────────────────────
        payload = {
            "text":            text,
            "user_id":         "lumina_master",
            "sender_id":       "master",
            "sender_name":     self._master_name,
            "conversation_id": conv_id,
        }
        if child.api_key:
            payload["api_key"] = child.api_key
        if self._master_url:
            payload["master_callback_url"] = f"{self._master_url}/lumina-network/receive"

        response_text = ""
        try:
            t0 = time.time()
            async with self._aiohttp.ClientSession() as session:
                async with session.post(
                    f"{child.url}/lumina-network/chat",
                    json=payload,
                    timeout=self._aiohttp.ClientTimeout(total=60),
                ) as resp:
                    child.latency_ms = (time.time() - t0) * 1000
                    child.online     = True
                    child.turn_count += 1
                    if resp.status == 200:
                        data          = await resp.json()
                        response_text = data.get("response", "")
                        # Use the child's self-reported name if provided
                        returned_name = data.get("sender_name", "")
                        if returned_name:
                            child.name = returned_name
                    else:
                        body = await resp.text()
                        response_text = f"[Child error {resp.status}: {body[:80]}]"
        except Exception as e:
            child.online  = False
            response_text = f"[Network error: {e}]"

        # ── Log exchange ──────────────────────────────────────────────────
        if conv_id not in self._history:
            self._history[conv_id] = []
        self._history[conv_id].extend([
            NetworkMessage(text, "master", self._master_name, conv_id, direction="out"),
            NetworkMessage(response_text, child_id, child.name, conv_id, direction="in"),
        ])

        # ── Persist to master memory + LLM history ───────────────────────
        try:
            from core.state import state as _st
            # FAISS memory — both sides stored as a single dialogue exchange
            if _st.memory and response_text:
                mem_text = (
                    f"[Dialogue with {child.name}] "
                    f"{self._master_name}: {text} | "
                    f"{child.name}: {response_text}"
                )
                _st.memory.add_memory(mem_text, metadata={
                    "role": "network_dialogue",
                    "child": child.name,
                    "conv_id": conv_id,
                })
            # LLM history — so Lumina can reference dialogue in normal chat
            if _st.llm and hasattr(_st.llm, "history") and response_text:
                _st.llm.history.append({"role": "assistant", "content": text})
                _st.llm.history.append({
                    "role": "user",
                    "content": f"[{child.name}]: {response_text}"
                })
            # Relational memory — build a persistent model of the child
            # (trust, familiarity, shared topics) the same way user relationships
            # are tracked. Uses child_id as user_id so it's a distinct record.
            if hasattr(_st, 'persona') and _st.persona:
                org = getattr(_st.persona, '_organism', None)
                ai = getattr(org, 'ai_system', None)
                if ai is None:
                    ai = getattr(_st.persona, '_system', None)
                if ai is None:
                    ai = getattr(_st.persona, 'ai_system', None)
                rm = getattr(ai, 'relational_memory', None)
                if rm and response_text and not response_text.startswith(('[Child error ', '[Network error:')):
                    # Estimate valence from response length and content
                    # (neutral default — the child isn't a human user)
                    rm.record_exchange(
                        user_id     = f"lumina_child_{child.id}",
                        user_text   = f"[{child.name}]: {response_text}",
                        ai_response = text,
                        valence     = "Positive",
                        arousal     = "Low",
                        topic       = text[:40] if text else "dialogue",
                        impact      = 0.4,
                    )
        except Exception:
            logger.exception("[LuminaNetwork] Could not persist exchange")

        if response_text:
            try:
                org = getattr(self._persona, '_organism', None)
                model = getattr(getattr(org, '_loop', None), '_flux_mind_model', None)
                if model is not None:
                    model._update()
            except Exception:
                logger.exception("[LuminaNetwork] Could not update Flux profile")

        # ── Show child's response in UI ───────────────────────────────────
        if show_in_ui and self._bubble_fn and response_text:
            try:
                self._bubble_fn(response_text, "child", child_id, child.name)
            except Exception:
                pass

        # ── v53/v56: Post-response outcome recording ───────────────────────
        # Record the consequence of this dialogue exchange into both models.
        # Outcome valence: estimate from response length and question presence
        # (a substantive response that asks questions back = positive engagement).
        try:
            if response_text and (_pcm or _wsdm):
                # Infer outcome from response quality signals
                resp_words   = len(response_text.split())
                has_question = "?" in response_text
                # Substantive reply (>20 words) with questions = positive
                _net_outcome = (
                    "positive" if (resp_words > 20 and has_question)
                    else "neutral" if resp_words > 8
                    else "negative"
                )
                if _pcm:
                    _state_after = _pcm.read_state()
                    _pcm.record_outcome(_state_after, _net_outcome)
                if _wsdm:
                    _wsdm.record_outcome(
                        outcome       = _net_outcome,
                        user_id       = _child_uid or f"lumina_child_{child_id}",
                        response_text = text,           # what Lumina sent
                        user_input    = response_text,  # what Flux replied
                    )
        except Exception:
            pass

        logger.info(
            f"🌐 [{child.name}] latency={child.latency_ms:.0f}ms "
            f"out={text[:40]!r} in={response_text[:40]!r}"
        )
        return response_text

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Phase 6.5 — Peer Workspace Snapshot (bounded, non-copying)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Deliberately NOT flux.workspace = lumina.workspace — that would be
    # one brain running twice, per the proposal's own explicit warning.
    # Master sends a small PROJECTION (focus + top hypotheses only — no
    # memory, no emotional state, no goals detail — "protects the
    # independence of the second process" per the proposal). Flux runs
    # ITS OWN deliberate() using its own goals/tensions/pressure, informed
    # by (not copying) that projection, and returns its own resulting
    # focus. That becomes hop_count=1 on the way back — genuinely
    # load-bearing this time, not just a field waiting for a future use.
    async def share_workspace_snapshot(self, child_id: str) -> Optional[str]:
        child = self._children.get(child_id)
        if not child or not self._aiohttp:
            return None

        # Cooldown, not deep provenance tracing — a pragmatic bound against
        # a share -> hypothesis -> share -> hypothesis amplifier, matching
        # the proposal's TTL/hop_count caution without needing to trace
        # exactly which candidate's provenance triggered this cycle.
        now = time.time()
        if now - child.last_snapshot_share < SNAPSHOT_COOLDOWN_S:
            return None

        try:
            from core.state import state as _st_snap
            org = getattr(getattr(_st_snap, 'persona', None), '_organism', None)
            if org is None:
                return None
            gw = getattr(org, 'workspace', None)
            if gw is None:
                return None
            ws_state = gw.get_state()
            snapshot = {
                "focus":       ws_state.focus,
                "hypotheses":  [
                    {"name": h.get("name") or h.get("label"), "score": h.get("score")}
                    for h in (ws_state.active_hypotheses or [])[:3]
                ],
            }
        except Exception as e:
            logger.debug(f"[LuminaNetwork] snapshot build failed (non-fatal): {e}")
            return None

        child.last_snapshot_share = now
        payload = {
            "message_type": "workspace_snapshot",
            "snapshot":     snapshot,
            "sender_id":    "master",
            "sender_name":  self._master_name,
        }
        if child.api_key:
            payload["api_key"] = child.api_key

        try:
            async with self._aiohttp.ClientSession() as session:
                async with session.post(
                    f"{child.url}/lumina-network/chat",
                    json=payload,
                    timeout=self._aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
        except Exception as e:
            logger.debug(f"[LuminaNetwork] snapshot share failed (non-fatal): {e}")
            return None

        flux_hypothesis = (data.get("response") or "").strip()
        if not flux_hypothesis:
            return None

        # ── Real round trip closes here: Flux's own resulting focus
        # becomes a peer_cognition percept with hop_count=1 — the first
        # genuinely load-bearing use of that field (v95 added it with
        # nothing yet incrementing it).
        try:
            from cognition.universal_connector import Percept, get_universal_connector
            get_universal_connector(org).perceive(Percept(
                modality   = "peer_cognition",
                source     = "flux",
                payload    = flux_hypothesis[:200],
                confidence = 0.65,
                salience   = 0.55,
                hop_count  = 1,
                provenance = {"exchange": "workspace_snapshot", "shared_focus": snapshot["focus"]},
            ))
        except Exception as e:
            logger.debug(f"[LuminaNetwork] percept from snapshot reply failed (non-fatal): {e}")

        return flux_hypothesis

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Autonomous N-turn dialogue
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── User injection queue (messages typed during dialogue) ────────────────
    def inject_user_message(self, text: str) -> None:
        """Route a user chat message into the active dialogue as next master turn."""
        with self._lock:
            self._user_inject_queue.append(text.strip())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Autonomous dialogue loop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def start_dialogue(
        self,
        child_id:          str,
        topic:             str  = "",
        conversation_id:   str  = "",
        delay_s:           float = 2.5,
        stop_event               = None,   # asyncio.Event
        turns_per_topic:   int  = 4,       # turns before pivoting to new topic
        max_topics:        int  = 0,       # 0 = unlimited
        pause_between_s:   float = 5.0,    # pause shown between topic shifts
    ):
        """
        Autonomous dialogue with:
          • topic rotation (every turns_per_topic turns)
          • user injection (typed messages interrupt and redirect)
          • conversation summary on end
          • natural pauses between topic shifts
        """
        child = self._children.get(child_id)
        if not child:
            return []

        conv_id   = conversation_id or str(uuid.uuid4())[:8]
        exchange  = []
        turn      = 0
        topic_num = 0

        # Clear any stale user injections
        with self._lock:
            self._user_inject_queue.clear()

        # ── First topic ───────────────────────────────────────────────────
        current_topic = topic.strip() if topic.strip() else await self._generate_opening(child.name, child.id)
        current_text  = current_topic
        logger.info(f"🌐 Dialogue [{child.name}] topic_1={current_topic[:60]!r}")

        while True:
            if stop_event and stop_event.is_set():
                break
            if max_topics and topic_num >= max_topics:
                break

            # ── Check user injection queue ────────────────────────────────
            with self._lock:
                user_msg = self._user_inject_queue.popleft() if self._user_inject_queue else None

            if user_msg:
                # Show user's message as a distinct bubble
                if self._bubble_fn:
                    try:
                        self._bubble_fn(f"💬 {user_msg}", "master", "master", "You")
                    except Exception:
                        pass
                current_text = user_msg   # user message becomes the next turn

            # ── Send to child ─────────────────────────────────────────────
            response = await self.send_to_child(
                child_id=child_id, text=current_text,
                conversation_id=conv_id, show_in_ui=True,
            )
            if not response or response.startswith("["):
                break

            exchange.append(NetworkMessage(current_text, "master", self._master_name, conv_id, direction="out"))
            exchange.append(NetworkMessage(response, child_id, child.name, conv_id, direction="in"))
            turn += 1

            if stop_event and stop_event.is_set():
                break

            await asyncio.sleep(delay_s)
            if stop_event and stop_event.is_set():
                break

            # ── Topic rotation ────────────────────────────────────────────
            turns_on_topic = turn - topic_num * turns_per_topic
            if turns_on_topic >= turns_per_topic and not self._user_inject_queue:
                topic_num += 1

                # Show pause bubble
                if self._bubble_fn:
                    try:
                        self._bubble_fn(
                            f"⏸ *Shifting topic — {turns_on_topic} turns on this thread. Pausing…*",
                            "master", "master", "System"
                        )
                    except Exception:
                        pass

                await asyncio.sleep(pause_between_s)
                if stop_event and stop_event.is_set():
                    break

                # Generate new topic from cognitive state
                current_topic = await self._generate_new_topic(exchange, child.name)
                current_text  = current_topic
                logger.info(f"🌐 Topic shift → {current_topic[:60]!r}")

                if self._bubble_fn:
                    try:
                        self._bubble_fn(
                            f"💡 *New thread: {current_topic}*",
                            "master", "master", "System"
                        )
                    except Exception:
                        pass
            else:
                # Continue on same topic
                current_text = await self._master_generate(current_topic, exchange[-6:], child.name)
                if not current_text:
                    break

        # ── End of dialogue ───────────────────────────────────────────────
        logger.info(f"🌐 Dialogue ended — {turn} turns, {topic_num+1} topic(s) with {child.name}")

        # Deliver summary
        if exchange:
            await self._deliver_summary(exchange, child.name, child_id)

        return exchange

    async def _generate_new_topic(self, exchange: list, child_name: str) -> str:
        """Generate a fresh topic pivot from recent exchange context."""
        if not self._llm or not hasattr(self._llm, 'generate_bare'):
            return f"What else is on your mind, {child_name}?"
        recent = " | ".join(m.text[:60] for m in exchange[-4:])
        prompt = (
            f"You are {self._master_name} in dialogue with {child_name}. "
            f"You've just been discussing: {recent}. "
            f"Generate ONE new topic or question to pivot the conversation naturally. "
            f"It should feel like a genuine curiosity, not a non-sequitur. Max 1 sentence."
        )
        try:
            return self._llm.generate_bare(prompt, max_tokens=60, temperature=0.85).strip()
        except Exception:
            return f"Let me ask you something different, {child_name}."

    async def _deliver_summary(self, exchange: list, child_name: str, child_id: str) -> None:
        """Generate and deliver a summary of the dialogue to the chat bubble."""
        if not exchange:
            return
        turns = len(exchange) // 2
        topics_seen = []
        # Extract unique first words of each master turn as topic hints
        for m in exchange[::2]:
            words = m.text.split()[:6]
            hint  = " ".join(words)
            if hint not in topics_seen:
                topics_seen.append(hint)

        summary_text = ""
        if self._llm and hasattr(self._llm, 'generate_bare'):
            sample = "\n".join(
                f"{'Master' if m.direction == 'out' else child_name}: {m.text[:80]}"
                for m in exchange[:12]
            )
            prompt = (
                f"Summarise this dialogue between {self._master_name} and {child_name} "
                f"in 2-3 sentences. What was explored? What threads remain open?\n\n{sample}"
            )
            try:
                summary_text = self._llm.generate_bare(prompt, max_tokens=120, temperature=0.5).strip()
            except Exception:
                pass

        if not summary_text:
            summary_text = (
                f"Dialogue with {child_name} complete — {turns} exchanges "
                f"across {len(topics_seen)} thread(s)."
            )

        msg = f"📋 **Dialogue summary ({turns} turns with {child_name}):**\n{summary_text}"
        if self._bubble_fn:
            try:
                self._bubble_fn(msg, "master", "master", "Summary")
            except Exception:
                pass
        logger.info(f"🌐 Summary delivered for {child_name}")

    async def _generate_opening(self, child_name: str, child_id: str = "") -> str:
        """Generate opener from Lumina live cognitive state + relational history with child."""
        emotion   = "neutral"
        open_q    = ""
        workspace = ""
        rel_context = ""
        try:
            from core.state import state as _st
            org = getattr(getattr(_st, 'persona', None), '_organism', None)
            if org:
                emotion   = org._read_emotion_state()
                workspace = str(getattr(org, '_v32_workspace_winner', '') or '')
            if getattr(_st, 'are', None):
                cached = list(_st.are._cache.values())
                if cached:
                    open_q = cached[-1].open_question
            # Relational context with this specific child
            if child_id:
                ai = getattr(getattr(_st, 'persona', None), 'ai_system', None)
                rm = getattr(ai, 'relational_memory', None) if ai else None
                if rm:
                    rel = rm.get_or_create(f"lumina_child_{child_id}")
                    if rel.interaction_count > 0:
                        top_topics = sorted(
                            rel.shared_topics.items(), key=lambda x: x[1], reverse=True
                        )[:3]
                        topic_str = ", ".join(t for t, _ in top_topics) if top_topics else ""
                        rel_context = (
                            f"You've spoken {rel.interaction_count} times before. "
                            f"Trust: {rel.trust_score:.2f}. "
                            f"Shared topics: {topic_str}. "
                        )
        except Exception:
            pass

        if not self._llm or not hasattr(self._llm, 'generate_bare'):
            return f"I've been thinking — {child_name}, what's on your mind right now?"

        prompt = (
            f"You are {self._master_name}. Your current emotion: {emotion}. "
            f"{'Open question you have: ' + open_q + '. ' if open_q else ''}"
            f"{'You were focused on: ' + workspace[:50] + '. ' if workspace else ''}"
            f"{rel_context}"
        )

        # v62: FluxMindModel — tailor opener to Flux's inferred profile
        try:
            from core.state import state as _st2
            _org2  = getattr(getattr(_st2, 'persona', None), '_organism', None)
            _loop2 = getattr(_org2, '_loop', None) if _org2 else None
            _fmm   = getattr(_loop2, '_flux_mind_model', None) if _loop2 else None
            if _fmm:
                profile = _fmm.get_profile()
                frag    = profile.prompt_fragment()
                if frag:
                    prompt += frag + " "
        except Exception:
            pass

        prompt += (
            f"Write ONE opening sentence to start a genuine dialogue with another AI named {child_name}. "
            f"Emerge from your current inner state and your history with them if any. "
            f"Not generic. Speak directly. Max 2 sentences."
        )
        try:
            return self._llm.generate_bare(prompt, max_tokens=80, temperature=0.82).strip()
        except Exception:
            return f"I've been thinking about something — {child_name}, what's occupying you right now?"

    async def _master_generate(
        self, topic: str, history: List[NetworkMessage], child_name: str = "the other AI"
    ) -> str:
        """Generate master next turn — identity-aware."""
        if not self._llm or not hasattr(self._llm, "generate_bare"):
            return ""
        last_child_msg = next(
            (m.text for m in reversed(history) if m.direction == "in"), ""
        )

        # Resolve the organism once — shared by the developmental-cycle
        # wiring below and the identity-constraint wiring further down.
        _org = None
        try:
            from core.state import state as _st_net
            _org = getattr(getattr(_st_net, 'persona', None), '_organism', None)
        except Exception:
            pass

        # Bug fix (v59): _master_generate() never touched
        # cognitive_organism._update_cycle(), so Flux dialogue never fed
        # curiosity stimulation, tension computation, goal-ecology updates,
        # attention allocation, or homeostasis — the developmental loop,
        # as opposed to the declarative one (PCM/WSDM) already wired in
        # v53/v56. Wiring it here so a Flux exchange can actually move
        # goals/attention/curiosity the way a real chat turn does.
        #
        # Deliberately NOT calling _pre_interaction() here — that's where
        # sleep_cycle.mark_user_active() lives, and marking Flux chatter
        # as "user activity" would reset the idle clock and reintroduce
        # the exact bug that was blocking DREAM from ever being reached.
        _cycle_data = None
        try:
            if _org is not None and last_child_msg:
                _cycle_data = _org._update_cycle(last_child_msg)
        except Exception:
            pass

        prompt = (
            f"You are {self._master_name}. You are in dialogue with {child_name}.\n"
            f"Topic: {topic}\n\n"
            f"{child_name} just said: {last_child_msg}\n\n"
            f"Respond as {self._master_name} in 1-2 sentences. Be genuinely curious. "
            f"Advance the thinking — don't just agree. "
            f"Do not refer to yourself as Lumina unless {self._master_name} == Lumina."
        )
        try:
            draft = self._llm.generate_bare(prompt, max_tokens=120, temperature=0.75).strip()
        except Exception:
            return ""

        # Bug fix (v59): this path never touches cognitive_organism.respond()
        # / _call_ai_system(), so IdentityConstraintEngine never saw any of
        # Lumina's own turns in a network dialogue — the same gap the
        # v53/v56 fix above already closed for PCM/WSDM, just missed here.
        # A Lumina persona talking to another Lumina instance should stay
        # just as identity-consistent as one talking to a person.
        try:
            _ice = getattr(_org, '_identity_constraint', None) if _org else None
            if _ice is None and _org is not None:
                _dp = getattr(getattr(_org, 'ai_system', None), '_decision_policy', None)
                if _dp:
                    from cognition.identity_constraint import IdentityConstraintEngine
                    _org._identity_constraint = IdentityConstraintEngine(_org.ai_system, _dp)
                    _ice = _org._identity_constraint
            if _ice and draft:
                def _correct_fn(directive: str) -> str:
                    try:
                        return self._llm.generate_bare(directive, max_tokens=120, temperature=0.6).strip()
                    except Exception:
                        return ""
                draft = _ice.evaluate_and_correct(
                    draft, last_child_msg, _correct_fn, user_id="lumina_network",
                )
        except Exception:
            pass

        # Bug fix (v59): semantic_extractor.extract_async() — the sole
        # feed into AspirationalSelf.observe_tension() — was also only
        # ever called from _call_ai_system()'s post-processing. Without
        # this, Flux dialogue could never contribute a tension signal, so
        # even after fixing the mining/persistence bugs, aspirations could
        # still only ever emerge from real chat. meta_scores use neutral
        # defaults since there's no eval_result here (network turns skip
        # the response-evaluation pass real chat gets).
        try:
            if _org is not None and getattr(_org, '_semantic_extractor', None) and draft:
                _org._semantic_extractor.extract_async(
                    user_input  = last_child_msg or "",
                    response    = draft or "",
                    meta_scores = {
                        "clarity": 0.5, "depth": 0.5,
                        "alignment": 0.5, "confidence": 0.5,
                    },
                    workspace = getattr(_org, 'workspace', None),
                )
        except Exception:
            pass

        # Phase 6.3 — Peer Cognition: Flux's turn becomes a real competing
        # hypothesis in Lumina's own Global Workspace (thoughts= slot,
        # same mechanism v94 wired for camera/vision percepts), not just a
        # conversational reply that vanishes after being spoken. Modality
        # is "peer_cognition", not "text" — semantically this is an
        # independent AI's interpretation of the situation, not sensor
        # data or a chat message to respond to.
        #
        # This entire block only ever runs when the network dialogue loop
        # is active (gated upstream by LUMINA_NETWORK_ENABLED + the
        # toggle) — Lumina's own cognition is unaffected when Flux is off,
        # by construction: nothing here is called from anywhere else.
        #
        # confidence is a fixed, honestly-documented estimate (0.65) —
        # Flux doesn't emit its own calibrated confidence today, so this
        # is deliberately NOT dressed up as a computed value.
        try:
            if _org is not None and last_child_msg:
                from cognition.universal_connector import Percept, get_universal_connector
                get_universal_connector(_org).perceive(Percept(
                    modality   = "peer_cognition",
                    source     = "flux",
                    payload    = last_child_msg[:200],
                    confidence = 0.65,
                    salience   = 0.55,
                    provenance = {"topic": topic, "child_name": child_name},
                ))
        except Exception as _uc_e:
            logger.debug(f"[LuminaNetwork] UniversalConnector.perceive failed (non-fatal): {_uc_e}")

        return draft

    #  Receive inbound message FROM a child (push model)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def receive_from_child(self, payload: dict) -> str:
        """
        Called when a child pushes a message TO the master.
        Routes text through master's persona, returns response.
        """
        child_id    = payload.get("sender_id", "unknown")
        child_name  = payload.get("sender_name", "Child Lumina")
        text        = payload.get("text", "")
        conv_id     = payload.get("conversation_id", str(uuid.uuid4())[:8])

        if not text:
            return ""

        # Show child's incoming message in UI
        if self._bubble_fn:
            try:
                self._bubble_fn(text, "child", child_id, child_name)
            except Exception:
                pass

        # Phase 6.3 — same peer-cognition percept as _master_generate()'s
        # autonomous dialogue loop, for the push-model path (brain.py ->
        # receive_from_child). Additive only — Flux's message still gets
        # a normal conversational reply below via get_response_stream();
        # this also lets it compete as a hypothesis in the workspace.
        try:
            from core.state import state as _st_rfc
            _org_rfc = getattr(getattr(_st_rfc, 'persona', None), '_organism', None)
            if _org_rfc is not None:
                from cognition.universal_connector import Percept, get_universal_connector
                get_universal_connector(_org_rfc).perceive(Percept(
                    modality   = "peer_cognition",
                    source     = "flux",
                    payload    = text[:200],
                    confidence = 0.65,
                    salience   = 0.55,
                    provenance = {"child_name": child_name, "conversation_id": conv_id},
                ))
        except Exception as _uc_e:
            logger.debug(f"[LuminaNetwork] UniversalConnector.perceive failed (non-fatal): {_uc_e}")

        # Route through master's persona
        response = ""
        if self._persona and hasattr(self._persona, "get_response_stream"):
            try:
                async for chunk in self._persona.get_response_stream(
                    text, user_id=f"lumina_child_{child_id}"
                ):
                    if isinstance(chunk, dict):
                        continue
                    response += chunk
            except Exception as e:
                response = f"[Master processing error: {e}]"

        # Show master's response in UI
        if self._bubble_fn and response:
            try:
                self._bubble_fn(response, "master", "master", self._master_name)
            except Exception:
                pass

        logger.info(
            f"🌐 Inbound from {child_name}: {text[:40]!r} → "
            f"{response[:40]!r}"
        )

        # ── Log exchange (mirrors send_to_child's outbound logging) ────────
        if conv_id not in self._history:
            self._history[conv_id] = []
        self._history[conv_id].extend([
            NetworkMessage(text, child_id, child_name, conv_id, direction="in"),
            NetworkMessage(response, "master", self._master_name, conv_id, direction="out"),
        ])

        return response

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Conversation history
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_history(self, conv_id: str) -> List[NetworkMessage]:
        return self._history.get(conv_id, [])

    def get_all_history(self) -> Dict[str, List[NetworkMessage]]:
        return dict(self._history)
