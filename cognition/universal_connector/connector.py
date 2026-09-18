"""
UniversalConnector — Phase 6.0/6.2 (minimal, evidence-scoped slice)
=====================================================================
One shared instance per organism (see get_universal_connector() below —
PresenceEngine and PerceptionLoop both feed the same buffer, otherwise
recursive_deliberation would only ever see whichever consumer happened
to construct its own separate connector).

perceive() does two real things for a sufficiently salient percept:
  1. Makes it durable — real semantic memory (v93) + a narrative chapter
     (v93, routes through the same v91 self_concept pipeline).
  2. Makes it a genuine competing candidate for attention THIS turn —
     recursive_deliberation.deliberate() already has an unused `thoughts`
     slot in WorkspaceCompetition.compete() (always called with
     thoughts=[]). A percept becomes a real "thought" candidate there,
     scored by the same fair competition goals already go through — not
     injected sideways via GlobalWorkspace.set_hypotheses(), which
     REPLACES active_hypotheses wholesale and would race with whatever
     deliberate() sets on the very next turn (verified: set_hypotheses()
     replaces the full list, so a background-thread camera tick calling
     it directly would just get overwritten a moment later).

Deliberately NOT built (would be speculation, no code to audit against):
  - act() for outbound device/actuator control
  - a modality registry/adapters/ package — one real modality (vision)
    plus mic, which is already fully connected via the ordinary chat path
    (STT transcription -> state.transcription_queue -> handle_send() ->
    the same respond() pipeline as typed text — verified, no gap there,
    so mic doesn't need routing through this connector at all)
  - Flux as a peer-cognition hypothesis source — already a real, running
    Lumina instance (brain-only, client of master, conversation mode) per
    the user, not a peer-cognition system to build; a real next step, but
    a distinct piece of work from this vision/mic slice
"""
from __future__ import annotations

import logging
import time
import json
import ipaddress
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Any, Dict, List, Optional

from cognition.universal_connector.event import Percept
from cognition.body_runtime import get_body_runtime

logger = logging.getLogger(__name__)

MIN_SALIENCE_FOR_MEMORY = 0.35   # below this, not worth a permanent memory/chapter
MIN_SALIENCE_FOR_WORKSPACE = 0.45  # below this, not worth competing for attention
RECENT_TTL_S = 180.0              # same order as GlobalWorkspace.HYPOTHESIS_TTL
MAX_RECENT = 5
MAX_HOP_COUNT = 2                 # anti-echo cap, per the peer-cognition proposal
DUPLICATE_WINDOW_S = 60.0         # same source + near-identical text within this
                                   # window is treated as one percept, not spammed
                                   # into the workspace competition every occurrence
                                   # (concretely matters for Flux's autonomous dialogue
                                   # loop, which can stay on one topic for many turns)

# Phase 6.4 — small, honest keyword heuristic (not real semantic disagreement
# detection). Same pattern already used elsewhere in this codebase for
# similar low-cost signals (VALUE_KEYWORDS, ETHICAL_ONLY_KEYWORDS).
_DISAGREEMENT_MARKERS = (
    "i don't think", "i disagree", "not sure that's right", "however,",
    "but actually", "i'd push back", "that doesn't add up", "i'm not convinced",
    "on the contrary", "actually, i think", "counterpoint",
)


class UniversalConnector:
    def __init__(self, organism: Any):
        self._organism = organism
        self._body_runtime = get_body_runtime(organism)
        self._recent: List[Percept] = []
        self._ha_task = None
        self._ha_stop = None
        self._ha_states: Dict[str, Any] = {}
        self._ha_entities: Dict[str, Dict[str, Any]] = {}
        logger.info("[UniversalConnector] initialised (Phase 6.0/6.2 slice)")

    @staticmethod
    def _body_plugin_enabled(config: Any, legacy_name: str) -> bool:
        """Read the new body-plugin switch while preserving old configs."""
        if legacy_name == "home_assistant":
            selected = getattr(config, "BODY_PLUGIN_HOME_ASSISTANT_ENABLED", None)
            return bool(getattr(config, "HOME_ASSISTANT_ENABLED", False) if selected is None else selected)
        return False

    def _body_setting(self, config: Any, key: str, fallback: Any = None) -> Any:
        value = self._body_runtime.config_value(key, getattr(config, key, fallback))
        return fallback if value is None else value

    def perceive(self, percept: Percept) -> None:
        if percept.hop_count > MAX_HOP_COUNT:
            logger.debug(
                f"[UniversalConnector] dropped percept from {percept.source} — "
                f"hop_count {percept.hop_count} exceeds cap {MAX_HOP_COUNT}"
            )
            return
        if self._is_recent_duplicate(percept):
            return

        # Phase 6.4 — "disagreement itself becomes a cognitive signal": for
        # a peer_cognition percept specifically, a small, honest keyword
        # heuristic (same style as VALUE_KEYWORDS/ETHICAL_ONLY_KEYWORDS
        # elsewhere in this codebase — not real semantic disagreement
        # detection, which would need an embedding comparison this
        # environment can't cheaply do) checks for explicit disagreement
        # markers and, if found, boosts REAL epistemic pressure via
        # PressureSystem.boost() — whose own docstring already says
        # "after detecting a contradiction". This is the same pressure
        # dimension already verified (closed-loop audit) to flow into
        # curiosity_drive -> derive_motivations() -> real goal formation,
        # so a disagreement can genuinely trigger investigation, not just
        # get logged.
        if percept.modality == "peer_cognition":
            self._maybe_boost_epistemic_pressure(percept)

        if percept.salience >= MIN_SALIENCE_FOR_WORKSPACE:
            self._recent.append(percept)
            self._recent = self._recent[-MAX_RECENT:]

        if percept.salience < MIN_SALIENCE_FOR_MEMORY:
            return

        self._store_memory(percept)
        self._record_chapter(percept)

    def discover_home_assistant(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Read permitted Home Assistant states without executing actions."""
        try:
            from managers.settings_manager import config
            enabled = bool(self._body_setting(config, "BODY_PLUGIN_HOME_ASSISTANT_ENABLED", getattr(config, "HOME_ASSISTANT_ENABLED", False)))
            connector_enabled = bool(getattr(config, "UNIVERSAL_CONNECTOR_ENABLED", False))
            body_enabled = bool(self._body_setting(config, "BODY_RUNTIME_ENABLED", True))
            base_url = str(self._body_setting(config, "HOME_ASSISTANT_URL", "") or "").strip().rstrip("/")
            token = str(self._body_setting(config, "HOME_ASSISTANT_TOKEN", "") or "").strip()
            verify_ssl = bool(self._body_setting(config, "HOME_ASSISTANT_VERIFY_SSL", True))
            allowed = {
                item.strip().lower()
                for item in str(self._body_setting(config, "HOME_ASSISTANT_ALLOWED_DOMAINS", "") or "").split(",")
                if item.strip()
            }
            if not connector_enabled or not body_enabled or not enabled:
                return {"ok": False, "status": "disabled", "entities": [], "message": "Universal Connector, Body Runtime, or Home Assistant plugin is disabled."}
            if not base_url or not token:
                return {"ok": False, "status": "not_configured", "entities": [], "message": "Home Assistant URL and access token are required."}
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return {"ok": False, "status": "invalid_url", "entities": [], "message": "Home Assistant URL must be an HTTP(S) address."}

            request = Request(
                f"{base_url}/api/states",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
            context = None
            if parsed.scheme == "https" and not verify_ssl:
                context = ssl._create_unverified_context()
            with urlopen(request, timeout=max(0.25, float(timeout)), context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                return {"ok": False, "status": "invalid_response", "entities": [], "message": "Home Assistant returned an invalid states response."}
            entities = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                entity_id = str(item.get("entity_id", ""))
                domain = entity_id.split(".", 1)[0].lower()
                if entity_id and (not allowed or domain in allowed):
                    attributes = item.get("attributes") or {}
                    device_class = attributes.get("device_class")
                    signal = None
                    if domain == "binary_sensor" and device_class in {"presence", "occupancy", "motion"}:
                        signal = "presence_detected" if str(item.get("state", "")).lower() == "on" else "no_presence"
                    entities.append({
                        "entity_id": entity_id,
                        "state": item.get("state"),
                        "friendly_name": attributes.get("friendly_name", entity_id),
                        "unit": attributes.get("unit_of_measurement"),
                        "domain": domain,
                        "device_class": device_class,
                        "signal": signal,
                        "last_updated": item.get("last_updated"),
                        "last_changed": item.get("last_changed"),
                    })
            entities.sort(key=lambda item: item["entity_id"])
            next_entities = {item["entity_id"]: item for item in entities}
            previous_entities = self._ha_entities
            changed_ids = {
                entity_id for entity_id, item in next_entities.items()
                if previous_entities.get(entity_id) != item
            }
            changed_ids.update(set(previous_entities) - set(next_entities))
            self._ha_entities = next_entities
            for item in entities:
                self._body_runtime.publish_observation(
                    source="home_assistant",
                    kind=str(item.get("domain") or "sensor"),
                    subject=self._entity_label(item),
                    value=("presence detected" if item.get("signal") == "presence_detected" else
                           "no presence" if item.get("signal") == "no_presence" else item.get("state")),
                    unit=str(item.get("unit") or ""),
                    confidence=0.98,
                    observed_at=self._parse_timestamp(item.get("last_updated") or item.get("last_changed")),
                    provenance={"entity_id": item.get("entity_id"), "device_class": item.get("device_class")},
                )
            if not previous_entities or len(previous_entities) != len(next_entities):
                logger.info(
                    "[UniversalConnector] Home Assistant discovery: %d permitted entities",
                    len(entities),
                )
            elif changed_ids:
                logger.info(
                    "[UniversalConnector] Home Assistant refresh: %d/%d entities changed",
                    len(changed_ids), len(entities),
                )
            else:
                # A five-second poll is intentional for presence latency, but
                # an unchanged snapshot should not flood the application log.
                logger.debug(
                    "[UniversalConnector] Home Assistant refresh unchanged: %d entities",
                    len(entities),
                )
            return {"ok": True, "status": "connected", "entities": entities, "count": len(entities)}
        except HTTPError as exc:
            status = "unauthorized" if exc.code in {401, 403} else "http_error"
            return {"ok": False, "status": status, "entities": [], "message": f"Home Assistant returned HTTP {exc.code}."}
        except (URLError, TimeoutError) as exc:
            return {"ok": False, "status": "unreachable", "entities": [], "message": f"Home Assistant is unreachable: {exc.reason if isinstance(exc, URLError) else exc}."}
        except Exception as exc:
            logger.warning("[UniversalConnector] Home Assistant discovery failed: %s", exc)
            return {"ok": False, "status": "error", "entities": [], "message": str(exc)}

    def refresh_home_assistant_for_prompt(self) -> bool:
        """Refresh selected sensor values immediately before prompt creation.

        The background poll remains the normal path. This short synchronous
        read closes the race where a sensor changes while the monitor is
        paused for an interactive turn, so current-value questions use the
        latest Home Assistant timestamp instead of the previous poll.
        """
        result = self.discover_home_assistant(timeout=1.5)
        if result.get("ok"):
            self._ingest_home_assistant_changes(result.get("entities", []))
            return True
        logger.debug("[UniversalConnector] prompt refresh unavailable: %s", result.get("message"))
        return False

    async def reconcile_home_assistant_monitor(self) -> None:
        """Start or stop the low-rate HA monitor after config changes."""
        try:
            from managers.settings_manager import config
            enabled = (
                bool(getattr(config, "UNIVERSAL_CONNECTOR_ENABLED", False))
                and bool(getattr(config, "BODY_RUNTIME_ENABLED", True))
                and bool(self._body_setting(config, "BODY_PLUGIN_HOME_ASSISTANT_ENABLED", getattr(config, "HOME_ASSISTANT_ENABLED", False)))
            )
        except Exception:
            enabled = False
        if enabled and self._ha_task is None:
            import asyncio
            self._ha_stop = asyncio.Event()
            self._ha_task = asyncio.create_task(self._home_assistant_loop())
            logger.info("[UniversalConnector] Home Assistant monitor started")
        elif not enabled and self._ha_task is not None:
            await self.stop_home_assistant_monitor()

    async def stop_home_assistant_monitor(self) -> None:
        """Stop the HA monitor without leaving a pending asyncio task."""
        task = self._ha_task
        if task is None:
            return
        if self._ha_stop is not None:
            self._ha_stop.set()
        try:
            await task
        except Exception:
            logger.debug("[UniversalConnector] Home Assistant monitor stopped with an error", exc_info=True)
        self._ha_task = None
        self._ha_stop = None

    async def _home_assistant_loop(self) -> None:
        import asyncio
        while self._ha_stop is not None and not self._ha_stop.is_set():
            try:
                # The interactive turn owns the local LLM and attention path;
                # do not add network work or percepts while it is active.
                from core.interface_api import interactive_turn_active
                if not interactive_turn_active():
                    result = await asyncio.to_thread(self.discover_home_assistant)
                    if result.get("ok"):
                        self._ingest_home_assistant_changes(result.get("entities", []))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("[UniversalConnector] Home Assistant monitor cycle failed: %s", exc)
            try:
                from managers.settings_manager import config
                interval = max(1, min(300, int(self._body_setting(config, "HOME_ASSISTANT_POLL_INTERVAL", 5))))
            except Exception:
                interval = 5
            try:
                await asyncio.wait_for(self._ha_stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def _ingest_home_assistant_changes(self, entities: List[Dict[str, Any]]) -> None:
        selected = self._selected_home_assistant_entities()
        current = {
            str(item.get("entity_id")): item for item in entities
            if item.get("entity_id") and (not selected or str(item.get("entity_id")) in selected)
        }
        if not self._ha_states:
            self._ha_states = {key: self._state_signature(item) for key, item in current.items()}
            for entity_id, item in current.items():
                signal = item.get("signal")
                if signal in {"presence_detected", "no_presence"}:
                    self._forward_external_presence(entity_id, signal == "presence_detected")
            logger.info("[UniversalConnector] Home Assistant baseline captured: %d entities", len(current))
            return
        for entity_id, item in current.items():
            signature = self._state_signature(item)
            if self._ha_states.get(entity_id) == signature:
                continue
            self._ha_states[entity_id] = signature
            signal = item.get("signal") or "state_changed"
            if signal in {"presence_detected", "no_presence"}:
                self._forward_external_presence(entity_id, signal == "presence_detected")
            payload = f"Home Assistant sensor {self._entity_label(item)}: {signal} ({item.get('state')})"
            self.perceive(Percept(
                modality="home_assistant",
                source="home_assistant",
                payload=payload,
                confidence=0.95,
                salience=0.72 if signal in {"presence_detected", "no_presence"} else 0.45,
                novelty=0.65,
                provenance={"entity_id": entity_id, "device_class": item.get("device_class")},
            ))
            logger.info("[UniversalConnector] Home Assistant change: %s -> %s", entity_id, signal)

    def home_assistant_context(self) -> str:
        """Return selected live HA states for the user-facing LLM context."""
        if not self._ha_entities:
            return ""
        selected = self._selected_home_assistant_entities()
        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            f"Observed from Home Assistant at {observed_at} (authoritative live snapshot).",
            "These selected external sensor data are relevant evidence about the current environment.",
            "Use a sensor only when relevant to the user's question; preserve its exact value and unit.",
            "Prefer the sensor timestamp over older conversation or memory data. If a requested value is absent, say it is unavailable.",
        ]
        for entity_id, item in self._ha_entities.items():
            if selected and entity_id not in selected:
                continue
            value = item.get("state")
            unit = item.get("unit") or ""
            label = self._entity_label(item)
            signal = item.get("signal")
            if signal == "presence_detected":
                value = "presence detected"
            elif signal == "no_presence":
                value = "no presence"
            updated = item.get("last_updated") or item.get("last_changed") or "timestamp unavailable"
            lines.append(f"- {label}: {value}{(' ' + unit) if unit else ''} (sensor updated {updated})")
        return "\n".join(lines)

    def _entity_label(self, item: Dict[str, Any]) -> str:
        entity_id = str(item.get("entity_id") or "sensor")
        try:
            from managers.settings_manager import config
            tags = json.loads(str(self._body_setting(config, "HOME_ASSISTANT_ENTITY_TAGS", "{}") or "{}"))
            tag = tags.get(entity_id) if isinstance(tags, dict) else None
            if isinstance(tag, str) and tag.strip():
                return tag.strip()
        except Exception:
            pass
        return entity_id.rsplit(".", 1)[-1].replace("_", " ")

    def _selected_home_assistant_entities(self) -> set[str]:
        try:
            from managers.settings_manager import config
            return {
                item.strip() for item in str(self._body_setting(config, "HOME_ASSISTANT_SELECTED_ENTITIES", "") or "").split(",")
                if item.strip()
            }
        except Exception:
            return set()

    @staticmethod
    def _state_signature(item: Dict[str, Any]) -> str:
        return f"{item.get('state')}|{item.get('signal')}|{item.get('unit')}"

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        if not value:
            return time.time()
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except Exception:
            return time.time()

    def _forward_external_presence(self, entity_id: str, present: bool) -> None:
        """Forward occupancy to PresenceEngine without assigning identity."""
        self._body_runtime.publish_observation(
            source="home_assistant",
            kind="presence",
            subject=entity_id,
            value="present" if present else "absent",
            confidence=0.98,
            provenance={"entity_id": entity_id},
        )
        try:
            from managers.settings_manager import config
            if not bool(self._body_setting(config, "HOME_ASSISTANT_PRESENCE_ENABLED", False)):
                return
            from core.state import state
            vision = getattr(state, "vision", None)
            presence = getattr(vision, "presence_engine", None) if vision is not None else None
            if presence is None:
                presence = getattr(state, "presence_engine", None)
            if presence is not None and hasattr(presence, "on_external_presence"):
                presence.on_external_presence(entity_id, present, source="home_assistant")
        except Exception as exc:
            logger.debug("[UniversalConnector] external presence forwarding failed: %s", exc)

    def _maybe_boost_epistemic_pressure(self, percept: Percept) -> None:
        try:
            text = str(percept.payload).lower()
            if not any(marker in text for marker in _DISAGREEMENT_MARKERS):
                return
            pressure = getattr(self._organism, "pressure", None)
            if pressure is None or not hasattr(pressure, "boost"):
                return
            pressure.boost("epistemic", amount=0.08)
            logger.debug(
                f"[UniversalConnector] disagreement marker in {percept.source} "
                f"percept -> epistemic pressure boosted"
            )
        except Exception as e:
            logger.debug(f"[UniversalConnector] disagreement boost failed (non-fatal): {e}")

    def _is_recent_duplicate(self, percept: Percept) -> bool:
        now = time.time()
        needle = str(percept.payload)[:60].lower()
        for p in self._recent:
            if p.source != percept.source:
                continue
            if now - p.timestamp > DUPLICATE_WINDOW_S:
                continue
            if str(p.payload)[:60].lower() == needle:
                return True
        return False

    def pending_thought_candidates(self) -> List[Dict]:
        """
        Recent, non-expired percepts as candidate dicts for
        WorkspaceCompetition.compete(thoughts=...) — the same shape as any
        other thought candidate there (priority/actionability), so they
        compete fairly rather than being force-injected as the focus.

        Two real bugs fixed here (found by an external analysis that
        actually ran the v97 code, not just read it — same standard this
        whole engagement holds itself to):

        Fix A — every percept previously produced the same candidate id
        (compete() defaults thought.get('id', 'unknown_thought') when the
        key is absent, which it always was here). WorkspaceCompetition
        tracks win-history and focus-persistence PER id (_win_counts), so
        distinct Flux hypotheses were accidentally sharing attention
        history — a genuinely new hypothesis could inherit an older one's
        stagnation penalty, or vice versa. Fixed with a deterministic id
        built from modality+source+timestamp — stable for the same
        percept across repeated compete() calls within its TTL life
        (consistent tracking of THAT hypothesis), distinct across
        different percepts (no more cross-contamination).

        Fix B — compete() reads thought.get("source", "thought") at the
        TOP level to classify which module/attention-channel a thought
        belongs to (_CANDIDATE_MODULE_MAP). This dict only ever put
        modality/source inside a nested source_data — never at the top
        level — so every percept silently fell through to the
        "curiosity_engine" default regardless of modality. Fixed by
        exposing source = percept.modality at the top level, and
        registering the real modalities in _CANDIDATE_MODULE_MAP
        (workspace_competition.py) so the attention economy can actually
        tell a Flux hypothesis apart from a camera percept apart from
        ordinary curiosity, instead of all three looking identical to it.
        """
        now = time.time()
        self._recent = [p for p in self._recent if now - p.timestamp < RECENT_TTL_S]
        out = []
        for p in self._recent:
            content = str(p.payload)[:200]
            out.append({
                # compete()'s THOUGHTS branch builds name/label from
                # content/thought_type (verified by direct testing — it
                # does NOT read a 'label' key), not from arbitrary keys,
                # so this must match that real shape or the percept's own
                # text never surfaces past source_data.
                "id":            f"percept_{p.modality}_{p.source}_{int(p.timestamp * 1000)}",
                "source":        p.modality,  # top-level — see Fix B above
                "content":       content,
                "thought_type":  "percept",
                "topic":         content[:80],
                "priority":      round(min(1.0, p.salience * p.confidence * (1.0 + 0.3 * p.novelty)), 3),
                "actionability": 0.3,  # a percept isn't directly actionable like a goal
                "source_data":   {"modality": p.modality, "source": p.source, "provenance": p.provenance},
            })
        return out

    # ── internal ─────────────────────────────────────────────────────────

    def _store_memory(self, percept: Percept) -> None:
        try:
            ai_system = getattr(self._organism, "ai_system", None)
            mem = getattr(ai_system, "memory_system", None)
            if mem is None or not hasattr(mem, "add_memory"):
                return
            text = self._describe(percept)
            # importance scales with salience/confidence, category tags the
            # modality so it's distinguishable from conversational memories
            importance = round(0.4 + 0.4 * percept.salience * percept.confidence, 2)
            mem.add_memory(text, importance, f"percept:{percept.modality}", "Neutral", "Low")
        except Exception as e:
            logger.debug(f"[UniversalConnector] memory store failed (non-fatal): {e}")

    def _record_chapter(self, percept: Percept) -> None:
        try:
            ni = getattr(self._organism, "narrative_identity", None)
            if ni is None or not hasattr(ni, "record_chapter"):
                return
            ni.record_chapter(
                title       = f"A moment via {percept.source}",
                description = self._describe(percept),
                emotion     = "aware",
                significance= round(min(0.6, 0.3 + 0.3 * percept.salience), 3),
            )
        except Exception as e:
            logger.debug(f"[UniversalConnector] chapter record failed (non-fatal): {e}")

    @staticmethod
    def _describe(percept: Percept) -> str:
        base = str(percept.payload)
        bits = [base]
        if percept.confidence < 1.0:
            bits.append(f"(confidence {percept.confidence:.2f})")
        if percept.novelty > 0.5:
            bits.append("— this felt unexpected")
        return " ".join(bits)


def get_universal_connector(organism: Any) -> UniversalConnector:
    """
    One shared connector per organism — PresenceEngine, PerceptionLoop,
    and recursive_deliberation.deliberate() all need the SAME recent-
    percepts buffer, not one each. Cached directly on the organism object
    (matches the lazy-init-on-first-use pattern already used throughout
    this codebase, e.g. internal_loop.py's try/except component inits)
    rather than requiring a change to CognitiveOrganism.__init__.
    """
    existing = getattr(organism, "_universal_connector", None)
    if existing is not None:
        return existing
    conn = UniversalConnector(organism)
    try:
        organism._universal_connector = conn
    except Exception:
        pass
    return conn
