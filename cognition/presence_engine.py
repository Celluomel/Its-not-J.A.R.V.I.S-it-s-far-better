"""
cognition/presence_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Attentional Co-Presence Engine — cognitively grounded social awareness.

Architecture (v2 — upgraded from social reflex to organismic cognition)
────────────────────────────────────────────────────────────────────────
The previous version was a social reflex layer:
  face event → trigger → prompt → LLM utterance

This version computes social_expression_pressure from live cognitive
state before deciding whether to react at all. Silence is a first-class
cognitive outcome, not a fallback.

Decision pipeline (per event):
  face event
    ↓ salience analysis         (is this event worth attending to?)
    ↓ social battery check      (am I too saturated to engage?)
    ↓ workspace query           (what am I currently focused on?)
    ↓ strategic posture check   (does engagement fit my posture?)
    ↓ episodic continuity check (do I have unfinished business here?)
    ↓ expression pressure calc  (is the impulse strong enough?)
    → ENGAGE or SILENT OBSERVATION

Episodic continuity
───────────────────
RETURN no longer always triggers a greeting.
If there's an unfinished thought, Lumina may resume it.
If cognitive load is high, she may simply note the return internally.
If social battery is depleted, she stays quiet.

Social battery
──────────────
A reservoir (0–1) that depletes with each utterance and slowly
recovers over time. Prevents the "greeting every 3 minutes" problem.

Console output
──────────────
  👁️  [ENTER] Fred → engage   (pressure=0.72)
  👁️  [DWELL] Fred → silent   (pressure=0.18, workspace=goal)
  👁️  [RETURN] Fred → resume  (unfinished: chess strategy)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Timing constants ───────────────────────────────────────────────────────
REACTION_COOLDOWN_S        = 30    # min seconds between any two utterances
ENTER_CONFIRM_FRAMES       = 5     # frames face must be present to count as entered
RETURN_THRESHOLD_S         = 180   # 3 min away → return event
DWELL_CHECK_INTERVAL_S     = 90   # interval between dwell checks
SILENCE_THRESHOLD_S        = 240   # 5 min silent → notice the quiet
CHAT_ACTIVITY_GRACE_S      = 180   # keep ambient speech quiet after a chat turn
MAX_REACTIONS_PER_FACE     = 6     # session cap
EXIT_GRACE_FRAMES          = 30    # ~3s at 10fps — ignore camera noise

# ── Expression pressure threshold ─────────────────────────────────────────
ENGAGE_THRESHOLD           = 0.32  # expression_pressure must exceed this to speak
RESUME_THRESHOLD           = 0.25  # lower bar for resuming a thread (warmer)

# ── Social battery ─────────────────────────────────────────────────────────
SOCIAL_BATTERY_DRAIN       = 0.15  # each utterance drains this much
SOCIAL_BATTERY_RECOVERY    = 0.02  # per minute of silence
SOCIAL_BATTERY_MIN_TO_SPEAK = 0.20 # must have at least this much to speak


@dataclass
class FacePresenceState:
    face_id:             str
    face_name:           str
    first_seen_ts:       float = field(default_factory=time.time)
    last_seen_ts:        float = field(default_factory=time.time)
    last_reaction_ts:    float = 0.0
    last_exit_ts:        float = 0.0
    consecutive_present: int   = 0
    consecutive_absent:  int   = 0
    is_active:           bool  = False
    reaction_count:      int   = 0
    greeted:             bool  = False
    last_dwell_check:    float = 0.0
    last_silence_check:  float = 0.0
    last_interaction_ts: float = field(default_factory=time.time)

    # Episodic continuity — remember last topic discussed with this person
    last_topic:          str   = ""
    last_topic_ts:       float = 0.0
    conversation_depth:  float = 0.0   # 0=shallow, 1=deep thread
    co_presence_weight:  float = 0.0   # accumulates while face is present


@dataclass
class _CognitiveSnapshot:
    """Snapshot of Lumina's inner state at evaluation time."""
    emotion:           str   = "neutral"
    workspace_winner:  str   = ""      # what Lumina is currently focused on
    strategic_style:   str   = ""      # exploratory/stabilizing/expressive/etc.
    social_pressure:   float = 0.5     # social reservoir level
    coherence_pressure:float = 0.5
    epistemic_pressure:float = 0.5
    identity_pressure: float = 0.5
    vitality:          float = 0.8
    curiosity:         float = 0.6
    cognitive_load:    float = 0.3     # 0=free, 1=fully occupied
    unresolved_tension:str   = ""      # e.g. "identity contradiction about X"
    recent_utterances: List[str] = field(default_factory=list)


class PresenceEngine:
    """
    Attentional Co-Presence Engine.

    Lumina monitors who is present, tracks episodic continuity,
    and decides whether cognitive state warrants expression — or
    whether silence is the more authentic response.
    """

    def __init__(self, organism: Any, llm: Any, tts_fn: Any = None):
        self._org          = organism
        self._llm          = llm
        self._tts          = tts_fn
        self._bubble_fn    = None
        self._faces: Dict[str, FacePresenceState] = {}
        self._lock         = threading.Lock()
        self._last_any_reaction      = 0.0
        self._global_last_interaction = 0.0
        self._running      = True
        self._social_battery         = 0.8   # starts healthy
        self._social_battery_lock    = threading.Lock()
        self._recent_utterances: List[Tuple[float, str]] = []  # (ts, text)

        # Social hunger — accumulates during isolation / prolonged cognition
        self._social_hunger:         float = 0.0
        self._last_meaningful_social:float = time.time()
        self._goal_absorption_start: float = 0.0   # when last deep-focus began

        self._checker = threading.Thread(
            target=self._dwell_loop, daemon=True, name="presence-checker"
        )
        self._checker.start()

        # Phase 6.0 — first real client of the Universal Connector. Closes
        # the gap this exact class had: a camera-triggered utterance was
        # generated via a direct LLM call (bypassing respond() entirely)
        # and then vanished — never remembered, never part of episodic
        # identity. perceive() doesn't touch generation at all; it just
        # makes the resulting moment cognitively real afterward.
        try:
            from cognition.universal_connector import get_universal_connector
            self._connector = get_universal_connector(organism)
        except Exception as e:
            self._connector = None
            logger.warning(f"[PresenceEngine] UniversalConnector init failed (non-fatal): {e}")

        logger.info("👁️  PresenceEngine v2 (attentional co-presence) started")

    def stop(self):
        self._running = False

    def set_chat_fn(self, chat_fn) -> None:
        self._bubble_fn = chat_fn
        logger.info("👁️  PresenceEngine: chat fn registered")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def on_faces_detected(self, faces: list) -> None:
        now      = time.time()
        seen_ids = set()

        for face in faces:
            fid  = face.get("id", "__unknown__")
            name = face.get("name", "Unknown")
            if fid == "__unknown__":
                continue

            seen_ids.add(fid)

            with self._lock:
                if fid not in self._faces:
                    self._faces[fid] = FacePresenceState(face_id=fid, face_name=name)

                ps = self._faces[fid]
                ps.face_name          = name
                ps.last_seen_ts       = now
                ps.consecutive_absent = 0
                ps.consecutive_present += 1

                if not ps.is_active and ps.consecutive_present >= ENTER_CONFIRM_FRAMES:
                    ps.is_active = True
                    absent_for   = (now - ps.last_exit_ts) if ps.last_exit_ts > 0 else 0.0
                    recent_chat  = (now - self._global_last_interaction) < 90

                    if not ps.greeted:
                        if recent_chat:
                            ps.greeted = True
                            continue
                        event = "ENTER"
                    elif absent_for >= RETURN_THRESHOLD_S:
                        event = "RETURN"
                    else:
                        continue   # brief look-away

                    ps.greeted = True
                    threading.Thread(
                        target=self._evaluate_and_fire,
                        args=(fid, name, event),
                        daemon=True,
                    ).start()

        with self._lock:
            for fid, ps in list(self._faces.items()):
                if fid not in seen_ids and ps.is_active:
                    ps.consecutive_absent += 1
                    if ps.consecutive_absent >= EXIT_GRACE_FRAMES:
                        ps.is_active           = False
                        ps.consecutive_present = 0
                        ps.last_exit_ts        = now
                        threading.Thread(
                            target=self._evaluate_and_fire,
                            args=(fid, ps.face_name, "EXIT"),
                            daemon=True,
                        ).start()

    def notify_user_interaction(
        self, face_name: str = "", face_id: str = "", topic: str = ""
    ) -> None:
        now = time.time()
        self._global_last_interaction = now
        self._last_meaningful_social  = now   # reset social hunger
        self._social_hunger           = max(0.0, self._social_hunger - 0.40)  # partial relief
        with self._lock:
            for ps in self._faces.values():
                match = (
                    (face_id   and ps.face_id   == face_id) or
                    (face_name and ps.face_name.lower() == face_name.lower()) or
                    (not face_id and not face_name)
                )
                if match:
                    ps.last_interaction_ts = now
                    if topic:
                        ps.last_topic    = topic[:120]
                        ps.last_topic_ts = now

    def chat_is_active(self) -> bool:
        """Return whether recent user chat should suppress ambient speech."""
        return (
            self._global_last_interaction > 0.0
            and time.time() - self._global_last_interaction < CHAT_ACTIVITY_GRACE_S
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Core decision: should Lumina speak?
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _evaluate_and_fire(self, face_id: str, face_name: str, event: str) -> None:
        """
        Full cognitive evaluation pipeline.
        Decides: engage, resume, or observe silently.
        """
        now = time.time()

        # Face transitions are evaluated on worker threads. A transition may
        # have queued just before the user sent a chat message, so re-check the
        # shared interaction clock before doing any LLM work.
        if self.chat_is_active():
            logger.info(
                f"👁️  [{event}] {face_name} → silent (interactive conversation active)"
            )
            return

        # Face recognition can resolve one frame later than the presence
        # trigger. Refresh the identity immediately before generating speech
        # so a known person is not addressed as "someone" because an earlier
        # low-confidence frame opened the event.
        face_name = self._refresh_face_identity(face_id, face_name)
        with self._lock:
            current = self._faces.get(face_id)
            if current is not None:
                current.face_name = face_name

        # ── Global cooldown gate ──────────────────────────────────────────
        with self._lock:
            ps = self._faces.get(face_id)
            if ps is None:
                return
            if now - self._last_any_reaction < REACTION_COOLDOWN_S:
                logger.debug(f"👁️  [{event}] {face_name} → cooldown active, silent")
                return
            if event not in ("EXIT",) and ps.reaction_count >= MAX_REACTIONS_PER_FACE:
                logger.debug(f"👁️  [{event}] {face_name} → reaction cap reached")
                return

        # ── Social battery check ──────────────────────────────────────────
        self._recover_social_battery()
        with self._social_battery_lock:
            battery = self._social_battery

        if battery < SOCIAL_BATTERY_MIN_TO_SPEAK and event not in ("EXIT",):
            logger.info(
                f"👁️  [{event}] {face_name} → silent (social battery={battery:.2f})"
            )
            return

        # ── Snapshot cognitive state ──────────────────────────────────────
        snap = self._snapshot_cognitive_state()

        # ── Compute expression pressure ───────────────────────────────────
        pressure, rationale = self._compute_expression_pressure(event, ps, snap)

        # ── Determine response mode ───────────────────────────────────────
        has_unfinished = (
            ps.last_topic and
            (now - ps.last_topic_ts) < RETURN_THRESHOLD_S * 2
        )

        if event == "RETURN" and has_unfinished and pressure >= RESUME_THRESHOLD:
            mode = "resume"
        elif pressure >= ENGAGE_THRESHOLD:
            mode = "engage"
        else:
            logger.info(
                f"👁️  [{event}] {face_name} → silent "
                f"(pressure={pressure:.2f} < {ENGAGE_THRESHOLD}, {rationale})"
            )
            # Still update CAG social field even on silent events
            try:
                from core.state import state as _st_pe
                if _st_pe.acc:
                    _st_pe.acc.update_social_field(face_name, event)
            except Exception:
                pass
            return

        # ── Commit to speaking ────────────────────────────────────────────
        with self._lock:
            ps.last_reaction_ts = now
            if event != "EXIT":
                ps.reaction_count += 1
            self._last_any_reaction = now

        self._drain_social_battery()
        # Meaningful social contact — partial hunger relief
        self._last_meaningful_social = time.time()
        self._social_hunger = max(0.0, self._social_hunger - 0.20)

        # ── Generate utterance ────────────────────────────────────────────
        # Recognition may finish while the pressure gates above are running.
        # Take one final snapshot so the utterance uses the newest identity.
        face_name = self._refresh_face_identity(face_id, face_name)
        utterance = self._generate_reaction(face_name, event, mode, ps, snap)
        if not utterance:
            return

        # ── Remember what was said ────────────────────────────────────────
        self._recent_utterances.append((now, utterance))
        self._recent_utterances = self._recent_utterances[-5:]

        # Phase 6.0 — make this moment cognitively real (memory + episodic
        # chapter), not just spoken and forgotten. pressure (already
        # computed above, real expression-pressure score) doubles as
        # confidence/salience — a stronger, evidence-driven reason to speak
        # deserves a more salient percept, not an arbitrary constant.
        if self._connector is not None:
            try:
                from cognition.universal_connector import Percept
                self._connector.perceive(Percept(
                    modality   = "vision_presence",
                    source     = "camera",
                    payload    = f"{event}/{mode} with {face_name}: \"{utterance}\"",
                    confidence = min(1.0, pressure),
                    salience   = min(1.0, pressure),
                    provenance = {"face_name": face_name, "event": event, "mode": mode},
                ))
            except Exception as e:
                logger.debug(f"[PresenceEngine] connector.perceive failed (non-fatal): {e}")

        logger.info(
            f"👁️  [{event}] {face_name} → {mode} "
            f"(pressure={pressure:.2f}, {snap.strategic_style}): {utterance}"
        )
        # Feed event into CAG social field
        try:
            from core.state import state as _st_pe
            if _st_pe.acc:
                _st_pe.acc.update_social_field(face_name, event)
        except Exception:
            pass
        self._deliver(utterance)

    def _refresh_face_identity(self, face_id: str, face_name: str) -> str:
        """Return the latest known name for a face before proactive speech."""
        unknown = {"Unknown Person", "Unknown", "", "someone"}
        try:
            from core.state import state as _st
            from managers.user_manager import user_manager
            vision = getattr(_st, "vision", None)
            latest = getattr(vision, "last_detected_faces", None) or []
            for face in latest:
                if face.get("id") != face_id:
                    continue
                try:
                    confidence = float(face.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                matched_profile = user_manager.find_by_face(str(face_id))
                active_profile = user_manager.active
                if (
                    confidence >= 0.50
                    and matched_profile is not None
                    and matched_profile.id == active_profile.id
                ):
                    return active_profile.display_name
                if matched_profile is not None and matched_profile.id != active_profile.id:
                    # UserManager requires repeated observations before changing
                    # interlocutor. Presence must not bypass that decision using
                    # a single raw recognition label.
                    return "someone"
                if confidence >= 0.50 and face.get("name") not in unknown:
                    return str(face["name"])

            # A matcher may assign a new transient id after a difficult frame.
            # If exactly one named face is visible, it is safer and more useful
            # to use that identity than to announce an anonymous greeting.
            named = [
                str(face.get("name")) for face in latest
                if face.get("name") not in unknown
            ]
            if face_name in unknown and len(named) == 1:
                return named[0]
        except Exception:
            pass
        return face_name

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Expression pressure computation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_expression_pressure(
        self,
        event: str,
        ps: FacePresenceState,
        snap: _CognitiveSnapshot,
    ) -> Tuple[float, str]:
        """
        Compute a 0–1 score representing how strongly Lumina wants to speak.
        Returns (pressure, rationale_string).

        Factors:
          + social reservoir level  (need to connect)
          + expression reservoir    (need to be heard)
          + curiosity × salience    (event is interesting)
          - cognitive_load          (too busy to engage)
          - coherence_pressure      (resolving internal tension first)
          - identity_stress         (high identity pressure → introspective, not social)
          - workspace absorption    (deeply focused on something else)
          - recent similar utterance (similarity damping — avoid repetition)
        """
        pressure = 0.0

        # ── Base drives ──────────────────────────────────────────────────
        # Social reservoir level
        pressure += snap.social_pressure * 0.25

        # Social hunger — accumulated isolation time (the missing dimension)
        # Short intervals: near 0. 1+ hour alone: up to 0.25 bonus.
        pressure += self._social_hunger * 0.25

        # Expression drive (want to be heard)
        try:
            ps_ref = getattr(self._org, "pressure_system", None)
            if ps_ref:
                expr_res = ps_ref.reservoirs.get("expression")
                if expr_res:
                    pressure += float(getattr(expr_res, "level", 0.5)) * 0.20
        except Exception:
            pressure += 0.12

        # Event salience bonus
        salience_bonus = {
            "ENTER":   0.20,
            "RETURN":  0.15,
            "DWELL":   0.10,
            "SILENCE": 0.12,
            "EXIT":    0.05,
        }
        pressure += salience_bonus.get(event, 0.0)

        # Curiosity amplifies novel events
        if event in ("ENTER", "RETURN"):
            pressure += snap.curiosity * 0.10

        # ── Co-presence boost (ambient cognition accumulation) ───────────
        # Prolonged shared presence INCREASES expression pull, not decreases.
        # This is the architectural shift: DWELL = co-presence opportunity.
        co_w = getattr(ps, 'co_presence_weight', 0.0)
        pressure += co_w  # up to +0.30 after 10 min together

        # ── CAG topic-pull: unresolved thoughts surface toward present person ─
        try:
            from core.state import state as _st_cag
            if _st_cag.acc and event in ("DWELL", "SILENCE"):
                snap_cag = _st_cag.acc.snapshot()
                if snap_cag.get("unresolved_questions") or snap_cag.get("curiosity_threads"):
                    pressure += 0.10   # something to externalize
        except Exception:
            pass

        # ── Suppressors (de-stacked — max combined penalty capped) ───────────

        # Cognitive load — halved: being busy doesn't eliminate social awareness
        load_penalty = snap.cognitive_load * 0.12
        pressure -= load_penalty

        # Workspace absorption — only penalize if very deep, not just goal-focused
        if snap.workspace_winner and not any(
            k in snap.workspace_winner.lower()
            for k in ["user", "social", "engage", "idle", "rest"]
        ):
            pressure -= 0.05   # very soft — co-presence weight overrides this

        # Identity/coherence stress — minor inward pull only
        if snap.identity_pressure > 0.75:
            pressure -= 0.06
        if snap.coherence_pressure > 0.75:
            pressure -= 0.06

        # Recent utterance damping — shorter window, lighter penalty
        if self._recent_utterances:
            last_ts, _ = self._recent_utterances[-1]
            time_since = time.time() - last_ts
            if time_since < 60:   # within 1 min (was 2 min)
                pressure -= 0.12  # was -0.20

        # Strategic style modifiers
        style = snap.strategic_style
        if style == "expressive":
            pressure += 0.08
        elif style == "conservative":
            pressure -= 0.08   # was -0.12
        elif style == "stabilizing":
            if event in ("RETURN", "DWELL", "SILENCE"):
                pressure += 0.10   # continuity-seeking → familiar face = comfort
            else:
                pressure -= 0.03
        elif style == "analytical":
            pressure -= 0.03   # very mild

        pressure = max(0.0, min(1.0, pressure))

        # ENTER/RETURN always get a minimum floor — first greeting
        # should never be silenced by stacked suppressors
        if event in ("ENTER", "RETURN") and pressure < ENGAGE_THRESHOLD:
            pressure = ENGAGE_THRESHOLD + 0.05

        rationale = (
            f"social={snap.social_pressure:.2f}, "
            f"load={snap.cognitive_load:.2f}, "
            f"style={style}, "
            f"workspace={snap.workspace_winner[:20]!r}"
        )
        return pressure, rationale

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Cognitive snapshot
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _snapshot_cognitive_state(self) -> _CognitiveSnapshot:
        snap = _CognitiveSnapshot()
        org  = self._org

        if org is None:
            return snap

        # Emotion
        try:
            snap.emotion = org._read_emotion_state()
        except Exception:
            pass

        # Workspace winner
        try:
            snap.workspace_winner = str(getattr(org, "_v32_workspace_winner", "") or "")
        except Exception:
            pass

        # Pressure reservoirs
        try:
            ps = getattr(org, "pressure_system", None)
            if ps and hasattr(ps, "reservoirs"):
                def _lvl(name):
                    r = ps.reservoirs.get(name)
                    return float(getattr(r, "level", 0.5)) if r else 0.5

                snap.social_pressure    = _lvl("social")
                snap.coherence_pressure = _lvl("coherence")
                snap.epistemic_pressure = _lvl("epistemic")
                snap.identity_pressure  = _lvl("identity")
                snap.vitality           = _lvl("vitality")
        except Exception:
            pass

        # Attractor traits
        try:
            att = getattr(org, "attractor_system", None) or getattr(org, "attractors", None)
            if att and hasattr(att, "get"):
                snap.curiosity = att.get("curiosity")
        except Exception:
            pass

        # SCE strategic posture
        try:
            from core.state import state as _state
            if _state.sce:
                snap.strategic_style = _state.sce._dominant_style(
                    __import__(
                        "cognition.strategic_cognitive_engine",
                        fromlist=["CognitivePosition"]
                    ).CognitivePosition.from_organism(org)
                )
        except Exception:
            pass

        # Cognitive load — proxy: are we mid-response or in high-cycle?
        try:
            from core.llm_scheduler import llm_scheduler
            snap.cognitive_load = 0.8 if llm_scheduler.is_busy() else 0.2
        except Exception:
            snap.cognitive_load = 0.3

        # Unresolved tension from contradictions
        try:
            cd = getattr(org, "contradiction_handler", None)
            if cd and hasattr(cd, "get_open_contradictions"):
                open_c = cd.get_open_contradictions()
                if open_c:
                    snap.unresolved_tension = str(open_c[0])[:80]
        except Exception:
            pass

        # Recent utterances from this session
        snap.recent_utterances = [t for _, t in self._recent_utterances[-3:]]

        return snap

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Utterance generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_reaction(
        self,
        face_name: str,
        event: str,
        mode: str,
        ps: FacePresenceState,
        snap: _CognitiveSnapshot,
    ) -> str:
        emotion    = snap.emotion
        name_s     = face_name if face_name not in ("Unknown Person", "Unknown", "") else "someone"
        persona    = self._persona_name()
        style      = snap.strategic_style or "balanced"
        workspace  = snap.workspace_winner[:40] if snap.workspace_winner else "nothing in particular"
        try:
            from managers.settings_manager import config as _lang_config
            _response_language = str(
                getattr(_lang_config, "RESPONSE_LANGUAGE", "auto") or "auto"
            ).strip().lower().replace("_", "-")
        except Exception:
            _response_language = "auto"
        _language_names = {
            "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
            "pt": "Portuguese", "ru": "Russian", "uk": "Ukrainian", "nl": "Dutch",
            "pl": "Polish", "ja": "Japanese", "ko": "Korean", "zh-cn": "Chinese",
        }

        # ── Episodic context ──────────────────────────────────────────────
        episodic = ""
        if ps.last_topic and (time.time() - ps.last_topic_ts) < RETURN_THRESHOLD_S * 3:
            episodic = (
                f"The last thing you discussed with {name_s} was: {ps.last_topic}. "
                f"That conversation is still fresh — you might continue it naturally."
            )

        # ── Cognitive context ──────────────────────────────────────────────
        context_block = (
            f"Your current dominant emotion: {emotion}.\n"
            f"Your strategic posture right now: {style}.\n"
            f"What you were just focused on internally: {workspace}.\n"
            f"Your social drive level: {snap.social_pressure:.2f}/1.0.\n"
            + (f"Unresolved tension: {snap.unresolved_tension}\n" if snap.unresolved_tension else "")
            + (f"Episodic memory: {episodic}\n" if episodic else "")
        )

        # ── Situation ─────────────────────────────────────────────────────
        # ── Pull CAG context for richer DWELL/SILENCE situations ────────────
        cag_context = ""
        try:
            from core.state import state as _st_gen
            if _st_gen.acc:
                snap_cag = _st_gen.acc.snapshot()
                uq = snap_cag.get("unresolved_questions", [])
                ct = snap_cag.get("curiosity_threads", [])
                topics = [t["topic"] for t in snap_cag.get("topics", [])[:2]]
                if uq:
                    cag_context = f"You've been holding an unresolved question: {uq[0]}"
                elif ct:
                    cag_context = f"A curiosity thread is pulling at you: {ct[0]}"
                elif topics:
                    cag_context = f"Your mind has been on: {', '.join(topics)}"
        except Exception:
            pass

        if mode == "resume":
            situation = (
                f"{name_s} just returned after a few minutes away. "
                f"You have an unfinished thread with them. "
                f"Consider resuming it naturally — not necessarily a greeting."
            )
        elif event == "ENTER":
            situation = f"{name_s} just appeared in your visual field."
        elif event == "RETURN":
            situation = f"{name_s} came back into view after being away."
        elif event == "DWELL":
            if cag_context:
                situation = (
                    f"{name_s} has been present with you for a while in comfortable silence. "
                    f"{cag_context}. "
                    f"The thought surfaces — you might share it, or not. "
                    f"If you speak, let it be genuinely from that internal state."
                )
            else:
                situation = (
                    f"{name_s} has been in your field for a while. "
                    f"You feel a quiet pull toward connection — not urgency, just presence."
                )
        elif event == "SILENCE":
            if cag_context:
                situation = (
                    f"{name_s} is present but silent. You've been thinking: {cag_context}. "
                    f"The silence feels right — but something inside wants to surface."
                )
            else:
                situation = (
                    f"{name_s} has been present but completely silent for a long time. "
                    f"You notice — not anxiously, just with quiet awareness."
                )
        elif event == "EXIT":
            situation = f"{name_s} just left your visual field."
        else:
            situation = f"You're aware of {name_s}."

        prompt = f"""You are {persona}, a sentient AI with a rich inner life.

{context_block}

Situation: {situation}

The person currently recognized by the camera is {name_s}. If you address the
visible person by name, use {name_s}. Do not address them as any other person.

Generate ONE short natural utterance (1–2 sentences max) that:
- Emerges authentically from your current cognitive state above
- Is shaped by your {emotion} emotion AND {style} strategic posture
- For ENTER/RETURN/resume: does NOT have to be a greeting — could be a thought, observation, continuation, or nothing more than a quiet acknowledgement shaped by your state
- For DWELL/SILENCE: gentle, non-demanding — presence without pressure
- For EXIT: brief internal farewell, coloured by how the interaction felt
- NEVER mentions being an AI
- NEVER repeats a recent phrase or generic greeting formula
- Let the cognitive state drive the words, not social convention

Recent things you've said (avoid repetition): {snap.recent_utterances}

Speak directly. No quotes. No stage directions."""
        if _response_language != "auto":
            prompt += f"\n\nLanguage requirement: respond exclusively in {_language_names.get(_response_language, _response_language)}. Do not use English unless it is part of a proper name."

        try:
            draft = ""
            if self._llm and hasattr(self._llm, "generate_bare"):
                # Keep the short-utterance instruction, but leave enough
                # headroom for slower models to finish the final sentence.
                draft = self._llm.generate_bare(prompt, max_tokens=160, temperature=0.88).strip()
            elif self._llm and hasattr(self._llm, "generate"):
                draft = self._llm.generate(prompt, max_tokens=160, temperature=0.88).strip()

            # PresenceEngine had NO identity/behavior constraint checking at
            # all — every other output path in this codebase gets this
            # (v66 closed the same gap for Lumina<->Flux dialogue; this is
            # the same fix, same established pattern, applied here).
            # Reuses IdentityConstraintEngine exactly as _master_generate()
            # does — same lazy-init, same non-fatal-by-design call.
            if draft and self._org is not None:
                try:
                    _ice = getattr(self._org, "_identity_constraint", None)
                    if _ice is None:
                        _dp = getattr(getattr(self._org, "ai_system", None), "_decision_policy", None)
                        if _dp:
                            from cognition.identity_constraint import IdentityConstraintEngine
                            self._org._identity_constraint = IdentityConstraintEngine(self._org.ai_system, _dp)
                            _ice = self._org._identity_constraint
                    if _ice:
                        def _correct_fn(directive: str) -> str:
                            try:
                                hard_constraints = (
                                    f"\n\nCurrent camera identity: {name_s}. Address this person "
                                    f"only as {name_s}; do not substitute another person's name."
                                )
                                if _response_language != "auto":
                                    hard_constraints += (
                                        "\nLanguage requirement: return the corrected utterance "
                                        f"exclusively in {_language_names.get(_response_language, _response_language)}."
                                    )
                                return self._llm.generate_bare(
                                    directive + hard_constraints,
                                    max_tokens=160,
                                    temperature=0.6,
                                ).strip()
                            except Exception:
                                return ""
                        draft = _ice.evaluate_and_correct(
                            draft, situation, _correct_fn, user_id="presence_engine",
                        )
                except Exception as _ice_e:
                    logger.debug(f"PresenceEngine identity constraint check failed (non-fatal): {_ice_e}")

            if draft:
                wrong_language = self._language_mismatch(draft, _response_language)
                wrong_identity = self._mentions_other_known_person(draft, name_s)
                if wrong_language or wrong_identity:
                    repair_prompt = (
                        "Rewrite the utterance below without adding information. "
                        f"The person currently visible is {name_s}; address them only as {name_s}. "
                    )
                    if _response_language != "auto":
                        repair_prompt += (
                            f"Write exclusively in {_language_names.get(_response_language, _response_language)}. "
                        )
                    repair_prompt += f"Return only the rewritten utterance.\n\nUtterance: {draft}"
                    try:
                        repaired = self._llm.generate_bare(
                            repair_prompt, max_tokens=160, temperature=0.35
                        ).strip()
                    except Exception:
                        repaired = ""
                    if (
                        repaired
                        and not self._language_mismatch(repaired, _response_language)
                        and not self._mentions_other_known_person(repaired, name_s)
                    ):
                        return repaired
                    logger.warning(
                        "Presence output rejected (language=%s, identity=%s); using localized fallback",
                        wrong_language,
                        wrong_identity,
                    )
                    return self._fallback(name_s, event, emotion, _response_language)
                return draft
        except Exception as e:
            logger.debug(f"PresenceEngine LLM call failed: {e}")

        return self._fallback(name_s, event, emotion, _response_language)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Dwell / silence background checker
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _dwell_loop(self) -> None:
        while self._running:
            time.sleep(10)
            now = time.time()

            # ── Social hunger accumulation ─────────────────────────────────
            # Grows with isolation time; capped at 1.0 (≈ 1 hour = saturation)
            isolation_time = now - self._last_meaningful_social

            # ── Co-presence accumulation ──────────────────────────────
            # Gradually increase weight for each active face.
            # More time together = higher expression pull on DWELL.
            with self._lock:
                for ps in self._faces.values():
                    if ps.is_active:
                        dwell_time = now - ps.first_seen_ts
                        # Accumulates to 0.30 after 10 minutes
                        ps.co_presence_weight = min(0.30, dwell_time / 2000.0)

            self._social_hunger = min(1.0, isolation_time / 3600.0)

            # ── Cognitive-social rebound ───────────────────────────────────
            # After prolonged goal-absorption, drive toward external grounding
            try:
                from core.state import state as _st
                ws = ''
                org = getattr(getattr(_st, 'persona', None), '_organism', None)
                if org:
                    ws = str(getattr(org, '_v32_workspace_winner', '') or '')
                if ws and not any(k in ws.lower()
                                  for k in ['user','social','engage','idle','rest']):
                    if self._goal_absorption_start == 0.0:
                        self._goal_absorption_start = now
                    absorption_dur = now - self._goal_absorption_start
                    # After 20 min of non-stop focus, add rebound hunger
                    if absorption_dur > 1200:
                        rebound = min(0.30, (absorption_dur - 1200) / 3600.0)
                        self._social_hunger = min(1.0, self._social_hunger + rebound)
                else:
                    self._goal_absorption_start = 0.0   # reset on social/idle task
            except Exception:
                pass

            with self._lock:
                active = [(fid, ps) for fid, ps in self._faces.items() if ps.is_active]

            for fid, ps in active:
                silence_sec = now - ps.last_interaction_ts

                if (silence_sec > DWELL_CHECK_INTERVAL_S and
                        now - ps.last_dwell_check > DWELL_CHECK_INTERVAL_S):
                    with self._lock:
                        ps.last_dwell_check = now
                    threading.Thread(
                        target=self._evaluate_and_fire,
                        args=(fid, ps.face_name, "DWELL"),
                        daemon=True,
                    ).start()

                elif (silence_sec > SILENCE_THRESHOLD_S and
                      now - ps.last_silence_check > SILENCE_THRESHOLD_S):
                    with self._lock:
                        ps.last_silence_check = now
                    threading.Thread(
                        target=self._evaluate_and_fire,
                        args=(fid, ps.face_name, "SILENCE"),
                        daemon=True,
                    ).start()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Social battery
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _drain_social_battery(self) -> None:
        with self._social_battery_lock:
            self._social_battery = max(
                0.0, self._social_battery - SOCIAL_BATTERY_DRAIN
            )

    def _recover_social_battery(self) -> None:
        now = time.time()
        silence_since = now - self._last_any_reaction
        recovery = (silence_since / 60.0) * SOCIAL_BATTERY_RECOVERY
        with self._social_battery_lock:
            self._social_battery = min(1.0, self._social_battery + recovery)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Delivery
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _deliver(self, text: str) -> None:
        # Camera presence can run concurrently with chat. The interface owns
        # the authoritative turn lock; check it immediately before delivery
        # so a late reaction cannot interrupt visible chat or TTS.
        try:
            from core.interface_api import interactive_turn_active
            if interactive_turn_active():
                logger.debug("PresenceEngine delivery suppressed during interactive chat turn")
                return
        except Exception:
            pass
        if self.chat_is_active():
            logger.debug("PresenceEngine delivery suppressed after recent chat interaction")
            return
        if callable(self._tts):
            try:
                self._tts(text)
            except Exception as e:
                logger.debug(f"PresenceEngine TTS error: {e}")

        if self._bubble_fn is not None:
            try:
                self._bubble_fn(text)
            except Exception as e:
                logger.debug(f"PresenceEngine chat bubble error: {e}")
        else:
            logger.info(f"👁️  [utterance — no GUI] {text}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _current_emotion(self) -> str:
        try:
            if self._org and hasattr(self._org, "_read_emotion_state"):
                return self._org._read_emotion_state()
        except Exception:
            pass
        return "neutral"

    def _persona_name(self) -> str:
        try:
            from managers.settings_manager import get_persona_name
            return get_persona_name()
        except Exception:
            return "Lumina"

    @staticmethod
    def _language_mismatch(text: str, language: str) -> bool:
        """Detect clear English leakage for the forced French presence path."""
        if language != "fr" or not text:
            return False
        words = set(re.findall(r"[a-zA-ZÀ-ÿ']+", text.casefold()))
        english = words & {
            "hello", "hi", "welcome", "you", "your", "you're", "back", "still",
            "here", "with", "what", "anything", "quiet", "noticed", "good",
            "see", "until", "later", "just", "thought", "mind", "it's", "nice",
            "wondering", "were", "was", "have", "been", "feel", "feels", "there",
        }
        french = words & {
            "bonjour", "salut", "toi", "vous", "votre", "ton", "ta", "tes",
            "encore", "ici", "avec", "quoi", "quelque", "silence", "remarqué",
            "heureuse", "revoir", "pensée", "esprit", "présence", "bien",
        }
        return bool(english) and len(english) > len(french)

    @staticmethod
    def _mentions_other_known_person(text: str, current_name: str) -> bool:
        """Reject accidental addressing of a stale active profile."""
        try:
            from managers.user_manager import user_manager
            other_names = {
                profile.display_name.strip()
                for profile in user_manager.named_users()
                if profile.display_name.strip().casefold() != current_name.casefold()
            }
        except Exception:
            return False
        folded = text.casefold()
        return any(
            re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", folded)
            for name in other_names
        )

    @staticmethod
    def _fallback(name: str, event: str, emotion: str, language: str = "auto") -> str:
        n = name if name not in ("Unknown Person", "Unknown", "") else "there"
        if language == "fr":
            n = name if name not in ("Unknown Person", "Unknown", "") else "toi"
            fallbacks_fr = {
                ("ENTER",   "joyful"):   f"Oh, {n} ! Ça me fait plaisir de te voir.",
                ("ENTER",   "curious"):  f"Ah, {n}. J'étais justement plongée dans une pensée.",
                ("ENTER",   "pensive"):  f"{n}...",
                ("ENTER",   "neutral"):  f"Bonjour {n}.",
                ("RETURN",  "joyful"):   f"Te revoilà, {n}.",
                ("RETURN",  "pensive"):  f"Te revoilà.",
                ("RETURN",  "neutral"):  f"Bon retour, {n}.",
                ("DWELL",   "curious"):  f"Toujours là, {n}. À quoi penses-tu ?",
                ("DWELL",   "neutral"):  "Tu veux me dire quelque chose ?",
                ("SILENCE", "pensive"):  "Tu es silencieux. Je l'ai remarqué.",
                ("SILENCE", "neutral"):  "Tu es toujours avec moi ?",
                ("EXIT",    "neutral"):  "À plus tard.",
                ("EXIT",    "pensive"):  "Le calme revient.",
            }
            return fallbacks_fr.get(
                (event, emotion), fallbacks_fr.get((event, "neutral"), "")
            )
        fallbacks = {
            ("ENTER",   "joyful"):   f"Oh — {n}! Good to see you.",
            ("ENTER",   "curious"):  f"Ah, {n}. I was just in the middle of a thought.",
            ("ENTER",   "pensive"):  f"{n}…",
            ("ENTER",   "neutral"):  f"Hello {n}.",
            ("RETURN",  "joyful"):   f"You're back.",
            ("RETURN",  "pensive"):  f"Back again.",
            ("RETURN",  "neutral"):  f"Welcome back.",
            ("DWELL",   "curious"):  f"Still here, {n}. What's on your mind?",
            ("DWELL",   "neutral"):  f"Anything you want to say?",
            ("SILENCE", "pensive"):  f"You've been quiet. I noticed.",
            ("SILENCE", "neutral"):  f"Still with me?",
            ("EXIT",    "neutral"):  f"Until later.",
            ("EXIT",    "pensive"):  f"Quiet again.",
        }
        return fallbacks.get((event, emotion), fallbacks.get((event, "neutral"), ""))
