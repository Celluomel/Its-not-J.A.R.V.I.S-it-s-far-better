"""
lumina_brain_bridge.py  —  PandoraBOX ↔ Brain Visualizer (v2 — reactive)
======================================================================

Changes v2
----------
  - Poll rate 2.0s → 0.5s  (4× faster)
  - EMA α 0.25 → 0.45      (faster visual response)
  - EventBus subscription: SURPRISE_DETECTED, CONTRADICTION_DETECTED,
    DRIVE_SPIKE → instant activation spikes, not waiting for next poll
  - All readers print a one-time warning when a module is missing so you
    can see what's connected vs falling back
  - Temporal pulse: activations decay toward a resting baseline between
    events so the brain is never fully frozen

Network → PandoraBOX module mapping
--------------------------------
  DMN  ← thought_stream + inner_monologue + narrative_identity
  FPN  ← goal_ecology.dominant_drive + cognitive_validator.health
  VAN  ← predictive_mind.surprise_index + tension.overall_arousal
  LIM  ← relational_memory + social pressure
  DAN  ← curiosity_engine.global_level + epistemic pressure
  SMN  ← expression pressure + verbosity
  VIS  ← ambient_vision + CCS observatory
"""

import logging
import math
import os
import sys
import threading
import time
from typing import Optional, Set

logger = logging.getLogger(__name__)

_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
if _VIZ_DIR not in sys.path:
    sys.path.insert(0, _VIZ_DIR)

from network_mapping import BrainState, create_default_brain_state, NETWORKS

# ── Tuning ─────────────────────────────────────────────────────────────────────
UPDATE_INTERVAL  = 0.5    # poll every 500ms (was 2s)
SMOOTHING_ALPHA  = 0.45   # EMA — faster visual response (was 0.25)
SPIKE_ALPHA      = 0.85   # EMA for instant event spikes
DECAY_RATE       = 0.04   # per-cycle decay toward resting baseline
RESTING          = {      # baseline when nothing specific is happening
    "DMN": 0.30, "FPN": 0.25, "VAN": 0.10,
    "LIM": 0.15, "DAN": 0.20, "SMN": 0.10, "VIS": 0.05,
}
MEMORY_SCALE_MAX = 2000
INTERACTION_MAX  = 200

# ── Helpers ────────────────────────────────────────────────────────────────────
def _ema(cur, tgt, alpha=SMOOTHING_ALPHA):
    return cur + alpha * (tgt - cur)

def _spike(cur, tgt, alpha=SPIKE_ALPHA):
    return cur + alpha * (tgt - cur)

def _clamp(v):
    return max(0.0, min(1.0, float(v)))

def _log_scale(value, max_val):
    if value <= 0: return 0.0
    return _clamp(math.log1p(value) / math.log1p(max_val))


class LuminaBrainBridge:

    def __init__(self, organism, window_title="PandoraBOX — Cognitive Brain"):
        self._o             = organism
        self._window_title  = window_title
        self._shared_state  = create_default_brain_state()
        self._running       = False
        self._poll_thread:   Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._warned: Set[str] = set()      # one-time missing-module warnings

        # Previous-value tracking for delta signals
        self._prev_chapter_count = 0
        self._prev_memory_count  = 0

        # Event spike buffers — set by event callbacks, consumed by next poll
        self._spike_VAN = 0.0
        self._spike_DAN = 0.0
        self._spike_DMN = 0.0
        self._spike_FPN = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        if self._running: return
        self._running = True
        self._subscribe_events()
        self._poll_once()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="BrainBridge-Poller", daemon=True)
        self._poll_thread.start()
        self._render_thread = threading.Thread(
            target=self._render_loop, name="BrainBridge-Renderer", daemon=True)
        self._render_thread.start()
        logger.info("[BrainBridge] v2 started — 0.5s poll, event spikes active")

    def stop(self):
        self._running = False

    @property
    def shared_state(self) -> BrainState:
        return self._shared_state

    # ── EventBus subscription ─────────────────────────────────────────────────

    def _subscribe_events(self):
        """Subscribe to high-frequency cognitive events for instant visual spikes."""
        try:
            o = self._o
            ai = getattr(o, 'ai_system', None)
            bus = None
            for attr in ('event_bus', '_event_bus', 'bus'):
                bus = getattr(o, attr, None) or (getattr(ai, attr, None) if ai else None)
                if bus: break
            if not bus:
                logger.debug("[BrainBridge] No EventBus found — falling back to poll-only")
                return

            def on_surprise(event):
                mag = float(getattr(event, 'magnitude', 0.6))
                self._spike_VAN = max(self._spike_VAN, mag)

            def on_contradiction(event):
                mag = float(getattr(event, 'pressure', 0.6))
                self._spike_VAN = max(self._spike_VAN, mag * 0.8)
                self._spike_DMN = max(self._spike_DMN, mag * 0.5)

            def on_drive_spike(event):
                drive = str(getattr(event, 'drive', '')).lower()
                mag   = float(getattr(event, 'intensity', 0.7))
                if 'curiosi' in drive or 'epistemi' in drive:
                    self._spike_DAN = max(self._spike_DAN, mag)
                elif 'goal' in drive or 'cohere' in drive:
                    self._spike_FPN = max(self._spike_FPN, mag)
                elif 'express' in drive or 'social' in drive:
                    self._spike_DMN = max(self._spike_DMN, mag * 0.6)

            for evt, cb in [
                ('SURPRISE_DETECTED',     on_surprise),
                ('CONTRADICTION_DETECTED', on_contradiction),
                ('DRIVE_SPIKE',           on_drive_spike),
            ]:
                try:
                    bus.subscribe(evt, cb)
                except Exception:
                    pass

            logger.info("[BrainBridge] EventBus subscribed: SURPRISE, CONTRADICTION, DRIVE_SPIKE")
        except Exception as e:
            logger.debug(f"[BrainBridge] Event subscription failed: {e}")

    # ── Poll loop ──────────────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                logger.debug(f"[BrainBridge] Poll error: {e}")
            time.sleep(UPDATE_INTERVAL)

    def _poll_once(self):
        o = self._o
        s = self._shared_state

        # Read all modules
        dmn = self._read_dmn(o)
        fpn = self._read_fpn(o)
        van = self._read_van(o)
        lim = self._read_lim(o)
        dan = self._read_dan(o)
        smn = self._read_smn(o)
        vis = self._read_vis(o)

        # Apply event spikes on top of polled values
        van = max(van, self._spike_VAN);  self._spike_VAN = 0.0
        dan = max(dan, self._spike_DAN);  self._spike_DAN = 0.0
        dmn = max(dmn, self._spike_DMN);  self._spike_DMN = 0.0
        fpn = max(fpn, self._spike_FPN);  self._spike_FPN = 0.0

        targets = {"DMN":dmn, "FPN":fpn, "VAN":van,
                   "LIM":lim, "DAN":dan, "SMN":smn, "VIS":vis}

        for net in NETWORKS:
            cur  = s.get_region_activation(net)
            tgt  = targets[net]
            # If target > current: fast approach; if below: also decay toward resting
            if tgt > cur:
                new = _clamp(_ema(cur, tgt))
            else:
                # Decay: blend current toward resting, then toward target
                toward_rest = cur - DECAY_RATE * (cur - RESTING[net])
                new = _clamp(_ema(toward_rest, tgt, 0.3))
            s.set_region(net, new)

        logger.debug(
            f"[BB] DMN={dmn:.2f} FPN={fpn:.2f} VAN={van:.2f} "
            f"LIM={lim:.2f} DAN={dan:.2f} SMN={smn:.2f} VIS={vis:.2f}"
        )
        # Live console readout every 5 polls (~2.5s)
        self._poll_count = getattr(self, '_poll_count', 0) + 1
        if self._poll_count % 5 == 1:
            s2 = self._shared_state
            def _bar(v, w=12):
                filled = int(v * w)
                return "█" * filled + "░" * (w - filled)
            print(
                f"\r🧠 "
                f"DMN {_bar(s2.get_region_activation('DMN'))} "
                f"DAN {_bar(s2.get_region_activation('DAN'))} "
                f"VAN {_bar(s2.get_region_activation('VAN'))} "
                f"FPN {_bar(s2.get_region_activation('FPN'))} "
                f"SMN {_bar(s2.get_region_activation('SMN'))} "
                f"LIM {_bar(s2.get_region_activation('LIM'))} "
                f"VIS {_bar(s2.get_region_activation('VIS'))}",
                end="", flush=True
            )

    # ── Module readers ─────────────────────────────────────────────────────────

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(f"[BrainBridge] {msg}")

    def _read_dmn(self, o) -> float:
        """DMN — thought_stream activity + narrative depth + workspace self-referential items"""
        try:
            score = 0.0; weight = 0.0

            # Thought stream: recency + volume
            ts = getattr(o, 'thought_stream', None)
            if ts:
                recent = ts.recent(8)
                now    = time.time()
                fresh  = sum(1 for t in recent if now - getattr(t, 'timestamp', now) < 90)
                total  = len(recent)
                # Boost: even 1-2 recent thoughts = meaningful DMN activity
                score += _clamp(fresh / 4.0 + total / 8.0 * 0.3) * 0.45
                weight += 0.45

            # Narrative identity: chapter count = accumulated self-model depth
            ni = getattr(o, 'narrative_identity', None)
            if ni:
                summ     = ni.summary()
                chapters = summ.get('chapters', 0)
                beliefs  = summ.get('beliefs', 0)
                delta    = max(0, chapters - self._prev_chapter_count)
                self._prev_chapter_count = chapters
                # Chapters growing = active narrative processing
                growth = _clamp(delta * 0.5)
                depth  = _log_scale(chapters + beliefs * 2, 120)
                score += _clamp(growth + depth * 0.6) * 0.35
                weight += 0.35

            # Meta-cognition reflection = DMN self-reference
            mc = getattr(o, 'meta_cognition', None)
            if mc:
                try:
                    summ = mc.summary() if callable(getattr(mc,'summary',None)) else {}
                    cycles = summ.get('reflection_cycles', summ.get('cycles', 0))
                    score += _log_scale(cycles, 200) * 0.20
                    weight += 0.20
                except Exception:
                    pass

            result = _clamp(score / weight) if weight > 0 else RESTING['DMN']
            # Floor: DMN never fully off while thought_stream exists
            return max(result, 0.25 if ts else RESTING['DMN'])
        except Exception as e:
            logger.debug(f"[BB] _read_dmn: {e}"); return RESTING['DMN']

    def _read_fpn(self, o) -> float:
        """FPN — goal_ecology urgency + cognitive_validator health"""
        try:
            score = 0.0

            energy = getattr(o, 'energy', None)
            e_level = _clamp(energy.level() / 100.0) if energy else 0.5

            goal_ecology = getattr(o, 'goal_ecology', None)
            if goal_ecology:
                drive   = goal_ecology.dominant_drive(energy.level() if energy else 100.0)
                urgency = _clamp(getattr(drive, 'urgency', 0.4))
                score  += urgency * 0.55
            else:
                self._warn_once('fpn_ge', "goal_ecology not found")
                score += 0.25

            ai = getattr(o, 'ai_system', None)
            cv = None
            if ai:
                for attr in ('_cognitive_validator', 'cognitive_validator'):
                    cv = getattr(ai, attr, None)
                    if cv: break
            if cv:
                health = cv.health_report().get('overall_health', 0.5)
                score += _clamp(float(health)) * 0.45
            else:
                score += e_level * 0.45

            return _clamp(score)
        except Exception as e:
            logger.debug(f"[BB] _read_fpn: {e}"); return RESTING['FPN']

    def _read_van(self, o) -> float:
        """VAN — surprise_index + tension arousal"""
        try:
            score = 0.0

            pm = getattr(o, 'predictive_mind', None)
            if pm:
                surprise = pm.stability_metrics().get('surprise_index', 0.0)
                score   += _clamp(surprise / 0.5) * 0.55
            else:
                self._warn_once('van_pm', "predictive_mind not found")

            te = getattr(o, 'tension_engine', None)
            if te:
                arousal = te.current().overall_arousal()
                score  += _clamp(arousal) * 0.45

            return _clamp(score)
        except Exception as e:
            logger.debug(f"[BB] _read_van: {e}"); return RESTING['VAN']

    def _read_lim(self, o) -> float:
        """LIM — relational memory (on ai_system) + social pressure"""
        try:
            score = 0.0

            # relational_memory lives on ai_system, not organism
            ai  = getattr(o, 'ai_system', None)
            uid = getattr(ai, '_active_user_id', None) if ai else None
            rm  = getattr(ai, 'relational_memory', None) if ai else None
            if rm and uid:
                rel   = rm.get_or_create(uid)
                count = getattr(rel, 'interaction_count', 0)
                score += _log_scale(count, INTERACTION_MAX) * 0.55
            elif not rm:
                self._warn_once('lim_rm', "relational_memory not on ai_system either — LIM uses pressure only")

            pressure = getattr(o, 'pressure', None)
            if pressure:
                soc = getattr(pressure, 'reservoirs', {}).get('social')
                if soc:
                    score += _clamp(soc.effective_pressure()) * 0.45

            return _clamp(score) if score > 0 else RESTING['LIM']
        except Exception as e:
            logger.debug(f"[BB] _read_lim: {e}"); return RESTING['LIM']

    def _read_dan(self, o) -> float:
        """DAN — curiosity engine + epistemic pressure + world model topic richness"""
        try:
            score = 0.0

            cu = None
            for attr in ('curiosity', 'curiosity_engine', '_curiosity'):
                cu = getattr(o, attr, None)
                if cu: break
            if cu:
                level = None
                for meth in ('global_level', 'level', 'get_level'):
                    fn = getattr(cu, meth, None)
                    if callable(fn):
                        level = fn(); break
                if level is not None:
                    score += _clamp(float(level)) * 0.50
            else:
                self._warn_once('dan_cu', "curiosity engine not found — DAN relies on epistemic + world model")

            pressure = getattr(o, 'pressure', None)
            if pressure:
                ep = getattr(pressure, 'reservoirs', {}).get('epistemic')
                if ep:
                    score += _clamp(ep.effective_pressure()) * 0.35

            # World model topic richness = directed knowledge/attention breadth
            wm = getattr(o, 'world_model', None)
            if wm:
                try:
                    summ   = wm.summary() if callable(getattr(wm, 'summary', None)) else {}
                    topics = summ.get('topics', summ.get('topic_count', 0))
                    score += _log_scale(topics, 300) * 0.15
                except Exception:
                    pass

            return _clamp(score) if score > 0 else RESTING['DAN']
        except Exception as e:
            logger.debug(f"[BB] _read_dan: {e}"); return RESTING['DAN']

    def _read_smn(self, o) -> float:
        """SMN — expression pressure + verbosity (output motor system)"""
        try:
            score = 0.0

            pressure = getattr(o, 'pressure', None)
            if pressure:
                expr = getattr(pressure, 'reservoirs', {}).get('expression')
                if expr:
                    score += _clamp(expr.effective_pressure()) * 0.65

            ai = getattr(o, 'ai_system', None)
            if ai:
                vl = getattr(ai, '_verbosity_level', None)
                if vl is not None:
                    score += _clamp(float(vl) / 3.0) * 0.35

            return _clamp(score) if score > 0 else RESTING['SMN']
        except Exception as e:
            logger.debug(f"[BB] _read_smn: {e}"); return RESTING['SMN']

    def _read_vis(self, o) -> float:
        """
        VIS — Real camera perception + ambient vision GW recency.

        Camera off  → RESTING (near zero) — PandoraBOX's eyes are closed
        Camera on   → base 0.30
        Recent ambient vision snapshot (<120s) → +0.45 scaled by recency
        Known face in last snapshot → +0.15 bonus
        CCS observatory → small modulator ×0.10
        """
        try:
            # ── Camera active? ───────────────────────────────────────────────
            # Try organism._vision_manager first, then fall back to state.vision
            vm = getattr(o, '_vision_manager', None)
            if vm is None:
                try:
                    from core.state import state as _state
                    vm = getattr(_state, 'vision', None)
                    if vm is not None:
                        o._vision_manager = vm   # cache for next poll
                except Exception:
                    pass
            camera_on = vm is not None and getattr(vm, 'camera_active', False)

            if not camera_on:
                return RESTING['VIS']   # eyes closed

            score = 0.30  # base: camera running

            # ── Recent ambient vision broadcasts in Global Workspace ─────────
            ws = getattr(o, 'workspace', None)
            if ws:
                try:
                    now = time.time()
                    for item in ws.recent(20):
                        src = getattr(item, 'source', '')
                        age = now - getattr(item, 'timestamp', now)
                        if age > 180:
                            continue
                        if src == 'ambient_vision':
                            score += 0.45 * max(0.0, 1.0 - age / 120.0)
                        elif src == 'ambient_vision.faces':
                            score += 0.15 * max(0.0, 1.0 - age / 120.0)
                except Exception:
                    pass

            # ── Observatory CCS (small modulator) ────────────────────────────
            obs = getattr(o, 'observatory', None)
            if obs:
                try:
                    snap = obs.latest() if hasattr(obs, 'latest') else None
                    ccs  = getattr(snap, 'ccs', None) if snap else None
                    if ccs is not None and float(ccs) > 0:
                        score += _clamp(float(ccs)) * 0.10
                except Exception:
                    pass

            return _clamp(score)
        except Exception as e:
            logger.debug(f"[BB] _read_vis: {e}"); return RESTING['VIS']

    # ── Renderer ───────────────────────────────────────────────────────────────

    def _render_loop(self):
        try:
            from brain_visualizer.brain_visualizer import run_brain_visualizer
            run_brain_visualizer(base_dir=_VIZ_DIR, shared_state=self._shared_state)
        except ImportError as e:
            logger.error(f"[BrainBridge] Missing dep: {e}\n  pip install PyOpenGL glfw")
        except Exception as e:
            logger.error(f"[BrainBridge] Renderer crashed: {e}")
        finally:
            self._running = False
