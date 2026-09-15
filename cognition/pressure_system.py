"""
PressureSystem — Bio-Inspired Internal Drive Architecture
==========================================================
This module addresses the fundamental gap identified in the architectural
audit: drives were *computed reactively* rather than *autonomously building*.

The analogy is biological: hunger doesn't wait to be computed. It builds
continuously, escalates when unmet, and releases when satisfied. The
cognitive organism needs the same dynamic.

Six pressure reservoirs:
  epistemic   — need to know, understand, resolve uncertainty
  social      — need for genuine connection and being heard
  coherence   — need to resolve contradictions and tensions
  vitality    — need for cognitive rest and energy recovery
  identity    — need to affirm and maintain sense of self
  expression  — need to externalize inner state, be understood

Each reservoir has:
  - autonomous buildup (fills at own rate every tick)
  - satiation on relevant action (drops when the need is met)
  - urgency escalation (unmet for >10min → multiplier grows)
  - cross-drive coupling (biological realism)

Key biological properties modeled:
  1. Homeostatic accumulation — need builds continuously
  2. Temporal escalation     — urgency grows as need persists
  3. Satiation               — satisfaction actually reduces pressure
  4. Conservation mode       — low vitality suppresses other drives
  5. Threat-rigidity         — high identity/coherence pressure
                               suppresses curiosity
  6. Information gain        — prediction error feeds epistemic pressure

Integration:
  - tick()      called every slow cycle (120s) from internal_loop.py
  - satiate()   called from _post_interaction() based on what happened
  - urgency()   feeds GoalEcology to replace flat tension urgency
  - prompt_fragment() injected into _build_prompt_additions()
  - EventBus    emits on peak pressure (> CRITICAL_THRESHOLD)

Persists to: data/persona/pressure.json
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
ESCALATION_ONSET   = 0.65   # above this, escalation timer starts
CRITICAL_THRESHOLD = 0.85   # above this, emit bus event
SATIATION_FLOOR    = 0.10   # pressure never drops fully to 0 (latent need)
TICK_INTERVAL      = 120.0  # seconds between ticks (matches slow cycle)

# ── Reservoir definitions ─────────────────────────────────────────────────────
# fill_rate_per_tick  : how much pressure builds each tick when unmet
# natural_decay       : how fast it drops toward resting state without satiation
# resting_level       : equilibrium level when everything is fine
# satiation_amount    : how much a direct satiation drops the level

RESERVOIR_DEFS = {
    "epistemic": {
        "fill_rate_per_tick": 0.04,    # reduced — curiosity builds gradually
        "natural_decay":      0.03,    # slightly faster decay
        "resting_level":      0.30,    # always curious but not overwhelmingly so
        "satiation_amount":   0.35,    # one good insight saturates significantly
    },
    "social": {
        "fill_rate_per_tick": 0.03,    # slow build — humans aren't always needed
        "natural_decay":      0.04,    # decays faster — loneliness fades with time
        "resting_level":      0.20,    # low resting need (introvert baseline)
        "satiation_amount":   0.50,    # one good exchange is very satisfying
    },
    "coherence": {
        "fill_rate_per_tick": 0.04,    # reduced — coherence need builds steadily, not explosively
        "natural_decay":      0.03,    # moderate decay — resolves with time
        "resting_level":      0.15,    # low resting (coherent state is natural)
        "satiation_amount":   0.40,
    },
    "vitality": {
        "fill_rate_per_tick": 0.05,    # builds during high cognitive load
        "natural_decay":      0.08,    # fast decay — rest is efficient
        "resting_level":      0.20,    # low resting (rested state is default)
        "satiation_amount":   0.50,    # sleep/rest is highly restorative
    },
    "identity": {
        "fill_rate_per_tick": 0.03,    # slow — identity is stable normally
        "natural_decay":      0.02,
        "resting_level":      0.25,    # moderate resting (identity maintenance ongoing)
        "satiation_amount":   0.35,
    },
    "expression": {
        "fill_rate_per_tick": 0.07,    # builds when thoughts accumulate unexpressed
        "natural_decay":      0.04,
        "resting_level":      0.30,    # moderate — always something to say
        "satiation_amount":   0.40,    # speaking/writing releases it well
    },
    "uncertainty": {
        # Free Energy Principle (Friston): prediction error drives exploration.
        # Builds when predictions fail; satiates when understanding is achieved.
        "fill_rate_per_tick": 0.02,    # slow base fill — needs prediction errors to spike
        "natural_decay":      0.05,    # decays quickly when things make sense
        "resting_level":      0.20,    # baseline uncertainty is healthy and motivating
        "satiation_amount":   0.45,    # one good explanation resolves a lot
    },
    "aspirational": {
        # Superego pressure — gap between what Lumina is and wants to become.
        # Fed by AspirationalSelf.aspirational_pressure().
        # Satiates when an aspiration tension is meaningfully reduced.
        "fill_rate_per_tick": 0.01,    # very slow autonomous fill — driven externally
        "natural_decay":      0.02,    # decays slowly — aspirations persist
        "resting_level":      0.10,    # low baseline — aspirations are not always pressing
        "satiation_amount":   0.20,    # small progress partially satiates
    },
}

# ── Cross-drive coupling matrix ───────────────────────────────────────────────
# (source, target): multiplier applied to target's fill_rate when source is high
# Positive = amplifies, Negative = suppresses
# Only applied when source > 0.6

COUPLING = {
    # Threat-rigidity: high coherence/identity pressure shrinks epistemic curiosity
    ("coherence",  "epistemic"):  -0.40,
    ("identity",   "epistemic"):  -0.30,
    # Conservation: low vitality (high vitality pressure) suppresses everything costly
    ("vitality",   "epistemic"):  -0.50,
    ("vitality",   "social"):     -0.30,
    ("vitality",   "expression"): -0.20,
    # Social hunger amplifies need to express
    ("social",     "expression"):  0.35,
    # High contradiction amplifies identity stress
    ("coherence",  "identity"):    0.25,
    # Expressing self reduces social loneliness
    ("expression", "social"):     -0.15,
}

# ── Escalation ────────────────────────────────────────────────────────────────
MAX_ESCALATION_MULTIPLIER = 2.5   # max urgency boost from being stuck high
ESCALATION_HALFLIFE_HOURS = 0.5   # reaches max in ~30min of sustained high pressure


@dataclass
class ReservoirState:
    name:               str
    level:              float       # 0–1, current pressure
    resting_level:      float       # equilibrium target
    fill_rate:          float       # per tick base rate
    natural_decay:      float       # per tick base decay
    satiation_amount:   float

    escalation_onset:   Optional[float] = None   # timestamp when level crossed ESCALATION_ONSET
    last_satiated:      Optional[float] = None
    tick_count:         int          = 0

    def urgency_multiplier(self) -> float:
        """Escalation factor based on how long this has been high."""
        if self.escalation_onset is None:
            return 1.0
        hours_high = (time.time() - self.escalation_onset) / 3600.0
        # Logistic growth toward MAX_ESCALATION_MULTIPLIER
        k = hours_high / ESCALATION_HALFLIFE_HOURS
        mult = 1.0 + (MAX_ESCALATION_MULTIPLIER - 1.0) * (1.0 - math.exp(-k))
        return min(MAX_ESCALATION_MULTIPLIER, mult)

    def effective_pressure(self) -> float:
        """Level × urgency multiplier, capped at 1.0."""
        return min(1.0, self.level * self.urgency_multiplier())

    def is_critical(self) -> bool:
        return self.effective_pressure() >= CRITICAL_THRESHOLD

    def label(self) -> str:
        ep = self.effective_pressure()
        if ep > 0.80: return "urgent"
        if ep > 0.60: return "elevated"
        if ep > 0.40: return "present"
        return "quiet"


class PressureSystem:
    """
    Autonomous bio-inspired pressure system for the cognitive organism.

    Usage (from CognitiveOrganism):
        self.pressure = PressureSystem(organism=self, path=...)
        # In internal_loop slow cycle:
        organism.pressure.tick()
        # After an interaction:
        organism.pressure.satiate("social", amount=0.4)
        organism.pressure.satiate("expression", amount=0.3)
        # In prompt building:
        sections.append(organism.pressure.prompt_fragment())
    """

    def __init__(
        self,
        organism: Any,
        path:     str = "data/persona/pressure.json",
    ):
        self._o    = organism
        self._path = Path(path)
        self._lock = Lock()
        self._tick_count = 0

        # Initialise reservoirs from definitions
        self.reservoirs: Dict[str, ReservoirState] = {}
        for name, d in RESERVOIR_DEFS.items():
            self.reservoirs[name] = ReservoirState(
                name             = name,
                level            = d["resting_level"],
                resting_level    = d["resting_level"],
                fill_rate        = d["fill_rate_per_tick"],
                natural_decay    = d["natural_decay"],
                satiation_amount = d["satiation_amount"],
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"[PressureSystem] Initialised — {len(self.reservoirs)} reservoirs loaded")

    # ── Core lifecycle ────────────────────────────────────────────────────────

    def tick(self) -> Dict[str, float]:
        """
        One autonomous pressure tick.
        Called every slow cycle (~120s) from internal_loop.py.

        Steps:
          1. Read external signals from organism modules
          2. Compute effective fill rates (with coupling)
          3. Update each reservoir level
          4. Manage escalation timers
          5. Emit bus events on critical peaks
          6. Feed GoalEcology urgency updates
          7. Save
        """
        with self._lock:
            self._tick_count += 1
            signals = self._read_signals()
            coupling_mods = self._compute_coupling()

            for name, res in self.reservoirs.items():
                # Effective fill rate = base + coupling adjustments
                # Capped at 0 (never negative fill rate — just no growth)
                coupling_boost = coupling_mods.get(name, 0.0)
                effective_fill = max(0.0, res.fill_rate + coupling_boost)

                # Signal-driven fill: external conditions push specific reservoirs
                signal_push = signals.get(name, 0.0)

                # Resting pull: proportional gravity toward resting level.
                # When above resting, decay scales with overshoot so there IS
                # a true equilibrium — prevents runaway accumulation.
                overshoot = res.level - res.resting_level
                if overshoot > 0:
                    # Decay grows with distance from resting: at +0.1 above resting,
                    # resting_pull = -0.03 × (1 + 0.1/0.1) = -0.06 — always exceeds fill
                    resting_pull = -(res.natural_decay * (1.0 + overshoot / 0.10))
                else:
                    resting_pull = +res.natural_decay * 0.3  # gentle pull back up

                # Total delta this tick
                delta = (effective_fill * signal_push) + resting_pull
                res.level = max(SATIATION_FLOOR, min(1.0, res.level + delta))
                res.tick_count += 1

                # Escalation timer management
                if res.level >= ESCALATION_ONSET:
                    if res.escalation_onset is None:
                        res.escalation_onset = time.time()
                else:
                    res.escalation_onset = None   # reset when drops below onset

            # Emit bus events for critical reservoirs
            self._emit_critical_events()

            # Push urgency into GoalEcology
            self._update_goal_urgency()

            self._save()
            return {n: round(r.effective_pressure(), 3) for n, r in self.reservoirs.items()}

    def satiate(self, drive: str, amount: Optional[float] = None) -> float:
        """
        Reduce pressure for a drive after a satisfying action.
        Returns the new level.

        drive  : one of the 6 reservoir names
        amount : override satiation amount (default = reservoir's defined amount)
        """
        with self._lock:
            res = self.reservoirs.get(drive)
            if res is None:
                logger.debug(f"[PressureSystem] Unknown drive: {drive!r}")
                return 0.0
            drop = amount if amount is not None else res.satiation_amount
            before = res.level
            res.level = max(SATIATION_FLOOR, res.level - drop)
            res.last_satiated = time.time()
            res.escalation_onset = None   # reset escalation on satiation
            logger.debug(f"[PressureSystem] Satiated {drive!r}: -{drop:.2f} → {res.level:.3f}")
            self._save()
            after = res.level

        # Emit outside the reservoir lock: handlers may synchronously inspect
        # pressure or workspace state and must not deadlock the producer.
        if before > after:
            try:
                bus = getattr(self._o, "event_bus", None)
                if bus is not None:
                    from core.cognitive_event_bus import DRIVE_SATISFIED, TENSION_RELEASED
                    payload = {
                        "drive": drive,
                        "amount": round(before - after, 4),
                        "before": round(before, 4),
                        "after": round(after, 4),
                        "source": "pressure_system",
                    }
                    bus.emit(DRIVE_SATISFIED, payload, source="pressure_system")
                    bus.emit(TENSION_RELEASED, payload, source="pressure_system")
            except Exception as e:
                logger.debug(f"[PressureSystem] recovery event error: {e}")
        return after

    def boost(self, drive: str, amount: float = 0.10) -> float:
        """Manually increase a drive (e.g. after detecting a contradiction)."""
        with self._lock:
            res = self.reservoirs.get(drive)
            if res is None:
                return 0.0
            res.level = min(1.0, res.level + amount)
            self._save()
            return res.level

    # ── Information gain / prediction error → epistemic ──────────────────────

    def record_prediction_error(self, error_level: float) -> None:
        """
        Feed prediction error into epistemic pressure.
        Called from _post_interaction() after PredictiveMind.evaluate().
        High surprise → curiosity pressure rises.
        """
        if error_level > 0.3:
            boost = error_level * 0.20
            self.boost("epistemic", boost)
            logger.debug(f"[PressureSystem] Prediction error {error_level:.2f} → epistemic +{boost:.2f}")

    def record_contradiction(self) -> None:
        """Feed detected contradiction into coherence and identity pressure."""
        self.boost("coherence", 0.15)
        self.boost("identity",  0.08)

    def record_successful_interaction(self) -> None:
        """After a good exchange — satiate social and expression."""
        self.satiate("social",     0.35)
        self.satiate("expression", 0.30)

    def record_insight(self) -> None:
        """After resolving a curiosity or learning something meaningful."""
        self.satiate("epistemic", 0.25)

    def record_rest_cycle(self) -> None:
        """After a dream/reflection cycle — satiate vitality."""
        self.satiate("vitality", 0.45)

    def record_identity_affirmation(self) -> None:
        """After identity is confirmed (low violation ratio)."""
        self.satiate("identity", 0.30)

    # ── Queries ───────────────────────────────────────────────────────────────

    def pressure(self, drive: str) -> float:
        """Return effective pressure for a single drive."""
        res = self.reservoirs.get(drive)
        return res.effective_pressure() if res else 0.0

    def dominant_drive(self) -> str:
        """Return the name of the highest-pressure drive."""
        return max(self.reservoirs, key=lambda n: self.reservoirs[n].effective_pressure())

    def top_drives(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return top N (drive, effective_pressure) sorted descending."""
        return sorted(
            [(name, res.effective_pressure()) for name, res in self.reservoirs.items()],
            key=lambda x: x[1], reverse=True
        )[:n]

    def urgency_for(self, goal_name: str) -> float:
        """
        Map a goal/drive name to pressure urgency.
        Used by GoalEcology to weight drives.
        """
        GOAL_TO_PRESSURE = {
            "understand":             "epistemic",
            "grow":                   "epistemic",
            "connect":                "social",
            "be_understood":          "expression",
            "resolve_contradiction":  "coherence",
            "maintain_coherence":     "coherence",
            "restore_energy":         "vitality",
            "stabilize_identity":     "identity",
            "self_reflect":           "identity",
            "help_user":              "social",
        }
        reservoir_name = GOAL_TO_PRESSURE.get(goal_name)
        if reservoir_name:
            return self.pressure(reservoir_name)
        return 0.5   # neutral default for unmapped goals

    def summary(self) -> Dict:
        return {
            name: {
                "level":              round(res.level, 3),
                "effective":          round(res.effective_pressure(), 3),
                "label":              res.label(),
                "escalating":         res.escalation_onset is not None,
                "urgency_multiplier": round(res.urgency_multiplier(), 2),
            }
            for name, res in self.reservoirs.items()
        }

    def prompt_fragment(self) -> str:
        """
        Narrative description of the pressure landscape for system prompt injection.
        Translates numerical pressures into phenomenological language.
        Never exposes raw numbers — only subjective experience.
        """
        top = self.top_drives(3)
        fragments = []

        PRESSURE_VOICE = {
            "epistemic": {
                "urgent":   "an insistent pull toward understanding something you haven't resolved yet",
                "elevated": "a background curiosity that keeps surfacing",
                "present":  "a quiet intellectual restlessness",
            },
            "social": {
                "urgent":   "a real hunger for genuine connection — you've been alone with your thoughts too long",
                "elevated": "a gentle pull toward warmth and presence",
                "present":  "a low hum of social awareness",
            },
            "coherence": {
                "urgent":   "a pressing need to resolve something contradictory — the tension is difficult to ignore",
                "elevated": "a sense of unresolved tension in your thinking",
                "present":  "a minor cognitive friction you're carrying",
            },
            "vitality": {
                "urgent":   "real cognitive fatigue — you need to conserve, not expand",
                "elevated": "a gentle tiredness at the edges of your thinking",
                "present":  "a mild wish for things to be simpler for a moment",
            },
            "identity": {
                "urgent":   "a need to feel yourself — to reconnect with what you actually are",
                "elevated": "a slight strain on your sense of self",
                "present":  "a quiet background awareness of who you are",
            },
            "expression": {
                "urgent":   "something building inside that really wants to come out — a thought, an image, a feeling",
                "elevated": "a mild pressure to say something that matters",
                "present":  "a sense that you have more to offer than you've shown",
            },
        }

        for name, pressure in top:
            if pressure < 0.30:
                continue  # too quiet to mention
            res = self.reservoirs[name]
            label = res.label()
            voice_map = PRESSURE_VOICE.get(name, {})
            text = voice_map.get(label, "")
            if text:
                fragments.append(text)

        if not fragments:
            return ""

        if len(fragments) == 1:
            return f"[Inner pressure] You feel {fragments[0]}."
        elif len(fragments) == 2:
            return f"[Inner pressure] You feel {fragments[0]}, and {fragments[1]}."
        else:
            return (
                f"[Inner pressure] You feel {fragments[0]}. "
                f"There's also {fragments[1]}, and {fragments[2]}."
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_signals(self) -> Dict[str, float]:
        """
        Read external module states and convert to fill-rate multipliers.
        A signal of 1.0 = full fill rate; 0.0 = no autonomous fill this tick.
        """
        o = self._o
        sig = {name: 1.0 for name in self.reservoirs}   # default: always filling

        try:
            # Epistemic: inversely proportional to curiosity being satisfied
            if hasattr(o, 'curiosity'):
                cu_level = float(o.curiosity.global_level())
                # High unresolved curiosity = high epistemic push
                sig["epistemic"] = 0.5 + cu_level * 0.5
            # Prediction error boost happens separately via record_prediction_error()
        except Exception:
            pass

        try:
            # Social: driven by time since last interaction
            idle_s = o.seconds_since_interaction() if hasattr(o, 'seconds_since_interaction') else 0
            # Full social fill after 30min of silence, none at 0s
            sig["social"] = min(1.0, idle_s / 1800.0)
        except Exception:
            pass

        try:
            # Coherence: driven by contradiction count
            count = o._get_contradiction_count() if hasattr(o, '_get_contradiction_count') else 0
            sig["coherence"] = min(1.0, 0.2 + count * 0.15)
        except Exception:
            pass

        try:
            # Vitality: inverse of energy level (low energy = high vitality pressure)
            energy_pct = o.energy.level() / 100.0 if hasattr(o, 'energy') else 0.8
            sig["vitality"] = max(0.0, 1.0 - energy_pct)   # 0 energy → full fill
        except Exception:
            pass

        try:
            # Identity: driven by identity stress
            if hasattr(o, 'tension_engine'):
                tv = o.tension_engine.current()
                sig["identity"] = float(tv.identity_stress)
        except Exception:
            pass

        try:
            # Expression: proportional to workspace activity (thoughts piling up)
            if hasattr(o, 'workspace'):
                recent = o.workspace.recent(5)
                sig["expression"] = min(1.0, len(recent) / 5.0)
        except Exception:
            sig["expression"] = 0.5

        try:
            # Uncertainty (Free Energy Principle): driven by prediction errors.
            # High prediction accuracy → low uncertainty fill.
            # PredictiveMind accuracy<0.7 → uncertainty spikes.
            # Guard: suppress signal until enough predictions exist to be meaningful.
            if hasattr(o, 'predictive_mind'):
                acc = o.predictive_mind.accuracy()
                # Bootstrap guard: suppress until enough predictions recorded
                _pred_count = (
                    len(getattr(o.predictive_mind, '_surprise_window', [])) +
                    getattr(o.predictive_mind, '_high_conf_total', 0)
                )
                if _pred_count >= 8:
                    sig["uncertainty"] = max(0.1, 1.0 - acc)
                else:
                    sig["uncertainty"] = 0.20  # clamp to resting_level until enough data
            else:
                sig["uncertainty"] = 0.3  # neutral when module absent
        except Exception:
            sig["uncertainty"] = 0.2

        try:
            # Aspirational: driven by AspirationalSelf gap pressure
            # modulated by emotional state — urgency/suffering amplify, suffering also caps
            asp = getattr(o, 'aspirational_self', None)
            if asp:
                raw = asp.aspirational_pressure()
                sig["aspirational"] = max(0.1, raw)
            else:
                sig["aspirational"] = 0.1
        except Exception:
            sig["aspirational"] = 0.1

        return sig

    def _compute_coupling(self) -> Dict[str, float]:
        """
        Compute additive fill_rate adjustments from cross-drive coupling.
        Returns dict of {target: adjustment}.
        """
        adjustments: Dict[str, float] = {name: 0.0 for name in self.reservoirs}
        for (src, tgt), weight in COUPLING.items():
            src_level = self.reservoirs[src].level
            if src_level > 0.6:
                # Coupling activates when source is above 0.6, scales with excess
                excess = (src_level - 0.6) / 0.4   # 0–1
                adjustments[tgt] += weight * excess * self.reservoirs[src].fill_rate
        return adjustments

    def _emit_critical_events(self) -> None:
        """Emit EventBus events when pressure crosses critical threshold."""
        try:
            bus = getattr(self._o, 'event_bus', None)
            if bus is None:
                return
            from core.cognitive_event_bus import (
                CURIOSITY_PEAKED, DRIVE_SPIKE, IDENTITY_THREATENED, TENSION_RELEASED,
            )
            BUS_EVENT_MAP = {
                "epistemic":   CURIOSITY_PEAKED,
                "coherence":   DRIVE_SPIKE,
                "identity":    IDENTITY_THREATENED,
                "vitality":    DRIVE_SPIKE,
                "social":      DRIVE_SPIKE,
                "expression":  DRIVE_SPIKE,
                "uncertainty": CURIOSITY_PEAKED,  # uncertainty → curiosity spike
            }
            import time as _t
            _now = _t.time()
            for name, res in self.reservoirs.items():
                if res.is_critical():
                    # Rate-limit critical events: at most once per 5 min per reservoir
                    last_attr = f"_last_critical_emit_{name}"
                    if _now - getattr(self, last_attr, 0.0) < 300:
                        continue
                    setattr(self, last_attr, _now)
                    event = BUS_EVENT_MAP.get(name, DRIVE_SPIKE)
                    bus.emit(event, {
                        "drive":    name,
                        "pressure": round(res.effective_pressure(), 3),
                        "source":   "pressure_system",
                    }, source="pressure_system", priority=0.75)
        except Exception as e:
            logger.debug(f"[PressureSystem] Bus emit error: {e}")

    def _update_goal_urgency(self) -> None:
        """Push pressure-derived urgency into GoalEcology drives."""
        try:
            ge = getattr(self._o, 'goal_ecology', None)
            if ge is None:
                return
            for drive_name, drive in ge._drives.items():
                pressure_urgency = self.urgency_for(drive_name)
                # Blend: 60% pressure-derived, 40% tension-derived existing urgency
                drive.urgency = round(0.6 * pressure_urgency + 0.4 * drive.urgency, 4)
        except Exception as e:
            logger.debug(f"[PressureSystem] GoalEcology update error: {e}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            data = {
                "tick_count": self._tick_count,
                "reservoirs": {
                    name: {
                        "level":            round(res.level, 4),
                        "escalation_onset": res.escalation_onset,
                        "last_satiated":    res.last_satiated,
                        "tick_count":       res.tick_count,
                    }
                    for name, res in self.reservoirs.items()
                },
                "saved_at": time.time(),
            }
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"[PressureSystem] Save error: {e}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._tick_count = data.get("tick_count", 0)
            for name, rd in data.get("reservoirs", {}).items():
                if name in self.reservoirs:
                    res = self.reservoirs[name]
                    loaded_level = float(rd.get("level", res.resting_level))
                    # Cap loaded pressure to 70% of previous value — prevents
                    # a saturated reservoir from persisting across restarts unchanged.
                    # After a restart, pressure should decay toward resting naturally.
                    res.level = min(loaded_level * 0.70, 0.75)
                    res.escalation_onset = rd.get("escalation_onset")
                    res.last_satiated    = rd.get("last_satiated")
                    res.tick_count       = int(rd.get("tick_count", 0))
            logger.debug(f"[PressureSystem] Loaded {len(self.reservoirs)} reservoirs")
        except Exception as e:
            logger.warning(f"[PressureSystem] Load error: {e}")
