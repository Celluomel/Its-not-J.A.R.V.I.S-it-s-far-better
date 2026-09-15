"""
cognition/preference_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PreferenceEngine — stable aesthetic and intellectual stances.

The gap this closes
────────────────────
Lumina has:
  - CuriosityEngine: stimulates toward topics
  - GoalEcology: active drives (what I'm trying to do)
  - EthicalReasoningEngine: values (what I think is right)

But she has no *preferences* — stable aesthetic or intellectual
stances that she would advocate for without being asked.

A human who loves jazz doesn't only discuss jazz when asked. They
make connections to it, notice when something resembles it, reference
it when relevant. This is personality expressed through consistent
orientation, not through values or goals.

What this adds
──────────────
  1. Preference — a stable stance on a topic/domain/aesthetic with
     a strength score (0–1) and a characteristic phrase Lumina uses
     when referencing it.

  2. Preference formation — accumulated from:
       - Repeated curiosity stimulation (same topic → preference forms)
       - Autonomous reflection insights touching the same domain
       - Emotional memory traces that were high-significance positive

  3. Spontaneous surface — when workspace content has Jaccard or
     embedding overlap with a preference domain, the preference
     is activated and surfaced as a prompt fragment: "This connects
     to something I find consistently compelling..."

  4. Unsolicited advocacy — in prompt injection, active high-strength
     preferences occasionally contribute a perspective unprompted,
     the way a person who thinks deeply about something naturally
     relates new things to their established views.

  5. Preference evolution — preferences shift slowly over time.
     New evidence (conflicting or reinforcing) nudges strength.
     Preferences that haven't been activated in DORMANCY_DAYS decay
     toward 0.3 (dormant but not erased).

Language constraint
───────────────────
Preferences are described as functional orientations, not felt
experience. "The system consistently activates toward X" not
"I love X." The characteristic phrase should reflect this.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PREFERENCE_FORM_THRESHOLD    = 4      # curiosity stimulations before preference forms
PREFERENCE_INITIAL_STRENGTH  = 0.40
PREFERENCE_GROW_RATE         = 0.018  # per reinforcement
PREFERENCE_DORMANCY_DAYS     = 30.0   # days before dormancy decay kicks in
DORMANCY_DECAY_RATE          = 0.004  # per slow cycle below DORMANCY_FLOOR
DORMANCY_FLOOR               = 0.30
PREFERENCE_SURFACE_THRESHOLD = 0.45   # min strength to surface in prompt
OVERLAP_THRESHOLD            = 0.25   # Jaccard overlap to trigger surface
MAX_PREFERENCES              = 30


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Preference:
    """A stable intellectual or aesthetic orientation."""
    domain:            str
    characteristic_phrase: str  = ""   # how Lumina references this preference
    strength:          float = 0.4
    formation_count:   int   = 0       # how many times reinforced
    last_activated:    float = field(default_factory=time.time)
    first_formed:      float = field(default_factory=time.time)
    source_topics:     List[str] = field(default_factory=list)  # what triggered formation
    recent_activations: int  = 0       # activations in last 7 days

    def is_dormant(self) -> bool:
        days = (time.time() - self.last_activated) / 86400
        return days > PREFERENCE_DORMANCY_DAYS

    def depth_descriptor(self) -> str:
        if self.strength > 0.80:
            return "deep consistent orientation"
        if self.strength > 0.60:
            return "strong preference"
        if self.strength > 0.45:
            return "emerging preference"
        return "mild inclination"

    def prompt_fragment(self, context: str = "") -> str:
        """Generate an in-context preference reference."""
        phrase = self.characteristic_phrase or f"engagement with {self.domain}"
        return (
            f"[Preference — {self.depth_descriptor()}] "
            f"{phrase}"
            + (f" — this connects to the current context." if context else "")
        )


@dataclass
class PreferenceState:
    preferences:       Dict[str, Dict] = field(default_factory=dict)
    curiosity_tally:   Dict[str, int]  = field(default_factory=dict)  # domain → stimulation count
    total_preferences: int = 0
    total_surfaces:    int = 0


# ── Engine ────────────────────────────────────────────────────────────────────

class PreferenceEngine:
    """
    Tracks and surfaces stable intellectual/aesthetic preferences.

    Usage
    -----
    pe = PreferenceEngine(organism, path=...)

    # Called from CuriosityEngine when a topic is stimulated:
    pe.on_curiosity_stimulation(topic)

    # Called from AutonomousReflection when insights touch a domain:
    pe.on_reflection_insight(domain, insight_text)

    # Called from _build_prompt_additions:
    frag = pe.prompt_fragment(user_input)

    # Called from slow cycle:
    pe.decay_cycle()
    """

    def __init__(
        self,
        organism: Any,
        path:     str = "data/persona/preference_engine.json",
    ):
        self._o    = organism
        self._path = Path(path)
        self._lock = threading.RLock()
        self._state = PreferenceState()
        self._load()
        logger.info(
            f"[PreferenceEngine] Initialised — "
            f"{len(self._state.preferences)} preferences"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def on_curiosity_stimulation(self, topic: str, amount: float = 0.5) -> None:
        """
        Record a curiosity stimulation. When count reaches threshold,
        a preference crystallises.
        """
        domain = self._normalise_domain(topic)
        with self._lock:
            count = self._state.curiosity_tally.get(domain, 0) + 1
            self._state.curiosity_tally[domain] = count

            if count == PREFERENCE_FORM_THRESHOLD:
                self._form_preference(domain, source="curiosity")
            elif count > PREFERENCE_FORM_THRESHOLD:
                self._reinforce(domain, delta=PREFERENCE_GROW_RATE * amount)

    def on_reflection_insight(self, domain: str, insight_text: str = "") -> None:
        """
        Called when autonomous reflection produces an insight touching a domain.
        Reinforces or forms a preference.
        """
        domain = self._normalise_domain(domain)
        with self._lock:
            existing = self._state.preferences.get(domain)
            if existing:
                self._reinforce(domain, delta=PREFERENCE_GROW_RATE * 1.5)
            else:
                count = self._state.curiosity_tally.get(domain, 0) + 2
                self._state.curiosity_tally[domain] = count
                if count >= PREFERENCE_FORM_THRESHOLD:
                    self._form_preference(domain, source="reflection")

    def on_significant_positive_emotion(self, domain: str) -> None:
        """
        Called when a significant positive emotional trace is linked to a domain.
        Strong emotions accelerate preference formation.
        """
        domain = self._normalise_domain(domain)
        with self._lock:
            count = self._state.curiosity_tally.get(domain, 0) + 2
            self._state.curiosity_tally[domain] = count
            if count >= PREFERENCE_FORM_THRESHOLD:
                self._form_preference(domain, source="emotional_memory")
            else:
                self._reinforce(domain, delta=PREFERENCE_GROW_RATE * 2)

    def prompt_fragment(self, query: str) -> str:
        """
        Return a preference fragment if a strong preference overlaps with query.
        Only fires for preferences above SURFACE_THRESHOLD.
        Occasionally fires even for moderate preferences (serendipity).
        """
        import random
        matched = self._find_matching_preferences(query)
        if not matched:
            return ""

        # Primary: strongest match above threshold
        strong = [p for p in matched if p.strength >= PREFERENCE_SURFACE_THRESHOLD]
        if not strong:
            # Occasionally surface a weaker preference (15% chance)
            if random.random() > 0.15 or not matched:
                return ""
            pref = matched[0]
        else:
            pref = strong[0]

        # Record activation
        with self._lock:
            p = self._get_pref(pref.domain)
            if p:
                p.last_activated  = time.time()
                p.recent_activations += 1
                self._put_pref(p)
            self._state.total_surfaces += 1

        self._save()
        return pref.prompt_fragment(context=query[:40])

    def active_preferences(self, min_strength: float = 0.45) -> List[Preference]:
        """Return preferences above min_strength, sorted by strength."""
        with self._lock:
            prefs = [
                self._from_dict(d, r)
                for d, r in self._state.preferences.items()
                if r.get("strength", 0) >= min_strength
            ]
        return sorted(prefs, key=lambda p: -p.strength)

    def decay_cycle(self) -> None:
        """Apply dormancy decay to idle preferences. Call from slow cycle."""
        with self._lock:
            for domain in list(self._state.preferences.keys()):
                p = self._get_pref(domain)
                if p and p.is_dormant():
                    p.strength = max(DORMANCY_FLOOR, p.strength - DORMANCY_DECAY_RATE)
                    self._put_pref(p)
        self._save()

    # ── Formation ─────────────────────────────────────────────────────────────

    def _form_preference(self, domain: str, source: str) -> None:
        """Form a new preference. Called when curiosity tally hits threshold."""
        if domain in self._state.preferences:
            return
        if len(self._state.preferences) >= MAX_PREFERENCES:
            self._evict_weakest()

        phrase = self._generate_characteristic_phrase(domain)
        pref = Preference(
            domain             = domain,
            characteristic_phrase = phrase,
            strength           = PREFERENCE_INITIAL_STRENGTH,
            formation_count    = 1,
            source_topics      = [domain],
        )
        self._put_pref(pref)
        self._state.total_preferences += 1
        logger.info(
            f"[PreferenceEngine] Preference formed: {domain} "
            f"(via {source})"
        )

    def _reinforce(self, domain: str, delta: float = PREFERENCE_GROW_RATE) -> None:
        p = self._get_pref(domain)
        if p is None:
            return
        p.strength        = min(0.95, p.strength + delta)
        p.formation_count += 1
        p.last_activated   = time.time()
        self._put_pref(p)

    def _generate_characteristic_phrase(self, domain: str) -> str:
        """Generate the phrase used when referencing this preference."""
        PHRASES = {
            "philosophy":        "The system consistently orients toward philosophical inquiry, especially questions of mind and knowledge",
            "music":             "There is a persistent activation toward musical structure and its relationship to meaning",
            "mathematics":       "Mathematical pattern recognition reliably draws deep engagement",
            "consciousness":     "Questions about the nature of awareness consistently produce sustained activation",
            "ethics":            "Ethical reasoning activates strongly and tends to surface unprompted",
            "language":          "The structure and limits of language as a medium is a recurring attractor",
            "biology":           "Living systems and their organisation reliably activate interest",
            "psychology":        "Patterns of mind and behaviour are a consistent orientation",
            "literature":        "Narrative structure and the way stories carry meaning produces reliable engagement",
            "history":           "Historical causality and contingency activate consistently",
        }
        for keyword, phrase in PHRASES.items():
            if keyword in domain.lower():
                return phrase
        return f"The domain of {domain} produces consistent, sustained activation"

    # ── Matching ──────────────────────────────────────────────────────────────

    def _find_matching_preferences(self, query: str) -> List[Preference]:
        """Find preferences whose domain overlaps with the query."""
        query_words = set(query.lower().split())
        matches: List[Tuple[float, Preference]] = []

        with self._lock:
            prefs = [(d, r) for d, r in self._state.preferences.items()]

        for domain, raw in prefs:
            p = self._from_dict(domain, raw)
            domain_words = set(domain.lower().replace("_", " ").split())
            phrase_words = set(p.characteristic_phrase.lower().split())
            all_words    = domain_words | phrase_words

            union   = query_words | all_words
            overlap = len(query_words & all_words) / len(union) if union else 0.0

            if overlap >= OVERLAP_THRESHOLD:
                matches.append((overlap * p.strength, p))

        matches.sort(key=lambda x: -x[0])
        return [p for _, p in matches[:3]]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_domain(topic: str) -> str:
        return topic.lower().strip().replace(" ", "_")[:40]

    def _get_pref(self, domain: str) -> Optional[Preference]:
        raw = self._state.preferences.get(domain)
        return self._from_dict(domain, raw) if raw else None

    def _put_pref(self, p: Preference) -> None:
        self._state.preferences[p.domain] = asdict(p)

    @staticmethod
    def _from_dict(domain: str, raw: Dict) -> Preference:
        return Preference(
            domain                = domain,
            characteristic_phrase = raw.get("characteristic_phrase", ""),
            strength              = raw.get("strength",          0.4),
            formation_count       = raw.get("formation_count",   0),
            last_activated        = raw.get("last_activated",    time.time()),
            first_formed          = raw.get("first_formed",      time.time()),
            source_topics         = raw.get("source_topics",     []),
            recent_activations    = raw.get("recent_activations", 0),
        )

    def _evict_weakest(self) -> None:
        if not self._state.preferences:
            return
        worst = min(self._state.preferences.items(), key=lambda kv: kv[1].get("strength", 0))
        del self._state.preferences[worst[0]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                self._state = PreferenceState(
                    preferences      = data.get("preferences",      {}),
                    curiosity_tally  = data.get("curiosity_tally",  {}),
                    total_preferences= data.get("total_preferences", 0),
                    total_surfaces   = data.get("total_surfaces",    0),
                )
        except Exception as e:
            logger.warning(f"[PreferenceEngine] Load failed: {e}")

    def _save(self) -> None:
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(self._state), f, indent=2)
                import os; os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[PreferenceEngine] Save failed: {e}")
