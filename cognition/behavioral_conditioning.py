"""
Behavioral Conditioning — Pass 2
==================================
Memories that actively constrain or bias future behavior.
When Lumina has repeatedly had bad outcomes with a specific
interaction pattern, a conditioning signal is stored.
Future interactions matching that pattern trigger a caution
signal that affects both the response and the evolution engine.

This is not just context injection — it modifies how the response
is generated (temperature, added caution phrasing) and triggers
a "behavioral_conditioning_hit" experience in the evolution engine,
ensuring the trait-level system learns from the pattern too.
"""
import json, logging, time, re, threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConditioningSignal:
    pattern:       str     # text pattern that triggers this signal
    signal_type:   str     # "caution" | "approach" | "avoid"
    strength:      float   # 0–1, decays over time
    formed_at:     float   = field(default_factory=time.time)
    last_triggered:float   = field(default_factory=time.time)
    trigger_count: int     = 0
    source:        str     = ""    # what event formed this conditioning


class BehavioralConditioningSystem:
    """
    Tracks patterns that have led to negative outcomes and ensures
    they influence future responses in a visceral way — not just
    as text in the prompt, but as a signal that changes how the
    response is generated.

    Positive conditioning (approach signals) can also be tracked.
    """
    MAX_SIGNALS      = 50
    DECAY_HALFLIFE_H = 72   # signals fade over 3 days without reinforcement

    def __init__(self, path: str = "conditioning.json", evolution_engine=None):
        self._path    = Path(path)
        self._lock    = threading.RLock()
        self._evo     = evolution_engine
        self._signals: List[ConditioningSignal] = []
        self._load()

    # ── Public ─────────────────────────────────────────────────────────────

    def add_signal(self, pattern: str, signal_type: str,
                   strength: float = 0.6, source: str = "") -> None:
        """
        Add or reinforce a conditioning signal.
        If a similar pattern exists, reinforce it instead of adding a new one.
        """
        with self._lock:
            for sig in self._signals:
                if self._patterns_similar(sig.pattern, pattern):
                    sig.strength = min(1.0, sig.strength + strength * 0.3)
                    sig.source   = source or sig.source
                    self._save()
                    return

            self._signals.append(ConditioningSignal(
                pattern=pattern[:200], signal_type=signal_type,
                strength=strength, source=source,
            ))
            # Trim to max
            if len(self._signals) > self.MAX_SIGNALS:
                self._signals.sort(key=lambda s: s.strength, reverse=True)
                self._signals = self._signals[:self.MAX_SIGNALS]
            self._save()

    def check_input(self, text: str) -> Dict[str, Any]:
        """
        Check an incoming message against all conditioning signals.
        Returns a dict with:
          - triggered: List of triggered signals
          - caution_level: 0–1 aggregate caution
          - approach_level: 0–1 aggregate approach
          - prompt_note: text to inject into system prompt (or empty)
        """
        with self._lock:
            now = time.time()
            triggered   = []
            caution_sum = 0.0
            approach_sum= 0.0

            for sig in self._signals:
                # Apply time decay
                elapsed_h = (now - sig.last_triggered) / 3600
                decayed_strength = sig.strength * (0.5 ** (elapsed_h / self.DECAY_HALFLIFE_H))
                if decayed_strength < 0.05:
                    continue
                if self._text_matches(text, sig.pattern):
                    triggered.append(sig)
                    sig.trigger_count  += 1
                    sig.last_triggered  = now
                    if sig.signal_type == "caution":
                        caution_sum  += decayed_strength
                    elif sig.signal_type == "approach":
                        approach_sum += decayed_strength

            if triggered and self._evo:
                self._evo.queue_experience("behavioral_conditioning_hit",
                                           intensity=min(1.5, caution_sum))
                self._save()

            # Build prompt note
            prompt_note = ""
            if caution_sum > 0.3:
                prompt_note = (f"⚠ Past experience suggests caution in this context "
                               f"(conditioning strength: {caution_sum:.2f}). "
                               f"Be especially thoughtful in your response.")
            elif approach_sum > 0.3:
                prompt_note = (f"✓ This is a context where you've done well before "
                               f"(approach signal: {approach_sum:.2f}).")

            return {
                "triggered":     [s.pattern for s in triggered],
                "caution_level": min(1.0, caution_sum),
                "approach_level":min(1.0, approach_sum),
                "prompt_note":   prompt_note,
                "temp_modifier": -0.08 * min(1.0, caution_sum),
            }

    def record_outcome(self, context: str, positive: bool, strength: float = 0.5) -> None:
        """
        Record the outcome of an interaction.
        Negative outcomes create or reinforce caution signals.
        Positive outcomes create or reinforce approach signals.
        """
        keywords = self._extract_keywords(context)
        if not keywords:
            return
        pattern = " ".join(keywords[:4])
        sig_type = "approach" if positive else "caution"
        self.add_signal(pattern, sig_type, strength=strength,
                       source=f"{'positive' if positive else 'negative'} outcome")

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            active = [s for s in self._signals if s.strength > 0.1]
            return {
                "total_signals": len(self._signals),
                "active_signals": len(active),
                "caution_patterns": [s.pattern for s in active if s.signal_type=="caution"][:5],
                "approach_patterns":[s.pattern for s in active if s.signal_type=="approach"][:5],
            }

    # ── Internals ───────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> List[str]:
        stopwords = {"i","you","the","a","an","is","are","was","were","be","been",
                     "have","has","had","do","does","did","will","would","could","should",
                     "this","that","these","those","and","or","but","for","with","about"}
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stopwords][:8]

    def _patterns_similar(self, p1: str, p2: str) -> bool:
        w1 = set(p1.lower().split())
        w2 = set(p2.lower().split())
        if not w1 or not w2: return False
        overlap = len(w1 & w2) / min(len(w1), len(w2))
        return overlap > 0.5

    def _text_matches(self, text: str, pattern: str) -> bool:
        text_lower    = text.lower()
        pattern_words = pattern.lower().split()
        matches = sum(1 for w in pattern_words if w in text_lower)
        return matches >= max(1, len(pattern_words) * 0.5)

    def _save(self):
        try:
            self._path.write_text(json.dumps(
                [asdict(s) for s in self._signals], indent=2))
        except Exception as e:
            logger.error(f"Conditioning save: {e}")

    def _load(self):
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            for sd in data:
                self._signals.append(ConditioningSignal(**{
                    k: v for k, v in sd.items()
                    if k in ConditioningSignal.__dataclass_fields__
                }))
            logger.info(f"Conditioning: {len(self._signals)} signals loaded")
        except Exception as e:
            logger.error(f"Conditioning load: {e}")
