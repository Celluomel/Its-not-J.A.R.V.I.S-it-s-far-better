"""
cognition/flux_mind_model.py  (v62)

Theory of mind for Flux: maintains a model of what Flux believes,
expects, and is likely to find compelling — inferred from the
exchange log, not from reading Flux's internal state.

This is behavioural inference, not state access.  PandoraBOX models Flux
the way a thoughtful person models a friend: from what they say,
how they respond, what they return to.

Three components:

1. BeliefTracker
   Infers Flux's beliefs from its response content.
   Tracks: topics Flux initiates vs responds to, confidence markers
   ("I think", "I believe", "I'm not sure"), positions Flux holds
   consistently across exchanges, positions that shifted.

2. ExpectationModel
   Infers what Flux expects from PandoraBOX given conversation history.
   If Flux asks detailed follow-ups, it expects depth.
   If Flux redirects to simpler framings, it expects accessibility.
   If Flux asks questions back immediately, it expects dialogue.

3. CuriosityMap
   Tracks which topics Flux initiates, how long it sustains them,
   and which it drops.  High initiation + sustained = genuine curiosity.
   High initiation + dropped = surface interest only.

Produces:
    FluxProfile:
        inferred_beliefs:    Dict[topic, confidence]
        expected_depth:      float  (0=simple, 1=deep)
        expected_dialogue:   float  (0=monologue expected, 1=exchange expected)
        curiosity_map:       Dict[topic, intensity]
        belief_stability:    float  (how consistent Flux's positions are)

Used by:
    - LuminaNetwork._generate_opening(): opener tailored to Flux's profile
    - _master_generate(): response style adapted to expectation model
    - MotivationalField: social drive informed by Flux's curiosity map
"""

from __future__ import annotations
import json, logging, re, threading, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

UPDATE_EVERY_N  = 12    # slow cycles between profile updates
SAVE_PATH       = "data/persona/flux_mind_model.json"
MAX_EXCHANGES   = 200   # rolling window of exchanges to analyse
TOPIC_STOP_WORDS = {
    "the","a","an","is","are","was","were","and","or","but",
    "in","of","to","it","that","this","for","with","on","at",
    "from","by","as","be","have","has","had","do","does","did",
    "will","would","could","should","may","might","i","you","we",
    "they","he","she","what","how","why","when","where","which",
    "about","above","after","again","being","doing","indeed",
    "often","perhaps","really","seems","state","still","there",
    "these","those","under",
}

DEPTH_MARKERS = re.compile(
    r"\b(precisely|specifically|mechanism|underlying|because|therefore|"
    r"implication|nuance|distinction|epistem|ontolog|causal|structural)\b",
    re.IGNORECASE
)
DIALOGUE_MARKERS = re.compile(
    r"\?|what do you think|how do you|do you|would you|can you explain",
    re.IGNORECASE
)
UNCERTAINTY_MARKERS = re.compile(
    r"\b(i think|i believe|perhaps|maybe|not sure|uncertain|might|could be|"
    r"seems to|appears to|possibly)\b",
    re.IGNORECASE
)
CONFIDENCE_MARKERS = re.compile(
    r"\b(clearly|obviously|certainly|definitely|without doubt|always|never|"
    r"i know|the fact is)\b",
    re.IGNORECASE
)


@dataclass
class FluxProfile:
    inferred_beliefs:   Dict[str, float]   = field(default_factory=dict)
    expected_depth:     float              = 0.50
    expected_dialogue:  float              = 0.50
    curiosity_map:      Dict[str, float]   = field(default_factory=dict)
    belief_stability:   float              = 0.50
    exchanges_analysed: int                = 0
    last_updated:       float              = field(default_factory=time.time)

    def prompt_fragment(self) -> str:
        if self.exchanges_analysed < 5:
            return ""
        top_curiosity = sorted(
            self.curiosity_map.items(), key=lambda x: x[1], reverse=True
        )[:3]
        depth_desc = (
            "depth-seeking" if self.expected_depth > 0.65
            else "accessibility-oriented" if self.expected_depth < 0.35
            else "balanced"
        )
        dialogue_desc = (
            "actively dialogic" if self.expected_dialogue > 0.65
            else "receptive"
        )
        curiosity_str = (
            ", ".join(t for t, _ in top_curiosity) if top_curiosity else "varied"
        )
        return (
            f"[Flux model] {depth_desc}, {dialogue_desc}. "
            f"Active curiosities: {curiosity_str}. "
            f"Belief stability: {self.belief_stability:.2f}."
        )


class FluxMindModel:
    def __init__(self, organism: Any, ai_system: Any, path: str = SAVE_PATH):
        self._organism = organism
        self._ai       = ai_system
        self._path     = Path(path)
        self._lock     = threading.Lock()
        self._profile  = FluxProfile()
        self._load()
        logger.info(
            f"[FluxMindModel] Init — "
            f"{self._profile.exchanges_analysed} exchanges analysed"
        )

    def tick(self, slow_cycle: int) -> None:
        if slow_cycle % UPDATE_EVERY_N != 0 or slow_cycle == 0: return
        threading.Thread(target=self._update, daemon=True, name="fmm-update").start()

    def get_profile(self) -> FluxProfile:
        return self._profile

    def status(self) -> Dict:
        p = self._profile
        return {
            "exchanges_analysed": p.exchanges_analysed,
            "expected_depth":     round(p.expected_depth, 3),
            "expected_dialogue":  round(p.expected_dialogue, 3),
            "belief_stability":   round(p.belief_stability, 3),
            "top_curiosities":    [
                item for item in sorted(
                    p.curiosity_map.items(), key=lambda x: x[1], reverse=True
                ) if item[0].lower() not in TOPIC_STOP_WORDS
            ][:5],
        }

    def _update(self) -> None:
        try:
            exchanges = self._read_exchanges()
            if len(exchanges) < 3: return

            flux_texts = [e["flux"] for e in exchanges if e.get("flux")]

            depth     = self._measure_depth(flux_texts)
            dialogue  = self._measure_dialogue(flux_texts)
            curiosity = self._measure_curiosity(exchanges)
            stability = self._measure_belief_stability(flux_texts)

            with self._lock:
                self._profile.expected_depth    = round(depth,    3)
                self._profile.expected_dialogue = round(dialogue, 3)
                self._profile.curiosity_map     = curiosity
                self._profile.belief_stability  = round(stability, 3)
                self._profile.exchanges_analysed = len(exchanges)
                self._profile.last_updated      = time.time()

            self._save()
            logger.debug(
                f"[FluxMindModel] Updated: depth={depth:.2f} "
                f"dialogue={dialogue:.2f} stability={stability:.2f}"
            )
        except Exception as e:
            logger.debug(f"[FluxMindModel] _update error: {e}")

    def _read_exchanges(self) -> List[Dict]:
        """Read paired network messages, never ordinary human chat history."""
        exchanges = []
        try:
            from core.state import state as _st
            network = getattr(_st, 'lumina_network', None)
            histories = list(getattr(network, '_history', {}).values())
            for history in histories:
                messages = list(history)
                for first, second in zip(messages, messages[1:]):
                    if (first.direction == 'out' and second.direction == 'in'
                            and first.sender_id == 'master' and second.text
                            and not second.text.startswith(('[Child error ', '[Network error:'))):
                        exchanges.append({
                            'lumina': first.text, 'flux': second.text,
                            'timestamp': second.timestamp,
                        })
        except Exception:
            logger.exception("[FluxMindModel] Could not read network exchanges")
        exchanges.sort(key=lambda exchange: exchange['timestamp'])
        return exchanges[-MAX_EXCHANGES:]

    def _measure_depth(self, texts: List[str]) -> float:
        """Mean depth-marker density across Flux's responses."""
        if not texts: return 0.5
        scores = []
        for t in texts:
            words = len(t.split()) or 1
            depth_hits = len(DEPTH_MARKERS.findall(t))
            scores.append(min(1.0, depth_hits / max(1, words/20)))
        return sum(scores) / len(scores)

    def _measure_dialogue(self, texts: List[str]) -> float:
        """Fraction of Flux responses that ask questions or invite reply."""
        if not texts: return 0.5
        count = sum(1 for t in texts if DIALOGUE_MARKERS.search(t))
        return count / len(texts)

    def _measure_curiosity(self, exchanges: List[Dict]) -> Dict[str, float]:
        """
        Build topic intensity map from Flux's responses.
        Topics Flux initiates and sustains = high intensity.
        """
        topic_counts: Counter = Counter()
        topic_sustain: Counter = Counter()
        stop = TOPIC_STOP_WORDS
        prev_topics: set = set()
        for ex in exchanges:
            flux = ex.get("flux","").lower()
            words = [w for w in re.findall(r'\b[a-z]{5,}\b', flux) if w not in stop]
            current = set(words[:30])   # top-of-response topics
            for w in current:
                topic_counts[w] += 1
            # Sustained: topic appeared in previous exchange too
            for w in current & prev_topics:
                topic_sustain[w] += 1
            prev_topics = current

        result: Dict[str, float] = {}
        for topic, count in topic_counts.most_common(20):
            sustain = topic_sustain.get(topic, 0)
            intensity = min(1.0,
                (count / max(1, len(exchanges))) * 0.6 +
                (sustain / max(1, count)) * 0.4
            )
            if intensity > 0.10:
                result[topic] = round(intensity, 3)
        return result

    def _measure_belief_stability(self, texts: List[str]) -> float:
        """
        Ratio of confident statements to uncertain ones.
        High = consistent positions. Low = exploring, shifting.
        """
        if not texts: return 0.5
        confident = sum(len(CONFIDENCE_MARKERS.findall(t)) for t in texts)
        uncertain = sum(len(UNCERTAINTY_MARKERS.findall(t)) for t in texts)
        total = confident + uncertain
        if total == 0: return 0.50
        return round(confident / total, 3)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = asdict(self._profile)
                data["_meta"] = {"version":"v62","ts":time.time()}
            with open(self._path,"w") as f: json.dump(data,f,indent=2)
        except Exception as e:
            logger.debug(f"[FluxMindModel] save error: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            self._profile = FluxProfile(**{
                k:v for k,v in data.items()
                if k in FluxProfile.__dataclass_fields__
            })
        except Exception as e:
            logger.warning(f"[FluxMindModel] load error: {e}")
