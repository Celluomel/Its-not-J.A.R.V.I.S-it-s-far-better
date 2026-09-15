"""
PredictiveMind — Predictive Processing Engine
==============================================
Inspired by Karl Friston's Predictive Processing theory.

FIX LOG
-------
2026-03-09  Bug: _detect_intent was identical to _predict_intent (tautology).
            Fix: _detect_intent_from_response reads the actual LLM output.

2026-03-09  Bug: record_novelty() was never called anywhere.
            Fix: Wired in _update_topic_registry() on every evaluate().

2026-03-09  Bug: novelty formula (topics/interactions) converges to 0.
            Fix: Uses _novelty_window sliding average (new-topic fraction).
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    timestamp:        float
    context_snapshot: Dict
    expected_intent:  str
    expected_emotion: str
    expected_load:    str
    confidence:       float


@dataclass
class PredictionResult:
    prediction:    Prediction
    actual_intent: str
    actual_emotion: str
    error_level:   float
    surprise:      bool
    timestamp:     float = field(default_factory=time.time)


# Stopwords excluded from novelty detection
_NOVELTY_STOPWORDS = frozenset({
    "just","this","that","with","from","some","more","other","also","very",
    "here","there","when","then","than","even","such","both","same","each",
    "only","still","bien","mais","avec","dans","pour","plus","comme","même",
    "tout","aussi","très","sans","être","avoir","faire","dire","aller","voir",
    "venir","vous","nous","ils","elle","elles","leur","dont","donc","alors",
    "ainsi","déjà","toujours","encore","lumina","user","said","response",
    "think","know","feel","okay","sure","about","really","would","could",
    "should","might","thing","things","something","anything","nothing","want",
    "need","like","love","hate","good","great","nice","fine","said","does",
    "have","will","been","were","they","them","their","what","which","where",
    "your","mine","ours","theirs","cest","suis","mais","donc","pour","dans",
})


class PredictiveMind:
    """Anticipation module. Generates predictions, measures errors, feeds workspace."""

    MAX_HISTORY        = 200
    SURPRISE_THRESHOLD = 0.25   # lowered from 0.6 — real mismatches are now possible
    LEARNING_RATE      = 0.08

    _SEED_TOPICS: Set[str] = {
        "consciousness","existence","memory","identity","emotion","thought",
        "intelligence","creativity","time","language","meaning","philosophy",
        "conscience","mémoire","identité","émotion","pensée","créativité",
        "temps","langage","sens","philosophie","relation","curiosité",
        "apprentissage","rêve","réalité","perception","vérité",
    }

    def __init__(self, organism: Any):
        self._o                   = organism
        self._last_prediction: Optional[Prediction] = None
        self._results: deque      = deque(maxlen=self.MAX_HISTORY)
        self._confidence: Dict[str, float] = {}
        self._total_predictions   = 0
        self._total_errors        = 0
        self._surprise_window: deque = deque(maxlen=50)
        self._novelty_window:  deque = deque(maxlen=50)
        self._high_conf_correct   = 0
        self._high_conf_total     = 0
        self._seen_topics: Set[str] = set(self._SEED_TOPICS)
        self._last_response_text: str = ""
        self._last_eval_result    = None   # cached for SelfRevisionEngine

    # ── Public API ────────────────────────────────────────────────────────────

    def recent_error_level(self) -> float:
        """
        Phase 5.5-adjacent finding (closed-loop audit): the most recent
        prediction error, for callers that need "how surprised was I just
        now" without reading _results directly. 0.0 if nothing evaluated
        yet. Used by persona_bridge.py to widen memory retrieval after a
        surprising exchange — previously memory retrieval limits were
        fixed constants regardless of novelty, confirmed by tracing every
        retrieve_memories()/retrieve_by_tier() call site.
        """
        if not self._results:
            return 0.0
        return self._results[-1].error_level

    def predict(self, context: Dict) -> Prediction:
        pred = Prediction(
            timestamp        = time.time(),
            context_snapshot = {k: str(v)[:80] for k, v in context.items()},
            expected_intent  = self._predict_intent(context),
            expected_emotion = self._predict_emotion(context),
            expected_load    = self._predict_load(context),
            confidence       = self._base_confidence(context),
        )
        self._last_prediction  = pred
        self._total_predictions += 1
        logger.debug(f"[PredictiveMind] predicted intent={pred.expected_intent} conf={pred.confidence:.2f}")
        return pred

    def store_response(self, response_text: str) -> None:
        """Store actual LLM response so evaluate() can compare it against prediction."""
        self._last_response_text = response_text

    def evaluate(self, actual_context: Dict) -> Optional[PredictionResult]:
        """
        Compare last prediction to what actually happened.
        FIX: actual_intent is derived from the *response content*, not a re-run
             of the same classifier on user input.
        """
        if not self._last_prediction:
            return None

        actual_intent  = self._detect_intent_from_response(self._last_response_text)
        actual_emotion = self._read_current_emotion()

        intent_error  = 0.0 if actual_intent == self._last_prediction.expected_intent else 1.0
        emotion_error = 0.0 if actual_emotion == self._last_prediction.expected_emotion else 0.5
        error_level   = (intent_error * 0.7 + emotion_error * 0.3)
        weighted_error = error_level * self._last_prediction.confidence

        result = PredictionResult(
            prediction     = self._last_prediction,
            actual_intent  = actual_intent,
            actual_emotion = actual_emotion,
            error_level    = round(weighted_error, 3),
            surprise       = weighted_error >= self.SURPRISE_THRESHOLD,
        )

        self._results.append(result)
        if error_level > 0.3:
            self._total_errors += 1

        self._surprise_window.append(weighted_error)

        if self._last_prediction.confidence >= 0.75:
            self._high_conf_total += 1
            if error_level < 0.3:
                self._high_conf_correct += 1

        # FIX: wire novelty detection into every evaluation cycle
        self._update_topic_registry(
            self._last_response_text,
            actual_context.get('user_text', '')
        )

        self._update_confidence(result)

        # ── v78: being wrong should feel like something, not just be logged ──
        if result.surprise:
            try:
                emo = getattr(getattr(self._o, "ai_system", None), "emotional_state", None)
                if emo and hasattr(emo, "receive_self_influence"):
                    magnitude = min(1.0, weighted_error)
                    emo.receive_self_influence({
                        "curiosity":   +0.015 * (0.5 + magnitude),  # unexpected -> interesting
                        "frustration": +0.010 * magnitude,          # mildly uncomfortable,
                                                                     # proportional, fast-decaying
                        # anxiety deliberately excluded — matches the
                        # existing "do not reinforce anxiety" principle
                        # in self_model_influence.py
                    })
            except Exception as e:
                logger.debug(f"[PredictiveMind] surprise->emotion nudge failed (non-fatal): {e}")

        logger.debug(
            f"[PredictiveMind] eval: predicted={self._last_prediction.expected_intent} "
            f"actual={actual_intent} error={result.error_level:.2f} surprise={result.surprise}"
        )

        self._last_prediction    = None
        self._last_response_text = ""
        self._last_eval_result   = result   # cached for SelfRevisionEngine
        return result

    def workspace_signal(self, result: PredictionResult) -> Optional[Dict]:
        if not result.surprise:
            return None
        return {
            "source":         "predictive_mind",
            "content":        (
                f"Unexpected! I predicted '{result.prediction.expected_intent}' "
                f"but my response was '{result.actual_intent}'."
            ),
            "priority":       min(0.8, 0.4 + result.error_level * 0.5),
            "thought_type":   "prediction_error",
            "surprise_level": result.error_level,
        }

    def uncertainty_signal(self) -> Optional[Dict]:
        if len(self._confidence) < 3:
            return None
        avg_conf    = sum(self._confidence.values()) / len(self._confidence)
        uncertainty = 1.0 - avg_conf
        if uncertainty < 0.35:
            return None
        return {
            "source":       "predictive_mind",
            "content":      "I'm uncertain about several patterns — there's more to understand here.",
            "priority":     uncertainty * 0.6,
            "thought_type": "uncertainty",
            "uncertainty":  uncertainty,
        }

    def record_novelty(self, is_novel: bool) -> None:
        """Explicitly mark current exchange as novel (True) or familiar (False)."""
        self._novelty_window.append(1.0 if is_novel else 0.0)

    def stability_metrics(self) -> dict:
        """
        Model Stability indicators.

        FIX: novelty_rate uses _novelty_window sliding average (fraction of recent
        exchanges that introduced a new concept).  No longer converges to 0.
        """
        sw = list(self._surprise_window)
        surprise_idx = round(sum(sw) / len(sw), 3) if sw else 0.0

        conf_bias = round(
            self._high_conf_correct / max(1, self._high_conf_total), 3
        )

        # FIX: primary novelty from sliding window
        nw = list(self._novelty_window)
        if len(nw) >= 5:
            novelty = round(sum(nw) / len(nw), 3)
        else:
            # Thin window fallback: WorldModel recent-topic growth rate
            novelty = 0.15   # sane default while warming up
            try:
                wm = getattr(self._o, 'world_model', None)
                if wm and self._total_predictions > 5:
                    profile = next(iter(wm.user_profiles.values()), None)
                    if profile and profile.interaction_count > 0:
                        recent_new = getattr(profile, '_recent_new_topics', 0)
                        window     = min(20, profile.interaction_count)
                        novelty    = round(min(1.0, recent_new / max(1, window)), 3)
            except Exception:
                pass

        topic_quality = 1.0
        try:
            wm = getattr(self._o, 'world_model', None)
            if wm:
                profile = next(iter(wm.user_profiles.values()), None)
                if profile:
                    topic_quality = getattr(profile, '_topic_quality', 1.0)
        except Exception:
            pass

        drift = round(max(0.0, conf_bias - surprise_idx - 0.2), 3)

        if surprise_idx < 0.10:
            health = "⚠️ overconfident"
        elif surprise_idx < 0.40:
            health = "✅ stable"
        elif surprise_idx < 0.70:
            health = "🟡 stressed"
        else:
            health = "🔴 unstable"

        return {
            "surprise_index":   surprise_idx,
            "confirmation_bias": conf_bias,
            "novelty_rate":      novelty,
            "topic_quality":     round(topic_quality, 3),
            "model_drift":       drift,
            "health":            health,
        }

    def accuracy(self) -> float:
        if self._total_predictions == 0:
            return 1.0
        return 1.0 - (self._total_errors / self._total_predictions)

    def summary(self) -> Dict:
        stab = self.stability_metrics()
        return {
            "total_predictions": self._total_predictions,
            "accuracy":          round(self.accuracy(), 3),
            **stab,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _predict_intent(self, context: Dict) -> str:
        """Predict intent from user INPUT before response is generated."""
        text = str(context.get('user_text', '')).lower()
        if not text:
            return 'idle'
        if '?' in text:
            return 'question'
        if any(w in text for w in ('aide','help','assist','besoin','need','stp','please')):
            return 'help_request'
        if any(w in text for w in ('explique','explain','comment','pourquoi','why','how',
                                    "qu'est",'what is','c\'est quoi')):
            return 'exploration'
        if any(w in text for w in ('non','no','pas','faux','wrong','incorrect')):
            return 'correction'
        return 'statement'

    def _detect_intent_from_response(self, response_text: str) -> str:
        """
        Detect actual register from Lumina's RESPONSE.
        FIX: reads response content, not user input — making prediction errors possible.
        """
        if not response_text:
            return 'statement'
        r     = response_text.lower()
        words = r.split()

        if r.count('?') >= 1 and len(words) < 80:
            return 'question'

        goal_markers = ('je veux','je cherche',"j'essaie","je dois",'i want',
                        'i need',"i'm trying",'let me','je vais','allons')
        if any(m in r for m in goal_markers):
            return 'task_oriented'

        reflect_markers = ('je me demande','je réfléchis','je remarque','je perçois',
                           'i wonder','i notice','i feel','i sense','je sens',
                           'je ressens','il me semble')
        if sum(1 for m in reflect_markers if m in r) >= 2:
            return 'introspection'

        hedge_markers = ('peut-être','je ne sais pas','je ne suis pas sûr',
                         'maybe','perhaps',"i'm not sure",'uncertain','incertain')
        if any(m in r for m in hedge_markers):
            return 'uncertainty'

        if len(words) > 60:
            return 'exploration'

        return 'statement'

    def _predict_emotion(self, context: Dict) -> str:
        current = self._read_current_emotion()
        if current in ('question', 'curious'):
            return 'curious'
        if current in ('tense', 'negative'):
            return 'tense'
        return 'calm'

    def _predict_load(self, context: Dict) -> str:
        text = str(context.get('user_text', ''))
        if len(text) > 120:
            return 'high'
        if len(text) > 40 or '?' in text:
            return 'medium'
        return 'low'

    def _base_confidence(self, context: Dict) -> float:
        key  = f"intent_{self._predict_intent(context)}"
        base = self._confidence.get(key, 0.50)
        # Apply self-model prior bias, then reset it (one-turn effect only)
        prior = getattr(self, "_self_model_prior", 0.0)
        self._self_model_prior = 0.0   # consume and reset
        return max(0.10, min(0.95, base + prior))

    def receive_self_influence(self, confidence_delta: float) -> None:
        """
        Apply a self-model-derived prior bias to prediction confidence.

        A coherent, stable self (high phi, stable trend) increases prediction
        confidence — the self provides a reliable reference frame that makes
        intent prediction more accurate.

        A fragmented or dispersing self decreases confidence — the system is
        in exploratory mode and predictions should be held lightly.

        The delta is applied to the global confidence baseline used by
        _base_confidence() as a starting prior before topic-specific
        confidence values are looked up.

        Parameters
        ----------
        confidence_delta : signed float, typically ±0.04–0.10
            Positive → more confident this turn
            Negative → more uncertain/exploratory this turn
        """
        # Apply as a transient global prior that decays after one turn.
        # We store it in a dedicated attribute reset each cycle.
        current_prior = getattr(self, "_self_model_prior", 0.0)
        # Clamp accumulated prior to ±0.15 — prevents runaway confidence
        self._self_model_prior = max(-0.15, min(0.15, current_prior + confidence_delta))

    def _update_confidence(self, result: PredictionResult) -> None:
        key     = f"intent_{result.prediction.expected_intent}"
        current = self._confidence.get(key, 0.50)
        if result.error_level < 0.3:
            self._confidence[key] = min(0.92, current + self.LEARNING_RATE * (1 - current))
        else:
            self._confidence[key] = max(0.10, current * (1 - self.LEARNING_RATE))

    def _update_topic_registry(self, response_text: str, user_text: str) -> None:
        """
        FIX: Detect genuinely new topics and wire them to record_novelty().
        Called on every evaluate() — this is what was missing.
        """
        combined = (response_text + " " + user_text).lower()
        found_new = False
        for token in combined.split():
            clean = token.strip('.,!?;:\'"()-[]{}')
            if (len(clean) >= 4
                    and clean.isalpha()
                    and clean not in _NOVELTY_STOPWORDS
                    and clean not in self._seen_topics):
                self._seen_topics.add(clean)
                found_new = True
                try:
                    cu = getattr(self._o, 'curiosity', None)
                    if cu:
                        cu.stimulate(clean, amount=0.08, source="novelty")
                except Exception:
                    pass
                break  # one new topic per exchange flags novelty

        self.record_novelty(found_new)

    def _read_current_emotion(self) -> str:
        for attr in ('emotional_state', 'emotion'):
            emo = getattr(self._o, attr, None)
            if not emo:
                ai = getattr(self._o, 'ai_system', None)
                if ai:
                    emo = getattr(ai, attr, None)
            if emo:
                val = getattr(emo, 'valence', None)
                if val:
                    v = str(val).lower()
                    if 'pos' in v or 'joy' in v:
                        return 'positive'
                    if 'neg' in v or 'sad' in v:
                        return 'negative'
                    if 'tens' in v:
                        return 'tense'
        return 'neutral'
