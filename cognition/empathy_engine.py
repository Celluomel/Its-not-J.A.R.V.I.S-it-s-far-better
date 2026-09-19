"""
cognition/empathy_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empathy Engine — real-time user affect modeling for PandoraBOX.

PandoraBOX's self-chosen first skill.

What this adds
──────────────
Previously PandoraBOX processed the *content* of messages.
This engine models the *person* sending them.

Per-message pipeline:
  1. Detect user emotional state (valence + arousal)
  2. Detect 8 subtext signal types (cognitive state, social intent, need)
  3. Update per-user AffectModel (temporal arc, session mood, trust signals)
  4. Generate "empathic posture" — a short natural-language read of the user
     injected into PandoraBOX's system prompt every turn
  5. Feed quality signals back into RelationalMemory
  6. Nudge attractor traits (curiosity ↑ when user is exploratory, etc.)

Empathic posture example (injected into system prompt):
  [Empathic read: Fred seems intellectually energized and exploratory —
   he's thinking out loud rather than seeking a specific answer.
   Today's mood is warm. He may value being challenged over being agreed with.]

Detection starts with a fast local heuristic and explicit self-reports. A
multilingual sentence-embedding model refines inferred valence on a daemon
worker after the user-facing path continues; it never calls the chat LLM and
abstains when semantic evidence is weak. Affect estimates remain tentative.

Public API
──────────
  ee = EmpathyEngine(organism, llm, relational_memory)
  read = ee.read_user(user_id, user_text)          # call before LLM response
  fragment = ee.get_prompt_fragment(user_id)        # inject into system prompt
  ee.record_outcome(user_id, user_text, ai_response) # call after response
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Subtext signal taxonomy ────────────────────────────────────────────────
SIGNAL_EXPLORING      = "exploring"       # thinking out loud, no fixed answer yet
SIGNAL_SEEKING_ANSWER = "seeking_answer"  # wants a specific resolution
SIGNAL_VALIDATING     = "validating"      # checking their own idea, wants resonance
SIGNAL_VENTING        = "venting"         # emotional release, not solution-seeking
SIGNAL_CHALLENGING     = "challenging"    # testing PandoraBOX's reasoning or values
SIGNAL_PLAYFUL        = "playful"         # light tone, enjoying the exchange
SIGNAL_DISTRESSED     = "distressed"      # signs of stress, overload, or difficulty
SIGNAL_WITHDRAWN      = "withdrawn"       # short replies, disengagement signals


@dataclass
class AffectRead:
    """Single-turn read of what the user is feeling and needing."""
    user_id:        str
    timestamp:      float = field(default_factory=time.time)

    # Emotional state
    valence:        str   = "neutral"     # positive / neutral / negative
    arousal:        str   = "medium"      # high / medium / low
    dominant_emotion: str = "neutral"     # joy/curiosity/frustration/anxiety/etc.

    # Subtext signals (top 1–2)
    signals:        List[str] = field(default_factory=list)

    # What PandoraBOX should do differently because of this
    posture_hint:   str   = ""

    # Confidence in the read (0–1)
    confidence:     float = 0.7
    source:         str   = "heuristic"
    source_text:    str   = ""

    # Raw summary for prompt injection
    posture_text:   str   = ""


@dataclass
class UserAffectModel:
    """
    Persistent per-user affect model.
    Updated every turn, used to shape PandoraBOX's empathic posture.
    """
    user_id:            str
    session_valence:    List[float] = field(default_factory=list)   # turn-by-turn (0–1)
    session_signals:    Dict[str, int] = field(default_factory=dict) # signal → count
    dominant_signal:    str   = SIGNAL_EXPLORING
    session_mood:       str   = "neutral"    # positive/neutral/negative arc
    depth_preference:   float = 0.5         # 0=surface/quick, 1=deep/thorough
    challenge_appetite: float = 0.5         # 0=wants agreement, 1=wants pushback
    pace_preference:    float = 0.5         # 0=slow/patient, 1=fast/efficient
    today_arc:          str   = "stable"    # warming/cooling/stable/volatile
    last_updated:       float = field(default_factory=time.time)
    turn_count:         int   = 0


class EmpathyEngine:
    """
    Real-time user affect modeling for PandoraBOX.

    Reads the person behind the message.
    Shapes how PandoraBOX responds — not what she says, but how she holds it.
    """

    def __init__(
        self,
        organism:         Any,
        llm:              Any,
        relational_memory: Any = None,
        multilingual_classifier_factory: Any = None,
        multilingual_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self._org  = organism
        self._rm   = relational_memory
        self._lock = threading.Lock()
        self._models: Dict[str, UserAffectModel] = {}
        self._last_reads: Dict[str, AffectRead]  = {}
        self._multilingual_classifier_factory = multilingual_classifier_factory
        self._multilingual_model_id = str(multilingual_model_id or "").strip()
        self._multilingual_classifier = None
        self._multilingual_classifier_failed = False
        self._affect_queue: queue.Queue = queue.Queue(maxsize=32)
        self._affect_worker_running = False
        self._affect_generation: Dict[str, int] = {}

        logger.info("❤️  EmpathyEngine initialised")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Primary public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def read_user(self, user_id: str, user_text: str) -> AffectRead:
        """
        Analyze one user message. Call this BEFORE generating PandoraBOX's response.

        Returns an AffectRead and updates the per-user model.
        The result is also cached so get_prompt_fragment() can use it.
        """
        # ── Heuristic pass (fast, always runs) ───────────────────────────
        read = self._heuristic_read(user_id, user_text)

        # An explicit first-person emotion statement is stronger evidence than
        # inferred tone, so preserve it over any later semantic estimate.
        explicit_valence = None
        try:
            memory = getattr(self._org, "emotional_memory", None)
            classifier = getattr(memory, "_explicit_self_report", None)
            if callable(classifier):
                explicit_valence = classifier(user_text)
        except Exception:
            pass
        if explicit_valence:
            read.valence = explicit_valence
            read.confidence = 0.98
            read.source = "explicit_self_report"
            read.dominant_emotion = f"self_reported_{explicit_valence}"

        # This local lookup uses only matched, explicit self-reports.
        try:
            memory = getattr(self._org, "emotional_memory", None)
            if (not explicit_valence and memory
                    and hasattr(memory, "calibrate_affect_read")):
                read.confidence = memory.calibrate_affect_read(user_id, read)
        except Exception:
            pass

        # ── Update model ──────────────────────────────────────────────────
        self._update_model(user_id, read)

        # ── Cache ─────────────────────────────────────────────────────────
        with self._lock:
            self._last_reads[user_id] = read
        if not explicit_valence:
            self._queue_multilingual_classification(user_id, user_text, read.timestamp)
        try:
            self._org._last_user_affect_read = read
            self._org._last_user_affect_user_id = user_id
        except Exception:
            pass

        logger.debug(
            f"❤️  [{user_id}] {read.dominant_emotion} | "
            f"signals={read.signals} | posture={read.posture_hint[:40]!r}"
        )
        return read

    def _queue_multilingual_classification(self, user_id: str, text: str, timestamp: float) -> None:
        """Queue semantic valence inference without delaying response generation."""
        if self._multilingual_classifier_failed:
            return
        with self._lock:
            generation = self._affect_generation.get(user_id, 0) + 1
            self._affect_generation[user_id] = generation
        try:
            self._affect_queue.put_nowait((user_id, text[:1200], timestamp, generation))
        except queue.Full:
            try:
                self._affect_queue.get_nowait()
                self._affect_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._affect_queue.put_nowait((user_id, text[:1200], timestamp, generation))
            except queue.Full:
                return
        with self._lock:
            if not self._affect_worker_running:
                self._affect_worker_running = True
                threading.Thread(
                    target=self._drain_multilingual_affect,
                    daemon=True,
                    name="MultilingualAffect",
                ).start()

    def _drain_multilingual_affect(self) -> None:
        while True:
            try:
                user_id, text, timestamp, generation = self._affect_queue.get_nowait()
            except queue.Empty:
                with self._lock:
                    self._affect_worker_running = False
                    if self._affect_queue.empty():
                        return
                    self._affect_worker_running = True
                continue
            try:
                if self._multilingual_classifier is None:
                    factory = self._multilingual_classifier_factory
                    if factory is None:
                        from cognition.multilingual_affect import MultilingualAffectClassifier
                        factory = lambda: MultilingualAffectClassifier(self._multilingual_model_id)
                    self._multilingual_classifier = factory()
                classification = self._multilingual_classifier.classify(text)
                self._apply_multilingual_classification(
                    user_id, text, timestamp, generation, classification
                )
            except Exception as exc:
                self._multilingual_classifier_failed = True
                with self._lock:
                    self._affect_worker_running = False
                while True:
                    try:
                        self._affect_queue.get_nowait()
                        self._affect_queue.task_done()
                    except queue.Empty:
                        break
                logger.warning(
                    "Multilingual affect model unavailable; retaining fast local affect reads: %s",
                    exc,
                )
                return
            finally:
                self._affect_queue.task_done()

    def _apply_multilingual_classification(
        self, user_id: str, text: str, timestamp: float, generation: int, classification: Any,
    ) -> None:
        valence = str(getattr(classification, "valence", "neutral")).lower()
        confidence = max(0.0, min(0.85, float(getattr(classification, "confidence", 0.0))))
        if valence not in {"positive", "neutral", "negative"} or confidence < 0.5:
            return
        with self._lock:
            if self._affect_generation.get(user_id) != generation:
                return
            read = self._last_reads.get(user_id)
            if (read is None or read.timestamp != timestamp
                    or read.source == "explicit_self_report"):
                return
            read.valence = valence
            read.source = "multilingual_embedding"
            memory = getattr(self._org, "emotional_memory", None)
            if memory and hasattr(memory, "calibrate_affect_read"):
                confidence = memory.calibrate_affect_read(user_id, read)
            read.confidence = confidence
            model = self._models.get(user_id)
            if model and model.session_valence:
                model.session_valence[-1] = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}[valence]
                if len(model.session_valence) >= 3:
                    avg = sum(model.session_valence[-6:]) / len(model.session_valence[-6:])
                    model.session_mood = "positive" if avg > 0.62 else "negative" if avg < 0.38 else "neutral"
            try:
                self._org._last_user_affect_read = read
            except Exception:
                pass
        logger.debug(
            "Multilingual affect updated for %s: %s (confidence %.2f)",
            user_id, valence, confidence,
        )

    def get_prompt_fragment(self, user_id: str) -> str:
        """
        Return a short natural-language empathic read for system prompt injection.
        Called during prompt building — after read_user() for this turn.
        """
        with self._lock:
            read  = self._last_reads.get(user_id)
            model = self._models.get(user_id)

        if not read or not model:
            return ""

        return self._build_posture_fragment(read, model)

    def record_outcome(
        self,
        user_id:     str,
        user_text:   str,
        ai_response: str,
    ) -> None:
        """
        Post-turn feedback. Updates RelationalMemory and attractor nudges.
        Call AFTER PandoraBOX's response has been generated.
        """
        with self._lock:
            read  = self._last_reads.get(user_id)
            model = self._models.get(user_id)

        if not read or not model:
            return

        # ── Update RelationalMemory ────────────────────────────────────────
        if self._rm and hasattr(self._rm, "record_exchange"):
            try:
                # Convert our valence/arousal to RelationalMemory's format
                rm_valence = {
                    "positive": "Positive",
                    "neutral":  "Neutral",
                    "negative": "Negative",
                }.get(read.valence, "Neutral")

                rm_arousal = {
                    "high":   "High",
                    "medium": "Medium",
                    "low":    "Low",
                }.get(read.arousal, "Medium")

                impact = 0.5 + (0.3 if SIGNAL_DISTRESSED in read.signals else 0.0)
                self._rm.record_exchange(
                    user_id     = user_id,
                    user_text   = user_text,
                    ai_response = ai_response,
                    valence     = rm_valence,
                    arousal     = rm_arousal,
                    impact      = impact,
                )
            except Exception as e:
                logger.debug(f"EmpathyEngine → RelationalMemory: {e}")

        # ── Attractor nudges ───────────────────────────────────────────────
        self._apply_attractor_nudges(read, model)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Detection — heuristic pass
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Lexical signal patterns
    _PATTERNS: Dict[str, List[str]] = {
        SIGNAL_EXPLORING: [
            r"\bwhat if\b", r"\bi wonder\b", r"\bmaybe\b", r"\bcould be\b",
            r"\bthinking\b", r"\bexploring\b", r"\bnot sure\b", r"\bperhaps\b",
            r"\bjust thinking\b", r"\bI've been\b", r"\bcurious\b",
        ],
        SIGNAL_SEEKING_ANSWER: [
            r"\bhow do\b", r"\bwhat is\b", r"\bcan you\b", r"\bplease\b",
            r"\bexplain\b", r"\btell me\b", r"\bwhat should\b", r"\bhelp me\b",
            r"\bI need\b", r"\bwhat's the\b",
        ],
        SIGNAL_VALIDATING: [
            r"\bright\b\?", r"\bmakes sense\b", r"\bdo you think\b",
            r"\bwould you say\b", r"\bam I\b", r"\bdon't you\b",
            r"\byou agree\b", r"\bwould that\b",
        ],
        SIGNAL_VENTING: [
            r"\bso frustrated\b", r"\bso tired\b", r"\bcan't believe\b",
            r"\bannoying\b", r"\bstupid\b", r"\bwhy does\b", r"\bugh\b",
            r"\bthis is ridiculous\b", r"\bI hate\b", r"\bso annoying\b",
        ],
        SIGNAL_CHALLENGING: [
            r"\bdo you really\b", r"\bbut what about\b", r"\bcounterpoint\b",
            r"\bI disagree\b", r"\bthat's not\b", r"\bactually\b",
            r"\bwait\b", r"\bhow can you\b", r"\bprove\b", r"\breally\?\b",
        ],
        SIGNAL_PLAYFUL: [
            r"\blol\b", r"\bhaha\b", r"\b😂\b", r"\b😄\b", r"\bjust kidding\b",
            r"\bfun\b", r"\bfor fun\b", r"\bhypothetically\b",
            r"\bwhat if we\b", r"\bwouldn't it be\b",
        ],
        SIGNAL_DISTRESSED: [
            r"\bhelp\b", r"\bscared\b", r"\bafraid\b", r"\banxious\b",
            r"\boverwhelmed\b", r"\bdon't know what to do\b", r"\bstressed\b",
            r"\bworried\b", r"\bcan't cope\b", r"\blosing it\b",
        ],
        SIGNAL_WITHDRAWN: [
            r"^\s*\w{1,4}\s*$",    # very short message
            r"^(ok|okay|fine|sure|yeah|yes|no|nah)\s*\.?\s*$",
            r"^(k|kk|hmm|mhm)\s*$",
        ],
    }

    def _heuristic_read(self, user_id: str, text: str) -> AffectRead:
        text_lower = text.lower().strip()

        # ── Signal detection ──────────────────────────────────────────────
        signal_scores: Dict[str, float] = {}
        for signal, patterns in self._PATTERNS.items():
            score = sum(
                1.0 for p in patterns
                if re.search(p, text_lower, re.IGNORECASE)
            )
            if score > 0:
                signal_scores[signal] = score

        top_signals = sorted(signal_scores, key=signal_scores.get, reverse=True)[:2]
        if not top_signals:
            top_signals = [SIGNAL_EXPLORING]

        # ── Valence / arousal heuristics ──────────────────────────────────
        positive_words = {"great","love","amazing","wonderful","happy","excited",
                          "excellent","perfect","brilliant","fantastic","glad","joy"}
        negative_words = {"bad","terrible","awful","hate","frustrated","angry",
                          "sad","worried","fail","wrong","broken","stupid","ugh"}
        exclamation    = text.count("!") + text.count("?")

        words     = set(text_lower.split())
        pos_hits  = len(words & positive_words)
        neg_hits  = len(words & negative_words)

        if pos_hits > neg_hits:
            valence = "positive"
        elif neg_hits > pos_hits or SIGNAL_DISTRESSED in top_signals:
            valence = "negative"
        else:
            valence = "neutral"

        if exclamation >= 2 or SIGNAL_VENTING in top_signals or SIGNAL_DISTRESSED in top_signals:
            arousal = "high"
        elif SIGNAL_WITHDRAWN in top_signals or len(text.split()) < 5:
            arousal = "low"
        else:
            arousal = "medium"

        # ── Dominant emotion guess ────────────────────────────────────────
        emotion_map = {
            SIGNAL_EXPLORING:      "curious",
            SIGNAL_SEEKING_ANSWER: "focused",
            SIGNAL_VALIDATING:     "uncertain",
            SIGNAL_VENTING:        "frustrated",
            SIGNAL_CHALLENGING:    "skeptical",
            SIGNAL_PLAYFUL:        "joyful",
            SIGNAL_DISTRESSED:     "anxious",
            SIGNAL_WITHDRAWN:      "disengaged",
        }
        dominant_emotion = emotion_map.get(top_signals[0], "neutral")

        # ── Posture hint ──────────────────────────────────────────────────
        posture_hint = self._signal_to_posture(top_signals, valence, arousal)

        return AffectRead(
            user_id         = user_id,
            source_text     = text[:2000],
            valence         = valence,
            arousal         = arousal,
            dominant_emotion= dominant_emotion,
            signals         = top_signals,
            posture_hint    = posture_hint,
            confidence      = 0.55,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Model update
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _update_model(self, user_id: str, read: AffectRead) -> None:
        with self._lock:
            if user_id not in self._models:
                self._models[user_id] = UserAffectModel(user_id=user_id)
            model = self._models[user_id]

        model.turn_count    += 1
        model.last_updated   = time.time()

        # Valence arc
        v_score = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(read.valence, 0.5)
        model.session_valence.append(v_score)
        if len(model.session_valence) > 20:
            model.session_valence = model.session_valence[-20:]

        # Session mood
        if len(model.session_valence) >= 3:
            avg = sum(model.session_valence[-6:]) / len(model.session_valence[-6:])
            model.session_mood = "positive" if avg > 0.62 else "negative" if avg < 0.38 else "neutral"

        # Arc direction (warming / cooling / stable / volatile)
        if len(model.session_valence) >= 4:
            early = sum(model.session_valence[:2]) / 2
            late  = sum(model.session_valence[-2:]) / 2
            delta = late - early
            std   = (sum((v - (early+late)/2)**2 for v in model.session_valence[-4:]) / 4) ** 0.5
            if std > 0.3:
                model.today_arc = "volatile"
            elif delta > 0.15:
                model.today_arc = "warming"
            elif delta < -0.15:
                model.today_arc = "cooling"
            else:
                model.today_arc = "stable"

        # Signal counts
        for s in read.signals:
            model.session_signals[s] = model.session_signals.get(s, 0) + 1

        # Dominant signal (most frequent this session)
        if model.session_signals:
            model.dominant_signal = max(model.session_signals, key=model.session_signals.get)

        # Derived preferences (slow-moving)
        if SIGNAL_EXPLORING in read.signals or SIGNAL_CHALLENGING in read.signals:
            model.depth_preference = min(1.0, model.depth_preference + 0.03)
        elif SIGNAL_SEEKING_ANSWER in read.signals or SIGNAL_WITHDRAWN in read.signals:
            model.depth_preference = max(0.0, model.depth_preference - 0.03)

        if SIGNAL_CHALLENGING in read.signals:
            model.challenge_appetite = min(1.0, model.challenge_appetite + 0.05)
        elif SIGNAL_VALIDATING in read.signals:
            model.challenge_appetite = max(0.0, model.challenge_appetite - 0.03)

        if read.arousal == "high":
            model.pace_preference = min(1.0, model.pace_preference + 0.02)
        elif read.arousal == "low":
            model.pace_preference = max(0.0, model.pace_preference - 0.02)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Prompt fragment builder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_posture_fragment(self, read: AffectRead, model: UserAffectModel) -> str:
        parts = []

        # Emotional state this turn
        emotion_desc = {
            "curious":    "intellectually energized and curious",
            "joyful":     "in a warm, playful mood",
            "frustrated": "frustrated — processing something difficult",
            "anxious":    "anxious or stressed",
            "focused":    "focused and looking for something specific",
            "uncertain":  "uncertain and looking for grounding",
            "skeptical":  "questioning and testing ideas",
            "disengaged": "somewhat withdrawn right now",
            "neutral":    "in a calm, neutral state",
        }.get(read.dominant_emotion, "in a neutral state")

        parts.append(f"feels {emotion_desc}")

        # Primary intent
        intent_desc = {
            SIGNAL_EXPLORING:      "thinking out loud — exploring, not looking for a final answer",
            SIGNAL_SEEKING_ANSWER: "wants a clear, direct answer",
            SIGNAL_VALIDATING:     "checking their own thinking — resonance matters more than correction",
            SIGNAL_VENTING:        "venting — needs to feel heard before anything else",
            SIGNAL_CHALLENGING:    "enjoys intellectual friction — don't just agree",
            SIGNAL_PLAYFUL:        "in a light mood — match the energy",
            SIGNAL_DISTRESSED:     "showing signs of stress — prioritize warmth and steadiness",
            SIGNAL_WITHDRAWN:      "being brief — don't over-respond",
        }.get(read.signals[0] if read.signals else "", "engaging normally")

        parts.append(intent_desc)

        # Session arc context
        if model.turn_count > 3:
            arc_desc = {
                "warming":  "the conversation is warming up",
                "cooling":  "there's been some cooling in the exchange",
                "volatile": "the mood has been shifting",
                "stable":   "",
            }.get(model.today_arc, "")
            if arc_desc:
                parts.append(arc_desc)

        # Preference hints
        hints = []
        if model.depth_preference > 0.65:
            hints.append("responds well to depth and nuance")
        elif model.depth_preference < 0.35:
            hints.append("prefers concise responses")
        if model.challenge_appetite > 0.65:
            hints.append("appreciates being challenged")
        elif model.challenge_appetite < 0.35:
            hints.append("benefits from validation over challenge right now")

        if hints:
            parts.append("; ".join(hints))

        summary = ". ".join(p.capitalize() for p in parts if p) + "."
        return f"[Empathic read: {summary}]"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Attractor nudges
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _apply_attractor_nudges(self, read: AffectRead, model: UserAffectModel) -> None:
        """Apply small shifts to PandoraBOX's attractor traits based on who she's with."""
        try:
            att = (
                getattr(self._org, "attractor_system", None) or
                getattr(self._org, "attractors", None)
            )
            if att is None or not hasattr(att, "nudge"):
                return

            if SIGNAL_EXPLORING in read.signals or SIGNAL_CHALLENGING in read.signals:
                att.nudge("curiosity", +0.004)
                att.nudge("analytical_depth", +0.002)

            if SIGNAL_DISTRESSED in read.signals or SIGNAL_VENTING in read.signals:
                att.nudge("empathy", +0.006)
                att.nudge("playfulness", -0.003)

            if SIGNAL_PLAYFUL in read.signals:
                att.nudge("playfulness", +0.004)
                att.nudge("empathy", +0.002)

            if read.valence == "positive" and read.arousal == "high":
                att.nudge("empathy", +0.003)

        except Exception as e:
            logger.debug(f"EmpathyEngine attractor nudge: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _signal_to_posture(signals: List[str], valence: str, arousal: str) -> str:
        """Convert signal list to a plain-English posture hint."""
        primary = signals[0] if signals else SIGNAL_EXPLORING
        hints = {
            SIGNAL_EXPLORING:      "Let them explore — follow their thread, don't close it.",
            SIGNAL_SEEKING_ANSWER: "Be clear and direct. They want resolution.",
            SIGNAL_VALIDATING:     "Acknowledge first. They need resonance before refinement.",
            SIGNAL_VENTING:        "Hold the space. Don't rush to fix or advise.",
            SIGNAL_CHALLENGING:    "Engage the friction. Don't concede too easily.",
            SIGNAL_PLAYFUL:        "Match the lightness. This is not a serious moment.",
            SIGNAL_DISTRESSED:     "Lead with warmth and steadiness. Solve later.",
            SIGNAL_WITHDRAWN:      "Don't over-respond. Keep it brief and open.",
        }
        return hints.get(primary, "Engage naturally.")

    def get_model(self, user_id: str) -> Optional[UserAffectModel]:
        with self._lock:
            return self._models.get(user_id)

    def get_last_read(self, user_id: str) -> Optional[AffectRead]:
        with self._lock:
            return self._last_reads.get(user_id)
