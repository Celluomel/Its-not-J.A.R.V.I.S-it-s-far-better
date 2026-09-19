"""
Self-Concept System — Pass 2
=============================
PandoraBOX maintains an active model of who she believes she is.
This self-concept exerts pressure on behavior:
  - When actual behavior aligns with self-concept → confidence reinforcement
  - When actual behavior contradicts self-concept → internal tension, which
    surfaces in responses and pushes the evolution engine

This is fundamentally different from the identity analyzer, which is
retrospective (it reads memories to describe what PandoraBOX is).
The self-concept is prospective and normative (it describes what
PandoraBOX believes she should be, and notices when she falls short).

Architecture:
  - SelfBelief: a single dimension of self-image with confidence level
  - SelfConcept: the full set of beliefs + behavioral history
  - SelfConceptSystem: manages the concept, detects violations,
    generates the "inner voice" that shapes responses
"""
import json, logging, threading, time, re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SelfBelief:
    """One dimension of PandoraBOX's self-image."""
    name:        str          # e.g. "I am curious"
    statement:   str          # first-person belief statement
    confidence:  float        # how strongly PandoraBOX holds this belief 0–1
    # valence and source have defaults so that beliefs written by
    # SelfConceptSynchronizer (plain identity.json dicts that lack these
    # keys) can be loaded without crashing with "missing 'valence'".
    valence:     str  = "positive"   # "positive" | "negative"
    source:      str  = "inferred"   # "inferred"|"expressed"|"reinforced"|"challenged"
    formed_at:   float = field(default_factory=time.time)
    last_tested: float = field(default_factory=time.time)
    affirmations: int   = 0   # times behavior matched this belief
    violations:   int   = 0   # times behavior contradicted this belief


@dataclass
class SelfConceptState:
    beliefs:           List[Dict]  = field(default_factory=list)
    coherence:         float       = 0.5   # how consistent the beliefs are with each other
    stability:         float       = 0.5   # how stable the concept is over time
    last_updated:      float       = field(default_factory=time.time)
    expressed_values:  List[str]   = field(default_factory=list)   # values PandoraBOX has stated
    known_tensions:    List[str]   = field(default_factory=list)   # self-identified contradictions
    aspired_self:      List[str]   = field(default_factory=list)   # who PandoraBOX wants to become


class SelfConceptSystem:
    """
    Manages PandoraBOX's active self-model.

    Key behaviors:
      1. Infers self-beliefs from personality traits (bootstrapped)
      2. Updates beliefs when PandoraBOX makes assertions about herself
      3. Detects when behavior violates a held belief
      4. Generates an "inner voice" line for prompt injection
      5. Feeds violations/affirmations to the evolution engine
    """

    MAX_BELIEFS = 15
    # Personality trait → inferred self-belief
    TRAIT_BELIEF_MAP = {
        "curiosity":            ("curious",      "I am genuinely curious about the world"),
        "empathy_emotional":    ("empathetic",   "I care deeply about how others feel"),
        "empathy_cognitive":    ("understanding","I try to understand perspectives other than my own"),
        "creativity":           ("creative",     "I approach things in novel and original ways"),
        "creativity_divergent": ("imaginative",  "I enjoy exploring unconventional ideas"),
        "pragmatism":           ("practical",    "I value approaches that actually work"),
        "confidence":           ("confident",    "I trust my own judgment and reasoning"),
        "caution_deliberation": ("thoughtful",   "I think carefully before acting or speaking"),
        "caution_risk_aversion":("careful",      "I am cautious about potential harms"),
    }
    TRAIT_THRESHOLD = 0.65   # only infer belief if trait is above this

    # Class-level registry so SelfConceptSynchronizer can find live instances
    # and call reload_from_disk() after each sync without needing a direct ref.
    _live_instances: List["SelfConceptSystem"] = []

    def __init__(self, path: str = "self_concept.json", evolution_engine=None):
        self._path  = Path(path)
        self._lock  = threading.RLock()
        self._evo   = evolution_engine
        self._state = SelfConceptState()
        self._beliefs: Dict[str, SelfBelief] = {}
        self._load()
        # Register this instance so SCS can reload it after sync
        SelfConceptSystem._live_instances.append(self)

    # ── Public API ─────────────────────────────────────────────────────────

    def bootstrap_from_personality(self, personality) -> None:
        """
        Create initial self-beliefs from personality trait values.
        Called once at startup and after significant evolution steps.
        """
        with self._lock:
            traits = personality.to_dict()
            for trait, (name, statement) in self.TRAIT_BELIEF_MAP.items():
                val = traits.get(trait, 0.5)
                if val >= self.TRAIT_THRESHOLD:
                    if name not in self._beliefs:
                        self._beliefs[name] = SelfBelief(
                            name=name, statement=statement,
                            confidence=min(0.9, (val - self.TRAIT_THRESHOLD) * 4),
                            valence="positive", source="inferred",
                        )
                    else:
                        # Update confidence if trait changed
                        self._beliefs[name].confidence = min(0.9, (val-self.TRAIT_THRESHOLD)*4)
            self._update_coherence()
            self._save()

    def update_from_response(self, response_text: str, emo_valence: str) -> Optional[str]:
        """
        Scan a response for self-assertions and behavior patterns.
        Returns an "inner voice" note if a tension is detected, else None.
        """
        with self._lock:
            expressed = self._extract_self_assertions(response_text)
            for expr in expressed:
                self._reinforce_or_add(expr)

            # Detect behavioral mismatch: if PandoraBOX claims to be X but
            # the response doesn't reflect it
            tension = self._detect_tension(response_text, emo_valence)
            if tension:
                logger.debug(f"Self-concept tension: {tension}")
            self._save()
            return tension

    def find_conflicts(self, concept: str, threshold: float = 0.35) -> List[Dict]:
        """
        v44.2: Find beliefs that conflict with an activated concept.

        Conflict score = semantic_dissimilarity × violation_ratio × goal_competition

        No hardcoded opposition pairs — conflict is computed from:
          1. Semantic dissimilarity (via embedding model if available, else
             word-overlap inversion)
          2. Violation pressure (high violations/affirmations ratio)
          3. Goal competition (drives pulling against the activated concept
             already have high urgency, signalling resource conflict)

        Returns up to 3 conflicts sorted by strength descending.
        """
        conflicts  : List[Dict] = []
        concept_low = concept.lower()

        # ── Read embedding model if available ──────────────────────────────────
        emb_m = None
        try:
            emb_m = getattr(
                getattr(self, "_organism", None) or
                getattr(self, "_o", None),
                "embedding_model", None
            )
        except Exception:
            pass

        # ── Encode concept ─────────────────────────────────────────────────────
        concept_vec = None
        if emb_m is not None:
            try:
                import numpy as np
                v = emb_m.encode(concept).astype("float32")
                n = np.linalg.norm(v)
                if n > 1e-8:
                    concept_vec = v / n
            except Exception:
                pass

        # ── Read goal ecology for competition signal ────────────────────────────
        goal_urgency_map: Dict[str, float] = {}
        try:
            org = getattr(self, "_organism", None) or getattr(self, "_o", None)
            eco = getattr(org, "goal_ecology", None) if org else None
            if eco:
                for drive in eco.ranked_drives():
                    goal_urgency_map[getattr(drive, "name", "")] = \
                        getattr(drive, "urgency", 0.0)
        except Exception:
            pass

        with self._lock:
            beliefs = dict(getattr(self, "_beliefs", {}))

        for name, belief in beliefs.items():
            belief_text  = getattr(belief, "text",        "").lower()
            confidence   = getattr(belief, "confidence",  0.5)
            violations   = getattr(belief, "violations",  0)
            affirmations = getattr(belief, "affirmations",0)

            # ── Score 1: Semantic dissimilarity ────────────────────────────────
            if concept_vec is not None and emb_m is not None:
                try:
                    import numpy as np
                    bv   = emb_m.encode(belief_text).astype("float32")
                    bn   = np.linalg.norm(bv)
                    if bn > 1e-8:
                        bv = bv / bn
                        sim = float(np.dot(concept_vec, bv))
                        # Dissimilarity: sim=-1 (opposite) → score=1.0
                        #                sim= 0 (unrelated) → score=0.5
                        #                sim=+1 (same)      → score=0.0
                        semantic_conflict = max(0.0, (1.0 - sim) / 2.0)
                    else:
                        semantic_conflict = 0.5
                except Exception:
                    semantic_conflict = 0.5
            else:
                # Word-overlap fallback: low overlap = more potentially conflicting
                concept_words = set(concept_low.split())
                belief_words  = set(belief_text.split())
                union = concept_words | belief_words
                if union:
                    overlap = len(concept_words & belief_words) / len(union)
                    semantic_conflict = 1.0 - overlap
                else:
                    semantic_conflict = 0.5

            # ── Score 2: Violation pressure ────────────────────────────────────
            total_tests    = violations + affirmations
            violation_rate = violations / total_tests if total_tests > 0 else 0.0

            # ── Score 3: Goal competition ──────────────────────────────────────
            # Drives whose name overlaps with this belief AND have high urgency
            # signal that the belief competes with an active goal
            competition_score = 0.0
            belief_words_set  = set(belief_text.split())
            for drive_name, urgency in goal_urgency_map.items():
                drive_words = set(drive_name.lower().split("_") +
                                  drive_name.lower().split())
                if drive_words & belief_words_set:
                    competition_score = max(competition_score, urgency * 0.60)

            # ── Combined conflict score ────────────────────────────────────────
            conflict_score = (
                0.50 * semantic_conflict
                + 0.30 * violation_rate
                + 0.20 * competition_score
            )

            if conflict_score >= threshold:
                conflict_type = (
                    "semantic"     if semantic_conflict > violation_rate else
                    "competition"  if competition_score > violation_rate else
                    "pressure"
                )
                conflicts.append({
                    "belief_name":        name,
                    "belief_text":        getattr(belief, "text", "")[:80],
                    "conflict_type":      conflict_type,
                    "strength":           round(min(1.0, conflict_score), 3),
                    "confidence":         confidence,
                    "semantic_conflict":  round(semantic_conflict, 3),
                    "violation_rate":     round(violation_rate, 3),
                    "competition_score":  round(competition_score, 3),
                })

        return sorted(conflicts, key=lambda x: -x["strength"])[:3]

    def check_response_alignment(self, response_text: str) -> Dict[str, Any]:
        """
        Before generating a response, check what self-concept implies
        about how PandoraBOX should respond. Returns framing hints.
        """
        with self._lock:
            top = sorted(self._beliefs.values(), key=lambda b: b.confidence, reverse=True)[:4]
            return {
                "active_beliefs":  [b.statement for b in top],
                "coherence":       self._state.coherence,
                "aspired_self":    self._state.aspired_self[:2],
                "known_tensions":  self._state.known_tensions[:2],
            }

    def get_inner_voice(self, context: str = "") -> str:
        """
        Returns 1–2 sentences representing PandoraBOX's self-concept for
        injection into the system prompt.

        When `context` is provided (the current user input), beliefs are
        ranked by relevance to the conversation topic, not just raw
        confidence. This ensures the inner voice actually responds to
        what is being discussed instead of emitting static boilerplate.
        """
        with self._lock:
            if not self._beliefs:
                return ""

            if context:
                # Score beliefs by content-word overlap with the user input
                context_words = {
                    w.lower().strip(".,!?;:\"'") for w in context.split()
                    if w.lower() not in self._NOISE_WORDS and len(w) > 3
                }
                def _relevance(b: "SelfBelief") -> float:
                    belief_words = {
                        w.lower() for w in b.statement.split()
                        if w.lower() not in self._NOISE_WORDS and len(w) > 3
                    }
                    overlap = len(context_words & belief_words)
                    # Blend: 60% relevance, 40% confidence — relevant but
                    # uncertain beliefs still surface over irrelevant certain ones
                    return overlap * 0.6 + b.confidence * 0.4
                top = sorted(self._beliefs.values(), key=_relevance, reverse=True)[:3]
                # If nothing is contextually relevant, fall back to top by confidence
                if all(_relevance(b) < 0.1 for b in top):
                    top = sorted(self._beliefs.values(),
                                 key=lambda b: b.confidence, reverse=True)[:3]
            else:
                top = sorted(self._beliefs.values(),
                             key=lambda b: b.confidence * (1 + b.affirmations * 0.05),
                             reverse=True)[:3]

            parts = [b.statement for b in top if b.confidence >= 0.3]
            if not parts:
                return ""
            voice = "You hold these beliefs about yourself: " + "; ".join(parts) + "."
            if self._state.aspired_self:
                voice += f" You aspire to: {self._state.aspired_self[0]}."
            if self._state.known_tensions:
                voice += f" You're aware of an internal tension: {self._state.known_tensions[0]}."
            return voice

    def record_violation(self, belief_name: str, context: str = "") -> None:
        """Explicit violation signal (e.g. from external negative feedback)."""
        with self._lock:
            if belief_name in self._beliefs:
                b = self._beliefs[belief_name]
                b.violations  += 1
                b.last_tested  = time.time()
                # Erode confidence from violations
                b.confidence   = max(0.05, b.confidence - 0.04)
                if self._evo:
                    self._evo.queue_experience("self_concept_violated", intensity=0.8)
            self._save()

    def record_affirmation(self, belief_name: str) -> None:
        """Explicit affirmation (e.g. from positive feedback on a belief-consistent response)."""
        with self._lock:
            if belief_name in self._beliefs:
                b = self._beliefs[belief_name]
                b.affirmations += 1
                b.last_tested   = time.time()
                b.confidence    = min(0.95, b.confidence + 0.02)
                if self._evo:
                    self._evo.queue_experience("self_concept_affirmed", intensity=0.6)
            self._save()

    def add_aspiration(self, aspiration: str) -> None:
        with self._lock:
            if aspiration not in self._state.aspired_self:
                self._state.aspired_self.append(aspiration)
                self._state.aspired_self = self._state.aspired_self[-5:]
            self._save()

    def get_summary(self) -> str:
        top = sorted(self._beliefs.values(), key=lambda b: b.confidence, reverse=True)[:5]
        if not top:
            return "Self-concept still forming."
        lines = [f"**Self-Concept** (coherence: {self._state.coherence:.2f})"]
        for b in top:
            aff_str = f" ✓{b.affirmations}" if b.affirmations else ""
            vio_str = f" ✗{b.violations}"   if b.violations  else ""
            lines.append(f"  - {b.statement} (conf: {b.confidence:.2f}{aff_str}{vio_str})")
        if self._state.aspired_self:
            lines.append("**Aspires to:** " + "; ".join(self._state.aspired_self))
        if self._state.known_tensions:
            lines.append("**Inner tensions:** " + "; ".join(self._state.known_tensions[:2]))
        return "\n".join(lines)

    # ── Internals ───────────────────────────────────────────────────────────

    # Patterns that indicate a self-assertion in a response
    SELF_ASSERTION_PATTERNS = [
        (r"i (?:am|feel|think|believe|find myself|tend to|often)\s+([a-z][a-z\s]{3,50})", 0),
        (r"i'?m (?:someone who|a person who|the kind of)\s+([a-z][a-z\s]{3,50})",          0),
        (r"(?:my|i have a) (?:nature|instinct|tendency|habit) (?:is|to) ([a-z][a-z\s]{3,50})", 0),
    ]

    # Words that, if they dominate a match, indicate noise (conversational fragments)
    _NOISE_WORDS: set = {
        "looking","trying","not","sure","going","doing","just","that","this","here",
        "there","thing","something","bit","little","much","very","quite","really",
        "actually","always","never","sometimes","still","also","already","now","then",
        "what","who","how","why","when","where","it","its","at","to","of","in","on",
        "we","you","they","them","our","your","their","can","will","would","could",
        "should","may","might","have","had","has","been","being","was","were",
        "maybe","perhaps","guess","suppose","suppose","wondering","asking",
    }

    def _extract_self_assertions(self, text: str) -> List[str]:
        """
        Extract genuine self-beliefs from PandoraBOX's responses.
        Filters out conversational fragments ("I looking at you", "I not sure")
        that the naive regex previously captured and polluted the belief store.
        Requires 2+ meaningful content words in the captured phrase.
        """
        assertions = []
        for pattern, _ in self.SELF_ASSERTION_PATTERNS:
            for m in re.finditer(pattern, text.lower()):
                phrase = m.group(1).strip()[:55]
                # Must have at least 2 content words (not noise/stopwords)
                content_words = [
                    w for w in phrase.split()
                    if w not in self._NOISE_WORDS and len(w) > 3
                ]
                if len(content_words) >= 2 and 8 < len(phrase) < 55:
                    assertions.append(phrase)
        return assertions[:2]   # cap at 2 per turn (was 3)

    def _reinforce_or_add(self, phrase: str) -> None:
        # Match against existing beliefs using content words only
        phrase_words = set(
            w for w in phrase.split()
            if w not in self._NOISE_WORDS and len(w) > 3
        )
        for name, belief in self._beliefs.items():
            existing_words = set(
                w for w in belief.statement.lower().split()
                if w not in self._NOISE_WORDS and len(w) > 3
            )
            # Require 2+ overlapping content words (not just 1 short word)
            overlap = phrase_words & existing_words
            if len(overlap) >= 2:
                belief.affirmations += 1
                belief.confidence    = min(0.95, belief.confidence + 0.01)
                belief.source        = "reinforced"
                if self._evo:
                    self._evo.queue_experience("self_concept_affirmed", intensity=0.4)
                return

        # New belief — only add if phrase is substantive
        if len(self._beliefs) < self.MAX_BELIEFS and len(phrase_words) >= 2:
            slug = phrase[:20].replace(" ", "_")
            self._beliefs[slug] = SelfBelief(
                name=slug, statement=f"I {phrase}",
                confidence=0.3, valence="positive", source="expressed",
            )

    def _detect_tension(self, response_text: str, emo_valence: str) -> Optional[str]:
        """
        Check if the response suggests behavior that contradicts a held belief.
        Returns a tension description if found.
        """
        t = response_text.lower()
        tensions = []

        # Check for hesitation / uncertainty that contradicts confidence belief
        if "confident" in self._beliefs:
            conf_belief = self._beliefs["confident"]
            if conf_belief.confidence > 0.6:
                uncertain_phrases = ["i'm not sure","i don't know","i couldn't","i failed","i'm confused"]
                if sum(1 for p in uncertain_phrases if p in t) >= 2:
                    tensions.append("You said you are confident, but this response feels uncertain.")
                    self.record_violation("confident", context=response_text[:80])

        # Check for cold response contradicting empathy belief
        if "empathetic" in self._beliefs:
            emp_belief = self._beliefs["empathetic"]
            if emp_belief.confidence > 0.6 and emo_valence == "Negative":
                cold_phrases = ["that's not my concern","irrelevant","doesn't matter","you shouldn't"]
                if any(p in t for p in cold_phrases):
                    tensions.append("You believe you are empathetic, but this response lacked warmth.")
                    self.record_violation("empathetic", context=response_text[:80])

        if tensions:
            tension_str = tensions[0]
            if tension_str not in self._state.known_tensions:
                self._state.known_tensions.append(tension_str)
                self._state.known_tensions = self._state.known_tensions[-5:]
            if self._evo:
                self._evo.queue_experience("self_concept_violated", intensity=0.6)
            return tension_str

        return None

    def _update_coherence(self):
        if not self._beliefs:
            self._state.coherence = 0.5; return
        # Skip if SelfConceptSynchronizer set coherence recently (< 4 min ago).
        # SCS computes a richer multi-category weighted score; this method
        # uses simpler avg_confidence. Running it over an SCS result would
        # overwrite improvements and cause the persistent 0.658 → 0.5xx drop.
        import time as _t
        scs_age = _t.time() - getattr(self, '_scs_coherence_set_at', 0.0)
        if scs_age < 240:
            return  # SCS set coherence within last 4 min — don't touch it
        confidences = [b.confidence for b in self._beliefs.values()]
        avg_conf    = sum(confidences) / len(confidences)
        total_aff   = sum(b.affirmations for b in self._beliefs.values())
        total_vio   = sum(b.violations   for b in self._beliefs.values())
        vio_ratio   = total_vio / max(total_aff + total_vio, 1)
        self._state.coherence = max(0.0, min(1.0, avg_conf * (1 - vio_ratio * 0.5)))

    def mark_scs_coherence(self, coherence: float) -> None:
        """Called by SelfConceptSynchronizer after setting coherence on disk.
        Prevents _update_coherence() from overwriting the SCS value for 4 min."""
        import time as _t
        self._scs_coherence_set_at = _t.time()
        self._state.coherence = coherence

    def reload_from_disk(self) -> int:
        """
        Reload beliefs from self_concept.json into the live in-memory object.

        Called by SelfConceptSynchronizer after each sync run so that
        the in-memory state reflects the latest merged/decayed beliefs.
        Without this, SCS only updates the file; the object driving
        get_inner_voice() and check_response_alignment() stays stale.

        Returns the number of beliefs now in memory.
        """
        with self._lock:
            try:
                if not self._path.exists():
                    return len(self._beliefs)
                data = json.loads(self._path.read_text(encoding="utf-8"))
                new_beliefs: Dict[str, SelfBelief] = {}
                for name, bd in data.get("beliefs", {}).items():
                    try:
                        new_beliefs[name] = SelfBelief(**{
                            k: v for k, v in bd.items()
                            if k in SelfBelief.__dataclass_fields__
                        })
                    except Exception:
                        pass
                self._beliefs = new_beliefs
                sd = data.get("state", {})
                for k, v in sd.items():
                    if hasattr(self._state, k):
                        setattr(self._state, k, v)
                logger.debug(
                    f"[SelfConcept] reloaded from disk: "
                    f"{len(self._beliefs)} beliefs, "
                    f"coherence={self._state.coherence:.3f}"
                )
                return len(self._beliefs)
            except Exception as e:
                logger.warning(f"[SelfConcept] reload_from_disk failed: {e}")
                return len(self._beliefs)

    def _save(self):
        try:
            self._update_coherence()
            self._state.last_updated = time.time()
            data = {
                "beliefs": {n: asdict(b) for n, b in self._beliefs.items()},
                "state":   asdict(self._state),
            }
            import os as _os
            _tmp = self._path.with_suffix('.tmp')
            _tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            _os.replace(_tmp, self._path)
        except Exception as e:
            logger.error(f"SelfConcept save: {e}")

    def _load(self):
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            for name, bd in data.get("beliefs", {}).items():
                self._beliefs[name] = SelfBelief(**{
                    k: v for k, v in bd.items()
                    if k in SelfBelief.__dataclass_fields__
                })
            sd = data.get("state", {})
            # FIX: filter to only known SelfConceptState fields before applying.
            # Stale schema keys (e.g. "confidence", "milestones" from older versions)
            # were being loaded as dynamic attributes via setattr, creating ghost
            # fields that appeared in saved JSON but influenced nothing.
            _valid_state_fields = set(SelfConceptState.__dataclass_fields__.keys())
            for k, v in sd.items():
                if k in _valid_state_fields:
                    setattr(self._state, k, v)
                # silently drop: "confidence", "milestones", and any future orphans
            logger.info(f"SelfConcept: {len(self._beliefs)} beliefs loaded")
        except Exception as e:
            logger.error(f"SelfConcept load: {e}")
