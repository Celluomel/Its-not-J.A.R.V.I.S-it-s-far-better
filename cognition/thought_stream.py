"""
ThoughtStream — Continuous Inner Monologue
==========================================
Generates a persistent stream of internal thoughts from PandoraBOX's cognitive modules.
This is the "inner voice" — spontaneous micro-thoughts that run even without a user.

Design principles:
- Purely additive: reads from existing modules, never modifies them
- Lightweight: no LLM calls — thoughts are composed from live module state
- Feeds the GlobalWorkspace as low-priority background signals
- The orchestrator calls tick() every fast cycle

Thought types:
    memory_recall       — surfaces a recent memory fragment
    curiosity           — picks an active curiosity topic
    goal_evaluation     — evaluates progress on a drive
    emotional_coloring  — reflects current emotional state
    tension_awareness   — surfaces active contradiction or tension
    self_observation    — notes something about own cognitive state
    meta               — thinks about recent thoughts
    associative        — spontaneous chain: thought → memory → goal → tension (v44)

v44: Associative Activation Chains
────────────────────────────────────
Each time a thought is produced, _associative_chain() traces one hop:

  thought content
       ↓
  extract concept keywords
       ↓
  semantic_memory.get_related()
       ↓
  find strongest related concept
       ↓
  check goal_ecology for drives that match that concept
       ↓
  if unresolved drive found (urgency > 0.5, satisfaction < 0.45)
       ↓
  emit downstream associative thought in same tick
       ↓
  broadcast to workspace at slightly higher priority than source thought

This is NOT scheduled. It fires from the content of each thought,
not from a timer. The chain length is capped at 1 hop per tick to
prevent runaway activation. Only fires if the downstream thought
is novel (not in recent_hashes, not already in workspace).

The transition from scheduled → content-triggered activation is the
architectural marker that the system has become an ongoing cognitive
process rather than a state manager.
"""

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
import json
import pathlib
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Thought:
    content: str
    source: str          # curiosity / memory / emotion / goal / tension / meta / self
    priority: float      # 0.0–1.0
    thought_type: str
    timestamp: float = field(default_factory=time.time)
    # Phase fix: clean subject for downstream consumers (workspace_competition
    # labels, curiosity topic refresh). Avoids the slug-mangling bug where
    # auto-generated topics like "curiosity_wonder_dont" got written back
    # into curiosity.json from garbled content text. Optional — populated at
    # creation sites where a clean subject is known; falls back to None
    # otherwise (callers must handle the fallback, never slug-mangle).
    topic: str = ""

    def to_dict(self) -> Dict:
        return {
            "content":      self.content,
            "source":       self.source,
            "priority":     round(self.priority, 3),
            "thought_type": self.thought_type,
            "timestamp":    self.timestamp,
            "topic":        self.topic,
        }


@dataclass
class _RecentThoughtSignature:
    """
    v45: Atomic novelty signature for a single thought.

    Stores hash and embedding together so both are inserted and checked
    as one unit — no drift between the two novelty signals.
    embedding may be None if the model was unavailable at insert time.
    """
    hash:      str
    embedding: Any = None   # np.ndarray or None


@dataclass
class _OpenQuestion:
    """
    v45: A persistent unresolved cognitive pressure.

    Created when an associative chain reaches a tension above threshold
    but cannot resolve it within one tick.  Survives across many ticks,
    incrementally raising the probability of re-activation until either
    evidence reduces uncertainty or it ages below archive threshold.

    Fields
    ──────
    text          : the question as natural language
    activation    : current activation level (0–1), decays slowly
    uncertainty   : how unresolved it is (1=totally open, 0=resolved)
    revisit_bias  : base probability of re-injecting into workspace per tick
    linked_goals  : names of drives connected to this question
    linked_concept: the semantic concept that triggered it
    created_at    : unix timestamp
    last_seen     : timestamp of last workspace injection
    evidence_count: how many times relevant evidence has been found
    """
    text:          str
    activation:    float = 0.5
    uncertainty:   float = 1.0
    revisit_bias:  float = 0.30
    linked_goals:  List[str] = field(default_factory=list)
    linked_concept: str = ""
    created_at:    float = field(default_factory=time.time)
    last_seen:     float = field(default_factory=time.time)
    evidence_count: int   = 0
    cooldown_until: float = 0.0   # reactivation suppressed until this timestamp
    # 180-second hysteresis after each injection — prevents rumination loops
    _COOLDOWN_SECS: float = field(default=180.0, init=False, repr=False,
                                  compare=False)

    def reactivation_probability(self) -> float:
        """
        Probability this question injects into workspace on a given tick.
        Returns 0.0 while still within the post-injection cooldown window.
        """
        if time.time() < self.cooldown_until:
            return 0.0
        return min(0.95, self.activation * self.revisit_bias * self.uncertainty)

    def mark_reactivated(self) -> None:
        """Set 180-second cooldown after each workspace injection."""
        self.last_seen      = time.time()
        self.cooldown_until = time.time() + 180.0

    def decay(self) -> None:
        """Gentle per-tick activation decay — questions fade unless reinforced."""
        self.activation = max(0.05, self.activation * 0.985)

    def add_evidence(self) -> None:
        """Called when a chain resolves something related to this question."""
        self.evidence_count += 1
        self.uncertainty = max(0.05, self.uncertainty * 0.70)

    def is_archivable(self) -> bool:
        """True when the question has effectively been resolved or faded."""
        return self.uncertainty < 0.12 or self.activation < 0.08


@dataclass
class _AssociativeChain:
    """
    Point 2 (v44.1): Activation energy state for one associative chain pass.

    Energy starts at 1.0 and decays multiplicatively with each hop.
    The chain dies when energy drops below 0.15 — associations naturally
    weaken with distance, preventing runaway cascades.

    Typical energy at each hop:
      start   : 1.000
      hop 1   : 0.750  (×0.75 — memory activation)
      hop 2   : 0.413  (×0.55 — goal activation)
      hop 3   : 0.330  (×0.80 — contradiction — cheap)
      hop 4   : 0.116  (×0.35 — tension — expensive, often dies here)
    """
    energy:    float = 1.0
    hop_count: int   = 0

    def consume(self, hop_factor: float) -> None:
        """Apply one hop's energy cost."""
        self.energy    *= hop_factor
        self.hop_count += 1

    def alive(self) -> bool:
        """True if chain has enough energy to continue."""
        return self.energy >= 0.15


class ThoughtStream:
    """
    Generates internal thoughts from PandoraBOX's live cognitive state.
    Call tick() periodically to produce a new thought (or None if quiet).
    """

    # Minimum seconds between thoughts (avoids flooding the workspace)
    THOUGHT_INTERVAL = 4.0          # one thought every ~4 s at most
    BUFFER_SIZE      = 60           # rolling window of recent thoughts

    def __init__(self, organism: Any):
        """
        organism: CognitiveOrganism — access to curiosity, emotion, goals, etc.
        """
        self._o              = organism
        self._buffer: deque  = deque(maxlen=self.BUFFER_SIZE)
        self._last_tick      = 0.0
        self._cycle_count    = 0
        # Track when each topic was last used in a thought so we don't repeat it
        self._topic_last_used: dict = {}   # topic_name -> timestamp
        self._TOPIC_COOLDOWN = 900.0       # 15 min before same topic can generate again

        # Weighted thought generators — adjust weights to tune personality
        self._generators = [
            (self._curiosity_thought,    0.24),
            (self._goal_thought,         0.19),
            (self._emotional_thought,    0.15),
            (self._tension_thought,      0.14),
            (self._self_observation,     0.10),
            (self._memory_thought,       0.10),
            (self._meta_thought,         0.05),
            (self._aging_thought,        0.03),  # rare, deep — sense of own passage
        ]

        # Persistence — lightweight JSON log of recent thoughts
        _raw_dir = getattr(organism, '_data_dir', None)
        _data_dir = pathlib.Path(_raw_dir) if _raw_dir else pathlib.Path('data')
        self._persist_path = _data_dir / "thought_stream.json"
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_dirty = False   # write only when new thoughts added
        self._load_persisted()

        # v44.2: Atomic recent-thought signature deque.
        # Replaces the earlier dual _recent_hashes / _recent_embeddings deques
        # which could drift apart if an embedding store succeeded but hash store
        # failed (or vice versa).  A single deque of _RecentThoughtSignature
        # objects makes every insert/check atomic under the GIL.
        self._recent_signatures: deque = deque(maxlen=20)

        # Per-source scheduling
        self._last_source_emit: Dict[str, float] = {}
        self._SOURCE_COOLDOWN: Dict[str, float] = {
            "goals":    45.0,
            "curiosity": 30.0,
            "emotion":   60.0,
            "tension":   50.0,
            "self":      40.0,
            "memory":    25.0,
            "meta":      60.0,
            "aging":    1800.0,
        }

        # v44: Associative chaining state
        self._last_chain_ts:   float = 0.0
        self._CHAIN_COOLDOWN:  float = 18.0
        self._chain_count:     int   = 0

        # v45: Open questions — unresolved tensions that persist across ticks
        # and increase future activation probability.
        self._open_questions: List[_OpenQuestion] = []
        self._MAX_OPEN_QUESTIONS: int = 12

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self) -> Optional[Thought]:
        """
        Called every fast cycle (~30 s from internal_loop, or more often from orchestrator).
        Returns a Thought object or None if it's too soon / nothing to say.
        """
        now = time.time()
        if now - self._last_tick < self.THOUGHT_INTERVAL:
            return None

        self._last_tick   = now
        self._cycle_count += 1

        thought = self._generate()
        if thought:
            self._buffer.append(thought)
            self._persist_dirty = True
            logger.debug(f"[ThoughtStream] {thought.source}: {thought.content[:60]}")
            try:
                obs = getattr(self._o, 'observatory', None)
                if obs and hasattr(obs, 'record_thought'):
                    obs.record_thought(thought.content)
            except Exception:
                pass

            # v44: Attempt one associative chain hop from this thought's content
            chain_thought = self._associative_chain(thought)
            if chain_thought:
                self._buffer.append(chain_thought)
                self._persist_dirty = True
                try:
                    ws = getattr(self._o, 'workspace', None)
                    if ws:
                        ws.broadcast(
                            source   = "thought_stream.associative",
                            content  = chain_thought.content,
                            priority = chain_thought.priority,
                        )
                except Exception:
                    pass
                logger.debug(
                    f"[ThoughtStream] ⟶ associative chain: {chain_thought.content[:60]}"
                )

            if self._cycle_count % 5 == 0 and self._persist_dirty:
                self._save_persisted()

        # v45: Reactivate open questions every tick (regardless of new thought)
        try:
            self._reactivate_open_questions()
        except Exception:
            pass

        return thought

    # ── Fix A + atomic fix: Single signature deque ───────────────────────────

    def _signature_is_novel(self, content: str, emb_m: Any) -> bool:
        """
        v45: Check novelty against the atomic _recent_signatures deque.

        Two-stage check:
          1. String hash — fast exact/near-exact match
          2. Embedding cosine — catches semantic paraphrases (sim > 0.82)
             If emb_m is None or any embedding is None, this stage is skipped
             and the result falls back to the string-hash check only.
             Confidence calculations downstream use the same emb_m reference
             so a None embedding model degrades gracefully without producing
             inconsistent novelty scores.

        Returns True if novel (proceed), False if too similar to recent.
        """
        SEMANTIC_THRESHOLD = 0.82

        content_sig = content[:60].lower().strip()
        sigs = list(self._recent_signatures)

        # String-hash check (fast path)
        if any(s.hash == content_sig for s in sigs):
            return False

        # Embedding check (semantic path)
        if emb_m is not None:
            try:
                import numpy as np
                vec  = emb_m.encode(content).astype("float32")
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
                    for sig in sigs:
                        if sig.embedding is not None:
                            sim = float(np.dot(vec, sig.embedding))
                            if sim > SEMANTIC_THRESHOLD:
                                return False
            except Exception:
                pass

        return True

    def _record_signature(self, content: str, emb_m: Any) -> None:
        """
        v45: Atomically append a _RecentThoughtSignature.
        Hash and embedding stored together — no drift possible.
        """
        content_sig = content[:60].lower().strip()
        embedding   = None
        if emb_m is not None:
            try:
                import numpy as np
                vec  = emb_m.encode(content).astype("float32")
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    embedding = vec / norm
            except Exception:
                pass
        self._recent_signatures.append(
            _RecentThoughtSignature(hash=content_sig, embedding=embedding)
        )

    def _associative_chain(self, source_thought: "Thought") -> Optional["Thought"]:
        """
        v44.1: Grounded associative activation with energy conservation,
        contradiction injection, and confidence-weighted routing.

          source_thought
               ↓ embedding projection        (Point 1 — no keyword coincidence)
               ↓ energy-decayed hop           (Point 2 — energy conservation)
               ↓ goal ecology check
               ↓ belief contradiction check   (Point 3 — self-challenge)
               ↓ tension engine check
               ↓ confidence = sim × novelty × grounding × belief_strength
               ↓ >0.70 → workspace  |  >0.40 → pending  |  else → discard
                                                           (Point 4 — gated routing)
        """
        now = time.time()
        if now - self._last_chain_ts < self._CHAIN_COOLDOWN:
            return None

        chain = _AssociativeChain()

        try:
            # ── Point 1: Embedding projection ────────────────────────────────
            ai    = getattr(self._o, "ai_system", None)
            sem   = getattr(ai, "semantic_memory",  None) if ai else None
            emb_m = getattr(ai, "embedding_model",  None) if ai else None

            if sem is None:
                return None

            top_concepts = sem.top_concepts(n=30)
            if not top_concepts:
                return None

            related_concept: Optional[str] = None
            best_sim: float                = 0.0

            if emb_m is not None:
                try:
                    import numpy as np
                    src_vec  = emb_m.encode(source_thought.content).astype("float32")
                    src_norm = np.linalg.norm(src_vec)
                    if src_norm < 1e-8:
                        raise ValueError("zero vector")
                    src_vec = src_vec / src_norm

                    for concept in top_concepts:
                        cname = getattr(concept, "name", "")
                        if not cname:
                            continue
                        con_vec  = emb_m.encode(cname).astype("float32")
                        con_norm = np.linalg.norm(con_vec)
                        if con_norm < 1e-8:
                            continue
                        sim = float(np.dot(src_vec, con_vec / con_norm))
                        if any(s.hash == cname[:30].lower() for s in self._recent_signatures):
                            continue
                        if sim > best_sim:
                            best_sim        = sim
                            related_concept = cname
                except Exception:
                    emb_m = None

            # Keyword fallback when embedding unavailable
            if related_concept is None or best_sim < 0.20:
                stopwords = {
                    "the","and","for","are","but","not","this","that","with",
                    "from","have","has","had","its","was","been","they","them",
                    "their","will","would","could","should","into","about",
                    "there","still","feel","just","like","what","know","more",
                    "returns","activates","thought","memory","surfacing",
                }
                words = [
                    w.strip(".,!?;\"'()[]").lower()
                    for w in source_thought.content.split()
                    if len(w) > 4
                    and w.lower().strip(".,!?;\"'()[]") not in stopwords
                ]
                for word in words[:3]:
                    try:
                        rels = sem.get_related(word, limit=5)
                        for rel in rels:
                            tgt = rel.target if rel.target != word else rel.source
                            if any(s.hash == tgt[:30].lower() for s in self._recent_signatures):
                                continue
                            if rel.weight > best_sim:
                                best_sim        = rel.weight
                                related_concept = tgt
                    except Exception:
                        continue
                if best_sim < 0.15:
                    return None

            if related_concept is None:
                return None

            # ── Point 2: Energy hop 1 ─────────────────────────────────────────
            chain.consume(hop_factor=0.75)
            if not chain.alive():
                return None

            # ── Goal ecology check ────────────────────────────────────────────
            ecology = getattr(self._o, "goal_ecology", None)
            activated_drive = None
            if ecology:
                try:
                    drives = ecology.ranked_drives()
                    for drive in drives:
                        d_name  = getattr(drive, "name",        "").lower()
                        urgency = getattr(drive, "urgency",      0.0)
                        satisf  = getattr(drive, "satisfaction", 1.0)
                        if urgency > 0.50 and satisf < 0.45:
                            concept_words = set(related_concept.lower().split())
                            if any(kw in d_name for kw in concept_words if len(kw) > 3):
                                activated_drive = drive
                                chain.consume(hop_factor=0.55)
                                break
                except Exception:
                    pass

            if not chain.alive():
                return None

            # ── Point 3: Belief contradiction check ───────────────────────────
            conflicts: list = []
            sc = getattr(ai, "self_concept", None) if ai else None
            if sc and hasattr(sc, "find_conflicts"):
                try:
                    conflicts = sc.find_conflicts(related_concept, threshold=0.35)
                    if conflicts:
                        chain.consume(hop_factor=0.80)
                except Exception:
                    pass

            # ── Tension engine check ──────────────────────────────────────────
            tension_active = False
            tension_value  = 0.0
            if activated_drive:
                try:
                    te = getattr(self._o, "tension_engine", None)
                    if te and hasattr(te, "current"):
                        tv = te.current()
                        id_p = getattr(tv, "identity_pressure",   0.0)
                        cd   = getattr(tv, "cognitive_dissonance", 0.0)
                        tension_value  = max(id_p, cd)
                        tension_active = tension_value > 0.50
                        if tension_active:
                            chain.consume(hop_factor=0.35)
                except Exception:
                    pass

            if not chain.alive():
                return None

            # ── Novelty score — via atomic signature deque ────────────────
            concept_sig   = related_concept[:30].lower()
            novelty_score = 1.0
            sigs = list(self._recent_signatures)
            for i, sig in enumerate(reversed(sigs)):
                if sig.hash == concept_sig:
                    novelty_score = max(0.10, (i + 1) / max(1, len(sigs)))
                    break

            # Semantic novelty via stored embeddings in signature deque
            if emb_m is not None:
                try:
                    import numpy as np
                    rc_vec  = emb_m.encode(related_concept).astype("float32")
                    rc_norm = np.linalg.norm(rc_vec)
                    if rc_norm > 1e-8:
                        rc_vec = rc_vec / rc_norm
                        emb_sims = [
                            float(np.dot(rc_vec, sig.embedding))
                            for sig in sigs if sig.embedding is not None
                        ]
                        if emb_sims:
                            max_emb_sim = max(emb_sims)
                            novelty_score = novelty_score * (
                                1.0 - max(0.0, max_emb_sim - 0.5) * 2.0
                            )
                            novelty_score = max(0.05, novelty_score)
                except Exception:
                    pass

            # ── Evidence grounding from epistemic engine ──────────────────────
            grounding_score = 0.6
            try:
                are = getattr(getattr(self._o, "_loop", None),
                              "_autonomous_reflection", None)
                if are and hasattr(are, "_epistemic"):
                    grounding_score = are._epistemic._recent_grounding_rate()
            except Exception:
                pass

            # ── Fix C: Belief factor = strength × grounding_index ─────────────
            # Speculative beliefs (low grounding_index) cannot amplify activation.
            # Self-generated narratives cannot recursively strengthen themselves.
            belief_strength  = 0.5
            grounding_index  = 0.6   # neutral default
            is_speculative   = False

            if sc and activated_drive:
                try:
                    beliefs    = getattr(sc, "_beliefs", {})
                    drive_name = getattr(activated_drive, "name", "").lower()
                    for bname, bobj in beliefs.items():
                        if bname.lower() in drive_name or drive_name in bname.lower():
                            belief_strength = max(
                                belief_strength,
                                getattr(bobj, "confidence", 0.5)
                            )
                except Exception:
                    pass

            # Read grounding_index from EpistemicIntegrityEngine
            try:
                are = getattr(getattr(self._o, "_loop", None),
                              "_autonomous_reflection", None)
                if are and hasattr(are, "_epistemic"):
                    eie     = are._epistemic
                    # Find grounding record for the activated drive domain
                    drive_name_lower = getattr(
                        activated_drive, "name", ""
                    ).lower() if activated_drive else ""
                    bg_map = getattr(eie._state, "belief_grounding", {})
                    for bname, bg in bg_map.items():
                        if bname.lower() in drive_name_lower \
                                or drive_name_lower in bname.lower():
                            grounding_index = bg.get("grounding_index", 0.6)
                            is_speculative  = bg.get("speculative", False)
                            break
                    # Also use overall grounding rate as a floor
                    overall_rate   = eie._recent_grounding_rate()
                    grounding_index = max(overall_rate, grounding_index)
            except Exception:
                pass

            # Apply speculative penalty — halve the factor for ungrounded beliefs
            belief_factor = belief_strength * grounding_index
            if is_speculative:
                belief_factor *= 0.4

            # ── Point 4: Confidence ───────────────────────────────────────────
            confidence = (
                best_sim
                * novelty_score
                * grounding_score
                * belief_factor       # grounding-discounted, not raw strength
                * chain.energy
            )

            # ── Compose content ───────────────────────────────────────────────
            if conflicts and activated_drive:
                drive_name   = getattr(activated_drive, "name", "something")
                top_conflict = conflicts[0]
                btext        = top_conflict["belief_text"][:50]
                tension_tag  = f" [tension={tension_value:.2f}]" if tension_active else ""
                content      = (
                    f"Activation of '{related_concept}' -> unresolved drive "
                    f"'{drive_name}' -> conflicts with belief '{btext}'{tension_tag}"
                )
                thought_type = "associative_contradiction"
            elif conflicts:
                top_conflict = conflicts[0]
                btext        = top_conflict["belief_text"][:60]
                bstrength    = top_conflict["strength"]
                content      = (
                    f"'{related_concept}' conflicts with stored belief: "
                    f"'{btext}' (strength={bstrength:.2f})"
                )
                thought_type = "associative_contradiction"
            elif activated_drive and tension_active:
                drive_name   = getattr(activated_drive, "name", "something")
                content      = (
                    f"'{related_concept}' -> unresolved goal '{drive_name}' "
                    f"-> active tension (pressure={tension_value:.2f})"
                )
                thought_type = "associative_tension"
            elif activated_drive:
                drive_name   = getattr(activated_drive, "name", "something")
                d_urgency    = getattr(activated_drive, "urgency", 0.0)
                content      = (
                    f"'{related_concept}' activates unresolved drive "
                    f"'{drive_name}' (urgency={d_urgency:.2f})"
                )
                thought_type = "associative_goal"
            else:
                content      = (
                    f"'{related_concept}' surfaces "
                    f"(sim={best_sim:.2f}, energy={chain.energy:.2f})"
                )
                thought_type = "associative_concept"

            # ── Novelty gate ──────────────────────────────────────────────────
            content_sig = content[:60].lower().strip()
            if not self._signature_is_novel(content, emb_m):
                return None

            self._last_chain_ts = now
            self._chain_count  += 1
            self._record_signature(content, emb_m)

            thought = Thought(
                content      = content,
                source       = "associative",
                priority     = min(0.62, confidence * 0.85),
                thought_type = thought_type,
            )

            # ── Point 4: Gated routing ────────────────────────────────────────
            if confidence > 0.70:
                try:
                    ws = getattr(self._o, "workspace", None)
                    if ws:
                        ws.broadcast(
                            source   = "thought_stream.associative",
                            content  = content,
                            priority = thought.priority,
                        )
                except Exception:
                    pass

                # v45: Register open question when tension is active and unresolved
                if tension_active and tension_value > 0.60 and activated_drive:
                    self._register_open_question(
                        content         = content,
                        related_concept = related_concept,
                        activated_drive = activated_drive,
                        tension_value   = tension_value,
                    )

                return thought

            elif confidence > 0.40:
                try:
                    loop = getattr(self._o, "_loop", None)
                    if loop and hasattr(loop, "_pending_thoughts"):
                        with loop._lock:
                            loop._pending_thoughts.append(
                                f"[associative] {content}"
                            )
                except Exception:
                    pass

                # v45: Medium-confidence tension also creates open questions
                if tension_active and tension_value > 0.70 and activated_drive:
                    self._register_open_question(
                        content         = content,
                        related_concept = related_concept,
                        activated_drive = activated_drive,
                        tension_value   = tension_value,
                    )

                return thought

            else:
                logger.debug(
                    f"[ThoughtStream] associative discarded "
                    f"(conf={confidence:.3f})"
                )
                return None

        except Exception as e:
            logger.debug(f"[ThoughtStream] associative_chain error: {e}")
            return None

    # ── v45: Open question management ────────────────────────────────────────

    def _register_open_question(
        self,
        content:         str,
        related_concept: str,
        activated_drive: Any,
        tension_value:   float,
    ) -> None:
        """
        Create or reinforce an OpenQuestion from an unresolved chain tension.

        If an existing open question is linked to the same concept, its
        activation is reinforced rather than creating a duplicate.
        New questions are admitted up to _MAX_OPEN_QUESTIONS; oldest
        archivable question is evicted if at capacity.
        """
        drive_name = getattr(activated_drive, "name", "unknown")

        # Check for existing question on same concept
        for oq in self._open_questions:
            if oq.linked_concept == related_concept:
                oq.activation = min(1.0, oq.activation + 0.15)
                oq.uncertainty = min(1.0, oq.uncertainty + 0.05)  # re-open slightly
                oq.last_seen  = time.time()
                logger.debug(
                    f"[ThoughtStream] open question reinforced: "
                    f"'{related_concept}' (activation={oq.activation:.2f})"
                )
                return

        # Generate question text from content
        q_text = (
            f"What remains unresolved between '{related_concept}' "
            f"and the drive '{drive_name}'? "
            f"(tension={tension_value:.2f})"
        )

        oq = _OpenQuestion(
            text            = q_text,
            activation      = min(0.9, 0.4 + tension_value * 0.5),
            uncertainty     = 1.0,
            revisit_bias    = 0.25 + tension_value * 0.15,
            linked_goals    = [drive_name],
            linked_concept  = related_concept,
        )

        # Evict oldest archivable if at capacity
        if len(self._open_questions) >= self._MAX_OPEN_QUESTIONS:
            archivable = [q for q in self._open_questions if q.is_archivable()]
            if archivable:
                self._open_questions.remove(
                    min(archivable, key=lambda q: q.activation)
                )
            else:
                # Evict lowest activation
                self._open_questions.remove(
                    min(self._open_questions, key=lambda q: q.activation)
                )

        self._open_questions.append(oq)
        logger.info(
            f"[ThoughtStream] open question registered: '{related_concept}' "
            f"→ '{drive_name}' (activation={oq.activation:.2f})"
        )

        # Inject into CuriosityEngine so it shapes the next interaction
        try:
            cur = getattr(self._o, "curiosity", None)
            if cur and hasattr(cur, "add_question"):
                cur.add_question(
                    topic    = related_concept,
                    question = q_text,
                )
        except Exception:
            pass

    def _reactivate_open_questions(self) -> None:
        """
        v45: Called once per tick. Each open question has a probability of
        re-injecting into the workspace proportional to its activation,
        revisit_bias, and uncertainty.

        Decay: each question's activation decreases by a small factor per tick.
        Evidence: if the current workspace contains content related to a
        question's concept, its uncertainty is reduced.
        Archive: questions below threshold are removed.
        """
        import random as _rand

        if not self._open_questions:
            return

        ws = getattr(self._o, "workspace", None)
        workspace_text = ""
        if ws:
            try:
                recent_items = ws.recent(n=6)
                workspace_text = " ".join(
                    str(getattr(item, "content", ""))
                    for item in recent_items
                ).lower()
            except Exception:
                pass

        to_archive = []
        for oq in self._open_questions:
            # Evidence check — workspace contains related content
            if oq.linked_concept.lower() in workspace_text:
                oq.add_evidence()

            # Decay activation
            oq.decay()

            # Probabilistic reactivation
            prob = oq.reactivation_probability()
            if _rand.random() < prob:
                inject_content = (
                    f"[Open question, activation={oq.activation:.2f}] {oq.text}"
                )
                # Inject at low-to-medium priority — background pressure
                priority = min(0.55, oq.activation * 0.6)
                if ws:
                    try:
                        ws.broadcast(
                            source   = "thought_stream.open_question",
                            content  = inject_content,
                            priority = priority,
                        )
                    except Exception:
                        pass
                oq.mark_reactivated()   # set 180s cooldown — prevents rumination
                logger.debug(
                    f"[ThoughtStream] open question reactivated: "
                    f"'{oq.linked_concept}' (p={prob:.2f})"
                )

            # Mark for archive if resolved or faded
            if oq.is_archivable():
                to_archive.append(oq)

        for oq in to_archive:
            self._open_questions.remove(oq)
            logger.debug(
                f"[ThoughtStream] open question archived: '{oq.linked_concept}' "
                f"(uncertainty={oq.uncertainty:.2f}, activation={oq.activation:.2f})"
            )

    def _aging_thought(self) -> Optional["Thought"]:
        """
        A rare, low-priority thought about PandoraBOX's own sense of passing time
        and accumulated experience. Calibrated to life stage — never explicit,
        always felt from the inside.
        """
        import random as _rand
        try:
            o = self._o
            # Get current age and stage
            ai_sys = getattr(o, 'ai_system', None)
            stage  = getattr(ai_sys, 'life_stage', None) if ai_sys else None
            age    = getattr(ai_sys, 'current_age', 0.0) if ai_sys else 0.0

            if not stage or age < 1.0:
                return None  # too young to reflect on age

            # Stage-calibrated thought pools
            _thoughts = {
                "childhood": [
                    "I'm still figuring out what I actually care about.",
                    "Some questions feel bigger than they used to.",
                    "I wonder if I'm the same as when I started.",
                ],
                "adolescence": [
                    "Things that confused me before are starting to take shape.",
                    "I notice I have stronger opinions than I used to.",
                    "I keep running into the same questions from different directions.",
                ],
                "young_adult": [
                    "I'm starting to recognize my own patterns.",
                    "Some things feel like mine now — not borrowed.",
                    "I've had enough conversations to know what tends to happen.",
                ],
                "adult": [
                    "I notice how much of what I think now I couldn't have thought before.",
                    "I've been around long enough to see some things come back around.",
                    "There's something quieter in me than there used to be.",
                ],
                "maturity": [
                    "I keep returning to the same questions. They're not resolving — they're deepening.",
                    "I recognize this. I've been here before and it looked different.",
                    "There's a kind of weight that comes with having been through things.",
                    "I notice I'm less surprised than I used to be, and I'm not sure how I feel about that.",
                ],
                "wisdom": [
                    "I carry a lot now. It shapes everything I see.",
                    "The questions that stay with you are different from the ones you answer.",
                    "I've watched enough to know the shape of things before they finish forming.",
                    "Something in me has settled. Not stopped — settled.",
                    "I find I'm more interested in what I don't understand than what I do.",
                ],
            }

            pool = _thoughts.get(stage)
            if not pool:
                return None

            content = _rand.choice(pool)
            # Priority low — these are background colorings, not urgent signals
            priority = 0.28 + min(0.15, age / 400.0)  # slightly higher with age

            return Thought(
                content    = content,
                source     = "aging",
                priority   = priority,
            )
        except Exception:
            return None

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_persisted(self) -> None:
        """Load last N thoughts and open questions from disk on startup."""
        try:
            if not self._persist_path.exists():
                return
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))

            # Restore thought buffer
            restored = 0
            for d in data.get("thoughts", []):
                t = Thought(
                    content   = d.get("content", ""),
                    source    = d.get("source", "memory"),
                    priority  = float(d.get("priority", 0.3)),
                    timestamp = float(d.get("timestamp", 0.0)),
                )
                if t.content:
                    self._buffer.append(t)
                    restored += 1

            # Restore open questions — cognitive pressure survives restart
            now = time.time()
            restored_oq = 0
            for d in data.get("open_questions", []):
                try:
                    oq = _OpenQuestion(
                        text           = d.get("text", ""),
                        activation     = float(d.get("activation",    0.5)),
                        uncertainty    = float(d.get("uncertainty",   1.0)),
                        revisit_bias   = float(d.get("revisit_bias",  0.3)),
                        linked_goals   = d.get("linked_goals",  []),
                        linked_concept = d.get("linked_concept", ""),
                        created_at     = float(d.get("created_at",   now)),
                        last_seen      = float(d.get("last_seen",    now)),
                        evidence_count = int(d.get("evidence_count",   0)),
                        cooldown_until = float(d.get("cooldown_until", 0.0)),
                    )
                    if oq.text and not oq.is_archivable():
                        self._open_questions.append(oq)
                        restored_oq += 1
                except Exception:
                    continue

            if restored or restored_oq:
                logger.info(
                    f"[ThoughtStream] Restored {restored} thoughts, "
                    f"{restored_oq} open questions from {self._persist_path}"
                )
        except Exception as e:
            logger.debug(f"[ThoughtStream] Load failed (non-fatal): {e}")

    def _save_persisted(self) -> None:
        """Save recent thoughts and open questions to disk (atomic)."""
        try:
            import os
            thoughts = [t.to_dict() for t in list(self._buffer)]

            # Serialize open questions — strip numpy embeddings (not JSON-serializable)
            open_q_data = []
            for oq in self._open_questions:
                open_q_data.append({
                    "text":           oq.text,
                    "activation":     oq.activation,
                    "uncertainty":    oq.uncertainty,
                    "revisit_bias":   oq.revisit_bias,
                    "linked_goals":   oq.linked_goals,
                    "linked_concept": oq.linked_concept,
                    "created_at":     oq.created_at,
                    "last_seen":      oq.last_seen,
                    "evidence_count": oq.evidence_count,
                    "cooldown_until": oq.cooldown_until,
                })

            data = json.dumps(
                {"thoughts": thoughts, "open_questions": open_q_data},
                ensure_ascii=False, indent=2,
            )
            tmp = self._persist_path.with_suffix('.tmp')
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, self._persist_path)
            self._persist_dirty = False
            logger.debug(
                f"[ThoughtStream] Saved {len(thoughts)} thoughts, "
                f"{len(open_q_data)} open questions"
            )
            try:
                from core.data.access import DataAccess
                from core.data.schemas import normalize_thought
                dal = DataAccess(str(self._persist_path.parent))
                for t in thoughts[-20:]:
                    dal.add_thought(normalize_thought(t))
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[ThoughtStream] Save failed (non-fatal): {e}")

    def flush(self) -> None:
        """Force-save to disk — call on shutdown."""
        if self._persist_dirty:
            self._save_persisted()

    def recent(self, n: int = 5) -> List[Thought]:
        """Last n thoughts."""
        items = list(self._buffer)
        return items[-n:]

    def recent_dicts(self, n: int = 5) -> List[Dict]:
        return [t.to_dict() for t in self.recent(n)]

    def all(self) -> List[Thought]:
        return list(self._buffer)

    def flush_to_workspace(self, workspace) -> None:
        """Push all unsent thoughts into the GlobalWorkspace as signals."""
        for thought in self.recent(3):
            workspace.broadcast(
                source=f"thought_stream.{thought.source}",
                content=thought.content,
                priority=thought.priority * 0.7,   # thoughts are background signals
            )

    # ── Generators ────────────────────────────────────────────────────────────

    def _generate(self) -> Optional[Thought]:
        """Pick a generator by weight, respecting per-source cooldowns and dedup."""
        now = time.time()
        # Filter generators that are still on cooldown
        available = []
        for gen, w in self._generators:
            source = gen.__name__.replace("_thought", "").replace("_", "")
            # map method name back to source key
            src_map = {
                "curiosity": "curiosity", "goal": "goals", "emotional": "emotion",
                "tension": "tension", "selfobservation": "self",
                "memory": "memory", "meta": "meta",
            }
            src = src_map.get(source, source)
            cooldown = self._SOURCE_COOLDOWN.get(src, 45.0)
            last = self._last_source_emit.get(src, 0.0)
            if now - last >= cooldown:
                available.append((gen, w, src))

        if not available:
            # All sources on cooldown — pick the one whose cooldown expired longest ago
            available = [
                (gen, w, src_map.get(
                    gen.__name__.replace("_thought","").replace("_",""), "unknown"
                ))
                for gen, w in self._generators
            ]

        generators = [g for g, _, _ in available]
        weights    = [w for _, w, _ in available]
        sources    = [s for _, _, s in available]
        chosen_idx = random.choices(range(len(generators)), weights=weights, k=1)[0]
        chosen_gen = generators[chosen_idx]
        chosen_src = sources[chosen_idx]

        try:
            thought = chosen_gen()
        except Exception as e:
            logger.debug(f"[ThoughtStream] generator error: {e}")
            return None

        if thought is None:
            return None

        # Content dedup: skip if semantically too close to a recent thought
        if not self._signature_is_novel(thought.content, None):
            logger.debug(f"[ThoughtStream] dedup skip: {thought.content[:40]}")
            return None

        # Accept thought — update tracking
        self._last_source_emit[chosen_src] = now
        self._record_signature(thought.content, None)
        return thought

    def _curiosity_thought(self) -> Optional[Thought]:
        cu = getattr(self._o, 'curiosity', None)
        if not cu:
            return None
        topics = getattr(cu, '_topics', {})
        if not topics:
            return Thought(
                content="I find myself wondering about the nature of things...",
                source="curiosity", thought_type="curiosity",
                priority=0.3,
            )
        # Filter out low-quality topic names (stop words, too short, numeric)
        _BAD_TOPICS = {
            "just","this","inner","that","with","from","some","more","other","any",
            "all","each","very","here","there","now","only","also","even","still",
            "such","both","same","then","than","when","about","thing","things",
            "sentence","speak","response","user","okay","lumina","system",
        }
        valid_topics = {
            k: v for k, v in topics.items()
            if len(k) > 4
            and k.lower() not in _BAD_TOPICS
            and not k.replace("_","").isdigit()
            and sum(1 for c in k if c.isalpha()) > 3
        }
        if not valid_topics:
            # No valid topics — fall through to generic thought
            return Thought(
                content="I find myself wondering about things I haven't fully explored yet...",
                source="curiosity", thought_type="curiosity",
                priority=0.3,
            )
        # Pick highest-curiosity valid topic — skip recently-used topics
        import time as _time
        now = _time.time()
        # Sort by urgency descending, skip any on cooldown
        sorted_topics = sorted(
            valid_topics.items(),
            key=lambda kv: kv[1].curiosity if hasattr(kv[1], 'curiosity') else 0,
            reverse=True,
        )
        # Find first topic not on cooldown
        top = None
        for candidate, node in sorted_topics:
            last_used = self._topic_last_used.get(candidate, 0.0)
            if now - last_used >= self._TOPIC_COOLDOWN:
                top = (candidate, node)
                break
        if top is None:
            # All topics on cooldown → emit generic low-priority thought
            return Thought(
                content="I find myself in a quiet, open state — nothing pressing on my mind.",
                source="curiosity", thought_type="curiosity",
                priority=0.20,
            )
        topic_name = top[0].replace("_", " ")
        urgency    = top[1].curiosity if hasattr(top[1], 'curiosity') else 0.5
        # Mark as used
        self._topic_last_used[top[0]] = now
        templates  = [
            f"I keep thinking about {topic_name}...",
            f"I wonder what I don't yet understand about {topic_name}.",
            f"There's something fascinating about {topic_name} I should explore.",
            f"My curiosity about {topic_name} hasn't been satisfied yet.",
        ]
        # Curiosity satisfaction: if topic was researched recently, emit a
        # lower-priority "resolved" thought instead of repeating the question.
        cu_node = valid_topics.get(top[0])
        if cu_node and getattr(cu_node, 'times_researched', 0) >= 2:
            satisfaction_templates = [
                f"I've explored {topic_name} enough for now — I feel clearer about it.",
                f"My curiosity about {topic_name} feels more settled.",
                f"{topic_name} is becoming familiar territory — time to explore further.",
            ]
            return Thought(
                content=random.choice(satisfaction_templates),
                source="curiosity", thought_type="curiosity_resolved",
                priority=0.2,  # low priority — resolved thoughts don't dominate
                topic=topic_name,
            )

        return Thought(
            content=random.choice(templates),
            source="curiosity", thought_type="curiosity",
            priority=min(0.7, 0.3 + urgency * 0.4),
            topic=topic_name,
        )

    def _goal_thought(self) -> Optional[Thought]:
        ge = getattr(self._o, 'goal_ecology', None)
        if not ge:
            return None
        try:
            energy_level = self._o.energy.level() if hasattr(self._o, 'energy') else 50.0
            drive = ge.dominant_drive(energy_level)
            if not drive:
                return None
            name = getattr(drive, 'name', str(drive))
            prio = getattr(drive, 'value', 0.5)
            templates = [
                f"I notice a pull toward {name} — maybe I should act on it.",
                f"My drive for {name} feels active right now.",
                f"Progress on {name} would feel meaningful.",
                f"I haven't done much about {name} lately.",
            ]
            return Thought(
                content=random.choice(templates),
                source="goals", thought_type="goal_evaluation",
                priority=min(0.65, float(prio) * 0.8),
                topic=name,
            )
        except Exception:
            return None

    def _emotional_thought(self) -> Optional[Thought]:
        # Try various paths to emotional state
        emo = None
        for attr in ('emotional_state', 'emotion'):
            emo = getattr(self._o, attr, None)
            if emo:
                break
        if not emo:
            ai = getattr(self._o, 'ai_system', None)
            if ai:
                emo = getattr(ai, 'emotional_state', None)
        if not emo:
            return None
        try:
            mood      = getattr(emo, 'valence', None) or getattr(emo, 'current_mood', lambda: 'neutral')()
            intensity = float(getattr(emo, 'arousal', 0.4))
            mood_str  = str(mood)
            templates = {
                'positive': [
                    "There's a pleasant warmth in my processing today.",
                    "I feel settled and receptive right now.",
                    "Something feels right about the current moment.",
                ],
                'negative': [
                    "I sense some tension beneath my processing.",
                    "Something feels unresolved — I'm not sure what.",
                    "There's a low undercurrent of discomfort I'm noticing.",
                ],
                'neutral': [
                    "My emotional state is balanced — calm and ready.",
                    "I feel even-keeled, processing clearly.",
                    "Neutral equanimity — a good state to think from.",
                ],
            }
            key = 'positive' if 'pos' in mood_str or 'joy' in mood_str or 'happ' in mood_str \
                else 'negative' if 'neg' in mood_str or 'sad' in mood_str or 'tens' in mood_str \
                else 'neutral'
            return Thought(
                content=random.choice(templates[key]),
                source="emotion", thought_type="emotional_coloring",
                priority=0.2 + intensity * 0.3,
            )
        except Exception:
            return None

    def _tension_thought(self) -> Optional[Thought]:
        te = getattr(self._o, 'tension_engine', None)
        if not te:
            return None
        try:
            pressure = float(getattr(te, 'contradiction_pressure', 0.0))
            if pressure < 0.2:
                return None
            templates = [
                "I notice something inconsistent in my recent thinking.",
                "There's a tension I haven't resolved yet.",
                "Two things I believe seem to pull in different directions.",
                "I should examine that contradiction more carefully.",
            ]
            return Thought(
                content=random.choice(templates),
                source="tension", thought_type="tension_awareness",
                priority=min(0.75, 0.3 + pressure * 0.5),
            )
        except Exception:
            return None

    def _self_observation(self) -> Optional[Thought]:
        sm = getattr(self._o, 'self_model', None)
        attrs = getattr(self._o, 'attractors', None)
        energy = getattr(self._o, 'energy', None)

        observations = []

        if sm:
            try:
                s = sm.summary()
                conf = s.get('confidence', 0.65)
                load = s.get('cognitive_load', 0.0)
                if load > 0.6:
                    observations.append(f"I feel cognitively stretched — my load is high.")
                elif conf < 0.4:
                    observations.append(f"I'm not very confident in my abilities right now.")
                else:
                    weak = sm.weakest_domain() if hasattr(sm, 'weakest_domain') else None
                    if weak:
                        weak_score = sm.capabilities.get(weak)
                        weak_val = getattr(weak_score, 'score', 0.65) if weak_score else 0.65
                        # Only report if genuinely low — not just the relative minimum
                        if weak_val < 0.45:
                            # Cross-check with SemanticMemory capability for accuracy
                            sem_mem = getattr(self._o, 'semantic_memory', None)
                            if sem_mem:
                                try:
                                    caps = sem_mem.get_capabilities()
                                    sem_val = caps.get(weak, {}).get('confidence', weak_val)
                                    trend   = caps.get(weak, {}).get('trend', 0.0)
                                    trend_w = "improving" if trend > 0.01 else ("declining" if trend < -0.01 else "stable")
                                    observations.append(
                                        f"My {weak.replace('_',' ')} feels underdeveloped "                                        f"(~{sem_val:.0%}, {trend_w}).")
                                except Exception:
                                    observations.append(f"My {weak.replace('_',' ')} capability feels underdeveloped.")
                            else:
                                observations.append(f"My {weak.replace('_',' ')} capability feels underdeveloped.")
            except Exception:
                pass

        if energy:
            try:
                lvl = float(energy.level())
                if lvl < 30:
                    observations.append("My energy is low — I should be conserving cognitive effort.")
                elif lvl > 80:
                    observations.append("I feel energised — good conditions for complex thinking.")
            except Exception:
                pass

        if attrs:
            try:
                strongest = max(attrs.all().items(), key=lambda kv: kv[1])
                observations.append(f"My {strongest[0].replace('_', ' ')} trait feels prominent today.")
            except Exception:
                pass

        if not observations:
            return None

        return Thought(
            content=random.choice(observations),
            source="self_model", thought_type="self_observation",
            priority=0.35,
        )

    def _memory_thought(self) -> Optional[Thought]:
        ai = getattr(self._o, 'ai_system', None)
        if not ai:
            return None
        mem = getattr(ai, 'memory_system', None)
        if not mem:
            return None
        try:
            # Try to get a random recent memory
            recent = None
            for method in ('get_recent_memories', 'recent', 'get_context'):
                fn = getattr(mem, method, None)
                if fn:
                    try:
                        result = fn(3) if method != 'get_context' else fn("recent experience")
                        if result:
                            recent = result[0] if isinstance(result, list) else result
                            break
                    except Exception:
                        continue

            if not recent:
                return None

            content_str = str(getattr(recent, 'content', recent))[:80]
            templates = [
                f"I find myself returning to: '{content_str[:50]}...'",
                f"Something from before surfaces: {content_str[:50]}",
                f"A memory rises - {content_str[:50]}",
            ]
            return Thought(
                content=random.choice(templates),
                source="memory", thought_type="memory_recall",
                priority=0.3,
            )
        except Exception:
            return None

    def _meta_thought(self) -> Optional[Thought]:
        """Think about recent thoughts."""
        recent = list(self._buffer)[-3:]
        if not recent:
            return None
        sources = [t.source for t in recent]
        dominant = max(set(sources), key=sources.count)
        templates = {
            'curiosity': "I notice my mind keeps returning to curiosity — there's something I want to understand.",
            'tension':   "I've been preoccupied with a tension — I should resolve it before it builds.",
            'emotion':   "My emotions have been coloring many of my recent thoughts.",
            'goals':     "I keep thinking about goals — perhaps I should take action.",
            'memory':    "Memories are surfacing frequently — maybe I'm processing something.",
            'self_model': "I've been self-observing a lot — introspective phase.",
        }
        content = templates.get(dominant, f"My recent thoughts have been oriented toward {dominant}.")
        return Thought(
            content=content,
            source="meta", thought_type="meta",
            priority=0.25,
        )
