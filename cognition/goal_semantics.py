"""
cognition/goal_semantics.py

Embedding-based semantic scoring for the goal system.

This module replaces static keyword-dictionary matching with real semantic
similarity. It is the shared backing for:

  - cognition/goal_quality_filter.py  (tension alignment scoring)
  - cognition/goal_engine.py          (goal-topic quality gate)

Model stack (see utils/shared_embedder.py):
  1. QUALITY tier  -> text-embed-nomic-embed-text-v1.5 via LM Studio
                      (http://localhost:1234/v1/embeddings), 768-dim.
  2. FAST tier     -> local all-MiniLM-L6-v2 (sentence-transformers), 384-dim,
                      used automatically when the LM Studio API is down.
  3. Neither       -> functions here return None so the CALLER falls back to
                      its original static dictionary logic.

Dimension safety: every public function produces ALL of its vectors inside a
SINGLE encode() batch, so all vectors in one computation always share the same
tier and dimension. We never compare a vector against one produced by a
different call (which could have used a different tier).

All public functions are exception-safe: any failure returns None (or an
empty dict) rather than raising, so a goal-cycle never crashes because the
embeddings backend is unavailable.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def embed(texts: Sequence[str]) -> Optional[np.ndarray]:
    """Encode a list of texts on the QUALITY tier (auto-fallback to FAST).

    Returns an (N, dim) float32 matrix, or None if no embedding tier is
    reachable. A single batched call guarantees every row shares one tier and
    dimension (dimension-safety).
    """
    try:
        from utils.shared_embedder import get_embedder
        emb = get_embedder(quality=True)
        mat = emb.encode([str(t) for t in texts])
        mat = np.asarray(mat, dtype="float32")
        if mat.ndim == 1:
            mat = mat[None, :]
        return mat
    except Exception as e:  # noqa: BLE001 - never break a goal cycle
        logger.warning(
            "[goal_semantics] embeddings unavailable (%s); "
            "callers should use dictionary fallback" % e
        )
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def max_cosine(query: np.ndarray, refs: Sequence[np.ndarray]) -> float:
    """Max cosine similarity between a query vector and a set of ref vectors."""
    best = 0.0
    for r in refs:
        c = _cosine(query, r)
        if c > best:
            best = c
    return best


# Calibration scale for semantic tension alignment.
#
# The ORIGINAL static scorer used `tension_level * 0.3` for a theme-word match
# (a binary "matched" mapped to a fixed 0.3). A nomic cosine for a genuinely
# aligned pair is typically ~0.5, so a raw `tension_level * cosine` is roughly
# 2x stronger than the static baseline — that would let a single-word noise
# goal's tension bonus drown out the -0.2 single-word penalty and get it
# *boosted*. To keep the semantic path on the SAME scoring scale as the static
# fallback (so the rest of the scorer stays balanced), we map cosine onto the
# static "match strength" scale: cosine 0.5 -> 0.3 (the old "matched" value).
#   TENSION_ALIGN_SCALE = 0.3 / 0.5 = 0.6
# Strong matches (cosine > 0.5) still score above the static baseline, so the
# semantic path remains strictly better at capturing real alignment — it just
# no longer overpowers the other scoring signals.
TENSION_ALIGN_SCALE = 0.6


# ── Semantic anchors ──────────────────────────────────────────────────────────
# Natural-language concept anchors, one per tension key. Used as the semantic
# reference for embedding-based tension alignment in the goal quality filter.
# Keys MUST match the tension keys produced by DataAccess.get_tensions().
TENSION_CONCEPTS = {
    "curiosity_drive": (
        "curiosity and inquiry: the drive to understand, explore, learn, "
        "discover, investigate, and research new knowledge"
    ),
    "identity_stress": (
        "identity and self: questions of personal values, purpose, meaning, "
        "integrity, and consciousness"
    ),
    "goal_pressure": (
        "goal progress: completing, achieving, resolving, and advancing toward "
        "meaningful objectives"
    ),
    "contradiction_pressure": (
        "resolving contradictions: reconciling, integrating, clarifying, and "
        "aligning conflicting beliefs or demands"
    ),
    "social_drive": (
        "social connection: relating to others, empathy, understanding people, "
        "and meaningful engagement"
    ),
    "knowledge_uncertainty": (
        "knowledge gaps: clarifying and resolving uncertainty by learning, "
        "researching, and testing assumptions"
    ),
}

# Diverse, meaningful, goal-worthy topic exemplars (cross-domain) and generic
# noise exemplars, used as semantic anchors for the goal engine's topic gate.
GOOD_TOPIC_EXEMPLARS = [
    "learning to build a small web application",
    "improving my sourdough bread baking technique",
    "understanding how consciousness emerges from cognition",
    "planning a trip to Japan",
    "writing a short story about memory",
    "researching the history of fractals and mathematics",
    "practicing guitar fingerpicking every day",
    "designing a garden for pollinators",
    "learning Spanish conversation",
    "building a home automation system",
]
NOISE_TOPIC_EXEMPLARS = [
    "something to do",
    "just thinking about things",
    "random stuff",
    "the thing about yesterday",
    "going to get some coffee",
    "whatever comes up",
    "a lot of things going on",
    "not sure what to do next",
]


# ── Public semantic scorers ───────────────────────────────────────────────────

def batch_tension_alignment(
    goal_names: Sequence[str],
    tensions: Dict[str, float],
    min_tension: float = 0.3,
) -> Optional[Dict[str, Tuple[float, str]]]:
    """Embedding-based tension alignment for many goals in ONE embedding batch.

    For each goal, computes the best (max) alignment across active tensions:
        alignment = tension_level * cosine(goal_name, tension_concept_anchor)
    (clamped to >= 0). This mirrors the static scorer's shape
    (`tension_level * 0.3`) but uses real semantic similarity.

    Returns {goal_name: (best_alignment, best_tension_key)} or None if
    embeddings are unavailable (caller should use its dictionary fallback).
    """
    names = [str(g) for g in goal_names]
    if not names:
        return {}

    active = {
        k: float(v)
        for k, v in tensions.items()
        if k in TENSION_CONCEPTS and float(v) >= min_tension
    }
    if not active:
        return {g: (0.0, "") for g in names}

    anchor_keys = list(active.keys())
    texts = list(names) + [TENSION_CONCEPTS[k] for k in anchor_keys]
    mat = embed(texts)
    if mat is None:
        return None

    goal_vecs = mat[: len(names)]
    anchor_vecs = mat[len(names):]

    out: Dict[str, Tuple[float, str]] = {}
    for i, g in enumerate(names):
        best = 0.0
        best_key = ""
        for j, key in enumerate(anchor_keys):
            sim = max(0.0, _cosine(goal_vecs[i], anchor_vecs[j]))
            alignment = float(active[key]) * sim * TENSION_ALIGN_SCALE
            if alignment > best:
                best = alignment
                best_key = key
        out[g] = (best, best_key)
    return out


def batch_topic_quality(
    topics: Sequence[str],
) -> Optional[Dict[str, Tuple[float, float, bool]]]:
    """Embedding-based "meaningfulness" score for many topics in ONE batch.

    For each topic, computes:
        good_sim  = max cosine(topic, GOOD_TOPIC_EXEMPLARS)
        noise_sim = max cosine(topic, NOISE_TOPIC_EXEMPLARS)
    A topic is considered goal-worthy (pass=True) when it is at least as close
    to good exemplars as to noise (with a small tolerance for nomic's
    compressed similarity band), and has a minimum meaningfulness:
        pass = (good_sim - noise_sim) > -0.03  AND  good_sim > 0.10

    The relative (good >= noise) criterion is the primary signal — it is
    robust to absolute similarity-scale differences between nomic and MiniLM.
    The -0.03 tolerance is deliberate: the goal pool was previously TOO narrow
    (a 30-word whitelist), so we bias slightly toward acceptance and let the
    downstream quality filter do the fine discrimination.

    Returns {topic: (good_sim, noise_sim, pass)} or None if embeddings are
    unavailable (caller should use its dictionary fallback).
    """
    topics = [str(t) for t in topics]
    if not topics:
        return {}

    texts = list(topics) + GOOD_TOPIC_EXEMPLARS + NOISE_TOPIC_EXEMPLARS
    mat = embed(texts)
    if mat is None:
        return None

    n = len(topics)
    topic_vecs = mat[:n]
    good_vecs = mat[n : n + len(GOOD_TOPIC_EXEMPLARS)]
    noise_vecs = mat[n + len(GOOD_TOPIC_EXEMPLARS):]

    out: Dict[str, Tuple[float, float, bool]] = {}
    for i, t in enumerate(topics):
        good_sim = max_cosine(topic_vecs[i], good_vecs)
        noise_sim = max_cosine(topic_vecs[i], noise_vecs)
        ok = (good_sim - noise_sim) > -0.03 and good_sim > 0.10
        out[t] = (float(good_sim), float(noise_sim), bool(ok))
    return out
