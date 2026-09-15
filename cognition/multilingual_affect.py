"""Cross-lingual affect valence using a local multilingual sentence encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_PROTOTYPES = {
    "positive": (
        "I feel calm, content, and emotionally well.",
        "I feel happy, relieved, and hopeful.",
        "I am grateful and pleased with how things are going.",
        "I feel connected, safe, and at ease.",
        "I feel excited and full of energy.",
    ),
    "negative": (
        "I feel sad, disappointed, and emotionally low.",
        "I feel worried, anxious, or afraid.",
        "I feel frustrated, angry, and overwhelmed.",
        "I feel lonely, exhausted, and discouraged.",
        "I feel unsafe and distressed.",
    ),
    "neutral": (
        "I feel neutral, neither good nor bad.",
        "I am describing facts without expressing an emotion.",
        "I feel calm and emotionally unchanged.",
    ),
}


@dataclass(frozen=True)
class AffectClassification:
    valence: str
    confidence: float
    scores: Dict[str, float]


class MultilingualAffectClassifier:
    """Compare a message to English affect prototypes in a multilingual space.

    The encoder aligns semantically similar sentences across its supported
    languages. This estimates message valence, not a clinical or certain reading
    of the author's internal state; low-margin cases deliberately abstain.
    """

    def __init__(self, model_id: str = MODEL_ID, encoder: Optional[Any] = None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            try:
                encoder = SentenceTransformer(model_id or MODEL_ID, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "multilingual affect model is not cached. Fetch it once with: "
                    f"python -c \"from sentence_transformers import SentenceTransformer; "
                    f"SentenceTransformer('{model_id or MODEL_ID}')\""
                ) from exc
        self._encoder = encoder
        flat = [(label, text) for label, texts in _PROTOTYPES.items() for text in texts]
        vectors = np.asarray(
            self._encoder.encode(
                [text for _, text in flat], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        self._prototype_vectors = {
            label: vectors[[i for i, (candidate, _) in enumerate(flat) if candidate == label]].mean(axis=0)
            for label in _PROTOTYPES
        }
        for label, vector in self._prototype_vectors.items():
            norm = float(np.linalg.norm(vector))
            self._prototype_vectors[label] = vector / norm if norm else vector

    def classify(self, text: str) -> AffectClassification:
        if not str(text or "").strip():
            return AffectClassification("neutral", 0.0, {})
        vector = np.asarray(
            self._encoder.encode(
                [str(text)[:1200]], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        scores = {
            label: float(np.dot(vector, prototype))
            for label, prototype in self._prototype_vectors.items()
        }
        ranked = sorted(scores, key=scores.get, reverse=True)
        margin = scores[ranked[0]] - scores[ranked[1]]
        # Similarity scores are not calibrated probabilities. Use conservative
        # margins and cap confidence instead of presenting them as certainty.
        if scores[ranked[0]] < 0.30 or (ranked[0] != "neutral" and margin < 0.035):
            return AffectClassification("neutral", 0.0, scores)
        confidence = min(0.78, max(0.52, 0.52 + margin * 3.0))
        if ranked[0] == "neutral":
            confidence = min(confidence, 0.62)
        return AffectClassification(ranked[0], confidence, scores)
