"""
utils/shared_embedder.py

Two-tier embedding stack with a task router.

  - FAST tier (default): local all-MiniLM-L6-v2 via sentence-transformers.
    In-process, lowest latency (384-dim). Use for HOT PATHS that run every
    cycle: memory search / FAISS, attention relevance, insight anchors.

  - QUALITY tier: nomic-embed-text-v1.5 via an OpenAI-compatible
    /v1/embeddings endpoint (LM Studio). Higher semantic fidelity (768-dim).
    Use for BATCH / SEMANTIC-CRITICAL work where a few hundred ms of API
    latency is fine: dedup, pruning, relevance ranking, coherence scoring.

IMPORTANT (dimension safety): the two tiers produce DIFFERENT dimensions
(MiniLM 384 vs nomic 768). Never mix vectors from the two tiers inside a
single computation, and never query a FAISS index built with one tier using
the other. Each call site must pick ONE tier and stay consistent with it.

Usage:
    from utils.shared_embedder import get_embedder
    emb = get_embedder()                                   # FAST (MiniLM)
    emb = get_embedder(quality=True)                        # QUALITY (nomic)
    emb = get_embedder("text-embedding-nomic-embed-text-v1.5", quality=True)

Both tiers expose a sentence-transformers-compatible interface:
    .encode(text_or_list) -> np.ndarray          # str -> (dim,), list -> (N,dim)
    .get_sentence_embedding_dimension() -> int

The quality tier AUTO-FALLS-BACK to the fast tier if the API is unavailable,
so callers never break when LM Studio / the embed model is down.
"""
from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

# Singletons
_fast_instance = None
_fast_name = ""
_quality_instance = None
_quality_name = ""

# Defaults
DEFAULT_FAST_MODEL    = "all-MiniLM-L6-v2"
DEFAULT_QUALITY_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_BASE_URL      = "http://localhost:1234/v1"


def _base_url_from_config():
    """Resolve the embeddings API base URL from config, with a safe default."""
    try:
        from managers.settings_manager import config
        for attr in ("EMBED_API_BASE_URL", "LLM_BASE_URL"):
            v = getattr(config, attr, "")
            if v:
                return str(v)
    except Exception:
        pass
    return DEFAULT_BASE_URL


def _quality_model_from_config():
    """Resolve the quality embedding model name from config, with a default."""
    try:
        from managers.settings_manager import config
        for attr in ("QUALITY_EMBED_MODEL", "MEMORY_COGNEE_EMBED_MODEL"):
            v = getattr(config, attr, "")
            if v and "embed" in str(v).lower():
                return str(v)
    except Exception:
        pass
    return DEFAULT_QUALITY_MODEL


class APIEmbedder:
    """Embedder backed by an OpenAI-compatible /v1/embeddings endpoint.

    sentence-transformers-compatible surface (.encode, .get_sentence_embedding_dimension).
    Falls back to the FAST tier on any API failure so callers never crash.
    """

    def __init__(self, model_name, base_url, timeout=60):
        self.model_name = model_name
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._dim = None

    def _post(self, texts):
        body = json.dumps({"model": self.model_name, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _encode_list(self, texts, batch_size=32):
        import numpy as np
        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            d = self._post(chunk)
            data = sorted(d.get("data", []), key=lambda x: x.get("index", 0))
            for item in data:
                out.append(item["embedding"])
        return np.asarray(out, dtype="float32")

    def encode(self, texts, **kw):
        single = isinstance(texts, str)
        try:
            mat = self._encode_list([texts] if single else list(texts))
            return mat[0] if single else mat
        except Exception as e:
            logger.warning(
                "[embedder] QUALITY tier API failed (%s); "
                "falling back to FAST tier for this call" % e
            )
            fast = _get_fast_embedder()
            if fast is None:
                raise
            return fast.encode(texts, **kw)

    def get_sentence_embedding_dimension(self):
        if self._dim is None:
            v = self.encode(["dimension calibration probe"])
            self._dim = int(v.shape[-1] if v.ndim > 1 else len(v))
        return self._dim


def _get_fast_embedder(model_name=DEFAULT_FAST_MODEL):
    """Return (creating on first use) the shared local SentenceTransformer."""
    global _fast_instance, _fast_name
    if _fast_instance is not None and _fast_name == model_name:
        return _fast_instance
    try:
        from sentence_transformers import SentenceTransformer
        try:
            _fast_instance = SentenceTransformer(model_name, local_files_only=True)
            logger.info("[embedder] FAST tier loaded from cache: %s" % model_name)
        except Exception:
            _fast_instance = SentenceTransformer(model_name)
            logger.info("[embedder] FAST tier downloaded: %s" % model_name)
        _fast_name = model_name
        return _fast_instance
    except ImportError:
        logger.warning(
            "[embedder] sentence-transformers not installed - FAST tier unavailable. "
            "Install: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.error("[embedder] FAST tier init failed: %s" % e)
        return None


def get_embedder(model_name=DEFAULT_FAST_MODEL, quality=False):
    """Return the appropriate embedder for the requested tier.

    quality=False (default) -> FAST tier (local MiniLM), lowest latency.
    quality=True            -> QUALITY tier (nomic via LM Studio API); if the
                               API cannot be reached it transparently falls back
                               to the FAST tier, so callers never break.

    Backward compatible: get_embedder("some-model") still returns a local
    SentenceTransformer for that model (FAST tier).
    """
    global _quality_instance, _quality_name

    if quality:
        if model_name and "embed" in model_name.lower():
            qmodel = model_name
        else:
            qmodel = _quality_model_from_config()
        if _quality_instance is not None and _quality_name == qmodel:
            return _quality_instance
        try:
            _quality_instance = APIEmbedder(qmodel, _base_url_from_config())
            _quality_name = qmodel
            logger.info("[embedder] QUALITY tier ready: %s" % qmodel)
            return _quality_instance
        except Exception as e:
            logger.warning("[embedder] QUALITY tier init failed (%s); using FAST tier" % e)

    return _get_fast_embedder(model_name)


def get_fast_embedder(model_name=DEFAULT_FAST_MODEL):
    return _get_fast_embedder(model_name)


def get_quality_embedder(model_name=""):
    return get_embedder(model_name, quality=True)
