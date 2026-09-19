import sys
"""
Memory Manager
- FAISS  : semantic vector search, persists to ~/.robot_agent/
- Dict   : simple recency buffer, zero deps
- Cognee : knowledge-graph memory (graph + vector unified)
           Requires pip install cognee
           Works with Ollama/LM Studio (no OpenAI key needed)
"""
import logging
import json
import time
import os
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Base
# ─────────────────────────────────────────────
class BaseMemoryBackend:
    def add_memory(self, content: str, metadata: Dict[str, Any] = None): pass
    def get_context(self, query: str, top_k: int = 5) -> str: return ""
    def clear(self): pass


# ─────────────────────────────────────────────
#  Dict (always-available fallback)
# ─────────────────────────────────────────────
class DictMemoryBackend(BaseMemoryBackend):
    """Simple recency-based in-memory store. Zero dependencies."""

    def __init__(self, max_entries: int = 100):
        self._store: List[Dict] = []
        self.max_entries = max_entries

    def add_memory(self, content: str, metadata: Dict[str, Any] = None):
        self._store.append({
            'content':  content,
            'metadata': metadata or {},
            'ts':       time.time()
        })
        if len(self._store) > self.max_entries:
            self._store.pop(0)

    def get_context(self, query: str, top_k: int = 5) -> str:
        recent = self._store[-top_k:]
        return "\n".join(
            f"[{e['metadata'].get('role', 'sys')}]: {e['content']}"
            for e in recent
        )

    def clear(self):
        self._store.clear()


# ─────────────────────────────────────────────
#  FAISS (semantic search)
# ─────────────────────────────────────────────
class FAISSMemoryBackend(BaseMemoryBackend):
    """
    Semantic memory using FAISS + sentence-transformers.
    Falls back to DictMemoryBackend if deps are missing.
    Persists index to config.MEMORY_FAISS_PATH (default: data/persona/faiss_index.bin).
    """

    # Class-level defaults — overridden per-instance in __init__ from config
    INDEX_PATH = Path('data/persona/faiss_index.bin')
    META_PATH  = Path('data/persona/faiss_index.meta.json')

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        # ── Resolve storage paths from user config ──────────────────────
        try:
            from managers.settings_manager import config as _sc
            _faiss_path = (_sc.MEMORY_FAISS_PATH or "").strip() or "data/persona/faiss_index.bin"
        except Exception:
            _faiss_path = "data/persona/faiss_index.bin"
        self.INDEX_PATH = Path(_faiss_path)
        self.META_PATH  = self.INDEX_PATH.with_suffix('.meta.json')
        self.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        self._fallback = DictMemoryBackend()
        self._ready    = False
        self._docs: List[Dict] = []

        try:
            from utils.shared_embedder import get_embedder
            import faiss
            import numpy as np

            self._np    = np
            self._faiss = faiss
            self._model = get_embedder(embed_model)
            if self._model is None:
                raise ImportError("Shared embedder returned None")
            
            self._dim   = self._model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatL2(self._dim)

            self._load_persisted()
            self._ready = True
            logger.info(f"✅ FAISS memory ready (dim={self._dim}, entries={self._index.ntotal})")

        except ImportError:
            logger.warning(
                "FAISS/sentence-transformers not installed — using dict memory. "
                "Install: pip install faiss-cpu sentence-transformers"
            )
        except OSError as e:
            # Model not found in cache and no internet connection
            logger.warning(
                f"⚠️ Sentence transformer model not cached and no internet connection. "
                f"Using dict memory fallback. To fix: connect to internet once to download model, "
                f"or manually download 'all-MiniLM-L6-v2' from HuggingFace. Error: {e}"
            )
        except Exception as e:
            logger.error(f"FAISS init error: {e}")

    def add_memory(self, content: str, metadata: Dict[str, Any] = None):
        self._fallback.add_memory(content, metadata)
        if not self._ready:
            return
        try:
            vec = self._model.encode([content]).astype('float32')
            # IndexIDMap (common on Windows faiss-wheels) requires add_with_ids()
            # IndexFlat uses plain add() — detect and dispatch correctly
            try:
                self._index.add(vec)
            except Exception:
                import numpy as _np
                _new_id = _np.array([self._index.ntotal], dtype='int64')
                self._index.add_with_ids(vec, _new_id)
            self._docs.append({'content': content, 'metadata': metadata or {}})
            self._persist()
        except Exception as e:
            logger.error(f"FAISS add error: {e}")

    def get_context(self, query: str, top_k: int = 5) -> str:
        if not self._ready or self._index.ntotal == 0:
            return self._fallback.get_context(query, top_k)
        try:
            vec = self._model.encode([query]).astype('float32')
            _, indices = self._index.search(vec, min(top_k, self._index.ntotal))
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._docs):
                    doc  = self._docs[idx]
                    role = doc['metadata'].get('role', 'sys')
                    results.append(f"[{role}]: {doc['content']}")
            return "\n".join(results)
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return self._fallback.get_context(query, top_k)

    def clear(self):
        self._fallback.clear()
        if self._ready:
            self._index = self._faiss.IndexFlatL2(self._dim)
            self._docs  = []
            self._persist()

    # ── Chaos Fetcher API ─────────────────────────────────────────────────────

    def get_random_episodic(self, limit: int = 50) -> Optional[Dict]:
        """
        Pull a random entry from the most recent `limit` stored memories.
        Returns a dict with 'content' and 'metadata' keys, or None if empty.
        Compatibility shim for code expecting a simple random episodic anchor.
        """
        import random
        pool = self._docs[-limit:] if self._docs else []
        return random.choice(pool) if pool else None

    def get_chaos_episodic(
        self,
        current_thought: str,
        candidate_pool: int = 64,
        strategy: str = "probe",
    ) -> Optional[Dict]:
        """
        Return the stored memory that is MAXIMALLY DISSIMILAR to current_thought.
        This is the Chaos Fetcher — designed to force bisociative collisions by
        maximising the creative gap between the current tension and its anchor.

        Works with IndexFlatL2: larger L2 distance == more dissimilar.

        Strategies
        ----------
        "probe"       — O(candidate_pool). Sample random entries, compute L2
                        distances, return the one furthest away. Best for hot path.
        "exhaustive"  — O(n). Ask FAISS to search all ntotal vectors and take
                        the last result (max distance). Guaranteed true anti-neighbour.
                        Use during reflection / sleep cycles.

        Returns a dict {'content': str, 'metadata': dict} or None if not ready.
        """
        if not self._ready or not self._docs or self._index.ntotal == 0:
            # Graceful fallback to random if FAISS isn't ready
            return self.get_random_episodic(candidate_pool)

        try:
            import numpy as np
            import random

            query_vec = self._model.encode([current_thought]).astype('float32')  # (1, dim)

            if strategy == "exhaustive":
                return self._chaos_exhaustive(query_vec)
            else:
                return self._chaos_probe(query_vec, candidate_pool)

        except Exception as e:
            logger.warning(f"[ChaosFilter] get_chaos_episodic error: {e} — falling back to random")
            return self.get_random_episodic(candidate_pool)

    def _chaos_probe(self, query_vec, candidate_pool: int) -> Optional[Dict]:
        """
        Sample `candidate_pool` random indices, compute L2 distances in batch,
        return the entry with the LARGEST distance (most dissimilar).
        """
        import numpy as np
        import random

        n = len(self._docs)
        pool_size = min(candidate_pool, n)
        indices = random.sample(range(n), pool_size)

        # Reconstruct vectors for sampled indices via FAISS reconstruct()
        # faiss.IndexFlat supports reconstruct(i) to retrieve stored vectors.
        candidate_vecs = np.zeros((pool_size, self._dim), dtype='float32')
        valid_indices = []
        for out_i, doc_i in enumerate(indices):
            try:
                self._index.reconstruct(doc_i, candidate_vecs[out_i])
                valid_indices.append((out_i, doc_i))
            except Exception:
                continue  # index/docs desync — skip

        if not valid_indices:
            return self.get_random_episodic(candidate_pool)

        # L2 distances: ||query - candidate||^2 — use numpy for batch efficiency
        # query_vec shape: (1, dim); candidate_vecs shape: (pool, dim)
        diff = candidate_vecs[[vi[0] for vi in valid_indices]] - query_vec  # (valid, dim)
        l2_distances = (diff * diff).sum(axis=1)  # (valid,)

        # Largest L2 distance == most dissimilar
        max_pos = int(l2_distances.argmax())
        best_doc_idx = valid_indices[max_pos][1]

        logger.debug(
            f"[ChaosFilter] probe({len(valid_indices)} candidates) → "
            f"doc[{best_doc_idx}] L2={l2_distances[max_pos]:.3f}"
        )
        return self._docs[best_doc_idx]

    def _chaos_exhaustive(self, query_vec) -> Optional[Dict]:
        """
        Full index scan via FAISS search(k=ntotal), return the last result
        (FAISS sorts ascending by L2, so index[-1] == maximum distance).
        Most accurate but O(n) — use for reflection cycles only.
        """
        n = self._index.ntotal
        k = min(n, len(self._docs))  # guard against index/docs desync
        distances, indices = self._index.search(query_vec, k)

        # Walk from the end (max distance) backward, find first valid doc index
        for raw_idx in reversed(indices[0]):
            if 0 <= raw_idx < len(self._docs):
                logger.debug(
                    f"[ChaosFilter] exhaustive → doc[{raw_idx}] "
                    f"L2={distances[0][-1]:.3f}"
                )
                return self._docs[raw_idx]

        return self.get_random_episodic()

    def _persist(self):
        try:
            self.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(self.INDEX_PATH))
            with open(self.META_PATH, 'w') as f:
                json.dump(self._docs, f)
        except Exception as e:
            logger.warning(f"FAISS persist error: {e}")

    def _load_persisted(self):
        try:
            if self.INDEX_PATH.exists():
                self._index = self._faiss.read_index(str(self.INDEX_PATH))
                if self.META_PATH.exists():
                    with open(self.META_PATH) as f:
                        self._docs = json.load(f)
                logger.info(f"Loaded {self._index.ntotal} memories from disk")
        except Exception as e:
            logger.warning(f"FAISS load error (starting fresh): {e}")


# ─────────────────────────────────────────────
#  Cognee (graph-based memory)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  MemPalace / Cognee replacement note
# ─────────────────────────────────────────────────────────────────────────────
# CogneeMemoryBackend has been removed.
#
# What Cognee provided:
#   - LLM-based knowledge graph extraction (cognify()) from memories
#   - Graph traversal search (INSIGHTS, GRAPH_COMPLETION)
#
# Why it was removed:
#   - cognify() requires structured JSON from the LLM — qwen2.5-7b reliably
#     fails this, causing constant "Cognee add error" messages
#   - LanceDB + kuzu database conflicts inside the NiceGUI process
#   - tiktoken import failures on Windows with the LM Studio endpoint
#   - The graph traversal it provided was NEVER in the memory_recall path
#     (GoalActionExecutor calls SemanticMemory.get_related_concepts() and
#     then FAISS.get_context() — Cognee search was never called)
#
# What replaces it:
#   PandoraBOX already has two memory backends that work:
#   1. FAISS   (1140+ episodic vectors, sentence-transformers, semantic search)
#   2. SemanticMemory SQLite (1793 concepts, graph relations, get_related_concepts())
#
#   EnhancedFAISSMemoryBackend (below) wraps FAISS and adds:
#   - SemanticMemory enrichment on every add (auto-concept extraction)
#   - Keyword-weighted hybrid search (embedding score + keyword overlap)
#     — the same technique that gives MemPalace its retrieval boost
#   - Category tagging (emotion, user_id, timestamp in metadata)
#
# MemPalace evaluation:
#   MemPalace uses ChromaDB + SQLite — equivalent to what PandoraBOX already has.
#   Its 96.6% LongMemEval score comes from raw verbatim ChromaDB storage with
#   all-MiniLM-L6-v2 embeddings — the same model FAISS already uses here.
#   The palace structure (wings/rooms) is for conversation retrieval across
#   sessions, not for internal cognitive architecture memory.
#   Installing MemPalace would add a second ChromaDB/SQLite layer on top of
#   existing working FAISS + SemanticMemory — no benefit, added complexity.
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedFAISSMemoryBackend(BaseMemoryBackend):
    """
    Drop-in replacement for CogneeMemoryBackend.

    Combines FAISS episodic vector search with SemanticMemory concept
    enrichment and keyword-weighted hybrid retrieval.

    No LLM required. No external service. No startup crashes.
    Works entirely within PandoraBOX's existing memory stack.

    Hybrid search formula (from MemPalace BENCHMARKS.md):
        fused_score = embedding_score × (1 + keyword_weight × overlap)
    This gives a +34% retrieval improvement over pure vector search
    on queries where the exact topic words appear in memories.
    """

    KEYWORD_WEIGHT = 0.4   # how much to boost keyword-matched results

    def __init__(self):
        self._faiss   = FAISSMemoryBackend()
        self._sem     = None   # SemanticMemory — wired in lazily from organism
        logger.info("✅ EnhancedFAISSMemoryBackend ready (FAISS + hybrid search)")

    def attach_semantic_memory(self, sem) -> None:
        """Call once organism is booted: EnhancedFAISS.attach_semantic_memory(o.semantic_memory)"""
        self._sem = sem
        logger.info("[EnhancedFAISS] SemanticMemory attached — concept enrichment active")

    def add_memory(self, content: str, metadata: dict = None) -> None:
        # 1. Store verbatim in FAISS (same as before)
        self._faiss.add_memory(content, metadata)

        # 2. Extract concepts into SemanticMemory — enriches get_related_concepts()
        #    which GoalActionExecutor uses for memory_recall. No LLM needed:
        #    we use a simple keyword heuristic (words > 5 chars, not stopwords).
        if self._sem and hasattr(self._sem, 'upsert_concept'):
            try:
                _stop = {
                    'about', 'above', 'after', 'again', 'being', 'could',
                    'doing', 'every', 'from', 'going', 'great', 'have',
                    'their', 'there', 'these', 'think', 'those', 'through',
                    'under', 'using', 'where', 'which', 'while', 'would',
                    'lumina', 'really', 'today', 'something',
                }
                words = [
                    w.lower().strip('.,!?\"\'()')
                    for w in content.split()
                    if len(w) > 5 and w.lower().strip('.,!?\"\'()') not in _stop
                    and w.isalpha()
                ]
                # Extract up to 6 content words as concepts
                for word in words[:6]:
                    self._sem.upsert_concept(word, strength_boost=0.05)
            except Exception:
                pass

    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Hybrid search: FAISS vector similarity + keyword overlap boost.

        Falls back to pure FAISS if vector search returns nothing.
        """
        # 1. Pure FAISS search returns top_k*2 candidates
        raw = self._faiss_raw_search(query, top_k=min(top_k * 2, 20))
        if not raw:
            return self._faiss.get_context(query, top_k)

        # 2. Re-rank with keyword overlap
        query_words = set(
            w.lower().strip('.,!?\"\'()')
            for w in query.split()
            if len(w) > 3
        )

        def hybrid_score(item):
            text, emb_score = item
            if not query_words:
                return emb_score
            text_words = set(
                w.lower().strip('.,!?\"\'()')
                for w in text.split()
            )
            overlap = len(query_words & text_words) / len(query_words)
            return emb_score * (1 + self.KEYWORD_WEIGHT * overlap)

        reranked = sorted(raw, key=hybrid_score, reverse=True)
        top = reranked[:top_k]
        return "\n---\n".join(text for text, _ in top) if top else ""

    def _faiss_raw_search(self, query: str, top_k: int = 10):
        """Return [(text, score)] from FAISS without joining."""
        try:
            if not self._faiss._index or self._faiss._index.ntotal == 0:
                return []
            import numpy as np
            emb = self._faiss._model.encode([query])[0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            emb_f = np.array([emb], dtype='float32')
            k = min(top_k, self._faiss._index.ntotal)
            distances, indices = self._faiss._index.search(emb_f, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                # BUG FIX: was self._faiss._metadata (doesn't exist).
                # Docs are stored as self._faiss._docs: List[Dict[content, metadata]]
                doc_i = int(idx)
                if doc_i >= len(self._faiss._docs):
                    continue
                doc = self._faiss._docs[doc_i]
                text = doc.get('content', '')
                if text:
                    # Convert L2 distance to similarity score (higher = better)
                    score = 1.0 / (1.0 + float(dist))
                    results.append((text, score))
            return results
        except Exception as e:
            logger.debug(f"[EnhancedFAISS] raw search error: {e}")
            return []

    # ── Chaos Fetcher proxies ─────────────────────────────────────────────────
    # Delegate to FAISSMemoryBackend so callers don't need to unwrap the layer.

    def get_random_episodic(self, limit: int = 50) -> Optional[Dict]:
        """Proxy to FAISSMemoryBackend.get_random_episodic()."""
        return self._faiss.get_random_episodic(limit)

    def get_chaos_episodic(
        self,
        current_thought: str,
        candidate_pool: int = 64,
        strategy: str = "probe",
    ) -> Optional[Dict]:
        """
        Proxy to FAISSMemoryBackend.get_chaos_episodic().
        Returns the memory most dissimilar to current_thought.
        strategy: 'probe' (fast, hot path) | 'exhaustive' (accurate, reflection)
        """
        return self._faiss.get_chaos_episodic(current_thought, candidate_pool, strategy)


def create_memory_manager(backend: str = None) -> BaseMemoryBackend:
    from managers.settings_manager import config
    backend = backend or config.MEMORY_BACKEND

    if backend in ('faiss', 'cognee'):
        # 'cognee' now routes to EnhancedFAISSMemoryBackend — same interface,
        # no LLM dependency, no startup crashes, better hybrid retrieval.
        return EnhancedFAISSMemoryBackend()
    else:
        return DictMemoryBackend()
