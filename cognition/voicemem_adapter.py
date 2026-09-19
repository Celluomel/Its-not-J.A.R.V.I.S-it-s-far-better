"""Optional VoiceMem bridge.

VoiceMem is deliberately kept behind this adapter.  The project memory,
Whisper/STT and TTS remain canonical; an unavailable or failing VoiceMem
installation must never affect a conversation.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VoiceMemAdapter:
    """Lazy, best-effort VoiceMem integration using PandoraBOX's existing transcript."""

    def __init__(self, enabled: bool = False, data_path: str = "data/persona/voicemem",
                 top_k: int = 5, local_mode: bool = True,
                 local_base_url: str = "http://localhost:1234/v1",
                 local_model: str = "local-model") -> None:
        self.enabled = bool(enabled)
        self.local_mode = bool(local_mode)
        self.local_base_url = str(local_base_url or "http://localhost:1234/v1").rstrip("/")
        self.local_model = str(local_model or "local-model")
        self.data_path = Path(data_path or "data/persona/voicemem")
        self.top_k = max(1, min(int(top_k or 5), 20))
        self._vm: Any = None
        self._error = ""
        self._lock = threading.RLock()
        self._latest: dict[str, list[dict[str, Any]]] = {}
        self._segments: dict[str, str] = {}
        self._ledger: dict[str, Any] = {"observations": [], "retrievals": [], "consolidations": 0}
        self._backend_mode = "local_fallback"
        self._state_path = self.data_path / "adapter_state.json"
        self._read_state()

    def _read_state(self) -> None:
        try:
            if self._state_path.exists():
                loaded = json.loads(self._state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._ledger.update(loaded)
        except Exception as exc:
            self._error = f"state read: {exc}"

    def _write_state(self) -> None:
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._ledger, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as exc:
            self._error = f"state write: {exc}"

    def _record(self, item: dict[str, Any]) -> None:
        with self._lock:
            observations = self._ledger.setdefault("observations", [])
            observations.append(item)
            del observations[:-500]
            self._write_state()

    def status(self) -> dict[str, Any]:
        available = importlib.util.find_spec("voicemem") is not None
        return {
            "enabled": self.enabled,
            "available": available,
            "ready": self._vm is not None,
            "data_path": str(self.data_path),
            "top_k": self.top_k,
            "local_mode": self.local_mode,
            "local_base_url": self.local_base_url,
            "local_model": self.local_model,
            "backend_mode": self._backend_mode,
            "error": self._error,
            "observations": len(self._ledger.get("observations", [])),
            "retrievals": len(self._ledger.get("retrievals", [])),
            "consolidations": self._ledger.get("consolidations", 0),
        }

    def _load(self) -> Any:
        if not self.enabled:
            return None
        with self._lock:
            if self._vm is not None:
                return self._vm
            try:
                module = importlib.import_module("voicemem")
                if self.local_mode:
                    self._enable_local_response_format_compatibility()
                self.data_path.mkdir(parents=True, exist_ok=True)
                cls = getattr(module, "VoiceMem", None)
                if cls is None:
                    raise RuntimeError("VoiceMem package exposes no VoiceMem class")
                if self.local_mode:
                    # The default VoiceMem constructor selects OpenAI-backed
                    # components. Inject local embedding and slot routing.
                    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder, shared_e5
                    from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
                    # Some VoiceMem subcomponents (notably ConflictResolver)
                    # read these settings from the environment instead of
                    # the top-level constructor arguments.
                    os.environ["OPENAI_API_KEY"] = "lm-studio-local"
                    os.environ["OPENAI_BASE_URL"] = self.local_base_url
                    os.environ["OPENAI_MODEL"] = self.local_model
                    self._vm = cls(
                        mode="text_mode", memory_root=str(self.data_path),
                        user_id="default", embedding=lambda: LocalE5Embedder(),
                        schema=lambda: LocalQueryClassifier(model=shared_e5()),
                        enable_emotion=False,
                        # OpenAI-compatible clients validate that a key value
                        # exists even for local servers. This is not a real
                        # credential and is sent only to the configured local
                        # endpoint.
                        api_key="lm-studio-local",
                        base_url=self.local_base_url,
                    )
                    self._backend_mode = "local"
                    return self._vm
                # Keep construction permissive across upstream API revisions.
                for kwargs in (
                    {"persist_dir": str(self.data_path)},
                    {"storage_path": str(self.data_path)},
                    {},
                ):
                    try:
                        self._vm = cls(**kwargs)
                        break
                    except TypeError:
                        continue
                if self._vm is None:
                    raise RuntimeError("unsupported VoiceMem constructor")
                return self._vm
            except Exception as exc:
                self._error = str(exc)
                logger.warning("VoiceMem unavailable; continuing with native memory: %s", exc)
                return None

    @staticmethod
    def _enable_local_response_format_compatibility() -> None:
        """Make VoiceMem's OpenAI client calls compatible with LM Studio.

        VoiceMem currently requests ``json_object``. LM Studio versions used
        by this project accept ``text`` or ``json_schema`` instead. The
        extraction prompt still requests JSON, so changing only the transport
        hint preserves VoiceMem's existing parser and avoids patching the
        installed third-party package.
        """
        try:
            openai = importlib.import_module("openai")
            original = getattr(openai, "OpenAI", None)
            if original is None or getattr(original, "_lumina_local_compat", False):
                return

            class _CompletionsProxy:
                def __init__(self, target: Any) -> None:
                    self._target = target

                def create(self, **kwargs: Any) -> Any:
                    response_format = kwargs.get("response_format")
                    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                        kwargs["response_format"] = {"type": "text"}
                    return self._target.create(**kwargs)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._target, name)

            class _ChatProxy:
                def __init__(self, target: Any) -> None:
                    self._target = target
                    self.completions = _CompletionsProxy(target.completions)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._target, name)

            class _LocalOpenAI:
                _lumina_local_compat = True

                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    self._target = original(*args, **kwargs)
                    self.chat = _ChatProxy(self._target.chat)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._target, name)

            openai.OpenAI = _LocalOpenAI
        except Exception as exc:
            logger.debug("Could not enable LM Studio response-format compatibility: %s", exc)

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "content", "memory", "summary", "description"):
                if value.get(key):
                    return str(value[key]).strip()
        for key in ("text", "content", "memory", "summary", "description"):
            candidate = getattr(value, key, None)
            if candidate:
                return str(candidate).strip()
        return str(value or "").strip()

    def ingest_text(self, text: str, user_id: str = "default") -> bool:
        if not self.enabled or not str(text or "").strip():
            return False
        transcript = str(text).strip()[:2000]
        vm = self._load()
        if vm is None:
            # Keep the local observation even when the optional upstream
            # package cannot initialise (for example, when its cloud client
            # expects OPENAI_API_KEY). This is visible as fallback evidence,
            # never as a falsely indexed VoiceMem memory.
            self._record({
                "timestamp": time.time(), "user_id": user_id,
                "text": transcript, "status": "fallback",
                "memory_side": "unclassified", "provenance": "existing_transcript",
            })
            return False
        try:
            # Text is supplied by PandoraBOX's existing STT/chat path.  This avoids
            # a second recognizer and prevents VoiceMem from taking the audio lock.
            vm.ingest(transcript)
            self._record({
                "timestamp": time.time(), "user_id": user_id,
                "text": transcript, "status": "ingested",
                "memory_side": "unclassified", "provenance": "existing_transcript",
            })
            self.consolidate(user_id)
            return True
        except Exception as exc:
            self._error = str(exc)
            self._record({
                "timestamp": time.time(), "user_id": user_id,
                "text": transcript, "status": "fallback",
                "memory_side": "unclassified", "provenance": "existing_transcript",
            })
            logger.warning("VoiceMem ingestion skipped: %s", exc)
            return False

    def ingest_segment(self, segment: str, user_id: str = "default", final: bool = False,
                       stream_id: str = "default") -> bool:
        """Accept incremental STT text and commit only complete utterances."""
        if not self.enabled or not str(segment or "").strip():
            return False
        with self._lock:
            current = self._segments.get(stream_id, "")
            self._segments[stream_id] = (current + " " + str(segment).strip()).strip()
            if not final:
                return True
            completed = self._segments.pop(stream_id, "")
        return self.ingest_text(completed, user_id=user_id)

    def consolidate(self, user_id: str = "default") -> int:
        """Collapse repeated transcript observations without inventing facts.

        Classification and promotion remain explicit future steps: raw voice
        observations are never silently promoted to durable personal facts.
        """
        with self._lock:
            seen: set[str] = set()
            kept: list[dict[str, Any]] = []
            for item in self._ledger.get("observations", []):
                if item.get("user_id") != user_id:
                    kept.append(item)
                    continue
                key = " ".join(str(item.get("text", "")).lower().split())
                if key and key in seen:
                    item = dict(item, status="duplicate")
                else:
                    seen.add(key)
                kept.append(item)
            self._ledger["observations"] = kept[-500:]
            self._ledger["consolidations"] = int(self._ledger.get("consolidations", 0)) + 1
            self._write_state()
            return len(self._ledger["observations"])

    def search(self, query: str, user_id: str = "default") -> list[dict[str, Any]]:
        if not self.enabled or not str(query or "").strip():
            return []
        vm = self._load()
        raw: Any = []
        if vm is not None:
            try:
                raw = vm.search(str(query).strip(), top_k=self.top_k)
            except TypeError:
                raw = vm.search(str(query).strip())
            except Exception as exc:
                self._error = str(exc)
                logger.warning("VoiceMem retrieval skipped: %s", exc)
        if isinstance(raw, dict):
            raw = raw.get("memories") or raw.get("results") or raw.get("items") or []
        elif raw is not None and not isinstance(raw, (list, tuple, set)):
            # VoiceMem's native API returns SearchResult, a structured object
            # whose public left-brain facts are exposed through this property.
            raw = getattr(raw, "result_leftbrain", None) or getattr(raw, "hits", None) or []
        result = []
        for item in list(raw or [])[: self.top_k]:
            text = self._text(item)
            if text:
                result.append({"text": text, "source": "voicemem", "memory_type": "voice_memory"})
        # Keep the feature useful without cloud credentials or a backend that
        # has not indexed the current space yet. This is deliberately a small,
        # transparent lexical fallback over the adapter's own transcript
        # ledger; it never promotes a result to a verified personal fact.
        if not result:
            result = self._local_search(query, user_id)
        self._latest[user_id] = result
        with self._lock:
            retrievals = self._ledger.setdefault("retrievals", [])
            retrievals.append({
                "timestamp": time.time(), "user_id": user_id,
                "query": str(query).strip()[:500], "result_count": len(result),
                "source": "voicemem" if raw and result and result[0].get("source") == "voicemem" else "local_fallback",
            })
            del retrievals[:-200]
            self._write_state()
        return result

    def _local_search(self, query: str, user_id: str) -> list[dict[str, Any]]:
        """Rank stored transcript observations when indexed retrieval is empty."""
        terms = {term.lower() for term in re.findall(r"\w+", str(query)) if len(term) > 2}
        if not terms:
            return []
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in self._ledger.get("observations", []):
            if item.get("user_id") != user_id or item.get("status") == "duplicate":
                continue
            text = str(item.get("text", "")).strip()
            words = {term.lower() for term in re.findall(r"\w+", text) if len(term) > 2}
            overlap = len(terms & words)
            if overlap:
                ranked.append((overlap / len(terms), item))
        ranked.sort(key=lambda pair: (pair[0], pair[1].get("timestamp", 0)), reverse=True)
        return [
            {"text": str(item["text"]), "source": "local_fallback", "memory_type": "voice_memory"}
            for _, item in ranked[: self.top_k]
        ]

    def evaluation(self, user_id: str | None = None) -> dict[str, Any]:
        """Return observable retrieval/consolidation evidence, not a quality claim."""
        retrievals = [r for r in self._ledger.get("retrievals", [])
                      if user_id is None or r.get("user_id") == user_id]
        nonempty = sum(1 for r in retrievals if r.get("result_count", 0) > 0)
        return {
            "retrievals": len(retrievals),
            "nonempty_retrievals": nonempty,
            "retrieval_hit_rate": round(nonempty / len(retrievals), 3) if retrievals else None,
            "observations": sum(1 for o in self._ledger.get("observations", [])
                                 if (user_id is None or o.get("user_id") == user_id)
                                 and o.get("status") in {"ingested", "fallback"}),
            "indexed_observations": sum(1 for o in self._ledger.get("observations", [])
                                         if (user_id is None or o.get("user_id") == user_id)
                                         and o.get("status") == "ingested"),
            "fallback_observations": sum(1 for o in self._ledger.get("observations", [])
                                          if (user_id is None or o.get("user_id") == user_id)
                                          and o.get("status") == "fallback"),
            "verified_memories": 0,
            "note": "Voice observations are not promoted to verified facts without confirmation.",
        }

    def cached(self, user_id: str | None = None) -> list[dict[str, Any]]:
        if user_id is not None:
            return list(self._latest.get(user_id, []))
        latest: list[dict[str, Any]] = []
        for items in self._latest.values():
            latest.extend(items)
        return latest[-self.top_k:]

    def space_snapshot(self, user_id: str | None = None) -> dict[str, Any]:
        """Return a UI-safe view of the local VoiceMem memory space."""
        with self._lock:
            observations = [
                dict(item) for item in self._ledger.get("observations", [])
                if user_id is None or item.get("user_id", "default") == user_id
            ][-12:]
            retrievals = [
                dict(item) for item in self._ledger.get("retrievals", [])
                if user_id is None or item.get("user_id", "default") == user_id
            ][-8:]
        return {
            "status": self.status(),
            "evaluation": self.evaluation(user_id),
            "observations": observations,
            "retrievals": retrievals,
            "cached": self.cached(user_id),
            "storage": str(self._state_path),
        }
