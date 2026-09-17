"""Optional VoiceMem bridge.

VoiceMem is deliberately kept behind this adapter.  The project memory,
Whisper/STT and TTS remain canonical; an unavailable or failing VoiceMem
installation must never affect a conversation.
"""
from __future__ import annotations

import importlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VoiceMemAdapter:
    """Lazy, best-effort VoiceMem integration using Lumina's existing transcript."""

    def __init__(self, enabled: bool = False, data_path: str = "data/persona/voicemem",
                 top_k: int = 5) -> None:
        self.enabled = bool(enabled)
        self.data_path = Path(data_path or "data/persona/voicemem")
        self.top_k = max(1, min(int(top_k or 5), 20))
        self._vm: Any = None
        self._error = ""
        self._lock = threading.RLock()
        self._latest: dict[str, list[dict[str, Any]]] = {}
        self._segments: dict[str, str] = {}
        self._ledger: dict[str, Any] = {"observations": [], "retrievals": [], "consolidations": 0}
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
                self.data_path.mkdir(parents=True, exist_ok=True)
                cls = getattr(module, "VoiceMem", None)
                if cls is None:
                    raise RuntimeError("VoiceMem package exposes no VoiceMem class")
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
    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "content", "memory", "summary", "description"):
                if value.get(key):
                    return str(value[key]).strip()
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
            # Text is supplied by Lumina's existing STT/chat path.  This avoids
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
        if vm is None:
            return []
        try:
            raw = vm.search(str(query).strip(), top_k=self.top_k)
        except TypeError:
            raw = vm.search(str(query).strip())
        except Exception as exc:
            self._error = str(exc)
            logger.warning("VoiceMem retrieval skipped: %s", exc)
            return []
        if isinstance(raw, dict):
            raw = raw.get("memories") or raw.get("results") or raw.get("items") or []
        result = []
        for item in list(raw or [])[: self.top_k]:
            text = self._text(item)
            if text:
                result.append({"text": text, "source": "voicemem", "memory_type": "voice_memory"})
        self._latest[user_id] = result
        with self._lock:
            retrievals = self._ledger.setdefault("retrievals", [])
            retrievals.append({
                "timestamp": time.time(), "user_id": user_id,
                "query": str(query).strip()[:500], "result_count": len(result),
            })
            del retrievals[:-200]
            self._write_state()
        return result

    def evaluation(self, user_id: str = "default") -> dict[str, Any]:
        """Return observable retrieval/consolidation evidence, not a quality claim."""
        retrievals = [r for r in self._ledger.get("retrievals", []) if r.get("user_id") == user_id]
        nonempty = sum(1 for r in retrievals if r.get("result_count", 0) > 0)
        return {
            "retrievals": len(retrievals),
            "nonempty_retrievals": nonempty,
            "retrieval_hit_rate": round(nonempty / len(retrievals), 3) if retrievals else None,
            "observations": sum(1 for o in self._ledger.get("observations", [])
                                 if o.get("user_id") == user_id and o.get("status") in {"ingested", "fallback"}),
            "indexed_observations": sum(1 for o in self._ledger.get("observations", [])
                                         if o.get("user_id") == user_id and o.get("status") == "ingested"),
            "fallback_observations": sum(1 for o in self._ledger.get("observations", [])
                                          if o.get("user_id") == user_id and o.get("status") == "fallback"),
            "verified_memories": 0,
            "note": "Voice observations are not promoted to verified facts without confirmation.",
        }

    def cached(self, user_id: str = "default") -> list[dict[str, Any]]:
        return list(self._latest.get(user_id, []))

    def space_snapshot(self, user_id: str = "default") -> dict[str, Any]:
        """Return a UI-safe view of the local VoiceMem memory space."""
        with self._lock:
            observations = [
                dict(item) for item in self._ledger.get("observations", [])
                if item.get("user_id", "default") == user_id
            ][-12:]
            retrievals = [
                dict(item) for item in self._ledger.get("retrievals", [])
                if item.get("user_id", "default") == user_id
            ][-8:]
        return {
            "status": self.status(),
            "evaluation": self.evaluation(user_id),
            "observations": observations,
            "retrievals": retrievals,
            "cached": self.cached(user_id),
            "storage": str(self._state_path),
        }
