"""Optional VoiceMem bridge.

VoiceMem is deliberately kept behind this adapter.  The project memory,
Whisper/STT and TTS remain canonical; an unavailable or failing VoiceMem
installation must never affect a conversation.
"""
from __future__ import annotations

import importlib
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

    def status(self) -> dict[str, Any]:
        available = importlib.util.find_spec("voicemem") is not None
        return {
            "enabled": self.enabled,
            "available": available,
            "ready": self._vm is not None,
            "data_path": str(self.data_path),
            "top_k": self.top_k,
            "error": self._error,
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
        vm = self._load()
        if vm is None:
            return False
        try:
            # Text is supplied by Lumina's existing STT/chat path.  This avoids
            # a second recognizer and prevents VoiceMem from taking the audio lock.
            vm.ingest(str(text).strip())
            return True
        except Exception as exc:
            self._error = str(exc)
            logger.warning("VoiceMem ingestion skipped: %s", exc)
            return False

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
        return result

    def cached(self, user_id: str = "default") -> list[dict[str, Any]]:
        return list(self._latest.get(user_id, []))

