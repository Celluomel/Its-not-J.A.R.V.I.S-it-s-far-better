"""
SessionManager — Conversation persistence for PandoraBOX.

Solves three real problems in the current architecture:

  Problem 1: _HISTORY_MAX=20 silently drops oldest turns (PandoraBOX just forgets)
  Problem 2: reload_llm() wipes history (saving LLM settings kills conversation)
  Problem 3: No shutdown hook — process kill = no save, no recovery

What this does:
  ✅ Saves history to disk after every exchange
  ✅ Compresses old turns into a summary block instead of silently dropping them
  ✅ Survives reload_llm() — history is saved before reload, restored after
  ✅ Registers app.on_shutdown for graceful final save
  ✅ On boot: reloads the last session's summary so PandoraBOX remembers context

Architecture:
  - Persists to: data/persona/session.json   (raw history, last N turns)
  - Persists to: data/persona/session_summary.json  (compressed older context)
  - LLMManager.history stays at MAX_RAW_TURNS (recent turns only)
  - Older turns are compressed into a SUMMARY block injected at position 0
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import threading
from typing import List, Dict, Optional, Callable, Any

logger = logging.getLogger("session_manager")

SESSION_PATH   = "data/persona/session.json"
SUMMARY_PATH   = "data/persona/session_summary.json"

# When raw history hits this, compress the oldest COMPRESS_WINDOW turns
COMPRESS_TRIGGER = 16   # 80% of default _HISTORY_MAX=20
COMPRESS_WINDOW  = 10   # number of oldest turns to compress into summary
MAX_RAW_TURNS    = 20   # hard cap after compression

_COMPRESS_SYSTEM = """You are a conversation summarizer for an AI assistant named PandoraBOX.
Given a list of conversation turns, write a dense factual summary that preserves:
- What the user was working on or asking about
- Key decisions, facts, or conclusions reached
- User's name if mentioned, their preferences, context
- Any technical details (file paths, config values, code structures) that were discussed

Write in third person past tense. Be dense and specific — this replaces the actual messages.
Start with: "[CONTEXT FROM EARLIER IN THIS CONVERSATION]"
Max 400 words. No markdown headers.
"""


class SessionManager:
    """
    Singleton — one instance shared across the app.
    Attach to an LLMManager instance to activate.
    """

    def __init__(self, llm_fn: Optional[Callable[[str, str], str]] = None):
        """
        llm_fn: callable(prompt, system) → str for compression.
        If None, compression falls back to a simple join (no LLM summary).
        Pass state.llm.generate_bare once the LLM is booted.
        """
        self._llm_fn   = llm_fn
        self._lock     = threading.Lock()
        self._dirty    = False   # True = unsaved changes pending
        self._summary: str = ""  # compressed older context
        os.makedirs("data/persona", exist_ok=True)

    # ── Attach / detach ───────────────────────────────────────────────────────
    def set_llm_fn(self, llm_fn: Callable):
        """Call after LLM boots so compression can use the real model."""
        self._llm_fn = llm_fn

    # ── After every exchange ──────────────────────────────────────────────────
    def on_exchange_complete(self, history: List[Dict]):
        """
        Called by LLMManager after each user/assistant exchange.
        Saves history to disk immediately (non-blocking via thread).
        Triggers compression if history is getting long.
        """
        self._dirty = True
        threading.Thread(
            target=self._save_and_maybe_compress,
            args=(list(history),),    # snapshot to avoid race
            daemon=True,
        ).start()

    def _save_and_maybe_compress(self, history: List[Dict]):
        with self._lock:
            self._save_raw(history)
            if len(history) >= COMPRESS_TRIGGER:
                self._compress(history)

    # ── Before / after reload_llm ─────────────────────────────────────────────
    def save_before_reload(self, history: List[Dict]) -> List[Dict]:
        """
        Call before reload_llm() to snapshot current history.
        Returns the history to restore afterward.
        """
        with self._lock:
            snapshot = list(history)
            self._save_raw(snapshot)
            logger.info(f"SessionManager: saved {len(snapshot)} turns before LLM reload")
            return snapshot

    def restore_after_reload(self, llm_manager: Any) -> int:
        """
        Call after reload_llm() to put history back.
        Returns number of turns restored.
        """
        with self._lock:
            saved = self._load_raw()
            if not saved:
                # Try injecting summary at minimum
                if self._summary:
                    llm_manager.history = [
                        {"role": "assistant", "content": self._summary}
                    ]
                    return 1
                return 0
            llm_manager.history = saved[-MAX_RAW_TURNS:]
            logger.info(f"SessionManager: restored {len(llm_manager.history)} turns after LLM reload")
            return len(llm_manager.history)

    # ── Boot ──────────────────────────────────────────────────────────────────
    def load_on_boot(self, llm_manager: Any) -> dict:
        """
        Called on app startup. Injects last session's summary + recent turns.
        Returns a dict with stats for the boot log.
        """
        with self._lock:
            # Load compressed summary
            summary_data = self._load_summary()
            self._summary = summary_data.get("text", "")
            summary_age   = summary_data.get("saved_at", "")

            # Load raw recent turns
            raw = self._load_raw()

            if not raw and not self._summary:
                return {"restored": False}

            restored_turns = 0
            new_history: List[Dict] = []

            # Inject summary block as first assistant turn.
            # CRITICAL: wrap with temporal frame so LLM knows this is a MEMORY,
            # not the current conversational state. Without this, the LLM reads
            # the summary as its last utterance and continues from that emotional
            # endpoint (e.g. "sweet dreams!" → treats it as end-of-night).
            if self._summary:
                _now_str = datetime.datetime.now().strftime("%H:%M on %A, %B %d")
                _saved_label = f" (saved: {summary_age})" if summary_age else ""
                framed = (
                    f"[MEMORY FROM PREVIOUS SESSION{_saved_label}]\n"
                    f"{self._summary}\n"
                    f"[END OF PREVIOUS SESSION MEMORY — "
                    f"it is now {_now_str}, this is a new session]"
                )
                new_history.append({
                    "role": "assistant",
                    "content": framed,
                })

            # Add recent turns (up to MAX_RAW_TURNS)
            recent = raw[-MAX_RAW_TURNS:] if raw else []
            new_history.extend(recent)
            restored_turns = len(recent)

            llm_manager.history = new_history

            logger.info(
                f"SessionManager boot: restored {restored_turns} recent turns "
                f"+ {'summary' if self._summary else 'no summary'} "
                f"(session from {summary_age or 'unknown'})"
            )
            return {
                "restored": True,
                "turns": restored_turns,
                "has_summary": bool(self._summary),
                "session_age": summary_age,
            }

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def on_shutdown(self, history: List[Dict]):
        """
        Called on graceful shutdown.
        Runs a final compression + save synchronously (blocking is fine here).
        """
        logger.info("SessionManager: shutdown save...")
        with self._lock:
            self._save_raw(history)
            if len(history) >= 4:
                self._compress(history, force=True)
            self._dirty = False
        logger.info("SessionManager: shutdown save complete")

    def clear(self, llm_manager: Any):
        """Wipe session — called from 'new conversation' button."""
        with self._lock:
            llm_manager.history.clear()
            self._summary = ""
            self._save_raw([])
            self._save_summary_data({"text": "", "saved_at": ""})
        logger.info("SessionManager: session cleared")

    # ── Compression ───────────────────────────────────────────────────────────
    def _compress(self, history: List[Dict], force: bool = False):
        """
        Compress the oldest COMPRESS_WINDOW turns into a summary block.
        After compression the history has MAX_RAW_TURNS - COMPRESS_WINDOW + 1 turns.
        """
        if len(history) < COMPRESS_WINDOW and not force:
            return

        to_compress = history[:COMPRESS_WINDOW]
        keep        = history[COMPRESS_WINDOW:]

        # Build a text block of the turns to compress
        turns_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content'][:800]}"
            for m in to_compress
            if isinstance(m, dict) and m.get("content")
        )

        new_summary = self._run_compression(turns_text)

        # Prepend previous summary to new one if exists
        if self._summary:
            new_summary = (
                self._summary.rstrip()
                + "\n\n[ADDITIONAL CONTEXT]\n"
                + new_summary.replace("[CONTEXT FROM EARLIER IN THIS CONVERSATION]\n", "")
            )

        self._summary = new_summary
        now = datetime.datetime.now().isoformat()
        self._save_summary_data({"text": new_summary, "saved_at": now})
        self._save_raw(keep)

        logger.info(
            f"SessionManager: compressed {len(to_compress)} turns → summary "
            f"({len(keep)} turns remain)"
        )

    def _run_compression(self, turns_text: str) -> str:
        """Run LLM compression or fallback to simple join."""
        if self._llm_fn:
            try:
                result = self._llm_fn(
                    f"Conversation turns to summarize:\n\n{turns_text}",
                    _COMPRESS_SYSTEM,
                )
                if result and len(result) > 50:
                    return result.strip()
            except Exception as e:
                logger.warning(f"SessionManager compression LLM failed: {e}")

        # Fallback: extract user messages from the text string
        lines = []
        for chunk in turns_text.split("\n\n"):
            if chunk.startswith("USER:"):
                snippet = chunk[5:].strip()[:200]
                if snippet:
                    lines.append(f"User said: {snippet}")
        return (
            "[CONTEXT FROM EARLIER IN THIS CONVERSATION]\n"
            + "Earlier topics covered: "
            + "; ".join(lines[:5])
        ) if lines else "[CONTEXT FROM EARLIER IN THIS CONVERSATION]\nPrevious conversation existed."

    # ── I/O ───────────────────────────────────────────────────────────────────
    def _save_raw(self, history: List[Dict]):
        try:
            with open(SESSION_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "history":  history,
                    "saved_at": datetime.datetime.now().isoformat(),
                    "turns":    len(history),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"SessionManager: save_raw failed: {e}")

    def _load_raw(self) -> List[Dict]:
        try:
            with open(SESSION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("history", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_summary_data(self, data: dict):
        try:
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"SessionManager: save_summary failed: {e}")

    def _load_summary(self) -> dict:
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # ── Status ────────────────────────────────────────────────────────────────
    def status(self) -> dict:
        raw  = self._load_raw()
        summ = self._load_summary()
        return {
            "raw_turns":      len(raw),
            "has_summary":    bool(self._summary),
            "summary_chars":  len(self._summary),
            "last_saved":     summ.get("saved_at", "never"),
            "dirty":          self._dirty,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
