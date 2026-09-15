"""
CognitiveStack — Layer 2 Conversation State
============================================
Lumina's architecture has three memory layers:

    Layer 1  Immediate context    (LLM prompt window)
    Layer 2  Conversation state   ← THIS MODULE
    Layer 3  Long-term memory     (FAISS / robot_memory)

The missing Layer 2 is why conversations lose thread: the system has no
persistent representation of *what is being discussed right now*.

When a user says "I revert you the question" the LLM collapses context
because nothing tracks the active topic stack. A human automatically pops
back to the previous topic frame. This module gives Lumina that capability.

Stack operations (detected from user phrasing)
-----------------------------------------------
PUSH    — new topic detected (similarity to active frame drops below threshold)
POP     — user returns to a previous topic ("going back to...", "I revert...")
UPDATE  — active topic is refined with new keywords
MERGE   — two topics are connected ("same way", "both apply to...")

The active frame + open question are injected into every system prompt via
prompt_context(), giving the LLM a stable conversational backbone.

Background decay
----------------
tick() is called each slow cycle (~2 min) from InternalThoughtLoop.
It decays frame confidence and prunes stale frames, keeping only what
is still cognitively "active". The top frame (current focus) is always kept.

Thread safety: all public methods hold a reentrant lock.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
DRIFT_THRESHOLD  = 0.18   # overlap below this → topic has shifted
CONFIDENCE_DECAY = 0.08   # confidence lost per slow-cycle tick (~2 min)
MIN_CONFIDENCE   = 0.15   # frames below this are pruned
MAX_STACK_DEPTH  = 8      # oldest frames beyond this are silently dropped
MAX_KEYWORDS     = 8      # keywords kept per frame

# ── Stop words (English + French ultra-common) ────────────────────────────────
_STOP = frozenset({
    "the","a","an","is","it","in","on","at","to","do","so","of","and","or",
    "but","for","with","this","that","what","how","why","when","where","you",
    "me","my","your","we","they","them","i","he","she","was","are","were","be",
    "je","tu","il","elle","nous","vous","les","des","une","est","que","qui",
    "pas","sur","par","au","du","dans","avec","pour","mais","donc","ainsi",
})

_STRIP_RE = re.compile(r"[^a-zA-Z\u00C0-\u024F0-9\s]", re.UNICODE)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TopicFrame:
    topic:           str
    active_question: str       = ""
    keywords:        List[str] = field(default_factory=list)
    confidence:      float     = 1.0
    unresolved:      List[str] = field(default_factory=list)
    created_at:      float     = field(default_factory=time.time)

    def touch(self) -> None:
        self.confidence = min(1.0, self.confidence + 0.10)

    def to_prompt_line(self) -> str:
        parts = [f"[TOPIC] {self.topic}"]
        if self.active_question:
            parts.append(f"[Q] {self.active_question}")
        if self.unresolved:
            parts.append(f"[OPEN] {'; '.join(self.unresolved[:2])}")
        return " | ".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _keywords(text: str) -> List[str]:
    clean = _STRIP_RE.sub(" ", text.lower())
    words = clean.split()
    seen: set = set()
    result = []
    for w in words:
        nw = unicodedata.normalize("NFD", w)
        nw = "".join(c for c in nw if unicodedata.category(c) != "Mn")
        if len(w) >= 4 and nw not in _STOP and w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= MAX_KEYWORDS * 3:
            break
    return result[:MAX_KEYWORDS]


def _overlap(kw_list: List[str], text: str) -> float:
    if not kw_list:
        return 0.0
    b = set(_keywords(text) + _keywords(text)[:50])
    a = set(kw_list)
    if not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Main class ────────────────────────────────────────────────────────────────

class CognitiveStack:
    """
    Conversation topic stack for Lumina.

    Typical usage in AgentController.process():

        # At top of each interaction, BEFORE building the prompt:
        stack_ctx = get_cognitive_stack().prompt_context()

        # AFTER receiving bot response:
        get_cognitive_stack().update_from_message(user_text, bot_response)
        get_cognitive_stack().mark_bot_question(bot_response)
    """

    def __init__(self) -> None:
        self._stack: List[TopicFrame] = []
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_from_message(
        self, text: str, bot_response: str = ""
    ) -> Tuple[str, Optional[TopicFrame]]:
        """
        Analyse user message, update the stack, return (operation, active_frame).
        operation in {"push", "pop", "update", "merge", "idle"}
        """
        with self._lock:
            lower = text.lower()

            # ── POP signals ───────────────────────────────────────────────────
            pop_signals = [
                "going back", "back to", "return to", "let's return",
                "i revert", "reverting", "as i was saying", "as we said",
                "earlier you", "you said earlier", "previously",
                "revenons", "comme je disais",
            ]
            if any(s in lower for s in pop_signals) and len(self._stack) > 1:
                self._stack.pop()
                logger.debug(f"[CognitiveStack] POP -> {self._top()!r}")
                return "pop", self._stack[-1] if self._stack else None

            # ── MERGE signals ─────────────────────────────────────────────────
            merge_signals = [
                "both", "same way", "similarly", "connected to", "link between",
                "same idea", "de meme", "de la meme facon",
            ]
            if any(s in lower for s in merge_signals) and len(self._stack) >= 2:
                merged = f"{self._stack[-2].topic} <-> {self._stack[-1].topic}"
                kw = list(dict.fromkeys(
                    self._stack[-2].keywords + self._stack[-1].keywords
                ))[:MAX_KEYWORDS]
                frame = TopicFrame(
                    topic=merged, keywords=kw,
                    active_question=self._stack[-1].active_question,
                )
                self._stack = self._stack[:-2] + [frame]
                logger.debug(f"[CognitiveStack] MERGE -> {merged!r}")
                return "merge", frame

            # ── Drift detection ───────────────────────────────────────────────
            new_kw = _keywords(text)
            drift = True
            sim = 0.0
            if self._stack:
                sim = _overlap(self._stack[-1].keywords, text)
                drift = sim < DRIFT_THRESHOLD

            if drift:
                # PUSH new frame
                topic = " ".join(new_kw[:3]) if new_kw else text[:40].strip()
                frame = TopicFrame(
                    topic=topic,
                    active_question=self._infer_question(text),
                    keywords=new_kw,
                )
                self._stack.append(frame)
                if len(self._stack) > MAX_STACK_DEPTH:
                    self._stack = self._stack[-MAX_STACK_DEPTH:]
                logger.debug(f"[CognitiveStack] PUSH -> {topic!r} (sim={sim:.2f})")
                return "push", frame
            else:
                # UPDATE current frame
                frame = self._stack[-1]
                frame.touch()
                existing = set(frame.keywords)
                for kw in new_kw:
                    if kw not in existing and len(frame.keywords) < MAX_KEYWORDS:
                        frame.keywords.append(kw)
                        existing.add(kw)
                # Record bot question as unresolved point
                if bot_response:
                    q = self._infer_question(bot_response)
                    if q and q not in frame.unresolved:
                        frame.unresolved.append(q)
                        frame.unresolved = frame.unresolved[-5:]
                logger.debug(f"[CognitiveStack] UPDATE -> {frame.topic!r}")
                return "update", frame

    def mark_bot_question(self, bot_response: str) -> None:
        """Record any question Lumina asked as an open point in the active frame."""
        with self._lock:
            if not self._stack:
                return
            q = self._infer_question(bot_response)
            if q and q not in self._stack[-1].unresolved:
                self._stack[-1].unresolved.append(q)
                self._stack[-1].unresolved = self._stack[-1].unresolved[-5:]

    def prompt_context(self, max_frames: int = 3) -> str:
        """Return a compact block for injection into the system prompt."""
        with self._lock:
            if not self._stack:
                return ""
            frames = self._stack[-max_frames:][::-1]
            lines = ["-- CONVERSATION THREAD --"]
            for i, f in enumerate(frames):
                prefix = "> " if i == 0 else "  "
                lines.append(prefix + f.to_prompt_line())
            return "\n".join(lines)

    def active_topic(self) -> str:
        with self._lock:
            return self._top()

    def active_question(self) -> str:
        with self._lock:
            return self._stack[-1].active_question if self._stack else ""

    def tick(self) -> None:
        """Background decay — called by InternalThoughtLoop._slow_cycle()."""
        with self._lock:
            for f in self._stack:
                f.confidence = max(0.0, f.confidence - CONFIDENCE_DECAY)
            if len(self._stack) > 1:
                self._stack = [
                    f for f in self._stack[:-1] if f.confidence >= MIN_CONFIDENCE
                ] + [self._stack[-1]]
            logger.debug(f"[CognitiveStack] tick: {len(self._stack)} frame(s)")

    def reset(self) -> None:
        with self._lock:
            self._stack.clear()

    def depth(self) -> int:
        with self._lock:
            return len(self._stack)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _top(self) -> str:
        return self._stack[-1].topic if self._stack else ""

    @staticmethod
    def _infer_question(text: str) -> str:
        for s in re.split(r"[.!?]+", text):
            if "?" in s and len(s.strip()) > 5:
                return s.strip()[:120]
        return ""


# ── Process-level singleton ───────────────────────────────────────────────────
_stack_singleton = CognitiveStack()

def get_cognitive_stack() -> CognitiveStack:
    return _stack_singleton
