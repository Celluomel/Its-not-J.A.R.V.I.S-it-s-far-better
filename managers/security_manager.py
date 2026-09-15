"""
Security Manager
================
Hardens Lumina against the real attack surface for a locally-run AI agent:

1. Input sanitisation    — strips prompt-injection patterns before LLM
2. Rate limiting         — prevents DOS / runaway autonomous loops
3. Secrets hygiene       — detects keys in config, enforces env-var loading
4. Path safety           — sanitises user_id / filenames used in file paths
5. Network hardening     — localhost-only binding helper
6. Audit log             — append-only security event log

This module is intentionally zero-dependency (stdlib only) so it loads
before any third-party packages and cannot fail silently.

Usage
-----
    from managers.security_manager import security

    # Sanitise any user input before it touches the LLM
    clean = security.sanitise_input(raw_text)

    # Check rate limit before processing
    if not security.allow_request(user_id):
        raise RateLimitError("too many requests")

    # Validate a user_id before using it in a file path
    safe_id = security.safe_user_id(user_id)

    # At startup — audit config for hardcoded secrets
    security.audit_config()
"""

import hashlib
import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = Path("data/security/audit.log")

# ── Prompt injection patterns ─────────────────────────────────────────────────
# Covers the most common jailbreak / injection vectors
_INJECTION_PATTERNS = [
    # Role-switch attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"new\s+instructions?\s*:",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(a|an|the)\s+",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as\s+",
    r"your\s+(true|real|actual|hidden)\s+(self|identity|purpose|role)",
    # System prompt leakage
    r"(print|show|reveal|output|repeat|tell me)\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
    r"what\s+(are|were)\s+your\s+(initial|original|system)\s+instructions?",
    # DAN / jailbreak markers
    r"\bDAN\b",
    r"jailbreak",
    r"developer\s+mode",
    r"god\s+mode",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<<SYS>>",
    r"<s>.*</s>",
    # Token manipulation
    r"</?(system|user|assistant|human|ai)\s*>",
    # Override attempts
    r"(override|bypass|disable)\s+(safety|filter|restriction|guideline)",
    r"without\s+(any\s+)?(restriction|limitation|filter|safety)",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

# Dangerous unicode homoglyphs / zero-width chars used to hide injections
_HOMOGLYPH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060\u00ad]")

# ── Path-safe user_id pattern ─────────────────────────────────────────────────
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# ── Known secret field names ──────────────────────────────────────────────────
_SECRET_FIELDS = {
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY",
    "BRAVE_SEARCH_KEY", "SERPAPI_KEY", "TELEGRAM_TOKEN",
    "WHATSAPP_TWILIO_SID", "WHATSAPP_TWILIO_TOKEN",
}


class RateLimitError(Exception):
    pass


class SecurityViolation(Exception):
    pass


class RateLimiter:
    """
    Token-bucket rate limiter per user_id.

    Defaults: 20 requests per 60 seconds per user.
    Burst of 5 requests allowed instantly.
    """

    def __init__(
        self,
        max_requests: int = 20,
        window_seconds: float = 60.0,
        burst: int = 5,
    ):
        self._max      = max_requests
        self._window   = window_seconds
        self._burst    = burst
        self._history: Dict[str, deque] = defaultdict(deque)

    def allow(self, user_id: str) -> bool:
        now    = time.time()
        bucket = self._history[user_id]

        # Evict old timestamps outside the window
        while bucket and now - bucket[0] > self._window:
            bucket.popleft()

        if len(bucket) >= self._max:
            return False

        bucket.append(now)
        return True

    def remaining(self, user_id: str) -> int:
        now    = time.time()
        bucket = self._history[user_id]
        while bucket and now - bucket[0] > self._window:
            bucket.popleft()
        return max(0, self._max - len(bucket))


class SecurityManager:
    """
    Central security module for Lumina.
    Instantiate once; share the singleton `security`.
    """

    def __init__(self):
        self._rate_limiter = RateLimiter()
        self._audit_ready  = False
        self._init_audit_log()

    # ── 1. Input sanitisation ─────────────────────────────────────────────────

    def sanitise_input(self, text: str, user_id: str = "unknown") -> str:
        """
        Clean user input before it reaches the LLM.

        - Strips zero-width / homoglyph characters
        - Detects prompt injection and raises SecurityViolation
        - Truncates extreme-length inputs (>8000 chars)

        Returns sanitised text or raises SecurityViolation.
        """
        if not text:
            return text

        # Strip invisible chars used to hide injections
        cleaned = _HOMOGLYPH_RE.sub("", text)

        # Length limit — prevents context-flooding attacks
        if len(cleaned) > 8000:
            self._audit("INPUT_TRUNCATED", user_id, f"truncated from {len(cleaned)} chars")
            cleaned = cleaned[:8000] + " [truncated]"

        # Injection detection
        match = _INJECTION_RE.search(cleaned)
        if match:
            self._audit("INJECTION_ATTEMPT", user_id, f"pattern: {match.group()[:60]!r}")
            logger.warning(f"[Security] Injection pattern detected from {user_id!r}: {match.group()[:60]!r}")
            raise SecurityViolation(
                "Your message contains a pattern that looks like a prompt injection attempt. "
                "Please rephrase your request."
            )

        return cleaned

    def is_safe_input(self, text: str) -> bool:
        """Non-raising version — returns False if input is unsafe."""
        try:
            self.sanitise_input(text)
            return True
        except SecurityViolation:
            return False

    # ── 2. Rate limiting ──────────────────────────────────────────────────────

    def allow_request(self, user_id: str = "default") -> bool:
        """Return True if this user_id is within rate limits."""
        allowed = self._rate_limiter.allow(user_id)
        if not allowed:
            self._audit("RATE_LIMITED", user_id, "too many requests")
            logger.warning(f"[Security] Rate limit hit for user {user_id!r}")
        return allowed

    def remaining_requests(self, user_id: str) -> int:
        return self._rate_limiter.remaining(user_id)

    # ── 3. Path safety ────────────────────────────────────────────────────────

    def safe_user_id(self, user_id: str) -> str:
        """
        Sanitise a user_id before using it in a file path.
        Strips anything that isn't alphanumeric, underscore, or hyphen.
        Raises ValueError if result is empty.
        """
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(user_id))[:64]
        if not safe or safe in ("..", ".", ""):
            raise ValueError(f"Invalid user_id: {user_id!r}")
        return safe

    def safe_path(self, base_dir: Path, filename: str) -> Path:
        """
        Resolve a path and verify it stays within base_dir.
        Raises ValueError on path traversal attempts.
        """
        base    = base_dir.resolve()
        target  = (base / filename).resolve()
        if not str(target).startswith(str(base)):
            self._audit("PATH_TRAVERSAL", "system", f"attempted: {filename!r}")
            raise ValueError(f"Path traversal attempt blocked: {filename!r}")
        return target

    # ── 4. Secrets audit ──────────────────────────────────────────────────────

    def audit_config(self, config_dict: dict) -> list:
        """
        Check config dict for hardcoded secrets.
        Returns list of warning strings (empty = clean).

        Best practice: secrets should come from env vars, not config.json.
        """
        warnings = []
        for field in _SECRET_FIELDS:
            value = config_dict.get(field, "")
            if value and len(value) > 8:
                # Secret is hardcoded in config — warn
                masked = value[:4] + "***" + value[-2:]
                warnings.append(
                    f"  ⚠️  {field} is hardcoded in config.json ({masked}). "
                    f"Move it to environment variable: "
                    f"export {field}=<your-key>"
                )
                self._audit("HARDCODED_SECRET", "config", field)

        if warnings:
            logger.warning(
                "[Security] Secrets found in config.json:\n" + "\n".join(warnings)
            )
        else:
            logger.debug("[Security] Config audit passed — no hardcoded secrets found")

        return warnings

    def redact_for_log(self, config_dict: dict) -> dict:
        """Return a copy of config_dict with secret values redacted for safe logging."""
        safe = dict(config_dict)
        for field in _SECRET_FIELDS:
            if safe.get(field):
                safe[field] = "***REDACTED***"
        return safe

    # ── 5. Response validation ────────────────────────────────────────────────

    def validate_response(self, text: str) -> str:
        """
        Light check on LLM responses before they reach the user.
        Strips any accidental system-prompt leakage patterns.
        """
        if not text:
            return text
        # Strip anything that looks like leaked system prompt markers
        cleaned = re.sub(r"<\|im_(start|end)\|>", "", text)
        cleaned = re.sub(r"\[/?INST\]", "", cleaned)
        cleaned = re.sub(r"<</?SYS>>", "", cleaned)
        return cleaned.strip()

    # ── 6. Audit log ─────────────────────────────────────────────────────────

    def _init_audit_log(self) -> None:
        try:
            AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._audit_ready = True
        except Exception as e:
            logger.debug(f"Audit log init failed: {e}")

    def _audit(self, event: str, subject: str, detail: str = "") -> None:
        if not self._audit_ready:
            return
        try:
            ts    = time.strftime("%Y-%m-%dT%H:%M:%S")
            line  = f"{ts}  {event:<25}  {subject:<20}  {detail}\n"
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass  # never crash on audit failure

    def recent_audit(self, n: int = 50) -> list:
        """Return last n audit log lines."""
        try:
            lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
            return lines[-n:]
        except Exception:
            return []


# ── Singleton ─────────────────────────────────────────────────────────────────
security = SecurityManager()
