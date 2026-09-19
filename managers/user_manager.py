"""
UserManager — Persistent user profiles with face recognition bridge.

Responsibilities:
  - Store and retrieve UserProfile records (data/users/users.json)
  - Track the currently active user across the session
  - Bridge face_id (vision) → user_id (memory/persona)
  - Auto-switch active user when a known face is detected
  - Provide user_id to RelationalMemory, FAISS memory, and AgentController

Profile fields:
  id           : slug, e.g. "alice"  (used as the key everywhere)
  display_name : "Alice"
  face_ids     : ["face_1", "face_3"]  ← links to vision_manager face_encodings
  notes        : free-text bio / preferences the AI should know
  color        : hex accent colour for UI avatar chip  e.g. "#6366f1"
  created_at   : ISO timestamp
  last_seen    : ISO timestamp (updated on every interaction)
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

USERS_FILE = Path("data/users/users.json")

# ── Palette for auto-assigned avatar colours ─────────────────────────────────
_PALETTE = [
    "#6366f1", "#8b5cf6", "#ec4899", "#f97316",
    "#22c55e", "#06b6d4", "#eab308", "#ef4444",
]


def _slugify(name: str) -> str:
    """Turn a display name into a safe slug: 'Alice B.' → 'alice_b'"""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"user_{int(time.time())}"


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    id:           str
    display_name: str
    face_ids:     List[str]  = field(default_factory=list)
    notes:        str        = ""          # free-text, injected into system prompt
    preferences:  List[str]  = field(default_factory=list)
    interests:    List[str]  = field(default_factory=list)
    characteristics: List[str] = field(default_factory=list)
    color:        str        = "#6366f1"
    created_at:   str        = field(default_factory=lambda: datetime.now().isoformat())
    last_seen:    str        = field(default_factory=lambda: datetime.now().isoformat())

    def touch(self):
        self.last_seen = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "UserProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def short_bio(self) -> str:
        """One-liner for prompt injection."""
        parts = [f"User's name: {self.display_name}."]
        if self.notes:
            parts.append(self.notes.strip())
        if self.preferences:
            parts.append("Preferences: " + "; ".join(self.preferences[:4]) + ".")
        if self.interests:
            parts.append("Interests: " + ", ".join(self.interests[:5]) + ".")
        if self.characteristics:
            parts.append("Known characteristics: " + ", ".join(self.characteristics[:4]) + ".")
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────────────────────

class UserManager:
    """
    Singleton — import and use `user_manager` from this module.

    Thread-safe: all mutations go through _lock.
    Auto-saves on every write.
    """

    GUEST_ID = "guest"

    def __init__(self, path: Path = USERS_FILE):
        self._path   = path
        self._lock   = threading.RLock()
        self._users: Dict[str, UserProfile] = {}
        self._prompt_cache: Dict[str, str] = {}
        self._active_id: str = self.GUEST_ID
        self._face_candidate_id: Optional[str] = None
        self._face_candidate_hits = 0
        self._face_candidate_at = 0.0
        self._load()
        self._ensure_guest()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create(
        self,
        display_name: str,
        face_ids: List[str] = None,
        notes: str = "",
        color: str = None,
    ) -> UserProfile:
        """Create a new user. Returns the profile."""
        with self._lock:
            base   = _slugify(display_name)
            uid    = base
            suffix = 2
            while uid in self._users:        # avoid slug collision
                uid = f"{base}_{suffix}"
                suffix += 1

            color = color or _PALETTE[len(self._users) % len(_PALETTE)]
            profile = UserProfile(
                id=uid,
                display_name=display_name,
                face_ids=face_ids or [],
                notes=notes,
                color=color,
            )
            self._users[uid] = profile
            self._prompt_cache.pop(uid, None)
            self._save()
            logger.info(f"👤 Created user: {uid} ({display_name})")
            return profile

    def get(self, user_id: str) -> Optional[UserProfile]:
        return self._users.get(user_id)

    def get_or_guest(self, user_id: str) -> UserProfile:
        return self._users.get(user_id) or self._users[self.GUEST_ID]

    def all_users(self) -> List[UserProfile]:
        with self._lock:
            return list(self._users.values())

    def named_users(self) -> List[UserProfile]:
        """All users except the guest placeholder."""
        return [u for u in self.all_users() if u.id != self.GUEST_ID]

    def update(
        self,
        user_id: str,
        display_name: str = None,
        notes: str = None,
        color: str = None,
    ) -> Optional[UserProfile]:
        with self._lock:
            profile = self._users.get(user_id)
            if not profile:
                return None
            if display_name is not None:
                profile.display_name = display_name
            if notes is not None:
                profile.notes = notes
            if color is not None:
                profile.color = color
            self._prompt_cache.pop(user_id, None)
            self._save()
            return profile

    def delete(self, user_id: str) -> bool:
        if user_id == self.GUEST_ID:
            return False
        with self._lock:
            if user_id not in self._users:
                return False
            del self._users[user_id]
            if self._active_id == user_id:
                self._active_id = self.GUEST_ID
            self._save()
            logger.info(f"🗑️  Deleted user: {user_id}")
            return True

    # ── Face ↔ User bridge ───────────────────────────────────────────────────

    def link_face(self, user_id: str, face_id: str) -> bool:
        """Associate a vision face_id with this user."""
        with self._lock:
            profile = self._users.get(user_id)
            if not profile:
                return False
            if face_id not in profile.face_ids:
                profile.face_ids.append(face_id)
                self._prompt_cache.pop(user_id, None)
                self._save()
                logger.info(f"🔗 Linked face {face_id} → user {user_id}")
            return True

    def unlink_face(self, user_id: str, face_id: str) -> bool:
        with self._lock:
            profile = self._users.get(user_id)
            if not profile or face_id not in profile.face_ids:
                return False
            profile.face_ids.remove(face_id)
            self._save()
            return True

    def find_by_face(self, face_id: str) -> Optional[UserProfile]:
        """Return the user who owns this face_id, or None."""
        for profile in self._users.values():
            if face_id in profile.face_ids:
                return profile
        return None

    def find_by_name(self, display_name: str) -> Optional[UserProfile]:
        """Return a named profile using a case-insensitive display-name match."""
        wanted = str(display_name or "").strip().casefold()
        if not wanted:
            return None
        for profile in self._users.values():
            if profile.id != self.GUEST_ID and profile.display_name.casefold() == wanted:
                return profile
        return None

    def resolve_faces(self, detected_faces: List[Dict]) -> Optional[UserProfile]:
        """
        Given a list of dicts from vision_manager.detect_faces(),
        return the highest-confidence known user found, or None.

        Call this every frame and pass the result to set_active_from_face().
        """
        best: Optional[UserProfile] = None
        best_conf = 0.0
        for face in detected_faces:
            fid  = face.get("id")
            conf = face.get("confidence", 0.0)
            if fid and conf > best_conf:
                profile = self.find_by_face(fid)
                if profile:
                    best      = profile
                    best_conf = conf
        return best

    # ── Active user ──────────────────────────────────────────────────────────

    @property
    def active_id(self) -> str:
        return self._active_id

    @property
    def active(self) -> UserProfile:
        return self.get_or_guest(self._active_id)

    def set_active(self, user_id: str) -> bool:
        """Manually switch the active user."""
        with self._lock:
            if user_id not in self._users:
                return False
            if self._active_id != user_id:
                self._active_id = user_id
                self._users[user_id].touch()
                self._save()
                logger.info(f"👤 Active user → {user_id}")
            return True

    def set_active_from_face(self, profile: Optional[UserProfile]) -> bool:
        """
        Called when vision detects faces.
        Only switches if confidence is high enough.
        Returns True if the active user changed.
        """
        if not profile:
            return False
        if profile.id == self._active_id:
            return False
        changed = self.set_active(profile.id)
        if changed:
            logger.info(f"🎥 Auto-switched to user '{profile.display_name}' via face recognition")
        return changed

    def sync_active_from_faces(self, detected_faces: List[Dict]) -> Optional[UserProfile]:
        """Resolve the strongest recognized face and make it the active profile.

        A face can be named in the vision database before an interlocutor
        profile exists.  In that case, reuse a same-name profile or create one
        and link the recognition id.  This keeps camera presence, chat memory,
        and the active interlocutor on the same identity.
        """
        profile = self.resolve_faces(detected_faces or [])
        if profile is None:
            candidates = []
            for face in detected_faces or []:
                face_id = str(face.get("id") or "").strip()
                name = str(face.get("name") or "").strip()
                if (
                    not face_id
                    or face_id == "__unknown__"
                    or name.casefold() in {"", "unknown", "unknown person", "known person"}
                ):
                    continue
                try:
                    confidence = float(face.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                candidates.append((confidence, face_id, name))

            if candidates:
                _, face_id, name = max(candidates, key=lambda item: item[0])
                profile = self.find_by_name(name)
                if profile is None:
                    profile = self.create(name, face_ids=[face_id])
                    logger.info("Created interlocutor profile for recognized face '%s'", name)
                elif face_id not in profile.face_ids:
                    self.link_face(profile.id, face_id)

        if profile:
            matching_confidences = [
                float(face.get("confidence", 0.0) or 0.0)
                for face in detected_faces or []
                if self.find_by_face(str(face.get("id") or "")) is profile
            ]
            confidence = max(matching_confidences, default=0.0)
            now = time.monotonic()
            if profile.id == self._active_id:
                self._face_candidate_id = None
                self._face_candidate_hits = 0
            elif confidence >= 0.50:
                if (
                    self._face_candidate_id == profile.id
                    and now - self._face_candidate_at <= 5.0
                ):
                    self._face_candidate_hits += 1
                else:
                    self._face_candidate_id = profile.id
                    self._face_candidate_hits = 1
                self._face_candidate_at = now
                if self._face_candidate_hits >= 3:
                    self.set_active_from_face(profile)
                    self._face_candidate_id = None
                    self._face_candidate_hits = 0
            profile.touch()
        return profile

    def touch_active(self):
        """Update last_seen for the active user."""
        with self._lock:
            self.active.touch()
            self._save()

    # ── Prompt helper ────────────────────────────────────────────────────────

    def active_context_for_prompt(self) -> str:
        """
        Returns a short string for injection into the system prompt.
        Empty string if the active user is just 'guest'.
        """
        profile = self.active
        if profile.id == self.GUEST_ID:
            return ""
        return profile.short_bio()

    def prompt_context_for(self, user_id: str) -> str:
        """Return a small cached profile block for the critical chat path."""
        profile = self.get(user_id)
        if not profile or profile.id == self.GUEST_ID:
            return ""
        with self._lock:
            cached = self._prompt_cache.get(profile.id)
            if cached is None:
                cached = profile.short_bio()[:1200]
                self._prompt_cache[profile.id] = cached
            return cached

    def learn_from_message(self, user_id: str, message: str) -> bool:
        """Persist only explicit preference, interest, and trait statements."""
        profile = self.get(user_id)
        if not profile or profile.id == self.GUEST_ID or not message:
            return False
        text = " ".join(str(message).strip().split())
        try:
            from managers.settings_manager import get_persona_name
            from cognition.identity_boundary import claims_persona_identity
            persona_name = get_persona_name()
            claims_persona_name = claims_persona_identity(text, persona_name)
            # Keep the safety boundary intact even when a plugin or test
            # replaces identity_boundary with an older implementation.
            if not claims_persona_name and persona_name:
                escaped = re.escape(" ".join(str(persona_name).split()))
                claims_persona_name = bool(re.search(
                    rf"\b(?:i am|i'm|je suis|moi c['’]est|mon nom est|je m['’]appelle)\s+"
                    rf"(?:the\s+|la\s+|le\s+|l['’])?{escaped}\b",
                    text,
                    flags=re.IGNORECASE,
                ))
        except Exception:
            claims_persona_name = False
        learned: List[tuple[str, str]] = []
        patterns = (
            ("preference", r"(?:i prefer|i like|i love|i don't like|i dislike|je préfère|j'aime|je n'aime pas)\s+([^.!?]{3,100})"),
            ("interest", r"(?:i am interested in|i'm interested in|i enjoy|je m'intéresse à|je m'interesse a)\s+([^.!?]{3,100})"),
            ("characteristic", r"(?:i am|i'm|je suis)\s+(?:a |an |un |une )?([^.!?]{3,70})"),
        )
        for kind, pattern in patterns:
            if kind == "characteristic" and claims_persona_name:
                continue
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1).strip(" ,;:")
                if value and len(value.split()) <= 16:
                    learned.append((kind, value))
        if not learned:
            return False
        changed = False
        with self._lock:
            for kind, value in learned[:3]:
                target = getattr(profile, kind + "s")
                if value.casefold() not in {item.casefold() for item in target}:
                    target.append(value)
                    del target[:-8]
                    changed = True
            if changed:
                self._prompt_cache.pop(profile.id, None)
                profile.touch()
                self._save()
        if changed:
            logger.info("Learned explicit profile details for %s", profile.id)
        return changed

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_id": self._active_id,
                "users":     {uid: p.to_dict() for uid, p in self._users.items()},
            }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"UserManager save error: {e}")

    def _load(self):
        try:
            if not self._path.exists():
                return
            data = json.loads(self._path.read_text())
            for uid, pd in data.get("users", {}).items():
                self._users[uid] = UserProfile.from_dict(pd)
            saved_active = data.get("active_id", self.GUEST_ID)
            if saved_active in self._users:
                self._active_id = saved_active
            logger.info(f"👥 Loaded {len(self._users)} users (active: {self._active_id})")
        except Exception as e:
            logger.error(f"UserManager load error: {e}")

    def _ensure_guest(self):
        if self.GUEST_ID not in self._users:
            self._users[self.GUEST_ID] = UserProfile(
                id=self.GUEST_ID,
                display_name="Guest",
                color="#64748b",
            )


# ── Singleton ─────────────────────────────────────────────────────────────────
user_manager = UserManager()
