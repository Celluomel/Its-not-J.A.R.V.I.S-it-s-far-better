"""
Vision Manager — Two-Tier Visual Memory
========================================
Tier 1 — Working Memory (JSON):
    visual_working_memory.json holds exactly three entries at all times:
      current  : what Lumina sees right now
      previous : what she saw before the last notable change
      delta    : one-sentence narrative of what changed between them
    This JSON is injected directly into every LLM prompt so Lumina always
    knows what is in front of her without polluting long-term memory.

Tier 2 — Long-Term Visual Memory (FAISS):
    Only scenes that cross the SURPRISE_THRESHOLD (cosine similarity < 0.08)
    are stored in FAISS. At 5-second polling this produces ~5–15 events/hour
    at normal activity, keeping the index lean and semantically useful.
    query_visual_memory(text) does real vector search over this store.

All existing functionality is preserved:
  face recognition, camera capture, encode loop, UI frame serving,
  sync/async analysis, vision LLM routing (separate / direct modes).
"""

import asyncio
import base64
import json
import logging
import os
import pickle
import platform
import threading
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    # face_recognition calls ``quit()`` (SystemExit) when its separate model
    # package is absent. Treat that optional dependency failure as disabled
    # vision instead of allowing it to terminate the whole server.
    with redirect_stdout(StringIO()):
        import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    FACE_RECOGNITION_ERROR = ''
except (ImportError, SystemExit):
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_ERROR = 'face_recognition_models is unavailable'

# ── Two-tier memory deps (graceful degradation if missing) ────────────────────
try:
    import faiss as _faiss_lib
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

# Surprise filter: cosine distance below this → scene is same → skip FAISS storage.
# 0.18 accounts for first-person VLM vocabulary variance ("sunlit" vs "cozy"):
# the same static scene generates ~0.08–0.15 distance due to word choice drift.
# Only genuine scene changes (person enters/leaves, object moved) exceed 0.18.
SURPRISE_THRESHOLD  = 0.18

# How often the autonomous perception loop checks the camera (seconds).
# 10s = ~360 checks/hour. Combined with the surprise filter, typically
# produces 5–20 FAISS entries/hour at normal activity.
PERCEPTION_INTERVAL = 10

# VLM temperature for narrative generation — low for consistency.
# High temperature produces different words for the same scene each tick,
# inflating cosine distance and triggering false-positive surprise events.
NARRATIVE_TEMPERATURE = 0.15

# Working memory JSON path
WORKING_MEMORY_PATH = Path("data/persona/visual_working_memory.json")

# FAISS long-term visual store paths
VISUAL_FAISS_PATH   = Path("data/persona/visual_faiss.bin")
VISUAL_META_PATH    = Path("data/persona/visual_faiss.meta.json")

# Embedding model (shared with memory_manager)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Human-readable timestamp format
TIME_FMT = "%I:%M %p"   # "3:06 PM"

# Narrative prompt — asks the VLM to sound like a witness, not a sensor
NARRATIVE_PROMPT = (
    "What do you see right now? "
    "Describe it like a human eyewitness in one sentence. "
    "Focus on people, actions, and objects. No technical details."
)

# Delta prompt — asks the VLM to compare two scenes
DELTA_PROMPT_TEMPLATE = (
    "Previous scene: \"{prev}\"\n"
    "Current scene: \"{curr}\"\n\n"
    "In one sentence, what specifically changed between these two scenes? "
    "If nothing meaningful changed, reply: 'No significant change.'"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Visual Working Memory (JSON)
# ─────────────────────────────────────────────────────────────────────────────

class VisualWorkingMemory:
    """
    Three-slot JSON working memory: current, previous, delta.
    Always tiny (< 2KB). Written atomically. Thread-safe.

    This is what gets injected into every LLM prompt via
    get_visual_context_for_prompt(). The LLM always knows:
      - what Lumina sees now
      - what she saw before the last change
      - what changed

    Nothing else is needed for conversational awareness.
    """

    _EMPTY = {
        "last_updated":   None,
        "session_events": 0,
        "current":        None,
        "previous":       None,
        "delta":          None,
    }

    def __init__(self, path: Path = WORKING_MEMORY_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state = self._load()
        logger.info(f"[WorkingMemory] Loaded — {self._state['session_events']} prior events")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, description: str, faces: List[str], delta: str) -> None:
        """
        Shift current → previous, install new description as current,
        store computed delta. Called only when surprise threshold is crossed.
        """
        now = datetime.now().strftime(TIME_FMT)
        with self._lock:
            old_current = self._state.get("current")
            self._state["previous"]       = old_current
            self._state["current"]        = {
                "time":        now,
                "description": description,
                "faces":       faces,
            }
            self._state["delta"]          = delta
            self._state["last_updated"]   = now
            self._state["session_events"] = self._state.get("session_events", 0) + 1
        self._save()

    def update_current_only(self, description: str, faces: List[str]) -> None:
        """
        Refresh just the current slot without shifting to previous.
        Used for low-surprise frames where the scene is essentially the same —
        keeps the timestamp fresh without cluttering working memory.
        """
        now = datetime.now().strftime(TIME_FMT)
        with self._lock:
            if self._state["current"]:
                self._state["current"]["description"] = description
                self._state["current"]["time"]        = now
                self._state["current"]["faces"]       = faces
            else:
                self._state["current"] = {
                    "time":        now,
                    "description": description,
                    "faces":       faces,
                }
            self._state["last_updated"] = now
        self._save()

    def get_context_block(self) -> str:
        """
        Return a plain-prose context block for injection into the system prompt.

        Uses narrative sentences rather than labeled slots (Now/Before/Change)
        to prevent the LLM from continuing the format in its own response.

        Example output:
          [Visual Perception]
          Right now I can see the user sitting at their desk holding a coffee mug.
          A few minutes ago I observed the user typing, with two monitors visible.
          What changed: the user stopped typing and picked up the mug.
          People present: Alice
        """
        with self._lock:
            cur  = self._state.get("current")
            prev = self._state.get("previous")
            delt = self._state.get("delta")

        if not cur:
            return ""

        lines = ["[Visual Perception]"]
        lines.append(f"Right now ({cur['time']}) I can see: {cur['description']}")

        if prev:
            lines.append(f"Earlier ({prev['time']}) I observed: {prev['description']}")

        if delt and delt not in ("No significant change.", "First observation.", "Scene changed.", "On-demand analysis."):
            lines.append(f"What changed: {delt}")

        faces = cur.get("faces", [])
        if faces:
            lines.append(f"People present: {', '.join(faces)}")

        return "\n".join(lines)

    def get_raw(self) -> Dict:
        with self._lock:
            import copy
            return copy.deepcopy(self._state)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            tmp = self._path.with_suffix(".tmp")
            with self._lock:
                payload = dict(self._state)
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"[WorkingMemory] save error: {e}")

    def _load(self) -> Dict:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                # Reset session_events counter on load (new session)
                data["session_events"] = 0
                return data
        except Exception as e:
            logger.warning(f"[WorkingMemory] load error: {e}")
        return dict(self._EMPTY)


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — Long-Term Visual FAISS Memory
# ─────────────────────────────────────────────────────────────────────────────

class VisualLongTermMemory:
    """
    FAISS-backed semantic store for notable visual events.
    Only receives entries that crossed the SURPRISE_THRESHOLD.

    query(text, top_k) does a real vector search — not keyword matching.
    Falls back gracefully to keyword search if FAISS deps are missing.
    """

    def __init__(
        self,
        index_path: Path = VISUAL_FAISS_PATH,
        meta_path:  Path = VISUAL_META_PATH,
        embed_model: str = EMBED_MODEL,
    ):
        self._index_path = index_path
        self._meta_path  = meta_path
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock  = threading.Lock()
        self._docs: List[Dict] = []
        self._model = None
        self._index = None
        self._dim   = 384   # all-MiniLM-L6-v2 default
        self._ready = False

        self._init_faiss(embed_model)

    def _init_faiss(self, embed_model: str) -> None:
        if not FAISS_AVAILABLE:
            logger.warning("[VisualLTM] faiss not installed — keyword fallback active")
            return
        try:
            from utils.shared_embedder import get_embedder
            self._model = get_embedder(embed_model)
            if self._model is None:
                raise ImportError("shared_embedder returned None")
            self._dim   = self._model.get_sentence_embedding_dimension()
            self._index = _faiss_lib.IndexFlatL2(self._dim)
            self._load()
            self._ready = True
            logger.info(
                f"[VisualLTM] FAISS ready (dim={self._dim}, "
                f"entries={self._index.ntotal})"
            )
        except Exception as e:
            logger.warning(f"[VisualLTM] FAISS init failed: {e} — keyword fallback")

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, description: str, faces: List[str], timestamp: str) -> None:
        """Store a notable visual event."""
        doc = {
            "time":        timestamp,
            "description": description,
            "faces":       faces,
        }
        with self._lock:
            self._docs.append(doc)

        if self._ready:
            try:
                vec = self._model.encode([description]).astype("float32")
                with self._lock:
                    self._index.add(vec)
                self._persist()
            except Exception as e:
                logger.warning(f"[VisualLTM] add error: {e}")

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text to a vector. Returns None if model unavailable."""
        if not self._ready:
            return None
        try:
            return self._model.encode([text]).astype("float32")
        except Exception:
            return None

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Return the top_k most semantically relevant stored events.
        Falls back to keyword matching if FAISS is unavailable.
        """
        if not self._docs:
            return []

        if self._ready:
            return self._faiss_search(query_text, top_k)
        return self._keyword_search(query_text, top_k)

    def _faiss_search(self, query_text: str, top_k: int) -> List[Dict]:
        try:
            vec = self._model.encode([query_text]).astype("float32")
            with self._lock:
                k = min(top_k, self._index.ntotal)
                if k == 0:
                    return []
                distances, indices = self._index.search(vec, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self._docs):
                    entry = dict(self._docs[idx])
                    entry["_score"] = float(1.0 / (1.0 + dist))
                    results.append(entry)
            return results
        except Exception as e:
            logger.warning(f"[VisualLTM] search error: {e}")
            return self._keyword_search(query_text, top_k)

    def _keyword_search(self, query_text: str, top_k: int) -> List[Dict]:
        q = query_text.lower()
        scored = []
        with self._lock:
            docs = list(self._docs)
        for doc in docs:
            score = sum(1 for w in q.split() if w in doc["description"].lower())
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    # ── Persistence ─────────────────────────────────────────────────────────

    def _persist(self) -> None:
        try:
            with self._lock:
                _faiss_lib.write_index(self._index, str(self._index_path))
                payload = self._docs[:]
            tmp = self._meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            os.replace(tmp, self._meta_path)
        except Exception as e:
            logger.warning(f"[VisualLTM] persist error: {e}")

    def _load(self) -> None:
        try:
            if self._index_path.exists():
                with self._lock:
                    self._index = _faiss_lib.read_index(str(self._index_path))
            if self._meta_path.exists():
                with self._lock:
                    self._docs = json.loads(self._meta_path.read_text())
            logger.info(f"[VisualLTM] Loaded {len(self._docs)} long-term visual memories")
        except Exception as e:
            logger.warning(f"[VisualLTM] load error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Surprise Filter
# ─────────────────────────────────────────────────────────────────────────────

class SurpriseFilter:
    """
    Compares the current description vector to the previous one.
    Returns True (is_surprising) when cosine distance > SURPRISE_THRESHOLD.

    Uses the same embedding model as the long-term store so vectors are
    directly comparable. Gracefully degrades to always-surprised if sklearn
    or the model are unavailable.
    """

    def __init__(self, ltm: VisualLongTermMemory, threshold: float = SURPRISE_THRESHOLD):
        self._ltm       = ltm
        self._threshold = threshold
        self._last_vec: Optional[np.ndarray] = None

    def is_surprising(self, description: str) -> Tuple[bool, float]:
        """
        Returns (surprising: bool, distance: float).
        distance = 1 - cosine_similarity (0 = identical, 2 = opposite).
        """
        vec = self._ltm.encode(description)
        if vec is None or not SKLEARN_AVAILABLE:
            # No model — always treat as surprising so we don't drop frames
            return True, 1.0

        if self._last_vec is None:
            self._last_vec = vec
            return True, 1.0

        try:
            sim = float(_cosine_sim(vec, self._last_vec)[0][0])
            distance = 1.0 - sim
            surprising = distance > self._threshold
            if surprising:
                self._last_vec = vec
            return surprising, round(distance, 4)
        except Exception:
            return True, 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Autonomous Perception Loop
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionLoop:
    """
    Background thread that periodically:
      1. Grabs the latest frame from the camera
      2. Sends it to the VLM for a narrative description
      3. Runs the surprise filter
      4. If surprising: generates a delta, updates working memory,
         stores in FAISS long-term memory
      5. If not surprising: refreshes only the timestamp in working memory

    This loop is entirely non-blocking — it runs in a daemon thread.
    The main camera capture loop is unaffected.
    """

    def __init__(
        self,
        vision_manager: "StreamingVisionManager",
        working_memory: VisualWorkingMemory,
        long_term:      VisualLongTermMemory,
        surprise_filter: SurpriseFilter,
        interval:       float = PERCEPTION_INTERVAL,
    ):
        self._vm      = vision_manager
        self._wm      = working_memory
        self._ltm     = long_term
        self._sf      = surprise_filter
        self._interval = interval
        self._thread:  Optional[threading.Thread] = None
        self._stop     = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="perception-loop",
        )
        self._thread.start()

        # Fast face tracking — runs at 4 FPS independently of the 10s perception tick.
        # Updates vm.last_detected_faces so the rectangle follows the face in real time.
        self._face_thread = threading.Thread(
            target=self._face_tracking_run,
            daemon=True,
            name="face-tracking",
        )
        self._face_thread.start()
        logger.info(f"[PerceptionLoop] Started (interval={self._interval}s, face-tracking=4fps)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._interval + 2)
        if getattr(self, '_face_thread', None):
            self._face_thread.join(timeout=2)
        logger.info("[PerceptionLoop] Stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.warning(f"[PerceptionLoop] tick error: {e}")
            self._stop.wait(self._interval)

    def _face_tracking_run(self) -> None:
        """
        Lightweight face detection at ~4 FPS (every 0.25 s).
        Only updates vm.last_detected_faces — does NOT write to memory or
        trigger working memory updates. That stays with _tick().

        Runs on a half-resolution frame for speed. On a modern CPU this
        takes ~15-40ms per frame with the HOG model.
        """
        FACE_INTERVAL      = 1.5   # ~0.7 FPS — HOG+encoding costs 30-70ms/call;
                                   # 4 FPS burned ~25% of a core and starved audio threads.
                                   # 0.7 FPS is plenty for identity recognition.
        FACE_INTERVAL_BUSY = 3.0   # further back-off while TTS/STT is active
        vm = self._vm
        while not self._stop.is_set():
            try:
                # Back off when TTS is playing or STT is transcribing so the
                # HOG model doesn't starve the audio threads of CPU time.
                if getattr(vm, 'audio_busy', False):
                    self._stop.wait(FACE_INTERVAL_BUSY)
                    continue

                if vm.camera_active and vm.face_detection_enabled:
                    frame = vm.get_current_frame()
                    if frame is not None:
                        faces = vm.detect_faces(frame)
                        # detect_faces already writes to vm.last_detected_faces
                        # The capture loop reads last_detected_faces to draw boxes
            except Exception as e:
                logger.debug(f"[FaceTracking] error: {e}")
            self._stop.wait(FACE_INTERVAL)

    def _tick(self) -> None:
        if not self._vm.camera_active:
            return

        frame = self._vm.get_current_frame()
        if frame is None:
            return

        # Get narrative description from VLM
        description = self._get_narrative(frame)
        if not description:
            return

        # Face detection (non-blocking, uses existing mechanism)
        face_names: List[str] = []
        if self._vm.face_detection_enabled:
            try:
                faces = self._vm.detect_faces(frame)
                face_names = [
                    f.get("name", "Unknown") for f in faces
                    if f.get("name", "Unknown") != "Unknown"
                ]
            except Exception:
                pass

        # The VLM describes the scene before face recognition runs.  Merge
        # recognised identities into the canonical narrative so the same
        # event reaches logs, visual memory, workspace perception, and the
        # PresenceEngine with both scene and person context.
        if face_names:
            names = face_names[:3]
            if len(names) == 1:
                people = names[0]
            elif len(names) == 2:
                people = f"{names[0]} and {names[1]}"
            else:
                people = ", ".join(names[:-1]) + f", and {names[-1]}"
            description = f"{description.rstrip('.')} Recognized person present: {people}."

        # Surprise filter
        surprising, distance = self._sf.is_surprising(description)

        if surprising:
            # Generate delta between previous and current
            delta = self._get_delta(description)

            # Update working memory (shifts current → previous)
            self._wm.update(description, face_names, delta)

            # Store in long-term FAISS visual store
            now_str = datetime.now().strftime(TIME_FMT)
            self._ltm.add(description, face_names, now_str)

            # ── Persist to main episodic memory (survives restarts) ───────
            # The PerceptionLoop's own FAISS is the fast visual store.
            # Writing here also ensures the main EnhancedMemorySystem / SQLite
            # index has a copy — so after restart, visual recall queries hit it.
            try:
                vm = self._vm
                em = getattr(vm, 'episodic_memory', None)
                if em is None:
                    from core.state import state as _st
                    persona = getattr(_st, 'persona', None)
                    org     = getattr(persona, '_organism', None) if persona else None
                    ai_sys  = getattr(org, 'ai_system', None) if org else None
                    em      = getattr(ai_sys, 'memory_system', None) if ai_sys else None
                    if em:
                        vm.episodic_memory = em   # cache for future ticks

                if em and hasattr(em, 'add_memory'):
                    face_note = (
                        f" [people: {', '.join(face_names[:3])}]"
                        if face_names else ""
                    )
                    em.add_memory(
                        f"[Visual] {description[:400]}{face_note}",
                        impact_score       = 0.50,
                        memory_type        = "visual_perception",
                        memory_tier        = "visual",
                        emotional_valence  = "Neutral",
                        arousal_level      = "Low",
                    )
                    logger.debug(
                        f"[PerceptionLoop] episodic persist ✅ "
                        f"(Δ={distance:.3f}{face_note})"
                    )
            except Exception as _ep:
                logger.debug(f"[PerceptionLoop] episodic persist error (non-fatal): {_ep}")

            logger.info(
                f"[PerceptionLoop] 👁 New event (Δ={distance:.3f}): "
                f"{description[:60]}"
            )

            # Phase 6.2 — feed this surprising visual event through the same
            # connector PresenceEngine uses (get_universal_connector shares
            # the buffer), so it becomes a real competing candidate in
            # recursive_deliberation.deliberate() this turn, not just a
            # memory write. distance (Jaccard/embedding distance from
            # SurpriseFilter, already the real novelty signal here) maps
            # directly to Percept.novelty and salience — a bigger surprise
            # is both more novel and more worth attention, same signal.
            try:
                from cognition.universal_connector import Percept, get_universal_connector
                from core.state import state as _st_pl
                persona = getattr(_st_pl, 'persona', None)
                org = getattr(persona, '_organism', None) if persona else None
                if org is not None:
                    # Computed independently — the face_note above is only
                    # assigned inside a conditional in the prior try block
                    # (never runs if em is None), so relying on it here
                    # would risk a NameError depending on memory_system
                    # availability.
                    _face_note = (
                        f" [people: {', '.join(face_names[:3])}]"
                        if face_names else ""
                    )
                    conn = get_universal_connector(org)
                    conn.perceive(Percept(
                        modality   = "vision_scene",
                        source     = "camera",
                        payload    = description[:200] + _face_note,
                        confidence = 0.8,
                        salience   = min(1.0, distance),
                        novelty    = min(1.0, distance),
                        provenance = {"faces": face_names},
                    ))
            except Exception as _uc_e:
                logger.debug(f"[PerceptionLoop] UniversalConnector.perceive failed (non-fatal): {_uc_e}")
        else:
            # Just refresh timestamp — no narrative shift
            self._wm.update_current_only(description, face_names)
            logger.debug(
                f"[PerceptionLoop] idle (Δ={distance:.3f}) — working memory refreshed"
            )

    def _get_narrative(self, frame: np.ndarray) -> Optional[str]:
        """Call the VLM synchronously (runs in daemon thread, blocking is fine)."""
        try:
            frame_b64 = self._vm.get_frame_as_base64(frame)
            if not frame_b64:
                return None

            llm = self._vm.llm
            if not llm:
                return None

            try:
                from managers.settings_manager import config as _vcfg
                _mode = getattr(_vcfg, "VISION_LLM_MODE", "separate")
            except Exception:
                _mode = "separate"

            if _mode == "direct" and hasattr(llm, "generate_with_vision"):
                response = llm.generate_with_vision(
                    prompt=NARRATIVE_PROMPT,
                    image_base64=frame_b64,
                    max_tokens=60,
                    temperature=NARRATIVE_TEMPERATURE,
                )
            elif hasattr(llm, "generate_with_vision"):
                response = llm.generate_with_vision(
                    prompt=NARRATIVE_PROMPT,
                    image_base64=frame_b64,
                    model=self._vm.lava_model,
                    max_tokens=60,
                    temperature=NARRATIVE_TEMPERATURE,
                )
            elif hasattr(llm, "generate_bare"):
                # Text-only fallback: describe from brightness + face data
                return None
            else:
                return None

            if isinstance(response, dict):
                text = response.get("text", response.get("response", ""))
            else:
                text = str(response)

            text = text.strip()
            return text if len(text) > 5 else None

        except Exception as e:
            logger.debug(f"[PerceptionLoop] narrative error: {e}")
            return None

    def _get_delta(self, current_description: str) -> str:
        """
        Ask the VLM (text-only) to compare previous and current descriptions.
        Returns "No significant change." if LLM is unavailable.
        """
        prev_state = self._wm.get_raw().get("current")
        if not prev_state:
            return "First observation."

        prev_desc = prev_state.get("description", "")
        if not prev_desc:
            return "First observation."

        prompt = DELTA_PROMPT_TEMPLATE.format(
            prev=prev_desc[:150],
            curr=current_description[:150],
        )
        try:
            llm = self._vm.llm
            if not llm or not hasattr(llm, "generate_bare"):
                return "Scene changed."

            response = llm.generate_bare(prompt, max_tokens=50, temperature=0.3)
            text = (response or "").strip()
            return text[:200] if text else "Scene changed."
        except Exception as e:
            logger.debug(f"[PerceptionLoop] delta error: {e}")
            return "Scene changed."


# ─────────────────────────────────────────────────────────────────────────────
#  StreamingVisionManager — Main Class
# ─────────────────────────────────────────────────────────────────────────────

class StreamingVisionManager:
    """
    Vision Manager with two-tier visual memory and LLaVA support.

    Tier 1 — Working Memory (JSON): current + previous + delta.
              Always injected into the LLM prompt via get_visual_context_for_prompt().
    Tier 2 — Long-Term Memory (FAISS): only notable events.
              Queried via query_visual_memory(text).

    All original functionality preserved:
      face recognition, camera capture, encode loop, UI frame serving,
      sync/async analysis, vision LLM routing (separate / direct modes).
    """

    def __init__(
        self,
        llm_manager=None,
        memory_manager=None,
        episodic_memory=None,
        lava_model: str = "llava:latest",
        camera_id: int = 0,
        target_fps: int = 10,
        frame_width: int = 640,
        frame_height: int = 480,
        perception_interval: float = PERCEPTION_INTERVAL,
    ):
        self.llm            = llm_manager
        self.memory         = memory_manager
        self.episodic_memory = episodic_memory
        self.lava_model     = lava_model

        # Camera settings
        self.camera_id    = camera_id
        self.target_fps   = target_fps
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # Camera state
        self.camera_active    = False
        self.current_frame    = None
        self.capture_thread   = None
        self.stop_event       = threading.Event()
        self.frame_lock       = threading.Lock()

        # UI optimisation — pre-encoded frame for NiceGUI timer
        self.last_encoded_frame = None
        self.last_clean_encoded_frame = None
        self.encode_lock        = threading.Lock()

        # Face recognition
        self.face_encodings:     Dict[str, np.ndarray] = {}
        self.face_avg_encodings: Dict[str, List]       = {}   # rolling pool of 10
        self.face_names:         Dict[str, str]        = {}
        self.face_first_seen:    Dict[str, str]        = {}   # ISO timestamp
        self.face_seen_count:    Dict[str, int]        = {}
        self.last_detected_faces: List[Dict]           = []
        self.face_detection_enabled = FACE_RECOGNITION_AVAILABLE

        # Set True by the audio layer during TTS playback or STT transcription.
        # The face-tracking thread backs off to avoid starving audio threads.
        self.audio_busy: bool = False

        self.faces_file = Path("data/faces/known_faces.pkl")
        self.faces_file.parent.mkdir(parents=True, exist_ok=True)

        # Legacy RAM cache (kept for backwards compatibility with UI code)
        self.vision_meta: List[Dict] = []
        self.VISION_META_MAX = 50

        # Vision capability check
        self.vision_supported = self._check_vision_support()
        if not self.vision_supported:
            logger.warning("⚠️ Current LLM provider does not support vision analysis")

        # ── Two-tier memory system ───────────────────────────────────────
        self.working_memory  = VisualWorkingMemory(WORKING_MEMORY_PATH)
        self.long_term       = VisualLongTermMemory(VISUAL_FAISS_PATH, VISUAL_META_PATH)
        self.surprise_filter = SurpriseFilter(self.long_term, SURPRISE_THRESHOLD)

        # Autonomous perception loop (starts when camera starts)
        self._perception_loop = PerceptionLoop(
            vision_manager  = self,
            working_memory  = self.working_memory,
            long_term       = self.long_term,
            surprise_filter = self.surprise_filter,
            interval        = perception_interval,
        )

        self._load_faces()
        logger.info(
            f"✅ Vision Manager initialised on {platform.system()} "
            f"(FAISS={'✅' if FAISS_AVAILABLE else '❌'}, "
            f"face_recognition={'✅' if FACE_RECOGNITION_AVAILABLE else '❌'})"
        )

        # ── Social presence engine ──────────────────────────────────────
        # Wired after init via attach_presence_engine() once state.llm is ready.
        self.presence_engine = None

    def attach_presence_engine(self, organism, llm, tts_fn=None) -> None:
        """
        Wire the PresenceEngine to this vision manager.
        Call from state.initialize() once persona + LLM are ready.
        """
        try:
            from cognition.presence_engine import PresenceEngine
            self.presence_engine = PresenceEngine(
                organism = organism,
                llm      = llm,
                tts_fn   = tts_fn,
            )
            logger.info("✅ PresenceEngine attached to VisionManager")
        except Exception as e:
            logger.error(f"PresenceEngine attach failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  Two-Tier Memory — Public API
    # ─────────────────────────────────────────────────────────────────────

    def get_visual_context_for_prompt(self) -> str:
        """
        Return a formatted context block for injection into an LLM prompt.
        Empty string if the camera has never observed anything.

        Example:
          [Visual Context]
          Now (3:06 PM): The user is looking at the screen holding a coffee cup.
          Before (3:01 PM): The user was typing with two monitors visible.
          Change: User stopped typing and picked up coffee.
          Present: Alice
        """
        return self.working_memory.get_context_block()

    def query_visual_memory(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search over long-term visual FAISS memory.
        Returns list of dicts: {time, description, faces, _score}.
        Falls back to keyword search if FAISS is unavailable.
        """
        return self.long_term.query(query_text, top_k)

    def get_working_memory(self) -> Dict:
        """Return the raw working memory state (for dashboard / debug)."""
        return self.working_memory.get_raw()

    # ─────────────────────────────────────────────────────────────────────
    #  Capability Check
    # ─────────────────────────────────────────────────────────────────────

    def _check_vision_support(self) -> bool:
        if not self.llm:
            return False
        for attr in ["generate_with_vision", "vision", "image", "multimodal"]:
            if hasattr(self.llm, attr):
                return True
        provider = getattr(self.llm, "provider", "").lower()
        if any(v in provider for v in ["ollama", "llava", "vision", "multimodal"]):
            return True
        model = getattr(self.llm, "model", "").lower()
        if any(v in model for v in [
            "llava", "vision", "bakllava", "cogvlm", "gemma-4", "gemma3",
            "gemma-3", "qwen2.5-vl", "qwen3-vl",
        ]):
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────────────────────────────────────

    def start_camera(self) -> bool:
        if self.camera_active:
            return True

        logger.info("Starting camera...")
        try:
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
            else:
                cap = cv2.VideoCapture(self.camera_id)

            if not cap.isOpened():
                logger.error("Failed to open camera")
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS,          self.target_fps)

            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                logger.error("Camera opened but cannot read frames")
                cap.release()
                return False

            logger.info(f"✅ Camera test successful — frame shape: {test_frame.shape}")

            with self.frame_lock:
                self.current_frame = test_frame.copy()

            self.camera_active = True
            self.stop_event.clear()

            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(cap,),
                daemon=True,
                name="camera-capture",
            )
            self.capture_thread.start()
            logger.info("✅ Camera capture thread started")

            # Start autonomous perception loop
            self._perception_loop.start()

            return True

        except cv2.error as e:
            logger.error(f"OpenCV error opening camera {self.camera_id}: {e}")
            raise RuntimeError(f"Camera {self.camera_id} unavailable")
        except Exception as e:
            logger.exception(f"Unexpected camera error: {e}")
            raise

    def _capture_loop(self, cap) -> None:
        """
        Background capture loop. All encoding happens here — never in the UI thread.
        Reduces UI thread CPU by ~70%.
        """
        logger.info("📹 Camera capture loop started")
        frame_count = 0

        while not self.stop_event.is_set() and self.camera_active:
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                    small_frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                    clean_frame = small_frame.copy()
                    # Keep the bitmap clean; the interface renders recognition HUD above the organism.
                    if self.face_detection_enabled and self.last_detected_faces:
                        small_frame = self.draw_faces_on_frame(small_frame, self.last_detected_faces)

                    _, buffer = cv2.imencode(
                        ".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    b64_str = base64.b64encode(buffer).decode("utf-8")
                    _, clean_buffer = cv2.imencode(
                        ".jpg", clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    clean_b64_str = base64.b64encode(clean_buffer).decode("utf-8")

                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    with self.encode_lock:
                        self.last_encoded_frame = b64_str
                        self.last_clean_encoded_frame = clean_b64_str

                    if frame_count % 100 == 0:
                        logger.debug(f"Captured {frame_count} frames")
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

        cap.release()
        logger.info("📹 Camera capture loop stopped")

    def stop_camera(self) -> None:
        logger.info("Stopping camera...")
        self._perception_loop.stop()
        self.camera_active = False
        self.stop_event.set()

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        with self.frame_lock:
            self.current_frame = None
        with self.encode_lock:
            self.last_encoded_frame = None
            self.last_clean_encoded_frame = None

        logger.info("Camera stopped")

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_latest_encoded_frame(self) -> Optional[str]:
        """Thread-safe pull for the NiceGUI UI timer."""
        with self.encode_lock:
            return self.last_encoded_frame

    def get_latest_clean_encoded_frame(self) -> Optional[str]:
        with self.encode_lock:
            return self.last_clean_encoded_frame or self.last_encoded_frame

    def get_frame_as_base64(self, frame: Optional[np.ndarray] = None) -> str:
        if frame is None:
            frame = self.get_current_frame()
        if frame is None:
            return ""
        try:
            _, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            logger.error(f"Base64 error: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────
    #  Frame Analysis  (async + sync wrappers — unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────

    async def analyze_frame_async(
        self,
        frame: Optional[np.ndarray] = None,
        prompt: str = (
            "Describe the visible scene naturally and concretely for a human. "
            "Mention the main person or objects, their setting, posture or activity, "
            "and notable background details. Start with the scene description, not "
            "image resolution, brightness, channels, or face-recognition diagnostics."
        ),
        max_tokens: int = 300,
    ) -> Tuple[str, List[Dict]]:
        """
        Analyze a frame and return (analysis_text, faces).
        Also routes the result through the two-tier memory pipeline.
        """
        if frame is None:
            frame = self.get_current_frame()
        if frame is None:
            return "No frame available", []

        try:
            faces: List[Dict] = []
            if self.face_detection_enabled:
                faces = await asyncio.to_thread(self.detect_faces, frame)

            basic_analysis = self._generate_basic_analysis(frame, faces)

            # A dedicated LAVA model is allowed to be multimodal even when
            # the primary text model is not.  Do not gate separate routing
            # on the primary model's capability flag.
            try:
                from managers.settings_manager import config as _vcfg
                _mode = getattr(_vcfg, "VISION_LLM_MODE", "separate")
            except Exception:
                _mode = "separate"
            vision_ready = bool(self.llm) and (
                _mode == "separate" and bool(self.lava_model)
                or _mode != "separate" and self.vision_supported
            )

            enhanced_analysis = None
            if vision_ready:
                try:
                    enhanced_analysis = await self._call_vision_llm(
                        frame, prompt, max_tokens=max_tokens
                    )
                except Exception as e:
                    logger.warning(f"Vision LLM call failed: {e}")

            if enhanced_analysis:
                final_analysis = enhanced_analysis
                if faces and "face" not in enhanced_analysis.lower():
                    final_analysis += f"\n\n{basic_analysis}"
            else:
                final_analysis = basic_analysis
                if not vision_ready:
                    final_analysis += (
                        "\n\nVision description is unavailable: configure a "
                        "multimodal model or a dedicated LAVA model."
                    )

            await self._save_to_memory(final_analysis, prompt, faces)
            return final_analysis, faces

        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return f"Error: {str(e)}", []

    def _generate_basic_analysis(self, frame: np.ndarray, faces: List[Dict]) -> str:
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1

        analysis = "**Image Analysis**\n"
        analysis += f"- Resolution: {width} x {height}\n"
        analysis += f"- Color channels: {channels}\n"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if brightness < 50:
            analysis += f"- Brightness: Very dark ({brightness:.1f})\n"
        elif brightness < 100:
            analysis += f"- Brightness: Dark ({brightness:.1f})\n"
        elif brightness < 150:
            analysis += f"- Brightness: Normal ({brightness:.1f})\n"
        elif brightness < 200:
            analysis += f"- Brightness: Bright ({brightness:.1f})\n"
        else:
            analysis += f"- Brightness: Very bright ({brightness:.1f})\n"

        if faces:
            analysis += f"- Detected {len(faces)} face(s)\n"
            for i, face in enumerate(faces):
                name = face.get("name", "Unknown")
                conf = face.get("confidence", 0)
                if conf > 0:
                    analysis += f"  • Face {i+1}: {name} (confidence: {conf:.0%})\n"
                else:
                    analysis += f"  • Face {i+1}: {name}\n"
        else:
            analysis += "- No faces detected\n"

        return analysis

    async def _call_vision_llm(
        self, frame: np.ndarray, prompt: str, max_tokens: int = 300
    ) -> Optional[str]:
        """
        Route to vision LLM.
        VISION_LLM_MODE = 'separate' → dedicated LLaVA model
        VISION_LLM_MODE = 'direct'   → main multimodal model
        """
        if not self.llm:
            return None

        frame_b64 = self.get_frame_as_base64(frame)
        if not frame_b64:
            return None

        try:
            from managers.settings_manager import config as _vcfg
            _mode = getattr(_vcfg, "VISION_LLM_MODE", "separate")
        except Exception:
            _mode = "separate"

        try:
            if _mode == "direct":
                if not hasattr(self.llm, "generate_with_vision"):
                    logger.warning("[Vision] direct mode but LLM has no generate_with_vision")
                    return None
                response = await asyncio.to_thread(
                    self.llm.generate_with_vision,
                    prompt=prompt,
                    image_base64=frame_b64,
                    max_tokens=max_tokens,
                )
            else:
                if hasattr(self.llm, "generate_with_vision"):
                    response = await asyncio.to_thread(
                        self.llm.generate_with_vision,
                        prompt=prompt,
                        image_base64=frame_b64,
                        model=self.lava_model,
                        max_tokens=max_tokens,
                    )
                elif hasattr(self.llm, "chat_with_vision"):
                    response = await asyncio.to_thread(
                        self.llm.chat_with_vision,
                        message=prompt,
                        image=frame_b64,
                    )
                else:
                    logger.warning("[Vision] No vision method on LLM")
                    return None

            if isinstance(response, dict):
                if not response.get("success", True):
                    logger.warning(f"[Vision] LLM failure: {response.get('text','')}")
                    return None
                return response.get("text", response.get("response", str(response)))
            return str(response)

        except Exception as e:
            logger.error(f"[Vision] LLM call error: {e}")
            return None

    def analyze_frame(
        self,
        frame: Optional[np.ndarray] = None,
        prompt: str = "Describe what you see in this image in detail.",
    ) -> Tuple[str, List[Dict]]:
        """Synchronous wrapper for analyze_frame_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.analyze_frame_async(frame, prompt), loop
                )
                return future.result(timeout=30.0)
            else:
                return asyncio.run(self.analyze_frame_async(frame, prompt))
        except Exception as e:
            logger.error(f"Sync analysis error: {e}")
            return f"Analysis failed: {str(e)}", []

    async def _save_to_memory(
        self, analysis: str, prompt: str, faces: List[Dict]
    ) -> None:
        """
        Save vision analysis to:
          1. Legacy RAM cache (vision_meta) — unchanged, for UI compatibility
          2. Episodic memory (EnhancedMemorySystem) — FAISS + SQLite
          3. Two-tier visual memory — via surprise filter

        The two-tier update is only triggered when the on-demand analysis is
        semantically different from the last perception loop observation.
        """
        try:
            ts = datetime.now().isoformat()
            memory_entry = {
                "id":        len(self.vision_meta),
                "timestamp": ts,
                "prompt":    prompt,
                "analysis":  analysis,
                "faces":     [
                    {"id": f.get("id", "unknown"), "name": f.get("name", "Unknown")}
                    for f in faces
                ],
            }
            self.vision_meta.append(memory_entry)
            if len(self.vision_meta) > self.VISION_META_MAX:
                self.vision_meta = self.vision_meta[-self.VISION_META_MAX:]

            # ── Episodic memory ───────────────────────────────────────────
            em = self.episodic_memory
            if em is None:
                try:
                    from core.state import state as _st
                    persona = getattr(_st, "persona", None)
                    org     = getattr(persona, "_organism", None) if persona else None
                    ai_sys  = getattr(org, "ai_system", None) if org else None
                    em      = getattr(ai_sys, "memory_system", None) if ai_sys else None
                    if em:
                        self.episodic_memory = em
                except Exception:
                    pass

            if em and hasattr(em, "add_memory"):
                face_note = ""
                if faces:
                    known = [
                        f.get("name", "Unknown") for f in faces
                        if f.get("name", "Unknown") != "Unknown"
                    ]
                    if known:
                        face_note = f" [faces: {', '.join(known[:3])}]"
                text = f"[Vision] {analysis[:500].strip()}{face_note}"
                try:
                    em.add_memory(
                        text,
                        impact_score      = 0.55,
                        memory_type       = "visual_perception",
                        memory_tier       = "visual",
                        emotional_valence = "Neutral",
                        arousal_level     = "Low",
                    )
                    logger.info(
                        f"✅ Vision persisted to episodic memory "
                        f"({len(analysis)} chars{face_note})"
                    )
                except Exception as _me:
                    logger.debug(f"Vision episodic persist failed (non-fatal): {_me}")

            # ── Two-tier: push on-demand analysis through surprise filter ─
            # This handles the case where analyze_frame_async() is called
            # directly (e.g. from UI button) rather than via the perception loop.
            face_names = [
                f.get("name", "Unknown") for f in faces
                if f.get("name", "Unknown") != "Unknown"
            ]
            surprising, distance = self.surprise_filter.is_surprising(analysis)
            if surprising:
                delta = "On-demand analysis."
                self.working_memory.update(analysis, face_names, delta)
                now_str = datetime.now().strftime(TIME_FMT)
                self.long_term.add(analysis, face_names, now_str)

        except Exception as e:
            logger.error(f"Failed to save vision to memory: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  Face Recognition (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────

    def detect_faces(self, frame: Optional[np.ndarray] = None) -> List[Dict]:
        if not self.face_detection_enabled:
            return []
        if frame is None:
            frame = self.get_current_frame()
        if frame is None:
            return []

        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations_small = face_recognition.face_locations(rgb_frame, model="hog")
            if not face_locations_small:
                return []

            face_locations = [
                (t * 2, r * 2, b * 2, l * 2)
                for t, r, b, l in face_locations_small
            ]
            # Only compute encodings if there are known faces to match against.
            # face_encodings() costs ~10-30ms per face — skip it when the database
            # is empty since there is nothing to compare against.
            if self.face_encodings:
                face_encodings_list = face_recognition.face_encodings(
                    rgb_frame, face_locations_small
                )
            else:
                face_encodings_list = [None] * len(face_locations_small)

            faces = []
            for face_encoding, face_location in zip(face_encodings_list, face_locations):
                face_id    = None
                face_name  = "Unknown"
                confidence = 0.0

                if self.face_encodings and face_encoding is not None:
                    # Use mean of pooled encodings for robust matching
                    distances = [
                        (kid, face_recognition.face_distance(
                            self.face_avg_encodings.get(kid, [kenc]), face_encoding
                        ).mean())
                        for kid, kenc in self.face_encodings.items()
                    ]
                    best_id, best_distance = min(distances, key=lambda x: x[1])
                    if best_distance < 0.55:   # tighter than 0.6 → fewer false positives
                        face_id    = best_id
                        face_name  = self.face_names.get(best_id, "Known Person")
                        confidence = round(1.0 - best_distance, 3)
                        self._update_avg_encoding(face_id, face_encoding)

                if face_id is None:
                    # New face — compute encoding if it was skipped (empty-DB fast path)
                    if face_encoding is None:
                        try:
                            enc_list = face_recognition.face_encodings(
                                rgb_frame, [face_location]
                            )
                            face_encoding = enc_list[0] if enc_list else None
                        except Exception:
                            face_encoding = None

                    if face_encoding is None:
                        # Still no encoding (very blurry / partial face) — show in
                        # overlay as "Unknown" but do NOT persist to avoid poisoning
                        # the face DB with None values.
                        face_id = "__unknown__"
                        face_name = "Unknown"
                    else:
                        # Register with thumbnail
                        face_id = f"face_{len(self.face_encodings) + 1}"
                        self.face_encodings[face_id]     = face_encoding
                        self.face_avg_encodings[face_id] = [face_encoding]
                        self.face_names[face_id]         = "Unknown Person"
                        self.face_first_seen[face_id]    = datetime.now().isoformat()
                        self.face_seen_count[face_id]    = 1
                        # Save thumbnail crop
                        top, right, bottom, left = face_location
                        fh, fw = frame.shape[:2]
                        crop = frame[
                            max(0, top - 20):min(fh, bottom + 20),
                            max(0, left - 20):min(fw, right + 20),
                        ]
                        if crop.size > 0:
                            thumb_path = self.faces_file.parent / f"{face_id}_thumb.jpg"
                            cv2.imwrite(str(thumb_path), crop)
                        logger.info(f"🆕 New face registered: {face_id}")
                        self._save_faces()
                else:
                    if face_id != "__unknown__":
                        self.face_seen_count[face_id] = self.face_seen_count.get(face_id, 0) + 1

                top, right, bottom, left = face_location
                faces.append({
                    "id":         face_id,
                    "name":       face_name,
                    "confidence": confidence,
                    "seen_count": self.face_seen_count.get(face_id, 1),
                    "first_seen": self.face_first_seen.get(face_id),
                    "location": {
                        "top": int(top), "right": int(right),
                        "bottom": int(bottom), "left": int(left),
                    },
                })

            self.last_detected_faces = faces

            # Identity is established by the camera, so synchronize it here
            # before PresenceEngine can generate an utterance.  API polling is
            # too late for autonomous ENTER/DWELL events.
            try:
                from managers.user_manager import user_manager
                user_manager.sync_active_from_faces(faces)
            except Exception as _user_sync_error:
                logger.debug(
                    "[FaceTracking] active-user sync failed (non-fatal): %s",
                    _user_sync_error,
                )

            # ── Feed presence engine ─────────────────────────────────────
            if self.presence_engine is not None and faces:
                try:
                    self.presence_engine.on_faces_detected(faces)
                except Exception as _pe:
                    pass   # never let presence engine crash detection

            return faces

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def _update_avg_encoding(self, face_id: str, new_encoding: np.ndarray) -> None:
        """Rolling window of 10 encodings per face — reduces jitter from lighting."""
        pool = self.face_avg_encodings.setdefault(face_id, [])
        pool.append(new_encoding)
        if len(pool) > 10:
            pool.pop(0)

    def update_face_name(self, face_id: str, name: str) -> None:
        if face_id in self.face_names:
            old_name = self.face_names[face_id]
            self.face_names[face_id] = name
            self._save_faces()
            logger.info(f"Updated face {face_id}: '{old_name}' → '{name}'")
            self._remember_face_label(face_id, name)

    def _remember_face_label(self, face_id: str, name: str) -> None:
        """Write a named-face event to episodic memory so it survives restarts."""
        try:
            from core.state import state as _st
            persona = getattr(_st, 'persona', None)
            org     = getattr(persona, '_organism', None) if persona else None
            ai_sys  = getattr(org, 'ai_system', None) if org else None
            em      = getattr(ai_sys, 'memory_system', None) if ai_sys else None
            if em and hasattr(em, 'add_memory'):
                first = self.face_first_seen.get(face_id, 'unknown time')
                em.add_memory(
                    f"[Face] I learned that the person I first saw at {first} "
                    f"is called {name}. Face ID: {face_id}.",
                    impact_score      = 0.70,
                    memory_type       = "face_recognition",
                    memory_tier       = "interaction",
                    emotional_valence = "Positive",
                    arousal_level     = "Low",
                )
        except Exception as e:
            logger.debug(f"[FaceRecog] episodic remember error: {e}")

    def delete_face(self, face_id: str) -> bool:
        if face_id in self.face_encodings:
            name = self.face_names.get(face_id, "Unknown")
            del self.face_encodings[face_id]
            self.face_avg_encodings.pop(face_id, None)
            self.face_names.pop(face_id, None)
            self.face_first_seen.pop(face_id, None)
            self.face_seen_count.pop(face_id, None)
            thumb = self.faces_file.parent / f"{face_id}_thumb.jpg"
            thumb.unlink(missing_ok=True)
            self._save_faces()
            logger.info(f"🗑️ Deleted face: {face_id} ({name})")
            return True
        return False

    def get_known_faces(self) -> List[Dict]:
        """All registered faces as list of dicts — for UI display."""
        result = []
        for face_id, name in self.face_names.items():
            thumb_path = self.faces_file.parent / f"{face_id}_thumb.jpg"
            result.append({
                "id":         face_id,
                "name":       name,
                "first_seen": self.face_first_seen.get(face_id),
                "seen_count": self.face_seen_count.get(face_id, 0),
                "thumb_path": str(thumb_path) if thumb_path.exists() else None,
            })
        return sorted(result, key=lambda x: x.get("seen_count", 0), reverse=True)

    def get_face_stats(self) -> Dict:
        """Summary stats for dashboard."""
        named = sum(1 for n in self.face_names.values() if n and n != "Unknown Person")
        return {
            "total_registered": len(self.face_encodings),
            "named":            named,
            "unnamed":          len(self.face_encodings) - named,
            "last_detected":    len(self.last_detected_faces),
        }

    def register_face_from_image(self, image_path: str, name: str) -> Tuple[bool, str]:
        """Register a face from a file — useful for bulk-adding known people."""
        if not self.face_detection_enabled:
            return False, "face_recognition not installed"
        try:
            img  = face_recognition.load_image_file(image_path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                return False, "No face detected in image"
            if len(encs) > 1:
                return False, f"Multiple faces ({len(encs)}) — use a single-person photo"
            enc     = encs[0]
            face_id = f"face_{len(self.face_encodings) + 1}"
            self.face_encodings[face_id]     = enc
            self.face_avg_encodings[face_id] = [enc]
            self.face_names[face_id]         = name
            self.face_first_seen[face_id]    = datetime.now().isoformat()
            self.face_seen_count[face_id]    = 0
            src = cv2.imread(image_path)
            if src is not None:
                cv2.imwrite(str(self.faces_file.parent / f"{face_id}_thumb.jpg"), src)
            self._save_faces()
            self._remember_face_label(face_id, name)
            logger.info(f"✅ Registered '{name}' from {image_path} as {face_id}")
            return True, f"Registered '{name}' as {face_id}"
        except Exception as e:
            return False, f"Error: {e}"

    def _save_faces(self) -> None:
        try:
            data = {
                "encodings":     self.face_encodings,
                "avg_encodings": {k: [e.tolist() for e in v]
                                  for k, v in self.face_avg_encodings.items()},
                "names":         self.face_names,
                "first_seen":    self.face_first_seen,
                "seen_count":    self.face_seen_count,
                "version":       "2.0",
                "saved_at":      datetime.now().isoformat(),
            }
            with open(self.faces_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"💾 Saved {len(self.face_encodings)} faces")
        except Exception as e:
            logger.error(f"Failed to save faces: {e}")

    def _load_faces(self) -> None:
        try:
            if self.faces_file.exists():
                with open(self.faces_file, "rb") as f:
                    data = pickle.load(f)
                self.face_encodings  = data.get("encodings", {})
                self.face_names      = data.get("names", {})
                self.face_first_seen = data.get("first_seen", {})
                self.face_seen_count = data.get("seen_count", {})
                raw_avg              = data.get("avg_encodings", {})
                self.face_avg_encodings = (
                    {k: [np.array(e) for e in v] for k, v in raw_avg.items()}
                    if raw_avg else
                    {k: [enc] for k, enc in self.face_encodings.items()}
                )
                logger.info(
                    f"📂 Loaded {len(self.face_encodings)} faces "
                    f"(v{data.get('version','1.0')}, saved: {data.get('saved_at','?')})"
                )
            else:
                logger.info("No saved faces — starting fresh")
        except Exception as e:
            logger.error(f"Failed to load faces: {e}")
            self.face_encodings = {}; self.face_avg_encodings = {}
            self.face_names     = {}; self.face_first_seen    = {}
            self.face_seen_count = {}

    def draw_faces_on_frame(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        annotated = frame.copy()
        for face in faces:
            loc  = face["location"]
            name = face["name"]
            conf = face.get("confidence", 0.0)

            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
            cv2.rectangle(
                annotated,
                (loc["left"], loc["top"]),
                (loc["right"], loc["bottom"]),
                color, 2,
            )

            label = name
            if conf > 0:
                label += f" ({conf:.0%})"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated,
                (loc["left"], loc["top"] - h - 10),
                (loc["left"] + w, loc["top"]),
                color, -1,
            )
            cv2.putText(
                annotated, label,
                (loc["left"], loc["top"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2,
            )
        return annotated

    # ─────────────────────────────────────────────────────────────────────
    #  Memory Search  (upgraded to FAISS — legacy signature preserved)
    # ─────────────────────────────────────────────────────────────────────

    def search_vision_memory(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[float, Dict]]:
        """
        Legacy signature: returns List[(score, memory_dict)].
        Now backed by FAISS semantic search over long-term visual memory.
        Falls back to keyword search if FAISS unavailable.
        """
        results = self.long_term.query(query, top_k)
        # Convert to legacy (score, dict) tuples
        return [(r.pop("_score", 1.0), r) for r in results]

    # ─────────────────────────────────────────────────────────────────────
    #  Status & Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        wm = self.working_memory.get_raw()
        return {
            "camera_active":      self.camera_active,
            "has_frame":          self.current_frame is not None,
            "face_detection":     self.face_detection_enabled,
            "known_faces":        len(self.face_encodings),
            "memory_count":       len(self.vision_meta),
            "vision_supported":   self.vision_supported,
            "platform":           platform.system(),
            # Two-tier status
            "working_memory":     {
                "has_current":    wm.get("current") is not None,
                "has_previous":   wm.get("previous") is not None,
                "session_events": wm.get("session_events", 0),
                "last_updated":   wm.get("last_updated"),
            },
            "long_term_events":   len(self.long_term._docs),
            "faiss_available":    FAISS_AVAILABLE,
            "perception_running": (
                self._perception_loop._thread is not None
                and self._perception_loop._thread.is_alive()
            ),
        }

    def cleanup(self) -> None:
        self.stop_camera()
        with self.frame_lock:
            self.current_frame = None
        with self.encode_lock:
            self.last_encoded_frame = None
        logger.info("Vision resources cleaned up")
