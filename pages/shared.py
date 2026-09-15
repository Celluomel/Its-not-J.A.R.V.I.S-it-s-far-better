"""
pages/shared.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared constants and thread-safe utilities used by ALL page modules.

  _ui_queue / _safe_ui / _drain_ui_queue  — thread-safe NiceGUI mutations
  on_settings_changed_callback            — external reload hook
  GLOBAL_CSS                              — compiled stylesheet
  _build_proactive_prompt(state)          — organism-aware proactive speech
"""
import logging
import threading
import queue as _queue_mod

from utils.enhanced_css import ENHANCED_GLOBAL_CSS
from core.state import state as _state          # used by _build_proactive_prompt

logger = logging.getLogger(__name__)

# ── Thread-safe UI update queue (module-level so all pages can use it) ───────
# NiceGUI element mutations (set_content, delete, etc.) must happen inside the
# event loop. Background threads push callables here; _drain_ui_queue() runs
# inside ui.timer callbacks where mutations are safe.
_ui_queue: "_queue_mod.SimpleQueue" = _queue_mod.SimpleQueue()

def _safe_ui(fn, *args, **kwargs):
    """Schedule a UI mutation from any thread; executed on next render tick."""
    _ui_queue.put_nowait((fn, args, kwargs))

def _drain_ui_queue():
    """Drain pending thread-safe UI updates. Call at top of any render timer."""
    drained = 0
    while not _ui_queue.empty() and drained < 30:
        try:
            fn, a, kw = _ui_queue.get_nowait()
            try:
                fn(*a, **kw)
            except Exception as _ue:
                import logging as _lg
                _lg.getLogger(__name__).debug(
                    f"[UIQueue] deferred update skipped (element gone): {_ue}"
                )
            drained += 1
        except Exception:
            break





# ─────────────────────────────────────────────
#  Global CSS — Enhanced Organism UI Design
# ─────────────────────────────────────────────
GLOBAL_CSS = ENHANCED_GLOBAL_CSS + """
/* ── App-level overrides only ───────────────────────────────── */

/* Remove NiceGUI's max-width constraint but keep the header padding intact */
.nicegui-content { max-width: unset !important; }

/* ── Fixed size video container ───────────────────────────────── */
.fixed-video-container {
    width: 100%;
    max-width: 800px;
    height: auto;
    margin: 0 auto;
    aspect-ratio: 4/3;
    overflow: hidden;
    border-radius: 12px;
    border: 2px solid #334155;
    background: #000;
    position: relative;
}
.fixed-video-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}
.bubble-child {
    background: linear-gradient(135deg, #3b1f6e 0%, #4c1d95 100%);
    border: 1px solid #6d28d9;
    color: #e9d5ff;
    padding: 10px 16px;
    border-radius: 16px 16px 16px 4px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.55;
    box-shadow: 0 2px 12px rgba(109,40,217,.25);
}
.bubble-master {
    background: linear-gradient(135deg, #134e4a 0%, #115e59 100%);
    border: 1px solid #0d9488;
    color: #ccfbf1;
    padding: 10px 16px;
    border-radius: 16px 16px 16px 4px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.55;
    box-shadow: 0 2px 12px rgba(13,148,136,.25);
}
.vision-overlay {
    background: rgba(0,0,0,0.7);
    padding: 4px 12px;
    border-radius: 20px;
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 10;
    color: #e2e8f0;
    font-size: 12px;
}

/* Ensure body and page can always scroll */
html, body { overflow-y: auto !important; height: auto !important; overflow-x: hidden !important; }
.q-page { overflow-y: visible !important; min-height: 100vh !important; }

/* Settings and vision pages need top padding for the fixed header */
.settings-page-content {
    padding-top: 72px !important;
    padding-bottom: 120px !important;
    position: relative !important;
    z-index: 10 !important;
}

/* Settings card wider on desktop */
.settings-page-content .settings-card {
    max-width: 900px !important;
    width: 100% !important;
    margin: 0 auto !important;
}

/* Settings tab panels must scroll freely — not clipped by chat layout */
.settings-panel {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
}

/* Prevent chat layout containers from leaking into /settings route */
body:has(.settings-page-content) .chat-container,
body:has(.settings-page-content) .floating-core-bar + *,
body:has(.settings-page-content) .left-panel,
body:has(.settings-page-content) .right-panel {
    display: none !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1); }
}
"""



def _build_proactive_prompt(state) -> str:
    """
    Build a proactive prompt grounded in Lumina's real cognitive state.
    Priority: ThoughtStream thought > curiosity topic > memory intrusion
              > emotional colour > minimal fallback.
    """
    try:
        pb  = getattr(state, 'persona', None)
        org = getattr(pb, '_organism', None) if pb else None

        # 1. Most recent non-meta inner thought
        if org and hasattr(org, 'thought_stream'):
            recent = org.thought_stream.recent(3)
            candidates = [t for t in recent if t.thought_type not in ('meta', 'self_model')]
            if candidates:
                best = max(candidates, key=lambda t: t.priority)
                if best.priority > 0.45:
                    return (
                        f"You just had this inner thought: '{best.content}'. "
                        f"Without mentioning that you had a thought, let it naturally surface "
                        f"as something you say. One sentence only. Speak naturally."
                    )

        # 2. Active curiosity topic
        if org and hasattr(org, 'curiosity'):
            top = org.curiosity.top_topic()
            lvl = org.curiosity.global_level() if hasattr(org.curiosity, 'global_level') else 0
            if top and lvl > 0.5:
                return (
                    f"You find yourself drawn to '{top}'. "
                    f"Share this in one natural, spontaneous sentence. One sentence only."
                )

        # 3. Recent memory intrusion from workspace
        if org and hasattr(org, 'workspace'):
            try:
                recent_ws = org.workspace.recent(8)
                for item in reversed(recent_ws):
                    src = str(getattr(item, 'source', ''))
                    if 'memory_intrusion' in src:
                        mem_text = str(getattr(item, 'content', ''))[:80]
                        if mem_text:
                            return (
                                f"Something just surfaced in your memory: '{mem_text}'. "
                                f"Let it come out naturally in one short sentence. One sentence only."
                            )
            except Exception:
                pass

        # 4. Emotional colour
        if org and hasattr(org, '_read_emotion_state'):
            emotion = org._read_emotion_state()
            if emotion not in ('neutral', 'unknown'):
                return (
                    f"You are feeling {emotion}. Without naming the emotion, "
                    f"let it colour one brief, natural thing you say. One sentence only."
                )

    except Exception:
        pass

    # 5. Minimal fallback
    return (
        "Something genuine is on your mind. Say it in one natural sentence. "
        "Don't ask a question. Don't explain. Just say it."
    )

