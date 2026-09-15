"""Global CSS Styles"""

GLOBAL_CSS = """
/* ── Reset / base ──────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body {
    margin: 0;
    background: #0f172a;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #e2e8f0;
    -webkit-font-smoothing: antialiased;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ── Chat bubbles ───────────────────────────────────────── */
.bubble-user {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    color: #fff;
    border-radius: 20px 20px 5px 20px;
    padding: 11px 18px;
    max-width: 74%;
    align-self: flex-end;
    margin: 4px 0;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
    font-size: 0.925rem;
    line-height: 1.55;
    word-break: break-word;
    animation: slideInRight 0.18s ease-out;
}

.bubble-bot {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 20px 20px 20px 5px;
    padding: 11px 18px;
    max-width: 74%;
    align-self: flex-start;
    margin: 4px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    font-size: 0.925rem;
    line-height: 1.55;
    word-break: break-word;
    border: 1px solid #334155;
    animation: slideInLeft 0.18s ease-out;
}

/* Markdown inside bot bubbles */
.bubble-bot p  { margin: 0 0 0.4em; }
.bubble-bot p:last-child { margin-bottom: 0; }
.bubble-bot code {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
    color: #7dd3fc;
}
.bubble-bot pre {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    overflow-x: auto;
    margin: 6px 0;
}

/* ── Chat wrap ──────────────────────────────────────────── */
.chat-wrap {
    height: calc(100vh - 132px);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    padding: 24px 20px 12px;
    gap: 2px;
    scroll-behavior: smooth;
}

/* ── Animations ─────────────────────────────────────────── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(12px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-dot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%           { transform: scale(1.0); opacity: 1; }
}

/* ── Typing indicator ───────────────────────────────────── */
.typing-bubble {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px 20px 20px 5px;
    padding: 12px 18px;
    align-self: flex-start;
    display: flex;
    align-items: center;
    gap: 5px;
    animation: fadeIn 0.18s ease-out;
}
.typing-dot {
    width: 7px; height: 7px;
    background: #64748b;
    border-radius: 50%;
    animation: pulse-dot 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Header ─────────────────────────────────────────────── */
.app-header {
    background: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 0 24px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-title { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; color: #f1f5f9; }
.app-subtitle { font-size: 0.72rem; color: #64748b; margin-top: 1px; }

/* ── Status pill ────────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    cursor: default;
}
.pill-memory   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-tts      { background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; }
.pill-clone    { background: #431407; color: #fb923c; border: 1px solid #7c2d12; }
.pill-clone-on { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-vision   { background: #1e1a3d; color: #c084fc; border: 1px solid #6b21a5; }

/* ── Footer input bar ───────────────────────────────────── */
.chat-footer {
    background: #0f172a;
    border-top: 1px solid #1e293b;
    padding: 12px 20px;
    height: 68px;
}
.input-wrap {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px 6px 14px;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.input-wrap:focus-within {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
}
.input-wrap .q-field { flex: 1; }
.input-wrap .q-field__control { background: transparent !important; }
.input-wrap .q-field__native,
.input-wrap .q-field__input { color: #e2e8f0 !important; }
.input-wrap .q-field--outlined .q-field__control::before { border: none !important; }
.input-wrap .q-field--outlined .q-field__control::after  { border: none !important; }
.input-wrap .q-field__label { color: #64748b !important; }

/* ── Send button ────────────────────────────────────────── */
.send-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border-radius: 10px !important;
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.4) !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
}
.send-btn:active { transform: scale(0.94) !important; }

/* ── Mode select ────────────────────────────────────────── */
.mode-select .q-field__control { background: #1e293b !important; border-radius: 10px !important; }
.mode-select .q-field__native,
.mode-select .q-field__input  { color: #e2e8f0 !important; font-size: 0.82rem !important; }
.mode-select .q-field--outlined .q-field__control::before { border-color: #334155 !important; border-radius: 10px !important; }
.mode-select .q-field--outlined .q-field__control::after  { border-color: #6366f1 !important; border-radius: 10px !important; }
.mode-select .q-field__label { color: #64748b !important; }

/* ── Status bar ─────────────────────────────────────────── */
.status-bar {
    position: fixed;
    bottom: 72px; left: 0; right: 0;
    text-align: center;
    font-size: 0.75rem;
    color: #64748b;
    pointer-events: none;
    height: 20px;
    line-height: 20px;
}

/* ── Settings page ──────────────────────────────────────── */
.settings-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    overflow: hidden;
}
.settings-tabs { background: #0f172a !important; border-bottom: 1px solid #334155 !important; }
.settings-tabs .q-tab { color: #64748b !important; font-size: 0.82rem !important; }
.settings-tabs .q-tab--active { color: #a5b4fc !important; }
.settings-tabs .q-tab-indicator { background: #6366f1 !important; }
.settings-panel { background: #1e293b !important; }

/* ── Form fields in dark mode ───────────────────────────── */
.dark-input .q-field__control { background: #0f172a !important; border-radius: 10px !important; }
.dark-input .q-field__native,
.dark-input .q-field__input  { color: #e2e8f0 !important; }
.dark-input .q-field--outlined .q-field__control::before { border-color: #334155 !important; border-radius: 10px !important; }
.dark-input .q-field--outlined .q-field__control::after  { border-color: #6366f1 !important; border-radius: 10px !important; }
.dark-input .q-field__label { color: #64748b !important; }
.dark-input .q-select__dropdown-icon { color: #64748b !important; }

/* ── Info cards (voice lab) ─────────────────────────────── */
.info-card {
    background: #0f172a;
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid #334155;
}
.info-card-amber  { border-color: #92400e; background: #1c1008; }
.info-card-purple { border-color: #4c1d95; background: #120d1f; }
.info-card-green  { border-color: #14532d; background: #0a1f12; }
.info-card-blue   { border-color: #1e3a5f; background: #0a1626; }

/* ── Clone badge ────────────────────────────────────────── */
.clone-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #052e16; border: 1px solid #16a34a;
    color: #4ade80; border-radius: 8px; padding: 5px 12px;
    font-size: 0.8rem; font-weight: 600;
}
.clone-badge-warn {
    background: #1c1008; border-color: #b45309; color: #fbbf24;
}

/* ── Status table ───────────────────────────────────────── */
.status-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #1e293b;
    font-size: 0.875rem;
}
.status-ok   { color: #4ade80; }
.status-fail { color: #f87171; }

/* ── Save button ────────────────────────────────────────── */
.save-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important; border-radius: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.01em !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
}

/* ── Section label ──────────────────────────────────────── */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #475569; margin-bottom: 10px;
}

/* ── Smooth progress bar ────────────────────────────────── */
.q-linear-progress { border-radius: 4px; }

/* ── Notification tweaks ────────────────────────────────── */
.q-notification { border-radius: 12px !important; }

/* ── Vision preview ─────────────────────────────────────── */
.vision-preview {
    border-radius: 12px;
    border: 2px solid #334155;
    overflow: hidden;
    background: #1a1a1a;
    position: relative;
    min-height: 360px;
    display: flex;
    justify-content: center;
    align-items: center;
}
.vision-frame {
    max-width: 100%;
    max-height: 480px;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
}
.vision-overlay {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(0,0,0,0.7);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    color: #c084fc;
    border: 1px solid #6b21a5;
    z-index: 10;
}

/* ── Fixed size video container ─────────────────────────── */
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

/* ── Connection status ──────────────────────────────────── */
.connection-status {
    position: fixed;
    bottom: 10px;
    right: 10px;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    z-index: 9999;
    pointer-events: none;
}
.connection-status.connected {
    background: #052e16;
    border-color: #166534;
    color: #4ade80;
}
.connection-status.reconnecting {
    background: #431407;
    border-color: #7c2d12;
    color: #fb923c;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Loading overlay ────────────────────────────────────── */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
}
"""