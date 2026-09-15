"""Enhanced Modern CSS - Full Bio-Organic Animation Suite"""

ENHANCED_GLOBAL_CSS = """
/* ═══════════════════════════════════════════════════════════
   ENHANCED BLOOM UI - ALIEN BIO-ORGANIC V3
   ═══════════════════════════════════════════════════════════ */

:root {
    --core-hue: 260;
    --core-sat: 70%;
    --core-light: 60%;
    --mic-level: 0;
    --orbit-speed: 18s;
    --bond-level: 0;
    --glow-intensity: 1;
}

/* ── 1. RESET & BASE ─────────────────────────────────────── */
*, *::before, *::after { 
    box-sizing: border-box; 
}

html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;           /* Lock body scroll — only chat-container scrolls */
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #e2e8f0;
    -webkit-font-smoothing: antialiased;
    /* 5-Layered Animated Gradient Background */
    background:
        radial-gradient(circle at 20% 20%, hsla(var(--core-hue), 60%, 25%, 0.8) 0%, transparent 35%),
        radial-gradient(circle at 80% 30%, hsla(calc(var(--core-hue) + 60), 70%, 30%, 0.6) 0%, transparent 40%),
        radial-gradient(circle at 50% 70%, hsla(calc(var(--core-hue) - 30), 65%, 28%, 0.7) 0%, transparent 45%),
        radial-gradient(circle at 30% 90%, hsla(calc(var(--core-hue) + 90), 60%, 25%, 0.5) 0%, transparent 40%),
        linear-gradient(135deg, #0a0e1a 0%, #020617 50%, #0f0a1f 100%);
    background-attachment: fixed;
}

/* Flowing Animated Grid Overlay */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: 
        linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridFlow 20s linear infinite;
}

/* ── 2. PARTICLE CANVAS ──────────────────────────────────── */
/* Canvas is self-injected by JS. This CSS ensures correct display if it exists. */
#particleCanvas {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    pointer-events: none !important;
    z-index: 3 !important;
    display: block !important;
}

/* ── 3. SCROLLBAR & GLOBALS ──────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.6); border-radius: 5px; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, hsla(var(--core-hue), 70%, 50%, 0.6), hsla(var(--core-hue), 70%, 40%, 0.8));
    border-radius: 5px;
    border: 2px solid rgba(15, 23, 42, 0.6);
    transition: all 0.3s;
}
::-webkit-scrollbar-thumb:hover { 
    background: linear-gradient(180deg, hsla(var(--core-hue), 80%, 60%, 0.8), hsla(var(--core-hue), 80%, 50%, 1));
    box-shadow: 0 0 20px hsla(var(--core-hue), 70%, 50%, 0.5);
}

/* ── 4. FLOATING GLASS HEADER ────────────────────────────── */
.floating-core-bar {
    position: fixed !important;
    top: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(1200px, 92%) !important;
    max-width: 1200px !important;
    backdrop-filter: blur(30px) saturate(200%) !important;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.6) 100%) !important;
    border: 2px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 28px !important;
    padding: 18px 36px !important;
    z-index: 9999 !important;
    isolation: isolate !important;
    box-shadow: 
        0 25px 80px rgba(0, 0, 0, 0.6),
        0 0 0 1px rgba(255, 255, 255, 0.08) inset,
        0 0 150px hsla(var(--core-hue), 70%, 50%, 0.25),
        0 10px 40px rgba(99, 102, 241, 0.2) !important;
    animation: headerFloat 10s ease-in-out infinite;
}

.floating-core-bar::before {
    content: '';
    position: absolute;
    inset: -2px;
    background: linear-gradient(135deg, hsla(var(--core-hue), 70%, 50%, 0.3), hsla(calc(var(--core-hue) + 40), 70%, 50%, 0.3));
    border-radius: 28px;
    filter: blur(20px);
    opacity: 0.5;
    z-index: -1;
    animation: glowPulse 4s ease-in-out infinite;
}

.core-logo {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, hsla(var(--core-hue), 80%, 60%, 0.3), hsla(calc(var(--core-hue) + 30), 80%, 50%, 0.3));
    display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.1);
}

.core-title {
    font-size: 1.2rem; font-weight: 600; letter-spacing: 2px;
    background: linear-gradient(135deg, hsl(var(--core-hue), 80%, 70%), hsl(calc(var(--core-hue) + 30), 80%, 60%));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── 5. MAIN LAYOUT & CONTAINERS ─────────────────────────── */
.main-layout {
    position: relative;
    z-index: 10;
    display: flex;
    height: calc(100vh - 80px);
    padding-top: 80px;
    padding-bottom: 0;
    gap: 30px;
    align-items: stretch;
    overflow: hidden;
}

.organism-container {
    flex: 0 0 auto;
    width: 580px;
    min-width: 580px;
    position: relative;
    top: 0;
    height: calc(100vh - 80px);
    max-height: calc(100vh - 80px);
    overflow: visible;             /* Allow petals to show outside bounds */
    display: flex;
    flex-direction: column;
    justify-content: center;       /* Vertically center the orb */
    align-items: center;
    margin-left: 30px;
    z-index: 100;
    padding-top: 20px;
    padding-bottom: 100px;         /* Space below for input bar */
    align-self: flex-start;
}

/* Force NiceGUI's own wrappers to fill the viewport so height:100% works in children */
.q-layout, .q-page-container, .q-page, .nicegui-content {
    height: 100% !important;
    min-height: unset !important;
}

.chat-container {
    flex: 1 1 0%;
    min-width: 0;
    margin-right: 40px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    height: calc(100vh - 160px);
    max-height: calc(100vh - 160px);
    overflow-y: auto;
    overflow-x: hidden;
    scroll-behavior: auto;
    padding-bottom: 80px;
    position: relative;
    z-index: 1;
}

/* ── 6. AI CORE & TRIPLE ORBITS ──────────────────────────── */
/* Single authoritative definition */
/* ai-core-wrapper MUST be >= orbit-3 (520px) so mouse stays inside when hovering rings */
.ai-core-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    width: 560px;
    height: 560px;
}

/* Triple Counter-Rotating Orbital Rings */
.orbit-ring { 
    position: absolute; 
    border-radius: 50%; 
    border: 2px solid transparent; 
    pointer-events: none; 
}
.orbit-1 { 
    width: 480px; height: 480px; 
    border-top-color: hsla(var(--core-hue), 80%, 60%, 0.4); 
    border-right-color: hsla(var(--core-hue), 80%, 60%, 0.1); 
    animation: rotate360 20s linear infinite; 
    filter: blur(1px);
}
.orbit-2 { 
    width: 440px; height: 440px; 
    border-bottom-color: hsla(calc(var(--core-hue) + 40), 70%, 50%, 0.3); 
    border-left-color: hsla(calc(var(--core-hue) + 40), 70%, 50%, 0.1); 
    animation: rotate360 25s linear infinite reverse; 
}
.orbit-3 { 
    width: 520px; height: 520px; 
    border-left-color: hsla(calc(var(--core-hue) - 30), 60%, 40%, 0.2); 
    animation: rotate360 35s linear infinite; 
}

/* Single authoritative .ai-core definition */
.ai-core {
    width: 380px;
    height: 380px;
    position: relative;
    border-radius: 50%;
    animation: symbioticBreath 12s ease-in-out infinite;
    transition: all 0.8s cubic-bezier(.25, .8, .25, 1);
    filter: drop-shadow(0 0 80px hsla(var(--core-hue), 70%, 50%, 0.4));
    transform: scale(calc(1 + var(--mic-level) * 0.08));
}

/* ── 7. EYE & INTERNAL LAYERS ────────────────────────────── */
.ai-eye { 
    width: 100% !important; 
    height: 100% !important; 
    border-radius: 50% !important; 
    border: 4px solid hsla(var(--core-hue), 80%, 60%, 0.8) !important; 
    box-shadow: 
        0 0 60px hsla(var(--core-hue), 80%, 60%, 0.6),
        inset 0 0 30px hsla(var(--core-hue), 80%, 60%, 0.4) !important; 
    overflow: hidden; 
    transition: all 0.5s ease; 
}

/* Dual Iris Rings */
.iris-ring {
    position: absolute; inset: 0; border-radius: 50%;
    background: conic-gradient(from 0deg, hsla(var(--core-hue), 90%, 60%, 0.5) 0%, transparent 20%, hsla(calc(var(--core-hue) + 60), 90%, 60%, 0.5) 40%, transparent 60%, hsla(var(--core-hue), 90%, 60%, 0.5) 100%);
    animation: irisRotate 25s linear infinite; pointer-events: none; filter: blur(1px);
}
.iris-ring::after {
    content: ''; position: absolute; inset: 15%; border-radius: 50%;
    background: conic-gradient(from 180deg, transparent 0%, hsla(calc(var(--core-hue) + 45), 80%, 55%, 0.4) 15%, transparent 30%, hsla(calc(var(--core-hue) + 90), 80%, 55%, 0.4) 45%, transparent 60%);
    animation: irisRotate 35s linear infinite reverse;
}

/* 5 Pulsing Vascular Veins */
.vascular-network {
    position: absolute; inset: 0; border-radius: 50%;
    background: 
        radial-gradient(circle at 20% 20%, hsla(var(--core-hue), 80%, 65%, 0.15) 0%, transparent 12%),
        radial-gradient(circle at 80% 20%, hsla(calc(var(--core-hue) + 30), 80%, 65%, 0.12) 0%, transparent 14%),
        radial-gradient(circle at 50% 80%, hsla(calc(var(--core-hue) + 60), 80%, 65%, 0.13) 0%, transparent 16%),
        radial-gradient(circle at 20% 80%, hsla(calc(var(--core-hue) - 30), 80%, 65%, 0.1) 0%, transparent 13%),
        radial-gradient(circle at 80% 80%, hsla(calc(var(--core-hue) + 90), 80%, 65%, 0.11) 0%, transparent 15%);
    animation: vascularPulse 10s ease-in-out infinite;
    pointer-events: none;
}

/* Multi-layered Pupil Glow */
.core-pupil {
    position: absolute; inset: 35%; border-radius: 50%;
    background: radial-gradient(circle at center, hsla(var(--core-hue), 95%, 75%, 0.8), hsla(var(--core-hue), 85%, 65%, 0.4) 40%, transparent 80%);
    box-shadow: 0 0 50px hsla(var(--core-hue), 95%, 75%, 0.6), 0 0 100px hsla(var(--core-hue), 90%, 70%, 0.4), inset 0 0 30px hsla(var(--core-hue), 95%, 75%, 0.3);
    animation: pupilPulse 5s ease-in-out infinite; pointer-events: none;
}

/* Triple Cascading Ripple Effects */
.core-ripple {
    position: absolute; inset: -30px; border-radius: 50%;
    border: 3px solid hsla(var(--core-hue), 80%, 50%, 0.6);
    animation: ripple 5s ease-out infinite; pointer-events: none;
}
.core-ripple::before, .core-ripple::after {
    content: ''; position: absolute; inset: -20px; border-radius: 50%;
    border: 2px solid hsla(var(--core-hue), 80%, 50%, 0.4);
    animation: ripple 5s ease-out infinite;
}
.core-ripple::before { animation-delay: 1.5s; }
.core-ripple::after { animation-delay: 3s; }

/* ── 8. NAVIGATION PETALS (6 Petals) ─────────────────────── */
/* Visibility is controlled by JavaScript (initPetals) for reliability in NiceGUI/Quasar */
/* Bloom hidden by default — pure CSS hover shows it.
   NO visibility property — it prevents CSS :hover from working.
   NO JS inline styles on this element ever. */
.bloom-container {
    position: absolute;
    width: 550px;
    height: 550px;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.5s cubic-bezier(.25,.8,.25,1);
}

/* CSS hover — works because nothing overwrites with inline styles */
.ai-core-wrapper:hover .bloom-container {
    opacity: 1;
    pointer-events: auto;
}

.petal {
    position: absolute; width: 85px; height: 85px; display: flex; align-items: center; justify-content: center;
    border-radius: 50%; backdrop-filter: blur(20px) saturate(200%);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(99, 102, 241, 0.08));
    border: 2px solid rgba(255, 255, 255, 0.15);
    transition: all 0.5s cubic-bezier(.25, .8, .25, 1); cursor: pointer;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 60px hsla(var(--core-hue), 70%, 60%, 0.2);
}

.petal::before {
    content: ''; position: absolute; inset: -3px; border-radius: 50%;
    background: linear-gradient(135deg, hsla(var(--core-hue), 70%, 50%, 0.4), hsla(calc(var(--core-hue) + 40), 70%, 50%, 0.4));
    filter: blur(15px); opacity: 0; transition: opacity 0.4s; z-index: -1;
}

.petal:hover::before { opacity: 1; }
.petal:hover {
    transform: scale(1.25) translateZ(20px);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(99, 102, 241, 0.2));
    border-color: hsla(var(--core-hue), 70%, 60%, 0.8);
    box-shadow: 0 15px 60px rgba(0, 0, 0, 0.6), 0 0 80px hsla(var(--core-hue), 70%, 60%, 0.5);
}

/* CSS-only petal tooltip using data-label — no NiceGUI tooltip widgets */
.petal[data-label]::after {
    content: attr(data-label);
    position: absolute;
    bottom: -28px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.9);
    color: #e2e8f0;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 6px;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
    border: 1px solid rgba(255,255,255,0.1);
    letter-spacing: 0.3px;
}
.petal:hover[data-label]::after {
    opacity: 1;
}

.petal-btn { font-size: 28px !important; transition: all 0.4s cubic-bezier(.25, .8, .25, 1); }
.petal:hover .petal-btn { font-size: 34px !important; filter: drop-shadow(0 4px 12px rgba(255, 255, 255, 0.6)); }

/* 6 Petal Layout — all within organism-container bounds */
.petal-1 { top: 20px;   left: 50%; transform: translateX(-50%); }
.petal-2 { top: 120px;  right: 10px; }
.petal-3 { bottom: 120px; right: 10px; }
.petal-4 { bottom: 20px; left: 50%; transform: translateX(-50%); }
.petal-5 { bottom: 120px; left: 10px; }
.petal-6 { top: 120px;  left: 10px; }
.petal-7 { top: 50%;    right: -20px; transform: translateY(-50%); }

/* ── 9. STATE FEEDBACK ───────────────────────────────────── */
.core-listening .ai-eye {
    border-color: rgba(74, 222, 128, 0.95) !important;
    box-shadow: 0 0 100px rgba(74, 222, 128, 0.7), 0 0 200px rgba(74, 222, 128, 0.4), inset 0 0 80px rgba(74, 222, 128, 0.2) !important;
    animation: listeningPulse 2.5s ease-in-out infinite;
}
.core-thinking .ai-eye {
    border-color: rgba(96, 165, 250, 0.95) !important;
    animation: thinkingPulse 3.5s ease-in-out infinite;
}
.core-speaking .ai-eye {
    border-color: rgba(192, 132, 252, 0.95) !important;
    box-shadow: 0 0 120px rgba(192, 132, 252, 0.8), 0 0 240px rgba(192, 132, 252, 0.5), inset 0 0 60px rgba(192, 132, 252, 0.3) !important;
    animation: speakingWave 1.8s ease-in-out infinite;
}
.core-face .core-ripple {
    border-color: rgba(248, 113, 113, 0.95);
    animation: faceDetected 2.5s ease-in-out infinite;
}

/* ── 10. FLOATING ORBS (DECORATIVE) ──────────────────────── */
.floating-orb { position: fixed; border-radius: 50%; pointer-events: none; z-index: 2; filter: blur(60px); opacity: 0.3; }
.orb-1 { width: 300px; height: 300px; background: radial-gradient(circle, hsla(var(--core-hue), 70%, 50%, 0.6), transparent); top: 20%; left: 10%; animation: float1 20s ease-in-out infinite; }
.orb-2 { width: 250px; height: 250px; background: radial-gradient(circle, hsla(calc(var(--core-hue) + 60), 70%, 50%, 0.5), transparent); top: 60%; right: 15%; animation: float2 25s ease-in-out infinite; }
.orb-3 { width: 200px; height: 200px; background: radial-gradient(circle, hsla(calc(var(--core-hue) + 120), 70%, 50%, 0.4), transparent); bottom: 20%; left: 20%; animation: float3 30s ease-in-out infinite; }

/* ── 11. GLASS CARD ──────────────────────────────────────── */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

/* ── 12. STATUS BAR & PILLS ──────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
.pill-active  { background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.35); color: #a5b4fc; }
.pill-tts     { background: rgba(168, 85, 247, 0.2); border: 1px solid rgba(168, 85, 247, 0.35); color: #d8b4fe; }
.pill-vision  { background: rgba(34, 197, 94, 0.2);  border: 1px solid rgba(34, 197, 94, 0.35);  color: #86efac; }
.pill-memory  { background: rgba(234, 179, 8, 0.2);  border: 1px solid rgba(234, 179, 8, 0.35);  color: #fde047; }
.pill-clone-on{ background: rgba(236, 72, 153, 0.2); border: 1px solid rgba(236, 72, 153, 0.35); color: #f9a8d4; }

/* Status bar settings button - controlled size, no overlap */
.status-bar-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 10px 16px;
    flex-wrap: wrap;
    gap: 8px;
}

/* FIX: settings button must not overflow the row */
.settings-btn,
.settings-btn.q-btn,
.settings-btn .q-btn__wrapper {
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    min-height: 34px !important;
    padding: 4px !important;
    font-size: 16px !important;
    flex-shrink: 0 !important;
}

/* Toggle button states — these beat Quasar's .text-primary default */
.toggle-btn-off .q-icon,
.toggle-btn-off { color: #475569 !important; }
.toggle-btn-indigo .q-icon,
.toggle-btn-indigo { color: #818cf8 !important; }
.toggle-btn-amber .q-icon,
.toggle-btn-amber { color: #f59e0b !important; }
.toggle-btn-red .q-icon,
.toggle-btn-red { color: #ef4444 !important; }

/* ── 13. CHAT INPUT AREA ─────────────────────────────────── */
.chat-input-container {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    padding: 10px 14px !important;
    background: rgba(10, 14, 26, 0.88) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 18px !important;
    box-shadow: 
        0 4px 30px rgba(0, 0, 0, 0.5),
        0 0 40px hsla(var(--core-hue), 70%, 50%, 0.08),
        inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.chat-input-container:focus-within {
    border-color: hsla(var(--core-hue), 70%, 60%, 0.6) !important;
    box-shadow: 
        0 4px 30px rgba(0, 0, 0, 0.5),
        0 0 60px hsla(var(--core-hue), 70%, 50%, 0.2),
        inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* Override ALL Quasar input field internals */
.chat-input-container .q-field,
.chat-input-container .q-field__inner,
.chat-input-container .q-field__control {
    background: transparent !important;
    color: #e2e8f0 !important;
    border: none !important;
    box-shadow: none !important;
    min-height: unset !important;
}
.chat-input-container .q-field__native,
.chat-input-container .q-field__input {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    padding: 0 !important;
}
.chat-input-container .q-field__native::placeholder {
    color: rgba(148, 163, 184, 0.5) !important;
}
/* Remove Quasar underline/outline on focused input */
.chat-input-container .q-field--outlined .q-field__control::before,
.chat-input-container .q-field--outlined .q-field__control::after {
    display: none !important;
}

/* voice mode select – compact */
.mode-select {
    min-width: 85px !important;
    max-width: 105px !important;
    flex-shrink: 0 !important;
}
.mode-select .q-field__control {
    height: 36px !important;
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}
.mode-select .q-field__native { color: #e2e8f0 !important; font-size: 0.85rem !important; }

/* text input wrapper */
.input-wrap { flex: 1 1 0%; min-width: 0; }
.input-wrap .q-field { width: 100% !important; }

/* send button */
.send-btn {
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    background: linear-gradient(135deg, hsla(var(--core-hue), 70%, 50%, 0.8), hsla(calc(var(--core-hue) + 30), 70%, 50%, 0.8)) !important;
    border-radius: 50% !important;
    flex-shrink: 0;
    transition: all 0.3s !important;
}
.send-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 0 20px hsla(var(--core-hue), 70%, 50%, 0.5) !important;
}

/* ── 14. CONNECTION STATUS ───────────────────────────────── */
.connection-status {
    position: fixed;
    bottom: 16px;
    right: 20px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 500;
    z-index: 999;
    transition: all 0.4s;
}
.connection-status.connected {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #86efac;
}
.connection-status.disconnected {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
}

/* ── 15. CHAT MESSAGES ───────────────────────────────────── */
.bubble-user {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    color: #fff;
    border-radius: 20px 20px 5px 20px;
    padding: 11px 18px;
    max-width: 74%;
    align-self: flex-end;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    position: relative;
    z-index: 1;
}
.bubble-bot, .bubble-assistant {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    color: #e2e8f0;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 20px 20px 20px 5px;
    padding: 11px 18px;
    max-width: 80%;
    animation: slideIn 0.3s ease-out;
    position: relative;
    z-index: 1;
}

/* Typing indicator */
.typing-bubble {
    display: flex; align-items: center; gap: 6px;
    padding: 12px 18px;
    background: rgba(30, 41, 59, 0.7);
    border-radius: 20px 20px 20px 5px;
    width: fit-content;
}
.typing-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: hsla(var(--core-hue), 70%, 60%, 0.8);
    animation: typingBounce 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── 16. GLOW TEXT ───────────────────────────────────────── */
.glow-text {
    background: linear-gradient(135deg, hsl(var(--core-hue), 80%, 70%), hsl(calc(var(--core-hue) + 30), 80%, 60%));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 10px hsla(var(--core-hue), 70%, 50%, 0.4));
}

/* ── 17. STATUS DOT ──────────────────────────────────────── */
.status-dot {
    display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #4ade80;
    box-shadow: 0 0 30px #4ade80, 0 0 60px rgba(74, 222, 128, 0.5); 
    animation: statusPulse 2.5s ease-in-out infinite; 
    position: relative;
}

/* ── 18. LOADING OVERLAY ─────────────────────────────────── */
.loading-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center;
    z-index: 10000;
}

/* ── 19. SAVE BTN ────────────────────────────────────────── */
.save-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
}

/* ── 20. ANIMATION KEYFRAMES ─────────────────────────────── */
@keyframes fluidShift { 0% { background-position: 20% 20%; } 100% { background-position: 25% 25%; } }
@keyframes gridFlow { 0% { transform: translate(0, 0); } 100% { transform: translate(40px, 40px); } }
@keyframes rotate360 { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes symbioticBreath { 0%, 100% { transform: scale(calc(1 + var(--mic-level) * 0.08)); filter: brightness(1); } 50% { transform: scale(calc(1.05 + var(--mic-level) * 0.08)); filter: brightness(1.15); } }
@keyframes vascularPulse { 0%, 100% { opacity: 0.7; transform: scale(1); } 50% { opacity: 1; transform: scale(1.03); } }
@keyframes pupilPulse { 0%, 100% { transform: scale(1); opacity: 0.7; } 50% { transform: scale(1.15); opacity: 1; } }
@keyframes ripple { 0% { transform: scale(1); opacity: 0.8; } 100% { transform: scale(1.6); opacity: 0; } }
@keyframes irisRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes listeningPulse { 0%, 100% { transform: scale(calc(1 + var(--mic-level)*0.08)); filter: brightness(1); } 50% { transform: scale(calc(1.08 + var(--mic-level)*0.08)); filter: brightness(1.2); } }
@keyframes thinkingPulse { 0%, 100% { box-shadow: 0 0 80px rgba(96, 165, 250, 0.6), 0 0 160px rgba(96, 165, 250, 0.3); } 50% { box-shadow: 0 0 160px rgba(96, 165, 250, 0.9), 0 0 240px rgba(96, 165, 250, 0.5); } }
@keyframes speakingWave { 0%, 100% { transform: scale(1); } 15% { transform: scale(1.04); } 30% { transform: scale(0.98); } 45% { transform: scale(1.02); } 60% { transform: scale(0.99); } 75% { transform: scale(1.01); } }
@keyframes faceDetected { 0%, 100% { transform: scale(1); opacity: 0.9; box-shadow: 0 0 60px rgba(248, 113, 113, 0.5); } 50% { transform: scale(1.25); opacity: 0.3; box-shadow: 0 0 100px rgba(248, 113, 113, 0.7); } }
@keyframes float1 { 0%, 100% { transform: translate(0, 0); } 33% { transform: translate(30px, -30px); } 66% { transform: translate(-20px, 20px); } }
@keyframes float2 { 0%, 100% { transform: translate(0, 0); } 33% { transform: translate(-40px, 30px); } 66% { transform: translate(20px, -25px); } }
@keyframes float3 { 0%, 100% { transform: translate(0, 0); } 33% { transform: translate(25px, 35px); } 66% { transform: translate(-30px, -20px); } }
@keyframes headerFloat { 0%, 100% { transform: translateX(-50%) translateY(0px) scale(1); } 50% { transform: translateX(-50%) translateY(-8px) scale(1.01); } }
@keyframes glowPulse { 0%, 100% { opacity: 0.5; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.05); } }
@keyframes statusPulse { 0%, 100% { opacity: 0.7; transform: scale(1); } 50% { opacity: 1; transform: scale(1.3); } }
@keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes typingBounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
@keyframes colorCycle { 0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(30deg); } }

/* ── 21. RESPONSIVE ──────────────────────────────────────── */
@media (max-width: 1100px) {
    .main-layout { flex-direction: column; align-items: center; padding-top: 90px; }
    .organism-container { position: relative; top: 0; width: 100%; min-width: unset; margin-left: 0; align-items: center; }
    .chat-container { width: 100%; margin-right: 0; padding: 0 16px; }
    .ai-core-wrapper { width: 360px; height: 360px; }
    .ai-core { width: 300px; height: 300px; }
    .orbit-1 { width: 390px; height: 390px; }
    .orbit-2 { width: 350px; height: 350px; }
    .orbit-3 { width: 420px; height: 420px; }
    .bloom-container { width: 420px; height: 420px; }
}

/* ══════════════════════════════════════════════════════════
   22. SETTINGS PAGE — App header, tabs, cards, dark inputs
   ══════════════════════════════════════════════════════════ */

/* Override Quasar's default blue q-header */
.q-header,
.q-header .q-toolbar,
header.q-header {
    background: rgba(10, 14, 26, 0.95) !important;
    backdrop-filter: blur(30px) saturate(180%) !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important;
    min-height: 64px !important;
}

/* Settings / Vision page header */
.app-header {
    background: rgba(10, 14, 26, 0.95) !important;
    backdrop-filter: blur(30px) !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5) !important;
}

.app-title {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    background: linear-gradient(135deg, hsl(var(--core-hue), 80%, 70%), hsl(calc(var(--core-hue) + 30), 80%, 60%)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* Settings card wrapper */
.settings-card {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 20px !important;
    overflow: visible !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* Tab bar — sticky so it stays visible while content scrolls */
.settings-tabs {
    background: rgba(10, 14, 26, 0.95) !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.15) !important;
    padding: 0 8px !important;
    position: sticky !important;
    top: 64px !important;
    z-index: 10 !important;
    border-radius: 20px 20px 0 0 !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: none !important;
}
.settings-tabs .q-tab {
    color: rgba(148, 163, 184, 0.7) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    min-height: 48px !important;
    padding: 0 16px !important;
    transition: color 0.2s !important;
}
.settings-tabs .q-tab--active {
    color: hsl(var(--core-hue), 80%, 70%) !important;
}
.settings-tabs .q-tab__indicator {
    background: linear-gradient(90deg, hsl(var(--core-hue), 80%, 60%), hsl(calc(var(--core-hue)+30), 80%, 60%)) !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}
.settings-tabs .q-tab:hover {
    color: #e2e8f0 !important;
    background: rgba(99, 102, 241, 0.08) !important;
}

/* Tab panels — scrollable content area */
.settings-panel {
    background: transparent !important;
    overflow-y: visible !important;
    min-height: 200px !important;
}

/* Section label */
.section-label {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: hsla(var(--core-hue), 70%, 65%, 0.8) !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
    display: block !important;
}

/* Info card */
.info-card {
    background: rgba(99, 102, 241, 0.06) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Dark styled inputs — override Quasar outlined */
.dark-input .q-field__control,
.dark-input.q-field .q-field__control {
    background: rgba(10, 14, 26, 0.6) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.dark-input .q-field__control:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
}
.dark-input .q-field__control::before {
    border-color: rgba(99, 102, 241, 0.2) !important;
}
.dark-input .q-field__control::after {
    border-color: hsl(var(--core-hue), 70%, 60%) !important;
}
.dark-input .q-field__native,
.dark-input .q-field__input,
.dark-input input {
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
.dark-input .q-field__label {
    color: rgba(148, 163, 184, 0.6) !important;
    font-size: 0.8rem !important;
}
.dark-input .q-field__marginal { color: rgba(148, 163, 184, 0.5) !important; }

/* Quasar select dropdown menu */
.q-menu {
    background: rgba(15, 23, 42, 0.97) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.6) !important;
}
.q-item { color: #e2e8f0 !important; }
.q-item:hover, .q-item--active { background: rgba(99, 102, 241, 0.15) !important; color: #e2e8f0 !important; }

/* Back to chat link */
.back-link {
    color: hsla(var(--core-hue), 70%, 65%, 0.8) !important;
    font-size: 0.85rem !important;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    transition: color 0.2s !important;
    cursor: pointer !important;
}
.back-link:hover { color: hsl(var(--core-hue), 80%, 70%) !important; }

/* Quasar sliders in settings */
.q-slider__track-container { background: rgba(99, 102, 241, 0.15) !important; }
.q-slider__track { background: linear-gradient(90deg, hsl(var(--core-hue), 70%, 55%), hsl(calc(var(--core-hue)+30), 70%, 55%)) !important; }
.q-slider__thumb { color: hsl(var(--core-hue), 80%, 65%) !important; }

/* Quasar switch */
.q-toggle__inner--truthy .q-toggle__track { background: hsl(var(--core-hue), 70%, 50%) !important; }

/* Expansion panels */
.q-expansion-item .q-item { background: rgba(15, 23, 42, 0.5) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
.q-expansion-item__content { background: rgba(10, 14, 26, 0.4) !important; border-radius: 0 0 10px 10px !important; }

/* Settings footer row */
.settings-footer {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
    padding: 16px 24px !important;
    border-top: 1px solid rgba(99, 102, 241, 0.15) !important;
    background: rgba(10, 14, 26, 0.4) !important;
    border-radius: 0 0 20px 20px !important;
}

"""
