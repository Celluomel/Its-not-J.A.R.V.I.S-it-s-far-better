"""Symbiotic AI Organism UI - Living Interface Layer"""
from nicegui import ui
import asyncio

# ══════════════════════════════════════════════════════════
# GLOBAL REFERENCES
# ══════════════════════════════════════════════════════════
core_element = None
camera_element = None
chat_container_ref = None
input_container_ref = None
status_bar_ref = None

# ══════════════════════════════════════════════════════════
# CORE STATE CONTROL
# ══════════════════════════════════════════════════════════
import logging as _logging
_organism_logger = _logging.getLogger(__name__)
_last_core_state: str = ''

def set_core_state(state: str):
    """Set the AI core's emotional/activity state.

    Safe to call from any context — NiceGUI element mutations are wrapped in
    try/except so a deleted core_element (tab close, page reload) is silent.
    Deduplicates: only applies CSS and logs when state actually changes.
    """
    global core_element, _last_core_state
    if not core_element:
        return
    if state == _last_core_state:
        return  # no change — skip CSS mutation and log spam
    try:
        core_element.classes(remove='core-listening core-thinking core-speaking core-face')
        if state:
            core_element.classes(add=f'core-{state}')
            _organism_logger.debug(f"🧬 Organism state: {state}")
        else:
            _organism_logger.debug("🧬 Organism state: idle")
        _last_core_state = state
    except Exception:
        # Element was deleted (client disconnected / tab closed) — reset ref
        core_element = None
        _last_core_state = ''

def update_user_hue(hue: int):
    """Update the organism's color based on emotional tone"""
    hue = max(180, min(320, hue))
    ui.run_javascript(f"if (window.userHue !== undefined) {{ window.userHue = {hue}; }}")

def set_camera_feed(frame_data: str):
    """Update the camera feed displayed in the AI eye.

    Called at 10 FPS from a ui.timer — silences errors from deleted elements
    (client disconnect) rather than flooding the console at camera framerate.
    """
    global camera_element
    if not camera_element or not frame_data:
        return
    try:
        camera_element.set_source(f'data:image/jpeg;base64,{frame_data}')
    except Exception:
        # Client disconnected or element deleted — clear ref so we stop trying
        camera_element = None

# ══════════════════════════════════════════════════════════
# UI CONSTRUCTION
# ══════════════════════════════════════════════════════════
def build_organism(show_header=True):
    """Build the complete symbiotic interface"""
    global core_element, camera_element, chat_container_ref, input_container_ref, status_bar_ref

    # Inject CSS & JS directly into the head
    from utils.enhanced_css import ENHANCED_GLOBAL_CSS
    ui.add_head_html(f"<style>\n{ENHANCED_GLOBAL_CSS}\n</style>")
    ui.add_head_html(_JS)

    # 1. Floating Header — fixed bar, always visible above scroll
    if show_header:
        with ui.element('div').classes('floating-core-bar'):
            # status_bar_ref: app.py fills this with name, pills, and action buttons
            status_bar_ref = ui.row().classes('items-center justify-between w-full gap-2')

    # 2. Floating Orbs (Background Decorative)
    ui.element('div').classes('floating-orb orb-1')
    ui.element('div').classes('floating-orb orb-2')
    ui.element('div').classes('floating-orb orb-3')

    # 3. Main Layout: Organism (Left) + Chat (Right)
    with ui.element('div').classes('main-layout'):
        
        # LEFT: Organism Container
        with ui.element('div').classes('organism-container'):
            with ui.element('div').classes('ai-core-wrapper'):
                
                # Triple Orbital Rings
                ui.element('div').classes('orbit-ring orbit-1')
                ui.element('div').classes('orbit-ring orbit-2')
                ui.element('div').classes('orbit-ring orbit-3')

                # AI Core
                core_element = ui.element('div').classes('ai-core')
                with core_element:
                    ui.element('div').classes('core-ripple')
                    ui.element('div').classes('iris-ring')
                    ui.element('div').classes('vascular-network')
                    camera_element = ui.interactive_image().classes('ai-eye')
                    ui.element('div').classes('core-pupil')

                # Bloom Navigation (6 Petals) — NO ui.tooltip() — CSS tooltips via data-label to avoid stuck labels
                with ui.element('div').classes('bloom-container'):
                    petals = [
                        ('🧠', '/llm', 'Neural Core'),
                        ('👁️', '/vision', 'Vision'),
                        ('⚙️', '/settings', 'Settings'),
                        ('💾', '/settings?tab=memory', 'Memory'),
                        ('🎙️', '/settings?tab=voice', 'Voice Lab'),
                        ('✨', '/lumina', 'PandoraBOX Mind'),
                        ('🔬', '/research', 'Research')
                    ]
                    for i, (icon, link, label) in enumerate(petals, 1):
                        petal_div = ui.element('div').classes(f'petal petal-{i}')
                        petal_div.props(f'data-label="{label}"')
                        with petal_div:
                            ui.button(icon, on_click=lambda l=link: ui.navigate.to(l)).props('flat round').classes('petal-btn')

            # ✨ Input Container anchored beneath the eye
            input_container_ref = ui.element('div').classes('w-full mt-6')

        # RIGHT: Chat Container
        chat_container_ref = ui.element('div').classes('chat-container').props('id="lumina-chat"')

    # NOTE: Particle canvas is injected by JS directly into document.body for reliable rendering
    
    return core_element, camera_element, chat_container_ref, input_container_ref, status_bar_ref


# ══════════════════════════════════════════════════════════
# JAVASCRIPT - Full Organism Behaviors & Particle Engine
# ══════════════════════════════════════════════════════════
_JS = """
<script>
// ── Auto-scroll: observe lumina-chat for new content ──────────────────
(function() {
  function attachChatObserver() {
    var chat = document.getElementById('lumina-chat');
    if (!chat) { setTimeout(attachChatObserver, 500); return; }

    // Track if user has manually scrolled up (don't hijack their scroll)
    var userScrolledUp = false;
    chat.addEventListener('scroll', function() {
      var distFromBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight;
      userScrolledUp = distFromBottom > 80;
    });

    var observer = new MutationObserver(function() {
      if (!userScrolledUp) {
        requestAnimationFrame(function() {
          chat.scrollTop = chat.scrollHeight;
        });
      }
    });
    observer.observe(chat, { childList: true, subtree: true, characterData: true });
  }
  document.addEventListener('DOMContentLoaded', attachChatObserver);
  setTimeout(attachChatObserver, 800);
})();

// Core state
window.userHue = 260;
let coreDriftHue = 260;
let surpriseMode = false;
let interactionLevel = 0;

// Core drift evolution
function evolveCoreDrift() {
    coreDriftHue += (Math.random() - 0.5) * 1.5;
    if (coreDriftHue < 180) coreDriftHue = 180;
    if (coreDriftHue > 320) coreDriftHue = 320;
}
setInterval(evolveCoreDrift, 8000);

// Surprise mode
function triggerSurprise() {
    surpriseMode = true;
    setTimeout(() => { surpriseMode = false; }, 20000);
}
setInterval(triggerSurprise, 240000);

// Blend hues
function blendHue() {
    let userWeight = surpriseMode ? 0.4 : 0.7;
    let coreWeight = surpriseMode ? 0.6 : 0.3;
    userWeight += interactionLevel * 0.1;
    
    let finalHue = (window.userHue * userWeight) + (coreDriftHue * coreWeight);
    document.documentElement.style.setProperty('--core-hue', finalHue);
}
setInterval(blendHue, 200);

// Microphone detection
async function initMic() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const ctx = new AudioContext();
        const analyser = ctx.createAnalyser();
        const source = ctx.createMediaStreamSource(stream);
        
        source.connect(analyser);
        analyser.fftSize = 256;
        
        const data = new Uint8Array(analyser.frequencyBinCount);
        let smoothed = 0;
        
        function update() {
            analyser.getByteFrequencyData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i++) sum += data[i];
            let level = sum / data.length / 255;
            
            smoothed = smoothed * 0.92 + level * 0.08;
            document.documentElement.style.setProperty('--mic-level', smoothed);
            
            if (smoothed > 0.1) {
                interactionLevel = Math.min(interactionLevel + 0.001, 0.3);
            } else {
                interactionLevel = Math.max(interactionLevel - 0.0005, 0);
            }
            requestAnimationFrame(update);
        }
        update();
    } catch (e) {
        console.log('Mic access not available or denied.');
    }
}
initMic();

// Circadian rhythm
function updateCircadian() {
    const hour = new Date().getHours();
    if (hour < 6 || hour > 20) {
        document.documentElement.style.setProperty('--core-light', '50%');
        document.documentElement.style.setProperty('--core-sat', '60%');
    } else if (hour < 10) {
        document.documentElement.style.setProperty('--core-light', '65%');
        document.documentElement.style.setProperty('--core-sat', '75%');
    } else if (hour < 17) {
        document.documentElement.style.setProperty('--core-light', '70%');
        document.documentElement.style.setProperty('--core-sat', '80%');
    } else {
        document.documentElement.style.setProperty('--core-light', '55%');
        document.documentElement.style.setProperty('--core-sat', '85%');
    }
}
setInterval(updateCircadian, 60000);
updateCircadian();

// ══════════════════════════════════════════════════════
// SAFE INIT - waits for NiceGUI DOM to settle
// ══════════════════════════════════════════════════════
function safeInit() {
    launchParticles();
    initInputStyle();
}

// Retry until NiceGUI has rendered the full DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(safeInit, 300));
} else {
    setTimeout(safeInit, 300);
}

// ══════════════════════════════════════════════════════
// PETALS: Pure CSS hover handles show/hide.
// NO JS touching bloom inline styles - inline styles
// override CSS :hover and break everything.
// ══════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════
// INPUT BAR STYLE - force dark glass look over Quasar defaults
// ══════════════════════════════════════════════════════
function initInputStyle() {
    const row = document.querySelector('.chat-input-container');
    if (!row) { setTimeout(initInputStyle, 200); return; }
    
    row.style.background = 'rgba(10, 14, 26, 0.85)';
    row.style.backdropFilter = 'blur(20px)';
    row.style.border = '1px solid rgba(99,102,241,0.3)';
    row.style.borderRadius = '16px';
    row.style.padding = '8px 12px';
    row.style.boxShadow = '0 4px 30px rgba(0,0,0,0.5), 0 0 40px hsla(260,70%,50%,0.1)';

    // Also style the Quasar input internals
    const inputs = row.querySelectorAll('.q-field__control, .q-field__native');
    inputs.forEach(el => {
        el.style.background = 'transparent';
        el.style.color = '#e2e8f0';
    });
    console.log('✅ Input bar styled');
}

// ══════════════════════════════════════════════════════
// PARTICLE SYSTEM - Self-injects canvas into body
// ══════════════════════════════════════════════════════
function launchParticles() {
    // Remove any existing canvas
    const old = document.getElementById('particleCanvas');
    if (old) old.remove();

    // Create and inject canvas directly into body
    const canvas = document.createElement('canvas');
    canvas.id = 'particleCanvas';
    canvas.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:3;display:block;';
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    window.addEventListener('resize', () => {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    const config = { particleCount: 80, connectionDistance: 150, particleSpeed: 0.3, mouseInfluence: 100 };
    const mouse = { x: null, y: null };
    window.addEventListener('mousemove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
    window.addEventListener('mouseleave', () => { mouse.x = null; mouse.y = null; });

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * config.particleSpeed;
            this.vy = (Math.random() - 0.5) * config.particleSpeed;
            this.radius = Math.random() * 2 + 1;
            this.opacity = Math.random() * 0.5 + 0.3;
            this.hue = 260;
        }
        update() {
            this.x += this.vx; this.y += this.vy;
            if (this.x < 0) this.x = canvas.width;
            if (this.x > canvas.width) this.x = 0;
            if (this.y < 0) this.y = canvas.height;
            if (this.y > canvas.height) this.y = 0;
            if (mouse.x && mouse.y) {
                const dx = this.x - mouse.x, dy = this.y - mouse.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < config.mouseInfluence && dist > 0) {
                    const force = (config.mouseInfluence - dist) / config.mouseInfluence;
                    this.vx += (dx / dist) * force * 0.1;
                    this.vy += (dy / dist) * force * 0.1;
                }
            }
            this.vx *= 0.99; this.vy *= 0.99;
            const spd = Math.sqrt(this.vx*this.vx + this.vy*this.vy);
            if (spd > config.particleSpeed * 2) {
                this.vx = (this.vx/spd)*config.particleSpeed*2;
                this.vy = (this.vy/spd)*config.particleSpeed*2;
            }
        }
        draw(h) {
            this.hue = h + (Math.random() * 60 - 30);
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue},70%,65%,${this.opacity})`;
            ctx.shadowBlur = 10;
            ctx.shadowColor = `hsla(${this.hue},80%,65%,0.6)`;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    const particles = Array.from({length: config.particleCount}, () => new Particle());

    function drawConnections(h) {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i+1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < config.connectionDistance) {
                    const op = (1 - dist/config.connectionDistance) * 0.25;
                    const g = ctx.createLinearGradient(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
                    g.addColorStop(0, `hsla(${particles[i].hue},60%,55%,${op})`);
                    g.addColorStop(1, `hsla(${particles[j].hue},60%,55%,${op})`);
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = g; ctx.lineWidth = 0.8; ctx.stroke();
                    if (dist < config.connectionDistance * 0.5) {
                        const t = (Date.now() % 2000) / 2000;
                        const px = particles[i].x + (particles[j].x - particles[i].x) * t;
                        const py = particles[i].y + (particles[j].y - particles[i].y) * t;
                        ctx.beginPath();
                        ctx.arc(px, py, 1.5, 0, Math.PI*2);
                        ctx.fillStyle = `hsla(${(particles[i].hue+particles[j].hue)/2},80%,75%,${op*3})`;
                        ctx.fill();
                    }
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const h = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--core-hue')) || 260;
        particles.forEach(p => { p.update(); p.draw(h); });
        drawConnections(h);
        requestAnimationFrame(animate);
    }
    animate();
    console.log('✅ Particles launched:', config.particleCount);
}
</script>
"""