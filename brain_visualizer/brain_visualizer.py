"""
brain_visualizer.py  —  Holographic Brain Renderer + Particle System + HUD
============================================================================
OpenGL 3.3 hologram — 7 Yeo functional networks, Fresnel shader, neural particles.

CONTROLS  (shown on-screen bottom-right)
---------
  Scroll wheel  : zoom in / out
  Left-drag     : orbit
  Right-drag    : pan
  SPACE         : pause / resume auto-orbit
  R             : reset camera
  ESC / Q       : quit
"""

import ctypes, math, os, sys
from typing import Dict, Optional
import numpy as np
import glfw
from OpenGL import GL as gl

from obj_loader import load_brain_mesh
from network_mapping import (
    create_default_brain_state, create_region_network_mapping,
    BrainState, NEON_COLORS, NETWORKS,
)
from shaders import create_brain_shader_program
from particles import ParticleSystem


# ── Global camera (plain dict — pyglfw wraps window in new object each callback)
_CAM: dict = {}

def _make_camera() -> dict:
    _CAM.clear()
    _CAM.update({
        'radius':      3.5,
        'azimuth':     0.0,
        'elevation':   0.15,
        'target':      np.zeros(3, dtype=np.float64),
        'mouse_left':  False,
        'mouse_right': False,
        'last_x':      0.0,
        'last_y':      0.0,
        'auto_orbit':  True,
        'orbit_speed': 0.10,
    })
    return _CAM


# ── GLFW callbacks ────────────────────────────────────────────────────────────
def _key_cb(window, key, sc, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_SPACE and action == glfw.PRESS:
        _CAM['auto_orbit'] = not _CAM['auto_orbit']
    elif key == glfw.KEY_R and action == glfw.PRESS:
        _CAM.update({'radius': 3.5, 'azimuth': 0.0, 'elevation': 0.15})
        _CAM['target'][:] = 0.0

def _mouse_btn_cb(window, btn, action, mods):
    pressed = (action == glfw.PRESS)
    if btn == glfw.MOUSE_BUTTON_LEFT:  _CAM['mouse_left']  = pressed
    if btn == glfw.MOUSE_BUTTON_RIGHT: _CAM['mouse_right'] = pressed
    if pressed:
        x, y = glfw.get_cursor_pos(window)
        _CAM['last_x'], _CAM['last_y'] = x, y

def _cursor_cb(window, x, y):
    dx = x - _CAM['last_x'];  dy = y - _CAM['last_y']
    _CAM['last_x'], _CAM['last_y'] = x, y
    if _CAM['mouse_left']:
        _CAM['auto_orbit'] = False
        _CAM['azimuth']   -= dx * 0.005
        _CAM['elevation']  = max(-1.4, min(1.4, _CAM['elevation'] - dy * 0.005))
    elif _CAM['mouse_right']:
        right = np.array([math.cos(_CAM['azimuth']), 0., -math.sin(_CAM['azimuth'])])
        spd = _CAM['radius'] * 0.001
        _CAM['target'] += right * (-dx * spd) + np.array([0., 1., 0.]) * (dy * spd)

def _scroll_cb(window, xoff, yoff):
    _CAM['radius'] = max(0.5, min(12.0, _CAM['radius'] * (0.92 if yoff > 0 else 1/0.92)))


# ── GL helpers ─────────────────────────────────────────────────────────────────
class GLResources:
    vao = vbo_vertices = vbo_normals = vbo_net_index = ibo = program = None
    num_indices = 0
    ulocs: Dict[str, int] = None

def _init_glfw(w=1280, h=720, title="PandoraBOX — Cognitive Brain"):
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    win = glfw.create_window(w, h, title, None, None)
    if not win:
        glfw.terminate(); raise RuntimeError("Window creation failed")
    glfw.make_context_current(win)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    gl.glDisable(gl.GL_CULL_FACE)   # transparent hologram needs both sides
    return win

def _create_gl_resources(mesh, region_to_network) -> GLResources:
    res = GLResources()
    verts   = np.array(mesh.vertices, dtype=np.float32)
    idx     = np.array(mesh.indices,  dtype=np.uint32);  res.num_indices = idx.size
    norms   = (np.array(mesh.normals, dtype=np.float32)
               if mesh.normals and len(mesh.normals) == len(mesh.vertices)
               else np.zeros_like(verts))
    n2i     = {n: i for i, n in enumerate(NETWORKS)}
    net_idx = np.array([float(n2i.get(region_to_network.get(r, NETWORKS[0]), 0))
                        for r in mesh.region_ids], dtype=np.float32)

    vao = gl.glGenVertexArrays(1);  gl.glBindVertexArray(vao)
    def _vbo(data, loc, size):
        buf = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        return buf
    res.vbo_vertices  = _vbo(verts,   0, 3)
    res.vbo_normals   = _vbo(norms,   1, 3)
    res.vbo_net_index = _vbo(net_idx, 2, 1)
    ibo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
    gl.glBindVertexArray(0)
    res.vao = vao;  res.ibo = ibo
    res.program = create_brain_shader_program()
    p = res.program
    res.ulocs = {n: gl.glGetUniformLocation(p, n) for n in
                 ("uModel","uView","uProj","uEyePos",
                  "uNetworkActivation","uNetworkColors",
                  "uFresnelPower","uFresnelIntensity","uBaseAlpha")}
    return res

def _persp(fovy, aspect, zn, zf):
    f = 1.0 / math.tan(fovy / 2)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0]=f/aspect; m[1,1]=f; m[2,2]=(zf+zn)/(zn-zf)
    m[2,3]=2*zf*zn/(zn-zf); m[3,2]=-1.0
    return m

def _look_at(eye, center, up):
    f = np.array(center) - np.array(eye);  f = f / np.linalg.norm(f)
    u = np.array(up, dtype=np.float64);    u = u / np.linalg.norm(u)
    s = np.cross(f, u);                    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0,:3]=s;  m[1,:3]=u;  m[2,:3]=-f
    m[0,3]=-np.dot(s,eye);  m[1,3]=-np.dot(u,eye);  m[2,3]=np.dot(f,eye)
    return m


# ── Particle shader ───────────────────────────────────────────────────────────
_PART_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uViewProj;
uniform float uPointSize;
out vec4 vColor;
void main(){
    vColor = aColor;
    gl_Position = uViewProj * vec4(aPos, 1.0);
    gl_PointSize = uPointSize * aColor.a + 1.5;
}
"""
_PART_FRAG = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    vec2 c = gl_PointCoord - 0.5;
    float d = dot(c, c);
    if(d > 0.25) discard;
    float alpha = vColor.a * (1.0 - d * 4.0);
    FragColor = vec4(vColor.rgb, alpha);
}
"""

def _compile_shader(src, stype):
    s = gl.glCreateShader(stype)
    gl.glShaderSource(s, src);  gl.glCompileShader(s)
    if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
        raise RuntimeError("Shader compile:\n" + gl.glGetShaderInfoLog(s).decode(errors='replace'))
    return s

def _make_particle_program():
    vs = _compile_shader(_PART_VERT, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(_PART_FRAG, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs);  gl.glAttachShader(prog, fs);  gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError("Particle link:\n" + gl.glGetProgramInfoLog(prog).decode(errors='replace'))
    gl.glDeleteShader(vs);  gl.glDeleteShader(fs)
    return prog

MAX_PARTICLES = 6000
_part_vao = _part_vbo_pos = _part_vbo_col = _part_prog = None
_part_uloc_vp = _part_uloc_ps = None

def _init_particle_gl():
    global _part_vao, _part_vbo_pos, _part_vbo_col, _part_prog, _part_uloc_vp, _part_uloc_ps
    _part_prog    = _make_particle_program()
    _part_uloc_vp = gl.glGetUniformLocation(_part_prog, "uViewProj")
    _part_uloc_ps = gl.glGetUniformLocation(_part_prog, "uPointSize")
    _part_vao = gl.glGenVertexArrays(1);  gl.glBindVertexArray(_part_vao)
    _part_vbo_pos = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _part_vbo_pos)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, MAX_PARTICLES*3*4, None, gl.GL_DYNAMIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    _part_vbo_col = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _part_vbo_col)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, MAX_PARTICLES*4*4, None, gl.GL_DYNAMIC_DRAW)
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glBindVertexArray(0)

def _draw_particles(ps: ParticleSystem, vp: np.ndarray):
    positions, colors = ps.get_render_data()
    n = len(positions) // 3
    if n == 0: return
    pos_arr = np.array(positions[:MAX_PARTICLES*3], dtype=np.float32)
    col_arr = np.array(colors[:MAX_PARTICLES*4],    dtype=np.float32)
    gl.glUseProgram(_part_prog)
    gl.glUniformMatrix4fv(_part_uloc_vp, 1, gl.GL_TRUE, vp)
    gl.glUniform1f(_part_uloc_ps, 12.0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _part_vbo_pos)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, pos_arr.nbytes, pos_arr)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _part_vbo_col)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, col_arr.nbytes, col_arr)
    gl.glBindVertexArray(_part_vao)
    gl.glDepthMask(gl.GL_FALSE)
    # Additive blending: particles ADD their color to background → neon glow effect
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glDrawArrays(gl.GL_POINTS, 0, min(n, MAX_PARTICLES))
    # Restore normal alpha blending for HUD
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glDepthMask(gl.GL_TRUE)
    gl.glBindVertexArray(0)


# ── HUD overlay ───────────────────────────────────────────────────────────────
_HUD_VERT = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
"""
_HUD_FRAG = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main(){ FragColor = texture(uTex, vUV); }
"""

_hud_prog = _hud_vao = _hud_tex = _hud_vbo = None
_hud_W = _hud_H = 0   # window size at last HUD build

# PandoraBOX-specific network labels (cognitive meaning, not just anatomical name)
NETWORK_LABELS = {
    "DMN": "Inner Monologue / Narrative",
    "DAN": "Curiosity / Epistemic Drive",
    "VAN": "Surprise / Salience",
    "SMN": "Expression / Output",
    "VIS": "Perception / Vision",
    "LIM": "Social / Relational",
    "FPN": "Goals / Executive Control",
}

def _build_hud_texture():
    """Render controls + legend into a single Pillow image → OpenGL texture."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None, 0, 0

    pad       = 14
    font_size = 14
    font = None
    for name in ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf", "Courier New.ttf",
                 "lucon.ttf", "Lucida Console.ttf"]:
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    line_h = font_size + 5

    # ── Controls section ──────────────────────────────────────────────────
    ctrl_lines = [
        ("CONTROLS", "header"),
        ("",          "gap"),
        ("Scroll      zoom in / out",  "text"),
        ("Left drag   orbit",          "text"),
        ("Right drag  pan",            "text"),
        ("SPACE       pause / resume", "text"),
        ("R           reset camera",   "text"),
        ("ESC / Q     quit",           "text"),
    ]

    # ── Legend section ────────────────────────────────────────────────────
    legend_lines = [("NETWORKS", "header"), ("", "gap")]
    for net in NETWORKS:
        r, g, b = NEON_COLORS[net]
        label = NETWORK_LABELS[net]
        legend_lines.append((f"{net}  {label}", "net", net))

    all_sections = [ctrl_lines, legend_lines]

    # Measure total width needed
    dummy = Image.new("RGBA", (1, 1))
    dd = ImageDraw.Draw(dummy)
    def tw(text): return int(dd.textlength(text, font=font))

    swatch = 12   # color square size
    swatch_gap = 6

    max_w = 0
    for section in all_sections:
        for item in section:
            extra = swatch + swatch_gap if len(item) > 2 else 0
            max_w = max(max_w, tw(item[0]) + extra)

    img_w = max_w + pad * 2 + 4
    total_lines = sum(len(s) for s in all_sections) + 1  # +1 gap between sections
    img_h = total_lines * line_h + pad * 2

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, img_w-1, img_h-1], radius=8,
                            fill=(8, 8, 25, 210))

    y = pad
    for si, section in enumerate(all_sections):
        if si > 0:
            y += line_h  # gap between sections
        for item in section:
            text, kind = item[0], item[1]
            if kind == "gap":
                y += line_h // 2
                continue
            if kind == "header":
                draw.text((pad, y), text, font=font, fill=(0, 220, 255, 255))
            elif kind == "net":
                net = item[2]
                r, g, b = NEON_COLORS[net]
                ri, gi, bi = int(r*255), int(g*255), int(b*255)
                # Colored square swatch
                sx = pad
                draw.rectangle([sx, y+2, sx+swatch, y+swatch+2],
                                fill=(ri, gi, bi, 230))
                draw.text((pad + swatch + swatch_gap, y), text,
                          font=font, fill=(ri, gi, bi, 220))
            else:
                draw.text((pad, y), text, font=font, fill=(190, 190, 215, 210))
            y += line_h

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    data = img.tobytes("raw", "RGBA")
    return data, img_w, img_h

def _init_hud():
    global _hud_prog, _hud_vao, _hud_tex, _hud_vbo
    vs = _compile_shader(_HUD_VERT, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(_HUD_FRAG, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs);  gl.glAttachShader(prog, fs);  gl.glLinkProgram(prog)
    gl.glDeleteShader(vs);  gl.glDeleteShader(fs)
    _hud_prog = prog

    _hud_vao = gl.glGenVertexArrays(1);  gl.glBindVertexArray(_hud_vao)
    _hud_vbo = gl.glGenBuffers(1);  gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _hud_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, 4*4*4, None, gl.GL_DYNAMIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
    gl.glBindVertexArray(0)

    tex_data, tw, th = _build_hud_texture()
    _hud_tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, _hud_tex)
    if tex_data:
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, tw, th, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, tex_data)
    else:
        # Fallback: 1×1 transparent pixel
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 1, 1, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, b'\x00\x00\x00\x00')
        tw, th = 0, 0
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tw, th

def _draw_hud(win_w, win_h, tex_w, tex_h):
    if not _hud_prog or tex_w == 0:
        return
    margin = 12
    # NDC coords for top-right quad
    x0 =  1.0 - 2.0 * (tex_w + margin) / win_w
    x1 =  1.0 - 2.0 *  margin          / win_w
    y1 =  1.0 - 2.0 *  margin          / win_h
    y0 =  1.0 - 2.0 * (tex_h + margin) / win_h
    quad = np.array([
        x0, y1,  0.0, 1.0,
        x1, y1,  1.0, 1.0,
        x0, y0,  0.0, 0.0,
        x1, y0,  1.0, 0.0,
    ], dtype=np.float32)
    gl.glUseProgram(_hud_prog)
    gl.glBindVertexArray(_hud_vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _hud_vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, quad.nbytes, quad)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, _hud_tex)
    gl.glUniform1i(gl.glGetUniformLocation(_hud_prog, "uTex"), 0)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glBindVertexArray(0)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_brain_visualizer(base_dir: str = ".", shared_state: Optional[BrainState] = None):
    import ctypes   # needed for HUD VBO offset pointer

    mesh = load_brain_mesh(base_dir)
    r2n  = create_region_network_mapping(mesh.region_ids)

    if shared_state is None:
        shared_state = create_default_brain_state()
        for net, val in zip(NETWORKS, [0.7, 0.7, 0.35, 0.3, 0.05, 0.15, 0.5]):
            shared_state.set_region(net, val)

    win = _init_glfw()
    cam = _make_camera()

    glfw.set_key_callback(win,          _key_cb)
    glfw.set_mouse_button_callback(win, _mouse_btn_cb)
    glfw.set_cursor_pos_callback(win,   _cursor_cb)
    glfw.set_scroll_callback(win,       _scroll_cb)

    res     = _create_gl_resources(mesh, r2n)
    _init_particle_gl()
    hud_tw, hud_th = _init_hud()

    ps         = ParticleSystem(max_particles=MAX_PARTICLES)
    # Precompute per-network surface vertices for localized particle spawning
    ps.set_surface_data(mesh.vertices, mesh.normals, mesh.region_ids, r2n)
    neon       = np.array([c for net in NETWORKS for c in NEON_COLORS[net]], dtype=np.float32)
    ul         = res.ulocs
    last       = glfw.get_time()
    spawn_acc  = 0.0

    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = glfw.get_time();  dt = min(now - last, 0.05);  last = now

        if cam['auto_orbit']:
            cam['azimuth'] += cam['orbit_speed'] * dt

        r, az, el = cam['radius'], cam['azimuth'], cam['elevation']
        eye = cam['target'] + np.array([
            r * math.cos(el) * math.sin(az),
            r * math.sin(el),
            r * math.cos(el) * math.cos(az),
        ])

        W, H = glfw.get_framebuffer_size(win)
        proj = _persp(math.radians(45.0), W / max(1, H), 0.05, 20.0)
        view = _look_at(eye, cam['target'], [0, 1, 0])
        vp   = (proj @ view).astype(np.float32)

        # Spawn particles — 15 per active network every 50 ms
        spawn_acc += dt
        if spawn_acc > 0.05:
            spawn_acc = 0.0
            for net in NETWORKS:
                act = shared_state.get_region_activation(net)
                if act > 0.30:   # networks with meaningful activation emit
                    # count scales as act^2 so low activity emits very little
                    count = max(1, int(act * act * 20))
                    ps.spawn_for_network(net, shared_state, count=count)
        ps.update(dt)

        gl.glViewport(0, 0, W, H)
        gl.glClearColor(0.0, 0.0, 0.02, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # ── Brain mesh
        model = np.identity(4, dtype=np.float32)
        gl.glUseProgram(res.program)
        gl.glUniformMatrix4fv(ul["uModel"], 1, gl.GL_TRUE, model)
        gl.glUniformMatrix4fv(ul["uView"],  1, gl.GL_TRUE, view)
        gl.glUniformMatrix4fv(ul["uProj"],  1, gl.GL_TRUE, proj)
        gl.glUniform3f(ul["uEyePos"], float(eye[0]), float(eye[1]), float(eye[2]))
        acts = np.array(shared_state.as_uniform_array(), dtype=np.float32)
        gl.glUniform1fv(ul["uNetworkActivation"], len(acts), acts)
        gl.glUniform3fv(ul["uNetworkColors"],     len(NETWORKS), neon)
        gl.glUniform1f(ul["uFresnelPower"],     3.0)
        gl.glUniform1f(ul["uFresnelIntensity"], 0.8)
        gl.glUniform1f(ul["uBaseAlpha"],        0.55)
        gl.glBindVertexArray(res.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, res.num_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # ── Particles
        _draw_particles(ps, vp)

        # ── HUD (bottom-right, no depth test)
        _draw_hud(W, H, hud_tw, hud_th)

        glfw.swap_buffers(win)

    glfw.terminate()


if __name__ == "__main__":
    run_brain_visualizer(os.path.dirname(os.path.abspath(__file__)))
