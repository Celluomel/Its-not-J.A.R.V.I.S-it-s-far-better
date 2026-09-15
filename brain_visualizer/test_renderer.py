"""
test_renderer.py — Minimal OpenGL sanity check (no brain.obj needed)
Run: python brain_visualizer/test_renderer.py
If you see a rotating colored sphere → OpenGL pipeline OK.
If window is blank/black → driver / GLFW issue.
"""
import math, sys
import numpy as np
import glfw
from OpenGL import GL as gl

VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
out vec3 vNorm;
out vec3 vPos;
void main(){
    vNorm = aNorm; vPos = aPos;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""
FRAG = """
#version 330 core
in vec3 vNorm; in vec3 vPos;
out vec4 FragColor;
uniform float uTime;
void main(){
    vec3 n = normalize(vNorm);
    vec3 light = normalize(vec3(1.0, 1.5, 2.0));
    float diff = max(dot(n, light), 0.15);
    // Colorful by position
    vec3 col = 0.5 + 0.5*cos(vec3(uTime) + vPos.xyz + vec3(0,2,4));
    FragColor = vec4(col * diff, 0.85);
}
"""

def make_sphere(rings=32, segs=32):
    verts, norms, idx = [], [], []
    for i in range(rings+1):
        phi = math.pi * i / rings
        for j in range(segs+1):
            theta = 2*math.pi * j / segs
            x = math.sin(phi)*math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi)*math.sin(theta)
            verts += [x, y, z]; norms += [x, y, z]
    for i in range(rings):
        for j in range(segs):
            a = i*(segs+1)+j; b = a+1; c = a+(segs+1); d = c+1
            idx += [a,c,b, b,c,d]
    return (np.array(verts, np.float32), np.array(norms, np.float32),
            np.array(idx, np.uint32))

def compile(src, t):
    s = gl.glCreateShader(t); gl.glShaderSource(s, src); gl.glCompileShader(s)
    if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(s).decode())
    return s

def persp(fov, asp, zn, zf):
    f = 1/math.tan(fov/2); m = np.zeros((4,4), np.float32)
    m[0,0]=f/asp; m[1,1]=f; m[2,2]=(zf+zn)/(zn-zf)
    m[2,3]=2*zf*zn/(zn-zf); m[3,2]=-1; return m

def look_at(eye, at, up):
    f=np.array(at)-np.array(eye,dtype=np.float32); f/=np.linalg.norm(f)
    u=np.array(up,dtype=np.float32); u/=np.linalg.norm(u)
    s=np.cross(f,u); s/=np.linalg.norm(s); u=np.cross(s,f)
    m=np.eye(4,dtype=np.float32)
    m[0,:3]=s; m[1,:3]=u; m[2,:3]=-f
    m[0,3]=-s.dot(eye); m[1,3]=-u.dot(eye); m[2,3]=f.dot(eye); return m

if __name__ == "__main__":
    if not glfw.init(): sys.exit("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    win = glfw.create_window(800, 600, "GL Test — rotating sphere", None, None)
    if not win: glfw.terminate(); sys.exit("Window failed")
    glfw.make_context_current(win)

    vs = compile(VERT, gl.GL_VERTEX_SHADER)
    fs = compile(FRAG, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs); gl.glAttachShader(prog, fs); gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())

    V, N, I = make_sphere()
    vao = gl.glGenVertexArrays(1); gl.glBindVertexArray(vao)
    def vbo(d, loc, sz):
        b = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, d.nbytes, d, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, sz, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    vbo(V, 0, 3); vbo(N, 1, 3)
    ibo = gl.glGenBuffers(1); gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, I.nbytes, I, gl.GL_STATIC_DRAW)
    gl.glBindVertexArray(0)

    gl.glEnable(gl.GL_DEPTH_TEST); gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glDisable(gl.GL_CULL_FACE)

    loc_mvp  = gl.glGetUniformLocation(prog, "uMVP")
    loc_time = gl.glGetUniformLocation(prog, "uTime")

    print("✅ OpenGL pipeline OK — you should see a rotating colorful sphere")
    print("   ESC to quit")

    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        t = glfw.get_time()
        W, H = glfw.get_framebuffer_size(win)
        gl.glViewport(0, 0, W, H)
        gl.glClearColor(0, 0, 0.05, 1); gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        eye = np.array([3.5*math.sin(t*0.4), 0.5, 3.5*math.cos(t*0.4)], np.float32)
        P = persp(math.radians(45), W/max(1,H), 0.05, 20)
        V2 = look_at(eye, [0,0,0], [0,1,0])
        MVP = P @ V2

        gl.glUseProgram(prog)
        gl.glUniformMatrix4fv(loc_mvp, 1, gl.GL_TRUE, MVP)
        gl.glUniform1f(loc_time, t)
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(I), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        glfw.swap_buffers(win)

    glfw.terminate()
