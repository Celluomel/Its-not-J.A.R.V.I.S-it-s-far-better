from OpenGL import GL as gl

VERT_SRC = r"""
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aNetIndex;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out float vNetIndex;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    mat3 normalMat = mat3(transpose(inverse(uModel)));
    vNormal = normalize(normalMat * aNormal);
    vNetIndex = aNetIndex;
    gl_Position = uProj * uView * worldPos;
}
"""

FRAG_SRC = r"""
#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in float vNetIndex;

out vec4 FragColor;

uniform vec3  uEyePos;
uniform float uNetworkActivation[7];
uniform vec3  uNetworkColors[7];
uniform float uFresnelPower;
uniform float uFresnelIntensity;
uniform float uBaseAlpha;

void main()
{
    vec3 V = normalize(uEyePos - vWorldPos);
    vec3 N = normalize(vNormal);

    // Fresnel — strong edge glow
    // Use abs() so back-faces (N pointing away) are also lit — hologram is two-sided
    float ndotv   = abs(dot(N, V));
    float fresnel = pow(1.0 - ndotv, uFresnelPower);

    int   idx = clamp(int(round(vNetIndex)), 0, 6);
    float act = uNetworkActivation[idx];
    vec3  baseColor = uNetworkColors[idx];

    // ── Neon / emissive look ──────────────────────────────────────────────
    // Core surface: dark unless facing camera (adds depth)
    float core = 0.05 + 0.20 * ndotv;

    // Neon bloom: brightest at edges (Fresnel) and scales hard with activation
    float bloom = fresnel * (0.6 + 1.4 * act);

    // Emissive inner glow: always-on dim haze
    float emissive = 0.15 + 0.35 * act;

    // Combine — multiply by 1.8 so colors exceed 1.0 before clamping = saturated neon
    vec3 color = baseColor * (core + bloom + emissive) * 1.8;

    // Hard-clamp with a slight over-saturation trick: push towards white at peaks
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(color, color + 0.15, smoothstep(0.7, 1.2, lum));
    color = clamp(color, 0.0, 1.0);

    // Alpha: translucent base + strong Fresnel rim
    float alpha = uBaseAlpha * (0.4 + 0.6 * act) + uFresnelIntensity * fresnel;
    alpha = clamp(alpha, 0.0, 1.0);

    FragColor = vec4(color, alpha);
}
"""

def _compile_shader(src, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, src)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader

def _link_program(vs, fs):
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(prog).decode("utf-8", errors="ignore")
        gl.glDeleteProgram(prog)
        raise RuntimeError(f"Program link error:\n{log}")
    gl.glDetachShader(prog, vs)
    gl.glDetachShader(prog, fs)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog

def create_brain_shader_program():
    vs = _compile_shader(VERT_SRC, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(FRAG_SRC, gl.GL_FRAGMENT_SHADER)
    return _link_program(vs, fs)
