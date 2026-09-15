"""
particles.py — Neural particle system, surface-localized by functional network.

Particles spawn ON the actual brain surface vertices belonging to each network,
not on a uniform sphere. This ensures visual correspondence between lit regions
and particle emission zones.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from network_mapping import NETWORKS, NEON_COLORS, BrainState


@dataclass
class Particle:
    pos:      Tuple[float, float, float]
    vel:      Tuple[float, float, float]
    life:     float
    max_life: float
    color:    Tuple[float, float, float]


class ParticleSystem:
    def __init__(self, max_particles: int = 6000):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        # Per-network surface sample points: list of (x,y,z, nx,ny,nz)
        self._surface: Dict[str, List[Tuple[float,...]]] = {n: [] for n in NETWORKS}

    def set_surface_data(
        self,
        vertices: List[Tuple[float,float,float]],
        normals:  Optional[List[Tuple[float,float,float]]],
        region_ids: List[int],
        region_to_network: Dict[int, str],
    ) -> None:
        """
        Precompute per-network vertex lists from the brain mesh.
        Call once after loading the mesh.
        Samples at most 4000 points per network to keep memory light.
        """
        MAX_PER_NET = 4000
        buckets: Dict[str, List] = {n: [] for n in NETWORKS}

        for i, (v, rid) in enumerate(zip(vertices, region_ids)):
            net = region_to_network.get(rid, NETWORKS[0])
            if len(buckets[net]) < MAX_PER_NET:
                n = normals[i] if normals and i < len(normals) else (0., 1., 0.)
                buckets[net].append((v[0], v[1], v[2], n[0], n[1], n[2]))

        self._surface = buckets
        counts = {n: len(v) for n, v in buckets.items()}
        print(f"[ParticleSystem] Surface vertices per network: {counts}")

    def spawn_for_network(
        self,
        net_name: str,
        brain_state: BrainState,
        count: int = 8,
    ) -> None:
        act = brain_state.get_region_activation(net_name)
        if act <= 0.01:
            return

        surface = self._surface.get(net_name, [])
        base_color = NEON_COLORS[net_name]

        for _ in range(count):
            if surface:
                # Pick a random surface vertex for this network
                sx, sy, sz, nx, ny, nz = random.choice(surface)
                # Small tangential jitter so not all particles stack
                jx = random.gauss(0, 0.015)
                jy = random.gauss(0, 0.015)
                jz = random.gauss(0, 0.015)
                rx = sx + jx;  ry = sy + jy;  rz = sz + jz
            else:
                # Fallback: uniform sphere at brain scale
                theta = random.uniform(0, 2 * math.pi)
                phi   = math.acos(random.uniform(-1, 1))
                r     = 1.15
                rx = r * math.sin(phi) * math.cos(theta)
                ry = r * math.cos(phi)
                rz = r * math.sin(phi) * math.sin(theta)
                nx, ny, nz = rx/r, ry/r, rz/r

            # Velocity: outward along surface normal + small random
            speed = random.uniform(0.04, 0.16) * (0.3 + 0.7 * act)
            vx = nx * speed + random.gauss(0, 0.02)
            vy = ny * speed + random.gauss(0, 0.02)
            vz = nz * speed + random.gauss(0, 0.02)

            life = random.uniform(1.5, 3.5) * (0.4 + 0.6 * act)

            self.particles.append(Particle(
                pos=(rx, ry, rz),
                vel=(vx, vy, vz),
                life=life,
                max_life=life,
                color=base_color,
            ))

        # Trim oldest
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

    def update(self, dt: float) -> None:
        alive = []
        for p in self.particles:
            if p.life <= 0.0:
                continue
            x, y, z = p.pos
            vx, vy, vz = p.vel
            p.pos  = (x + vx * dt, y + vy * dt, z + vz * dt)
            p.life -= dt
            alive.append(p)
        self.particles = alive

    def get_render_data(self):
        positions, colors = [], []
        for p in self.particles:
            positions.extend(p.pos)
            r, g, b = p.color
            alpha = max(0.0, p.life / max(p.max_life, 1e-5))
            # Boost toward white at peak life so all colors are clearly visible
            # against the dark background regardless of hue
            t = alpha  # 1.0 = just spawned, 0.0 = dying
            boost = 0.35 * t
            r2 = min(1.0, r + boost)
            g2 = min(1.0, g + boost)
            b2 = min(1.0, b + boost)
            colors.extend([r2, g2, b2, alpha])
        return positions, colors
