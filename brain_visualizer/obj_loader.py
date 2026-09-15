"""
obj_loader.py — Brain mesh OBJ loader.

FIX 2026-03-10 (Bug 4)
-----------------------
  Problem: _infer_encoding mapped region IDs from the R channel of vertex colors.
           Our neon palette has only 4 unique R values for 7 networks:
             R ∈ {0.10, 0.20, 0.60, 1.00}
           → VAN / VIS / LIM all had R=1.0 → same region_id → 3 networks invisible.

  Fix: The brain.obj generator now encodes the network index (0-6) directly as
       R = index / 6.0 (evenly spaced in [0, 1]).  The loader detects this encoding
       by checking whether all unique R values are multiples of 1/6 within tolerance,
       and if so reads the network index directly — no ambiguity.

  Fallback: If the OBJ uses a different encoding (e.g. an external mesh), the
            original _infer_encoding heuristic still runs.
"""

import os
import math

OBJ_FILENAME = "brain.obj"

# Number of networks — must match network_mapping.NETWORKS length
N_NETWORKS = 7


class BrainMesh:
    def __init__(self, vertices, normals, indices, region_ids):
        self.vertices   = vertices
        self.normals    = normals
        self.indices    = indices
        self.region_ids = region_ids


def _parse_vertex_color(tokens):
    """tokens: list of strings starting after 'v x y z'. Returns (r,g,b) or None."""
    if len(tokens) < 6:
        return None
    try:
        r, g, b = float(tokens[3]), float(tokens[4]), float(tokens[5])
    except ValueError:
        return None
    if r > 1.0 or g > 1.0 or b > 1.0:
        r /= 255.0; g /= 255.0; b /= 255.0
    return (max(0., min(1., r)), max(0., min(1., g)), max(0., min(1., b)))


def _is_network_index_encoding(region_colors) -> bool:
    """
    Check if R values are evenly spaced as idx/(N_NETWORKS-1).
    This is the encoding used by our brain.obj generator.
    Expected values: 0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6
    """
    if not region_colors:
        return False
    step = 1.0 / (N_NETWORKS - 1)
    for r, g, b in region_colors[:500]:   # sample first 500 for speed
        # R should be close to one of the 7 expected values
        nearest = round(r / step) * step
        if abs(r - nearest) > 0.02:       # 2% tolerance
            return False
    return True


def _infer_encoding(region_colors):
    """
    Fallback heuristic for external meshes with unknown encoding.
    Returns a function color_to_region_id((r,g,b)) -> int.
    """
    if not region_colors:
        return lambda c: 0

    rs = [c[0] for c in region_colors]
    unique_rs = sorted(set(rs))

    if len(unique_rs) > 1:
        diffs = [abs(unique_rs[i+1] - unique_rs[i]) for i in range(len(unique_rs) - 1)]
        avg_step = sum(diffs) / len(diffs)
    else:
        avg_step = 0.0

    if max(rs) > 0.5 and 0.002 < avg_step < 0.01:
        # 0-255 integer encoding
        return lambda c: int(round(c[0] * 255.0))
    else:
        n_est = max(1, len(unique_rs))
        return lambda c: int(round(c[0] * (n_est - 1)))


def load_brain_mesh(base_dir: str) -> BrainMesh:
    obj_path = os.path.join(base_dir, OBJ_FILENAME)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    temp_vertices, temp_normals, temp_colors = [], [], []
    indices = []

    print(f"Loading OBJ: {obj_path}")
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.split()
            if not parts:
                continue
            t = parts[0]
            if t == 'v':
                if len(parts) < 4:
                    continue
                temp_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                temp_colors.append(_parse_vertex_color(parts[1:]))
            elif t == 'vn':
                if len(parts) < 4:
                    continue
                temp_normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif t == 'f':
                face = [int(v.split('/')[0]) - 1 for v in parts[1:]]
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        indices.extend([face[0], face[i], face[i + 1]])

    # Determine region encoding
    valid_colors = [c for c in temp_colors if c is not None]

    if _is_network_index_encoding(valid_colors):
        # Fix 4: direct network index decoding — R = net_idx / (N_NETWORKS - 1)
        step = 1.0 / (N_NETWORKS - 1)
        def color_to_region_id(c):
            net_idx = int(round(c[0] / step))
            net_idx = max(0, min(N_NETWORKS - 1, net_idx))
            # Map network index to first region_id of that network
            # 180 regions / 7 networks = 26 each; this gives stable per-network IDs
            return net_idx * 26
        print("Region encoding: network-index (direct, 7 networks)")
    else:
        color_to_region_id = _infer_encoding(valid_colors)
        print("Region encoding: heuristic (external mesh)")

    region_ids = [
        color_to_region_id(c) if c is not None else 0
        for c in temp_colors
    ]

    normals = temp_normals if temp_normals else None
    print(f"Loaded {len(temp_vertices):,} vertices, {len(indices)//3:,} triangles.")
    print(f"Sample region_ids: {region_ids[:8]}")

    return BrainMesh(temp_vertices, normals, indices, region_ids)
