"""
convert_brain.py — Convert any external brain.obj to PandoraBOX's format
=====================================================================
Usage (from anywhere):
    python brain_visualizer/convert_brain.py 997.obj
    python brain_visualizer/convert_brain.py C:/Users/fred/Downloads/997.obj
"""

import sys, os, math

N_NETWORKS = 7

def normalize_vec(v):
    l = math.sqrt(sum(x*x for x in v))
    return tuple(x/l for x in v) if l > 1e-9 else (0., 1., 0.)

def compute_normals(verts, faces):
    norms = [[0.,0.,0.] for _ in verts]
    for (i0,i1,i2) in faces:
        ax,ay,az=verts[i0]; bx,by,bz=verts[i1]; cx,cy,cz=verts[i2]
        ex,ey,ez=bx-ax,by-ay,bz-az; fx,fy,fz=cx-ax,cy-ay,cz-az
        nx=ey*fz-ez*fy; ny=ez*fx-ex*fz; nz=ex*fy-ey*fx
        for idx in (i0,i1,i2):
            norms[idx][0]+=nx; norms[idx][1]+=ny; norms[idx][2]+=nz
    return [normalize_vec(n) for n in norms]

def assign_net_idx(x, y, z):
    pos = (x*1.7 + y*2.3 + z*1.1 + 2.5) / 5.0
    return int(abs(pos) * N_NETWORKS) % N_NETWORKS

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python brain_visualizer/convert_brain.py path/to/brain.obj")
        sys.exit(1)

    src = sys.argv[1]

    # Search in multiple locations so the user can run from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        src,
        os.path.join(os.getcwd(), src),
        os.path.join(script_dir, src),
        os.path.join(script_dir, '..', src),
    ]
    resolved = None
    for c in candidates:
        if os.path.exists(c):
            resolved = os.path.abspath(c)
            break

    if not resolved:
        print(f"Error: cannot find '{src}'")
        print("Searched in:")
        for c in candidates:
            print(f"  {os.path.abspath(c)}")
        sys.exit(1)

    print(f"Loading {resolved} ...")
    verts, faces = [], []
    with open(resolved, encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == 'v' and not p[0].startswith('vn') and not p[0].startswith('vt'):
                if len(p) >= 4:
                    verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif p[0] == 'f':
                idxs = [int(x.split('/')[0]) - 1 for x in p[1:]]
                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append((idxs[0], idxs[i], idxs[i+1]))

    if len(verts) == 0:
        print("Error: no vertices found in OBJ file")
        sys.exit(1)

    print(f"  {len(verts):,} verts, {len(faces):,} faces")

    # Re-center at origin
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    cz = sum(v[2] for v in verts) / len(verts)
    verts = [(x-cx, y-cy, z-cz) for x,y,z in verts]

    # Normalize scale: 95th-percentile radius → 1.0
    dists = sorted(math.sqrt(x*x+y*y+z*z) for x,y,z in verts)
    scale = dists[int(len(dists)*0.95)]
    if scale > 0:
        verts = [(x/scale, y/scale, z/scale) for x,y,z in verts]
    print(f"  Normalized: 95pct radius = 1.0")

    net_indices = [assign_net_idx(x,y,z) for x,y,z in verts]
    norms = compute_normals(verts, faces)

    out = os.path.join(script_dir, "brain.obj")
    step = 1.0 / (N_NETWORKS - 1)
    with open(out, 'w') as f:
        f.write(f"# Converted brain — {len(verts)} verts, 7 Yeo networks\n")
        for (x,y,z), nidx in zip(verts, net_indices):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {nidx*step:.5f} 0.5000 0.5000\n")
        for n in norms:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for (i0,i1,i2) in faces:
            f.write(f"f {i0+1}//{i0+1} {i1+1}//{i1+1} {i2+1}//{i2+1}\n")

    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"✅ Written: {out}  ({size_mb:.1f} MB)")
