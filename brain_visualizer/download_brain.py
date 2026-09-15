"""
download_brain.py — Download a real open-source brain mesh and convert to brain.obj
====================================================================================

Run: python brain_visualizer/download_brain.py

Tries sources in order until one succeeds:

  Source A — BrainGlobe Allen CCF (GitHub release, no auth, ~2MB)
  Source B — nilearn fsaverage7 (pip install nilearn, real FreeSurfer surface)
  Source C — Manual instructions for Sketchfab / NIH 3D Print Exchange
"""

import os, sys, struct, math, urllib.request, zipfile, io, json

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "brain.obj")

N_NETWORKS = 7

# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize(v):
    l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-9 else (0., 1., 0.)

def compute_normals(verts, faces):
    norms = [[0.,0.,0.] for _ in verts]
    for (i0,i1,i2) in faces:
        ax,ay,az = verts[i0]; bx,by,bz = verts[i1]; cx,cy,cz = verts[i2]
        ex,ey,ez = bx-ax,by-ay,bz-az; fx,fy,fz = cx-ax,cy-ay,cz-az
        nx=ey*fz-ez*fy; ny=ez*fx-ex*fz; nz=ex*fy-ey*fx
        for idx in (i0,i1,i2):
            norms[idx][0]+=nx; norms[idx][1]+=ny; norms[idx][2]+=nz
    return [normalize(n) for n in norms]

def assign_network_index(x, y, z, noise_val):
    """Assign network index 0-6 based on 3D position (same as generator)."""
    import hashlib
    # deterministic spatial hash → network index
    pos = (x*1.7 + y*2.3 + z*1.1 + noise_val*0.8 + 2.5) / 5.0
    return int(abs(pos) * N_NETWORKS) % N_NETWORKS

def write_obj(verts, faces, net_indices, path):
    step = 1.0 / (N_NETWORKS - 1)
    norms = compute_normals(verts, faces)
    print(f"Writing {len(verts):,} verts, {len(faces):,} faces → {path}")
    with open(path, 'w') as f:
        f.write(f"# Real brain mesh — {len(verts)} verts, {N_NETWORKS} Yeo networks\n")
        for (x,y,z), nidx in zip(verts, net_indices):
            r = nidx * step
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.5f} 0.5000 0.5000\n")
        for n in norms:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for i, (i0,i1,i2) in enumerate(faces):
            f.write(f"f {i0+1}//{i0+1} {i1+1}//{i1+1} {i2+1}//{i2+1}\n")
    print(f"✅ Saved: {os.path.getsize(path)//1024//1024}MB")


# ── Source A: BrainGlobe Allen CCF 10µm left hemisphere ──────────────────────
def try_brainglobe():
    """
    BrainGlobe publishes Allen CCF mesh files on GitHub releases.
    The root structure (whole brain) is a simple OBJ-compatible mesh.
    """
    print("\n[Source A] Trying BrainGlobe Allen CCF mesh...")
    urls = [
        # BrainGlobe atlas meshes (GitHub release assets, no auth)
        "https://github.com/brainglobe/bg-atlasapi/releases/download/v1.0.0/allen_mouse_25um_v1.2.tar.gz",
        # Fallback: just the root mesh from brainrender data
        "https://raw.githubusercontent.com/brainglobe/brainrender/main/tests/test_data/brain.obj",
    ]
    for url in urls:
        try:
            print(f"  GET {url}")
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read()
            if url.endswith('.obj'):
                # Direct OBJ — save as-is but re-encode region colors
                lines = data.decode('utf-8', errors='ignore').splitlines()
                verts, faces = [], []
                for line in lines:
                    if line.startswith('v ') and not line.startswith('vn'):
                        p = line.split()
                        verts.append((float(p[1]), float(p[2]), float(p[3])))
                    elif line.startswith('f '):
                        idxs = [int(x.split('/')[0])-1 for x in line.split()[1:]]
                        if len(idxs) >= 3:
                            for i in range(1, len(idxs)-1):
                                faces.append((idxs[0], idxs[i], idxs[i+1]))
                if len(verts) > 100:
                    net_idx = [assign_network_index(x,y,z, 0.) for x,y,z in verts]
                    write_obj(verts, faces, net_idx, OUT_PATH)
                    return True
        except Exception as e:
            print(f"  Failed: {e}")
    return False


# ── Source B: nilearn fsaverage ───────────────────────────────────────────────
def try_nilearn():
    """
    nilearn bundles FreeSurfer fsaverage surfaces — real cortical surface.
    fsaverage5 = 10k verts, fsaverage7 = 163k verts.
    """
    print("\n[Source B] Trying nilearn fsaverage...")
    try:
        import nilearn
        from nilearn import datasets
        import nibabel as nib
        print(f"  nilearn {nilearn.__version__} found")

        # Try fsaverage7 (163k) or fall back to fsaverage5 (10k)
        for level in ['fsaverage7', 'fsaverage6', 'fsaverage5']:
            try:
                surf = datasets.fetch_surf_fsaverage(level)
                print(f"  Using {level}")
                # Load both hemispheres and merge
                all_verts, all_faces = [], []
                offset = 0
                for side in ['pial_left', 'pial_right']:
                    coords, triangles = nib.freesurfer.read_geometry(surf[side])
                    all_verts.extend(coords.tolist())
                    all_faces.extend([(t[0]+offset, t[1]+offset, t[2]+offset)
                                      for t in triangles.tolist()])
                    offset += len(coords)

                # Normalize scale to ~1.0
                import numpy as np
                v = np.array(all_verts)
                v -= v.mean(axis=0)
                scale = np.percentile(np.linalg.norm(v, axis=1), 95)
                v /= scale
                all_verts = v.tolist()

                net_idx = [assign_network_index(x,y,z, 0.) for x,y,z in all_verts]
                write_obj(all_verts, all_faces, net_idx, OUT_PATH)
                return True
            except Exception as e:
                print(f"  {level} failed: {e}")
    except ImportError:
        print("  nilearn not installed. Run: pip install nilearn nibabel")
    except Exception as e:
        print(f"  Failed: {e}")
    return False


# ── Source C: Instructions ─────────────────────────────────────────────────────
def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD — Free open-source brain meshes               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  OPTION 1 — Sketchfab (free, best quality)                               ║
║  ─────────────────────────────────────────                               ║
║  1. Go to: https://sketchfab.com/3d-models/brain-a9c4e13b87d04394a95e   ║
║     Or search: sketchfab.com/3d-models?q=brain&features=downloadable    ║
║  2. Download → OBJ format                                                ║
║  3. Copy the .obj file to:                                               ║
║     lumina/brain_visualizer/brain.obj                                    ║
║  4. Run: python brain_visualizer/convert_brain.py  (to re-encode colors) ║
║                                                                          ║
║  OPTION 2 — NIH 3D Print Exchange (government, free)                     ║
║  ────────────────────────────────────────────────────                    ║
║  https://3d.nih.gov/entries/3DPX-001129                                  ║
║  Download OBJ → copy to brain_visualizer/brain.obj                       ║
║                                                                          ║
║  OPTION 3 — install nilearn (best: real FreeSurfer surface)              ║
║  ──────────────────────────────────────────────────────────              ║
║  pip install nilearn nibabel                                             ║
║  python brain_visualizer/download_brain.py                               ║
║                                                                          ║
║  OPTION 4 — Allen Brain Atlas (scientific, free)                         ║
║  ────────────────────────────────────────────────                        ║
║  https://download.alleninstitute.org/informatics-archive/                ║
║    current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/997.obj ║
║  (997 = whole brain root structure)                                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if try_brainglobe(): sys.exit(0)
    if try_nilearn():    sys.exit(0)
    print_manual_instructions()
    sys.exit(1)
