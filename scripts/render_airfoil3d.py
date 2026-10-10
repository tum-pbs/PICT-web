"""Render the 3D airfoil channel flow recorded by sim_airfoil3d.py:
  airfoil3d_midspan_vorticity.mp4   span-wise vorticity in the mid-span plane, like the 2D videos of render_flow_examples.py
  airfoil3d_vortices.mp4            vortex surfaces (Q-criterion) colored by span-wise vorticity, rendered with vedo
with posters (last frame) and a snapshot strip of the vortices. The vedo frames are rendered in a child process like volume_render.py;
vedo may live in another Python environment, set VEDO_PYTHON to its interpreter.
usage: python render_airfoil3d.py <data dir> <out dir> [--q Q iso value in (U/c)^2] [--test]"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np

EDGE, WALL, SOLID_COLOR = "#898781", "#c3c2b7", "#3a3a3a"


def _child(src, out_dir):
    # runs in the vedo environment: only numpy and vedo
    import vedo
    with open(os.path.join(src, "scene.json")) as file:
        scene = json.load(file)
    q = np.load(os.path.join(src, "q.npy"), mmap_mode="r"); wz = np.load(os.path.join(src, "wz.npy"), mmap_mode="r")
    plotter = vedo.Plotter(size=scene["size"], offscreen=True, bg=scene["bg"])
    x0, x1, y0, y1, z0, z1 = scene["box"]
    # airfoil extruded over the span, the caps are triangle fans (the profile is convex)
    p = np.asarray(scene["airfoil"]) # N, 2
    n = len(p)
    verts = [(x, y, z0) for x, y in p] + [(x, y, z1) for x, y in p] + [(*p.mean(axis=0), z0), (*p.mean(axis=0), z1)]
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    faces += [[2*n, (i + 1) % n, i] for i in range(n)] + [[2*n + 1, n + i, n + (i + 1) % n] for i in range(n)]
    airfoil = vedo.Mesh([verts, faces]).c(SOLID_COLOR).lighting(ambient=0.45, diffuse=0.6, specular=0.05)
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.count_nonzero(corners[a] != corners[b]) == 1]
    box = vedo.Lines([corners[a] for a, _ in edges], [corners[b] for _, b in edges]).c(EDGE).lw(1.5).lighting("off")
    wall = vedo.Mesh([[[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [[0, 1, 2, 3]]]).c(WALL).alpha(0.5).lighting("off")
    plotter.add(airfoil, box, wall)
    cam = scene["camera"]
    plotter.camera.SetPosition(cam["position"]); plotter.camera.SetFocalPoint(cam["focal_point"])
    plotter.camera.SetViewUp(cam["view_up"]); plotter.camera.SetViewAngle(cam["view_angle"])
    plotter.show(resetcam=False)
    grid = dict(spacing=scene["spacing"], origin=scene["origin"])
    actor = None
    for i in range(len(q)):
        if actor is not None:
            plotter.remove(actor)
        actor = vedo.Volume(np.asarray(q[i], dtype=np.float32), **grid).isosurface(scene["iso_value"]).compute_normals()
        actor.probe(vedo.Volume(np.asarray(wz[i], dtype=np.float32), **grid)) # the probed values replace "input_scalars"
        actor.cmap(scene["cmap"], "input_scalars", vmin=scene["vmin"], vmax=scene["vmax"])
        actor.phong().lighting(ambient=0.3, diffuse=0.75, specular=0.1)
        plotter.add(actor)
        plotter.render()
        plotter.screenshot(os.path.join(out_dir, "f_%05d.png" % i))
    plotter.close()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    _child(sys.argv[2], sys.argv[3])
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Polygon
from matplotlib.path import Path
import vizlib as V

parser = argparse.ArgumentParser()
parser.add_argument("src"); parser.add_argument("out")
parser.add_argument("--q", type=float, default=2.0, help="Q of the vortex surfaces in units of (U/c)^2")
parser.add_argument("--test", action="store_true", help="render the last frame only, as <name>_test.png")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
d = np.load(os.path.join(args.src, "airfoil3d_frames.npz"))
TIME_STEP, TIME_OFFSET = float(d["time_step"]), float(d["time_offset"])
w_cells = d["w"] # read once: every access to a member of the npz file decompresses all frames
n_frames = w_cells.shape[0]
frames = [n_frames - 1] if args.test else list(range(n_frames))
FPS = 15 # as the 2D airfoil video: one shedding period is ca. 2.7 time units
LIGHT_BG = "#f2f1ee"
airfoil = d["airfoil"].T # N, 2
SOLID = dict(facecolor=SOLID_COLOR, edgecolor="#1e1e1e", lw=1.2, zorder=5)

def nice(value):
    # round to 2 significant digits
    e = 10**np.floor(np.log10(value))
    return float(np.round(value/e, 1)*e)

def encode(tmp, name, n, fps):
    if args.test:
        shutil.copy(os.path.join(tmp, "f_%05d.png" % (n - 1)), os.path.join(args.out, name + "_test.png"))
    else:
        V.encode_mp4(os.path.join(tmp, "f_%05d.png"), os.path.join(args.out, name + ".mp4"), fps=fps, crf=24)
        shutil.copy(os.path.join(tmp, "f_%05d.png" % (n - 1)), os.path.join(args.out, name + "_poster.png"))

# --- mid-span plane, as the 2D airfoil video of render_flow_examples.py ---
cx, cy = d["cx"], d["cy"]
bx, by = d["boundary_points"]; bcells = d["boundary_cells"]
_, unique = np.unique(np.round(np.stack([bx, by], 1), 6), axis=0, return_index=True)
bx, by, bcells = bx[unique], by[unique], bcells[unique]
source = np.concatenate([np.arange(len(cx)), bcells])
xl, yl, PX_W = (-1.5, 4.5), (-0.7, 0.7), 2400
PX_H = int(round(PX_W*(yl[1] - yl[0])/(xl[1] - xl[0])))
resampler = V.BarycentricResampler(np.concatenate([cx, bx]), np.concatenate([cy, by]), xl, yl, PX_W, PX_H,
    mask_fn=lambda x, y: Path(airfoil).contains_points(np.stack([x, y], axis=-1)))
w_lim = nice(np.percentile(np.abs(w_cells), 99.5))
cmap = V.PICT_DIV.copy(); cmap.set_bad(LIGHT_BG)
fig = plt.figure(figsize=(PX_W/100, PX_H/100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); fig.patch.set_facecolor(LIGHT_BG)
im = ax.imshow(np.zeros((PX_H, PX_W)), origin="lower", extent=(*xl, *yl), cmap=cmap, vmin=-w_lim, vmax=w_lim, interpolation="bilinear")
ax.add_patch(Polygon(airfoil, closed=True, **SOLID)); ax.set_xlim(*xl); ax.set_ylim(*yl)
tmp = tempfile.mkdtemp()
for n, i in enumerate(frames):
    im.set_data(resampler(w_cells[i][source]))
    fig.savefig(os.path.join(tmp, "f_%05d.png" % n), dpi=100, facecolor=LIGHT_BG)
plt.close(fig)
encode(tmp, "airfoil3d_midspan_vorticity", len(frames), FPS); shutil.rmtree(tmp)
print("rendered airfoil3d_midspan_vorticity: %d frames, color range [%.3g, %.3g]" % (len(frames), -w_lim, w_lim))

# --- vortex surfaces with vedo ---
in_vel, span = float(d["in_vel"]), float(d["span"])
spacing, origin = d["vol_spacing"].tolist(), d["vol_origin"].tolist()
q_iso = args.q*in_vel**2 # chord length 1
def camera(focal, dist, el, az, angle=19):
    el, az = np.radians(el), np.radians(az)
    return dict(position=tuple(float(c) for c in np.array(focal) + dist*np.array([np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)])),
        focal_point=tuple(focal), view_up=(0, 1, 0), view_angle=angle)
scene = dict(size=(1600, 1000), bg=LIGHT_BG, spacing=spacing, origin=origin, iso_value=q_iso, vmin=-w_lim, vmax=w_lim,
    cmap=[to_hex(c) for c in V.PICT_DIV(np.linspace(0, 1, 256))], airfoil=airfoil.tolist(),
    box=(origin[0] - 0.5*spacing[0], 4.5, -0.7, 0.7, -0.5*span, 0.5*span), camera=camera((1.6, -0.2, 0.0), 9.0, 32, -38))
tmp = tempfile.mkdtemp()
src = os.path.join(tmp, "src"); out = os.path.join(tmp, "frames"); os.makedirs(src); os.makedirs(out)
# light smoothing along the span: the span-wise cells (0.0146) do not fit the voxels (0.025), which leaves a regular ripple on the surfaces
from scipy.ndimage import gaussian_filter1d
for key, name in (("vol_q", "q.npy"), ("vol_wz", "wz.npy")):
    v = gaussian_filter1d(d[key][frames].astype(np.float32), sigma=1.0, axis=3, mode="wrap") # frames, x, y, z; periodic span
    np.save(os.path.join(src, name), np.ascontiguousarray(v.astype(np.float16)))
with open(os.path.join(src, "scene.json"), "w") as file:
    json.dump(scene, file)
env = dict(os.environ); env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
proc = subprocess.run([os.environ.get("VEDO_PYTHON", sys.executable), os.path.abspath(__file__), "--child", src, out], capture_output=True, env=env,
    timeout=120 + 20*len(frames))
if proc.returncode != 0:
    raise RuntimeError("vedo child exited %d: %s" % (proc.returncode, proc.stderr.decode(errors="replace")[-2000:]))
images = [os.path.join(out, "f_%05d.png" % n) for n in range(len(frames))]
blank = [p for p in images if not os.path.exists(p) or plt.imread(p)[..., :3].std() < 1e-3] # a broken GL stack returns blank images
if blank:
    raise RuntimeError("vedo: %d of %d images are missing or blank" % (len(blank), len(images)))
encode(out, "airfoil3d_vortices", len(frames), FPS)
if not args.test:
    snaps = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    fig, axs = plt.subplots(1, len(snaps), figsize=(4.2*len(snaps), 4.2*1000/1600 + 0.35), dpi=150)
    fig.patch.set_facecolor("#05121c")
    for a, s in zip(axs, snaps):
        a.imshow(plt.imread(images[s])); a.axis("off"); a.set_title("t = %g" % (TIME_OFFSET + frames[s]*TIME_STEP), color="#c3c2b7", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=1 - 0.354/(4.2*1000/1600 + 0.35), wspace=0.05)
    fig.savefig(os.path.join(args.out, "airfoil3d_vortices_strip.png"), facecolor=fig.get_facecolor()); plt.close(fig)
shutil.rmtree(tmp)
print("rendered airfoil3d_vortices: %d frames, Q = %.3g (%.2g (U/c)^2), color range [%.3g, %.3g]" % (len(frames), q_iso, args.q, -w_lim, w_lim))
