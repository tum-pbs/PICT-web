"""3D renders of the channel flow recorded by sim_flow_examples.py (case tcf) with volume_render.py, replacing the first 3D renders of
render_flow_examples.py. Full channel and lower half (wall to centreline):
  tcf_vortices[_half]: Q-criterion surfaces at Q+ = 0.001, colored by u'+
  tcf_volume[_half]:   u' scaled by its rms at each height, transparent below 1 rms, nearly opaque from 1.15 rms
vedo may live in another Python environment, set VEDO_PYTHON to its interpreter (see volume_render.py).
usage: python render_tcf3d.py <data npz> <out dir> [--q Q+] [--lo rms] [--hi rms] [--only name ...] [--suffix name] [--test]"""
import argparse, os, sys, shutil, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
import volume_render as VR

parser = argparse.ArgumentParser()
parser.add_argument("src"); parser.add_argument("out")
parser.add_argument("--q", type=float, default=0.001, help="Q+ of the vortex surfaces")
parser.add_argument("--lo", type=float, default=1.0, help="volume: transparent below this many rms")
parser.add_argument("--hi", type=float, default=1.15, help="volume: nearly opaque from this many rms; a wider band speckles with the CPU mapper")
parser.add_argument("--only", nargs="+", default=None, help="render only these, e.g. tcf_volume")
parser.add_argument("--suffix", default="", help="appended to the output names")
parser.add_argument("--test", action="store_true", help="render frame 200 only, as <name>_test.png")
args = parser.parse_args()
SRC, OUT, TEST = args.src, args.out, args.test
d = np.load(SRC)
TIME_STEP, time_offset = float(d["time_step"]), float(d["time_offset"])
cy, y_walls = d["tcf_y"], d["tcf_y_walls"]; length, depth = float(d["tcf_length"]), float(d["tcf_depth"])
u_cells, q_cells = d["vol_u"], d["vol_q"] # read once: every access to a member of the npz file decompresses all frames
frames = [200] if TEST else list(range(u_cells.shape[0]))
FPS, Q_ISO, SCALED_LO, SCALED_HI, SCALED_LIM = 25, args.q, args.lo, args.hi, 3.5
PICT_DIV = LinearSegmentedColormap.from_list("pict_div", ["#041d2d", "#0d4a6e", "#3f8fb8", "#f2f1ee", "#df8f74", "#a23b28", "#4a140b"]) # vizlib.PICT_DIV

# grid as in render_flow_examples.py, with 129 points in y so that the centreline is a grid plane
ny = 129
ys = np.linspace(y_walls[0], y_walls[1], ny); y_nodes = np.concatenate([[y_walls[0]], cy, [y_walls[1]]])
idx = np.clip(np.searchsorted(y_nodes, ys) - 1, 0, len(y_nodes) - 2)
wy = ((ys - y_nodes[idx])/(y_nodes[idx + 1] - y_nodes[idx])).astype(np.float32)[None, :, None]
def to_grid(v):
    # z, y, x cells -> x, y, z grid points: periodic directions on the vertices, y uniform with 0 on the walls
    v = np.pad(v.astype(np.float32), ((0, 0), (1, 1), (0, 0)))
    v = v[:, idx]*(1 - wy) + v[:, idx + 1]*wy
    for axis in (0, 2):
        v = 0.5*(v + np.roll(v, 1, axis=axis)); v = np.concatenate([v, v.take([0], axis=axis)], axis=axis)
    return v.transpose(2, 1, 0).astype(np.float16)
# rms of u' at each height over the run, symmetric in the two halves
urms = np.sqrt((u_cells[::5].astype(np.float32)**2).mean(axis=(0, 1, 3)))
urms = 0.5*(urms + urms[::-1])
vol_u = np.stack([to_grid(u_cells[i]) for i in frames])
vol_s = np.stack([to_grid(u_cells[i].astype(np.float32)/urms[None, :, None]) for i in frames])
vol_q = np.stack([to_grid(q_cells[i]) for i in frames])
u_lim = 6.3 # color range of u'+, as in the earlier renders
j_mid = ny//2 + 1 # grid points of the lower half, up to the centreline

x0, z0 = -length/2, -depth/2
def camera(focal, dist, el, az=-28, angle=19):
    el, az = np.radians(el), np.radians(az)
    return dict(position=tuple(float(c) for c in np.array(focal) + dist*np.array([np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)])),
        focal_point=focal, view_up=(0, 1, 0), view_angle=angle)
base = dict(bg="#f2f1ee", spacing=(length/(vol_u.shape[1] - 1), (y_walls[1] - y_walls[0])/(ny - 1), depth/(vol_u.shape[3] - 1)),
    origin=(x0, float(y_walls[0]), z0), cmap=[to_hex(c) for c in PICT_DIV(np.linspace(0, 1, 256))])
# the wall plane half a grid cell below the volume: where they coincide, the CPU volume mapper ends the rays at random pixels (dots)
y_floor = float(y_walls[0]) - 0.5*base["spacing"][1]
full = dict(base, size=(1600, 1200), box=(x0, -x0, y_floor, float(y_walls[1]), z0, -z0), camera=camera((0, 0, 0), 10.5, 22))
half = dict(base, size=(1600, 1000), box=(x0, -x0, y_floor, float(ys[j_mid - 1]), z0, -z0), camera=camera((0, -0.5, 0), 8.6, 26))
ramp = [(-SCALED_LIM, 1.0), (-SCALED_HI, 0.85), (-SCALED_LO, 0.0), (SCALED_LO, 0.0), (SCALED_HI, 0.85), (SCALED_LIM, 1.0)]
volume = dict(mode="volume", vmin=-SCALED_LIM, vmax=SCALED_LIM, alpha=ramp, alpha_unit=0.03)
vortices = dict(mode="isosurface", vmin=-u_lim, vmax=u_lim, iso_value=Q_ISO)
renders = [("tcf_vortices", dict(full, **vortices), dict(iso=vol_q, color=vol_u)),
    ("tcf_vortices_half", dict(half, **vortices), dict(iso=vol_q[:, :, :j_mid], color=vol_u[:, :, :j_mid])),
    ("tcf_volume", dict(full, **volume), dict(volume=vol_s)),
    ("tcf_volume_half", dict(half, **volume), dict(volume=vol_s[:, :, :j_mid]))]

def save_image_strip(path, images, times, panel_w=3.6):
    # rendered images side by side, in the style of save_strip in render_flow_examples.py
    fig_h = panel_w*images[0].shape[0]/images[0].shape[1] + 0.35
    fig, axs = plt.subplots(1, len(images), figsize=(panel_w*len(images), fig_h), dpi=150)
    fig.patch.set_facecolor("#05121c")
    for ax, image, t in zip(axs, images, times):
        ax.imshow(image); ax.axis("off"); ax.set_title("t = %g" % t, color="#c3c2b7", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=1 - 0.354/fig_h, wspace=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)

os.makedirs(OUT, exist_ok=True)
if args.only:
    renders = [r for r in renders if r[0] in args.only]
for name, scene, fields in renders:
    name += args.suffix
    tmp = tempfile.mkdtemp()
    paths = VR.render_frames(tmp, scene, fields)
    if TEST:
        shutil.copy(paths[0], os.path.join(OUT, name + "_test.png")); shutil.rmtree(tmp); print(name, "test frame"); continue
    snaps = np.linspace(0, len(paths) - 1, 5).round().astype(int)
    save_image_strip(os.path.join(OUT, name + "_strip.png"), [plt.imread(paths[i]) for i in snaps], time_offset + snaps*TIME_STEP)
    # as vizlib.encode_mp4 with crf 24
    subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "f_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", os.path.join(OUT, name + ".mp4")], check=True)
    shutil.copy(paths[-1], os.path.join(OUT, name + "_poster.png")); shutil.rmtree(tmp)
    print("rendered %s: %d frames" % (name, len(paths)), flush=True)
