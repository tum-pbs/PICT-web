"""Render the fields recorded by sim_flow_examples.py as MP4 videos, poster images and snapshot strips.

usage: python render_flow_examples.py <case> <data_dir> <out_dir>
Writes <case>_vorticity.mp4 (and <case>_smoke.mp4 for the passive scalar of bfs and karman, <case>_temperature.mp4 for rbc)
with posters (last frame), plus snapshot strips for the periodic turbulence cases and rbc.
The 3D channel flow (tcf) gets <case>_streaks.mp4 (stream-wise velocity fluctuations near the wall, x-z plane) and <case>_section.mp4 (the same in an x-y plane);
its 3D renders are made by render_tcf3d.py.
The videos are encoded with vizlib.encode_mp4, set FFMPEG to an ffmpeg with libx264 if the one on the PATH has none."""
import os, sys, shutil, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.path import Path
import vizlib as V

CASE = sys.argv[1]; SRC = sys.argv[2]; OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(SRC, "%s_frames.npz" % CASE))
TIME_STEP = float(d["time_step"])

LIGHT_BG = "#f2f1ee"   # neutral midpoint of the diverging map = zero vorticity
DARK_BG = "#07131c"
SMOKE = LinearSegmentedColormap.from_list("smoke", ["#07131c", "#0d3550", "#1f6d96", "#79b8d6", "#e9f4fa"]) # as in viz_buoyancy.py
SOLID = dict(facecolor="#3a3a3a", edgecolor="#1e1e1e", lw=1.2, zorder=5)
SEPARATOR = {LIGHT_BG: "#c3c2b7", DARK_BG: "#30343f"}
TWO_PI = 2*np.pi

def nice(value):
    # round to 2 significant digits
    e = 10**np.floor(np.log10(value))
    return float(np.round(value/e, 1)*e)

def save_strip(name, images, times, extent, cmap, vmin, vmax, panel_w=2.6):
    # snapshots side by side, in the style of the buoyancy strip
    fig_h = panel_w*(extent[3] - extent[2])/(extent[1] - extent[0]) + 0.35
    fig, axs = plt.subplots(1, len(images), figsize=(panel_w*len(images), fig_h), dpi=150)
    fig.patch.set_facecolor("#05121c")
    for ax, image, t in zip(axs, images, times):
        ax.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("t = %g" % t, color="#c3c2b7", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=1 - 0.354/fig_h, wspace=0.05)
    fig.savefig(os.path.join(OUT, name + "_strip.png"), facecolor=fig.get_facecolor())
    plt.close(fig)

def encode(tmp, name, n_frames, fps):
    V.encode_mp4(os.path.join(tmp, "f_%05d.png"), os.path.join(OUT, name + ".mp4"), fps=fps, crf=24)
    shutil.copy(os.path.join(tmp, "f_%05d.png" % (n_frames - 1)), os.path.join(OUT, name + "_poster.png"))
    shutil.rmtree(tmp)

if CASE == "tcf":
    # 3D channel flow: slices of the stream-wise velocity on the uniform x-z grid and the wall-refined x-y grid
    u_wall = float(d["tcf_u_wall"]); length = float(d["tcf_length"]); depth = float(d["tcf_depth"])
    x0, x1 = -length/2, length/2
    z0, z1 = -depth/2, depth/2
    y_walls = d["tcf_y_walls"]; cy = d["tcf_y"]
    FPS = 25
    uxz, uxy = d["uxz"], d["uxy"] # read once: every access to a member of the npz file decompresses all frames
    frames = np.arange(uxz.shape[0])

    # near-wall streaks: fluctuation of the stream-wise velocity around the plane average, in wall units.
    # the plane is periodic, so the cells are wrapped by one cell for the bilinear interpolation up to the domain border.
    def streaks(i):
        u = uxz[i]
        u = (u - u.mean())/u_wall
        return np.pad(u, 1, mode="wrap")
    dx, dz = length/uxz.shape[2], depth/uxz.shape[1]
    ext_xz = (x0 - dx, x1 + dx, z0 - dz, z1 + dz)
    def y_resampling(n):
        # the wall-refined y-direction is resampled to n uniform points from wall to wall, the wall values are 0 (no-slip)
        ys = np.linspace(y_walls[0], y_walls[1], n)
        y_nodes = np.concatenate([[y_walls[0]], cy, [y_walls[1]]])
        idx = np.clip(np.searchsorted(y_nodes, ys) - 1, 0, len(y_nodes) - 2)
        return idx, ((ys - y_nodes[idx])/(y_nodes[idx + 1] - y_nodes[idx])).astype(np.float32)
    # x-y plane: fluctuation of the stream-wise velocity around the mean of each row, in wall units.
    PX_W = 1600
    px_h = int(round(PX_W*(y_walls[1] - y_walls[0])/length))
    idx, wy = y_resampling(px_h)
    wy = wy[:, None]
    def section(i):
        u = uxy[i]
        u = (u - u.mean(axis=1, keepdims=True))/u_wall
        u = np.concatenate([np.zeros_like(u[:1]), u, np.zeros_like(u[:1])], axis=0)
        u = u[idx]*(1 - wy) + u[idx + 1]*wy
        return np.pad(u, ((0, 0), (1, 1)), mode="wrap")
    ext_xy = (x0 - dx, x1 + dx, y_walls[0], y_walls[1])

    def render(name, image_fn, extent, cmap, vmin, vmax, bg, crop):
        w = PX_W; h = int(round(PX_W*(crop[3] - crop[2])/(crop[1] - crop[0])))
        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); fig.patch.set_facecolor(bg)
        im = ax.imshow(image_fn(0), origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[2], crop[3])
        tmp = tempfile.mkdtemp()
        for n, i in enumerate(frames):
            im.set_data(image_fn(i))
            fig.savefig(os.path.join(tmp, "f_%05d.png" % n), dpi=100, facecolor=bg)
        plt.close(fig)
        encode(tmp, name, len(frames), FPS)
        print("rendered %s: %d frames, color range [%.3g, %.3g]" % (name, len(frames), vmin, vmax))

    last = uxz[len(frames)//2:]
    s_lim = nice(np.percentile(np.abs(last - last.mean(axis=(1, 2), keepdims=True)), 99.5)/u_wall)
    cmap = V.PICT_DIV.copy()
    render(CASE + "_streaks", streaks, ext_xz, cmap, -s_lim, s_lim, LIGHT_BG, (x0, x1, z0, z1))
    last = uxy[len(frames)//2:]
    x_lim = nice(np.percentile(np.abs(last - last.mean(axis=2, keepdims=True)), 99.5)/u_wall)
    render(CASE + "_section", section, ext_xy, cmap, -x_lim, x_lim, LIGHT_BG, (x0, x1, y_walls[0], y_walls[1]))
    snaps = np.linspace(0, len(frames) - 1, 5).round().astype(int)
    time_offset = float(d["time_offset"]) if "time_offset" in d.files else 0.0
    save_strip(CASE + "_streaks", [streaks(i) for i in snaps], time_offset + snaps*TIME_STEP, ext_xz, cmap, -s_lim, s_lim, panel_w=3.6)
    print("Re_tau: start %.1f, end %.1f" % (d["re_tau"][0], d["re_tau"][-1]))
    sys.exit(0)

cx, cy = d["cx"], d["cy"]
block_shapes = d["block_shapes"]

# geometry of the setups in sim_flow_examples.py. views are stacked from top to bottom: (xlim, ylim)
# solids are drawn on top of the field and masked in the interpolation: ("rect", x, y, w, h), ("circle", x, y, r), ("polygon", points Nx2)
PX_W = 1600
SOLIDS = []
FPS = 25
if CASE == "bfs":
    PX_W = 2400 # the 40 x 2 channel is 20:1, too thin at page width: its two halves are stacked
    VIEWS = [((0, 20), (0, 2)), ((20, 40), (0, 2))]
    SOLIDS = [("rect", 0, 0, 5, 1)]
    FRAMES = slice(0, None, 2)
elif CASE == "karman":
    VIEWS = [((-3.75, 12.25), (-4, 4))]
    SOLIDS = [("rect", -0.75, -1, 1.5, 2)]
    FRAMES = slice(100, None, 2) # vortex shedding is established after ca. 10 time units
elif CASE in ("decaying", "kolmogorov"):
    PX_W = 800
    VIEWS = [((0, TWO_PI), (0, TWO_PI))]
    FRAMES = slice(0, None) if CASE == "decaying" else slice(100, None); FPS = 20 # Kolmogorov flow: skip the transition from the random initial state
elif CASE == "cylinder":
    PX_W = 2400 # the 22 x 4.1 channel at the true aspect ratio
    VIEWS = [((-2, 20), (-2, 2.1))]
    SOLIDS = [("circle", 0, 0, 0.5)]
    FRAMES = slice(200, None, 2) # the shedding starts after ca. 20 time units
elif CASE == "airfoil":
    PX_W = 2400 # the 6 x 1.4 channel at the true aspect ratio
    VIEWS = [((-1.5, 4.5), (-0.7, 0.7))]
    SOLIDS = [("polygon", d["airfoil"].T)]
    FRAMES = slice(40, None); FPS = 15 # the vortex shedding starts after ca. 10 time units, one shedding period is ca. 11 frames
elif CASE == "rbc":
    VIEWS = [((0, np.pi), (-0.5, 0.5))]
    FRAMES = slice(0, None)
else:
    raise ValueError("unknown case '%s'" % CASE)
PERIODIC = CASE in ("decaying", "kolmogorov")
GAP = 10
VIEW_H = [int(round(PX_W*(yl[1] - yl[0])/(xl[1] - xl[0]))) for xl, yl in VIEWS]
PX_H = sum(VIEW_H) + GAP*(len(VIEWS) - 1)
DETAIL_BOX = (VIEWS[1][0][0], VIEWS[1][1][0], VIEWS[1][0][1] - VIEWS[1][0][0], VIEWS[1][1][1] - VIEWS[1][1][0]) if len(VIEWS) > 1 else None

def in_solid(x, y):
    m = np.zeros(x.shape, dtype=bool)
    for solid in SOLIDS:
        if solid[0] == "rect":
            sx, sy, sw, sh = solid[1:]
            m |= (x > sx) & (x < sx + sw) & (y > sy) & (y < sy + sh)
        elif solid[0] == "circle":
            sx, sy, r = solid[1:]
            m |= (x - sx)**2 + (y - sy)**2 < r*r
        elif solid[0] == "polygon":
            m |= Path(solid[1]).contains_points(np.stack([x, y], axis=-1))
    return m

def solid_patch(solid):
    if solid[0] == "rect":
        return Rectangle(solid[1:3], *solid[3:], **SOLID)
    if solid[0] == "circle":
        return Circle(solid[1:3], solid[3], **SOLID)
    return Polygon(solid[1], closed=True, **SOLID)

if PERIODIC:
    # single uniform periodic block: the cell data is already an image
    res = int(block_shapes[0][0])
    with_boundary = lambda values: values
    resamplers = [lambda values: values.reshape(res, res)]
else:
    # the boundary points take the value of their adjacent cell, they extend the interpolation to the walls and the in- and outflow
    bx, by = d["boundary_points"]; bcells = d["boundary_cells"]
    _, unique = np.unique(np.round(np.stack([bx, by], 1), 6), axis=0, return_index=True) # corner vertices are shared between blocks
    bx, by, bcells = bx[unique], by[unique], bcells[unique]
    px, py = np.concatenate([cx, bx]), np.concatenate([cy, by])
    source = np.concatenate([np.arange(len(cx)), bcells]) # cell of each interpolation point
    if "periodic_x" in d.files:
        # periodic in x: copies of the points shifted by the domain length close the gaps at the periodic boundaries
        L = float(d["periodic_x"])
        left, right = px < 0.1*L, px > 0.9*L
        source = np.concatenate([source, source[left], source[right]])
        px, py = np.concatenate([px, px[left] + L, px[right] - L]), np.concatenate([py, py[left], py[right]])
    with_boundary = lambda values: values[source]
    resamplers = [V.BarycentricResampler(px, py, xl, yl, PX_W, h, mask_fn=in_solid if SOLIDS else None) for (xl, yl), h in zip(VIEWS, VIEW_H)]

def make_canvas(cmap, vmin, vmax, bg):
    fig = plt.figure(figsize=(PX_W/100, PX_H/100), dpi=100)
    fig.patch.set_facecolor(SEPARATOR[bg]) # visible in the gap between stacked views
    images = []
    top = PX_H
    for (xl, yl), h in zip(VIEWS, VIEW_H):
        top -= h
        ax = fig.add_axes([0, top/PX_H, 1, h/PX_H])
        top -= GAP
        ax.axis("off"); ax.set_facecolor(bg)
        images.append(ax.imshow(np.zeros((h, PX_W)), origin="lower", extent=(*xl, *yl), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear"))
        for solid in SOLIDS:
            ax.add_patch(solid_patch(solid))
        if DETAIL_BOX is not None and xl[1] - xl[0] > DETAIL_BOX[2]:
            ax.add_patch(Rectangle(DETAIL_BOX[:2], *DETAIL_BOX[2:], fill=False, edgecolor="#898781", lw=1.0, ls="--", zorder=6))
        ax.set_xlim(*xl); ax.set_ylim(*yl)
    return fig, images

def render_video(name, frames, cmap, vmin, vmax, bg):
    fig, images = make_canvas(cmap, vmin, vmax, bg)
    tmp = tempfile.mkdtemp()
    for i, values in enumerate(frames):
        values = with_boundary(values)
        for im, resampler in zip(images, resamplers):
            im.set_data(resampler(values))
        fig.savefig(os.path.join(tmp, "f_%05d.png" % i), dpi=100, facecolor=SEPARATOR[bg])
    plt.close(fig)
    encode(tmp, name, len(frames), FPS)
    print("rendered %s: %d frames, color range [%.3g, %.3g]" % (name, len(frames), vmin, vmax))

def render_strip(name, frames, times, cmap, vmin, vmax):
    (xl, yl), resampler = VIEWS[0], resamplers[0]
    save_strip(name, [resampler(with_boundary(values)) for values in frames], times, (*xl, *yl), cmap, vmin, vmax,
        panel_w=2.6 if PERIODIC else 3.6)

idx = np.arange(d["w"].shape[0])[FRAMES]
w = d["w"][idx]
w_lim = nice(np.percentile(np.abs(w), 99.5)) # fixed for all frames, so that decay and growth stay visible
cmap = V.PICT_DIV.copy(); cmap.set_bad(LIGHT_BG)
render_video(CASE + "_vorticity", w, cmap, -w_lim, w_lim, LIGHT_BG)
if PERIODIC:
    # the strip covers the whole simulation, including the frames skipped in the video
    snaps = np.linspace(0, d["w"].shape[0] - 1, 5).round().astype(int)
    render_strip(CASE + "_vorticity", d["w"][snaps], snaps*TIME_STEP, cmap, -w_lim, w_lim)
if "s" in d.files and CASE == "rbc":
    # temperature between the cold (0, blue) and the hot wall (1, rust)
    s = d["s"][idx]
    cmap = V.PICT_DIV.copy(); cmap.set_bad(LIGHT_BG)
    render_video(CASE + "_temperature", s, cmap, 0, 1, LIGHT_BG)
    snaps = np.linspace(0, d["s"].shape[0] - 1, 5).round().astype(int)
    render_strip(CASE + "_temperature", d["s"][snaps], snaps*TIME_STEP, cmap, 0, 1)
elif "s" in d.files:
    s = d["s"][idx]
    cmap = SMOKE.copy(); cmap.set_bad(DARK_BG)
    render_video(CASE + "_smoke", s, cmap, 0, nice(np.percentile(s, 99.9)), DARK_BG)
