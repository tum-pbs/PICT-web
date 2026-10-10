"""Shared helpers to render PICT multi-block fields for the website."""
import os, subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import PISOtorch

PICT_BLUE = "#0d4a6e"
PICT_RUST = "#a23b28"

# diverging map in the PICT logo colors: deep blue <-> white <-> rust
PICT_DIV = LinearSegmentedColormap.from_list(
    "pict_div", ["#041d2d", "#0d4a6e", "#3f8fb8", "#f2f1ee", "#df8f74", "#a23b28", "#4a140b"])
# sequential map for magnitudes / scalars
PICT_SEQ = LinearSegmentedColormap.from_list(
    "pict_seq", ["#05121c", "#0d4a6e", "#2f8db3", "#8fd0dc", "#f4efe6"])
PICT_SMOKE = LinearSegmentedColormap.from_list(
    "pict_smoke", ["#06131c", "#123a52", "#a23b28", "#e9a07a", "#fdf3e6"])


def block_vertices_2d(block):
    vc = block.vertexCoordinates
    return vc[0, 0].detach().cpu().numpy(), vc[0, 1].detach().cpu().numpy()


def vorticity_2d(domain):
    out = []
    for bg in PISOtorch.ComputeSpatialVelocityGradients(domain):
        du, dv = bg[0], bg[1]
        out.append((dv[0, 0] - du[0, 1]).detach().cpu().numpy())
    return out


class CirculationVorticity2D:
    """Vorticity of 2D multi-block domains from the circulation around each cell (Green-Gauss): w = (1/A) * sum over edges of u.dl.
    The velocities at the cell vertices are the mean of all cells sharing the vertex, also across connected blocks, and the boundary
    velocity on closed boundaries. In contrast to vorticity_2d (per-block finite differences) this does not show jumps where the grid
    lines of connected blocks change direction, e.g. in O-grids around obstacles."""
    def __init__(self, domain):
        coords, shapes = [], []
        for b in domain.getBlocks():
            vc = b.vertexCoordinates[0].detach().cpu().numpy() # 2, y+1, x+1
            coords.append(vc.reshape(2, -1)); shapes.append(vc.shape[1:])
        allc = np.concatenate(coords, axis=1)
        # global vertex index of each block vertex, vertices at the same position are shared
        _, vid = np.unique(np.round(allc.T/np.max(np.abs(allc)), 7), axis=0, return_inverse=True)
        vid = vid.reshape(-1)
        self.n_vert = vid.max() + 1
        self.corners, self.edges, self.areas, self.bounds = [], [], [], []
        off = 0
        for bi, (vc, (ny1, nx1)) in enumerate(zip(coords, shapes)):
            ids = vid[off:off + ny1*nx1].reshape(ny1, nx1); off += ny1*nx1
            self.corners.append(np.stack([ids[:-1,:-1], ids[:-1,1:], ids[1:,1:], ids[1:,:-1]])) # 4 corners per cell, counterclockwise
            p = vc.reshape(2, ny1, nx1)
            P = np.stack([p[:,:-1,:-1], p[:,:-1,1:], p[:,1:,1:], p[:,1:,:-1]]) # 4, 2, y, x
            self.edges.append(np.roll(P, -1, axis=0) - P)
            x, y = P[:,0], P[:,1]
            self.areas.append(0.5*np.sum(x*np.roll(y, -1, axis=0) - np.roll(x, -1, axis=0)*y, axis=0)) # signed (shoelace)
            block = domain.getBlock(bi)
            for side, line in enumerate([ids[:,0], ids[:,-1], ids[0,:], ids[-1,:]]):
                if isinstance(block.getBoundary(side), PISOtorch.FixedBoundary):
                    self.bounds.append((bi, side, line))

    def __call__(self, domain):
        vel = np.zeros((2, self.n_vert)); count = np.zeros(self.n_vert)
        for block, c in zip(domain.getBlocks(), self.corners):
            v = block.velocity[0].detach().cpu().numpy()
            for k in range(4):
                np.add.at(vel[0], c[k].ravel(), v[0].ravel()); np.add.at(vel[1], c[k].ravel(), v[1].ravel()); np.add.at(count, c[k].ravel(), 1)
        vel /= np.maximum(count, 1)
        # vertices on closed boundaries: mean of the adjacent boundary faces
        bvel = np.zeros((2, self.n_vert)); bcount = np.zeros(self.n_vert)
        for bi, side, line in self.bounds:
            v = domain.getBlock(bi).getBoundary(side).velocity.detach().cpu().numpy().reshape(2, -1)
            if v.shape[1] == 1:
                v = np.repeat(v, len(line) - 1, axis=1)
            for k in range(2):
                np.add.at(bvel[k], line[:-1], v[k]); np.add.at(bvel[k], line[1:], v[k])
            np.add.at(bcount, line[:-1], 1); np.add.at(bcount, line[1:], 1)
        on_bound = bcount > 0
        vel[:, on_bound] = bvel[:, on_bound]/bcount[on_bound]
        out = []
        for c, E, A in zip(self.corners, self.edges, self.areas):
            U = np.moveaxis(vel[:, c], 1, 0) # 4, 2, y, x
            U = 0.5*(U + np.roll(U, -1, axis=0)) # mean velocity of each edge
            out.append(np.sum(U[:,0]*E[:,0] + U[:,1]*E[:,1], axis=0)/A)
        return out


def speed_2d(domain):
    return [torch.linalg.vector_norm(b.velocity[0], dim=0).detach().cpu().numpy() for b in domain.getBlocks()]


def scalar_2d(domain, channel=0):
    return [b.passiveScalar[0, channel].detach().cpu().numpy() for b in domain.getBlocks()]


def draw_blocks(ax, domain, fields, cmap, vmin, vmax):
    for block, f in zip(domain.getBlocks(), fields):
        X, Y = block_vertices_2d(block)
        ax.pcolormesh(X, Y, f, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True, antialiased=False, linewidth=0)


def new_canvas(width_px, height_px, xlim, ylim, bg="#05121c", dpi=100):
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return fig, ax


def encode_mp4(frame_pattern, out_path, fps=24, crf=26):
    # needs an ffmpeg build with libx264; set the FFMPEG environment variable to choose the binary
    cmd = [os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", frame_pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
           "-movflags", "+faststart", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_path]
    subprocess.run(cmd, check=True)


class BarycentricResampler:
    """Linear interpolation from scattered cell centers onto a regular pixel grid.
    The Delaunay triangulation and barycentric weights are computed once and reused for every frame."""
    def __init__(self, cx, cy, xlim, ylim, width, height, mask_fn=None):
        from scipy.spatial import Delaunay
        tri = Delaunay(np.stack([cx, cy], axis=1))
        xs = np.linspace(xlim[0], xlim[1], width)
        ys = np.linspace(ylim[0], ylim[1], height)
        X, Y = np.meshgrid(xs, ys)
        q = np.stack([X.ravel(), Y.ravel()], axis=1)
        simplex = tri.find_simplex(q)
        valid = simplex >= 0
        if mask_fn is not None:
            valid &= ~mask_fn(q[:, 0], q[:, 1])
        s = simplex[valid]
        T = tri.transform[s]
        b = np.einsum("ijk,ik->ij", T[:, :2], q[valid] - T[:, 2])
        self.bary = np.concatenate([b, 1 - b.sum(axis=1, keepdims=True)], axis=1).astype(np.float32)
        self.verts = tri.simplices[s]
        self.valid = valid
        self.shape = (height, width)

    def __call__(self, values):
        out = np.full(self.valid.shape, np.nan, dtype=np.float32)
        out[self.valid] = np.sum(values[self.verts] * self.bary, axis=1)
        return out.reshape(self.shape)


# ---------------- chart styling (dataviz reference palette, validated) ----------------
CHART = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834"], ordinal=["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark":  dict(surface="#1f222b", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#30343f", axis="#454a56",
                  series=["#3987e5", "#d95926"], ordinal=["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}


def chart_axes(ax, mode):
    c = CHART[mode]
    ax.set_facecolor(c["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"]); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["muted"], labelcolor=c["ink2"], labelsize=9, length=3, width=0.8)
    ax.grid(True, color=c["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(c["ink2"]); ax.yaxis.label.set_color(c["ink2"])
    ax.title.set_color(c["ink"])


def chart_rc():
    matplotlib.rcParams.update({"font.family": "Liberation Sans", "font.size": 10, "axes.titlesize": 11,
                                "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
                                "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})
