"""Render the recorded vortex-street vorticity as a smooth MP4 + poster images."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import vizlib as V

SRC = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
frames_dir = os.path.join(OUT, "frames"); os.makedirs(frames_dir, exist_ok=True)
d = np.load(os.path.join(SRC, "vortex_frames.npz"))
cx, cy, W = d["cx"], d["cy"], d["w"]

PX_W, PX_H = 1600, 560
XL = (-1.4, 7.6)
half_h = (XL[1] - XL[0]) * PX_H / PX_W / 2
YL = (-half_h, half_h)
BG = "#f2f1ee"   # neutral midpoint of the diverging map = zero vorticity
res = V.BarycentricResampler(cx, cy, XL, YL, PX_W, PX_H, mask_fn=lambda x, y: (x * x + y * y) < 0.25)
cmap = V.PICT_DIV.copy()
cmap.set_bad(BG)

def draw(field, path, vlim=4.0):
    fig = plt.figure(figsize=(PX_W / 100, PX_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.imshow(res(field), origin="lower", extent=(*XL, *YL), cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 0.5, facecolor="#3a3a3a", edgecolor="#1e1e1e", lw=1.2, zorder=5))
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)

for i in range(W.shape[0]):
    draw(W[i], os.path.join(frames_dir, "f_%05d.png" % i))
V.encode_mp4(os.path.join(frames_dir, "f_%05d.png"), os.path.join(OUT, "vortex_street.mp4"), fps=20, crf=24)
# poster = last frame
import shutil
shutil.copy(os.path.join(frames_dir, "f_%05d.png" % (W.shape[0] - 1)), os.path.join(OUT, "vortex_street_poster.png"))
print("rendered", W.shape[0])
