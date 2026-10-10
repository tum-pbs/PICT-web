"""Multi-block grids used by the PICT samples, colored by block (light and dark variants)."""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lib.data.shapes as shapes
from lib.util.logging import get_logger
import airfoil_sample as af
import PISOtorch_simulation
PICT_DIR = os.path.dirname(os.path.abspath(PISOtorch_simulation.__file__))  # the PICT checkout on PYTHONPATH
import vortex_street_refined_sample as vsr

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
af.LOG = get_logger("grids")
vsr.LOG = af.LOG

# categorical slots (fixed order) from the validated reference palette
CAT = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
       "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
SURF = {"light": "#ffffff", "dark": "#1f222b"}
INK2 = {"light": "#52514e", "dark": "#c3c2b7"}

def as_np(g):
    g = g.detach().cpu().numpy()
    return g[0, 0], g[0, 1]

# 1) ring (O-grid) around a cylinder, 4 blocks, as in vortex_street_torus_sample.py
ring = [shapes.make_torus_2D(16, r1=0.5, r2=8, start_angle=a, angle=-90) for a in (90, 0, -90, -180)]
# 2) C-grid around NACA 0012, 3 blocks, as in airfoil_sample.py
dom, _, _ = af.makeAirfoil_v2(os.path.join(PICT_DIR, "sample_airfoils", "naca0012_sharp.dat"), airfoil_res_div=4)
cgrid = dom.getVertexCoordinates()
# 3) 8 blocks around a square obstacle, as in vortex_street_refined_sample.py
ring8 = vsr.make_refined_grid(obs_x_res=8, obs_y_res=8, obs_x_size=1, obs_y_size=1, top_bot_size=2, top_bot_grow_factor=1.2,
                              in_size=2, in_grow_factor=1.2, out_size=8, out_grow_factor=1.2)

panels = [("Ring grid around a cylinder · 4 blocks", ring, (-2.6, 2.6), (-2.6, 2.6)),
          ("C-grid around a NACA 0012 airfoil · 3 blocks", cgrid, (-0.6, 2.4), (-1.0, 1.0)),
          ("Graded grid around a square obstacle · 8 blocks", ring8, (-2.5, 8.5), (-2.5, 2.5))]

for mode in ("light", "dark"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), dpi=150, gridspec_kw={"width_ratios": [1, 1.5, 2.2]})
    fig.patch.set_facecolor(SURF[mode])
    for ax, (title, grids, xl, yl) in zip(axs, panels):
        ax.set_facecolor(SURF[mode])
        for i, g in enumerate(grids):
            X, Y = as_np(g)
            c = CAT[mode][i % 8]
            lw = 0.45
            for j in range(X.shape[0]):
                ax.plot(X[j], Y[j], color=c, lw=lw, solid_capstyle="round")
            for j in range(X.shape[1]):
                ax.plot(X[:, j], Y[:, j], color=c, lw=lw, solid_capstyle="round")
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, color=INK2[mode], fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grids_%s.png" % mode), facecolor=SURF[mode], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
print("done")
