"""Zoomed C-grid of airfoil_sample.py (full resolution), colored by block, light/dark."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib.util.logging import get_logger
import airfoil_sample as af
import PISOtorch_simulation
PICT_DIR = os.path.dirname(os.path.abspath(PISOtorch_simulation.__file__))  # the PICT checkout on PYTHONPATH
OUT = sys.argv[1]
af.LOG = get_logger("grid")
dom, _, _ = af.makeAirfoil_v2(os.path.join(PICT_DIR, "sample_airfoils", "naca0012_sharp.dat"), airfoil_res_div=1)
grids = [g.detach().cpu().numpy() for g in dom.getVertexCoordinates()]
print("block sizes", [g.shape for g in grids])
CAT = {"light": ["#2a78d6", "#eb6834", "#1baf7a"], "dark": ["#3987e5", "#d95926", "#199e70"]}
SURF = {"light": "#ffffff", "dark": "#1f222b"}
for mode in ("light", "dark"):
    for name, xl, yl, lw, size in [("airfoil_grid", (-0.25, 1.6), (-0.45, 0.45), 0.35, (8.2, 4.0)), ("airfoil_grid_full", (-2.2, 5.2), (-2.1, 2.1), 0.25, (10, 5.7))]:
        fig = plt.figure(figsize=size, dpi=150); ax = fig.add_axes([0, 0, 1, 1]); fig.patch.set_facecolor(SURF[mode])
        for i, g in enumerate(grids):
            X, Y = g[0, 0], g[0, 1]
            for j in range(X.shape[0]): ax.plot(X[j], Y[j], color=CAT[mode][i], lw=lw)
            for j in range(X.shape[1]): ax.plot(X[:, j], Y[:, j], color=CAT[mode][i], lw=lw)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
        fig.savefig(os.path.join(OUT, "%s_%s.png" % (name, mode)), facecolor=SURF[mode])
        plt.close(fig)
print("ok")
