"""Cavity centerline profiles vs. Ghia et al. (1982) and a streamline plot (light + dark variants)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import vizlib as V

SRC, OUT = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(SRC, "cavity.npz"))
RES = [16, 32, 64]
# validated 3-step ordinal ramps (light: light->dark, dark: dark->light)
RAMP = {"light": ["#86b6ef", "#2a78d6", "#0d366b"], "dark": ["#1c5cab", "#5598e7", "#b7d3f6"]}

# Ghia et al. 1982, as included in PICT's tests/validations.py
ref_y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
ref_u = {100: [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000],
         1000: [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000]}
ref_x = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
ref_v = {100: [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000],
         1000: [0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]}

def profiles(Re, N):
    k = "%d_%d__" % (Re, N)
    y = np.concatenate([[0], d[k + "y"], [1]]); u = np.concatenate([[0], d[k + "u"], [1]])
    x = np.concatenate([[0], d[k + "x"], [1]]); v = np.concatenate([[0], d[k + "v"], [0]])
    return y, u, x, v

# error table (interior reference points)
table = {}
for Re in (100, 1000):
    for N in RES:
        y, u, x, v = profiles(Re, N)
        eu = np.interp(ref_y, y, u)[1:-1] - np.array(ref_u[Re])[1:-1]
        ev = np.interp(ref_x, x, v)[1:-1] - np.array(ref_v[Re])[1:-1]
        table["%d_%d" % (Re, N)] = dict(rmse_u=float(np.sqrt(np.mean(eu ** 2))), rmse_v=float(np.sqrt(np.mean(ev ** 2))),
                                       max_u=float(np.abs(eu).max()), max_v=float(np.abs(ev).max()))
json.dump(table, open(os.path.join(OUT, "cavity_errors.json"), "w"), indent=1)

V.chart_rc()
for mode in ("light", "dark"):
    c = V.CHART[mode]
    matplotlib.rcParams["axes.titlecolor"] = c["ink"]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.2), dpi=170)
    fig.patch.set_facecolor(c["surface"])
    for row, Re in enumerate((100, 1000)):
        axu, axv = axs[row]
        for ax in (axu, axv):
            V.chart_axes(ax, mode)
        for i, N in enumerate(RES):
            y, u, x, v = profiles(Re, N)
            axu.plot(u, y, color=RAMP[mode][i], lw=2)
            axv.plot(x, v, color=RAMP[mode][i], lw=2)
        axu.plot(ref_u[Re], ref_y, "o", ms=6.5, color=c["ink"], mec=c["surface"], mew=1.6, zorder=5)
        axv.plot(ref_x, ref_v[Re], "o", ms=6.5, color=c["ink"], mec=c["surface"], mew=1.6, zorder=5)
        axu.set_title("Re = %d · u on the vertical center line" % Re)
        axv.set_title("Re = %d · v on the horizontal center line" % Re)
        axu.set_xlabel("u"); axu.set_ylabel("y")
        axv.set_xlabel("x"); axv.set_ylabel("v")
        axu.set_ylim(0, 1); axv.set_xlim(0, 1)
    handles = [Line2D([], [], color=RAMP[mode][i], lw=2, label="PICT %d²" % N) for i, N in enumerate(RES)]
    handles.append(Line2D([], [], ls="none", marker="o", ms=6.5, color=c["ink"], mec=c["surface"], mew=1.6, label="Ghia et al. (1982)"))
    leg = fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    for t in leg.get_texts():
        t.set_color(c["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.2, w_pad=2.5)
    fig.savefig(os.path.join(OUT, "cavity_ghia_%s.png" % mode), facecolor=c["surface"])
    plt.close(fig)

# streamlines of the Re=1000 run at 64^2 (uniform grid -> cell centers form a regular lattice)
k = "1000_64__"
vel = d[k + "vel"]; cx = d[k + "cx"][0]; cy = d[k + "cy"][:, 0]
speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
for mode in ("light", "dark"):
    c = V.CHART[mode]
    if mode == "light":
        cmap = LinearSegmentedColormap.from_list("speed_l", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
        line = "#0b0b0b"
    else:
        cmap = LinearSegmentedColormap.from_list("speed_d", [c["surface"], "#123b66", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])
        line = "#ffffff"
    fig = plt.figure(figsize=(5.2, 5.2), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); fig.patch.set_facecolor(c["surface"])
    ax.imshow(speed, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    # stream function psi (u = dpsi/dy), integrated upwards from the bottom wall (psi = 0)
    dy = 1.0 / vel.shape[1]
    psi = np.cumsum(vel[0], axis=0) * dy - 0.5 * vel[0] * dy
    levels_primary = [-0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-3, -1e-4]
    levels_corner = [1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3]
    ax.contour(cx, cy, psi, levels=levels_primary, colors=line, linewidths=0.8, negative_linestyles="solid")
    ax.contour(cx, cy, psi, levels=levels_corner, colors=line, linewidths=0.6)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=c["axis"], lw=1.2)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    fig.savefig(os.path.join(OUT, "cavity_streamlines_%s.png" % mode), facecolor=c["surface"])
    plt.close(fig)
print(json.dumps(table, indent=1))
