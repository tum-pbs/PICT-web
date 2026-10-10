"""Multi-block grids of the newer forward-simulation samples, made with the grid functions and parameters of the samples themselves,
colored by block (light and dark variants, in the style of viz_grids.py). All grid lines are drawn; each figure shows the part of
the domain given in VIEW.

usage: python viz_example_grids.py <out_dir>   (PICT on the PYTHONPATH, needs a GPU for the Karman setup)"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import backward_facing_step_sample as bfs
import karman_vortex_street_sample as karman
import cylinder_channel_sample as cylinder
import airfoil_channel_sample as airfoil

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# categorical slots (fixed order) from the validated reference palette, as in viz_grids.py
CAT = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
       "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
SURF = {"light": "#ffffff", "dark": "#1f222b"}

def as_np(g):
    g = g.detach().cpu().numpy()
    return g[0, 0], g[0, 1]

# backward_facing_step_sample(res_scale=8)
grids_bfs = bfs.make_BFS_grids(h=1.0, s=1.0, l=5.0, L=32.0, buffer_length=3.0, res_l=32, res_L=256, res_y=64, res_buffer=16, ratio=(20,2))
# karman_vortex_street_sample(res_scale=16, obstacle_height=2.0)
res_scale = 16
_, _, _, grids_karman = karman.make8BlockKarmanSetup(x=16*res_scale, y=8*res_scale, x_in=int(1.5*res_scale), y_in=int(2.0*res_scale), x_pos=3*res_scale,
    in_vel=1.0, in_var=0.5, viscosity=2.0/600, scale=1/res_scale, refinement_ratio=3, obstacle_res_factor=1.5, wake_perturbation=True)
# cylinder_channel_sample(res=32)
grids_cylinder = cylinder.make_cylinder_channel_grids(32, cylinder_radius=0.5, cylinder_offset_y=0.05, ring_thickness=0.5, quad_thickness=1.0,
    channel_length=22.0, wake_refinement_base=0.95)
# airfoil_channel_sample(attack_angle=10)
airfoil_path = os.path.join(os.path.dirname(os.path.abspath(airfoil.__file__)), "sample_airfoils", "naca0012_sharp.dat")
grids_airfoil = airfoil.make_airfoil_channel_grids(airfoil_path, attack_angle=10, channel_height=1.4, inflow_x=-1.5, outflow_x=4.5, front_x=-0.5,
    res_normal=96, normal_base=0.97, tail_grow_mul=1.01)

# name: grids, view (xlim, ylim), figure width in inches
CASES = {
    "bfs_grid": (grids_bfs, ((0, 12), (0, 2)), 12),
    "karman_grid": (grids_karman, ((-2.75, 4.25), (-2.5, 2.5)), 8),
    "cylinder_grid": (grids_cylinder, ((-2, 4), (-2, 2.1)), 9),
    "airfoil_channel_grid": (grids_airfoil, ((-1.5, 2.0), (-0.7, 0.7)), 11),
}

for name, (grids, (xl, yl), width) in CASES.items():
    for mode in ("light", "dark"):
        height = width*(yl[1] - yl[0])/(xl[1] - xl[0])
        fig = plt.figure(figsize=(width, height), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_facecolor(SURF[mode]); ax.set_facecolor(SURF[mode])
        for i, g in enumerate(grids):
            X, Y = as_np(g)
            c = CAT[mode][i % 8]
            for j in range(X.shape[0]):
                ax.plot(X[j], Y[j], color=c, lw=0.35, solid_capstyle="round")
            for j in range(X.shape[1]):
                ax.plot(X[:, j], Y[:, j], color=c, lw=0.35, solid_capstyle="round")
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(os.path.join(OUT, "%s_%s.png" % (name, mode)), facecolor=SURF[mode])
        plt.close(fig)
    print(name, "%d blocks, %d cells" % (len(grids), sum((g.size(-1) - 1)*(g.size(-2) - 1) for g in grids)))
