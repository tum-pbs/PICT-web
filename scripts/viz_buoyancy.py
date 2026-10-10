"""Same setup as buoyancy_sample.py (rising plume driven by a passive-scalar buoyancy force),
plus a frame renderer for the scalar field."""
import os, sys, time, glob
import numpy as np
import torch
import vizlib as V
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import PISOtorch, PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.logging import setup_run

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
frames = os.path.join(OUT, "frames"); os.makedirs(frames, exist_ok=True)
setup_run(OUT, name="log")
cuda_device = torch.device("cuda")
dtype = torch.float32

# --- identical to buoyancy_sample.py ---
dims = 2
x, y = 16, 32
res_factor = 4
viscosity = 2e-4
buoyancy_factor = 0.1
time_step = 0.25
iterations = 200
x_base, y_base = x, y
x *= res_factor; y *= res_factor
grid = shapes.make_wall_refined_ortho_grid(x, y, corner_upper=(1, y / x), wall_refinement=[], base=1.10)
grid = grid.cuda().contiguous()
viscosity = torch.tensor([viscosity], dtype=dtype)
domain = PISOtorch.Domain(dims, viscosity, passiveScalarChannels=1, name="CavityDomain", device=cuda_device, dtype=dtype)
block = domain.CreateBlock(vertexCoordinates=grid, name="CavityBlock")
block.CloseAllBoundaries()
domain.PrepareSolve()
torch.manual_seed(83342666)
density_inflow = torch.rand((1, 1, y_base // 16, x_base // 4), dtype=dtype, device=cuda_device)
density_inflow = torch.repeat_interleave(density_inflow, res_factor, dim=2)
density_inflow = torch.repeat_interleave(density_inflow, res_factor, dim=3)
density_inflow = torch.nn.functional.pad(density_inflow, ((x - x // 4) // 2, (x - x // 4) // 2, 2 * y // 16, y - (3 * y // 16)))

def density_inflow_fn(domain, **kwargs):
    block = domain.getBlock(0)
    block.setPassiveScalar(torch.maximum(block.passiveScalar, density_inflow).contiguous())
    domain.UpdateDomainData()

vel_src_velX_pad = torch.zeros_like(block.passiveScalar)
def buoyancy_fn(domain, time_step, **kwargs):
    block = domain.getBlock(0)
    block.setVelocitySource(torch.cat([vel_src_velX_pad, block.passiveScalar * buoyancy_factor], dim=1))
    domain.UpdateDomainData()

def clamp_density(domain, **kwargs):
    block = domain.getBlock(0)
    block.setPassiveScalar(torch.clamp(block.passiveScalar, min=0).contiguous())
    domain.UpdateDomainData()

prep_fn = {"PRE": [density_inflow_fn], "PRE_VELOCITY_SETUP": [buoyancy_fn], "POST": clamp_density}
# --- end of sample setup ---

SMOKE = LinearSegmentedColormap.from_list("smoke", ["#07131c", "#0d3550", "#1f6d96", "#79b8d6", "#e9f4fa"])
W, H = 400, 800
snap_its = [25, 50, 100, 150, 200]
snaps = {}

def render(domain, it, **kw):
    d = V.scalar_2d(domain)
    if it in snap_its:
        snaps[it] = d[0].copy()
    fig, ax = V.new_canvas(W, H, (0, 1), (0, 2), bg="#07131c")
    ax.imshow(d[0], origin="lower", extent=(0, 1, 0, 2), cmap=SMOKE, vmin=0, vmax=1.0, interpolation="bilinear")
    fig.savefig(os.path.join(frames, "f_%05d.png" % it), dpi=100)
    plt.close(fig)

sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
    substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, pressure_tol=1e-5,
    advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
    velocity_corrector="FD", non_orthogonal=False,
    log_interval=0, log_images=False, log_fn=render)
t0 = time.time()
sim.run(iterations, log_domain=False)
print("SIM TIME %.1f s, %d steps" % (time.time() - t0, sim.total_step))
V.encode_mp4(os.path.join(frames, "f_%05d.png"), os.path.join(OUT, "buoyancy.mp4"), fps=24)

# snapshot strip
X, Y = V.block_vertices_2d(domain.getBlock(0))
fig, axs = plt.subplots(1, len(snap_its), figsize=(2.2 * len(snap_its), 4.2), dpi=150)
fig.patch.set_facecolor("#07131c")
for ax, it in zip(axs, snap_its):
    ax.imshow(snaps[it], origin="lower", extent=(0, 1, 0, 2), cmap=SMOKE, vmin=0, vmax=1, interpolation="bilinear")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("t = %g" % (it * time_step), color="#c3c2b7", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0.05)
fig.savefig(os.path.join(OUT, "buoyancy_strip.png"), facecolor=fig.get_facecolor())
print("done")
