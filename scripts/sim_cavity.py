"""2D lid-driven cavity (lid at +y moving in +x, as in Ghia et al. 1982) at several resolutions.
Saves centerline profiles and the finest velocity field to an npz for plotting."""
import os, sys, time, json
import numpy as np
import torch
import PISOtorch, PISOtorch_simulation
import lib.data.shapes as shapes
from lib.util.logging import setup_run

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
setup_run(OUT, name="log")
cuda = torch.device("cuda")
dtype = torch.float32

def run(N, Re, t_end):
    grid = shapes.make_wall_refined_ortho_grid(N, N, wall_refinement=[], dtype=dtype).cuda().contiguous()
    domain = PISOtorch.Domain(2, torch.tensor([1.0 / Re], dtype=dtype), name="Cavity", device=cuda, dtype=dtype, passiveScalarChannels=0)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Cavity")
    block.CloseAllBoundaries()
    block.CloseBoundary("+y", torch.tensor([[1.0, 0.0]], device=cuda, dtype=dtype))
    domain.PrepareSolve()
    sim = PISOtorch_simulation.Simulation(domain=domain, time_step=1.0, substeps="ADAPTIVE",
        corrector_steps=2, non_orthogonal=False, velocity_corrector="FD",
        log_interval=0, log_images=False)
    t0 = time.time()
    sim.run(int(t_end), log_domain=False)
    wall = time.time() - t0
    vel = block.velocity[0].detach().cpu().numpy()  # [2, y, x]
    cc = block.getCellCoordinates()[0].detach().cpu().numpy()  # [2, y, x]
    h = N // 2
    u_center = vel[0, :, h - 1:h + 1].mean(-1)
    v_center = vel[1, h - 1:h + 1, :].mean(0)
    y = cc[1, :, 0]; x = cc[0, 0, :]
    return dict(u=u_center, y=y, v=v_center, x=x, vel=vel, cx=cc[0], cy=cc[1], steps=sim.total_step, wall=wall)

results = {}
for Re, t_end in [(100, 40), (1000, 100)]:
    for N in [16, 32, 64, 128]:
        r = run(N, Re, t_end)
        print("Re %d N %d: %d steps, %.1f s" % (Re, N, r["steps"], r["wall"]), flush=True)
        results["%d_%d" % (Re, N)] = r
        np.savez_compressed(os.path.join(OUT, "cavity.npz"), **{k + "__" + kk: v for k, r in results.items() for kk, v in r.items()})
print("saved")
