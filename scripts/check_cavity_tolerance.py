"""2D lid-driven cavity at Re=1000 (same setup as sim_cavity.py): RMS deviation of u from Ghia et al. (1982)
and the mean number of pressure CG iterations, on 64^2 and 128^2 grids with the default float32 tolerances
and on 128^2 with advection_tol = pressure_tol = 1e-7. Prints one line per run."""
import time
import numpy as np
import torch
import PISOtorch, PISOtorch_simulation
import lib.data.shapes as shapes

# count the iterations of the CG (pressure) solves; PISOtorch_diff looks up PISOtorch.SolveLinear at call time
_solve_linear = PISOtorch.SolveLinear
cg_iterations = []
def counting_solve_linear(*args, **kwargs):
    infos = _solve_linear(*args, **kwargs)
    use_BiCG = args[6] if len(args) > 6 else kwargs.get("useBiCG", False)
    if not use_BiCG:
        cg_iterations.extend(info.usedIterations for info in infos)
    return infos
PISOtorch.SolveLinear = counting_solve_linear

# Ghia et al. 1982, Re = 1000, u on the vertical center line (as in PICT's tests/validations.py)
ref_y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
ref_u = [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000]

def run(N, tol=None, Re=1000, t_end=100, dtype=torch.float32):
    cg_iterations.clear()
    grid = shapes.make_wall_refined_ortho_grid(N, N, wall_refinement=[], dtype=dtype).cuda().contiguous()
    domain = PISOtorch.Domain(2, torch.tensor([1.0 / Re], dtype=dtype), name="Cavity", device=torch.device("cuda"), dtype=dtype, passiveScalarChannels=0)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Cavity")
    block.CloseAllBoundaries()
    block.CloseBoundary("+y", torch.tensor([[1.0, 0.0]], device="cuda", dtype=dtype))
    domain.PrepareSolve()
    sim = PISOtorch_simulation.Simulation(domain=domain, time_step=1.0, substeps="ADAPTIVE", non_orthogonal=False,
                                          advection_tol=tol, pressure_tol=tol, log_interval=0, log_images=False)
    t0 = time.time()
    sim.run(t_end, log_domain=False)
    vel = block.velocity[0].detach().cpu().numpy()
    cc = block.getCellCoordinates()[0].detach().cpu().numpy()
    h = N // 2
    u = np.concatenate([[0], vel[0, :, h - 1:h + 1].mean(-1), [1]])
    y = np.concatenate([[0], cc[1, :, 0], [1]])
    dev = np.interp(ref_y, y, u)[1:-1] - np.array(ref_u)[1:-1]
    print("N=%d tol=%s: RMS deviation of u = %.4f, mean CG iterations = %.1f (%.0f s)" % (
        N, "default" if tol is None else tol, np.sqrt(np.mean(dev ** 2)), np.mean(cg_iterations), time.time() - t0), flush=True)

run(64)
run(128)
run(128, tol=1e-7)
