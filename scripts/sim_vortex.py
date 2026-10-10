"""Vortex street behind a cylinder on the 4-block ring grid of vortex_street_torus_sample.py (Re=200).
Records the vorticity of all blocks per output iteration for offline rendering."""
import os, sys, time
import numpy as np
import torch
import vizlib as V
import PISOtorch, PISOtorch_simulation
from lib.util.logging import setup_run, get_logger
import vortex_street_torus_sample as vs

OUT = sys.argv[1]; ITER = int(sys.argv[2]); FRAME_START = int(sys.argv[3])
RES = int(sys.argv[4]) if len(sys.argv) > 4 else 48
os.makedirs(OUT, exist_ok=True)
setup_run(OUT, name="log")
vs.LOG = get_logger("viz")

Re = 200.0
domain, prep_fn, _ = vs.make4BlockTorus_VortexStreet(RES, r1=0.5, r2=8, rot_vel=0, vel_in=1, viscosity=1.0 / Re, dtype=torch.float32)
for block in domain.getBlocks():   # small asymmetric disturbance to trigger shedding early
    cc = block.getCellCoordinates()
    x, y = cc[:, 0:1], cc[:, 1:2]
    block.velocity[:, 1:2] += 0.4 * torch.exp(-((x - 1.2) ** 2 + (y - 0.3) ** 2) / 0.3)

centers = [b.getCellCoordinates()[0].detach().cpu().numpy() for b in domain.getBlocks()]
frames = []

def record(domain, it, **kw):
    if it >= FRAME_START:
        frames.append(np.concatenate([w.reshape(-1) for w in V.vorticity_2d(domain)]).astype(np.float32))

sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
        substeps="ADAPTIVE", time_step=0.25, corrector_steps=2, advection_tol=1e-6, pressure_tol=1e-6,
        advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
        velocity_corrector="FD", non_orthogonal=False,
        log_interval=0, log_images=False, log_fn=record)
sim.preconditionBiCG = False
sim.BiCG_precondition_fallback = True
sim.make_divergence_free()
t0 = time.time()
sim.run(ITER, log_domain=False)
print("SIM TIME %.1f s for %d iterations, %d steps" % (time.time() - t0, ITER, sim.total_step))
cx = np.concatenate([c[0].reshape(-1) for c in centers]); cy = np.concatenate([c[1].reshape(-1) for c in centers])
np.savez_compressed(os.path.join(OUT, "vortex_frames.npz"), cx=cx, cy=cy, w=np.stack(frames))
print("saved", len(frames))
