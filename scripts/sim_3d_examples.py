"""Run the 3D examples of the PICT samples and record their fields per output iteration for render_3d_examples.py.
The sample functions run unchanged (normal PICT output in ./test_runs/<time>_<case>-3d-sample); the fields are recorded through the
log function of their Simulation, which is called after every output iteration (the sample's own log function still runs).

Cases (the 3D calls of the samples' main sections):
  cylinder   cylinder_channel_sample.py, Re 250, res 32 with 32 cells over a periodic span of 4, from the developed 2D flow
             records: span-wise vorticity in the mid-span plane per cell of the 2D grid (circulation around the cells, as the 2D video),
                      the velocity of all cells, per span-wise layer of the 2D grid. The renderer interpolates it to a uniform grid and
                      computes the vortex criteria there: per-block finite differences on the O-grid show the block junctions.
  airfoil    airfoil_channel_sample.py, Re 3000, angle of attack 10, 96 cells over a periodic span of 1.4, from the developed 2D flow
             records the same as cylinder; the velocity is written per frame to <out_dir>/vel/%05d.npy (27 MB per frame)
  bfs        backward_facing_step_sample.py, Re 1400, res_scale 8 with 32 cells over a periodic span of 4, from the developed 2D flow
             records the same as cylinder plus the passive scalar, written per frame to <out_dir>/vel and <out_dir>/s
  rbc        rayleigh_benard_sample.py, Ra 8e5, 96 x 61 x 96 cells
             records: temperature of all cells (uniform in x and z, wall-refined in y), Nusselt number
  decaying   decaying_turbulence_sample.py, 128^3 cells, viscosity 3e-3
             records: z-vorticity and Q-criterion of all cells (uniform grid), kinetic energy and enstrophy

usage: python sim_3d_examples.py <case> <out_dir> <iterations> <time step> [<saved 2D domain>] (cylinder, airfoil and bfs)"""
import os, sys, time
import numpy as np
import torch
import vizlib as V
import PISOtorch, PISOtorch_simulation
from lib.util.logging import setup_run, get_logger, close_logging

CASE, OUT, ITER, TIME_STEP = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
frames = {} # name -> list of per-frame arrays
scalars = {} # name -> list of floats
extra = dict(time_step=TIME_STEP)

def velocity_gradients(domain):
    # per block: G[i, j] = du_i/dx_j, z, y, x. PISOtorch returns per block and velocity component i the gradient [1, j, z, y, x]
    return [torch.stack([g[0] for g in grads], dim=0) for grads in PISOtorch.ComputeSpatialVelocityGradients(domain)]

def q_criterion(G):
    # Q = (|rotation rate|^2 - |strain rate|^2)/2, positive in vortex cores
    S = 0.5*(G + G.transpose(0, 1)); R = 0.5*(G - G.transpose(0, 1))
    return 0.5*(torch.sum(R*R, dim=(0, 1)) - torch.sum(S*S, dim=(0, 1)))

DISK = set() # frames written to <out_dir>/<name>/%05d.npy right away instead of being kept in memory
def add(name, value):
    if name in DISK:
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        np.save(os.path.join(OUT, name, "%05d.npy" % len(frames.setdefault(name, []))), value)
        frames[name].append(None)
    else:
        frames.setdefault(name, []).append(value)

def extruded_recorder(domain_2d, res_z, depth, outflow_blocks, record_scalar=False):
    # For the 3D setups extruded from a 2D multi-block grid with a periodic span. domain_2d: the 2D setup on the same grid.
    # Records the span-wise vorticity in the mid-span plane per cell of the 2D grid (circulation around the cells, as the 2D videos)
    # and the velocity (and passive scalar) of all cells per span-wise layer.
    k_mid = res_z//2
    circulation = V.CirculationVorticity2D(domain_2d) # the 2D setup holds the velocity of the mid-span plane
    centers = [b.getCellCoordinates()[0].detach().cpu().numpy() for b in domain_2d.getBlocks()]
    # boundary points of the 2D grid for the interpolation in the renderer, as in sim_flow_examples.py
    # walls: boundaries with a static zero velocity, the others are the in- and outflow
    boundary_points, boundary_cells, boundary_wall = [], [], []
    offset = 0
    for block in domain_2d.getBlocks():
        vc = block.vertexCoordinates[0].detach().cpu().numpy()
        ny, nx = vc.shape[1] - 1, vc.shape[2] - 1
        cells = offset + np.arange(ny*nx).reshape(ny, nx)
        for side, (verts, side_cells) in enumerate([(vc[:,:,0], cells[:,0]), (vc[:,:,-1], cells[:,-1]), (vc[:,0,:], cells[0,:]), (vc[:,-1,:], cells[-1,:])]):
            bound = block.getBoundary(side)
            if isinstance(bound, PISOtorch.FixedBoundary):
                boundary_points.append(np.concatenate([verts[:,:1], 0.5*(verts[:,1:] + verts[:,:-1]), verts[:,-1:]], axis=1))
                boundary_cells.append(np.concatenate([side_cells[:1], side_cells, side_cells[-1:]]))
                is_wall = bound.isVelocityStatic and torch.count_nonzero(bound.velocity).item() == 0
                boundary_wall.append(np.full(len(side_cells) + 2, is_wall))
        offset += ny*nx
    extra.update(cx=np.concatenate([c[0].reshape(-1) for c in centers]), cy=np.concatenate([c[1].reshape(-1) for c in centers]),
        boundary_points=np.concatenate(boundary_points, axis=1).astype(np.float32), boundary_cells=np.concatenate(boundary_cells),
        boundary_wall=np.concatenate(boundary_wall), span=depth, res_z=res_z)

    def record(domain):
        for block, block_2d in zip(domain.getBlocks(), domain_2d.getBlocks()):
            block_2d.setVelocity(block.velocity[:,:2,k_mid].contiguous())
        for idx in outflow_blocks: # varying outflow velocity
            domain_2d.getBlock(idx).getBoundary("+x").setVelocity(domain.getBlock(idx).getBoundary("+x").velocity[:,:2,k_mid].contiguous())
        add("w", np.concatenate([w.reshape(-1) for w in circulation(domain_2d)]).astype(np.float32))
        # velocity of all cells: component, span-wise layer, cells of the 2D grid (in the order of cx, cy)
        add("vel", torch.cat([b.velocity[0].reshape(3, res_z, -1) for b in domain.getBlocks()], dim=-1).detach().cpu().numpy().astype(np.float16))
        if record_scalar:
            add("s", torch.cat([b.passiveScalar[0,0].reshape(res_z, -1) for b in domain.getBlocks()], dim=-1).detach().cpu().numpy().astype(np.float16))
        # span-wise velocity, grows with the 3D instability of the wake
        scalars.setdefault("w_rms", []).append(np.sqrt(sum(torch.sum(b.velocity[0,2]**2).item() for b in domain.getBlocks())/sum(b.velocity[0,2].numel() for b in domain.getBlocks())))
    return record

if CASE == "cylinder":
    import cylinder_channel_sample as ex
    LOAD_2D = sys.argv[5]
    Re, res, res_z, depth = 250, 32, 32, 4.0 # as the 3D example of the sample
    grids = ex.make_cylinder_channel_grids(res)
    domain_2d, _, _ = ex.make5BlockCylinderChannelSetup(grids, in_vel=1.0, viscosity=1/Re)
    record = extruded_recorder(domain_2d, res_z, depth, outflow_blocks=[4])

    run_name = "cylinder-channel-3d-sample"
    def run(run_dir, stop_handler):
        ex.cylinder_channel_sample(run_dir, name="Re250_r32_3D", res=res, Re=Re, res_z=res_z, load_2d_domain_path=LOAD_2D,
            iterations=ITER, time_step=TIME_STEP, dp=False, STOP_FN=stop_handler)

elif CASE == "airfoil":
    import airfoil_channel_sample as ex
    LOAD_2D = sys.argv[5]
    Re, attack_angle, in_vel, res_z, depth = 3000, 10, 0.3, 96, 1.4 # as the 3D example of the sample
    airfoil_path = os.path.join(os.path.dirname(os.path.abspath(ex.__file__)), "sample_airfoils", "naca0012_sharp.dat")
    grids = ex.make_airfoil_channel_grids(airfoil_path, attack_angle=attack_angle)
    domain_2d, _, _ = ex.make6BlockAirfoilChannelSetup(grids, in_vel=in_vel, viscosity=in_vel/Re)
    DISK.add("vel")
    record = extruded_recorder(domain_2d, res_z, depth, outflow_blocks=[4, 5])
    extra.update(airfoil=ex.read_airfoil(airfoil_path, attack_angle=attack_angle)[1][0,:,0].numpy().astype(np.float32), in_vel=in_vel) # 2, N

    run_name = "airfoil-channel-3d-sample"
    def run(run_dir, stop_handler):
        ex.airfoil_channel_sample(run_dir, name="Re3000_aoa10_3D", Re=Re, attack_angle=attack_angle, res_z=res_z, load_2d_domain_path=LOAD_2D,
            iterations=ITER, time_step=TIME_STEP, dp=False, STOP_FN=stop_handler)

elif CASE == "bfs":
    import backward_facing_step_sample as ex
    LOAD_2D = sys.argv[5]
    res_scale, Re, h, s_h, res_z, depth = 8, 1400, 1.0, 1.0, 32, 4.0 # as the 3D example of the sample
    H = h + s_h
    grids = ex.make_BFS_grids(h=h, s=s_h, l=5*h, L=32*h, buffer_length=3*h,
        res_l=4*res_scale, res_L=32*res_scale, res_y=int(4*res_scale*H), res_buffer=2*res_scale, ratio=(20,2))
    domain_2d, _, _ = ex.make5BlockBackwardFacingStepSetup(grids, h=h, s=s_h, in_vel=1.0, in_var=0.4, viscosity=2*h/Re)
    DISK.update(["vel", "s"])
    record = extruded_recorder(domain_2d, res_z, depth, outflow_blocks=[3, 4], record_scalar=True)

    run_name = "backward-facing-step-3d-sample"
    def run(run_dir, stop_handler):
        ex.backward_facing_step_sample(run_dir, name="Re1400_r8_3D", res_scale=res_scale, Re=Re, res_z=res_z, depth=depth, load_2d_domain_path=LOAD_2D,
            iterations=ITER, time_step=TIME_STEP, dp=False, STOP_FN=stop_handler)

elif CASE == "rbc":
    import rayleigh_benard_sample as ex
    res_x, Ra, Pr, aspect = 96, 8e5, 0.7, np.pi # as the 3D example of the sample
    res_y = round(2*res_x/aspect)
    grid = ex.makeRayleighBenardSetup(res_x, res_y, aspect, 1.0, 1.0, 1.0)[2] # x-y grid only, for the cell coordinates
    extra.update(cy=ex.shapes.coords_to_center_coords(grid)[0,1,:,0].numpy(), y_walls=np.array([-0.5, 0.5]), length=aspect, Ra=Ra, Pr=Pr)
    nusselt_fn = None

    def record(domain):
        global nusselt_fn
        if nusselt_fn is None:
            nusselt_fn = ex.make_nusselt_fn(domain, Ra, Pr)
        add("T", domain.getBlock(0).passiveScalar[0,0].detach().cpu().numpy().astype(np.float16)) # z, y, x
        scalars.setdefault("nusselt", []).append(nusselt_fn(domain))

    run_name = "rayleigh-benard-3d-sample"
    def run(run_dir, stop_handler):
        ex.rayleigh_benard_sample(run_dir, name="Ra8e5_r96_3D", res_x=res_x, res_z=res_x, Ra=Ra, Pr=Pr, adaptive_CFL=0.5,
            iterations=ITER, time_step=TIME_STEP, dp=False, STOP_FN=stop_handler)

elif CASE == "decaying":
    import decaying_turbulence_sample as ex
    res, viscosity = 128, 3e-3 # as the 3D example of the sample
    extra.update(domain_size=2*np.pi, viscosity=viscosity)

    def record(domain):
        G = velocity_gradients(domain)[0]
        wz = G[1,0] - G[0,1]
        add("wz", wz.detach().cpu().numpy().astype(np.float16)) # z, y, x
        add("q", q_criterion(G).detach().cpu().numpy().astype(np.float16))
        u = domain.getBlock(0).velocity[0]
        scalars.setdefault("energy", []).append(0.5*torch.mean(torch.sum(u**2, dim=0)).item())
        w = torch.stack([G[2,1] - G[1,2], G[0,2] - G[2,0], wz])
        scalars.setdefault("enstrophy", []).append(0.5*torch.mean(torch.sum(w**2, dim=0)).item())

    run_name = "decaying-turbulence-3d-sample"
    def run(run_dir, stop_handler):
        ex.decaying_turbulence_sample(run_dir, name="3D_r128_visc3e-3", dims=3, res=res, viscosity=viscosity,
            iterations=ITER, time_step=TIME_STEP, dp=False, STOP_FN=stop_handler)
else:
    raise ValueError("unknown case '%s'" % CASE)

# the samples create their Simulation themselves: record through its log function, after the sample's own one
class RecordingSimulation(PISOtorch_simulation.Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sample_log_fn = self.log_fn
        def log_fn(domain, **kwargs):
            if sample_log_fn is not None:
                sample_log_fn(domain=domain, **kwargs)
            record(domain)
        self.log_fn = log_fn
PISOtorch_simulation.Simulation = RecordingSimulation

run_dir = setup_run("./test_runs", name=run_name)
LOG = get_logger("Main")
stop_handler = PISOtorch_simulation.StopHandler(LOG)
stop_handler.register_signal()
t0 = time.time()
run(run_dir, stop_handler)
stop_handler.unregister_signal()
print("SIM TIME %.1f s for %d iterations" % (time.time() - t0, ITER), flush=True)
# the first record is after the first iteration
data = dict(extra, time_offset=TIME_STEP, **{k: np.stack(v) for k, v in frames.items() if k not in DISK}, **{k: np.array(v) for k, v in scalars.items()})
np.savez_compressed(os.path.join(OUT, "%s3d_frames.npz" % CASE), **data)
print("saved", len(next(iter(frames.values()))), "frames:", {k: (v[0].shape if k not in DISK else "on disk") for k, v in frames.items()}, flush=True)
close_logging()
