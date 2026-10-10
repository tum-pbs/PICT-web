"""Record the 3D airfoil channel flow (airfoil_channel_sample.py with res_z=96) for render_airfoil3d.py. Per output iteration:
  w        span-wise vorticity in the mid-span plane, per cell of the 2D grid, from the circulation around the cells (as the 2D videos)
  vol_q    Q-criterion on a uniform grid, Q = (|rotation rate|^2 - |strain rate|^2)/2
  vol_wz   span-wise vorticity on the same grid
Two modes:
  --sample <saved 2D domain> <its simulated time>
      runs the 3D example of airfoil_channel_sample.py itself (start from the 2D flow, 120 iterations, normal PICT output in
      ./test_runs/<time>_airfoil-channel-3d-sample), recording every iteration through the log function of its Simulation
  --continue <saved 3D domain> <its simulated time> <iterations> <time step>
      continues a saved 3D domain with the solver settings of the sample
usage: python sim_airfoil3d.py <out_dir> --sample|--continue ..."""
import os, sys, time
import numpy as np
import torch
from matplotlib.path import Path
import vizlib as V
import PISOtorch, PISOtorch_simulation
import lib.util.domain_io as domain_io
from lib.util.logging import setup_run, get_logger, close_logging
from lib.util.output import _resample_block_data
import airfoil_channel_sample as ex

OUT, MODE = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
cuda_device = torch.device("cuda")
dtype = torch.float32
Re, attack_angle, in_vel, res_z, depth = 3000, 10, 0.3, 96, 1.4 # as the 3D example of the sample
VOXEL = 0.025 # cell size of the uniform grid for the 3D renders
X_MIN = -0.5 # the uniform grid starts in front of the airfoil, the inflow region is left out

airfoil_path = os.path.join(os.path.dirname(os.path.abspath(ex.__file__)), "sample_airfoils", "naca0012_sharp.dat")
grids = ex.make_airfoil_channel_grids(airfoil_path, attack_angle=attack_angle, dtype=dtype)
_, airfoil = ex.read_airfoil(airfoil_path, attack_angle=attack_angle)
airfoil = airfoil[0,:,0].numpy() # 2, N
k_mid = res_z//2 # cells at z = 0.007, the middle of the span

# mid-span plane: the 2D setup on the same grid holds the velocity of the plane for the circulation around the cells
domain_2d, _, _ = ex.make6BlockAirfoilChannelSetup(grids, in_vel=in_vel, viscosity=in_vel/Re, dtype=dtype)
circulation = V.CirculationVorticity2D(domain_2d)
def mid_span_vorticity(domain):
    for block, block_2d in zip(domain.getBlocks(), domain_2d.getBlocks()):
        block_2d.setVelocity(block.velocity[:,:2,k_mid].contiguous())
    for idx in (4, 5): # varying outflow velocity
        domain_2d.getBlock(idx).getBoundary("+x").setVelocity(domain.getBlock(idx).getBoundary("+x").velocity[:,:2,k_mid].contiguous())
    return np.concatenate([w.reshape(-1) for w in circulation(domain_2d)]).astype(np.float32)

# cell centers and boundary points of the 2D grid for the interpolation in the renderer, as in sim_flow_examples.py
centers = [b.getCellCoordinates()[0].detach().cpu().numpy() for b in domain_2d.getBlocks()]
boundary_points, boundary_cells = [], []
offset = 0
for block in domain_2d.getBlocks():
    vc = block.vertexCoordinates[0].detach().cpu().numpy()
    ny, nx = vc.shape[1] - 1, vc.shape[2] - 1
    cells = offset + np.arange(ny*nx).reshape(ny, nx)
    for side, (verts, side_cells) in enumerate([(vc[:,:,0], cells[:,0]), (vc[:,:,-1], cells[:,-1]), (vc[:,0,:], cells[0,:]), (vc[:,-1,:], cells[-1,:])]):
        if isinstance(block.getBoundary(side), PISOtorch.FixedBoundary):
            boundary_points.append(np.concatenate([verts[:,:1], 0.5*(verts[:,1:] + verts[:,:-1]), verts[:,-1:]], axis=1))
            boundary_cells.append(np.concatenate([side_cells[:1], side_cells, side_cells[-1:]]))
    offset += ny*nx

class VolumeGrid:
    # uniform grid of the 3D renders: resampled over the whole domain (cell values to voxels, holes filled), then cropped to x >= X_MIN
    def __init__(self, domain):
        coords = torch.cat([b.vertexCoordinates.reshape(3, -1) for b in domain.getBlocks()], dim=-1)
        lower, upper = torch.min(coords, dim=-1)[0].tolist(), torch.max(coords, dim=-1)[0].tolist()
        self.shape = [int(round((u - l)/VOXEL)) for l, u in zip(lower, upper)] # x, y, z
        max_cell = max(torch.max(torch.abs(b.vertexCoordinates[:,:,1:,1:,1:] - b.vertexCoordinates[:,:,:-1,:-1,:-1])).item() for b in domain.getBlocks())
        self.fill_steps = int(np.ceil(0.5*max_cell/VOXEL))
        # as in lib.data.resample.make_uniform_transform_AABB_outer: cubic voxels with the largest spacing, centered on the domain
        self.spacing = max((u - l)/n for l, u, n in zip(lower, upper, self.shape))
        origin = [0.5*(l + u) - self.spacing*(n/2 - 0.5) for l, u, n in zip(lower, upper, self.shape)] # center of the first voxel
        self.i0 = int(np.ceil((X_MIN - origin[0])/self.spacing))
        self.origin = [origin[0] + self.i0*self.spacing, origin[1], origin[2]]
        # voxels inside the airfoil (the resampling fills them from the neighbors): Q below any iso level, no vorticity
        vx = self.origin[0] + np.arange(self.shape[0] - self.i0)*self.spacing
        vy = self.origin[1] + np.arange(self.shape[1])*self.spacing
        VX, VY = np.meshgrid(vx, vy, indexing="ij")
        self.inside = Path(airfoil.T).contains_points(np.stack([VX.ravel(), VY.ravel()], axis=1)).reshape(VX.shape) # x, y

    def __call__(self, domain):
        domain.UpdateDomainData()
        q_list, wz_list = [], []
        for grads in PISOtorch.ComputeSpatialVelocityGradients(domain):
            G = torch.stack([g[0] for g in grads], dim=1) # (velocity component i, derivative direction j), z, y, x
            S = 0.5*(G + G.transpose(0, 1)); R = 0.5*(G - G.transpose(0, 1))
            q_list.append((0.5*(torch.sum(R*R, dim=(0, 1)) - torch.sum(S*S, dim=(0, 1))))[None, None].contiguous())
            wz_list.append((G[1,0] - G[0,1])[None, None].contiguous())
        vcs = [b.vertexCoordinates for b in domain.getBlocks()]
        q, wz = [_resample_block_data(data, vcs, self.shape, 3, fill_max_steps=self.fill_steps)[0,0].permute(2, 1, 0)[self.i0:].detach().cpu().numpy()
            for data in (q_list, wz_list)] # x, y, z
        q[self.inside] = -1e3; wz[self.inside] = 0
        return q.astype(np.float16), wz.astype(np.float16)

w_slice, vol_q, vol_wz = [], [], []
volume_grid = None
def record(domain, **kwargs):
    global volume_grid
    if volume_grid is None:
        volume_grid = VolumeGrid(domain)
    w_slice.append(mid_span_vorticity(domain))
    q, wz = volume_grid(domain)
    vol_q.append(q); vol_wz.append(wz)

if MODE == "--sample":
    LOAD_2D, TIME_OFFSET = sys.argv[3], float(sys.argv[4])
    TIME_STEP, ITER = 0.25, 120 # as the 3D example of the sample
    # the sample creates its Simulation itself: record through its log function (no other change to the run)
    class RecordingSimulation(PISOtorch_simulation.Simulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log_fn = record
    PISOtorch_simulation.Simulation = RecordingSimulation
    run_dir = setup_run("./test_runs", name="airfoil-channel-3d-sample")
    LOG = get_logger("Main")
    stop_handler = PISOtorch_simulation.StopHandler(LOG)
    stop_handler.register_signal()
    t0 = time.time()
    ex.airfoil_channel_sample(run_dir, name="Re3000_aoa10_3D", Re=Re, attack_angle=attack_angle, res_z=res_z, load_2d_domain_path=LOAD_2D,
        iterations=ITER, time_step=TIME_STEP, dp=False, STOP_FN=stop_handler)
    stop_handler.unregister_signal()
    print("SIM TIME %.1f s for %d iterations" % (time.time() - t0, ITER))
    TIME_OFFSET += TIME_STEP # the first record is after the first iteration
elif MODE == "--continue":
    LOAD_3D, TIME_OFFSET, ITER, TIME_STEP = sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
    setup_run(OUT, name="log")
    domain = domain_io.load_domain(LOAD_3D, dtype=dtype, device=cuda_device)
    domain.PrepareSolve()
    assert domain.getSpatialDims() == 3 and domain.getBlock(0).velocity.size(2) == res_z
    # outflow update as in make6BlockAirfoilChannelSetup, the outflow blocks are 4 and 5
    char_vel = torch.tensor([[in_vel, 0, 0]], device=cuda_device, dtype=dtype)
    def prep_fn(domain, time_step, **kwargs):
        PISOtorch_simulation.update_advective_boundaries(domain, [domain.getBlock(idx).getBoundary("+x") for idx in (4, 5)], char_vel, time_step.cuda())
    sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn={"PRE": prep_fn},
            substeps="ADAPTIVE", time_step=TIME_STEP, corrector_steps=2, advection_tol=1e-6, pressure_tol=1e-7, # as the 3D sample
            advect_non_ortho_steps=2, pressure_non_ortho_steps=4, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=True, log_interval=0, log_images=False, log_fn=record)
    sim.solver_double_fallback = True
    sim.preconditionBiCG = False
    sim.BiCG_precondition_fallback = True
    record(domain) # initial state
    t0 = time.time()
    sim.run(ITER, log_domain=False)
    print("SIM TIME %.1f s for %d iterations, %d steps" % (time.time() - t0, ITER, sim.total_step))
else:
    raise ValueError("unknown mode '%s'" % MODE)

cx = np.concatenate([c[0].reshape(-1) for c in centers]); cy = np.concatenate([c[1].reshape(-1) for c in centers])
np.savez_compressed(os.path.join(OUT, "airfoil3d_frames.npz"), time_step=TIME_STEP, time_offset=TIME_OFFSET, cx=cx, cy=cy, w=np.stack(w_slice),
    boundary_points=np.concatenate(boundary_points, axis=1).astype(np.float32), boundary_cells=np.concatenate(boundary_cells), airfoil=airfoil.astype(np.float32),
    vol_q=np.stack(vol_q), vol_wz=np.stack(vol_wz), vol_origin=np.array(volume_grid.origin), vol_spacing=np.array([volume_grid.spacing]*3),
    span=depth, in_vel=in_vel)
print("saved", len(w_slice), "frames, volume", vol_q[0].shape)
if MODE == "--sample":
    close_logging()
