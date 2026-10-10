"""Run one of the flow samples of PICT and record the vorticity (and passive scalar) of all blocks per output iteration
for offline rendering with render_flow_examples.py.

Cases (setups and solver settings as in the sample scripts):
  bfs          backward_facing_step_sample.py, Re = 1400, res_scale 8
  karman       karman_vortex_street_sample.py, Re = 600, res_scale 16
  decaying     decaying_turbulence_sample.py, 256 x 256
  kolmogorov   kolmogorov_flow_sample.py, 256 x 256
  cylinder     cylinder_channel_sample.py, Re = 100, res 32
  airfoil      airfoil_channel_sample.py, Re = 3000, angle of attack 10
  rbc          rayleigh_benard_sample.py, Ra = 8e5, 96 x 61
  tcf          turbulent_channel_flow_sample.py, Re_tau = 180, small channel 64^3 (3D: records slices of the stream-wise velocity,
               and the whole channel's stream-wise velocity fluctuation and Q-criterion for the volume renders)
               becomes turbulent after ca. 8 ETT (190 time units), so continue from the final domain of the sample run

usage: python sim_flow_examples.py <case> <out_dir> <iterations> [<saved domain to continue from> <its simulated time>] (tcf only)"""
import os, sys, time
import numpy as np
import torch
import vizlib as V
import PISOtorch, PISOtorch_simulation
from lib.util.logging import setup_run

CASE = sys.argv[1]; OUT = sys.argv[2]; ITER = int(sys.argv[3])
LOAD_PATH = sys.argv[4] if len(sys.argv) > 4 else None # tcf only: continue from a saved domain, e.g. the final state of the sample
TIME_OFFSET = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0 # simulated time of the loaded domain, for the labels of the snapshots
os.makedirs(OUT, exist_ok=True)
setup_run(OUT, name="log")
dtype = torch.float32
TIME_STEP = 0.1 # output time step of the bfs, karman, decaying, kolmogorov and cylinder samples
record_scalar = CASE in ("bfs", "karman", "rbc")
prep_fn = None
prep_fn_after_projection = None # outflow updates are set after the initial projection, as in the samples
# solver settings of the samples, the first four use the same settings
sim_args = dict(pressure_tol=1e-6, advect_non_ortho_steps=1, pressure_non_ortho_steps=1, non_orthogonal=False)
sim_attrs = {} # attributes of the Simulation set after construction
extra = {} # additional data to save

if CASE == "bfs":
    import backward_facing_step_sample as ex
    res_scale, Re, h, s = 8, 1400, 1.0, 1.0
    H = h + s
    grids = ex.make_BFS_grids(h=h, s=s, l=5*h, L=32*h, buffer_length=3*h,
        res_l=4*res_scale, res_L=32*res_scale, res_y=int(4*res_scale*H), res_buffer=2*res_scale, ratio=(20,2), dtype=dtype)
    domain, prep_fn_after_projection, _ = ex.make5BlockBackwardFacingStepSetup(grids, h=h, s=s, in_vel=1.0, in_var=0.4, viscosity=2*h/Re, dtype=dtype)
    sim_args.update(adaptive_CFL=0.8)
elif CASE == "karman":
    import karman_vortex_street_sample as ex
    res_scale, Re, obstacle_height = 16, 600, 2.0
    domain, prep_fn_after_projection, _, _ = ex.make8BlockKarmanSetup(x=16*res_scale, y=8*res_scale, x_in=int(1.5*res_scale),
        y_in=int(obstacle_height*res_scale), x_pos=3*res_scale, in_vel=1.0, in_var=0.5, viscosity=obstacle_height/Re, scale=1/res_scale, dtype=dtype)
    sim_args.update(adaptive_CFL=0.8)
elif CASE in ("decaying", "kolmogorov"):
    if CASE == "decaying":
        import decaying_turbulence_sample as ex
    else:
        import kolmogorov_flow_sample as ex
    res = 256
    max_velocity = 4.2 if CASE == "decaying" else 7.0
    velocity = ex.make_filtered_velocity_field(res, 2*np.pi, max_velocity, 4, seed=0, dtype=dtype)
    domain = ex.makePeriodicTurbulenceDomain(res, 2*np.pi, 1e-3, velocity, dtype=dtype)
    sim_args.update(adaptive_CFL=0.5)
    if CASE == "kolmogorov":
        prep_fn = {"PRE": ex.make_kolmogorov_forcing_fn(domain, 4.0, 4, -0.1)}
elif CASE == "cylinder":
    import cylinder_channel_sample as ex
    res, Re = 32, 100
    grids = ex.make_cylinder_channel_grids(res, dtype=dtype)
    domain, prep_fn_after_projection, _ = ex.make5BlockCylinderChannelSetup(grids, in_vel=1.0, viscosity=1.0/Re, dtype=dtype)
    sim_args.update(adaptive_CFL=0.8, pressure_tol=1e-5, non_orthogonal=True)
    sim_attrs.update(preconditionBiCG=False, BiCG_precondition_fallback=True, solver_double_fallback=True)
elif CASE == "airfoil":
    import airfoil_channel_sample as ex
    TIME_STEP = 0.25
    Re, attack_angle, in_vel = 3000, 10, 0.3
    airfoil_path = os.path.join(os.path.dirname(os.path.abspath(ex.__file__)), "sample_airfoils", "naca0012_sharp.dat")
    grids = ex.make_airfoil_channel_grids(airfoil_path, attack_angle=attack_angle, dtype=dtype)
    domain, prep_fn_after_projection, _ = ex.make6BlockAirfoilChannelSetup(grids, in_vel=in_vel, viscosity=in_vel/Re, dtype=dtype)
    sim_args.update(adaptive_CFL=0.8, advection_tol=1e-6, pressure_tol=1e-6, advect_non_ortho_steps=2, pressure_non_ortho_steps=4, non_orthogonal=True)
    sim_attrs.update(preconditionBiCG=False, BiCG_precondition_fallback=True, solver_double_fallback=True)
    _, airfoil_coords = ex.read_airfoil(airfoil_path, attack_angle=attack_angle)
    extra["airfoil"] = airfoil_coords[0,:,0].numpy().astype(np.float32) # 2, N
elif CASE == "rbc":
    import rayleigh_benard_sample as ex
    TIME_STEP = 0.25 # finer than the sample for a smooth video: the plumes oscillate with a period of ca. 5 time units
    res_x, Ra, Pr, aspect = 96, 8e5, 0.7, np.pi
    res_y = round(2*res_x/aspect)
    domain, prep_fn, _ = ex.makeRayleighBenardSetup(res_x, res_y, aspect, 1.0, np.sqrt(Pr/Ra), 1/np.sqrt(Ra*Pr), seed=0, dtype=dtype)
    sim_args.update(adaptive_CFL=0.5, pressure_tol=1e-5)
    extra["periodic_x"] = aspect # the renderer wraps the cells around
    nusselt_fn = ex.make_nusselt_fn(domain, Ra, Pr)
    extra["nusselt"] = []
elif CASE == "tcf":
    import turbulent_channel_flow_sample as ex
    import lib.data.TCF_tools as TCF_tools
    import lib.util.domain_io as domain_io
    TIME_STEP = 0.25 # finer than the sample for a smooth video: the near-wall streaks move by ca. 0.4 per time unit (10 u_tau)
    res, Re_wall, length, depth = 64, 180, np.pi, np.pi/2
    if LOAD_PATH:
        domain = domain_io.load_domain(LOAD_PATH, dtype=dtype, device=torch.device("cuda"))
        domain.PrepareSolve()
        grid = domain.getBlock(0).vertexCoordinates
    else:
        domain, grid = ex.makeChannelFlowSetup(res, res, res, length, depth, Re_wall, dtype=dtype)
    forcing_fn, get_wall_shear_stress = ex.make_dynamic_forcing_fn(domain)
    prep_fn = {"PRE": forcing_fn}
    sim_args.update(adaptive_CFL=0.1, advection_use_BiCG=True, advection_tol=1e-6, pressure_tol=1e-6, non_orthogonal=True)
    sim_attrs.update(preconditionBiCG=False, BiCG_precondition_fallback=True)
    Re_center = TCF_tools.Re_wall_to_cl(Re_wall)
    viscosity = 1/Re_center
    u_wall = Re_wall/Re_center
    cell_coords = domain.getBlock(0).getCellCoordinates()[0].detach().cpu().numpy() # 3, z, y, x
    cy_line = cell_coords[1, 0, :, 0]
    y_plus = (cy_line + 1)*u_wall/viscosity
    j_wall = int(np.argmin(np.abs(y_plus - 15))) # x-z plane in the buffer layer of the lower wall, where the streaks are strongest
    k_mid = cell_coords.shape[1]//2 # x-y plane in the middle of the span
    extra.update(time_offset=TIME_OFFSET, tcf_x=cell_coords[0, 0, 0, :], tcf_y=cy_line, tcf_z=cell_coords[2, :, 0, 0], tcf_y_plus=y_plus[j_wall], tcf_u_wall=u_wall,
        tcf_y_walls=np.array([grid[0,1,0,0,0].item(), grid[0,1,0,-1,0].item()]), tcf_length=length, tcf_depth=depth)
    uxz, uxy, re_tau = [], [], []
    vol_u, vol_q = [], [] # 3D fields in wall units, float16 to keep the 401 frames at ca. 200 MB each
else:
    raise ValueError("unknown case '%s'" % CASE)

centers = [b.getCellCoordinates()[0].detach().cpu().numpy() for b in domain.getBlocks()]
vorticity, scalar = [], []
# curved multi-block grids: vorticity from the circulation around the cells, the per-block finite differences show the block connections
vorticity_fn = V.CirculationVorticity2D(domain) if CASE in ("cylinder", "airfoil") else V.vorticity_2d

# Points on the domain boundary (walls, in- and outflow) that take the value of the adjacent cell when rendering: face centers and end vertices.
# Without them the interpolation from the cell centers does not reach the boundary and connects cells across the obstacles.
boundary_points, boundary_cells = [], []
if CASE != "tcf":
    offset = 0
    for block in domain.getBlocks():
        vc = block.vertexCoordinates[0].detach().cpu().numpy() # 2, y+1, x+1
        ny, nx = vc.shape[1] - 1, vc.shape[2] - 1
        cells = offset + np.arange(ny*nx).reshape(ny, nx)
        for side, (verts, side_cells) in enumerate([(vc[:,:,0], cells[:,0]), (vc[:,:,-1], cells[:,-1]), (vc[:,0,:], cells[0,:]), (vc[:,-1,:], cells[-1,:])]):
            if isinstance(block.getBoundary(side), PISOtorch.FixedBoundary):
                boundary_points.append(np.concatenate([verts[:,:1], 0.5*(verts[:,1:] + verts[:,:-1]), verts[:,-1:]], axis=1))
                boundary_cells.append(np.concatenate([side_cells[:1], side_cells, side_cells[-1:]]))
        offset += ny*nx

def record(domain, **kwargs):
    if CASE == "tcf":
        u = domain.getBlock(0).velocity[0,0] # z, y, x
        uxz.append(u[:, j_wall, :].detach().cpu().numpy().astype(np.float32))
        uxy.append(u[k_mid].detach().cpu().numpy().astype(np.float32))
        re_tau.append(np.sqrt(get_wall_shear_stress(domain).item())/viscosity)
        # stream-wise velocity fluctuation around the mean profile of the x-z planes, and the Q-criterion
        # Q = -1/2 du_i/dx_j du_j/dx_i (positive in vortex cores) from the gradients on the wall-refined grid
        vol_u.append(((u - u.mean(dim=(0, 2), keepdim=True))/u_wall).detach().cpu().numpy().astype(np.float16))
        grads = PISOtorch.ComputeSpatialVelocityGradients(domain)[0] # [component i][0, j]: du_i/dx_j
        q = -0.5*sum(grads[i][0, j]*grads[j][0, i] for i in range(3) for j in range(3))
        vol_q.append((q*(viscosity/u_wall**2)**2).detach().cpu().numpy().astype(np.float16))
        return
    vorticity.append(np.concatenate([w.reshape(-1) for w in vorticity_fn(domain)]).astype(np.float32))
    if record_scalar:
        scalar.append(np.concatenate([d.reshape(-1) for d in V.scalar_2d(domain)]).astype(np.float32))
    if CASE == "rbc":
        extra["nusselt"].append(nusselt_fn(domain))

sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn,
        substeps="ADAPTIVE", time_step=TIME_STEP, corrector_steps=2, pressure_return_best_result=True, velocity_corrector="FD",
        log_interval=0, log_images=False, log_fn=record, **sim_args)
for name, value in sim_attrs.items():
    setattr(sim, name, value)
if CASE == "tcf":
    PISOtorch.CopyVelocityResultFromBlocks(domain)
if not LOAD_PATH:
    sim.make_divergence_free(max_iter=sim.linear_solve_max_iterations)
if prep_fn_after_projection is not None:
    sim.prep_fn = prep_fn_after_projection
record(domain) # initial state, t = 0
t0 = time.time()
sim.run(ITER, log_domain=False)
print("SIM TIME %.1f s for %d iterations, %d steps" % (time.time() - t0, ITER, sim.total_step))

if CASE == "tcf":
    data = dict(time_step=TIME_STEP, uxz=np.stack(uxz), uxy=np.stack(uxy), re_tau=np.array(re_tau),
        vol_u=np.stack(vol_u), vol_q=np.stack(vol_q), **extra)
    np.savez_compressed(os.path.join(OUT, "%s_frames.npz" % CASE), **data)
    print("saved", len(uxz), "frames")
    sys.exit(0)

cx = np.concatenate([c[0].reshape(-1) for c in centers]); cy = np.concatenate([c[1].reshape(-1) for c in centers])
block_shapes = np.array([c.shape[1:] for c in centers]) # (y, x) cells per block
data = dict(cx=cx, cy=cy, block_shapes=block_shapes, time_step=TIME_STEP, w=np.stack(vorticity), **extra)
if boundary_points:
    data["boundary_points"] = np.concatenate(boundary_points, axis=1).astype(np.float32)
    data["boundary_cells"] = np.concatenate(boundary_cells)
if record_scalar:
    data["s"] = np.stack(scalar)
np.savez_compressed(os.path.join(OUT, "%s_frames.npz" % CASE), **data)
print("saved", len(vorticity), "frames")
