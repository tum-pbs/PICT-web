# Running Simulations

`PISOtorch_simulation.Simulation` advances a prepared `Domain` in time with the PISO algorithm. It handles time stepping, linear solver settings, user callbacks, logging and output. The same class runs the [differentiable mode](differentiable.md).

```python
import PISOtorch_simulation

sim = PISOtorch_simulation.Simulation(
    domain=domain,              # must have been finalized with domain.PrepareSolve()
    time_step=0.25,             # physical time per iteration (see below)
    substeps="ADAPTIVE",        # CFL-based sub-stepping
    corrector_steps=2,          # PISO pressure corrector steps
    non_orthogonal=False,       # True for skewed meshes
    pressure_tol=1e-5,
    velocity_corrector="FD",
    log_dir=run_dir, log_interval=1,
)
sim.run(iterations=100)
```

## Time stepping

| `substeps` | One iteration of `run()` advances | Notes |
|---|---|---|
| integer `n ≥ 1` | `n × time_step` | `n` PISO steps of fixed size `time_step` |
| `"ADAPTIVE"` | exactly `time_step` | Split into sub-steps so that the CFL number stays below `adaptive_CFL` (default 0.8). After every sub-step, the size of the remaining sub-steps is recomputed from the current maximum velocity (in computational, i.e. cell, units, including boundary values). |

The time integration is implicit Euler. The cavity, obstacle, airfoil and buoyancy samples use adaptive steps with the default limit of 0.8, as does the paper's vortex street. For the turbulent channel flow validation, the paper uses CFL < 0.1, which "was necessary to reduce numerical diffusion introduced by larger time steps", and `channel_flow_sim.py` notes that higher values might result in a laminar flow. `iterations` counts the output steps: logging, image output and `log_fn` run once per iteration, while the `prep_fn` [callbacks](#callbacks) run in every sub-step.

`run(iterations, static=True)` only advects the passive scalars with the current, frozen velocity field. `make_divergence_free()` runs a single pressure projection, which is useful for initial conditions.

## The PISO step

Each (sub-)step consists of:

1. **Passive scalar advection** (if the domain has scalars): an implicit advection–diffusion solve for all scalar channels. The channels are solved as a batch with one matrix, or one by one if per-channel diffusivities or boundary condition types are set.
2. **Velocity predictor:** the implicit advection–diffusion solve for all velocity components (BiCGStab).
3. **`corrector_steps` pressure correctors** (default 2): a pressure Poisson solve (CG), then a velocity correction.

Details and equations are on the [Numerical Method](../background/method.md) page.

## Solver parameters

| Parameter | Default | Meaning |
|---|---|---|
| `corrector_steps` | `2` | number of PISO pressure corrector steps |
| `non_orthogonal` | `True` | enable the non-orthogonal correction terms (needed for skewed meshes; set `False` on orthogonal meshes for speed) |
| `advect_non_ortho_steps`, `pressure_non_ortho_steps` | `1`, `1` | non-orthogonal corrector iterations of the advection and pressure solves |
| `advection_use_BiCG` | `True` | BiCGStab for the (non-symmetric) advection–diffusion systems |
| `pressure_use_BiCG` | `False` | CG for the (symmetric) pressure system |
| `preconditionBiCG` | `False` | use an ILU(0) preconditioner for BiCGStab |
| `BiCG_precondition_fallback` | `True` | if an unpreconditioned BiCGStab solve fails, retry with the preconditioner |
| `advection_tol`, `pressure_tol` | `None` | residual tolerances; `None` selects 1e-5 for float32 and 1e-8 for float64 |
| `pressure_return_best_result` | `False` | if CG does not converge, continue with the best iterate instead of raising an error |
| `velocity_corrector` | `"FD"` | pressure gradient for the velocity correction: `"FD"`, `"FVM_CENTER"` or `"FVM_FACE"`. **Only `"FD"` is differentiable.** |
| `normalize_pressure_result` | `True` | subtract the mean pressure after each solve |
| `advect_passive_scalar` | `True` | advect passive scalars, if the domain has any |
| `advect_reuse_result`, `pressure_reuse_result` | `False` | use the previous result as the initial guess (ignored in differentiable mode; reusing the advection result can destabilize turbulent simulations) |
| `convergence_tol` | `None` | stop when the maximum velocity change of a step drops below this value (steady-state problems) |
| `differentiable` | `False` | use the autograd-enabled backend, see [Differentiable Simulation](differentiable.md) |

The maximum number of solver iterations is set by the attribute `sim.linear_solve_max_iterations` (default 5000). A failed solve raises `PISOtorch_diff.LinsolveError`. `run()` catches it, logs the error and stops the simulation.

## Callbacks

Custom logic is injected with **callbacks** passed as `prep_fn`, a dictionary mapping a hook name to a function or a list of functions. All callbacks are called with keyword arguments, so they should accept `**kwargs`:

```python
def apply_forcing(domain, time_step, total_step, **kwargs):
    domain.getBlock(0).setVelocitySource(compute_forcing(domain))
    domain.UpdateDomainData()      # always update after replacing tensors

prep_fn = {"PRE": [apply_forcing]}
PISOtorch_simulation.append_prep_fn(prep_fn, "POST", my_post_step_fn)   # helpers to add hooks
sim = PISOtorch_simulation.Simulation(domain=domain, prep_fn=prep_fn, ...)
```

Hooks in the order in which they are called during one PISO step:

| Hook | Called | Typical use |
|---|---|---|
| `PRE` | at the start of every (sub-)step | update inflow/outflow boundaries, inject forcing or network output, set an SGS viscosity |
| `POST_SCALAR_SETUP` | after the scalar advection RHS is built | modify the scalar RHS |
| `PRE_VELOCITY_SETUP` | before the velocity advection system is built | set velocity sources that depend on the new scalar (e.g. buoyancy) |
| `POST_VELOCITY_SETUP` | after the velocity RHS is built | modify the velocity RHS |
| `POST_PREDICTION` | after the predictor solve | inspect or modify the predicted velocity |
| `POST_PRESSURE_SETUP` | after the pressure system is built (every corrector) | |
| `POST_PRESSURE_RESULT` | after every pressure solve | |
| `POST_PRESSURE_NON_ORTHO` | after the non-orthogonal pressure iterations of a corrector | |
| `POST_VELOCITY_CORRECTION` | after every velocity correction | |
| `POST` | at the end of the step, after results are written to the blocks | clamp fields, accumulate statistics |

Every hook receives `domain`, `local_step`, `time_step`, `total_step` and `total_time`. `POST_SCALAR_SETUP` and `POST_VELOCITY_SETUP` also receive `no_step`, the index of the non-orthogonal iteration. `time_step` is a CPU tensor, so use `time_step.cuda()` when combining it with GPU tensors. `make_divergence_free()` and `run(static=True)` also call the `PRE` hook.

`log_fn(domain, out_dir, it, out_it, total_step)` is called once after every iteration, e.g. for custom diagnostics or rendering.

## Output and logging

```python
from lib.util.logging import setup_run, get_logger, close_logging

run_dir = setup_run("./test_runs", name="my-run")   # creates ./test_runs/<timestamp>_my-run/ with a log file
LOG = get_logger("Main")
stop_handler = PISOtorch_simulation.StopHandler(LOG)
stop_handler.register_signal()                      # Ctrl-C stops the simulation gracefully
sim = PISOtorch_simulation.Simulation(domain=domain, log_dir=run_dir, log_interval=1,
                                      stop_fn=stop_handler, save_domain_name="domain", ...)
sim.run(100)
stop_handler.unregister_signal()
close_logging()
```

| Parameter | Effect |
|---|---|
| `log_dir`, `log_interval` | output directory and interval (in iterations) of statistics and files; `0` disables them |
| `log_images` | write velocity, pressure, scalar and velocity-source images (PNG); **on by default**, see the note below |
| `output_resampling_shape`, `output_resampling_coords` | additionally resample the fields of all blocks onto one uniform image in physical space |
| `output_resampling_fill_max_steps` | fill holes left by the resampling of coarse curvilinear cells |
| `norm_vel` | normalize the velocity images by the current maximum velocity |
| `block_layout`, `output_mode3D` | arrangement of block images and the 3D reduction (`"slice"`) |
| `log_vtk` | write VTK files (requires `pyvista`) for ParaView & co. |
| `save_domain_name` | save the final state with [`save_domain`](concepts.md#saving-and-loading) |
| `stop_fn` | callable returning `True` to stop, e.g. a `StopHandler` |

!!! warning "Without a `log_dir`, switch off `log_images`"
    `log_images=True` is the default, and `run()` writes the images of the initial state even when `log_interval=0`. Without a `log_dir`, `run()` therefore stops with `RuntimeError: no log_dir set.` For runs without output, e.g. inside an optimization loop, pass `log_images=False`.

At every log interval, the log contains the maximum velocity and magnitude, pressure statistics and the velocity divergence. Further helpers:

- `lib.util.GPU_info.get_available_GPU_id()` picks the least-used GPU.
- `lib.util.memory_usage.MemoryUsage` tracks GPU memory.
- `lib.util.profiling.SAMPLE` is a profiling context manager.
