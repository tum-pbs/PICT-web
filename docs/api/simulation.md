# PISOtorch_simulation

```python
import PISOtorch_simulation
```

The simulation driver. It composes the [`PISOtorch`](pisotorch.md) operators into PISO time steps. See [Running Simulations](../guide/simulation.md) for a guided introduction.

## Simulation

```python
PISOtorch_simulation.Simulation(domain=None, *, time_step=1.0, substeps=1, corrector_steps=2, adaptive_CFL=0.8,
    prep_fn=None, advection_use_BiCG=True, pressure_use_BiCG=False,
    scipy_solve_advection=False, scipy_solve_pressure=False,
    preconditionBiCG=False, BiCG_precondition_fallback=True,
    advection_tol=None, pressure_tol=None, convergence_tol=None, solver_double_fallback=False,
    advect_non_ortho_steps=1, pressure_non_ortho_steps=1,
    advect_reuse_result=False, pressure_reuse_result=False,
    normalize_pressure_result=True, pressure_return_best_result=False,
    advect_passive_scalar=True, pressure_time_step_normalized=False,
    velocity_corrector="FD", non_orthogonal=True,
    differentiable=False, exclude_advection_solve_gradients=False, exclude_pressure_solve_gradients=False,
    log_dir=None, log_interval=0, log_images=True, log_vtk=False, norm_vel=False, block_layout=None,
    output_mode3D="slice", log_fn=None,
    output_resampling_coords=None, output_resampling_shape=None, output_resampling_fill_max_steps=0,
    save_domain_name=None, stop_fn=lambda: False)
```

All arguments after `domain` are keyword-only. They can also be changed later as attributes of the simulation, most of them through properties with type checks.

!!! note
    `log_images=True` is the default, and `run()` writes the images of the initial state right away. Without a `log_dir`, it stops with `RuntimeError: no log_dir set.`, so pass `log_images=False` for runs without output.

**Time stepping**

| Parameter | Default | Description |
|---|---|---|
| `domain` | `None` | a `PISOtorch.Domain` on which `PrepareSolve()` was called |
| `time_step` | `1.0` | time step size; with `"ADAPTIVE"`, the time per iteration |
| `substeps` | `1` | fixed number of steps per iteration (each of size `time_step`), or `"ADAPTIVE"` |
| `adaptive_CFL` | `0.8` | target CFL number for adaptive sub-stepping |
| `corrector_steps` | `2` | PISO pressure correctors per step |
| `convergence_tol` | `None` | stop once the maximum velocity change per step falls below this value |

**Linear solvers**

| Parameter | Default | Description |
|---|---|---|
| `advection_use_BiCG` / `pressure_use_BiCG` | `True` / `False` | BiCGStab instead of CG |
| `preconditionBiCG` | `False` | ILU(0)-preconditioned BiCGStab |
| `BiCG_precondition_fallback` | `True` | retry a failed unpreconditioned BiCGStab solve with the preconditioner |
| `advection_tol` / `pressure_tol` | `None` | residual tolerances; `None` selects 1e-5 (float32) or 1e-8 (float64) |
| `pressure_return_best_result` | `False` | continue with the best CG iterate if the pressure solve does not converge |
| `solver_double_fallback` | `False` | retry a failed single-precision solve in double precision |
| `scipy_solve_advection` / `scipy_solve_pressure` | `False` | solve on the CPU with SciPy (debugging) |
| `advect_reuse_result` / `pressure_reuse_result` | `False` | previous result as the initial guess (ignored when differentiable) |
| `normalize_pressure_result` | `True` | subtract the mean pressure |
| `pressure_time_step_normalized` | `False` | divide the right-hand side of the pressure system by the time step (`timeStepNorm` of the pressure operators) |

The attribute `linear_solve_max_iterations` (default 5000) limits the iterations of each solve.

**Discretization**

| Parameter | Default | Description |
|---|---|---|
| `non_orthogonal` | `True` | non-orthogonal correction for skewed grids |
| `advect_non_ortho_steps` / `pressure_non_ortho_steps` | `1` / `1` | non-orthogonal corrector iterations |
| `velocity_corrector` | `"FD"` | `"FD"`, `"FVM_CENTER"` or `"FVM_FACE"`; only `"FD"` is differentiable |
| `advect_passive_scalar` | `True` | advect the passive scalars |

**Differentiability**

| Parameter | Default | Description |
|---|---|---|
| `differentiable` | `False` | route all operators through `PISOtorch_diff` |
| `exclude_advection_solve_gradients` | `False` | run the advection solves under `torch.no_grad()` (drops \(J^{\text{Adv}}\)) |
| `exclude_pressure_solve_gradients` | `False` | run the pressure solves under `torch.no_grad()` (drops \(J^{P}\)) |

**Callbacks, output and control**

| Parameter | Default | Description |
|---|---|---|
| `prep_fn` | `None` | dict of [callbacks](../guide/simulation.md#callbacks) |
| `log_fn` | `None` | `log_fn(domain=, out_dir=, it=, out_it=, total_step=)`, called after every iteration |
| `log_dir`, `log_interval` | `None`, `0` | output directory and interval (iterations) |
| `log_images`, `log_vtk` | `True`, `False` | image and VTK output |
| `norm_vel`, `block_layout`, `output_mode3D` | `False`, `None`, `"slice"` | image appearance |
| `output_resampling_coords`, `output_resampling_shape`, `output_resampling_fill_max_steps` | `None`, `None`, `0` | resample all blocks onto one uniform image |
| `save_domain_name` | `None` | save the final state into `log_dir` |
| `stop_fn` | `lambda: False` | checked regularly; return `True` to stop |

**Methods**

| Method | Description |
|---|---|
| `run(iterations, static=False, log_domain=True)` | run `iterations` output steps; `static=True` only advects passive scalars. Returns `False` if the boundary fluxes are unbalanced or the velocity becomes NaN. A failed linear solve is logged and stops the run. |
| `make_divergence_free(iterations=1, max_iter=1000)` | pressure projection of the current velocity field |
| `advect_static(iterations, time_step=None)` | passive-scalar-only advection |
| `save_domain(name="domain")`, `save_domain_images(...)`, `save_domain_vtk(...)` | manual output |
| `reset_step_counters()` | reset `total_step`, `total_time` and the output counters |
| `linear_solve(csrMat, rhs, x=None, use_BiCG=False, tol=None, max_iter=None, ...)` | solve with the simulation's settings and backend |

`total_step` and `total_time` count the performed PISO steps and the simulated time.

## Boundary helpers

| Function | Description |
|---|---|
| `balance_boundary_fluxes(domain, free_bounds, tol=None)` | rescale the velocity of the boundaries in `free_bounds` so that the net boundary flux is zero |
| `update_advective_boundaries(domain, bounds, velms, dt, tol=None)` | advective outflow update for `bounds`; `velms` is a global `[1, dims]` velocity or one tensor per boundary; then re-balances the fluxes |
| `get_max_time_step(domain, time_step_target, CFL_cond=0.8, with_transformations=True)` | time step and number of sub-steps for a CFL condition |

## Callback helpers

| Function | Description |
|---|---|
| `append_prep_fn(prep_fn, name, fn)` | add `fn` at the end of hook `name` |
| `prepend_prep_fn(prep_fn, name, fn)` | add `fn` at the start of hook `name` |

## StopHandler

```python
stop_handler = PISOtorch_simulation.StopHandler(log=None)
stop_handler.register_signal()     # SIGINT (Ctrl-C) sets the stop flag
...
Simulation(..., stop_fn=stop_handler)
stop_handler.unregister_signal()
```

`stop_handler()` returns whether a stop was requested. `stop()` requests one manually, and `reset()` clears the flag.
