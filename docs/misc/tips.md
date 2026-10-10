# Tips & Tricks

## Setting up simulations

- **Start from a sample.** `minimal_lid_sample.py` is a complete setup in under 100 lines. Copy it and change the mesh and boundaries step by step.
- **Look at your mesh first.** `lib.util.output.plot_grids(domain.getVertexCoordinates(), color=[...], path=out_dir)` plots all blocks, one color per block. It shows wrong connections, flipped blocks and bad cells before you spend GPU time on them.
- **Make the initial state consistent.** Call `sim.make_divergence_free()` before `run()` whenever the initial velocity was set by hand or the boundaries changed.
- **Balance open boundaries.** With inflow and outflow, call `PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)` once during setup, and use `update_advective_boundaries` in a `PRE` callback.
- **Disable what you don't need.** Pass `passiveScalarChannels=0` if you do not need scalars, which saves a linear solve in every step. On orthogonal meshes, `non_orthogonal=False` skips the non-orthogonal terms, which vanish there anyway.

## Robustness

- **Adaptive time stepping** (`substeps="ADAPTIVE"`) keeps the CFL number below `adaptive_CFL`. Turbulent flows need small values, e.g. 0.1 for the channel flow.
- **Difficult pressure solves:** `pressure_return_best_result=True` continues with the best iterate instead of aborting. Check the log for warnings about solves that did not converge.
- **Difficult advection solves:** keep `BiCG_precondition_fallback=True`, which retries with an ILU(0) preconditioner. The obstacle samples use this combination:

    ```python
    sim.preconditionBiCG = False
    sim.BiCG_precondition_fallback = True
    ```

- **Double precision** (`dtype=torch.float64`) tightens the default solver tolerance from 1e-5 to 1e-8. This is useful for validation runs and gradient checks.
- **Fine grids need tighter tolerances.** The tolerances are absolute (RMS of the residual), so on fine grids a solve can count as converged after very few iterations. See the [note on the Validation page](../background/validation.md#2d-lid-driven-cavity).
- **Watch the log.** Every `log_interval`, the simulation logs the maximum velocity, pressure statistics and the velocity divergence. A growing divergence or velocity usually points to boundary or mesh problems.

## Output and analysis

- **Physical-space images.** Pass `output_resampling_shape=[...]` to get images resampled onto a uniform grid, in addition to the per-block images in computational space.
- **ParaView.** `log_vtk=True` writes VTK multi-block files (requires `pyvista`).
- **Custom diagnostics.** A `log_fn` gets the domain after every iteration, e.g. to compute vorticity with `PISOtorch.ComputeSpatialVelocityGradients(domain)` or to accumulate statistics with `lib.data.online_statistics`.
- **Checkpoints.** `save_domain_name="domain"` saves the final state. `lib.util.domain_io.load_domain` restores it, and you call `PrepareSolve()` afterwards. The channel-flow scripts build their warm-up and data pipeline on this (`load_run_id`).

## Learning and optimization

- **Fresh copy per step, detach after.** For every optimization step, create the working domain with `domain.Copy()`, call `PrepareSolve()`, simulate differentiably, call `backward()`, and finish with `domain.Detach()`.
- **Warm up without gradients.** Run the first steps with `differentiable=False` and only the last steps differentiably. This saves memory and time, and randomizing the number of warm-up steps diversifies the training states.
- **Approximate gradients first.** `exclude_advection_solve_gradients` and `exclude_pressure_solve_gradients` remove the backward linear solves. This is often enough early in training. Re-enable them for fine-tuning.
- **Curricula over the unroll length.** Increase the number of differentiable steps during training (e.g. 4 → 8 → 16) for long-term stability.
- **Convolutions across blocks.** Use `MultiblockConv` instead of per-block convolutions so that the receptive field crosses block interfaces seamlessly.
- **Verify gradients.** For custom operators or new setups, compare against finite differences in double precision, as `tests/gradcheck.py` does.
