# Gotchas

Things that commonly trip up new users, collected from the code.

## Imports and setup

- **Import `torch` before `PISOtorch`.** The extension links against PyTorch's libraries. Importing it first fails with errors like `ImportError: libc10.so: cannot open shared object file`.
- **Run scripts from the repository root.** `PISOtorch_simulation`, `PISOtorch_diff` and `lib` are not installed, only the compiled `PISOtorch` module is.
- **`run_tests.py` selects GPU 0.** The script sets `CUDA_VISIBLE_DEVICES="0"`. Edit it if that GPU is busy.

## Tensors

- **Batch size 1.** Every field tensor must have `N = 1`.
- **`zyx` vs. `xyz`.** Spatial dimensions are ordered `zyx` (a 2D velocity has shape `[1, 2, y, x]`), but vector components and coordinates are ordered `xyz` (channel 0 is u).
- **Devices.** Fields, boundary values and vertex coordinates must be contiguous CUDA tensors. The domain viscosity and the time step are CPU tensors of shape `[1]`.
- **At least 3 cells** along every axis of every block.
- **`passiveScalarChannels` defaults to 1.** Pass `0` if you do not need scalars, otherwise an extra solve runs every step.

## Domain life cycle

- **`PrepareSolve()` is mandatory** after creating a domain and after `Copy()`, `Clone()` or `load_domain()`. `Simulation` refuses domains that are not initialized.
- **`UpdateDomainData()` after replacing tensors.** Methods like `setVelocity`, `setVelocitySource` and `setPassiveScalar` store a new tensor reference, and the packed GPU structure only sees it after `UpdateDomainData()`. Callbacks that set tensors should end with it.
- **`Copy()` shares tensors.** A *non-differentiable* simulation on a copy modifies the original domain's fields in place. Use `Clone()` for an independent copy.
- **`Detach()` after every optimization step.** Otherwise reference cycles between the domain and the autograd graph keep GPU memory alive until training crashes.

## Boundaries

- **New blocks are periodic** on all faces.
- **Closing a face also closes its partner.** For a periodic face this is the opposite face of the same block, which becomes a no-slip wall. For a connected face it is the face of the other block.
- **Connecting a face that was periodic** turns the *opposite* face of that block into a no-slip wall. Set up connections first and boundary values afterwards, as the samples do.
- **Unbalanced fluxes stop the simulation.** If the net boundary flux exceeds the pressure tolerance, `run()` logs a warning and returns `False` without simulating. Use `balance_boundary_fluxes`.
- **Outflow boundaries must be varying.** `update_advective_boundaries` raises an error for static boundary velocities; call `makeVelocityVarying()` first. A passive-scalar outflow must be a varying Dirichlet scalar.
- **Mixed block orientation.** Blocks with different handedness trigger a warning and can cause problems; check `domain.GetCoordinateOrientation()`.

## Simulation parameters

- **`substeps=n` vs. `"ADAPTIVE"`.** With an integer, one iteration advances `n × time_step`. With `"ADAPTIVE"`, it advances exactly `time_step`.
- **No `log_dir`? Set `log_images=False`.** Image output is on by default and `run()` writes the initial state right away, so a `Simulation` without `log_dir` stops with `RuntimeError: no log_dir set.`
- **`non_orthogonal=True` is the default.** On orthogonal meshes the non-orthogonal terms vanish and `False` gives the same result. The differentiable workflows of the repository use `False`.
- **`time_step` in callbacks is a CPU tensor.** Use `time_step.cuda()` when combining it with GPU tensors.

## Physics and gradients

- **A block viscosity replaces the global viscosity** on that block, it is not added to it.
- **The Smagorinsky coefficient is not squared.** `SGSviscosityIncompressibleSmagorinsky(domain, C)` computes \(C\,\Delta^2|\bar S|\), so pass \(C_s^2\).
- **Only the `"FD"` velocity corrector is differentiable.**
- **No mesh gradients.** Coordinates and transformation metrics are constants for autograd.
