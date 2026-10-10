# PISOtorch_diff

```python
import PISOtorch_diff
```

Autograd-enabled versions of the PISO operators. Each function wraps the corresponding [`PISOtorch`](pisotorch.md) operator in a `torch.autograd.Function`, whose backward pass calls the hand-written gradient kernel. `Simulation(differentiable=True)` uses this module automatically, so you only need it directly for custom algorithms. See [Differentiable Simulation](../guide/differentiable.md).

## Operators

The PISO operators take the same arguments as their `PISOtorch` counterparts; the linear solve and `matmul` have their own signatures:

| Function | Tracked inputs (receive gradients) |
|---|---|
| `SetupAdvectionMatrix(domain, time_step, non_ortho_flags, forPassiveScalar=False, passiveScalarChannel=0)` | velocity, boundary velocities, viscosity (global and block / scalar) |
| `SetupAdvectionScalar(domain, time_step, non_ortho_flags)` | passive scalars, boundary scalars and velocities, viscosity and scalar diffusivity (+ scalar result for non-orthogonal terms) |
| `SetupAdvectionVelocity(domain, time_step, non_ortho_flags, apply_pressure_gradient=False)` | velocity, velocity source, boundary velocities, viscosities (+ velocity result for non-orthogonal terms) |
| `SetupPressureCorrection(domain, time_step, non_ortho_flags, use_face_transform=False, timeStepNorm=False)` | velocity, velocity source, velocity result, boundary velocities, viscosities, `A`, `C` (+ pressure result for non-orthogonal terms) |
| `SetupPressureMatrix(...)`, `SetupPressureRHS(...)`, `SetupPressureRHSdiv(...)` | the parts of the above |
| `CorrectVelocity(domain, time_step, version, timeStepNorm=False)` | pressure, `A`, pressure RHS |
| `CopyVelocityResultToBlocks`, `CopyVelocityResultFromBlocks`, `CopyPressureResultToBlocks`, `CopyScalarResultToBlocks`, `CopyScalarResultFromBlocks` (all `(domain)`) | the copied fields; the backward pass of `CopyScalarResultFromBlocks` is not implemented (it raises `NotImplementedError`) |
| `linear_solve_GPU(csrMat, rhs, transpose=False, use_BiCG=False, tol=None, max_iter=5000, return_best_result=False, double_fallback=False, BiCG_with_preconditioner=True, BiCG_precondition_fallback=False)` | matrix values and right-hand side; backward via the adjoint solve |
| `matmul(vectorMatrixA, vectorMatrixB, transposeA=False, invertA=False, transposeB=False, invertB=False, transposeOutput=False, invertOutput=False)` | both operands (no gradients through inverted matrices) |

Grid transformations are inputs to most operators, but they are treated as constants: there are no gradients with respect to the mesh.

## Gradient control

| Function | Description |
|---|---|
| `exclude_gradient(function_name, grad_name, is_gradient_input)` | zero a specific gradient of an operator, e.g. `exclude_gradient("SetupAdvectionMatrix", "VISCOSITY_GRAD", False)`. `grad_name` must end in `_GRAD`. `is_gradient_input` selects incoming (`True`) or outgoing (`False`) gradients. |
| `clear_excluded_gradients()` | remove all exclusions |

The coarse-grained alternative is `Simulation(exclude_advection_solve_gradients=True, exclude_pressure_solve_gradients=True)`.

## Domain helpers

| Function | Description |
|---|---|
| `detach_domain(domain)`, `detach_domain_fwd(domain)`, `detach_domain_grad(domain)` | Python equivalents of `Domain.Detach()` / `DetachFwd()` / `DetachGrad()` |
| `reset_domain(domain)` | re-create zero fields on all blocks and call `PrepareSolve()` |

## Errors

`LinsolveError` (a `RuntimeError`) is raised when a linear solve does not converge, unless the best result is accepted, or when the residual becomes non-finite.
