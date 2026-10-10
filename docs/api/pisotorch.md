# PISOtorch (core module)

```python
import torch      # import torch first, PISOtorch links against it
import PISOtorch
```

The compiled core of PICT. It contains the data structures, the PISO operators with their backward kernels, the sparse linear solvers, and grid, resampling and per-cell linear-algebra utilities.

## Domain

```python
PISOtorch.Domain(spatialDims, viscosity, name, dtype, device, passiveScalarChannels=1, scalarViscosity=None)
```

| Argument | Description |
|---|---|
| `spatialDims` | 1, 2 or 3 |
| `viscosity` | global (kinematic) viscosity, CPU tensor of shape `[1]` |
| `name` | name used in logs and output |
| `dtype` | `torch.float32` or `torch.float64` |
| `device` | the CUDA device |
| `passiveScalarChannels` | number of passive scalar fields; **defaults to 1**, use 0 for none |
| `scalarViscosity` | optional scalar diffusivity: a 1D CUDA tensor with 1 value or one value per channel |

**Blocks**

| Method | Description |
|---|---|
| `CreateBlock(velocity=None, pressure=None, passiveScalar=None, vertexCoordinates=None, name="Block")` | create and add a block; at least one tensor must be given, the others are zero-initialized; vertex coordinates generate the transformation metrics |
| `CreateBlockWithSize(size, name="Block")` | create and add a zero-initialized block with unit cells; `size` is a `PISOtorch.Int4` |
| `AddBlock(block)`, `getBlock(index)`, `getBlocks()`, `getNumBlocks()` | manage and access blocks |

**Setup and copies**

| Method | Description |
|---|---|
| `PrepareSolve()` | validate the setup and create the secondary tensors (matrices, RHS, results, gradients); required before simulating |
| `UpdateDomainData()` | refresh the packed GPU structure after tensors were replaced |
| `IsInitialized()` | whether `PrepareSolve()` was called |
| `Copy(newName=None)` | copy the structure, **share** the primary tensors (velocity, pressure, ...) |
| `Clone(newName=None)` | copy the structure and **clone** the primary tensors |
| `Detach()`, `DetachFwd()`, `DetachGrad()` | detach all (forward / gradient) tensors from the autograd graph |

**Queries**

| Method | Description |
|---|---|
| `getSpatialDims()`, `getTotalSize()`, `getDevice()`, `getDtype()` | basic information; `getTotalSize()` is the total number of cells of all blocks |
| `hasPassiveScalar()`, `getPassiveScalarChannels()` | passive scalar setup |
| `getMaxVelocity(withBounds=True, computational=False)` | maximum absolute velocity component, optionally including boundaries and in computational (cell) units |
| `getMaxVelocityMagnitude(withBounds=True, computational=False)` | maximum velocity magnitude |
| `hasVertexCoordinates()`, `getVertexCoordinates()` | vertex coordinate tensors of all blocks |
| `GetCoordinateOrientation()` | the sign of the cell determinants if all blocks share the same orientation, 0 otherwise |
| `GetBoundaryFluxBalance()`, `CheckBoundaryFluxBalance(eps=1e-5)` | net flux through all boundaries (must be ~0 for incompressible flow) |
| `hasPrescribedBoundary()`, `isAllFixedBoundariesPassiveScalarTypeStatic()` | boundary queries |

**Viscosity**

| Member | Description |
|---|---|
| `viscosity`, `setViscosity(v)` | global viscosity |
| `hasBlockViscosity()` | whether any block has its own viscosity |
| `passiveScalarViscosity`, `setScalarViscosity(v)`, `clearScalarViscosity()`, `hasPassiveScalarViscosity()`, `isPassiveScalarViscosityStatic()` | scalar diffusivity |

**Solver data.** These are read-only attributes. The vectors have `set*` / `Create*` methods (e.g. `setVelocityRHS`, `CreateVelocityRHS`), `A` has `setA` / `CreateA`, and the matrices `C` and `P` are `CSRmatrix` objects whose values are set with `setValue` / `CreateValue`:

| Attribute | Content |
|---|---|
| `C` | advection–diffusion matrix (`CSRmatrix`) |
| `A` | diagonal of `C` |
| `P` | pressure matrix (`CSRmatrix`) |
| `velocityRHS`, `velocityResult` | right-hand side and result of the velocity solve (all blocks, flattened) |
| `scalarRHS`, `scalarResult` | the same for passive scalars |
| `pressureRHS`, `pressureRHSdiv`, `pressureResult` | \(\mathbf{h}\), \(\nabla\cdot\mathbf{h}\) and the pressure solution |

`CreateVelocityOnBlocks()`, `CreatePressureOnBlocks()` and `CreatePassiveScalarOnBlocks()` (re)create zero fields on all blocks. *(grad)* Every solver tensor has a gradient counterpart (`CGrad`, `AGrad`, `PGrad`, `velocityResultGrad`, ...), together with `CreateVelocityGradOnBlocks()` and similar.

## Block

Blocks are created with `Domain.CreateBlock` or `Domain.CreateBlockWithSize`.

**Fields**

| Member | Description |
|---|---|
| `name` | block name |
| `velocity`, `setVelocity(t)`, `getVelocity(computational)`, `CreateVelocity()` | velocity `[1, dims, (z,) y, x]`; `getVelocity(True)` returns it transformed to computational space with the inverse metrics |
| `pressure`, `setPressure(t)`, `CreatePressure()` | pressure `[1, 1, (z,) y, x]` |
| `passiveScalar`, `setPassiveScalar(t)`, `CreatePassiveScalar()`, `hasPassiveScalar()`, `getPassiveScalarChannels()` | passive scalars `[1, S, (z,) y, x]` |
| `velocitySource`, `setVelocitySource(t)`, `clearVelocitySource()`, `hasVelocitySource()`, `isVelocitySourceStatic`, `CreateVelocitySource(createStatic)` | momentum source, `[1, dims]` or per cell |
| `viscosity`, `setViscosity(t)`, `clearViscosity()`, `hasViscosity()`, `isViscosityStatic()` | block viscosity `[1, 1]` or per cell; replaces the global viscosity for the velocity (not for passive scalars) |

**Geometry**

| Member | Description |
|---|---|
| `vertexCoordinates`, `setVertexCoordinates(t)`, `hasVertexCoordinates()` | cell corner coordinates; setting them recomputes all metrics |
| `getCellCoordinates()` | cell centers |
| `getCellSizes()` | cell volumes `[1, 1, (z,) y, x]` |
| `transform`, `setTransform(transform, faceTransform=None)`, `hasTransform()` | per-cell metrics (matrix, inverse, determinant; channels last) |
| `faceTransform`, `hasFaceTransform()` | per-face metrics |
| `clearCoordsTransforms()` | back to unit cells |
| `GetCoordinateOrientation()` | orientation of the block's cells |

**Boundaries**

| Method | Description |
|---|---|
| `getBoundary(face)`, `getBoundaries()`, `getFixedBoundaries()` | access boundaries; `getFixedBoundaries()` returns `(index, boundary)` pairs |
| `CloseBoundary(face, velocity=None, passiveScalar=None)` | make `face` a `FixedBoundary` (Dirichlet); closes periodic/connected partners too |
| `CloseAllBoundaries()` | no-slip walls on all faces |
| `ConnectBlock(faceString, otherBlock, otherFaceString, axis1String="", axis2String="")` | connect two blocks (see [Grids](../guide/grids.md#connecting-blocks)); an overload takes face indices |
| `MakePeriodic(axis)`, `MakeAllPeriodic()` | periodic boundaries |
| `hasPrescribedBoundary()`, `isAllFixedBoundariesPassiveScalarTypeStatic()` | boundary queries |

**Other members**

| Member | Description |
|---|---|
| `getSizes()`, `getStrides()`, `getDim()`, `getSpatialDims()`, `getDevice()` | shape information (`Int4` in `xyzw` order) |
| `getMaxVelocity(withBounds=True, computational=False)`, `getMaxVelocityMagnitude(...)` | velocity maxima |
| `getParentDomain()` | the owning domain |
| `globalOffset`, `csrOffset`, `ComputeCSRSize()` | position of the block in the global vectors and matrices |
| `Detach()`, `DetachFwd()`, `DetachGrad()` | detach tensors |
| *(grad)* `velocityGrad`, `pressureGrad`, `passiveScalarGrad`, `velocitySourceGrad`, `viscosityGrad` (+ `set*`/`Create*`) | gradient tensors |

## Boundaries

**`FixedBoundary`** prescribes values on a face. It is created by `CloseBoundary` / `CloseAllBoundaries`.

| Member | Description |
|---|---|
| `velocity`, `setVelocity(t)`, `CreateVelocity(createStatic)` | boundary velocity, static `[1, dims]` or varying `[1, dims, ...]` with size 1 normal to the face |
| `velocityType`, `setVelocityType(t)`, `isVelocityStatic` | velocity condition type (only `DIRICHLET` is supported) and whether the velocity is static |
| `getVelocity(computational)` | velocity broadcast to the face shape, optionally in computational components |
| `getVelocityVarying(size=None)`, `makeVelocityVarying(size=None)` | broadcast a static velocity (and set it) |
| `GetFluxes()` | boundary-normal fluxes per boundary cell |
| `passiveScalar`, `setPassiveScalar(t)`, `CreatePassiveScalar(createStatic)`, `hasPassiveScalar()`, `getPassiveScalarChannels()`, `isPassiveScalarStatic()` | scalar boundary values |
| `passiveScalarTypes`, `setPassiveScalarType(type_or_list)`, `isPassiveScalarBoundaryTypeStatic()` | Dirichlet/Neumann per channel |
| `transform`, `hasTransform()`, `setTransform(t)`, `clearTransform()` | face metrics |
| `getSizes()`, `hasSize()`, `getStrides()`, `getSpatialDims()`, `getDevice()` | shape information |
| *(grad)* `velocityGrad`, `passiveScalarGrad` (+ `set*`/`Create*`) | gradients of the boundary values |

**`PeriodicBoundary`** has no parameters. **`ConnectedBoundary`** provides `getConnectedBlock()` and `axes`, the connected face and the axis mapping. The base class **`Boundary`** provides `type` and `getParentDomain()`. `StaticDirichletBoundary` and `VaryingDirichletBoundary` are deprecated legacy boundary types; use `FixedBoundary`.

## Enums and small types

| Type | Values / members |
|---|---|
| `Int4(x=0, y=0, z=0, w=0)` | integer 4-vector; `.x .y .z .w`, indexable |
| `BoundaryType` | `DIRICHLET`, `DIRICHLET_VARYING`, `NEUMANN`, `FIXED`, `CONNECTED`, `PERIODIC` |
| `BoundaryConditionType` | `DIRICHLET`, `NEUMANN` |
| `ConvergenceCriterion` | `NORM2`, `NORM2_NORMALIZED`, `ABS_SUM`, `ABS_MEAN`, `ABS_MAX` |
| `BoundarySampling` | `CONSTANT`, `CLAMP` (for resampling) |
| `CSRmatrix(value, index, row)` | sparse matrix: `value`, `index`, `row`, `setValue`, `CreateValue`, `getRows()`, `getSize()`, `copy()`, `clone()`, `detach()`, `WithZeroValue()`, `toType(dtype)` |
| `LinearSolverResultInfo` | `finalResidual`, `usedIterations`, `converged`, `isFiniteResidual` |

## PISO operators

These are the building blocks of a PISO step, normally orchestrated by [`Simulation`](simulation.md). `timeStep` is a CPU tensor of shape `[1]`.

| Function | Description |
|---|---|
| `SetupAdvectionMatrix(domain, timeStep, nonOrthoFlags, forPassiveScalar=False, passiveScalarChannel=0)` | assemble the implicit advection–diffusion matrix `C` (and `A`); used for velocity and passive scalars |
| `SetupAdvectionScalar(domain, timeStep, nonOrthoFlags)` | right-hand side for passive scalar advection |
| `SetupAdvectionVelocity(domain, timeStep, nonOrthoFlags, applyPressureGradient=False)` | right-hand side for velocity advection |
| `SetupPressureCorrection(domain, timeStep, nonOrthoMode=0, useFaceTransform=False, timeStepNorm=False)` | pressure matrix, RHS and divergence of RHS |
| `SetupPressureMatrix(...)`, `SetupPressureRHS(...)`, `SetupPressureRHSdiv(...)` | the individual parts (for non-orthogonal iterations) |
| `CorrectVelocity(domain, timeStep, version=0, timeStepNorm=False)` | velocity correction with the pressure gradient; `Simulation` uses version 1 (`"FD"`), 5 (`"FVM_CENTER"`) or 6 (`"FVM_FACE"`) |
| `Copy{Velocity,Pressure,Scalar}Result{To,From}Blocks(domain)` | copy between the global result vectors and the block fields |
| `ComputeVelocityDivergence(domain)` | per-cell velocity divergence as a flat global vector (shaped like `domain.pressureResult`) |
| `ComputePressureGradient(domain, useFVM, gradientInterpolation)` | pressure gradient as a flat global vector (shaped like `domain.velocityResult`); `useFVM=False` uses the gradient of the `"FD"` corrector (`gradientInterpolation` is then ignored), `useFVM=True` the finite-volume gradient with `gradientInterpolation=0` (interpolated metrics, as `"FVM_CENTER"`) or `3` (face metrics, as `"FVM_FACE"`); `channel_flow_learning.py` calls it with `False, 0`. Copy to the blocks with `domain.setVelocityResult(...)`, `UpdateDomainData()`, `CopyVelocityResultToBlocks(domain)` |
| `ComputeSpatialVelocityGradients(domain)` | \(\partial u_i/\partial x_j\) per block: nested list `[blocks][components]` of `NCDHW` tensors |
| `SGSviscosityIncompressibleSmagorinsky(domain, coefficient)` | Smagorinsky eddy viscosity per block (forward only) |
| *(grad)* `SetupAdvectionMatrixGrad`, `SetupAdvectionScalarGrad`, `SetupAdvectionVelocityGrad`, `SetupPressureCorrectionGrad`, `SetupPressureMatrixGrad`, `SetupPressureRHSGrad`, `SetupPressureRHSdivGrad`, `CorrectVelocityGrad`, `Copy*ResultGrad*` | backward kernels |

**Non-orthogonal flags** (`nonOrthoFlags`, a bit field; the values are defined as constants of `PISOtorch_simulation.Simulation`):

| Flag | Value | Meaning |
|---|---|---|
| `NON_ORTHO_DIRECT_MATRIX` | 1 | implicit non-orthogonal terms of the direct neighbors |
| `NON_ORTHO_DIRECT_RHS` | 2 | explicit (deferred) direct terms; less stable than the matrix variant |
| `NON_ORTHO_DIAGONAL_MATRIX` | 4 | not implemented |
| `NON_ORTHO_DIAGONAL_RHS` | 8 | explicit diagonal-neighbor terms |
| `NON_ORTHO_CENTER_MATRIX` | 16 | implicit center contribution |

`Simulation(non_orthogonal=True)` uses `16 | 1 | 8`; `0` means orthogonal.

## Linear solver

```python
infos = PISOtorch.SolveLinear(A, RHS, x, maxIterations, tolerance,
                              convergenceCriterion=PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED,
                              useBiCG=False, matrixRankDeficient=False, residualResetSteps=0,
                              transposeA=False, printResidual=False, returnBestResult=False,
                              BiCGwithPreconditioner=True)
```

Solves `A x = RHS` in place on the GPU, using CG, or BiCGStab with `useBiCG=True`, ILU(0)-preconditioned if `BiCGwithPreconditioner=True` (the default of the binding; `Simulation` passes its own `preconditionBiCG` setting). `maxIterations` (`torch.IntTensor([n])`) and `tolerance` (a 1-element tensor of the RHS dtype) are CPU tensors and must be passed explicitly: the binding declares scalar defaults (1000, 1e-8), but calling `SolveLinear(A, RHS, x)` without them fails with a `TypeError` because the arguments are tensors. A RHS that is a multiple of the matrix size is solved as a batch, and one `LinearSolverResultInfo` is returned per RHS. `transposeA=True` solves with \(A^\mathsf{T}\), which the backward pass uses.

`SparseOuterProduct(a, b, out_pattern)` computes \(\mathbf{a}\mathbf{b}^\mathsf{T}\) only on the sparsity pattern of `out_pattern`; this gives the matrix gradient of a linear solve. `TransformVectors(vectors, transforms, inverse)` applies per-cell transformations.

## Grids and metrics

| Function | Description |
|---|---|
| `CoordsToTransforms(vertexCoordinates)` | per-cell metrics from vertex coordinates |
| `CoordsToFaceTransforms(vertexCoordinates)` | per-face metrics |
| `MakeGrid2DNonUniformScale`, `MakeGridNDNonUniformScaleNormalized`, `MakeGridNDExpScaleNormalized`, `MakeCoordsNDNonUniformScaleNormalized` | GPU generators for refined grids and their transformations |

## Per-cell linear algebra

These functions operate on small vectors or matrices stored in the channel dimension of `NCDHW` tensors, with matrices flattened in row-major order:

| Function | Description |
|---|---|
| `matmul(A, B, transposeA=False, invertA=False, transposeB=False, invertB=False, transposeOutput=False, invertOutput=False)` | matrix/vector products per cell; *(grad)* `matmulGrad` |
| `InvertMatrix(matrices, inPlace=False)`, `VectorToDiagMatrix(vectors)` | inversion, diagonal matrices |
| `EigenDecomposition(matrices, outputEigenvalues=True, outputEigenvectors=True, normalizeEigenvectors=False)` | closed-form eigen decomposition of symmetric 2×2/3×3 matrices |
| `MakeBasisUnique(basisMatrices, sortingVectors, inPlace=False)` | make orthogonal bases unique (ordering, signs, handedness) |

## Resampling

| Function | Description |
|---|---|
| `SampleTransformedGridGlobalToLocal(globalData, globalTransform, localCoords, boundarySamplingMode, constantValue)` | gather from a uniform (globally transformed) grid onto curvilinear cells |
| `SampleTransformedGridLocalToGlobal(localData, localCoords, globalTransform, globalShape, fillMaxSteps=0)` | scatter curvilinear cells onto a uniform grid; returns data and weights |
| `SampleTransformedGridLocalToGlobalMulti(...)` | the same for a list of blocks |

The Python wrappers in [`lib.data.resample`](lib.md#resampling-libdataresample) are easier to use.
