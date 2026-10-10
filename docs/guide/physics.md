# Sources, Scalars & Viscosity

PICT solves the incompressible Navier–Stokes equations in kinematic form, without a density field:

\[
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\otimes\mathbf{u}) = -\nabla p + \nabla\cdot(\nu\nabla\mathbf{u}) + \mathbf{S}, \qquad \nabla\cdot\mathbf{u} = 0,
\]

together with optional passive scalars \(\phi_k\) that are advected and diffused by the flow:

\[
\frac{\partial \phi_k}{\partial t} + \nabla\cdot(\phi_k\,\mathbf{u}) = \nabla\cdot(\kappa_k \nabla \phi_k).
\]

This page covers the terms you control: the source \(\mathbf{S}\), the scalars \(\phi_k\), and the viscosities \(\nu\) and \(\kappa_k\).

## Velocity sources (forcing)

A **velocity source** is an acceleration added to the momentum equation. It enters both the predictor and the pressure correction:

```python
block.setVelocitySource(torch.tensor([[1.0, 0.0]], device=cuda_device, dtype=dtype))  # static: [1, dims]
block.setVelocitySource(S_field)          # per cell: [1, dims, (z,) y, x]
block.clearVelocitySource()
domain.UpdateDomainData()                 # after changing it during a simulation
```

Typical uses:

- a constant **pressure-gradient forcing** that drives a periodic channel,
- **body forces** such as buoyancy (below),
- **learned corrections**: a neural network predicts \(\mathbf{S}\) in every step. The source is differentiable, so gradients flow back into the network (see [Training Neural Networks](learning.md)).

Sources that change over time are set in a callback. As an example, the turbulent channel flow (`channel_flow_sim.py`) recomputes a uniform streamwise forcing in every step from an estimate of the mean wall shear stress:

```python
# d_y: distances of the first and the last cell centers from the walls at y = -1 and y = +1
pos_y = torch.mean(domain.getBlock(0).getCellCoordinates()[0, 1], dim=(0, 2))
d_y = (1 + pos_y[0].cpu().numpy(), 1 - pos_y[-1].cpu().numpy())

def pfn_set_forcing(domain, **kwargs):              # a PRE callback
    block = domain.getBlock(0)
    viscosity = domain.viscosity.to(domain.getDevice())
    mean_vel_u = torch.mean(block.velocity[0, 0], dim=(0, 2))       # mean u over x and z, per y
    tau_wall_n = viscosity * mean_vel_u[0] / d_y[0]                 # lower wall
    tau_wall_p = viscosity * mean_vel_u[-1] / d_y[-1]               # upper wall
    forcing = (tau_wall_n + tau_wall_p) * 0.5
    block.setVelocitySource(torch.tensor([[forcing, 0, 0]], dtype=domain.getDtype(), device=domain.getDevice()))
```

## Passive scalars

Passive scalars are additional fields such as smoke density, temperature or dye. They are transported by the flow but do not act on it, unless you couple them yourself.

```python
domain = PISOtorch.Domain(dims, viscosity, name="D", device=cuda_device, dtype=dtype,
                          passiveScalarChannels=2,                  # number of scalar fields
                          scalarViscosity=torch.tensor([1e-3, 5e-4], device=cuda_device, dtype=dtype))
block.passiveScalar        # [1, 2, (z,) y, x]
block.setPassiveScalar(t)  # replace the field (then call domain.UpdateDomainData())
```

- The **diffusivity** is given by `scalarViscosity` (or `domain.setScalarViscosity`), a 1D CUDA tensor with one value for all channels or one value per channel. Without it, the scalars use the global viscosity of the domain. Block viscosities are not applied to scalars.
- All channels are solved together with one matrix, unless per-channel diffusivities or per-channel boundary condition types are set. Then each channel gets its own solve.
- Boundary values and types (Dirichlet/Neumann) are set per fixed boundary, see [Boundary Conditions](boundaries.md#passive-scalar-boundary-conditions).

!!! tip "No scalars? Say so."
    The `Domain` constructor defaults to `passiveScalarChannels=1`, which costs an extra linear solve per step. Pass `passiveScalarChannels=0` if you do not need scalars.

### Buoyancy: coupling a scalar back into the flow

The buoyancy sample (`buoyancy_sample.py`) turns a density scalar into an upward force with three callbacks: one feeds density into a source region, one converts density into a velocity source, and one clamps the density to non-negative values:

```python
vel_src_x = torch.zeros_like(block.passiveScalar)

def density_inflow_fn(domain, **kwargs):              # PRE: keep the source region filled
    block = domain.getBlock(0)
    block.setPassiveScalar(torch.maximum(block.passiveScalar, density_inflow).contiguous())
    domain.UpdateDomainData()

def buoyancy_fn(domain, time_step, **kwargs):          # PRE_VELOCITY_SETUP: density -> upward force
    block = domain.getBlock(0)
    block.setVelocitySource(torch.cat([vel_src_x, block.passiveScalar * buoyancy_factor], dim=1))
    domain.UpdateDomainData()

def clamp_density(domain, **kwargs):                   # POST: keep the density non-negative
    block = domain.getBlock(0)
    block.setPassiveScalar(torch.clamp(block.passiveScalar, min=0).contiguous())
    domain.UpdateDomainData()

prep_fn = {"PRE": [density_inflow_fn], "PRE_VELOCITY_SETUP": [buoyancy_fn], "POST": clamp_density}
```

The `PRE_VELOCITY_SETUP` hook runs after the scalar has been advected and before the momentum system is assembled, so the force uses the up-to-date density. See the [Buoyant Plume](../examples/buoyancy.md) example for the result.

## Viscosity

| Level | How | Shape |
|---|---|---|
| global | `PISOtorch.Domain(dims, viscosity, ...)`, `domain.setViscosity(v)` | `[1]` |
| per block | `block.setViscosity(v)` | `[1, 1]` |
| per cell | `block.setViscosity(v)` | `[1, 1, (z,) y, x]` |

A block viscosity **replaces** the domain viscosity on that block. To add an eddy viscosity to the molecular one, add the two yourself. `block.clearViscosity()` returns to the global value. The advection–diffusion matrices are differentiable with respect to all three levels of viscosity.

## Turbulence modeling

PICT resolves the flow on the given grid; there is no built-in turbulence model in the PISO loop. A Smagorinsky helper and per-cell viscosities let you add one:

```python
coefficient = torch.tensor([C], dtype=dtype)   # CPU tensor
nu_t = PISOtorch.SGSviscosityIncompressibleSmagorinsky(domain, coefficient)   # list: one tensor per block
```

computes \(\nu_t = C\,\Delta^2\,|\bar S|\) with \(|\bar S| = \sqrt{2\bar S_{ij}\bar S_{ij}}\) and \(\Delta^2\) the largest squared cell-edge length. The coefficient is used as given, so pass \(C_s^2\) for the classical \((C_s\Delta)^2|\bar S|\) form. This helper is forward-only, i.e. not differentiable. `channel_flow_sim.py` applies it in a `PRE` callback, optionally with van Driest damping near the walls:

```python
def add_block_SGS_viscosity(domain, **kwargs):
    domain.UpdateDomainData()
    SGS_viscosities = PISOtorch.SGSviscosityIncompressibleSmagorinsky(domain, SGS_coefficient)
    base_viscosity = domain.viscosity.to(cuda_device)
    for idx, (block, visc) in enumerate(zip(domain.getBlocks(), SGS_viscosities)):
        if use_van_driest:
            visc = visc * van_driest_scale_sqr[idx]
        block.setViscosity(visc + base_viscosity)    # block viscosity replaces the global one
    domain.UpdateDomainData()
```

The approach the paper focuses on is a **learned** model: a CNN predicts a corrective forcing \(\mathbf{S}_\theta\) for a coarse simulation and is trained through the differentiable solver. See [Turbulent Channel Flow SGS Model](../examples/turbulent_channel_flow.md).

## Further utilities

| Function | Purpose |
|---|---|
| `PISOtorch.ComputeVelocityDivergence(domain)` | per-cell velocity divergence |
| `PISOtorch.ComputePressureGradient(domain, useFVM, gradientInterpolation)` | pressure gradient (e.g. `False, 0` for the finite-difference gradient); both return flat global vectors, see the [API reference](../api/pisotorch.md#piso-operators) |
| `PISOtorch.ComputeSpatialVelocityGradients(domain)` | \(\partial u_i/\partial x_j\) in physical space for every block, e.g. for vorticity or strain-rate features |
| `PISOtorch.matmul`, `InvertMatrix`, `EigenDecomposition`, `MakeBasisUnique` | per-cell small linear algebra on matrices stored in the channel dimension (e.g. invariant tensor features for learned closures); `PISOtorch_diff.matmul` is the differentiable version |
| `lib.data.online_statistics` | online (streaming) mean, (co)variance, higher moments, PSD, TKE budgets and temporal correlation, written in PyTorch |
