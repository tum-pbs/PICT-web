# Boundary Conditions

Every face of every block has exactly one boundary:

| Boundary | Created by | Meaning |
|---|---|---|
| `PeriodicBoundary` | default for new blocks; `block.MakePeriodic(axis)` | the face wraps around to the opposite face of the same block |
| `ConnectedBoundary` | `block.ConnectBlock(...)` | the face is glued to a face of another block (see [Grids](grids.md#connecting-blocks)) |
| `FixedBoundary` | `block.CloseBoundary(...)`, `block.CloseAllBoundaries()` | prescribed values: Dirichlet velocity, Dirichlet or Neumann passive scalars |

Use `block.getBoundary("-y")` (or a face index 0–5) to access a boundary, and `block.getFixedBoundaries()` to list all `(index, boundary)` pairs of fixed boundaries.

## Walls

```python
block.CloseAllBoundaries()   # no-slip walls on all faces: velocity = 0, passive scalar = 0 (Dirichlet)
block.CloseBoundary("-y")    # a single no-slip wall
```

Closing a face that was periodic **also closes the opposite face**, because both faces belonged to the same periodic pair. Likewise, closing a connected face also closes the face of the block it was connected to. For example, `CloseBoundary("-y")` on a fresh block creates walls at `-y` and `+y`, which turns a periodic box into a periodic channel. Connecting a periodic face to another block also turns its opposite face into a wall (see [Grids](grids.md#connecting-blocks)).

## Prescribed velocity: moving walls and inflow

```python
block.CloseBoundary(face, velocity=None, passiveScalar=None)
```

The velocity is given in physical `xyz` components, either

- **static**, one value for the whole face, with shape `[1, dims]`, or
- **varying**, one value per boundary cell, with shape `[1, dims, (z,) y, x]`, where the face-normal axis has size 1. For the `-y` face of a 2D block with 32 cells in x, the shape is `[1, 2, 1, 32]`.

```python
# lid-driven cavity (minimal_lid_sample.py)
block.CloseAllBoundaries()
lid_vel = torch.tensor([[1.0, 0.0]], device=domain.getDevice(), dtype=dtype)
block.CloseBoundary("-y", lid_vel)            # or: block.getBoundary("-y").setVelocity(lid_vel)
```

A **rotating obstacle** is a wall with a tangential velocity per boundary cell. `vortex_street_torus_sample.py` computes the tangential direction of every cell on the inner ring and sets it as a varying velocity:

```python
for block, bound_vel_part in zip(torus_blocks, bound_vel_parts):   # bound_vel_part: [1, dims, 1, x]
    block.getBoundary("-y").setVelocity(bound_vel_part.contiguous())
```

The most important `FixedBoundary` methods:

| Method | Purpose |
|---|---|
| `setVelocity(t)` / `velocity` | set or read the boundary velocity (static or varying) |
| `makeVelocityVarying()` | broadcast a static velocity to a per-cell tensor, e.g. for outflow boundaries |
| `getVelocity(computational)` | velocity broadcast to the boundary shape, optionally transformed to computational space |
| `GetFluxes()` | boundary-normal fluxes of all boundary cells |
| `setPassiveScalar(t)`, `CreatePassiveScalar(createStatic)` | passive scalar boundary values |
| `setPassiveScalarType(t)` | `PISOtorch.BoundaryConditionType.DIRICHLET` or `.NEUMANN`, one value or a list per scalar channel |

## Passive scalar boundary conditions

Passive scalars use the value stored on the fixed boundary, interpreted according to the boundary condition type:

- **Dirichlet** (default): the value is prescribed at the face.
- **Neumann**: the value is the prescribed normal gradient. With the default value 0, the wall is insulated (no diffusive flux).

```python
# insulated obstacle walls for the scalar (vortex_street_sample.py)
scalar_type = PISOtorch.BoundaryConditionType.NEUMANN   # can also be a list, one entry per channel
obstacle_block.getBoundary("-y").setPassiveScalarType(scalar_type)

# scalar inflow profile
inflow_block.getBoundary("-x").setPassiveScalar(profile.cuda())   # [1, channels, (z,) y, 1]
```

## Pressure

Pressure needs no boundary conditions from you. At fixed boundaries the pressure equation uses a zero normal gradient. Pressure is only defined up to a constant, so by default the mean is subtracted after every pressure solve (`Simulation(normalize_pressure_result=True)`).

## Inflow and outflow

An inflow is a fixed boundary with the inflow velocity. An **outflow** needs more care. PICT implements a non-reflecting **advective outflow** by updating Dirichlet values on the outflow faces every time step:

1. Make the outflow velocity varying (and the scalar, if used), so every boundary cell can have its own value.
2. Balance the boundary fluxes, so that exactly as much fluid leaves the domain as enters it.
3. Before every step, advect the boundary values out of the domain with a characteristic velocity (typically the inflow velocity) and re-balance the fluxes.

```python
# from airfoil_sample.py
in_vel = torch.tensor([[1.0, 0.0]], device=cuda_device, dtype=dtype)
block_tail_upper.CloseBoundary("+x", in_vel)
block_tail_upper.getBoundary("+x").makeVelocityVarying()
block_tail_lower.CloseBoundary("+x", in_vel)
block_tail_lower.getBoundary("+x").makeVelocityVarying()

out_bounds = [block_tail_upper.getBoundary("+x"), block_tail_lower.getBoundary("+x")]
PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)      # step 2

char_vel = in_vel.clone()
def update_outflow(domain, time_step, **kwargs):                       # step 3
    out_bounds = [domain.getBlock(1).getBoundary("+x"), domain.getBlock(2).getBoundary("+x")]
    PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())

prep_fn = {"PRE": update_outflow}   # passed to Simulation(prep_fn=prep_fn)
```

`update_advective_boundaries` performs an implicit upwind step of \(\partial_t \phi + U_n \partial_n \phi = 0\) between the boundary face and the adjacent cell center:

\[
\phi_b \leftarrow \frac{\phi_b + \alpha\, \phi_P}{1+\alpha}, \qquad \alpha = 2\,\Delta t\, U_n ,
\]

where \(U_n\) is the characteristic velocity normal to the face, in computational (cell) units. It then calls `balance_boundary_fluxes` again. Velocities and passive scalars are both updated. A passive-scalar outflow needs varying Dirichlet values: `bound.CreatePassiveScalar(False)`.

## Global flux balance

For incompressible flow, the net flux through all boundaries must be zero, otherwise the pressure equation has no solution. `domain.GetBoundaryFluxBalance()` returns the net flux and `domain.CheckBoundaryFluxBalance(eps)` tests it. **`Simulation.run()` refuses to start** if the imbalance exceeds the pressure tolerance; it logs a warning and returns `False`. Use `balance_boundary_fluxes` to fix the outflow.

## Consistent initial conditions

An initial velocity field usually does not match the boundary conditions and is not divergence-free. A pressure projection fixes both:

```python
sim = PISOtorch_simulation.Simulation(domain=domain, ...)
sim.make_divergence_free()   # projection onto a divergence-free field that satisfies the boundaries
sim.run(iterations)
```
