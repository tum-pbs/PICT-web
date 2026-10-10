# Kármán Vortex Street in a Channel

!!! source "Source"

    - **Script:** [`karman_vortex_street_sample.py`](https://github.com/tum-pbs/PICT/blob/main/karman_vortex_street_sample.py)
    - **Setup:** adapted from the code of [INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](https://openreview.net/forum?id=s3Uk3lrfjy) (Wei et al., NeurIPS 2025)
    - **Run:** `python karman_vortex_street_sample.py` from the repository root

A rectangular obstacle sits in a closed channel and sheds alternating vortices into its wake. The domain is the 8-block ring of `vortex_street_sample.py` (see [Vortex Street](vortex_street.md#square-obstacle-in-a-channel)), with every block refined towards the obstacle and a finer resolution in the blocks next to it. A short perturbation of the initial velocity behind the obstacle breaks the symmetry, so the shedding starts early.

=== "Vorticity"

    <figure class="pict-media" markdown>
      <video autoplay loop muted playsinline poster="../../imgs/flows/karman_vorticity_poster.webp">
        <source src="../../assets/videos/karman_vorticity.mp4" type="video/mp4">
      </video>
      <figcaption>Vorticity at Re = 600 (blue: clockwise, red: counter-clockwise) in the full 16 × 8 channel, from t = 10, when the shedding is established, to t = 80. Every second iteration of 0.1 time units is shown. Simulated with PICT and rendered for this website.</figcaption>
    </figure>

=== "Passive scalar"

    <figure class="pict-media" markdown>
      <video autoplay loop muted playsinline poster="../../imgs/flows/karman_smoke_poster.webp">
        <source src="../../assets/videos/karman_smoke.mp4" type="video/mp4">
      </video>
      <figcaption>The passive scalar of the same run. It enters with a Gaussian profile through the part of the inflow in front of the obstacle. Simulated with PICT and rendered for this website.</figcaption>
    </figure>

**Setup** (default run of the sample)

- Channel of 16 × 8 with no-slip walls at the top and bottom, a 1.5 × 2 obstacle in the middle of the height, 3 units behind the inflow
- 8 blocks with 36,864 cells: 16 cells per unit length in the outer blocks (`res_scale=16`), 24 in the blocks next to the obstacle, refined towards the obstacle with a largest-to-smallest cell ratio of 3
- Gaussian inflow profile over the channel height with a peak velocity of 0.56 and a mean of 0.42 (`in_vel=1` times a normal density with variance 0.5 over the height mapped to [−1, 1]), and a Gaussian passive scalar profile in front of the obstacle
- Re = 600, based on the obstacle height and the inflow velocity scale `in_vel=1` (viscosity 2/600)
- 800 iterations of 0.1 time units with adaptive sub-steps

## Grid

<figure markdown>
  ![Grid around the rectangular obstacle](../imgs/flows/karman_grid_light.webp#only-light){ style="max-width: 32rem" }
  ![Grid around the rectangular obstacle](../imgs/flows/karman_grid_dark.webp#only-dark){ style="max-width: 32rem" }
  <figcaption>The grid around the obstacle (x = −2.75 to 4.25, y = −2.5 to 2.5), one color per block, every grid line drawn. The missing center block is the obstacle. Drawn with the setup function of the sample by <code>scripts/viz_example_grids.py</code>.</figcaption>
</figure>

`make8BlockKarmanSetup` takes the sizes in cells. With `obstacle_res_factor=1.5`, the blocks next to the obstacle get 1.5× more cells while the physical size stays the same. Each block is refined towards the obstacle with `make_weights_ratio` (the same helper as in the [backward-facing step](backward_facing_step.md#grid)), so that the ratio of the largest to the smallest cell does not depend on the resolution:

```python title="karman_vortex_street_sample.py" linenums="243" hl_lines="1 2"
    domain, prep_fn, layout, grids = make8BlockKarmanSetup(x=16*res_scale, y=8*res_scale, x_in=int(1.5*res_scale), y_in=int(obstacle_height*res_scale), x_pos=3*res_scale,
        in_vel=in_vel, in_var=0.5, viscosity=viscosity, scale=1/res_scale, refinement_ratio=3, obstacle_res_factor=1.5, wake_perturbation=wake_perturbation, dtype=dtype)
```

## Blocks and boundaries

The four faces around the missing center block are closed, which makes them the obstacle walls. The ring connections replace the default periodic boundaries, so the outer top and bottom faces of the channel become walls as well:

```python title="karman_vortex_street_sample.py" linenums="156" hl_lines="3 4 5 6"
    # set boundaries
    # close obstacle
    Btm.CloseBoundary("-y")
    Bml.CloseBoundary("+x")
    Bmr.CloseBoundary("-x")
    Bbm.CloseBoundary("+y")

    # connect 8-block ring
    # the outer boundaries of the channel are closed as the default periodic boundaries are replaced by the connections
    Btl.ConnectBlock("+x", Btm, "-x", "-y")
    Btm.ConnectBlock("+x", Btr, "-x", "-y")

    Btr.ConnectBlock("-y", Bmr, "+y", "-x")
    Bmr.ConnectBlock("-y", Bbr, "+y", "-x")

    Bbl.ConnectBlock("+x", Bbm, "-x", "-y")
    Bbm.ConnectBlock("+x", Bbr, "-x", "-y")

    Btl.ConnectBlock("-y", Bml, "+y", "-x")
    Bml.ConnectBlock("-y", Bbl, "+y", "-x")
```

The inflow profile is split into the parts of the three left blocks. The three outflow faces start with the mean inflow velocity, and the `PRE` callback advects them with the maximum inflow velocity:

```python title="karman_vortex_street_sample.py" linenums="178" hl_lines="2 6 10 11 12 16 20 33 37 38 39"
    # gaussian profile over the cells of the channel height
    inflow = shapes.get_grid_normal_dist([y], [0], [in_var], dtype=dtype)
    inflow = torch.reshape(inflow, [1,1,y,1])*in_vel
    mean_in = torch.mean(inflow).item()
    max_in = torch.max(inflow).item()
    char_vel = torch.tensor([[max_in, 0]], device=cuda_device, dtype=dtype) # NC
    inflow = torch.cat([inflow, torch.zeros_like(inflow)], axis=1)
    inflow_blocks = [_.to(cuda_device).contiguous() for _ in torch.split(inflow, y_sizes, dim=-2)]

    Btl.getBoundary("-x").setVelocity(inflow_blocks[2])
    Bml.getBoundary("-x").setVelocity(inflow_blocks[1])
    Bbl.getBoundary("-x").setVelocity(inflow_blocks[0])

    # passive scalar inflow in front of the obstacle
    inflow_scalar_mid = shapes.get_grid_normal_dist([y_in], [0], [in_var], dtype=dtype)
    Bml.getBoundary("-x").setPassiveScalar(torch.reshape(inflow_scalar_mid, [1,1,y_in,1]).to(cuda_device).contiguous())

    # also set up outflow boundaries at the end of the channel.
    # these will need special treatment during the simulation
    out_vel = torch.tensor([mean_in, 0], device=cuda_device, dtype=dtype).reshape([1,dims,1,1])

    Btr.getBoundary("+x").setVelocity((torch.ones_like(inflow_blocks[2]) * out_vel).contiguous())
    Bmr.getBoundary("+x").setVelocity((torch.ones_like(inflow_blocks[1]) * out_vel).contiguous())
    Bbr.getBoundary("+x").setVelocity((torch.ones_like(inflow_blocks[0]) * out_vel).contiguous())

    # create varying passive scalar for outflow
    Btr.getBoundary("+x").CreatePassiveScalar(False)
    Bmr.getBoundary("+x").CreatePassiveScalar(False)
    Bbr.getBoundary("+x").CreatePassiveScalar(False)

    # make the boundaries consistent/divergence-free
    out_bounds = [Btr.getBoundary("+x"), Bmr.getBoundary("+x"), Bbr.getBoundary("+x")]
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)

    # callback function to update the outflow during the simulation
    out_bound_indices = [domain.getBlocks().index(Btr), domain.getBlocks().index(Bmr), domain.getBlocks().index(Bbr)]
    def prep_fn(domain, time_step, **kwargs):
        out_bounds = [domain.getBlock(idx).getBoundary("+x") for idx in out_bound_indices]
        PISOtorch_simulation.update_advective_boundaries(domain, out_bounds, char_vel, time_step.cuda())
```

## Triggering the shedding

A symmetric setup sheds vortices only once small numerical errors have grown. To start the shedding early, the sample gives the wake block an upward velocity in a horizontal band through its center. The band follows the first part of a sine wave, and the initial projection turns it into a divergence-free perturbation:

```python title="karman_vortex_street_sample.py" linenums="60" hl_lines="7 8"
def make_wake_perturbation(res_x, res_y, half_width, dtype=torch.float32):
    # Upward velocity in a horizontal band of 2*half_width cells through the center of the wake block,
    # following the first part of a sine wave. It breaks the symmetry of the flow to trigger the vortex shedding.
    num_samples = max(30, 2*half_width)
    amplitude = max(5, 10*half_width/30)
    profile = amplitude*torch.sin(torch.linspace(0, 2*np.pi, num_samples, dtype=dtype))[:2*half_width]
    velocity = torch.zeros([1, 2, res_y, res_x], dtype=dtype)
    velocity[0, 1, res_y//2-half_width:res_y//2+half_width, :] = profile.reshape([-1, 1])
    return velocity
```

```python title="karman_vortex_street_sample.py" linenums="153"
    if wake_perturbation:
        Bmr.setVelocity(make_wake_perturbation(x_sizes[2], y_sizes[1], (y_sizes[1]+1)//int(obstacle_res_factor*4), dtype=dtype).to(cuda_device).contiguous())
```

Pass `wake_perturbation=False` to `karman_vortex_street_sample` to start from the unperturbed flow.

## Running the simulation

```python title="karman_vortex_street_sample.py" linenums="254" hl_lines="2 4 11 14 15"
    sim = PISOtorch_simulation.Simulation(domain=domain, block_layout=layout, prep_fn=None,
            substeps="ADAPTIVE", time_step=time_step, corrector_steps=2, pressure_tol=1e-6,
            advect_non_ortho_steps=1, pressure_non_ortho_steps=1, pressure_return_best_result=True,
            velocity_corrector="FD", non_orthogonal=False,
            log_interval=1, norm_vel=True,
            output_resampling_shape=resample_shape, output_resampling_fill_max_steps=resample_fill_steps,
            log_dir=log_dir, save_domain_name="domain",
            stop_fn=STOP_FN)

    # initialize the domain with velocity that matches in- and outflow
    sim.make_divergence_free()

    # set the outflow update only after the initial projection, which uses a time step of 1
    sim.prep_fn = prep_fn
    sim.run(iterations)
```

The output images are resampled to a uniform grid with twice the resolution of the outer blocks. A commented run at `res_scale=8` is faster, but according to the script it shows stronger numerical oscillations at the obstacle corners. `res_scale` corresponds to 16/`downsample_factor` of the INC code.

??? info "How the video was made"

    `scripts/sim_flow_examples.py karman` builds the domain with `make8BlockKarmanSetup` of the sample and runs it with the solver settings of the sample. A log callback records the vorticity and the scalar after every iteration. `scripts/render_flow_examples.py` interpolates the cell values to pixels and encodes the video. The color range is fixed for the whole run.
