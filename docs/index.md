---
title: Getting Started
---

<div class="pict-hero" markdown>

![PICT logo](imgs/pict_icon_transparent.webp){ .pict-hero__logo }

<div class="pict-hero__body" markdown>

# PICT

<p class="pict-hero__tagline">A differentiable, GPU-accelerated multi-block <strong class="pict-letter">PI</strong>SO solver with <strong class="pict-letter">C</strong>UDA for Py<strong class="pict-letter">T</strong>orch, built for simulation-coupled learning in fluid dynamics.</p>

[:fontawesome-solid-file-lines: Paper (JCP)](https://doi.org/10.1016/j.jcp.2025.114433){ .md-button .md-button--primary }
[:simple-arxiv: arXiv](https://arxiv.org/abs/2505.16992){ .md-button }
[:fontawesome-brands-github: Code](https://github.com/tum-pbs/PICT){ .md-button }

</div>
</div>

<figure class="pict-media" markdown>
  <video autoplay loop muted playsinline poster="imgs/vortex_street_poster.webp">
    <source src="assets/videos/vortex_street.mp4" type="video/mp4">
  </video>
  <figcaption>Vortex street behind a cylinder at Re = 200, simulated with PICT on a curvilinear 4-block ring grid (vorticity shown). See the <a href="examples/vortex_street/">example</a>.</figcaption>
</figure>

PICT solves the incompressible Navier–Stokes equations with the PISO algorithm on the GPU and is fully differentiable. Its fields are plain PyTorch tensors and every solver step supports autograd, so you can put neural networks *inside* the simulation and train them by backpropagating through many solver steps. The paper uses this to train correctors for coarse 2D simulations and a 3D sub-grid-scale model for turbulence that is learned from flow statistics alone.

- **Validated PISO solver** on flexible, deformed **multi-block** domains in 2D and 3D: O-grids, C-grids, wall-refined channels
- **Seamless PyTorch integration** for AI, deep learning and neural networks
- **Fully differentiable**, with hand-written backward CUDA kernels and adjoint linear solves
- **Modular CUDA operators** for efficient simulation at training and inference time

## Installation

```bash
git clone https://github.com/tum-pbs/PICT.git && cd PICT
conda create --name pisotorch --file environment.txt
conda activate pisotorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install cuda-toolkit=12.8 -c nvidia/label/cuda-12.8.1
cd extensions && python setup.py install && cd ..
```

Requires Linux and an NVIDIA GPU. 👉 [Full installation guide](installation.md), including tested PyTorch/CUDA versions and troubleshooting.

## PICT Paper

<div class="pict-paper" markdown>
<div class="pict-paper__title">PICT – A Differentiable, GPU-Accelerated Multi-Block PISO Solver for Simulation-Coupled Learning Tasks in Fluid Dynamics</div>
<div class="pict-paper__authors">Aleksandra Franz, Hao Wei, Luca Guastoni, Nils Thuerey · Technical University of Munich</div>
<div class="pict-paper__venue">Journal of Computational Physics 544 (2026) 114433 · open access</div>
</div>

- [📄 Journal of Computational Physics](https://doi.org/10.1016/j.jcp.2025.114433)
- [📄 arXiv preprint](https://arxiv.org/abs/2505.16992)
- [🧪 Results from the paper](background/results.md)

## Quickstart

A lid-driven cavity in a few lines, and an optimization through the solver:

=== "Simulate"

    ```python
    import torch
    import PISOtorch, PISOtorch_simulation
    import lib.data.shapes as shapes

    dtype = torch.float32
    grid = shapes.make_wall_refined_ortho_grid(32, 32, wall_refinement=["-x", "+x", "-y", "+y"], base=1.10)

    domain = PISOtorch.Domain(2, torch.tensor([1e-3], dtype=dtype), name="Cavity",
                              device=torch.device("cuda"), dtype=dtype, passiveScalarChannels=0)
    block = domain.CreateBlock(vertexCoordinates=grid.cuda().contiguous(), name="Cavity")
    block.CloseAllBoundaries()                                         # no-slip walls
    block.CloseBoundary("-y", torch.tensor([[1.0, 0.0]], device="cuda", dtype=dtype))  # moving lid
    domain.PrepareSolve()

    sim = PISOtorch_simulation.Simulation(domain=domain, time_step=0.25, substeps="ADAPTIVE",
                                          non_orthogonal=False, log_dir="./test_runs/cavity", log_interval=10)
    sim.run(iterations=100)                                            # Re = 1000, 25 time units
    ```

=== "Differentiate"

    ```python
    # recover an unknown forcing from a target flow by backpropagating through the solver
    # domain: a prepared PISOtorch.Domain, here the periodic 2D channel of the "Learning a Forcing" example
    # target_velocity: the velocity of a reference run with the unknown forcing
    forcing = torch.zeros([1, 2], device="cuda", dtype=dtype, requires_grad=True)
    optimizer = torch.optim.SGD([forcing], lr=1e-2)

    for step in range(20):
        opt_domain = domain.Copy()
        opt_domain.getBlock(0).setVelocitySource(forcing)
        opt_domain.PrepareSolve()
        PISOtorch_simulation.Simulation(opt_domain, time_step=0.1, non_orthogonal=False,
                                        differentiable=True, log_images=False).run(3, log_domain=False)
        loss = torch.sum((opt_domain.getBlock(0).velocity - target_velocity) ** 2)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
        opt_domain.Detach()
    ```

    See [Learning a Forcing](examples/learning_forcing.md) for the complete example.

Run scripts from the repository root, which contains `PISOtorch_simulation.py` and `lib/`. The [examples](examples/index.md) go much further, from vortex streets and airfoils to learned turbulence models.

## Example gallery

Benchmark flows from the sample scripts, simulated with PICT and rendered for this website. Each example page names its source script and walks through the code of the setup.

<div class="grid cards" markdown>

-   [![Backward-facing step, 3D](imgs/flows/bfs3d_vortices_streamlines_poster.webp)](examples/backward_facing_step.md)

    **[Backward-Facing Step](examples/backward_facing_step.md)** · 2D and 3D, Re = 1400

-   [![Kármán vortex street](imgs/flows/karman_vorticity_poster.webp)](examples/karman_vortex_street.md)

    **[Kármán Vortex Street](examples/karman_vortex_street.md)** · 8 blocks, Re = 600

-   [![Kolmogorov flow](imgs/flows/kolmogorov_vorticity_poster.webp)](examples/periodic_turbulence.md)

    **[Decaying & Kolmogorov Turbulence](examples/periodic_turbulence.md)** · periodic 2D and 3D

-   [![Cylinder in a channel, 3D](imgs/flows/cylinder3d_vortices_poster.webp)](examples/cylinder_channel.md)

    **[Cylinder in a Channel](examples/cylinder_channel.md)** · O-grid, 2D and 3D

-   [![Airfoil in a channel, 3D](imgs/flows/airfoil3d_vortices_poster.webp)](examples/airfoil_channel.md)

    **[Airfoil in a Channel](examples/airfoil_channel.md)** · 2D and 3D, Re = 3000

-   [![Rayleigh–Bénard convection](imgs/flows/rbc_temperature_poster.webp)](examples/rayleigh_benard.md)

    **[Rayleigh–Bénard Convection](examples/rayleigh_benard.md)** · 2D and 3D, Ra = 8·10<sup>5</sup>

-   [![Turbulent channel flow](imgs/flows/tcf_volume_poster.webp)](examples/channel_flow_re180.md)

    **[Turbulent Channel Flow](examples/channel_flow_re180.md)** · 3D, Re<sub>τ</sub> = 180

-   [![Vortex street behind a cylinder](imgs/vortex_street_poster.webp){ .thumb-left }](examples/index.md)

    **[All examples](examples/index.md)** · from the lid-driven cavity to learned turbulence models

</div>

## Background

Many problems in fluid dynamics benefit from combining numerical solvers with machine learning: learned corrections for coarse simulations, turbulence closures, flow control and inverse problems. Training such hybrid models works best when the network sees the consequences of its outputs over many simulation steps. This requires a solver that is accurate, fast, and **differentiable**.

PICT provides this for incompressible flows:

1. **The forward solver.** A finite-volume PISO solver on collocated, curvilinear multi-block grids, with implicit time integration, GPU linear solvers (CG and BiCGStab) and support for passive scalars and source terms. It is [validated](background/validation.md) on lid-driven cavities, Poiseuille flow, a backward-facing step and a 3D turbulent channel flow, including against OpenFOAM on the same mesh.
2. **The gradients.** Every operator of the PISO step has an analytic backward kernel, and linear solves are differentiated with adjoint solves. Gradient paths can be pruned to [trade accuracy for speed](guide/differentiable.md#gradient-paths-trading-accuracy-for-speed).
3. **The learning interface.** Fields are PyTorch tensors, networks are injected through [callbacks](guide/simulation.md#callbacks), and [multi-block convolutions](guide/learning.md#multi-block-convolutions) operate seamlessly across block boundaries.

In the paper, PICT is used to train correctors for a 2D vortex street and a backward-facing step. It is also used to train a **3D sub-grid-scale model for turbulent channel flow from reference statistics alone**. The resulting corrected low-resolution simulation is about 40× faster than OpenFOAM at higher resolution, while matching the reference statistics more accurately. Read more in [Numerical Method](background/method.md) and [Results from the Paper](background/results.md).

## Citation

If you find PICT useful for your research, please cite:

```bibtex
@article{franz2026pict,
  title     = {{PICT} -- A Differentiable, {GPU}-Accelerated Multi-Block {PISO} Solver for Simulation-Coupled Learning Tasks in Fluid Dynamics},
  author    = {Franz, Aleksandra and Wei, Hao and Guastoni, Luca and Thuerey, Nils},
  journal   = {Journal of Computational Physics},
  volume    = {544},
  pages     = {114433},
  year      = {2026},
  doi       = {10.1016/j.jcp.2025.114433}
}
```

(Feel free to also give the project a star on [GitHub](https://github.com/tum-pbs/PICT) if you like it.)

## Funding

This work was supported by the European Research Council (ERC-2019-COG #863850 SpaTe) and by the DFG Research Unit FOR 2987/1.

## License

The PICT code is released under the [Apache License 2.0](https://github.com/tum-pbs/PICT/blob/main/LICENSE). The paper is open access under CC BY 4.0.

---

> PICT is developed by the [Physics-based Simulation group](https://ge.in.tum.de/) at the Technical University of Munich &nbsp;&middot;&nbsp;
> [Other publications of the lab](https://ge.in.tum.de/publications/) &nbsp;&middot;&nbsp;
> GitHub [@tum-pbs](https://github.com/tum-pbs)
