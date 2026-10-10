# FAQ

??? question "What does PICT stand for?"
    **P**ISO solver with CUDA for Py**T**orch. The logo spells it out as "Differentiable **PI**SO-Solver with **C**UDA for Py**T**orch". Inside the code, the compiled module is called `PISOtorch`.

??? question "Does PICT run on the CPU?"
    No. All operators are CUDA kernels, and fields must be CUDA tensors. A CUDA-capable NVIDIA GPU is required. The SciPy options of `Simulation` (`scipy_solve_advection`, `scipy_solve_pressure`) only move the linear solves to the CPU, for debugging.

??? question "Can I use several GPUs?"
    Not for one simulation: a domain lives on a single GPU. Independent simulations can of course run on different GPUs. Most samples pick the least-used GPU with `lib.util.GPU_info.get_available_GPU_id`.

??? question "Which kinds of flows can I simulate?"
    Incompressible, single-phase flows in 2D and 3D (the data structures also accept 1D), governed by the Navier–Stokes equations in kinematic form, with optional passive scalars and arbitrary source terms. There is no density field and no compressibility. Effects like buoyancy are added through [velocity sources](../guide/physics.md#buoyancy-coupling-a-scalar-back-into-the-flow).

??? question "Which meshes are supported?"
    Structured multi-block meshes: every block is a regular grid that can be stretched and curved, and blocks connect conformingly, face to face. This covers channels, cavities, rings around cylinders, C-grids around airfoils and many more, see [Grids](../guide/grids.md). Unstructured meshes are not supported.

??? question "Which turbulence models are included?"
    PICT simulates the resolved flow. It provides a Smagorinsky helper and per-cell viscosities to build eddy-viscosity models (see [Turbulence modeling](../guide/physics.md#turbulence-modeling)). The main focus is on **learned** closures trained through the differentiable solver, see the [channel flow example](../examples/turbulent_channel_flow.md).

??? question "What can I differentiate with respect to?"
    Initial states, velocity sources (including network outputs), viscosities, passive scalars and boundary values. There are no gradients with respect to the mesh or the time step size. See [Differentiable Simulation](../guide/differentiable.md#what-you-can-differentiate).

??? question "Can I train with mini-batches?"
    A domain holds exactly one sample (batch size 1). To average over several samples, e.g. initial conditions, simulate them one after another and accumulate their gradients before the optimizer step.

??? question "Which precision should I use?"
    `float32` is the default and was used for all experiments in the paper. `float64` is supported throughout and is useful for gradient checks and tight tolerances. Half precision is not supported.

??? question "How fast is it?"
    It depends on the resolution, the number of solver iterations and the GPU. As a reference from the paper: a 3D turbulent channel flow at 64 × 32 × 32 with a learned SGS model simulates 20 eddy turnover times in 223 s on a single GPU. The linear solves typically take 70–90% of the runtime.

??? question "Where does the output go?"
    The samples call `lib.util.logging.setup_run("./test_runs", name=...)`, which creates `./test_runs/<timestamp>_<name>/` with a log file, images, and saved domains.

??? question "How do I cite PICT?"
    See [Citation](../index.md#citation).

??? question "Under which license is PICT released?"
    The code is released under the Apache License 2.0. The paper is open access under CC BY 4.0.
