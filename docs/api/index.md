# API Reference

PICT is used through three Python-facing layers:

| Module | Kind | Use it for |
|---|---|---|
| [`PISOtorch`](pisotorch.md) | compiled C++/CUDA extension (`extensions/`) | data structures (`Domain`, `Block`, boundaries), the individual PISO operators, the sparse linear solvers, grid and resampling utilities |
| [`PISOtorch_simulation`](simulation.md) | Python (repository root) | the `Simulation` class that runs complete PISO time steps, plus boundary helpers |
| [`PISOtorch_diff`](diff.md) | Python (repository root) | autograd-enabled versions of the PISO operators; used automatically by `Simulation(differentiable=True)` |
| [`lib`](lib.md) | Python package (repository root) | mesh generation, I/O, output and plotting, multi-block CNN layers, turbulence statistics, logging |

`PISOtorch` is a compiled CUDA extension without generated API documentation. These pages are written from the binding definitions in `extensions/PISOtorch.cpp` and from the Python sources, and the listed names were checked against a compiled build. The docstrings in the bindings and the comments in the sample scripts are the authoritative low-level references.

## Conventions

- **Tensors** are PyTorch tensors in channels-first layout (`NCDHW`, `NCHW`, `NCW`) with batch size 1. Spatial dimensions are in `zyx` order and vector components in `xyz` order (see [Core Concepts](../guide/concepts.md#tensor-layout)).
- **Devices:** field data must be on the CUDA device of the domain and contiguous. The global viscosity and the time step are small **CPU** tensors of shape `[1]`.
- **Faces** are named `"-x"`, `"+x"`, `"-y"`, `"+y"`, `"-z"`, `"+z"`, or given as indices 0–5 in that order. Axes are `"x"`, `"y"`, `"z"` or 0–2.
- **`PISOtorch.Int4`** holds integer sizes in `xyzw` order, e.g. `block.getSizes()` returns the cell counts as `Int4`.
- Functions and attributes marked *(grad)* belong to the backward pass. They are compiled in by default (`WITH_GRAD` is defined in `extensions/domain_structs_gpu.h`), and `PISOtorch_diff` uses them.
