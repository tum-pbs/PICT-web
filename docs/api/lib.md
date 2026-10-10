# Helper library (`lib`)

The `lib/` package in the repository root contains the Python tooling around the solver. It is imported from the repository root, e.g. `import lib.data.shapes as shapes`.

## Mesh generation: `lib.data.shapes`

| Function | Description |
|---|---|
| `make_wall_refined_ortho_grid(res_x, res_y, corner_lower=(0,0), corner_upper=(1,1), wall_refinement=[], base=1.05, dtype=torch.float32)` | 2D orthogonal grid with exponential refinement towards the listed walls |
| `extrude_grid_z(grid, res_z, start_z=0, end_z=1, weights_z=None, exp_base=1.05)` | extrude a 2D grid to 3D |
| `make_torus_2D(res, r1, r2, start_angle, angle, offset=None, dtype=torch.float32)` | ring segment with approximately square cells |
| `generate_grid_vertices_2D(res, corner_vertices, border_vertices=None, x_weights=None, y_weights=None, dtype=torch.float32)` | four-sided block interpolated from its corners and optional border curves |
| `interpolate_vertices_from_borders_2D(borders, x_weights=None, y_weights=None)` | the underlying interpolation from four borders |
| `make_weights_linear(res)`, `make_weights_exp(res, base, refinement)`, `make_weights_exp_global(...)`, `make_weights_cos(res, refinement)` | 1D vertex distributions; `refinement` is `"START"`, `"END"` or `"BOTH"` |
| `rotate_grid(grid, angle, axis=None, center="CENTER", distance_scaling=None)` | rotate or swirl a grid (e.g. for robustness tests) |
| `extrapolate_boundary_layers(grid, boundaries)` | add extrapolated cell layers at boundaries |

## Resampling: `lib.data.resample`

These functions convert between curvilinear block data and uniform grids, e.g. to compare with reference data or to feed networks that need a regular grid:

| Function | Description |
|---|---|
| `sample_coords_to_uniform_grid(data, coords, out_shape, is_cell_coords=False, transform_uniform="AABB_OUTER", fill_max_steps=0)` | scatter one block onto a uniform grid covering its bounding box |
| `sample_multi_coords_to_uniform_grid(data_list, coords_list, out_shape, ...)` | the same for several blocks into one grid |
| `sample_coords_from_uniform_grid(data, coords, is_cell_coords=False, transform_uniform="AABB_OUTER", boundary_mode="CLAMP")` | gather uniform-grid data onto a block |
| `sample_multi_coords_from_uniform_grid(data, coords_list, ...)` | the same for several blocks |
| `sample_transform_to_uniform_grid(...)`, `sample_transform_from_uniform_grid(...)` | variants based on transformation metrics instead of coordinates |

## Neural network layers: `lib.modules.multiblock_cnn`

| Class | Description |
|---|---|
| `MultiblockConv(domain, fixed_boundary_padding, in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None)` | 1D/2D/3D convolution over a list of block tensors, padded with data from connected blocks; `kernel_size` 3, 5 or 7; fixed boundaries are zero-padded (`fixed_boundary_padding="ZERO"`) |
| `MultiblockConv_naive(dims, *conv_args, **conv_kwargs)` | per-block convolution without cross-block padding |
| `MultiblockReLU()` | ReLU over a list of tensors |
| `AxesMapping` | helper describing axis permutations and flips between connected blocks |

See [Training Neural Networks](../guide/learning.md#multi-block-convolutions).

## Turbulence statistics: `lib.data.online_statistics`

Streaming statistics, written in PyTorch so that they can be differentiated:

| Class | Accumulates |
|---|---|
| `WelfordOnlineParallel_Torch(dims, ...)` | mean, variance and standard deviation over the given dimensions (Welford's algorithm) |
| `CovarianceOnlineParallel_Torch(dims, ...)` | covariances (e.g. Reynolds shear stress) |
| `MultivariateMomentsOnlineParallel_Torch(moments, spatial_dims=3, avg_dims=[0,2])` | higher-order (co-)moments, e.g. skewness and flatness |
| `TurbulentEnergyBudgetsOnlineParallel_Torch(avg_dims=[0,2], grid_coordinates=None, with_forcing=False, u_wall=None)` | terms of the turbulent kinetic energy budget |
| `PSDOnline_Torch(...)` | power spectral densities |
| `TemporalTwoPointCorrelation_Online_torch(dims, ...)` | temporal two-point correlation |

## Channel-flow tools

| Module | Contents |
|---|---|
| `lib.data.TCF_tools` | unit conversions for channel flows: `Re_wall_to_cl`, `Re_cl_to_wall`, `ETT_to_t`, `t_to_ETT`, `t_to_t_wall`, `t_wall_to_t`, `vel_to_vel_wall`, `pos_to_pos_wall`; statistics containers `PISOTCFProfile`, `VelocityStats` |
| `lib.data.torroja` | readers for the Torroja DNS data: `TorrojaProfile(base_path, Re)`, `TorrojaBalances(base_path, Re)`, `TorrojaSpectra(base_path, Re)` |
| `lib.data.OpenFOAM_profile` | `OpenFOAMProfile`, a reader for OpenFOAM channel statistics |

## Input/output: `lib.util`

| Function / class | Description |
|---|---|
| `domain_io.save_domain(domain, path)` | save structure (`path.json`) and tensors (`path.npz`) |
| `domain_io.load_domain(path, dtype=None, device=None, with_scalar=True)` | load a saved domain; call `PrepareSolve()` afterwards |
| `output.plot_grids(grids, color="tab:blue", path=None, name="grid", type="svg", linewidth=1.0)` | plot block grids |
| `output.save_domain_images(domain, out_dir, it, ...)` | write velocity (`v`), pressure (`p`), scalar (`d`) and source (`vs`) images; `-r` variants are resampled |
| `output.ttonp(tensor)`, `output.ntonp(x)` | tensor / number to NumPy |
| `outputVtk.save_vtk(domain, out_dir, it, vertex_coord_list)` | VTK multi-block output via `pyvista` |
| `logging.setup_run(base_dir, name="TEST")` | create `<base_dir>/<timestamp>_<name>/` with a log file; returns the path |
| `logging.get_logger(name)`, `logging.close_logging()` | logging helpers |
| `GPU_info.get_available_GPU_id(active_mem_threshold=0.05, default=None)` | index of the least-used GPU, for `CUDA_VISIBLE_DEVICES` |
| `memory_usage.MemoryUsage(logger)` | `check_memory(name)` / `print_max_memory()` for GPU memory tracking |
| `profiling.SAMPLE(name)` | context manager for profiling code sections |
