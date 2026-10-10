# Installation

PICT consists of three parts:

| Part | Location | How you get it |
|---|---|---|
| `PISOtorch`, the compiled C++/CUDA core (data structures, PISO operators, linear solvers, backward kernels) | `extensions/` | compiled and installed into your environment with `setup.py` |
| `PISOtorch_simulation` (simulation loop) and `PISOtorch_diff` (autograd wrappers) | repository root | plain Python modules, imported from the repository root |
| `lib/` (grid generation, I/O, output, multi-block CNN layers, turbulence statistics) | `lib/` | plain Python package, imported from the repository root |

Only the CUDA extension is installed. Run your scripts from the repository root, or add the repository root to your `PYTHONPATH`.

!!! info "Requirements"
    - Linux and an NVIDIA GPU. PICT runs on a **single GPU**; all fields live in GPU memory.
    - A PyTorch build together with a **matching** CUDA toolkit (`nvcc`) and a compatible `gcc`. The conda environment below provides all of them.

## Setup with conda

The PICT solver and the CUDA toolkit are installed via conda, PyTorch via pip.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and clone the repository:

    ```bash
    git clone https://github.com/tum-pbs/PICT.git
    cd PICT
    ```

2. Create the environment from the `environment.txt` in the repository and activate it:

    ```bash
    conda create --name pisotorch --file environment.txt
    conda activate pisotorch
    python --version   # should be a recent Python 3.10
    ```

3. Install PyTorch for CUDA 12.8 and the matching CUDA toolkit:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    conda install cuda-toolkit=12.8 -c nvidia/label/cuda-12.8.1
    ```

4. Compile and install the custom CUDA kernels:

    ```bash
    cd extensions
    python setup.py install
    cd ..
    ```

5. Check the installation (from the repository root):

    ```python
    import torch
    import PISOtorch             # compiled core module
    import PISOtorch_simulation  # simulation loop, imported from the repository root
    print(torch.cuda.is_available(), PISOtorch.Domain)
    ```

### Tested versions

PICT uses Python 3.10.x by default. To use another Python version, change it in `environment.txt` before creating the environment. PICT works with PyTorch on either CUDA 11 or CUDA 12. These combinations have been tested:

| PyTorch | CUDA |
|---|---|
| 1.13 | 11.8 |
| 2.6 | 11.8 |
| 2.7 | 11.8 |
| 2.7 | 12.8 |

For example, to use PyTorch 2.6 with CUDA 11.8, replace step 3 with:

```bash
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
conda install cuda-toolkit=11.8 -c nvidia/label/cuda-11.8.0
```

## Testing

```bash
python run_tests.py
```

This runs the tests of the compiled kernels: short PISO simulations on a set of test domains in forward and differentiable mode, `gradcheck` tests of the backward passes of the PISO operators, small optimizations through the solver, and domain copy/clone/save/load. The tests are extensive and take about **30 minutes**. `run_tests.py` sets `os.environ["CUDA_VISIBLE_DEVICES"]="0"` at the top of the file; change it to an available GPU on your machine (use `nvidia-smi` to check GPU usage).

!!! note
    As of April 2025, around 5 tests fail, but most of them should work.

The physical validation cases (plane Poiseuille flow and lid-driven cavities compared against reference data) are run with:

```bash
python run_validations.py
```

See [Validation](background/validation.md) for the results.

## Optional components

**Simplex noise kernels.** The turbulent channel flow example initializes its velocity with a Reichardt profile perturbed by curl noise, and the noise is computed by a second small extension:

```bash
cd extensions/noise
python setup_noise.py install
```

This installs the `SimplexNoiseVariations` module: GPU noise fields in 2D and 3D (single precision), with simplex, Worley, fBm, ridged, gradient and curl variants.

**VTK output.** `Simulation(log_vtk=True)` writes VTK files through [`pyvista`](https://pyvista.org/), which is already part of `environment.txt`.

**Turbulence reference statistics.** The channel-flow learning example needs the DNS statistics from the [Torroja database](https://torroja.dmt.upm.es/channels/data/), see [Turbulent Channel Flow SGS Model](examples/turbulent_channel_flow.md).

## Build issues

- **The wrong CUDA toolkit is picked up.** PyTorch's extension builder takes the CUDA installation from the `CUDA_HOME` environment variable if it is set. If it finds a system-wide CUDA instead of the toolkit in your environment, the build fails with errors like `No such file or directory: '/usr/lib/cuda/bin/nvcc'`. Point it to the environment explicitly:

    ```bash
    export CUDA_HOME=$CONDA_PREFIX
    ```

- **PyTorch 1.13.** With PyTorch 1.13, importing the module can fail with `ImportError: arg(): could not convert default argument into a Python object`. PyTorch 1.13 binds `c10::optional` but not its `nullopt` default. Enable the small `type_caster` block near the top of `extensions/PISOtorch.cpp` (it is commented out and marked `needed for torch 1.13`), then rebuild.

## Troubleshooting

The environment setup should contain all tools and libraries in compatible versions, including the `nvcc` and `gcc` compilers. Versions matter and have to be compatible:

- the torch packages are built for a specific Python version,
- the torch version needs a specific CUDA version,
- the CUDA version dictates the `nvcc` and CUDA library versions,
- the linear solver implementations need specific CUDA library versions,
- `nvcc` limits the supported `gcc` versions.

In the default setup this means torch 2.7, CUDA (runtime, `nvcc`, libraries) 12.8, and gcc 6.0 to 11.5. If you update torch to a newer CUDA version, the CUDA version(s) might have to be updated accordingly, which in turn might require updating the solver to use newer library functions.

??? question "GCC: the wrong compiler is used"
    The conda-installed gcc does not supersede the system gcc. It has a different name like `x86_64-conda-linux-gnu-cc`, which should be set in the `CC` environment variable. It should be used automatically when running `setup.py`.

??? question "Sysconfig: `_sysconfigdata_x86_64_conda_cos7_linux_gnu.py` is missing"
    With some older Python versions, the `_sysconfigdata_x86_64_conda_cos7_linux_gnu.py` file needed to build the CUDA kernels was not generated. This might be caused by a Python 3.9.x version that is too old.

    For the issues observed so far (`cos7` required but only `cos6` available), a workaround is to copy the file and replace `cos6` with `cos7` in the filename and in the file. The file is in `$CONDA_PREFIX/lib/python3.x`, with `$CONDA_PREFIX` being something like `~/miniconda3/envs/pisotorch`:

    ```bash
    cd ~/miniconda3/envs/pisotorch/lib/python3.9/
    cp _sysconfigdata_x86_64_conda_cos6_linux_gnu.py _sysconfigdata_x86_64_conda_cos7_linux_gnu.py
    ```

??? question "Dynamic linker: a library is not found"
    The dynamic linker might not find required libraries (or specific versions of them), which shows up as a file-not-found error.

    1. Make sure that the linker finds the libraries in the environment:

        ```bash
        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}
        ```

    2. If the issue persists, the required library is missing from the environment.

    To automate step 1, add the following to the file `env_vars.sh` in `$CONDA_PREFIX/etc/conda/activate.d` (create it if missing):

    ```bash
    export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}
    ```

    and to the file of the same name in `$CONDA_PREFIX/etc/conda/deactivate.d`:

    ```bash
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
    unset OLD_LD_LIBRARY_PATH
    ```
