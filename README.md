# PICT website

Source of the documentation and project website of [PICT](https://github.com/tum-pbs/PICT), the differentiable, GPU-accelerated multi-block PISO solver. The site is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), with the same setup as the [APEBench documentation](https://tum-pbs.github.io/apebench/).

## Build and preview

```bash
pip install -r requirements.txt
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # static site in ./site, fails on broken links
```

## Deploy

`.github/workflows/docs.yml` builds the site with `--strict` and publishes it to GitHub Pages on every push to `main`. To use it, enable *Settings → Pages → Source: GitHub Actions* in the repository. Alternatively, run `mkdocs gh-deploy` to push the built site to a `gh-pages` branch. Adjust `site_url` in `mkdocs.yml` if the site is not served from `https://tum-pbs.github.io/PICT/`.

## Structure

```text
mkdocs.yml                 site configuration and navigation
docs/
  index.md                 landing page (Getting Started)
  installation.md
  guide/                   user guide: concepts, grids, boundaries, simulation, physics, differentiability, learning
  background/              numerical method, validation, results from the paper
  examples/                walkthroughs of the sample scripts
  api/                     API reference (written from the pybind11 bindings and Python sources)
  misc/                    FAQ, tips, gotchas
  imgs/                    figures; imgs/paper/ holds figures from the paper (CC BY 4.0)
  assets/videos/           simulation videos
  stylesheets/extra.css    PICT colors and page components
  javascripts/mathjax.js   MathJax configuration
scripts/                   scripts that produced the simulation visuals and charts (see below)
```

## Figures and videos

- `docs/imgs/paper/` contains figures from Franz et al., *PICT – A Differentiable, GPU-Accelerated Multi-Block PISO Solver for Simulation-Coupled Learning Tasks in Fluid Dynamics*, Journal of Computational Physics 544 (2026) 114433, published open access under CC BY 4.0. The pages credit them with "(Franz et al. 2026, Fig. N)", where the figure numbers follow the arXiv version of the paper (arXiv:2505.16992v2; appendix figures are numbered B.15–C.26 there).
- `docs/imgs/PISO_graph.png`, `PISOtorch_overview.png` and `obstacle-flow_vis.jpg` (and its crop `airfoil_flow.jpg`) are taken from the PICT repository. The icons `pict_icon.png`, `pict_icon_transparent.webp` and `favicon.png` are crops of the PICT logo.
- The vortex-street and buoyancy videos, the multi-block grid figures, the cavity validation and streamline plots, and the learning curves were produced with the PICT solver itself, using the scripts in `scripts/`. To regenerate them, build PICT, then run the scripts with the PICT repository on the `PYTHONPATH`:

    ```bash
    export PYTHONPATH=/path/to/PICT:$PWD/scripts
    python scripts/sim_vortex.py out/vortex 320 160 48 && python scripts/render_vortex.py out/vortex out/vortex_render
    python scripts/viz_buoyancy.py out/buoyancy
    python scripts/sim_cavity.py out/cavity && python scripts/plot_cavity.py out/cavity out/cavity
    python scripts/sim_learning.py out/learning && python scripts/plot_learning.py out/learning out/learning
    python scripts/viz_grids.py out/grids
    python scripts/viz_airfoil_grid.py out/airfoil
    ```

- `scripts/check_cavity_tolerance.py` reproduces the numbers in the "Solver tolerances on fine grids" note on the Validation page (it only prints results).
- The videos, posters and snapshot strips of the benchmark-flow examples (`docs/assets/videos/<case>_*.mp4`, `docs/imgs/flows/`) come from runs of the newer sample scripts with their setup functions and solver settings. `sim_flow_examples.py` records the fields of every iteration, and the render scripts turn them into videos. The videos are encoded with the `ffmpeg` given by the `FFMPEG` variable, which must support libx264. The 3D renders use [vedo](https://vedo.embl.es/), which may live in another Python environment: set `VEDO_PYTHON` to its interpreter.

    ```bash
    # 2D cases: bfs karman decaying kolmogorov cylinder airfoil rbc (iterations as on the example pages)
    python scripts/sim_flow_examples.py bfs out/bfs 1000 && python scripts/render_flow_examples.py bfs out/bfs out/bfs
    # 3D channel flow: run turbulent_channel_flow_sample.py first, then continue from its final domain (t = 400)
    python scripts/sim_flow_examples.py tcf out/tcf 400 <run_dir>/0000_channel_flow_Re180_small/domain 400
    python scripts/render_flow_examples.py tcf out/tcf out/tcf && python scripts/render_tcf3d.py out/tcf/tcf_frames.npz out/tcf --only tcf_volume
    # 3D airfoil: the 3D run of airfoil_channel_sample.py, starting from the final domain of its 2D run (t = 60)
    python scripts/sim_airfoil3d.py out/airfoil3d --sample <run_dir>/0000_airfoil_channel_Re3000_aoa10/domain 60
    python scripts/render_airfoil3d.py out/airfoil3d out/airfoil3d
    # 3D runs of the samples, run by sim_3d_examples.py through the sample functions: <iterations> <time step> [<final domain of the 2D run>]
    # bfs and cylinder start from their 2D runs: the default bfs run and the commented cylinder run at Re = 250 (both t = 100)
    python scripts/sim_3d_examples.py bfs out/bfs3d 500 0.2 <run_dir>/0000_BFS_Re1400_r8/domain
    python scripts/render_3d_examples.py bfs out/bfs3d out/bfs3d --only bfs3d_vortices_streamlines
    python scripts/sim_3d_examples.py cylinder out/cylinder3d 400 0.2 <run_dir>/<n>_cylinder_channel_Re250_r32/domain
    python scripts/render_3d_examples.py cylinder out/cylinder3d out/cylinder3d --only cylinder3d_vortices cylinder3d_volume
    python scripts/sim_3d_examples.py rbc out/rbc3d 600 0.5 && python scripts/render_3d_examples.py rbc out/rbc3d out/rbc3d
    python scripts/sim_3d_examples.py decaying out/decaying3d 200 0.1 && python scripts/render_3d_examples.py decaying out/decaying3d out/decaying3d
    # block-colored grid figures of the multi-block samples
    python scripts/viz_example_grids.py out/grids
    ```

    The iterations of the videos: bfs 1000, karman 800, decaying 200, kolmogorov 400, cylinder 1000, airfoil 240, rbc 600 (output every 0.25 instead of 1.0), tcf 400 (output every 0.25); in 3D: bfs 500, cylinder 400, rbc 600 (output every 0.5 instead of 1.0), decaying 200. The PNG posters, strips and grids were converted to WebP for the site. The 3D videos larger than 6 MB (bfs3d, cylinder3d_vortices, rbc3d, decaying3d_vortices) were re-encoded for the site with libx264 at crf 27, scaled to 1600 (bfs, cylinder) or 1200 (rbc, decaying) pixels wide, and their posters scaled to match.
- `scripts/check_excerpts.py <PICT repository>` checks that every code excerpt with a file title and start line (```` ```python title="x.py" linenums="N" ````) is a verbatim copy of those lines of the solver code. Re-run it after the solver code changes and update the line numbers it reports.
