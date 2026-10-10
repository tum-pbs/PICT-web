"""Render the 3D examples recorded by sim_3d_examples.py as 3D videos with poster images (last frame) and snapshot strips, all with vedo:
  cylinder, airfoil, bfs:  <case>3d_vortices  vortex surfaces (Q-criterion) colored by the span-wise vorticity; for the step inside a translucent
                                              volume of the span-wise vorticity, which shows the laminar shear layer (Q = 0 in pure shear)
                           <case>3d_volume    ray-cast volume of the span-wise vorticity, the quantity of the 2D videos
                           bfs3d_smoke        surface of the passive scalar (smoke jet), colored by the span-wise velocity
                           bfs3d_velocity     ray-cast volume of the stream-wise velocity: reverse flow (recirculation bubbles) and the fast core of the jet
                           bfs3d_streamlines  instantaneous streamlines from the inflow and in the recirculation bubble, colored by stream-wise velocity
                           bfs3d_vortices_streamlines, bfs3d_velocity_streamlines: the vortices and the velocity with dark streamlines in the same scene
                           (the step is rendered from the inflow to x = 28, after its 3D transition)
  rbc:                     rbc3d_plumes       plume surfaces (temperature deviation from the mean at each height) colored by the deviation
                           rbc3d_volume       ray-cast volume of the temperature deviation scaled by its rms at each height, like tcf_volume.mp4
  decaying:                decaying3d_vortices  vortex surfaces (Q relative to the current mean enstrophy) colored by z-vorticity
                           decaying3d_volume    ray-cast volume of the z-vorticity scaled by its current rms
The fields are interpolated to uniform grids in physical space: barycentric in each span-wise layer of the curvilinear multi-block grids,
linear over the wall-refined y-axis of RBC with the wall temperatures. The vortex criteria are computed on these grids.
vedo runs in a child process like render_airfoil3d.py and volume_render.py; it may live in another Python environment, set VEDO_PYTHON to
its interpreter. Set FFMPEG to an ffmpeg with libx264.
usage: python render_3d_examples.py <case> <data dir> <out dir> [--test] [--only name ...] [--iso value] [--frames slice]"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np

EDGE, WALL, SOLID_COLOR = "#898781", "#c3c2b7", "#3a3a3a"


def _child(src, out_dir):
    # runs in the vedo environment: only numpy and vedo
    import vedo
    from vtkmodules.vtkCommonCore import vtkMultiThreader
    vtkMultiThreader.SetGlobalMaximumNumberOfThreads(8) # half of the cores: the CPU ray casting runs next to the GPU simulations
    from vtkmodules.vtkRenderingCore import vtkRenderer
    with open(os.path.join(src, "scene.json")) as file:
        scene = json.load(file)
    fields = {key: np.load(os.path.join(src, key + ".npy"), mmap_mode="r") for key in scene["fields"]}
    plotter = vedo.Plotter(size=scene["size"], offscreen=True, bg=scene["bg"])
    x0, x1, y0, y1, z0, z1 = scene["box"]
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.count_nonzero(corners[a] != corners[b]) == 1]
    def lines(pairs):
        return vedo.Lines([corners[a] for a, _ in pairs], [corners[b] for _, b in pairs]).c(EDGE).lw(1.5).lighting("off")
    cam = scene["camera"]
    # the edges at the corner farthest from the camera are behind the field, the others are drawn on top
    far = int(np.argmax(np.linalg.norm(corners - np.asarray(cam["position"]), axis=1)))
    plotter.add(lines([e for e in edges if far in e]))
    if scene.get("wall") is not None: # wall plane at y = wall
        yw = scene["wall"]
        plotter.add(vedo.Mesh([[[x0, yw, z0], [x1, yw, z0], [x1, yw, z1], [x0, yw, z1]], [[0, 1, 2, 3]]]).c(WALL).alpha(0.5).lighting("off"))
    for profile in scene.get("solids", []):
        # solid extruded over the span, the caps are triangle fans (the profile must be convex)
        p = np.asarray(profile); n = len(p)
        verts = [(x, y, z0) for x, y in p] + [(x, y, z1) for x, y in p] + [(*p.mean(axis=0), z0), (*p.mean(axis=0), z1)]
        faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        faces += [[2*n, (i + 1) % n, i] for i in range(n)] + [[2*n + 1, n + i, n + (i + 1) % n] for i in range(n)]
        plotter.add(vedo.Mesh([verts, faces]).c(SOLID_COLOR).lighting(ambient=0.45, diffuse=0.6, specular=0.05))
    overlay = vtkRenderer()
    overlay.SetLayer(1); overlay.InteractiveOff(); overlay.SetActiveCamera(plotter.camera)
    overlay.AddActor(lines([e for e in edges if far not in e]).actor)
    plotter.window.SetNumberOfLayers(2); plotter.window.AddRenderer(overlay)
    plotter.camera.SetPosition(cam["position"]); plotter.camera.SetFocalPoint(cam["focal_point"])
    plotter.camera.SetViewUp(cam["view_up"]); plotter.camera.SetViewAngle(cam["view_angle"])
    plotter.show(resetcam=False)
    # white ambient light: the CPU volume mapper scales the ambient term by the lights' ambient color, black by default
    lights = plotter.renderer.GetLights()
    lights.InitTraversal()
    for _ in range(lights.GetNumberOfItems()):
        lights.GetNextItem().SetAmbientColor(1, 1, 1)
    grid = dict(spacing=scene["spacing"], origin=scene["origin"])
    # the scene is drawn from layers (e.g. vortex surfaces in a translucent volume), each with its own mode, settings and fields (role -> file key)
    layers = [dict(scene, **layer) for layer in scene["layers"]] if "layers" in scene else [dict(scene, fields={key: key for key in scene["fields"]})]
    actors = []
    for i in range(scene["n_frames"]):
        for actor in actors:
            plotter.remove(actor)
        actors = [make_actor(layer, i, fields, grid) for layer in layers]
        for actor in actors:
            plotter.add(actor)
        plotter.render()
        plotter.screenshot(os.path.join(out_dir, "f_%05d.png" % i))
    plotter.close()


def make_actor(scene, i, fields, grid):
    # one layer of frame i (runs in the vedo environment)
    import vedo
    field = lambda role: np.asarray(fields[scene["fields"][role]][i], dtype=np.float32)
    if scene["mode"] == "streamlines":
        # instantaneous streamlines through the seed points, traced in both directions with the maximum length of their group, drawn as tubes
        # colored by the stream-wise velocity or in line_color. The image data has the origin and spacing of the volumes and surfaces.
        from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
        from vtkmodules.vtkFiltersCore import vtkTubeFilter, vtkAppendPolyData
        from vtkmodules.util.numpy_support import numpy_to_vtk
        vel = field("vel") # 3, x, y, z
        image = vtkImageData()
        image.SetDimensions(*vel.shape[1:]); image.SetSpacing(*scene["spacing"]); image.SetOrigin(*scene["origin"])
        vectors = numpy_to_vtk(np.ascontiguousarray(np.stack([c.ravel(order="F") for c in vel], axis=1)), deep=True) # x fastest
        vectors.SetName("velocity")
        image.GetPointData().SetVectors(vectors)
        append = vtkAppendPolyData()
        for group in scene["seed_groups"]:
            seeds = vtkPoints()
            for p in group["points"]:
                seeds.InsertNextPoint(*p)
            seed_data = vtkPolyData(); seed_data.SetPoints(seeds)
            tracer = vtkStreamTracer()
            tracer.SetInputData(image); tracer.SetSourceData(seed_data)
            tracer.SetIntegratorTypeToRungeKutta45(); tracer.SetIntegrationDirectionToBoth()
            tracer.SetInitialIntegrationStep(0.2); tracer.SetMaximumIntegrationStep(0.5) # in cell lengths
            tracer.SetMaximumPropagation(group["max_length"]); tracer.SetMaximumNumberOfSteps(20000); tracer.SetTerminalSpeed(1e-4)
            tracer.Update()
            append.AddInputData(tracer.GetOutput())
        append.Update()
        tubes = vtkTubeFilter()
        tubes.SetInputConnection(append.GetOutputPort()); tubes.SetRadius(scene["tube_radius"]); tubes.SetNumberOfSides(10); tubes.CappingOn()
        tubes.Update()
        actor = vedo.Mesh(tubes.GetOutput())
        if scene.get("line_color"):
            actor.c(scene["line_color"])
        else:
            actor.pointdata["u"] = vedo.utils.vtk2numpy(actor.dataset.GetPointData().GetArray("velocity"))[:, 0]
            actor.cmap(scene["cmap"], "u", vmin=scene["vmin"], vmax=scene["vmax"])
        actor.phong().lighting(ambient=0.35, diffuse=0.7, specular=0.1)
    elif scene["mode"] == "volume":
        # ray-cast volume as in volume_render.py: CPU ray casting with fine samples, the GPU mapper of VTK 9.5 draws wood-grain stripes
        actor = vedo.Volume(field("volume"), **grid)
        actor.cmap(scene["cmap"], vmin=scene["vmin"], vmax=scene["vmax"])
        # the range of the ramp is given: otherwise vedo adds ramp end points at the data range, which makes the volume opaque in frames
        # whose values lie within the ramp (e.g. the quiet initial state)
        actor.alpha(scene["alpha"], vmin=scene["vmin"], vmax=scene["vmax"]).alpha_unit(scene["alpha_unit"])
        actor.shade(True)
        actor.properties.SetAmbient(0.35); actor.properties.SetDiffuse(0.7); actor.properties.SetSpecular(0.15)
        from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
        mapper = vtkFixedPointVolumeRayCastMapper()
        mapper.SetInputData(actor.dataset)
        mapper.SetAutoAdjustSampleDistances(False)
        mapper.SetSampleDistance(0.25*min(scene["spacing"]))
        mapper.SetImageSampleDistance(1.0)
        actor.actor.SetMapper(mapper)
    else:
        actor = vedo.Volume(field("iso"), **grid).isosurface(scene["iso_value"]).compute_normals()
        actor.probe(vedo.Volume(field("color"), **grid)) # the probed values replace "input_scalars"
        actor.cmap(scene["cmap"], "input_scalars", vmin=scene["vmin"], vmax=scene["vmax"])
        actor.phong().lighting(ambient=0.3, diffuse=0.75, specular=0.1)
    return actor


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    _child(sys.argv[2], sys.argv[3])
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.path import Path
import vizlib as V
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("case", choices=["cylinder", "airfoil", "bfs", "rbc", "decaying"]); parser.add_argument("src"); parser.add_argument("out")
parser.add_argument("--test", action="store_true", help="render one frame only, as <name>_test.png")
parser.add_argument("--only", nargs="+", default=None, help="render only these, e.g. rbc3d_plumes")
parser.add_argument("--iso", type=float, default=None, help="iso value of the 3D surfaces, replaces the case's default")
parser.add_argument("--frames", default=None, help="frames to render as a python slice, e.g. '100:' (default: per case)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
CASE = args.case
d = np.load(os.path.join(args.src, "%s3d_frames.npz" % CASE))
TIME_STEP, TIME_OFFSET = float(d["time_step"]), float(d["time_offset"])
LIGHT_BG = "#f2f1ee" # neutral midpoint of the diverging map
STRIP_BG = "#05121c"
SOLID = dict(facecolor=SOLID_COLOR, edgecolor="#1e1e1e", lw=1.2, zorder=5)
CMAP_HEX = [to_hex(c) for c in V.PICT_DIV(np.linspace(0, 1, 256))]

def nice(value):
    # round to 2 significant digits
    e = 10**np.floor(np.log10(value))
    return float(np.round(value/e, 1)*e)

def wanted(name):
    return args.only is None or name in args.only

def encode(tmp, name, n, fps):
    if args.test:
        shutil.copy(os.path.join(tmp, "f_%05d.png" % (n - 1)), os.path.join(args.out, name + "_test.png"))
    else:
        V.encode_mp4(os.path.join(tmp, "f_%05d.png"), os.path.join(args.out, name + ".mp4"), fps=fps, crf=24)
        shutil.copy(os.path.join(tmp, "f_%05d.png" % (n - 1)), os.path.join(args.out, name + "_poster.png"))
    shutil.rmtree(tmp)

def save_strip(name, images, times, extent=None, cmap=None, vmin=None, vmax=None, panel_w=3.6):
    # snapshots side by side, in the style of the strips of render_flow_examples.py; images are fields (with extent) or rendered RGB images
    aspect = (extent[3] - extent[2])/(extent[1] - extent[0]) if extent is not None else images[0].shape[0]/images[0].shape[1]
    fig_h = panel_w*aspect + 0.35
    fig, axs = plt.subplots(1, len(images), figsize=(panel_w*len(images), fig_h), dpi=150)
    fig.patch.set_facecolor(STRIP_BG)
    for ax, image, t in zip(axs, images, times):
        if extent is not None:
            ax.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_aspect("equal")
        else:
            ax.imshow(image)
        ax.axis("off")
        ax.set_title("t = %g" % t, color="#c3c2b7", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=1 - 0.354/fig_h, wspace=0.05)
    fig.savefig(os.path.join(args.out, name + "_strip.png"), facecolor=fig.get_facecolor())
    plt.close(fig)

def camera(focal, dist, el, az, angle=19):
    el, az = np.radians(el), np.radians(az)
    return dict(position=tuple(float(c) for c in np.array(focal) + dist*np.array([np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)])),
        focal_point=tuple(float(c) for c in focal), view_up=(0, 1, 0), view_angle=angle)

def render_surfaces(name, scene, iso, color, frames, fps, strip_frames=4):
    # isosurfaces of iso, colored by color. iso, color: frames, x, y, z on the grid of the scene
    render_scene(name, dict(scene, mode="isosurface"), dict(iso=iso, color=color), frames, fps, strip_frames)

def render_volume(name, scene, volume, frames, fps, strip_frames=4):
    # ray-cast volume, scene: cmap (default PICT_DIV), vmin, vmax, alpha: [(value, opacity)] per alpha_unit length. volume: frames, x, y, z
    render_scene(name, dict(scene, mode="volume"), dict(volume=volume), frames, fps, strip_frames)

def render_scene(name, scene, fields, frames, fps, strip_frames):
    # fields: key -> frames, x, y, z array (3, x, y, z per frame for velocities).
    # scene: one layer (mode and its settings), or "layers": a list of such dicts with "fields": role -> key, drawn together
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "src"); out = os.path.join(tmp, "frames"); os.makedirs(src); os.makedirs(out)
    for key, values in fields.items():
        np.save(os.path.join(src, key + ".npy"), np.ascontiguousarray(values, dtype=np.float16))
    n_frames = len(next(iter(fields.values())))
    with open(os.path.join(src, "scene.json"), "w") as file:
        json.dump(dict(dict(cmap=CMAP_HEX), **scene, fields=list(fields), n_frames=n_frames, bg=LIGHT_BG), file)
    env = dict(os.environ); env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    proc = subprocess.run([os.environ.get("VEDO_PYTHON", sys.executable), os.path.abspath(__file__), "--child", src, out], capture_output=True, env=env,
        timeout=300 + 60*n_frames)
    if proc.returncode != 0:
        raise RuntimeError("vedo child exited %d: %s" % (proc.returncode, proc.stderr.decode(errors="replace")[-2000:]))
    images = [os.path.join(out, "f_%05d.png" % n) for n in range(n_frames)]
    blank = [p for p in images if not os.path.exists(p) or plt.imread(p)[..., :3].std() < 1e-3] # a broken GL stack returns blank images
    if blank:
        raise RuntimeError("vedo: %d of %d images are missing or blank" % (len(blank), len(images)))
    if not args.test:
        snaps = np.linspace(0, len(images) - 1, strip_frames).round().astype(int)
        save_strip(name, [plt.imread(images[s]) for s in snaps], [TIME_OFFSET + frames[s]*TIME_STEP for s in snaps], panel_w=4.2)
    encode(out, name, len(images), fps)
    shutil.rmtree(tmp, ignore_errors=True)
    describe = lambda layer: "%s %s, color range [%.3g, %.3g]" % (layer["mode"], {"isosurface": lambda: "iso %.3g" % layer["iso_value"],
        "volume": lambda: "opacity %s" % layer["alpha"],
        "streamlines": lambda: "%d seeds" % sum(len(g["points"]) for g in layer["seed_groups"])}[layer["mode"]](), layer["vmin"], layer["vmax"])
    print("rendered %s: %d frames, %s" % (name, len(images), "; ".join(describe(dict(scene, **layer)) for layer in scene.get("layers", [{}]))),
        flush=True)

def select(n_frames, default):
    # frames of the videos; the color ranges are set from all of them, also in test mode
    return np.arange(n_frames)[eval("np.s_[%s]" % (args.frames or default))]

def rendered(frames):
    # frames to render: the last one only in test mode
    return frames[-1:] if args.test else frames

def periodic_to_vertices(v, axes):
    # cell values -> values on the vertices along periodic axes (mean of the two adjacent cells, the first plane repeated at the end)
    for axis in axes:
        v = 0.5*(v + np.roll(v, 1, axis=axis))
        v = np.concatenate([v, v.take([0], axis=axis)], axis=axis)
    return v


def segment_distance(px, py, polygon):
    # distance of the points (px, py) to the closed polygon (N, 2)
    a = polygon; b = np.roll(polygon, -1, axis=0); ab = b - a
    t = np.clip(((px[..., None] - a[:, 0])*ab[:, 0] + (py[..., None] - a[:, 1])*ab[:, 1])/np.maximum(np.sum(ab*ab, axis=1), 1e-12), 0, 1)
    return np.min(np.hypot(px[..., None] - (a[:, 0] + t*ab[:, 0]), py[..., None] - (a[:, 1] + t*ab[:, 1])), axis=-1)

# 3D setups extruded from a 2D multi-block grid with a periodic span: xl, yl: the channel; fps: as the 2D videos; H: spacing of the grid of the
# vortex criteria (about the span-wise cell size or finer); iso: Q of the vortex surfaces (U/D = 1 for the cylinder, U/c = 0.3 for the airfoil,
# U/h = 1 for the step; below 0.3 the surfaces follow the interpolation noise in the shear layer, where Q is about 0); volume: |span-wise vorticity| where the volume becomes visible and opaque, relative to the color range;
# volume_wall: distances from the lower and upper channel wall within which the vorticity is left out of the volume (their boundary layers);
# view: x-range of the renders, None for the whole channel. For the step the upper wall's boundary layer is left out up to 0.4: in the inflow
# channel it is an opaque sheet above the shear layer that separates at the step, and the view ends after the 3D transition (x = 15 - 27).
EXTRUDED = dict(
    cylinder=dict(xl=(-2.0, 20.0), yl=(-2.0, 2.1), fps=25, H=0.05, iso=1.0, volume=(0.1, 0.16), volume_wall=(0.35, 0.35), view=None, size=(2000, 900),
        camera=dict(focal=(8.0, -0.6, 0.0), dist=30.0, el=30, az=-32)),
    airfoil=dict(xl=(-1.5, 4.5), yl=(-0.7, 0.7), fps=15, H=0.02, iso=0.18, volume=(0.1, 0.16), volume_wall=(0.1, 0.1), view=None, size=(2000, 900),
        camera=dict(focal=(1.5, -0.15, 0.0), dist=9.4, el=30, az=-32)),
    bfs=dict(xl=(0.0, 40.0), yl=(0.0, 2.0), fps=25, H=0.05, iso=0.3, volume=(0.25, 0.4), volume_wall=(0.1, 0.4), view=(0.0, 28.0), size=(2400, 760),
        camera=dict(focal=(13.5, 0.6, 0.0), dist=31.0, el=14, az=-30), alpha_unit=0.6))

if CASE in EXTRUDED:
    cfg = EXTRUDED[CASE]
    FPS, xl, yl = cfg["fps"], cfg["xl"], cfg["yl"]
    if CASE == "cylinder":
        solid = np.array([(0.5*np.cos(a), 0.5*np.sin(a)) for a in np.linspace(0, 2*np.pi, 96, endpoint=False)])
    elif CASE == "airfoil":
        solid = d["airfoil"].T.astype(np.float64) # N, 2
    else:
        solid = np.array([(0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (0.0, 1.0)]) # the step
    in_solid = lambda x, y: Path(solid).contains_points(np.stack([x, y], axis=-1))
    w_cells = d["w"] # read once: every access to a member of the npz file decompresses all frames
    frames = select(w_cells.shape[0], ":")
    w_lim = nice(np.percentile(np.abs(w_cells[frames]), 99.5)) # color range of the span-wise vorticity, as the 2D videos
    frames = rendered(frames)
    cx, cy = d["cx"], d["cy"]
    bx, by = d["boundary_points"]; bcells, bwall = d["boundary_cells"], d["boundary_wall"]
    _, unique = np.unique(np.round(np.stack([bx, by], 1), 6), axis=0, return_index=True) # corner vertices are shared between blocks
    bx, by, bcells, bwall = bx[unique], by[unique], bcells[unique], bwall[unique]
    # The velocity of the cells is interpolated to a uniform grid: barycentric in each span-wise layer of the curvilinear 2D grid, with the
    # boundary velocity on the walls and the solid (0) and the adjacent cells at the in- and outflow, then linearly along the periodic span.
    # Q and the span-wise vorticity follow from central differences on that grid: per-block finite differences show the block junctions,
    # and PICT's resampling fills the large span-wise cells with constant values, which leaves steps on the surfaces.
    # The boundary layers are too thin for the grid: the differences across the walls give large values of both signs, and for the cylinder
    # a small kink of the velocity where the blocks around it meet the wake block at the channel walls (x = 2) shows as a thin rod.
    # Q is left out within two grid spacings of the solid and three of the channel walls.
    span, res_z, H = float(d["span"]), int(d["res_z"]), cfg["H"]
    nx, ny, nz = int(round((xl[1] - xl[0])/H)) + 1, int(round((yl[1] - yl[0])/H)) + 1, int(round(span/H))
    resampler = V.BarycentricResampler(np.concatenate([cx, bx]), np.concatenate([cy, by]), xl, yl, nx, ny, mask_fn=in_solid)
    source = np.concatenate([np.arange(len(cx)), bcells])
    wall_scale = np.concatenate([np.ones(len(cx)), ~bwall]).astype(np.float32) # wall velocity 0
    pos = np.arange(nz)*H/(span/res_z) - 0.5 # grid planes in units of the span-wise cells, whose centers are at k + 0.5
    k0 = np.floor(pos).astype(int); fz = (pos - k0).astype(np.float32)
    gx, gy = np.meshgrid(np.linspace(*xl, nx), np.linspace(*yl, ny), indexing="ij")
    channel_wall = (gy < yl[0] + 3*H) | (gy > yl[1] - 3*H)
    thick_channel_wall = (gy < yl[0] + cfg["volume_wall"][0]) | (gy > yl[1] - cfg["volume_wall"][1])
    near_solid = (segment_distance(gx, gy, solid) < 2*H) | in_solid(gx.ravel(), gy.ravel()).reshape(gx.shape)
    near_wall = near_solid | channel_wall
    def load(key, i):
        if key in d.files:
            return d[key][i]
        return np.load(os.path.join(args.src, key, "%05d.npy" % i)) # written per frame by the recorder
    def to_grid(cell_values, scale):
        # cell_values: channels*layers, cells -> channels, x, y, z
        v = cell_values.astype(np.float32).T[source]*scale[:, None] # points, channel*layer
        out = np.zeros((resampler.valid.size, v.shape[1]), np.float32) # grid points in the solid keep 0 (the wall velocity)
        out[resampler.valid] = np.einsum("pk,pkc->pc", resampler.bary, v[resampler.verts])
        out = out.reshape(ny, nx, -1, res_z)
        out = out[..., k0 % res_z]*(1 - fz) + out[..., (k0 + 1) % res_z]*fz # periodic span
        return out.transpose(2, 1, 0, 3)
    close = lambda f: np.concatenate([f, f[..., :1]], axis=-1).astype(np.float16) # the first span-wise plane again at the end of the span
    need = set() # fields required by the requested renders
    bfs_wanted = lambda *names: CASE == "bfs" and any(wanted(name) for name in names)
    if wanted(CASE + "3d_vortices") or bfs_wanted("bfs3d_vortices_streamlines"): need |= {"q", "wz"}
    if cfg["volume"] is not None and wanted(CASE + "3d_volume"): need |= {"wz_volume"}
    if bfs_wanted("bfs3d_vortices", "bfs3d_vortices_streamlines"): need |= {"wz_volume"} # the shear layer around the vortex surfaces
    if bfs_wanted("bfs3d_smoke"): need |= {"s", "w"}
    if bfs_wanted("bfs3d_streamlines", "bfs3d_vortices_streamlines", "bfs3d_velocity_streamlines"): need |= {"vel"}
    if bfs_wanted("bfs3d_velocity", "bfs3d_velocity_streamlines"): need |= {"u"}
    vx = cfg["view"] or xl
    i0, i1 = int(round((vx[0] - xl[0])/H)), int(round((vx[1] - xl[0])/H)) + 1 # grid points of the view in x
    def fields(i):
        u = to_grid(load("vel", i).reshape(3*res_z, -1), wall_scale) # component, x, y, z
        out = {}
        if "vel" in need: # periodic span closed, cropped to the view
            out["vel"] = np.concatenate([u, u[..., :1]], axis=-1)[:, i0:i1].astype(np.float16)
        if "u" in need: # stream-wise velocity
            out["u"] = close(u[0])
        if not need & {"q", "wz", "wz_volume", "w", "s"}:
            return out
        G = np.empty((3, 3) + u.shape[1:], np.float32) # G[i, j] = du_i/dx_j
        for c in range(3):
            G[c, 0] = np.gradient(u[c], H, axis=0); G[c, 1] = np.gradient(u[c], H, axis=1)
            G[c, 2] = (np.roll(u[c], -1, axis=2) - np.roll(u[c], 1, axis=2))/(2*H)
        S = 0.5*(G + G.transpose(1, 0, 2, 3, 4)); R = 0.5*(G - G.transpose(1, 0, 2, 3, 4))
        q = 0.5*(np.sum(R*R, axis=(0, 1)) - np.sum(S*S, axis=(0, 1)))
        q[near_wall] = -1.0
        wz = G[1, 0] - G[0, 1]
        out.update(q=close(q), wz=close(wz))
        # for the volume, the boundary layers are left out: on the solid it is thinner than the grid spacing like for Q, at the channel walls
        # it thickens where the flow passes the solid. The vorticity is smoothed over about a grid spacing: sheets only a few ray samples thick
        # are drawn with stripes by the CPU mapper.
        wz[thick_channel_wall | near_solid] = 0
        out["wz_volume"] = close(gaussian_filter(wz, sigma=1.0, mode=("nearest", "nearest", "wrap")))
        if CASE == "bfs": # passive scalar, zero gradient at the walls: the boundary points take the adjacent cells
            out["s"] = close(to_grid(load("s", i), np.ones_like(wall_scale))[0])
            out["w"] = close(u[2]) # span-wise velocity
        return out
    data = [fields(i) for i in frames]
    stacks = {}
    def stack(key):
        # frames, x, y, z (component first for "vel") cropped to the view. Built once and shared by the renders, the per-frame data is released.
        if key not in stacks:
            stacks[key] = np.stack([data[n].pop(key) if key == "vel" else data[n].pop(key)[i0:i1] for n in range(len(data))])
        return stacks[key]
    base = dict(size=cfg["size"], spacing=[H]*3, origin=[xl[0] + i0*H, yl[0], -0.5*span], solids=[solid.tolist()],
        box=(vx[0], vx[1], yl[0], yl[1], -0.5*span, 0.5*span), camera=camera(**cfg["camera"], angle=19))
    if CASE == "bfs":
        # The renders of the step are made from layers drawn together in one scene: all fields of a frame come from the same velocity on the same
        # grid (origin, spacing), so surfaces, volumes and streamlines line up.
        # Q is zero in a pure shear flow: the laminar shear layer that separates at the step has no Q > 0 until it rolls up (x ~ 15), and
        # the surfaces alone leave the first half of the channel empty. The shear layer is drawn as a translucent volume of the span-wise
        # vorticity (as bfs3d_volume, lighter) around the vortex surfaces; its colors span 1.4 times its opacity range, the dark ends of the
        # map make the thick inflow layer nearly black.
        lo, hi = cfg["volume"]
        ramp = [(-w_lim, 0.5), (-hi*w_lim, 0.35), (-lo*w_lim, 0.0), (lo*w_lim, 0.0), (hi*w_lim, 0.35), (w_lim, 0.5)]
        vortex_layers = [dict(mode="volume", fields=dict(volume="wz_volume"), alpha=ramp, alpha_unit=cfg["alpha_unit"], vmin=-1.4*w_lim, vmax=1.4*w_lim),
            dict(mode="isosurface", fields=dict(iso="q", color="wz"), iso_value=args.iso or cfg["iso"], vmin=-w_lim, vmax=w_lim)]
        vortex_fields = lambda: dict(q=stack("q"), wz=stack("wz"), wz_volume=stack("wz_volume"))
        # The stream-wise velocity is shown on a two-slope scale: the reverse flow in the recirculation bubbles (-0.03 at the step corner, about
        # -0.2 at its center) would be white next to the jet (up to 1.5) on a symmetric scale. Negative values are divided by U_NEG, positive
        # ones by U_POS. The colors span [-1.4, 1.4]: the dark ends of the map would turn the jet and the bubble nearly black.
        # The volume shows the reverse flow (blue, from u < -0.016) and the fast core of the jet (red, from u > 0.9). It is smoothed over about
        # a grid spacing like the vorticity volume; the thin reverse-flow layer of the upper wall (0.1, a few cells) is left out, the CPU mapper
        # draws it as a flat grey bar.
        U_NEG, U_POS, U_COLOR = 0.2, 1.5, 1.4
        ramp = [(-1.0, 0.95), (-0.3, 0.7), (-0.08, 0.0), (0.6, 0.0), (0.85, 0.3), (1.0, 0.5)]
        velocity_layer = dict(mode="volume", fields=dict(volume="u_volume"), alpha=ramp, alpha_unit=cfg["alpha_unit"], vmin=-U_COLOR, vmax=U_COLOR)
        def velocity_fields():
            if "u_volume" not in stacks:
                u = gaussian_filter(stack("u").astype(np.float32), sigma=(0, 1.0, 1.0, 1.0), mode=("nearest", "nearest", "nearest", "wrap"))
                u[:, :, np.linspace(*yl, len(gy[0])) > yl[1] - 0.1] = 0
                stacks["u_volume"] = np.clip(np.where(u < 0, u/U_NEG, u/U_POS), -1, 1).astype(np.float16)
                del stacks["u"]
            return dict(u_volume=stacks["u_volume"])
        # Instantaneous streamlines over the fields: from a rake across the inflow over the recirculation bubble, and through the bubble,
        # drawn in one dark color so that the colors stay those of the field. The streamlines of the bubble spiral; they are cut at 22 units
        # in each direction (one or two turns), otherwise the turns pile up into coils.
        zs = np.array([-1.25, 0.15, 1.4]) # not on the grid planes, the periodic border or the front and back faces of the box
        streamline_layer = dict(mode="streamlines", fields=dict(vel="vel"), tube_radius=0.02, line_color="#2b2b2b",
            seed_groups=[dict(points=[(0.25, y, z) for y in (1.2, 1.4, 1.6, 1.8) for z in zs], max_length=60.0),
                dict(points=[(x, 0.3, z) for x in (7.0, 11.0, 15.0) for z in zs], max_length=22.0)])
        # the wall plane half a grid spacing below the volumes: where they coincide, the CPU volume mapper ends the rays at random pixels (dots)
        wall = yl[0] - 0.5*H
        # --- vortex surfaces colored by the span-wise vorticity, in the shear layer ---
        if wanted("bfs3d_vortices"):
            render_scene("bfs3d_vortices", dict(base, layers=vortex_layers, vmin=-w_lim, vmax=w_lim, wall=wall), vortex_fields(), frames, FPS, 4)
        if wanted("bfs3d_vortices_streamlines"):
            render_scene("bfs3d_vortices_streamlines", dict(base, layers=vortex_layers + [streamline_layer], vmin=-w_lim, vmax=w_lim, wall=wall),
                dict(vortex_fields(), vel=stack("vel")), frames, FPS, 4)
        # --- stream-wise velocity ---
        if wanted("bfs3d_velocity"):
            render_scene("bfs3d_velocity", dict(base, layers=[velocity_layer], vmin=-U_COLOR, vmax=U_COLOR, wall=wall), velocity_fields(), frames, FPS, 4)
        if wanted("bfs3d_velocity_streamlines"):
            render_scene("bfs3d_velocity_streamlines", dict(base, layers=[velocity_layer, streamline_layer], vmin=-U_COLOR, vmax=U_COLOR, wall=wall),
                dict(velocity_fields(), vel=stack("vel")), frames, FPS, 4)
    elif wanted(CASE + "3d_vortices"): # --- vortex surfaces colored by the span-wise vorticity ---
        render_surfaces(CASE + "3d_vortices", dict(base, iso_value=args.iso or cfg["iso"], vmin=-w_lim, vmax=w_lim, wall=yl[0]), stack("q"), stack("wz"),
            frames, FPS)
    # --- volume of the span-wise vorticity, the quantity of the 2D videos: transparent below cfg["volume"][0]*w_lim, nearly opaque from [1]*w_lim ---
    if cfg["volume"] is not None and wanted(CASE + "3d_volume"):
        lo, hi = cfg["volume"]
        ramp = [(-w_lim, 1.0), (-hi*w_lim, 0.85), (-lo*w_lim, 0.0), (lo*w_lim, 0.0), (hi*w_lim, 0.85), (w_lim, 1.0)]
        # the wall plane half a grid spacing below the volume: where they coincide, the CPU volume mapper ends the rays at random pixels (dots)
        render_volume(CASE + "3d_volume", dict(base, vmin=-w_lim, vmax=w_lim, alpha=ramp, alpha_unit=cfg.get("alpha_unit", 4*H), wall=yl[0] - 0.5*H),
            stack("wz_volume"), frames, FPS)
    # --- passive scalar (smoke) of the step, like bfs_smoke.mp4: the surface at 40% of the inflow maximum, the interface of the smoke jet,
    # colored by the span-wise velocity (white where the flow is two-dimensional) ---
    if bfs_wanted("bfs3d_smoke"):
        s_lim = nice(np.percentile(stack("s")[len(data)//2:].astype(np.float32), 99.9))
        w_span = nice(np.percentile(np.abs(stack("w")[len(data)//2:].astype(np.float32)), 99.5))
        render_surfaces("bfs3d_smoke", dict(base, iso_value=0.4*s_lim, vmin=-w_span, vmax=w_span, wall=yl[0]), stack("s"), stack("w"), frames, FPS)
    # --- streamlines of the step: from the inflow over the recirculation bubble, and inside the bubble behind the step, colored by the
    # stream-wise velocity (blue: reverse flow) ---
    if bfs_wanted("bfs3d_streamlines"):
        zs = np.linspace(-0.5*span, 0.5*span, 7)[1:-1] + 0.15 # not on grid planes or the periodic border
        inflow = [(0.25, y, z) for y in (1.2, 1.4, 1.6, 1.8) for z in zs]
        bubble = [(x, 0.3, z) for x in (7.0, 10.0, 13.0, 16.0, 19.0, 22.0) for z in zs]
        u_lim = 0.6 # the reverse flow in the bubble is ca. -0.2, the jet saturates
        render_scene("bfs3d_streamlines", dict(base, mode="streamlines", seed_groups=[dict(points=inflow + bubble, max_length=60.0)], tube_radius=0.03,
            vmin=-u_lim, vmax=u_lim, wall=yl[0]), dict(vel=stack("vel")), frames, FPS, strip_frames=4)
    del data, stacks

elif CASE == "rbc":
    FPS = 25 # one frame per 0.5 free-fall times
    T = d["T"] # frames, z, y, x
    frames = select(T.shape[0], ":")
    length = float(d["length"]); cy = d["cy"]; y_walls = d["y_walls"]
    res_z, res_y, res_x = T.shape[1:]
    def y_resampling(n):
        # the wall-refined y-direction is resampled to n uniform points from wall to wall, with the wall temperatures (1 at the bottom, 0 at the top)
        ys = np.linspace(y_walls[0], y_walls[1], n)
        y_nodes = np.concatenate([[y_walls[0]], cy, [y_walls[1]]])
        idx = np.clip(np.searchsorted(y_nodes, ys) - 1, 0, len(y_nodes) - 2)
        return idx, ((ys - y_nodes[idx])/(y_nodes[idx + 1] - y_nodes[idx])).astype(np.float32)
    def with_walls(t, axis):
        shape = list(t.shape); shape[axis] = 1
        return np.concatenate([np.ones(shape, t.dtype), t, np.zeros(shape, t.dtype)], axis=axis)
    ny = 129 # finer than the wall-normal cells of the bulk: the plume roots are thin sheets at the walls
    idx, wy = y_resampling(ny); wy = wy[None, :, None]
    def to_grid(t):
        # z, y, x cells -> x, y, z grid points: periodic x and z on the vertices, y uniform with the wall temperatures
        t = with_walls(t.astype(np.float32), 1)
        t = t[:, idx]*(1 - wy) + t[:, idx + 1]*wy
        return periodic_to_vertices(t, (0, 2)).transpose(2, 1, 0)
    # deviation from the mean temperature at each height (removes the thermal boundary layers, which are uniform layers at the walls)
    deviation = lambda t: t - t.mean(axis=(0, 2), keepdims=True)
    developed = np.stack([deviation(to_grid(T[i])) for i in frames[len(frames)//2::10]])
    theta_lim = nice(np.percentile(np.abs(developed), 99.5))
    theta_rms = np.sqrt(np.mean(developed**2, axis=(0, 1, 3))) # at each height, over the developed half of the run
    theta_rms = 0.5*(theta_rms + theta_rms[::-1]) # symmetric in the two halves
    del developed
    frames = rendered(frames)
    theta = np.stack([deviation(to_grid(T[i])) for i in frames]).astype(np.float16)
    spacing = [length/res_x, (y_walls[1] - y_walls[0])/(ny - 1), length/res_z]
    base = dict(size=(1600, 1100), spacing=spacing, origin=[0.0, float(y_walls[0]), 0.0],
        box=(0.0, length, float(y_walls[0]), float(y_walls[1]), 0.0, length), camera=camera((0.5*length, -0.1, 0.5*length), 8.6, 24, -30))
    # --- plume surfaces ---
    if wanted("rbc3d_plumes"):
        render_surfaces("rbc3d_plumes", dict(base, iso_value=args.iso or 0.15, vmin=-theta_lim, vmax=theta_lim, wall=float(y_walls[0])),
            np.abs(theta), theta, frames, FPS, strip_frames=5)
    # --- temperature volume, like the TCF volume: deviation scaled by its rms at each height, transparent below 1 rms, nearly opaque from 1.15 rms ---
    if wanted("rbc3d_volume"):
        scaled = (theta.astype(np.float32)/np.maximum(theta_rms, 1e-6)[None, None, :, None]).astype(np.float16)
        scaled[:, :, [0, -1]] = 0 # the walls have the wall temperature, no deviation
        ramp = [(-3.5, 1.0), (-1.15, 0.85), (-1.0, 0.0), (1.0, 0.0), (1.15, 0.85), (3.5, 1.0)]
        render_volume("rbc3d_volume", dict(base, vmin=-3.5, vmax=3.5, alpha=ramp, alpha_unit=0.03, wall=float(y_walls[0]) - 0.5*spacing[1]),
            scaled, frames, FPS, strip_frames=5)

elif CASE == "decaying":
    FPS = 20 # as the 2D decaying video
    wz, q = d["wz"], d["q"] # frames, z, y, x
    frames = select(wz.shape[0], ":")
    size = float(d["domain_size"])
    res = wz.shape[-1]
    w_lim = nice(np.percentile(np.abs(wz[frames]), 99.5)) # fixed for all frames, so that the decay stays visible
    # --- z-vorticity volume, the quantity of the 2D decaying video: scaled by its current rms (the vorticity decays by more than an order of
    # magnitude), transparent below 2 rms and nearly opaque from 2.3 rms ---
    if wanted("decaying3d_volume"):
        to_grid = lambda v: periodic_to_vertices(v.astype(np.float32), (0, 1, 2)).transpose(2, 1, 0) # z, y, x cells -> x, y, z vertices
        vframes = rendered(frames)
        h = size/res
        def scaled(i):
            w = wz[i].astype(np.float32)
            return to_grid(w/np.sqrt(np.mean(w**2))).astype(np.float16)
        ramp = [(-5.0, 1.0), (-2.3, 0.85), (-2.0, 0.0), (2.0, 0.0), (2.3, 0.85), (5.0, 1.0)]
        scene = dict(size=(1400, 1200), spacing=[h]*3, origin=[0.0]*3, vmin=-5.0, vmax=5.0, alpha=ramp, alpha_unit=0.1,
            box=(0.0, size, 0.0, size, 0.0, size), wall=None, camera=camera((0.5*size,)*3, 27.0, 24, -32))
        render_volume("decaying3d_volume", scene, np.stack([scaled(i) for i in vframes]), vframes, FPS, strip_frames=5)
    # --- vortex surfaces ---
    if wanted("decaying3d_vortices"):
        # the surfaces are at a fixed Q relative to the current mean enstrophy (<|vorticity|^2>/2), which decays by more than two orders
        # of magnitude: at a fixed Q the box would be empty after a few time units. The decay shows in the coarsening vortices and in their
        # colors, which use a fixed range.
        to_grid = lambda v: periodic_to_vertices(v.astype(np.float32), (0, 1, 2)).transpose(2, 1, 0) # z, y, x cells -> x, y, z vertices
        enstrophy = d["enstrophy"]
        frames = rendered(frames)
        h = size/res
        scene = dict(size=(1400, 1200), spacing=[h]*3, origin=[0.0]*3, iso_value=args.iso or 1.0, vmin=-w_lim, vmax=w_lim,
            box=(0.0, size, 0.0, size, 0.0, size), wall=None, camera=camera((0.5*size,)*3, 27.0, 24, -32))
        render_surfaces("decaying3d_vortices", scene, np.stack([to_grid(q[i])/enstrophy[i] for i in frames]), np.stack([to_grid(wz[i]) for i in frames]),
            frames, FPS, strip_frames=5)
