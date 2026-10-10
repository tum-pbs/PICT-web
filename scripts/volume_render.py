"""Volume renders of 3D fields with vedo for the videos of render_flow_examples.py, after ReDiT's equidit/run_volume.py.

The render runs in a child process with a timeout: on a broken offscreen GL stack VTK calls abort(), which takes the calling
process with it, and on some it returns blank frames, so every image is checked for content. The child needs only numpy and
vedo (no PISOtorch or torch), so it can run in another Python environment: set VEDO_PYTHON to its interpreter (default: the
current one). VTK renders offscreen through EGL, no X display is needed.

    paths = render_frames(frame_dir, scene, {"volume": u})   # u: frames, x, y, z on a uniform grid

scene (JSON):
  size, bg              image size in pixels and background color
  spacing, origin       of the uniform grid, the arrays are indexed [frame, x, y, z]
  box                   (x0, x1, y0, y1, z0, z1) drawn as grey edges, with the wall at y0 as a plane
  camera                position, focal_point, view_up, view_angle
  cmap, vmin, vmax      list of colors spread evenly over [vmin, vmax]
  mode "volume"         ray-cast "volume" with the opacity ramp "alpha" ([(value, opacity)]) per "alpha_unit" length, shaded
  mode "isosurface"     surface of "iso" at "iso_value", colored by "color"
"""
import json, os, subprocess, sys, tempfile
import numpy as np

SECONDS_PER_FRAME = 10 # a healthy render of a 65 x 128 x 65 volume takes ca. 0.6 s on the CPU; a bad GL stack hangs
EDGE = "#898781"
WALL = "#c3c2b7"


def render_frames(frame_dir, scene, fields):
    """Render one image per frame to <frame_dir>/f_%05d.png and return their paths."""
    os.makedirs(frame_dir, exist_ok=True)
    n_frames = len(next(iter(fields.values())))
    with tempfile.TemporaryDirectory() as tmp:
        for key, values in fields.items():
            np.save(os.path.join(tmp, key + ".npy"), np.ascontiguousarray(values))
        with open(os.path.join(tmp, "scene.json"), "w") as file:
            json.dump(dict(scene, fields=list(fields)), file)
        env = dict(os.environ)
        env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
        cmd = [os.environ.get("VEDO_PYTHON", sys.executable), os.path.abspath(__file__), tmp, frame_dir]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=60 + SECONDS_PER_FRAME*n_frames, env=env)
        except subprocess.TimeoutExpired:
            raise RuntimeError("volume render: no result after %d s" % (60 + SECONDS_PER_FRAME*n_frames)) from None
    if proc.returncode != 0:
        tail = " | ".join(proc.stderr.decode(errors="replace").strip().splitlines()[-3:])
        raise RuntimeError("volume render: child exited %d%s: %s" % (proc.returncode, " (SIGABRT)" if proc.returncode == -6 else "", tail))
    paths = [os.path.join(frame_dir, "f_%05d.png" % i) for i in range(n_frames)]
    blank = [p for p in paths if _uniform(p)]
    if blank:
        raise RuntimeError("volume render: %d of %d images are missing or blank, e.g. %s" % (len(blank), n_frames, blank[0]))
    return paths


def _uniform(png):
    # True if the image is missing or a single color (what a broken GL stack returns)
    if not os.path.exists(png):
        return True
    import matplotlib.image as mpimg
    return float(np.asarray(mpimg.imread(png))[..., :3].std()) < 1e-3


def _box_actors(vedo, box, eye):
    """The edges of the box behind the field (the three at the corner farthest from the camera) with the wall at y0, and the
    others, which are in front of the field or on its outline: they are drawn on top, as the CPU volume mapper breaks lines
    on the border of the volume into dashes."""
    x0, x1, y0, y1, z0, z1 = box
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    far = int(np.argmax(np.linalg.norm(corners - np.asarray(eye), axis=1)))
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.count_nonzero(corners[a] != corners[b]) == 1]
    def lines(pairs):
        return vedo.Lines([corners[a] for a, _ in pairs], [corners[b] for _, b in pairs]).c(EDGE).lw(1.5).lighting("off")
    wall = vedo.Mesh([[[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [[0, 1, 2, 3]]]).c(WALL).alpha(0.5).lighting("off")
    return [lines([e for e in edges if far in e]), wall], lines([e for e in edges if far not in e])


def _field_actor(vedo, scene, fields, i):
    grid = dict(spacing=scene["spacing"], origin=scene["origin"])
    if scene["mode"] == "volume":
        volume = vedo.Volume(np.asarray(fields["volume"][i], dtype=np.float32), **grid)
        volume.cmap(scene["cmap"], vmin=scene["vmin"], vmax=scene["vmax"])
        volume.alpha(scene["alpha"]).alpha_unit(scene["alpha_unit"])
        volume.shade(True)
        volume.properties.SetAmbient(0.35); volume.properties.SetDiffuse(0.7); volume.properties.SetSpecular(0.15)
        # CPU ray casting with fine samples: the GPU mapper of VTK 9.5 draws wood-grain stripes where opaque structures meet the
        # border of the volume and on surfaces seen at grazing angles, at any sample distance
        from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
        mapper = vtkFixedPointVolumeRayCastMapper()
        mapper.SetInputData(volume.dataset)
        mapper.SetAutoAdjustSampleDistances(False)
        mapper.SetSampleDistance(0.25*min(scene["spacing"]))
        mapper.SetImageSampleDistance(1.0)
        volume.actor.SetMapper(mapper)
        return volume
    iso = vedo.Volume(np.asarray(fields["iso"][i], dtype=np.float32), **grid).isosurface(scene["iso_value"]).compute_normals()
    iso.probe(vedo.Volume(np.asarray(fields["color"][i], dtype=np.float32), **grid)) # the probed values replace "input_scalars"
    iso.cmap(scene["cmap"], "input_scalars", vmin=scene["vmin"], vmax=scene["vmax"])
    return iso.phong().lighting(ambient=0.3, diffuse=0.75, specular=0.1)


def _child(src, out_dir):
    import vedo
    with open(os.path.join(src, "scene.json")) as file:
        scene = json.load(file)
    fields = {key: np.load(os.path.join(src, key + ".npy"), mmap_mode="r") for key in scene["fields"]}
    plotter = vedo.Plotter(size=scene["size"], offscreen=True, bg=scene["bg"])
    cam = scene["camera"]
    behind, on_top = _box_actors(vedo, scene["box"], cam["position"])
    plotter.add(*behind)
    from vtkmodules.vtkRenderingCore import vtkRenderer
    overlay = vtkRenderer()
    overlay.SetLayer(1); overlay.InteractiveOff(); overlay.SetActiveCamera(plotter.camera)
    overlay.AddActor(on_top.actor)
    plotter.window.SetNumberOfLayers(2); plotter.window.AddRenderer(overlay)
    plotter.camera.SetPosition(cam["position"]); plotter.camera.SetFocalPoint(cam["focal_point"])
    plotter.camera.SetViewUp(cam["view_up"]); plotter.camera.SetViewAngle(cam["view_angle"])
    plotter.show(resetcam=False)
    # white ambient light: the CPU volume mapper scales the ambient term by the lights' ambient color, black by default
    lights = plotter.renderer.GetLights()
    lights.InitTraversal()
    for _ in range(lights.GetNumberOfItems()):
        lights.GetNextItem().SetAmbientColor(1, 1, 1)
    actor = None
    for i in range(len(fields[scene["fields"][0]])):
        if actor is not None:
            plotter.remove(actor)
        actor = _field_actor(vedo, scene, fields, i)
        plotter.add(actor)
        plotter.render()
        plotter.screenshot(os.path.join(out_dir, "f_%05d.png" % i))
    plotter.close()


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])
