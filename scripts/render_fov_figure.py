#!/usr/bin/env python3
"""Offscreen render of the Rhody 2 URDF with every sensor field of view drawn.

Produces a multi-panel figure (hero perspective + three orthographic views +
legend) suitable for a thesis or a slide, plus a printed table of pairwise
coverage overlap.

The point of the figure is that nothing in it is hand-drawn. The vehicle mesh,
the sensor poses and the FOV solids all come from urdf/rhody2.urdf.xacro
expanded with fov:=true, walked through the same fixed-joint chain that
robot_state_publisher puts on /tf. If the calibration in the URDF is wrong, the
picture is wrong in the same way -- which is what makes it evidence rather than
illustration.

Requires: vtk, numpy, matplotlib, yaml, and `xacro` on PATH
(source /opt/ros/humble/setup.bash).

Usage:
    python3 scripts/render_fov_figure.py
    python3 scripts/render_fov_figure.py --out docs/figures/rhody2_sensor_fov.png
    python3 scripts/render_fov_figure.py --panel hero      # single view, no montage
"""

import argparse
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import vtk
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (DEFAULT_DPI, DEFAULT_WIDTH_IN, PT_CAPTION,  # noqa: E402
                          PT_KEY_LABEL, PT_KEY_STATS, PT_PANEL_TITLE,
                          PT_SCALE_LABEL, wrap_to_width)

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# URDF -> forward kinematics
# --------------------------------------------------------------------------

def rpy_to_matrix(rpy):
    """URDF fixed-axis convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def pose_of(element, default=np.eye(4)):
    """Read a URDF <origin> child into a 4x4 homogeneous transform."""
    origin = element.find('origin') if element is not None else None
    if origin is None:
        return default.copy()
    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
    rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def expand_xacro(path, fov=True):
    xacro = 'xacro'
    if subprocess.call(['which', xacro], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL) != 0:
        distro = os.environ.get('ROS_DISTRO', 'humble')
        xacro = f'/opt/ros/{distro}/bin/xacro'
        if not os.path.exists(xacro):
            sys.exit('xacro not found; source /opt/ros/<distro>/setup.bash first')
    cmd = [xacro, path] + ([f'fov:=true'] if fov else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f'xacro failed:\n{out.stderr}')
    return out.stdout


def link_transforms(urdf_xml):
    """Resolve every link's pose in base_link. All joints here are fixed."""
    root = ET.fromstring(urdf_xml)
    joints = {}
    for j in root.findall('joint'):
        child = j.find('child').get('link')
        joints[child] = (j.find('parent').get('link'), pose_of(j))

    links = {l.get('name'): l for l in root.findall('link')}
    cache = {}

    def resolve(name):
        if name in cache:
            return cache[name]
        if name not in joints:                 # root
            cache[name] = np.eye(4)
            return cache[name]
        parent, T = joints[name]
        cache[name] = resolve(parent) @ T
        return cache[name]

    return {n: resolve(n) for n in links}, links


def resolve_package_uri(uri):
    if uri.startswith('package://rhody/'):
        return os.path.join(PKG_DIR, uri[len('package://rhody/'):])
    return uri


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------

def vtk_matrix(T):
    m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            m.SetElement(i, j, T[i, j])
    return m


def load_vehicle(glb_path):
    """Vehicle hull as a single polydata, already in base_link coordinates.

    NOTE: do NOT apply the -90 deg X rotation that rhody2.urdf.xacro puts on the
    base_link visual. That rotation exists because RViz loads the Collada
    conversion without the glTF root node transform; vtkGLTFReader *does* apply
    the node hierarchy, so the Y-up -> Z-up flip is already baked in here.
    Cross-check that it landed right: the mesh top sits at z = 0.312 m, and the
    URDF puts usbl_link -- which is bolted to the top plate -- at z = 0.3098 m.
    """
    reader = vtk.vtkGLTFReader()
    reader.SetFileName(glb_path)
    reader.ApplyDeformationsToGeometryOn()
    reader.Update()

    append = vtk.vtkAppendPolyData()
    it = reader.GetOutput().NewIterator()
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        block = it.GetCurrentDataObject()
        if block and block.IsA('vtkPolyData') and block.GetNumberOfPoints():
            append.AddInputData(block)
        it.GoToNextItem()
    append.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(append.GetOutputPort())
    normals.SetFeatureAngle(45.0)
    normals.Update()
    return normals.GetOutput()


def surface_actor(polydata, T, rgb, alpha):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetUserMatrix(vtk_matrix(T))
    prop = actor.GetProperty()
    prop.SetColor(*rgb)
    prop.SetOpacity(alpha)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.65)
    prop.SetSpecular(0.0)
    return actor


def edge_actor(polydata, T, rgb, width=1.4):
    """Silhouette/boundary lines. Translucent volumes alone read as mush in
    print; the outlines are what make the coverage boundaries legible."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polydata)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOn()
    edges.SetFeatureAngle(25.0)
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.ColoringOff()
    edges.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(edges.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetUserMatrix(vtk_matrix(T))
    prop = actor.GetProperty()
    prop.SetColor(*[c * 0.75 for c in rgb])
    prop.SetLineWidth(width)
    prop.SetLighting(False)
    prop.SetOpacity(0.9)
    return actor


def triad_actors(T, length=0.35, shaft=0.012):
    """RGB body-frame triad at base_link: x forward, y port, z up (REP-103)."""
    actors = []
    axes = [((1, 0, 0), (0.85, 0.15, 0.15)),
            ((0, 1, 0), (0.15, 0.65, 0.20)),
            ((0, 0, 1), (0.20, 0.35, 0.90))]
    for direction, color in axes:
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(24)
        arrow.SetShaftResolution(24)
        arrow.SetShaftRadius(shaft / length)
        arrow.SetTipRadius(3.0 * shaft / length)
        arrow.SetTipLength(0.22)

        # vtkArrowSource points along +x; rotate it onto the requested axis.
        d = np.array(direction, dtype=float)
        x = np.array([1.0, 0.0, 0.0])
        v = np.cross(x, d)
        R = np.eye(3)
        if np.linalg.norm(v) > 1e-9:
            c = float(np.dot(x, d))
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx / (1.0 + c)
        A = np.eye(4)
        A[:3, :3] = R * length
        actors.append(surface_actor(arrow.GetOutput() if arrow.Update() is None
                                    else arrow.GetOutput(), T @ A, color, 1.0))
        actors[-1].GetProperty().SetAmbient(0.4)
        actors[-1].GetProperty().SetDiffuse(0.6)
    return actors


def build_renderer(cfg, transforms, links, show_triad=True):
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.SetUseDepthPeeling(True)
    ren.SetMaximumNumberOfPeels(100)
    ren.SetOcclusionRatio(0.0)
    ren.SetUseFXAA(True)

    # Vehicle.
    hull = load_vehicle(os.path.join(PKG_DIR, 'meshes', 'rhody2.glb'))
    hull_actor = surface_actor(hull, np.eye(4), (0.42, 0.45, 0.50), 1.0)
    hull_actor.GetProperty().SetAmbient(0.25)
    hull_actor.GetProperty().SetDiffuse(0.75)
    hull_actor.GetProperty().SetSpecular(0.25)
    hull_actor.GetProperty().SetSpecularPower(30)
    ren.AddActor(hull_actor)

    # Map each FOV link back to the sensor whose colour it should take.
    colours = {name: (spec['color'], spec['alpha'])
               for name, spec in cfg['sensors'].items()}
    link_to_sensor = {
        'voyis_left_fov': 'voyis_stereo',
        'voyis_right_fov': 'voyis_stereo',
        'fls_fov': 'gemini_fls',
        'waterlinked_fov': 'waterlinked_3d',
        'dvl_fov': 'nucleus_dvl',
    }

    readers = {}
    for link_name, sensor in link_to_sensor.items():
        if link_name not in links:
            continue
        rgb, alpha = colours[sensor]
        T_link = transforms[link_name]
        # One link can carry several visuals (the DVL's 3 beams + altimeter).
        for visual in links[link_name].findall('visual'):
            mesh = visual.find('geometry/mesh')
            path = resolve_package_uri(mesh.get('filename'))
            if path not in readers:
                r = vtk.vtkSTLReader()
                r.SetFileName(path)
                r.Update()
                readers[path] = r.GetOutput()
            T = T_link @ pose_of(visual)
            ren.AddActor(surface_actor(readers[path], T, rgb, alpha))
            ren.AddActor(edge_actor(readers[path], T, rgb))

    if show_triad:
        for a in triad_actors(np.eye(4)):
            ren.AddActor(a)

    return ren


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------
# focal point sits forward of base_link so the coverage cone, not the hull,
# is centred in frame.
FOCUS = (0.75, 0.0, 0.0)

# The hero camera looks from aft-starboard, i.e. across the vehicle from the
# side the Gemini is *not* on. That is the one family of angles where the FLS
# fan and the 3D-sonar wedge separate visually instead of stacking; from the
# port side or dead ahead the orange disappears inside the purple and the
# overlap the figure exists to show becomes invisible.
# Titles are kept short on purpose: at print size the bottom row is only about
# 2 in wide per panel, and anything longer than ~20 characters runs into its
# neighbour. What each view is showing belongs in the caption, not the title.
VIEWS = {
    'hero':      dict(pos=(-2.2, -3.0, 1.9), up=(0, 0, 1), parallel=False,
                      title='Perspective — aft, starboard quarter'),
    'top':       dict(pos=(0.75, 0.0, 6.0), up=(1, 0, 0), parallel=True,
                      title='Plan — azimuth'),
    'starboard': dict(pos=(0.75, -6.0, 0.0), up=(0, 0, 1), parallel=True,
                      title='Starboard — elevation'),
    'bow':       dict(pos=(6.0, 0.0, 0.0), up=(0, 0, 1), parallel=True,
                      title='Bow-on — DVL beams'),
}


def crop_to_content(arr, margin=14, tol=6):
    """Trim the white border VTK leaves around the scene.

    Framing four different cameras by hand is fiddly and breaks the moment the
    display range changes; cropping to the rendered content is scale-invariant
    and keeps the panels tight whatever the geometry does.
    """
    ink = np.any(arr < (255 - tol), axis=2)
    if not ink.any():
        return arr
    rows, cols = np.where(ink)
    r0 = max(rows.min() - margin, 0)
    r1 = min(rows.max() + margin + 1, arr.shape[0])
    c0 = max(cols.min() - margin, 0)
    c1 = min(cols.max() + margin + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def render_view(ren, view, size):
    """Render one view. Returns (RGB array, pixels-per-metre or None).

    The pixel scale is exact only under parallel projection, where
    ParallelScale is the viewport half-height in world units; perspective
    panels get None and no scale bar.
    """
    cam = vtk.vtkCamera()
    cam.SetPosition(*view['pos'])
    cam.SetFocalPoint(*FOCUS)
    cam.SetViewUp(*view['up'])
    cam.SetParallelProjection(view['parallel'])
    ren.SetActiveCamera(cam)
    ren.ResetCamera()
    if view['parallel']:
        cam.SetParallelScale(cam.GetParallelScale() * 0.92)
    else:
        cam.Zoom(1.35)

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetAlphaBitPlanes(1)
    win.SetMultiSamples(0)          # required for depth peeling
    win.AddRenderer(ren)
    win.SetSize(*size)

    kit = vtk.vtkLightKit()
    kit.SetKeyLightIntensity(1.05)
    kit.SetKeyToFillRatio(2.2)
    kit.SetKeyToHeadRatio(3.0)
    ren.RemoveAllLights()
    kit.AddLightsToRenderer(ren)

    win.Render()
    if not ren.GetLastRenderingUsedDepthPeeling():
        print('  warning: depth peeling unavailable; transparency order may be '
              'wrong on overlapping volumes', file=sys.stderr)

    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.SetInputBufferTypeToRGB()
    grab.ReadFrontBufferOff()
    grab.Update()

    img = grab.GetOutput()
    w, h, _ = img.GetDimensions()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    arr = vtk_to_numpy(img.GetPointData().GetScalars())
    arr = arr.reshape(h, w, -1)[::-1]          # VTK origin is bottom-left

    px_per_m = h / (2.0 * cam.GetParallelScale()) if view['parallel'] else None
    win.Finalize()
    return crop_to_content(arr), px_per_m


# --------------------------------------------------------------------------
# Coverage overlap, computed analytically rather than from the meshes
# --------------------------------------------------------------------------

def in_fov(points_base, T_sensor, spec, r_max):
    """Boolean mask: which base_link points fall inside this sensor's cone.

    The test has to match the solid the mesh generator built, and the two
    sensor families are genuinely different shapes:
      frustum -- a pyramid, bounded by four planes and a *planar* far clip
      fan     -- a spherical sector, bounded in az/el and by a *range* sphere
    Testing a pyramid with spherical az/el bounds (or clipping it to a sphere)
    silently shrinks it, which is exactly the kind of error that would make the
    overlap percentages quietly wrong.
    """
    R = T_sensor[:3, :3]
    t = T_sensor[:3, 3]
    local = (points_base - t) @ R                # world -> sensor frame
    x, y, z = local[:, 0], local[:, 1], local[:, 2]

    if spec['type'] == 'frustum':
        tan_az = np.tan(np.radians(spec['az_fov_deg']) / 2.0)
        tan_el = np.tan(np.radians(spec['el_fov_deg']) / 2.0)
        return ((x > 0) & (x <= r_max)
                & (np.abs(y) <= x * tan_az)
                & (np.abs(z) <= x * tan_el))

    rng = np.linalg.norm(local, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arcsin(np.clip(z / np.maximum(rng, 1e-12), -1.0, 1.0)))
    return ((rng > 0) & (rng <= r_max)
            & (np.abs(az) <= spec['az_fov_deg'] / 2.0)
            & (np.abs(el) <= spec['el_fov_deg'] / 2.0))


FOV_LINKS = {
    'voyis_stereo': ['voyis_left_fov', 'voyis_right_fov'],
    'gemini_fls': ['fls_fov'],
    'waterlinked_3d': ['waterlinked_fov'],
}


def overlap_table(cfg, transforms, n=2000000, seed=0):
    """Monte-Carlo pairwise coverage overlap out to the display range.

    Uses the analytic containment test, not the triangulated meshes, so the
    numbers do not inherit the tessellation error of the STLs.
    """
    r = float(cfg['display_range_m'])

    # Sample box must enclose every cone, and the sensors are mounted forward
    # of base_link -- a box centred on the origin would clip the far end of the
    # frustums and under-report every volume.
    origins = [transforms[ln][:3, 3] for names in FOV_LINKS.values()
               for ln in names if ln in transforms]
    lo = np.min(origins, axis=0) - r
    hi = np.max(origins, axis=0) + r

    pts = np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))
    cell = float(np.prod(hi - lo)) / n

    masks, volumes = {}, {}
    for sensor, link_names in FOV_LINKS.items():
        spec = cfg['sensors'][sensor]
        per_cam = [in_fov(pts, transforms[ln], spec, r) for ln in link_names
                   if ln in transforms]
        if not per_cam:
            continue
        # Only the intersection of the two frustums yields depth; the sonars
        # each have a single aperture, so their links simply union.
        reduce = np.logical_and if sensor == 'voyis_stereo' else np.logical_or
        masks[sensor] = reduce.reduce(per_cam)
        volumes[sensor] = masks[sensor].sum() * cell

    return masks, volumes, cell


# --------------------------------------------------------------------------

def draw_scale_bar(ax, arr, px_per_m, metres=1.0):
    """A 1 m rule, drawn in image-pixel coordinates so it stays exact.

    Without it the figure is ambiguous: a 73 deg frustum truncated at 2 m looks
    identical to one truncated at 20 m, and the reader has no way to judge how
    much water the vehicle actually sees.
    """
    if not px_per_m:
        return
    h, w = arr.shape[:2]
    length = metres * px_per_m
    if length > 0.8 * w:                        # would not fit; halve it
        metres, length = metres / 2.0, length / 2.0
    x0 = w - length - 0.045 * w
    y = h - 0.055 * h
    ax.plot([x0, x0 + length], [y, y], color='#222222', lw=1.9,
            solid_capstyle='butt', clip_on=False)
    for x in (x0, x0 + length):
        ax.plot([x, x], [y - 0.014 * h, y + 0.014 * h], color='#222222',
                lw=1.9, clip_on=False)
    ax.text(x0 + length / 2.0, y - 0.024 * h,
            f'{metres:g} m' if metres != 1 else '1 m',
            ha='center', va='bottom', fontsize=PT_SCALE_LABEL, color='#222222')


SHORT = {
    'voyis_stereo': 'stereo',
    'gemini_fls': 'FLS',
    'waterlinked_3d': '3D sonar',
}

# Print sizing lives in figure_style so this plate and the annotated
# photograph plate cannot drift to different label sizes in the same
# document. See that module for why the canvas is page-width.
FIG_ASPECT = 13.6 / 10.4          # width / height, unchanged from the montage


def key_panel(ax, cfg, volumes, pairs):
    """Colour key, geometry per sensor, and the overlap numbers.

    Lives in the corner the portrait hero render leaves empty, which is also
    the right place for it: the reader meets the colours before scanning the
    orthographic panels below.

    Everything here is on a height budget -- the cell is only about 2.4 in tall
    at print size. Two lines per sensor rather than three is what makes the key
    and the overlap table coexist; the long-form notes live in
    config/sensor_fov.yaml.
    """
    from matplotlib.patches import Patch
    ax.axis('off')

    handles, labels = [], []
    for spec in cfg['sensors'].values():
        handles.append(Patch(facecolor=spec['color'], alpha=0.6,
                             edgecolor=[c * 0.7 for c in spec['color']]))
        if spec['type'] == 'beams':
            geom = (f"{spec['n_beams']} × {spec['beamwidth_deg']:.0f}° "
                    f"@ {spec['slant_deg']:.0f}° slant")
        else:
            geom = (f"{spec['az_fov_deg']:.1f}° × "
                    f"{spec['el_fov_deg']:.1f}°").replace('.0°', '°')
        labels.append(f"{spec['label']}\n{geom}, {spec['range_note']}")

    leg = ax.legend(handles, labels, loc='upper left', frameon=False,
                    fontsize=PT_KEY_LABEL, handlelength=1.2, handleheight=1.35,
                    borderpad=0, labelspacing=0.95, handletextpad=0.7,
                    bbox_to_anchor=(-0.03, 1.03))
    for text in leg.get_texts():
        text.set_linespacing(1.4)

    # Monospace so the columns line up; at this size a proportional font makes
    # the numbers look ragged rather than tabular.
    #
    # Anchored to the BOTTOM of the cell while the legend is anchored to the
    # top. Placing both from the top means any change to the legend's height
    # (a longer sensor name, one more sensor) silently overruns this block.
    r = cfg['display_range_m']
    # Pad against the rendered pair string, not the sum of the two names --
    # the ' ∩ ' separator is 3 more characters and omitting it runs the label
    # straight into the number.
    width = max(len(f'{SHORT[a]} ∩ {SHORT[b]}') for a, b, *_ in pairs) + 2
    lines = [f'Pairwise overlap within {r:.0f} m']
    for a, b, both, fa, fb in pairs:
        pair = f'{SHORT[a]} ∩ {SHORT[b]}'
        lines.append(f' {pair:<{width}s}{both:.2f} m³ ({fa:.0f}/{fb:.0f}%)')
    lines.append('Volumes ' + ' '.join(
        f'{SHORT[k]} {v:.1f}' for k, v in volumes.items()) + ' m³')

    ax.text(-0.03, 0.0, '\n'.join(lines), transform=ax.transAxes,
            ha='left', va='bottom', fontsize=PT_KEY_STATS, color='#333333',
            family='DejaVu Sans Mono', linespacing=1.45)



def compose(panels, scales, cfg, volumes, pairs, out_path, single=None,
            width_in=DEFAULT_WIDTH_IN, dpi=DEFAULT_DPI):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if single:
        fig, ax = plt.subplots(figsize=(width_in, width_in * 0.8), dpi=dpi)
        ax.imshow(panels[single])
        ax.axis('off')
        ax.set_title(VIEWS[single]['title'], fontsize=PT_PANEL_TITLE,
                     color='#333333')
        draw_scale_bar(ax, panels[single], scales[single])
        fig.savefig(out_path, facecolor='white', bbox_inches='tight')
        pdf_path = os.path.splitext(out_path)[0] + '.pdf'
        fig.savefig(pdf_path, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        return pdf_path

    # No suptitle: in a dissertation the \caption carries the title, and a
    # second one baked into the image reads as a duplicate. Top margin is
    # tightened to reclaim the space it used to occupy.
    fig = plt.figure(figsize=(width_in, width_in / FIG_ASPECT), dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0],
                          width_ratios=[1.0, 1.0, 1.28],
                          hspace=0.13, wspace=0.03,
                          left=0.010, right=0.990, top=0.962, bottom=0.098)

    ax_hero = fig.add_subplot(gs[0, 0:2])
    ax_hero.imshow(panels['hero'])
    ax_hero.axis('off')
    ax_hero.set_title(VIEWS['hero']['title'], fontsize=PT_PANEL_TITLE, pad=4,
                      color='#333333')

    key_panel(fig.add_subplot(gs[0, 2]), cfg, volumes, pairs)

    for key, cell in [('top', gs[1, 0]), ('starboard', gs[1, 1]),
                      ('bow', gs[1, 2])]:
        ax = fig.add_subplot(cell)
        ax.imshow(panels[key])
        ax.axis('off')
        ax.set_title(VIEWS[key]['title'], fontsize=PT_PANEL_TITLE, pad=4,
                     color='#333333')
        draw_scale_bar(ax, panels[key], scales[key])

    # Wrapped against a measured character budget rather than matplotlib's
    # wrap=True, which measures against the figure edge and silently runs text
    # off both sides. The budget follows --width-in, so the caption re-wraps
    # correctly when the figure is built for a different text block.
    # Only what a reader needs while looking at the plate. The provenance
    # sentence belongs in the LaTeX \\caption -- main() prints a ready-to-paste
    # copy -- and duplicating it here just costs two lines of figure height.
    r = cfg['display_range_m']
    caption = (
        f'Volumes truncated at {r:.0f} m for legibility; true maximum ranges at '
        f'right. Stereo volume is the two-frustum intersection, i.e. the '
        f'depth-yielding volume only. '
        f'Triad at base_link: red = x forward, green = y port, blue = z up.'
    )
    fig.text(0.5, 0.010, wrap_to_width(caption, width_in, PT_CAPTION),
             ha='center', va='bottom', fontsize=PT_CAPTION, color='#444444',
             linespacing=1.5)

    fig.savefig(out_path, facecolor='white')
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(pdf_path, facecolor='white')
    plt.close(fig)
    return pdf_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(PKG_DIR, 'config', 'sensor_fov.yaml'))
    ap.add_argument('--urdf', default=os.path.join(PKG_DIR, 'urdf', 'rhody2.urdf.xacro'))
    ap.add_argument('--out', default=os.path.join(PKG_DIR, 'docs', 'figures',
                                                  'rhody2_sensor_fov.png'))
    ap.add_argument('--panel', choices=sorted(VIEWS), default=None,
                    help='render a single view instead of the montage')
    ap.add_argument('--width-in', type=float, default=DEFAULT_WIDTH_IN,
                    help='printed width in inches; set this to your LaTeX '
                         '\\textwidth so the in-figure point sizes come out '
                         'as intended at width=\\textwidth (default 6.5)')
    ap.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                    help='output resolution (default 400)')
    ap.add_argument('--width', type=int, default=2600,
                    help='offscreen render width in pixels, per panel')
    ap.add_argument('--no-triad', action='store_true')
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print(f'expanding {os.path.relpath(args.urdf, PKG_DIR)} with fov:=true')
    transforms, links = link_transforms(expand_xacro(args.urdf, fov=True))
    print(f'  {len(links)} links resolved in base_link')

    ren = build_renderer(cfg, transforms, links, show_triad=not args.no_triad)

    wanted = [args.panel] if args.panel else list(VIEWS)
    panels, scales = {}, {}
    for key in wanted:
        w = args.width if (key == 'hero' or args.panel) else args.width * 2 // 3
        h = int(w * (0.62 if key == 'hero' and not args.panel else 0.82))
        panels[key], scales[key] = render_view(ren, VIEWS[key], (w, h))
        ph, pw = panels[key].shape[:2]
        print(f'rendering {key} at {w}x{h} -> {pw}x{ph} after crop')

    print('computing coverage overlap (Monte Carlo)')
    masks, volumes, cell = overlap_table(cfg, transforms)

    # Pairwise overlap, the number Chapter 3 actually needs.
    pairs = []
    names = list(masks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = np.logical_and(masks[a], masks[b]).sum() * cell
            pairs.append((a, b, both, 100 * both / volumes[a],
                          100 * both / volumes[b]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pdf = compose(panels, scales, cfg, volumes, pairs, args.out,
                  single=args.panel, width_in=args.width_in, dpi=args.dpi)

    print('\npairwise coverage overlap within '
          f"{cfg['display_range_m']:.0f} m (Monte Carlo):")
    for a, b, both, fa, fb in pairs:
        print(f'  {a:<15s} & {b:<15s}  {both:6.2f} m³   '
              f'({fa:5.1f}% of {a}, {fb:5.1f}% of {b})')

    print(f'\nwrote {args.out}\n      {pdf}')

    if not args.panel:
        # The figure carries only what you need while looking at it; the
        # provenance belongs in the LaTeX caption, so hand it over ready to
        # paste rather than making it a thing to remember.
        print(f"""
LaTeX (figure is laid out for a {args.width_in:g} in text block):

  \\begin{{figure}}[tb]
    \\centering
    \\includegraphics[width=\\textwidth]{{figures/{os.path.basename(pdf)}}}
    \\caption[Rhody 2 sensor coverage]{{Sensor coverage of the Rhody 2
      vehicle, rendered from the URDF. The vehicle hull, the sensor poses and
      every field-of-view solid are taken from \\texttt{{rhody2.urdf.xacro}}
      expanded with \\texttt{{fov:=true}}, walked through the same fixed-joint
      chain \\texttt{{robot\\_state\\_publisher}} publishes on
      \\texttt{{/tf}}, so the coverage shown is the calibrated geometry rather
      than a sketch. The Gemini additionally reaches 120\\,m at 720\\,kHz. The
      Water Linked 3D sonar is not yet installed: both its field of view and
      its mounting pose are placeholders.}}
    \\label{{fig:rhody2-sensor-fov}}
  \\end{{figure}}
""")

    if abs(args.width_in - 6.5) < 1e-9:
        print('note: type is sized for a 6.5 in text block. If your '
              '\\textwidth differs,\n      re-run with --width-in <inches> so '
              'the point sizes land as intended.')


if __name__ == '__main__':
    main()
