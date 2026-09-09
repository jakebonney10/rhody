#!/usr/bin/env python3
"""Bake the sensor field-of-view solids in config/sensor_fov.yaml into meshes.

Writes binary STL into meshes/fov/, one solid per sensor, which
urdf/sensor_fov.xacro then hangs off the corresponding sensor link.

STL rather than Collada on purpose: STL carries no material, so the FOV colour
and -- more importantly -- the alpha come from the URDF <material>, where they
can be read and tweaked next to the geometry. A .dae would ship its own
material and RViz's override behaviour for it is less predictable.

Every solid is generated in the sensor link's own frame, REP-103 style:
  +x boresight, +y port, +z up, apex at the origin
so the xacro attaches it with an identity origin. The DVL beam cone is the one
exception -- it opens along -z (down), and the xacro rotates a copy onto each
of the three slanted beams.

Usage:
    python3 scripts/gen_fov_meshes.py [--config PATH] [--outdir PATH]
"""

import argparse
import os
import sys

import numpy as np
import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# Solid construction
#
# Each builder returns (verts Nx3, faces Mx3). Winding is fixed afterwards by
# orient_outward(), so builders can emit triangles in whatever order is
# convenient -- every shape here is convex, which is what makes that work.
# --------------------------------------------------------------------------

def orient_outward(verts, faces):
    """Flip any triangle whose normal points back toward the solid's centre.

    Valid only for convex solids (all of ours are: pyramid, spherical sector
    with < 180 deg of sweep, cone). Saves hand-deriving a winding order per
    shape and getting one of them backwards.
    """
    centre = verts.mean(axis=0)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    outward = (a + b + c) / 3.0 - centre
    flip = np.einsum('ij,ij->i', normals, outward) < 0.0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return verts, faces


def direction(az, el):
    """Unit vectors for azimuth about +z (positive to port) and elevation."""
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], axis=-1)


def build_frustum(az_fov_deg, el_fov_deg, r):
    """Camera frustum: rectangular pyramid with a *planar* far face.

    Planar because a pinhole camera's far clip is an image plane, unlike the
    constant-range arc a sonar returns.
    """
    hw = r * np.tan(np.radians(az_fov_deg) / 2.0)
    hh = r * np.tan(np.radians(el_fov_deg) / 2.0)
    verts = np.array([
        [0.0, 0.0, 0.0],     # 0 apex / entrance pupil
        [r,  hw,  hh],       # 1
        [r, -hw,  hh],       # 2
        [r, -hw, -hh],       # 3
        [r,  hw, -hh],       # 4
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],   # sides
        [1, 2, 3], [1, 3, 4],                          # far face
    ])
    return orient_outward(verts, faces)


def build_fan(az_fov_deg, el_fov_deg, r, n_az=64, n_el=12):
    """Sonar sector: spherical cap over an az x el window, apex at the array.

    The far face is a constant-range spherical cap, not a plane -- a sonar
    resolves range, so its coverage boundary is an arc. At 120 deg of azimuth
    that difference is very visible and worth getting right.
    """
    az = np.linspace(-np.radians(az_fov_deg) / 2.0,
                     np.radians(az_fov_deg) / 2.0, n_az + 1)
    el = np.linspace(-np.radians(el_fov_deg) / 2.0,
                     np.radians(el_fov_deg) / 2.0, n_el + 1)
    az_g, el_g = np.meshgrid(az, el, indexing='ij')
    cap = direction(az_g, el_g).reshape(-1, 3) * r

    verts = np.vstack([np.zeros((1, 3)), cap])   # index 0 is the apex
    idx = np.arange(1, cap.shape[0] + 1).reshape(n_az + 1, n_el + 1)

    faces = []
    # Spherical cap, quads split into triangles.
    for i in range(n_az):
        for j in range(n_el):
            a, b = idx[i, j], idx[i + 1, j]
            c, d = idx[i + 1, j + 1], idx[i, j + 1]
            faces += [[a, b, c], [a, c, d]]
    # Four side walls, each edge of the cap fanned back to the apex.
    for j in range(n_el):                      # az extremes
        faces += [[0, idx[0, j], idx[0, j + 1]],
                  [0, idx[n_az, j], idx[n_az, j + 1]]]
    for i in range(n_az):                      # el extremes
        faces += [[0, idx[i, 0], idx[i + 1, 0]],
                  [0, idx[i, n_el], idx[i + 1, n_el]]]

    return orient_outward(verts, np.array(faces))


def build_beam_cone(beamwidth_deg, r, n=48):
    """A single acoustic beam opening along -z, spherical end cap.

    Used for the DVL's three slanted beams and its central altimeter; the xacro
    rotates a copy of this solid onto each beam axis.
    """
    half = np.radians(beamwidth_deg) / 2.0
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.sin(half) * np.cos(phi),
                     r * np.sin(half) * np.sin(phi),
                     np.full(n, -r * np.cos(half))], axis=-1)

    verts = np.vstack([np.zeros((1, 3)), ring, [[0.0, 0.0, -r]]])
    cap_centre = n + 1

    faces = []
    for i in range(n):
        a, b = 1 + i, 1 + (i + 1) % n
        faces += [[0, a, b], [cap_centre, a, b]]

    return orient_outward(verts, np.array(faces))


# --------------------------------------------------------------------------
# Binary STL
# --------------------------------------------------------------------------

def write_stl(path, verts, faces, header=b''):
    tris = verts[faces]                                   # M x 3 x 3
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0)

    record = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    data = np.zeros(len(faces), dtype=record)
    data['n'] = normals
    data['v'] = tris

    with open(path, 'wb') as fh:
        fh.write(header[:79].ljust(80, b'\0'))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(data.tobytes())


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(PKG_DIR, 'config', 'sensor_fov.yaml'))
    ap.add_argument('--outdir', default=os.path.join(PKG_DIR, 'meshes', 'fov'))
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(args.outdir, exist_ok=True)
    default_range = float(cfg['display_range_m'])

    written = []
    for name, spec in cfg['sensors'].items():
        r = float(spec.get('display_range_m', default_range))
        kind = spec['type']
        header = f'rhody FOV: {name} r={r}m (generated, do not edit)'.encode()

        if kind == 'frustum':
            solids = {f'{name}_frustum': build_frustum(
                spec['az_fov_deg'], spec['el_fov_deg'], r)}
        elif kind == 'fan':
            solids = {f'{name}_fan': build_fan(
                spec['az_fov_deg'], spec['el_fov_deg'], r)}
        elif kind == 'beams':
            solids = {f'{name}_beam': build_beam_cone(spec['beamwidth_deg'], r)}
            if spec.get('include_altimeter'):
                solids[f'{name}_altimeter'] = build_beam_cone(
                    spec['altimeter_beamwidth_deg'], r)
        else:
            sys.exit(f"unknown FOV type '{kind}' for sensor '{name}'")

        for stem, (verts, faces) in solids.items():
            path = os.path.join(args.outdir, f'{stem}.stl')
            write_stl(path, verts, faces, header)
            written.append((os.path.basename(path), len(verts), len(faces)))

    width = max(len(n) for n, _, _ in written)
    print(f'wrote {len(written)} FOV meshes to {args.outdir}')
    for name, nv, nf in written:
        print(f'  {name:<{width}}  {nv:6d} verts  {nf:6d} tris')


if __name__ == '__main__':
    main()
