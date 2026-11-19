"""Radial sections of a rotor blade from its surface mesh, for blade-element solvers.

Two cuts are offered. ``plane`` uses the propeller slicer
(:func:`cfdbender.geometry.stages.slice_blade`): each station is the plane at distance ``r``
from the shaft, normal to the blade's radial direction rotated by the station's skew, and chord,
pitch and shape are measured with the propeller builder's own definitions. ``cylinder`` (the
default) cuts the cylinder r = const and unrolls it to (arc length, axial); there the leading
edge is the point farthest ahead along the rotation. This module states the convention a
blade-element solver reads, each choice an argument with this default:

- the rotation axis is ``x`` and thrust points along +x;
- the hub center is the origin;
- the rotor turns counter-clockwise seen from +x looking aft (``ccw_from_front``);
- the blade is the largest connected mesh component;
- stations are cell-centered between the blade's innermost and outermost radial distances,
  cosine (clustered toward both ends) or uniform, never on an end;
- on a plane cut the leading edge is the rounder end of the section, or, when both ends are
  equally round, the end ahead in the direction of motion measured on a reference section; the
  chord runs from it to the trailing-edge midpoint;
- beta is the angle in degrees between the chord line and the rotation plane, positive when
  the leading edge sits ahead along the thrust axis;
- the shape is in unit-chord coordinates with x from the leading edge (0) to the trailing
  edge (1) and y positive on the thrust side, ordered from the trailing edge over the upper
  side and back under the lower.

Lengths stay in the mesh's own units.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.geometry._blade_types import (
    AXES,
    CUTS,
    ROTATIONS,
    SPACINGS,
    BladeSection,
    BladeSliceError,
    Vec,
    parse_obj,
    station_radii,
)
from cfdbender.geometry._cylinder_cut import cylinder_sections
from cfdbender.geometry._propeller_slice import BladeFrame, BladeSurface
from cfdbender.geometry._propeller_slice import slice_blade as _slice_planes
from cfdbender.mesh.objio import weld

__all__ = [
    "AXES",
    "CUTS",
    "ROTATIONS",
    "SPACINGS",
    "BladeSection",
    "BladeSliceError",
    "parse_obj",
    "radial_extent",
    "slice_blade",
    "station_radii",
]

#: The propeller slicer needs at least this many stations to fit its end laws.
_FEWEST_CUTS = 4


def _blade(text: str, axis: str, *, welded: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices and the triangles of the largest connected component.

    With ``welded``, coincident vertices are merged first, so a surface written as separate
    patches or as one triangle per face is one component, as the propeller slicer sees it.
    """
    if axis not in AXES:
        raise BladeSliceError(f"axis must be one of: {', '.join(AXES)}")
    vertices, faces = parse_obj(text)
    if not faces:
        raise BladeSliceError("OBJ has no faces")
    xyz = np.asarray(vertices, dtype=float)
    triangles = np.asarray(faces, dtype=np.int64)
    if welded:
        xyz, triangles = weld(xyz, triangles)
        if not len(triangles):
            raise BladeSliceError("OBJ has no faces")
    rows = np.repeat(np.arange(len(triangles)), 3)
    incidence = coo_matrix((np.ones(rows.size), (rows, triangles.ravel())))
    graph = (incidence @ incidence.T).tocsr()
    _, label = connected_components(graph, directed=False)
    kept = triangles[label == np.bincount(label).argmax()]
    used, compact = np.unique(kept, return_inverse=True)
    return xyz[used], compact.reshape(-1, 3)


def radial_extent(
    obj_text: str,
    *,
    axis: str = "x",
    hub_center: Vec = (0.0, 0.0, 0.0),
    cut: str = "cylinder",
) -> tuple[float, float]:
    """Return the smallest and largest distance of the blade's vertices from the axis.

    Parameters
    ----------
    obj_text : str
        Wavefront OBJ text, as for :func:`slice_blade`.
    axis : str
        Rotation axis ``"x"``, ``"y"`` or ``"z"``.
    hub_center : tuple of float
        A point on the rotation axis.
    cut : str
        The cut the sections are taken with, so the blade is the one that cut sees: on a
        ``"plane"`` cut coincident vertices are merged before the largest component is taken.

    Returns
    -------
    tuple of float
        The innermost and outermost distance, in model units.
    """
    if cut not in CUTS:
        raise BladeSliceError(f"cut must be one of: {', '.join(CUTS)}")
    xyz, triangles = _blade(obj_text, axis, welded=cut == "plane")
    a = np.asarray(AXES[axis])
    q = xyz[np.unique(triangles)] - np.asarray(hub_center, dtype=float)
    radii = np.sqrt(np.maximum(np.einsum("ij,ij->i", q, q) - (q @ a) ** 2, 0.0))
    return float(radii.min()), float(radii.max())


def slice_blade(
    obj_text: str,
    *,
    n_sections: int,
    spacing: str = "cosine",
    cut: str = "cylinder",
    axis: str = "x",
    hub_center: Vec = (0.0, 0.0, 0.0),
    rotation: str = "ccw_from_front",
) -> tuple[BladeSection, ...]:
    """Cut a blade OBJ at radial stations and measure radius, chord, beta and shape.

    Parameters
    ----------
    obj_text : str
        Wavefront OBJ text holding one blade or a mesh whose largest connected component is
        the blade. Lengths stay in the model's own units.
    n_sections : int
        Number of stations between the root and tip of the blade.
    spacing : str
        ``"cosine"`` (default) or ``"uniform"`` station distribution.
    cut : str
        ``"cylinder"`` (default), the cylinder r = const unrolled, or ``"plane"``, the
        skew-following plane of the propeller slicer.
    axis : str
        Rotation axis ``"x"``, ``"y"`` or ``"z"``; thrust points along it.
    hub_center : tuple of float
        A point on the rotation axis.
    rotation : str
        ``"ccw_from_front"`` (default) or ``"cw_from_front"``. It orients the cylinder cut; a
        plane cut measures the direction of motion from the blade's reference section.

    Returns
    -------
    tuple of BladeSection
        Sections in ascending radius.

    Raises
    ------
    BladeSliceError
        Unknown choice, empty geometry or a station whose cut has no closed outline.
    """
    for value, allowed, name in (
        (spacing, SPACINGS, "spacing"),
        (cut, CUTS, "cut"),
        (rotation, ROTATIONS, "rotation"),
        (axis, tuple(AXES), "axis"),
    ):
        if value not in allowed:
            raise BladeSliceError(f"{name} must be one of: {', '.join(allowed)}")
    if type(n_sections) is not int or n_sections < 1:
        raise BladeSliceError("n_sections must be a positive integer")
    if cut == "cylinder":
        return cylinder_sections(
            obj_text,
            n_sections=n_sections,
            spacing=spacing,
            cut=cut,
            axis=axis,
            hub_center=hub_center,
            rotation=rotation,
        )
    xyz, triangles = _blade(obj_text, axis, welded=True)
    thrust = np.asarray(AXES[axis])
    origin = np.asarray(hub_center, dtype=float)
    used = np.unique(triangles)
    centroid = xyz[used].mean(axis=0) - origin
    radial = centroid - (centroid @ thrust) * thrust
    length = float(np.linalg.norm(radial))
    if length < 1e-12:
        raise BladeSliceError("radial direction is undefined: blade centroid lies on the axis")
    e_r = radial / length
    distance = (xyz[used] - origin) @ e_r
    wanted = station_radii(float(distance.min()), float(distance.max()), n_sections, spacing)
    tip = float(distance.max())
    cuts = sorted(set(wanted) | set(station_radii(wanted[0], wanted[-1], _FEWEST_CUTS, "uniform")))
    if n_sections == 1:
        cuts = station_radii(float(distance.min()), tip, _FEWEST_CUTS, "uniform") + wanted
        cuts = sorted(set(cuts))
    frame = BladeFrame(origin, thrust, e_r)
    try:
        result = _slice_planes(
            BladeSurface(xyz, triangles, frame), blades=1, stations=[r / tip for r in cuts]
        )
    except (InputArtifactError, InputValueError) as error:
        raise BladeSliceError(str(error)) from error
    radius = result["measured"]["radius_mm"]
    rows = {round(r, 12): row for r, row in zip(cuts, result["table"], strict=True)}
    sections = []
    for r in wanted:
        row = rows[round(r, 12)]
        upper = [(float(x), float(y)) for x, y in row["profile"]["upper"]]
        lower = [(float(x), float(y)) for x, y in row["profile"]["lower"]]
        beta = float(row["beta_deg"])
        if abs(beta) > 90.0:
            # The slicer named the end behind in the measured direction of motion the leading
            # edge (two ends of nearly equal width). A blade element's leading edge is the end
            # ahead: turn the section about its mid-chord.
            beta -= math.copysign(180.0, beta)
            upper, lower = (
                [(1.0 - x, -y) for x, y in lower[::-1]],
                [(1.0 - x, -y) for x, y in upper[::-1]],
            )
        shape = tuple(upper[::-1] + lower[1:])
        sections.append(BladeSection(row["r_R"] * radius, row["c_R"] * radius, beta, shape))
    return tuple(sections)
