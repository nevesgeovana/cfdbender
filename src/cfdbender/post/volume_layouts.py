"""Where the sample points of a volume cut and of probes are placed.

The user states only the extremities of a rectangle or the radii of a circle, or the points, lines,
rectangles and circles of a probe entry; the points are laid here, in the frame the entry names,
in meters, with the arithmetic of the layouts they are recorded against (see
``tools/record_pyfs_volume_layouts.py``). Nothing here reads a file or a solver.

A volume cut is laid by :func:`volume_section_points`:

* a rectangle between two diagonal corners ``(u0, v0, u1, v1)`` on a plane, with ``points`` stations
  along each of the two axes, both ends included, ``(25, 25)`` when not stated; each extra
  refinement layer bisects every interval, so the counts become ``(n - 1) * 2 ** (layers - 1) + 1``.
  The points run with the second axis fastest;
* a circle between an inner and an outer radius, ``points = (radial, azimuthal)`` stations; ring
  ``i`` has radius ``inner + (outer - inner) * i / (radial - 1)`` and ``azimuthal`` points at angles
  ``2 pi j / azimuthal`` from the first in-plane axis; a ring of zero radius is one point.

The plane is ``XY``, ``XZ`` or ``YZ``: the in-plane axes in that written order, and ``offset_m``
along the remaining axis. The probes are laid by :func:`line_points`, :func:`rectangle_points` and
:func:`circle_points`, rounded to five decimals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_SECTION_POINTS",
    "LayoutError",
    "circle_points",
    "line_points",
    "rectangle_points",
    "volume_section_points",
]

#: The station counts of a rectangle that states none.
DEFAULT_SECTION_POINTS: tuple[int, int] = (25, 25)

#: The decimals the probe shapes are rounded to.
_PROBE_DECIMALS = 5


class LayoutError(ValueError):
    """A declaration that does not lay a set of points."""


def _vector(value: Sequence[float], what: str) -> list[float]:
    if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
        raise LayoutError(f"{what} is {value!r}; it is three finite numbers")
    return [float(x) for x in value]


def volume_section_points(
    shape: str,
    plane: str,
    *,
    corners_m: Sequence[float] | None = None,
    radii_m: Sequence[float] | None = None,
    points: Sequence[int] | None = None,
    refinement_layers: int = 1,
    offset_m: float = 0.0,
    center_m: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Lay the points of a volume cut, in the declared frame, in meters.

    Parameters
    ----------
    shape : {'rectangle', 'circle'}
        The extremities the user states.
    plane : {'XY', 'XZ', 'YZ'}
        The plane of the cut.
    corners_m : sequence of four floats
        Rectangle: ``u0, v0, u1, v1``, the lower and upper diagonal coordinates.
    radii_m : sequence of two floats
        Circle: the inner and outer radius, ``0 <= r1 < r2``.
    points : sequence of two ints, optional
        Rectangle: the stations along each axis, both at least 2, ``(25, 25)`` by default.
        Circle: the radial (at least 2) and azimuthal (at least 3) counts, required.
    refinement_layers : int, default 1
        Rectangle: each extra layer bisects every interval.
    offset_m : float, default 0
        Distance along the remaining axis of the plane.
    center_m : sequence of two floats, optional
        Circle: the centre in the plane's own coordinates (``u, v``); ``(0, 0)`` by default.
        With ``0 < r1`` the cut is an annulus: no point lies inside the inner radius. A rectangle
        takes none.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 3)``, in the order the declaration lays them.

    Raises
    ------
    LayoutError
        If a key of the other shape is stated, a needed key is missing, a count is too small, a
        corner pair encloses no area or the radii are not ``0 <= r1 < r2``.

    Examples
    --------
    >>> volume_section_points("rectangle", "YZ", corners_m=(0, 0, 1, 2), points=(2, 3)).shape
    (6, 3)
    """
    if plane not in ("XY", "XZ", "YZ"):
        raise LayoutError(f"the plane is {plane!r}; it is one of 'XY', 'XZ' or 'YZ'")
    if shape == "rectangle":
        if center_m is not None:
            raise LayoutError("a rectangle takes corners_m; center_m is a circle's")
        if radii_m is not None:
            raise LayoutError("a rectangle takes corners_m; radii_m is a circle's")
        if corners_m is None or len(corners_m) != 4:
            raise LayoutError("a rectangle takes corners_m, the four numbers u0, v0, u1, v1")
        counts = tuple(points) if points is not None else DEFAULT_SECTION_POINTS
        if len(counts) != 2 or min(counts) < 2:
            raise LayoutError("a rectangle needs at least 2 stations along each axis")
        if refinement_layers < 1:
            raise LayoutError("refinement_layers is at least 1")
        u0, v0, u1, v1 = (float(x) for x in corners_m)
        if u0 == u1 or v0 == v1:
            raise LayoutError(f"corners_m {list(corners_m)} enclose no area")
        count_u = (counts[0] - 1) * 2 ** (refinement_layers - 1) + 1
        count_v = (counts[1] - 1) * 2 ** (refinement_layers - 1) + 1
        pairs = [
            (u0 + (u1 - u0) * i / (count_u - 1), v0 + (v1 - v0) * j / (count_v - 1))
            for i in range(count_u)
            for j in range(count_v)
        ]
    elif shape == "circle":
        if corners_m is not None:
            raise LayoutError("a circle takes radii_m; corners_m is a rectangle's")
        if radii_m is None or len(radii_m) != 2:
            raise LayoutError("a circle takes radii_m, the inner and outer radius")
        if points is None or len(points) != 2:
            raise LayoutError("a circle takes points, the radial and azimuthal counts")
        radial, azimuth = (int(x) for x in points)
        if radial < 2 or azimuth < 3:
            raise LayoutError("a circle needs at least 2 radial and 3 azimuthal stations")
        inner, outer = float(radii_m[0]), float(radii_m[1])
        if not 0.0 <= inner < outer:
            raise LayoutError(f"radii_m {list(radii_m)} are not 0 <= r1 < r2")
        if center_m is None:
            centre_u, centre_v = 0.0, 0.0
        elif len(center_m) != 2 or not all(math.isfinite(float(x)) for x in center_m):
            raise LayoutError(f"center_m is {list(center_m)!r}; it is two finite numbers")
        else:
            centre_u, centre_v = float(center_m[0]), float(center_m[1])
        pairs = []
        for i in range(radial):
            radius = inner + (outer - inner) * i / (radial - 1)
            if radius == 0:
                pairs.append((centre_u, centre_v))
            else:
                pairs.extend(
                    (
                        centre_u + radius * math.cos(2 * math.pi * j / azimuth),
                        centre_v + radius * math.sin(2 * math.pi * j / azimuth),
                    )
                    for j in range(azimuth)
                )
    else:
        raise LayoutError(f"the shape is {shape!r}; it is 'rectangle' or 'circle'")
    rows = []
    for u, v in pairs:
        local = {"XY": [u, v, offset_m], "XZ": [u, offset_m, v], "YZ": [offset_m, u, v]}[plane]
        rows.append(local)
    return np.array(rows, dtype=np.float64)


def line_points(
    start: Sequence[float], end: Sequence[float], points: int, scale: float = 1.0
) -> NDArray[np.float64]:
    """Lay the points of a probe line, both ends included, rounded to five decimals.

    Parameters
    ----------
    start, end : sequence of float
        The ends in the entry's frame and unit.
    points : int
        The count, at least 2.
    scale : float, default 1
        The factor from the entry's unit to meters.

    Returns
    -------
    numpy.ndarray
        Shape ``(points, 3)``.
    """
    if points < 2:
        raise LayoutError(f"a probe line has at least two points, both ends included; got {points}")
    a, b = _vector(start, "start"), _vector(end, "end")
    rows = [
        [
            round((p + (q - p) * (step / (points - 1))) * scale, _PROBE_DECIMALS)
            for p, q in zip(a, b, strict=True)
        ]
        for step in range(points)
    ]
    return np.array(rows, dtype=np.float64)


def rectangle_points(
    origin: Sequence[float],
    along_u: Sequence[float],
    along_v: Sequence[float],
    points_u: int,
    points_v: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Lay the grid of a probe rectangle given by three vertices, v fastest.

    The grid runs from ``origin`` toward ``along_u`` in ``points_u`` stations and toward
    ``along_v`` in ``points_v``, both ends included, rounded to five decimals.
    """
    if points_u < 2 or points_v < 2:
        raise LayoutError(
            "a probe rectangle has at least two points along each direction, both ends included"
        )
    o, u, v = _vector(origin, "origin"), _vector(along_u, "along_u"), _vector(along_v, "along_v")
    edge_u = [b - a for a, b in zip(o, u, strict=True)]
    edge_v = [b - a for a, b in zip(o, v, strict=True)]
    cross = [
        edge_u[1] * edge_v[2] - edge_u[2] * edge_v[1],
        edge_u[2] * edge_v[0] - edge_u[0] * edge_v[2],
        edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0],
    ]
    area = math.sqrt(sum(x * x for x in cross))
    reach = math.sqrt(sum(x * x for x in edge_u)) * math.sqrt(sum(x * x for x in edge_v))
    if not (math.isfinite(area) and area > 1e-12 * reach):
        raise LayoutError("the rectangle's two edges are collinear or zero, so it has no area")
    rows = []
    for index_u in range(points_u):
        fraction_u = index_u / (points_u - 1)
        for index_v in range(points_v):
            fraction_v = index_v / (points_v - 1)
            rows.append(
                [
                    round(
                        (o[axis] + edge_u[axis] * fraction_u + edge_v[axis] * fraction_v) * scale,
                        _PROBE_DECIMALS,
                    )
                    for axis in range(3)
                ]
            )
    return np.array(rows, dtype=np.float64)


def circle_points(
    center: Sequence[float],
    normal: Sequence[float],
    radius: float,
    points_radial: int,
    points_azimuth: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Lay the polar grid of a probe circle; the center appears once.

    ``points_radial`` stations run from the center to the rim, both included, and
    ``points_azimuth`` around each ring. The two in-plane axes come from the normal and the
    world axis least aligned with it. Rounded to five decimals.
    """
    if not (math.isfinite(radius) and radius > 0.0):
        raise LayoutError(f"the circle radius is {radius!r}; it is finite and positive")
    c = _vector(center, "center")
    n = _vector(normal, "normal")
    length = math.sqrt(sum(x * x for x in n))
    if length == 0.0:
        raise LayoutError("a probe circle needs a normal with a direction; got a zero vector")
    if points_radial < 2 or points_azimuth < 1:
        raise LayoutError("a probe circle has at least two radial stations and one azimuth point")
    n = [x / length for x in n]
    least = min(range(3), key=lambda axis: abs(n[axis]))
    seed = [1.0 if axis == least else 0.0 for axis in range(3)]
    first = [
        seed[1] * n[2] - seed[2] * n[1],
        seed[2] * n[0] - seed[0] * n[2],
        seed[0] * n[1] - seed[1] * n[0],
    ]
    span = math.sqrt(sum(x * x for x in first))
    first = [x / span for x in first]
    second = [
        n[1] * first[2] - n[2] * first[1],
        n[2] * first[0] - n[0] * first[2],
        n[0] * first[1] - n[1] * first[0],
    ]
    rows: list[list[float]] = []
    for index_r in range(points_radial):
        ring = radius * index_r / (points_radial - 1)
        if ring == 0.0:
            rows.append([round(x * scale, _PROBE_DECIMALS) for x in c])
            continue
        for index_a in range(points_azimuth):
            angle = 2.0 * math.pi * index_a / points_azimuth
            rows.append(
                [
                    round(
                        (
                            c[axis]
                            + ring
                            * (math.cos(angle) * first[axis] + math.sin(angle) * second[axis])
                        )
                        * scale,
                        _PROBE_DECIMALS,
                    )
                    for axis in range(3)
                ]
            )
    return np.array(rows, dtype=np.float64)
