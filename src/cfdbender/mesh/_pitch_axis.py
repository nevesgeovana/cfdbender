"""The pitch axis of each blade from the rotor axis and a reference point.

``[repitch] rotor_axis`` is the direction of the rotor's rotation and ``reference_point`` places
the pitch axes. A scalar is a coordinate along the rotor axis: it locates the plane perpendicular
to the rotor axis, and every blade's pitch axis lies in that plane, along the blade's span
direction projected on it, through the hub center. The hub center is the centroid of the named
hub's cut by the plane when the cut closes (a full wheel), or the center of the circle fitted to
the cut when it is an arc (a sector); the cut crosses each blade root, so its two crossings of a
root loop are joined by a chord. A three-vector is one point for every blade.

The span direction is the principal axis of the blade's nodes, the eigenvector of their spread,
with no snap to a global axis: a snap would catch a blade lying near an axis and not its
neighbors, so the blades of a wheel would turn about lines that do not map onto each other. Its
sense is towards the tip.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cfdbender._errors import GeometryWarning, InputValueError, warn
from cfdbender.mesh._repitch_grid import edges, root_loop
from cfdbender.mesh._wheel import parse_axis

Array = NDArray[np.float64]
Edge = tuple[int, int]

#: A span direction whose part across the rotor axis is shorter than this is along the axis.
_ALONG_AXIS = 1e-6


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise InputValueError(f"repitch: {message}. No file was written.")


def span_direction(points: Array, towards: Array) -> Array:
    """Return the unit principal axis of a blade's nodes, with the sense away from ``towards``."""
    centred = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, -1]
    if float(axis @ (points.mean(axis=0) - towards)) < 0.0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def plane_basis(axis: Array) -> tuple[Array, Array]:
    """Return two unit vectors that, with ``axis``, make a right-handed frame."""
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(axis)))] = 1.0
    first = seed - (seed @ axis) * axis
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


@dataclass
class _Cut:
    """The crossings of a surface by a plane, as points on edges, and the links between them."""

    points: dict[Edge, Array] = field(default_factory=dict)
    links: dict[Edge, list[Edge]] = field(default_factory=lambda: defaultdict(list))

    def link(self, a: Edge, b: Edge) -> None:
        self.links[a].append(b)
        self.links[b].append(a)


def _cross_face(cut: _Cut, verts: Array, face: list[int], side: Array) -> None:
    """Add the crossings of one face and link them in pairs around the face."""
    crossing = []
    for a, b in zip(face, [*face[1:], face[0]], strict=True):
        if (side[a] >= 0.0) == (side[b] >= 0.0):
            continue
        key = (min(a, b), max(a, b))
        if key not in cut.points:
            t = side[a] / (side[a] - side[b])
            cut.points[key] = verts[a] + t * (verts[b] - verts[a])
        crossing.append(key)
    for k in range(0, len(crossing) - 1, 2):
        cut.link(crossing[k], crossing[k + 1])


def _bridge(cut: _Cut, roots: dict[str, set[Edge]]) -> None:
    """Join the two crossings of each root by a chord across it, for every family on the hub.

    ``roots`` holds, per family other than the hub, the hub's edges it shares: a blade's root
    loop whether the blade is named or not, so that the cut of a full wheel closes when only
    some of its blades turn.
    """
    for name, shared in roots.items():
        ends = [e for e in cut.points if e in shared and len(cut.links[e]) == 1]
        _need(
            len(ends) in (0, 2),
            f"the plane of reference_point crosses the root of {name!r} {len(ends)} times, "
            "not twice; choose a reference_point where the plane crosses each root once on each "
            "side",
        )
        if ends:
            cut.link(ends[0], ends[1])


def _shared_edges(families: dict[str, list[list[int]]], hubs: list[str]) -> dict[str, set[Edge]]:
    """Return, per family other than the hubs, the hub edges its faces share."""
    hub_edges = {e for n in hubs for f in families[n] for e in edges(f)}
    found = {}
    for name, faces in families.items():
        if name in hubs:
            continue
        shared = {e for f in faces for e in edges(f)} & hub_edges
        if shared:
            found[name] = shared
    return found


def _chains(cut: _Cut) -> list[tuple[list[Edge], bool]]:
    """Return the cut's pieces as ordered crossings, each with whether it closes."""
    seen: set[Edge] = set()
    pieces = []
    starts = sorted(cut.points, key=lambda e: (len(cut.links[e]) != 1, e))
    for start in starts:
        if start in seen:
            continue
        chain, previous = [start], None
        seen.add(start)
        while True:
            ahead = [e for e in cut.links[chain[-1]] if e != previous and e not in seen]
            if not ahead:
                break
            previous = chain[-1]
            chain.append(ahead[0])
            seen.add(ahead[0])
        closed = len(chain) > 2 and chain[0] in cut.links[chain[-1]]
        pieces.append((chain, closed))
    return pieces


def _circle(xy: Array) -> tuple[Array, float, float]:
    """Return the center, the radius and the largest radial residual of a least-squares circle."""
    a = np.c_[xy, np.ones(len(xy))]
    rhs = -(xy**2).sum(axis=1)
    (d, e, f), *_ = np.linalg.lstsq(a, rhs, rcond=None)
    centre = np.array([-0.5 * d, -0.5 * e])
    radius = math.sqrt(max(float(centre @ centre - f), 0.0))
    residual = float(np.abs(np.linalg.norm(xy - centre, axis=1) - radius).max())
    return centre, radius, residual


def _centroid(xy: Array) -> Array:
    """Return the area centroid of a closed polygon."""
    x, y = xy[:, 0], xy[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum()
    _need(bool(abs(area) > 0.0), "the hub's cut by the plane of reference_point encloses no area")
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6.0 * area)


def hub_centre(
    verts: Array, hub: list[list[int]], roots: dict[str, set[Edge]], axis: Array, station: float
) -> dict[str, Any]:
    """Return the hub center in the plane ``x . axis = station`` and how it was found."""
    side = verts @ axis - station
    cut = _Cut()
    for face in hub:
        _cross_face(cut, verts, face, side)
    _need(bool(cut.points), f"the plane of reference_point {station!r} does not cut the hub")
    _bridge(cut, roots)
    pieces = _chains(cut)
    _need(
        len(pieces) == 1,
        f"the hub's cut by the plane of reference_point {station!r} is {len(pieces)} pieces, "
        "not one closed loop (a full wheel) or one arc (a sector)",
    )
    chain, closed = pieces[0]
    first, second = plane_basis(axis)
    points = np.array([cut.points[e] for e in chain])
    xy = np.c_[points @ first, points @ second]
    fitted, radius, residual = _circle(xy)
    middle = _centroid(xy) if closed else fitted
    point = station * axis + middle[0] * first + middle[1] * second
    return {
        "point": point.tolist(),
        "cut": "closed loop" if closed else "arc",
        "cut_points": len(points),
        "fitted_radius": radius,
        "fit_residual": residual,
    }


def root_miss(verts: Array, loop: list[int], origin: Array, direction: Array) -> float:
    """Return how far the pitch axis passes outside the root loop seen along it, 0 inside."""
    first, second = plane_basis(direction)
    q = verts[loop] - origin
    xy = np.c_[q @ first, q @ second]
    following = np.roll(xy, -1, axis=0)
    inside = False
    for (x0, y0), (x1, y1) in zip(xy, following, strict=True):
        if (y0 > 0.0) != (y1 > 0.0) and x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0) > 0.0:
            inside = not inside
    if inside:
        return 0.0
    side = following - xy
    t = np.clip(-(xy * side).sum(axis=1) / np.maximum((side * side).sum(axis=1), 1e-300), 0, 1)
    return float(np.linalg.norm(xy + t[:, None] * side, axis=1).min())


def _station(cfg: dict[str, Any], axis: Array) -> tuple[float | None, Array | None]:
    """Return the scalar station, or the point, that ``reference_point`` states."""
    value = cfg["reference_point"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _need(math.isfinite(float(value)), "repitch.reference_point must be finite")
        return float(value), None
    point = np.asarray(value if isinstance(value, list) else [], dtype=float)
    _need(
        point.shape == (3,) and bool(np.isfinite(point).all()),
        "repitch.reference_point must be a coordinate along rotor_axis or a point [x, y, z]",
    )
    return None, point


def pitch_axes(
    verts: Array, families: dict[str, list[list[int]]], blades: list[str], hubs: list[str], cfg: Any
) -> tuple[dict[str, tuple[Array, Array]], dict[str, Any]]:
    """Return each blade's pitch axis (origin, direction) and the report of how it was found."""
    _need(
        "rotor_axis" in cfg and "reference_point" in cfg,
        "a blade with no axis of its own needs repitch.rotor_axis and repitch.reference_point",
    )
    axis = parse_axis(cfg["rotor_axis"])
    hub_faces = [f for n in hubs for f in families[n]]
    roots = {name: root_loop(families[name], hub_faces, name) for name in blades}
    station, point = _station(cfg, axis)
    report: dict[str, Any] = {"rotor_axis": axis.tolist()}
    if point is None:
        assert station is not None
        report["reference_station"] = station
        shared = _shared_edges(families, hubs)
        report["hub_centre"] = hub_centre(verts, hub_faces, shared, axis, station)
        point = np.asarray(report["hub_centre"]["point"])
    else:
        report["reference_point"] = point.tolist()
    found: dict[str, tuple[Array, Array]] = {}
    report["blades"] = {}
    for name in blades:
        ids = np.unique([v for f in families[name] for v in f])
        # Towards the tip is away from the root, wherever reference_point lies on the pitch line.
        span = span_direction(verts[ids], verts[roots[name]].mean(axis=0))
        across = span - (span @ axis) * axis
        _need(
            float(np.linalg.norm(across)) > _ALONG_AXIS,
            f"blade {name!r} spans along rotor_axis, so no pitch axis lies across it",
        )
        direction = across / np.linalg.norm(across)
        miss = root_miss(verts, roots[name], point, direction)
        found[name] = (point.copy(), direction)
        report["blades"][name] = {
            "span_direction": span.tolist(),
            "pitch_axis_direction": direction.tolist(),
            "pitch_axis_miss_of_root": miss,
        }
        if miss > 0.0:
            warn(
                f"repitch: the pitch axis of blade {name!r} passes {miss:.6g} (mesh units) "
                "outside its root loop; check reference_point",
                GeometryWarning,
            )
    return found, report


#: A shaft or datum shorter than this has no direction.
_AXIS_TOLERANCE = 1e-12


def frame_basis_for_shaft(
    shaft: tuple[float, float, float],
    datum: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the ``x_axis`` and ``y_axis`` of a frame whose third axis is ``shaft``.

    ``x_axis`` is the blade datum projected into the disk plane, which makes an azimuth well
    defined on a tilted rotor; ``y_axis`` completes the right-handed basis. A shaft that is
    not a unit vector is normalized first, so a scaled shaft gives the same frame.

    Raises
    ------
    InputValueError
        The shaft has no direction, or the datum lies along it.
    """
    shaft_length = math.sqrt(sum(component * component for component in shaft))
    if shaft_length <= _AXIS_TOLERANCE:
        raise InputValueError(f"the shaft {shaft} has no direction")
    shaft = (shaft[0] / shaft_length, shaft[1] / shaft_length, shaft[2] / shaft_length)
    along = sum(a * b for a, b in zip(shaft, datum, strict=True))
    projected = tuple(d - along * s for s, d in zip(shaft, datum, strict=True))
    length = math.sqrt(sum(component * component for component in projected))
    if length <= _AXIS_TOLERANCE:
        raise InputValueError(
            f"the blade datum {datum} lies along the shaft {shaft}, so it projects to "
            "nothing in the disk plane and no azimuth can be measured from it"
        )
    x_axis = (projected[0] / length, projected[1] / length, projected[2] / length)
    y_axis = (
        shaft[1] * x_axis[2] - shaft[2] * x_axis[1],
        shaft[2] * x_axis[0] - shaft[0] * x_axis[2],
        shaft[0] * x_axis[1] - shaft[1] * x_axis[0],
    )
    return x_axis, y_axis
