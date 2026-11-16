"""Blade sections cut on cylinders r = const, unrolled to (arc length, axial).

The cylinder cut of :func:`cfdbender.geometry.blade_sections.slice_blade`. The leading edge is
the section point farthest ahead along the rotation direction, the trailing edge the point
farthest from it, and the chord is that distance. Only the standard library is used.
"""

from __future__ import annotations

import math

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


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _scale(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def _largest_component(faces: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for face in faces:
        for other in face[1:]:
            parent[find(other)] = find(face[0])
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for face in faces:
        groups.setdefault(find(face[0]), []).append(face)
    return max(groups.values(), key=len)


def _chains(segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in segments:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    seen: set[tuple[int, int]] = set()
    chains = []
    starts = [k for k, v in adjacency.items() if len(v) == 1] + list(adjacency)
    for start in starts:
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        while True:
            following = [n for n in adjacency[chain[-1]] if n not in seen]
            if not following:
                break
            chain.append(following[0])
            seen.add(following[0])
        chains.append(chain)
    return chains


def _section(points: list[tuple[float, float]], closed: bool, label: str) -> BladeSection:
    leading = max(range(len(points)), key=lambda i: (points[i][0], -i))
    le = points[leading]
    trailing = max(range(len(points)), key=lambda i: math.dist(points[i], le))
    te = points[trailing]
    chord = math.dist(le, te)
    if chord <= 0.0:
        raise BladeSliceError(f"{label}: section has zero chord")
    beta = math.degrees(math.atan2(le[1] - te[1], le[0] - te[0]))
    ux, uy = (te[0] - le[0]) / chord, (te[1] - le[1]) / chord
    nx, ny = -uy, ux
    if ny < 0:
        nx, ny = -nx, -ny
    local = [
        (
            ((p[0] - le[0]) * ux + (p[1] - le[1]) * uy) / chord,
            ((p[0] - le[0]) * nx + (p[1] - le[1]) * ny) / chord,
        )
        for p in points
    ]
    count = len(points)
    if closed:
        order = [(trailing + k) % count for k in range(count)] + [trailing]
    else:
        order = list(range(count))
    split = order.index(leading)
    first, second = order[: split + 1], order[split:]
    if not first or not second:
        raise BladeSliceError(f"{label}: cannot split the section at its leading edge")
    mean_first = sum(local[i][1] for i in first) / len(first)
    mean_second = sum(local[i][1] for i in second) / len(second)
    if mean_first >= mean_second:
        walk = first + second[1:]
    else:
        walk = second[::-1] + first[::-1][1:]
    return BladeSection(0.0, chord, beta, tuple(local[i] for i in walk))


def cylinder_sections(
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
        Wavefront OBJ text holding one blade or a mesh whose largest connected
        component is the blade. Lengths stay in the model's own units.
    n_sections : int
        Number of stations between the root and tip radii of the blade.
    spacing : str
        ``"cosine"`` (default) or ``"uniform"`` station distribution.
    cut : str
        ``"cylinder"`` (default, unrolled) or ``"plane"``.
    axis : str
        Rotation axis ``"x"``, ``"y"`` or ``"z"``; thrust points along it.
    hub_center : tuple of float
        A point on the rotation axis.
    rotation : str
        ``"ccw_from_front"`` (default) or ``"cw_from_front"``.

    Returns
    -------
    tuple of BladeSection
        Sections in ascending radius.

    Raises
    ------
    BladeSliceError
        Unknown choice, empty geometry or a station whose cut has no outline.
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
    vertices, all_faces = parse_obj(obj_text)
    if not all_faces:
        raise BladeSliceError("OBJ has no faces")
    faces = _largest_component(all_faces)
    used = sorted({i for face in faces for i in face})
    a = AXES[axis]
    local = {i: _sub(vertices[i], hub_center) for i in used}
    centroid = tuple(sum(local[i][k] for i in used) / len(used) for k in range(3))
    perp = _sub(centroid, _scale(a, _dot(centroid, a)))  # type: ignore[arg-type]
    length = math.sqrt(_dot(perp, perp))
    if length < 1e-12:
        raise BladeSliceError("radial direction is undefined: blade centroid lies on the axis")
    e_r = _scale(perp, 1.0 / length)
    e_t = _cross(a, e_r)
    if rotation == "cw_from_front":
        e_t = _scale(e_t, -1.0)
    rho = {i: math.sqrt(max(_dot(q, q) - _dot(q, a) ** 2, 0.0)) for i, q in local.items()}
    radial = {i: _dot(local[i], e_r) for i in used}
    measure = rho if cut == "cylinder" else radial
    sections = []
    for station in station_radii(min(measure.values()), max(measure.values()), n_sections, spacing):
        label = f"station r={station:.6g}"
        level = {i: measure[i] - station for i in used}
        points: dict[tuple[int, int], Vec] = {}
        segments = []
        for face in faces:
            crossings = []
            for p, q in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                if (level[p] >= 0.0) != (level[q] >= 0.0):
                    key = (min(p, q), max(p, q))
                    if key not in points:
                        t = level[key[0]] / (level[key[0]] - level[key[1]])
                        u, v = local[key[0]], local[key[1]]
                        points[key] = (
                            u[0] + t * (v[0] - u[0]),
                            u[1] + t * (v[1] - u[1]),
                            u[2] + t * (v[2] - u[2]),
                        )
                    crossings.append(key)
            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
        if not segments:
            raise BladeSliceError(f"{label}: the cut does not meet the blade")

        flats = {
            k: (
                station * math.atan2(_dot(q, e_t), _dot(q, e_r))
                if cut == "cylinder"
                else _dot(q, e_t),
                _dot(q, a),
            )
            for k, q in points.items()
        }

        def perimeter(
            chain: list[tuple[int, int]], flats: dict[tuple[int, int], tuple[float, float]] = flats
        ) -> float:
            xy = [flats[k] for k in chain]
            return sum(math.dist(xy[i], xy[i + 1]) for i in range(len(xy) - 1))

        chain = max(_chains(segments), key=perimeter)
        if len(chain) < 3:
            raise BladeSliceError(f"{label}: the cut outline has fewer than three points")
        closed = (chain[0], chain[-1]) in segments or (chain[-1], chain[0]) in segments
        outline = [flats[k] for k in chain]
        result = _section(outline, closed, label)
        sections.append(BladeSection(station, result.chord, result.beta_deg, result.shape))
    return tuple(sections)
