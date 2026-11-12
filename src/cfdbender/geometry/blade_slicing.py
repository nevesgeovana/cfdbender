"""Slice a three-dimensional rotor blade into radial sections.

Convention, each choice an argument with this default:

- the rotation axis is ``x`` and thrust points along +x;
- the hub centre is the origin;
- the rotor turns counter-clockwise seen from +x looking aft (``ccw_from_front``);
- the blade is the largest connected mesh component;
- the radial direction runs from the axis to the blade centroid;
- a station is cut on the cylinder r = const, unrolled to (arc length, axial),
  or on the plane normal to the radial direction (``cut="plane"``);
- the leading edge is the section point farthest ahead along the rotation
  direction, the trailing edge is the section point farthest from it, and the
  chord is that distance;
- beta is the angle in degrees between the chord line and the rotation plane,
  positive when the leading edge sits ahead along the thrust axis (nose up into
  the oncoming relative flow);
- the shape is expressed in unit-chord coordinates with x running from the
  leading edge (0) to the trailing edge (1) and y positive on the thrust side,
  ordered from the trailing edge over the upper side and back under the lower.

Only the standard library is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec = tuple[float, float, float]
AXES: dict[str, Vec] = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
SPACINGS = ("cosine", "uniform")
CUTS = ("cylinder", "plane")
ROTATIONS = ("ccw_from_front", "cw_from_front")


class BladeSliceError(ValueError):
    """The blade cannot be sliced with the requested convention."""


@dataclass(frozen=True)
class BladeSection:
    """One radial station: radius and chord in model units, beta in degrees."""

    radius: float
    chord: float
    beta_deg: float
    shape: tuple[tuple[float, float], ...]


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


def parse_obj(text: str) -> tuple[list[Vec], list[tuple[int, int, int]]]:
    """Read vertices and triangulated faces from Wavefront OBJ text.

    Parameters
    ----------
    text : str
        OBJ content. Only ``v`` and ``f`` records are read; polygons are fanned
        into triangles and negative indices count back from the last vertex.

    Returns
    -------
    tuple
        Vertex list and zero-based triangle index triples.
    """
    vertices: list[Vec] = []
    faces: list[tuple[int, int, int]] = []
    for number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "v":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                ids = []
                for token in parts[1:]:
                    index = int(token.split("/")[0])
                    ids.append(index - 1 if index > 0 else len(vertices) + index)
                if len(ids) < 3 or any(not 0 <= i < len(vertices) for i in ids):
                    raise ValueError("face needs three valid vertex references")
                faces.extend((ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1))
        except (ValueError, IndexError) as error:
            raise BladeSliceError(f"OBJ line {number}: cannot read {line.strip()!r}") from error
    return vertices, faces


def station_radii(r_min: float, r_max: float, count: int, spacing: str) -> list[float]:
    """Return ``count`` cell-centred radii between the extremes, never on the ends.

    Parameters
    ----------
    r_min, r_max : float
        Inner and outer radius of the blade.
    count : int
        Number of stations.
    spacing : str
        ``"uniform"`` or ``"cosine"`` (clustered toward both ends).
    """
    if spacing not in SPACINGS:
        raise BladeSliceError(f"spacing must be one of: {', '.join(SPACINGS)}")
    if count < 1:
        raise BladeSliceError("n_sections must be a positive integer")
    result = []
    for i in range(count):
        fraction = (i + 0.5) / count
        if spacing == "cosine":
            fraction = 0.5 * (1.0 - math.cos(math.pi * fraction))
        result.append(r_min + (r_max - r_min) * fraction)
    return result


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


def radial_extent(
    obj_text: str, *, axis: str = "x", hub_center: Vec = (0.0, 0.0, 0.0)
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
    """
    if axis not in AXES:
        raise BladeSliceError(f"axis must be one of: {', '.join(AXES)}")
    vertices, faces = parse_obj(obj_text)
    if not faces:
        raise BladeSliceError("OBJ has no faces")
    a = AXES[axis]
    radii = []
    for i in {i for face in _largest_component(faces) for i in face}:
        q = _sub(vertices[i], hub_center)
        radii.append(math.sqrt(max(_dot(q, q) - _dot(q, a) ** 2, 0.0)))
    return min(radii), max(radii)


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
