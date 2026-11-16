"""Read full-precision polynomial geometry from a single native BREP face.

The adapter writes this kernel-owned representation for measurement only.
Unsupported versions, locations and geometry types fail closed. No IGES
rounding, surface fitting, or point sampling supplies the area certificate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from cfdbender.geometry._geom_area_domain import _beziers, _refuse


@dataclass
class _Curve:
    degree: int
    knots: list[float]
    points: np.ndarray
    weights: list[float]
    line: bool = False


class _Tokens:
    def __init__(self, text: str) -> None:
        self.values = iter(text.split())

    def number(self) -> float:
        return float(next(self.values))

    def integer(self) -> int:
        return int(next(self.values))

    def numbers(self, count: int) -> list[float]:
        return [self.number() for _ in range(count)]


def _edge_pieces(curve: _Curve, low: float, high: float, reverse: bool) -> list[_Curve]:
    """Restrict on native knots before reversing; normalize each exact Bezier span."""
    if curve.line:
        points = np.array([curve.points[0] + t * curve.points[1] for t in (low, high)])
        pieces = [_Curve(1, [0, 0, 1, 1], points, [1, 1])]
    else:
        weights = np.asarray(curve.weights)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise _refuse("native curve weights must be finite and positive")
        homogeneous = np.column_stack([curve.points * weights[:, None], weights])
        spline = BSpline(curve.knots, homogeneous, curve.degree)
        for value in (low, high):
            count = int(np.count_nonzero(spline.t == value))
            if count < curve.degree:
                spline = spline.insert_knot(value, curve.degree - count)
        knots, poles = _beziers(np.asarray(spline.t), curve.degree, np.asarray(spline.c))
        values = np.unique(knots)
        pieces = []
        for a, b in zip(values[:-1], values[1:], strict=True):
            if a < low or b > high:
                continue
            span = int(np.searchsorted(knots, a, side="right") - 1)
            net = poles[span - curve.degree : span + 1]
            pieces.append(
                _Curve(
                    curve.degree,
                    [0.0] * (curve.degree + 1) + [1.0] * (curve.degree + 1),
                    net[:, :2] / net[:, 2, None],
                    net[:, 2].tolist(),
                )
            )
    if reverse:
        pieces = [_Curve(c.degree, c.knots, c.points[::-1], c.weights[::-1]) for c in pieces[::-1]]
    return pieces


def _curve(tokens: _Tokens, dimension: int) -> _Curve:
    kind = tokens.integer()
    if kind == 8:
        tokens.numbers(2)  # Native trimmed-curve wrapper; edge ranges govern below.
        return _curve(tokens, dimension)
    if kind == 1:
        return _Curve(
            1, [], np.array(tokens.numbers(2 * dimension)).reshape(2, dimension), [], True
        )
    if kind != 7:
        raise _refuse(f"native curve type {kind} has no polynomial certificate")
    rational, periodic = tokens.integer(), tokens.integer()
    degree, poles, count = tokens.integer(), tokens.integer(), tokens.integer()
    if periodic:
        raise _refuse("periodic native curves require a seam certificate")
    rows = np.array(tokens.numbers(poles * (dimension + rational))).reshape(
        poles, dimension + rational
    )
    points, weights = (
        rows[:, :dimension],
        rows[:, dimension].tolist() if rational else [1.0] * poles,
    )
    knots = []
    for _ in range(count):
        knot, multiplicity = tokens.number(), tokens.integer()
        knots.extend([knot] * multiplicity)
    return _Curve(degree, knots, points, weights)


def _surface(tokens: _Tokens) -> dict[str, Any]:
    kind = tokens.integer()
    if kind == 1:
        values = np.array(tokens.numbers(12)).reshape(4, 3)
        return {"plane": values}
    if kind != 9:
        raise _refuse(f"native surface type {kind} has no polynomial certificate")
    flags = [tokens.integer() for _ in range(4)]
    if any(flags):
        raise _refuse("rational or periodic native surfaces require a separate certificate")
    pu, pv, nu, nv, nku, nkv = [tokens.integer() for _ in range(6)]
    points = np.array(tokens.numbers(3 * nu * nv)).reshape(nu, nv, 3)
    knots = []
    for count in (nku, nkv):
        vector = []
        for _ in range(count):
            value, multiplicity = tokens.number(), tokens.integer()
            vector.extend([value] * multiplicity)
        knots.append(vector)
    return {"points": points, "knots": knots, "degrees": [pu, pv]}


def _section(text: str, name: str, end: str) -> tuple[int, str]:
    match = re.search(rf"(?m)^{name} (\d+)\s*$", text)
    following = re.search(rf"(?m)^{end} \d+\s*$", text)
    if match is None or following is None or following.start() <= match.end():
        raise _refuse(f"missing native {name} section")
    return int(match[1]), text[match.end() : following.start()]


def _references(text: str) -> list[tuple[int, int]]:
    return [
        (int(shape), int(location))
        for shape, location in re.findall(r"([+-]\d+)\s+(\d+)", text.split("*")[0])
    ]


def _wire_order(
    edges: list[tuple[int, int]],
    shapes: dict[int, tuple[str, str, list[tuple[int, int]]]],
) -> list[int]:
    """Follow oriented kernel vertices; BREP storage need not follow the wire."""
    connections = []
    for edge, _ in edges:
        vertices = shapes[abs(edge)][2]
        if len(vertices) != 2 or any(
            location or shapes[abs(vertex)][0] != "Ve" for vertex, location in vertices
        ):
            raise _refuse("native edge needs exactly two unlocated topological vertex references")
        starts = [v for v, _ in vertices if v > 0]
        ends = [-v for v, _ in vertices if v < 0]
        if len(starts) != 1 or len(ends) != 1:
            raise _refuse("native edge vertex orientations are ambiguous")
        start, end = starts[0], ends[0]
        connections.append((start, end) if edge > 0 else (end, start))
    if not connections:
        raise _refuse("native wire has no oriented edges")
    order = [0]
    remaining = set(range(1, len(edges)))
    while remaining:
        following = [i for i in remaining if connections[i][0] == connections[order[-1]][1]]
        if len(following) != 1:
            raise _refuse("native wire has disconnected or ambiguous vertex topology")
        order.append(following[0])
        remaining.remove(following[0])
    if connections[order[-1]][1] != connections[0][0]:
        raise _refuse("native wire vertex topology does not close")
    return order


def brep_definition(text: str) -> dict[str, Any]:
    """Return a full-precision measurement document for one native face."""
    if "CASCADE Topology V1," not in text or not re.search(r"(?m)^Locations 0\s*$", text):
        raise _refuse("native BREP version or located geometry is unsupported")
    count, payload = _section(text, "Curve2ds", "Curves")
    tokens = _Tokens(payload)
    curves = [_curve(tokens, 2) for _ in range(count)]
    count, payload = _section(text, "Surfaces", "Triangulations")
    tokens = _Tokens(payload)
    surfaces = [_surface(tokens) for _ in range(count)]
    count, topology = _section(text + "\nEnd 0\n", "TShapes", "End")
    matches = list(re.finditer(r"(?m)^(Ve|Ed|Wi|Fa|Sh|So|Cs|Co)\s*$", topology))
    if len(matches) != count:
        raise _refuse("native topology count is inconsistent")
    shapes = {}
    for index, match in enumerate(matches):
        block = topology[
            match.end() : matches[index + 1].start() if index + 1 < count else len(topology)
        ]
        flags = re.search(r"(?m)^[01]{7}\s*$", block)
        if flags is None:
            raise _refuse("native shape flags are absent")
        shapes[count - index] = (
            match[1],
            block[: flags.start()].strip(),
            _references(block[flags.end() :]),
        )
    faces = [v for v in shapes.values() if v[0] == "Fa"]
    if len(faces) != 1:
        raise _refuse(f"native face count {len(faces)} differs from limit 1")
    _, data, wires = faces[0]
    fields = data.split()
    if len(fields) != 4 or int(fields[3]):
        raise _refuse("native face location is unsupported")
    surface_index = int(fields[2])
    surface = surfaces[surface_index - 1]
    space_curves = []
    if "plane" in surface:
        count, payload = _section(text, "Curves", "Polygon3D")
        tokens = _Tokens(payload)
        space_curves = [_curve(tokens, 3) for _ in range(count)]
    loops = []
    for wire, location in wires:
        if location or shapes[abs(wire)][0] != "Wi":
            raise _refuse("native wire location or type is unsupported")
        edges = shapes[abs(wire)][2]
        if wire < 0:
            edges = [(-edge, loc) for edge, loc in edges[::-1]]
        edge_pieces: list[list[tuple[_Curve, float, float]]] = []
        for edge, location in edges:
            if location or shapes[abs(edge)][0] != "Ed":
                raise _refuse("native edge location or type is unsupported")
            records = [
                line.split() for line in shapes[abs(edge)][1].splitlines()[1:] if line.strip()
            ]
            candidates = [r for r in records if r[0] == "2" and int(r[2]) == surface_index]
            if any(r[0] == "3" for r in records):
                raise _refuse("native seam edges require a periodic certificate")
            if candidates:
                if len(candidates) != 1 or int(candidates[0][3]):
                    raise _refuse("native edge has ambiguous pcurves or locations")
                record = candidates[0]
                curve = curves[int(record[1]) - 1]
            elif "plane" in surface:
                candidates = [r for r in records if r[0] == "1"]
                if len(candidates) != 1 or int(candidates[0][2]):
                    raise _refuse("planar edge has no unique unlocated space curve")
                record = candidates[0]
                original = space_curves[int(record[1]) - 1]
                origin, _, xaxis, yaxis = surface["plane"]
                points = original.points.copy()
                points[0] -= origin
                if not original.line:
                    points[1:] -= origin
                points = np.stack([points @ xaxis, points @ yaxis], axis=-1)
                curve = _Curve(
                    original.degree, original.knots, points, original.weights, original.line
                )
            else:
                raise _refuse("native curved edge has no pcurve on the measured surface")
            low, high = float(record[-2]), float(record[-1])
            if not np.isfinite([low, high]).all() or low >= high:
                raise _refuse("native edge has an invalid parameter range")
            if not curve.line and not (
                curve.knots[curve.degree] <= low < high <= curve.knots[-curve.degree - 1]
            ):
                start, end = curve.knots[curve.degree], curve.knots[-curve.degree - 1]
                overshoot = max(start - low, high - end, 0.0)
                if overshoot > 2e-12 * (end - start) or not np.all(
                    np.asarray(curve.weights) == curve.weights[0]
                ):
                    raise _refuse(
                        f"native edge range [{low}, {high}] exceeds represented range "
                        f"[{start}, {end}]"
                    )
                # Native endpoints can differ by a rounding unit. Restrict
                # them only after bounding the omitted polynomial extension
                # inside the integration's explicit UV reconstruction strip.
                t, poles = _beziers(np.asarray(curve.knots), curve.degree, curve.points)
                step = float(np.diff(np.unique(t)).min())
                slope = curve.degree * float(np.abs(np.diff(poles, axis=0)).max()) / step
                extension = overshoot * slope * (1 + 2 * overshoot / step) ** (curve.degree - 1)
                allowance = 2e-12 * max(1.0, float(np.abs(poles).max()))
                if not np.isfinite(extension) or extension > allowance:
                    raise _refuse(
                        f"native edge range error {extension} UV exceeds limit {allowance} UV"
                    )
                low, high = max(low, start), min(high, end)
                if low >= high:
                    raise _refuse("native edge range has no represented interval")
            edge_pieces.append(
                [(member, 0.0, 1.0) for member in _edge_pieces(curve, low, high, edge < 0)]
            )
        loop = [member for i in _wire_order(edges, shapes) for member in edge_pieces[i]]
        loops.append(loop)
    if "plane" in surface:
        points = np.concatenate([c.points for loop in loops for c, _, _ in loop])
        lower, upper = points.min(axis=0), points.max(axis=0)
        if np.any(lower >= upper):
            raise _refuse("native planar face has an empty parameter domain")
        origin, _, xaxis, yaxis = surface["plane"]
        net = np.array(
            [
                [origin + u * xaxis + v * yaxis for v in (lower[1], upper[1])]
                for u in (lower[0], upper[0])
            ]
        )
        surface = {
            "points": net,
            "degrees": [1, 1],
            "knots": [[lower[0]] * 2 + [upper[0]] * 2, [lower[1]] * 2 + [upper[1]] * 2],
        }
    entities: list[dict[str, Any]] = []

    def entity(kind: int, values: list[Any]) -> int:
        ident = 2 * len(entities) + 1
        entities.append(
            {"id": ident, "type": kind, "parameters": [str(v) for v in values], "d1": ["0"] * 9}
        )
        return ident

    pu, pv = surface["degrees"]
    net, (ku, kv) = surface["points"], surface["knots"]
    nu, nv = net.shape[:2]
    base = entity(
        128,
        [
            128,
            nu - 1,
            nv - 1,
            pu,
            pv,
            0,
            0,
            1,
            0,
            0,
            *ku,
            *kv,
            *([1] * (nu * nv)),
            *net.swapaxes(0, 1).reshape(-1),
            ku[pu],
            ku[-pu - 1],
            kv[pv],
            kv[-pv - 1],
        ],
    )
    boundaries = []
    for loop in loops:
        members = []
        for curve, low, high in loop:
            xyz = np.column_stack([curve.points, np.zeros(len(curve.points))])
            members.append(
                entity(
                    126,
                    [
                        126,
                        len(xyz) - 1,
                        curve.degree,
                        0,
                        0,
                        1,
                        0,
                        *curve.knots,
                        *curve.weights,
                        *xyz.reshape(-1),
                        low,
                        high,
                        0,
                        0,
                        1,
                    ],
                )
            )
        composite = entity(102, [102, len(members), *members])
        boundaries.append(entity(142, [142, 1, base, composite, 0, 1]))
    entity(144, [144, base, 1, len(boundaries) - 1, boundaries[0], *boundaries[1:]])
    return {"global": [""] * 13 + ["6"], "entities": entities}
