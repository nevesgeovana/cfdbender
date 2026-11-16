"""The measured checks of the junction sub-steps, and the pure geometry they measure with.

Pipeline role: pure part of stage S10 of the geometry pipeline. The
assembly of the kernel stages runs the sub-steps (load, pieces alone, the cut,
the projection, the split, the removal, the write and the sewing) and hands the numbers it
measured to the functions here, which compute distances, chains, inside tests, the flood of a
removal and the verdict of each named check. Nothing here imports a kernel.

The dominant imprints on the dominated, the dominated loses its inside, and the dominant is
split along the junction too. No area constant
decides a removal: the faces a piece keeps are those its seeds reach without crossing a junction.

Every limit below carries the sentence that says where it comes from. They are lengths of the
geometry's own tolerances (half the sewing tolerance, the reader's resolution), not values of any
aircraft.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError

from ._records import check_entry

__all__ = [
    "CHAIN_END_GAP_M",
    "CREASE_GAP_M",
    "EDGE_ON_SURFACE_M",
    "GROUP_FUZZY_M",
    "JUNCTION_GAP_M",
    "J_END_ON_CUT_M",
    "J_LOOP_CLOSURE_M",
    "J_ON_LOOP_M",
    "KEPT_ON_LOOP_M",
    "LOOP_CLOSURE_M",
    "MIN_JUNCTION_FRACTION",
    "OPENING_M",
    "REGENERATE_M",
    "RIM_MARGIN_M",
    "SEW_FINE_M",
    "SEW_M",
    "SEW_WIDE_M",
    "SHARED_PAIR_END_M",
    "SHARED_PAIR_NEAR_M",
    "SLIVER_WIDTH_M",
    "SYMMETRY_M",
    "WRITTEN_PAIR_M",
    "ZERO_WIDTH_M",
    "Operation",
    "chain",
    "classify_free_edges",
    "curve_distance",
    "distance_to_polyline",
    "distance_to_triangles",
    "dominance_plan",
    "dominant_inside",
    "drop_zero_width",
    "flood",
    "introduced",
    "lonely_ends",
    "loop_closure",
    "mean_width",
    "measured",
    "min_junction",
    "mirror_match",
    "pair_curves",
    "parity_inside",
    "plane_offset",
    "points_outside",
    "pole_change",
    "ray_hits",
    "polyline_length",
    "slivers",
]

#: The sewing tolerance of the assembly checks, 0.01 mm: the reader's own resolution of written
#: junctions as measured on the reference configurations.
SEW_M = 1e-5
#: The second, wider sewing of the written file, 0.05 mm.
SEW_WIDE_M = 5e-5
#: The finer sewing of a piece alone and of a join, 0.001 mm.
SEW_FINE_M = 1e-6
#: A junction curve lies on both its surfaces within half the sewing tolerance.
JUNCTION_GAP_M = SEW_M / 2
#: A body join's crease lies on both bodies within 0.001 mm.
CREASE_GAP_M = 1e-6
#: Two junction chain ends closer than 0.02 mm are one end.
CHAIN_END_GAP_M = 2e-5
#: The curves of a rebuilt loop meet within half the sewing tolerance.
LOOP_CLOSURE_M = SEW_M / 2
#: A free edge within 0.02 mm of a partner is the opening that partner closes.
OPENING_M = 2e-5
#: A projected junction lies on its host's loops within half the sewing tolerance.
J_ON_LOOP_M = SEW_M / 2
#: The loops of a split host face close within 0.02 mm.
J_LOOP_CLOSURE_M = 2e-5
#: An open end of a projected junction lies on the cut's junction within 0.05 mm.
J_END_ON_CUT_M = 5e-5
#: The dominant's edges lie on the loops a host keeps within 0.01 mm.
KEPT_ON_LOOP_M = 1e-5
#: A face edge lies on the face's own surface within 0.05 mm.
EDGE_ON_SURFACE_M = 5e-5
#: Curves of two pieces within 0.1 mm of each other are a shared pair.
SHARED_PAIR_NEAR_M = 1e-4
#: The ends of a shared pair meet within 0.2 mm.
SHARED_PAIR_END_M = 2e-4
#: A written shared pair is one curve: no measurable distance (identical tokens).
WRITTEN_PAIR_M = 1e-9
#: The edges of a half model lie on its symmetry plane within 0.001 mm.
SYMMETRY_M = 1e-6
#: A piece regenerated from its own inputs moves no pole by more than 1e-6 mm.
REGENERATE_M = 1e-9
#: A rim point within 0.01 mm of the dominant's surface counts as on it.
RIM_MARGIN_M = 1e-5
#: A face narrower than 5 mm on average is a sliver. Those the
#: pieces bring alone are counted and listed, not refused (generator thin faces);
#: a sliver the assembly introduces fails ``sew.slivers`` (none allowed).
SLIVER_WIDTH_M = 5e-3
#: A region of a split face narrower than 0.05 mm on average has no area worth a face; it is
#: dropped and counted.
ZERO_WIDTH_M = 5e-5
#: A junction shorter than this fraction of the smaller piece's largest extent means the pieces
#: of a cut do not intersect. Relative, so the library builds no length in and a model is judged
#: alike at any scale: 100 mm per pair on a piece about 2.5 m across, where
#: the 10 mm junction it refused lies a factor of ten below.
MIN_JUNCTION_FRACTION = 0.04
#: The fuzzy value of a group fragment, 0.05 mm.
GROUP_FUZZY_M = 5e-5


def measured(
    check_id: str,
    value: float,
    limit: float,
    *,
    unit: str,
    cause: str,
    bound: str = "max",
    where: Mapping[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Return the verdict of one measured check, its class and cause named when it fails.

    ``cause`` starts with the class a failure is read as, ``STRATEGY`` (the operation cannot
    give the result even with correct code) or ``BUG`` (the operation is right, its inputs pass,
    and its output violates its own definition), and says in one line what a failure means.
    """
    value = float(value)
    ok = value <= limit if bound == "max" else value >= limit
    text = note if ok else "; ".join(p for p in (note, f"class {cause}") if p)
    return check_entry(
        check_id,
        "passed" if ok else "failed",
        value,
        float(limit),
        unit=unit,
        where=dict(where) if where else None,
        note=text,
        bound=bound,
    )


# ---- curves ------------------------------------------------------------------------------


def polyline_length(points: np.ndarray) -> float:
    """Return the length of the polyline through ``points``."""
    p = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Return the distance of each point to the polyline, point to segment (never to a sample)."""
    p = np.asarray(points, float).reshape(-1, 3)
    line = np.asarray(polyline, float).reshape(-1, 3)
    if len(line) == 1:
        return np.linalg.norm(p - line[0], axis=1)
    a, b = line[:-1], line[1:]
    ab = b - a
    length2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    out = np.empty(len(p))
    for start in range(0, len(p), 256):
        chunk = p[start : start + 256, None, :]
        t = np.clip(np.einsum("nij,ij->ni", chunk - a, ab) / length2, 0.0, 1.0)
        nearest = a + t[..., None] * ab
        out[start : start + 256] = np.linalg.norm(chunk - nearest, axis=2).min(axis=1)
    return out


def curve_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the largest distance of either sampled curve to the other one."""
    return float(max(distance_to_polyline(a, b).max(), distance_to_polyline(b, a).max()))


def chain(
    curves: Sequence[tuple[Sequence[float], Sequence[float]]], gap: float
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Chain curves end to end and return the chains and the ends that meet no other end.

    ``curves`` holds ``(start, end)`` of each curve. Two ends within ``gap`` are joined; an end
    left alone is returned as ``(curve, 0 or 1)``. A closed chain has no lonely end.
    """
    ends = [np.asarray(e, float) for c in curves for e in c]
    count = len(ends)
    parent = list(range(len(curves)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lonely: list[tuple[int, int]] = []
    for i in range(count):
        others = [j for j in range(count) if j != i]
        distances = [float(np.linalg.norm(ends[i] - ends[j])) for j in others]
        near = [j for j, d in zip(others, distances, strict=True) if d <= gap]
        if not near:
            lonely.append((i // 2, i % 2))
        for j in near:
            parent[root(i // 2)] = root(j // 2)
    groups: dict[int, list[int]] = {}
    for k in range(len(curves)):
        groups.setdefault(root(k), []).append(k)
    return list(groups.values()), lonely


def lonely_ends(
    curves: Sequence[tuple[Sequence[float], Sequence[float]]],
    lonely: Sequence[tuple[int, int]],
    tags: Sequence[int],
) -> dict[str, Any]:
    """Describe the lonely ends of a chain for a failure record: where each is, how far it is.

    ``curves`` and ``lonely`` are the input and the second result of :func:`chain`; ``tags``
    names each curve. Each end carries its curve tag, its side, its point and the distance to
    the nearest other end of the chain, so a record read later shows why the chain stops.
    """
    ends = [np.asarray(e, float) for c in curves for e in c]
    described = []
    for curve, side in lonely:
        i = 2 * curve + side
        gaps = [float(np.linalg.norm(ends[i] - ends[j])) for j in range(len(ends)) if j != i]
        described.append(
            {
                "curve": int(tags[curve]),
                "side": int(side),
                "xyz_m": [float(v) for v in ends[i]],
                "nearest_m": min(gaps, default=None),
            }
        )
    return {"curves": [int(t) for t in tags], "count": len(tags), "lonely_ends": described}


def loop_closure(loop: Sequence[tuple[Sequence[float], Sequence[float]]]) -> float:
    """Return the largest gap between the end of a curve and the start of the next one."""
    gaps = [
        float(np.linalg.norm(np.subtract(loop[k][1], loop[(k + 1) % len(loop)][0])))
        for k in range(len(loop))
    ]
    return max(gaps) if gaps else 0.0


def pair_curves(
    curves: Sequence[tuple[str, np.ndarray]], *, near: float, end_gap: float
) -> list[tuple[int, int, float]]:
    """Return the pairs of curves of two pieces that join the same two vertices and lie together.

    ``curves`` holds ``(piece, samples)``. A pair has both ends within ``end_gap`` (in either
    sense) and every sample of each within ``near`` of the other; its third item is that largest
    distance, the gap before the two are made one curve.
    """
    pairs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            (pa, a), (pb, b) = curves[i], curves[j]
            if pa == pb:
                continue
            same = max(np.linalg.norm(a[0] - b[0]), np.linalg.norm(a[-1] - b[-1]))
            crossed = max(np.linalg.norm(a[0] - b[-1]), np.linalg.norm(a[-1] - b[0]))
            if min(same, crossed) > end_gap:
                continue
            gap = curve_distance(a, b)
            if gap <= near:
                pairs.append((i, j, gap))
    return pairs


def plane_offset(edges: Iterable[np.ndarray], axis: int) -> float:
    """Return the largest distance of the edges' samples from the plane ``axis`` = 0."""
    values = [float(np.abs(np.asarray(e, float)[:, axis]).max()) for e in edges]
    return max(values) if values else 0.0


def mirror_match(left: Sequence[np.ndarray], right: Sequence[np.ndarray], axis: int) -> float:
    """Return how far the edges of one half lie from the mirrored edges of the other half."""
    flip = np.ones(3)
    flip[axis] = -1.0
    mirrored = [np.asarray(e, float) * flip for e in right]
    if not left or not mirrored:
        return 0.0
    return max(
        float(min(distance_to_polyline(e, m).max() for m in mirrored))
        for e in (np.asarray(x, float) for x in left)
    )


def pole_change(first: Sequence[Any], again: Sequence[Any]) -> float:
    """Return the largest pole move between two builds of a piece, patch by patch.

    Parameters
    ----------
    first : sequence of Patch
        Patches of the original build, with control points in meters.
    again : sequence of Patch
        Patches of the repeated build, in the same order and coordinate frame.

    Returns
    -------
    float
        Largest control-point displacement in meters.

    Raises
    ------
    InputValueError
        If ``first`` and ``again`` have different patch counts or control-net shapes.
    """
    if len(first) != len(again):
        raise InputValueError(
            f"first and again pole nets differ: {len(first)} patches against {len(again)}; "
            "compare builds with matching patch counts and control-net shapes"
        )
    change = 0.0
    for a, b in zip(first, again, strict=True):
        pa, pb = np.asarray(a.controls, float), np.asarray(b.controls, float)
        if pa.shape != pb.shape:
            raise InputValueError(
                f"first and again pole nets differ: {pa.shape} against {pb.shape}; "
                "compare builds with matching patch counts and control-net shapes"
            )
        change = max(change, float(np.linalg.norm(pa - pb, axis=-1).max()))
    return change


# ---- triangles: distance and inside --------------------------------------------------------


def _closest_on_triangles(p: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, for one point, the closest point of each triangle (Ericson's regions)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp, cp = p - b, p - c
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) > 0, va + vb + vc, 1.0)
    v, w = vb / denom, vc / denom
    out = a + ab * v[:, None] + ac * w[:, None]
    safe = lambda x: np.where(np.abs(x) > 0, x, 1.0)  # noqa: E731
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out[edge_ab] = (a + ab * (d1 / safe(d1 - d3))[:, None])[edge_ab]
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out[edge_ac] = (a + ac * (d2 / safe(d2 - d6))[:, None])[edge_ac]
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / safe((d4 - d3) + (d5 - d6))
    out[edge_bc] = (b + (c - b) * t[:, None])[edge_bc]
    out[(d1 <= 0) & (d2 <= 0)] = a[(d1 <= 0) & (d2 <= 0)]
    out[(d3 >= 0) & (d4 <= d3)] = b[(d3 >= 0) & (d4 <= d3)]
    out[(d6 >= 0) & (d5 <= d6)] = c[(d6 >= 0) & (d5 <= d6)]
    return out, np.linalg.norm(out - p, axis=1)


def distance_to_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return the distance of each point to the nearest of the triangles."""
    tris = np.asarray(triangles, float).reshape(-1, 3, 3)
    return np.array(
        [_closest_on_triangles(p, tris)[1].min() for p in np.asarray(points, float).reshape(-1, 3)]
    )


#: Three ray directions, tilted off the axes so a ray grazes no edge of an axis-aligned hull.
_RAYS = np.array([[1e-3, 2e-3, 1.0], [1.0, 1e-3, 2.3e-3], [1.7e-3, 1.0, 1.1e-3]])


def _crossings(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> int:
    """Return how many triangles the ray from ``origin`` crosses (Moeller and Trumbore)."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - tris[:, 0]
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * (q @ direction)
    t = inv * np.einsum("ij,ij->i", e2, q)
    return int(np.count_nonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)))


def ray_hits(point: np.ndarray, direction: np.ndarray, triangles: np.ndarray) -> bool:
    """Whether the line through ``point`` along ``direction``, either sense, meets a triangle."""
    tris = np.asarray(triangles, float).reshape(-1, 3, 3)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    p = np.asarray(point, float)
    return _crossings(p, d, tris) > 0 or _crossings(p, -d, tris) > 0


def parity_inside(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Whether each point is inside the closed hull: odd crossings, by the vote of three rays."""
    tris = np.asarray(triangles, float).reshape(-1, 3, 3)
    out = []
    for p in np.asarray(points, float).reshape(-1, 3):
        votes = sum(_crossings(p, d / np.linalg.norm(d), tris) % 2 for d in _RAYS)
        out.append(votes >= 2)
    return np.array(out, bool)


#: Outward normals already computed, by the bytes of the triangles and the hull; a run asks the
#: same dominant thousands of times, and each answer costs a parity test per triangle.
_NORMALS: dict[int, np.ndarray] = {}


def _outward_normals(tris: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Return each triangle's unit normal turned out of the hull (parity of a nudged centroid)."""
    key = hash((tris.tobytes(), hull.tobytes()))
    if key not in _NORMALS:
        if len(_NORMALS) > 64:
            _NORMALS.clear()
        _NORMALS[key] = _normals_out(tris, hull)
    return _NORMALS[key]


def _normals_out(tris: np.ndarray, hull: np.ndarray) -> np.ndarray:
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
    size = float(np.ptp(hull.reshape(-1, 3), axis=0).max())
    nudged = tris.mean(axis=1) + normal * size * 1e-6
    inward = parity_inside(nudged, hull)
    normal[inward] *= -1.0
    return normal


def dominant_inside(
    points: np.ndarray, current: np.ndarray, hull: np.ndarray, *, near: float
) -> np.ndarray:
    """Whether each point lies inside a dominant, by its current faces with the parity fallback.

    A point within ``near`` of the dominant's current faces is decided by the outward normal at
    its closest point there. Where the faces next to a point are gone (removed as inside another
    piece), that closest point is far and its normal orients nothing, so beyond ``near`` the
    point is decided by the parity of the dominant's whole hull ``hull``.
    """
    tris = np.asarray(current, float).reshape(-1, 3, 3)
    whole = np.asarray(hull, float).reshape(-1, 3, 3)
    normals = _outward_normals(tris, whole)
    out = []
    for p in np.asarray(points, float).reshape(-1, 3):
        closest, distance = _closest_on_triangles(p, tris)
        k = int(np.argmin(distance))
        if distance[k] <= near:
            out.append(bool(np.dot(p - closest[k], normals[k]) < 0.0))
        else:
            out.append(bool(parity_inside(p[None], whole)[0]))
    return np.array(out, bool)


def points_outside(points: np.ndarray, hull: np.ndarray, *, margin: float) -> int:
    """Return how many points are outside the hull by more than ``margin``."""
    p = np.asarray(points, float).reshape(-1, 3)
    inside = parity_inside(p, hull)
    distance = distance_to_triangles(p, hull)
    return int(np.count_nonzero(~inside & (distance > margin)))


def classify_free_edges(
    edges: Sequence[np.ndarray],
    *,
    distance: Callable[[np.ndarray], np.ndarray],
    inside: Callable[[np.ndarray], np.ndarray],
    plane_axis: int | None,
) -> list[str]:
    """Return the class of each free edge: symmetry, opening, buried or unexplained.

    An edge on the symmetry plane (when the model has one) is ``symmetry``; one whose samples all
    lie within :data:`OPENING_M` of a partner is the ``opening`` that partner closes; one whose
    samples all lie inside a partner is ``buried``; anything else is ``unexplained``.
    ``distance`` and ``inside`` measure against the partners of the edge's piece.
    """
    classes = []
    for edge in edges:
        samples = np.asarray(edge, float)
        if plane_axis is not None and np.abs(samples[:, plane_axis]).max() <= SYMMETRY_M:
            classes.append("symmetry")
        elif float(np.max(distance(samples))) <= OPENING_M:
            classes.append("opening")
        elif bool(np.all(inside(samples))):
            classes.append("buried")
        else:
            classes.append("unexplained")
    return classes


# ---- removal, slivers, introduced edges ----------------------------------------------------


def flood(
    neighbours: Mapping[Hashable, Iterable[tuple[Hashable, Hashable]]],
    *,
    seeds: Iterable[Hashable],
    barriers: set[Hashable],
) -> set[Hashable]:
    """Return the faces reached from the seeds through shared edges that are not barriers.

    ``neighbours[face]`` lists ``(edge, other face)``. The barriers are the junction curves, so a
    piece keeps what its own exposed side reaches and loses what a junction encloses.
    """
    kept = set(seeds)
    queue = deque(kept)
    while queue:
        face = queue.popleft()
        for edge, other in neighbours.get(face, ()):
            if edge in barriers or other in kept:
                continue
            kept.add(other)
            queue.append(other)
    return kept


def mean_width(area: float, perimeter: float) -> float:
    """Return a face's mean width, twice its area over its perimeter (a strip of w by l: w)."""
    return 2.0 * float(area) / float(perimeter) if perimeter > 0 else 0.0


def drop_zero_width(faces: Sequence[Any], widths: Sequence[float]) -> tuple[list[Any], list[Any]]:
    """Return the faces kept and those dropped as narrower than :data:`ZERO_WIDTH_M`."""
    keep = [f for f, w in zip(faces, widths, strict=True) if w >= ZERO_WIDTH_M]
    dropped = [f for f, w in zip(faces, widths, strict=True) if w < ZERO_WIDTH_M]
    return keep, dropped


def slivers(widths: Sequence[float]) -> list[int]:
    """Return the faces narrower than :data:`SLIVER_WIDTH_M`, by index (listed, never refused)."""
    return [k for k, w in enumerate(widths) if w < SLIVER_WIDTH_M]


def min_junction(*extents: float) -> float:
    """Return the shortest junction two pieces may have, from their largest extents in metres.

    It is :data:`MIN_JUNCTION_FRACTION` of the smaller extent, so the limit scales with the
    pieces it judges.
    """
    return MIN_JUNCTION_FRACTION * float(min(extents))


def introduced(total: int, inputs: int) -> int:
    """Return the free edges an assembly introduced: its total less those the pieces brought."""
    return max(0, int(total) - int(inputs))


# ---- the dominance table, applied to the members of one configuration ----------------------


@dataclass(frozen=True)
class Operation:
    """One pair of a dominance row: who imprints on whom, and how."""

    operation: str
    dominant: str
    dominated: str
    row: int
    tolerance_m: float = 5e-5
    contact_deg: float = 10.0


def dominance_plan(
    rows: Sequence[Mapping[str, Any]], members: Sequence[str], warnings: list[str]
) -> list[Operation]:
    """Return the operations of the dominance rows on ``members``: cuts first, then projections.

    A row naming a component that is not a member of the configuration is warned about and its
    pairs with that name skipped. Cuts run before projections whatever the order of the rows,
    because a projected footprint closes on the cut's junction (a projection
    before the cut trimmed the host off the fillet's edge).
    """
    present = set(members)
    cuts: list[Operation] = []
    projects: list[Operation] = []
    for index, row in enumerate(rows):
        absent = [
            n for n in [*_names(row["dominant"]), *_names(row["dominated"])] if n not in present
        ]
        for name in dict.fromkeys(absent):
            warnings.append(
                f"dominance[{index}]: {name} is not a member of this configuration, row skipped"
            )
        for dominant in _names(row["dominant"]):
            for dominated in _names(row["dominated"]):
                if dominant not in present or dominated not in present:
                    continue
                op = Operation(
                    str(row.get("operation", "auto")),
                    dominant,
                    dominated,
                    index,
                    float(row.get("tolerance_m", 5e-5)),
                    float(row.get("contact_deg", 10.0)),
                )
                (projects if op.operation == "project" else cuts).append(op)
    return cuts + projects


def _names(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else [str(v) for v in value]
