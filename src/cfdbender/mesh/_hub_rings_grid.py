"""The geometry of hub rings around a blade root: march, hole, annulus.

Every step works in a chart of the hub around one root: :class:`._repitch.RingChart` with no
rotor frame, the plane of the patch's largest spread, because ``[hub_rings]`` takes no axis.
Rings are marched outward from the root loop in that chart, normal to the loop, with the normals
smoothed along the loop so that the corners of a blunt trailing edge and the fine spacing at a
leading edge do not fold, then lifted onto the source hub by the chart. The hole is the hub
faces the rings reach plus one layer; the annulus between the outer ring and the hole's boundary
is first triangulated by walking the two loops together, or, where a jagged boundary stops
the walk, by clipping the ears of the band cut open along one bridge.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cfdbender.mesh._repitch import RingChart
from cfdbender.mesh._repitch_grid import edges, incidence

Array = NDArray[np.float64]
Edge = tuple[int, int]

#: Passes of the binomial smoothing of the marching normals along a ring; each pass spreads a
#: corner's turn over one more node on either side.
NORMAL_PASSES = 4
#: The chart holds the hub nodes within the root's extent plus this many ring widths.
CHART_WIDTHS = 3.0


def signed_area(xy: Array) -> float:
    """Return the signed area of a closed polygon in the chart, positive counterclockwise."""
    x, y = xy[:, 0], xy[:, 1]
    return float(0.5 * (x * np.roll(y, -1) - np.roll(x, -1) * y).sum())


def inside(points: Array, polygon: Array) -> NDArray[np.bool_]:
    """Return which chart points lie inside a closed polygon (crossing number)."""
    result = np.zeros(len(points), dtype=bool)
    ahead = np.roll(polygon, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(polygon, ahead, strict=True):
        if y0 == y1:
            continue
        spans = (points[:, 1] > y0) != (points[:, 1] > y1)
        at = x0 + (points[:, 1] - y0) * (x1 - x0) / (y1 - y0)
        result ^= spans & (points[:, 0] < at)
    return result


@dataclass
class Region:
    """The source hub around one root: its chart, its nodes and its triangles in local numbering."""

    chart: RingChart
    ids: NDArray[np.int64]
    triangles: NDArray[np.int64]
    xy: Array


def region(verts: Array, faces: list[list[int]], loop: list[int], width: float) -> Region:
    """Return the chart of the hub faces near a root loop."""
    centre = verts[loop].mean(axis=0)
    reach = float(np.linalg.norm(verts[loop] - centre, axis=1).max()) + CHART_WIDTHS * width
    near = [f for f in faces if np.linalg.norm(verts[f] - centre, axis=1).min() <= reach]
    ids = np.unique([v for f in near for v in f])
    local = {int(v): k for k, v in enumerate(ids)}
    tris = [[local[f[0]], local[f[j]], local[f[j + 1]]] for f in near for j in range(1, len(f) - 1)]
    chart = RingChart(verts[ids], None)
    return Region(chart, ids, np.asarray(tris, dtype=np.int64), chart.coordinates(verts[ids]))


def _normals(row: Array) -> Array:
    """Return the outward unit normals of a counterclockwise ring, smoothed along it."""
    side = np.roll(row, -1, axis=0) - row
    side /= np.maximum(np.linalg.norm(side, axis=1), 1e-300)[:, None]
    out = np.c_[side[:, 1], -side[:, 0]]
    normal = out + np.roll(out, 1, axis=0)
    for _ in range(NORMAL_PASSES):
        normal = np.roll(normal, 1, axis=0) + 2.0 * normal + np.roll(normal, -1, axis=0)
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
    return normal


def folded(inner: Array, outer: Array) -> bool:
    """Return whether a ring of quads between two counterclockwise rows turns over anywhere."""
    a, b = inner, np.roll(inner, -1, axis=0)
    c, d = np.roll(outer, -1, axis=0), outer
    for p, q, r in ((a, d, c), (a, c, b), (a, d, b), (d, c, b)):
        cross = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (
            r[:, 0] - p[:, 0]
        )
        if np.any(cross <= 0.0):
            return True
    return False


def march(root: Array, rings: int, width: float) -> Array | None:
    """Return the rows of a counterclockwise root marched outward in the chart; None on a fold."""
    step = width / rings
    rows = [root]
    for _ in range(rings):
        nxt = rows[-1] + step * _normals(rows[-1])
        if folded(rows[-1], nxt):
            return None
        rows.append(nxt)
    return np.array(rows)


def islands_filled(faces: list[list[int]], taken: set[int]) -> set[int]:
    """Return the taken faces with every pocket of untaken faces they enclose added."""
    rest = [i for i in range(len(faces)) if i not in taken]
    by_edge = incidence([faces[i] for i in rest])
    parent = list(range(len(rest)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for owners in by_edge.values():
        for other in owners[1:]:
            parent[find(other)] = find(owners[0])
    groups: dict[int, list[int]] = defaultdict(list)
    for k, i in enumerate(rest):
        groups[find(k)].append(i)
    if len(groups) < 2:
        return set(taken)
    largest = max(groups.values(), key=len)
    return set(taken) | {i for g in groups.values() if g is not largest for i in g}


def one_layer(faces: list[list[int]], taken: set[int]) -> set[int]:
    """Return the taken faces with every face sharing a node with them added."""
    nodes = {v for i in taken for v in faces[i]}
    return set(taken) | {i for i, f in enumerate(faces) if nodes & set(f)}


def boundary_loop(faces: list[list[int]], taken: set[int], root: list[int]) -> list[int] | None:
    """Return the one closed boundary of the taken faces other than the root loop, or None."""
    owners = incidence([faces[i] for i in sorted(taken)])
    skip = set(edges(root))
    open_edges = [e for e, own in owners.items() if len(own) == 1 and e not in skip]
    links: dict[int, list[int]] = defaultdict(list)
    for a, b in open_edges:
        links[a].append(b)
        links[b].append(a)
    if not links or any(len(v) != 2 for v in links.values()):
        return None
    loop, previous = [min(links)], -1
    while len(loop) < len(links):
        nxt = next(v for v in links[loop[-1]] if v != previous)
        if nxt == loop[0]:
            return None
        previous = loop[-1]
        loop.append(nxt)
    return loop


def _turn(points: Array, tri: tuple[int, int, int]) -> float:
    """Return twice the signed area of a chart triangle; negative is the walk's orientation."""
    p, q, r = points[list(tri)]
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _arc(loop: Array) -> Array:
    """Return the fraction of a closed loop's length at each node and at the return to its first."""
    side = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)
    return np.r_[0.0, np.cumsum(side)] / side.sum()


def walk_loops(inner: Array, outer: Array) -> list[tuple[int, int, int]] | None:
    """Triangulate between two counterclockwise loops by walking them together (two pointers).

    Nodes of ``inner`` are numbered 0 to n - 1 and those of ``outer`` n onward, ``outer`` taken
    from its node nearest the first of ``inner``. Each triangle advances one loop by one node:
    the one whose next node lies earlier along its loop's length, so that the two stay level
    when their spacings differ, or the other when that triangle would turn over. Returns None
    when both would, so that no triangle turns over in the chart.
    """
    n, m = len(inner), len(outer)
    start = int(np.argmin(np.linalg.norm(outer - inner[0], axis=1)))
    outer = np.roll(outer, -start, axis=0)
    points = np.vstack([inner, outer])
    t_in, t_out = _arc(inner), _arc(outer)
    i = j = 0
    tris: list[tuple[int, int, int]] = []
    while i < n or j < m:
        step_in = (i % n, (i + 1) % n, n + j % m) if i < n else None
        step_out = (i % n, n + (j + 1) % m, n + j % m) if j < m else None
        fine = [t for t in (step_in, step_out) if t is not None and _turn(points, t) < 0.0]
        if not fine:
            return None
        inward = step_in in fine and (step_out not in fine or t_in[i + 1] <= t_out[j + 1])
        tris.append(step_in if inward else step_out)  # type: ignore[arg-type]
        i, j = (i + 1, j) if inward else (i, j + 1)
    return [tuple(v if v < n else n + (v - n + start) % m for v in t) for t in tris]  # type: ignore[misc]


def _crosses(p: Array, q: Array, a: Array, b: Array) -> NDArray[np.bool_]:
    """Return which segments a-b the segment p-q crosses properly (not at an end)."""

    def side(u: Array, v: Array, w: Array) -> Array:
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (
            w[..., 0] - u[..., 0]
        )

    d1, d2 = side(p, q, a), side(p, q, b)
    d3, d4 = side(a, b, p), side(a, b, q)
    return (d1 * d2 < 0.0) & (d3 * d4 < 0.0)


def _bridge(inner: Array, outer: Array) -> tuple[int, int] | None:
    """Return the shortest segment from an inner node to an outer node that crosses no side."""
    n, m = len(inner), len(outer)
    sides = np.vstack(
        [np.c_[inner, np.roll(inner, -1, axis=0)], np.c_[outer, np.roll(outer, -1, axis=0)]]
    )
    a, b = sides[:, :2], sides[:, 2:]
    order = np.argsort(np.linalg.norm(inner[:, None] - outer[None], axis=2), axis=None)
    for flat in order[: n * m]:
        i, j = divmod(int(flat), m)
        if not _crosses(inner[i], outer[j], a, b).any():
            return i, j
    return None


def clip_annulus(inner: Array, outer: Array) -> list[tuple[int, int, int]] | None:
    """Triangulate between two counterclockwise loops by ear clipping, through one bridge.

    Numbering as :func:`walk_loops`, triangles in the walk's orientation. The band is cut open
    along the shortest segment from the inner loop to the outer that crosses no side, and its
    ears are clipped, each the valid one whose least angle is greatest. Returns None when no
    bridge or no ear is found.
    """
    n, m = len(inner), len(outer)
    found = _bridge(inner, outer)
    if found is None:
        return None
    i, j = found
    ring = [n + (j + k) % m for k in range(m + 1)] + [(i - k) % n for k in range(n + 1)]
    points = np.vstack([inner, outer])
    tris: list[tuple[int, int, int]] = []
    while len(ring) > 3:
        ear = _best_ear(points, ring)
        if ear is None:
            return None
        k = ear
        tris.append((ring[k - 1], ring[k], ring[(k + 1) % len(ring)]))
        del ring[k]
    tris.append((ring[0], ring[1], ring[2]))
    return [(c, b, a) for a, b, c in tris]


def _best_ear(points: Array, ring: list[int]) -> int | None:
    """Return the corner of a counterclockwise polygon whose ear is valid with the widest angle."""
    ids = np.asarray(ring)
    p, q, r = points[np.roll(ids, 1)], points[ids], points[np.roll(ids, -1)]
    area = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
    candidates = np.flatnonzero(area > 0.0)
    best, score = None, -1.0
    others = points[ids]
    for k in candidates.tolist():
        tri = np.array([p[k], q[k], r[k]])
        own = [ids[k - 1], ids[k], ids[(k + 1) % len(ids)]]
        mask = ~np.isin(ids, own)
        if _covers(tri, others[mask]):
            continue
        angle = _least_angle(tri)
        if angle > score:
            best, score = k, angle
    return best


def _covers(tri: Array, pts: Array) -> bool:
    """Return whether a counterclockwise triangle holds any of the points, its sides included."""
    if not len(pts):
        return False
    inside_all = np.ones(len(pts), dtype=bool)
    for k in range(3):
        a, b = tri[k], tri[(k + 1) % 3]
        inside_all &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) >= 0.0
    return bool(inside_all.any())


def _least_angle(tri: Array) -> float:
    """Return the least angle of a triangle, in radians."""
    out = np.pi
    for k in range(3):
        u, v = tri[(k + 1) % 3] - tri[k], tri[(k + 2) % 3] - tri[k]
        cosine = float(u @ v) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-300)
        out = min(out, float(np.arccos(min(1.0, max(-1.0, cosine)))))
    return out
