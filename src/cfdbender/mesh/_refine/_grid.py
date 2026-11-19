"""The structured grid of a family, recovered and resampled..

A GRID FAMILY is recovered from the connectivity alone, never from the vertex
order of the file: quadrilaterals, or quadrilaterals each split into two
triangles, laid out as a SHEET (an open grid with four corners) or as a TUBE
(closed chordwise) whose two ends are each OPEN (a boundary loop), closed by
a POLE fan (triangles around one node) or closed by a ZIPPER cap
(a strip pairing the upper node i with the lower node n - i, a triangle at
each end of the strip). The trailing edge is the grid line through the
family's trailing-edge vertices; it is chordwise index 0. A SMOOTH TUBE (a
body of revolution: a spinner, a nacelle barrel) holds no trailing-edge vertex
and no edge whose dihedral exceeds ``_geometry.RIDGE_DEGREES``; its seam,
chordwise index 0, is the lowest source vertex index of its first station,
and its chordwise direction is resampled by a periodic spline, so the closed
section has no knot and no kink at the seam.

The surface the orchestrator calls:

- :func:`recover_grid` ``(verts, faces, te_vertices)`` returns the
  :class:`Grid` of a family, or the reason, as text, why none was recovered
  ( names it in ``refine.json`` and in the refusal of ``method = "grid"``).
- :func:`check_factors` ``(grid, family, chordwise=, spanwise=)`` raises the refusal of a factor
  that is not finite and positive or that is below
  1/m for the m source intervals of its index direction; call it for every
  family before any family is resampled.
- :func:`refine_grid` ``(verts, faces, grid, chordwise=, spanwise=)`` returns
  a :class:`GridLevel`: the new points, the faces as indices into them, the
  trailing-edge mid-points and the per-family report.
- :func:`not_a_knot` is the cubic spline the resampling uses, and
  :func:`periodic` the closed one of a smooth tube's sections, both written
  with numpy alone: this module imports neither scipy nor rtree.

Resampling is a cubic spline in INDEX space, so the clustering of the
source is kept and only the counts change: ``round(f m)`` intervals per index
direction, counted per stretch between knots (the trailing edge to the
leading edge and back, around a tube). The trailing edge, the leading edge
and the end stations sit at integer parameters, where the spline returns the
source node itself. Each new cell is split along the diagonal of the source
cell its midpoint falls in.

Face order: when the source's lateral faces form one sweep (each cell's
faces consecutive, the cells in row or column order from one start cell),
the level is written in that sweep, each cell's faces following the source
cell's template; otherwise every face takes the position of the nearest
source face (``_geometry.order_like``). The end faces of a tube start on the
vertex playing the role the source's end faces start on. At factor 1 the
level's faces equal the source's in coordinates and in order.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import numpy
from numpy.typing import NDArray

from cfdbender._errors import InputArtifactError
from cfdbender.mesh._refine._geometry import (
    RIDGE_DEGREES,
    Faces,
    Points,
    dihedral_degrees,
    edge_faces,
    order_like,
    orient_like,
)
from cfdbender.mesh._refine._obj import KIND

Edge = tuple[int, int]
Template = tuple[tuple[tuple[int, int], ...], ...]


class _NotAGridError(Exception):
    """The reason, as its message, why a family's grid was not recovered."""


@dataclass(frozen=True, eq=False)
class Grid:
    """The structured grid of one family, as recovered from its connectivity.

    Attributes
    ----------
    layout : str
        ``"sheet"`` or ``"tube"``.
    ids : numpy.ndarray
        Source vertex indices, shape (K, n): K spanwise stations of n
        chordwise nodes; column 0 is the trailing edge.
    ends : tuple of str
        The kind of each end (station 0, station K - 1): ``"edge"`` for a
        sheet; ``"open"``, ``"pole"`` or ``"zipper"`` for a tube.
    pole : int or None
        The pole vertex of a fan closing the far end.
    i_le : int
        The chordwise index of the leading edge.
    cell_diag : numpy.ndarray or None
        Per lateral cell, shape (K - 1, cells): 0 when its two triangles share
        the diagonal (k, i)-(k+1, i+1), 1 when they share (k, i+1)-(k+1, i);
        None for quadrilaterals.
    smooth : bool
        A tube without trailing edge: column 0 is the seam, ``i_le`` is 0 and
        the chordwise direction is resampled by the periodic spline.
    root_pole : int or None
        The pole vertex of a fan closing station 0, when a fan closes both
        ends; a single fan is always the far end's.
    """

    layout: str
    ids: NDArray[numpy.int64]
    ends: tuple[str, str]
    pole: int | None
    i_le: int
    cell_diag: NDArray[numpy.int64] | None
    smooth: bool = False
    root_pole: int | None = None

    @property
    def poles(self) -> tuple[int | None, int | None]:
        """Return the pole of each end (station 0, station K - 1), None where there is none."""
        return self.root_pole, self.pole

    @property
    def wrap(self) -> bool:
        """Return whether the chordwise direction closes on itself (a tube)."""
        return self.layout == "tube"

    @property
    def intervals(self) -> tuple[int, int]:
        """Return the source's chordwise and spanwise interval counts."""
        k, n = self.ids.shape
        return (n if self.wrap else n - 1), k - 1

    def describe(self) -> str:
        """Return the recovery reason of a grid that was recovered."""
        k, n = self.ids.shape
        split = "quadrilaterals" if self.cell_diag is None else "split quadrilaterals"
        layout = "smooth tube without trailing edge" if self.smooth else self.layout
        return f"a {layout} of {k} stations by {n} nodes across the reference line, of {split}"


@dataclass(frozen=True, eq=False)
class GridLevel:
    """One resampled grid family.

    Attributes
    ----------
    points : numpy.ndarray
        The new nodes, shape (P, 3); the far end's pole, then station 0's,
        when there are, are last.
    faces : list of list of int
        The new faces as indices into ``points``, in the order of.
    te_midpoints : numpy.ndarray
        One mid-point per trailing-edge edge, shape (K' - 1, 3).
    report : dict
        The family's entry of ``refine.json``: layout, ends, split, the
        intervals and faces before and after, the face order used.
    """

    points: Points
    faces: Faces
    te_midpoints: Points
    report: dict[str, object]


# ---------------------------------------------------------------- the spline


def not_a_knot(values: NDArray[numpy.float64], params: NDArray[numpy.float64]) -> Points:
    """Evaluate the not-a-knot cubic spline through values at the indices 0..m.

    Parameters
    ----------
    values : numpy.ndarray
        Shape (m + 1, ...): the nodes at the integer parameters 0 to m.
    params : numpy.ndarray
        The parameters to evaluate at, within [0, m].

    Returns
    -------
    numpy.ndarray
        Shape (len(params), ...). An integer parameter returns its node
        exactly; with fewer than four nodes the interpolation is linear.
    """
    y = numpy.asarray(values, dtype=float)
    p = numpy.asarray(params, dtype=float).reshape(-1)
    m = y.shape[0] - 1
    flat = y.reshape(m + 1, -1)
    if m < 3:
        x = numpy.arange(m + 1)
        cols = [numpy.interp(p, x, flat[:, j]) for j in range(flat.shape[1])]
        out = numpy.stack(cols, axis=1)
    else:
        out = _evaluate(flat, _second_derivatives(flat), p)
    exact = (p == numpy.floor(p)) & (p >= 0) & (p <= m)
    out[exact] = flat[p[exact].astype(numpy.int64)]
    return out.reshape((len(p),) + y.shape[1:])


def periodic(values: NDArray[numpy.float64], params: NDArray[numpy.float64]) -> Points:
    """Evaluate the periodic cubic spline through values at the indices 0..n-1, period n.

    Parameters
    ----------
    values : numpy.ndarray
        Shape (n, ...): the nodes of a closed loop at the integer parameters
        0 to n - 1; the node after n - 1 is node 0.
    params : numpy.ndarray
        The parameters to evaluate at, within [0, n).

    Returns
    -------
    numpy.ndarray
        Shape (len(params), ...). An integer parameter returns its node
        exactly. The spline and its first two derivatives are continuous all
        around the loop, the seam included; with fewer than three nodes the
        interpolation is linear.
    """
    y = numpy.asarray(values, dtype=float)
    p = numpy.asarray(params, dtype=float).reshape(-1)
    n = y.shape[0]
    flat = y.reshape(n, -1)
    closed = numpy.vstack([flat, flat[:1]])
    if n < 3:
        x = numpy.arange(n + 1)
        out = numpy.stack([numpy.interp(p, x, closed[:, j]) for j in range(flat.shape[1])], axis=1)
    else:
        # M(i-1) + 4 M(i) + M(i+1) = 6 (y(i+1) - 2 y(i) + y(i-1)) around the loop: a
        # circulant system, diagonal in the discrete Fourier basis.
        rhs = 6.0 * (numpy.roll(flat, -1, axis=0) - 2.0 * flat + numpy.roll(flat, 1, axis=0))
        eigen = 4.0 + 2.0 * numpy.cos(2.0 * numpy.pi * numpy.arange(n) / n)
        second = numpy.fft.ifft(numpy.fft.fft(rhs, axis=0) / eigen[:, None], axis=0).real
        out = _evaluate(closed, numpy.vstack([second, second[:1]]), p)
    exact = (p == numpy.floor(p)) & (p >= 0) & (p < n)
    out[exact] = flat[p[exact].astype(numpy.int64)]
    return out.reshape((len(p),) + y.shape[1:])


def _second_derivatives(flat: Points) -> Points:
    """Solve the not-a-knot system of unit spacing for the second derivatives.

    The not-a-knot rows (M0 - 2 M1 + M2 = 0 and its mirror) fold into the
    first and last interior rows, giving M1 and M(m-1) directly; the rows
    between are tridiagonal and strictly diagonally dominant.
    """
    m = flat.shape[0] - 1
    r = 6.0 * (flat[2:] - 2.0 * flat[1:-1] + flat[:-2])  # rows 1..m-1
    second = numpy.zeros_like(flat)
    second[1], second[m - 1] = r[0] / 6.0, r[-1] / 6.0
    inner = r[1:-1].copy()  # rows 2..m-2
    if len(inner):
        inner[0] -= second[1]
        inner[-1] -= second[m - 1]
        second[2 : m - 1] = _tridiagonal(inner)
    second[0] = 2.0 * second[1] - second[2]
    second[m] = 2.0 * second[m - 1] - second[m - 2]
    return second


def _tridiagonal(rhs: Points) -> Points:
    """Solve the (1, 4, 1) tridiagonal system for every column of rhs."""
    q = len(rhs)
    upper = numpy.zeros(q)
    d = numpy.zeros_like(rhs)
    upper[0], d[0] = 0.25, rhs[0] / 4.0
    for i in range(1, q):
        w = 4.0 - upper[i - 1]
        upper[i] = 1.0 / w
        d[i] = (rhs[i] - d[i - 1]) / w
    for i in range(q - 2, -1, -1):
        d[i] -= upper[i] * d[i + 1]
    return d


def _evaluate(flat: Points, second: Points, p: NDArray[numpy.float64]) -> Points:
    """Evaluate the cubic of each interval at the parameters."""
    m = flat.shape[0] - 1
    i = numpy.clip(numpy.floor(p), 0, m - 1).astype(numpy.int64)
    t = (p - i)[:, None]
    y0, y1, m0, m1 = flat[i], flat[i + 1], second[i], second[i + 1]
    slope = (y1 - y0) - (2.0 * m0 + m1) / 6.0
    return y0 + t * (slope + t * (0.5 * m0 + t * (m1 - m0) / 6.0))


# ------------------------------------------------------------- the recovery


def recover_grid(verts: Points, faces: Faces, te_vertices: Collection[int]) -> Grid | str:
    """Recover the structured grid of a family from its connectivity.

    Parameters
    ----------
    verts : numpy.ndarray
        All the mesh's vertices, shape (V, 3).
    faces : list of list of int
        The family's faces, as indices into ``verts``.
    te_vertices : collection of int
        The family's trailing-edge vertices.

    Returns
    -------
    Grid or str
        The grid, or the reason no grid was recovered.
    """
    try:
        return _recover(numpy.asarray(verts, dtype=float), faces, frozenset(te_vertices))
    except _NotAGridError as reason:
        return str(reason)


@dataclass(frozen=True, eq=False)
class _Part:
    """A family's faces, its lateral faces (zipper caps removed), their edges and cap nodes."""

    faces: Faces
    lmap: Mapping[Edge, list[int]]
    cap_verts: set[int]
    lateral: Faces | None = None

    @property
    def side(self) -> Faces:
        """Return the lateral faces."""
        return self.faces if self.lateral is None else self.lateral


def _recover(verts: Points, faces: Faces, te: frozenset[int]) -> Grid:
    """Return the grid of a family or raise the reason it has none.

    The layout is first read from the boundary: four corners, each in one
    face, make a sheet, and a boundary without corners a tube. A family of
    triangles that reading misses is then tried as a sheet of split
    quadrilaterals, its corners read from its cells, and a family with no
    boundary as a tube closed by a pole fan at each end.
    """
    try:
        return _from_boundary(verts, faces, te)
    except _NotAGridError:
        grid = _split_sheet(verts, faces, te) or _two_poles(verts, faces, te)
        if grid is None:
            raise
        return grid


def _from_boundary(verts: Points, faces: Faces, te: frozenset[int]) -> Grid:
    """Return the grid whose layout the boundary's corners and loops give."""
    removed, cap_verts = _zipper_caps(faces)
    lateral = [f for k, f in enumerate(faces) if k not in removed]
    lmap = edge_faces(lateral)
    layout, seed = _layout_and_seed(verts, lateral, lmap)
    return _grid_of(verts, te, _Part(faces, lmap, cap_verts, lateral), layout, seed)


def _grid_of(verts: Points, te: frozenset[int], part: _Part, layout: str, seed: list[int]) -> Grid:
    """Return the grid layered from the seed chain, or raise the reason it is none."""
    adj = _adjacency(part.lmap)
    smooth = layout == "tube" and not te and _without_ridge(verts, part.faces)
    rows, pole = _rows(verts, seed, adj, layout == "tube", te, smooth=smooth)
    ids = numpy.array(rows, dtype=numpy.int64)
    if layout == "sheet":
        ids = _sheet_trailing_edge_first(ids, te)
        ends, i_le = ("edge", "edge"), ids.shape[1] - 1
    else:
        ends, i_le = _tube_ends(verts, ids, pole, part.cap_verts, te, smooth=smooth)
    diag = _cell_diagonals(ids, part.side, part.lmap, layout == "tube", pole)
    grid = Grid(layout, ids, ends, pole, i_le, diag, smooth)
    _check_covers(grid, part.faces)
    return grid


def _without_ridge(verts: Points, faces: Faces) -> bool:
    """Return whether no edge of the family is sharper than the ridge angle."""
    return all(d <= RIDGE_DEGREES for d in dihedral_degrees(verts, faces).values())


def _adjacency(edges: Iterable[Edge]) -> dict[int, set[int]]:
    """Return each vertex's neighbors along the edges."""
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _vertex_faces(faces: Faces) -> dict[int, list[int]]:
    """Return the faces of each vertex."""
    out: dict[int, list[int]] = defaultdict(list)
    for k, f in enumerate(faces):
        for v in f:
            out[v].append(k)
    return out


def _zipper_caps(faces: Faces) -> tuple[set[int], set[int]]:
    """Return the faces of the zipper caps and their vertices.

    A cap is a strip of quadrilaterals between two triangles whose apex has
    three faces. A triangle that starts no strip stays lateral (a pole fan).
    """
    tris = [k for k, f in enumerate(faces) if len(f) == 3]
    removed: set[int] = set()
    if not tris or len(tris) == len(faces):
        return removed, set()
    emap, vface = edge_faces(faces), _vertex_faces(faces)
    for t in tris:
        if t not in removed:
            removed.update(_strip_from(t, faces, emap, vface) or [])
    return removed, {v for k in removed for v in faces[k]}


def _strip_from(
    start: int, faces: Faces, emap: Mapping[Edge, list[int]], vface: Mapping[int, list[int]]
) -> list[int] | None:
    """Walk the quadrilateral strip from a triangle whose apex has three faces."""
    apex = [v for v in faces[start] if len(vface[v]) == 3]
    if len(apex) != 1:
        return None
    a, b = (v for v in faces[start] if v != apex[0])
    strip, cur, entry = [start], start, (min(a, b), max(a, b))
    while len(strip) <= len(faces):
        nxt = [g for g in emap[entry] if g != cur]
        if len(nxt) != 1:
            return None
        cur = nxt[0]
        strip.append(cur)
        if len(faces[cur]) == 3:
            # A cap holds a quadrilateral between its two triangles; two triangles side by
            # side are the corner of a triangulated grid (F-326).
            return strip if len(strip) > 2 else None
        found = _opposite_edge(faces[cur], entry)
        if found is None:
            return None
        entry = found
    return None


def _opposite_edge(quad: list[int], entry: Edge) -> Edge | None:
    """Return the edge of a quadrilateral opposite the entry edge."""
    if len(quad) != 4:
        return None
    hits = [i for i in range(4) if {quad[i], quad[(i + 1) % 4]} == set(entry)]
    if len(hits) != 1:
        return None
    p, q = quad[(hits[0] + 2) % 4], quad[(hits[0] + 3) % 4]
    return (min(p, q), max(p, q))


def _chain_order(vs: Iterable[int], adj: Mapping[int, set[int]]) -> tuple[list[int], bool]:
    """Order a set of vertices as one chain or cycle along adj; return it and if it closes."""
    members = set(vs)
    inner = {v: sorted(u for u in adj[v] if u in members) for v in members}
    if any(len(n) > 2 for n in inner.values()):
        raise _NotAGridError("a layer is not a simple chain")
    ends = sorted(v for v, n in inner.items() if len(n) == 1)
    start = ends[0] if ends else min(members)
    out, visited, cur = [start], {start}, start
    while True:
        nxt = [u for u in inner[cur] if u not in visited]
        if not nxt:
            break
        cur = nxt[0]
        out.append(cur)
        visited.add(cur)
    if len(out) != len(members):
        raise _NotAGridError("a layer is disconnected")
    return out, not ends


def _component(v: int, adj: Mapping[int, set[int]]) -> set[int]:
    """Return the vertices connected to v along adj."""
    out, stack = {v}, [v]
    while stack:
        for u in adj[stack.pop()]:
            if u not in out:
                out.add(u)
                stack.append(u)
    return out


def _layout_and_seed(
    verts: Points, lateral: Faces, lmap: Mapping[Edge, list[int]]
) -> tuple[str, list[int]]:
    """Return the layout and the boundary chain the layering starts from."""
    loops = _loops(lmap)
    count = Counter(v for f in lateral for v in f)
    corners = sorted(v for loop in loops for v in loop if count[v] == 1)
    if len(corners) == 4 and len(loops) == 1:
        at = sorted(loops[0].index(c) for c in corners)
        return "sheet", loops[0][at[0] : at[1] + 1]
    if not corners and len(loops) in (1, 2):
        norms = [float(numpy.linalg.norm(verts[loop].mean(axis=0))) for loop in loops]
        return "tube", loops[int(numpy.argmin(norms))]
    raise _NotAGridError(f"{len(loops)} boundary loops and {len(corners)} corners")


def _loops(lmap: Mapping[Edge, list[int]]) -> list[list[int]]:
    """Return the boundary loops, each in order along its edges, from its lowest vertex."""
    badj = _adjacency(e for e, fs in lmap.items() if len(fs) == 1)
    loops: list[list[int]] = []
    seen: set[int] = set()
    for v in sorted(badj):
        if v not in seen:
            loops.append(_chain_order(_component(v, badj), badj)[0])
            seen.update(loops[-1])
    return loops


def _two_poles(verts: Points, faces: Faces, te: frozenset[int]) -> Grid | None:
    """Return the grid of a tube closed by a fan of triangles around a pole at each end.

    Such a family has no boundary, so the boundary gives no layout. A
    candidate root pole lies only in triangles, as many as divide the
    family's nodes less the two poles; candidates are tried by that count,
    largest first, then from the one whose ring center is nearest the
    origin (the end a tube's layering starts from). The two poles of such a
    tube are the only nodes with their fan size, so a size that more than two
    nodes share names no pole: an unstructured closed triangulation, whose
    nodes share a handful of sizes, offers no candidate and is refused at once
    (IND3-01). Without the candidate's
    fan the family is a tube open at that ring and closed by a pole at the
    far end, and the fan is the triangles of that ring around the
    candidate; the first candidate for which both hold is the root pole.
    None when there is a boundary or no candidate holds.
    """
    if any(len(fs) != 2 for fs in edge_faces(faces).values()):
        return None
    vface = _vertex_faces(faces)
    nodes = len(vface)
    sizes = Counter(len(fs) for fs in vface.values())

    def center(v: int) -> float:
        ring = {u for k in vface[v] for u in faces[k] if u != v}
        return float(numpy.linalg.norm(verts[sorted(ring)].mean(axis=0)))

    candidates = sorted(
        (
            v
            for v, fs in vface.items()
            if len(fs) >= 3
            and sizes[len(fs)] <= 2
            and (nodes - 2) % len(fs) == 0
            and all(len(faces[k]) == 3 for k in fs)
        ),
        key=lambda v: (-len(vface[v]), center(v), v),
    )
    for root in candidates:
        fan = set(vface[root])
        try:
            grid = _from_boundary(verts, [f for k, f in enumerate(faces) if k not in fan], te)
        except _NotAGridError:
            continue
        n = grid.ids.shape[1]
        ring = [int(v) for v in grid.ids[0]]
        want = Counter(frozenset((root, ring[i], ring[(i + 1) % n])) for i in range(n))
        if grid.ends[1] != "pole" or want != Counter(frozenset(faces[k]) for k in fan):
            continue
        # the rest of the family is covered by the grid and the fan by the root pole
        return dataclasses.replace(grid, ends=("pole", "pole"), root_pole=root)
    return None


def _sides(perimeter: int, nodes: int) -> tuple[int, int] | None:
    """Return the interval counts m <= n of a sheet with this boundary and node count.

    A sheet of m by n intervals has 2 (m + n) boundary nodes and
    (m + 1)(n + 1) nodes; None when no such pair of whole numbers exists.
    """
    if perimeter % 2:
        return None
    s = perimeter // 2
    disc = s * s - 4 * (nodes - s - 1)
    root = math.isqrt(disc) if disc >= 0 else -1
    if root < 0 or root * root != disc or (s - root) % 2 or s - root < 2:
        return None
    return (s - root) // 2, (s + root) // 2


def _split_sheet(verts: Points, faces: Faces, te: frozenset[int]) -> Grid | None:
    """Return the grid of a sheet of split quadrilaterals, its corners read from the cells.

    A corner of such a sheet lies in one triangle or in two, and so may a
    node of its sides, so a single face does not mark the corners. The
    boundary loop's P nodes and the family's V nodes give the sides,
    m + n = P / 2 and (m + 1)(n + 1) = V; every placement of the corners at
    the loop positions p, p + m, p + m + n and p + 2m + n that holds each
    node lying in one triangle, and no node lying in more than two, is a
    candidate. A candidate is the grid when the stations layered from one of
    its sides are chains of equal length, each aligned node for node with
    the one before along an edge, and the grid holds every node and every
    face (the checks of any sheet). None when no candidate is; two different
    grids are refused as ambiguous.
    """
    if not faces or any(len(f) != 3 for f in faces):
        return None
    lmap = edge_faces(faces)
    loops = _loops(lmap)
    count = Counter(v for f in faces for v in f)
    sides = _sides(len(loops[0]), len(count)) if len(loops) == 1 else None
    if sides is None:
        return None
    loop, (m, n), size = loops[0], sides, len(loops[0])
    single = {v for v in loop if count[v] == 1}
    part = _Part(faces, lmap, set())
    tried: set[frozenset[int]] = set()
    found: list[Grid] = []
    for p in range(size):
        corners = frozenset(loop[(p + q) % size] for q in (0, m, m + n, 2 * m + n))
        if corners in tried or not single <= corners or any(count[c] > 2 for c in corners):
            continue
        tried.add(corners)
        try:
            grid = _grid_of(verts, te, part, "sheet", [loop[(p + q) % size] for q in range(m + 1)])
        except _NotAGridError:
            continue
        found.append(grid)
    if len(found) > 1:
        raise _NotAGridError(f"{len(found)} sheets of split quadrilaterals fit its cells")
    return found[0] if found else None


def _layers(seed: list[int], adj: Mapping[int, set[int]]) -> list[list[int]]:
    """Return the vertices by graph distance from the seed chain."""
    dist = dict.fromkeys(seed, 0)
    queue = deque(seed)
    while queue:
        v = queue.popleft()
        for u in sorted(adj[v]):
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)
    layers: list[list[int]] = [[] for _ in range(max(dist.values()) + 1)]
    for v, d in dist.items():
        layers[d].append(v)
    return layers


def _rows(
    verts: Points,
    seed: list[int],
    adj: Mapping[int, set[int]],
    tube: bool,
    te: frozenset[int],
    *,
    smooth: bool = False,
) -> tuple[list[list[int]], int | None]:
    """Return the stations, each ordered like the one before it, and a far pole.

    A tube starts at a trailing-edge vertex of its first station; a smooth tube
    at the lowest source vertex index of it (its seam).
    """
    layers = _layers(seed, adj)
    pole = layers.pop()[0] if tube and len(layers) > 1 and len(layers[-1]) == 1 else None
    sizes = {len(layer) for layer in layers}
    if len(sizes) != 1:
        raise _NotAGridError(f"layer sizes differ: {sorted(sizes)[:6]}")
    if len(layers) < 2:
        raise _NotAGridError("fewer than two stations")
    first, _ = _chain_order(layers[0], adj)
    if tube:
        at = [first.index(min(first))] if smooth else [i for i, v in enumerate(first) if v in te]
        if not at:
            raise _NotAGridError("no trailing-edge vertex on the end loop")
        first = first[at[0] :] + first[: at[0]]
    rows = [first]
    for k in range(1, len(layers)):
        rows.append(_align(verts, layers[k], rows[-1], adj, k=k, te=te))
    return rows, pole


def _align(
    verts: Points,
    layer: list[int],
    prev: list[int],
    adj: Mapping[int, set[int]],
    *,
    k: int,
    te: frozenset[int] = frozenset(),
) -> list[int]:
    """Order a station so each node neighbors the node of the previous station.

    Of the alignments that hold, the one that puts a trailing-edge node under
    every trailing-edge node of the previous station wins, and the shortest
    among those: a twisted grid's shortest alignment can shift the trailing
    edge off its column (F-328).
    """
    cand, cyclic = _chain_order(layer, adj)
    best, best_d = None, (math.inf, math.inf)
    for seq in (cand, cand[::-1]):
        shifts = [s for s, v in enumerate(seq) if v in adj[prev[0]]] if cyclic else [0]
        for s in shifts:
            option = seq[s:] + seq[:s]
            if all(u in adj[w] for u, w in zip(option, prev, strict=True)):
                off = sum((u in te) != (w in te) for u, w in zip(option, prev, strict=True))
                d = (off, float(numpy.linalg.norm(verts[option] - verts[prev], axis=1).sum()))
                if d < best_d:
                    best, best_d = option, d
    if best is None:
        raise _NotAGridError(f"layer {k} does not align with layer {k - 1}")
    return best


def _te_columns(ids: NDArray[numpy.int64], te: frozenset[int]) -> list[int]:
    """Return the columns whose every node is a trailing-edge vertex."""
    return [i for i in range(ids.shape[1]) if all(int(v) in te for v in ids[:, i])]


def _sheet_trailing_edge_first(
    ids: NDArray[numpy.int64], te: frozenset[int]
) -> NDArray[numpy.int64]:
    """Orient a sheet so the trailing edge is chordwise index 0.

    A SHEET NEEDS A TRAILING EDGE: the trailing edge is the one
    grid line that fixes which of the sheet's edges is chordwise index 0, and
    a family that names none has no such line, so it is refused for that reason.
    Only a smooth tube is recovered without one, because its seam stands in
    for it; a sheet has no seam to stand in.
    """
    if not te:
        raise _NotAGridError(
            "a sheet needs a trailing edge: the family has no trailing-edge vertex, and only "
            "a smooth tube is recovered without one"
        )
    cols = _te_columns(ids, te)
    if not cols:
        if not _te_columns(ids.T, te):
            return _sheet_incomplete_trailing_edge_first(ids, te)
        ids = ids.T.copy()
        cols = _te_columns(ids, te)
    return ids[:, ::-1].copy() if cols[0] != 0 else ids


def _sheet_incomplete_trailing_edge_first(
    ids: NDArray[numpy.int64], te: frozenset[int]
) -> NDArray[numpy.int64]:
    """Orient a sheet on the boundary line that holds every listed trailing-edge vertex.

    An incomplete trailing-edge points file lists part of the line only. When every
    trailing-edge vertex of the sheet lies on one boundary line (chordwise index 0 or
    last, in either index direction) and they are at least half of that line's nodes,
    that line is the trailing edge; the level completes it and warns. A few points on
    another boundary (a root or tip chord) are not taken for it. Otherwise the trailing
    edge is not a grid line.
    """
    on = {int(v) for v in ids.ravel()} & set(te)
    best = None
    for grid in (ids, ids.T):
        for col in (0, grid.shape[1] - 1):
            line = {int(v) for v in grid[:, col]}
            share = len(on) / len(line)
            if len(on) >= 2 and on <= line and share >= 0.5 and (best is None or share > best[0]):
                best = (share, grid[:, ::-1].copy() if col != 0 else grid.copy())
    if best is None:
        raise _NotAGridError("the trailing edge is not a grid line")
    return best[1]


def _tube_ends(
    verts: Points,
    ids: NDArray[numpy.int64],
    pole: int | None,
    cap_verts: set[int],
    te: frozenset[int],
    *,
    smooth: bool = False,
) -> tuple[tuple[str, str], int]:
    """Return the kind of each end of a tube and its leading-edge index (0 for a smooth tube)."""
    if smooth:
        if cap_verts:
            raise _NotAGridError("a zipper cap needs a trailing edge")
        return ("open", "pole" if pole is not None else "open"), 0
    if not all(int(v) in te for v in ids[:, 0]):
        raise _NotAGridError("the reference line is not the grid's first line across it")
    n = ids.shape[1]
    mid = ids[ids.shape[0] // 2]
    i_le = int(numpy.argmax(numpy.linalg.norm(verts[mid] - verts[mid[0]], axis=1)))

    def kind(ring: NDArray[numpy.int64]) -> str:
        return "zipper" if cap_verts and set(ring.tolist()) <= cap_verts else "open"

    ends = (kind(ids[0]), "pole" if pole is not None else kind(ids[-1]))
    if "zipper" in ends and i_le * 2 != n:
        raise _NotAGridError("a zipper cap needs the leading edge opposite the trailing edge")
    return ends, i_le


def _cell_diagonals(
    ids: NDArray[numpy.int64],
    lateral: Faces,
    lmap: Mapping[Edge, list[int]],
    tube: bool,
    pole: int | None,
) -> NDArray[numpy.int64] | None:
    """Return the diagonal of each lateral cell's triangle split, or None for quadrilaterals."""
    sizes = {len(f) for f in lateral if pole is None or pole not in f}
    if sizes == {4}:
        return None
    if sizes != {3}:
        raise _NotAGridError("a lateral grid that mixes triangles and quads")
    n = ids.shape[1]
    cells = n if tube else n - 1
    a, c = ids[:-1, :cells], ids[1:, (numpy.arange(cells) + 1) % n]
    rows = [
        [
            0 if (min(x, y), max(x, y)) in lmap else 1
            for x, y in zip(ra.tolist(), rc.tolist(), strict=True)
        ]
        for ra, rc in zip(a, c, strict=True)
    ]
    return numpy.array(rows, dtype=numpy.int64)


def _check_covers(grid: Grid, faces: Faces) -> None:
    """Refuse a grid that leaves a vertex or a face of the family out."""
    used = {v for f in faces for v in f}
    poles = {p for p in grid.poles if p is not None}
    held = set(grid.ids.ravel().tolist()) | poles
    if used != held:
        raise _NotAGridError(f"{len(used - held)} vertices outside the grid")
    k, n = grid.ids.shape
    cells = (k - 1) * grid.intervals[0] * (1 if grid.cell_diag is None else 2)
    expected = cells + n * len(poles) + grid.i_le * grid.ends.count("zipper")
    if expected != len(faces):
        raise _NotAGridError(f"{len(faces)} faces where the grid has {expected}")


# ------------------------------------------------------------- the factors


def _count(m: int, factor: float) -> int:
    """Return round(f m), at least one interval."""
    return max(1, round(m * factor))


def _spanwise_params(k: int, factor: float) -> NDArray[numpy.float64]:
    """Return the new spanwise parameters of a grid of ``k`` stations.

    The ``k - 1`` source intervals are resampled by the factor; the end
    stations stay at their integer parameters.
    """
    return numpy.linspace(0, k - 1, _count(k - 1, factor) + 1)


def check_factors(grid: Grid, family: str, *, chordwise: float, spanwise: float) -> None:
    """Refuse a factor that is not finite and positive, or that leaves no interval.

    On a tube the chordwise factor is also refused when the section it
    leaves cannot close: fewer than three nodes around, or fewer than two
    intervals on each half of a zipper-capped section, which its end caps
    need. Every refusal comes before any family is resampled.

    Parameters
    ----------
    grid : Grid
        The family's recovered grid.
    family : str
        The family's name, for the message.
    chordwise, spanwise : float
        The factor of each index direction.

    Raises
    ------
    InputArtifactError
        Naming the family, the direction and the value; nothing was written.
    """
    chord_m, span_m = grid.intervals
    for direction, value, m in zip(
        ("normal", "parallel"), (chordwise, spanwise), (chord_m, span_m), strict=True
    ):
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not (number and math.isfinite(value) and value > 0):
            raise InputArtifactError(
                f"family {family}: the {direction} factor is {value!r}, not a finite number "
                f"greater than zero. Give a {direction} factor greater than zero and run again. "
                "Nothing was written.",
                kind=KIND,
                where={"families": [family], "nodes": grid.ids.ravel().tolist()},
            )
        if round(value * m, 9) >= 1.0:  # 1/m itself is the least: (1/49) * 49 < 1 in floats
            continue
        raise InputArtifactError(
            f"family {family}: the {direction} factor is {value:g}, below 1/{m}: scaled by it, "
            f"its {m} source intervals come to fewer than one. Give a {direction} factor of "
            f"at least 1/{m}, or leave force_factors at True, the default, to raise it with a "
            "warning, and run again. Nothing was written.",
            kind=KIND,
            where={"families": [family], "nodes": grid.ids.ravel().tolist()},
        )
    if grid.wrap:
        _check_section(grid, family, chordwise)


def _check_section(grid: Grid, family: str, chordwise: float) -> None:
    """Refuse a chordwise factor whose section a tube's caps cannot carry."""
    s, new_le = _chordwise_params(grid, chordwise)
    if "zipper" in grid.ends and new_le < 2:
        half = grid.i_le
        reason = (
            f"leaves {new_le} interval(s) on each half of the zipper-capped section of "
            f"{half} per half, below the 2 its end caps need"
        )
        least = f"2/{half}"
    elif len(s) < 3:
        n = grid.intervals[0]
        reason = f"leaves {len(s)} node(s) around the tube's section of {n}, below 3"
        least = f"3/{n}"
    else:
        return
    raise InputArtifactError(
        f"family {family}: the normal factor is {chordwise:g}, which {reason}. Give a "
        f"normal factor of at least {least}, or leave force_factors at True, the default, to "
        "raise it with a warning, and run again. Nothing was written.",
        kind=KIND,
        where={"families": [family], "nodes": grid.ids.ravel().tolist()},
    )


def least_factors(grid: Grid, chordwise: float, spanwise: float) -> tuple[float, float]:
    """Return the factors raised to the least the grid allows (``force_factors``).

    A factor below ``1/m`` for the ``m`` source intervals of its direction
    becomes ``1/m``; on a tube the chordwise factor is also raised to the
    least its section's caps carry (:func:`_check_section`). A factor that
    is not a finite number above zero is returned as it is, for
    :func:`check_factors` to refuse.
    """
    chord_m, span_m = grid.intervals
    raised = []
    for value, m in ((chordwise, chord_m), (spanwise, span_m)):
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if number and math.isfinite(value) and value > 0 and value * m < 1.0:
            value = 1.0 / m
        raised.append(value)
    chordwise, spanwise = raised
    if grid.wrap and isinstance(chordwise, (int, float)) and chordwise > 0:
        if "zipper" in grid.ends:
            chordwise = max(chordwise, 2.0 / grid.i_le)
        else:
            chordwise = max(chordwise, 3.0 / grid.intervals[0])
    return chordwise, spanwise


# ------------------------------------------------------------- the resampling


@dataclass(frozen=True, eq=False)
class _Sampled:
    """The resampled grid: nodes, their indices, and where each new cell came from."""

    nodes: Points  # (K', nc, 3)
    idx: NDArray[numpy.int64]  # (K', nc)
    new_le: int
    old_k: NDArray[numpy.int64]  # per new spanwise cell, its source station cell
    old_i: NDArray[numpy.int64]  # per new chordwise cell, its source chordwise cell


def _chordwise_params(grid: Grid, factor: float) -> tuple[NDArray[numpy.float64], int]:
    """Return the new chordwise parameters and the leading edge's new index."""
    n = grid.ids.shape[1]
    if not grid.wrap:
        s = numpy.linspace(0, n - 1, _count(n - 1, factor) + 1)
        return s, len(s) - 1
    if grid.smooth:
        return numpy.linspace(0, n, _count(n, factor) + 1)[:-1], 0
    m1, m2 = grid.i_le, n - grid.i_le
    n1 = _count(m1, factor)
    n2 = n1 if "zipper" in grid.ends else _count(m2, factor)
    upper = numpy.linspace(0, m1, n1 + 1)[:-1]
    lower = m1 + numpy.linspace(0, m2, n2 + 1)[:-1]
    return numpy.concatenate([upper, lower]), n1


def _resample(source: Points, grid: Grid, chordwise: float, spanwise: float) -> _Sampled:
    """Resample the source nodes (K, n, 3) along both index directions."""
    k, n = grid.ids.shape
    s, new_le = _chordwise_params(grid, chordwise)
    if grid.smooth:
        across = periodic(source.transpose(1, 0, 2), s).transpose(1, 0, 2)
    else:
        chain = numpy.concatenate([source, source[:, :1]], axis=1) if grid.wrap else source
        across = not_a_knot(chain.transpose(1, 0, 2), s).transpose(1, 0, 2)
    t = _spanwise_params(k, spanwise)
    nodes = not_a_knot(across, t)
    old_k = numpy.minimum(numpy.floor(0.5 * (t[:-1] + t[1:])).astype(numpy.int64), k - 2)
    ends = numpy.append(s, n) if grid.wrap else s
    mids = numpy.floor(0.5 * (ends[:-1] + ends[1:])).astype(numpy.int64)
    old_i = mids % n if grid.wrap else numpy.minimum(mids, n - 2)
    idx = numpy.arange(nodes.shape[0] * nodes.shape[1]).reshape(nodes.shape[:2])
    return _Sampled(nodes, idx, new_le, old_k, old_i)


def _lateral_faces(sampled: _Sampled, grid: Grid) -> Faces:
    """Return the new lateral cells, each split along its source cell's diagonal."""
    idx = sampled.idx
    kp, nc = idx.shape
    diag = None
    if grid.cell_diag is not None:
        diag = grid.cell_diag[numpy.ix_(sampled.old_k, sampled.old_i)]
    out: Faces = []
    for k in range(kp - 1):
        for i in range(nc if grid.wrap else nc - 1):
            j = (i + 1) % nc
            a, b, c, d = (int(x) for x in (idx[k, i], idx[k, j], idx[k + 1, j], idx[k + 1, i]))
            if diag is None:
                out.append([a, b, c, d])
            elif diag[k, i] == 0:
                out += [[a, b, c], [a, c, d]]
            else:
                out += [[a, b, d], [b, c, d]]
    return out


def _zipper_cap(ring: list[int], h: int) -> Faces:
    """Return the cap pairing the upper node i of a ring with the lower node n - i."""
    nc = len(ring)
    cap = [[ring[0], ring[1], ring[nc - 1]]]
    cap += [[ring[i], ring[i + 1], ring[nc - i - 1], ring[nc - i]] for i in range(1, h - 1)]
    cap.append([ring[h - 1], ring[h], ring[h + 1]])
    return cap


def _end_faces(sampled: _Sampled, grid: Grid, poles: tuple[int | None, int | None]) -> Faces:
    """Return the pole fans and the zipper caps of the new grid's ends.

    ``poles`` holds the new index of each end's pole (station 0, the far
    end), None where the end has none.
    """
    out: Faces = []
    for end, k, pole in zip(grid.ends, (0, sampled.idx.shape[0] - 1), poles, strict=True):
        ring = [int(v) for v in sampled.idx[k]]
        nc = len(ring)
        if end == "pole" and pole is not None:
            fan = [[ring[i], ring[(i + 1) % nc], pole] for i in range(nc)]
            # station 0 runs ring[i] -> ring[i+1] in its lateral faces, so its fan runs back
            out += [f[::-1] for f in fan] if k == 0 else fan
        elif end == "zipper":
            cap = _zipper_cap(ring, sampled.new_le)
            # station 0 runs ring[i] -> ring[i+1] in its lateral faces, so its cap runs back
            out += [f[::-1] for f in cap] if k == 0 else cap
    return out


def refine_grid(
    verts: Points,
    faces: Faces,
    grid: Grid,
    *,
    chordwise: float,
    spanwise: float,
    family: str = "family",
) -> GridLevel:
    """Resample a grid family by its chordwise and spanwise factors.

    Parameters
    ----------
    verts : numpy.ndarray
        All the mesh's vertices, shape (V, 3).
    faces : list of list of int
        The family's source faces, as indices into ``verts``.
    grid : Grid
        The family's grid, from :func:`recover_grid`.
    chordwise, spanwise : float
        The factor of each index direction.
    family : str
        The family's name, for a refusal.

    Returns
    -------
    GridLevel
        The new points, faces, trailing-edge mid-points and report.

    Raises
    ------
    InputArtifactError
        The refusal of :func:`check_factors`.
    """
    check_factors(grid, family, chordwise=chordwise, spanwise=spanwise)
    verts = numpy.asarray(verts, dtype=float)
    sampled = _resample(verts[grid.ids], grid, chordwise, spanwise)
    points = sampled.nodes.reshape(-1, 3)
    new_poles: list[int | None] = [None, None]
    for end in (1, 0):
        pole = grid.poles[end]
        if pole is not None:
            points = numpy.vstack([points, verts[pole][None, :]])
            new_poles[end] = len(points) - 1
    poles = (new_poles[0], new_poles[1])
    copied = _identity_copy(grid, faces, sampled, verts[grid.ids], poles)
    if copied is not None:
        ordered, swept_order = copied, _source_sweep(grid, faces) is not None
    else:
        built = _lateral_faces(sampled, grid) + _end_faces(sampled, grid, poles)
        built = orient_like(points, built, verts, faces)
        swept = _emit_like(grid, faces, sampled, built, (points, verts))
        ordered = swept if swept is not None else order_like(points, built, verts, faces)
        swept_order = swept is not None
        if grid.wrap:
            new = _EndFrame.of(sampled.idx, sampled.new_le, poles)
            old = _EndFrame.of(grid.ids, grid.i_le, grid.poles)
            ordered = _align_end_rotation(ordered, new, old, faces, (points, verts))
    te = 0.5 * (sampled.nodes[:-1, 0] + sampled.nodes[1:, 0])
    if grid.smooth:
        te = numpy.zeros((0, 3))
    report = _report(grid, sampled, len(faces), len(ordered), swept_order)
    return GridLevel(points, ordered, te, report)


def _identity_copy(
    grid: Grid,
    faces: Faces,
    sampled: _Sampled,
    source: Points,
    poles: tuple[int | None, int | None],
) -> Faces | None:
    """Return the source's faces on the new nodes when the resampling kept every node.

    At factor 1 in both directions the new node ``(k, i)`` is the source's
    node ``(k, i)``, so each source face is written as it is: in its order,
    with its winding and its own start vertex, cap faces included. Return
    None when the resampling moved, added or removed a node.
    """
    if not numpy.array_equal(sampled.nodes, source):
        return None
    new_of = {int(v): int(w) for v, w in zip(grid.ids.ravel(), sampled.idx.ravel(), strict=True)}
    for old, new in zip(grid.poles, poles, strict=True):
        if old is not None and new is not None:
            new_of[int(old)] = new
    return [[new_of[v] for v in f] for f in faces]


def _report(
    grid: Grid, sampled: _Sampled, before: int, after: int, swept: bool
) -> dict[str, object]:
    """Return the family's entry of refine.json."""
    kp, nc = sampled.idx.shape
    chord_after = nc if grid.wrap else nc - 1
    chord, span = grid.intervals
    return {
        "layout": grid.layout,
        "ends": list(grid.ends),
        "split": "quadrilaterals" if grid.cell_diag is None else "triangles",
        "grid": f"{grid.ids.shape[0]}x{grid.ids.shape[1]} -> {kp}x{nc}",
        "intervals": {"chordwise": [chord, chord_after], "spanwise": [span, kp - 1]},
        "faces": [before, after],
        "order": "sweep" if swept else "nearest",
        "chordwise_spline": "periodic" if grid.smooth else "not-a-knot",
    }


# ------------------------------------------------------------- the face order


@dataclass(frozen=True, eq=False)
class _Sweep:
    """The sweep the source's lateral faces are written in."""

    k_major: bool
    i_dir: int
    k_dir: int
    start_i: int
    templates: dict[tuple[int, int], Template]
    end_faces: list[int]
    ends_first: bool


def _source_cells(
    faces: Faces, where: Mapping[int, tuple[int, int]], n: int, wrap: bool
) -> tuple[dict[tuple[int, int], list[int]], list[int]]:
    """Return the source faces of each lateral cell, and the positions of the end faces."""
    cells: dict[tuple[int, int], list[int]] = {}
    end_faces: list[int] = []
    for pos, f in enumerate(faces):
        ki = [where.get(v) for v in f]
        located = [x for x in ki if x is not None]
        ks = sorted({k for k, _ in located})
        cols = sorted({i for _, i in located})
        if len(located) != len(f) or len(ks) != 2 or ks[1] != ks[0] + 1 or len(cols) != 2:
            end_faces.append(pos)
            continue
        a, b = cols
        cells.setdefault((ks[0], b if (wrap and a == 0 and b == n - 1) else a), []).append(pos)
    return cells, end_faces


def _sequence(count: int, start: int, step: int, wrap: bool) -> list[int]:
    """Return the cells of one index direction in the order a sweep visits them."""
    if wrap:
        return [(start + step * t) % count for t in range(count)]
    return list(range(count)) if step == 1 else list(range(count - 1, -1, -1))


def _cells_in_order(ks: list[int], cols: list[int], k_major: bool) -> list[tuple[int, int]]:
    """Return the cells of a sweep: rows of cells, or columns of cells."""
    if k_major:
        return [(k, i) for k in ks for i in cols]
    return [(k, i) for i in cols for k in ks]


def _source_sweep(grid: Grid, faces: Faces) -> _Sweep | None:
    """Return the sweep of the source's lateral faces, or None when they are not one sweep."""
    k, n = grid.ids.shape
    where = {int(v): (kk, ii) for (kk, ii), v in numpy.ndenumerate(grid.ids)}
    cells, end_faces = _source_cells(faces, where, n, grid.wrap)
    if not cells or any(ps != list(range(ps[0], ps[0] + len(ps))) for ps in cells.values()):
        return None
    pos = {c: ps[0] for c, ps in cells.items()}
    # THE SWEEP IS READ FROM ITS START CELL, NOT FROM CELLS ZERO AND ONE (F-329): across the
    # ordering seam of a sweep that starts inside a ring those two cells are not neighbors
    # in the file. The first cell in file order gives the start row and the start column,
    # and the direction of the sweep is the combination whose order the source reproduces.
    # ONE ROW OR ONE COLUMN OF CELLS STATES NO STEP ACROSS IT: such a source
    # runs along its one direction, forward across the other.
    start_k, start_i = min(pos, key=pos.__getitem__)
    if start_k not in (0, k - 2):
        return None
    k_dir = 1 if start_k == 0 else -1
    one_column = (0, 1) not in pos and (1, 0) in pos
    order = (False, True) if one_column else (True, False)
    expected = sorted(cells, key=pos.__getitem__)
    ks = _sequence(k - 1, 0, k_dir, False)
    for k_major in order:
        for i_dir in (1, -1):
            cols = _sequence(grid.intervals[0], start_i, i_dir, grid.wrap)
            if _cells_in_order(ks, cols, k_major) == expected:
                break
        else:
            continue
        break
    else:
        return None
    templates = {
        c: _template(c, [faces[p] for p in ps], where, n, grid.wrap) for c, ps in cells.items()
    }
    ends_first = bool(end_faces) and min(end_faces) < min(pos.values())
    return _Sweep(k_major, i_dir, k_dir, start_i, templates, end_faces, ends_first)


def _template(
    cell: tuple[int, int],
    cell_faces: Faces,
    where: Mapping[int, tuple[int, int]],
    n: int,
    wrap: bool,
) -> Template:
    """Return a cell's faces in file order, each vertex as its offset from the cell's origin."""
    k0, i0 = cell
    out = []
    for f in cell_faces:
        offsets = []
        for v in f:
            k, i = where[v]
            offsets.append((k - k0, (i - i0) % n if wrap else i - i0))
        out.append(tuple(offsets))
    return tuple(out)


def _new_start(sweep: _Sweep, grid: Grid, ncell: int) -> int:
    """Return the new chordwise cell the sweep starts at."""
    old = grid.intervals[0]
    if sweep.start_i == 0:
        return 0
    if grid.wrap and sweep.start_i == old - 1:
        return ncell - 1
    return round(sweep.start_i * (ncell - 1) / max(1, old - 1))


def _emit_like(
    grid: Grid, faces: Faces, sampled: _Sampled, built: Faces, geometry: tuple[Points, Points]
) -> Faces | None:
    """Write the new grid in the source's sweep, or return None when the source is not one."""
    sweep = _source_sweep(grid, faces)
    if sweep is None:
        return None
    points, verts = geometry
    idx = sampled.idx
    kp, nc = idx.shape
    ncell = nc if grid.wrap else nc - 1
    uniform = set(sweep.templates.values())
    ks = _sequence(kp - 1, 0, sweep.k_dir, False)
    cols = _sequence(ncell, _new_start(sweep, grid, ncell), sweep.i_dir, grid.wrap)
    lateral: Faces = []
    for k, i in _cells_in_order(ks, cols, sweep.k_major):
        cell = (int(sampled.old_k[k]), int(sampled.old_i[i]))
        template = next(iter(uniform)) if len(uniform) == 1 else sweep.templates.get(cell)
        if template is None:
            return None
        for tf in template:
            lateral.append(
                [int(idx[k + dk, (i + di) % nc if grid.wrap else i + di]) for dk, di in tf]
            )
    held = {tuple(sorted(f)) for f in lateral}
    rest = [f for f in built if tuple(sorted(f)) not in held]
    if len(lateral) + len(rest) != len(built):
        return None
    if rest and sweep.end_faces:
        rest = order_like(points, rest, verts, [faces[p] for p in sweep.end_faces])
    return rest + lateral if sweep.ends_first else lateral + rest


@dataclass(frozen=True, eq=False)
class _EndFrame:
    """The two end rings of a tube, for naming the role of each vertex of an end face."""

    vmap: dict[int, tuple[int, int]]
    h: int
    nc: int
    poles: tuple[int, ...]

    @classmethod
    def of(
        cls, ids: NDArray[numpy.int64], h: int, poles: tuple[int | None, int | None]
    ) -> _EndFrame:
        """Return the frame of a grid's first and last stations and its poles."""
        rings = {0: ids[0], 1: ids[-1]}
        vmap = {int(v): (e, p) for e, r in rings.items() for p, v in enumerate(r)}
        return cls(vmap, h, ids.shape[1], tuple(int(p) for p in poles if p is not None))

    def roles(self, face: list[int]) -> tuple[int, str, list[str]] | None:
        """Return (end, kind, role of each vertex) of an end face, or None for a lateral face."""
        pole = next((p for p in self.poles if p in face), None)
        if pole is not None:
            return self._fan(face, pole)
        m = [self.vmap.get(v) for v in face]
        located = [x for x in m if x is not None]
        if len(located) != len(face) or len({e for e, _ in located}) != 1:
            return None
        pos = [p for _, p in located]
        kind = _cap_kind(len(face), pos, self.h)
        if kind is None:
            return None
        up = sorted(p for p in pos if 0 < p < self.h)
        lo = sorted((p for p in pos if p > self.h), reverse=True)
        return located[0][0], kind, [_cap_role(p, self.h, kind == "quad", up, lo) for p in pos]

    def _fan(self, face: list[int], pole: int) -> tuple[int, str, list[str]] | None:
        """Return the roles of a fan triangle: the pole, then the ring nodes in ring order."""
        ring = [self.vmap.get(v) for v in face if v != pole]
        located = [x for x in ring if x is not None]
        if len(located) != 2 or located[0][0] != located[1][0]:
            return None
        (e, p), (_, q) = located
        r0 = p if (p + 1) % self.nc == q else q
        roles = ["P" if v == pole else ("R0" if self.vmap[v][1] == r0 else "R1") for v in face]
        return e, "fan", roles


def _cap_kind(size: int, pos: list[int], h: int) -> str | None:
    """Return the kind of a cap face: the trailing-edge or leading-edge triangle, or a quad."""
    if size == 3 and 0 in pos:
        return "tri_te"
    if size == 3 and h in pos:
        return "tri_le"
    return "quad" if size == 4 else None


def _cap_role(p: int, h: int, quad: bool, up: list[int], lo: list[int]) -> str:
    """Return the role of a ring position in a cap face."""
    if p == 0:
        return "TE"
    if p == h:
        return "LE"
    if p < h:
        return ("U0" if p == up[0] else "U1") if quad else "U"
    return ("L0" if p == lo[0] else "L1") if quad else "L"


def _align_end_rotation(
    new_faces: Faces,
    new: _EndFrame,
    old: _EndFrame,
    old_faces: Faces,
    geometry: tuple[Points, Points],
) -> Faces:
    """Start every new end face on the vertex role its nearest source end face starts on.

    The source end face is the one of the same end and kind (fan triangle,
    trailing-edge or leading-edge cap triangle, cap quadrilateral) whose
    centroid is nearest to the new face's, so each face follows its own
    source face and never the most common start of its kind.
    """
    points, verts = geometry
    found: dict[tuple[int, str], tuple[list[str], list[Points]]] = {}
    for f in old_faces:
        r = old.roles(f)
        if r:
            roles, centroids = found.setdefault((r[0], r[1]), ([], []))
            roles.append(r[2][0])
            centroids.append(verts[f].mean(axis=0))
    near = {key: (roles, numpy.asarray(c)) for key, (roles, c) in found.items()}
    out: Faces = []
    for f in new_faces:
        r = new.roles(f)
        source = near.get((r[0], r[1])) if r else None
        role = None
        if source is not None:
            gap = numpy.linalg.norm(source[1] - points[f].mean(axis=0), axis=1)
            role = source[0][int(numpy.argmin(gap))]
        if r and role in r[2]:
            s = r[2].index(role)
            f = f[s:] + f[:s]
        out.append(f)
    return out
