"""The multiblock grid of an all-quadrilateral family, recovered and resampled..

A family of quadrilaterals that is not one sheet or one tube (a nacelle whose
pylon junction cuts a hole, a fuselage sheet whose boundary has six corners)
is still structured piecewise. Its LAYOUT is found from the connectivity
alone:

1. the SINGULAR vertices are the interior vertices with other than four
   edges or faces, and the boundary vertices with other than two faces (a
   convex corner has one, a re-entrant corner three);
2. from each singular vertex, along each of its interior edges, a SEPARATRIX
   follows the grid line straight through every regular vertex (the edge of
   the four that shares no face with the edge it arrived by) until it reaches
   the boundary or a singular vertex. A grid line through an edge sharper than
   ``_geometry.RIDGE_DEGREES``, or through a trailing-edge edge, is traced the
   same way in both directions, so no spline crosses a ridge;
3. the separatrices and the boundary cut the family into PATCHES, each a
   rectangle of rows by columns of quadrilaterals, and the cut into ARCS,
   each the whole side of one patch or the side two patches share. Lines
   traced to their ends cross one another rather than stopping, so a side
   never meets a corner of the patch beside it: the arcs are the sides;
4. a cut line no patch corner needs, between patches of one family and ending
   on the boundary where the sides beside it run straight on, is joined
   across (:func:`_loose_lines`), so a vertex where two boundaries
   only touch does not leave the family in patches of one interval.

Resampling keeps that structure: each arc is resampled ONCE, by the
cubic spline of :func:`._grid.not_a_knot` through its source nodes in index
space, to ``round(f m)`` intervals of its m; opposite sides of a patch hold
the same m, so they agree. A patch's interior is the tensor spline of its
nodes, as a single grid's is, and its boundary nodes are the arcs' nodes, so
two patches sharing an arc share its nodes exactly. The source's nodes at
integer parameters are kept bit for bit, the layout's corners among them.
The count of each arc is given per arc (:func:`resample`): ``factor`` applies
to every arc, and ``chordwise`` and ``spanwise`` by the arc's direction, read
from the family's trailing edge (:func:`te_directions`).

Face order: every new face takes the position of the source face whose
centroid is nearest (``_geometry.order_like``), each patch oriented like the
source faces it replaces; at factor 1 the family is the source in coordinates
and in order. A family with a face that is not a quadrilateral, or whose
layout is one patch (a sheet or a tube, recovered or refused by
:mod:`._grid`), or whose cut leaves a patch that is not four-sided, has no
multiblock grid, and falls back as before.

GRIDS THAT SHARE NODES ARE RESAMPLED TOGETHER. The grid families of
quadrilaterals that touch, sheets and multiblock families alike, are taken
as one surface: the edges where two of them meet are cut, and the grid line
through each is traced on to its end like a separatrix
(:func:`recover_joined`), so the union is cut into four-sided
patches each inside one family, and every curve two families share is the
side of a patch, an ARC, resampled ONCE: both families hold the same nodes on
it. Each new cell is written back to the family of its patch, oriented like
that patch's source faces, and each family's faces are written in its own
source order (the nearest source face's position). A lone multiblock
family is the same layout with one family.

THE COUNT OF AN ARC is ``max(1, round(f m))`` of its m source
intervals, f the factor of each family holding it in the arc's direction
(``factor`` in both; a sheet's ``chordwise`` along its stations and
``spanwise`` along its columns, a multiblock family's by its trailing
edge, :func:`te_directions`). Two families giving one arc different counts
are refused naming both and the curve, before any work. A side whose
``f m`` is below 1 is kept and reported, as on a lone multiblock family.

A family that cannot join a layout, a tube or a family with a triangle, and
the touching grids whose union is not a layout of four-sided patches, are left
to the rule before this one: the first
of them in the source's family order is a grid, and a later one sharing its
nodes is remeshed, or refused under ``method = "grid"``.

This module imports neither scipy nor rtree.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy
from numpy.typing import NDArray

from cfdbender._errors import GeometryWarning, warn
from cfdbender.mesh._refine._geometry import (
    RIDGE_DEGREES,
    Faces,
    Points,
    dihedral_degrees,
    edge_faces,
    order_like,
)
from cfdbender.mesh._refine._grid import Grid, GridLevel, not_a_knot, recover_grid

Edge = tuple[int, int]
#: The four sides of a patch: its first row, its last row, its first column, its last column.
SIDES = ("first row", "last row", "first column", "last column")


class _NoLayoutError(Exception):
    """The reason, as its message, why a family's multiblock layout was not recovered."""


class _OnePatchError(Exception):
    """The layout is a single patch: the family is a sheet or a tube, judged by :mod:`._grid`."""


@dataclass(frozen=True, eq=False)
class Blocks:
    """The multiblock grid of one family, as recovered from its connectivity.

    Attributes
    ----------
    patches : tuple of numpy.ndarray
        Per patch, the source vertex indices, shape (rows + 1, columns + 1).
    cells : tuple of numpy.ndarray
        Per patch, the source face of each cell, shape (rows, columns).
    arcs : tuple of tuple of int
        The source vertex chain of each arc, from one patch corner to another.
    sides : tuple of tuple of (int, bool)
        Per patch, for each of :data:`SIDES`, its arc and whether the side runs
        against the arc's direction.
    te_arcs : frozenset of int
        The arcs whose every edge joins two trailing-edge vertices.
    """

    patches: tuple[NDArray[numpy.int64], ...]
    cells: tuple[NDArray[numpy.int64], ...]
    arcs: tuple[tuple[int, ...], ...]
    sides: tuple[tuple[tuple[int, bool], ...], ...]
    te_arcs: frozenset[int]

    layout = "multiblock"

    def describe(self) -> str:
        """Return the recovery reason of a multiblock grid that was recovered."""
        return f"a multiblock of {len(self.patches)} four-sided patches of quadrilaterals"


# ------------------------------------------------------------- the recovery


def recover_family(
    verts: Points, faces: Faces, te_vertices: Collection[int]
) -> Grid | Blocks | str:
    """Recover a family's grid: a sheet or a tube, else a multiblock grid of quadrilaterals.

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
    Grid or Blocks or str
        The grid, or the reason no grid was recovered. A family whose faces
        are not all quadrilaterals, or whose layout is one patch, gets the
        reason of :func:`._grid.recover_grid` alone.
    """
    found = recover_grid(verts, faces, te_vertices)
    if isinstance(found, Grid) or any(len(f) != 4 for f in faces):
        return found
    try:
        return _recover(numpy.asarray(verts, dtype=float), faces, frozenset(te_vertices))
    except _OnePatchError:
        return found
    except _NoLayoutError as why:
        return f"{found}; no multiblock layout: {why}"


def _key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, eq=False)
class _Topology:
    """The connectivity the layout is traced on."""

    faces: Faces
    emap: Mapping[Edge, list[int]]
    adj: Mapping[int, set[int]]
    stops: frozenset[int]  # the boundary and singular vertices, where a separatrix ends
    singular: tuple[int, ...]

    @classmethod
    def of(cls, faces: Faces) -> _Topology:
        """Return the topology of a family of quadrilaterals."""
        emap = edge_faces(faces)
        if any(len(fs) > 2 for fs in emap.values()):
            raise _NoLayoutError("an edge is shared by more than two faces")
        adj: dict[int, set[int]] = defaultdict(set)
        for a, b in emap:
            adj[a].add(b)
            adj[b].add(a)
        count = Counter(v for f in faces for v in f)
        boundary = {v for e, fs in emap.items() if len(fs) == 1 for v in e}
        singular = sorted(
            v
            for v in adj
            if (v in boundary and count[v] != 2)
            or (v not in boundary and (len(adj[v]) != 4 or count[v] != 4))
        )
        return cls(faces, emap, adj, frozenset(boundary | set(singular)), tuple(singular))

    def straight(self, a: int, b: int) -> int:
        """Return the vertex after b on the grid line a-b, through the regular vertex b."""
        beside = {x for k in self.emap[_key(a, b)] for x in _quad_neighbours(self.faces[k], b)}
        ahead = self.adj[b] - beside - {a}
        if len(ahead) != 1:
            raise _NoLayoutError(f"the grid line through vertex {b + 1} does not go straight on")
        return ahead.pop()

    def trace(self, a: int, b: int) -> list[Edge]:
        """Return the edges of the separatrix leaving a along a-b, to its end."""
        out: list[Edge] = []
        seen: set[tuple[int, int]] = set()
        while (a, b) not in seen:
            seen.add((a, b))
            out.append(_key(a, b))
            if b in self.stops:
                break
            a, b = b, self.straight(a, b)
        return out


def _quad_neighbours(quad: Sequence[int], v: int) -> tuple[int, int]:
    """Return the two vertices next to v around a face."""
    i = list(quad).index(v)
    return quad[i - 1], quad[(i + 1) % len(quad)]


def _seeds(
    verts: Points, topo: _Topology, te: frozenset[int], lines: frozenset[Edge] | None = None
) -> list[tuple[int, int]]:
    """Return the first edge of every separatrix: the singular vertices', then the ridges'."""
    seeds = [
        (v, u)
        for v in topo.singular
        for u in sorted(topo.adj[v])
        if len(topo.emap[_key(v, u)]) == 2
    ]
    for a, b in sorted(_features(verts, topo, te, lines)):
        seeds += [(a, b), (b, a)]
    return seeds


def _features(
    verts: Points, topo: _Topology, te: frozenset[int], lines: frozenset[Edge] | None = None
) -> set[Edge]:
    """Return the ridge and trailing-edge edges, each traced as a separatrix both ways.

    Given ``lines`` (a joined layout's reference edges), a trailing-edge
    edge is one of them; otherwise it is an edge whose two ends are in ``te``.
    """
    sharp = dihedral_degrees(verts, topo.faces)
    return {
        e
        for e, angle in sharp.items()
        if angle > RIDGE_DEGREES or (e in lines if lines is not None else e[0] in te and e[1] in te)
    }


#: A family's boundary is cut into sides where it turns by more than this, in degrees.
#: A turn at or under it does not split a side, so patches there still count as one side
#: (the same rule the side split uses, and the loose-line test of :class:`_Cuts`).
SIDE_CORNER_DEGREES = 45.0


@dataclass(frozen=True, eq=False)
class _Cuts:
    """The arcs of a traced layout, the patches holding each, for finding loose lines."""

    verts: Points
    arcs: Sequence[tuple[int, ...]]
    sides: Sequence[Sequence[tuple[int, bool]]]
    held: Mapping[int, list[int]]

    def rail(self, p: int, n: int, e: int) -> int | None:
        """Return the side of patch p other than arc n that ends at vertex e."""
        arcs = self.arcs
        found = [a for a, _ in self.sides[p] if a != n and e in (arcs[a][0], arcs[a][-1])]
        return found[0] if len(found) == 1 else None

    def straight(self, a: int, b: int, e: int) -> bool:
        """Return whether arcs a and b, both leaving e, run on within SIDE_CORNER_DEGREES."""
        u, w = (
            self.verts[self.arcs[x][1 if self.arcs[x][0] == e else -2]] - self.verts[e]
            for x in (a, b)
        )
        norm = max(float(numpy.linalg.norm(u) * numpy.linalg.norm(w)), 1e-300)
        return -float(u @ w) / norm >= math.cos(math.radians(SIDE_CORNER_DEGREES))

    def beyond(self, n: int, e: int) -> int | None:
        """Return n at a free end at e, the arc it runs on into across four patches, else None."""
        p, q = self.held[n]
        rp, rq = self.rail(p, n, e), self.rail(q, n, e)
        if rp is None or rq is None or rp == rq:
            return None
        if self.held[rp] == [p] and self.held[rq] == [q]:
            return n if self.straight(rp, rq, e) else None
        p2 = next((x for x in self.held[rp] if x != p), None)
        q2 = next((x for x in self.held[rq] if x != q), None)
        if len(self.held[rp]) != 2 or len(self.held[rq]) != 2 or p2 is None or q2 is None:
            return None
        c = self.rail(p2, rp, e)
        ahead = {p2, q2}
        if len(ahead) != 2 or ahead & {p, q} or c is None or c != self.rail(q2, rq, e):
            return None
        return c if set(self.held[c]) == ahead else None

    def eligible(self, n: int, kept: Collection[Edge]) -> bool:
        """Return whether arc n lies between two patches and holds no kept edge."""
        arc = self.arcs[n]
        return (
            len(self.held[n]) == 2
            and arc[0] != arc[-1]
            and not any(_key(a, b) in kept for a, b in zip(arc, arc[1:], strict=False))
        )


def _root(parent: dict[int, int], n: int) -> int:
    while parent[n] != n:
        parent[n] = parent[parent[n]]
        n = parent[n]
    return n


def _loose_lines(
    verts: Points,
    arcs: Sequence[tuple[int, ...]],
    sides: Sequence[Sequence[tuple[int, bool]]],
    kept: Collection[Edge],
) -> set[Edge]:
    """Return the edges of the cut lines no patch corner needs, so their patches join.

    A separatrix from a vertex where two boundaries only touch (a node a
    neighbour holds on a curve the family spans with one edge) cuts the family
    across its grid lines where it has no corner, and every patch side of one
    interval keeps its count. A LOOSE line is a chain of arcs, each between two
    patches of one family and holding no ridge, trailing-edge or join edge
    (``kept``), continued through vertices where four patches meet, whose two
    ends are boundary vertices where the sides beside it are held by those
    patches alone and run on within ``SIDE_CORNER_DEGREES``. Joining across it
    leaves every arc a whole side of its patches.
    """
    held: dict[int, list[int]] = defaultdict(list)
    for p, mine in enumerate(sides):
        for arc, _ in mine:
            held[arc].append(p)
    cuts = _Cuts(verts, arcs, sides, held)
    parent = {n: n for n in range(len(arcs)) if cuts.eligible(n, kept)}
    bad: set[int] = set()
    for n in list(parent):
        for e in (arcs[n][0], arcs[n][-1]):
            c = cuts.beyond(n, e)
            if c is None or c not in parent:
                bad.add(n)
            else:
                parent[_root(parent, c)] = _root(parent, n)
    spoiled = {_root(parent, n) for n in bad}
    loose = [n for n in parent if _root(parent, n) not in spoiled]
    return {_key(a, b) for n in loose for a, b in zip(arcs[n], arcs[n][1:], strict=False)}


def recover_joined(
    verts: Points,
    faces: Faces,
    te_vertices: Collection[int],
    joins: Collection[Edge],
    te_lines: Collection[Edge] | None = None,
) -> Blocks | str:
    """Recover the multiblock grid of several families' quadrilaterals taken as one.

    ``joins`` are the edges where two of the families meet: they are cut, and
    the grid line through each is traced on in both directions like a
    separatrix, so every patch lies inside one family and every curve two
    families share is the side of a patch. ``te_lines``, when given, are the
    families' reference edges: only they are traced as trailing edges, never an
    edge that merely joins two reference nodes, such as each edge across a
    one-element strip between two reference lines. Returns the
    reason, as text, when no such layout exists.
    """
    if any(len(f) != 4 for f in faces):
        return "a face is not a quadrilateral"
    lines = None if te_lines is None else frozenset(_key(a, b) for a, b in te_lines)
    try:
        return _recover(
            numpy.asarray(verts, dtype=float),
            faces,
            frozenset(te_vertices),
            frozenset(joins),
            lines,
        )
    except _OnePatchError:
        return "the faces form one patch"
    except _NoLayoutError as why:
        return str(why)


def _recover(
    verts: Points,
    faces: Faces,
    te: frozenset[int],
    joins: frozenset[Edge] = frozenset(),
    lines: frozenset[Edge] | None = None,
) -> Blocks:
    """Return the multiblock grid of a family of quadrilaterals or raise why it has none."""
    topo = _Topology.of(faces)
    cut = {e for e, fs in topo.emap.items() if len(fs) == 1} | joins
    seeds = _seeds(verts, topo, te, lines)
    seeds += [s for a, b in sorted(joins) for s in ((a, b), (b, a))]
    for a, b in seeds:
        cut.update(topo.trace(a, b))
    groups = _patches(topo, cut)
    if len(groups) < 2:
        raise _OnePatchError
    grids = [_patch_grid(topo, group, cut) for group in groups]
    arcs, sides = _arcs([ids for ids, _ in grids], cut)
    loose = _loose_lines(verts, arcs, sides, _features(verts, topo, te, lines) | joins)
    if loose:
        try:
            merged = _patches(topo, cut - loose)
            if len(merged) >= 2:
                grids = [_patch_grid(topo, group, cut - loose) for group in merged]
                arcs, sides = _arcs([ids for ids, _ in grids], cut - loose)
        except _NoLayoutError:
            pass  # the joined patches are not four-sided: the layout stays as traced
    te_arcs = frozenset(n for n, arc in enumerate(arcs) if all(v in te for v in arc))
    return Blocks(
        tuple(ids for ids, _ in grids),
        tuple(cells for _, cells in grids),
        arcs,
        sides,
        te_arcs,
    )


def sheet_blocks(grid: Grid, faces: Faces, te: Collection[int]) -> Blocks:
    """Represent a recovered quadrilateral sheet as one planning patch."""
    ids = grid.ids
    lookup = {frozenset(f): i for i, f in enumerate(faces)}
    cells = numpy.array(
        [
            [
                lookup[frozenset(int(v) for v in ids[k : k + 2, i : i + 2].flat)]
                for i in range(ids.shape[1] - 1)
            ]
            for k in range(ids.shape[0] - 1)
        ],
        dtype=numpy.int64,
    )
    boundary = {e for e, fs in edge_faces(faces).items() if len(fs) == 1}
    arcs, sides = _arcs([ids], boundary)
    return Blocks(
        (ids,),
        (cells,),
        arcs,
        sides,
        frozenset(a for a, arc in enumerate(arcs) if all(v in te for v in arc)),
    )


def _patches(topo: _Topology, cut: set[Edge]) -> list[list[int]]:
    """Return the faces of each patch, joined across the edges not cut, by first face."""
    parent = list(range(len(topo.faces)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for e, fs in topo.emap.items():
        if len(fs) == 2 and e not in cut:
            parent[root(fs[0])] = root(fs[1])
    groups: dict[int, list[int]] = defaultdict(list)
    for k in range(len(topo.faces)):
        groups[root(k)].append(k)
    return sorted(groups.values(), key=lambda g: g[0])


def _across(
    topo: _Topology, face: int, u: int, v: int, cut: set[Edge]
) -> tuple[int, int, int] | None:
    """Return the face beyond the edge u-v of a patch and its vertices next to u and to v."""
    if _key(u, v) in cut:
        return None
    other = [k for k in topo.emap[_key(u, v)] if k != face]
    if not other:
        return None
    quad = topo.faces[other[0]]
    nu = [x for x in _quad_neighbours(quad, u) if x != v]
    nv = [x for x in _quad_neighbours(quad, v) if x != u]
    return other[0], nu[0], nv[0]


def _first_cell(topo: _Topology, group: list[int]) -> tuple[int, list[list[int]]]:
    """Return the patch's corner cell: the face of its lowest one-face corner, as a 2 by 2 grid."""
    count = Counter(v for k in group for v in topo.faces[k])
    corners = sorted(v for v, c in count.items() if c == 1)
    if len(corners) != 4:
        raise _NoLayoutError(f"a patch has {len(corners)} corners, not four")
    c = corners[0]
    face = next(k for k in group if c in topo.faces[k])
    quad = topo.faces[face]
    i = quad.index(c)
    a, x, b = quad[(i + 1) % 4], quad[(i + 2) % 4], quad[(i + 3) % 4]
    if b < a:
        a, b = b, a
    return face, [[c, a], [b, x]]


def _patch_grid(
    topo: _Topology, group: list[int], cut: set[Edge]
) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64]]:
    """Return a patch's nodes (rows + 1, columns + 1) and its cells' source faces, or raise."""
    face, rows = _first_cell(topo, group)
    cells = [[face]]
    while True:  # the first row of cells, column by column
        nxt = _across(topo, cells[0][-1], rows[0][-1], rows[1][-1], cut)
        if nxt is None:
            break
        if nxt[0] in cells[0]:
            raise _NoLayoutError("a patch closes on itself")
        cells[0].append(nxt[0])
        rows[0].append(nxt[1])
        rows[1].append(nxt[2])
    while True:  # each next row of cells, from the row above
        below = [
            _across(topo, k, rows[-1][j], rows[-1][j + 1], cut) for j, k in enumerate(cells[-1])
        ]
        if all(b is None for b in below):
            break
        if len(cells) * len(cells[0]) >= len(group):
            raise _NoLayoutError("a patch closes on itself")
        cells.append(_row_below(below, rows))
    ids = numpy.array(rows, dtype=numpy.int64)
    held = numpy.array(cells, dtype=numpy.int64)
    if sorted(held.ravel().tolist()) != sorted(group):
        raise _NoLayoutError("a patch is not a rectangle of rows by columns")
    return ids, held


def _row_below(below: list[tuple[int, int, int] | None], rows: list[list[int]]) -> list[int]:
    """Return the next row of cells, appending its nodes to ``rows``, or raise when ragged."""
    if any(b is None for b in below):
        raise _NoLayoutError("a patch is not a rectangle of rows by columns")
    found = [b for b in below if b is not None]
    row = [found[0][1]] + [b[2] for b in found]
    joined = all(found[j][2] == found[j + 1][1] for j in range(len(found) - 1))
    if not joined or len(set(row)) != len(row):
        raise _NoLayoutError("a patch is not a rectangle of rows by columns")
    rows.append(row)
    return [b[0] for b in found]


def _side_chains(ids: NDArray[numpy.int64]) -> list[tuple[int, ...]]:
    """Return a patch's four sides, in the order of :data:`SIDES`."""
    return [tuple(int(v) for v in s) for s in (ids[0], ids[-1], ids[:, 0], ids[:, -1])]


def _arcs(
    patches: list[NDArray[numpy.int64]], cut: set[Edge]
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, bool], ...], ...]]:
    """Return the arcs (each patch side, once) and each patch's sides as (arc, reversed)."""
    index: dict[tuple[int, ...], int] = {}
    holder: dict[Edge, int] = {}
    sides = []
    for ids in patches:
        mine = []
        for chain in _side_chains(ids):
            flip = (chain[0], chain[1]) > (chain[-1], chain[-2])
            canonical = chain[::-1] if flip else chain
            arc = index.setdefault(canonical, len(index))
            for a, b in zip(canonical, canonical[1:], strict=False):
                if holder.setdefault(_key(a, b), arc) != arc:
                    raise _NoLayoutError("two patches meet at a T-junction")
            mine.append((arc, flip))
        sides.append(tuple(mine))
    if set(holder) != cut:
        raise _NoLayoutError("a cut edge lies on no patch side")
    return tuple(index), tuple(sides)


# ------------------------------------------------------------- the factors


#: An arc runs SPANWISE when its edges run, on average weighted by length, within 45
#: degrees of the trailing-edge segment nearest each: the mean of |cos| at least cos 45.
#: Absolute cosine tolerance 1e-14 includes the boundary despite floating-point roundoff.
SPANWISE_COS = math.cos(math.radians(45.0))


def longest_boundary_side(verts: Points, blocks: Blocks, patches: Sequence[int]) -> Points:
    """Return the longest side of a family's boundary as segments, shape (n, 2, 3).

    The boundary is the arcs one patch of the family holds; it is cut into sides
    where it turns by more than :data:`SIDE_CORNER_DEGREES`, and the segments of
    the longest side are returned in their order along it.
    """
    held: dict[int, int] = {}
    for p in patches:
        for arc, _ in blocks.sides[p]:
            held[arc] = held.get(arc, 0) + 1
    touching: dict[int, list[int]] = {}
    for arc, count in held.items():
        if count != 1:
            continue
        chain = list(blocks.arcs[arc])
        for a, b in zip(chain, chain[1:], strict=False):
            touching.setdefault(a, []).append(b)
            touching.setdefault(b, []).append(a)
    pts = numpy.asarray(verts, dtype=float)
    sides = boundary_runs(pts, touching)
    if not sides:  # a closed family: every arc is held by two of its patches
        return numpy.empty((0, 2, 3))
    best = max(
        sides, key=lambda run: float(numpy.linalg.norm(numpy.diff(pts[run], axis=0), axis=1).sum())
    )
    return numpy.stack([pts[best[:-1]], pts[best[1:]]], axis=1)


def boundary_runs(pts: Points, touching: dict[int, list[int]]) -> list[list[int]]:
    """Return the runs of boundary vertices between corners (a turn, or a branch)."""
    most = math.cos(math.radians(SIDE_CORNER_DEGREES))

    def corner(v: int) -> bool:
        n = touching[v]
        if len(n) != 2:
            return True
        u, w = pts[v] - pts[n[0]], pts[n[1]] - pts[v]
        cos = float(u @ w) / max(float(numpy.linalg.norm(u) * numpy.linalg.norm(w)), 1e-300)
        return cos < most

    runs: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    corners = [v for v in touching if corner(v)]
    starts = corners + [v for v in touching if v not in set(corners)]
    for start in starts:
        for nxt in touching[start]:
            if (min(start, nxt), max(start, nxt)) in seen:
                continue
            run, prev, cur = [start], start, nxt
            while True:
                seen.add((min(prev, cur), max(prev, cur)))
                run.append(cur)
                if cur == start or corner(cur):
                    break
                prev, cur = cur, next(n for n in touching[cur] if n != prev)
            runs.append(run)
    return runs


def runs_spanwise(verts: Points, chain: Sequence[int], te: Points) -> bool:
    """Return whether an arc runs along the trailing edge (spanwise) rather than across it.

    ``te`` holds the trailing-edge segments, shape (T, 2, 3). Each edge of the
    arc is compared with the segment nearest its mid-point; the arc's value
    is the length-weighted mean of the absolute cosine between the two, so an
    arc wrapping round a leading edge, whose ends almost meet, is read by its
    edges and not by its chord.
    """
    return along(verts, chain, te) >= SPANWISE_COS - 1e-14


def along(verts: Points, chain: Sequence[int], te: Points) -> float:
    """Return how much an arc runs along the reference segments: the weighted mean |cos|."""
    p = numpy.asarray(verts, dtype=float)[list(chain)]
    edge = p[1:] - p[:-1]
    mid = 0.5 * (p[1:] + p[:-1])
    a, ab = te[:, 0], te[:, 1] - te[:, 0]
    t = ((mid[:, None, :] - a[None]) * ab[None]).sum(axis=2)
    t = numpy.clip(t / numpy.maximum((ab * ab).sum(axis=1), 1e-300)[None], 0.0, 1.0)
    gap = numpy.linalg.norm(a[None] + t[..., None] * ab[None] - mid[:, None, :], axis=2)
    near = ab[numpy.argmin(gap, axis=1)]
    length = numpy.linalg.norm(edge, axis=1)
    scale = numpy.maximum(length * numpy.linalg.norm(near, axis=1), 1e-300)
    cos = numpy.abs((edge * near).sum(axis=1)) / scale
    return float((cos * length).sum() / max(float(length.sum()), 1e-300))


def te_directions(
    verts: Points,
    blocks: Blocks,
    patches: Sequence[int],
    te: Points,
    *,
    family: str,
    what: str,
    patch_runs: dict[int, dict[int, str]] | None = None,
) -> dict[int, str]:
    """Return ``"chordwise"`` or ``"spanwise"`` for each arc of a family's patches.

    Each patch is told by its own two pairs of opposite sides: the pair that runs
    more along the reference line (:func:`along`) is parallel (spanwise), the
    other normal, so its rows and columns always differ. An arc a patch shares
    with one already told keeps its direction and orients the patch. Where the
    shared arcs disagree within a patch, that patch keeps its own geometric
    orientation, warned; nothing is refused. ``patch_runs`` preserves each patch's
    decision where a shared arc has conflicting local orientations.

    Parameters
    ----------
    verts : numpy.ndarray
        All the mesh's vertices, shape (V, 3).
    blocks : Blocks
        The layout.
    patches : sequence of int
        The family's patches, numbered 1, 2, ... in this order in a refusal.
    te : numpy.ndarray
        The family's trailing-edge segments, shape (T, 2, 3).
    family : str
        The source and family, for a refusal.
    what : str
        The family's layout as a refusal names it.
    patch_runs : dict, optional
        Retain each patch's orientation on conflict instead of the earlier family-wide fallback.

    Notes
    -----
    A family with no trailing edge is told against the longest side of its
    boundary (:func:`longest_boundary_side`), warned.
    """
    all_arcs = {arc for q in patches for arc, _ in blocks.sides[q]}
    if not len(te):
        te = longest_boundary_side(verts, blocks, patches)
        if not len(te):
            warn(
                f"{family}: {what} has no trailing edge listed and no boundary to orient it; "
                "every arc of the family takes the parallel factor.",
                GeometryWarning,
                stacklevel=4,
            )
            return dict.fromkeys(all_arcs, "spanwise")
        ends = " and ".join(
            "(" + ", ".join(f"{float(c):.6g}" for c in point) + ")"
            for point in (te[0, 0], te[-1, 1])
        )
        warn(
            f"{family}: no trailing edge is listed for {what}; its arcs are told parallel or "
            f"normal against the longest side of its boundary, {len(te)} edges from {ends} "
            "(mesh units). List its trailing edge in a [[perimeter]] file to orient it by "
            "that line.",
            GeometryWarning,
            stacklevel=4,
        )
    out: dict[int, str] = {}
    for n, p in _telling_order(verts, blocks, patches, te, out):
        arcs = [arc for arc, _ in blocks.sides[p]]
        rows_along = _patch_rows_along(verts, blocks, arcs, te, out)
        if rows_along is None:
            pts = numpy.asarray(verts, dtype=float)[blocks.patches[p].ravel()]
            where = ", ".join(f"{v:.3g}" for v in pts.mean(axis=0))
            warn(
                f"{family}: the sides patch {n} of {what} (centroid {where}) shares with its "
                "neighbours run both parallel and normal; this patch keeps its own "
                "geometric orientation.",
                GeometryWarning,
                stacklevel=4,
            )
            rows_along = _patch_rows_along(verts, blocks, arcs, te, {})
        row, column = ("spanwise", "chordwise") if rows_along else ("chordwise", "spanwise")
        if patch_runs is not None:
            patch_runs[p] = {a: row if i < 2 else column for i, a in enumerate(arcs)}
        for i, arc in enumerate(arcs):
            out.setdefault(arc, row if i < 2 else column)
    return out


def _telling_order(
    verts: Points, blocks: Blocks, patches: Sequence[int], te: Points, told: dict[int, str]
):
    """Yield (number, patch) in the order the patches are told.

    The next patch is one that shares an arc already told, else any; among those,
    the one whose two pairs of sides differ most in how much they run along the
    reference, so the result does not hang on the source's face numbering.
    ``told`` is read as it fills.
    """
    margin = {}
    for p in patches:
        score = [along(verts, blocks.arcs[a], te) for a, _ in blocks.sides[p]]
        margin[p] = abs((score[0] + score[1]) - (score[2] + score[3]))
    left = list(patches)
    while left:
        linked = [p for p in left if any(a in told for a, _ in blocks.sides[p])]
        p = min(linked or left, key=lambda q: (-margin[q], patches.index(q)))
        left.remove(p)
        yield patches.index(p) + 1, p


def _patch_rows_along(
    verts: Points, blocks: Blocks, arcs: list[int], te: Points, told: dict[int, str]
) -> bool | None:
    """Return whether a patch's rows run along the reference; None when its told arcs disagree.

    Arcs already told (shared with an earlier patch) decide; else the pair of
    opposite sides with the larger mean :func:`along` is the one along it.
    """
    votes = {
        ("spanwise" if i < 2 else "chordwise") == told[a] for i, a in enumerate(arcs) if a in told
    }
    if len(votes) > 1:
        return None
    if votes:
        return votes.pop()
    score = [along(verts, blocks.arcs[a], te) for a in arcs]
    return (score[0] + score[1]) >= (score[2] + score[3])


def kept_sides(
    blocks: Blocks,
    verts: NDArray[numpy.float64],
    factor: float | Mapping[int, float],
    *,
    patches: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Return the patch sides a factor below 1/m cannot coarsen, with where they lie.

    A side of m source intervals with ``factor * m < 1`` keeps
    ``max(1, round(factor * m))`` intervals rather than refusing the family,
    so the rest of the family is resampled as asked. Each entry names
    the patch (1-based, in the order of ``patches``), the side, both counts, and
    the patch's centroid and bounding box in the mesh's units (three
    significant figures), so the side can be found in a viewer.

    Parameters
    ----------
    blocks : Blocks
        The recovered multiblock grid.
    verts : numpy.ndarray
        The source vertices, shape (n, 3).
    factor : float or mapping of int to float
        The family's factor, or the factor of each arc of its patches.
    patches : sequence of int, optional
        The family's patches (default: every patch of the layout).

    Returns
    -------
    list of dict
        One entry per kept side, in patch then :data:`SIDES` order; empty when none.
    """

    def _sig(values: NDArray[numpy.float64]) -> list[float]:
        return [float(f"{v:.3g}") for v in values]

    kept: list[dict[str, object]] = []
    mine = range(len(blocks.patches)) if patches is None else patches
    for n, p in enumerate(mine, 1):
        pts = verts[blocks.patches[p].ravel()]
        for side, (arc, _) in zip(SIDES, blocks.sides[p], strict=True):
            m = len(blocks.arcs[arc]) - 1
            f = factor[arc] if isinstance(factor, Mapping) else factor
            if f * m < 1.0:
                kept.append(
                    {
                        "patch": n,
                        "side": side,
                        "source_intervals": m,
                        "intervals": count(m, f),
                        "centroid": _sig(pts.mean(axis=0)),
                        "bbox_min": _sig(pts.min(axis=0)),
                        "bbox_max": _sig(pts.max(axis=0)),
                    }
                )
    return kept


def count(m: int, factor: float) -> int:
    """Return round(f m), at least one interval."""
    return max(1, round(m * factor))


def sheet_directions(grid: Grid, blocks: Blocks, arcs: Collection[int]) -> dict[int, str]:
    """Return ``"chordwise"`` or ``"spanwise"`` for each arc lying in a sheet grid.

    A sheet keeps its index directions when it is resampled with other grids: an arc along one of
    its stations is chordwise, one along a
    column spanwise, as :func:`._grid.refine_grid` reads them.
    """
    place = {int(v): (k, i) for (k, i), v in numpy.ndenumerate(grid.ids)}
    out = {}
    for arc in arcs:
        a, b = (place[v] for v in blocks.arcs[arc][:2])
        out[arc] = "chordwise" if a[0] == b[0] else "spanwise"
    return out


# ------------------------------------------------------------- the resampling


class _Nodes:
    """The level's nodes, each made once and named by where it comes from."""

    def __init__(self) -> None:
        self.index: dict[tuple[object, ...], int] = {}
        self.points: list[Points] = []

    def node(self, key: tuple[object, ...], point: Points) -> int:
        """Return the index of a named node, making it at ``point`` the first time."""
        if key not in self.index:
            self.index[key] = len(self.points)
            self.points.append(point)
        return self.index[key]


def _patch_nodes(
    verts: Points,
    blocks: Blocks,
    p: int,
    arcs: list[Points],
    nodes: _Nodes,
    *,
    counts: Sequence[int],
) -> NDArray[numpy.int64]:
    """Return the level's node index of every point of a patch's new grid.

    Its columns are its first row's arc's count, its rows its first column's.
    """
    ids = blocks.patches[p]
    rows, cols = ids.shape[0] - 1, ids.shape[1] - 1
    s = numpy.linspace(0, cols, counts[blocks.sides[p][0][0]] + 1)
    t = numpy.linspace(0, rows, counts[blocks.sides[p][2][0]] + 1)
    across = not_a_knot(verts[ids].transpose(1, 0, 2), s).transpose(1, 0, 2)
    inside = not_a_knot(across, t)
    kp, cp = inside.shape[:2]
    out = numpy.empty((kp, cp), dtype=numpy.int64)
    corners = {(0, 0): ids[0, 0], (0, cp - 1): ids[0, -1], (kp - 1, 0): ids[-1, 0]}
    corners[(kp - 1, cp - 1)] = ids[-1, -1]
    lines = dict(zip(SIDES, blocks.sides[p], strict=True))
    for r in range(kp):
        for c in range(cp):
            if (r, c) in corners:
                v = int(corners[(r, c)])
                out[r, c] = nodes.node(("v", v), verts[v])
                continue
            on = _on_side(r, c, kp, cp)
            if on is None:
                out[r, c] = nodes.node(("p", p, r, c), inside[r, c])
                continue
            side, at, count = on
            arc, flip = lines[side]
            pos = count - at if flip else at
            out[r, c] = nodes.node(("a", arc, pos), arcs[arc][pos])
    return out


def _on_side(r: int, c: int, kp: int, cp: int) -> tuple[str, int, int] | None:
    """Return the side a new grid point lies on, its position along it and the side's count."""
    if r == 0:
        return SIDES[0], c, cp - 1
    if r == kp - 1:
        return SIDES[1], c, cp - 1
    if c == 0:
        return SIDES[2], r, kp - 1
    if c == cp - 1:
        return SIDES[3], r, kp - 1
    return None


def _forward(quad: Sequence[int], a: int, b: int) -> bool:
    """Return whether a face runs from a to b (rather than from b to a)."""
    i = list(quad).index(a)
    return quad[(i + 1) % len(quad)] == b


@dataclass(frozen=True, eq=False)
class Resampled:
    """A layout resampled: the new nodes, each patch's new faces and shapes, the trailing edge.

    Attributes
    ----------
    points : numpy.ndarray
        The new nodes, shape (P, 3), each arc's made once.
    cells : tuple of list of list of int
        Per patch, its new faces, oriented like its source faces.
    shapes : tuple of tuple
        Per patch, its node array's shape before and after.
    te_midpoints : numpy.ndarray
        One mid-point per new trailing-edge edge, shape (T, 3).
    """

    points: Points
    cells: tuple[Faces, ...]
    shapes: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    te_midpoints: Points


def resample(
    verts: Points,
    faces: Faces,
    blocks: Blocks,
    counts: Sequence[int],
    *,
    keep: Collection[int] = (),
) -> Resampled:
    """Resample every arc of a layout to its count, then every patch inside its arcs.

    ``counts`` holds each arc's new interval count; opposite sides of a patch
    must hold the same. ``faces`` are the faces the layout was recovered on.
    Patches in ``keep`` retain their original faces and coordinates without interpolation.
    """
    verts = numpy.asarray(verts, dtype=float)
    arcs = [
        verts[list(arc)]
        if counts[n] == len(arc) - 1
        else not_a_knot(verts[list(arc)], numpy.linspace(0, len(arc) - 1, counts[n] + 1))
        for n, arc in enumerate(blocks.arcs)
    ]
    nodes = _Nodes()
    cells: list[Faces] = []
    shapes = []
    for p, ids in enumerate(blocks.patches):
        if p in keep:
            cells.append(
                [
                    [nodes.node(("v", v), verts[v]) for v in faces[int(cell)]]
                    for cell in blocks.cells[p].flat
                ]
            )
            shapes.append((tuple(ids.shape), tuple(ids.shape)))
            continue
        grid = _patch_nodes(verts, blocks, p, arcs, nodes, counts=counts)
        same = _forward(faces[int(blocks.cells[p][0, 0])], int(ids[0, 0]), int(ids[0, 1]))
        built: Faces = []
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1] - 1):
                quad = [
                    int(grid[r, c]),
                    int(grid[r, c + 1]),
                    int(grid[r + 1, c + 1]),
                    int(grid[r + 1, c]),
                ]
                built.append(quad if same else [quad[0], quad[3], quad[2], quad[1]])
        cells.append(built)
        shapes.append((tuple(ids.shape), tuple(grid.shape)))
    points = numpy.array(nodes.points, dtype=float).reshape(-1, 3)
    te = [0.5 * (arcs[a][:-1] + arcs[a][1:]) for a in sorted(blocks.te_arcs)]
    midpoints = numpy.vstack(te) if te else numpy.zeros((0, 3))
    return Resampled(points, tuple(cells), tuple(shapes), midpoints)


def report(
    blocks: Blocks, done: Resampled, patches: Sequence[int], faces: tuple[int, int]
) -> dict[str, object]:
    """Return the ``refine.json`` entry of the family holding ``patches`` of a resampled layout."""
    shapes = [done.shapes[p] for p in patches]
    return {
        "layout": "multiblock",
        "split": "quadrilaterals",
        "patches": len(patches),
        "blocks": [
            {"rows": [a[0] - 1, b[0] - 1], "columns": [a[1] - 1, b[1] - 1]} for a, b in shapes
        ],
        "arcs": len({arc for p in patches for arc, _ in blocks.sides[p]}),
        "faces": list(faces),
        "order": "nearest",
    }


def refine_blocks(verts: Points, faces: Faces, blocks: Blocks, *, factor: float) -> GridLevel:
    """Resample a multiblock family by its factor.

    Parameters
    ----------
    verts : numpy.ndarray
        All the mesh's vertices, shape (V, 3).
    faces : list of list of int
        The family's source faces, as indices into ``verts``.
    blocks : Blocks
        The family's layout, from :func:`recover_family`.
    factor : float
        The factor of every arc's interval count.

    Returns
    -------
    GridLevel
        The new points, faces, trailing-edge mid-points and report.
    """
    done = resample(verts, faces, blocks, [count(len(arc) - 1, factor) for arc in blocks.arcs])
    built = [f for cell in done.cells for f in cell]
    ordered = order_like(done.points, built, numpy.asarray(verts, dtype=float), faces)
    entry = report(blocks, done, range(len(blocks.patches)), (len(faces), len(ordered)))
    return GridLevel(done.points, ordered, done.te_midpoints, entry)


# Compatibility exports keep existing planner callers on their established path.
_LAYOUT_EXPORTS = (
    "ArcDirection",
    "Layout",
    "_Planner",
    "_joinable",
    "_touching",
    "_join",
    "_direction_links",
    "propagate_directions",
    "_reference_directions",
    "warn_direction_fallback",
    "_perimeters",
    "arc_bands",
    "_lead_rank",
    "_band_lead",
    "_band_counts",
    "_warn_kept",
    "plan",
    "_skip_unled",
    "_skipped_layout",
    "refine_layout",
    "Recovered",
)


def __getattr__(name: str):
    """Resolve layout exports after the recovery module has initialized."""
    if name in _LAYOUT_EXPORTS:
        from cfdbender.mesh._refine import _layout

        return getattr(_layout, name)
    raise AttributeError(name)
