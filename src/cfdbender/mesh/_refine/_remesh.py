"""Remeshing of the families a refinement does not resample as a grid.

The target edge length at a point is the source's local edge length there
divided by the family's factor, so the source's distribution of sizes is kept
and only scaled. Passes of edge split, edge collapse, edge flip and tangential
smoothing change the panel count and never the shape: every node is projected
back onto the source surface. A family at factor 1 beside a refined side
takes instead, at the nodes of the curve they share, the length of the new
shared edges there, and returns to its own target over
:data:`SHARED_LAYERS` face layers.

CURVES are the edges whose nodes move only along the source curve they lie
on and are split and collapsed along it, never flipped: open boundaries, the
trailing edge, the edges whose dihedral exceeds the ridge angle, and an
interface between two families remeshed together (remeshed once, so both
share it). An interface with a family outside the group is FROZEN: its nodes
never move, unless :meth:`Remesher.conform` first rebuilds it on the new
nodes of a grid family refined before it.

QUAD-DOMINANT families (the refinement file's ``elements``) are remeshed into
triangles first, so sizes, curves, frozen and conformed interfaces and the
trailing edge are all honored as above; their triangles are
then paired into quadrilaterals across edges that are no curve, each pair
merged only into a convex quadrilateral within ``QUAD_ANGLE_BAND`` and
``QUAD_WARP_DEGREES`` of :mod:`._geometry`, best quality first; the
triangles left unpaired stay. Their free nodes are then smoothed and
projected onto the source surface as above, curve, fixed and corner nodes
held, and a move that would break a face is undone. A family in the default
mode, ``"triangles"``, is returned exactly as the remesh ends.

The projection onto the source surface searches a spatial index (scipy and
rtree, through trimesh), and every import of trimesh, scipy and rtree here is
deferred: a refinement of grid families alone imports none of them.

Two sections live in their own modules below this one, and every name of
theirs that other modules read is imported here, so each import path of this
module holds: :mod:`._remesh_group` (the group's local numbering, its targets,
its curves and its source order) and :mod:`._remesh_pairs` (the pairing of
triangles into quadrilaterals, and the element report).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy

from cfdbender.mesh._refine._config import QUAD_DOMINANT, TRIANGLES
from cfdbender.mesh._refine._geometry import (
    QUAD_ANGLE_BAND,
    QUAD_WARP_DEGREES,
    RIDGE_DEGREES,
    Faces,
    Points,
    degenerate,
    min_angle_degrees,
    quad_shapes,
)
from cfdbender.mesh._refine._geometry import (
    SHARED_LAYERS as SHARED_LAYERS,
)

# Preserve this public import path (release parity).
from cfdbender.mesh._refine._geometry import (
    dihedral_degrees as dihedral_degrees,
)
from cfdbender.mesh._refine._geometry import (
    edge_faces as edge_faces,
)
from cfdbender.mesh._refine._geometry import (
    order_like as order_like,
)
from cfdbender.mesh._refine._geometry import (
    orient_like as orient_like,
)
from cfdbender.mesh._refine._obj import ObjMesh
from cfdbender.mesh._refine._remesh_group import (
    BOUNDARY as BOUNDARY,
)
from cfdbender.mesh._refine._remesh_group import (
    FROZEN,
    TRAILING_EDGE,
    Edge,
    Patch,
    _curves,
    _ek,
    _group,
    _ordered,
    _targets,
    shared_targets,
)
from cfdbender.mesh._refine._remesh_group import (
    INTERFACE as INTERFACE,
)
from cfdbender.mesh._refine._remesh_group import (
    RIDGE as RIDGE,
)
from cfdbender.mesh._refine._remesh_pairs import (
    AUGMENT_DEPTH as AUGMENT_DEPTH,
)
from cfdbender.mesh._refine._remesh_pairs import (
    _face_normal,
    _matching,
    _quad_of,
    _unit_normal,
)
from cfdbender.mesh._refine._remesh_pairs import (
    element_report as element_report,
)

#: The passes of split, collapse, flip and smoothing one remesh runs.
PASSES = 8
#: An edge longer than this multiple of its target length is split.
SPLIT_RATIO = 4.0 / 3.0
#: An edge shorter than this multiple of its target length is collapsed.
COLLAPSE_RATIO = 4.0 / 5.0
#: A collapse is refused when a moved face's normal keeps less than this cosine.
COLLAPSE_COSINE = 0.2
#: A flip is refused when a new face's normal keeps less than this cosine.
FLIP_COSINE = 0.3
#: Quad-dominant: the passes of smoothing after the pairing, an engineering choice: on the
#: development fixtures more passes improved the skewness little (an unrecorded measurement).
SMOOTH_PASSES = 3
#: Quad-dominant: a smoothing move is undone when a face's normal keeps less than this cosine.
SMOOTH_COSINE = 0.9
#: A grid node this close to an interface node (a fraction of the interface's size) keeps it.
CONFORM_FRACTION = 1e-6
#: The passes of the repair of degenerate faces at most; a pass repairs every face it can,
#: and the bound keeps a face no flip or collapse improves from being tried forever.
REPAIR_PASSES = 4


def _keeps_direction(n0: Points, n1: Points, cosine: float) -> bool:
    """Return whether n1 turns away from n0 by less than the given cosine allows."""
    return bool(numpy.dot(n0, n1) > cosine * numpy.linalg.norm(n0) * numpy.linalg.norm(n1))


class Remesher:
    """Split, collapse, flip and smooth a patch towards its target edge lengths.

    Parameters
    ----------
    patch : Patch
        The triangles, their curves, fixed nodes and target lengths.
    """

    def __init__(self, patch: Patch) -> None:
        import trimesh

        self.V = list(numpy.asarray(patch.verts, dtype=float))
        self.F: dict[int, list[int]] = {i: list(f) for i, f in enumerate(patch.faces)}
        self.L: dict[int, str] = dict(enumerate(patch.labels))
        self.next_face = len(patch.faces)
        self.E: dict[Edge, set[int]] = defaultdict(set)
        self.VF: dict[int, set[int]] = defaultdict(set)
        for i, f in self.F.items():
            self._index_face(i, f)
        self.fixed = set(patch.fixed)
        self.frozen = set(patch.frozen)
        self.CE = dict(patch.curve)
        self.vcurve: dict[int, set[str]] = {}
        ends: dict[int, list[int]] = defaultdict(list)
        for (a, b), key in self.CE.items():
            self.vcurve.setdefault(a, set()).add(key)
            self.vcurve.setdefault(b, set()).add(key)
            ends[a].append(b)
            ends[b].append(a)
        self.corner = {v for v, ks in self.vcurve.items() if len(ks) > 1 or self._turns(v, ends[v])}
        self.h = [float(x) for x in patch.target]
        self.h_source = numpy.asarray(patch.target, dtype=float)
        self.interfaces: list[set[int]] = []
        self.surface = trimesh.Trimesh(
            numpy.asarray(patch.verts), numpy.asarray(patch.faces), process=False
        )
        self.curves = self._source_curves(patch)

    def _turns(self, v: int, ends: list[int]) -> bool:
        """Return whether a curve node is a corner: the end of a curve, a branch, or a sharp turn.

        A node where its curve turns by more than the ridge angle keeps its
        place, as a node on two curves does, so a corner of a boundary is
        not cut by smoothing along it.
        """
        if len(ends) != 2:
            return True
        a, p, b = self._points([ends[0], v, ends[1]])
        before, after = p - a, b - p
        return not _keeps_direction(before, after, math.cos(math.radians(RIDGE_DEGREES)))

    def _source_curves(self, patch: Patch) -> dict[str, tuple[Points, Points]]:
        """Return the source segments of each curve, the targets of its nodes' projection."""
        segments: dict[str, list[Edge]] = defaultdict(list)
        for edge, key in patch.curve.items():
            segments[key].append(edge)
        verts = numpy.asarray(patch.verts, dtype=float)
        return {
            key: (verts[[a for a, _ in es]], verts[[b for _, b in es]])
            for key, es in segments.items()
        }

    def _points(self, ids: Sequence[int]) -> Points:
        """Return the coordinates of some nodes, shape (n, 3)."""
        return numpy.array([self.V[v] for v in ids]).reshape(-1, 3)

    # ---- bookkeeping

    def _index_face(self, i: int, f: list[int]) -> None:
        for a, b in zip(f, f[1:] + f[:1], strict=True):
            self.E[_ek(a, b)].add(i)
        for v in f:
            self.VF[v].add(i)

    def _del_face(self, i: int) -> list[int]:
        f = self.F.pop(i)
        self.L.pop(i)
        for a, b in zip(f, f[1:] + f[:1], strict=True):
            k = _ek(a, b)
            self.E[k].discard(i)
            if not self.E[k]:
                del self.E[k]
        for v in f:
            self.VF[v].discard(i)
        return f

    def _new_face(self, f: list[int], label: str) -> None:
        i = self.next_face
        self.next_face += 1
        self.F[i] = f
        self.L[i] = label
        self._index_face(i, f)

    def neighbours(self, v: int) -> set[int]:
        """Return the vertices that share a face with v."""
        out: set[int] = set()
        for i in self.VF[v]:
            out.update(self.F[i])
        out.discard(v)
        return out

    def _length(self, k: Edge) -> float:
        return float(numpy.linalg.norm(self.V[k[0]] - self.V[k[1]]))

    def _target(self, k: Edge) -> float:
        return 0.5 * (self.h[k[0]] + self.h[k[1]])

    def _curve_neighbours(self, v: int) -> list[int]:
        return [u for u in sorted(self.neighbours(v)) if _ek(u, v) in self.CE]

    # ---- operations

    def split(self, a: int, b: int) -> int:
        """Split the edge a-b at its mid-point and return the new vertex."""
        k = _ek(a, b)
        m = len(self.V)
        self.V.append(0.5 * (self.V[a] + self.V[b]))
        self.h.append(0.5 * (self.h[a] + self.h[b]))
        key = self.CE.pop(k, None)
        for i in sorted(self.E[k]):
            f, label = self.F[i], self.L[i]
            self._del_face(i)
            j = f.index(a)
            if f[(j + 1) % 3] == b:
                c = f[(j + 2) % 3]
                self._new_face([a, m, c], label)
                self._new_face([m, b, c], label)
            else:
                c = f[(j + 1) % 3]
                self._new_face([a, c, m], label)
                self._new_face([m, c, b], label)
        if key is not None:
            self.CE[_ek(a, m)] = key
            self.CE[_ek(m, b)] = key
            self.vcurve[m] = {key}
        if k in self.frozen:
            self.frozen.discard(k)
            self.frozen |= {_ek(a, m), _ek(m, b)}
        return m

    def split_at(self, a: int, b: int, point: Points) -> int:
        """Split the edge a-b at the given point and return the new vertex."""
        m = self.split(a, b)
        self.V[m] = numpy.asarray(point, dtype=float).copy()
        return m

    def _link_ok(self, a: int, b: int) -> bool:
        """Return whether collapsing a-b keeps the surface a manifold (the link condition)."""
        k = _ek(a, b)
        opposite = {v for i in self.E[k] for v in self.F[i] if v not in (a, b)}
        return (self.neighbours(a) & self.neighbours(b)) == opposite

    def _no_long_edge(self, a: int, b: int) -> bool:
        """Return whether collapsing a onto b leaves no edge above the split length."""
        hb, pb = self.h[b], self.V[b]
        return all(
            numpy.linalg.norm(self.V[v] - pb) <= SPLIT_RATIO * 0.5 * (self.h[v] + hb)
            for v in self.neighbours(a)
            if v != b
        )

    def _no_fold(self, a: int, b: int) -> bool:
        """Return whether collapsing a onto b turns no face over."""
        for i in self.VF[a]:
            f = self.F[i]
            if b in f:
                continue
            before = numpy.array([self.V[x] for x in f])
            after = numpy.array([self.V[b] if x == a else self.V[x] for x in f])
            if not _keeps_direction(_unit_normal(before), _unit_normal(after), COLLAPSE_COSINE):
                return False
        return True

    def can_collapse(self, a: int, b: int) -> bool:
        """Return whether a may collapse onto b."""
        if a in self.fixed or a in self.corner:
            return False
        on_curve = _ek(a, b) in self.CE
        if (a in self.vcurve) != on_curve:
            return False
        return self._link_ok(a, b) and self._no_long_edge(a, b) and self._no_fold(a, b)

    def collapse(self, a: int, b: int) -> None:
        """Collapse vertex a onto vertex b."""
        k = _ek(a, b)
        around = sorted(self.neighbours(a))
        for i in sorted(self.E[k]):
            self._del_face(i)
        self.CE.pop(k, None)
        for i in sorted(self.VF[a]):
            f, label = self.F[i], self.L[i]
            self._del_face(i)
            self._new_face([b if x == a else x for x in f], label)
        for o in around:
            key = self.CE.pop(_ek(a, o), None)
            if key is not None and o != b:
                self.CE[_ek(o, b)] = key
        self.vcurve.pop(a, None)
        if self.frozen:
            renamed = {_ek(b if x == a else x, b if y == a else y) for x, y in self.frozen}
            self.frozen = {e for e in renamed if e[0] != e[1]}

    def _valence_gain(self, a: int, b: int, c: int, d: int) -> bool:
        """Return whether flipping a-b into c-d moves the valences towards their ideal."""
        valence = {v: len(self.neighbours(v)) for v in (a, b, c, d)}

        def deviation(v: int, n: int) -> int:
            return abs(n - (4 if v in self.vcurve else 6))

        before = sum(deviation(v, valence[v]) for v in (a, b, c, d))
        after = (
            deviation(a, valence[a] - 1)
            + deviation(b, valence[b] - 1)
            + deviation(c, valence[c] + 1)
            + deviation(d, valence[d] + 1)
        )
        return after < before

    def try_flip(self, a: int, b: int) -> bool:
        """Flip the edge a-b when it is not a curve and the valences improve."""
        k = _ek(a, b)
        fs = sorted(self.E.get(k, ()))
        if len(fs) != 2 or k in self.CE or self.L[fs[0]] != self.L[fs[1]]:
            return False
        f0, f1 = self.F[fs[0]], self.F[fs[1]]
        c = next(v for v in f0 if v not in (a, b))
        d = next(v for v in f1 if v not in (a, b))
        if _ek(c, d) in self.E or not self._valence_gain(a, b, c, d):
            return False
        j = f0.index(a)
        n0, n1 = ([a, d, c], [b, c, d]) if f0[(j + 1) % 3] == b else ([a, c, d], [b, d, c])
        old = _unit_normal(numpy.array([self.V[x] for x in f0]))
        for g in (n0, n1):
            if not _keeps_direction(
                old, _unit_normal(numpy.array([self.V[x] for x in g])), FLIP_COSINE
            ):
                return False
        label = self.L[fs[0]]
        self._del_face(fs[0])
        self._del_face(fs[1])
        self._new_face(n0, label)
        self._new_face(n1, label)
        return True

    # ---- smoothing and projection

    def _tangent_move(self, v: int) -> Points:
        """Return v moved halfway to its neighbors' centroid within its tangent plane."""
        centroid = numpy.mean([self.V[u] for u in sorted(self.neighbours(v))], axis=0)
        normal = numpy.zeros(3)
        for i in sorted(self.VF[v]):
            normal += _unit_normal(numpy.array([self.V[x] for x in self.F[i]]))
        normal /= max(float(numpy.linalg.norm(normal)), 1e-300)
        d = centroid - self.V[v]
        return self.V[v] + 0.5 * (d - numpy.dot(d, normal) * normal)

    def _curve_move(self, v: int) -> Points | None:
        """Return v moved towards its two curve neighbors, or None at a curve's end."""
        ends = self._curve_neighbours(v)
        if len(ends) != 2:
            return None
        return 0.5 * self.V[v] + 0.25 * (self.V[ends[0]] + self.V[ends[1]])

    def _onto_curve(self, v: int, p: Points) -> Points:
        """Return the point of v's source curve nearest p, found in the real space."""
        key = min(self.vcurve[v])
        if key not in self.curves:
            return self.V[v]
        a, b = self.curves[key]
        ab = b - a
        t = numpy.clip(
            ((p - a) * ab).sum(axis=1) / numpy.maximum((ab * ab).sum(axis=1), 1e-300), 0, 1
        )
        on = a + t[:, None] * ab
        return on[int(numpy.argmin(numpy.linalg.norm(on - p, axis=1)))]

    def _onto_surface(self, moves: dict[int, Points]) -> None:
        """Place the free moved vertices on the source surface and read their target there."""
        from trimesh.proximity import closest_point
        from trimesh.triangles import points_to_barycentric

        free = [v for v in moves if v not in self.vcurve]
        if not free:
            return
        q, _, tid = closest_point(self.surface, numpy.array([moves[v] for v in free]))
        bary = points_to_barycentric(self.surface.triangles[tid], q)
        hq = (bary * self.h_source[self.surface.faces[tid]]).sum(axis=1)
        for v, p, h in zip(free, q, hq, strict=True):
            self.V[v] = numpy.asarray(p, dtype=float)
            self.h[v] = float(h)

    def smooth_and_project(self) -> None:
        """Smooth every movable vertex and project it back onto the source surface or curve."""
        moves: dict[int, Points] = {}
        for v in sorted(v for v, fs in self.VF.items() if fs):
            if v in self.fixed or v in self.corner:
                continue
            p = self._curve_move(v) if v in self.vcurve else self._tangent_move(v)
            if p is not None:
                moves[v] = p
        before = {v: (self.V[v].copy(), self.h[v]) for v in moves}
        self._onto_surface(moves)
        for v in [v for v in moves if v in self.vcurve]:
            self.V[v] = self._onto_curve(v, moves[v])
        self._undo_folding_moves(before)

    def _undo_folding_moves(self, before: dict[int, tuple[Points, float]]) -> None:
        """Put back the moved vertices of every face a move flattened or turned over (F-307).

        A face whose normal keeps less than :data:`COLLAPSE_COSINE` of its direction
        returns all its moved vertices to where they were, until no face is left.
        """
        moved = set(before)
        for _ in range(len(before) + 1):
            bad: set[int] = set()
            for v in moved:
                for i in self.VF[v]:
                    face = self.F[i]
                    old = numpy.array([before[x][0] if x in before else self.V[x] for x in face])
                    new = numpy.array([self.V[x] for x in face])
                    if not _keeps_direction(_unit_normal(old), _unit_normal(new), COLLAPSE_COSINE):
                        bad.update(x for x in face if x in moved)
            if not bad:
                return
            for x in bad:
                self.V[x] = before[x][0]
                self.h[x] = before[x][1]
            moved -= bad

    # ---- the passes

    def _held(self, k: Edge) -> bool:
        """Return whether an edge may not be split (frozen, or fixed at both ends on a curve)."""
        if k in self.frozen:
            return True
        both_fixed = k[0] in self.fixed and k[1] in self.fixed
        return both_fixed and (k in self.CE or len(self.E[k]) == 1)

    def _split_long(self) -> None:
        for k in [k for k in list(self.E) if self._length(k) > SPLIT_RATIO * self._target(k)]:
            if k in self.E and not self._held(k):
                self.split(*k)

    def _collapse_short(self) -> None:
        for k in sorted(self.E, key=self._length):
            if k not in self.E or self._length(k) >= COLLAPSE_RATIO * self._target(k):
                continue
            a, b = k
            if self.can_collapse(a, b):
                self.collapse(a, b)
            elif self.can_collapse(b, a):
                self.collapse(b, a)

    def run(self, passes: int = PASSES) -> None:
        """Run the passes of split, collapse, flip and smoothing."""
        for _ in range(passes):
            self._split_long()
            self._collapse_short()
            for k in list(self.E):
                if k in self.E:
                    self.try_flip(*k)
            self.smooth_and_project()

    # ---- repairing degenerate faces

    def degenerate_faces(self) -> list[int]:
        """Return the faces of near-zero area or of a sliver's angle, in face order.

        A face is degenerate by :func:`._geometry.degenerate`, the audit's rule:
        its area below ``DEGENERATE_AREA_FRACTION`` of its family's median face
        area, or its smallest angle below ``DEGENERATE_ANGLE_DEGREES``.
        """
        ids = sorted(self.F)
        if not ids:
            return []
        p = [self._points(self.F[i]) for i in ids]
        area = numpy.array([0.5 * float(numpy.linalg.norm(_face_normal(q))) for q in p])
        names = sorted(set(self.L.values()))
        family = numpy.array([names.index(self.L[i]) for i in ids], dtype=numpy.int64)
        angle = numpy.array([min_angle_degrees(q) for q in p])
        return [i for i, bad in zip(ids, degenerate(area, angle, family), strict=True) if bad]

    def _held_node(self, v: int) -> bool:
        """Return whether a repair may not move or remove v (a curve, fixed or corner node)."""
        return v in self.fixed or v in self.corner or v in self.vcurve

    def _repair_flip(self, k: Edge) -> bool:
        """Flip k when it joins two faces of one family and the worse face's angle grows."""
        fs = sorted(self.E.get(k, ()))
        if len(fs) != 2 or k in self.CE or k in self.frozen or self.L[fs[0]] != self.L[fs[1]]:
            return False
        a, b = k
        f0, f1 = self.F[fs[0]], self.F[fs[1]]
        if len(f0) != 3 or len(f1) != 3:
            return False
        c = next(v for v in f0 if v not in (a, b))
        d = next(v for v in f1 if v not in (a, b))
        if _ek(c, d) in self.E:
            return False
        j = f0.index(a)
        n0, n1 = ([a, d, c], [b, c, d]) if f0[(j + 1) % 3] == b else ([a, c, d], [b, d, c])
        before = [self._points(f) for f in (f0, f1)]
        after = [self._points(f) for f in (n0, n1)]
        normal = _unit_normal(before[0]) + _unit_normal(before[1])
        if not all(_keeps_direction(normal, _unit_normal(q), FLIP_COSINE) for q in after):
            return False
        if min(map(min_angle_degrees, after)) <= min(map(min_angle_degrees, before)):
            return False
        label = self.L[fs[0]]
        self._del_face(fs[0])
        self._del_face(fs[1])
        self._new_face(n0, label)
        self._new_face(n1, label)
        return True

    def _repair_collapse(self, k: Edge) -> bool:
        """Collapse a free end of k onto the other when the surface stays a manifold, unfolded."""
        for a, b in (k, k[::-1]):
            if not self._held_node(a) and self._link_ok(a, b) and self._no_fold(a, b):
                self.collapse(a, b)
                return True
        return False

    def repair_degenerate(self) -> list[int]:
        """Repair the degenerate faces a remesh left; return the faces left degenerate.

        Each degenerate triangle has its longest edge flipped with its
        neighbor of the same family when that makes the worse of the two
        faces' least angle larger, else its shortest edge collapsed onto one
        end. A node on a curve (an open boundary, the trailing edge or a
        listed perimeter, a ridge, an interface), a node shared with a family
        outside the group and a corner never move nor go: a face held by them
        is left, and the audit reports it. A mesh with no degenerate face is
        not touched. At most :data:`REPAIR_PASSES` passes run.
        """
        for _ in range(REPAIR_PASSES):
            bad = self.degenerate_faces()
            repaired = False
            for i in bad:
                f = self.F.get(i)
                if f is None or len(f) != 3:
                    continue
                edges = sorted(
                    (_ek(a, b) for a, b in zip(f, f[1:] + f[:1], strict=True)),
                    key=lambda e: (self._length(e), e),
                )
                if self._repair_flip(edges[-1]) or self._repair_collapse(edges[0]):
                    repaired = True
            if not repaired:
                return bad
        return self.degenerate_faces()

    # ---- conforming an interface to a grid's new nodes

    def _place(self, p: Points, chain: set[Edge], keep: set[int], tol: float) -> int:
        """Keep the chain node at p, or insert one there; return 1 when a node was inserted."""
        on = sorted({v for k in chain for v in k})
        d = numpy.linalg.norm(numpy.array([self.V[v] for v in on]) - p, axis=1)
        j = int(numpy.argmin(d))
        if d[j] < tol and on[j] not in keep:
            keep.add(on[j])
            self.V[on[j]] = p.copy()
            return 0
        k = min(sorted(chain), key=lambda e: _segment_distance(self.V[e[0]], self.V[e[1]], p))
        m = self.split_at(k[0], k[1], p)
        chain.discard(k)
        for e in (_ek(k[0], m), _ek(m, k[1])):
            chain.add(e)
            self.frozen.add(e)
            self.CE.setdefault(e, FROZEN)
        keep.add(m)
        return 1

    def _drop(self, o: int, chain: set[Edge], keep: set[int]) -> bool:
        """Collapse a chain node no new node keeps onto a chain neighbor; return success."""
        ends = [x for k in sorted(chain) if o in k for x in k if x != o]
        order = sorted(
            ends, key=lambda x: (x not in keep, float(numpy.linalg.norm(self.V[x] - self.V[o])), x)
        )
        for b in order:
            # F-308: the interface removal takes the fold guard every other collapse takes.
            if not self._link_ok(o, b) or not self._no_fold(o, b):
                continue
            self.collapse(o, b)
            renamed = {_ek(b if x == o else x, b if y == o else y) for x, y in chain}
            chain.clear()
            chain.update(e for e in renamed if e[0] != e[1])
            self.fixed.discard(o)
            self.corner.discard(o)
            return True
        return False

    def conform(self, old: Iterable[int], new_points: Points) -> dict[str, int]:
        """Rebuild a frozen interface on the new nodes of a grid refined before it.

        Each new point is inserted on the interface at its own position (an
        old node it coincides with is kept and takes the point exactly), each
        old node no new point keeps is collapsed along the interface, and the
        result is fixed node for node, so the two families weld again.

        Parameters
        ----------
        old : iterable of int
            The local vertices of the interface in the source.
        new_points : ndarray
            The grid's new nodes on that interface, in the real space.

        Returns
        -------
        dict
            The counts ``inserted``, ``kept``, ``removed`` and ``not_removed``.
        """
        old_set = set(old)
        chain = {
            k for k, fs in self.E.items() if k[0] in old_set and k[1] in old_set and len(fs) == 1
        }
        if not chain:
            self.interfaces.append(set())
            return {"inserted": 0, "kept": 0, "removed": 0, "not_removed": 0}
        span = numpy.array([self.V[v] for v in sorted(old_set)])
        tol = CONFORM_FRACTION * max(float(numpy.linalg.norm(numpy.ptp(span, axis=0))), 1e-300)
        keep: set[int] = set()
        inserted = 0
        for q in numpy.asarray(new_points, dtype=float).reshape(-1, 3):
            inserted += self._place(q, chain, keep, tol)
        dropped = [self._drop(o, chain, keep) for o in sorted({v for k in chain for v in k} - keep)]
        self.fixed |= keep
        self.interfaces.append(keep)
        return {
            "inserted": inserted,
            "kept": len(keep) - inserted,
            "removed": sum(dropped),
            "not_removed": len(dropped) - sum(dropped),
        }

    def interface_chain(self, index: int) -> set[Edge]:
        """Return the edges of a conformed interface, between the new nodes it holds.

        Parameters
        ----------
        index : int
            The interface, in the order :meth:`conform` was called.

        Returns
        -------
        set of tuple
            The patch's boundary edges whose two nodes are the interface's.
        """
        keep = self.interfaces[index]
        return {k for k, fs in self.E.items() if len(fs) == 1 and k[0] in keep and k[1] in keep}

    def retarget(self, targets: Mapping[int, float]) -> None:
        """Set the target length at nodes, and the source's field of targets at its own nodes.

        Parameters
        ----------
        targets : mapping of int to float
            The new target length per node.
        """
        for v, h in targets.items():
            self.h[v] = float(h)
            if v < len(self.h_source):
                self.h_source[v] = float(h)

    # ---- quad-dominant: pairing triangles into quadrilaterals

    def _all_points(self) -> Points:
        """Return the coordinates of every node, shape (n, 3)."""
        return self._points(range(len(self.V)))

    def _mergeable(self, k: Edge, labels: set[str]) -> tuple[int, int] | None:
        """Return the two triangles an edge may be merged across, or None.

        It may when it is no curve (an open boundary, the trailing edge, a
        ridge, an interface, a frozen edge) and its two faces are triangles
        of one quad-dominant family.
        """
        fs = self.E.get(k, set())
        if len(fs) != 2 or k in self.CE or k in self.frozen:
            return None
        i, j = sorted(fs)
        same = self.L[i] == self.L[j] and self.L[i] in labels
        return (i, j) if same and len(self.F[i]) == len(self.F[j]) == 3 else None

    def _merge_candidates(self, labels: set[str]) -> tuple[list[tuple[int, int]], Faces, Points]:
        """Return every pair of triangles that merges into a valid quadrilateral, and its quality.

        A quadrilateral is valid when it is convex, every interior angle lies
        in ``QUAD_ANGLE_BAND`` and its warp is at most ``QUAD_WARP_DEGREES``,
        all measured in the real space. Its quality is one minus its
        equiangle skewness, 1 for a rectangle.
        """
        pairs: list[tuple[int, int]] = []
        quads: Faces = []
        for k in sorted(self.E):
            pair = self._mergeable(k, labels)
            quad = None if pair is None else _quad_of(self.F[pair[0]], self.F[pair[1]], k)
            if pair is not None and quad is not None:
                pairs.append(pair)
                quads.append(quad)
        if not quads:
            return [], [], numpy.zeros(0)
        angle, warp, convex = quad_shapes(self._all_points()[numpy.array(quads)])
        low, high = QUAD_ANGLE_BAND
        valid = convex & (angle.min(axis=1) >= low) & (angle.max(axis=1) <= high)
        valid &= warp <= QUAD_WARP_DEGREES
        quality = 1.0 - numpy.abs(angle - 90.0).max(axis=1) / 90.0
        keep = numpy.nonzero(valid)[0].tolist()
        return [pairs[c] for c in keep], [quads[c] for c in keep], quality[keep]

    def pair_quads(self, labels: Iterable[str]) -> int:
        """Merge pairs of triangles of the given families into quadrilaterals; return how many.

        The pairs are chosen best quality first (a greedy matching), then
        the matching is grown along alternating paths of up to
        :data:`AUGMENT_DEPTH` matched pairs, each of which merges one more
        pair, so the share of quadrilaterals is high. A quadrilateral runs
        through its corners the way its two triangles did, so it keeps their
        orientation. Curves are never merged across; nodes do not move.
        """
        pairs, quads, quality = self._merge_candidates(set(labels))
        mate = _matching(pairs, quality)
        merged = 0
        for (i, j), quad in zip(pairs, quads, strict=True):
            if mate.get(i) == j and i in self.F and j in self.F:
                label = self.L[i]
                self._del_face(i)
                self._del_face(j)
                self._new_face(quad, label)
                merged += 1
        return merged

    def _edge_neighbors(self, v: int) -> list[int]:
        """Return the nodes joined to v by a face edge (not across a quadrilateral)."""
        out: set[int] = set()
        for i in self.VF[v]:
            f = self.F[i]
            j = f.index(v)
            out.update((f[j - 1], f[(j + 1) % len(f)]))
        return sorted(out)

    def _face_valid(self, i: int, real: Points, normal: Points) -> bool:
        """Return whether a face keeps its normal's direction and, a quadrilateral, its shape."""
        p = real[self.F[i]]
        if not _keeps_direction(normal, _face_normal(p), SMOOTH_COSINE):
            return False
        if len(p) != 4:
            return True
        angle, warp, convex = quad_shapes(p[None])
        low, high = QUAD_ANGLE_BAND
        inside = low <= float(angle.min()) and float(angle.max()) <= high
        return bool(convex[0]) and inside and float(warp[0]) <= QUAD_WARP_DEGREES

    def smooth_quads(self, labels: Iterable[str]) -> int:
        """Smooth the free nodes of quad-dominant families once; return how many moved.

        A node moves halfway to the centroid of its edge neighbors within its
        tangent plane and is projected onto the source surface, as the
        remesh's own smoothing does. Nodes on a curve, fixed, frozen or at a
        corner never move, nor any node of a family not in ``labels``. A move
        that turns a face by more than :data:`SMOOTH_COSINE` allows or leaves
        a quadrilateral outside the merge bounds is undone, with the moves of
        the face's other nodes, until every face is valid again.
        """
        families = set(labels)
        free = [
            v
            for v in sorted(v for v, fs in self.VF.items() if fs)
            if v not in self.fixed
            and v not in self.vcurve
            and v not in self.corner
            and all(self.L[i] in families for i in self.VF[v])
        ]
        if not free:
            return 0
        before = {v: (self.V[v].copy(), self.h[v]) for v in free}
        real = self._all_points()
        touched = sorted({i for v in free for i in self.VF[v]})
        normals = {i: _face_normal(real[self.F[i]]) for i in touched}
        self._onto_surface({v: self._tangent_quad_move(v) for v in free})
        moved = set(free)
        while True:
            real = self._all_points()
            bad = [i for i in touched if not self._face_valid(i, real, normals[i])]
            back = {v for i in bad for v in self.F[i] if v in moved}
            if not back:
                return len(moved)
            for v in sorted(back):
                self.V[v], self.h[v] = before[v]
            moved -= back

    def _tangent_quad_move(self, v: int) -> Points:
        """Return v moved halfway to its edge neighbors' centroid within its tangent plane."""
        centroid = numpy.mean([self.V[u] for u in self._edge_neighbors(v)], axis=0)
        normal = numpy.zeros(3)
        for i in sorted(self.VF[v]):
            normal += _face_normal(numpy.array([self.V[x] for x in self.F[i]]))
        normal /= max(float(numpy.linalg.norm(normal)), 1e-300)
        d = centroid - self.V[v]
        return self.V[v] + 0.5 * (d - numpy.dot(d, normal) * normal)

    # ---- the result

    def edge_ratio_median(self) -> float:
        """Return the median ratio of a non-curve edge's length to its target."""
        ratios = [self._length(k) / self._target(k) for k in self.E if k not in self.CE]
        return float(numpy.median(ratios)) if ratios else float("nan")

    def result(self) -> tuple[Points, dict[str, Faces], Points]:
        """Return the points, the faces per family and the trailing-edge mid-points."""
        used = sorted({v for f in self.F.values() for v in f})
        renumber = {v: i for i, v in enumerate(used)}
        points = self._points(used)
        families: dict[str, Faces] = defaultdict(list)
        for i, f in self.F.items():
            families[self.L[i]].append([renumber[v] for v in f])
        te = [k for k, key in self.CE.items() if key == TRAILING_EDGE]
        mid = 0.5 * (self._points([a for a, _ in te]) + self._points([b for _, b in te]))
        return points, dict(families), mid


def _segment_distance(a: Points, b: Points, p: Points) -> float:
    """Return the distance from p to the segment a-b."""
    ab = b - a
    t = numpy.clip(numpy.dot(p - a, ab) / max(float(numpy.dot(ab, ab)), 1e-300), 0, 1)
    return float(numpy.linalg.norm(a + t * ab - p))


class Remeshed(NamedTuple):
    """The remesh of a group of families.

    Attributes
    ----------
    points : ndarray
        The group's new nodes.
    families : dict
        Each family's new triangles in the source's face order.
    trailing_edge : ndarray
        The mid-point of every trailing-edge edge of the new mesh.
    info : dict
        What ``refine.json`` reports of the remesh (JSON values only).
    """

    points: Points
    families: dict[str, Faces]
    trailing_edge: Points
    info: dict[str, Any]


def _interface_key(old: Iterable[int], new: Points) -> tuple[frozenset[int], bytes]:
    """Return what identifies an interface entry: its source vertices and its new nodes."""
    return frozenset(int(v) for v in old), numpy.asarray(new, dtype=float).reshape(-1, 3).tobytes()


def _outside_edges(mesh: ObjMesh, others: Sequence[str], local: Mapping[int, int]) -> set[Edge]:
    """Return the edges of the faces of families outside the group, in its local numbering."""
    return {
        _ek(local[a], local[b])
        for n in others
        for f in mesh.families[n]
        for a, b in zip(f, f[1:] + f[:1], strict=True)
        if a in local and b in local
    }


def _take_shared(
    remesher: Remesher,
    factors: Mapping[str, float],
    local: Mapping[int, int],
    conform: Sequence[tuple[Iterable[int], Points]],
    shared: Sequence[tuple[Iterable[int], Points]],
    *,
    outside: set[Edge],
) -> dict[str, dict[str, float]]:
    """Set the targets of at the curves a family shares; return the record of.

    A family at factor 1 takes them on each conformed interface ``shared``
    names, the new shared edges of a refined side; a family at another
    factor takes them on each frozen interface no ``conform`` entry rebuilt,
    the shared edges an unchanged neighbor keeps: an edge of a face of
    the neighbor (``outside``, local numbering), so a diagonal the group's
    triangulation draws between two shared nodes is not one. A node of a
    face of the other kind keeps its own target.
    """
    keys = {_interface_key(old, new) for old, new in shared}
    chosen = [i for i, (old, new) in enumerate(conform) if _interface_key(old, new) in keys]
    rebuilt = {local[v] for old, _ in conform for v in old if v in local}
    rebuilt.update(*remesher.interfaces)
    faces = list(remesher.F.values())
    labels = [remesher.L[i] for i in remesher.F]
    other = {v for f, n in zip(faces, labels, strict=True) if factors[n] != 1.0 for v in f}
    unit = {v for f, n in zip(faces, labels, strict=True) if factors[n] == 1.0 for v in f}
    kept = {
        e
        for e in remesher.frozen & outside
        if e in remesher.E and not set(e) <= rebuilt and not set(e) <= unit
    }
    sources = {local[v] for i in chosen for v in conform[i][0] if v in local}
    rules = [
        (other, [remesher.interface_chain(i) for i in chosen], sources),
        (unit, [kept] if kept else [], set()),
    ]
    record: dict[str, dict[str, float]] = {}
    for held, chains, sources in rules:
        targets, found = shared_targets(
            numpy.asarray(remesher.V, dtype=float),
            faces,
            labels,
            own=numpy.asarray(remesher.h, dtype=float),
            held=held,
            chains=chains,
            sources=sources,
        )
        remesher.retarget(targets)
        record.update(found)
    return record


def refine_group(
    mesh: ObjMesh,
    names: Sequence[str],
    te_vertices: Iterable[int],
    factor: float | Mapping[str, float],
    *,
    conform: Sequence[tuple[Iterable[int], Points]] = (),
    elements: str | Mapping[str, str] = TRIANGLES,
    shared: Sequence[tuple[Iterable[int], Points]] = (),
    te_edges: set[tuple[int, int]] | None = None,
) -> Remeshed:
    """Remesh a group of families together.

    An interface between two families of the group is a curve remeshed once
    and shared by both; an interface with any other family of the mesh is
    frozen node for node, unless ``conform`` rebuilds it first on the new
    nodes of a grid family refined before it.

    Parameters
    ----------
    mesh : ObjMesh
        The source mesh; families outside ``names`` are read, never changed.
    names : sequence of str
        The families remeshed together, in the source's family order.
    te_vertices : iterable of int
        The source vertices of the trailing edge (mesh numbering).
    factor : float or mapping of str to float
        One factor for every family, or a factor per family.
    conform : sequence of (iterable of int, ndarray), optional
        For each interface with a grid family refined first, its source
        vertices (mesh numbering) and the grid's new nodes on it.
    elements : str or mapping of str to str, optional
        One element mode for every family, or a mode per family:
        ``"triangles"`` (the default; the remesh is returned as it ends) or
        ``"quad-dominant"`` (after the remesh, the family's triangles are
        paired into quadrilaterals by :meth:`Remesher.pair_quads` and its free
        nodes smoothed :data:`SMOOTH_PASSES` times by
        :meth:`Remesher.smooth_quads`).
    shared : sequence of (iterable of int, ndarray), optional
        The entries of ``conform`` whose grid is a refined side: once each is
        conformed, a family of the group at factor 1 takes, along and near
        it, the length of the new shared edges (:func:`shared_targets`), and
        ``info["shared_edge_target_mesh_units"]`` records it per family in the
        mesh's coordinate units; the level writer converts declared units to meters.
    te_edges : set of tuple of int, optional
        Exact source trailing-edge edges, in mesh numbering. When supplied,
        only these edges and their splits are returned as shedding a wake (the
        solver's reading of that is not measured by a dated report; this
        function returns geometry only); vertex membership
        alone is retained for callers that do not supply an edge list.

    Returns
    -------
    Remeshed
        The points, the faces per family, the trailing-edge mid-points and the info.

    """
    factors = dict(factor) if isinstance(factor, Mapping) else {n: float(factor) for n in names}
    group = _group(mesh, names)
    local = {v: i for i, v in enumerate(group.ids)}
    others = [n for n in mesh.families if n not in names]
    fixed = {local[v] for n in others for v in mesh.family_vertices(n).tolist() if v in local}
    te = {local[v] for v in te_vertices if v in local}
    listed = (
        {e for e in edge_faces(group.faces) if e[0] in te and e[1] in te}
        if te_edges is None
        else {
            (min(local[a], local[b]), max(local[a], local[b]))
            for a, b in te_edges
            if a in local and b in local
        }
    )
    curve, frozen = _curves(group, fixed, listed)
    patch = Patch(group.verts, group.faces, group.labels, _targets(group, factors), curve)
    patch.fixed, patch.frozen = fixed, frozen
    remesher = Remesher(patch)
    conformed = [
        remesher.conform([local[v] for v in old if v in local], new) for old, new in conform
    ]
    outside = _outside_edges(mesh, others, local)
    at_curve = _take_shared(remesher, factors, local, conform, shared, outside=outside)
    remesher.run()
    remesher.repair_degenerate()
    modes = dict(elements) if isinstance(elements, Mapping) else dict.fromkeys(names, elements)
    quad_families = [n for n in names if modes.get(n, TRIANGLES) == QUAD_DOMINANT]
    if quad_families:
        remesher.pair_quads(quad_families)
        for _ in range(SMOOTH_PASSES):
            remesher.smooth_quads(quad_families)
    points, families, te_mid = remesher.result()
    ratio = remesher.edge_ratio_median()
    kinds: dict[str, int] = defaultdict(int)
    for key in curve.values():
        kinds[key.partition(":")[0]] += 1
    info: dict[str, Any] = {
        "method": "remesh",
        "remeshed_together": list(names),
        "fixed_interface_nodes": len(fixed),
        "curve_edges": dict(sorted(kinds.items())),
        # JSON null when every edge is a curve: NaN is no JSON a strict reader accepts.
        "edge_ratio_median": None if math.isnan(ratio) else round(ratio, 6),
    }
    if conformed:
        info["conformed_interfaces"] = conformed
    if at_curve:
        info["shared_edge_target_mesh_units"] = at_curve
    return Remeshed(points, _ordered(points, families, group), te_mid, info)
