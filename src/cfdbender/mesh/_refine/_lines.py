"""The lines a grid is oriented by: the [[perimeter]] node lists and a family's trailing edge.

A ``[[perimeter]]`` list gives, per family, the mesh edges joining two of its
nodes (:func:`perimeter_lines`); a family on several lines keeps the first that
recovers its grid, the exported lists first (:func:`choose_perimeter_line`). A
grid family with no line takes the longest side of its boundary as reference
(:func:`assume_trailing_edge`), and an incomplete line on a sheet is completed
from the grid (:func:`grid_with_complete_te_edges`). Only the exported lists
are the level's trailing edge (:func:`on_exported`).

The functions read and write the refinement's working state, the ``_Work`` of
:mod:`._level`, passed in as ``work``.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

from cfdbender._errors import GeometryWarning, warn
from cfdbender.mesh._refine import _blocks, _geometry, _grid, _obj
from cfdbender.mesh._refine._config import RefineRequest
from cfdbender.mesh._refine._obj import Faces, ObjMesh

Points = NDArray[numpy.float64]
_Work = Any  # the working state of ._level, which imports this module


def perimeter_lines(
    source: Path, request: RefineRequest, mesh: ObjMesh
) -> (
    tuple[
        set[tuple[int, int]],
        set[tuple[int, int]],
        list[set[tuple[int, int]]],
        dict[str, set[tuple[int, int]]],
    ]
    | None
):
    """Return the ``[[perimeter]]`` source edges: all, exported, per list, per family's rearmost.

    An edge is a perimeter's when its two ends are nodes of that list. A list
    that is missing, or whose nodes join no edge, is warned and skipped, and
    so is a node that matches no vertex; None without ``[[perimeter]]``.

    Of each list, the rearmost edges of every family (:func:`._obj.rearmost_edges`)
    are a set of their own, ahead of the whole list, so a grid is oriented by its
    trailing edge first; the last element maps each family to its own rearmost
    edges over every list, the line its trailing edge is taken from before any
    other listed line on it. A list with ``te_families`` exports those families'
    rearmost edges (a name of ``[components]`` stands for its members); one with
    ``export_te_per_family`` and no ``te_families`` exports all its edges.
    """
    if not request.perimeters:
        return None
    every: set[tuple[int, int]] = set()
    exported: set[tuple[int, int]] = set()
    sets: list[set[tuple[int, int]]] = []
    rears: dict[str, set[tuple[int, int]]] = {}
    for perimeter in sorted(request.perimeters, key=lambda p: not p.exported):
        found = _listed_edges(perimeter, mesh)
        if found is None:
            continue
        lines = set().union(*found.values())
        rear = {
            name: _obj.rearmost_edges(
                mesh.verts,
                mesh.families[name],
                _obj.rearmost_perimeter(mesh.verts, mesh.families[name], edges, perimeter.aft),
                perimeter.aft,
            )
            for name, edges in found.items()
        }
        every |= lines
        rearmost = set().union(*rear.values())
        if rearmost:
            sets.append(rearmost)
        sets.append(lines)
        for name, edges in rear.items():
            rears[name] = rears.get(name, set()) | edges
        if perimeter.te_families:
            for name in te_members(perimeter, request, mesh):
                exported |= rear[name]
        elif perimeter.export_te_per_family:
            exported |= lines
    return every, exported, sets, rears


def _listed_edges(perimeter: Any, mesh: ObjMesh) -> dict[str, set[tuple[int, int]]] | None:
    """Return a list's edges per family; None, warned, for a list missing, empty or joining none."""
    if not perimeter.file.is_file():
        warn(
            f"{perimeter.file}: no such [[perimeter]] file; it is skipped",
            GeometryWarning,
            stacklevel=4,
        )
        return None
    nodes = _obj.read_node_list(perimeter.file)
    if not nodes.ids:
        warn(
            f"{perimeter.file}: no node is read (Id, Name, X, Y, Z); it is skipped",
            GeometryWarning,
            stacklevel=4,
        )
        return None
    found, unmatched = _obj.perimeter_edges(mesh.verts, mesh.families, nodes)
    if unmatched:
        shown = ", ".join(str(i) for i in unmatched[:5])
        warn(
            f"{perimeter.file}: {len(unmatched)} of {len(nodes.ids)} node(s) match no mesh "
            f"vertex and are skipped (ids {shown}{' and more' if len(unmatched) > 5 else ''}); "
            "the list and the OBJ must be saved from the same model, in the same unit",
            GeometryWarning,
            stacklevel=4,
        )
    if not any(found.values()):
        warn(
            f"{perimeter.file}: no mesh edge joins two listed nodes; it is skipped",
            GeometryWarning,
            stacklevel=4,
        )
        return None
    return found


def te_members(perimeter: Any, request: RefineRequest, mesh: ObjMesh) -> list[str]:
    """Return the mesh families a list's ``te_families`` names, warning a name of neither."""
    out: list[str] = []
    for name in perimeter.te_families:
        if name in request.components:
            out += [m for m in request.components[name] if m in mesh.families]
        elif name in mesh.families:
            out.append(name)
        else:
            warn(
                f"{perimeter.file}: te_families names {name!r}, neither a family of the mesh nor "
                "a component of [components]; it is skipped",
                GeometryWarning,
                stacklevel=4,
            )
    return list(dict.fromkeys(out))


def exported_by_family(mesh: ObjMesh, exported: set[tuple[int, int]]) -> dict[str, Points]:
    out = {}
    for name, faces in mesh.families.items():
        edges, _ = _obj.edge_midpoints(mesh.verts, faces)
        on = {(int(a), int(b)) for a, b in edges} & exported
        if on:
            out[name] = _obj.midpoints(mesh.verts, on)
    return out


def grid_with_complete_te_edges(
    work: _Work, name: str, found: _grid.Grid | _blocks.Blocks
) -> _grid.Grid | _blocks.Blocks | str:
    """Return the recovered grid, or why the listed trailing-edge edges are not grid lines."""
    if isinstance(found, _blocks.Blocks):
        chains = [found.arcs[i] for i in found.te_arcs]
    elif found.smooth:
        chains = []
    else:
        chains = [found.ids[:, 0].tolist()]
    represented = {
        (min(a, b), max(a, b)) for chain in chains for a, b in zip(chain, chain[1:], strict=False)
    }
    listed = work.te_edges[name]
    if represented and listed < represented and not isinstance(found, _blocks.Blocks):
        # Every listed edge lies on the grid's trailing-edge line, so the line is the
        # trailing edge. A line edge with an end no list holds is a node left out:
        # complete it and say where. One whose two ends are listed is a gap the lists
        # leave on purpose (a wake gap), and stays out.
        nodes = {v for edge in listed for v in edge}
        left_out = {e for e in represented - listed if not (e[0] in nodes and e[1] in nodes)}
        if left_out:
            warn(
                f"{work.source} family {name}: the perimeter list is incomplete; the grid "
                f"line completes it. " + te_mismatch(work, listed | left_out, listed),
                GeometryWarning,
                stacklevel=2,
            )
            work.te_edges[name] = listed | left_out
            work.te_of[name] = {v for edge in work.te_edges[name] for v in edge}
        return found
    if represented != listed:
        return (
            "the listed trailing-edge edges are not complete grid lines; use remesh. "
            + te_mismatch(work, represented, work.te_edges[name])
        )
    return found


def te_mismatch(
    work: _Work, line: set[tuple[int, int]], listed: set[tuple[int, int]], shown: int = 5
) -> str:
    """Name where the listed trailing edge and the grid's trailing-edge line differ."""
    verts = work.mesh.verts

    def where(edges: set[tuple[int, int]]) -> str:
        mids = sorted(
            tuple(round(float(c), 6) for c in (verts[a] + verts[b]) / 2.0) for a, b in edges
        )
        text = "; ".join(f"({x}, {y}, {z})" for x, y, z in mids[:shown])
        return text + (f"; and {len(mids) - shown} more" if len(mids) > shown else "")

    missing, extra = line - listed, listed - line
    if not line:
        return te_pieces(work, listed)
    parts = [f"the grid line has {len(line)} edges and the perimeter lists matched {len(listed)}"]
    if missing:
        parts.append(f"{len(missing)} line edges have no point, mid-points {where(missing)}")
    if extra:
        parts.append(f"{len(extra)} matched edges are off the line, mid-points {where(extra)}")
    return "; ".join(parts) + " (mesh units)"


def te_pieces(work: _Work, listed: set[tuple[int, int]]) -> str:
    """Name the connected pieces of the listed trailing edge and their end points."""
    verts = work.mesh.verts
    touching: dict[int, list[int]] = {}
    for a, b in listed:
        touching.setdefault(a, []).append(b)
        touching.setdefault(b, []).append(a)
    seen: set[int] = set()
    pieces = []
    for start in sorted(touching):
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            members.append(v)
            for n in touching[v]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        ends = [v for v in members if len(touching[v]) == 1]
        pieces.append((len(members) - 1, ends))

    def point(v: int) -> str:
        x, y, z = (round(float(c), 6) for c in verts[v])
        return f"({x}, {y}, {z})"

    text = "; ".join(
        f"piece {i}: {edges} edges, ends {' and '.join(point(v) for v in ends) or 'closed loop'}"
        for i, (edges, ends) in enumerate(sorted(pieces, key=lambda p: -p[0]), start=1)
    )
    count = f"{len(listed)} edges in {len(pieces)} pieces"
    return f"the perimeter lists matched {count}: {text} (mesh units)"


#: A boundary loop has a corner, and a path of edges is cut, where it turns by more
#: than this angle in degrees.
CORNER_DEGREES = 45.0


def turns(pts: Points, before: int, at: int, after: int) -> bool:
    """Return whether the path before -> at -> after turns by more than :data:`CORNER_DEGREES`."""
    u, w = pts[at] - pts[before], pts[after] - pts[at]
    cos = float(u @ w) / max(float(numpy.linalg.norm(u) * numpy.linalg.norm(w)), 1e-300)
    return cos < numpy.cos(numpy.radians(CORNER_DEGREES))


def _single_loop(faces: Faces) -> list[int]:
    """Return the family's boundary as one loop of vertices, empty when it is not one loop."""
    count: dict[tuple[int, int], int] = {}
    for f in faces:
        for a, b in zip(f, [*f[1:], f[0]], strict=True):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    touching: dict[int, list[int]] = {}
    for (a, b), n in count.items():
        if n == 1:
            touching.setdefault(a, []).append(b)
            touching.setdefault(b, []).append(a)
    if not touching or any(len(n) != 2 for n in touching.values()):
        return []
    start = min(touching)
    loop, prev = [start], None
    while True:
        here = loop[-1]
        nxt = touching[here][0] if touching[here][0] != prev else touching[here][1]
        if nxt == start:
            break
        prev = here
        loop.append(nxt)
    return loop if len(loop) == len(touching) else []


def boundary_sides(verts: Points, faces: Faces) -> list[list[int]]:
    """Return the sides of a family's single boundary loop, split at its corners.

    A corner is a boundary vertex where the loop turns by more than
    :data:`CORNER_DEGREES`. A family whose boundary is not one loop, or that
    has fewer than four corners, has no sides.
    """
    loop = _single_loop(faces)
    pts = numpy.asarray(verts, dtype=float)
    n = len(loop)
    corners = [i for i, v in enumerate(loop) if turns(pts, loop[i - 1], v, loop[(i + 1) % n])]
    if len(corners) < 4:
        return []
    sides = []
    for k, i in enumerate(corners):
        j = corners[(k + 1) % len(corners)]
        sides.append(loop[i : j + 1] if j > i else loop[i:] + loop[: j + 1])
    return sides


def side_length(verts: Points, side: list[int]) -> float:
    pts = numpy.asarray(verts, dtype=float)[side]
    return float(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1).sum())


def assume_trailing_edge(work: _Work, name: str) -> None:
    """Give a family that found no trailing edge the longest side of its perimeter.

    The reference line of a family with no trailing edge listed for it is the
    longest side of its boundary, taken as parallel; when the grid is not
    recovered with it, the next longest is tried. The choice is warned. It applies
    to a sheet, which needs a line to orient its grid; a multiblock family tells
    its arcs against the longest side of its own boundary
    (:func:`._blocks.te_directions`), and a tube is left alone.
    """
    verts, faces = work.mesh.verts, work.mesh.families[name]
    bare = _blocks.recover_family(verts, faces, ())
    if not (isinstance(bare, str) and bare.startswith("a sheet needs a trailing edge")):
        return
    sides = boundary_sides(verts, faces)
    if not sides:
        return
    pts = numpy.asarray(verts, dtype=float)
    for side in sorted(sides, key=lambda s: -side_length(verts, s)):
        found = _blocks.recover_family(verts, faces, side)
        if isinstance(found, str):
            continue
        edges = {(min(a, b), max(a, b)) for a, b in zip(side, side[1:], strict=False)}
        work.te_of[name] = set(side)
        work.te_edges[name] = edges
        work.assumed.add(name)
        ends = " and ".join(
            "(" + ", ".join(str(round(float(c), 6)) for c in pts[v]) + ")"
            for v in (side[0], side[-1])
        )
        warn(
            f"{work.source} family {name}: no trailing edge is listed for it; its parallel and "
            f"normal directions are taken against the longest side of its boundary, {len(edges)} "
            f"edges from {ends} (mesh units), which is not written as a trailing edge. List its "
            "trailing edge in a [[perimeter]] file to orient it by that line.",
            GeometryWarning,
            stacklevel=2,
        )
        return


def choose_perimeter_line(work: _Work, name: str) -> None:
    """Keep the perimeter line on a family that recovers its grid, trying each list in turn.

    The family's own rearmost listed edges come first (the edges its surface
    lies forward of, :func:`._obj.rearmost_edges`), so a rear panel whose front
    edge is the listed rearmost line of the panel ahead keeps its own trailing
    edge, not the longer hinge. Then the exported lists. Of each list, the
    whole of it on the family is tried first (a trailing edge in pieces stays
    whole), then each connected piece from the longest, then each piece cut at
    its corners, so a trailing edge and a root listed together in one file are
    told apart. A line is kept when the grid recovered with it holds it on its
    trailing-edge line; a family no line orients keeps its lines as found, for
    the refusal or remesh to name.
    """
    edges = work.te_edges[name]
    verts = work.mesh.verts
    lists = [edges & listed for listed in work.perimeter_sets if edges & listed]
    own = edges & work.te_rear.get(name, set())
    lines: list[set[tuple[int, int]]] = [own] if own else []
    # All the lists together next: a trailing edge split over lists (a wake gap) stays whole.
    lines += [set().union(*lists)] if len(lists) > 1 else []
    for on in lists:
        parts = sorted(pieces(on), key=lambda line: -length(verts, line))
        cut = [c for part in parts for c in corners(verts, part)]
        lines += [on, *parts, *sorted(cut, key=lambda line: -length(verts, line))]
    for line in lines:
        vertices = {v for edge in line for v in edge}
        found = _blocks.recover_family(verts, work.mesh.families[name], vertices)
        if not isinstance(found, str) and on_grid_line(found, line):
            work.te_edges[name] = line
            work.te_of[name] = vertices
            return


def on_grid_line(found: _grid.Grid | _blocks.Blocks, line: set[tuple[int, int]]) -> bool:
    """Return whether a line lies on the grid's trailing-edge line (is all of it, on blocks)."""
    if isinstance(found, _blocks.Blocks):
        chains = [found.arcs[i] for i in found.te_arcs]
    elif found.smooth:
        chains = []
    else:
        chains = [found.ids[:, 0].tolist()]
    represented = {
        (min(a, b), max(a, b)) for chain in chains for a, b in zip(chain, chain[1:], strict=False)
    }
    if isinstance(found, _blocks.Blocks):
        return bool(represented) and represented == line
    return bool(represented) and line <= represented


def corners(verts: Points, piece: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Return a path of edges cut where it turns (:func:`turns`); a branch is not cut."""
    touching: dict[int, list[int]] = {}
    for a, b in piece:
        touching.setdefault(a, []).append(b)
        touching.setdefault(b, []).append(a)
    ends = [v for v, n in touching.items() if len(n) == 1]
    if len(ends) != 2 or any(len(n) > 2 for n in touching.values()):
        return []
    path = [ends[0]]
    while len(path) < len(touching):
        path.append(next(n for n in touching[path[-1]] if n not in path[-2:-1]))
    pts = numpy.asarray(verts, dtype=float)
    cuts = [0]
    cuts += [i for i in range(1, len(path) - 1) if turns(pts, path[i - 1], path[i], path[i + 1])]
    cuts.append(len(path) - 1)
    if len(cuts) == 2:
        return []
    return [
        {(min(a, b), max(a, b)) for a, b in zip(path[i:j], path[i + 1 : j + 1], strict=True)}
        for i, j in zip(cuts, cuts[1:], strict=False)
    ]


def pieces(edges: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Return the connected pieces of a set of edges."""
    touching: dict[int, set[int]] = {}
    for a, b in edges:
        touching.setdefault(a, set()).add(b)
        touching.setdefault(b, set()).add(a)
    pieces: list[set[tuple[int, int]]] = []
    seen: set[int] = set()
    for start in touching:
        if start in seen:
            continue
        stack, members = [start], {start}
        while stack:
            for n in touching[stack.pop()] - members:
                members.add(n)
                stack.append(n)
        seen |= members
        pieces.append({e for e in edges if e[0] in members})
    return pieces


def length(verts: Points, edges: set[tuple[int, int]]) -> float:
    return sum(float(numpy.linalg.norm(verts[a] - verts[b])) for a, b in edges)


def reach(work: _Work) -> float:
    """Return how far from its source line a refined grid's node lies at most."""
    return _geometry.GRID_SURFACE_DISTANCE * work.scale


def refined_edge_ends(
    verts: Points, faces: Faces, te: Points, tolerance: float
) -> tuple[Points, Points]:
    """Return the two ends of the refined edge each trailing-edge point is the mid-point of.

    The ends of a coarsened edge are source nodes and lie on their source line,
    where the mid-point of a chord across a curved line does not (its sagitta
    grows with the chord). A point that is no edge's mid-point stands for both
    its ends.
    """
    pts = numpy.asarray(te, dtype=float).reshape(-1, 3)
    a, b = pts.copy(), pts.copy()
    if not len(pts) or not faces:
        return a, b
    edges, mid = _obj.edge_midpoints(verts, faces)
    distance, index = _geometry.NearestIndex(mid).query(pts)
    hit = distance < tolerance
    v = numpy.asarray(verts, dtype=float)
    a[hit] = v[edges[index[hit], 0]]
    b[hit] = v[edges[index[hit], 1]]
    return a, b


def on_line(
    work: _Work, mids: Points, a: Points, b: Points, edges: set[tuple[int, int]]
) -> NDArray[numpy.bool_]:
    """Return per trailing-edge point whether it lies on the listed source ``edges``.

    A point within the grid's surface distance of the edges does. So
    does the mid-point of a coarsened chord across a curved line, off it by its
    sagitta: its ends ``a`` and ``b`` are source nodes of the edges, joined along
    them by a path no longer than twice the chord. A chord across a gap left in
    the list (a wake gap) finds no such path and is not written.
    """
    pts = numpy.asarray(mids, dtype=float).reshape(-1, 3)
    if not len(pts) or not edges:
        return numpy.zeros(len(pts), dtype=bool)
    within = reach(work)
    verts = numpy.asarray(work.mesh.verts, dtype=float)
    sa, sb = segments(verts, edges)
    distance, _ = _geometry.segment_distances(pts, sa, sb)
    keep = distance <= within
    ends_a = numpy.asarray(a, dtype=float).reshape(-1, 3)
    ends_b = numpy.asarray(b, dtype=float).reshape(-1, 3)
    nodes = sorted({v for e in edges for v in e})
    index = _geometry.NearestIndex(verts[nodes])
    da, ia = index.query(ends_a)
    db, ib = index.query(ends_b)
    chords = numpy.linalg.norm(ends_a - ends_b, axis=1)
    joined: dict[int, list[tuple[int, float]]] = {}
    for u, v in edges:
        span = float(numpy.linalg.norm(verts[u] - verts[v]))
        joined.setdefault(u, []).append((v, span))
        joined.setdefault(v, []).append((u, span))
    for i in numpy.flatnonzero(~keep & (da <= within) & (db <= within) & (ia != ib)):
        keep[i] = _path_within(joined, nodes[int(ia[i])], nodes[int(ib[i])], 2.0 * float(chords[i]))
    return keep


def _path_within(
    joined: Mapping[int, list[tuple[int, float]]], start: int, goal: int, limit: float
) -> bool:
    """Return whether ``start`` reaches ``goal`` along ``joined`` by a path within ``limit``."""
    best = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        so_far, node = heapq.heappop(queue)
        if node == goal:
            return True
        if so_far > best.get(node, numpy.inf):
            continue
        for other, span in joined.get(node, ()):
            total = so_far + span
            if total <= limit and total < best.get(other, numpy.inf):
                best[other] = total
                heapq.heappush(queue, (total, other))
    return False


def on_exported(
    work: _Work, te: Points, verts: Points | None = None, faces: Faces | None = None
) -> Points:
    """Drop the level's trailing-edge points that lie on a list not exported.

    A point on a line of an exported ``[[perimeter]]`` list is kept, and so is
    a point on no list (a trailing edge a family assumed), except under
    ``te_families``, where only the rearmost lines picked are trailing edges and
    a line a grid assumed to orient itself is not written. Whether a point lies
    on a line is :func:`on_line`'s, given the level's ``verts`` and ``faces``
    for the ends of the edge each point is the mid-point of.
    """
    assert work.te_sets is not None
    exported, other = work.te_sets
    picked = any(p.te_families for p in work.request.perimeters)
    if not len(te) or (not other and not picked):
        return te
    if verts is not None and faces is not None:
        a, b = refined_edge_ends(verts, faces, te, work.te_tolerance)
    else:
        a, b = te, te
    if picked:
        keep = numpy.zeros(len(te), dtype=bool)
    else:
        keep = ~on_line(work, te, a, b, other)
    keep |= on_line(work, te, a, b, exported)
    return te[keep]


def own_te_files(
    work: _Work,
    folder: Path,
    verts: Points,
    faces: Mapping[str, Faces],
    te: Points,
    *,
    stems: Mapping[str, str],
) -> dict[str, Path]:
    """Write each family's ``<level>.<family>.te.txt``: the level's points on its edges.

    Under ``te_families`` a family's file holds its own rearmost line alone: a
    listed line it shares with the family ahead of it (a hinge) is that family's.
    """
    picked = any(p.te_families for p in work.request.perimeters)
    every = [f for fs in faces.values() for f in fs]
    a, b = refined_edge_ends(verts, every, te, work.te_tolerance)
    own: dict[str, Path] = {}
    for name, family_faces in faces.items():
        _, mid = _obj.edge_midpoints(verts, family_faces)
        distance, _ = _geometry.NearestIndex(mid).query(te)
        on = distance < work.te_tolerance
        if picked and work.te_rear.get(name):
            on &= on_line(work, te, a, b, work.te_rear[name])
        points = te[on]
        if len(points):
            own[name] = _obj.write_te(
                folder / f"{folder.name}.{stems[name]}{_obj.TE_SUFFIX}", work.te_unit, points
            )
    return own


def on_listed(
    work: _Work,
    mids: Points,
    listed: set[tuple[int, int]],
    points: Points | None = None,
    faces: Faces | None = None,
) -> Points:
    """Keep a refined grid's trailing-edge points that lie on a listed source edge.

    The grid writes its whole trailing-edge line; a gap the lists leave on
    purpose (a wake gap) is not written. Without listed edges every point stays.
    Given the level's ``points`` and ``faces``, the ends of the edge each point
    is the mid-point of decide (:func:`refined_edge_ends`), so a coarsened chord
    across a curved listed line is kept with its ends on the line.
    """
    if not len(mids) or not listed:
        return mids
    if points is not None and faces is not None:
        a, b = refined_edge_ends(points, faces, mids, work.te_tolerance)
    else:
        a, b = mids, mids
    return mids[on_line(work, mids, a, b, listed)]


def segments(verts: Points, edges: set[tuple[int, int]]) -> tuple[Points, Points]:
    """Return the two ends of each edge, in a stable order."""
    pairs = sorted(edges)
    return (
        verts[[a for a, _ in pairs]].reshape(-1, 3),
        verts[[b for _, b in pairs]].reshape(-1, 3),
    )


def te_vertices(mesh: ObjMesh, family: str, points: Points, tolerance: float) -> set[int]:
    """Return the trailing-edge vertices: the ends of the edges whose mid-point is listed."""
    return {v for edge in listed_edges(mesh, family, points, tolerance) for v in edge}


def listed_edges(
    mesh: ObjMesh, family: str, points: Points, tolerance: float
) -> set[tuple[int, int]]:
    """Keep the listed edges themselves; their vertex set cannot represent a wake gap.

    A point names the edge whose mid-point lies closer than ``tolerance``, the
    request's ``point_fraction`` of the mesh size.
    """
    if not len(points):
        return set()
    edges, mid = _obj.edge_midpoints(mesh.verts, mesh.families[family])
    dist, index = _geometry.NearestIndex(mid).query(points)
    hit = dist < tolerance
    return {(int(a), int(b)) for a, b in edges[index[hit]]}
