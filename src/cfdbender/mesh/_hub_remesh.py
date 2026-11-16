"""Remesh a hub band between two frozen loops, shared by hub-rings and repitch.

Both verbs take a hole of hub faces out around a blade root and fill the annulus between an inner
loop (the outer ring of ``hub-rings``, the turned root of the repitch remesh route) and the hole's
boundary: first triangulated in a chart of the hub, then remeshed quad-dominant by the remesh
``refine`` uses, both loops frozen and every free node projected on the source hub. The kept faces
keep their place and order, and each blade's new faces take the place of its first face taken,
because the solver's results depend on face order. The checks of the result live here too, so
both verbs hold their output to the source in one way: broken invariants (a kept node, face or
open edge changed, an edge over two faces, a frozen loop node moved) refuse; the quality of the
new cells is measured and returned for the verb to warn.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from cfdbender._errors import GeometryWarning, InputValueError, warn
from cfdbender.mesh._hub_rings_grid import (
    Region,
    _best_ear,
    _normals,
    boundary_loop,
    clip_annulus,
    folded,
    inside,
    islands_filled,
    march,
    one_layer,
    region,
    signed_area,
    walk_loops,
)
from cfdbender.mesh._refine._obj import ObjMesh
from cfdbender.mesh._refine._remesh import (
    PASSES,
    SMOOTH_PASSES,
    Remesher,
    element_report,
)
from cfdbender.mesh._refine._remesh_group import FROZEN, Patch
from cfdbender.mesh._repitch import HubSurface, RingChart, _nearest
from cfdbender.mesh._repitch_grid import edges, incidence, quality, root_loop

Edge = tuple[int, int]
Triangles = list[tuple[int, int, int]]

#: The widths tried, each this fraction of the one before, when a band does not fit.
SHRINK, TRIES = 0.9, 44
#: The remesh passes tried in turn on an annulus, while a pass count turns a cell over.
PASSES_TRIED = (PASSES, 4, 2, 1, 0)


def need(ok: bool, message: str, verb: str) -> None:
    """Refuse with the verb's name when a condition does not hold."""
    if not ok:
        raise InputValueError(f"{verb}: {message}. No file was written.")


class Plan(Protocol):
    """What the remesh of one blade's band reads: the blade, the hub family, the hole."""

    name: str
    family: str
    taken: set[int]
    hole: list[int]
    tris: Triangles


def hub_family(mesh: ObjMesh, hubs: list[str], loop: list[int], blade: str, verb: str) -> str:
    """Return the one hub family the faces along a root loop belong to."""
    root = set(edges(loop))
    found = {n for n in hubs for f in mesh.families[n] if root & set(edges(f))}
    need(
        len(found) == 1,
        f"blade {blade!r}: the hub faces along its root belong to {sorted(found)}, not one family",
        verb,
    )
    return found.pop()


def forbidden(mesh: ObjMesh, family: str, blade: str, gone: set[int]) -> set[int]:
    """Return the hub faces never taken: on an open boundary, another family or another hole."""
    owners = incidence([f for fs in mesh.families.values() for f in fs])
    banned = {v for e, own in owners.items() if len(own) == 1 for v in e}
    for name, faces in mesh.families.items():
        if name not in (family, blade):
            banned.update(v for f in faces for v in f)
    faces = mesh.families[family]
    return {i for i, f in enumerate(faces) if banned & set(f)} | gone


def root_edge_median(mesh: ObjMesh, family: str, loop: list[int], verb: str) -> float:
    """Return the median length of the hub edges that leave a root loop (its local cell size)."""
    on = set(loop)
    lengths = [
        float(np.linalg.norm(mesh.verts[a] - mesh.verts[b]))
        for a, b in incidence(mesh.families[family])
        if (a in on) != (b in on)
    ]
    need(bool(lengths), f"no hub edge of {family!r} leaves the root loop", verb)
    return float(np.median(lengths))


def reached(faces: list[list[int]], area: Region, rows: np.ndarray, loop: list[int]) -> set[int]:
    """Return the hub faces a march reaches: inside its outer row, holding a row node, or root."""
    local = {int(v): k for k, v in enumerate(area.ids)}
    points = rows[1:].reshape(-1, 2)
    root = set(edges(loop))
    found = {i for i, f in enumerate(faces) if root & set(edges(f))}
    for i, face in enumerate(faces):
        if any(v not in local for v in face):
            continue
        xy = area.xy[[local[v] for v in face]]
        if inside(xy, rows[-1]).any() or inside(points, xy).any():
            found.add(i)
    return found


@dataclass
class Hole:
    """The hub faces a band takes, the hole's boundary loop and the annulus's first triangles."""

    taken: set[int]
    hole: list[int]
    tris: Triangles
    route: str


def cut(
    verts: np.ndarray,
    faces: list[list[int]],
    area: Region,
    reach: tuple[np.ndarray, list[int]],
    inner: np.ndarray,
    *,
    banned: set[int],
) -> Hole | None:
    """Take the faces a march reaches plus one layer; triangulate from ``inner`` to the hole.

    ``reach`` holds the marched chart rows and the root loop they start from; ``inner`` is the
    annulus's inner loop in the chart, counterclockwise. Returns None when the hole would take a
    banned face, has no single boundary, or neither first triangulation closes.
    """
    found = hole_of(verts, faces, area, reach, banned=banned)
    if found is None:
        return None
    return join(inner, found[2], found[:2])


def hole_of(
    verts: np.ndarray,
    faces: list[list[int]],
    area: Region,
    reach: tuple[np.ndarray, list[int]],
    *,
    banned: set[int],
) -> tuple[set[int], list[int], np.ndarray] | None:
    """Return the faces a march reaches plus one layer, the hole's loop and its chart points.

    The loop runs counterclockwise in the chart. None when the hole would take a banned face or
    has no single boundary.
    """
    rows, loop = reach
    taken = islands_filled(faces, one_layer(faces, reached(faces, area, rows, loop)))
    hole = None if taken & banned else boundary_loop(faces, taken, loop)
    if hole is None:
        return None
    hole_xy = area.chart.coordinates(verts[hole])
    if signed_area(hole_xy) < 0.0:
        hole, hole_xy = hole[::-1], hole_xy[::-1]
    return taken, hole, hole_xy


def join(inner: np.ndarray, hole_xy: np.ndarray, found: tuple[set[int], list[int]]) -> Hole | None:
    """Triangulate from ``inner`` to the hole's loop: by the two-pointer walk, else by ears."""
    tris, route = walk_loops(inner, hole_xy), "two-pointer walk"
    if tris is None:
        tris, route = clip_annulus(inner, hole_xy), "ear clipping"
    if tris is None:
        return None
    return Hole(found[0], found[1], tris, route)


def targets(mesh: ObjMesh, family: str) -> np.ndarray:
    """Return, per node, the mean length of the hub edges at it (0 off the hub)."""
    total = np.zeros(len(mesh.verts))
    count = np.zeros(len(mesh.verts))
    for a, b in incidence(mesh.families[family]):
        length = float(np.linalg.norm(mesh.verts[a] - mesh.verts[b]))
        total[[a, b]] += length
        count[[a, b]] += 1
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _loop_target(points: np.ndarray) -> np.ndarray:
    side = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return 0.5 * (side + np.roll(side, 1))


def _remesher(
    mesh: ObjMesh, plan: Plan, inner: np.ndarray, surface: HubSurface
) -> tuple[Remesher, Patch]:
    """Return the remesher of an annulus, both loops frozen, and the patch it starts from."""
    n, m = len(inner), len(plan.hole)
    hole = mesh.verts[plan.hole]
    loops = [[(k, (k + 1) % n) for k in range(n)], [(n + k, n + (k + 1) % m) for k in range(m)]]
    frozen = {(min(a, b), max(a, b)) for loop in loops for a, b in loop}
    patch = Patch(
        np.vstack([inner, hole]),
        [list(t) for t in plan.tris],
        [plan.family] * len(plan.tris),
        np.r_[_loop_target(inner), _loop_target(hole)],
        dict.fromkeys(frozen, FROZEN),
        set(range(n + m)),
        frozen,
    )
    remesher = Remesher(patch)
    # Every node is projected on the source hub, not on the first triangulation.
    remesher.surface = surface.mesh
    remesher.h_source = targets(mesh, plan.family)
    if getattr(plan, "sizing", "source") == "loops":
        remesher.h_source = loop_sizing(mesh, plan, inner, remesher.h_source)
    return remesher, patch


def loop_sizing(mesh: ObjMesh, plan: Any, inner: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return the remesh's target lengths with the band's nodes sized from its two loops.

    At each node of the faces the band takes and of its root, the target is the side length of
    the nearest inner-loop node and of the nearest hole-loop node, weighted by the distances to
    them; elsewhere the source hub's own lengths stay.
    """
    from scipy.spatial import cKDTree

    hub = mesh.families[plan.family]
    near = sorted({v for i in plan.taken for v in hub[i]} | set(plan.loop))
    hole = mesh.verts[plan.hole]
    d_in, i_in = cKDTree(inner).query(mesh.verts[near])
    d_out, i_out = cKDTree(hole).query(mesh.verts[near])
    w = d_in / np.maximum(d_in + d_out, 1e-300)
    sized = source.copy()
    sized[near] = (1.0 - w) * _loop_target(inner)[i_in] + w * _loop_target(hole)[i_out]
    return sized


def _finished(
    remesher: Remesher, plan: Plan, patch: Patch, surface: HubSurface, verb: str
) -> tuple[np.ndarray, list[list[int]]]:
    """Repair, pair and smooth a remesher after its passes; return its points and faces."""
    remesher.repair_degenerate()
    remesher.pair_quads([plan.family])
    for _ in range(SMOOTH_PASSES):
        remesher.smooth_quads([plan.family])
    points, families, _ = remesher.result()
    frozen = len(patch.verts)
    need(
        np.array_equal(points[:frozen], patch.verts),
        f"blade {plan.name!r}: the annulus remesh moved a frozen loop node",
        verb,
    )
    # A node a pass made last (a split) can sit on a chord; every free node ends on the hub.
    if len(points) > frozen:
        points[frozen:] = surface.nearest(points[frozen:])[0]
    return points, families[plan.family]


def remeshed(
    mesh: ObjMesh, plan: Plan, inner: np.ndarray, surface: HubSurface, passes: int, *, verb: str
) -> tuple[np.ndarray, list[list[int]]]:
    """Remesh the annulus with a number of passes; return its points and faces.

    The points are the inner loop (0..n), the hole's loop (n..n+m) and the free nodes after.
    """
    remesher, patch = _remesher(mesh, plan, inner, surface)
    remesher.run(passes)
    return _finished(remesher, plan, patch, surface, verb)


def remesh_series(
    mesh: ObjMesh, plan: Plan, inner: np.ndarray, surface: HubSurface, verb: str
) -> dict[int, tuple[np.ndarray, list[list[int]]]]:
    """Return :func:`remeshed` at every pass count of :data:`PASSES_TRIED`, the passes shared.

    A pass of the remesher does not depend on how many follow, so the passes run once, from
    the fewest up, and each count is finished on a copy; the result at each count is the one
    :func:`remeshed` gives. The copy shares only the source hub, which nothing changes.
    """
    remesher, patch = _remesher(mesh, plan, inner, surface)
    done, found = 0, {}
    for passes in sorted(PASSES_TRIED):
        remesher.run(passes - done)
        done = passes
        twin = copy.deepcopy(remesher, {id(surface.mesh): surface.mesh})
        found[passes] = _finished(twin, plan, patch, surface, verb)
    return found


def turned_cells(
    points: np.ndarray, faces: list[list[int]], surface: HubSurface, sense: int
) -> int:
    """Count the cells whose normal, in the written sense, opposes the nearest source face's."""
    centres = np.array([points[f].mean(axis=0) for f in faces])
    _, _, face_of = _nearest(surface.mesh, centres)
    count = 0
    for face, tri in zip(faces, face_of, strict=True):
        p = points[face]
        normal = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        count += sense * float(normal @ surface.mesh.face_normals[tri]) <= 0.0
    return count


Counter = Callable[[np.ndarray, list[list[int]], HubSurface, int], int]


def annulus(
    mesh: ObjMesh,
    plan: Plan,
    inner: np.ndarray,
    surface: HubSurface,
    sense: int,
    *,
    turned: Counter = turned_cells,
    verb: str,
) -> tuple[np.ndarray, list[list[int]], int]:
    """Remesh the annulus quad-dominant; return its points, faces (inner 0.., hole n..), passes.

    The remesh of ``refine`` runs its full passes; when they leave a cell turned against the
    source hub, fewer passes are tried, down to none, where the first triangulation, valid by
    construction, is only paired and smoothed.
    """
    for passes in PASSES_TRIED:
        points, faces = remeshed(mesh, plan, inner, surface, passes, verb=verb)
        if not turned(points, faces, surface, sense):
            break
    return points, faces, passes


def forward(faces: list[list[int]], loop: list[int]) -> bool:
    """Return whether the hub face on the root's first edge runs that edge in the loop's order."""
    a, b = loop[0], loop[1]
    face = next(f for f in faces if a in f and b in f)
    return face[(face.index(a) + 1) % len(face)] == b


def numbered(
    faces: list[list[int]], loops: tuple[Any, list[int]], first: int, free: int, ahead: bool
) -> list[list[int]]:
    """Return annulus faces in mesh numbering: inner loop, hole loop, then ``free`` new nodes.

    The new nodes are numbered from ``first``; ``ahead`` is :func:`forward` of the root, and a
    root run the other way writes every face reversed.
    """
    inner, hole = loops
    local = np.r_[inner, hole, first + np.arange(free)]
    return [[int(local[v]) for v in (f if ahead else f[::-1])] for f in faces]


@dataclass
class Built:
    """What the bands add: points per blade, faces per blade in mesh numbering."""

    points: list[np.ndarray]
    faces: dict[str, list[list[int]]]
    quads: dict[str, dict[str, Any]]


@dataclass
class Result:
    """The rebuilt mesh with what the checks need to compare it with the source."""

    source: ObjMesh
    mesh: ObjMesh
    plans: list[Any]
    index: np.ndarray
    added: dict[str, list[list[int]]]
    kept: list[tuple[str, int]]
    annulus: dict[str, dict[str, Any]]


def assemble(source: ObjMesh, plans: list[Any], built: Built) -> Result:
    """Drop the taken faces and the nodes only they used; put each blade's new faces in place."""
    taken = {(p.family, i): p.name for p in plans for i in p.taken}
    used = {
        v
        for name, fs in source.families.items()
        for i, f in enumerate(fs)
        if (name, i) not in taken
        for v in f
    }
    count = len(source.verts)
    # A hole node whose every kept face another blade's hole took (two holes that touch) is
    # still a corner of both annuli: it stays, or their faces would name a removed node.
    used |= {v for faces in built.faces.values() for f in faces for v in f if v < count}
    alive = np.array([i in used for i in range(count)])
    every = np.vstack([source.verts, *built.points]) if built.points else source.verts
    keep = np.r_[alive, np.ones(len(every) - count, dtype=bool)]
    index = np.where(keep, np.cumsum(keep) - 1, -1)
    families: dict[str, list[list[int]]] = {}
    kept: list[tuple[str, int]] = []
    for name, faces in source.families.items():
        rows, placed = [], set()
        for i, face in enumerate(faces):
            face_of = taken.get((name, i))
            if face_of is None:
                rows.append([int(index[v]) for v in face])
                kept.append((name, i))
            elif face_of not in placed:
                placed.add(face_of)
                rows += [[int(index[v]) for v in f] for f in built.faces[face_of]]
        families[name] = rows
    text = source.vertex_text
    new_text = (
        ([t for t, a in zip(text, alive, strict=True) if a] + [None] * int(keep[count:].sum()))
        if len(text) == count
        else []
    )
    mesh = ObjMesh(every[keep], families, list(source.header), source.family_tag, new_text)
    added = {
        name: [[int(index[v]) for v in f] for f in faces] for name, faces in built.faces.items()
    }
    return Result(source, mesh, plans, index, added, kept, built.quads)


def written_blocks(blocks: list[tuple[str, int]], r: Result) -> list[tuple[str, int]]:
    """Return the face blocks of the output, in the source's file order.

    ``blocks`` are the source's (:func:`~cfdbender.mesh._refine._obj.face_blocks`): a
    family that returns after another has one block per run. Each block keeps its kept faces,
    and the block that holds a blade's first taken face also holds that blade's new faces, as
    :func:`assemble` places them; a block left empty is dropped and the runs it separated join.
    """
    taken = {(p.family, i): p.name for p in r.plans for i in p.taken}
    out: list[tuple[str, int]] = []
    start: dict[str, int] = {}
    placed: set[str] = set()
    for name, count in blocks:
        first = start.get(name, 0)
        start[name] = first + count
        size = 0
        for i in range(first, first + count):
            face_of = taken.get((name, i))
            if face_of is None:
                size += 1
            elif face_of not in placed:
                placed.add(face_of)
                size += len(r.added[face_of])
        if not size:
            continue
        if out and out[-1][0] == name:
            out[-1] = (name, out[-1][1] + size)
        else:
            out.append((name, size))
    return out


def chord_deviation(verts: np.ndarray, faces: list[list[int]]) -> float:
    """Return the largest sagitta of the source hub's facets, length times turn over eight.

    Two faces sharing an edge of length L whose normals differ by an angle t stand for a
    surface curved about that edge; the arc through it leaves the chord by about L t / 8.
    """
    normals = []
    for face in faces:
        p = verts[face]
        n = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        normals.append(n / max(float(np.linalg.norm(n)), 1e-300))
    worst = 0.0
    for (a, b), own in incidence(faces).items():
        if len(own) == 2:
            turn = math.acos(min(1.0, max(-1.0, float(normals[own[0]] @ normals[own[1]]))))
            worst = max(worst, float(np.linalg.norm(verts[a] - verts[b])) * turn / 8.0)
    return worst


def cell_findings(r: Result, surface: HubSurface) -> tuple[dict[str, Any], int]:
    """Return the new cells' quality and the count of new cells turned against the source hub."""
    new = [f for faces in r.added.values() for f in faces]
    shape = quality(r.mesh.verts, new)
    centres = np.array([r.mesh.verts[f].mean(axis=0) for f in new])
    _, _, face_of = _nearest(surface.mesh, centres)
    count = 0
    for face, tri in zip(new, face_of, strict=True):
        p = r.mesh.verts[face]
        normal = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        count += float(normal @ surface.mesh.face_normals[tri]) <= 0.0
    return shape, count


def cell_warnings(shape: dict[str, Any], count: int) -> list[str]:
    """Return the sentences that state folded and turned new cells with their numbers."""
    found = []
    if shape["folded_cells"]:
        found.append(
            f"{shape['folded_cells']} new hub cell(s) fold; the least scaled Jacobian of the new "
            f"cells is {shape['min_scaled_jacobian']:.6g}"
        )
    if count:
        found.append(f"{count} new hub cell(s) are turned against the source hub")
    return found


def distance_warning(
    r: Result, surface: HubSurface, allowed: float, skip: frozenset[int] = frozenset()
) -> tuple[float, list[str]]:
    """Return the largest distance of a new node to the source hub, and a warning beyond it.

    The nodes of the new faces are measured, but those in ``skip`` (output numbering): the
    turned root of repitch, which follows its blade and whose stand-off is reported apart.
    """
    added = sorted({v for faces in r.added.values() for f in faces for v in f} - skip)
    distance = surface.nearest(r.mesh.verts[added])[1]
    worst = int(np.argmax(distance))
    largest = float(distance[worst])
    if largest <= allowed:
        return largest, []
    return largest, [
        f"new hub node {added[worst]} (zero-based, in the output) lies {largest:.6g} from the "
        f"source hub, beyond the source's own chord deviation {allowed:.6g}"
    ]


def check_kept(r: Result, verb: str) -> None:
    """Refuse a kept node moved or rewritten, or a kept face out of place or order."""
    src, out = r.source, r.mesh
    alive = sorted({v for n, i in r.kept for v in src.families[n][i]})
    mapped = r.index[alive]
    need(np.array_equal(out.verts[mapped], src.verts[alive]), "a kept node moved", verb)
    if out.vertex_text and src.vertex_text:
        same = all(
            out.vertex_text[j] == src.vertex_text[i] for i, j in zip(alive, mapped, strict=True)
        )
        need(same, "a kept node is written differently from the source", verb)
    expected = [(n, [int(r.index[v]) for v in src.families[n][i]]) for n, i in r.kept]
    added = {tuple(f) for faces in r.added.values() for f in faces}
    actual = [(n, f) for n, faces in out.families.items() for f in faces if tuple(f) not in added]
    need(
        actual == expected,
        "a kept face differs from the source in place, family or node order",
        verb,
    )
    check_order(r, verb)


def check_order(r: Result, verb: str) -> None:
    """Refuse new faces anywhere but at the place of their blade's first taken face."""
    first = {p.name: min(p.taken) for p in r.plans}
    for plan in r.plans:
        before = sum(1 for n, i in r.kept if n == plan.family and i < first[plan.name])
        before += sum(
            len(r.added[p.name])
            for p in r.plans
            if p.family == plan.family and first[p.name] < first[plan.name]
        )
        place = r.mesh.families[plan.family][before : before + len(r.added[plan.name])]
        need(
            place == r.added[plan.name],
            f"blade {plan.name!r}: its new faces are not where its first taken face was",
            verb,
        )


def open_edges(mesh: ObjMesh) -> tuple[set[frozenset], int]:
    """Return the edges one face owns, by the places of their ends, and the edges over two faces."""
    owners = incidence([f for fs in mesh.families.values() for f in fs])
    places = {
        frozenset(tuple(mesh.verts[v].tolist()) for v in e)
        for e, own in owners.items()
        if len(own) == 1
    }
    return places, sum(len(own) > 2 for own in owners.values())


def edge_counts(r: Result, verb: str) -> dict[str, int]:
    """Return the open edges and edges over two faces; refuse a change of them or a turned cell."""
    now, over = open_edges(r.mesh)
    was, over0 = open_edges(r.source)
    need(now == was, f"open edges differ from the source: {len(now ^ was)} edge(s) changed", verb)
    need(over <= over0, f"{over - over0} edge(s) now have more than two faces", verb)
    for plan in r.plans:
        runs: dict[Edge, int] = {}
        for face in r.mesh.families[plan.family]:
            for a, b in zip(face, [*face[1:], face[0]], strict=True):
                runs[(a, b)] = runs.get((a, b), 0) + 1
        new_edges = {e for f in r.added[plan.name] for e in zip(f, [*f[1:], f[0]], strict=True)}
        need(
            all(runs.get(e, 0) == 1 for e in new_edges),
            f"blade {plan.name!r}: a new hub cell runs an edge the way its neighbour does",
            verb,
        )
    return {"open_edges": len(now), "edges_over_two_faces": over}


def area(verts: np.ndarray, faces: list[list[int]]) -> float:
    """Return the total area of polygons, each by its vector area."""
    return sum(
        0.5 * float(np.linalg.norm(np.cross(verts[f], np.roll(verts[f], -1, axis=0)).sum(axis=0)))
        for f in faces
    )


def kept_edges(r: Result) -> set[Edge]:
    """Return the source edges of the kept faces."""
    return {e for n, i in r.kept for e in edges(r.source.families[n][i])}


def new_by_old(source: ObjMesh, mesh: ObjMesh, index: np.ndarray) -> np.ndarray:
    """Return each source node's place in the output, NaN for a node removed."""
    out = np.full_like(source.verts, np.nan)
    alive = index[: len(source.verts)] >= 0
    out[alive] = mesh.verts[index[: len(source.verts)][alive]]
    return out


# The remesh route of repitch: one band per blade around its turned root.

#: Source cells across the default band beyond the root's largest move: two, the depth of the two
#: complete quad rings the ring route needs around a root.
BAND_CELLS = 2
#: Wider bands tried beside the first, each one more source cell across, while they fit.
WIDER = 2
#: Rows marched around every turned root, at least: two, as the ring route needs two rings.
MIN_ROWS = 2
#: Each thinner row height tried is this fraction of the one before, down to the root's finest
#: spacing, before a row is dropped.
THIN = 0.8
#: Least distance, in row heights, from the outer row to the hole's boundary.
CLEAR = 1.0
#: A row is at most this many times as high as the shorter root side at its node.
ROW_ASPECT = 2.0
#: Root nodes each way over which the row heights are smoothed.
ROW_SPAN = 2
_REPITCH = "repitch"


@dataclass
class Root:
    """What a band is cut around: the blade, its hub family, root loop, cell and barred faces."""

    name: str
    family: str
    loop: list[int]
    cell: float
    banned: set[int]
    foot: list[list[int]] = field(default_factory=list)
    share: float = math.inf


def footprint(mesh: ObjMesh, hub_faces: list[list[int]], loop: list[int]) -> list[list[int]]:
    """Return triangles closing the hub under a blade's root loop, run as the hub's faces run.

    The loop is triangulated in the plane of its largest spread by clipping, each time, the ear
    with the widest least angle; an empty list when no ear is found.
    """
    chart = RingChart(mesh.verts[loop], None)
    xy = chart.coordinates(mesh.verts[loop])
    ccw = list(range(len(loop)))
    if signed_area(xy) < 0.0:
        ccw.reverse()
    ring = list(ccw)
    tris = []
    while len(ring) > 3:
        k = _best_ear(xy, ring)
        if k is None:
            return []
        tris.append([ring[k - 1], ring[k], ring[(k + 1) % len(ring)]])
        del ring[k]
    tris.append(ring)
    # A counterclockwise ear runs the root's edges as the loop does; the hub face across a root
    # edge runs it the other way when it runs the loop forward.
    ahead = forward(hub_faces, [loop[i] for i in ccw])
    return [[loop[i] for i in (t[::-1] if ahead else t)] for t in tris]


@dataclass
class Band:
    """One blade's band: its root loop (counterclockwise in the chart), the hole and annulus."""

    name: str
    family: str
    loop: list[int]
    taken: set[int]
    hole: list[int]
    tris: Triangles
    route: str
    width: float
    rows: Any = None
    heights: Any = None
    graded: bool = True
    sizing: str = "source"


def root_move(verts: np.ndarray, moved: np.ndarray, loop: list[int]) -> float:
    """Return the largest move of a root node by the blade's turn."""
    return float(np.linalg.norm(moved[loop] - verts[loop], axis=1).max())


def band_at(
    mesh: ObjMesh, moved: np.ndarray, root: Root, width: float, *, graded: bool = True
) -> Band | None:
    """Return the band of hub faces within ``width`` of the old root, or None when it does not fit.

    The band's edge is marched from the old root in steps of at most one source cell, so the
    normals are taken again at least once a cell; the faces it reaches plus one layer leave. The
    annulus runs from the turned root to the hole's boundary. Rows are marched from the turned
    root, one local cell apart, as many as stay a cell inside the hole (the hole reaches past
    the width, one layer of source faces beyond it), and lifted on the hub; they are written as
    quads, as hub-rings writes its rings, so the cells around the root have the source's size,
    and only the gap from the last row to the hole is triangulated and remeshed. When a row
    folds or leaves the hole, or the triangulation does not close, fewer rows are tried, down
    to none. None when the march folds, the turned root turns over in the chart,
    the hole would take a barred face or no triangulation closes.
    """
    faces = mesh.families[root.family]
    zone = region(mesh.verts, faces + root.foot, root.loop, width)
    loop = list(root.loop)
    xy = zone.chart.coordinates(mesh.verts[loop])
    if signed_area(xy) < 0.0:
        loop = [loop[0], *loop[:0:-1]]
        xy = xy[[0, *range(len(loop) - 1, 0, -1)]]
    rows = march(xy, max(1, math.ceil(width / root.cell)), width)
    inner = zone.chart.coordinates(moved[loop])
    if rows is None or signed_area(inner) <= 0.0:
        return None
    found = hole_of(mesh.verts, faces, zone, (rows, loop), banned=root.banned)
    if found is None:
        return None
    taken, hole, hole_xy = found
    finest = float(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).min())
    for count, heights in row_tries(inner, hole_xy, (root.cell, finest), graded=graded):
        ring = _march_by(inner, count, heights) if count else inner[None]
        if ring is None or (count and not inside(ring[1:].reshape(-1, 2), hole_xy).all()):
            continue
        if count and clearance(ring[-1], hole_xy) < CLEAR * float(heights.min()):
            continue
        joined = join(ring[-1], hole_xy, (taken, hole))
        if joined is None:
            continue
        lifted = zone.chart.lift(ring[1:], mesh.verts[zone.ids], zone.triangles) if count else None
        made = Band(root.name, root.family, loop, taken, hole, joined.tris, joined.route, width)
        made.rows, made.heights, made.graded = lifted, heights, graded
        return made
    return None


def row_heights(inner: np.ndarray, hole_xy: np.ndarray, count: int, cell: float) -> np.ndarray:
    """Return, per root node, the height of each of ``count`` rows marched from it.

    One local cell, less where the hole is near (a row's height is left beyond the last row)
    and less where the root's sides are short (at most :data:`ROW_ASPECT` times the shorter side
    at the node, as at a blunt trailing edge), so the rows thin only where they must. The
    limits are smoothed along the root over :data:`ROW_SPAN` nodes each way without any node
    passing its own.
    """
    side = np.linalg.norm(np.roll(inner, -1, axis=0) - inner, axis=1)
    shorter = np.minimum(side, np.roll(side, 1))
    limit = np.minimum(np.minimum(cell, _gaps(inner, hole_xy) / (count + 1)), ROW_ASPECT * shorter)
    shifts = range(-ROW_SPAN, ROW_SPAN + 1)
    low = np.min([np.roll(limit, s) for s in shifts], axis=0)
    return np.asarray(np.mean([np.roll(low, s) for s in shifts], axis=0))


def row_tries(
    inner: np.ndarray, hole_xy: np.ndarray, sizes: tuple[float, float], *, graded: bool
) -> list[tuple[int, np.ndarray]]:
    """Return the (rows, heights per root node) tried around a turned root, the first preferred.

    As many rows as the hole leaves room for (at least :data:`MIN_ROWS`), each of the heights of
    :func:`row_heights` (``graded``) or all of one height, a local cell or the room the nearest
    point of the hole leaves; where they do not fit, the rows thin by :data:`THIN` at a time,
    until the thinnest would be below the root's finest spacing, before one row is dropped;
    last, no rows. ``sizes`` holds the local cell and the root's finest spacing.
    """
    cell, finest = sizes
    tries = []
    room = _gap(inner, hole_xy)
    for count in range(max(MIN_ROWS, math.floor(room / cell) - 1), 0, -1):
        heights = (
            row_heights(inner, hole_xy, count, cell)
            if graded
            else np.full(len(inner), min(cell, room / (count + 1)))
        )
        while heights.min() >= finest:
            tries.append((count, heights))
            heights = heights * THIN
    return [*tries, (0, np.zeros(len(inner)))]


def _march_by(root: np.ndarray, rows: int, heights: np.ndarray) -> np.ndarray | None:
    """Return the rows of a counterclockwise root marched out by per-node heights, or None."""
    made = [root]
    for _ in range(rows):
        nxt = made[-1] + heights[:, None] * _normals(made[-1])
        if folded(made[-1], nxt):
            return None
        made.append(nxt)
    return np.array(made)


def clearance(one: np.ndarray, other: np.ndarray) -> float:
    """Return the least chart distance between two closed loops, node to side both ways."""
    return min(_gap(one, other), _gap(other, one))


def _gap(inner: np.ndarray, loop: np.ndarray) -> float:
    """Return the least distance in the chart from the inner loop's nodes to a closed loop."""
    return float(_gaps(inner, loop).min())


def _gaps(inner: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Return, per node of ``inner``, its least distance in the chart to a closed loop."""
    a, b = loop, np.roll(loop, -1, axis=0)
    ab = b - a
    along = np.einsum("pk,sk->ps", inner, ab) - (a * ab).sum(axis=1)[None, :]
    t = np.clip(along / np.maximum((ab * ab).sum(axis=1), 1e-300)[None, :], 0.0, 1.0)
    near = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.asarray(np.linalg.norm(near - inner[:, None, :], axis=2).min(axis=1))


def _capped(root: Root, asked: float, cap: float, stated: bool, rows: float) -> float:
    """Return the first band width: the one asked, kept to the root's share of the gap.

    A stated ``band_width`` lowered to the share is warned with both numbers. A share that
    leaves no room (the share less one local cell at or below zero) is warned, and the band is
    no wider than the marched rows need (``rows``: the root's move plus their depth), never
    negative.
    """
    if cap <= 0.0:
        warn(
            f"repitch: blade {root.name!r}: its share of the gap to the nearest other root "
            f"({root.share:.6g}) leaves no room beyond one local cell ({root.cell:.6g}); the "
            f"band holds the marched rows only, width {min(asked, rows):.6g}",
            GeometryWarning,
        )
        return min(asked, rows)
    if stated and asked > cap:
        warn(
            f"repitch: blade {root.name!r}: band_width {asked:.6g} is lowered to {cap:.6g}, its "
            "share of the gap to the nearest other root less one local cell",
            GeometryWarning,
        )
    return min(asked, cap)


def bands_tried(
    mesh: ObjMesh, moved: np.ndarray, root: Root, stated: float | None
) -> tuple[list[Band], dict[str, Any]]:
    """Return the bands to remesh around one root and how their width was found.

    The first width is the stated ``band_width``, else the root's largest move plus
    :data:`BAND_CELLS` local cells (the median length of the hub edges leaving the root). When its
    band does not fit (it would take a face on an open boundary, another family or another
    blade's band), the width shrinks by :data:`SHRINK` until one fits, and that is warned.
    Otherwise up to :data:`WIDER` bands, each one local cell wider, join while they fit.
    """
    shift = root_move(mesh.verts, moved, root.loop)
    asked = stated if stated is not None else shift + BAND_CELLS * root.cell
    # On a wheel each band keeps to its share of the gap to the next root, less the one layer
    # of faces a hole takes beyond its width (about one local cell).
    cap = root.share - root.cell
    width = _capped(root, asked, cap, stated is not None, shift + MIN_ROWS * root.cell)
    info: dict[str, Any] = {
        "rule": "stated" if stated is not None else "root move plus two local cells",
        "asked": asked,
        "share_of_gap": None if math.isinf(root.share) else root.share,
        "root_largest_move": shift,
        "local_cell_size": root.cell,
        "shrunk_from": None,
    }
    first = band_at(mesh, moved, root, width)
    if first is not None:
        bands = [first]
        for k in range(1, WIDER + 1):
            if width + k * root.cell > cap:
                break
            wider = band_at(mesh, moved, root, width + k * root.cell)
            if wider is None:
                break
            bands.append(wider)
        return bands + _uniform(mesh, moved, root, bands), info
    for k in range(1, TRIES + 1):
        first = band_at(mesh, moved, root, width * SHRINK**k)
        if first is not None:
            break
    need(
        first is not None,
        f"blade {root.name!r}: no hub band down to a hundredth of the width {width:.6g} fits "
        f"on hub {root.family!r} without an open boundary (a seam), another family or another "
        "blade's band; state [repitch] band_width, or hub_method = 'rings' with [hub_rings]",
        _REPITCH,
    )
    assert first is not None
    warn(
        f"repitch: blade {root.name!r}: the hub band of width {width:.6g} reaches an open "
        f"boundary (a seam), another family or another blade's band; it is shrunk to "
        f"{first.width:.6g}",
        GeometryWarning,
    )
    info["shrunk_from"] = width
    return [first, *_uniform(mesh, moved, root, [first])], info


def _uniform(mesh: ObjMesh, moved: np.ndarray, root: Root, bands: list[Band]) -> list[Band]:
    """Return the bands of the same widths with rows of one height, where they fit."""
    made = [band_at(mesh, moved, root, b.width, graded=False) for b in bands]
    return [b for b in made if b is not None]


@dataclass
class Candidate:
    """One remesh of one band: its pass count, points, faces, turned cells and shape."""

    band: Band
    passes: int
    points: np.ndarray
    faces: list[list[int]]
    ahead: bool
    turned: int
    shape: dict[str, Any]
    margin: float = 0.0


def _inner(moved: np.ndarray, band: Band) -> np.ndarray:
    """Return the annulus's inner loop: the last held row, or the turned root without rows."""
    return moved[band.loop] if band.rows is None else band.rows[-1]


#: A cell scoring below this (its least corner Jacobian against the source hub's normal, or its
#: shortest side over its longest) has its free nodes moved by :func:`polish`.
POLISH_FLOOR = 0.5
#: Sweeps of :func:`polish` at most; it stops when a sweep moves no node.
POLISH_SWEEPS = 12
#: The steps a node is tried at, in its cells' mean edge length, in eight tangent directions.
POLISH_STEPS = (0.5, 0.25, 0.1, 0.05)
#: How an annulus remesh is sized, in the order tried: by the source hub's cells, then by
#: its two loops (:func:`loop_sizing`).
SIZINGS = ("source", "loops")


def _scores(q: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Score cells stacked as (..., corners, 3) against reference normals.

    The score is the lesser of the least corner Jacobian along the normal and the shortest side
    over the longest.
    """
    side = np.roll(q, -1, axis=-2) - q
    lengths = np.linalg.norm(side, axis=-1)
    corners = np.cross(side, -np.roll(side, 1, axis=-2))
    jac = (corners @ normal) / np.maximum(lengths * np.roll(lengths, 1, axis=-1), 1e-300)
    ratio = lengths.min(axis=-1) / np.maximum(lengths.max(axis=-1), 1e-300)
    return np.minimum(jac.min(axis=-1), ratio)


def _places(points: np.ndarray, v: int, ring: list[int], normal: np.ndarray) -> np.ndarray:
    """Return where a node is tried: its neighbours' centroid and steps about it, unprojected."""
    h = float(np.mean(np.linalg.norm(points[ring] - points[v], axis=1)))
    t1 = np.cross(normal, np.eye(3)[int(np.argmin(np.abs(normal)))])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    turns = [t1, -t1, t2, -t2, t1 + t2, t1 - t2, t2 - t1, -t1 - t2]
    steps = [points[v] + s * h * d / np.linalg.norm(d) for s in POLISH_STEPS for d in turns]
    return np.array([points[ring].mean(axis=0), *steps])


def _move(points: np.ndarray, v: int, cells: list[int], faces: list[list[int]], ref: Any) -> bool:
    """Move node ``v`` to its best tried place on the hub when that raises its worst cell."""
    normals, surface = ref
    ring = sorted({u for i in cells for u in faces[i] if u != v})
    normal = normals[cells].mean(axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-300)
    tries = surface.nearest(_places(points, v, ring, normal))[0]
    worst, now = np.full(len(tries), np.inf), np.inf
    for i in cells:
        q = np.repeat(points[faces[i]][None], len(tries), axis=0)
        q[:, faces[i].index(v)] = tries
        worst = np.minimum(worst, _scores(q, normals[i]))
        now = min(now, float(_scores(points[faces[i]], normals[i])))
    k = int(np.argmax(worst))
    if worst[k] <= now + 1e-9:
        return False
    points[v] = tries[k]
    return True


def polish(
    points: np.ndarray, faces: list[list[int]], frozen: int, surface: HubSurface, sense: int
) -> np.ndarray:
    """Return an annulus's points with the free nodes of its worst cells moved to better places.

    Every free node of a cell scoring below :data:`POLISH_FLOOR` is tried at its neighbours'
    centroid and at :data:`POLISH_STEPS` of its mean edge in eight directions of the tangent
    plane, each projected on the source hub, and takes the place that most raises the least
    score of its cells, if any does. The score is taken against the source hub's normal, so a
    cell turned over scores below zero and is lifted first. The first ``frozen`` points (the
    loops) never move; the faces do not change. At most :data:`POLISH_SWEEPS` sweeps.
    """
    points = points.copy()
    cells: dict[int, list[int]] = {}
    for i, face in enumerate(faces):
        for v in face:
            cells.setdefault(v, []).append(i)
    centres = np.array([points[f].mean(axis=0) for f in faces])
    normals = sense * surface.mesh.face_normals[_nearest(surface.mesh, centres)[2]]
    for _ in range(POLISH_SWEEPS):
        low = {
            v
            for i, face in enumerate(faces)
            if float(_scores(points[face], normals[i])) < POLISH_FLOOR
            for v in face
            if v >= frozen
        }
        moved = [_move(points, v, cells[v], faces, (normals, surface)) for v in sorted(low)]
        if not any(moved):
            break
    return points


def target_of(source_band: dict[str, Any]) -> dict[str, float]:
    """Return a band's quality target from its source band's numbers.

    The output band's least scaled Jacobian is to be at least half the source band's, and its
    largest aspect at most twice the source band's.
    """
    return {
        "least_scaled_jacobian": 0.5 * source_band["min_scaled_jacobian"],
        "largest_aspect": 2.0 * source_band["max_aspect"],
    }


def _margin(shape: dict[str, Any], target: dict[str, float]) -> float:
    """Return how far a band's shape clears its target: 1 or more when it meets it."""
    floor, ceiling = target["least_scaled_jacobian"], target["largest_aspect"]
    sj = shape["min_scaled_jacobian"]
    over = sj / floor if floor > 0.0 else 1.0 + sj - floor
    return min(over, ceiling / max(shape["max_aspect"], 1e-300))


def _candidate(band: Band, passes: int, made: Any, target: dict[str, float]) -> Candidate:
    """Polish one remesh of a band and measure it as written, held rows and annulus together."""
    (points, faces), moved, surface, ahead = made
    sense = 1 if ahead else -1
    points = polish(points, faces, len(band.loop) + len(band.hole), surface, sense)
    turned = turned_cells(points, faces, surface, sense)
    found = Candidate(band, passes, points, faces, ahead, turned, {})
    written, new, _ = _rows_and_annulus(found, len(moved))
    found.shape = quality(np.vstack([moved, *new]), written)
    found.margin = _margin(found.shape, target)
    return found


def best_candidate(
    mesh: ObjMesh, moved: np.ndarray, bands: list[Band], surface: HubSurface
) -> tuple[Candidate, list[dict[str, Any]]]:
    """Remesh the bands until one meets its target; return the best and the table.

    The bands come as :func:`bands_tried` gives them (graded rows at each width, then rows of
    one height); each is remeshed sized by the source hub's cells, then by its two loops
    (:func:`loop_sizing`). Fewer passes of the fallback are tried only while a cell is left
    turned against the source hub. Each remesh is polished (:func:`polish`) and measured as
    written, its held rows with its annulus. The first with no cell turned that meets its target
    (:func:`target_of`) ends the search; else the best has the fewest turned cells and, among
    those, the largest margin to its target: the lesser of its least scaled Jacobian over the
    target's and the target's largest aspect over its own.
    """
    tried: list[Candidate] = []
    for band in bands:
        hub = mesh.families[band.family]
        target = target_of(band_numbers(mesh.verts, [hub[i] for i in band.taken]))
        for sizing in SIZINGS:
            plan = dataclasses.replace(band, sizing=sizing)
            if _remesh_one(plan, (mesh, moved, surface), target, tried):
                return tried[-1], _table(tried)
    return max(tried, key=lambda c: (-c.turned, c.margin)), _table(tried)


def _remesh_one(
    band: Band, inputs: tuple[ObjMesh, np.ndarray, HubSurface], target: Any, tried: list[Candidate]
) -> bool:
    """Remesh a band at the pass counts it needs into ``tried``; True when one meets its target."""
    mesh, moved, surface = inputs
    ahead = forward(mesh.families[band.family], band.loop)
    series = remesh_series(mesh, band, _inner(moved, band), surface, _REPITCH)
    for passes in PASSES_TRIED:
        tried.append(_candidate(band, passes, (series[passes], moved, surface, ahead), target))
        if tried[-1].turned == 0:
            return tried[-1].margin >= 1.0
    return False


def _table(tried: list[Candidate]) -> list[dict[str, Any]]:
    """Return the record's line for every candidate tried."""
    return [
        {
            "band_width": c.band.width,
            "rows": "graded" if c.band.graded else "uniform",
            "sizing": c.band.sizing,
            "passes": c.passes,
            "turned_cells": c.turned,
            "min_scaled_jacobian": c.shape["min_scaled_jacobian"],
            "max_aspect": c.shape["max_aspect"],
            "target_margin": c.margin,
            "quad_share": element_report("quad-dominant", c.faces)["quad_share"],
        }
        for c in tried
    ]


def band_numbers(verts: np.ndarray, faces: list[list[int]]) -> dict[str, Any]:
    """Return a band's quality, its quad share and its mean edge length."""
    sides = [np.linalg.norm(np.roll(verts[f], -1, axis=0) - verts[f], axis=1) for f in faces]
    return {
        **quality(verts, faces),
        "quad_share": element_report("quad-dominant", faces)["quad_share"],
        "mean_edge": float(np.mean(np.concatenate(sides))),
    }


def _blade_numbers(
    mesh: ObjMesh, best: Candidate, info: dict[str, Any], table: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return what the record says of one blade's band; the output band is filled in later."""
    band = best.band
    return {
        "hub_family": band.family,
        "band_width": band.width,
        "band_width_rule": info["rule"],
        "band_width_asked": info["asked"],
        "shrunk_from": info["shrunk_from"],
        "local_cell_size": info["local_cell_size"],
        "root_largest_move": info["root_largest_move"],
        "share_of_gap": info["share_of_gap"],
        "candidate_widths": sorted({c["band_width"] for c in table}),
        "root_loop_nodes": len(band.loop),
        "held_rows": 0 if band.rows is None else len(band.rows),
        "row_height": 0.0 if band.heights is None else float(band.heights.min()),
        "row_height_largest": 0.0 if band.heights is None else float(band.heights.max()),
        "faces_taken": len(band.taken),
        "hole_boundary_nodes": len(band.hole),
        "first_triangulation": band.route,
        "rows": "graded" if band.graded else "uniform",
        "sizing": band.sizing,
        "passes": best.passes,
        "turned_cells": best.turned,
        "source_band": band_numbers(
            mesh.verts, [mesh.families[band.family][i] for i in band.taken]
        ),
        "candidates": table,
    }


def _rows_and_annulus(best: Candidate, start: int) -> tuple[list[list[int]], list[np.ndarray], int]:
    """Return a band's faces in mesh numbering from ``start``, its new points and their count.

    The held rows are written as quads from the root outward, as hub-rings writes its rings,
    then the remeshed annulus from the last row to the hole.
    """
    band = best.band
    n, m = len(band.loop), len(band.hole)
    count = 0 if band.rows is None else len(band.rows)
    ids = np.vstack(
        [np.asarray(band.loop)[None, :], start + np.arange(count * n).reshape(count, n)]
    )
    quads = [
        [ids[k, j], ids[k, (j + 1) % n], ids[k + 1, (j + 1) % n], ids[k + 1, j]]
        if best.ahead
        else [ids[k, j], ids[k + 1, j], ids[k + 1, (j + 1) % n], ids[k, (j + 1) % n]]
        for k in range(count)
        for j in range(n)
    ]
    free = len(best.points) - n - m
    ring = numbered(best.faces, (ids[-1], band.hole), start + count * n, free, best.ahead)
    points = ([] if band.rows is None else [band.rows.reshape(-1, 3)]) + [best.points[n + m :]]
    faces = [[int(v) for v in q] for q in quads] + ring
    return faces, points, count * n + free


def gap_shares(verts: np.ndarray, roots: dict[str, list[int]]) -> dict[str, float]:
    """Return each root's share of the gap to its nearest other root: half the least distance.

    The blades of a wheel are planned together: a band reaches no farther than its share, less
    the layer of faces a hole takes beyond its width, so no band takes room the next needs and
    none is shrunk to nothing by the bands planned before it. A lone root has no limit.
    """
    from scipy.spatial import cKDTree

    shares = {}
    for name, loop in roots.items():
        others = [v for other, ids in roots.items() if other != name for v in ids]
        if not others:
            shares[name] = math.inf
            continue
        shares[name] = 0.5 * float(cKDTree(verts[others]).query(verts[loop])[0].min())
    return shares


def remesh_bands(
    source: ObjMesh, turned: ObjMesh, names: tuple[list[str], list[str]], stated: float | None
) -> tuple[Result, HubSurface, dict[str, dict[str, Any]]]:
    """Remesh the hub band around every turned blade root, unchecked; return the result.

    ``turned`` is the source with the blades turned; every band is cut and measured against the
    same source hub, in the order the blades are named, and no band takes a face another took.
    """
    blades, hubs = names
    hub_faces = [f for n in hubs for f in source.families[n]]
    roots = {name: root_loop(source.families[name], hub_faces, name) for name in blades}
    feet = {name: footprint(source, hub_faces, loop) for name, loop in roots.items()}
    # The source hub closed under each blade's old root: a turned root uncovers part of the
    # footprint, and the band's nodes there need a surface to lie on.
    surface = HubSurface(source.verts, hub_faces + [f for foot in feet.values() for f in foot])
    shares = gap_shares(source.verts, roots)
    gone: dict[str, set[int]] = {}
    built = Built([], {}, {})
    start, plans, numbers = len(source.verts), [], {}
    for name in blades:
        loop = roots[name]
        family = hub_family(source, hubs, loop, name, _REPITCH)
        banned = forbidden(source, family, name, gone.setdefault(family, set()))
        cell = root_edge_median(source, family, loop, _REPITCH)
        root = Root(name, family, loop, cell, banned, feet[name], shares[name])
        bands, info = bands_tried(source, turned.verts, root, stated)
        best, table = best_candidate(source, turned.verts, bands, surface)
        gone[family] |= best.band.taken
        faces, points, made = _rows_and_annulus(best, start)
        built.faces[name] = faces
        built.points += points
        built.quads[name] = element_report("quad-dominant", faces)
        start += made
        plans.append(best.band)
        numbers[name] = _blade_numbers(source, best, info, table)
    every = np.vstack([turned.verts, *built.points])
    for name, blade in numbers.items():
        blade["faces_added"] = len(built.faces[name])
        blade["output_band"] = band_numbers(every, built.faces[name])
        blade["size_ratio"] = blade["output_band"]["mean_edge"] / blade["source_band"]["mean_edge"]
        blade["target"] = _target_check(name, blade)
    return assemble(turned, plans, built), surface, numbers


def target_miss(name: str, blade: dict[str, Any]) -> str | None:
    """Return the sentence that states a blade's band missing its target, or None when met."""
    target = target_of(blade["source_band"])
    out = blade["output_band"]
    sj, aspect = out["min_scaled_jacobian"], out["max_aspect"]
    if sj >= target["least_scaled_jacobian"] and aspect <= target["largest_aspect"]:
        return None
    return (
        f"blade {name!r}: the remeshed hub band misses its quality target: least "
        f"scaled Jacobian {sj:.4f} against {target['least_scaled_jacobian']:.4f} (half the "
        f"source band's), largest aspect {aspect:.4f} against "
        f"{target['largest_aspect']:.4f} (twice the source band's), with "
        f"{blade['held_rows']} rows of {blade['row_height']:.6g} in a band of "
        f"{blade['band_width']:.6g}"
    )


def _target_check(name: str, blade: dict[str, Any]) -> dict[str, Any]:
    """Hold a blade's output band to its target; warn of a miss with its numbers."""
    miss = target_miss(name, blade)
    if miss is not None:
        warn(f"repitch: {miss}; the mesh is written", GeometryWarning)
    return {**target_of(blade["source_band"]), "met": miss is None}
