"""The surface mesh value, its orientation, its checks and its OBJ interchange.

Pipeline role: pure numerical stage of the geometry package, downstream of the
mesher in the workspace. A :class:`Mesh` holds nodes, polygon cells and the
component and source surface of each cell. This module orients the winding of a
skin outward, repairs inverted and sliver triangles, measures the acceptance
checks (inverted, degenerate, non-manifold, free edges off the symmetry plane,
shared-edge direction, distance to the source surface) and reads and writes OBJ
with one group per component. It imports no geometry kernel.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import cfdbender._textio as _textio
from cfdbender._errors import InputValueError

__all__ = [
    "Mesh",
    "check_mesh",
    "coincident_nodes",
    "flip_slivers",
    "orient_mesh",
    "oriented_source_normals",
    "paired_triangles",
    "read_obj",
    "repair_inverted",
    "write_obj",
    "write_record",
]


@dataclass
class Mesh:
    """Nodes ``xyz`` ``(n, 3)``, polygon ``cells`` (node index tuples), and per-cell labels."""

    xyz: np.ndarray
    cells: list[tuple[int, ...]]
    components: list[str]
    surfaces: list[int]
    node_tags: np.ndarray


def directed_edges(cells):
    """Map each undirected edge to the ``(cell, direction)`` uses that walk it."""
    edges = defaultdict(list)
    for i, cell in enumerate(cells):
        for a, b in zip(cell, cell[1:] + cell[:1], strict=True):
            edges[tuple(sorted((a, b)))].append((i, 1 if a < b else -1))
    return edges


def inconsistent_edges(cells):
    """Return the shared edges that two cells walk in the same direction."""
    return [
        list(e)
        for e, uses in directed_edges(cells).items()
        if len(uses) == 2 and uses[0][1] == uses[1][1]
    ]


def area_vectors(mesh):
    """Return the area vector of every cell, fan-triangulated from its first node."""
    return np.array(
        [
            sum(
                (
                    np.cross(mesh.xyz[c[j]] - mesh.xyz[c[0]], mesh.xyz[c[j + 1]] - mesh.xyz[c[0]])
                    / 2
                    for j in range(1, len(c) - 1)
                ),
                start=np.zeros(3),
            )
            for c in mesh.cells
        ]
    )


def _propagate_winding(mesh):
    adjacent = defaultdict(list)
    for uses in directed_edges(mesh.cells).values():
        if len(uses) > 2:
            raise InputValueError("mesh: cannot orient a nonmanifold skin")
        if len(uses) == 2:
            (a, da), (b, db) = uses
            adjacent[a].append((b, -da * db))
            adjacent[b].append((a, -da * db))
    signs, islands = {}, []
    for seed in range(len(mesh.cells)):
        if seed in signs:
            continue
        signs[seed], queue, island = 1, deque([seed]), []
        while queue:
            a = queue.popleft()
            island.append(a)
            for b, relation in adjacent[a]:
                required = signs[a] * relation
                if b in signs and signs[b] != required:
                    raise InputValueError("mesh: skin has contradictory winding constraints")
                if b not in signs:
                    signs[b] = required
                    queue.append(b)
        islands.append(island)
    return signs, islands


def _island_score(mesh, cells, vectors, island, sense, *, axis_origin, direction, reference):
    if sense == "body":
        centers = np.array([mesh.xyz[list(cells[i])].mean(axis=0) for i in island])
        radial = centers - np.asarray(axis_origin)
        radial -= (radial @ direction)[:, None] * direction
        return float(np.sum(vectors[island] * radial))
    if sense == "reference":
        return float(np.sum(vectors[island] @ np.asarray(reference)))
    if sense == "mirror_volume":
        used = set(island)
        for edge, holders in directed_edges(cells).items():
            if (
                len(holders) == 1
                and holders[0][0] in used
                and np.any(abs(mesh.xyz[list(edge), 1]) > 1e-7)
            ):
                raise InputValueError("mesh: mirror volume requires all free edges on y=0")
        score = 0.0
        for i in island:
            p = mesh.xyz[list(cells[i])]
            for j in range(1, len(p) - 1):
                score += float(p[0] @ np.cross(p[j], p[j + 1])) / 3
        return score
    raise InputValueError(f"sense: unknown outward sense {sense!r}")


def orient_mesh(
    mesh, sense="mirror_volume", axis_origin=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), reference=None
):
    """Orient the winding: propagate over shared edges, then one outward sign per connected skin.

    ``mirror_volume`` assumes every open edge lies on y=0 (reversing the reflected skin closes it).
    ``body`` integrates flux away from the straight axis; ``reference`` uses a supplied vector.
    Only connectivity changes. Returns ``(mesh, record)``.

    Parameters
    ----------
    mesh : Mesh
        The mesh to orient.
    sense : str, optional
        Outward sense: ``mirror_volume``, ``body`` or ``reference``; default ``mirror_volume``.
    axis_origin : sequence of float, optional
        A point on the body axis; default the origin.
    axis : sequence of float, optional
        Direction of the body axis; default x.
    reference : sequence of float, optional
        Reference vector the outward flux is measured along; used by the ``reference`` sense.

    Returns
    -------
    tuple
        ``(mesh, record)``: the oriented mesh and a record of the orientation.

    Raises
    ------
    InputValueError
        If the skin is nonmanifold, its winding constraints contradict, the outward sense is
        indeterminate or unknown, or a mirror volume has a free edge off y=0.
    """
    signs, islands = _propagate_winding(mesh)
    cells = [c if signs[i] > 0 else tuple(reversed(c)) for i, c in enumerate(mesh.cells)]
    vectors = area_vectors(replace(mesh, cells=cells))
    direction = np.asarray(axis, float)
    direction /= np.linalg.norm(direction)
    scores = []
    for island in islands:
        score = _island_score(
            mesh,
            cells,
            vectors,
            island,
            sense,
            axis_origin=axis_origin,
            direction=direction,
            reference=reference,
        )
        if abs(score) <= 1e-14:
            raise InputValueError("mesh: physical outward sense is indeterminate")
        if score < 0:
            for i in island:
                cells[i] = tuple(reversed(cells[i]))
        scores.append(abs(score))
    return replace(mesh, cells=cells), {
        "sense": sense,
        "positive_scores": scores,
        "connected_skins": len(islands),
        "flipped_cells": sum(a != b for a, b in zip(mesh.cells, cells, strict=True)),
        "inconsistent_shared_edge_directions": len(inconsistent_edges(cells)),
    }


def oriented_source_normals(mesh, normals):
    """Choose one source-normal sign per source face, never one per cell.

    A locally folded cell therefore stays inverted against the oriented source face.

    Parameters
    ----------
    mesh : Mesh
        The oriented mesh.
    normals : array_like
        Source normals, one per cell, shape ``(cells, 3)``.

    Returns
    -------
    numpy.ndarray
        A copy of ``normals`` with the sign of each source face chosen to agree with the mesh.
    """
    result = np.array(normals, copy=True)
    vectors = area_vectors(mesh)
    surfaces = np.asarray(mesh.surfaces)
    for surface in set(mesh.surfaces):
        mask = surfaces == surface
        if np.sum(vectors[mask] * result[mask]) < 0:
            result[mask] *= -1
    return result


def edge_incidence(cells):
    """Map each undirected edge to the cells that use it."""
    edges = defaultdict(list)
    for index, cell in enumerate(cells):
        for a, b in zip(cell, cell[1:] + cell[:1], strict=True):
            edges[tuple(sorted((a, b)))].append(index)
    return dict(edges)


def corner_quality(xyz, cells, normals, area_tolerance=1e-14):
    """Return the indices of inverted cells and of degenerate cells."""
    inverted, degenerate = [], []
    for index, (cell, normal) in enumerate(zip(cells, normals, strict=True)):
        p = xyz[list(cell)]
        cross = np.cross(np.roll(p, -1, axis=0) - p, np.roll(p, 1, axis=0) - p)
        if np.any(cross @ normal < -area_tolerance):
            inverted.append(index)
        if len(set(cell)) != len(cell) or np.any(np.linalg.norm(cross, axis=1) <= area_tolerance):
            degenerate.append(index)
    return inverted, degenerate


def check_mesh(
    mesh,
    normals,
    distances,
    transition_surfaces=(),
    symmetry_tolerance=1e-7,
    *,
    distance_tolerance=5e-4,
    area_tolerance=1e-14,
):
    """Measure the acceptance checks of an oriented mesh and return them as one record.

    ``normals`` holds one oriented source normal per cell and ``distances`` the distance of each
    node to every incident source surface. ``accepted`` covers those checks only: not physical
    spacing and not solver setup.

    Parameters
    ----------
    mesh : Mesh
        The oriented mesh.
    normals : array_like
        One oriented source normal per cell, shape ``(cells, 3)``.
    distances : array_like
        Distance of each node to its source surfaces, shape ``(nodes,)``.
    transition_surfaces : sequence of int, optional
        Surfaces allowed to hold triangles; default none.
    symmetry_tolerance : float, optional
        Largest ``|y|`` of a free edge node, metres; default 1e-7.
    distance_tolerance : float, optional
        Largest accepted node-to-source distance, metres; default 5e-4.
    area_tolerance : float, optional
        Area below which a cell is degenerate, square metres; default 1e-14.

    Returns
    -------
    dict
        The record of counts, cell lists, tolerances and the ``accepted`` flag.

    Raises
    ------
    InputValueError
        If the shapes of ``normals`` or ``distances`` do not match the mesh, or any value is
        nonfinite, a normal is zero or a distance is negative.
    """
    xyz = np.asarray(mesh.xyz)
    normals, distances = np.asarray(normals), np.asarray(distances)
    if normals.shape != (len(mesh.cells), 3) or distances.shape != (len(xyz),):
        raise InputValueError("normals: one reference normal per cell and one distance per node")
    if (
        not np.all(np.isfinite(xyz))
        or not np.all(np.isfinite(normals))
        or not np.all(np.isfinite(distances))
        or np.any(distances < 0)
        or np.any(np.linalg.norm(normals, axis=1) == 0)
    ):
        raise InputValueError("normals: must be finite and nonzero, distances nonnegative")
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    inverted, degenerate = corner_quality(xyz, mesh.cells, normals, area_tolerance)
    edges = edge_incidence(mesh.cells)
    free = [list(e) for e, faces in edges.items() if len(faces) == 1]
    off = [e for e in free if np.any(np.abs(xyz[e, 1]) > symmetry_tolerance)]
    nonmanifold = [list(e) for e, faces in edges.items() if len(faces) > 2]
    nonquad = [i for i, c in enumerate(mesh.cells) if len(c) != 4]
    forbidden = [
        i for i in nonquad if mesh.surfaces[i] not in transition_surfaces or len(mesh.cells[i]) != 3
    ]
    inconsistent = inconsistent_edges(mesh.cells)
    record = {
        "nodes": len(xyz),
        "cells": len(mesh.cells),
        "non_quad_count": len(nonquad),
        "non_quad_cells": nonquad,
        "forbidden_non_quad_cells": forbidden,
        "inverted_count": len(inverted),
        "inverted_cells": inverted,
        "degenerate_count": len(degenerate),
        "degenerate_cells": degenerate,
        "non_manifold_count": len(nonmanifold),
        "non_manifold_edges": nonmanifold,
        "free_edge_count": len(free),
        "free_edges_off_y0_count": len(off),
        "free_edges_off_y0": off,
        "source_distance_max_m": float(np.max(distances, initial=0)),
        "source_distance_tolerance_m": distance_tolerance,
        "symmetry_tolerance_m": symmetry_tolerance,
        "area_tolerance_m2": area_tolerance,
        "inconsistent_shared_edge_directions": len(inconsistent),
        "inconsistent_shared_edges": inconsistent,
    }
    record["accepted"] = (
        not (forbidden or inverted or degenerate or nonmanifold or off or inconsistent)
        and record["source_distance_max_m"] <= distance_tolerance
    )
    return record


def coincident_nodes(xyz, tolerance=1e-9):
    """Return the pairs of distinct nodes closer than ``tolerance``: a junction not node to node."""
    from scipy.spatial import cKDTree

    return sorted(cKDTree(np.asarray(xyz)).query_pairs(tolerance))


def write_record(path, record):
    """Write a check record as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _textio.write_text(path, json.dumps(record, indent=2) + "\n")


def _quality(p):
    """Twice the area over the squared longest edge: small for a sliver triangle."""
    edges = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    return float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / np.max(edges) ** 2)


def _flip_candidate(mesh, cells, surfaces, edge_cells, i, *, threshold):
    c = cells[i]
    if len(c) != 3 or _quality(mesh.xyz[list(c)]) >= threshold:
        return None
    lengths = [np.linalg.norm(mesh.xyz[c[(k + 1) % 3]] - mesh.xyz[c[k]]) for k in range(3)]
    k = int(np.argmax(lengths))
    p, q, r = c[k], c[(k + 1) % 3], c[(k + 2) % 3]
    others = [j for j in edge_cells.get(tuple(sorted((p, q))), []) if j != i]
    if len(others) != 1:
        return None
    j = others[0]
    n = cells[j]
    if len(n) != 3 or surfaces[j] != surfaces[i]:
        return None
    d = next(x for x in n if x not in (p, q))
    new = [(p, d, r), (d, q, r)]
    before = min(_quality(mesh.xyz[list(c)]), _quality(mesh.xyz[list(n)]))
    after = min(_quality(mesh.xyz[list(t)]) for t in new)
    return None if after <= before * 1.05 else (j, new)


def flip_slivers(mesh, threshold=0.15, passes=400):
    """Flip the longest edge of sliver triangles, with a same-face triangle across it.

    The new triangles keep the winding of the sliver; a flip is taken only when it raises the
    worst quality. Returns ``(mesh, number of flips)``.
    """
    cells = list(mesh.cells)
    surfaces = list(mesh.surfaces)
    components = list(mesh.components)
    flips = 0
    for _ in range(passes):
        edge_cells = defaultdict(list)
        for i, c in enumerate(cells):
            for a, b in zip(c, c[1:] + c[:1], strict=True):
                edge_cells[tuple(sorted((a, b)))].append(i)
        changed = False
        for i in range(len(cells)):
            found = _flip_candidate(mesh, cells, surfaces, edge_cells, i, threshold=threshold)
            if found is not None:
                cells[i], cells[found[0]] = found[1]
                changed = True
                flips += 1
                break
        if not changed:
            break
    return replace(mesh, cells=cells, surfaces=surfaces, components=components), flips


def repair_inverted(mesh, normals, inverted, area_tolerance=1e-14):
    """Flip an edge of each inverted triangle when the same-face neighbour gives two sound ones.

    The mesh must be consistently oriented and ``normals`` oriented with it. Cell indices are
    preserved. Returns ``(mesh, flips)``.

    Parameters
    ----------
    mesh : Mesh
        The consistently oriented mesh.
    normals : array_like
        Oriented source normals, one per cell.
    inverted : sequence of int
        Indices of the inverted cells.
    area_tolerance : float, optional
        Smallest signed area of a sound triangle, square metres; default 1e-14.

    Returns
    -------
    tuple
        ``(mesh, flips)``: the repaired mesh and the number of edges flipped.
    """
    cells = list(mesh.cells)
    flips = 0

    def sound(t, normal):
        p = mesh.xyz[list(t)]
        cross = np.cross(np.roll(p, -1, axis=0) - p, np.roll(p, 1, axis=0) - p)
        return bool(np.all(cross @ normal > area_tolerance))

    for i in inverted:
        c = cells[i]
        if len(c) != 3:
            continue
        for k in range(3):
            p, q, r = c[k], c[(k + 1) % 3], c[(k + 2) % 3]
            neighbour = [
                j
                for j, o in enumerate(cells)
                if j != i
                and len(o) == 3
                and mesh.surfaces[j] == mesh.surfaces[i]
                and (q, p) in list(zip(o, o[1:] + o[:1], strict=True))
            ]
            if not neighbour:
                continue
            j = neighbour[0]
            d = next(x for x in cells[j] if x not in (p, q))
            new = [(p, d, r), (d, q, r)]
            if sound(new[0], normals[i]) and sound(new[1], normals[j]):
                cells[i], cells[j] = new
                flips += 1
                break
    return replace(mesh, cells=cells), flips


def paired_triangles(mesh):
    """Split every quadrilateral into the two reciprocal triangles a solver mesh stores.

    Returns ``(triangles, partners, slots, holders)``: a quadrilateral ``(a, b, c, d)`` becomes
    ``(a, b, c)`` and ``(a, c, d)`` partnered with shared-diagonal slots 3 and 1; a triangle has
    partner 0 and slot 0. Partners are one-based.
    """
    triangles, partners, slots, holders = [], [], [], []
    for cell, holder in zip(mesh.cells, mesh.components, strict=True):
        if len(cell) == 4:
            a, b, c, d = cell
            index = len(triangles)
            triangles.extend([(a, b, c), (a, c, d)])
            partners.extend([index + 2, index + 1])
            slots.extend([3, 1])
            holders.extend([holder, holder])
        elif len(cell) == 3:
            triangles.append(cell)
            partners.append(0)
            slots.append(0)
            holders.append(holder)
        else:
            raise InputValueError("mesh: triangles and quadrilaterals only")
    return np.asarray(triangles), partners, slots, holders


def write_obj(path, mesh):
    """Write the mesh as OBJ, one group per component, polygon topology preserved.

    Parameters
    ----------
    path : str or Path
        File to write.
    mesh : Mesh
        The mesh to write.

    Raises
    ------
    InputValueError
        If a component name is empty or contains whitespace.
    """
    lines = ["# polygon mesh; one group per component"]
    lines += ["v " + " ".join(format(float(x), ".17g") for x in p) for p in mesh.xyz]
    for name in dict.fromkeys(mesh.components):
        if not name or any(c.isspace() for c in name):
            raise InputValueError("component: OBJ names must be nonempty and contain no whitespace")
        lines.append("g " + name)
        lines += [
            "f " + " ".join(str(n + 1) for n in c)
            for c, holder in zip(mesh.cells, mesh.components, strict=True)
            if holder == name
        ]
    _textio.write_text(path, "\n".join(lines) + "\n", encoding="ascii")


def read_obj(path):
    """Read an OBJ written by :func:`write_obj` back into a :class:`Mesh`."""
    xyz, cells, components, holder = [], [], [], None
    for line in Path(path).read_text(encoding="ascii").splitlines():
        items = line.split()
        if not items or items[0].startswith("#"):
            continue
        if items[0] == "v":
            xyz.append(list(map(float, items[1:])))
        elif items[0] == "g":
            holder = items[1]
        elif items[0] == "f":
            if holder is None:
                raise InputValueError("OBJ: a face has no component group")
            cells.append(tuple(int(n) - 1 for n in items[1:]))
            components.append(holder)
    return Mesh(np.asarray(xyz), cells, components, [0] * len(cells), np.arange(len(xyz)))
