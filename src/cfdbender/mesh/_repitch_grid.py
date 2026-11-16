"""Recover source hub rings and rebuild their interior with the spinner's Winslow method.

The source connectivity is retained. Transport carries the root's local cell frame through
the available rings; surface projection and safeguarded smoothing retain the hub interior.
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from cfdbender._errors import InputValueError
from cfdbender.mesh._repitch import HubSurface, RingChart
from cfdbender.mesh._sector_winslow import winslow
from cfdbender.mesh._wheel import rotate

Array = NDArray[np.float64]
Edge = tuple[int, int]


def edges(face: Sequence[int]) -> list[Edge]:
    """Return the unordered sides of a polygon in its source order."""
    return [(min(a, b), max(a, b)) for a, b in zip(face, [*face[1:], face[0]], strict=True)]


def incidence(faces: list[list[int]]) -> dict[Edge, list[int]]:
    """Index each edge by its incident faces."""
    found: dict[Edge, list[int]] = defaultdict(list)
    for i, face in enumerate(faces):
        for edge in edges(face):
            found[edge].append(i)
    return dict(found)


def root_loop(blade: list[list[int]], hub: list[list[int]], name: str) -> list[int]:
    """Find the single closed boundary shared by a blade and the named hubs."""
    be, he = incidence(blade), incidence(hub)
    shared = [e for e, own in be.items() if len(own) == 1 and len(he.get(e, [])) == 1]
    links: dict[int, list[int]] = defaultdict(list)
    for a, b in shared:
        links[a].append(b)
        links[b].append(a)
    if not links or any(len(v) != 2 for v in links.values()):
        raise InputValueError(f"blade {name!r}: no closed manifold root loop on repitch.hub")
    loop = [min(links)]
    previous = -1
    while True:
        nxt = next(v for v in links[loop[-1]] if v != previous)
        if nxt == loop[0]:
            break
        if nxt in loop:
            raise InputValueError(f"blade {name!r}: root loop repeats a node")
        previous = loop[-1]
        loop.append(nxt)
    if len(loop) != len(links):
        raise InputValueError(f"blade {name!r}: multiple root loops; select one blade family")
    return loop


def rings(root: list[int], faces: list[list[int]]) -> NDArray[np.int64]:
    """Walk complete quad rings outward, stopping at a boundary or a topology transition."""
    by_edge = incidence(faces)
    used: set[int] = set()
    rows = [root]
    for _ in range(len(faces)):
        row = rows[-1]
        following: dict[int, int] = {}
        layer: set[int] = set()
        for a, b in zip(row, row[1:] + row[:1], strict=True):
            candidates = [i for i in by_edge.get((min(a, b), max(a, b)), []) if i not in used]
            if len(candidates) != 1 or len(faces[candidates[0]]) != 4:
                return np.array(rows)
            i = candidates[0]
            face = faces[i]
            layer.add(i)
            for v, other in ((a, b), (b, a)):
                neighbours = [u for e in edges(face) if v in e for u in e if u != v and u != other]
                if len(neighbours) != 1:
                    return np.array(rows)
                u = neighbours[0]
                if v in following and following[v] != u:
                    return np.array(rows)
                following[v] = u
        new = [following[v] for v in row]
        if len(layer) != len(row) or len(set(new)) != len(row):
            return np.array(rows)
        if set(new) & {v for r in rows for v in r}:
            return np.array(rows)
        rows.append(new)
        used.update(layer)
    return np.array(rows)


def rebuild(
    old: Array,
    new: Array,
    ids: NDArray[np.int64],
    motion: tuple[Array, Array, float],
    surface: HubSurface,
    *,
    frame: tuple[Array, Array] | None,
) -> None:
    """Transport all available rings, then backtrack a surface-chart Winslow candidate."""
    if len(ids) < 3:
        raise InputValueError(
            "hub has fewer than two complete quad rings; supply an O-grid hub, or state "
            "[repitch] hub_method = 'remesh' to remesh the hub band around each turned root. "
            "A mesh whose trailing edge was collapsed (collapse-te) has a closure triangle on "
            "the hub: repitch first, then run collapse-te on the repitched mesh"
        )
    if np.array_equal(old[ids[0]], new[ids[0]]):
        return
    before = old[ids]
    origin, axis, angle = motion
    rotated = rotate(before.reshape(-1, 3), axis, origin, angle).reshape(before.shape)
    # Zero slope at the root carries its local cell frame with the rigid blade. Every
    # available ring participates; only the outer loop has zero displacement.
    weight = 1 - np.linspace(0, 1, len(ids))[:, None, None] ** 2
    trial = before + weight * (rotated - before)
    trial[1:-1], _ = surface.nearest(trial[1:-1])
    trial[0], trial[-1] = new[ids[0]], before[-1]
    chart = RingChart(before, frame)
    uv = chart.coordinates(trial)
    smoothed = winslow(uv)
    mapping = np.full(len(old), -1, dtype=np.int64)
    mapping[ids.ravel()] = np.arange(ids.size)
    triangles = mapping[surface.mesh.faces]
    triangles = triangles[np.all(triangles >= 0, axis=1)]
    faces = [
        [
            k * ids.shape[1] + j,
            k * ids.shape[1] + (j + 1) % ids.shape[1],
            (k + 1) * ids.shape[1] + (j + 1) % ids.shape[1],
            (k + 1) * ids.shape[1] + j,
        ]
        for k in range(len(ids) - 1)
        for j in range(ids.shape[1])
    ]
    best = quality(trial.reshape(-1, 3), faces)["min_scaled_jacobian"]
    for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        candidate = trial.copy()
        candidate[1:-1] = chart.lift((uv + step * (smoothed - uv))[1:-1], before, triangles)
        score = quality(candidate.reshape(-1, 3), faces)["min_scaled_jacobian"]
        if score > best:
            trial, best = candidate, score
    new[ids[1:-1]] = trial[1:-1]


def quality(points: Array, faces: list[list[int]]) -> dict[str, float]:
    """Return corner scaled-Jacobian extrema, aspect extrema and folded-cell count."""
    values, aspects, folded = [], [], 0
    for face in faces:
        q = points[face]
        side = np.roll(q, -1, axis=0) - q
        lengths = np.linalg.norm(side, axis=1)
        normals = np.cross(side, -np.roll(side, 1, axis=0))
        normal = normals.sum(axis=0)
        norm = np.linalg.norm(normal)
        denom = lengths * np.roll(lengths, 1) * norm
        jac = np.divide(normals @ normal, denom, out=np.zeros(len(face)), where=denom > 0)
        values.extend(jac.tolist())
        aspects.append(float(lengths.max() / max(float(lengths.min()), 1e-300)))
        folded += int(np.any(jac <= 0))
    return {
        "min_scaled_jacobian": min(values),
        "max_scaled_jacobian": max(values),
        "min_aspect": min(aspects),
        "max_aspect": max(aspects),
        "folded_cells": folded,
    }
