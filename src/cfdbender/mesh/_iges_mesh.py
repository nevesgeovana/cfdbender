"""The gmsh half of ``geometry iges-grid``: join the faces, set the laws, mesh, read the elements.

The faces of every component are healed and fragmented together, so an edge two faces share
becomes one curve and its nodes are shared. Four-sided patches of a structured component get
transfinite curves with the laws of the grid control file and are recombined into quads; every
other patch is meshed quad-dominant (MeshAdapt, Blossom recombination) at its component's size,
graded from any curve shorter than it. The opposite sides of a four-sided patch must take one
count, so the curves that must agree are grouped and each group takes the count of the first
component that asks; every other request in the group is recorded as overridden.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cfdbender._errors import GeometryWarning, warn
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.mesh._cad import Face
from cfdbender.mesh._gmsh import GmshKernel, NamedModel
from cfdbender.mesh._iges_grid_config import (
    DEFAULT_RATIO,
    TE_ROLES,
    ComponentGrid,
    Direction,
    FaceLayout,
    GridConfig,
    Layout,
)

__all__ = [
    "JUNCTION_SAMPLES",
    "Meshed",
    "assign_layouts",
    "balance_counts",
    "cut_layout",
    "drop_unbalanced",
    "junction_checks",
    "junction_warnings",
    "mesh_model",
    "structured_blocks",
]

#: gmsh's mesh element type numbers of the linear triangle and quadrangle.
_TRIANGLE, _QUAD = 2, 3
#: gmsh 2D algorithm 1, MeshAdapt, and recombination algorithm 1, Blossom. MeshAdapt keeps the
#: component's size on a face whose parametrization is stretched and meshes a narrow face
#: without slivers, where Frontal-Delaunay for quads (8) left cells many sizes long across both
#: in the release validation assembly (200 sliver faces with 8, 29 with MeshAdapt).
_MESH_ADAPT, _BLOSSOM = 1, 1
#: How fast the size grows with the distance from a short curve: a cell about a third larger
#: than its neighbor nearer the curve.
_GROWTH = 0.3
#: The least quality of a quad Blossom may make: 1 less the largest departure of a corner from
#: 90 degrees over 90. 6/90 refuses a quad with a corner over 174 or under 6 degrees, which stays
#: two triangles, so recombination makes no sliver (the record's limit is 5 degrees).
_RECOMBINE_QUALITY = 6.0 / 90.0


@dataclass
class Meshed:
    """The mesh of every component: nodes, faces per component and what the run recorded."""

    nodes: NDArray[np.float64]
    faces: dict[str, list[list[int]]]
    jacobian: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    #: The mesh edges on true boundaries of the body, as sorted node index pairs.
    boundary: set[tuple[int, int]] = field(default_factory=set)
    record: dict[str, Any] = field(default_factory=dict)
    #: The mesh edges on the rims of each component's trailing-edge face (a layout role), as
    #: sorted node index pairs; empty for a component with no such face.
    te_edges: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    #: The edges of each component's other trailing-edge rim, which is not marked.
    te_dropped: dict[str, int] = field(default_factory=dict)


class _Groups:
    """Union-find over curve tags: the curves that must take one count."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, a: int) -> int:
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


@dataclass
class _Side:
    """One pair of opposite curves of a four-sided patch, as a component asks for it."""

    component: str
    curves: tuple[int, int]
    direction: str
    count: int
    law: str
    ratio: float | None
    axis: NDArray[np.float64]


def _heal(gmsh: Any, model: NamedModel, tolerance: float) -> str:
    """Fix degenerate edges and sew the faces; small edges are left as they are.

    Fixing small edges took the zero-length apex edge off the two half cones of a
    nacelle nose, and the join after it gave both halves, and both half rings behind them, one
    half circle of the four: the mesh of the other half folded across the nose and four faces met
    along each of its edges.
    """
    tags = [(2, t) for t in model.faces]
    out = gmsh.model.occ.healShapes(tags, tolerance, True, False, False, True, False)
    gmsh.model.occ.synchronize()
    healed = [t for d, t in out if d == 2]
    if len(healed) == len(model.faces):
        # One healed face per face, in the order given. gmsh numbers them again from the lowest
        # free tag, so a model that lost its first faces (a selection of components) gets new
        # tags: the model follows them and each face keeps its component.
        model.faces[:] = healed
        return "healed (degenerate edges fixed, faces sewn)"
    return "healing changed the faces and was not kept"


def _by_tag(model: NamedModel, roles: list[str] | None) -> dict[int, str] | None:
    """Return the layout role of each face of the model by tag, or ``None`` when there is none.

    The roles follow the faces' order, which healing keeps when it keeps one face per face; a
    healing that changed the faces leaves them unmatched and no role is used.
    """
    if not roles or len(roles) != len(model.faces):
        return None
    return {tag: role for tag, role in zip(model.faces, roles, strict=True) if role}


def _crossings(
    gmsh: Any, face_of: dict[int, str], warnings: list[str]
) -> dict[tuple[str, ...], int]:
    """Count and warn the curves more than two faces meet along, by the components that meet."""
    crossings: dict[tuple[str, ...], int] = {}
    for _, curve in gmsh.model.getEntities(1):
        up, _ = gmsh.model.getAdjacencies(1, curve)
        if len(up) > 2:
            names = tuple(sorted({face_of.get(int(t), "?") for t in up}))
            crossings[names] = crossings.get(names, 0) + 1
    for names, count in sorted(crossings.items()):
        warnings.append(
            f"more than two faces meet along {count} curve(s) of {' and '.join(names)}: a face "
            "crosses another without a trimmed junction; after the mesh the faces that lie "
            "inside the assembly are removed (remove_inner_faces in [defaults])"
        )
    return crossings


def _join(gmsh: Any, faces: list[int]) -> list[list[tuple[int, int]]]:
    """Fragment the faces against each other; return the pieces of each face, in their order.

    A single face has nothing to be joined to, and gmsh's fragment of one shape returns no map
    at all: it is kept whole, its own only piece.
    """
    if len(faces) < 2:
        return [[(2, t)] for t in faces]
    _, mapping = gmsh.model.occ.fragment([(2, t) for t in faces], [])
    gmsh.model.occ.synchronize()
    return [list(pieces) for pieces in mapping]


def _fragment(
    gmsh: Any,
    by_name: dict[str, list[int]],
    order: list[str],
    tolerance: float,
    warnings: list[str],
    *,
    roles: dict[int, str] | None = None,
) -> tuple[dict[str, list[int]], dict[str, Any], dict[int, str]]:
    """Fragment every face against every other; return the faces of each component after.

    ``roles`` gives the layout role of some input faces, by tag; the third result gives the role
    of every face left, each piece taking the role of the face it came from.
    """
    inputs = [(name, t) for name in order for t in by_name[name]]
    gmsh.option.setNumber("Geometry.ToleranceBoolean", tolerance)
    mapping = _join(gmsh, [t for _, t in inputs])
    face_of: dict[int, str] = {}
    piece_role: dict[int, str] = {}
    overlaps: dict[tuple[str, str], int] = {}
    for (name, source), pieces in zip(inputs, mapping, strict=True):
        for dim, tag in pieces:
            if dim != 2:
                continue
            if roles and source in roles and tag not in face_of:
                piece_role[tag] = roles[source]
            if tag in face_of and face_of[tag] != name:
                key = (face_of[tag], name)
                overlaps[key] = overlaps.get(key, 0) + 1
                continue
            face_of.setdefault(tag, name)
    for (first, second), count in overlaps.items():
        warnings.append(
            f"{count} face(s) of {second} lie on faces of {first} and were given to {first}"
        )
    faces: dict[str, list[int]] = {name: [] for name in order}
    for tag, name in face_of.items():
        faces[name].append(tag)
    for name in faces:
        faces[name].sort()
    crossings = _crossings(gmsh, face_of, warnings)
    record = {
        "faces_before": len(inputs),
        "faces_after": len(face_of),
        "split": sum(1 for pieces in mapping if sum(1 for d, _ in pieces if d == 2) > 1),
        "crossings": [{"components": list(k), "curves": v} for k, v in sorted(crossings.items())],
    }
    return faces, record, piece_role


def _ends(gmsh: Any, curve: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the points at a curve's first and last parameter, the order its mesh runs in.

    The order ``getBoundary`` lists a curve's end points in need not be that one: after the join
    a curve can keep another face's orientation, and a law clustered by the listed order would
    then cluster toward the wrong end of that curve only.
    """
    lo, hi = gmsh.model.getParametrizationBounds(1, curve)
    start = np.asarray(gmsh.model.getValue(1, curve, [lo[0]]), dtype=float)
    end = np.asarray(gmsh.model.getValue(1, curve, [hi[0]]), dtype=float)
    return start, end


def _span_axis(gmsh: Any, faces: list[int], grid: ComponentGrid) -> NDArray[np.float64]:
    if grid.span_axis is not None:
        return np.asarray(grid.span_axis, dtype=float)
    # x is the streamwise axis, along which no lifting surface spans: the span axis is y or z,
    # whichever the component extends along more (y for a wing or a pylon, z for a fin).
    boxes = np.array([gmsh.model.getBoundingBox(2, t) for t in faces], dtype=float)
    extent = boxes[:, 3:].max(axis=0) - boxes[:, :3].min(axis=0)
    axis = np.zeros(3)
    axis[1 if extent[1] >= extent[2] else 2] = 1.0
    return axis


#: A patch is four-sided only when each of its four corners is below this angle. A curve that
#: meets the next nearly straight on (a side the join cut in two, a three-cornered patch whose
#: degenerate edge left) is a point on a side, not a corner, and the transfinite map flattens the
#: cell there; 160 degrees keeps every real corner of a wing, tail or body panel and leaves a
#: margin above them.
CORNER_LIMIT_DEG = 160.0


def _inward(gmsh: Any, curve: int) -> dict[int, NDArray[np.float64]]:
    """Return the unit tangent of a curve at each of its end points, pointing into the curve."""
    ends = [abs(t) for _, t in gmsh.model.getBoundary([(1, curve)], oriented=True)]
    where = {p: np.asarray(gmsh.model.getValue(0, p, []), dtype=float) for p in ends}
    lo, hi = gmsh.model.getParametrizationBounds(1, curve)
    out: dict[int, NDArray[np.float64]] = {}
    for at, sign in ((lo[0], 1.0), (hi[0], -1.0)):
        # The end a parameter bound is at, by position: the boundary's order need not follow it.
        here = np.asarray(gmsh.model.getValue(1, curve, [at]), dtype=float)
        point = min(where, key=lambda p: float(np.linalg.norm(where[p] - here)))
        d = sign * np.asarray(gmsh.model.getDerivative(1, curve, [at]), dtype=float)
        n = float(np.linalg.norm(d))
        if n > 0:
            out[point] = d / n
    return out


def corner_angles(gmsh: Any, loop: list[int]) -> list[float]:
    """Return the angle at each point where two consecutive curves of a loop meet, in degrees."""
    tangents = [_inward(gmsh, c) for c in loop]
    angles = []
    for i, here in enumerate(tangents):
        after = tangents[(i + 1) % len(loop)]
        shared = set(here) & set(after)
        if not shared:
            angles.append(180.0)
            continue
        point = shared.pop()
        cosine = float(np.clip(here[point] @ after[point], -1.0, 1.0))
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def _four_sides(gmsh: Any, face: int) -> list[list[int]] | None:
    """Return the four sides of a face as lists of curves, or ``None`` when it has no four sides.

    A face of four curves has a corner at each of them. A face of more curves is four-sided when
    exactly four of its points are corners and the others are points on a side (a side the join
    cut in two, as the vertex of the one-face tip cap cuts a skin edge): the side is then a
    composite of the curves between two corners.
    """
    loop = [abs(t) for _, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True)]
    if len(loop) < 4 or len(set(loop)) != len(loop):
        return None
    angles = corner_angles(gmsh, loop)
    corners = [i for i, a in enumerate(angles) if a < CORNER_LIMIT_DEG]
    if len(corners) != 4:
        return None
    sides = []
    for k in range(4):
        start, stop = corners[k - 1] + 1, corners[k] + 1
        sides.append(loop[start:stop] if stop > start else loop[start:] + loop[:stop])
    return sides


def _corner_points(gmsh: Any, face: int, sides: list[list[int]]) -> list[int]:
    """Return the point tag of each corner of a four-sided face: the start of each side."""
    signed = [t for _, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True)]
    first = {abs(t): t for t in signed}
    corners = []
    for side in sides:
        curve = side[0]
        _, down = gmsh.model.getAdjacencies(1, curve)
        corners.append(int(down[0] if first[curve] > 0 else down[1]))
    return corners


def _shares(counts: int, lengths: list[float]) -> list[int]:
    """Split a cell count over the curves of a composite side in proportion to their lengths."""
    total = sum(lengths)
    parts = [max(1, round(counts * length / total)) for length in lengths]
    while sum(parts) > counts and max(parts) > 1:
        parts[parts.index(max(parts))] -= 1
    while sum(parts) < counts:
        parts[max(range(len(parts)), key=lambda i: lengths[i] / parts[i])] += 1
    return parts


def _bump(
    law: Direction, count: int, grid: ComponentGrid, direction: str, warnings: list[str]
) -> float:
    """Return the end-to-middle ratio of a cosine law: given, else the clusters, else its own."""
    if law.ratio is not None:
        return law.ratio
    if direction == "chord":
        clusters = [c for c in (grid.le_cluster, grid.te_cluster) if c is not None]
        if clusters:
            if len(set(clusters)) > 1:
                warnings.append(
                    f"{grid.name}: the cosine law is symmetric, so the smaller of le_cluster and "
                    f"te_cluster, {min(clusters):g}, is used at both edges"
                )
            return min(clusters)
    return math.tan(math.pi / (2 * max(count, 1)))


def _span_pair(gmsh: Any, pairs: list[tuple[int, int]], axis: NDArray[np.float64]) -> int:
    """Return the index of the pair of opposite sides that runs closer to the span axis."""
    align = []
    for pair in pairs:
        total = 0.0
        for curve in pair:
            a, b = _ends(gmsh, curve)
            d = b - a
            n = float(np.linalg.norm(d))
            total += abs(float(d @ axis)) / n if n > 0 else 0.0
        align.append(total)
    return int(np.argmax(align))


def _side_law(gmsh, grid, direction, pair_of, *, size, strips, warnings):
    """Choose the requested spacing law, including the one-cell thin-strip rule."""
    spec = getattr(grid, direction) or Direction(count=None)
    length = max(sum(gmsh.model.occ.getMass(1, c) for c in side) for side in pair_of)
    if length < size:
        # A side shorter than the component's size is a strip, such as the base of a
        # blunt trailing edge: one cell across it, whatever the law of its direction.
        if spec.count is not None and spec.count != 1:
            key = (direction, spec.count)
            strips[key] = strips.get(key, 0) + 1
        spec = Direction(count=1)
    count = spec.count or max(1, round(length / size))
    ratio = spec.ratio
    if spec.law == "cosine":
        ratio = _bump(spec, count, grid, direction, warnings)
    elif spec.law in DEFAULT_RATIO and ratio is None:
        ratio = DEFAULT_RATIO[spec.law]
    return spec, count, ratio


def _composite_sides(gmsh, name, pair_of, law, *, pair, axis, direction, face, warnings):
    """Distribute opposite composite sides while preserving each curve request."""
    spec, count, ratio = law
    sides = []
    if all(len(side) == 1 for side in pair_of):
        return [_Side(name, pair, direction, count, spec.law, ratio, axis)]
    count = max(count, *(len(side) for side in pair_of))
    if spec.law != "uniform" and count > 1:
        warnings.append(
            f"{name}: face {face} has a {direction} side of several curves, whose "
            f"{count} cells are shared over them by length, uniform on each"
        )
    for side in pair_of:
        if len(side) == 1:
            sides.append(_Side(name, (side[0], side[0]), direction, count, spec.law, ratio, axis))
            continue
        shares = _shares(count, [float(gmsh.model.occ.getMass(1, c)) for c in side])
        for curve, share in zip(side, shares, strict=True):
            sides.append(_Side(name, (curve, curve), direction, share, "uniform", None, axis))
    return sides


def _sides(
    gmsh: Any,
    name: str,
    grid: ComponentGrid,
    faces: list[int],
    *,
    size: float,
    warnings: list[str],
    corners: dict[int, list[int]] | None = None,
) -> tuple[list[_Side], list[int]]:
    """Return the two sides of every four-sided patch of a component, and those patches.

    A patch with a composite side (a side of several curves) is given its four corner points in
    ``corners``, by face, and the cells of that side are shared over its curves by length.
    """
    axis = _span_axis(gmsh, faces, grid)
    sides: list[_Side] = []
    patches: list[int] = []
    strips: dict[tuple[str, int], int] = {}

    def long(side: list[int]) -> int:
        return max(side, key=lambda c: gmsh.model.occ.getMass(1, c))

    for face in faces:
        loop = _four_sides(gmsh, face)
        if loop is None or not grid.structured:
            continue
        pairs_of = [(loop[0], loop[2]), (loop[1], loop[3])]
        pairs = [(long(a), long(b)) for a, b in pairs_of]
        span_index = _span_pair(gmsh, pairs, axis)
        patches.append(face)
        if corners is not None and any(len(side) > 1 for side in loop):
            corners[face] = _corner_points(gmsh, face, loop)
        for index, pair_of in enumerate(pairs_of):
            pair = pairs[index]
            direction = "span" if index == span_index else "chord"
            law = _side_law(
                gmsh, grid, direction, pair_of, size=size, strips=strips, warnings=warnings
            )
            sides += _composite_sides(
                gmsh,
                name,
                pair_of,
                law,
                pair=pair,
                axis=axis,
                direction=direction,
                face=face,
                warnings=warnings,
            )
    for (direction, asked), patches_cut in sorted(strips.items()):
        warnings.append(
            f"{name}: {patches_cut} patch(es) with a {direction} side shorter than the size "
            f"{size:g} take 1 cell across it, not the {direction} count of {asked}"
        )
    return sides, patches


def _transfinite(
    gmsh: Any, sides: list[_Side], loops: dict[int, list[list[int]]], warnings: list[str]
) -> tuple[list[dict[str, Any]], dict[int, tuple[int, int]], set[int]]:
    """Give every group of curves one count and law, with equal opposite totals on each patch.

    The first request of a group sets its count. A patch with a composite side can then hold
    unequal totals on two opposite sides; :func:`balance_counts` raises the smaller side until
    they are equal, and a raise is warned. Return the overridden requests, the two totals of
    every patch that holds, and the patches that cannot hold equal totals.
    """
    groups = _Groups()
    for side in sides:
        groups.union(*side.curves)
    chosen: dict[int, _Side] = {}
    overrides: dict[tuple[str, str, int, int], int] = {}
    for side in sides:
        first = chosen.setdefault(groups.find(side.curves[0]), side)
        if first.count != side.count:
            key = (side.component, first.component, side.count, first.count)
            overrides[key] = overrides.get(key, 0) + 1
    counts = {root: side.count for root, side in chosen.items()}
    lengths = {
        c: float(gmsh.model.occ.getMass(1, c)) for loop in loops.values() for s in loop for c in s
    }
    owners: dict[int, set[str]] = {}
    for side in sides:
        owners.setdefault(groups.find(side.curves[0]), set()).add(side.component)
    pinned = frozenset(root for root, names in owners.items() if len(names) > 1)
    unbalanced = balance_counts(groups, counts, loops, lengths, pinned=pinned)
    _warn_raised(chosen, counts, warnings)
    done: set[int] = set()
    for side in sides:
        root = groups.find(side.curves[0])
        for curve in side.curves:
            if curve not in done:
                done.add(curve)
                _set_curve(gmsh, curve, replace(chosen[root], count=counts[root]))
    totals = {
        face: (_total(groups, counts, loop[0]), _total(groups, counts, loop[1]))
        for face, loop in loops.items()
        if face not in unbalanced
    }
    report = [
        {"component": a, "wanted": w, "used": u, "from": b, "pairs": n}
        for (a, b, w, u), n in sorted(overrides.items())
    ]
    return report, totals, unbalanced


#: The passes :func:`balance_counts` makes over the patches before it gives up on equal totals.
_BALANCE_PASSES = 1000


def _total(groups: _Groups, counts: dict[int, int], side: list[int]) -> int:
    return sum(counts[groups.find(c)] for c in side)


def balance_counts(
    groups: _Groups,
    counts: dict[int, int],
    loops: dict[int, list[list[int]]],
    lengths: dict[int, float],
    *,
    pinned: frozenset[int] = frozenset(),
) -> set[int]:
    """Raise group counts until the opposite sides of every patch hold equal totals.

    ``counts`` (by group root) is changed in place. A count is only ever raised, one cell at a
    time, on the side with the smaller total, on the curve of that side with the longest cells
    whose group does not run as often along the other side. A group in ``pinned`` (one that
    several components ask for) is raised only when no other curve of the side can be. Return
    the faces whose opposite totals are still unequal: no raise can fix them, or the passes ran
    out.
    """
    pairs = [(face, loop[k], loop[k + 2]) for face, loop in loops.items() for k in (0, 1)]
    for _ in range(_BALANCE_PASSES):
        moved = [_raise_smaller(groups, counts, lengths, (a, b), pinned) for _, a, b in pairs]
        if not any(moved):
            break
    return {f for f, a, b in pairs if _total(groups, counts, a) != _total(groups, counts, b)}


def _raise_smaller(
    groups: _Groups,
    counts: dict[int, int],
    lengths: dict[int, float],
    pair: tuple[list[int], list[int]],
    pinned: frozenset[int],
) -> bool:
    """Raise one group of the smaller of two opposite sides by one cell; False when none can."""
    a, b = pair
    ta, tb = _total(groups, counts, a), _total(groups, counts, b)
    if ta == tb:
        return False
    small, big = (a, b) if ta < tb else (b, a)
    gain: dict[int, int] = {}
    for curve in small:
        gain[groups.find(curve)] = gain.get(groups.find(curve), 0) + 1
    for curve in big:
        gain[groups.find(curve)] = gain.get(groups.find(curve), 0) - 1
    free = [c for c in small if gain[groups.find(c)] > 0]
    if not free:
        return False
    curve = max(
        free,
        key=lambda c: (groups.find(c) not in pinned, lengths[c] / counts[groups.find(c)]),
    )
    counts[groups.find(curve)] += 1
    return True


def _warn_raised(chosen: dict[int, _Side], counts: dict[int, int], warnings: list[str]) -> None:
    for root, side in sorted(chosen.items()):
        if counts[root] != side.count:
            warnings.append(
                f"{side.component}: a {side.direction} curve takes {counts[root]} cells, not "
                f"{side.count}, so that the opposite sides of its patch hold equal totals"
            )


def structured_blocks(
    gmsh: Any,
    patches: dict[str, list[int]],
    totals: dict[int, tuple[int, int]],
    warnings: list[str],
) -> dict[str, int]:
    """Count, by component, the planned blocks the mesh realized as structured grids.

    A block is counted when its face holds only quads and exactly the product of its two
    totals of them; any other planned block is warned by face and not counted.
    """
    counted: dict[str, int] = {}
    for name, four in patches.items():
        counted[name] = 0
        for face in four:
            types, tags, _ = gmsh.model.mesh.getElements(2, face)
            cells = sum(len(t) for t in tags)
            quads = sum(len(t) for k, t in zip(types, tags, strict=True) if k == _QUAD)
            a, b = totals[face]
            if quads == cells == a * b:
                counted[name] += 1
                continue
            warnings.append(
                f"{name}: face {face} was planned as a {a} x {b} block but its mesh has "
                f"{quads} quads in {cells} cells; it is not counted as transfinite"
            )
    return counted


def drop_unbalanced(
    patches: dict[str, list[int]],
    corners: dict[int, list[int]],
    unbalanced: set[int],
    warnings: list[str],
) -> None:
    """Mesh the patches without equal opposite totals quad-dominant, and say so by face."""
    for name, four in patches.items():
        for face in [f for f in four if f in unbalanced]:
            four.remove(face)
            corners.pop(face, None)
            warnings.append(
                f"{name}: face {face} cannot hold equal cell totals on its opposite sides; "
                "it is meshed quad-dominant and not counted as transfinite"
            )


def _set_curve(gmsh: Any, curve: int, side: _Side) -> None:
    nodes = side.count + 1
    if side.law == "uniform" or side.count == 1:
        gmsh.model.mesh.setTransfiniteCurve(curve, nodes, "Progression", 1.0)
        return
    if side.law == "cosine":
        gmsh.model.mesh.setTransfiniteCurve(curve, nodes, "Bump", float(side.ratio or 1.0))
        return
    a, b = _ends(gmsh, curve)
    if side.direction == "span":
        # Toward the tip: the end farther from the plane of symmetry, along the span axis.
        at_start = abs(float(a @ side.axis)) > abs(float(b @ side.axis))
    else:
        # Toward the downstream end: the end with the larger x.
        at_start = a[0] > b[0]
    ratio = float(side.ratio or 1.0)
    kind = "Progression" if side.law == "geometric" else "Beta"
    # A positive coefficient clusters at the curve's start, a negative one at its end.
    gmsh.model.mesh.setTransfiniteCurve(curve, nodes, kind, ratio if at_start else -ratio)


def _point_sizes(
    gmsh: Any, faces: dict[str, list[int]], order: list[str], sizes: dict[str, float]
) -> None:
    """Set the mesh size: its component's at every point, and graded about every short curve.

    A point takes the size of the first component in the grid file whose faces reach it. A curve
    shorter than the size at its ends, such as a piece the join cut off, gets one cell; a
    quad-dominant face meshed at the full size around it would close that cell with a sliver. So
    the size about such a curve is its length, growing by :data:`_GROWTH` of the distance from it
    until it is the full size again: the cells grow from the curve and only near it.
    """
    size: dict[int, float] = {}
    for name in order:
        points = gmsh.model.getBoundary(
            [(2, t) for t in faces[name]], combined=False, recursive=True
        )
        for dim, tag in points:
            if dim == 0:
                size.setdefault(abs(tag), sizes[name])
    for point, value in size.items():
        gmsh.model.mesh.setSize([(0, point)], value)
    fields = []
    for _, curve in gmsh.model.getEntities(1):
        ends = [abs(t) for d, t in gmsh.model.getBoundary([(1, curve)]) if d == 0]
        full = min((size[p] for p in ends if p in size), default=0.0)
        length = gmsh.model.occ.getMass(1, curve)
        if not 0 < length < full:
            continue
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", [curve])
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", length)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", full)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", (full - length) / _GROWTH)
        gmsh.model.mesh.field.setNumber(threshold, "StopAtDistMax", 1)
        fields.append(threshold)
    if fields:
        least = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(least, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(least)


def _set_surfaces(gmsh, faces, order, patches, corners):
    """Apply transfinite corner constraints and recombination to the planned surfaces."""
    for name in order:
        for face in patches.get(name, []):
            if face in corners:
                gmsh.model.mesh.setTransfiniteSurface(face, "Left", corners[face])
            else:
                gmsh.model.mesh.setTransfiniteSurface(face)
        for face in faces[name]:
            gmsh.model.mesh.setRecombine(2, face)


def mesh_model(
    gmsh: Any,
    model: NamedModel,
    config: GridConfig,
    order: list[str],
    sizes: dict[str, float],
    *,
    tolerance: float,
    reopen: Callable[[], NamedModel] | None = None,
    roles: list[str] | None = None,
) -> Meshed:
    """Join, mesh and read back every component of an imported model.

    With ``reopen`` (which clears gmsh and imports the files again) and no stated tolerance, a
    gap the files leave between faces that should meet is measured after the first join and,
    when :data:`_CLOSE_FACTOR` times the widest is above that join's tolerance, the files are
    imported and joined again at it (:func:`_near_miss_gaps`).

    ``roles`` is the layout role of each face of ``model``, in its order (an empty string for a
    face with none); ``reopen`` refreshes it in place when it imports the files again. A face
    with the trailing-edge role gives the component's trailing-edge rims in ``te_edges``.
    """
    warnings: list[str] = []
    healing = _heal(gmsh, model, tolerance)
    faces, fragment, piece_role = _fragment(
        gmsh, model.components(), order, tolerance, warnings, roles=_by_tag(model, roles)
    )
    closed: list[dict[str, Any]] = []
    if config.tolerance is None and reopen is not None:
        closed, joined = _near_miss_gaps(gmsh, faces, sizes)
        if joined > tolerance:
            first = {"tolerance": tolerance, **fragment}
            model, tolerance = reopen(), joined
            warnings.clear()
            healing = _heal(gmsh, model, tolerance)
            faces, fragment, piece_role = _fragment(
                gmsh, model.components(), order, tolerance, warnings, roles=_by_tag(model, roles)
            )
            fragment["first_join"] = first
            for entry in closed:
                warnings.append(
                    f"the files leave a gap of up to {entry['gap']:.4g} between faces of "
                    f"{' and '.join(entry['components'])} along {entry['curves']} curve(s); "
                    f"the faces were joined again at {joined:.4g}"
                )
        else:
            closed = []
    # Said now, before the mesh: on overlapping or crossing faces the mesher can take long, and
    # whoever waits should already know why. The record keeps them with the others.
    for message in warnings:
        warn(f"iges-grid: {message}", GeometryWarning)
    fragment["warned"] = len(warnings)
    sides: list[_Side] = []
    patches: dict[str, list[int]] = {}
    corners: dict[int, list[int]] = {}
    for name in order:
        grid = config.components.get(name, ComponentGrid(name=name))
        if not faces[name]:
            continue
        found, four = _sides(
            gmsh, name, grid, faces[name], size=sizes[name], warnings=warnings, corners=corners
        )
        sides += found
        patches[name] = four
    loops = {f: _four_sides(gmsh, f) or [] for four in patches.values() for f in four}
    overrides, totals, unbalanced = _transfinite(gmsh, sides, loops, warnings)
    drop_unbalanced(patches, corners, unbalanced, warnings)
    _point_sizes(gmsh, faces, order, sizes)
    _set_surfaces(gmsh, faces, order, patches, corners)
    gmsh.option.setNumber("Mesh.Algorithm", _MESH_ADAPT)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", _BLOSSOM)
    gmsh.option.setNumber("Mesh.RecombineMinimumQuality", _RECOMBINE_QUALITY)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max(sizes.values()))
    gmsh.model.mesh.generate(2)
    structured = structured_blocks(gmsh, patches, totals, warnings)
    meshed = _read(
        gmsh,
        faces,
        order,
        warnings,
        {
            "healing": healing,
            "fragment": fragment,
            "tolerance": tolerance,
            "closed_gaps": closed,
            "transfinite_patches": structured,
            "count_overrides": overrides,
        },
    )
    face_of = {f: n for n, fs in faces.items() for f in fs}
    meshed.boundary, meshed.record["unsewn_curves"] = _boundary_edges(gmsh, face_of, sizes)
    meshed.te_edges, meshed.te_dropped = _role_edges(gmsh, faces, order, piece_role)
    return meshed


#: A free curve that lies on another within this share of the smaller of their components'
#: sizes is a join the file missed: a grid of that size cannot show an opening so narrow, and a
#: wider one, such as a blunt trailing edge left open, is a boundary of the body.
NEAR_MISS_SHARE = 0.01
#: The second join that closes a near miss takes this multiple of the widest gap measured.
_CLOSE_FACTOR = 1.5


def lies_on(
    params: NDArray[np.float64],
    bounds: tuple[float, float],
    gaps: NDArray[np.float64],
    reach: float,
) -> bool:
    """Return whether sampled points lie on a curve: every gap within reach, on its own span.

    ``params`` are the parameters of the closest points on the curve and ``bounds`` its
    parameter range. A projection outside the range lands on the curve's extension, where the
    next piece of a curve the join cut continues: such points lie beside the curve, not on it.
    """
    lo, hi = bounds
    margin = 1e-9 * max(abs(hi - lo), 1.0)
    inside = bool(np.all(params >= lo - margin) and np.all(params <= hi + margin))
    return inside and bool(np.all(gaps <= reach))


def _near_misses(
    gmsh: Any, face_of: dict[int, str], sizes: dict[str, float]
) -> list[tuple[int, int, float]]:
    """Return each free curve that lies on another free curve, that curve and the widest gap.

    A free curve bounds one face. Five points of it are projected on each other free curve
    whose box comes within reach, the reach being :data:`NEAR_MISS_SHARE` of the smaller size
    of the two curves' components.
    """
    free: dict[int, str] = {}
    for _, curve in gmsh.model.getEntities(1):
        up, _ = gmsh.model.getAdjacencies(1, curve)
        if len(up) == 1:
            free[curve] = face_of.get(int(up[0]), "")
    samples: dict[int, NDArray[np.float64]] = {}
    boxes: dict[int, NDArray[np.float64]] = {}
    for curve in free:
        lo, hi = gmsh.model.getParametrizationBounds(1, curve)
        ts = np.linspace(lo[0], hi[0], 7)[1:-1]
        samples[curve] = np.asarray(gmsh.model.getValue(1, curve, ts), dtype=float)
        boxes[curve] = np.asarray(gmsh.model.getBoundingBox(1, curve), dtype=float)
    smallest = min(sizes.values()) if sizes else 0.0
    out: list[tuple[int, int, float]] = []
    for a, name_a in free.items():
        for b, name_b in free.items():
            if a == b:
                continue
            reach = NEAR_MISS_SHARE * min(sizes.get(name_a, smallest), sizes.get(name_b, smallest))
            box_a, box_b = boxes[a], boxes[b]
            if np.any(box_a[:3] > box_b[3:] + reach) or np.any(box_b[:3] > box_a[3:] + reach):
                continue
            closest, params = gmsh.model.getClosestPoint(1, b, samples[a])
            gaps = np.linalg.norm(
                np.asarray(closest).reshape(-1, 3) - samples[a].reshape(-1, 3), axis=1
            )
            lo, hi = gmsh.model.getParametrizationBounds(1, b)
            if lies_on(np.asarray(params, dtype=float), (lo[0], hi[0]), gaps, reach):
                out.append((a, b, float(gaps.max())))
                break
    return out


def _near_miss_gaps(
    gmsh: Any, faces: dict[str, list[int]], sizes: dict[str, float]
) -> tuple[list[dict[str, Any]], float]:
    """Return the near misses of a join by pair of components, and the tolerance that closes them.

    The first join's tolerance is a share of the assembly; a gap the files leave between two
    faces that should meet can be wider. Each such gap is measured (:func:`_near_misses`); the
    closing tolerance is :data:`_CLOSE_FACTOR` times the widest, 0 when there is none.
    """
    face_of = {f: n for n, fs in faces.items() for f in fs}
    misses = _near_misses(gmsh, face_of, sizes)
    pairs: dict[tuple[str, ...], dict[str, Any]] = {}
    for a, b, gap in misses:
        names = tuple(
            sorted({face_of.get(int(gmsh.model.getAdjacencies(1, c)[0][0]), "") for c in (a, b)})
        )
        entry = pairs.setdefault(names, {"components": list(names), "curves": 0, "gap": 0.0})
        entry["curves"] += 1
        entry["gap"] = max(entry["gap"], gap)
    widest = max((gap for *_, gap in misses), default=0.0)
    return [pairs[k] for k in sorted(pairs)], _CLOSE_FACTOR * widest


def _boundary_edges(
    gmsh: Any, face_of: dict[int, str], sizes: dict[str, float]
) -> tuple[set[tuple[int, int]], int]:
    """Return the mesh edges on true boundaries of the body, and the free curves left unsewn.

    A true boundary is a curve that bounds one face and lies on no other such curve: the open
    root of a wing, the open base of a blunt trailing edge. A free curve that lies on another
    (:func:`_near_misses`) is an edge the join missed, and its mesh edges are not boundaries.
    """
    tags, _, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    unsewn = {a for a, _, _ in _near_misses(gmsh, face_of, sizes)}
    free = [
        curve
        for _, curve in gmsh.model.getEntities(1)
        if len(gmsh.model.getAdjacencies(1, curve)[0]) == 1
    ]
    boundary: set[tuple[int, int]] = set()
    for curve in free:
        if curve in unsewn:
            continue
        types, _, enodes = gmsh.model.mesh.getElements(1, curve)
        for kind, nodes in zip(types, enodes, strict=True):
            if int(kind) != 1:
                continue
            for a, b in np.asarray(nodes, dtype=np.int64).reshape(-1, 2):
                i, j = index[int(a)], index[int(b)]
                boundary.add((min(i, j), max(i, j)))
    return boundary, len(unsewn)


def _read(
    gmsh: Any,
    faces: dict[str, list[int]],
    order: list[str],
    warnings: list[str],
    record: dict[str, Any],
) -> Meshed:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
    out: dict[str, list[list[int]]] = {}
    jacobian: dict[str, float] = {}
    for name in order:
        polys: list[list[int]] = []
        elements: list[int] = []
        empty = 0
        for face in faces[name]:
            types, etags, enodes = gmsh.model.mesh.getElements(2, face)
            held = 0
            for kind, et, en in zip(types, etags, enodes, strict=True):
                width = {_TRIANGLE: 3, _QUAD: 4}.get(int(kind))
                if width is None:
                    continue
                rows = np.asarray(en, dtype=np.int64).reshape(-1, width)
                for row in rows:
                    poly = [index[int(v)] for v in row]
                    poly = [v for i, v in enumerate(poly) if v != poly[i - 1]]
                    if len(set(poly)) >= 3:
                        polys.append(poly)
                        held += 1
                elements += [int(e) for e in et]
            empty += held == 0
        if empty:
            warnings.append(f"{name}: {empty} face(s) could not be meshed and are missing")
        out[name] = polys
        if elements:
            quality = gmsh.model.mesh.getElementQualities(elements, "minSJ")
            jacobian[name] = float(np.min(quality))
    return Meshed(nodes, out, jacobian, warnings=warnings, record=record)


#: The junction nodes whose distance to the original faces is measured, at most per pair.
JUNCTION_SAMPLES = 200


def junction_checks(
    verts: NDArray[np.float64],
    families: dict[str, list[list[int]]],
    distance: Callable[[str, NDArray[np.float64]], float],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Check each junction, the nodes two components share, against both original components.

    A node of the junction of ``a`` and ``b`` lies on the curve where the two surfaces cross,
    so it must lie on the original face set of each. The distance of at most
    :data:`JUNCTION_SAMPLES` nodes (evenly spaced in node order) to each component is
    measured with the geometry kernel; ``within_tolerance`` compares the larger one with the
    join tolerance. The nodes are counted whole.
    """
    names = [n for n in families if families[n]]
    nodes = {n: {v for f in families[n] for v in f} for n in names}
    out: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sorted(nodes[a] & nodes[b])
            if not common:
                continue
            step = math.ceil(len(common) / JUNCTION_SAMPLES)
            points = verts[common[::step]]
            gaps = {a: distance(a, points), b: distance(b, points)}
            out.append(
                {
                    "components": [a, b],
                    "nodes": len(common),
                    "checked": len(points),
                    "distance": gaps,
                    "tolerance": tolerance,
                    "within_tolerance": max(gaps.values()) <= tolerance,
                }
            )
    return out


def junction_warnings(junctions: list[dict[str, Any]]) -> list[str]:
    """Return one warning for each junction whose nodes lie past the join tolerance."""
    return [
        f"junction of {' and '.join(entry['components'])}: {entry['nodes']} node(s) lie "
        f"up to {max(entry['distance'].values()):.4g} from the original faces of the "
        f"components, past the join tolerance {entry['tolerance']:.4g}"
        for entry in junctions
        if not entry["within_tolerance"]
    ]


def _role_edges(
    gmsh: Any, faces: dict[str, list[int]], order: list[str], piece_role: dict[int, str]
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, int]]:
    """Return the mesh edges of each component's trailing edge from its layout roles.

    A rim is a curve of a face with a trailing-edge role (:data:`TE_ROLES`) that a skin face of
    the same component also bounds: the line along which the blunt base meets the upper or the
    lower skin (a cap's curve is the end of the base, not a rim). A blunt base has two rims and
    one is marked: the side whose curves are longer, the first by role name on a tie, as the
    geometric search keeps the longer of two opposite rims. The edges are sorted node index
    pairs of the mesh, as :func:`_boundary_edges`; the second result counts the edges of the
    rim not marked.
    """
    base_faces = {
        name: {t for t in faces[name] if piece_role.get(t) in TE_ROLES}
        for name in order
        if any(piece_role.get(t) in TE_ROLES for t in faces[name])
    }
    if not base_faces:
        return {}, {}
    tags, _, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    out: dict[str, set[tuple[int, int]]] = {}
    dropped: dict[str, int] = {}
    for name, base in base_faces.items():
        skin = {t for t in faces[name] if t not in base and piece_role.get(t) != "cap"}
        sides: dict[str, set[int]] = {}
        for face in base:
            for _, curve in gmsh.model.getBoundary(
                [(2, face)], combined=False, oriented=False, recursive=False
            ):
                up, _ = gmsh.model.getAdjacencies(1, curve)
                for tag in skin & {int(t) for t in up}:
                    sides.setdefault(piece_role.get(tag, ""), set()).add(curve)
        length = {
            side: sum(float(gmsh.model.occ.getMass(1, c)) for c in curves)
            for side, curves in sides.items()
        }
        longest = max(length.values(), default=0.0)
        kept = min((s for s, v in length.items() if v >= longest * (1.0 - _RIM_TIE)), default=None)
        edges = {side: _curve_edges(gmsh, curves, index) for side, curves in sides.items()}
        out[name] = edges[kept] if kept is not None else set()
        dropped[name] = sum(len(e) for s, e in edges.items() if s != kept)
    return out, dropped


def _curve_edges(gmsh: Any, curves: set[int], index: dict[int, int]) -> set[tuple[int, int]]:
    """Return the mesh edges of the curves as sorted node index pairs."""
    edges: set[tuple[int, int]] = set()
    for curve in curves:
        types, _, enodes = gmsh.model.mesh.getElements(1, curve)
        for kind, nodes in zip(types, enodes, strict=True):
            if int(kind) != 1:
                continue
            for a, b in np.asarray(nodes, dtype=np.int64).reshape(-1, 2):
                i, j = index[int(a)], index[int(b)]
                edges.add((min(i, j), max(i, j)))
    return edges


#: Two rims whose lengths differ by less than this share are of one length.
_RIM_TIE = 1e-9


#: A re-imported face's area matches the sidecar's within this share of it.
_AREA_MATCH = 1e-4


def assign_layouts(
    gmsh: Any, model: NamedModel, layouts: list[Layout | None], notes: list[str]
) -> tuple[dict[int, tuple[FaceLayout, int]], list[dict[str, Any]]]:
    """Match the faces of each source file to the faces of its layout sidecar, by order.

    ``layouts`` holds one entry per source file, in the files' order. A file's faces take its
    sidecar's faces in order when the count, the component of every face and the area of every
    face agree; any disagreement leaves the whole file with no layout and a line in ``notes``
    (the file is a foreign one). Return the layout face of each matched face by tag with the
    index of its sidecar in the summaries, and one summary per sidecar read.
    """
    assigned: dict[int, tuple[FaceLayout, int]] = {}
    summaries: list[dict[str, Any]] = []
    scale = float(METRES_PER_UNIT[model.unit]) ** 2
    offset = 0
    for source, layout in zip(model.sources, layouts, strict=True):
        count = int(source["faces"])
        tags = model.faces[offset : offset + count]
        names = model.names[offset : offset + count]
        offset += count
        if layout is None:
            continue
        summary: dict[str, Any] = {
            "file": layout.path.name,
            "component": layout.component,
            "faces": len(layout.faces),
            "used": False,
            "split_faces": 0,
            "pieces": 0,
        }
        summaries.append(summary)
        why = _mismatch(gmsh, layout, source["file"], tags, names, scale=scale)
        if why is not None:
            summary["reason"] = why
            notes.append(
                f"{layout.path.name}: {why}; it does not belong to this file, no layout is "
                "used and the file is gridded as a foreign one"
            )
            continue
        summary["used"] = True
        for tag, face in zip(tags, layout.faces, strict=True):
            assigned[tag] = (face, len(summaries) - 1)
    return assigned, summaries


def _mismatch(
    gmsh: Any, layout: Layout, file: str, tags: list[int], names: list[str], *, scale: float
) -> str | None:
    """Say why a sidecar's faces are not the faces of the file, or ``None`` when they are."""
    if len(tags) != len(layout.faces):
        return f"it lists {len(layout.faces)} faces and {file} holds {len(tags)}"
    if set(names) != {layout.component}:
        return f"its component {layout.component} is not the one {file} names"
    for tag, face in zip(tags, layout.faces, strict=True):
        area = float(gmsh.model.occ.getMass(2, tag)) * scale
        if face.area_m2 is not None and abs(area - face.area_m2) > _AREA_MATCH * face.area_m2:
            return f"its face {face.name} has area {face.area_m2:.6g} m2 and {file} {area:.6g} m2"
    return None


def cut_layout(
    gmsh: Any,
    model: NamedModel,
    assigned: dict[int, tuple[FaceLayout, int]],
    summaries: list[dict[str, Any]],
) -> list[str]:
    """Split each face of the model that has a layout at its lines; return each face's role.

    Every line strictly inside a face's own parameter bounds cuts it along the iso line of its
    own base, keeping its trims (:meth:`GmshKernel.split_imported_face_uv`), so the pieces are
    the layout's blocks. ``model`` is changed in place: a face becomes its pieces, in order, each
    with the face's component. The roles are the sidecar's, one per face of the changed model,
    an empty string for a face with no layout.
    """
    kernel = GmshKernel(gmsh)  # never closed here: the caller's session owns gmsh
    faces: list[int] = []
    names: list[str] = []
    roles: list[str] = []
    for tag, name in zip(model.faces, model.names, strict=True):
        found = assigned.get(tag)
        if found is None:
            faces.append(tag)
            names.append(name)
            roles.append("")
            continue
        face, index = found
        pieces = [Face(tag)]
        for direction, lines in (("u", face.u), ("v", face.v)):
            for value in lines[1:-1]:
                cut = [kernel.split_imported_face_uv(p, direction, value) for p in pieces]
                pieces = [p for group in cut for p in group]
        summaries[index]["split_faces"] += int(len(pieces) > 1)
        summaries[index]["pieces"] += len(pieces)
        faces += [p.tag for p in pieces]
        names += [name] * len(pieces)
        roles += [face.role] * len(pieces)
    gmsh.model.occ.synchronize()
    # In ascending tag order, as the import gave them: healing returns its faces in that order
    # whatever order they were given in, and the roles follow the faces by position.
    ordered = sorted(range(len(faces)), key=faces.__getitem__)
    model.faces[:] = [faces[i] for i in ordered]
    model.names[:] = [names[i] for i in ordered]
    return [roles[i] for i in ordered]
