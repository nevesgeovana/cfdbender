"""The OpenCASCADE steps of the patch mesher: transfinite meshing, the planar tip cap and smoothing.

Pipeline role: the geometry-kernel half of the workspace mesher. The caller owns an initialized
gmsh model (see :func:`cfdbender.mesh._gmsh.session`). A four-sided source surface is
meshed transfinite and recombined from prescribed edge nodes; a planar flat tip cap is meshed
apart as one all-quad region bounded by the rim nodes of the skin; the interior nodes of chosen
faces are Laplacian smoothed back onto their surface; the distance of every node to its source
surface is measured. The pure parts (spacing laws, orientation, checks) live in the geometry
package, which never imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from cfdbender._errors import InputValueError
from cfdbender.geometry.stages import Mesh
from cfdbender.mesh._gmsh import closest_on_face, gmsh_module

__all__ = [
    "CAP_SURFACE",
    "MeshPatch",
    "Transition",
    "mesh_cap",
    "mesh_patches",
    "ordered_edges",
    "rim_edges",
    "rim_loop",
    "smooth_faces",
    "source_metrics",
    "uv_edges",
]

#: Pseudo surface id of the cells of the planar tip cap, which has no OpenCASCADE surface.
CAP_SURFACE = -1
ALGORITHMS = (8, 6, 5)
MIN_SIN = 0.02  # smallest sine of a cap quadrilateral corner (about 1.1 degrees)
#: Global kernel options the planar cap sets for its own model and restores afterwards.
CAP_OPTIONS = (
    "Mesh.Algorithm",
    "Mesh.RecombinationAlgorithm",
    "Mesh.RecombineMinimumQuality",
    "Mesh.RecombineOptimizeTopology",
    "Mesh.Smoothing",
)
TRANSITION_INTERFACES = (
    "wing-fuselage",
    "L1-nacelle-wing",
    "L3-pylon-nacelle",
    "L3-pylon-fuselage",
)


@dataclass
class MeshPatch:
    """A four-sided source surface with its signed edge walk and its edge node requests.

    ``sides`` gives, for a face with more than four boundary curves (a block a junction curve
    crosses, delta F.2), its four sides in walk order, each a run of signed curves; the face
    is transfinite between the first vertices of the sides, and opposite sides carry equal
    node intervals.
    """

    surface: int
    edges: tuple[int, int, int, int]
    u: np.ndarray
    v: np.ndarray
    component: str = "COMPONENT"
    boundary_nodes: dict | None = None
    sides: tuple[tuple[int, ...], ...] | None = None


@dataclass
class Transition:
    """A face meshed unstructured and recombined, with its own node requests per boundary edge."""

    surface: int
    nodes: dict[int, np.ndarray]
    interface: str
    component: str = "INTERFACE"

    def __post_init__(self):
        if self.interface not in TRANSITION_INTERFACES:
            raise InputValueError(
                f"interface: {self.interface!r} is not a permitted transition; "
                f"available: {list(TRANSITION_INTERFACES)}"
            )


def curve_endpoints(model, tag):
    """Return the two vertex tags of a curve in increasing native curve parameter order."""
    vertices = [t for _, t in model.getBoundary([(1, abs(tag))], oriented=False)]
    if len(vertices) != 2:
        raise InputValueError(f"curve {tag}: must have two distinct endpoints")
    lo, _ = model.getParametrizationBounds(1, abs(tag))
    start = np.asarray(model.getValue(1, abs(tag), lo))
    vertices.sort(key=lambda t: np.linalg.norm(np.asarray(model.getValue(0, t, [])) - start))
    return tuple(vertices)


def ordered_edges(model, surface):
    """Return a closed signed edge walk of a four-sided surface, starting at the smallest tag."""
    pending = sorted(t for _, t in model.getBoundary([(2, surface)], oriented=False))
    if len(pending) != 4:
        raise InputValueError(f"surface {surface}: expected four edges, got {len(pending)}")
    edge = pending.pop(0)
    start, current = curve_endpoints(model, edge)
    result = [edge]
    while pending:
        for e in pending:
            a, b = curve_endpoints(model, e)
            if current in (a, b):
                result.append(e if current == a else -e)
                current = b if current == a else a
                pending.remove(e)
                break
        else:
            raise InputValueError("surface: patch edges do not form a connected loop")
    if current != start:
        raise InputValueError("surface: patch edge loop is not closed")
    return tuple(result)


def uv_edges(model, surface):
    """Return the signed boundary walk from UV=(0, 0), following increasing u first."""
    lo, hi = model.getParametrizationBounds(2, surface)
    corners = [
        model.getValue(2, surface, uv)
        for uv in [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    ]
    edges = [e for _, e in model.getBoundary([(2, surface)], oriented=False)]
    result = []
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        matches = []
        for edge in edges:
            p, q = (np.asarray(model.getValue(0, tag, [])) for tag in curve_endpoints(model, edge))
            if np.linalg.norm(p - a) < 1e-6 and np.linalg.norm(q - b) < 1e-6:
                matches.append(edge)
            elif np.linalg.norm(q - a) < 1e-6 and np.linalg.norm(p - b) < 1e-6:
                matches.append(-edge)
        if len(matches) != 1:
            raise InputValueError(f"surface {surface}: tensor boundary is not one four-sided patch")
        result.extend(matches)
    return tuple(result)


def _validate_nodes(nodes):
    x = np.asarray(nodes, float)
    if (
        x.ndim != 1
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or x[0] != 0
        or x[-1] != 1
        or np.any(np.diff(x) <= 0)
    ):
        raise InputValueError("edge nodes: must increase strictly from zero to one")
    return x


def _edge_requests(patch):
    if patch.boundary_nodes is not None:
        return {edge: _validate_nodes(x) for edge, x in patch.boundary_nodes.items()}
    u, v = _validate_nodes(patch.u), _validate_nodes(patch.v)
    walks = [u, v, 1 - u[::-1], 1 - v[::-1]]
    return {
        abs(edge): x if edge > 0 else 1 - x[::-1]
        for edge, x in zip(patch.edges, walks, strict=True)
    }


def _set_curve_nodes(model, edge, positions):
    tags, _, param = model.mesh.getNodes(1, edge, includeBoundary=False)
    if len(tags) != len(positions) - 2:
        raise InputValueError(f"curve {edge}: generated node count differs from the request")
    lo, hi = model.getParametrizationBounds(1, edge)
    for tag, fraction in zip(np.asarray(tags)[np.argsort(param)], positions[1:-1], strict=True):
        t = float(lo[0] + fraction * (hi[0] - lo[0]))
        model.mesh.setNode(int(tag), model.getValue(1, edge, [t]), [t])


def _extract_mesh(model, patches):
    tags, xyz, _ = model.mesh.getNodes()
    tags, xyz = np.asarray(tags), np.asarray(xyz).reshape(-1, 3)
    order = np.argsort(tags)
    tags, xyz = tags[order], xyz[order]
    lookup = {int(t): i for i, t in enumerate(tags)}
    cells, components, surfaces = [], [], []
    for patch in patches:
        types, _, node_blocks = model.mesh.getElements(2, patch.surface)
        for kind, block in zip(types, node_blocks, strict=True):
            if kind not in (2, 3):
                raise InputValueError(f"surface {patch.surface}: unsupported element type {kind}")
            for row in np.asarray(block).reshape(-1, 3 if kind == 2 else 4):
                cells.append(tuple(lookup[int(t)] for t in row))
                components.append(patch.component)
                surfaces.append(patch.surface)
    used = sorted({n for cell in cells for n in cell})
    compact = {n: i for i, n in enumerate(used)}
    return Mesh(
        xyz[used],
        [tuple(compact[n] for n in cell) for cell in cells],
        components,
        surfaces,
        tags[used],
    )


def _register_patch(model, patch, requests):
    requested = (
        {e: _validate_nodes(x) for e, x in patch.nodes.items()}
        if isinstance(patch, Transition)
        else _edge_requests(patch)
    )
    if set(requested) != {t for _, t in model.getBoundary([(2, patch.surface)], oriented=False)}:
        raise InputValueError(f"surface {patch.surface}: patch edge list differs from the source")
    for edge, x in requested.items():
        if edge in requests and (
            len(x) != len(requests[edge]) or not np.allclose(x, requests[edge], atol=1e-12, rtol=0)
        ):
            raise InputValueError(f"curve {edge}: shared curve has incompatible spacing requests")
        requests[edge] = x
    if not isinstance(patch, Transition):
        sides = patch.sides or tuple((e,) for e in patch.edges)
        sizes = [sum(len(requested[abs(e)]) - 1 for e in side) for side in sides]
        if sizes[0] != sizes[2] or sizes[1] != sizes[3]:
            raise InputValueError(f"surface {patch.surface}: opposite edge counts differ")
        corners = [curve_endpoints(model, side[0])[0 if side[0] > 0 else 1] for side in sides]
        model.mesh.setTransfiniteSurface(patch.surface, cornerTags=corners)
    model.mesh.setRecombine(2, patch.surface)


def mesh_patches(model, patches):
    """Prescribe every edge once, then transfinite and recombine each surface; return the mesh."""
    requests: dict[int, np.ndarray] = {}
    for patch in patches:
        _register_patch(model, patch, requests)
    for edge, x in requests.items():
        model.mesh.setTransfiniteCurve(edge, len(x))
    model.mesh.generate(1)
    for edge, x in requests.items():
        _set_curve_nodes(model, edge, x)
    model.mesh.generate(2)
    return _extract_mesh(model, patches)


def smooth_faces(model, mesh, faces, iterations=15, relaxation=0.5):
    """Return a mesh whose interior nodes of the given faces are Laplacian smoothed on the faces.

    A node is interior when every cell using it belongs to one of ``faces``. The smoothed point is
    projected with the OpenCASCADE closest-point query, so the node stays on its source surface.
    """
    faces = set(faces)
    surfaces = np.asarray(mesh.surfaces)
    holders: dict[int, set[int]] = {}
    for cell, surface in zip(mesh.cells, surfaces, strict=True):
        for n in cell:
            holders.setdefault(n, set()).add(int(surface))
    interior = {
        n: next(iter(s)) for n, s in holders.items() if len(s) == 1 and next(iter(s)) in faces
    }
    neighbours: dict[int, set[int]] = {n: set() for n in interior}
    for cell in mesh.cells:
        for a, b in zip(cell, cell[1:] + cell[:1], strict=True):
            if a in neighbours:
                neighbours[a].add(b)
            if b in neighbours:
                neighbours[b].add(a)
    xyz = np.array(mesh.xyz, copy=True)
    by_face: dict[int, list[int]] = {}
    for n, s in interior.items():
        by_face.setdefault(s, []).append(n)
    for _ in range(iterations):
        target = {n: xyz[list(neighbours[n])].mean(axis=0) for n in interior}
        for s, nodes in by_face.items():
            guess = np.array([xyz[n] + relaxation * (target[n] - xyz[n]) for n in nodes])
            projected, _ = model.getClosestPoint(2, s, guess.ravel())
            xyz[nodes] = np.asarray(projected).reshape(-1, 3)
    return replace(mesh, xyz=xyz)


def source_metrics(model, mesh, planes=None):
    """Return ``(normals, distances)`` of the mesh against every incident source surface.

    ``planes`` maps a pseudo surface id to ``(point, unit normal)`` for a region meshed outside
    the OpenCASCADE model, such as the planar flat tip cap.
    """
    planes = planes or {}
    distances = np.zeros(len(mesh.xyz))
    normals = np.zeros((len(mesh.cells), 3))
    surfaces = np.asarray(mesh.surfaces)
    for surface in sorted(set(mesh.surfaces)):
        faces = np.flatnonzero(surfaces == surface)
        nodes = sorted({n for f in faces for n in mesh.cells[f]})
        if surface in planes:
            point, normal = planes[surface]
            d = np.abs((mesh.xyz[nodes] - point) @ normal)
            distances[nodes] = np.maximum(distances[nodes], d)
            normals[faces] = normal
            continue
        projected, _ = model.getClosestPoint(2, surface, mesh.xyz[nodes].ravel())
        d = np.linalg.norm(mesh.xyz[nodes] - np.asarray(projected).reshape(-1, 3), axis=1)
        distances[nodes] = np.maximum(distances[nodes], d)
        centers = np.array([mesh.xyz[list(mesh.cells[f])].mean(axis=0) for f in faces])
        _, uv = closest_on_face(model, surface, centers)
        normals[faces] = np.asarray(model.getNormal(surface, uv.ravel())).reshape(-1, 3)
    return normals, distances


def rim_edges(model, cap_surfaces):
    """Return the edges bounding exactly one cap surface: the rim shared with the skin."""
    count: dict[int, int] = {}
    for s in cap_surfaces:
        for _, e in model.getBoundary([(2, s)], oriented=False):
            if model.occ.getMass(1, e) > 1e-12:  # the collapsed end of a one-face cap is a point
                count[e] = count.get(e, 0) + 1
    return sorted(e for e, n in count.items() if n == 1)


def rim_loop(model, mesh, edges):
    """Return the ordered node indices of the closed rim, walked over the skin nodes on the rim."""
    lookup = {int(t): i for i, t in enumerate(mesh.node_tags)}
    neighbours: dict[int, list[int]] = {}
    for e in edges:
        tags, _, param = model.mesh.getNodes(1, e, includeBoundary=True)
        seq = [int(t) for t in np.asarray(tags)[np.argsort(param)]]
        for a, b in zip(seq, seq[1:], strict=False):
            neighbours.setdefault(a, []).append(b)
            neighbours.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in neighbours.values()):
        raise InputValueError("cap: the rim is not one simple closed loop")
    start = min(neighbours)
    loop, previous, here = [start], None, start
    while True:
        a, b = neighbours[here]
        nxt = a if a != previous else b
        if nxt == start:
            break
        loop.append(nxt)
        previous, here = here, nxt
    if len(loop) != len(neighbours):
        raise InputValueError("cap: the rim is not one closed loop")
    return [lookup[t] for t in loop]


def _chains(ring):
    """Split the closed rim at the trailing-edge base midpoint and the leading-edge point.

    The base midpoint is the middle node of the one collinear consecutive triple at the aft end;
    the leading-edge point is the rim node farthest from it. Returns
    ``(te, le, forward, backward)``:
    positions in ``ring`` of the two arcs from te to le, each excluding both ends.
    """
    n = len(ring)
    best = None
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        u, v = b - a, c - b
        cross = np.linalg.norm(np.cross(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))
        if cross < 1e-9:
            best = i if best is None else best
            if best != i:
                raise InputValueError(
                    "cap: more than one collinear rim triple; the trailing-edge base is ambiguous"
                )
    if best is None:
        raise InputValueError("cap: no collinear rim triple at the trailing-edge base")
    te = best
    le = int(np.argmax(np.linalg.norm(ring - ring[te], axis=1)))
    forward = [(te + k) % n for k in range(1, (le - te) % n)]
    backward = [(te - k) % n for k in range(1, (te - le) % n)]
    return te, le, forward, backward


def _cut_stations(ring, te, le, arc_a, cut_shift):
    direction = ring[le] - ring[te]
    length = np.linalg.norm(direction)
    direction = direction / length
    station = sorted({float((ring[i] - ring[te]) @ direction) for i in arc_a[1:]})
    station = [s for s in station if 0.05 * length < s < 0.995 * length]
    if cut_shift > 0:
        station = sorted(station + [station[-1] + 0.5 * (length - station[-1])])
    elif cut_shift < 0:
        station = station[:-1]
    return [ring[te] + s * direction for s in station]


def _build_cap_halves(gmsh: Any, ring, te, le, arcs, *, cut):
    g = gmsh.model.geo

    def size(a, b, c):
        return 0.5 * (np.linalg.norm(a - b) + np.linalg.norm(c - b))

    n = len(ring)
    points = {
        i: g.addPoint(*ring[i], size(ring[i - 1], ring[i], ring[(i + 1) % n])) for i in range(n)
    }
    cut_ends = [ring[te], *cut, ring[le]]
    cut_points = [points[te]]
    for j, p in enumerate(cut, start=1):
        cut_points.append(g.addPoint(*p, size(cut_ends[j - 1], p, cut_ends[j + 1])))
    cut_points.append(points[le])
    cut_lines = [g.addLine(a, b) for a, b in zip(cut_points, cut_points[1:], strict=False)]

    def arc_lines(sequence):
        chain = [te, *sequence, le]
        return [g.addLine(points[a], points[b]) for a, b in zip(chain, chain[1:], strict=False)]

    lines_a, lines_b = arc_lines(arcs[0]), arc_lines(arcs[1])
    loop_a = lines_a + [-line for line in reversed(cut_lines)]
    loop_b = [-line for line in reversed(lines_b)] + cut_lines
    g.addPlaneSurface([g.addCurveLoop(loop_a)], 1)
    g.addPlaneSurface([g.addCurveLoop(loop_b)], 2)
    g.synchronize()
    for line in lines_a + lines_b + cut_lines:
        gmsh.model.mesh.setTransfiniteCurve(line, 2)
    for s in (1, 2):
        gmsh.model.mesh.setRecombine(2, s)
    return points


def _generate_quads(gmsh: Any, algorithms, n, n_cut):
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.0)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    last = None
    for algorithm in algorithms:
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:  # gmsh raises its own untyped error on a failed meshing
            last = f"algorithm {algorithm}: {exc}"
            continue
        found = [[int(t) for t in gmsh.model.mesh.getElements(2, s)[0]] for s in (1, 2)]
        if found == [[3], [3]]:
            return
        last = f"algorithm {algorithm}: element types {found}"
    raise InputValueError(
        f"cap: halves are not all quadrilateral (rim {n} nodes, cut {n_cut}); {last}"
    )


def _mesh_cap_once(mesh, loop, component, cut_shift=0, algorithms=None):
    """Append an all-quad mesh of the planar region bounded by ``loop``; return ``(mesh, normal)``.

    The cap is split along the straight chord line from the trailing-edge base midpoint to the
    leading-edge point, so no quadrilateral holds the 180 degree corner of the base. The cut
    nodes sit at the chord stations of the rim nodes on one arc (``cut_shift`` adds or drops a
    node when the half boundaries are odd).
    """
    gmsh = gmsh_module("meshing the planar tip cap")
    xyz = mesh.xyz
    ring = xyz[loop]
    centre = ring.mean(axis=0)
    _, _, vt = np.linalg.svd(ring - centre)
    normal = vt[2]
    if np.max(np.abs((ring - centre) @ normal)) > 1e-8:
        raise InputValueError("cap: the rim is not planar")
    te, le, arc_a, arc_b = _chains(ring)
    cut = _cut_stations(ring, te, le, arc_a, cut_shift)
    previous = gmsh.model.getCurrent()
    keys = (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromCurvature",
        # The cap's algorithm and recombination options are restored too, so a later pass of
        # the caller's model (a transition face retried) meshes as the first pass did.
        *CAP_OPTIONS,
    )
    saved_options = {k: gmsh.option.getNumber(k) for k in keys}
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.model.add("cap")
    try:
        points = _build_cap_halves(gmsh, ring, te, le, (arc_a, arc_b), cut=cut)
        _generate_quads(gmsh, algorithms or ALGORITHMS, len(ring), len(cut))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        position = {int(t): i for i, t in enumerate(tags)}
        boundary = {}
        for i in range(len(ring)):
            t, _, _ = gmsh.model.mesh.getNodes(0, points[i])
            boundary[int(t[0])] = loop[i]
        quads = []
        for s in (1, 2):
            quads.extend(np.asarray(gmsh.model.mesh.getElements(2, s)[2][0]).reshape(-1, 4))
    finally:
        gmsh.model.remove()
        gmsh.model.setCurrent(previous)
        for k, v in saved_options.items():
            gmsh.option.setNumber(k, v)
    new_xyz, index = [], {}
    base = len(xyz)
    for t, i in position.items():
        if t in boundary:
            index[t] = boundary[t]
        else:
            index[t] = base + len(new_xyz)
            new_xyz.append(coords[i])
    cells = [tuple(index[int(t)] for t in q) for q in quads]
    return replace(
        mesh,
        xyz=np.vstack([xyz, np.array(new_xyz).reshape(-1, 3)]),
        node_tags=np.r_[mesh.node_tags, -np.arange(1, len(new_xyz) + 1)],
        cells=mesh.cells + cells,
        components=mesh.components + [component] * len(cells),
        surfaces=mesh.surfaces + [CAP_SURFACE] * len(cells),
    ), normal


def _corner_sines(xyz, cells):
    sines = []
    for cell in cells:
        p = xyz[list(cell)]
        a, b = np.roll(p, -1, axis=0) - p, np.roll(p, 1, axis=0) - p
        sines.append(
            np.linalg.norm(np.cross(a, b), axis=1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        )
    return np.array(sines)


def mesh_cap(mesh, loop, component):
    """Mesh the planar cap, trying cut variants and algorithms until no corner is collinear."""
    last = None
    for shift in (0, 1, -1, 2, -2):
        for algorithms in ((8, 6, 5), (6, 8, 5), (5, 6, 8)):
            try:
                result, normal = _mesh_cap_once(mesh, loop, component, shift, algorithms)
            except InputValueError as exc:
                last = str(exc)
                continue
            sines = _corner_sines(result.xyz, result.cells[len(mesh.cells) :])
            if sines.min() >= MIN_SIN:
                return result, normal
            last = (
                f"cut shift {shift} algorithms {algorithms}: smallest corner sine {sines.min():.4g}"
            )
    raise InputValueError(
        f"cap: no meshing variant gave an all-quad mesh without collinear corners; {last}"
    )


def _dominant_algorithm(gmsh: Any, algorithms, n: int):
    """Mesh the open ``cap`` surface with the first algorithm whose cells are all sound."""
    last = None
    for algorithm in algorithms:
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:  # gmsh raises its own untyped error on a failed meshing
            last = f"algorithm {algorithm}: {exc}"
            continue
        kinds, _, blocks = gmsh.model.mesh.getElements(2, 1)
        if len(kinds) and set(int(k) for k in kinds) <= {2, 3}:
            return
        last = f"algorithm {algorithm}: element types {[int(k) for k in kinds]}"
    raise InputValueError(f"cap: no quad-dominant mesh of the rim ({n} nodes); {last}")


def _split_collinear_quads(coords, position, cells):
    """Split each quad holding a collinear corner along the diagonal from that corner.

    A quad that closes over a straight run of the rim (the base midpoint, a smooth outline node)
    has a corner near 180 degrees; as two triangles the corner is divided and both are sound.
    """
    result = []
    for cell in cells:
        cell = [int(t) for t in cell]
        if len(cell) == 4:
            sines = _corner_sines(coords, [[position[t] for t in cell]])[0]
            k = int(np.argmin(sines))
            if sines[k] < MIN_SIN:
                a, b, c, d = (cell[(k + j) % 4] for j in range(4))
                result += [[a, b, c], [a, c, d]]
                continue
        result.append(cell)
    return result


def mesh_cap_dominant(mesh, loop, component):
    """Append a quad-dominant mesh of the planar region bounded by ``loop``.

    MESH_DESIGN section 8 lets the tip cap be quad-dominant: the region is meshed free, recombined
    with the blossom pairing, and the triangles it keeps are reported apart from the structured
    rows. The boundary nodes are the rim nodes; each variant is refused when a corner of any cell
    is collinear (smallest sine under :data:`MIN_SIN`). Returns ``(mesh, normal)``.
    """
    gmsh = gmsh_module("meshing the planar tip cap")
    ring = mesh.xyz[loop]
    centre = ring.mean(axis=0)
    normal = np.linalg.svd(ring - centre)[2][2]
    if np.max(np.abs((ring - centre) @ normal)) > 1e-8:
        raise InputValueError("cap: the rim is not planar")
    previous = gmsh.model.getCurrent()
    keys = ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeExtendFromBoundary", *CAP_OPTIONS)
    saved_options = {k: gmsh.option.getNumber(k) for k in keys}
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.0)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    gmsh.model.add("cap")
    try:
        position, coords, cells, boundary = _dominant_cap_cells(gmsh, ring, loop)
    finally:
        gmsh.model.remove()
        gmsh.model.setCurrent(previous)
        for k, v in saved_options.items():
            gmsh.option.setNumber(k, v)
    new_xyz, index = [], {}
    base = len(mesh.xyz)
    for t in position:
        if t in boundary:
            index[t] = boundary[t]
        else:
            index[t] = base + len(new_xyz)
            new_xyz.append(coords[position[t]])
    new_cells = [tuple(index[int(t)] for t in c) for c in cells]
    return replace(
        mesh,
        xyz=np.vstack([mesh.xyz, np.array(new_xyz).reshape(-1, 3)]),
        node_tags=np.r_[mesh.node_tags, -np.arange(1, len(new_xyz) + 1)],
        cells=mesh.cells + new_cells,
        components=mesh.components + [component] * len(new_cells),
        surfaces=mesh.surfaces + [CAP_SURFACE] * len(new_cells),
    ), normal


def _dominant_cap_cells(gmsh, ring, loop):
    """Build the planar polygon and retain the best sound quad-dominant variant."""
    n = len(ring)
    g = gmsh.model.geo
    size = [
        0.5 * (np.linalg.norm(ring[i] - ring[i - 1]) + np.linalg.norm(ring[(i + 1) % n] - ring[i]))
        for i in range(n)
    ]
    points = [g.addPoint(*ring[i], size[i]) for i in range(n)]
    lines = [g.addLine(points[i], points[(i + 1) % n]) for i in range(n)]
    g.addPlaneSurface([g.addCurveLoop(lines)], 1)
    g.synchronize()
    for line in lines:
        gmsh.model.mesh.setTransfiniteCurve(line, 2)
    gmsh.model.mesh.setRecombine(2, 1)
    best = None
    for algorithms in ((8, 6, 5), (6, 8, 5), (5, 6, 8)):
        _dominant_algorithm(gmsh, algorithms, n)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        position = {int(t): i for i, t in enumerate(tags)}
        cells = []
        for kind, block in zip(*gmsh.model.mesh.getElements(2, 1)[::2], strict=True):
            cells.extend(np.asarray(block).reshape(-1, 3 if kind == 2 else 4))
        cells = _split_collinear_quads(coords, position, cells)
        smallest = min(
            float(_corner_sines(coords, [[position[int(t)] for t in c]]).min()) for c in cells
        )
        if best is None or smallest > best[0]:
            boundary = {}
            for i in range(n):
                t, _, _ = gmsh.model.mesh.getNodes(0, points[i])
                boundary[int(t[0])] = loop[i]
            best = (smallest, position, coords, cells, boundary)
        if smallest >= MIN_SIN:
            break
    if best[0] < MIN_SIN:
        raise InputValueError(f"cap: smallest corner sine {best[0]:.4g} on every variant")
    return best[1:]
