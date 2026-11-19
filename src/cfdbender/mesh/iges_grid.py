"""Grid a named IGES assembly into one OBJ, one family per component.

Each IGES file is imported into one OpenCASCADE model (the gmsh kernel) with every face
named by the NAME of its level (:mod:`cfdbender.geometry._iges_names`). The faces are
healed and fragmented together so the edges two components share are conformal, four-sided
patches are meshed transfinite with the laws of the grid control file and recombined into quads,
and every other patch is meshed quad-dominant at its component's size
(:mod:`cfdbender.mesh._iges_mesh`). The OBJ holds one family per component in the
order of the control file, each oriented outward; ``<out>.boundaries.toml`` is the boundaries
file ``cfd-mesh`` reads, with one points file per trailing edge found; and
``<out>.json`` records the nodes, faces, quad share, least scaled Jacobian and largest aspect of
every component, the conformal check and every warning. Quality is warned with its numbers and
never refused.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import (
    GeometryWarning,
    InputValueError,
    NotApplicableWarning,
    warn,
)
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.geometry._iges_names import (
    DEFAULT_NAME_ENCODING,
    named_curves,
    read_iges_names,
    require_name_encoding,
    resolve_components,
)
from cfdbender.geometry._iges_skin import outer_skin
from cfdbender.mesh._gmsh import gmsh_module, import_named, keep_components, session
from cfdbender.mesh._gmsh_worker import run_isolated
from cfdbender.mesh._iges_grid_config import GridConfig, read_grid, read_layout
from cfdbender.mesh._iges_mesh import (
    assign_layouts,
    cut_layout,
    junction_checks,
    junction_warnings,
    mesh_model,
)
from cfdbender.mesh._refine._obj import ObjMesh, midpoints, obj_text, write_te

__all__ = ["grid_iges"]

_PURPOSE = "Gridding an IGES assembly"
#: The share of the assembly's diagonal that is the default target edge when none is given.
_DEFAULT_SIZE_SHARE = 0.01
#: The share of the assembly's diagonal within which two faces are joined when none is given.
_DEFAULT_TOLERANCE_SHARE = 1e-6
#: Quality below which a component is warned (never refused): least scaled Jacobian, largest
#: aspect (longest edge over shortest edge of one face).
WARN_JACOBIAN, WARN_ASPECT = 0.1, 20.0
#: A sliver is a face with a corner under this angle or over 180 degrees less it, in degrees: a
#: panel the solver's surface vorticity can degenerate on.
SLIVER_DEG = 5.0
#: The half angle of the cone about +x (downstream) within which the bisector of a trailing
#: edge's two normals points.
_AFT_CONE_DEG = 60.0
#: A node lies on a named trailing-edge curve within this share of its component size.
_ON_CURVE_SHARE = 0.01
#: Two trailing edges that meet at a node belong to one line when the line turns there by less
#: than this, in degrees; a tip's base edge turns about 90 degrees from the rims it joins.
_RIM_TURN_DEG = 60.0

Faces = list[list[int]]


def _order(config: GridConfig, found: list[str], known: list[str] | None = None) -> list[str]:
    """Return the components in the control file's order, then the others in the files' order.

    ``known`` lists every component the files carry when ``found`` is only the selected ones:
    a table naming a component that is known but not selected is skipped without a warning.
    """
    every = found if known is None else known
    order = [name for name in config.components if name in found]
    for name in config.components:
        if name not in every:
            warn(
                f"grid: component.{name} names no component of the IGES files "
                f"({', '.join(every)}) and is skipped",
                NotApplicableWarning,
            )
    return order + [name for name in found if name not in order]


def _edges(face: list[int]) -> list[tuple[int, int]]:
    return list(zip(face, face[1:] + face[:1], strict=True))


def _normal(verts: NDArray[np.float64], face: list[int]) -> NDArray[np.float64]:
    """Newell's area vector of a polygon (its length is twice the area)."""
    p = verts[face]
    q = np.roll(p, -1, axis=0)
    return np.array(
        [
            np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
            np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
            np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
        ]
    )


def _walk(
    out: Faces, owners: dict[tuple[int, int], list[int]], start: int, seen: list[bool]
) -> tuple[list[int], int]:
    """Orient the connected piece of ``start`` like it, in place; return the piece and the flips.

    A neighbor across an edge two faces share that runs the edge the same way is reversed.
    """
    piece, stack = [start], [start]
    seen[start] = True
    flips = 0
    while stack:
        i = stack.pop()
        for a, b in set(_edges(out[i])):
            sharing = owners[(min(a, b), max(a, b))]
            if len(sharing) != 2:
                continue
            for j in sharing:
                if j == i or seen[j]:
                    continue
                if (a, b) in set(_edges(out[j])):
                    out[j].reverse()
                    flips += 1
                seen[j] = True
                piece.append(j)
                stack.append(j)
    return piece, flips


def orient(verts: NDArray[np.float64], faces: Faces) -> tuple[Faces, int]:
    """Orient the faces of one component consistently and outward; return them and the flips.

    Faces are walked across every edge two of them share; a neighbor that runs the shared edge
    the same way is reversed. Each connected piece is then turned so that its faces point away
    from its center on the whole (the sum of each face's area vector dotted with its offset from
    the piece's center is positive), which is outward for a closed body and for a wing.
    """
    owners: dict[tuple[int, int], list[int]] = {}
    for i, face in enumerate(faces):
        for a, b in _edges(face):
            owners.setdefault((min(a, b), max(a, b)), []).append(i)
    out = [list(f) for f in faces]
    seen = [False] * len(out)
    flips = 0
    for start in range(len(out)):
        if seen[start]:
            continue
        piece, turned = _walk(out, owners, start, seen)
        flips += turned
        centre = verts[np.unique([v for i in piece for v in out[i]])].mean(axis=0)
        outward = sum(
            float(_normal(verts, out[i]) @ (verts[out[i]].mean(axis=0) - centre)) for i in piece
        )
        if outward < 0:
            for i in piece:
                out[i].reverse()
            flips += len(piece)
    return out, flips


def conformal(faces: Faces, boundary: set[tuple[int, int]]) -> dict[str, int]:
    """Count the open edges of the whole mesh and those that are not true boundaries.

    An open edge is used by one face. It is a true boundary when it lies on a curve of the
    geometry that bounds one face and lies on no other (``boundary``, sorted node pairs), such
    as the open root of a wing. Every other open edge is a mismatch: two faces meet there without
    sharing nodes, or a face is missing.
    """
    count: dict[tuple[int, int], int] = {}
    for face in faces:
        for a, b in _edges(face):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    open_edges = [e for e, n in count.items() if n == 1]
    true = sum(1 for e in open_edges if e in boundary)
    return {
        "open_edges": len(open_edges),
        "boundary_edges": true,
        "conformal_mismatches": len(open_edges) - true,
    }


def _compact(
    nodes: NDArray[np.float64],
    families: dict[str, Faces],
    boundary: set[tuple[int, int]],
    marked: dict[str, set[tuple[int, int]]] | None = None,
) -> tuple[
    NDArray[np.float64], dict[str, Faces], set[tuple[int, int]], dict[str, set[tuple[int, int]]]
]:
    """Keep the nodes the faces use, renumbered in order of first use, with the boundary.

    ``marked`` is more sets of edges by component (the trailing-edge rims of a layout role),
    renumbered the same way.
    """
    order: dict[int, int] = {}
    for faces in families.values():
        for face in faces:
            for v in face:
                order.setdefault(v, len(order))
    keep = np.fromiter(order, dtype=np.int64, count=len(order))
    out = {n: [[order[v] for v in f] for f in fs] for n, fs in families.items()}

    def renumber(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {
            (min(order[a], order[b]), max(order[a], order[b]))
            for a, b in pairs
            if a in order and b in order
        }

    return nodes[keep], out, renumber(boundary), {n: renumber(e) for n, e in (marked or {}).items()}


def _corners(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the angle at each corner of one face, in degrees (0 where an edge has no length)."""
    before = np.roll(p, 1, axis=0) - p
    after = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    dot = np.einsum("ij,ij->i", before, after)
    cosine = np.divide(dot, lengths, out=np.ones_like(dot), where=lengths > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def quality(verts: NDArray[np.float64], faces: Faces) -> dict[str, Any]:
    """Return the faces, nodes, quad share, largest aspect and corner angles of one component."""
    quads = sum(1 for f in faces if len(f) == 4)
    aspect = 0.0
    smallest, largest, slivers = 180.0, 0.0, 0
    for face in faces:
        p = verts[face]
        e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        if e.min() > 0:
            aspect = max(aspect, float(e.max() / e.min()))
        else:
            aspect = math.inf
        angles = _corners(p)
        low, high = float(angles.min()), float(angles.max())
        smallest, largest = min(smallest, low), max(largest, high)
        slivers += low < SLIVER_DEG or high > 180.0 - SLIVER_DEG
    return {
        "faces": len(faces),
        "nodes": len({v for f in faces for v in f}),
        "quads": quads,
        "triangles": len(faces) - quads,
        "quad_share": quads / len(faces) if faces else 0.0,
        "largest_aspect": aspect,
        "smallest_corner_deg": smallest,
        "largest_corner_deg": largest,
        "sliver_faces": slivers,
    }


def trailing_edges(
    verts: NDArray[np.float64], faces: Faces, angle_deg: float
) -> set[tuple[int, int]]:
    """Return the sharp edges of one component that face downstream: its trailing edges.

    An edge two faces share is a trailing edge when the faces' outward normals differ by more
    than ``angle_deg`` and their bisector points downstream (+x) within
    :data:`_AFT_CONE_DEG`. A leading edge points upstream and a tip cap sideways, so neither is
    taken; on a blunt trailing edge both rims are, and :func:`one_rim` keeps one of them.
    """
    owners: dict[tuple[int, int], list[int]] = {}
    for i, face in enumerate(faces):
        for a, b in _edges(face):
            owners.setdefault((min(a, b), max(a, b)), []).append(i)
    normals = [_normal(verts, f) for f in faces]
    normals = [n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n for n in normals]
    limit = math.cos(math.radians(angle_deg))
    aft = math.cos(math.radians(_AFT_CONE_DEG))
    sharp = set()
    for edge, pair in owners.items():
        if len(pair) != 2:
            continue
        n1, n2 = normals[pair[0]], normals[pair[1]]
        bisector = n1 + n2
        length = float(np.linalg.norm(bisector))
        if float(n1 @ n2) < limit and length > 0 and bisector[0] / length > aft:
            sharp.add(edge)
    return sharp


def _lines(verts: NDArray[np.float64], edges: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Split edges into lines, returned longest first, then by node.

    Two edges at a node are one line when only they meet there and the line turns by less than
    :data:`_RIM_TURN_DEG`.
    """
    at: dict[int, list[tuple[int, int]]] = {}
    for edge in edges:
        for v in edge:
            at.setdefault(v, []).append(edge)
    parent = {edge: edge for edge in edges}

    def find(edge: tuple[int, int]) -> tuple[int, int]:
        while parent[edge] != edge:
            parent[edge] = parent[parent[edge]]
            edge = parent[edge]
        return edge

    limit = math.cos(math.radians(_RIM_TURN_DEG))
    for v, meeting in at.items():
        if len(meeting) != 2:
            continue
        (a, b) = (verts[e[0] if e[1] == v else e[1]] - verts[v] for e in meeting)
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        # A straight line through v has its two edges pointing opposite ways from it.
        if la > 0 and lb > 0 and -float(a @ b) / (la * lb) > limit:
            parent[find(meeting[0])] = find(meeting[1])
    groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for edge in edges:
        groups.setdefault(find(edge), set()).add(edge)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def one_rim(
    verts: NDArray[np.float64], faces: Faces, edges: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], int]:
    """Keep one rim of each blunt trailing edge; return the edges kept and the number dropped.

    A blunt trailing edge is found as two rims, lines of edges that lie on opposite sides of one
    face of its base strip. Marking both would mark two lines a base cell apart, and one
    rim carries the trailing edge. Of two opposite rims the longer is kept (on a tie the one
    with the lower node), and a line that joins a dropped rim to a kept one, a tip's base edge,
    is dropped with it. A sharp
    trailing edge is one line with nothing opposite and is kept whole.
    """
    lines = _lines(verts, edges)
    line_of = {edge: i for i, line in enumerate(lines) for edge in line}
    opposite = _opposite(faces, line_of, len(lines))
    dropped: set[int] = set()
    for i in range(len(lines)):
        # Lines come longest first, so a line still kept when its turn comes is the longer rim.
        if i not in dropped:
            dropped |= opposite[i]
    meets = _meets(line_of, len(lines))
    for i in range(len(lines)):
        if i in dropped or opposite[i]:
            continue
        ends = [m for m in meets[i] if m]
        # A tip's base edge meets the dropped rim alone at one end and the kept rim at the other;
        # a sharp line going on where a blunt edge closes meets both rims at one node, and stays.
        if any(m <= dropped for m in ends) and any(m - dropped for m in ends):
            dropped.add(i)
    kept = {e for i, line in enumerate(lines) if i not in dropped for e in line}
    return kept, len(edges) - len(kept)


def _opposite(faces: Faces, line_of: dict[tuple[int, int], int], count: int) -> dict[int, set[int]]:
    """Return, per line, the lines with an edge on the far side of one of its faces."""
    opposite: dict[int, set[int]] = {i: set() for i in range(count)}
    for face in faces:
        on = [(a, b) for a, b in ((min(e), max(e)) for e in _edges(face)) if (a, b) in line_of]
        for i, e1 in enumerate(on):
            for e2 in on[i + 1 :]:
                if not set(e1) & set(e2) and line_of[e1] != line_of[e2]:
                    opposite[line_of[e1]].add(line_of[e2])
                    opposite[line_of[e2]].add(line_of[e1])
    return opposite


def _meets(line_of: dict[tuple[int, int], int], count: int) -> dict[int, list[set[int]]]:
    """Return, per line, the other lines that meet it at each of its nodes."""
    at: dict[int, set[int]] = {}
    for edge, i in line_of.items():
        for v in edge:
            at.setdefault(v, set()).add(i)
    meets: dict[int, list[set[int]]] = {i: [] for i in range(count)}
    for meeting in at.values():
        for i in meeting:
            meets[i].append(meeting - {i})
    return meets


def _inner(
    config: GridConfig, meshed: Any, order: list[str], warnings: list[str]
) -> tuple[dict[str, Faces], dict[str, dict[str, float]], str]:
    """Return the families without the faces inside the assembly, what went, and the note."""
    families = {name: meshed.faces[name] for name in order if meshed.faces.get(name)}
    if not config.remove_inner_faces:
        if meshed.record.get("fragment", {}).get("crossings"):
            warnings.append(
                "remove_inner_faces = false: the faces that lie inside the assembly where "
                "components cross are kept"
            )
        return families, {}, "kept whole: remove_inner_faces = false"
    families, removed, skin = outer_skin(meshed.nodes, families, symmetry=config.symmetry)
    if skin.startswith("kept whole"):
        warnings.append(
            "the faces that lie inside the assembly where components cross are kept, every "
            f"face of the assembly with them ({skin.removeprefix('kept whole: ')})"
        )
    for name, gone in removed.items():
        warnings.append(
            f"{name}: {gone['faces']:g} face(s) of area {gone['area']:.4g} lie inside the "
            "assembly and were removed"
        )
    return families, removed, skin


def _toml_key(name: str) -> str:
    return json.dumps(name, ensure_ascii=False)


def grid_iges(
    sources: Sequence[str | Path],
    grid: str | Path,
    out: str | Path,
    *,
    name_encoding: str = DEFAULT_NAME_ENCODING,
    components: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Grid the named components of IGES files into one OBJ with its boundaries file and record.

    Parameters
    ----------
    sources : sequence of str or Path
        IGES files, read only; their faces are named by the NAME of their level.
    grid : str or Path
        The grid control file (``grid.toml``).
    name_encoding : str, optional
        The text encoding of the NAME bytes in the IGES files; default ``"latin-1"``.
    components : sequence of str, optional
        Grid only these components, named as the files name them (an exact name, or one that
        differs only in case); the others leave the model before the join. An unknown or
        ambiguous name is refused with the names found. Every component when omitted.
    out : str or Path
        The OBJ to write; ``<out>.boundaries.toml``, ``<out>.json`` and one
        ``<stem>.<component>.te.txt`` per component with a trailing edge are written beside it.

    Returns
    -------
    dict
        The record written to ``<out>.json``.

    Raises
    ------
    MissingExtraError
        The gmsh library cannot load (``KernelUnavailableError``).
    InputValueError, InputArtifactError
        The control file or an IGES file is refused, or the geometry kernel failed (it runs in
        a worker process of its own, so a native fault ends only that worker and is reported
        with its cause); nothing was written.
    """
    require_name_encoding(name_encoding)
    target = Path(out)
    if target.suffix.lower() != ".obj":
        raise InputValueError(f"grid: {target} must be an .obj file. Nothing was written.")
    read_grid(grid)  # a control file that cannot be read is refused before a worker starts
    gmsh_module(_PURPOSE)
    return run_isolated(
        f"{__name__}:_grid",
        {
            "sources": [str(s) for s in sources],
            "grid": str(grid),
            "out": str(target),
            "name_encoding": name_encoding,
            "components": None if components is None else list(components),
        },
        _grid,
    )


def _grid(
    sources: Sequence[str | Path],
    grid: str | Path,
    out: str | Path,
    name_encoding: str,
    components: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the grid in this process; :func:`grid_iges` calls it in a worker."""
    target = Path(out)
    config = read_grid(grid)
    notes: list[str] = []
    with session(_PURPOSE) as gmsh:
        layouts = [read_layout(source, notes) for source in sources]
        model = import_named(gmsh, sources, name_encoding=name_encoding)
        known = list(model.components())
        found = known if components is None else resolve_components(known, components)
        assigned, laid = assign_layouts(gmsh, model, layouts, notes)
        if components is not None:
            model = keep_components(gmsh, model, found)
        roles = cut_layout(gmsh, model, assigned, laid)
        model.notes.extend(notes)
        order = _order(config, found, known)
        # The faces' own extent, not the model's: a degenerate edge of an imported face can leave a
        # free curve far outside the part (y = 2000 for a 2 m wing), and every tolerance below
        # is a share of the diagonal.
        boxes = [gmsh.model.getBoundingBox(2, tag) for tag in model.faces]
        low = [min(b[k] for b in boxes) for k in range(3)]
        high = [max(b[k + 3] for b in boxes) for k in range(3)]
        diagonal = math.dist(low, high)
        default = config.size or _DEFAULT_SIZE_SHARE * diagonal
        sizes = {}
        for name in order:
            grid_of = config.components.get(name)
            if grid_of is None:
                warn(
                    f"grid: component {name} has no [component.{name}] table and is meshed "
                    f"unstructured at the defaults size {default:g}",
                    NotApplicableWarning,
                )
            sizes[name] = grid_of.size if grid_of and grid_of.size else default
        tolerance = config.tolerance or _DEFAULT_TOLERANCE_SHARE * diagonal

        def reopen() -> Any:
            gmsh.clear()
            again = import_named(gmsh, sources, name_encoding=name_encoding)
            held, again_laid = assign_layouts(gmsh, again, layouts, [])
            again = again if components is None else keep_components(gmsh, again, found)
            roles[:] = cut_layout(gmsh, again, held, again_laid)
            return again

        meshed = mesh_model(
            gmsh, model, config, order, sizes, tolerance=tolerance, reopen=reopen, roles=roles
        )
        meshed.record["layout"] = {"sidecars": laid} if laid else None
        originals: dict[str, list[int]] = {}

        def distance(name: str, points: NDArray[np.float64]) -> float:
            """Largest distance of ``points`` to the original, untrimmed faces of ``name``."""
            if not originals:  # the files are read once more, only when a junction is checked
                gmsh.clear()
                again = import_named(gmsh, sources, name_encoding=name_encoding)
                if components is not None:
                    again = keep_components(gmsh, again, found)
                originals.update(again.components())
            gap = np.full(len(points), np.inf)
            for face in originals[name]:
                for i, point in enumerate(points):
                    nearest, _ = gmsh.model.getClosestPoint(2, face, point.tolist())
                    gap[i] = min(gap[i], math.dist(nearest, point))
            return float(gap.max())

        return _write(
            target,
            model,
            config,
            order,
            sizes,
            meshed=meshed,
            default=default,
            curves=_curves(sources, model.unit, name_encoding),
            selection=None
            if components is None
            else {
                "requested": list(components),
                "selected": found,
                "left_out": [n for n in known if n not in found],
            },
            junction=(distance, tolerance),
        )


def _curves(
    sources: Sequence[str | Path], unit: str, name_encoding: str
) -> dict[str, list[NDArray[np.float64]]]:
    """Return the named curves of every source, scaled to the model's unit."""
    out: dict[str, list[NDArray[np.float64]]] = {}
    for source in sources:
        factor = float(
            METRES_PER_UNIT[read_iges_names(source, name_encoding=name_encoding).unit or unit]
            / METRES_PER_UNIT[unit]
        )
        for name, lines in named_curves(source, name_encoding=name_encoding).items():
            out.setdefault(name, []).extend(line * factor for line in lines)
    return out


def edges_on(
    verts: NDArray[np.float64],
    faces: Faces,
    lines: Sequence[NDArray[np.float64]],
    reach: float,
) -> set[tuple[int, int]]:
    """Return the edges of the faces whose two nodes lie within ``reach`` of a polyline."""
    if not lines:
        return set()
    starts = np.vstack([line[:-1] for line in lines])
    spans = np.vstack([np.diff(line, axis=0) for line in lines])
    squared = np.maximum(np.einsum("ij,ij->i", spans, spans), 1e-300)
    used = sorted({v for f in faces for v in f})
    points = verts[used]
    near = np.zeros(len(used), dtype=bool)
    for start, span, length in zip(starts, spans, squared, strict=True):
        t = np.clip((points - start) @ span / length, 0.0, 1.0)
        near |= np.linalg.norm(start + t[:, None] * span - points, axis=1) <= reach
    on = {v for v, hit in zip(used, near, strict=True) if hit}
    return {(min(a, b), max(a, b)) for f in faces for a, b in _edges(f) if a in on and b in on}


def _quality_warnings(name: str, stats: dict[str, Any]) -> list[str]:
    jacobian = stats["least_scaled_jacobian"]
    out = []
    if jacobian is not None and jacobian < WARN_JACOBIAN:
        out.append(f"{name}: least scaled Jacobian {jacobian:.3g} is below {WARN_JACOBIAN}")
    if stats["largest_aspect"] > WARN_ASPECT:
        out.append(f"{name}: largest aspect {stats['largest_aspect']:.3g} is above {WARN_ASPECT}")
    if stats["sliver_faces"]:
        out.append(
            f"{name}: {stats['sliver_faces']} sliver face(s) with a corner under {SLIVER_DEG:g} "
            f"or over {180 - SLIVER_DEG:g} degrees (corners from "
            f"{stats['smallest_corner_deg']:.3g} to {stats['largest_corner_deg']:.4g})"
        )
    return out


def _trailing(
    verts: NDArray[np.float64],
    faces: Faces,
    grid: Any,
    config: GridConfig,
    *,
    curves: dict[str, list[NDArray[np.float64]]],
    size: float,
    role: tuple[set[tuple[int, int]], int] | None = None,
) -> tuple[set[tuple[int, int]], list[str], int, str]:
    """Return the trailing edges a component asks for, what to warn, the rim edges dropped, source.

    ``"auto"`` takes the rim of the face a layout sidecar gives the trailing-edge role (``role``:
    its edges, already renumbered, and the edges of the other rim) and, when there is none, the
    geometric search; the source says which.
    """
    name = grid.name
    if grid.trailing_edge == "auto":
        used = {(min(a, b), max(a, b)) for f in faces for a, b in _edges(f)}
        if role is not None and role[0] & used:
            return role[0] & used, [], role[1], "layout role"
        angle = grid.te_angle_deg or config.te_angle_deg
        found, dropped = one_rim(verts, faces, trailing_edges(verts, faces, angle))
        said = (
            []
            if found
            else [f'{name}: trailing_edge = "auto" found no edge sharper than {angle:g} degrees']
        )
        if role is not None:
            said.append(
                f"{name}: the layout's trailing-edge face has no rim on the grid, "
                "so the search was used"
            )
        return found, said, dropped, "geometric search"
    lines: list[NDArray[np.float64]] = []
    said = []
    for curve in grid.trailing_edge:
        if curve in curves:
            lines += curves[curve]
        else:
            said.append(f"{name}: trailing_edge names {curve}, and no IGES curve level is named so")
    found = edges_on(verts, faces, lines, _ON_CURVE_SHARE * size)
    if lines and not found:
        said.append(f"{name}: no edge of the grid lies on {', '.join(grid.trailing_edge)}")
    return found, said, 0, "named curves"


#: The characters a file name cannot hold on any platform the package runs on.
_NOT_IN_A_FILE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _te_path(target: Path, name: str, taken: set[str]) -> Path:
    """Return ``<stem>.<component>.te.txt`` beside ``target``, the name made a file name.

    Each character a file name cannot hold becomes ``_``; a name another component already took
    that way gets ``_2``, ``_3``, ... Names are compared without case, as the file systems of
    Windows and macOS compare them, so ``Wing`` and ``wing`` never share one file. The boundaries
    file names the file, so the component keeps its own name.
    """
    safe = _NOT_IN_A_FILE_NAME.sub("_", name)
    held = {other.casefold() for other in taken}
    path, n = target.with_name(f"{target.stem}.{safe}.te.txt"), 2
    while path.name.casefold() in held:
        path, n = target.with_name(f"{target.stem}.{safe}_{n}.te.txt"), n + 1
    return path


def _write(
    target: Path,
    model: Any,
    config: GridConfig,
    order: list[str],
    sizes: dict[str, float],
    *,
    meshed: Any,
    default: float,
    curves: dict[str, list[NDArray[np.float64]]],
    selection: dict[str, Any] | None = None,
    junction: tuple[Callable[[str, NDArray[np.float64]], float], float] | None = None,
) -> dict[str, Any]:
    joined = meshed.record["fragment"]["warned"]
    warnings = list(model.notes) + list(meshed.warnings)
    families, removed, skin = _inner(config, meshed, order, warnings)
    verts, families, boundary, rims = _compact(
        meshed.nodes, families, meshed.boundary, meshed.te_edges
    )
    flips: dict[str, int] = {}
    for name in families:
        families[name], flips[name] = orient(verts, families[name])
    every = [f for faces in families.values() for f in faces]
    check = conformal(every, boundary)
    junctions = [] if junction is None else junction_checks(verts, families, *junction)
    per: dict[str, Any] = {}
    te: dict[str, set[tuple[int, int]]] = {}
    dropped: dict[str, int] = {}
    sources: dict[str, str] = {}
    for name, faces in families.items():
        stats = quality(verts, faces)
        stats["least_scaled_jacobian"] = meshed.jacobian.get(name)
        stats["size"] = sizes[name]
        stats["reoriented_faces"] = flips[name]
        stats["inner_faces_removed"] = int(removed.get(name, {}).get("faces", 0))
        stats["inner_area_removed"] = float(removed.get(name, {}).get("area", 0.0))
        per[name] = stats
        warnings += _quality_warnings(name, stats)
        grid = config.components.get(name)
        if grid is not None and grid.trailing_edge is not None:
            found, said, dropped[name], source = _trailing(
                verts,
                faces,
                grid,
                config,
                curves=curves,
                size=sizes[name],
                role=(rims[name], meshed.te_dropped.get(name, 0)) if name in rims else None,
            )
            sources[name] = source
            warnings += said
            if found:
                te[name] = found
    warnings += junction_warnings(junctions)
    if check["conformal_mismatches"]:
        warnings.append(
            f"{check['conformal_mismatches']} open edge(s) are not boundaries of the body: two "
            "faces meet there without sharing nodes, or a face is missing"
        )
    # The join's warnings were said before the mesh; the record keeps every one.
    notes = len(model.notes)
    for message in warnings[:notes] + warnings[notes + joined :]:
        warn(f"iges-grid: {message}", GeometryWarning)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = obj_text(ObjMesh(verts, families), note=f"cfd-mesh iges-grid, unit {model.unit}")
    _textio.write_text(target, text)
    te_files: dict[str, str] = {}
    for name, edges in te.items():
        path = _te_path(target, name, set(te_files.values()))
        write_te(path, model.unit, midpoints(verts, edges))
        te_files[name] = path.name
    counts = [len(f) for f in families.values()]
    lines = [
        f"boundaries = [{', '.join(_toml_key(n) for n in families)}]",
        f"mesh_faces = {sum(counts)}",
        f"boundary_faces = [{', '.join(str(c) for c in counts)}]",
        f"mesh_sha256 = {_toml_key(file_sha256(target))}",
        "",
        "[import]",
        f"units = {_toml_key(model.unit)}",
    ]
    if te_files:
        for name, file in te_files.items():
            lines += [
                "",
                f"[trailing_edges.{_toml_key(name)}]",
                f"file = {_toml_key(file)}",
                'type = "STANDARD"',
            ]
    else:
        lines += ["", "[trailing_edges]", "none = true"]
    sidecar = Path(str(target) + ".boundaries.toml")
    _textio.write_lines(sidecar, lines)
    record = {
        "output": target.name,
        "sha256": file_sha256(target),
        "unit": model.unit,
        "sources": model.sources,
        "defaults_size": default,
        "nodes": len(verts),
        "faces": sum(counts),
        "components": per,
        "conformal": check,
        "junctions": junctions,
        "selection": selection,
        "trailing_edges": {
            name: {
                "file": te_files[name],
                "edges": len(te[name]),
                "dropped_rim_edges": dropped.get(name, 0),
                "source": sources[name],
            }
            for name in te_files
        },
        **meshed.record,
        "inner_faces": skin,
        "symmetry": config.symmetry,
        "warnings": warnings,
    }
    _textio.write_text(
        Path(str(target) + ".json"), json.dumps(record, indent=2, default=str) + "\n"
    )
    return record
