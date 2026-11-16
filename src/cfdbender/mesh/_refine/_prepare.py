"""Prepare an OBJ for the refinement: families split into patches, a refine file and a node list.

``cfd-mesh prepare-to-refine`` (optional; the refinement reads
any OBJ and refine file without it) writes, beside the mesh:

1. ``<stem>_patches.obj``: every family cut into PATCHES, named
   ``<family>_<NNN>_grid`` or ``<family>_<NNN>_mixed``. A family is first cut
   along its sharp edges (:data:`._geometry.RIDGE_DEGREES`); in each piece, the
   quadrilaterals joined edge to edge whose layout the refinement recovers as a
   grid (a sheet, a tube or a multiblock grid, :mod:`._blocks`) are a grid patch
   each, and the other faces, joined edge to edge, a mixed patch each. The
   original families keep their order; within each, the patches are numbered
   from the largest extent along y down, and each patch keeps its faces in the
   source's order.
2. ``<stem>_patches.refine.toml``: the planner's declared patches, ``method = "grid"``,
   ``"remesh"`` or ``"auto"`` at factor 1 for the user to edit. Directional leads
   carry governing flags; fully propagated followers have grid_auto tables. It uses
   the quad-dominant remesh,
   ``[components]`` grouping each family's patches back into the family, and a
   ``[[perimeter]]`` naming the node list with ``te_families``.
3. ``<stem>_patches.nodes.txt``: the node list the refinement reads (an ANSA SET
   node-window export, ``Id, Name, X, Y, Z``): for each grid patch, the nodes of
   the rearmost line of its outer boundary (:func:`._obj.rearmost_edges`) whose
   edges lie in or near the plane across the aft direction (the yz plane for
   +x), within ``max_angle`` (:func:`near_yz`),
   ordered along y. A node two patches share is listed
   once.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._errors import (
    GeometryWarning,
    InputArtifactError,
    warn,
)
from cfdbender.mesh._refine import _blocks, _geometry, _level, _lines, _nastran, _obj
from cfdbender.mesh._refine._obj import Faces, ObjMesh

Points = NDArray[numpy.float64]

#: The suffix of the prepared files' stem.
PATCHES_SUFFIX = "_patches"
#: The largest angle, in degrees, between a trailing-edge edge and the yz plane when none
#: is stated: a trailing edge runs across the flow (along y on a wing, along z on a fin),
#: at its sweep from that plane, while a tip or a root line runs along the flow.
MAX_ANGLE = 45.0
#: A quadrilateral piece of fewer faces than this is not a grid patch: the islands of
#: quadrilaterals a quad-dominant mesh holds among its triangles stay in their mixed patch.
GRID_MIN_FACES = 4


@dataclass(frozen=True)
class PreparedMesh:
    """What prepare-to-refine wrote: the patches OBJ, the refine file, the node list, the patches.

    ``patches`` maps each original family to its patch names in order.
    """

    obj: Path
    config: Path
    nodes: Path
    patches: dict[str, list[str]]
    #: The grid patches governing either direction.
    governing: tuple[str, ...] = ()
    #: The visual-inspect PDF, one page per original family, when asked for.
    inspect: Path | None = None


def prepare_to_refine(
    mesh: str | Path,
    *,
    max_angle: float = MAX_ANGLE,
    aft: tuple[float, float, float] = _obj.AFT,
    overwrite: bool = False,
    unit: str = "MILLIMETER",
    obj_unit: str = "METER",
    visual_inspect: bool = False,
    keep_within_component: bool = False,
    out_dir: str | Path | None = None,
) -> PreparedMesh:
    """Split a mesh's families into patches and write its refine file and node list.

    The mesh is an OBJ, whose families are cut at their sharp edges, or an ANSA
    NASTRAN export (``.nas``, ``.bdf``, ``.dat``) written with the FE perimeters
    on, whose families are cut along those perimeters, so each patch is one of
    the pre-processor's meshing macros (:mod:`._nastran`).

    Parameters
    ----------
    mesh : str or pathlib.Path
        The source OBJ or NASTRAN export; it is never modified.
    max_angle : float
        The largest angle, in degrees, between a listed edge and the plane
        across ``aft`` (default :data:`MAX_ANGLE`).
    aft : tuple of float
        The direction from the nose to the tail (default +x).
    overwrite : bool
        Replace the prepared files of an earlier run.
    unit : str
        A NASTRAN source's length unit (default MILLIMETER); read only for one.
    obj_unit : str
        The length unit a NASTRAN source is written in (default METER).
    visual_inspect : bool
        Also write ``<stem>_patches.inspect.pdf``: one page per original family,
        each patch colored and named, with GOVERNING PARALLEL and GOVERNING NORMAL labels.
    keep_within_component : bool
        Select one governor per direction across each original family's grid patches,
        including disconnected layouts, and persist the option under ``[refine]``.
    out_dir : str or pathlib.Path, optional
        Folder for prepared files, created when needed. Default: beside the source.

    Returns
    -------
    PreparedMesh
        The three files written and the patches of each family.

    Raises
    ------
    InputArtifactError
        The mesh cannot be read, or a prepared file exists and ``overwrite`` is false.
    """
    source = Path(mesh)
    aft = _obj.aft_direction(aft, str(source))
    cuts: frozenset[tuple[int, int]] | None = None
    if _nastran.is_nastran(source):
        read = _nastran.read_nastran(source, unit=unit, obj_unit=obj_unit)
        obj, cuts = read.mesh, (read.perimeter or None)
    else:
        obj = _obj.read_obj(source)
    stem = source.stem + PATCHES_SUFFIX
    folder = source.parent if out_dir is None else Path(out_dir)
    targets = [
        folder / f"{stem}.obj",
        folder / f"{stem}.refine.toml",
        folder / f"{stem}.nodes.txt",
    ]
    if visual_inspect:
        targets.append(folder / f"{stem}.inspect.pdf")
    existing = [t for t in targets if t.exists()]
    if existing and not overwrite:
        raise InputArtifactError(
            f"{existing[0]}: exists; pass overwrite (CLI: --overwrite) to replace the prepared "
            "files. Nothing was written.",
            kind=_obj.KIND,
        )
    families: dict[str, Faces] = {}
    kinds: dict[str, str] = {}
    patches: dict[str, list[str]] = {}
    for name, faces in obj.families.items():
        split = (
            _patches(obj.verts, faces) if cuts is None else _macro_patches(obj.verts, faces, cuts)
        )
        found = sorted(split, key=lambda p: -_extent_y(obj.verts, faces, p[1]))
        patches[name] = []
        for number, (kind, members) in enumerate(found, start=1):
            patch = f"{name}_{number:03d}_{kind}"
            families[patch] = [faces[i] for i in sorted(members)]
            kinds[patch] = kind
            patches[name].append(patch)
    rows, listed = _te_rows(obj.verts, families, kinds, max_angle, aft)
    note = f"patches of {source.name}, prepared by cfdbender prepare-to-refine"
    out = ObjMesh(obj.verts, families, list(obj.header), obj.family_tag, list(obj.vertex_text))
    folder.mkdir(parents=True, exist_ok=True)
    _obj.write_obj(out, targets[0], note)
    _textio.write_text(targets[2], "\n".join(["#  Id, Name, X , Y, Z", *rows]) + "\n")
    te_families = [n for n, ps in patches.items() if any(p in listed for p in ps)]
    planned = _level.plan_prepared(
        targets[0],
        targets[2],
        patches,
        kinds,
        te_families,
        keep_within_component=keep_within_component,
    )
    governing, followers, excluded, methods = _declarations(planned, kinds, keep_within_component)
    _textio.write_text(
        targets[1],
        _toml(
            source,
            targets[2],
            patches,
            kinds,
            te_families,
            governing=governing,
            followers=followers,
            excluded=excluded,
            methods=methods,
            layouts=planned.layouts,
            keep_within_component=keep_within_component,
        ),
    )
    inspect = None
    if visual_inspect:
        try:
            patches_pdf = importlib.import_module("cfdbender.mesh._visual_inspect").patches_pdf
        except ModuleNotFoundError:
            raise InputArtifactError(
                f"{source}: the visual inspection PDF is not available in this "
                "installation; run again without it. Nothing was written.",
                kind=_obj.KIND,
            ) from None

        unit_name = obj_unit.upper() if cuts is not None or _nastran.is_nastran(source) else None
        inspect = patches_pdf(
            targets[0], patches, governing, targets[3], mesh_unit=unit_name, layouts=planned.layouts
        )
    return PreparedMesh(
        targets[0], targets[1], targets[2], patches, tuple(sorted(governing)), inspect
    )


def _declarations(planned, kinds, keep_within_component):
    """Read prepared tables and fallback comments from the completed planner result."""
    governing = {
        n: s.leads for n, s in planned.specs.items() if n in planned.grids and any(s.leads)
    }
    followers: set[str] = set()
    excluded: dict[str, tuple[str, str]] = {}
    for name, layout in planned.layouts.items():
        directions = [d for p in layout.patches_of(name) for d in layout.directions[p].values()]
        if directions and all(d.provenance.startswith("propagated_from:") for d in directions):
            followers.add(name)
        reasons = {d.provenance for d in directions if d.provenance.startswith("own_orientation_")}
        if reasons:
            leads = ", ".join(n for n in layout.names if n in governing)
            excluded[name] = (
                leads,
                ", ".join(sorted(r.removeprefix("own_orientation_") for r in reasons)),
            )
    methods = {n: "grid" if n in planned.grids else "auto" for n in kinds if kinds[n] == "grid"}
    if keep_within_component:
        followers.update(n for n, s in planned.specs.items() if s.method == "grid_auto")
    return governing, followers, excluded, methods


def _patches(verts: Points, faces: Faces) -> list[tuple[str, list[int]]]:
    """Return a family's patches as (kind, face indices): its grids first, then the rest."""
    dihedral = _geometry.dihedral_degrees(verts, faces)
    sharp = {e for e, angle in dihedral.items() if angle > _geometry.RIDGE_DEGREES}
    holders = _geometry.edge_faces(faces)
    quads = {i for i, f in enumerate(faces) if len(f) == 4}
    out: list[tuple[str, list[int]]] = []
    taken: set[int] = set()
    for piece in _joined(quads, holders, sharp, quads):
        if len(piece) >= GRID_MIN_FACES and _is_grid(verts, [faces[i] for i in piece]):
            out.append(("grid", piece))
            taken.update(piece)
    rest = set(range(len(faces))) - taken
    for piece in _joined(rest, holders, sharp, rest):
        out.append(("mixed", piece))
    return out


def _macro_patches(
    verts: Points, faces: Faces, cuts: frozenset[tuple[int, int]]
) -> list[tuple[str, list[int]]]:
    """Return a family's patches cut along the given edges; a grid where one is recovered."""
    holders = _geometry.edge_faces(faces)
    every = set(range(len(faces)))
    out: list[tuple[str, list[int]]] = []
    for piece in _joined(every, holders, set(cuts), every):
        quads = all(len(faces[i]) == 4 for i in piece)
        grid = quads and len(piece) >= GRID_MIN_FACES and _is_grid(verts, [faces[i] for i in piece])
        out.append(("grid" if grid else "mixed", piece))
    return out


def _joined(
    members: set[int],
    holders: dict[tuple[int, int], list[int]],
    sharp: set[tuple[int, int]],
    allowed: set[int],
) -> list[list[int]]:
    """Return the members joined through edges that are not sharp, each piece once."""
    neighbours: dict[int, set[int]] = {i: set() for i in members}
    for edge, fs in holders.items():
        if edge in sharp:
            continue
        inside = [f for f in fs if f in allowed]
        if len(inside) == 2:
            a, b = inside
            neighbours[a].add(b)
            neighbours[b].add(a)
    pieces: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(members):
        if start in seen:
            continue
        stack, piece = [start], {start}
        while stack:
            for n in neighbours[stack.pop()] - piece:
                piece.add(n)
                stack.append(n)
        seen |= piece
        pieces.append(sorted(piece))
    return pieces


#: The share of a perimeter line's edges found as trailing edge above which the
#: whole line is taken: the edges near its ends that the angle tests miss, where
#: a blade's trailing edge bends into its junction with the hub, are the same
#: line. A line is cut where the perimeter branches, at a corner of a grid
#: patch and where it turns by more than :data:`LINE_CORNER_DEGREES`, so a tip
#: or a root line stays apart.
LINE_MAJORITY = 0.5

#: The turn, in degrees, that cuts a perimeter line: a trailing edge meets a tip
#: or a root line at about a right angle, while its bend into a fillet is gentler.
LINE_CORNER_DEGREES = 60.0


def rearmost_lines(
    verts: Points,
    faces: Faces,
    aft: tuple[float, float, float] = _obj.AFT,
    max_angle: float = MAX_ANGLE,
) -> set[tuple[int, int]]:
    """Return a family's trailing edge found from the mesh alone.

    The family is cut along its sharp edges into pieces; a piece's perimeter is
    its boundary edges and the sharp edges inside it (a sharp trailing edge
    joins the upper and the lower surface of a wing whose leading edge is
    smooth). Of each piece, only the perimeter reaching furthest aft is read
    (:func:`._obj.rearmost_perimeter`), and of it the edges the piece lies
    forward of (:func:`._obj.rearmost_edges`) that lie in or near the plane
    across ``aft`` (the yz plane for +x), within ``max_angle`` (:func:`near_yz`).
    Where a chain of the edges found (edges sharing a vertex) holds a fold (the
    upper and the lower surface meeting at a sharp trailing edge,
    :func:`._obj.folded_edges`), only its folds are kept: a closed tip cap is a
    piece of its own, and the contour it meets the blade along, which joins the
    trailing edge at its tip, is not a trailing edge. A chain with no fold (the
    rim of a blunt base) keeps every edge, whatever else of the family touches it.
    A line of the perimeter (its boundary and sharp edges, cut where they branch
    or turn) more than :data:`LINE_MAJORITY` of whose edges are found is taken
    whole, to its corners.
    """
    dihedral = _geometry.dihedral_degrees(verts, faces)
    sharp = {e for e, angle in dihedral.items() if angle > _geometry.RIDGE_DEGREES}
    holders = _geometry.edge_faces(faces)
    every = set(range(len(faces)))
    found: set[tuple[int, int]] = set()
    for piece in _joined(every, holders, sharp, every):
        members = [faces[i] for i in piece]
        held = _geometry.edge_faces(members)
        perimeter = {e for e, fs in held.items() if len(fs) == 1 or e in sharp}
        rear = _obj.rearmost_perimeter(verts, members, perimeter, aft)
        found |= _obj.rearmost_edges(verts, members, rear, aft)
        # A fold is read on the whole family: a sharp leading edge leaves the upper and
        # the lower surface in two pieces, each holding the trailing edge on one face.
        folds = _obj.folded_edges(verts, faces, rear)
        found |= _obj.rearmost_edges(verts, faces, rear, aft) & folds
    folds = _obj.folded_edges(verts, faces, found)
    if folds:
        chain = _obj.vertex_components([list(e) for e in found])
        folded_chains = {chain[a] for a, _ in folds}
        found = {e for e in found if e in folds or chain[e[0]] not in folded_chains}
    feature = sharp | {e for e, fs in holders.items() if len(fs) == 1}
    kept = near_yz(verts, found, max_angle, aft)
    return whole_lines(verts, faces, feature, kept, max_angle=max_angle, aft=aft)


def whole_lines(
    verts: Points,
    faces: Faces,
    perimeter: set[tuple[int, int]],
    found: set[tuple[int, int]],
    *,
    max_angle: float = MAX_ANGLE,
    aft: tuple[float, float, float] = _obj.AFT,
) -> set[tuple[int, int]]:
    """Return ``found`` with every line of ``perimeter`` mostly found taken whole.

    A line is a chain of perimeter edges cut where the perimeter branches or
    turns (:func:`perimeter_lines`); when more than :data:`LINE_MAJORITY` of its
    edges are in ``found``, its edges within ``max_angle`` of the plane across
    ``aft`` (:func:`near_yz`) are added. A tip that continues the trailing edge
    at a gentle corner runs along the flow and stays out; the edges a trailing
    edge's bend leaves out of the rearmost tests come back within ``max_angle``.
    """
    out = set(found)
    for line in perimeter_lines(verts, faces, perimeter):
        if len(line & found) > LINE_MAJORITY * len(line):
            out |= near_yz(verts, line, max_angle, aft)
    return out


def perimeter_lines(
    verts: Points, faces: Faces, perimeter: set[tuple[int, int]]
) -> list[set[tuple[int, int]]]:
    """Return the lines of a family's perimeter edges.

    A line runs through a vertex that joins two of the edges, that more than one
    face holds (a vertex one face holds is a grid patch's corner) and where the
    line turns by :data:`LINE_CORNER_DEGREES` or less.
    """
    touching: dict[int, list[int]] = {}
    for a, b in perimeter:
        touching.setdefault(a, []).append(b)
        touching.setdefault(b, []).append(a)
    held: dict[int, int] = {}
    for face in faces:
        for v in face:
            held[v] = held.get(v, 0) + 1
    pts = numpy.asarray(verts, dtype=float)
    most = numpy.cos(numpy.radians(LINE_CORNER_DEGREES))
    through = {
        v
        for v, n in touching.items()
        if len(n) == 2 and held.get(v, 0) > 1 and _turn_cosine(pts, n[0], v, n[1]) >= most
    }
    lines: list[set[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for edge in sorted(perimeter):
        if edge in seen:
            continue
        line, stack = {edge}, [v for v in edge if v in through]
        while stack:
            v = stack.pop()
            for n in touching[v]:
                step = (min(v, n), max(v, n))
                if step not in line:
                    line.add(step)
                    stack += [n] if n in through else []
        seen |= line
        lines.append(line)
    return lines


def _turn_cosine(pts: Points, before: int, at: int, after: int) -> float:
    """Return the cosine of the turn of the path before -> at -> after."""
    u, w = pts[at] - pts[before], pts[after] - pts[at]
    return float(u @ w) / max(float(numpy.linalg.norm(u) * numpy.linalg.norm(w)), 1e-300)


def near_yz(
    verts: Points,
    edges: set[tuple[int, int]],
    max_angle: float = MAX_ANGLE,
    aft: tuple[float, float, float] = _obj.AFT,
) -> set[tuple[int, int]]:
    """Return the edges whose tangent makes ``max_angle`` degrees or less with the plane across aft.

    The plane is the one normal to ``aft``, the yz plane for the default +x: a
    trailing edge runs across the flow, whichever axis the flow runs along.
    """
    most = numpy.sin(numpy.radians(max_angle))
    axis = numpy.asarray(aft, dtype=float)
    axis = axis / float(numpy.linalg.norm(axis))
    out = set()
    for a, b in edges:
        tangent = verts[b] - verts[a]
        length = float(numpy.linalg.norm(tangent))
        if length > 0.0 and abs(float(tangent @ axis)) <= most * length:
            out.add((a, b))
    return out


def _is_grid(verts: Points, faces: Faces) -> bool:
    """Return whether the refinement recovers the quadrilaterals as a grid (a sheet by a side)."""
    found = _blocks.recover_family(verts, faces, ())
    if not isinstance(found, str):
        return True
    for side in sorted(_lines.boundary_sides(verts, faces), key=lambda s: -len(s)):
        if not isinstance(_blocks.recover_family(verts, faces, side), str):
            return True
    return False


def _extent_y(verts: Points, faces: Faces, members: list[int]) -> float:
    ys = verts[sorted({v for i in members for v in faces[i]}), 1]
    return float(ys.max() - ys.min())


def _te_rows(
    verts: Points,
    families: dict[str, Faces],
    kinds: dict[str, str],
    max_angle: float,
    aft: tuple[float, float, float],
) -> tuple[list[str], set[str]]:
    """Return the node-list rows of the grid patches' rearmost lines, and the patches listed."""
    rows: list[str] = []
    listed: set[str] = set()
    written: set[int] = set()
    for name, faces in families.items():
        if kinds[name] != "grid":
            continue
        held = _geometry.edge_faces(faces)
        ring = _obj.rearmost_perimeter(
            verts, faces, {e for e, fs in held.items() if len(fs) == 1}, aft
        )
        edges = near_yz(verts, _obj.rearmost_edges(verts, faces, ring, aft), max_angle, aft)
        nodes = sorted({v for e in edges for v in e} - written, key=lambda v: float(verts[v, 1]))
        if not nodes:
            continue
        listed.add(name)
        written.update(nodes)
        for v in nodes:
            x, y, z = (repr(float(c)) for c in verts[v])
            rows.append(f"{v + 1}, {name}, {x}, {y}, {z}")
    if not rows:
        warn(
            "prepare-to-refine: no grid patch has a rearmost line within "
            f"{max_angle:g} degrees of the plane across the aft direction; the node list is empty",
            GeometryWarning,
            stacklevel=3,
        )
    return rows, listed


def _toml(
    source: Path,
    nodes: Path,
    patches: dict[str, list[str]],
    kinds: dict[str, str],
    te_families: list[str],
    *,
    governing: dict[str, tuple[bool, bool]] | None = None,
    followers: set[str] | frozenset[str] = frozenset(),
    excluded: dict[str, tuple[str, str]] | None = None,
    methods: dict[str, str] | None = None,
    layouts=None,
    keep_within_component=False,
) -> str:
    """Return the refine file text: one table per patch, the components and the perimeter."""
    lines = _toml_header(source, keep_within_component)
    for members in patches.values():
        for patch in members:
            layout = (layouts or {}).get(patch)
            if patch in followers and layout is not None:
                sources = layout.governing_by_family.get(patch, layout.governing)
                lines.append(
                    f"# follows {sources['parallel']} (parallel) and {sources['normal']} (normal)"
                )
                lines += [f"[families.{_key(patch)}]", 'method = "grid_auto"', ""]
                continue
            if layout is not None and any((governing or {}).get(patch, ())):
                labels = [label for label, (lead, _) in layout.leads.items() if lead == patch]
                lines.append("# leads " + ", ".join(labels))
            method = (methods or {}).get(patch, "grid" if kinds[patch] == "grid" else "remesh")
            if method == "auto":
                lines.append(
                    "# The planner cannot refine this patch as a grid here; use auto's fallback."
                )
            if excluded and patch in excluded:
                lead, reason = excluded[patch]
                lines.append(f"# not a follower of {lead}: {reason}")
            lines += [f"[families.{_key(patch)}]", f'method = "{method}"', "factor = 1.0"]
            normal, parallel = (governing or {}).get(patch, (False, False))
            if parallel:
                lines.append("governing_parallel = true")
            if normal:
                lines.append("governing_normal = true")
            lines.append("")
    lines.append("[components]")
    for family, members in patches.items():
        listed = ", ".join(f'"{m}"' for m in members)
        lines.append(f"{_key(family)} = [{listed}]")
    lines += [
        "",
        "# The rearmost line of each grid patch, its edges near the yz plane; the",
        "# trailing edges of te_families are written per family into the level.",
        "[[perimeter]]",
        f'file = "{nodes.name}"',
        "te_families = [" + ", ".join(f'"{n}"' for n in te_families) + "]",
        "",
    ]
    return "\n".join(lines)


def _toml_header(source, keep_within_component):
    """Introduce the prepared refine table and its optional component mode."""
    lines = [
        f"# Refine file prepared by cfdbender prepare-to-refine from {source.name}.",
        "# Edit the factors (and the method) of each patch; the [components] put each",
        "# family's patches back together in the level.",
        "",
        "[refine]",
        'elements = "quad-dominant"',
        "",
    ]
    if keep_within_component:
        lines.insert(-1, "keep_within_component = true")
    return lines


def _key(name: str) -> str:
    """Return a TOML key: the name bare when it is one, else quoted."""
    bare = bool(name) and all(c.isascii() and (c.isalnum() or c in "_-") for c in name)
    return name if bare else '"' + name.replace('"', "'") + '"'
