"""Write a level: the orchestration behind ``cfd-mesh refine`` and ``refine_mesh``..

The order, from the source to the level folder:

1. read the mesh, the request (:mod:`._config`) and the ``[[perimeter]]``
   node lists that orient the grids and give the trailing edges (:mod:`._lines`);
2. recover the grid of every family asked to change (a sheet, a tube, or a
   multiblock grid of quadrilaterals, :mod:`._blocks`) and decide its
   method, refusing before any work;
3. refine the grid families first, in the source's family order; grids that
   share nodes are resampled as one layout, so both hold the same nodes on
   every curve they share, and a grid that cannot join one is remeshed when
   it shares nodes with a grid refined before it (:mod:`._blocks`);
4. hand each neighbour the grid's new nodes on their shared curve: a remeshed
   neighbour is rebuilt on them, an unchanged one is remeshed at
   factor 1 in a band of two face layers, or whole when the band
   would hold every face of it; a family remeshed at factor 1 takes,
   along and near that curve, the length of the new shared edges;
5. remesh the other families, together where they touch;
6. assemble in the source's family order, weld the shared nodes, put each
   band back into its family in the source's order, group the components, write the level into a
   temporary folder, audit it there, and rename it once complete; an existing level is replaced
   only
   then.

The boundaries TOML and the ``[[perimeter]]`` node lists are read in step 1,
before any resampling.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import os
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

import cfdbender._textio as _textio
from cfdbender._errors import (
    GeometryWarning,
    InputArtifactError,
    warn,
)
from cfdbender._lengths import scale as length_scale
from cfdbender.mesh._refine import (
    _blocks,
    _config,
    _geometry,
    _grid,
    _lines,
    _obj,
)
from cfdbender.mesh._refine._audit import MeshAudit, audit_mesh
from cfdbender.mesh._refine._config import GRID_AUTO, FamilySpec, RefineRequest
from cfdbender.mesh._refine._geometry import NearestIndex, order_like
from cfdbender.mesh._refine._obj import Faces, ObjMesh
from cfdbender.mesh._refine._staging import (
    PREVIOUS_SUFFIX,
    _place,
    _publish,
    _release,
    staging_prefix,
)
from cfdbender.mesh._sidecars import TRAILING_EDGES_TABLE, ensure_inventory, inventory_sidecar
from cfdbender.mesh._te_points import quad_families, write_level_trailing_edges

Points = NDArray[numpy.float64]

#: The suffix of a band family while it is remeshed; it never reaches the level.
BAND_SUFFIX = "~band"
#: The trailing-edge points file of a mesh, when its boundaries file names none.
TE_SUFFIX = _obj.TE_SUFFIX
#: Version 2 names native-unit target lengths explicitly when mesh units are absent.
REFINE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RefinedMesh:
    """What a refinement wrote: the level folder, its files and the report of each family.

    Attributes
    ----------
    folder : pathlib.Path
        The level folder ``<out-dir>/<stem>_<tag>/``.
    obj : pathlib.Path
        The level's OBJ.
    files : tuple of pathlib.Path
        Every file written, the OBJ first.
    report : dict
        The ``families`` entry of ``refine.json``: per family, the method,
        the reason, the layout and the counts before and after.
    audit : MeshAudit or None
        The audit written beside the level, or None when audit=False.
    """

    folder: Path
    obj: Path
    files: tuple[Path, ...]
    report: dict[str, Any]
    audit: MeshAudit | None = None


@dataclass
class _Work:
    """The state a refinement carries from one step to the next."""

    source: Path
    mesh: ObjMesh
    request: RefineRequest
    te_unit: str  # the unit line of the level's points files: the mesh's unit
    te_points: Points  # the mid-points of the exported [[perimeter]] edges, in the mesh's unit
    te_listed: bool  # whether a [[perimeter]] list exports trailing edges
    names: list[str]
    original: dict[str, Faces]
    fam_v: dict[str, set[int]]
    te_of: dict[str, set[int]]
    scale: float
    grids: dict[str, _grid.Grid | _blocks.Blocks] = field(default_factory=dict)
    remesh: list[str] = field(default_factory=list)
    report: dict[str, dict[str, Any]] = field(default_factory=dict)
    parts: dict[str, tuple[Points, dict[str, Faces], Points]] = field(default_factory=dict)
    conform: dict[str, list[tuple[set[int], Points]]] = field(default_factory=dict)
    # the entries of conform whose grid is a refined side
    shared: dict[str, list[tuple[set[int], Points]]] = field(default_factory=dict)
    # the unchanged neighbours a band would have covered whole, and their record
    rebuilt: dict[str, dict[str, Any]] = field(default_factory=dict)
    # the layout of each multiblock family and of each grid resampled with others
    layouts: dict[str, _blocks.Layout] = field(default_factory=dict)
    # the progress line callback of refine_mesh; silent by default
    say: Callable[[str], None] = field(default=lambda _line: None)
    bands: dict[str, str] = field(default_factory=dict)
    # The families oriented by the longest side of their boundary, a reference line that is
    # not written as a trailing edge (_lines.assume_trailing_edge).
    assumed: set[str] = field(default_factory=set)
    # The grid_auto families and the governing family selected by the planner.
    followers: dict[str, str] = field(default_factory=dict)
    component_governors: dict[str, dict[str, str]] = field(default_factory=dict)
    audit_enabled: bool = True
    remesh_reason: str = "the refinement file states remesh"
    sidecar: str | None = None  # the level's boundaries file, rewritten before any work
    unchanged: list[str] = field(default_factory=list)  # the families G6 compares
    # the selected families of a summary tag, taken with the tag before any work
    tag_selection: list[str] | None = None
    # The working copy of the request's specs: a factor-1 family leaves it, a band or a
    # rebuilt family joins it; the request itself, which refine.json records, never changes.
    specs: dict[str, FamilySpec] = field(default_factory=dict)
    te_edges: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    # The source edges, as segment ends in the mesh's unit, of the [[perimeter]] lists
    # marked export_te_per_family and of the others; None without [[perimeter]].
    te_export: tuple[tuple[Points, Points], tuple[Points, Points]] | None = None
    # Per source family, the mid-points of its edges on an exported [[perimeter]] list,
    # which refine.json records for the audit's G4 (the source has no points file).
    te_perimeter: dict[str, Points] = field(default_factory=dict)
    # The source edges of each [[perimeter]] list, the exported ones first: a family's
    # line is chosen from them in that order (_choose_perimeter_line).
    perimeter_sets: list[set[tuple[int, int]]] = field(default_factory=list)
    # Per source family, its own rearmost listed edges (the edges its surface lies forward
    # of): the line its trailing edge is taken from, and the only one written for it.
    te_rear: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    # The same two sets of source edges as te_export, the exported and the others.
    te_sets: tuple[set[tuple[int, int]], set[tuple[int, int]]] | None = None

    @property
    def te_tolerance(self) -> float:
        """Return how close a trailing-edge point lies to the mid-point of the edge it names."""
        return self.request.point_fraction * self.scale


def _refuse(context: str, reason: str, *, where: dict | None = None) -> InputArtifactError:
    return InputArtifactError(
        f"{context}: {reason}. Nothing was written.", kind=_obj.KIND, where=where
    )


# --------------------------------------------------------------------- reading


def _start(source: Path, request: RefineRequest, mesh: ObjMesh) -> _Work:
    # A grid's parallel and normal factors need no trailing edge: a family without one
    # takes the longest side of its boundary as reference (_lines.assume_trailing_edge,
    # _blocks.te_directions), which is not written as a trailing edge.
    perimeter = _lines.perimeter_lines(source, request, mesh)
    # The level's points files are in the mesh's unit, as the boundaries file states it.
    unit = _obj.mesh_unit(source) or "METER"
    # The mid-points of every listed edge orient the grids (te_of, te_edges); only the
    # exported ones are the level's trailing edge (te_points).
    points = _obj.midpoints(mesh.verts, perimeter[0]) if perimeter else numpy.zeros((0, 3))
    scale = mesh.size
    tolerance = request.point_fraction * scale
    names = list(mesh.families)
    return _Work(
        source=source,
        mesh=mesh,
        request=request,
        te_unit=unit,
        te_points=_obj.midpoints(mesh.verts, perimeter[1]) if perimeter else points,
        te_listed=bool(perimeter and perimeter[1]),
        names=names,
        original={n: list(fs) for n, fs in mesh.families.items()},
        fam_v={n: set(mesh.family_vertices(n).tolist()) for n in names},
        te_of={n: _lines.te_vertices(mesh, n, points, tolerance) for n in names},
        scale=scale,
        specs={n: request.specs.get(n, FamilySpec(method="keep")) for n in names},
        te_edges={n: _lines.listed_edges(mesh, n, points, tolerance) for n in names},
        perimeter_sets=perimeter[2] if perimeter else [],
        te_rear=perimeter[3] if perimeter else {},
        te_sets=(perimeter[1], perimeter[0] - perimeter[1]) if perimeter else None,
        te_perimeter=_lines.exported_by_family(mesh, perimeter[1]) if perimeter else {},
        te_export=None
        if not perimeter
        else (
            _lines.segments(mesh.verts, perimeter[1]),
            _lines.segments(mesh.verts, perimeter[0] - perimeter[1]),
        ),
    )


# ------------------------------------------------------------------- deciding


def plan_prepared(source, nodes, components, kinds, te_families, *, keep_within_component=False):
    """Plan prepared patches with the exact OBJ and node list written for refinement."""
    request = RefineRequest(
        specs={n: FamilySpec(method="grid" if k == "grid" else "remesh") for n, k in kinds.items()},
        components=components,
        perimeters=(_config.Perimeter(nodes, te_families=tuple(te_families)),),
        accept_remesh=True,
        keep_within_component=keep_within_component,
    )
    work = _start(source, request, _obj.read_obj(source))
    _decide(work, prepare=True)
    return work


def _decide(work: _Work, *, prepare: bool = False) -> None:
    """Recover the grids and choose each family's method, refusing before any work.."""
    names = [n for n in work.names if n in work.specs]
    for i, name in enumerate(names, 1):
        work.say(f"family {i} of {len(names)}: {name}, recovering its layout")
        _decide_family(work, name)
    _grid_after_grid(work, prepare=prepare)
    for name in work.grids:
        if work.specs[name].method == "keep" and name not in work.layouts:
            del work.specs[name]
            work.report[name] = {"method": "copied"}
    for name in work.remesh:
        spec = work.specs[name]
        if spec.chordwise is None and spec.spanwise is None:
            continue
        why = work.report.get(name, {}).get("reason", "no grid was recovered")
        if not work.request.accept_remesh:
            raise _refuse(
                f"{work.source} family {name}",
                f"parallel_factor and normal_factor apply to a grid, and this family is "
                f"remeshed ({why}); give factor, or accept the remesh with factor "
                f"{spec.factor:g} with --accept-remesh (refine_mesh: accept_remesh=True)",
                where={"families": [name]},
            )
        warn(
            f"{work.source} family {name}: parallel_factor and normal_factor apply to a grid "
            f"and are skipped; it is remeshed with factor {spec.factor:g} ({why})",
            GeometryWarning,
            stacklevel=2,
        )


def _refuse_or_remesh(work: _Work, name: str, found: str) -> None:
    """Refuse a family whose grid was demanded but not recovered, else choose its remesh.

    Under ``accept_remesh`` the demanded grid is warned and remeshed instead. A
    grid_auto family that is not a grid is kept unchanged.
    """
    spec = work.specs[name]
    if spec.method in (GRID_AUTO, "keep"):
        del work.specs[name]
        work.followers.pop(name, None)
        work.report[name] = {"method": "unchanged", "reason": f"{spec.method}, not a grid: {found}"}
        if spec.method == "keep":
            work.report[name] = {"method": "copied"}
        return
    if spec.method == "grid":
        _demanded_grid(work, name, f"method is grid and no grid was recovered: {found}")
    _remesh_or_keep(work, name, f"no grid recovered: {found}")


def _demanded_grid(work: _Work, name: str, why: str, *, where: dict | None = None) -> None:
    """Refuse a ``method = "grid"`` family that is no grid, or warn under ``accept_remesh``."""
    if not work.request.accept_remesh:
        raise _refuse(
            f"{work.source} family {name}",
            why,
            where=where or {"families": [name], "nodes": sorted(work.fam_v[name])},
        )
    warn(
        f"{work.source} family {name}: {why}; it is remeshed with factor "
        f"{work.specs[name].factor:g} (accept_remesh, CLI --accept-remesh)",
        GeometryWarning,
        stacklevel=4,
    )


def _decide_family(work: _Work, name: str) -> None:
    """Recover one selected family's grid, or choose its remesh, refusing before any work."""
    spec = work.specs[name]
    if spec.method == "remesh":
        _remesh_or_keep(work, name, work.remesh_reason)
        return
    directional = spec.chordwise is not None or spec.spanwise is not None
    if work.te_export is not None and work.te_edges[name]:
        _lines.choose_perimeter_line(work, name)
    if (spec.method in ("grid", GRID_AUTO, "keep") or directional) and not work.te_of[name]:
        # A grid demanded of a sheet, or asked for parallel and normal factors, that
        # found no trailing edge takes the longest side of its boundary as reference
        # (_lines.assume_trailing_edge), not written as a trailing edge, and warns.
        _lines.assume_trailing_edge(work, name)
    found = _blocks.recover_family(work.mesh.verts, work.mesh.families[name], work.te_of[name])
    if not isinstance(found, str) and work.te_edges[name]:
        found = _lines.grid_with_complete_te_edges(work, name, found)
    if isinstance(found, str):
        _refuse_or_remesh(work, name, found)
        return
    if not isinstance(found, _blocks.Blocks):
        chordwise, spanwise = spec.directions
        if work.request.force_factors:
            chordwise, spanwise = _forced_factors(work, name, found, chordwise, spanwise)
        _grid.check_factors(found, name, chordwise=chordwise, spanwise=spanwise)
    work.grids[name] = found


def _forced_factors(
    work: _Work, name: str, grid: _grid.Grid, chordwise: float, spanwise: float
) -> tuple[float, float]:
    """Raise a grid's factors to the least it allows, warn, and keep them in its spec.

    ``force_factors`` (the default; CLI ``--strict-factors`` refuses): the level's counts
    differ from the factors asked, which ``refine.json`` keeps under ``specs``; the
    family's record shows the intervals written.
    """
    least = _grid.least_factors(grid, chordwise, spanwise)
    if least == (chordwise, spanwise):
        return chordwise, spanwise
    warn(
        f"{work.source} family {name}: the normal and parallel factors {chordwise:g} and "
        f"{spanwise:g} are below the least the grid allows and are raised to {least[0]:g} "
        f"and {least[1]:g}; pass --strict-factors (force_factors=False) to refuse it instead",
        GeometryWarning,
        stacklevel=4,
    )
    work.specs[name] = dataclasses.replace(work.specs[name], chordwise=least[0], spanwise=least[1])
    return least


def _te_rows(work: _Work, points: Points) -> list[list[float]]:
    """Return trailing-edge mid-points as the level's points file writes them ( G4).

    Each coordinate is the decimal the points file writes
    (:func:`_obj.coordinates`), so the audit reads in ``refine.json`` the
    line it would read in that file.
    """
    return [[float(w) for w in _obj.coordinates(p).split()] for p in points]


def _grid_after_grid(work: _Work, *, prepare: bool = False) -> None:
    """Resample grids that share nodes as one layout; remesh or refuse one that cannot join.

    (:func:`._blocks.plan`).
    """
    work.layouts, late = _blocks.plan(
        work.mesh.verts,
        work.mesh.families,
        work.grids,
        work.specs,
        work.te_edges,
        where=str(work.source),
        accept=work.request.accept_remesh,
        followers=work.followers,
        assumed=work.assumed,
        prepare=prepare,
        components=work.request.components if work.request.keep_within_component else None,
        component_governors=work.component_governors,
    )
    # Followers use the layout's propagated axes and per-arc minimum counts.
    # Their independently recovered Grid axes may be rotated; checking those
    # here can refuse valid counts or change factors after the plan is final.
    for name, first, why in late:
        if first == name:  # its own layout was refused and accepted as a remesh
            reason = f"its layout cannot be refined as asked{why}"
            demanded = f"method is grid and {reason}"
        else:
            reason = f"shares nodes with the grid {first}{why}"
            demanded = (
                f"method is grid and it shares nodes with {first}, refined as a grid first{why}"
            )
        if work.specs[name].method == "grid":
            _demanded_grid(
                work,
                name,
                demanded,
                where={
                    "families": list(dict.fromkeys((name, first))),
                    "nodes": sorted(work.fam_v[name] & work.fam_v[first]),
                },
            )
        del work.grids[name]
        _remesh_or_keep(work, name, reason)
    work.remesh.sort(key=work.names.index)


def _remesh_or_keep(work: _Work, name: str, reason: str) -> None:
    """Remesh a family, or keep it unchanged when every factor it is given is 1.

    A remeshed family at factor 1 is an UNCHANGED FAMILY: it is
    copied as the source holds it, and a band of it is remeshed only where a
    grid beside it changed their shared curve. A family the
    refinement file states ``method = "remesh"`` is remeshed at any factor.
    """
    spec = work.specs[name]
    if spec.unchanged and spec.method in ("auto", "keep"):
        del work.specs[name]
        work.report[name] = {"method": "unchanged", "reason": f"factor 1 ({reason})"}
        return
    work.remesh.append(name)
    work.report[name] = {"method": "remesh", "reason": reason}


def _refine_grids(work: _Work) -> None:
    """Refine every grid family and record what its neighbours must take."""
    levels: dict[str, _grid.GridLevel] = {}
    for i, (name, grid) in enumerate(work.grids.items(), 1):
        if name not in work.specs:
            continue
        chordwise, spanwise = work.specs[name].directions
        work.say(f"grid {i} of {len(work.grids)}: {name} ({grid.describe()})")
        layout = work.layouts.get(name)  # every multiblock family has one
        mates = layout.names if layout is not None else (name,)
        if layout is not None or not isinstance(grid, _grid.Grid):
            if name not in levels:
                levels.update(
                    _blocks.refine_layout(work.mesh.verts, work.mesh.families, work.layouts[name])
                )
            level = levels[name]
        else:
            level = _grid.refine_grid(
                work.mesh.verts,
                work.mesh.families[name],
                grid,
                chordwise=chordwise,
                spanwise=spanwise,
                family=name,
            )
        info = _grid_info(work, name, grid, level, layout)
        interfaces: dict[str, str] = {}
        for other in work.names:
            old = work.fam_v[name] & work.fam_v[other] if other not in mates else set()
            new = _interface_points(work, name, old, level) if old else None
            if new is None or _unchanged_curve(work, old, new):
                continue
            work.conform.setdefault(other, []).append((old, new))
            work.shared.setdefault(other, []).append((old, new))
            interfaces[other] = f"{len(old)} -> {len(new)} nodes"
            if other not in work.specs:
                continuity = work.report.setdefault(other, {}).setdefault(
                    "continuity", {"band": "remeshed", "because": []}
                )
                continuity["because"].append(name)
        if interfaces:
            info["interfaces"] = interfaces
        work.report[name] = info
        if name == mates[0]:  # a layout's families are one part, under its first family
            fams = {n: levels[n].faces for n in mates} if layout else {name: level.faces}
            listed = set().union(*(work.te_edges[n] for n in mates))
            every = [f for fs in fams.values() for f in fs]
            mids = _lines.on_listed(work, level.te_midpoints, listed, level.points, every)
            work.parts[name] = (level.points, fams, mids)


def _grid_info(
    work: _Work,
    name: str,
    grid: _grid.Grid | _blocks.Blocks,
    level: _grid.GridLevel,
    layout: _blocks.Layout | None,
) -> dict[str, Any]:
    """Return a grid family's refine.json entry, before its interfaces."""
    info: dict[str, Any] = dict(level.report, method="grid", reason=grid.describe())
    if layout is not None:
        info["kept_sides"] = layout.kept[name]
        if name in layout.continuity:
            info["continuity"] = layout.continuity[name]
        if name in layout.unchanged:
            info["method"] = "unchanged"
    if name in work.followers:  # grid_auto
        directions = (
            [d for p in layout.patches_of(name) for d in layout.directions[p].values()]
            if layout is not None
            else []
        )
        sources = list(dict.fromkeys(d.source_family for d in directions))
        fallback = next(
            (d.provenance for d in directions if d.provenance.startswith("own_orientation_")), None
        )
        direction = (
            "own reference: " + fallback.removeprefix("own_orientation_")
            if fallback
            else "from " + ", ".join(sources)
            if sources
            else None
        )
        if direction is None:
            direction = "own reference: unlinked"
            _blocks.warn_direction_fallback(
                name, work.followers[name], "unlinked", where=str(work.source)
            )
        info["grid_auto"] = {
            "follows": work.followers[name],
            "factors": list(work.specs[name].directions),
            "directions": direction,
        }
    if work.request.elements_of(name) != _config.TRIANGLES:
        info["elements"] = (
            f"{work.request.elements_of(name)} ignored: it applies to a remeshed family, "
            "and a grid keeps its cells"
        )
    return info


def _interface_points(
    work: _Work, name: str, old: set[int], level: _grid.GridLevel
) -> Points | None:
    """Return the grid's new boundary nodes on the curve it shares with a neighbour.

    The curve is the grid's source boundary edges whose two ends are shared
    (``old``). A new boundary node is on it when it lies within
    :data:`._geometry.ON_CURVE_FRACTION` of a segment's length from one of
    those edges AND one of its own boundary edges runs along the curve: both
    ends within :data:`._geometry.ALONG_FRACTION` of the EDGE's length from it (an edge
    leaving the curve at a right angle has its far end one length away). A node of another
    boundary curve next to the curve's end, closer to it than a quarter of a
    curve segment, is no node of it: its edges run away from the curve.
    """
    source_edges = [
        e for e in _geometry.boundary_edges(work.mesh.families[name]) if e[0] in old and e[1] in old
    ]
    if not source_edges:
        return None
    a = work.mesh.verts[[e[0] for e in source_edges]]
    b = work.mesh.verts[[e[1] for e in source_edges]]
    length = numpy.linalg.norm(b - a, axis=1)
    ab = b - a

    def distance(p: Points) -> tuple[float, int]:
        t = numpy.clip(
            ((p - a) * ab).sum(axis=1) / numpy.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0
        )
        d = numpy.linalg.norm(a + t[:, None] * ab - p, axis=1)
        j = int(numpy.argmin(d))
        return float(d[j]), j

    edges = _geometry.boundary_edges(level.faces)
    near = {}
    for v in sorted({v for e in edges for v in e}):
        d, j = distance(level.points[v])
        if d < _geometry.ON_CURVE_FRACTION * length[j]:
            near[v] = d
    along: set[int] = set()
    for u, w in edges:
        if u in near and w in near:
            span = float(numpy.linalg.norm(level.points[u] - level.points[w]))
            if max(near[u], near[w]) < _geometry.ALONG_FRACTION * span:
                along.update((u, w))
    keep = [v for v in near if v in along]
    return level.points[keep] if keep else None


def _unchanged_curve(work: _Work, old: set[int], new: Points) -> bool:
    """Return whether the grid kept exactly the source nodes of a shared curve."""
    olds = work.mesh.verts[sorted(old)]
    tolerance = _geometry.DUPLICATE_FRACTION * work.scale
    return len(new) == len(olds) and float(NearestIndex(olds).query(new)[0].max()) < tolerance


def _bands(work: _Work) -> None:
    """Cut a band of face layers out of each unchanged neighbour whose curve changed.

    A band that would hold every face of the family leaves no remainder: the
    family is rebuilt whole at factor 1 instead, as if stated ``factor = 1``,
    ``method = "remesh"``, with one warning and its record.
    """
    for name in [n for n in work.names if n in work.conform and n not in work.specs]:
        seed = set().union(*(old for old, _ in work.conform[name]))
        faces = work.mesh.families[name]
        band = band_of_faces(faces, seed, _geometry.BAND_LAYERS)
        if len(band) == len(faces):
            _rebuild_whole(work, name, len(band))
            continue
        bname = name + BAND_SUFFIX
        families: dict[str, Faces] = {}
        for n, fs in work.mesh.families.items():
            if n == name:
                families[name] = [f for i, f in enumerate(fs) if i not in band]
                families[bname] = [f for i, f in enumerate(fs) if i in band]
            else:
                families[n] = fs
        work.mesh.families = families
        work.names = list(families)
        for n in (name, bname):
            work.fam_v[n] = set(work.mesh.family_vertices(n).tolist())
            work.te_of[n] = _lines.te_vertices(work.mesh, n, work.te_points, work.te_tolerance)
            work.te_edges[n] = _lines.listed_edges(work.mesh, n, work.te_points, work.te_tolerance)
        work.specs[bname] = FamilySpec(factor=1.0, method="remesh")
        work.remesh.append(bname)
        work.conform[bname] = work.conform.pop(name)
        if name in work.shared:
            work.shared[bname] = work.shared.pop(name)
        work.bands[bname] = name
        work.report[name] = {
            **work.report.get(name, {}),
            "method": "unchanged",
            "band": len(band),
            "faces": len(work.original[name]),
        }


def _rebuild_whole(work: _Work, name: str, band: int) -> None:
    """Remesh a neighbour the band covers whole at factor 1, warn and record it."""
    work.specs[name] = FamilySpec(factor=1.0, method="remesh")
    work.remesh.append(name)
    work.remesh.sort(key=work.names.index)
    faces = len(work.original[name])
    work.rebuilt[name] = {
        "method": "rebuilt_whole",
        "reason": "band_covers_family",
        "band": band,
        "faces": faces,
    }
    work.report[name] = {**work.report.get(name, {}), **work.rebuilt[name]}
    warn(
        f"{work.source} family {name}: the band of {_geometry.BAND_LAYERS} face layers beside "
        f"the refined grid covers all {faces} faces of the family, so it was rebuilt whole at "
        'factor 1, as if stated factor = 1, method = "remesh"',
        GeometryWarning,
        stacklevel=4,  # warn's caller is _rebuild_whole: past _bands and refine_mesh, the user
    )


def band_of_faces(faces: Faces, seed: set[int], layers: int) -> set[int]:
    """Return the faces within ``layers`` face layers of the seed vertices."""
    vertices, selected = set(seed), set()
    for _ in range(layers):
        new = [
            i for i, f in enumerate(faces) if i not in selected and any(v in vertices for v in f)
        ]
        selected.update(new)
        vertices |= {v for i in new for v in faces[i]}
    return selected


def _remesh_groups(work: _Work) -> None:
    """Remesh the other families, together where they share nodes."""
    if not work.remesh:
        return
    from cfdbender.mesh._refine._remesh import element_report, refine_group

    groups: list[list[str]] = []
    for name in work.remesh:
        hit = [g for g in groups if any(_together(work, name, m) for m in g)]
        merged = [name] + [m for g in hit for m in g]
        groups = [g for g in groups if g not in hit] + [merged]
    for i, group in enumerate(groups, 1):
        ordered = [n for n in work.names if n in group]
        work.say(
            f"remesh {i} of {len(groups)}: {', '.join(ordered)} "
            f"({sum(len(work.original.get(n, ())) for n in ordered)} source faces)"
        )
        te = set().union(*(work.te_of[n] for n in ordered))
        factors = {n: work.specs[n].factor for n in ordered}
        conform = [c for n in ordered for c in work.conform.get(n, [])]
        modes = {n: work.request.elements_of(work.bands.get(n, n)) for n in ordered}
        shared = [c for n in ordered for c in work.shared.get(n, [])]
        done = refine_group(
            work.mesh,
            ordered,
            te,
            factors,
            conform=conform,
            elements=modes,
            shared=shared,
            te_edges=set().union(*(work.te_edges[n] for n in ordered)),
        )
        at_curve = done.info.get("shared_edge_target_mesh_units", {})
        unit = _obj.mesh_unit(work.source)
        to_metres = None if unit is None else length_scale(unit, "METER")
        target_key = (
            "shared_edge_target_mesh_units" if to_metres is None else "shared_edge_target_m"
        )
        if to_metres is not None:
            at_curve = {
                n: {key: value * to_metres for key, value in measures.items()}
                for n, measures in at_curve.items()
            }
        info = {k: v for k, v in done.info.items() if k != "shared_edge_target_mesh_units"}
        for n in ordered:
            entry = work.report.setdefault(n, {"method": "remesh"})
            entry.update(
                info,
                faces_before=len(work.mesh.families[n]),
                faces_after=len(done.families.get(n, [])),
                **element_report(modes[n], done.families.get(n, [])),
            )
            entry.update(work.rebuilt.get(n, {}))
            if n in at_curve and n not in work.bands:  # , under the rebuilt family
                entry[target_key] = at_curve[n]
        for n in ordered:
            if n in work.bands and n in at_curve:  # , under the band's family
                work.report[work.bands[n]][target_key] = at_curve[n]
        work.parts[ordered[0]] = (done.points, done.families, done.trailing_edge)


def _together(work: _Work, name: str, other: str) -> bool:
    """Return whether two remeshed families are remeshed in one group: they share nodes."""
    return bool(work.fam_v[name] & work.fam_v[other])


# ------------------------------------------------------------------ assembling


def _assemble(work: _Work) -> tuple[Points, dict[str, Faces], Points]:
    """Assemble the parts in the source's family order; families not asked are copied unchanged."""
    blocks: list[Points] = []
    faces: dict[str, Faces] = {}
    te: list[Points] = []
    offset = 0
    copied_te = numpy.zeros(len(work.te_points), dtype=bool)
    changed = set(work.specs)
    for name in work.names:
        if name in work.parts:
            pts, fams, mids = work.parts[name]
            blocks.append(pts)
            faces.update({n: [[v + offset for v in f] for f in fs] for n, fs in fams.items()})
            offset += len(pts)
            if len(mids) and name not in work.assumed:
                te.append(mids)
        elif name not in changed:
            ids = sorted(work.fam_v[name])
            local = {v: i + offset for i, v in enumerate(ids)}
            blocks.append(work.mesh.verts[ids])
            faces[name] = [[local[v] for v in f] for f in work.mesh.families[name]]
            offset += len(ids)
            hit = _te_points_of(work, name) & ~copied_te
            if hit.any():
                te.append(work.te_points[hit])
                copied_te |= hit
    verts, faces = _weld(
        numpy.vstack(blocks), {n: faces[n] for n in work.names if n in faces}, work.scale
    )
    for band, family in work.bands.items():
        faces[family] = order_like(
            verts, faces[family] + faces.pop(band), work.mesh.verts, work.original[family]
        )
    midpoints = _distinct(numpy.vstack(te) if te else numpy.zeros((0, 3)), work.te_tolerance)
    verts, faces = _source_numbering(verts, faces, work.mesh.verts, work.scale)
    return verts, faces, midpoints


def _distinct(points: Points, tolerance: float) -> Points:
    """Return the trailing-edge mid-points with each edge's named once, the first kept.

    A grid whose trailing-edge nodes did not change and an unchanged family
    holding the same edges both write them; the workflow refuses a points file
    naming one edge twice, so a point within ``tolerance`` (the request's
    ``point_fraction`` of the size) of one kept before is dropped.
    """
    kept: list[int] = []
    for i, point in enumerate(points):
        if not kept or float(numpy.linalg.norm(points[kept] - point, axis=1).min()) >= tolerance:
            kept.append(i)
    return points[kept]


def _source_numbering(
    verts: Points, faces: dict[str, Faces], source: Points, scale: float
) -> tuple[Points, dict[str, Faces]]:
    """Renumber the level's nodes the way the source numbers them.

    A node at a source node's position takes that node's place in the source's
    order; the new nodes follow, in their order of first use. A level that
    keeps every source node, a grid at factor 1, is written with the source's
    vertex list.
    """
    dist, index = _geometry.NearestIndex(source).query(verts)
    kept = dist <= _geometry.DUPLICATE_FRACTION * scale
    key = numpy.where(kept, index, len(source) + numpy.arange(len(verts)))
    order = numpy.argsort(key, kind="stable")
    rank = numpy.empty(len(verts), dtype=numpy.int64)
    rank[order] = numpy.arange(len(verts))
    renumbered = {n: [[int(rank[v]) for v in f] for f in fs] for n, fs in faces.items()}
    return verts[order], renumbered


def _source_text(work: _Work, verts: Points) -> list[str | None]:
    """Return, per level node, the source's coordinate text when the node is a source node.

    A node equal in every coordinate to the source node at its position is
    written as the source wrote it, so a node the refinement kept is the
    source's in its text too; any other node is written by
    :func:`._obj.coordinates`.
    """
    if not len(verts) or len(work.mesh.vertex_text) != len(work.mesh.verts):
        return []
    dist, index = NearestIndex(work.mesh.verts).query(verts)
    same = (dist == 0.0) & numpy.all(work.mesh.verts[index] == verts, axis=1)
    return [work.mesh.vertex_text[int(i)] if s else None for i, s in zip(index, same, strict=True)]


def _te_points_of(work: _Work, name: str) -> NDArray[numpy.bool_]:
    """Return which listed trailing-edge points lie on an edge of the family."""
    if not len(work.te_points):
        return numpy.zeros(0, dtype=bool)
    _, mid = _obj.edge_midpoints(work.mesh.verts, work.mesh.families[name])
    dist, _ = NearestIndex(mid).query(work.te_points)
    return dist < work.te_tolerance


def _weld(
    verts: Points, families: dict[str, Faces], scale: float
) -> tuple[Points, dict[str, Faces]]:
    """Weld nodes that coincide within DUPLICATE_FRACTION of the size; drop unused ones.

    The nodes kept are numbered in their order of first use.
    """
    tolerance = max(_geometry.DUPLICATE_FRACTION * scale, 1e-300)
    root = _geometry.weld_roots(verts, tolerance)
    welded = {n: [[int(root[v]) for v in f] for f in fs] for n, fs in families.items()}
    return _obj.compact(verts, welded)


def _components(work: _Work, faces: dict[str, Faces]) -> dict[str, Faces]:
    """Write each component's members as one family, in the source's family order."""
    holder = {m: c for c, members in work.request.components.items() for m in members}
    out: dict[str, Faces] = {}
    for name in [n for n in work.names if n not in work.bands]:
        out.setdefault(holder.get(name, name), []).extend(faces[name])
    return out


# --------------------------------------------------------------------- writing


def _boundaries_text(work: _Work, stem: str) -> str | None:
    """Return the source's declarations with only component names changed.

    Inventory values are refreshed when written. Declaration formatting and
    comments are preserved, and every component rewrite is checked by parsing.

    Raises
    ------
    InputArtifactError
        A value the rewrite cannot place; nothing was written. It is called
        before any family is resampled.
    """
    read = _obj.read_sidecar(work.source)
    if read is None:
        return None
    side, data = read
    text = side.read_bytes().decode("utf-8")
    holder = {m: c for c, members in work.request.components.items() for m in members}
    listed = data.get("boundaries")
    if holder and isinstance(listed, list) and all(isinstance(n, str) for n in listed):
        names = list(dict.fromkeys(holder.get(n, n) for n in listed))
        changed = copy.deepcopy(data)
        changed["boundaries"] = names
        text = _obj.rewrite_value(side, text, "boundaries", listed, changed)
        data = changed
    # F-1492: a detection selector (``detect = { surfaces = [...] }``) names families too,
    # and a grouped level no longer holds the members it names.
    paths: list[tuple[str, ...]] = [
        (name,) for name in (TRAILING_EDGES_TABLE, "wake_termination", "base_regions")
    ]
    trailing = data.get(TRAILING_EDGES_TABLE, {})
    if isinstance(trailing, dict):
        paths += [(TRAILING_EDGES_TABLE, label) for label in trailing]
    for path in paths:
        table = data
        for key in path:
            table = table.get(key, {}) if isinstance(table, dict) else {}
        detect = table.get("detect") if isinstance(table, dict) else None
        surfaces = detect.get("surfaces") if isinstance(detect, dict) else None
        if holder and isinstance(surfaces, list) and all(isinstance(n, str) for n in surfaces):
            names = list(dict.fromkeys(holder.get(n, n) for n in surfaces))
            if names != surfaces:
                changed = copy.deepcopy(data)
                entry = changed
                for key in path:
                    entry = entry[key]
                entry["detect"]["surfaces"] = names
                text = _obj.rewrite_value(side, text, "surfaces", surfaces, changed, value=names)
                data = changed
    return _obj.base_component_names(side, text, data, holder)


def _direction_record(layout: _blocks.Layout | None, name: str) -> dict[str, Any]:
    """Count distinct retained arc orientations; conflicting local records remain distinct."""
    records = (
        {(a, d) for p in layout.patches_of(name) for a, d in layout.directions[p].items()}
        if layout is not None
        else set()
    )
    provenance: dict[str, int] = {}
    for _, direction in records:
        key = direction.provenance
        provenance[key] = provenance.get(key, 0) + 1
    return {
        "directions": {
            d: sum(told.direction == d for _, told in records) for d in ("parallel", "normal")
        },
        "provenance": dict(sorted(provenance.items())),
    }


def _layout_record(layout: _blocks.Layout) -> dict[str, Any]:
    """Record each count band's arcs and the lead selected by the planner."""
    bands: dict[int, list[int]] = {}
    for arc, band in _blocks.arc_bands(layout.blocks, range(len(layout.blocks.arcs))).items():
        bands.setdefault(band, []).append(arc)
    return {
        "families": list(layout.names),
        "bands": [
            {
                "id": band,
                "arcs": arcs,
                "lead": layout.band_leads[band][0],
                "direction": layout.band_leads[band][1],
                "label": layout.labels[arcs[0]],
                "dominant_arc": layout.leads[layout.labels[arcs[0]]][1],
                "count": layout.counts[arcs[0]],
            }
            for band, arcs in sorted(bands.items())
        ],
    }


def _record(work: _Work, stem: str, counts: Mapping[str, int], has_te: bool) -> dict[str, Any]:
    # every selected family and its factor, a factor-1 family kept unchanged
    # included; the method each got is in "families".
    specs = {
        n: {k: v for k, v in vars(s).items() if v is not None}
        for n, s in work.request.specs.items()
    }
    shortened = {} if work.tag_selection is None else {"tag_selection": work.tag_selection}
    return {
        "schema_version": REFINE_SCHEMA_VERSION,
        "source": work.source.name,
        "level": stem,
        "config": None if work.request.config is None else work.request.config.name,
        # The source names no points file: its trailing edges are [[perimeter]] lists.
        "trailing_edge_files": {
            "source": None,
            "level": f"{stem}{TE_SUFFIX}" if has_te else None,
        },
        "trailing_edge_point_fraction": work.request.point_fraction,
        "specs": specs,
        "components": work.request.components,
        "ignored_families_tables": list(work.request.ignored),
        "missing_families_tables": list(work.request.missing),
        "unchanged_grids": list(work.unchanged),
        "families": {
            n: {
                **_with_perimeter(work, n, work.report.get(n, {"method": "copied"})),
                **(
                    _direction_record(work.layouts.get(n), n)
                    if n in work.grids and n in work.specs
                    else {}
                ),
            }
            for n in work.names
            if n not in work.bands
        },
        "layouts": [
            _layout_record(layout)
            for layout in {lay.names: lay for lay in work.layouts.values()}.values()
        ],
        "faces": dict(counts),
        **shortened,
        **({"audit": "skipped (--audit-off)"} if not work.audit_enabled else {}),
        **({"component_governors": work.component_governors} if work.component_governors else {}),
    }


def _with_perimeter(work: _Work, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Return a family's record with its source trailing edge when an exported list gives it.

    A grid's line, completed where the list left a node out, is the one recorded.
    """
    if name not in work.te_perimeter or entry.get("trailing_edge", "perimeter") != "perimeter":
        return entry
    points = work.te_perimeter[name]
    if name in work.grids and work.te_edges.get(name):
        points = _obj.midpoints(work.mesh.verts, work.te_edges[name])
    return {**entry, "trailing_edge": "perimeter", "trailing_edge_points": _te_rows(work, points)}


#: The type each per-component trailing-edge file of a level's boundaries file is given,
#: the one a detection gives every edge it marks; the user edits it there.
DEFAULT_TE_TYPE = "STANDARD"


def _with_te_files(text: str, files: Mapping[str, Path]) -> str:
    """Return a boundaries file declaring each component's trailing-edge file.

    One ``[trailing_edges.<component>]`` table per file, the form
    the sidecar readers accept, each given :data:`DEFAULT_TE_TYPE` for the user to
    edit before marking. It is called only when the source declares none.
    """
    lines = text.rstrip("\n").splitlines()
    lines += [
        "",
        "# The level's trailing edges, one points file per component, written by the refine.",
        "# Edit each type (STANDARD or VORTEX_SHEDDING) before using the file.",
    ]
    for name, path in files.items():
        key = name if name.replace("_", "").replace("-", "").isalnum() else f'"{name}"'
        lines += [
            f"[{TRAILING_EDGES_TABLE}.{key}]",
            f'file = "{path.name}"',
            f'type = "{DEFAULT_TE_TYPE}"',
        ]
    out = "\n".join(lines) + "\n"
    tomllib.loads(out)
    return out


def _warn_stale_declarations(work: _Work, mesh: ObjMesh, folder: Path) -> None:
    """Warn when a preserved marking points file cannot mark the refined mesh."""
    if work.sidecar is None:
        return
    declarations = tomllib.loads(work.sidecar).get("trailing_edges", {})
    if not isinstance(declarations, dict):
        return
    faces = [face for group in mesh.families.values() for face in group]
    _, midpoints = _obj.edge_midpoints(mesh.verts, faces)
    for entry in declarations.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            continue
        named = folder / entry["file"]
        try:
            points = _obj.points_from_file(work.source, named)
            assert points is not None
            distance, _ = NearestIndex(midpoints).query(points.points)
            tolerance = float(entry.get("tolerance", 0.0001))
            conversion = length_scale(_obj.mesh_unit(work.source) or "METER", "METER")
            stale = not len(points.points) or bool(
                numpy.any(distance * (conversion or 1.0) > tolerance)
            )
        except (InputArtifactError, OSError, ValueError, TypeError):
            stale = True
        if stale:
            warn(
                _obj.stale_declaration_message(entry["file"], folder),
                GeometryWarning,
                stacklevel=3,
            )


def _quad_te_files(
    work: _Work,
    folder: Path,
    verts: Points,
    faces: dict[str, Faces],
    te: Points,
    *,
    stems: Mapping[str, str],
) -> list[Path]:
    """Write the quad families' trailing-edge node lists of the level, all and per family.

    The edges of each quad family whose mid-points lie on the level's trailing
    edge go to ``<level>.nodes.txt`` and ``<level>.<family>.nodes.txt``, node lists
    (:func:`cfdbender.mesh._te_points.write_level_trailing_edges`).
    """
    if not len(te):
        return []
    index = NearestIndex(numpy.asarray(te, dtype=float))
    picked = any(p.te_families for p in work.request.perimeters)
    found: dict[str, set[tuple[int, int]]] = {}
    for name in quad_families(faces):
        edges, mid = _obj.edge_midpoints(verts, faces[name])
        near = index.query(mid)[0] < work.te_tolerance
        if picked and work.te_rear.get(name):
            # The family's own rearmost line alone, as its <level>.<family>.te.txt.
            pts = numpy.asarray(verts, dtype=float)
            near &= _lines.on_line(
                work, mid, pts[edges[:, 0]], pts[edges[:, 1]], work.te_rear[name]
            )
        on = {(int(a), int(b)) for (a, b), hit in zip(edges, near, strict=True) if hit}
        if on:
            found[stems[name]] = on
    written = write_level_trailing_edges(verts, found, folder, folder.name)
    _obj.restore_node_names(written, dict(stems))
    return written


class _NoInspection:
    """The refinement's inspection hook when no visual inspection is installed: it does nothing."""

    def __init__(self, source: Path, obj: ObjMesh, enabled: bool) -> None:
        if enabled:
            raise _refuse(
                str(source),
                "the visual inspection PDF is not available in this installation; "
                "run again without it",
            )
        self.work: Any = None

    def __enter__(self) -> _NoInspection:
        return self

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        return None

    def write(self, writer: Any, work: Any, *args: Any) -> Any:
        return writer(work, *args)


def _inspection(source: Path, obj: ObjMesh, enabled: bool) -> Any:
    """Return the inspection hook: the visual one when its module is installed, else a no-op."""
    try:
        module = importlib.import_module("cfdbender.mesh._inspect_pages")
    except ModuleNotFoundError:
        return _NoInspection(source, obj, enabled)
    return module.Inspection(source, obj, enabled)


def _write(
    work: _Work, folder: Path, verts: Points, faces: dict[str, Faces], te: Points
) -> list[Path]:
    """Write the level's files into ``folder`` and return them, the OBJ first."""
    stem = folder.name
    every = [f for fs in faces.values() for f in fs]
    if work.te_export is not None:
        te = _lines.on_exported(work, te, verts, every)
    stems = _obj.output_stems(list(faces)) if len(te) else {n: n for n in faces}
    mesh = ObjMesh(
        verts, faces, list(work.mesh.header), work.mesh.family_tag, _source_text(work, verts)
    )
    note = f"refined by cfdbender from {work.source.name}"
    written = [_obj.write_obj(mesh, folder / f"{stem}.obj", note)]
    # The level's trailing edges under the level's name: the exported [[perimeter]]
    # lines, else the trailing edge the level holds (an assumed one).
    if work.te_listed or len(te):
        target = folder / f"{stem}{TE_SUFFIX}"
        written.append(_obj.write_te(target, work.te_unit, te))
    own: dict[str, Path] = {}
    if work.te_export is not None:
        own = _lines.own_te_files(work, folder, verts, faces, te, stems=stems)
    # The trailing edges a [[perimeter]] list exports are declared per component in the
    # level's boundaries file, for the user to type, unless the
    # source's boundaries file declares its own, which are kept (and warned when stale).
    declared = work.sidecar is not None and TRAILING_EDGES_TABLE in tomllib.loads(work.sidecar)
    listed = own if work.te_export is not None and not declared else {}
    target = inventory_sidecar(written[0], "obj")
    if work.sidecar is not None:  # its face counts and digest are the level OBJ's
        side = inventory_sidecar(work.source)
        text = _obj.level_inventory(side, work.sidecar, written[0])
        _textio.write_text(target, _with_te_files(text, listed) if listed else text)
        written.append(target)
    elif listed:
        written.append(ensure_inventory(written[0], kind="obj"))
        _textio.write_text(target, _with_te_files(target.read_text(encoding="utf-8"), listed))
    if not listed:
        _warn_stale_declarations(work, mesh, folder)
    written += own.values()
    written += _quad_te_files(work, folder, verts, faces, te, stems=stems)
    record = folder / f"{stem}.refine.json"
    data = _record(
        work, stem, {n: len(fs) for n, fs in faces.items()}, work.te_listed or bool(len(te))
    )
    renamed = {n: s for n, s in stems.items() if n != s}
    if renamed:
        data["renamed_family_outputs"] = renamed
    _textio.write_json(
        record,
        data,
        indent=1,
    )
    written.append(record)
    return written


# ---------------------------------------------------------------------- entry


def _renamed_keyword(
    old: str, old_value: float | None, new: str, value: float | None
) -> float | None:
    """Return a factor given under its earlier keyword, warning that it was renamed."""
    if old_value is None:
        return value
    if value is not None:
        warn(f"refine_mesh: {old} is skipped; {new} is given", GeometryWarning, stacklevel=3)
        return value
    warn(f"refine_mesh: {old} is read as {new}; pass {new}", GeometryWarning, stacklevel=3)
    return old_value


def refine_mesh(  # noqa: PLR0913 (one keyword per option of the command)
    mesh: str | os.PathLike[str],
    factor: float | None = None,
    *,
    families: Sequence[str] | None = None,
    parallel: float | None = None,
    normal: float | None = None,
    config: str | Path | None = None,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
    keep_first_row: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
    accept_remesh: bool = False,
    accept_unknown: bool = False,
    force_factors: bool = True,
    visual_inspect: bool = False,
    keep_within_component: bool | None = None,
    audit: bool = True,
    remesh: bool = False,
    chordwise: float | None = None,
    spanwise: float | None = None,
) -> RefinedMesh | None:
    """Refine or coarsen a panel mesh from its OBJ into a new geometry folder.

    Parameters
    ----------
    mesh : path-like
        The source OBJ, kept unchanged. A refusal may write a diagnostic PDF beside it.
    factor : float, optional
        The factor of every selected family; without it, and without
        ``parallel`` or ``normal``, the factors come from the refinement file.
    families : sequence of str, optional
        The families to change (default: every family).
    parallel, normal : float, optional
        The factor of every selected grid family along its reference perimeter
        (``parallel``) and across it (``normal``), each replacing ``factor`` there.
    config : path-like, optional
        The refinement file (default: ``<stem>.refine.toml`` beside the mesh).
    out_dir : path-like, optional
        Where the level folder goes (default: the parent of the source's folder).
    overwrite : bool
        Replace an existing level folder.
    keep_first_row : sequence of str, optional
        An earlier option, warned and skipped when given: a row kept at an
        interface is a family of its own.
    progress : callable, optional
        Called with one line per stage and per family as the work goes (read,
        decide, grid, remesh, write, audit), so a long refinement shows it is
        moving; the command prints them. None says nothing.
    accept_remesh : bool
        Remesh with ``factor``, with a warning, a family whose grid is not
        recovered although it states ``parallel_factor`` or ``normal_factor``
        or ``method = "grid"``; by default such a family is refused.
    accept_unknown : bool
        Warn and skip, instead of refusing, an entry the refinement does not
        know or cannot read: an unknown table or key of the refinement file,
        a value it cannot read (the key takes its default), an implicit
        refinement file that cannot be read, a family table or component
        member the mesh does not hold, and a selected family it does not hold.
    force_factors : bool
        Raise a grid factor below the least its grid allows (``1/m`` for its
        ``m`` intervals, the least a tube's caps carry) to that least, with a
        warning (the default); False refuses such a factor (CLI ``--strict-factors``).
    visual_inspect : bool
        Write ``<level>.refine-governors.pdf`` beside the level OBJ: a summary page, one page
        per governing family and direction, then the patches and directions pages.
    keep_within_component : bool, optional
        Share directional governors across each declared component. None (the default) reads
        ``keep_within_component`` under ``[refine]`` of the refine file; True or False overrides it.
        Without components the existing layout behavior is retained.
    audit : bool
        Audit the written level against its source. False skips the audit and records
        that choice; the level can be audited later with geometry audit.
    remesh : bool
        Remesh every selected family, including structured grids at factor 1. Element
        modes, markings, interfaces, audits and output publication follow refinement.
    chordwise, spanwise : float, optional
        The earlier names of ``normal`` and ``parallel``, read with a warning;
        beside the new name, the earlier one is warned and skipped.

    Returns
    -------
    RefinedMesh or None
        The level folder, its files, the per-family report and the audit.
        None, with an EmptySelectionWarning and no level written, when no family is selected.

    Raises
    ------
    InputArtifactError
        A refinement refusal. After reading the source, a diagnostic PDF is attempted.

    Examples
    --------
    >>> from cfdbender.mesh._refine import refine_mesh
    >>> level = refine_mesh("geometries/W1/W1.obj", 2)  # doctest: +SKIP
    >>> level.folder.name  # doctest: +SKIP
    'W1_R2'
    """
    source = Path(mesh)
    if not source.is_file():
        raise _refuse(str(source), "no such OBJ file; name an existing OBJ and run again")
    obj = _obj.read_obj(source)
    with _inspection(source, obj, visual_inspect) as inspection:
        parallel = _renamed_keyword("spanwise", spanwise, "parallel", parallel)
        normal = _renamed_keyword("chordwise", chordwise, "normal", normal)
        request = _config.resolve_request(
            source,
            list(obj.families),
            factor=factor,
            families=[families] if isinstance(families, str) else families,
            parallel=parallel,
            normal=normal,
            config=config,
            keep_first_row=keep_first_row,
            accept_remesh=accept_remesh,
            accept_unknown=accept_unknown,
            force_factors=force_factors,
            keep_within_component=keep_within_component,
        )
        if request is None:
            # No family selected: warned by the request, and nothing is written.
            return None
        if remesh:
            request = dataclasses.replace(
                request,
                specs={
                    name: dataclasses.replace(spec, method="remesh")
                    for name, spec in request.specs.items()
                },
            )
        stem = f"{source.stem}_{_config.level_tag(request, list(obj.families), str(source))}"
        if request.filename:
            if request.tag:
                warn(
                    f"{request.config} [refine] states filename and tag; the level is named "
                    f"{request.filename!r} and the tag is skipped",
                    GeometryWarning,
                    stacklevel=2,
                )
            stem = request.filename
        folder = _place(source, stem, out_dir, overwrite)
        work = inspection.work = _start(source, request, obj)
        if remesh:
            work.remesh_reason = "the call requested remesh"
        work.audit_enabled = audit
        if progress is not None:
            work.say = progress
        work.say(
            f"read {source.name}: {len(obj.families)} families, "
            f"{sum(len(f) for f in obj.families.values())} faces; level {stem}"
        )
        work.tag_selection = _config.tag_selection(request, list(obj.families))
        work.sidecar = _boundaries_text(work, stem)
        _decide(work)
        work.say(f"decided: {len(work.grids)} grid, {len(work.remesh)} remeshed")
        _refine_grids(work)
        work.say("bands and shared curves")
        _bands(work)
        _remesh_groups(work)
        work.say("assembling the level")
        verts, faces, te = _assemble(work)
        faces = _components(work, faces)
        work.unchanged = _unchanged_grids(work, list(faces))
        folder.parent.mkdir(parents=True, exist_ok=True)
        # The run's own folder: created here under a unique name, and the only one it removes.
        owned = Path(tempfile.mkdtemp(prefix=staging_prefix(folder.name), dir=folder.parent))
        try:
            staging = owned / folder.name
            staging.mkdir()
            work.say(f"writing {len(verts)} nodes, {sum(len(f) for f in faces.values())} faces")
            written = inspection.write(_write, work, staging, verts, faces, te)
            result = _audit_level(work, written[0], folder)
            _publish(staging, folder, owned / f"{folder.name}{PREVIOUS_SUFFIX}", overwrite)
        finally:
            _release(owned, folder)
        if result is not None:
            written.append(result.path)
            result = dataclasses.replace(
                result, mesh=folder / written[0].name, path=folder / result.path.name
            )
        files = tuple(folder / p.name for p in written)
        report = json.loads((folder / f"{stem}.refine.json").read_text(encoding="utf-8"))[
            "families"
        ]
        return RefinedMesh(folder=folder, obj=files[0], files=files, report=report, audit=result)


def _audit_level(work: _Work, obj: Path, folder: Path) -> MeshAudit | None:
    """Audit a staged level when requested, otherwise name the later audit command."""
    if not work.audit_enabled:
        work.say(
            f'audit skipped (--audit-off); run cfd-mesh audit "{folder / obj.name}" '
            f'--against "{work.source}"'
        )
        return None
    work.say("auditing the level against its source")
    audit = audit_mesh(obj, against=work.source, unchanged_grids=work.unchanged)
    if audit.degenerate_faces:  # what no repair could reach is warned, never refused
        warn(
            f"{folder / obj.name}: {len(audit.degenerate_faces)} degenerate face(s) are left "
            f"(an area below {_geometry.DEGENERATE_AREA_FRACTION:g} of their family's median "
            f"or an angle below {_geometry.DEGENERATE_ANGLE_DEGREES:g} degree, on nodes a remesh "
            "may not move or in a family not remeshed); each is listed with its family, face "
            f"index and centroid under degenerate_faces in {folder / audit.path.name}; inspect "
            "those faces, or re-cut the source near them (a node on a boundary or shared with "
            "another family is not moved by refine)",
            GeometryWarning,
            stacklevel=3,
        )
    return audit


def _unchanged_grids(work: _Work, families: list[str]) -> list[str]:
    """Return the level's families G6 compares with the source's.

    A grid family written at factor 1, and a component whose every member is
    one, in the level's family order. ``refine.json`` lists them, so the
    audit of a saved level judges G6 as the refinement's own audit does.
    """
    ones = {
        n
        for n in work.grids
        if work.specs.get(n, FamilySpec()).directions == (1.0, 1.0)
        and "continuity" not in work.report.get(n, {})
    }
    components = work.request.components
    return [
        n
        for n in families
        if (n in components and all(m in ones for m in components[n]))
        or (n not in components and n in ones)
    ]
