"""Rebuild the named hub around each blade root as rings of quads, on an OBJ.

A hub that is quad-dominant has no complete rings of quads around a blade root, which repitch's
ring route needs. Per blade, ``rings`` rows of quads with as many nodes as the root loop are
marched outward on the hub, the hub faces they reach plus one layer leave, and the annulus
between the outer ring and that hole is remeshed quad-dominant by the remesh ``refine`` uses,
with both loops frozen and every node on the source hub (:mod:`._hub_remesh`, which repitch's
remesh route shares). Blade nodes and root nodes never move; every other family, the hub's open
boundaries (a sector's seams) and the hub faces away from the roots stay to the bit. The kept hub
faces keep their order and the new faces take the place of the first face taken, rings from the
root outward, because the solver's results depend on face order. The quality of the new cells
(folded or turned cells, a node off the hub) is warned with its numbers, and the mesh written.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cfdbender._textio as _textio
from cfdbender._digest import file_sha256
from cfdbender._errors import (
    GeometryWarning,
    InputValueError,
    NotApplicableWarning,
    warn,
)
from cfdbender.mesh._hub_remesh import (
    PASSES_TRIED,
    SHRINK,
    TRIES,
    Built,
    Result,
    annulus,
    area,
    assemble,
    cell_findings,
    cell_warnings,
    check_kept,
    chord_deviation,
    cut,
    distance_warning,
    edge_counts,
    forbidden,
    forward,
    hub_family,
    kept_edges,
    new_by_old,
    root_edge_median,
    written_blocks,
)
from cfdbender.mesh._hub_remesh import turned_cells as _turned_cells
from cfdbender.mesh._hub_rings_grid import (
    Region,
    march,
    region,
    signed_area,
)
from cfdbender.mesh._refine._obj import (
    ObjMesh,
    face_blocks,
    mesh_unit,
    obj_text,
    read_obj,
)
from cfdbender.mesh._refine._remesh import PASSES, element_report
from cfdbender.mesh._repitch import HubSurface
from cfdbender.mesh._repitch_grid import edges, quality, rings, root_loop
from cfdbender.mesh._repitch_io import (
    Rewrite,
    attachments,
    morphing_table,
    read_back,
    read_control,
    sidecar_reads_back,
)

__all__ = ["hub_rings_mesh"]

Edge = tuple[int, int]

#: The widths tried, each this fraction of the one before, when the rings do not fit.
_SHRINK, _TRIES = SHRINK, TRIES
#: The remesh passes tried in turn on an annulus, while a pass count turns a cell over.
_PASSES = PASSES_TRIED
_VERB = "hub_rings"


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise InputValueError(f"hub_rings: {message}. No file was written.")


def _names(cfg: dict[str, Any], key: str, mesh: ObjMesh) -> list[str]:
    names = cfg.get(key, [])
    _need(
        isinstance(names, list) and bool(names) and all(isinstance(n, str) for n in names),
        f"hub_rings.{key} must name one or more families",
    )
    _need(len(set(names)) == len(names), f"hub_rings.{key} repeats a family")
    absent = sorted(set(names) - mesh.families.keys())
    _need(not absent, f"hub_rings.{key} names absent families {absent}")
    return list(names)


def read_config(data: dict[str, Any], mesh: ObjMesh) -> dict[str, Any]:
    """Return the ``[hub_rings]`` table of a morphing file, checked against a mesh."""
    cfg = morphing_table(data, "hub_rings", ("blades", "hub"))
    _need(isinstance(cfg, dict), "the morphing file needs a [hub_rings] table")
    for key in sorted(set(cfg) - {"blades", "hub", "rings", "band_width"}):
        warn(f"hub_rings.{key} does not apply and is skipped", NotApplicableWarning)
    count = cfg.get("rings")
    _need(
        isinstance(count, int) and not isinstance(count, bool) and count >= 1,
        f"hub_rings.rings must be a whole number of rings, 1 or more, not {count!r}",
    )
    for key in ("band_width",):
        value = cfg.get(key)
        _need(
            key not in cfg
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0.0
            ),
            f"hub_rings.{key} must be a positive length in the mesh's unit, not {value!r}",
        )
    width = cfg.get("band_width")
    blades, hubs = _names(cfg, "blades", mesh), _names(cfg, "hub", mesh)
    _need(not set(blades) & set(hubs), "hub_rings.blades and hub must be disjoint")
    return {"blades": blades, "hub": hubs, "rings": count, "band_width": width}


@dataclass
class _Blade:
    """One blade's rings: the root loop, the rows, the faces taken and the hole's boundary."""

    name: str
    family: str
    loop: list[int]
    rows: np.ndarray
    taken: set[int]
    hole: list[int]
    width: float
    area: Region
    tris: list[tuple[int, int, int]]
    route: str


def _default_width(mesh: ObjMesh, family: str, loop: list[int], count: int) -> float:
    """Return ``rings`` times the median length of the hub edges leaving the root loop."""
    return count * root_edge_median(mesh, family, loop, _VERB)


@dataclass
class _Seed:
    """What one attempt at a blade's rings is asked: the root, its hub family, count and width."""

    name: str
    family: str
    loop: list[int]
    count: int
    width: float


def _attempt(mesh: ObjMesh, seed: _Seed, banned: set[int]) -> _Blade | None:
    """Return the blade's rings at the seed's width, or None when they fold or do not fit."""
    faces = mesh.families[seed.family]
    zone = region(mesh.verts, faces, seed.loop, seed.width)
    loop = list(seed.loop)
    root = zone.chart.coordinates(mesh.verts[loop])
    if signed_area(root) < 0.0:
        loop = [loop[0], *loop[:0:-1]]
        root = root[[0, *range(len(loop) - 1, 0, -1)]]
    rows = march(root, seed.count, seed.width)
    if rows is None:
        return None
    hole = cut(mesh.verts, faces, zone, (rows, loop), rows[-1], banned=banned)
    if hole is None:
        return None
    return _Blade(
        seed.name,
        seed.family,
        loop,
        rows,
        hole.taken,
        hole.hole,
        seed.width,
        zone,
        hole.tris,
        hole.route,
    )


def _seed(
    mesh: ObjMesh, name: str, cfg: dict[str, Any], gone: dict[str, set[int]]
) -> tuple[_Seed, set[int]]:
    """Return what one blade's rings are asked and the hub faces they may not take."""
    hub_faces = [f for n in cfg["hub"] for f in mesh.families[n]]
    loop = root_loop(mesh.families[name], hub_faces, name)
    family = hub_family(mesh, cfg["hub"], loop, name, _VERB)
    banned = forbidden(mesh, family, name, gone.setdefault(family, set()))
    width = cfg["band_width"] or _default_width(mesh, family, loop, cfg["rings"])
    return _Seed(name, family, loop, cfg["rings"], width), banned


def _fits_every_blade(mesh: ObjMesh, cfg: dict[str, Any], width: float) -> bool:
    """Return whether the rings of every named blade fit at one stated width, planned in turn.

    The width a refusal names is the ``band_width`` a user then states for every blade, so it is
    judged on them all: judged on the refused blade alone, the faces it then takes can leave no
    room for the next blade.
    """
    gone: dict[str, set[int]] = {}
    for name in cfg["blades"]:
        seed, banned = _seed(mesh, name, {**cfg, "band_width": width}, gone)
        found = _attempt(mesh, seed, banned)
        if found is None:
            return False
        gone[seed.family] |= found.taken
    return True


def _plan(mesh: ObjMesh, name: str, cfg: dict[str, Any], gone: dict[str, set[int]]) -> _Blade:
    """Plan one blade's rings, refusing rings that do not fit and naming the width that does."""
    seed, banned = _seed(mesh, name, cfg, gone)
    family, width = seed.family, seed.width
    found = _attempt(mesh, seed, banned)
    if found is not None:
        gone[family] |= found.taken
        return found
    fits = None
    for k in range(1, _TRIES + 1):
        # The band_width named is the one tried, to the digits the message prints.
        trial = float(f"{width * _SHRINK**k:.6g}")
        if _fits_every_blade(mesh, cfg, trial):
            fits = trial
            break
    raise InputValueError(
        f"hub_rings: blade {name!r}: {cfg['rings']} rings of width {width:.6g} do not fit on "
        f"hub {family!r}: they fold, or the faces they need reach an open boundary (a seam), "
        "another family or another blade's rings; "
        + (
            f"a width of {fits:.6g} fits every blade"
            if fits
            else "no width down to a hundredth of it fits every blade"
        )
        + ". No file was written."
    )


def _fell_back(name: str, passes: int, elements: dict[str, Any], shape: dict[str, Any]) -> None:
    """Warn that a blade's annulus kept fewer remesh passes than ``refine`` runs, and what it is.

    The record alone held the pass count, and its ``elements`` names the mode asked, so an
    annulus left mostly triangles, or with a sliver, went out with exit 0 and nothing a user
    reads at the command line.
    """
    warn(
        f"hub_rings: blade {name!r}: the annulus remesh turned a cell against the hub at "
        f"{PASSES} passes and kept {passes}; its cells are {elements['quad_share']:.0%} quads, "
        f"the least scaled Jacobian {shape['min_scaled_jacobian']:.3g} and the largest aspect "
        f"{shape['max_aspect']:.3g}. Inspect the hub around that root, or state a smaller width",
        GeometryWarning,
    )


def _new_faces(mesh: ObjMesh, plans: list[_Blade], surface: HubSurface) -> Built:
    """Return the new nodes and the new faces of every blade, numbered after the source nodes."""
    built = Built([], {}, {})
    start = len(mesh.verts)
    for plan in plans:
        count, n = len(plan.rows) - 1, len(plan.loop)
        lifted = plan.area.chart.lift(plan.rows[1:], mesh.verts[plan.area.ids], plan.area.triangles)
        ahead = forward(mesh.families[plan.family], plan.loop)
        # _turned_cells is read here, at the call, so that it is this module's name.
        points, ring_band, passes = annulus(
            mesh, plan, lifted[-1], surface, 1 if ahead else -1, turned=_turned_cells, verb=_VERB
        )
        ids = np.vstack(
            [np.asarray(plan.loop)[None, :], start + np.arange(count * n).reshape(count, n)]
        )
        quads = [
            [ids[k, j], ids[k, (j + 1) % n], ids[k + 1, (j + 1) % n], ids[k + 1, j]]
            if ahead
            else [ids[k, j], ids[k + 1, j], ids[k + 1, (j + 1) % n], ids[k, (j + 1) % n]]
            for k in range(count)
            for j in range(n)
        ]
        extra = start + count * n
        local = np.r_[ids[-1], plan.hole, extra + np.arange(len(points) - n - len(plan.hole))]
        ring_faces = [[int(local[v]) for v in (f if ahead else f[::-1])] for f in ring_band]
        built.faces[plan.name] = [[int(v) for v in q] for q in quads] + ring_faces
        built.points += [lifted.reshape(-1, 3), points[n + len(plan.hole) :]]
        built.quads[plan.name] = {**element_report("quad-dominant", ring_band), "passes": passes}
        if passes != PASSES:
            _fell_back(plan.name, passes, built.quads[plan.name], quality(points, ring_band))
        start = extra + len(points) - n - len(plan.hole)
    return built


def _check_rings(r: Result) -> list[int]:
    """Refuse a blade whose output hub does not hold exactly the rings asked, from its root."""
    found = []
    for plan in r.plans:
        hub = r.mesh.families[plan.family]
        loop = [int(r.index[v]) for v in plan.loop]
        count = len(rings(loop, hub)) - 1
        asked = len(plan.rows) - 1
        _need(
            count == asked,
            f"blade {plan.name!r}: the output holds {count} complete rings, not {asked}",
        )
        found.append(count)
    return found


def verify_rings(r: Result, surface: HubSurface, hubs: list[str]) -> dict[str, Any]:
    """Run every check of the rebuilt hub before anything is written.

    Broken invariants refuse (the rings asked, a kept node or face, the place of the new faces,
    open and non-manifold edges). Quality is warned with its numbers and listed under
    ``warnings``: folded new cells and their least scaled Jacobian, cells turned against the
    source hub, and the worst new node off the hub beyond the source's chord deviation.
    """
    checks: dict[str, Any] = {"ring_counts": _check_rings(r)}
    shape, turned = cell_findings(r, surface)
    checks["new_cell_quality"] = shape
    checks["turned_cells"] = turned
    found = cell_warnings(shape, turned)
    check_kept(r, _VERB)
    checks["manifold"] = edge_counts(r, _VERB)
    hub_faces = [f for n in hubs for f in r.source.families[n]]
    allowed = chord_deviation(r.source.verts, hub_faces)
    distance, off = distance_warning(r, surface, allowed)
    checks["largest_distance_to_source_hub"] = distance
    checks["source_chord_deviation"] = allowed
    found += off
    before = area(r.source.verts, hub_faces)
    after = area(r.mesh.verts, [f for n in hubs for f in r.mesh.families[n]])
    checks["hub_area"] = {"source": before, "output": after, "relative_change": after / before - 1}
    for sentence in found:
        warn(f"hub_rings: {sentence}; the mesh is written", GeometryWarning)
    checks["warnings"] = found
    return checks


def build_rings(source: ObjMesh, cfg: dict[str, Any]) -> tuple[Result, HubSurface]:
    """Plan and build every named blade's rings, unchecked; return the result and the hub surface.

    :func:`verify_rings` holds the result to the source before anything is written.
    """
    gone: dict[str, set[int]] = {}
    plans = [_plan(source, name, cfg, gone) for name in cfg["blades"]]
    surface = HubSurface(source.verts, [f for n in cfg["hub"] for f in source.families[n]])
    return assemble(source, plans, _new_faces(source, plans, surface)), surface


def rings_step(source: ObjMesh, cfg: dict[str, Any]) -> tuple[Result, dict[str, Any], Rewrite, Any]:
    """Rebuild the hub around every named blade in memory.

    Returns the result (the mesh, and where its new faces sit), the report, the rewrite of the
    boundaries file and each source node's place in the output (NaN for a node removed).

    Every check runs here; nothing is written. The rewrite carries the boundaries file through
    the new node list: a marking on a removed node or edge is refused.
    """
    result, surface = build_rings(source, cfg)
    plans = result.plans
    checks = verify_rings(result, surface, cfg["hub"])
    report = _report(result, cfg, checks)
    removed = {i for i in range(len(source.verts)) if result.index[i] < 0}
    kept = kept_edges(result)
    rw = Rewrite(
        verb="hub_rings",
        counts=(
            sum(len(f) for f in result.mesh.families.values()),
            [len(f) for f in result.mesh.families.values()],
        ),
        gone=removed,
        dead={
            e
            for plan in plans
            for i in plan.taken
            for e in edges(source.families[plan.family][i])
            if e not in kept
        },
    )
    return result, report, rw, new_by_old(source, result.mesh, result.index)


def _report(r: Result, cfg: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    blades = {}
    for plan in r.plans:
        added = r.added[plan.name]
        blades[plan.name] = {
            "hub_family": plan.family,
            "rings": len(plan.rows) - 1,
            "width": plan.width,
            "root_loop_nodes": len(plan.loop),
            "faces_taken": len(plan.taken),
            "faces_added": len(added),
            "hole_boundary_nodes": len(plan.hole),
            "annulus_first_triangulation": plan.route,
            "annulus": r.annulus[plan.name],
        }
    return {
        "rings": cfg["rings"],
        "width_stated": cfg["band_width"],
        "blades": blades,
        "checks": {"passed": not checks["warnings"], **checks},
    }


def hub_rings_mesh(mesh: str | Path, morphing: str | Path, out: str | Path) -> dict[str, Any]:
    """Rebuild the named hub around each named blade as rings of quads; write after every check.

    Parameters
    ----------
    mesh : str or Path
        Source OBJ, read only. A saved simulation (.fsm) is refused.
    morphing : str or Path
        TOML control file with a ``[hub_rings]`` table: ``rings`` (a whole number), optionally
        ``band_width`` (total ring height in the mesh's unit), and ``blades`` and ``hub``
        of its own or at the top of the file.
    out : str or Path
        New OBJ. Existing outputs are refused.

    Returns
    -------
    dict
        The per-blade counts and the checks, also written as ``<out>.hub_rings.json``.
        ``checks["passed"]`` is false when a quality warning was raised, and
        ``checks["warnings"]`` lists them.
    """
    path, control, target = Path(mesh), Path(morphing), Path(out)
    record = Path(str(target) + ".hub_rings.json")
    _need(
        path.suffix.lower() == ".obj",
        f"mesh {path} must be an .obj: the operation works on the OBJ, not on a saved .fsm",
    )
    _need(target.suffix.lower() == ".obj", "--out must be an .obj")
    _need(target.resolve() != path.resolve(), "--out must differ from the source")
    _need(not target.exists() and not record.exists(), "output or report exists; choose new --out")
    source = read_obj(path)
    cfg = read_config(read_control(control, _VERB), source)
    result, report, rw, moved = rings_step(source, cfg)
    built = result.mesh
    # The faces in the source's order: a family that returns after another keeps its runs.
    payload = obj_text(built, blocks=written_blocks(face_blocks(path), result)).encode("utf-8")
    read_back(built, payload, "hub_rings")
    report["checks"]["readback_equal"] = True
    extra = attachments(path, target, source, moved, payload, rewrite=rw)
    sidecar_reads_back(extra, target, "hub_rings")
    for file in extra:
        _need(not file.exists(), f"attachment {file} exists; choose new --out")
    report["mesh_unit"] = mesh_unit(path)
    report["source_sha256"] = file_sha256(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    for file, content in extra.items():
        file.write_bytes(content)
    report["output_sha256"] = file_sha256(target)
    report["attachments"] = sorted(p.name for p in extra)
    _textio.write_text(record, json.dumps(report, indent=2) + "\n")
    return report
