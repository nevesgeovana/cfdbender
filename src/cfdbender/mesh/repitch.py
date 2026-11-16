"""Small rigid blade pitch changes with the hub rebuilt around each root.

Two routes rebuild the hub, chosen by ``[repitch] hub_method``. The ring route (``"rings"``)
recovers the complete quad rings around each root and moves their nodes: it replaces only
coordinates, so connectivity, quad pairing, face order, trailing-edge flags, base regions and
wake terminations keep their identities. The
remesh route (``"remesh"``) takes the hub band around each root out and remeshes it
quad-dominant between the turned root and the band's unmoved boundary, every free node on the
source hub and the kept faces in place and order (:mod:`._hub_remesh`). ``"auto"``, the default,
takes the ring route where every named root has two complete quad rings and the remesh route
otherwise. Quality (folded, reversed or turned cells, a node off the hub) is warned with its
numbers and the mesh written; broken input and broken invariants refuse.
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
from cfdbender._lengths import METRES_PER_UNIT
from cfdbender.mesh._hub_remesh import (
    area,
    cell_findings,
    cell_warnings,
    check_kept,
    chord_deviation,
    distance_warning,
    edge_counts,
    kept_edges,
    new_by_old,
    remesh_bands,
    target_miss,
    written_blocks,
)
from cfdbender.mesh._pitch_axis import frame_basis_for_shaft, pitch_axes
from cfdbender.mesh._refine._obj import (
    ObjMesh,
    face_blocks,
    mesh_unit,
    obj_text,
    read_obj,
)
from cfdbender.mesh._repitch import HubSurface, surface_checks
from cfdbender.mesh._repitch_grid import (
    edges,
    incidence,
    quality,
    rebuild,
    rings,
    root_loop,
)
from cfdbender.mesh._repitch_io import (
    Rewrite,
    attachments,
    morphing_table,
    read_back,
    read_control,
    read_rotor,
    sidecar_reads_back,
)
from cfdbender.mesh._wheel import Faces, parse_axis, rotate, wheel_symmetry

__all__ = ["repitch_mesh"]

#: The routes ``[repitch] hub_method`` names; ``auto`` is the default.
HUB_METHODS = ("auto", "rings", "remesh")


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise InputValueError(f"repitch: {message}. No file was written.")


def _read(path: Path) -> ObjMesh:
    _need(
        path.suffix.lower() == ".obj",
        f"mesh {path} must be an .obj (a saved simulation is not accepted here)",
    )
    return read_obj(path)


def _degrees(cfg: dict[str, Any], key: str, default: float) -> float:
    """Return an angle of the table in degrees, refusing a value that is not a number.

    A boolean is refused too: ``true`` would read as one degree, as ``hub_rings.width`` would.
    """
    value = cfg.get(key, default)
    _need(not isinstance(value, bool), f"{key} must be a number of degrees, not {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        _need(False, f"{key} must be a number of degrees, not {value!r}")
        raise


def _config(path: Path) -> dict[str, Any]:
    data = read_control(path, "repitch")
    cfg = morphing_table(data, "repitch", ("blades", "hub"))
    _need(isinstance(cfg, dict), f"{path} needs a [repitch] table")
    for key in set(cfg) - {
        "angle_deg",
        "max_angle_deg",
        "blades",
        "hub",
        "keep",
        "axis",
        "axes",
        "reference",
        "rotor",
        "rotor_axis",
        "reference_point",
        "hub_method",
        "band_width",
    }:
        warn(f"repitch.{key} does not apply and is skipped", NotApplicableWarning)
    method = cfg.get("hub_method", "auto")
    _need(
        method in HUB_METHODS,
        f"hub_method {method!r} must be one of {', '.join(repr(m) for m in HUB_METHODS)}",
    )
    width = cfg.get("band_width")
    _need(
        width is None
        or (
            isinstance(width, (int, float))
            and not isinstance(width, bool)
            and math.isfinite(width)
            and width > 0.0
        ),
        f"band_width must be a positive length in the mesh's unit, not {width!r}",
    )
    cfg["hub_method"] = method
    _need(isinstance(cfg.get("axes", {}), dict), "repitch.axes must be a table of blade axes")
    keep = cfg.get("keep", [])
    _need(
        isinstance(keep, list) and all(isinstance(n, str) for n in keep),
        f"repitch.keep must be a list of family names, not {keep!r}",
    )
    angle = _degrees(cfg, "angle_deg", math.nan)
    limit = _degrees(cfg, "max_angle_deg", 5.0)
    _need(
        math.isfinite(limit) and 0 < limit <= 5,
        f"max_angle_deg {limit!r} must be positive and at most 5 degrees",
    )
    _need(
        math.isfinite(angle) and abs(angle) <= limit,
        f"angle_deg {angle!r} exceeds max_angle_deg {limit:g}; use a smaller pitch change",
    )
    cfg["angle_deg"] = angle
    return cfg


def _own_axis(name: str, cfg: dict[str, Any]) -> Any:
    """Return the axis the table states for one blade (``axes`` before ``axis``), or None."""
    return cfg.get("axes", {}).get(name, cfg.get("axis"))


def _axis(name: str, cfg: dict[str, Any], rotor: Any, derived: dict[str, Any]) -> tuple[Any, Any]:
    spec = _own_axis(name, cfg)
    if spec is None and name in derived:
        return derived[name]
    spec = "auto" if spec is None else spec
    if isinstance(spec, dict):
        origin = np.asarray(spec.get("origin", []), dtype=float)
        _need(
            origin.shape == (3,) and bool(np.isfinite(origin).all()),
            f"blade {name!r}: repitch.axes.{name}.origin needs three finite coordinates",
        )
        return origin, parse_axis(spec.get("direction", []))
    _need(spec == "auto", f"blade {name!r}: axis must be auto or origin/direction")
    _need(
        rotor is not None and name in rotor.get("families_blades", []),
        f"blade {name!r} has no reference axis; add repitch.rotor_axis and "
        f"repitch.reference_point, repitch.axes.{name}.origin and direction, or "
        "repitch.reference",
    )
    main_axis = parse_axis(rotor["axis"])
    datum = rotor.get("blade1", {})
    _need(
        "zero" in datum and "azimuth_deg" in datum,
        f"blade {name!r}: reference rotor needs blade1.zero and blade1.azimuth_deg",
    )
    zero = parse_axis(datum["zero"])
    zero = np.asarray(frame_basis_for_shaft(tuple(main_axis), tuple(zero))[0])
    names = rotor["families_blades"]
    angle = float(datum["azimuth_deg"]) + names.index(name) * 360 / len(names)
    direction = rotate(zero[None, :], main_axis, np.zeros(3), angle)[0]
    sense = rotor.get("rpm_sign", 1)
    _need(sense in (-1, 1), "reference rpm_sign must be +1 or -1")
    direction *= -sense
    origin = np.array([float(rotor.get(k + "_m", 0)) for k in ("x", "y", "z")])
    return origin, direction


def _selection(cfg: dict[str, Any], mesh: ObjMesh, rotor: Any) -> tuple[list[str], list[str]]:
    blades = cfg.get("blades", [])
    if blades == "all" or blades == ["all"]:
        _need(rotor is not None, "blades=all needs repitch.reference")
        blades = rotor["families_blades"]
    hubs = cfg.get("hub", [])
    for key, names in (("blades", blades), ("hub", hubs)):
        _need(
            isinstance(names, list) and bool(names) and all(isinstance(n, str) for n in names),
            f"repitch.{key} must explicitly name nonempty families",
        )
        _need(len(set(names)) == len(names), f"repitch.{key} repeats a family")
        _need(
            not set(names) - mesh.families.keys(),
            f"repitch.{key} names absent families {sorted(set(names) - mesh.families.keys())}",
        )
    _need(not set(blades) & set(hubs), "blades and hub must be disjoint")
    keep = cfg.get("keep", [])
    _need(not set(keep) & set(blades + hubs), "keep overlaps blades or hub")
    for name in set(keep) - mesh.families.keys():
        warn(f"repitch.keep {name!r} is absent and skipped", NotApplicableWarning)
    return blades, hubs


def _blade_check(old: Any, new: Any, ids: Any, origin: Any, axis: Any, *, angle: float) -> dict:
    a, b = old[ids] - origin, new[ids] - origin
    axial = a @ axis
    u, v = a - np.outer(axial, axis), b - np.outer(b @ axis, axis)
    radius = np.linalg.norm(u, axis=1)
    usable = radius > max(float(radius.max()) * 1e-10, 1e-14)
    measured = np.degrees(
        np.arctan2(np.cross(u[usable], v[usable]) @ axis, np.sum(u[usable] * v[usable], axis=1))
    )
    error = float(np.max(np.abs(radius - np.linalg.norm(v, axis=1))))
    rigid_error = float(
        np.max(np.linalg.norm(rotate(new[ids], axis, origin, -angle) - old[ids], axis=1))
    )
    _need(
        rigid_error < 1e-11 * max(float(np.linalg.norm(a, axis=1).max()), 1),
        "blade shape is not a rigid rotation",
    )
    _need(
        bool(len(measured)) and bool(np.allclose(measured, angle, atol=1e-8, rtol=0)),
        "measured blade pitch change differs from angle_deg",
    )
    _need(error < 1e-11 * max(float(radius.max()), 1), "blade distance from pitch axis changed")
    selected = np.argsort(axial[usable])
    samples = selected[np.linspace(0, len(selected) - 1, 5).astype(int)]
    return {
        "connectivity_unchanged": True,
        "face_order_unchanged": True,
        "axis_distance_max_error": error,
        "rigid_max_error": rigid_error,
        "pitch_samples_deg": measured[samples].tolist(),
        "span_samples_mesh_unit": axial[usable][samples].tolist(),
        "pitch_min_deg": float(measured.min()),
        "pitch_max_deg": float(measured.max()),
    }


def repitch_mesh(mesh: str | Path, morphing: str | Path, out: str | Path) -> dict[str, Any]:
    """Repitch named blades, rebuild the hub around their roots and write after every check.

    Parameters
    ----------
    mesh : str or Path
        Source OBJ, read only.
    morphing : str or Path
        TOML control file; axes use the mesh coordinates and angle uses degrees. ``hub_method``
        (``"auto"``, ``"rings"`` or ``"remesh"``) chooses how the hub is rebuilt and
        ``band_width`` states the remesh band's width in the mesh's unit.
    out : str or Path
        New OBJ. Existing outputs are refused.

    Returns
    -------
    dict
        The checks also written beside the output as ``<out>.repitch.json``: the route each blade
        took, and ``checks["passed"]`` false with ``checks["warnings"]`` when quality was warned.

    Raises
    ------
    InputValueError
        When the input is broken or the reference rotor states no axis; nothing is written.
    """
    path, control, target = Path(mesh), Path(morphing), Path(out)
    record = Path(str(target) + ".repitch.json")
    _need(target.resolve() != path.resolve(), "--out must differ from the source")
    _need(target.suffix.lower() == path.suffix.lower(), "--out must keep the source suffix")
    _need(not target.exists() and not record.exists(), "output or report exists; choose new --out")
    cfg = _config(control)
    source = _read(path)
    try:
        rotor = read_rotor(cfg, control, path)
    except KeyError as error:
        raise InputValueError(
            f"repitch: the rotor of the reference states no {error}. No file was written."
        ) from error
    blades, hubs = _selection(cfg, source, rotor)
    unit = mesh_unit(path)
    if unit is None:
        warn(
            "repitch: mesh unit is unstated; surface distances assume metres", NotApplicableWarning
        )
        unit = "METER"
    _need(unit in METRES_PER_UNIT, f"mesh unit {unit!r} has no metre scale; state a known unit")
    scale = float(METRES_PER_UNIT[unit])
    new, report, made = _deform(source, cfg, rotor, (blades, hubs), scale=scale)
    report["distance_unit"] = "m"
    report["source_sha256"] = file_sha256(path)
    if made is None:
        _write_output(path, target, source, new)
    else:
        _write_remeshed(path, target, source, made)
    _textio.write_text(record, json.dumps(report, indent=2) + "\n")
    return report


def _pitch_frame(
    source: ObjMesh, cfg: dict, rotor: Any, blades: list, hubs: list
) -> tuple[dict, Any, Any]:
    """Return the derived pitch axes, the hub chart's rotor frame and how they were found.

    A blade with an axis of its own (``axes`` or ``axis``) keeps it; the others take the rule of
    ``rotor_axis`` and ``reference_point`` when the table states either, else the reference's.
    """
    derived: dict[str, Any] = {}
    found = None
    needs = [n for n in blades if _own_axis(n, cfg) is None]
    if needs and ("rotor_axis" in cfg or "reference_point" in cfg):
        derived, found = pitch_axes(source.verts, source.families, needs, hubs, cfg)
    frame = None
    if rotor is not None:
        frame = (
            np.array([rotor.get(k + "_m", 0.0) for k in ("x", "y", "z")]),
            parse_axis(rotor["axis"]),
        )
    elif derived:
        frame = (next(iter(derived.values()))[0], parse_axis(cfg["rotor_axis"]))
    return derived, frame, found


def _turn_blades(
    source: ObjMesh, new: Any, cfg: dict, axes: tuple[Any, dict], names: tuple[list, list]
) -> tuple[list, list, set[int], dict]:
    """Turn each blade rigidly about its pitch axis; return its rings, motions and checks."""
    old, (rotor, derived), (blades, hubs) = source.verts, axes, names
    hub_faces = [f for n in hubs for f in source.families[n]]
    held = {
        v for n in source.families if n not in blades + hubs for f in source.families[n] for v in f
    }
    patches: list = []
    motions: list = []
    used: set[int] = set()
    report: dict = {}
    for name in blades:
        ids = source.family_vertices(name)
        _need(not set(ids) & (held | used), f"blade {name!r} shares nodes with a kept/other blade")
        used.update(ids.tolist())
        origin, axis = _axis(name, cfg, rotor, derived)
        if cfg["angle_deg"]:
            new[ids] = rotate(old[ids], axis, origin, cfg["angle_deg"])
        loop = root_loop(source.families[name], hub_faces, name)
        patches.append(rings(loop, hub_faces))
        motions.append((origin, axis, cfg["angle_deg"]))
        report[name] = {
            "faces": len(source.families[name]),
            "axis_origin": origin.tolist(),
            "axis_direction": axis.tolist(),
            **_blade_check(old, new, ids, origin, axis, angle=cfg["angle_deg"]),
        }
    return patches, motions, used | held, report


def _closure(loop: list[int], hub_faces: list[list[int]]) -> bool:
    """Return whether a root shows a collapsed trailing edge: one closure triangle on a ring.

    After collapse-te, the hub face on every root edge is a quad, one per edge, and the only
    other hub face at the root is the triangle that closed the strip. The ring route cannot
    carry that ring, and repitch on a collapsed mesh names the working order (FR-3668).
    """
    by_edge = incidence(hub_faces)
    first = [by_edge.get(e, []) for e in edges(loop)]
    if any(len(own) != 1 or len(hub_faces[own[0]]) != 4 for own in first):
        return False
    ring = {own[0] for own in first}
    on = set(loop)
    others = [i for i, f in enumerate(hub_faces) if i not in ring and on & set(f)]
    return len(others) == 1 and len(hub_faces[others[0]]) == 3


def _route(cfg: dict, patches: list, hub_faces: list[list[int]]) -> str:
    """Return the route the hub takes: ``rings`` or ``remesh``.

    ``hub_method = "rings"`` takes the rings; ``"remesh"`` remeshes.
    ``"auto"`` takes the rings where every named root has two complete quad rings and remeshes
    otherwise, save a root with the closure triangle of a collapsed trailing edge, which keeps
    the ring route and its refusal naming the order repitch, then collapse-te.
    """
    asked = cfg["hub_method"]
    if asked == "rings":
        return "rings"
    if asked == "remesh":
        return "remesh"
    if all(len(p) >= 3 for p in patches):
        return "rings"
    if any(_closure([int(v) for v in p[0]], hub_faces) for p in patches):
        return "rings"
    return "remesh"


def _deform(
    source: ObjMesh,
    cfg: dict,
    rotor: Any,
    names: tuple[list, list],
    *,
    scale: float = 1.0,
) -> tuple[Any, dict, Any]:
    """Compute the new mesh and checks without touching output files.

    Returns the new coordinates, the report, and for the remesh route the rebuilt mesh with
    what carries the boundaries file (None on the ring route).
    """
    blades, hubs = names
    new = source.verts.copy()
    hub_faces = [f for n in hubs for f in source.families[n]]
    kept = [n for n in source.families if n not in blades + hubs]
    report: dict[str, Any] = {"angle_deg": cfg["angle_deg"], "kept": kept, "blades": {}, "hub": {}}
    derived, frame, found = _pitch_frame(source, cfg, rotor, blades, hubs)
    if found is not None:
        report["pitch_axes"] = found
    patches, motions, taken, report["blades"] = _turn_blades(
        source, new, cfg, (rotor, derived), (blades, hubs)
    )
    route = _route(cfg, patches, hub_faces)
    report["hub_method"] = cfg["hub_method"]
    for name in blades:
        report["blades"][name]["hub_route"] = route
    wheel = _periodic_set(cfg, rotor, blades, hubs, frame)
    if route == "remesh":
        made = _remesh(source, new, cfg, names, report, scale=scale)
        _periodicity_remeshed(source, made.mesh, wheel, report["checks"])
        return new, report, made
    _rings_route(source, new, (patches, motions, taken), (hubs, frame), report, scale=scale)
    _periodicity(source, new, wheel, report["checks"])
    return new, report, None


def _rings_route(
    source: ObjMesh, new: Any, turned: tuple, where: tuple, report: dict, *, scale: float
) -> None:
    """Rebuild the complete quad rings around each root in ``new`` and fill the checks."""
    patches, motions, taken = turned
    hubs, frame = where
    old = source.verts
    hub_faces = [f for n in hubs for f in source.families[n]]
    held = {v for n in report["kept"] for f in source.families[n] for v in f}
    interiors: set[int] = set()
    outer_nodes = {int(v) for patch in patches for v in patch[-1]}
    surface = HubSurface(old, hub_faces)
    for patch, motion in zip(patches, motions, strict=True):
        inner = set(patch[1:-1].ravel().tolist())
        _need(
            not inner & (taken | interiors | outer_nodes),
            "hub rings overlap a blade or another patch boundary",
        )
        interiors.update(inner)
        rebuild(old, new, patch, motion, surface, frame=frame)
    _need(bool(np.isfinite(new).all()), "hub smoothing produced nonfinite coordinates")
    # An edge over two faces where nothing moves (an airframe imprinted on itself beside an
    # installed rotor) is no broken input for the ring route; one at a moved node is.
    moving = set(np.flatnonzero((new != old).any(axis=1)).tolist())
    near = [f for fs in source.families.values() for f in fs if moving.intersection(f)]
    edge_map = incidence(near)
    _need(
        not any(len(owners) > 2 and moving & set(e) for e, owners in edge_map.items()),
        "non-manifold edge at a moved hub or blade node",
    )
    distances = surface_checks(surface, old, new, {int(v) for p in patches for v in p[0]}, scale)
    report["hub"] = _hub_checks(source, new, hubs)
    boundary = {v for e, owners in incidence(hub_faces).items() if len(owners) == 1 for v in e}
    fixed = sorted((boundary - (taken - held)) | held)
    _need(np.array_equal(old[fixed], new[fixed]), "non-root hub boundary moved")
    found = distances.pop("warnings") + [w for h in report["hub"].values() for w in h["warnings"]]
    report["checks"] = {
        "passed": not found,
        "warnings": found,
        "non_manifold_edges": 0,
        "root_gap": 0.0,
        "hub_boundary_unmoved": True,
        "triangles_added": 0,
        "ring_counts": [len(p) - 1 for p in patches],
        **distances,
    }


def _hub_checks(source: ObjMesh, new: Any, hubs: list) -> dict:
    """Measure the hub's quality against the source; warn of folded and reversed cells.

    A folded cell (a corner's scaled Jacobian at or below zero) and a reversed cell (its normal
    against the source cell's) are quality: each is warned with its count, the least scaled
    Jacobian beside the source's, and listed under the hub's ``warnings``.
    """
    old = source.verts
    report = {}
    for name in hubs:
        faces = source.families[name]
        before, after = quality(old, faces), quality(new, faces)
        reversed_cells = 0
        for face in faces:
            a, b = old[face], new[face]
            na = np.cross(a, np.roll(a, -1, axis=0)).sum(axis=0)
            nb = np.cross(b, np.roll(b, -1, axis=0)).sum(axis=0)
            reversed_cells += float(na @ nb) <= 0
        found = []
        if after["folded_cells"]:
            found.append(
                f"hub {name!r} has {after['folded_cells']} folded cell(s); the least scaled "
                f"Jacobian is {after['min_scaled_jacobian']:.6g} (source "
                f"{before['min_scaled_jacobian']:.6g})"
            )
        if reversed_cells:
            found.append(f"hub {name!r} has {reversed_cells} reversed cell(s) against the source")
        for sentence in found:
            warn(f"repitch: {sentence}; the mesh is written", GeometryWarning)
        report[name] = {
            "source_cells": len(faces),
            "output_cells": len(faces),
            "source_quality": before,
            "output_quality": after,
            "reversed_cells": reversed_cells,
            "warnings": found,
        }
    return report


def _periodic_set(cfg: dict, rotor: Any, blades: list, hubs: list, frame: Any) -> Any:
    """Return the wheel the periodicity check reads (blades, hubs, axis, origin, copies), or None.

    A reference that declares several blades turns it on, about its own axis, over every blade
    it declares; without one, several named blades turn it on, about ``rotor_axis`` through the
    hub centre. Only the blades and the hub are read, so another body in the file (a wing, a
    fuselage) does not take part.
    """
    if rotor is not None and len(rotor["families_blades"]) > 1:
        origin = np.array([rotor.get(k + "_m", 0.0) for k in ("x", "y", "z")])
        names = list(rotor["families_blades"])
        return names, hubs, parse_axis(rotor["axis"]), origin, len(names)
    if rotor is None and frame is not None and len(blades) > 1:
        return blades, hubs, frame[1], frame[0], len(blades)
    return None


def _symmetry(verts: Any, faces: list, axis: Any, origin: Any, copies: int) -> dict:
    """Return the N-fold invariance of some faces, read on their own nodes."""
    ids = np.unique([v for f in faces for v in f])
    local = np.full(len(verts), -1, dtype=np.int64)
    local[ids] = np.arange(len(ids))
    picked = Faces.of([[int(local[v]) for v in f] for f in faces])
    return wheel_symmetry(verts[ids], picked, axis, origin, copies=copies, tol=1e-7)["invariance"]


def _whole(inv: dict) -> bool:
    return bool(inv["nodes_mapped"] == inv["nodes"] and inv["faces_mapped"] == inv["faces"])


def _periodicity(source: ObjMesh, new: Any, wheel: Any, checks: dict) -> None:
    """Check the blades and hub of a wheel; refuse only what the repitch made unperiodic."""
    if wheel is None:
        warn(
            "repitch: full-wheel periodicity does not apply without a wheel reference or "
            "several named blades about rotor_axis; skipped",
            NotApplicableWarning,
        )
        return
    names, hubs, axis, origin, copies = wheel
    faces = [f for n in [*names, *hubs] if n in source.families for f in source.families[n]]
    inv = _symmetry(new, faces, axis, origin, copies)
    checks["periodicity"] = inv
    if _whole(inv):
        return
    _need(
        not _whole(_symmetry(source.verts, faces, axis, origin, copies)),
        "wheel periodicity failed; repitch every blade by the same angle",
    )
    warn(
        "repitch: the blades and hub of the source are not periodic either, so the wheel "
        "periodicity is reported and does not refuse",
        NotApplicableWarning,
    )


@dataclass
class _Remeshed:
    """The remesh route's output: the mesh, the boundaries rewrite, each source node's place.

    ``result`` is the remesh result, which places the new faces in the source's face blocks.
    """

    mesh: ObjMesh
    rewrite: Rewrite
    moved: Any
    result: Any


def _turned_mesh(source: ObjMesh, new: Any) -> ObjMesh:
    """Return the source with the blades turned; a moved node's source text is not kept."""
    text = list(source.vertex_text) if len(source.vertex_text) == len(source.verts) else []
    if text:
        for i in np.flatnonzero(np.any(source.verts != new, axis=1)).tolist():
            text[i] = None
    return ObjMesh(new, source.families, list(source.header), source.family_tag, text)


def _band_notes(numbers: dict[str, dict]) -> list[str]:
    """Warn of each band left worse than the source band it replaced, with both numbers."""
    notes = []
    for name, blade in numbers.items():
        src, out = blade["source_band"], blade["output_band"]
        if (
            out["min_scaled_jacobian"] < src["min_scaled_jacobian"]
            or out["max_aspect"] > src["max_aspect"]
        ):
            notes.append(
                f"blade {name!r}: the remeshed hub band is worse than the source band it "
                f"replaces: least scaled Jacobian {out['min_scaled_jacobian']:.4g} (source "
                f"{src['min_scaled_jacobian']:.4g}), largest aspect {out['max_aspect']:.4g} "
                f"(source {src['max_aspect']:.4g})"
            )
    return notes


def _remesh(
    source: ObjMesh, new: Any, cfg: dict, names: tuple[list, list], report: dict, *, scale: float
) -> _Remeshed:
    """Remesh the hub band around every turned root and fill the checks; nothing is written.

    Broken invariants refuse (a kept node, face or open edge changed, an edge over two faces, a
    frozen loop node moved, a nonfinite node); quality is warned with its numbers.
    """
    blades, hubs = names
    old = source.verts
    hub_faces = [f for n in hubs for f in source.families[n]]
    result, surface, report["hub_remesh"] = remesh_bands(
        source, _turned_mesh(source, new), names, cfg.get("band_width")
    )
    out = result.mesh
    _need(bool(np.isfinite(out.verts).all()), "the hub remesh produced nonfinite coordinates")
    check_kept(result, "repitch")
    manifold = edge_counts(result, "repitch")
    shape, turned = cell_findings(result, surface)
    allowed = chord_deviation(old, hub_faces)
    skip = frozenset(int(result.index[v]) for n in blades for v in source.family_vertices(n))
    distance, off = distance_warning(result, surface, allowed, skip)
    found = cell_warnings(shape, turned) + off
    for sentence in found:
        warn(f"repitch: {sentence}; the mesh is written", GeometryWarning)
    roots = {v for name in blades for v in root_loop(source.families[name], hub_faces, name)}
    # The stand-off of the root is measured against the source hub alone, as the ring route does.
    distances = surface_checks(HubSurface(old, hub_faces), old, new, roots, scale)
    found += distances.pop("warnings")
    notes = _band_notes(report["hub_remesh"])
    for sentence in notes:
        warn(f"repitch: {sentence}", GeometryWarning)
    # A band that misses its target was warned as it was kept; it is listed with the others.
    found += [s for n, b in report["hub_remesh"].items() if (s := target_miss(n, b)) is not None]
    report["hub"] = {
        name: {
            "source_cells": len(source.families[name]),
            "output_cells": len(out.families[name]),
            "source_quality": quality(old, source.families[name]),
            "output_quality": quality(out.verts, out.families[name]),
        }
        for name in hubs
    }
    hub_out = [f for n in hubs for f in out.families[n]]
    report["checks"] = {
        "passed": not (found + notes),
        "warnings": found + notes,
        "non_manifold_edges": manifold["edges_over_two_faces"],
        "open_edges": manifold["open_edges"],
        "hub_boundary_unmoved": True,
        "new_cell_quality": shape,
        "turned_cells": turned,
        "largest_distance_to_source_hub": distance,
        "source_chord_deviation": allowed,
        "hub_area": {"source": area(old, hub_faces), "output": area(out.verts, hub_out)},
        **distances,
    }
    moved = new_by_old(source, out, result.index)
    return _Remeshed(out, _rewrite(source, result), moved, result)


def _rewrite(source: ObjMesh, result: Any) -> Rewrite:
    """Return how the boundaries file follows the remesh: removed nodes and edges, new counts."""
    kept = kept_edges(result)
    return Rewrite(
        verb="repitch",
        counts=(
            sum(len(f) for f in result.mesh.families.values()),
            [len(f) for f in result.mesh.families.values()],
        ),
        gone={i for i in range(len(source.verts)) if result.index[i] < 0},
        dead={
            e
            for band in result.plans
            for i in band.taken
            for e in edges(source.families[band.family][i])
            if e not in kept
        },
    )


def _periodicity_remeshed(source: ObjMesh, mesh: ObjMesh, wheel: Any, checks: dict) -> None:
    """Check a remeshed wheel: refuse blades the repitch made unperiodic, warn of the bands.

    Each blade's band is remeshed on its own, so the hub may not map node for node from one
    blade to the next even when every blade turned by the same angle; the blades must.
    """
    if wheel is None:
        _periodicity(source, mesh.verts, None, checks)
        return
    names, hubs, axis, origin, copies = wheel
    faces = [f for n in [*names, *hubs] if n in mesh.families for f in mesh.families[n]]
    inv = _symmetry(mesh.verts, faces, axis, origin, copies)
    checks["periodicity"] = inv
    if _whole(inv):
        return
    blades_now = [f for n in names if n in mesh.families for f in mesh.families[n]]
    blades_was = [f for n in names if n in source.families for f in source.families[n]]
    _need(
        _whole(_symmetry(mesh.verts, blades_now, axis, origin, copies))
        or not _whole(_symmetry(source.verts, blades_was, axis, origin, copies)),
        "wheel periodicity failed; repitch every blade by the same angle",
    )
    warn(
        f"repitch: the remeshed hub bands are not periodic node for node about the rotor axis "
        f"({inv['nodes_mapped']} of {inv['nodes']} nodes map); the blades are",
        GeometryWarning,
    )


def _write_remeshed(path: Path, target: Path, source: ObjMesh, made: _Remeshed) -> None:
    """Write the remeshed OBJ and its boundaries file after they read back as prepared.

    The faces are written in the source's order: a family that returns after another keeps a
    block for each run, the new faces in the block of the first face they replace.
    """
    blocks = written_blocks(face_blocks(path), made.result)
    payload = obj_text(made.mesh, blocks=blocks).encode("utf-8")
    read_back(made.mesh, payload, "repitch")
    extra = attachments(path, target, source, made.moved, payload, rewrite=made.rewrite)
    sidecar_reads_back(extra, target, "repitch")
    for file in extra:
        _need(not file.exists(), f"attachment {file} exists; choose new --out")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    for file, content in extra.items():
        file.write_bytes(content)


def _write_output(path: Path, target: Path, source: ObjMesh, new: Any) -> None:
    """Prepare every payload and validate readback before writing any output."""
    text = obj_text(ObjMesh(new, source.families, source.header), blocks=face_blocks(path))
    payload = text.encode("utf-8")
    extra = attachments(path, target, source, new, payload)
    for file in extra:
        _need(not file.exists(), f"attachment {file} exists; choose new --out")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    for file, content in extra.items():
        file.write_bytes(content)
