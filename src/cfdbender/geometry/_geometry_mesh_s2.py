"""Strategy 2 of a generated configuration meshed as one full quad structured grid.

Pipeline role: the workspace row of the full-grid meshing (``MESH_ROUTE full_grid``). The
configuration's members are added to one OpenCASCADE model and fragmented exactly as for
strategy 1; then every exposed face becomes a block. A face trimmed by a junction keeps its
junction curve as one side of the block (a composite side when the curve is cut in pieces),
which builds the collar and C-H topology around a footprint from the face's own corners (delta
F.2). One integer cell count per edge is solved for the whole configuration by
:func:`cfdbender.geometry._mesh_blocks.solve_counts`: opposite block sides equal, the
lifting root ring holding ``2 chordwise.cells + base cells`` on every side, the body azimuth
ring holding its count, and the cell width carried across every junction under
``defaults.junction.max_growth``. Every block is then meshed transfinite and recombined, so no
triangle is made; a face that has no four corners is reported by name as not full grid.

The record carries the plan, the counts, the not-full-grid faces, and the strategy 2 acceptance
rows of MESH_STRATEGY section 5, each with its measured value.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import cfdbender._textio as _textio
from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_mesh_occ import (
    CAP_SURFACE,
    mesh_cap,
    mesh_cap_dominant,
    rim_edges,
    rim_loop,
)
from cfdbender.geometry.stages import (
    Mesh,
    azimuth_count,
    azimuth_split,
    block_from_walk,
    carry_pairs_growth,
    check_entry,
    check_triangles,
    component_topology,
    irregular_nodes,
    mesh_controls,
    ring_count,
    solve_counts,
)

__all__ = ["mesh_full_grid", "plan_strategy2"]

#: The ratio of successive cell widths across a junction when the controls give none
#: (MESH_STRATEGY example ``defaults.junction.max_growth``).
DEFAULT_MAX_GROWTH = 1.2

#: Two edges continue one grid line across a vertex when they leave it this close to opposite.
_CONTINUE_DEG = 60.0

#: Body controls of MESH_STRATEGY 2.2 when none is given.
_BODY_DEFAULTS = {"x_cells": 80, "azimuth_cells": 24}

_ROUTE = "strategy 2 (full grid)"


# ---- plan -------------------------------------------------------------------------------------


def _members(resolved: Mapping, mesh: Mapping) -> list[str]:
    configuration = mesh.get("configuration")
    if configuration is None:
        return list(resolved["components"])
    table = resolved.get("configurations", {})
    if configuration not in table:
        raise InputValueError(
            f"mesh.configuration {configuration!r}: unknown; available {sorted(table)}"
        )
    return list(table[configuration])


def _raw(source: Any) -> dict:
    import tomllib

    if isinstance(source, (str, Path)):
        return tomllib.loads(Path(source).read_text(encoding="utf-8"))
    return json.loads(json.dumps(source, default=str))


def _names(raw: Mapping, members: list[str], base: Path | None) -> set[str]:
    iges = raw.get("mesh", {}).get("iges")
    if not iges:
        return set(members)
    from cfdbender.geometry._geometry_mesh_s1 import iges_names

    path = Path(iges)
    return iges_names(path if path.is_absolute() or base is None else base / path)


def _lifting_row(name: str, table: Mapping, defaults: Mapping) -> dict:
    row: dict[str, Any] = {}
    for direction in ("chordwise", "spanwise"):
        merged = {**defaults.get("lifting_surface", {}).get(direction, {}), **table[direction]}
        row[direction] = {"cells": int(merged["cells"]), "law": merged["law"]}
    te = {**defaults.get("lifting_surface", {}).get("chordwise", {}), **table["chordwise"]}
    row["te_cells"] = int(te.get("te_cells", 1))
    if table.get("trailing_edge", "blunt") != "blunt":
        raise InputValueError(f"{name}.trailing_edge: {_ROUTE} keeps the blunt base (one row)")
    return row


def _body_row(name: str, table: Mapping, defaults: Mapping, spec: Mapping) -> dict:
    merged = {**_BODY_DEFAULTS, **defaults.get("body", {}), **table}
    if "x_step_m" in table:
        merged.pop("x_cells", None)
    if {"azimuth_rule", "azimuth_step_deg"} & table.keys():
        merged.pop("azimuth_cells", None)
    row: dict[str, Any] = {"length_m": float(spec["length"])}
    row["x_step_m"] = float(merged.get("x_step_m") or row["length_m"] / merged["x_cells"])
    if "azimuth_rule" in merged:
        row["azimuth_rule"] = dict(merged["azimuth_rule"])
    elif "azimuth_step_deg" in merged:
        row["azimuth_cells"] = max(1, round(180.0 / float(merged["azimuth_step_deg"])))
    else:
        row["azimuth_cells"] = int(merged["azimuth_cells"])
    return row


def _component_rows(controls: Mapping, components: Mapping, members: list[str]) -> dict:
    rows = {}
    defaults = controls["defaults"]
    for name in members:
        spec = components[name]
        table = controls["components"].get(name, {})
        row = {
            "type": spec["type"],
            "role": table.get("role", "follower" if not table else "primary"),
            "blocks": component_topology(name, spec["type"], table.get("blocks")),
        }
        if spec["type"] == "lifting_surface" and row["role"] != "follower":
            row.update(_lifting_row(name, table, defaults))
        elif spec["type"] == "body":
            row.update(_body_row(name, table, defaults, spec))
        rows[name] = row
    return rows


def _junctions(controls: Mapping, members: list[str]) -> list[dict]:
    rows = []
    for index, raw in enumerate(controls.get("junctions", []) or []):
        between = list(raw.get("between", []))
        if len(between) != 2 or any(name not in members for name in between):
            raise InputValueError(
                f"junctions[{index}].between: two members of the configuration required"
            )
        count = raw.get("ring_count", "auto")
        if count != "auto" and (type(count) is not int or count < 1):
            raise InputValueError(f"junctions[{index}].ring_count: auto or a positive integer")
        rows.append(
            {"name": raw.get("name", "-".join(between)), "between": between, "ring_count": count}
        )
    return rows


def plan_strategy2(resolved: Mapping, source: Any, *, base: Path | None = None) -> dict:
    """Validate mesh controls against the configuration and plan strategy 2, without a kernel.

    Parameters
    ----------
    resolved : mapping
        The resolved configuration.
    source : mapping or path
        The mesh controls; ``mesh.strategy`` must be 2.
    base : Path, optional
        The folder relative ``mesh.iges`` paths are read from.

    Returns
    -------
    dict
        Members, per component rows (topology, counts, steps), junctions, ``max_growth``,
        warnings and the validated controls.

    Raises
    ------
    InputValueError
        For a schema error (E01 to E16, E07 for any triangle allowance) or a control this route
        cannot honor.
    """
    from cfdbender.geometry.stages import unpack

    raw = _raw(source)
    mesh = raw.get("mesh", {}) if isinstance(raw.get("mesh"), dict) else {}
    if mesh.get("strategy", 1) != 2:
        raise InputValueError("mesh.strategy: the full_grid route meshes strategy 2 controls")
    members = _members(resolved, mesh)
    controls = mesh_controls(raw, members=set(members), names=_names(raw, members, base))
    if controls["mesh"]["variant"] != "clean":
        raise InputValueError("mesh.variant: the generated mesh route supports only clean")
    components = unpack(resolved)["components"]
    junction = controls["defaults"].get("junction", {})
    max_growth = float(junction.get("max_growth", DEFAULT_MAX_GROWTH))
    if not (math.isfinite(max_growth) and max_growth > 1):
        raise InputValueError("defaults.junction.max_growth: a number above 1 required")
    return {
        "strategy": 2,
        "variant": controls["mesh"]["variant"],
        "configuration": controls["mesh"].get("configuration"),
        "members": members,
        "components": _component_rows(controls, components, members),
        "junctions": _junctions(controls, members),
        "max_growth": max_growth,
        "warnings": list(controls["warnings"]),
        "controls": controls,
    }


# ---- kernel model -----------------------------------------------------------------------------


def _assembly(model, resolved: Mapping, members: list[str]) -> dict:
    """Add and fragment the members; return face holders, source families and the audit."""
    from cfdbender.geometry._geometry_generate import add_occ_patch, assemble
    from cfdbender.geometry.stages import assembly_regions, generate_components

    sources = generate_components(resolved)
    regions = assembly_regions(resolved, sources)
    groups: dict[str, list[int]] = {}
    family: dict[int, str] = {}
    for name in members:
        groups[name] = []
        for patch in sources[name]:
            tag = add_occ_patch(model.occ, patch)
            groups[name].append(tag)
            family[tag] = patch.family
    model.occ.synchronize()
    exposed, audit = assemble(model, groups, {k: regions[k] for k in members})
    if audit["free_edges_off_symmetry"] or audit["nonmanifold_edges"]:
        raise InputValueError("assembly: source topology gate failed (free or non-manifold edges)")
    holder = {int(s): name for name, tags in exposed.items() for s in tags}
    # The one-face tip cap is meshed apart, quad-dominant (MESH_DESIGN section 8), not as a block.
    cap = {s: holder.pop(s) for s in sorted(holder) if family.get(s) == "tip_cap"}
    return {"holder": holder, "family": family, "audit": audit, "cap": cap}


def _points(model, edge: int, count: int = 9) -> np.ndarray:
    lo, hi = model.getParametrizationBounds(1, abs(edge))
    xyz = np.asarray(model.getValue(1, abs(edge), np.linspace(lo[0], hi[0], count)))
    xyz = xyz.reshape(-1, 3)
    return xyz if edge > 0 else xyz[::-1]


def face_edges(model, face: int) -> list[int]:
    """Return the tags of a face's boundary edges, leaving out a degenerated edge.

    A patch with one side collapsed to a point (the one-face planar tip cap closing at the
    leading edge) keeps that side as an edge with no end vertices; it bounds no cells and takes
    no count, so the walk and the count classes skip it.
    """
    return [
        abs(t)
        for _, t in model.getBoundary([(2, face)], oriented=False)
        if model.getBoundary([(1, abs(t))], oriented=False)
    ]


def _ends(model, edge: int) -> tuple[int, int]:
    """Return the edge's (start, end) vertex tags in its native parameter direction."""
    tags = [t for _, t in model.getBoundary([(1, abs(edge))], oriented=False)]
    start = _points(model, abs(edge), 2)[0]
    tags.sort(key=lambda t: float(np.linalg.norm(np.asarray(model.getValue(0, t, [])) - start)))
    return (tags[0], tags[-1])


def _walk(model, face: int, ends: dict) -> list[int]:
    """Return the closed signed edge walk of a face (one outer loop)."""
    pending = sorted(face_edges(model, face))
    first = pending.pop(0)
    walk, current = [first], ends[first][1]
    while pending:
        step = next((e for e in pending if current in ends[e]), None)
        if step is None:
            raise InputValueError(f"face {face}: its boundary is not one closed loop")
        pending.remove(step)
        forward = ends[step][0] == current
        walk.append(step if forward else -step)
        current = ends[step][1] if forward else ends[step][0]
    return walk


def _turns(model, walk: list[int]) -> list[float]:
    """Return the turn in degrees at the start vertex of every edge of a walk."""
    polylines = [_points(model, e) for e in walk]
    turns = []
    for k, line in enumerate(polylines):
        before = polylines[k - 1]
        a, b = before[-1] - before[-2], line[1] - line[0]
        cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
        turns.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return turns


def _faces(model, state: dict) -> dict:
    """Walk every face once: edge ends, walks, turns and lengths."""
    edges = {e for s in state["holder"] for e in face_edges(model, s)}
    ends = {e: _ends(model, e) for e in edges}
    walks = {face: _walk(model, face, ends) for face in sorted(state["holder"])}
    return {
        "ends": ends,
        "walks": walks,
        "turns": {face: _turns(model, walk) for face, walk in walks.items()},
        "lengths": {e: float(model.occ.getMass(1, e)) for e in edges},
    }


def _blocks(state: dict, *, cusps: bool) -> None:
    """Make every face a block; list the faces that have no four corners."""
    kernel = state["kernel"]
    blocks, other = [], []
    for face, walk in kernel["walks"].items():
        name = state["holder"][face]
        block = block_from_walk(f"{name}.{face}", name, walk, kernel["turns"][face], cusps=cusps)
        if block is None:
            other.append(face)
        else:
            blocks.append((face, block))
    kernel.update(blocks=blocks, other=other, cusps=cusps)


# ---- counts -----------------------------------------------------------------------------------


def _edge_owners(model, state: dict) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for face, name in state["holder"].items():
        for e in face_edges(model, face):
            result.setdefault(e, set()).add(name)
    return result


def _direction(model, edge: int) -> np.ndarray:
    line = _points(model, edge, 2)
    vector = line[1] - line[0]
    return vector / max(float(np.linalg.norm(vector)), 1e-300)


def _junction_edges(state: dict, a: str, b: str) -> list[int]:
    return sorted(
        int(row["edge"])
        for row in state["audit"]["intersection_curves"]
        if {a, b} <= set(row["components"])
    )


def _lifting_targets(model, plan, state, holders, name) -> tuple[dict, list]:
    """Chord and span cell sizes of one lifting member and its root ring totals."""
    row = plan["components"][name]
    targets, totals = {}, []
    hosts = [m for m in plan["members"] if m != name and _junction_edges(state, name, m)]
    ring = sorted({e for m in hosts for e in _junction_edges(state, name, m)})
    families = {state["family"].get(f, "") for f, n in state["holder"].items() if n == name}
    bases = {f for f in families if f.startswith("trailing_")}
    count = ring_count(row["chordwise"]["cells"], max(row["te_cells"], len(bases)))
    row["base_cells"] = max(row["te_cells"], len(bases))
    lengths = state["kernel"]["lengths"]
    span = [e for e, n in holders.items() if name in n and abs(_direction(model, e)[1]) > 0.7]
    ys = [y for e in span for y in _points(model, e, 2)[:, 1]]
    span_m = (max(ys) - min(ys)) / row["spanwise"]["cells"] if ys else 1.0
    chord_m = sum(lengths[e] for e in ring) / count if ring else span_m
    for e, names in holders.items():
        if name in names:
            targets[e] = span_m if e in span else chord_m
    if ring:
        totals.append((ring, count))
    row["ring_count"] = count
    return targets, totals


def _body_stations(model, state, holders, name) -> dict[float, list[int]]:
    """Group the body's azimuth edges by the axial station they lie on."""
    lengths = state["kernel"]["lengths"]
    stations: dict[float, list[int]] = {}
    for e, names in holders.items():
        if names == {name} and abs(_direction(model, e)[0]) < 1e-6 and lengths[e] > 0:
            x = round(float(_points(model, e, 2)[0][0]), 9)
            stations.setdefault(x, []).append(e)
    return stations


def _body_targets(model, plan, state, holders, name) -> tuple[dict, list]:
    row = plan["components"][name]
    lengths = state["kernel"]["lengths"]
    stations = _body_stations(model, state, holders, name)
    full = {x: edges for x, edges in stations.items() if len(edges) > 1}
    if "azimuth_rule" in row and full:
        measured = [
            [x, sum(lengths[e] for e in edges), row["x_step_m"]] for x, edges in full.items()
        ]
        n_az, reference = azimuth_count(measured, row["azimuth_rule"])
        row["azimuth_reference_x_m"] = reference
    else:
        n_az = row.get("azimuth_cells", _BODY_DEFAULTS["azimuth_cells"])
    row["azimuth_count"] = int(n_az)
    targets = {e: row["x_step_m"] for e, names in holders.items() if name in names}
    totals = []
    if full:
        x0 = min(full)
        edges = full[x0]
        split = azimuth_split(int(n_az), [lengths[e] for e in edges])
        row["azimuth_split"] = split
        totals.append((edges, int(n_az)))
    return targets, totals


def _carry(model, state, holders) -> list[tuple[int, int]]:
    """Edge pairs that continue one grid line across a junction vertex, owned by two members."""
    ends, lengths = state["kernel"]["ends"], state["kernel"]["lengths"]
    junction = {int(row["edge"]) for row in state["audit"]["intersection_curves"]}
    at: dict[int, list[int]] = {}
    for e, (a, b) in ends.items():
        if e not in junction and lengths[e] > 0:
            at.setdefault(a, []).append(e)
            at.setdefault(b, []).append(-e)
    on_junction = {v for e in junction for v in ends[e]}
    pairs = []
    for vertex in sorted(on_junction & set(at)):
        leaving = [(e, _direction(model, e) * (1 if e > 0 else -1)) for e in at[vertex]]
        for i, (e1, d1) in enumerate(leaving):
            for e2, d2 in leaving[i + 1 :]:
                straight = np.dot(d1, d2) < -math.cos(math.radians(_CONTINUE_DEG))
                if straight and holders[abs(e1)] != holders[abs(e2)]:
                    pairs.append((abs(e1), abs(e2)))
    return sorted(set(pairs))


def _targets(model, plan: dict, state: dict, holders) -> tuple[dict[int, float], list]:
    """Return the target cell size of every edge and the count totals the controls pin."""
    targets: dict[int, float] = {}
    totals: list = []
    for name, row in plan["components"].items():
        if row["type"] == "lifting_surface" and "chordwise" in row:
            t, r = _lifting_targets(model, plan, state, holders, name)
        elif row["type"] == "body":
            t, r = _body_targets(model, plan, state, holders, name)
        else:
            continue
        for e, size in t.items():
            targets[e] = min(size, targets.get(e, size))
        totals += r
    for e in state["kernel"]["lengths"]:
        targets.setdefault(e, min(targets.values()) if targets else 1.0)
    for junction in plan["junctions"]:
        if junction["ring_count"] != "auto":
            totals.append((_junction_edges(state, *junction["between"]), junction["ring_count"]))
    return targets, totals


def _solve(model, plan: dict, state: dict) -> dict:
    holders = _edge_owners(model, state)
    targets, totals = _targets(model, plan, state, holders)
    carry = _carry(model, state, holders)

    def solve() -> dict[int, int]:
        kernel = state["kernel"]
        return solve_counts(
            [b for _, b in kernel["blocks"]],
            kernel["lengths"],
            targets,
            totals=totals,
            carry=carry,
            max_growth=plan["max_growth"],
            even=[[abs(e) for e in kernel["walks"][f]] for f in kernel["other"]],
        )

    _blocks(state, cusps=True)
    try:
        counts = solve()
    except InputValueError as error:
        if not str(error).startswith(("E08", "E09")):
            raise
        # A block cornered on a cusp can tie a ring to an odd count: mesh those faces with
        # quads only, unstructured, and report them as not full grid.
        _blocks(state, cusps=False)
        counts = solve()
    return {"counts": counts, "carry": carry, "totals": totals, "holders": holders}


# ---- mesh -------------------------------------------------------------------------------------


def _generate(gmsh, state: dict, counts: Mapping[int, int]) -> None:
    model = gmsh.model
    for edge, n in counts.items():
        model.mesh.setTransfiniteCurve(edge, n + 1)
    ends = state["kernel"]["ends"]
    for face, block in state["kernel"]["blocks"]:
        walk = state["kernel"]["walks"][face]
        corners = [ends[abs(walk[k])][0 if walk[k] > 0 else 1] for k in block.corners]
        model.mesh.setTransfiniteSurface(face, cornerTags=corners)
        model.mesh.setRecombine(2, face)
    for face in state["kernel"]["other"]:
        model.mesh.setRecombine(2, face)
    # Blossom: a perfect matching closes a face with an even boundary with quads only.
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.0)
    for face in state["kernel"]["other"]:
        model.mesh.setAlgorithm(2, face, 8)
    model.mesh.generate(2)


def _mesh_cap(model, state: dict, mesh: Mesh, rim: list[int]) -> Mesh:
    """Mesh the removed tip cap on the rim nodes, all quad when it can be, else quad-dominant."""
    if not rim:
        return mesh
    (name,) = set(state["cap"].values())
    loop = rim_loop(model, mesh, rim)
    try:
        mesh, normal = mesh_cap(mesh, loop, name)
    except InputValueError:
        mesh, normal = mesh_cap_dominant(mesh, loop, name)
    state["cap_plane"] = (mesh.xyz[loop].mean(axis=0), normal)
    return mesh


def _extract(model, holder: Mapping[int, str]) -> Mesh:
    tags, xyz, _ = model.mesh.getNodes()
    tags, xyz = np.asarray(tags), np.asarray(xyz).reshape(-1, 3)
    order = np.argsort(tags)
    tags, xyz = tags[order], xyz[order]
    lookup = {int(t): i for i, t in enumerate(tags)}
    cells, components, surfaces = [], [], []
    for face, name in sorted(holder.items()):
        kinds, _, blocks = model.mesh.getElements(2, face)
        for kind, block in zip(kinds, blocks, strict=True):
            width = 3 if kind == 2 else 4
            for row in np.asarray(block).reshape(-1, width):
                cells.append(tuple(lookup[int(t)] for t in row))
                components.append(name)
                surfaces.append(face)
    used = sorted({n for cell in cells for n in cell})
    compact = {n: i for i, n in enumerate(used)}
    return Mesh(
        xyz[used], [tuple(compact[n] for n in c) for c in cells], components, surfaces, tags[used]
    )


def _poles(model, plan: dict, state: dict) -> list[list[float]]:
    """Return the kernel vertices of a body cap where exactly three faces of that body meet."""
    points = []
    for name, row in plan["components"].items():
        if row["type"] != "body":
            continue
        ends = {"nose": "nose", "rear": "base"}
        faces = [
            f
            for f, n in state["holder"].items()
            if n == name
            and row["blocks"]["poles"].get(ends.get(state["family"].get(f, ""), "")) == "valence3"
        ]
        around: dict[int, int] = {}
        for f in faces:
            for v in {
                v for e in state["kernel"]["walks"][f] for v in state["kernel"]["ends"][abs(e)]
            }:
                around[v] = around.get(v, 0) + 1
        for v, k in around.items():
            point = np.asarray(model.getValue(0, v, []))
            if k == 3 and abs(point[1]) > 1e-7:
                points.append(point.tolist())
    return points


# ---- checks -----------------------------------------------------------------------------------


#: Largest distance, m, of a face's sampled points from their plane for the face to be planar.
PLANAR_TOLERANCE_M = 1e-9


def planar_faces(model, faces) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return ``{face: (point, unit normal)}`` for the planar faces among ``faces``.

    A planar face is measured against its plane, as strategy 1 measures its tip cap: the
    kernel's closest point on the one-face cap, whose leading-edge side is collapsed to a
    point, can converge to that point from a node on the trailing-edge base (measured: the
    base midpoint projected 0.125 m away, on the leading edge).
    """
    planes = {}
    for face in sorted(int(f) for f in faces):
        lo, hi = model.getParametrizationBounds(2, face)
        uv = [(u, v) for u in np.linspace(lo[0], hi[0], 5) for v in np.linspace(lo[1], hi[1], 5)]
        points = np.asarray(model.getValue(2, face, np.ravel(uv))).reshape(-1, 3)
        centre = points.mean(axis=0)
        normal = np.linalg.svd(points - centre)[2][2]
        if np.max(np.abs((points - centre) @ normal)) <= PLANAR_TOLERANCE_M:
            planes[face] = (centre, normal)
    return planes


def _quality_checks(model, mesh: Mesh, cap_plane=None) -> list[dict]:
    from cfdbender.geometry._geometry_mesh_occ import source_metrics
    from cfdbender.geometry.stages import check_mesh, orient_mesh, oriented_source_normals

    oriented, _ = orient_mesh(mesh, "mirror_volume")
    planes = planar_faces(model, {s for s in set(mesh.surfaces) if s != CAP_SURFACE})
    if cap_plane is not None:
        planes[CAP_SURFACE] = cap_plane
    normals, distances = source_metrics(model, oriented, planes)
    record = check_mesh(oriented, oriented_source_normals(oriented, normals), distances)

    def faces(cells):
        names = (f"{oriented.components[i]}.{int(oriented.surfaces[i])}" for i in cells)
        return {"by_surface": sorted(set(names))}

    return [
        _check("mesh.inverted", record["inverted_count"], where=faces(record["inverted_cells"])),
        _check(
            "mesh.degenerate", record["degenerate_count"], where=faces(record["degenerate_cells"])
        ),
        _check("mesh.node_to_node", record["non_manifold_count"]),
        _check(
            "mesh.source_distance",
            float(np.max(distances)) if len(distances) else 0.0,
            5e-4,
            unit="m",
        ),
    ]


def _check(check_id: str, value: float, limit: float = 0.0, *, where=None, unit="") -> dict:
    status = "passed" if value <= limit else "failed"
    return check_entry(check_id, status, float(value), float(limit), where=where, unit=unit)


def _ring_checks(plan: dict, state: dict, solved: dict) -> list[dict]:
    """E08: the ring count measured on each side of every lifting root junction."""
    checks = []
    counts = solved["counts"]
    for name, row in plan["components"].items():
        if "ring_count" not in row:
            continue
        for host in plan["members"]:
            ring = _junction_edges(state, name, host) if host != name else []
            if not ring:
                continue
            sides = {
                member: sum(counts[e] for e in ring if member in solved["holders"][e])
                for member in (name, host)
            }
            spread = max(sides.values()) - min(sides.values())
            entry = _check("E08.ring_counts", spread, where={"junction": f"{name}-{host}", **sides})
            entry["planned"] = row["ring_count"]
            checks.append(entry)
    return checks


def _acceptance(model, plan: dict, state: dict, solved: dict, mesh: Mesh) -> list[dict]:
    counts, lengths = solved["counts"], state["kernel"]["lengths"]
    skin = [i for i, s in enumerate(mesh.surfaces) if s != CAP_SURFACE]
    triangles = sum(len(mesh.cells[i]) == 3 for i in skin)
    in_cap = sum(
        len(c) == 3 for c, s in zip(mesh.cells, mesh.surfaces, strict=True) if s == CAP_SURFACE
    )
    skin_cells = [mesh.cells[i] for i in skin]
    try:
        check_triangles(skin_cells, [mesh.components[i] for i in skin])
    except InputValueError:
        pass
    poles, other = irregular_nodes(
        mesh.xyz, skin_cells, _poles(model, plan, state), tolerance_m=1e-6
    )
    opposite = 0
    for _, block in state["kernel"]["blocks"]:
        sums = [sum(counts[e] for e in side) for side in block.sides]
        opposite += int(sums[0] != sums[2] or sums[1] != sums[3])
    growth = carry_pairs_growth(counts, lengths, solved["carry"])
    checks = [
        _check("E07.triangles", triangles, where={"tip_cap": in_cap}),
        _check("E09.opposite_sides", opposite),
        _check(
            "E10.carry_growth", growth, plan["max_growth"], where={"pairs": len(solved["carry"])}
        ),
        check_entry(
            "E11.valence3_poles",
            "passed",
            float(poles),
            float(poles),
            bound="min",
            note="valence 3 nodes at the declared nose and base closures",
        ),
        _check("mesh.irregular_nodes.junction", len(other)),
        _check("mesh.not_full_grid_faces", len(state["kernel"]["other"])),
    ]
    quality = _quality_checks(model, mesh, state.get("cap_plane"))
    return checks + _ring_checks(plan, state, solved) + quality


# ---- entry ------------------------------------------------------------------------------------


def _kernel(plan: dict, resolved: Mapping) -> tuple[Mesh, dict]:
    from cfdbender.mesh._gmsh import session

    with session("meshing a configuration with strategy 2") as gmsh:
        model = gmsh.model
        model.add("full_grid")
        state = _assembly(model, resolved, plan["members"])
        state["kernel"] = _faces(model, state)
        solved = _solve(model, plan, state)
        rim = rim_edges(model, list(state["cap"]))
        model.removeEntities([(2, s) for s in state["cap"]])
        _generate(gmsh, state, solved["counts"])
        mesh = _mesh_cap(model, state, _extract(model, state["holder"]), rim)
        checks = _acceptance(model, plan, state, solved, mesh)
        dims = sorted({dim for dim, _ in model.getEntities() if dim >= 2})
    record = {
        "checks": checks + [_check("mesh.surfaces_only", sum(d != 2 for d in dims))],
        "blocks": len(state["kernel"]["blocks"]),
        "not_full_grid": [f"{state['holder'][f]}.{f}" for f in state["kernel"]["other"]],
        "edges": len(solved["counts"]),
        "ring_totals": [int(n) for _, n in solved["totals"]],
    }
    return mesh, record


def _counts(mesh: Mesh) -> dict:
    quads = sum(len(c) == 4 for c in mesh.cells)
    return {
        "vertices": len(mesh.xyz),
        "quads": quads,
        "tris": sum(len(c) == 3 for c in mesh.cells),
        "pct_quad": 100.0 * quads / max(1, len(mesh.cells)),
        "by_component": {n: mesh.components.count(n) for n in dict.fromkeys(mesh.components)},
    }


def _write(mesh: Mesh, record: dict, *, out, descriptor: str, fsm_template) -> dict[str, str]:
    from cfdbender._digest import file_sha256
    from cfdbender.geometry.stages import write_obj

    if fsm_template is not None:
        raise InputValueError(
            "fsm_template: a saved simulation of another solver is not written by this package; "
            "the mesh is written as OBJ with its record"
        )
    folder = Path(out)
    targets = {"obj": folder / f"{descriptor}.obj", "record": folder / f"{descriptor}.mesh.json"}
    taken = [str(p) for p in targets.values() if p.exists()]
    if taken:
        raise InputValueError(f"out: {taken} exist; nothing was written")
    folder.mkdir(parents=True, exist_ok=True)
    write_obj(targets["obj"], mesh)
    written = {k: str(v) for k, v in targets.items() if k != "record"}
    record["outputs"] = {
        k: {"path": Path(v).name, "sha256": file_sha256(v)} for k, v in written.items()
    }
    _textio.write_text(targets["record"], json.dumps(record, indent=2, sort_keys=True, default=str))
    return {**written, "record": str(targets["record"])}


def mesh_full_grid(
    config: Any,
    controls: Any,
    *,
    dry_run: bool = False,
    out: str | Path | None = None,
    descriptor: str | None = None,
    fsm_template: str | Path | None = None,
):
    """Mesh a generated configuration with strategy 2 (full grid) from its mesh controls.

    Parameters
    ----------
    config : mapping or path
        A configuration (resolved or not) or the path of its TOML file.
    controls : mapping or path
        The mesh controls, ``mesh.strategy = 2``: ``[defaults]`` (``junction.max_growth``),
        ``[components.<name>]`` (``blocks``, lifting ``chordwise`` and ``spanwise``, body
        ``x_cells`` or ``x_step_m`` and ``azimuth_cells`` or ``azimuth_rule``) and
        ``[[junctions]]`` (``between``, ``ring_count``).
    dry_run : bool
        Validate and plan only; no kernel is opened and nothing is written.
    out : str or Path, optional
        Folder for ``<descriptor>.obj``, ``<descriptor>.mesh.json`` and, with
        ``fsm_template``, ``<descriptor>.fsm``; an existing file is refused.
    descriptor : str, optional
        File stem; ``mesh.output.descriptor`` when absent, else ``mesh``.
    fsm_template : str or Path, optional
        A fresh saved simulation of the target build.

    Returns
    -------
    ConfigurationMesh
        The mesh (None on a dry run), the record (plan, counts, every strategy 2 acceptance row
        with its measured value, ``accepted``) and the written files.

    Raises
    ------
    InputValueError
        For a schema error (E01 to E16), counts that cannot hold the blocks (E09), the rings
        (E08) or the carry growth (E10).
    """
    from cfdbender.geometry._geometry_mesh import ConfigurationMesh
    from cfdbender.geometry.config import load_config, resolve_config

    base = Path(controls).parent if isinstance(controls, (str, Path)) else None
    resolved = load_config(config) if isinstance(config, (str, Path)) else resolve_config(config)
    plan = plan_strategy2(resolved, controls, base=base)
    raw = (
        Path(controls).read_bytes()
        if base is not None
        else json.dumps(controls, sort_keys=True, default=str).encode()
    )
    record: dict[str, Any] = {
        **{k: plan[k] for k in ("strategy", "variant", "configuration", "members", "components")},
        "junctions": plan["junctions"],
        "max_growth": plan["max_growth"],
        "controls_sha256": hashlib.sha256(raw).hexdigest(),
        "warnings": [*resolved["warnings"], *plan["warnings"]],
    }
    if dry_run:
        return ConfigurationMesh(None, {**record, "dry_run": True}, {})
    mesh, kernel = _kernel(plan, resolved)
    record.update(kernel, dry_run=False, counts=_counts(mesh))
    record["accepted"] = all(c["status"] != "failed" for c in record["checks"])
    written: dict[str, str] = {}
    if out is not None:
        stem = descriptor or plan["controls"]["mesh"].get("output", {}).get("descriptor", "mesh")
        written = _write(mesh, record, out=out, descriptor=stem, fsm_template=fsm_template)
    return ConfigurationMesh(mesh, record, written)
