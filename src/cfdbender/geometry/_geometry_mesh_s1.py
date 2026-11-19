"""Strategy 1 boundary laws and measured surface-mesh policy.

Pipeline role: workspace mesh planning and acceptance, downstream of the pure
schema. Shared edges use one ordered node list. Measurements are made on the
realized mesh, independently of the requested controls.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from cfdbender._errors import InputArtifactError, InputValueError
from cfdbender.geometry._geometry_mesh_body import azimuth_from_rule, body_aspect_band
from cfdbender.geometry._mesh_schema import mesh_measurements
from cfdbender.geometry.stages import (
    Mesh,
    check_entry,
    load_mesh_controls,
    mesh_controls,
)


def follower_edges(controls: Mapping[str, Any], edge_nodes: Mapping) -> dict:
    """Copy neighbor boundary divisions in dependency and declared priority order.

    Node identities and their order are preserved. A follower may not prescribe
    its own edge distribution; flexible interiors are handled after boundaries.
    """
    result = deepcopy(dict(edge_nodes))
    for name in controls["order"]:
        component = controls["components"][name]
        if component["role"] != "follower":
            result.setdefault(name, {})
            continue
        if result.get(name):
            raise InputValueError(f"{name}: follower cannot own a distribution")
        result[name] = {}
        for neighbour in component["follows"]:
            for edge, nodes in result.get(neighbour, {}).items():
                result[name].setdefault(edge, deepcopy(nodes))
    return result


def _check(name: str, value: float, limit: float = 0.0, *, where=None, unit="") -> dict:
    return check_entry(
        name, "passed" if value <= limit else "failed", value, limit, where=where, unit=unit
    )


def _canonical(cell) -> tuple:
    values = tuple(cell)
    return min(values[i:] + values[:i] for i in range(len(values)))


def structured_check(mesh, node_ids, component: str, kind: str) -> dict:
    """Compare oriented mesh cells with every cell of the declared rectangular grid."""
    ids = np.asarray(node_ids, dtype=int)
    if ids.ndim != 2 or min(ids.shape) < 2 or len(set(ids.flat)) != ids.size:
        return _check("mesh.structured", 1, where={"component": component, "kind": kind})
    expected = {
        _canonical((ids[j, i], ids[j, i + 1], ids[j + 1, i + 1], ids[j + 1, i]))
        for j in range(ids.shape[0] - 1)
        for i in range(ids.shape[1] - 1)
    }
    cells = [c for c, name in zip(mesh.cells, mesh.components, strict=True) if name == component]
    realized = {_canonical(c) for c in cells}
    defects = len(expected ^ realized) + len(cells) - len(realized)
    return _check("mesh.structured", defects, where={"component": component, "kind": kind})


def row_check(xyz, node_ids, kind: str) -> dict:
    """Measure constant-y deviation on all rows, including the root row."""
    ids = np.asarray(node_ids, dtype=int)
    points = np.asarray(xyz)[ids[:1] if kind == "fillet" else ids]
    deviation = float(np.max(np.ptp(points[:, :, 1], axis=1)))
    return _check("mesh.flow_aligned_rows", deviation, 1e-12, where={"kind": kind}, unit="m")


def trailing_edge_check(node_ids) -> dict:
    """Measure the number of base cells across the blunt trailing edge."""
    across = np.asarray(node_ids).shape[1] - 1
    return _check("mesh.trailing_edge_one_quad", abs(across - 1), where={"cells_across": across})


def axial_check(xyz, node_ids, spacing_m: float) -> dict:
    """Measure uniform axial stations without a junction blend or a local refinement."""
    stations = np.asarray(xyz)[np.asarray(node_ids, dtype=int), 0]
    deviation = max(
        float(np.max(np.ptp(stations, axis=1))),
        float(np.max(np.abs(np.diff(stations, axis=0) - spacing_m))),
    )
    return _check("mesh.fuselage_uniform_axial", deviation, 1e-12, unit="m")


def transition_checks(
    mesh, component: str, region: str, limits: Mapping, *, intersects=()
) -> list[dict]:
    """Measure region eligibility, triangle fraction, quad aspect and corner skew.

    Lifting surfaces and fillets stay structured even when marked flexible.
    Junction ratio targets are deliberately not applied here.
    """
    cells = [
        cell for cell, name in zip(mesh.cells, mesh.components, strict=True) if name == component
    ]
    triangles = sum(len(cell) == 3 for cell in cells)
    allowed = region == "fairing" or (
        region == "aft_patch" and bool(set(intersects) & {"pylon", "vertical_tail"})
    )
    checks = [_check("mesh.transition.region", int(bool(triangles) and not allowed))]
    checks.append(
        _check(
            "mesh.transition.triangle_fraction",
            triangles / max(1, len(cells)),
            float(limits.get("max_triangle_fraction", 0)),
        )
    )
    aspect, skew = 0.0, 0.0
    for cell in cells:
        if len(cell) != 4:
            continue
        points = mesh.xyz[list(cell)]
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        if min(lengths) <= 0:
            return [*checks, _check("mesh.transition.degenerate", 1)]
        aspect = max(aspect, float(max(lengths) / min(lengths)))
        cosines = np.sum(-np.roll(edges, 1, axis=0) * edges, axis=1) / (
            np.roll(lengths, 1) * lengths
        )
        skew = max(skew, float(np.max(np.abs(np.degrees(np.arccos(np.clip(cosines, -1, 1))) - 90))))
    checks.append(_check("mesh.transition.aspect", aspect, float(limits.get("max_aspect", 8))))
    checks.append(
        _check("mesh.transition.skew", skew, float(limits.get("max_skew_deg", 50)), unit="deg")
    )
    return checks


def fillet_checks(xyz, node_ids, wing_root, wing_next, *, tolerance_m: float) -> list[dict]:
    """Measure shared root identities and uniform cross spacing from the wing law."""
    ids = np.asarray(node_ids, dtype=int)
    root, next_row = np.asarray(wing_root, dtype=int), np.asarray(wing_next, dtype=int)
    points = np.asarray(xyz)
    shared = int(not np.array_equal(ids[0], root))
    first_span = float(np.median(np.linalg.norm(points[next_row] - points[root], axis=1)))
    across = np.linalg.norm(np.diff(points[ids], axis=0), axis=2)
    return [
        _check("mesh.fillet.shared_root", shared),
        _check(
            "mesh.fillet.root_refinement",
            float(np.max(np.abs(across - first_span))),
            tolerance_m,
            unit="m",
        ),
    ]


def keel_check(keel_x, fuselage_x) -> dict:
    """Measure the keel edge's distance to the fuselage axial stations."""
    keel, body = np.asarray(keel_x), np.asarray(fuselage_x)
    if not len(keel) or not len(body):
        return _check("mesh.fairing.keel_stations", 1)
    distance = np.min(np.abs(keel[:, None] - body[None, :]), axis=1)
    return _check("mesh.fairing.keel_stations", float(max(distance)), 1e-12, unit="m")


def fairing_boundary_check(mesh, component: str, boundary_edges, small_m: float) -> dict:
    """Count small cells outside the single row touching the prescribed boundary."""
    boundary = {tuple(sorted(edge)) for edge in boundary_edges}
    defects = 0
    for cell, name in zip(mesh.cells, mesh.components, strict=True):
        if name != component:
            continue
        p = mesh.xyz[list(cell)]
        area = sum(
            float(np.linalg.norm(np.cross(p[j] - p[0], p[j + 1] - p[0]))) / 2
            for j in range(1, len(cell) - 1)
        )
        touches = any(
            tuple(sorted(edge)) in boundary for edge in zip(cell, cell[1:] + cell[:1], strict=True)
        )
        defects += int(np.sqrt(area) < small_m and not touches)
    return _check("mesh.fairing.small_boundary_row", defects)


def aft_growth_checks(paths, *, first_m: float, far_m: float) -> list[dict]:
    """Measure new aft distributions from the fillet cross spacing to body spacing."""
    widths = [np.diff(np.asarray(path, dtype=float)) for path in paths]
    if not widths or any(len(w) < 2 or np.any(w <= 0) for w in widths):
        return [_check("mesh.fairing.aft_monotone", 1)]
    growth = max(float(max(w[1:] / w[:-1])) for w in widths)
    ends = max(max(abs(float(w[0]) - first_m), abs(float(w[-1]) - far_m)) for w in widths)
    # A strict inequality remains strict in the machine-readable record.
    return [
        _check("mesh.fairing.aft_end_spacing", ends, 1e-12, unit="m"),
        _check("mesh.fairing.aft_growth", growth, float(np.nextafter(1.2, 0.0))),
    ]


def cut_check(mesh, node_ids, intercept: float, slope: float, *, required: set[str]) -> dict:
    """Check a TE-plane cut is a mesh-edge chain across all required components."""
    ids = [int(i) for i in node_ids]
    edges: dict[tuple, set[str]] = {}
    for cell, name in zip(mesh.cells, mesh.components, strict=True):
        for edge in zip(cell, cell[1:] + cell[:1], strict=True):
            edges.setdefault(tuple(sorted(edge)), set()).add(name)
    components: set[str] = set()
    missing = int(len(ids) < 2)
    for edge in zip(ids[:-1], ids[1:], strict=True):
        key = tuple(sorted(edge))
        missing += int(key not in edges)
        components.update(edges.get(key, set()))
    if ids:
        points = mesh.xyz[ids]
        missing += int(np.max(np.abs(points[:, 0] - intercept - slope * points[:, 1])) > 1e-10)
    missing += len(required - components)
    return _check("mesh.trailing_edge.cut_continuation", missing)


def omission_checks(
    mesh, excluded_surfaces: set[int], source_before: str, source_after: str
) -> list[dict]:
    """Measure omitted CAD-tip faces and unchanged source identity independently."""
    return [
        _check("mesh.fillet.tip_omitted", len(set(mesh.surfaces) & excluded_surfaces)),
        _check("mesh.source.unchanged", int(source_before != source_after)),
    ]


def surface_only_check(dimensions) -> dict:
    """Refuse volume, prism and solid entities in the realized entity inventory."""
    return _check("mesh.surfaces_only", sum(int(value != 2) for value in dimensions))


def body_aspect_checks(xyz, node_ids, band) -> list[dict]:
    """Check realized circumferential over axial edge lengths at every station."""
    points = np.asarray(xyz)[np.asarray(node_ids, dtype=int)]
    cross = np.linalg.norm(np.diff(points, axis=1), axis=2)
    axial = np.linalg.norm(np.diff(points, axis=0), axis=2)
    checks = []
    for j in range(len(axial)):
        for i in range(cross.shape[1]):
            if axial[j, i] <= 0:
                raise InputValueError("E14: zero axial spacing in realized body grid")
            for k in (j, j + 1):
                checks.append(
                    mesh_measurements(
                        "E14", f"station {k}, azimuth {i}", float(cross[k, i] / axial[j, i]), band
                    )
                )
    return checks


def _collapsed_nodes(mesh: Mesh, groups, tolerance_m: float):
    xyz = mesh.xyz.copy()
    if not np.all(np.isfinite(xyz)):
        raise InputValueError("collapse distance: mesh coordinates must be finite")
    representatives = np.arange(len(xyz))
    used: set[int] = set()
    distance = 0.0
    for group in groups:
        nodes = [int(v) for v in group]
        if (
            len(nodes) < 2
            or len(set(nodes)) != len(nodes)
            or min(nodes) < 0
            or max(nodes) >= len(xyz)
            or set(nodes) & used
        ):
            raise InputValueError("collapse: disjoint valid upper/lower TE pairs required")
        used.update(nodes)
        middle = (xyz[nodes[0]] + xyz[nodes[-1]]) / 2
        distance = max(distance, float(np.max(np.linalg.norm(xyz[nodes] - middle, axis=1))))
        if not np.isfinite(distance) or distance > tolerance_m:
            raise InputValueError(f"collapse distance {distance} exceeds tolerance {tolerance_m} m")
        xyz[nodes[0]] = middle
        representatives[nodes[1:]] = nodes[0]
    return xyz, representatives, used, distance


def _reduced(cell: tuple[int, ...]) -> tuple[int, ...]:
    """Drop a node repeated next to itself (a quad with one collapsed edge becomes a triangle)."""
    result = [v for i, v in enumerate(cell) if v != cell[i - 1]]
    return tuple(result)


def collapse_trailing_edge(
    mesh: Mesh,
    pairs,
    base_cells: set[int],
    *,
    tolerance_m: float,
    reducible: frozenset = frozenset(),
) -> tuple[Mesh, list[dict]]:
    """Collapse explicit upper/lower TE node groups at their midpoints in a new mesh.

    A group is an upper node, any nodes across the base, and a lower node; it collapses to
    the midpoint of its two ends. Only declared base cells are removed. A ``reducible`` cell
    (a cap or transition cell) that loses one edge becomes a triangle; any other cell that
    would degenerate refuses the operation. The source geometry and mesh are untouched.
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or not pairs or not base_cells:
        raise InputValueError(
            "collapse: positive distance tolerance, TE pairs and base cells required"
        )
    xyz, representatives, used, distance = _collapsed_nodes(mesh, pairs, tolerance_m)
    if not base_cells <= set(range(len(mesh.cells))):
        raise InputValueError("collapse: base cell indices outside mesh")
    cells, components, surfaces = [], [], []
    for i, cell in enumerate(mesh.cells):
        new_cell = tuple(int(representatives[j]) for j in cell)
        degenerate = len(set(new_cell)) != len(new_cell)
        if i in base_cells:
            if not degenerate or not set(cell) <= used:
                raise InputValueError("collapse: declared base cell does not collapse to its TE")
            continue
        reduced = _reduced(new_cell)
        if degenerate and i in reducible and len(set(reduced)) == len(reduced) >= 3:
            new_cell = reduced
        elif degenerate:
            raise InputValueError("collapse: undeclared base or skin cell would degenerate")
        cells.append(new_cell)
        components.append(mesh.components[i])
        surfaces.append(mesh.surfaces[i])
    keep = sorted({j for cell in cells for j in cell})
    indices = {node: i for i, node in enumerate(keep)}
    collapsed = Mesh(
        xyz[keep],
        [tuple(indices[j] for j in cell) for cell in cells],
        components,
        surfaces,
        mesh.node_tags[keep].copy(),
    )
    return collapsed, [
        _check("mesh.te.collapse_distance", distance, tolerance_m, unit="m"),
        _check("mesh.te.base_removed", abs(len(mesh.cells) - len(cells) - len(base_cells))),
    ]


def merge_trailing_edge_base(
    mesh: Mesh, groups, base_cells: set[int], *, reducible: frozenset = frozenset()
) -> Mesh:
    """Leave one quad across a blunt base (MESH_DESIGN section 8) in a new mesh.

    A group is an upper node, the nodes across the base and a lower node, as
    :func:`trailing_edge_groups` finds them. In each span strip the base cell holding two
    upper nodes takes the two lower nodes in place of its nodes across the base, and the other
    base cells of the strip go. A node across the base left in a ``reducible`` cell (a cap or
    transition cell) merges onto its upper node, a quad losing a corner becoming a triangle; in
    any other cell the merge refuses. A base already one quad across is returned unchanged.
    """
    head = {int(g[0]): g for g in groups}
    across = {int(v): g for g in groups for v in g[1:-1]}
    if not across:
        return mesh
    lower = {int(g[-1]) for g in groups}
    cells, components, surfaces = [], [], []
    for i, cell in enumerate(mesh.cells):
        if i in base_cells:
            if sum(int(v) in head for v in cell) != 2:
                if not {int(v) for v in cell} <= set(across) | lower:
                    raise InputValueError("merge: a base cell is outside the TE groups")
                continue
            new_cell = tuple(int(across[v][-1]) if v in across else int(v) for v in cell)
        else:
            new_cell = _reduced(tuple(int(across[v][0]) if v in across else int(v) for v in cell))
            touched = any(int(v) in across for v in cell)
            if touched and (i not in reducible or len(set(new_cell)) < 3):
                raise InputValueError("merge: a skin cell holds a node across the base")
        cells.append(new_cell)
        components.append(mesh.components[i])
        surfaces.append(mesh.surfaces[i])
    keep = sorted({j for cell in cells for j in cell})
    indices = {node: k for k, node in enumerate(keep)}
    return Mesh(
        mesh.xyz[keep],
        [tuple(indices[j] for j in cell) for cell in cells],
        components,
        surfaces,
        mesh.node_tags[keep].copy(),
    )


#: Tolerance of a free edge on the symmetry plane y = 0 (MESH_STRATEGY section 5), m.
SYMMETRY_TOLERANCE_M = 1e-7
#: Smallest corner sine of a cell that is not degenerate (MESH_STRATEGY section 5).
DEGENERATE_SINE = 0.02


def edge_uses(mesh: Mesh) -> dict[tuple[int, int], list[int]]:
    """Map every undirected mesh edge to the cells that use it."""
    uses: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(mesh.cells):
        for a, b in zip(cell, cell[1:] + cell[:1], strict=True):
            uses.setdefault((min(a, b), max(a, b)), []).append(index)
    return uses


def shared_curves(mesh: Mesh) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Discover the curves two components share: the edges used by a cell of each."""
    curves: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for edge, cells in edge_uses(mesh).items():
        names = sorted({mesh.components[c] for c in cells})
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                curves.setdefault((a, b), []).append(edge)
    return curves


def _on_symmetry(mesh: Mesh, edge) -> bool:
    return bool(np.all(np.abs(mesh.xyz[list(edge), 1]) <= SYMMETRY_TOLERANCE_M))


def follower_checks(mesh: Mesh, controls: Mapping[str, Any]) -> list[dict]:
    """Check each follower shares a curve, node to node, with every neighbor it follows.

    The shared curves are discovered on the realized mesh (edges used by a cell of each
    component); a neighbor with none is named. Hanging edges are the free-edge check's.
    """
    curves = shared_curves(mesh)
    checks = []
    for name, component in controls["components"].items():
        if component["role"] != "follower" or name not in mesh.components:
            continue
        missing = [n for n in component["follows"] if not curves.get(tuple(sorted((name, n))))]
        found = {n: len(curves.get(tuple(sorted((name, n))), [])) for n in component["follows"]}
        checks.append(
            _check(
                "mesh.follower.shared_curves",
                len(missing),
                where={"component": name, "missing_neighbours": missing, "shared_edges": found},
            )
        )
    return checks


def _corner_measures(points: np.ndarray) -> tuple[float, float, float]:
    """Return the smallest corner sine, the edge aspect and the largest skew of one cell."""
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) <= 0:
        return 0.0, float("inf"), 90.0
    previous = -np.roll(edges, 1, axis=0)
    norms = np.roll(lengths, 1) * lengths
    sines = np.linalg.norm(np.cross(previous, edges), axis=1) / norms
    angles = np.degrees(np.arccos(np.clip(np.sum(previous * edges, axis=1) / norms, -1, 1)))
    skew = float(np.max(np.abs(angles - 90))) if len(points) == 4 else 0.0
    return float(np.min(sines)), float(max(lengths) / min(lengths)), skew


def sliver_surfaces(mesh: Mesh, faces) -> set[int]:
    """Return the faces among ``faces`` holding a cell the ``mesh.degenerate`` row refuses.

    A cell is refused when its smallest corner sine is under :data:`DEGENERATE_SINE`; the
    strategy 1 mesher retries such a transition face with its next algorithm.
    """
    faces = set(faces)
    return {
        int(surface)
        for cell, surface in zip(mesh.cells, mesh.surfaces, strict=True)
        if surface in faces and _corner_measures(mesh.xyz[list(cell)])[0] < DEGENERATE_SINE
    }


def _limit_check(check_id: str, value: float, table: Mapping, key: str, unit: str = "") -> dict:
    if key in table:
        return _check(check_id, value, float(table[key]), unit=unit)
    return check_entry(check_id, "skipped", note=f"no {key} given; measured {value:.6g}", unit=unit)


def _quality(
    mesh: Mesh, transition: set[str], quality: Mapping[str, Any], base: frozenset = frozenset()
) -> list[dict]:
    """Measure degeneracy, quad aspect and skew; ``base`` holds the TE base faces.

    The aspect limit applies everywhere except the cell across a blunt trailing-edge base (one
    quad across a base far thinner than the spanwise cell): those cells are left out of the
    limit and reported under ``where.te_base_exempt`` with their count and largest aspect.
    """
    degenerate: dict[str, int] = {}
    aspect, skew, worst = 0.0, 0.0, ""
    exempt: dict[str, Any] = {"cells": 0, "max_aspect": 0.0, "surfaces": []}
    for cell, name, surface in zip(mesh.cells, mesh.components, mesh.surfaces, strict=True):
        sine, ratio, angle = _corner_measures(mesh.xyz[list(cell)])
        if sine < DEGENERATE_SINE:
            degenerate[f"{name}.{int(surface)}"] = degenerate.get(f"{name}.{int(surface)}", 0) + 1
        if name not in transition and len(cell) == 4:
            skew = max(skew, angle)
            if int(surface) in base:
                label = f"{name}.{int(surface)}"
                exempt["cells"] += 1
                exempt["max_aspect"] = max(exempt["max_aspect"], ratio)
                if label not in exempt["surfaces"]:
                    exempt["surfaces"].append(label)
                continue
            worst = f"{name}.{int(surface)}" if ratio > aspect else worst
            aspect = max(aspect, ratio)
    aspect_check = _limit_check("mesh.quality.aspect", aspect, quality, "max_aspect")
    aspect_check["where"] = {"surface": worst, "te_base_exempt": exempt}
    return [
        _check("mesh.degenerate", sum(degenerate.values()), where={"by_surface": degenerate}),
        aspect_check,
        _limit_check("mesh.quality.skew", skew, quality, "max_skew_deg", "deg"),
    ]


def _triangle_checks(
    mesh: Mesh, limits: Mapping[str, Mapping], caps: frozenset = frozenset()
) -> list[dict]:
    """Triangles only in transition components and tip caps, each under its stated fraction.

    A tip cap may be quad-dominant (MESH_DESIGN section 8): the lifting-triangle check
    excludes cap faces only; their triangles are reported apart.
    """
    totals: dict[str, list[int]] = {}
    in_caps = 0
    for cell, name, surface in zip(mesh.cells, mesh.components, mesh.surfaces, strict=True):
        if int(surface) in caps:
            in_caps += int(len(cell) == 3)
            continue
        counts = totals.setdefault(name, [0, 0])
        counts[0] += int(len(cell) == 3)
        counts[1] += 1
    outside = {n: c[0] for n, c in totals.items() if n not in limits and c[0]}
    checks = [
        _check(
            "mesh.triangles.region",
            sum(outside.values()),
            where={"by_component": outside, "tip_caps": in_caps},
        )
    ]
    for name, table in sorted(limits.items()):
        if name in totals:
            fraction = totals[name][0] / totals[name][1]
            entry = _limit_check(
                "mesh.triangles.fraction", fraction, table, "max_triangle_fraction"
            )
            entry["where"] = {"component": name}
            checks.append(entry)
    return checks


def _irregular_nodes(mesh: Mesh, transition: set[str], caps: frozenset = frozenset()) -> dict:
    """Count interior nodes of one structured surface whose valence is not four.

    Nodes of transition components and of tip caps (quad-dominant, MESH_DESIGN section 8)
    are free.
    """
    boundary: set[int] = set()
    for edge, cells in edge_uses(mesh).items():
        if len(cells) == 1 or len({mesh.surfaces[c] for c in cells}) != 1:
            boundary.update(edge)
    incident: dict[int, list[int]] = {}
    for index, cell in enumerate(mesh.cells):
        for v in cell:
            incident.setdefault(v, []).append(index)
    found: dict[str, int] = {}
    for node, cells in incident.items():
        if (
            node not in boundary
            and not any(mesh.components[c] in transition for c in cells)
            and not any(int(mesh.surfaces[c]) in caps for c in cells)
            and len(cells) != 4
        ):
            where = f"{mesh.components[cells[0]]}.{int(mesh.surfaces[cells[0]])}"
            found[where] = found.get(where, 0) + 1
    return _check("mesh.irregular_nodes", sum(found.values()), where={"by_surface": found})


def _topology(mesh: Mesh) -> list[dict]:
    uses = edge_uses(mesh)
    over = sum(1 for cells in uses.values() if len(cells) > 2)
    free = sum(1 for e, cells in uses.items() if len(cells) == 1 and not _on_symmetry(mesh, e))
    return [
        _check("mesh.node_to_node", over),
        _check("mesh.free_edges", free, where={"tolerance_m": SYMMETRY_TOLERANCE_M}),
    ]


def acceptance_checks(
    mesh: Mesh, record: Mapping[str, Any], controls: Mapping[str, Any], transition: Mapping
) -> list[dict]:
    """Measure the strategy 1 acceptance table of the surface-mesh design on a realized mesh.

    ``record`` is the kernel record (inverted cells, source distance, spacing rules);
    ``transition`` maps each transition component to its transition table. Every other check
    is measured here on the cells and nodes, independently of the requested controls.
    """
    names = set(transition)
    caps = frozenset(
        int(s) for s, f in record.get("surface_families", {}).items() if f == "tip_cap"
    )
    checks = [_check("mesh.inverted", int(record["inverted_count"]))]
    base = frozenset(
        int(s) for s, f in record.get("surface_families", {}).items() if f.startswith("trailing")
    )
    checks += _quality(mesh, names, controls.get("defaults", {}).get("quality", {}), base)
    checks += _topology(mesh)
    checks += _triangle_checks(mesh, transition, caps)
    checks.append(_irregular_nodes(mesh, names, caps))
    families = {int(k): v for k, v in record.get("surface_families", {}).items()}
    checks += body_aspect_band(mesh, families, controls)
    failures = record.get("spacing_rules", {}).get("failures", [None])
    checks.append(_check("mesh.spacing_laws", len(failures)))
    checks.append(
        _check(
            "mesh.source_distance",
            float(record["source_distance_max_m"]),
            float(record["source_distance_tolerance_m"]),
            unit="m",
        )
    )
    return checks + follower_checks(mesh, controls)


#: Component types the generated route meshes as followers of their hosts when the controls
#: give them no table (MESH_STRATEGY 2.3, followers by default).
_DEFAULT_FOLLOWERS = {"wing_fuselage_fairing": "quad_dominant", "fillet": "full_quad"}
#: Keys of a direction table the generated route cannot honor, refused by name.
_ROUTE_REFUSED = {
    "chordwise": ("first_cell", "root_cells"),
    "spanwise": ("first_cell", "root_cells"),
}
_ROUTE = "strategy 1 on the generated route"


def iges_names(path: str | Path) -> set[str]:
    """Return the component names an IGES file carries as 406 form 15 NAME properties."""
    from cfdbender.geometry._iges_write import read_iges

    try:
        document = read_iges(path)
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as error:
        raise InputArtifactError(f"mesh.iges {Path(path).name}: cannot be read: {error}") from error
    names = set()
    for entity in document["entities"]:
        if entity["type"] == 406 and int(entity["d2"][4]) == 15 and len(entity["parameters"]) > 2:
            text = entity["parameters"][2]
            names.add(text.split("H", 1)[1] if "H" in text else text)
    return names


def _members(resolved: Mapping, mesh: Mapping) -> list[str]:
    components = list(resolved["components"])
    configuration = mesh.get("configuration")
    if configuration is None:
        return components
    table = resolved.get("configurations", {})
    if configuration not in table:
        raise InputValueError(
            f"mesh.configuration {configuration!r}: unknown; available {sorted(table)}"
        )
    return list(table[configuration])


def _with_default_followers(raw: dict, components: Mapping, members, warnings) -> dict:
    tables = raw.setdefault("components", {})
    strategy = raw.get("mesh", {}).get("strategy", 1) if isinstance(raw.get("mesh"), dict) else 1
    for name in members:
        kind = components[name]["type"]
        if name not in tables and kind in _DEFAULT_FOLLOWERS:
            hosts = list(components[name].get("hosts", []))
            fill = _DEFAULT_FOLLOWERS[kind] if strategy == 1 else "full_quad"
            tables[name] = {"role": "follower", "follows": hosts, "fill": fill}
            warnings.append(
                f"{name}: no mesh table; follower of {hosts}, fill {tables[name]['fill']}"
            )
        elif name not in tables and kind == "body":
            tables[name] = {"role": "primary"}
            warnings.append(f"{name}: no mesh table; primary body from defaults.body")
        elif name in tables and kind in _DEFAULT_FOLLOWERS:
            tables[name].setdefault("type", kind)
    for name, table in tables.items():
        if name in components and isinstance(table, dict):
            kind = components[name]["type"]
            if kind in ("lifting_surface", "body"):
                table.setdefault("type", kind)
    return raw


def _direction(component: Mapping, defaults: Mapping, direction: str, name: str) -> dict:
    table = {**defaults.get("lifting_surface", {}).get(direction, {}), **component[direction]}
    for key in _ROUTE_REFUSED[direction]:
        if table.get(key):
            raise InputValueError(f"{name}.{direction}.{key}: not available in {_ROUTE}")
    return table


def _lifting_controls(component: Mapping, defaults: Mapping, name: str) -> dict:
    chord = _direction(component, defaults, "chordwise", name)
    span = _direction(component, defaults, "spanwise", name)
    if chord["law"] not in ("geometric_cap", "constant"):
        raise InputValueError(
            f"{name}.chordwise.law {chord['law']!r}: {_ROUTE} meshes geometric_cap or constant"
        )
    if span["law"] != "constant":
        raise InputValueError(
            f"{name}.spanwise.law {span['law']!r}: {_ROUTE} meshes constant "
            "spanwise cells with an optional tip run"
        )
    controls: dict[str, Any] = load_mesh_controls()["lifting_surface"]
    controls.update(
        chord_cells=chord["cells"],
        chord_growth=1.0 if chord["law"] == "constant" else chord.get("ratio", 1.15),
        growth_until=chord.get("growth_until", controls["growth_until"]),
        span_cells=span["cells"],
        tip_cells=span.get("tip_cells", controls["tip_cells"]),
        tip_ratio=span.get("tip_ratio", controls["tip_ratio"]),
        thickness_cells=chord.get("te_cells", controls["thickness_cells"]),
    )
    if chord.get("hinge_rows"):
        controls["hinge_rows"] = tuple(sorted(set(chord["hinge_rows"])))
    return controls


def _body_controls(component: Mapping, spec: Mapping, defaults: Mapping, name: str) -> dict:
    table = {**defaults.get("body", {}), **component}
    controls = load_mesh_controls()["body"]
    if "x_step_m" in table:
        controls["x_cells"] = max(1, round(spec["length"] / table["x_step_m"]))
    elif "x_cells" in table:
        controls["x_cells"] = table["x_cells"]
    if "azimuth_rule" in table:
        controls.update(azimuth_from_rule(spec, table["azimuth_rule"], controls["x_cells"]))
    elif "azimuth_step_deg" in table:
        controls["azimuth_cells"] = max(1, round(180.0 / table["azimuth_step_deg"]))
    elif "azimuth_cells" in table:
        controls["azimuth_cells"] = table["azimuth_cells"]
    return controls


def _transition(controls: Mapping, components: Mapping) -> dict[str, dict]:
    """Return the components that may hold triangles, with their transition tables.

    Lifting surfaces stay structured even when marked flexible; a follower filled
    quad-dominant and a flexible body are transition components.
    """
    result = {}
    default = controls["defaults"].get("transition", {})
    for name, c in controls["components"].items():
        kind = components[name]["type"]
        if kind == "lifting_surface":
            continue
        if c["role"] == "flexible" or c.get("fill") == "quad_dominant":
            result[name] = {**default, **c.get("transition", {})}
    return result


def _route_members(components: Mapping, members: list[str]) -> list[str]:
    """Return the route's member order: one component alone, or body, wing and fairing."""
    if len(members) == 1 and components[members[0]]["type"] in ("body", "lifting_surface"):
        return members
    fairings = [m for m in members if components[m]["type"] == "wing_fuselage_fairing"]
    if len(fairings) == 1:
        body, wing = components[fairings[0]]["hosts"]
        if set(members) == {body, wing, fairings[0]}:
            return [body, wing, fairings[0]]
    raise InputValueError(
        f"mesh.configuration: {_ROUTE} meshes one body or lifting surface alone, or a body, "
        f"its wing and their wing_fuselage_fairing; members {sorted(members)} are not meshed yet"
    )


def _planned_rows(controls: Mapping, l4: Mapping, components: Mapping) -> dict[str, dict]:
    rows = {}
    for name, c in controls["components"].items():
        kind = components[name]["type"]
        row: dict[str, Any] = {"type": kind, "role": c["role"]}
        if c["role"] == "follower":
            row.update(follows=c["follows"], fill=c.get("fill", "full_quad"))
        elif kind == "lifting_surface":
            lc = l4["lifting_surface"]
            row.update(
                chord_cells_per_side=lc["chord_cells"],
                span_cells=lc["span_cells"],
                tip_cells=lc["tip_cells"],
                te_cells=lc["thickness_cells"],
                hinge_rows=list(lc.get("hinge_rows", ())),
                trailing_edge=c.get("trailing_edge", "blunt"),
            )
        elif kind == "body":
            bc = l4["body"]
            row.update(
                x_cells=bc["x_cells"],
                x_step_m=components[name]["length"] / bc["x_cells"],
                azimuth_cells=bc["azimuth_cells"],
            )
        rows[name] = row
    return rows


def plan_strategy1(resolved: Mapping, source: Any, *, base: Path | None = None) -> dict:
    """Validate mesh controls against the configuration and plan strategy 1, without a kernel.

    Membership comes from ``mesh.configuration`` (every component when absent), the IGES names
    from ``mesh.iges`` (the generated component names when absent). A fairing or fillet with no
    table follows its hosts. Returns the validated controls, the kernel controls, the realized
    counts per component, the transition components and every warning; nothing is written.
    """
    raw = (
        tomllib.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, (str, Path))
        else deepcopy(dict(source))
    )
    from cfdbender.geometry.stages import unpack

    components = unpack(resolved)["components"]
    mesh = raw.get("mesh", {}) if isinstance(raw.get("mesh", {}), Mapping) else {}
    members = _members({**resolved, "components": components}, mesh)
    names = set(components)
    if "iges" in mesh:
        names = iges_names(Path(base or ".") / mesh["iges"])
    warnings: list[str] = []
    raw = _with_default_followers(raw, components, members, warnings)
    controls = mesh_controls(raw, members=set(members), names=names)
    if controls["mesh"]["variant"] != "clean":
        raise InputValueError("mesh.variant: the generated mesh route supports only clean")
    if controls["mesh"]["strategy"] != 1:
        raise InputValueError("mesh.strategy 2: the full-grid strategy is not on this route")
    order = _route_members(components, members)
    missing = [m for m in order if m not in controls["components"]]
    if missing:
        raise InputValueError(f"{missing}: no mesh controls; give a table per member")
    l4: dict[str, dict[str, Any]] = load_mesh_controls()
    defaults = controls["defaults"]
    for name, c in controls["components"].items():
        kind = components[name]["type"]
        if c["role"] != "follower" and kind == "lifting_surface":
            l4["lifting_surface"] = _lifting_controls(c, defaults, name)
        elif c["role"] != "follower" and kind == "body":
            l4["body"] = _body_controls(c, components[name], defaults, name)
        if "families" in c:
            warnings.append(f"{name}.families: per-family overrides are not applied by {_ROUTE}")
    return {
        "strategy": 1,
        "variant": controls["mesh"]["variant"],
        "configuration": mesh.get("configuration"),
        "members": order,
        "order": controls["order"],
        "controls": controls,
        "kernel_controls": l4,
        "components": _planned_rows(controls, l4, components),
        "transition": _transition(controls, components),
        "warnings": [*controls["warnings"], *warnings],
    }


def _te_sides(mesh: Mesh, family: list[str]) -> dict[int, str]:
    side: dict[int, str] = {}
    for i, f in enumerate(family):
        for prefix in ("upper", "lower"):
            if f.startswith(prefix):
                side.update(dict.fromkeys(mesh.cells[i], prefix))
    return side


def _te_links(mesh: Mesh, base: set[int], side: Mapping[int, str]):
    links: dict[int, set[int]] = {}
    lines: set[tuple[int, int]] = set()
    for i in base:
        cell = mesh.cells[i]
        for a, b in zip(cell, cell[1:] + cell[:1], strict=True):
            sa, sb = side.get(a, "inside"), side.get(b, "inside")
            if sa == sb and sa != "inside":
                lines.add((min(a, b), max(a, b)))
            elif sa != sb:
                links.setdefault(a, set()).add(b)
                links.setdefault(b, set()).add(a)
    return links, lines


def _te_chain(start: int, links: Mapping[int, set[int]], side: Mapping[int, str]) -> list[int]:
    chain, previous, node = [start], None, start
    while side.get(node) != "lower":
        following = [v for v in links[node] if v != previous and v not in chain]
        if len(following) != 1:
            break
        previous, node = node, following[0]
        chain.append(node)
    return chain


def trailing_edge_groups(
    mesh: Mesh, families: Mapping[int, str], component: str
) -> tuple[set[int], list[list[int]], list[tuple[int, int]]]:
    """Find a lifting component's TE base cells, its across-base node groups and TE lines.

    Base cells are the component's ``trailing*`` family cells. A node on the upper (lower)
    TE line also belongs to an ``upper*`` (``lower*``) skin cell; a node in base cells only is
    inside the base. A group is a connected run of base edges joining different classes,
    ordered upper to lower; the TE line edges join two nodes of the same skin class.
    """
    family = [
        families.get(int(s), "") if name == component else ""
        for s, name in zip(mesh.surfaces, mesh.components, strict=True)
    ]
    base = {i for i, f in enumerate(family) if f.startswith("trailing")}
    side = _te_sides(mesh, family)
    links, lines = _te_links(mesh, base, side)
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(v for v in links if side.get(v) == "upper"):
        chain = [] if start in seen else _te_chain(start, links, side)
        if chain and side.get(chain[-1]) == "lower":
            groups.append(chain)
            seen.update(chain)
    return base, groups, sorted(lines)


def cells_across(groups: list[list[int]]) -> dict:
    """Measure the base cells across every TE station (the settled rule: one quad across)."""
    counts = sorted({len(group) - 1 for group in groups}) or [0]
    worst = max(abs(c - 1) for c in counts)
    return _check("mesh.trailing_edge_one_quad", worst, where={"cells_across": counts})


def _single_valued(values: np.ndarray, over: np.ndarray, tolerance: float) -> bool:
    """Whether ``values`` is a function of ``over``: ``over`` strictly monotone along the curve."""
    steps = np.diff(over)
    return bool(len(steps)) and (
        bool(np.all(steps > tolerance)) or bool(np.all(steps < -tolerance))
    )


def junction_blocks(
    faces: Mapping[str, Mapping[str, Any]],
    junctions: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
) -> dict[str, list[dict]]:
    """Classify each layout block of each face against the junction curves crossing it.

    ``faces`` follows the layout sidecar of the design (face name to ``{"u": [...], "v":
    [...]}``, the iso lines bounding its blocks, each a sorted list of parameters with both
    ends); ``junctions`` maps a face name to the (n, 2) (u, v) samples of each junction
    pcurve on it. A block no junction enters is ``transfinite``; a block a junction crosses as
    a function of one parameter keeps four sides with one curved side
    (``transfinite_curved``, naming the parameter the curve is single valued in); any other
    crossing is ``quad_dominant``, which strategy 2 reports as not full grid.
    """
    result: dict[str, list[dict]] = {}
    for name, layout in faces.items():
        u_lines = np.asarray(layout["u"], dtype=float)
        v_lines = np.asarray(layout["v"], dtype=float)
        curves = [np.asarray(c, dtype=float).reshape(-1, 2) for c in junctions.get(name, [])]
        blocks = []
        for i in range(len(u_lines) - 1):
            for j in range(len(v_lines) - 1):
                box = (u_lines[i], u_lines[i + 1], v_lines[j], v_lines[j + 1])
                blocks.append(_block_kind(name, (i, j), box, curves, tolerance=tolerance))
        result[name] = blocks
    return result


def _block_kind(name, index, box, curves, *, tolerance) -> dict:
    u0, u1, v0, v1 = box
    inside = [
        c[
            (c[:, 0] > u0 + tolerance)
            & (c[:, 0] < u1 - tolerance)
            & (c[:, 1] > v0 + tolerance)
            & (c[:, 1] < v1 - tolerance)
        ]
        for c in curves
    ]
    inside = [c for c in inside if len(c)]
    entry: dict[str, Any] = {"face": name, "block": list(index), "box": [u0, u1, v0, v1]}
    if not inside:
        return {**entry, "kind": "transfinite"}
    if len(inside) == 1:
        c = inside[0]
        for parameter, over, values in (("u", c[:, 0], c[:, 1]), ("v", c[:, 1], c[:, 0])):
            if _single_valued(values, over, tolerance):
                return {**entry, "kind": "transfinite_curved", "single_valued_in": parameter}
    return {**entry, "kind": "quad_dominant"}
