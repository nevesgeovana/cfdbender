"""Meshing a generated parametric configuration into one all-quad surface mesh (FR-1221).

Pipeline role: the workspace row of the geometry meshing. The pure geometry package gives the
placed patches of every component, the spacing laws, the rules and the mesh checks; this module
adds the patches to one OpenCASCADE model, prescribes the edge nodes of each structured patch
from its role (spanwise, chordwise, body band, pole), meshes them transfinite and recombined,
meshes the planar flat tip cap apart from its rim, meshes the wing-fuselage transition ring
quad-dominant, orients the skin outward, measures the spacing rules and the mesh checks, and
returns the mesh with one record.

gmsh is a dependency of the package; :func:`mesh_generated` raises ``KernelUnavailableError``,
naming the library, when it cannot load.

Patch roles. A body band carries its axial station interval in ``Patch.bay`` and a pole patch
carries none. The wing patches of a configuration with a fairing are listed family by family, six
families times (inner, outer, outboard) spanwise bays, then the tip cap, so the junction bay of a
patch is its position in that order (:func:`junction_bays`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cfdbender._textio as _textio
from cfdbender._errors import InputValueError
from cfdbender.geometry._geometry_generate import add_occ_patch, assemble
from cfdbender.geometry._geometry_mesh_junction import (
    VIRTUAL_BASE,
    distribute,
    junction_faces,
)
from cfdbender.geometry._geometry_mesh_occ import (
    CAP_SURFACE,
    MeshPatch,
    Transition,
    curve_endpoints,
    mesh_cap,
    mesh_patches,
    rim_edges,
    rim_loop,
    smooth_faces,
    source_metrics,
    uv_edges,
)
from cfdbender.geometry.config import load_config, resolve_config
from cfdbender.geometry.stages import (
    Mesh,
    Patch,
    assembly_regions,
    body,
    check_body,
    check_lifting,
    check_mesh,
    chord_fraction,
    chord_nodes,
    constant,
    count_classes,
    generate_components,
    insert_seed,
    lifting_surface,
    load_mesh_controls,
    orient_mesh,
    oriented_source_normals,
    repair_inverted,
    tip_cap,
    tip_refinement,
    unpack,
    wing_in_body_frame,
    write_obj,
)
from cfdbender.mesh._gmsh import session

__all__ = [
    "ConfigurationMesh",
    "GeneratedMesh",
    "junction_bays",
    "mesh_configuration",
    "mesh_generated",
]

JUNCTION_BAYS = ("inner", "outer", "outboard")
#: Quad-dominant algorithms of a transition face, in the order a defective face retries them:
#: frontal-Delaunay for quads, frontal-Delaunay, Delaunay, then MeshAdapt.
RING_ALGORITHMS = (8, 6, 5, 1)


@dataclass
class GeneratedMesh:
    """The mesh of a generated configuration and the record of its checks."""

    mesh: Mesh
    record: dict[str, Any]


def junction_bays(patches: list[Patch]) -> list[str | None]:
    """Return the spanwise junction bay of each wing patch of a configuration with a fairing.

    The skin is listed family by family, each with an inner, an outer and an outboard bay; the
    tip cap follows and has none.
    """
    result: list[str | None] = []
    counter = 0
    for patch in patches:
        if patch.family == "tip_cap":
            result.append(None)
        else:
            result.append(JUNCTION_BAYS[counter % 3])
            counter += 1
    return result


def patch_roles(patch: Patch, kind: str, junction: str | None = None) -> dict[str, Any]:
    """Return the roles the seed laws read: family, bay, axial interval and junction bay."""
    roles: dict[str, Any] = {"family": patch.family}
    if patch.bay is not None:
        roles["axial" if kind == "body" else "bay"] = list(patch.bay)
    if junction is not None:
        roles["junction_bay"] = junction
    return roles


def _seed_bay(component, controls, roles, e, seeds):
    a, b = roles["bay"]
    span = tip_refinement(controls["span_cells"], controls["tip_cells"], controls["tip_ratio"])
    # Preserve global stations; a geometric bay boundary must be a mesh station.
    span = np.unique(np.r_[a, span[(span > a) & (span < b)], b])
    span = (span - a) / (b - a)
    insert_seed(seeds, e[1], span)
    insert_seed(seeds, -e[3], span)
    for edge, eta in [(e[0], a), (-e[2], b)]:
        x = chord_nodes(component, controls, roles["family"], eta)
        if x is not None:
            insert_seed(seeds, edge, x)


def _seed_outboard(model, component, controls, roles, e, *, seeds, span_count):
    # Outboard of the transition ring: constant spanwise cells of the standalone width, finer at
    # the tip. The parameter directions of the junction patches differ between families, so the
    # spanwise pair is found from the geometry and each law along its own walking direction.
    def ends(signed):
        a, b = curve_endpoints(model, abs(signed))
        if signed < 0:
            a, b = b, a
        return np.asarray(model.getValue(0, a, [])), np.asarray(model.getValue(0, b, []))

    walks = [e[0], e[1], -e[2], -e[3]]
    rise = [abs(ends(w)[1][1] - ends(w)[0][1]) for w in walks]
    span_pair = (
        (walks[0], walks[2]) if rise[0] + rise[2] > rise[1] + rise[3] else (walks[1], walks[3])
    )
    chord_pair = (walks[1], walks[3]) if span_pair == (walks[0], walks[2]) else (walks[0], walks[2])
    law = tip_refinement(controls["span_cells"], controls["tip_cells"], controls["tip_ratio"])
    half = component["planform"]["span"] / 2
    width = float(law[1] - law[0]) * half
    tip = width * sum(controls["tip_ratio"] ** -j for j in range(1, controls["tip_cells"] + 1))
    lengths = [model.occ.getMass(1, abs(t)) for t in span_pair]
    count = span_count or max(
        controls["tip_cells"] + 1,
        int(round((np.mean(lengths) - tip) / width)) + controls["tip_cells"],
    )
    span = tip_refinement(count, controls["tip_cells"], controls["tip_ratio"])
    for w in span_pair:
        a, b = ends(w)
        insert_seed(seeds, w, span if b[1] > a[1] else 1 - span[::-1])
    for w in chord_pair:
        a, b = ends(w)
        if roles["family"].startswith("trailing"):
            insert_seed(seeds, w, constant(controls["thickness_cells"]))
            continue
        eta = 0.0 if 0.5 * (a[1] + b[1]) < 0.5 * half else 1.0
        x = chord_nodes(component, controls, roles["family"], eta)
        xa, xb = chord_fraction(component, np.array([a, b]))
        insert_seed(seeds, w, x if xb > xa else 1 - x[::-1])


def _seed_body_band(component, controls, roles, e, seeds, *, strict):
    a, b = roles["axial"]
    count = max(1, int(round((b - a) * controls["x_cells"])))
    if strict and abs(count / controls["x_cells"] - (b - a)) > 1e-10:
        raise InputValueError("body: stations must coincide with the constant x mesh spacing")
    axial = constant(count)
    # Native u is azimuth and v is axial in the body bands.
    sectors = 2 * (len(component["azimuths"]) - 1)
    if controls["azimuth_cells"] % sectors:
        raise InputValueError("body: azimuth count must divide the generated half-sector layout")
    azimuth = constant(controls["azimuth_cells"] // sectors)
    insert_seed(seeds, e[0], azimuth)
    insert_seed(seeds, -e[2], azimuth)
    insert_seed(seeds, e[1], axial)
    insert_seed(seeds, -e[3], axial)


def seed_patch(model, component, controls, roles, e, *, seeds, strict=True, span_count=None):
    """Insert the physical spacing laws of one tensor patch into ``seeds`` (edge tag to nodes)."""
    if component["type"] == "lifting_surface" and "bay" in roles:
        _seed_bay(component, controls, roles, e, seeds)
    elif component["type"] == "lifting_surface" and roles.get("junction_bay") == "outboard":
        _seed_outboard(model, component, controls, roles, e, seeds=seeds, span_count=span_count)
    elif component["type"] == "body" and "axial" in roles:
        _seed_body_band(component, controls, roles, e, seeds, strict=strict)


def _edge_polylines(model, edges):
    """Return the realized mesh nodes of each curve, ordered by native parameter, ends included."""
    result = {}
    for e in edges:
        _, xyz, param = model.mesh.getNodes(1, abs(e), includeBoundary=True)
        result[abs(e)] = np.asarray(xyz).reshape(-1, 3)[np.argsort(param)]
    return result


def _patch_items(patches, roles_by_surface):
    return [
        {
            "surface": p.surface,
            "family": roles_by_surface[p.surface]["family"],
            "u": [abs(p.edges[0]), abs(p.edges[2])],
            "v": [abs(p.edges[1]), abs(p.edges[3])],
            "axial": "axial" in roles_by_surface[p.surface],
        }
        for p in patches
    ]


def _component_patches(model, component, source, controls, name, *, split_cap):
    surfaces = [add_occ_patch(model.occ, p) for p in source]
    model.occ.synchronize()
    model.occ.removeAllDuplicates()
    model.occ.synchronize()
    if {s for _, s in model.getEntities(2)} != set(surfaces):
        raise InputValueError("source: sewing changed source face identities")
    entries = [
        (patch_roles(p, component["type"]), s) for p, s in zip(source, surfaces, strict=True)
    ]
    cap = (
        [s for p, s in zip(source, surfaces, strict=True) if p.family == "tip_cap"]
        if split_cap
        else []
    )
    entries = [(r, s) for r, s in entries if s not in cap]
    loops = {s: uv_edges(model, s) for _, s in entries}
    seeds: dict[int, np.ndarray] = {}
    for roles, surface in entries:
        seed_patch(model, component, controls, roles, loops[surface], seeds=seeds)
    if component["type"] == "body":
        lengths = {abs(e): model.occ.getMass(1, abs(e)) for loop in loops.values() for e in loop}
        counts = count_classes(loops, seeds, 4, lengths, component["length"] / controls["x_cells"])
    else:
        counts = count_classes(loops, seeds, controls.get("thickness_cells", 4))
    edge_nodes = {edge: seeds.get(edge, constant(count)) for edge, count in counts.items()}
    patches = [
        MeshPatch(
            s,
            loops[s],
            constant(1),
            constant(1),
            name,
            {abs(e): edge_nodes[abs(e)] for e in loops[s]},
        )
        for _, s in entries
    ]
    return patches, entries, cap


def _mesh_component(gmsh, name, component, controls, source):
    model = gmsh.model
    model.add(name)
    lifting = component["type"] == "lifting_surface"
    patches, entries, cap = _component_patches(
        model, component, source, controls, name, split_cap=lifting
    )
    rim = rim_edges(model, cap) if cap else []
    if cap:
        model.removeEntities([(2, s) for s in cap])
    mesh = mesh_patches(model, patches)
    roles = {s: r for r, s in entries}
    polylines = _edge_polylines(model, {abs(e) for p in patches for e in p.edges})
    check = check_lifting if lifting else check_body
    rows, failures = check(component, controls, _patch_items(patches, roles), polylines)
    planes = {}
    if cap:
        loop = rim_loop(model, mesh, rim)
        mesh, normal = mesh_cap(mesh, loop, name)
        planes[CAP_SURFACE] = (mesh.xyz[loop].mean(axis=0), normal)
    mesh, orientation = orient_mesh(
        mesh, "body" if component["type"] == "body" else "mirror_volume"
    )
    normals, distances = source_metrics(model, mesh, planes)
    normals = oriented_source_normals(mesh, normals)
    record = check_mesh(mesh, normals, distances)
    record["orientation"] = orientation
    record["source_patches"] = len(entries) + len(cap)
    record["spacing_rules"] = {
        "checked_edges": len(rows),
        "failures": failures,
        "passed": not failures,
        "rows": rows,
    }
    record["cap_cells"] = sum(1 for s in mesh.surfaces if s == CAP_SURFACE)
    record["surface_families"] = {
        **{int(s): r["family"] for s, r in roles.items()},
        **{int(s): "tip_cap" for s in cap},
        CAP_SURFACE: "tip_cap",
    }
    return mesh, record


def _wb_sources(resolved, member_names):
    components = unpack(resolved)["components"]
    sources = generate_components(resolved)
    body_name, wing_name = member_names[0], member_names[1]
    roles: dict[int, dict] = {}
    bays = junction_bays(sources[wing_name])
    return components, sources, bays, (body_name, wing_name), roles


def _wb_model(model, resolved, member_names, *, blocks=False):
    """Add, fragment and audit the members; return the state the ring meshing reads.

    With ``blocks`` (strategy 1, delta F.2) the primary faces next to the fairing are
    classified: a block that stays four-sided is meshed structured and only the fairing, with
    the faces that cannot be, is meshed quad-dominant (``junction`` rows in the state).
    """
    components, sources, bays, (body_name, wing_name), roles = _wb_sources(resolved, member_names)
    regions = assembly_regions(resolved, sources)
    groups = {}
    for name in member_names:
        groups[name] = []
        for index, patch in enumerate(sources[name]):
            surface = add_occ_patch(model.occ, patch)
            groups[name].append(surface)
            junction = bays[index] if name == wing_name else None
            roles[surface] = patch_roles(patch, components[name]["type"], junction)
    model.occ.synchronize()
    exposed, audit = assemble(model, groups, {k: regions[k] for k in member_names})
    if (
        audit["free_edges_off_symmetry"]
        or audit["nonmanifold_edges"]
        or audit["non_ring_nonquadrilateral_faces"]
    ):
        raise InputValueError(
            "assembly: source topology gate failed (free, non-manifold or non-quad faces)"
        )
    rings = set(audit["transition_ring_faces"])
    holder = {s: name for name, tags in exposed.items() for s in tags}
    cap = [
        s
        for s, name in holder.items()
        if name == wing_name and roles.get(s, {}).get("family") == "tip_cap" and s not in rings
    ]
    junction: dict[int, dict] = {}
    mesh_owner = dict(holder)
    if blocks and len(member_names) > 2:
        junction = junction_faces(model, rings, holder, member_names[2])
        for s, row in junction.items():
            if row["mesh"] == "structured":
                rings.discard(s)
            else:
                mesh_owner[s] = member_names[2]
    structured = [s for s in holder if s not in rings and s not in cap]
    for s in structured:
        if s not in roles and s not in junction:
            raise InputValueError(f"face {s}: structured face lost its source roles")
    host_wing = wing_in_body_frame(components, body_name, wing_name)
    return {
        "components": components,
        "host_wing": host_wing,
        "roles": roles,
        "audit": audit,
        "rings": rings,
        "holder": holder,
        "cap": cap,
        "structured": structured,
        "junction": junction,
        "mesh_owner": mesh_owner,
    }


def _junction_loops(model, state, lengths):
    """Add the count loops of the structured junction faces; return loops and composite sides.

    A face of four curves is walked from UV (0, 0) like a source patch; a side of several
    curves becomes one virtual curve of the count classes, split afterwards by length.
    """
    loops, composite = {}, {}
    for s, row in state["junction"].items():
        if row["mesh"] != "structured":
            continue
        if "sides" not in row:
            loops[s] = uv_edges(model, s)
            continue
        loop = []
        for side in row["sides"]:
            if len(side) == 1:
                loop.append(side[0])
                continue
            virtual = VIRTUAL_BASE + len(composite)
            composite[virtual] = (s, [abs(e) for e in side])
            lengths[virtual] = sum(lengths[abs(e)] for e in side)
            loop.append(virtual)
        loops[s] = tuple(loop)
    return loops, composite


def _carry_junction_seeds(model, state, seeds):
    """Give the free side of a four-curve junction block the law of its opposite side.

    The junction curve of a wing bay next to the fairing carries no law of its own; walked in
    the same parameter direction as the opposite side, it takes that side's nodes, so the
    chordwise rows of the block stay the rows of the wing.
    """
    for s, row in state.get("junction", {}).items():
        if row["mesh"] != "structured" or "sides" in row:
            continue
        e = uv_edges(model, s)
        for i in range(4):
            mine, other = e[i], -e[(i + 2) % 4]
            if abs(mine) in seeds or abs(other) not in seeds:
                continue
            nodes = seeds[abs(other)]
            insert_seed(seeds, mine, nodes if other > 0 else 1 - nodes[::-1])


def _wb_nodes(model, state, controls, member_names, ring_cell_m):
    """Seed every structured edge, solve the count classes, and return (nodes, loops, seeds)."""
    names = {"body": member_names[0], "wing": member_names[1]}
    components, host_wing = state["components"], state["host_wing"]
    structured, holder, roles, cap = (
        state["structured"],
        state["holder"],
        state["roles"],
        state["cap"],
    )
    junction = state.get("junction", {})
    loops = {s: uv_edges(model, s) for s in structured if s not in junction}
    seeds: dict[int, np.ndarray] = {}
    body_c, wing_c = components[names["body"]], components[names["wing"]]
    outboard = [s for s in structured if roles.get(s, {}).get("junction_bay") == "outboard"]
    lc = controls["lifting_surface"]
    law = np.diff(np.asarray(tip_refinement(lc["span_cells"], lc["tip_cells"], lc["tip_ratio"])))
    width = float(law[0]) * wing_c["planform"]["span"] / 2
    tip_length = width * sum(lc["tip_ratio"] ** -j for j in range(1, lc["tip_cells"] + 1))

    def span_length(s):
        lengths = [model.occ.getMass(1, abs(t)) for t in loops[s]]
        return max((lengths[0] + lengths[2]) / 2, (lengths[1] + lengths[3]) / 2)

    mean_span = np.mean([span_length(s) for s in outboard]) if outboard else 0.0
    span_count = (
        max(lc["tip_cells"] + 1, int(round((mean_span - tip_length) / width)) + lc["tip_cells"])
        if outboard
        else None
    )
    for s in structured:
        if s in junction:
            continue
        is_body = holder[s] == names["body"]
        seed_patch(
            model,
            body_c if is_body else host_wing,
            controls["body" if is_body else "lifting_surface"],
            roles[s],
            loops[s],
            seeds=seeds,
            strict=False,
            span_count=span_count,
        )
    _carry_junction_seeds(model, state, seeds)
    all_edges = {
        t for s in holder if s not in cap for _, t in model.getBoundary([(2, s)], oriented=False)
    }
    lengths = {e: model.occ.getMass(1, e) for e in all_edges}
    x_step = body_c["length"] / controls["body"]["x_cells"]
    extra, composite = _junction_loops(model, state, lengths)
    loops.update(extra)
    counts = count_classes(
        loops,
        seeds,
        1,
        {e: lengths[e] for e in {abs(t) for lp in loops.values() for t in lp}},
        x_step,
    )
    nodes = {}
    for virtual, (surface, edges) in composite.items():
        cells = distribute(counts[virtual], [lengths[e] for e in edges])
        if cells is None:
            raise InputValueError(
                f"face {surface}: junction side of {len(edges)} curves takes "
                f"{counts[virtual]} cells; mesh the face with the transition"
            )
        nodes.update({e: constant(c) for e, c in zip(edges, cells, strict=True)})
    for e in all_edges:
        if e in nodes:
            continue
        if e in seeds:
            nodes[e] = seeds[e]
        elif e in counts:
            nodes[e] = constant(counts[e])
        else:
            nodes[e] = constant(max(1, int(round(lengths[e] / ring_cell_m))))
    return nodes, loops


def _wb_patches(model, state, nodes, loops, member_names):
    patches = []
    holders = state.get("mesh_owner", state["holder"])
    for s in sorted(state["structured"]):
        sides = state.get("junction", {}).get(s, {}).get("sides")
        walk = tuple(side[0] for side in sides) if sides else loops[s]
        edges = [e for side in sides for e in side] if sides else loops[s]
        boundary = {abs(t): nodes[abs(t)] for t in edges}
        patches.append(
            MeshPatch(s, walk, constant(1), constant(1), holders[s], boundary, sides=sides)
        )
    for s in sorted(state["rings"]):
        edges = [t for _, t in model.getBoundary([(2, s)], oriented=False)]
        patches.append(Transition(s, {t: nodes[t] for t in edges}, "wing-fuselage", holders[s]))
    return patches


def _wb_pass(model, state, controls, patches, rim, *, member_names):
    """Mesh once, cap, orient, measure and repair; return mesh, record, orientation, repairs."""
    rings = state["rings"]
    mesh = smooth_faces(model, mesh_patches(model, patches), rings)
    loop = rim_loop(model, mesh, rim)
    mesh, normal = mesh_cap(mesh, loop, member_names[1])
    planes = {CAP_SURFACE: (mesh.xyz[loop].mean(axis=0), normal)}
    mesh, orientation = orient_mesh(mesh, "mirror_volume")
    normals, distances = source_metrics(model, mesh, planes)
    normals = oriented_source_normals(mesh, normals)
    record = check_mesh(mesh, normals, distances, transition_surfaces=rings)
    repairs = 0
    for _ in range(8):
        if not record["inverted_cells"]:
            break
        repaired, count = repair_inverted(mesh, normals, record["inverted_cells"])
        if not count:
            break
        mesh, repairs = repaired, repairs + count
        record = check_mesh(mesh, normals, distances, transition_surfaces=rings)
    return mesh, record, orientation, repairs


def _wb_spacing_rules(model, state, controls, patches, member_names):
    body_name, wing_name = member_names[0], member_names[1]
    roles = state["roles"]
    # A junction block cut from a source patch carries no source law of its own.
    kept = [p for p in patches if isinstance(p, MeshPatch) and p.surface in roles]
    bodies = [p for p in kept if p.component == body_name]
    wings = [p for p in kept if p.component == wing_name]
    poly = _edge_polylines(model, {abs(e) for p in bodies + wings for e in p.edges})
    body_rows, body_fail = check_body(
        state["components"][body_name], controls["body"], _patch_items(bodies, roles), poly
    )
    wing_rows, wing_fail = check_lifting(
        state["host_wing"], controls["lifting_surface"], _patch_items(wings, roles), poly
    )
    return body_rows + wing_rows, body_fail + wing_fail


def _mesh_wb(
    gmsh,
    resolved,
    controls,
    member_names,
    ring_cell_m,
    *,
    ring_size_limits,
    ring_plan,
    blocks=False,
):
    from cfdbender.geometry._geometry_mesh_s1 import sliver_surfaces

    model = gmsh.model
    model.add("assembly")
    state = _wb_model(model, resolved, member_names, blocks=blocks)
    nodes, loops = _wb_nodes(model, state, controls, member_names, ring_cell_m)
    patches = _wb_patches(model, state, nodes, loops, member_names)
    rim = rim_edges(model, state["cap"])
    model.removeEntities([(2, s) for s in state["cap"]])
    rings = state["rings"]
    if ring_size_limits:
        # Very short ring edges would otherwise pull tiny cells over whole faces.
        gmsh.option.setNumber("Mesh.MeshSizeMin", ring_size_limits[0])
        gmsh.option.setNumber("Mesh.MeshSizeMax", ring_size_limits[1])
    plan = {s: list((ring_plan or {}).get(s, RING_ALGORITHMS)) for s in rings}
    attempt = {s: 0 for s in rings}
    history, rows, failures = [], [], []
    for _ in range(len(RING_ALGORITHMS) * 3):
        for s in rings:
            model.mesh.setAlgorithm(2, s, plan[s][attempt[s]])
        model.mesh.clear()
        mesh, record, orientation, repairs = _wb_pass(
            model, state, controls, patches, rim, member_names=member_names
        )
        if not history:
            rows, failures = _wb_spacing_rules(model, state, controls, patches, member_names)
        bad = {mesh.surfaces[i] for i in record["inverted_cells"] + record["degenerate_cells"]}
        bad |= sliver_surfaces(mesh, rings)
        history.append({"defective_surfaces": sorted(int(x) for x in bad)})
        if not bad or not bad <= rings:
            break
        exhausted = [s for s in bad if attempt[s] + 1 >= len(plan[s])]
        if exhausted:
            record["exhausted_transition_faces"] = sorted(int(x) for x in exhausted)
            break
        for s in bad:
            attempt[s] += 1
    record["ring_algorithm_history"] = history
    record["inverted_triangle_edge_flips"] = repairs
    record["ring_algorithms"] = {int(s): int(plan[s][attempt[s]]) for s in sorted(rings)}
    record["orientation"] = orientation
    record["transition_surfaces"] = sorted(rings)
    record["component_cells"] = {n: mesh.components.count(n) for n in member_names}
    record["structured_faces"] = len(state["structured"])
    if state["junction"]:
        record["junction_faces"] = [
            {k: v for k, v in row.items() if k != "sides"}
            for _, row in sorted(state["junction"].items())
        ]
    record["triangles_by_surface"] = {
        int(s): sum(1 for i, c in enumerate(mesh.cells) if len(c) == 3 and mesh.surfaces[i] == s)
        for s in rings
    }
    record["spacing_rules"] = {
        "checked_edges": len(rows),
        "failures": failures,
        "passed": not failures,
        "rows": rows,
    }
    keys = (
        "free_edges_off_symmetry",
        "nonmanifold_edges",
        "transition_ring_faces",
        "non_ring_nonquadrilateral_faces",
    )
    record["audit"] = {k: state["audit"][k] for k in keys}
    record["surface_families"] = {
        **{int(s): r["family"] for s, r in state["roles"].items() if "family" in r},
        CAP_SURFACE: "tip_cap",
    }
    return mesh, record


def _assembly_members(resolved):
    components = unpack(resolved)["components"]
    fairings = [n for n, c in components.items() if c["type"] == "wing_fuselage_fairing"]
    if len(fairings) != 1:
        raise InputValueError(
            f"component: assembly meshing needs exactly one wing_fuselage_fairing, "
            f"found {fairings}; "
            "name one component to mesh it alone"
        )
    body_name, wing_name = components[fairings[0]]["hosts"]
    return body_name, wing_name, fairings[0]


def mesh_generated(
    config: Any,
    *,
    component: str | None = None,
    controls: Any = None,
    obj: str | Path | None = None,
) -> GeneratedMesh:
    """Mesh a generated parametric configuration into an all-quad surface mesh.

    Parameters
    ----------
    config : mapping or path
        A configuration (resolved or not) or the path of its TOML file.
    component : str, optional
        Name of one body or lifting surface to mesh alone in its own frame (a lifting surface
        carries its flat tip cap). When omitted, the whole assembly of a body, its attached wing
        and the wing-fuselage fairing is meshed, the fairing faces as one quad-dominant
        transition ring.
    controls : mapping or path, optional
        Mesh controls overriding the defaults, by table (``lifting_surface``, ``body``,
        ``checks``, ``assembly``) as a mapping or a TOML file.
    obj : str or Path, optional
        Write the mesh as OBJ, one group per component.

    Returns
    -------
    GeneratedMesh
        The oriented mesh and its record: counts, the acceptance checks, the spacing rules
        measured on the realized edge nodes, the orientation, and the ``warnings`` of the
        configuration: every key it skipped.

    Raises
    ------
    InputValueError
        For an unknown component, an unsupported assembly or controls, a topology gate that
        fails, or a mesh that cannot be built all-quad.
    cfdbender.mesh._extras.MissingExtraError
        When the gmsh library cannot load (``KernelUnavailableError``).
    """
    resolved = load_config(config) if isinstance(config, (str, Path)) else resolve_config(config)
    if isinstance(controls, dict) and "mesh" in controls:
        result = mesh_configuration(resolved, controls, obj=obj)
        assert result.mesh is not None
        return GeneratedMesh(mesh=result.mesh, record=result.record)
    values = load_mesh_controls(controls)
    components = unpack(resolved)["components"]
    if component is not None and component not in components:
        raise InputValueError(f"component: {component!r} unknown; available: {sorted(components)}")
    members = None if component is not None else _assembly_members(resolved)
    with session("meshing a generated parametric configuration") as gmsh:
        if component is not None:
            spec = components[component]
            if spec["type"] not in ("body", "lifting_surface"):
                raise InputValueError(f"{component}.type: only body and lifting_surface mesh alone")
            if spec["type"] == "body":
                source = body(spec)
            else:
                source = lifting_surface(spec) + tip_cap(spec)
            mesh, record = _mesh_component(gmsh, component, spec, values[spec["type"]], source)
        else:
            assert members is not None
            mesh, record = _mesh_wb(
                gmsh, resolved, values, members, 0.15, ring_size_limits=(0.06, 0.3), ring_plan=None
            )
    if obj is not None:
        write_obj(obj, mesh)
    return GeneratedMesh(mesh=mesh, record={**record, "warnings": list(resolved["warnings"])})


@dataclass
class ConfigurationMesh:
    """A strategy 1 mesh of a configuration (None on a dry run), its record and written files."""

    mesh: Mesh | None
    record: dict[str, Any]
    written: dict[str, str]


def _kernel_mesh(plan, resolved):
    """Mesh the planned members in one kernel session; return mesh, record, entity dimensions."""
    components = unpack(resolved)["components"]
    members = plan["members"]
    values = plan["kernel_controls"]
    with session("meshing a configuration with strategy 1") as gmsh:
        if len(members) == 1:
            name = members[0]
            spec = components[name]
            source = body(spec) if spec["type"] == "body" else lifting_surface(spec) + tip_cap(spec)
            mesh, record = _mesh_component(gmsh, name, spec, values[spec["type"]], source)
        else:
            # The fairing is meshed from the sizes of its structured neighbors (delta F.2).
            body_spec = components[members[0]]
            mesh, record = _mesh_wb(
                gmsh,
                resolved,
                values,
                members,
                body_spec["length"] / values["body"]["x_cells"],
                ring_size_limits=None,
                ring_plan=None,
                blocks=True,
            )
        dims = [dim for dim, _ in gmsh.model.getEntities() if dim >= 2]
    return mesh, record, dims


def _trailing_edges(mesh, record, plan):
    """Collapse the bases asked for and return the mesh, its TE checks and the TE line edges."""
    from cfdbender.geometry._geometry_mesh_s1 import (
        cells_across,
        collapse_trailing_edge,
        merge_trailing_edge_base,
        trailing_edge_groups,
    )

    checks, marked = [], []
    families = record.get("surface_families", {})
    for name, row in plan["components"].items():
        if (
            row["type"] != "lifting_surface"
            or row["role"] == "follower"
            or name not in mesh.components
        ):
            continue
        base, groups, lines = trailing_edge_groups(mesh, families, name)
        cap = {i for i, s in enumerate(mesh.surfaces) if families.get(int(s)) == "tip_cap"}
        soft = frozenset(
            cap | {i for i, n in enumerate(mesh.components) if n in plan["transition"]}
        )
        if row.get("trailing_edge") != "collapsed":
            # A base generated as two patches is merged to one quad across (MESH_DESIGN 8).
            mesh = merge_trailing_edge_base(mesh, groups, base, reducible=soft)
            base, groups, lines = trailing_edge_groups(mesh, families, name)
            checks.append(cells_across(groups))
            marked += [tuple(int(t) for t in mesh.node_tags[[a, b]]) for a, b in lines]
            continue
        checks.append(cells_across(groups))
        tolerance = plan["controls"]["components"][name].get("collapse_tolerance_m")
        if tolerance is None:
            raise InputValueError(
                f"{name}.collapse_tolerance_m: required with trailing_edge collapsed"
            )
        upper = {g[0] for g in groups}
        marked += [
            tuple(int(t) for t in mesh.node_tags[[a, b]])
            for a, b in lines
            if a in upper and b in upper
        ]
        mesh, collapsed = collapse_trailing_edge(
            mesh, groups, base, tolerance_m=float(tolerance), reducible=soft
        )
        checks += collapsed
    tags = {int(t): i for i, t in enumerate(mesh.node_tags)}
    return mesh, checks, [(tags[a], tags[b]) for a, b in marked]


def _counts(mesh):
    quads = sum(len(c) == 4 for c in mesh.cells)
    tris = sum(len(c) == 3 for c in mesh.cells)
    return {
        "vertices": len(mesh.xyz),
        "quads": quads,
        "tris": tris,
        "pct_quad": 100.0 * quads / max(1, len(mesh.cells)),
        "by_component": {n: mesh.components.count(n) for n in dict.fromkeys(mesh.components)},
    }


def _write_outputs(mesh, record, marks, *, out, descriptor, fsm_template):
    import json

    from cfdbender._digest import file_sha256

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


def _summary(plan, digest):
    return {
        "strategy": plan["strategy"],
        "variant": plan["variant"],
        "configuration": plan["configuration"],
        "members": plan["members"],
        "order": plan["order"],
        "controls_sha256": digest,
        "components": plan["components"],
        "transition_components": sorted(plan["transition"]),
        "warnings": plan["warnings"],
    }


def _strategy(controls: Any) -> Any:
    """Return ``mesh.strategy`` of the controls (a mapping or a TOML path), 1 when absent."""
    import tomllib

    raw = (
        tomllib.loads(Path(controls).read_text(encoding="utf-8"))
        if isinstance(controls, (str, Path))
        else controls
    )
    mesh = raw.get("mesh", {}) if isinstance(raw, dict) else {}
    return mesh.get("strategy", 1) if isinstance(mesh, dict) else 1


def mesh_configuration(
    config: Any,
    controls: Any,
    *,
    dry_run: bool = False,
    out: str | Path | None = None,
    descriptor: str | None = None,
    fsm_template: str | Path | None = None,
    obj: str | Path | None = None,
) -> ConfigurationMesh:
    """Mesh a generated configuration from its mesh controls.

    Strategy 1 is meshed here; ``mesh.strategy = 2`` (full grid) goes to
    :func:`~cfdbender.geometry._geometry_mesh_s2.mesh_full_grid` with the same arguments.

    Parameters
    ----------
    config : mapping or path
        A configuration (resolved or not) or the path of its TOML file.
    controls : mapping or path
        The mesh controls: ``[mesh]`` (strategy 1, configuration, variant, iges, output),
        ``[defaults]`` and one ``[components.<name>]`` table per member.
    dry_run : bool
        Validate and plan only: the record lists the realized counts and the warnings, no
        kernel is opened and nothing is written.
    out : str or Path, optional
        Folder for ``<descriptor>.obj``, ``<descriptor>.mesh.json`` and, with
        ``fsm_template``, ``<descriptor>.fsm``; an existing file is refused.
    descriptor : str, optional
        File stem; ``mesh.output.descriptor`` when absent, else ``mesh``.
    fsm_template : str or Path, optional
        Not supported: a saved simulation of another solver is refused; the OBJ is written.
    obj : str or Path, optional
        Write only the OBJ here.

    Returns
    -------
    ConfigurationMesh
        The mesh (None on a dry run), the generation record (plan, counts, every acceptance
        check with its measured value, ``accepted``) and the written files.

    Raises
    ------
    InputValueError
        For a schema error (named E01 to E16) or a control this route cannot honor.
    cfdbender.mesh._extras.MissingExtraError
        When the gmsh library cannot load (``KernelUnavailableError``).
    """
    import hashlib
    import json

    from cfdbender.geometry._geometry_mesh_s1 import (
        acceptance_checks,
        plan_strategy1,
        surface_only_check,
    )

    if _strategy(controls) == 2:
        from cfdbender.geometry._geometry_mesh_s2 import mesh_full_grid

        result = mesh_full_grid(
            config,
            controls,
            dry_run=dry_run,
            out=out,
            descriptor=descriptor,
            fsm_template=fsm_template,
        )
        if obj is not None and result.mesh is not None:
            write_obj(obj, result.mesh)
            result.written["obj"] = str(obj)
        return result
    base = Path(controls).parent if isinstance(controls, (str, Path)) else None
    resolved = load_config(config) if isinstance(config, (str, Path)) else resolve_config(config)
    plan = plan_strategy1(resolved, controls, base=base)
    raw = (
        Path(controls).read_bytes()
        if base is not None
        else json.dumps(controls, sort_keys=True, default=str).encode()
    )
    record: dict[str, Any] = _summary(plan, hashlib.sha256(raw).hexdigest())
    record["warnings"] = [*resolved["warnings"], *record["warnings"]]
    if dry_run:
        return ConfigurationMesh(None, {**record, "dry_run": True}, {})
    mesh, kernel, dims = _kernel_mesh(plan, resolved)
    mesh, te_checks, marks = _trailing_edges(mesh, kernel, plan)
    checks = acceptance_checks(mesh, kernel, plan["controls"], plan["transition"])
    checks += [surface_only_check(dims), *te_checks]
    record.update(
        dry_run=False,
        counts=_counts(mesh),
        checks=checks,
        accepted=all(c["status"] != "failed" for c in checks),
        kernel={
            k: kernel[k]
            for k in (
                "orientation",
                "spacing_rules",
                "junction_faces",
                "ring_algorithms",
                "ring_algorithm_history",
            )
            if k in kernel
        },
    )
    written: dict[str, str] = {}
    if obj is not None:
        write_obj(obj, mesh)
        written["obj"] = str(obj)
    if out is not None:
        stem = descriptor or plan["controls"]["mesh"].get("output", {}).get("descriptor", "mesh")
        written.update(
            _write_outputs(mesh, record, marks, out=out, descriptor=stem, fsm_template=fsm_template)
        )
    return ConfigurationMesh(mesh, record, written)
