"""Run the shipped readers and audit their mesh against retained face roles.

Pipeline role: S12.C converts the exported assembly, S12.G grids its named
retained faces, and S12 measures the mesh topology and sidecar trailing edges.
The mesher establishes true CAD boundaries before the independent mesh audit.
No geometric trailing-edge search is used for these native exports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cfdbender._errors import GeometryError
from cfdbender.geometry.stages import check_entry, unpack, write_config, write_record
from cfdbender.mesh._refine._obj import read_obj, write_te


def measured(name, value, *, where=None):
    """Return a zero-defect reader check with its measured location."""
    note = (
        f"{name}: measured {value} defects, limit <= 0 defects; "
        f"{json.dumps(where or {}, sort_keys=True)}. "
        "Check the component faces and reader boundaries, then rebuild."
    )
    return check_entry(
        name,
        "passed" if value == 0 else "failed",
        value,
        0,
        unit="defects",
        where=where,
        note=note if value else "",
    )


def convert(folder, manifest):
    """Exercise the shipped converter and compare its read-back face counts."""
    from cfdbender.geometry._iges_write import read_iges
    from cfdbender.geometry.iges_convert import convert_iges

    target = folder / "S12-converted.igs"
    expected = {p["component"]: p["faces"] for p in manifest["pieces"]}
    try:
        document = read_iges(folder / manifest["iges"])
        written = sum(e["type"] == 144 for e in document["entities"])
        errors = abs(written - sum(expected.values()))
        if errors:
            record = {"error": "written face count differs from the manifest", "faces": written}
        else:
            record = convert_iges([folder / manifest["iges"]], target)
            errors = abs(record["faces"] - sum(expected.values()))
            errors += record["components"] != expected
    except GeometryError as error:
        record, errors = {"error": str(error)}, 1
    return {
        "checks": [measured("gate.convert", errors, where={"components": list(expected)})],
        "outputs": {
            "inputs": {"file": manifest["iges"]},
            "converted": str(target),
            "conversion": record,
        },
    }


def holders(mesh):
    """Index undirected mesh edges by all incident face families."""
    edges = defaultdict(list)
    for name, faces in mesh.families.items():
        for face in faces:
            for a, b in zip(face, face[1:] + face[:1], strict=True):
                edges[tuple(sorted((a, b)))].append(name)
    return edges


def edge_key(mesh, edge):
    """Represent an edge geometrically, independent of OBJ node numbering."""
    return tuple(sorted(tuple(float(v) for v in mesh.verts[n]) for n in edge))


def grid(folder, manifest):
    """Grid retained faces once and retain the reader's validated boundary reference."""
    from cfdbender.mesh.iges_grid import grid_iges

    faces = [f for p in manifest["pieces"] for f in p["face_records"]]
    control, target = folder / "S12-grid.toml", folder / "S12.obj"
    config = {
        "defaults": {"remove_inner_faces": False},
        "component": {f["name"]: {"chord": {"count": 16}, "span": {"count": 16}} for f in faces},
    }
    write_config(config, control)
    crossings = []
    try:
        record = grid_iges([folder / f["iges"] for f in faces], control, target)
        mesh = read_obj(target)
        edges = holders(mesh)
        errors = record["conformal"]["conformal_mismatches"]
        crossings = [names for names in edges.values() if len(names) > 2]
        errors += len(crossings)
        errors += set(mesh.families) != {f["name"] for f in faces}
        reference = {
            "boundary": [edge_key(mesh, e) for e, names in edges.items() if len(names) == 1],
            "faces": {n: len(fs) for n, fs in mesh.families.items()},
            "validated": errors == 0,
        }
        write_record(folder / "S12-boundary.json", reference)
    except GeometryError as error:
        record, errors = {"error": str(error)}, 1
    affected = sorted({name for names in crossings for name in names})
    components = sorted(
        p["component"]
        for p in manifest["pieces"]
        if not affected or any(f["name"] in affected for f in p["face_records"])
    )
    check = measured(
        "gate.grid", errors, where={"components": components, "faces": affected} if errors else None
    )
    if errors:
        action = (
            "Add the intended [[dominance]] trimming relationships between these components, "
            "then rebuild."
            if crossings
            else "Repair junctions and face definitions in the configuration, then rebuild."
        )
        check["unit"] = "defects"
        check["note"] = (
            f"gate.grid: measured {errors} defects, limit <= 0 defects; "
            f"components {', '.join(components)}; faces {', '.join(affected) or 'see S12.G.json'}. "
            f"{len(crossings)} edges have more than two incident faces. "
            f"{record.get('error', '')} {action}"
        )
    return {
        "checks": [check],
        "warnings": record.get("warnings", []),
        "outputs": {
            "inputs": {"files": [f["iges"] for f in faces]},
            "mesh": str(target),
            "grid": record,
        },
    }


def trailing_edges(folder, manifest, config, mesh, edges):
    """Find the upper rim of each TE face from retained sidecar roles."""
    checks, results = [], {}
    for piece in manifest["pieces"]:
        name = piece["component"]
        if config["components"][name]["type"] != "lifting_surface":
            continue
        layout = json.loads((folder / piece["layout"]).read_text(encoding="utf-8"))
        roles = {f["name"]: f["role"] for f in layout["faces"]}
        found = [
            edge
            for edge, names in edges.items()
            if len(names) == 2 and {roles.get(n) for n in names} == {"upper", "trailing_edge"}
        ]
        length = sum(float(np.linalg.norm(mesh.verts[a] - mesh.verts[b])) for a, b in found)
        checks.append(
            measured("gate.te_found", int(not found or length <= 0), where={"component": name})
        )
        result = {"edges": len(found), "length_m": length, "source": "sidecar roles"}
        if found:
            path = folder / f"S12-{name}.te.txt"
            write_te(path, "m", np.asarray([(mesh.verts[a] + mesh.verts[b]) / 2 for a, b in found]))
            result["file"] = str(path)
        results[name] = result
    if not checks:
        checks.append(check_entry("gate.te_found", "skipped", note="no lifting pieces"))
    return checks, results


def audit(folder, manifest, resolved):
    """Re-read the OBJ and reject new open seams, triple edges and absent TE rims."""
    mesh = read_obj(folder / "S12.obj")
    reference = json.loads((folder / "S12-boundary.json").read_text(encoding="utf-8"))
    boundary = {tuple(tuple(p) for p in e) for e in reference["boundary"]}
    edges = holders(mesh)
    missing = sum(len(ns) == 1 and edge_key(mesh, e) not in boundary for e, ns in edges.items())
    triple = sum(len(ns) > 2 for ns in edges.values())
    changed = {n: len(fs) for n, fs in mesh.families.items()} != reference["faces"]
    errors = missing + triple + int(changed) + int(not reference["validated"])
    checks, trailing = trailing_edges(folder, manifest, unpack(resolved), mesh, edges)
    return {
        "checks": [
            measured(
                "gate.conformal",
                errors,
                where={"components": [p["component"] for p in manifest["pieces"]]},
            ),
            *checks,
        ],
        "outputs": {
            "inputs": {"mesh": "S12.obj", "boundary": "S12-boundary.json"},
            "new_open_edges": missing,
            "triple_edges": triple,
            "trailing_edges": trailing,
        },
    }


def reader_check(stage, folder, manifest, resolved):
    """Dispatch one reader sub-step without introducing a second assembly scheme."""
    folder = Path(folder)
    if stage == "S12.C":
        return convert(folder, manifest)
    if stage == "S12.G":
        return grid(folder, manifest)
    return audit(folder, manifest, resolved)
