"""Export the retained assembly faces and their measured source provenance.

Pipeline role: S11 writes the faces still owned by each piece after S10. The
manifest records measurements made on the live model, before any reimport.
Layout sidecars carry source bases and lines without resampling the geometry.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from cfdbender._errors import InputArtifactError
from cfdbender._textio import write_text
from cfdbender.geometry._boundary import sha256_file
from cfdbender.geometry._geom_area import measure_area
from cfdbender.geometry._geom_area_domain import NUMERICAL_RTOL
from cfdbender.geometry._geom_assembly import Assembly
from cfdbender.geometry._geom_parts import plain
from cfdbender.geometry.stages import (
    build_id,
    check_entry,
    manifest_checks,
    manifest_record,
    parameter_id,
    sidecar_identity,
    surface_id,
    unpack,
    write_record,
)
from cfdbender.geometry.stages import junctions as jn
from cfdbender.mesh._gmsh_worker import run_isolated

#: Absolute geometric verification tolerance in metres; not an area acceptance limit.
_CORRESPONDENCE_M = 1e-7


def _face_key(document, entity):
    """Include the complete trimmed entity, excluding its name property pointers."""
    from cfdbender.geometry._iges_write import surface_key

    bare = {**entity, "parameters": entity["parameters"][: 5 + int(entity["parameters"][3])]}
    return surface_key(document, bare)


def _label_faces(path, references):
    """Bind directory labels to isolated trimmed entities, never just their base."""
    from cfdbender.geometry._iges_write import read_iges, serialize_iges

    keys = {}
    for identity, reference in references:
        document = read_iges(reference)
        faces = [e for e in document["entities"] if e["type"] == 144]
        if len(faces) != 1:
            raise InputArtifactError(f"{reference}: expected one retained face. Rebuild.")
        key = _face_key(document, faces[0])
        if key in keys:
            raise InputArtifactError(f"{path}: duplicated retained face identity. Rebuild.")
        keys[key] = identity
    document = read_iges(path)
    order = []
    for entity in document["entities"]:
        if entity["type"] != 144:
            continue
        identity = keys.get(_face_key(document, entity))
        if identity is None or identity in order:
            raise InputArtifactError(f"{path}: unverified retained face identity. Rebuild.")
        entity["d2"][7] = identity
        order.append(identity)
    if len(order) != len(keys):
        raise InputArtifactError(f"{path}: missing retained face identity. Rebuild.")
    write_text(path, serialize_iges(document), encoding="ascii")
    return order


def _face_geometry(path):
    """Fingerprint complete trimmed entities in directory order and file units."""
    from cfdbender.geometry._iges_write import read_iges

    document = read_iges(path)
    return [
        sha256(
            json.dumps(
                [document["global"][12:15], _face_key(document, entity)], separators=(",", ":")
            ).encode("ascii")
        ).hexdigest()
        for entity in document["entities"]
        if entity["type"] == 144
    ]


def _face_property_names(path):
    """Read the explicit NAME pointer of each trimmed face, refusing broken links."""
    from cfdbender.geometry._iges_write import read_iges

    document = read_iges(path)
    by_id = {e["id"]: e for e in document["entities"]}
    names = []
    for entity in document["entities"]:
        if entity["type"] != 144:
            continue
        name = ""
        try:
            values = entity["parameters"]
            tail = values[5 + int(values[3]) :]
            if len(tail) == 3 and tail[:2] == ["0", "1"]:
                prop = by_id[int(tail[2])]
                parameters = prop["parameters"]
                if (
                    prop["type"] == 406
                    and int(prop["d2"][4]) == 15
                    and len(parameters) == 3
                    and parameters[:2] == ["406", "1"]
                ):
                    count, separator, value = parameters[2].partition("H")
                    if separator and int(count) == len(value):
                        name = value
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        names.append(name)
    return names


def _export_units(resolved, names):
    """Choose the definition's mm for propeller-only groups; other groups use kernel metres."""
    components = unpack(resolved or {}).get("components", {})
    if names and all(
        components.get(n, {}).get("type") == "propeller"
        and components[n].get("definition", {}).get("output", {}).get("units") == "mm"
        for n in names
    ):
        return {"unit": "mm"}
    return {}


def write_assembly(
    assembly: Assembly,
    target: Path,
    *,
    resolved=None,
    configuration=None,
    variant="clean",
    selection=None,
) -> dict[str, Any]:
    """Write one file per piece and per face, plus the combined sewing input.

    Per-face files let a fresh kernel import each face alone, so adjacency
    healing cannot conceal a missing trim or an excessive face area.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = assembly.export_state()
    manifest["iges"] = target.name
    manifest["components"] = dict(manifest["components"])
    manifest["face_orders"] = {}
    face_number = 0
    source_free: list[list[list[float]]] = []
    for name, faces in assembly.faces.items():
        piece = manifest["components"][name]
        piece["iges"] = f"{name}.igs"
        piece["layout"] = f"{name}.layout.json"
        units = _export_units(resolved, [name])
        assembly.k.export_iges({name: faces}, target.parent / piece["iges"], **units)
        topology = assembly.k.inspect_file(target.parent / piece["iges"], jn.SEW_M)
        piece["free_edges"] = sum(len(holders) == 1 for _, holders in topology.edges)
        source_free.extend(
            points.tolist() for points, holders in topology.edges if len(holders) == 1
        )
        model = assembly.models[name]
        bases = {
            key: {
                "controls": p.controls.tolist(),
                "knots_u": p.knots_u,
                "knots_v": p.knots_v,
                "mult_u": p.multiplicities_u,
                "mult_v": p.multiplicities_v,
                "degree_u": p.degree_u,
                "degree_v": p.degree_v,
            }
            for key, p in model.bases.items()
        }
        for i, (face, entry) in enumerate(zip(faces, piece["faces"], strict=True)):
            face_number += 1
            entry["iges_identity"] = f"F{face_number:07d}"
            entry["centroid_m"] = list(assembly.k.centroid(face))
            entry["bbox_m"] = plain(assembly.k.bbox(face))
            entry["area_m2"], entry["area_error_m2"] = measure_area(assembly.k, face)
            entry["iges"] = f"{name}.face-{i}.igs"
            assembly.k.export_iges({entry["name"]: [face]}, target.parent / entry["iges"], **units)
            entry["source_faces"] = [t.to_dict() for t in model.trims if t.base == entry["base"]]
            centre = np.mean(assembly.k.uv_bounds(face), axis=0)
            roles = set()
            for trim in model.trims:
                bounds = np.asarray(trim.loop).reshape(-1, 2)
                if (
                    trim.base == entry["base"]
                    and np.all(centre >= bounds.min(axis=0))
                    and np.all(centre <= bounds.max(axis=0))
                ):
                    roles.add(trim.role)
            if len(roles) != 1:
                raise InputArtifactError(
                    f"{entry['name']}: retained face has no unique source role", kind="geometry"
                )
            entry["role"] = roles.pop()
            entry["lines"] = model.layout["lines"]
        references = [
            (entry["iges_identity"], target.parent / entry["iges"]) for entry in piece["faces"]
        ]
        for entry in piece["faces"]:
            path = target.parent / entry["iges"]
            manifest["face_orders"][entry["iges"]] = _label_faces(
                path, [(entry["iges_identity"], path)]
            )
        manifest["face_orders"][piece["iges"]] = _label_faces(
            target.parent / piece["iges"], references
        )
        piece["sha256"] = sha256_file(target.parent / piece["iges"])
        write_record(
            target.parent / piece["layout"],
            plain(
                {
                    **model.layout,
                    "bases": bases,
                    "base_order": list(model.bases),
                    "source_trims": [t.to_dict() for t in model.trims],
                    "faces": piece["faces"],
                    "source_surface_id": model.source_surface_id,
                }
            ),
        )
    if not assembly.plan:
        manifest["input_free_edges"] = source_free
    combined_units = _export_units(resolved, assembly.faces)
    assembly.k.export_iges(assembly.faces, target, **combined_units)
    with TemporaryDirectory(prefix="face-identities-", dir=target.parent) as temporary:
        references = []
        for name, faces in assembly.faces.items():
            entries = manifest["components"][name]["faces"]
            for face, entry in zip(faces, entries, strict=True):
                reference = target.parent / entry["iges"]
                if _export_units(resolved, [name]) != combined_units:
                    # Compare entities in the combined file's units, including
                    # a millimetre component inside a metre assembly.
                    reference = Path(temporary) / entry["iges"]
                    assembly.k.export_iges({entry["name"]: [face]}, reference, **combined_units)
                references.append((entry["iges_identity"], reference))
        manifest["face_orders"][target.name] = _label_faces(target, references)
    manifest["face_geometry"] = {
        file: _face_geometry(target.parent / file) for file in manifest["face_orders"]
    }
    config = (
        unpack(resolved)
        if resolved is not None
        else {
            "components": {
                n: {"version": p.version, "source_surface_id": p.source_surface_id}
                for n, p in assembly.models.items()
            }
        }
    )
    parameter = parameter_id(config)
    try:
        package = metadata.version("cfdbender")
    except metadata.PackageNotFoundError:
        package = "0.0.0+uninstalled"
    identity = surface_id(
        {n: list(p.bases.values()) for n, p in assembly.models.items()},
        {n: p.trims for n, p in assembly.models.items()},
    )
    coverage: dict[str, Any] = next(
        (r["outputs"].get("pieces", {}) for r in assembly.records if r["stage"] == "S10.D8b"), {}
    )
    manifest = manifest_record(
        manifest,
        configuration=configuration,
        identities={
            "parameter_id": parameter,
            "surface_id": identity,
            "build_id": build_id(parameter, package, assembly.k.name, assembly.k.version),
        },
        dominance=config.get("dominance", []),
        coverage=coverage,
    )
    manifest["variant"] = variant
    manifest["control_selection"] = selection or {}
    write_record(target.parent / "manifest.json", manifest)
    return {
        "iges": str(target),
        "manifest": str(target.parent / "manifest.json"),
        "surface_id": manifest["surface_id"],
        "build_id": manifest["build_id"],
        "faces": {name: len(faces) for name, faces in assembly.faces.items()},
        "warnings": [],
        "audit": {"source": "retained assembly faces"},
        "sources": {
            n: {"version": p.version, "source_surface_id": p.source_surface_id}
            for n, p in assembly.models.items()
        },
    }


def check_export(stage, resolved, *, configuration, members, out, source) -> dict[str, Any]:
    """Measure one export sub-step against the retained source record."""
    result = _check_one(
        stage, resolved, configuration=configuration, members=members, out=out, source=source
    )
    for index, alternative in enumerate(source.get("alternatives", []), 1):
        measured = _check_one(
            stage,
            resolved,
            configuration=configuration,
            members=members,
            out=Path(alternative["manifest"]).parent,
            source=alternative,
        )
        result["checks"].extend(
            {**c, "where": {**c["where"], "position": index}} for c in measured["checks"]
        )
        result.setdefault("warnings", []).extend(measured.get("warnings", []))
        result["outputs"].setdefault("positions", {})[str(index)] = measured["outputs"]
    return result


def _check_one(stage, resolved, *, configuration, members, out, source):
    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    if stage == "S11":
        return write_checks(Path(out), manifest)
    if stage == "S11.P":
        return profile_checks(Path(out), manifest, source)
    if stage == "S11.R":
        return read_back(Path(out), manifest, float(unpack(resolved)["acceptance"]["area_excess"]))
    if stage == "S11.S":
        return sew_back(Path(out), manifest)
    if stage in ("S12.C", "S12.G", "S12"):
        from cfdbender.geometry._geom_reader import reader_check

        return reader_check(stage, out, manifest, resolved)
    if stage in ("S13.R", "S13"):
        from cfdbender.geometry._geom_roundtrip import identify, regenerate

        return (
            regenerate(resolved, out, members, manifest, source)
            if stage == "S13.R"
            else identify(resolved, out, manifest)
        )
    checks = manifest_checks(resolved, configuration, members, manifest, source["sources"])
    return {"checks": checks, "outputs": {"inputs": {"members": members}}}


def write_checks(folder, manifest):
    """Count emitted faces and solid entities in each file immediately after writing."""
    from cfdbender.geometry._iges_write import read_iges
    from cfdbender.geometry.iges_profile import SOLID_ENTITIES

    location = {"components": [p["component"] for p in manifest["pieces"]]}
    expected = {manifest["iges"]: sum(p["faces"] for p in manifest["pieces"])}
    for piece in manifest["pieces"]:
        expected[piece["file"]] = piece["faces"]
        expected.update({f["iges"]: 1 for f in piece["face_records"]})
    missing, solid = 0, 0
    for file, count in expected.items():
        entities = read_iges(folder / file)["entities"]
        missing = max(missing, abs(sum(e["type"] == 144 for e in entities) - count))
        solid += sum(e["type"] in SOLID_ENTITIES for e in entities)
    return {
        "checks": [
            _measured("write.face_count", missing, 0, unit="faces", where=location),
            _measured("profile.surfaces_only", solid, 0, unit="solid entities", where=location),
        ],
        "outputs": {"inputs": {"expected_faces": expected}},
    }


def profile_checks(folder, manifest, source):
    """Measure the reader flags and every published base against source provenance."""
    from cfdbender.geometry._geom_gate import base_displacement, check_profile, read_bases
    from cfdbender.geometry._iges_write import read_iges
    from cfdbender.geometry.iges_profile import SOLID_ENTITIES

    location = {"components": [p["component"] for p in manifest["pieces"]]}
    files, combined, shared = {}, {}, 0.0
    for piece in manifest["pieces"]:
        name = piece["component"]
        layout = json.loads((folder / piece["layout"]).read_text(encoding="utf-8"))
        identity = sidecar_identity(name, layout)
        shared = max(shared, float(identity != source["sources"][name]["source_surface_id"]))
        written_bases = {}
        for face in piece["face_records"]:
            if face["base"] not in written_bases:
                written_bases[face["base"]] = read_bases(folder / face["iges"])[face["name"]]
        combined[name] = list(written_bases.values())
        files[piece["file"]] = {name: combined[name]}
        for face in piece["face_records"]:
            files[face["iges"]] = {face["name"]: [written_bases[face["base"]]]}
    files[manifest["iges"]] = combined
    solid, flags = 0, 0
    for file, bases in files.items():
        document = read_iges(folder / file)
        by_id = {e["id"]: e for e in document["entities"]}
        solid += sum(e["type"] in SOLID_ENTITIES for e in by_id.values())
        for entity in by_id.values():
            if entity["type"] == 142:
                values = entity["parameters"]
                parameter, model = by_id.get(int(values[3])), by_id.get(int(values[4]))
                flags += int(
                    values[5] != "1"
                    or parameter is None
                    or model is None
                    or parameter["d1"][8][4:6] != "05"
                )
        shared = max(shared, base_displacement(folder / file, bases))
    report = None if solid else check_profile(folder / manifest["iges"])
    failures = flags + (len(report.failed) if report is not None else solid)
    return {
        "checks": [
            _measured("profile.reader", failures, 0, unit="defects", where=location),
            _measured(
                "profile.base_shared", shared, 1e-12, unit="coefficient difference", where=location
            ),
        ],
        "warnings": []
        if report is None
        else [str(f) for f in report.findings if f.severity == "warn"],
        "outputs": {
            "inputs": {"files": list(files)},
            "findings": [] if report is None else [vars(f) for f in report.findings],
        },
    }


def inspect_record(file: str, tolerance: float | None) -> dict[str, Any]:
    """Read one file alone in a fresh worker; return measured topology in metres."""
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("export read-back")
    try:
        topology = kernel.inspect_file(file, tolerance)
        areas = topology.areas
        errors: tuple[float, ...] = ()
        centroids, boxes, identities = [], [], []
        names = topology.names
        if tolerance is None:
            from cfdbender.geometry._iges_write import read_iges

            imported = kernel.import_iges(file)
            names = imported.names
            measurements = [measure_area(kernel, face) for face in imported.faces]
            areas = tuple(a for a, _ in measurements)
            errors = tuple(e for _, e in measurements)
            centroids = [kernel.centroid(face) for face in imported.faces]
            boxes = [kernel.bbox(face) for face in imported.faces]
            identities = [
                e["d2"][7].strip() for e in read_iges(file)["entities"] if e["type"] == 144
            ]
        return plain(
            {
                "areas": areas,
                "area_errors": errors,
                "centroids": centroids,
                "boxes": boxes,
                "identities": identities,
                "geometry": _face_geometry(file) if tolerance is None else [],
                "perimeters": topology.perimeters,
                "edges": [{"samples": points, "faces": faces} for points, faces in topology.edges],
                "names": names,
                "property_names": _face_property_names(file) if tolerance is None else [],
                "pid": os.getpid(),
            }
        )
    finally:
        kernel.close()


def _inspect(path: Path, tolerance: float | None = None) -> dict[str, Any]:
    return run_isolated(
        f"{__name__}:inspect_record", {"file": str(path), "tolerance": tolerance}, inspect_record
    )


def _measured(check, value, limit, *, unit, where=None):
    actions = {
        "area.read_back": (
            "Check the cap and trim definitions and face identities, then rebuild and export again."
        ),
        "area.source_excess": "Check the retained trims against the source, then rebuild.",
        "reimport.each_component": "Check for missing or extra component faces, then rebuild.",
        "reimport.free_edges": "Repair the component's open boundaries, then rebuild.",
        "reimport.triple_edges": "Remove overlapping component faces, then rebuild.",
    }
    location = ", ".join(f"{key}={item}" for key, item in (where or {}).items())
    note = (
        f"{check}: measured {value} {unit}, limit <= {limit} {unit}"
        f"{'; ' + location if location else ''}. "
        + actions.get(check, "Check the exported faces and boundary definitions, then rebuild.")
    )
    return check_entry(
        check,
        "passed" if value <= limit else "failed",
        value,
        limit,
        unit=unit,
        where=where,
        note=note if value > limit else "",
    )


def _proved_areas(values, errors):
    """Whether every positive finite value has an explicit numerical certificate."""
    try:
        a, e = np.asarray(values, float), np.asarray(errors, float)
        return bool(
            a.ndim == e.ndim == 1
            and len(a) > 0
            and a.shape == e.shape
            and np.isfinite(a).all()
            and np.isfinite(e).all()
            and np.all(a > 0)
            and np.all(e >= 0)
            and np.all(e <= NUMERICAL_RTOL * a)
        )
    except (ValueError, TypeError):
        return False


def _area_difference(got, got_error, wanted, wanted_error):
    """Bound relative change for every value in both measurement intervals."""
    return (abs(got - wanted) + got_error + wanted_error) / (wanted - wanted_error)


def _corresponding_faces(record, faces, order, geometry=None):
    """Validate directory identity and the unique geometry of every imported face.

    Directory order is only a candidate correspondence. Unique centroids or
    bounding boxes validate the kernel's import order in metres. A box can
    describe the underlying surface, so its fallback also requires unchanged
    serialized trim geometry recorded when the source was written.
    Ambiguous geometry cannot establish which imported face carries a label.
    """
    identities = record.get("identities", [])
    expected = [face.get("iges_identity") for face in faces]
    if (
        not order
        or identities != order
        or not all(isinstance(label, str) and label for label in expected)
        or not all(isinstance(label, str) and label for label in identities)
        or len(expected) != len(set(expected))
        or set(expected) != set(identities)
        or len(identities) != len(expected)
        or len(record["areas"]) != len(expected)
    ):
        return None, "missing, duplicated or changed face identity/order"
    indices = [expected.index(label) for label in identities]
    reasons = []
    verified = np.zeros(len(faces), dtype=bool)
    for source_key, imported_key, width in (("centroid_m", "centroids", 3), ("bbox_m", "boxes", 6)):
        try:
            source = np.array([face[source_key] for face in faces], float).reshape(len(faces), -1)
            imported = np.array(record[imported_key], float).reshape(len(faces), -1)
            if source.shape != (len(faces), width) or imported.shape != source.shape:
                raise ValueError("invalid certificate dimensions")
            if not np.isfinite(source).all() or not np.isfinite(imported).all():
                raise ValueError("nonfinite certificate")
            matches = np.max(np.abs(imported[:, None, :] - source[None, :, :]), axis=2)
            close = matches <= _CORRESPONDENCE_M
            # Each face needs its own unique geometric witness. A cap can
            # have a stable box while a curved skin has a stable centroid.
            rows, columns = close.sum(axis=1), close.sum(axis=0)
            witness = np.array(
                [close[i, j] and rows[i] == columns[j] == 1 for i, j in enumerate(indices)]
            )
            if source_key == "bbox_m":
                imported_geometry = record.get("geometry", [])
                if (
                    not geometry
                    or len(geometry) != len(faces)
                    or len(imported_geometry) != len(faces)
                ):
                    witness[:] = False
                else:
                    witness &= [
                        bool(a) and a == b for a, b in zip(geometry, imported_geometry, strict=True)
                    ]
                reasons.append("bounding-box fallback requires unchanged recorded trim geometry")
            verified |= witness
            if verified.all():
                return [faces[i] for i in indices], ""
            distance = max(matches[i, j] for i, j in enumerate(indices))
            reasons.append(
                f"{source_key}: measured distance {distance} m, limit {_CORRESPONDENCE_M} m; "
                f"matches per face {close.sum(axis=1).tolist()}, limit exactly one"
            )
        except (KeyError, ValueError, TypeError):
            reasons.append(f"{source_key}: missing or invalid geometric certificate")
    return None, "unverified or ambiguous geometry; " + "; ".join(reasons)


def read_back(folder: Path, manifest: dict, area_excess: float) -> dict[str, Any]:
    """Measure component count, free edges and all corresponding trimmed face areas.

    Component, isolated and combined area reads have no sewing.
    Free-edge comparison uses a separate import sewn
    at 1e-5 m. A failed import counts as a lost face and a full area loss.
    """
    count_gap, free_gap, triple, worst, excess = 0, 0, 0, 0.0, 0.0
    where: dict[str, Any] = {}
    locations: dict[str, set[str]] = {key: set() for key in ("count", "free", "triple", "excess")}
    pids, failures, files = [], [], []

    def read(file, tolerance=None):
        files.append(file)
        try:
            record = _inspect(folder / file, tolerance)
            if tolerance is None and not _proved_areas(record["areas"], record.get("area_errors")):
                raise InputArtifactError(
                    f"{file}: measured area or error is unverified; numerical relative limit "
                    f"{NUMERICAL_RTOL}. Rebuild the component and export again.",
                    kind="geometry",
                )
            pids.append(record["pid"])
            return record
        except (InputArtifactError, OSError, ValueError) as error:
            failures.append({"file": file, "error": str(error)})
            return {"areas": [], "area_errors": [], "edges": []}

    def compare(file, record, faces, component, *, isolated=False):
        nonlocal worst, where
        paired, reason = _corresponding_faces(
            record,
            faces,
            manifest.get("face_orders", {}).get(file),
            manifest.get("face_geometry", {}).get(file),
        )
        if paired is not None:
            names = [f["name"] if isolated else f.get("component", component) for f in paired]
            if record.get("names") != names or record.get("property_names") != names:
                paired = None
                reason = (
                    f"missing or changed face name; expected {names}, "
                    f"directory names {record.get('names')}, "
                    f"NAME pointers {record.get('property_names')}"
                )
        if paired is None or not _proved_areas(
            [f["area_m2"] for f in faces], [f.get("area_error_m2") for f in faces]
        ):
            location = {
                "component": component,
                "face": ", ".join(f["name"] for f in faces),
                "file": file,
                "verification": reason or "source area unverified",
            }
            if worst <= 1.0:
                worst, where = 1.0, location
            failure = _measured("area.read_back", 1.0, 1e-3, unit="relative", where=location)
            failure["note"] = (
                "area.read_back: not measured, the imported faces are not matched to the source "
                f"({location['verification']}); component={component}, face={location['face']}, "
                f"file={file}. Check the face identities and NAME properties, then rebuild and "
                "export again."
            )
            failures.append({"file": file, "error": failure["note"]})
            return
        for got, error, face in zip(record["areas"], record["area_errors"], paired, strict=True):
            relative = _area_difference(got, error, face["area_m2"], face["area_error_m2"])
            if relative >= worst:
                worst, where = (
                    relative,
                    {
                        "component": face.get("component", component),
                        "face": face["name"],
                        "file": file,
                        "expected_m2": face["area_m2"],
                        "read_m2": got,
                        "verification": "bounded, correspondence verified",
                        "uncertainty_m2": error + face["area_error_m2"],
                    },
                )

    for piece in manifest["pieces"]:
        name = piece["component"]
        expected = [float(f["area_m2"]) for f in piece["face_records"]]
        imported = read(piece["file"])
        areas = imported["areas"]
        area_errors = imported["area_errors"]
        piece["reimport_area_m2"] = sum(areas)
        count_gap = max(count_gap, abs(len(areas) - len(expected)))
        if len(areas) != len(expected):
            locations["count"].add(name)
        compare(piece["file"], imported, piece["face_records"], name)
        source_area = float(piece["source_area_m2"])
        component_excess = (sum(areas) + sum(area_errors) - source_area) / max(source_area, 1e-30)
        excess = max(excess, component_excess)
        if component_excess > area_excess:
            locations["excess"].add(name)
        sewn = read(piece["file"], 1e-5)
        free = sum(len(e["faces"]) == 1 for e in sewn["edges"])
        free_gap = max(free_gap, abs(free - piece["free_edges"]))
        component_triple = sum(len(e["faces"]) > 2 for e in sewn["edges"])
        triple += component_triple
        if free != piece["free_edges"]:
            locations["free"].add(name)
        if component_triple:
            locations["triple"].add(name)
        for face in piece["face_records"]:
            alone = read(face["iges"])
            count_gap = max(count_gap, abs(len(alone["areas"]) - 1))
            if len(alone["areas"]) != 1:
                locations["count"].add(name)
            compare(face["iges"], alone, [face], name, isolated=True)
    combined_file = manifest.get("iges") or "<missing combined IGES>"
    retained = [
        {**face, "component": piece["component"]}
        for piece in manifest["pieces"]
        for face in piece["face_records"]
    ]
    combined = read(combined_file)
    count_gap = max(count_gap, abs(len(combined["areas"]) - len(retained)))
    if len(combined["areas"]) != len(retained):
        locations["count"].update(p["component"] for p in manifest["pieces"])
    compare(
        combined_file,
        combined,
        retained,
        ", ".join(piece["component"] for piece in manifest["pieces"]),
    )
    manifest["combined_reimport_area_m2"] = sum(combined["areas"])
    write_record(folder / "manifest.json", manifest)
    return {
        "checks": [
            _measured("area.read_back", worst, 1e-3, unit="relative", where=where),
            _measured(
                "area.source_excess",
                excess,
                area_excess,
                unit="relative",
                where={"components": sorted(locations["excess"])},
            ),
            _measured(
                "reimport.each_component",
                count_gap,
                0,
                unit="faces",
                where={"components": sorted(locations["count"])},
            ),
            _measured(
                "reimport.free_edges",
                free_gap,
                0,
                unit="edges",
                where={"components": sorted(locations["free"])},
            ),
            _measured(
                "reimport.triple_edges",
                triple,
                0,
                unit="edges",
                where={"components": sorted(locations["triple"])},
            ),
        ],
        "outputs": {"inputs": {"files": files}, "worker_pids": pids, "failures": failures},
    }


def sew_back(folder: Path, manifest: dict) -> dict[str, Any]:
    """Read the combined file once per tolerance and compare with the input openings.

    Faces from separately imported files are never treated as sewn together.
    Thin faces already present in the input are listed, not treated as new defects.
    """
    reference = [np.asarray(p, float) for p in manifest["input_free_edges"]]
    location = {"components": [p["component"] for p in manifest.get("pieces", [])]}
    axis = manifest.get("plane_axis")

    def brought(points):
        if axis is not None and float(np.abs(points[:, axis]).max()) <= jn.SYMMETRY_M:
            return True
        return any(
            float(jn.distance_to_polyline(points, p).max()) <= jn.OPENING_M for p in reference
        )

    imports = []
    for tolerance in (jn.SEW_M, jn.SEW_WIDE_M):
        topology = _inspect(folder / manifest["iges"], tolerance)
        free = [np.asarray(e["samples"]) for e in topology["edges"] if len(e["faces"]) == 1]
        new = [p.tolist() for p in free if not brought(p)]
        widths = [
            jn.mean_width(a, p)
            for a, p in zip(topology["areas"], topology["perimeters"], strict=True)
        ]
        thin = [
            {"face": i, "area_m2": topology["areas"][i], "width_m": widths[i]}
            for i in jn.slivers(widths)
        ]
        imports.append(
            {
                "tolerance_m": tolerance,
                "pid": topology["pid"],
                "free_edges": len(free),
                "introduced": new,
                "triple_edges": sum(len(e["faces"]) > 2 for e in topology["edges"]),
                "slivers": thin,
                "slivers_introduced": max(0, len(thin) - manifest["input_slivers"]),
            }
        )
    return {
        "checks": [
            _measured(
                "sew.free_introduced",
                max(len(r["introduced"]) for r in imports),
                0,
                unit="edges",
                where=location,
            ),
            _measured(
                "sew.triple_edges",
                max(r["triple_edges"] for r in imports),
                0,
                unit="edges",
                where=location,
            ),
            _measured(
                "sew.slivers",
                max(r["slivers_introduced"] for r in imports),
                0,
                unit="faces",
                where=location,
            ),
        ],
        "outputs": {
            "inputs": {
                "file": manifest["iges"],
                "free_edges": len(reference),
                "slivers": manifest["input_slivers"],
            },
            "imports": imports,
        },
    }


def generate_record(
    config: Any, path: str, configuration: str | None, variant="clean", selection=None
) -> dict[str, Any]:
    """Build unassembled source pieces on a kernel for a configuration with no operations."""
    from cfdbender.geometry._geom_propeller import PropellerAssembly
    from cfdbender.geometry.stages import generate_pieces, unpack
    from cfdbender.mesh._gmsh import open_kernel

    plain_config = unpack(config)
    every = generate_pieces(config, variant=variant, selection=selection)
    members = (
        list(every) if configuration is None else plain_config["configurations"][configuration]
    )
    kernel = open_kernel("writing source pieces")
    try:
        factory = (
            PropellerAssembly
            if any(plain_config["components"][n]["type"] == "propeller" for n in members)
            else Assembly
        )
        assembly = factory(kernel, {n: every[n] for n in members}, [], out=Path(path).parent)
        record = assembly.step("S10.D1")
        if any(c["status"] == "failed" for c in record["checks"]):
            raise InputArtifactError(str(record["checks"]), kind="geometry")
        written = write_assembly(
            assembly,
            Path(path),
            resolved=config,
            configuration=configuration,
            variant=variant,
            selection=selection,
        )
        written["sections"] = {
            n: p.layout["sections"] for n, p in every.items() if "sections" in p.layout
        }
        return written
    finally:
        kernel.close()
