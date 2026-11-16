"""export checks and distance regression.

Reproduction: measure a point on a unit plane whose v domain is [0, 10000].
Its distance must stay below 1e-9 of the model size near a parameter bound.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# A cold worker imports this module by its dotted name, without the test folder on its path.
sys.path.insert(0, str(Path(__file__).parent))
from kernel_support import kernel_or_skip  # noqa: E402

from cfdbender.geometry._geom_gate import check_profile
from cfdbender.geometry._iges_write import read_iges, serialize_iges
from cfdbender.geometry.iges_profile import ReaderProfile


def _applied_config():
    return {
        "components": {
            "panel": {
                "type": "lifting_surface",
                "planform": {"sref_m2": 4.0, "aspect_ratio": 4.0, "taper": 1.0},
                "cap": {"type": "flat"},
                "root": "flat",
                "control": [
                    {
                        "name": "flap",
                        "kind": "plain",
                        "eta": [0.2, 0.7],
                        "chord_ratio": 0.25,
                        "gap_span": 0.01,
                        "gap_chord": 0.02,
                        "deflection_deg": [15],
                    }
                ],
            }
        },
        "variants": {"deployed": ["panel.flap"]},
    }


def test_geom_kernel_export_triangular_face_keeps_its_surface_with_four_boundary_segments(tmp_path):
    from cfdbender.mesh._gmsh import open_kernel

    kernel_or_skip()
    kernel = open_kernel("a retained triangular face")
    path = tmp_path / "triangle.igs"
    try:
        face = kernel.add_planar_face([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        kernel.export_iges({"triangle": [face]}, path)
        profile = check_profile(path)
        assert profile.passed, profile.failed
        topology = kernel.inspect_file(path, None)
        assert len(topology.areas) == 1
        assert topology.areas[0] == pytest.approx(0.5, rel=1e-9)
        document = read_iges(path)
        by_id = {e["id"]: e for e in document["entities"]}
        boundary = next(e for e in document["entities"] if e["type"] == 142)
        for pointer in boundary["parameters"][3:5]:
            assert by_id[int(pointer)]["parameters"][1] == "4"
        assert boundary["parameters"][5] == "1"
    finally:
        kernel.close()


def test_geom_kernel_export_converter_retains_compatible_named_nurbs_coefficients(
    tmp_path, monkeypatch
):
    from cfdbender._errors import InputArtifactError
    from cfdbender.geometry._geom_gate import base_displacement, read_bases
    from cfdbender.geometry.iges_convert import convert_iges
    from cfdbender.mesh._gmsh import gmsh_module, open_kernel

    kernel_or_skip()
    source, target = tmp_path / "source.igs", tmp_path / "converted.igs"
    kernel = open_kernel("a compatible named B-spline at a translated datum")
    try:
        face = kernel.add_bspline_surface(
            [[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0.2)]],
            knots_u=[0, 1],
            knots_v=[0, 1],
            mult_u=[2, 2],
            mult_v=[2, 2],
            deg_u=1,
            deg_v=1,
        )
        kernel.export_iges({"part": [face]}, source)
    finally:
        kernel.close()
    document = read_iges(source)
    offset = 10000.123456789013
    for entity in document["entities"]:
        values, kind = entity["parameters"], entity["type"]
        if kind == 128:
            ku, kv, du, dv = map(int, values[1:5])
            count = (ku + 1) * (kv + 1)
            start = 10 + ku + du + 2 + kv + dv + 2 + count
            indices = range(start, start + 3 * count, 3)
        elif kind == 126 and entity["d1"][8][4:6] != "05":
            k, degree = map(int, values[1:3])
            start = 7 + k + degree + 2 + k + 1
            indices = range(start, start + 3 * (k + 1), 3)
        elif kind == 110 and entity["d1"][8][4:6] != "05":
            indices = (1, 4)
        else:
            continue
        for index in indices:
            values[index] = format(float(values[index]) + offset, ".17g")
    document["global"][19] = str(offset + 1)
    source.write_text(serialize_iges(document), encoding="ascii")
    expected = {name: [base] for name, base in read_bases(source).items()}
    record = convert_iges([source], target)
    assert record["faces"] == 1 and record["components"] == {"part": 1}
    assert base_displacement(target, expected) <= 1e-12
    assert record["readback"]["retained_compatible_nurbs"] is True
    gmsh = gmsh_module("a retained-file count mutation")
    importing, calls = gmsh.model.occ.importShapes, []

    def drop_second_read(path, *args, **kwargs):
        calls.append(path)
        faces = importing(path, *args, **kwargs)
        return faces if len(calls) == 1 else []

    monkeypatch.setattr(gmsh.model.occ, "importShapes", drop_second_read)
    monkeypatch.setenv("CFDBENDER_GMSH_ISOLATE", "0")
    refused = tmp_path / "lost-face.igs"
    with pytest.raises(InputArtifactError, match="faces"):
        convert_iges([source], refused)
    assert not refused.exists()


def test_geom_kernel_export_empty_variant_reaches_the_export_api(tmp_path):
    from cfdbender.geometry import pipeline as _geom_run
    from cfdbender.geometry.stages import stage_row

    kernel_or_skip()
    raw = {"components": {"shell": {"type": "body"}}, "variants": {"deployed": []}}
    selected = [_geom_run.Stage(stage_row(s), _geom_run.STAGE_FUNCTIONS[s]) for s in ("S0", "S11")]
    report = _geom_run.run_geometry(
        raw, variant="deployed", out=tmp_path, until="S11", stages=selected
    )
    assert report["status"] == "accepted", report
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "deployed"
    assert [p["component"] for p in manifest["pieces"]] == ["shell"]


def _keep_boundary_for_ci(report, folder, stage):
    """Copy a failed run's boundary folder to ``PYFS_GEOM_DIAG`` (CI uploads it on failure).

    The clean junction chain failed on some Linux runners only (2 of 3 runs of one shard) and
    passed on every machine it could be measured on, so the failing boundary is kept for reading.
    """
    import os
    import shutil
    from pathlib import Path

    target = os.environ.get("PYFS_GEOM_DIAG")
    if target and report["status"] != "accepted" and folder.exists():
        destination = Path(target) / f"{stage}-boundary"
        shutil.copytree(folder, destination, dirs_exist_ok=True)
        (destination / "report.json").write_text(json.dumps(report, default=str), "utf-8")


@pytest.mark.parametrize("gap", [0.02, 0.002], ids=["P0570-clear", "P0570-interpenetrating"])
def test_geom_kernel_export_applied_variant_runs_shipped_reader_and_mesh_checks(tmp_path, gap):
    from cfdbender.geometry import pipeline as _geom_run
    from cfdbender.geometry.stages import stage_row

    kernel_or_skip()
    selected = [
        _geom_run.Stage(stage_row(s), _geom_run.STAGE_FUNCTIONS[s])
        for s in ("S0", "S11", "S11.P", "S12.C", "S12.G", "S12")
    ]
    raw = _applied_config()
    raw["components"]["panel"]["control"][0]["gap_chord"] = gap
    report = _geom_run.run_geometry(
        raw, variant="deployed", out=tmp_path, until="S12", stages=selected
    )
    if gap == 0.002:
        assert (report["status"], report["stage"], report["check"]) == (
            "failed",
            "S12.G",
            "gate.grid",
        )
        assert report["measured"] > 0
        return
    assert report["status"] == "accepted", report
    for stage in ("S12.C", "S12.G", "S12"):
        record = json.loads((tmp_path / f"{stage}.json").read_text(encoding="utf-8"))
        assert record["checks"] and all(c["status"] == "passed" for c in record["checks"]), record


@pytest.mark.parametrize("u", [0.001, 0.999], ids=["P0570-low-bound", "P0570-high-bound"])
@pytest.mark.parametrize("quadratic", [False, True], ids=["P0570-uniform", "P0570-independent"])
def test_geom_kernel_export_anisotropic_surface_distance(tmp_path, u, quadratic):
    values = [128, 1, 1, 1, 1, 0, 0, 1, 0, 0]
    values += [0, 0, 1, 1, 0, 0, 10000, 10000]
    values += [1, 1, 1, 1]
    values += [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]
    values += [0, 1, 0, 10000]
    corners = [(u, 0.12345), (u + 0.00001, 0.12345), (u + 0.00001, 0.12346), (u, 0.12346)]
    entities = [
        _entity(1, 128, values),
        _entity(3, 144, [144, 1, 1, 0, 5, 0, 1, 11]),
        _entity(5, 142, [142, 1, 1, 7, 9, 1]),
        _entity(7, 102, [102, 4, 13, 17, 21, 25]),
        _entity(9, 102, [102, 4, 15, 19, 23, 27]),
        _entity(11, 406, [406, 1, "5Hplate"]),
    ]
    entities[3]["d1"][8] = "00000500"
    entities[5]["d2"][4] = f"{15:8d}"
    for i, (x, y) in enumerate(corners):
        xx, yy = corners[(i + 1) % 4]
        entities += [
            _entity(13 + 4 * i, 110, [110, x, y * 10000, 0, xx, yy * 10000, 0]),
            _entity(15 + 4 * i, 110, [110, x, y, 0, xx, yy, 0]),
        ]
        entities[-2]["d1"][8] = "00000500"
        if quadratic:
            # Same straight physical edge with a quadratic time parameter.
            # Its nine samples differ from those of the linear pcurve, so the
            # distance search must refine an independently sampled point.
            model = entities[-1]
            a, b = [x, y, 0], [xx, yy, 0]
            values = [126, 2, 2, 1, 0, 1, 0]
            values += [0, 0, 0, 1, 1, 1] + [1, 1, 1]
            values += [*a, *a, *b, 0, 1, 0, 0, 1]
            model.update(type=126, parameters=[str(v) for v in values])
            model["d1"][0] = model["d2"][0] = f"{126:8d}"
    globals_ = ["0"] * 25
    globals_[18] = str(1e-9 * np.sqrt(2))
    document = {"start": ["distance fixture"], "global": globals_, "entities": entities}
    path = tmp_path / "distance.igs"
    path.write_text(serialize_iges(document), encoding="ascii")
    profile = ReaderProfile(degrees_u=(1,), degrees_v=(1,))
    assert check_profile(path, profile).passed
    # A parallel displacement must still be detected at the same resolution.
    for entity in entities:
        if entity["id"] in (15, 19, 23, 27):
            for z in (18, 21, 24) if quadratic else (3, 6):
                entity["parameters"][z] = "0.00001"
    path.write_text(serialize_iges(document), encoding="ascii")
    assert {f.check for f in check_profile(path, profile).failed} == {"profile.loop_surface"}


def _entity(ident, kind, values):
    return {
        "id": ident,
        "type": kind,
        "parameters": [str(v) for v in values],
        "d1": [f"{kind:8d}"] + [f"{0:8d}"] * 7 + ["00000000"],
        "d2": [f"{kind:8d}"] + [f"{0:8d}"] * 8,
    }


@pytest.mark.parametrize("end", ["tip", "root"])
def test_geom_kernel_export_flat_cap_passes_reader_profile_with_exact_closed_rims(tmp_path, end):
    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.stages import cap_edge_gap, lifting_model, unpack

    kernel_or_skip()
    component = {"type": "lifting_surface", "cap": {"type": "flat" if end == "tip" else "none"}}
    if end == "root":
        component["root"] = "flat"
    raw = {"components": {"wing": component}}
    model = lifting_model(unpack(resolve_config(raw))["components"]["wing"])
    assert next(c for c in model.records if c["id"] == "closure.rim_shared")["value"] <= 1e-12
    assert cap_edge_gap(model) <= 1e-12
    nose = model.caps[f"{end}_cap"]
    for cap in model.caps.values():
        grid = cap.evaluate(31)
        # The leading-edge end of the one-face outline cap is a point; its other edges are curves.
        for edge in (grid[-1], grid[:, 0], grid[:, -1]):
            assert np.linalg.norm(np.diff(edge, axis=0), axis=1).sum() > 1e-6
        normals = np.cross(np.diff(grid, axis=0)[:, :-1], np.diff(grid, axis=1)[:-1])
        assert np.all(normals[1:-1, 1:-1, 1] > 0) or np.all(normals[1:-1, 1:-1, 1] < 0)
    changed = np.array(nose.controls, copy=True)
    changed[0, :, 2] += 1e-4
    broken = replace(model, caps={**model.caps, f"{end}_cap": replace(nose, controls=changed)})
    assert cap_edge_gap(broken) >= 9e-5
    report = run_geometry(raw, out=tmp_path, until="S11.P")
    assert report["status"] in ("accepted", "incomplete"), report
    record = json.loads((tmp_path / "S11.P.json").read_text(encoding="utf-8"))
    assert record["status"] == "passed", record
    assert check_profile(tmp_path / "wing.igs").passed


@pytest.mark.parametrize("preference", [1, 3], ids=["P0570-default", "P0570-either"])
def test_geom_kernel_export_written_boundaries_carry_both_curves(tmp_path, preference):
    kernel_or_skip()
    from cfdbender.mesh._gmsh import open_kernel

    kernel = open_kernel("export regression")
    try:
        face = kernel.add_planar_face(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        )
        path = tmp_path / "piece.igs"
        if preference == 3:
            with pytest.warns(UserWarning, match="IGES.PREF: 3 .*PREF 1 is written instead"):
                kernel.export_iges({"plate": [face]}, path, preference=preference)
        else:
            kernel.export_iges({"plate": [face]}, path, preference=preference)
        document = read_iges(path)
        by_id = {e["id"]: e for e in document["entities"]}
        boundaries = [e for e in document["entities"] if e["type"] == 142]
        assert boundaries
        for boundary in boundaries:
            values = boundary["parameters"]
            assert values[5] == "1"
            assert int(values[3]) in by_id
            assert int(values[4]) in by_id
            assert by_id[int(values[3])]["d1"][8][4:6] == "05"
        read = kernel.inspect_file(path, None)
        assert len(read.areas) == 1
        assert read.areas[0] == pytest.approx(1.0, rel=1e-6)
    finally:
        kernel.close()


@pytest.mark.parametrize("gap", [0.0, 0.00013], ids=["P0570-control", "P0570-cap-edge-gap"])
def test_geom_kernel_export_cap_boundary_check_reaches_run_log(tmp_path, gap):
    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._boundary import unpack
    from cfdbender.geometry._caps import lifting_model
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.pipeline import Stage, StageOutcome, default_stages

    raw = {"components": {"main": {"type": "lifting_surface", "cap": {"type": "rounded"}}}}
    component = unpack(resolve_config(raw))["components"]["main"]
    model = lifting_model(component)
    corner = model.caps["tip_cap_corner"]
    controls = np.array(corner.controls, copy=True)
    # Move the whole side except its skin-rim endpoint: the rim-only check misses this gap.
    controls[0, 1:, 2] += gap
    caps = {**model.caps, "tip_cap_corner": replace(corner, controls=controls)}
    model = replace(model, caps=caps)

    def supply_model(context):
        context.state["lifting_models"] = {"main": model}
        return StageOutcome()

    plan = [Stage(s.row, supply_model) if s.row.stage == "S7" else s for s in default_stages()]
    report = run_geometry(raw, out=tmp_path, until="S8", stages=plan)
    if gap:
        assert report["status"] == "failed"
        assert (report["stage"], report["check"]) == ("S8", "closure.cap_edge_gap")
    else:
        checks = json.loads((tmp_path / "S8.json").read_text(encoding="utf-8"))["checks"]
        found = [c for c in checks if c["id"] == "closure.cap_edge_gap"]
        assert found and found[0]["status"] == "passed"
    assert "closure.cap_edge_gap" in (tmp_path / "geom-run.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["body", "lifting_surface"], ids=["P0570-body", "P0570-skin"])
def test_geom_kernel_export_export_keeps_component_version_and_trimmed_faces(tmp_path, kind):
    kernel_or_skip()
    from cfdbender.geometry._identify import run_geometry

    raw = {"components": {"shell": {"type": kind, "version": "2.1.0"}}}
    report = run_geometry(raw, out=tmp_path, until="S11")
    assert report["status"] != "failed", report
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file(), "S11 must retain each source piece and its written faces"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    piece = next(p for p in manifest["pieces"] if p["component"] == "shell")
    assert piece["version"] == "2.1.0"
    assert len(piece["source_surface_id"]) == 64
    assert piece["faces"] and all(f["area_m2"] > 0 for f in piece["face_records"])
    sidecar = json.loads((tmp_path / "shell.layout.json").read_text(encoding="utf-8"))
    assert len(sidecar["faces"]) == piece["faces"]
    assert all(f["base"] in sidecar["bases"] for f in sidecar["faces"])
    if kind == "lifting_surface":
        skin_faces = [f for f in sidecar["faces"] if f["base"] == "skin"]
        assert len(skin_faces) >= 2
        assert {f["role"] for f in skin_faces} == {"upper", "lower"}
        assert {t["role"] for f in skin_faces for t in f["source_faces"]} >= {"upper", "lower"}
    document = read_iges(tmp_path / piece["file"])
    assert len([e for e in document["entities"] if e["type"] == 144]) == piece["faces"]


def _cold_roundtrip():
    """Run the regeneration check with empty process-local fitting caches."""
    import sys
    from pathlib import Path

    from cfdbender.geometry._geom_export import check_export

    request, reply = map(Path, sys.argv[1:3])
    data = json.loads(request.read_text(encoding="utf-8"))
    calls = []

    def trace(frame, event, _arg):
        if event == "call" and frame.f_code.co_name in ("_joint_fit", "open_kernel"):
            calls.append(frame.f_code.co_name)

    sys.setprofile(trace)
    try:
        record = check_export(
            "S13.R",
            data["resolved"],
            configuration=None,
            members=data.get("members", ["part"]),
            out=request.parent,
            source=data["written"],
        )
    finally:
        sys.setprofile(None)
    reply.write_text(json.dumps({"record": record, "calls": calls}), encoding="utf-8")


def _run_cold_roundtrip(folder, resolved, written, members):
    import os
    import subprocess
    import sys

    request, reply = folder / "cold-request.json", folder / "cold-reply.json"
    request.write_text(
        json.dumps({"resolved": resolved, "written": written, "members": members}), encoding="utf-8"
    )
    code = (
        "from tests.tier1_offline.test_geom_kernel_export import _cold_roundtrip; _cold_roundtrip()"
    )
    with (folder / "cold-worker.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-c", code, str(request), str(reply)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=dict(os.environ),
        )
    return json.loads(reply.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "built", ["2.1.0", "1.1.0"], ids=["P0570-version-control", "P0570-wrong-version"]
)
def test_geom_kernel_export_manifest_checks_the_built_version_against_inherited_registry(
    tmp_path, built
):
    kernel_or_skip()
    from cfdbender.geometry._identify import run_geometry

    raw = {
        "components": {"shell": {"type": "body", "version": built}},
        "configurations": {
            "base": {"members": [{"component": "shell", "version": "2.1.0"}]},
            "derived": {"extends": "base", "members": ["shell"]},
        },
    }
    # An explicit derived member inherits the same declared version, independent of generation.
    raw["configurations"]["derived"]["members"] = [{"component": "shell", "version": "2.1.0"}]
    report = run_geometry(raw, configuration="derived", out=tmp_path, until="S11.V")
    checks = [
        check
        for path in tmp_path.glob("S*.json")
        for check in json.loads(path.read_text(encoding="utf-8")).get("checks", [])
        if check["id"] == "manifest.piece_version"
    ]
    assert checks and all(
        c["status"] == ("passed" if built == "2.1.0" else "failed") for c in checks
    )
    if built != "2.1.0":
        assert report["check"] == "manifest.piece_version"
        assert "1.1.0" in str(checks) and "2.1.0" in str(checks)


@pytest.mark.parametrize("variant", ["clean", "deployed"], ids=["P0570-allowed", "P0570-excluded"])
def test_geom_kernel_export_selected_configuration_limits_its_variants(tmp_path, variant):
    from cfdbender.geometry._identify import run_geometry

    raw = {
        "components": {"shell": {"type": "body"}},
        "variants": {"deployed": []},
        "configurations": {"one": {"members": ["shell"], "variants": ["clean"]}},
    }
    report = run_geometry(raw, configuration="one", variant=variant, out=tmp_path, until="S0")
    assert report["status"] == ("accepted" if variant == "clean" else "refused")


def test_geom_kernel_export_empty_variant_exports_its_declared_geometry(tmp_path):
    kernel_or_skip()
    from cfdbender.geometry._identify import run_geometry

    raw = {"components": {"shell": {"type": "body"}}, "variants": {"deployed": []}}
    report = run_geometry(raw, variant="deployed", out=tmp_path, until="S11")
    assert report["status"] in ("accepted", "incomplete")
    record = json.loads((tmp_path / "S11.json").read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert list(tmp_path.glob("*.igs"))
    assert (
        json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["variant"]
        == "deployed"
    )


def _written_halves(folder, height=1.0):
    """Two exact half-width trims on one rectangular base."""
    kernel_or_skip()
    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._geom_assembly import Assembly
    from cfdbender.geometry._geom_export import write_assembly
    from cfdbender.geometry.stages import GeneratedPiece, Patch
    from cfdbender.mesh._gmsh import open_kernel

    patch = Patch(
        "skin",
        np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]),
        (0.0, 1.0),
        (0.0, 1.0),
        (2, 2),
        (2, 2),
        1,
        1,
    )
    piece = GeneratedPiece.from_patches("plate", "1.0.0", [patch])
    if height != 1.0:
        controls = patch.controls.copy()
        controls[..., 1] *= height
        piece = GeneratedPiece.from_patches("plate", "1.0.0", [replace(patch, controls=controls)])
    trims = []
    for name, low, high in (("left", 0.0, 0.5), ("right", 0.5, 1.0)):
        points = ((low, 0.0), (high, 0.0), (high, 1.0), (low, 1.0))
        trims.append(
            replace(
                piece.trims[0],
                face=name,
                role=name,
                loop=tuple((points[i], points[(i + 1) % 4]) for i in range(4)),
            )
        )
    piece = replace(piece, trims=tuple(trims))
    kernel = open_kernel("area acceptance fixture")
    try:
        assembly = Assembly(kernel, {"plate": piece}, [], out=folder)
        loaded = assembly.step("S10.D1")
        assert all(c["status"] == "passed" for c in loaded["checks"])
        written = write_assembly(assembly, folder / "combined.igs")
    finally:
        kernel.close()
    resolved = resolve_config({"components": {"plate": {"type": "body"}}})
    return resolved, written


@pytest.mark.parametrize("lost", [False, True], ids=["P0570-area-control", "P0570-lost-trim"])
def test_geom_kernel_export_fresh_readback_catches_a_trimmed_face_written_untrimmed(tmp_path, lost):
    import os

    from cfdbender.geometry._geom_export import check_export

    resolved, written = _written_halves(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    face = manifest["pieces"][0]["face_records"][0]
    assert face["area_m2"] == pytest.approx(0.5, abs=1e-12)
    if lost:
        path = tmp_path / face["iges"]
        document = read_iges(path)
        trimmed = next(e for e in document["entities"] if e["type"] == 144)
        trimmed["parameters"][2] = "0"
        trimmed["parameters"][4] = "0"
        path.write_text(serialize_iges(document), encoding="ascii")
    record = check_export(
        "S11.R", resolved, configuration=None, members=["plate"], out=tmp_path, source=written
    )
    areas = [c for c in record["checks"] if c["id"] == "area.read_back"]
    assert areas, "export must measure the area of every face read alone"
    assert areas[0]["status"] == ("failed" if lost else "passed")
    assert areas[0]["limit"] == 1e-3
    if lost:
        assert areas[0]["measured"] == pytest.approx(1.0, rel=1e-6)
    else:
        assert all(c["status"] == "passed" for c in record["checks"])
        assert all(pid != os.getpid() for pid in record["outputs"]["worker_pids"])


def test_geom_kernel_export_reimport_has_its_own_stage_record_and_run_log(tmp_path):
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.pipeline import Stage, StageOutcome, default_stages
    from cfdbender.geometry.stages import check_entry

    def supply(context):
        _, written = _written_halves(context.out)
        context.state["export"] = written
        return StageOutcome(
            checks=[check_entry("profile.surfaces_only", "passed", 0, 0)], outputs=written
        )

    plan = [Stage(s.row, supply) if s.row.stage == "S11" else s for s in default_stages()]
    report = run_geometry(
        {"components": {"plate": {"type": "body"}}}, out=tmp_path, until="S11.R", stages=plan
    )
    assert report["status"] != "failed", report
    record = json.loads((tmp_path / "S11.R.json").read_text(encoding="utf-8"))
    assert all(c["status"] == "passed" for c in record["checks"])
    log = (tmp_path / "geom-run.log").read_text(encoding="utf-8")
    assert "S11.R" in log and "area.read_back" in log and "reimport.free_edges" in log


@pytest.mark.parametrize("missing", [False, True], ids=["P0570-sew-control", "P0570-new-opening"])
@pytest.mark.parametrize("height", [1.0, 0.001], ids=["P0570-wide", "P0570-input-slivers"])
def test_geom_kernel_export_sewing_checks_introduced_edges_against_the_input_reference(
    tmp_path, missing, height
):
    from cfdbender.geometry._geom_export import check_export

    resolved, written = _written_halves(tmp_path, height)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_slivers"] == (2 if height < 0.005 else 0)
    if missing:
        face = manifest["pieces"][0]["face_records"][0]
        (tmp_path / manifest["iges"]).write_bytes((tmp_path / face["iges"]).read_bytes())
    record = check_export(
        "S11.S", resolved, configuration=None, members=["plate"], out=tmp_path, source=written
    )
    checks = {c["id"]: c for c in record["checks"]}
    assert "sew.free_introduced" in checks, "sewing must distinguish original and new openings"
    assert checks["sew.free_introduced"]["status"] == ("failed" if missing else "passed")
    assert checks["sew.triple_edges"]["measured"] == 0
    assert {row["tolerance_m"] for row in record["outputs"]["imports"]} == {1e-5, 5e-5}
    if missing:
        assert checks["sew.free_introduced"]["measured"] == 1
    else:
        assert all(c["status"] == "passed" for c in checks.values())


@pytest.mark.parametrize(
    "trimmed,fault",
    [(False, ""), (True, ""), (True, "delete"), (False, "displace")],
    ids=["P0570-coverage-control", "P0570-original-trim", "P0570-deleted", "P0570-off-source"],
)
def test_geom_kernel_export_coverage_measures_source_trim_and_world_distance(
    tmp_path, trimmed, fault
):
    kernel_or_skip()
    from cfdbender.geometry._geom_assembly import Assembly
    from cfdbender.geometry.stages import GeneratedPiece, Patch
    from cfdbender.mesh._gmsh import open_kernel

    patch = Patch(
        "plate",
        np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]),
        (0.0, 1.0),
        (0.0, 1.0),
        (2, 2),
        (2, 2),
        1,
        1,
    )
    piece = GeneratedPiece.from_patches("plate", "1.0.0", [patch])
    if trimmed:
        corners = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
        trim = replace(
            piece.trims[0], loop=tuple((corners[i], corners[(i + 1) % 4]) for i in range(4))
        )
        piece = replace(piece, trims=(trim,))
    kernel = open_kernel("source coverage fixture")
    try:
        assembly = Assembly(kernel, {"plate": piece}, [], out=tmp_path)
        assembly.step("S10.D1")
        if fault == "delete":
            kernel.remove([assembly.faces["plate"].pop()])
        elif fault == "displace":
            kernel.transform(
                assembly.faces["plate"],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.00013], [0, 0, 0, 1]],
            )
        record = assembly.step("S10.D8b")
    finally:
        kernel.close()
    checks = {c["id"]: c for c in record["checks"]}
    assert checks["coverage.skin"]["status"] == ("failed" if fault else "passed"), record
    ledger = record["outputs"]["pieces"]["plate"]
    assert {"covered_m2", "removed_m2", "missing_m2"} <= set(ledger)
    assert ledger["covered_m2"] + ledger["removed_m2"] + ledger["missing_m2"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert checks["coverage.area_sum"]["status"] == "passed"
    assert ledger["missing_m2"] == pytest.approx(
        (0.5 if trimmed else 1.0) if fault else 0.0, abs=1e-12
    )


@pytest.mark.parametrize(
    "lost", ["", "tip", "aft"], ids=["P0570-F1-control", "P0570-F1-kept-face", "P0570-F1-aft-of-te"]
)
def test_geom_kernel_export_coverage_catches_losses_that_reimport_and_triple_audit_miss(
    tmp_path, lost
):
    kernel_or_skip()
    from test_geom_kernel_assembly import box, wing

    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._geom_assembly import Assembly
    from cfdbender.geometry._geom_export import check_export, write_assembly
    from cfdbender.geometry.stages import GeneratedPiece
    from cfdbender.mesh._gmsh import open_kernel

    source = {"body": box((0.0, -1.0, -1.0), (4.0, 1.0, 1.0), split_x=2.5), "foil": wing()}
    pieces = {
        name: GeneratedPiece.from_patches(name, "1.0.0", patches)
        for name, patches in source.items()
    }
    rows = [{"dominant": ["body"], "dominated": ["foil"], "operation": "cut"}]
    kernel = open_kernel("coverage lesson fixtures")
    try:
        assembly = Assembly(
            kernel, pieces, rows, out=tmp_path, kinds={"body": "body", "foil": "lifting_surface"}
        )
        for stage, _ in assembly.STEPS:
            record = assembly.step(stage)
            assert all(c["status"] != "failed" for c in record["checks"]), record
            if stage == "S10.D8":
                break
        if lost:
            holder = "body" if lost == "aft" else "foil"
            candidates = (
                [
                    f
                    for f in assembly.faces[holder]
                    if (kernel.centroid(f)[0] > 2.5 and kernel.centroid(f)[1] > 0.99)
                    if lost == "aft"
                ]
                if lost == "aft"
                else [f for f in assembly.faces[holder] if kernel.centroid(f)[1] > 2.99]
            )
            assert len(candidates) == 1
            assert kernel.area(candidates[0]) == pytest.approx(
                3.0 if lost == "aft" else 0.2, rel=1e-6
            )
            kernel.remove(candidates)
            assembly.faces[holder].remove(candidates[0])
        coverage = assembly.step("S10.D8b")
        written = write_assembly(assembly, tmp_path / "combined.igs")
    finally:
        kernel.close()
    checks = {c["id"]: c for c in coverage["checks"]}
    assert checks["coverage.skin"]["status"] == ("failed" if lost else "passed")
    assert checks["coverage.area_sum"]["status"] == "passed"
    if lost:
        assert checks["coverage.skin"]["where"]["missing_m2"] == pytest.approx(
            3.0 if lost == "aft" else 0.2, rel=1e-3
        )
    config = resolve_config({"components": {n: {"type": "body"} for n in pieces}})
    read = check_export(
        "S11.R", config, configuration=None, members=list(pieces), out=tmp_path, source=written
    )
    assert all(c["status"] == "passed" for c in read["checks"]), read
    sewn = check_export(
        "S11.S", config, configuration=None, members=list(pieces), out=tmp_path, source=written
    )
    assert next(c for c in sewn["checks"] if c["id"] == "sew.triple_edges")["status"] == "passed"


def test_geom_kernel_export_export_manifest_has_public_piece_records_and_original_loops(tmp_path):
    import hashlib

    kernel_or_skip()
    _written_halves(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("format") == "cfdbender-geom-manifest/1"
    assert manifest["variant"] == "clean"
    assert len(manifest["surface_id"]) == 64
    assert len(manifest["build_id"]) == 64
    assert len(manifest["parameter_id"]) == 64
    assert len(manifest["pieces"]) == 1
    piece = manifest["pieces"][0]
    assert piece["component"] == "plate" and piece["faces"] == 2
    assert piece["trimmed_area_m2"] == pytest.approx(1.0, abs=1e-12)
    assert piece["source_area_m2"] == pytest.approx(1.0, abs=1e-12)
    assert piece["sha256"] == hashlib.sha256((tmp_path / piece["file"]).read_bytes()).hexdigest()
    sidecar = json.loads((tmp_path / piece["layout"]).read_text(encoding="utf-8"))
    assert sidecar["base_order"] == ["skin_0"]
    assert len(sidecar["source_trims"]) == 2


@pytest.mark.parametrize(
    "changed",
    ["", "face", "piece", "combined", "sidecar"],
    ids=[
        "P0570-base-control",
        "P0570-face-base",
        "P0570-piece-base",
        "P0570-combined-base",
        "P0570-sidecar-base",
    ],
)
def test_geom_kernel_export_profile_checks_every_written_base_and_source_sidecar(tmp_path, changed):
    from cfdbender.geometry._geom_export import check_export

    kernel_or_skip()
    resolved, written = _written_halves(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    piece = manifest["pieces"][0]
    if changed == "sidecar":
        path = tmp_path / piece["layout"]
        layout = json.loads(path.read_text(encoding="utf-8"))
        layout["bases"]["skin_0"]["controls"][0][0][2] += 1e-10
        path.write_text(json.dumps(layout), encoding="utf-8")
    elif changed:
        path = (
            tmp_path
            / {
                "face": piece["face_records"][0]["iges"],
                "piece": piece["file"],
                "combined": manifest["iges"],
            }[changed]
        )
        document = read_iges(path)
        values = next(e["parameters"] for e in document["entities"] if e["type"] == 128)
        ku, kv, du, dv = map(int, values[1:5])
        pole = 10 + ku + du + 2 + kv + dv + 2 + (ku + 1) * (kv + 1)
        values[pole + 2] = str(float(values[pole + 2]) + 1e-10)
        path.write_text(serialize_iges(document), encoding="ascii")
    record = check_export(
        "S11.P", resolved, configuration=None, members=["plate"], out=tmp_path, source=written
    )
    checks = {c["id"]: c for c in record["checks"]}
    assert "profile.base_shared" in checks, record
    assert checks["profile.base_shared"]["status"] == ("failed" if changed else "passed"), record
    assert checks["profile.reader"]["status"] == "passed", record


@pytest.mark.parametrize("fault", ["pref", "pcurve"], ids=["P0570-PREF-2", "P0570-no-pcurve"])
def test_geom_kernel_export_export_profile_rejects_an_unreadable_boundary(tmp_path, fault):
    from cfdbender.geometry._geom_export import check_export

    kernel_or_skip()
    resolved, written = _written_halves(tmp_path)
    path = tmp_path / "combined.igs"
    document = read_iges(path)
    boundary = next(e for e in document["entities"] if e["type"] == 142)
    boundary["parameters"][5 if fault == "pref" else 3] = "2" if fault == "pref" else "0"
    path.write_text(serialize_iges(document), encoding="ascii")
    record = check_export(
        "S11.P", resolved, configuration=None, members=["plate"], out=tmp_path, source=written
    )
    checks = {c["id"]: c for c in record["checks"]}
    assert "profile.reader" in checks, record
    assert checks["profile.reader"]["status"] == "failed", record


def test_geom_kernel_export_write_check_counts_faces_in_the_actual_file(tmp_path):
    from cfdbender.geometry._geom_export import check_export

    kernel_or_skip()
    resolved, written = _written_halves(tmp_path)
    path = tmp_path / "combined.igs"
    document = read_iges(path)
    face = next(e for e in document["entities"] if e["type"] == 144)
    document["entities"].remove(face)
    path.write_text(serialize_iges(document), encoding="ascii")
    record = check_export(
        "S11", resolved, configuration=None, members=["plate"], out=tmp_path, source=written
    )
    checks = {c["id"]: c for c in record["checks"]}
    assert "write.face_count" in checks, record
    assert checks["write.face_count"]["status"] == "failed", record
    assert checks["write.face_count"]["measured"] == 1


@pytest.mark.parametrize(
    "fault",
    ["", "pref", "base"],
    ids=["P0570-profile-run-control", "P0570-profile-run-pref", "P0570-profile-run-base"],
)
def test_geom_kernel_export_profile_failure_stops_its_substep_and_reaches_the_run_log(
    tmp_path, fault
):
    from cfdbender.geometry._geom_export import check_export
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.pipeline import Stage, StageOutcome, default_stages

    kernel_or_skip()

    def write(context):
        _, written = _written_halves(context.out)
        context.state["export"] = written
        if fault == "pref":
            path = context.out / "combined.igs"
            document = read_iges(path)
            next(e for e in document["entities"] if e["type"] == 142)["parameters"][5] = "2"
            path.write_text(serialize_iges(document), encoding="ascii")
        elif fault == "base":
            path = context.out / "plate.layout.json"
            layout = json.loads(path.read_text(encoding="utf-8"))
            layout["bases"]["skin_0"]["controls"][0][0][2] += 1e-10
            path.write_text(json.dumps(layout), encoding="utf-8")
        return StageOutcome(
            **check_export(
                "S11",
                context.resolved,
                configuration=None,
                members=context.members,
                out=context.out,
                source=written,
            )
        )

    stages = {s.row.stage: s for s in default_stages()}
    report = run_geometry(
        {"components": {"plate": {"type": "body"}}},
        out=tmp_path,
        until="S11.P",
        stages=[Stage(stages["S11"].row, write), stages["S11.P"]],
    )
    assert report["status"] == ("failed" if fault else "accepted"), report
    profile = json.loads((tmp_path / "S11.P.json").read_text(encoding="utf-8"))
    assert profile["outputs"]["inputs"]["files"]
    log = (tmp_path / "geom-run.log").read_text(encoding="utf-8")
    assert "S11.P" in log and "profile.base_shared" in log and "profile.reader" in log
    if fault:
        assert report["stage"] == "S11.P"
        assert report["check"] == ("profile.reader" if fault == "pref" else "profile.base_shared")


@pytest.mark.parametrize(
    "kind", ["body", "lifting_surface"], ids=["P0570-body-roundtrip", "P0570-wing-roundtrip"]
)
@pytest.mark.parametrize(
    "wrong", [False, True], ids=["P0570-source-control", "P0570-relabeled-source"]
)
def test_geom_kernel_export_roundtrip_regenerates_stored_surfaces_without_a_section_fit(
    tmp_path, kind, wrong
):
    import tomllib
    from pathlib import Path

    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._geom_export import generate_record

    kernel_or_skip()
    component = {"type": kind}
    if kind == "lifting_surface":
        component["cap"] = {"type": "none"}
    resolved = resolve_config({"components": {"part": component}})
    written = generate_record(resolved, str(tmp_path / "combined.igs"), None)
    if wrong:
        written["sources"]["part"]["source_surface_id"] = "0" * 64
    answer = _run_cold_roundtrip(tmp_path, resolved, written, ["part"])
    record, calls = answer["record"], answer["calls"]
    checks = {c["id"]: c for c in record["checks"]}
    assert "roundtrip.same_surface" in checks, record
    assert checks["roundtrip.same_surface"]["status"] == ("failed" if wrong else "passed"), record
    assert not calls, calls
    stored = tomllib.loads(Path(record["outputs"]["resolved_toml"]).read_text(encoding="utf-8"))
    assert stored["format"] == "cfdbender-geometry-resolved"
    if kind == "lifting_surface":
        sections = stored["components"]["part"]["sections"]
        assert sections["form"] == "bspline" and not sections["airfoils"]
        assert sections["items"] and all("bspline" in item for item in sections["items"])


@pytest.mark.parametrize(
    "kind",
    ["body", "lifting_surface", "invalid"],
    ids=["P0570-no-reverse", "P0570-reverse-control", "P0570-reverse-flat-plate"],
)
def test_geom_kernel_export_roundtrip_identifies_supported_pieces_and_reports_bad_geometry(
    tmp_path, kind
):
    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._geom_export import check_export, generate_record

    kernel_or_skip()
    if kind == "invalid":
        _, written = _written_halves(tmp_path)
        name = "plate"
        resolved = resolve_config({"components": {name: {"type": "lifting_surface"}}})
    else:
        name = "part"
        component = {"type": kind}
        if kind == "lifting_surface":
            component["cap"] = {"type": "none"}
        resolved = resolve_config({"components": {name: component}})
        written = generate_record(resolved, str(tmp_path / "combined.igs"), None)
    record = check_export(
        "S13", resolved, configuration=None, members=[name], out=tmp_path, source=written
    )
    checks = [c for c in record["checks"] if c["id"] == "roundtrip.identify"]
    assert checks, record
    assert (
        checks[0]["status"]
        == {"body": "skipped", "lifting_surface": "passed", "invalid": "failed"}[kind]
    ), record
    assert checks[0]["where"]["component"] == name


@pytest.mark.parametrize("wrong", [False, True], ids=["P0570-run-control", "P0570-run-source"])
def test_geom_kernel_export_roundtrip_records_failure_and_written_identities(tmp_path, wrong):
    from cfdbender.geometry._geom_export import generate_record
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.pipeline import Stage, StageOutcome, default_stages

    kernel_or_skip()

    def write(context):
        written = generate_record(context.resolved, str(context.out / "combined.igs"), None)
        context.state["assembly"] = {"export": written}
        if wrong:
            written["sources"]["part"]["source_surface_id"] = "0" * 64
        return StageOutcome()

    stages = {s.row.stage: s for s in default_stages()}
    report = run_geometry(
        {"components": {"part": {"type": "lifting_surface", "cap": {"type": "none"}}}},
        out=tmp_path,
        until="S13.R",
        stages=[Stage(stages["S0"].row, write), stages["S11"], stages["S13.R"]],
    )
    assert report["status"] == ("failed" if wrong else "incomplete"), report
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert report["surface_id"] == manifest["surface_id"]
    assert report["build_id"] == manifest["build_id"]
    log = (tmp_path / "geom-run.log").read_text(encoding="utf-8")
    assert "S13.R" in log and "roundtrip.same_surface" in log
    if wrong:
        assert report["stage"] == "S13.R" and report["check"] == "roundtrip.same_surface"


def _reader_piece(folder):
    from test_geom_kernel_assembly import wing

    from cfdbender.geometry import resolve_config
    from cfdbender.geometry._geom_assembly import Assembly
    from cfdbender.geometry._geom_export import write_assembly
    from cfdbender.geometry.stages import GeneratedPiece
    from cfdbender.mesh._gmsh import open_kernel

    roles = {"xmax": "trailing_edge", "zmax": "upper", "zmin": "lower"}
    patches = [replace(p, role=roles.get(p.family, "cap")) for p in wing()]
    piece = GeneratedPiece.from_patches("wing", "1.0.0", patches)
    resolved = resolve_config({"components": {"wing": {"type": "lifting_surface"}}})
    kernel = open_kernel("reader gate fixture")
    try:
        assembly = Assembly(kernel, {"wing": piece}, [], out=folder)
        assembly.step("S10.D1")
        written = write_assembly(assembly, folder / "combined.igs", resolved=resolved)
    finally:
        kernel.close()
    return resolved, written


@pytest.mark.parametrize(
    "fault", ["", "seam", "te"], ids=["P0570-reader-control", "P0570-open-seam", "P0570-no-TE"]
)
def test_geom_kernel_export_reader_checks_actual_mesh_and_sidecar_trailing_edge(tmp_path, fault):
    from pathlib import Path

    from cfdbender.geometry._geom_export import check_export
    from cfdbender.mesh._refine._obj import obj_text, read_obj

    kernel_or_skip()
    resolved, written = _reader_piece(tmp_path)

    def run(stage):
        return check_export(
            stage, resolved, configuration=None, members=["wing"], out=tmp_path, source=written
        )

    converted = run("S12.C")
    assert any(c["id"] == "gate.convert" for c in converted["checks"]), converted
    assert all(c["status"] == "passed" for c in converted["checks"]), converted
    gridded = run("S12.G")
    assert any(c["id"] == "gate.grid" for c in gridded["checks"]), gridded
    assert all(c["status"] == "passed" for c in gridded["checks"]), gridded
    layout_path = tmp_path / "wing.layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if fault == "te":
        for face in layout["faces"]:
            if face["role"] == "trailing_edge":
                face["role"] = "cap"
        layout_path.write_text(json.dumps(layout), encoding="utf-8")
    elif fault == "seam":
        path = Path(gridded["outputs"]["mesh"])
        mesh = read_obj(path)
        upper = next(f["name"] for f in layout["faces"] if f["role"] == "upper")
        nodes = mesh.family_vertices(upper)
        copies = dict(zip(nodes, range(len(mesh.verts), len(mesh.verts) + len(nodes)), strict=True))
        mesh.verts = np.concatenate((mesh.verts, mesh.verts[nodes]))
        mesh.families[upper] = [[copies[n] for n in f] for f in mesh.families[upper]]
        path.write_text(obj_text(mesh), encoding="utf-8")
    audited = run("S12")
    checks = {c["id"]: c for c in audited["checks"]}
    assert {"gate.conformal", "gate.te_found"} <= set(checks), audited
    assert checks["gate.conformal"]["status"] == ("failed" if fault == "seam" else "passed")
    assert checks["gate.te_found"]["status"] == ("passed" if not fault else "failed")
    if not fault:
        assert audited["outputs"]["trailing_edges"]["wing"]["edges"] > 0


@pytest.mark.parametrize("wrong", [False, True], ids=["P0570-reader-log", "P0570-reader-stop"])
def test_geom_kernel_export_reader_substeps_record_their_inputs_checks_and_failure(tmp_path, wrong):
    from cfdbender.geometry._geom_export import check_export
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.pipeline import Stage, StageOutcome, default_stages

    kernel_or_skip()
    stages = {s.row.stage: s for s in default_stages()}

    def write(context):
        _, written = _reader_piece(context.out)
        context.state["export"] = written
        return StageOutcome(
            **check_export(
                "S11",
                context.resolved,
                configuration=None,
                members=context.members,
                out=context.out,
                source=written,
            )
        )

    def grid(context):
        outcome = stages["S12.G"].run(context)
        if wrong:
            path = context.out / "wing.layout.json"
            layout = json.loads(path.read_text(encoding="utf-8"))
            for face in layout["faces"]:
                if face["role"] == "trailing_edge":
                    face["role"] = "cap"
            path.write_text(json.dumps(layout), encoding="utf-8")
        return outcome

    report = run_geometry(
        {"components": {"wing": {"type": "lifting_surface"}}},
        out=tmp_path,
        until="S12",
        stages=[
            Stage(stages["S11"].row, write),
            stages["S12.C"],
            Stage(stages["S12.G"].row, grid),
            stages["S12"],
        ],
    )
    assert report["status"] == ("failed" if wrong else "accepted"), report
    log = (tmp_path / "geom-run.log").read_text(encoding="utf-8")
    for stage, check in (
        ("S12.C", "gate.convert"),
        ("S12.G", "gate.grid"),
        ("S12", "gate.te_found"),
    ):
        record = json.loads((tmp_path / f"{stage}.json").read_text(encoding="utf-8"))
        assert record["outputs"]["inputs"]
        assert any(c["id"] == check for c in record["checks"])
        assert stage in log and check in log
    if wrong:
        assert report["stage"] == "S12" and report["check"] == "gate.te_found"


def test_geom_kernel_export_reader_conversion_rejects_a_deleted_written_face(tmp_path):
    from cfdbender.geometry._geom_export import check_export

    kernel_or_skip()
    resolved, written = _reader_piece(tmp_path)
    path = tmp_path / "combined.igs"
    document = read_iges(path)
    document["entities"].remove(next(e for e in document["entities"] if e["type"] == 144))
    path.write_text(serialize_iges(document), encoding="ascii")
    record = check_export(
        "S12.C", resolved, configuration=None, members=["wing"], out=tmp_path, source=written
    )
    checks = {c["id"]: c for c in record["checks"]}
    assert checks["gate.convert"]["status"] == "failed", record


def test_geom_kernel_export_missing_export_face_is_rejected_before_native_conversion(tmp_path):
    """A missing directory entity must not reach the native IGES reader."""
    from cfdbender.geometry._geom_export import check_export

    kernel_or_skip()
    resolved, written = _reader_piece(tmp_path)
    path = tmp_path / "combined.igs"
    document = read_iges(path)
    document["entities"].remove(next(e for e in document["entities"] if e["type"] == 144))
    path.write_text(serialize_iges(document), encoding="ascii")
    record = check_export(
        "S12.C", resolved, configuration=None, members=["wing"], out=tmp_path, source=written
    )
    assert record["checks"][0]["status"] == "failed"
    assert record["outputs"]["conversion"] == {
        "error": "written face count differs from the manifest",
        "faces": 4,
    }
    assert not (tmp_path / "S12-converted.igs").exists()


def test_geom_kernel_export_clean_lifting_export_runs_all_export_reader_and_roundtrip_substeps(
    tmp_path,
):
    from cfdbender.geometry._identify import run_geometry
    from cfdbender.geometry.pipeline import default_stages

    kernel_or_skip()
    stages = [s for s in default_stages() if s.row.stage.startswith(("S11", "S12", "S13"))]
    report = run_geometry(
        {"components": {"wing": {"type": "lifting_surface", "cap": {"type": "none"}}}},
        out=tmp_path,
        until="S13",
        stages=stages,
    )
    assert report["status"] == "accepted", report
    assert not report["not_run"], report
    for stage in stages:
        record = json.loads((tmp_path / f"{stage.row.stage}.json").read_text(encoding="utf-8"))
        assert record["status"] == "passed", record
        assert record["outputs"]["inputs"], record
