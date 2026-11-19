"""Collect writes each point's declared products with pyflightstream's names and columns."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import warnings

import pytest
import test_runtypes_b1_workflows as rt
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace
from wiring_support import block_volume, box_surface

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import workspace_path
from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import CONTEXT, _cell
from cfdbender.post.product_frames import posed, product_frames
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

PPROC = """history = true
[sections]
count = 4
[[sections.distributions]]
families = "wing"
planes = ["XZ"]
integrate = true
[[sections.distributions]]
families = "tail"
planes = ["XZ"]
[volume_section]
shape = "rectangle"
plane = "XZ"
frame = "REFERENCE"
corners_m = [0.0, -0.5, 1.0, 0.5]
points = [3, 3]
offset_m = 0.5
[[probes]]
frame = "REFERENCE"
points = 3
parameters = ["VX", "MACH", "CP_FREE", "CP_REF"]
[[probes.lines]]
start = [0.0, 0.5, 0.0]
end = [1.0, 0.5, 0.0]
"""
LEAD = ["POL", "STEP", "time_s", "FAMILY", "PLANE", "ROTOR", "AZIMUTH", *CONTEXT]


def table(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def collected(root, pproc=PPROC):
    workspace(root)
    (root / "inputs/pproc/p001.toml").write_text(pproc, encoding="ascii")
    plan = plan_workspace(root, "study")
    run_api(plan, check_ram=lambda: None)
    assert cli(["collect", "study", "--workspace", str(root)]) == 0
    name = plan.points[0].point_id.split("/", 1)[1]
    return root / "post/study", name


def test_the_sections_files_of_a_steady_point(tmp_path):
    post, name = collected(tmp_path)
    loads = post / f"sections/P1001/{name}_sloads_wing.csv"
    cp = post / f"sections/P1001/{name}_cp_wing.csv"
    header = loads.read_text().splitlines()[0].split(",")
    assert header == [
        *LEAD,
        *("Offset", "Chord", "X_QC", "Z_QC", "Fx", "Fz", "Moment"),
        *("Strip_length", "Fx_int", "Fz_int", "My_int"),
    ]
    rows = table(loads)
    assert len(rows) == 4
    assert {(r["POL"], r["FAMILY"], r["PLANE"], r["ROTOR"], r["AZIMUTH"]) for r in rows} == {
        ("1001", "wing", "XZ", "NA", "NA")
    }
    assert {r["STEP"] for r in rows} == {"2"} and {r["time_s"] for r in rows} == {"NA"}
    for row in rows:
        assert math.isclose(float(row["Fz"]), 100.0, rel_tol=1e-9)
        assert abs(float(row["Fx"])) < 1e-9
        assert math.isclose(float(row["Chord"]), math.hypot(1.0, 0.1))  # the farthest pair
    assert math.isclose(sum(float(r["Fz_int"]) for r in rows), 100.0, rel_tol=1e-9)
    cp_header = cp.read_text().splitlines()[0].split(",")
    assert cp_header[: len(LEAD) + 1] == [*LEAD, "SECTION"]
    assert not {"CF", "Theta", "Delta*", "Delta", "H"} & set(cp_header)
    assert {"Cp", "Mach", "vx", "vtot"} <= set(cp_header) and "Cp_ref" not in cp_header


def test_the_volume_section_and_the_probes(tmp_path):
    post, name = collected(tmp_path)
    vtk = post / f"fields/P1001/{name}_vsec.vtk"
    assert "POINTS 9 float" in vtk.read_text()
    assert (post / f"fields/P1001/{name}_vsec.vtk.provenance.json").is_file()
    rows = table(post / f"probes/P1001/{name}_probes.csv")
    header = list(rows[0])
    assert header == [
        "POL",
        "PROBE",
        "X",
        "Y",
        "Z",
        "FRAME",
        "STEP",
        *CONTEXT,
        "VX",
        "MACH_LOCAL",
        "CP_FREE",
    ]
    assert [float(r["VX"]) for r in rows] == pytest.approx([10.0, 10.5, 11.0])
    assert [float(r["CP_FREE"]) for r in rows] == pytest.approx([0.0, 50 / 60, 100 / 60])
    assert {r["STEP"] for r in rows} == {"NA"}


def test_the_manifest_names_every_product_its_kind_and_what_is_absent_or_skipped(tmp_path):
    post, name = collected(tmp_path)
    data = json.loads((post / "products.json").read_text())
    by_path = {entry["path"]: entry for entry in data["products"]}
    cp = by_path[f"post/study/sections/P1001/{name}_cp_wing.csv"]
    assert (cp["product"], cp["kind"], cp["point"]) == ("sections_cp", "instant", f"1001/{name}")
    assert cp["absent_columns"]["CF"] == "the mock writes no boundary layer"
    assert cp["absent_columns"]["Cp_ref"] == "the adapter does not fill it"
    probes = by_path[f"post/study/probes/P1001/{name}_probes.csv"]
    assert probes["absent_columns"] == {"CP_REF": "the mock states no reference speed"}
    skipped = {entry["product"]: entry["reason"] for entry in data["skipped"]}
    reason = skipped[f"sections/P1001/{name}_cp_tail.csv"]
    assert "families ['tail'] name no family of the surface file (it holds: wing)" in reason
    log = (post / "post.log").read_text()
    assert f"wrote post/study/sections/P1001/{name}_sloads_wing.csv" in log
    assert data["complete"] is True


def test_collect_again_rewrites_the_same_bytes(tmp_path):
    post, _ = collected(tmp_path)
    before = {p: p.read_bytes() for p in post.rglob("*") if p.is_file()}
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert {p: p.read_bytes() for p in post.rglob("*") if p.is_file()} == before


def test_a_missing_file_skips_its_products_and_keeps_the_others_and_the_polar(tmp_path):
    workspace(tmp_path)
    (tmp_path / "inputs/pproc/p001.toml").write_text(PPROC, encoding="ascii")
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    (workspace_path(tmp_path, plan.points[0].directory) / "volume.vtu").unlink()
    assert cli(["collect", "study", "--workspace", str(tmp_path)]) == 0
    post = tmp_path / "post/study"
    name = plan.points[0].point_id.split("/", 1)[1]
    data = json.loads((post / "products.json").read_text())
    skipped = {entry["product"]: entry["reason"] for entry in data["skipped"]}
    assert "volume.vtu: the run did not write it" in skipped[f"probes/P1001/{name}_probes.csv"]
    assert f"fields/P1001/{name}_vsec.vtk" in skipped
    assert (post / f"sections/P1001/{name}_sloads_wing.csv").is_file()
    assert (post / "polars").is_dir()


def test_the_polars_of_a_point_declaring_products_are_those_of_one_declaring_none(tmp_path):
    def polars(root):
        folder = root / "post/study/polars"
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}

    collected(tmp_path / "a")
    collected(tmp_path / "b", pproc="history = true\n")
    assert polars(tmp_path / "a") == polars(tmp_path / "b")
    assert not (tmp_path / "b/post/study/sections").exists()


def test_the_rotor_frames_follow_the_reference(tmp_path):
    reference = {"ROTOR": {**_rotor(), "families_blades": ["B1", "B2", "B3", "B4"]}}
    frames = product_frames(reference, moment_point=(1.0, 2.0, 3.0))
    assert frames["MRP"].origin == (1.0, 2.0, 3.0)
    assert frames["ROTOR_SMRP"].axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # blade two of four sits a quarter turn ahead, right-handed about +x: z goes to -y
    second = frames["ROTOR_RMRP2"].axes
    assert second[2] == pytest.approx((0.0, -1.0, 0.0))
    turning = product_frames(reference, steps_per_revolution=36.0)["ROTOR_RMRP"]
    assert turning.spin is not None
    # rpm_sign -1: after 9 steps of 36 the frame has turned a quarter turn the other way
    assert posed(turning, 9).axes[2] == pytest.approx((0.0, 1.0, 0.0))


def _rotor():
    return {
        "kind": "rotor",
        "x_m": 0.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "axis": "X",
        "rpm_sign": -1,
        "diameter_m": 0.4,
        "blade1": {"azimuth_deg": 0.0, "zero": "Z"},
    }


SU2_SECTIONS = """history = true
[[sections.distributions]]
families = "B1"
planes = ["XZ"]
count = 4
[[probes]]
parameters = ["VX", "VELOCITY", "CP_REF", "BL_SKIN_FRICTION"]
kind = "instantaneous"
points = 3
[[probes.lines]]
start = [0.0, 0.5, 0.0]
end = [1.0, 0.5, 0.0]
[[probes]]
frame = "REFERENCE"
parameters = ["MACH"]
points = 2
[[probes.lines]]
start = [0.0, 0.5, 0.0]
end = [1.0, 0.5, 0.0]
"""


def _su2_window(root):
    rt.workspace(
        root,
        workflow="unsteady_rotor",
        var="DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_ITERS_AVG: 2 / EXPORT_UNSTEADY_LAST_ITER: 2",
        setup='MARKER_PLOTTING = ["B1"]\nMARKER_MONITORING = ["B1"]\n',
    )
    (root / "inputs/pproc/p001.toml").write_text(SU2_SECTIONS, encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    adapter = load_solver("su2")
    inputs = adapter.product_inputs(point)
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    assert inputs.free_stream is not None
    for k in (34, 35):
        box_surface(folder / f"surface_flow_{k:05d}.vtu", inputs.free_stream.p_inf)
        block_volume(folder / f"flow_{k:05d}.vtu", inputs.free_stream.p_inf)
    return root, point, adapter


def test_a_time_accurate_su2_point_writes_the_series_of_its_export_window(tmp_path):
    root, point, adapter = _su2_window(tmp_path)
    found = point_products(
        root,
        "7001",
        "DP",
        point,
        adapter,
        lead=dict.fromkeys(CONTEXT, "NA"),
        moment_point=(0.0, 0.0, 0.0),
        cell=_cell,
        target=lambda relative: root / "post/rt" / relative,
    )
    files = {
        path.relative_to(root / "post/rt").as_posix(): facts for path, facts in found.files.items()
    }
    loads = files["sections/P7001/DP_sloads_B1.csv"]
    rows = list(csv.DictReader(io.StringIO(loads["text"])))
    assert [r["STEP"] for r in rows] == ["35"] * 4 + ["36"] * 4
    # blade one of a left-handed rotor at 10 degrees a step: step 35 is 10 degrees, 36 is 0
    assert {(r["STEP"], r["AZIMUTH"], r["ROTOR"]) for r in rows} == {
        ("35", "10.0", "ROTOR"),
        ("36", "0.0", "ROTOR"),
    }
    assert all(math.isclose(float(r["Fz"]), 100.0, rel_tol=1e-9) for r in rows)
    assert (loads["kind"], loads["steps_tabled"]) == ("history", [35, 36])
    cp = files["sections/P7001/DP_cp_B1.csv"]
    assert set(cp["absent_columns"]) == {"Theta", "Delta*", "Delta", "H"}
    assert "CF" in cp["text"].splitlines()[0].split(",")
    history = list(csv.DictReader(io.StringIO(files["probes/P7001/DP_probes_history.csv"]["text"])))
    assert [r["STEP"] for r in history] == ["35"] * 3 + ["36"] * 3
    last = list(csv.DictReader(io.StringIO(files["probes/P7001/DP_probes.csv"]["text"])))
    assert [r["STEP"] for r in last] == ["36", "36"]
    assert files["probes/P7001/DP_probes_history.csv"]["absent_columns"] == {
        "BL_SKIN_FRICTION": "SU2 writes no boundary-layer integral thickness or shape factor"
    }
    assert "CP_REF" in history[0] and "VELOCITY" in history[0]


def test_su2_names_the_steps_its_run_did_not_write(tmp_path):
    root, point, adapter = _su2_window(tmp_path)
    folder = workspace_path(root, point.point.directory)
    (folder / "flow_00034.vtu").unlink()
    inputs = adapter.product_inputs(point)
    assert [f.step for f in inputs.volume] == [36]
    assert (f"{point.point.directory}/flow_00034.vtu", "the run did not write it") in inputs.missing
