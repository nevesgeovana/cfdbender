"""The products of a sliding-mesh point are read from the per-zone files SU2 writes."""

from __future__ import annotations

import csv
import io
import json
import warnings

import test_runtypes_b1_workflows as rt
from wiring_support import block_volume, box_surface

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import workspace_path
from cfdbender.matrix._point_products import point_products
from cfdbender.matrix.exports import CONTEXT, _cell
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

SLIDING = '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_OUT", "X_IN"]\n'
MESH = (
    "NZONE= 2\nIZONE= 1\nNDIME= 3\nNMARK= 2\nMARKER_TAG= wing\nMARKER_ELEMS= 1\n"
    "MARKER_TAG= X_OUT\nMARKER_ELEMS= 1\nIZONE= 2\nNDIME= 3\nNMARK= 2\nMARKER_TAG= B1\n"
    "MARKER_ELEMS= 1\nMARKER_TAG= X_IN\nMARKER_ELEMS= 1\n"
)
PPROC = """history = true
[[sections.distributions]]
families = "wing"
planes = ["XZ"]
count = 4
[[sections.distributions]]
families = "B1"
planes = ["XZ"]
count = 4
[[sections.distributions]]
families = ["wing", "B1"]
planes = ["XZ"]
count = 4
[volume_section]
shape = "rectangle"
plane = "XZ"
frame = "REFERENCE"
corners_m = [-0.5, -0.5, 5.0, 0.5]
points = [4, 2]
offset_m = 0.5
[[probes]]
frame = "REFERENCE"
parameters = ["VX", "MACH"]
points = 3
[[probes.lines]]
start = [0.0, 0.5, 0.0]
end = [4.5, 0.5, 0.0]
"""
SHIFT = 4.0
STEPS = (35, 36)


def _point(root, zones=(0, 1), pproc=PPROC, markers='["wing", "B1"]'):
    rt.workspace(
        root,
        workflow="unsteady_rotor",
        var="DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_ITERS_AVG: 2 / EXPORT_UNSTEADY_LAST_ITER: 2",
        setup=f"MARKER_PLOTTING = {markers}\n" + SLIDING,
    )
    (root / "inputs/pproc/p001.toml").write_text(pproc, encoding="ascii")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    adapter = load_solver("su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    (folder / "mesh.su2").write_text(MESH, encoding="ascii")
    p_inf = adapter.product_inputs(point).free_stream.p_inf
    for step in STEPS:
        for zone in zones:
            shift = SHIFT * zone
            box_surface(folder / f"surface_flow_{zone}_{step - 1:05d}.vtu", p_inf, shift_x=shift)
            block_volume(folder / f"flow_{zone}_{step - 1:05d}.vtu", p_inf, shift_x=shift)
    return root, point, adapter, folder


def _products(root, point, adapter):
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
    return {
        path.relative_to(root / "post/rt").as_posix(): facts for path, facts in found.files.items()
    }, found


def _rows(facts):
    return list(csv.DictReader(io.StringIO(facts["text"])))


def test_the_per_zone_files_are_named_with_their_zone_and_step(tmp_path):
    _, point, adapter, folder = _point(tmp_path)
    inputs = adapter.product_inputs(point)
    base = point.point.directory
    assert [(f.step, f.zone, f.path) for f in inputs.volume] == [
        (s, z, f"{base}/flow_{z}_{s - 1:05d}.vtu") for s in STEPS for z in (0, 1)
    ]
    assert [f.zone for f in inputs.surface] == [0, 1, 0, 1]
    assert not inputs.missing
    (folder / "flow_1_00034.vtu").unlink()
    assert (f"{base}/flow_1_00034.vtu", "the run did not write it") in adapter.product_inputs(
        point
    ).missing


def test_a_probe_is_found_in_whichever_zone_holds_it_and_a_miss_is_reported(tmp_path):
    root, point, adapter, _ = _point(tmp_path)
    files, _ = _products(root, point, adapter)
    rows = _rows(files["probes/P7001/DP_probes.csv"])
    assert [r["STEP"] for r in rows] == ["36"] * 3
    assert [r["X"] for r in rows] == ["0.0", "2.25", "4.5"]
    assert [r["VX"] for r in rows] == ["10.0", "NA", "14.5"]
    assert files["probes/P7001/DP_probes.csv"]["points_not_found"] == [
        "STEP 36: 1 point(s) in no cell"
    ]


def test_the_volume_cut_spans_the_zones(tmp_path):
    root, point, adapter, _ = _point(tmp_path)
    files, found = _products(root, point, adapter)
    assert not [s for s in found.skipped if "vsec" in s[0]], found.skipped
    # four x positions from -0.5 to 5 by two y: -0.5 and 1.33 lie in zone 0 (x -1 to 2), 3.17
    # and 5 in zone 1 (x 3 to 6); none is left out
    provenance = json.loads(files["fields/P7001/DP_vsec_step_36.vtk.provenance.json"]["text"])
    assert (provenance["declared_points"], provenance["sample_count"]) == (8, 8)
    assert provenance["source_file"] == "flow_0_00035.vtu, flow_1_00035.vtu"


def test_a_section_of_the_stationary_markers_cuts_the_stationary_zone_file(tmp_path):
    root, point, adapter, _ = _point(tmp_path)
    files, _ = _products(root, point, adapter)
    rows = _rows(files["sections/P7001/DP_sloads_wing.csv"])
    assert {r["STEP"] for r in rows} == {"35", "36"}
    assert files["sections/P7001/DP_sloads_wing.csv"]["sources"] == [
        f"{point.point.directory}/surface_flow_0_{s - 1:05d}.vtu" for s in STEPS
    ]
    assert all(abs(float(r["Fz"]) - 100.0) < 1e-6 for r in rows)


def test_a_section_of_the_rotating_markers_cuts_the_rotating_zone_file(tmp_path):
    root, point, adapter, _ = _point(tmp_path)
    files, _ = _products(root, point, adapter)
    facts = files["sections/P7001/DP_sloads_B1.csv"]
    assert facts["sources"] == [
        f"{point.point.directory}/surface_flow_1_{s - 1:05d}.vtu" for s in STEPS
    ]
    assert min(float(r["X_QC"]) for r in _rows(facts)) >= SHIFT - 1e-9


def test_markers_that_span_the_zones_cut_both_files_in_one_table(tmp_path):
    root, point, adapter, _ = _point(tmp_path)
    files, _ = _products(root, point, adapter)
    facts = files["sections/P7001/DP_sloads_wing-B1.csv"]
    assert facts["sources"] == [
        f"{point.point.directory}/surface_flow_{z}_{s - 1:05d}.vtu" for s in STEPS for z in (0, 1)
    ]
    single = _rows(files["sections/P7001/DP_sloads_wing.csv"])
    both = _rows(facts)
    assert len(both) == len(single)
    assert list(both[0]) == list(single[0])
    # the loads of both bodies are summed at each station
    assert all(abs(float(r["Fz"]) - 200.0) < 1e-6 for r in both)


def test_an_entry_cutting_part_of_a_zone_file_is_refused_with_the_reason(tmp_path):
    root, point, adapter, folder = _point(tmp_path, markers='["wing", "tail", "B1"]')
    stationary = "NMARK= 3\nMARKER_TAG= tail\nMARKER_ELEMS= 1\nMARKER_TAG= wing"
    (folder / "mesh.su2").write_text(
        MESH.replace("NMARK= 2\nMARKER_TAG= wing", stationary), encoding="ascii"
    )
    _, found = _products(root, point, adapter)
    reasons = [reason for path, reason in found.skipped if "sloads_wing.csv" in path]
    assert reasons and "all of" in reasons[0] and "tail" in reasons[0]


def test_the_zone_of_each_marker_is_read_from_the_mesh(tmp_path):
    from cfdbender.solvers.su2._products import zone_markers

    path = tmp_path / "mesh.su2"
    path.write_text(MESH, encoding="ascii")
    assert zone_markers(path) == {"wing": 0, "X_OUT": 0, "B1": 1, "X_IN": 1}
    path.write_text("NDIME= 3\nMARKER_TAG= a\n  MARKER_TAG = b % note\n", encoding="ascii")
    assert zone_markers(path) == {"a": 0, "b": 0}


def _samplers(tmp_path, shift_second):
    from cfdbender.post.volume import VolumeSampler
    from cfdbender.post.vtu import read_vtu

    block_volume(tmp_path / "a.vtu", 1000.0)
    block_volume(tmp_path / "b.vtu", 5000.0, shift_x=shift_second)
    return [VolumeSampler(read_vtu(tmp_path / name)) for name in ("a.vtu", "b.vtu")]


def test_a_point_in_two_zones_takes_the_value_of_the_first_zone_in_order(tmp_path):
    from cfdbender.post.volume import ZonedSampler

    first, second = _samplers(tmp_path, 0.0)
    point = [[0.5, 0.5, 0.5]]
    a = first.sample(point, ["Pressure"]).values["Pressure"][0]
    b = second.sample(point, ["Pressure"]).values["Pressure"][0]
    assert a != b
    forward = ZonedSampler([first, second]).sample(point, ["Pressure"])
    backward = ZonedSampler([second, first]).sample(point, ["Pressure"])
    assert forward.values["Pressure"][0] == a
    assert backward.values["Pressure"][0] == b


def test_a_point_on_a_face_is_found_in_the_zone_that_has_the_face(tmp_path):
    from cfdbender.post.volume import ZonedSampler

    first, second = _samplers(tmp_path, 4.0)
    # x -1 face of the first block, y -1 face of the second (x 3 to 6), and the gap between
    result = ZonedSampler([first, second]).sample([[-1.0, 0.0, 0.0], [5.0, -1.0, 0.0], [2.5, 0, 0]])
    assert list(result.found) == [True, True, False]
