"""The run types' open items: origins of carried keys, provenance, left-out points, segments."""

from __future__ import annotations

import json
import sys
import warnings
from types import SimpleNamespace

import test_runtypes_b1_workflows as rt
from test_p0000_item3c2_run import cli, run_api
from test_p0000_item3c_adapter import workspace as mock_workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.matrix._unsteady_polar import left_out
from cfdbender.solvers import mock
from cfdbender.solvers.su2._execute import run
from cfdbender.workspace.planning import plan_workspace

WINDOW = "DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_ITERS_AVG: 2"


def _plan(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return plan_workspace(root, "rt")


def test_the_keys_a_run_type_carries_are_listed_with_their_origin(tmp_path, capsys):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=WINDOW)
    point = _plan(tmp_path).points[0]
    carried = {k for k, origin in point.origins.items() if origin == "run type unsteady_rotor"}
    assert {"TIME_DOMAIN", "TIME_STEP", "TIME_ITER", "GRID_MOVEMENT", "ROTATION_RATE"} <= carried
    assert "SOLVER" not in carried
    cli(["plan", "rt", "--workspace", str(tmp_path), "--verbose"])
    assert "setup keys from run type unsteady_rotor: " in capsys.readouterr().out


def test_a_steady_row_carries_no_key(tmp_path):
    rt.workspace(tmp_path, condition="MACH:0.0529, ALPHA:0, BETA:0")
    point = _plan(tmp_path).points[0]
    assert not [o for o in point.origins.values() if o.startswith("run type")]


def test_the_record_states_the_run_type_its_declarations_and_the_turbulence_model(tmp_path):
    rt.workspace(tmp_path, workflow="unsteady_rotor", var=WINDOW)
    plan = _plan(tmp_path)
    RunRegistry(tmp_path).record(plan)
    row = json.loads((tmp_path / "runs.json").read_text())["records"][0]
    assert row["provenance"] == {
        "run_type": "unsteady_rotor",
        "declarations": "su2/unsteady_rotor",
        "numerics_id": None,
        "turbulence_model": "SA",
        "turbulence_model_from": "setup",
    }
    assert RunRegistry(tmp_path).read()[0].point.provenance == row["provenance"]


def test_a_model_the_row_chose_is_said_to_come_from_the_row(tmp_path):
    rt.workspace(tmp_path, workflow="qsteady_rotor", var="TURBULENCE_MODEL: SST")
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text().replace('KIND_TURB_MODEL = "SA"\n', ""))
    provenance = _plan(tmp_path).points[0].provenance
    assert (provenance["turbulence_model"], provenance["turbulence_model_from"]) == ("SST", "row")


def test_a_steady_record_without_a_model_keeps_its_bytes(tmp_path):
    rt.workspace(tmp_path, condition="MACH:0.0529, ALPHA:0, BETA:0")
    RunRegistry(tmp_path).record(_plan(tmp_path))
    assert "provenance" not in json.loads((tmp_path / "runs.json").read_text())["records"][0]


def test_a_time_accurate_point_left_out_of_the_unsteady_polar_is_named():
    rows = [
        {"pol": "7001", "point_id": "7001/DP-A", "result": {"time_average": {"skipped": "short"}}},
        {"pol": "7001", "point_id": "7001/DP-B", "result": {"time_average": {"means": {}}}},
    ]
    assert left_out(rows, "rt") == (
        ("polars/P7001_rt_uns_avg.csv#point=7001/DP-A", "left out: short"),
    )


def test_collect_names_the_left_out_point_in_the_manifest(tmp_path, monkeypatch):
    mock_workspace(tmp_path)
    plan = plan_workspace(tmp_path, "study")
    run_api(plan, check_ram=lambda: None)
    original = mock.collect

    def collect(point):
        found = original(point)
        data = {**found.to_dict(), "time_average": {"skipped": "the history ends at step 3"}}
        return SimpleNamespace(
            to_dict=lambda: data,
            classification=found.classification,
            solver_version=found.solver_version,
        )

    monkeypatch.setattr(mock, "collect", collect)
    cli(["collect", "study", "--workspace", str(tmp_path)])
    data = json.loads((tmp_path / "post/study/products.json").read_text())
    point = plan.points[0].point_id
    assert {
        "product": f"polars/P1001_study_uns_avg.csv#point={point}",
        "reason": "left out: the history ends at step 3",
    } in data["skipped"]


FAKE = """import sys
from pathlib import Path
name = sys.argv[-1]
with Path("ran.txt").open("a") as stream:
    stream.write(name + chr(10))
print("SU2 Release 8.6.0", flush=True)
sys.exit(3 if Path("fail_first").exists() else 0)
"""


def _segments(root, fail=False):
    script = root / "fake_su2.py"
    script.write_text(FAKE, encoding="ascii")
    rt.workspace(
        root,
        workflow="unsteady_rotor",
        var=WINDOW + " / EXPORT_UNSTEADY_LAST_ITER: 2",
        setup=f'[solver]\nroute = "executable"\nexecutable = {json.dumps(script.as_posix())}\n',
    )
    point = load_point(root, _plan(root).points[0], solver="su2")
    folder = root / point.point.directory
    if fail:
        original = sys.modules["cfdbender.solvers.su2._execute"].stage_files

        def staged(p, values, f):
            original(p, values, f)
            (f / "fail_first").write_text("x")

        return point, folder, staged
    return point, folder, None


def test_a_local_run_runs_both_segments_in_order(tmp_path):
    point, folder, _ = _segments(tmp_path)
    run(point, lambda: None)
    assert (folder / "ran.txt").read_text().split() == ["su2.cfg", "su2_segment2.cfg"]
    assert json.loads((folder / "execution.json").read_text())["exit_code"] == 0


def test_a_failed_first_segment_does_not_start_the_second(tmp_path, monkeypatch):
    point, folder, staged = _segments(tmp_path, fail=True)
    monkeypatch.setattr("cfdbender.solvers.su2._execute.stage_files", staged)
    run(point, lambda: None)
    assert (folder / "ran.txt").read_text().split() == ["su2.cfg"]
    assert json.loads((folder / "execution.json").read_text())["exit_code"] == 3
