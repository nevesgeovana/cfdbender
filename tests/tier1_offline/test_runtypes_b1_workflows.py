"""Run types are selected by the WORKFLOW cell, as pyflightstream selects them, and carry keys."""

from __future__ import annotations

import math
import warnings

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import PlanError
from cfdbender.core.workspace import init_workspace
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.solvers.su2._config import config, extra_files
from cfdbender.workspace.planning import plan_workspace

HEADER = (
    "POL | RUN | DESCRIPTION | FLIGHT_CONDITION | SWEEP_VALUES | MESH | REF | SETUP | PPROC "
    "| SYMMETRY | SYMMETRY_LOADS | WORKFLOW | VAR_NAMES_VALUES"
)
ROTOR = """
[ROTOR]
kind = "rotor"
axis = "X"
rpm_sign = -1
diameter_m = 0.4064
families_blades = ["B1", "B2", "B3", "B4", "B5", "B6"]
blade1 = { azimuth_deg = 0.0, zero = "Z" }
"""
BASE_SETUP = (
    'iterations = 40\nSOLVER = "RANS"\nKIND_TURB_MODEL = "SA"\nMARKER_FAR = ["farfield"]\n'
    "freestream_temperature = 288.15\nGAS_CONSTANT = 287.058\nGAMMA_VALUE = 1.4\n"
)
SPEED = 0.0529 * math.sqrt(1.4 * 287.058 * 288.15)


def workspace(
    root,
    *,
    workflow="-",
    condition="MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:0.4",
    var="-",
    setup="",
    symmetry="-",
    rotor=ROTOR,
):
    files = {
        "inputs/setups/s001.toml": BASE_SETUP + setup,
        "inputs/references/r001.toml": "area = 1.0\nlength = 0.5\nmoment_origin = [0.0, 0.0, 0.0]"
        + rotor,
        "inputs/pproc/p001.toml": "history = true",
        "inputs/mesh/rotor.su2": "NDIME= 3",
        "inputs/matrices/rt.fs": "\n".join(
            [
                HEADER,
                "-" * 60,
                f" 7001 | 1 | rotor | {condition} | - | rotor.su2 | 001 | 001 | 001 "
                f"| {symmetry} | - | {workflow} | {var} ",
            ]
        ),
    }
    init_workspace(root, solver="su2", available_solvers=available_solvers())
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text + "\n", encoding="ascii")
    return root


def deck(root):
    """The planned point, its SU2 keys and the files beside su2.cfg."""
    plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    text = load_solver("su2").translate(point)
    values = config(
        point.setup,
        point.reference,
        point.post,
        point.point.condition,
        geometry=point.geometry,
        run_type=point.point.run_type,
    )
    lines = dict(line.split("= ", 1) for line in text.splitlines())
    return plan.points[0], lines, extra_files(values)


def numbers(text):
    return [float(item) for item in text.split()]


@pytest.mark.parametrize("workflow", ["LEGACY", "Steady", "mrf"])
def test_a_workflow_that_is_not_a_run_type_is_refused(tmp_path, workflow):
    workspace(tmp_path, workflow=workflow)
    with pytest.raises(PlanError, match=f"WORKFLOW value '{workflow}' of POL 7001 names no known"):
        plan_workspace(tmp_path, "rt")


def test_an_unstated_workflow_is_steady_and_its_record_says_nothing_new(tmp_path):
    workspace(tmp_path, condition="MACH:0.0529, ALPHA:0", rotor="")
    point, lines, files = deck(tmp_path)
    assert point.run_type == "steady" and files == {}
    other = workspace(tmp_path / "b", workflow="steady", condition="MACH:0.0529, ALPHA:0", rotor="")
    assert deck(other)[1] == lines


def test_qsteady_rotor_carries_the_rotating_frame_the_setup_stated_before(tmp_path):
    workspace(tmp_path / "old", setup='GRID_MOVEMENT = "ROTATING_FRAME"\n')
    workspace(tmp_path / "new", workflow="qsteady_rotor")
    _, old, _ = deck(tmp_path / "old")
    point, new, _ = deck(tmp_path / "new")
    assert point.run_type == "qsteady_rotor"
    assert new == old, "the run type writes what the setup wrote: no key differs"


def test_qsteady_rotor_takes_its_speed_from_rpm(tmp_path):
    workspace(tmp_path, workflow="qsteady_rotor", condition="MACH:0.0529, ALPHA:0, RPM:6000")
    _, lines, _ = deck(tmp_path)
    assert numbers(lines["ROTATION_RATE"]) == pytest.approx(
        [-2.0 * math.pi * 100.0, 0.0, 0.0], rel=1e-12
    )


def test_qsteady_rotor_refuses_an_angle_and_a_missing_rotor(tmp_path):
    workspace(tmp_path / "a", workflow="qsteady_rotor", condition="MACH:0.0529, ALPHA:2, RPM:600")
    with pytest.raises(PlanError, match="an angle makes the inflow vary with azimuth"):
        plan_workspace(tmp_path / "a", "rt")
    workspace(tmp_path / "b", workflow="qsteady_rotor", rotor="")
    with pytest.raises(PlanError, match="WORKFLOW qsteady_rotor needs the rotor declaration"):
        plan_workspace(tmp_path / "b", "rt")


def test_qsteady_rotor_writes_the_periodic_pair_from_the_symmetry(tmp_path):
    workspace(
        tmp_path,
        workflow="qsteady_rotor",
        symmetry="PERIODIC",
        var="PERIODIC_COPIES: 6",
        setup='periodic_markers = ["left", "right"]\n',
    )
    _, lines, _ = deck(tmp_path)
    assert lines["MARKER_PERIODIC"] == "( left, right, 0, 0, 0, 60, 0, 0, 0, 0, 0 )"


def test_unsteady_writes_the_time_keys_from_the_explicit_clock(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 200 / LAST_ITERS_AVG: 100",
    )
    point, lines, files = deck(tmp_path)
    assert (lines["TIME_DOMAIN"], lines["TIME_MARCHING"]) == ("YES", "DUAL_TIME_STEPPING-2ND_ORDER")
    assert (float(lines["TIME_STEP"]), lines["TIME_ITER"], lines["INNER_ITER"]) == (
        0.001,
        "200",
        "40",
    )
    assert lines["WINDOW_START_ITER"] == "100", "step 101 is SU2's time iteration 100"
    assert "ITER" not in lines and "GRID_MOVEMENT" not in lines and files == {}
    for key in ("DELTA_TIME", "TIME_ITERATIONS", "LAST_ITERS_AVG"):
        assert key not in lines


def test_unsteady_without_a_window_is_refused(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 200",
    )
    with pytest.raises(PlanError, match="names the run type unsteady and states no LAST_ITERS_AVG"):
        plan_workspace(tmp_path, "rt")


@pytest.mark.parametrize(
    ("rpm", "theta", "revs", "avg", "step", "iters", "start"),
    [
        (1000.0, 3.0, 5.0, 2.5, 0.0005, 600, 300),  # the colleague's hover clock
        (972.0, 1.5, 5.0, 2.5, 2.57201646090535e-4, 1200, 600),  # the colleague's twin clock
    ],
)
def test_unsteady_rotor_reproduces_the_colleagues_clocks(
    tmp_path, rpm, theta, revs, avg, step, iters, start
):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition=f"MACH:0.0529, ALPHA:0, RPM:{rpm:g}",
        var=f"DELTA_THETA: {theta} / REVOLUTIONS: {revs} / LAST_REVS_AVG: {avg}",
    )
    _, lines, files = deck(tmp_path)
    assert float(lines["TIME_STEP"]) == pytest.approx(step, rel=1e-12)
    assert (int(lines["TIME_ITER"]), int(lines["WINDOW_START_ITER"])) == (iters, start)
    assert lines["GRID_MOVEMENT"] == "RIGID_MOTION" and files == {}
    assert numbers(lines["ROTATION_RATE"]) == pytest.approx(
        [-2.0 * math.pi * rpm / 60.0, 0.0, 0.0], rel=1e-12
    )


def test_unsteady_rotor_with_a_sliding_interface_writes_one_file_per_zone(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 3 / REVOLUTIONS: 5 / LAST_REVS_AVG: 2.5",
        setup='[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_BOXD1", "X_BOX1"]\n',
    )
    point, lines, files = deck(tmp_path)
    assert lines["MULTIZONE"] == "YES" and lines["CONFIG_LIST"] == "( zone_0.cfg, zone_1.cfg )"
    assert lines["MARKER_ZONE_INTERFACE"] == lines["MARKER_FLUID_INTERFACE"]
    assert (lines["OUTER_ITER"], lines["INNER_ITER"]) == ("40", "1")
    assert "GRID_MOVEMENT" not in lines and "ROTATION_RATE" not in lines
    zone = dict(line.split("= ", 1) for line in files["zone_1.cfg"].splitlines())
    # the colleague's turning zone file states exactly these four keys
    assert set(zone) == {"GRID_MOVEMENT", "MACH_MOTION", "MOTION_ORIGIN", "ROTATION_RATE"}
    assert files["zone_0.cfg"] == "GRID_MOVEMENT= NONE\n"
    assert any(output.endswith("/zone_1.cfg") for output in point.outputs)


def test_a_native_key_of_the_setup_wins_over_a_carried_one(tmp_path):
    workspace(
        tmp_path,
        workflow="unsteady",
        condition="MACH:0.0529, ALPHA:0",
        rotor="",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 20 / LAST_ITERS_AVG: 10",
        setup='TIME_MARCHING = "DUAL_TIME_STEPPING-1ST_ORDER"\n',
    )
    assert deck(tmp_path)[1]["TIME_MARCHING"] == "DUAL_TIME_STEPPING-1ST_ORDER"


def test_a_variable_of_another_run_type_is_warned_and_skipped(tmp_path):
    workspace(tmp_path, workflow="qsteady_rotor", var="DELTA_THETA: 3")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, lines, _ = deck(tmp_path)
    said = [str(w.message) for w in caught]
    assert any(
        "DELTA_THETA is not registered for run type 'qsteady_rotor'. It is read by unsteady, "
        "unsteady_rotor" in text
        for text in said
    )
    assert "DELTA_THETA" not in lines and "TIME_DOMAIN" not in lines
