"""A time-accurate point reports its window means; its rotor row is formed from the mean loads."""

from __future__ import annotations

import math
import warnings

import pytest
from test_runtypes_b1_workflows import workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.plan import workspace_path
from cfdbender.matrix.exports import write_tables
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

STEPS = 36  # three revolutions of twelve steps
VAR = "DELTA_THETA: 30 / REVOLUTIONS: 3 / LAST_REVS_AVG: 1"


def loads(step, drift=0.0):
    """CFx, CFy, CFz, CMx, CMy, CMz at a step: a periodic part, and a drift per revolution."""
    phase = 2.0 * math.pi * step / 12.0
    grow = 1.0 + drift * ((step - 1) // 12)
    return (
        -0.05 * grow + 0.01 * math.cos(phase),
        0.002 * math.sin(phase),
        0.003 * math.cos(phase),
        0.004 * grow + 0.001 * math.sin(phase),
        0.0,
        0.0,
    )


def history(last=STEPS, drift=0.0):
    lines = ["Time_Iter,Outer_Iter,Inner_Iter,rms[Rho],CFx,CFy,CFz,CMx,CMy,CMz"]
    for step in range(1, last + 1):
        for inner in (0, 1):  # two inner iterations; the last row of a step is its state
            values = loads(step, drift) if inner else (9.0,) * 6
            lines.append(
                f"{step - 1},0,{inner},-6.0," + ",".join(format(v, ".17g") for v in values)
            )
    return "\n".join(lines) + "\n"


def collected(root, last=STEPS, drift=0.0):
    workspace(
        root,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var=VAR,
    )
    plan = plan_workspace(root, "rt")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    solver = load_solver("su2")
    (folder / "su2.cfg").write_text(solver.translate(point), encoding="ascii")
    (folder / "history.csv").write_text(history(last, drift), encoding="ascii")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solver.collect(point)
    return result, plan, [str(w.message) for w in caught]


def test_the_result_is_the_mean_over_the_window_and_the_rotor_row_is_of_the_mean(tmp_path):
    result, _, _ = collected(tmp_path)
    data = result.to_dict()
    averaged = data["time_average"]
    assert (averaged["FIRST_STEP"], averaged["LAST_STEP"], averaged["STEPS"]) == (25, 36, 12)
    expected = [sum(loads(s)[i] for s in range(25, 37)) / 12 for i in range(6)]
    for name, value in zip(("CFx", "CFy", "CFz", "CMx", "CMy", "CMz"), expected, strict=True):
        assert averaged["means"][name] == pytest.approx(value, rel=1e-12, abs=1e-15)
        assert data["coefficients"][name]["value"] == pytest.approx(value, rel=1e-12, abs=1e-15)
    row = data["rotor"]["row"]
    assert row["ETA"] == pytest.approx(row["J"] * row["CT"] / row["CP"], rel=1e-12)
    assert data["rotor"]["window"] == {"FIRST_STEP": 25, "LAST_STEP": 36, "STEPS": 12}
    # the mean force is periodic-free: CFx's mean over a whole revolution is its steady part
    q_area = data["rotor"]["loads"]["force_n"][0] / expected[0]
    assert data["rotor"]["loads"]["force_n"][0] == pytest.approx(-0.05 * q_area, rel=1e-9)


def test_one_row_per_complete_revolution_with_its_drift(tmp_path):
    result, _, said = collected(tmp_path, drift=0.05)
    table = result.to_dict()["time_average"]["per_revolution"]
    assert [(r["REVOLUTION"], r["FIRST_STEP"], r["LAST_STEP"]) for r in table] == [
        (1, 1, 12),
        (2, 13, 24),
        (3, 25, 36),
    ]
    assert table[0]["CFx_DRIFT_PCT"] == "NA"
    assert table[2]["CFx_DRIFT_PCT"] == pytest.approx(-100 * (1.10 / 1.05 - 1), rel=1e-9)
    assert any("CFx" in text for text in said), "a five per cent drift is warned"


def test_a_history_that_does_not_cover_the_window_is_left_out_with_the_reason(tmp_path):
    result, _, _ = collected(tmp_path, last=30)
    data = result.to_dict()
    assert "steps 25 to 36" in data["time_average"]["skipped"]
    assert "row" not in data["rotor"] and "steps 25 to 36" in data["rotor"]["skipped"]


def test_the_unsteady_polar_and_the_rotor_table_carry_the_window(tmp_path):
    result, plan, _ = collected(tmp_path)
    point = plan.points[0]
    rows = [
        {
            "point_id": point.point_id,
            "pol": point.pol,
            "condition": dict(point.condition),
            "status": "COMPLETED",
            "result": result.to_dict(),
        }
    ]
    paths = write_tables(tmp_path, "inputs/matrices/rt.fs", rows, "su2")
    polar = next(p for p in paths if p.name == "P7001_rt_uns_avg.csv")
    header, line = polar.read_text(encoding="ascii").splitlines()
    names = header.split(",")
    assert names[:5] == ["POL", "FIRST_STEP", "LAST_STEP", "STEPS", "ALPHA"]
    assert "BREF" not in names and "MESH_FACES" not in names
    assert names[-2:] == ["RPM_ROTOR", "DIAMETER_ROTOR"]
    cells = dict(zip(names, line.split(","), strict=True))
    assert (cells["FIRST_STEP"], cells["LAST_STEP"], cells["STEPS"]) == ("25", "36", "12")
    assert any(p.name == "P7001-ROTOR_rotor.csv" for p in paths)
