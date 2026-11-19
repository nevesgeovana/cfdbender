"""Every reported unsteady coefficient uses the averaging window."""

from __future__ import annotations

import pytest
from test_runtypes_b1_workflows import workspace

from cfdbender.core.adapter import load_point
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("body_loads", [False, True])
def test_lift_and_drag_are_window_means(tmp_path, body_loads):
    workspace(
        tmp_path,
        workflow="unsteady",
        rotor="",
        condition="MACH:0.05",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 4 / LAST_ITERS_AVG: 2",
    )
    plan = plan_workspace(tmp_path, "rt")
    point = load_point(tmp_path, plan.points[0], solver="su2")
    solver = load_solver("su2")
    folder = tmp_path / point.point.directory
    folder.mkdir(parents=True)
    (folder / "su2.cfg").write_text(solver.translate(point), encoding="ascii")
    header = "Time_Iter,Inner_Iter,rms[Rho],CL,CD"
    if body_loads:
        header += ",CFx,CFy,CFz,CMx,CMy,CMz"
    lines = [header]
    for step in range(4):
        lines.append(f"{step},0,-6,{step},{step / 10}" + (",1,2,3,4,5,6" if body_loads else ""))
    (folder / "history.csv").write_text("\n".join(lines) + "\n", encoding="ascii")
    data = solver.collect(point).to_dict()
    assert data["coefficients"]["CL"]["value"] == pytest.approx(2.5)
    assert data["coefficients"]["CD"]["value"] == pytest.approx(0.25)
    assert data["time_average"]["means"]["CL"] == pytest.approx(2.5)
