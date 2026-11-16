"""P060: explicit row conditions take precedence over fallback setup values."""

import pytest
import test_p0000_item3d_config as su2_case
import test_p0000_openfoam_adapter as foam_case
import test_p0000_xfoil_plan as xfoil_case

from cfdbender.core.adapter import load_point
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("solver", ["su2", "openfoam", "xfoil"])
def test_p060_row_conditions_replace_defaults_without_hiding_unknown_keys(tmp_path, solver):
    if solver == "su2":
        su2_case.workspace(tmp_path)
        defaults = "mach = 0.1\nalpha = 0\n"
    elif solver == "openfoam":
        foam_case.case(tmp_path, sweep="4")
        setup = tmp_path / "inputs/setups/s001.toml"
        setup.write_text(setup.read_text().replace("speed = 10.0\nnu = 1.5e-5\n", ""))
        matrix = tmp_path / "inputs/matrices/study.fs"
        matrix.write_text(matrix.read_text().replace("ALPHA:sweep", "MACH:0.1,REmi:1,ALPHA:sweep"))
        defaults = "speed = 20\nnu = 1.5e-5\n"
    else:
        xfoil_case.workspace(tmp_path, condition="MACH:0.3,ALPHA:4", sweep="-")
        defaults = "mach = 0.1\n"
    init = tmp_path / "init.toml"
    init.write_text(init.read_text() + "\n[defaults]\n" + defaults)
    try:
        plan = plan_workspace(tmp_path, "study")
    except ValueError as error:
        pytest.fail(f"the explicit row must override setup defaults: {error}")
    context = load_point(tmp_path, plan.points[0], solver=solver)
    assert load_solver(solver).translate(context)
    assert getattr(plan.points[0], "defaults_omitted", ())
