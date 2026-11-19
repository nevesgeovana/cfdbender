"""Run-type declarations outrank workspace defaults, but not explicit row inputs."""

from __future__ import annotations

import pytest
from test_runtypes_b1_workflows import deck, workspace


@pytest.mark.parametrize("home", ["defaults", "numerics", "setup", "row"])
def test_run_type_time_marching_has_its_documented_precedence(tmp_path, home):
    workspace(
        tmp_path,
        workflow="unsteady",
        rotor="",
        condition="MACH:0.05",
        var="DELTA_TIME: 0.001 / TIME_ITERATIONS: 10 / LAST_ITERS_AVG: 5"
        + (' / TIME_MARCHING: "TIME_STEPPING"' if home == "row" else ""),
        setup='TIME_MARCHING = "TIME_STEPPING"\n' if home == "setup" else "",
    )
    init = tmp_path / "init.toml"
    init.write_text(init.read_text() + '\n[defaults]\nTIME_MARCHING = "TIME_STEPPING"\n')
    if home == "numerics":
        path = tmp_path / "inputs/numerics/n001.toml"
        path.parent.mkdir(exist_ok=True)
        path.write_text('TIME_MARCHING = "TIME_STEPPING"\n')
        matrix = tmp_path / "inputs/matrices/rt.fs"
        matrix.write_text(
            matrix.read_text()
            .replace("| SETUP |", "| NUMERICS | SETUP |")
            .replace("| rotor.su2 | 001 |", "| rotor.su2 | 001 | 001 |")
        )
    point, values, _ = deck(tmp_path)
    expected = "DUAL_TIME_STEPPING-2ND_ORDER" if home == "defaults" else "TIME_STEPPING"
    assert values["TIME_MARCHING"] == expected
    assert point.origins["TIME_MARCHING"] == ("run type unsteady" if home == "defaults" else home)


@pytest.mark.parametrize("home", ["defaults", "setup"])
def test_sliding_declaration_outranks_defaults(tmp_path, home):
    workspace(
        tmp_path,
        workflow="unsteady_rotor",
        condition="MACH:0.0529, ALPHA:0, RPM:1000",
        var="DELTA_THETA: 3 / REVOLUTIONS: 5 / LAST_REVS_AVG: 2.5",
        setup=('KIND_INTERPOLATION = "NEAREST_NEIGHBOR"\n' if home == "setup" else "")
        + '[sliding]\nzones = 2\nrotating_zone = 1\ninterface = ["X_BOXD1", "X_BOX1"]\n',
    )
    init = tmp_path / "init.toml"
    init.write_text(init.read_text() + '\n[defaults]\nKIND_INTERPOLATION = "NEAREST_NEIGHBOR"\n')
    point, values, _ = deck(tmp_path)
    assert values["KIND_INTERPOLATION"] == (
        "WEIGHTED_AVERAGE" if home == "defaults" else "NEAREST_NEIGHBOR"
    )
    assert point.origins["KIND_INTERPOLATION"] == (
        "run type unsteady_rotor" if home == "defaults" else "setup"
    )
