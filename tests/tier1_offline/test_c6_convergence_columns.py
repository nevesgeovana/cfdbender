"""OpenFOAM polar convergence reads named columns and the integrated force set."""

import math

import pytest

from cfdbender.core.convergence import read_openfoam


@pytest.mark.parametrize(
    "header, row", [("Cl Time Cd", "0.8 7 0.02"), ("Cd Cl Time", "0.02 0.8 7")]
)
def test_force_time_can_be_any_column(tmp_path, header, row):
    (tmp_path / "log.simpleFoam").write_text(
        "Time = 7\nSolving for p, Initial residual = 0.001, Final residual = 0\n"
    )
    table = tmp_path / "postProcessing/forceCoeffs/0/coefficient.dat"
    table.parent.mkdir(parents=True)
    table.write_text(f"# {header}\n{row}\n")
    measured = read_openfoam(tmp_path)
    assert measured[0].lift == 0.8
    assert measured[0].drag == 0.02


def test_group_cannot_substitute_for_missing_total(tmp_path):
    (tmp_path / "log.simpleFoam").write_text(
        "Time = 7\nSolving for p, Initial residual = 0.001, Final residual = 0\n"
    )
    table = tmp_path / "postProcessing/forceCoeffs_wing/0/coefficient.dat"
    table.parent.mkdir(parents=True)
    table.write_text("# Time Cl Cd\n7 0.8 0.02\n")
    with pytest.raises(ValueError, match="forceCoeffs"):
        read_openfoam(tmp_path)


def test_missing_force_sample_stays_unknown(tmp_path):
    (tmp_path / "log.simpleFoam").write_text(
        "Time = 7\nSolving for p, Initial residual = 0.001, Final residual = 0\n"
    )
    table = tmp_path / "postProcessing/forceCoeffs/0/coefficient.dat"
    table.parent.mkdir(parents=True)
    table.write_text("# Time Cl Cd\n6 0.8 0.02\n")
    assert math.isnan(read_openfoam(tmp_path)[0].lift)
