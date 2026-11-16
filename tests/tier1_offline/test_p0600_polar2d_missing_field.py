"""P0600: a missing solved-field residual cannot establish convergence."""

from cfdbender.core.convergence import FoamRow, judge_openfoam


def test_p0600_missing_field_does_not_pass_cauchy():
    rows = [FoamRow(0, {"U": 1.0, "p": 1.0}, 1.0, 0.1)]
    rows.extend(FoamRow(n, {"U": 0.0001}, 1.0, 0.1) for n in range(1, 252))
    assert judge_openfoam(rows).criterion is None


def test_p0600_later_field_must_also_reach_target():
    rows = [FoamRow(0, {"U": 1e-7}, 1.0, 0.1), FoamRow(1, {"U": 1e-7, "p": 0.1}, 1.0, 0.1)]
    assert judge_openfoam(rows).criterion is None
