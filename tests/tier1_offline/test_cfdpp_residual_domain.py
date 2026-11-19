"""Invalid residuals fail, and a zero initial residual does not crash collection."""

import pytest
from cfdpp_support import builder

from cfdbender.solvers.cfdpp import _history


@pytest.mark.parametrize("residuals", [[1.0, -1.0], [-1.0]])
def test_negative_residual_is_failed_evidence(residuals: list[float]) -> None:
    evidence = _history.build_evidence(
        builder().info0_text(residuals),
        None,
        "",
        exit_code=0,
        iteration_limit=1000,
        convergence_orders=6.0,
    )
    assert _history.classify(evidence) == "failed"
    assert "non-negative" in evidence.problem


def test_zero_initial_residual_followed_by_positive_is_not_converged() -> None:
    evidence = _history.build_evidence(
        builder().info0_text([0.0, 1.0]),
        None,
        "",
        exit_code=0,
        iteration_limit=1000,
        convergence_orders=6.0,
        series={"CL": [0.5, 0.5], "CD": [0.01, 0.01]},
    )
    assert _history.classify(evidence) == "failed"


def test_extreme_finite_residual_growth_does_not_crash_cauchy_judges() -> None:
    from cfdbender.core.convergence import StepRow, judge_cfdpp

    residuals = [1e-300, 1e300]
    evidence = _history.build_evidence(
        builder().info0_text(residuals),
        None,
        "",
        exit_code=0,
        iteration_limit=1000,
        convergence_orders=6.0,
        series={"CL": [0.5, 0.5], "CD": [0.01, 0.01]},
    )
    assert _history.classify(evidence) == "failed"
    verdict = judge_cfdpp(
        [StepRow(1, residuals[0], 0.5, 0.01), StepRow(2, residuals[1], 0.5, 0.01)],
        orders=6.0,
        elems=1,
    )
    assert not verdict.converged
