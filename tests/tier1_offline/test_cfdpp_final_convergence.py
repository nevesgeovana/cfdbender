"""A transient convergence crossing must not certify a later divergent state."""

import pytest
from cfdpp_support import builder

from cfdbender.solvers.cfdpp import _history


@pytest.mark.parametrize("criterion", ["R", "C"])
def test_convergence_must_hold_at_the_last_step(criterion: str) -> None:
    residuals = [1.0, 1e-8, 1.0] if criterion == "R" else [1.0] + [1e-4] * 300 + [1.0]
    series = {"CL": [0.5] * len(residuals), "CD": [0.01] * len(residuals)}
    evidence = _history.build_evidence(
        builder().info0_text(residuals),
        None,
        "",
        exit_code=0,
        iteration_limit=1000,
        convergence_orders=6.0,
        series=series,
    )
    assert _history.classify(evidence) == "failed"
    assert evidence.criterion == "none"
