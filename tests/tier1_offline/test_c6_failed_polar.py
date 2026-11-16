"""A residual target cannot turn a failed solver execution into a usable polar row."""

import pytest

from cfdbender.core.convergence import Verdict
from cfdbender.matrix.polar2d import _reason


@pytest.mark.parametrize("classification", ["failed", "converged", "iteration_limit"])
def test_polar_respects_failed_execution(classification):
    row = {"status": "COMPLETED", "result": {"classification": classification}}
    reason = _reason(row, Verdict("R", 1), {"CL": 0.5, "CD": 0.01, "CM": 0.02})
    if classification == "failed":
        assert reason is not None and "failed" in reason
    else:
        assert reason is None
