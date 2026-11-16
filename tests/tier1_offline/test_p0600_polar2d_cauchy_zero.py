"""P0600: a zero outside the Cauchy window cannot poison later windows."""

from cfdbender.core.convergence import cauchy_passes


def test_p0600_cauchy_recovers_after_zero_leaves_window():
    values = [1.0, 0.0] + [1.0] * 251
    assert cauchy_passes(values, 250, 1e-6)[-1]
