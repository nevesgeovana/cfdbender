"""P0000: shared helpers retain numeric, walltime and RAM safety semantics."""

import importlib
import math
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("solver", ["su2", "xfoil", "openfoam", "pyfr"])
def test_p0000_solver_host_helpers_are_neutral(solver):
    config = importlib.import_module(f"cfdbender.solvers.{solver}._config")
    execution = importlib.import_module(f"cfdbender.solvers.{solver}._execute")
    for helper in (
        config.number,
        execution._ram.require_ram,
        execution._timeout,
        execution._kill_tree,
        execution.process_lease,
    ):
        assert helper.__module__.startswith("cfdbender.solvers._host.")


@pytest.mark.parametrize("value", [True, False, "1", None, math.inf, -math.inf, math.nan])
def test_p0000_solver_host_number_refuses_nonfinite_and_nonnumeric(value):
    from cfdbender.solvers.su2._config import number

    with pytest.raises(ValueError, match="alpha must be a finite number"):
        number(value, "alpha")


@pytest.mark.parametrize(
    "walltime,expected", [(None, 600), ("2s", 2), ("2m", 120), ("2h", 600), ("2d", 600)]
)
def test_p0000_solver_host_timeout_bounds(walltime, expected):
    from cfdbender.solvers.su2._execute import _timeout

    point = SimpleNamespace(point=SimpleNamespace(walltime=walltime))
    assert _timeout(point, {"timeout_s": 600}) == expected


@pytest.mark.parametrize(
    "estimate,free,accept",
    [
        (0.25, 0.75, True),
        (0, 0.5, True),
        (0.25, 0.749, False),
        (0, 0.499, False),
        (-1, 2, False),
        (math.inf, 2, False),
        (0, math.nan, False),
    ],
)
def test_p0000_solver_host_ram_floor_and_estimate(estimate, free, accept):
    from cfdbender.solvers.su2._ram import require_ram

    if accept:
        assert require_ram(estimate, measure=lambda: free) is None
    else:
        with pytest.raises(ValueError, match="0.5 GiB floor"):
            require_ram(estimate, measure=lambda: free)
