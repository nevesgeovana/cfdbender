"""SU2INV: a free-stream-scaled SU2 deck converges to rms -12 unless the user states a level.

SU2's own NACA 0012 case states CONV_RESIDUAL_MINVAL -12 with FREESTREAM_PRESS_EQ_ONE; SU2's
built-in default is -8. cfdbender classifies a point as converged only when SU2's log says the
criteria were satisfied, so the deck's CONV_RESIDUAL_MINVAL is the classifier's threshold.
"""

from __future__ import annotations

import pytest
import test_p0000_item3d_config as su2_case

from cfdbender.core.results import ExecutionResult, Iteration
from cfdbender.solvers.su2 import classify


def _lines(root, extra=""):
    su2_case.workspace(root)
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text(encoding="utf-8") + extra + "\n", encoding="utf-8")
    return su2_case.api().translate(su2_case.prepared(root)).splitlines()


def test_scaled_deck_defaults_to_minus_twelve(tmp_path):
    lines = _lines(tmp_path)
    assert "REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE" in lines
    assert "CONV_RESIDUAL_MINVAL= -12" in lines


@pytest.mark.parametrize(
    "extra,line",
    [
        ("convergence_residual = -9", "CONV_RESIDUAL_MINVAL= -9"),
        ("CONV_RESIDUAL_MINVAL = -10", "CONV_RESIDUAL_MINVAL= -10"),
    ],
)
def test_stated_level_wins(tmp_path, extra, line):
    lines = _lines(tmp_path, extra)
    assert line in lines
    assert sum(item.startswith("CONV_RESIDUAL_MINVAL=") for item in lines) == 1


def test_dimensional_deck_keeps_su2_default(tmp_path):
    lines = _lines(tmp_path, 'REF_DIMENSIONALIZATION = "DIMENSIONAL"')
    assert not any(item.startswith("CONV_RESIDUAL_MINVAL=") for item in lines)


def _result(converged, iteration, limit=100):
    row = Iteration(iteration=iteration, residual=1e-11, converged=converged)
    return ExecutionResult(
        exit_code=0, history=(row,), iteration_limit=limit, solver_version="8.6.0"
    )


def test_classifier_threshold_is_su2s_verdict():
    assert classify(_result(True, 50)) == "converged"
    assert classify(_result(False, 100)) == "iteration_limit"
