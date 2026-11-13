"""Adapter contract C-12: a setup value that the matrix row also determines must agree with it.

SU2 is covered by ``test_su2_matrix_columns.py::test_su2_setup_and_matrix_must_agree``; the
same refusal for OpenFOAM and PyFR is asserted here, with the matching value accepted.
"""

from __future__ import annotations

import pytest
from test_matrix_columns_all_solvers import build

from cfdbender.core.adapter import load_point
from cfdbender.workspace.planning import plan_workspace


def _context(tmp_path, solver):
    root = build(tmp_path, solver)
    return load_point(root, plan_workspace(root, "full").points[0], solver=solver)


def _openfoam(context, setup):
    from cfdbender.solvers.openfoam._config import config

    return config(
        setup, context.reference, context.post, context.point.condition, geometry=context.geometry
    )


def _pyfr(context, setup):
    from cfdbender.solvers.pyfr._config import config

    return config(setup, context.point.condition, context.reference, context.geometry)


CASES = [
    ("openfoam", _openfoam, {"speed": 99.0}, "speed"),
    ("openfoam", _openfoam, {"mesh": "inputs/mesh/other.msh"}, "mesh"),
    ("pyfr", _pyfr, {"mach": 0.9}, "mach"),
    ("pyfr", _pyfr, {"mesh": "inputs/mesh/other.msh"}, "mesh"),
]


@pytest.mark.parametrize(("solver", "build_case", "stated", "key"), CASES)
def test_c12_a_setup_value_that_disagrees_with_the_row_is_refused(
    tmp_path, solver, build_case, stated, key
):
    context = _context(tmp_path, solver)
    with pytest.raises(ValueError, match=f"conflict: {key} is"):
        build_case(context, {**context.setup, **stated})


@pytest.mark.parametrize("solver", ["openfoam", "pyfr"])
def test_c12_a_setup_value_that_agrees_with_the_row_is_accepted(tmp_path, solver):
    context = _context(tmp_path, solver)
    build_case = {"openfoam": _openfoam, "pyfr": _pyfr}[solver]
    case = build_case(context, dict(context.setup))
    stated = {"speed": case.speed} if solver == "openfoam" else {"mach": case.mach}
    assert build_case(context, {**context.setup, **stated}) == case
