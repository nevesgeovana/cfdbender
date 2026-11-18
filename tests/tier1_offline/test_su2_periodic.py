"""SYMMETRY, SYMMETRY_LOADS and PERIODIC_COPIES of an SU2 rotor row, as the FlightStream side."""

from __future__ import annotations

import warnings

import pytest
import test_su2_matrix_columns as columns
from test_su2_advance_ratio import FRAME, REFERENCE_BLOCK, build

from cfdbender.core.matrix import MatrixError
from cfdbender.core.plan import PlanError
from cfdbender.workspace.planning import plan_workspace

PERIODIC = 'MARKER_PERIODIC = ["right", "left", 0.0, 0.0, 0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n'
CONTROLS = (
    "PERIODIC_COPIES",
    "SYMMETRY",
    "SYMMETRY_LOADS",
    "symmetry_loads",
    "DELTA_THETA",
    "REVOLUTIONS",
)


def periodic(root, copies="6", setup=FRAME + PERIODIC, symmetry="PERIODIC", extra="", **kwargs):
    var = f"PERIODIC_COPIES: {copies}" if copies is not None else "-"
    if extra:
        var = f"{var} / {extra}" if var != "-" else extra
    return build(root, setup=setup, var=var, symmetry=symmetry, **kwargs)


def test_the_row_controls_never_reach_the_configuration(tmp_path):
    periodic(tmp_path, extra="DELTA_THETA: 5 / REVOLUTIONS: 2", loads="true")
    with pytest.warns(UserWarning) as caught:
        deck = columns.deck(tmp_path)
    assert not set(CONTROLS) & set(deck)
    assert "ROTATION_RATE" in deck and "MARKER_PERIODIC" in deck
    texts = [str(w.message) for w in caught]
    for key in ("DELTA_THETA", "REVOLUTIONS"):
        assert any(f"{key} is skipped" in t for t in texts)
        assert any("time-accurate run type" in t and "does not offer yet" in t for t in texts)


def test_a_consistent_periodic_row_warns_of_nothing(tmp_path):
    periodic(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deck = columns.deck(tmp_path)
    assert not set(CONTROLS) & set(deck)


@pytest.mark.parametrize(
    ("copies", "symmetry", "message"),
    [
        (None, "PERIODIC", "needs PERIODIC_COPIES"),
        ("0", "PERIODIC", "positive whole number"),
        ("-3", "PERIODIC", "positive whole number"),
        ("2.5", "PERIODIC", "positive whole number"),
        ("6", "NONE", "PERIODIC_COPIES needs SYMMETRY: PERIODIC"),
        ("6", "-", "PERIODIC_COPIES needs SYMMETRY: PERIODIC"),
    ],
)
def test_the_symmetry_refusals_at_planning(tmp_path, copies, symmetry, message):
    periodic(tmp_path, copies, symmetry=symmetry)
    with pytest.raises(PlanError, match=message):
        plan_workspace(tmp_path, "dpw")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symmetry": "SIDEWAYS"}, "reads NONE, PERIODIC or -"),
        ({"symmetry": "MIRROR"}, "reads NONE, PERIODIC or -"),
        ({"loads": "maybe"}, "SYMMETRY_LOADS must be true, false or -"),
    ],
)
def test_a_bad_symmetry_cell_is_refused_by_the_matrix(tmp_path, kwargs, message):
    periodic(tmp_path, **kwargs)
    with pytest.raises((MatrixError, PlanError), match=message):
        plan_workspace(tmp_path, "dpw")


def test_the_angle_of_the_periodic_boundary_is_checked_against_the_copies(tmp_path):
    periodic(tmp_path, copies="4")
    with pytest.warns(
        UserWarning, match="60 degrees, which makes 6 copies, and PERIODIC_COPIES is 4"
    ):
        columns.deck(tmp_path)


def test_a_periodic_row_without_a_periodic_boundary_warns(tmp_path):
    periodic(tmp_path, setup=FRAME)
    with pytest.warns(
        UserWarning, match="SYMMETRY is PERIODIC but the setup states no MARKER_PERIODIC"
    ):
        columns.deck(tmp_path)


def test_a_periodic_boundary_without_the_symmetry_warns_that_the_loads_are_the_sectors(tmp_path):
    build(tmp_path, setup=FRAME + PERIODIC)
    with pytest.warns(UserWarning, match="MARKER_PERIODIC but SYMMETRY is not PERIODIC"):
        columns.deck(tmp_path)


def test_a_non_rotor_row_with_a_periodic_boundary_is_left_alone(tmp_path):
    build(tmp_path, "MACH:0.0529, ALPHA:0", setup=PERIODIC, block="")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*legacy mesh folder.*")
        warnings.filterwarnings("ignore", message=".*MACH_NUMBER.*")
        columns.deck(tmp_path)


def test_the_symmetry_cells_are_still_refused_by_solvers_that_do_not_read_them(tmp_path):
    from test_matrix_columns_all_solvers import build as build_other

    build_other(tmp_path, "openfoam")
    matrix = tmp_path / "inputs/matrices/full.fs"
    lines = matrix.read_text(encoding="ascii").splitlines()
    names = [x.strip() for x in lines[0].split("|")]
    cells = lines[2].split("|")
    cells[names.index("SYMMETRY")] = " PERIODIC "
    lines[2] = "|".join(cells)
    matrix.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(PlanError, match="SYMMETRY 'PERIODIC' is not read by solver 'openfoam'"):
        plan_workspace(tmp_path, "full")


def test_a_reference_file_may_carry_the_whole_flightstream_rotor_part(tmp_path):
    block = REFERENCE_BLOCK + '\n[aliases]\nairframe = ["N"]\nrotor_all = ["ROTOR"]\n'
    periodic(tmp_path, block=block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = columns.deck(tmp_path)
    assert "ROTOR" not in deck and "ROTATION_RATE" in deck
