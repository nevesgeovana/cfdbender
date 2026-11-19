"""The typed product declarations of the adapter contract; SU2's and the mock's."""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import test_su2_matrix_columns as columns
from test_p0000_item3c_adapter import workspace as mock_workspace

from cfdbender.core.adapter import load_point
from cfdbender.core.contract import (
    SURFACE,
    VOLUME,
    FreeStream,
    ProductSupport,
    product_support,
)
from cfdbender.core.plan import plan_matrix, workspace_path
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

SECTIONS = '[[sections.distributions]]\nfamilies = "wing"\nplanes = ["XZ"]\n'
PROBES = '[[probes]]\nparameters = ["VX"]\n[[probes.lines]]\nstart = [0, 0, 0]\nend = [1, 0, 0]\n'
VOLUME_SECTION = (
    '[volume_section]\nshape = "circle"\nplane = "YZ"\nradii_m = [0.0, 1.0]\npoints = [2, 4]\n'
)
BL = {"Theta", "Delta*", "Delta", "H"}


def test_su2_declares_the_three_products_and_what_it_cannot_fill():
    declared = product_support(load_solver("su2"))
    assert set(declared) == {"sections", "volume_section", "probes"}
    assert declared["sections"].source == SURFACE
    assert set(declared["sections"].unfillable) == BL
    assert {"pressure", "Mach", "vx", "vy", "vz", "Cp_ref", "CF"} <= set(
        declared["sections"].columns
    )
    assert declared["volume_section"].source == VOLUME
    assert set(declared["probes"].unfillable) == {
        "BL_MOMENTUM_THICKNESS",
        "BL_DISPLACEMENT_THICKNESS",
        "BL_TOTAL_THICKNESS",
        "BL_SHAPE_FACTOR",
        "BL_SKIN_FRICTION",
        "BL_TRANSITION_MARKER",
    }


def test_an_adapter_that_declares_nothing_supplies_nothing():
    assert product_support(load_solver("openfoam")) == {}


@pytest.mark.parametrize(
    "products",
    [
        {"slices": ProductSupport(SURFACE, {})},
        {"sections": ProductSupport("disk", {})},
        {"sections": ProductSupport(SURFACE, {"H": "h"}, {"H": "none"})},
        {"sections": {"source": "surface"}},
    ],
)
def test_a_malformed_declaration_is_a_developer_error(products):
    with pytest.raises(ValueError, match="PRODUCTS"):
        product_support(SimpleNamespace(PRODUCTS=products, __name__="fake"))


def test_su2_velocity_columns_come_from_momentum_over_density():
    free = FreeStream(100.0, 50.0, 1.0, 10.0, "test")
    fields = {
        "Momentum": np.array([[2.0, 4.0, 4.0]]),
        "Density": np.array([2.0]),
        "Pressure": np.array([150.0]),
        "Skin_Friction_Coefficient": np.array([[3.0, 4.0, 0.0]]),
    }
    sections = product_support(load_solver("su2"))["sections"].columns
    probes = product_support(load_solver("su2"))["probes"].columns
    assert [float(sections[c](fields, free)[0]) for c in ("vx", "vy", "vz")] == [1.0, 2.0, 2.0]
    assert float(sections["CF"](fields, free)[0]) == 5.0
    assert float(probes["VELOCITY"](fields, free)[0]) == 3.0
    assert float(probes["CP_FREE"](fields, free)[0]) == 1.0
    assert float(probes["STATIC_PRESSURE_RATIO"](fields, free)[0]) == 1.5


def _su2(root, pproc="", setup=""):
    columns.workspace(root, setup=setup)
    (root / "inputs/pproc/p001.toml").write_text("history = true\n" + pproc, encoding="ascii")
    return root


def test_su2_writes_no_output_list_when_no_product_is_declared(tmp_path):
    assert "OUTPUT_FILES" not in columns.deck(_su2(tmp_path))


@pytest.mark.parametrize(
    ("pproc", "setup", "expected"),
    [
        (SECTIONS, "", "( RESTART, PARAVIEW, SURFACE_PARAVIEW )"),
        (PROBES, 'OUTPUT_FILES = ["RESTART"]', "( RESTART, PARAVIEW )"),
        (
            VOLUME_SECTION,
            'OUTPUT_FILES = ["RESTART", "SURFACE_CSV"]',
            "( RESTART, SURFACE_CSV, PARAVIEW )",
        ),
        (
            SECTIONS + PROBES,
            'OUTPUT_FILES = ["RESTART"]',
            "( RESTART, SURFACE_PARAVIEW, PARAVIEW )",
        ),
    ],
)
def test_su2_asks_for_the_files_a_declared_product_reads(tmp_path, pproc, setup, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deck = columns.deck(_su2(tmp_path, pproc, setup))
    assert deck["OUTPUT_FILES"] == expected
    assert not [w for w in caught if "is skipped" in str(w.message)]


def test_su2_product_inputs_of_a_steady_point(tmp_path):
    root = _su2(tmp_path, SECTIONS + PROBES, 'MARKER_EULER = ["wing", "tail"]')
    plan = plan_workspace(root, "dpw")
    point = load_point(root, plan.points[0], solver="su2")
    folder = workspace_path(root, plan.points[0].directory)
    folder.mkdir(parents=True)
    (folder / "surface_flow.vtu").write_text("x", encoding="ascii")
    found = load_solver("su2").product_inputs(point)
    assert [f.path for f in found.surface] == [f"{plan.points[0].directory}/surface_flow.vtu"]
    assert found.surface[0].step is None and found.volume == ()
    assert found.missing == ((f"{plan.points[0].directory}/flow.vtu", "the run did not write it"),)
    assert found.families == ("wing", "tail") and not found.families_separable
    free = found.free_stream
    assert free is not None
    assert math.isclose(free.q_ref, 0.5 * free.density * free.speed**2)
    deck = columns.deck(root)
    temperature, gas = (float(deck[k]) for k in ("FREESTREAM_TEMPERATURE", "GAS_CONSTANT"))
    assert math.isclose(free.p_inf, free.density * gas * temperature, rel_tol=1e-12)
    assert found.notes["velocity_frame"] == "absolute"


def test_the_mock_declares_the_products_and_writes_their_files_only_when_asked(tmp_path):
    mock_workspace(tmp_path)
    mock = load_solver("mock")
    plain = plan_matrix(tmp_path, "study", validators={"mock": mock.validate_row})
    point = load_point(tmp_path, plain.points[0], solver="mock")
    mock.execute(point, check_ram=lambda: None)
    folder = workspace_path(tmp_path, plain.points[0].directory)
    assert not (folder / "surface.vtu").exists() and not (folder / "volume.vtu").exists()

    other = tmp_path / "declared"
    mock_workspace(other)
    (other / "inputs/pproc/p001.toml").write_text("history = true\n" + SECTIONS + PROBES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = plan_matrix(other, "study", validators={"mock": mock.validate_row})
    assert not [w for w in caught if "skipped" in str(w.message)]
    point = load_point(other, plan.points[0], solver="mock")
    mock.execute(point, check_ram=lambda: None)
    found = mock.product_inputs(point)
    assert [s.path.rsplit("/", 1)[1] for s in (*found.surface, *found.volume)] == [
        "surface.vtu",
        "volume.vtu",
    ]
    assert found.families == ("wing",) and found.free_stream is not None
