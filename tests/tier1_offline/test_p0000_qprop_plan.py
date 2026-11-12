"""P0000: init --solver qprop, plan operating points and translate each point.

Usage: cfd-workspace init --solver qprop; cfd-matrix plan study --local.
The s file names the blade (an OBJ under inputs/geometry or a polars-2D set),
n_sections, spacing and cut. Operating points are FLIGHT_CONDITION keys V, RPM or
J, BETA07 or DBETA, RHO, MU, A. Planning is read-only; translate writes nothing.
"""

from __future__ import annotations

import hashlib
import math

import pytest
from qprop_workspace import (
    CONDITION,
    adapter,
    local_reynolds,
    prepared,
    setup_text,
    tree,
    workspace,
)

from cfdbender.core.adapter import load_point
from cfdbender.geometry.blade_slicing import slice_blade
from cfdbender.matrix.cli import main as matrix_cli
from cfdbender.workspace.planning import plan_workspace


def files_of(text):
    """Split a translate text into its named files."""
    parts = text.split("// FILE: ")[1:]
    return {part.split("\n", 1)[0]: part.split("\n", 1)[1] for part in parts}


def test_p0000_qprop_plan_translate_twice_is_byte_identical(tmp_path):
    texts = []
    for name in ("first", "second"):
        root = workspace(tmp_path / name)
        before = tree(root)
        assert matrix_cli(["plan", "study", "--workspace", str(root), "--local"]) == 0
        plan = plan_workspace(root, "study")
        assert len(plan.points) == 1
        text = adapter().translate(load_point(root, plan.points[0], solver="qprop"))
        assert text.isascii()
        assert tree(root) == before, "planning and translating write nothing"
        texts.append(text.encode("ascii"))
    assert texts[0] == texts[1]


def test_p0000_qprop_translate_lists_sections_command_and_xfoil_decks(tmp_path):
    root = workspace(tmp_path)
    files = files_of(adapter().translate(prepared(root)))
    assert files["command"].split() == ["qprop", "propeller.prop", "none", "20", "6000"]
    assert files["qcon.def"].splitlines()[0].split()[0] == "1.2"
    rows = [line.split() for line in files["sections.plan"].splitlines() if line[0] != "#"]
    expected = slice_blade(
        (root / "inputs/geometry/blade.obj").read_text(), n_sections=5, cut="plane"
    )
    assert len(rows) == 5
    for row, section in zip(rows, expected, strict=True):
        assert float(row[1]) == pytest.approx(section.radius, rel=1e-5)
        assert float(row[2]) == pytest.approx(section.chord, rel=1e-5)
        assert float(row[3]) == pytest.approx(section.beta_deg, abs=1e-3)
        re = local_reynolds(1.2, 1.8e-5, 20, 6000, section.radius, section.chord)
        assert float(row[4]) == pytest.approx(re, abs=1.0)
        assert float(row[5]) == pytest.approx(
            math.hypot(20, 2 * math.pi * 100 * section.radius) / 340, abs=1e-3
        )
        assert row[6] == "xfoil"
    for index in range(1, 6):
        deck = files[f"xfoil/sec_{index:02d}/xfoil.in"].splitlines()
        assert deck[:5] == ["PLOP", "G", "", "LOAD airfoil.dat", ""]
        assert f"VISC {float(rows[index - 1][4]):.0f}" in deck
        assert "ASEQ -8 16 1" in deck
        airfoil = files[f"xfoil/sec_{index:02d}/airfoil.dat"].splitlines()
        assert airfoil[0].strip() and len(airfoil) > 20
        assert airfoil[1].split() == ["1.000000", "0.000000"]


def test_p0000_qprop_matrix_j_sets_rpm_from_speed_and_diameter(tmp_path):
    root = workspace(
        tmp_path, condition="V:20,J:sweep,BETA07:30,RHO:1.2,MU:1.8e-5", sweep="0.8,1.0"
    )
    plan = plan_workspace(root, "study")
    commands = [
        files_of(adapter().translate(load_point(root, p, solver="qprop")))["command"].split()
        for p in plan.points
    ]
    # rpm = 60 V / (J D) with D = 2 x 0.2 m
    assert [float(c[4]) for c in commands] == pytest.approx([3750.0, 3000.0])


def test_p0000_qprop_n_sections_can_be_a_matrix_condition(tmp_path):
    root = workspace(tmp_path, condition=CONDITION + ",N_SECTIONS:7")
    rows = files_of(adapter().translate(prepared(root)))["sections.plan"].splitlines()
    assert len([line for line in rows if line[0] != "#"]) == 7


def test_p0000_qprop_plan_tracks_the_blade_geometry_digest(tmp_path):
    root = workspace(tmp_path)
    point = prepared(root)
    blade = root / "inputs/geometry/blade.obj"
    assert (
        point.point.inputs_sha256["inputs/geometry/blade.obj"]
        == hashlib.sha256(blade.read_bytes()).hexdigest()
    )
    blade.write_bytes(blade.read_bytes() + b"# changed\n")
    with pytest.raises(ValueError, match="changed"):
        adapter().translate(point)


def test_p0000_qprop_plan_claims_declared_outputs(tmp_path):
    root = workspace(tmp_path)
    names = {p.rsplit("/", 1)[1] for p in plan_workspace(root, "study").points[0].outputs}
    assert names == {
        "propeller.prop",
        "qcon.def",
        "stdout.log",
        "stderr.log",
        "execution.json",
        "sections.json",
    }


@pytest.mark.parametrize(
    "setup,condition,problem",
    [
        (None, "V:20", "RPM or J"),
        (None, "V:20,RPM:6000,J:0.8", "only one of RPM and J"),
        (None, "V:20,RPM:6000,BETA07:30,DBETA:1", "BETA07 or DBETA"),
        (None, "V:20,RPM:6000,FLAP:1", "unsupported QPROP condition"),
        (None, "V:20,RPM:-5", "RPM"),
        (None, "V:0,J:0.5", "J needs V"),
        (None, "V:20,RPM:6000,N_SECTIONS:2.5", "N_SECTIONS"),
        (setup_text(blade="inputs/geometry/missing.obj"), CONDITION, "missing"),
        (None, CONDITION + ",RHO:0", "RHO"),
    ],
)
def test_p0000_qprop_plan_refuses_bad_conditions(tmp_path, setup, condition, problem):
    root = workspace(tmp_path, setup=setup, condition=condition)
    with pytest.raises(ValueError, match=problem):
        plan_workspace(root, "study")


@pytest.mark.parametrize(
    "overrides,problem",
    [
        ({"solver_2d": "mses"}, "solver_2d.*xfoil"),
        ({"spacing": "log"}, "spacing"),
        ({"cut": "box"}, "cut"),
        ({"rotation": "left"}, "rotation"),
        ({"n_sections": 0}, "n_sections"),
        ({"axis": "q"}, "axis"),
        ({"mystery": 1}, "unsupported QPROP setup: mystery"),
        ({"n_blades": 0}, "n_blades"),
        ({"blade": None}, "blade"),
        ({"blade": "inputs/other/blade.obj"}, "blade"),
    ],
)
def test_p0000_qprop_plan_refuses_bad_setup_keys(tmp_path, overrides, problem):
    root = workspace(tmp_path, setup=setup_text(**overrides))
    with pytest.raises(ValueError, match=problem):
        plan_workspace(root, "study")


def test_p0000_qprop_solver_2d_refusal_names_the_accepted_values(tmp_path):
    root = workspace(tmp_path, setup=setup_text(solver_2d="mses"))
    with pytest.raises(ValueError) as caught:
        plan_workspace(root, "study")
    assert "solver_2d" in str(caught.value) and "xfoil" in str(caught.value)
    assert "mses" in str(caught.value)


def test_p0000_qprop_solver_2d_accepts_xfoil_explicitly(tmp_path):
    root = workspace(tmp_path, setup=setup_text(solver_2d="xfoil"))
    assert len(plan_workspace(root, "study").points) == 1
