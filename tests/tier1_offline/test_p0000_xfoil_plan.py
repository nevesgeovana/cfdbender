"""P0000: init --solver xfoil, plan an alpha sweep, then translate each point.

Usage: cfd-workspace init --solver xfoil; cfd-matrix plan study --local.
Setup: airfoil="NACA 0012", reynolds=1000000. Reference and post may be empty.
Matrix: POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES followed by
1001|1|001|001|001|alpha:sweep|0,2,4. Planning itself remains read-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cfdbender.core.adapter import load_point
from cfdbender.matrix.cli import main as matrix_cli
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.cli import main as workspace_cli
from cfdbender.workspace.planning import plan_workspace

FAKE = Path(__file__).resolve().parents[1] / "fixtures/xfoil/fake_xfoil.py"


def adapter():
    assert "xfoil" in available_solvers(), "P0000 XFOIL must be registered"
    return load_solver("xfoil")


def workspace(root, *, setup=None, condition="alpha:sweep", sweep="0,2,4", options=""):
    assert workspace_cli(["init", "--solver", "xfoil", "--workspace", str(root)]) == 0
    files = {
        "inputs/setups/s001.toml": (
            'airfoil = "NACA 0012"\nreynolds = 1000000\n' if setup is None else setup
        )
        + "\n[solver]\nexecutable = "
        + json.dumps(FAKE.as_posix())
        + "\n"
        + options,
        "inputs/references/r001.toml": "",
        "inputs/pproc/p001.toml": "",
        "inputs/matrices/study.fs": (
            "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
            f"1001|1|001|001|001|{condition}|{sweep}\n"
        ),
    }
    for name, text in files.items():
        (root / name).write_bytes(text.encode("ascii"))
    return root


def prepared(root):
    return load_point(root, plan_workspace(root, "study").points[0], solver="xfoil")


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_p0000_xfoil_init_plan_deterministic_sweep(tmp_path):
    decks = []
    for name in ("first", "second"):
        root = workspace(tmp_path / name)
        assert 'solver = "xfoil"' in (root / "init.toml").read_text()
        before = tree(root)
        assert matrix_cli(["plan", "study", "--workspace", str(root), "--local"]) == 0
        plan = plan_workspace(root, "study")
        assert [p.condition["ALPHA"] for p in plan.points] == [0, 2, 4]
        texts = [adapter().translate(load_point(root, p, solver="xfoil")) for p in plan.points]
        assert all(text.isascii() for text in texts)
        assert all(any(path.endswith("/xfoil.in") for path in p.outputs) for p in plan.points)
        assert tree(root) == before
        decks.append([text.encode("ascii") for text in texts])
    assert decks[0] == decks[1]
    assert decks[0][2] == (
        b"PLOP\nG\n\nNACA 0012\nPANE\nOPER\nVISC 1000000\nMACH 0\nITER 200\n"
        b"PACC\npolar.txt\n\nALFA 4\n\n\nQUIT\n"
    )


@pytest.mark.parametrize("designation", ["NACA 0012", "NACA 23012"])
def test_p0000_xfoil_naca_and_inviscid_defaults(tmp_path, designation):
    workspace(tmp_path, setup=f'airfoil = "{designation}"\n', condition="alpha:4", sweep="")
    text = adapter().translate(prepared(tmp_path))
    assert f"NACA {designation[5:]}\nPANE\nOPER\nMACH 0\nITER 200\n" in text
    assert "VISC" not in text


def test_p0000_xfoil_airfoil_digest_and_panel_transition_options(tmp_path):
    workspace(
        tmp_path,
        setup='airfoil_file = "inputs/airfoils/profile.dat"\n'
        "reynolds = 500000\nmach = 0.1\nncrit = 7\niterations = 80\npanels = 160\n",
        condition="alpha:4",
        sweep="",
    )
    airfoil = tmp_path / "inputs/airfoils/profile.dat"
    airfoil.parent.mkdir()
    airfoil.write_bytes(b"NACA 0012\n1.0 0.0\n0.0 0.0\n1.0 0.0\n")
    point = prepared(tmp_path)
    assert (
        point.point.inputs_sha256["inputs/airfoils/profile.dat"]
        == hashlib.sha256(airfoil.read_bytes()).hexdigest()
    )
    text = adapter().translate(point)
    assert "LOAD airfoil.dat\nPPAR\nN 160\n\n\nPANE\nOPER\n" in text
    assert "VISC 500000\nMACH 0.1\nITER 80\nVPAR\nN 7\n\nPACC" in text
    airfoil.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        adapter().translate(point)


@pytest.mark.parametrize(
    "setup,condition,sweep,problem",
    [
        ('airfoil="NACA 123"', "alpha:4", "", "airfoil"),
        ('airfoil="NACA 0012"\nairfoil_file="inputs/airfoils/a.dat"', "alpha:4", "", "one"),
        ('airfoil="NACA 0012"\nreynolds=-1', "alpha:4", "", "reynolds"),
        ('airfoil="NACA 0012"\nmach=nan', "alpha:4", "", "mach"),
        ('airfoil="NACA 0012"\nncrit=true', "alpha:4", "", "ncrit"),
        ('airfoil="NACA 0012"\niterations=1.5', "alpha:4", "", "iterations"),
        ('airfoil="NACA 0012"\npanels=0', "alpha:4", "", "panels"),
        ('airfoil="NACA 0012"', "mach:sweep", "0,0.1", "alpha"),
        ('airfoil="NACA 0012"', "", "", "alpha"),
    ],
)
def test_p0000_xfoil_invalid_inputs_refused_before_writes(
    tmp_path, setup, condition, sweep, problem
):
    workspace(tmp_path, setup=setup + "\n", condition=condition, sweep=sweep)
    before = tree(tmp_path)
    with pytest.raises(ValueError, match=problem):
        plan_workspace(tmp_path, "study")
    assert tree(tmp_path) == before
