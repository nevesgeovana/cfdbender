"""P0000: the XFOIL adapter exposes its guarded execution path for a keystroke script."""

from __future__ import annotations

import json

import pytest
from qprop_workspace import FAKE_XFOIL

from cfdbender.solvers import load_solver

SCRIPT = (
    "PLOP\nG\n\nLOAD airfoil.dat\n\nPANE\nOPER\nVISC 100000\nMACH 0.100\nITER 150\n"
    "PACC\npolar.txt\n\nASEQ -4 10 0.5\n\nQUIT\n"
)


def test_p0000_xfoil_run_script_is_the_shared_execution_path(tmp_path):
    folder = tmp_path / "work" / "case"
    load_solver("xfoil").run_script(
        tmp_path,
        folder,
        SCRIPT,
        {"airfoil.dat": b"x\n1 0\n0 0\n"},
        {"executable": str(FAKE_XFOIL)},
        lambda: None,
    )
    assert (folder / "xfoil.in").read_text() == SCRIPT
    assert (folder / "airfoil.dat").read_bytes() == b"x\n1 0\n0 0\n"
    assert (folder / "polar.txt").is_file() and (folder / "stderr.log").is_file()
    assert json.loads((folder / "execution.json").read_text())["exit_code"] == 0
    with pytest.raises(ValueError, match="existing"):
        load_solver("xfoil").run_script(
            tmp_path, folder, SCRIPT, {}, {"executable": str(FAKE_XFOIL)}, lambda: None
        )


def test_p0000_xfoil_run_script_refuses_unknown_options_and_checks_ram(tmp_path):
    with pytest.raises(ValueError, match="unknown XFOIL solver options"):
        load_solver("xfoil").run_script(
            tmp_path, tmp_path / "a", SCRIPT, {}, {"bogus": 1}, lambda: None
        )
    calls = []
    load_solver("xfoil").run_script(
        tmp_path, tmp_path / "b", SCRIPT, {"airfoil.dat": b"x\n"},
        {"executable": str(FAKE_XFOIL)}, lambda: calls.append(1),
    )  # fmt: skip
    assert len(calls) >= 3
