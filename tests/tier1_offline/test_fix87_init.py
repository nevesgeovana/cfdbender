"""Usability fixes that concern what `cfd-workspace init` writes."""

from __future__ import annotations

import pytest

from cfdbender.workspace.cli import main


@pytest.mark.parametrize("solver", ["mock", "su2", "openfoam", "xfoil", "pyfr", "qprop"])
def test_u08_init_writes_inputs_mesh_for_every_solver(tmp_path, solver, capsys):
    root = tmp_path / "w"
    assert main(["init", str(root), "--solver", solver]) == 0
    assert (root / "inputs/mesh").is_dir()
    assert not (root / "inputs/geometries").exists()
    assert "inputs/geometries" not in capsys.readouterr().out
