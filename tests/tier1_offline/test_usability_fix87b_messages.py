"""U-17 and U-30: refusals name what the user wrote, in one wording across solvers."""

from __future__ import annotations

from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, run


def plan_error(capsys, root):
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 2
    return err


def test_u17_su2_names_what_the_user_wrote_not_internal_keys(tmp_path, capsys):
    root = build(tmp_path / "su2", "su2")
    setup = root / "inputs/setups/s001.toml"
    reference = root / "inputs/references/r001.toml"
    matrix = root / "inputs/matrices/full.fs"
    edit(setup, "iterations = 40\n", "")
    assert "inputs/setups/s001.toml needs iterations" in plan_error(capsys, root)
    edit(setup, "SOLVER", "iterations = 40\nSOLVER")
    edit(reference, "area = 1.0\n", "")
    assert "inputs/references/r001.toml needs area" in plan_error(capsys, root)
    edit(reference, "length = 1.0", "area = 1.0\nlength = 1.0")
    edit(matrix, "| model.su2 |", "| - |")
    err = plan_error(capsys, root)
    assert "GEOMETRY names no mesh and inputs/setups/s001.toml states none" in err
    edit(matrix, "| - | 001 | 001 | 001 |", "| model.su2 | 001 | 001 | 001 |")
    edit(matrix, "MACH:0.5", "MACH:0")
    err = plan_error(capsys, root)
    assert "REYNOLDS_NUMBER" not in err and "FLIGHT_CONDITION" in err


def test_u17_openfoam_and_pyfr_say_absent_not_non_finite(tmp_path, capsys):
    root = build(tmp_path / "of", "openfoam")
    edit(root / "inputs/matrices/full.fs", "TASmps:10, REmi:0.6, ", "")
    err = plan_error(capsys, root)
    assert "FLIGHT_CONDITION needs MACH or TASmps" in err and "finite" not in err
    root = build(tmp_path / "pf", "pyfr")
    edit(root / "inputs/setups/s001.toml", "order = 3\n", "")
    err = plan_error(capsys, root)
    assert "inputs/setups/s001.toml needs order" in err and "finite" not in err


def test_u30_wording_is_for_the_user_not_for_the_adapter(tmp_path, capsys):
    root = build(tmp_path / "of", "openfoam")
    edit(
        root / "inputs/setups/s001.toml", 'application = "simpleFoam"', 'application = "pimpleFoam"'
    )
    err = plan_error(capsys, root)
    assert "POC" not in err and "inputs/setups/s001.toml" in err
    root = build(tmp_path / "su2", "su2")
    matrix = root / "inputs/matrices/full.fs"
    edit(matrix, "| 1.0 | - |", "| 1.0 | unsteady |")
    err = plan_error(capsys, root)
    assert "in this adapter" not in err and "WORKFLOW" in err
    root = build(tmp_path / "pf", "pyfr")
    edit(root / "inputs/setups/s001.toml", "order = 3\n", "order = 3\niteratons = 3\n")
    err = plan_error(capsys, root)
    assert "['" not in err and "iteratons" in err and "inputs/setups/s001.toml" in err
    root = build(tmp_path / "su2b", "su2")
    edit(root / "inputs/setups/s001.toml", "iterations = 40", "iteratons = 40\niterations = 40")
    err = plan_error(capsys, root)
    assert "iteratons" in err and "inputs/setups/s001.toml" in err
    assert "uppercase" not in err or "closest" in err
