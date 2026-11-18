"""U-31 and U-32: quiet executable route, what a solver assumes, and matrix header messages."""

from __future__ import annotations

import pytest
from test_conformance_workspace import executables, state_no_build
from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, plan_error_text, run


def standin(tmp_path, solver):
    root = build(tmp_path / solver, solver)
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes() + executables(root, solver).encode("ascii"))
    state_no_build(root)
    return root


def test_u31_a_stated_executable_selects_the_executable_route_without_a_warning(tmp_path, capsys):
    root = standin(tmp_path, "su2")
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    code, out, err = run(capsys, "run", "full", "--workspace", root, "--local")
    assert code == 0
    assert "pysu2" not in err and "Warnings" not in err


def test_u31_console_lines_keep_their_order_when_streams_are_captured(tmp_path, capsys):
    root = standin(tmp_path, "su2")
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    _, out, _ = run(capsys, "run", "full", "--workspace", root, "--local")
    assert out.splitlines()[0].startswith("[1/2]")


def test_u32_xfoil_designation_must_be_a_real_naca_five_digit(tmp_path, capsys):
    root = build(tmp_path / "xfoil", "xfoil")
    edit(root / "inputs/setups/s001.toml", "NACA 0012", "NACA 00123")
    err = plan_error_text(capsys, root)
    assert "NACA" in err and "inputs/setups/s001.toml" in err


def test_u32_a_row_stating_only_alpha_warns_what_the_solver_assumes(tmp_path, capsys):
    root = build(tmp_path / "su2", "su2")
    edit(
        root / "inputs/matrices/full.fs",
        "MACH:0.5, ALTFT:10000, ALPHA:sweep, BETA:0",
        "ALPHA:sweep",
    )
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 0 and "state no MACH_NUMBER" in err and "SU2" in err
    root = build(tmp_path / "xfoil", "xfoil")
    edit(root / "inputs/matrices/full.fs", "MACH:0.1, REmi:1, ALPHA:sweep", "ALPHA:sweep")
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 0 and "inviscid" in err


def test_u32_the_table_states_sea_level_where_it_was_assumed(tmp_path, capsys):
    root = standin(tmp_path, "mock")
    pass
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    assert run(capsys, "run", "full", "--workspace", root, "--local")[0] == 0
    assert run(capsys, "collect", "full", "--workspace", root)[0] == 0
    table = next((root / "post/full/polars").glob("FAM-ACOEF-P*.csv"))
    header, first = table.read_text().splitlines()[:2]
    cells = dict(zip(header.split(","), first.split(","), strict=True))
    assert cells["ALT"] == "0.0"


@pytest.mark.parametrize(
    ("old", "new", "expected", "absent"),
    [
        (" | ", ",", "columns are separated by |", "(none)"),
        (
            "| SETUP |",
            "| SETT |",
            "SETT (closest known column: SETUP)",
            "unsupported columns: (none)",
        ),
        ("| FS_BUILD |", "| FS_BUILD | HIDDEN2 |", "HIDDEN2", "missing columns: (none)"),
    ],
)
def test_u32_matrix_header_messages(tmp_path, capsys, old, new, expected, absent):
    root = build(tmp_path / "mock", "mock")
    matrix = root / "inputs/matrices/full.fs"
    header, rest = matrix.read_text(encoding="ascii").split("\n", 1)
    matrix.write_text(header.replace(old, new) + "\n" + rest, encoding="ascii", newline="\n")
    err = plan_error_text(capsys, root)
    assert expected in err and absent not in err
