"""Usability findings of the first-campaign walk, each pinned as the user sees it."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli_route import main as matrix_main
from test_matrix_columns_all_solvers import build

NL = chr(10)


@pytest.fixture
def mock(tmp_path):
    return build(tmp_path / "w", "mock")


def run(capsys, *argv):
    code = matrix_main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    assert old in text, old
    path.write_text(text.replace(old, new), encoding="ascii", newline=NL)


# U-12 -------------------------------------------------------------------------------


def test_u12_a_missing_matrix_names_the_matrices_found(mock, capsys):
    code, _, err = run(capsys, "plan", "m1", "--workspace", mock)
    assert code == 2
    assert "no matrix m1 in inputs/matrices/ (found: full)" in err
    assert "Errno" not in err


def test_u12_a_missing_setup_names_the_file_the_row_asks_for(mock, capsys):
    (mock / "inputs/setups/s001.toml").unlink()
    code, _, err = run(capsys, "plan", "full", "--workspace", mock)
    assert code == 2
    assert "SET s001 names inputs/setups/s001.toml, which does not exist" in err
    assert "Errno" not in err


def test_u12_u13_outside_a_workspace_every_verb_refuses_the_same_way(tmp_path, capsys):
    folder = tmp_path / "plain"
    folder.mkdir()
    for verb in (("plan", "full"), ("status",), ("log",), ("history", "7001"), ("collect",)):
        code, out, err = run(capsys, *verb, "--workspace", folder)
        assert code == 2, verb
        assert "is not a cfdbender workspace (no init.toml); run cfd-workspace init" in err, verb
        assert "available solvers" not in err and "Errno" not in err and out == "", verb


# U-14 -------------------------------------------------------------------------------


def test_u14_empty_results_say_why(mock, capsys):
    code, out, _ = run(capsys, "status", "--workspace", mock)
    assert code == 0 and "no recorded sim yet: run a matrix first" in out
    matrix = mock / "inputs/matrices/full.fs"
    lines = matrix.read_text(encoding="ascii").splitlines()
    header = [c.strip() for c in lines[0].split("|")]
    run_at = header.index("RUN")
    cells = lines[2].split("|")
    cells[run_at] = " 0 "
    lines[2] = "|".join(cells)
    matrix.write_text(NL.join(lines) + NL, encoding="ascii", newline=NL)
    code, out, _ = run(capsys, "plan", "full", "--workspace", mock)
    assert code == 0 and "no active row: every row of full.fs has RUN = 0" in out
    code, out, _ = run(capsys, "run", "full", "--workspace", mock)
    assert code == 0 and "no active row: every row of full.fs has RUN = 0" in out


def test_u14_status_failed_with_none_failed_says_so(mock, capsys):
    edit(mock / "inputs/matrices/full.fs", "| 2h | 1.0 |", "| 2h | - |")
    assert run(capsys, "plan", "full", "--workspace", mock)[0] == 0
    assert run(capsys, "run", "full", "--workspace", mock, "--local")[0] == 0
    code, out, _ = run(capsys, "status", "--failed", "--workspace", mock)
    assert code == 0 and "no failed point" in out


# U-15 -------------------------------------------------------------------------------


def test_u15_every_row_refusal_starts_with_the_matrix_path_and_the_row(mock, capsys):
    matrix = mock / "inputs/matrices/full.fs"
    edit(matrix, "MACH:0.2", "MACH:abc")
    code, _, err = run(capsys, "plan", "full", "--workspace", mock)
    assert code == 2
    assert err.startswith("inputs/matrices/full.fs, data row 1 (POL 7001)")
    assert "\\" not in err.split(":")[0]


def test_u15_an_adapter_refusal_names_the_setup_file_it_concerns(mock, capsys):
    setup = mock / "inputs/setups/s001.toml"
    edit(setup, "iterations", "iteratons")
    code, _, err = run(capsys, "plan", "full", "--workspace", mock)
    assert code == 2
    assert err.startswith("inputs/matrices/full.fs, data row 1 (POL 7001)")
    assert "inputs/setups/s001.toml" in err


def plan_error_text(capsys, root):
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 2
    return err
