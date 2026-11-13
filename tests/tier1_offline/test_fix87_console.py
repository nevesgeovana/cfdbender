"""Walk findings U-04, U-10, U-11 and U-24: what the console says about failed points and times."""

from __future__ import annotations

from test_conformance_workspace import SOLVERS, build, local
from test_matrix_columns_all_solvers import CASES  # noqa: F401

from cfdbender.matrix._console import clock, status_word
from cfdbender.matrix.cli import main
from cfdbender.matrix.receipt import write_receipt
from cfdbender.workspace.planning import plan_workspace

assert "mock" in SOLVERS


def _failed_mock(tmp_path):
    root = build(tmp_path, "mock")
    setup = root / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text(encoding="ascii").replace("converged", "failed"))
    matrix = root / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii").replace("| 2h | 1.0 |", "| 2h | - |"))
    write_receipt(plan_workspace(root, "full"))
    return root


def test_u10_a_failed_point_reads_failed_not_completed():
    assert status_word("COMPLETED", "failed", "FAILED_EXECUTION: exit code 1") == "FAILED_EXECUTION"
    assert status_word("COMPLETED", "failed", None) == "FAILED"
    assert status_word("COMPLETED", "converged", None) == "COMPLETED"
    assert status_word("RUNNING", None, None) == "RUNNING"


def test_u24_times_print_as_local_time_to_the_second():
    shown = clock("2026-11-13T12:32:48.116504+00:00")
    assert len(shown) == 19 and shown[10] == " " and "." not in shown
    assert clock("not a time") == "not a time"


def test_u04_u11_resume_says_how_to_redo_a_failed_point_and_a_refusal_says_what_to_type(
    tmp_path, capsys
):
    root = _failed_mock(tmp_path)
    assert local(root) == 2
    out = capsys.readouterr()
    assert "FAILED" in out.out and "COMPLETED failed" not in out.out
    assert local(root) == 2
    err = capsys.readouterr().err
    assert "--resume" in err and "--force-rerun POINT" in err, err
    assert local(root, "--resume") == 0
    out = capsys.readouterr().out
    assert "skipped: COMPLETED failed; redo it with --force-rerun 7001/" in out
    assert main(["status", "--workspace", str(root), "--points"]) == 0
    status = capsys.readouterr().out
    assert "FAILED" in status
