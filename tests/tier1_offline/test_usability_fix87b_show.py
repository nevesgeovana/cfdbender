"""U-23: show is an outcome-first card; hashes and empty fields only under --verbose."""

from __future__ import annotations

from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, run


def finished(tmp_path, capsys):
    root = build(tmp_path / "mock", "mock")
    edit(root / "inputs/matrices/full.fs", "| 2h | 1.0 |", "| 2h | - |")
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    assert run(capsys, "run", "full", "--workspace", root, "--local")[0] == 0
    assert run(capsys, "collect", "full", "--workspace", root)[0] == 0
    return root


def test_u23_a_point_card_leads_with_the_outcome(tmp_path, capsys):
    root = finished(tmp_path, capsys)
    code, out, _ = run(capsys, "show", "7001/DP-M200AL+000", "--workspace", root)
    lines = out.splitlines()
    assert code == 0 and len(lines) <= 12
    assert lines[0].startswith("7001/DP-M200AL+000  COMPLETED converged")
    assert "sha256" not in out and "error:" not in out and "inputs_sha256" not in out
    assert "iterations " in out and "sims/sim_7001/datapoints/DP-M200AL+000" in out
    assert "CZ=" in out


def test_u23_a_sim_lists_its_points_one_line_each(tmp_path, capsys):
    root = finished(tmp_path, capsys)
    code, out, _ = run(capsys, "show", "7001", "--workspace", root)
    lines = [line for line in out.splitlines() if line.startswith("  7001/")]
    assert code == 0 and len(lines) == 2 and len(out.splitlines()) <= 6


def test_u23_verbose_prints_the_full_record_without_empty_fields(tmp_path, capsys):
    root = finished(tmp_path, capsys)
    code, out, _ = run(capsys, "show", "7001/DP-M200AL+000", "--workspace", root, "--verbose")
    assert code == 0 and "inputs_sha256:" in out and "  error:" not in out
