"""U-18, U-19, U-20, U-22 and U-25: what plan, collect, post and log tell the user."""

from __future__ import annotations

import json

from test_conformance_workspace import executables, state_no_build
from test_matrix_columns_all_solvers import build
from test_usability_fix87b import edit, run


def standin_workspace(tmp_path, solver="su2"):
    root = build(tmp_path / solver, solver)
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes() + executables(root, solver).encode("ascii"))
    return root


# U-18 -------------------------------------------------------------------------------


def test_u18_a_build_nobody_declares_is_warned_at_plan(tmp_path, capsys):
    root = standin_workspace(tmp_path)
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 0
    assert "SOLVER_BUILD 1.0 is carried but no inputs/executables.toml declares it" in err


def test_u18_a_declared_file_without_the_build_refuses_at_plan(tmp_path, capsys):
    root = standin_workspace(tmp_path)
    (root / "inputs/executables.toml").write_text('[su2]\n"9.9" = "x"\n', encoding="ascii")
    code, _, err = run(capsys, "plan", "full", "--workspace", root)
    assert code == 2
    assert "SOLVER_BUILD '1.0' is not declared under [su2]" in err


# U-19 -------------------------------------------------------------------------------


def test_u19_plan_prints_one_line_per_point_and_a_closing_line(tmp_path, capsys):
    root = standin_workspace(tmp_path, "openfoam")
    state_no_build(root)
    code, out, _ = run(capsys, "plan", "full", "--workspace", root)
    assert code == 0
    lines = out.splitlines()
    assert "output:" not in out and "would:" not in out
    assert len([line for line in lines if line.startswith("  7001/")]) == 2
    assert lines[-1] == "2 point(s) planned; receipt post/full/plan.json; next: cfd-matrix run full"
    code, out, _ = run(capsys, "plan", "full", "--workspace", root, "--verbose")
    assert "output:" in out and "would:" in out and out.splitlines()[-1] == lines[-1]


# U-20 -------------------------------------------------------------------------------


def test_u20_collect_reports_recorded_points_the_matrix_no_longer_has(tmp_path, capsys):
    root = build(tmp_path / "mock", "mock")
    edit(root / "inputs/matrices/full.fs", "| 2h | 1.0 |", "| 2h | - |")
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    assert run(capsys, "run", "full", "--workspace", root, "--local")[0] == 0
    edit(root / "inputs/matrices/full.fs", "MACH:0.2", "MACH:0.3")
    assert run(capsys, "plan", "full", "--workspace", root)[0] == 0
    code, out, _ = run(capsys, "collect", "full", "--workspace", root)
    assert "2 recorded points are not in full.fs any more" in out
    assert "7001/DP-M200AL+000" in out and "cfd-matrix delete-sims" in out
    assert code == 0


# U-22 -------------------------------------------------------------------------------


def test_u22_collect_and_post_print_workspace_relative_paths_the_same_way(tmp_path, capsys):
    root = build(tmp_path / "mock", "mock")
    edit(root / "inputs/matrices/full.fs", "| 2h | 1.0 |", "| 2h | - |")
    run(capsys, "plan", "full", "--workspace", root)
    run(capsys, "run", "full", "--workspace", root, "--local")
    _, collected, _ = run(capsys, "collect", "full", "--workspace", root)
    _, posted, _ = run(capsys, "post", "full", "--workspace", root)
    assert "wrote post/full/campaign_sweep.csv" in collected
    assert "wrote post/full/campaign_sweep.csv" in posted
    assert str(root) not in collected and str(root) not in posted


# U-25 -------------------------------------------------------------------------------


def test_u25_the_registration_event_is_named_run_not_plan(tmp_path, capsys):
    root = build(tmp_path / "mock", "mock")
    edit(root / "inputs/matrices/full.fs", "| 2h | 1.0 |", "| 2h | - |")
    run(capsys, "plan", "full", "--workspace", root)
    run(capsys, "run", "full", "--workspace", root, "--local")
    code, out, _ = run(capsys, "log", "--workspace", root, "--json")
    rows = json.loads(out)
    assert code == 0
    assert not [r for r in rows if r["stage"] == "plan" and r["event"] == "registered"]
    assert [r for r in rows if r["stage"] == "run" and r["event"] == "registered"]
