"""``cfd-matrix status`` prints what ``pyfs-matrix status`` prints, from cfdbender's records.

One row per polar (a simulation of a matrix) with its recorded and planned points and its
status, or one row per datapoint; the totals; and, per matrix, whether its plan and its post are
current. A point the plan receipt names and no record carries reads ``planned``. Nothing is
written. The shapes follow pyflightstream's tier-1 status tests and its table.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from test_matrix_columns_all_solvers import build

from cfdbender.matrix.cli import main

HEAD = ["SIM", "POLAR", "SWEEP", "POINTS", "MATRIX", "STATUS"]


@pytest.fixture
def root(tmp_path):
    made = build(tmp_path, "mock")
    matrix = made / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii").replace("| 2h | 1.0 |", "| 2h | - |"))
    for moved in (made / "inputs/geometries").iterdir():
        moved.rename(made / "inputs/mesh" / moved.name)
    (made / "inputs/geometries").rmdir()
    return made


def _go(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    streams = capsys.readouterr()
    return code, streams.out, streams.err


def _rows(out: str) -> list[list[str]]:
    lines = out.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("SIM"))
    rows = []
    for line in lines[first:]:
        if " polar(s), " in line or " datapoint(s): " in line:
            break
        rows.append([cell.strip() for cell in line.split("  ") if cell.strip()])
    return rows


def test_a_planned_matrix_reads_planned_with_the_plan_footer(root, capsys):
    assert main(["plan", "full", "--workspace", str(root)]) == 0
    code, out, _ = _go(capsys, "status", "--workspace", str(root))
    assert code == 0
    rows = _rows(out)
    assert rows[0] == HEAD
    assert rows[1] == ["7001", "M200AL+sweep", "alpha", "0/2", "full", "planned"]
    assert "1 polar(s), 2 datapoint(s): planned" in out
    assert "plan/full: made from the matrix on disk" in out
    assert "post/full: no post yet" in out


def test_a_run_and_collected_matrix_counts_its_points_and_dates_its_post(root, capsys):
    for argv in (["plan", "full"], ["run", "full", "--local"], ["collect", "full"]):
        assert main([*argv, "--workspace", str(root)]) == 0
    _, out, _ = _go(capsys, "status", "--workspace", str(root))
    assert _rows(out)[1] == ["7001", "M200AL+sweep", "alpha", "2/2", "full", "COMPLETED"]
    assert "1 polar(s), 2 datapoint(s): COMPLETED" in out
    assert "post/full: written " in out and ", complete" in out
    assert "not yet posted" not in out


def test_the_matrix_changed_after_its_plan_makes_the_plan_stale(root, capsys):
    assert main(["plan", "full", "--workspace", str(root)]) == 0
    matrix = root / "inputs/matrices/full.fs"
    matrix.write_text(matrix.read_text(encoding="ascii") + "\n", encoding="ascii")
    _, out, _ = _go(capsys, "status", "--workspace", str(root))
    assert "plan/full: made from another revision of the matrix; plan it again" in out


def test_points_prints_one_row_per_datapoint_with_its_columns(root, capsys):
    for argv in (["plan", "full"], ["run", "full", "--local"]):
        assert main([*argv, "--workspace", str(root)]) == 0
    _, out, _ = _go(capsys, "status", "--points", "--workspace", str(root))
    rows = _rows(out)
    assert rows[0][:4] == ["SIM", "POINT", "MATRIX", "STATUS"]
    assert out.splitlines()[[line.startswith("SIM") for line in out.splitlines()].index(True)].split() == [
        "SIM", "POINT", "MATRIX", "STATUS", "WAS", "ITER", "RESIDUAL", "WALL", "BUILD", "WHERE"
    ]
    assert [r[1] for r in rows[1:]] == ["M200AL+000", "M200AL+020"]
    assert "2 datapoint(s): COMPLETED" in out
    assert "plan/full:" in out


def test_selection_words_are_checked_and_an_unknown_sim_is_named(root, capsys):
    for argv in (["plan", "full"], ["run", "full", "--local"]):
        assert main([*argv, "--workspace", str(root)]) == 0
    code, _, err = _go(capsys, "status", "--status", "SUCCESS", "--workspace", str(root))
    assert code == 2 and "status (CLI: --status)" in err and "'SUCCESS'" in err
    code, out, err = _go(capsys, "status", "--sims", "7001,9999", "--workspace", str(root))
    assert code == 0 and "simulation 9999 holds no record and no planned point" in err
    assert [row[0] for row in _rows(out)[1:]] == ["7001"]
    code, _, err = _go(capsys, "status", "--sims", "[9999]", "--workspace", str(root))
    assert code == 1 and "9999" in err
    _, out, _ = _go(capsys, "status", "--status", "COMPLETED", "--workspace", str(root))
    assert len(_rows(out)) == 2
    _, out, _ = _go(capsys, "status", "--status", "FAILED_*", "--workspace", str(root))
    assert len(_rows(out)) == 1
    _, out, _ = _go(capsys, "status", "--matrix", "full.fs", "--workspace", str(root))
    assert len(_rows(out)) == 2


def test_json_and_csv_print_the_same_rows_on_stdout_alone(root, capsys):
    for argv in (["plan", "full"], ["run", "full", "--local"]):
        assert main([*argv, "--workspace", str(root)]) == 0
    _, out, err = _go(capsys, "status", "--json", "--workspace", str(root))
    document = json.loads(out)
    assert document["schema"].startswith("cfdbender.status")
    assert document["rows"][0]["points"] == "2/2" and document["rows"][0]["recorded"] == 2
    assert "cfd-matrix status" in err
    _, out, _ = _go(capsys, "status", "--csv", "--workspace", str(root))
    table = list(csv.reader(io.StringIO(out)))
    assert table[0] == [c.lower() for c in HEAD]
    assert table[1][:4] == ["7001", "M200AL+sweep", "alpha", "2/2"]


def _tree(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_status_writes_nothing(root, capsys):
    for argv in (["plan", "full"], ["run", "full", "--local"], ["collect", "full"]):
        assert main([*argv, "--workspace", str(root)]) == 0
    before = _tree(root)
    for flags in ((), ("--points",), ("--json",), ("--csv",), ("--failed",)):
        assert _go(capsys, "status", "--workspace", str(root), *flags)[0] == 0
    assert _tree(root) == before
