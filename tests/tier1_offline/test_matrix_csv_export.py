"""collect writes campaign_sweep.csv and polars/P<POL>.csv, one row per point."""

from __future__ import annotations

import csv
import json

from test_matrix_selection import three

from cfdbender.matrix.cli import main


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_collect_writes_one_row_per_point_with_every_coefficient(tmp_path, capsys):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    post = tmp_path / "post/study"
    assert f"collected table: {post / 'campaign_sweep.csv'}" in capsys.readouterr().out
    summary = json.loads((post / "campaign_summary.json").read_bytes())["points"]
    rows = read(post / "campaign_sweep.csv")
    assert len(rows) == len(summary) == 4
    header = list(rows[0])
    assert header[:10] == [
        "POL",
        "point",
        "ALPHA",
        "status",
        "classification",
        "iterations",
        "residual",
        "wall_time_s",
        "solver_version",
        "CX",
    ]
    assert header[9:] == ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]
    for row, point in zip(rows, summary, strict=True):
        result = point["result"]
        assert row["POL"] == point["pol"]
        assert row["point"] == point["point_id"].split("/", 1)[1]
        assert float(row["ALPHA"]) == point["condition"]["ALPHA"]
        assert row["status"] == "COMPLETED" and row["classification"] == "converged"
        assert int(row["iterations"]) == result["history"][-1]["iteration"]
        assert float(row["residual"]) == result["history"][-1]["residual"]
        assert float(row["wall_time_s"]) >= 0
        assert row["solver_version"] == "mock-1"
        for name, value in result["coefficients"].items():
            assert float(row[name]) == value["value"]
    polar = read(post / "polars/P1001.csv")
    assert [r["point"] for r in polar] == ["DP-AL+000", "DP-AL+020"]
    assert polar == rows[:2]
    assert sorted(p.name for p in (post / "polars").iterdir()) == [
        "P1001.csv",
        "P1002.csv",
        "P1003.csv",
    ]


def test_an_unread_point_leaves_empty_cells(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path), "--sims", "1001"]) == 0
    from dataclasses import replace

    from cfdbender.core.registry import RunRegistry
    from cfdbender.workspace.planning import plan_workspace

    plan = plan_workspace(tmp_path, "study")
    RunRegistry(tmp_path).record(replace(plan, points=plan.points[2:3]))
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 3
    rows = read(tmp_path / "post/study/polars/P1002.csv")
    assert rows[0]["status"] == "PLANNED"
    assert rows[0]["classification"] == rows[0]["iterations"] == rows[0]["CX"] == ""
