"""collect writes the family polar files, one row per point, and no campaign sweep."""

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
    out = capsys.readouterr().out
    assert "wrote post/study/polars/FAM-ACOEF-P1001.csv" in out and "campaign_sweep" not in out
    assert not (post / "campaign_sweep.csv").exists()
    summary = json.loads((post / "campaign_summary.json").read_bytes())["points"]
    rows = [
        row
        for pol in ("1001", "1002", "1003")
        for row in read(post / f"polars/FAM-ACOEF-P{pol}.csv")
    ]
    assert len(rows) == len(summary) == 4
    header = list(rows[0])
    assert header[:4] == ["POL", "POINT_ID", "POINT", "RUN_TYPE"] and header[4] == "ALPHA"
    assert header[-7:] == ["CX", "CY", "CZ", "Cl", "Cm", "Cn", "MESH_FACES"]
    for row, point in zip(rows, summary, strict=True):
        result = point["result"]
        assert row["POL"] == point["pol"]
        assert row["POINT"] == point["point_id"].split("/", 1)[1]
        assert float(row["ALPHA"]) == point["condition"]["ALPHA"]
        for name, value in result["coefficients"].items():
            assert float(row[name]) == value["value"]
    polar = read(post / "polars/FAM-ACOEF-P1001.csv")
    assert [r["POINT"] for r in polar] == ["DP-AL+000", "DP-AL+020"]
    assert polar == rows[:2]
    assert not any(p.name.startswith("P1001.") for p in (post / "polars").iterdir())


def test_an_unread_point_leaves_empty_cells(tmp_path):
    three(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path), "--sims", "1001"]) == 0
    from dataclasses import replace

    from cfdbender.core.registry import RunRegistry
    from cfdbender.workspace.planning import plan_workspace

    plan = plan_workspace(tmp_path, "study")
    RunRegistry(tmp_path).record(replace(plan, points=plan.points[2:3]))
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 3
    rows = read(tmp_path / "post/study/polars/FAM-ACOEF-P1002.csv")
    assert rows[0]["CX"] == rows[0]["CDW"] == "NA"
