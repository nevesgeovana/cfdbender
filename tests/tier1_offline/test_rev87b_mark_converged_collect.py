"""F8-002: a point marked converged stays collectable; its result keeps its row."""

from __future__ import annotations

import csv
import json

from test_cli_verbs import FIRST, ran

from cfdbender.matrix.cli import main


def test_a_point_marked_converged_is_collected_with_its_result(tmp_path, capsys):
    ran(tmp_path, "iteration_limit")
    mark = ["mark-converged", "--sims", "1001", "--reason", "read the history", "--apply"]
    assert main([*mark, "--workspace", str(tmp_path)]) == 0
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 0
    assert "collected 2, failed 0" in capsys.readouterr().out
    rows = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())["points"]
    row = next(r for r in rows if r["point_id"] == FIRST)
    assert row["classification"] == "converged" and row["error"] is None
    assert row["result"]["classification"] == "iteration_limit", "the numerical class is kept"
    with (tmp_path / "post/study/campaign_sweep.csv").open(newline="") as stream:
        cells = next(r for r in csv.DictReader(stream) if r["POL"] == "1001")
    assert cells["classification"] == "converged" and cells["CL"] != "NA"


def test_a_record_that_differs_from_its_evidence_without_a_mark_is_still_refused(tmp_path):
    ran(tmp_path, "iteration_limit")
    path = tmp_path / "runs.json"
    path.write_text(path.read_text().replace('"iteration_limit"', '"converged"', 1))
    assert main(["collect", "study", "--workspace", str(tmp_path)]) == 1
    rows = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())["points"]
    assert "differs from record" in rows[0]["error"]
