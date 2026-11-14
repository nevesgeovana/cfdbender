"""A matrix that grows after points ran keeps its finished points valid.

Staleness of a point is decided by the digests of the files that point read (its s, r and p
files, its geometry), never by the matrix file, which every point shares and which a new row
changes.
"""

from __future__ import annotations

import json

from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def _grow(root):
    matrix = root / "inputs/matrices/study.fs"
    matrix.write_text(matrix.read_text().rstrip() + "\n1002|1|001|001|001\n", encoding="ascii")


def test_appending_a_row_after_points_ran_keeps_the_old_points_collectable(tmp_path, capsys):
    workspace(tmp_path)
    ws = str(tmp_path)
    assert main(["plan", "study", "--workspace", ws]) == 0
    assert main(["run", "study", "--workspace", ws]) == 0
    assert main(["collect", "study", "--workspace", ws]) == 0
    old = {r.point_id for r in RunRegistry(tmp_path).read()}
    _grow(tmp_path)
    capsys.readouterr()
    assert main(["plan", "study", "--workspace", ws]) == 0
    assert main(["run", "study", "--resume", "--workspace", ws]) == 0
    capsys.readouterr()
    assert main(["collect", "study", "--workspace", ws]) == 0, capsys.readouterr().err
    assert main(["post", "study", "--workspace", ws]) == 0, capsys.readouterr().err
    records = RunRegistry(tmp_path).read()
    assert old < {r.point_id for r in records}
    assert all(r.status == "COMPLETED" for r in records)


def test_a_changed_setup_still_makes_a_finished_point_stale(tmp_path, capsys):
    workspace(tmp_path)
    ws = str(tmp_path)
    assert main(["plan", "study", "--workspace", ws]) == 0
    assert main(["run", "study", "--workspace", ws]) == 0
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(setup.read_text(encoding="ascii") + "# edited\n", encoding="ascii")
    capsys.readouterr()
    assert main(["collect", "study", "--workspace", ws]) != 0
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    assert "input changed after planning" in summary["points"][0]["error"]
