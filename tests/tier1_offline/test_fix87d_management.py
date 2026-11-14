"""Closing-review decisions: running points, shared sims, mark reasons and --prefer-other."""

from __future__ import annotations

import json
import shutil

from test_cli_verbs import FIRST, grown, out, ran

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.workspace.planning import plan_workspace


def _running(root):
    grown(root)
    registry = RunRegistry(root)
    registry.record(plan_workspace(root, "study"))
    registry.transition(FIRST, "RUNNING")


def test_mark_failed_refuses_a_running_point_and_says_to_stop_the_run(tmp_path, capsys):
    _running(tmp_path)
    args = ["mark-failed", "--sims", "1001", "--apply", "--workspace", str(tmp_path)]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "RUNNING" in err and "stop the run first" in err
    assert {r.point_id: r for r in RunRegistry(tmp_path).read()}[FIRST].status == "RUNNING"


def test_delete_sims_needs_force_for_a_running_point(tmp_path, capsys):
    _running(tmp_path)
    base = [
        "delete-sims",
        "--sims",
        "1001",
        "--apply",
        "--matrix-products",
        "points-only",
        "--workspace",
        str(tmp_path),
    ]
    assert main(base) == 2
    assert "still RUNNING; --force" in capsys.readouterr().err
    assert FIRST in {r.point_id for r in RunRegistry(tmp_path).read()}
    assert main([*base, "--force"]) == 0
    assert FIRST not in {r.point_id for r in RunRegistry(tmp_path).read()}


def test_delete_sims_refuses_a_sim_another_manifest_still_records(tmp_path, capsys):
    ran(tmp_path)
    shutil.copy(tmp_path / "runs.json", tmp_path / "alt.json")
    args = [
        "delete-sims",
        "--sims",
        "1001",
        "--runs",
        "alt.json",
        "--apply",
        "--matrix-products",
        "points-only",
    ]
    assert main([*args, "--workspace", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "runs.json" in err and "still records" in err
    assert (tmp_path / "sims/sim_1001").is_dir()


def test_a_marked_failed_point_shows_the_reason_in_status_and_the_summary(tmp_path, capsys):
    ran(tmp_path)
    mark = ["mark-failed", "--sims", "1001", "--reason", "wrong mesh", "--apply"]
    assert main([*mark, "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["status", "--points", "--workspace", str(tmp_path)]) == 0
    assert "wrong mesh" in out(capsys)
    main(["collect", "study", "--workspace", str(tmp_path)])
    summary = json.loads((tmp_path / "post/study/campaign_summary.json").read_bytes())
    row = next(p for p in summary["points"] if p["point_id"] == FIRST)
    assert "wrong mesh" in str(row["error"])


def test_sync_prefer_other_without_overwrite_refuses(tmp_path, capsys):
    root, other = tmp_path / "main", tmp_path / "other"
    ran(root)
    ran(other)
    (root / "inputs/sync-workspaces.toml").write_text(
        f'[workspaces]\nother = "{other.as_posix()}"\n', encoding="utf-8"
    )
    base = ["sync", "all", "--prefer-other", "--workspace", str(root)]
    assert main(base) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main([*base, "--overwrite"]) == 0
