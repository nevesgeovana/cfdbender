"""--sims and --points narrow plan, run and collect; unknown ids refuse before any write."""

from __future__ import annotations

import json

import pytest
from test_p0000_item3c_adapter import workspace
from test_p0000_item3f_plan import tree

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def three(root):
    workspace(root)
    matrix = root / "inputs/matrices/study.fs"
    matrix.write_text(
        "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
        "1001|1|001|001|001|ALPHA:sweep|0,2\n"
        "1002|1|001|001|001|ALPHA:0|\n"
        "1003|1|001|001|001|ALPHA:0|\n",
        encoding="ascii",
    )


def cli(root, command, *options):
    return main([command, "study", "--workspace", str(root), *options])


def recorded(root):
    return sorted(r.point_id for r in RunRegistry(root).read())


def test_plan_lists_only_the_selected_simulations(tmp_path, capsys):
    three(tmp_path)
    assert cli(tmp_path, "plan", "--sims", "1001,1003") == 0
    out = capsys.readouterr().out
    assert "3 point(s)" in out and "1002/" not in out


@pytest.mark.parametrize("sims", [["1001", "1003"], ["1001,1003"]])
def test_run_runs_only_the_selected_simulations(tmp_path, sims):
    three(tmp_path)
    assert cli(tmp_path, "run", "--sims", *sims) == 0
    assert recorded(tmp_path) == ["1001/DP-AL+000", "1001/DP-AL+020", "1003/DP-AL+000"]


def test_run_one_point_then_resume_the_rest(tmp_path):
    three(tmp_path)
    assert cli(tmp_path, "run", "--sims", "1001", "--points", "DP-AL+020") == 0
    assert recorded(tmp_path) == ["1001/DP-AL+020"]
    assert cli(tmp_path, "run", "--points", "1002/DP-AL+000") == 0
    assert cli(tmp_path, "run", "--resume") == 0
    assert len(recorded(tmp_path)) == 4


@pytest.mark.parametrize("command", ["plan", "run"])
@pytest.mark.parametrize(
    "options,word",
    [
        (["--sims", "1009"], "--sims 1009"),
        (["--sims", "1002", "--points", "DP-AL+020"], "--points"),
    ],
)
def test_unknown_selection_refuses_without_writes(tmp_path, capsys, command, options, word):
    three(tmp_path)
    before = tree(tmp_path)
    assert cli(tmp_path, command, *options) == 2
    assert word in capsys.readouterr().err
    assert before == tree(tmp_path)


def test_collect_reads_only_the_selection_and_keeps_other_rows(tmp_path):
    three(tmp_path)
    assert cli(tmp_path, "run") == 0
    assert cli(tmp_path, "collect") == 0
    summary = tmp_path / "post/study/campaign_summary.json"
    full = json.loads(summary.read_bytes())
    assert main(["collect", "study", "--workspace", str(tmp_path), "--sims", "1002"]) == 0
    assert json.loads(summary.read_bytes()) == full
    assert main(["collect", "study", "--workspace", str(tmp_path), "--sims", "1009"]) == 2
