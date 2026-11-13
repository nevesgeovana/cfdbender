"""Gap 11: the activity log and live logs live under logs/ and the log verb reads them."""

from __future__ import annotations

import json

import pytest
from test_cli_verbs import grown, out, snapshot
from test_p0000_item3c_adapter import workspace

from cfdbender.core import mesh
from cfdbender.matrix.cli import main


def activity(root):
    path = root / "logs/activity.log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def legacy_mesh_workspace(root):
    workspace(root)
    legacy = root / "inputs/geometry"
    legacy.mkdir(parents=True)
    (legacy / "wing.dat").write_text("x")
    (root / "inputs/matrices/study.fs").write_text(
        "POL|RUN|SET|REF|PPROC|GEOMETRY\n1001|1|001|001|001|wing.dat\n", encoding="ascii"
    )
    mesh._warned.clear()


def test_a_run_writes_a_live_log_of_what_the_console_said(tmp_path, capsys):
    grown(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    printed = out(capsys)
    (live,) = sorted((tmp_path / "logs").glob("run-*.log"))
    text = live.read_text(encoding="utf-8")
    assert text.startswith("# run started ")
    for line in printed.splitlines():
        assert line in text.splitlines()
    assert "\r" not in text


def test_every_logged_command_leaves_started_and_finished_activity(tmp_path, capsys):
    grown(tmp_path)
    for command in (["plan", "study"], ["run", "study"], ["post", "study"], ["collect", "study"]):
        assert main([*command, "--workspace", str(tmp_path)]) == 0
    events = activity(tmp_path)
    assert [(e["stage"], e["event"]) for e in events] == [
        ("run", "started"),
        ("run", "finished"),
        ("post", "started"),
        ("post", "finished"),
        ("collect", "started"),
        ("collect", "finished"),
    ]
    assert all(e["exit_code"] == 0 for e in events if e["event"] == "finished")
    assert all(e["timestamp"] for e in events)
    text = (tmp_path / "logs/activity.log").read_text(encoding="utf-8")
    assert "[run] started" in text and "[collect] finished" in text
    capsys.readouterr()


def test_a_refused_command_is_logged_as_failed(tmp_path, capsys):
    workspace(tmp_path, "failed")
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 2
    last = activity(tmp_path)[-1]
    assert (last["stage"], last["event"], last["exit_code"]) == ("run", "failed", 2)
    capsys.readouterr()


def test_the_log_verb_reads_the_activity_log(tmp_path, capsys):
    grown(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["log", "--stage", "command", "--json", "--workspace", str(tmp_path)]) == 0
    rows = json.loads(out(capsys))
    assert [r["event"] for r in rows] == ["run started", "run finished"]
    assert main(["log", "--problems", "--json", "--workspace", str(tmp_path)]) == 0
    assert json.loads(out(capsys)) == []


def test_warnings_are_always_written_to_the_warnings_log(tmp_path, capsys):
    legacy_mesh_workspace(tmp_path)
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert "Warnings (1)" in capsys.readouterr().err
    assert "legacy mesh folder" in (tmp_path / "logs/warnings.log").read_text(encoding="utf-8")


def test_warning_off_only_silences_the_console(tmp_path, capsys):
    legacy_mesh_workspace(tmp_path)
    assert main(["--warning-off", "plan", "study", "--workspace", str(tmp_path)]) == 0
    assert "Warnings" not in capsys.readouterr().err
    assert "legacy mesh folder" in (tmp_path / "logs/warnings.log").read_text(encoding="utf-8")


def test_the_live_log_keeps_the_warnings_block_too(tmp_path, capsys):
    legacy_mesh_workspace(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    capsys.readouterr()
    (live,) = sorted((tmp_path / "logs").glob("run-*.log"))
    assert "Warnings (1)" in live.read_text(encoding="utf-8")


@pytest.mark.parametrize("verb", [["status"], ["log"], ["show", "1001"]])
def test_read_only_verbs_write_no_log(tmp_path, capsys, verb):
    grown(tmp_path)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    before = snapshot(tmp_path)
    assert main([*verb, "--workspace", str(tmp_path)]) == 0
    assert snapshot(tmp_path) == before
    capsys.readouterr()


def test_a_folder_that_is_no_workspace_gets_no_logs(tmp_path, capsys):
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 2
    assert not (tmp_path / "logs").exists()
    capsys.readouterr()


def test_plan_only_previews_and_leaves_no_live_log_or_activity(tmp_path, capsys):
    grown(tmp_path)
    before = snapshot(tmp_path)
    assert main(["plan", "study", "--workspace", str(tmp_path)]) == 0
    assert not (tmp_path / "logs").exists()
    assert {k for k in snapshot(tmp_path)} - set(before) <= {"post/study/plan.json"}
    capsys.readouterr()
