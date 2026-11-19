"""cfd-matrix run --force-rerun-all runs every selected point again whatever its recorded state."""

from __future__ import annotations

import pytest
from test_p0000_item3f_plan import cli as hpc_cli
from test_p0000_item3f_plan import prepared
from test_p0000_item3f_registry import evidence
from test_submit_batch import FIELDS, planned, setup

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main
from cfdbender.workspace.planning import plan_workspace

FIRST = "1001/DP-NOCOND"
STAMP = "2026-11-10T12:00:0"
WARNING = "a job of the earlier submission may still be running; cancelling it is your action"


def leftovers(root, point):
    """Files an earlier attempt leaves in the point folder."""
    folder = root / point.directory
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("stdout.log", "stderr.log", "history.csv", "job.exit"):
        (folder / name).write_text("earlier attempt\n", encoding="ascii")
    (folder / "restart").mkdir(exist_ok=True)
    (folder / "restart/state.bin").write_text("earlier", encoding="ascii")


def build(root, state):
    """Record the first point in ``state`` and return its point."""
    plan = plan_workspace(root, "study")
    point = plan.points[0]
    if state == "unrecorded":
        return point
    registry = RunRegistry(root)
    registry.record(plan, created_at=f"{STAMP}0+00:00")
    if state == "planned":
        return point
    registry.transition(point.point_id, "RUNNING", timestamp=f"{STAMP}1+00:00")
    if state == "running":
        leftovers(root, point)
        return point
    if state == "running-no-folder":
        return point
    data = evidence(root, point)
    leftovers(root, point)
    registry.transition(point.point_id, "SUBMITTED", timestamp=f"{STAMP}2+00:00", submission=data)
    if state == "submitted":
        return point
    for name in point.outputs:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("numerical output", encoding="utf-8")
    registry.transition(
        point.point_id,
        "COMPLETED",
        timestamp=f"{STAMP}3+00:00",
        classification="failed" if state == "failed" else "converged",
        error="exit status 1" if state == "failed" else None,
    )
    return point


STATES = [
    "unrecorded",
    "planned",
    "running",
    "running-no-folder",
    "submitted",
    "completed",
    "failed",
]
ALIVE = {"running", "running-no-folder", "submitted"}


def run_all(root, *options):
    return main(["run", "study", "--workspace", str(root), "--local", *options])


@pytest.mark.parametrize("state", STATES)
def test_force_rerun_all_runs_the_point_whatever_its_state(tmp_path, capsys, state):
    prepared(tmp_path)
    point = build(tmp_path, state)
    assert run_all(tmp_path, "--force-rerun-all") == 0
    out = capsys.readouterr()
    record = RunRegistry(tmp_path).read()[0]
    assert record.status == "COMPLETED" and record.classification != "failed"
    assert record.started_at > f"{STAMP}3+00:00" or state in ("unrecorded", "planned")
    assert (WARNING in out.out + out.err) == (state in ALIVE)
    if state in ALIVE:
        assert (point.point_id in out.out + out.err) and "warning" in (out.out + out.err).lower()


@pytest.mark.parametrize("state", ["running", "submitted", "completed", "failed"])
def test_force_rerun_all_archives_the_earlier_attempt_whole(tmp_path, state):
    prepared(tmp_path)
    point = build(tmp_path, state)
    if state in ("completed", "failed"):
        leftovers(tmp_path, point)
    assert run_all(tmp_path, "--force-rerun-all") == 0
    (archive,) = (tmp_path / "archive").iterdir()
    old = archive / point.directory
    for name in ("stdout.log", "history.csv", "job.exit", "restart/state.bin"):
        assert (old / name).read_text(encoding="ascii").startswith("earlier"), name
    new = tmp_path / point.directory
    assert not (new / "job.exit").exists(), "nothing of the earlier attempt stays"
    assert not (new / "restart/state.bin").exists()
    assert (new / "history.csv").read_text(encoding="ascii") != "earlier attempt\n"
    (entry,) = RunRegistry(tmp_path).history()
    assert entry["record"]["status"] == ("COMPLETED" if state == "failed" else state.upper())


@pytest.mark.parametrize("state", STATES)
def test_without_the_flag_the_state_still_refuses(tmp_path, state):
    prepared(tmp_path)
    build(tmp_path, state)
    expected = 0 if state in ("unrecorded", "planned") else 2
    assert run_all(tmp_path) == expected


def test_force_rerun_all_leaves_a_dead_writers_lock_taken_over(tmp_path):
    prepared(tmp_path)
    build(tmp_path, "submitted")
    (tmp_path / "runs.json.lock").write_text("2147483646", encoding="ascii")
    assert run_all(tmp_path, "--force-rerun-all") == 0


def test_force_rerun_all_still_refuses_what_is_not_a_state(tmp_path, capsys):
    prepared(tmp_path)
    build(tmp_path, "submitted")
    (tmp_path / "inputs/setups/s001.toml").write_text("ncpus = 2\nsolver_version = 'x'\n")
    assert run_all(tmp_path, "--force-rerun-all") == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize("state", ["submitted", "running", "completed"])
def test_force_rerun_all_on_a_cluster_submits_again(tmp_path, capsys, state):
    prepared(tmp_path)
    point = build(tmp_path, state)
    assert hpc_cli(tmp_path, "plan") == 0
    assert hpc_cli(tmp_path, "run", "--force-rerun-all") == 0
    out = capsys.readouterr()
    (record,) = RunRegistry(tmp_path).read()
    assert record.status == "SUBMITTED" and record.started_at > f"{STAMP}3+00:00"
    assert (WARNING in out.out + out.err) == (state != "completed")
    assert (tmp_path / "archive").is_dir() and point.directory


def test_force_rerun_all_under_batch_submits_the_batch_again(tmp_path, capsys):
    setup(tmp_path)
    planned(tmp_path, capsys, n=1)
    assert hpc_cli(tmp_path, "run", "--batch", "1") == 0
    first = {r.point_id: r.status for r in RunRegistry(tmp_path).read()}
    assert set(first.values()) == {"SUBMITTED"}
    capsys.readouterr()
    assert hpc_cli(tmp_path, "plan", "--batch", "1") == 0, "the plan names a fresh batch folder"
    capsys.readouterr()
    assert hpc_cli(tmp_path, "run", "--batch", "1", "--force-rerun-all") == 0
    out = capsys.readouterr()
    assert (out.out + out.err).count(WARNING) == 3
    assert {r.status for r in RunRegistry(tmp_path).read()} == {"SUBMITTED"}
    assert len(RunRegistry(tmp_path).history()) == 3
    assert (tmp_path / "sims/batch/study_b1/BATCH-1001-1003.sh").is_file()
    assert (tmp_path / "sims/batch/study_b2/BATCH-1001-1003.sh").is_file()
    assert FIELDS


def test_force_rerun_all_under_points_selection_leaves_the_others(tmp_path, capsys):
    prepared(tmp_path)
    matrix = tmp_path / "inputs/matrices/study.fs"
    matrix.write_text(matrix.read_text().rstrip() + "\n1002|1|001|001|001\n", encoding="ascii")
    assert hpc_cli(tmp_path, "plan") == 0
    assert hpc_cli(tmp_path, "run") == 0
    before = {r.point_id: r.started_at for r in RunRegistry(tmp_path).read()}
    capsys.readouterr()
    assert hpc_cli(tmp_path, "run", "--points", "1002/DP-NOCOND", "--force-rerun-all") == 0
    after = {r.point_id: r.started_at for r in RunRegistry(tmp_path).read()}
    changed = [p for p in before if before[p] != after[p]]
    assert len(changed) == 1 and changed[0].startswith("1002/")
    assert len(RunRegistry(tmp_path).history()) == 1
