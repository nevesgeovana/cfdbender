"""cfd-matrix run --resume, --force-rerun POINT and --force-rerun-all archive, never delete."""

from __future__ import annotations

import json

import pytest
from test_p0000_item3c_adapter import workspace
from test_p0000_item3f_plan import tree

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix import rerun
from cfdbender.matrix.cli import main
from cfdbender.workspace.planning import plan_workspace

FIRST = "1001/DP-NOCOND"
SECOND = "1002/DP-NOCOND"


def cli(root, *options):
    return main(["run", "study", "--workspace", str(root), *options])


def grow(root):
    matrix = root / "inputs/matrices/study.fs"
    matrix.write_text(matrix.read_text().rstrip() + "\n1002|1|001|001|001\n", encoding="ascii")


def statuses(root):
    return {r.point_id: r.status for r in RunRegistry(root).read()}


def test_resume_runs_only_the_points_a_grown_matrix_added(tmp_path, capsys):
    workspace(tmp_path)
    assert cli(tmp_path) == 0
    first = (tmp_path / "sims/sim_1001/datapoints/DP-NOCOND/history.csv").read_bytes()
    grow(tmp_path)
    assert cli(tmp_path) == 2, "a recorded point still refuses a run without --resume"
    assert cli(tmp_path, "--resume") == 0
    assert f"{FIRST}  skipped: COMPLETED" in capsys.readouterr().out
    assert statuses(tmp_path) == {FIRST: "COMPLETED", SECOND: "COMPLETED"}
    assert (tmp_path / "sims/sim_1001/datapoints/DP-NOCOND/history.csv").read_bytes() == first
    assert not (tmp_path / "archive").exists() or not any((tmp_path / "archive").iterdir())


def test_force_rerun_archives_record_and_folder_then_runs_again(tmp_path, capsys):
    workspace(tmp_path)
    assert cli(tmp_path) == 0
    old = RunRegistry(tmp_path).read()[0]
    manifest = (tmp_path / "runs.json").read_bytes()
    assert cli(tmp_path, "--force-rerun", "DP-NOCOND") == 0
    out = capsys.readouterr().out
    archives = list((tmp_path / "archive").iterdir())
    assert len(archives) == 1 and str(archives[0]) in out
    assert (archives[0] / "runs.json").read_bytes() == manifest
    assert (archives[0] / "sims/sim_1001/datapoints/DP-NOCOND/history.csv").is_file()
    new = RunRegistry(tmp_path).read()[0]
    assert new.status == "COMPLETED" and new.started_at > old.started_at
    history = RunRegistry(tmp_path).history()
    assert len(history) == 1
    assert history[0]["record"]["started_at"] == old.started_at
    assert history[0]["archive"] == archives[0].relative_to(tmp_path).as_posix()
    document = json.loads((tmp_path / "runs.json").read_bytes())
    assert set(document) == {"schema_version", "records", "history"}


def test_force_rerun_all_redoes_every_recorded_point(tmp_path):
    workspace(tmp_path)
    grow(tmp_path)
    assert cli(tmp_path) == 0
    assert cli(tmp_path, "--force-rerun-all") == 0
    assert len(RunRegistry(tmp_path).history()) == 2
    assert statuses(tmp_path) == {FIRST: "COMPLETED", SECOND: "COMPLETED"}


@pytest.mark.parametrize(
    "options",
    [
        ["--force-rerun", "DP-ABSENT"],
        ["--force-rerun-all", "--resume"],
        ["--force-rerun-all", "--force-rerun", "DP-NOCOND"],
    ],
)
def test_bad_rerun_requests_refuse_before_any_change(tmp_path, capsys, options):
    workspace(tmp_path)
    assert cli(tmp_path) == 0
    before = tree(tmp_path)
    assert cli(tmp_path, *options) == 2
    assert "force-rerun" in capsys.readouterr().err
    assert before == tree(tmp_path)


def test_ambiguous_point_name_needs_its_pol(tmp_path, capsys):
    workspace(tmp_path)
    grow(tmp_path)
    assert cli(tmp_path) == 0
    assert cli(tmp_path, "--force-rerun", "DP-NOCOND") == 2
    assert "POL/DP-name" in capsys.readouterr().err
    assert cli(tmp_path, "--force-rerun", SECOND) == 0
    assert len(RunRegistry(tmp_path).history()) == 1


def crashed(root):
    """Leave the first point RUNNING with a partial folder, as a killed run does."""
    workspace(root)
    plan = plan_workspace(root, "study")
    registry = RunRegistry(root)
    registry.record(plan)
    registry.transition(FIRST, "RUNNING")
    folder = root / plan.points[0].directory
    folder.mkdir(parents=True)
    (folder / "mock.in").write_text("partial", encoding="ascii")


def test_resume_reruns_a_running_point_whose_process_is_gone(tmp_path, monkeypatch):
    crashed(tmp_path)
    monkeypatch.setattr(rerun, "host_slot_free", lambda: True)
    assert cli(tmp_path) == 2, "RUNNING still refuses without --resume"
    assert cli(tmp_path, "--resume") == 0
    assert statuses(tmp_path) == {FIRST: "COMPLETED"}
    (archive,) = (tmp_path / "archive").iterdir()
    staged = archive / "sims/sim_1001/datapoints/DP-NOCOND/mock.in"
    assert staged.read_text() == "partial"
    assert RunRegistry(tmp_path).history()[0]["record"]["status"] == "RUNNING"


def test_resume_leaves_a_running_point_while_a_solver_holds_the_slot(tmp_path, monkeypatch, capsys):
    crashed(tmp_path)
    monkeypatch.setattr(rerun, "host_slot_free", lambda: False)
    before = tree(tmp_path)
    assert cli(tmp_path, "--resume") == 0
    assert "skipped: RUNNING and its process may be alive" in capsys.readouterr().out
    assert before == tree(tmp_path)
    assert cli(tmp_path, "--force-rerun", FIRST) == 0
    assert statuses(tmp_path) == {FIRST: "COMPLETED"}


def test_resume_skips_a_submitted_point(tmp_path, capsys):
    from test_p0000_item3f_registry import submitted

    submitted(tmp_path)
    assert cli(tmp_path, "--resume", "--local") == 0
    assert f"{FIRST}  skipped: SUBMITTED" in capsys.readouterr().out
    assert statuses(tmp_path) == {FIRST: "SUBMITTED"}


def test_archived_su2_point_keeps_its_linked_mesh(tmp_path):
    from test_p0000_su2_cgns import mesh_point

    mesh_point(tmp_path, ".su2")
    assert cli(tmp_path, "--local") == 0
    assert cli(tmp_path, "--local", "--force-rerun-all") == 0
    (archive,) = (tmp_path / "archive").iterdir()
    archived = next(archive.rglob("mesh.su2"))
    assert archived.samefile(tmp_path / "inputs/meshes/tiny.su2")
