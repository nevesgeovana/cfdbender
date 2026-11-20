"""A job that ended without job.exit is declared with ``collect --ended``, then continued.

A cluster job cancelled or killed before SU2 wrote its history (or a point of a batch that
never started) leaves its point SUBMITTED with no job.exit, which a plain collect cannot tell
from a queued job. ``cfd-matrix collect <matrix> --ended POINT`` seals it COMPLETED failed,
keeping its files; ``run --continue`` then continues it from its latest restart pair, and a
point with no pair runs again (REV080A-003). A job killed while SU2 marched has its history,
and a plain collect seals it failed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

from test_080_continuation import TWO, cont_cluster, finished, grow, kill_after
from test_p0000_item3f_plan import profile
from test_su2_time_restart import CLOCK, FAKE, call, keys, workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.solvers.su2._time_history import time_history

EVERY = CLOCK + " / RESTART_EVERY_REV: 0.25"
ENDED = "job ended without job.exit (declared with --ended)"


def quiet(argv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(argv)


def cluster_profile(root):
    path = profile(
        root,
        fields={"name": "{job_name}", "working_directory": "{workdir}", "config": "{config_path}"},
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.index("[builds]")], encoding="utf-8")


def su2(folder):
    with (folder / "stdout.log").open("wb") as log:
        subprocess.run([sys.executable, str(FAKE), "su2.cfg"], cwd=folder, stdout=log, check=True)


def killed(root):
    """A submitted point whose job SU2 ran in and the walltime killed after iteration 27."""
    workspace(root, EVERY, restart=False)
    cluster_profile(root)
    assert quiet(["plan", "ta", "--workspace", str(root)]) == 0
    assert quiet(["run", "ta", "--workspace", str(root)]) == 0
    record = RunRegistry(root).read()[-1]
    assert record.status == "SUBMITTED", record.error
    folder = root / record.point.directory
    su2(folder)
    kill_after(folder, 27)
    for path in folder.glob("*.vtu"):
        path.unlink()
    return record, folder


def test_rev080a_003_a_job_ended_before_its_history_is_declared_and_continued(tmp_path):
    """A continuation job cancelled before SU2 wrote anything stays SUBMITTED until declared."""
    record, folder = finished(tmp_path, EVERY)
    cluster_profile(tmp_path)
    grow(tmp_path, TWO + " / RESTART_EVERY_REV: 0.25")
    assert cont_cluster(tmp_path) == 0
    # The job never runs: a plain collect cannot tell it from a job still queued.
    assert quiet(["collect", "ta", "--workspace", str(tmp_path)]) == 3
    assert RunRegistry(tmp_path).read()[-1].status == "SUBMITTED"
    point_id = RunRegistry(tmp_path).read()[-1].point_id
    before = sorted(p.name for p in folder.iterdir())
    quiet(["collect", "ta", "--ended", point_id, "--workspace", str(tmp_path)])
    sealed = RunRegistry(tmp_path).read()[-1]
    assert (sealed.status, sealed.classification) == ("COMPLETED", "failed")
    assert ENDED in (sealed.error or "")
    assert sorted(p.name for p in folder.iterdir()) == before
    summary = json.loads((tmp_path / "post/ta/campaign_summary.json").read_text())
    assert [row["classification"] for row in summary["points"]] == ["failed"]
    assert cont_cluster(tmp_path) == 0
    assert keys(folder / "su2.cfg")["RESTART_ITER"] == "36"
    su2(folder)
    assert quiet(["collect", "ta", "--workspace", str(tmp_path)]) == 0
    done = RunRegistry(tmp_path).read()[-1]
    assert done.status == "COMPLETED" and done.classification != "failed", done.error
    assert list(time_history(folder).steps) == list(range(1, 73))


def test_rev080a_003_a_job_killed_while_marching_is_collected_failed_and_continued(tmp_path):
    """SU2 wrote its history before the kill: a plain collect seals it failed, no --ended."""
    record, folder = killed(tmp_path)
    assert quiet(["collect", "ta", "--workspace", str(tmp_path)]) != 0
    sealed = RunRegistry(tmp_path).read()[-1]
    assert (sealed.status, sealed.classification) == ("COMPLETED", "failed")
    # The same REVOLUTIONS: the point continues from its latest periodic pair, 26 and 27.
    assert cont_cluster(tmp_path) == 0
    cfg = keys(folder / "su2.cfg")
    assert (cfg["RESTART_ITER"], cfg["TIME_ITER"]) == ("28", "36")
    su2(folder)
    assert quiet(["collect", "ta", "--workspace", str(tmp_path)]) == 0
    done = RunRegistry(tmp_path).read()[-1]
    assert done.status == "COMPLETED" and done.classification != "failed", done.error
    assert list(time_history(folder).steps) == list(range(1, 37))


def test_rev080a_003_ended_refuses_a_point_that_is_not_submitted(tmp_path, capsys):
    record, folder = killed(tmp_path)
    assert quiet(["collect", "ta", "--ended", record.point_id, "--workspace", str(tmp_path)])
    capsys.readouterr()
    before = (tmp_path / "runs.json").read_bytes()
    assert quiet(["collect", "ta", "--ended", record.point_id, "--workspace", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "--ended" in err and "COMPLETED" in err
    assert (tmp_path / "runs.json").read_bytes() == before


def test_rev080a_003_ended_refuses_a_point_whose_job_wrote_job_exit(tmp_path, capsys):
    record, folder = killed(tmp_path)
    (folder / "job.exit").write_text("0\n")
    before = (tmp_path / "runs.json").read_bytes()
    assert quiet(["collect", "ta", "--ended", record.point_id, "--workspace", str(tmp_path)]) == 2
    assert "job.exit" in capsys.readouterr().err
    assert (tmp_path / "runs.json").read_bytes() == before


def test_rev080a_003_ended_refuses_an_unknown_point(tmp_path, capsys):
    killed(tmp_path)
    assert quiet(["collect", "ta", "--ended", "7002/DP-NOPE", "--workspace", str(tmp_path)]) == 2
    assert "7002/DP-NOPE" in capsys.readouterr().err


def test_rev080a_003_continue_without_a_restart_pair_names_the_files(tmp_path, capsys):
    record, folder = killed(tmp_path)
    for path in folder.glob("restart_flow_*.dat"):
        path.unlink()
    quiet(["collect", "ta", "--ended", record.point_id, "--workspace", str(tmp_path)])
    capsys.readouterr()
    assert cont_cluster(tmp_path) == 2
    err = capsys.readouterr().err
    assert "restart_flow_00034.dat" in err and "restart_flow_00035.dat" in err
    assert not (folder / "continued").exists()


def test_rev080a_003_a_batch_point_never_started_is_sealed_and_runs_again(tmp_path):
    from test_batch_per_polar import setup
    from test_p0000_item3f_plan import cli

    setup(tmp_path)
    assert cli(tmp_path, "plan", "--batch-per-polar") == 0
    assert cli(tmp_path, "run", "--batch-per-polar") == 0
    point = next(r for r in RunRegistry(tmp_path).read() if r.point.pol == "1002")
    assert point.status == "SUBMITTED"
    assert cli(tmp_path, "collect", "--ended", point.point_id) in (0, 1, 2)
    sealed = next(r for r in RunRegistry(tmp_path).read() if r.point_id == point.point_id)
    assert (sealed.status, sealed.classification) == ("COMPLETED", "failed")
    assert ENDED in (sealed.error or "")
    # The profile holds batch fields only, so the point runs again as a grouped job.
    assert cli(tmp_path, "plan", "--batch-per-polar", "--sims", "1002") == 0
    assert cli(tmp_path, "run", "--batch-per-polar", "--sims", "1002", "--force-rerun-all") == 0
    again = next(r for r in RunRegistry(tmp_path).read() if r.point_id == point.point_id)
    assert again.status == "SUBMITTED"
