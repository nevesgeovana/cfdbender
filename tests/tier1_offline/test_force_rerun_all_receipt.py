"""--force-rerun-all after a grouped submission needs no new plan; a real change still does."""

from __future__ import annotations

from test_p0000_item3f_plan import cli
from test_submit_batch import planned, setup

from cfdbender.core.registry import RunRegistry

WARNING = "a job of the earlier submission may still be running"


def first_attempt(tmp_path, capsys, n=1):
    setup(tmp_path)
    planned(tmp_path, capsys, n=n)
    assert cli(tmp_path, "run", "--batch", str(n)) == 0
    for record in RunRegistry(tmp_path).read():
        folder = tmp_path / record.point.directory
        (folder / "stdout.log").write_text("earlier\n", encoding="ascii")
        (folder / "job.exit").write_text("137\n", encoding="ascii")
    capsys.readouterr()


def test_forced_rerun_of_a_batch_needs_no_new_plan(tmp_path, capsys):
    first_attempt(tmp_path, capsys)
    assert cli(tmp_path, "run", "--batch", "1", "--force-rerun-all") == 0
    out = capsys.readouterr()
    assert (out.out + out.err).count(WARNING) == 3
    assert (tmp_path / "sims/batch/study_b1").is_dir() and (
        tmp_path / "sims/batch/study_b2"
    ).is_dir()
    records = RunRegistry(tmp_path).read()
    assert {r.status for r in records} == {"SUBMITTED"}
    assert {r.submission["job"]["label"] for r in records} == {"study_b2"}
    (archive,) = (tmp_path / "archive").iterdir()
    for record in records:
        old = archive / record.point.directory
        assert (old / "stdout.log").read_text(encoding="ascii") == "earlier\n"
        assert (old / "job.exit").read_text(encoding="ascii") == "137\n"
        assert not (tmp_path / record.point.directory / "job.exit").exists()
        assert not (tmp_path / record.point.directory / "stdout.log").exists()


def test_without_the_flag_the_renumbered_batch_still_asks_for_a_plan(tmp_path, capsys):
    first_attempt(tmp_path, capsys)
    assert cli(tmp_path, "run", "--batch", "1") == 2
    assert "plan it again" in capsys.readouterr().err


def test_a_real_change_of_grouping_still_refuses_under_the_flag(tmp_path, capsys):
    first_attempt(tmp_path, capsys, n=1)
    assert cli(tmp_path, "run", "--batch", "2", "--force-rerun-all") == 2
    assert "plan" in capsys.readouterr().err
    assert {r.status for r in RunRegistry(tmp_path).read()} == {"SUBMITTED"}
    assert not (tmp_path / "archive").exists()


def test_a_folder_without_a_record_still_refuses_under_the_flag(tmp_path, capsys):
    first_attempt(tmp_path, capsys)
    registry = RunRegistry(tmp_path)
    point = registry.read()[0].point
    registry.archive((point.point_id,))
    stray = tmp_path / point.directory
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "kept.txt").write_text("unknown data", encoding="ascii")
    assert cli(tmp_path, "plan", "--batch", "1") == 0
    capsys.readouterr()
    assert cli(tmp_path, "run", "--batch", "1", "--force-rerun-all") == 2
    assert "existing output; refusing overwrite" in capsys.readouterr().err
    assert (stray / "kept.txt").read_text(encoding="ascii") == "unknown data"
