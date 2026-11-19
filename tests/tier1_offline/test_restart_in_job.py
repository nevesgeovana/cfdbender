"""A RESTART whose source runs earlier in the same batch job is staged by the job script."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import test_runtypes_b1_workflows as rt
from test_p0000_item3f_plan import profile
from test_submit_batch import FIELDS

from cfdbender.core.adapter import load_point
from cfdbender.core.registry import RunRegistry
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

FAKE = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2_transient.py"
HEADER = rt.HEADER + " | RESTART"
CONDITION = "MACH:0.0529, ALPHA:0, BETA:0, ADVANCE_RATIO:0.4"
CLOCK = "DELTA_THETA: 10 / REVOLUTIONS: 1 / LAST_REVS_AVG: 0.25"
WRAPPER = """#!/bin/sh
# Stand-in for SU2_CFD: the transient fake, with two switches for the failure tests.
case "$PWD" in
  *sim_7001*)
    [ -f ../../../../fail-source ] && exit 3
    "{python}" "{fake}" "$@" || exit $?
    [ -f ../../../../drop-restart ] && rm -f restart_flow.dat
    exit 0;;
esac
exec "{python}" "{fake}" "$@"
"""


def row(pol, workflow, var="-", restart="-", condition=CONDITION, sweep="-", extra=""):
    return (
        f" {pol} | 1 | rotor | {condition} | {sweep} | rotor.su2 | 001 | 001 | 001 | - | - "
        f"| {workflow} | {var}{extra} | {restart} "
    )


def build(root, rows, *, setup=""):
    """A workspace with an HPC profile for grouped jobs and the matrix ``m`` of ``rows``."""
    wrapper = root / "tools/su2.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        WRAPPER.format(python=Path(sys.executable).as_posix(), fake=FAKE.as_posix()), newline="\n"
    )
    rt.workspace(
        root,
        setup='RESTART_FILENAME = "restart_flow.dat"\n'
        + setup
        + f'[solver]\nroute = "executable"\nexecutable = {json.dumps(wrapper.as_posix())}\n',
    )
    path = profile(
        root, fields={k: v for k, v in FIELDS.items() if k in ("mode", "command", "name", "sims")}
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.index("[builds]")], encoding="utf-8")
    write_matrix(root, rows)


def write_matrix(root, rows):
    path = root / "inputs/matrices/m.fs"
    path.write_text("\n".join([HEADER, "-" * 60, *rows]) + "\n", encoding="ascii")


def source_id(root, rows):
    """The point id of the first row, planned on its own."""
    write_matrix(root, rows[:1])
    return plan_workspace(root, "m").points[0].point_id


def steady_then_unsteady(root, **kwargs):
    source = source_id_for(root)
    rows = [
        row(7001, "qsteady_rotor"),
        row(7002, "unsteady_rotor", CLOCK, source),
    ]
    write_matrix(root, rows)
    return source, rows


def source_id_for(root):
    write_matrix(root, [row(7001, "qsteady_rotor")])
    return plan_workspace(root, "m").points[0].point_id


def batch(root, *options, command="run"):
    return cli_m(root, command, "--batch", "1", *options)


def cli_m(root, command, *options):
    from cfdbender.matrix.cli import main

    try:
        return main([command, "m", "--workspace", str(root), *options])
    except SystemExit as error:
        return int(error.code or 0)


def script(root):
    (path,) = (root / "sims/batch").glob("*/BATCH-*.sh")
    return path


def run_script(root, search_path=None):
    """Run the job script; ``search_path`` replaces PATH for the whole script."""
    path = script(root)
    env = dict(os.environ)
    if search_path is not None:
        env["PATH"] = search_path
    return subprocess.run(["sh", path.name], cwd=path.parent, env=env, capture_output=True)


def shell_only_path():
    """A PATH holding the shell's own folder and no python, or None when one is there."""
    sh_folder = str(Path(shutil.which("sh") or "").parent)
    found = [shutil.which(name, path=sh_folder) for name in ("python", "python3", "py")]
    return None if any(found) else sh_folder


needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")


def point_of(root, pol):
    return next(r for r in RunRegistry(root).read() if r.point.pol == str(pol))


def test_the_plan_shows_the_dependency_and_the_run_is_not_refused(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source, _ = steady_then_unsteady(tmp_path)
    assert batch(tmp_path, command="plan") == 0
    out = capsys.readouterr().out
    assert "7002/" in out and f"RESTART {source}: fed inside the job" in out
    assert batch(tmp_path) == 0
    records = {r.point.pol: r for r in RunRegistry(tmp_path).read()}
    assert {r.status for r in records.values()} == {"SUBMITTED"}
    text = script(tmp_path).read_text()
    assert text.index("# " + source) < text.index("RESTART from " + source)
    assert "cp " in text and "restart-input_00001.dat" in text
    folder = tmp_path / records["7002"].point.directory
    pending = json.loads((folder / "restart-pending.json").read_text())
    assert pending["source"] == source
    assert sorted(pending["stage"]) == ["restart-input_00000.dat", "restart-input_00001.dat"]
    assert not list(folder.glob("restart-input_*.dat")), "nothing is staged before the source ends"
    assert (folder / "su2.cfg").read_text().count("RESTART_SOL= YES") == 1


@needs_sh
def test_the_script_stages_after_the_source_and_collect_reads_both(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    steady_then_unsteady(tmp_path)
    assert batch(tmp_path) == 0
    cfg = tmp_path / point_of(tmp_path, 7002).point.directory / "su2.cfg"
    before = hashlib.sha256(cfg.read_bytes()).hexdigest()
    done = run_script(tmp_path)
    assert done.returncode == 0, done.stderr.decode()
    folder = cfg.parent
    for name in ("restart-input_00000.dat", "restart-input_00001.dat"):
        assert (folder / name).read_bytes() == b"steady solution"
    sidecar = folder / "restart-staged.sha256"
    if shutil.which("sha256sum"):
        digests = {line.split()[0] for line in sidecar.read_text().splitlines()}
        assert digests == {hashlib.sha256(b"steady solution").hexdigest()}
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == before, "su2.cfg is never edited"
    submitted = point_of(tmp_path, 7002).submission["config_sha256"]
    assert submitted == before
    for pol in (7001, 7002):
        assert (
            tmp_path / point_of(tmp_path, pol).point.directory / "job.exit"
        ).read_text() == "0\n"
    assert cli_m(tmp_path, "collect") != 2
    classes = {r.point.pol: r.classification for r in RunRegistry(tmp_path).read()}
    assert "failed" not in classes.values(), classes


@needs_sh
def test_a_failed_source_leaves_the_dependent_unrun_and_the_batch_goes_on(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    write_matrix(
        tmp_path,
        [
            row(7001, "qsteady_rotor"),
            row(7002, "unsteady_rotor", CLOCK, source),
            row(7003, "qsteady_rotor"),
        ],
    )
    assert batch(tmp_path) == 0
    (tmp_path / "fail-source").write_text("x")
    assert run_script(tmp_path).returncode == 1
    dependent = tmp_path / point_of(tmp_path, 7002).point.directory
    assert (dependent / "job.exit").read_text() == "97\n"
    assert (dependent / "stderr.log").read_text().count("\n") == 1
    assert (
        f"RESTART {source}: the source ended with status 3"
        in (dependent / "stderr.log").read_text()
    )
    assert not (dependent / "stdout.log").exists() or not (dependent / "stdout.log").read_text()
    assert (tmp_path / point_of(tmp_path, 7003).point.directory / "job.exit").read_text() == "0\n"


@needs_sh
def test_a_missing_restart_file_is_a_distinct_status_with_one_line(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    steady_then_unsteady(tmp_path)
    assert batch(tmp_path) == 0
    (tmp_path / "drop-restart").write_text("x")
    run_script(tmp_path)
    dependent = tmp_path / point_of(tmp_path, 7002).point.directory
    assert (dependent / "job.exit").read_text() == "98\n"
    line = (dependent / "stderr.log").read_text()
    assert line.count("\n") == 1 and "wrote no restart file" in line
    assert not list(dependent.glob("restart-input_*.dat"))


def test_the_plan_time_rendering_equals_the_one_read_from_a_finished_source(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source, rows = steady_then_unsteady(tmp_path)
    write_matrix(tmp_path, rows[:1])
    assert batch(tmp_path) == 0
    assert run_script(tmp_path).returncode == 0
    assert cli_m(tmp_path, "collect") != 2
    assert RunRegistry(tmp_path).read()[0].status == "COMPLETED"
    write_matrix(tmp_path, rows)
    plan = plan_workspace(tmp_path, "m")
    src_point = next(p for p in plan.points if p.point_id == source)
    dep = next(p for p in plan.points if p.pol == "7002")
    recorded = RunRegistry(tmp_path).read()[0]
    adapter = load_solver("su2")
    disk = adapter.translate(
        load_point(
            tmp_path,
            replace(dep, restart_directory=recorded.point.directory),
            solver="su2",
        )
    )
    rendered = adapter.translate(load_point(tmp_path, src_point, solver="su2"))
    predicted = adapter.translate(
        load_point(
            tmp_path,
            replace(dep, restart_in_job=True, restart_directory=src_point.directory),
            solver="su2",
            restart_source_cfg=rendered,
        )
    )
    assert predicted == disk
    assert "RESTART_SOL= YES" in disk


def test_a_dependent_before_its_source_is_reordered_within_the_job(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    write_matrix(tmp_path, [row(7002, "unsteady_rotor", CLOCK, source), row(7001, "qsteady_rotor")])
    assert batch(tmp_path) == 0
    text = script(tmp_path).read_text()
    assert text.index("# " + source) < text.index("RESTART from " + source)


def test_one_source_feeds_every_point_of_a_sweep_row(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    sweep = "MACH:0.0529, ALPHA:sweep, BETA:0, ADVANCE_RATIO:0.4"
    write_matrix(
        tmp_path,
        [row(7001, "qsteady_rotor"), row(7002, "unsteady_rotor", CLOCK, source, sweep, "0, 2")],
    )
    assert batch(tmp_path) == 0
    fed = [r for r in RunRegistry(tmp_path).read() if r.point.pol == "7002"]
    assert len(fed) == 2 and all(r.point.restart_in_job for r in fed)
    assert script(tmp_path).read_text().count("restart-staged.sha256") == 2


def test_it_still_refuses_when_the_source_is_not_in_the_same_job(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source, _ = steady_then_unsteady(tmp_path)
    assert cli_m(tmp_path, "plan", "--hpc", "h001") == 0
    assert cli_m(tmp_path, "run", "--hpc", "h001") == 2
    assert f"RESTART {source}: no recorded point; run the source point first" in (
        capsys.readouterr().err
    )
    assert (
        batch(
            tmp_path,
            "--points",
            "7002/" + plan_workspace(tmp_path, "m").points[1].point_id.split("/", 1)[1],
        )
        == 2
    )
    assert f"RESTART {source}: no recorded point; run the source point first" in (
        capsys.readouterr().err
    )


def test_it_refuses_two_batches_that_split_the_source_from_its_dependent(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    steady_then_unsteady(tmp_path)
    assert cli_m(tmp_path, "plan", "--batch", "2") == 0
    assert cli_m(tmp_path, "run", "--batch", "2") == 2
    assert "RESTART" in capsys.readouterr().err
    assert not (tmp_path / "runs.json").exists() or not RunRegistry(tmp_path).read()


def test_a_source_whose_restart_format_is_left_to_run_time_is_refused(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")], setup='OUTPUT_FILES = ["RESTART_ASCII"]\n')
    steady_then_unsteady(tmp_path)
    assert batch(tmp_path) == 2
    assert "add RESTART to the OUTPUT_FILES" in capsys.readouterr().err


def test_a_forced_rerun_of_both_rows_in_one_batch_feeds_again(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    steady_then_unsteady(tmp_path)
    assert batch(tmp_path) == 0
    assert batch(tmp_path, "--force-rerun-all") == 0
    dependent = point_of(tmp_path, 7002)
    assert dependent.status == "SUBMITTED" and dependent.point.restart_in_job
    assert dependent.point.restart_directory.startswith("sims/")
    pending = json.loads(
        (tmp_path / dependent.point.directory / "restart-pending.json").read_text()
    )
    assert not any("archive" in path for path in pending["stage"].values())


@needs_sh
def test_the_staging_comes_before_a_per_point_wrapper(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    wrap = tmp_path / "wrap.sh"
    wrap.write_text("#!/bin/sh\necho wrapped > wrapped.txt\n", newline="\n")
    write_matrix(
        tmp_path,
        [
            row(7001, "qsteady_rotor"),
            row(7002, "unsteady_rotor", CLOCK + " / hpc.application_id: wrap.sh", source),
        ],
    )
    assert batch(tmp_path) == 0
    text = script(tmp_path).read_text()
    block = text[text.index("RESTART from " + source) :]
    assert block.index("cp ") < block.index("wrap.sh")


def test_points_that_feed_each_other_are_refused_by_name(tmp_path):
    from cfdbender.matrix.batch import in_job_order

    build(tmp_path, [row(7001, "qsteady_rotor")])
    source, rows = steady_then_unsteady(tmp_path)
    first, second = plan_workspace(tmp_path, "m").points
    crossed = [
        (replace(first, restart_from=second.point_id), None),
        (replace(second, restart_from=first.point_id), None),
    ]
    with pytest.raises(ValueError, match="feed each other"):
        in_job_order(crossed, frozenset())
