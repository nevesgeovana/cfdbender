"""Edges of the in-job RESTART staging: the interpreter, quoting, order, collect and receipts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest
from test_restart_in_job import (
    CLOCK,
    batch,
    build,
    cli_m,
    needs_sh,
    point_of,
    row,
    run_script,
    script,
    shell_only_path,
    source_id_for,
    write_matrix,
)

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.batch import STAGE_MARK, _dependent_block
from cfdbender.workspace.planning import plan_workspace


def three_rows(root):
    build(root, [row(7001, "qsteady_rotor")])
    source = source_id_for(root)
    write_matrix(
        root,
        [
            row(7001, "qsteady_rotor"),
            row(7002, "unsteady_rotor", CLOCK, source),
            row(7003, "qsteady_rotor"),
        ],
    )
    return source


def submit(root, *options):
    assert batch(root, *options, command="plan") == 0
    return batch(root, *options)


def exit_of(root, pol):
    return (root / point_of(root, pol).point.directory / "job.exit").read_text()


def test_the_script_names_no_python_and_needs_only_a_posix_shell(tmp_path):
    three_rows(tmp_path)
    assert submit(tmp_path) == 0
    text = script(tmp_path).read_text()
    assert "python" not in text.lower() and "stage_restart" not in text
    assert "CFDBENDER_PYTHON" not in text and "@@STAGE" not in text


@needs_sh
def test_the_script_runs_on_a_path_that_holds_no_python(tmp_path):
    search_path = shell_only_path()
    if search_path is None:
        pytest.skip("the shell's folder holds a python")
    three_rows(tmp_path)
    assert submit(tmp_path) == 0
    done = run_script(tmp_path, search_path)
    assert done.returncode == 0, done.stderr.decode()
    folder = tmp_path / point_of(tmp_path, 7002).point.directory
    for name in ("restart-input_00000.dat", "restart-input_00001.dat"):
        assert (folder / name).read_bytes() == b"steady solution"
    assert [exit_of(tmp_path, p) for p in (7001, 7002, 7003)] == ["0\n"] * 3


@needs_sh
def test_a_missing_source_file_is_status_98_naming_it_and_the_job_goes_on(tmp_path):
    three_rows(tmp_path)
    assert submit(tmp_path) == 0
    (tmp_path / "drop-restart").write_text("x")
    run_script(tmp_path)
    assert exit_of(tmp_path, 7002) == "98\n"
    log = (tmp_path / point_of(tmp_path, 7002).point.directory / "stderr.log").read_text()
    assert log.count("\n") == 1 and "restart_flow.dat" in log
    assert exit_of(tmp_path, 7003) == "0\n", "the next independent run still ran"


def test_a_source_with_an_ascii_restart_is_refused_naming_the_key(tmp_path, capsys):
    build(tmp_path, [row(7001, "qsteady_rotor")], setup="WRT_BINARY_RESTART = false\n")
    source = source_id_for(tmp_path)
    write_matrix(tmp_path, [row(7001, "qsteady_rotor"), row(7002, "unsteady_rotor", CLOCK, source)])
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path) == 2
    assert "set WRT_BINARY_RESTART to YES" in capsys.readouterr().err


@needs_sh
def test_a_forced_rerun_that_puts_the_source_first_is_not_a_new_grouping(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    assert submit(tmp_path) == 0
    assert run_script(tmp_path).returncode == 0
    assert cli_m(tmp_path, "collect") != 2
    write_matrix(tmp_path, [row(7002, "unsteady_rotor", CLOCK, source), row(7001, "qsteady_rotor")])
    assert batch(tmp_path, command="plan") == 0
    assert batch(tmp_path, "--force-rerun-all") == 0
    scripts = sorted((tmp_path / "sims/batch").glob("*/BATCH-*.sh"))
    newest = scripts[-1].read_text()
    assert newest.index("# " + source) < newest.index("RESTART from " + source)


@needs_sh
def test_the_97_note_survives_quotes_and_percent_signs_and_a_killed_source(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    plan = plan_workspace(tmp_path, "m")
    odd = "1/DP-a'b%s$x"
    point = replace(plan.points[0], restart_from=odd)
    block = _dependent_block(point, "dep", ["true"], "src")
    block = block.replace(STAGE_MARK.format(point_id=point.point_id), ":")
    (tmp_path / "dep").mkdir()
    (tmp_path / "src").mkdir()
    body = "failed=0\n" + block + "\nexit $failed\n"
    (tmp_path / "t.sh").write_text(body, newline="\n")
    assert subprocess.run(["sh", "-n", "t.sh"], cwd=tmp_path).returncode == 0
    (tmp_path / "src/job.exit").write_text("3\n")
    subprocess.run(["sh", "t.sh"], cwd=tmp_path, check=False)
    note = (tmp_path / "dep/stderr.log").read_text()
    assert note == f"RESTART {odd}: the source ended with status 3; this run was not started\n"
    assert (tmp_path / "dep/job.exit").read_text() == "97\n"
    (tmp_path / "src/job.exit").unlink()
    (tmp_path / "dep/job.exit").unlink()
    subprocess.run(["sh", "t.sh"], cwd=tmp_path, check=False)
    assert "ended with status missing;" in (tmp_path / "dep/stderr.log").read_text()
    assert (tmp_path / "dep/job.exit").read_text() == "97\n"


@needs_sh
def test_a_workspace_path_with_a_space_runs_the_whole_flow(tmp_path):
    root = tmp_path / "my work"
    root.mkdir()
    three_rows(root)
    assert submit(root) == 0
    assert run_script(root).returncode == 0
    assert [exit_of(root, p) for p in (7001, 7002, 7003)] == ["0\n"] * 3


@needs_sh
def test_the_staged_files_exist_when_a_per_point_wrapper_starts(tmp_path):
    build(tmp_path, [row(7001, "qsteady_rotor")])
    source = source_id_for(tmp_path)
    (tmp_path / "wrap.sh").write_text(
        "#!/bin/sh\nls restart-input_00000.dat restart-input_00001.dat > seen.txt 2>&1\n",
        newline="\n",
    )
    write_matrix(
        tmp_path,
        [
            row(7001, "qsteady_rotor"),
            row(7002, "unsteady_rotor", CLOCK + " / hpc.application_id: wrap.sh", source),
        ],
    )
    assert submit(tmp_path) == 0
    assert run_script(tmp_path).returncode == 0
    seen = (tmp_path / point_of(tmp_path, 7002).point.directory / "seen.txt").read_text()
    assert "No such file" not in seen and seen.count("restart-input_0000") == 2


@needs_sh
@pytest.mark.parametrize("why", ["source-failed", "staging-failed"])
def test_a_run_that_was_not_started_is_failed_with_its_reason_never_outstanding(tmp_path, why):
    three_rows(tmp_path)
    assert submit(tmp_path) == 0
    if why == "source-failed":
        (tmp_path / "fail-source").write_text("x")
        run_script(tmp_path)
        reason = "the source ended with status 3"
    else:
        (tmp_path / "drop-restart").write_text("x")
        run_script(tmp_path)
        reason = "wrote no restart file"
    code = cli_m(tmp_path, "collect")
    assert code != 3, "nothing may stay outstanding"
    record = point_of(tmp_path, 7002)
    assert record.status == "COMPLETED" and record.classification == "failed"
    assert "exit status" in record.error and reason in record.error
    assert record.error.count("\n") == 0
    from cfdbender.matrix.cli import main

    assert main(["status", "--workspace", str(tmp_path)]) in (0, 1)
    document = json.loads((tmp_path / "post/m/campaign_summary.json").read_text())
    states = {p["point_id"].split("/")[0]: p["classification"] for p in document["points"]}
    assert states["7002"] == "failed" and states["7003"] != "failed"
    assert all(r.status == "COMPLETED" for r in RunRegistry(tmp_path).read())
