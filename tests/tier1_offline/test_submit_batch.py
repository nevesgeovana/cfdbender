"""cfd-matrix run --batch N: rows split into scheduler jobs, one generated script per batch."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_p0000_item3f_collect import outputs
from test_p0000_item3f_plan import cli, prepared, profile
from test_p053_rowcl_cluster import matrix

from cfdbender.core.matrix import parse_matrix
from cfdbender.core.registry import RunRegistry
from cfdbender.hpc._batch_split import RowUnit, split_rows
from cfdbender.matrix.batch import STATUS_FILE, batch_count

FIELDS = {
    "mode": "custom",
    "command": "{script_path}",
    "name": "{job_name}",
    "queue": "{cluster}",
    "cpus": "{ncpus}",
    "sims": "{sims}",
    "batch": "{batch_id}",
}
APP = 'hpc.application_id: "tools/stub.sh"'
STUB = """#!/bin/sh
echo "$PWD $1" >> "$(dirname "$0")/trace.txt"
case "$PWD" in *sim_1002*) exit 3;; esac
exit 0
"""


def units(*weights, group=("g",)):
    return [RowUnit(str(1001 + i), i + 1, group, w) for i, w in enumerate(weights)]


def setup(tmp_path, cells=None):
    prepared(tmp_path, fields=FIELDS)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/stub.sh").write_text(STUB, newline="\n")
    matrix(tmp_path, cells or [f"cluster: one / {APP}"] * 3)


def planned(tmp_path, capsys, n=2):
    assert cli(tmp_path, "plan", "--batch", str(n)) == 0
    return capsys.readouterr().out


def script_of(tmp_path, name="BATCH-1001-1003.sh"):
    return tmp_path / "sims/batch/study_b1" / name


def test_batch_split_contiguous_optimal_and_never_cuts_a_row():
    runs, warnings = split_rows(units(1, 1, 1, 1, 1), 2)
    assert [[u.sim_id for u in run] for run in runs] == [["1001", "1002"], ["1003", "1004", "1005"]]
    assert warnings == []
    runs, _ = split_rows(units(5, 1, 1, 1), 2)
    assert [[u.sim_id for u in run] for run in runs] == [["1001"], ["1002", "1003", "1004"]]


def test_batch_split_remainder_and_warnings():
    runs, warnings = split_rows(units(1, 1), 5)
    assert len(runs) == 2 and "asks for more jobs than the 2 row(s)" in warnings[0]
    mixed = [*units(1, 1, group=("a",)), RowUnit("2001", 3, ("b",), 1)]
    runs, warnings = split_rows(mixed, 1)
    assert len(runs) == 2 and "fewer jobs than the 2 group(s)" in warnings[0]


def test_batch_count_is_a_whole_number_of_at_least_one():
    assert batch_count("3") == 3
    for bad in ("x", "0", "-2"):
        with pytest.raises(Exception, match="batch"):
            batch_count(bad)


def test_batch_plan_prints_one_descriptor_per_batch_and_records_grouping(tmp_path, capsys):
    setup(tmp_path)
    out = planned(tmp_path, capsys)
    assert out.count("would submit:") == 2 and out.count('"mode": "custom"') == 2
    assert "batch 2: study_b2 holds row(s) 1002, 1003" in out
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    jobs = receipt["grouping"]["jobs"]
    assert receipt["grouping"]["requested"] == 2
    assert [j["sims"] for j in jobs] == [["1001"], ["1002", "1003"]]
    assert jobs[0]["dir"] == "sims/batch/study_b1"
    assert not (tmp_path / "sims").exists()


def test_batch_descriptor_points_at_the_script_relative_to_the_job_folder(tmp_path, capsys):
    setup(tmp_path)
    out = planned(tmp_path, capsys, n=1)
    assert '"command": "BATCH-1001-1003.sh"' in out
    assert '"batch": 1' in out and '"sims": "1001,1002,1003"' in out


def test_batch_run_writes_the_script_and_the_records(tmp_path, capsys):
    setup(tmp_path)
    planned(tmp_path, capsys, n=1)
    assert cli(tmp_path, "run", "--batch", "1") == 0
    text = script_of(tmp_path).read_bytes().decode()
    assert "\r" not in text and text.startswith("#!/bin/sh\n")
    assert "set -u" in text and text.rstrip().endswith('exit "$failed"')
    order = [text.index(f"sim_{pol}") for pol in ("1001", "1002", "1003")]
    assert order == sorted(order)
    assert "( cd ../../sim_1001/datapoints/" in text and "/tmp" not in text
    assert text.count("> stdout.log 2> stderr.log") == 3
    assert "sh ../../../../tools/stub.sh" in text or "sh ../../../../../tools/stub.sh" in text
    records = RunRegistry(tmp_path).read()
    assert {r.submission["job"]["label"] for r in records} == {"study_b1"}
    assert records[0].submission["descriptor"].startswith("sims/batch/study_b1/")
    assert records[0].submission["job_end_files"][0].endswith(STATUS_FILE)
    assert (tmp_path / "sims/batch/study_b1/submit.json").is_file()


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_batch_script_runs_every_run_continues_after_failure_and_collect_reads_status(
    tmp_path, capsys
):
    setup(tmp_path)
    planned(tmp_path, capsys, n=1)
    assert cli(tmp_path, "run", "--batch", "1") == 0
    script = script_of(tmp_path)
    done = subprocess.run(["sh", script.name], cwd=script.parent, capture_output=True, check=False)
    assert done.returncode == 1
    trace = (tmp_path / "tools/trace.txt").read_text().splitlines()
    assert len(trace) == 3 and "sim_1002" in trace[1] and trace[2].count("sim_1003") == 1
    assert all(line.endswith("mock.in") for line in trace)
    records = RunRegistry(tmp_path).read()
    codes = {r.point.pol: (tmp_path / r.point.directory / STATUS_FILE).read_text() for r in records}
    assert codes == {"1001": "0\n", "1002": "3\n", "1003": "0\n"}
    for record in records:
        outputs(tmp_path, record.point)
    assert cli(tmp_path, "collect") != 2
    final = {r.point.pol: r for r in RunRegistry(tmp_path).read()}
    assert final["1002"].classification == "failed"
    assert "exit status 3" in final["1002"].error
    assert final["1001"].classification != "failed" and final["1003"].classification != "failed"


def test_batch_rows_disagreeing_on_a_job_level_value_are_refused_naming_both(tmp_path, capsys):
    setup(
        tmp_path,
        [
            f"cluster: one / {APP} / hpc.mode: solver",
            f"cluster: one / {APP} / hpc.mode: custom",
        ],
    )
    assert cli(tmp_path, "plan", "--batch", "1") == 2
    error = capsys.readouterr().err
    assert (
        "rows 1001 and 1002 of batch study_b1 disagree on hpc.mode ('solver' and 'custom')" in error
    )
    assert cli(tmp_path, "plan", "--batch", "2") == 0


def test_batch_rows_may_differ_in_their_application(tmp_path, capsys):
    setup(tmp_path, [f"cluster: one / {APP}", "cluster: one / hpc.application_id: other-tool"])
    assert cli(tmp_path, "run", "--batch", "1") == 2
    assert "plan" in capsys.readouterr().err
    planned(tmp_path, capsys, n=1)
    assert cli(tmp_path, "run", "--batch", "1") == 0
    text = script_of(tmp_path, "BATCH-1001-1002.sh").read_text()
    assert "other-tool " in text and "stub.sh" in text


def test_batch_needs_a_profile_a_scheduler_and_no_warm_start(tmp_path, capsys):
    setup(tmp_path)
    assert cli(tmp_path, "plan", "--batch", "1", "--local") == 2
    assert "cannot be combined with --local" in capsys.readouterr().err
    planned(tmp_path, capsys, n=1)
    assert cli(tmp_path, "run", "--batch", "1", "--restart") == 2
    assert "does not take --restart" in capsys.readouterr().err
    assert cli(tmp_path, "run", "--batch", "2") == 2
    assert "made for --batch 1, not 2" in capsys.readouterr().err
    (tmp_path / "inputs/hpc/h001.toml").rename(tmp_path / "inputs/h001.bak")
    matrix(tmp_path, ["-"] * 3)
    assert cli(tmp_path, "plan", "--batch", "1") == 2
    assert "no HPC profile is selected" in capsys.readouterr().err


def test_batch_run_needs_a_batch_plan(tmp_path, capsys):
    setup(tmp_path)
    profile(tmp_path, fields={"mode": "solver", "queue": "{cluster}", "name": "{job_name}"})
    assert cli(tmp_path, "plan") == 0
    capsys.readouterr()
    assert cli(tmp_path, "run", "--batch", "1") == 2
    assert "plan holds no grouping" in capsys.readouterr().err


def test_without_the_flag_plan_and_run_are_unchanged(tmp_path, capsys):
    setup(tmp_path, ["cluster: one"] * 2)
    profile(tmp_path, fields={"mode": "solver", "queue": "{cluster}", "name": "{job_name}"})
    assert cli(tmp_path) == 0
    out = capsys.readouterr().out
    assert out.count("would submit:") == 2 and "batch 1" not in out
    receipt = json.loads((tmp_path / "post/study/plan.json").read_text())
    assert "grouping" not in receipt
    assert cli(tmp_path, "run") == 0
    for record in RunRegistry(tmp_path).read():
        assert "job" not in record.submission
        assert record.submission["descriptor"].endswith("submit.json")
    assert not (tmp_path / "sims/batch").exists()


def test_a_quoted_slash_is_kept_and_an_unquoted_cell_splits_as_before():
    head = b"POL|RUN|SETUP|REF|PPROC|VAR_NAMES_VALUES\n"
    row = parse_matrix(head + b"1|1|001|001|001|a: 1 / b: 2 / c: [1, 2]\n", Path("m.fs"))[0]
    assert dict(row.variables) == {"a": 1, "b": 2, "c": [1, 2]}
    row = parse_matrix(head + b'1|1|001|001|001|a: "x/y" / b: 2\n', Path("m.fs"))[0]
    assert dict(row.variables) == {"a": "x/y", "b": 2}
