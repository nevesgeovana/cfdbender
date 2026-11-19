"""CFD++ staging and execution with the builder's stand-ins, and the conformance
assertions of cfdbender.testing for the adapter.

The stand-in solver parses the generated input with the public parser and writes the
output files by the scenario named in the raw keyword ``zscenario``; the stand-in tool
partitions or asks for a stop. Nothing here runs the licensed solver.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from cfdpp_support import builder
from cfdpp_workspace import SETUP, STOP_TOOL, matrix, use_toy, with_keywords, workspace

import cfdbender.solvers.cfdpp as cfdpp
from cfdbender import testing
from cfdbender.core.adapter import load_point
from cfdbender.core.executables import bind_executable
from cfdbender.core.plan import PlanError, workspace_path
from cfdbender.core.registry import RegistryError, RunRegistry
from cfdbender.core.run import execute_plan
from cfdbender.solvers._host._execute import _NO_WINDOW
from cfdbender.solvers.cfdpp import _execute
from cfdbender.workspace.planning import plan_workspace

ONE = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:10", "-")


@pytest.fixture(autouse=True)
def _toy(monkeypatch):
    use_toy(monkeypatch)


def scenario(name: str, setup: str = SETUP, iterations: int | None = None) -> str:
    text = (
        setup
        if iterations is None
        else setup.replace("iterations = 200", f"iterations = {iterations}")
    )
    return with_keywords(f'{builder().SCENARIO_KEYWORD} = "{name}"\n', text)


def planned(root: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return plan_workspace(root, "study")


def points(root: Path):
    return [
        bind_executable(load_point(root, point, solver="cfdpp"), cfdpp)
        for point in planned(root).points
    ]


def make(tmp_path: Path, name: str = "ws", **kwargs):
    kwargs.setdefault("table", ONE)
    root = workspace(tmp_path / name, **kwargs)
    return root, points(root)[0]


def folder(point) -> Path:
    return workspace_path(point.root, point.point.directory)


# --- the steps ------------------------------------------------------------------------

ENTRY = {
    "serial": "/opt/solver/serial",
    "mpi": "/opt/solver/mpi",
    "mpi_launcher": ["mpirun", "-np", "{ncpus}"],
    "partitioner": ["/opt/solver/part", "pmetis", "{ncpus}"],
}


def test_cfdpp_execute_argv_is_serial_for_one_process():
    steps = _execute.argv(ENTRY, 1, partitioned=True)
    assert [(s.name, s.argv, s.log) for s in steps] == [
        ("solver", ("/opt/solver/serial",), "stdout.log")
    ]


def test_cfdpp_execute_argv_is_mpi_with_ncpus_filled_above_one():
    steps = _execute.argv(ENTRY, 4, partitioned=False)
    assert [(s.name, s.argv) for s in steps] == [
        ("partition", ("/opt/solver/part", "pmetis", "4")),
        ("solver", ("mpirun", "-np", "4", "/opt/solver/mpi")),
    ]
    assert [s.name for s in _execute.argv(ENTRY, 4, partitioned=True)] == ["solver"]


def test_cfdpp_execute_a_missing_field_names_it():
    with pytest.raises(ValueError) as caught:
        _execute.argv({"serial": "x"}, 2, partitioned=True, build="1.0")
    assert str(caught.value) == (
        "CFD++: NCPUS 2 needs the mpi_launcher field of build '1.0' in inputs/executables.toml"
    )


def test_cfdpp_execute_a_missing_mpi_field_is_refused_at_planning(tmp_path):
    root = workspace(tmp_path / "ws", table=matrix("MACH:0.15, REmi:6, TK:300, ALPHA:0", "-",
                                                   ncpus=2))  # fmt: skip
    path = root / "inputs/executables.toml"
    path.write_text(
        "\n".join(line for line in path.read_text().splitlines() if not line.startswith("mpi ")),
        encoding="ascii",
    )
    with pytest.raises(PlanError, match="NCPUS 2 needs the mpi field of build '1.0'"):
        planned(root)


def test_cfdpp_execute_the_job_script_holds_the_same_steps():
    steps = _execute.argv(ENTRY, 4, partitioned=False)
    assert _execute.job_script(steps, 4) == (
        "#!/bin/sh\n"
        "# Written by cfdbender: the ordered steps of one CFD++ point.\n"
        'cd "$(dirname "$0")" || exit 1\n'
        "set -e\n"
        "finish() {\n"
        "    code=$?\n"
        '    printf \'{"exit_code": %s}\\n\' "$code" > execution.json.tmp\n'
        "    mv execution.json.tmp execution.json\n"
        "    touch job.done\n"
        '    exit "$code"\n'
        "}\n"
        "trap finish 0\n"
        "/opt/solver/part pmetis 4 > log.partition 2>&1\n"
        "test -f mcpusin.bin.4 || exit 1\n"
        "mpirun -np 4 /opt/solver/mpi < /dev/null > stdout.log 2> stderr.log\n"
    )


# --- a local run ----------------------------------------------------------------------


def test_cfdpp_execute_a_converging_point(tmp_path):
    root, point = make(tmp_path)
    result = cfdpp.execute(point, check_ram=lambda: None)
    assert cfdpp.classify(result) == "converged"
    assert result.criterion == "R" and result.solver_version == "1.0"
    record = json.loads((folder(point) / "execution.json").read_text())
    serial = point.setup["solver"]["serial"]
    assert record["steps"] == [{"name": "solver", "argv": [sys.executable, serial], "exit_code": 0}]
    assert record["exit_code"] == 0 and record["ncpus"] == 1
    assert record["timed_out"] is False and record["stop_requested"] is False
    assert record["executable_sha256"] == hashlib.sha256(Path(serial).read_bytes()).hexdigest()
    assert record["schema_status"] == "verified"
    assert record["raw_keywords"] == []


def test_cfdpp_execute_grid_files_are_linked_not_copied(tmp_path):
    root, point = make(tmp_path)
    cfdpp.execute(point, check_ram=lambda: None)
    grid = root / "inputs/mesh/naca.cfdpp"
    for name in ("nodesin.bin", "cellsin.bin", "exbcsin.bin", "mcfd.bc"):
        assert os.path.samefile(folder(point) / name, grid / name)


def test_cfdpp_execute_launches_without_a_shell_or_a_window(tmp_path, monkeypatch):
    root, point = make(tmp_path)
    seen = []
    real = subprocess.Popen

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    cfdpp.execute(point, check_ram=lambda: None)
    assert seen and all(k["shell"] is False and k["creationflags"] == _NO_WINDOW for k in seen)
    assert all(k["stdin"] == subprocess.DEVNULL for k in seen)


def test_cfdpp_execute_on_several_processes_partitions_first(tmp_path):
    table = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:10", "-", ncpus=2)
    root, point = make(tmp_path, table=table)
    result = cfdpp.execute(point, check_ram=lambda: None)
    assert cfdpp.classify(result) == "converged"
    record = json.loads((folder(point) / "execution.json").read_text())
    assert [step["name"] for step in record["steps"]] == ["partition", "solver"]
    launcher, mpi = point.setup["solver"]["mpi_launcher"][0], point.setup["solver"]["mpi"]
    assert record["steps"][1]["argv"] == [sys.executable, launcher, "-n", "2", mpi]
    assert (folder(point) / "mcpusin.bin.2").is_file()
    assert "partitioned for 2" in (folder(point) / "log.partition").read_text()


def test_cfdpp_execute_a_partition_step_without_its_file_fails_the_point(tmp_path):
    table = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:10", "-", ncpus=2)
    root = workspace(tmp_path / "ws", table=table)
    path = root / "inputs/executables.toml"
    path.write_text(path.read_text().replace('"partition", "{ncpus}"', '"idle"'), encoding="ascii")
    point = points(root)[0]
    result = cfdpp.execute(point, check_ram=lambda: None)
    assert cfdpp.classify(result) == "failed"
    record = json.loads((folder(point) / "execution.json").read_text())
    assert record["problem"] == "the partition step left no mcpusin.bin.2; see log.partition"
    assert [step["name"] for step in record["steps"]] == ["partition"]


def test_cfdpp_execute_a_staged_partition_file_skips_the_partitioner(tmp_path):
    table = matrix("MACH:0.15, REmi:6, TK:300, ALPHA:10", "-", ncpus=2)
    root = workspace(tmp_path / "ws", table=table)
    (root / "inputs/mesh/naca.cfdpp/mcpusin.bin.2").write_bytes(bytes(16))
    point = points(root)[0]
    assert "partition" not in point.point.actions
    cfdpp.execute(point, check_ram=lambda: None)
    record = json.loads((folder(point) / "execution.json").read_text())
    assert [step["name"] for step in record["steps"]] == ["solver"]
    assert os.path.samefile(
        folder(point) / "mcpusin.bin.2", root / "inputs/mesh/naca.cfdpp/mcpusin.bin.2"
    )


def test_cfdpp_execute_a_timeout_asks_for_a_stop_then_fails_the_point(tmp_path):
    root, point = make(
        tmp_path, setup=scenario("hang"), init_options="timeout_s = 1\nstop_grace_s = 20\n"
    )
    result = cfdpp.execute(point, check_ram=lambda: None)
    assert cfdpp.classify(result) == "failed"
    record = json.loads((folder(point) / "execution.json").read_text())
    assert record["timed_out"] is True and record["stop_requested"] is True
    assert record["exit_code"] == 124
    assert (folder(point) / builder().STOP_REQUEST).is_file()
    assert (folder(point) / "mcfd.info0").is_file()  # the solver wrote its files on the stop


def test_cfdpp_execute_without_a_stop_tool_a_timeout_kills_at_once(tmp_path):
    root = workspace(tmp_path / "ws", table=ONE, setup=scenario("hang"),
                     init_options="timeout_s = 1\nstop_grace_s = 20\n")  # fmt: skip
    path = root / "inputs/executables.toml"
    path.write_text(
        "\n".join(line for line in path.read_text().splitlines() if "tools_dir" not in line),
        encoding="ascii",
    )
    point = points(root)[0]
    cfdpp.execute(point, check_ram=lambda: None)
    record = json.loads((folder(point) / "execution.json").read_text())
    assert record["timed_out"] is True and record["stop_requested"] is False
    assert not (folder(point) / builder().STOP_REQUEST).exists()


def test_cfdpp_execute_the_ram_guard_runs_before_and_during(tmp_path):
    root, point = make(tmp_path)
    with pytest.raises(ValueError, match="RAM"):
        cfdpp.execute(point, check_ram=lambda: (_ for _ in ()).throw(ValueError("RAM low")))
    assert not folder(point).exists()
    root, point = make(tmp_path, name="ws2", setup=scenario("hang"),
                       init_options="stop_grace_s = 20\n")  # fmt: skip
    calls = []

    def guard():
        calls.append(None)
        if len(calls) > 6:
            raise ValueError("RAM low during the run")

    with pytest.raises(ValueError, match="during the run"):
        cfdpp.execute(point, check_ram=guard)
    assert (folder(point) / builder().STOP_REQUEST).is_file()


def test_cfdpp_execute_a_missing_executable_is_refused_before_any_write(tmp_path):
    root = workspace(tmp_path / "ws", table=ONE)
    path = root / "inputs/executables.toml"
    path.write_text(
        "\n".join(
            'serial = "/nowhere/cfdpp-serial"' if line.startswith("serial ") else line
            for line in path.read_text().splitlines()
        ),
        encoding="ascii",
    )
    plan = planned(root)
    with pytest.raises(RegistryError) as caught:
        execute_plan(plan, cfdpp)
    text = str(caught.value)
    assert text.startswith("CFD++ executable not found: '") and "cfdpp-serial" in text
    assert text.endswith(
        "'; correct the serial field of build '1.0' in inputs/executables.local.toml (this "
        "machine's paths) or inputs/executables.toml"
    )
    assert not (root / "sims/sim_1001").exists()


def test_cfdpp_execute_failure_is_read_from_files_not_the_exit_code(tmp_path):
    for index, (name, expected) in enumerate(
        [("fail", "failed"), ("nan", "failed"), ("none", "failed"), ("exit1", "failed")]
    ):
        root, point = make(tmp_path, name=f"ws{index}", setup=scenario(name))
        result = cfdpp.execute(point, check_ram=lambda: None)
        assert cfdpp.classify(result) == expected, name
    root, point = make(tmp_path, name="wsfail", setup=scenario("fail"))
    result = cfdpp.execute(point, check_ram=lambda: None)
    assert result.exit_code == 0 and result.failures == ("negative_volume",)


def test_cfdpp_execute_a_local_run_through_the_registry(tmp_path):
    root = workspace(tmp_path / "ws", table=matrix(sweep="0,10"))
    records = execute_plan(planned(root), cfdpp)
    assert [record.classification for record in records] == ["converged", "converged"]
    assert {record.solver_version for record in records} == {"1.0"}
    assert all(r.status == "COMPLETED" for r in RunRegistry(root).read())


# --- staging for a cluster ----------------------------------------------------------


def test_cfdpp_stage_writes_the_same_files_as_a_local_run(tmp_path):
    root, point = make(tmp_path)
    cfdpp.stage(point)
    staged = folder(point)
    names = sorted(path.name for path in staged.iterdir())
    assert names == [
        "cellsin.bin", "cfdpp_job.sh", "exbcsin.bin", "mcfd.bc", "mcfd.inp", "nodesin.bin"
    ]  # fmt: skip
    inp = (staged / "mcfd.inp").read_bytes()
    script = (staged / "cfdpp_job.sh").read_bytes()
    root2, local = make(tmp_path, name="ws2")
    cfdpp.execute(local, check_ram=lambda: None)
    assert (folder(local) / "mcfd.inp").read_bytes() == inp
    assert (folder(local) / "cfdpp_job.sh").read_bytes().replace(
        str(root2.name + "-bin").encode(), str(root.name + "-bin").encode()
    ) == script
    with pytest.raises(ValueError, match="refusing overwrite"):
        cfdpp.stage(point)


def test_cfdpp_job_outputs_are_the_execution_record(tmp_path):
    root, point = make(tmp_path)
    assert set(cfdpp.job_output_files(point.point)) == {"stdout.log", "mcfd.info0", "mcfd.info1"}


# --- restart --------------------------------------------------------------------------


def test_cfdpp_restart_requires_and_copies_the_saved_solution(tmp_path):
    root = workspace(tmp_path / "ws", table=matrix(sweep="0,10"))
    source, point = points(root)
    restarted = dataclasses.replace(
        point,
        point=dataclasses.replace(
            point.point, restart_from=source.point.point_id, restart_directory=None
        ),
    )
    with pytest.raises(ValueError, match="no recorded point"):
        cfdpp.restart_files(restarted)
    cfdpp.execute(source, check_ram=lambda: None)
    with_source = dataclasses.replace(
        restarted,
        point=dataclasses.replace(restarted.point, restart_directory=source.point.directory),
    )
    files = cfdpp.restart_files(with_source)
    assert list(files) == ["cdepsout.bin"]
    assert "trestart 1\n" in cfdpp.translate(with_source)
    cfdpp.stage(with_source)
    assert (folder(with_source) / "cdepsout.bin").read_bytes() == b"synthetic solution"
    (folder(source) / "cdepsout.bin").unlink()
    with pytest.raises(ValueError, match="solution is missing"):
        cfdpp.restart_files(with_source)


# --- collect --------------------------------------------------------------------------


def test_cfdpp_collect_reads_the_last_step_of_the_force_file(tmp_path):
    root, point = make(tmp_path)
    cfdpp.execute(point, check_ram=lambda: None)
    result = cfdpp.collect(point)
    case = cfdpp._point_case(point)[0]
    qa = case.freestream.dynamic_pressure * 1.0
    # Selector 1 (the airfoil wall) of the builder's known forces: (8, 0, 450) N.
    assert result.coefficients["CFx"].value == pytest.approx(8.0 / qa)
    assert result.coefficients["CFy"].value == pytest.approx(0.0)
    assert result.coefficients["Fz"].value == pytest.approx(450.0)
    assert result.coefficients["CFx_inv"].value == pytest.approx(6.0 / qa)
    assert result.classification == "converged"
    data = result.to_dict()
    assert data["dimensions"] == 2 and data["schema_status"] == "verified"
    assert data["criterion"] == "R" and data["raw_keywords"] == []


def test_cfdpp_collect_works_without_execution_json_and_reads_groups(tmp_path):
    text = SETUP + '\n[groups]\nmain = ["airfoil"]\n'
    root, point = make(tmp_path, setup=text)
    cfdpp.execute(point, check_ram=lambda: None)
    (folder(point) / "execution.json").unlink()
    result = cfdpp.collect(point)
    assert set(result.force_groups) == {"TOTAL", "main"}
    assert result.force_groups["main"]["CFx"] == result.coefficients["CFx"]


def test_cfdpp_collect_carries_the_raw_keywords_as_a_silent_field(tmp_path):
    root, point = make(tmp_path, setup=with_keywords("zfree = 3\n"))
    cfdpp.execute(point, check_ram=lambda: None)
    data = cfdpp.collect(point).to_dict()
    assert data["raw_keywords"] == [
        {"keyword": "zfree", "value": 3, "section": "extras", "evidence": "unknown"}
    ]


# --- the conformance assertions of cfdbender.testing --------------------------------


def test_cfdpp_conformance_translation_reads_init_and_is_deterministic(tmp_path):
    root, point = make(tmp_path)
    testing.assert_deterministic(cfdpp, point)
    testing.assert_reads_init(cfdpp, point)


def test_cfdpp_conformance_classifications(tmp_path):
    cases = {
        "converged": make(tmp_path, name="c")[1],
        "iteration_limit": make(tmp_path, name="l", setup=scenario("limit", iterations=40))[1],
        "failed": make(tmp_path, name="f", setup=scenario("fail"))[1],
    }
    testing.assert_classifications(cfdpp, cases)


def test_cfdpp_conformance_exit_evidence_results_and_isolation(tmp_path):
    testing.assert_exit_evidence(cfdpp, make(tmp_path, name="a")[1])
    testing.assert_results(cfdpp, make(tmp_path, name="b")[1])
    testing.assert_execution_isolation(cfdpp, make(tmp_path, name="c")[1])
    testing.assert_no_overwrite(cfdpp, make(tmp_path, name="d")[1])


def test_cfdpp_registration_and_capabilities():
    from cfdbender.solvers import available_solvers, load_solver

    assert "cfdpp" in available_solvers()
    adapter = load_solver("cfdpp")
    assert adapter.CAPABILITIES == frozenset({"cluster_staging", "restart", "boundary_groups"})
    assert adapter.SOLVER_NAME == "cfdpp" and adapter.STAGED_CONFIG == "mcfd.inp"
    assert adapter.validate_row.geometry_folders is True
    assert STOP_TOOL  # the toy database names its stop tool with role = "stop"


def test_cfdpp_aborted_launch_records_failure_even_with_complete_outputs(tmp_path, monkeypatch):
    root, point = make(tmp_path)

    def interrupted(step, target, *args):
        data = builder()
        (target / "mcfd.info0").write_text(data.info0_converged_text(), encoding="ascii")
        (target / "mcfd.info1").write_text(data.info1_known_text(1, 121), encoding="ascii")
        raise ValueError("RAM low during the run")

    monkeypatch.setattr(_execute, "_launch", interrupted)
    with pytest.raises(ValueError, match="RAM low"):
        cfdpp.execute(point, check_ram=lambda: None)
    record = json.loads((folder(point) / "execution.json").read_text())
    assert record["exit_code"] != 0
    assert "RAM low" in record["problem"]
    assert cfdpp.collect(point).classification == "failed"


def test_cfdpp_init_line_endings_control_translated_and_staged_bytes(tmp_path):
    root, point = make(tmp_path, init_options='line_endings = "crlf"\n')
    text = cfdpp.translate(point)
    assert "\r\n" in text
    assert "\n" not in text.replace("\r\n", "")
    cfdpp.stage(point)
    assert (folder(point) / "mcfd.inp").read_bytes() == text.encode("ascii")


def test_cfdpp_failed_cluster_job_is_collectable_without_solver_outputs(tmp_path):
    from types import SimpleNamespace

    from cfdbender.matrix.collection import _job_outputs
    from cfdbender.matrix.submission import job_outputs

    root, point = make(tmp_path)
    target = folder(point)
    target.mkdir(parents=True)
    (target / "execution.json").write_text('{"exit_code": 7}', encoding="ascii")
    record = SimpleNamespace(
        solver="cfdpp",
        point=point.point,
        submission={"job_outputs": job_outputs(cfdpp, point.point)},
    )
    assert _job_outputs(record, root) == (point.point.directory + "/execution.json",)


def test_application_cluster_job_without_execution_record_is_collectable(tmp_path):
    from types import SimpleNamespace

    from cfdbender.matrix.collection import _job_outputs
    from cfdbender.matrix.submission import job_outputs

    root, point = make(tmp_path)
    submission = {"job_outputs": job_outputs(cfdpp, point.point)}
    record = SimpleNamespace(solver="cfdpp", point=point.point, submission=submission)
    outputs = _job_outputs(record)
    assert {Path(name).name for name in outputs} == {"stdout.log", "mcfd.info0", "mcfd.info1"}
