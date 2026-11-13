"""Gap 8: the solver version is recorded on every run record; a SOLVER_BUILD mismatch is refused."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_p0000_item3c_adapter import workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.core.run import execute_plan
from cfdbender.matrix.cli import main
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace


def with_build(root, build):
    workspace(root)
    matrix = root / "inputs/matrices/study.fs"
    matrix.write_text(
        "POL|RUN|SET|REF|PPROC|SOLVER_BUILD\n" + ("1001|1|001|001|001|" + (build or "-") + "\n"),
        encoding="ascii",
    )


def test_a_completed_record_carries_the_version_the_solver_reported(tmp_path):
    with_build(tmp_path, None)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    (record,) = RunRegistry(tmp_path).read()
    assert record.solver_version == "mock-1"
    stored = json.loads((tmp_path / "runs.json").read_text())
    assert stored["records"][0]["solver_version"] == "mock-1"


def test_a_planned_or_running_record_holds_no_version(tmp_path):
    with_build(tmp_path, None)
    RunRegistry(tmp_path).record(plan_workspace(tmp_path, "study"))
    (record,) = RunRegistry(tmp_path).read()
    assert record.status == "PLANNED" and record.solver_version is None
    assert "solver_version" not in json.loads((tmp_path / "runs.json").read_text())["records"][0]


def test_a_row_build_equal_to_the_reported_version_runs(tmp_path):
    with_build(tmp_path, "mock-1")
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    (record,) = RunRegistry(tmp_path).read()
    assert record.classification == "converged"
    assert record.point.solver_build == "mock-1" and record.solver_version == "mock-1"


def test_a_row_build_that_differs_from_the_reported_version_is_refused(tmp_path, capsys):
    with_build(tmp_path, "9.9.9")
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 2
    (record,) = RunRegistry(tmp_path).read()
    assert record.classification == "failed"
    assert record.solver_version == "mock-1"
    message = record.error or ""
    assert "SOLVER_BUILD" in message and "9.9.9" in message and "mock-1" in message
    assert "SOLVER_BUILD" in capsys.readouterr().err


def reporting(version):
    mock = load_solver("mock")

    def execute(point, **kwargs):
        return replace(mock.execute(point, **kwargs), solver_version=version)

    return SimpleNamespace(execute=execute, classify=mock.classify)


@pytest.mark.parametrize(
    "build,version,refused",
    [
        ("8.3.0", "8.3.0", False),
        ("8.3.0-mpi", "8.3.0", False),
        ("8.3.0", "8.4.0", True),
        ("8.3", "8.3.0", True),
        ("8.3.0", "unknown", False),
    ],
)
def test_the_build_matches_the_reported_version_or_an_alias_of_it(
    tmp_path, build, version, refused
):
    with_build(tmp_path, build)
    plan = plan_workspace(tmp_path, "study")
    (record,) = execute_plan(plan, reporting(version), check_ram=lambda: None)
    assert record.solver_version == version
    assert (record.classification == "failed") is refused
    assert bool(record.error and "SOLVER_BUILD" in record.error) is refused


PYFR_STANDIN = (
    "import pathlib,sys\n"
    "if '--version' in sys.argv:\n"
    "    print('PyFR 2.0.3'); sys.exit(0)\n"
    "if 'import' in sys.argv: pathlib.Path('mesh.pyfrm').write_text('mesh')\n"
    "else: pathlib.Path('forces.csv').write_text('t,px,py,vx,vy\\n'"
    " + '0,0.1,0.3,0,0\\n1,0.1,0.3,0,0\\n2,0.1,0.3,0,0\\n3,0.1,0.3,0,0\\n')\n"
)


def pyfr_case(root, standin_text):
    from test_p0000_pyfr_adapter import case

    plan = case(root)
    (root / "standin.py").write_text(standin_text, encoding="ascii")
    return plan


def test_pyfr_asks_its_executable_for_the_version(tmp_path):
    from cfdbender.core.adapter import load_point

    plan = pyfr_case(tmp_path, PYFR_STANDIN)
    adapter = load_solver("pyfr")
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    result = adapter.execute(point, check_ram=lambda: None)
    assert result.solver_version == "2.0.3"
    assert adapter.collect(point).solver_version == "2.0.3"


def test_pyfr_without_a_version_answer_reports_unknown_and_stays_clean(tmp_path):
    from cfdbender.core.adapter import load_point

    plan = pyfr_case(tmp_path, PYFR_STANDIN.replace("--version", "--no-such-flag"))
    adapter = load_solver("pyfr")
    point = load_point(plan.root, plan.points[0], solver="pyfr")
    result = adapter.execute(point, check_ram=lambda: None)
    assert result.solver_version == "unknown"
    folder = tmp_path / plan.points[0].directory
    assert sorted(p.name for p in folder.iterdir() if p.is_file()) == [
        "case.ini",
        "execution.json",
        "forces.csv",
        "mesh.msh",
        "mesh.pyfrm",
        "stderr.log",
        "stdout.log",
    ]
    assert sys.executable


def test_a_rebuilt_record_carries_the_version_of_the_collected_result(tmp_path):
    with_build(tmp_path, None)
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    (tmp_path / "runs.json").unlink()
    assert main(["rebuild", "--apply", "--workspace", str(tmp_path)]) == 0
    assert RunRegistry(tmp_path).read()[0].solver_version == "mock-1"
