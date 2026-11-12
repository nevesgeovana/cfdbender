"""inputs/executables.toml and its local overlay pick a local run's executable by SOLVER_BUILD."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import test_p0000_item3d_config as su2_case
from test_p0000_item3f_plan import tree

from cfdbender.core.executables import resolve_executable
from cfdbender.core.registry import RunRegistry
from cfdbender.core.settings import WorkspaceError
from cfdbender.matrix.cli import main

FAKE = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2.py"
MPI = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_mpi.py"


def case(root, build="8.3.0", setup_solver=""):
    su2_case.workspace(root, extra=setup_solver)
    matrix = root / "inputs/matrices/study.fs"
    header, row = matrix.read_text(encoding="ascii").strip().splitlines()
    matrix.write_text(f"{header}|SOLVER_BUILD\n{row}|{build}\n", encoding="ascii")


def declare(root, text, local=None):
    (root / "inputs/executables.toml").write_text(text, encoding="utf-8")
    if local is not None:
        (root / "inputs/executables.local.toml").write_text(local, encoding="utf-8")


def run(root):
    return main(["run", "study", "--workspace", str(root), "--local"])


def test_the_build_named_by_the_row_runs_its_declared_executable(tmp_path):
    case(tmp_path)
    declare(tmp_path, '[su2]\n"8.3.0" = ' + json.dumps(FAKE.as_posix()) + "\n")
    assert run(tmp_path) == 0
    record = RunRegistry(tmp_path).read()[0]
    assert record.classification == "converged"
    folder = tmp_path / record.point.directory
    assert json.loads((folder / "execution.json").read_bytes())["route"] == "executable"
    assert "SU2 Release 8.6.0" in (folder / "stdout.log").read_text()


def test_the_local_overlay_replaces_path_and_adds_the_launcher(tmp_path):
    case(tmp_path)
    declare(
        tmp_path,
        '[su2]\n"8.3.0" = "/not/on/this/machine/SU2_CFD"\n',
        '[su2]\n"8.3.0" = { path = '
        + json.dumps(FAKE.as_posix())
        + ", launcher = ["
        + ", ".join(json.dumps(t) for t in ("python", MPI.as_posix(), "-np", "192"))
        + "] }\n",
    )
    found = resolve_executable(tmp_path, "su2", "8.3.0")
    assert found.path == str(FAKE)
    assert found.launcher == ("python", MPI.as_posix(), "-np", "192")


def test_the_row_build_wins_over_the_setup_solver_table(tmp_path):
    case(tmp_path, setup_solver='[solver]\nroute = "executable"\nexecutable = "absent-su2"\n')
    declare(tmp_path, '[su2]\n"8.3.0" = ' + json.dumps(FAKE.as_posix()) + "\n")
    assert run(tmp_path) == 0
    assert RunRegistry(tmp_path).read()[0].classification == "converged"


@pytest.mark.parametrize(
    "declared,local,word",
    [
        ('[su2]\n"8.6.0" = "SU2_CFD"\n', None, "is not declared under [su2]"),
        ('[su2]\n"8.3.0" = "SU2_CFD"\n', '[su2]\n"9.0.0" = "SU2_CFD"\n', "is not declared in"),
        ('[su2]\n"8.3.0" = { launcher = "mpirun" }\n', None, "launcher must be a list"),
        ("[su2\n", None, "cannot read executables"),
    ],
)
def test_a_missing_or_malformed_build_refuses_before_any_write(
    tmp_path, capsys, declared, local, word
):
    case(tmp_path)
    declare(tmp_path, declared, local)
    before = tree(tmp_path)
    assert run(tmp_path) == 2
    assert word in capsys.readouterr().err
    assert before == tree(tmp_path)


def test_without_the_registry_the_setup_solver_table_still_runs(tmp_path):
    case(
        tmp_path,
        setup_solver='[solver]\nroute = "executable"\nexecutable = ' + json.dumps(FAKE.as_posix()),
    )
    assert run(tmp_path) == 0
    with pytest.raises(WorkspaceError, match="does not exist"):
        resolve_executable(tmp_path, "su2", "8.3.0")


def test_a_cluster_submission_ignores_the_local_registry(tmp_path):
    from test_p0000_item3f_plan import profile

    case(tmp_path, build="mock-1")
    declare(tmp_path, '[su2]\n"8.3.0" = "SU2_CFD"\n')
    profile(tmp_path, fields={"config": "{config_path}"})
    assert main(["run", "study", "--workspace", str(tmp_path)]) == 0
    assert RunRegistry(tmp_path).read()[0].status == "SUBMITTED"
