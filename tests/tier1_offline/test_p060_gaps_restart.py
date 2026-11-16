"""P060: RESTART selects a recorded solution; run --restart continues each point."""

import json
from pathlib import Path

import pytest
import test_p0000_item3d_config as su2_case
import test_p0000_openfoam_adapter as foam_case
import test_p0000_pyfr_adapter as pyfr_case
from test_p0000_item3c_adapter import workspace as mock_workspace

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.cli import main


def call(argv):
    try:
        return main(argv)
    except SystemExit as error:
        return error.code


def prepared(root, solver):
    if solver == "su2":
        fake = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2.py"
        su2_case.workspace(
            root, extra='[solver]\nroute="executable"\nexecutable=' + json.dumps(fake.as_posix())
        )
    elif solver == "openfoam":
        foam_case.case(root, sweep="4")
    else:
        pyfr_case.case(root)
        (root / "standin.py").write_text(
            "import pathlib,sys,configparser\n"
            "if 'import' in sys.argv: pathlib.Path('mesh.pyfrm').write_text('mesh')\n"
            "else:\n"
            " c=configparser.ConfigParser(); c.read('case.ini')\n"
            " start=c.getfloat('solver-time-integrator','tstart')\n"
            " pathlib.Path('forces.csv').write_text('t,px,py,vx,vy\\n' + ''.join(\n"
            " f'{start+i},0.1,0.3,0,0\\n' for i in range(4)))\n"
        )
    argv = ["run", "study", "--local", "--workspace", str(root)]
    assert call(argv) == 0
    source = RunRegistry(root).read()[0]
    return source, argv


def solution(root, source, solver):
    name = {"su2": "restart_flow.dat", "openfoam": "4/U", "pyfr": "solution-3.pyfrs"}[solver]
    path = root / source.point.directory / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(b"recorded solution")
    if solver == "openfoam":
        (path.parent / "p").write_bytes(b"pressure")
    return name


def assert_staged(root, record, solver):
    folder = root / record.point.directory
    if solver == "su2":
        config = (folder / "su2.cfg").read_text()
        assert "RESTART_SOL= YES" in config
        assert "SOLUTION_FILENAME= restart-input.dat" in config
        path = folder / "restart-input.dat"
    elif solver == "openfoam":
        assert "startFrom latestTime;" in (folder / "system/controlDict").read_text()
        path = folder / "4/U"
        assert (folder / "4/p").read_bytes() == b"pressure"
    else:
        argv = json.loads((folder / "execution.json").read_bytes())["argv"][-1]
        assert "restart" in argv and "restart.pyfrs" in argv
        assert "tend = 6" in (folder / "case.ini").read_text()
        path = folder / "restart.pyfrs"
    assert path.read_bytes() == b"recorded solution"


@pytest.mark.parametrize("solver", ["su2", "openfoam", "pyfr"])
def test_p060_matrix_restart_copies_the_named_point_solution(tmp_path, solver):
    source, argv = prepared(tmp_path, solver)
    solution(tmp_path, source, solver)
    matrix = tmp_path / "inputs/matrices/study.fs"
    header, row = matrix.read_text().strip().splitlines()
    matrix.write_text(
        header + "|RESTART\n" + row.replace("1001|", "1002|", 1) + "|" + source.point_id + "\n"
    )
    assert call(["plan", *argv[1:]]) == 0
    assert call(argv) == 0
    record = RunRegistry(tmp_path).read()[-1]
    assert getattr(record.point, "restart_from", None) == source.point_id
    assert getattr(record.point, "restart_sha256", {})
    assert_staged(tmp_path, record, solver)


@pytest.mark.parametrize("solver", ["su2", "openfoam", "pyfr"])
def test_p060_run_restart_archives_then_copies_its_own_last_solution(tmp_path, solver):
    source, argv = prepared(tmp_path, solver)
    name = solution(tmp_path, source, solver)
    assert call([*argv, "--restart"]) == 0
    registry = RunRegistry(tmp_path)
    record = registry.read()[0]
    assert getattr(record.point, "restart_from", None) == source.point_id
    history = registry.history()
    assert len(history) == 1
    assert (tmp_path / history[0]["archive"] / source.point.directory / name).is_file()
    assert_staged(tmp_path, record, solver)


@pytest.mark.parametrize("solver", ["su2", "openfoam", "pyfr"])
def test_p060_missing_restart_solution_names_point_and_preserves_evidence(tmp_path, solver, capsys):
    source, argv = prepared(tmp_path, solver)
    before = (tmp_path / "runs.json").read_bytes()
    assert call([*argv, "--restart"]) == 2
    assert source.point_id in capsys.readouterr().err
    assert (tmp_path / "runs.json").read_bytes() == before
    assert not (tmp_path / "archive").exists() or not list((tmp_path / "archive").iterdir())


def test_p060_unsupported_restart_warns_and_skips_or_refuses(tmp_path, capsys):
    mock_workspace(tmp_path)
    argv = ["run", "study", "--local", "--workspace", str(tmp_path), "--restart"]
    assert call(argv) == 0
    assert "restart" in capsys.readouterr().err and RunRegistry(tmp_path).read()


def test_p060_unsupported_restart_honors_warning_refusals(tmp_path, capsys):
    mock_workspace(tmp_path)
    argv = [
        "--warning-refusals",
        "run",
        "study",
        "--local",
        "--workspace",
        str(tmp_path),
        "--restart",
    ]
    assert call(argv) == 2
    assert "does not support restart" in capsys.readouterr().err
    assert not (tmp_path / "runs.json").exists()
