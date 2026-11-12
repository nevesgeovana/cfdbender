"""P0000 HPC-3: [solver] processes=12 decomposes, solves and reconstructs."""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest
from test_p0000_openfoam_adapter import api, case, point, split

from cfdbender.solvers.openfoam._config import options


@pytest.mark.parametrize(
    "launcher,command",
    [(None, "mpirun"), ('["mpi run", "--bind-to", "core"]', "'mpi run' --bind-to core")],
)
def test_p0000_openfoam_parallel_script_and_collection(tmp_path, launcher, command):
    extra = "processes = 12\n" + ("" if launcher is None else f"mpi_launcher = {launcher}\n")
    try:
        plan = case(tmp_path, sweep="4", extra=extra)
    except ValueError as error:
        pytest.fail(f"P0000-HPC-3: parallel setup refused: {error}")
    files = split(api().translate(point(plan)))
    assert "numberOfSubdomains 12;" in files["system/decomposeParDict"]
    assert "method scotch;" in files["system/decomposeParDict"]
    script = files["Allrun"]
    commands = [
        "foamDictionary",
        "decomposePar > log.decomposePar",
        f"{command} -np 12 simpleFoam -parallel > log.simpleFoam",
        "reconstructPar -latestTime > log.reconstructPar",
    ]
    assert [script.index(value) for value in commands] == sorted(
        script.index(value) for value in commands
    )
    assert f"{point(plan).point.directory}/system/decomposeParDict" in plan.points[0].outputs
    api().execute(point(plan), check_ram=lambda: None)
    result = api().collect(point(plan))
    assert result.classification == "converged"
    assert result.coefficients["Cl"].value == 0.47
    assert (
        tmp_path / plan.points[0].directory / "postProcessing/forceCoeffs/0/coefficient.dat"
    ).is_file()


def test_p0000_openfoam_parallel_serial_unchanged(tmp_path):
    plan = case(tmp_path, sweep="4")
    files = split(api().translate(point(plan)))
    assert "system/decomposeParDict" not in files
    assert "simpleFoam > log.simpleFoam 2>&1\n" in files["Allrun"]
    assert "decomposePar" not in files["Allrun"]
    assert "reconstructPar" not in files["Allrun"]


@pytest.mark.parametrize("processes", [True, 0, -1, 1.5, "12"])
def test_p0000_openfoam_parallel_process_validation(processes):
    with pytest.raises(ValueError, match="processes.*positive integer"):
        options({"solver": {"processes": processes}})


@pytest.mark.parametrize("launcher", [[], "mpirun", [1], [""], ["bad\nvalue"]])
def test_p0000_openfoam_parallel_launcher_validation(launcher):
    with pytest.raises(ValueError, match="mpi_launcher.*nonempty.*strings"):
        options({"solver": {"mpi_launcher": launcher}})


@pytest.mark.parametrize("failure", [None, "decomposePar", "mpirun", "reconstructPar"])
def test_p0000_openfoam_parallel_allrun_offline(tmp_path, monkeypatch, failure):
    from cfdbender.solvers.openfoam import _execute

    bash = (
        Path("C:/Program Files/Git/bin/bash.exe")
        if os.name == "nt"
        else Path(shutil.which("bash") or "/missing")
    )
    if not bash.is_file():
        pytest.skip("native bash unavailable; WSL is never used")
    plan = case(tmp_path, sweep="4", extra="processes = 12\n")
    tools = tmp_path / "offline-tools"
    tools.mkdir()
    fixture = Path(__file__).resolve().parents[1] / "fixtures/openfoam/fake_foam_command.py"
    for tool in (
        "gmshToFoam",
        "foamDictionary",
        "decomposePar",
        "mpirun",
        "simpleFoam",
        "reconstructPar",
    ):
        wrapper = tools / tool
        wrapper.write_text(
            "#!/usr/bin/env bash\nexec "
            + shlex.join([Path(sys.executable).as_posix(), fixture.as_posix(), tool])
            + ' "$@"\n',
            encoding="ascii",
            newline="\n",
        )
        wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("CFDBENDER_OPENFOAM_BASHRC", raising=False)
    monkeypatch.setenv("FAKE_OPENFOAM_MODE", "converged")
    monkeypatch.setenv("FAKE_FAIL_TOOL", failure or "")
    monkeypatch.setattr(_execute, "argv", lambda opts, root: [str(bash), "Allrun"])
    result = api().execute(point(plan), check_ram=lambda: None)
    folder = tmp_path / plan.points[0].directory
    commands = (folder / "commands.txt").read_text().splitlines()
    assert commands[0] == "gmshToFoam mesh.msh"
    assert commands[4] == "decomposePar "
    if failure is None:
        assert commands[5:] == [
            "mpirun -np 12 simpleFoam -parallel",
            "simpleFoam -parallel",
            "reconstructPar -latestTime",
        ]
        assert api().classify(result) == "converged"
    else:
        assert json.loads((folder / "execution.json").read_text())["exit_code"] == 7
        assert api().classify(result) == "failed"
        assert commands[-1].split()[0] == failure
