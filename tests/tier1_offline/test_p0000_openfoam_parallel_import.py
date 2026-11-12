"""P0000 HPC-4: mesh_converter='convert {mesh} -case {case}' is user supplied."""

import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest
from test_p0000_openfoam_adapter import api, case, point, split

from cfdbender.solvers.openfoam._config import config
from cfdbender.workspace.planning import plan_workspace


@pytest.mark.parametrize("suffix", [".cgns", ".ugrid"])
def test_p0000_openfoam_parallel_import_tracking_staging(tmp_path, monkeypatch, suffix):
    case(tmp_path, sweep="4")
    setup = tmp_path / "inputs/setups/s001.toml"
    text = setup.read_text().replace("tiny.msh", f"tiny{suffix}")
    text = 'mesh_converter = "userConverter {mesh} -case {case}"\n' + text
    setup.write_text(text, encoding="ascii")
    payload = b"synthetic binary mesh\x00"
    mesh = f"inputs/meshes/tiny{suffix}"
    (tmp_path / mesh).write_bytes(payload)
    try:
        plan = plan_workspace(tmp_path, "study")
    except ValueError as error:
        pytest.fail(f"P0000-HPC-4: converter setup refused: {error}")
    prepared = point(plan)
    assert prepared.point.inputs_sha256[mesh] == hashlib.sha256(payload).hexdigest()
    assert f"{prepared.point.directory}/mesh{suffix}" in prepared.point.outputs
    script = split(api().translate(prepared))["Allrun"]
    expected = f"userConverter mesh{suffix} -case ."
    assert expected in script and "gmshToFoam" not in script
    assert script.index(expected) < script.index("foamDictionary")
    from cfdbender.solvers.openfoam import _execute

    def capture(command, folder, opts, timeout, check_ram):
        assert (folder / f"mesh{suffix}").read_bytes() == payload
        assert not (folder / "mesh.msh").exists()

    monkeypatch.setattr(_execute, "_launch", capture)
    _execute.run(prepared, api().translate(prepared), lambda: None)
    (tmp_path / mesh).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        api().translate(prepared)


@pytest.mark.parametrize(
    "converter", [None, "", "convert {unknown}", "convert {mesh", 3, "convert\nbad"]
)
def test_p0000_openfoam_parallel_import_converter_validation(converter):
    setup = {"mesh": "inputs/meshes/grid.cgns", "application": "simpleFoam"}
    if converter is not None:
        setup["mesh_converter"] = converter
    with pytest.raises(ValueError, match="mesh_converter"):
        config(setup, {}, {}, {})


@pytest.mark.parametrize("fail", [False, True])
def test_p0000_openfoam_parallel_import_allrun_converter(tmp_path, monkeypatch, fail):
    from cfdbender.solvers.openfoam import _execute

    bash = (
        Path("C:/Program Files/Git/bin/bash.exe")
        if os.name == "nt"
        else Path(shutil.which("bash") or "/missing")
    )
    if not bash.is_file():
        pytest.skip("native bash unavailable; WSL is never used")
    case(tmp_path, sweep="4")
    fixtures = Path(__file__).resolve().parents[1] / "fixtures/openfoam"
    command = (
        shlex.join(
            [Path(sys.executable).as_posix(), (fixtures / "fake_mesh_converter.py").as_posix()]
        )
        + " {mesh} {case}"
    )
    setup = tmp_path / "inputs/setups/s001.toml"
    text = setup.read_text().replace("tiny.msh", "tiny.ugrid")
    setup.write_text("mesh_converter = " + json.dumps(command) + "\n" + text, encoding="ascii")
    (tmp_path / "inputs/meshes/tiny.ugrid").write_bytes(b"synthetic binary mesh\x00")
    plan = plan_workspace(tmp_path, "study")
    tools = tmp_path / "offline-tools"
    tools.mkdir()
    for tool in ("foamDictionary", "simpleFoam"):
        wrapper = tools / tool
        wrapper.write_text(
            "#!/usr/bin/env bash\nexec "
            + shlex.join(
                [
                    Path(sys.executable).as_posix(),
                    (fixtures / "fake_foam_command.py").as_posix(),
                    tool,
                ]
            )
            + ' "$@"\n',
            encoding="ascii",
            newline="\n",
        )
        wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("CFDBENDER_OPENFOAM_BASHRC", raising=False)
    monkeypatch.setenv("FAKE_OPENFOAM_MODE", "converged")
    monkeypatch.setenv("FAKE_CONVERTER_FAIL", "1" if fail else "0")
    monkeypatch.setattr(_execute, "argv", lambda opts, root: [str(bash), "Allrun"])
    result = api().execute(point(plan), check_ram=lambda: None)
    folder = tmp_path / plan.points[0].directory
    assert json.loads((folder / "converter-args.json").read_text()) == ["mesh.ugrid", "."]
    assert (folder / "log.meshConverter").is_file()
    if fail:
        assert result.exit_code == 9
        assert not (folder / "commands.txt").exists()
        assert api().classify(result) == "failed"
    else:
        assert (folder / "commands.txt").read_text().splitlines()[-1] == "simpleFoam "
        assert api().classify(result) == "converged"
