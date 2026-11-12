"""P0000: launcher argv is prepended without resolving a guest executable."""

import json
import sys

import pytest
from test_p0000_xfoil_plan import FAKE, adapter, prepared, workspace


def execution():
    adapter()
    from cfdbender.solvers.xfoil import _execute

    return _execute


def test_p0000_xfoil_launcher_exact_prefix_and_verbatim_guest(tmp_path):
    launcher = ["wsl", "-d", "Ubuntu", "--"]
    guest = "/home/example/bin/xfoil"
    assert execution().argv({"executable": guest, "launcher": launcher}, tmp_path) == [
        *launcher,
        guest,
    ]
    assert execution().argv({"executable": "~/bin/xfoil", "launcher": launcher}, tmp_path) == [
        *launcher,
        "~/bin/xfoil",
    ]


def test_p0000_xfoil_python_environment_and_path_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDBENDER_XFOIL_EXE", str(FAKE))
    assert execution().argv({}, tmp_path) == [sys.executable, str(FAKE)]
    assert execution().argv({"executable": str(FAKE)}, tmp_path) == [sys.executable, str(FAKE)]
    monkeypatch.setenv("CFDBENDER_XFOIL_EXE", "missing-xfoil")
    with pytest.raises(ValueError, match="executable not found"):
        execution().argv({}, tmp_path)


def test_p0000_xfoil_launcher_execution_keeps_point_cwd(tmp_path):
    workspace(tmp_path, condition="alpha:4", sweep="")
    setup = tmp_path / "inputs/setups/s001.toml"
    setup.write_text(
        'airfoil="NACA 0012"\n[solver]\nexecutable="/guest/bin/xfoil"\nlauncher='
        + json.dumps([sys.executable, str(FAKE)])
        + "\n",
        encoding="ascii",
    )
    point = prepared(tmp_path)
    result = adapter().execute(point, check_ram=lambda: None)
    assert adapter().classify(result) == "converged"
    folder = tmp_path / point.point.directory
    metadata = json.loads((folder / "execution.json").read_bytes())
    assert metadata["argv"] == [sys.executable, str(FAKE), "/guest/bin/xfoil"]
    assert (folder / "polar.txt").is_file()


@pytest.mark.parametrize("option", ['launcher="wsl"', "launcher=[1]", 'launcher=[""]'])
def test_p0000_xfoil_launcher_is_validated_during_planning(tmp_path, option):
    workspace(tmp_path, options=option + "\n")
    with pytest.raises(ValueError, match="launcher"):
        prepared(tmp_path)
