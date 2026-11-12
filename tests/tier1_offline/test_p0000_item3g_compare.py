"""P0000: route comparison reports measured outcomes and honest differences."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_p0000_item3d_execute import FAKE
from test_p0000_item3g_worker import wrapper as _wrapper

from cfdbender.solvers.su2 import _ram

wrapper = _wrapper
TOOL = Path(__file__).resolve().parents[2] / "tools/compare_su2_routes.py"
MESH = Path(__file__).resolve().parents[1] / "fixtures/su2/naca0012_small.su2"


def tool():
    assert TOOL.is_file(), "provide the executable-vs-pysu2 comparison tool"
    spec = importlib.util.spec_from_file_location("compare_su2_routes", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0000_item3g_compare_both_routes(tmp_path, wrapper, monkeypatch, capsys):
    monkeypatch.setenv("CFDBENDER_SU2_EXE", str(FAKE))
    monkeypatch.setenv("CFDBENDER_PYSU2_PATH", str(wrapper))
    output = tmp_path / "comparison"
    assert tool().main(["--output", str(output)]) == 0
    text = capsys.readouterr().out
    for value in (
        "route",
        "wall",
        "peak RAM",
        "classification",
        "CL",
        "CD",
        "executable",
        "pysu2",
        "converged",
        "not measured",
        "max abs",
        "0",
    ):
        assert value in text
    for route in ("executable", "pysu2"):
        metadata = list((output / route).glob("sims/*/datapoints/*/execution.json"))
        assert len(metadata) == 1
        assert json.loads(metadata[0].read_bytes())["route"] == route
        assert (
            list((output / route).glob("sims/*/datapoints/*/mesh.su2"))[0].read_bytes()
            == MESH.read_bytes()
        )
    with pytest.raises(ValueError, match="exist"):
        tool().compare(output, MESH, str(FAKE))


def test_p0000_item3g_compare_disagreement(tmp_path, wrapper, monkeypatch, capsys):
    driver = tmp_path / "failed.py"
    driver.write_text("import sys; print('failed executable'); sys.exit(1)")
    driver.chmod(0o755)
    monkeypatch.setenv("CFDBENDER_SU2_EXE", str(driver))
    monkeypatch.setenv("CFDBENDER_PYSU2_PATH", str(wrapper))
    assert tool().main(["--output", str(tmp_path / "comparison")]) == 1
    assert "failed" in capsys.readouterr().out


def test_p0000_item3g_compare_measures_nonzero_difference(tmp_path, wrapper, monkeypatch, capsys):
    driver = tmp_path / "different.py"
    driver.write_text(FAKE.read_text().replace("0.3, 0.02", "0.4, 0.04"))
    driver.chmod(0o755)
    monkeypatch.setenv("CFDBENDER_SU2_EXE", str(driver))
    monkeypatch.setenv("CFDBENDER_PYSU2_PATH", str(wrapper))
    assert tool().main(["--output", str(tmp_path / "comparison")]) == 0
    assert "max abs" in capsys.readouterr().out
    # Saved results rather than rounded screen columns must drive the difference.
    assert tool().compare(tmp_path / "second", MESH, str(driver))[1] == pytest.approx(0.1)


def test_p0000_item3g_compare_ram_refusal_precedes_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_ram, "free_ram_gib", lambda: 0.6)
    output = tmp_path / "comparison"
    assert tool().main(["--output", str(output)]) == 2
    assert "RAM" in capsys.readouterr().err
    assert not output.exists()


def test_p0000_item3g_compare_exe_beside_wrapper(tmp_path, monkeypatch):
    directory = tmp_path / "install"
    directory.mkdir()
    candidate = directory / "SU2_CFD"
    candidate.touch()
    monkeypatch.delenv("CFDBENDER_SU2_EXE", raising=False)
    monkeypatch.setenv("CFDBENDER_PYSU2_PATH", str(directory))
    assert tool().executable() == str(candidate)


def test_p0000_item3g_compare_missing_executable_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CFDBENDER_SU2_EXE", raising=False)
    monkeypatch.setenv("CFDBENDER_PYSU2_PATH", str(tmp_path / "absent"))
    output = tmp_path / "comparison"
    assert tool().main(["--output", str(output)]) == 2
    assert "CFDBENDER_SU2_EXE" in capsys.readouterr().err
    assert not output.exists()
