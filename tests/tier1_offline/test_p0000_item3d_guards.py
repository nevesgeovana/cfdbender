"""P0000: refuse concurrent launch, stale inputs and unavailable explicit routes."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_p0000_item3d_config import prepared
from test_p0000_item3d_execute import case, execution_api


def test_p0000_item3d_process_lock_prevents_second_interpreter(tmp_path):
    point = case(tmp_path)
    import cfdbender.solvers.su2 as su2

    code = chr(10).join(
        [
            "import sys, time",
            "sys.path.insert(0, sys.argv[1])",
            "try:",
            "    from cfdbender.solvers.su2._single import process_lease",
            "except ImportError:",
            "    from contextlib import nullcontext as process_lease",
            "with process_lease():",
            "    print('owned', flush=True)",
            "    time.sleep(10)",
        ]
    )
    source = Path(__file__).resolve().parents[2] / "src"
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(source)],
        stdout=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout.readline().strip() == "owned"
        with pytest.raises(ValueError, match="one process"):
            su2.execute(point, check_ram=lambda: None)
        assert not (tmp_path / point.point.directory).exists()
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()


@pytest.mark.parametrize("available", [False, True])
def test_p0000_item3d_explicit_pysu2_never_launches_executable(tmp_path, monkeypatch, available):
    point = case(tmp_path)
    source = tmp_path / "inputs/setups/s001.toml"
    source.write_text(source.read_text().replace('route = "executable"', 'route = "pysu2"'))
    point = prepared(tmp_path)
    if available:
        monkeypatch.setitem(sys.modules, "pysu2", SimpleNamespace())
    else:
        monkeypatch.setitem(sys.modules, "pysu2", None)
    with pytest.raises(ValueError, match="pysu2"):
        execution_api().execute(point, check_ram=lambda: None)
    assert not (tmp_path / point.point.directory).exists()


def test_p0000_item3d_default_fallback_runs_with_warning(tmp_path, monkeypatch):
    case(tmp_path)
    source = tmp_path / "inputs/setups/s001.toml"
    source.write_text(source.read_text().replace('route = "executable"', ""))
    monkeypatch.setitem(sys.modules, "pysu2", None)
    with pytest.warns(RuntimeWarning, match="falling back.*executable"):
        result = execution_api().execute(prepared(tmp_path), check_ram=lambda: None)
    assert execution_api().classify(result) == "converged"


def test_p0000_item3d_environment_executable_and_default_threads(tmp_path, monkeypatch):
    case(tmp_path)
    source = tmp_path / "inputs/setups/s001.toml"
    lines = [
        line
        for line in source.read_text().splitlines()
        if not line.startswith(("executable =", "threads ="))
    ]
    source.write_text(chr(10).join(lines))
    from test_p0000_item3d_execute import FAKE

    monkeypatch.setenv("CFDBENDER_SU2_EXE", str(FAKE))
    point = prepared(tmp_path)
    result = execution_api().execute(point, check_ram=lambda: None)
    assert execution_api().classify(result) == "converged"
    assert "OMP_NUM_THREADS=1" in (tmp_path / point.point.directory / "stdout.log").read_text()


def test_p0000_item3d_missing_executable_refuses_before_writes(tmp_path, monkeypatch):
    case(tmp_path)
    source = tmp_path / "inputs/setups/s001.toml"
    lines = [
        line for line in source.read_text().splitlines() if not line.startswith("executable =")
    ]
    source.write_text(chr(10).join(lines))
    monkeypatch.setenv("CFDBENDER_SU2_EXE", "no-such-su2-executable")
    point = prepared(tmp_path)
    with pytest.raises(ValueError, match="executable.*not found"):
        execution_api().execute(point, check_ram=lambda: None)
    assert not (tmp_path / point.point.directory).exists()


def test_p0000_item3d_collect_reads_used_cfg_references(tmp_path):
    point = case(tmp_path)
    adapter = execution_api()
    adapter.execute(point, check_ram=lambda: None)
    cfg = tmp_path / point.point.directory / "su2.cfg"
    cfg.write_text(cfg.read_text().replace("REF_AREA= 2", "REF_AREA= 7"))
    assert adapter.collect(point).references.area == 7
    # Run/collect orchestration separately protects these bytes with output digests.
    metadata = json.loads((cfg.parent / "execution.json").read_text())
    assert metadata["route"] == "executable"


def test_review6_timeout_kills_descendant(tmp_path):
    import json
    import time

    from test_p0000_item3d_config import prepared, workspace
    from test_p0000_item3d_execute import execution_api

    marker = tmp_path / "survived"
    child = tmp_path / "child.py"
    child.write_text(
        "import time, pathlib; time.sleep(2); pathlib.Path("
        + repr(str(marker))
        + ").write_text('alive')",
        encoding="utf-8",
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import subprocess, sys, time; subprocess.Popen([sys.executable, "
        + repr(str(child))
        + "]); time.sleep(10)",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    workspace(
        tmp_path,
        extra='[solver]\nroute = "executable"\nexecutable = '
        + json.dumps(driver.as_posix())
        + "\ntimeout_s = 0.8",
    )
    execution_api().execute(prepared(tmp_path), check_ram=lambda: None)
    time.sleep(2.2)
    assert not marker.exists(), "timeout left a descendant process running"
