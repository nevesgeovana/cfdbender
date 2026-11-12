"""P0000: run a fake executable, read numerical evidence and preserve outputs."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_p0000_item3d_config import api, prepared, workspace

FAKE = Path(__file__).resolve().parents[1] / "fixtures/su2/fake_su2.py"


def test_p0000_cifix_fake_is_discoverable(tmp_path, monkeypatch):
    from cfdbender.solvers.su2._execute import _argv

    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    assert _argv({"executable": str(FAKE)}, tmp_path) == [sys.executable, str(FAKE), "su2.cfg"]


def test_p0000_cifix2_relative_environment_path(tmp_path, monkeypatch):
    from cfdbender.solvers.su2._execute import _argv

    script = tmp_path / "bin" / "solver.py"
    script.parent.mkdir()
    script.write_text("pass", encoding="utf-8")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("CFDBENDER_SU2_EXE", "bin/solver.py")
    assert _argv({}, tmp_path) == [sys.executable, str(script), "su2.cfg"]
    monkeypatch.setenv("CFDBENDER_SU2_EXE", "bin/missing.py")
    with pytest.raises(ValueError, match="SU2 executable not found"):
        _argv({}, tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows executable suffix resolution")
def test_p0000_cifix2_path_without_exe_suffix_resolves(tmp_path, monkeypatch):
    from cfdbender.solvers.su2._execute import _argv

    program = tmp_path / "bin" / "SU2_CFD.exe"
    program.parent.mkdir()
    program.write_bytes(b"")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    argv = _argv({"executable": str(tmp_path / "bin" / "SU2_CFD")}, tmp_path)
    assert [os.path.normcase(argv[0]), argv[1]] == [os.path.normcase(str(program)), "su2.cfg"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable permissions")
def test_p0000_cifix2_path_requires_executable_permission(tmp_path):
    from cfdbender.solvers.su2._execute import _argv

    program = tmp_path / "solver"
    program.write_text("program", encoding="utf-8")
    program.chmod(0o644)
    with pytest.raises(ValueError, match="SU2 executable not found"):
        _argv({"executable": str(program)}, tmp_path)
    program.chmod(0o755)
    assert _argv({"executable": str(program)}, tmp_path) == [str(program), "su2.cfg"]


def execution_api():
    module = api()
    if hasattr(module, "execute"):
        return module
    return SimpleNamespace(
        execute=lambda *args, **kwargs: None,
        classify=lambda result: None,
        collect=lambda point: None,
    )


def case(root, outcome="converged", *, timeout=5):
    workspace(
        root,
        extra=chr(10).join(
            [
                f'FAKE_CASE = "{outcome}"',
                "[solver]",
                'route = "executable"',
                "executable = " + json.dumps(FAKE.as_posix()),
                "threads = 2",
                f"timeout_s = {timeout}",
            ]
        ),
    )
    return prepared(root)


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("converged", "converged"),
        ("iteration_limit", "iteration_limit"),
        ("failed", "failed"),
        ("nan", "failed"),
        ("missing", "failed"),
        ("exit_only", "iteration_limit"),
    ],
)
def test_p0000_item3d_execute_classify_collect(tmp_path, outcome, expected):
    point = case(tmp_path, outcome)
    adapter = execution_api()
    checks = []
    result = adapter.execute(point, check_ram=lambda: checks.append(1))
    assert adapter.classify(result) == expected, "classify numerical evidence, not Exit Success"
    assert len(checks) >= 2
    collected = adapter.collect(point)
    assert collected.classification == expected
    assert collected.solver_version == "8.6.0"
    assert collected.history == result.history
    assert collected.references.area == 2
    assert collected.references.length == 3
    assert collected.references.moment_origin == (1, 2, 3)
    if outcome in ("nan", "missing"):
        assert result.exit_code == 0, "numerical failure must preserve the actual process exit"
    folder = tmp_path / point.point.directory
    assert (folder / "mesh.su2").read_bytes() == (tmp_path / "inputs/meshes/tiny.su2").read_bytes()
    assert "OMP_NUM_THREADS=2" in (folder / "stdout.log").read_text()
    assert "diagnostic" in (folder / "stderr.log").read_text()
    if expected != "failed":
        assert collected.coefficients["CL"].axis == "z"
        assert collected.coefficients["CD"].axis == "x"
        assert collected.coefficients["CSF"].axis == "y"
        assert collected.coefficients["CMy"].axis == "y"
        assert collected.coefficients["CMy"].kind == "moment"
        assert collected.coefficients["CL"].value == 0.3
        assert (
            "wind" in collected.references.axis_system
            and "body" in collected.references.axis_system
        )
        assert result.history[0].iteration == 1
        assert result.history[0].residual == pytest.approx(0.01)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, OSError)):
        adapter.execute(point, check_ram=lambda: None)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_p0000_item3d_timeout_preserves_channels(tmp_path):
    point = case(tmp_path, "timeout", timeout=0.5)
    adapter = execution_api()
    result = adapter.execute(point, check_ram=lambda: None)
    assert adapter.classify(result) == "failed"
    metadata = json.loads((tmp_path / point.point.directory / "execution.json").read_text())
    assert metadata["timed_out"] is True
    assert "SU2 Release" in (tmp_path / point.point.directory / "stdout.log").read_text()


def test_p0000_item3d_exit_success_without_history_is_failed(tmp_path):
    adapter = execution_api()
    result = adapter.execute(case(tmp_path), check_ram=lambda: None)
    assert result is not None, "execution must supply numerical evidence"
    assert adapter.classify(replace(result, history=())) == "failed"
    assert adapter.classify(replace(result, exit_code=1)) == "failed"


@pytest.mark.parametrize(
    "free,estimate,accept",
    [
        (0.74, 0.25, False),
        (0.75, 0.25, True),
        (float("nan"), 0.25, False),
        (2, -1, False),
        (float("inf"), 0.25, False),
    ],
)
def test_p0000_item3d_ram_estimate_with_fake_measurement(free, estimate, accept):
    try:
        ram = importlib.import_module("cfdbender.solvers.su2._ram")
    except ModuleNotFoundError:
        ram = SimpleNamespace(require_ram=lambda *args, **kwargs: False)
    if accept:
        assert ram.require_ram(estimate, measure=lambda: free) is None
    else:
        with pytest.raises(ValueError, match="RAM"):
            ram.require_ram(estimate, measure=lambda: free)


def test_p0000_item3d_low_ram_refuses_before_writes(tmp_path, monkeypatch):
    point = case(tmp_path)
    adapter = execution_api()
    if hasattr(adapter, "execute") and adapter is api():
        ram = importlib.import_module("cfdbender.solvers.su2._ram")
        monkeypatch.setattr(ram, "free_ram_gib", lambda: 0.6)
    with pytest.raises(ValueError, match="RAM"):
        adapter.execute(point, check_ram=lambda: None)
    assert not (tmp_path / point.point.directory).exists()


def test_p0000_item3d_ram_guard_during_process_reaps_child(tmp_path):
    point = case(tmp_path, "timeout")
    adapter = execution_api()
    calls = []

    def check():
        calls.append(1)
        if len(calls) >= 4:
            raise ValueError("RAM floor threatened")

    with pytest.raises(ValueError, match="RAM"):
        adapter.execute(point, check_ram=check)
    folder = tmp_path / point.point.directory
    assert json.loads((folder / "execution.json").read_text())["exit_code"] != 0


def test_review6_convergence_evidence_on_stderr(tmp_path):
    point = case(tmp_path)
    adapter = execution_api()
    adapter.execute(point, check_ram=lambda: None)
    folder = tmp_path / point.point.directory
    text = (folder / "stdout.log").read_text()
    (folder / "stdout.log").write_text("", encoding="utf-8")
    (folder / "stderr.log").write_text(text.replace("\n", "\r\n"), encoding="utf-8", newline="")
    result = adapter.collect(point)
    assert result.classification == "converged"
    assert result.solver_version == "8.6.0"
