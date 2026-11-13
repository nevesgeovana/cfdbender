"""Usability fixes of GOAL-087 arm 10 that concern `cfd-matrix run`: refusals before any point,
live progress, one wording for a missing executable."""

from __future__ import annotations

import re

import pytest
from test_conformance_workspace import case, local, state_no_build  # noqa: F401

from cfdbender.core.registry import RunRegistry
from cfdbender.matrix.receipt import write_receipt
from cfdbender.workspace.planning import plan_workspace

REAL = ["su2", "openfoam", "xfoil", "pyfr", "qprop"]


def _planned(root):
    state_no_build(root)
    write_receipt(plan_workspace(root, "full"))


def _break_executable(root, solver):
    path = root / "inputs/setups/s001.toml"
    text = path.read_text(encoding="ascii")
    key = "executable"
    text = re.sub(r'^executable = "[^"]+"', 'executable = "absent-solver-87"', text, flags=re.M)
    if solver == "openfoam":
        text = text.replace('executable = "absent-solver-87"', 'launcher = ["absent-launcher-87"]')
    path.write_text(text, encoding="ascii")
    return key


def test_u05_a_missing_executable_refuses_the_run_before_any_record(case, capsys):  # noqa: F811
    solver, root = case
    if solver in ("mock", "openfoam"):
        pytest.skip("no host executable to resolve")
    _planned(root)
    _break_executable(root, solver)
    assert local(root) == 2
    err = capsys.readouterr().err
    assert "matrix not run" in err and "executable not found" in err, err
    assert "absent-solver-87" in err and "executable" in err
    assert RunRegistry(root).read() == ()
    assert not (root / "sims").exists() or not list((root / "sims").rglob("datapoints/*"))


def test_u05_a_busy_host_slot_refuses_the_run_before_any_record(case, monkeypatch, capsys):  # noqa: F811
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock runs no process")
    _planned(root)
    from cfdbender.solvers._host import _exe

    def busy(name):
        raise ValueError(f"{name} is already running; only one process at a time")

    from contextlib import contextmanager

    @contextmanager
    def lease(name):
        busy(name)
        yield

    monkeypatch.setattr(_exe, "process_lease", lease)
    assert local(root) == 2
    err = capsys.readouterr().err
    assert "matrix not run" in err and "holds this machine" in err, err
    assert RunRegistry(root).read() == ()


def test_u05_the_ram_floor_refuses_the_run_before_any_record(case, monkeypatch, capsys):  # noqa: F811
    solver, root = case
    if solver == "mock":
        pytest.skip("the mock runs no process")
    _planned(root)
    monkeypatch.setattr("cfdbender.core.run.free_ram_gib", lambda: 0.1)
    monkeypatch.setattr("cfdbender.solvers._host._ram.free_ram_gib", lambda: 0.1)
    assert local(root) == 2
    err = capsys.readouterr().err
    assert "matrix not run" in err and "RAM" in err, err
    assert RunRegistry(root).read() == ()


def test_u03_run_prints_a_line_as_each_point_starts_and_ends(case, capsys):  # noqa: F811
    solver, root = case
    _planned(root)
    assert local(root) == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[")]
    total = len(RunRegistry(root).read())
    assert len(lines) == 2 * total, lines
    assert re.match(rf"\[1/{total}\] \S+  running", lines[0])
    assert re.match(rf"\[1/{total}\] \S+  COMPLETED \w+ in \d+ s", lines[1])


@pytest.mark.parametrize("solver", REAL)
def test_u09_one_wording_for_a_missing_executable(solver, tmp_path):
    from test_conformance_workspace import build, executables

    from cfdbender.core.adapter import load_point
    from cfdbender.solvers import load_solver

    root = build(tmp_path, solver)
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(setup.read_bytes() + executables(root, solver).encode("ascii"))
    if solver == "xfoil":
        matrix = root / "inputs/matrices/full.fs"
        matrix.write_text(matrix.read_text(encoding="ascii").replace("MACH:0.1", "MACH:0"))
    _break_executable(root, solver)
    adapter = load_solver(solver)
    point = load_point(root, plan_workspace(root, "full").points[0], solver=solver)
    with pytest.raises(ValueError) as caught:
        adapter.preflight(point)
    message = str(caught.value)
    assert "executable not found" in message or "launcher" in message, message
    assert "WinError" not in message and "O sistema" not in message
    if solver != "openfoam":
        assert "SOLVER_BUILD" in message and "[solver]" in message
