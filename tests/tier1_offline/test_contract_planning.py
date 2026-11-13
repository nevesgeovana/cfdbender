"""Adapter contract, registration and planning clauses, over the six adapters.

Each test names its clause of docs/adapter-contract.md in its docstring and runs for
su2, openfoam, xfoil, pyfr, qprop and mock with the offline stand-ins of
test_conformance_workspace. Nothing here launches a real solver.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_conformance_workspace import SOLVERS, case, data_row  # noqa: F401

from cfdbender.core.adapter import SolverAdapter
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

SRC = Path(__file__).resolve().parents[2] / "src"


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def edit_row(root: Path, column: int, value: str) -> None:
    path = root / "inputs/matrices/full.fs"
    lines = path.read_text(encoding="ascii").splitlines()
    cells = lines[2].split("|")
    cells[column] = f" {value} "
    lines[2] = "|".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.mark.parametrize("solver", SOLVERS)
def test_c02_every_registered_adapter_exposes_the_five_functions(solver):
    """C-02: the adapter module satisfies the SolverAdapter protocol."""
    assert isinstance(load_solver(solver), SolverAdapter)


def test_c02_a_module_missing_a_function_is_not_an_adapter():
    """C-02: the protocol check can fail, so the test above is not a constant."""
    import types

    assert not isinstance(types.ModuleType("empty"), SolverAdapter)


INERT = (
    "import subprocess, sys\n"
    "def refuse(*a, **k):\n"
    "    raise AssertionError('import launched a process')\n"
    "subprocess.Popen = refuse\n"
    "before = set(sys.modules)\n"
    "import cfdbender.solvers.{name}\n"
    "new = {{m.split('.')[0] for m in set(sys.modules) - before}}\n"
    "bound = new & {{'pysu2', 'SU2_PY', 'pyfr', 'mpi4py', 'gmsh', 'cgnslib', 'h5py'}}\n"
    "assert not bound, bound\n"
)


@pytest.mark.parametrize("solver", SOLVERS)
def test_c05_importing_an_adapter_binds_no_solver_and_launches_nothing(solver):
    """C-05: a fresh interpreter imports the adapter without a binding or a child process."""
    code = INERT.format(name=solver)
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert done.returncode == 0, done.stderr


def test_c05_the_inertness_probe_detects_a_launch():
    """C-05: control, the probe goes red when an import launches a process."""
    code = INERT.format(name="mock").replace(
        "import cfdbender.solvers.mock", "import cfdbender.solvers.mock; subprocess.Popen(['x'])"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert done.returncode != 0 and "launched a process" in done.stderr


def test_c06_planning_writes_nothing_to_the_workspace(case):  # noqa: F811
    """C-06: plan_workspace and validate_row only read."""
    solver, root = case
    before = snapshot(root)
    assert plan_workspace(root, "full").points, solver
    assert snapshot(root) == before


def test_c08_c16_an_unimplemented_workflow_is_refused_naming_it(case):  # noqa: F811
    """C-08 and C-16: a WORKFLOW other than steady is refused and the refusal names it."""
    solver, root = case
    edit_row(root, 17, "invalid")
    before = snapshot(root)
    with pytest.raises(ValueError, match="(?i)workflow"):
        plan_workspace(root, "full")
    assert snapshot(root) == before


def test_c08_c09_an_unknown_setup_key_is_refused_naming_the_key(case):  # noqa: F811
    """C-08 and C-09: an unknown top-level setup key is refused, and the key is named."""
    solver, root = case
    setup = root / "inputs/setups/s001.toml"
    setup.write_bytes(b"zz_unknown_key = 1\n" + setup.read_bytes())
    with pytest.raises(ValueError, match="zz_unknown_key"):
        plan_workspace(root, "full")


# solver: (condition cell, sweep cell) with a negative or impossible last value.
INVALID_LAST = {
    "su2": ("MACH:sweep, ALTFT:10000, ALPHA:0, BETA:0", "0.5,-1"),
    "openfoam": ("TASmps:sweep, REmi:0.6, ALPHA:0, BETA:0", "10,-5"),
    "xfoil": ("MACH:0, REmi:1, ALPHA:sweep", "0,2,1e999"),
    "pyfr": ("MACH:sweep, REmi:0.01, ALPHA:0, BETA:0", "0.2,-1"),
    "mock": ("MACH:sweep, ALPHA:0", "0.2,-1"),
    "qprop": ("TASmps:20, RPM:sweep, BETA07:30, ALPHA:0, BETA:0", "6000,-1"),
}


def test_c17_every_sweep_point_is_validated_at_planning(case):  # noqa: F811
    """C-17: a sweep whose last value is invalid is refused at planning, before any write."""
    solver, root = case
    condition, sweep = INVALID_LAST[solver]
    edit_row(root, 6, condition)
    edit_row(root, 7, sweep.rsplit(",", 1)[0])
    assert plan_workspace(root, "full").points, "control: the valid prefix plans"
    edit_row(root, 7, sweep)
    before = snapshot(root)
    with pytest.raises(ValueError):
        plan_workspace(root, "full")
    assert snapshot(root) == before
