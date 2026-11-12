"""P0000: sibling imports fail the guard and each adapter names its own refusals.

Reproduce by executing a prepared offline point with only 0.49 GiB available.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_p0000_item3d_execute import case as su2_point
from test_p0000_openfoam_adapter import case as foam_case
from test_p0000_openfoam_adapter import point as foam_point
from test_p0000_pyfr_adapter import case as pyfr_case
from test_p0000_xfoil_plan import prepared as xfoil_point
from test_p0000_xfoil_plan import workspace as xfoil_workspace

from cfdbender.core.adapter import load_point

ROOT = Path(__file__).resolve().parents[2]


def prepared(root, solver):
    if solver == "su2":
        return su2_point(root)
    if solver == "xfoil":
        xfoil_workspace(root)
        return xfoil_point(root)
    if solver == "pyfr":
        plan = pyfr_case(root)
        return load_point(plan.root, plan.points[0], solver="pyfr")
    return foam_point(foam_case(root))


@pytest.mark.parametrize("solver", ["xfoil", "openfoam", "pyfr"])
def test_p0000_solver_host_import_isolation(solver):
    code = (
        f"import cfdbender.solvers.{solver}; import sys; "
        "assert not any(n == 'cfdbender.solvers.su2' or "
        "n.startswith('cfdbender.solvers.su2.') for n in sys.modules), "
        "sorted(n for n in sys.modules if n.startswith('cfdbender.solvers.su2'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "solver,label", [("su2", "SU2"), ("xfoil", "XFOIL"), ("openfoam", "OpenFOAM"), ("pyfr", "PyFR")]
)
@pytest.mark.parametrize("unavailable", [False, True])
def test_p0000_solver_host_ram_refusal_names_solver(
    tmp_path, monkeypatch, solver, label, unavailable
):
    point = prepared(tmp_path, solver)
    execution = importlib.import_module(f"cfdbender.solvers.{solver}._execute")

    def measure():
        if unavailable:
            raise OSError("measurement failed")
        return 0.49

    monkeypatch.setattr(execution._ram, "free_ram_gib", measure)
    adapter = importlib.import_module(f"cfdbender.solvers.{solver}")
    with pytest.raises(ValueError, match=f"refusing {label}"):
        adapter.execute(point, check_ram=lambda: None)
    assert not (point.root / point.point.directory).exists()


@pytest.mark.parametrize(
    "solver,label", [("su2", "SU2"), ("xfoil", "XFOIL"), ("openfoam", "OpenFOAM"), ("pyfr", "PyFR")]
)
def test_p0000_solver_host_thread_refusal_names_solver(tmp_path, solver, label):
    point = prepared(tmp_path, solver)
    execution = importlib.import_module(f"cfdbender.solvers.{solver}._execute")
    with execution._RUNNING:
        with pytest.raises(ValueError, match=f"{label} is already running"):
            importlib.import_module(f"cfdbender.solvers.{solver}").execute(
                point, check_ram=lambda: None
            )
    assert not (point.root / point.point.directory).exists()
