"""P0000 real-solver test: opt in by pointing CFDBENDER_SU2_EXE at SU2_CFD.

The skipif marker keeps ordinary offline runs solver-free. No numerical
convergence is assumed from process success or from a small iteration budget.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from test_p0000_item3d_config import plan, prepared, workspace
from test_p0000_item3d_execute import execution_api

from cfdbender.core.registry import RunRegistry
from cfdbender.core.run import execute_plan

EXE = os.environ.get("CFDBENDER_SU2_EXE", "")
GENERATOR = Path(__file__).resolve().parents[1] / "fixtures/su2/tiny_case.py"


@pytest.mark.skipif(
    not EXE or not Path(EXE).is_file(), reason="set CFDBENDER_SU2_EXE to an existing SU2_CFD"
)
def test_p0000_item3d_real_solver_naca0012(tmp_path):
    pytest.importorskip("gmsh", reason="gmsh Python API is required to generate the tiny SU2 mesh")
    from cfdbender.solvers.su2 import _ram

    _ram.require_ram(0.25)
    workspace(tmp_path)
    spec = importlib.util.spec_from_file_location("tiny_su2_case", GENERATOR)
    assert spec is not None and spec.loader is not None and GENERATOR.is_file(), (
        "a tiny mesh generator is required for the real-solver proof"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    mesh = tmp_path / "inputs/meshes/tiny.su2"
    cells = generator.generate(mesh)
    assert 100 < cells <= 5000
    mesh_text = mesh.read_text()
    assert "airfoil" in mesh_text and "farfield" in mesh_text
    (tmp_path / "inputs/setups/s001.toml").write_text(
        chr(10).join(
            [
                'mesh = "inputs/meshes/tiny.su2"',
                "iterations = 40",
                'SOLVER = "EULER"',
                'MATH_PROBLEM = "DIRECT"',
                "RESTART_SOL = false",
                "freestream_temperature = 288.15",
                'INIT_OPTION = "TD_CONDITIONS"',
                'CONV_NUM_METHOD_FLOW = "JST"',
                'NUM_METHOD_GRAD = "WEIGHTED_LEAST_SQUARES"',
                "CFL_NUMBER = 2.0",
                'MARKER_EULER = ["airfoil"]',
                'MARKER_FAR = ["farfield"]',
                'MARKER_MONITORING = ["airfoil"]',
                'CONV_FIELD = "RMS_DENSITY"',
                "CONV_RESIDUAL_MINVAL = -14",
                "CONV_STARTITER = 0",
                'OUTPUT_FILES = ["RESTART_ASCII"]',
                'HISTORY_OUTPUT = ["ITER", "RMS_RES", "AERO_COEFF"]',
                "[solver]",
                'route = "executable"',
                "threads = 1",
                "timeout_s = 120",
                "ram_estimate_gib = 0.25",
            ]
        ),
        encoding="ascii",
    )
    (tmp_path / "inputs/references/r001.toml").write_text(
        chr(10).join(
            [
                "area = 1.0",
                "length = 1.0",
                "moment_origin = [0.25, 0.0, 0.0]",
            ]
        ),
        encoding="ascii",
    )
    run_plan = plan(tmp_path)
    records = execute_plan(run_plan, execution_api())
    assert len(records) == 1
    point = prepared(tmp_path)
    folder = tmp_path / point.point.directory
    print("SU2 real run directory:", folder)
    print("SU2 mesh cells:", cells)
    print("SU2 stdout transcript:")
    print((folder / "stdout.log").read_text(encoding="utf-8", errors="replace"))
    print("SU2 stderr transcript:")
    print((folder / "stderr.log").read_text(encoding="utf-8", errors="replace"))
    print("SU2 execution:", (folder / "execution.json").read_text())
    result = execution_api().collect(point)
    print("SU2 neutral result:", json.dumps(result.to_dict(), sort_keys=True))
    assert result.solver_version != "unknown"
    assert result.classification == "iteration_limit"
    assert len(result.history) == 40
    assert "CL" in result.coefficients and "CD" in result.coefficients
    assert result.references.area == 1 and result.references.length == 1
    assert result.references.moment_origin == (0.25, 0, 0)
    assert records[0].classification == result.classification
    assert RunRegistry(tmp_path).read() == records
    assert "inputs/meshes/tiny.su2" in records[0].point.inputs_sha256
