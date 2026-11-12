"""Prepare the common 1,280-cell, 40-iteration inviscid SU2 acceptance case.

The mesh is supplied explicitly so installed-wheel tests require no gmsh.
Numerical settings match the executable-route tiny-case acceptance test.
"""

from __future__ import annotations

import json
from pathlib import Path

from cfdbender.core.plan import RunPlan, plan_matrix
from cfdbender.solvers.su2 import validate_row


def prepare(root: Path, mesh: Path, route: str, executable: str | None = None) -> RunPlan:
    """Create fresh reference inputs and return their digest-tracked plan.

    Parameters
    ----------
    root : Path
        New workspace; an existing directory refuses before writing.
    mesh : Path
        Committed small mesh, read without generation or modification.
    route : str
        Explicit executable or pysu2 selection.
    executable : str, optional
        Executable path; otherwise the usual adapter environment applies.
    """
    data = mesh.read_bytes()
    root.mkdir(parents=True, exist_ok=False)
    files = {
        "init.toml": 'schema_version=1\nsolver="su2"\n',
        "inputs/setups/s001.toml": chr(10).join(
            [
                'mesh="inputs/meshes/naca0012_small.su2"',
                "iterations=40",
                'SOLVER="EULER"',
                'MATH_PROBLEM="DIRECT"',
                "RESTART_SOL=false",
                "freestream_temperature=288.15",
                'INIT_OPTION="TD_CONDITIONS"',
                'CONV_NUM_METHOD_FLOW="JST"',
                'NUM_METHOD_GRAD="WEIGHTED_LEAST_SQUARES"',
                "CFL_NUMBER=2.0",
                'MARKER_EULER=["airfoil"]',
                'MARKER_FAR=["farfield"]',
                'MARKER_MONITORING=["airfoil"]',
                'CONV_FIELD="RMS_DENSITY"',
                "CONV_RESIDUAL_MINVAL=-14",
                "CONV_STARTITER=0",
                'OUTPUT_FILES=["RESTART_ASCII"]',
                'HISTORY_OUTPUT=["ITER", "RMS_RES", "AERO_COEFF"]',
                "[solver]",
                "route=" + json.dumps(route),
                "threads=1",
                "timeout_s=120",
                "ram_estimate_gib=0.25",
                *(["executable=" + json.dumps(executable)] if executable else []),
            ]
        ),
        "inputs/references/r001.toml": "area=1.0\nlength=1.0\nmoment_origin=[0.25,0,0]\n",
        "inputs/pproc/p001.toml": "history=true\noutput_frequency=5\n",
        "inputs/matrices/reference.fs": (
            "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION\n"
            "---|---|---|---|-----|----------------\n"
            "1001|1|001|001|001|MACH:0.3,ALPHA:2.0\n"
        ),
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    target = root / "inputs/meshes/naca0012_small.su2"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return plan_matrix(
        root, "reference", validators={"su2": lambda row, inputs: validate_row(row, inputs)}
    )
