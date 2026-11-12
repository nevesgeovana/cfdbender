"""Workspace builders and independent oracles for the QPROP adapter tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

from qprop_support import blade_obj

from cfdbender.core.adapter import load_point
from cfdbender.solvers import available_solvers, load_solver
from cfdbender.workspace.cli import main as workspace_cli
from cfdbender.workspace.planning import plan_workspace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/qprop"
FAKE_QPROP = FIXTURES / "fake_qprop.py"
FAKE_XFOIL = FIXTURES / "fake_xfoil.py"
CONDITION = "V:20,RPM:6000,BETA07:30,RHO:1.2,MU:1.8e-5,A:340"
DEFAULTS = {
    "blade": "inputs/geometry/blade.obj",
    "n_blades": 3,
    "n_sections": 5,
    "spacing": "cosine",
    "cut": "plane",
    "radius": 0.2,
}


def setup_text(**overrides):
    """Return s-file top-level TOML from the defaults; None removes a key."""
    values = {**DEFAULTS, **overrides}
    return "".join(
        f"{key} = {json.dumps(value)}\n" for key, value in values.items() if value is not None
    )


SETUP = setup_text()


def adapter():
    assert "qprop" in available_solvers(), "QPROP must be registered"
    return load_solver("qprop")


def workspace(root, *, setup=None, condition=CONDITION, sweep="", options="", xfoil=True):
    assert workspace_cli(["init", "--solver", "qprop", "--workspace", str(root)]) == 0
    solver = "\n[solver]\nexecutable = " + json.dumps(FAKE_QPROP.as_posix()) + "\n"
    if xfoil:
        solver += "xfoil_executable = " + json.dumps(FAKE_XFOIL.as_posix()) + "\n"
    files = {
        "inputs/setups/s001.toml": (SETUP if setup is None else setup) + solver + options,
        "inputs/references/r001.toml": "",
        "inputs/pproc/p001.toml": "",
        "inputs/matrices/study.fs": (
            "POL|RUN|SET|REF|PPROC|FLIGHT_CONDITION|SWEEP_VALUES\n"
            f"1001|1|001|001|001|{condition}|{sweep}\n"
        ),
    }
    for name, text in files.items():
        (root / name).write_bytes(text.encode("ascii"))
    geometry = root / "inputs/geometry"
    geometry.mkdir(parents=True, exist_ok=True)
    (geometry / "blade.obj").write_bytes(blade_obj(rings=31).encode("ascii"))
    return root


def prepared(root, index=0):
    return load_point(root, plan_workspace(root, "study").points[index], solver="qprop")


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def local_reynolds(rho, mu, speed, rpm, radius, chord):
    """Independent section Reynolds number from the relative speed."""
    return rho * math.hypot(speed, 2 * math.pi * rpm / 60 * radius) * chord / mu


def prop_stations(folder):
    """Return the 13-column station rows of a written propeller file as floats."""
    lines = (Path(folder) / "propeller.prop").read_text(encoding="ascii").splitlines()
    return [
        [float(v) for v in line.split()]
        for line in lines
        if line and line[0] != "#" and len(line.split()) == 13
    ]


def polar_text(cl0):
    """Return an XFOIL-format polar with CL = cl0 + 0.11 alpha and a CLCD0 at alpha = 2."""
    lines = [
        "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr",
        "  ------ -------- --------- --------- -------- -------- --------",
    ]
    for alpha in range(-8, 17):
        cl = cl0 + 0.11 * alpha
        cd = 0.01 + 0.02 * (cl - (cl0 + 0.22)) ** 2
        lines.append(f" {alpha:7.3f} {cl:8.4f} {cd:9.5f}   0.00046  -0.0000   0.6870   0.6870")
    return "\n".join(lines) + "\n"


def table_set(root, *, rows=None):
    """Write a polars-2D set: a blade table, one airfoil and two given polars."""
    folder = root / "inputs/polars-2D/set1"
    folder.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        "0.06,0.030,50,a1.dat,p1.txt",
        "0.12,0.026,38,a1.dat,p2.txt",
        "0.19,0.021,22,a1.dat,",
    ]
    (folder / "blade.csv").write_text("r,chord,beta,airfoil,polar\n" + "\n".join(rows) + "\n")
    (folder / "a1.dat").write_text("section\n1.0 0.0\n0.5 0.06\n0.0 0.0\n0.5 -0.04\n1.0 0.0\n")
    (folder / "p1.txt").write_text(polar_text(0.2))
    (folder / "p2.txt").write_text(polar_text(0.3))
    return folder
