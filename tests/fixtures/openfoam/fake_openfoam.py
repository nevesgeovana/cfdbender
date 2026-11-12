"""Offline ESI v2412 stand-in: inspect the case and emit named force coefficients."""

import os
import time
from pathlib import Path

for name in (
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "0/U",
    "0/p",
    "mesh.msh",
    "Allrun",
):
    assert Path(name).is_file(), name
if "SpalartAllmaras" in Path("constant/turbulenceProperties").read_text():
    assert Path("0/nut").is_file() and Path("0/nuTilda").is_file()
assert os.environ["OMP_NUM_THREADS"] == "1"
mode = os.environ.get("FAKE_OPENFOAM_MODE", "converged")
if mode == "timeout":
    time.sleep(60)
log = [
    "Build : OpenFOAM-v2412",
    "Time = 1",
    "smoothSolver: Solving for Ux, Initial residual = 0.01, Final residual = 1e-9, No Iterations 2",
]
if mode != "exit_only":
    log += ["Time = 4" if mode == "iteration_limit" else "Time = 2"]
    for field in ("Ux", "Uy", "p", "nuTilda"):
        residual = "1e-7" if mode == "residual" else "0.01"
        log.append(
            f"smoothSolver: Solving for {field}, Initial residual = {residual}, "
            "Final residual = 1e-10, No Iterations 2"
        )
if mode == "converged":
    log.append("SIMPLE solution converged in 2 iterations")
if mode == "fatal":
    log.append("--> FOAM FATAL ERROR: planted failure")
Path("log.simpleFoam").write_text("\n".join(log) + "\n", encoding="ascii")
if mode != "missing":
    target = Path("postProcessing/forceCoeffs/0/coefficient.dat")
    target.parent.mkdir(parents=True)
    target.write_text(
        "# Force coefficients\n"
        "# Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch CmRoll CmYaw Cs Cs(f) Cs(r)\n"
        "1 0.04 0.02 0.02 0.41 0.2 0.21 -0.02 0 0 0 0 0\n"
        + ("4" if mode == "iteration_limit" else "2")
        + " 0.031 0.015 0.016 0.47 0.22 0.25 -0.025 0 0 0 0 0\n",
        encoding="ascii",
    )
raise SystemExit(1 if mode == "nonzero" else 0)
