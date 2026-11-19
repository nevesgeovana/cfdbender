"""A synthetic CFD++ workspace on the toy database, for the adapter's offline tests.

Every test that plans or runs a CFD++ point points the adapter at the invented toy
database (``use_toy``): the real database of 22.5.U3 cannot write an input until its
layout returns from the licensed machine. The stand-in executables are written by the
fixture builder into the test's temporary folder.
"""

from __future__ import annotations

import json
from pathlib import Path

from cfdpp_support import builder

from cfdbender.core import init_workspace
from cfdbender.solvers import available_solvers

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "cfdpp" / "toy_schema"
VERSION = "1.0"
STOP_TOOL = "toystop"
GRID = "naca.cfdpp"

#: A complete setup for the toy database: every key the database binds, with the two
#: families of the builder's 2D grid.
SETUP = """\
iterations = 200
convergence_orders = 6.0
cfl = 1.0
cfl_max = 50.0
cfl_ramp_steps = [1, 100]
turbulence = "SpalartAllmaras"
viscosity_ratio = 3.0

[physics]
method = "preconditioned"
preconditioning_velocity = 50.0

[fluid]
gas_constant = 287.05287
gamma = 1.4
viscosity = "sutherland"
sutherland_viscosity = 1.716e-5
sutherland_temperature = 273.15
sutherland_constant = 110.4

[numerics]
multigrid_levels = 4
temporal_smoothing = 0.75
limiter = "continuous"
blend_first_order = [1.0, 1.0]
blend_steps = [1, 1]
turbulence_start_step = 11

[boundaries.airfoil]
type = "wall"
treatment = "solve-to-wall"
thermal = "adiabatic"

[boundaries.farfield]
type = "farfield"
"""
REFERENCE = "area = 1.0\nlength = 1.0\nmoment_origin = [0.25, 0.0, 0.0]\n"
POST = "history = true\noutput_frequency = 100\nrestart_frequency = 100\n"
HEADER = (
    "POL  | RUN | DESCRIPTION | FLIGHT_CONDITION | SWEEP_VALUES | MESH | REF | SETUP | "
    "PPROC | NCPUS | WALLTIME | SOLVER_BUILD | WORKFLOW"
)


def use_toy(monkeypatch) -> None:
    """Point the adapter at the toy database for the rest of the test."""
    from cfdbender.solvers.cfdpp import _schema

    monkeypatch.setattr(_schema, "DATA", TOY)


def matrix(
    condition: str = "MACH:0.15, REmi:6, TK:300, ALPHA:sweep",
    sweep: str = "0,10",
    *,
    geometry: str = GRID,
    ncpus: int = 1,
    build: str = VERSION,
    workflow: str = "steady",
) -> str:
    """Return a one-row matrix."""
    rule = "|".join("-" * 5 for _ in HEADER.split("|"))
    row = (
        f"1001 | 1 | toy | {condition} | {sweep} | {geometry} | 001 | 001 | 001 | {ncpus} | 1h | "
        f"{build} | {workflow}"
    )
    return f"{HEADER}\n{rule}\n{row}\n"


def workspace(
    root: Path,
    *,
    setup: str = SETUP,
    reference: str = REFERENCE,
    post: str = POST,
    table: str | None = None,
    grid: str = "2d",
    executables: bool = True,
    init_options: str = "",
) -> Path:
    """Create a CFD++ workspace with a grid folder, s/r/p files, a matrix and the stand-ins."""
    init_workspace(root, solver="cfdpp", available_solvers=available_solvers())
    build = builder()
    folder = root / "inputs" / "mesh" / GRID
    if grid == "2d":
        build.write_grid2d(folder)
    else:
        build.write_grid3d(folder)
    (root / "inputs/setups/s001.toml").write_text(setup, encoding="ascii", newline="")
    (root / "inputs/references/r001.toml").write_text(reference, encoding="ascii")
    (root / "inputs/pproc/p001.toml").write_text(post, encoding="ascii")
    (root / "inputs/matrices/study.fs").write_text(table or matrix(), encoding="ascii")
    if init_options:
        (root / "init.toml").write_text(
            f'schema_version = 1\n\n[solver]\nname = "cfdpp"\n{init_options}',
            encoding="ascii",
        )
    if executables:
        paths = build.write_standins(root.parent / f"{root.name}-bin", TOY, VERSION, STOP_TOOL)
        entry = {
            "serial": paths["solver"].as_posix(),
            "mpi": paths["solver"].as_posix(),
            "mpi_launcher": [paths["solver"].as_posix(), "-n", "{ncpus}"],
            "partitioner": [paths["tool"].as_posix(), "partition", "{ncpus}"],
            "tools_dir": paths["tools_dir"].as_posix(),
        }
        lines = [f'[cfdpp."{VERSION}"]']
        lines += [f"{key} = {json.dumps(value)}" for key, value in entry.items()]
        (root / "inputs/executables.toml").write_text("\n".join(lines) + "\n", encoding="ascii")
    return root


def with_keywords(text: str, setup: str = SETUP) -> str:
    """Return the setup with a [keywords] table appended."""
    return setup + "\n[keywords]\n" + text
