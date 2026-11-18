"""Offline SU2, OpenFOAM and PyFR workspaces for the SETUP and NUMERICS columns.

Each workspace is built from the same pieces: the numerical lines of the setup, the other
lines, the files around them and a matrix. A test chooses the matrix header and where the
numerical lines live (in the setup, as before, or in a numerics file).
"""

from __future__ import annotations

import json
from pathlib import Path

from cfdbender.core.adapter import load_point
from cfdbender.solvers import load_solver
from cfdbender.workspace.planning import plan_workspace

FAKE = Path(__file__).resolve().parents[1] / "fixtures/openfoam/fake_openfoam.py"

#: solver -> (numerical setup lines, other setup lines, mesh file name and text, condition)
PARTS: dict[str, tuple[list[str], list[str], tuple[str, str], str]] = {
    "su2": (
        ["iterations = 40", 'convergence_field = "RMS_DENSITY"', "CFL_NUMBER = 2.0"],
        [
            'mesh = "inputs/meshes/tiny.su2"',
            "freestream_temperature = 288.15",
            "reynolds = 1000000",
            'SOLVER = "EULER"',
        ],
        ("inputs/meshes/tiny.su2", "NDIME= 2\n"),
        "MACH:0.3,ALPHA:2.0",
    ),
    "openfoam": (
        ["iterations = 4"],
        [
            'mesh = "inputs/meshes/tiny.msh"',
            'application = "simpleFoam"',
            "speed = 10.0",
            "nu = 1.5e-5",
            'turbulence = "SpalartAllmaras"',
            "[solver]",
            "executable = " + json.dumps(FAKE.as_posix()),
        ],
        ("inputs/meshes/tiny.msh", "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"),
        "ALPHA:4",
    ),
    "pyfr": (
        ["order = 3", "tend = 3", "window_start = 0"],
        [
            'mesh = "inputs/meshes/tiny.msh"',
            "mach = 0.2",
            "reynolds = 10000",
            "[solver]",
            "executable = " + json.dumps(FAKE.as_posix()),
        ],
        ("inputs/meshes/tiny.msh", "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"),
        "ALPHA:4",
    ),
}
REFERENCE = {
    "su2": "area = 2.0\nlength = 3.0\nmoment_origin = [1.0, 2.0, 3.0]\n",
    "openfoam": "area = 0.1\nlength = 1.0\nmoment_origin = [0.25, 0.0, 0.0]\n",
    "pyfr": "",
}


def build(
    root: Path,
    solver: str,
    *,
    columns: str = "POL|RUN|SETUP|REF|PPROC|FLIGHT_CONDITION",
    cells: str = "1001|1|001|001|001|{condition}",
    numerics: dict[str, str] | None = None,
    split: bool = False,
    extra_setup: list[str] | None = None,
    init_extra: str = "",
    rows: int = 1,
) -> Path:
    """Write a workspace of ``solver`` and return its root.

    Parameters
    ----------
    columns, cells : str
        The matrix header and one row; ``{condition}`` is the solver's flight condition.
    numerics : dict, optional
        Extra files to write, by workspace-relative name.
    split : bool
        Put the numerical lines of the setup in ``inputs/numerics/n001.toml`` instead.
    extra_setup : list of str, optional
        Lines appended to the setup, before its ``[solver]`` table.
    rows : int
        Number of matrix rows; the pol numbers count up from 1001.
    """
    numeric, other, (mesh, mesh_text), condition = PARTS[solver]
    head = [line for line in other if line.startswith(("mesh", "application"))]
    tables = other[other.index("[solver]") :] if "[solver]" in other else []
    rest = [line for line in other if line not in head and line not in tables]
    setup = [*head, *rest, *(extra_setup or []), *([] if split else numeric), *tables]
    files = {
        "init.toml": f'schema_version = 1\nsolver = "{solver}"\n{init_extra}',
        "inputs/setups/s001.toml": "\n".join(setup) + "\n",
        "inputs/references/r001.toml": REFERENCE[solver],
        "inputs/pproc/p001.toml": "history = true\n",
        mesh: mesh_text,
    }
    if split:
        files["inputs/numerics/n001.toml"] = "\n".join(numeric) + "\n"
    files.update(numerics or {})
    row = cells.replace("{condition}", condition)
    body = [row.replace("1001", str(1001 + i), 1) for i in range(rows)]
    files["inputs/matrices/study.fs"] = "\n".join([columns, *body]) + "\n"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def decks(root: Path, solver: str) -> list[str]:
    """Plan the workspace and return the translated solver input of every point."""
    plan = plan_workspace(root, "study")
    return [
        load_solver(solver).translate(load_point(root, point, solver=solver))
        for point in plan.points
    ]


def digests(root: Path) -> dict[str, dict[str, str]]:
    """Return each point's input digests without the matrix file, which differs by header."""
    plan = plan_workspace(root, "study")
    return {
        point.point_id: {k: v for k, v in point.inputs_sha256.items() if not k.endswith(".fs")}
        for point in plan.points
    }
