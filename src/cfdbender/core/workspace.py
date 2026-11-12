"""Create the empty campaign layout without importing a solver adapter."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from cfdbender.core.settings import WorkspaceError, _validate

INPUT_KINDS = (
    "geometries",
    "references",
    "setups",
    "pproc",
    "profiles",
    "freestreams",
    "calibrations",
    "hpc",
    "fsi",
    "matrices",
    "management",
)
#: The input folders the solver adapters of this package read. The other kinds
#: belong to the FlightStream layout and stay in a mock workspace.
SOLVER_INPUT_KINDS = ("geometries", "references", "setups", "pproc", "hpc", "matrices")
_LIBRARY_SOLVERS = frozenset({"su2", "openfoam", "pyfr", "xfoil", "qprop"})


def input_kinds(solver: str) -> tuple[str, ...]:
    """Return the input folders a new workspace of this solver is created with.

    Parameters
    ----------
    solver : str
        Registered solver name.

    Returns
    -------
    tuple of str
        Folder names under inputs/, in creation order.
    """
    return SOLVER_INPUT_KINDS if solver in _LIBRARY_SOLVERS else INPUT_KINDS


def init_workspace(root: str | Path, *, solver: str, available_solvers: Collection[str]) -> Path:
    """Create a workspace and bind it to one registered solver.

    Parameters
    ----------
    root : str or Path
        Campaign root to create or complete, resolved once.
    solver : str
        Exact registered solver name to put in init.toml.
    available_solvers : Collection[str]
        Names passed as data by the application, without loading adapters.

    Returns
    -------
    Path
        The absolute workspace root.

    Raises
    ------
    WorkspaceError
        If init.toml already exists, the solver is unknown, or a layout path is a file.
    OSError
        If the filesystem cannot create the workspace.
    """
    target = Path(root).resolve()
    declaration = target / "init.toml"
    if declaration.exists() or declaration.is_symlink():
        raise WorkspaceError(f"{declaration} already exists; choose a new workspace root")
    config = _validate({"schema_version": 1, "solver": solver}, declaration, available_solvers)
    folders = [
        target,
        target / "inputs",
        *(target / "inputs" / kind for kind in input_kinds(config.solver)),
        *(target / name for name in ("sims", "post", "archive")),
    ]
    for folder in folders:
        if folder.exists() and not folder.is_dir():
            raise WorkspaceError(f"{folder}: not a directory")
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects the one-solver binding from an overlapping init.
    with declaration.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"schema_version = 1\nsolver = {json.dumps(config.solver)}\n")
    # This machine's executable paths stay out of version control.
    ignore = target / "inputs" / ".gitignore"
    if not ignore.exists():
        ignore.write_text("executables.local.toml\n", encoding="utf-8", newline="\n")
    return target
