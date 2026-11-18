"""Create the empty campaign layout without importing a solver adapter."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from cfdbender.core.inputs_guide import NUMERICS_SOLVERS, guide_text
from cfdbender.core.settings import WorkspaceError, _validate

INPUT_KINDS = (
    "mesh",
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
SOLVER_INPUT_KINDS = ("mesh", "references", "setups", "pproc", "hpc", "matrices")
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
    if solver not in _LIBRARY_SOLVERS:
        return INPUT_KINDS
    if solver in NUMERICS_SOLVERS:
        return (*SOLVER_INPUT_KINDS[:3], "numerics", *SOLVER_INPUT_KINDS[3:])
    return SOLVER_INPUT_KINDS


def _declaration_text(solver: str) -> str:
    """Return init.toml: QPROP states its section settings where the user can see them."""
    if solver != "qprop":
        return f"schema_version = 1\nsolver = {json.dumps(solver)}\n"
    return (
        'schema_version = 1\n\n[solver]\nname = "qprop"\n'
        "# Seconds one XFOIL section polar may run; a setup file [solver] value overrides it.\n"
        "section_timeout_s = 60\n"
        "# warn: drop the angles that did not converge and run on the rest; fail: fail the point.\n"
        'on_unconverged_section = "warn"\n'
    )


def init_workspace(
    root: str | Path,
    *,
    solver: str,
    available_solvers: Collection[str],
    declaration_only: bool = False,
) -> Path:
    """Create a workspace and bind it to one registered solver.

    Parameters
    ----------
    root : str or Path
        Campaign root to create or complete, resolved once.
    solver : str
        Exact registered solver name to put in init.toml.
    available_solvers : Collection[str]
        Names passed as data by the application, without loading adapters.
    declaration_only : bool, default False
        Write only init.toml; the caller already made the layout (FlightStream keeps
        pyflightstream's own tree, with inputs/geometries).

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
    if declaration_only:
        target.mkdir(parents=True, exist_ok=True)
        with declaration.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_declaration_text(config.solver))
        return target
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
        stream.write(_declaration_text(config.solver))
    # This machine's executable paths stay out of version control.
    ignore = target / "inputs" / ".gitignore"
    if not ignore.exists():
        ignore.write_text("executables.local.toml\n", encoding="utf-8", newline="\n")
    ignore = target / ".gitignore"
    if not ignore.is_symlink():
        previous = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if "/.input-digests.cache" not in previous.splitlines():
            with ignore.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    ("\n" if previous and not previous.endswith("\n") else "")
                    + "/.input-digests.cache\n/.input-digests-*.tmp\n"
                )
    guide = target / "inputs" / "INPUTS.md"
    if not guide.exists():
        guide.write_text(guide_text(config.solver), encoding="utf-8", newline="\n")
    return target
