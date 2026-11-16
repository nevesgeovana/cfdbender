"""One resolver and one wording for a solver executable that cannot be found."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from cfdbender.solvers._host import _ram
from cfdbender.solvers._host._single import process_lease as process_lease


def missing_executable(solver: str, tried: object, key: str = "executable") -> ValueError:
    """Return the one refusal every adapter raises for an executable that is not there.

    Parameters
    ----------
    solver : str
        Display name of the solver.
    tried : object
        The value that was looked for.
    key : str, optional
        The setup key the user sets, as written in the [solver] table.
    """
    return ValueError(
        f"{solver} executable not found: {str(tried)!r}; set {key} in the [solver] table of the "
        "setup file, or declare SOLVER_BUILD in inputs/executables.toml"
    )


def resolve_executable(value: str, root: Path, solver: str, key: str = "executable") -> Path:
    """Resolve an executable name or path on this machine, or refuse with one wording.

    Parameters
    ----------
    value : str
        Name on PATH, or an absolute or workspace-relative path.
    root : Path
        Workspace root for relative paths.
    solver : str
        Display name for the refusal.
    key : str, optional
        The setup key the user sets, named in the refusal.

    Returns
    -------
    Path
        The existing file.
    """
    executable = Path(value).expanduser()
    is_path = executable.is_absolute() or any(sep in value for sep in (os.sep, os.altsep) if sep)
    if is_path:
        if not executable.is_absolute():
            executable = root / executable
        allowed = executable.suffix == ".py" or os.name == "nt" or os.access(executable, os.X_OK)
        resolved = str(executable) if executable.is_file() and allowed else None
        if resolved is None and not executable.exists():
            # A path written without its Windows suffix still resolves.
            resolved = shutil.which(str(executable))
    else:
        resolved = shutil.which(value)
    if resolved is None or not Path(resolved).is_file():
        raise missing_executable(solver, value, key)
    return Path(resolved)


def check_host_free(solver: str, estimate_gib: float = 0.0) -> None:
    """Refuse up front when this machine cannot start a solver run now.

    Free RAM minus the solver's estimate must stay above the floor. The optional
    machine-wide process lease is acquired by the local run command.

    Parameters
    ----------
    solver : str
        Display name for the refusal.
    estimate_gib : float, optional
        Memory the solver needs before it starts.
    """
    _ram.require_ram(estimate_gib, solver=solver)
