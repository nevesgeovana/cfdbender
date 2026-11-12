"""Read the workspace's single solver declaration without importing adapters."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path


class WorkspaceError(ValueError):
    """A workspace declaration or layout cannot be used as requested."""


@dataclass(frozen=True)
class WorkspaceConfig:
    """Validated root declaration of one solver.

    Attributes
    ----------
    schema_version : int
        Supported declaration format, currently 1.
    solver : str
        Registered solver name supplied by the composing application.
    """

    schema_version: int
    solver: str
    solver_options: Mapping[str, object] = field(default_factory=dict)


def _validate(
    values: Mapping[str, object], source: Path, available_solvers: Collection[str]
) -> WorkspaceConfig:
    def refuse(problem: str) -> WorkspaceError:
        choices = ", ".join(sorted(available_solvers)) or "(none)"
        return WorkspaceError(f"{source}: {problem}; available solvers: {choices}")

    unknown = set(values) - {"schema_version", "solver"}
    if unknown:
        raise refuse(f"unknown keys: {', '.join(sorted(unknown))}")
    version = values.get("schema_version")
    if type(version) is not int or version != 1:
        raise refuse("schema_version must be the integer 1")
    solver = values.get("solver")
    solver_options: dict[str, object] = {}
    if isinstance(solver, dict):
        solver_options = {k: v for k, v in solver.items() if k != "name"}
        for key, value in solver_options.items():
            if type(value) not in (str, int, float, bool) or (
                type(value) is float and not math.isfinite(value)
            ):
                raise refuse(f"solver option {key} must be a finite scalar")
        solver = solver.get("name")
    if not isinstance(solver, str) or not solver:
        raise refuse("solver must be a non-empty string")
    if solver not in available_solvers:
        raise refuse(f"unknown solver {solver!r}")
    return WorkspaceConfig(schema_version=version, solver=solver, solver_options=solver_options)


def read_init(root: str | Path, *, available_solvers: Collection[str]) -> WorkspaceConfig:
    """Read and validate init.toml at a workspace root.

    Parameters
    ----------
    root : str or Path
        Workspace root, resolved once against the current directory.
    available_solvers : Collection[str]
        Registered solver names, passed as data by the application.

    Returns
    -------
    WorkspaceConfig
        The validated declaration; no adapter is loaded.

    Raises
    ------
    WorkspaceError
        If the file cannot be read, is malformed, or declares unsupported values.
    """
    source = Path(root).resolve() / "init.toml"
    try:
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        choices = ", ".join(sorted(available_solvers)) or "(none)"
        problem = "invalid TOML" if isinstance(error, tomllib.TOMLDecodeError) else "cannot read"
        raise WorkspaceError(
            f"{source}: {problem}: {error}; available solvers: {choices}"
        ) from error
    return _validate(values, source, available_solvers)
