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
    warning_refusals: bool = False
    defaults: Mapping[str, object] = field(default_factory=dict)
    one_run_at_a_time: bool = False


def _validate(
    values: Mapping[str, object], source: Path, available_solvers: Collection[str]
) -> WorkspaceConfig:
    def refuse(problem: str) -> WorkspaceError:
        choices = ", ".join(sorted(available_solvers)) or "(none)"
        return WorkspaceError(f"{source}: {problem}; available solvers: {choices}")

    unknown = set(values) - {"schema_version", "solver", "warning_refusals", "defaults"}
    if unknown:
        raise refuse(f"unknown keys: {', '.join(sorted(unknown))}")
    defaults = values.get("defaults", {})
    if not isinstance(defaults, dict):
        raise refuse("defaults must be a table of setup keys")
    refusals = values.get("warning_refusals", False)
    if type(refusals) is not bool:
        raise refuse("warning_refusals must be true or false")
    version = values.get("schema_version")
    if type(version) is not int or version != 1:
        raise refuse("schema_version must be the integer 1")
    solver = values.get("solver")
    solver_options: dict[str, object] = {}
    one_run_at_a_time = False
    if isinstance(solver, dict):
        one_run_at_a_time = solver.get("one_run_at_a_time", False)
        if type(one_run_at_a_time) is not bool:
            raise refuse("one_run_at_a_time must be a boolean")
        solver_options = {k: v for k, v in solver.items() if k not in ("name", "one_run_at_a_time")}
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
    return WorkspaceConfig(
        schema_version=version,
        solver=solver,
        solver_options=solver_options,
        one_run_at_a_time=one_run_at_a_time,
        warning_refusals=refusals,
        defaults=defaults,
    )


def setup_defaults(
    defaults: Mapping[str, object], setup: Mapping[str, object], *, omitted: Collection[str] = ()
) -> Mapping[str, object]:
    """Merge setup values over workspace defaults, recursively for tables.

    Parameters
    ----------
    defaults : mapping
        The init.toml defaults table. It is never modified.
    setup : mapping
        Explicit setup values, which win over defaults. Row overrides apply last.
    omitted : collection of str, optional
        Known setup keys that explicit row columns replace, declared by the adapter.

    Returns
    -------
    mapping
        A new setup, validated by the adapter just as an explicit setup is.
    """
    result = {key: value for key, value in defaults.items() if key not in omitted}
    for key, value in setup.items():
        previous = result.get(key)
        result[key] = (
            dict(setup_defaults(previous, value))
            if isinstance(previous, Mapping) and isinstance(value, Mapping)
            else value
        )
    return result


def require_workspace(root: str | Path) -> Path:
    """Return the resolved workspace root, refusing a folder that has no init.toml.

    Parameters
    ----------
    root : str or Path
        Folder a command was pointed at.

    Raises
    ------
    WorkspaceError
        The folder is not a cfdbender workspace.
    """
    target = Path(root).resolve()
    if not (target / "init.toml").exists():
        raise WorkspaceError(
            f"{target} is not a cfdbender workspace (no init.toml); "
            "run cfd-workspace init or pass --workspace"
        )
    return target


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
    source = require_workspace(root) / "init.toml"
    try:
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        choices = ", ".join(sorted(available_solvers)) or "(none)"
        problem = "invalid TOML" if isinstance(error, tomllib.TOMLDecodeError) else "cannot read"
        raise WorkspaceError(
            f"{source}: {problem}: {error}; available solvers: {choices}"
        ) from error
    return _validate(values, source, available_solvers)


def read_warning_refusals(root: str | Path) -> bool:
    """Return the ``warning_refusals`` option of a workspace's init.toml (False when unset).

    A folder without a readable init.toml answers False: the command's own workspace check
    refuses it with its own message.

    Raises
    ------
    WorkspaceError
        The option is present and is not true or false.
    """
    source = Path(root) / "init.toml"
    try:
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    value = values.get("warning_refusals", False)
    if type(value) is not bool:
        raise WorkspaceError(f"{source}: warning_refusals must be true or false")
    return value
