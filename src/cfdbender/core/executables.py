"""Resolve a row's SOLVER_BUILD to this machine's executable for a local run.

``inputs/executables.toml`` declares, per solver, the builds a matrix may name,
and ``inputs/executables.local.toml`` beside it (kept out of version control)
gives this machine's paths for builds the first file declares::

    [su2]
    "8.3.0" = "/opt/su2-8.3.0/bin/SU2_CFD"
    "8.3.0-mpi" = { path = "/opt/su2-8.3.0/bin/SU2_CFD", launcher = ["mpirun", "-np", "8"] }

A bare string is the path; a table holds ``path`` and an optional ``launcher``
list. A local entry replaces the fields it states and may not add a build. A
relative path resolves from the workspace root. A cluster job takes its build
from the HPC profile's ``[builds]`` table instead; these files serve local runs.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cfdbender.core.adapter import AdapterPoint, load_point
from cfdbender.core.plan import InputSession
from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.settings import WorkspaceError

DECLARED = "inputs/executables.toml"
LOCAL = "inputs/executables.local.toml"


@dataclass(frozen=True)
class Executable:
    """One build's executable path and launcher tokens on this machine."""

    build: str
    path: str
    launcher: tuple[str, ...] = ()


def _read(path: Path) -> Mapping[str, object]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WorkspaceError(f"{path}: cannot read executables: {error}") from error


def _entry(value: object, where: str) -> dict[str, object]:
    if isinstance(value, str) and value:
        return {"path": value}
    if not isinstance(value, dict) or set(value) - {"path", "launcher"}:
        raise WorkspaceError(f"{where}: give a path, or a table of path and launcher")
    launcher = value.get("launcher", [])
    if not isinstance(launcher, list) or not all(isinstance(t, str) and t for t in launcher):
        raise WorkspaceError(f"{where}: launcher must be a list of strings")
    if "path" in value and (not isinstance(value["path"], str) or not value["path"]):
        raise WorkspaceError(f"{where}: path must be a non-empty string")
    return dict(value)


def bind_executable(point: AdapterPoint, adapter: object) -> AdapterPoint:
    """Return the point with its build's executable in the setup's [solver] table.

    The adapter's EXECUTABLE_KEYS maps ``path`` and ``launcher`` to its own
    [solver] option names, and optional BUILD_OPTIONS adds fixed options (SU2
    selects its executable route). A point without SOLVER_BUILD, a workspace
    without inputs/executables.toml, or an adapter without EXECUTABLE_KEYS keeps
    the setup's [solver] table as written.

    Parameters
    ----------
    point : AdapterPoint
        Prepared, digest-checked point.
    adapter : object
        The workspace's adapter module.

    Returns
    -------
    AdapterPoint
        The point, or a copy whose setup carries the build's options.
    """
    keys = getattr(adapter, "EXECUTABLE_KEYS", None)
    build = point.point.solver_build
    if keys is None or build is None or not has_registry(point.root):
        return point
    found = resolve_executable(point.root, point.solver, build)
    table = point.setup.get("solver", {})
    solver = dict(table) if isinstance(table, Mapping) else {}
    solver.update(getattr(adapter, "BUILD_OPTIONS", {}))
    solver[keys["path"]] = found.path
    solver[keys["launcher"]] = list(found.launcher)
    return replace(point, setup={**point.setup, "solver": solver})


def check_builds(root: Path, solver: str, adapter: object, builds: Iterable[str | None]) -> None:
    """Judge at plan time the builds a local run will bind, as a run binds them.

    A build nobody declares is warned when the workspace has no inputs/executables.toml
    (a plan may be made here for a cluster) and refused when the file exists but lacks it.

    Parameters
    ----------
    root : Path
        Workspace root.
    solver : str
        The init.toml solver.
    adapter : object
        The workspace's adapter module; one without EXECUTABLE_KEYS binds nothing.
    builds : iterable of str or None
        The SOLVER_BUILD of every planned point.

    Raises
    ------
    WorkspaceError
        The workspace declares executables and a planned build is not among them.
    """
    if getattr(adapter, "EXECUTABLE_KEYS", None) is None:
        return
    for build in sorted({b for b in builds if b}):
        if has_registry(root):
            resolve_executable(root, solver, build)
        else:
            warn_or_refuse(
                f"SOLVER_BUILD {build} is carried but no {DECLARED} declares it; a local run "
                "uses the executable the setup names",
                UserWarning,
                stacklevel=2,
            )


def check_local_executables(
    root: Path, solver: str, adapter: object, points: Iterable[Any]
) -> None:
    """Refuse a local plan whose solver executable is not on this machine.

    A local plan binds each point's build as a run does and asks the adapter's own
    ``check_executable``, the check ``run`` makes first, so plan and run say the same sentence.
    Nothing is launched and no host slot is taken. A plan made for a cluster does not call this.

    Parameters
    ----------
    root : Path
        Workspace root.
    solver : str
        The init.toml solver.
    adapter : object
        The workspace's adapter module; one without ``check_executable`` looks for nothing.
    points : iterable of Point
        The planned points.

    Raises
    ------
    ValueError
        The executable a point would run does not exist (the one missing-executable sentence).
    WorkspaceError
        A planned build is not declared.
    """
    check = getattr(adapter, "check_executable", None)
    if check is None:
        return
    session = InputSession(root)
    for point in points:
        check(bind_executable(load_point(root, point, solver=solver, session=session), adapter))


def has_registry(root: Path) -> bool:
    """Return True when the workspace declares inputs/executables.toml."""
    return (root / DECLARED).is_file()


def resolve_executable(root: Path, solver: str, build: str) -> Executable:
    """Return the executable a local run of ``build`` uses.

    Parameters
    ----------
    root : Path
        Workspace root.
    solver : str
        The init.toml solver; its table in the files is read.
    build : str
        The row's SOLVER_BUILD.

    Returns
    -------
    Executable
        The declared entry with this machine's overlay applied.

    Raises
    ------
    WorkspaceError
        A missing file or build, a local build not declared, or a malformed entry.
    """
    declared_file = root / DECLARED
    if not declared_file.is_file():
        raise WorkspaceError(
            f"SOLVER_BUILD {build!r}: {DECLARED} does not exist; declare the build there"
        )
    tables = []
    for name in (DECLARED, LOCAL):
        table = _read(root / name).get(solver, {})
        if not isinstance(table, dict):
            raise WorkspaceError(f"{name}: [{solver}] must be a table of builds")
        tables.append(table)
    declared, local = tables
    if extra := sorted(set(local) - set(declared)):
        raise WorkspaceError(
            f"{LOCAL}: build {extra[0]!r} of [{solver}] is not declared in {DECLARED}"
        )
    if build not in declared:
        known = ", ".join(sorted(declared)) or "none"
        raise WorkspaceError(
            f"SOLVER_BUILD {build!r} is not declared under [{solver}] in {DECLARED} "
            f"(declared: {known}); add it with its executable path"
        )
    entry = _entry(declared[build], f"{DECLARED} [{solver}] {build!r}")
    if build in local:
        entry.update(_entry(local[build], f"{LOCAL} [{solver}] {build!r}"))
    if "path" not in entry:
        raise WorkspaceError(f"{DECLARED} [{solver}] {build!r}: no path")
    path = Path(str(entry["path"])).expanduser()
    if not path.is_absolute() and any(sep in str(entry["path"]) for sep in "/\\"):
        path = root / path
    launcher = entry.get("launcher", [])
    assert isinstance(launcher, list)
    return Executable(build, str(path), tuple(str(token) for token in launcher))


def build_mismatch(build: str | None, reported: str) -> str | None:
    """Return a refusal when the row's SOLVER_BUILD differs from the version the solver printed.

    A build equal to the reported version matches, and so does a build that
    names an alias of it (``8.3.0-mpi`` for ``8.3.0``). A solver that did not
    report a version (``unknown`` or empty) cannot be checked and is not refused.

    Parameters
    ----------
    build : str or None
        The row's SOLVER_BUILD, or None when the row states none.
    reported : str
        The version the adapter read from the solver.

    Returns
    -------
    str or None
        The message to record as the failure, or None when the point may stand.
    """
    seen = reported.strip()
    if not build or not seen or seen.casefold() == "unknown":
        return None
    wanted = build.casefold()
    if wanted == seen.casefold() or wanted.startswith(seen.casefold() + "-"):
        return None
    return (
        f"SOLVER_BUILD mismatch: the row names {build!r} but the solver reported version "
        f"{seen!r}; point SOLVER_BUILD at the executable of that build in "
        f"{DECLARED}, or correct the row"
    )
