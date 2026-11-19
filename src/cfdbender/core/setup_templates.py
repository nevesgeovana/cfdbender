"""Setup templates: ready s files a solver ships as data, found without importing the solver.

A template is a file ``solvers/<solver>/setup_templates/<name>.toml``: a complete s file as
text whose first line is a comment holding its one-line description. Adding a file adds a
template; no code changes. The core reads the files through the package resources, so it never
imports an adapter.
"""

from __future__ import annotations

import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from cfdbender.core.settings import WorkspaceError

FOLDER = "setup_templates"
SUFFIX = ".toml"
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SETUP = re.compile(r"s(\d{3})\.toml")
ID_DIGITS = 3


def _folder(solver: str) -> Traversable | None:
    if not _WORD.fullmatch(solver):
        raise WorkspaceError(f"{solver!r} is not a solver name")
    folder = resources.files("cfdbender") / "solvers" / solver / FOLDER
    return folder if folder.is_dir() else None


def available(solver: str) -> tuple[str, ...]:
    """Return the template names of a solver, sorted; empty when it ships none."""
    folder = _folder(solver)
    if folder is None:
        return ()
    return tuple(
        sorted(
            item.name.removesuffix(SUFFIX)
            for item in folder.iterdir()
            if item.is_file() and item.name.endswith(SUFFIX)
        )
    )


def _check(solver: str, name: str) -> None:
    names = available(solver)
    if name not in names:
        listed = ", ".join(names) if names else "none"
        raise WorkspaceError(f"unknown setup template {name!r} for {solver}; available: {listed}")


def read(solver: str, name: str) -> str:
    """Return the template text exactly as the data file holds it.

    Raises
    ------
    WorkspaceError
        The solver ships no template of that name; the message lists the ones it has.
    """
    _check(solver, name)
    folder = _folder(solver)
    assert folder is not None
    return (folder / f"{name}{SUFFIX}").read_bytes().decode("utf-8")


def describe(solver: str, name: str) -> str:
    """Return the description: the first line of the file, without its comment mark.

    Raises
    ------
    WorkspaceError
        The solver ships no such template, or the file does not start with a comment line.
    """
    first = read(solver, name).lstrip("﻿").splitlines()[:1]
    if not first or not first[0].startswith("#") or not first[0].lstrip("# ").strip():
        raise WorkspaceError(
            f"setup template {name!r} of {solver} must start with a comment line that describes it"
        )
    return first[0].lstrip("# ").strip()


def next_free_id(setups: Path) -> str:
    """Return the id after the highest ``s<id>.toml`` in a folder, as three digits."""
    used = [
        int(match.group(1))
        for path in (setups.iterdir() if setups.is_dir() else ())
        if (match := _SETUP.fullmatch(path.name))
    ]
    number = max(used, default=0) + 1
    if number >= 10**ID_DIGITS:
        raise WorkspaceError(f"{setups}: no setup id is left after {max(used):0{ID_DIGITS}d}")
    return f"{number:0{ID_DIGITS}d}"


def write_setup(root: Path, solver: str, name: str, setup_id: str | None) -> Path:
    """Write ``inputs/setups/s<id>.toml`` of a workspace from a template; overwrite nothing.

    Parameters
    ----------
    root : Path
        The campaign workspace.
    solver : str
        The workspace's solver.
    name : str
        The template name.
    setup_id : str, optional
        Three digits; the next free id when omitted.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    WorkspaceError
        Unknown template, malformed id, or the file already exists.
    """
    text = read(solver, name)
    if setup_id is not None and not re.fullmatch(rf"\d{{{ID_DIGITS}}}", setup_id):
        raise WorkspaceError(f"--id {setup_id!r}: a setup id is {ID_DIGITS} digits, such as 001")
    setups = root / "inputs" / "setups"
    target = setups / f"s{setup_id or next_free_id(setups)}{SUFFIX}"
    setups.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(text.encode("utf-8"))
    except FileExistsError:
        raise WorkspaceError(
            f"{target.relative_to(root).as_posix()} already exists; nothing was overwritten. "
            "Pick another --id or leave it out for the next free one"
        ) from None
    return target
