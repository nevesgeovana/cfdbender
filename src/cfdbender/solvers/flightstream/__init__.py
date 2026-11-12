"""Delegate a FlightStream campaign to pyflightstream through the facade.

Create with ``cfd-workspace init --solver flightstream campaign``. In that
workspace, ``cfd-matrix plan m`` and both commands' help use upstream's CLI.
Install the optional engine with ``pip install cfdbender[flightstream]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cfdbender import _pyfs
from cfdbender.core import WorkspaceError, read_init
from cfdbender.core import init_workspace as write_init
from cfdbender.solvers import available_solvers
from cfdbender.solvers.flightstream.reader import FlightStreamPoint, read_workspace

__all__ = ["FlightStreamPoint", "read_workspace"]

SOLVER_NAME = "flightstream"
require_support = _pyfs.require_support


def init_workspace(root: str) -> Path:
    """Initialize upstream's tree and add cfdbender's standard solver declaration.

    Parameters
    ----------
    root : str
        Campaign directory, resolved against the current directory.

    Returns
    -------
    Path
        The initialized absolute root.

    Raises
    ------
    WorkspaceError
        If the solver declaration exists or the optional engine is absent.
    """
    target = Path(root).resolve()
    declaration = target / "init.toml"
    if declaration.exists() or declaration.is_symlink():
        raise WorkspaceError(f"{declaration} already exists; choose a new workspace root")
    _pyfs.init_workspace(target)
    return write_init(target, solver=SOLVER_NAME, available_solvers=(SOLVER_NAME,))


def _root(argv: list[str], workspace: bool) -> Path:
    selected = None
    for index, word in enumerate(argv):
        if word == "--":
            break
        option, separator, value = word.partition("=")
        if len(option) > 2 and "--workspace".startswith(option):
            if separator:
                selected = Path(value)
            elif index + 1 < len(argv):
                selected = Path(argv[index + 1])
    if selected is not None:
        return selected
    # The upstream workspace parser accepts this global flag before the verb,
    # and accepts options before positional roots on these three verbs.
    words = argv
    if words and words[0].startswith("--") and "--warning-refusals".startswith(words[0]):
        words = words[1:]
    if workspace and words and words[0] in {"init", "migrate", "archive"}:
        positional = []
        after_separator = False
        for word in words[1:]:
            if word == "--" and not after_separator:
                after_separator = True
            elif after_separator or not word.startswith("-"):
                positional.append(word)
        if positional and (words[0] != "archive" or len(positional) > 1):
            return Path(positional[0])
    return Path.cwd()


def dispatch(argv: list[str], *, workspace: bool) -> int | None:
    """Forward before local parsing when the selected root declares FlightStream.

    Parameters
    ----------
    argv : list[str]
        Original command arguments, forwarded unchanged.
    workspace : bool
        Select pyfs-workspace rather than pyfs-matrix.

    Returns
    -------
    int or None
        Upstream status, or None to use cfdbender's existing parser. Missing
        support returns 2; upstream argparse exits propagate unchanged.
    """
    if (
        workspace
        and argv[:1] == ["init"]
        and any(word == "--solver" or word.startswith("--solver=") for word in argv)
    ):
        return None
    root = _root(argv, workspace)
    if not (root / "init.toml").is_file():
        return None
    try:
        config = read_init(root, available_solvers=available_solvers())
        if config.solver != SOLVER_NAME:
            return None
        return _pyfs.workspace_main(argv) if workspace else _pyfs.matrix_main(argv)
    except WorkspaceError as error:
        print(str(error), file=sys.stderr)
        return 2
