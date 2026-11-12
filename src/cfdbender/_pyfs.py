"""Lazy access to pyflightstream's public workspace and console entry points."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from cfdbender.core.settings import WorkspaceError

_INSTALL = "install the FlightStream support: pip install cfdbender[flightstream]"


def _workspace_type() -> type[Any]:
    try:
        from pyflightstream.workspace import CampaignWorkspace
    except ModuleNotFoundError as error:
        if error.name not in {"pyflightstream", "pyflightstream.workspace"}:
            raise
        raise WorkspaceError(_INSTALL) from error
    return cast(type[Any], CampaignWorkspace)


def require_support() -> None:
    """Refuse with the install instruction when the optional engine is absent."""
    _workspace_type()


def init_workspace(root: Path) -> None:
    """Create the upstream workspace tree through its exported workspace class.

    Parameters
    ----------
    root : Path
        Campaign directory passed unchanged to CampaignWorkspace.init.
    """
    _workspace_type().init(root)


def matrix_main(argv: list[str]) -> int:
    """Call the pyfs-matrix console target with unchanged arguments.

    Parameters
    ----------
    argv : list[str]
        Arguments excluding the executable name.

    Returns
    -------
    int
        Upstream exit status; argparse SystemExit propagates unchanged.
    """
    require_support()
    from pyflightstream.run.cli import main

    return cast(Callable[[list[str]], int], main)(argv)


def workspace_main(argv: list[str]) -> int:
    """Call the pyfs-workspace console target with unchanged arguments.

    Parameters
    ----------
    argv : list[str]
        Arguments excluding the executable name.

    Returns
    -------
    int
        Upstream exit status; argparse SystemExit propagates unchanged.
    """
    require_support()
    from pyflightstream.workspace.cli import main

    return cast(Callable[[list[str]], int], main)(argv)


def read_ledger(root: Path) -> Any:
    """Read the upstream public ledger without writing campaign products.

    Parameters
    ----------
    root : Path
        FlightStream workspace directory.

    Returns
    -------
    Any
        Upstream Ledger, whose public points and card methods return dictionaries.
    """
    require_support()
    from pyflightstream.workspace.ledger import read_ledger

    return read_ledger(root)
