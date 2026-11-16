"""The one home of warn-and-skip: a warning by default, a refusal under warning_refusals.

An entry that does not apply to a row, a key a solver does not read or a line that cannot be
read is skipped with a warning and the command goes on. With ``--warning-refusals`` (or
``warning_refusals = true`` in init.toml) the same skip is a refusal, so a campaign that must
not skip anything stops at the first one. Every such site calls :func:`warn_or_refuse`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

from cfdbender.core.settings import WorkspaceError

HINT = "--warning-refusals, or warning_refusals = true in init.toml, makes this a refusal"
_state = {"on": False}


def refusing() -> bool:
    """Return True while skips are refused."""
    return _state["on"]


@contextmanager
def warning_refusals(on: bool) -> Iterator[None]:
    """Make every :func:`warn_or_refuse` inside the block refuse (``on``) or warn."""
    previous = _state["on"]
    _state["on"] = bool(on)
    try:
        yield
    finally:
        _state["on"] = previous


def warn_or_refuse(
    message: str, category: type[Warning] = UserWarning, *, stacklevel: int = 1
) -> None:
    """Warn that something is skipped, or refuse it under warning_refusals.

    Parameters
    ----------
    message : str
        What is skipped and why.
    category : type of Warning, optional
        The warning class used when the skip only warns.
    stacklevel : int, optional
        Stack level as for ``warnings.warn`` (1 is the line that calls this function).

    Raises
    ------
    WorkspaceError
        Under warning_refusals, with the message and how the refusal was asked for.
    """
    if _state["on"]:
        raise WorkspaceError(f"{message} ({HINT})")
    warnings.warn(message, category, stacklevel=stacklevel + 1)


class UnusedClusterWarning(UserWarning):
    """A matrix cluster value is unused by the selected profile."""


def warn_unused_cluster(row: str) -> None:
    """Name an ignored row value through the shared warning-refusal policy."""
    warn_or_refuse(
        f"row {row}: cluster value ignored; the profile has no {{cluster}} placeholder",
        UnusedClusterWarning,
        stacklevel=2,
    )
