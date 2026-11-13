"""One optional sink for progress lines raised below the console, such as one per QPROP section.

The matrix console installs a sink for the length of a local run; code that cannot see the
console calls :func:`emit`, which does nothing when no sink is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

_SINK: Callable[[str], None] | None = None


def emit(text: str) -> None:
    """Send one progress line to the installed sink, if any.

    Parameters
    ----------
    text : str
        The line, without a newline.
    """
    if _SINK is not None:
        _SINK(text)


@contextmanager
def sink(write: Callable[[str], None]) -> Iterator[None]:
    """Install ``write`` as the progress sink while the block runs.

    Parameters
    ----------
    write : callable
        Called with every emitted line.
    """
    global _SINK
    previous, _SINK = _SINK, write
    try:
        yield
    finally:
        _SINK = previous
