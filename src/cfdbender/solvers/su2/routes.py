"""Choose an SU2 route, preserving explicit user intent."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from importlib import util
from importlib.machinery import PathFinder

ROUTES = ("pysu2", "executable")


def select_route(
    route: str | None = None,
    *,
    pysu2_path: str | None = None,
    importer: Callable[[str], object] | None = None,
) -> str:
    """Prefer pysu2 and announce automatic executable fallback.

    Parameters
    ----------
    route : str, optional
        Explicit pysu2 or executable selection. None prefers pysu2.
    pysu2_path : str, optional
        Extra worker module directory, searched before the interpreter path.
    importer : callable, optional
        Legacy capability-test hook. Production uses find_spec, never imports.

    Returns
    -------
    str
        Selected registered route. Selection does not launch a driver.
    """
    if route is not None and route not in ROUTES:
        raise ValueError("solver.route must be pysu2 or executable")
    if route == "executable":
        return route
    try:
        if importer is not None:
            importer("pysu2")
        elif not (
            (pysu2_path and PathFinder.find_spec("pysu2", [pysu2_path])) or util.find_spec("pysu2")
        ):
            raise ImportError("pysu2 module not found on the interpreter path or pysu2_path")
    except (ImportError, OSError, ValueError) as error:
        if route == "pysu2":
            raise ValueError(
                f"pysu2 is unavailable: {error}. Build SU2 with the Python wrapper "
                "(-Denable-pywrapper=true) on Linux or WSL using a compatible Python; "
                "the Windows interpreter cannot import that Linux build. Set solver.pysu2_path "
                "or CFDBENDER_PYSU2_PATH to its install directory. "
                "The executable route is available: choose solver.route='executable'."
            ) from error
        warnings.warn(
            f"pysu2 is unavailable: {error}; falling back to the executable route",
            RuntimeWarning,
            stacklevel=2,
        )
        return "executable"
    return "pysu2"
