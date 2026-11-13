"""Enforce the shared 0.5 GiB free RAM floor with an estimate; core.run measures it."""

from __future__ import annotations

import math
from collections.abc import Callable

from cfdbender.core.run import FLOOR_GIB as FLOOR_GIB
from cfdbender.core.run import free_ram_gib as free_ram_gib


def require_ram(
    estimate_gib: float, *, solver: str = "SU2", measure: Callable[[], float] | None = None
) -> None:
    """Refuse a launch or poll that would breach the host RAM floor.

    Parameters
    ----------
    estimate_gib : float
        Additional memory needed before launch; zero when polling.
    solver : str
        Solver name for diagnostics. The default preserves legacy SU2 callers.
    measure : callable, optional
        Injectable available-memory measurement in GiB.
    """
    try:
        free = (measure or free_ram_gib)()
    except (OSError, ValueError) as error:
        raise ValueError(f"RAM measurement unavailable; refusing {solver}") from error
    if (
        not math.isfinite(free)
        or not math.isfinite(estimate_gib)
        or free < 0
        or estimate_gib < 0
        or free - estimate_gib < FLOOR_GIB
    ):
        raise ValueError(
            f"free RAM {free:.3f} GiB minus estimate {estimate_gib:.3f} GiB "
            f"would breach the 0.5 GiB floor; refusing {solver}"
        )
