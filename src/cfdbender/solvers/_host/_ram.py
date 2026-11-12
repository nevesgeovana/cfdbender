"""Measure host memory and enforce the shared 0.5 GiB free RAM floor."""

from __future__ import annotations

import ctypes
import math
import sys
from collections.abc import Callable
from pathlib import Path


def free_ram_gib() -> float:
    """Return available host RAM in GiB using the native operating system."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page",
                    "free_page",
                    "virtual",
                    "free_virtual",
                    "ext",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.available) / 2**30
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise OSError("MemAvailable missing")


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
        or free - estimate_gib < 0.5
    ):
        raise ValueError(
            f"free RAM {free:.3f} GiB minus estimate {estimate_gib:.3f} GiB "
            f"would breach the 0.5 GiB floor; refusing {solver}"
        )
