"""The typed time history of a time-accurate SU2 point: one row per time step.

SU2 writes ``history.csv`` with ``Time_Iter`` (0-based) and, inside each time step, its inner
or outer iterations. The adapter keeps the LAST row of each time step (the converged state of
that step) and names it step ``Time_Iter + 1``, the 1-based step of
:mod:`cfdbender.post.time_average`.

TO VERIFY ON A REAL TIME-ACCURATE RUN: the column names of a time-accurate and of a multizone
history. They are in ONE table, :data:`LOAD_COLUMNS` and :func:`zone_column`; the names below
are the single-zone ones of a steady SU2 8.6.0 history and the ``[i]`` zone suffix of the
colleague's multizone ``HISTORY_OUTPUT`` groups, read from configuration files, not from a
time-accurate history.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from cfdbender.post.time_average import TimeHistory
from cfdbender.solvers.su2._history import history_headers

#: The history column of each force and moment coefficient (TO VERIFY on a real run).
LOAD_COLUMNS = ("CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
#: The time-step counter of a time-accurate history (TO VERIFY on a real run).
TIME_COUNTER = "Time_Iter"


def zone_column(name: str, zone: int | None) -> str:
    """Return the history column of ``name`` for one zone (TO VERIFY): ``CFx[1]``."""
    return name if zone is None else f"{name}[{zone}]"


def time_history(
    folder: Path, *, zone: int | None = None, columns: tuple[str, ...] = LOAD_COLUMNS
) -> TimeHistory:
    """Read ``history.csv`` of a time-accurate point into a :class:`TimeHistory`.

    Parameters
    ----------
    folder : Path
        The point folder.
    zone : int or None, optional
        The zone whose loads are read on a multizone run; None for one zone.
    columns : tuple of str, optional
        The neutral column names to read; the history keeps them under these names.

    Raises
    ------
    ValueError
        No time-step counter, a column missing, or a value that is not finite.
    """
    path = folder / "history.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        headers = history_headers(next(reader), path)
        if TIME_COUNTER not in headers:
            raise ValueError(f"{path}: no {TIME_COUNTER} column, so it is not a time history")
        wanted = {name: zone_column(name, zone) for name in columns}
        missing = [column for column in wanted.values() if column not in headers]
        if missing:
            raise ValueError(f"{path}: no column {', '.join(missing)}")
        last: dict[int, list[float]] = {}
        for cells in reader:
            row = dict(zip(headers, cells, strict=False))
            step = int(float(row[TIME_COUNTER])) + 1
            values = [float(row[column]) for column in wanted.values()]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: step {step} holds a value that is not finite")
            last[step] = values
    steps = sorted(last)
    data = np.array([last[step] for step in steps], dtype=float).reshape(len(steps), len(columns))
    return TimeHistory(
        steps=np.array(steps, dtype=int),
        columns={name: data[:, index] for index, name in enumerate(columns)},
    )
