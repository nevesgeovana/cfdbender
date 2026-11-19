"""The typed time history of a time-accurate SU2 point: one row per time step.

SU2 writes ``history.csv`` for a cold start and ``history_<RESTART_ITER>.csv`` (five digits)
for a restarted run, with ``Time_Iter`` (0-based) and, inside each time step, its inner or outer
iterations; a point with an export window has one history per segment, joined in step order
(:func:`~cfdbender.solvers.su2._history.history_table`). The adapter keeps the LAST row of each
time step (the converged state of that step) and names it with the row's 1-based step,
``Time_Iter + 1 - offset``, where the offset is the ``RESTART_ITER`` of ``su2.cfg`` of a point
started from a steady solution (0 for a cold start), so the history always carries steps 1 to
the row's last.

The column names are in ONE table, :data:`LOAD_COLUMNS` and :func:`zone_column`, measured on a
real SU2 8.6.0 time-accurate history: the loads are ``CFx`` ... ``CMz`` and the time counter is
``Time_Iter``; with ``AERO_COEFF_SURF`` the loads of one marker are ``CFx(blade)``. A multizone
history appends the zone index in square brackets, ``CFx[1]`` and ``CFx[1](blade)``, as SU2's
own multizone output forms them (``CMultizoneOutput``: the field name, then ``[zone]``, then the
marker); no multizone time-accurate history has been run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cfdbender.post.time_average import TimeHistory
from cfdbender.solvers.su2._history import history_names, history_table, step_offset

#: The history column of each force and moment coefficient (SU2 8.6.0).
LOAD_COLUMNS = ("CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
#: The time-step counter of a time-accurate history (SU2 8.6.0).
TIME_COUNTER = "Time_Iter"


def zone_column(name: str, zone: int | None, marker: str | None = None) -> str:
    """Return the history column of ``name``: ``CFx``, ``CFx(blade)``, ``CFx[1](blade)``.

    Parameters
    ----------
    name : str
        The coefficient, SU2's spelling.
    zone : int or None
        The zone of a multizone run; None for one zone.
    marker : str or None, optional
        The marker of a per-marker column (``AERO_COEFF_SURF``); None for the total.
    """
    zoned = name if zone is None else f"{name}[{zone}]"
    return zoned if marker is None else f"{zoned}({marker})"


def time_history(
    folder: Path,
    *,
    zone: int | None = None,
    columns: tuple[str, ...] = LOAD_COLUMNS,
    marker: str | None = None,
) -> TimeHistory:
    """Read the history of a time-accurate point into a :class:`TimeHistory`.

    Parameters
    ----------
    folder : Path
        The point folder.
    zone : int or None, optional
        The zone whose loads are read on a multizone run; None for one zone.
    columns : tuple of str, optional
        The neutral column names to read; the history keeps them under these names.
    marker : str or None, optional
        The marker whose own loads are read; None for the total.

    Raises
    ------
    ValueError
        A missing history (naming the files looked for), a gap or an overlap between
        segments, empty or malformed rows, missing or duplicated columns, invalid time-step
        counters, or values that are not finite.
    """
    headers, rows = history_table(folder, TIME_COUNTER)
    path = folder / " + ".join(history_names(folder))
    offset = step_offset(folder)
    if TIME_COUNTER not in headers:
        raise ValueError(f"{path}: no {TIME_COUNTER} column, so it is not a time history")
    wanted = {name: zone_column(name, zone, marker) for name in columns}
    missing = [column for column in wanted.values() if column not in headers]
    if missing:
        raise ValueError(f"{path}: no column {', '.join(missing)}")
    last: dict[int, list[float]] = {}
    previous = 0
    for line, cells in enumerate(rows, 2):
        if len(cells) != len(headers):
            raise ValueError(f"{path}: line {line} has {len(cells)} cells, expected {len(headers)}")
        row = dict(zip(headers, cells, strict=True))
        try:
            counter = float(row[TIME_COUNTER])
            values = [float(row[column]) for column in wanted.values()]
        except ValueError as error:
            raise ValueError(f"{path}: line {line}: {error}") from error
        if (
            not math.isfinite(counter)
            or not counter.is_integer()
            or counter < offset
            or counter >= 2**63 - 1
        ):
            raise ValueError(
                f"{path}: line {line} needs a finite whole {TIME_COUNTER} of at least {offset}"
            )
        step = int(counter) + 1 - offset
        if step < previous:
            raise ValueError(f"{path}: line {line}: time steps must not decrease")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: step {step} holds a value that is not finite")
        last[step] = values
        previous = step
    if not last:
        raise ValueError(f"{path}: no history rows")
    steps = list(last)
    data = np.array([last[step] for step in steps], dtype=float).reshape(len(steps), len(columns))
    return TimeHistory(
        steps=np.array(steps, dtype=int),
        columns={name: data[:, index] for index, name in enumerate(columns)},
    )
