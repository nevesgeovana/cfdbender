"""Per-revolution means of a history, their drift, and the warning that judges it.

A rotor history is cut into COMPLETE revolutions on the rotor's own clock, and each column is
averaged over each of them by :func:`~cfdbender.post.time_average.window_average`, the one
average. From the second revolution the drift of a mean from the previous one is stated in per
cent (:func:`revolution_drift_pct`). The drift of the LAST revolution is judged by
:func:`revolution_drift_warnings` against the SCALE of its load group, with a limit of
:data:`DEFAULT_DRIFT_LIMIT_PCT` per cent by default. The warning is advice: nothing here
blocks anything.

Everything is pure; the semantics, names and sentences match the time-domain post-processing
of pyflightstream.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from cfdbender.core.settings import WorkspaceError
from cfdbender.post.time_average import TimeHistory, window_average

__all__ = [
    "DEFAULT_DRIFT_LIMIT_PCT",
    "FORCE_PLOT_PARAMETERS",
    "is_force_or_moment_column",
    "last_complete_revolution",
    "load_kind",
    "per_revolution_table",
    "revolution_drift_pct",
    "revolution_drift_warnings",
    "whole_revolution_steps",
]

#: The drift limit when none is declared, in per cent of the scale of the load.
DEFAULT_DRIFT_LIMIT_PCT = 1.0

#: The force plot parameters by the short name a column is named for, each paired with the
#: parameter it measures and the units it is sampled in. A load column is named
#: ``<short>_<group>``.
FORCE_PLOT_PARAMETERS: dict[str, tuple[str, str]] = {
    "CL": ("CL", "COEFFICIENTS"),
    "CDI": ("CDI", "COEFFICIENTS"),
    "CDO": ("CDO", "COEFFICIENTS"),
    "CDP": ("CDP", "COEFFICIENTS"),
    "CDV": ("CDV", "COEFFICIENTS"),
    "CD": ("CD", "COEFFICIENTS"),
    "FX": ("FORCE_X", "NEWTONS"),
    "FY": ("FORCE_Y", "NEWTONS"),
    "FZ": ("FORCE_Z", "NEWTONS"),
    "MX": ("MOMENT_X", "NEWTONS"),
    "MY": ("MOMENT_Y", "NEWTONS"),
    "MZ": ("MOMENT_Z", "NEWTONS"),
}


def is_force_or_moment_column(name: str) -> bool:
    """Whether a column is a force or a moment, or a force coefficient of one.

    A column is named ``<parameter>_<group>`` (``FX_MRP_TOTAL``) and the parameter is one of
    :data:`FORCE_PLOT_PARAMETERS`. The parameter is what is asked, so a group that happens to
    begin with ``F`` is not mistaken for a force.

    Examples
    --------
    >>> is_force_or_moment_column("FX_MRP_TOTAL"), is_force_or_moment_column("CDI_WING")
    (True, True)
    >>> is_force_or_moment_column("VX_probe1"), is_force_or_moment_column("FXX_TOTAL")
    (False, False)
    """
    return name.split("_", 1)[0] in FORCE_PLOT_PARAMETERS


def load_kind(name: str) -> tuple[str, str] | None:
    """Return the ``(kind, group)`` of a load column, or None for a column that is no load.

    The KIND is what the parameter measures (``force`` for ``FX FY FZ``, ``moment`` for
    ``MX MY MZ``, ``coefficient`` for ``CL CDI CDO CDP CDV CD``) and the GROUP is the text
    after the parameter. The columns of one kind and one group are the components of one load
    of one body in one frame.

    Examples
    --------
    >>> load_kind("FY_PROP_X"), load_kind("CL_WING"), load_kind("MACH_P1")
    (('force', 'PROP_X'), ('coefficient', 'WING'), None)
    """
    parameter, _, group = name.partition("_")
    if parameter not in FORCE_PLOT_PARAMETERS:
        return None
    measured, units = FORCE_PLOT_PARAMETERS[parameter]
    return ("coefficient" if units == "COEFFICIENTS" else measured.split("_")[0].lower(), group)


def whole_revolution_steps(steps_per_revolution: float) -> int:
    """Return an exact whole-step period, refusing non-finite or fractional clocks.

    Raises
    ------
    WorkspaceError
        If the clock is not finite or has a fractional remainder beyond two ULP.

    Examples
    --------
    >>> whole_revolution_steps(36.0)
    36
    """
    if not math.isfinite(float(steps_per_revolution)):
        raise WorkspaceError(
            f"steps per revolution must be a finite number, got {steps_per_revolution!r}"
        )
    nearest = round(steps_per_revolution)
    if abs(steps_per_revolution - nearest) > 2 * math.ulp(steps_per_revolution):
        raise WorkspaceError(
            f"{steps_per_revolution:g} steps per revolution has fractional remainder "
            f"{steps_per_revolution % 1:g}; a complete revolution cannot be represented "
            "by whole solver steps without rounding. Choose the time step so one revolution "
            "is a whole number of steps (delta_time = 60 / (rpm * N) for N steps per revolution)"
        )
    return nearest


def last_complete_revolution(
    first_step: int, last_step: int, steps_per_revolution: float
) -> tuple[int, int] | None:
    """Return the first and last step of the last complete revolution, or None.

    Revolution k is steps ``first + (k - 1) N`` to ``first + k N - 1``, counted from the
    history's first step, with ``N`` an integral number of steps per revolution. Fractional
    clocks are refused rather than rounded to another period.

    Raises
    ------
    WorkspaceError
        If ``steps_per_revolution`` is not finite or has a fractional remainder.

    Examples
    --------
    >>> last_complete_revolution(3, 13, 4.0)
    (7, 10)
    >>> last_complete_revolution(3, 5, 4.0) is None
    True
    """
    steps = whole_revolution_steps(steps_per_revolution)
    if steps < 1 or last_step < first_step:
        return None
    complete = (last_step - first_step + 1) // steps
    if complete < 1:
        return None
    start = first_step + (complete - 1) * steps
    return start, start + steps - 1


def per_revolution_table(
    history: TimeHistory,
    columns: Sequence[str],
    *,
    revolution_steps: int,
    read_steps: set[int] | None = None,
) -> tuple[list[tuple[int, int]], list[dict[str, float]], int]:
    """Cut a history into COMPLETE revolutions and take each column's mean.

    Revolution ``k`` is rows ``(k-1) * revolution_steps`` to ``k * revolution_steps - 1`` of
    the history, counted from its first row, so a history that begins at step one cuts at
    steps 1 to N, N+1 to 2N and so on. The mean is
    :func:`~cfdbender.post.time_average.window_average`, so this table cannot disagree with
    the time average about what a mean is. A trailing partial revolution is NOT averaged: its
    length differs, so its mean would be the mean of another thing under the same name.

    Parameters
    ----------
    history : TimeHistory
        The history.
    columns : sequence of str
        The columns to average.
    revolution_steps : int
        Solver steps in one revolution; at least one.
    read_steps : set of int, optional
        Receives the steps the averages read.

    Returns
    -------
    tuple
        The (first, last) step of each complete revolution, each revolution's mean by column,
        and the number of rows left over in the partial one.

    Raises
    ------
    WorkspaceError
        If ``revolution_steps`` is below one.
    """
    if revolution_steps < 1:
        raise WorkspaceError("a revolution of under one solver step cannot be cut")
    rows = len(history.steps)
    complete = rows // revolution_steps
    windows: list[tuple[int, int]] = []
    means: list[dict[str, float]] = []
    for index in range(complete):
        first = int(history.steps[index * revolution_steps])
        last = int(history.steps[(index + 1) * revolution_steps - 1])
        average = window_average(history, window=(first, last), read_steps=read_steps)
        windows.append((first, last))
        means.append({name: float(average.means[name]) for name in columns})
    return windows, means, rows - complete * revolution_steps


def revolution_drift_pct(current: float, previous: float) -> float | None:
    """Return the drift of one mean from the previous one, in per cent, or None (`NA`).

    ``(current - previous) / |previous| * 100``: the sign says whether the mean rose or fell
    whatever the sign of the quantity. None where the previous mean is zero, which has no
    relative change, and where either mean is not a number.

    Examples
    --------
    >>> revolution_drift_pct(101.0, 100.0)
    1.0
    >>> revolution_drift_pct(-99.0, -100.0)
    1.0
    >>> revolution_drift_pct(1.0, 0.0) is None
    True
    """
    if previous == 0.0 or not math.isfinite(previous) or not math.isfinite(current):
        return None
    difference = current - previous
    ratio = (
        difference / abs(previous)
        if math.isfinite(difference)
        else current / abs(previous) - math.copysign(1.0, previous)
    )
    return ratio * 100.0


def revolution_drift_warnings(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    shown: Mapping[str, str],
    *,
    limit_pct: float,
    kind_of: Callable[[str], tuple[str, str] | None] = load_kind,
) -> list[str]:
    """Judge the last revolution of each force and moment column against its SCALE.

    ``kind_of`` gives the ``(kind, group)`` of a column, or None for a column that is no load
    (:func:`load_kind` by default). The columns of one kind and one group are the components
    of ONE load of one body in one frame, so an in-plane force is judged against the thrust
    and an in-plane moment against the torque. The scale is the largest magnitude among the
    PREVIOUS revolution's means of that kind and group, the revolution the relative drift
    divides by; a column warns when ``|mean_now - mean_prev| > limit_pct / 100 * scale``. For
    the column that IS the largest this is the relative drift of the table, unchanged; a
    near-zero component is not judged against itself, whose relative change is noise.

    ``shown`` maps each column to the name the table prints. A column that is no load, whose
    group has no finite nonzero mean, or whose own means are not numbers is never warned
    about. The warning never blocks anything.

    Parameters
    ----------
    previous : mapping of str to float
        Each column's mean over the previous revolution.
    current : mapping of str to float
        Each column's mean over the last revolution.
    shown : mapping of str to str
        Each column's name as the table prints it.
    limit_pct : float
        The drift allowed, in per cent of the scale.
    kind_of : callable, optional
        The classifier of a column name into ``(kind, group)``.

    Returns
    -------
    list of str
        One clause per warned column, in column order, stating the change against the scale.

    Examples
    --------
    >>> before = {"FX_P": 100.0, "FY_P": 0.02, "MACH_P1": 0.1}
    >>> names = {name: name for name in before}
    >>> after = {"FX_P": 100.1, "FY_P": 0.005, "MACH_P1": 0.2}
    >>> revolution_drift_warnings(before, after, names, limit_pct=1.0)
    []
    >>> (clause,) = revolution_drift_warnings(before, {**after, "FY_P": 2.5}, names, limit_pct=1.0)
    >>> clause.split(" (")[0]
    'column FY_P drifts +2.4800 per cent of its scale 100'
    """
    kinds: dict[str, tuple[str, str]] = {}
    largest: dict[tuple[str, str], str] = {}
    for name, value in previous.items():
        key = kind_of(name)
        if key is None:
            continue
        kinds[name] = key
        best = largest.get(key)
        if math.isfinite(value) and (best is None or abs(value) > abs(previous[best])):
            largest[key] = name
    clauses: list[str] = []
    for name, key in kinds.items():
        by = largest.get(key)
        if by is None or previous[by] == 0.0:
            continue
        scale = abs(previous[by])
        change = current[name] - previous[name]
        pct = change / scale * 100.0
        if abs(pct) > limit_pct:  # NaN compares False: a mean that is no number is quiet
            clauses.append(
                f"column {shown[name]} drifts {pct:+.4f} per cent of its scale {scale:.6g} "
                f"(the magnitude of {shown[by]}, the largest {key[0]} mean of its group in "
                f"the earlier revolution; a change of {change:+.6g})"
            )
    return clauses
