"""The time average of a history over a window of solver steps.

The mean of a column over the window is the arithmetic mean of the rows whose step lies in it,
equal weights. A history that does not hold every step of the window is not averaged:
:func:`window_is_covered` gives the reason the point is left out. The window itself is resolved
by :mod:`cfdbender.core.unsteady_window`. Nothing here reads a file or a solver; the semantics,
names and sentences match the time-domain post-processing of pyflightstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cfdbender.core.settings import WorkspaceError

__all__ = [
    "TimeHistory",
    "WindowAverage",
    "refuse_uncovered_window",
    "step_clock",
    "window_average",
    "window_is_covered",
]


def step_clock(cells: Sequence[object] | np.ndarray) -> np.ndarray:
    """Return the steps a history's step cells state, the ordinal where they cannot.

    A cell that is not a finite whole number, or a clock that is not strictly
    increasing, gives the WHOLE table its ordinal (row k is step k), never a mix
    of stated and ordinal steps.

    Parameters
    ----------
    cells : sequence
        The column's cells, as text or numbers, in table order.

    Returns
    -------
    numpy.ndarray
        One whole step per cell.

    Examples
    --------
    >>> step_clock(["3", "4", "5"]).tolist()
    [3, 4, 5]
    >>> step_clock(["3", "3", "5"]).tolist()
    [1, 2, 3]
    """
    n_rows = len(cells)
    ordinal = np.arange(1, n_rows + 1, dtype=int)
    try:
        read = [float(cell) for cell in cells]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ordinal
    stated = np.asarray(read, dtype=float)
    if not n_rows or not bool(np.all(np.isfinite(stated))):
        return ordinal
    whole = np.rint(stated)
    if not bool(np.all(np.abs(stated - whole) < 1e-9)):
        # A fractional clock is a time and not a step; the windows are whole steps.
        return ordinal
    steps = whole.astype(int)
    if n_rows > 1 and not bool(np.all(np.diff(steps) > 0)):
        # Duplicated or out of order: selecting a window by value would pick rows
        # the window did not name.
        return ordinal
    return steps


@dataclass(frozen=True)
class TimeHistory:
    """A scalar history: one value per column at every solver step.

    Attributes
    ----------
    steps : numpy.ndarray
        Solver step of each row, 1-based whole numbers, shape ``(n_steps,)``.
    columns : dict of str to numpy.ndarray
        One entry per column, shape ``(n_steps,)``.
    time_s : numpy.ndarray or None
        Solution time of each row in seconds, when the history carries one. It is not a
        column and is never averaged.
    """

    steps: np.ndarray
    columns: dict[str, np.ndarray]
    time_s: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        """Refuse a column that does not hold one value per step."""
        for name, values in self.columns.items():
            if np.shape(values) != np.shape(self.steps):
                raise WorkspaceError(
                    f"column {name!r} holds {np.size(values)} value(s) for "
                    f"{np.size(self.steps)} step(s); a history has one value per step"
                )

    @property
    def n_frames(self) -> int:
        """The number of rows of the history."""
        return int(len(self.steps))


@dataclass(frozen=True)
class WindowAverage:
    """The mean of every column of a :class:`TimeHistory` over one window.

    Attributes
    ----------
    window : tuple of int
        Inclusive ``(first_step, last_step)``, as declared by the caller.
    n_frames : int
        How many rows fell inside the window; never zero.
    means : dict of str to float
        Each column's arithmetic mean over those rows.
    source_window : tuple of int
        First and last step actually averaged.
    """

    window: tuple[int, int]
    n_frames: int
    means: dict[str, float]
    source_window: tuple[int, int]


def window_average(
    history: TimeHistory, *, window: tuple[int, int], read_steps: set[int] | None = None
) -> WindowAverage:
    """Average a history over one declared window of solver steps.

    The only implementation of this average: a per-revolution table is this
    function applied once per revolution, so two tables cannot disagree about what a
    mean is. It is pure.

    Parameters
    ----------
    history : TimeHistory
        The history.
    window : tuple of int, keyword-only
        Inclusive ``(first_step, last_step)`` in solver steps.
    read_steps : set of int or None
        When supplied, collects the steps used by this average.

    Returns
    -------
    WindowAverage
        Per-column means over the rows inside the window.

    Raises
    ------
    WorkspaceError
        If the history holds no row, the window is given backwards, or the window holds
        no row. An average of nothing is still a number, which is why it is refused.
    """
    if history.n_frames == 0:
        raise WorkspaceError("the series holds no frame")
    first, last = int(window[0]), int(window[1])
    if last < first:
        raise WorkspaceError(
            f"the window ({first}, {last}) ends before it starts; it is an inclusive "
            "(first_step, last_step) pair in solver steps"
        )
    steps = history.steps
    inside = (steps >= first) & (steps <= last)
    if read_steps is not None:
        read_steps.update(int(step) for step in steps[inside])
    count = int(inside.sum())
    if count == 0:
        raise WorkspaceError(
            f"the window ({first}, {last}) holds no frame of a series running from "
            f"{int(steps[0])} to {int(steps[-1])} in steps of "
            f"{int(steps[1] - steps[0]) if len(steps) > 1 else 1}. "
            "An average over no frame is still a number, which is why this is refused "
            "rather than returned"
        )
    return WindowAverage(
        window=(first, last),
        n_frames=count,
        # Reduced as a one-column table, the shape the arithmetic was written for.
        means={
            name: float(values[inside][:, None].mean(axis=0)[0])
            for name, values in history.columns.items()
        },
        source_window=(int(steps[inside].min()), int(steps[inside].max())),
    )


def window_is_covered(history: TimeHistory, window: tuple[int, int]) -> str | None:
    """Return why a history does not hold every step of ``window``, or None when it does.

    A partial cover is not a cover: a point that stopped part-way through would
    average the part and sit beside a full one with the same label. The caller
    leaves the point out and reports this reason.

    Parameters
    ----------
    history : TimeHistory
        The history.
    window : tuple of int
        Inclusive ``(first_step, last_step)``.

    Returns
    -------
    str or None
        The reason, or None where every step of the window is present.
    """
    steps = np.asarray(history.steps, dtype=int)
    if np.array_equal(
        steps[(steps >= window[0]) & (steps <= window[1])],
        np.arange(window[0], window[1] + 1),
    ):
        return None
    held = f"steps {int(steps[0])} to {int(steps[-1])}" if len(steps) else "no step"
    return (
        f"the row states steps {window[0]} to {window[1]} and the history holds {held}, "
        "with a missing sample or gap in that window"
    )


def refuse_uncovered_window(
    history: TimeHistory, window: tuple[int, int], *, reduction: str, index: int = 1
) -> None:
    """Refuse a window that reaches outside the steps the history holds.

    Parameters
    ----------
    history : TimeHistory
        The history.
    window : tuple of int
        Inclusive ``(first_step, last_step)``.
    reduction : str
        The name of the reduction, for the sentence.
    index : int, optional
        The window's number within the reduction, from one.

    Raises
    ------
    WorkspaceError
        If the window starts before the first step or ends after the last one of the history.
    """
    first_step = int(history.steps[0]) if len(history.steps) else 1
    last_step = int(history.steps[-1]) if len(history.steps) else 0
    first, last = int(window[0]), int(window[1])
    if last > last_step or first < first_step:
        raise WorkspaceError(
            f"the history runs from step {first_step} to step {last_step} and the "
            f"{reduction} window {index} spans steps {first} to {last}, so the history "
            "does not cover the window the row states; a shorter history averaged as a "
            "whole one would be an average of a run that did not finish writing"
        )
