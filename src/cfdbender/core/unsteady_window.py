"""The averaging window of an unsteady point and the time average over it.

The window is a MATRIX input, stated once on the row and shared by every
time-domain product of the point. Exactly one of two keys states it, and there
is no default:

* ``LAST_REVS_AVG``: a count of revolutions of the clock rotor (a float);
* ``LAST_ITERS_AVG``: a count of solver steps, the only key of a row that turns
  no rotor.

The window ENDS at the last step of the run, inclusive, steps counted from one.
A count of revolutions has no length without a clock, so it resolves to None
rather than to a guess; a fractional length is rounded to the nearest whole step,
halves to the even one. A window longer than the run is the whole run.

The mean over the window is in :mod:`cfdbender.post.time_average`. Nothing here reads a
file or a solver and nothing imports numpy; the semantics, names and sentences match the
time-domain post-processing of pyflightstream.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from cfdbender.core.settings import WorkspaceError

__all__ = [
    "LAST_ITERS_AVG",
    "LAST_REVS_AVG",
    "averaging_span",
    "averaging_steps",
    "stated_key",
]

#: The two keys, spelled once. Upper case and exact, like every other row key.
LAST_REVS_AVG = "LAST_REVS_AVG"
LAST_ITERS_AVG = "LAST_ITERS_AVG"

#: The token of a cell that states nothing.
NOT_APPLICABLE = "NA"


def _positive(variables: Mapping[str, object], key: str) -> float | None:
    stated = variables.get(key)
    if stated is None or str(stated).strip() in ("", "-", NOT_APPLICABLE):
        return None
    try:
        value = float(str(stated).strip()) if not isinstance(stated, bool) else math.nan
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value <= 0.0:
        raise WorkspaceError(
            f"averaging window {key} = {stated!r} must be a finite positive number"
        )
    return value


def stated_key(variables: Mapping[str, object]) -> tuple[str, float] | None:
    """Return ``(key, value)`` of the ONE averaging key the row states, or None.

    None where the row states neither key (including the unstated cell ``-``).
    An invalid value or both keys raise WorkspaceError: picking a fallback
    would hide a contradictory or invalid window.

    Parameters
    ----------
    variables : mapping
        The matrix row's variables, keyed by their upper-case names.

    Returns
    -------
    tuple of (str, float) or None
        The key (``LAST_REVS_AVG`` or ``LAST_ITERS_AVG``) and its positive value,
        or None.

    Raises
    ------
    WorkspaceError
        If a value is not a finite positive number, or both keys are stated.

    Examples
    --------
    >>> stated_key({"LAST_REVS_AVG": "2.5"})
    ('LAST_REVS_AVG', 2.5)
    >>> stated_key({"LAST_ITERS_AVG": "-"}) is None
    True
    """
    revs = _positive(variables, LAST_REVS_AVG)
    iters = _positive(variables, LAST_ITERS_AVG)
    if revs is not None and iters is not None:
        raise WorkspaceError(
            f"averaging window states both {LAST_REVS_AVG} = {revs:g} and "
            f"{LAST_ITERS_AVG} = {iters:g}, two ways to say one window; "
            "state only one averaging key"
        )
    if revs is None and iters is None:
        return None
    return (LAST_REVS_AVG, revs) if revs is not None else (LAST_ITERS_AVG, float(iters or 0.0))


def averaging_steps(variables: Mapping[str, object], *, per_revolution: float | None) -> int | None:
    """Return the window's LENGTH in solver steps on a clock of ``per_revolution``.

    A count of iterations is already in steps and is the same for every rotor. A
    count of revolutions has no length without a clock, and resolves to None
    rather than to a guess.

    Parameters
    ----------
    variables : mapping
        The matrix row's variables, keyed by their upper-case names.
    per_revolution : float or None
        Solver steps in one revolution of the rotor whose clock counts the
        window. None where the row has no clock.

    Returns
    -------
    int or None
        The window's length in steps, at least 1, or None where the row states
        no key or a count of revolutions meets no clock. A fractional number of
        steps is rounded to the nearest whole step, halves to the even one (2.5
        steps is 2, 3.5 is 4). Invalid values and windows shorter than one step
        raise WorkspaceError.

    Raises
    ------
    WorkspaceError
        If the stated averaging window is nonnumeric, nonfinite or shorter than one step. The
        run length is unknown here, so a window longer than the run is clipped to it later, by
        :func:`averaging_span`, and never refused.
    """
    stated = stated_key(variables)
    if stated is None:
        return None
    key, value = stated
    if key == LAST_REVS_AVG and (
        per_revolution is None or not math.isfinite(per_revolution) or per_revolution <= 0
    ):
        return None
    length = value if key == LAST_ITERS_AVG else value * float(per_revolution or 0)
    if not math.isfinite(length) or length < 1:
        raise WorkspaceError(
            f"averaging window {key} = {value:g} spans {length:g} solver step(s), "
            "which averages nothing or has no finite length. State a finite window "
            "of at least one step."
        )
    return int(round(length))


def averaging_span(
    variables: Mapping[str, object], *, last_step: int, per_revolution: float | None
) -> tuple[int, int] | None:
    """Return the inclusive, 1-based window ``(first, last)`` ending at ``last_step``.

    Longer than the run is the whole run, never a step before the first.

    Parameters
    ----------
    variables : mapping
        The matrix row's variables, keyed by their upper-case names.
    last_step : int
        The last solver step of the run, 1-based. The window ends here.
    per_revolution : float or None
        Solver steps in one revolution of the clock rotor; None where there is
        none.

    Returns
    -------
    tuple of (int, int) or None
        The inclusive 1-based ``(first, last)`` steps of the window, or None
        where the length is unknown or ``last_step`` is not positive.

    Examples
    --------
    >>> averaging_span({"LAST_REVS_AVG": "0.5"}, last_step=1000, per_revolution=250.0)
    (876, 1000)
    >>> averaging_span({"LAST_ITERS_AVG": "100"}, last_step=1000, per_revolution=None)
    (901, 1000)
    """
    steps = averaging_steps(variables, per_revolution=per_revolution)
    if steps is None or last_step <= 0:
        return None
    return (max(int(last_step) - steps + 1, 1), int(last_step))
