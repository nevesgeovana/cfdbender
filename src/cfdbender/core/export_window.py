"""The export window of a time-accurate row: the steps whose surface and volume files are written.

A row states at most one of pyflightstream's four keys:

* ``EXPORT_UNSTEADY_AFTER_ITER: n``: the exports begin at step ``n`` (1-based);
* ``EXPORT_UNSTEADY_AFTER_REV: r``: they begin at step ``ceil(r S - 1e-9)``;
* ``EXPORT_UNSTEADY_LAST_ITER: n``: they cover the last ``n`` steps, from ``N - n + 1``;
* ``EXPORT_UNSTEADY_LAST_REV: r``: they cover the last ``ceil(r S - 1e-9)`` steps;

with ``S`` the steps per revolution of the rotor clock and ``N`` the run's ``TIME_ITERATIONS``.
Every window ends at step ``N``. A revolutions form on a row with no rotor clock is warned and
skipped; two keys, a value that is not positive, a window of no step or longer than the run, a
threshold beyond the run, and a window that begins after the averaging window are refused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from cfdbender.core.refusals import warn_or_refuse
from cfdbender.core.unsteady_clock import revolution_steps

AFTER_ITER = "EXPORT_UNSTEADY_AFTER_ITER"
AFTER_REV = "EXPORT_UNSTEADY_AFTER_REV"
LAST_ITER = "EXPORT_UNSTEADY_LAST_ITER"
LAST_REV = "EXPORT_UNSTEADY_LAST_REV"
EXPORT_KEYS = (LAST_REV, LAST_ITER, AFTER_REV, AFTER_ITER)


@dataclass(frozen=True)
class ExportWindow:
    """The resolved export window: its stated form and value and its steps.

    Attributes
    ----------
    key : str
        The stated key.
    value : float
        The stated value.
    first_step, last_step : int
        The first and last exported steps, 1-based and inclusive.
    """

    key: str
    value: float
    first_step: int
    last_step: int

    def record(self) -> dict[str, object]:
        """Return the form, the written value and the resolved steps."""
        return {
            "form": self.key,
            "value": self.value,
            "first_step": self.first_step,
            "last_step": self.last_step,
        }


def _who(pol: str) -> str:
    return f"POL {pol}" if pol else "the row"


def _value(pol: str, key: str, raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise ValueError(f"{_who(pol)} states {key} as {raw!r}, which is not a finite number")
    value = float(raw)
    if value <= 0.0:
        what = (
            "how much of the end of the run the per-step exports cover"
            if key in (LAST_ITER, LAST_REV)
            else "the step the exports begin on"
        )
        raise ValueError(
            f"{_who(pol)} states {key} as {raw!r}, and it is {what}, so it is a positive number"
        )
    if key in (AFTER_ITER, LAST_ITER) and value != int(value):
        raise ValueError(
            f"{_who(pol)} states {key} as {raw!r}, which is not a whole number of steps"
        )
    return value


def export_window(
    variables: Mapping[str, object],
    *,
    pol: str,
    time_iterations: int,
    steps_per_revolution: float | None,
    averaging_first_step: int | None = None,
) -> ExportWindow | None:
    """Resolve the row's export window, or None when it states none.

    Parameters
    ----------
    variables : mapping
        The row's variables.
    pol : str
        The row's POL, named by every message.
    time_iterations : int
        The run's last step.
    steps_per_revolution : float or None
        The rotor clock; None on a row that turns nothing.
    averaging_first_step : int or None, optional
        The first step of the averaging window; an export window that begins later is
        refused.

    Raises
    ------
    ValueError
        Two keys, a value that is not positive (or not whole for an iterations form), a
        window of no step or longer than the run, a threshold beyond the run, or a window
        that begins after the averaging window.
    """
    stated = [key for key in EXPORT_KEYS if key in variables]
    if not stated:
        return None
    if len(stated) > 1:
        raise ValueError(
            f"{_who(pol)} states {' and '.join(stated)} both. The per-step exports begin at "
            "ONE step; keep one"
        )
    (key,) = stated
    value = _value(pol, key, variables[key])
    if key in (LAST_REV, AFTER_REV) and not steps_per_revolution:
        warn_or_refuse(
            f"{_who(pol)} states {key} and the row has no rotor clock: a revolution is counted on "
            f"a rotor the run turns. State '{key.replace('_REV', '_ITER')}: <steps>' instead; "
            "the key is skipped and every step's files follow the setup",
            UserWarning,
            stacklevel=2,
        )
        return None
    if key in (LAST_REV, AFTER_REV):
        steps = revolution_steps(value, float(steps_per_revolution or 0.0))
    else:
        steps = int(value)
    if key in (LAST_ITER, LAST_REV):
        if steps < 1:
            raise ValueError(
                f"{_who(pol)} states {key} as {value:g}, which rounds to no time step of this "
                f"{time_iterations}-step run, so nothing would be exported"
            )
        if steps > time_iterations:
            raise ValueError(
                f"{_who(pol)} states {key} as {value:g}, which is more than the "
                f"{time_iterations} steps the run is long. Lower it to at most "
                f"{time_iterations} steps, which exports every step, or lengthen the run"
            )
        first = time_iterations - steps + 1
    else:
        first = max(steps, 1)
        if first > time_iterations:
            raise ValueError(
                f"{_who(pol)} states {key} as {value:g}, which is step {first}, and the run is "
                f"{time_iterations} steps long, so nothing would be exported. Lower it, or "
                "lengthen the run"
            )
    if averaging_first_step is not None and first > averaging_first_step:
        raise ValueError(
            f"{_who(pol)} states {key} as {value:g}, which is step {first}, and the averaging "
            f"window begins at step {averaging_first_step}: the steps of the window before the "
            "exports would never be exported. Begin the exports at that step or earlier"
        )
    return ExportWindow(key, value, first, time_iterations)
