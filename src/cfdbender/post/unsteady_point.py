"""The reported result of a time-accurate point: window means, per-revolution means and drift.

The result of a time-accurate point is the arithmetic mean of each history column over the
averaging window (:func:`cfdbender.post.time_average.window_average`, the one average). A
rotor's row is formed from the MEAN force and moment, never as the mean of per-step
coefficients and never from the last step. A history that does not hold every step of the
window is not averaged: the point is left out with the reason. On a rotor clock of a whole
number of steps per revolution the history is also cut into complete revolutions
(:func:`cfdbender.post.revolutions.per_revolution_table`) and the last revolution's drift is
judged (:func:`cfdbender.post.revolutions.revolution_drift_warnings`); a drift warning never
blocks anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from cfdbender.core.settings import WorkspaceError
from cfdbender.post.revolutions import (
    DEFAULT_DRIFT_LIMIT_PCT,
    load_kind,
    per_revolution_table,
    revolution_drift_pct,
    revolution_drift_warnings,
    whole_revolution_steps,
)
from cfdbender.post.time_average import TimeHistory, window_average, window_is_covered


def reduce_point(
    history: TimeHistory,
    columns: Sequence[str],
    *,
    window: tuple[int, int],
    steps_per_revolution: float | None,
    kind_of: Callable[[str], tuple[str, str] | None] = load_kind,
    drift_limit_pct: float = DEFAULT_DRIFT_LIMIT_PCT,
) -> dict[str, object]:
    """Return the window means, the per-revolution table and the drift warnings of a point.

    Returns
    -------
    dict
        ``FIRST_STEP``, ``LAST_STEP``, ``STEPS`` and ``means``; ``per_revolution`` (one row
        per complete revolution: ``REVOLUTION``, ``FIRST_STEP``, ``LAST_STEP``, each column
        and ``<column>_DRIFT_PCT``) and ``warnings``. A history that does not cover the window
        gives ``skipped`` with the reason instead of the means.
    """
    first, last = window
    out: dict[str, object] = {"FIRST_STEP": first, "LAST_STEP": last, "STEPS": last - first + 1}
    reason = window_is_covered(history, window)
    if reason is not None:
        out["skipped"] = reason
        return out
    average = window_average(history, window=window)
    out["means"] = {name: float(average.means[name]) for name in columns}
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    if steps_per_revolution:
        try:
            period = whole_revolution_steps(steps_per_revolution)
        except WorkspaceError as error:
            warnings.append(f"per_revolution is skipped: {error}")
        else:
            spans, means, _ = per_revolution_table(history, columns, revolution_steps=period)
            for index, ((start, end), mean) in enumerate(zip(spans, means, strict=True)):
                row: dict[str, object] = {"REVOLUTION": index + 1, "FIRST_STEP": start}
                row["LAST_STEP"] = end
                for name in columns:
                    row[name] = mean[name]
                    drift = (
                        revolution_drift_pct(mean[name], means[index - 1][name]) if index else None
                    )
                    row[f"{name}_DRIFT_PCT"] = "NA" if drift is None else drift
                rows.append(row)
            if len(means) >= 2:
                warnings.extend(
                    revolution_drift_warnings(
                        means[-2],
                        means[-1],
                        {name: name for name in columns},
                        limit_pct=drift_limit_pct,
                        kind_of=kind_of,
                    )
                )
    out["per_revolution"] = rows
    out["warnings"] = warnings
    return out
