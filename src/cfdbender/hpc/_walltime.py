"""Numeric grouped walltime rules, independent of any solver or executor."""

from __future__ import annotations

import math
from dataclasses import dataclass

BEST_FACTOR = 1.25


@dataclass(frozen=True)
class BudgetRow:
    """One row and its points, with budgets supplied by the planning caller."""

    sim_id: str
    run_ids: tuple[str, ...]
    point_seconds: tuple[float | None, ...]
    walltime_cell_s: float | None
    walltime_cell_text: str | None
    best: bool
    margin_s: float
    timeout_s: float | None = None


@dataclass(frozen=True)
class BudgetGroup:
    """The rows of one job, in their input order."""

    units: tuple[BudgetRow, ...]


def _unestimated(split: BudgetGroup) -> list[str]:
    """Return the points with neither a recorded estimate nor a row cell to fall back on."""
    return [
        run_id
        for unit in split.units
        for run_id, seconds in zip(unit.run_ids, unit.point_seconds, strict=True)
        if seconds is None and unit.walltime_cell_s is None
    ]


def best_walltime_s(estimate_s: float, margin_s: float, factor: float = BEST_FACTOR) -> int:
    """Return the BEST walltime: ``factor x estimate + margin``, ceiled to the minute (FR-364).

    Parameters
    ----------
    estimate_s : float
        The job's estimate in seconds.
    margin_s : float
        The walltime margin in seconds.
    factor : float, optional
        The safety factor, 1.25 by default.

    Returns
    -------
    int
        Whole seconds, a multiple of 60.
    """
    return math.ceil((factor * estimate_s + margin_s) / 60.0 - 1e-9) * 60


def walltime_text(seconds: int, *, split: BudgetGroup | None = None) -> str:
    """Return a job's wall clock as the WALLTIME cell writes it: whole minutes, else seconds.

    Parameters
    ----------
    seconds : int
        The wall clock in seconds.
    split : BudgetGroup or None, optional
        Unused; kept so callers that pass the job keep working.

    Returns
    -------
    str
        ``<minutes>m`` when the seconds are whole minutes, else ``<seconds>s`` (``68s``), so a
        clock summed from the rows' own cells is written exactly and never rounded up.
    """
    del split
    return f"{seconds // 60}m" if seconds % 60 == 0 else f"{seconds}s"


def _minutes(seconds: float) -> str:
    """Return a duration in whole hours and minutes for a sentence."""
    total = int(round(seconds / 60.0))
    return f"{total // 60}h{total % 60:02d}m"


def _cells_of(split: BudgetGroup) -> list[BudgetRow]:
    """Return the rows of the job that state a wall clock of their own."""
    return [unit for unit in split.units if unit.walltime_cell_s is not None]


def _unstated_walltime(
    split: BudgetGroup, cells: list[BudgetRow], max_walltime_s: float | None
) -> tuple[int | None, str, bool | None, float | None, list[str], str | None] | None:
    """Return the judgment of a job holding a row with no WALLTIME that is not BEST, else None.

    A ROW THAT STATES NO WALL CLOCK AND IS NOT BEST adds nothing to the sum, so
    the sum would bound its points at zero seconds: the job asks
    the profile's limit, said, or is refused where there is none.
    """
    unstated = [
        unit.sim_id for unit in split.units if unit.walltime_cell_s is None and not unit.best
    ]
    if not unstated:
        return None
    names = ", ".join(unstated)
    if max_walltime_s is None:
        return (
            None,
            "none",
            None,
            None,
            [],
            f"rows {names} state no WALLTIME and are not BEST, and share a job with rows "
            f"that state one ({', '.join(unit.sim_id for unit in cells)}), so the job's "
            "walltime cannot budget their points and the profile states no max_walltime "
            "to ask for; write a wall clock with its unit or BEST in those rows.",
        )
    # The points no estimate prices are named too.
    unpriced = _unestimated(split)
    warning = (
        f"rows {names} state no WALLTIME and are not BEST, so the cells of the job do not "
        "budget their points; the job asks the profile's max_walltime."
        + (f" No estimate for {', '.join(unpriced)}." if unpriced else "")
    )
    return int(max_walltime_s), "max_walltime", None, None, [warning], None


def _need_of(split: BudgetGroup, estimate_s: float | None) -> int | None:
    """Return the walltime the job's estimate needs, or None where no point is priced."""
    if estimate_s is None or not any(
        value is not None for unit in split.units for value in unit.point_seconds
    ):
        return None
    return best_walltime_s(estimate_s, max(unit.margin_s for unit in split.units))


def _priced_seconds(split: BudgetGroup, cells: list[BudgetRow]) -> int | None:
    """Return the sum of the cells' budgets with every row priced, or None where one cannot be.

    EVERY UNIT IS PRICED: a row that states no WALLTIME shares
    the job and added 0 s, as BEST does by its estimate. None where such a row
    has a point no estimate prices.
    """
    seconds = int(sum((unit.walltime_cell_s or 0.0) * max(len(unit.run_ids), 1) for unit in cells))
    for unit in split.units:
        if unit.walltime_cell_s is None:
            if any(value is None for value in unit.point_seconds):
                return None
            seconds += best_walltime_s(
                sum(value for value in unit.point_seconds if value is not None), unit.margin_s
            )
    return seconds


def _cell_above_the_limit(cells: list[BudgetRow], max_walltime_s: float | None) -> str | None:
    """Return the refusal of the first row whose own WALLTIME cell is above the limit, else None."""
    if max_walltime_s is None:
        return None
    for unit in cells:
        if (unit.walltime_cell_s or 0.0) > max_walltime_s:
            return (
                f"the WALLTIME {unit.walltime_cell_text} of row {unit.sim_id} is above the "
                f"cluster's max_walltime of {_minutes(max_walltime_s)}; write a cell the cluster "
                "grants, or BEST."
            )
    return None


def _from_cells(
    split: BudgetGroup, estimate_s: float | None, max_walltime_s: float | None
) -> tuple[int | None, str, bool | None, float | None, list[str], str | None]:
    """Return the walltime of a job whose rows state cells: the SUM of its points' budgets.

    A row's WALLTIME cell is the budget of ONE of its points (it states the
    wall clock a single datapoint is given), so a job that runs several points in
    one instance asks for the sum over its points of their rows' cells. A sum above the
    profile's ``max_walltime`` is capped there with a warning suggesting a larger ``n``; one
    point's own cell above it is refused.
    """
    cells = _cells_of(split)
    warnings: list[str] = []
    refusal = _cell_above_the_limit(cells, max_walltime_s)
    if refusal is not None:
        return None, "matrix", None, None, warnings, refusal
    unstated = _unstated_walltime(split, cells, max_walltime_s)
    if unstated is not None:
        return unstated
    priced = _priced_seconds(split, cells)
    if priced is None:
        return _from_estimate(split, None, max_walltime_s)
    seconds = priced
    if max_walltime_s is not None and seconds > max_walltime_s:
        warnings.append(
            f"the job's points ask {_minutes(seconds)} together, above the cluster's max_walltime "
            f"of {_minutes(max_walltime_s)}; the job asks the maximum, and a larger --batch n "
            "makes shorter jobs."
        )
        seconds = int(max_walltime_s)
        # THE CAP IS COMPARED WITH THE ESTIMATE (F-263): the cells' sum above the limit does
        # not say the job's estimated need does not fit under it.
        need = _need_of(split, estimate_s)
        if need is None:
            return seconds, "max_walltime", False, None, warnings, None
        if need <= seconds:
            return seconds, "max_walltime", True, None, warnings, None
        return seconds, "max_walltime", False, float(need - seconds), warnings, None
    need = _need_of(split, estimate_s)
    if need is None or estimate_s is None:
        return seconds, "matrix", None, None, warnings, None
    if need <= seconds:
        return seconds, "matrix", True, None, warnings, None
    short = float(need - seconds)
    warnings.append(
        f"the job's points ask {_minutes(seconds)} together and its estimate needs {need // 60}m "
        f"(1.25 x {_minutes(estimate_s)} plus the margin), {_minutes(short)} short; the cells "
        "are kept, and a larger --batch n makes shorter jobs."
    )
    return seconds, "matrix", False, short, warnings, None


def _from_estimate(
    split: BudgetGroup, estimate_s: float | None, max_walltime_s: float | None
) -> tuple[int | None, str, bool | None, float | None, list[str], str | None]:
    """Return the walltime of an all-BEST job: its estimate, else the limit, else a block."""
    if estimate_s is None:
        names = ", ".join(_unestimated(split))
        if max_walltime_s is None:
            return (
                None,
                "none",
                None,
                None,
                [],
                f"no estimate for {names}, and the profile states no max_walltime to ask for; "
                "write a wall clock with its unit in those rows, or run one point of the row "
                "first so the fit has a sample.",
            )
        return (
            int(max_walltime_s),
            "max_walltime",
            None,
            None,
            [f"no estimate for {names}; the job asks the profile's max_walltime."],
            None,
        )
    wanted = best_walltime_s(estimate_s, max(unit.margin_s for unit in split.units))
    if max_walltime_s is not None and wanted > max_walltime_s:
        short = wanted - max_walltime_s
        return (
            int(max_walltime_s),
            "max_walltime",
            False,
            float(short),
            [
                f"BEST asks {wanted // 60}m, above the cluster's max_walltime of "
                f"{_minutes(max_walltime_s)}, so the job asks the limit and is {_minutes(short)} "
                "short; a larger --batch n makes shorter jobs."
            ],
            None,
        )
    return wanted, "BEST", True, None, [], None


def job_walltime(
    split: BudgetGroup, estimate_s: float | None, *, max_walltime_s: float | None
) -> tuple[int | None, str, bool | None, float | None, list[str], str | None]:
    """Decide the wall clock one job asks for (FR-364, FR-377).

    No cell and a BEST row: the job is BEST, ``ceil(1.25 x estimate + margin)`` to the minute,
    capped at ``max_walltime_s`` with a warning that suggests a larger ``n``. Otherwise the job
    takes the sum of the cells per point and each BEST row's priced estimate; a
    row that states no WALLTIME and is not BEST sends the job to ``max_walltime_s`` with a
    warning, and the plan is refused where the profile states none. A point with no estimate
    takes ``max_walltime_s`` with a warning naming it, or blocks the plan where the profile
    states none. A cell above
    ``max_walltime_s`` is refused; a shortfall is named only with a recorded point estimate. A job
    whose rows all state nothing asks for none.

    Parameters
    ----------
    split : BudgetGroup
        The job's rows.
    estimate_s : float or None
        The job's estimate from :func:`job_estimate`.
    max_walltime_s : float or None
        The profile's ``max_walltime`` in seconds.

    Returns
    -------
    tuple
        ``(walltime_s, source, fits, shortfall_s, warnings, refusal)``; ``source`` is one of
        ``BEST``, ``matrix``, ``max_walltime`` and ``none``; ``refusal`` is the sentence that
        blocks the plan, or None.
    """
    if _cells_of(split):
        return _from_cells(split, estimate_s, max_walltime_s)
    if any(unit.best for unit in split.units):
        return _from_estimate(split, estimate_s, max_walltime_s)
    return None, "none", None, None, [], None
